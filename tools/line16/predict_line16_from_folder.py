#!/usr/bin/env python3
"""Predict full .fss line #16 for a setup folder.

Pipeline:
1) detect orientation template rectangles from orientation images
2) predict B/CH/MM + TH/P1..P5 for each flip
3) build line #16 string
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_line16_params_model import (
    ParamRow,
    choose_device,
    decode_prediction,
    encode_single_row_features,
    load_checkpoint_for_inference,
)
from orientation_template_runtime import RectRegressor, normalize_box_order

ORIENT_IMAGE_RE = re.compile(
    r"^image_orientation_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$",
    re.IGNORECASE,
)
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")


@dataclass
class ImageRectPrediction:
    image_path: Path
    width: int
    height: int
    box: Tuple[float, float, float, float]  # top,left,bottom,right
    score: float


@dataclass
class SetupContext:
    folder_path: Path
    fss_path: Path
    orient_path: Optional[Path]
    fss_id_echo: int
    fss_id_probe: int
    fss_probe_type: int
    fss_video_x: int
    fss_video_y: int
    fss_group_orientation: int
    orient_available: Tuple[int, int, int, int]


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def _infer_manufacturer(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    if re.match(r"^bk(?:[\\s,_-]|\\d)", name):
        return "BK"
    if name.startswith("esaote"):
        return "Esaote"
    if name.startswith("hitachi"):
        return "Hitachi"
    if name.startswith("ge"):
        return "GE"
    if name.startswith("mindray"):
        return "Mindray"
    if name.startswith("canon"):
        return "Canon"
    if name.startswith("philips"):
        return "Philips"
    if name.startswith("toshiba"):
        return "Toshiba"
    if name.startswith("siemens"):
        return "Siemens"
    if name.startswith("koelis"):
        return "Koelis"
    if name.startswith("biopsee"):
        return "Biopsee"
    if name.startswith("terason"):
        return "Terason"
    if name.startswith("sonostar"):
        return "Sonostar"
    if name.startswith("exactvu"):
        return "ExactVu"
    if name.startswith("alpinion"):
        return "Alpinion"
    if name.startswith("vinno"):
        return "Vinno"
    token = re.split(r"[\\s,_-]+", folder_name.strip())[0]
    return token if token else "UNKNOWN"


def _clamp_box_tlbr(
    top: float,
    left: float,
    bottom: float,
    right: float,
    video_x: int,
    video_y: int,
) -> Tuple[int, int, int, int]:
    top_i = int(round(max(0.0, min(top, video_y - 1.0))))
    left_i = int(round(max(0.0, min(left, video_x - 1.0))))
    bottom_i = int(round(max(1.0, min(bottom, float(video_y)))))
    right_i = int(round(max(1.0, min(right, float(video_x)))))
    if bottom_i <= top_i:
        bottom_i = min(video_y, top_i + 1)
    if right_i <= left_i:
        right_i = min(video_x, left_i + 1)
    return top_i, left_i, bottom_i, right_i


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _extract_rect_line(lines: Sequence[str]) -> int:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines):
            if RECT_RE.match(lines[candidate - 1].strip()):
                return candidate
    for idx in range(1, min(20, len(lines)) + 1):
        if RECT_RE.match(lines[idx - 1].strip()):
            return idx
    return 11


def _parse_int(lines: Sequence[str], one_based_idx: int, default: int) -> int:
    if one_based_idx < 1 or one_based_idx > len(lines):
        return default
    txt = lines[one_based_idx - 1].strip()
    if not txt:
        return default
    try:
        return int(float(txt))
    except ValueError:
        return default


def _parse_orient_available(orient_path: Optional[Path]) -> Tuple[int, int, int, int]:
    if orient_path is None or not orient_path.exists():
        return (1, 1, 1, 1)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(orient_path, encoding="utf-8")
    if "Orientations" not in parser:
        return (1, 1, 1, 1)

    sec = parser["Orientations"]

    def _to_bool_int(key: str) -> int:
        value = sec.get(key, "true").strip().lower()
        return 1 if value in {"1", "true", "yes", "on"} else 0

    return (
        _to_bool_int("FLIP_NO"),
        _to_bool_int("FLIP_LR"),
        _to_bool_int("FLIP_UD"),
        _to_bool_int("FLIP_LR_UD"),
    )


def _find_fss(folder_path: Path, explicit_fss: Optional[Path]) -> Path:
    if explicit_fss is not None:
        fss = explicit_fss.expanduser().resolve()
        if not fss.exists():
            raise FileNotFoundError(f"FSS file not found: {fss}")
        return fss

    setup_dir = folder_path / "DB_setup"
    fss_files = sorted(p for p in setup_dir.glob("*.fss") if p.is_file())
    if not fss_files:
        raise FileNotFoundError(f"No .fss found in {setup_dir}")
    if len(fss_files) > 1:
        raise RuntimeError(f"Expected one .fss in {setup_dir}, found {len(fss_files)}")
    return fss_files[0]


def _extract_context(folder_path: Path, explicit_fss: Optional[Path]) -> SetupContext:
    fss_path = _find_fss(folder_path, explicit_fss)
    lines = _read_lines(fss_path)
    rect_line = _extract_rect_line(lines)
    offset = rect_line - 11

    fss_id_echo = _parse_int(lines, 2 + offset, default=0)
    fss_id_probe = _parse_int(lines, 3 + offset, default=0)
    fss_probe_type = _parse_int(lines, 4 + offset, default=0)
    fss_video_x = _parse_int(lines, 9 + offset, default=0)
    fss_video_y = _parse_int(lines, 10 + offset, default=0)
    fss_group_orientation = _parse_int(lines, 12 + offset, default=4)

    if fss_video_x <= 0:
        fss_video_x = 1920
    if fss_video_y <= 0:
        fss_video_y = 1080

    setup_dir = fss_path.parent
    suffix = fss_path.stem.split("_")[-1]
    orient_path = setup_dir / f"setup_{suffix}.orient"
    orient_available = _parse_orient_available(orient_path if orient_path.exists() else None)

    return SetupContext(
        folder_path=folder_path,
        fss_path=fss_path,
        orient_path=orient_path if orient_path.exists() else None,
        fss_id_echo=fss_id_echo,
        fss_id_probe=fss_id_probe,
        fss_probe_type=fss_probe_type,
        fss_video_x=fss_video_x,
        fss_video_y=fss_video_y,
        fss_group_orientation=fss_group_orientation,
        orient_available=orient_available,
    )


def _collect_orientation_images(folder_path: Path) -> List[Path]:
    image_dir = folder_path / "image_samples"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image_samples folder: {image_dir}")

    pairs: List[Tuple[int, Path]] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        m = ORIENT_IMAGE_RE.match(path.name)
        if not m:
            continue
        idx = int(m.group(1))
        pairs.append((idx, path))

    if not pairs:
        raise RuntimeError(f"No image_orientation_setup_* files found in {image_dir}")

    pairs.sort(key=lambda item: item[0])
    return [path for _, path in pairs]


def _iou_tlbr(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    inter_t = max(at, bt)
    inter_l = max(al, bl)
    inter_b = min(ab, bb)
    inter_r = min(ar, br)
    ih = max(0.0, inter_b - inter_t)
    iw = max(0.0, inter_r - inter_l)
    inter = ih * iw
    area_a = max(0.0, ab - at) * max(0.0, ar - al)
    area_b = max(0.0, bb - bt) * max(0.0, br - bl)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _predict_rectangles(
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    device: torch.device,
    max_images: int,
) -> Tuple[List[ImageRectPrediction], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ck_args = checkpoint.get("args", {})
    image_size = int(ck_args.get("image_size", 320))

    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    chosen_images = list(image_paths)
    if max_images > 0 and len(chosen_images) > max_images:
        chosen_images = chosen_images[:max_images]

    out: List[ImageRectPrediction] = []
    with torch.no_grad():
        for image_path in chosen_images:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                resized = TF.resize(
                    rgb,
                    size=[image_size, image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                tensor = TF.to_tensor(resized)
                tensor = (tensor - mean) / std
                tensor = tensor.unsqueeze(0).to(device)

            pred = model(tensor)
            pred = normalize_box_order(pred).squeeze(0).detach().cpu().numpy()
            x1 = float(pred[0]) * width
            y1 = float(pred[1]) * height
            x2 = float(pred[2]) * width
            y2 = float(pred[3]) * height

            top = max(0.0, min(y1, y2))
            left = max(0.0, min(x1, x2))
            bottom = max(top + 1.0, min(float(height), max(y1, y2)))
            right = max(left + 1.0, min(float(width), max(x1, x2)))

            score = float((bottom - top) * (right - left) / max(1.0, width * height))
            out.append(
                ImageRectPrediction(
                    image_path=image_path,
                    width=width,
                    height=height,
                    box=(top, left, bottom, right),
                    score=score,
                )
            )

    return out, image_size


def _cluster_boxes(
    predictions: Sequence[ImageRectPrediction],
    max_clusters: int,
) -> List[Tuple[float, float, float, float]]:
    if not predictions:
        return []

    clusters: List[List[Tuple[float, float, float, float]]] = []
    for pred in predictions:
        box = pred.box
        assigned = False
        for cluster in clusters:
            ref = tuple(float(np.median([b[i] for b in cluster])) for i in range(4))
            iou = _iou_tlbr(box, ref)
            c1 = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
            c2 = ((ref[0] + ref[2]) * 0.5, (ref[1] + ref[3]) * 0.5)
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            h = max(1.0, ref[2] - ref[0])
            w = max(1.0, ref[3] - ref[1])
            norm_dist = dist / max(1.0, math.hypot(h, w))
            if iou >= 0.55 or norm_dist <= 0.35:
                cluster.append(box)
                assigned = True
                break
        if not assigned:
            clusters.append([box])

    clusters.sort(key=lambda group: (-len(group), -np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in group])))
    clusters = clusters[:max_clusters]

    out = []
    for group in clusters:
        out.append(tuple(float(np.median([b[i] for b in group])) for i in range(4)))
    return out


def _flip_lr(box: Tuple[float, float, float, float], video_x: int) -> Tuple[float, float, float, float]:
    t, l, b, r = box
    return t, float(video_x) - r, b, float(video_x) - l


def _flip_ud(box: Tuple[float, float, float, float], video_y: int) -> Tuple[float, float, float, float]:
    t, l, b, r = box
    return float(video_y) - b, l, float(video_y) - t, r


def _assign_flip_boxes(
    clusters: Sequence[Tuple[float, float, float, float]],
    context: SetupContext,
) -> List[Tuple[int, int, int, int]]:
    if not clusters:
        # Fallback center rectangle.
        default_box = (
            context.fss_video_y * 0.08,
            context.fss_video_x * 0.20,
            context.fss_video_y * 0.18,
            context.fss_video_x * 0.60,
        )
        clusters = [default_box]

    base = clusters[0]
    candidates = list(clusters)
    candidates.extend([
        _flip_lr(base, context.fss_video_x),
        _flip_ud(base, context.fss_video_y),
        _flip_ud(_flip_lr(base, context.fss_video_x), context.fss_video_y),
    ])

    # Keep unique-ish candidates.
    uniq: List[Tuple[float, float, float, float]] = []
    for box in candidates:
        if any(_iou_tlbr(box, kept) > 0.85 for kept in uniq):
            continue
        uniq.append(box)

    # Ensure at least 4 candidates.
    while len(uniq) < 4:
        uniq.append(base)

    # Assign by spatial order: top-left, top-right, bottom-left, bottom-right.
    scored = []
    for box in uniq:
        t, l, b, r = box
        cy = (t + b) * 0.5
        cx = (l + r) * 0.5
        scored.append((cy, cx, box))
    scored.sort(key=lambda item: (item[0], item[1]))

    top = sorted(scored[:2], key=lambda item: item[1])
    bottom = sorted(scored[2:4], key=lambda item: item[1]) if len(scored) >= 4 else top

    mapping = [
        top[0][2] if top else base,     # flip 0
        top[1][2] if len(top) > 1 else base,  # flip 1
        bottom[0][2] if bottom else base,     # flip 2
        bottom[1][2] if len(bottom) > 1 else base,  # flip 3
    ]

    out: List[Tuple[int, int, int, int]] = []
    for flip_idx, box in enumerate(mapping):
        if context.orient_available[flip_idx] == 0:
            box = mapping[0]
        out.append(_clamp_box_tlbr(box[0], box[1], box[2], box[3], context.fss_video_x, context.fss_video_y))
    return out


def _format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{float(round(value)):.1f}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0.0"


def _build_param_row_for_flip(
    context: SetupContext,
    flip_idx: int,
    rect: Tuple[int, int, int, int],
) -> ParamRow:
    top, left, bottom, right = rect
    rect_width = max(1.0, float(right - left + 1))
    rect_height = max(1.0, float(bottom - top + 1))
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    vx = float(max(1, context.fss_video_x))
    vy = float(max(1, context.fss_video_y))

    return ParamRow(
        sample_id=f"{context.folder_path.name}::flip{flip_idx}",
        split="predict",
        dataset_folder=context.folder_path.name,
        manufacturer=_infer_manufacturer(context.folder_path.name),
        model_name=context.folder_path.name,
        fss_path=context.fss_path.as_posix(),
        fss_id_echo=context.fss_id_echo,
        fss_id_probe=context.fss_id_probe,
        fss_probe_type=context.fss_probe_type,
        fss_video_x=context.fss_video_x,
        fss_video_y=context.fss_video_y,
        fss_group_orientation=context.fss_group_orientation,
        flip_idx=flip_idx,
        flip_available=context.orient_available[flip_idx],
        rect_top=float(top),
        rect_left=float(left),
        rect_bottom=float(bottom),
        rect_right=float(right),
        rect_width=rect_width,
        rect_height=rect_height,
        rect_cx_norm=float(cx / vx),
        rect_cy_norm=float(cy / vy),
        rect_area_norm=float((rect_width * rect_height) / (vx * vy)),
        b_value=1,
        channel=7,
        threshold=1.0,
        p1=20.0,
        p2=120.0,
        p3=0.0,
        p4=0.0,
        p5=0.0,
        mm=6,
    )


def _load_mm_priors(
    manifest_path: Optional[Path],
    split: str = "train",
) -> Dict[str, Tuple[int, float]]:
    if manifest_path is None or not manifest_path.exists():
        return {}

    by_manufacturer: Dict[str, Dict[int, int]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        seen: set[Tuple[str, str]] = set()
        for row in reader:
            if row.get("split", "").strip().lower() != split:
                continue
            fss_path = row.get("fss_path", "")
            flip_idx = row.get("flip_idx", "")
            dedup_key = (fss_path, flip_idx)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            manufacturer = row.get("manufacturer", "").strip()
            try:
                mm_value = int(float(row.get("mm", "6")))
            except ValueError:
                continue
            bucket = by_manufacturer.setdefault(manufacturer, {})
            bucket[mm_value] = bucket.get(mm_value, 0) + 1

    out: Dict[str, Tuple[int, float]] = {}
    for manufacturer, counter in by_manufacturer.items():
        total = float(sum(counter.values()))
        if total <= 0:
            continue
        mm_value, count = max(counter.items(), key=lambda item: item[1])
        out[manufacturer] = (int(mm_value), float(count / total))
    return out


def _predict_params_for_flip(
    model: torch.nn.Module,
    preprocessor,
    row: ParamRow,
    device: torch.device,
) -> Dict[str, float]:
    feats = encode_single_row_features(row, preprocessor)
    num = torch.tensor(feats.numeric, dtype=torch.float32, device=device)
    cat = torch.tensor(feats.categorical, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(num, cat)
    return decode_prediction(outputs, preprocessor)


def _build_line16(
    flip_rects: Sequence[Tuple[int, int, int, int]],
    flip_params: Sequence[Dict[str, float]],
) -> str:
    segments: List[str] = []
    for idx in range(4):
        top, left, bottom, right = flip_rects[idx]
        params = flip_params[idx]
        b_value = int(round(params["b_value"]))
        channel = int(round(params["channel"]))
        mm = int(round(params["mm"]))

        th = _format_float(max(0.0, float(params["threshold"])))
        p1 = _format_float(max(0.0, float(params["p1"])))
        p2 = _format_float(max(0.0, float(params["p2"])))
        # In the current dataset these are fixed to zero across configurations.
        p3 = "0.0"
        p4 = "0.0"
        p5 = "0.0"

        segment = (
            f"{top}|{left}|{bottom}|{right}|{b_value}|"
            f"{channel}:{th}:{p1}:{p2}:{p3}:{p4}:{p5}|{mm}|"
        )
        segments.append(segment)
    return ";".join(segments) + ";"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict full line #16 for a setup folder.")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--fss", type=Path, default=None)
    parser.add_argument(
        "--orientation-checkpoint",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_v1/best_model.pt"),
    )
    parser.add_argument(
        "--vendor-rect-map",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_vendor/vendor_rect_map_selected.json"),
    )
    parser.add_argument(
        "--params-checkpoint",
        type=Path,
        default=Path("artifacts/30_models/line16_params_training_v2_nobalance/best_model.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--priors-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/manifest_line16_params.csv"),
    )
    parser.add_argument("--mm-prior-min-fraction", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-line16", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    folder_path = args.folder.expanduser().resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    context = _extract_context(folder_path, args.fss)
    image_paths = _collect_orientation_images(folder_path)

    device = choose_device(args.device)

    orientation_ckpt_default = args.orientation_checkpoint.expanduser().resolve()
    params_ckpt = args.params_checkpoint.expanduser().resolve()
    inferred_manufacturer = _infer_manufacturer(folder_path.name)

    orientation_ckpt = orientation_ckpt_default
    if args.vendor_rect_map is not None:
        vendor_map_path = args.vendor_rect_map.expanduser().resolve()
        if vendor_map_path.exists():
            try:
                vendor_map = json.loads(vendor_map_path.read_text(encoding="utf-8"))
                candidate = vendor_map.get(inferred_manufacturer)
                if isinstance(candidate, str) and candidate.strip():
                    path_candidate = Path(candidate).expanduser()
                    if not path_candidate.is_absolute():
                        path_candidate = (vendor_map_path.parent / path_candidate).resolve()
                    else:
                        path_candidate = path_candidate.resolve()
                    if path_candidate.exists():
                        orientation_ckpt = path_candidate
            except json.JSONDecodeError:
                pass

    rect_preds, image_size = _predict_rectangles(
        image_paths=image_paths,
        checkpoint_path=orientation_ckpt,
        device=device,
        max_images=args.max_images,
    )
    clusters = _cluster_boxes(rect_preds, max_clusters=4)
    flip_rects = _assign_flip_boxes(clusters, context)

    params_model, preprocessor, _ = load_checkpoint_for_inference(params_ckpt, device=device)
    mm_priors = _load_mm_priors(
        args.priors_manifest.expanduser().resolve() if args.priors_manifest is not None else None
    )

    flip_params: List[Dict[str, float]] = []
    for flip_idx in range(4):
        row = _build_param_row_for_flip(context, flip_idx, flip_rects[flip_idx])
        pred = _predict_params_for_flip(params_model, preprocessor, row, device=device)

        prior = mm_priors.get(inferred_manufacturer)
        if prior is not None and prior[1] >= float(args.mm_prior_min_fraction):
            pred["mm"] = float(prior[0])

        flip_params.append(pred)

    line16 = _build_line16(flip_rects, flip_params)

    result = {
        "folder": folder_path.as_posix(),
        "fss_path": context.fss_path.as_posix(),
        "orient_path": context.orient_path.as_posix() if context.orient_path else None,
        "fss_meta": {
            "fss_id_echo": context.fss_id_echo,
            "fss_id_probe": context.fss_id_probe,
            "fss_probe_type": context.fss_probe_type,
            "fss_video_x": context.fss_video_x,
            "fss_video_y": context.fss_video_y,
            "fss_group_orientation": context.fss_group_orientation,
            "orient_available": list(context.orient_available),
            "manufacturer_inferred": inferred_manufacturer,
        },
        "image_count": len(rect_preds),
        "orientation_checkpoint_used": orientation_ckpt.as_posix(),
        "orientation_model_image_size": image_size,
        "cluster_count": len(clusters),
        "predicted_flip_rects": [
            {
                "flip_idx": idx,
                "top": rect[0],
                "left": rect[1],
                "bottom": rect[2],
                "right": rect[3],
            }
            for idx, rect in enumerate(flip_rects)
        ],
        "predicted_flip_params": [
            {
                "flip_idx": idx,
                "b_value": int(round(pred["b_value"])),
                "channel": int(round(pred["channel"])),
                "threshold": float(pred["threshold"]),
                "p1": float(pred["p1"]),
                "p2": float(pred["p2"]),
                "p3": float(pred["p3"]),
                "p4": float(pred["p4"]),
                "p5": float(pred["p5"]),
                "mm": int(round(pred["mm"])),
            }
            for idx, pred in enumerate(flip_params)
        ],
        "line16": line16,
    }

    if args.output_json is not None:
        out_json = args.output_json.expanduser().resolve()
    else:
        out_json = (Path("artifacts") / "line16_predictions" / f"{folder_path.name}.json").resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_line16 is not None:
        out_line16 = args.output_line16.expanduser().resolve()
        out_line16.parent.mkdir(parents=True, exist_ok=True)
        out_line16.write_text(line16 + "\n", encoding="utf-8")

    print(line16, flush=True)
    print(f"JSON: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
