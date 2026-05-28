#!/usr/bin/env python3
"""Vendor-routed inference for ultrasound rectangle detection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_ultrasound_rect_net import (
    RectRegressor,
    SampleRow,
    _clamp_box,
    box_iou,
    choose_device,
    load_manifest_rows,
    normalize_box_order,
)
from train_ultrasound_vendor_classifier import VendorClassifier


@dataclass(frozen=True)
class RoutedSample:
    row: SampleRow
    width: int
    height: int
    gt_x1: float
    gt_y1: float
    gt_x2: float
    gt_y2: float


class RoutedDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SampleRow],
        vendor_image_size: int,
        rect_image_size: int,
    ) -> None:
        self.rows = list(rows)
        self.vendor_image_size = int(vendor_image_size)
        self.rect_image_size = int(rect_image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        with Image.open(row.image_path) as img:
            image = img.convert("RGB")
            width, height = image.size
            gt_x1, gt_y1, gt_x2, gt_y2 = _clamp_box(
                row.x1, row.y1, row.x2, row.y2, width, height
            )

            vendor_image = TF.resize(
                image,
                size=[self.vendor_image_size, self.vendor_image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            rect_image = TF.resize(
                image,
                size=[self.rect_image_size, self.rect_image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            vendor_tensor = TF.to_tensor(vendor_image)
            vendor_tensor = (vendor_tensor - self.mean) / self.std
            rect_tensor = TF.to_tensor(rect_image)
            rect_tensor = (rect_tensor - self.mean) / self.std

        target = torch.tensor(
            [
                gt_x1 / width,
                gt_y1 / height,
                gt_x2 / width,
                gt_y2 / height,
            ],
            dtype=torch.float32,
        )
        sample = RoutedSample(
            row=row,
            width=width,
            height=height,
            gt_x1=gt_x1,
            gt_y1=gt_y1,
            gt_x2=gt_x2,
            gt_y2=gt_y2,
        )
        return vendor_tensor, rect_tensor, target, sample


def collate_routed(batch):
    vendor_images = torch.stack([b[0] for b in batch], dim=0)
    rect_images = torch.stack([b[1] for b in batch], dim=0)
    targets = torch.stack([b[2] for b in batch], dim=0)
    samples = [b[3] for b in batch]
    return vendor_images, rect_images, targets, samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inferenza bbox con routing vendor: usa detector specializzato "
            "se confidenza vendor alta, altrimenti fallback detector globale."
        )
    )
    parser.add_argument(
        "--checkpoint-global",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"),
        help="Checkpoint detector rettangolo globale.",
    )
    parser.add_argument(
        "--checkpoint-vendor",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"),
        help="Checkpoint classificatore vendor.",
    )
    parser.add_argument(
        "--vendor-rect-map",
        type=Path,
        default=None,
        help="JSON map vendor->checkpoint rect specializzato.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/rect_inference_vendor_routing"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--vendor-conf-threshold",
        type=float,
        default=0.90,
        help="Usa detector specializzato solo se confidenza vendor >= soglia.",
    )
    parser.add_argument(
        "--enable-vendor-ocr-lowconf",
        action="store_true",
        help="Se conf vendor < soglia, prova OCR per correggere il vendor prima del routing.",
    )
    parser.add_argument("--vendor-ocr-lang", type=str, default="eng")
    parser.add_argument("--vendor-ocr-psm", type=int, default=6)
    parser.add_argument("--vendor-ocr-timeout-sec", type=float, default=4.0)
    parser.add_argument(
        "--vendor-ocr-min-hit-count",
        type=int,
        default=1,
        help="Numero minimo keyword OCR per applicare override vendor.",
    )
    parser.add_argument(
        "--vendor-ocr-max-lowconf-samples",
        type=int,
        default=0,
        help="Limite campioni low-conf su cui fare OCR (0=tutti).",
    )
    parser.add_argument(
        "--vendor-image-size",
        type=int,
        default=0,
        help="Override image-size rete vendor. 0=da checkpoint.",
    )
    parser.add_argument(
        "--rect-image-size",
        type=int,
        default=0,
        help="Override image-size rete rect. 0=da checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-overlays", type=int, default=40)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Usa solo i primi N campioni dello split (0=tutti).",
    )
    return parser


def _to_abs_box(
    pred_norm: torch.Tensor,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = float(pred_norm[0]) * width
    y1 = float(pred_norm[1]) * height
    x2 = float(pred_norm[2]) * width
    y2 = float(pred_norm[3]) * height
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, width, height)
    return x1, y1, x2, y2


def _summary_iou(rows: List[Dict[str, object]], field: str) -> Dict[str, float]:
    vals = [float(r[field]) for r in rows if math.isfinite(float(r[field]))]
    if not vals:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "ge_050": 0.0,
            "ge_075": 0.0,
            "ge_090": 0.0,
        }
    ordered = sorted(vals)

    def q(prob: float) -> float:
        idx = int(round((len(ordered) - 1) * prob))
        idx = max(0, min(idx, len(ordered) - 1))
        return ordered[idx]

    return {
        "mean": float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "p10": float(q(0.10)),
        "p25": float(q(0.25)),
        "p75": float(q(0.75)),
        "p90": float(q(0.90)),
        "ge_050": float(sum(1 for v in vals if v >= 0.50) / len(vals)),
        "ge_075": float(sum(1 for v in vals if v >= 0.75) / len(vals)),
        "ge_090": float(sum(1 for v in vals if v >= 0.90) / len(vals)),
    }


def _checkpoint_image_size(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        try:
            value = int(args.get("image_size", 0) or 0)
        except Exception:
            value = 0
    else:
        value = 0
    return value


def _load_rect_model(
    checkpoint_path: Path,
    device: torch.device,
) -> RectRegressor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_vendor_map(
    map_path: Optional[Path],
    vendor_class_names: Sequence[str],
    global_checkpoint: Path,
) -> Dict[str, Path]:
    del global_checkpoint  # Mapping esplicito: fallback globale gestito a runtime.
    if map_path is None:
        return {}
    if not map_path.exists():
        raise RuntimeError(f"vendor-rect-map non trovato: {map_path}")
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("vendor-rect-map deve essere un oggetto JSON vendor->checkpoint.")

    class_lookup = {name.lower(): name for name in vendor_class_names}
    out: Dict[str, Path] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        vendor_key = k.strip()
        if not vendor_key:
            continue
        if vendor_key in class_lookup:
            vendor_name = class_lookup[vendor_key]
        else:
            vendor_name = class_lookup.get(vendor_key.lower(), vendor_key)
        ckpt_path = Path(v).expanduser().resolve()
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint rect per vendor '{vendor_name}' non trovato: {ckpt_path}")
        out[vendor_name] = ckpt_path
    return out


def _build_ocr_keyword_map(class_names: Sequence[str]) -> Dict[str, List[str]]:
    base_map: Dict[str, List[str]] = {
        "BK": ["bk", "profocus", "flexfocus"],
        "Esaote": ["esaote", "mylab"],
        "Hitachi": ["hitachi", "arietta"],
        "GE": ["logiq", "voluson"],
        "Mindray": ["mindray", "resona", "te7"],
        "Canon": ["canon", "aplio"],
        "Philips": ["philips", "affiniti", "epiq"],
        "Toshiba": ["toshiba", "aplio"],
        "Siemens": ["siemens", "acuson"],
        "Koelis": ["koelis"],
        "Biopsee": ["biopsee", "biojet"],
        "ExactVu": ["exactvu"],
        "Alpinion": ["alpinion", "ecube"],
        "Terason": ["terason"],
        "Sonostar": ["sonostar"],
        "Vinno": ["vinno"],
    }
    return {name: base_map.get(name, [name.lower()]) for name in class_names}


def _run_ocr_text(
    image_path: Path,
    lang: str,
    psm: int,
    timeout_sec: float,
) -> str:
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--oem",
                "1",
                "--psm",
                str(psm),
                "-l",
                lang,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return "__TESSERACT_NOT_FOUND__"
    except Exception:
        return ""
    return (proc.stdout or "").lower()


def _ocr_pick_vendor_from_text(
    text: str,
    keyword_map: Dict[str, Sequence[str]],
) -> Tuple[str, int]:
    if not text:
        return "", 0
    if text == "__TESSERACT_NOT_FOUND__":
        return "", 0

    best_vendor = ""
    best_hits = 0
    for vendor, keywords in keyword_map.items():
        hit_count = 0
        for keyword in keywords:
            if keyword and keyword in text:
                hit_count += 1
        if hit_count > best_hits:
            best_vendor = vendor
            best_hits = hit_count
    return best_vendor, best_hits


def _write_group_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    by_vendor: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_vendor[str(row["true_vendor"])].append(
            (float(row["routed_iou"]), float(row["global_iou"]))
        )

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "vendor",
                "num_samples",
                "routed_mean_iou",
                "global_mean_iou",
                "delta_routed_minus_global",
            ]
        )
        for vendor in sorted(by_vendor):
            vals = by_vendor[vendor]
            routed = sum(v[0] for v in vals) / len(vals)
            global_v = sum(v[1] for v in vals) / len(vals)
            writer.writerow(
                [
                    vendor,
                    len(vals),
                    f"{routed:.6f}",
                    f"{global_v:.6f}",
                    f"{(routed - global_v):.6f}",
                ]
            )


def _save_overlays(rows: List[Dict[str, object]], output_dir: Path, n: int) -> None:
    if n <= 0:
        return
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: float(r["routed_iou"]))
    chosen = sorted_rows[:n] + sorted_rows[-n:]
    for idx, row in enumerate(chosen):
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        draw = ImageDraw.Draw(image)

        gt = (
            float(row["gt_x1"]),
            float(row["gt_y1"]),
            float(row["gt_x2"]),
            float(row["gt_y2"]),
        )
        pred = (
            float(row["pred_x1"]),
            float(row["pred_y1"]),
            float(row["pred_x2"]),
            float(row["pred_y2"]),
        )
        draw.rectangle(gt, outline=(0, 255, 0), width=3)
        draw.rectangle(pred, outline=(255, 0, 0), width=3)

        rank = "worst" if idx < n else "best"
        score = float(row["routed_iou"])
        out_name = (
            f"{rank}_{idx:03d}_iou_{score:.4f}_"
            f"{row['route_type']}_{row['pred_vendor']}_{image_path.name}"
        )
        image.save(overlay_dir / out_name)


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest.expanduser().resolve()
    vendor_ckpt_path = args.checkpoint_vendor.expanduser().resolve()
    global_rect_ckpt_path = args.checkpoint_global.expanduser().resolve()

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    rows, _ = load_manifest_rows(manifest_path, strict_manifest=False)
    split_rows = [r for r in rows if r.split == args.split.lower()]
    if args.max_samples > 0:
        split_rows = split_rows[: args.max_samples]
    if not split_rows:
        raise RuntimeError(f"Nessun campione per split '{args.split}'.")
    print(f"Split: {args.split} | samples: {len(split_rows)}", flush=True)

    vendor_ckpt = torch.load(vendor_ckpt_path, map_location="cpu", weights_only=False)
    vendor_class_names = vendor_ckpt.get("class_names")
    if not vendor_class_names:
        raise RuntimeError("Checkpoint vendor senza class_names.")
    vendor_class_names = [str(x) for x in vendor_class_names]

    if args.vendor_ocr_min_hit_count < 1:
        raise ValueError("--vendor-ocr-min-hit-count deve essere >= 1.")
    if args.vendor_ocr_max_lowconf_samples < 0:
        raise ValueError("--vendor-ocr-max-lowconf-samples deve essere >= 0.")

    vendor_image_size = int(args.vendor_image_size)
    if vendor_image_size <= 0:
        vendor_image_size = int(vendor_ckpt.get("args", {}).get("image_size", 320))
    vendor_model = VendorClassifier(
        num_classes=len(vendor_class_names),
        pretrained=False,
    ).to(device)
    vendor_model.load_state_dict(vendor_ckpt["model_state_dict"])
    vendor_model.eval()

    vendor_rect_paths = _load_vendor_map(
        map_path=args.vendor_rect_map.expanduser().resolve() if args.vendor_rect_map else None,
        vendor_class_names=vendor_class_names,
        global_checkpoint=global_rect_ckpt_path,
    )

    global_rect_ckpt = torch.load(global_rect_ckpt_path, map_location="cpu", weights_only=False)
    rect_image_size = int(args.rect_image_size)
    rect_image_size_source = "arg"
    if rect_image_size <= 0:
        specialized_size_set = set()
        for ckpt_path in vendor_rect_paths.values():
            size = _checkpoint_image_size(ckpt_path)
            if size > 0:
                specialized_size_set.add(size)
        specialized_sizes = sorted(specialized_size_set)
        if len(specialized_sizes) == 1:
            rect_image_size = int(specialized_sizes[0])
            rect_image_size_source = "specialized_checkpoint"
        else:
            rect_image_size = int(global_rect_ckpt.get("args", {}).get("image_size", 320))
            rect_image_size_source = "global_checkpoint"
    if rect_image_size_source == "arg":
        rect_image_size_source = "cli_override"
    global_rect_model = RectRegressor(pretrained=False).to(device)
    global_rect_model.load_state_dict(global_rect_ckpt["model_state_dict"])
    global_rect_model.eval()

    unique_rect_paths = sorted(
        {p.resolve() for p in vendor_rect_paths.values()} | {global_rect_ckpt_path.resolve()}
    )
    rect_models_by_path: Dict[Path, RectRegressor] = {}
    for ckpt_path in unique_rect_paths:
        rect_models_by_path[ckpt_path] = _load_rect_model(ckpt_path, device=device)
    vendor_rect_models = {
        vendor: rect_models_by_path[ckpt_path.resolve()]
        for vendor, ckpt_path in vendor_rect_paths.items()
    }

    print(
        f"Vendor classes: {len(vendor_class_names)} | "
        f"vendor_image_size={vendor_image_size} | rect_image_size={rect_image_size} "
        f"({rect_image_size_source})",
        flush=True,
    )
    print(
        f"Vendor-map entries: {len(vendor_rect_paths)} | "
        f"unique rect models loaded: {len(rect_models_by_path)}",
        flush=True,
    )
    if args.enable_vendor_ocr_lowconf:
        print(
            "OCR low-conf enabled | "
            f"lang={args.vendor_ocr_lang}, psm={args.vendor_ocr_psm}, "
            f"timeout={args.vendor_ocr_timeout_sec:.1f}s, "
            f"min_hits={args.vendor_ocr_min_hit_count}",
            flush=True,
        )

    dataset = RoutedDataset(
        rows=split_rows,
        vendor_image_size=vendor_image_size,
        rect_image_size=rect_image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_routed,
        drop_last=False,
    )

    out_rows: List[Dict[str, object]] = []
    route_type_counts: Dict[str, int] = defaultdict(int)
    routed_by_vendor: Dict[str, int] = defaultdict(int)
    vendor_correct = 0
    vendor_total = 0
    ocr_keyword_map = _build_ocr_keyword_map(vendor_class_names)
    ocr_cache: Dict[str, Tuple[str, int, bool]] = {}
    ocr_stats: Counter[str] = Counter()
    ocr_engine_available = True

    with torch.no_grad():
        for vendor_images, rect_images, targets, samples in loader:
            vendor_images = vendor_images.to(device, non_blocking=True)
            rect_images = rect_images.to(device, non_blocking=True)
            # NOTE:
            # On Apple MPS we observed intermittent target corruption (GT -> zeros)
            # when transferring labels with non_blocking=True. Keep this transfer
            # blocking to preserve deterministic/valid GT values during evaluation.
            targets = normalize_box_order(targets.to(device, non_blocking=False))

            vendor_logits = vendor_model(vendor_images)
            vendor_probs = torch.softmax(vendor_logits, dim=1)
            vendor_conf, vendor_idx = torch.max(vendor_probs, dim=1)
            vendor_probs_cpu = vendor_probs.cpu()
            vendor_conf_cpu = vendor_conf.cpu()
            vendor_idx_cpu = vendor_idx.cpu()

            global_preds = normalize_box_order(global_rect_model(rect_images))
            routed_preds = global_preds.clone()

            route_type = ["fallback_low_conf"] * len(samples)
            route_vendor = [""] * len(samples)
            route_ckpt = [global_rect_ckpt_path.as_posix()] * len(samples)
            vendor_source = ["cnn_high_conf"] * len(samples)
            ocr_vendor = [""] * len(samples)
            ocr_hit_count = [0] * len(samples)
            ocr_applied = [0] * len(samples)
            vendor_groups: Dict[str, List[int]] = defaultdict(list)

            for i, sample in enumerate(samples):
                pred_vendor = vendor_class_names[int(vendor_idx_cpu[i].item())]
                conf = float(vendor_conf_cpu[i].item())
                vendor_total += 1
                if pred_vendor == sample.row.manufacturer:
                    vendor_correct += 1

                selected_vendor = pred_vendor
                is_low_conf = conf < float(args.vendor_conf_threshold)
                if is_low_conf:
                    vendor_source[i] = "cnn_low_conf"
                    route_type[i] = "fallback_low_conf"

                    can_try_ocr = (
                        args.enable_vendor_ocr_lowconf
                        and ocr_engine_available
                        and (
                            args.vendor_ocr_max_lowconf_samples <= 0
                            or ocr_stats["lowconf_processed"] < args.vendor_ocr_max_lowconf_samples
                        )
                    )
                    if can_try_ocr:
                        ocr_stats["lowconf_processed"] += 1
                        ocr_applied[i] = 1
                        image_key = sample.row.image_path.as_posix()

                        cached = ocr_cache.get(image_key)
                        if cached is None:
                            text = _run_ocr_text(
                                sample.row.image_path,
                                lang=args.vendor_ocr_lang,
                                psm=args.vendor_ocr_psm,
                                timeout_sec=args.vendor_ocr_timeout_sec,
                            )
                            if text == "__TESSERACT_NOT_FOUND__":
                                ocr_engine_available = False
                                ocr_stats["tesseract_not_found"] += 1
                                cached = ("", 0, False)
                            else:
                                has_text = bool(text)
                                vendor_ocr_pick = ""
                                vendor_ocr_hits = 0
                                if has_text:
                                    vendor_ocr_pick, vendor_ocr_hits = _ocr_pick_vendor_from_text(
                                        text=text,
                                        keyword_map=ocr_keyword_map,
                                    )
                                cached = (vendor_ocr_pick, vendor_ocr_hits, has_text)
                            ocr_cache[image_key] = cached

                        ocr_pick, ocr_hits, has_text = cached
                        if has_text:
                            ocr_stats["lowconf_with_text"] += 1
                        if ocr_pick and ocr_hits >= args.vendor_ocr_min_hit_count:
                            ocr_stats["lowconf_with_hit"] += 1
                            ocr_vendor[i] = ocr_pick
                            ocr_hit_count[i] = int(ocr_hits)
                            if ocr_pick != selected_vendor:
                                selected_vendor = ocr_pick
                                vendor_source[i] = "ocr_override_low_conf"
                                ocr_stats["lowconf_overrides"] += 1
                            else:
                                vendor_source[i] = "ocr_confirm_low_conf"
                    elif args.enable_vendor_ocr_lowconf and not ocr_engine_available:
                        ocr_stats["lowconf_skipped_ocr_unavailable"] += 1
                    elif args.enable_vendor_ocr_lowconf:
                        ocr_stats["lowconf_skipped_ocr_cap"] += 1

                specialized_ckpt = vendor_rect_paths.get(selected_vendor)
                if specialized_ckpt is not None:
                    vendor_groups[selected_vendor].append(i)
                    route_vendor[i] = selected_vendor
                    route_ckpt[i] = specialized_ckpt.as_posix()
                    if is_low_conf and vendor_source[i].startswith("ocr_"):
                        route_type[i] = "vendor_specialized_ocr_low_conf"
                    elif is_low_conf:
                        route_type[i] = "vendor_specialized_low_conf"
                    else:
                        route_type[i] = "vendor_specialized"
                else:
                    route_vendor[i] = selected_vendor
                    route_ckpt[i] = global_rect_ckpt_path.as_posix()
                    if is_low_conf and vendor_source[i] == "ocr_override_low_conf":
                        route_type[i] = "fallback_low_conf_ocr_override"
                    elif is_low_conf and vendor_source[i] == "ocr_confirm_low_conf":
                        route_type[i] = "fallback_low_conf_ocr_confirm"
                    elif not is_low_conf:
                        route_type[i] = "fallback_no_specialized_vendor"
                    else:
                        route_type[i] = "fallback_low_conf"

            for pred_vendor, idxs in vendor_groups.items():
                model = vendor_rect_models[pred_vendor]
                idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device)
                sub_images = torch.index_select(rect_images, dim=0, index=idx_tensor)
                sub_preds = normalize_box_order(model(sub_images))
                routed_preds.index_copy_(0, idx_tensor, sub_preds)

            routed_iou = box_iou(routed_preds, targets).detach().cpu()
            global_iou = box_iou(global_preds, targets).detach().cpu()
            routed_preds_cpu = routed_preds.detach().cpu()
            global_preds_cpu = global_preds.detach().cpu()
            targets_cpu = targets.detach().cpu()

            for i, sample in enumerate(samples):
                true_vendor = sample.row.manufacturer
                pred_vendor = vendor_class_names[int(vendor_idx_cpu[i].item())]
                pred_prob = float(vendor_probs_cpu[i, vendor_idx_cpu[i]].item())

                pred_x1, pred_y1, pred_x2, pred_y2 = _to_abs_box(
                    routed_preds_cpu[i], sample.width, sample.height
                )
                global_x1, global_y1, global_x2, global_y2 = _to_abs_box(
                    global_preds_cpu[i], sample.width, sample.height
                )

                tgt = targets_cpu[i]
                gt_x1 = float(tgt[0]) * sample.width
                gt_y1 = float(tgt[1]) * sample.height
                gt_x2 = float(tgt[2]) * sample.width
                gt_y2 = float(tgt[3]) * sample.height

                route_type_counts[route_type[i]] += 1
                if route_type[i].startswith("vendor_specialized"):
                    routed_by_vendor[route_vendor[i]] += 1

                out_rows.append(
                    {
                        "image_path": sample.row.image_path.as_posix(),
                        "split": sample.row.split,
                        "true_vendor": true_vendor,
                        "model_name": sample.row.model_name,
                        "pred_vendor": pred_vendor,
                        "pred_vendor_conf": pred_prob,
                        "vendor_source": vendor_source[i],
                        "ocr_vendor": ocr_vendor[i],
                        "ocr_hit_count": int(ocr_hit_count[i]),
                        "ocr_applied": int(ocr_applied[i]),
                        "route_type": route_type[i],
                        "route_vendor": route_vendor[i],
                        "route_checkpoint": route_ckpt[i],
                        "width": sample.width,
                        "height": sample.height,
                        "gt_x1": gt_x1,
                        "gt_y1": gt_y1,
                        "gt_x2": gt_x2,
                        "gt_y2": gt_y2,
                        "pred_x1": pred_x1,
                        "pred_y1": pred_y1,
                        "pred_x2": pred_x2,
                        "pred_y2": pred_y2,
                        "global_x1": global_x1,
                        "global_y1": global_y1,
                        "global_x2": global_x2,
                        "global_y2": global_y2,
                        "routed_iou": float(routed_iou[i].item()),
                        "global_iou": float(global_iou[i].item()),
                        "delta_iou": float(routed_iou[i].item() - global_iou[i].item()),
                    }
                )

    routed_summary = _summary_iou(out_rows, field="routed_iou")
    global_summary = _summary_iou(out_rows, field="global_iou")
    delta_mean = float(
        sum(float(r["delta_iou"]) for r in out_rows) / max(1, len(out_rows))
    )

    summary = {
        "split": args.split,
        "num_samples": len(out_rows),
        "device": str(device),
        "checkpoint_global": global_rect_ckpt_path.as_posix(),
        "checkpoint_vendor": vendor_ckpt_path.as_posix(),
        "vendor_map_path": args.vendor_rect_map.as_posix() if args.vendor_rect_map else "",
        "vendor_map_entries": len(vendor_rect_paths),
        "vendor_conf_threshold": float(args.vendor_conf_threshold),
        "vendor_image_size": vendor_image_size,
        "rect_image_size": rect_image_size,
        "rect_image_size_source": rect_image_size_source,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "vendor_top1_accuracy": float(vendor_correct / max(1, vendor_total)),
        "route_type_counts": dict(sorted(route_type_counts.items())),
        "vendor_specialized_usage_by_vendor": dict(sorted(routed_by_vendor.items())),
        "nan_iou_rows": {
            "routed": int(sum(1 for r in out_rows if not math.isfinite(float(r["routed_iou"])))),
            "global": int(sum(1 for r in out_rows if not math.isfinite(float(r["global_iou"])))),
        },
        "ocr_lowconf": {
            "enabled": bool(args.enable_vendor_ocr_lowconf),
            "ocr_engine_available": bool(ocr_engine_available),
            "lowconf_processed": int(ocr_stats.get("lowconf_processed", 0)),
            "lowconf_with_text": int(ocr_stats.get("lowconf_with_text", 0)),
            "lowconf_with_hit": int(ocr_stats.get("lowconf_with_hit", 0)),
            "lowconf_overrides": int(ocr_stats.get("lowconf_overrides", 0)),
            "lowconf_skipped_ocr_cap": int(ocr_stats.get("lowconf_skipped_ocr_cap", 0)),
            "lowconf_skipped_ocr_unavailable": int(ocr_stats.get("lowconf_skipped_ocr_unavailable", 0)),
            "tesseract_not_found": int(ocr_stats.get("tesseract_not_found", 0)),
            "lang": args.vendor_ocr_lang,
            "psm": int(args.vendor_ocr_psm),
            "timeout_sec": float(args.vendor_ocr_timeout_sec),
            "min_hit_count": int(args.vendor_ocr_min_hit_count),
            "max_lowconf_samples": int(args.vendor_ocr_max_lowconf_samples),
        },
        "routed_iou": routed_summary,
        "global_iou": global_summary,
        "delta_mean_iou_routed_minus_global": delta_mean,
    }

    preds_csv = output_dir / f"predictions_routed_{args.split}.csv"
    with preds_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_path",
                "split",
                "true_vendor",
                "model_name",
                "pred_vendor",
                "pred_vendor_conf",
                "vendor_source",
                "ocr_vendor",
                "ocr_hit_count",
                "ocr_applied",
                "route_type",
                "route_vendor",
                "route_checkpoint",
                "width",
                "height",
                "gt_x1",
                "gt_y1",
                "gt_x2",
                "gt_y2",
                "pred_x1",
                "pred_y1",
                "pred_x2",
                "pred_y2",
                "global_x1",
                "global_y1",
                "global_x2",
                "global_y2",
                "routed_iou",
                "global_iou",
                "delta_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    _write_group_csv(out_rows, output_dir / f"per_true_vendor_{args.split}.csv")

    summary_json = output_dir / f"summary_{args.split}.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_txt = output_dir / f"summary_{args.split}.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"Split: {args.split}",
                f"Samples: {len(out_rows)}",
                f"Vendor top1 accuracy: {summary['vendor_top1_accuracy']:.4f}",
                f"Route threshold: {args.vendor_conf_threshold:.2f}",
                f"Route counts: {summary['route_type_counts']}",
                (
                    "OCR low-conf: "
                    f"processed={summary['ocr_lowconf']['lowconf_processed']}, "
                    f"with_hit={summary['ocr_lowconf']['lowconf_with_hit']}, "
                    f"overrides={summary['ocr_lowconf']['lowconf_overrides']}"
                ),
                f"Routed mean IoU: {summary['routed_iou']['mean']:.4f}",
                f"Global mean IoU: {summary['global_iou']['mean']:.4f}",
                f"Delta mean IoU (routed-global): {summary['delta_mean_iou_routed_minus_global']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _save_overlays(out_rows, output_dir, n=args.save_overlays)

    print(f"Predictions: {preds_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    print(
        "Mean IoU routed/global: "
        f"{summary['routed_iou']['mean']:.4f}/{summary['global_iou']['mean']:.4f} "
        f"(delta={summary['delta_mean_iou_routed_minus_global']:.6f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
