#!/usr/bin/env python3
"""Build a symbol-level detection dataset (small boxes) from existing manifests.

This script uses tiny orientation templates (orientation_0..3.png) found in
UpdateEsiConfiguration folders and performs normalized cross-correlation (NCC)
inside the coarse line16 envelope.

Output:
- symbol_template_bank.csv
- symbol_detection_manifest.csv
- summary.json / summary.txt
- optional preview overlays
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ORIENTATION_NAMES = {0: "NF", 1: "LR", 2: "UD", 3: "LRUD"}
TEMPLATE_RE = re.compile(r"orientation_([0-3])\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE)
DEFAULT_SOURCE_TYPES = (
    "th_orientation_positive",
    "depth_find_nf",
    "depth_find_lr",
    "depth_find_ud",
    "depth_find_lrud",
)


@dataclass(frozen=True)
class CoarseRect:
    top: int
    left: int
    bottom: int
    right: int
    orientation_available: int


@dataclass(frozen=True)
class FrameRow:
    setup_id: str
    split: str
    dataset_folder: str
    manufacturer: str
    model_name: str
    fss_path: str
    image_path: Path
    image_name: str
    source_type: str
    orientation_hint: int
    fallback_rect: Optional[Tuple[int, int, int, int]]


@dataclass(frozen=True)
class TemplateBankRow:
    template_id: str
    orientation_idx: int
    template_path: Path
    width: int
    height: int
    mean: float
    std: float
    tensor_zero_mean: torch.Tensor
    tensor_ones: torch.Tensor
    tensor_norm: float


@dataclass(frozen=True)
class MatchResult:
    score: float
    x: int
    y: int
    template: TemplateBankRow


def _parse_int_optional(value: str) -> Optional[int]:
    cleaned = (value or "").strip()
    if cleaned == "":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_coarse_rect_map(path: Path) -> Dict[Tuple[str, int], CoarseRect]:
    out: Dict[Tuple[str, int], CoarseRect] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            setup_id = row.get("setup_id", "").strip()
            orientation_idx = _parse_int_optional(row.get("orientation_idx", ""))
            top = _parse_int_optional(row.get("rect_top", ""))
            left = _parse_int_optional(row.get("rect_left", ""))
            bottom = _parse_int_optional(row.get("rect_bottom", ""))
            right = _parse_int_optional(row.get("rect_right", ""))
            orient_available = _parse_int_optional(row.get("orientation_available", ""))
            if (
                not setup_id
                or orientation_idx is None
                or top is None
                or left is None
                or bottom is None
                or right is None
            ):
                continue
            out[(setup_id, orientation_idx)] = CoarseRect(
                top=top,
                left=left,
                bottom=bottom,
                right=right,
                orientation_available=0 if orient_available is None else orient_available,
            )
    return out


def _load_frames(path: Path, source_types: set[str], max_rows: int) -> List[FrameRow]:
    rows: List[FrameRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            orientation_hint = _parse_int_optional(raw.get("orientation_hint", ""))
            if orientation_hint is None or orientation_hint not in (0, 1, 2, 3):
                continue
            source_type = (raw.get("source_type", "") or "").strip()
            if source_types and source_type not in source_types:
                continue
            image_path = Path(raw.get("image_path", "")).expanduser().resolve()
            if not image_path.exists():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            top = _parse_int_optional(raw.get("target_rect_top", ""))
            left = _parse_int_optional(raw.get("target_rect_left", ""))
            bottom = _parse_int_optional(raw.get("target_rect_bottom", ""))
            right = _parse_int_optional(raw.get("target_rect_right", ""))
            fallback_rect = None
            if top is not None and left is not None and bottom is not None and right is not None:
                fallback_rect = (top, left, bottom, right)

            rows.append(
                FrameRow(
                    setup_id=(raw.get("setup_id", "") or "").strip(),
                    split=((raw.get("split", "") or "").strip().lower() or "train"),
                    dataset_folder=(raw.get("dataset_folder", "") or "").strip(),
                    manufacturer=(raw.get("manufacturer", "") or "").strip(),
                    model_name=(raw.get("model_name", "") or "").strip(),
                    fss_path=(raw.get("fss_path", "") or "").strip(),
                    image_path=image_path,
                    image_name=(raw.get("image_name", "") or image_path.name).strip(),
                    source_type=source_type,
                    orientation_hint=orientation_hint,
                    fallback_rect=fallback_rect,
                )
            )
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _to_gray_tensor(path: Path, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    with Image.open(path) as img:
        gray = img.convert("L")
        width, height = gray.size
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device), width, height


def _prepare_templates(dataset_roots: Sequence[Path], device: torch.device) -> List[TemplateBankRow]:
    templates: List[TemplateBankRow] = []
    seen: set[Tuple[int, int, int, bytes]] = set()
    next_id = 0

    for root in dataset_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            m = TEMPLATE_RE.match(path.name)
            if not m:
                continue
            orientation_idx = int(m.group(1))
            try:
                with Image.open(path) as img:
                    gray = img.convert("L")
                    arr = np.asarray(gray, dtype=np.float32) / 255.0
            except Exception:
                continue

            h, w = int(arr.shape[0]), int(arr.shape[1])
            if h < 6 or w < 6:
                continue
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-6:
                continue

            key = (orientation_idx, h, w, arr.tobytes())
            if key in seen:
                continue
            seen.add(key)

            tpl = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            tpl = tpl.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            tpl_zm = tpl - tpl.mean()
            tpl_norm = float(torch.sqrt(torch.clamp((tpl_zm * tpl_zm).sum(), min=1e-12)).item())
            if tpl_norm < 1e-6:
                continue

            templates.append(
                TemplateBankRow(
                    template_id=f"tpl_{next_id:04d}",
                    orientation_idx=orientation_idx,
                    template_path=path.resolve(),
                    width=w,
                    height=h,
                    mean=mean,
                    std=std,
                    tensor_zero_mean=tpl_zm,
                    tensor_ones=torch.ones_like(tpl_zm),
                    tensor_norm=tpl_norm,
                )
            )
            next_id += 1

    return templates


def _ncc_best_match(search_img: torch.Tensor, template: TemplateBankRow) -> Optional[Tuple[float, int, int]]:
    # search_img: [H,W] in [0,1], float32, already on target device.
    h, w = int(search_img.shape[0]), int(search_img.shape[1])
    th, tw = template.height, template.width
    if h < th or w < tw:
        return None

    x = search_img.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    n = float(th * tw)

    num = F.conv2d(x, template.tensor_zero_mean)
    sum_x = F.conv2d(x, template.tensor_ones)
    sum_x2 = F.conv2d(x * x, template.tensor_ones)

    var_x = torch.clamp(sum_x2 - (sum_x * sum_x) / n, min=0.0)
    den = torch.sqrt(var_x) * template.tensor_norm

    # Guard against near-constant windows where NCC is numerically unstable.
    valid = den > 1e-6
    ncc = torch.where(valid, num / (den + 1e-8), torch.full_like(num, -1.0))
    ncc = torch.clamp(ncc, min=-1.0, max=1.0)
    ncc = torch.nan_to_num(ncc, nan=-1.0, posinf=-1.0, neginf=-1.0)

    flat = ncc.view(-1)
    best_val, best_idx = torch.max(flat, dim=0)
    idx = int(best_idx.item())
    out_w = int(ncc.shape[-1])
    y = idx // out_w
    x0 = idx % out_w
    return float(best_val.item()), x0, y


def _find_best_match(search_img: torch.Tensor, templates: Sequence[TemplateBankRow]) -> Optional[MatchResult]:
    best: Optional[MatchResult] = None
    for template in templates:
        m = _ncc_best_match(search_img, template)
        if m is None:
            continue
        score, x, y = m
        if best is None or score > best.score:
            best = MatchResult(score=score, x=x, y=y, template=template)
    return best


def _draw_preview(
    image_path: Path,
    coarse_rect: Tuple[int, int, int, int],
    symbol_rect: Optional[Tuple[int, int, int, int]],
    out_path: Path,
) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle((coarse_rect[1], coarse_rect[0], coarse_rect[3], coarse_rect[2]), outline=(255, 220, 0), width=3)
    if symbol_rect is not None:
        draw.rectangle((symbol_rect[1], symbol_rect[0], symbol_rect[3], symbol_rect[2]), outline=(255, 64, 64), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare small-box symbol detection dataset.")
    p.add_argument(
        "--frame-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv"),
    )
    p.add_argument(
        "--setup-orientation-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbolic_dataset/setup_orientation_manifest.csv"),
    )
    p.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[Path("Dataset"), Path("Dataset L_T")],
        help="Roots used to discover orientation_0..3 template images.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset"),
    )
    p.add_argument(
        "--source-types",
        type=str,
        default=",".join(DEFAULT_SOURCE_TYPES),
        help="Comma-separated source_type values from frame_manifest.",
    )
    p.add_argument(
        "--search-margin-px",
        type=int,
        default=24,
        help="Extra margin around coarse line16 rect for template search.",
    )
    p.add_argument(
        "--min-match-score",
        type=float,
        default=0.45,
        help="NCC threshold to auto-accept pseudo-box labels.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for processed frames (0 = no cap).",
    )
    p.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="How many best and worst accepted previews to export.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force torch device (e.g. cpu, mps, cuda).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_types = {s.strip() for s in args.source_types.split(",") if s.strip()}
    device = _choose_device(args.device)

    coarse_map = _load_coarse_rect_map(args.setup_orientation_manifest.expanduser().resolve())
    frames = _load_frames(
        args.frame_manifest.expanduser().resolve(),
        source_types=source_types,
        max_rows=max(0, int(args.max_rows)),
    )
    if not frames:
        raise RuntimeError("No frame rows selected. Check --source-types and input manifest.")

    template_rows = _prepare_templates(
        [p.expanduser().resolve() for p in args.dataset_roots],
        device=device,
    )
    if not template_rows:
        raise RuntimeError("No orientation templates found (orientation_0..3.png).")
    templates_by_orientation: Dict[int, List[TemplateBankRow]] = {0: [], 1: [], 2: [], 3: []}
    for tpl in template_rows:
        templates_by_orientation[tpl.orientation_idx].append(tpl)

    template_bank_csv = output_dir / "symbol_template_bank.csv"
    with template_bank_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "template_id",
                "orientation_idx",
                "orientation_name",
                "template_path",
                "template_width",
                "template_height",
                "template_mean",
                "template_std",
            ],
        )
        writer.writeheader()
        for tpl in template_rows:
            writer.writerow(
                {
                    "template_id": tpl.template_id,
                    "orientation_idx": tpl.orientation_idx,
                    "orientation_name": ORIENTATION_NAMES.get(tpl.orientation_idx, ""),
                    "template_path": tpl.template_path.as_posix(),
                    "template_width": tpl.width,
                    "template_height": tpl.height,
                    "template_mean": f"{tpl.mean:.6f}",
                    "template_std": f"{tpl.std:.6f}",
                }
            )

    manifest_csv = output_dir / "symbol_detection_manifest.csv"
    fields = [
        "sample_id",
        "label_kind",
        "auto_accept",
        "match_score",
        "template_id",
        "template_path",
        "template_width",
        "template_height",
        "setup_id",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "image_path",
        "image_name",
        "source_type",
        "orientation_idx",
        "orientation_name",
        "coarse_rect_top",
        "coarse_rect_left",
        "coarse_rect_bottom",
        "coarse_rect_right",
        "search_rect_top",
        "search_rect_left",
        "search_rect_bottom",
        "search_rect_right",
        "symbol_rect_top",
        "symbol_rect_left",
        "symbol_rect_bottom",
        "symbol_rect_right",
        "symbol_rect_width",
        "symbol_rect_height",
    ]

    processed = 0
    accepted_scores: List[float] = []
    by_kind: Dict[str, int] = {}
    by_orientation_accept: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    by_orientation_total: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    preview_rows: List[Dict[str, object]] = []

    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for idx, row in enumerate(frames, start=1):
            orientation_idx = row.orientation_hint
            by_orientation_total[orientation_idx] += 1

            coarse = coarse_map.get((row.setup_id, orientation_idx))
            if coarse is None and row.fallback_rect is not None:
                coarse = CoarseRect(
                    top=row.fallback_rect[0],
                    left=row.fallback_rect[1],
                    bottom=row.fallback_rect[2],
                    right=row.fallback_rect[3],
                    orientation_available=1,
                )

            if coarse is None:
                label_kind = "review_no_rect"
                by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
                writer.writerow(
                    {
                        "sample_id": f"sym_{idx:08d}",
                        "label_kind": label_kind,
                        "auto_accept": 0,
                        "match_score": "",
                        "template_id": "",
                        "template_path": "",
                        "template_width": "",
                        "template_height": "",
                        "setup_id": row.setup_id,
                        "split": row.split,
                        "dataset_folder": row.dataset_folder,
                        "manufacturer": row.manufacturer,
                        "model_name": row.model_name,
                        "fss_path": row.fss_path,
                        "image_path": row.image_path.as_posix(),
                        "image_name": row.image_name,
                        "source_type": row.source_type,
                        "orientation_idx": orientation_idx,
                        "orientation_name": ORIENTATION_NAMES[orientation_idx],
                        "coarse_rect_top": "",
                        "coarse_rect_left": "",
                        "coarse_rect_bottom": "",
                        "coarse_rect_right": "",
                        "search_rect_top": "",
                        "search_rect_left": "",
                        "search_rect_bottom": "",
                        "search_rect_right": "",
                        "symbol_rect_top": "",
                        "symbol_rect_left": "",
                        "symbol_rect_bottom": "",
                        "symbol_rect_right": "",
                        "symbol_rect_width": "",
                        "symbol_rect_height": "",
                    }
                )
                processed += 1
                continue

            templates = templates_by_orientation.get(orientation_idx, [])
            if not templates:
                label_kind = "review_no_template"
                by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
                processed += 1
                continue

            try:
                gray, width, height = _to_gray_tensor(row.image_path, device=device)
            except Exception:
                label_kind = "review_open_error"
                by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
                processed += 1
                continue

            margin = max(0, int(args.search_margin_px))
            s_top = max(0, coarse.top - margin)
            s_left = max(0, coarse.left - margin)
            s_bottom = min(height - 1, coarse.bottom + margin)
            s_right = min(width - 1, coarse.right + margin)
            if s_bottom <= s_top or s_right <= s_left:
                label_kind = "review_bad_search_rect"
                by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
                processed += 1
                continue

            search_img = gray[s_top : (s_bottom + 1), s_left : (s_right + 1)]
            best = _find_best_match(search_img, templates)
            if best is None:
                label_kind = "review_no_match"
                by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
                processed += 1
                continue

            y0 = s_top + best.y
            x0 = s_left + best.x
            y1 = y0 + best.template.height - 1
            x1 = x0 + best.template.width - 1
            y1 = min(height - 1, y1)
            x1 = min(width - 1, x1)

            accepted = 1 if best.score >= float(args.min_match_score) else 0
            label_kind = "pseudo_box" if accepted else "review_no_match"
            by_kind[label_kind] = by_kind.get(label_kind, 0) + 1
            if accepted:
                accepted_scores.append(best.score)
                by_orientation_accept[orientation_idx] += 1
                preview_rows.append(
                    {
                        "score": best.score,
                        "image_path": row.image_path,
                        "coarse_rect": (coarse.top, coarse.left, coarse.bottom, coarse.right),
                        "symbol_rect": (y0, x0, y1, x1),
                        "orientation_idx": orientation_idx,
                    }
                )

            writer.writerow(
                {
                    "sample_id": f"sym_{idx:08d}",
                    "label_kind": label_kind,
                    "auto_accept": accepted,
                    "match_score": f"{best.score:.6f}",
                    "template_id": best.template.template_id,
                    "template_path": best.template.template_path.as_posix(),
                    "template_width": best.template.width,
                    "template_height": best.template.height,
                    "setup_id": row.setup_id,
                    "split": row.split,
                    "dataset_folder": row.dataset_folder,
                    "manufacturer": row.manufacturer,
                    "model_name": row.model_name,
                    "fss_path": row.fss_path,
                    "image_path": row.image_path.as_posix(),
                    "image_name": row.image_name,
                    "source_type": row.source_type,
                    "orientation_idx": orientation_idx,
                    "orientation_name": ORIENTATION_NAMES[orientation_idx],
                    "coarse_rect_top": coarse.top,
                    "coarse_rect_left": coarse.left,
                    "coarse_rect_bottom": coarse.bottom,
                    "coarse_rect_right": coarse.right,
                    "search_rect_top": s_top,
                    "search_rect_left": s_left,
                    "search_rect_bottom": s_bottom,
                    "search_rect_right": s_right,
                    "symbol_rect_top": y0,
                    "symbol_rect_left": x0,
                    "symbol_rect_bottom": y1,
                    "symbol_rect_right": x1,
                    "symbol_rect_width": (x1 - x0 + 1),
                    "symbol_rect_height": (y1 - y0 + 1),
                }
            )
            processed += 1
            if processed % 500 == 0:
                print(f"processed {processed} frames...", flush=True)

    preview_count = max(0, int(args.preview_count))
    if preview_count > 0 and preview_rows:
        preview_rows = sorted(preview_rows, key=lambda d: float(d["score"]))
        worst = preview_rows[:preview_count]
        best = list(reversed(preview_rows[-preview_count:]))
        preview_dir = output_dir / "previews"
        for i, item in enumerate(worst, start=1):
            out = preview_dir / f"worst_{i:03d}_score_{item['score']:.4f}_{Path(item['image_path']).name}"
            _draw_preview(
                image_path=Path(item["image_path"]),
                coarse_rect=item["coarse_rect"],  # type: ignore[arg-type]
                symbol_rect=item["symbol_rect"],  # type: ignore[arg-type]
                out_path=out,
            )
        for i, item in enumerate(best, start=1):
            out = preview_dir / f"best_{i:03d}_score_{item['score']:.4f}_{Path(item['image_path']).name}"
            _draw_preview(
                image_path=Path(item["image_path"]),
                coarse_rect=item["coarse_rect"],  # type: ignore[arg-type]
                symbol_rect=item["symbol_rect"],  # type: ignore[arg-type]
                out_path=out,
            )

    accepted_ratio = (len(accepted_scores) / processed) if processed > 0 else 0.0
    score_stats = {}
    if accepted_scores:
        score_stats = {
            "mean": float(sum(accepted_scores) / len(accepted_scores)),
            "median": float(statistics.median(accepted_scores)),
            "min": float(min(accepted_scores)),
            "max": float(max(accepted_scores)),
        }
    else:
        score_stats = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    summary = {
        "device": str(device),
        "templates_total": len(template_rows),
        "templates_per_orientation": {
            ORIENTATION_NAMES[k]: len(templates_by_orientation.get(k, [])) for k in (0, 1, 2, 3)
        },
        "frames_total": len(frames),
        "frames_processed": processed,
        "label_kind_counts": dict(sorted(by_kind.items())),
        "auto_accept_count": len(accepted_scores),
        "auto_accept_ratio": accepted_ratio,
        "auto_accept_score_stats": score_stats,
        "orientation_counts_total": {ORIENTATION_NAMES[k]: int(v) for k, v in by_orientation_total.items()},
        "orientation_counts_auto_accept": {ORIENTATION_NAMES[k]: int(v) for k, v in by_orientation_accept.items()},
        "min_match_score": float(args.min_match_score),
        "search_margin_px": int(args.search_margin_px),
        "source_types": sorted(list(source_types)),
    }

    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_txt.write_text(
        "\n".join(
            [
                f"Templates: {summary['templates_total']}",
                f"Frames processed: {summary['frames_processed']}",
                f"Auto-accepted pseudo-boxes: {summary['auto_accept_count']} ({summary['auto_accept_ratio']:.3f})",
                f"Min match score: {summary['min_match_score']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Template bank: {template_bank_csv}", flush=True)
    print(f"Symbol manifest: {manifest_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    print(
        f"Processed {processed} | Pseudo-box {len(accepted_scores)} | "
        f"Acceptance {accepted_ratio:.3f} | Device {device}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
