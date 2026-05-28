#!/usr/bin/env python3
"""Evaluate orientation template detector and export preview overlays.

Generates:
- predictions CSV on selected split
- summary JSON
- best/worst overlays per model
- optional overall best/worst overlays
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from orientation_template_runtime import (
    RectRegressor,
    SampleRow,
    _clamp_box,
    choose_device,
    load_manifest_rows,
    normalize_box_order,
)


@dataclass(frozen=True)
class EvalRow:
    row: SampleRow
    width: int
    height: int
    pred_box: Tuple[float, float, float, float]
    gt_box: Tuple[float, float, float, float]
    iou: float
    candidate_idx: int
    checkpoint_used: str


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return inter / union


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def _load_vendor_map(path: Optional[Path]) -> Dict[str, Path]:
    if path is None:
        return {}
    p = path.expanduser().resolve()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Path] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        ckpt = Path(value).expanduser()
        if not ckpt.is_absolute():
            ckpt = (p.parent / ckpt).resolve()
        else:
            ckpt = ckpt.resolve()
        if ckpt.exists():
            out[key] = ckpt
    return out


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    cache: Dict[str, Tuple[RectRegressor, int]],
) -> Tuple[RectRegressor, int]:
    key = checkpoint_path.as_posix()
    if key in cache:
        return cache[key]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("args", {}).get("image_size", 320))

    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cache[key] = (model, image_size)
    return model, image_size


def _predict_box(
    image_path: Path,
    model: RectRegressor,
    image_size: int,
    device: torch.device,
) -> Tuple[int, int, Tuple[float, float, float, float]]:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

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

    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
        pred = normalize_box_order(pred).squeeze(0).detach().cpu()

    x1 = float(pred[0]) * float(width)
    y1 = float(pred[1]) * float(height)
    x2 = float(pred[2]) * float(width)
    y2 = float(pred[3]) * float(height)
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, width, height)
    return width, height, (x1, y1, x2, y2)


def _choose_best_candidate(
    pred_box: Tuple[float, float, float, float],
    candidates: Sequence[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Tuple[int, Tuple[float, float, float, float], float]:
    best_idx = 0
    best_iou = -1.0
    best_box = candidates[0]
    for idx, cand in enumerate(candidates):
        x1, y1, x2, y2 = _clamp_box(cand[0], cand[1], cand[2], cand[3], width, height)
        gt = (x1, y1, x2, y2)
        iou = _box_iou(pred_box, gt)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
            best_box = gt
    return best_idx, best_box, best_iou


def _draw_overlay(
    image_path: Path,
    pred_box: Tuple[float, float, float, float],
    gt_box: Tuple[float, float, float, float],
    out_path: Path,
) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle(gt_box, outline=(0, 255, 0), width=3)
    draw.rectangle(pred_box, outline=(255, 0, 0), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path)


def _write_predictions_csv(rows: Sequence[EvalRow], path: Path) -> None:
    fields = [
        "image_path",
        "manufacturer",
        "model_name",
        "split",
        "width",
        "height",
        "iou",
        "candidate_idx",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "checkpoint_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "image_path": item.row.image_path.as_posix(),
                    "manufacturer": item.row.manufacturer,
                    "model_name": item.row.model_name,
                    "split": item.row.split,
                    "width": item.width,
                    "height": item.height,
                    "iou": f"{item.iou:.6f}",
                    "candidate_idx": item.candidate_idx,
                    "pred_x1": f"{item.pred_box[0]:.3f}",
                    "pred_y1": f"{item.pred_box[1]:.3f}",
                    "pred_x2": f"{item.pred_box[2]:.3f}",
                    "pred_y2": f"{item.pred_box[3]:.3f}",
                    "gt_x1": f"{item.gt_box[0]:.3f}",
                    "gt_y1": f"{item.gt_box[1]:.3f}",
                    "gt_x2": f"{item.gt_box[2]:.3f}",
                    "gt_y2": f"{item.gt_box[3]:.3f}",
                    "checkpoint_used": item.checkpoint_used,
                }
            )


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean_iou": 0.0, "min_iou": 0.0, "max_iou": 0.0}
    return {
        "count": float(len(values)),
        "mean_iou": float(sum(values) / len(values)),
        "min_iou": float(min(values)),
        "max_iou": float(max(values)),
    }


def _write_previews(
    rows: Sequence[EvalRow],
    output_dir: Path,
    per_model_k: int,
    overall_k: int,
) -> None:
    by_model: Dict[str, List[EvalRow]] = {}
    for item in rows:
        by_model.setdefault(item.row.model_name, []).append(item)

    model_root = output_dir / "previews_by_model"
    for model_name, group in sorted(by_model.items()):
        sorted_group = sorted(group, key=lambda x: x.iou)
        worst = sorted_group[: max(0, per_model_k)]
        best = list(reversed(sorted_group[-max(0, per_model_k) :]))
        model_dir = model_root / _sanitize_name(model_name)
        for idx, item in enumerate(worst, start=1):
            out_name = f"worst_{idx:02d}_iou_{item.iou:.4f}_{item.row.image_path.name}"
            _draw_overlay(item.row.image_path, item.pred_box, item.gt_box, model_dir / out_name)
        for idx, item in enumerate(best, start=1):
            out_name = f"best_{idx:02d}_iou_{item.iou:.4f}_{item.row.image_path.name}"
            _draw_overlay(item.row.image_path, item.pred_box, item.gt_box, model_dir / out_name)

    if overall_k > 0:
        overall_dir = output_dir / "previews_overall"
        sorted_rows = sorted(rows, key=lambda x: x.iou)
        for idx, item in enumerate(sorted_rows[:overall_k], start=1):
            out_name = f"worst_{idx:02d}_iou_{item.iou:.4f}_{_sanitize_name(item.row.model_name)}_{item.row.image_path.name}"
            _draw_overlay(item.row.image_path, item.pred_box, item.gt_box, overall_dir / out_name)
        for idx, item in enumerate(reversed(sorted_rows[-overall_k:]), start=1):
            out_name = f"best_{idx:02d}_iou_{item.iou:.4f}_{_sanitize_name(item.row.model_name)}_{item.row.image_path.name}"
            _draw_overlay(item.row.image_path, item.pred_box, item.gt_box, overall_dir / out_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate orientation template detector and export previews.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_v1/best_model.pt"),
    )
    parser.add_argument(
        "--vendor-rect-map",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_vendor/vendor_rect_map_selected.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/orientation_rect_eval"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--per-model-k", type=int, default=1)
    parser.add_argument("--overall-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    manifest = args.manifest.expanduser().resolve()
    default_ckpt = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = load_manifest_rows(manifest, strict_manifest=False)
    split = args.split.strip().lower()
    rows = [row for row in rows if row.split == split]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No samples for split '{split}'.")

    vendor_map = _load_vendor_map(args.vendor_rect_map)
    device = choose_device(args.device)

    cache: Dict[str, Tuple[RectRegressor, int]] = {}
    eval_rows: List[EvalRow] = []

    for row in rows:
        checkpoint_path = vendor_map.get(row.manufacturer, default_ckpt)
        model, image_size = _load_model(checkpoint_path, device=device, cache=cache)
        width, height, pred_box = _predict_box(row.image_path, model, image_size=image_size, device=device)
        cand_idx, gt_box, best_iou = _choose_best_candidate(pred_box, row.candidates, width, height)

        eval_rows.append(
            EvalRow(
                row=row,
                width=width,
                height=height,
                pred_box=pred_box,
                gt_box=gt_box,
                iou=best_iou,
                candidate_idx=cand_idx,
                checkpoint_used=checkpoint_path.as_posix(),
            )
        )

    pred_csv = output_dir / f"predictions_{split}.csv"
    _write_predictions_csv(eval_rows, pred_csv)

    by_model: Dict[str, List[float]] = {}
    by_manufacturer: Dict[str, List[float]] = {}
    for item in eval_rows:
        by_model.setdefault(item.row.model_name, []).append(item.iou)
        by_manufacturer.setdefault(item.row.manufacturer, []).append(item.iou)

    summary = {
        "split": split,
        "num_samples": len(eval_rows),
        "num_missing_manifest_rows": missing,
        "overall": _summary_stats([x.iou for x in eval_rows]),
        "per_manufacturer": {
            k: _summary_stats(v) for k, v in sorted(by_manufacturer.items())
        },
        "per_model": {
            k: _summary_stats(v) for k, v in sorted(by_model.items())
        },
        "checkpoints_used": sorted({item.checkpoint_used for item in eval_rows}),
    }

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_previews(
        eval_rows,
        output_dir=output_dir,
        per_model_k=max(0, args.per_model_k),
        overall_k=max(0, args.overall_k),
    )

    print(f"Samples: {len(eval_rows)}", flush=True)
    print(f"Overall mean IoU: {summary['overall']['mean_iou']:.4f}", flush=True)
    print(f"Predictions CSV: {pred_csv}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)
    print(f"Preview dir: {output_dir / 'previews_by_model'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
