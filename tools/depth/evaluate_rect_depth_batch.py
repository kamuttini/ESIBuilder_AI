#!/usr/bin/env python3
"""Batch-evaluate the RECT_DEPTH OCR/ranking baseline against legacy .fss boxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from rect_depth_hybrid import (
    DepthCluster,
    cluster_tokens,
    collect_depth_tokens,
    find_config_dirs,
    iter_images,
    parse_rect_depth_checks,
    parse_rect_echo,
)


Box = Tuple[float, float, float, float]


def _safe_slug(text: str, max_len: int = 110) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _box_iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: Box, b: Box) -> float:
    ax = 0.5 * (a[0] + a[2])
    ay = 0.5 * (a[1] + a[3])
    bx = 0.5 * (b[0] + b[2])
    by = 0.5 * (b[1] + b[3])
    return math.hypot(ax - bx, ay - by)


def _legacy_depth_box(fss_path: Path) -> Optional[Box]:
    try:
        checks, _depths = parse_rect_depth_checks(fss_path)
    except Exception:
        return None
    nf = [c for c in checks if c.flip_state == "nf"]
    checks = nf or checks
    if not checks:
        return None
    left = float(round(sum(c.left for c in checks) / len(checks)))
    top = float(round(sum(c.top for c in checks) / len(checks)))
    right = float(round(sum(c.right for c in checks) / len(checks)))
    bottom = float(round(sum(c.bottom for c in checks) / len(checks)))
    return (left, top, right, bottom)


def _expected_depths(fss_path: Path) -> List[float]:
    try:
        _checks, depths = parse_rect_depth_checks(fss_path)
        return depths
    except Exception:
        return []


def _draw_preview(
    image_path: Path,
    pred_box: Optional[Box],
    legacy_box: Optional[Box],
    rect_echo: Optional[Box],
    label: str,
    out_path: Path,
) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        def rect(box: Optional[Box], color: Tuple[int, int, int], width: int) -> None:
            if not box:
                return
            draw.rectangle([float(v) for v in box], outline=color, width=width)

        rect(rect_echo, (0, 180, 255), 4)
        rect(legacy_box, (255, 215, 0), 6)
        rect(pred_box, (20, 230, 90), 5)
        text_box = draw.textbbox((12, 12), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((12, 12), label, fill=(255, 255, 255), font=font)
        im.thumbnail((900, 560), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, quality=90)


def _collect_fss_jobs(roots: Sequence[Path], max_folders: int) -> List[Tuple[Path, Path]]:
    jobs: List[Tuple[Path, Path]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for config_dir in find_config_dirs(root):
            fss_files = sorted((config_dir / "DB_setup").glob("*.fss"))
            if not fss_files:
                continue
            key = config_dir.resolve().as_posix()
            if key in seen:
                continue
            seen.add(key)
            jobs.append((config_dir, fss_files[0]))
            if max_folders and len(jobs) >= max_folders:
                return jobs
    return jobs


def _row_for_failure(config_dir: Path, fss_path: Path, status: str, message: str) -> Dict[str, object]:
    return {
        "config_dir": config_dir.as_posix(),
        "fss_path": fss_path.as_posix(),
        "status": status,
        "message": message,
        "image_count": 0,
        "token_count": 0,
        "candidate_count": 0,
        "best_score": "",
        "center_error_px": "",
        "iou": "",
        "legacy_box": "",
        "pred_box": "",
        "best_reason": "",
        "preview": "",
    }


def evaluate(args: argparse.Namespace) -> None:
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    output_dir = Path(args.output_dir).expanduser().resolve()
    preview_dir = output_dir / "previews"
    jobs = _collect_fss_jobs(roots, int(args.max_folders))
    rows: List[Dict[str, object]] = []

    for idx, (config_dir, fss_path) in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {config_dir}")
        image_dir = config_dir / "image_samples"
        images = iter_images(image_dir)
        if args.max_images and len(images) > args.max_images:
            images = images[: args.max_images]
        if not images:
            rows.append(_row_for_failure(config_dir, fss_path, "no_images", "No images found"))
            continue

        legacy_box = _legacy_depth_box(fss_path)
        if legacy_box is None:
            rows.append(_row_for_failure(config_dir, fss_path, "no_legacy", "No RECT_DEPTH in .fss"))
            continue
        expected_depths = _expected_depths(fss_path)
        rect_echo = parse_rect_echo(fss_path)

        try:
            tokens = collect_depth_tokens(
                images=images,
                timeout=float(args.ocr_timeout),
                max_side=int(args.ocr_max_side),
                min_conf=float(args.min_ocr_conf),
                expected_depths_mm=expected_depths,
            )
            clusters = cluster_tokens(
                tokens=tokens,
                bin_px=float(args.cluster_bin_px),
                image_count=len(images),
                expected_depths_mm=expected_depths,
                rect_echo=rect_echo,
            )
        except Exception as exc:
            rows.append(_row_for_failure(config_dir, fss_path, "error", str(exc)))
            continue

        best: Optional[DepthCluster] = clusters[0] if clusters else None
        pred_box: Optional[Box] = None
        center_error = ""
        iou = ""
        status = "no_candidate"
        message = ""
        if best is not None:
            pred_box = (best.left, best.top, best.right, best.bottom)
            center_error_value = _center_distance(pred_box, legacy_box)
            iou_value = _box_iou(pred_box, legacy_box)
            center_error = f"{center_error_value:.2f}"
            iou = f"{iou_value:.4f}"
            status = "ok" if center_error_value <= float(args.pass_center_px) else "miss"
            message = f"center_error={center_error}; iou={iou}"

        preview_path = ""
        if args.preview_limit <= 0 or len([r for r in rows if r.get("preview")]) < args.preview_limit:
            preview_name = f"{idx:03d}_{_safe_slug(config_dir.name)}.jpg"
            preview_out = preview_dir / preview_name
            _draw_preview(images[0], pred_box, legacy_box, rect_echo, status, preview_out)
            preview_path = preview_out.as_posix()

        rows.append(
            {
                "config_dir": config_dir.as_posix(),
                "fss_path": fss_path.as_posix(),
                "status": status,
                "message": message,
                "image_count": len(images),
                "token_count": len(tokens),
                "candidate_count": len(clusters),
                "best_score": f"{best.score:.4f}" if best else "",
                "center_error_px": center_error,
                "iou": iou,
                "legacy_box": "|".join(f"{v:.1f}" for v in legacy_box),
                "pred_box": "|".join(f"{v:.1f}" for v in pred_box) if pred_box else "",
                "best_reason": best.reason if best else "",
                "preview": preview_path,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "batch_results.csv"
    fieldnames = [
        "config_dir",
        "fss_path",
        "status",
        "message",
        "image_count",
        "token_count",
        "candidate_count",
        "best_score",
        "center_error_px",
        "iou",
        "legacy_box",
        "pred_box",
        "best_reason",
        "preview",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_errors = [float(r["center_error_px"]) for r in rows if str(r.get("center_error_px", ""))]
    ok_count = sum(1 for r in rows if r["status"] == "ok")
    summary = {
        "roots": [p.as_posix() for p in roots],
        "folders": len(rows),
        "ok": ok_count,
        "miss": sum(1 for r in rows if r["status"] == "miss"),
        "no_candidate": sum(1 for r in rows if r["status"] == "no_candidate"),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "pass_center_px": float(args.pass_center_px),
        "median_center_error_px": float(np_median(numeric_errors)) if numeric_errors else None,
        "mean_center_error_px": float(sum(numeric_errors) / len(numeric_errors)) if numeric_errors else None,
        "results_csv": csv_path.as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def np_median(values: Sequence[float]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch test RECT_DEPTH detection across many config folders.")
    parser.add_argument("--roots", nargs="+", required=True, help="Dataset roots to scan.")
    parser.add_argument("--output-dir", required=True, help="Output report directory.")
    parser.add_argument("--max-folders", type=int, default=40, help="Limit scanned config folders.")
    parser.add_argument("--max-images", type=int, default=12, help="Images per folder to OCR.")
    parser.add_argument("--ocr-timeout", type=float, default=8.0)
    parser.add_argument("--ocr-max-side", type=int, default=1800)
    parser.add_argument("--min-ocr-conf", type=float, default=5.0)
    parser.add_argument("--cluster-bin-px", type=float, default=42.0)
    parser.add_argument("--pass-center-px", type=float, default=45.0)
    parser.add_argument("--preview-limit", type=int, default=40)
    parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
