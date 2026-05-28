#!/usr/bin/env python3
"""Filter negative/proibite rows and scan transrectal LT rect dataset for anomalous images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_KEYWORD_REGEX = r"(?i)(negative|proibite)"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _read_manifest(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _is_negative_or_proibite(row: Dict[str, str], keyword_re: re.Pattern[str]) -> bool:
    fields = (
        row.get("image_path", ""),
        row.get("image_folder", ""),
        row.get("dataset_folder", ""),
        row.get("model_name", ""),
        row.get("fss_path", ""),
        row.get("source_name", ""),
    )
    blob = " | ".join(fields)
    return keyword_re.search(blob) is not None


def _clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1i = max(0, min(int(round(x1)), width - 1))
    y1i = max(0, min(int(round(y1)), height - 1))
    x2i = max(1, min(int(round(x2)), width))
    y2i = max(1, min(int(round(y2)), height))
    if x2i <= x1i:
        x2i = min(width, x1i + 1)
    if y2i <= y1i:
        y2i = min(height, y1i + 1)
    return x1i, y1i, x2i, y2i


def _resolve_pred_crop(row: Dict[str, str], width: int, height: int) -> Tuple[int, int, int, int, str]:
    n_left = _safe_float(row.get("pred_left_norm"), default=float("nan"))
    n_top = _safe_float(row.get("pred_top_norm"), default=float("nan"))
    n_right = _safe_float(row.get("pred_right_norm"), default=float("nan"))
    n_bottom = _safe_float(row.get("pred_bottom_norm"), default=float("nan"))

    if all(math.isfinite(v) for v in (n_left, n_top, n_right, n_bottom)):
        x1 = n_left * width
        y1 = n_top * height
        x2 = n_right * width
        y2 = n_bottom * height
        x1i, y1i, x2i, y2i = _clip_box(x1, y1, x2, y2, width, height)
        return x1i, y1i, x2i, y2i, "pred_norm"

    a_left = _safe_float(row.get("pred_left"), default=float("nan"))
    a_top = _safe_float(row.get("pred_top"), default=float("nan"))
    a_right = _safe_float(row.get("pred_right"), default=float("nan"))
    a_bottom = _safe_float(row.get("pred_bottom"), default=float("nan"))
    if all(math.isfinite(v) for v in (a_left, a_top, a_right, a_bottom)):
        x1i, y1i, x2i, y2i = _clip_box(a_left, a_top, a_right, a_bottom, width, height)
        return x1i, y1i, x2i, y2i, "pred_abs"

    return 0, 0, width, height, "full_image_fallback"


def _entropy_uint8(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _scan_row(row: Dict[str, str]) -> Dict[str, object]:
    image_path = Path((row.get("image_path") or "").strip())
    out: Dict[str, object] = {
        "image_path": image_path.as_posix(),
        "split": row.get("split", ""),
        "label_lt": row.get("label_lt", ""),
        "manufacturer": row.get("manufacturer", ""),
        "model_name": row.get("model_name", ""),
        "fss_id_probe": row.get("fss_id_probe", ""),
    }

    if not image_path.exists():
        out.update(
            {
                "scan_status": "missing_image",
                "is_suspect": 1,
                "suspect_score": 100,
                "suspect_reason": "missing_image",
            }
        )
        return out

    try:
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"))
    except Exception:
        out.update(
            {
                "scan_status": "read_error",
                "is_suspect": 1,
                "suspect_score": 100,
                "suspect_reason": "read_error",
            }
        )
        return out

    if gray.ndim != 2 or gray.size == 0:
        out.update(
            {
                "scan_status": "invalid_image",
                "is_suspect": 1,
                "suspect_score": 100,
                "suspect_reason": "invalid_image",
            }
        )
        return out

    height, width = gray.shape
    x1, y1, x2, y2, crop_source = _resolve_pred_crop(row, width=width, height=height)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        crop = gray
        x1, y1, x2, y2 = 0, 0, width, height
        crop_source = "full_image_empty_crop_fallback"

    min_v = int(np.min(crop))
    max_v = int(np.max(crop))
    mean_v = float(np.mean(crop))
    std_v = float(np.std(crop))
    unique_v = int(np.unique(crop).size)
    dynamic_range = int(max_v - min_v)
    entropy = _entropy_uint8(crop)

    crop_h, crop_w = crop.shape
    area_frac = float(crop.size / max(1, gray.size))
    aspect = float(crop_w / max(1, crop_h))
    black_ratio = float(np.mean(crop <= 3))
    white_ratio = float(np.mean(crop >= 252))
    nonzero_ratio = float(np.mean(crop > 0))

    gx = np.abs(np.diff(crop.astype(np.float32), axis=1))
    gy = np.abs(np.diff(crop.astype(np.float32), axis=0))
    edge_mean = float(0.5 * (gx.mean() if gx.size > 0 else 0.0) + 0.5 * (gy.mean() if gy.size > 0 else 0.0))

    flags: List[str] = []
    score = 0
    if std_v == 0.0 or unique_v <= 1:
        flags.append("uniform_exact")
        score += 90
    if std_v < 3.0 and dynamic_range < 16 and entropy < 2.0:
        flags.append("uniform_strong")
        score += 70
    if std_v < 6.0 and dynamic_range < 26 and entropy < 3.0 and edge_mean < 2.0:
        flags.append("uniform_like")
        score += 40
    if mean_v < 8.0 and black_ratio > 0.95:
        flags.append("mostly_black")
        score += 70
    if mean_v > 245.0 and white_ratio > 0.95:
        flags.append("mostly_white")
        score += 60
    if area_frac < 0.015:
        flags.append("tiny_crop")
        score += 50
    if min(crop_w, crop_h) < 20:
        flags.append("tiny_side")
        score += 35
    if aspect > 8.0 or aspect < 0.125:
        flags.append("extreme_aspect")
        score += 25
    if nonzero_ratio < 0.02:
        flags.append("almost_all_zero")
        score += 30
    if edge_mean < 0.8 and entropy < 2.5:
        flags.append("very_low_texture")
        score += 20

    is_suspect = int(score >= 40 or ("uniform_exact" in flags))
    reason = "|".join(flags)

    out.update(
        {
            "scan_status": "ok",
            "is_suspect": is_suspect,
            "suspect_score": score,
            "suspect_reason": reason,
            "crop_source": crop_source,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "image_width": width,
            "image_height": height,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "crop_area_fraction": f"{area_frac:.6f}",
            "crop_aspect_ratio": f"{aspect:.6f}",
            "pix_mean": f"{mean_v:.6f}",
            "pix_std": f"{std_v:.6f}",
            "pix_min": min_v,
            "pix_max": max_v,
            "pix_dynamic_range": dynamic_range,
            "pix_unique": unique_v,
            "pix_entropy": f"{entropy:.6f}",
            "pix_black_ratio": f"{black_ratio:.6f}",
            "pix_white_ratio": f"{white_ratio:.6f}",
            "pix_nonzero_ratio": f"{nonzero_ratio:.6f}",
            "edge_mean_absdiff": f"{edge_mean:.6f}",
        }
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter transrectal LT manifest from negative/proibite entries and scan image crops "
            "for uniform/strange anomalies."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/manifest_transrectal_lt_rect.csv"),
        help="Input manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/quality_scan"),
        help="Output folder.",
    )
    parser.add_argument(
        "--keyword-regex",
        type=str,
        default=DEFAULT_KEYWORD_REGEX,
        help="Regex to exclude negative/proibite rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keyword_re = re.compile(args.keyword_regex)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    rows, fields = _read_manifest(manifest_path)
    excluded: List[Dict[str, str]] = []
    kept: List[Dict[str, str]] = []
    for row in rows:
        if _is_negative_or_proibite(row, keyword_re):
            excluded.append(row)
        else:
            kept.append(row)

    clean_manifest = output_dir / "manifest_transrectal_lt_rect_clean.csv"
    excluded_csv = output_dir / "excluded_negative_proibite_rows.csv"
    _write_csv(clean_manifest, kept, fields)
    _write_csv(excluded_csv, excluded, fields)

    metrics_rows: List[Dict[str, object]] = []
    suspects_rows: List[Dict[str, object]] = []
    counters = Counter()
    for idx, row in enumerate(kept, start=1):
        rec = _scan_row(row)
        metrics_rows.append(rec)
        counters[str(rec.get("scan_status", "unknown"))] += 1
        if _safe_int(rec.get("is_suspect"), default=0) == 1:
            suspects_rows.append(rec)
        if idx % 1000 == 0:
            print(f"[scan] {idx}/{len(kept)}", flush=True)

    suspects_rows.sort(key=lambda r: (int(r.get("suspect_score", 0)), str(r.get("image_path", ""))), reverse=True)

    metric_fields = [
        "image_path",
        "split",
        "label_lt",
        "manufacturer",
        "model_name",
        "fss_id_probe",
        "scan_status",
        "is_suspect",
        "suspect_score",
        "suspect_reason",
        "crop_source",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "image_width",
        "image_height",
        "crop_width",
        "crop_height",
        "crop_area_fraction",
        "crop_aspect_ratio",
        "pix_mean",
        "pix_std",
        "pix_min",
        "pix_max",
        "pix_dynamic_range",
        "pix_unique",
        "pix_entropy",
        "pix_black_ratio",
        "pix_white_ratio",
        "pix_nonzero_ratio",
        "edge_mean_absdiff",
    ]
    all_metrics_csv = output_dir / "quality_metrics_all_rows.csv"
    suspects_csv = output_dir / "quality_suspects.csv"
    _write_csv(all_metrics_csv, metrics_rows, metric_fields)
    _write_csv(suspects_csv, suspects_rows, metric_fields)

    by_reason = Counter()
    by_label = Counter()
    for rec in suspects_rows:
        by_label[str(rec.get("label_lt", ""))] += 1
        reasons = str(rec.get("suspect_reason", "")).split("|")
        for reason in reasons:
            reason = reason.strip()
            if reason:
                by_reason[reason] += 1

    summary = {
        "input_manifest": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "keyword_regex": args.keyword_regex,
        "rows_total_input": len(rows),
        "rows_excluded_negative_proibite": len(excluded),
        "rows_clean": len(kept),
        "scan_status_counts": dict(sorted(counters.items())),
        "suspects_total": len(suspects_rows),
        "suspects_by_label": dict(sorted(by_label.items())),
        "suspect_reasons": dict(sorted(by_reason.items(), key=lambda x: (-x[1], x[0]))),
        "paths": {
            "clean_manifest": clean_manifest.as_posix(),
            "excluded_rows_csv": excluded_csv.as_posix(),
            "quality_metrics_all_rows": all_metrics_csv.as_posix(),
            "quality_suspects_csv": suspects_csv.as_posix(),
        },
    }
    summary_json = output_dir / "quality_scan_summary.json"
    summary_txt = output_dir / "quality_scan_summary.txt"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"rows_total_input: {summary['rows_total_input']}",
        f"rows_excluded_negative_proibite: {summary['rows_excluded_negative_proibite']}",
        f"rows_clean: {summary['rows_clean']}",
        f"suspects_total: {summary['suspects_total']}",
        "",
        "scan_status_counts:",
    ]
    for k, v in sorted(summary["scan_status_counts"].items()):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("suspects_by_label:")
    for k, v in sorted(summary["suspects_by_label"].items()):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("suspect_reasons:")
    for k, v in summary["suspect_reasons"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"clean_manifest: {clean_manifest.as_posix()}")
    lines.append(f"suspects_csv: {suspects_csv.as_posix()}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Input rows: {len(rows)}", flush=True)
    print(f"Excluded negative/proibite: {len(excluded)}", flush=True)
    print(f"Clean rows: {len(kept)}", flush=True)
    print(f"Suspects: {len(suspects_rows)}", flush=True)
    print(f"Clean manifest: {clean_manifest}", flush=True)
    print(f"Suspects CSV: {suspects_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
