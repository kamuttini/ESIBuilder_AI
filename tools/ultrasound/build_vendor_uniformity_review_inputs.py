#!/usr/bin/env python3
"""Build one-sample-per-folder manifest with GT uniformity metrics for manual review."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a one-image-per-folder manifest with GT crop uniformity metrics "
            "(std/unique/range), to manually review possible flat-color template crops."
        )
    )
    p.add_argument("--manifest", type=Path, required=True, help="Training manifest CSV.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--selection",
        type=str,
        default="most_uniform",
        choices=("first", "middle", "last", "random", "most_uniform"),
        help=(
            "How to choose one representative image per folder. "
            "'most_uniform' scans all images and picks the most flat-looking crop."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for random selection.")
    p.add_argument(
        "--std-threshold",
        type=float,
        default=1.0,
        help="Threshold for std intensity to mark a row as uniform_suspect.",
    )
    p.add_argument(
        "--unique-threshold",
        type=int,
        default=3,
        help="Threshold for number of unique intensity values to mark uniform_suspect.",
    )
    p.add_argument(
        "--range-threshold",
        type=float,
        default=2.0,
        help="Threshold for (max-min) intensity to mark uniform_suspect.",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for full-image scan.")
    p.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N scanned rows.",
    )
    return p.parse_args()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and not value.strip():
            return int(default)
        if isinstance(value, float) and math.isnan(value):
            return int(default)
        return int(round(float(value)))
    except Exception:
        return int(default)


def _pick_one(rows: List[Dict[str, object]], mode: str, rng: random.Random) -> Dict[str, object]:
    ordered = sorted(rows, key=lambda r: str(r.get("image_path", "")))
    if not ordered:
        raise RuntimeError("Cannot pick from empty rows.")
    if mode == "first":
        return ordered[0]
    if mode == "last":
        return ordered[-1]
    if mode == "random":
        return ordered[rng.randrange(len(ordered))]
    return ordered[len(ordered) // 2]


def _pick_most_uniform(rows: List[Dict[str, object]]) -> Dict[str, object]:
    def _rank(row: Dict[str, object]) -> tuple[float, float, float, str]:
        # Lower is "more uniform/suspicious"
        std = float(row.get("gt_std", 1e9))
        uniq = float(row.get("gt_unique", 1e9))
        rng = float(row.get("gt_range", 1e9))
        path = str(row.get("image_path", ""))
        return (std, uniq, rng, path)

    return min(rows, key=_rank)


def _bbox_from_row(row: Dict[str, object]) -> tuple[int, int, int, int]:
    x1 = _safe_int(row.get("bbox_xmin", row.get("bbox_left", 0)))
    y1 = _safe_int(row.get("bbox_ymin", row.get("bbox_top", 0)))
    x2 = _safe_int(row.get("bbox_xmax", row.get("bbox_right", x1 + 1)))
    y2 = _safe_int(row.get("bbox_ymax", row.get("bbox_bottom", y1 + 1)))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def _to_gray_intensity(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop.astype(np.uint8, copy=False)
    if crop.ndim == 3:
        # Fast proxy intensity: use first channel only.
        return crop[..., 0].astype(np.uint8, copy=False)
    return crop.astype(np.uint8, copy=False)


def _compute_metrics(image_path: str, bbox: tuple[int, int, int, int]) -> Optional[Dict[str, float]]:
    p = Path(image_path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            arr = np.array(im)
    except Exception:
        return None
    if arr.ndim < 2:
        return None

    h, w = arr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = _to_gray_intensity(crop)
    mn = int(np.min(gray))
    mx = int(np.max(gray))
    std = float(np.std(gray))
    mean = float(np.mean(gray))
    # Faster than np.unique for small-value-domain images (uint8).
    hist = np.bincount(gray.ravel(), minlength=256)
    uniq = int(np.count_nonzero(hist))
    return {
        "bbox_xmin": int(x1),
        "bbox_ymin": int(y1),
        "bbox_xmax": int(x2),
        "bbox_ymax": int(y2),
        "gt_min": float(mn),
        "gt_max": float(mx),
        "gt_mean": mean,
        "gt_std": std,
        "gt_unique": int(uniq),
        "gt_range": float(mx - mn),
    }


def _scan_task(task: tuple[int, str, int, int, int, int]) -> tuple[int, str, Optional[Dict[str, float]]]:
    idx, image_path, x1, y1, x2, y2 = task
    bbox = (x1, y1, x2, y2)
    p = Path(image_path)
    if not p.exists():
        return idx, "missing", None
    metrics = _compute_metrics(image_path=image_path, bbox=bbox)
    if metrics is None:
        return idx, "invalid", None
    return idx, "ok", metrics


def _apply_uniform_flags(
    row: Dict[str, object],
    std_threshold: float,
    unique_threshold: int,
    range_threshold: float,
) -> Dict[str, object]:
    gt_max = float(row.get("gt_max", 0.0))
    gt_std = float(row.get("gt_std", 1e9))
    gt_unique = int(row.get("gt_unique", 0))
    gt_range = float(row.get("gt_range", 1e9))

    is_black = int(gt_max <= 0.0)
    is_uniform_exact = int(gt_std == 0.0 or gt_unique <= 1)
    is_uniform_like = int(
        gt_std <= float(std_threshold)
        or gt_unique <= int(unique_threshold)
        or gt_range <= float(range_threshold)
    )
    reasons: List[str] = []
    if is_black:
        reasons.append("black_exact")
    if gt_std <= float(std_threshold):
        reasons.append(f"std<={std_threshold}")
    if gt_unique <= int(unique_threshold):
        reasons.append(f"unique<={unique_threshold}")
    if gt_range <= float(range_threshold):
        reasons.append(f"range<={range_threshold}")

    row["uniform_black_exact"] = int(is_black)
    row["uniform_exact"] = int(is_uniform_exact)
    row["uniform_suspect"] = int(is_uniform_like)
    row["uniform_reason"] = "|".join(reasons)
    return row


def main() -> int:
    args = _parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if df.empty:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    rows = df.to_dict(orient="records")
    by_folder_source: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        folder = str(row.get("dataset_folder", "") or "").strip()
        if not folder:
            folder = str(row.get("group_id", "") or "").strip()
        if not folder:
            continue
        by_folder_source[folder].append(idx)

    if not by_folder_source:
        raise RuntimeError("No folder groups found in manifest.")

    rng = random.Random(int(args.seed))
    tasks: List[tuple[int, str, int, int, int, int]] = []
    for idx, row in enumerate(rows):
        image_path = str(row.get("image_path", "") or "").strip()
        x1, y1, x2, y2 = _bbox_from_row(row)
        tasks.append((idx, image_path, x1, y1, x2, y2))

    workers = max(1, int(args.workers))
    workers = min(workers, os.cpu_count() or workers)
    print(f"Scanning all rows: {len(tasks)} | workers={workers}", flush=True)

    missing = 0
    invalid = 0
    analyzed_all_rows: List[Dict[str, object]] = []
    by_folder_analyzed: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            result_iter = pool.imap_unordered(_scan_task, tasks, chunksize=64)
            for done, (idx, status, metrics) in enumerate(result_iter, start=1):
                row = rows[idx]
                if status == "missing":
                    missing += 1
                elif status == "invalid" or metrics is None:
                    invalid += 1
                else:
                    out_row: Dict[str, object] = dict(row)
                    out_row.update(metrics)
                    folder = str(out_row.get("dataset_folder", "") or str(out_row.get("group_id", ""))).strip()
                    out_row["review_folder_key"] = folder
                    _apply_uniform_flags(
                        out_row,
                        std_threshold=float(args.std_threshold),
                        unique_threshold=int(args.unique_threshold),
                        range_threshold=float(args.range_threshold),
                    )
                    analyzed_all_rows.append(out_row)
                    by_folder_analyzed[folder].append(out_row)

                if done % max(1, int(args.progress_every)) == 0:
                    print(
                        f"progress {done}/{len(tasks)} analyzed={len(analyzed_all_rows)} "
                        f"missing={missing} invalid={invalid}",
                        flush=True,
                    )
    else:
        result_iter = map(_scan_task, tasks)
        for done, (idx, status, metrics) in enumerate(result_iter, start=1):
            row = rows[idx]
            if status == "missing":
                missing += 1
            elif status == "invalid" or metrics is None:
                invalid += 1
            else:
                out_row = dict(row)
                out_row.update(metrics)
                folder = str(out_row.get("dataset_folder", "") or str(out_row.get("group_id", ""))).strip()
                out_row["review_folder_key"] = folder
                _apply_uniform_flags(
                    out_row,
                    std_threshold=float(args.std_threshold),
                    unique_threshold=int(args.unique_threshold),
                    range_threshold=float(args.range_threshold),
                )
                analyzed_all_rows.append(out_row)
                by_folder_analyzed[folder].append(out_row)
            if done % max(1, int(args.progress_every)) == 0:
                print(
                    f"progress {done}/{len(tasks)} analyzed={len(analyzed_all_rows)} "
                    f"missing={missing} invalid={invalid}",
                    flush=True,
                )

    if not analyzed_all_rows:
        raise RuntimeError("No analyzable rows found after full scan.")

    selection_mode = str(args.selection)
    selected_rows: List[Dict[str, object]] = []
    for folder in sorted(by_folder_analyzed.keys()):
        rows_folder = by_folder_analyzed[folder]
        if selection_mode == "most_uniform":
            picked = _pick_most_uniform(rows_folder)
        else:
            picked = _pick_one(rows_folder, mode=selection_mode, rng=rng)
        out_row = dict(picked)
        if int(out_row.get("uniform_suspect", 0)) == 1:
            out_row["manual_folder_action"] = "uniform_suspect"
        selected_rows.append(out_row)

    all_df = pd.DataFrame(analyzed_all_rows)
    selected_df = pd.DataFrame(selected_rows)

    all_metrics_csv = out_dir / "uniformity_metrics_all_rows.csv"
    all_df.to_csv(all_metrics_csv, index=False)

    out_manifest = out_dir / "manifest_one_per_folder_uniformity.csv"
    selected_df.to_csv(out_manifest, index=False)

    suspects_all_df = all_df[all_df["uniform_suspect"] == 1].copy()
    suspects_all_csv = out_dir / "uniform_suspects_all_rows.csv"
    suspects_all_df.to_csv(suspects_all_csv, index=False)

    suspects_selected_df = selected_df[selected_df["uniform_suspect"] == 1].copy()
    suspects_selected_csv = out_dir / "uniform_suspects_one_per_folder.csv"
    suspects_selected_df.to_csv(suspects_selected_csv, index=False)

    suspect_counter_selected: Counter[str] = Counter(
        str(v) for v in selected_df[selected_df["uniform_suspect"] == 1]["manufacturer"].tolist()
    )
    vendor_counter_selected: Counter[str] = Counter(str(v) for v in selected_df["manufacturer"].tolist())

    summary = {
        "source_manifest": manifest_path.as_posix(),
        "output_manifest": out_manifest.as_posix(),
        "output_all_metrics_csv": all_metrics_csv.as_posix(),
        "output_suspects_all_rows_csv": suspects_all_csv.as_posix(),
        "output_suspects_one_per_folder_csv": suspects_selected_csv.as_posix(),
        "selection_mode": selection_mode,
        "seed": int(args.seed),
        "thresholds": {
            "std_threshold": float(args.std_threshold),
            "unique_threshold": int(args.unique_threshold),
            "range_threshold": float(args.range_threshold),
        },
        "rows_total_in_manifest": int(len(rows)),
        "folders_total": int(len(by_folder_source)),
        "rows_scanned_ok": int(len(all_df)),
        "missing_images": int(missing),
        "invalid_bbox_or_crop": int(invalid),
        "suspects_all_rows_total": int(len(suspects_all_df)),
        "black_exact_all_rows_total": int(all_df["uniform_black_exact"].sum()),
        "uniform_exact_all_rows_total": int(all_df["uniform_exact"].sum()),
        "rows_selected_one_per_folder": int(len(selected_df)),
        "suspects_one_per_folder_total": int(len(suspects_selected_df)),
        "rows_per_vendor_one_per_folder": dict(sorted((k, int(v)) for k, v in vendor_counter_selected.items())),
        "suspects_per_vendor_one_per_folder": dict(
            sorted((k, int(v)) for k, v in suspect_counter_selected.items())
        ),
    }
    summary_path = out_dir / "uniformity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source manifest: {manifest_path}", flush=True)
    print(f"All-rows metrics CSV: {all_metrics_csv}", flush=True)
    print(f"One-per-folder manifest: {out_manifest}", flush=True)
    print(f"Uniform suspects all-rows CSV: {suspects_all_csv}", flush=True)
    print(f"Uniform suspects one-per-folder CSV: {suspects_selected_csv}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(
        f"Rows scanned OK: {summary['rows_scanned_ok']} | Suspects(all): {summary['suspects_all_rows_total']} | "
        f"One-per-folder: {summary['rows_selected_one_per_folder']} | Suspects(selected): {summary['suspects_one_per_folder_total']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
