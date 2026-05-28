#!/usr/bin/env python3
"""Audit vendor template GT crops that are completely black.

This script scans a manifest row by row, extracts the GT crop using bbox_xmin/ymin/xmax/ymax,
flags rows where crop pixels are all zero, generates a visual report, and exports a filtered
manifest without black GT rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit black GT crops in vendor manifest")
    parser.add_argument("--manifest", required=True, help="Input CSV manifest")
    parser.add_argument("--output-dir", required=True, help="Output directory for report + filtered manifest")
    parser.add_argument("--chunksize", type=int, default=1000, help="CSV read chunk size")
    parser.add_argument("--progress-every", type=int, default=2000, help="Print progress every N rows")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for image scan")
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(round(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _sanitize_name(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_len]


def _overlay_with_bbox(image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    out = image.convert("RGB")
    drawer = ImageDraw.Draw(out)
    drawer.rectangle([x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)], outline=(255, 0, 0), width=3)
    return out


def _scan_one(
    task: tuple[int, str, int | None, int | None, int | None, int | None]
) -> tuple[int, str, tuple[int, int, int, int] | None]:
    local_idx, image_path, x1, y1, x2, y2 = task
    if None in (x1, y1, x2, y2):
        return local_idx, "invalid", None

    try:
        with Image.open(image_path) as im:
            image_np = np.array(im)
    except Exception:
        return local_idx, "missing", None

    if image_np.ndim < 2:
        return local_idx, "missing", None

    h, w = image_np.shape[:2]
    x1_i = max(0, min(int(x1), w - 1))
    y1_i = max(0, min(int(y1), h - 1))
    x2_i = max(0, min(int(x2), w))
    y2_i = max(0, min(int(y2), h))

    if x2_i <= x1_i or y2_i <= y1_i:
        return local_idx, "invalid", None

    crop = image_np[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return local_idx, "invalid", None

    if int(np.max(crop)) == 0:
        return local_idx, "black", (x1_i, y1_i, x2_i, y2_i)
    return local_idx, "ok", (x1_i, y1_i, x2_i, y2_i)


def _write_html(report_path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows_html: list[str] = []
    for rec in records:
        rows_html.append(
            f"""
            <div class="card">
              <div class="meta">
                <div><b>Row</b>: {rec['row_index']}</div>
                <div><b>Vendor</b>: {rec['manufacturer']}</div>
                <div><b>Folder</b>: {rec['dataset_folder']}</div>
                <div><b>Split</b>: {rec['split']}</div>
                <div><b>BBox</b>: ({rec['bbox_xmin']},{rec['bbox_ymin']})-({rec['bbox_xmax']},{rec['bbox_ymax']})</div>
                <div><b>Image</b>: {rec['image_path']}</div>
              </div>
              <div class="images">
                <div>
                  <div class="label">Overlay</div>
                  <img src="{rec['overlay_rel']}" alt="overlay" loading="lazy" />
                </div>
                <div>
                  <div class="label">GT crop (black)</div>
                  <img src="{rec['crop_rel']}" alt="crop" loading="lazy" />
                </div>
              </div>
            </div>
            """
        )

    summary_html = (
        f"Scanned: {summary['total_rows_scanned']} | "
        f"Black GT: {summary['black_gt_rows']} | "
        f"Kept: {summary['kept_rows']} | "
        f"Missing image: {summary['missing_image_rows']} | "
        f"Invalid bbox: {summary['invalid_bbox_rows']}"
    )

    vendors_html = "".join(
        f"<li><b>{vendor}</b>: {count}</li>" for vendor, count in summary["black_by_vendor"].items()
    )
    if not vendors_html:
        vendors_html = "<li>None</li>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Black GT Audit</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 20px;
      background: #f5f7fb;
      color: #1f2937;
    }}
    .header {{
      background: #fff;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 14px;
    }}
    .meta {{
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 10px;
      word-break: break-word;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .images img {{
      width: 100%;
      height: auto;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #111;
    }}
    .label {{
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #374151;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1 style="margin:0 0 8px 0;">Black GT Audit</h1>
    <div style="font-size:14px;">{summary_html}</div>
    <div style="margin-top:10px;">
      <b>Black GT by vendor</b>
      <ul style="margin:6px 0 0 18px;">{vendors_html}</ul>
    </div>
  </div>
  <div class="cards">
    {''.join(rows_html) if rows_html else '<div class="card"><b>No completely black GT crop found.</b></div>'}
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"
    overlay_dir = out_dir / "overlays"
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    black_rows_csv = out_dir / "black_gt_rows.csv"
    filtered_manifest_csv = out_dir / "manifest_training_ready_ui_corrected_clean_no_black_gt.csv"
    summary_json = out_dir / "black_gt_summary.json"
    report_html = out_dir / "index.html"

    expected_cols = {"image_path", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"}
    header_cols = list(pd.read_csv(manifest_path, nrows=0).columns)
    missing_cols = sorted(expected_cols - set(header_cols))
    if missing_cols:
        raise ValueError(f"Missing required columns in manifest: {missing_cols}")

    started_at = time.time()
    total_rows = 0
    black_rows = 0
    missing_images = 0
    invalid_bbox = 0
    preview_export_errors = 0
    header_written = False
    black_records: list[dict[str, Any]] = []
    black_by_vendor: Counter[str] = Counter()
    workers = max(1, int(args.workers))
    if workers > 1:
        workers = min(workers, os.cpu_count() or workers)
        print(f"[info] parallel scan enabled with workers={workers}", flush=True)

    executor: concurrent.futures.ProcessPoolExecutor | None = None
    try:
        if workers > 1:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

        for chunk in pd.read_csv(manifest_path, chunksize=args.chunksize):
            keep_mask = np.ones(len(chunk), dtype=bool)
            rows = chunk.to_dict(orient="records")
            abs_row_idx: list[int] = []
            tasks: list[tuple[int, str, int | None, int | None, int | None, int | None]] = []

            for idx, row in enumerate(rows):
                total_rows += 1
                abs_row_idx.append(total_rows)
                tasks.append(
                    (
                        idx,
                        str(row.get("image_path", "")),
                        _safe_int(row.get("bbox_xmin")),
                        _safe_int(row.get("bbox_ymin")),
                        _safe_int(row.get("bbox_xmax")),
                        _safe_int(row.get("bbox_ymax")),
                    )
                )

            if executor is not None:
                results = executor.map(_scan_one, tasks, chunksize=64)
            else:
                results = map(_scan_one, tasks)

            for local_idx, status, bbox in results:
                row = rows[local_idx]
                row_index = abs_row_idx[local_idx]
                image_path = str(row.get("image_path", ""))

                if status == "missing":
                    missing_images += 1
                    continue
                if status == "invalid":
                    invalid_bbox += 1
                    continue
                if status != "black" or bbox is None:
                    continue

                keep_mask[local_idx] = False
                black_rows += 1
                x1, y1, x2, y2 = bbox

                vendor = str(row.get("manufacturer", "unknown"))
                folder = str(row.get("dataset_folder", ""))
                split = str(row.get("split", ""))
                stem = _sanitize_name(f"row_{row_index:06d}_{vendor}_{Path(image_path).stem}")
                crop_path = crop_dir / f"{stem}.png"
                overlay_path = overlay_dir / f"{stem}.jpg"
                crop_rel = f"crops/{crop_path.name}"
                overlay_rel = f"overlays/{overlay_path.name}"

                try:
                    with Image.open(image_path) as im:
                        pil_image = im.copy()
                    image_np = np.array(pil_image)
                    crop = image_np[y1:y2, x1:x2]
                    Image.fromarray(crop).save(crop_path)
                    overlay = _overlay_with_bbox(pil_image, x1, y1, x2, y2)
                    if overlay.width > 1000:
                        scale = 1000.0 / overlay.width
                        new_w = 1000
                        new_h = max(1, int(round(overlay.height * scale)))
                        overlay = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    overlay.save(overlay_path, quality=90)
                except Exception:
                    preview_export_errors += 1
                    crop_rel = ""
                    overlay_rel = ""

                black_by_vendor[vendor] += 1
                black_records.append(
                    {
                        "row_index": row_index,
                        "manufacturer": vendor,
                        "dataset_folder": folder,
                        "split": split,
                        "image_path": image_path,
                        "bbox_xmin": x1,
                        "bbox_ymin": y1,
                        "bbox_xmax": x2,
                        "bbox_ymax": y2,
                        "crop_path": str(crop_path),
                        "overlay_path": str(overlay_path),
                        "crop_rel": crop_rel,
                        "overlay_rel": overlay_rel,
                    }
                )

            filtered_chunk = chunk.loc[keep_mask]
            filtered_chunk.to_csv(
                filtered_manifest_csv,
                mode="w" if not header_written else "a",
                header=not header_written,
                index=False,
            )
            header_written = True

            if total_rows % args.progress_every == 0:
                elapsed = time.time() - started_at
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] rows={total_rows} black={black_rows} "
                    f"missing={missing_images} invalid={invalid_bbox} rate={rate:.1f} rows/s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    black_df = pd.DataFrame(black_records)
    black_df.to_csv(black_rows_csv, index=False)

    summary = {
        "manifest_path": str(manifest_path),
        "total_rows_scanned": total_rows,
        "black_gt_rows": black_rows,
        "kept_rows": int(total_rows - black_rows),
        "missing_image_rows": missing_images,
        "invalid_bbox_rows": invalid_bbox,
        "preview_export_errors": preview_export_errors,
        "black_by_vendor": dict(sorted(black_by_vendor.items())),
        "black_rows_csv": str(black_rows_csv),
        "filtered_manifest_csv": str(filtered_manifest_csv),
        "report_html": str(report_html),
        "generated_at_epoch": int(time.time()),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_html(report_html, black_records, summary)

    elapsed = time.time() - started_at
    print(
        f"[done] scanned={total_rows} black={black_rows} kept={total_rows - black_rows} "
        f"missing={missing_images} invalid={invalid_bbox} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    print(f"[out] summary={summary_json}", flush=True)
    print(f"[out] black_rows={black_rows_csv}", flush=True)
    print(f"[out] filtered_manifest={filtered_manifest_csv}", flush=True)
    print(f"[out] report_html={report_html}", flush=True)


if __name__ == "__main__":
    main()
