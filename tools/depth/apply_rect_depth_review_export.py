#!/usr/bin/env python3
"""Apply RECT_DEPTH review corrections to a legacy manifest.

The review HTML exports rows keyed by the sampled image path and carries manual
``corrected_gt_box`` values in ``left|top|right|bottom`` order.  This utility
turns that review export into a manifest that existing training/evaluation code
can consume, while preserving review metadata in extra columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Box = Tuple[float, float, float, float]


def _norm_path(text: str) -> str:
    return str(text or "").strip()


def _norm_int(text: object) -> str:
    try:
        return str(int(float(str(text).strip())))
    except Exception:
        return str(text or "").strip()


def _review_match_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        _norm_path(str(row.get("image_path") or row.get("source_image") or "")),
        _norm_int(row.get("setup_id")),
        _norm_int(row.get("depth_index0")),
        str(row.get("flip_state") or "").strip().lower(),
    )


def _manifest_match_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        _norm_path(row.get("source_image", "")),
        _norm_int(row.get("setup_id")),
        _norm_int(row.get("depth_index0")),
        str(row.get("flip_state") or "").strip().lower(),
    )


def _parse_box(text: object) -> Optional[Box]:
    if text is None:
        return None
    nums = [float(x) for x in re.split(r"[|,;\s]+", str(text).strip()) if x]
    if len(nums) != 4 or any(not math.isfinite(x) for x in nums):
        return None
    x1, y1, x2, y2 = nums
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_to_manifest_fields(box: Box) -> Dict[str, str]:
    x1, y1, x2, y2 = box
    # Legacy manifest stores inclusive right/bottom-derived width/height.
    return {
        "left": str(int(round(x1))),
        "top": str(int(round(y1))),
        "right": str(int(round(x2))),
        "bottom": str(int(round(y2))),
        "width": str(max(1, int(round(x2 - x1)))),
        "height": str(max(1, int(round(y2 - y1)))),
    }


def load_review(path: Path) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, object]], Counter]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("rows", [])
    out: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    stats: Counter = Counter(raw_rows=len(rows))
    for row in rows:
        key = _review_match_key(row)
        if not key[0]:
            stats["missing_image_path"] += 1
            continue
        if key in out:
            stats["duplicate_review_key"] += 1
        out[key] = row
        if row.get("exclude"):
            stats["review_excluded"] += 1
        if str(row.get("comment") or "").strip():
            stats["review_commented"] += 1
        if _parse_box(row.get("corrected_gt_box") or row.get("correctedBox")):
            stats["review_corrected"] += 1
    stats["review_unique_keys"] = len(out)
    return out, stats


def apply_review(
    manifest: Path,
    review_json: Path,
    output_csv: Path,
    only_reviewed: bool,
    require_corrected: bool,
) -> Dict[str, object]:
    review, stats = load_review(review_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise RuntimeError(f"{manifest} has no CSV header")
        fieldnames = list(reader.fieldnames)
        extra_fields = [
            "review_key",
            "review_exclude",
            "review_comment",
            "review_corrected_gt_box",
            "review_updated_at",
        ]
        for field in extra_fields:
            if field not in fieldnames:
                fieldnames.append(field)

        output_rows: List[Dict[str, str]] = []
        for row in reader:
            stats["manifest_rows"] += 1
            key = _manifest_match_key(row)
            review_row = review.get(key)
            if not review_row:
                if only_reviewed:
                    continue
                output_rows.append(row)
                continue

            stats["matched_manifest_rows"] += 1
            if review_row.get("exclude"):
                stats["excluded_matched_rows"] += 1
                continue

            corrected_text = str(review_row.get("corrected_gt_box") or review_row.get("correctedBox") or "").strip()
            corrected_box = _parse_box(corrected_text)
            if require_corrected and corrected_box is None:
                stats["matched_without_corrected_box"] += 1
                continue

            out_row = dict(row)
            if corrected_box is not None:
                out_row.update(_box_to_manifest_fields(corrected_box))
                stats["applied_corrected_box"] += 1
            out_row["review_key"] = str(review_row.get("key") or "")
            out_row["review_exclude"] = str(bool(review_row.get("exclude"))).lower()
            out_row["review_comment"] = str(review_row.get("comment") or "")
            out_row["review_corrected_gt_box"] = corrected_text
            out_row["review_updated_at"] = str(review_row.get("updated_at") or "")
            output_rows.append(out_row)

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    matched_keys = {_manifest_match_key(row) for row in output_rows if row.get("review_key")}
    stats["review_keys_not_in_output"] = len(set(review) - matched_keys)
    stats["output_rows"] = len(output_rows)
    stats["output_configs"] = len({r.get("config_folder", "") for r in output_rows})
    stats["output_reviewed_rows"] = sum(1 for r in output_rows if r.get("review_key"))
    stats["output_commented_rows"] = sum(1 for r in output_rows if str(r.get("review_comment") or "").strip())
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply RECT_DEPTH review export corrections to a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--only-reviewed", action="store_true", help="Write only rows that appear in the review export.")
    parser.add_argument("--require-corrected", action="store_true", help="Drop reviewed rows without corrected_gt_box.")
    args = parser.parse_args()

    summary = apply_review(
        manifest=args.manifest.expanduser().resolve(),
        review_json=args.review_json.expanduser().resolve(),
        output_csv=args.output_csv.expanduser().resolve(),
        only_reviewed=bool(args.only_reviewed),
        require_corrected=bool(args.require_corrected),
    )
    if args.summary_json:
        args.summary_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.expanduser().resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
