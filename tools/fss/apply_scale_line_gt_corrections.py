#!/usr/bin/env python3
"""Apply per-sample scale-line GT corrections to a manifest CSV.

Expected corrections CSV columns (preferred):
- sample_id
- x_gt_new
- y_top_gt_new
- y_bottom_gt_new

Accepted aliases:
- x_gt_new: x_new, x_corr, x
- y_top_gt_new: y_top_new, y1_new, y_top_corr, y1
- y_bottom_gt_new: y_bottom_new, y2_new, y_bottom_corr, y2

Notes:
- Coordinates are interpreted in manifest/video pixel coordinates.
- Updates x1=x2=x_gt_new and y1=min(y_top,y_bottom), y2=max(y_top,y_bottom).
- If present, normalized fields (x1_norm/x2_norm/y1_norm/y2_norm), length_px and
  is_inside_frame are updated coherently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Correction:
    sample_id: str
    x_new: float
    y_top_new: float
    y_bottom_new: float
    note: str
    source_csv: str
    source_row: int


def _first_non_empty(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _parse_float(text: str) -> Optional[float]:
    t = str(text).strip()
    if not t:
        return None
    try:
        v = float(t)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _parse_int(text: str) -> Optional[int]:
    t = str(text).strip()
    if not t:
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _fmt(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def load_corrections(paths: Sequence[Path]) -> Tuple[Dict[str, Correction], Dict[str, int]]:
    corrections: Dict[str, Correction] = {}
    stats = {
        "files_read": 0,
        "rows_total": 0,
        "rows_valid": 0,
        "rows_invalid": 0,
        "rows_missing_sample_id": 0,
        "rows_missing_coords": 0,
        "rows_overwritten_by_later_file": 0,
    }

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Corrections CSV not found: {path}")

        stats["files_read"] += 1
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader, start=2):
                stats["rows_total"] += 1

                sample_id = _first_non_empty(row, ["sample_id", "id", "sample"])
                if not sample_id:
                    stats["rows_missing_sample_id"] += 1
                    stats["rows_invalid"] += 1
                    continue

                x_new = _parse_float(_first_non_empty(row, ["x_gt_new", "x_new", "x_corr", "x"]))
                y_top_new = _parse_float(
                    _first_non_empty(row, ["y_top_gt_new", "y_top_new", "y1_new", "y_top_corr", "y1"])
                )
                y_bottom_new = _parse_float(
                    _first_non_empty(row, ["y_bottom_gt_new", "y_bottom_new", "y2_new", "y_bottom_corr", "y2"])
                )

                if x_new is None or y_top_new is None or y_bottom_new is None:
                    stats["rows_missing_coords"] += 1
                    stats["rows_invalid"] += 1
                    continue

                if sample_id in corrections:
                    stats["rows_overwritten_by_later_file"] += 1

                corrections[sample_id] = Correction(
                    sample_id=sample_id,
                    x_new=float(x_new),
                    y_top_new=float(y_top_new),
                    y_bottom_new=float(y_bottom_new),
                    note=_first_non_empty(row, ["note", "notes", "comment"]),
                    source_csv=path.as_posix(),
                    source_row=row_idx,
                )
                stats["rows_valid"] += 1

    return corrections, stats


def apply_corrections_to_rows(
    rows: List[Dict[str, str]],
    corrections: Dict[str, Correction],
    clamp_to_frame: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], Dict[str, int]]:
    applied_rows: List[Dict[str, object]] = []
    matched_sample_ids: set[str] = set()
    stats = {
        "manifest_rows_total": len(rows),
        "manifest_rows_touched": 0,
        "manifest_rows_changed": 0,
        "manifest_rows_missing_video_size": 0,
    }

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        corr = corrections.get(sample_id)
        if corr is None:
            continue

        matched_sample_ids.add(sample_id)
        stats["manifest_rows_touched"] += 1

        vx = _parse_int(row.get("video_x_size", ""))
        vy = _parse_int(row.get("video_y_size", ""))
        if vx is None or vy is None or vx <= 0 or vy <= 0:
            stats["manifest_rows_missing_video_size"] += 1
            continue

        x_old = _parse_float(row.get("x1", ""))
        y1_old = _parse_float(row.get("y1", ""))
        y2_old = _parse_float(row.get("y2", ""))

        x_new = float(corr.x_new)
        y1_new = float(min(corr.y_top_new, corr.y_bottom_new))
        y2_new = float(max(corr.y_top_new, corr.y_bottom_new))

        if clamp_to_frame:
            x_new = float(max(0.0, min(float(vx - 1), x_new)))
            y1_new = float(max(0.0, min(float(vy - 1), y1_new)))
            y2_new = float(max(0.0, min(float(vy - 1), y2_new)))

        row["x1"] = _fmt(x_new)
        row["x2"] = _fmt(x_new)
        row["y1"] = _fmt(y1_new)
        row["y2"] = _fmt(y2_new)

        if "x1_norm" in row:
            row["x1_norm"] = _fmt(x_new / float(vx))
        if "x2_norm" in row:
            row["x2_norm"] = _fmt(x_new / float(vx))
        if "y1_norm" in row:
            row["y1_norm"] = _fmt(y1_new / float(vy))
        if "y2_norm" in row:
            row["y2_norm"] = _fmt(y2_new / float(vy))

        if "length_px" in row:
            row["length_px"] = _fmt(abs(y2_new - y1_new))
        if "is_inside_frame" in row:
            row["is_inside_frame"] = "1"

        changed = False
        if x_old is None or y1_old is None or y2_old is None:
            changed = True
        else:
            changed = (
                abs(x_new - x_old) > 1e-6
                or abs(y1_new - y1_old) > 1e-6
                or abs(y2_new - y2_old) > 1e-6
            )

        if changed:
            stats["manifest_rows_changed"] += 1

        applied_rows.append(
            {
                "sample_id": sample_id,
                "source_csv": corr.source_csv,
                "source_row": corr.source_row,
                "x_old": "" if x_old is None else _fmt(x_old),
                "y1_old": "" if y1_old is None else _fmt(y1_old),
                "y2_old": "" if y2_old is None else _fmt(y2_old),
                "x_new": _fmt(x_new),
                "y1_new": _fmt(y1_new),
                "y2_new": _fmt(y2_new),
                "changed": 1 if changed else 0,
                "note": corr.note,
            }
        )

    unmatched = sorted(set(corrections.keys()) - matched_sample_ids)
    stats["correction_sample_ids_total"] = len(corrections)
    stats["correction_sample_ids_matched"] = len(matched_sample_ids)
    stats["correction_sample_ids_unmatched"] = len(unmatched)

    return rows, applied_rows, {**stats, "_unmatched_list_len": len(unmatched)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply scale-line GT corrections CSV to manifest CSV.")
    p.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV to patch.")
    p.add_argument(
        "--corrections-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV files exported from GT review HTML (or compatible format).",
    )
    p.add_argument("--output-manifest", type=Path, required=True, help="Output patched manifest CSV.")
    p.add_argument(
        "--applied-csv",
        type=Path,
        default=None,
        help="Optional audit CSV of applied rows (default: <output-manifest stem>_applied.csv).",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path (default: <output-manifest stem>_summary.json).",
    )
    p.add_argument("--no-clamp", action="store_true", help="Do not clamp corrected coordinates to frame bounds.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if one or more correction sample_id values are not found in the manifest.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    corrections_csvs = [p.expanduser().resolve() for p in args.corrections_csv]

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    corrections, corr_stats = load_corrections(corrections_csvs)
    if not corrections:
        raise RuntimeError("No valid corrections were loaded from CSV files.")

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    patched_rows, applied_rows, apply_stats = apply_corrections_to_rows(
        rows=rows,
        corrections=corrections,
        clamp_to_frame=not args.no_clamp,
    )

    unmatched_sample_ids: List[str] = []
    matched_ids = {r["sample_id"] for r in applied_rows}
    for sid in sorted(corrections.keys()):
        if sid not in matched_ids:
            unmatched_sample_ids.append(sid)

    if args.strict and unmatched_sample_ids:
        raise RuntimeError(
            "Found correction sample_id values missing from manifest (strict mode): "
            + ", ".join(unmatched_sample_ids[:20])
            + (" ..." if len(unmatched_sample_ids) > 20 else "")
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patched_rows)

    applied_csv = args.applied_csv
    if applied_csv is None:
        applied_csv = output_manifest.with_name(output_manifest.stem + "_applied.csv")
    applied_csv = applied_csv.expanduser().resolve()
    applied_csv.parent.mkdir(parents=True, exist_ok=True)
    with applied_csv.open("w", encoding="utf-8", newline="") as fh:
        header = [
            "sample_id",
            "source_csv",
            "source_row",
            "x_old",
            "y1_old",
            "y2_old",
            "x_new",
            "y1_new",
            "y2_new",
            "changed",
            "note",
        ]
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(applied_rows)

    summary_json = args.summary_json
    if summary_json is None:
        summary_json = output_manifest.with_name(output_manifest.stem + "_summary.json")
    summary_json = summary_json.expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "manifest_in": manifest.as_posix(),
        "manifest_out": output_manifest.as_posix(),
        "corrections_csv": [p.as_posix() for p in corrections_csvs],
        "applied_csv": applied_csv.as_posix(),
        "clamp_to_frame": bool(not args.no_clamp),
        "strict": bool(args.strict),
        "corrections_load_stats": corr_stats,
        "apply_stats": {
            k: v
            for k, v in apply_stats.items()
            if not str(k).startswith("_")
        },
        "unmatched_sample_ids_count": len(unmatched_sample_ids),
        "unmatched_sample_ids_preview": unmatched_sample_ids[:50],
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
