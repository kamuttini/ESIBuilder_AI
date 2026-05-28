#!/usr/bin/env python3
"""Apply real image exclusions CSV to a transrectal LT manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _load_excluded_paths(decisions_csv: Path) -> Set[str]:
    excluded: Set[str] = set()
    with decisions_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            exclude_real = (row.get("exclude_real") or "").strip()
            if exclude_real == "1":
                excluded.add(image_path)
    return excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply exported real exclusions to create final training manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/manifest_transrectal_lt_rect_training_clean.csv"),
        help="Input manifest to filter.",
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        required=True,
        help="CSV exported from quality_exclusion_review.html (transrectal_real_exclusions.csv or all decisions).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/manifest_transrectal_lt_rect_training_final.csv"),
        help="Output manifest after exclusions.",
    )
    parser.add_argument(
        "--excluded-rows-csv",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/excluded_real_rows.csv"),
        help="Rows removed from manifest.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    decisions_csv = args.decisions_csv.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    excluded_rows_csv = args.excluded_rows_csv.expanduser().resolve()

    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest non trovato: {manifest}")
    if not decisions_csv.is_file():
        raise FileNotFoundError(f"Decisions CSV non trovato: {decisions_csv}")

    excluded_paths = _load_excluded_paths(decisions_csv)
    rows, fields = _read_csv(manifest)

    kept: List[Dict[str, str]] = []
    removed: List[Dict[str, str]] = []
    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        if image_path and image_path in excluded_paths:
            removed.append(row)
        else:
            kept.append(row)

    _write_csv(output_manifest, kept, fields)
    _write_csv(excluded_rows_csv, removed, fields)

    print(f"Input rows: {len(rows)}", flush=True)
    print(f"Excluded paths in decisions: {len(excluded_paths)}", flush=True)
    print(f"Removed rows: {len(removed)}", flush=True)
    print(f"Output rows: {len(kept)}", flush=True)
    print(f"Output manifest: {output_manifest}", flush=True)
    print(f"Removed rows CSV: {excluded_rows_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
