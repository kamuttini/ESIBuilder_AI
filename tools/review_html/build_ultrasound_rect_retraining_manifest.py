#!/usr/bin/env python3
"""Build a retraining manifest from exported correction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:  # noqa: BLE001
        return default


def _parse_correction_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "vendor",
            "group_name",
            "image_path",
            "rel_path",
            "pred_top",
            "pred_left",
            "pred_bottom",
            "pred_right",
            "global_top",
            "global_left",
            "global_bottom",
            "global_right",
            "corr_top",
            "corr_left",
            "corr_bottom",
            "corr_right",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path.name}: missing required columns {sorted(missing)}")

        for row in reader:
            image_path = Path(str(row["image_path"])).expanduser().resolve()
            if not image_path.exists():
                continue
            corr_top = _to_int(str(row["corr_top"]))
            corr_left = _to_int(str(row["corr_left"]))
            corr_bottom = _to_int(str(row["corr_bottom"]))
            corr_right = _to_int(str(row["corr_right"]))
            if corr_bottom <= corr_top or corr_right <= corr_left:
                continue
            rows.append(
                {
                    "source_file": path.as_posix(),
                    "sample_id": str(row["sample_id"]),
                    "vendor": str(row["vendor"]),
                    "group_name": str(row["group_name"]),
                    "image_path": image_path.as_posix(),
                    "rel_path": str(row["rel_path"]),
                    "pred_top": _to_int(str(row["pred_top"])),
                    "pred_left": _to_int(str(row["pred_left"])),
                    "pred_bottom": _to_int(str(row["pred_bottom"])),
                    "pred_right": _to_int(str(row["pred_right"])),
                    "global_top": _to_int(str(row["global_top"])),
                    "global_left": _to_int(str(row["global_left"])),
                    "global_bottom": _to_int(str(row["global_bottom"])),
                    "global_right": _to_int(str(row["global_right"])),
                    "corr_top": corr_top,
                    "corr_left": corr_left,
                    "corr_bottom": corr_bottom,
                    "corr_right": corr_right,
                    "note": str(row.get("note", "")),
                }
            )
    return rows


def _write_manifest(rows: List[Dict[str, object]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_file",
                "sample_id",
                "vendor",
                "group_name",
                "image_path",
                "rel_path",
                "pred_top",
                "pred_left",
                "pred_bottom",
                "pred_right",
                "global_top",
                "global_left",
                "global_bottom",
                "global_right",
                "target_top",
                "target_left",
                "target_bottom",
                "target_right",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": row["source_file"],
                    "sample_id": row["sample_id"],
                    "vendor": row["vendor"],
                    "group_name": row["group_name"],
                    "image_path": row["image_path"],
                    "rel_path": row["rel_path"],
                    "pred_top": row["pred_top"],
                    "pred_left": row["pred_left"],
                    "pred_bottom": row["pred_bottom"],
                    "pred_right": row["pred_right"],
                    "global_top": row["global_top"],
                    "global_left": row["global_left"],
                    "global_bottom": row["global_bottom"],
                    "global_right": row["global_right"],
                    "target_top": row["corr_top"],
                    "target_left": row["corr_left"],
                    "target_bottom": row["corr_bottom"],
                    "target_right": row["corr_right"],
                    "note": row["note"],
                }
            )


def _write_summary(rows: List[Dict[str, object]], corrections_files: List[Path], summary_json: Path) -> None:
    vendor_counts = Counter(str(r["vendor"]) for r in rows)
    group_counts = Counter(str(r["group_name"]) for r in rows)
    source_counts = Counter(str(r["source_file"]) for r in rows)

    payload = {
        "num_correction_files": len(corrections_files),
        "num_rows": len(rows),
        "vendors": dict(sorted(vendor_counts.items())),
        "groups_count": len(group_counts),
        "top_groups": [
            {"group_name": g, "rows": c}
            for g, c in group_counts.most_common(25)
        ],
        "rows_per_source_file": dict(sorted(source_counts.items())),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolida i corrections_*.csv esportati dalle review HTML in un manifest unico per retraining."
    )
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        required=True,
        help="Directory dove hai salvato i file corrections_*.csv esportati dal browser.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/ultrasound_rect_retraining_from_reviews"),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="corrections_*.csv",
        help="Glob pattern dei file correzione (default: corrections_*.csv).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    reviews_dir = args.reviews_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reviews_dir.is_dir():
        raise FileNotFoundError(f"reviews-dir not found: {reviews_dir}")

    files = sorted(reviews_dir.rglob(args.pattern))
    if not files:
        raise RuntimeError(f"No correction files found with pattern '{args.pattern}' under {reviews_dir}")

    all_rows: List[Dict[str, object]] = []
    errors: Dict[str, str] = {}
    for path in files:
        try:
            all_rows.extend(_parse_correction_csv(path))
        except Exception as exc:  # noqa: BLE001
            errors[path.as_posix()] = str(exc)

    if not all_rows:
        raise RuntimeError("No valid correction rows parsed.")

    manifest_csv = output_dir / "retraining_manifest.csv"
    _write_manifest(all_rows, manifest_csv)
    summary_json = output_dir / "retraining_manifest_summary.json"
    _write_summary(all_rows, files, summary_json)

    if errors:
        (output_dir / "retraining_manifest_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Correction files found: {len(files)}", flush=True)
    print(f"Rows exported: {len(all_rows)}", flush=True)
    print(f"Manifest: {manifest_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    if errors:
        print(f"Files with parse errors: {len(errors)} (see retraining_manifest_errors.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
