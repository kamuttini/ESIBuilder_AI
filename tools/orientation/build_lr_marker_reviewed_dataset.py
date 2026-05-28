#!/usr/bin/env python3
"""Aggregate reviewed SU/GIU-driven LR marker batches into a train-ready v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


EXCLUDE_DECISIONS = {"needs_fix", "rejected"}


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _load_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sample_id = row.get("sample_id", "")
            if sample_id:
                out[sample_id] = dict(row)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _count(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = row.get(field, "") or "(empty)"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _is_dark_template(path: Path, max_value: int) -> bool:
    try:
        with Image.open(path) as img:
            extrema = img.convert("L").getextrema()
    except Exception:
        return False
    if extrema is None:
        return False
    _min_value, template_max = extrema
    return int(template_max) <= int(max_value)


def _blank_marker_setup_reason(row: Dict[str, str], max_value: int, cache: Dict[str, str]) -> str:
    config_path = row.get("config_path", "")
    fss_path = row.get("fss_path", "")
    if not config_path or not fss_path:
        return ""
    setup_token = Path(fss_path).stem.split("_")[-1]
    if not setup_token:
        return ""
    setup_dir = (Path(config_path) / "DB_echo" / f"setup_{setup_token}").expanduser().resolve()
    cache_key = setup_dir.as_posix()
    if cache_key in cache:
        return cache[cache_key]
    templates = sorted(setup_dir.glob("orientation_*.png"))
    if not templates:
        cache[cache_key] = ""
        return ""
    if all(_is_dark_template(path, max_value=max_value) for path in templates):
        names = ",".join(path.name for path in templates)
        cache[cache_key] = f"{setup_dir.as_posix()} | all {len(templates)} templates max<={max_value}: {names}"
        return cache[cache_key]
    cache[cache_key] = ""
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reviewed LR marker dataset v1.")
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("artifacts/20_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_reviewed_v1"),
    )
    parser.add_argument(
        "--exclude-future-vendors",
        action="store_true",
        default=True,
        help="Exclude folders containing future_vendor_*.json.",
    )
    parser.add_argument(
        "--exclude-vendors",
        nargs="*",
        default=["Biopsee"],
        help="Vendors excluded from the train-ready aggregate.",
    )
    parser.add_argument(
        "--exclude-blank-marker-templates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows whose DB_echo setup has only black/near-black orientation templates.",
    )
    parser.add_argument(
        "--blank-template-max-value",
        type=int,
        default=3,
        help="8-bit max pixel threshold used to consider a template visually black.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_root = args.review_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    all_rows: List[Dict[str, str]] = []
    excluded_rows: List[Dict[str, str]] = []
    fields: List[str] = []
    vendor_summaries: Dict[str, Dict[str, object]] = {}
    blank_setup_cache: Dict[str, str] = {}
    excluded_vendor_norms = {value.strip().lower() for value in args.exclude_vendors if value.strip()}

    for folder in sorted(review_root.glob("lr_marker_sugiu_v2_review_*")):
        manifest = folder / "manifest.csv"
        if not manifest.exists():
            continue
        future_files = list(folder.glob("future_vendor_*.json"))
        manifest_rows, manifest_fields = _read_csv(manifest)
        if not fields:
            fields = list(manifest_fields)
            for extra in [
                "review_batch",
                "review_decision",
                "review_note",
                "review_updated_at_epoch",
                "blank_marker_template_excluded",
                "blank_marker_template_reason",
            ]:
                if extra not in fields:
                    fields.append(extra)

        decisions = _load_decisions(folder / "review_decisions.csv")
        vendor = manifest_rows[0].get("manufacturer", folder.name) if manifest_rows else folder.name
        included = 0
        excluded = 0

        for row in manifest_rows:
            decision = decisions.get(row.get("sample_id", ""), {})
            decision_value = decision.get("decision", "")
            out = dict(row)
            out["review_batch"] = folder.name
            out["review_decision"] = decision_value or "implicit_accepted_after_visual_review"
            out["review_note"] = decision.get("note", "")
            out["review_updated_at_epoch"] = decision.get("updated_at_epoch", "")
            blank_reason = _blank_marker_setup_reason(
                row,
                max_value=int(args.blank_template_max_value),
                cache=blank_setup_cache,
            )
            out["blank_marker_template_excluded"] = "1" if blank_reason else "0"
            out["blank_marker_template_reason"] = blank_reason

            should_exclude = decision_value in EXCLUDE_DECISIONS
            if blank_reason and bool(args.exclude_blank_marker_templates):
                should_exclude = True
                if not decision_value:
                    out["review_decision"] = "blank_marker_template_excluded"
            if vendor.lower() in excluded_vendor_norms:
                should_exclude = True
                if not decision_value:
                    out["review_decision"] = "vendor_excluded"
            if future_files and bool(args.exclude_future_vendors):
                should_exclude = True
                out["review_decision"] = decision_value or "future_vendor_excluded"

            if should_exclude:
                excluded_rows.append(out)
                excluded += 1
            else:
                all_rows.append(out)
                included += 1

        vendor_summaries[vendor] = {
            "review_folder": folder.as_posix(),
            "future_vendor": bool(future_files),
            "vendor_excluded": vendor.lower() in excluded_vendor_norms,
            "rows_total": len(manifest_rows),
            "included": included,
            "excluded": excluded,
            "decision_counts": _count(
                [
                    {
                        "decision": decisions.get(row.get("sample_id", ""), {}).get(
                            "decision",
                            "implicit_accepted_after_visual_review",
                        )
                    }
                    for row in manifest_rows
                ],
                "decision",
            ),
            "sugiu_counts": _count(manifest_rows, "su_giu_pred"),
            "lr_counts": _count(manifest_rows, "lr_label"),
            "blank_marker_template_rows": sum(
                1
                for row in manifest_rows
                if _blank_marker_setup_reason(
                    row,
                    max_value=int(args.blank_template_max_value),
                    cache=blank_setup_cache,
                )
            ),
        }

    if not all_rows:
        raise RuntimeError("No included rows found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "manifest_train_ready.csv", all_rows, fields)
    _write_csv(output_dir / "manifest_excluded.csv", excluded_rows, fields)

    by_vendor_dir = output_dir / "by_vendor"
    by_vendor: Dict[str, List[Dict[str, str]]] = {}
    for row in all_rows:
        by_vendor.setdefault(row.get("manufacturer", "UNKNOWN") or "UNKNOWN", []).append(row)
    for vendor, rows in sorted(by_vendor.items()):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in vendor).strip("_") or "UNKNOWN"
        _write_csv(by_vendor_dir / safe / "train_ready.csv", rows, fields)

    summary = {
        "output_dir": output_dir.as_posix(),
        "included_rows": len(all_rows),
        "excluded_rows": len(excluded_rows),
        "included_by_vendor": _count(all_rows, "manufacturer"),
        "excluded_by_vendor": _count(excluded_rows, "manufacturer"),
        "included_sugiu_counts": _count(all_rows, "su_giu_pred"),
        "included_lr_counts": _count(all_rows, "lr_label"),
        "included_review_decision_counts": _count(all_rows, "review_decision"),
        "excluded_review_decision_counts": _count(excluded_rows, "review_decision"),
        "included_blank_marker_template_counts": _count(all_rows, "blank_marker_template_excluded"),
        "excluded_blank_marker_template_counts": _count(excluded_rows, "blank_marker_template_excluded"),
        "vendors": vendor_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# LR marker SU/GIU reviewed v1",
        "",
        f"Included rows: {len(all_rows)}",
        f"Excluded rows: {len(excluded_rows)}",
        "",
        "Included by vendor:",
    ]
    for vendor, count in summary["included_by_vendor"].items():
        readme.append(f"- {vendor}: {count}")
    readme.extend(["", "Excluded by vendor:"])
    for vendor, count in summary["excluded_by_vendor"].items():
        readme.append(f"- {vendor}: {count}")
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"Output: {output_dir}", flush=True)
    print(f"Included rows: {len(all_rows)}", flush=True)
    print(f"Excluded rows: {len(excluded_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
