#!/usr/bin/env python3
"""Train scale-line image model per vendor from the global scale manifest.

This script:
- loads the global scale manifest
- filters rows by optional exclude labels (fusion / negative / proibite)
- infers vendor from path tokens
- writes one manifest per vendor
- runs train_scale_line_image_model.py once per eligible vendor
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


VENDOR_RULES = [
    ("alpinion", "Alpinion"),
    ("bk", "BK"),
    ("biopsee", "Biopsee"),
    ("canon", "Canon"),
    ("chison", "Chison"),
    ("esaote", "Esaote"),
    ("exactvu", "ExactVu"),
    ("fujifilm", "Fujifilm"),
    ("ge", "GE"),
    ("general electric", "GE"),
    ("hitachi", "Hitachi"),
    ("mindray", "Mindray"),
    ("philips", "Philips"),
    ("samsung", "Samsung"),
    ("sonostar", "Sonostar"),
    ("terason", "Terason"),
    ("vinno", "Vinno"),
    ("koelis", "Koelis"),
    ("siemens", "Siemens"),
]


def infer_vendor(text: str) -> str:
    t = text.lower()
    for key, vendor in VENDOR_RULES:
        if key in t:
            return vendor
    return "Unknown"


def row_matches_exclusion(text: str, label: str) -> bool:
    t = text.lower()
    if label == "fusion":
        return "fusion" in t
    if label == "negative":
        return "negative" in t
    if label == "proibite":
        return any(
            tok in t
            for tok in (
                "proibite",
                "proibita",
                "proibiti",
                "proibito",
                "prohibited",
                "sbagliat",
                "non usare",
                "non_usare",
            )
        )
    return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train per-vendor scale image models.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/23_scale_manifest_ssd_esi1_n1_leakfree/manifest_scale_train_val_test_leakfree.csv"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/32_scale_image_model_per_vendor_20260416"),
    )
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=360)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-review-rows", type=int, default=140)
    p.add_argument("--min-train", type=int, default=20)
    p.add_argument("--min-val", type=int, default=5)
    p.add_argument("--min-test", type=int, default=5)
    p.add_argument("--vendor", type=str, default="", help="Optional comma-separated vendor whitelist.")
    p.add_argument("--exclude-fusion", action="store_true")
    p.add_argument("--exclude-negative", action="store_true")
    p.add_argument("--exclude-proibite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _vendor_allowed(vendor: str, allowed: Optional[Sequence[str]]) -> bool:
    if not allowed:
        return True
    aset = {x.strip().lower() for x in allowed if x.strip()}
    return vendor.lower() in aset


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    out_root = args.output_root.expanduser().resolve()
    manifests_dir = out_root / "manifests"
    runs_dir = out_root / "runs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    exclude_labels: List[str] = []
    if args.exclude_fusion:
        exclude_labels.append("fusion")
    if args.exclude_negative:
        exclude_labels.append("negative")
    if args.exclude_proibite:
        exclude_labels.append("proibite")

    allowed_vendors = [x.strip() for x in args.vendor.split(",") if x.strip()] if args.vendor else []

    rows_by_vendor: Dict[str, List[dict]] = defaultdict(list)
    split_counts_by_vendor: Dict[str, Counter[str]] = defaultdict(Counter)
    excluded_by_filter: Counter[str] = Counter()
    rows_in = 0
    rows_excluded = 0

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        fieldnames = list(rd.fieldnames or [])
        if not fieldnames:
            raise RuntimeError("Empty manifest or missing header.")
        for row in rd:
            rows_in += 1
            text = " ".join(
                (
                    row.get("fss_path", ""),
                    row.get("setup_id", ""),
                    row.get("sample_id", ""),
                )
            )
            matched = [label for label in exclude_labels if row_matches_exclusion(text, label)]
            if matched:
                rows_excluded += 1
                for label in matched:
                    excluded_by_filter[label] += 1
                continue

            vendor = infer_vendor(text)
            if not _vendor_allowed(vendor, allowed_vendors):
                continue
            rows_by_vendor[vendor].append(row)
            split = (row.get("split") or "").strip().lower()
            if split in {"train", "val", "test"}:
                split_counts_by_vendor[vendor][split] += 1

    manifest_paths: Dict[str, str] = {}
    for vendor, rows in sorted(rows_by_vendor.items()):
        out_csv = manifests_dir / f"manifest_scale_{vendor.lower()}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)
        manifest_paths[vendor] = out_csv.as_posix()

    run_records = []
    for vendor in sorted(rows_by_vendor.keys()):
        counts = split_counts_by_vendor[vendor]
        eligible = (
            counts.get("train", 0) >= args.min_train
            and counts.get("val", 0) >= args.min_val
            and counts.get("test", 0) >= args.min_test
        )

        out_dir = runs_dir / vendor.lower()
        rec = {
            "vendor": vendor,
            "manifest": manifest_paths[vendor],
            "rows_total": int(sum(counts.values())),
            "rows_by_split": {
                "train": int(counts.get("train", 0)),
                "val": int(counts.get("val", 0)),
                "test": int(counts.get("test", 0)),
            },
            "eligible": bool(eligible),
            "output_dir": out_dir.as_posix(),
            "status": "skipped",
            "returncode": None,
            "elapsed_sec": 0.0,
        }
        if not eligible:
            rec["status"] = "skipped_not_enough_split_rows"
            run_records.append(rec)
            continue
        if args.dry_run:
            rec["status"] = "dry_run"
            run_records.append(rec)
            continue

        cmd = [
            sys.executable,
            "tools/fss/train_scale_line_image_model.py",
            "--manifest",
            manifest_paths[vendor],
            "--output-dir",
            out_dir.as_posix(),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--image-width",
            str(args.image_width),
            "--image-height",
            str(args.image_height),
            "--seed",
            str(args.seed),
            "--max-review-rows",
            str(args.max_review_rows),
        ]

        print(
            f"[run] vendor={vendor} rows={rec['rows_total']} "
            f"(train={rec['rows_by_split']['train']}, val={rec['rows_by_split']['val']}, test={rec['rows_by_split']['test']})",
            flush=True,
        )
        t0 = time.time()
        proc = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        rec["elapsed_sec"] = float(elapsed)
        rec["returncode"] = int(proc.returncode)
        rec["status"] = "ok" if proc.returncode == 0 else "failed"
        run_records.append(rec)

    summary = {
        "source_manifest": manifest.as_posix(),
        "output_root": out_root.as_posix(),
        "exclude_filters_enabled": exclude_labels,
        "rows_in": int(rows_in),
        "rows_excluded": int(rows_excluded),
        "rows_excluded_by_filter": dict(excluded_by_filter),
        "vendors_total": len(rows_by_vendor),
        "vendors_eligible": int(sum(1 for r in run_records if r.get("eligible"))),
        "vendors_ok": int(sum(1 for r in run_records if r.get("status") == "ok")),
        "vendors_failed": int(sum(1 for r in run_records if r.get("status") == "failed")),
        "vendors_skipped": int(sum(1 for r in run_records if str(r.get("status", "")).startswith("skipped"))),
        "run_records": run_records,
    }
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
