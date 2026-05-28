#!/usr/bin/env python3
"""Train vendor-specific rect models and export a vendor->checkpoint map."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SPLITS = ("train", "val", "test")


def _parse_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        for token in str(raw).replace(";", ",").split(","):
            value = token.strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
    return out


def _slugify_vendor(value: str) -> str:
    out: List[str] = []
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "vendor"


def _load_manifest_rows(manifest_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest without header: {manifest_path}")
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fields, rows


def _group_by_vendor(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        vendor = (row.get("manufacturer") or "").strip()
        if not vendor:
            continue
        grouped[vendor].append(row)
    return dict(sorted(grouped.items()))


def _split_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split = (row.get("split") or "").strip().lower()
        if split in counts:
            counts[split] += 1
    return counts


def _write_manifest(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_train_cmd(
    *,
    python_bin: str,
    trainer_script: Path,
    manifest_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd: List[str] = [
        python_bin,
        trainer_script.as_posix(),
        "--manifest",
        manifest_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
    ]
    if args.max_train_samples > 0:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples > 0:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_test_samples > 0:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.pretrained:
        cmd.append("--pretrained")
    if args.strict_manifest:
        cmd.append("--strict-manifest")
    if args.log_interval > 0:
        cmd.extend(["--log-interval", str(args.log_interval)])
    return cmd


def _load_metric(path: Path, key: str, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    value: object = data
    for token in key.split("."):
        if not isinstance(value, dict):
            value = ""
            break
        value = value.get(token, "")
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train rect models per vendor.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/40_outputs_eval/vendor_template_retraining_from_review_final_20260318/manifest_training_ready.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/rect_training_vendor"),
    )
    p.add_argument(
        "--trainer-script",
        type=Path,
        default=Path("tools/ultrasound/train_ultrasound_rect_net.py"),
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument(
        "--vendor",
        action="append",
        default=[],
        help="Restrict to selected vendors (repeat or comma-separated).",
    )
    p.add_argument("--min-train-samples", type=int, default=1)
    p.add_argument("--min-val-samples", type=int, default=1)
    p.add_argument("--min-test-samples", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--strict-manifest", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    trainer_script = args.trainer_script.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    fields, rows = _load_manifest_rows(manifest_path)
    grouped = _group_by_vendor(rows)
    requested = _parse_list(args.vendor)

    if requested:
        lookup = {name.lower(): name for name in grouped}
        filtered: Dict[str, List[Dict[str, str]]] = {}
        missing: List[str] = []
        for req in requested:
            key = req.lower()
            actual = lookup.get(key)
            if actual is None:
                missing.append(req)
                continue
            filtered[actual] = grouped[actual]
        grouped = dict(sorted(filtered.items()))
        if missing:
            print(f"Warning: vendors not found in manifest: {missing}", flush=True)

    if not grouped:
        raise RuntimeError("No vendors available to train.")

    manifests_dir = output_dir / "vendor_manifests"
    models_dir = output_dir / "vendor_models"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []
    vendor_map: Dict[str, str] = {}

    for vendor, vendor_rows in grouped.items():
        counts = _split_counts(vendor_rows)
        total = int(sum(counts.values()))
        slug = _slugify_vendor(vendor)
        vendor_manifest = manifests_dir / f"{slug}.csv"
        vendor_output = models_dir / slug

        row_base = {
            "vendor": vendor,
            "slug": slug,
            "rows_total": str(total),
            "train_rows": str(counts["train"]),
            "val_rows": str(counts["val"]),
            "test_rows": str(counts["test"]),
            "status": "",
            "reason": "",
            "best_model": "",
            "best_epoch": "",
            "best_val_iou": "",
            "test_iou": "",
        }

        if (
            counts["train"] < args.min_train_samples
            or counts["val"] < args.min_val_samples
            or counts["test"] < args.min_test_samples
        ):
            row = dict(row_base)
            row["status"] = "skipped"
            row["reason"] = "insufficient_split_support"
            summary_rows.append(row)
            print(
                f"[{vendor}] skipped train/val/test={counts['train']}/{counts['val']}/{counts['test']}",
                flush=True,
            )
            continue

        _write_manifest(vendor_manifest, fields, vendor_rows)
        cmd = _build_train_cmd(
            python_bin=args.python_bin,
            trainer_script=trainer_script,
            manifest_path=vendor_manifest,
            output_dir=vendor_output,
            args=args,
        )
        print(
            f"[{vendor}] start train/val/test={counts['train']}/{counts['val']}/{counts['test']} -> {vendor_output}",
            flush=True,
        )
        if args.dry_run:
            row = dict(row_base)
            row["status"] = "planned"
            row["reason"] = "dry_run"
            summary_rows.append(row)
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            row = dict(row_base)
            row["status"] = "failed"
            row["reason"] = f"trainer_exit_{exc.returncode}"
            summary_rows.append(row)
            print(f"[{vendor}] failed exit={exc.returncode}", flush=True)
            continue

        best_model = (vendor_output / "best_model.pt").resolve()
        metrics_path = vendor_output / "metrics.json"

        row = dict(row_base)
        if best_model.exists():
            row["best_model"] = best_model.as_posix()
            vendor_map[vendor] = best_model.as_posix()
        row["best_epoch"] = _load_metric(metrics_path, "best_epoch")
        row["best_val_iou"] = _load_metric(metrics_path, "best_val_iou")
        row["test_iou"] = _load_metric(metrics_path, "test.mean_iou")
        row["status"] = "trained" if best_model.exists() else "failed"
        row["reason"] = "ok" if best_model.exists() else "missing_checkpoint"
        summary_rows.append(row)
        print(
            f"[{vendor}] done best_epoch={row['best_epoch']} best_val_iou={row['best_val_iou']}",
            flush=True,
        )

    summary_csv = output_dir / "vendor_training_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        fields_out = [
            "vendor",
            "slug",
            "rows_total",
            "train_rows",
            "val_rows",
            "test_rows",
            "status",
            "reason",
            "best_model",
            "best_epoch",
            "best_val_iou",
            "test_iou",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields_out)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    vendor_map_path = output_dir / "vendor_rect_map_all.json"
    vendor_map_path.write_text(
        json.dumps(vendor_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_json = output_dir / "vendor_training_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "manifest": manifest_path.as_posix(),
                "trainer_script": trainer_script.as_posix(),
                "output_dir": output_dir.as_posix(),
                "vendors_considered": len(grouped),
                "vendors_trained": sum(1 for r in summary_rows if r["status"] == "trained"),
                "vendor_map_path": vendor_map_path.as_posix(),
                "summary_csv": summary_csv.as_posix(),
                "rows": summary_rows,
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Vendor map: {vendor_map_path}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
