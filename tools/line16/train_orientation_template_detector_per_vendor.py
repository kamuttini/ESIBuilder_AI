#!/usr/bin/env python3
"""Train orientation template detector per vendor and build routing map.

Workflow:
1) read global manifest and discover vendor counts by split
2) train per-vendor checkpoints via a user-provided trainer script
3) compare vendor test IoU vs global reference IoU per vendor
4) export vendor summary CSV + selected vendor->checkpoint JSON map
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VendorStats:
    vendor: str
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


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


def _load_manifest_vendor_stats(manifest_path: Path) -> Dict[str, VendorStats]:
    counters: Dict[str, Dict[str, int]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vendor = (row.get("manufacturer") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not vendor or split not in SPLITS:
                continue
            bucket = counters.setdefault(vendor, {"train": 0, "val": 0, "test": 0})
            bucket[split] += 1

    out: Dict[str, VendorStats] = {}
    for vendor, counts in sorted(counters.items()):
        out[vendor] = VendorStats(
            vendor=vendor,
            train=int(counts["train"]),
            val=int(counts["val"]),
            test=int(counts["test"]),
        )
    return out


def _load_global_per_vendor_iou(global_metrics_path: Optional[Path]) -> Dict[str, float]:
    if global_metrics_path is None:
        return {}
    if not global_metrics_path.exists():
        print(
            f"Warning: global metrics not found ({global_metrics_path}). "
            "Per-vendor delta vs global will be unavailable.",
            flush=True,
        )
        return {}

    data = json.loads(global_metrics_path.read_text(encoding="utf-8"))
    test = data.get("test", {}) if isinstance(data.get("test"), dict) else {}
    per = test.get("per_manufacturer_best_iou", {})
    if not isinstance(per, dict):
        per = data.get("per_manufacturer_best_iou", {})
    if not isinstance(per, dict):
        return {}

    out: Dict[str, float] = {}
    for k, v in per.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _load_vendor_test_iou(metrics_path: Path) -> float:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = data.get("test", {})
    value = test.get("mean_best_iou", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_optional_iou(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _train_vendor(
    python_bin: str,
    trainer_path: Path,
    manifest_path: Path,
    output_dir: Path,
    vendor: str,
    args: argparse.Namespace,
) -> None:
    cmd: List[str] = [
        python_bin,
        trainer_path.as_posix(),
        "--manifest",
        manifest_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--manufacturer",
        vendor,
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

    subprocess.run(cmd, check=True)


def _check_python_bin_has_torch(python_bin: str) -> None:
    probe = subprocess.run(
        [python_bin, "-c", "import torch"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    stderr = (probe.stderr or "").strip()
    raise RuntimeError(
        "Selected python interpreter cannot import torch. "
        "Use --python-bin pointing to the environment used for training "
        f"(failed interpreter: {python_bin}). "
        + (f"Details: {stderr}" if stderr else "")
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train orientation template detector per vendor.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_vendor"),
    )
    p.add_argument(
        "--trainer-script",
        type=Path,
        default=None,
        help=(
            "Path to single-model trainer script. "
            "Required: the previous default trainer was removed and must be redesigned."
        ),
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the trainer script.",
    )
    p.add_argument(
        "--global-metrics",
        type=Path,
        default=Path("artifacts/30_models/orientation_rect_training_v1/test_metrics_recomputed.json"),
        help="Global-model metrics.json used as baseline per vendor.",
    )
    p.add_argument(
        "--vendor",
        action="append",
        default=[],
        help="Restrict to selected vendor(s). Repeat or pass comma-separated values.",
    )
    p.add_argument("--min-train-samples", type=int, default=40)
    p.add_argument("--min-val-samples", type=int, default=8)
    p.add_argument("--min-test-samples", type=int, default=8)
    p.add_argument("--min-delta-iou", type=float, default=0.01)
    p.add_argument(
        "--min-test-samples-for-selection",
        type=int,
        default=16,
        help="Require at least this many test samples to include vendor in routing map.",
    )
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--strict-manifest", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument(
        "--allow-selection-without-global-baseline",
        action="store_true",
        help=(
            "Allow selecting vendor models even when the vendor is missing from global baseline metrics. "
            "Default behavior is conservative: do not select in that case."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned trainings only.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.trainer_script is None:
        raise RuntimeError(
            "--trainer-script is required. "
            "train_orientation_template_detector.py was removed and must be replaced with the new implementation."
        )
    trainer_path = args.trainer_script.expanduser().resolve()
    global_metrics_path = args.global_metrics.expanduser().resolve() if args.global_metrics else None

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vendor_stats = _load_manifest_vendor_stats(manifest_path)
    selected_vendors = _parse_list(args.vendor)

    if selected_vendors:
        lookup = {k.lower(): k for k in vendor_stats}
        filtered: Dict[str, VendorStats] = {}
        missing: List[str] = []
        for req in selected_vendors:
            key = req.lower()
            actual = lookup.get(key)
            if not actual:
                missing.append(req)
                continue
            filtered[actual] = vendor_stats[actual]
        vendor_stats = dict(sorted(filtered.items()))
        if missing:
            print(f"Warning: requested vendor(s) not found in manifest: {missing}", flush=True)

    if not vendor_stats:
        raise RuntimeError("No vendors available after filtering.")

    global_iou = _load_global_per_vendor_iou(global_metrics_path)
    if not args.dry_run:
        _check_python_bin_has_torch(str(args.python_bin))

    summary_rows: List[Dict[str, object]] = []
    selected_map: Dict[str, str] = {}

    for vendor, stats in sorted(vendor_stats.items()):
        enough_samples = (
            stats.train >= args.min_train_samples
            and stats.val >= args.min_val_samples
            and stats.test >= args.min_test_samples
        )

        vendor_output_dir = output_dir / vendor
        metrics_path = vendor_output_dir / "metrics.json"
        best_model_path = vendor_output_dir / "best_model.pt"

        if not enough_samples:
            summary_rows.append(
                {
                    "vendor": vendor,
                    "train_samples": stats.train,
                    "val_samples": stats.val,
                    "test_samples": stats.test,
                    "global_test_best_iou": _fmt_optional_iou(global_iou.get(vendor)),
                    "vendor_test_best_iou": "",
                    "delta_vendor_minus_global": "",
                    "use_vendor_model": 0,
                    "reason": "insufficient_samples",
                }
            )
            continue

        print(
            f"[{vendor}] train/val/test={stats.train}/{stats.val}/{stats.test} -> {vendor_output_dir}",
            flush=True,
        )
        if args.dry_run:
            vendor_test_iou = float("nan")
            global_vendor_iou = global_iou.get(vendor)
            delta: Optional[float] = None
            reason = "dry_run"
            use_vendor_model = 0
        else:
            _train_vendor(
                python_bin=str(args.python_bin),
                trainer_path=trainer_path,
                manifest_path=manifest_path,
                output_dir=vendor_output_dir,
                vendor=vendor,
                args=args,
            )
            if not metrics_path.exists():
                raise RuntimeError(f"Training completed but metrics missing: {metrics_path}")
            vendor_test_iou = _load_vendor_test_iou(metrics_path)
            global_vendor_iou = global_iou.get(vendor)
            min_test_support = int(args.min_test_samples_for_selection)
            support_ok = stats.test >= min_test_support

            if global_vendor_iou is None:
                if args.allow_selection_without_global_baseline:
                    delta = vendor_test_iou
                    use_vendor_model = int(
                        delta >= float(args.min_delta_iou)
                        and support_ok
                        and best_model_path.exists()
                    )
                    reason = "selected_no_global_baseline" if use_vendor_model else "not_selected_no_global_baseline"
                else:
                    delta = None
                    use_vendor_model = 0
                    reason = "no_global_baseline"
            else:
                delta = vendor_test_iou - float(global_vendor_iou)
                use_vendor_model = int(
                    delta >= float(args.min_delta_iou)
                    and support_ok
                    and best_model_path.exists()
                )
                reason = "selected" if use_vendor_model else "not_better_than_global"

            if not support_ok and use_vendor_model == 0:
                reason = "low_test_support"
            if use_vendor_model:
                selected_map[vendor] = best_model_path.resolve().as_posix()

        if args.dry_run or delta is None:
            delta_value = ""
        else:
            delta_value = f"{float(delta):.6f}"

        summary_rows.append(
            {
                "vendor": vendor,
                "train_samples": stats.train,
                "val_samples": stats.val,
                "test_samples": stats.test,
                "global_test_best_iou": _fmt_optional_iou(global_vendor_iou),
                "vendor_test_best_iou": "" if args.dry_run else f"{vendor_test_iou:.6f}",
                "delta_vendor_minus_global": delta_value,
                "use_vendor_model": use_vendor_model,
                "reason": reason,
            }
        )

    summary_csv = output_dir / "vendor_vs_global_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "vendor",
            "train_samples",
            "val_samples",
            "test_samples",
            "global_test_best_iou",
            "vendor_test_best_iou",
            "delta_vendor_minus_global",
            "use_vendor_model",
            "reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    selected_map_path = output_dir / "vendor_rect_map_selected.json"
    selected_map_path.write_text(json.dumps(selected_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_json = output_dir / "vendor_training_summary.json"
    summary_payload = {
        "manifest": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "global_metrics": global_metrics_path.as_posix() if global_metrics_path else None,
        "selection_rules": {
            "min_train_samples": int(args.min_train_samples),
            "min_val_samples": int(args.min_val_samples),
            "min_test_samples": int(args.min_test_samples),
            "min_delta_iou": float(args.min_delta_iou),
            "min_test_samples_for_selection": int(args.min_test_samples_for_selection),
        },
        "vendors_considered": len(vendor_stats),
        "vendors_selected": len(selected_map),
        "selected_vendor_models": selected_map,
        "summary_csv": summary_csv.as_posix(),
        "vendor_rect_map_selected": selected_map_path.as_posix(),
        "rows": summary_rows,
        "dry_run": bool(args.dry_run),
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Selected map: {selected_map_path}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Vendors selected: {len(selected_map)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
