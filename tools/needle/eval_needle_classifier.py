#!/usr/bin/env python3
"""Evaluate a saved needle classifier on the cached crops, without retraining.

Useful to score the best checkpoint of a run that was stopped early, or to
re-score an existing model after the crops or the split have been rebuilt.
Writes the same `metrics.json` and `predictions_<split>.csv` the trainer writes,
so the review gallery works on the output unchanged.

Example:
  python3 tools/needle/eval_needle_classifier.py \
    --model artifacts/91_needle_models/resnet18_256_cpu/best_model.pt \
    --crops-dir artifacts/90_needle_dataset/v1/crops \
    --output-dir artifacts/91_needle_models/resnet18_256_cpu_eval --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_needle_classifier import (
    CropDataset,
    average_precision,
    best_f1_threshold,
    binary_metrics,
    build_model,
    choose_device,
    grouped_metrics,
    load_rows,
    predict,
    review_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--sensitivity-exclude-regex", type=str, default="guid",
                        help="Also report metrics with these leaf directories removed, as a "
                             "sensitivity check on the folder-derived labels. Default \"guid\": "
                             "the needle-GUIDE folders hold a guide overlay and no needle.")
    args = parser.parse_args()

    device = choose_device(args.device or None)
    checkpoint = torch.load(args.model.expanduser().resolve(), map_location="cpu", weights_only=False)
    image_size = int(checkpoint["image_size"])
    model = build_model(checkpoint["arch"], dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    rows = load_rows(args.crops_dir.expanduser().resolve())
    by_split: Dict[str, List] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_of: Dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        if not by_split[split]:
            continue
        loader = DataLoader(
            CropDataset(by_split[split], image_size, train=False, jitter=0.0),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        scores_of[split] = predict(model, loader, device, tta=not args.no_tta)
        print(f"{split}: {len(by_split[split])} immagini valutate", flush=True)

    val_labels = np.array([r.label for r in by_split["val"]])
    threshold, _ = best_f1_threshold(val_labels, scores_of["val"])
    policy = review_policy(val_labels, scores_of["val"], args.target_precision, args.target_recall)

    report: Dict[str, object] = {
        "source_model": str(args.model),
        "arch": checkpoint["arch"],
        "image_size": image_size,
        "source_epoch": checkpoint.get("epoch"),
        "best_val_ap": checkpoint.get("val_ap"),
        "threshold_from_val": threshold,
        "review_policy_from_val": policy,
        "splits": {},
    }

    for split, scores in scores_of.items():
        split_rows = by_split[split]
        labels = np.array([r.label for r in split_rows])
        per_vendor: Dict[str, Dict[str, float]] = {}
        for vendor in sorted({r.vendor for r in split_rows}):
            idx = np.array([i for i, r in enumerate(split_rows) if r.vendor == vendor])
            if idx.size >= 10:
                per_vendor[vendor] = binary_metrics(labels[idx], scores[idx], threshold)
        report["splits"][split] = {
            "per_image": binary_metrics(labels, scores, threshold),
            "per_leaf_dir": grouped_metrics(
                split_rows, scores, threshold, key=lambda r: f"{r.group}||{r.leaf_dir}"
            ),
            "per_vendor": per_vendor,
        }
        with (out_dir / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rel_path", "group", "leaf_dir", "vendor", "label", "score"])
            for row, score in zip(split_rows, scores):
                writer.writerow([row.rel_path, row.group, row.leaf_dir, row.vendor, row.label, f"{score:.6f}"])

    if args.sensitivity_exclude_regex:
        import re as _re

        pattern = _re.compile(args.sensitivity_exclude_regex, _re.IGNORECASE)
        sensitivity: Dict[str, object] = {}
        for split, scores in scores_of.items():
            keep = [i for i, r in enumerate(by_split[split]) if not pattern.search(r.leaf_dir)]
            if len(keep) == len(by_split[split]):
                continue
            idx = np.array(keep)
            labels = np.array([by_split[split][i].label for i in keep])
            sensitivity[split] = {
                "removed_images": len(by_split[split]) - len(keep),
                "per_image": binary_metrics(labels, scores[idx], threshold),
            }
        if sensitivity:
            report["sensitivity_excluding"] = {
                "regex": args.sensitivity_exclude_regex,
                "splits": sensitivity,
            }

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nsoglia da val {threshold:.3f}  (policy: {policy})")
    for split in ("val", "test"):
        if split not in report["splits"]:
            continue
        pi = report["splits"][split]["per_image"]
        pl = report["splits"][split]["per_leaf_dir"]
        print(
            f"{split:5s} immagine: AP {pi['ap']:.4f} AUC {pi['auc']:.4f} "
            f"P {pi['precision']:.3f} R {pi['recall']:.3f} F1 {pi['f1']:.3f} acc {pi['accuracy']:.4f}"
        )
        print(
            f"      cartella: AP {pl['ap']:.4f} AUC {pl['auc']:.4f} "
            f"P {pl['precision']:.3f} R {pl['recall']:.3f} F1 {pl['f1']:.3f} (n={pl['n']})"
        )
    print(f"report: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
