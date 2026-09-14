#!/usr/bin/env python3
"""Build the needle-vs-nonneedle manifest from raw acquisition folders.

Labels are weak/folder-derived, exactly as the acquisitions are already
organised on the SSD: any image whose relative path contains a directory
matching --positive-regex (default "agh", i.e. AGHI / Verifica Aghi / aghi ...)
is a positive; every other image is a negative.

Two properties matter for this dataset and are enforced here:

1. Leak-free split at acquisition level. The top-level acquisition directory is
   the group; every image of a group lands in a single split. Consecutive video
   frames are near duplicates, so any finer split would leak.
2. Redundancy control. A leaf directory often holds hundreds of nearly identical
   frames. Images are subsampled with a uniform stride over the sorted file list
   so the kept subset spans the whole acquisition instead of one burst.

Negatives are capped per leaf directory, with a larger budget inside the groups
that also contain positives: those are the hard, in-domain negatives (same
machine, same session, often also water/phantom).

Example:
  python3 tools/needle/prepare_needle_dataset.py \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
    --output-dir artifacts/90_needle_dataset/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}

MANIFEST_FIELDS = [
    "image_path", "rel_path", "group", "leaf_dir", "label", "split", "vendor_hint",
]

# Coarse vendor guess from the acquisition folder name. Used only to stratify the
# split and to report metrics per vendor -- never as a model input.
VENDOR_PATTERNS: Sequence[Tuple[str, str]] = (
    ("Esaote", r"esaote|mylab|echolaser|xpro"),
    ("GE", r"\bge\b|logiq|voluson"),
    ("BK", r"\bbk\b|flexfocus|profocus"),
    ("Philips", r"philips|affiniti|epiq|cx50"),
    ("Hitachi", r"hitachi|arietta|aloka"),
    ("Canon", r"canon|toshiba|aplio"),
    ("Siemens", r"siemens|acuson"),
    ("Mindray", r"mindray|resona"),
    ("Terason", r"terason|usmart"),
    ("Samsung", r"samsung|\brs8\b"),
    ("Supersonic", r"supersonic|aixplorer"),
    ("Alpinion", r"alpinion|ecube|xcube"),
    ("Vinno", r"vinno"),
    ("Sonostar", r"sonostar|uprobe"),
    ("Fukuda", r"fukuda"),
    ("Chison", r"chison"),
    ("Sonoscape", r"sonoscape"),
    ("Butterfly", r"butterfly"),
)


def guess_vendor(group: str) -> str:
    low = group.lower()
    for name, pattern in VENDOR_PATTERNS:
        if re.search(pattern, low):
            return name
    return "UNKNOWN"


def stride_sample(items: Sequence[Path], cap: int) -> List[Path]:
    """Keep at most `cap` items, uniformly spread over the sorted sequence."""
    total = len(items)
    if cap <= 0 or total <= cap:
        return list(items)
    step = total / float(cap)
    picked = []
    seen = set()
    for i in range(cap):
        idx = min(total - 1, int(round(i * step)))
        if idx in seen:
            idx = min(total - 1, idx + 1)
            while idx in seen and idx + 1 < total:
                idx += 1
        if idx in seen:
            continue
        seen.add(idx)
        picked.append(items[idx])
    return picked


def scan_dataset(
    root: Path, positive_regex: re.Pattern, exclude_regex: Optional[re.Pattern] = None
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, Dict[str, List[Path]]]]:
    """Return (positives, negatives) as {group: {leaf_dir_rel: [paths]}}."""
    positives: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    negatives: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        parts = rel.split(os.sep)
        group = parts[0]
        images = sorted(
            Path(dirpath) / f
            for f in filenames
            if Path(f).suffix.lower() in IMAGE_EXTS and not f.startswith("._")
        )
        if not images:
            continue
        if exclude_regex is not None and any(exclude_regex.search(p) for p in parts):
            continue
        target = positives if any(positive_regex.search(p) for p in parts) else negatives
        target[group][rel] = images

    return positives, negatives


def assign_splits(
    weights: Dict[str, int], fractions: Dict[str, float], rng: random.Random
) -> Dict[str, str]:
    """Greedily assign groups to splits so each split hits its target weight share.

    Heaviest groups first: a single acquisition can hold thousands of frames, so
    random assignment would routinely blow up one split.
    """
    total = sum(weights.values()) or 1
    targets = {name: frac * total for name, frac in fractions.items()}
    current = {name: 0.0 for name in fractions}
    assignment: Dict[str, str] = {}

    ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    for group, weight in ordered:
        # pick the split that is furthest below its target, in absolute terms
        best = max(fractions, key=lambda name: (targets[name] - current[name], rng.random()))
        assignment[group] = best
        current[best] += weight
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-regex", type=str, default="agh")
    parser.add_argument("--exclude-regex", type=str, default="",
                        help="Drop images whose path contains a directory matching this regex. "
                             "Use \"guid\" to leave out the needle-GUIDE folders: those carry the "
                             "guide overlay without any needle and without water, a different "
                             "concept from 'needles in water'.")
    parser.add_argument("--pos-cap-per-dir", type=int, default=60)
    parser.add_argument("--neg-cap-per-dir-pos-group", type=int, default=25)
    parser.add_argument("--neg-cap-per-dir-other", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset-root non trovata: {root}")

    positive_regex = re.compile(args.positive_regex, re.IGNORECASE)
    exclude_regex = re.compile(args.exclude_regex, re.IGNORECASE) if args.exclude_regex else None
    positives, negatives = scan_dataset(root, positive_regex, exclude_regex)

    pos_groups = set(positives)
    all_groups = pos_groups | set(negatives)

    # Apply the per-directory caps FIRST, then balance the splits on the counts
    # that will actually be written out. Balancing on raw counts would be wrong:
    # the caps compress the huge acquisitions much more than the small ones.
    kept_pos: Dict[str, List[Tuple[str, List[Path]]]] = defaultdict(list)
    kept_neg: Dict[str, List[Tuple[str, List[Path]]]] = defaultdict(list)

    for group, leaves in positives.items():
        for leaf, images in sorted(leaves.items()):
            kept_pos[group].append((leaf, stride_sample(images, args.pos_cap_per_dir)))

    for group, leaves in negatives.items():
        cap = (
            args.neg_cap_per_dir_pos_group
            if group in pos_groups
            else args.neg_cap_per_dir_other
        )
        for leaf, images in sorted(leaves.items()):
            kept_neg[group].append((leaf, stride_sample(images, cap)))

    test_frac = max(0.0, 1.0 - args.train_frac - args.val_frac)
    fractions = {"train": args.train_frac, "val": args.val_frac, "test": test_frac}

    # Groups carrying positives drive the split balance: they are the scarce
    # resource. Negative-only groups are balanced separately on their own weight.
    pos_weights = {g: sum(len(p) for _, p in kept_pos[g]) for g in pos_groups}
    split_of = assign_splits(pos_weights, fractions, rng)

    neg_only = {
        g: sum(len(p) for _, p in kept_neg[g])
        for g in all_groups
        if g not in pos_groups
    }
    split_of.update(assign_splits(neg_only, fractions, rng))

    rows: List[Dict[str, object]] = []

    def emit(paths: Sequence[Path], group: str, leaf: str, label: int) -> None:
        for path in paths:
            rows.append(
                {
                    "image_path": str(path),
                    "rel_path": str(path.relative_to(root)),
                    "group": group,
                    "leaf_dir": leaf,
                    "label": label,
                    "split": split_of[group],
                    "vendor_hint": guess_vendor(group),
                }
            )

    for group, leaves in kept_pos.items():
        for leaf, images in leaves:
            emit(images, group, leaf, 1)
    for group, leaves in kept_neg.items():
        for leaf, images in leaves:
            emit(images, group, leaf, 0)

    rows.sort(key=lambda r: (str(r["split"]), str(r["group"]), str(r["rel_path"])))

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_needle.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    per_split = defaultdict(Counter)
    per_split_groups = defaultdict(set)
    per_vendor = defaultdict(Counter)
    for row in rows:
        per_split[row["split"]][int(row["label"])] += 1
        per_split_groups[row["split"]].add(row["group"])
        per_vendor[row["vendor_hint"]][int(row["label"])] += 1

    summary = {
        "dataset_root": str(root),
        "positive_regex": args.positive_regex,
        "exclude_regex": args.exclude_regex,
        "caps": {
            "pos_per_dir": args.pos_cap_per_dir,
            "neg_per_dir_pos_group": args.neg_cap_per_dir_pos_group,
            "neg_per_dir_other": args.neg_cap_per_dir_other,
        },
        "groups_total": len(all_groups),
        "groups_with_positives": len(pos_groups),
        "images_total": len(rows),
        "splits": {
            split: {
                "pos": counts[1],
                "neg": counts[0],
                "groups": len(per_split_groups[split]),
            }
            for split, counts in sorted(per_split.items())
        },
        "vendors": {
            vendor: {"pos": counts[1], "neg": counts[0]}
            for vendor, counts in sorted(per_vendor.items())
        },
        "split_of_group": split_of,
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"manifest: {manifest_path}  ({len(rows)} immagini)")
    for split, counts in sorted(per_split.items()):
        print(
            f"  {split:5s}  pos={counts[1]:6d}  neg={counts[0]:6d}  "
            f"gruppi={len(per_split_groups[split]):4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
