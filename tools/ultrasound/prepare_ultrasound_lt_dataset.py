#!/usr/bin/env python3
"""Build a binary L/T manifest from raw ultrasound acquisition trees."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {
    ".spotlight-v100",
    "system volume information",
    "$recycle.bin",
    ".trashes",
    ".fseventsd",
}

LABEL_L = "L"
LABEL_T = "T"
LABEL_AMBIGUOUS = "AMB"
CLASS_TARGET = {LABEL_L: 0, LABEL_T: 1}

TOKENS_L = {
    "L",
    "LINEARE",
    "LINEAR",
    "LINEARI",
}
TOKENS_T = {
    "T",
    "TRASVERSALE",
    "TRASVERSALI",
    "TRANSVERSE",
    "TRANSVERSAL",
    "TRANSVERSALE",
    "TRANS",
    "TRASV",
}

VENDOR_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("BK", re.compile(r"(^|[^a-z0-9])(bk|biojet|flexfocus|profocus|specto)([^a-z0-9]|$)")),
    ("Esaote", re.compile(r"(^|[^a-z0-9])(esaote|mylab|echolaser)([^a-z0-9]|$)")),
    ("Hitachi", re.compile(r"(^|[^a-z0-9])(hitachi|arietta|aloka|fujifilm)([^a-z0-9]|$)")),
    ("GE", re.compile(r"(^|[^a-z0-9])(ge|logiq|voluson|versana)([^a-z0-9]|$)")),
    ("Mindray", re.compile(r"(^|[^a-z0-9])(mindray|resona|consona|mx7|te7|dc70)([^a-z0-9]|$)")),
    ("Canon", re.compile(r"(^|[^a-z0-9])(canon|aplio)([^a-z0-9]|$)")),
    ("Philips", re.compile(r"(^|[^a-z0-9])(philips|affiniti|affinity|cx50)([^a-z0-9]|$)")),
    ("Toshiba", re.compile(r"(^|[^a-z0-9])toshiba([^a-z0-9]|$)")),
    ("Siemens", re.compile(r"(^|[^a-z0-9])(siemens|acuson)([^a-z0-9]|$)")),
    ("Koelis", re.compile(r"(^|[^a-z0-9])koelis([^a-z0-9]|$)")),
    ("Biopsee", re.compile(r"(^|[^a-z0-9])biopsee([^a-z0-9]|$)")),
    ("Terason", re.compile(r"(^|[^a-z0-9])(terason|usmart|smart[ _-]*3200t)([^a-z0-9]|$)")),
    ("Sonostar", re.compile(r"(^|[^a-z0-9])(sonostar|uprobe)([^a-z0-9]|$)")),
    ("ExactVu", re.compile(r"(^|[^a-z0-9])(exactvu|exact[ _-]*vu|edap)([^a-z0-9]|$)")),
    ("Alpinion", re.compile(r"(^|[^a-z0-9])(alpinion|e-?cube|xcube)([^a-z0-9]|$)")),
    ("Vinno", re.compile(r"(^|[^a-z0-9])vinno([^a-z0-9]|$)")),
)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    source_rel_path: Path
    acquisition_root: str
    manufacturer: str
    label: str
    group_id: str
    label_anchor_rel: str
    label_source: str


@dataclass
class GroupRecord:
    group_id: str
    label: str
    manufacturer: str
    acquisition_root: str
    label_anchor_rel: str
    samples: List[SampleRecord]

    @property
    def image_count(self) -> int:
        return len(self.samples)


def _normalize_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def _tokenize(value: str) -> List[str]:
    normalized = _normalize_text(value)
    return [tok for tok in re.split(r"[^a-z0-9]+", normalized) if tok]


def _label_from_text(value: str) -> Optional[str]:
    has_l = False
    has_t = False
    for token in _tokenize(value):
        up = token.upper()
        if up in TOKENS_L or up.startswith("LINEAR"):
            has_l = True
        if up in TOKENS_T or up.startswith("TRANSV") or up.startswith("TRASV"):
            has_t = True
    if has_l and has_t:
        return LABEL_AMBIGUOUS
    if has_l:
        return LABEL_L
    if has_t:
        return LABEL_T
    return None


def infer_manufacturer(text: str) -> str:
    normalized = _normalize_text(text)
    best_vendor = "UNKNOWN"
    best_pos: Optional[int] = None
    for vendor, pattern in VENDOR_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        pos = match.start()
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_vendor = vendor
    return best_vendor


def _iter_image_paths(
    dataset_root: Path,
    exclude_dir_re: Optional[re.Pattern[str]],
) -> Iterable[Path]:
    for root_dir, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [
            d
            for d in dirnames
            if d.lower() not in SKIP_DIR_NAMES and (not exclude_dir_re or not exclude_dir_re.search(d))
        ]
        current_dir = Path(root_dir)
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            if filename.startswith("._"):
                continue
            yield (current_dir / filename).resolve()


def _resolve_label_from_path(
    image_path: Path,
    dataset_root: Path,
) -> Tuple[Optional[str], Optional[Path], str]:
    ambiguous_anchor: Optional[Path] = None
    current = image_path.parent
    while True:
        label = _label_from_text(current.name)
        if label == LABEL_L or label == LABEL_T:
            return label, current, "ancestor_name"
        if label == LABEL_AMBIGUOUS and ambiguous_anchor is None:
            ambiguous_anchor = current

        if current == dataset_root:
            break
        if dataset_root not in current.parents:
            break
        current = current.parent

    rel_text = image_path.relative_to(dataset_root).as_posix()
    rel_label = _label_from_text(rel_text)
    if rel_label == LABEL_L or rel_label == LABEL_T:
        return rel_label, None, "full_path_tokens"
    if rel_label == LABEL_AMBIGUOUS:
        return LABEL_AMBIGUOUS, ambiguous_anchor, "full_path_ambiguous"
    if ambiguous_anchor is not None:
        return LABEL_AMBIGUOUS, ambiguous_anchor, "ancestor_name_ambiguous"
    return None, None, ""


def _compute_split_counts(
    n_items: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    if n_items <= 0:
        return {split: 0 for split in SPLITS}
    if n_items == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n_items == 2:
        return {"train": 1, "val": 1, "test": 0}
    if n_items == 3:
        return {"train": 1, "val": 1, "test": 1}

    raw = {split: n_items * ratios[split] for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}

    while sum(counts.values()) < n_items:
        split = max(
            SPLITS,
            key=lambda s: (raw[s] - counts[s], ratios[s], s == "train"),
        )
        counts[split] += 1

    minimums = {"train": 1, "val": 1, "test": 1}
    for split in SPLITS:
        while counts[split] < minimums[split]:
            donors = [s for s in SPLITS if counts[s] > minimums[s]]
            if not donors:
                break
            donor = max(donors, key=lambda s: counts[s])
            counts[donor] -= 1
            counts[split] += 1

    diff = n_items - sum(counts.values())
    if diff != 0:
        counts["train"] += diff
    return counts


def assign_group_splits(
    groups: Sequence[GroupRecord],
    ratios: Dict[str, float],
    seed: int,
    stratify_by_vendor: bool,
) -> Dict[str, str]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, ...], List[GroupRecord]] = defaultdict(list)
    for rec in groups:
        if stratify_by_vendor:
            key = (rec.label, rec.manufacturer)
        else:
            key = (rec.label,)
        grouped[key].append(rec)

    assignments: Dict[str, str] = {}
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        counts = _compute_split_counts(len(bucket), ratios)

        idx = 0
        for split in SPLITS:
            take = counts[split]
            for rec in bucket[idx : idx + take]:
                assignments[rec.group_id] = split
            idx += take

    if len(assignments) != len(groups):
        missing = [g.group_id for g in groups if g.group_id not in assignments]
        raise RuntimeError(f"Split assignment incompleto: gruppi mancanti={missing[:10]}")
    return assignments


def _write_manifest(path: Path, groups: Sequence[GroupRecord], assignments: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "split",
        "label_lt",
        "class_target",
        "group_id",
        "group_size_images",
        "acquisition_root",
        "manufacturer",
        "label_anchor_rel",
        "label_source",
        "source_rel_path",
        "source_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for group in sorted(groups, key=lambda g: g.group_id.lower()):
            split = assignments[group.group_id]
            for sample in sorted(group.samples, key=lambda s: s.source_rel_path.as_posix().lower()):
                writer.writerow(
                    {
                        "image_path": sample.image_path.as_posix(),
                        "split": split,
                        "label_lt": sample.label,
                        "class_target": CLASS_TARGET[sample.label],
                        "group_id": sample.group_id,
                        "group_size_images": group.image_count,
                        "acquisition_root": sample.acquisition_root,
                        "manufacturer": sample.manufacturer,
                        "label_anchor_rel": sample.label_anchor_rel,
                        "label_source": sample.label_source,
                        "source_rel_path": sample.source_rel_path.as_posix(),
                        "source_name": sample.image_path.name,
                    }
                )


def _write_group_table(path: Path, groups: Sequence[GroupRecord], assignments: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group_id",
        "split",
        "label_lt",
        "manufacturer",
        "acquisition_root",
        "label_anchor_rel",
        "image_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for group in sorted(groups, key=lambda g: g.group_id.lower()):
            writer.writerow(
                {
                    "group_id": group.group_id,
                    "split": assignments[group.group_id],
                    "label_lt": group.label,
                    "manufacturer": group.manufacturer,
                    "acquisition_root": group.acquisition_root,
                    "label_anchor_rel": group.label_anchor_rel,
                    "image_count": group.image_count,
                }
            )


def _build_summary(
    groups: Sequence[GroupRecord],
    assignments: Dict[str, str],
    scanned_images: int,
    skipped_no_label: int,
    skipped_ambiguous: int,
    skipped_small_groups: int,
) -> Dict[str, object]:
    image_by_split_label: Dict[str, Dict[str, int]] = {
        split: {LABEL_L: 0, LABEL_T: 0} for split in SPLITS
    }
    group_by_split_label: Dict[str, Dict[str, int]] = {
        split: {LABEL_L: 0, LABEL_T: 0} for split in SPLITS
    }
    image_by_split_vendor: Dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    total_images = 0
    for group in groups:
        split = assignments[group.group_id]
        total_images += group.image_count
        group_by_split_label[split][group.label] += 1
        image_by_split_label[split][group.label] += group.image_count
        image_by_split_vendor[split][group.manufacturer] += group.image_count

    labels_total = Counter(group.label for group in groups)
    vendors_total = Counter(group.manufacturer for group in groups)

    return {
        "scanned_images": scanned_images,
        "kept_images": total_images,
        "skipped_no_label": skipped_no_label,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_small_groups": skipped_small_groups,
        "groups_total": len(groups),
        "groups_by_label": dict(sorted(labels_total.items())),
        "groups_by_manufacturer": dict(sorted(vendors_total.items())),
        "images_by_split_label": image_by_split_label,
        "groups_by_split_label": group_by_split_label,
        "images_by_split_manufacturer": {
            split: dict(sorted(counter.items())) for split, counter in image_by_split_vendor.items()
        },
    }


def _summary_to_text(summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"scanned_images: {summary['scanned_images']}")
    lines.append(f"kept_images: {summary['kept_images']}")
    lines.append(f"skipped_no_label: {summary['skipped_no_label']}")
    lines.append(f"skipped_ambiguous: {summary['skipped_ambiguous']}")
    lines.append(f"skipped_small_groups: {summary['skipped_small_groups']}")
    lines.append(f"groups_total: {summary['groups_total']}")
    lines.append("")
    lines.append("groups_by_label:")
    for label, count in (summary.get("groups_by_label") or {}).items():
        lines.append(f"  - {label}: {count}")
    lines.append("")
    lines.append("images_by_split_label:")
    by_split = summary.get("images_by_split_label") or {}
    for split in SPLITS:
        row = by_split.get(split, {})
        lines.append(f"  - {split}: L={row.get('L', 0)} T={row.get('T', 0)}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a binary L/T dataset manifest from raw acquisition folders.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"),
        help="Radice da scandire in modo ricorsivo (default: SSD_esi1_n3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3"),
        help="Cartella output manifest/split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--min-images-per-group",
        type=int,
        default=1,
        help="Scarta gruppi con meno immagini del valore impostato.",
    )
    parser.add_argument(
        "--max-images-per-group",
        type=int,
        default=0,
        help="0=tutte; se >0 campiona al massimo N immagini per gruppo.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="0=tutti; utile per smoke test su pochi gruppi.",
    )
    parser.add_argument(
        "--exclude-dir-regex",
        type=str,
        default=None,
        help="Regex opzionale per escludere directory durante la scansione.",
    )
    parser.add_argument(
        "--no-stratify-vendor",
        action="store_true",
        help="Stratifica solo per classe L/T (default: stratifica anche per vendor).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root non trovato: {dataset_root}")

    ratios_raw = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
    }
    if any(v < 0.0 for v in ratios_raw.values()):
        raise ValueError("Le ratio devono essere >= 0.")
    ratio_sum = sum(ratios_raw.values())
    if ratio_sum <= 0:
        raise ValueError("La somma train/val/test ratio deve essere > 0.")
    ratios = {k: v / ratio_sum for k, v in ratios_raw.items()}

    exclude_dir_re = re.compile(args.exclude_dir_regex) if args.exclude_dir_regex else None
    scanned_images = 0
    skipped_no_label = 0
    skipped_ambiguous = 0

    grouped_samples: Dict[str, GroupRecord] = {}
    for image_path in _iter_image_paths(dataset_root, exclude_dir_re):
        scanned_images += 1
        rel = image_path.relative_to(dataset_root)
        label, anchor_dir, label_source = _resolve_label_from_path(image_path, dataset_root)
        if label is None:
            skipped_no_label += 1
            continue
        if label == LABEL_AMBIGUOUS:
            skipped_ambiguous += 1
            continue

        rel_parts = rel.parts
        acquisition_root = rel_parts[0] if rel_parts else "UNKNOWN_ACQ"
        manufacturer = infer_manufacturer(acquisition_root)

        if anchor_dir is not None and dataset_root in anchor_dir.parents:
            group_dir = anchor_dir
            label_anchor_rel = anchor_dir.relative_to(dataset_root).as_posix()
        else:
            group_dir = image_path.parent
            label_anchor_rel = image_path.parent.relative_to(dataset_root).as_posix()

        # Keep label in the grouping key to avoid accidental L/T mixing
        # when a broad parent folder can contain both acquisition branches.
        group_base_rel = group_dir.relative_to(dataset_root).as_posix()
        group_id = f"{group_base_rel}::label_{label.lower()}"

        sample = SampleRecord(
            image_path=image_path,
            source_rel_path=rel,
            acquisition_root=acquisition_root,
            manufacturer=manufacturer,
            label=label,
            group_id=group_id,
            label_anchor_rel=label_anchor_rel,
            label_source=label_source,
        )

        if group_id not in grouped_samples:
            grouped_samples[group_id] = GroupRecord(
                group_id=group_id,
                label=label,
                manufacturer=manufacturer,
                acquisition_root=acquisition_root,
                label_anchor_rel=label_anchor_rel,
                samples=[],
            )
        grouped_samples[group_id].samples.append(sample)

    rng = random.Random(args.seed)
    skipped_small_groups = 0
    groups: List[GroupRecord] = []
    for group_id in sorted(grouped_samples):
        group = grouped_samples[group_id]
        if args.max_images_per_group > 0 and len(group.samples) > args.max_images_per_group:
            pool = list(group.samples)
            rng.shuffle(pool)
            group.samples = pool[: args.max_images_per_group]

        if len(group.samples) < args.min_images_per_group:
            skipped_small_groups += 1
            continue
        groups.append(group)

    if args.limit_groups > 0 and len(groups) > args.limit_groups:
        rng.shuffle(groups)
        groups = sorted(groups[: args.limit_groups], key=lambda g: g.group_id.lower())

    if not groups:
        raise RuntimeError(
            "Nessun gruppo valido trovato. Controlla dataset root e regole di labeling L/T."
        )

    assignments = assign_group_splits(
        groups=groups,
        ratios=ratios,
        seed=args.seed,
        stratify_by_vendor=not args.no_stratify_vendor,
    )

    manifest_path = output_dir / "manifest_lt.csv"
    groups_csv_path = output_dir / "groups_lt.csv"
    summary_json_path = output_dir / "summary_lt.json"
    summary_txt_path = output_dir / "summary_lt.txt"

    _write_manifest(manifest_path, groups, assignments)
    _write_group_table(groups_csv_path, groups, assignments)

    summary = _build_summary(
        groups=groups,
        assignments=assignments,
        scanned_images=scanned_images,
        skipped_no_label=skipped_no_label,
        skipped_ambiguous=skipped_ambiguous,
        skipped_small_groups=skipped_small_groups,
    )
    summary["dataset_root"] = dataset_root.as_posix()
    summary["output_dir"] = output_dir.as_posix()
    summary["seed"] = args.seed
    summary["ratios_normalized"] = ratios
    summary["stratify_by_vendor"] = not args.no_stratify_vendor
    summary["max_images_per_group"] = int(args.max_images_per_group)
    summary["min_images_per_group"] = int(args.min_images_per_group)

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_txt_path.write_text(_summary_to_text(summary), encoding="utf-8")

    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Scanned images: {scanned_images}", flush=True)
    print(
        f"Kept images: {summary['kept_images']} | groups: {summary['groups_total']} | "
        f"skipped(no-label/amb/small): {skipped_no_label}/{skipped_ambiguous}/{skipped_small_groups}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Group table: {groups_csv_path}", flush=True)
    print(f"Summary JSON: {summary_json_path}", flush=True)
    print(f"Summary TXT: {summary_txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
