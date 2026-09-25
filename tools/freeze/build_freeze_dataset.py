#!/usr/bin/env python3
"""Build the freeze/proibite dataset from legacy ESIBuilder projects.

Ground truth comes from the legacy workspaces (one folder per setup):

    <workspace>/DB_setup/setup_<ID>.freeze          rectTemplate + flags
    <workspace>/image_samples/image_proibite_setup_<i>.png   sample screenshot
    <workspace>/DB_echo/setup_<ID>/proibited_screen_<i>_0.png  template crop

The legacy application renames every sample to `image_proibite_setup_<i>.png`,
losing the state it represents (freeze, CFM, PW, zoom, ...). We recover that
label by matching the sample byte-for-byte against the original acquisition
files, whose names carry the state (`PROIBITE/FREEZE.png` and friends).

Outputs, under --output-dir:

    manifest_proibite.csv    one row per [Freeze_i] entry
    setups.csv               one row per setup
    unmatched_samples.txt    samples with no original acquisition file
    summary.json             counts, class distribution, split sizes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS_DIR / "ultrasound"))

from freeze_io import Rect, read_freeze  # noqa: E402
from prepare_ultrasound_rect_dataset import infer_manufacturer  # noqa: E402

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", "__pycache__"}

# Ordered: the first pattern that matches the original file name wins, so that
# e.g. "FREEZE_CFM.png" is filed under freeze rather than cfm.
CLASS_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("freeze", r"FREEZE|FROZEN|CONGELAT"),
    ("cfm", r"CFM|COLORDOPPLER|COLOR|CPA|PDI|POWERDOPPLER|^CF$|^PD$"),
    ("pw", r"PW|SPECTRAL|DOPPLER|^SV$"),
    ("mmode", r"MMODE|^M$|^MM$"),
    ("zoom", r"ZOOM"),
    ("split", r"SPLIT|QUAD|DUAL|DIVIDI"),
    ("elasto", r"ELASTO|^ELA$|SHEARWAVE"),
    ("biopsy", r"BIOPSIA|BIOPSY|NEEDLE"),
    ("contrast", r"CONTRAST|^CNTI$|^CEUS$"),
    ("measure", r"MEASURE|MISURA"),
)
GENERIC_SAMPLE_RE = re.compile(r"^image_proibite_setup_\d+\.png$", re.IGNORECASE)


@dataclass
class EntryRow:
    entry_id: str
    workspace: str
    setup_id: Optional[int]
    manufacturer: str
    freeze_index: int
    class_label: str
    class_source: str
    original_name: str
    original_path: str
    sample_path: str
    crop_path: str
    image_width: Optional[int]
    image_height: Optional[int]
    template_x: int
    template_y: int
    template_w: int
    template_h: int
    find_x: int
    find_y: int
    find_w: int
    find_h: int
    find_follows_rule: bool
    is_screen_saver: bool
    value_find: bool
    freeze_path: str
    split: str = ""


def _walk_files(root: Path) -> Iterable[Path]:
    for dir_path, dir_names, file_names in os.walk(root):
        dir_names[:] = [d for d in dir_names if not d.startswith(".") and d not in SKIP_DIRS]
        for name in file_names:
            if name.startswith("."):
                continue
            yield Path(dir_path) / name


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalized_stem(name: str) -> str:
    stem = Path(name).stem.upper()
    return re.sub(r"[^A-Z]", "", stem)


def classify_original_name(name: str) -> str:
    """Map an original acquisition file name to a forbidden-state class."""
    stem = _normalized_stem(name)
    if not stem:
        return "unknown"
    for label, pattern in CLASS_PATTERNS:
        if re.search(pattern, stem):
            return label
    return "unknown"


def _is_acquisition_candidate(path: Path) -> bool:
    """Files that can plausibly be the original of a legacy proibite sample."""
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    lowered = str(path).lower()
    name = path.name.lower()
    return "proibit" in lowered or "freeze" in name or "frozen" in name


def build_original_index(acquisition_roots: Sequence[Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for root in acquisition_roots:
        if not root.exists():
            print(f"WARNING: acquisition root not found: {root}")
            continue
        for path in _walk_files(root):
            if _is_acquisition_candidate(path):
                index[_md5(path)].append(path)
    return index


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _pick_original(paths: Sequence[Path]) -> Path:
    """Prefer an informative name over a generic one when several files match."""
    named = [p for p in paths if not GENERIC_SAMPLE_RE.match(p.name)]
    pool = named or list(paths)
    return sorted(pool, key=lambda p: (classify_original_name(p.name) == "unknown", str(p)))[0]


def collect_entries(
    legacy_root: Path,
    original_index: Dict[str, List[Path]],
) -> Tuple[List[EntryRow], List[str]]:
    rows: List[EntryRow] = []
    unmatched: List[str] = []

    freeze_paths = sorted(p for p in _walk_files(legacy_root) if p.suffix == ".freeze")
    print(f"legacy .freeze files: {len(freeze_paths)}")

    for freeze_path in freeze_paths:
        freeze = read_freeze(freeze_path)
        workspace_parts = freeze_path.relative_to(legacy_root).parts
        workspace = workspace_parts[0]
        workspace_dir = legacy_root / workspace
        manufacturer = infer_manufacturer(workspace)
        setup_id = freeze.setup_id
        echo_dir = workspace_dir / "DB_echo" / f"setup_{setup_id}"

        for entry in freeze.entries:
            sample_path = workspace_dir / "image_samples" / f"image_proibite_setup_{entry.index}.png"
            if not sample_path.exists():
                # Some early projects kept the sample under its original name.
                legacy_name = Path(entry.file_name.replace("\\", "/")).name
                alternative = workspace_dir / "image_samples" / legacy_name
                sample_path = alternative if alternative.exists() else sample_path

            crop_path = echo_dir / f"proibited_screen_{entry.index}_0.png"

            class_source = "legacy_name"
            original_name = Path(entry.file_name.replace("\\", "/")).name
            original_path = ""
            if GENERIC_SAMPLE_RE.match(original_name) or not original_name:
                class_source = "unmatched"
                original_name = ""
                if sample_path.exists():
                    matches = original_index.get(_md5(sample_path), [])
                    if matches:
                        picked = _pick_original(matches)
                        original_name = picked.name
                        original_path = str(picked)
                        class_source = "content_match"
                    else:
                        unmatched.append(str(sample_path))

            class_label = classify_original_name(original_name) if original_name else "unknown"
            size = _image_size(sample_path) if sample_path.exists() else None

            rows.append(
                EntryRow(
                    entry_id=f"{workspace}#{setup_id}#{entry.index}",
                    workspace=workspace,
                    setup_id=setup_id,
                    manufacturer=manufacturer,
                    freeze_index=entry.index,
                    class_label=class_label,
                    class_source=class_source,
                    original_name=original_name,
                    original_path=original_path,
                    sample_path=str(sample_path) if sample_path.exists() else "",
                    crop_path=str(crop_path) if crop_path.exists() else "",
                    image_width=size[0] if size else None,
                    image_height=size[1] if size else None,
                    template_x=entry.rect_template.x,
                    template_y=entry.rect_template.y,
                    template_w=entry.rect_template.w,
                    template_h=entry.rect_template.h,
                    find_x=entry.rect_find.x,
                    find_y=entry.rect_find.y,
                    find_w=entry.rect_find.w,
                    find_h=entry.rect_find.h,
                    find_follows_rule=entry.find_margin_matches_rule,
                    is_screen_saver=entry.is_screen_saver,
                    value_find=entry.value_find,
                    freeze_path=str(freeze_path),
                )
            )

    return rows, unmatched


def assign_splits(
    rows: Sequence[EntryRow],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    """Leak-free split: a whole workspace goes to a single split.

    Workspaces are shuffled inside each manufacturer so that every vendor is
    represented in train/val/test whenever it has enough workspaces.
    """
    by_manufacturer: Dict[str, List[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.workspace in seen:
            continue
        seen.add(row.workspace)
        by_manufacturer[row.manufacturer].append(row.workspace)

    rng = random.Random(seed)
    assignment: Dict[str, str] = {}
    for manufacturer, workspaces in sorted(by_manufacturer.items()):
        pool = sorted(workspaces)
        rng.shuffle(pool)
        total = len(pool)
        n_val = int(round(total * val_ratio))
        n_test = int(round(total * test_ratio))
        if total >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        while n_val + n_test >= total and n_val + n_test > 0:
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1
        for position, workspace in enumerate(pool):
            if position < n_test:
                assignment[workspace] = "test"
            elif position < n_test + n_val:
                assignment[workspace] = "val"
            else:
                assignment[workspace] = "train"
    return assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n1"),
        help="Root holding the legacy ESIBuilder workspaces (default: %(default)s).",
    )
    parser.add_argument(
        "--acquisition-root",
        type=Path,
        action="append",
        default=None,
        help="Acquisition root used to recover original file names (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: %(default)s).")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    acquisition_roots = args.acquisition_root or [
        Path("/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"),
        Path("/Volumes/SSD_esi1_n3/ACQUISITION"),
    ]

    if not args.legacy_root.exists():
        print(f"ERROR: legacy root not found: {args.legacy_root}")
        return 2

    print("indexing acquisition candidates ...")
    original_index = build_original_index(acquisition_roots)
    print(f"distinct acquisition hashes: {len(original_index)}")

    rows, unmatched = collect_entries(args.legacy_root, original_index)
    if not rows:
        print("ERROR: no freeze entries found")
        return 2

    assignment = assign_splits(rows, args.seed, args.val_ratio, args.test_ratio)
    for row in rows:
        row.split = assignment.get(row.workspace, "train")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_proibite.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    setups: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = f"{row.workspace}#{row.setup_id}"
        record = setups.setdefault(
            key,
            {
                "workspace": row.workspace,
                "setup_id": row.setup_id,
                "manufacturer": row.manufacturer,
                "split": row.split,
                "freeze_path": row.freeze_path,
                "num_entries": 0,
                "num_freeze": 0,
                "num_unknown": 0,
            },
        )
        record["num_entries"] = int(record["num_entries"]) + 1
        if row.class_label == "freeze":
            record["num_freeze"] = int(record["num_freeze"]) + 1
        if row.class_label == "unknown":
            record["num_unknown"] = int(record["num_unknown"]) + 1

    setups_path = output_dir / "setups.csv"
    with open(setups_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(setups.values())).keys()))
        writer.writeheader()
        for record in setups.values():
            writer.writerow(record)

    (output_dir / "unmatched_samples.txt").write_text("\n".join(unmatched), encoding="utf-8")

    class_counts = Counter(row.class_label for row in rows)
    summary = {
        "legacy_root": str(args.legacy_root),
        "acquisition_roots": [str(p) for p in acquisition_roots],
        "seed": args.seed,
        "num_entries": len(rows),
        "num_setups": len(setups),
        "num_workspaces": len({row.workspace for row in rows}),
        "class_counts": dict(class_counts.most_common()),
        "class_source_counts": dict(Counter(row.class_source for row in rows)),
        "split_entries": dict(Counter(row.split for row in rows)),
        "split_workspaces": dict(Counter(assignment.values())),
        "manufacturer_counts": dict(Counter(row.manufacturer for row in rows).most_common()),
        "freeze_entries": class_counts.get("freeze", 0),
        "freeze_workspaces": len({row.workspace for row in rows if row.class_label == "freeze"}),
        "entries_missing_sample": sum(1 for row in rows if not row.sample_path),
        "entries_missing_crop": sum(1 for row in rows if not row.crop_path),
        "unmatched_samples": len(unmatched),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
