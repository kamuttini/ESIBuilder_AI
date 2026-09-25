#!/usr/bin/env python3
"""CLI: compare two .freeze files (exact and semantic compatibility).

Quality gate for the freeze/proibite block, mirroring tools/fss/compare_fss.py.

Semantic compatibility ignores the absolute `strFileName` paths (the legacy
files carry Windows paths of whoever built the setup) and accepts a template
rectangle whose IoU with the reference is at least --min-iou.

Exit codes:
    0  semantically compatible
    1  incompatible
    2  input files missing or unreadable
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_io import FreezeEntry, FreezeFile, read_freeze  # noqa: E402

DEFAULT_MIN_IOU = 0.70


@dataclass
class EntryDiff:
    index: int
    ok: bool
    iou: Optional[float]
    reasons: List[str]


@dataclass
class FreezeComparison:
    reference_path: Path
    candidate_path: Path
    byte_identical: bool
    count_equal: bool
    semantic_equal: bool
    mean_iou: Optional[float]
    entry_diffs: List[EntryDiff]


def compare_freeze(
    reference_path: Path,
    candidate_path: Path,
    min_iou: float = DEFAULT_MIN_IOU,
    compare_file_names: bool = False,
) -> FreezeComparison:
    reference = read_freeze(reference_path)
    candidate = read_freeze(candidate_path)

    byte_identical = reference_path.read_bytes() == candidate_path.read_bytes()
    count_equal = len(reference.entries) == len(candidate.entries)

    diffs: List[EntryDiff] = []
    ious: List[float] = []
    for entry in reference.entries:
        other = candidate.entry_by_index(entry.index)
        if other is None:
            diffs.append(EntryDiff(entry.index, False, None, ["missing in candidate"]))
            continue
        diffs.append(_compare_entry(entry, other, min_iou, compare_file_names))
        ious.append(entry.rect_template.iou(other.rect_template))

    for entry in candidate.entries:
        if reference.entry_by_index(entry.index) is None:
            diffs.append(EntryDiff(entry.index, False, None, ["extra in candidate"]))

    semantic_equal = count_equal and all(diff.ok for diff in diffs)
    mean_iou = sum(ious) / len(ious) if ious else None

    return FreezeComparison(
        reference_path=reference_path,
        candidate_path=candidate_path,
        byte_identical=byte_identical,
        count_equal=count_equal,
        semantic_equal=semantic_equal,
        mean_iou=mean_iou,
        entry_diffs=sorted(diffs, key=lambda item: item.index),
    )


def _compare_entry(
    reference: FreezeEntry,
    candidate: FreezeEntry,
    min_iou: float,
    compare_file_names: bool,
) -> EntryDiff:
    reasons: List[str] = []
    iou = reference.rect_template.iou(candidate.rect_template)
    if iou < min_iou:
        reasons.append(
            f"rectTemplate IoU {iou:.3f} < {min_iou:.2f} "
            f"(ref {reference.rect_template.to_qt()}, cand {candidate.rect_template.to_qt()})"
        )
    if reference.value_find != candidate.value_find:
        reasons.append("bValueFind differs")
    if reference.is_screen_saver != candidate.is_screen_saver:
        reasons.append("isScreenSaver differs")
    if not candidate.has_done:
        reasons.append("bThisHasDone is false in candidate")
    if compare_file_names and Path(reference.file_name).name != Path(candidate.file_name).name:
        reasons.append("strFileName basename differs")
    return EntryDiff(reference.index, not reasons, iou, reasons)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two ESI .freeze files for compatibility."
    )
    parser.add_argument("reference", type=Path, help="Legacy or expected .freeze file")
    parser.add_argument("candidate", type=Path, help="Generated .freeze file to validate")
    parser.add_argument(
        "--min-iou",
        type=float,
        default=DEFAULT_MIN_IOU,
        help=f"Minimum rectTemplate IoU to accept an entry (default: {DEFAULT_MIN_IOU}).",
    )
    parser.add_argument(
        "--compare-file-names",
        action="store_true",
        help="Also require the strFileName basenames to match.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for path in (args.reference, args.candidate):
        if not path.is_file():
            print(f"ERROR: missing file {path}")
            return 2

    result = compare_freeze(
        args.reference,
        args.candidate,
        min_iou=args.min_iou,
        compare_file_names=args.compare_file_names,
    )

    print(f"Reference: {result.reference_path}")
    print(f"Candidate: {result.candidate_path}")
    print(f"Byte-identical: {'YES' if result.byte_identical else 'NO'}")
    print(f"Semantic-compatible: {'YES' if result.semantic_equal else 'NO'}")
    print(f"Same entry count: {'YES' if result.count_equal else 'NO'}")
    if result.mean_iou is not None:
        print(f"Mean rectTemplate IoU: {result.mean_iou:.4f}")
    print("")

    failures = [diff for diff in result.entry_diffs if not diff.ok]
    if not failures:
        print("No entry differences to report.")
        return 0 if result.semantic_equal else 1

    print("Entry details:")
    for diff in failures:
        for reason in diff.reasons:
            print(f"- [DIFF] Freeze_{diff.index}: {reason}")

    return 0 if result.semantic_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
