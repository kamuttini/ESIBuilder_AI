#!/usr/bin/env python3
"""CLI: compare two .fss files (exact and semantic compatibility)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fss_compat import compare_fss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two ESI .fss files for compatibility."
    )
    parser.add_argument("reference", type=Path, help="Legacy or expected .fss file")
    parser.add_argument("candidate", type=Path, help="Generated .fss file to validate")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Numeric tolerance for semantic comparison (default: 1e-6).",
    )
    parser.add_argument(
        "--show-equal-fields",
        action="store_true",
        help="Also print fields that are equal.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = compare_fss(args.reference, args.candidate, float_tolerance=args.tol)

    print(f"Reference: {result.reference_path}")
    print(f"Candidate: {result.candidate_path}")
    print(f"Byte-identical: {'YES' if result.byte_identical else 'NO'}")
    print(f"Semantic-compatible: {'YES' if result.semantic_equal else 'NO'}")
    print(f"Same line count: {'YES' if result.line_count_equal else 'NO'}")
    print("")

    rows = result.field_results
    if not args.show_equal_fields:
        rows = [row for row in rows if not row.semantic_equal]

    if not rows:
        print("No field differences to report.")
        return 0

    print("Field details:")
    for row in rows:
        status = "OK" if row.semantic_equal else "DIFF"
        print(f"- [{status}] #{row.field_id:02d} {row.field_name}: {row.reason}")
        if not row.semantic_equal:
            print(f"  ref : {_truncate(row.reference)}")
            print(f"  cand: {_truncate(row.candidate)}")

    return 0 if result.semantic_equal else 1


def _truncate(value: str | None, limit: int = 220) -> str:
    if value is None:
        return "<missing>"
    if len(value) <= limit:
        return value
    return value[:limit] + "... [truncated]"


if __name__ == "__main__":
    sys.exit(main())
