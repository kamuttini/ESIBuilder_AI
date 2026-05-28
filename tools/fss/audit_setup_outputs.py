#!/usr/bin/env python3
"""Audit expected ESIBuilder output layout for each setup_<ID>.fss."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import List

from fss_compat import load_fss


SETUP_RE = re.compile(r"setup_(\d+)\.fss$")


@dataclass
class SetupAuditResult:
    setup_id: int
    ok: bool
    missing_items: List[str]
    checked_items: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DB_setup + DB_echo consistency for generated setups."
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        type=Path,
        help="Folder containing DB_setup and DB_echo (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()

    db_setup = root / "DB_setup"
    db_echo = root / "DB_echo"

    if not db_setup.exists():
        print(f"ERROR: missing folder {db_setup}")
        return 2
    if not db_echo.exists():
        print(f"ERROR: missing folder {db_echo}")
        return 2

    fss_files = sorted(db_setup.glob("setup_*.fss"))
    if not fss_files:
        print(f"ERROR: no setup_*.fss found in {db_setup}")
        return 2

    results: List[SetupAuditResult] = []
    for fss_path in fss_files:
        result = _audit_single_setup(fss_path, db_echo)
        results.append(result)

    print(f"Audited setups: {len(results)}")
    print("")
    failing = 0
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(
            f"setup_{result.setup_id}: {status} "
            f"(checked {result.checked_items} expected items)"
        )
        if result.note:
            print(f"  note: {result.note}")
        if result.missing_items:
            failing += 1
            print("  missing:")
            for item in result.missing_items:
                print(f"  - {item}")

    print("")
    print(f"Summary: {len(results) - failing} OK, {failing} FAIL")
    return 0 if failing == 0 else 1


def _audit_single_setup(fss_path: Path, db_echo: Path) -> SetupAuditResult:
    match = SETUP_RE.search(fss_path.name)
    if not match:
        return SetupAuditResult(
            setup_id=-1,
            ok=False,
            missing_items=[f"Invalid filename: {fss_path.name}"],
            checked_items=0,
            note="",
        )

    setup_id = int(match.group(1))
    setup_echo_dir = db_echo / f"setup_{setup_id}"

    fss = load_fss(fss_path)
    expected_files = _expected_echo_files(fss)

    missing: List[str] = []
    if not setup_echo_dir.exists():
        missing.append(str(setup_echo_dir))
        return SetupAuditResult(
            setup_id=setup_id,
            ok=False,
            missing_items=missing,
            checked_items=len(expected_files),
            note="Missing setup directory under DB_echo.",
        )

    for filename in expected_files:
        if not (setup_echo_dir / filename).exists():
            missing.append(filename)

    return SetupAuditResult(
        setup_id=setup_id,
        ok=(len(missing) == 0),
        missing_items=missing,
        checked_items=len(expected_files),
        note="",
    )


def _expected_echo_files(fss) -> List[str]:
    files: List[str] = []

    # Always expected from createFssFromThis.
    files.extend(
        [
            "echo_name.png",
            "probe_name.png",
            "orientation_0.png",
            "orientation_1.png",
            "orientation_2.png",
            "orientation_3.png",
        ]
    )

    prohibited_count = _count_non_empty_csv_groups(fss.fields.get(15, ""))
    for idx in range(prohibited_count):
        files.append(f"proibited_screen_{idx}_0.png")

    depth_count = _count_pipe_values(fss.fields.get(18, ""))
    for idx in range(depth_count):
        files.append(f"depth_{idx}.png")

    probe_type = _parse_int(fss.fields.get(4, ""))
    if probe_type in {3, 4}:  # PROBE_TYPE_TRANS_L / PROBE_TYPE_TRANS_T
        for idx in range(4):
            files.append(f"trans_type_image_{idx}.png")

    return files


def _count_non_empty_csv_groups(value: str) -> int:
    groups = [token.strip() for token in value.split(",")]
    return len([group for group in groups if group != ""])


def _count_pipe_values(value: str) -> int:
    values = [token.strip() for token in value.split("|")]
    return len([token for token in values if token != ""])


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


if __name__ == "__main__":
    sys.exit(main())

