#!/usr/bin/env python3
"""Round-trip check: parse a legacy .fss, re-render it with fss_writer, compare byte by byte.

This is the strongest available proof that the writer speaks the legacy dialect: if a real
file survives parse -> render unchanged, every separator and number format is right.

Uso:
  python3 tools/app/selftest_roundtrip.py OldSoftwareEsiBuilder/templates/DB_setup/*.fss
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fss_writer import FssDocument, MatchParams, ScaleLine, TemplateBlock  # noqa: E402


def _params(token: str) -> MatchParams:
    parts = token.split(":")
    return MatchParams(
        channel=int(float(parts[0])),
        threshold=float(parts[1]),
        p1=float(parts[2]),
        p2=float(parts[3]),
        p3=float(parts[4]),
        p4=float(parts[5]),
        p5=float(parts[6]),
    )


def _block(raw: str, with_second_image: bool = False) -> TemplateBlock:
    tokens = [t for t in raw.split("|") if t != ""]
    block = TemplateBlock(
        top=int(tokens[0]),
        left=int(tokens[1]),
        bottom=int(tokens[2]),
        right=int(tokens[3]),
        check=int(tokens[4]),
        params=_params(tokens[5]),
        match_method=int(tokens[6]),
    )
    if with_second_image and len(tokens) > 7:
        block.second_image = int(tokens[7])
        if len(tokens) > 8:
            block.params2 = _params(tokens[8])
    return block


def _blocks(raw: str, with_second_image: bool = False) -> List[TemplateBlock]:
    return [_block(part, with_second_image) for part in raw.split(";") if part.strip()]


def _groups(raw: str, with_second_image: bool = False) -> List[List[TemplateBlock]]:
    if not raw.strip():
        return []
    return [_blocks(part, with_second_image) for part in raw.split(",") if part.strip()]


def _floats(raw: str) -> List[float]:
    return [float(part) for part in raw.split("|") if part.strip()]


def parse_fss(path: Path) -> FssDocument:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    line = lambda n: lines[n - 1] if len(lines) >= n else ""  # noqa: E731

    doc = FssDocument(
        version=line(1),
        id_echo=int(line(2)),
        id_probe=int(line(3)),
        probe_type=int(line(4)),
        kit_needle_guide=int(line(5)),
        video_input=int(line(6)),
        video_input_size=(int(line(7)), int(line(8))),
        video_size=(int(line(9)), int(line(10))),
        group_orientation=int(line(12)),
        vect_depth=_floats(line(18)),
        pixel_ratio_x=_floats(line(19)),
        pixel_ratio_y=_floats(line(20)),
        angles=_floats(line(23)),
    )
    rect = [int(v) for v in line(11).split("|") if v.strip()]
    if len(rect) == 4:
        doc.rect_echo = tuple(rect)
    if line(13).strip():
        doc.rect_name_echo = _block(line(13))
    if line(14).strip():
        doc.rect_name_probe = _block(line(14))
    doc.proibited_screen = _groups(line(15))
    doc.rect_orientation = _blocks(line(16))
    doc.rect_depth = _groups(line(17), with_second_image=True)
    doc.scale_lines = [
        ScaleLine(
            x1=int(float(f[0])), x2=int(float(f[1])), y1=int(float(f[2])), y2=int(float(f[3])),
            length_mm=float(f[4]), tick_mm=float(f[5]), label_side=int(float(f[6])),
        )
        for f in (
            [t for t in entry.split("|") if t.strip()]
            for entry in line(21).split(";")
            if entry.strip()
        )
    ]
    doc.centre_distance = [_floats(row) for row in line(22).split(";") if row.strip()]
    if len(lines) >= 26:
        doc.rect_trans = _blocks(line(24))
        raw_next = line(25).strip()
        doc.id_next_probe = int(raw_next) if raw_next not in ("", "|") else None
        doc.biplana_recognition_mode = int(line(26) or 0)
    return doc


def check(path: Path) -> Optional[str]:
    original = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rendered = parse_fss(path).render_lines()
    if len(original) != len(rendered):
        return f"{len(original)} righe originali, {len(rendered)} rigenerate"
    diffs = []
    for index, (want, got) in enumerate(zip(original, rendered), start=1):
        if want.strip() != got.strip():
            diffs.append(
                f"riga #{index:02d} differisce"
                f"\n    originale:  ...{want.strip()[-60:]}"
                f"\n    rigenerata: ...{got.strip()[-60:]}"
            )
    return "\n  ".join(diffs) if diffs else None


def main(argv: List[str]) -> int:
    paths = [Path(arg) for arg in argv]
    if not paths:
        print("uso: selftest_roundtrip.py <file.fss> [...]")
        return 2
    failures = 0
    for path in paths:
        problem = check(path)
        if problem:
            failures += 1
            print(f"DIVERSO  {path.name}: {problem}")
        else:
            print(f"identico {path.name}")
    print()
    print(
        f"{len(paths) - failures}/{len(paths)} file rigenerati identici"
        if failures
        else f"tutti i {len(paths)} file legacy rigenerati identici"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
