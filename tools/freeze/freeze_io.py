#!/usr/bin/env python3
"""Read and write legacy ESIBuilder `setup_<ID>.freeze` files.

The legacy Qt application stores the "proibite" (forbidden ultrasound screen
states: freeze, CFM, PW, zoom, ...) of a setup in a QSettings INI file:

    [Freeze_0]
    nIndexOfListSetup=0
    strFileName=C:/.../PROIBITE/FREEZE.png
    rectTemplate=@Rect(1457 339 36 21)
    rectFind=@Rect(1454 336 42 27)
    bValueFind=true
    bThisHasDone=true
    isScreenSaver=false

    [Settings]
    ListFile=image_samples/image_proibite_setup_0.png, ...

Alongside the INI, two image artifacts are produced per entry:

    image_samples/image_proibite_setup_<i>.png        full sample screenshot
    DB_echo/setup_<ID>/proibited_screen_<i>_0.png     crop of rectTemplate

`rectFind` is not an independent annotation: it is `rectTemplate` grown by
FIND_MARGIN_PX on each side and clamped to the image. Measured on the 700
ground-truth entries of SSD_esi1_n1 the rule holds for 673/700 on X and
689/700 on Y; the remaining cases are templates touching the image border
plus a handful of manual edits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FIND_MARGIN_PX = 3
RECT_RE = re.compile(r"@Rect\((-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\)")
GROUP_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
FREEZE_GROUP_RE = re.compile(r"^Freeze_(?P<index>\d+)$")


@dataclass(frozen=True)
class Rect:
    """Qt-style rectangle: top-left corner plus size, all in pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        """Exclusive right edge."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Exclusive bottom edge."""
        return self.y + self.h

    @property
    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def to_qt(self) -> str:
        return f"@Rect({self.x} {self.y} {self.w} {self.h})"

    @classmethod
    def from_qt(cls, value: str) -> Optional["Rect"]:
        match = RECT_RE.search(value or "")
        if match is None:
            return None
        return cls(*(int(g) for g in match.groups()))

    def expanded(
        self,
        margin: int = FIND_MARGIN_PX,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> "Rect":
        """Grow by `margin` on each side, clamped to `image_size` if given."""
        x1 = self.x - margin
        y1 = self.y - margin
        x2 = self.x2 + margin
        y2 = self.y2 + margin
        if image_size is not None:
            width, height = image_size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: "Rect") -> float:
        inter_w = min(self.x2, other.x2) - max(self.x, other.x)
        inter_h = min(self.y2, other.y2) - max(self.y, other.y)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter = inter_w * inter_h
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FreezeEntry:
    """One forbidden-state template, i.e. one `[Freeze_i]` group."""

    index: int
    index_of_list_setup: int
    file_name: str
    rect_template: Rect
    rect_find: Rect
    value_find: bool = True
    has_done: bool = True
    is_screen_saver: bool = False

    @property
    def find_margin_matches_rule(self) -> bool:
        expected = self.rect_template.expanded()
        return expected == self.rect_find


@dataclass
class FreezeFile:
    """Full content of a `setup_<ID>.freeze` file."""

    setup_id: Optional[int] = None
    entries: List[FreezeEntry] = field(default_factory=list)
    list_file: List[str] = field(default_factory=list)
    source_path: Optional[Path] = None

    def entry_by_index(self, index: int) -> Optional[FreezeEntry]:
        for entry in self.entries:
            if entry.index == index:
                return entry
        return None


def _parse_bool(value: str) -> Optional[bool]:
    text = (value or "").strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _setup_id_from_name(path: Path) -> Optional[int]:
    match = re.match(r"setup_(\d+)\.freeze$", path.name)
    return int(match.group(1)) if match else None


def parse_freeze_text(text: str, source_path: Optional[Path] = None) -> FreezeFile:
    """Parse the INI body of a `.freeze` file.

    Unknown groups and keys are ignored: the legacy writer only ever emits
    `[Freeze_i]` groups plus `[Settings]`.
    """
    groups: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = GROUP_RE.match(line)
        if match is not None:
            current = match.group("name")
            groups.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        groups[current][key.strip()] = value.strip()

    entries: List[FreezeEntry] = []
    for name, values in groups.items():
        match = FREEZE_GROUP_RE.match(name)
        if match is None:
            continue
        rect_template = Rect.from_qt(values.get("rectTemplate", ""))
        rect_find = Rect.from_qt(values.get("rectFind", ""))
        if rect_template is None or rect_find is None:
            # The legacy loader rejects entries without both rectangles.
            continue
        try:
            index_of_list = int(values.get("nIndexOfListSetup", ""))
        except ValueError:
            continue
        entries.append(
            FreezeEntry(
                index=int(match.group("index")),
                index_of_list_setup=index_of_list,
                file_name=values.get("strFileName", ""),
                rect_template=rect_template,
                rect_find=rect_find,
                value_find=bool(_parse_bool(values.get("bValueFind", "true"))),
                has_done=bool(_parse_bool(values.get("bThisHasDone", "true"))),
                is_screen_saver=bool(_parse_bool(values.get("isScreenSaver", "false"))),
            )
        )
    entries.sort(key=lambda item: item.index)

    raw_list = groups.get("Settings", {}).get("ListFile", "")
    list_file = [item.strip() for item in raw_list.split(",") if item.strip()]

    return FreezeFile(
        setup_id=_setup_id_from_name(source_path) if source_path else None,
        entries=entries,
        list_file=list_file,
        source_path=source_path,
    )


def read_freeze(path: Path) -> FreezeFile:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_freeze_text(text, source_path=Path(path))


def render_freeze(freeze: FreezeFile) -> str:
    """Render a `.freeze` file the way the legacy QSettings writer does.

    QSettings emits groups in lexicographic order, so a setup with more than
    ten entries writes Freeze_1, Freeze_10, Freeze_11, Freeze_2, ... Sorting
    numerically instead would produce a valid but non-identical file.
    """
    blocks: List[str] = []
    for entry in sorted(freeze.entries, key=lambda item: f"Freeze_{item.index}"):
        blocks.append(
            "\n".join(
                [
                    f"[Freeze_{entry.index}]",
                    f"nIndexOfListSetup={entry.index_of_list_setup}",
                    f"strFileName={entry.file_name}",
                    f"rectTemplate={entry.rect_template.to_qt()}",
                    f"rectFind={entry.rect_find.to_qt()}",
                    f"bValueFind={'true' if entry.value_find else 'false'}",
                    f"bThisHasDone={'true' if entry.has_done else 'false'}",
                    f"isScreenSaver={'true' if entry.is_screen_saver else 'false'}",
                ]
            )
        )
    blocks.append("[Settings]\nListFile=" + ", ".join(freeze.list_file))
    return "\n\n".join(blocks) + "\n"


def write_freeze(freeze: FreezeFile, path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_freeze(freeze), encoding="utf-8")
    return out


def build_freeze_file(
    samples: Sequence[Tuple[str, Rect, bool]],
    image_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    setup_id: Optional[int] = None,
    list_file: Optional[Sequence[str]] = None,
) -> FreezeFile:
    """Assemble a FreezeFile from (file_name, rect_template, is_screen_saver).

    `rect_find` is derived with the legacy margin rule. `bValueFind` is always
    true: it is true in all 700 ground-truth entries on SSD_esi1_n1.
    """
    entries: List[FreezeEntry] = []
    for position, (file_name, rect_template, is_screen_saver) in enumerate(samples):
        size = image_sizes[position] if image_sizes is not None else None
        entries.append(
            FreezeEntry(
                index=position,
                index_of_list_setup=position,
                file_name=file_name,
                rect_template=rect_template,
                rect_find=rect_template.expanded(image_size=size),
                value_find=True,
                has_done=True,
                is_screen_saver=is_screen_saver,
            )
        )
    if list_file is None:
        list_file = [
            f"image_samples/image_proibite_setup_{position}.png"
            for position in range(len(entries))
        ]
    return FreezeFile(setup_id=setup_id, entries=entries, list_file=list(list_file))
