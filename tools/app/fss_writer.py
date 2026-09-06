"""Serializer for ESI ``.fss`` setup files.

This is the component that was missing: the pipeline predicts values, this module turns
them into the 23 (single probe) or 26 (biplane) lines ESI reads back.

Separator rules, read off templates/DB_setup/*.fss and spiegazione_file_fss.md:

* a template block is ``TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|``
  and grows a ``BM|TH2|`` tail on #17 only;
* a *group* of blocks is ``;``-joined and closed by a trailing ``;``;
* several groups on one line are ``,``-joined, which is why legacy files show ``;,``;
* plain vectors (#18, #19, #20, #23) are ``|``-joined and closed by a trailing ``|``.

Numbers keep the legacy formatting: coordinates as integers, match parameters in
scientific notation with six decimals (``4.748910e+07``), everything else ``%g``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from constants import (
    BIPLANE_PROBE_TYPES,
    FSS_LINE_NAMES,
    DEFAULT_CHANNEL,
    DEFAULT_MATCH_METHOD,
    DEFAULT_P1,
    DEFAULT_P2,
    FSS_LINES_BIPLANE,
    FSS_LINES_SINGLE_PROBE,
    ORIENTATION_KEYS,
    RECT_MAX_HEIGHT,
    RECT_MAX_WIDTH,
)


def _i(value: float | int) -> str:
    return str(int(round(float(value))))


def _g(value: float | int) -> str:
    return f"{float(value):g}"


def _e(value: float | int) -> str:
    return f"{float(value):e}"


@dataclass
class MatchParams:
    """The ``CH:TH:P1:P2:P3:P4:P5`` group."""

    threshold: float = 0.0
    channel: int = DEFAULT_CHANNEL
    p1: float = DEFAULT_P1
    p2: float = DEFAULT_P2
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0

    def render(self) -> str:
        parts = [_i(self.channel)] + [
            _e(v) for v in (self.threshold, self.p1, self.p2, self.p3, self.p4, self.p5)
        ]
        return ":".join(parts)

    @classmethod
    def from_dict(cls, data: Dict) -> "MatchParams":
        return cls(
            threshold=float(data.get("threshold", 0.0)),
            channel=int(data.get("channel", DEFAULT_CHANNEL)),
            p1=float(data.get("p1", DEFAULT_P1)),
            p2=float(data.get("p2", DEFAULT_P2)),
            p3=float(data.get("p3", 0.0)),
            p4=float(data.get("p4", 0.0)),
            p5=float(data.get("p5", 0.0)),
        )


@dataclass
class TemplateBlock:
    """One rectangle plus its match criterion."""

    top: int
    left: int
    bottom: int
    right: int
    check: int = 1  # field B
    params: MatchParams = field(default_factory=MatchParams)
    match_method: int = DEFAULT_MATCH_METHOD
    # #17 only: BM tells ESI a second image must be checked, TH2 carries its criterion.
    second_image: int = 0
    params2: Optional[MatchParams] = None

    def render(self, with_second_image: bool = False) -> str:
        parts = [
            _i(self.top),
            _i(self.left),
            _i(self.bottom),
            _i(self.right),
            _i(self.check),
            self.params.render(),
            _i(self.match_method),
        ]
        if with_second_image:
            parts.append(_i(self.second_image))
            parts.append((self.params2 or MatchParams(p1=DEFAULT_P1, p2=DEFAULT_P2)).render())
        return "|".join(parts) + "|"

    @classmethod
    def from_dict(cls, data: Dict) -> "TemplateBlock":
        return cls(
            top=int(data["top"]),
            left=int(data["left"]),
            bottom=int(data["bottom"]),
            right=int(data["right"]),
            check=int(data.get("check", 1)),
            params=MatchParams.from_dict(data.get("params", {})),
            match_method=int(data.get("match_method", DEFAULT_MATCH_METHOD)),
            second_image=int(data.get("second_image", 0)),
            params2=MatchParams.from_dict(data["params2"]) if data.get("params2") else None,
        )


@dataclass
class ScaleLine:
    """One entry of #21: where to draw the ESI scale for a given depth."""

    x1: int
    x2: int
    y1: int
    y2: int
    length_mm: float
    tick_mm: float
    label_side: int  # -1 left, +1 right

    def render(self) -> str:
        parts = [
            _i(self.x1),
            _i(self.x2),
            _i(self.y1),
            _i(self.y2),
            _g(self.length_mm),
            _g(self.tick_mm),
            _i(self.label_side),
        ]
        return "|".join(parts) + "|"

    @classmethod
    def from_dict(cls, data: Dict) -> "ScaleLine":
        return cls(
            x1=int(data["x1"]),
            x2=int(data["x2"]),
            y1=int(data["y1"]),
            y2=int(data["y2"]),
            length_mm=float(data["length_mm"]),
            tick_mm=float(data.get("tick_mm", 0.5)),
            label_side=int(data.get("label_side", -1)),
        )


def _group(blocks: Sequence[TemplateBlock], with_second_image: bool = False) -> str:
    """``;``-joined blocks, closed by a trailing ``;``."""
    if not blocks:
        return ""
    return ";".join(b.render(with_second_image) for b in blocks) + ";"


def _groups(groups: Sequence[Sequence[TemplateBlock]], with_second_image: bool = False) -> str:
    """Every group is closed by ``,`` on top of its own trailing ``;``, hence the ``;,``.

    Legacy files close the last group too: lines #15 and #17 end with ``;,``.
    """
    return "".join(_group(group, with_second_image) + "," for group in groups)


def _vector(values: Sequence[float], fmt=_g) -> str:
    if not values:
        return ""
    return "|".join(fmt(v) for v in values) + "|"


@dataclass
class FssDocument:
    """Every field of a .fss file, in file order."""

    version: str = ""  # #01 - open point: which version ESI accepts
    id_echo: int = 0  # #02
    id_probe: int = 0  # #03
    probe_type: int = 0  # #04
    kit_needle_guide: int = 0  # #05
    video_input: int = 0  # #06
    video_input_size: tuple[int, int] = (0, 0)  # #07 #08
    video_size: tuple[int, int] = (0, 0)  # #09 #10 (image sample)
    rect_echo: Optional[tuple[int, int, int, int]] = None  # #11 top/left/bottom/right
    group_orientation: int = 4  # #12
    rect_name_echo: Optional[TemplateBlock] = None  # #13
    rect_name_probe: Optional[TemplateBlock] = None  # #14
    proibited_screen: List[List[TemplateBlock]] = field(default_factory=list)  # #15
    rect_orientation: List[TemplateBlock] = field(default_factory=list)  # #16, 4 blocks
    rect_depth: List[List[TemplateBlock]] = field(default_factory=list)  # #17, per depth
    vect_depth: List[float] = field(default_factory=list)  # #18
    pixel_ratio_x: List[float] = field(default_factory=list)  # #19
    pixel_ratio_y: List[float] = field(default_factory=list)  # #20
    scale_lines: List[ScaleLine] = field(default_factory=list)  # #21
    centre_distance: List[List[float]] = field(default_factory=list)  # #22, per depth
    angles: List[float] = field(default_factory=list)  # #23
    rect_trans: List[TemplateBlock] = field(default_factory=list)  # #24, 4 blocks
    id_next_probe: Optional[int] = None  # #25
    biplana_recognition_mode: Optional[int] = None  # #26

    @property
    def is_biplane(self) -> bool:
        return self.probe_type in BIPLANE_PROBE_TYPES

    # -- rendering ---------------------------------------------------------
    def render_lines(self) -> List[str]:
        lines = [
            self.version,  # 01
            _i(self.id_echo),  # 02
            _i(self.id_probe),  # 03
            _i(self.probe_type),  # 04
            _i(self.kit_needle_guide),  # 05
            _i(self.video_input),  # 06
            _i(self.video_input_size[0]),  # 07
            _i(self.video_input_size[1]),  # 08
            _i(self.video_size[0]),  # 09
            _i(self.video_size[1]),  # 10
            _vector(list(self.rect_echo), _i) if self.rect_echo else "",  # 11
            _i(self.group_orientation),  # 12
            self.rect_name_echo.render() if self.rect_name_echo else "",  # 13
            self.rect_name_probe.render() if self.rect_name_probe else "",  # 14
            _groups(self.proibited_screen),  # 15
            _group(self.rect_orientation),  # 16
            _groups(self.rect_depth, with_second_image=True),  # 17
            _vector(self.vect_depth),  # 18
            _vector(self.pixel_ratio_x),  # 19
            _vector(self.pixel_ratio_y),  # 20
            "".join(s.render() + ";" for s in self.scale_lines),  # 21
            "".join(_vector(row) + ";" for row in self.centre_distance),  # 22
            _vector(self.angles),  # 23
        ]
        if self.is_biplane:
            lines.append(_group(self.rect_trans))  # 24
            lines.append(_i(self.id_next_probe) if self.id_next_probe is not None else "|")  # 25
            lines.append(_i(self.biplana_recognition_mode or 0))  # 26
        return lines

    def render(self) -> str:
        return "\n".join(self.render_lines()) + "\n"

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Problems that must be fixed before the file can be handed to ESI."""
        problems: List[str] = []

        if not self.version.strip():
            problems.append("#01 versione non impostata (punto aperto della specifica)")
        for label, value in (("#02 ID_ECHO", self.id_echo), ("#03 ID_PROBE", self.id_probe)):
            if value <= 0:
                problems.append(f"{label} non impostato")
        if self.probe_type <= 0:
            problems.append("#04 PROBE_TYPE non impostato")
        if self.video_input not in (0, 1):
            problems.append("#06 VIDEO_INPUT deve essere 0 (HDMI) o 1 (VGA)")
        for label, size in (("#07/#08", self.video_input_size), ("#09/#10", self.video_size)):
            if size[0] <= 0 or size[1] <= 0:
                problems.append(f"{label} risoluzione non impostata")

        if self.rect_echo is None:
            problems.append("#11 RECT_ECHO mancante")
        else:
            top, left, bottom, right = self.rect_echo
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                problems.append("#11 RECT_ECHO degenere (bottom<=top oppure right<=left)")
            elif width > RECT_MAX_WIDTH or height > RECT_MAX_HEIGHT:
                problems.append(
                    f"#11 RECT_ECHO {width}x{height} oltre il limite ESI "
                    f"{RECT_MAX_WIDTH}x{RECT_MAX_HEIGHT}"
                )
            sample_w, sample_h = self.video_size
            if sample_w and sample_h and (right > sample_w - 1 or bottom > sample_h - 1):
                problems.append("#11 RECT_ECHO fuori dall'immagine campione")

        if self.rect_orientation and len(self.rect_orientation) != len(ORIENTATION_KEYS):
            problems.append(
                f"#16 attesi {len(ORIENTATION_KEYS)} box "
                f"({'/'.join(ORIENTATION_KEYS)}), trovati {len(self.rect_orientation)}"
            )

        depths = len(self.vect_depth)
        if depths == 0:
            problems.append("#18 nessuna depth")
        for label, values in (
            ("#19 PIXEL_RATIO_X", self.pixel_ratio_x),
            ("#20 PIXEL_RATIO_Y", self.pixel_ratio_y),
        ):
            if depths and len(values) != depths:
                problems.append(f"{label}: {len(values)} valori per {depths} depth")
        if depths and self.scale_lines and len(self.scale_lines) != depths:
            problems.append(f"#21 SCALE_LINE: {len(self.scale_lines)} voci per {depths} depth")
        if depths and self.rect_depth and len(self.rect_depth) != depths:
            problems.append(f"#17 RECT_DEPTH: {len(self.rect_depth)} gruppi per {depths} depth")
        if self.centre_distance:
            if depths and len(self.centre_distance) != depths:
                problems.append(
                    f"#22 CENTRE_DISTANCE: {len(self.centre_distance)} righe per {depths} depth"
                )
            expected_angles = len(self.angles)
            for idx, row in enumerate(self.centre_distance):
                if expected_angles and len(row) != expected_angles:
                    problems.append(
                        f"#22 depth {idx + 1}: {len(row)} distanze per {expected_angles} angoli"
                    )
                    break

        if self.is_biplane:
            if len(self.rect_trans) not in (0, len(ORIENTATION_KEYS)):
                problems.append("#24 RECT_TRANS: attesi 4 box di transizione")
            if self.id_next_probe is None:
                problems.append("#25 ID_NEXT_PROBE mancante per una sonda biplana")

        lines = self.render_lines()
        expected = FSS_LINES_BIPLANE if self.is_biplane else FSS_LINES_SINGLE_PROBE
        if len(lines) != expected:
            problems.append(f"il file avrebbe {len(lines)} righe invece di {expected}")

        # No legacy file in templates/DB_setup has an empty line: ESI expects every field.
        already_reported = " ".join(problems)
        for number, value in enumerate(lines, start=1):
            if value.strip():
                continue
            tag = f"#{number:02d}"
            if tag in already_reported:
                continue
            problems.append(f"{tag} {FSS_LINE_NAMES.get(number, '?')} vuota")
        return problems


def write_fss(document: FssDocument, path) -> List[str]:
    """Write the file and return the validation problems found (empty means clean)."""
    from pathlib import Path

    problems = document.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.render(), encoding="utf-8")
    return problems
