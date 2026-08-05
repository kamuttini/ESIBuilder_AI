#!/usr/bin/env python3
"""Shared primitives for the scale (SCALE_LINE, .fss line 21) block.

Legacy .fss facts established on 2026-07-29 by inspecting real setups:

* line 11 ``RECT_ECHO``  -> ``x1|y1|x2|y2|``  (ultrasound rectangle)
* line 18 ``VECT_DEPTH`` -> ``d0|d1|...|``    (depth values, mm, one per depth index)
* line 19 ``PIXEL_RATIO_X`` / 20 ``PIXEL_RATIO_Y`` -> mm per pixel, one per depth index
* line 21 ``SCALE_LINE``  -> ``x1|x2|y1|y2|length_mm|tick|side|`` groups joined by ``;``

In the legacy Qt app this line is labelled "Linea calcolo pixel ratio, punto zero e
posizione scala" (``wdgpagedepthvalue.cpp``): the operator drags a *vertical* segment
(x1 == x2) from the tick labelled ``0`` down to some other labelled tick, and declares
its physical length. So the stored triple carries three things, and only the first two
are objective:

1. the calibration ``mm_per_px = length_mm / length_px`` (equals ``PIXEL_RATIO_Y``),
2. the zero point ``y_top`` and the scale column ``x``,
3. ``y_bottom``, which is *where the operator happened to stop* — any labelled tick
   below zero is legally valid, so this endpoint is a convention, not a fact.

That is why ``length_mm`` is frequently well below the ``VECT_DEPTH`` of the same
depth index, and why regressing ``y_bottom`` directly (runs ``32_``/``34_``/``36_``)
was fighting label noise it could not win against.

Derived quantities::

    length_px     = y2 - y1
    mm_per_px     = length_mm / length_px          == PIXEL_RATIO_Y
    n_intervals   = length_mm / (tick_cm * 10)
    tick_pitch_px = length_px / n_intervals

``tick`` is expressed in **cm** while ``length_mm`` is in **mm** (legacy inconsistency;
``qscaledline.cpp`` prints tick labels as ``i * tick``, matching the "0 cm / 1 / 2.0"
overlay drawn by the scanner).

Contrary to the 2026-04-16 intake note ("scala dentro il rect"), on real BK setups
the scale sits *outside* the rectangle, in the black margin to its right. The audit
tool measures this per vendor instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# .fss field ids we care about
# --------------------------------------------------------------------------- #

F_VERSION = 1
F_ID_ECHO = 2
F_ID_PROBE = 3
F_PROBE_TYPE = 4
F_VIDEO_X_SIZE = 9
F_VIDEO_Y_SIZE = 10
F_RECT_ECHO = 11
F_RECT_DEPTH = 17
F_VECT_DEPTH = 18
F_PIXEL_RATIO_X = 19
F_PIXEL_RATIO_Y = 20
F_SCALE_LINE = 21

# --------------------------------------------------------------------------- #
# vendor inference (kept in sync with tools/fss/train_scale_line_image_model_per_vendor.py)
# --------------------------------------------------------------------------- #

VENDOR_RULES: Sequence[Tuple[str, str]] = (
    ("alpinion", "Alpinion"),
    ("bk", "BK"),
    ("canon", "Canon"),
    ("toshiba", "Canon"),
    ("esaote", "Esaote"),
    ("mylab", "Esaote"),
    ("exactvu", "ExactVu"),
    ("edap", "ExactVu"),
    ("hitachi", "Hitachi"),
    ("aloka", "Hitachi"),
    ("mindray", "Mindray"),
    ("philips", "Philips"),
    ("siemens", "Siemens"),
    ("sonostar", "Sonostar"),
    ("terason", "Terason"),
    ("vinno", "Vinno"),
    ("koelis", "Koelis"),
    ("ge ", "GE"),
    ("logiq", "GE"),
    ("voluson", "GE"),
)


def infer_vendor(text: str) -> str:
    """Vendor from a free-text config folder name. ``Unknown`` when no rule fires."""
    t = f" {text.lower()} "
    # longest key first so "sonostar" wins over "bk" inside odd folder names
    for key, vendor in sorted(VENDOR_RULES, key=lambda kv: -len(kv[0])):
        if key in t:
            return vendor
    return "Unknown"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def read_fss_fields(path: Path) -> Dict[int, str]:
    """Read a .fss into ``{line_number: raw_text}`` (1-based, CR/LF stripped)."""
    raw = path.read_bytes().decode("utf-8", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return {i + 1: line for i, line in enumerate(lines)}


def _pipe_tokens(text: str) -> List[str]:
    return [t for t in text.strip().split("|") if t != ""]


def parse_float_vector(text: str) -> List[float]:
    out: List[float] = []
    for tok in _pipe_tokens(text):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


@dataclass(frozen=True)
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_rect_echo(text: str) -> Optional[Rect]:
    """``x1|y1|x2|y2|`` -> Rect. Returns None when malformed."""
    toks = _pipe_tokens(text)
    if len(toks) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(t))) for t in toks[:4])
    except ValueError:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Rect(x1, y1, x2, y2)


@dataclass(frozen=True)
class ScaleEntry:
    """One SCALE_LINE group, i.e. the scale for a single depth index."""

    depth_index: int
    x1: float
    x2: float
    y1: float
    y2: float
    length_mm: float
    tick_cm: float
    side_raw: int

    @property
    def x(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def y_zero(self) -> float:
        """``y1`` is the zero end of the calibration line, not the topmost point.

        21.7% of the n1 corpus (1146/5290 rows) stores ``y1 > y2``: on UD-flipped
        acquisitions the scanner prints ``0.0 cm`` at the *bottom*. Collapsing the pair
        to (min, max) — as the earlier ``32_``/``34_``/``36_`` targets did — silently
        put the zero at the wrong end on a fifth of the dataset.
        """
        return self.y1

    @property
    def y_far(self) -> float:
        """The other end, at ``length_mm`` from the zero."""
        return self.y2

    @property
    def zero_at(self) -> str:
        return "bottom" if self.y1 > self.y2 else "top"

    @property
    def y_top(self) -> float:
        return min(self.y1, self.y2)

    @property
    def y_bottom(self) -> float:
        return max(self.y1, self.y2)

    @property
    def length_px(self) -> float:
        return self.y_bottom - self.y_top

    @property
    def is_vertical(self) -> bool:
        return abs(self.x1 - self.x2) <= 1.0

    @property
    def n_intervals(self) -> Optional[float]:
        """How many minor ticks fit between the ``0`` tick and the end of the segment."""
        step_mm = self.tick_cm * 10.0
        if step_mm <= 0:
            return None
        return self.length_mm / step_mm

    @property
    def tick_pitch_px(self) -> Optional[float]:
        n = self.n_intervals
        if not n:
            return None
        return self.length_px / n

    @property
    def mm_per_px(self) -> Optional[float]:
        """The objective part of the label: must equal ``PIXEL_RATIO_Y``."""
        if self.length_px <= 0:
            return None
        return self.length_mm / self.length_px


def parse_scale_line(text: str) -> List[ScaleEntry]:
    """Parse line 21 into one ScaleEntry per depth index (order = depth index)."""
    entries: List[ScaleEntry] = []
    for idx, group in enumerate(text.split(";")):
        group = group.strip()
        if not group:
            continue
        toks = _pipe_tokens(group)
        if len(toks) < 6:
            continue
        try:
            x1, x2, y1, y2 = (float(t) for t in toks[:4])
            length_mm = float(toks[4])
            tick_cm = float(toks[5])
        except ValueError:
            continue
        try:
            side_raw = int(float(toks[6])) if len(toks) > 6 else 0
        except ValueError:
            side_raw = 0
        entries.append(
            ScaleEntry(
                depth_index=idx,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                length_mm=length_mm,
                tick_cm=tick_cm,
                side_raw=side_raw,
            )
        )
    return entries


@dataclass
class ScaleSetup:
    """Everything the scale block needs from one setup_NNN.fss."""

    fss_path: Path
    config_folder: str
    setup_id: str
    vendor: str
    id_echo: str
    id_probe: str
    video_x_size: Optional[int]
    video_y_size: Optional[int]
    rect: Optional[Rect]
    vect_depth: List[float]
    pixel_ratio_x: List[float]
    pixel_ratio_y: List[float]
    scale_entries: List[ScaleEntry]
    n_lines: int


def load_scale_setup(fss_path: Path) -> ScaleSetup:
    fields = read_fss_fields(fss_path)
    config_folder = _config_folder_of(fss_path)

    def _int(fid: int) -> Optional[int]:
        try:
            return int(round(float(fields.get(fid, "").strip())))
        except (TypeError, ValueError):
            return None

    return ScaleSetup(
        fss_path=fss_path,
        config_folder=config_folder,
        setup_id=fss_path.stem.replace("setup_", ""),
        vendor=infer_vendor(config_folder),
        id_echo=fields.get(F_ID_ECHO, "").strip(),
        id_probe=fields.get(F_ID_PROBE, "").strip(),
        video_x_size=_int(F_VIDEO_X_SIZE),
        video_y_size=_int(F_VIDEO_Y_SIZE),
        rect=parse_rect_echo(fields.get(F_RECT_ECHO, "")),
        vect_depth=parse_float_vector(fields.get(F_VECT_DEPTH, "")),
        pixel_ratio_x=parse_float_vector(fields.get(F_PIXEL_RATIO_X, "")),
        pixel_ratio_y=parse_float_vector(fields.get(F_PIXEL_RATIO_Y, "")),
        scale_entries=parse_scale_line(fields.get(F_SCALE_LINE, "")),
        n_lines=len([k for k in fields if fields[k].strip() != ""]),
    )


def _config_folder_of(fss_path: Path) -> str:
    """``<root>/<config folder>/DB_setup/setup_N.fss`` -> ``<config folder>``."""
    parts = fss_path.parts
    if len(parts) >= 3 and parts[-2] == "DB_setup":
        return parts[-3]
    return fss_path.parent.name


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_INSIDE = "inside"
SIDE_UNKNOWN = "unknown"


def side_vs_rect(x: float, rect: Optional[Rect]) -> str:
    """Where the scale column sits relative to the ultrasound rectangle."""
    if rect is None:
        return SIDE_UNKNOWN
    if x > rect.x2:
        return SIDE_RIGHT
    if x < rect.x1:
        return SIDE_LEFT
    return SIDE_INSIDE


def signed_offset_from_rect(x: float, rect: Optional[Rect]) -> Optional[float]:
    """Distance in px from the nearest vertical rect edge (positive = outside)."""
    if rect is None:
        return None
    if x > rect.x2:
        return x - rect.x2
    if x < rect.x1:
        return rect.x1 - x
    return -min(x - rect.x1, rect.x2 - x)


# --------------------------------------------------------------------------- #
# image resolution
# --------------------------------------------------------------------------- #

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Ordered by how much we trust the pairing between file name and depth index.
# ``image_depth_value_setup_{i}`` is the frame the legacy app captured while the
# operator was setting depth index ``i``: that is the reliable one.
IMAGE_STEM_PATTERNS = (
    "image_depth_value_setup_{i}",
    "image_depth_find_no_flip_setup_{i}",
    "image_depth_find_flip_ud_setup_{i}",
    "image_depth_find_flip_lr_setup_{i}",
    "image_depth_find_flip_lrud_setup_{i}",
)


def find_depth_image(
    fss_path: Path,
    depth_index: int,
    patterns: Sequence[str] = IMAGE_STEM_PATTERNS,
) -> Optional[Path]:
    """Resolve the full-frame image for ``depth_index``.

    Unlike the older loader in ``tools/fss/train_scale_line_image_model.py`` this
    function never falls back to "any image in the folder": a wrong image/depth
    pairing silently poisons the ground truth, so we would rather return None.
    """
    samples = fss_path.parent.parent / "image_samples"
    if not samples.is_dir():
        return None
    for pattern in patterns:
        stem = pattern.format(i=depth_index)
        for ext in IMAGE_EXTS:
            cand = samples / f"{stem}{ext}"
            if cand.is_file() and not cand.name.startswith("._"):
                return cand
    return None


def iter_scale_setups(root: Path):
    """Yield ScaleSetup for every ``*/DB_setup/setup_*.fss`` under ``root``."""
    for fss in sorted(root.glob("*/DB_setup/setup_*.fss")):
        if fss.name.startswith("._"):
            continue
        try:
            yield load_scale_setup(fss)
        except Exception as exc:  # noqa: BLE001 - audit must not die on one file
            print(f"[warn] cannot load {fss}: {exc}", flush=True)
