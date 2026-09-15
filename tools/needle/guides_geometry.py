#!/usr/bin/env python3
"""Legacy geometry of the needle-guide lines: .fss + .ndg -> the line drawn on the frame.

Port of `QLineGuide::upDateLine` (OldSoftwareEsiBuilder/qlineguide.cpp). It exists so the
new app can draw exactly what the old one drew, and so a measured line can be turned back
into the numbers ESI reads.

What the two .fss lines mean (qfilefss.h):

- #23 ANGLE, one angle per guide-line family, degrees in -180..+180 against the X axis.
  It is the *measured* angle, not the nominal angle of the kit.
- #22 CENTRE_DISTANCE, millimetres from the top of RECT_ECHO to the point where the FIRST
  line crosses the central vertical, one group per depth and inside it one value per angle.

Coordinates follow the original: `m_rectView` is built as QRect((0,0), rectEcho.size()), so
everything is relative to the RECT_ECHO crop. Add (rect.left, rect.top) for frame coordinates.

Two traps kept from the original:

- a vertical line (|angle| == 90) measures its distance horizontally, not vertically, and the
  original adds `rectView.center().x() * pixelRatio_X` to it;
- lines after the first are spaced by the .ndg distances divided by cos(angle), because the
  spacing is measured along the probe axis, not along the line.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Rect:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


@dataclass
class Setup:
    """The part of a .fss this module needs."""

    path: Path
    probe_type: int
    kit_id: int
    video_size: Tuple[int, int]
    rect_echo: Rect
    depths: List[float]
    pixel_ratio_x: List[float]
    pixel_ratio_y: List[float]
    centre_distance: List[List[float]]   # [depth][angle]
    angles: List[float]                  # [angle]

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    def consistent(self) -> bool:
        """The three vectors must agree, otherwise the file is not usable as ground truth."""
        if not self.angles or not self.depths or not self.centre_distance:
            return False
        if len(self.centre_distance) != len(self.depths):
            return False
        if any(len(row) != len(self.angles) for row in self.centre_distance):
            return False
        return len(self.pixel_ratio_x) == len(self.depths) == len(self.pixel_ratio_y)


def _floats(line: str) -> List[float]:
    out = []
    for token in line.split("|"):
        token = token.strip()
        if token:
            try:
                out.append(float(token))
            except ValueError:
                pass
    return out


def read_setup(path: Path) -> Optional[Setup]:
    try:
        lines = Path(path).read_text(encoding="latin-1").splitlines()
    except OSError:
        return None
    if len(lines) < 23:
        return None
    try:
        probe_type = int(lines[3].strip())
        kit_id = int(lines[4].strip())
        video = (int(lines[8].strip()), int(lines[9].strip()))
        top, left, bottom, right = (int(v) for v in _floats(lines[10])[:4])
    except (ValueError, IndexError):
        return None

    centre = [_floats(group) for group in lines[21].split(";") if group.strip()]
    return Setup(
        path=Path(path),
        probe_type=probe_type,
        kit_id=kit_id,
        video_size=video,
        rect_echo=Rect(top=top, left=left, bottom=bottom, right=right),
        depths=_floats(lines[17]),
        pixel_ratio_x=_floats(lines[18]),
        pixel_ratio_y=_floats(lines[19]),
        centre_distance=centre,
        angles=_floats(lines[22]),
    )


def read_ndg(path: Path) -> Optional[List[List[float]]]:
    """Distances between consecutive guide lines, per angle.

    Binary despite what the header comment in qfilendg.h says: a QDataStream
    QVector<QVector<float>>, big endian, with the floats written as doubles.
    Returns [angle][line] distances in mm from the first line.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    try:
        offset = 0
        (n_angles,) = struct.unpack_from(">i", raw, offset)
        offset += 4
        if not 0 < n_angles < 1000:
            return None
        out: List[List[float]] = []
        for _ in range(n_angles):
            (count,) = struct.unpack_from(">i", raw, offset)
            offset += 4
            if not 0 <= count < 10000:
                return None
            out.append(list(struct.unpack_from(f">{count}d", raw, offset)))
            offset += 8 * count
        return out
    except struct.error:
        return None


def centre_distance_mm(setup: Setup, depth_index: int, angle_index: int, line_index: int,
                       line_distances: Sequence[float]) -> float:
    """Distance of line `line_index`, following the original's spacing rule."""
    angle = setup.angles[angle_index]
    distance = setup.centre_distance[depth_index][angle_index]

    if abs(angle) == 90.0:
        centre_x = (setup.rect_echo.width - 1) / 2.0
        distance += centre_x * setup.pixel_ratio_x[depth_index]

    if line_index > 0:
        if abs(angle) != 90.0:
            # the spacing is measured along the probe axis, so it stretches with the tilt
            angle_rad = math.radians(angle if abs(angle) < 90 else 180.0 - angle)
            distance += line_distances[line_index - 1] / math.cos(angle_rad)
        else:
            distance += line_distances[line_index - 1]
    return distance


def line_in_crop(setup: Setup, depth_index: int, angle_index: int, line_index: int = 0,
                 line_distances: Sequence[float] = ()) -> Optional[Tuple[float, float, float, float]]:
    """Endpoints of one guide line, in RECT_ECHO crop coordinates.

    Mirrors the branches of upDateLine: vertical, horizontal, and the tilted case where the
    original walks the line out to the sides of the rectangle.
    """
    if not setup.consistent():
        return None
    if not (0 <= depth_index < len(setup.depths) and 0 <= angle_index < len(setup.angles)):
        return None

    angle = setup.angles[angle_index]
    ratio_x = setup.pixel_ratio_x[depth_index]
    ratio_y = setup.pixel_ratio_y[depth_index]
    if ratio_x <= 0 or ratio_y <= 0:
        return None

    distance = centre_distance_mm(setup, depth_index, angle_index, line_index, line_distances)
    y_centre = round(distance / ratio_y) - 1
    x_centre = round(distance / ratio_x) - 1

    width = setup.rect_echo.width
    height = setup.rect_echo.height
    right = width - 1
    bottom = height - 1
    centre_x = (width - 1) / 2.0

    if angle == 90.0:
        return (x_centre, 0.0, x_centre, bottom)
    if angle == -90.0:
        return (x_centre, bottom, x_centre, 0.0)
    if angle == 0.0:
        return (0.0, y_centre, right, y_centre)
    if abs(angle) == 180.0:
        return (right, y_centre, 0.0, y_centre)

    # tilted: rise from the centre out to both sides, in millimetres, then back to pixels
    mm_left = (centre_x + 1) * ratio_x
    mm_right = (right - centre_x + 1) * ratio_x

    if 0 < angle < 90:
        rad = math.radians(angle)
        y_left = y_centre - round(mm_left * math.tan(rad) / ratio_y) + 1
        y_right = y_centre + round(mm_right * math.tan(rad) / ratio_y) - 1
        return (0.0, y_left, right, y_right)
    if 90 < angle < 180:
        rad = math.pi - math.radians(angle)
        y_left = y_centre + round(mm_left * math.tan(rad) / ratio_y) - 1
        y_right = y_centre - round(mm_right * math.tan(rad) / ratio_y) + 1
        return (right, y_right, 0.0, y_left)
    if -90 < angle < 0:
        rad = math.radians(-angle)
        y_left = y_centre + round(mm_left * math.tan(rad) / ratio_y) - 1
        y_right = y_centre - round(mm_right * math.tan(rad) / ratio_y) + 1
        return (0.0, y_left, right, y_right)
    # -180 < angle < -90
    rad = math.pi - math.radians(-angle)
    y_left = y_centre - round(mm_left * math.tan(rad) / ratio_y) + 1
    y_right = y_centre + round(mm_right * math.tan(rad) / ratio_y) - 1
    return (right, y_right, 0.0, y_left)


def clip_to_rect(segment: Tuple[float, float, float, float], width: int, height: int
                 ) -> Optional[Tuple[float, float, float, float]]:
    """Cut the line down to the rectangle, keeping its direction.

    `line_in_crop` builds the tilted cases by rising from the centre out to x=0 and x=right,
    so the endpoints often sit above or below the rectangle. The original clips them against
    the four sides (the QLineF::intersects cascade in upDateLine); this is that step, done
    with a parametric clip so the ends land on the border instead of outside it.
    """
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - 0.0), (dx, (width - 1) - x1),
                 (-dy, y1 - 0.0), (dy, (height - 1) - y1)):
        if p == 0:
            if q < 0:
                return None          # parallel and outside
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy)


def line_in_frame(setup: Setup, depth_index: int, angle_index: int, line_index: int = 0,
                  line_distances: Sequence[float] = ()) -> Optional[Tuple[float, float, float, float]]:
    """Same line, in full-frame coordinates."""
    local = line_in_crop(setup, depth_index, angle_index, line_index, line_distances)
    if local is None:
        return None
    clipped = clip_to_rect(local, setup.rect_echo.width, setup.rect_echo.height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    dx, dy = setup.rect_echo.left, setup.rect_echo.top
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def measure_from_points(setup: Setup, depth_index: int, p1: Tuple[float, float],
                        p2: Tuple[float, float], in_frame: bool = True) -> Optional[Tuple[float, float]]:
    """The inverse: two points traced on the needle -> (angle in degrees, centre distance in mm).

    This is what the calibration session produces; `line_in_crop` is its exact reverse, so a
    round trip through the two must land back on the same numbers.
    """
    if not setup.consistent() or not (0 <= depth_index < len(setup.depths)):
        return None
    ratio_x = setup.pixel_ratio_x[depth_index]
    ratio_y = setup.pixel_ratio_y[depth_index]
    if ratio_x <= 0 or ratio_y <= 0:
        return None

    x1, y1 = p1
    x2, y2 = p2
    if in_frame:
        x1 -= setup.rect_echo.left
        y1 -= setup.rect_echo.top
        x2 -= setup.rect_echo.left
        y2 -= setup.rect_echo.top

    # the angle lives in millimetre space, not pixel space: the two ratios differ
    dx_mm = (x2 - x1) * ratio_x
    dy_mm = (y2 - y1) * ratio_y
    if dx_mm == 0 and dy_mm == 0:
        return None
    angle = math.degrees(math.atan2(dy_mm, dx_mm))

    centre_x = (setup.rect_echo.width - 1) / 2.0
    if abs(abs(angle) - 90.0) < 1e-6:
        x_at_centre = (x1 + x2) / 2.0
        return angle, (x_at_centre + 1) * ratio_x - centre_x * ratio_x

    if x2 != x1:
        y_at_centre = y1 + (y2 - y1) * (centre_x - x1) / (x2 - x1)
    else:
        y_at_centre = (y1 + y2) / 2.0
    return angle, (y_at_centre + 1) * ratio_y


if __name__ == "__main__":  # quick self-check on one file
    import sys

    setup = read_setup(Path(sys.argv[1]))
    if setup is None:
        raise SystemExit("setup non leggibile")
    print(f"probe_type={setup.probe_type} kit={setup.kit_id} angoli={setup.angles}")
    print(f"rect={setup.rect_echo} depth={len(setup.depths)} coerente={setup.consistent()}")
    for d in range(min(3, len(setup.depths))):
        print(f"  depth {setup.depths[d]:6.1f} mm -> linea {line_in_frame(setup, d, 0)}")
