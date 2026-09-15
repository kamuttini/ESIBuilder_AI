"""Turn raw line candidates into the one or two needles a calibration frame can contain.

Four rules, all from Camilla, all about what a needle is rather than about what scores high:

1. a frame holds one or two needles, never more;
2. when there are two, they are parallel -- they come from the same guide;
3. candidates that nearly touch and run the same way are one needle cut in pieces by the
   detector, and belong back together as a single segment;
4. a needle shows reverberation: bright repeats parallel to it, below it. A bright straight
   thing without them is usually a reflection in the water.

The detector's own score cannot express any of this: it judges each segment alone, so it
happily returns four pieces of the same needle, or a needle and a reflection with no way to
prefer one. These rules are what the operator uses by eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Refined:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    angle_deg: float
    reverberation: float
    merged_from: int
    source_score: float


def _angle(p1, p2) -> float:
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def _angle_gap(a: float, b: float) -> float:
    """Between directions, so opposite orientations of one line count as equal."""
    d = abs((a - b + 90) % 180 - 90)
    return d


def _point_line_distance(point, p1, p2) -> float:
    (x0, y0), (x1, y1), (x2, y2) = point, p1, p2
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return math.hypot(x0 - x1, y0 - y1)
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / norm


def _extent(points: Sequence[Tuple[float, float]], angle_deg: float):
    """The two endpoints of the segment spanning all points along `angle_deg`."""
    rad = math.radians(angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    projected = [(p[0] * ux + p[1] * uy, p) for p in points]
    projected.sort(key=lambda z: z[0])
    return projected[0][1], projected[-1][1]


def merge_collinear(candidates, angle_tol: float = 6.0, offset_tol: float = 24.0,
                    gap_tol: float = 180.0) -> List[Refined]:
    """Rule 3: pieces of one needle become one segment.

    Two candidates merge when they point the same way, sit on the same line, and their ends are
    close enough that the space between them is a break in the detection rather than a gap
    between two different needles.

    The tolerances are measured, not chosen. On the six pairs Camilla marked as one needle cut
    in two, the angles differ by at most 4.4 degrees, the perpendicular offset reaches 19.5
    pixels and the gap along the line reaches 162 -- against my originals of 6, 14 and 60, so
    two of the three were too tight to merge what she says belongs together.

    The perpendicular offset stays the strict one on purpose. Two real needles in a frame are
    parallel and near each other, so a generous gap along the line is safe -- far apart on the
    same line means one needle with a hole in the detection -- while a generous offset would
    start merging the two needles into one.
    """
    groups: List[List] = []
    for cand in sorted(candidates, key=lambda c: -getattr(c, "confidence", 0.0)):
        angle = _angle(cand.p1, cand.p2)
        placed = False
        for group in groups:
            ref = group[0]
            ref_angle = _angle(ref.p1, ref.p2)
            if _angle_gap(angle, ref_angle) > angle_tol:
                continue
            offset = max(_point_line_distance(cand.p1, ref.p1, ref.p2),
                         _point_line_distance(cand.p2, ref.p1, ref.p2))
            if offset > offset_tol:
                continue
            ends = [ref.p1, ref.p2, cand.p1, cand.p2]
            gap = min(math.hypot(a[0] - b[0], a[1] - b[1])
                      for a in (cand.p1, cand.p2) for b in (ref.p1, ref.p2))
            span = _extent(ends, ref_angle)
            length = math.hypot(span[1][0] - span[0][0], span[1][1] - span[0][1])
            overlaps = gap <= gap_tol or length <= max(
                math.hypot(ref.p2[0] - ref.p1[0], ref.p2[1] - ref.p1[1]),
                math.hypot(cand.p2[0] - cand.p1[0], cand.p2[1] - cand.p1[1])) + gap_tol
            if not overlaps:
                continue
            group.append(cand)
            placed = True
            break
        if not placed:
            groups.append([cand])

    out: List[Refined] = []
    for group in groups:
        points = [p for c in group for p in (c.p1, c.p2)]
        angle = _angle(group[0].p1, group[0].p2)
        p1, p2 = _extent(points, angle)
        out.append(Refined(p1=p1, p2=p2, angle_deg=_angle(p1, p2), reverberation=0.0,
                           merged_from=len(group),
                           source_score=max(getattr(c, "confidence", 0.0) for c in group)))
    return out


def line_brightness(gray: np.ndarray, p1, p2, samples: int = 128) -> float:
    """Median intensity along the line.

    The single strongest signal there is. Measured on Camilla's traced needles against segments
    that are certainly not hers: the needles sit at 94, the wrong ones at 2 -- the fallback
    detector searches every line of the frame and happily returns one through the black outside
    the sector, where nothing can be a needle.
    """
    values = _profile_along(gray, p1, p2, samples)
    return 0.0 if values is None else float(np.median(values))


def _profile_along(gray: np.ndarray, p1, p2, samples: int, offset: float = 0.0):
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 10:
        return None
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    t = np.linspace(0.0, 1.0, samples)
    xs = x1 + t * (x2 - x1) + nx * offset
    ys = y1 + t * (y2 - y1) + ny * offset
    h, w = gray.shape
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    return gray[yi, xi].astype(np.float32)


def ridge_score(gray: np.ndarray, p1, p2, offsets=(4, 8, 12, 16, 20, 24)) -> float:
    """Rule 4, measured the way that actually separates: how much the profile *varies* going down.

    A needle carries reverberation, so the brightness stepping away from it rises and falls in
    bands; a reflection in water fades smoothly. Counting how many bands are brighter than their
    neighbours -- the first attempt -- made precision worse. The spread of the differences
    between successive bands separates cleanly: 8.9 on the traced needles against 0.7 on
    segments that are certainly not needles.
    """
    levels = []
    for offset in offsets:
        values = _profile_along(gray, p1, p2, 96, offset=float(offset))
        if values is None:
            return 0.0
        levels.append(float(np.median(values)))
    return float(np.std(np.diff(levels)))


def reverberation_score(gray: np.ndarray, p1, p2, step: int = 7, repeats: int = 4,
                        samples: int = 96) -> float:
    """First attempt at rule 4, kept for reference. `ridge_score` replaced it: see its docstring."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 12:
        return 0.0
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    h, w = gray.shape
    t = np.linspace(0.0, 1.0, samples)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    def band(offset: float) -> float:
        xi = np.clip(np.round(xs + nx * offset).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys + ny * offset).astype(int), 0, h - 1)
        return float(np.median(gray[yi, xi].astype(np.float32)))

    best = 0.0
    for sign in (1.0, -1.0):
        hits = 0
        for k in range(1, repeats + 1):
            on = band(sign * step * k)
            between = 0.5 * (band(sign * step * (k - 0.5)) + band(sign * step * (k + 0.5)))
            if on - between > 3.0:
                hits += 1
        best = max(best, hits / float(repeats))
    return best


def _inclination(angle_deg: float) -> float:
    """Degrees away from horizontal, ignoring direction."""
    a = abs(angle_deg) % 180.0
    return min(a, 180.0 - a)


def refine(candidates, gray: np.ndarray, rect: Optional[Tuple[int, int, int, int]] = None,
           max_needles: int = 2, parallel_tol: float = 8.0, min_brightness: float = 30.0,
           min_ridge: float = 2.0, min_inclination: float = 8.0,
           max_start_depth: float = 0.55, min_vertical_span: float = 0.0) -> List[Refined]:
    """Apply the rules, best first.

    Two of them come from Camilla noticing that the detector prefers the wrong mark when
    several are present, and that the wrong ones are the flat ones low in the image while a
    needle enters from the top at a clear angle. Measured on 104 traced needles against 74 of
    the detector's own errors, that is not an impression:

      inclination from horizontal   needles 34.9 deg, errors 16.3, separation 1.07
      nearly flat (under 8 deg)     0 of 104 needles, 27 of 74 errors
      starting below mid-depth      3 of 104 needles, 28 of 74 errors

    A third candidate rule is available and OFF by default, and the reason is instructive. A
    needle crosses a third of the rectangle's height while an error crosses eight percent; under
    15% sit 17 of 181 needles against 81 of 150 errors, which as a single feature is the best
    trade of the lot. End to end on frames never used for tuning it makes things worse -- median
    error 3.48 degrees becomes 9.74 -- because when the detector has only found a short piece of
    the true needle, the cut removes it and the fallback picks something else entirely. A feature
    that separates well is not the same as a filter that helps, and only the end-to-end number
    can tell them apart.

    A fourth pattern is deliberately NOT a rule. 81% of needles descend to the right, against
    47% of errors, but the 19% that go left are not mistakes: they come from five specific
    acquisitions, presumably mounted or flipped the other way. Direction is a property of the
    acquisition, not of needles, so it belongs in a consistency check within a folder rather
    than in a filter that would throw away every needle of those five.

    So a flat candidate is discarded outright, and one that begins in the lower half is too --
    the first costs no needle at all, the second costs three in a hundred and removes more than
    a third of the errors.
    """
    if not candidates:
        return []
    merged = merge_collinear(candidates)
    if min_inclination > 0:
        dritti = [n for n in merged if _inclination(n.angle_deg) >= min_inclination]
        merged = dritti or merged
    if rect is not None and max_start_depth < 1.0:
        top, height = rect[1], max(1, rect[3] - rect[1])
        alti = [n for n in merged
                if (min(n.p1[1], n.p2[1]) - top) / height <= max_start_depth]
        merged = alti or merged
    if rect is not None and min_vertical_span > 0:
        height = max(1, rect[3] - rect[1])
        lunghi = [n for n in merged
                  if abs(n.p2[1] - n.p1[1]) / height >= min_vertical_span]
        merged = lunghi or merged
    scored = []
    for needle in merged:
        brightness = line_brightness(gray, needle.p1, needle.p2)
        ridge = ridge_score(gray, needle.p1, needle.p2)
        scored.append((replace(needle, reverberation=ridge), brightness, ridge))

    # a dark line is not a needle whatever else it has going for it
    kept = [(n, b, r) for n, b, r in scored if b >= min_brightness and r >= min_ridge]
    if not kept:
        kept = sorted(scored, key=lambda z: -z[1])[:1]

    kept.sort(key=lambda z: -(z[1] * (1.0 + z[2])))
    kept = [n for n, _, _ in kept]

    out = [kept[0]]
    # rule 2: a second needle only if it is parallel to the first
    for needle in kept[1:]:
        if len(out) >= max_needles:
            break
        if _angle_gap(needle.angle_deg, out[0].angle_deg) <= parallel_tol:
            out.append(needle)
    return out
