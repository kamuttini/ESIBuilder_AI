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


def merge_collinear(candidates, angle_tol: float = 6.0, offset_tol: float = 14.0,
                    gap_tol: float = 60.0) -> List[Refined]:
    """Rule 3: pieces of one needle become one segment.

    Two candidates merge when they point the same way, sit on the same line, and their ends are
    close enough that the space between them is a break in the detection rather than a gap
    between two different needles.
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


def reverberation_score(gray: np.ndarray, p1, p2, step: int = 7, repeats: int = 4,
                        samples: int = 96) -> float:
    """Rule 4: how much the line repeats itself below, which a reflection does not do.

    Samples the image on parallel lines at increasing distance on both sides and asks how many
    of them are still brighter than the background between them. Reverberation decays but stays
    banded; a reflection fades smoothly and scores near zero.
    """
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


def refine(candidates, gray: np.ndarray, max_needles: int = 2,
           parallel_tol: float = 8.0, min_reverberation: float = 0.0) -> List[Refined]:
    """Apply the four rules, best first."""
    if not candidates:
        return []
    merged = merge_collinear(candidates)
    for i, needle in enumerate(merged):
        merged[i] = replace(needle, reverberation=reverberation_score(gray, needle.p1, needle.p2))

    # a needle with reverberation beats a brighter thing without it
    merged.sort(key=lambda n: -(n.source_score * (1.0 + 2.0 * n.reverberation)))
    if min_reverberation > 0:
        kept = [n for n in merged if n.reverberation >= min_reverberation] or merged[:1]
    else:
        kept = merged

    out = [kept[0]]
    # rule 2: a second needle only if it is parallel to the first
    for needle in kept[1:]:
        if len(out) >= max_needles:
            break
        if _angle_gap(needle.angle_deg, out[0].angle_deg) <= parallel_tol:
            out.append(needle)
    return out
