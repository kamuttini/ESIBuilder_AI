#!/usr/bin/env python3
"""Consolidate per-image scale predictions into a coherent per-setup answer.

Why this stage exists
---------------------
The detector looks at one frame at a time, and on a single frame a wrong calibration is
invisible: nothing in the image contradicts it. Measured on BK, 9 of 27 ``accepted``
predictions were off by 30–519% on ``mm_per_px`` while looking perfectly self-consistent.

A configuration *folder*, though, is highly constrained. Measured on the 5210 clean GT
rows of ``SSD_esi1_n1`` (2026-07-29, setups with ≥ 4 depth indices and a homogeneous
direction, n = 371):

| quantity | behaviour across the depth indices of one setup |
|---|---|
| direction (zero at top/bottom) | homogeneous in **371 / 376** setups |
| ``mm_per_px`` | monotonically increasing with the depth index in **345 / 376 (91.8%)** |
| ``y_zero`` | median step between adjacent depths **1.0 px** (p90 15 px) |
| ``x`` | median step between adjacent depths **2.0 px** (p90 19 px) |

Note what is *not* true: ``x`` is monotone in only 22% of setups and ``y_zero`` in 68%,
and the scale genuinely moves — it gets shorter and the zero shifts as the depth changes.
So this is deliberately **not** a "one value per folder" model. It is local coherence:
each depth index may drift from its neighbours by a couple of pixels, and anything that
jumps far from them is the outlier, not the trend.

What the stage does
-------------------
1. **Direction** by majority vote; candidates disagreeing with it are dropped.
2. **``mm_per_px``** by robust isotonic regression (pool-adjacent-violators) after a
   local-median outlier filter. This is the step that kills the 30–519% errors.
3. **``x`` and ``y_zero``** by local-median filtering along the depth sequence.
4. **Gap filling**: depth indices where no ladder was found inherit ``x`` / ``y_zero``
   from their neighbours and ``mm_per_px`` from the isotonic curve, and are marked
   ``interpolated`` so a reviewer can see they were not observed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #

# A candidate whose mm_per_px differs from the local median by more than this is an
# outlier. 25% is far wider than the real depth-to-depth change between adjacent
# indices and far narrower than the 30%+ failures we need to catch.
# Tolerance on the relative residual from the robust log-linear trend of mm_per_px vs
# depth index. Measured on the 5210 clean GT rows: residual median 3.9%, p90 12.1%,
# p99 30.7%, max 51.7%. 0.35 therefore keeps ~99% of genuine rows.
#
# Note what does NOT work, because it was tried: comparing against the median of the
# whole setup. mm_per_px legitimately spans a factor 4.6 (median) within one setup — the
# depth goes from 20 to 80 mm — so any global factor either passes the 30% errors or
# rejects most good rows. Between *adjacent* depth indices, though, the ratio sits in
# [0.85, 1.35] 99% of the time and essentially never decreases (p01 = 0.998). The trend,
# not the level, is the invariant.
TS_RESIDUAL_TOL = 0.35
# Beyond this the raw value is replaced by the fitted trend rather than emitted as-is:
# 12.1% is the p90 of the genuine residual, so a row further out than that is better
# described by its folder than by its own frame. Without this, a 31% error still sits
# inside TS_RESIDUAL_TOL and would be accepted verbatim.
MM_PER_PX_SNAP_TOL = 0.12

# Positional tolerances vs the robust trend, in px. From the same measurement:
# x residual p90 18, p99 59; y_zero residual p90 20, p99 91. The scale genuinely moves
# and shortens with depth, so these have to be generous — but the failures we need to
# catch are hundreds of px (671 px observed), not tens.
X_TOL_PX = 60.0
Y_TOL_PX = 90.0
# Window (in depth indices, each side) used for the local medians.
LOCAL_WINDOW = 2
# Below this many usable candidates a setup cannot vote on itself.
MIN_CANDIDATES = 3
# Physically plausible calibration for an ultrasound frame.
MM_PER_PX_MIN = 0.005
MM_PER_PX_MAX = 0.60


@dataclass
class ScaleCandidate:
    """One per-image detector output, as fed into the consensus."""

    depth_index: int
    x: Optional[float] = None
    y_zero: Optional[float] = None
    mm_per_px: Optional[float] = None
    direction: int = 1
    confidence: float = 0.0
    status: str = "reject"
    n_labels: int = 0
    # A weak anchor still gets a value and can be corrected/filled, but must not define the
    # robust trend: e.g. a single-label geometry calibration, which fixes coverage but can
    # pick the wrong 5-vs-10 mm step and would drag the Theil-Sen fit if it anchored it.
    weak_anchor: bool = False

    @property
    def usable(self) -> bool:
        return (
            self.x is not None
            and self.y_zero is not None
            and self.mm_per_px is not None
            and MM_PER_PX_MIN <= self.mm_per_px <= MM_PER_PX_MAX
        )


@dataclass
class ScaleConsensus:
    """The consolidated answer for one depth index of one setup."""

    depth_index: int
    status: str  # accepted | review | reject
    source: str  # detected | corrected | interpolated | none
    x: Optional[float] = None
    y_zero: Optional[float] = None
    mm_per_px: Optional[float] = None
    direction: int = 1
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _isotonic(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> List[float]:
    """Pool-adjacent-violators: nearest non-decreasing sequence in weighted L2.

    Written out rather than pulled from scikit-learn to keep the scale block free of a
    heavy dependency it does not otherwise need (relevant for the Windows deploy).
    """
    n = len(values)
    if n == 0:
        return []
    w = list(weights) if weights is not None else [1.0] * n
    # Each block: (weighted mean, total weight, length)
    blocks: List[Tuple[float, float, int]] = []
    for v, wi in zip(values, w):
        blocks.append((float(v), float(wi), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, l2 = blocks.pop()
            m1, w1, l1 = blocks.pop()
            tot = w1 + w2
            blocks.append(((m1 * w1 + m2 * w2) / tot if tot else m1, tot, l1 + l2))
    out: List[float] = []
    for mean, _, length in blocks:
        out.extend([mean] * length)
    return out


def _theil_sen_log(points: Sequence[Tuple[int, float]]):
    """Robust log-linear trend of a positive quantity vs depth index.

    Returns a callable ``index -> expected value``, or None when there is not enough to
    fit. Theil-Sen takes the median of all pairwise slopes, so a minority of grossly
    wrong points cannot drag the line.
    """
    pts = [(i, v) for i, v in points if v and v > 0]
    if len(pts) < 3:
        return None
    xs = [float(i) for i, _ in pts]
    ys = [math.log(v) for _, v in pts]
    slopes = [
        (ys[b] - ys[a]) / (xs[b] - xs[a])
        for a in range(len(pts))
        for b in range(a + 1, len(pts))
        if xs[b] != xs[a]
    ]
    if not slopes:
        return None
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys)])
    return lambda idx: math.exp(slope * float(idx) + intercept)


def _theil_sen_linear(points: Sequence[Tuple[int, float]]):
    """Robust linear trend vs depth index. Returns ``index -> expected`` or None."""
    pts = list(points)
    if len(pts) < 3:
        return None
    xs = [float(i) for i, _ in pts]
    ys = [float(v) for _, v in pts]
    slopes = [
        (ys[b] - ys[a]) / (xs[b] - xs[a])
        for a in range(len(pts))
        for b in range(a + 1, len(pts))
        if xs[b] != xs[a]
    ]
    if not slopes:
        return None
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys)])
    return lambda idx: slope * float(idx) + intercept


def _local_median(
    by_index: Dict[int, float],
    index: int,
    window: int = LOCAL_WINDOW,
    exclude_self: bool = True,
) -> Optional[float]:
    """Median of a quantity over the neighbouring depth indices."""
    vals = [
        v
        for i, v in by_index.items()
        if abs(i - index) <= window and not (exclude_self and i == index)
    ]
    if not vals:
        return None
    return statistics.median(vals)


def _interpolate(by_index: Dict[int, float], index: int) -> Optional[float]:
    """Linear interpolation between the nearest observed neighbours on each side."""
    if not by_index:
        return None
    lower = [i for i in by_index if i < index]
    upper = [i for i in by_index if i > index]
    if lower and upper:
        a, b = max(lower), min(upper)
        t = (index - a) / (b - a)
        return by_index[a] + t * (by_index[b] - by_index[a])
    nearest = min(by_index, key=lambda i: abs(i - index))
    return by_index[nearest]


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def consolidate_setup(
    candidates: Sequence[ScaleCandidate],
    depth_indices: Optional[Sequence[int]] = None,
) -> List[ScaleConsensus]:
    """Turn per-image candidates into a coherent per-setup answer.

    ``depth_indices`` lets the caller ask for indices that produced no candidate at all,
    so gaps come back as ``interpolated`` rows instead of silently disappearing.
    """
    indices = sorted(set(depth_indices) if depth_indices else {c.depth_index for c in candidates})
    by_idx = {c.depth_index: c for c in candidates}
    usable = [c for c in candidates if c.usable]

    if len(usable) < MIN_CANDIDATES:
        # Not enough evidence to vote: pass the raw candidates through as review.
        return [
            ScaleConsensus(
                depth_index=i,
                status="review" if (i in by_idx and by_idx[i].usable) else "reject",
                source="detected" if (i in by_idx and by_idx[i].usable) else "none",
                x=by_idx[i].x if i in by_idx else None,
                y_zero=by_idx[i].y_zero if i in by_idx else None,
                mm_per_px=by_idx[i].mm_per_px if i in by_idx else None,
                direction=by_idx[i].direction if i in by_idx else 1,
                confidence=by_idx[i].confidence if i in by_idx else 0.0,
                notes=["setup_too_small_for_consensus"],
            )
            for i in indices
        ]

    # --- 1. direction by majority vote -------------------------------------
    n_voters = len(usable)
    down = sum(c.confidence + 0.1 for c in usable if c.direction > 0)
    up = sum(c.confidence + 0.1 for c in usable if c.direction < 0)
    direction = 1 if down >= up else -1
    n_agree = sum(1 for c in usable if c.direction == direction)
    dir_unanimous = n_agree == n_voters
    usable = [c for c in usable if c.direction == direction]

    # --- 2. mm_per_px: robust log-linear trend, then isotonic fit -----------
    # Theil-Sen (median of pairwise slopes) on log(mm_per_px) vs depth index: it has a
    # real breakdown point, which matters because the outliers we must survive were ~1/3
    # of the accepted rows on BK.
    #
    # Fit the trend on the *strong* anchors only. Weak ones (single-label geometry) still
    # get corrected/filled from the trend below, but must not define it: on BK they picked
    # the wrong 5-vs-10 mm step and, when allowed to anchor, dragged the fit and promoted
    # their own wrong value (accepted quality 98%->89%). Excluding them is a no-op when
    # there are none (the plain baseline), so this cannot regress the label-only path.
    anchors = [c for c in usable if not c.weak_anchor]
    if len(anchors) < 3:
        anchors = usable  # too few strong anchors to fit: fall back to everything
    trend = _theil_sen_log(
        [(c.depth_index, float(c.mm_per_px)) for c in anchors]  # type: ignore[arg-type]
    )
    outlier_idx: set[int] = set()
    inliers: List[ScaleCandidate] = []
    for c in usable:
        expected = trend(c.depth_index) if trend else None
        if expected and expected > 0:
            if abs(float(c.mm_per_px) / expected - 1.0) > TS_RESIDUAL_TOL:  # type: ignore[arg-type]
                outlier_idx.add(c.depth_index)
                continue
        inliers.append(c)

    if len(inliers) < 2:  # the filter ate everything: keep the raw set for review
        inliers, outlier_idx = list(usable), set()

    inliers.sort(key=lambda c: c.depth_index)
    fitted = _isotonic(
        [float(c.mm_per_px) for c in inliers],  # type: ignore[arg-type]
        [max(0.05, c.confidence) for c in inliers],
    )
    mm_fit = {c.depth_index: v for c, v in zip(inliers, fitted)}
    # Refit the trend on the inliers only, so gaps are filled from a clean curve.
    trend_clean = _theil_sen_log([(i, v) for i, v in mm_fit.items()]) or trend

    def _mm_for_gap(idx: int) -> Optional[float]:
        if trend_clean is not None:
            return trend_clean(idx)
        return _interpolate(mm_fit, idx)

    # --- 3. x and y_zero: robust linear trend ------------------------------
    # A local median was tried here first and does not work: one bad y_zero sits inside
    # its own neighbours' window and drags their median, so a single failure condemns the
    # rows either side of it. Theil-Sen looks at all pairs, so the bad row is outvoted.
    x_by_idx = {c.depth_index: float(c.x) for c in inliers}  # type: ignore[arg-type]
    y_by_idx = {c.depth_index: float(c.y_zero) for c in inliers}  # type: ignore[arg-type]
    x_trend = _theil_sen_linear(sorted(x_by_idx.items()))
    y_trend = _theil_sen_linear(sorted(y_by_idx.items()))

    def _cleaned(
        src: Dict[int, float],
        idx: int,
        trend_fn,
        tol: float,
    ) -> Tuple[Optional[float], bool]:
        """Value at idx, replaced by the fitted trend when it jumps away from it."""
        if idx not in src:
            return None, False
        if trend_fn is None:
            ref = _local_median(src, idx)
            if ref is not None and abs(src[idx] - ref) > tol:
                return ref, True
            return src[idx], False
        expected = trend_fn(idx)
        if abs(src[idx] - expected) > tol:
            return expected, True
        return src[idx], False

    # --- 4. assemble, filling the gaps -------------------------------------
    out: List[ScaleConsensus] = []
    for i in indices:
        notes: List[str] = []
        if not dir_unanimous:
            notes.append(f"direction_vote_{n_agree}/{n_voters}")

        if i in mm_fit:
            x_val, x_fixed = _cleaned(x_by_idx, i, x_trend, X_TOL_PX)
            y_val, y_fixed = _cleaned(y_by_idx, i, y_trend, Y_TOL_PX)
            raw = by_idx[i]
            mm_val = mm_fit[i]
            mm_fixed = False
            if trend_clean is not None:
                expected = trend_clean(i)
                if expected > 0 and abs(mm_val / expected - 1.0) > MM_PER_PX_SNAP_TOL:
                    mm_val, mm_fixed = expected, True
            drift = (
                abs(mm_val - float(raw.mm_per_px)) / mm_val if mm_val else 0.0  # type: ignore[arg-type]
            )
            source = "detected"
            if x_fixed or y_fixed or mm_fixed or drift > 0.01:
                source = "corrected"
                if x_fixed:
                    notes.append("x_snapped_to_setup_trend")
                if y_fixed:
                    notes.append("y_zero_snapped_to_setup_trend")
                if mm_fixed:
                    notes.append("mm_per_px_snapped_to_setup_trend")
                if drift > 0.01:
                    notes.append(f"mm_per_px_shift_{drift:.1%}")
            # Agreeing with the setup direction is not by itself a licence to auto-accept: a
            # weak-evidence row (single-label geometry) can sit inside the trend tolerance and
            # would be promoted on direction alone. Measured on BK that added 9 accepted rows
            # of which 6 had the wrong step (accepted quality 98%->89%). Weak rows stay in
            # review — still corrected, still filling coverage, but confirmed by the operator.
            status = (
                "accepted"
                if (dir_unanimous or raw.direction == direction) and not raw.weak_anchor
                else "review"
            )
            confidence = raw.confidence
        elif i in outlier_idx:
            # Detected but incoherent with its own folder: this is the case the
            # per-image policy used to wave through.
            x_val = _interpolate(x_by_idx, i)
            y_val = _interpolate(y_by_idx, i)
            mm_val = _mm_for_gap(i)
            source = "interpolated"
            status = "review"
            notes.append("rejected_as_incoherent_with_setup")
            confidence = 0.3
        else:
            x_val = _interpolate(x_by_idx, i)
            y_val = _interpolate(y_by_idx, i)
            mm_val = _mm_for_gap(i)
            source = "interpolated" if mm_val is not None else "none"
            status = "review" if mm_val is not None else "reject"
            notes.append("no_detection_filled_from_neighbours")
            confidence = 0.25

        out.append(
            ScaleConsensus(
                depth_index=i,
                status=status,
                source=source,
                x=x_val,
                y_zero=y_val,
                mm_per_px=mm_val,
                direction=direction,
                confidence=confidence,
                notes=notes,
            )
        )
    return out
