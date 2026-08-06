#!/usr/bin/env python3
"""Deterministic scale detector: tick ladder first, OCR numbers for calibration.

Rationale (see docs/scala_strategia_per_vendor_2026-07-29.md)

The scale a scanner overlays is a *ruler*: a column of short, equally spaced bright
dashes, with numeric labels every N dashes and often a unit ("cm") next to the zero.
That is a strongly structured object, so we detect it instead of regressing it:

  stage 1  tick candidates   short bright horizontal dashes (connected components)
  stage 2  ladder grouping   dashes sharing an x, y positions fitting y0 + k*pitch
  stage 3  OCR labels        read the numbers beside the ladder
  stage 4  calibration       linear fit value_mm = mm_per_px * (y - y_zero)
  stage 5  confidence        consensus of the fit -> accepted / review / reject

Stage 4 is the point of the whole design: two correctly-read labels at two known y
positions give ``mm_per_px`` and ``y_zero`` directly, which is exactly what .fss
line 21 (and lines 19/20) encode. No depth value and no learned prior needed.

Output is a ``ScalePrediction``; the ``.fss`` line 21 segment is then
``x | x | y_zero | y_zero + length_mm / mm_per_px | length_mm | 0.5 | -1``.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# tunables (a ScaleProfile can override any of them per vendor)
# --------------------------------------------------------------------------- #


@dataclass
class ScaleProfile:
    """Per-vendor priors. Defaults are the vendor-agnostic starting point."""

    name: str = "default"

    # stage 1 - what a tick dash looks like, in px
    tick_min_w: int = 3
    tick_max_w: int = 34
    tick_min_h: int = 1
    tick_max_h: int = 9
    bright_percentile: float = 99.0
    bright_min_value: int = 120

    # stage 2 - what a ladder looks like
    x_tolerance: int = 4  # dashes within this many px share a column
    min_ticks: int = 5
    min_pitch: float = 8.0
    max_pitch: float = 220.0
    pitch_residual_max: float = 2.5  # px, median abs residual of the y0 + k*pitch fit

    # search bands, as a fraction of image width, measured from the rect edge
    # (positive = away from the rect interior). None = search the full width.
    right_band_px: Tuple[int, int] = (-60, 420)
    left_band_px: Tuple[int, int] = (-60, 420)
    prefer_side: str = "right"
    side_bonus: float = 0.12

    # stage 3 - OCR
    label_strip_w: int = 140  # how far from the ladder the numbers can sit
    label_strip_gap: int = 4  # keep the ticks themselves out of the OCR strip
    ocr_upscale: int = 3
    # psm 6 reads the label column best; psm 11 (sparse text) is tried only when 6 came
    # back with too little to calibrate, because OCR dominates the runtime.
    ocr_psms: Tuple[int, ...] = (6,)
    ocr_psms_fallback: Tuple[int, ...] = (11,)
    ocr_min_conf: float = 25.0
    label_snap_cm: float = 0.5  # printed labels are multiples of this
    label_snap_tol_cm: float = 0.16
    label_max_cm: float = 40.0

    # stage 5 - acceptance
    min_labels_for_accept: int = 3
    calib_residual_max_mm: float = 0.9
    # With only two labels the linear fit is exact and therefore unverifiable, so we
    # instead demand that the ladder pitch corroborate them: mm_per_px * pitch must
    # land on a tick step a scanner would actually print.
    plausible_tick_mm: Tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0, 20.0)
    tick_step_rel_tol: float = 0.08


PROFILES: Dict[str, ScaleProfile] = {
    "default": ScaleProfile(),
    # Filled in from artifacts/37_scale_gt_audit_20260729/scale_gt_profiles.json as
    # each vendor is validated; keeping them explicit documents what we measured.
    "BK": ScaleProfile(name="BK", right_band_px=(-80, 320), min_pitch=20.0),
    "Esaote": ScaleProfile(name="Esaote", right_band_px=(-60, 520)),
    "Hitachi": ScaleProfile(name="Hitachi", right_band_px=(-60, 1000)),
    "GE": ScaleProfile(name="GE", right_band_px=(-60, 950)),
    "Canon": ScaleProfile(name="Canon", right_band_px=(-60, 800)),
    "Mindray": ScaleProfile(name="Mindray", right_band_px=(-60, 750)),
    "Alpinion": ScaleProfile(name="Alpinion", prefer_side="left", left_band_px=(-80, 200)),
}


def profile_for(vendor: str) -> ScaleProfile:
    return PROFILES.get(vendor, PROFILES["default"])


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class TickLadder:
    x: float
    ticks_y: List[float]
    pitch: float
    y_first: float
    residual: float
    score: float
    side: str
    n_ticks: int


@dataclass
class ScalePrediction:
    ok: bool
    status: str  # accepted | review | reject
    reason: str = ""
    x: Optional[float] = None
    y_zero: Optional[float] = None
    mm_per_px: Optional[float] = None
    tick_pitch_px: Optional[float] = None
    tick_step_mm: Optional[float] = None
    y_last_tick: Optional[float] = None  # bottom-most detected tick
    y_first_tick: Optional[float] = None  # top-most detected tick
    ticks_y: List[float] = field(default_factory=list)  # every tick of the chosen ladder
    # +1 = values grow downwards (zero at the top), -1 = zero at the bottom (UD flip)
    direction: int = 1
    span_mm: Optional[float] = None
    labels: List[Tuple[float, float]] = field(default_factory=list)  # (y, value_mm) used for calib
    labels_all: List[Tuple[float, float]] = field(default_factory=list)  # (y, value_mm) every OCR read
    label_side: Optional[str] = None
    calib_residual_mm: Optional[float] = None
    confidence: float = 0.0
    ladder_score: float = 0.0
    n_ticks: int = 0
    debug: Dict[str, object] = field(default_factory=dict)

    def fss_segment(self, length_mm: Optional[float] = None) -> Optional[Dict[str, float]]:
        """The .fss line-21 group implied by this prediction."""
        if not self.ok or self.mm_per_px is None or self.y_zero is None or self.x is None:
            return None
        if length_mm is None:
            length_mm = self.default_length_mm()
        if length_mm is None or length_mm <= 0:
            return None
        return {
            "x1": round(self.x),
            "x2": round(self.x),
            # y1 carries the zero, y2 the far end — the order encodes the direction,
            # exactly as the legacy files do.
            "y1": round(self.y_zero),
            "y2": round(self.y_zero + self.direction * length_mm / self.mm_per_px),
            "length_mm": length_mm,
            "tick": 0.5,
            "side": -1,
        }

    def default_length_mm(self) -> Optional[float]:
        """Extend the segment to the deepest tick, snapped to the tick grid.

        The old convention floored ``span_mm`` to a whole centimetre. Because the ticks
        sit at 0.5 cm, that structurally left the half-centimetre tick beyond the last
        whole-cm mark *outside* the drawn segment: measured on BK/Esaote, ~1 tick fell
        out in 62-85% of rows, and the gap was exactly the floor loss (median 4.6-6.7 mm)
        — detection already reached the ruler end. Snapping to the tick step instead lands
        the far end on the deepest tick. length_mm/y2 do not enter strict_ok or the
        calibration, so this only changes coverage, not correctness.
        """
        if self.span_mm is None or self.span_mm <= 0:
            return None
        step = self.tick_step_mm if (self.tick_step_mm and self.tick_step_mm > 0) else 5.0
        return max(1, round(self.span_mm / step)) * step


# --------------------------------------------------------------------------- #
# stage 1 - tick candidates
# --------------------------------------------------------------------------- #


def _bright_mask(gray: np.ndarray, prof: ScaleProfile) -> np.ndarray:
    """Threshold locally, because the scale is often much dimmer than the anatomy.

    A single global threshold fails on the whole BK FlexFocus/Profocus family, whose
    ruler is drawn in dark grey while the ultrasound content is bright: the 99th
    percentile of the frame lands well above the ticks. Inside a lateral band the
    ticks *are* the brightest thing, so the percentile is taken per band by the caller
    and only a small absolute floor is enforced here.
    """
    finite = gray[gray > 0]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    thr = float(np.percentile(finite, prof.bright_percentile))
    thr = max(thr, float(prof.bright_min_value))
    thr = min(thr, float(gray.max()) - 1.0)
    return (gray >= thr).astype(np.uint8)


def _tophat_mask(gray: np.ndarray, prof: ScaleProfile) -> Optional[np.ndarray]:
    """A morphological tick mask that does not depend on absolute brightness.

    The percentile bright mask misses the faint dark-grey rulers (BK FlexFocus/Profocus)
    and the deep end of long rulers, where each tick fades below the frame percentile.
    A CLAHE + horizontal top-hat pops every short bright dash regardless of its absolute
    level — the technique the depth block uses to read scale labels on textured
    backgrounds (``rect_depth_hybrid.py`` ``scale_tophat``). Returned as an extra source
    of candidates; the dash-shape filter and the grid fit still reject spurious blobs.
    """
    # High-recall tick source (top-hat) is opt-in: it lifts ladder-found and tick recall
    # sharply, but the extra minor ticks currently mis-snap OCR labels and regress
    # calibration (see docs). Kept behind a flag until the label->tick assignment is made
    # robust; the correction tool turns it on to surface more rulers for human review.
    if cv2 is None or gray.size == 0 or not os.environ.get("SCALE_HIGH_RECALL"):
        return None
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        kw = max(9, int(prof.tick_max_w) + 8)
        kh = max(5, int(prof.tick_max_h) + 4)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        th = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, kern)
        th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
        binm = cv2.adaptiveThreshold(
            th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -3
        )
        return (binm > 0).astype(np.uint8)
    except cv2.error:
        return None


def _detach_tall(mask: np.ndarray) -> np.ndarray:
    """Remove tall vertical structures (the drawn scale line) so attached ticks split off."""
    tall = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21)))
    tall = cv2.dilate(tall, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return cv2.subtract(mask, tall)


def find_tick_dashes(
    gray: np.ndarray,
    prof: ScaleProfile,
    x_offset: int = 0,
) -> List[Tuple[float, float, int, int]]:
    """Return ``(cx, cy, w, h)`` of every component shaped like a tick dash.

    Candidate masks are unioned: the raw bright mask (ticks alone in the black margin),
    a CLAHE+top-hat mask (faint dark-grey ticks the percentile misses), and for each the
    tall-structure-removed variant (ticks attached to a drawn scale line, which would
    otherwise merge into one long component). Extra candidates only raise recall — the
    dash-shape filter here and the uniform-grid fit downstream drop the spurious ones.
    """
    mask = _bright_mask(gray, prof)
    sources = [mask, _detach_tall(mask)]
    tophat = _tophat_mask(gray, prof)
    if tophat is not None:
        sources += [tophat, _detach_tall(tophat)]

    out: List[Tuple[float, float, int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for m in sources:
        n, _, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (prof.tick_min_w <= w <= prof.tick_max_w):
                continue
            if not (prof.tick_min_h <= h <= prof.tick_max_h):
                continue
            if area < 0.4 * w * h:  # must be a solid dash, not a diagonal wisp
                continue
            cx, cy = float(centroids[i][0]) + x_offset, float(centroids[i][1])
            key = (int(round(cx)), int(round(cy)))
            if key in seen:
                continue
            seen.add(key)
            out.append((cx, cy, int(w), int(h)))
    return out


# --------------------------------------------------------------------------- #
# stage 2 - ladder grouping
# --------------------------------------------------------------------------- #


def _fit_uniform_grid(ys: Sequence[float], prof: ScaleProfile) -> Optional[Tuple[float, float, float, int]]:
    """Fit ``y_k = y0 + k * pitch`` to ys, tolerating missing and spurious ticks.

    Returns ``(y0, pitch, median_residual, n_used)``.
    """
    ys = sorted(ys)
    if len(ys) < prof.min_ticks:
        return None
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a >= prof.min_pitch * 0.5]
    if not diffs:
        return None

    # Candidate pitches: the smallest gaps are the minor-tick step; also try
    # dividing larger gaps in case minor ticks are missing in places.
    cands: List[float] = []
    base = float(np.median(diffs))
    for d in (base, float(np.percentile(diffs, 25)), min(diffs)):
        for k in (1, 2, 3):
            p = d / k
            if prof.min_pitch <= p <= prof.max_pitch:
                cands.append(p)
    if not cands:
        return None

    best: Optional[Tuple[float, float, float, int]] = None
    best_score = -1e9
    # Largest pitch first: a submultiple of the true pitch fits the same ticks just as
    # well but claims ticks that are not there, so it must not win on a tie.
    for pitch in sorted(set(round(c, 3) for c in cands), reverse=True):
        # phase from the residuals of all ys modulo pitch (circular mean)
        ang = 2 * math.pi * (np.asarray(ys) % pitch) / pitch
        phase = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        offset = (phase / (2 * math.pi)) * pitch
        ks = np.round((np.asarray(ys) - offset) / pitch)
        fitted = offset + ks * pitch
        resid = np.abs(np.asarray(ys) - fitted)
        keep = resid <= prof.pitch_residual_max
        n_used = int(keep.sum())
        if n_used < prof.min_ticks:
            continue
        med = float(np.median(resid[keep])) if n_used else 99.0
        # Coverage is what separates the real pitch from a submultiple of it. A grid
        # three times too fine fits the same ticks with a *smaller* residual — every y
        # is closer to some finer grid point — but leaves two thirds of its slots empty.
        # Measured on BK + Esaote, gt_pitch/pred_pitch was an exact 2 or 3 in 29 of 33
        # correctly localised rows before this term was added.
        used_ys = np.asarray(ys)[keep]
        span = float(used_ys.max() - used_ys.min())
        slots = span / pitch + 1.0
        coverage = n_used / slots if slots > 0 else 0.0
        score = n_used * min(1.0, coverage) - med
        if score > best_score:
            best_score = score
            y0 = float(np.min(fitted[keep]))
            best = (y0, pitch, med, n_used)
    return best


def group_ladders(
    dashes: Sequence[Tuple[float, float, int, int]],
    gray_shape: Tuple[int, int],
    prof: ScaleProfile,
    bands: Sequence[Tuple[int, int, str]],
) -> List[TickLadder]:
    """Cluster dashes into vertical ladders, one per candidate x column."""
    ladders: List[TickLadder] = []
    h, w = gray_shape
    for x_lo, x_hi, side in bands:
        in_band = [d for d in dashes if x_lo <= d[0] <= x_hi]
        if len(in_band) < prof.min_ticks:
            continue
        # cluster by x (dashes of one ruler share a right edge / centre)
        by_x: Dict[int, List[Tuple[float, float, int, int]]] = {}
        for d in sorted(in_band, key=lambda t: t[0]):
            placed = False
            for key in list(by_x):
                if abs(key - d[0]) <= prof.x_tolerance:
                    by_x[key].append(d)
                    placed = True
                    break
            if not placed:
                by_x[int(round(d[0]))] = [d]

        for key, group in by_x.items():
            if len(group) < prof.min_ticks:
                continue
            fit = _fit_uniform_grid([g[1] for g in group], prof)
            if fit is None:
                continue
            y0, pitch, resid, n_used = fit
            ys = sorted(g[1] for g in group)
            span = ys[-1] - ys[0]
            if span <= 0:
                continue
            coverage = n_used / max(1.0, span / pitch + 1.0)
            score = (
                0.5 * min(1.0, n_used / 12.0)
                + 0.3 * min(1.0, coverage)
                + 0.2 * max(0.0, 1.0 - resid / prof.pitch_residual_max)
            )
            if side == prof.prefer_side:
                score += prof.side_bonus
            ladders.append(
                TickLadder(
                    x=float(np.median([g[0] for g in group])),
                    ticks_y=ys,
                    pitch=pitch,
                    y_first=y0,
                    residual=resid,
                    score=score,
                    side=side,
                    n_ticks=n_used,
                )
            )
    ladders.sort(key=lambda l: -l.score)
    return ladders


# --------------------------------------------------------------------------- #
# stage 3 - OCR the labels
# --------------------------------------------------------------------------- #

_OCR_CONFIG = "--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789."


def _parse_number(text: str) -> Optional[float]:
    t = text.strip().replace(",", ".").replace(" ", "")
    if not t:
        return None
    # tesseract loves stray dots; keep the first plausible number
    buf = ""
    for ch in t:
        if ch.isdigit() or (ch == "." and "." not in buf and buf):
            buf += ch
        elif buf:
            break
    buf = buf.rstrip(".")
    if not buf:
        return None
    try:
        return float(buf)
    except ValueError:
        return None


def _snap_label(value: float, prof: ScaleProfile) -> Optional[float]:
    """Printed labels are multiples of the tick step; snap or drop.

    This is what turns "2.1" (a misread of "2.0") into a usable observation and
    throws away OCR hallucinations that land nowhere near a valid label.
    """
    if value < 0 or value > prof.label_max_cm:
        return None
    step = prof.label_snap_cm
    snapped = round(value / step) * step
    if abs(snapped - value) > prof.label_snap_tol_cm:
        return None
    return round(snapped, 3)


def _ocr_strip(
    gray: np.ndarray,
    prof: ScaleProfile,
    psms: Optional[Sequence[int]] = None,
) -> List[Tuple[float, float, float]]:
    """OCR one vertical strip in a single call. Returns ``[(y_in_strip, value, conf)]``."""
    import pytesseract

    if gray.size == 0 or gray.shape[0] < 8 or gray.shape[1] < 8:
        return []
    psms = prof.ocr_psms if psms is None else psms
    up_gray = cv2.resize(
        gray, None, fx=prof.ocr_upscale, fy=prof.ocr_upscale, interpolation=cv2.INTER_CUBIC
    )

    def _binarize(variant: str) -> np.ndarray:
        # Preprocessing variants borrowed from the depth block's scale-label OCR
        # (rect_depth_hybrid.py): CLAHE rescues dim/dark-grey rulers, horizontal-line
        # suppression peels the drawn scale bar off the digits. tesseract wants dark
        # text on light, hence the final invert.
        a = up_gray
        if variant == "clahe":
            a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)
            b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif variant == "line_suppressed":
            a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)
            kw = max(17, min(61, int(round(0.22 * a.shape[1]))))
            horiz = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
            b = cv2.threshold(cv2.subtract(a, horiz), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:  # base
            b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.bitwise_not(b)

    out: Dict[Tuple[int, str], Tuple[float, float, float]] = {}

    def _run(up: np.ndarray) -> None:
        for psm in psms:
            try:
                data = pytesseract.image_to_data(
                    up, config=_OCR_CONFIG.format(psm=psm), output_type=pytesseract.Output.DICT
                )
            except Exception:  # noqa: BLE001 - OCR must never break the pipeline
                continue
            for i, raw in enumerate(data.get("text", [])):
                if not raw or not raw.strip():
                    continue
                try:
                    conf = float(data["conf"][i])
                except (KeyError, ValueError, TypeError):
                    conf = -1.0
                if conf < prof.ocr_min_conf:
                    continue
                value = _parse_number(raw)
                if value is None:
                    continue
                y_centre = (data["top"][i] + data["height"][i] / 2.0) / prof.ocr_upscale
                key = (int(round(y_centre)), raw.strip())
                prev = out.get(key)
                if prev is None or conf > prev[2]:
                    out[key] = (y_centre, value, conf)

    _run(_binarize("base"))
    # Enhanced fallbacks only when the plain pass is thin — OCR dominates the runtime, so
    # we do not pay for CLAHE / line-suppression on the strips that already read cleanly.
    if len(out) < 2 and os.environ.get("SCALE_HIGH_RECALL"):
        _run(_binarize("clahe"))
        _run(_binarize("line_suppressed"))
    return sorted(out.values(), key=lambda t: t[0])


def read_labels(
    gray: np.ndarray,
    ladder: TickLadder,
    prof: ScaleProfile,
) -> Tuple[List[Tuple[float, float]], str]:
    """OCR the numbers beside a ladder and snap each to its tick.

    The OCR bounding box centre is only roughly aligned with the tick it labels (a
    "2.0" glyph box is not centred on the tick), so the tick y — which we measured to
    sub-pixel accuracy in stage 2 — is what we keep. OCR only supplies the *value*.
    """
    h, w = gray.shape
    # Fixed margin, deliberately *not* proportional to the pitch: the strip is
    # Otsu-thresholded as a whole, so letting its height follow the pitch made the
    # binarisation — and therefore which digits tesseract sees — depend on an unrelated
    # parameter. That coupling silently cost labels when the pitch estimate improved.
    margin = 30
    y_lo = int(max(0, ladder.ticks_y[0] - margin))
    y_hi = int(min(h, ladder.ticks_y[-1] + margin))
    if y_hi - y_lo < 10:
        return [], ""

    best: List[Tuple[float, float]] = []
    best_side = ""
    snap_tol = max(6.0, ladder.pitch * 0.5)
    # Legacy line 21 stores label_side = -1 on all 5290 rows of the n1 corpus, i.e. the
    # numbers are printed to the left of the ticks. Try that side first and skip the
    # other one when it already reads cleanly — OCR is the slow part of the pipeline.
    for side, sign in (("left", -1), ("right", +1)):
        if len(best) >= 3:
            break
        if sign < 0:
            x1 = int(max(0, ladder.x - prof.label_strip_gap - prof.label_strip_w))
            x2 = int(max(1, ladder.x - prof.label_strip_gap))
        else:
            x1 = int(min(w - 1, ladder.x + prof.label_strip_gap))
            x2 = int(min(w, ladder.x + prof.label_strip_gap + prof.label_strip_w))
        if x2 - x1 < 12:
            continue

        strip = gray[y_lo:y_hi, x1:x2]
        hits = _ocr_strip(strip, prof)
        raw = _collect(hits, y_lo, prof)
        if len(raw) < 2 and prof.ocr_psms_fallback:
            raw = _collect(_ocr_strip(strip, prof, prof.ocr_psms_fallback), y_lo, prof) or raw
        pairs = _assign_labels_to_ticks(raw, ladder, snap_tol, prof)
        if len(pairs) > len(best):
            best, best_side = pairs, side
    return best, best_side


def _collect(
    hits: Sequence[Tuple[float, float, float]],
    y_offset: float,
    prof: ScaleProfile,
) -> List[Tuple[float, float, float]]:
    """Keep the OCR hits that snap to a printable label value."""
    out: List[Tuple[float, float, float]] = []
    for y_rel, raw_value, conf in hits:
        value = _snap_label(raw_value, prof)
        if value is not None:
            out.append((y_offset + y_rel, value, conf))
    return out


def _coherent_assignment(
    raw: Sequence[Tuple[float, float, float]],
    ladder: TickLadder,
    slope: float,
    intercept: float,
    reach: float,
    prof: ScaleProfile,
) -> Optional[List[Tuple[float, float]]]:
    """Choose the label->tick assignment whose implied tick step is a real printed step.

    Each label gets a shortlist of nearby ticks (its own and the neighbours it could slip
    onto). We enumerate the combinations, fit ``value_mm`` against the *tick* y of each
    combination, and keep the one whose ``mm_per_px * pitch`` lands on ``plausible_tick_mm``
    with the smallest fit residual. Returns None when nothing is corroborated, so the
    caller can fall back to the nearest-value assignment.
    """
    if ladder.pitch <= 0 or not ladder.ticks_y:
        return None
    # The readings themselves are the unreliable part, not only which tick they belong to.
    # Measured on Esaote with the high-recall OCR passes: a strip yielded 5.0 cm then 4.0 cm
    # at *increasing* y (values must not decrease along a depth scale), and elsewhere
    # "2,4,3,7,9" where the printed numbers were 8.8/9.8/10.1 — single digits misread out of
    # two-digit labels, plus digits picked up from the image. Choosing tick assignments
    # while trusting those values cannot recover: the value is what is wrong.
    #
    # So select the largest subset of readings that is mutually consistent — monotonic in y
    # and implying a printed tick step a scanner really uses — and drop the rest.
    pool = sorted(raw, key=lambda r: r[0])[:6]  # by y; cap keeps the subset search small
    if len(pool) < 2:
        return None

    def _tick_for(y_ocr: float) -> Optional[float]:
        near = [t for t in ladder.ticks_y if abs(t - y_ocr) <= reach]
        return min(near, key=lambda t: abs(t - y_ocr)) if near else None

    best: Optional[Tuple[Tuple[int, float], List[Tuple[float, float]]]] = None
    for size in range(min(4, len(pool)), 1, -1):
        for subset in itertools.combinations(pool, size):
            vals = [r[1] for r in subset]
            if len(set(vals)) != len(vals):  # one printed number cannot repeat
                continue
            # monotonic in y: strictly increasing or strictly decreasing values
            inc = all(b > a for a, b in zip(vals, vals[1:]))
            dec = all(b < a for a, b in zip(vals, vals[1:]))
            if not (inc or dec):
                continue
            ticks = [_tick_for(r[0]) for r in subset]
            if any(t is None for t in ticks) or len(set(ticks)) != len(ticks):
                continue
            ts = np.asarray(ticks, dtype=float)
            vs = np.asarray(vals, dtype=float) * 10.0
            if float(ts.max() - ts.min()) < 4.0:
                continue
            A = np.vstack([ts, np.ones_like(ts)]).T
            sol, *_ = np.linalg.lstsq(A, vs, rcond=None)
            s, b = float(sol[0]), float(sol[1])
            if s == 0:
                continue
            _snapped, _mult, corroborated = _corroborate_step(abs(s) * ladder.pitch, prof)
            if not corroborated:
                continue
            resid = float(np.median(np.abs(vs - (s * ts + b))))
            key = (-size, resid)  # more agreeing labels first, then tightest fit
            if best is None or key < best[0]:
                best = (key, [(float(t), float(v)) for t, v in zip(ts, vals)])
        if best is not None:
            break  # a consistent subset of this size exists: no need to go smaller
    if best is None:
        return None
    return sorted(best[1])


def _assign_labels_to_ticks(
    raw: Sequence[Tuple[float, float, float]],
    ladder: TickLadder,
    snap_tol: float,
    prof: ScaleProfile,
) -> List[Tuple[float, float]]:
    """Attach each OCR reading to the tick it labels.

    Snapping to the nearest tick is not enough: a scanner does not centre the glyph box
    on its tick (a "2.0" box sits a few px above it), so a naive nearest-tick snap slips
    by one tick whenever that offset exceeds half the pitch. One slipped tick moves the
    extrapolated zero by a whole tick step — the dominant ``y_zero`` error in the first
    BK run.

    So we first fit on the *OCR* y positions, which is enough to know roughly how many
    mm a pixel is worth, then re-attach every label to the tick whose predicted value is
    closest to what we read, and finally keep the precise tick y.
    """
    if len(raw) < 2:
        return [(y, v) for y, v, _ in raw]

    ys = np.asarray([r[0] for r in raw], dtype=float)
    vs = np.asarray([r[1] for r in raw], dtype=float) * 10.0  # mm
    A = np.vstack([ys, np.ones_like(ys)]).T
    sol, *_ = np.linalg.lstsq(A, vs, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])
    if slope == 0:
        return [(min(ladder.ticks_y, key=lambda t: abs(t - y)), v) for y, v, _ in raw]

    # Pick the assignment that is *self-consistent*, not the one that is nearest.
    #
    # Nearest-by-predicted-value breaks down as soon as the minor ticks are all detected:
    # the OCR fit is only roughly right, so a label can land on the half-centimetre tick
    # next to its own (measured on Esaote: "4 cm" snapped to y=422 instead of y=384, one
    # pitch away, which alone moved mm_per_px from 0.268 to 0.322 — a 20% calibration
    # error). Only *one* label slips, so the slope changes and no downstream robust fit can
    # undo it. What does distinguish right from wrong is independent evidence: the implied
    # printed step ``mm_per_px * pitch`` must be a step a scanner actually prints. In that
    # same frame the correct assignment gives 9.9 mm (=1 cm) and the slipped one 12.1 mm,
    # which is not a real step. So we enumerate the few plausible assignments and keep the
    # corroborated one.
    reach = max(snap_tol, 2 * ladder.pitch)
    coherent = _coherent_assignment(raw, ladder, slope, intercept, reach, prof)
    if coherent is not None:
        return coherent

    # Fallback: score every (label, tick) pair, then assign greedily and **injectively**:
    # two different printed numbers cannot label the same tick. Without this constraint a
    # coarse (correct) pitch collapses two labels onto one tick, which drops the pair
    # count below the two needed to calibrate.
    cands: List[Tuple[float, float, float, float]] = []  # (err, tick, value_cm, conf)
    for y_ocr, value_cm, conf in raw:
        target_mm = value_cm * 10.0
        for t in ladder.ticks_y:
            if abs(t - y_ocr) > reach:
                continue
            cands.append((abs((slope * t + intercept) - target_mm), t, value_cm, conf))
    cands.sort(key=lambda c: (c[0], -c[3]))

    out: Dict[float, float] = {}
    used_values: set[float] = set()
    for _, tick, value_cm, _conf in cands:
        if tick in out or value_cm in used_values:
            continue
        out[tick] = value_cm
        used_values.add(value_cm)
    return sorted(out.items())


# --------------------------------------------------------------------------- #
# stage 4 - calibration from the labels
# --------------------------------------------------------------------------- #


def calibrate_from_labels(
    labels: Sequence[Tuple[float, float]],
    ladder: TickLadder,
) -> Optional[Tuple[float, float, float, List[Tuple[float, float]], float]]:
    """Fit ``value_mm = slope * (y - y_zero)`` over label pairs, either direction.

    Labels are in cm (that is how scanners print them and how legacy line 21 stores
    the tick step), so ``value_mm = 10 * value_cm``.

    The slope may be **negative**: on UD-flipped acquisitions the scanner prints 0 at
    the bottom and the numbers grow upwards. ``mm_per_px`` returned is ``|slope|``, and
    ``y_zero`` is where the fit crosses zero regardless of direction — which is exactly
    the ``y1`` the legacy .fss stores.

    Returns ``(mm_per_px, y_zero, residual_mm, inliers, tick_step_mm)``.
    """
    pts = [(y, v * 10.0) for y, v in labels]
    if len(pts) < 2:
        return None

    best: Optional[Tuple[float, float, float, List[Tuple[float, float]], float]] = None
    best_key: Tuple[int, float] = (-1, -1e9)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (y1, v1), (y2, v2) = pts[i], pts[j]
            if abs(y2 - y1) < 4 or abs(v2 - v1) < 1e-6:
                continue
            slope = (v2 - v1) / (y2 - y1)  # mm per px, signed
            if slope == 0:
                continue
            y_zero = y1 - v1 / slope
            resid = [abs(v - slope * (y - y_zero)) for y, v in pts]
            inliers = [p for p, r in zip(pts, resid) if r <= 1.5]
            if len(inliers) < 2:
                continue
            # least squares refit on the inliers
            ys = np.asarray([p[0] for p in inliers], dtype=float)
            vs = np.asarray([p[1] for p in inliers], dtype=float)
            A = np.vstack([ys, np.ones_like(ys)]).T
            sol, *_ = np.linalg.lstsq(A, vs, rcond=None)
            slope_r, intercept = float(sol[0]), float(sol[1])
            if slope_r == 0:
                continue
            y_zero_r = -intercept / slope_r
            res_r = float(np.median(np.abs(vs - (slope_r * ys + intercept))))
            step_mm = abs(slope_r) * ladder.pitch
            key = (len(inliers), -res_r)
            if key > best_key:
                best_key = key
                best = (abs(slope_r), y_zero_r, res_r, inliers, step_mm)
    return best


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def _refine_zero(
    pairs: Sequence[Tuple[float, float]],
    ladder: TickLadder,
    mm_per_px: float,
    direction: int,
    fit_zero: float,
) -> Tuple[float, str]:
    """Pin the zero to a real tick when the label values leave it ambiguous.

    With only two labels the fit is exact for *any* consistent misreading: reading
    "1, 2" where the scanner printed "2, 3" gives the right mm/px and a zero one whole
    centimetre off. Nothing in the labels can settle it — but the ruler itself can: a
    scanner draws the ladder starting at zero, so the zero coincides with the ladder's
    extreme tick on the side the values count up from.

    Returns ``(y_zero, source)`` where source is ``fit`` or ``ladder_end``.
    """
    if not ladder.ticks_y or mm_per_px <= 0:
        return fit_zero, "fit"
    end = ladder.ticks_y[0] if direction > 0 else ladder.ticks_y[-1]

    def consistency(zero: float) -> float:
        if not pairs:
            return 0.0
        return sum(
            abs(direction * (ty - zero) * mm_per_px - v * 10.0) for ty, v in pairs
        ) / len(pairs)

    err_fit, err_end = consistency(fit_zero), consistency(end)
    # Override only when the ladder end explains the labels *as well as* the fit does.
    # A looser rule (half a tick step of slack) was tried and lost accuracy: on the BK
    # families whose zero tick is too faint to detect, the ladder end is genuinely not
    # the zero, and the extrapolated fit is the better answer.
    if err_end <= err_fit + 0.5:
        return end, "ladder_end"
    return fit_zero, "fit"


def _corroborate_step(step_mm: float, prof: ScaleProfile) -> Tuple[float, int, bool]:
    """Snap ``step_mm`` to a printable tick step, allowing the pitch to be a submultiple.

    Returns ``(step_mm_corrected, pitch_multiplier, corroborated)``.
    """
    best: Tuple[float, int, bool] = (step_mm, 1, False)
    best_err = float("inf")
    for mult in (1, 2, 3, 4, 5):
        cand = step_mm * mult
        nearest = min(prof.plausible_tick_mm, key=lambda t: abs(t - cand))
        err = abs(cand - nearest) / nearest
        if err <= prof.tick_step_rel_tol and err < best_err:
            best_err = err
            best = (cand, mult, True)
    return best


def calibrate_from_geometry(
    ladder: TickLadder,
    reads_cm: Sequence[Tuple[float, float]],
    prof: ScaleProfile,
    tol: float = 0.12,
) -> Optional[Tuple[float, float, int, float, int]]:
    """Calibrate from tick geometry when the labels are too few to fit a line.

    The label-fit path needs two correctly-read numbers; on the zoomed-in / dim views that
    is often not available (measured: only 10-44% of frames get two correct reads). But the
    tick *pitch* is measured to ~1 px, and the printed step is one of a tiny set of values.
    An oracle test showed that ``step / pitch`` reproduces the GT mm_per_px to <2% on 82-88%
    of Esaote/BK/Hitachi frames *if the step is known*. So we use a single read only to pick
    the step, then take the precision from the pitch:

      mm_per_px_raw = value_mm / |y_read - y_zero|      # rough, carries the zero noise
      step          = snap(mm_per_px_raw * pitch)       # to a real printed step
      mm_per_px     = step / pitch                      # precise, from the pitch

    Returns ``(mm_per_px, y_zero, direction, step_mm, n_agree)`` or None. n_agree is how many
    reads the calibration explains — 1 means a single-label guess (caller should keep it in
    review), >=2 means corroborated.
    """
    if ladder.pitch <= 0 or len(ladder.ticks_y) < 2 or not reads_cm:
        return None
    ends = [min(ladder.ticks_y), max(ladder.ticks_y)]
    best: Optional[Tuple[Tuple[int, float], float, float, int, float]] = None
    for y0 in ends:
        direction = 1 if y0 == ends[0] else -1
        for y_r, v_cm in reads_cm:
            d = abs(y_r - y0)
            if v_cm <= 0 or d < ladder.pitch:
                continue
            step_raw = (v_cm * 10.0 / d) * ladder.pitch
            nearest = min(prof.plausible_tick_mm, key=lambda t: abs(t - step_raw) / t)
            if abs(step_raw - nearest) / nearest > tol:
                continue
            mmpp = nearest / ladder.pitch
            resids = [abs(mmpp * abs(yy - y0) - vv * 10.0) for yy, vv in reads_cm]
            agree = sum(1 for r in resids if r <= 3.0)  # within 3 mm
            key = (agree, -float(np.median(resids)))
            if best is None or key > best[0]:
                best = (key, mmpp, y0, direction, nearest)
    if best is None:
        return None
    _key, mmpp, y0, direction, step = best
    return mmpp, y0, direction, step, best[0][0]


def _bands_for(
    width: int,
    rect: Optional[Tuple[int, int, int, int]],
    prof: ScaleProfile,
) -> List[Tuple[int, int, str]]:
    if rect is None:
        mid = width // 2
        return [(mid, width - 1, "right"), (0, mid, "left")]
    _, _, rx2, _ = rect[0], rect[1], rect[2], rect[3]
    rx1 = rect[0]
    lo_r, hi_r = prof.right_band_px
    lo_l, hi_l = prof.left_band_px
    return [
        (max(0, rx2 - abs(lo_r)), min(width - 1, rx2 + hi_r), "right"),
        (max(0, rx1 - hi_l), min(width - 1, rx1 + abs(lo_l)), "left"),
    ]


def detect_scale(
    image_bgr: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    vendor: str = "",
    profile: Optional[ScaleProfile] = None,
    max_ladders: int = 4,
    prior_x: Optional[float] = None,
    prior_band_px: int = 70,
) -> ScalePrediction:
    """Detect the scale in a full-frame acquisition screenshot.

    ``prior_x`` is the ruler column suggested by the heatmap network. When given, the
    search collapses to a narrow band around it instead of scanning both sides of the
    rect. That is the whole point of adding the network: choosing the column is the
    decision the threshold-based search gets wrong, while finding the ticks *inside* the
    right column is what the classical stage does well.
    """
    prof = profile or profile_for(vendor)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape

    if prior_x is not None:
        side = "right" if (rect is None or prior_x >= (rect[0] + rect[2]) / 2) else "left"
        bands = [
            (
                max(0, int(prior_x) - prior_band_px),
                min(w - 1, int(prior_x) + prior_band_px),
                side,
            )
        ]
    else:
        bands = _bands_for(w, rect, prof)
    # Threshold inside each band: the ruler is the brightest thing in the lateral
    # margin even when it is dim compared with the anatomy in the middle.
    ladders: List[TickLadder] = []
    n_dashes = 0
    for x_lo, x_hi, side in bands:
        x_lo, x_hi = max(0, int(x_lo)), min(w, int(x_hi) + 1)
        if x_hi - x_lo < 8:
            continue
        band = gray[:, x_lo:x_hi]
        dashes = find_tick_dashes(band, prof, x_offset=x_lo)
        n_dashes += len(dashes)
        ladders.extend(group_ladders(dashes, (h, w), prof, [(x_lo, x_hi, side)]))

    if not ladders and prior_x is not None:
        # The prior band was empty: fall back to the geometric bands before giving up.
        for x_lo, x_hi, side in _bands_for(w, rect, prof):
            x_lo, x_hi = max(0, int(x_lo)), min(w, int(x_hi) + 1)
            if x_hi - x_lo < 8:
                continue
            dashes = find_tick_dashes(gray[:, x_lo:x_hi], prof, x_offset=x_lo)
            n_dashes += len(dashes)
            ladders.extend(group_ladders(dashes, (h, w), prof, [(x_lo, x_hi, side)]))
    if not ladders:
        # Last resort: whole frame with a global threshold.
        dashes = find_tick_dashes(gray, prof)
        n_dashes = max(n_dashes, len(dashes))
        ladders = group_ladders(dashes, (h, w), prof, [(0, w - 1, prof.prefer_side)])
    if not ladders:
        return ScalePrediction(
            ok=False,
            status="reject",
            reason="no_ladder" if n_dashes >= prof.min_ticks else "no_tick_dashes",
            debug={"n_dashes": n_dashes},
        )
    ladders.sort(key=lambda l: -l.score)

    best: Optional[ScalePrediction] = None
    for ladder in ladders[:max_ladders]:
        labels, side = read_labels(gray, ladder, prof)
        calib = calibrate_from_labels(labels, ladder) if len(labels) >= 2 else None
        # Geometry calibration (mm_per_px = step/pitch from a single label) fills the frames
        # the label-fit cannot. It is safe to run by default now that the consensus treats
        # its single-label output as a weak anchor (marked review, excluded from the trend
        # fit), so it adds coverage without dragging the robust trend. SCALE_NO_GEOM disables.
        geo = (calibrate_from_geometry(ladder, labels, prof)
               if (calib is None and labels and not os.environ.get("SCALE_NO_GEOM")) else None)
        if calib is None and geo is not None:
            # A single readable number is enough to pick the step; the pitch gives the
            # precision. Kept in review (the operator confirms) unless several reads agree,
            # since one number cannot rule out the 5-vs-10 mm ambiguity on its own.
            mm_per_px, y0, direction, step_mm, agree = geo
            y_zero = y0
            span_mm = max(
                (direction * (t - y_zero) * mm_per_px for t in ladder.ticks_y), default=0.0
            )
            cand = ScalePrediction(
                ok=True,
                # Never auto-accept: this rests on one readable number plus a guessed step,
                # the weakest evidence class in the chain. Measured on BK, auto-accepting the
                # "2 reads agree" case added 9 accepted of which 6 were wrong (98%->89%).
                # As review it still supplies coverage and an interpolation anchor, and the
                # operator confirms — which is exactly the policy for uncertain proposals.
                status="review",
                reason="geometry_step_over_pitch" + ("" if agree >= 2 else "_single_label"),
                x=ladder.x,
                y_zero=y_zero,
                mm_per_px=mm_per_px,
                tick_pitch_px=ladder.pitch,
                tick_step_mm=step_mm,
                y_last_tick=ladder.ticks_y[-1],
                y_first_tick=ladder.ticks_y[0],
                ticks_y=list(ladder.ticks_y),
                direction=direction,
                span_mm=span_mm,
                labels=[(y, v * 10.0) for y, v in labels],
                labels_all=[(y, v * 10.0) for y, v in labels],
                label_side=side or None,
                confidence=0.3 * ladder.score + 0.1 * agree,
                ladder_score=ladder.score,
                n_ticks=ladder.n_ticks,
                debug={"calib_source": "geometry"},
            )
        elif calib is None:
            cand = ScalePrediction(
                ok=False,
                status="review",
                reason="ladder_without_readable_labels",
                x=ladder.x,
                y_zero=ladder.y_first,
                tick_pitch_px=ladder.pitch,
                y_last_tick=ladder.ticks_y[-1],
                y_first_tick=ladder.ticks_y[0],
                ticks_y=list(ladder.ticks_y),
                label_side=side or None,
                # read_labels works in cm; both fields are documented in mm.
                labels=[(y, v * 10.0) for y, v in labels],
                labels_all=[(y, v * 10.0) for y, v in labels],
                confidence=0.25 * ladder.score,
                ladder_score=ladder.score,
                n_ticks=ladder.n_ticks,
            )
        else:
            mm_per_px, y_zero, resid, inliers, step_mm = calib
            n_lab = len(inliers)
            # Direction from the labels themselves: no need for the orientation block.
            ys = [p[0] for p in inliers]
            vs = [p[1] for p in inliers]
            inliers_cm = [(y, v / 10.0) for y, v in inliers]
            direction = 1 if (vs[-1] - vs[0]) * (ys[-1] - ys[0]) >= 0 else -1
            y_zero, zero_source = _refine_zero(inliers_cm, ladder, mm_per_px, direction, y_zero)
            span_mm = max(
                (direction * (t - y_zero) * mm_per_px for t in ladder.ticks_y), default=0.0
            )
            # The grid fit can lock onto a submultiple of the real pitch when spurious
            # dashes sit between ticks. The calibration does not care (it comes from the
            # labels), so instead of failing we look for the integer multiple of the
            # pitch that lands on a tick step a scanner would print.
            step_mm, pitch_mult, corroborated = _corroborate_step(step_mm, prof)
            conf = (
                0.40 * min(1.0, n_lab / 4.0)
                + 0.20 * max(0.0, 1.0 - resid / 2.0)
                + 0.15 * (1.0 if corroborated else 0.0)
                + 0.25 * ladder.score
            )
            if n_lab >= prof.min_labels_for_accept and resid <= prof.calib_residual_max_mm:
                status, reason = "accepted", ""
            elif n_lab == 2 and corroborated:
                status, reason = "accepted", "two_labels_corroborated_by_pitch"
            else:
                status = "review"
                reason = (
                    f"labels={n_lab} residual_mm={resid:.2f} "
                    f"step_mm={step_mm:.2f} x{pitch_mult} corroborated={int(corroborated)}"
                )
            cand = ScalePrediction(
                ok=True,
                status=status,
                reason=reason,
                x=ladder.x,
                y_zero=y_zero,
                mm_per_px=mm_per_px,
                tick_pitch_px=ladder.pitch,
                tick_step_mm=step_mm,
                y_last_tick=ladder.ticks_y[-1],
                y_first_tick=ladder.ticks_y[0],
                ticks_y=list(ladder.ticks_y),
                direction=direction,
                span_mm=span_mm,
                labels=[(y, v) for y, v in inliers],  # already mm
                labels_all=[(y, v * 10.0) for y, v in labels],  # read_labels gives cm
                label_side=side or None,
                calib_residual_mm=resid,
                confidence=min(1.0, conf),
                ladder_score=ladder.score,
                n_ticks=ladder.n_ticks,
                debug={"zero_source": zero_source, "pitch_multiplier": pitch_mult},
            )
        if best is None or cand.confidence > best.confidence:
            best = cand
        # The ladders are ordered by score, so a confident accept on an early one means
        # there is nothing to gain from OCR-ing the rest.
        if best.status == "accepted" and best.confidence >= 0.7:
            break
    assert best is not None
    best.debug["n_dashes"] = n_dashes
    best.debug["n_ladders"] = len(ladders)
    return best


def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img
