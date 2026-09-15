#!/usr/bin/env python3
"""Find the needle in a calibration frame: the bright straight segment inside RECT_ECHO.

The needle reads as a straight run of pixels much brighter than the speckle around it, which
is what the operator looks for when tracing it by hand. So: keep the brightest pixels, let a
probabilistic Hough find the collinear runs among them, then score the candidates by how
bright and how long they are rather than trusting Hough's own ordering.

Two things inside RECT_ECHO are bright and straight but are not needles, and both are handled
here: the grey ramp, a solid vertical bar hugging one side, and the vendor watermark. A needle
is a *thin ridge* -- bright with darker pixels on both sides -- while the ramp is a wide block,
so candidates are also scored on how much darker their surroundings are.

Returns the segment in the coordinates of the frame it was given, so it can go straight into
`guides_geometry.measure_from_points`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    score: float
    brightness: float
    contrast: float
    length: float

    @property
    def angle_deg(self) -> float:
        """Image convention, y downwards, the same one the legacy planning angle uses."""
        return math.degrees(math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0]))


def _profile(gray: np.ndarray, p1: Sequence[float], p2: Sequence[float],
             samples: int, offset: float = 0.0) -> np.ndarray:
    """Sample the image along a segment, optionally shifted sideways by `offset` pixels."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return np.zeros(0, dtype=np.float32)
    nx, ny = -dy / length, dx / length          # unit normal
    t = np.linspace(0.0, 1.0, samples)
    xs = x1 + t * dx + nx * offset
    ys = y1 + t * dy + ny * offset
    h, w = gray.shape
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    return gray[yi, xi].astype(np.float32)


def detect_in_crop(crop_gray: np.ndarray, min_length_frac: float = 0.25,
                   bright_percentile: Optional[float] = None, max_candidates: int = 120,
                   ) -> Optional[Detection]:
    """Best candidate over several brightness cuts.

    A single percentile cannot serve every frame: on a dark water bath the needle is the top
    1% of pixels, while on a bright tissue view that same cut keeps half the image and Hough
    finds nothing. Several cuts are tried and the candidates compete on the same score, so the
    frame decides which cut suited it instead of a constant chosen in advance.
    """
    if bright_percentile is not None:
        return _detect_at(crop_gray, min_length_frac, bright_percentile, max_candidates)
    best: Optional[Detection] = None
    for percentile in (99.5, 99.0, 98.0, 96.0, 93.0):
        found = _detect_at(crop_gray, min_length_frac, percentile, max_candidates)
        if found is not None and (best is None or found.score > best.score):
            best = found
    return best


def _detect_at(crop_gray: np.ndarray, min_length_frac: float,
               bright_percentile: float, max_candidates: int) -> Optional[Detection]:
    h, w = crop_gray.shape
    if h < 16 or w < 16:
        return None

    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    threshold = float(np.percentile(blurred, bright_percentile))
    if threshold <= 1:
        return None
    mask = (blurred >= threshold).astype(np.uint8) * 255

    min_length = max(20.0, min_length_frac * math.hypot(w, h) * 0.5)
    # The grey ramp is a bright vertical bar hugging a side of RECT_ECHO; it is not a needle
    # and it is the single most common false positive, so that band of columns is dropped.
    edge = max(4, int(0.04 * w))
    lines = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 360.0, threshold=30,
        minLineLength=int(min_length), maxLineGap=14,
    )
    if lines is None:
        return None

    best: Optional[Detection] = None
    for entry in lines[:max_candidates]:
        x1, y1, x2, y2 = (float(v) for v in entry[0])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        near_edge = max(x1, x2) < edge or min(x1, x2) > w - edge
        if near_edge and abs(x2 - x1) < 0.2 * abs(y2 - y1):
            continue
        on = _profile(blurred, (x1, y1), (x2, y2), samples=64)
        if on.size == 0:
            continue
        # A needle is a ridge: dark on BOTH sides. The average of the two sides is not
        # enough, because a tissue band or the edge of the grey ramp is bright on one side
        # and dark on the other, and the average lets it through. The weaker side decides.
        above = _profile(blurred, (x1, y1), (x2, y2), 64, offset=+6.0)
        below = _profile(blurred, (x1, y1), (x2, y2), 64, offset=-6.0)
        brightness = float(np.median(on))
        contrast = float(brightness - max(np.median(above), np.median(below)))
        if contrast <= 6:
            continue
        score = contrast * math.log1p(length)
        if best is None or score > best.score:
            best = Detection((x1, y1), (x2, y2), score, brightness, contrast, length)
    return best


def detect_in_frame(frame_gray: np.ndarray, rect: Tuple[int, int, int, int],
                    **kwargs) -> Optional[Detection]:
    """`rect` is (left, top, right, bottom) in frame coordinates."""
    left, top, right, bottom = rect
    h, w = frame_gray.shape
    left = max(0, min(int(left), w - 1))
    top = max(0, min(int(top), h - 1))
    right = max(left + 1, min(int(right), w))
    bottom = max(top + 1, min(int(bottom), h))
    crop = frame_gray[top:bottom, left:right]
    found = detect_in_crop(crop, **kwargs)
    if found is None:
        return None
    return Detection(
        p1=(found.p1[0] + left, found.p1[1] + top),
        p2=(found.p2[0] + left, found.p2[1] + top),
        score=found.score, brightness=found.brightness,
        contrast=found.contrast, length=found.length,
    )


def load_gray(path) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image
