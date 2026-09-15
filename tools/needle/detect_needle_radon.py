"""Find the needle by searching the line parameters directly, instead of thresholding first.

The Hough baseline keeps the brightest pixels and looks for collinear runs among them. That
throws away the intensities before the search starts, so the answer depends on a threshold
chosen before knowing what is in the frame, and the angular resolution is whatever the
accumulator was given. For a quantity that has to be right to one degree, both are avoidable.

Here the image is rotated to each candidate angle and every row becomes a candidate line, so
a row's mean is the intensity along that line. A needle is a ridge, so the score is the row's
mean minus the mean a few pixels above and below -- the same ridge test as before, but applied
to every line in the frame rather than to a handful of survivors of a threshold.

Coarse to fine: one degree over the half circle, then a tenth of a degree around each peak,
which costs about as much as the coarse pass and lands well inside the one-degree budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class RadonDetection:
    """A line in crop coordinates, described the way the .fss describes it."""

    angle_deg: float
    row: float            # distance from the top of the crop, along the rotated frame
    score: float
    support: float        # fraction of the line that is actually bright
    p1: Tuple[float, float]
    p2: Tuple[float, float]


def _rotate(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate about the centre, keeping the whole frame, with a validity mask."""
    h, w = image.shape
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w = int(h * sin + w * cos)
    out_h = int(h * cos + w * sin)
    matrix[0, 2] += out_w / 2.0 - centre[0]
    matrix[1, 2] += out_h / 2.0 - centre[1]
    rotated = cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                             borderValue=0)
    mask = cv2.warpAffine(np.ones_like(image, dtype=np.uint8), matrix, (out_w, out_h),
                          flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, mask


def _row_scores(image: np.ndarray, angle_deg: float, gap: int, min_cover: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge score of every horizontal line of the rotated frame."""
    rotated, mask = _rotate(image, angle_deg)
    counts = mask.sum(axis=1).astype(np.float32)
    width = float(image.shape[1])
    valid = counts >= max(8.0, min_cover * width)
    sums = (rotated.astype(np.float32) * mask).sum(axis=1)
    means = np.divide(sums, np.maximum(counts, 1.0))

    # neighbours at two distances, as in the Hough version: a wide band stays bright close by
    shifted = []
    for offset in (gap, -gap, 2 * gap, -2 * gap):
        shifted.append(np.roll(means, offset))
    background = np.maximum.reduce(shifted)
    score = np.where(valid, means - background, -1e9)
    return score, means


def detect(crop_gray: np.ndarray, angle_range: Tuple[float, float] = (-85.0, 85.0),
           coarse_step: float = 1.0, fine_step: float = 0.1, gap: int = 6,
           min_cover: float = 0.25, top_k: int = 4, min_score: float = 8.0,
           ) -> List[RadonDetection]:
    h, w = crop_gray.shape
    if h < 24 or w < 24:
        return []
    image = cv2.GaussianBlur(crop_gray, (3, 3), 0)

    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, coarse_step)
    best_per_angle: List[Tuple[float, float, int]] = []
    for angle in angles:
        score, _ = _row_scores(image, float(angle), gap, min_cover)
        row = int(np.argmax(score))
        best_per_angle.append((float(score[row]), float(angle), row))

    best_per_angle.sort(reverse=True)
    picked: List[RadonDetection] = []
    for coarse_score, coarse_angle, _ in best_per_angle:
        if coarse_score < min_score:
            break
        if any(abs(((coarse_angle - d.angle_deg) + 90) % 180 - 90) < 6.0 for d in picked):
            continue
        # refine around the peak, where a tenth of a degree costs nothing
        best: Optional[Tuple[float, float, int]] = None
        for angle in np.arange(coarse_angle - coarse_step, coarse_angle + coarse_step + 1e-9,
                               fine_step):
            score, _ = _row_scores(image, float(angle), gap, min_cover)
            row = int(np.argmax(score))
            if best is None or score[row] > best[0]:
                best = (float(score[row]), float(angle), row)
        if best is None:
            continue
        score, angle, row = best
        segment = _segment_of(crop_gray.shape, angle, row)
        if segment is None:
            continue
        support = _support(image, segment, gap)
        picked.append(RadonDetection(angle_deg=-angle, row=float(row), score=score,
                                     support=support, p1=segment[0], p2=segment[1]))
        if len(picked) >= top_k:
            break
    return picked


def _segment_of(shape: Tuple[int, int], angle_deg: float, row: int
                ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Map a row of the rotated frame back to a segment in the original crop."""
    h, w = shape
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w = int(h * sin + w * cos)
    out_h = int(h * cos + w * sin)
    matrix[0, 2] += out_w / 2.0 - centre[0]
    matrix[1, 2] += out_h / 2.0 - centre[1]
    inverse = cv2.invertAffineTransform(matrix)
    points = np.array([[0.0, float(row), 1.0], [float(out_w), float(row), 1.0]]).T
    back = inverse @ points
    p1 = (float(back[0, 0]), float(back[1, 0]))
    p2 = (float(back[0, 1]), float(back[1, 1]))
    clipped = _clip(p1, p2, w, h)
    return clipped


def _clip(p1, p2, w, h):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1), (dx, (w - 1) - x1), (-dy, y1), (dy, (h - 1) - y1)):
        if p == 0:
            if q < 0:
                return None
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
    if t0 >= t1:
        return None
    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))


def _support(image: np.ndarray, segment, gap: int, samples: int = 96) -> float:
    """How much of the line is genuinely brighter than its surroundings.

    A long line through mostly empty space can win on the mean alone; this asks what share of
    it is actually ridge, which separates a needle from a lucky alignment.
    """
    (x1, y1), (x2, y2) = segment
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 8:
        return 0.0
    t = np.linspace(0.0, 1.0, samples)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    h, w = image.shape

    def sample(ox: float, oy: float) -> np.ndarray:
        xi = np.clip(np.round(xs + ox).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys + oy).astype(int), 0, h - 1)
        return image[yi, xi].astype(np.float32)

    on = sample(0.0, 0.0)
    side = np.maximum(sample(nx * gap, ny * gap), sample(-nx * gap, -ny * gap))
    return float(np.mean(on - side > 4.0))


def detect_in_frame(frame_gray: np.ndarray, rect: Tuple[int, int, int, int], **kwargs
                    ) -> List[RadonDetection]:
    left, top, right, bottom = rect
    h, w = frame_gray.shape
    left = max(0, min(int(left), w - 1))
    top = max(0, min(int(top), h - 1))
    right = max(left + 1, min(int(right), w))
    bottom = max(top + 1, min(int(bottom), h))
    out = []
    for d in detect(frame_gray[top:bottom, left:right], **kwargs):
        out.append(RadonDetection(
            angle_deg=d.angle_deg, row=d.row, score=d.score, support=d.support,
            p1=(d.p1[0] + left, d.p1[1] + top), p2=(d.p2[0] + left, d.p2[1] + top)))
    return out
