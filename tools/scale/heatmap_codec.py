#!/usr/bin/env python3
"""Encode targets into 1-D heatmaps and decode predictions back, in numpy only.

Kept free of torch on purpose: this is the part with the arithmetic that can silently be
wrong (bin/pixel conversions, sub-bin decoding), so it must be testable and runnable
anywhere — including on a machine where the training environment is not installed.

Convention throughout: a heatmap of ``n_bins`` covers the normalised range [0, 1) of one
image axis, and bin ``k`` is centred on ``(k + 0.5) / n_bins``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def bin_centres(n_bins: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins


def encode_gaussian(
    value_norm: float,
    n_bins: int,
    sigma_bins: float = 2.0,
) -> np.ndarray:
    """Soft target: Gaussian centred on ``value_norm``, normalised to sum 1.

    A soft target rather than a one-hot for the usual reason — it gives the network a
    gradient telling it *which way* it is wrong, instead of only that it is wrong.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    centre_bin = float(value_norm) * n_bins - 0.5
    k = np.arange(n_bins, dtype=np.float64)
    h = np.exp(-0.5 * ((k - centre_bin) / max(1e-6, sigma_bins)) ** 2)
    total = h.sum()
    if total <= 0:  # target far outside the range: fall back to the nearest bin
        h = np.zeros(n_bins, dtype=np.float64)
        h[int(np.clip(round(centre_bin), 0, n_bins - 1))] = 1.0
        return h
    return h / total


def decode_soft_argmax(
    heat: np.ndarray,
    window_bins: int = 4,
) -> Tuple[float, float]:
    """Locate the peak with sub-bin precision.

    Returns ``(value_norm, confidence)``. The centre of mass is taken over a window
    around the argmax rather than the whole array, so a second ruler elsewhere in the
    frame cannot drag the estimate into the empty space between the two peaks — which is
    exactly what a global expectation would do on fusion layouts.
    """
    heat = np.asarray(heat, dtype=np.float64).ravel()
    n = heat.size
    if n == 0:
        return 0.0, 0.0
    h = np.clip(heat, 0.0, None)
    total = h.sum()
    if total <= 0:
        return 0.5, 0.0
    peak = int(np.argmax(h))
    lo, hi = max(0, peak - window_bins), min(n, peak + window_bins + 1)
    w = h[lo:hi]
    idx = np.arange(lo, hi, dtype=np.float64)
    centre_bin = float((w * idx).sum() / w.sum())
    value_norm = (centre_bin + 0.5) / n
    # Confidence: how much of the whole distribution sits under the chosen peak.
    confidence = float(w.sum() / total)
    return value_norm, confidence


def peak_sharpness(heat: np.ndarray, window_bins: int = 4) -> float:
    """Ratio between the best peak and the best peak elsewhere.

    Near 1 means the network saw two equally good rulers (fusion layouts, or a mirrored
    panel) and its answer is a coin flip — useful as a review trigger.
    """
    h = np.clip(np.asarray(heat, dtype=np.float64).ravel(), 0.0, None)
    n = h.size
    if n == 0 or h.max() <= 0:
        return 0.0
    peak = int(np.argmax(h))
    masked = h.copy()
    lo, hi = max(0, peak - 2 * window_bins), min(n, peak + 2 * window_bins + 1)
    masked[lo:hi] = 0.0
    second = float(masked.max())
    if second <= 0:
        return float("inf")
    return float(h[peak] / second)


def norm_to_px(value_norm: float, size_px: int) -> float:
    return float(value_norm) * float(size_px)


def span_mm_to_mm_per_px(log_span_mm: float, image_h: int) -> float:
    """Invert the ``log_span_mm`` target back into mm per original pixel.

    The network predicts ``log(mm_per_px * image_height)`` — the number of millimetres the
    frame spans vertically — because that is invariant to the resize applied to the input
    and comparable across vendors with different video resolutions.
    """
    if image_h <= 0:
        raise ValueError("image_h must be positive")
    return float(np.exp(log_span_mm)) / float(image_h)


def mm_per_px_to_log_span(mm_per_px: float, image_h: int) -> float:
    if mm_per_px <= 0 or image_h <= 0:
        raise ValueError("mm_per_px and image_h must be positive")
    return float(np.log(mm_per_px * image_h))


def decode_prediction(
    x_heat: np.ndarray,
    y_heat: np.ndarray,
    log_span_mm: float,
    direction_logits: Optional[np.ndarray],
    image_w: int,
    image_h: int,
) -> dict:
    """Full decode of one network output into image-space quantities."""
    x_norm, x_conf = decode_soft_argmax(x_heat)
    y_norm, y_conf = decode_soft_argmax(y_heat)
    direction = 1
    dir_conf = 0.0
    if direction_logits is not None:
        logits = np.asarray(direction_logits, dtype=np.float64).ravel()
        if logits.size >= 2:
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            # class 0 = zero at top (+1), class 1 = zero at bottom (-1)
            direction = 1 if p[0] >= p[1] else -1
            dir_conf = float(max(p[0], p[1]))
    return {
        "x": norm_to_px(x_norm, image_w),
        "y_zero": norm_to_px(y_norm, image_h),
        "mm_per_px": span_mm_to_mm_per_px(log_span_mm, image_h),
        "direction": direction,
        "x_confidence": x_conf,
        "y_confidence": y_conf,
        "direction_confidence": dir_conf,
        "x_sharpness": peak_sharpness(x_heat),
        "y_sharpness": peak_sharpness(y_heat),
    }
