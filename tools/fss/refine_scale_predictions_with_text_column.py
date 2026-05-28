#!/usr/bin/env python3
"""Refine scale-line x predictions using vertical number-column anchor.

Workflow per run-dir:
- read train/val/test predictions CSV emitted by train_scale_line_image_model.py
- detect a text-column x anchor from each image
- calibrate vendor/run-specific offset on train: x_gt ~= x_col + offset
- apply blended correction on val/test (and train for diagnostics)
- write refined CSVs + summary metrics before/after
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class PredRow:
    sample_id: str
    split: str
    image_path: str
    x_gt: float
    y_top_gt: float
    y_bottom_gt: float
    x_pred: float
    y_top_pred: float
    y_bottom_pred: float


@dataclass(frozen=True)
class AnchorDet:
    x_image: float
    score: float
    quality: float
    runs: int
    coverage: float
    longest_ratio: float


@dataclass(frozen=True)
class TickDet:
    x_line_image: float
    score: float
    quality: float
    runs: int
    coverage: float
    periodicity: float
    vertical_support: float
    tick_rows: Tuple[int, ...]
    tick_step: float


def _f(x: str) -> float:
    return float(str(x).strip())


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))[:120]


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    b = mask.astype(np.int8)
    starts = int(np.sum((b[1:] == 1) & (b[:-1] == 0)))
    if b[0] == 1:
        starts += 1
    return starts


def _run_starts(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=np.int32)
    prev = np.zeros_like(mask, dtype=bool)
    prev[1:] = mask[:-1]
    return np.flatnonzero(mask & (~prev))


def _run_centers(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=np.float32)
    centers: List[float] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask.astype(bool)):
        if v and not in_run:
            in_run = True
            start = i
        if (not v) and in_run:
            end = i - 1
            centers.append(0.5 * (start + end))
            in_run = False
    if in_run:
        end = mask.size - 1
        centers.append(0.5 * (start + end))
    if not centers:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def _periodicity_score(mask: np.ndarray) -> float:
    starts = _run_starts(mask)
    if starts.size < 5:
        return 0.0
    diffs = np.diff(starts).astype(np.float32)
    med = float(np.median(diffs))
    if med < 2.0 or med > 50.0:
        return 0.0
    mad = float(np.median(np.abs(diffs - med)))
    return float(1.0 / (1.0 + mad))


def load_rows(pred_csv: Path) -> List[PredRow]:
    rows: List[PredRow] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rows.append(
                PredRow(
                    sample_id=str(r["sample_id"]),
                    split=str(r["split"]),
                    image_path=str(r["image_path"]),
                    x_gt=_f(r["x_gt"]),
                    y_top_gt=_f(r["y_top_gt"]),
                    y_bottom_gt=_f(r["y_bottom_gt"]),
                    x_pred=_f(r["x_pred"]),
                    y_top_pred=_f(r["y_top_pred"]),
                    y_bottom_pred=_f(r["y_bottom_pred"]),
                )
            )
    return rows


def build_video_size_map(manifest: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            sid = str(r.get("sample_id", "")).strip()
            if not sid:
                continue
            vw = r.get("video_x_size")
            vh = r.get("video_y_size")
            if vw is None or vh is None:
                continue
            try:
                out[sid] = (float(vw), float(vh))
            except Exception:
                continue
    return out


def detect_text_column_anchor(
    gray: np.ndarray,
    x_pred_image: float,
    y_top_image: float,
    y_bottom_image: float,
    search_radius_x: int = 220,
) -> Optional[AnchorDet]:
    H, W = gray.shape
    if H < 40 or W < 40:
        return None

    gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    blur = np.asarray(Image.fromarray(gray_u8, mode="L").filter(ImageFilter.BoxBlur(radius=3.0)), dtype=np.float32)
    hi = np.abs(gray - blur)

    gx = np.zeros_like(hi, dtype=np.float32)
    gx[:, 1:-1] = np.abs(hi[:, 2:] - hi[:, :-2])
    gy = np.zeros_like(hi, dtype=np.float32)
    gy[1:-1, :] = np.abs(hi[2:, :] - hi[:-2, :])
    feat = 0.75 * gx + 0.25 * gy

    thr = float(np.percentile(feat, 91.5))
    mask = feat > max(4.0, thr)

    yp0 = int(max(0.0, min(y_top_image, y_bottom_image)))
    yp1 = int(min(float(H), max(y_top_image, y_bottom_image)))
    pad = int(max(14.0, 0.08 * H))
    y0 = max(0, yp0 - pad)
    y1 = min(H, yp1 + pad)
    if y1 - y0 < int(0.16 * H):
        y0 = int(0.05 * H)
        y1 = int(0.95 * H)
    if y1 <= y0:
        return None

    # Search only near predicted line x (robust to global side-text artifacts).
    cx = int(round(float(x_pred_image)))
    x_min = max(int(0.12 * W), cx - int(search_radius_x))
    x_max = min(int(0.96 * W), cx + int(search_radius_x))
    if x_max <= x_min:
        return None

    best: Optional[AnchorDet] = None
    best_score = -1e9

    for x in range(x_min, x_max):
        xl = max(0, x - 2)
        xr = min(W, x + 3)
        band = mask[y0:y1, xl:xr]
        if band.size == 0:
            continue

        col = band.any(axis=1)
        if col.size >= 7:
            col = np.convolve(col.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0

        coverage = float(np.mean(col))
        if coverage < 0.02 or coverage > 0.78:
            continue

        runs = _count_runs(col)
        if runs < 2:
            continue
        longest = _longest_true_run(col)
        longest_ratio = float(longest / max(1, col.size))

        local_energy = float(np.mean(feat[y0:y1, xl:xr]))
        # Prefer text-like vertical stacks: many runs, medium coverage, short longest run.
        shape_score = (
            7.0 * float(runs)
            + 130.0 * coverage
            - 170.0 * max(0.0, longest_ratio - 0.25)
            + 0.06 * local_energy
        )
        # Keep anchor very near model prediction to avoid opposite-side false hits.
        dist = abs(float(x) - float(x_pred_image))
        if dist > float(search_radius_x):
            continue
        dist_bonus = max(0.0, 55.0 - 0.34 * dist)
        score = shape_score + dist_bonus - 0.02 * dist

        cov_pref = max(0.0, 1.0 - abs(coverage - 0.22) / 0.22)
        run_pref = min(1.0, runs / 14.0)
        long_pref = max(0.0, 1.0 - max(0.0, longest_ratio - 0.20) / 0.45)
        quality = float(np.clip(0.45 * run_pref + 0.30 * cov_pref + 0.25 * long_pref, 0.0, 1.0))

        if score > best_score:
            best_score = score
            best = AnchorDet(
                x_image=float(x),
                score=float(score),
                quality=quality,
                runs=int(runs),
                coverage=float(coverage),
                longest_ratio=float(longest_ratio),
            )

    return best


def detect_tick_line_anchor(
    gray: np.ndarray,
    x_pred_image: float,
    y_top_image: float,
    y_bottom_image: float,
    search_radius_x: int = 170,
) -> Optional[TickDet]:
    H, W = gray.shape
    if H < 40 or W < 40:
        return None

    gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    blur = np.asarray(Image.fromarray(gray_u8, mode="L").filter(ImageFilter.BoxBlur(radius=2.0)), dtype=np.float32)
    hi = np.abs(gray - blur)
    gx = np.zeros_like(hi, dtype=np.float32)
    gx[:, 1:-1] = np.abs(hi[:, 2:] - hi[:, :-2])
    gy = np.zeros_like(hi, dtype=np.float32)
    gy[1:-1, :] = np.abs(hi[2:, :] - hi[:-2, :])
    feat = 0.72 * gx + 0.28 * gy

    yp0 = int(max(0.0, min(y_top_image, y_bottom_image)))
    yp1 = int(min(float(H), max(y_top_image, y_bottom_image)))
    pad = int(max(14.0, 0.08 * H))
    y0 = max(0, yp0 - pad)
    y1 = min(H, yp1 + pad)
    if y1 - y0 < int(0.18 * H):
        y0 = int(0.05 * H)
        y1 = int(0.95 * H)
    if y1 <= y0:
        return None

    cx = int(round(float(x_pred_image)))
    x_min = max(int(0.10 * W), cx - int(search_radius_x))
    x_max = min(int(0.98 * W), cx + int(search_radius_x))
    if x_max <= x_min:
        return None

    base_thr = float(np.percentile(feat[y0:y1, :], 82.0))
    best: Optional[TickDet] = None
    best_score = -1e9

    for x in range(x_min, x_max):
        xl0 = max(0, x - 14)
        xl1 = max(xl0 + 1, x)
        xr0 = min(W - 1, x + 1)
        xr1 = min(W, x + 9)
        xc0 = max(0, x - 1)
        xc1 = min(W, x + 2)
        if xl1 <= xl0 or xr1 <= xr0 or xc1 <= xc0:
            continue

        left = feat[y0:y1, xl0:xl1]
        right = feat[y0:y1, xr0:xr1]
        center = feat[y0:y1, xc0:xc1]
        if left.size == 0 or right.size == 0 or center.size == 0:
            continue

        left_m = left.mean(axis=1)
        right_m = right.mean(axis=1)
        center_m = center.mean(axis=1)
        row_hits = (left_m > max(3.5, 0.75 * base_thr)) & (left_m > right_m * 1.18)
        if row_hits.size >= 9:
            row_hits = np.convolve(row_hits.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0
        coverage = float(np.mean(row_hits))
        if coverage < 0.02 or coverage > 0.80:
            continue

        runs = _count_runs(row_hits)
        if runs < 2:
            continue
        periodicity = _periodicity_score(row_hits)
        centers = _run_centers(row_hits)
        if centers.size < 2:
            continue
        center_diffs = np.diff(centers).astype(np.float32)
        tick_step = float(np.median(center_diffs)) if center_diffs.size else float("nan")
        vertical_support = float(np.mean(center_m > max(3.0, 0.70 * base_thr)))

        right_penalty = float(np.mean(np.maximum(0.0, right_m - 0.85 * left_m)))
        dist = abs(float(x) - float(x_pred_image))
        if dist > float(search_radius_x):
            continue
        score = (
            210.0 * coverage
            + 5.5 * float(runs)
            + 42.0 * periodicity
            + 75.0 * vertical_support
            - 8.5 * right_penalty
            + max(0.0, 36.0 - 0.24 * dist)
        )

        run_pref = min(1.0, runs / 16.0)
        cov_pref = max(0.0, 1.0 - abs(coverage - 0.22) / 0.22)
        per_pref = min(1.0, periodicity / 0.60)
        vert_pref = min(1.0, vertical_support / 0.50)
        quality = float(np.clip(0.33 * run_pref + 0.25 * cov_pref + 0.22 * per_pref + 0.20 * vert_pref, 0.0, 1.0))

        if score > best_score:
            best_score = score
            abs_tick_rows = tuple(int(round(y0 + float(c))) for c in centers.tolist())
            best = TickDet(
                x_line_image=float(x),
                score=float(score),
                quality=quality,
                runs=int(runs),
                coverage=float(coverage),
                periodicity=float(periodicity),
                vertical_support=float(vertical_support),
                tick_rows=abs_tick_rows,
                tick_step=tick_step,
            )

    return best


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def refine_rows(
    rows: Sequence[PredRow],
    video_size_map: Dict[str, Tuple[float, float]],
    offset_med: float,
    min_anchor_quality: float,
    min_tick_quality: float,
    max_refine_shift_px: float,
    tick_line_offset_px: float,
    hard_tick_quality: float,
    refine_y_from_zero_tick: bool,
    max_y_refine_shift_px: float,
    y_zero_min_tick_quality: float,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    out: List[Dict[str, object]] = []
    signed_before: List[float] = []
    signed_after: List[float] = []
    used_anchor = 0
    missing_video_w = 0
    missing_anchor = 0
    used_ticks = 0
    used_numbers = 0
    y_refined_rows = 0

    for r in rows:
        x_ref = float(r.x_pred)
        anchor_x_video = float("nan")
        anchor_q = 0.0
        anchor_runs = 0
        anchor_cov = float("nan")
        anchor_long = float("nan")
        tick_q = 0.0
        tick_runs = 0
        tick_cov = float("nan")
        tick_periodicity = float("nan")
        tick_vertical_support = float("nan")
        tick_step = float("nan")
        zero_tick_y_video = float("nan")
        y_top_ref = float(r.y_top_pred)
        y_bottom_ref = float(r.y_bottom_pred)
        anchor_source = "none"

        vsize = video_size_map.get(r.sample_id)
        if vsize is None:
            missing_video_w += 1
        else:
            vw, vh = vsize
            if vw <= 1 or vh <= 1:
                missing_video_w += 1
                continue
            img = Image.open(r.image_path).convert("L")
            gray = np.asarray(img, dtype=np.float32)
            H, W = gray.shape
            sx = float(W) / float(vw)
            sy = float(H) / float(vh)
            x_pred_img = float(r.x_pred) * sx
            y_top_img = float(r.y_top_pred) * sy
            y_bottom_img = float(r.y_bottom_pred) * sy
            det_tick = detect_tick_line_anchor(
                gray,
                x_pred_image=x_pred_img,
                y_top_image=y_top_img,
                y_bottom_image=y_bottom_img,
            )
            det = detect_text_column_anchor(
                gray,
                x_pred_image=x_pred_img,
                y_top_image=y_top_img,
                y_bottom_image=y_bottom_img,
            )
            if det_tick is not None:
                tick_q = float(det_tick.quality)
                tick_runs = int(det_tick.runs)
                tick_cov = float(det_tick.coverage)
                tick_periodicity = float(det_tick.periodicity)
                tick_vertical_support = float(det_tick.vertical_support)
                tick_step = float(det_tick.tick_step)
            if det_tick is not None and det_tick.quality >= min_tick_quality:
                x_line_video = float(det_tick.x_line_image * (float(vw) / float(W)))
                # Place scale line just to the right of ticks so ticks stay visible.
                x_target = x_line_video + float(tick_line_offset_px)
                if tick_q >= hard_tick_quality:
                    x_ref = float(np.clip(x_target, 0.0, max(0.0, vw - 1.0)))
                else:
                    w = float(np.clip(0.45 + 0.45 * tick_q, 0.45, 0.90))
                    delta = float(x_target - float(r.x_pred))
                    delta = float(np.clip(delta, -max_refine_shift_px, max_refine_shift_px))
                    x_target = float(r.x_pred) + delta
                    x_ref = (1.0 - w) * float(r.x_pred) + w * float(x_target)
                    x_ref = float(np.clip(x_ref, 0.0, max(0.0, vw - 1.0)))
                used_anchor += 1
                used_ticks += 1
                anchor_source = "ticks"
                if refine_y_from_zero_tick and det_tick.tick_rows and tick_q >= y_zero_min_tick_quality:
                    y_ticks_video = np.asarray(
                        [float(yi) * (float(vh) / float(H)) for yi in det_tick.tick_rows],
                        dtype=np.float32,
                    )
                    if y_ticks_video.size >= 6:
                        y_zero = float(np.min(y_ticks_video))
                        y_end = float(np.max(y_ticks_video))
                        if y_end > y_zero + 3.0:
                            zero_tick_y_video = y_zero
                            pred_span = max(4.0, float(r.y_bottom_pred) - float(r.y_top_pred))
                            tick_span = max(1.0, y_end - y_zero)
                            span_ratio = tick_span / pred_span

                            # Default: move whole segment so y_top starts at zero tick, preserving length.
                            if tick_q >= hard_tick_quality:
                                dy = float(np.clip(y_zero - float(r.y_top_pred), -max_y_refine_shift_px, max_y_refine_shift_px))
                            else:
                                wy = float(np.clip(0.30 + 0.55 * tick_q, 0.30, 0.85))
                                dy_raw = float(np.clip(y_zero - float(r.y_top_pred), -max_y_refine_shift_px, max_y_refine_shift_px))
                                dy = wy * dy_raw
                            y_top_ref = float(r.y_top_pred) + dy
                            y_bottom_ref = float(r.y_bottom_pred) + dy

                            # Optional stretch only when tick-span is very consistent with predicted span.
                            if 0.85 <= span_ratio <= 1.18 and tick_q >= hard_tick_quality:
                                y_top_ref = y_zero
                                y_bottom_ref = y_end
                            y_top_ref = float(np.clip(y_top_ref, 0.0, max(0.0, vh - 1.0)))
                            y_bottom_ref = float(np.clip(y_bottom_ref, 0.0, max(0.0, vh - 1.0)))
                            if y_top_ref > y_bottom_ref:
                                y_top_ref, y_bottom_ref = y_bottom_ref, y_top_ref
                            y_refined_rows += 1
            elif det is not None and det.quality >= min_anchor_quality:
                anchor_x_video = float(det.x_image * (float(vw) / float(W)))
                anchor_q = float(det.quality)
                anchor_runs = int(det.runs)
                anchor_cov = float(det.coverage)
                anchor_long = float(det.longest_ratio)
                x_target = anchor_x_video + offset_med
                w = float(np.clip(0.20 + 0.65 * anchor_q, 0.20, 0.85))
                delta = float(x_target - float(r.x_pred))
                delta = float(np.clip(delta, -max_refine_shift_px, max_refine_shift_px))
                x_target = float(r.x_pred) + delta
                x_ref = (1.0 - w) * float(r.x_pred) + w * float(x_target)
                x_ref = float(np.clip(x_ref, 0.0, max(0.0, vw - 1.0)))
                used_anchor += 1
                used_numbers += 1
                anchor_source = "numbers"
            else:
                missing_anchor += 1

        x_err_before = float(r.x_pred - r.x_gt)
        x_err_after = float(x_ref - r.x_gt)
        signed_before.append(x_err_before)
        signed_after.append(x_err_after)
        y_err_before = 0.5 * (abs(float(r.y_top_pred - r.y_top_gt)) + abs(float(r.y_bottom_pred - r.y_bottom_gt)))
        y_err_after = 0.5 * (abs(float(y_top_ref - r.y_top_gt)) + abs(float(y_bottom_ref - r.y_bottom_gt)))

        out.append(
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "image_path": r.image_path,
                "x_gt": r.x_gt,
                "y_top_gt": r.y_top_gt,
                "y_bottom_gt": r.y_bottom_gt,
                "x_pred": r.x_pred,
                "x_pred_refined": x_ref,
                "y_top_pred": r.y_top_pred,
                "y_bottom_pred": r.y_bottom_pred,
                "y_top_pred_refined": y_top_ref,
                "y_bottom_pred_refined": y_bottom_ref,
                "x_err_before": x_err_before,
                "x_err_after": x_err_after,
                "y_err": y_err_before,
                "y_err_before": y_err_before,
                "y_err_after": y_err_after,
                "anchor_x_video": anchor_x_video,
                "anchor_quality": anchor_q,
                "anchor_runs": anchor_runs,
                "anchor_coverage": anchor_cov,
                "anchor_longest_ratio": anchor_long,
                "tick_quality": tick_q,
                "tick_runs": tick_runs,
                "tick_coverage": tick_cov,
                "tick_periodicity": tick_periodicity,
                "tick_vertical_support": tick_vertical_support,
                "tick_step_px_image": tick_step,
                "zero_tick_y_video": zero_tick_y_video,
                "anchor_source": anchor_source,
            }
        )

    stats = {
        "rows": float(len(out)),
        "anchor_used_rows": float(used_anchor),
        "anchor_used_ratio": float(used_anchor / max(1, len(out))),
        "missing_video_w_rows": float(missing_video_w),
        "missing_anchor_rows": float(missing_anchor),
        "used_ticks_rows": float(used_ticks),
        "used_numbers_rows": float(used_numbers),
        "y_refined_rows": float(y_refined_rows),
        "signed_x_err_mean_before": float(np.mean(signed_before)) if signed_before else float("nan"),
        "signed_x_err_mean_after": float(np.mean(signed_after)) if signed_after else float("nan"),
    }
    return out, stats


def compute_metrics(rows: Sequence[Dict[str, object]], refined: bool) -> Dict[str, float]:
    x_key = "x_err_after" if refined else "x_err_before"
    if refined:
        y_key = "y_err_after" if rows and ("y_err_after" in rows[0]) else "y_err"
    else:
        y_key = "y_err_before" if rows and ("y_err_before" in rows[0]) else "y_err"
    x_abs = [abs(float(r[x_key])) for r in rows]
    y_abs = [float(r[y_key]) for r in rows]
    return {
        "rows": float(len(rows)),
        "mae_x_px": float(np.mean(x_abs)) if x_abs else float("nan"),
        "mae_y_px": float(np.mean(y_abs)) if y_abs else float("nan"),
        "score_xy": float(np.mean(x_abs) + np.mean(y_abs)) if x_abs else float("nan"),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def draw_refined_preview(row: Dict[str, object], out_path: Path) -> None:
    img = Image.open(str(row["image_path"])).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    x_gt = float(row["x_gt"])
    y_top_gt = float(row["y_top_gt"])
    y_bottom_gt = float(row["y_bottom_gt"])
    y_top_pred = float(row["y_top_pred"])
    y_bottom_pred = float(row["y_bottom_pred"])
    y1_pred = min(y_top_pred, y_bottom_pred)
    y2_pred = max(y_top_pred, y_bottom_pred)
    y_top_refined = float(row.get("y_top_pred_refined", y_top_pred))
    y_bottom_refined = float(row.get("y_bottom_pred_refined", y_bottom_pred))
    y1_ref = min(y_top_refined, y_bottom_refined)
    y2_ref = max(y_top_refined, y_bottom_refined)
    x_pred = float(row["x_pred"])
    x_ref = float(row["x_pred_refined"])

    # GT line (green), raw prediction (magenta), refined prediction (cyan, thinner).
    draw.line((x_gt, y_top_gt, x_gt, y_bottom_gt), fill=(0, 255, 110, 255), width=5)
    draw.line((x_pred, y1_pred, x_pred, y2_pred), fill=(255, 0, 220, 210), width=3)
    draw.line((x_ref, y1_ref, x_ref, y2_ref), fill=(0, 230, 255, 235), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def build_refined_review_html(
    out_dir: Path,
    split_name: str,
    rows: Sequence[Dict[str, object]],
    max_rows: int,
) -> Optional[Path]:
    if not rows:
        return None
    scored = sorted(
        rows,
        key=lambda r: abs(float(r["x_err_after"])) + float(r["y_err"]),
        reverse=True,
    )[: max(0, int(max_rows))]
    preview_dir = out_dir / f"review_{split_name}_refined_previews"
    html_path = out_dir / f"review_{split_name}_refined.html"
    trs: List[str] = []
    for i, r in enumerate(scored, start=1):
        sample_id = str(r.get("sample_id", f"row_{i}"))
        out_img = preview_dir / f"{i:04d}_{_safe_slug(sample_id)}.jpg"
        try:
            draw_refined_preview(r, out_img)
            img_tag = (
                f'<a href="{preview_dir.name}/{out_img.name}" target="_blank">'
                f'<img src="{preview_dir.name}/{out_img.name}" loading="lazy" /></a>'
            )
        except Exception:
            img_tag = "preview_error"
        trs.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{sample_id}</td>"
            f"<td>{abs(float(r['x_err_before'])):.2f}</td>"
            f"<td>{abs(float(r['x_err_after'])):.2f}</td>"
            f"<td>{float(r.get('y_err_before', r['y_err'])):.2f}</td>"
            f"<td>{float(r.get('y_err_after', r['y_err'])):.2f}</td>"
            f"<td>{str(r.get('anchor_source','none'))}</td>"
            "</tr>"
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Refined Scale Review - {split_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 520px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Refined Scale Review - {split_name}</h1>
  <p>Overlay: <b>green=GT</b>, <b>magenta=raw pred</b>, <b>cyan=refined pred (thin)</b></p>
  <table>
    <thead>
      <tr><th>#</th><th>Preview</th><th>sample_id</th><th>x_err_before</th><th>x_err_after</th><th>y_err_before</th><th>y_err_after</th><th>anchor</th></tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def run_refine_for_dir(
    run_dir: Path,
    min_anchor_quality: float,
    min_tick_quality: float,
    max_refine_shift_px: float,
    tick_line_offset_px: float,
    hard_tick_quality: float,
    max_review_rows: int,
    refine_y_from_zero_tick: bool,
    max_y_refine_shift_px: float,
    y_zero_min_tick_quality: float,
) -> Dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = Path(summary["manifest"]).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    video_size_map = build_video_size_map(manifest)

    rows_by_split: Dict[str, List[PredRow]] = {}
    for split in ("train", "val", "test"):
        pred_csv = run_dir / f"{split}_predictions_best.csv"
        rows_by_split[split] = load_rows(pred_csv)

    # Train calibration: robust median offset x_gt - x_anchor.
    train_offsets: List[float] = []
    train_anchor_quality: List[float] = []
    for r in rows_by_split["train"]:
        vsize = video_size_map.get(r.sample_id)
        if vsize is None:
            continue
        vw, vh = vsize
        if vw <= 1 or vh <= 1:
            continue
        gray = np.asarray(Image.open(r.image_path).convert("L"), dtype=np.float32)
        H, W = gray.shape
        sx = float(W) / float(vw)
        sy = float(H) / float(vh)
        x_pred_img = float(r.x_pred) * sx
        y_top_img = float(r.y_top_pred) * sy
        y_bottom_img = float(r.y_bottom_pred) * sy
        det = detect_text_column_anchor(
            gray,
            x_pred_image=x_pred_img,
            y_top_image=y_top_img,
            y_bottom_image=y_bottom_img,
        )
        if det is None or det.quality < min_anchor_quality:
            continue
        x_anchor_video = float(det.x_image * (float(vw) / float(W)))
        train_offsets.append(float(r.x_gt - x_anchor_video))
        train_anchor_quality.append(float(det.quality))

    if len(train_offsets) < 10:
        raise RuntimeError(
            f"Not enough train anchors in {run_dir.name}: {len(train_offsets)} < 10. "
            "Try lowering --min-anchor-quality."
        )

    offset_med = _median(train_offsets)
    offset_p10 = float(np.percentile(np.asarray(train_offsets, dtype=np.float64), 10.0))
    offset_p90 = float(np.percentile(np.asarray(train_offsets, dtype=np.float64), 90.0))

    out_dir = run_dir / "refined_text_column"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_reports: Dict[str, Dict[str, object]] = {}
    review_paths: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        refined_rows, refine_stats = refine_rows(
            rows_by_split[split],
            video_size_map=video_size_map,
            offset_med=offset_med,
            min_anchor_quality=min_anchor_quality,
            min_tick_quality=min_tick_quality,
            max_refine_shift_px=max_refine_shift_px,
            tick_line_offset_px=tick_line_offset_px,
            hard_tick_quality=hard_tick_quality,
            refine_y_from_zero_tick=refine_y_from_zero_tick,
            max_y_refine_shift_px=max_y_refine_shift_px,
            y_zero_min_tick_quality=y_zero_min_tick_quality,
        )
        before = compute_metrics(refined_rows, refined=False)
        after = compute_metrics(refined_rows, refined=True)
        write_csv(out_dir / f"{split}_predictions_refined.csv", refined_rows)
        if split in {"val", "test"}:
            html_path = build_refined_review_html(
                out_dir=out_dir,
                split_name=split,
                rows=refined_rows,
                max_rows=max_review_rows,
            )
            if html_path is not None:
                review_paths[f"{split}_review_html"] = html_path.as_posix()
        split_reports[split] = {
            "before": before,
            "after": after,
            "delta_score_xy": float(after["score_xy"] - before["score_xy"]),
            "delta_mae_x_px": float(after["mae_x_px"] - before["mae_x_px"]),
            "delta_mae_y_px": float(after["mae_y_px"] - before["mae_y_px"]),
            "refine_stats": refine_stats,
        }

    report = {
        "run_dir": run_dir.as_posix(),
        "manifest": manifest.as_posix(),
        "min_anchor_quality": float(min_anchor_quality),
        "min_tick_quality": float(min_tick_quality),
        "max_refine_shift_px": float(max_refine_shift_px),
        "tick_line_offset_px": float(tick_line_offset_px),
        "hard_tick_quality": float(hard_tick_quality),
        "max_review_rows": int(max_review_rows),
        "refine_y_from_zero_tick": bool(refine_y_from_zero_tick),
        "max_y_refine_shift_px": float(max_y_refine_shift_px),
        "y_zero_min_tick_quality": float(y_zero_min_tick_quality),
        "offset_calibration": {
            "train_offsets_n": int(len(train_offsets)),
            "offset_median_px": float(offset_med),
            "offset_p10_px": float(offset_p10),
            "offset_p90_px": float(offset_p90),
            "train_anchor_quality_mean": float(np.mean(train_anchor_quality)),
        },
        "splits": split_reports,
        "artifacts": {
            "train_refined_csv": (out_dir / "train_predictions_refined.csv").as_posix(),
            "val_refined_csv": (out_dir / "val_predictions_refined.csv").as_posix(),
            "test_refined_csv": (out_dir / "test_predictions_refined.csv").as_posix(),
            **review_paths,
        },
    }
    (out_dir / "refine_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refine scale x predictions using text-column anchor.")
    p.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories containing *_predictions_best.csv + summary.json",
    )
    p.add_argument("--min-anchor-quality", type=float, default=0.28)
    p.add_argument("--min-tick-quality", type=float, default=0.30)
    p.add_argument("--max-refine-shift-px", type=float, default=14.0)
    p.add_argument(
        "--tick-line-offset-px",
        type=float,
        default=1.0,
        help="Desired x offset of line with respect to detected tick column (positive = right).",
    )
    p.add_argument(
        "--hard-tick-quality",
        type=float,
        default=0.58,
        help="If tick quality >= threshold, enforce hard snap to tick+offset.",
    )
    p.add_argument("--max-review-rows", type=int, default=140)
    p.add_argument(
        "--refine-y-from-zero-tick",
        action="store_true",
        help="Also refine y_top/y_bottom by anchoring y_top to the zero-tick candidate and y_bottom to last tick.",
    )
    p.add_argument(
        "--max-y-refine-shift-px",
        type=float,
        default=80.0,
        help="Maximum per-side y shift applied by tick-based y refinement.",
    )
    p.add_argument(
        "--y-zero-min-tick-quality",
        type=float,
        default=0.90,
        help="Minimum tick quality required before applying zero-tick y refinement.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    reports = []
    for rd in args.run_dirs:
        rep = run_refine_for_dir(
            rd,
            min_anchor_quality=float(args.min_anchor_quality),
            min_tick_quality=float(args.min_tick_quality),
            max_refine_shift_px=float(args.max_refine_shift_px),
            tick_line_offset_px=float(args.tick_line_offset_px),
            hard_tick_quality=float(args.hard_tick_quality),
            max_review_rows=int(args.max_review_rows),
            refine_y_from_zero_tick=bool(args.refine_y_from_zero_tick),
            max_y_refine_shift_px=float(args.max_y_refine_shift_px),
            y_zero_min_tick_quality=float(args.y_zero_min_tick_quality),
        )
        reports.append(rep)
        print(
            json.dumps(
                {
                    "run_dir": rep["run_dir"],
                    "offset_median_px": rep["offset_calibration"]["offset_median_px"],
                    "val_before": rep["splits"]["val"]["before"]["score_xy"],
                    "val_after": rep["splits"]["val"]["after"]["score_xy"],
                    "test_before": rep["splits"]["test"]["before"]["score_xy"],
                    "test_after": rep["splits"]["test"]["after"]["score_xy"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
