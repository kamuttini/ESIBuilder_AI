#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from rect_red_reusable import (
        build_overlays_by_image,
        clamp_rect01,
        choose_top_segment_global,
        compute_folder_global_rects,
        median_rect,
        normalize_segment,
    )
except ModuleNotFoundError:
    from tools.ultrasound.rect_red_reusable import (
        build_overlays_by_image,
        clamp_rect01,
        choose_top_segment_global,
        compute_folder_global_rects,
        median_rect,
        normalize_segment,
    )


_LONG_HINT_RE = re.compile(r"\b(estesa|esteso|piu lungo|piu estesa|allunga|estendi|orizzont)")
_LEFT_HINT_RE = re.compile(r"\b(sinistra|sx|verso sinistra|a sinistra)\b")
_RIGHT_HINT_RE = re.compile(r"\b(destra|dx|verso destra|a destra)\b")
_OUTSIDE_ANY_RE = re.compile(r"\b(fuori|esce|uscire|oltre|out)\b")
_OUTSIDE_RECT_RE = re.compile(r"\b(rect|rettangolo|box)\b")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)


def _normalize_orientation_label(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if txt == "su":
        return "su"
    if txt == "giu":
        return "giu"
    return "unknown"


def rect_norm_from_tlbr(
    *,
    top: Any,
    left: Any,
    bottom: Any,
    right: Any,
    width: Any,
    height: Any,
) -> Optional[Dict[str, float]]:
    w = _safe_float(width, 0.0)
    h = _safe_float(height, 0.0)
    if w <= 1.0 or h <= 1.0:
        return None

    t = _safe_float(top, float("nan"))
    l = _safe_float(left, float("nan"))
    b = _safe_float(bottom, float("nan"))
    r = _safe_float(right, float("nan"))
    if t != t or l != l or b != b or r != r:
        return None

    y1 = min(t, b)
    y2 = max(t, b)
    x1 = min(l, r)
    x2 = max(l, r)
    if y2 <= y1 or x2 <= x1:
        return None

    rect = clamp_rect01(
        {
            "x": x1 / w,
            "y": y1 / h,
            "w": (x2 - x1) / w,
            "h": (y2 - y1) / h,
        }
    )
    if rect is None:
        return None
    return {
        "x": float(rect.x),
        "y": float(rect.y),
        "w": float(rect.w),
        "h": float(rect.h),
    }


def rect_norm_to_tlbr(
    rect_norm: Mapping[str, Any],
    *,
    width: Any,
    height: Any,
) -> Optional[Dict[str, int]]:
    rect = clamp_rect01(rect_norm)
    if rect is None:
        return None
    w = _safe_int(width, 0)
    h = _safe_int(height, 0)
    if w <= 1 or h <= 1:
        return None

    left = int(round(float(rect.x) * float(w)))
    top = int(round(float(rect.y) * float(h)))
    right = int(round(float(rect.x + rect.w) * float(w)))
    bottom = int(round(float(rect.y + rect.h) * float(h)))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))

    return {
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
    }


def _normalize_note_text(value: Any) -> str:
    txt = str(value or "").lower().strip()
    return txt


def _note_hints_from_text(note_text: Any) -> Dict[str, float | bool]:
    txt = _normalize_note_text(note_text)
    if not txt:
        return {
            "extra_left": 0.0,
            "extra_right": 0.0,
            "extra_y": 0.0,
            "allow_outside": False,
        }
    long_hint = _LONG_HINT_RE.search(txt) is not None
    left_hint = _LEFT_HINT_RE.search(txt) is not None
    right_hint = _RIGHT_HINT_RE.search(txt) is not None
    outside_hint = (_OUTSIDE_ANY_RE.search(txt) is not None) and (_OUTSIDE_RECT_RE.search(txt) is not None)
    return {
        "extra_left": (0.04 if long_hint else 0.0) + (0.08 if left_hint else 0.0) + (0.10 if outside_hint else 0.0),
        "extra_right": (0.04 if long_hint else 0.0) + (0.08 if right_hint else 0.0) + (0.10 if outside_hint else 0.0),
        "extra_y": 0.05 if outside_hint else 0.0,
        "allow_outside": bool(outside_hint or long_hint),
    }


def _row_continuous_extents(
    row_gray: np.ndarray,
    *,
    row_start_x: int,
    thr: float,
    max_gray: float,
    max_dark_gap: int,
) -> Dict[str, float]:
    is_white = (row_gray >= float(thr)) & (row_gray <= float(max_gray))

    best_first = -1
    best_last = -1
    best_count = 0
    best_sum = 0.0
    best_span = 0
    best_max_gap_seen = 0

    run_first = -1
    run_last_white = -1
    run_count = 0
    run_sum = 0.0
    dark_gap = 0
    run_max_gap_seen = 0

    def flush_run() -> None:
        nonlocal best_first, best_last, best_count, best_sum, best_span, best_max_gap_seen
        if run_first < 0 or run_last_white < run_first:
            return
        span = run_last_white - run_first + 1
        if span <= 0:
            return
        if span > best_span:
            best_span = span
            best_first = run_first
            best_last = run_last_white
            best_count = run_count
            best_sum = run_sum
            best_max_gap_seen = run_max_gap_seen

    def reset_run() -> None:
        nonlocal run_first, run_last_white, run_count, run_sum, dark_gap, run_max_gap_seen
        run_first = -1
        run_last_white = -1
        run_count = 0
        run_sum = 0.0
        dark_gap = 0
        run_max_gap_seen = 0

    for offset, white in enumerate(is_white.tolist()):
        x = int(row_start_x + offset)
        g = float(row_gray[offset])
        if white:
            if run_first < 0:
                run_first = x
                run_last_white = x
                run_count = 1
                run_sum = g
                dark_gap = 0
                run_max_gap_seen = 0
                continue

            if dark_gap > max_dark_gap:
                flush_run()
                reset_run()
                run_first = x
                run_last_white = x
                run_count = 1
                run_sum = g
                continue

            if dark_gap > 0:
                run_max_gap_seen = max(run_max_gap_seen, dark_gap)
            run_last_white = x
            run_count += 1
            run_sum += g
            dark_gap = 0
            continue

        if run_first >= 0:
            dark_gap += 1
            if dark_gap > max_dark_gap:
                flush_run()
                reset_run()

    flush_run()

    if best_first < 0 or best_last < best_first:
        return {
            "first": -1.0,
            "last": -1.0,
            "span": 0.0,
            "count": 0.0,
            "mean": 0.0,
            "density": 0.0,
            "max_gap": 0.0,
        }

    span = float(best_last - best_first + 1)
    density = float(best_count / max(1.0, span))
    return {
        "first": float(best_first),
        "last": float(best_last),
        "span": float(span),
        "count": float(best_count),
        "mean": float(best_sum / max(1, best_count)),
        "density": float(density),
        "max_gap": float(best_max_gap_seen),
    }


def estimate_horizontal_segment_from_path(
    *,
    image_path: Path,
    rect_norm: Optional[Mapping[str, Any]],
    orientation_label: str,
    bright_thr: float = 70.0,
    note_text: str = "",
) -> Optional[Dict[str, Any]]:
    path = Path(image_path).expanduser()
    if not path.is_file():
        return None

    with Image.open(path) as img:
        src = img.convert("RGB")
        src_w, src_h = src.size
        if src_w < 4 or src_h < 4:
            return None

        scale = min(1.0, 1024.0 / float(max(src_w, src_h)))
        w = max(96, int(round(float(src_w) * scale)))
        h = max(96, int(round(float(src_h) * scale)))
        if (w, h) != (src_w, src_h):
            src = src.resize((w, h), Image.Resampling.BILINEAR)
        arr = np.asarray(src, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    gray = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])
    if gray.shape[0] != h or gray.shape[1] != w:
        return None

    hints = _note_hints_from_text(note_text)
    prefer_high = _normalize_orientation_label(orientation_label) == "su"

    rx = 0
    ry = 0
    rw = int(w)
    rh = int(h)
    rect_pix: Optional[Dict[str, int]] = None
    rr = clamp_rect01(rect_norm) if rect_norm is not None else None

    if rr is not None:
        rx0 = max(0, min(w - 2, int(np.floor(float(rr.x) * float(w)))))
        ry0 = max(0, min(h - 2, int(np.floor(float(rr.y) * float(h)))))
        rw0 = max(2, min(w - rx0, int(np.ceil(float(rr.w) * float(w)))))
        rh0 = max(2, min(h - ry0, int(np.ceil(float(rr.h) * float(h)))))
        rect_pix = {
            "left": int(rx0),
            "right": int(rx0 + rw0 - 1),
            "top": int(ry0),
            "bottom": int(ry0 + rh0 - 1),
        }

        left_expand = int(round((0.10 + float(hints["extra_left"])) * float(rw0)))
        right_expand = int(round((0.10 + float(hints["extra_right"])) * float(rw0)))
        y_expand = int(round((0.03 + float(hints["extra_y"])) * float(rh0)))

        x_l = max(0, rx0 - max(0, left_expand))
        x_r = min(w - 1, rx0 + rw0 - 1 + max(0, right_expand))
        y_t = max(0, ry0 - max(0, y_expand))
        y_b = min(h - 1, ry0 + rh0 - 1 + max(0, y_expand))

        rx = int(x_l)
        ry = int(y_t)
        rw = max(2, int(x_r - x_l + 1))
        rh = max(2, int(y_b - y_t + 1))

    if rw < 24 or rh < 24:
        return None

    pad_y = max(0, int(np.floor(0.05 * float(rh))))
    y0 = max(0, int(ry + pad_y))
    y1 = min(h - 1, int(ry + rh - 1 - pad_y))
    if y1 <= y0:
        return None

    bright_base = _clamp(_safe_float(bright_thr, 70.0), 0.0, 255.0)
    thr_hi = _clamp(max(bright_base, 52.0), 20.0, 245.0)
    thr_lo = _clamp(max(bright_base - 18.0, 36.0), 16.0, 235.0)
    min_span_ref = max(20, int(np.floor(float(rr.w) * float(w) * 0.75))) if rr is not None else int(rw)
    min_span = max(10, int(np.floor(0.20 * float(min_span_ref))))
    min_white_count = max(8, int(np.floor(0.05 * float(rw))))
    max_dark_gap = max(2, min(7, int(round(0.008 * float(rw)))))

    best: Optional[Dict[str, float]] = None
    best_score = -1e12
    candidates: List[Dict[str, float]] = []
    y_center = float(ry) + (0.62 * float(rh))

    for y in range(y0, y1 + 1):
        row = gray[y, rx : (rx + rw)]
        if row.size <= 0:
            continue
        hi = _row_continuous_extents(
            row,
            row_start_x=rx,
            thr=thr_hi,
            max_gray=250.0,
            max_dark_gap=max_dark_gap,
        )
        lo = _row_continuous_extents(
            row,
            row_start_x=rx,
            thr=thr_lo,
            max_gray=255.0,
            max_dark_gap=max_dark_gap,
        )

        hi_ok = (hi["span"] >= float(min_span)) and (hi["count"] >= float(min_white_count)) and (hi["density"] >= 0.18)
        lo_ok = (lo["span"] >= float(min_span)) and (lo["count"] >= float(min_white_count)) and (lo["density"] >= 0.14)
        chosen = hi if hi_ok else (lo if lo_ok else None)
        if chosen is None:
            continue

        overlap_px = 0.0
        if rect_pix is not None:
            y_tol = max(1, int(round(0.004 * float(h))))
            inside_y = (y >= (rect_pix["top"] - y_tol)) and (y <= (rect_pix["bottom"] + y_tol))
            overlap_px = float(
                min(int(chosen["last"]), rect_pix["right"]) - max(int(chosen["first"]), rect_pix["left"]) + 1
            )
            if (not inside_y) or (overlap_px <= 0.0):
                continue

        y_penalty = abs(float(y) - y_center) / float(max(1, rh))
        y_penalty_w = 18.0 if bool(hints["allow_outside"]) else 24.0
        overlap_bonus = (0.12 * overlap_px) if rect_pix is not None else 0.0
        score = float(chosen["span"]) + (0.08 * float(chosen["mean"])) + (0.12 * float(chosen["count"])) + overlap_bonus - (y_penalty_w * y_penalty)

        cand = {
            "y": float(y),
            "start": float(chosen["first"]),
            "end": float(chosen["last"]),
            "length": float(chosen["span"]),
            "max_gap": float(chosen["max_gap"]),
            "overlap_px": float(overlap_px),
            "score": float(score),
        }
        candidates.append(cand)
        if float(score) > best_score:
            best_score = float(score)
            best = cand

    if best is None and not candidates:
        return None

    def overlap_len(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return float(min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1.0)

    selection_policy = "best_score_fallback"
    if candidates:
        max_span = max(float(c["length"]) for c in candidates)
        span_near_top = max(float(min_span), float(np.floor(max_span * 0.92)))
        attach_dy = 2.0
        attach_min_overlap = max(2.0, float(np.floor(0.03 * float(rw))))

        def _sort_key(item: Mapping[str, float]) -> Tuple[float, float, float]:
            yv = float(item["y"])
            if prefer_high:
                y_key = yv
            else:
                y_key = -yv
            return (
                y_key,
                -float(item["length"]),
                -float(item["score"]),
            )

        sorted_by_vertical = sorted(candidates, key=_sort_key)

        attached: List[Dict[str, float]] = []
        for c in sorted_by_vertical:
            if float(c["length"]) < span_near_top:
                continue
            ok_attach = False
            for u in candidates:
                if prefer_high:
                    if float(u["y"]) <= float(c["y"]):
                        continue
                    if (float(u["y"]) - float(c["y"])) > attach_dy:
                        continue
                else:
                    if float(u["y"]) >= float(c["y"]):
                        continue
                    if (float(c["y"]) - float(u["y"])) > attach_dy:
                        continue
                if overlap_len(c, u) >= attach_min_overlap:
                    ok_attach = True
                    break
            if ok_attach:
                attached.append(c)

        if attached:
            best = attached[0]
            selection_policy = "highest_attached_nearmax" if prefer_high else "lowest_attached_nearmax"
        else:
            near_top = [c for c in sorted_by_vertical if float(c["length"]) >= span_near_top]
            if near_top:
                best = near_top[0]
                selection_policy = "highest_nearmax" if prefer_high else "lowest_nearmax"

    if best is None:
        return None

    x1 = _clamp(float(best["start"]) / float(w), 0.0, 1.0)
    x2 = _clamp((float(best["end"]) + 1.0) / float(w), 0.0, 1.0)
    if x2 <= x1:
        return None
    yv = _clamp((float(best["y"]) + 0.5) / float(h), 0.0, 1.0)
    length_norm = float(x2 - x1)

    return {
        "x1": float(x1),
        "x2": float(x2),
        "y": float(yv),
        "length_norm": float(length_norm),
        "length_px": float(length_norm * float(src_w)),
        "source": "auto",
        "method": "row_contiguous_white_extrema",
        "selection_policy": selection_policy,
        "vertical_preference": "high" if prefer_high else "low",
        "max_dark_gap_seen": float(best.get("max_gap", 0.0)),
        "max_dark_gap_limit": float(max_dark_gap),
        "rect_overlap_px": float(best.get("overlap_px", 0.0)),
    }


def _segment_ref_to_dict(seg_ref: Any) -> Dict[str, Any]:
    if seg_ref is None:
        return {}
    seg = getattr(seg_ref, "segment", None)
    if seg is None:
        return {}
    return {
        "image_id": str(getattr(seg_ref, "image_id", "") or ""),
        "len_px": float(getattr(seg_ref, "len_px", 0.0) or 0.0),
        "len_norm": float(getattr(seg_ref, "len_norm", 0.0) or 0.0),
        "image_width": float(getattr(seg_ref, "image_width", 0.0) or 0.0),
        "segment": {
            "x1": float(getattr(seg, "x1", 0.0) or 0.0),
            "x2": float(getattr(seg, "x2", 0.0) or 0.0),
            "y": float(getattr(seg, "y", 0.0) or 0.0),
            "length_norm": float(getattr(seg, "length_norm", 0.0) or 0.0),
            "length_px": float(getattr(seg, "length_px", 0.0) or 0.0) if getattr(seg, "length_px", None) is not None else None,
            "source": str(getattr(seg, "source", "") or ""),
        },
    }


def _is_better_segment(a: Any, b: Any) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    a_px = float(getattr(a, "len_px", 0.0) or 0.0)
    b_px = float(getattr(b, "len_px", 0.0) or 0.0)
    if a_px > (b_px + 1e-6):
        return True
    a_norm = float(getattr(a, "len_norm", 0.0) or 0.0)
    b_norm = float(getattr(b, "len_norm", 0.0) or 0.0)
    if abs(a_px - b_px) <= 1e-6 and a_norm > (b_norm + 1e-9):
        return True
    return False


def _preferred_output_size(
    records_out: Sequence[Mapping[str, Any]],
    output_width: int,
    output_height: int,
) -> Tuple[int, int]:
    ow = int(output_width or 0)
    oh = int(output_height or 0)
    if ow > 1 and oh > 1:
        return ow, oh

    wh_counter: Counter[Tuple[int, int]] = Counter()
    for rec in records_out:
        w = _safe_int(rec.get("image_width", 0), 0)
        h = _safe_int(rec.get("image_height", 0), 0)
        if w > 1 and h > 1:
            wh_counter[(w, h)] += 1
    if wh_counter:
        (w, h), _ = wh_counter.most_common(1)[0]
        return int(w), int(h)
    return 0, 0


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    qq = _clamp(float(q), 0.0, 1.0)
    pos = qq * float(len(clean) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - float(lo)
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _norm_rect_iou(a: Any, b: Any) -> float:
    ar = clamp_rect01(a)
    br = clamp_rect01(b)
    if ar is None or br is None:
        return 0.0
    ax1 = float(ar.x)
    ay1 = float(ar.y)
    ax2 = float(ar.x + ar.w)
    ay2 = float(ar.y + ar.h)
    bx1 = float(br.x)
    by1 = float(br.y)
    bx2 = float(br.x + br.w)
    by2 = float(br.y + br.h)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-8:
        return 0.0
    return float(inter_area / union)


def _validate_line11_red_candidate(
    *,
    base_rect_norm: Mapping[str, Any],
    red_rect_norm: Mapping[str, Any],
    rects: Sequence[Any],
    parsed_records: Sequence[Mapping[str, Any]],
    top_segment: Any,
) -> Dict[str, Any]:
    base = clamp_rect01(base_rect_norm)
    red = clamp_rect01(red_rect_norm)
    widths = [float(r.w) for r in rects if r is not None and float(r.w) > 0.0]
    median_w = _quantile(widths, 0.5)
    q1_w = _quantile(widths, 0.25)
    q3_w = _quantile(widths, 0.75)
    iqr_w = (
        float(q3_w - q1_w)
        if q1_w is not None and q3_w is not None
        else None
    )
    iqr_ratio = (
        float(iqr_w / max(float(median_w), 1e-8))
        if iqr_w is not None and median_w is not None
        else None
    )

    top_id = str(getattr(top_segment, "image_id", "") or "").strip() if top_segment is not None else ""
    source_rect = None
    if top_id:
        for item in parsed_records:
            if str(item.get("image_id", "") or "") == top_id:
                source_rect = clamp_rect01(item.get("pred_rect_norm"))
                break

    source_iou = _norm_rect_iou(source_rect, base) if source_rect is not None and base is not None else None
    base_iou = _norm_rect_iou(base, red) if base is not None and red is not None else 0.0
    red_vs_base = (
        float(red.w / max(float(base.w), 1e-8))
        if base is not None and red is not None
        else None
    )
    red_vs_median = (
        float(red.w / max(float(median_w), 1e-8))
        if red is not None and median_w is not None
        else None
    )

    min_source_iou = 0.70
    max_stable_iqr_ratio = 0.035
    min_width_ratio_on_stable = 0.90
    min_base_iou_on_stable = 0.88
    stable_rects = (
        len(widths) >= 8
        and iqr_ratio is not None
        and iqr_ratio <= max_stable_iqr_ratio
    )

    accepted = True
    reason = "ok"
    if source_iou is not None and source_iou < min_source_iou:
        accepted = False
        reason = "top_segment_source_rect_outlier"
    elif (
        stable_rects
        and red_vs_median is not None
        and red_vs_base is not None
        and (red_vs_median < min_width_ratio_on_stable or red_vs_base < min_width_ratio_on_stable)
        and base_iou < min_base_iou_on_stable
    ):
        accepted = False
        reason = "red_rect_shrinks_stable_per_image_median"

    return {
        "accepted": bool(accepted),
        "reason": reason,
        "top_segment_image_id": top_id,
        "top_segment_source_iou_vs_base": source_iou,
        "base_iou_vs_red": float(base_iou),
        "base_width_norm": float(base.w) if base is not None else None,
        "red_width_norm": float(red.w) if red is not None else None,
        "red_width_vs_base": red_vs_base,
        "per_image_width_median_norm": median_w,
        "per_image_width_q1_norm": q1_w,
        "per_image_width_q3_norm": q3_w,
        "per_image_width_iqr_norm": iqr_w,
        "per_image_width_iqr_ratio": iqr_ratio,
        "red_width_vs_per_image_median": red_vs_median,
        "per_image_rects_stable": bool(stable_rects),
        "thresholds": {
            "min_source_iou": min_source_iou,
            "max_stable_iqr_ratio": max_stable_iqr_ratio,
            "min_width_ratio_on_stable": min_width_ratio_on_stable,
            "min_base_iou_on_stable": min_base_iou_on_stable,
        },
    }


def compute_rect_red_pipeline(
    *,
    records: Sequence[Mapping[str, Any]],
    output_width: int = 0,
    output_height: int = 0,
    margin_pct: float = 5.0,
    bright_thr: float = 70.0,
    detect_segments: bool = True,
) -> Dict[str, Any]:
    parsed: List[Dict[str, Any]] = []
    reusable_records: List[Dict[str, Any]] = []
    per_image_rect_by_image: Dict[str, Dict[str, float]] = {}

    for idx, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            continue

        image_id = str(
            raw.get("image_id")
            or raw.get("id")
            or raw.get("image_rel")
            or raw.get("image_path")
            or f"image_{idx:05d}"
        ).strip()
        if not image_id:
            image_id = f"image_{idx:05d}"

        image_path_txt = str(raw.get("image_path") or raw.get("abs_path") or "").strip()
        image_path = Path(image_path_txt).expanduser() if image_path_txt else None
        image_width = _safe_int(raw.get("image_width", raw.get("width", 0)), 0)
        image_height = _safe_int(raw.get("image_height", raw.get("height", 0)), 0)

        rect_norm = clamp_rect01(raw.get("pred_rect_norm") or raw.get("pred_rect"))
        if rect_norm is None:
            rect_tlbr = raw.get("pred_rect_tlbr")
            if isinstance(rect_tlbr, Mapping):
                rect_norm_dict = rect_norm_from_tlbr(
                    top=rect_tlbr.get("top"),
                    left=rect_tlbr.get("left"),
                    bottom=rect_tlbr.get("bottom"),
                    right=rect_tlbr.get("right"),
                    width=image_width,
                    height=image_height,
                )
                rect_norm = clamp_rect01(rect_norm_dict) if rect_norm_dict is not None else None
        if rect_norm is None:
            continue

        orientation_label = _normalize_orientation_label(
            raw.get("orientation_label")
            or raw.get("pred_label")
            or raw.get("label")
            or raw.get("group_name")
        )

        auto_segment = normalize_segment(raw.get("auto_segment"))
        if auto_segment is None and detect_segments and image_path is not None and image_path.is_file():
            seg_auto = estimate_horizontal_segment_from_path(
                image_path=image_path,
                rect_norm={
                    "x": float(rect_norm.x),
                    "y": float(rect_norm.y),
                    "w": float(rect_norm.w),
                    "h": float(rect_norm.h),
                },
                orientation_label=orientation_label,
                bright_thr=float(bright_thr),
                note_text=str(raw.get("note_text", "") or ""),
            )
            if seg_auto is not None:
                auto_segment = normalize_segment(seg_auto)

        rec_parsed: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path.as_posix() if image_path is not None else "",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "orientation_label": orientation_label,
            "pred_rect_norm": {
                "x": float(rect_norm.x),
                "y": float(rect_norm.y),
                "w": float(rect_norm.w),
                "h": float(rect_norm.h),
            },
            "auto_segment": {
                "x1": float(auto_segment.x1),
                "x2": float(auto_segment.x2),
                "y": float(auto_segment.y),
                "length_norm": float(auto_segment.length_norm),
                "length_px": float(auto_segment.length_px) if auto_segment.length_px is not None else None,
                "source": str(auto_segment.source or ""),
            } if auto_segment is not None else None,
        }
        parsed.append(rec_parsed)

        reusable_rec: Dict[str, Any] = {
            "id": image_id,
            "image_id": image_id,
            "group_name": orientation_label,
            "label": orientation_label,
            "width": int(image_width),
            "height": int(image_height),
        }
        if auto_segment is not None:
            reusable_rec["auto_segment"] = {
                "x1": float(auto_segment.x1),
                "x2": float(auto_segment.x2),
                "y": float(auto_segment.y),
                "length_norm": float(auto_segment.length_norm),
                "length_px": float(auto_segment.length_px) if auto_segment.length_px is not None else None,
                "source": str(auto_segment.source or ""),
            }
        reusable_records.append(reusable_rec)

        per_image_rect_by_image[image_id] = {
            "x": float(rect_norm.x),
            "y": float(rect_norm.y),
            "w": float(rect_norm.w),
            "h": float(rect_norm.h),
        }

    if not parsed:
        return {
            "available": False,
            "error": "no_valid_records",
            "records_total": 0,
            "records_with_segment": 0,
        }

    rects = [clamp_rect01(item["pred_rect_norm"]) for item in parsed]
    rects = [r for r in rects if r is not None]
    base_rect = median_rect(rects)
    if base_rect is None:
        return {
            "available": False,
            "error": "base_rect_unavailable",
            "records_total": len(parsed),
            "records_with_segment": 0,
        }

    group_counts: Dict[str, int] = {
        "su": 0,
        "giu": 0,
        "unknown": 0,
    }
    for item in parsed:
        lab = _normalize_orientation_label(item.get("orientation_label"))
        group_counts[lab] = int(group_counts.get(lab, 0) + 1)

    groups_present = {str(item.get("orientation_label", "unknown")) for item in parsed}
    if not groups_present:
        groups_present = {"unknown"}

    base_rect_by_group = {
        g: {
            "x": float(base_rect.x),
            "y": float(base_rect.y),
            "w": float(base_rect.w),
            "h": float(base_rect.h),
        }
        for g in groups_present
    }
    base_meta_by_group = {
        g: {
            "source": "median_pred_rect_per_image",
            "n_explicit": 0,
            "n_pred": int(len(rects)),
        }
        for g in groups_present
    }

    records_su = [r for r in reusable_records if _normalize_orientation_label(r.get("label")) == "su"]
    records_giu = [r for r in reusable_records if _normalize_orientation_label(r.get("label")) == "giu"]
    top_su = choose_top_segment_global(records_su) if records_su else None
    top_giu = choose_top_segment_global(records_giu) if records_giu else None
    top_global = choose_top_segment_global(reusable_records)

    winner_group = "unknown"
    winner_reason = "no_segment_available"
    winner_top = None

    if top_su is not None and top_giu is not None:
        if _is_better_segment(top_su, top_giu):
            winner_group = "su"
            winner_top = top_su
        else:
            winner_group = "giu"
            winner_top = top_giu
        winner_reason = "longest_segment_between_su_giu"
    elif top_su is not None:
        winner_group = "su"
        winner_top = top_su
        winner_reason = "only_su_has_segment"
    elif top_giu is not None:
        winner_group = "giu"
        winner_top = top_giu
        winner_reason = "only_giu_has_segment"
    elif top_global is not None:
        winner_top = top_global
        top_id = str(getattr(top_global, "image_id", "") or "")
        label_by_id = {
            str(it.get("image_id", "")): _normalize_orientation_label(it.get("orientation_label"))
            for it in parsed
        }
        winner_group = label_by_id.get(top_id, "unknown")
        winner_reason = "fallback_global_top_segment"

    seg_hint = str(getattr(winner_top, "image_id", "") or "") if winner_top is not None else ""
    if not seg_hint and top_global is not None:
        seg_hint = str(getattr(top_global, "image_id", "") or "")

    computation = compute_folder_global_rects(
        records=reusable_records,
        base_rect_by_group=base_rect_by_group,
        base_meta_by_group=base_meta_by_group,
        margin_pct=float(margin_pct),
        seg_image_id_hint=seg_hint or None,
    )
    if computation is None:
        return {
            "available": False,
            "error": "compute_folder_global_rects_failed",
            "records_total": len(parsed),
            "records_with_segment": 0,
        }

    overlays_by_image = build_overlays_by_image(
        records=reusable_records,
        computation=computation,
        per_image_rect_by_image=per_image_rect_by_image,
    )

    out_w, out_h = _preferred_output_size(
        records_out=parsed,
        output_width=int(output_width),
        output_height=int(output_height),
    )

    base_rect_norm_dict = {
        "x": float(computation.unified_base_rect.x),
        "y": float(computation.unified_base_rect.y),
        "w": float(computation.unified_base_rect.w),
        "h": float(computation.unified_base_rect.h),
    }
    red_rect_norm_dict = {
        "x": float(computation.adjusted_red_rect.x),
        "y": float(computation.adjusted_red_rect.y),
        "w": float(computation.adjusted_red_rect.w),
        "h": float(computation.adjusted_red_rect.h),
    }

    base_tlbr = rect_norm_to_tlbr(base_rect_norm_dict, width=out_w, height=out_h)
    red_tlbr = rect_norm_to_tlbr(red_rect_norm_dict, width=out_w, height=out_h)

    line11_base = ""
    line11_red = ""
    if base_tlbr is not None:
        line11_base = f"{base_tlbr['top']}|{base_tlbr['left']}|{base_tlbr['bottom']}|{base_tlbr['right']}|"
    if red_tlbr is not None:
        line11_red = f"{red_tlbr['top']}|{red_tlbr['left']}|{red_tlbr['bottom']}|{red_tlbr['right']}|"

    red_validation = _validate_line11_red_candidate(
        base_rect_norm=base_rect_norm_dict,
        red_rect_norm=red_rect_norm_dict,
        rects=rects,
        parsed_records=parsed,
        top_segment=computation.top_segment,
    )

    out_records: List[Dict[str, Any]] = []
    for item in parsed:
        image_id = str(item.get("image_id", "") or "")
        pred_rect_norm = item.get("pred_rect_norm", {})
        w = _safe_int(item.get("image_width", 0), 0)
        h = _safe_int(item.get("image_height", 0), 0)
        pred_rect_tlbr = rect_norm_to_tlbr(pred_rect_norm, width=w, height=h)
        out_records.append(
            {
                "image_id": image_id,
                "image_path": str(item.get("image_path", "") or ""),
                "image_width": int(w),
                "image_height": int(h),
                "orientation_label": _normalize_orientation_label(item.get("orientation_label")),
                "pred_rect_norm": pred_rect_norm,
                "pred_rect_tlbr": pred_rect_tlbr or {},
                "auto_segment": item.get("auto_segment") or {},
                "overlay": overlays_by_image.get(image_id, {}),
            }
        )

    records_with_segment = sum(
        1
        for item in out_records
        if isinstance(item.get("auto_segment"), Mapping)
        and normalize_segment(item.get("auto_segment")) is not None
    )

    return {
        "available": True,
        "error": "",
        "algorithm": "segment_top_red_rect",
        "margin_pct": float(_clamp(float(margin_pct), 0.0, 30.0)),
        "bright_thr": float(_clamp(float(bright_thr), 0.0, 255.0)),
        "output_size": {
            "width": int(out_w),
            "height": int(out_h),
        },
        "line11_base": {
            "text": line11_base,
            "rect_tlbr": base_tlbr or {},
            "rect_norm": base_rect_norm_dict,
        },
        "line11_red": {
            "text": line11_red,
            "rect_tlbr": red_tlbr or {},
            "rect_norm": red_rect_norm_dict,
        },
        "group_counts": {
            "su": int(group_counts.get("su", 0)),
            "giu": int(group_counts.get("giu", 0)),
            "unknown": int(group_counts.get("unknown", 0)),
        },
        "winner_group": winner_group,
        "winner_reason": winner_reason,
        "top_segment_selected": _segment_ref_to_dict(computation.top_segment),
        "top_segment_su": _segment_ref_to_dict(top_su),
        "top_segment_giu": _segment_ref_to_dict(top_giu),
        "top_segment_global": _segment_ref_to_dict(top_global),
        "line11_red_accepted": bool(red_validation.get("accepted", True)),
        "line11_red_reject_reason": (
            ""
            if bool(red_validation.get("accepted", True))
            else str(red_validation.get("reason", "") or "rect_red_rejected")
        ),
        "line11_red_validation": red_validation,
        "records_total": int(len(out_records)),
        "records_with_segment": int(records_with_segment),
        "records": out_records,
    }
