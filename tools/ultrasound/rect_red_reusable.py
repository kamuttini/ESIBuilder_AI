#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Segment:
    x1: float
    x2: float
    y: float
    length_norm: float
    length_px: float | None
    source: str = ""


@dataclass(frozen=True)
class TopSegmentRef:
    image_id: str
    segment: Segment
    len_px: float
    len_norm: float
    image_width: float


@dataclass(frozen=True)
class GlobalRectComputation:
    unified_base_rect: Rect
    adjusted_red_rect: Rect
    top_segment: TopSegmentRef | None
    margin_pct: float
    margin_norm: float
    base_by_group: dict[str, Rect]
    red_by_group: dict[str, Rect]
    meta_by_group: dict[str, dict[str, Any]]

    def to_jsonable(self) -> dict[str, Any]:
        top: dict[str, Any] | None = None
        if self.top_segment is not None:
            top = {
                "image_id": self.top_segment.image_id,
                "segment": asdict(self.top_segment.segment),
                "len_px": self.top_segment.len_px,
                "len_norm": self.top_segment.len_norm,
                "image_width": self.top_segment.image_width,
            }
        return {
            "unified_base_rect": asdict(self.unified_base_rect),
            "adjusted_red_rect": asdict(self.adjusted_red_rect),
            "top_segment": top,
            "margin_pct": self.margin_pct,
            "margin_norm": self.margin_norm,
            "base_by_group": {k: asdict(v) for k, v in self.base_by_group.items()},
            "red_by_group": {k: asdict(v) for k, v in self.red_by_group.items()},
            "meta_by_group": self.meta_by_group,
        }


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _to_float(v: Any) -> float | None:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if n != n:  # NaN
        return None
    return n


def clamp_rect01(rect: Mapping[str, Any] | Rect | None) -> Rect | None:
    if rect is None:
        return None
    x = _to_float(rect.x if isinstance(rect, Rect) else rect.get("x"))
    y = _to_float(rect.y if isinstance(rect, Rect) else rect.get("y"))
    w = _to_float(rect.w if isinstance(rect, Rect) else rect.get("w"))
    h = _to_float(rect.h if isinstance(rect, Rect) else rect.get("h"))
    if x is None or y is None or w is None or h is None:
        return None
    xx = clamp(x, 0.0, 1.0)
    yy = clamp(y, 0.0, 1.0)
    ww = clamp(w, 0.001, 1.0)
    hh = clamp(h, 0.001, 1.0)
    if xx + ww > 1.0:
        ww = 1.0 - xx
    if yy + hh > 1.0:
        hh = 1.0 - yy
    if ww <= 0.0 or hh <= 0.0:
        return None
    return Rect(x=xx, y=yy, w=ww, h=hh)


def rect_from_edges(
    left: Any,
    top: Any,
    right: Any,
    bottom: Any,
    width: Any = None,
    height: Any = None,
) -> Rect | None:
    l = _to_float(left)
    t = _to_float(top)
    r = _to_float(right)
    b = _to_float(bottom)
    if l is None or t is None or r is None or b is None:
        return None
    x1, x2 = (l, r) if l <= r else (r, l)
    y1, y2 = (t, b) if t <= b else (b, t)

    w = _to_float(width)
    h = _to_float(height)
    if w is not None and h is not None and w > 1.0 and h > 1.0:
        return clamp_rect01(
            {
                "x": x1 / w,
                "y": y1 / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
            }
        )

    if x1 >= -0.1 and y1 >= -0.1 and x2 <= 1.5 and y2 <= 1.5:
        return clamp_rect01({"x": x1, "y": y1, "w": (x2 - x1), "h": (y2 - y1)})
    return None


def pred_rect_from_row(row: Mapping[str, Any]) -> Rect | None:
    width = row.get("width", row.get("image_width", row.get("w")))
    height = row.get("height", row.get("image_height", row.get("h")))

    rect = rect_from_edges(
        row.get("pred_x1"),
        row.get("pred_y1"),
        row.get("pred_x2"),
        row.get("pred_y2"),
        width,
        height,
    )
    if rect is not None:
        return rect

    rect = rect_from_edges(
        row.get("pred_left"),
        row.get("pred_top"),
        row.get("pred_right"),
        row.get("pred_bottom"),
        width,
        height,
    )
    if rect is not None:
        return rect

    return rect_from_edges(
        row.get("pred_left_norm"),
        row.get("pred_top_norm"),
        row.get("pred_right_norm"),
        row.get("pred_bottom_norm"),
        1,
        1,
    )


def global_rect_from_row(row: Mapping[str, Any]) -> Rect | None:
    width = row.get("width", row.get("image_width", row.get("w")))
    height = row.get("height", row.get("image_height", row.get("h")))

    rect = rect_from_edges(
        row.get("global_x1"),
        row.get("global_y1"),
        row.get("global_x2"),
        row.get("global_y2"),
        width,
        height,
    )
    if rect is not None:
        return rect

    rect = rect_from_edges(
        row.get("global_left"),
        row.get("global_top"),
        row.get("global_right"),
        row.get("global_bottom"),
        width,
        height,
    )
    if rect is not None:
        return rect

    return rect_from_edges(
        row.get("global_left_norm"),
        row.get("global_top_norm"),
        row.get("global_right_norm"),
        row.get("global_bottom_norm"),
        1,
        1,
    )


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    m = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[m])
    return 0.5 * (float(ordered[m - 1]) + float(ordered[m]))


def median_rect(rects: Sequence[Rect]) -> Rect | None:
    if not rects:
        return None
    x = _median([r.x for r in rects])
    y = _median([r.y for r in rects])
    w = _median([r.w for r in rects])
    h = _median([r.h for r in rects])
    if x is None or y is None or w is None or h is None:
        return None
    return clamp_rect01({"x": x, "y": y, "w": w, "h": h})


def normalize_segment(seg: Mapping[str, Any] | Segment | None) -> Segment | None:
    if seg is None:
        return None
    x1 = _to_float(seg.x1 if isinstance(seg, Segment) else seg.get("x1"))
    x2 = _to_float(seg.x2 if isinstance(seg, Segment) else seg.get("x2"))
    y = _to_float(seg.y if isinstance(seg, Segment) else seg.get("y"))
    if x1 is None or x2 is None or y is None:
        return None
    xx1 = clamp(x1, 0.0, 1.0)
    xx2 = clamp(x2, 0.0, 1.0)
    if xx2 < xx1:
        xx1, xx2 = xx2, xx1
    yy = clamp(y, 0.0, 1.0)
    length_norm = max(0.0, xx2 - xx1)
    length_px = _to_float(seg.length_px if isinstance(seg, Segment) else seg.get("length_px"))
    source = seg.source if isinstance(seg, Segment) else str(seg.get("source", ""))
    return Segment(
        x1=xx1,
        x2=xx2,
        y=yy,
        length_norm=length_norm,
        length_px=length_px,
        source=source,
    )


def _group_name_for_record(rec: Mapping[str, Any]) -> str:
    for key in ("groupName", "group_name", "group"):
        val = str(rec.get(key, "")).strip()
        if val:
            return val
    return ""


def _image_id_for_record(rec: Mapping[str, Any]) -> str:
    for key in ("id", "image_id", "relPath", "rel_path", "name", "file_name"):
        val = rec.get(key)
        if val is not None:
            text = str(val).strip()
            if text:
                return text
    return ""


def image_width_for_record(rec: Mapping[str, Any]) -> float:
    for key in ("width", "image_width", "_imgNaturalWidth", "naturalWidth", "w"):
        n = _to_float(rec.get(key))
        if n is not None and n > 1.0:
            return n
    return 0.0


def segment_metrics_for_record(
    rec: Mapping[str, Any],
    manual_segments_by_image: Mapping[str, Mapping[str, Any]] | None = None,
) -> Segment | None:
    image_id = _image_id_for_record(rec)

    raw_manual: Mapping[str, Any] | None = None
    if manual_segments_by_image and image_id and image_id in manual_segments_by_image:
        raw_manual = manual_segments_by_image[image_id]
    elif isinstance(rec.get("manual_segment"), Mapping):
        raw_manual = rec.get("manual_segment")

    if raw_manual is not None:
        seg = normalize_segment(raw_manual)
        if seg is not None:
            w = image_width_for_record(rec)
            length_px = seg.length_px
            if (length_px is None or length_px <= 0) and w > 1.0:
                length_px = seg.length_norm * w
            return Segment(
                x1=seg.x1,
                x2=seg.x2,
                y=seg.y,
                length_norm=seg.length_norm,
                length_px=length_px,
                source="manual",
            )

    raw_auto: Mapping[str, Any] | None = None
    if isinstance(rec.get("auto_segment"), Mapping):
        raw_auto = rec.get("auto_segment")
    elif isinstance(rec.get("segment"), Mapping):
        raw_auto = rec.get("segment")

    if raw_auto is None:
        return None

    seg = normalize_segment(raw_auto)
    if seg is None:
        return None
    w = image_width_for_record(rec)
    length_px = seg.length_px
    if (length_px is None or length_px <= 0) and w > 1.0:
        length_px = seg.length_norm * w
    return Segment(
        x1=seg.x1,
        x2=seg.x2,
        y=seg.y,
        length_norm=seg.length_norm,
        length_px=length_px,
        source="auto",
    )


def _segment_is_better(cand: TopSegmentRef, prev: TopSegmentRef | None) -> bool:
    if prev is None:
        return True
    if cand.len_px > (prev.len_px + 1e-6):
        return True
    if abs(cand.len_px - prev.len_px) <= 1e-6 and cand.len_norm > (prev.len_norm + 1e-9):
        return True
    return False


def choose_top_segment_global(
    records: Sequence[Mapping[str, Any]],
    manual_segments_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    seg_image_id_hint: str | None = None,
) -> TopSegmentRef | None:
    if seg_image_id_hint:
        hint = str(seg_image_id_hint).strip()
        for rec in records:
            rid = _image_id_for_record(rec)
            if rid != hint:
                continue
            seg = segment_metrics_for_record(rec, manual_segments_by_image)
            if seg is None:
                break
            return TopSegmentRef(
                image_id=rid,
                segment=seg,
                len_px=float(seg.length_px or 0.0),
                len_norm=float(seg.length_norm),
                image_width=image_width_for_record(rec),
            )

    best: TopSegmentRef | None = None
    for rec in records:
        rid = _image_id_for_record(rec)
        if not rid:
            continue
        seg = segment_metrics_for_record(rec, manual_segments_by_image)
        if seg is None:
            continue
        cand = TopSegmentRef(
            image_id=rid,
            segment=seg,
            len_px=float(seg.length_px or 0.0),
            len_norm=float(seg.length_norm),
            image_width=image_width_for_record(rec),
        )
        if _segment_is_better(cand, best):
            best = cand
    return best


def adjusted_global_rect_by_segment(
    base_rect: Mapping[str, Any] | Rect | None,
    seg: Mapping[str, Any] | Segment | None,
    margin_pct: float = 5.0,
) -> Rect | None:
    br = clamp_rect01(base_rect)
    if br is None:
        return None
    if seg is None:
        return br
    ss = normalize_segment(seg)
    if ss is None:
        return br

    seg_left = clamp(min(ss.x1, ss.x2), 0.0, 1.0)
    seg_right = clamp(max(ss.x1, ss.x2), 0.0, 1.0)
    seg_w = max(0.001, seg_right - seg_left)
    seg_mid = 0.5 * (seg_left + seg_right)

    margin_norm = clamp((_to_float(margin_pct) or 0.0) / 100.0, 0.0, 0.30)
    ww = clamp(seg_w + (2.0 * margin_norm), 0.001, 1.0)
    xx = seg_mid - (0.5 * ww)
    xx = clamp(xx, 0.0, 1.0 - ww)
    return clamp_rect01({"x": xx, "y": br.y, "w": ww, "h": br.h})


def build_unified_base_rect(
    explicit_global_rects: Sequence[Mapping[str, Any] | Rect],
    pred_rects_fallback: Sequence[Mapping[str, Any] | Rect],
) -> tuple[Rect | None, dict[str, Any]]:
    explicit = [r for rr in explicit_global_rects if (r := clamp_rect01(rr)) is not None]
    pred = [r for rr in pred_rects_fallback if (r := clamp_rect01(rr)) is not None]
    if explicit:
        base = median_rect(explicit)
        return base, {
            "source": "csv_global_explicit_unified",
            "n_explicit": len(explicit),
            "n_pred": len(pred),
        }
    base = median_rect(pred)
    return base, {
        "source": "csv_median_from_pred_unified",
        "n_explicit": 0,
        "n_pred": len(pred),
    }


def compute_folder_global_rects(
    records: Sequence[Mapping[str, Any]],
    base_rect_by_group: Mapping[str, Mapping[str, Any] | Rect],
    base_meta_by_group: Mapping[str, Mapping[str, Any]] | None = None,
    manual_segments_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    margin_pct: float = 5.0,
    seg_image_id_hint: str | None = None,
) -> GlobalRectComputation | None:
    base_rects = [r for rr in base_rect_by_group.values() if (r := clamp_rect01(rr)) is not None]
    unified_base = median_rect(base_rects)
    if unified_base is None:
        return None

    top_seg = choose_top_segment_global(
        records=records,
        manual_segments_by_image=manual_segments_by_image,
        seg_image_id_hint=seg_image_id_hint,
    )
    adjusted_red = adjusted_global_rect_by_segment(
        base_rect=unified_base,
        seg=top_seg.segment if top_seg is not None else None,
        margin_pct=margin_pct,
    )
    if adjusted_red is None:
        return None

    group_names: set[str] = set()
    group_names.update(str(k).strip() for k in base_rect_by_group.keys() if str(k).strip())
    group_names.update(_group_name_for_record(rec) for rec in records if _group_name_for_record(rec))
    if not group_names:
        group_names.add("__ALL__")

    n_explicit = 0
    n_pred = 0
    source = "global_unified"
    if base_meta_by_group:
        for meta in base_meta_by_group.values():
            n_explicit += int(_to_float(meta.get("n_explicit")) or 0)
            n_pred += int(_to_float(meta.get("n_pred")) or 0)
            src = str(meta.get("source", ""))
            if "explicit" in src:
                source = "csv_global_explicit_unified"
            elif source == "global_unified" and src:
                source = src
    if n_explicit == 0 and n_pred == 0:
        n_pred = len(base_rects)

    margin_norm = clamp((_to_float(margin_pct) or 0.0) / 100.0, 0.0, 0.30)
    base_by_group = {g: unified_base for g in group_names}
    red_by_group = {g: adjusted_red for g in group_names}
    meta_by_group: dict[str, dict[str, Any]] = {}
    for g in group_names:
        meta_by_group[g] = {
            "source": "segment_adjusted_unified",
            "base_source": source,
            "base_n_explicit": n_explicit,
            "base_n_pred": n_pred,
            "margin_pct": float(margin_pct),
            "margin_norm": float(margin_norm),
            "seg_image_id": top_seg.image_id if top_seg is not None else "",
            "seg_len_px": top_seg.len_px if top_seg is not None else None,
            "seg_len_norm": top_seg.len_norm if top_seg is not None else None,
            "seg_x1": top_seg.segment.x1 if top_seg is not None else None,
            "seg_x2": top_seg.segment.x2 if top_seg is not None else None,
            "seg_y": top_seg.segment.y if top_seg is not None else None,
        }

    return GlobalRectComputation(
        unified_base_rect=unified_base,
        adjusted_red_rect=adjusted_red,
        top_segment=top_seg,
        margin_pct=float(margin_pct),
        margin_norm=float(margin_norm),
        base_by_group=base_by_group,
        red_by_group=red_by_group,
        meta_by_group=meta_by_group,
    )


def _rect_axes(rect: Rect) -> dict[str, float]:
    x_mid = rect.x + (0.5 * rect.w)
    y_mid = rect.y + (0.5 * rect.h)
    return {
        "x_mid": x_mid,
        "y_mid": y_mid,
    }


def _rect_to_bounds(rect: Rect) -> dict[str, float]:
    return {
        "x1": rect.x,
        "y1": rect.y,
        "x2": rect.x + rect.w,
        "y2": rect.y + rect.h,
        "w": rect.w,
        "h": rect.h,
    }


def _norm_x_to_px(x: float, image_width: float) -> float | None:
    if image_width > 1.0:
        return x * image_width
    return None


def _norm_y_to_px(y: float, image_height: float) -> float | None:
    if image_height > 1.0:
        return y * image_height
    return None


def _segment_to_overlay(
    seg: Segment,
    image_width: float,
    image_height: float,
    label_prefix: str,
) -> dict[str, Any]:
    x_mid = 0.5 * (seg.x1 + seg.x2)
    length_norm = max(0.0, seg.x2 - seg.x1)
    length_px = seg.length_px
    if (length_px is None or length_px <= 0) and image_width > 1.0:
        length_px = length_norm * image_width
    if length_px is not None and length_px > 0:
        label = f"{label_prefix} {length_px:.1f} px"
    else:
        label = f"{label_prefix} {length_norm * 100:.1f}%"
    return {
        "line_norm": {"x1": seg.x1, "y1": seg.y, "x2": seg.x2, "y2": seg.y},
        "line_px": {
            "x1": _norm_x_to_px(seg.x1, image_width),
            "y1": _norm_y_to_px(seg.y, image_height),
            "x2": _norm_x_to_px(seg.x2, image_width),
            "y2": _norm_y_to_px(seg.y, image_height),
        },
        "median_norm": {"x": x_mid, "y": seg.y},
        "median_px": {
            "x": _norm_x_to_px(x_mid, image_width),
            "y": _norm_y_to_px(seg.y, image_height),
        },
        "length_norm": length_norm,
        "length_px": length_px,
        "label": label,
        "source": seg.source,
    }


def _segment_margins_vs_rect(
    seg: Segment,
    rect: Rect,
    image_width: float,
) -> dict[str, Any]:
    seg_left = min(seg.x1, seg.x2)
    seg_right = max(seg.x1, seg.x2)
    rect_left = rect.x
    rect_right = rect.x + rect.w

    left_a = min(seg_left, rect_left)
    left_b = max(seg_left, rect_left)
    right_a = min(seg_right, rect_right)
    right_b = max(seg_right, rect_right)

    left_norm = abs(seg_left - rect_left)
    right_norm = abs(rect_right - seg_right)
    left_px = (left_norm * image_width) if image_width > 1.0 else None
    right_px = (right_norm * image_width) if image_width > 1.0 else None

    if left_px is not None:
        left_label = f"marg sx {left_px:.1f} px"
    else:
        left_label = f"marg sx {left_norm * 100:.1f}%"
    if right_px is not None:
        right_label = f"marg dx {right_px:.1f} px"
    else:
        right_label = f"marg dx {right_norm * 100:.1f}%"

    return {
        "left_band_norm": {"x1": left_a, "x2": left_b, "y1": rect.y, "y2": rect.y + rect.h},
        "right_band_norm": {"x1": right_a, "x2": right_b, "y1": rect.y, "y2": rect.y + rect.h},
        "left_margin_norm": left_norm,
        "right_margin_norm": right_norm,
        "left_margin_px": left_px,
        "right_margin_px": right_px,
        "left_label": left_label,
        "right_label": right_label,
    }


def build_overlay_payload_for_record(
    rec: Mapping[str, Any],
    computation: GlobalRectComputation,
    manual_segments_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    per_image_rect_by_image: Mapping[str, Mapping[str, Any] | Rect] | None = None,
) -> dict[str, Any]:
    image_id = _image_id_for_record(rec)
    if not image_id:
        return {}
    image_width = image_width_for_record(rec)
    image_height = _to_float(rec.get("height")) or _to_float(rec.get("image_height")) or _to_float(rec.get("h")) or 0.0
    group_name = _group_name_for_record(rec) or "__ALL__"

    label = str(rec.get("label", "")).strip().lower()
    if label not in {"su", "giu"}:
        op = rec.get("orientationPred")
        if isinstance(op, Mapping):
            ol = str(op.get("label", "")).strip().lower()
            if ol in {"su", "giu"}:
                label = ol
    if not label:
        label = "unknown"

    yellow_rect: Rect | None = None
    if per_image_rect_by_image and image_id in per_image_rect_by_image:
        yellow_rect = clamp_rect01(per_image_rect_by_image[image_id])

    gray_rect = computation.base_by_group.get(group_name, computation.unified_base_rect)
    red_rect = computation.red_by_group.get(group_name, computation.adjusted_red_rect)

    seg_current = segment_metrics_for_record(rec, manual_segments_by_image=manual_segments_by_image)
    seg_top = computation.top_segment.segment if computation.top_segment is not None else None

    overlays: dict[str, Any] = {
        "rect_global_gray": {
            "bounds_norm": _rect_to_bounds(gray_rect),
            "axes_norm": _rect_axes(gray_rect),
            "label": "globale cartella base",
        },
        "rect_global_red": {
            "bounds_norm": _rect_to_bounds(red_rect),
            "axes_norm": _rect_axes(red_rect),
            "label": "globale cartella+seg top",
        },
    }

    if yellow_rect is not None:
        overlays["rect_yellow_per_image"] = {
            "bounds_norm": _rect_to_bounds(yellow_rect),
            "axes_norm": _rect_axes(yellow_rect),
            "label": "rete immagine",
        }

    if seg_current is not None:
        overlays["segment_current"] = _segment_to_overlay(
            seg=seg_current,
            image_width=image_width,
            image_height=image_height,
            label_prefix="segmento",
        )

    if seg_top is not None:
        overlays["segment_top"] = _segment_to_overlay(
            seg=seg_top,
            image_width=image_width,
            image_height=image_height,
            label_prefix="segmento top",
        )
        overlays["segment_margins"] = _segment_margins_vs_rect(
            seg=seg_top,
            rect=red_rect,
            image_width=image_width,
        )

    return {
        "image_id": image_id,
        "group_name": group_name,
        "label": label,
        "image_size": {
            "width": image_width if image_width > 0 else None,
            "height": image_height if image_height > 0 else None,
        },
        "overlays": overlays,
    }


def build_overlays_by_image(
    records: Sequence[Mapping[str, Any]],
    computation: GlobalRectComputation,
    manual_segments_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    per_image_rect_by_image: Mapping[str, Mapping[str, Any] | Rect] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        item = build_overlay_payload_for_record(
            rec=rec,
            computation=computation,
            manual_segments_by_image=manual_segments_by_image,
            per_image_rect_by_image=per_image_rect_by_image,
        )
        image_id = str(item.get("image_id", "")).strip()
        if image_id:
            out[image_id] = item
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reusable rect-red algorithm: unify base rect, choose top segment "
            "and compute folder-global adjusted rect."
        )
    )
    p.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input JSON with records/base rects/segments (see docs).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the computed global rect result.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))

    records = payload.get("records", [])
    base_rect_by_group = payload.get("base_rect_by_group", {})
    base_meta_by_group = payload.get("base_meta_by_group", {})
    manual_segments_by_image = payload.get("manual_segments_by_image", {})
    margin_pct = float(payload.get("margin_pct", 5.0))
    seg_image_id_hint = payload.get("seg_image_id_hint")
    include_overlays = bool(payload.get("include_overlays", True))
    per_image_rect_by_image_raw = payload.get("per_image_rect_by_image", {})
    per_image_rect_by_image: dict[str, Rect] = {}
    if isinstance(per_image_rect_by_image_raw, Mapping):
        for k, v in per_image_rect_by_image_raw.items():
            rr = clamp_rect01(v if isinstance(v, Mapping) else None)
            if rr is not None:
                per_image_rect_by_image[str(k)] = rr

    result = compute_folder_global_rects(
        records=records,
        base_rect_by_group=base_rect_by_group,
        base_meta_by_group=base_meta_by_group,
        manual_segments_by_image=manual_segments_by_image,
        margin_pct=margin_pct,
        seg_image_id_hint=seg_image_id_hint,
    )
    if result is None:
        out = {"ok": False, "error": "Unable to compute global rects."}
    else:
        out = {"ok": True, **result.to_jsonable()}
        if include_overlays:
            out["overlays_by_image"] = build_overlays_by_image(
                records=records,
                computation=result,
                manual_segments_by_image=manual_segments_by_image,
                per_image_rect_by_image=per_image_rect_by_image,
            )
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.output_json}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
