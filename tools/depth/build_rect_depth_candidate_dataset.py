#!/usr/bin/env python3
"""Build a supervised OCR-candidate dataset for RECT_DEPTH ranking.

This is the bridge between the manual GT review and the hybrid model:

- OCR/classical code generates recurring candidate clusters for each folder.
- Manual ``corrected_gt_box`` labels decide which candidate/token is correct.
- The output CSV is small tabular data suitable for a conservative sklearn
  ranker, blended later with hard constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from rect_depth_hybrid import (
    DepthCluster,
    DepthToken,
    cluster_tokens,
    collect_depth_tokens,
    parse_rect_depth_checks,
    parse_rect_echo,
)


Box = Tuple[float, float, float, float]
_IMAGE_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}


def _f(text: object, default: float = float("nan")) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def _parse_box(text: object) -> Optional[Box]:
    nums = [_f(x) for x in re.split(r"[|,;\s]+", str(text or "").strip()) if x]
    if len(nums) != 4 or any(not math.isfinite(x) for x in nums):
        return None
    x1, y1, x2, y2 = nums
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _row_box(row: Dict[str, str]) -> Box:
    return (_f(row["left"]), _f(row["top"]), _f(row["right"]), _f(row["bottom"]))


def _box_iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: Box, b: Box) -> float:
    return math.hypot(0.5 * (a[0] + a[2]) - 0.5 * (b[0] + b[2]), 0.5 * (a[1] + a[3]) - 0.5 * (b[1] + b[3]))


def _token_box(token: DepthToken) -> Box:
    w = token.word
    return (float(w.left), float(w.top), float(w.right), float(w.bottom))


def _cluster_box(cluster: DepthCluster) -> Box:
    return (float(cluster.left), float(cluster.top), float(cluster.right), float(cluster.bottom))


def _image_size(path: str) -> Tuple[int, int]:
    if path in _IMAGE_SIZE_CACHE:
        return _IMAGE_SIZE_CACHE[path]
    try:
        with Image.open(path) as im:
            size = (int(im.size[0]), int(im.size[1]))
    except Exception:
        size = (1, 1)
    _IMAGE_SIZE_CACHE[path] = size
    return size


def _clamp_box(box: Box, width: int, height: int) -> Optional[Box]:
    x0, y0, x1, y1 = box
    left = max(0.0, min(float(width - 1), float(x0)))
    top = max(0.0, min(float(height - 1), float(y0)))
    right = max(left + 1.0, min(float(width), float(x1)))
    bottom = max(top + 1.0, min(float(height), float(y1)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _embedded_value_windows(token: Optional[DepthToken]) -> List[Tuple[str, Box]]:
    if token is None:
        return []
    text = str(token.word.text or "")
    if len(text) < 5 or token.word.width < 110:
        return []
    matches = list(re.finditer(r"\d+(?:[.,]\d+)?", text))
    if not matches:
        return []
    out: List[Tuple[str, Box]] = []
    n = max(1, len(text))
    target = float(token.numeric_value)
    # A wide OCR word can contain neighbouring UI text, while the useful
    # direct expression is compact (D/P/R/Depth + value + mm/cm).  Add an
    # exact crop spanning that expression so the final RECT_DEPTH box begins
    # at the marker and finishes after the unit instead of retaining noise.
    for direct in re.finditer(
        r"(?i)(depth|dep|deph|dept|dpth|(?<![a-z0-9])[dpr](?![a-z]))\s*[:=./-]?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm)(?![a-z])",
        text,
    ):
        try:
            value = float(direct.group(2).replace(",", "."))
        except Exception:
            continue
        if abs(value - target) > max(0.015, 0.01 * max(1.0, abs(target))):
            continue
        x0 = token.word.left + token.word.width * direct.start(1) / n
        x1 = token.word.left + token.word.width * direct.end(3) / n
        y0 = token.word.top - max(5.0, 0.30 * token.word.height)
        y1 = token.word.bottom + max(6.0, 0.40 * token.word.height)
        out.append(("direct_label_unit_window", (x0 - 4.0, y0, x1 + 4.0, y1)))
    for match in matches:
        try:
            value = float(match.group(0).replace(",", "."))
        except Exception:
            continue
        if abs(value - target) > max(0.015, 0.01 * max(1.0, abs(target))):
            continue
        start = match.start()
        end = match.end()
        while start > 0 and text[start - 1].lower() in {":", ".", "/", "-", "d", "p", "r", "f"}:
            start -= 1
        while end < len(text) and text[end:end + 1].lower() in {":", ".", "/", "-", "c", "m"}:
            end += 1
        x0 = token.word.left + token.word.width * max(0, start - 2) / n
        x1 = token.word.left + token.word.width * min(n, end + 4) / n
        y0 = token.word.top - max(7.0, 0.45 * token.word.height)
        y1 = token.word.bottom + max(9.0, 0.65 * token.word.height)
        out.append(("embedded_value_window", (x0 - 8.0, y0, x1 + 8.0, y1)))
        cx = 0.5 * (x0 + x1)
        out.append(("embedded_value_120x42", (cx - 60.0, y0, cx + 60.0, y1)))
    return out


def _unique_texts(texts: Iterable[str], limit: int = 10) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _candidate_ocr_text(cluster: DepthCluster, image_path: str, token: Optional[DepthToken]) -> str:
    if token is not None:
        return str(token.word.text or "")
    same_image = [
        t.word.text
        for t in sorted(cluster.tokens, key=lambda item: (-item.word.conf, item.word.top, item.word.left))
        if t.word.image_path.as_posix() == image_path
    ]
    texts = same_image or [
        t.word.text
        for t in sorted(cluster.tokens, key=lambda item: (-item.word.conf, item.word.top, item.word.left))
    ]
    return " | ".join(_unique_texts(texts, limit=8))


def _digit_group_count(text: str) -> int:
    return len(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


def _has_fps_ips(text: str) -> bool:
    compact = re.sub(r"[^a-z]", "", str(text or "").lower())
    spaced = re.sub(r"[^a-z]+", " ", str(text or "").lower())
    words = set(spaced.split())
    return "fps" in compact or "ips" in compact or bool(words.intersection({"fp", "ip"}))


def _has_forbidden_marker(text: str) -> bool:
    low = str(text or "").lower()
    compact = re.sub(r"[^a-z0-9%]", "", low)
    return bool(
        re.search(r"\b(?:mhz|hz)\b", low)
        or "%" in low
        or re.search(r"(?<![a-z0-9])\d+(?:\.\d+)?\s*d\s*[b8]\b", low)
        or "tis" in compact
        or re.search(r"\bmi\b", low)
        or "mi" in compact[:8]
        or re.search(r"\betd\b", low)
        or "etd" in compact
        or re.search(r"\bc\s*2\b", low)
        or "c2" in compact
    )


def _has_probe_model_marker(text: str) -> bool:
    low = str(text or "").lower()
    compact = re.sub(r"[^a-z0-9-]", "", low)
    patterns = [
        r"\b(?:tlc|trt)\s*\d{1,2}\s*[-/]?\s*\d{1,2}\b",
        r"\b(?:la|ac|sl|ca|si)\s*\d{2,5}[a-z]?\b",
        r"\bl\s*\d\s*[-/]\s*\d{1,2}[a-z]?\b",
        r"\bf\s*\d{5,}\b",
    ]
    if any(re.search(pattern, low) for pattern in patterns):
        return True
    compact_hits = [
        "tlc313",
        "trt33",
        "la332",
        "la523",
        "ac2541",
        "sl1543",
        "ca541",
        "si2c41",
        "l415",
    ]
    return any(hit in compact for hit in compact_hits)


def _has_time_like_text(text: str) -> bool:
    low = str(text or "").lower()
    return bool(re.search(r"(?<![a-z0-9])(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d(?:\s*:\s*[0-5]\d)?(?![a-z0-9])", low))


def _normalize_unit_ocr_confusions(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").replace(",", ".").strip())
    clean = re.sub(r"(?<=\d)[il|](?=\s*(?:cm|c|em|tm)(?![a-z]))", "1", clean, flags=re.I)
    clean = re.sub(r"(?<=\d)(?:em|tm)(?![a-z])", "cm", clean, flags=re.I)
    return clean


def _bad_suffix_after_number(text: str) -> bool:
    low = _normalize_unit_ocr_confusions(text).lower()
    for match in re.finditer(r"\d+(?:\.\d+)?\s*([a-z%]+)", low):
        suffix = match.group(1)
        if suffix.startswith(("cm", "mm")):
            continue
        if suffix == "c" and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*c[^a-z0-9]*", low):
            continue
        return True
    return False


def _ocr_has_letter_hint(text: str) -> bool:
    low = _normalize_unit_ocr_confusions(text).lower()
    return bool(
        re.search(r"\bdepth\b", low)
        or re.search(r"(?<![a-z0-9])[dpr](?![a-z])\s*[:./-]?\s*\d", low.replace(" ", ""))
        or re.search(r"\d\s*(cm|mm)\b", low)
    )


def _ocr_text_has_hint_letter(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(
        re.search(rf"(?<![a-z0-9]){target}(?![a-z])\s*[:./-]?\s*\d", low)
        or re.search(rf"\d\s*{target}($|[^a-z0-9])", low)
    )


def _hint_left_of_number(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"(?<![a-z0-9]){target}(?![a-z])\s*[:./-]?\s*\d", low))


def _hint_right_of_number(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"\d\s*{target}($|[^a-z0-9])", low))


def _candidate_variants(base_box: Box, image_path: str, token: Optional[DepthToken] = None) -> List[Tuple[str, Box]]:
    width, height = _image_size(image_path)
    variants: List[Tuple[str, Box]] = []

    def add(name: str, box: Box) -> None:
        clamped = _clamp_box(box, width, height)
        if clamped and all(_center_distance(clamped, old_box) > 2.0 or name != old_name for old_name, old_box in variants):
            variants.append((name, clamped))

    x0, y0, x1, y1 = base_box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    add("base", base_box)
    for name, box in _embedded_value_windows(token):
        add(name, box)
    if bw <= 55.0 or bh <= 18.0:
        # Hitachi-style accessory depth: OCR often sees only a tiny fragment of
        # the value/scale. The visible GT box consistently extends right/down.
        add("expand_right_105x38", (x0 - 10.0, y0 - 9.0, x0 + 105.0, y0 + 38.0))
        add("expand_center_105x38", (cx - 48.0, cy - 18.0, cx + 57.0, cy + 20.0))
        add("expand_left_125x42", (x1 - 125.0, y0 - 10.0, x1 + 14.0, y0 + 42.0))
        add("expand_wide_165x44", (cx - 82.0, cy - 20.0, cx + 83.0, cy + 24.0))
    return variants


def _best_token_for_row(cluster: DepthCluster, image_path: str, target_depth_mm: float) -> Optional[DepthToken]:
    tokens = [t for t in cluster.tokens if t.word.image_path.as_posix() == image_path]
    if not tokens:
        return None

    def key(token: DepthToken) -> Tuple[float, float]:
        value = float(token.snapped_depth_mm or token.numeric_value)
        return (abs(value - target_depth_mm), -float(token.word.conf))

    return min(tokens, key=key)


def _expected_depths(fss_path: Path) -> List[float]:
    try:
        _checks, depths = parse_rect_depth_checks(fss_path)
        return depths
    except Exception:
        return []


def _expected_depths_from_rows(rows: Sequence[Dict[str, str]]) -> List[float]:
    vals: List[float] = []
    for row in rows:
        value = _f(row.get("depth_mm"))
        if math.isfinite(value) and value > 0 and all(abs(value - old) > 1e-6 for old in vals):
            vals.append(value)
    return vals


def _safe_slug(text: str, max_len: int = 80) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _uses_roi_ocr(config_folder: str, roi_all: bool, roi_vendors: str) -> bool:
    if roi_all:
        return True
    vendors = [v.strip().lower() for v in str(roi_vendors or "").split(",") if v.strip()]
    if not vendors:
        return False
    low = config_folder.lower()
    return any(v in low for v in vendors)


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    rows = _load_rows(args.review_manifest.expanduser().resolve())
    if args.max_rows:
        rows = rows[: int(args.max_rows)]

    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("config_folder", ""), row.get("fss_path", ""))].append(row)
    group_items = list(groups.items())
    if args.config_contains:
        terms = [t.strip().lower() for t in str(args.config_contains).split(",") if t.strip()]
        group_items = [
            item for item in group_items
            if any(term in item[0][0].lower() for term in terms)
        ]
    if args.max_configs:
        group_items = group_items[: int(args.max_configs)]

    out_rows: List[Dict[str, object]] = []
    stats: Counter = Counter(input_rows=len(rows), groups=len(group_items))

    for group_index, ((config_folder, fss_path_text), group_rows) in enumerate(group_items, start=1):
        fss_path = Path(fss_path_text)
        images = sorted({Path(r["source_image"]) for r in group_rows if r.get("source_image")})
        images = [p for p in images if p.exists()]
        if not images:
            stats["groups_no_images"] += 1
            continue

        expected = _expected_depths(fss_path)
        if not expected:
            expected = _expected_depths_from_rows(group_rows)
        rect_echo = parse_rect_echo(fss_path) if fss_path.exists() else None
        use_roi_ocr = _uses_roi_ocr(config_folder, bool(args.ocr_roi_passes), args.ocr_roi_vendors)
        try:
            tokens = collect_depth_tokens(
                images=images,
                timeout=float(args.ocr_timeout),
                max_side=int(args.ocr_max_side),
                min_conf=float(args.min_ocr_conf),
                expected_depths_mm=expected,
                rect_echo=rect_echo,
                roi_passes=use_roi_ocr,
            )
            clusters = cluster_tokens(
                tokens=tokens,
                bin_px=float(args.cluster_bin_px),
                image_count=len(images),
                expected_depths_mm=expected,
                rect_echo=rect_echo,
            )
        except Exception as exc:
            stats["groups_ocr_error"] += 1
            if args.verbose:
                print(f"OCR error {config_folder}: {exc}", flush=True)
            continue

        if use_roi_ocr:
            stats["groups_roi_ocr"] += 1
        stats["tokens"] += len(tokens)
        stats["clusters"] += len(clusters)
        if not clusters:
            stats["groups_no_clusters"] += 1
            continue

        for row in group_rows:
            gt_box = _row_box(row)
            image_path = Path(row["source_image"]).as_posix()
            depth_mm = _f(row.get("depth_mm"))
            for rank, cluster in enumerate(clusters[: int(args.max_candidates_per_sample)], start=1):
                token = _best_token_for_row(cluster, image_path, depth_mm)
                base_box = _token_box(token) if token else _cluster_box(cluster)
                for variant_index, (box_variant, pred_box) in enumerate(_candidate_variants(base_box, image_path, token)):
                    ocr_text = _candidate_ocr_text(cluster, image_path, token)
                    digit_groups = _digit_group_count(ocr_text)
                    pred_width = pred_box[2] - pred_box[0]
                    pred_height = pred_box[3] - pred_box[1]
                    aspect = pred_width / max(1.0, pred_height)
                    iou = _box_iou(pred_box, gt_box)
                    center_px = _center_distance(pred_box, gt_box)
                    is_positive = int(iou >= float(args.positive_iou) or center_px <= float(args.positive_center_px))
                    if is_positive:
                        stats["positive_rows"] += 1
                    source = "token" if token else "cluster"
                    out_rows.append(
                        {
                            "sample_key": row.get("review_key") or f"{row.get('setup_id')}_{row.get('depth_index0')}_{row.get('flip_state')}",
                            "config_folder": config_folder,
                            "fss_path": fss_path_text,
                            "source_image": image_path,
                            "setup_id": row.get("setup_id", ""),
                            "depth_index0": row.get("depth_index0", ""),
                            "depth_mm": row.get("depth_mm", ""),
                            "flip_state": row.get("flip_state", ""),
                            "candidate_rank": rank + 0.01 * variant_index,
                            "candidate_source": source,
                            "box_variant": box_variant,
                            "box_variant_expanded": int(box_variant != "base"),
                            "box_variant_embedded": int(box_variant.startswith("embedded_value")),
                            "label": is_positive,
                            "iou": f"{iou:.5f}",
                            "center_error_px": f"{center_px:.3f}",
                            "cluster_score": f"{cluster.score:.5f}",
                            "token_count": len(cluster.tokens),
                            "image_support": len(cluster.images),
                            "unique_values": cluster.unique_values,
                            "cm_ratio": f"{cluster.cm_ratio:.5f}",
                            "mm_ratio": f"{cluster.mm_ratio:.5f}",
                            "depth_hint_ratio": f"{cluster.depth_hint_ratio:.5f}",
                            "d_hint_ratio": f"{cluster.d_hint_ratio:.5f}",
                            "p_hint_ratio": f"{cluster.p_hint_ratio:.5f}",
                            "r_hint_ratio": f"{cluster.r_hint_ratio:.5f}",
                            "scale_hint_ratio": f"{cluster.scale_hint_ratio:.5f}",
                            "fps_ips_ratio": f"{cluster.fps_ips_ratio:.5f}",
                            "side_score": f"{cluster.side_score:.5f}",
                            "accessory_score": f"{cluster.accessory_score:.5f}",
                            "echo_center_penalty": f"{cluster.echo_center_penalty:.5f}",
                            "expected_ratio": f"{cluster.expected_ratio:.5f}",
                            "plausible_ratio": f"{cluster.plausible_ratio:.5f}",
                            "ocr_conf": f"{token.word.conf:.3f}" if token else "",
                            "ocr_text": ocr_text,
                            "ocr_text_len": len(ocr_text),
                            "ocr_digit_group_count": digit_groups,
                            "ocr_has_fps_ips": int(_has_fps_ips(ocr_text) or (token.has_fps_ips_hint if token else cluster.fps_ips_ratio > 0.0)),
                            "ocr_has_forbidden_marker": int(_has_forbidden_marker(ocr_text)),
                            "ocr_has_hz_text": int(bool(re.search(r"\b(?:mhz|hz)\b", ocr_text, re.I))),
                            "ocr_has_percent": int("%" in ocr_text),
                            "ocr_has_mi_tis": int(bool(re.search(r"\bmi\b|\btis\b", ocr_text, re.I))),
                            "ocr_has_c2_text": int(bool(re.search(r"\bc\s*2\b", ocr_text, re.I))),
                            "ocr_has_probe_model": int(_has_probe_model_marker(ocr_text)),
                            "ocr_has_time_like_text": int(_has_time_like_text(ocr_text)),
                            "ocr_bad_suffix_after_number": int(_bad_suffix_after_number(ocr_text)),
                            "ocr_has_letter_hint": int(_ocr_has_letter_hint(ocr_text)),
                            "ocr_text_has_d": int(_ocr_text_has_hint_letter(ocr_text, "d")),
                            "ocr_text_has_p": int(_ocr_text_has_hint_letter(ocr_text, "p")),
                            "ocr_text_has_r": int(_ocr_text_has_hint_letter(ocr_text, "r")),
                            "ocr_text_has_depth": int(bool(re.search(r"\bdepth\b", ocr_text, re.I))),
                            "ocr_d_left_of_number": int(_hint_left_of_number(ocr_text, "d")),
                            "ocr_d_right_of_number": int(_hint_right_of_number(ocr_text, "d")),
                            "ocr_has_cm_text": int(bool(re.search(r"\bcm\b", ocr_text, re.I))),
                            "ocr_has_mm_text": int(bool(re.search(r"\bmm\b", ocr_text, re.I))),
                            "ocr_numeric_value": f"{token.numeric_value:.3f}" if token else "",
                            "ocr_snapped_depth_mm": f"{float(token.snapped_depth_mm):.3f}" if token and token.snapped_depth_mm is not None else "",
                            "ocr_snap_mode": token.snap_mode if token else "",
                            "ocr_multi_number": int(digit_groups > 2),
                            "ocr_single_depth_expr": int(digit_groups == 1 and (_ocr_has_letter_hint(ocr_text) or token is not None)),
                            "wide_text_box": int(pred_width > 220.0 and aspect > 4.0),
                            "snap_error_mm": f"{token.snap_error_mm:.3f}" if token and math.isfinite(token.snap_error_mm) else "",
                            "value_error_mm": f"{abs(float(token.snapped_depth_mm or token.numeric_value) - depth_mm):.3f}" if token else "",
                            "pred_width": f"{pred_width:.2f}",
                            "pred_height": f"{pred_height:.2f}",
                            "pred_left": f"{pred_box[0]:.2f}",
                            "pred_top": f"{pred_box[1]:.2f}",
                            "pred_right": f"{pred_box[2]:.2f}",
                            "pred_bottom": f"{pred_box[3]:.2f}",
                            "gt_left": row.get("left", ""),
                            "gt_top": row.get("top", ""),
                            "gt_right": row.get("right", ""),
                            "gt_bottom": row.get("bottom", ""),
                            "review_comment": row.get("review_comment", ""),
                            "cluster_reason": cluster.reason,
                        }
                    )

        if args.verbose:
            print(
                f"[{group_index}/{len(group_items)}] {config_folder}: "
                f"{len(images)} images, {len(tokens)} tokens, {len(clusters)} clusters",
                flush=True,
            )

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        if fieldnames:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    stats["candidate_rows"] = len(out_rows)
    stats["samples_with_positive"] = len({r["sample_key"] for r in out_rows if int(r["label"])})
    stats["output_csv"] = output_csv.as_posix()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised OCR candidate rows for RECT_DEPTH ranking.")
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--config-contains", default="")
    parser.add_argument("--max-candidates-per-sample", type=int, default=12)
    parser.add_argument("--ocr-timeout", type=float, default=8.0)
    parser.add_argument("--ocr-max-side", type=int, default=1600)
    parser.add_argument("--min-ocr-conf", type=float, default=18.0)
    parser.add_argument("--ocr-roi-passes", action="store_true")
    parser.add_argument("--ocr-roi-vendors", default="")
    parser.add_argument("--cluster-bin-px", type=float, default=36.0)
    parser.add_argument("--positive-iou", type=float, default=0.20)
    parser.add_argument("--positive-center-px", type=float, default=35.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = build_dataset(args)
    if args.summary_json:
        out = args.summary_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
