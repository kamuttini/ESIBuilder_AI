#!/usr/bin/env python3
"""Relabel and augment RECT_DEPTH candidate rows from a corrected manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Box = Tuple[float, float, float, float]


def _f(value: object, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_box(row: Dict[str, str], prefix: str = "") -> Box:
    p = f"{prefix}_" if prefix else ""
    return (_f(row.get(f"{p}left")), _f(row.get(f"{p}top")), _f(row.get(f"{p}right")), _f(row.get(f"{p}bottom")))


def _parse_box(text: object) -> Optional[Box]:
    vals = [_f(x, float("nan")) for x in str(text or "").replace(",", "|").split("|") if str(x).strip()]
    if len(vals) != 4 or any(not math.isfinite(x) for x in vals):
        return None
    left, top, right, bottom = vals
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


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


def _hint_flags(hint: str) -> Dict[str, float]:
    low = hint.lower().strip()
    return {
        "manual_hint_present": 1.0 if low else 0.0,
        "manual_hint_r": 1.0 if low == "r" else 0.0,
        "manual_hint_p": 1.0 if low == "p" or low.startswith("p e") else 0.0,
        "manual_hint_d": 1.0 if low == "d" or low.startswith("d e") else 0.0,
        "manual_hint_depth": 1.0 if low.startswith("depth") else 0.0,
        "manual_hint_cm": 1.0 if "cm" in low else 0.0,
        "manual_hint_mm": 1.0 if "mm" in low else 0.0,
        "manual_hint_scale": 1.0 if "scala" in low or "valore massimo" in low else 0.0,
    }


def _hint_scores(hint: str, row: Dict[str, str], manual_prior: bool = False) -> Dict[str, str]:
    flags = _hint_flags(hint)
    mapping = {
        "manual_hint_r": _f(row.get("r_hint_ratio")),
        "manual_hint_p": _f(row.get("p_hint_ratio")),
        "manual_hint_d": _f(row.get("d_hint_ratio")),
        "manual_hint_depth": _f(row.get("depth_hint_ratio")),
        "manual_hint_cm": _f(row.get("cm_ratio")),
        "manual_hint_mm": _f(row.get("mm_ratio")),
        "manual_hint_scale": _f(row.get("scale_hint_ratio")),
    }
    active = [mapping[k] for k, v in flags.items() if k != "manual_hint_present" and v > 0]
    inactive = [mapping[k] for k, v in flags.items() if k != "manual_hint_present" and v <= 0]
    if manual_prior and flags["manual_hint_present"]:
        match = 1.0
        mismatch = 0.0
    else:
        match = sum(active) / len(active) if active else 0.0
        mismatch = max(inactive) if inactive else 0.0
    out = {k: f"{v:.5f}" for k, v in flags.items()}
    out["manual_hint_match_score"] = f"{match:.5f}"
    out["manual_hint_mismatch_score"] = f"{mismatch:.5f}"
    return out


def _ocr_text_has_hint_letter(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"{target}\s*[:./-]?\s*\d", low) or re.search(rf"\d\s*{target}($|[^a-z0-9])", low))


def _has_forbidden_marker(text: str) -> bool:
    low = str(text or "").lower()
    compact = re.sub(r"[^a-z0-9%]", "", low)
    return bool(
        re.search(r"\b(?:mhz|hz)\b", low)
        or "%" in low
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


def _bad_suffix_after_number(text: str) -> bool:
    low = str(text or "").lower().replace(",", ".")
    for match in re.finditer(r"\d+(?:\.\d+)?\s*([a-z%]+)", low):
        suffix = match.group(1)
        if suffix.startswith(("cm", "mm")):
            continue
        return True
    return False


def _hint_left_of_number(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"{target}\s*[:./-]?\s*\d", low))


def _hint_right_of_number(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"\d\s*{target}($|[^a-z0-9])", low))


def _ocr_text_flags(text: str) -> Dict[str, str]:
    raw = str(text or "")
    return {
        "ocr_has_forbidden_marker": "1" if _has_forbidden_marker(raw) else "0",
        "ocr_has_hz_text": "1" if re.search(r"\b(?:mhz|hz)\b", raw, re.I) else "0",
        "ocr_has_percent": "1" if "%" in raw else "0",
        "ocr_has_mi_tis": "1" if re.search(r"\bmi\b|\btis\b", raw, re.I) else "0",
        "ocr_has_c2_text": "1" if re.search(r"\bc\s*2\b", raw, re.I) else "0",
        "ocr_has_probe_model": "1" if _has_probe_model_marker(raw) else "0",
        "ocr_has_time_like_text": "1" if _has_time_like_text(raw) else "0",
        "ocr_bad_suffix_after_number": "1" if _bad_suffix_after_number(raw) else "0",
        "ocr_text_has_d": "1" if _ocr_text_has_hint_letter(raw, "d") else "0",
        "ocr_text_has_p": "1" if _ocr_text_has_hint_letter(raw, "p") else "0",
        "ocr_text_has_r": "1" if _ocr_text_has_hint_letter(raw, "r") else "0",
        "ocr_text_has_depth": "1" if re.search(r"\bdepth\b", raw, re.I) else "0",
        "ocr_d_left_of_number": "1" if _hint_left_of_number(raw, "d") else "0",
        "ocr_d_right_of_number": "1" if _hint_right_of_number(raw, "d") else "0",
    }


def _sample_key(row: Dict[str, str]) -> str:
    return row.get("review_key") or f"{row.get('setup_id')}_{row.get('depth_index0')}_{row.get('flip_state')}"


def _label(pred: Box, gt: Box, positive_iou: float, positive_center_px: float) -> Tuple[int, float, float]:
    iou = _box_iou(pred, gt)
    center = _center_distance(pred, gt)
    return int(iou >= positive_iou or center <= positive_center_px), iou, center


def _blank_row(fieldnames: Iterable[str]) -> Dict[str, str]:
    return {field: "" for field in fieldnames}


def _manual_prior_row(
    manifest_row: Dict[str, str],
    fieldnames: List[str],
    positive_iou: float,
    positive_center_px: float,
) -> Optional[Dict[str, str]]:
    prior = _parse_box(manifest_row.get("ranker_review_box_prior"))
    if not prior:
        return None
    gt = _row_box(manifest_row)
    label, iou, center = _label(prior, gt, positive_iou, positive_center_px)
    row = _blank_row(fieldnames)
    row.update(
        {
            "sample_key": _sample_key(manifest_row),
            "config_folder": manifest_row.get("config_folder", ""),
            "fss_path": manifest_row.get("fss_path", ""),
            "source_image": manifest_row.get("source_image", ""),
            "setup_id": manifest_row.get("setup_id", ""),
            "depth_index0": manifest_row.get("depth_index0", ""),
            "depth_mm": manifest_row.get("depth_mm", ""),
            "flip_state": manifest_row.get("flip_state", ""),
            "candidate_rank": "0.0",
            "candidate_source": "manual_prior",
            "box_variant": "ranker_review_config_prior",
            "box_variant_expanded": "1",
            "box_variant_embedded": "0",
            "manual_prior_candidate": "1",
            "label": str(label),
            "iou": f"{iou:.5f}",
            "center_error_px": f"{center:.3f}",
            "cluster_score": "0.00000",
            "token_count": "0",
            "image_support": "0",
            "unique_values": "0",
            "pred_left": f"{prior[0]:.2f}",
            "pred_top": f"{prior[1]:.2f}",
            "pred_right": f"{prior[2]:.2f}",
            "pred_bottom": f"{prior[3]:.2f}",
            "pred_width": f"{prior[2] - prior[0]:.2f}",
            "pred_height": f"{prior[3] - prior[1]:.2f}",
            "gt_left": manifest_row.get("left", ""),
            "gt_top": manifest_row.get("top", ""),
            "gt_right": manifest_row.get("right", ""),
            "gt_bottom": manifest_row.get("bottom", ""),
            "review_comment": manifest_row.get("ranker_review_comment") or manifest_row.get("review_comment", ""),
            "ranker_review_hint": manifest_row.get("ranker_review_hint", ""),
            "cluster_reason": f"manual_prior_from_ranker_review; propagated={manifest_row.get('ranker_review_propagated', '')}; hint={manifest_row.get('ranker_review_hint', '')}",
        }
    )
    for key in [
        "cm_ratio",
        "mm_ratio",
        "depth_hint_ratio",
        "d_hint_ratio",
        "p_hint_ratio",
        "r_hint_ratio",
        "scale_hint_ratio",
        "fps_ips_ratio",
        "side_score",
        "accessory_score",
        "echo_center_penalty",
        "expected_ratio",
        "plausible_ratio",
        "ocr_conf",
        "ocr_text_len",
        "ocr_digit_group_count",
        "ocr_has_fps_ips",
        "ocr_has_forbidden_marker",
        "ocr_has_hz_text",
        "ocr_has_percent",
        "ocr_has_mi_tis",
        "ocr_has_c2_text",
        "ocr_has_probe_model",
        "ocr_has_time_like_text",
        "ocr_bad_suffix_after_number",
        "ocr_has_letter_hint",
        "ocr_text_has_d",
        "ocr_text_has_p",
        "ocr_text_has_r",
        "ocr_text_has_depth",
        "ocr_d_left_of_number",
        "ocr_d_right_of_number",
        "ocr_has_cm_text",
        "ocr_has_mm_text",
        "ocr_numeric_value",
        "ocr_snapped_depth_mm",
        "ocr_multi_number",
        "ocr_single_depth_expr",
        "wide_text_box",
        "snap_error_mm",
        "value_error_mm",
    ]:
        row.setdefault(key, "0")
        row[key] = row[key] or "0"
    row.setdefault("ocr_text", "")
    row.setdefault("ocr_snap_mode", "")
    row.update(_ocr_text_flags(row.get("ocr_text", "")))
    row.update(_hint_scores(manifest_row.get("ranker_review_hint", ""), row, manual_prior=True))
    return row


def build(args: argparse.Namespace) -> Dict[str, object]:
    candidates = _read_csv(args.candidates_csv.expanduser().resolve())
    manifest_rows = _read_csv(args.manifest.expanduser().resolve())
    manifest_by_key = {_sample_key(row): row for row in manifest_rows}
    fieldnames = list(candidates[0].keys()) if candidates else []
    extra_fields = [
        "manual_prior_candidate",
        "ranker_review_hint",
        "manual_hint_present",
        "manual_hint_r",
        "manual_hint_p",
        "manual_hint_d",
        "manual_hint_depth",
        "manual_hint_cm",
        "manual_hint_mm",
        "manual_hint_scale",
        "manual_hint_match_score",
        "manual_hint_mismatch_score",
        "ocr_text_has_d",
        "ocr_text_has_p",
        "ocr_text_has_r",
        "ocr_text_has_depth",
        "ocr_has_forbidden_marker",
        "ocr_has_hz_text",
        "ocr_has_percent",
        "ocr_has_mi_tis",
        "ocr_has_c2_text",
        "ocr_has_probe_model",
        "ocr_has_time_like_text",
        "ocr_bad_suffix_after_number",
        "ocr_d_left_of_number",
        "ocr_d_right_of_number",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    out_rows: List[Dict[str, str]] = []
    stats: Counter = Counter(input_candidate_rows=len(candidates), manifest_rows=len(manifest_rows))
    positive_iou = float(args.positive_iou)
    positive_center_px = float(args.positive_center_px)
    for row in candidates:
        sample_key = row.get("sample_key", "")
        manifest = manifest_by_key.get(sample_key)
        out = dict(row)
        out.update(_ocr_text_flags(out.get("ocr_text", "")))
        out["manual_prior_candidate"] = "0"
        if manifest:
            gt = _row_box(manifest)
            pred = (_f(row.get("pred_left")), _f(row.get("pred_top")), _f(row.get("pred_right")), _f(row.get("pred_bottom")))
            label, iou, center = _label(pred, gt, positive_iou, positive_center_px)
            out.update(
                {
                    "label": str(label),
                    "iou": f"{iou:.5f}",
                    "center_error_px": f"{center:.3f}",
                    "gt_left": manifest.get("left", ""),
                    "gt_top": manifest.get("top", ""),
                    "gt_right": manifest.get("right", ""),
                    "gt_bottom": manifest.get("bottom", ""),
                    "review_comment": manifest.get("ranker_review_comment") or manifest.get("review_comment", ""),
                    "ranker_review_hint": manifest.get("ranker_review_hint", ""),
                }
            )
            if label:
                stats["positive_rows_after_relabel"] += 1
            if manifest.get("ranker_review_hint"):
                stats["candidate_rows_with_hint"] += 1
            out.update(_hint_scores(manifest.get("ranker_review_hint", ""), out))
        else:
            stats["candidate_rows_without_manifest"] += 1
            out["ranker_review_hint"] = ""
            out.update(_hint_scores("", out))
        out_rows.append(out)

    existing_manual = {r.get("sample_key") for r in out_rows if r.get("candidate_source") == "manual_prior"}
    for manifest in manifest_rows:
        key = _sample_key(manifest)
        if key in existing_manual:
            continue
        row = _manual_prior_row(manifest, fieldnames, positive_iou, positive_center_px)
        if row:
            out_rows.append(row)
            stats["manual_prior_rows_added"] += 1
            if row.get("label") == "1":
                stats["manual_prior_positive_rows"] += 1

    output = args.output_csv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    stats["output_rows"] = len(out_rows)
    stats["samples_with_positive"] = len({r["sample_key"] for r in out_rows if str(r.get("label")) == "1"})
    stats["output_csv"] = output.as_posix()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel RECT_DEPTH candidates from a corrected manifest.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--positive-iou", type=float, default=0.20)
    parser.add_argument("--positive-center-px", type=float, default=35.0)
    args = parser.parse_args()
    summary = build(args)
    if args.summary_json:
        out = args.summary_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
