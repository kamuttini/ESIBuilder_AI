#!/usr/bin/env python3
"""Autonomous RECT_DEPTH recognition from OCR/classical candidates.

This is the production-facing bridge after the review/ranker experiments:

- it does not consume manual per-image review hints;
- it can consume vendor/probe context from the existing pipeline;
- it uses OCR/classical constraints to validate and rank candidates;
- it prefers explicit UI values (Depth/D/P/R + number, optional cm/mm);
- if no explicit UI value is present, it falls back to scale values and prefers
  the largest scale value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional at runtime
    cv2 = None

try:
    import joblib
except Exception:  # pragma: no cover - optional at runtime
    joblib = None

from rect_depth_hybrid import (
    DepthCluster,
    DepthToken,
    cluster_tokens,
    collect_depth_tokens,
    iter_images,
    parse_rect_depth_checks,
    parse_rect_echo,
)
from build_rect_depth_candidate_dataset import (
    Box,
    _candidate_ocr_text,
    _candidate_variants,
    _cluster_box,
    _digit_group_count,
    _has_forbidden_marker,
    _has_fps_ips,
    _has_probe_model_marker,
    _has_time_like_text,
    _hint_right_of_number,
    _token_box,
)
from train_rect_depth_candidate_ranker import FEATURES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANKER_MODEL = REPO_ROOT / "artifacts/24_rect_depth_hybrid/candidate_ranker_v11_bk_reviews/ranker.joblib"


@dataclass(frozen=True)
class DepthProfile:
    vendor: str = ""
    probe: str = ""
    prefer_direct: bool = True
    scale_fallback: bool = True
    scale_max_required: bool = True
    scale_cm_bias: float = 0.0
    scale_mm_bias: float = 0.0
    direct_letter_bias: float = 0.0
    scale_side_preference: str = "right"
    stable_scale_position_review: bool = True
    notes: Tuple[str, ...] = ()


def _f(value: object, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_slug(text: str, max_len: int = 90) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _read_json(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    p = path.expanduser().resolve()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _find_first_nested(data: object, keys: Sequence[str]) -> str:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value).strip()
        for value in data.values():
            found = _find_first_nested(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_first_nested(value, keys)
            if found:
                return found
    return ""


def _infer_vendor_from_text(text: str) -> str:
    low = text.lower()
    checks = [
        ("BK", r"\bbk\b|bk3000|bk5000|flexfocus|profocus|specto"),
        ("Esaote", r"esaote|mylab"),
        ("Hitachi", r"hitachi|arietta|aloka"),
        ("GE", r"\bge\b|logiq|voluson"),
        ("Mindray", r"mindray|resona|te7"),
        ("Philips", r"philips|affiniti|epiq"),
        ("Biopsee", r"biopsee|biojet"),
        ("Koelis", r"koelis"),
        ("Terason", r"terason"),
        ("Sonostar", r"sonostar"),
        ("Canon", r"canon|aplio|toshiba"),
        ("Alpinion", r"alpinion|ecube"),
        ("Siemens", r"siemens|acuson"),
        ("Vinno", r"vinno"),
        ("ExactVu", r"exactvu"),
    ]
    for name, pattern in checks:
        if re.search(pattern, low):
            return name
    return ""


def _infer_probe_from_text(text: str) -> str:
    hits = re.findall(
        r"\b(?:tlc\s*3\s*[-/]?\s*13|trt\s*33|la\s*332e?|la\s*523|ac\s*2541|sl\s*1543|ca\s*541|si\s*2c41|"
        r"e14cl4b|18l5|8848|8823|6c2|l\s*4\s*[-/]?\s*15|cl4416r1?|c41l47rp)\b",
        text,
        flags=re.I,
    )
    return hits[0].strip() if hits else ""


def resolve_context(folder: Path, vendor: str, probe: str, context_json: Optional[Path]) -> Dict[str, object]:
    payload = _read_json(context_json)
    context_text = " ".join([folder.name, folder.as_posix(), json.dumps(payload, ensure_ascii=False)])
    vendor_out = vendor.strip() or _find_first_nested(
        payload,
        [
            "vendor_predicted",
            "predicted_vendor",
            "vendor_pred",
            "predicted_name",
            "manufacturer",
            "manufacturer_inferred",
        ],
    )
    if vendor_out:
        vendor_out = _infer_vendor_from_text(vendor_out) or vendor_out
    if not vendor_out:
        vendor_out = _infer_vendor_from_text(context_text)

    probe_out = probe.strip() or _find_first_nested(
        payload,
        [
            "predicted_probe_id",
            "predicted_probe",
            "probe_name",
            "predicted_probe_name",
            "line_03_id_probe",
            "line_04_probe_type",
        ],
    )
    if not probe_out:
        probe_out = _infer_probe_from_text(context_text)
    return {
        "vendor": vendor_out,
        "probe": probe_out,
        "context_source": context_json.expanduser().as_posix() if context_json else "folder_name_or_cli",
    }


def profile_for_context(vendor: str, probe: str) -> DepthProfile:
    low_vendor = vendor.lower()
    low_probe = probe.lower()
    notes: List[str] = []
    profile = DepthProfile(vendor=vendor, probe=probe)
    if "bk" in low_vendor:
        notes.append("BK: scala laterale frequente; se non c'e label esplicita, preferire massimo scala")
        profile = DepthProfile(
            vendor=vendor,
            probe=probe,
            prefer_direct=True,
            scale_fallback=True,
            scale_max_required=True,
            scale_cm_bias=0.10,
            scale_mm_bias=0.04,
            direct_letter_bias=0.02,
            scale_side_preference="right",
            stable_scale_position_review=True,
            notes=tuple(notes),
        )
    elif "esaote" in low_vendor:
        notes.append("Esaote: preferire label D/P/Depth con mm; penalizzare modelli sonda e timestamp")
        profile = DepthProfile(
            vendor=vendor,
            probe=probe,
            prefer_direct=True,
            scale_fallback=True,
            scale_max_required=True,
            scale_cm_bias=0.02,
            scale_mm_bias=0.08,
            direct_letter_bias=0.08,
            scale_side_preference="right",
            stable_scale_position_review=True,
            notes=tuple(notes),
        )
    elif "hitachi" in low_vendor:
        notes.append("Hitachi: label R/P/D frequenti; crop embedded utile")
        profile = DepthProfile(
            vendor=vendor,
            probe=probe,
            prefer_direct=True,
            scale_fallback=True,
            scale_max_required=True,
            scale_cm_bias=0.03,
            scale_mm_bias=0.06,
            direct_letter_bias=0.10,
            scale_side_preference="right",
            stable_scale_position_review=True,
            notes=tuple(notes),
        )
    if re.search(r"e14cl4b|8848|8823|18l5|6c2", low_probe):
        notes.append("sonda BK riconosciuta: mantenere bias scala/cm")
        if "bk" not in low_vendor:
            profile = DepthProfile(
                vendor=vendor,
                probe=probe,
                prefer_direct=True,
                scale_fallback=True,
                scale_max_required=True,
                scale_cm_bias=max(profile.scale_cm_bias, 0.08),
                scale_mm_bias=profile.scale_mm_bias,
                direct_letter_bias=profile.direct_letter_bias,
                scale_side_preference=profile.scale_side_preference,
                stable_scale_position_review=profile.stable_scale_position_review,
                notes=tuple(notes),
            )
    return profile if profile.notes else DepthProfile(vendor=vendor, probe=probe, notes=tuple(notes))


def _expected_depths(fss_path: Optional[Path]) -> List[float]:
    if not fss_path:
        return []
    p = fss_path.expanduser().resolve()
    if not p.exists():
        return []
    try:
        _checks, depths = parse_rect_depth_checks(p)
        return [float(v) for v in depths if math.isfinite(float(v)) and float(v) > 0]
    except Exception:
        return []


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in re.split(r"[,;|\s]+", str(text or "").strip()):
        if not part:
            continue
        try:
            value = float(part.replace(",", "."))
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            values.append(value)
    return values


def _parse_box_arg(text: str) -> Optional[Box]:
    values = _parse_float_list(text)
    if len(values) != 4:
        return None
    left, top, right, bottom = values
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _clean_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace(",", ".").strip())


def _clean_numeric_expression(text: str, *, require_unit: bool = False) -> bool:
    """True only for one number with an optional allowed depth unit.

    This deliberately rejects a trailing letter such as ``dB`` or a number
    embedded in another UI value.  It is used both as a lexical guard and as a
    check that the selected box does not contain unrelated OCR text.
    """
    unit = r"(?:\s*(?:cm|mm))" if require_unit else r"(?:\s*(?:cm|mm))?"
    return bool(re.fullmatch(rf"\d+(?:\.\d+)?{unit}\s*", _clean_ocr_text(text), flags=re.I))


def _direct_label_value_matches(text: str) -> List[Tuple[int, float]]:
    """Parse only a complete D/P/R/Depth + number expression.

    D, P and R are valid only as a single label immediately to the left of the
    number.  After the number, the only accepted letters are the optional
    ``cm``/``mm`` unit.  This makes a local label a strong signal without
    turning incidental letters or wide OCR crops into a direct depth.
    """
    raw = _clean_ocr_text(text)
    patterns = (
        r"(?i)(depth|dep|deph|dept|dpth)\s*[:=./-]?\s*(\d+(?:\.\d+)?)(?:\s*(?:cm|mm))?\s*",
        r"(?i)([dpr])\s*[:=./-]?\s*(\d+(?:\.\d+)?)(?:\s*(?:cm|mm))?\s*",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, raw)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            return [(0, value)]
    return []


def _first_direct_label_value(text: str) -> Optional[Tuple[int, float]]:
    matches = _direct_label_value_matches(text)
    return matches[0] if matches else None


def _embedded_direct_unit_expression(text: str) -> Optional[Tuple[str, float]]:
    """Extract a clean D/P/R/Depth + value + unit from a noisy OCR word.

    OCR may concatenate nearby UI labels (for example ``13-TEID65mm/M42``).
    The direct expression itself is still unambiguous, because it starts with
    D/P/R/Depth and ends immediately after the only allowed suffix, mm or cm.
    """
    raw = _clean_ocr_text(text)
    match = re.search(
        r"(?i)(depth|dep|deph|dept|dpth|[dpr])\s*[:=./-]?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm)(?![a-z])",
        raw,
    )
    if not match:
        return None
    try:
        value = float(match.group(2).replace(",", "."))
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    label = match.group(1)
    normalized_label = "Depth" if label.lower().startswith("d") and len(label) > 1 else label.upper()
    unit = match.group(3).lower()
    depth_mm = value if unit == "mm" else value * 10.0
    return f"{normalized_label}{match.group(2).replace(',', '.')} {unit}", depth_mm


def _token_matches_direct_label_value(token: DepthToken) -> float:
    first = _first_direct_label_value(token.word.text)
    if first:
        _start, value = first
    else:
        embedded = _embedded_direct_unit_expression(token.word.text)
        if not embedded:
            return 0.0
        raw_match = re.search(r"\d+(?:[.,]\d+)?", embedded[0])
        if not raw_match:
            return 0.0
        value = float(raw_match.group(0).replace(",", "."))
    tolerance = max(0.03, 0.015 * max(1.0, abs(value)))
    return 1.0 if abs(float(token.numeric_value) - value) <= tolerance else 0.0


def _bad_suffix_after_number_autonomous(text: str) -> bool:
    low = str(text or "").lower().replace(",", ".")
    first_pair = _first_direct_label_value(low)
    first_pair_start = first_pair[0] if first_pair else None
    for match in re.finditer(r"\d+(?:\.\d+)?\s*([a-z%]+)", low):
        suffix = match.group(1)
        if suffix.startswith(("cm", "mm")):
            continue
        if suffix == "c" and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*c[^a-z0-9]*", low):
            continue
        if suffix.startswith(("em", "tm")) and re.fullmatch(r"\s*(?:s0|\d+(?:\.\d+)?)\s*(?:em|tm)[^a-z0-9]*", low):
            continue
        # Hitachi often OCR-concatenates repeated label/value pairs:
        # R:9.00R:64 or R:10.0R6:64. In that case the trailing R starts
        # the next label, it is not a unit suffix for 9.00/10.0.
        if (
            first_pair_start is not None
            and suffix[:1] in {"d", "p", "r"}
            and (
                (
                    match.start() > first_pair_start
                    and re.match(r"\s*\d?\s*[:./-]?\s*\d", low[match.end(1) :])
                )
                or (match.start(1) <= first_pair_start < match.end(1))
            )
        ):
            continue
        if first_pair_start is not None and match.start() < first_pair_start:
            continue
        return True
    return False


def _has_db_unit_suffix_text(text: str) -> bool:
    low = str(text or "").lower().replace(",", ".")
    return bool(re.search(r"(?<![a-z0-9])\d+(?:\.\d+)?\s*d\s*[b8]\b", low))


def _has_forbidden_marker_autonomous(text: str) -> bool:
    return _has_forbidden_marker(text) or _has_db_unit_suffix_text(text)


def _has_non_depth_direct_marker(text: str) -> bool:
    """Reject UI modes that resemble a letter-plus-number depth label."""
    low = _clean_ocr_text(text).lower()
    return bool(
        re.search(r"\bfr\s*\d", low)
        or re.search(r"\b\d+(?:\.\d+)?\s*d\b", low)
    )


def _box_from_token_or_cluster(cluster: DepthCluster, image_path: str, profile: DepthProfile) -> Tuple[Optional[DepthToken], Box]:
    tokens = [t for t in cluster.tokens if t.word.image_path.as_posix() == image_path]
    if not tokens:
        return None, _cluster_box(cluster)

    def is_direct(token: DepthToken) -> bool:
        text = token.word.text or ""
        return bool(
            token.has_depth_hint
            or _has_depth_hint_text(text)
            or _strict_hint_left_of_number(text, "d")
            or _strict_hint_left_of_number(text, "p")
            or _strict_hint_left_of_number(text, "r")
        )

    scale_mode = profile.scale_fallback and any(t.has_scale_hint for t in tokens) and not any(is_direct(t) for t in tokens)

    def key(token: DepthToken) -> Tuple[float, float, float, float, float]:
        value = float(token.snapped_depth_mm or token.numeric_value)
        direct = 1.0 if is_direct(token) else 0.0
        direct_pair = _token_matches_direct_label_value(token)
        unit = 1.0 if token.has_cm_hint or token.has_mm_hint else 0.0
        scale = 1.0 if token.has_scale_hint else 0.0
        value_key = value if scale_mode and profile.scale_max_required else -abs(value)
        return (direct, direct_pair, unit, scale, value_key + 0.001 * float(token.word.conf))

    best = max(tokens, key=key)
    return best, _token_box(best)


def _candidate_depth_mm(token: Optional[DepthToken]) -> float:
    if token is None:
        return 0.0
    value = token.snapped_depth_mm if token.snapped_depth_mm is not None else token.numeric_value
    return float(value) if math.isfinite(float(value)) else 0.0


def _row_center(row: Dict[str, object]) -> Tuple[float, float]:
    return (
        0.5 * (_f(row.get("pred_left")) + _f(row.get("pred_right"))),
        0.5 * (_f(row.get("pred_top")) + _f(row.get("pred_bottom"))),
    )


def _row_has_strict_direct_text(row: Dict[str, object]) -> bool:
    text = str(row.get("ocr_text") or "")
    return _first_direct_label_value(text) is not None


def _row_has_depth_above_number_context(row: Dict[str, object]) -> bool:
    """Accept the rare ``Depth``-above-number layout without loosening D/P/R."""
    return _clean_numeric_expression(str(row.get("ocr_text") or "")) and _f(row.get("depth_hint_ratio")) >= 0.35


def _row_has_direct_evidence(row: Dict[str, object]) -> bool:
    return _row_has_strict_direct_text(row) or _row_has_depth_above_number_context(row)


def _unit_value_match(text: str) -> Optional[Tuple[float, str]]:
    low = str(text or "").lower().replace(",", ".")
    if re.search(r"(?<![a-z0-9])s\.\d+\s*(?:cm|c)(?![a-z])", low):
        low = re.sub(r"(?<![a-z0-9])s(?=\.\d+\s*(?:cm|c)(?![a-z]))", "3", low)
    low = re.sub(r"(?<![a-z0-9])s0(?=\s*(?:cm|c|em|tm)(?![a-z]))", "3.0", low)
    low = re.sub(r"(?<=\d)(?:em|tm)(?![a-z])", "cm", low)
    for pattern, unit in (
        (r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*cm\b", "cm"),
        (r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*mm\b", "mm"),
        (r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*c(?![a-z])", "cm_ocr_truncated"),
    ):
        match = re.search(pattern, low)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            return value, unit
    return None


def _compact_dpr_label_value(text: str) -> bool:
    compact = re.sub(r"[^a-z0-9.]", "", str(text or "").lower())
    return bool(re.search(r"[dpr][a-z]{0,2}[dpr]\d", compact))


def _row_has_unit_value_text(row: Dict[str, object]) -> bool:
    return _unit_value_match(str(row.get("ocr_text") or "")) is not None


def _row_has_explicit_depth_unit_text(row: Dict[str, object]) -> bool:
    if _has_db_unit_suffix_text(str(row.get("ocr_text") or "")):
        return False
    return _unit_value_match(str(row.get("ocr_text") or "")) is not None


def _unit_value_depth_mm(text: str) -> float:
    match = _unit_value_match(text)
    if not match:
        return 0.0
    value, unit = match
    if unit == "mm":
        return value
    if unit.startswith("cm"):
        return value if value > 35.0 else value * 10.0
    return 0.0


def _row_has_zero_depth_value_text(row: Dict[str, object]) -> bool:
    text = str(row.get("ocr_text") or "").lower().replace(",", ".")
    return bool(re.search(r"(?<![\d.])0(?:\.0+)?\s*(?:cm|mm|c|m)?(?![\d.])", text))


def _row_has_decimal_unit_value(row: Dict[str, object]) -> bool:
    text = str(row.get("ocr_text") or "").lower().replace(",", ".")
    return bool(
        re.search(r"(?<![a-z0-9])\d+\.\d+\s*(?:cm|mm|c)(?![a-z])", text)
        or re.search(r"(?<![a-z0-9])s\.\d+\s*(?:cm|c)(?![a-z])", text)
        or re.search(r"(?<![a-z0-9])s0\s*(?:cm|c|em|tm)(?![a-z])", text)
    )


def _row_has_dirty_scale_cm_text(row: Dict[str, object]) -> bool:
    text = str(row.get("ocr_text") or "").lower().replace(",", ".")
    return bool(
        re.search(r"(?<![a-z0-9])\d+\.\d+\s*(?:m|cem)(?![a-z])", text)
        or re.search(r"(?<![a-z0-9])\d{2,3}\s*(?:m|cem)(?![a-z])", text)
    )


def _scale_endpoint_corrected_depth_mm(row: Dict[str, object]) -> float:
    text = str(row.get("ocr_text") or "").lower().replace(",", ".").strip()
    current = _f(row.get("depth_mm"))
    match = re.fullmatch(r"(\d+\.\d+)\s*(?:m|cem)[^a-z0-9]*", text)
    if match:
        try:
            return float(match.group(1)) * 10.0
        except Exception:
            return current
    match = re.fullmatch(r"(\d{2,3})\s*(?:m|cem|cm)[^a-z0-9]*", text)
    if match:
        try:
            value = float(match.group(1))
        except Exception:
            return current
        if 10.0 <= value <= 160.0 and (current <= 0 or current > 160.0 or _flag(row, "ocr_bad_suffix_after_number")):
            return value
    if current > 160.0:
        return 0.0
    return current


def _row_has_scale_endpoint_number_text(row: Dict[str, object]) -> bool:
    text = str(row.get("ocr_text") or "").lower().replace(",", ".")
    if _flag(row, "ocr_multi_number"):
        return False
    if _has_fps_ips(text) or _has_forbidden_marker_autonomous(text) or _has_probe_model_marker(text) or _has_time_like_text(text):
        return False
    return bool(
        _unit_value_match(text) is not None
        or _row_has_dirty_scale_cm_text(row)
        or re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[^a-z0-9]*", text)
    )


def _looks_like_scale_edge_value(row: Dict[str, object]) -> bool:
    if str(row.get("candidate_source") or "").startswith("scale_") and str(row.get("candidate_source") or "").endswith("_column"):
        return True
    if _row_has_strict_direct_text(row):
        return False
    if not (_row_has_unit_value_text(row) or _row_has_dirty_scale_cm_text(row)):
        return False
    if _f(row.get("depth_mm")) <= 0:
        return False
    image = str(row.get("source_image") or "")
    width, height = _image_size(image) if image else (1, 1)
    cx, cy = _row_center(row)
    if width <= 1 or height <= 1:
        return False
    x_ratio = cx / float(width)
    y_ratio = cy / float(height)
    # Scale labels are usually attached to the vertical scale edge and the
    # maximum value is the bottom-most unit value. This catches right-side
    # Philips labels such as "3.0cm" and left-side scale labels without
    # turning a genuine D/P/R/Depth UI value into a scale candidate.
    if y_ratio >= 0.70 and 0.12 <= x_ratio <= 0.95:
        return True
    if y_ratio >= 0.58 and (x_ratio <= 0.28 or x_ratio >= 0.62):
        return True
    return False


def _scale_side(row: Dict[str, object]) -> str:
    source = str(row.get("candidate_source") or "")
    if source == "scale_right_column":
        return "right"
    if source == "scale_left_column":
        return "left"
    image = str(row.get("source_image") or "")
    width, _height = _image_size(image) if image else (1, 1)
    if width <= 1:
        return "unknown"
    cx, _cy = _row_center(row)
    ratio = cx / float(width)
    if ratio >= 0.56:
        return "right"
    if ratio <= 0.44:
        return "left"
    return "unknown"


def _preferred_scale_side(profile: DepthProfile) -> str:
    return _ordered_scale_sides(profile.scale_side_preference)[0]


def _opposite_scale_side(side: str) -> str:
    return "left" if side == "right" else "right"


def _row_direct_hint_ratio(row: Dict[str, object]) -> float:
    return max(
        _f(row.get("depth_hint_ratio")),
        _f(row.get("d_hint_ratio")),
        _f(row.get("p_hint_ratio")),
        _f(row.get("r_hint_ratio")),
    )


def _candidate_mode(row: Dict[str, object]) -> str:
    text = str(row.get("ocr_text") or "")
    if _looks_like_scale_edge_value(row):
        return "scale"
    if _row_has_strict_direct_text(row) or _row_has_depth_above_number_context(row):
        return "direct_label"
    unit_or_scale = (
        _f(row.get("scale_hint_ratio")) > 0
        or _f(row.get("cm_ratio")) > 0
        or _f(row.get("mm_ratio")) > 0
        or _f(row.get("ocr_has_cm_text")) > 0
        or _f(row.get("ocr_has_mm_text")) > 0
    )
    if unit_or_scale:
        return "scale"
    return "numeric_accessory"


def _strict_hint_left_of_number(text: str, letter: str) -> bool:
    low = str(text or "").lower().replace(" ", "")
    target = re.escape(letter.lower())
    return bool(re.search(rf"(?<![a-z0-9]){target}\s*[:./-]?\s*\d", low))


def _has_depth_hint_text(text: str) -> bool:
    compact = re.sub(r"[^a-z]", "", str(text or "").lower())
    return "depth" in compact or compact in {"dep", "deph", "dept", "dpth"}


def _candidate_row(
    *,
    folder: Path,
    image_path: str,
    rank: int,
    variant_index: int,
    cluster: DepthCluster,
    token: Optional[DepthToken],
    box_variant: str,
    pred_box: Box,
    profile: DepthProfile,
) -> Dict[str, object]:
    raw_ocr_text = _candidate_ocr_text(cluster, image_path, token)
    embedded_direct = _embedded_direct_unit_expression(raw_ocr_text) if token is not None else None
    ocr_text = embedded_direct[0] if embedded_direct else raw_ocr_text
    digit_groups = _digit_group_count(ocr_text)
    pred_width = pred_box[2] - pred_box[0]
    pred_height = pred_box[3] - pred_box[1]
    aspect = pred_width / max(1.0, pred_height)
    depth_mm = embedded_direct[1] if embedded_direct else _candidate_depth_mm(token)
    source = "token" if token else "cluster"
    row: Dict[str, object] = {
        "sample_key": Path(image_path).name,
        "config_folder": folder.name,
        "fss_path": "",
        "source_image": image_path,
        "setup_id": "",
        "depth_index0": "",
        "depth_mm": f"{depth_mm:.3f}" if depth_mm else "",
        "flip_state": "",
        "candidate_rank": rank + 0.01 * variant_index,
        "candidate_source": source,
        "box_variant": box_variant,
        "box_variant_expanded": int(box_variant != "base"),
        "box_variant_embedded": int(box_variant.startswith("embedded_value")),
        "label": "0",
        "iou": "0",
        "center_error_px": "0",
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
        "ocr_has_forbidden_marker": int(_has_forbidden_marker_autonomous(ocr_text)),
        "ocr_has_db_suffix": int(_has_db_unit_suffix_text(ocr_text)),
        "ocr_has_hz_text": int(bool(re.search(r"\b(?:mhz|hz)\b", ocr_text, re.I))),
        "ocr_has_percent": int("%" in ocr_text),
        "ocr_has_mi_tis": int(bool(re.search(r"\bmi\b|\btis\b", ocr_text, re.I))),
        "ocr_has_c2_text": int(bool(re.search(r"\bc\s*2\b", ocr_text, re.I))),
        "ocr_has_probe_model": int(_has_probe_model_marker(ocr_text)),
        "ocr_has_time_like_text": int(_has_time_like_text(ocr_text)),
        "ocr_bad_suffix_after_number": int(_bad_suffix_after_number_autonomous(ocr_text)),
        "ocr_has_letter_hint": int(
            _has_depth_hint_text(ocr_text)
            or _strict_hint_left_of_number(ocr_text, "d")
            or _strict_hint_left_of_number(ocr_text, "p")
            or _strict_hint_left_of_number(ocr_text, "r")
            or bool(re.search(r"\d\s*(cm|mm)\b", ocr_text, re.I))
        ),
        "ocr_text_has_d": int(_strict_hint_left_of_number(ocr_text, "d")),
        "ocr_text_has_p": int(_strict_hint_left_of_number(ocr_text, "p")),
        "ocr_text_has_r": int(_strict_hint_left_of_number(ocr_text, "r")),
        "ocr_text_has_depth": int(_has_depth_hint_text(ocr_text)),
        "ocr_d_left_of_number": int(_strict_hint_left_of_number(ocr_text, "d")),
        "ocr_d_right_of_number": int(_hint_right_of_number(ocr_text, "d")),
        "ocr_p_left_of_number": int(_strict_hint_left_of_number(ocr_text, "p")),
        "ocr_p_right_of_number": int(_hint_right_of_number(ocr_text, "p")),
        "ocr_r_left_of_number": int(_strict_hint_left_of_number(ocr_text, "r")),
        "ocr_r_right_of_number": int(_hint_right_of_number(ocr_text, "r")),
        "ocr_has_cm_text": int(bool(re.search(r"\bcm\b", ocr_text, re.I))),
        "ocr_has_mm_text": int(bool(re.search(r"\bmm\b", ocr_text, re.I))),
        "ocr_numeric_value": f"{token.numeric_value:.3f}" if token else "",
        "ocr_snapped_depth_mm": f"{float(token.snapped_depth_mm):.3f}" if token and token.snapped_depth_mm is not None else "",
        "ocr_snap_mode": token.snap_mode if token else "",
        "ocr_multi_number": int(digit_groups > 2),
        "ocr_single_depth_expr": int(
            digit_groups == 1
            and (
                _has_depth_hint_text(ocr_text)
                or _strict_hint_left_of_number(ocr_text, "d")
                or _strict_hint_left_of_number(ocr_text, "p")
                or _strict_hint_left_of_number(ocr_text, "r")
                or token is not None
            )
        ),
        "wide_text_box": int(pred_width > 220.0 and aspect > 4.0),
        "snap_error_mm": f"{token.snap_error_mm:.3f}" if token and math.isfinite(token.snap_error_mm) else "",
        "value_error_mm": "0",
        "pred_width": f"{pred_width:.2f}",
        "pred_height": f"{pred_height:.2f}",
        "pred_left": f"{pred_box[0]:.2f}",
        "pred_top": f"{pred_box[1]:.2f}",
        "pred_right": f"{pred_box[2]:.2f}",
        "pred_bottom": f"{pred_box[3]:.2f}",
        "gt_left": "",
        "gt_top": "",
        "gt_right": "",
        "gt_bottom": "",
        "review_comment": "",
        "cluster_reason": cluster.reason,
        "manual_prior_candidate": "0",
        "ranker_review_hint": "",
        "manual_hint_present": "0",
        "manual_hint_r": "0",
        "manual_hint_p": "0",
        "manual_hint_d": "0",
        "manual_hint_depth": "0",
        "manual_hint_cm": "0",
        "manual_hint_mm": "0",
        "manual_hint_scale": "0",
        "manual_hint_match_score": "0",
        "manual_hint_mismatch_score": "0",
    }
    mode = _candidate_mode(row)
    row["autonomous_mode"] = mode
    row["vendor_profile"] = profile.vendor
    row["probe_profile"] = profile.probe
    return row


def _feature_matrix(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> List[List[float]]:
    return [[_f(row.get(feature, "")) for feature in features] for row in rows]


def _image_size(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return 1, 1


def _scale_depth_mm(token: DepthToken, expected_depths: Sequence[float]) -> Tuple[float, str]:
    value = float(token.numeric_value)
    if token.has_mm_hint:
        return value, "scale_mm"
    if token.has_cm_hint:
        if value > 35.0:
            return value, "scale_cm_implicit_decimal"
        return value * 10.0, "scale_cm_x10"
    if expected_depths and token.snapped_depth_mm is not None and math.isfinite(float(token.snapped_depth_mm)):
        return float(token.snapped_depth_mm), token.snap_mode or "scale_expected_snap"
    if value <= 35.0:
        return value * 10.0, "scale_guess_cm_x10"
    return value, "scale_raw_mm"


def _ordered_scale_sides(preference: str) -> List[str]:
    pref = str(preference or "right").lower()
    if pref == "left":
        return ["left", "right"]
    if pref == "auto":
        return ["right", "left"]
    return ["right", "left"]


def _scale_side_bands(rect_echo: Optional[Box], image_size: Tuple[int, int], preference: str = "right") -> List[Tuple[str, Box]]:
    width, height = image_size
    side_order = _ordered_scale_sides(preference)
    bands_by_side: Dict[str, Box] = {}
    if rect_echo:
        left, top, right, bottom = rect_echo
        echo_w = max(1.0, right - left)
        bands_by_side["right"] = (
            max(0.0, right - max(140.0, 0.22 * echo_w)),
            max(0.0, top - 80.0),
            min(float(width), right + max(110.0, 0.10 * echo_w)),
            min(float(height), bottom + 95.0),
        )
        bands_by_side["left"] = (
            max(0.0, left - max(90.0, 0.08 * echo_w)),
            max(0.0, top - 80.0),
            min(float(width), left + max(140.0, 0.22 * echo_w)),
            min(float(height), bottom + 95.0),
        )
    else:
        bands_by_side["right"] = (0.56 * width, 0.05 * height, float(width), 0.98 * height)
        bands_by_side["left"] = (0.0, 0.05 * height, 0.44 * width, 0.98 * height)
    return [(side, bands_by_side[side]) for side in side_order if side in bands_by_side]


def _is_scale_token_candidate(token: DepthToken, band: Box) -> bool:
    text = token.word.text or ""
    if _has_fps_ips(text) or _has_forbidden_marker_autonomous(text) or _has_non_depth_direct_marker(text) or _has_probe_model_marker(text) or _has_time_like_text(text):
        return False
    if _bad_suffix_after_number_autonomous(text):
        return False
    if _has_depth_hint_text(text) or _first_direct_label_value(text):
        return False
    if _compact_dpr_label_value(text):
        return False
    if _strict_hint_left_of_number(text, "d") or _strict_hint_left_of_number(text, "p") or _strict_hint_left_of_number(text, "r"):
        return False
    x0, y0, x1, y1 = band
    return x0 <= token.word.x_center <= x1 and y0 <= token.word.y_center <= y1


def _best_scale_column(
    tokens: Sequence[DepthToken],
    rect_echo: Optional[Box],
    image_size: Tuple[int, int],
    profile: DepthProfile,
    visual_marks: Sequence[Box] = (),
) -> Optional[Tuple[str, List[DepthToken]]]:
    candidates_by_side: Dict[str, List[DepthToken]] = {}
    preferred_side = _ordered_scale_sides(profile.scale_side_preference)[0]
    for side, band in _scale_side_bands(rect_echo, image_size, profile.scale_side_preference):
        side_candidates = [t for t in tokens if _is_scale_token_candidate(t, band)]
        if visual_marks:
            # Once we see actual ticks, the scale column must live alongside
            # them.  A side band alone is too broad: it also contains values
            # from controls and the settings panel.
            side_candidates = [
                t
                for t in side_candidates
                if any(
                    abs(t.word.x_center - _box_center(mark)[0]) <= 115.0
                    and abs(t.word.y_center - _box_center(mark)[1]) <= 55.0
                    for mark in visual_marks
                )
            ]
        else:
            # Without detected ticks retain the old, confidence-based fallback
            # for uncommon layouts.  Provisional OCR digits are only trusted
            # in the explicit tick-lane branch above.
            side_candidates = [t for t in side_candidates if t.word.conf >= 18.0]
        if side_candidates:
            candidates_by_side[side] = side_candidates
    if sum(len(v) for v in candidates_by_side.values()) < 2:
        return None
    by_bin: Dict[int, List[DepthToken]] = defaultdict(list)
    side_by_bin: Dict[int, str] = {}
    for side, candidates in candidates_by_side.items():
        for token in candidates:
            key = int(round(token.word.x_center / 42.0))
            by_bin[key].append(token)
            side_by_bin[key] = side
    best: Optional[List[DepthToken]] = None
    best_side = "unknown"
    best_score = -1.0
    for key, group in by_bin.items():
        if len(group) < 2:
            continue
        ys = sorted(t.word.y_center for t in group)
        span = ys[-1] - ys[0] if ys else 0.0
        distinct_y = len({int(round(y / 24.0)) for y in ys})
        if distinct_y < 2 or span < 35.0:
            continue
        xs = [t.word.x_center for t in group]
        x_spread = max(xs) - min(xs) if xs else 999.0
        score = 2.0 * distinct_y + min(4.0, span / 90.0) - 0.03 * x_spread
        side = side_by_bin.get(key, "unknown")
        if side == preferred_side:
            score += 1.8
        else:
            score -= 0.5
        if score > best_score:
            best = group
            best_side = side
            best_score = score
    return (best_side, best) if best else None


def _add_scale_column_rows(
    rows: List[Dict[str, object]],
    *,
    folder: Path,
    image_path: Path,
    image_tokens: Sequence[DepthToken],
    rect_echo: Optional[Box],
    expected_depths: Sequence[float],
    profile: DepthProfile,
    rank_base: int,
) -> int:
    image_text = image_path.as_posix()
    size = _image_size(image_text)
    best_column = _best_scale_column(
        image_tokens,
        rect_echo,
        size,
        profile,
        visual_marks=_detect_visual_scale_marks(image_text),
    )
    if not best_column:
        return 0
    side, column = best_column
    scored = []
    for token in column:
        depth_mm, mode = _scale_depth_mm(token, expected_depths)
        if depth_mm > 0 and math.isfinite(depth_mm):
            scored.append((depth_mm, mode, token))
    # Thin digits next to tick marks are sometimes merged into a larger number
    # (for example `1` plus noise becoming `15`). If the same column already
    # contains a short regular sequence, keep its compatible values and drop a
    # distant one-digit extension rather than treating it as the scale maximum.
    raw_small = sorted({round(float(token.numeric_value), 3) for _depth, _mode, token in scored if 0 < token.numeric_value <= 12.0})
    if len(raw_small) >= 3:
        steps = [b - a for a, b in zip(raw_small, raw_small[1:]) if b - a > 0]
        step = steps[len(steps) // 2] if steps else 1.0
        ceiling = max(raw_small) + max(3.0, 3.0 * step)
        compatible = [item for item in scored if item[2].numeric_value <= ceiling]
        if compatible:
            scored = compatible
    if not scored:
        return 0
    depth_mm, mode, token = max(scored, key=lambda item: item[0])
    cluster = DepthCluster(cluster_id=-1, tokens=list(column))
    cluster.score = 9.0
    cluster.cm_ratio = 1.0 if any(t.has_cm_hint for t in column) else 0.0
    cluster.mm_ratio = 1.0 if any(t.has_mm_hint for t in column) else 0.0
    cluster.expected_ratio = 1.0 if expected_depths else 0.0
    cluster.plausible_ratio = 1.0
    cluster.support_ratio = 1.0
    cluster.scale_hint_ratio = 1.0
    cluster.side_score = 1.0
    cluster.accessory_score = 1.0
    cluster.echo_center_penalty = 0.0
    cluster.unique_values = len({round(_scale_depth_mm(t, expected_depths)[0], 1) for t in column})
    side_label = "destra" if side == "right" else "sinistra" if side == "left" else "laterale"
    cluster.reason = (
        f"scale_{side}_column; numeri/tacche allineati a {side_label} del rettangolo eco; "
        f"n={len(column)}; max_depth={depth_mm:.1f}; mode={mode}"
    )
    count = 0
    for variant_index, (box_variant, pred_box) in enumerate(_candidate_variants(_token_box(token), image_text, token)):
        row = _candidate_row(
            folder=folder,
            image_path=image_text,
            rank=rank_base,
            variant_index=variant_index,
            cluster=cluster,
            token=token,
            box_variant=box_variant,
            pred_box=pred_box,
            profile=profile,
        )
        row["candidate_source"] = f"scale_{side}_column"
        row["autonomous_mode"] = "scale"
        row["depth_mm"] = f"{depth_mm:.3f}"
        row["ocr_snapped_depth_mm"] = f"{depth_mm:.3f}"
        row["ocr_snap_mode"] = mode
        row["scale_hint_ratio"] = "1.00000"
        row["cluster_reason"] = cluster.reason
        rows.append(row)
        count += 1
    return count


def _load_ranker(path: Path) -> Tuple[Optional[object], List[str], str]:
    if not path:
        return None, FEATURES, ""
    p = path.expanduser().resolve()
    if not p.exists() or joblib is None:
        return None, FEATURES, ""
    bundle = joblib.load(p)
    return bundle.get("model"), list(bundle.get("features") or FEATURES), p.as_posix()


def _flag(row: Dict[str, object], key: str) -> bool:
    return _f(row.get(key)) >= 1.0


def _rule_score(row: Dict[str, object], profile: DepthProfile) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    mode = str(row.get("autonomous_mode") or "")
    ocr_text = str(row.get("ocr_text") or "")
    has_direct_pair = _first_direct_label_value(ocr_text) is not None
    direct = mode == "direct_label"
    scale = mode == "scale"

    if direct:
        score += 0.25
        reasons.append("preferenza valore informativo D/P/R/Depth")
        if has_direct_pair:
            score += 0.24
            reasons.append("coppia label-valore completa")
            if "hitachi" in profile.vendor.lower():
                score += 0.08
                reasons.append("profilo Hitachi R/P/D")
        elif _row_has_depth_above_number_context(row):
            score += 0.18
            reasons.append("Depth sopra/al fianco del numero pulito")
        if _row_has_unit_value_text(row):
            score += 0.16
            reasons.append("unità esplicita sul direct")
        if _flag(row, "interface_unit_promoted"):
            score += 0.22
            reasons.append("unità stabile nell'interfaccia cartella")
    elif scale and profile.scale_fallback:
        scale_strength = max(_f(row.get("scale_hint_ratio")), _f(row.get("cm_ratio")), _f(row.get("mm_ratio")), _f(row.get("ocr_has_cm_text")), _f(row.get("ocr_has_mm_text")))
        score += 0.13 * scale_strength
        reasons.append("fallback scala")
        source = str(row.get("candidate_source") or "")
        if source == "scale_right_column":
            score += 0.30
            reasons.append("colonna scala destra")
        elif source == "scale_left_column":
            score += 0.10
            reasons.append("colonna scala sinistra fallback")
        elif _row_is_in_visual_scale_lane(row) and _row_has_unit_value_text(row):
            score += 0.12
            reasons.append("valore vicino alle tacche scala")
        if profile.scale_cm_bias and _f(row.get("cm_ratio")) > 0:
            score += profile.scale_cm_bias * _f(row.get("cm_ratio"))
            reasons.append("profilo scala/cm")
        if profile.scale_mm_bias and _f(row.get("mm_ratio")) > 0:
            score += profile.scale_mm_bias * _f(row.get("mm_ratio"))
            reasons.append("profilo scala/mm")
    else:
        score -= 0.06
        reasons.append("numero accessorio senza label/scala")

    if _flag(row, "ocr_has_cm_text") or _f(row.get("cm_ratio")) > 0:
        score += 0.06
        reasons.append("suffix/unità cm")
    if _flag(row, "ocr_has_mm_text") or _f(row.get("mm_ratio")) > 0:
        score += 0.06
        reasons.append("suffix/unità mm")
    if _f(row.get("accessory_score")) > 0:
        score += 0.04 * _f(row.get("accessory_score"))
        reasons.append("posizione accessoria/laterale")
    if _f(row.get("echo_center_penalty")) > 0.20:
        penalty = 0.14 * _f(row.get("echo_center_penalty"))
        score -= penalty
        reasons.append(f"penalità centro eco {penalty:.2f}")

    if _flag(row, "ocr_has_fps_ips") or _f(row.get("fps_ips_ratio")) > 0:
        score -= 0.35
        reasons.append("reject fps/ips")
    if _flag(row, "ocr_has_forbidden_marker"):
        score -= 0.28
        reasons.append("reject marker Hz/MI/TIS/ETD/C2/%")
    if _flag(row, "ocr_has_db_suffix"):
        score -= 0.30
        reasons.append("reject suffisso dB")
    if _flag(row, "ocr_has_probe_model"):
        score -= 0.28
        reasons.append("reject modello sonda")
    if _flag(row, "ocr_has_time_like_text"):
        score -= 0.24
        reasons.append("reject timestamp/ora")
    if _flag(row, "ocr_bad_suffix_after_number"):
        score -= 0.18
        reasons.append("reject suffisso non cm/mm")
    if _flag(row, "ocr_d_right_of_number") and not _flag(row, "ocr_d_left_of_number"):
        score -= 0.18
        reasons.append("reject D dopo numero")
    if _flag(row, "ocr_p_right_of_number") and not _flag(row, "ocr_p_left_of_number"):
        score -= 0.10
        reasons.append("P dopo numero sospetto")
    if _flag(row, "ocr_r_right_of_number") and not _flag(row, "ocr_r_left_of_number"):
        score -= 0.10
        reasons.append("R dopo numero sospetto")
    if _flag(row, "ocr_multi_number") and not scale and not has_direct_pair:
        score -= 0.08
        reasons.append("troppi numeri fuori scala")
    elif _flag(row, "ocr_multi_number") and has_direct_pair:
        reasons.append("più numeri, scelta prima coppia")

    return score, reasons


def _apply_scale_max_rule(rows: List[Dict[str, object]], profile: DepthProfile) -> None:
    if not profile.scale_fallback or not profile.scale_max_required:
        return
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)
    for image_rows in by_image.values():
        scale_rows = [
            row
            for row in image_rows
            if (
                row.get("autonomous_mode") == "scale"
                and _f(row.get("autonomous_valid")) > 0
                and _row_is_in_visual_scale_lane(row)
                and _row_is_scale_endpoint(row, image_rows)
            )
        ]
        if not scale_rows:
            continue
        preferred_side = _preferred_scale_side(profile)
        fallback_side = _opposite_scale_side(preferred_side)
        preferred_rows = [row for row in scale_rows if _scale_side(row) == preferred_side and _f(row.get("depth_mm")) > 0]
        fallback_rows = [row for row in scale_rows if _scale_side(row) == fallback_side and _f(row.get("depth_mm")) > 0]
        preferred_unit_rows = [
            row
            for row in preferred_rows
            if _row_has_unit_value_text(row) or _flag(row, "ocr_has_cm_text") or _flag(row, "ocr_has_mm_text") or _f(row.get("cm_ratio")) > 0 or _f(row.get("mm_ratio")) > 0
        ]
        fallback_unit_rows = [
            row
            for row in fallback_rows
            if _row_has_unit_value_text(row) or _flag(row, "ocr_has_cm_text") or _flag(row, "ocr_has_mm_text") or _f(row.get("cm_ratio")) > 0 or _f(row.get("mm_ratio")) > 0
        ]
        preferred_decimal_unit_rows = [row for row in preferred_unit_rows if _row_has_decimal_unit_value(row)]
        fallback_decimal_unit_rows = [row for row in fallback_unit_rows if _row_has_decimal_unit_value(row)]
        scoring_rows = (
            preferred_decimal_unit_rows
            or preferred_unit_rows
            or preferred_rows
            or fallback_decimal_unit_rows
            or fallback_unit_rows
            or fallback_rows
            or [row for row in scale_rows if _f(row.get("depth_mm")) > 0]
        )
        values = [_f(row.get("depth_mm")) for row in scoring_rows if _f(row.get("depth_mm")) > 0]
        if not values:
            continue
        max_value = max(values)
        for row in scale_rows:
            side = _scale_side(row)
            in_preferred_scale_side = row in scoring_rows
            if not in_preferred_scale_side:
                row["autonomous_score"] = f"{max(0.0, min(1.0, _f(row.get('autonomous_score')) - 0.14)):.5f}"
                if preferred_rows and side == fallback_side:
                    row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; scala {fallback_side} subordinata a scala {preferred_side}"
                else:
                    row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scala fuori lato preferito"
                continue
            if abs(_f(row.get("depth_mm")) - max_value) <= max(0.5, 0.01 * max_value):
                row["autonomous_score"] = f"{max(0.0, min(1.0, _f(row.get('autonomous_score')) + 0.16)):.5f}"
                side_reason = f"scala {preferred_side}" if preferred_rows else f"scala {fallback_side} fallback" if fallback_rows else "scala laterale"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; massimo valore {side_reason}"
            else:
                row["autonomous_score"] = f"{max(0.0, min(1.0, _f(row.get('autonomous_score')) - 0.08)):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; non massimo scala"


def _enforce_scale_endpoint_rule(rows: List[Dict[str, object]]) -> None:
    """Never promote a middle tick value as the depth of a scale.

    The highest depth is necessarily one of the two scale endpoints.  This is
    deliberately independent of the folder-level unit strategy: OCR may read
    an isolated digit in the middle of a scale before it manages to read the
    ``cm``/``mm`` label.  When tick detection is unavailable we do not invent
    an endpoint from loose geometry; that case remains available for review.
    """
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    for image_rows in by_image.values():
        image = next((str(row.get("source_image") or "") for row in image_rows if row.get("source_image")), "")
        if len(_detect_visual_scale_marks(image)) < 2:
            continue
        for row in image_rows:
            if row.get("autonomous_mode") != "scale":
                continue
            if not _row_is_in_visual_scale_lane(row):
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = (
                    str(row.get("autonomous_reason") or "")
                    + "; scartato: valore fuori dalla corsia delle tacche della scala"
                )
                continue
            if _row_is_scale_endpoint(row, image_rows):
                continue
            row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
            row["autonomous_reason"] = (
                str(row.get("autonomous_reason") or "")
                + "; scartato: valore interno alla scala, la depth puo' essere solo a un'estremita'"
            )


def _enforce_scale_max_value_rule(rows: List[Dict[str, object]]) -> None:
    """Require the selected scale reading to be the largest readable endpoint.

    A learned score can be useful for OCR quality, but it must never overturn
    the scale semantics: once two endpoint values are readable, the smaller
    one is not the depth.  This also resolves duplicate OCR hypotheses on the
    same tick (for example a raw ``2`` and its ``2 cm`` interpretation).
    """
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    for image_rows in by_image.values():
        image = next((str(row.get("source_image") or "") for row in image_rows if row.get("source_image")), "")
        if len(_detect_visual_scale_marks(image)) < 2:
            continue
        endpoints = [
            row
            for row in image_rows
            if (
                row.get("autonomous_mode") == "scale"
                and _f(row.get("autonomous_valid")) > 0
                and _f(row.get("depth_mm")) > 0
                and _row_is_in_visual_scale_lane(row)
                and _row_is_scale_endpoint(row, image_rows)
                and _row_has_isolated_scale_number(row)
            )
        ]
        if len(endpoints) < 2:
            continue
        max_depth = max(_f(row.get("depth_mm")) for row in endpoints)
        tolerance = max(0.5, 0.01 * max_depth)
        for row in endpoints:
            depth = _f(row.get("depth_mm"))
            if depth >= max_depth - tolerance:
                continue
            row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
            row["autonomous_reason"] = (
                str(row.get("autonomous_reason") or "")
                + f"; scartato: {depth:g} mm non e' il massimo endpoint OCR ({max_depth:g} mm)"
            )


def _apply_folder_consistency_rules(rows: List[Dict[str, object]]) -> None:
    images = {str(row.get("source_image") or "") for row in rows if row.get("source_image")}
    image_count = max(1, len(images))
    direct_rows = [
        row
        for row in rows
        if row.get("autonomous_mode") == "direct_label" and _f(row.get("autonomous_valid")) > 0
    ]
    groups: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for row in direct_rows:
        cx = 0.5 * (_f(row.get("pred_left")) + _f(row.get("pred_right")))
        cy = 0.5 * (_f(row.get("pred_top")) + _f(row.get("pred_bottom")))
        groups[(int(round(cx / 52.0)), int(round(cy / 52.0)))].add(str(row.get("source_image") or ""))
    min_support = max(2, int(math.ceil(0.30 * image_count)))
    for row in direct_rows:
        cx = 0.5 * (_f(row.get("pred_left")) + _f(row.get("pred_right")))
        cy = 0.5 * (_f(row.get("pred_top")) + _f(row.get("pred_bottom")))
        support = len(groups.get((int(round(cx / 52.0)), int(round(cy / 52.0))), set()))
        if support >= min_support:
            row["autonomous_score"] = f"{min(1.0, _f(row.get('autonomous_score')) + 0.12):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; posizione direct stabile {support}/{image_count}"
        elif image_count >= 4:
            row["autonomous_score"] = f"{max(0.0, _f(row.get('autonomous_score')) - 0.04):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; posizione direct poco stabile"

    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)
    for image_rows in by_image.values():
        direct_best = max(
            [_f(row.get("autonomous_score")) for row in image_rows if row.get("autonomous_mode") == "direct_label" and _f(row.get("autonomous_valid")) > 0],
            default=0.0,
        )
        if direct_best < 0.52:
            continue
        for row in image_rows:
            if row.get("autonomous_mode") == "scale" and _f(row.get("autonomous_valid")) > 0:
                row["autonomous_score"] = f"{max(0.0, _f(row.get('autonomous_score')) - 0.18):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scala subordinata a valore interfaccia"


def _is_strict_direct_candidate(row: Dict[str, object]) -> bool:
    if row.get("autonomous_mode") != "direct_label" or _f(row.get("autonomous_valid")) <= 0:
        return False
    if _f(row.get("depth_mm")) <= 0:
        return False
    return _row_has_direct_evidence(row) or _flag(row, "interface_unit_promoted")


def _is_scale_strategy_candidate(row: Dict[str, object]) -> bool:
    if _f(row.get("autonomous_valid")) <= 0 or _f(row.get("depth_mm")) <= 0:
        return False
    return row.get("autonomous_mode") == "scale" and _row_is_in_visual_scale_lane(row)


def _value_bucket(value: float) -> float:
    return round(float(value) * 2.0) / 2.0


def _best_by_score(rows: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    best: Optional[Dict[str, object]] = None
    best_score = -1.0
    for row in rows:
        score = _f(row.get("autonomous_score"))
        if score > best_score:
            best = row
            best_score = score
    return best


def _row_has_direct_unit_evidence(row: Dict[str, object]) -> bool:
    """True only for the full direct grammar with an explicit depth unit."""
    text = _clean_ocr_text(str(row.get("ocr_text") or ""))
    return bool(
        re.fullmatch(
            r"(?i)(?:depth|dep|deph|dept|dpth|[dpr])\s*[:=./-]?\s*\d+(?:\.\d+)?\s*(?:mm|cm)",
            text,
        )
    )


def _apply_folder_direct_unit_strategy(rows: List[Dict[str, object]]) -> None:
    """Lock a repeated, varying D/P/R/Depth + unit method for a folder.

    This is stronger than a single-image ranker score.  If an explicit direct
    expression occurs across most images and its value changes, it is the
    common depth method for that folder.  A fixed accessory value such as
    ``Print 4`` cannot compete with it.
    """
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 4:
        return
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    selected_by_image: Dict[str, Dict[str, object]] = {}
    for image, image_rows in by_image.items():
        candidates = [
            row
            for row in image_rows
            if (
                row.get("autonomous_mode") == "direct_label"
                and _f(row.get("autonomous_valid")) > 0
                and _f(row.get("depth_mm")) > 0
                and _row_has_direct_unit_evidence(row)
            )
        ]
        if not candidates:
            continue
        selected_by_image[image] = max(
            candidates,
            key=lambda row: (
                1 if str(row.get("box_variant") or "") == "direct_label_unit_window" else 0,
                _f(row.get("autonomous_score")),
                _f(row.get("ranker_score")),
            ),
        )

    min_support = max(4, int(math.ceil(0.50 * image_count)))
    if len(selected_by_image) < min_support:
        return
    values = {_value_bucket(_f(row.get("depth_mm"))) for row in selected_by_image.values() if _f(row.get("depth_mm")) > 0}
    if len(values) < 3 or max(values) - min(values) < 4.0:
        return

    strategy_reason = (
        "strategia cartella: direct D/P/R/Depth + cm/mm variabile "
        f"{len(selected_by_image)}/{image_count}; altri candidati subordinati"
    )
    for image, image_rows in by_image.items():
        selected = selected_by_image.get(image)
        if selected is None:
            # The folder has already established the direct method.  If one
            # frame loses the D/P/R glyph but still reads a clean unit value,
            # keep that frame in the direct workflow as a review candidate
            # rather than silently falling back to the scale.
            unit_fallbacks = [
                row
                for row in image_rows
                if (
                    str(row.get("candidate_source") or "") == "token"
                    and _f(row.get("autonomous_valid")) > 0
                    and not _flag(row, "ocr_has_forbidden_marker")
                    and not _flag(row, "ocr_has_db_suffix")
                    and not _flag(row, "ocr_has_time_like_text")
                    and _unit_value_depth_mm(str(row.get("ocr_text") or "")) > 0
                    and _unit_value_depth_mm(str(row.get("ocr_text") or "")) <= 350.0
                )
            ]
            if unit_fallbacks:
                selected = max(
                    unit_fallbacks,
                    key=lambda row: (
                        _unit_value_depth_mm(str(row.get("ocr_text") or "")),
                        _f(row.get("autonomous_score")),
                    ),
                )
                fallback_depth = _unit_value_depth_mm(str(selected.get("ocr_text") or ""))
                selected["autonomous_mode"] = "direct_label"
                selected["depth_mm"] = f"{fallback_depth:.3f}"
                selected["ocr_snapped_depth_mm"] = f"{fallback_depth:.3f}"
                selected["autonomous_valid"] = "1"
                selected["autonomous_score"] = "0.58000"
                selected["folder_strategy"] = "direct_label_unit_stable"
                selected["autonomous_reason"] = (
                    str(selected.get("autonomous_reason") or "")
                    + "; review direct: la cartella usa D/P/R/Depth + unita', marker non letto in questo frame"
                )
            else:
                continue
        selected["folder_strategy"] = "direct_label_unit_stable"
        selected["autonomous_valid"] = "1"
        if _row_has_direct_unit_evidence(selected):
            selected["autonomous_score"] = f"{max(_f(selected.get('autonomous_score')), 0.92):.5f}"
        selected["autonomous_reason"] = str(selected.get("autonomous_reason") or "") + f"; {strategy_reason}"
        for row in image_rows:
            if row is selected:
                continue
            if row.get("autonomous_mode") in {"scale", "numeric_accessory", "direct_label"}:
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; subordinato a direct D/P/R/Depth + unita' della cartella"


def _direct_d_value_without_unit(row: Dict[str, object]) -> Optional[float]:
    """Parse the strict no-unit direct form used by a few vendor interfaces."""
    text = _clean_ocr_text(str(row.get("ocr_text") or ""))
    match = re.fullmatch(r"(?i)d\s*[:=./-]?\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _apply_folder_direct_d_strategy(rows: List[Dict[str, object]]) -> None:
    """Use repeated, varying ``D + number`` as one folder-wide method.

    Without cm/mm a D label remains a strong direct clue only when it appears
    across the folder with changing values.  Small decimal values in that
    coherent method are interpreted as centimetres (D 2.0 -> 20 mm), matching
    the vendor's unitless depth presentation.
    """
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 4:
        return
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    selected_by_image: Dict[str, Tuple[Dict[str, object], float]] = {}
    for image, image_rows in by_image.items():
        candidates: List[Tuple[Dict[str, object], float]] = []
        for row in image_rows:
            if row.get("autonomous_mode") != "direct_label" or _f(row.get("autonomous_valid")) <= 0:
                continue
            value = _direct_d_value_without_unit(row)
            if value is not None:
                candidates.append((row, value))
        if candidates:
            selected_by_image[image] = max(candidates, key=lambda item: (_f(item[0].get("autonomous_score")), _f(item[0].get("ranker_score"))))

    min_support = max(4, int(math.ceil(0.50 * image_count)))
    if len(selected_by_image) < min_support:
        return
    values = {round(value, 3) for _row, value in selected_by_image.values()}
    if len(values) < 3 or max(values) - min(values) < 0.8:
        return
    implicit_cm = max(values) <= 16.0
    factor = 10.0 if implicit_cm else 1.0
    strategy_reason = (
        "strategia cartella: direct D + valore variabile senza unita' "
        f"{len(selected_by_image)}/{image_count}; "
        + ("interpretazione cm implicita" if implicit_cm else "interpretazione mm")
    )

    for image, image_rows in by_image.items():
        selected_pair = selected_by_image.get(image)
        if selected_pair is None:
            continue
        selected, raw_value = selected_pair
        depth_mm = raw_value * factor
        selected["autonomous_mode"] = "direct_label"
        selected["autonomous_valid"] = "1"
        selected["depth_mm"] = f"{depth_mm:.3f}"
        selected["ocr_snapped_depth_mm"] = f"{depth_mm:.3f}"
        selected["ocr_snap_mode"] = "direct_d_implicit_cm_x10" if implicit_cm else "direct_d_raw_mm"
        selected["folder_strategy"] = "direct_label_d_stable"
        selected["autonomous_score"] = f"{max(_f(selected.get('autonomous_score')), 0.92):.5f}"
        selected["autonomous_reason"] = str(selected.get("autonomous_reason") or "") + f"; {strategy_reason}"
        for row in image_rows:
            if row is selected:
                continue
            if row.get("autonomous_mode") in {"scale", "numeric_accessory", "direct_label"}:
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; subordinato a direct D variabile della cartella"


def _apply_folder_strategy_rules(rows: List[Dict[str, object]]) -> None:
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 4:
        return

    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    strict_direct_best: Dict[str, Dict[str, object]] = {}
    scale_best: Dict[str, Dict[str, object]] = {}
    for image in images:
        items = by_image.get(image, [])
        direct = _best_by_score(row for row in items if _is_strict_direct_candidate(row))
        scale = _best_by_score(row for row in items if _is_scale_strategy_candidate(row))
        if direct:
            strict_direct_best[image] = direct
        if scale:
            scale_best[image] = scale

    min_direct_support = max(4, int(math.ceil(0.35 * image_count)))
    min_scale_support = max(4, int(math.ceil(0.45 * image_count)))
    if len(strict_direct_best) < min_direct_support or len(scale_best) < min_scale_support:
        return

    direct_values = [_value_bucket(_f(row.get("depth_mm"))) for row in strict_direct_best.values()]
    scale_values = [_value_bucket(_f(row.get("depth_mm"))) for row in scale_best.values()]
    if not direct_values or not scale_values:
        return

    direct_counts = Counter(direct_values)
    scale_unique = {value for value in scale_values if value > 0}
    direct_majority = direct_counts.most_common(1)[0][1] / float(len(direct_values))
    direct_constant = direct_majority >= 0.68 and len(direct_counts) <= max(2, int(math.ceil(0.16 * len(direct_values))))
    scale_varies = len(scale_unique) >= max(3, int(math.ceil(0.18 * len(scale_values)))) and (max(scale_unique) - min(scale_unique) >= 8.0)
    if not (direct_constant and scale_varies):
        return

    strategy_reason = (
        "strategia cartella: valore D/P/R/Depth quasi costante "
        f"({direct_counts.most_common(1)[0][0]:g}) mentre massimo scala varia "
        f"({len(scale_unique)} valori)"
    )
    for row in rows:
        if _is_scale_strategy_candidate(row):
            row["autonomous_mode"] = "scale"
            row["folder_strategy"] = "scale_from_variable_scale"
            reason = str(row.get("autonomous_reason") or "")
            if "non massimo scala" in reason or "subordinata a scala destra" in reason or "fuori lato preferito" in reason:
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.58):.5f}"
            else:
                boost = 0.36 if _looks_like_scale_edge_value(row) else 0.24
                row["autonomous_score"] = f"{min(1.0, _f(row.get('autonomous_score')) + boost):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; {strategy_reason}"
        elif _is_strict_direct_candidate(row):
            row["folder_strategy"] = "scale_from_variable_scale"
            row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.24):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; allarme: {strategy_reason}"


def _is_scale_unit_strategy_candidate(row: Dict[str, object]) -> bool:
    if row.get("autonomous_mode") != "scale":
        return False
    if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
        return False
    if _flag(row, "ocr_bad_suffix_after_number") and not _row_has_dirty_scale_cm_text(row):
        return False
    if not (_row_has_explicit_depth_unit_text(row) or _row_has_dirty_scale_cm_text(row)):
        return False
    return _row_has_isolated_scale_number(row) and _scale_unit_strategy_depth_mm(row) > 0 and _row_is_in_visual_scale_lane(row)


def _scale_unit_strategy_depth_mm(row: Dict[str, object]) -> float:
    current = _f(row.get("depth_mm"))
    if 0 < current <= 160.0:
        return current
    source = str(row.get("candidate_source") or "")
    if not (source == "token" or source.startswith("scale_")):
        return 0.0
    text = str(row.get("ocr_text") or "")
    # Cluster OCR sometimes returns strings like "4cm | 4em"; without a
    # normalized token value that is not a single depth candidate.
    if _flag(row, "ocr_multi_number"):
        return 0.0
    parsed = _unit_value_depth_mm(text)
    if 0 < parsed <= 160.0:
        return parsed
    if _row_has_dirty_scale_cm_text(row):
        corrected = _scale_endpoint_corrected_depth_mm(row)
        if 0 < corrected <= 160.0:
            return corrected
    return 0.0


def _row_has_isolated_scale_number(row: Dict[str, object]) -> bool:
    """Require a real OCR number and its own box, never an aggregate cluster."""
    source = str(row.get("candidate_source") or "")
    if not (source == "token" or source.startswith("scale_")):
        return False
    text = _clean_ocr_text(str(row.get("ocr_text") or "")).lower()
    # The tick can be fused to the digit (``6-``) and a tight OCR crop can
    # leave harmless punctuation after a correct unit (``6cm*``).  Keep those
    # local artifacts, but never admit an additional alphabetic suffix.
    return bool(re.fullmatch(r"\d+(?:\.\d*)?(?:[-–])?\s*(?:cm|mm|c|em|tm)?[.,*]*", text))


def _scale_unit_strategy_key(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    variant = str(row.get("box_variant") or "")
    base_bonus = 0.08 if variant == "base" else 0.0
    width = _f(row.get("pred_width"))
    compact_bonus = max(0.0, min(0.04, (170.0 - width) / 2500.0)) if width > 0 else 0.0
    return (_scale_unit_strategy_depth_mm(row), _f(row.get("autonomous_score")) + base_bonus + compact_bonus, _f(row.get("ranker_score")), -width)


def _scale_unit_numeric_fallback_depth_mm(row: Dict[str, object]) -> float:
    current = _f(row.get("depth_mm"))
    text = str(row.get("ocr_text") or "").lower().replace(",", ".").strip()
    if current > 160.0 and re.fullmatch(r"\d+(?:\.\d+)?\W*", text):
        try:
            raw = float(re.search(r"\d+(?:\.\d+)?", text).group(0))
        except Exception:
            raw = 0.0
        if 0.5 < raw <= 160.0:
            return raw
    return current


def _is_scale_unit_numeric_fallback_candidate(row: Dict[str, object], lane_min_x: float, lane_max_x: float) -> bool:
    if row.get("autonomous_mode") != "scale":
        return False
    if _f(row.get("autonomous_valid")) <= 0:
        return False
    if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
        return False
    if _flag(row, "ocr_bad_suffix_after_number") and not _row_has_dirty_scale_cm_text(row):
        return False
    if _row_has_explicit_depth_unit_text(row) or _row_has_dirty_scale_cm_text(row):
        return False
    if not _row_has_isolated_scale_number(row):
        return False
    if not _row_is_in_visual_scale_lane(row):
        return False
    depth = _scale_unit_numeric_fallback_depth_mm(row)
    if depth <= 0 or depth > 160.0:
        return False
    image = str(row.get("source_image") or "")
    width, height = _image_size(image) if image else (1, 1)
    if width <= 1 or height <= 1:
        return False
    cx, cy = _row_center(row)
    if not (lane_min_x - 85.0 <= cx <= lane_max_x + 85.0):
        return False
    if cx / float(width) > 0.90:
        return False
    y_ratio = cy / float(height)
    if y_ratio < 0.12 or y_ratio > 0.78:
        return False
    return True


def _scale_unit_numeric_fallback_key(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    variant = str(row.get("box_variant") or "")
    base_bonus = 0.08 if variant == "base" else 0.0
    width = _f(row.get("pred_width"))
    return (_scale_unit_numeric_fallback_depth_mm(row), _f(row.get("ranker_score")) + base_bonus, _f(row.get("autonomous_score")), -width)


def _scale_unit_cluster_review_depth_mm(row: Dict[str, object]) -> float:
    if _f(row.get("depth_mm")) > 0:
        return 0.0
    if _flag(row, "ocr_multi_number"):
        return 0.0
    text = str(row.get("ocr_text") or "")
    low = text.lower().replace(",", ".")
    low = re.sub(r"(?<=\d)(?:em|tm)(?![a-z])", "cm", low)
    if not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:cm|mm|c)[^a-z0-9]*", low):
        return 0.0
    depth = _unit_value_depth_mm(text)
    return depth if 0 < depth <= 160.0 else 0.0


def _is_scale_unit_cluster_review_candidate(row: Dict[str, object]) -> bool:
    source = str(row.get("candidate_source") or "")
    if source == "token" or source.startswith("scale_"):
        return False
    if row.get("autonomous_mode") != "scale":
        return False
    if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
        return False
    if _flag(row, "ocr_bad_suffix_after_number") and not _row_has_dirty_scale_cm_text(row):
        return False
    reason = str(row.get("autonomous_reason") or "")
    if "fallback scala" not in reason and "colonna scala" not in reason:
        return False
    image = str(row.get("source_image") or "")
    width, height = _image_size(image) if image else (1, 1)
    if width <= 1 or height <= 1:
        return False
    _cx, cy = _row_center(row)
    y_ratio = cy / float(height)
    if y_ratio < 0.12 or y_ratio > 0.78:
        return False
    return _scale_unit_cluster_review_depth_mm(row) > 0


def _scale_unit_cluster_review_key(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    variant = str(row.get("box_variant") or "")
    base_bonus = 0.08 if variant == "base" else 0.0
    width = _f(row.get("pred_width"))
    return (_scale_unit_cluster_review_depth_mm(row), _f(row.get("ranker_score")) + base_bonus, _f(row.get("autonomous_score")), -width)


@lru_cache(maxsize=512)
def _detect_visual_scale_marks(image: str) -> Tuple[Box, ...]:
    if cv2 is None or not image:
        return tuple()
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return tuple()
    height, width = im.shape[:2]
    if width <= 1 or height <= 1:
        return tuple()
    x0, y0, x1, y1 = int(0.45 * width), int(0.12 * height), int(0.90 * width), int(0.78 * height)
    crop = im[y0:y1, x0:x1]
    _ret, thresh = cv2.threshold(crop, 180, 255, cv2.THRESH_BINARY)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8)
    components: List[Tuple[float, float, float, float, float, float, float]] = []
    tick_points: List[Tuple[float, float]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < 5 or area > 1500 or w < 2 or h < 2:
            continue
        cx = x0 + float(centroids[idx][0])
        cy = y0 + float(centroids[idx][1])
        gx = float(x0 + x)
        gy = float(y0 + y)
        components.append((gx, gy, float(w), float(h), float(area), cx, cy))
        if 3 <= w <= 18 and 2 <= h <= 12 and 5 <= area <= 180 and 0.55 * width <= cx <= 0.90 * width:
            tick_points.append((cx, cy))
    if not tick_points:
        return tuple()
    x_groups: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for cx, cy in tick_points:
        x_groups[int(round(cx / 12.0))].append((cx, cy))
    best_group: List[Tuple[float, float]] = []
    for group in x_groups.values():
        ys = [cy for _cx, cy in group]
        distinct_y = len({int(round(y / 30.0)) for y in ys})
        span = max(ys) - min(ys) if ys else 0.0
        if distinct_y < 2 or span < 80.0:
            continue
        if not best_group or (len(group), span) > (len(best_group), max(y for _x, y in best_group) - min(y for _x, y in best_group)):
            best_group = group
    if not best_group:
        return tuple()
    lane_xs = sorted(cx for cx, _cy in best_group)
    lane_x = lane_xs[len(lane_xs) // 2]
    near_lane = sorted(
        [c for c in components if lane_x - 95.0 <= c[5] <= lane_x + 25.0 and 0.12 * height <= c[6] <= 0.78 * height],
        key=lambda c: c[6],
    )
    rows: List[Dict[str, object]] = []
    for component in near_lane:
        for row in rows:
            if abs(float(row["cy"]) - component[6]) <= 22.0:
                items = row["items"]
                assert isinstance(items, list)
                items.append(component)
                row["cy"] = sum(float(item[6]) for item in items) / float(len(items))
                break
        else:
            rows.append({"cy": component[6], "items": [component]})
    marks: List[Box] = []
    for row in rows:
        items = row["items"]
        if not isinstance(items, list):
            continue
        has_tick = any(abs(float(item[5]) - lane_x) <= 10.0 and 3 <= item[2] <= 18 and 2 <= item[3] <= 12 and item[4] <= 180 for item in items)
        has_left_text = any(float(item[5]) < lane_x - 8.0 and item[4] >= 5 for item in items)
        if not has_tick or not has_left_text:
            continue
        left = min(float(item[0]) for item in items)
        top = min(float(item[1]) for item in items)
        right = max(float(item[0]) + float(item[2]) for item in items)
        bottom = max(float(item[1]) + float(item[3]) for item in items)
        if right - left > 120.0 or bottom - top > 45.0:
            continue
        marks.append((max(0.0, left - 5.0), max(0.0, top - 6.0), min(float(width), right + 8.0), min(float(height), bottom + 6.0)))
    marks.sort(key=lambda box: 0.5 * (box[1] + box[3]))
    return tuple(marks)


def _row_is_in_visual_scale_lane(row: Dict[str, object]) -> bool:
    """Require scale values to sit alongside a detected vertical tick lane.

    A detected lane is a hard spatial guard: it removes settings-panel values
    that happen to have ``cm``/``mm``.  When no lane can be detected we retain
    the older, conservative source/edge fallback so uncommon vendor layouts
    are reviewed rather than silently discarded.
    """
    image = str(row.get("source_image") or "")
    marks = _detect_visual_scale_marks(image)
    source = str(row.get("candidate_source") or "")
    if not marks:
        return source.startswith("scale_") or _looks_like_scale_edge_value(row)
    cx, cy = _row_center(row)
    return any(abs(cx - _box_center(mark)[0]) <= 115.0 and abs(cy - _box_center(mark)[1]) <= 55.0 for mark in marks)


def _box_center(box: Box) -> Tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _row_scale_mark_index(row: Dict[str, object], marks: Sequence[Box]) -> Optional[int]:
    if not marks:
        return None
    cx, cy = _row_center(row)
    nearby = [
        (idx, abs(cy - _box_center(mark)[1]), abs(cx - _box_center(mark)[0]))
        for idx, mark in enumerate(marks)
        if abs(cx - _box_center(mark)[0]) <= 115.0 and abs(cy - _box_center(mark)[1]) <= 55.0
    ]
    if not nearby:
        return None
    return min(nearby, key=lambda item: (item[1], item[2]))[0]


def _scale_endpoint_indices(image_rows: Sequence[Dict[str, object]]) -> Tuple[int, ...]:
    """Return the only marker positions eligible to be the scale maximum."""
    image = next((str(row.get("source_image") or "") for row in image_rows if row.get("source_image")), "")
    marks = _detect_visual_scale_marks(image)
    if len(marks) < 2:
        return tuple()
    last = len(marks) - 1
    zero_at = {0: False, last: False}
    for row in image_rows:
        idx = _row_scale_mark_index(row, marks)
        if idx in zero_at and _row_has_zero_depth_value_text(row):
            zero_at[idx] = True
    if zero_at[0] and not zero_at[last]:
        return (last,)
    if zero_at[last] and not zero_at[0]:
        return (0,)
    return (0, last)


def _row_is_scale_endpoint(row: Dict[str, object], image_rows: Sequence[Dict[str, object]]) -> bool:
    """A maximum scale value is always on one of the two vertical endpoints."""
    image = str(row.get("source_image") or "")
    marks = _detect_visual_scale_marks(image)
    if len(marks) >= 2:
        idx = _row_scale_mark_index(row, marks)
        return idx is not None and idx in _scale_endpoint_indices(image_rows)

    # Conservative fallback for layouts where tick detection failed: use only
    # the top/bottom rows of the already validated scale lane.
    lane_rows = [candidate for candidate in image_rows if _row_is_in_visual_scale_lane(candidate)]
    ys = [_row_center(candidate)[1] for candidate in lane_rows]
    if len(ys) < 2:
        return False
    top_y, bottom_y = min(ys), max(ys)
    cy = _row_center(row)[1]
    top_zero = any(abs(_row_center(candidate)[1] - top_y) <= 42.0 and _row_has_zero_depth_value_text(candidate) for candidate in lane_rows)
    bottom_zero = any(abs(_row_center(candidate)[1] - bottom_y) <= 42.0 and _row_has_zero_depth_value_text(candidate) for candidate in lane_rows)
    if top_zero and not bottom_zero:
        return abs(cy - bottom_y) <= 42.0
    if bottom_zero and not top_zero:
        return abs(cy - top_y) <= 42.0
    return abs(cy - top_y) <= 42.0 or abs(cy - bottom_y) <= 42.0


def _row_box(row: Dict[str, object]) -> Box:
    return (
        _f(row.get("pred_left")),
        _f(row.get("pred_top")),
        _f(row.get("pred_right")),
        _f(row.get("pred_bottom")),
    )


def _row_near_box(row: Dict[str, object], box: Box, *, x_tol: float = 85.0, y_tol: float = 55.0) -> bool:
    cx, cy = _row_center(row)
    bx, by = _box_center(box)
    return abs(cx - bx) <= x_tol and abs(cy - by) <= y_tol


def _scale_mark_depth_from_nearby_rows(mark: Box, image_rows: Sequence[Dict[str, object]]) -> float:
    mx, my = _box_center(mark)
    depths: List[float] = []
    for row in image_rows:
        if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
            continue
        if row.get("autonomous_mode") != "scale" and str(row.get("candidate_source") or "") != "token":
            continue
        depth = _scale_unit_strategy_depth_mm(row) or _scale_unit_numeric_fallback_depth_mm(row)
        if depth <= 0 or depth > 160.0:
            continue
        if depth < 10.0 and not _row_has_explicit_depth_unit_text(row):
            depth *= 10.0
        cx, cy = _row_center(row)
        if abs(cx - mx) <= 85.0 and abs(cy - my) <= 55.0:
            depths.append(depth)
    return max(depths) if depths else 0.0


def _scale_mark_zero_hint(mark: Box, image_rows: Sequence[Dict[str, object]]) -> bool:
    mx, my = _box_center(mark)
    for row in image_rows:
        cx, cy = _row_center(row)
        if abs(cx - mx) > 85.0 or abs(cy - my) > 55.0:
            continue
        text = str(row.get("ocr_text") or "").lower().replace(",", ".")
        if _row_has_zero_depth_value_text(row) or re.fullmatch(r"\s*0(?:\.0+)?\s*(?:cm|mm|c|m)?[^a-z0-9]*", text):
            return True
    return False


def _visual_scale_endpoint_choice(marks: Sequence[Box], mark_depths: Sequence[float], zero_hints: Sequence[bool]) -> Tuple[int, float, str]:
    last = len(marks) - 1
    if zero_hints and zero_hints[0] and not zero_hints[last]:
        return last, 0.0, "visual_scale_endpoint_opposite_zero"
    if zero_hints and zero_hints[last] and not zero_hints[0]:
        return 0, 0.0, "visual_scale_endpoint_opposite_zero"

    valid = [(idx, depth) for idx, depth in enumerate(mark_depths) if depth > 0]
    if len(valid) >= 2:
        first_idx, first_val = valid[0]
        last_idx, last_val = valid[-1]
        if last_val > first_val:
            return last, 0.0, "visual_scale_endpoint_monotonic_down"
        if last_val < first_val:
            return 0, 0.0, "visual_scale_endpoint_monotonic_up"
        return (last if last_idx >= first_idx else 0), 0.0, "visual_scale_endpoint_monotonic_tie"
    if len(valid) == 1:
        idx, _depth = valid[0]
        # With a single interior OCR value we cannot trust its magnitude, but
        # the scale is monotonic, so the maximum is one of the endpoints. The
        # usual ultrasound orientation grows downward; if the single value is
        # already at an endpoint, use that endpoint.
        if idx == 0 or idx == last:
            return idx, mark_depths[idx], "visual_scale_endpoint_single_ocr"
        return last, 0.0, "visual_scale_endpoint_single_ocr_default_down"
    first_depth = mark_depths[0] if mark_depths else 0.0
    last_depth = mark_depths[last] if mark_depths else 0.0
    if first_depth > 0 or last_depth > 0:
        index = 0 if first_depth >= last_depth else last
        return index, mark_depths[index], "visual_scale_endpoint_ocr"
    return last, 0.0, "visual_scale_endpoint_default_down"


def _infer_visual_scale_endpoint_depth(index: int, mark_depths: Sequence[float], zero_hints: Sequence[bool]) -> float:
    depth = mark_depths[index] if 0 <= index < len(mark_depths) else 0.0
    last = len(mark_depths) - 1
    if zero_hints and zero_hints[0] and index == last:
        return float(last * 10)
    if zero_hints and zero_hints[last] and index == 0:
        return float(last * 10)
    valid = [(idx, value) for idx, value in enumerate(mark_depths) if value > 0]
    if valid:
        nearest_idx, nearest_depth = min(valid, key=lambda item: abs(item[0] - index))
        inferred = nearest_depth + abs(index - nearest_idx) * 10.0
        if 0 < inferred <= 160.0 and inferred > depth:
            return inferred
    if depth > 0:
        return depth
    inferred = float((last + 1) * 10 if index == last else last * 10)
    return inferred if 0 < inferred <= 160.0 else 0.0


def _visual_scale_max_candidate(image: str, image_rows: Sequence[Dict[str, object]]) -> Optional[Tuple[Box, float, str]]:
    marks = list(_detect_visual_scale_marks(image))
    if len(marks) < 2:
        return None
    mark_depths = [_scale_mark_depth_from_nearby_rows(mark, image_rows) for mark in marks]
    zero_hints = [_scale_mark_zero_hint(mark, image_rows) for mark in marks]
    if mark_depths and mark_depths[0] <= 0 and marks[0][2] - marks[0][0] > 35.0:
        zero_hints[0] = True
    if mark_depths and mark_depths[-1] <= 0 and marks[-1][2] - marks[-1][0] > 35.0:
        zero_hints[-1] = True
    index, depth, mode = _visual_scale_endpoint_choice(marks, mark_depths, zero_hints)
    if depth <= 0:
        depth = _infer_visual_scale_endpoint_depth(index, mark_depths, zero_hints)
    if depth > 0:
        return marks[index], depth, mode

    # If OCR gives no numeric help, use this only as review and still pick an
    # endpoint, because scale values are monotonic and the maximum is there.
    if len(marks) > 5:
        return None
    index = len(marks) - 1
    depth = float(len(marks) * 10)
    if depth <= 0 or depth > 160.0:
        return None
    return marks[index], depth, "visual_scale_endpoint_inferred"


def _add_visual_scale_review_row(rows: List[Dict[str, object]], image: str, image_rows: Sequence[Dict[str, object]], strategy_reason: str) -> Optional[Dict[str, object]]:
    visual = _visual_scale_max_candidate(image, image_rows)
    if not visual or not image_rows:
        return None
    box, depth, mode = visual
    template = dict(image_rows[0])
    left, top, right, bottom = box
    template.update(
        {
            "depth_mm": f"{depth:.3f}",
            "candidate_rank": "0.5",
            "candidate_source": "visual_scale_mark",
            "box_variant": "base",
            "box_variant_expanded": "0",
            "box_variant_embedded": "0",
            "ocr_text": f"scala visuale max {depth / 10.0:g}cm",
            "ocr_numeric_value": f"{depth / 10.0:g}",
            "ocr_snapped_depth_mm": f"{depth:.3f}",
            "ocr_snap_mode": mode,
            "ocr_multi_number": "0",
            "pred_left": f"{left:.2f}",
            "pred_top": f"{top:.2f}",
            "pred_right": f"{right:.2f}",
            "pred_bottom": f"{bottom:.2f}",
            "pred_width": f"{right - left:.2f}",
            "pred_height": f"{bottom - top:.2f}",
            "autonomous_mode": "scale",
            "autonomous_valid": "1",
            "folder_strategy": "scale_unit_required",
            "ranker_score": "0.00000",
            "autonomous_rule_delta": "0.00000",
            "autonomous_score": "0.61000",
            "autonomous_status": "review",
            "autonomous_reason": f"review: massimo scala ricavato dai valori/tacche visibili; {strategy_reason}",
            "cluster_reason": "visual_scale_mark",
        }
    )
    rows.append(template)
    return template


def _folder_prefers_scale(rows: Sequence[Dict[str, object]]) -> bool:
    """Choose one family-level strategy before applying the unit rule.

    A stable, changing direct value wins.  A stable direct value that is nearly
    constant while the tick-aligned scale varies is treated as accessory UI and
    the scale wins for the whole folder.  Weak/scattered direct detections never
    suppress a supported scale.
    """
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 4:
        return False
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)

    direct_by_image: Dict[str, Dict[str, object]] = {}
    scale_by_image: Dict[str, Dict[str, object]] = {}
    for image, image_rows in by_image.items():
        direct = _best_by_score(row for row in image_rows if _is_strict_direct_candidate(row))
        scale = _best_by_score(row for row in image_rows if _is_scale_strategy_candidate(row))
        if direct:
            direct_by_image[image] = direct
        if scale:
            scale_by_image[image] = scale

    min_direct = max(4, int(math.ceil(0.35 * image_count)))
    min_scale = max(4, int(math.ceil(0.20 * image_count)))
    if len(scale_by_image) < min_scale:
        return False
    if len(direct_by_image) < min_direct:
        return True

    groups: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for image, row in direct_by_image.items():
        cx, cy = _row_center(row)
        groups[(int(round(cx / 52.0)), int(round(cy / 52.0)))].add(image)
    best_group = max(groups, key=lambda key: len(groups[key])) if groups else None
    stable_images = groups.get(best_group, set()) if best_group is not None else set()
    if len(stable_images) < min_direct:
        return True

    direct_values = [_value_bucket(_f(direct_by_image[image].get("depth_mm"))) for image in stable_images]
    scale_values = [_value_bucket(_f(row.get("depth_mm"))) for row in scale_by_image.values()]
    direct_counts = Counter(value for value in direct_values if value > 0)
    scale_unique = {value for value in scale_values if value > 0}
    if not direct_counts or not scale_unique:
        return False
    direct_majority = direct_counts.most_common(1)[0][1] / float(len(direct_values))
    direct_constant = direct_majority >= 0.68 and len(direct_counts) <= max(2, int(math.ceil(0.16 * len(direct_values))))
    scale_varies = len(scale_unique) >= max(3, int(math.ceil(0.18 * len(scale_values)))) and (max(scale_unique) - min(scale_unique) >= 8.0)
    return direct_constant and scale_varies


def _apply_folder_scale_unit_strategy(rows: List[Dict[str, object]]) -> None:
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 4 or not _folder_prefers_scale(rows):
        return

    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        image = str(row.get("source_image") or "")
        by_image[image].append(row)

    unit_by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for image, image_rows in by_image.items():
        unit_by_image[image] = [
            row
            for row in image_rows
            if _is_scale_unit_strategy_candidate(row) and _row_is_scale_endpoint(row, image_rows)
        ]
    unit_by_image = defaultdict(list, {image: items for image, items in unit_by_image.items() if items})

    support = len(unit_by_image)
    min_support = max(4, int(math.ceil(0.20 * image_count)))
    if support < min_support:
        return

    strategy_reason = f"strategia cartella: valori scala con unita' cm/mm {support}/{image_count}; numeri senza unita' subordinati"
    selected_by_image: Dict[str, Dict[str, object]] = {}
    for image, items in unit_by_image.items():
        selected_by_image[image] = max(items, key=_scale_unit_strategy_key)

    lane_centers = [_row_center(row)[0] for row in selected_by_image.values()]
    lane_min_x = min(lane_centers) if lane_centers else 0.0
    lane_max_x = max(lane_centers) if lane_centers else 0.0

    fallback_by_image: Dict[str, Dict[str, object]] = {}
    for image, image_rows in by_image.items():
        numeric_fallbacks = [
            row
            for row in image_rows
            if _is_scale_unit_numeric_fallback_candidate(row, lane_min_x, lane_max_x)
            and _row_is_scale_endpoint(row, image_rows)
        ]
        if numeric_fallbacks:
            fallback_by_image[image] = max(numeric_fallbacks, key=_scale_unit_numeric_fallback_key)

    unit_cluster_review_by_image: Dict[str, Dict[str, object]] = {}
    for image, image_rows in by_image.items():
        if image in selected_by_image or image in fallback_by_image:
            continue
        cluster_fallbacks = [
            row
            for row in image_rows
            if _is_scale_unit_cluster_review_candidate(row) and _row_is_scale_endpoint(row, image_rows)
        ]
        if cluster_fallbacks:
            unit_cluster_review_by_image[image] = max(cluster_fallbacks, key=_scale_unit_cluster_review_key)

    for image, image_rows in by_image.items():
        selected = selected_by_image.get(image)
        fallback = fallback_by_image.get(image)
        unit_cluster_review = unit_cluster_review_by_image.get(image)
        selected_depth_for_override = _scale_unit_strategy_depth_mm(selected) if selected else 0.0
        fallback_depth_for_override = _scale_unit_numeric_fallback_depth_mm(fallback) if fallback else 0.0
        numeric_overrides_selected = bool(
            selected
            and fallback
            and fallback_depth_for_override > selected_depth_for_override + max(0.5, 0.01 * selected_depth_for_override)
        )
        for row in image_rows:
            if row.get("autonomous_mode") == "scale" and not _row_has_isolated_scale_number(row):
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scartato: cluster OCR senza token numerico e box affidabile"
                continue
            if (
                row.get("autonomous_mode") == "scale"
                and _row_is_in_visual_scale_lane(row)
                and not _row_is_scale_endpoint(row, image_rows)
            ):
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scartato: valore interno alla scala, il massimo e' solo a un'estremita'"
                continue
            if row is selected:
                if numeric_overrides_selected:
                    row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.55):.5f}"
                    row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; subordinato: massimo scala numerico nella stessa corsia, unità non letta"
                    continue
                selected_depth = _scale_unit_strategy_depth_mm(row)
                row["autonomous_mode"] = "scale"
                row["autonomous_valid"] = "1"
                row["folder_strategy"] = "scale_unit_required"
                row["depth_mm"] = f"{selected_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{selected_depth:.3f}"
                if not row.get("ocr_snap_mode"):
                    row["ocr_snap_mode"] = "scale_unit_required_unit_text"
                row["autonomous_score"] = f"{max(_f(row.get('autonomous_score')), 0.84):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; {strategy_reason}"
                continue
            selected_depth = _scale_unit_strategy_depth_mm(selected) if selected else 0.0
            row_depth = _scale_unit_strategy_depth_mm(row)
            if selected and _is_scale_unit_strategy_candidate(row) and abs(row_depth - selected_depth) <= max(0.5, 0.01 * selected_depth):
                row["autonomous_mode"] = "scale"
                row["autonomous_valid"] = "1"
                row["folder_strategy"] = "scale_unit_required"
                row["depth_mm"] = f"{selected_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{selected_depth:.3f}"
                if not row.get("ocr_snap_mode"):
                    row["ocr_snap_mode"] = "scale_unit_required_unit_variant"
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.55):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; variante subordinata al candidato con unita' scelto"
                continue
            if row is fallback:
                fallback_depth = _scale_unit_numeric_fallback_depth_mm(row)
                row["autonomous_mode"] = "scale"
                row["autonomous_valid"] = "1"
                row["folder_strategy"] = "scale_unit_required"
                row["depth_mm"] = f"{fallback_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{fallback_depth:.3f}"
                row["ocr_snap_mode"] = "scale_unit_required_numeric_review"
                row["autonomous_score"] = f"{min(max(_f(row.get('autonomous_score')), 0.56), 0.60):.5f}"
                reason = "review: unità attesa ma non letta sul massimo scala" if numeric_overrides_selected else "review: unità non letta, usato miglior numero nella corsia scala"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; {reason}; {strategy_reason}"
                continue
            if row is unit_cluster_review:
                review_depth = _scale_unit_cluster_review_depth_mm(row)
                row["autonomous_mode"] = "scale"
                row["autonomous_valid"] = "1"
                row["folder_strategy"] = "scale_unit_required"
                row["depth_mm"] = f"{review_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{review_depth:.3f}"
                row["ocr_snap_mode"] = "scale_unit_required_unit_cluster_review"
                row["autonomous_score"] = f"{min(max(_f(row.get('autonomous_score')), 0.56), 0.60):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; review: unita' OCR trovata ma non agganciata a token numerico singolo; {strategy_reason}"
                continue
            if row.get("autonomous_mode") != "scale" or not _row_has_explicit_depth_unit_text(row):
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.22):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scartato: cartella richiede valore scala con unita'"


def _scale_endpoint_candidate(row: Dict[str, object], profile: DepthProfile) -> bool:
    if not profile.scale_fallback:
        return False
    if _scale_endpoint_corrected_depth_mm(row) <= 0.5:
        return False
    if str(row.get("candidate_source") or "") != "token":
        return False
    if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
        return False
    if _flag(row, "ocr_bad_suffix_after_number") and not _row_has_dirty_scale_cm_text(row):
        return False
    if not _row_has_scale_endpoint_number_text(row):
        return False
    image = str(row.get("source_image") or "")
    width, height = _image_size(image) if image else (1, 1)
    if width <= 1 or height <= 1:
        return False
    cx, cy = _row_center(row)
    x_ratio = cx / float(width)
    y_ratio = cy / float(height)
    if not (0.55 <= x_ratio <= 0.86):
        return False
    if not (y_ratio <= 0.24 or y_ratio >= 0.67):
        return False
    return True


def _scale_endpoint_key(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    variant = str(row.get("box_variant") or "")
    base_bonus = 0.10 if variant == "base" else 0.0
    unit_bonus = 0.04 if _row_has_unit_value_text(row) or _row_has_dirty_scale_cm_text(row) else 0.0
    dirty_penalty = 0.02 if _flag(row, "ocr_bad_suffix_after_number") else 0.0
    width = _f(row.get("pred_width"))
    compact_bonus = max(0.0, min(0.04, (170.0 - width) / 2500.0)) if width > 0 else 0.0
    return (_f(row.get("ranker_score")) + base_bonus + unit_bonus + compact_bonus - dirty_penalty, _scale_endpoint_corrected_depth_mm(row), -width, -_f(row.get("candidate_rank")))


def _scale_endpoint_review_fallback_candidate(row: Dict[str, object], lane_cx: float) -> bool:
    if str(row.get("candidate_source") or "") == "token":
        return False
    if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
        return False
    text = str(row.get("ocr_text") or "").lower()
    if not (_unit_value_match(text) or _row_has_dirty_scale_cm_text(row) or re.search(r"\d+(?:\.\d+)?\s*(?:cm|mm|c|m|cem)", text)):
        return False
    image = str(row.get("source_image") or "")
    width, height = _image_size(image) if image else (1, 1)
    if width <= 1 or height <= 1:
        return False
    cx, cy = _row_center(row)
    if abs(cx - lane_cx) > 150.0:
        return False
    y_ratio = cy / float(height)
    return y_ratio <= 0.24 or y_ratio >= 0.67


def _apply_folder_scale_endpoint_rules(rows: List[Dict[str, object]], profile: DepthProfile) -> None:
    images = sorted({str(row.get("source_image") or "") for row in rows if row.get("source_image")})
    image_count = len(images)
    if image_count < 8 or not profile.scale_fallback:
        return
    if "bk" not in str(profile.vendor or "").lower():
        return

    endpoint_rows = [row for row in rows if _scale_endpoint_candidate(row, profile)]
    if not endpoint_rows:
        return

    groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    support_by_bin: Dict[int, set[str]] = defaultdict(set)
    for row in endpoint_rows:
        cx, _cy = _row_center(row)
        key = int(round(cx / 52.0))
        groups[key].append(row)
        support_by_bin[key].add(str(row.get("source_image") or ""))
    best_bin = max(support_by_bin, key=lambda key: (len(support_by_bin[key]), len(groups[key])))
    initial_lane_cx = _median([_row_center(row)[0] for row in groups.get(best_bin, [])])
    lane_rows = [row for row in endpoint_rows if abs(_row_center(row)[0] - initial_lane_cx) <= 130.0]
    support = len({str(row.get("source_image") or "") for row in lane_rows})
    min_support = max(8, int(math.ceil(0.45 * image_count)))
    if support < min_support:
        return

    selected_by_image: Dict[str, Dict[str, object]] = {}
    for row in lane_rows:
        image = str(row.get("source_image") or "")
        old = selected_by_image.get(image)
        if old is None or _scale_endpoint_key(row) > _scale_endpoint_key(old):
            selected_by_image[image] = row
    if len(selected_by_image) < min_support:
        return

    strategy_reason = (
        "strategia cartella: endpoint scala stabile "
        f"{support}/{image_count}; 0.0 non selezionabile, scelto valore non-zero lato opposto"
    )
    lane_cx_values = [_row_center(row)[0] for row in lane_rows]
    lane_cx = _median(lane_cx_values)
    fallback_by_image: Dict[str, Dict[str, object]] = {}
    for row in rows:
        image = str(row.get("source_image") or "")
        if image in selected_by_image:
            continue
        if not _scale_endpoint_review_fallback_candidate(row, lane_cx):
            continue
        old = fallback_by_image.get(image)
        if old is None or _f(row.get("ranker_score")) > _f(old.get("ranker_score")):
            fallback_by_image[image] = row

    for row in rows:
        image = str(row.get("source_image") or "")
        selected = selected_by_image.get(image)
        fallback = fallback_by_image.get(image)
        if not selected and not fallback:
            continue
        if row is fallback:
            row["autonomous_mode"] = "scale"
            row["autonomous_valid"] = "1"
            row["folder_strategy"] = "scale_endpoint_lane"
            row["depth_mm"] = ""
            row["ocr_snapped_depth_mm"] = ""
            row["ocr_snap_mode"] = "scale_endpoint_lane_review"
            row["autonomous_score"] = f"{max(_f(row.get('autonomous_score')), 0.50):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + f"; review: OCR non separato ma {strategy_reason}"
            continue
        if not selected:
            cx, _cy = _row_center(row)
            if abs(cx - lane_cx) > 180.0 or row.get("autonomous_mode") != "scale":
                row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.18):.5f}"
                row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scartato: cartella usa corsia scala stabile"
            continue
        if row is selected:
            corrected_depth = _scale_endpoint_corrected_depth_mm(row)
            row["autonomous_mode"] = "scale"
            row["autonomous_valid"] = "1"
            row["folder_strategy"] = "scale_endpoint_lane"
            if corrected_depth > 0:
                row["depth_mm"] = f"{corrected_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{corrected_depth:.3f}"
                row["ocr_snap_mode"] = "scale_endpoint_lane"
            row["autonomous_score"] = f"{max(_f(row.get('autonomous_score')), 0.82):.5f}"
            reason = str(row.get("autonomous_reason") or "")
            if _row_has_dirty_scale_cm_text(row):
                reason += "; OCR scala: suffisso m interpretato come cm troncato"
            row["autonomous_reason"] = reason + f"; {strategy_reason}"
            continue

        selected_depth = _scale_endpoint_corrected_depth_mm(selected)
        same_depth = abs(_scale_endpoint_corrected_depth_mm(row) - selected_depth) <= max(0.5, 0.01 * selected_depth)
        if _scale_endpoint_candidate(row, profile) and same_depth:
            row["autonomous_mode"] = "scale"
            row["autonomous_valid"] = "1"
            row["folder_strategy"] = "scale_endpoint_lane"
            if selected_depth > 0:
                row["depth_mm"] = f"{selected_depth:.3f}"
                row["ocr_snapped_depth_mm"] = f"{selected_depth:.3f}"
                row["ocr_snap_mode"] = "scale_endpoint_lane_variant"
            row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.55):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; variante box subordinata all'endpoint scala scelto"
            continue

        cx, _cy = _row_center(row)
        far_from_lane = abs(cx - lane_cx) > 180.0
        if far_from_lane or row.get("autonomous_mode") != "scale":
            row["autonomous_score"] = f"{min(_f(row.get('autonomous_score')), 0.18):.5f}"
            row["autonomous_reason"] = str(row.get("autonomous_reason") or "") + "; scartato: fuori corsia scala stabile di cartella"


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _add_stable_scale_position_reviews(rows: List[Dict[str, object]], profile: DepthProfile) -> None:
    if not profile.stable_scale_position_review:
        return
    if not any(row.get("folder_strategy") == "scale_from_variable_scale" for row in rows):
        return
    preferred_side = _preferred_scale_side(profile)
    reference_rows = [
        row
        for row in rows
        if row.get("autonomous_mode") == "scale"
        and _f(row.get("autonomous_valid")) > 0
        and _f(row.get("depth_mm")) > 0
        and _scale_side(row) == preferred_side
        and _row_has_unit_value_text(row)
        and _f(row.get("autonomous_score")) >= 0.55
    ]
    if len(reference_rows) < 4:
        return
    stable_box = (
        _median([_f(row.get("pred_left")) for row in reference_rows]),
        _median([_f(row.get("pred_top")) for row in reference_rows]),
        _median([_f(row.get("pred_right")) for row in reference_rows]),
        _median([_f(row.get("pred_bottom")) for row in reference_rows]),
    )
    if stable_box[2] <= stable_box[0] or stable_box[3] <= stable_box[1]:
        return

    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)
    for image, image_rows in by_image.items():
        best_scale = max(
            [
                _f(row.get("autonomous_score"))
                for row in image_rows
                if row.get("autonomous_mode") == "scale"
                and _f(row.get("autonomous_valid")) > 0
                and _f(row.get("depth_mm")) > 0
                and _scale_side(row) == preferred_side
            ],
            default=0.0,
        )
        if best_scale >= 0.35:
            continue
        template = image_rows[0] if image_rows else reference_rows[0]
        fallback = dict(template)
        fallback.update(
            {
                "sample_key": Path(image).name,
                "source_image": image,
                "depth_mm": "",
                "candidate_rank": "999",
                "candidate_source": "scale_stable_position_review",
                "box_variant": "stable_scale_position",
                "box_variant_expanded": "0",
                "box_variant_embedded": "0",
                "cluster_score": "0",
                "token_count": "0",
                "image_support": len(reference_rows),
                "unique_values": "",
                "cm_ratio": "1.00000",
                "mm_ratio": "0.00000",
                "depth_hint_ratio": "0.00000",
                "d_hint_ratio": "0.00000",
                "p_hint_ratio": "0.00000",
                "r_hint_ratio": "0.00000",
                "scale_hint_ratio": "1.00000",
                "fps_ips_ratio": "0.00000",
                "side_score": "1.00000",
                "accessory_score": "1.00000",
                "echo_center_penalty": "0.00000",
                "ocr_conf": "",
                "ocr_text": "OCR non letto nella posizione scala stabile",
                "ocr_text_len": "0",
                "ocr_digit_group_count": "0",
                "ocr_has_fps_ips": "0",
                "ocr_has_forbidden_marker": "0",
                "ocr_has_probe_model": "0",
                "ocr_has_time_like_text": "0",
                "ocr_bad_suffix_after_number": "0",
                "ocr_has_letter_hint": "0",
                "ocr_text_has_d": "0",
                "ocr_text_has_p": "0",
                "ocr_text_has_r": "0",
                "ocr_text_has_depth": "0",
                "ocr_d_left_of_number": "0",
                "ocr_d_right_of_number": "0",
                "ocr_p_left_of_number": "0",
                "ocr_p_right_of_number": "0",
                "ocr_r_left_of_number": "0",
                "ocr_r_right_of_number": "0",
                "ocr_has_cm_text": "0",
                "ocr_has_mm_text": "0",
                "ocr_numeric_value": "",
                "ocr_snapped_depth_mm": "",
                "ocr_snap_mode": "stable_scale_position_review",
                "ocr_multi_number": "0",
                "ocr_single_depth_expr": "0",
                "wide_text_box": "0",
                "snap_error_mm": "",
                "pred_width": f"{stable_box[2] - stable_box[0]:.2f}",
                "pred_height": f"{stable_box[3] - stable_box[1]:.2f}",
                "pred_left": f"{stable_box[0]:.2f}",
                "pred_top": f"{stable_box[1]:.2f}",
                "pred_right": f"{stable_box[2]:.2f}",
                "pred_bottom": f"{stable_box[3]:.2f}",
                "cluster_reason": f"stable_scale_position_review; support={len(reference_rows)}",
                "autonomous_mode": "scale",
                "ranker_score": "0.00000",
                "autonomous_rule_delta": "0.00000",
                "autonomous_score": "0.50000",
                "autonomous_valid": "1",
                "autonomous_reason": f"review scala: OCR non letto ma la cartella usa la scala; box sulla posizione stabile dell'indicatore lato {preferred_side}",
                "folder_strategy": "scale_from_variable_scale",
            }
        )
        rows.append(fallback)


def _promote_stable_unit_interface_rows(rows: List[Dict[str, object]]) -> None:
    images = {str(row.get("source_image") or "") for row in rows if row.get("source_image")}
    image_count = max(1, len(images))
    if image_count < 2:
        return
    candidates = [
        row
        for row in rows
        if row.get("autonomous_mode") == "scale"
        and not str(row.get("candidate_source") or "").startswith("scale_")
        and not _looks_like_scale_edge_value(row)
        and not _row_is_in_visual_scale_lane(row)
        and _clean_numeric_expression(str(row.get("ocr_text") or ""), require_unit=True)
        and _f(row.get("depth_mm")) > 0
    ]
    groups: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for row in candidates:
        cx = 0.5 * (_f(row.get("pred_left")) + _f(row.get("pred_right")))
        cy = 0.5 * (_f(row.get("pred_top")) + _f(row.get("pred_bottom")))
        groups[(int(round(cx / 52.0)), int(round(cy / 52.0)))].add(str(row.get("source_image") or ""))
    min_support = max(2, int(math.ceil(0.30 * image_count)))
    for row in candidates:
        cx = 0.5 * (_f(row.get("pred_left")) + _f(row.get("pred_right")))
        cy = 0.5 * (_f(row.get("pred_top")) + _f(row.get("pred_bottom")))
        support = len(groups.get((int(round(cx / 52.0)), int(round(cy / 52.0))), set()))
        if support >= min_support:
            row["autonomous_mode"] = "direct_label"
            row["interface_unit_promoted"] = "1"
            row["cluster_reason"] = str(row.get("cluster_reason") or "") + f"; stable_unit_interface={support}/{image_count}"


def _score_rows(rows: List[Dict[str, object]], profile: DepthProfile, ranker_model: Optional[object], features: Sequence[str]) -> None:
    _promote_stable_unit_interface_rows(rows)
    if rows and ranker_model is not None:
        try:
            scores = ranker_model.predict_proba(_feature_matrix(rows, features))[:, 1]
        except Exception:
            scores = [0.0 for _ in rows]
    else:
        scores = [0.0 for _ in rows]
    for row, model_score in zip(rows, scores):
        valid = 1
        invalid_reasons: List[str] = []
        if _f(row.get("depth_mm")) <= 0:
            valid = 0
            if _row_has_zero_depth_value_text(row):
                invalid_reasons.append("0.0 non e' una depth valida")
            else:
                invalid_reasons.append("nessun valore numerico depth")
        if row.get("autonomous_mode") == "direct_label" and not _row_has_direct_evidence(row) and not _flag(row, "interface_unit_promoted"):
            valid = 0
            invalid_reasons.append("label direct non completa")
        if _flag(row, "ocr_has_fps_ips") or _flag(row, "ocr_has_forbidden_marker") or _flag(row, "ocr_has_db_suffix") or _flag(row, "ocr_has_probe_model") or _flag(row, "ocr_has_time_like_text"):
            valid = 0
            invalid_reasons.append("suffisso dB non valido" if _flag(row, "ocr_has_db_suffix") else "marker OCR non depth")
        if _has_non_depth_direct_marker(str(row.get("ocr_text") or "")):
            valid = 0
            invalid_reasons.append("marker 2D/FR non valido per depth")
        if _flag(row, "ocr_bad_suffix_after_number"):
            valid = 0
            invalid_reasons.append("suffisso non valido")
        rule_delta, reasons = _rule_score(row, profile)
        base = float(model_score) if math.isfinite(float(model_score)) else 0.0
        # Keep the learned ranker useful, but make OCR validation strong enough
        # to reject plausible-looking non-depth numbers.
        final = max(0.0, min(1.0, (0.72 * base) + rule_delta))
        if valid <= 0:
            final = 0.0
        row["ranker_score"] = f"{base:.5f}"
        row["autonomous_rule_delta"] = f"{rule_delta:.5f}"
        row["autonomous_score"] = f"{final:.5f}"
        row["autonomous_valid"] = str(valid)
        row["autonomous_reason"] = "; ".join(reasons + invalid_reasons)
    _apply_scale_max_rule(rows, profile)
    _enforce_scale_endpoint_rule(rows)
    _enforce_scale_max_value_rule(rows)
    _apply_folder_consistency_rules(rows)
    _apply_folder_strategy_rules(rows)
    _apply_folder_scale_unit_strategy(rows)
    _apply_folder_direct_unit_strategy(rows)
    _apply_folder_direct_d_strategy(rows)
    for row in rows:
        score = _f(row.get("autonomous_score"))
        if _f(row.get("autonomous_valid")) <= 0 or score < 0.35:
            status = "reject"
        elif score >= 0.62:
            status = "accepted"
        else:
            status = "review"
        row["autonomous_status"] = status


def build_rows(
    *,
    folder: Path,
    images: Sequence[Path],
    expected_depths: Sequence[float],
    rect_echo: Optional[Box],
    profile: DepthProfile,
    max_candidates_per_sample: int,
    ocr_timeout: float,
    ocr_max_side: int,
    min_ocr_conf: float,
    cluster_bin_px: float,
    roi_passes: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    tokens = collect_depth_tokens(
        images=images,
        timeout=ocr_timeout,
        max_side=ocr_max_side,
        min_conf=min_ocr_conf,
        expected_depths_mm=expected_depths,
        rect_echo=rect_echo,
        roi_passes=roi_passes,
    )
    clusters = cluster_tokens(
        tokens=tokens,
        bin_px=cluster_bin_px,
        image_count=len(images),
        expected_depths_mm=expected_depths,
        rect_echo=rect_echo,
    )
    rows: List[Dict[str, object]] = []
    tokens_by_image: Dict[str, List[DepthToken]] = defaultdict(list)
    for token in tokens:
        tokens_by_image[token.word.image_path.as_posix()].append(token)
    scale_column_rows = 0
    for image_path in images:
        image_text = image_path.as_posix()
        for rank, cluster in enumerate(clusters[:max_candidates_per_sample], start=1):
            token, base_box = _box_from_token_or_cluster(cluster, image_text, profile)
            for variant_index, (box_variant, pred_box) in enumerate(_candidate_variants(base_box, image_text, token)):
                rows.append(
                    _candidate_row(
                        folder=folder,
                        image_path=image_text,
                        rank=rank,
                        variant_index=variant_index,
                        cluster=cluster,
                        token=token,
                        box_variant=box_variant,
                        pred_box=pred_box,
                        profile=profile,
                    )
                )
        scale_column_rows += _add_scale_column_rows(
            rows,
            folder=folder,
            image_path=image_path,
            image_tokens=tokens_by_image.get(image_text, []),
            rect_echo=rect_echo,
            expected_depths=expected_depths,
            profile=profile,
            rank_base=max_candidates_per_sample + 1,
        )
    summary = {
        "token_count": len(tokens),
        "cluster_count": len(clusters),
        "candidate_rows": len(rows),
        "scale_column_rows": scale_column_rows,
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("source_image") or "")].append(row)
    out: List[Dict[str, object]] = []
    for image, items in sorted(by_image.items()):
        ranked = sorted(items, key=lambda row: _f(row.get("autonomous_score")), reverse=True)
        top = ranked[0] if ranked else {}
        best_direct = next((row for row in ranked if row.get("autonomous_mode") == "direct_label"), None)
        best_scale = next((row for row in ranked if row.get("autonomous_mode") == "scale"), None)
        out.append(
            {
                "image_path": image,
                "status": top.get("autonomous_status", "missing"),
                "score": top.get("autonomous_score", ""),
                "ranker_score": top.get("ranker_score", ""),
                "mode": top.get("autonomous_mode", ""),
                "depth_mm": top.get("depth_mm", ""),
                "left": top.get("pred_left", ""),
                "top": top.get("pred_top", ""),
                "right": top.get("pred_right", ""),
                "bottom": top.get("pred_bottom", ""),
                "ocr_text": top.get("ocr_text", ""),
                "reason": top.get("autonomous_reason", ""),
                "best_direct_score": best_direct.get("autonomous_score", "") if best_direct else "",
                "best_direct_text": best_direct.get("ocr_text", "") if best_direct else "",
                "best_scale_score": best_scale.get("autonomous_score", "") if best_scale else "",
                "best_scale_value_mm": best_scale.get("depth_mm", "") if best_scale else "",
                "best_scale_text": best_scale.get("ocr_text", "") if best_scale else "",
                "candidates": len(items),
            }
        )
    return out


def predict(args: argparse.Namespace) -> Dict[str, object]:
    folder = args.folder.expanduser().resolve()
    image_dir = folder / "image_samples" if (folder / "image_samples").is_dir() else folder
    images = iter_images(image_dir)
    if args.max_images and len(images) > args.max_images:
        images = images[: int(args.max_images)]
    if not images:
        raise RuntimeError(f"Nessuna immagine trovata in {image_dir}")

    fss_path = args.fss.expanduser().resolve() if args.fss else None
    expected = _parse_float_list(args.expected_depths) if args.expected_depths else _expected_depths(fss_path)
    rect_echo = _parse_box_arg(args.rect_echo) if args.rect_echo else None
    if rect_echo is None and fss_path and fss_path.exists():
        rect_echo = parse_rect_echo(fss_path)
    context = resolve_context(folder, args.vendor or "", args.probe or "", args.context_json)
    vendor_template = str(args.vendor_template or "").strip()
    if vendor_template:
        context["vendor_template"] = vendor_template
    profile_vendor = vendor_template or str(context.get("vendor") or "")
    profile = profile_for_context(profile_vendor, str(context.get("probe") or ""))
    scale_side = str(args.scale_side_preference or "").strip().lower()
    scale_side_aliases = {"destra": "right", "dx": "right", "right": "right", "sinistra": "left", "sx": "left", "left": "left", "auto": "auto"}
    if scale_side:
        profile = replace(profile, scale_side_preference=scale_side_aliases.get(scale_side, profile.scale_side_preference))
    ranker_model, features, ranker_path = _load_ranker(args.ranker_model.expanduser().resolve() if args.ranker_model else DEFAULT_RANKER_MODEL)

    rows, gen_summary = build_rows(
        folder=folder,
        images=images,
        expected_depths=expected,
        rect_echo=rect_echo,
        profile=profile,
        max_candidates_per_sample=int(args.max_candidates_per_sample),
        ocr_timeout=float(args.ocr_timeout),
        ocr_max_side=int(args.ocr_max_side),
        min_ocr_conf=float(args.min_ocr_conf),
        cluster_bin_px=float(args.cluster_bin_px),
        roi_passes=bool(args.ocr_roi_passes),
    )
    _score_rows(rows, profile, ranker_model, features)
    predictions = _prediction_rows(rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = output_dir / "rect_depth_autonomous_candidates.csv"
    prediction_csv = output_dir / "rect_depth_autonomous_predictions.csv"
    _write_csv(candidate_csv, rows)
    _write_csv(prediction_csv, predictions)

    status_counts = Counter(str(row.get("status") or "missing") for row in predictions)
    mode_counts = Counter(str(row.get("mode") or "") for row in predictions)
    strategy_counts = Counter(str(row.get("folder_strategy") or "") for row in rows if row.get("folder_strategy"))
    summary = {
        "folder": folder.as_posix(),
        "image_count": len(images),
        "context": context,
        "profile": {
            "vendor": profile.vendor,
            "probe": profile.probe,
            "prefer_direct": profile.prefer_direct,
            "scale_fallback": profile.scale_fallback,
            "scale_max_required": profile.scale_max_required,
            "scale_cm_bias": profile.scale_cm_bias,
            "scale_mm_bias": profile.scale_mm_bias,
            "direct_letter_bias": profile.direct_letter_bias,
            "scale_side_preference": profile.scale_side_preference,
            "stable_scale_position_review": profile.stable_scale_position_review,
            "notes": list(profile.notes),
        },
        "ranker_model": ranker_path,
        "expected_depths_mm": list(expected),
        "rect_echo": list(rect_echo) if rect_echo else None,
        "generation": gen_summary,
        "prediction_status": dict(status_counts),
        "prediction_modes": dict(mode_counts),
        "folder_strategy": dict(strategy_counts),
        "candidate_csv": candidate_csv.as_posix(),
        "prediction_csv": prediction_csv.as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous RECT_DEPTH OCR/classical/ranker predictor.")
    parser.add_argument("--folder", required=True, type=Path, help="Config folder or image_samples folder.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fss", type=Path, default=None, help="Optional .fss for RECT_ECHO and expected-depth snapping.")
    parser.add_argument("--context-json", type=Path, default=None, help="Optional pipeline JSON containing vendor/probe predictions.")
    parser.add_argument("--vendor", default="", help="Optional vendor override from pipeline.")
    parser.add_argument("--probe", default="", help="Optional probe/probe-id override from pipeline.")
    parser.add_argument("--vendor-template", default="", help="Optional vendor profile/template to apply when it differs from free-text vendor.")
    parser.add_argument("--scale-side-preference", default="", choices=["", "auto", "right", "left", "destra", "sinistra", "dx", "sx"], help="Preferred scale side. Default comes from vendor profile, usually right.")
    parser.add_argument("--rect-echo", default="", help="Optional manual ultrasound rectangle as left,top,right,bottom.")
    parser.add_argument("--expected-depths", default="", help="Optional comma/space separated expected depth values in mm.")
    parser.add_argument("--ranker-model", type=Path, default=DEFAULT_RANKER_MODEL)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--max-candidates-per-sample", type=int, default=24)
    parser.add_argument("--ocr-timeout", type=float, default=8.0)
    parser.add_argument("--ocr-max-side", type=int, default=1800)
    parser.add_argument("--min-ocr-conf", type=float, default=18.0)
    parser.add_argument("--ocr-roi-passes", action="store_true", default=True)
    parser.add_argument("--no-ocr-roi-passes", action="store_false", dest="ocr_roi_passes")
    parser.add_argument("--cluster-bin-px", type=float, default=36.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict(args)


if __name__ == "__main__":
    main()
