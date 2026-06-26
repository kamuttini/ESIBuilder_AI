#!/usr/bin/env python3
"""Build an interactive review HTML for RECT_DEPTH ranker/OCR results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib


def _f(text: object, default: float = 0.0) -> float:
    try:
        out = float(str(text).strip())
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _i(text: object, default: int = 0) -> int:
    try:
        return int(float(str(text).strip()))
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _extract_review_data(review_html: Path) -> Dict[str, Dict[str, object]]:
    html = review_html.read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise RuntimeError(f"Cannot find embedded data JSON in {review_html}")
    rows = json.loads(match.group(1))
    return {f"{r['split']}:{r['sample_id']}:{r['global_idx']}": r for r in rows}


def _vendor(config_folder: str) -> str:
    low = config_folder.lower()
    for vendor in [
        "esaote",
        "bk",
        "hitachi",
        "ge",
        "mindray",
        "philips",
        "biopsee",
        "koelis",
        "canon",
        "alpinion",
        "siemens",
        "terason",
        "vinno",
        "exactvu",
        "sonostar",
    ]:
        if vendor in low:
            return vendor.title()
    return config_folder.split("_")[0].split(" ")[0] or "Unknown"


def _feature_matrix(rows: Sequence[Dict[str, str]], features: Sequence[str]) -> List[List[float]]:
    return [[_f(row.get(feature, "")) for feature in features] for row in rows]


def _is_hitachi_candidate(row: Dict[str, object]) -> bool:
    return "hitachi" in str(row.get("config_folder", "")).lower()


def _is_bk_candidate(row: Dict[str, object]) -> bool:
    config = str(row.get("config_folder", "")).lower()
    return bool(re.search(r"(^|[^a-z])bk([^a-z]|$)|bk3000|bk5000|flexfocus|profocus|specto", config))


def _append_reason(row: Dict[str, object], reason: str) -> None:
    old = str(row.get("postprocess_reason") or "")
    row["postprocess_reason"] = f"{old}; {reason}" if old else reason


def _truthy(row: Dict[str, object], key: str) -> bool:
    return _f(row.get(key)) >= 1.0


def _postprocess_score(row: Dict[str, object]) -> float:
    score = _f(row.get("ranker_score"))
    if _f(row.get("manual_prior_candidate")) >= 1.0:
        score = max(score, 0.999)
        _append_reason(row, "prior manuale")
    elif _f(row.get("manual_hint_present")) > 0.0:
        match = _f(row.get("manual_hint_match_score"))
        mismatch = _f(row.get("manual_hint_mismatch_score"))
        score += (0.02 * match) - (0.03 * mismatch)
        if match or mismatch:
            _append_reason(row, f"hint {match:.2f}/{mismatch:.2f}")

    if _f(row.get("manual_prior_candidate")) < 1.0:
        if _truthy(row, "ocr_has_fps_ips") or _f(row.get("fps_ips_ratio")) > 0.0:
            score -= 0.35
            _append_reason(row, "fps/ips non depth")
        if _truthy(row, "ocr_has_forbidden_marker"):
            score -= 0.22
            _append_reason(row, "marker non depth")
        if _truthy(row, "ocr_has_probe_model"):
            score -= 0.28
            _append_reason(row, "modello sonda")
        if _truthy(row, "ocr_has_time_like_text"):
            score -= 0.24
            _append_reason(row, "timestamp/ora")
        if _truthy(row, "ocr_bad_suffix_after_number"):
            score -= 0.10
            _append_reason(row, "suffisso non cm/mm")
        if _truthy(row, "ocr_d_right_of_number") and not _truthy(row, "ocr_d_left_of_number"):
            score -= 0.14
            _append_reason(row, "D dopo numero non depth")

        if _f(row.get("manual_hint_d")) > 0.0:
            if _truthy(row, "ocr_text_has_d") and _truthy(row, "ocr_d_left_of_number"):
                score += 0.07
                _append_reason(row, "testo D")
            elif _truthy(row, "ocr_text_has_d") and not _truthy(row, "ocr_d_left_of_number"):
                score -= 0.12
                _append_reason(row, "D non a sinistra")
            elif _f(row.get("manual_hint_present")) > 0.0:
                score -= 0.05
                _append_reason(row, "manca D nel crop")
        if _f(row.get("manual_hint_p")) > 0.0:
            if _truthy(row, "ocr_text_has_p"):
                score += 0.08
                _append_reason(row, "testo P")
            elif _f(row.get("p_hint_ratio")) <= 0.05:
                score -= 0.05
                _append_reason(row, "manca P nel crop")
        if _f(row.get("manual_hint_r")) > 0.0:
            if _truthy(row, "ocr_text_has_r"):
                score += 0.08
                _append_reason(row, "testo R")
            elif _f(row.get("r_hint_ratio")) <= 0.05:
                score -= 0.05
                _append_reason(row, "manca R nel crop")
        if _f(row.get("manual_hint_depth")) > 0.0:
            if _truthy(row, "ocr_text_has_depth"):
                score += 0.08
                _append_reason(row, "testo Depth")
            else:
                score -= 0.04
                _append_reason(row, "manca Depth nel crop")
        if _f(row.get("manual_hint_cm")) > 0.0 and _truthy(row, "ocr_has_cm_text"):
            score += 0.07
            _append_reason(row, "testo cm")
        if _f(row.get("manual_hint_mm")) > 0.0 and _truthy(row, "ocr_has_mm_text"):
            score += 0.06
            _append_reason(row, "testo mm")
        if _f(row.get("manual_hint_scale")) > 0.0 and _f(row.get("scale_hint_ratio")) > 0.0:
            score += 0.035 * _f(row.get("scale_hint_ratio"))
            _append_reason(row, "scala coerente")

    if _is_bk_candidate(row) and _f(row.get("manual_prior_candidate")) < 1.0:
        scale_ratio = _f(row.get("scale_hint_ratio"))
        cm_ratio = _f(row.get("cm_ratio"))
        mm_ratio = _f(row.get("mm_ratio"))
        hint_scale = _f(row.get("manual_hint_scale"))
        hint_cm = _f(row.get("manual_hint_cm"))
        hint_mm = _f(row.get("manual_hint_mm"))
        if hint_scale > 0.0:
            if scale_ratio > 0.0:
                score += 0.12 * scale_ratio
                _append_reason(row, "scala BK")
            else:
                score -= 0.045
                _append_reason(row, "manca scala BK")
            if hint_cm > 0.0 and cm_ratio > 0.0:
                score += 0.08 * cm_ratio
                _append_reason(row, "cm scala BK")
            if hint_mm > 0.0 and mm_ratio > 0.0:
                score += 0.07 * mm_ratio
                _append_reason(row, "mm scala BK")
            if (_truthy(row, "ocr_text_has_d") or _truthy(row, "ocr_text_has_p") or _truthy(row, "ocr_text_has_r")) and scale_ratio <= 0.05 and max(cm_ratio, mm_ratio) <= 0.05:
                score -= 0.06
                _append_reason(row, "lettera senza scala BK")
        elif hint_cm > 0.0 and cm_ratio > 0.0:
            score += 0.06 * cm_ratio
            _append_reason(row, "cm BK")

    if _is_hitachi_candidate(row) and _f(row.get("manual_prior_candidate")) < 1.0:
        width = _f(row.get("pred_width"))
        height = _f(row.get("pred_height"))
        digit_groups = _f(row.get("ocr_digit_group_count"))
        variant = str(row.get("box_variant") or "")
        hint_r = _f(row.get("manual_hint_r"))
        hint_p = _f(row.get("manual_hint_p"))
        hint_scale = _f(row.get("manual_hint_scale"))
        r_ratio = _f(row.get("r_hint_ratio"))
        p_ratio = _f(row.get("p_hint_ratio"))
        scale_ratio = _f(row.get("scale_hint_ratio"))
        text_has_r = _f(row.get("ocr_text_has_r"))
        text_has_p = _f(row.get("ocr_text_has_p"))
        text_matches_manual_letter = (hint_r > 0.0 and text_has_r > 0.0) or (hint_p > 0.0 and text_has_p > 0.0)

        if variant.startswith("embedded_value"):
            score += 0.035
            _append_reason(row, "crop embedded")
        if width > 220.0 and height > 0 and width / max(1.0, height) > 4.0:
            score -= 0.035 if text_matches_manual_letter else 0.06
            _append_reason(row, "box largo")
        if digit_groups >= 4 and hint_scale < 1.0:
            score -= 0.04 if text_matches_manual_letter else 0.12
            _append_reason(row, "troppi numeri")
        elif digit_groups == 0 and str(row.get("ocr_text") or "").strip():
            score -= 0.08
            _append_reason(row, "testo senza numero")
        if hint_r > 0.0:
            if text_has_r > 0.0:
                score += 0.10
                _append_reason(row, "testo R")
            else:
                score += 0.025 * r_ratio
                score -= 0.08
                _append_reason(row, "manca R nel crop")
            if text_has_r <= 0.0 and r_ratio <= 0.05 and scale_ratio > 0.20:
                score -= 0.06
                _append_reason(row, "R preferito a scala")
        if hint_p > 0.0:
            if text_has_p > 0.0:
                score += 0.10
                _append_reason(row, "testo P")
            else:
                score += 0.025 * p_ratio
                score -= 0.08
                _append_reason(row, "manca P nel crop")
            if text_has_p <= 0.0 and p_ratio <= 0.05 and scale_ratio > 0.20:
                score -= 0.06
                _append_reason(row, "P preferito a scala")
    return max(0.0, min(1.0, score))


def _rank_root(row: Dict[str, object]) -> int:
    try:
        return int(float(str(row.get("candidate_rank") or "999999")))
    except Exception:
        return 999999


def _postprocess_group(rows: Sequence[Dict[str, object]]) -> None:
    if not rows or not any(_is_hitachi_candidate(row) for row in rows):
        return

    by_root: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_root[_rank_root(row)].append(row)

    for siblings in by_root.values():
        base_rows = [row for row in siblings if str(row.get("box_variant") or "") == "base"]
        embedded_rows = [row for row in siblings if str(row.get("box_variant") or "").startswith("embedded_value")]
        if not base_rows or not embedded_rows:
            continue
        best_embedded = max(embedded_rows, key=lambda row: _f(row.get("postprocess_score")))
        embedded_err = _f(best_embedded.get("value_error_mm"), 999999.0)
        embedded_digits = _f(best_embedded.get("ocr_digit_group_count"))
        embedded_ok = embedded_digits <= 2 and _f(best_embedded.get("ocr_has_fps_ips")) < 1.0 and _f(best_embedded.get("fps_ips_ratio")) <= 0.0
        for base in base_rows:
            width = _f(base.get("pred_width"))
            height = _f(base.get("pred_height"))
            if width <= 220.0 or height <= 0 or width / max(1.0, height) <= 4.0:
                continue
            base_err = _f(base.get("value_error_mm"), 999999.0)
            if embedded_ok and (embedded_err <= min(25.0, base_err + 6.0) or _f(best_embedded.get("manual_hint_match_score")) > 0.0):
                best_embedded["postprocess_score"] = max(
                    _f(best_embedded.get("postprocess_score")),
                    min(1.0, _f(base.get("postprocess_score")) + 0.045),
                )
                base["postprocess_score"] = max(0.0, _f(base.get("postprocess_score")) - 0.05)
                _append_reason(best_embedded, "preferito a base larga")
                _append_reason(base, "demote base larga")



def _box_pct(row: Dict[str, object], prefix: str, orig_w: float, orig_h: float) -> Optional[Dict[str, float]]:
    x1 = _f(row.get(f"{prefix}_left"))
    y1 = _f(row.get(f"{prefix}_top"))
    x2 = _f(row.get(f"{prefix}_right"))
    y2 = _f(row.get(f"{prefix}_bottom"))
    if orig_w <= 0 or orig_h <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return {
        "left": x1 / orig_w * 100.0,
        "top": y1 / orig_h * 100.0,
        "width": (x2 - x1) / orig_w * 100.0,
        "height": (y2 - y1) / orig_h * 100.0,
    }


def _gt_pct(row: Dict[str, str], orig_w: float, orig_h: float) -> Optional[Dict[str, float]]:
    x1 = _f(row.get("left"))
    y1 = _f(row.get("top"))
    x2 = _f(row.get("right"))
    y2 = _f(row.get("bottom"))
    if orig_w <= 0 or orig_h <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return {
        "left": x1 / orig_w * 100.0,
        "top": y1 / orig_h * 100.0,
        "width": (x2 - x1) / orig_w * 100.0,
        "height": (y2 - y1) / orig_h * 100.0,
    }


def _comment_hints(text: str) -> List[str]:
    low = f" {text.lower()} "
    hints: List[str] = []
    checks = [
        ("R", r"\br\b"),
        ("D", r"\bd\b|depth"),
        ("P", r"\bp\b"),
        ("cm", r"\bcm\b"),
        ("mm", r"\bmm\b"),
        ("scala", r"scala|scale"),
    ]
    for label, pattern in checks:
        if re.search(pattern, low):
            hints.append(label)
    return hints


def _suggestion(
    *,
    status: str,
    vendor: str,
    top: Optional[Dict[str, object]],
    oracle: Optional[Dict[str, object]],
    near: Optional[Dict[str, object]],
    review_comment: str,
) -> str:
    parts: List[str] = []
    hints = _comment_hints(review_comment)
    vendor_low = vendor.lower()
    if status == "ok":
        parts.append("ok: il ranker sceglie un candidato vicino alla GT")
        if top:
            parts.append(f"score={_f(top.get('ranker_score')):.2f}, err={_f(top.get('center_error_px')):.0f}px")
        parts.append("controllare solo se il box include troppo contesto o taglia unita/lettera")
    elif status == "ranker_miss":
        parts.append("ranker_miss: il candidato giusto esiste ma non e il primo")
        if oracle:
            parts.append(f"oracle rank={oracle.get('candidate_rank')}, err={_f(oracle.get('center_error_px')):.0f}px")
        parts.append("aumentare il peso di vicinanza a GT/side-hint e penalizzare candidati centrali")
    elif status == "no_candidate":
        parts.append("no_candidate: OCR/classica non genera nessun candidato")
        parts.append("serve una passata ROI dedicata su fasce laterali/superiori/inferiori")
    elif status == "near_miss":
        parts.append("near_miss: OCR trova un candidato vicino alla GT ma box/centro non sono ancora corretti")
        if top:
            parts.append(f"top err={_f(top.get('center_error_px')):.0f}px")
        if near:
            parts.append(f"near err={_f(near.get('center_error_px')):.0f}px")
        parts.append("utile per postprocess: espandere/ricentrare box usando contesto R/P/D/scala")
    else:
        parts.append("candidate_miss: esistono candidati ma nessuno aggancia la GT")
        if top:
            parts.append(f"top err={_f(top.get('center_error_px')):.0f}px")
        parts.append("probabile problema di crop OCR, soglia testo o box candidate troppo stretto")

    if vendor_low in {"hitachi", "biopsee"}:
        parts.append("priorita: cercare valore vicino a R/P/D e scala laterale, anche senza cm/mm")
    elif vendor_low == "ge":
        parts.append("priorita GE: molti candidati sono vicini, valutare espansione box e tolleranza centro")
    elif vendor_low in {"bk", "esaote"}:
        parts.append("priorita: mantenere cm/mm ma non renderli obbligatori")

    if hints:
        parts.append("commento GT cita " + "/".join(hints))
    return "; ".join(parts)


def _clip_text(text: object, limit: int = 96) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _yes(row: Optional[Dict[str, object]], key: str) -> bool:
    return bool(row) and _f(row.get(key)) >= 1.0


def _ocr_factor_lines(candidate: Optional[Dict[str, object]]) -> List[str]:
    if not candidate:
        return ["nessun candidato OCR/classico disponibile"]
    out: List[str] = []
    text = _clip_text(candidate.get("ocr_text"), 120)
    if text:
        out.append(f"OCR top: '{text}'")
    else:
        out.append("OCR top vuoto o non leggibile")
    if _yes(candidate, "ocr_text_has_depth"):
        out.append("contiene Depth")
    letters = [label for label, key in [("D", "ocr_text_has_d"), ("P", "ocr_text_has_p"), ("R", "ocr_text_has_r")] if _yes(candidate, key)]
    if letters:
        out.append("lettera hint nel crop: " + "/".join(letters))
    units = [label for label, key in [("cm", "ocr_has_cm_text"), ("mm", "ocr_has_mm_text")] if _yes(candidate, key)]
    if units:
        out.append("unita letta: " + "/".join(units))
    digit_groups = _f(candidate.get("ocr_digit_group_count"))
    if digit_groups:
        out.append(f"gruppi numerici OCR: {digit_groups:.0f}")
    value_error = _f(candidate.get("value_error_mm"), float("nan"))
    if math.isfinite(value_error) and value_error:
        out.append(f"errore valore vs depth attesa: {value_error:.1f} mm")
    elif math.isfinite(value_error):
        out.append("valore coerente con la depth attesa")
    if _f(candidate.get("manual_hint_match_score")) > 0.0:
        out.append(f"match con hint review: {_f(candidate.get('manual_hint_match_score')):.2f}")
    return out


def _ocr_warning_lines(candidate: Optional[Dict[str, object]]) -> List[str]:
    if not candidate:
        return []
    checks = [
        ("fps/ips", "ocr_has_fps_ips"),
        ("Hz/MI/TIS/C2/% non depth", "ocr_has_forbidden_marker"),
        ("modello sonda", "ocr_has_probe_model"),
        ("timestamp/ora", "ocr_has_time_like_text"),
        ("suffisso diverso da cm/mm", "ocr_bad_suffix_after_number"),
        ("troppi numeri nel crop", "ocr_multi_number"),
    ]
    out = [label for label, key in checks if _yes(candidate, key)]
    if _f(candidate.get("manual_hint_mismatch_score")) > 0.0:
        out.append(f"mismatch hint review: {_f(candidate.get('manual_hint_mismatch_score')):.2f}")
    return out


def _non_ocr_factor_lines(candidate: Optional[Dict[str, object]]) -> List[str]:
    if not candidate:
        return ["nessun box candidato generato"]
    out = [
        f"score postprocess {_f(candidate.get('postprocess_score')):.3f} da ranker {_f(candidate.get('ranker_score')):.3f}",
        f"rank {candidate.get('candidate_rank')} · source {candidate.get('candidate_source')} · variante {candidate.get('box_variant') or 'base'}",
        f"IoU GT {_f(candidate.get('iou')):.2f} · centro {_f(candidate.get('center_error_px')):.0f}px",
        f"box {float(_f(candidate.get('pred_width'))):.0f}x{float(_f(candidate.get('pred_height'))):.0f}px",
    ]
    if _yes(candidate, "manual_prior_candidate"):
        out.append("prior manuale importato dalle review")
    if _yes(candidate, "box_variant_embedded"):
        out.append("crop ricentrato sul valore embedded")
    reason = _clip_text(candidate.get("postprocess_reason"), 140)
    if reason:
        out.append("regole applicate: " + reason)
    return out


def _discarded_factor_lines(
    candidates: Sequence[Dict[str, object]],
    top: Optional[Dict[str, object]],
) -> List[str]:
    if not candidates:
        return []
    counts: Counter = Counter()
    high_raw_counts: Counter = Counter()
    top_score = _f(top.get("postprocess_score")) if top else -1.0
    top_raw = _f(top.get("ranker_score")) if top else -1.0
    checks = [
        ("fps/ips", "ocr_has_fps_ips"),
        ("Hz/MI/TIS/C2/%", "ocr_has_forbidden_marker"),
        ("modello sonda", "ocr_has_probe_model"),
        ("timestamp/ora", "ocr_has_time_like_text"),
        ("suffisso non cm/mm", "ocr_bad_suffix_after_number"),
        ("troppi numeri", "ocr_multi_number"),
        ("box largo", "wide_text_box"),
    ]
    for candidate in candidates:
        if top is not None and candidate is top:
            continue
        active = [label for label, key in checks if _yes(candidate, key)]
        for label in active:
            counts[label] += 1
        if active and _f(candidate.get("ranker_score")) >= max(0.05, top_raw - 0.03) and _f(candidate.get("postprocess_score")) <= top_score + 0.02:
            for label in active:
                high_raw_counts[label] += 1
    out: List[str] = []
    if counts:
        out.append("candidati penalizzati: " + ", ".join(f"{label} x{count}" for label, count in counts.most_common(5)))
    if high_raw_counts:
        out.append("penalita decisive vicino al top: " + ", ".join(f"{label} x{count}" for label, count in high_raw_counts.most_common(4)))
    return out


def _decision_factors(
    *,
    status: str,
    top: Optional[Dict[str, object]],
    oracle: Optional[Dict[str, object]],
    near: Optional[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
) -> Dict[str, List[str]]:
    decision: List[str] = []
    if not top:
        decision.append("nessun candidato generato: il limite e nella fase OCR/candidate generation")
    elif status == "ok":
        decision.append("scelto il top: supera soglia GT per IoU o distanza centro")
    elif status == "ranker_miss":
        decision.append("il candidato corretto esiste, ma il postprocess ha ordinato prima un altro box")
        if oracle:
            decision.append(f"oracle: rank {oracle.get('candidate_rank')} con centro {_f(oracle.get('center_error_px')):.0f}px")
    elif status == "near_miss":
        decision.append("il candidato piu vicino e nella zona giusta ma il box non aggancia ancora la GT")
        if near:
            decision.append(f"near: centro {_f(near.get('center_error_px')):.0f}px")
    else:
        decision.append("i candidati generati non intercettano la GT: probabile crop OCR o ROI insufficiente")
    warnings = _ocr_warning_lines(top)
    return {
        "decision": decision,
        "ocr": _ocr_factor_lines(top),
        "ocr_warnings": warnings,
        "non_ocr": _non_ocr_factor_lines(top),
        "discarded": _discarded_factor_lines(candidates, top),
    }


def _html_shell(data_json: str, summary_json: str) -> str:
    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RECT_DEPTH OCR/ranker review</title>
<style>
:root{{--bg:#090b0d;--panel:#14181c;--line:#2a333c;--text:#edf3f7;--muted:#91a0ad;--gt:#45b8ff;--corr:#eaff5c;--top:#ff6570;--ok:#25d366;--oracle:#b078ff;--near:#ff9f43;--warn:#f2cc60}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:30;background:rgba(9,11,13,.96);border-bottom:1px solid var(--line);padding:14px 18px}}h1{{margin:0 0 8px;font-size:21px}}
.summary,.controls,.actions{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 9px}}.metric b{{font-size:18px}}
.vendor-stats{{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}}.vendor-chip{{background:#10151a;border:1px solid #2b3540;border-radius:7px;padding:6px 8px;color:#dbe7ee;font-size:12px}}.vendor-chip b{{font-size:13px}}
input,select,textarea,button{{background:#0d1115;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:inherit}}button{{cursor:pointer}}.controls{{margin-top:10px}}.controls input{{min-width:280px;flex:1}}
.legend{{color:var(--muted)}}.sw{{display:inline-block;width:12px;height:12px;border:2px solid;margin:0 4px -2px 10px}}.sw.gt{{border-color:var(--gt)}}.sw.top{{border-color:var(--top)}}.sw.ok{{border-color:var(--ok)}}.sw.oracle{{border-color:var(--oracle)}}.sw.near{{border-color:var(--near)}}
main{{padding:18px 20px 44px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}.card.ok{{box-shadow:0 0 0 2px rgba(37,211,102,.28) inset}}.card.ranker_miss{{box-shadow:0 0 0 2px rgba(242,204,96,.32) inset}}.card.near_miss{{box-shadow:0 0 0 2px rgba(255,159,67,.32) inset}}.card.candidate_miss,.card.no_candidate{{box-shadow:0 0 0 2px rgba(255,101,112,.25) inset}}
.thumb{{width:100%;height:285px;background:#050607;border:0;border-bottom:1px solid var(--line);padding:0;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:zoom-in}}.stage,.lb-stage{{position:relative;display:inline-block;line-height:0;max-width:100%;max-height:100%}}.stage img{{display:block;max-width:100%;max-height:285px;width:auto;height:auto}}.lb-stage img{{display:block;max-width:100%;max-height:90vh;width:auto;height:auto}}
.box{{position:absolute;border:3px solid;pointer-events:none;min-width:7px;min-height:7px}}.box.gt{{border-color:var(--gt)}}.box.corrected{{border-color:var(--corr);border-style:dashed}}.box.top{{border-color:var(--top)}}.box.top.okbox{{border-color:var(--ok)}}.box.oracle{{border-color:var(--oracle);border-style:dotted}}.box.near{{border-color:var(--near);border-style:dashed}}body.hide-gt .box.gt,body.hide-top .box.top,body.hide-oracle .box.oracle{{display:none}}.lb-stage.draw{{cursor:crosshair;touch-action:none;user-select:none}}
.body{{padding:10px;display:grid;gap:8px}}.title{{display:flex;gap:6px;align-items:center;min-width:0}}.title b{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.pill{{font-size:12px;color:#071018;background:#d9c36c;border-radius:999px;padding:2px 7px}}.pill.ok{{background:var(--ok)}}.pill.ranker_miss{{background:var(--warn)}}.pill.near_miss{{background:var(--near)}}.pill.candidate_miss,.pill.no_candidate{{background:var(--top);color:white}}
.meta{{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}}.meta div,.candidate{{background:#0d1115;border:1px solid #202830;border-radius:6px;padding:6px;min-width:0}}.meta span,.candidate span,label span{{display:block;color:var(--muted);font-size:11px}}.meta b{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.candidate-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}.candidate b{{font-size:12px}}.path{{font-size:12px;color:var(--muted);word-break:break-word}}.empty{{display:none;color:var(--muted);text-align:center;margin:40px}}
.suggestion{{background:#0f1a20;border:1px solid #263743;border-left:3px solid #5fb5ff;border-radius:6px;padding:7px 8px;color:#d6e7f0;font-size:12px}}
.decision{{display:grid;gap:6px;background:#10161a;border:1px solid #25313a;border-radius:6px;padding:7px 8px;color:#dce8ef;font-size:12px}}.decision div{{min-width:0}}.decision span{{display:block;color:#91a0ad;font-size:11px;margin-bottom:2px}}.decision ul{{margin:0;padding-left:16px}}.decision li{{margin:1px 0}}.decision .warn span{{color:#f2cc60}}
.review-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.gt-tools{{display:flex;gap:8px;flex-wrap:wrap}}.draw-on{{border-color:var(--corr);box-shadow:0 0 0 1px var(--corr) inset}}.hint-badge{{color:var(--corr);font-size:12px}}
.lightbox{{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.94);display:none;gap:16px;padding:22px}}.lightbox.open{{display:flex}}.lb-image{{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}.lb-panel{{flex:0 0 430px;background:#10151a;border:1px solid #303a43;border-radius:8px;padding:14px;display:grid;gap:10px;overflow:auto}}.lb-panel h2{{margin:0;font-size:17px}}.sub{{color:var(--muted);font-size:12px;word-break:break-word}}.lb-buttons{{display:flex;flex-wrap:wrap;gap:8px}}.close,.nav{{position:fixed;background:#111820;color:white;border:1px solid #48525c}}.close{{top:14px;right:16px}}.nav{{top:50%;transform:translateY(-50%);font-size:28px;width:46px;height:62px}}.prev{{left:16px}}.next{{right:460px}}.saved{{min-height:18px;color:var(--ok);font-size:12px}}
@media(max-width:900px){{.lightbox{{flex-direction:column;overflow:auto}}.lb-panel{{flex:0 0 auto}}.next{{right:16px}}.grid{{grid-template-columns:1fr}}.controls input{{min-width:160px}}}}
</style>
</head>
<body>
<header>
  <h1>RECT_DEPTH OCR/ranker review</h1>
  <div id="summary" class="summary"></div>
  <div class="controls">
    <input id="q" autocomplete="off" placeholder="Cerca vendor, config, sample, commento">
    <select id="status" autocomplete="off"><option value="all">tutti gli stati</option><option value="candidate_miss">candidate_miss</option><option value="near_miss">near_miss</option><option value="ranker_miss">ranker_miss</option><option value="ok">ok</option><option value="no_candidate">no_candidate</option></select>
    <select id="vendor" autocomplete="off"><option value="all">tutti i vendor</option></select>
    <select id="sort" autocomplete="off"><option value="status">stato/errore</option><option value="ranker_score">score ranker</option><option value="center_error">errore centro</option><option value="vendor">vendor/config</option><option value="dataset">ordine dataset</option></select>
  </div>
  <div class="actions">
    <button id="toggleGt">GT on/off</button><button id="toggleTop">Top on/off</button><button id="toggleOracle">Oracle on/off</button>
    <button id="exportJson">Esporta commenti JSON</button>
    <span class="legend"><span id="visibleCount"></span> visibili <span class="sw gt"></span>GT <span class="sw" style="border-color:var(--corr)"></span>GT corretta <span class="sw top"></span>top ranker <span class="sw ok"></span>top corretto <span class="sw oracle"></span>oracle <span class="sw near"></span>near</span>
  </div>
  <div id="vendorStats" class="vendor-stats"></div>
</header>
<main><div id="grid" class="grid"></div><div id="empty" class="empty">Nessun risultato con questi filtri.</div></main>
<div id="lightbox" class="lightbox"><button id="close" class="close">Chiudi</button><button id="prev" class="nav prev">‹</button><button id="next" class="nav next">›</button><div class="lb-image"><span id="lbStage" class="lb-stage"><img id="lbImg" alt=""><span class="box gt"></span><span class="box corrected"></span><span class="box top"></span><span class="box oracle"></span><span class="box near"></span></span></div><aside class="lb-panel"><h2 id="lbTitle"></h2><div id="lbSub" class="sub"></div><div id="lbCandidates"></div><label><span>Analisi suggerita</span><div id="lbSuggestion" class="suggestion"></div></label><div class="review-grid"><label><span>Hint da seguire</span><select id="lbHint"></select></label><label><span>GT corretta</span><div class="gt-tools"><button id="lbDrawGt">Disegna GT</button><button id="lbUseTopGt">GT = top</button><button id="lbResetGt">Ripristina</button><button id="lbClearGt">Cancella</button></div></label></div><label><span>Commento review modello/OCR</span><textarea id="lbComment" placeholder="Es: OCR manca R, scala bassa, candidato vicino ma box troppo stretto..."></textarea></label><div class="lb-buttons"><button id="lbUseSuggestion">Usa suggerimento</button><button data-note="ocr_manca_valore">OCR manca valore</button><button data-note="box_troppo_stretto">Box stretto</button><button data-note="scala_laterale">Scala laterale</button><button data-note="hitachi_r">Hitachi/R</button><button data-note="gt_da_ricontrollare">GT da ricontrollare</button></div><div id="lbPath" class="sub"></div><div id="lbSaved" class="saved"></div></aside></div>
<script id="data" type="application/json">{data_json}</script>
<script id="summaryData" type="application/json">{summary_json}</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const SUMMARY=JSON.parse(document.getElementById('summaryData').textContent);
const storageKey='rectDepthRankerReview:v1';
const $=id=>document.getElementById(id);
const HINT_OPTIONS=['','D','P','R','Depth','D e mm','D e cm','P e mm','P e cm','Depth e mm','valore massimo scala','valore massimo scala con cm','cm','mm'];
let notes=loadNotes();let current=-1;let drawGt=false;let drawStart=null;
function loadNotes(){{try{{return JSON.parse(localStorage.getItem(storageKey)||'{{}}')||{{}}}}catch(e){{return {{}}}}}}
function saveNotes(){{localStorage.setItem(storageKey,JSON.stringify(notes));updateCount();}}
function esc(s){{return String(s??'').replace(/[&<>"]/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[ch]));}}
function styleBox(b){{return b?`left:${{b.left}}%;top:${{b.top}}%;width:${{b.width}}%;height:${{b.height}}%;`:''}}
function boxSet(el,b){{if(!b){{el.style.display='none';return}}el.style.display='block';el.style.left=b.left+'%';el.style.top=b.top+'%';el.style.width=b.width+'%';el.style.height=b.height+'%';}}
function noteFor(r){{return notes[r.sample_key]||{{}}}}
function saveFor(r,patch){{notes[r.sample_key]={{...noteFor(r),...patch,updated_at:new Date().toISOString(),status:r.status,vendor:r.vendor,config_folder:r.config_folder,source_image:r.source_image}};saveNotes();}}
function correctedBox(r){{return noteFor(r).corrected_gt_box_pct||null}}
function pctPoint(e){{const rect=$('lbStage').getBoundingClientRect();return {{x:Math.max(0,Math.min(100,(e.clientX-rect.left)/rect.width*100)),y:Math.max(0,Math.min(100,(e.clientY-rect.top)/rect.height*100))}}}}
function normPctBox(a,b){{const left=Math.min(a.x,b.x),top=Math.min(a.y,b.y),right=Math.max(a.x,b.x),bottom=Math.max(a.y,b.y);if(right-left<0.5||bottom-top<0.5)return null;return {{left,top,width:right-left,height:bottom-top}}}}
function pctToPx(box,r){{if(!box||!r.orig_w||!r.orig_h)return null;return {{left:Math.round(box.left*r.orig_w/100),top:Math.round(box.top*r.orig_h/100),right:Math.round((box.left+box.width)*r.orig_w/100),bottom:Math.round((box.top+box.height)*r.orig_h/100)}}}}
function pctRect(b){{return b?{{left:b.left,top:b.top,right:b.left+b.width,bottom:b.top+b.height}}:null}}
function pctIou(a,b){{a=pctRect(a);b=pctRect(b);if(!a||!b)return 0;const ix=Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left));const iy=Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));const inter=ix*iy;const aa=Math.max(0,a.right-a.left)*Math.max(0,a.bottom-a.top);const bb=Math.max(0,b.right-b.left)*Math.max(0,b.bottom-b.top);const u=aa+bb-inter;return u>0?inter/u:0}}
function pctCenterPx(a,b,r){{a=pctRect(a);b=pctRect(b);if(!a||!b)return 999999;const ax=(a.left+a.right)*0.5*r.orig_w/100,ay=(a.top+a.bottom)*0.5*r.orig_h/100,bx=(b.left+b.right)*0.5*r.orig_w/100,by=(b.top+b.bottom)*0.5*r.orig_h/100;return Math.hypot(ax-bx,ay-by)}}
function boxEval(box,gt,r){{const iou=pctIou(box,gt);const center=pctCenterPx(box,gt,r);return {{iou,center,ok:!!box&&!!gt&&(iou>=0.20||center<=35)}}}}
function effectiveEval(r){{const gt=correctedBox(r)||r.gt_box_pct;const changed=!!correctedBox(r);const top=boxEval(r.top_box_pct,gt,r),oracle=boxEval(r.oracle_box_pct,gt,r),near=boxEval(r.near_box_pct,gt,r);let status=r.status;if(changed){{if(!r.top)status='no_candidate';else if(top.ok)status='ok';else if(oracle.ok)status='ranker_miss';else if(near.ok)status='near_miss';else status='candidate_miss';}}return {{status,changed,top,oracle,near}}}}
function evalLine(r){{const e=effectiveEval(r);if(!e.changed)return '';return `GT corretta: ${{e.status}} · top iou ${{e.top.iou.toFixed(2)}} · centro ${{Math.round(e.top.center)}}px`}}
function markSaved(txt='salvato'){{$('lbSaved').textContent=txt;}}
function metric(label,value){{return `<div class="metric"><b>${{value}}</b> ${{label}}</div>`}}
function factorsText(r){{const f=r.decision_factors||{{}};return ['decision','ocr','ocr_warnings','discarded','non_ocr'].flatMap(k=>f[k]||[]).join(' ');}}
function factorLines(items){{return (items||[]).map(x=>`<li>${{esc(x)}}</li>`).join('')||'<li>nessun segnale</li>';}}
function decisionHtml(r){{const f=r.decision_factors||{{}};const warn=(f.ocr_warnings||[]).length?`<div class="warn"><span>Warning OCR</span><ul>${{factorLines(f.ocr_warnings)}}</ul></div>`:'';const discarded=(f.discarded||[]).length?`<div class="warn"><span>Candidati scartati</span><ul>${{factorLines(f.discarded)}}</ul></div>`:'';return `<div class="decision"><div><span>Decisione</span><ul>${{factorLines(f.decision)}}</ul></div><div><span>OCR determinante</span><ul>${{factorLines(f.ocr)}}</ul></div>${{warn}}${{discarded}}<div><span>Non-OCR</span><ul>${{factorLines(f.non_ocr)}}</ul></div></div>`;}}
function currentStatusCounts(){{const c={{ok:0,near_miss:0,ranker_miss:0,candidate_miss:0,no_candidate:0}};DATA.forEach(r=>{{const st=effectiveEval(r).status;c[st]=(c[st]||0)+1;}});return c}}
function buildSummary(){{const c=currentStatusCounts();$('summary').innerHTML=metric('campioni',SUMMARY.samples)+metric('ok',c.ok||0)+metric('near_miss',c.near_miss||0)+metric('ranker_miss',c.ranker_miss||0)+metric('candidate_miss',c.candidate_miss||0)+metric('no_candidate',c.no_candidate||0)+metric('vendor',SUMMARY.vendors);}}
function buildVendors(){{const vendors=[...new Set(DATA.map(r=>r.vendor))].sort();$('vendor').insertAdjacentHTML('beforeend',vendors.map(v=>`<option value="${{esc(v)}}">${{esc(v)}}</option>`).join(''));}}
function resetInitialFilters(){{$('q').value='';$('status').value='all';$('vendor').value='all';$('sort').value='status';}}
function buildVendorStats(){{const by={{}};DATA.forEach(r=>{{const st=effectiveEval(r).status;by[r.vendor]??={{total:0,ok:0,near_miss:0,ranker_miss:0,candidate_miss:0,no_candidate:0}};by[r.vendor].total++;by[r.vendor][st]++;}});$('vendorStats').innerHTML=Object.entries(by).sort((a,b)=>(b[1].candidate_miss+b[1].no_candidate)-(a[1].candidate_miss+a[1].no_candidate)).map(([v,c])=>`<button class="vendor-chip" data-vendor-filter="${{esc(v)}}"><b>${{esc(v)}}</b> ${{c.total}} · ok ${{c.ok}} · near ${{c.near_miss}} · miss ${{c.candidate_miss+c.no_candidate}} · rank ${{c.ranker_miss}}</button>`).join('');document.querySelectorAll('[data-vendor-filter]').forEach(b=>b.onclick=()=>{{$('vendor').value=b.dataset.vendorFilter;apply();}});}}
function cardHtml(r,i){{const saved=noteFor(r);const note=saved.comment||'';const hint=saved.hint||'';const learnedHint=r.ranker_review_hint||'';const learnedComment=r.ranker_review_comment||'';const corr=saved.corrected_gt_box_pct||null;const ev=effectiveEval(r);const status=ev.status;const topClass=status==='ok'?'okbox':'';const localLine=evalLine(r);return `<article class="card ${{status}}" data-index="${{i}}" data-key="${{esc(r.sample_key)}}" data-vendor="${{esc(r.vendor)}}" data-status="${{status}}" data-config="${{esc(r.config_folder)}}" data-search="${{esc((r.vendor+' '+r.config_folder+' '+r.sample_key+' '+r.review_comment+' '+learnedComment+' '+note+' '+hint+' '+learnedHint+' '+factorsText(r)).toLowerCase())}}">
  <button class="thumb" type="button" onclick="openCardByIndex(${{i}})" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();openCardByIndex(${{i}})}}"><span class="stage"><img src="${{esc(r.asset_rel)}}" alt="${{esc(r.sample_key)}}" loading="lazy" decoding="async"><span class="box gt" style="${{styleBox(r.gt_box_pct)}}"></span><span class="box corrected" style="${{styleBox(corr)}}"></span><span class="box top ${{topClass}}" style="${{styleBox(r.top_box_pct)}}"></span><span class="box oracle" style="${{styleBox(r.oracle_box_pct)}}"></span><span class="box near" style="${{styleBox(r.near_box_pct)}}"></span></span></button>
  <div class="body"><div class="title"><span class="pill ${{status}}">${{status}}${{ev.changed?' *':''}}</span><span class="pill">${{esc(r.vendor)}}</span><b title="${{esc(r.sample_key)}}">${{esc(r.sample_key)}}</b></div>
  <div class="meta"><div><span>config</span><b title="${{esc(r.config_folder)}}">${{esc(r.config_folder)}}</b></div><div><span>depth/flip</span><b>${{esc(r.depth_mm)}} mm · ${{esc(r.flip_state)}}</b></div><div><span>top postprocess</span><b>${{r.top?`score ${{(r.top.postprocess_score??r.top.ranker_score).toFixed(3)}} · raw ${{r.top.ranker_score.toFixed(3)}} · err ${{r.top.center_error_px}}px`:'nessuno'}}</b></div><div><span>oracle</span><b>${{r.oracle?`rank ${{r.oracle.candidate_rank}} · err ${{r.oracle.center_error_px}}px`:'nessuno'}}</b></div></div>
  <div class="candidate-grid">${{candidateMini(r.top,'top')}}${{candidateMini(r.oracle,'oracle')}}${{candidateMini(r.near,'near')}}</div>
  ${{localLine?`<div class="hint-badge">${{esc(localLine)}}</div>`:''}}
  ${{decisionHtml(r)}}
  <div class="suggestion">${{esc(r.suggestion)}}</div>
  ${{learnedHint||learnedComment?`<div class="hint-badge">imparato: ${{esc(learnedHint||'commento')}}${{learnedComment?` · ${{esc(learnedComment)}}`:''}}</div>`:''}}
  ${{hint?`<div class="hint-badge">hint: ${{esc(hint)}}</div>`:''}}
  <label><span>Nota</span><textarea data-note>${{esc(note)}}</textarea></label>
  <div class="path">${{esc(r.source_image)}}</div></div></article>`}}
function candidateMini(c,label){{if(!c)return `<div class="candidate"><span>${{label}}</span><b>nessuno</b></div>`;const manual=String(c.manual_prior_candidate||'0')==='1'?' · prior manuale':'';const learned=c.ranker_review_hint?` · hint ${{esc(c.ranker_review_hint)}}`:'';const match=Number(c.manual_hint_match_score||0);const mismatch=Number(c.manual_hint_mismatch_score||0);const matchText=(match||mismatch)?` · match ${{match.toFixed(2)}}/${{mismatch.toFixed(2)}}`:'';const score=Number(c.postprocess_score??c.ranker_score??0);const raw=Number(c.ranker_score??0);const warn=[];if(String(c.ocr_has_fps_ips||'0')==='1'||Number(c.fps_ips_ratio||0)>0)warn.push('fps/ips');if(String(c.ocr_has_forbidden_marker||'0')==='1')warn.push('marker');if(String(c.ocr_has_probe_model||'0')==='1')warn.push('sonda');if(String(c.ocr_has_time_like_text||'0')==='1')warn.push('ora');if(String(c.ocr_bad_suffix_after_number||'0')==='1')warn.push('suffisso');const ocr=c.ocr_text?`<span>OCR: ${{esc(c.ocr_text)}} · nums ${{c.ocr_digit_group_count||0}}${{warn.length?' · warn '+esc(warn.join('/')):''}}${{c.postprocess_reason?` · ${{esc(c.postprocess_reason)}}`:''}}</span>`:'';return `<div class="candidate"><span>${{label}}</span><b>rank ${{c.candidate_rank}} · ${{c.candidate_source}} · ${{c.box_variant||'base'}}${{manual}}${{learned}}</b><span>score ${{score.toFixed(3)}} · raw ${{raw.toFixed(3)}} · iou ${{c.iou}} · center ${{c.center_error_px}} · w/h ${{c.pred_width||''}}/${{c.pred_height||''}} · hint D/P/R ${{c.d_hint_ratio}}/${{c.p_hint_ratio}}/${{c.r_hint_ratio}}${{matchText}}</span>${{ocr}}</div>`}}
function render(){{buildSummary();buildVendorStats();$('grid').innerHTML=DATA.map(cardHtml).join('');bindCards();apply();}}
function bindCards(){{document.querySelectorAll('.thumb').forEach(btn=>{{btn.onclick=()=>openFromCard(btn.closest('.card'));btn.onkeydown=e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();openFromCard(btn.closest('.card'));}}}};}});document.querySelectorAll('textarea[data-note]').forEach(t=>{{t.oninput=()=>{{const key=t.closest('.card').dataset.key;notes[key]={{...(notes[key]||{{}}),comment:t.value,updated_at:new Date().toISOString()}};saveNotes();}}}});}}
$('grid').addEventListener('click',e=>{{const btn=e.target.closest('.thumb');if(!btn)return;openFromCard(btn.closest('.card'));}});
function visibleCards(){{return [...document.querySelectorAll('.card')].filter(c=>c.style.display!=='none')}}
function rankStatus(s){{return {{candidate_miss:0,no_candidate:1,near_miss:2,ranker_miss:3,ok:4}}[s]??5}}
function apply(){{const q=$('q').value.toLowerCase().trim(),st=$('status').value,v=$('vendor').value,sort=$('sort').value;const cards=[...document.querySelectorAll('.card')];cards.forEach(c=>{{let ok=true;if(st!=='all')ok=ok&&c.dataset.status===st;if(v!=='all')ok=ok&&c.dataset.vendor===v;if(q)ok=ok&&c.dataset.search.includes(q);c.style.display=ok?'':'none';}});cards.sort((a,b)=>{{const ra=DATA[+a.dataset.index],rb=DATA[+b.dataset.index];if(sort==='ranker_score')return (rb.top?.ranker_score||-1)-(ra.top?.ranker_score||-1);if(sort==='center_error')return (+(ra.top?.center_error_px||99999))-(+(rb.top?.center_error_px||99999));if(sort==='vendor')return a.dataset.vendor.localeCompare(b.dataset.vendor)||a.dataset.config.localeCompare(b.dataset.config);if(sort==='status')return rankStatus(a.dataset.status)-rankStatus(b.dataset.status)||(+b.dataset.index)-(+a.dataset.index);return (+a.dataset.index)-(+b.dataset.index);}}).forEach(c=>$('grid').appendChild(c));const n=cards.filter(c=>c.style.display!=='none').length;$('visibleCount').textContent=n;$('empty').style.display=n?'none':'block';}}
function rowForCurrent(){{const c=visibleCards()[current];return c?DATA[+c.dataset.index]:null}}
function openCardByIndex(i){{const card=document.querySelector(`.card[data-index="${{i}}"]`);if(card)openFromCard(card);}}
function openFromCard(card){{current=visibleCards().indexOf(card);openLightbox();}}
function syncCardReview(r){{const card=document.querySelector(`.card[data-key="${{CSS.escape(r.sample_key)}}"]`);if(!card)return;const saved=noteFor(r);const corr=card.querySelector('.box.corrected');if(corr)boxSet(corr,saved.corrected_gt_box_pct||null);const ta=card.querySelector('textarea[data-note]');if(ta)ta.value=saved.comment||'';}}
function refreshAfterGtChange(r,msg){{const key=r.sample_key;render();const card=document.querySelector(`.card[data-key="${{CSS.escape(key)}}"]`);if(card)current=visibleCards().indexOf(card);if($('lightbox').classList.contains('open'))openLightbox();markSaved(msg);}}
function setDrawGt(on){{drawGt=!!on;$('lbDrawGt').classList.toggle('draw-on',drawGt);$('lbDrawGt').textContent=drawGt?'Disegno attivo':'Disegna GT';$('lbStage').classList.toggle('draw',drawGt);}}
function openLightbox(){{const r=rowForCurrent();if(!r)return;const ev=effectiveEval(r);setDrawGt(false);$('lightbox').classList.add('open');$('lbImg').src=r.asset_rel;$('lbTitle').textContent=`${{ev.status}}${{ev.changed?' *':''}} · ${{r.vendor}} · ${{r.sample_key}}`;$('lbSub').textContent=`${{r.config_folder}} · ${{r.depth_mm}} mm · ${{r.flip_state}}`;$('lbPath').textContent=r.source_image;boxSet(document.querySelector('#lbStage .gt'),r.gt_box_pct);boxSet(document.querySelector('#lbStage .corrected'),correctedBox(r));boxSet(document.querySelector('#lbStage .top'),r.top_box_pct);document.querySelector('#lbStage .top').classList.toggle('okbox',ev.status==='ok');boxSet(document.querySelector('#lbStage .oracle'),r.oracle_box_pct);boxSet(document.querySelector('#lbStage .near'),r.near_box_pct);$('lbCandidates').innerHTML=`<div class="candidate-grid">${{candidateMini(r.top,'top ranker')}}${{candidateMini(r.oracle,'oracle')}}${{candidateMini(r.near,'near')}}</div>${{ev.changed?`<div class="hint-badge">${{esc(evalLine(r))}}</div>`:''}}${{decisionHtml(r)}}<div class="sub">Commento GT: ${{esc(r.review_comment||'')}}${{r.ranker_review_hint?` · hint imparato: ${{esc(r.ranker_review_hint)}}`:''}}${{r.ranker_review_comment?` · nota importata: ${{esc(r.ranker_review_comment)}}`:''}}</div>`;$('lbSuggestion').textContent=r.suggestion||'';$('lbComment').value=noteFor(r).comment||'';$('lbHint').value=noteFor(r).hint||'';$('lbSaved').textContent='';}}
function closeLightbox(){{$('lightbox').classList.remove('open')}}
function move(d){{const cards=visibleCards();if(!cards.length)return;current=(current+d+cards.length)%cards.length;openLightbox();}}
function isTyping(el){{return !!(el&&el.closest&&el.closest('input,textarea,select,[contenteditable="true"]'))}}
function updateCount(){{const n=Object.values(notes).filter(v=>v&&v.comment).length;document.title=`RECT_DEPTH ranker review (${{n}} note)`;}}
function download(name,type,content){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{{type}}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
function buildHints(){{$('lbHint').innerHTML=HINT_OPTIONS.map(v=>`<option value="${{esc(v)}}">${{v?esc(v):'nessun hint strutturato'}}</option>`).join('');}}
buildSummary();buildVendors();resetInitialFilters();buildVendorStats();buildHints();render();updateCount();setTimeout(()=>{{resetInitialFilters();apply();}},0);setTimeout(()=>{{resetInitialFilters();apply();}},250);
['q','status','vendor','sort'].forEach(id=>$(id).addEventListener('input',apply));
$('toggleGt').onclick=()=>document.body.classList.toggle('hide-gt');$('toggleTop').onclick=()=>document.body.classList.toggle('hide-top');$('toggleOracle').onclick=()=>document.body.classList.toggle('hide-oracle');
$('close').onclick=closeLightbox;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('lbComment').oninput=e=>{{const r=rowForCurrent();if(!r)return;saveFor(r,{{comment:e.target.value}});syncCardReview(r);markSaved();}};
$('lbHint').onchange=e=>{{const r=rowForCurrent();if(!r)return;saveFor(r,{{hint:e.target.value}});markSaved('hint salvato');}};
$('lbDrawGt').onclick=()=>setDrawGt(!drawGt);
$('lbUseTopGt').onclick=()=>{{const r=rowForCurrent();if(!r)return;if(!r.top_box_pct){{markSaved('top ranker assente');return}}saveFor(r,{{corrected_gt_box_pct:r.top_box_pct,corrected_gt_box_px:pctToPx(r.top_box_pct,r),correction_source:'top_ranker'}});refreshAfterGtChange(r,'GT corretta = top ranker; punteggi ricalcolati');}};
$('lbResetGt').onclick=()=>{{const r=rowForCurrent();if(!r)return;saveFor(r,{{corrected_gt_box_pct:r.gt_box_pct||null,corrected_gt_box_px:pctToPx(r.gt_box_pct,r),correction_source:'original_gt'}});refreshAfterGtChange(r,'GT corretta = GT originale; punteggi ricalcolati');}};
$('lbClearGt').onclick=()=>{{const r=rowForCurrent();if(!r)return;const saved={{...noteFor(r)}};delete saved.corrected_gt_box_pct;delete saved.corrected_gt_box_px;delete saved.correction_source;notes[r.sample_key]=saved;saveNotes();refreshAfterGtChange(r,'GT corretta cancellata; punteggi ricalcolati');}};
$('lbStage').addEventListener('pointerdown',e=>{{if(!drawGt)return;e.preventDefault();drawStart=pctPoint(e);$('lbStage').setPointerCapture?.(e.pointerId);boxSet(document.querySelector('#lbStage .corrected'),{{left:drawStart.x,top:drawStart.y,width:0.8,height:0.8}});}});
$('lbStage').addEventListener('pointermove',e=>{{if(!drawGt||!drawStart)return;e.preventDefault();const box=normPctBox(drawStart,pctPoint(e));if(box)boxSet(document.querySelector('#lbStage .corrected'),box);}});
$('lbStage').addEventListener('pointerup',e=>{{if(!drawGt||!drawStart)return;e.preventDefault();const r=rowForCurrent();const box=normPctBox(drawStart,pctPoint(e));drawStart=null;if(!r||!box){{markSaved('box troppo piccolo');return}}saveFor(r,{{corrected_gt_box_pct:box,corrected_gt_box_px:pctToPx(box,r),correction_source:'drawn_gt'}});refreshAfterGtChange(r,'GT corretta salvata; punteggi ricalcolati');}});
$('lbUseSuggestion').onclick=()=>{{const r=rowForCurrent();if(!r||!r.suggestion)return;$('lbComment').value=r.suggestion;$('lbComment').dispatchEvent(new Event('input',{{bubbles:true}}));}};
document.querySelectorAll('.lb-buttons [data-note]').forEach(b=>b.onclick=()=>{{const r=rowForCurrent();if(!r)return;const add=b.dataset.note;const cur=$('lbComment').value.trim();$('lbComment').value=cur?cur+'; '+add:add;$('lbComment').dispatchEvent(new Event('input',{{bubbles:true}}));}});
document.addEventListener('keydown',e=>{{if(!$('lightbox').classList.contains('open')||isTyping(e.target))return;if(e.key==='Escape')closeLightbox();if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);}});
$('exportJson').onclick=()=>download('rect_depth_ranker_review_notes.json','application/json',JSON.stringify(Object.entries(notes).map(([key,value])=>{{const r=DATA.find(row=>row.sample_key===key);const box=value.corrected_gt_box_pct;return {{key,...value,corrected_gt_box_px:value.corrected_gt_box_px||(r?pctToPx(box,r):null)}}}}),null,2));
</script>
</body>
</html>"""


def build(args: argparse.Namespace) -> Dict[str, object]:
    manifest_rows = _read_csv(args.review_manifest.expanduser().resolve())
    if args.config_contains:
        terms = [term.strip().lower() for term in str(args.config_contains).split(",") if term.strip()]
        manifest_rows = [
            row for row in manifest_rows
            if any(term in row.get("config_folder", "").lower() for term in terms)
        ]
    candidate_rows = _read_csv(args.candidates_csv.expanduser().resolve())
    asset_rows = _extract_review_data(args.review_html.expanduser().resolve())
    model_bundle = joblib.load(args.ranker_model.expanduser().resolve())
    model = model_bundle["model"]
    features = model_bundle["features"]
    if candidate_rows:
        scores = model.predict_proba(_feature_matrix(candidate_rows, features))[:, 1]
        for row, score in zip(candidate_rows, scores):
            row["ranker_score"] = float(score)
            row["postprocess_score"] = _postprocess_score(row)

    by_sample: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        by_sample[row["sample_key"]].append(row)
    for sample_rows in by_sample.values():
        _postprocess_group(sample_rows)

    records: List[Dict[str, object]] = []
    status_counts: Counter = Counter()
    vendors: set[str] = set()
    for order, row in enumerate(manifest_rows):
        key = row.get("review_key") or f"{row.get('setup_id')}_{row.get('depth_index0')}_{row.get('flip_state')}"
        if args.only_candidate_samples and key not in by_sample:
            continue
        asset = asset_rows.get(key)
        if not asset:
            continue
        orig_w = float(asset.get("orig_w") or 1)
        orig_h = float(asset.get("orig_h") or 1)
        candidates = by_sample.get(key, [])
        top = max(candidates, key=lambda c: float(c.get("postprocess_score", c.get("ranker_score", 0.0)))) if candidates else None
        positives = [c for c in candidates if int(float(c.get("label", "0") or 0)) == 1]
        oracle = min(positives, key=lambda c: float(c.get("center_error_px", "999999"))) if positives else None
        near = None
        if candidates:
            best_by_center = min(candidates, key=lambda c: float(c.get("center_error_px", "999999")))
            if float(best_by_center.get("center_error_px", "999999")) <= float(args.near_center_px):
                near = best_by_center
        if not candidates:
            status = "no_candidate"
        elif top and int(float(top.get("label", "0") or 0)) == 1:
            status = "ok"
        elif oracle:
            status = "ranker_miss"
        elif near:
            status = "near_miss"
        else:
            status = "candidate_miss"
        vendor = _vendor(row.get("config_folder", ""))
        vendors.add(vendor)
        status_counts[status] += 1
        top_box = _box_pct(top, "pred", orig_w, orig_h) if top else None
        oracle_box = _box_pct(oracle, "pred", orig_w, orig_h) if oracle else None
        near_box = _box_pct(near, "pred", orig_w, orig_h) if near else None
        review_context = " ".join(
            part
            for part in [
                row.get("review_comment", ""),
                row.get("ranker_review_comment", ""),
                row.get("ranker_review_hint", ""),
            ]
            if part
        )
        record = {
            "order": order,
            "sample_key": key,
            "vendor": vendor,
            "status": status,
            "config_folder": row.get("config_folder", ""),
            "source_image": row.get("source_image", ""),
            "depth_mm": row.get("depth_mm", ""),
            "flip_state": row.get("flip_state", ""),
            "orig_w": orig_w,
            "orig_h": orig_h,
            "review_comment": row.get("review_comment", ""),
            "ranker_review_hint": row.get("ranker_review_hint", ""),
            "ranker_review_comment": row.get("ranker_review_comment", ""),
            "ranker_review_propagated": row.get("ranker_review_propagated", ""),
            "ranker_review_box_prior": row.get("ranker_review_box_prior", ""),
            "asset_rel": "../bbox_smoke_cpu/" + str(asset.get("asset_rel", "")),
            "gt_box_pct": _gt_pct(row, orig_w, orig_h),
            "top_box_pct": top_box,
            "oracle_box_pct": oracle_box,
            "near_box_pct": near_box,
            "top": _candidate_payload(top) if top else None,
            "oracle": _candidate_payload(oracle) if oracle else None,
            "near": _candidate_payload(near) if near else None,
            "suggestion": _suggestion(
                status=status,
                vendor=vendor,
                top=top,
                oracle=oracle,
                near=near,
                review_comment=review_context,
            ),
            "decision_factors": _decision_factors(status=status, top=top, oracle=oracle, near=near, candidates=candidates),
        }
        records.append(record)

    summary = {
        "samples": len(records),
        "status_counts": dict(status_counts),
        "vendors": len(vendors),
        "input_manifest": args.review_manifest.as_posix(),
        "candidate_rows": args.candidates_csv.as_posix(),
        "ranker_model": args.ranker_model.as_posix(),
        "config_contains": str(args.config_contains or ""),
        "near_center_px": float(args.near_center_px),
        "selection_score": "postprocess_score",
        "postprocess_rules": "manual_prior_candidate hard boost; comment-derived hints; global forbidden-marker/probe/timestamp/suffix/D-after-number penalties; BK scale/cm postprocess; Hitachi wide/multi-number and embedded-crop postprocess",
    }
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        _html_shell(
            data_json=json.dumps(records, ensure_ascii=False, separators=(",", ":")),
            summary_json=json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    return {"html": output_html.as_posix(), **summary}


def _candidate_payload(row: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not row:
        return None
    keys = [
        "candidate_rank",
        "candidate_source",
        "box_variant",
        "box_variant_expanded",
        "label",
        "iou",
        "center_error_px",
        "cluster_score",
        "ranker_score",
        "postprocess_score",
        "cm_ratio",
        "mm_ratio",
        "d_hint_ratio",
        "p_hint_ratio",
        "r_hint_ratio",
        "scale_hint_ratio",
        "fps_ips_ratio",
        "value_error_mm",
        "pred_width",
        "pred_height",
        "ocr_text",
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
        "ocr_multi_number",
        "ocr_single_depth_expr",
        "wide_text_box",
        "postprocess_reason",
        "cluster_reason",
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
    ]
    out: Dict[str, object] = {}
    for key in keys:
        value = row.get(key, "")
        if key in {"ranker_score", "postprocess_score"}:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive ranker/OCR review HTML.")
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--ranker-model", required=True, type=Path)
    parser.add_argument("--review-html", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--config-contains", default="")
    parser.add_argument("--near-center-px", type=float, default=80.0)
    parser.add_argument("--only-candidate-samples", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    if args.summary_json:
        out = args.summary_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
