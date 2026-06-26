#!/usr/bin/env python3
"""Apply ranker review notes back to a RECT_DEPTH manifest.

The ranker review UI exports sparse corrections: corrected GT boxes, comments,
and structured hints.  For some ultrasound configurations the corrected box is
stable across depth images, so this utility can propagate the median corrected
box and the dominant hint to every row in the same configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Box = Tuple[float, float, float, float]


def _f(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _box_from_note(note: Dict[str, object]) -> Optional[Box]:
    box = note.get("corrected_gt_box_px")
    if isinstance(box, dict):
        vals = (_f(box.get("left")), _f(box.get("top")), _f(box.get("right")), _f(box.get("bottom")))
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        vals = tuple(_f(x) for x in box)  # type: ignore[assignment]
    else:
        return None
    if len(vals) != 4 or any(not math.isfinite(x) for x in vals):
        return None
    left, top, right, bottom = vals
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _box_text(box: Optional[Box]) -> str:
    if not box:
        return ""
    return "|".join(str(int(round(v))) for v in box)


def _median_box(boxes: Iterable[Box]) -> Optional[Box]:
    vals = list(boxes)
    if not vals:
        return None
    return tuple(float(statistics.median([b[i] for b in vals])) for i in range(4))  # type: ignore[return-value]


def _apply_box(row: Dict[str, str], box: Box) -> None:
    left, top, right, bottom = box
    row["left"] = str(int(round(left)))
    row["top"] = str(int(round(top)))
    row["right"] = str(int(round(right)))
    row["bottom"] = str(int(round(bottom)))
    row["width"] = str(max(1, int(round(right - left))))
    row["height"] = str(max(1, int(round(bottom - top))))


def _vendor_matches(config: str, vendors: set[str]) -> bool:
    low = config.lower()
    return any(v in low for v in vendors)


def _norm_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _hint_from_note(note: Optional[Dict[str, object]]) -> Tuple[str, str]:
    """Return an explicit or comment-derived review hint.

    Review comments often contain the useful rule even when the structured
    select was left empty: e.g. "fps non e un hint ... preferire R".  Keep
    explicit hints authoritative, then use conservative lexical rules.
    """
    if not note:
        return "", ""
    explicit = str(note.get("hint") or "").strip()
    if explicit:
        return explicit, "explicit"

    text = _norm_text(" ".join([str(note.get("comment") or ""), str(note.get("config_folder") or "")]))
    if not text:
        return "", ""

    negative_marker_comment = bool(
        re.search(r"\b(?:non\s+(?:e|è)\s+un\s+marker|non\s+pu[oò]\s+stare|non\s+pu[oò])\b", text)
        or re.search(r"\b(?:hz|mi|c2|etd|fps|ips)\s+non\b", text)
    )
    if negative_marker_comment:
        if "valore massimo scala" in text or "massimo valore scala" in text:
            return "valore massimo scala", "comment"
        if re.search(r"\bcm\b", text) and not re.search(r"\b(?:depth|d|p|r)\b", text):
            return "cm", "comment"
        if re.search(r"\bmm\b", text) and not re.search(r"\b(?:depth|d|p|r)\b", text):
            return "mm", "comment"
        return "", "negative_comment"

    prefer_r = (
        re.search(r"\bprefer\w*\s+(?:la\s+)?(?:lettera\s+)?r\b", text)
        or re.search(r"\br\s+(?:e|è)\s+da\s+preferire\b", text)
        or re.search(r"\bse\s+c.?e\s+r\b", text)
        or re.search(r"\btop\s+ranker\s+con\s+hint\s+r\b", text)
        or re.search(r"\bhint\s+r\b", text)
        or ("fps" in text and re.search(r"\br\b", text))
    )
    if prefer_r:
        return "R", "comment"

    prefer_p = (
        re.search(r"\bprefer\w*\s+(?:la\s+)?(?:lettera\s+)?p\b", text)
        or re.search(r"\bp\s+(?:e|è)\s+da\s+preferire\b", text)
        or ("ips" in text and re.search(r"\bp\b", text))
    )
    if prefer_p:
        return "P", "comment"

    if "valore massimo scala" in text or "massimo valore scala" in text:
        return "valore massimo scala", "comment"
    if re.search(r"\bdepth\b", text):
        if re.search(r"\bmm\b", text):
            return "Depth e mm", "comment"
        return "Depth", "comment"
    if re.search(r"\bd\b", text):
        if re.search(r"\bmm\b", text):
            return "D e mm", "comment"
        if re.search(r"\bcm\b", text):
            return "D e cm", "comment"
        return "D", "comment"
    if re.search(r"\bp\b", text):
        if re.search(r"\bmm\b", text):
            return "P e mm", "comment"
        if re.search(r"\bcm\b", text):
            return "P e cm", "comment"
        return "P", "comment"
    if re.search(r"\bcm\b", text):
        return "cm", "comment"
    if re.search(r"\bmm\b", text):
        return "mm", "comment"
    return "", ""


def build(args: argparse.Namespace) -> Dict[str, object]:
    manifest_rows = _read_csv(args.manifest.expanduser().resolve())
    notes_raw = json.loads(args.notes_json.expanduser().read_text(encoding="utf-8"))
    notes: List[Dict[str, object]] = notes_raw if isinstance(notes_raw, list) else notes_raw.get("rows", [])
    notes_by_key = {str(n.get("key") or ""): n for n in notes if n.get("key")}
    propagate_vendors = {v.strip().lower() for v in str(args.propagate_vendors or "").split(",") if v.strip()}

    boxes_by_config: Dict[str, List[Box]] = defaultdict(list)
    hints_by_config: Dict[str, Counter] = defaultdict(Counter)
    for note in notes:
        config = str(note.get("config_folder") or "")
        vendor = str(note.get("vendor") or "")
        if not config or (propagate_vendors and vendor.lower() not in propagate_vendors and not _vendor_matches(config, propagate_vendors)):
            continue
        box = _box_from_note(note)
        if box and str(note.get("correction_source") or "").strip() != "top_ranker":
            boxes_by_config[config].append(box)
        hint, _source = _hint_from_note(note)
        if hint:
            hints_by_config[config][hint] += 1

    config_box = {cfg: _median_box(boxes) for cfg, boxes in boxes_by_config.items()}
    config_hint = {
        cfg: counts.most_common(1)[0][0]
        for cfg, counts in hints_by_config.items()
        if counts
    }

    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    extra = [
        "ranker_review_hint",
        "ranker_review_comment",
        "ranker_review_corrected_gt_box",
        "ranker_review_box_prior",
        "ranker_review_propagated",
        "ranker_review_updated_at",
        "ranker_review_hint_source",
    ]
    for field in extra:
        if field not in fieldnames:
            fieldnames.append(field)

    stats: Counter = Counter(raw_notes=len(notes), manifest_rows=len(manifest_rows))
    out_rows: List[Dict[str, str]] = []
    for row in manifest_rows:
        out = dict(row)
        key = row.get("review_key", "")
        note = notes_by_key.get(key)
        config = row.get("config_folder", "")
        direct_box = _box_from_note(note) if note else None
        propagated_box = config_box.get(config)
        direct_hint, hint_source = _hint_from_note(note)
        propagated_hint = config_hint.get(config, "")

        applied_box: Optional[Box] = None
        propagated = False
        if direct_box:
            applied_box = direct_box
            stats["direct_corrected_boxes"] += 1
        elif propagated_box and _vendor_matches(config, propagate_vendors):
            applied_box = propagated_box
            propagated = True
            stats["propagated_corrected_boxes"] += 1

        if applied_box:
            _apply_box(out, applied_box)
            out["ranker_review_box_prior"] = _box_text(applied_box)
        else:
            out["ranker_review_box_prior"] = ""

        hint = direct_hint or (propagated_hint if _vendor_matches(config, propagate_vendors) else "")
        if not hint_source and hint and hint == propagated_hint:
            hint_source = "propagated"
        if hint:
            stats["rows_with_ranker_hint"] += 1
        if hint_source == "comment":
            stats["rows_with_comment_inferred_hint"] += 1
        if propagated and hint == propagated_hint:
            stats["rows_with_propagated_hint"] += 1

        if note:
            stats["direct_review_rows"] += 1
        out["ranker_review_hint"] = hint
        out["ranker_review_comment"] = str(note.get("comment") or "") if note else ""
        out["ranker_review_corrected_gt_box"] = _box_text(direct_box)
        out["ranker_review_propagated"] = "true" if propagated else "false"
        out["ranker_review_updated_at"] = str(note.get("updated_at") or "") if note else ""
        out["ranker_review_hint_source"] = hint_source
        out_rows.append(out)

    output = args.output_csv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    stats["output_rows"] = len(out_rows)
    stats["configs_with_box_prior"] = len([v for v in config_box.values() if v])
    stats["configs_with_hint_prior"] = len(config_hint)
    stats["output_csv"] = output.as_posix()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply sparse ranker review notes to a RECT_DEPTH manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--notes-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--propagate-vendors", default="hitachi")
    args = parser.parse_args()
    summary = build(args)
    if args.summary_json:
        out = args.summary_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
