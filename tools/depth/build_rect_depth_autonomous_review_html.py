#!/usr/bin/env python3
"""Build a visual review page for autonomous RECT_DEPTH predictions."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


Box = Tuple[float, float, float, float]


def _f(value: object, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
        return out
    except Exception:
        return default


def _safe_slug(text: str, max_len: int = 120) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _box_from_row(row: Dict[str, str], prefix: str = "") -> Optional[Box]:
    keys = [prefix + name for name in ("left", "top", "right", "bottom")]
    vals = [_f(row.get(key), float("nan")) for key in keys]
    if any(v != v for v in vals):
        return None
    left, top, right, bottom = vals
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _pct_box(box: Optional[Box], size: Optional[Tuple[int, int]]) -> Optional[Dict[str, float]]:
    if not box or not size:
        return None
    w, h = size
    if w <= 0 or h <= 0:
        return None
    left, top, right, bottom = box
    return {
        "left": 100.0 * left / w,
        "top": 100.0 * top / h,
        "width": 100.0 * (right - left) / w,
        "height": 100.0 * (bottom - top) / h,
    }


def _copy_asset(src_text: str, assets_dir: Path, index: int, vendor: str) -> Tuple[str, Optional[Tuple[int, int]], bool]:
    src = Path(src_text)
    assets_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".png"
    dest = assets_dir / f"{index:04d}_{_safe_slug(vendor, 40)}_{_safe_slug(src.stem, 80)}{suffix}"
    if not src.exists() and not dest.exists():
        return "", None, False
    if not dest.exists():
        shutil.copy2(src, dest)
    try:
        with Image.open(dest) as im:
            size = (int(im.size[0]), int(im.size[1]))
    except Exception:
        size = None
    return dest.name, size, True


def _detect_visual_scale_marks(image_path: Path) -> List[Box]:
    if cv2 is None or not image_path.exists():
        return []
    im = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return []
    height, width = im.shape[:2]
    if width <= 1 or height <= 1:
        return []
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
        return []
    x_groups: Dict[int, List[Tuple[float, float]]] = {}
    for cx, cy in tick_points:
        x_groups.setdefault(int(round(cx / 12.0)), []).append((cx, cy))
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
        return []
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
                if isinstance(items, list):
                    items.append(component)
                    row["cy"] = sum(float(item[6]) for item in items) / float(len(items))
                break
        else:
            rows.append({"cy": component[6], "items": [component]})
    marks: List[Box] = []
    for row in rows:
        items = row.get("items")
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
    return marks


def _center(box: Box) -> Tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _candidate_depth_for_scale_mark(row: Dict[str, str]) -> float:
    source = str(row.get("candidate_source") or "")
    if not (source == "token" or source.startswith("scale_")):
        return 0.0
    text = str(row.get("ocr_text") or "").strip().lower().replace(",", ".")
    if not re.fullmatch(r"\d+(?:\.\d*)?\s*(?:cm|mm|c|em|tm)?", text):
        return 0.0
    depth = _f(row.get("depth_mm"))
    if depth <= 0 or depth > 160:
        return 0.0
    has_unit = "cm" in text or "mm" in text
    if depth < 10 and not has_unit:
        depth *= 10.0
    return depth if 0 < depth <= 160 else 0.0


def _candidate_zero_for_scale_mark(row: Dict[str, str]) -> bool:
    source = str(row.get("candidate_source") or "")
    if not (source == "token" or source.startswith("scale_")):
        return False
    text = str(row.get("ocr_text") or "").lower().replace(",", ".")
    return bool(re.search(r"(?<![\d.])0(?:\.0+)?\s*(?:cm|mm|c|m)?(?![\d.])", text))


def _choose_scale_endpoint(mark_depths: Sequence[float], zero_hints: Sequence[bool]) -> Optional[int]:
    if not mark_depths:
        return None
    last = len(mark_depths) - 1
    if zero_hints and zero_hints[0] and not zero_hints[last]:
        endpoints = [last]
    elif zero_hints and zero_hints[last] and not zero_hints[0]:
        endpoints = [0]
    else:
        endpoints = [0, last]
    readable = [(idx, mark_depths[idx]) for idx in endpoints if mark_depths[idx] > 0]
    if readable:
        return max(readable, key=lambda item: item[1])[0]
    # OCR may read only intermediate tick values. Their vertical trend can show
    # which endpoint is the maximum, but there is deliberately no inferred mm
    # label attached to that red marker.
    known = [(idx, depth) for idx, depth in enumerate(mark_depths) if depth > 0]
    if len(known) >= 2:
        return last if known[-1][1] > known[0][1] else 0
    return None


def _infer_scale_endpoint_depth(index: int, mark_depths: Sequence[float], zero_hints: Sequence[bool]) -> float:
    if not mark_depths:
        return 0.0
    last = len(mark_depths) - 1
    if zero_hints and ((zero_hints[0] and index == last) or (zero_hints[last] and index == 0)):
        return float(last * 10)
    valid = [(idx, value) for idx, value in enumerate(mark_depths) if value > 0]
    if valid:
        near_idx, near_depth = min(valid, key=lambda item: abs(item[0] - index))
        inferred = near_depth + abs(index - near_idx) * 10.0
        if 0 < inferred <= 160 and inferred > mark_depths[index]:
            return inferred
    if mark_depths[index] > 0:
        return mark_depths[index]
    inferred = float((last + 1) * 10 if index == last else last * 10)
    return inferred if 0 < inferred <= 160 else 0.0


def _scale_marks_for_image(image_path: Path, size: Optional[Tuple[int, int]], image_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    marks = _detect_visual_scale_marks(image_path)
    if not marks or not size:
        return []
    mark_depths: List[float] = []
    zero_hints: List[bool] = []
    for mark in marks:
        mx, my = _center(mark)
        depths: List[float] = []
        zeros = False
        for row in image_rows:
            box = (
                _f(row.get("pred_left")),
                _f(row.get("pred_top")),
                _f(row.get("pred_right")),
                _f(row.get("pred_bottom")),
            )
            cx, cy = _center(box)
            if abs(cx - mx) <= 85 and abs(cy - my) <= 55:
                depth = _candidate_depth_for_scale_mark(row)
                if depth > 0:
                    depths.append(depth)
                zeros = zeros or _candidate_zero_for_scale_mark(row)
        mark_depths.append(max(depths) if depths else 0.0)
        zero_hints.append(zeros)
    # The overlay is evidential: yellow shows every detected tick/value row and
    # red can only be one of the two scale endpoints (opposite an observed
    # zero). Never manufacture a depth by counting ticks or assuming a value.
    max_index = _choose_scale_endpoint(mark_depths, zero_hints)
    uncertain_endpoints = max_index is None and len(marks) >= 2
    out: List[Dict[str, object]] = []
    for idx, mark in enumerate(marks):
        display_depth = mark_depths[idx]
        out.append(
            {
                "box_pct": _pct_box(mark, size),
                "depth_mm": f"{display_depth:.0f}" if display_depth else "",
                "is_max": idx == max_index or (uncertain_endpoints and idx in {0, len(marks) - 1}),
            }
        )
    return out


def _candidate_key(row: Dict[str, str]) -> Tuple[int, float]:
    valid = 1 if _f(row.get("autonomous_valid")) > 0 else 0
    return valid, _f(row.get("autonomous_score"))


def build_records(batch_dir: Path, output_html: Path, max_candidates: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    assets_dir = output_html.parent / "autonomous_review_assets"
    missing_assets = 0
    copied_assets = 0

    for run_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        pred_csv = run_dir / "rect_depth_autonomous_predictions.csv"
        cand_csv = run_dir / "rect_depth_autonomous_candidates.csv"
        summary_json = run_dir / "summary.json"
        if not pred_csv.exists() or not cand_csv.exists():
            continue
        summary = _read_json(summary_json)
        profile = summary.get("profile") if isinstance(summary.get("profile"), dict) else {}
        vendor = str(profile.get("vendor") or run_dir.name)
        probe = str(profile.get("probe") or "")
        pred_rows = _read_csv(pred_csv)
        cand_rows = _read_csv(cand_csv)
        by_image: Dict[str, List[Dict[str, str]]] = {}
        for row in cand_rows:
            by_image.setdefault(str(row.get("source_image") or ""), []).append(row)

        for pred in pred_rows:
            idx = len(records)
            image_path = str(pred.get("image_path") or "")
            asset_name, size, copied = _copy_asset(image_path, assets_dir, idx, vendor)
            copied_assets += int(copied)
            missing_assets += int(not copied)
            image_rows = by_image.get(image_path, [])
            candidates = sorted(image_rows, key=_candidate_key, reverse=True)[:max_candidates]
            scale_marks = _scale_marks_for_image(assets_dir / asset_name, size, image_rows) if asset_name else []
            records.append(
                {
                    "order": idx,
                    "run": run_dir.name,
                    "vendor": vendor,
                    "probe": probe,
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "asset_rel": f"autonomous_review_assets/{asset_name}" if asset_name else "",
                    "image_missing": not copied,
                    "orig_w": size[0] if size else None,
                    "orig_h": size[1] if size else None,
                    "status": pred.get("status", ""),
                    "mode": pred.get("mode", ""),
                    "score": pred.get("score", ""),
                    "ranker_score": pred.get("ranker_score", ""),
                    "depth_mm": pred.get("depth_mm", ""),
                    "ocr_text": pred.get("ocr_text", ""),
                    "reason": pred.get("reason", ""),
                    "box_pct": _pct_box(_box_from_row(pred), size),
                    "scale_marks": scale_marks,
                    "candidates": [
                        {
                            "score": row.get("autonomous_score", ""),
                            "ranker_score": row.get("ranker_score", ""),
                            "status": row.get("autonomous_status", ""),
                            "valid": row.get("autonomous_valid", ""),
                            "mode": row.get("autonomous_mode", ""),
                            "depth_mm": row.get("depth_mm", ""),
                            "ocr_text": row.get("ocr_text", ""),
                            "reason": row.get("autonomous_reason", ""),
                            "box_variant": row.get("box_variant", ""),
                            "box_pct": _pct_box(
                                (
                                    _f(row.get("pred_left")),
                                    _f(row.get("pred_top")),
                                    _f(row.get("pred_right")),
                                    _f(row.get("pred_bottom")),
                                ),
                                size,
                            ),
                        }
                        for row in candidates
                    ],
                }
            )

    summary = {
        "records": len(records),
        "status_counts": dict(Counter(str(r["status"]) for r in records)),
        "mode_counts": dict(Counter(str(r["mode"]) for r in records)),
        "vendors": dict(Counter(str(r["vendor"]) for r in records)),
        "copied_assets": copied_assets,
        "missing_assets": missing_assets,
        "batch_dir": batch_dir.as_posix(),
    }
    return records, summary


def write_html(records: Sequence[Dict[str, object]], summary: Dict[str, object], output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(records, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RECT_DEPTH autonomous review</title>
<style>
:root{{--bg:#0b0f12;--panel:#151a1f;--line:#2d3842;--text:#edf4f7;--muted:#94a4af;--ok:#22c55e;--review:#f59e0b;--reject:#ef4444;--box:#48b9ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(11,15,18,.96);border-bottom:1px solid var(--line);padding:14px 18px}}h1{{margin:0 0 8px;font-size:20px}}
.row{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 9px}}.metric b{{font-size:18px}}
input,select,button,textarea{{background:#0d1216;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:inherit}}button{{cursor:pointer}}input{{min-width:300px;flex:1}}
main{{padding:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}.card.accepted{{box-shadow:0 0 0 2px rgba(34,197,94,.28) inset}}.card.review{{box-shadow:0 0 0 2px rgba(245,158,11,.28) inset}}.card.reject{{box-shadow:0 0 0 2px rgba(239,68,68,.25) inset}}
.thumb{{height:260px;background:#050708;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:zoom-in}}.stage{{position:relative;display:inline-block;line-height:0;max-width:100%;max-height:100%}}.stage img{{display:block;max-width:100%;max-height:260px;width:auto;height:auto}}.missing{{padding:18px;text-align:center;color:var(--muted);word-break:break-word}}
.box{{position:absolute;border:2px solid var(--box);pointer-events:none;min-width:5px;min-height:5px;opacity:.72;box-shadow:0 0 0 1px rgba(0,0,0,.45)}}.box.cand{{border-color:#facc15;border-style:dashed;display:none;opacity:.50}}.box.scale-mark{{border-color:#facc15;background:rgba(250,204,21,.10);opacity:.95;z-index:3}}.box.scale-mark.max{{border-color:#ef4444;background:rgba(239,68,68,.14);box-shadow:0 0 0 1px rgba(0,0,0,.65),0 0 0 3px rgba(239,68,68,.24);z-index:4}}.scale-label{{position:absolute;left:0;top:-18px;background:rgba(5,7,8,.78);border:1px solid currentColor;border-radius:4px;color:#facc15;font-size:11px;line-height:1;padding:2px 4px;white-space:nowrap}}.scale-mark.max .scale-label{{color:#ff6b6b}}body.show-candidates .box.cand{{display:block}}body.hide-main-box .box.top,body.hide-boxes .box,body.hide-scale-values .box.scale-mark{{display:none}}
.body{{padding:10px;display:grid;gap:7px}}.title{{display:flex;gap:6px;align-items:center;min-width:0}}.title b{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.pill{{font-size:12px;border-radius:999px;padding:2px 7px;color:#061014;background:var(--muted)}}.pill.accepted{{background:var(--ok)}}.pill.review{{background:var(--review)}}.pill.reject{{background:var(--reject);color:white}}
.meta{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}.meta div,.reason,.cand{{background:#0d1216;border:1px solid #24303a;border-radius:6px;padding:6px;min-width:0}}.meta span,.cand span{{display:block;color:var(--muted);font-size:11px}}.meta b{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.reason{{color:#d8e5eb;font-size:12px}}.path{{font-size:12px;color:var(--muted);word-break:break-word}}.empty{{display:none;color:var(--muted);text-align:center;padding:40px}}
.lightbox{{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.94);display:none;gap:16px;padding:22px}}.lightbox.open{{display:flex}}.lb-img{{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden}}.lb-img .stage img{{max-height:92vh;max-width:100%}}.panel{{width:430px;max-width:42vw;background:#10161b;border:1px solid #33404b;border-radius:8px;padding:14px;display:grid;gap:10px;overflow:auto}}.panel h2{{margin:0;font-size:17px}}.lb-tools{{display:flex;gap:8px;flex-wrap:wrap}}textarea{{min-height:90px}}.close,.nav{{position:fixed;background:#111820;color:white;border:1px solid #55616b}}.close{{top:14px;right:16px}}.nav{{top:50%;transform:translateY(-50%);font-size:28px;width:46px;height:62px}}.prev{{left:16px}}.next{{right:462px}}.saved{{color:var(--ok);font-size:12px;min-height:18px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.lightbox{{flex-direction:column;overflow:auto}}.panel{{width:auto;max-width:none}}.next{{right:16px}}input{{min-width:180px}}}}
</style>
</head>
<body>
<header>
  <h1>RECT_DEPTH autonomous review</h1>
  <div id="summary" class="row"></div>
  <div class="row" style="margin-top:10px">
    <input id="q" placeholder="Cerca vendor, file, OCR, reason">
    <select id="status"><option value="all">tutti gli stati</option><option>accepted</option><option>review</option><option>reject</option></select>
    <select id="vendor"><option value="all">tutti i vendor</option></select>
    <select id="mode"><option value="all">tutti i modi</option><option>direct_label</option><option>scale</option><option>numeric_accessory</option></select>
    <button id="toggleMainBox">Box top on/off</button>
    <button id="toggleScaleMarks">Valori scala on/off</button>
    <button id="toggleCandidates">Candidati on/off</button>
    <button id="toggleBoxes">Tutti box on/off</button>
    <button id="exportJson">Esporta commenti</button>
  </div>
</header>
<main><div id="grid" class="grid"></div><div id="empty" class="empty">Nessun risultato con questi filtri.</div></main>
<div id="lightbox" class="lightbox">
  <button id="close" class="close">Chiudi</button><button id="prev" class="nav prev">‹</button><button id="next" class="nav next">›</button>
  <div class="lb-img"><span id="lbStage" class="stage"></span></div>
  <aside class="panel"><h2 id="lbTitle"></h2><div id="lbMeta"></div><div class="lb-tools"><button id="lbToggleMainBox">Box top</button><button id="lbToggleScaleMarks">Valori scala</button><button id="lbToggleCandidates">Candidati</button><button id="lbToggleBoxes">Senza box</button></div><div id="lbCands"></div><label>Commento<textarea id="comment"></textarea></label><button id="useReason">Usa spiegazione come commento</button><div id="saved" class="saved"></div></aside>
</div>
<script id="data" type="application/json">{data_json}</script>
<script id="summaryData" type="application/json">{summary_json}</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const SUMMARY=JSON.parse(document.getElementById('summaryData').textContent);
const KEY='rectDepthAutonomousReview:v1';
let notes=loadNotes(), current=-1;
const $=id=>document.getElementById(id);
function esc(s){{return String(s??'').replace(/[&<>"]/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[ch]))}}
function loadNotes(){{try{{return JSON.parse(localStorage.getItem(KEY)||'{{}}')||{{}}}}catch(e){{return {{}}}}}}
function saveNotes(){{localStorage.setItem(KEY,JSON.stringify(notes));}}
function metric(k,v){{return `<div class="metric"><b>${{esc(v)}}</b> ${{esc(k)}}</div>`}}
function boxStyle(b){{return b?`left:${{b.left}}%;top:${{b.top}}%;width:${{b.width}}%;height:${{b.height}}%;`:''}}
function imageHtml(r, large=false){{if(!r.asset_rel)return `<div class="missing">immagine non copiata<br>${{esc(r.image_path)}}</div>`;let boxes='';(r.scale_marks||[]).forEach(m=>{{if(m.box_pct){{const label=m.is_max?(m.depth_mm?`MAX ${{m.depth_mm}}mm`:'MAX? OCR assente'):(m.depth_mm?`${{m.depth_mm}}mm`:'scala');boxes+=`<span class="box scale-mark ${{m.is_max?'max':''}}" style="${{boxStyle(m.box_pct)}}"><span class="scale-label">${{esc(label)}}</span></span>`;}}}});if(r.box_pct&&r.status!=='reject')boxes+=`<span class="box top" style="${{boxStyle(r.box_pct)}}"></span>`;(r.candidates||[]).slice(1,4).forEach(c=>{{if(c.box_pct)boxes+=`<span class="box cand" style="${{boxStyle(c.box_pct)}}"></span>`}});return `<span class="stage"><img src="${{esc(r.asset_rel)}}" alt="">${{boxes}}</span>`}}
function card(r,i){{return `<article class="card ${{esc(r.status)}}" data-i="${{i}}"><div class="thumb">${{imageHtml(r)}}</div><div class="body"><div class="title"><span class="pill ${{esc(r.status)}}">${{esc(r.status)}}</span><b>${{esc(r.image_name)}}</b></div><div class="meta"><div><span>vendor/probe</span><b>${{esc(r.vendor)}} ${{esc(r.probe)}}</b></div><div><span>modo</span><b>${{esc(r.mode)}}</b></div><div><span>depth</span><b>${{esc(r.depth_mm)}} mm</b></div><div><span>score</span><b>${{esc(r.score)}} · ranker ${{esc(r.ranker_score)}}</b></div></div><div class="reason"><b>OCR</b> ${{esc(r.ocr_text||'')}}<br><b>Decisione</b> ${{esc(r.reason||'')}}</div><div class="path">${{esc(r.run)}} · ${{esc(r.image_path)}}</div></div></article>`}}
function filtered(){{const q=$('q').value.toLowerCase(), st=$('status').value, ven=$('vendor').value, mode=$('mode').value;return DATA.filter(r=>(st==='all'||r.status===st)&&(ven==='all'||r.vendor===ven)&&(mode==='all'||r.mode===mode)&&[r.vendor,r.probe,r.image_name,r.ocr_text,r.reason,r.image_path].join(' ').toLowerCase().includes(q));}}
function render(){{const rows=filtered();$('grid').innerHTML=rows.map((r)=>card(r,DATA.indexOf(r))).join('');$('empty').style.display=rows.length?'none':'block';document.querySelectorAll('.card').forEach(el=>el.onclick=()=>openLb(+el.dataset.i));}}
function renderSummary(){{$('summary').innerHTML=metric('record',SUMMARY.records)+metric('accepted',SUMMARY.status_counts.accepted||0)+metric('review',SUMMARY.status_counts.review||0)+metric('reject',SUMMARY.status_counts.reject||0)+metric('asset copiati',SUMMARY.copied_assets)+metric('asset mancanti',SUMMARY.missing_assets);}}
function setupFilters(){{[...new Set(DATA.map(r=>r.vendor))].sort().forEach(v=>$('vendor').insertAdjacentHTML('beforeend',`<option>${{esc(v)}}</option>`));['q','status','vendor','mode'].forEach(id=>$(id).addEventListener('input',render));}}
function candHtml(r){{return (r.candidates||[]).map((c,i)=>`<div class="cand"><span>#${{i+1}} ${{esc(c.status)}} · ${{esc(c.mode)}} · valid ${{esc(c.valid)}} · ${{esc(c.box_variant)}}</span><b>${{esc(c.depth_mm)}} mm · score ${{esc(c.score)}} · ranker ${{esc(c.ranker_score)}}</b><div>${{esc(c.ocr_text)}}</div><div>${{esc(c.reason)}}</div></div>`).join('')}}
function openLb(i){{current=i;const r=DATA[i];$('lightbox').classList.add('open');$('lbStage').innerHTML=imageHtml(r,true);$('lbTitle').textContent=r.image_name;$('lbMeta').innerHTML=`<div class="reason"><b>${{esc(r.vendor)}} ${{esc(r.probe)}}</b><br>${{esc(r.status)}} · ${{esc(r.mode)}} · ${{esc(r.depth_mm)}} mm · score ${{esc(r.score)}}<br>${{esc(r.reason)}}<br><span class="path">${{esc(r.image_path)}}</span></div>`;$('lbCands').innerHTML=candHtml(r);$('comment').value=(notes[r.image_path]||{{}}).comment||'';$('saved').textContent='';}}
function saveCurrent(){{if(current<0)return;const r=DATA[current];notes[r.image_path]={{...(notes[r.image_path]||{{}}),comment:$('comment').value,updated_at:new Date().toISOString(),vendor:r.vendor,status:r.status,mode:r.mode,depth_mm:r.depth_mm,ocr_text:r.ocr_text,reason:r.reason}};saveNotes();$('saved').textContent='salvato';}}
$('close').onclick=()=>$('lightbox').classList.remove('open');$('prev').onclick=()=>openLb((current-1+DATA.length)%DATA.length);$('next').onclick=()=>openLb((current+1)%DATA.length);$('comment').addEventListener('input',()=>{{clearTimeout(window.t);window.t=setTimeout(saveCurrent,200)}});$('useReason').onclick=()=>{{if(current>=0){{$('comment').value=DATA[current].reason||'';saveCurrent();}}}};function toggleBodyClass(cls){{document.body.classList.toggle(cls)}};$('toggleMainBox').onclick=()=>toggleBodyClass('hide-main-box');$('toggleScaleMarks').onclick=()=>toggleBodyClass('hide-scale-values');$('toggleCandidates').onclick=()=>toggleBodyClass('show-candidates');$('toggleBoxes').onclick=()=>toggleBodyClass('hide-boxes');$('lbToggleMainBox').onclick=()=>toggleBodyClass('hide-main-box');$('lbToggleScaleMarks').onclick=()=>toggleBodyClass('hide-scale-values');$('lbToggleCandidates').onclick=()=>toggleBodyClass('show-candidates');$('lbToggleBoxes').onclick=()=>toggleBodyClass('hide-boxes');$('exportJson').onclick=()=>{{const blob=new Blob([JSON.stringify(notes,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='rect_depth_autonomous_review_notes.json';a.click();URL.revokeObjectURL(a.href);}};
document.addEventListener('keydown',e=>{{if(!$('lightbox').classList.contains('open'))return;if(e.key==='Escape')$('lightbox').classList.remove('open');if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();}});
renderSummary();setupFilters();render();
</script>
</body></html>"""
    output_html.write_text(page, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build autonomous RECT_DEPTH visual review HTML.")
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, summary = build_records(args.batch_dir.expanduser().resolve(), args.output_html.expanduser().resolve(), args.max_candidates)
    write_html(records, summary, args.output_html.expanduser().resolve())
    print(json.dumps({"output_html": args.output_html.expanduser().resolve().as_posix(), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
