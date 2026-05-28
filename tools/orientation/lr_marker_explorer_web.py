#!/usr/bin/env python3
"""Interactive marker explorer for reviewed LR marker datasets.

Features:
- Browse train-ready and excluded/review rows.
- Show the full source image and the ultrasound crop.
- Show the DB_echo orientation template(s) expected for the configuration.
- Search the marker template interactively in ROI, crop, full image, top half,
  or bottom half.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
try:
    from skimage.feature import match_template as skimage_match_template
except Exception:  # pragma: no cover - optional speed path
    skimage_match_template = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

Rect = Tuple[int, int, int, int]  # top,left,bottom,right inclusive


@dataclass(frozen=True)
class Template:
    path: Path
    width: int
    height: int
    array: np.ndarray
    array_zero_mean: np.ndarray
    array_norm: float
    tensor_zero_mean: torch.Tensor
    tensor_ones: torch.Tensor
    tensor_norm: float


@dataclass
class AppState:
    rows: List[Dict[str, str]]
    fields: List[str]
    output_dir: Path
    search_decisions: Dict[str, Dict[str, str]]
    template_cache: Dict[str, Optional[Template]]


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _rect_abs(row: Dict[str, str], prefix: str) -> Optional[Rect]:
    vals = [
        _parse_int(row.get(f"{prefix}_top_abs", "")),
        _parse_int(row.get(f"{prefix}_left_abs", "")),
        _parse_int(row.get(f"{prefix}_bottom_abs", "")),
        _parse_int(row.get(f"{prefix}_right_abs", "")),
    ]
    if any(v is None for v in vals):
        return None
    return tuple(int(v) for v in vals)  # type: ignore[return-value]


def _rect_crop(row: Dict[str, str], prefix: str) -> Optional[Rect]:
    vals = [
        _parse_int(row.get(f"{prefix}_top_crop", "")),
        _parse_int(row.get(f"{prefix}_left_crop", "")),
        _parse_int(row.get(f"{prefix}_bottom_crop", "")),
        _parse_int(row.get(f"{prefix}_right_crop", "")),
    ]
    if any(v is None for v in vals):
        return None
    return tuple(int(v) for v in vals)  # type: ignore[return-value]


def _crop_to_abs(rect: Optional[Rect], echo: Optional[Rect]) -> Optional[Rect]:
    if rect is None or echo is None:
        return None
    top, left, bottom, right = rect
    echo_top, echo_left, _echo_bottom, _echo_right = echo
    return top + echo_top, left + echo_left, bottom + echo_top, right + echo_left


def _abs_to_crop(rect: Optional[Rect], echo: Optional[Rect]) -> Optional[Rect]:
    if rect is None or echo is None:
        return None
    top, left, bottom, right = rect
    echo_top, echo_left, _echo_bottom, _echo_right = echo
    return top - echo_top, left - echo_left, bottom - echo_top, right - echo_left


def _box(rect: Optional[Rect]) -> Optional[Dict[str, int]]:
    if rect is None:
        return None
    top, left, bottom, right = rect
    return {
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
        "width": int(right - left + 1),
        "height": int(bottom - top + 1),
    }


def _clip_rect(rect: Rect, width: int, height: int) -> Optional[Rect]:
    top, left, bottom, right = rect
    top = max(0, min(height - 1, top))
    left = max(0, min(width - 1, left))
    bottom = max(0, min(height - 1, bottom))
    right = max(0, min(width - 1, right))
    if bottom <= top or right <= left:
        return None
    return top, left, bottom, right


def _read_manifest(path: Path, source: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = []
        for idx, row in enumerate(reader):
            row = dict(row)
            row["_source_manifest"] = source
            row["_source_index"] = str(idx)
            row["_uid"] = f"{source}::{row.get('review_batch', '')}::{row.get('sample_id', '')}::{idx}"
            rows.append(row)
        return rows, fields


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "item"


def _load_search_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            uid = row.get("uid", "")
            if uid:
                out[uid] = dict(row)
    return out


def _write_search_decisions(path: Path, decisions: Dict[str, Dict[str, str]]) -> None:
    fields = [
        "uid",
        "source_manifest",
        "sample_id",
        "manufacturer",
        "review_batch",
        "action",
        "note",
        "scope",
        "score",
        "template_path",
        "top_abs",
        "left_abs",
        "bottom_abs",
        "right_abs",
        "marker_side",
        "lr_label_it",
        "image_path",
        "config_path",
        "updated_at_epoch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for uid in sorted(decisions):
            writer.writerow({field: decisions[uid].get(field, "") for field in fields})
    tmp.replace(path)


def _count(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = row.get(field, "") or "(empty)"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _template_size(path: Path) -> Tuple[int, int]:
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return 0, 0


def _template_paths_for_row(row: Dict[str, str]) -> List[Path]:
    paths: List[Path] = []
    explicit = row.get("template_path", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            paths.append(p)

    config = Path(row.get("config_path", "")).expanduser()
    fss = Path(row.get("fss_path", "")).expanduser()
    setup_token = ""
    if fss.name:
        setup_token = fss.stem.split("_")[-1]
    db_echo = config / "DB_echo"
    preferred = db_echo / f"setup_{setup_token}" if setup_token else None
    if preferred is not None and preferred.is_dir():
        for p in sorted(preferred.glob("orientation_*.png")):
            resolved = p.resolve()
            if resolved not in paths:
                paths.append(resolved)
    if db_echo.is_dir():
        for p in sorted(db_echo.rglob("orientation_*.png")):
            resolved = p.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _load_template(path: Path) -> Optional[Template]:
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if width < 4 or height < 4:
        return None
    if float(arr.std()) < 1e-6:
        return None
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    zm_arr = arr - float(arr.mean())
    norm = float(np.sqrt(max(float((zm_arr * zm_arr).sum()), 1e-12)))
    if norm < 1e-6:
        return None
    zm = torch.from_numpy(zm_arr).float().unsqueeze(0).unsqueeze(0)
    return Template(
        path=path.resolve(),
        width=width,
        height=height,
        array=arr,
        array_zero_mean=zm_arr,
        array_norm=norm,
        tensor_zero_mean=zm,
        tensor_ones=torch.ones_like(zm),
        tensor_norm=norm,
    )


def _ncc_map(search_img: torch.Tensor, template: Template) -> Optional[torch.Tensor]:
    height, width = int(search_img.shape[0]), int(search_img.shape[1])
    if height < template.height or width < template.width:
        return None
    x = search_img.float().unsqueeze(0).unsqueeze(0)
    n = float(template.width * template.height)
    numerator = F.conv2d(x, template.tensor_zero_mean)
    sum_x = F.conv2d(x, template.tensor_ones)
    sum_x2 = F.conv2d(x * x, template.tensor_ones)
    var_x = torch.clamp(sum_x2 - (sum_x * sum_x) / n, min=0.0)
    denominator = torch.sqrt(var_x) * template.tensor_norm
    valid = denominator > 1e-6
    ncc = torch.where(valid, numerator / (denominator + 1e-8), torch.full_like(numerator, -1.0))
    return torch.nan_to_num(torch.clamp(ncc[0, 0], min=-1.0, max=1.0), nan=-1.0, posinf=-1.0, neginf=-1.0)


def _exact_ncc_at(search_arr: np.ndarray, template: Template, y: int, x: int) -> float:
    patch = search_arr[y : y + template.height, x : x + template.width]
    if patch.shape != template.array.shape:
        return -1.0
    n = float(template.width * template.height)
    sum_x = float(patch.sum())
    sum_x2 = float((patch * patch).sum())
    variance = max(sum_x2 - (sum_x * sum_x) / n, 0.0)
    denominator = math.sqrt(variance) * template.array_norm
    if denominator <= 1e-6:
        return -1.0
    score = float((patch * template.array_zero_mean).sum()) / (denominator + 1e-8)
    return max(-1.0, min(1.0, score))


def _rough_positions(search_arr: np.ndarray, template: Template, count: int) -> Optional[List[Tuple[int, int]]]:
    if skimage_match_template is None:
        return None
    try:
        rough = skimage_match_template(search_arr, template.array, pad_input=False)
    except Exception:
        return None
    if rough.size <= 0:
        return []
    k = min(max(1, count), int(rough.size))
    flat = rough.reshape(-1)
    idxs = np.argpartition(flat, -k)[-k:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    out_w = int(rough.shape[1])
    return [(int(idx) // out_w, int(idx) % out_w) for idx in idxs.tolist()]


def _candidate_far_enough(candidate: Dict[str, object], accepted: Sequence[Dict[str, object]]) -> bool:
    cbox = candidate["box_abs"]
    assert isinstance(cbox, dict)
    cx = (float(cbox["left"]) + float(cbox["right"])) / 2.0
    cy = (float(cbox["top"]) + float(cbox["bottom"])) / 2.0
    cw = float(cbox["width"])
    ch = float(cbox["height"])
    for item in accepted:
        ibox = item["box_abs"]
        assert isinstance(ibox, dict)
        ix = (float(ibox["left"]) + float(ibox["right"])) / 2.0
        iy = (float(ibox["top"]) + float(ibox["bottom"])) / 2.0
        threshold = max(cw, ch, float(ibox["width"]), float(ibox["height"])) * 1.2
        if math.hypot(cx - ix, cy - iy) <= threshold:
            return False
    return True


def _search_scope_rect(row: Dict[str, str], scope: str, image_size: Tuple[int, int]) -> Optional[Rect]:
    width, height = image_size
    echo = _rect_abs(row, "echo_rect")
    roi_abs = _crop_to_abs(_rect_crop(row, "roi"), echo)
    if scope == "roi":
        return _clip_rect(roi_abs, width, height) if roi_abs else None
    if scope == "crop":
        return _clip_rect(echo, width, height) if echo else None
    if scope == "full":
        return 0, 0, height - 1, width - 1
    if echo is None:
        return None
    echo = _clip_rect(echo, width, height)
    if echo is None:
        return None
    top, left, bottom, right = echo
    cut = top + ((bottom - top + 1) // 2)
    if scope == "top":
        return top, left, max(top, cut - 1), right
    if scope == "bottom":
        return min(bottom, cut), left, bottom, right
    return None


def _search_marker(state: AppState, row: Dict[str, str], scope: str, max_candidates: int) -> Dict[str, object]:
    image_path = Path(row.get("image_path", "")).expanduser().resolve()
    with Image.open(image_path) as img:
        gray_full = img.convert("L")
        width, height = gray_full.size
        rect = _search_scope_rect(row, scope, (width, height))
        if rect is None:
            return {"scope": scope, "candidates": [], "error": "search rect unavailable"}
        top, left, bottom, right = rect
        search_img = gray_full.crop((left, top, right + 1, bottom + 1))
        arr = np.asarray(search_img, dtype=np.float32) / 255.0
    search_tensor = torch.from_numpy(arr).float()

    templates: List[Template] = []
    for path in _template_paths_for_row(row):
        key = path.as_posix()
        if key not in state.template_cache:
            state.template_cache[key] = _load_template(path)
        tpl = state.template_cache[key]
        if tpl is not None:
            templates.append(tpl)

    candidates: List[Dict[str, object]] = []
    for template in templates:
        rough_count = min(max(max_candidates * 256, 512), max(1, int(arr.size)))
        positions = _rough_positions(arr, template, rough_count)
        scored: List[Tuple[float, int, int]] = []
        if positions is not None:
            for y, x in positions:
                scored.append((_exact_ncc_at(arr, template, y, x), y, x))
            scored.sort(key=lambda item: item[0], reverse=True)
        else:
            ncc = _ncc_map(search_tensor, template)
            if ncc is None:
                continue
            flat = ncc.reshape(-1)
            k = min(max_candidates * 8, int(flat.numel()))
            if k <= 0:
                continue
            vals, idxs = torch.topk(flat, k=k)
            out_w = int(ncc.shape[1])
            scored = [(float(val), int(idx) // out_w, int(idx) % out_w) for val, idx in zip(vals.tolist(), idxs.tolist())]
        for val, y, x in scored:
            abs_rect = (top + y, left + x, top + y + template.height - 1, left + x + template.width - 1)
            echo = _rect_abs(row, "echo_rect")
            cand = {
                "score": float(val),
                "scope": scope,
                "template_path": template.path.as_posix(),
                "box_abs": _box(abs_rect),
                "box_crop": _box(_abs_to_crop(abs_rect, echo)),
            }
            if _candidate_far_enough(cand, candidates):
                candidates.append(cand)
            if len(candidates) >= max_candidates:
                break
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        candidates = candidates[:max_candidates]
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return {"scope": scope, "search_rect_abs": _box(rect), "candidates": candidates[:max_candidates]}


def _marker_side_from_abs_box(row: Dict[str, str], box_abs: Dict[str, object]) -> Tuple[str, str]:
    echo = _rect_abs(row, "echo_rect")
    left = _parse_int(box_abs.get("left", ""))
    right = _parse_int(box_abs.get("right", ""))
    if echo is None or left is None or right is None:
        return "", ""
    _top, echo_left, _bottom, echo_right = echo
    marker_cx = (left + right) / 2.0
    echo_cx = (echo_left + echo_right) / 2.0
    side = "left" if marker_cx < echo_cx else "right"
    label = "non_specchiata" if side == "left" else "specchiata_a_destra"
    return side, label


def _row_payload(state: AppState, idx: int) -> Dict[str, object]:
    row = state.rows[idx]
    uid = row["_uid"]
    echo_abs = _rect_abs(row, "echo_rect")
    roi_crop = _rect_crop(row, "roi")
    roi_abs = _crop_to_abs(roi_crop, echo_abs)
    marker_crop = _rect_crop(row, "marker")
    marker_abs = _crop_to_abs(marker_crop, echo_abs)
    if marker_abs is None:
        marker_abs = _rect_abs(row, "marker")
        marker_crop = _abs_to_crop(marker_abs, echo_abs)
    templates = []
    for i, path in enumerate(_template_paths_for_row(row)):
        width, height = _template_size(path)
        templates.append(
            {
                "index": i,
                "path": path.as_posix(),
                "url": f"/api/template?uid={quote(uid)}&index={i}",
                "name": path.name,
                "width": width,
                "height": height,
            }
        )
    return {
        "index": idx,
        "uid": uid,
        "sample_id": row.get("sample_id", ""),
        "source_manifest": row.get("_source_manifest", ""),
        "manufacturer": row.get("manufacturer", ""),
        "review_batch": row.get("review_batch", ""),
        "review_decision": row.get("review_decision", ""),
        "review_note": row.get("review_note", ""),
        "review_reason": row.get("review_reason", ""),
        "label_kind": row.get("label_kind", ""),
        "config_folder": row.get("config_folder", ""),
        "image_name": row.get("image_name", ""),
        "image_path": row.get("image_path", ""),
        "template_path": row.get("template_path", ""),
        "su_giu_pred": row.get("su_giu_pred", ""),
        "su_giu_conf": row.get("su_giu_conf", ""),
        "lr_label_it": row.get("lr_label_it", ""),
        "detected_marker_side": row.get("detected_marker_side", ""),
        "match_score": row.get("match_score", ""),
        "full_url": f"/api/full?uid={quote(uid)}",
        "crop_url": f"/api/crop?uid={quote(uid)}",
        "boxes": {
            "echo_abs": _box(echo_abs),
            "roi_abs": _box(roi_abs),
            "marker_abs": _box(marker_abs),
            "roi_crop": _box(roi_crop),
            "marker_crop": _box(marker_crop),
        },
        "templates": templates,
        "search_decision": state.search_decisions.get(uid, {}),
    }


def _image_bytes(row: Dict[str, str], crop: bool) -> bytes:
    image_path = Path(row.get("image_path", "")).expanduser().resolve()
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if crop:
            echo = _rect_abs(row, "echo_rect")
            if echo is not None:
                top, left, bottom, right = echo
                rect = _clip_rect(echo, rgb.width, rgb.height)
                if rect is not None:
                    top, left, bottom, right = rect
                    rgb = rgb.crop((left, top, right + 1, bottom + 1))
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


HTML = r"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LR Marker Explorer</title>
  <style>
    *{box-sizing:border-box}
    body{margin:0;background:#f4f6f8;color:#17202a;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
    header,.filters{background:#fff;border-bottom:1px solid #d0d5dd;padding:10px 14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
    h1{font-size:16px;margin:0 18px 0 0}
    select,input,button,textarea{font:inherit;border:1px solid #cfd6e0;border-radius:6px;background:#fff;padding:7px}
    button{cursor:pointer}.primary{background:#2563eb;border-color:#2563eb;color:#fff}.good{background:#16803c;border-color:#16803c;color:#fff}.warn{background:#b7791f;border-color:#b7791f;color:#fff}
    main{display:grid;grid-template-columns:minmax(0,1fr)420px;gap:12px;padding:12px}
    .panel{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:10px;min-width:0}
    .views{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .wrap{position:relative;background:#111827;overflow:auto;max-height:calc(100vh - 250px);border-radius:6px}
    .wrap img{display:block;width:100%;height:auto}.wrap canvas{position:absolute;left:0;top:0;pointer-events:auto}
    .caption{font-size:12px;color:#667085;margin:0 0 5px}.bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
    .kv{display:grid;grid-template-columns:135px 1fr;gap:5px 8px}.kv div:nth-child(odd){color:#667085}.kv div:nth-child(even){font:12px ui-monospace,Menlo,monospace;overflow-wrap:anywhere}
    .templates{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.tpl{border:1px solid #cfd6e0;border-radius:6px;padding:6px;background:#fafafa;cursor:pointer}.tpl.active{border-color:#2563eb;background:#eff6ff}.tpl img{image-rendering:pixelated;max-width:90px;max-height:70px;display:block;background:#111827}
    .results{max-height:220px;overflow:auto;border-top:1px solid #e4e7ec;margin-top:10px;padding-top:8px}.result{padding:6px;border:1px solid #d0d5dd;border-radius:6px;margin-bottom:6px;background:#fafafa;cursor:pointer}.result.active{border-color:#06b6d4;background:#ecfeff}
    textarea{width:100%;min-height:64px;margin-top:8px}.muted{color:#667085}.legend{font-size:12px;color:#667085;margin-bottom:8px}
    @media(max-width:1200px){main{grid-template-columns:1fr}.views{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header><h1>LR Marker Explorer</h1><span id="status" class="muted"></span></header>
<section class="filters">
  <select id="mode"><option value="excluded" selected>Da rivedere/esclusi</option><option value="train">Train-ready</option><option value="all">Tutti</option></select>
  <select id="vendor"></select>
  <select id="decision"></select>
  <input id="q" placeholder="cerca vendor, sample, config">
</section>
<main>
  <section class="panel">
    <div class="views">
      <div><div class="caption">Immagine completa</div><div class="wrap"><img id="fullImg"><canvas id="fullCanvas"></canvas></div></div>
      <div><div class="caption">Crop rettangolo ecografico</div><div class="wrap"><img id="cropImg"><canvas id="cropCanvas"></canvas></div></div>
    </div>
    <div class="bar">
      <button id="prev">← Prec</button><button id="next" class="primary">Succ →</button>
      <span id="pos" class="muted"></span>
    </div>
  </section>
  <aside class="panel">
    <div class="legend">giallo: rettangolo eco · blu: ROI su/giu · rosso: marker corrente · ciano: ricerca template</div>
    <div id="meta" class="kv"></div>
    <div class="templates" id="templates"></div>
    <div class="bar">
      <button data-scope="roi">Cerca ROI</button>
      <button data-scope="crop">Cerca crop</button>
      <button data-scope="full">Cerca full</button>
      <button data-scope="top">Cerca alto</button>
      <button data-scope="bottom">Cerca basso</button>
      <button id="manualToggle">Correzione manuale</button>
    </div>
    <div class="results" id="results"></div>
    <textarea id="note" placeholder="nota opzionale"></textarea>
    <div class="bar">
      <button class="good" id="saveCandidate">Salva correzione</button>
      <button class="warn" id="keepFuture">Lascia da rivedere</button>
    </div>
    <div id="msg" class="muted"></div>
  </aside>
</main>
<script>
const S={meta:null,index:0,total:0,sample:null,candidates:[],selected:null,selectedTemplate:null,manual:false};
const $=id=>document.getElementById(id);
async function J(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw Error(d.error||'errore');return d}
function esc(v){return String(v||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
function params(){return new URLSearchParams({mode:$('mode').value,vendor:$('vendor').value,decision:$('decision').value,q:$('q').value})}
function fill(id,label,vals){const e=$(id),cur=e.value;e.innerHTML='<option value="">'+label+'</option>';vals.forEach(v=>e.innerHTML+='<option>'+esc(v)+'</option>');if(vals.includes(cur))e.value=cur}
async function loadMeta(){S.meta=await J('/api/meta');fill('vendor','Vendor',S.meta.vendors);fill('decision','Decisione',S.meta.decisions_available);$('status').textContent=`${S.meta.total} righe · ${JSON.stringify(S.meta.source_counts)}`}
async function load(i){const p=params();p.set('index',Math.max(0,i));const d=await J('/api/sample?'+p);S.index=d.index;S.total=d.total;S.sample=d.sample;S.candidates=[];S.selected=null;S.selectedTemplate=null;$('pos').textContent=d.total?`${d.index+1}/${d.total}`:'nessuna riga';$('results').innerHTML='';if(!d.sample)return;renderMeta(d.sample);renderTemplates(d.sample);$('fullImg').onload=draw;$('cropImg').onload=draw;$('fullImg').src=d.sample.full_url+'&t='+Date.now();$('cropImg').src=d.sample.crop_url+'&t='+Date.now()}
function renderMeta(s){const rows=[['uid',s.uid],['vendor',s.manufacturer],['decision',s.review_decision],['reason',s.review_reason],['batch',s.review_batch],['su/giu',`${s.su_giu_pred} conf ${s.su_giu_conf}`],['LR',`${s.lr_label_it} ${s.detected_marker_side}`],['score',s.match_score],['config',s.config_folder],['file',s.image_name]];$('meta').innerHTML=rows.map(r=>`<div>${r[0]}</div><div>${esc(r[1])}</div>`).join('');$('note').value=s.search_decision?.note||''}
function renderTemplates(s){S.selectedTemplate=s.templates[0]||null;$('templates').innerHTML=s.templates.map((t,i)=>`<div class="tpl ${i===0?'active':''}" data-i="${i}"><img src="${t.url}" title="${esc(t.path)}"><div class="muted">${esc(t.name)} · ${t.width}x${t.height}</div></div>`).join('')||'<span class="muted">Nessun template trovato</span>';document.querySelectorAll('.tpl').forEach(el=>el.onclick=()=>{S.selectedTemplate=S.sample.templates[Number(el.dataset.i)];document.querySelectorAll('.tpl').forEach(x=>x.classList.remove('active'));el.classList.add('active')})}
function boxAbsToCrop(b){const e=S.sample.boxes.echo_abs;if(!e||!b)return null;return {top:b.top-e.top,left:b.left-e.left,bottom:b.bottom-e.top,right:b.right-e.left,width:b.width,height:b.height}}
function boxCropToAbs(b){const e=S.sample.boxes.echo_abs;if(!e||!b)return null;return {top:b.top+e.top,left:b.left+e.left,bottom:b.bottom+e.top,right:b.right+e.left,width:b.width,height:b.height}}
function rect(ctx,b,color,w,dash,sx,sy){if(!b)return;const l=b.left*sx,t=b.top*sy,r=b.right*sx,bt=b.bottom*sy;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=w;if(dash)ctx.setLineDash(dash);ctx.strokeRect(l+.5,t+.5,Math.max(1,r-l),Math.max(1,bt-t));ctx.restore()}
function boxForSpace(box,space,source){if(!box)return null;if(space===source)return box;if(space==='abs'&&source==='crop')return boxCropToAbs(box);if(space==='crop'&&source==='abs')return boxAbsToCrop(box);return box}
function drawCanvas(imgId,canvasId,space){const s=S.sample,img=$(imgId),c=$(canvasId);if(!s||!img.naturalWidth)return;const r=img.getBoundingClientRect();const dpr=window.devicePixelRatio||1;const displayW=Math.max(1,Math.round(r.width));const displayH=Math.max(1,Math.round(r.height));const sx=displayW/img.naturalWidth,sy=displayH/img.naturalHeight;c.width=Math.round(displayW*dpr);c.height=Math.round(displayH*dpr);c.style.width=displayW+'px';c.style.height=displayH+'px';c.style.cursor=S.manual?'crosshair':'default';const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,displayW,displayH);const b=s.boxes;if(space==='abs'){rect(x,b.echo_abs,'#f59e0b',3,[8,4],sx,sy);rect(x,boxCropToAbs(b.roi_crop),'#3b82f6',3,[8,4],sx,sy);rect(x,boxCropToAbs(b.marker_crop),'#ef4444',3,null,sx,sy)}else{rect(x,b.roi_crop,'#3b82f6',3,[8,4],sx,sy);rect(x,b.marker_crop,'#ef4444',3,null,sx,sy)};for(const cand of S.candidates){rect(x,space==='abs'?boxForSpace(cand.box_abs||cand.box_crop,space,cand.box_abs?'abs':'crop'):boxForSpace(cand.box_crop||cand.box_abs,space,cand.box_crop?'crop':'abs'),'#06b6d4',2,[4,3],sx,sy)};if(S.selected){rect(x,space==='abs'?boxForSpace(S.selected.box_abs||S.selected.box_crop,space,S.selected.box_abs?'abs':'crop'):boxForSpace(S.selected.box_crop||S.selected.box_abs,space,S.selected.box_crop?'crop':'abs'),'#22c55e',4,null,sx,sy)}}
function draw(){drawCanvas('fullImg','fullCanvas','abs');drawCanvas('cropImg','cropCanvas','crop')}
function clampBox(cx,cy,w,h,maxW,maxH){let l=Math.round(cx-w/2),t=Math.round(cy-h/2);l=Math.max(0,Math.min(maxW-w,l));t=Math.max(0,Math.min(maxH-h,t));return {top:t,left:l,bottom:t+h-1,right:l+w-1,width:w,height:h}}
function placeManual(evt,space){if(!S.manual||!S.sample)return;const c=evt.currentTarget,img=$(space==='abs'?'fullImg':'cropImg'),r=c.getBoundingClientRect();const x=(evt.clientX-r.left)*img.naturalWidth/r.width,y=(evt.clientY-r.top)*img.naturalHeight/r.height;const tpl=S.selectedTemplate||{};const old=S.sample.boxes.marker_crop||S.sample.boxes.marker_abs||{};const w=Math.max(4,Number(tpl.width||old.width||24)),h=Math.max(4,Number(tpl.height||old.height||24));let boxAbs,boxCrop;if(space==='abs'){boxAbs=clampBox(x,y,w,h,img.naturalWidth,img.naturalHeight);boxCrop=boxAbsToCrop(boxAbs)}else{boxCrop=clampBox(x,y,w,h,img.naturalWidth,img.naturalHeight);boxAbs=boxCropToAbs(boxCrop)}S.selected={score:'manual',scope:'manual_'+space,template_path:tpl.path||'',box_abs:boxAbs,box_crop:boxCrop};S.candidates=[S.selected];renderResults();draw();$('msg').textContent='correzione manuale pronta'}
async function search(scope){$('msg').textContent='cerco...';const d=await J(`/api/search?uid=${encodeURIComponent(S.sample.uid)}&scope=${scope}&n=12`);S.candidates=d.candidates||[];S.selected=S.candidates[0]||null;renderResults();draw();$('msg').textContent=`${S.candidates.length} candidate in ${scope}`}
function scoreText(v){const n=Number(v);return Number.isFinite(n)?n.toFixed(4):esc(v)}
function renderResults(){$('results').innerHTML=S.candidates.map((c,i)=>`<div class="result ${i===0?'active':''}" data-i="${i}"><b>${i+1}</b> score ${scoreText(c.score)} · ${esc(c.scope)}<br><span class="muted">${esc((c.template_path||'').split('/').slice(-3).join('/'))}</span></div>`).join('');document.querySelectorAll('.result').forEach(el=>el.onclick=()=>{S.selected=S.candidates[Number(el.dataset.i)];document.querySelectorAll('.result').forEach(x=>x.classList.remove('active'));el.classList.add('active');draw()})}
async function save(action){if(!S.sample)return;if(action==='candidate_found'&&!S.selected){$('msg').textContent='seleziona una candidate o piazza una correzione manuale';return}const payload={uid:S.sample.uid,action,note:$('note').value,candidate:S.selected};const d=await J('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});$('msg').textContent=d.message;await loadMeta()}
['mode','vendor','decision'].forEach(id=>$(id).onchange=()=>load(0));$('q').oninput=()=>{clearTimeout(S.t);S.t=setTimeout(()=>load(0),250)};$('prev').onclick=()=>load(S.index-1);$('next').onclick=()=>load(S.index+1);document.querySelectorAll('button[data-scope]').forEach(b=>b.onclick=()=>search(b.dataset.scope));$('manualToggle').onclick=()=>{S.manual=!S.manual;$('manualToggle').classList.toggle('primary',S.manual);$('msg').textContent=S.manual?'clicca sulla completa o sul crop':'correzione manuale disattivata';draw()};$('fullCanvas').onclick=e=>placeManual(e,'abs');$('cropCanvas').onclick=e=>placeManual(e,'crop');$('saveCandidate').onclick=()=>save('candidate_found');$('keepFuture').onclick=()=>save('keep_review');window.onresize=draw;window.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight')load(S.index+1);if(e.key==='ArrowLeft')load(S.index-1)};
loadMeta().then(()=>load(0)).catch(e=>$('status').textContent=e.message);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: AppState

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)

    def _json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def _bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def _row_by_uid(self, uid: str) -> Optional[Dict[str, str]]:
        return next((row for row in self.state.rows if row["_uid"] == uid), None)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self._bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/meta":
            self._json(
                {
                    "total": len(self.state.rows),
                    "vendors": sorted({row.get("manufacturer", "") for row in self.state.rows if row.get("manufacturer", "")}),
                    "decisions_available": sorted({row.get("review_decision", "") for row in self.state.rows if row.get("review_decision", "")}),
                    "source_counts": _count(self.state.rows, "_source_manifest"),
                    "vendor_counts": _count(self.state.rows, "manufacturer"),
                }
            )
            return
        if parsed.path == "/api/sample":
            filters = {key: (query.get(key, [""])[0] or "") for key in ("mode", "vendor", "decision", "q")}
            rows = self._filtered_rows(filters)
            idx = _parse_int(query.get("index", ["0"])[0]) or 0
            if not rows:
                self._json({"total": 0, "index": 0, "sample": None})
                return
            idx = max(0, min(idx, len(rows) - 1))
            source_idx = self.state.rows.index(rows[idx])
            self._json({"total": len(rows), "index": idx, "source_index": source_idx, "sample": _row_payload(self.state, source_idx)})
            return
        if parsed.path in {"/api/full", "/api/crop"}:
            uid = unquote(query.get("uid", [""])[0])
            row = self._row_by_uid(uid)
            if row is None:
                self._json({"error": "unknown uid"}, 404)
                return
            try:
                self._bytes(_image_bytes(row, crop=(parsed.path == "/api/crop")), "image/jpeg")
            except Exception as exc:  # pylint: disable=broad-except
                self._json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/template":
            uid = unquote(query.get("uid", [""])[0])
            index = _parse_int(query.get("index", ["0"])[0]) or 0
            row = self._row_by_uid(uid)
            if row is None:
                self._json({"error": "unknown uid"}, 404)
                return
            paths = _template_paths_for_row(row)
            if index < 0 or index >= len(paths):
                self._json({"error": "template index unavailable"}, 404)
                return
            self._bytes(paths[index].read_bytes(), "image/png")
            return
        if parsed.path == "/api/search":
            uid = unquote(query.get("uid", [""])[0])
            scope = query.get("scope", ["roi"])[0]
            max_candidates = _parse_int(query.get("n", ["10"])[0]) or 10
            row = self._row_by_uid(uid)
            if row is None:
                self._json({"error": "unknown uid"}, 404)
                return
            try:
                self._json(_search_marker(self.state, row, scope=scope, max_candidates=max_candidates))
            except Exception as exc:  # pylint: disable=broad-except
                self._json({"error": str(exc)}, 500)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        uid = str(payload.get("uid", ""))
        row = self._row_by_uid(uid)
        if row is None:
            self._json({"error": "unknown uid"}, 404)
            return
        action = str(payload.get("action", ""))
        candidate = payload.get("candidate") or {}
        box_abs = candidate.get("box_abs") if isinstance(candidate, dict) else {}
        marker_side, lr_label_it = _marker_side_from_abs_box(row, box_abs) if isinstance(box_abs, dict) else ("", "")
        decision = {
            "uid": uid,
            "source_manifest": row.get("_source_manifest", ""),
            "sample_id": row.get("sample_id", ""),
            "manufacturer": row.get("manufacturer", ""),
            "review_batch": row.get("review_batch", ""),
            "action": action,
            "note": str(payload.get("note", "") or ""),
            "scope": str(candidate.get("scope", "")) if isinstance(candidate, dict) else "",
            "score": str(candidate.get("score", "")) if isinstance(candidate, dict) else "",
            "template_path": str(candidate.get("template_path", "")) if isinstance(candidate, dict) else "",
            "top_abs": str(box_abs.get("top", "")) if isinstance(box_abs, dict) else "",
            "left_abs": str(box_abs.get("left", "")) if isinstance(box_abs, dict) else "",
            "bottom_abs": str(box_abs.get("bottom", "")) if isinstance(box_abs, dict) else "",
            "right_abs": str(box_abs.get("right", "")) if isinstance(box_abs, dict) else "",
            "marker_side": marker_side,
            "lr_label_it": lr_label_it,
            "image_path": row.get("image_path", ""),
            "config_path": row.get("config_path", ""),
            "updated_at_epoch": f"{time.time():.3f}",
        }
        self.state.search_decisions[uid] = decision
        _write_search_decisions(self.state.output_dir / "marker_explorer_decisions.csv", self.state.search_decisions)
        self._json({"ok": True, "message": f"salvato: {action}"})

    def _filtered_rows(self, filters: Dict[str, str]) -> List[Dict[str, str]]:
        mode = filters.get("mode", "excluded")
        vendor = filters.get("vendor", "")
        decision = filters.get("decision", "")
        query = filters.get("q", "").strip().lower()
        out: List[Dict[str, str]] = []
        for row in self.state.rows:
            if mode == "excluded" and row.get("_source_manifest") != "excluded":
                continue
            if mode == "train" and row.get("_source_manifest") != "train":
                continue
            if vendor and row.get("manufacturer", "") != vendor:
                continue
            if decision and row.get("review_decision", "") != decision:
                continue
            if query:
                blob = " ".join(
                    [
                        row.get("sample_id", ""),
                        row.get("manufacturer", ""),
                        row.get("config_folder", ""),
                        row.get("image_name", ""),
                        row.get("review_reason", ""),
                        row.get("review_note", ""),
                    ]
                ).lower()
                if query not in blob:
                    continue
            out.append(row)
        return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve LR marker explorer.")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_reviewed_v1/manifest_train_ready.csv"),
    )
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_reviewed_v1/manifest_excluded.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_marker_explorer"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_fields = _read_manifest(args.train_manifest.expanduser().resolve(), "train")
    excluded_rows, excluded_fields = _read_manifest(args.excluded_manifest.expanduser().resolve(), "excluded")
    rows = excluded_rows + train_rows
    fields = sorted(set(train_fields + excluded_fields))
    state = AppState(
        rows=rows,
        fields=fields,
        output_dir=output_dir,
        search_decisions=_load_search_decisions(output_dir / "marker_explorer_decisions.csv"),
        template_cache={},
    )
    Handler.state = state
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"Marker explorer: http://{args.host}:{args.port}", flush=True)
    print(f"Rows: {len(rows)} (excluded={len(excluded_rows)}, train={len(train_rows)})", flush=True)
    print(f"Decisions: {output_dir / 'marker_explorer_decisions.csv'}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
