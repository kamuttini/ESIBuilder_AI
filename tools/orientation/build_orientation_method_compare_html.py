#!/usr/bin/env python3
"""Build an interactive HTML comparison for old vs bundle marker orientation logic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        return default
    return default


def _to_int(value: object, default: Optional[int] = None) -> Optional[int]:
    f = _to_float(value, None)
    if f is None:
        return default
    return int(round(f))


def _file_url(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        return Path(path_text).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _key(row: Dict[str, str]) -> str:
    image_path = str(row.get("image_path", "") or "").strip()
    if image_path:
        return image_path
    folder = str(row.get("folder_name", "") or row.get("folder_path", "") or "").strip()
    idx = str(row.get("image_index", "") or "").strip()
    return f"{folder}::{idx}"


def _rect_from(row: Dict[str, str], prefix: str = "") -> Optional[Tuple[int, int, int, int]]:
    names = (
        f"{prefix}top_abs",
        f"{prefix}left_abs",
        f"{prefix}bottom_abs",
        f"{prefix}right_abs",
    )
    vals = [_to_int(row.get(name), None) for name in names]
    if any(v is None for v in vals):
        return None
    t, l, b, r = [int(v) for v in vals if v is not None]
    if b <= t or r <= l:
        return None
    return t, l, b, r


def _marker_rect(row: Dict[str, str]) -> Optional[Tuple[int, int, int, int]]:
    return _rect_from(row, "marker_")


def _echo_rect(row: Dict[str, str]) -> Optional[Tuple[int, int, int, int]]:
    return _rect_from(row, "echo_rect_")


def _orientation_group_old(row: Dict[str, str]) -> str:
    group = str(row.get("quadrant_group", "") or "").strip()
    if group:
        return group
    side = str(row.get("detected_marker_side", "") or "").strip().lower()
    vertical = str(row.get("su_giu_pred", "") or "").strip().lower()
    if side == "left" and vertical == "su":
        return "NF"
    if side == "right" and vertical == "su":
        return "LR"
    if side == "left" and vertical == "giu":
        return "UD"
    if side == "right" and vertical == "giu":
        return "LRUD"
    lr_label = str(row.get("lr_label", "") or "").strip()
    return lr_label


def _orientation_group_new(row: Dict[str, str]) -> str:
    return str(
        row.get("orientation_group_from_official_rect_axes", "")
        or row.get("orientation_group", "")
        or ""
    ).strip()


def _score(row: Dict[str, str]) -> Optional[float]:
    return _to_float(row.get("match_score"), None)


def _status(row: Dict[str, str]) -> str:
    return str(row.get("status", "") or "").strip()


def _row_payload(row: Optional[Dict[str, str]], method: str) -> Dict[str, object]:
    if row is None:
        return {"present": False}
    group = _orientation_group_old(row) if method == "old" else _orientation_group_new(row)
    return {
        "present": True,
        "status": _status(row),
        "review_reason": str(row.get("review_reason", "") or ""),
        "group": group,
        "score": _score(row),
        "side": str(row.get("detected_marker_side_by_official_rect_axis", "") or row.get("detected_marker_side", "") or ""),
        "vertical": str(row.get("vertical_by_official_rect_axis", "") or row.get("su_giu_pred", "") or row.get("vertical_final", "") or ""),
        "search": str(row.get("search_strategy", "") or row.get("search_scope", "") or ""),
        "template": str(row.get("template_name", "") or Path(str(row.get("template_path", "") or "")).name),
        "template_path": str(row.get("template_path", "") or ""),
        "echo": _echo_rect(row),
        "marker": _marker_rect(row),
    }


def _load_boxes(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    out: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in rows:
        folder = str(row.get("folder_name", "") or "").strip()
        group = str(row.get("orientation_group", "") or "").strip()
        vals = [_to_int(row.get(k), None) for k in ("top", "left", "bottom", "right")]
        if not folder or not group or any(v is None for v in vals):
            continue
        t, l, b, r = [int(v) for v in vals if v is not None]
        out.setdefault(folder, {}).setdefault(group, []).append(
            {
                "top": t,
                "left": l,
                "bottom": b,
                "right": r,
                "boxes": _to_int(row.get("boxes"), 0) or 0,
                "score": _to_float(row.get("median_score"), None),
            }
        )
    return out


def _center(rect: object) -> Optional[Tuple[float, float]]:
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        t, l, b, r = [float(v) for v in rect]
    except Exception:
        return None
    return (l + r) / 2.0, (t + b) / 2.0


def _marker_center_delta(old_payload: Dict[str, object], new_payload: Dict[str, object]) -> Optional[float]:
    old_c = _center(old_payload.get("marker"))
    new_c = _center(new_payload.get("marker"))
    if old_c is None or new_c is None:
        return None
    dx = new_c[0] - old_c[0]
    dy = new_c[1] - old_c[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _merge_rows(old_rows: Sequence[Dict[str, str]], new_rows: Sequence[Dict[str, str]], box_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    old_by_key = {_key(r): r for r in old_rows}
    new_by_key = {_key(r): r for r in new_rows}
    keys = sorted(set(old_by_key) | set(new_by_key))
    boxes = _load_boxes(box_rows)
    items: List[Dict[str, object]] = []
    for idx, key in enumerate(keys):
        old = old_by_key.get(key)
        new = new_by_key.get(key)
        src = new or old or {}
        image_path = str(src.get("image_path", "") or "")
        folder_name = str(src.get("folder_name", "") or "")
        old_payload = _row_payload(old, "old")
        new_payload = _row_payload(new, "new")
        old_group = str(old_payload.get("group", "") or "")
        new_group = str(new_payload.get("group", "") or "")
        old_score = old_payload.get("score")
        new_score = new_payload.get("score")
        marker_delta = _marker_center_delta(old_payload, new_payload)
        status_delta = str(old_payload.get("status", "") or "") != str(new_payload.get("status", "") or "")
        group_changed = bool(old_group and new_group and old_group != new_group)
        marker_changed = marker_delta is not None and marker_delta > 2.0
        items.append(
            {
                "idx": idx,
                "key": key,
                "image_path": image_path,
                "image_url": _file_url(image_path),
                "folder_name": folder_name,
                "folder_path": str(src.get("folder_path", "") or ""),
                "image_index": str(src.get("image_index", "") or ""),
                "old": old_payload,
                "new": new_payload,
                "boxes": boxes.get(folder_name, {}),
                "same_group": bool(old_group and new_group and old_group == new_group),
                "group_changed": group_changed,
                "status_changed": status_delta,
                "marker_changed": marker_changed,
                "marker_center_delta_px": marker_delta,
                "changed": bool(group_changed or status_delta or marker_changed),
                "group_delta": f"{old_group or '-'} -> {new_group or '-'}",
                "score_delta": (
                    None
                    if not isinstance(old_score, float) or not isinstance(new_score, float)
                    else float(new_score - old_score)
                ),
            }
        )
    return items


def _materialize_image_assets(items: Sequence[Dict[str, object]], output_html: Path, mode: str) -> Dict[str, object]:
    mode = str(mode or "copy").strip().lower()
    if mode == "none":
        return {"mode": "none", "assets": 0, "asset_dir": "", "errors_count": 0, "errors": []}
    if mode not in {"copy", "symlink"}:
        raise ValueError(f"Unsupported asset mode: {mode}")

    output_html = output_html.expanduser().resolve()
    asset_dir = output_html.parent / f"{output_html.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    materialized = 0
    errors: List[Dict[str, str]] = []

    for idx, item in enumerate(items):
        image_path_s = str(item.get("image_path", "") or "").strip()
        if not image_path_s:
            continue
        src = Path(image_path_s).expanduser().resolve()
        if not src.is_file():
            errors.append({"image_path": image_path_s, "error": "missing_source"})
            continue
        suffix = src.suffix.lower() or ".png"
        dest = asset_dir / f"img_{idx:06d}{suffix}"
        try:
            if not dest.exists():
                if mode == "symlink":
                    dest.symlink_to(src)
                else:
                    shutil.copy2(src, dest)
            materialized += 1
            item["image_url"] = f"{asset_dir.name}/{dest.name}"
        except Exception as exc:
            errors.append({"image_path": image_path_s, "error": str(exc)})
            item["image_url"] = _file_url(image_path_s)

    return {
        "mode": mode,
        "assets": materialized,
        "asset_dir": asset_dir.as_posix(),
        "errors_count": len(errors),
        "errors": errors[:25],
    }


def _summary(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    both = [x for x in items if x["old"].get("present") and x["new"].get("present")]  # type: ignore[index]
    changed = [x for x in both if x.get("changed")]
    marker_changed = [x for x in both if x.get("marker_changed")]
    only_old = [x for x in items if x["old"].get("present") and not x["new"].get("present")]  # type: ignore[index]
    only_new = [x for x in items if x["new"].get("present") and not x["old"].get("present")]  # type: ignore[index]
    new_review = [x for x in items if x["new"].get("status") == "review"]  # type: ignore[index]
    old_review = [x for x in items if x["old"].get("status") == "review"]  # type: ignore[index]
    return {
        "items": len(items),
        "both": len(both),
        "changed": len(changed),
        "marker_changed_gt2px": len(marker_changed),
        "only_old": len(only_old),
        "only_new": len(only_new),
        "old_review": len(old_review),
        "new_review": len(new_review),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confronto orientamento: vecchio vs nuovo</title>
<style>
:root {
  --bg: #10161f;
  --panel: #172231;
  --panel-2: #203044;
  --ink: #edf3f8;
  --muted: #9badbf;
  --old: #f97316;
  --new: #22c55e;
  --echo: #facc15;
  --danger: #fb7185;
  --line: rgba(255,255,255,.14);
}
* { box-sizing: border-box; }
body { margin: 0; background: radial-gradient(circle at 20% 0%, #21364f 0, #10161f 38%, #0a0f16 100%); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: rgba(16,22,31,.93); backdrop-filter: blur(14px); border-bottom: 1px solid var(--line); padding: 14px 18px; }
h1 { margin: 0 0 10px; font-size: 20px; letter-spacing: .01em; }
.bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.pill, button, select, input { border: 1px solid var(--line); background: var(--panel-2); color: var(--ink); border-radius: 999px; padding: 8px 12px; }
button { cursor: pointer; }
button:hover { border-color: rgba(255,255,255,.35); }
input { min-width: 260px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: var(--muted); font-size: 13px; }
main { padding: 18px; display: grid; grid-template-columns: minmax(330px, 430px) 1fr; gap: 16px; }
aside { background: rgba(23,34,49,.86); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; height: calc(100vh - 120px); display: flex; flex-direction: column; }
.list { overflow: auto; padding: 10px; display: grid; gap: 8px; }
.row { text-align: left; border: 1px solid var(--line); border-radius: 14px; padding: 10px; background: rgba(255,255,255,.035); color: var(--ink); }
.row.active { outline: 2px solid #38bdf8; background: rgba(56,189,248,.13); }
.row.changed { border-color: rgba(251,113,133,.55); }
.row small { color: var(--muted); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.viewer { background: rgba(23,34,49,.72); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; min-height: calc(100vh - 120px); }
.meta { padding: 14px 16px; border-bottom: 1px solid var(--line); display: grid; gap: 10px; }
.cards { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 10px; }
.card { background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 14px; padding: 10px; }
.card.old { border-left: 4px solid var(--old); }
.card.new { border-left: 4px solid var(--new); }
.card h3 { margin: 0 0 6px; font-size: 15px; }
.kv { display: grid; grid-template-columns: 92px 1fr; gap: 5px 10px; font-size: 13px; }
.kv b { color: var(--muted); font-weight: 600; }
.stage { padding: 14px; }
.canvasWrap { width: 100%; background: #05070a; border-radius: 14px; overflow: auto; border: 1px solid var(--line); }
canvas { display: block; max-width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: var(--muted); font-size: 13px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.help { color: var(--muted); font-size: 12px; }
.empty { color: var(--muted); padding: 28px; text-align: center; }
@media (max-width: 980px) {
  main { grid-template-columns: 1fr; }
  aside { height: 330px; }
  .cards { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Confronto orientamento: vecchio metodo vs nuovo bundle</h1>
  <div class="bar">
    <button id="prev">← Precedente</button>
    <button id="next">Successiva →</button>
    <select id="filter">
      <option value="all">Tutte</option>
      <option value="changed">Solo diverse</option>
      <option value="review">Solo review</option>
      <option value="only_new">Solo nuovo</option>
      <option value="only_old">Solo vecchio</option>
    </select>
    <input id="search" placeholder="Cerca folder, path, gruppo...">
    <label class="pill"><input type="checkbox" id="showEcho" checked> rect+assi</label>
    <label class="pill"><input type="checkbox" id="showOld" checked> vecchio</label>
    <label class="pill"><input type="checkbox" id="showNew" checked> nuovo</label>
    <label class="pill"><input type="checkbox" id="showBoxes" checked> 4 box</label>
  </div>
  <div class="stats" id="stats"></div>
</header>
<main>
  <aside>
    <div class="list" id="list"></div>
  </aside>
  <section class="viewer">
    <div class="meta" id="meta"></div>
    <div class="stage">
      <div class="canvasWrap"><canvas id="canvas"></canvas></div>
      <div class="legend">
        <span><i class="sw" style="background:var(--echo)"></i>Rettangolo ecografico + assi mediani</span>
        <span><i class="sw" style="background:var(--old)"></i>Marker vecchio</span>
        <span><i class="sw" style="background:var(--new)"></i>Marker nuovo</span>
        <span><i class="sw" style="background:#38bdf8"></i>Box finali orientamento</span>
      </div>
      <p class="help">Scorciatoie: freccia destra/sinistra per cambiare immagine, `1` rect, `2` vecchio, `3` nuovo, `4` box.</p>
    </div>
  </section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const items = payload.items || [];
let filtered = [];
let active = 0;
const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const img = new Image();

function fmtScore(v) {
  return typeof v === 'number' ? v.toFixed(4) : '-';
}
function rectLabel(r) {
  if (!r) return '-';
  return `${r[0]}, ${r[1]}, ${r[2]}, ${r[3]}`;
}
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function itemText(it) {
  return `${it.folder_name} ${it.image_path} ${it.group_delta} ${it.old?.status} ${it.new?.status}`.toLowerCase();
}
function applyFilter() {
  const mode = $('filter').value;
  const q = $('search').value.trim().toLowerCase();
  filtered = items.filter(it => {
    if (q && !itemText(it).includes(q)) return false;
    const oldP = !!it.old?.present;
    const newP = !!it.new?.present;
    if (mode === 'changed') return oldP && newP && !!it.changed;
    if (mode === 'review') return it.old?.status === 'review' || it.new?.status === 'review';
    if (mode === 'only_new') return newP && !oldP;
    if (mode === 'only_old') return oldP && !newP;
    return true;
  });
  if (active >= filtered.length) active = Math.max(0, filtered.length - 1);
  renderList();
  renderActive();
}
function renderStats() {
  const s = payload.summary || {};
  $('stats').innerHTML = [
    `totale ${s.items ?? items.length}`,
    `entrambi ${s.both ?? '-'}`,
    `diverse ${s.changed ?? '-'}`,
    `marker >2px ${s.marker_changed_gt2px ?? '-'}`,
    `solo vecchio ${s.only_old ?? '-'}`,
    `solo nuovo ${s.only_new ?? '-'}`,
    `review vecchio ${s.old_review ?? '-'}`,
    `review nuovo ${s.new_review ?? '-'}`,
    `visibili ${filtered.length}`,
  ].map(x => `<span class="pill">${esc(x)}</span>`).join('');
}
function renderList() {
  renderStats();
  const list = $('list');
  if (!filtered.length) {
    list.innerHTML = '<div class="empty">Nessuna immagine per questo filtro</div>';
    return;
  }
  list.innerHTML = filtered.map((it, i) => `
    <button class="row ${i === active ? 'active' : ''} ${it.changed ? 'changed' : ''}" data-i="${i}">
      <b>#${it.image_index || it.idx} ${esc(it.group_delta)}</b>
      <small>${esc(it.folder_name || '')}</small>
      <small>${esc((it.image_path || '').split('/').slice(-2).join('/'))}</small>
      <small>old ${esc(it.old?.status || '-')} ${fmtScore(it.old?.score)} | new ${esc(it.new?.status || '-')} ${fmtScore(it.new?.score)} | Δmarker ${typeof it.marker_center_delta_px === 'number' ? it.marker_center_delta_px.toFixed(1) + 'px' : '-'}</small>
    </button>`).join('');
  [...list.querySelectorAll('.row')].forEach(btn => {
    btn.addEventListener('click', () => {
      active = Number(btn.dataset.i || 0);
      renderList();
      renderActive();
    });
  });
}
function card(title, method, cls) {
  if (!method?.present) {
    return `<div class="card ${cls}"><h3>${title}</h3><div class="kv"><b>presente</b><span>no</span></div></div>`;
  }
  return `<div class="card ${cls}">
    <h3>${title}</h3>
    <div class="kv">
      <b>gruppo</b><span>${esc(method.group || '-')}</span>
      <b>status</b><span>${esc(method.status || '-')}</span>
      <b>score</b><span>${fmtScore(method.score)}</span>
      <b>side/vert</b><span>${esc(method.side || '-')} / ${esc(method.vertical || '-')}</span>
      <b>marker</b><span>${esc(rectLabel(method.marker))}</span>
      <b>search</b><span>${esc(method.search || '-')}</span>
      <b>template</b><span title="${esc(method.template_path || '')}">${esc(method.template || '-')}</span>
      <b>review</b><span>${esc(method.review_reason || '-')}</span>
    </div>
  </div>`;
}
function renderActive() {
  if (!filtered.length) {
    $('meta').innerHTML = '<div class="empty">Nessuna immagine selezionata</div>';
    clearCanvas();
    return;
  }
  const it = filtered[active];
  $('meta').innerHTML = `
    <div>
      <b>${esc(it.folder_name || '')}</b>
      <div class="help">${esc(it.image_path || '')}</div>
    </div>
    <div class="cards">
      ${card('Vecchio metodo', it.old, 'old')}
      ${card('Nuovo metodo bundle', it.new, 'new')}
    </div>`;
  loadImage(it);
}
function clearCanvas() {
  canvas.width = 900;
  canvas.height = 520;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#05070a';
  ctx.fillRect(0,0,canvas.width,canvas.height);
}
function drawRect(rect, color, width=3, label='') {
  if (!rect) return;
  const [t,l,b,r] = rect.map(Number);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(l, t, r-l, b-t);
  if (label) {
    ctx.font = '18px sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(label, l + 4, Math.max(20, t - 6));
  }
  ctx.restore();
}
function drawAxes(rect) {
  if (!rect) return;
  const [t,l,b,r] = rect.map(Number);
  const mx = (l+r)/2, my = (t+b)/2;
  ctx.save();
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--echo').trim();
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath(); ctx.moveTo(mx,t); ctx.lineTo(mx,b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(l,my); ctx.lineTo(r,my); ctx.stroke();
  ctx.restore();
}
function drawBoxes(boxes) {
  const colors = {NF:'#38bdf8', LR:'#fb7185', UD:'#60a5fa', LRUD:'#c084fc'};
  Object.entries(boxes || {}).forEach(([group, arr]) => {
    (arr || []).forEach(box => drawRect([box.top, box.left, box.bottom, box.right], colors[group] || '#38bdf8', 3, `${group} box`));
  });
}
function loadImage(it) {
  clearCanvas();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    drawOverlay(it);
  };
  img.onerror = () => {
    clearCanvas();
    ctx.fillStyle = '#edf3f8';
    ctx.font = '18px sans-serif';
    ctx.fillText('Immagine non caricata', 30, 50);
    ctx.fillStyle = '#9badbf';
    ctx.fillText(it.image_path || '', 30, 82);
  };
  img.src = it.image_url || '';
}
function drawOverlay(it) {
  const echo = it.new?.echo || it.old?.echo;
  if ($('showEcho').checked) {
    drawRect(echo, getComputedStyle(document.documentElement).getPropertyValue('--echo').trim(), 3, 'rect');
    drawAxes(echo);
  }
  if ($('showBoxes').checked) drawBoxes(it.boxes);
  if ($('showOld').checked) drawRect(it.old?.marker, getComputedStyle(document.documentElement).getPropertyValue('--old').trim(), 5, 'old');
  if ($('showNew').checked) drawRect(it.new?.marker, getComputedStyle(document.documentElement).getPropertyValue('--new').trim(), 5, 'new');
}
function step(delta) {
  if (!filtered.length) return;
  active = (active + delta + filtered.length) % filtered.length;
  renderList();
  renderActive();
}
['filter','search'].forEach(id => $(id).addEventListener('input', applyFilter));
['showEcho','showOld','showNew','showBoxes'].forEach(id => $(id).addEventListener('change', renderActive));
$('prev').addEventListener('click', () => step(-1));
$('next').addEventListener('click', () => step(1));
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft') step(-1);
  if (ev.key === 'ArrowRight') step(1);
  if (ev.key === '1') { $('showEcho').checked = !$('showEcho').checked; renderActive(); }
  if (ev.key === '2') { $('showOld').checked = !$('showOld').checked; renderActive(); }
  if (ev.key === '3') { $('showNew').checked = !$('showNew').checked; renderActive(); }
  if (ev.key === '4') { $('showBoxes').checked = !$('showBoxes').checked; renderActive(); }
});
applyFilter();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build old-vs-new orientation marker comparison HTML.")
    p.add_argument("--old-csv", type=Path, default=None, help="Current pipeline lr_marker_per_image_predictions.csv.")
    p.add_argument("--new-csv", type=Path, required=True, help="Bundle adapter bundle_marker_per_image_predictions.csv.")
    p.add_argument("--boxes-csv", type=Path, default=None, help="Bundle adapter bundle_orientation_boxes.csv.")
    p.add_argument("--output-html", type=Path, required=True)
    p.add_argument(
        "--asset-mode",
        choices=("copy", "symlink", "none"),
        default="copy",
        help="How to make images loadable from local HTML. copy is most robust for the in-app browser.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    old_rows = _read_csv(args.old_csv)
    new_rows = _read_csv(args.new_csv)
    box_rows = _read_csv(args.boxes_csv)
    items = _merge_rows(old_rows, new_rows, box_rows)
    output = args.output_html.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_summary = _materialize_image_assets(items, output, mode=str(args.asset_mode))
    payload = {
        "old_csv": "" if args.old_csv is None else args.old_csv.expanduser().resolve().as_posix(),
        "new_csv": args.new_csv.expanduser().resolve().as_posix(),
        "boxes_csv": "" if args.boxes_csv is None else args.boxes_csv.expanduser().resolve().as_posix(),
        "assets": assets_summary,
        "summary": _summary(items),
        "items": items,
    }
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    print(json.dumps({"output_html": output.as_posix(), "items": len(items), "summary": payload["summary"], "assets": assets_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
