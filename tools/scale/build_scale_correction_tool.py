"""Build a self-contained HTML tool to review and correct scale detections.

Samples a *varied* set of images (stratified by vendor and detection outcome), runs the
current scale pipeline on each, and renders a single portable HTML page where the user
can drag the ruler column, the zero and the far end, raise flags and leave a comment.
The corrections export to CSV/JSON with coordinates in *original* frame pixels, so they
can be fed back to tune profiles / fix detection / seed a regression set.

Usage (from repo root):
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/build_scale_correction_tool.py \
    --predictions artifacts/41_scale_chain_eval/con_rete/predictions.csv \
    --heatmap-models artifacts/40_scale_heatmap_models_clean \
    --n 48 --out artifacts/42_scale_correction_tool/index.html

  # offline self-test, no SSD, synthetic rulers:
  python tools/scale/build_scale_correction_tool.py --demo --n 6 \
    --out artifacts/42_scale_correction_tool/demo.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scale.detect_scale_ladder import detect_scale, profile_for, load_image  # noqa: E402


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _bucket(row: Dict[str, str]) -> str:
    """Stratification cell: outcome (if known) crossed with fusion vs single-panel."""
    src = (row.get("source") or "").strip() or "?"
    fus = "fusion" if _truthy(row.get("flag_fusion")) else "single"
    return f"{src}|{fus}"


def stratified_sample(rows: List[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    """Balance the sample across (vendor, outcome) cells, hardest cells first."""
    rng = random.Random(seed)
    cells: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        cells[(r.get("vendor", "?"), _bucket(r))].append(r)
    for c in cells.values():
        rng.shuffle(c)
    # round-robin across cells so every vendor/outcome gets represented
    order = sorted(cells.keys(), key=lambda k: (-len(cells[k]), k))
    picked: List[Dict[str, str]] = []
    i = 0
    while len(picked) < n and any(cells[k] for k in order):
        k = order[i % len(order)]
        if cells[k]:
            picked.append(cells[k].pop())
        i += 1
    return picked[:n]


# --------------------------------------------------------------------------- #
# per-image overlay + crop
# --------------------------------------------------------------------------- #
def _rect_from_row(row: Dict[str, str]) -> Optional[Tuple[int, int, int, int]]:
    try:
        x1, y1, x2, y2 = (float(row[k]) for k in ("rect_x1", "rect_y1", "rect_x2", "rect_y2"))
        if x2 > x1 and y2 > y1:
            return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _encode_crop(gray_or_bgr: np.ndarray, cap_w: int = 1000, quality: int = 72) -> str:
    img = gray_or_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if w > cap_w:
        img = cv2.resize(img, (cap_w, int(round(h * cap_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_item(
    row: Dict[str, str],
    gray: np.ndarray,
    registry=None,
) -> Optional[dict]:
    h, w = gray.shape
    vendor = row.get("vendor", "")
    rect = _rect_from_row(row)

    prior_x = None
    if registry is not None:
        try:
            prior = registry.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), vendor)
            if prior is not None and not prior.ambiguous_column:
                prior_x = prior.x
        except Exception:  # noqa: BLE001
            prior_x = None

    pred = detect_scale(gray, rect=rect, vendor=vendor, prior_x=prior_x)

    # far end of the segment implied by the prediction
    px = pred.x
    y_zero = pred.y_zero
    y_far = None
    if pred.ok and pred.mm_per_px and y_zero is not None:
        length_mm = pred.default_length_mm()
        if length_mm:
            y_far = y_zero + pred.direction * length_mm / pred.mm_per_px

    # Show the FULL frame: the overlay must be judged against the whole image, and a
    # wrong-column detection (e.g. a fusion montage where the ruler is on the far side)
    # has to be obvious. A crop centred on the predicted column used to hide the real
    # ruler exactly when the prediction was wrong.
    crop_x0 = 0
    crop_x1 = w
    crop = gray
    crop_w = w

    def fx(x):  # original px -> fraction of crop width
        return None if x is None else max(0.0, min(1.0, (x - crop_x0) / max(1, crop_w)))

    def fy(y):  # original px -> fraction of image height
        return None if y is None else max(0.0, min(1.0, y / max(1, h)))

    # every number the OCR recognised on the scale (not just the calibration inliers),
    # given as [y_fraction, value_cm]; drawn tied to the column handle in JS
    lab_src = pred.labels_all or pred.labels or []
    labels = [[fy(ly), round(lv / 10.0, 1)] for (ly, lv) in lab_src]

    return {
        "sample_id": row.get("sample_id", ""),
        "vendor": vendor,
        "config_folder": row.get("config_folder", ""),
        "depth_index": row.get("depth_index", ""),
        "image_path": row.get("image_path", ""),
        "source": row.get("source", ""),
        "status": pred.status,
        "reason": pred.reason,
        "img_w": w,
        "img_h": h,
        "crop_x0": crop_x0,
        "crop_w": crop_w,
        "img_uri": _encode_crop(crop),
        "pred": {
            "x": fx(px),
            "y_zero": fy(y_zero),
            "y_far": fy(y_far),
            "zero_at": "top" if pred.direction >= 0 else "bottom",
            "mm_per_px": pred.mm_per_px,
            "labels": labels,
            "ok": bool(pred.ok),
        },
    }


# --------------------------------------------------------------------------- #
# demo (synthetic rulers, no SSD needed)
# --------------------------------------------------------------------------- #
def _demo_rows(n: int) -> List[Tuple[Dict[str, str], np.ndarray]]:
    rng = random.Random(0)
    out = []
    for i in range(n):
        h, w = 700, 900
        img = np.full((h, w), 12, np.uint8)
        img[:, : int(w * 0.62)] = rng.randint(30, 90)  # fake anatomy block
        x = int(w * 0.72) + rng.randint(-20, 20)
        pitch = rng.choice([16, 22, 30])
        y0 = rng.randint(60, 120)
        flip = rng.random() < 0.3
        n_ticks = rng.randint(8, 22)
        for k in range(n_ticks):
            y = y0 + k * pitch
            if y > h - 20:
                break
            length = 18 if k % 2 == 0 else 10
            cv2.line(img, (x, y), (x + length, y), 235, 2)
            if k % 2 == 0:
                cv2.putText(img, str(k // 2), (x - 26, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 220, 1)
        if flip:
            img = cv2.flip(img, 0)
        row = {
            "sample_id": f"demo_{i}",
            "vendor": rng.choice(["BK", "Esaote", "Hitachi"]),
            "config_folder": f"Demo_Folder_{i}",
            "depth_index": str(i % 4),
            "image_path": f"<synthetic demo {i}>",
            "source": rng.choice(["detected", "interpolated", "none"]),
        }
        out.append((row, img))
    return out


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def render_html(items: List[dict], title: str) -> str:
    data = json.dumps(items, ensure_ascii=False)
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--ink:#e8e8ea;--mut:#9aa0aa;--line:#2a2f3a;
        --col:#4aa3ff;--zero:#37d67a;--far:#ffb020;--flag:#ff5470;--add:#ff5ec8;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
  header{position:sticky;top:0;z-index:5;background:#12141a;border-bottom:1px solid var(--line);
         padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header b{font-size:15px}
  .stat{color:var(--mut)} .stat b{color:var(--ink)}
  button{background:#232732;color:var(--ink);border:1px solid var(--line);border-radius:7px;
         padding:6px 10px;cursor:pointer;font-size:13px}
  button:hover{border-color:#3a4150}
  button.primary{background:#1f6feb;border-color:#1f6feb}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px;padding:14px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
  .card.done{outline:2px solid var(--zero)}
  .card.flag{outline:2px solid var(--flag)}
  .stage{position:relative;background:#000;user-select:none;touch-action:none}
  .stage img{display:block;width:100%;height:auto;pointer-events:none}
  .stage svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
  .meta{padding:8px 10px;display:flex;flex-direction:column;gap:6px}
  .muted{color:var(--mut);font-size:12px}
  .flags{display:flex;flex-wrap:wrap;gap:5px}
  .flags button{padding:3px 7px;font-size:12px}
  .flags button.on{background:var(--flag);border-color:var(--flag)}
  button.addon{background:var(--add);border-color:var(--add);color:#111}
  .stage.adding{cursor:crosshair}
  textarea{width:100%;background:#12141a;color:var(--ink);border:1px solid var(--line);
           border-radius:6px;padding:6px;resize:vertical;font:13px inherit}
  .rowbtns{display:flex;gap:6px}
  .hint{font-size:11px;color:var(--mut)}
  .handle{cursor:ns-resize} .colline{cursor:ew-resize}
</style></head>
<body>
<header>
  <b>__TITLE__</b>
  <span class="stat">tot <b id="s-tot">0</b></span>
  <span class="stat">riviste <b id="s-done">0</b></span>
  <span class="stat">flag <b id="s-flag">0</b></span>
  <span style="flex:1"></span>
  <span class="hint">trascina&nbsp;<span style="color:var(--col)">colonna</span> /
    <span style="color:var(--zero)">zero</span> /
    <span style="color:var(--far)">estremo</span></span>
  <button id="exp-csv" class="primary">Esporta CSV</button>
  <button id="exp-json">JSON</button>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = "scale_corr_" + (location.pathname.split('/').pop() || 'tool');
let ST = {};
try { ST = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ ST = {}; }
const FLAGS = [["col","colonna errata"],["noruler","no righello"],["flip","verso invertito"],["fusion","fusion sbagliata"]];

function itemState(id){ if(!ST[id]) ST[id]={}; return ST[id]; }
function save(){ localStorage.setItem(KEY, JSON.stringify(ST)); refreshStats(); }
function corr(d){
  const s = itemState(d.sample_id); const p = d.pred;
  if(!s.c){
    // Always give all three handles a position so they can be dragged. Ones the detector
    // did not predict get a default and a "to place" (dashed) style via the *set flags.
    s.c = {
      x:      p.x!=null ? p.x : 0.82,        xset: p.x!=null,
      y_zero: p.y_zero!=null ? p.y_zero : 0.14, zset: p.y_zero!=null,
      y_far:  p.y_far!=null ? p.y_far : (p.y_zero!=null ? Math.min(0.96, p.y_zero+0.4) : 0.82), fset: p.y_far!=null,
      zero_at: p.zero_at || 'top'
    };
  } else if(s.c.xset===undefined){
    // migrate annotations saved before the *set flags existed
    s.c.xset = s.c.x!=null; s.c.zset = s.c.y_zero!=null; s.c.fset = s.c.y_far!=null;
  }
  return s.c;
}
function toOrigX(d,fx){ return Math.round(d.crop_x0 + fx*d.crop_w); }
function toOrigY(d,fy){ return Math.round(fy*d.img_h); }

function drawOverlay(svg,d){
  const c = corr(d);
  const st = itemState(d.sample_id);
  const X = (c.x!=null?c.x:0.5)*100;
  const yz = c.y_zero!=null?c.y_zero*100:null;
  const yf = c.y_far!=null?c.y_far*100:null;
  let s = "";
  // everything sits ON the ruler column so it marks the exact detected point.
  // Handles not predicted (xset/zset/fset false) are dashed = "da posizionare".
  s += `<line class="colline" data-h="col" x1="${X}%" y1="0" x2="${X}%" y2="100%" stroke="var(--col)" stroke-width="1" stroke-dasharray="${c.xset?'2 7':'1 4'}" opacity="${c.xset?0.5:0.35}"/>`;
  if(yz!=null && yf!=null)
    s += `<line x1="${X}%" y1="${yz}%" x2="${X}%" y2="${yf}%" stroke="var(--far)" stroke-width="2" opacity="${(c.zset&&c.fset)?0.55:0.3}"/>`;
  // every recognised number on the scale, tied to the column
  (d.pred.labels||[]).forEach(l=>{ if(l[0]!=null&&l[1]!=null)
    s += `<text x="${X}%" y="${l[0]*100}%" dx="-14" dy="3" fill="var(--zero)" font-size="10" opacity="0.8" text-anchor="end">${l[1]}</text>`; });
  if(yz!=null) s += `<circle class="handle" data-h="zero" cx="${X}%" cy="${yz}%" r="5" fill="var(--zero)" fill-opacity="${c.zset?0.4:0.12}" stroke="var(--zero)" stroke-width="1.5" stroke-dasharray="${c.zset?'':'2 2'}"/>`;
  if(yf!=null) s += `<rect class="handle" data-h="far" x="${X}%" y="${yf}%" width="9" height="9" transform="translate(-4.5,-4.5)" fill="var(--far)" fill-opacity="${c.fset?0.4:0.12}" stroke="var(--far)" stroke-width="1.5" stroke-dasharray="${c.fset?'':'2 2'}"/>`;
  // numbers the user added by hand (OCR missed them): click to delete
  (st.added||[]).forEach((a,i)=>{
    s += `<g data-add="${i}" style="cursor:pointer">`
       + `<circle cx="${X}%" cy="${a.fy*100}%" r="4" fill="var(--add)" fill-opacity="0.55" stroke="var(--add)" stroke-width="1.5"/>`
       + `<text x="${X}%" y="${a.fy*100}%" dx="10" dy="3" fill="var(--add)" font-size="11" font-weight="bold">${a.v}</text></g>`;
  });
  svg.innerHTML = s;
}

const ADDMODE = {};
function attachDrag(stage,svg,d){
  let cur=null;
  const rect=()=>stage.getBoundingClientRect();
  function pos(e){ const r=rect(); return {fx:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
                                          fy:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))}; }
  svg.addEventListener('pointerdown',e=>{
    if(ADDMODE[d.sample_id]){
      const st=itemState(d.sample_id);
      const g=e.target.closest('[data-add]');
      if(g){ st.added.splice(+g.getAttribute('data-add'),1); }  // click a magenta mark to delete
      else { const v=prompt('Valore in cm a questa altezza (es. 2 o 2.5):');
             if(v!==null && v.trim()!==''){ st.added=st.added||[]; st.added.push({fy:pos(e).fy, v:v.trim()}); } }
      st.reviewed=true; save(); drawOverlay(svg,d); e.preventDefault(); return;
    }
    const h=e.target.getAttribute('data-h'); if(!h) return;
    cur=h; svg.setPointerCapture(e.pointerId); e.preventDefault();
  });
  svg.addEventListener('pointermove',e=>{
    if(!cur) return; const p=pos(e); const c=corr(d);
    if(cur==='col'){ c.x=p.fx; c.xset=true; }
    else if(cur==='zero'){ c.y_zero=p.fy; c.zset=true; }
    else if(cur==='far'){ c.y_far=p.fy; c.fset=true; }
    if(c.y_zero!=null&&c.y_far!=null) c.zero_at = c.y_zero<=c.y_far?'top':'bottom';
    drawOverlay(svg,d);
  });
  svg.addEventListener('pointerup',e=>{ if(cur){ cur=null; itemState(d.sample_id).reviewed=true; save(); } });
}

function card(d){
  const el=document.createElement('article'); el.className='card'; el.id='card-'+d.sample_id;
  el.innerHTML = `
    <div class="stage" id="stage-${d.sample_id}">
      <img src="${d.img_uri}" alt="">
      <svg id="svg-${d.sample_id}"></svg>
    </div>
    <div class="meta">
      <div><b>${d.vendor}</b> · <span class="muted">${d.config_folder} · d${d.depth_index} · src:${d.source} · ${d.status}</span></div>
      <div class="flags" id="flags-${d.sample_id}">
        ${FLAGS.map(f=>`<button data-f="${f[0]}">${f[1]}</button>`).join('')}
      </div>
      <textarea id="note-${d.sample_id}" rows="2" placeholder="commento..."></textarea>
      <div class="rowbtns">
        <button data-act="addnum">+ numero</button>
        <button data-act="ok">va bene</button>
        <button data-act="reset">reset</button>
      </div>
      <div class="hint">+ numero: clicca sulla scala dove c'è un numero non letto e digita il valore (clicca un mark magenta per cancellarlo)</div>
    </div>`;
  return el;
}

function wire(d){
  const stage=document.getElementById('stage-'+d.sample_id);
  const svg=document.getElementById('svg-'+d.sample_id);
  // no viewBox: SVG user units == displayed px, so "%" positions map to the shown image
  // and handle radii/sizes stay fixed on screen.
  drawOverlay(svg,d); attachDrag(stage,svg,d);
  const s=itemState(d.sample_id);
  const fb=document.getElementById('flags-'+d.sample_id);
  fb.querySelectorAll('button').forEach(b=>{
    const f=b.getAttribute('data-f'); if(s.flags&&s.flags[f]) b.classList.add('on');
    b.onclick=()=>{ s.flags=s.flags||{}; s.flags[f]=!s.flags[f]; b.classList.toggle('on'); s.reviewed=true; save(); paint(d); };
  });
  const ta=document.getElementById('note-'+d.sample_id); ta.value=s.note||'';
  ta.oninput=()=>{ s.note=ta.value; save(); };
  const el=document.getElementById('card-'+d.sample_id);
  el.querySelector('[data-act="ok"]').onclick=()=>{ s.reviewed=true; s.ok=true; save(); paint(d); };
  el.querySelector('[data-act="reset"]').onclick=()=>{ s.c=null; s.ok=false; s.added=[]; drawOverlay(svg,d); save(); paint(d); };
  const addBtn=el.querySelector('[data-act="addnum"]');
  addBtn.onclick=()=>{ ADDMODE[d.sample_id]=!ADDMODE[d.sample_id];
    addBtn.classList.toggle('addon',ADDMODE[d.sample_id]); stage.classList.toggle('adding',ADDMODE[d.sample_id]); };
  paint(d);
}
function paint(d){
  const el=document.getElementById('card-'+d.sample_id); const s=itemState(d.sample_id);
  const anyFlag=s.flags&&Object.values(s.flags).some(Boolean);
  el.classList.toggle('flag',!!anyFlag);
  el.classList.toggle('done',!!s.reviewed&&!anyFlag);
}
function refreshStats(){
  let done=0,flag=0;
  DATA.forEach(d=>{ const s=ST[d.sample_id]||{}; if(s.reviewed) done++;
    if(s.flags&&Object.values(s.flags).some(Boolean)) flag++; });
  document.getElementById('s-tot').textContent=DATA.length;
  document.getElementById('s-done').textContent=done;
  document.getElementById('s-flag').textContent=flag;
}
function rows(){
  const out=[["sample_id","vendor","config_folder","depth_index","image_path",
    "pred_x","pred_y_zero","pred_y_far","corr_x","corr_y_zero","corr_y_far","zero_at",
    "x_set","zero_set","far_set","added_labels","flags","reviewed","ok","comment"]];
  DATA.forEach(d=>{ const s=ST[d.sample_id]||{}; const c=s.c||{};
    const flags=s.flags?Object.keys(s.flags).filter(k=>s.flags[k]).join("|"):"";
    const added=(s.added||[]).map(a=>toOrigY(d,a.fy)+":"+a.v).join("|");  // y_px:cm pairs
    if(!s.reviewed && !flags && !(s.note) && !added) return; // export only touched rows
    out.push([d.sample_id,d.vendor,d.config_folder,d.depth_index,d.image_path,
      d.pred.x!=null?toOrigX(d,d.pred.x):"", d.pred.y_zero!=null?toOrigY(d,d.pred.y_zero):"", d.pred.y_far!=null?toOrigY(d,d.pred.y_far):"",
      c.x!=null?toOrigX(d,c.x):"", c.y_zero!=null?toOrigY(d,c.y_zero):"", c.y_far!=null?toOrigY(d,c.y_far):"",
      c.zero_at||d.pred.zero_at||"", c.xset?1:0, c.zset?1:0, c.fset?1:0,
      added, flags, s.reviewed?1:0, s.ok?1:0, (s.note||"").replace(/\n/g," ")]);
  });
  return out;
}
function dl(name,blob){ const u=URL.createObjectURL(blob); const a=document.createElement('a');
  a.href=u; a.download=name; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(u); }
document.getElementById('exp-csv').onclick=()=>{
  const csv=rows().map(r=>r.map(v=>`"${String(v??'').replace(/"/g,'""')}"`).join(',')).join('\n');
  dl('scale_corrections.csv', new Blob([csv],{type:'text/csv;charset=utf-8;'})); };
document.getElementById('exp-json').onclick=()=>{
  dl('scale_corrections.json', new Blob([JSON.stringify(ST,null,2)],{type:'application/json'})); };

const grid=document.getElementById('grid');
DATA.forEach(d=>{ grid.appendChild(card(d)); });
DATA.forEach(d=>wire(d));
refreshStats();
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _remap(path: str, remaps: List[str]) -> str:
    for rm in remaps:
        a, _, b = rm.partition("=")
        if a and path.startswith(a):
            return b + path[len(a):]
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, help="predictions.csv to sample from (has image_path, vendor, source).")
    ap.add_argument("--gt", type=Path, help="Alternative source: an audit GT CSV.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heatmap-models", type=Path, default=None)
    ap.add_argument("--heatmap-device", type=str, default="auto")
    ap.add_argument("--path-remap", action="append", default=[], help="orig=replacement prefix remap.")
    ap.add_argument("--vendors", type=str, default="", help="Comma list to keep (e.g. BK,Esaote,Hitachi,GE,Mindray,Canon).")
    ap.add_argument("--title", type=str, default="Correzione scala")
    ap.add_argument("--demo", action="store_true", help="Synthetic rulers, no images needed.")
    args = ap.parse_args(argv)

    # This is a human-review tool: prefer high recall (find more rulers/numbers to show
    # and correct) over the auto-accept precision the default detector optimises for.
    os.environ.setdefault("SCALE_HIGH_RECALL", "1")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    items: List[dict] = []
    if args.demo:
        for row, img in _demo_rows(args.n):
            it = build_item(row, img, registry=None)
            if it:
                items.append(it)
    else:
        src = args.predictions or args.gt
        if src is None:
            ap.error("pass --predictions or --gt (or --demo)")
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        rows = [r for r in rows if r.get("image_path")]
        if args.vendors:
            keep = {v.strip() for v in args.vendors.split(",") if v.strip()}
            rows = [r for r in rows if r.get("vendor") in keep]
        picked = stratified_sample(rows, args.n, args.seed)

        registry = None
        if args.heatmap_models is not None:
            from predict_scale_heatmap import load_registry, describe_registry  # noqa
            registry = load_registry(args.heatmap_models, args.heatmap_device)
            print(f"[heatmap] {describe_registry(registry)}")

        for r in picked:
            p = _remap(r["image_path"], args.path_remap)
            img = load_image(p)
            if img is None:
                print(f"[skip] unreadable {p}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            it = build_item(r, gray, registry=registry)
            if it:
                items.append(it)

    html = render_html(items, args.title)
    args.out.write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"[ok] {len(items)} images -> {args.out}  ({kb:.0f} KB)")
    # brief coverage of the sample: vendor, detection status, fusion share
    byv: Dict[str, int] = defaultdict(int)
    bystatus: Dict[str, int] = defaultdict(int)
    for it in items:
        byv[it["vendor"]] += 1
        bystatus[it["status"]] += 1
    print("[sample vendor] " + ", ".join(f"{k}:{v}" for k, v in sorted(byv.items())))
    print("[sample status] " + ", ".join(f"{k}:{v}" for k, v in sorted(bystatus.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
