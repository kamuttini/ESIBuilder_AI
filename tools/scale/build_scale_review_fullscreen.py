"""Full-screen review tool for the scale block: one frame at a time, arrows to move.

Built for actually reviewing a batch: the frame fills the window, the arrows walk through the
folder, the overlay can be switched off to see the untouched image, the three handles can be
dragged to correct the geometry, and flags + comments export to CSV so the corrections come
back into the pipeline (``ingest_scale_corrections.py``).

Keyboard
  left / right   previous / next frame in this folder
  up / down      previous / next folder
  h              hide/show the overlay
  0              reset zoom
  1..5           toggle the five flags
  esc            leave the comment field

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/build_scale_review_fullscreen.py \
    --folders-file /tmp/vendor_all.txt --root /Volumes/SSD_esi1_n1 \
    --out artifacts/48_scale_review/index.html
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from scale.consolidate_scale_setup import ScaleCandidate, _theil_sen_log, consolidate_setup  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", REPO / "tools/scale/build_scale_folder_report.py")
rep = importlib.util.module_from_spec(_spec)
sys.modules["rep"] = rep
_spec.loader.exec_module(rep)


def collect_folder(
    folder: str,
    vendor_hint: str,
    max_images: int,
    cap_w: int,
    pattern: str = "image_depth_value_setup_*.png",
    use_consensus: bool = True,
) -> Optional[dict]:
    base = os.path.join(folder, "image_samples")
    if not os.path.isdir(base):
        base = folder  # a plain directory of images works too
    paths = sorted(glob.glob(os.path.join(base, pattern)))[:max_images]
    if not paths:
        return None
    vendor = vendor_hint or rep.guess_vendor(folder)

    cases: List[dict] = []
    for p in paths:
        c = rep.process_image(p, vendor)
        if c:
            c["path"] = p
            # Material outside the configured corpus has no setup_N in the name; fall back to
            # the trailing number, then to position, purely so the frames have a stable order.
            if c["depth_index"] < 0:
                m = re.search(r"(\d+)\.png$", os.path.basename(p), re.I)
                c["depth_index"] = int(m.group(1)) if m else len(cases)
            cases.append(c)
    if not cases:
        return None

    rep.enforce_direction_groups(cases)
    reliable = [
        c for c in cases
        if c["pred"].mm_per_px and c["pred"].tick_pitch_px and c["depth_index"] >= 0
        and c["pred"].status == "accepted"
        and rep._step_fit(c["pred"].mm_per_px * c["pred"].tick_pitch_px)[1] <= 0.08
    ]
    trend = _theil_sen_log([(c["depth_index"], float(c["pred"].mm_per_px)) for c in reliable])
    ref = ({c["depth_index"]: float(trend(c["depth_index"])) for c in cases if c["depth_index"] >= 0}
           if trend is not None else {})
    rep.repair_step_incoherence(cases, ref)

    cands = [
        ScaleCandidate(depth_index=c["depth_index"], x=c["pred"].x, y_zero=c["pred"].y_zero,
                       mm_per_px=c["pred"].mm_per_px, direction=c["pred"].direction,
                       confidence=c["pred"].confidence, status=c["pred"].status,
                       n_labels=len(c["pred"].labels or []),
                       weak_anchor=bool(c["pred"].debug.get("calib_source") == "geometry"
                                        or c["pred"].debug.get("out_of_band")))
        for c in cases if c["depth_index"] >= 0
    ]
    cons: Dict[int, object] = {}
    # The consensus assumes mm_per_px moves monotonically with the depth index. On raw
    # material (movie frames, arbitrary order) that assumption is false, so it is skipped
    # rather than producing a confident-looking but meaningless trend.
    if cands and use_consensus:
        for k in consolidate_setup(cands, depth_indices=sorted({c.depth_index for c in cands})):
            cons[k.depth_index] = k

    frames = []
    for c in sorted(cases, key=lambda c: c["depth_index"]):
        p = c["pred"]
        s = min(1.0, cap_w / float(c["w"]))
        k = cons.get(c["depth_index"])
        far = p.y_last_tick if p.direction >= 0 else p.y_first_tick
        # Everything below is in ORIGINAL frame pixels; the page maps them with `scale`.
        frames.append({
            "name": c["name"],
            "depth": c["depth_index"],
            "img": c["img"],
            "w": c["w"], "h": c["h"], "scale": round(s, 6),
            "status": c["status"],
            "reason": c["reason"],
            "x": p.x, "y_zero": p.y_zero, "y_far": far,
            "ticks": [round(t, 1) for t in (p.ticks_y or [])],
            "labels": [[round(y, 1), round(v / 10.0, 2)] for y, v in (p.labels_all or [])],
            "mm_per_px": p.mm_per_px,
            "pitch": p.tick_pitch_px,
            "step_mm": (round(p.mm_per_px * p.tick_pitch_px, 2)
                        if (p.mm_per_px and p.tick_pitch_px) else None),
            "n_ticks": c["n_ticks"],
            "zero_at": c["zero_at"],
            "cons_mm": (round(k.mm_per_px, 6) if (k is not None and getattr(k, "mm_per_px", None))
                        else None),
            "cons_src": (getattr(k, "source", None) if k is not None else None),
            "fixes": c.get("fixes", []),
        })

    st: Dict[str, int] = {}
    for c in cases:
        st[c["status"]] = st.get(c["status"], 0) + 1
    return {"vendor": vendor or "default", "folder": os.path.basename(folder.rstrip("/\\")),
            "status": st, "frames": frames}


PAGE = r"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisione scala</title>
<style>
 :root{--bg:#0b0d11;--panel:#14171e;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
       --good:#37d67a;--bad:#ff5470;--warn:#ffb020;--add:#ff5ec8}
 *{box-sizing:border-box}
 html,body{height:100%;margin:0}
 body{background:var(--bg);color:var(--ink);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
      display:flex;flex-direction:column;overflow:hidden}
 .top{display:flex;gap:10px;align-items:center;padding:7px 12px;background:var(--panel);
      border-bottom:1px solid var(--line);flex-wrap:wrap;flex:0 0 auto}
 .top b{font-size:14px}
 select,button,input{background:#232732;color:var(--ink);border:1px solid var(--line);
                     border-radius:6px;padding:4px 9px;font-size:12px}
 button{cursor:pointer}
 button.on{background:var(--bad);border-color:var(--bad);color:#fff}
 button.tgl.on{background:var(--good);border-color:var(--good);color:#0b0d11}
 .muted{color:var(--mut)}
 .pill{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);
       font-size:11px;color:var(--mut)}
 .pill.good{border-color:var(--good);color:var(--good)}
 .pill.warn{border-color:var(--warn);color:var(--warn)}
 .pill.bad{border-color:var(--bad);color:var(--bad)}
 #stage{flex:1 1 auto;position:relative;overflow:hidden;background:#000;cursor:grab}
 #stage.dragging{cursor:grabbing}
 #wrap{position:absolute;left:0;top:0;transform-origin:0 0}
 #img{display:block;max-width:none}
 #ovl{position:absolute;inset:0;width:100%;height:100%}
 body.hideovl #ovl{display:none}
 .bottom{display:flex;gap:8px;align-items:center;padding:7px 12px;background:var(--panel);
         border-top:1px solid var(--line);flex-wrap:wrap;flex:0 0 auto}
 .bottom input.note{flex:1 1 320px;min-width:220px}
 .hint{font-size:11px;color:var(--mut)}
 .handle{cursor:ns-resize} .colline{cursor:ew-resize}
 .k{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;
    margin-right:3px}
</style></head><body>
<div class="top">
  <select id="fsel"></select>
  <button id="pf" title="cartella precedente (freccia su)">▲</button>
  <button id="nf" title="cartella successiva (freccia giu)">▼</button>
  <b id="counter">–</b>
  <button id="pv" title="frame precedente (freccia sinistra)">◀</button>
  <button id="nx" title="frame successivo (freccia destra)">▶</button>
  <span id="meta" class="muted"></span>
  <span style="flex:1"></span>
  <button id="tgl" class="tgl on" title="h">elaborazione: ON</button>
  <button id="rz" title="0">reset zoom</button>
  <button id="exp">Esporta CSV</button>
  <span class="muted">commentati <b id="cnt">0</b></span>
</div>
<div id="stage"><div id="wrap"><img id="img" alt=""><svg id="ovl"></svg></div></div>
<div class="bottom">
  <span class="hint"><i class="k" style="background:#ffd24a"></i>tacche
    <i class="k" style="background:#37d67a"></i>zero
    <i class="k" style="background:#a000c8"></i>ultima
    <i class="k" style="background:#ff5000"></i>letti
    <i class="k" style="background:#00b4ff"></i>colonna</span>
  <span id="flags"></span>
  <input id="note" class="note" type="text" placeholder="commento su questo frame...">
  <span class="hint">← → frame · ↑ ↓ cartella · h overlay · trascina le maniglie per correggere</span>
</div>
<script>
const DATA = __DATA__;
const FLAGS = [["col","colonna"],["zero","zero"],["ultima","ultima tacca"],
               ["valori","valori"],["noruler","no righello"]];
const SKEY = 'scale_review_' + (location.pathname.split('/').pop() || 'x');
let ST = {}; try { ST = JSON.parse(localStorage.getItem(SKEY)||'{}'); } catch(e){ ST={}; }
let fi = 0, ii = 0, z = 1, tx = 0, ty = 0;

const $ = id => document.getElementById(id);
const cur = () => DATA[fi].frames[ii];
const key = () => DATA[fi].vendor+'|'+DATA[fi].folder+'|'+cur().depth+'|'+cur().name;
function stGet(){ const k=key(); if(!ST[k]) ST[k]={flags:{},note:'',corr:null}; return ST[k]; }
function save(){ localStorage.setItem(SKEY, JSON.stringify(ST)); count(); }
function count(){
  let n=0; for(const k in ST){ const s=ST[k];
    if((s.note&&s.note.trim())||Object.values(s.flags||{}).some(Boolean)||s.corr) n++; }
  $('cnt').textContent=n;
}

// ---- overlay, in original-frame pixels scaled by frame.scale ----
function drawOvl(){
  const f = cur(), s = f.scale, st = stGet();
  const c = st.corr || {};
  const X  = (c.x ?? f.x), YZ = (c.y_zero ?? f.y_zero), YF = (c.y_far ?? f.y_far);
  const W = Math.round(f.w*s), H = Math.round(f.h*s);
  const ovl = $('ovl');
  ovl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  ovl.setAttribute('preserveAspectRatio','none');
  if (X == null){ ovl.innerHTML=''; return; }
  const px = X*s;
  let o = `<line class="colline" data-h="col" x1="${px}" y1="0" x2="${px}" y2="${H}"
            stroke="#00b4ff" stroke-width="1" stroke-dasharray="4 6" opacity="0.6"/>`;
  const many = f.ticks.length > 18;
  f.ticks.forEach((t,i)=>{
    const y=t*s;
    o += `<line x1="${px-9}" y1="${y}" x2="${px+9}" y2="${y}" stroke="#ffd24a"
            stroke-width="1.3" opacity="0.95"/>`;
    if (f.mm_per_px && YZ!=null && (!many || i%2===0)){
      const v = (f.zero_at==='alto'?1:-1)*(t-YZ)*f.mm_per_px/10;
      o += `<text x="${px+12}" y="${y+4}" fill="#ffd24a" font-size="11">${v.toFixed(1)}</text>`;
    }
  });
  f.labels.forEach(([y,v])=>{
    o += `<circle cx="${px}" cy="${y*s}" r="5" fill="none" stroke="#ff5000" stroke-width="2"/>`
       + `<text x="${px-14}" y="${y*s+4}" fill="#ff8c46" font-size="11" text-anchor="end">${v}</text>`;
  });
  if (YZ!=null) o += `<circle class="handle" data-h="zero" cx="${px}" cy="${YZ*s}" r="8"
      fill="#37d67a" fill-opacity="0.35" stroke="#37d67a" stroke-width="2.5"/>`
      + `<text x="${px+14}" y="${YZ*s-9}" fill="#37d67a" font-size="12" font-weight="bold">0</text>`;
  if (YF!=null) o += `<rect class="handle" data-h="far" x="${px}" y="${YF*s}" width="13" height="13"
      transform="translate(-6.5,-6.5)" fill="#a000c8" fill-opacity="0.75" stroke="#fff"
      stroke-width="1"/>`;
  ovl.innerHTML = o;
}

function layout(){
  const f = cur(), s = f.scale;
  const W = Math.round(f.w*s), H = Math.round(f.h*s);
  const img = $('img'); img.src = f.img; img.width = W; img.height = H;
  $('wrap').style.width = W+'px'; $('wrap').style.height = H+'px';
  // fit the frame in the stage on load
  const r = $('stage').getBoundingClientRect();
  z = Math.min(r.width/W, r.height/H); tx = (r.width - W*z)/2; ty = (r.height - H*z)/2;
  applyT();
}
function applyT(){ $('wrap').style.transform = `translate(${tx}px,${ty}px) scale(${z})`; }

function show(){
  const f = cur(), d = DATA[fi], st = stGet();
  $('counter').textContent = `${ii+1}/${d.frames.length}`;
  const cls = {accepted:'good',review:'warn',reject:'bad'}[f.status]||'';
  $('meta').innerHTML =
    `<span class="pill ${cls}">${f.status}</span> depth <b>${f.depth}</b> `
    + `<span class="pill">mm/px ${f.mm_per_px?f.mm_per_px.toFixed(4):'n/d'}</span>`
    + `<span class="pill">consenso ${f.cons_mm?f.cons_mm.toFixed(4):'n/d'}${f.cons_src?' · '+f.cons_src:''}</span>`
    + `<span class="pill">tacche ${f.n_ticks}</span>`
    + `<span class="pill">passo ${f.step_mm??'n/d'} mm</span>`
    + `<span class="pill">zero in ${f.zero_at}</span>`
    + (st.corr?' <span class="pill warn">corretto a mano</span>':'')
    + (f.fixes.length?` <span class="pill warn">${f.fixes.length} correzioni auto</span>`:'');
  $('note').value = st.note || '';
  [...$('flags').children].forEach(b=>b.classList.toggle('on', !!(st.flags||{})[b.dataset.f]));
  layout(); drawOvl();
}

// ---- navigation ----
function goFrame(d){ const n=DATA[fi].frames.length; ii=(ii+d+n)%n; show(); }
function goFolder(d){ fi=(fi+d+DATA.length)%DATA.length; ii=0; $('fsel').value=String(fi); show(); }

DATA.forEach((d,i)=>{
  const o=document.createElement('option'); o.value=String(i);
  const acc=d.status.accepted||0;
  o.textContent=`${d.vendor} — ${d.folder}  (${d.frames.length} depth, ${acc} accepted)`;
  $('fsel').appendChild(o);
});
$('fsel').addEventListener('change',e=>{ fi=+e.target.value; ii=0; show(); });
$('pv').onclick=()=>goFrame(-1); $('nx').onclick=()=>goFrame(1);
$('pf').onclick=()=>goFolder(-1); $('nf').onclick=()=>goFolder(1);
$('rz').onclick=()=>{ layout(); };
$('tgl').onclick=()=>{
  document.body.classList.toggle('hideovl');
  const on=!document.body.classList.contains('hideovl');
  $('tgl').classList.toggle('on',on); $('tgl').textContent='elaborazione: '+(on?'ON':'OFF');
};

FLAGS.forEach(([f,lab],i)=>{
  const b=document.createElement('button'); b.dataset.f=f; b.textContent=`${i+1} ${lab}`;
  b.onclick=()=>{ const st=stGet(); st.flags=st.flags||{}; st.flags[f]=!st.flags[f];
    b.classList.toggle('on'); save(); show(); };
  $('flags').appendChild(b);
});
$('note').addEventListener('input',()=>{ stGet().note=$('note').value; save(); });

document.addEventListener('keydown',e=>{
  if(e.target===$('note')){ if(e.key==='Escape') $('note').blur(); return; }
  if(e.key==='ArrowRight'){ goFrame(1); e.preventDefault(); }
  else if(e.key==='ArrowLeft'){ goFrame(-1); e.preventDefault(); }
  else if(e.key==='ArrowDown'){ goFolder(1); e.preventDefault(); }
  else if(e.key==='ArrowUp'){ goFolder(-1); e.preventDefault(); }
  else if(e.key==='h'||e.key==='H'){ $('tgl').click(); }
  else if(e.key==='0'){ layout(); }
  else if('12345'.includes(e.key)){ const b=$('flags').children[+e.key-1]; if(b) b.click(); }
});

// ---- zoom, pan, and dragging the handles ----
const stage=$('stage');
let drag=null, sx=0, sy=0;
stage.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=stage.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
  const f=e.deltaY<0?1.15:1/1.15, nz=Math.max(0.05,Math.min(20,z*f));
  tx=cx-(cx-tx)*(nz/z); ty=cy-(cy-ty)*(nz/z); z=nz; applyT();
},{passive:false});
stage.addEventListener('pointerdown',e=>{
  const h=e.target.getAttribute && e.target.getAttribute('data-h');
  if(h){ drag={mode:'handle',h}; }
  else { drag={mode:'pan'}; sx=e.clientX-tx; sy=e.clientY-ty; stage.classList.add('dragging'); }
  stage.setPointerCapture(e.pointerId); e.preventDefault();
});
stage.addEventListener('pointermove',e=>{
  if(!drag) return;
  if(drag.mode==='pan'){ tx=e.clientX-sx; ty=e.clientY-sy; applyT(); return; }
  // handle: screen -> original frame pixels
  const f=cur(), r=$('wrap').getBoundingClientRect();
  const ox=(e.clientX-r.left)/z/f.scale, oy=(e.clientY-r.top)/z/f.scale;
  const st=stGet();
  st.corr = st.corr || {x:f.x, y_zero:f.y_zero, y_far:f.y_far};
  if(drag.h==='col') st.corr.x=Math.max(0,Math.min(f.w,ox));
  else if(drag.h==='zero') st.corr.y_zero=Math.max(0,Math.min(f.h,oy));
  else if(drag.h==='far') st.corr.y_far=Math.max(0,Math.min(f.h,oy));
  drawOvl();
});
stage.addEventListener('pointerup',()=>{
  if(drag&&drag.mode==='handle'){ save(); show(); }
  drag=null; stage.classList.remove('dragging');
});

// ---- export ----
function esc(v){ return '"'+String(v??'').replace(/"/g,'""')+'"'; }
$('exp').onclick=()=>{
  const out=[["vendor","cartella","depth","immagine","stato","pred_x","pred_y_zero","pred_y_far",
              "corr_x","corr_y_zero","corr_y_far","mm_per_px","mm_per_px_consenso",
              "flag","commento"]];
  DATA.forEach(d=>d.frames.forEach(f=>{
    const k=d.vendor+'|'+d.folder+'|'+f.depth+'|'+f.name, s=ST[k];
    if(!s) return;
    const fl=Object.keys(s.flags||{}).filter(x=>s.flags[x]).join('|');
    const note=(s.note||'').trim(), c=s.corr;
    if(!fl && !note && !c) return;
    out.push([d.vendor,d.folder,f.depth,f.name,f.status,
      f.x?.toFixed?.(1)??'', f.y_zero?.toFixed?.(1)??'', f.y_far?.toFixed?.(1)??'',
      c?Math.round(c.x):'', c?Math.round(c.y_zero):'', c?Math.round(c.y_far):'',
      f.mm_per_px??'', f.cons_mm??'', fl, note]);
  }));
  if(out.length===1){ alert('Niente da esportare: nessun commento, flag o correzione.'); return; }
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out.map(r=>r.map(esc).join(',')).join('\n')],
    {type:'text/csv;charset=utf-8;'}));
  a.download='revisione_scala.csv'; document.body.appendChild(a); a.click(); a.remove();
};

window.addEventListener('resize',()=>layout());
$('fsel').value='0'; show(); count();
</script></body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folders-file", type=Path,
                    help="TSV vendor<TAB>folder (relative to --root).")
    ap.add_argument("--folder", action="append", default=[],
                    help="A single folder; repeatable. Alternative to --folders-file.")
    ap.add_argument("--root", default="/Volumes/SSD_esi1_n1")
    ap.add_argument("--max-images", type=int, default=14)
    ap.add_argument("--pattern", default="image_depth_value_setup_*.png",
                    help="Which images to take. Use '*.png' on raw material.")
    ap.add_argument("--no-consensus", action="store_true",
                    help="Skip the setup consensus: right for frames that are not depth steps.")
    ap.add_argument("--cap-width", type=int, default=1500,
                    help="Embedded frame width; higher = sharper zoom, heavier page.")
    ap.add_argument("--high-recall", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    entries: List[Tuple[str, str]] = []
    if args.folders_file and args.folders_file.exists():
        for line in args.folders_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.split("\t")
                entries.append((parts[0].strip(), parts[-1].strip()))
    for f in args.folder:
        entries.append(("", f))
    if not entries:
        ap.error("serve --folders-file o almeno un --folder")

    out_data: List[dict] = []
    for i, (vendor, folder) in enumerate(entries, 1):
        path = folder if os.path.isabs(folder) else os.path.join(args.root, folder)
        print(f"[{i}/{len(entries)}] {vendor or '?'} · {os.path.basename(path)[:44]}", flush=True)
        hint = "" if vendor.endswith("-hard") else vendor
        d = collect_folder(path, hint, args.max_images, args.cap_width,
                           args.pattern, not args.no_consensus)
        if not d:
            print("      nessuna immagine utilizzabile", flush=True)
            continue
        if vendor:
            d["vendor"] = vendor
        out_data.append(d)
        print(f"      {d['status']}  {len(d['frames'])} frame", flush=True)

    if not out_data:
        print("[error] nessuna cartella elaborata")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(out_data, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    n = sum(len(d["frames"]) for d in out_data)
    print(f"[ok] {len(out_data)} cartelle, {n} frame -> {args.out} ({len(html)/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
