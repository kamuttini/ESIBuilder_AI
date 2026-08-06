"""One HTML page with the scale block's state across many vendors.

Takes several configuration folders (one per vendor, typically), runs the same pipeline the
per-folder report uses — detection, folder coherence rules, setup consensus — and renders a
single page: a cross-vendor summary on top, then a collapsible section per vendor with the
per-depth table and a few representative frames, zoomable, with the overlay toggle.

Kept to a handful of frames per vendor on purpose: the point is comparing vendors at a
glance, not reviewing every depth (``build_scale_folder_report.py`` does that).

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/build_scale_vendor_overview.py \
    --folders-file /tmp/vendor_sel.txt --per-vendor 3 \
    --out artifacts/47_scale_stato/tutti_i_vendor.html
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
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


def run_folder(folder: str, vendor_hint: str, per_vendor: int, max_images: int) -> Optional[dict]:
    base = os.path.join(folder, "image_samples")
    if not os.path.isdir(base):
        return None
    paths = sorted(glob.glob(os.path.join(base, "image_depth_value_setup_*.png")))[:max_images]
    if not paths:
        return None
    vendor = vendor_hint or rep.guess_vendor(folder)
    cases: List[dict] = []
    for p in paths:
        c = rep.process_image(p, vendor)
        if c:
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
    if cands:
        for k in consolidate_setup(cands, depth_indices=sorted({c.depth_index for c in cands})):
            cons[k.depth_index] = k

    # refresh the overlays after the coherence pass, then keep a few frames to show
    for c in cases:
        s = min(1.0, 1100.0 / float(c["w"]))
        vw = int(c["w"] * s) if s < 1.0 else c["w"]
        vh = int(c["h"] * s) if s < 1.0 else c["h"]
        n = len(c["pred"].ticks_y or [])
        c["svg"] = rep._svg(vw, vh, s, c["pred"], 1 if n <= 14 else (2 if n <= 28 else 4))

    # prefer showing a mix: one accepted, one review, one reject when available
    shown: List[dict] = []
    for want in ("accepted", "review", "reject"):
        for c in cases:
            if len(shown) >= per_vendor:
                break
            if c["status"] == want and c not in shown:
                shown.append(c)
                break
    for c in cases:
        if len(shown) >= per_vendor:
            break
        if c not in shown:
            shown.append(c)

    st: Dict[str, int] = {}
    for c in cases:
        st[c["status"]] = st.get(c["status"], 0) + 1
    return {
        "vendor": vendor or "default", "folder": os.path.basename(folder),
        "n": len(cases), "status": st,
        "fixes": sum(1 for c in cases if c.get("fixes")),
        "cases": cases, "shown": shown, "cons": cons,
    }


def render(results: List[dict], per_vendor: int) -> str:
    def pill(s: str, n: int) -> str:
        cls = {"accepted": "good", "review": "warn", "reject": "bad"}.get(s, "")
        return f'<span class="pill {cls}">{s} {n}</span>'

    rows = []
    for r in results:
        acc = r["status"].get("accepted", 0)
        pct = 100 * acc / r["n"] if r["n"] else 0
        cls = "good" if pct >= 60 else ("warn" if pct > 0 else "bad")
        rows.append(
            f"<tr><td><b>{r['vendor']}</b></td><td class='fold'>{r['folder'][:46]}</td>"
            f"<td>{r['n']}</td><td class='{cls}'>{acc} ({pct:.0f}%)</td>"
            f"<td>{r['status'].get('review',0)}</td><td>{r['status'].get('reject',0)}</td>"
            f"<td>{r['fixes']}</td></tr>"
        )

    sections = []
    for r in results:
        drows = []
        for c in sorted(r["cases"], key=lambda c: c["depth_index"]):
            k = r["cons"].get(c["depth_index"])
            cm = f"{k.mm_per_px:.4f}" if (k is not None and getattr(k, "mm_per_px", None)) else "-"
            src = getattr(k, "source", "-") if k is not None else "-"
            cls = {"accepted": "good", "review": "warn", "reject": "bad"}.get(c["status"], "")
            fx = "<br>".join(c.get("fixes", []))
            drows.append(
                f"<tr><td>{c['depth_index']}</td><td class='{cls}'>{c['status']}</td>"
                f"<td>{c['mm_per_px'] or '-'}</td><td>{cm}</td><td>{src}</td>"
                f"<td>{c['n_ticks']}</td><td>{c['step_mm'] or '-'}</td><td>{c['n_labels']}</td>"
                f"<td>{c['zero_at']}</td><td class='fx'>{fx}</td></tr>"
            )
        cards = []
        for c in r["shown"]:
            cls = {"accepted": "good", "review": "warn", "reject": "bad"}.get(c["status"], "")
            key = f"{r['vendor']}|{r['folder']}|{c['depth_index']}|{c['name']}"
            cards.append(f"""
            <div class="card" data-key="{key}" data-vendor="{r['vendor']}"
                 data-folder="{r['folder']}" data-depth="{c['depth_index']}"
                 data-name="{c['name']}" data-status="{c['status']}"
                 data-mmpx="{c['mm_per_px'] or ''}">
              <div class="hd">depth <b>{c['depth_index']}</b>
                <span class="pill {cls}">{c['status']}</span>
                <span class="pill">mm/px {c['mm_per_px'] or 'n/d'}</span>
                <span class="pill">tacche {c['n_ticks']}</span></div>
              <div class="viewer"><div class="stage">
                <img class="full" src="{c['img']}" alt="">{c['svg']}</div>
                <div class="zbar"><button data-z="out">−</button><span class="zlvl">100%</span>
                  <button data-z="in">+</button><button data-z="reset">reset</button>
                  <button class="tgl" data-on="1">nascondi</button>
                  <button class="fs" title="schermo intero">⛶ schermo intero</button></div>
                <div class="cbar">
                  <div class="flags">
                    <button data-f="col">colonna errata</button>
                    <button data-f="zero">zero errato</button>
                    <button data-f="ultima">ultima tacca errata</button>
                    <button data-f="valori">valori errati</button>
                    <button data-f="noruler">righello assente</button>
                  </div>
                  <input class="note" type="text" placeholder="commento per questo frame...">
                </div>
              </div>
            </div>""")
        pills = " ".join(pill(s, n) for s, n in sorted(r["status"].items()))
        sections.append(f"""
        <details><summary><b>{r['vendor']}</b> — {r['folder'][:50]}
            <span class="muted">{r['n']} depth</span> {pills}</summary>
          <table class="detail"><tr><th>depth</th><th>stato</th><th>mm/px imm.</th>
            <th>mm/px consenso</th><th>fonte</th><th>tacche</th><th>passo mm</th>
            <th>etich.</th><th>zero</th><th>correzioni</th></tr>{''.join(drows)}</table>
          <div class="grid">{''.join(cards)}</div>
        </details>""")

    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scala — stato per vendor</title>
<style>
 :root{{--bg:#0f1115;--card:#171a21;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
        --good:#37d67a;--bad:#ff5470;--warn:#ffb020}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
 header{{padding:18px 24px;border-bottom:1px solid var(--line);background:#12141a}}
 h1{{margin:0 0 5px;font-size:19px}} h2{{font-size:16px;margin:24px 0 8px}}
 .wrap{{max-width:1240px;margin:0 auto;padding:0 24px 50px}}
 .muted{{color:var(--mut)}}
 button{{background:#232732;color:var(--ink);border:1px solid var(--line);border-radius:7px;
         padding:4px 9px;cursor:pointer;font-size:12px}}
 table{{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:13px}}
 th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
 th{{color:var(--mut);font-weight:600}}
 td.good{{color:var(--good);font-weight:700}} td.warn{{color:var(--warn)}}
 td.bad{{color:var(--bad);font-weight:700}} td.fold{{font-size:12px}}
 td.fx{{color:var(--warn);font-size:11px}}
 table.detail{{font-size:12px}}
 details{{background:var(--card);border:1px solid var(--line);border-radius:9px;
          margin:9px 0;padding:10px 14px}}
 summary{{cursor:pointer;font-size:14px}}
 .pill{{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);
        font-size:11px;margin-left:5px;color:var(--mut)}}
 .pill.good{{border-color:var(--good);color:var(--good)}}
 .pill.warn{{border-color:var(--warn);color:var(--warn)}}
 .pill.bad{{border-color:var(--bad);color:var(--bad)}}
 .grid{{display:flex;gap:12px;flex-wrap:wrap}}
 .card{{background:#12141a;border:1px solid var(--line);border-radius:8px;overflow:hidden;
        flex:1 1 340px;min-width:320px}}
 .hd{{padding:7px 10px;border-bottom:1px solid var(--line);font-size:12px}}
 .viewer{{position:relative;overflow:hidden;background:#000;max-height:60vh;cursor:grab}}
 .viewer.dragging{{cursor:grabbing}}
 .viewer .stage{{position:relative;transform-origin:0 0;transition:transform .05s linear}}
 .viewer img.full{{display:block;width:100%;height:auto}}
 svg.ovl{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
 .hideovl svg.ovl{{display:none}}
 .zbar{{position:absolute;left:8px;bottom:8px;display:flex;gap:5px;align-items:center;
        background:rgba(10,12,17,.85);border:1px solid var(--line);border-radius:7px;
        padding:4px 7px;font-size:11px}}
 .zbar .zlvl{{min-width:40px;text-align:center;color:var(--mut)}}
 .zbar button.tgl{{border-color:var(--good);color:var(--good)}}
 .zbar button.tgl[data-on="0"]{{border-color:var(--line);color:var(--mut)}}
 /* comment bar lives inside the viewer so it stays reachable in fullscreen */
 .cbar{{position:absolute;right:8px;bottom:8px;left:auto;display:flex;gap:6px;
        align-items:center;flex-wrap:wrap;justify-content:flex-end;max-width:70%}}
 .cbar .flags{{display:flex;gap:4px;flex-wrap:wrap}}
 .cbar .flags button{{background:rgba(10,12,17,.85);font-size:11px;padding:3px 7px}}
 .cbar .flags button.on{{background:var(--bad);border-color:var(--bad);color:#fff}}
 .cbar input.note{{background:rgba(10,12,17,.9);color:var(--ink);border:1px solid var(--line);
                   border-radius:6px;padding:4px 8px;font-size:12px;width:260px}}
 .card.hasnote{{outline:2px solid var(--warn)}}
 /* fullscreen: the viewer takes the whole screen and the image is not clipped */
 .viewer:fullscreen{{max-height:none;height:100vh;width:100vw;background:#000;
                     display:flex;align-items:center;justify-content:center}}
 .viewer:fullscreen .stage{{width:100%}}
 .viewer:fullscreen img.full{{max-height:100vh;width:auto;margin:0 auto}}
 .viewer:fullscreen .cbar input.note{{width:420px}}
 .legend span{{margin-right:13px;font-size:12px}}
 .k{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;
     margin-right:4px}}
</style></head><body>
<header>
  <h1>Blocco scala — stato per vendor</h1>
  <div class="muted">Una cartella per vendor, stesso percorso della produzione: detection →
    regole di coerenza di cartella → consenso di setup. Nessuna ground truth usata: quello che
    vedi e' cio' che il sistema propone.</div>
  <div style="margin-top:8px" class="legend muted">
    <span><i class="k" style="background:#ffd24a"></i>tacche + valore (cm)</span>
    <span><i class="k" style="background:#37d67a"></i>zero</span>
    <span><i class="k" style="background:#a000c8"></i>ultima tacca</span>
    <span><i class="k" style="background:#ff5000"></i>numero letto</span>
    <span><i class="k" style="background:#00b4ff"></i>colonna</span>
    <button id="tgl-all" data-on="1">nascondi elaborazioni su tutte</button>
  </div>
  <div style="margin-top:9px" class="muted">
    <b>⛶ schermo intero</b> su ogni frame · flag rapidi e commento nella barra in basso a destra
    (restano salvati nel browser) ·
    <button id="exp-cm">Esporta commenti CSV</button>
    <span class="muted">frame commentati: <b id="cm-count">0</b></span>
  </div>
</header>
<div class="wrap">
<h2>Riepilogo</h2>
<div class="muted">"review" non e' un fallimento: e' il sistema che propone e chiede conferma.
  Il numero che conta per il deploy e' la <b>purezza</b> degli accepted, non il loro numero.</div>
<table><tr><th>vendor</th><th>cartella</th><th>depth</th><th>accepted</th><th>review</th>
  <th>reject</th><th>correzioni coerenza</th></tr>{''.join(rows)}</table>
<h2>Dettaglio per vendor ({per_vendor} frame per vendor)</h2>
{''.join(sections)}
</div>
<script>
document.querySelectorAll('.viewer').forEach(v => {{
  const st=v.querySelector('.stage'), lvl=v.querySelector('.zlvl');
  const card=v.closest('.card'), tgl=v.querySelector('.tgl');
  let z=1,tx=0,ty=0,drag=false,sx=0,sy=0;
  const apply=()=>{{
    const r=v.getBoundingClientRect();
    const mx=Math.max(0,r.width*(z-1)), my=Math.max(0,r.height*(z-1));
    tx=Math.min(0,Math.max(-mx,tx)); ty=Math.min(0,Math.max(-my,ty));
    st.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{z}})`;
    lvl.textContent=Math.round(z*100)+'%';
  }};
  const zoomAt=(f,cx,cy)=>{{
    const nz=Math.min(12,Math.max(1,z*f));
    tx=cx-(cx-tx)*(nz/z); ty=cy-(cy-ty)*(nz/z); z=nz;
    if(z===1){{tx=0;ty=0;}} apply();
  }};
  v.addEventListener('wheel',e=>{{e.preventDefault();const r=v.getBoundingClientRect();
    zoomAt(e.deltaY<0?1.15:1/1.15,e.clientX-r.left,e.clientY-r.top);}},{{passive:false}});
  v.addEventListener('pointerdown',e=>{{if(e.target.closest('.zbar')||e.target.closest('.cbar'))return;
    drag=true;sx=e.clientX-tx;sy=e.clientY-ty;v.classList.add('dragging');v.setPointerCapture(e.pointerId);}});
  v.addEventListener('pointermove',e=>{{if(drag){{tx=e.clientX-sx;ty=e.clientY-sy;apply();}}}});
  v.addEventListener('pointerup',()=>{{drag=false;v.classList.remove('dragging');}});
  v.addEventListener('dblclick',()=>{{z=1;tx=0;ty=0;apply();}});
  v.querySelectorAll('.zbar button[data-z]').forEach(b=>b.addEventListener('click',()=>{{
    const r=v.getBoundingClientRect();
    if(b.dataset.z==='in')zoomAt(1.4,r.width/2,r.height/2);
    else if(b.dataset.z==='out')zoomAt(1/1.4,r.width/2,r.height/2);
    else {{z=1;tx=0;ty=0;apply();}}
  }}));
  tgl.addEventListener('click',()=>{{
    const on=tgl.dataset.on==='1'; tgl.dataset.on=on?'0':'1';
    tgl.textContent=on?'mostra':'nascondi'; card.classList.toggle('hideovl',on);
  }});
  // fullscreen on the viewer itself: the zoom/pan handlers keep working as they are
  const fsb=v.querySelector('.fs');
  if(fsb) fsb.addEventListener('click',()=>{{
    if(document.fullscreenElement===v) document.exitFullscreen();
    else if(v.requestFullscreen) v.requestFullscreen();
  }});
  document.addEventListener('fullscreenchange',()=>{{ z=1;tx=0;ty=0;apply(); }});
  apply();
}});

// ---- commenti: persistenti in locale, esportabili ----
const CKEY='scale_overview_'+(location.pathname.split('/').pop()||'x');
let CM={{}};
try{{ CM=JSON.parse(localStorage.getItem(CKEY)||'{{}}'); }}catch(e){{ CM={{}}; }}
function cmSave(){{ localStorage.setItem(CKEY,JSON.stringify(CM)); refreshCount(); }}
function cmGet(k){{ if(!CM[k]) CM[k]={{flags:{{}},note:''}}; return CM[k]; }}
document.querySelectorAll('.card').forEach(card=>{{
  const k=card.dataset.key, st=cmGet(k);
  const note=card.querySelector('.note'); note.value=st.note||'';
  const paint=()=>{{
    const any=(st.note&&st.note.trim())||Object.values(st.flags||{{}}).some(Boolean);
    card.classList.toggle('hasnote',!!any);
  }};
  note.addEventListener('input',()=>{{ st.note=note.value; cmSave(); paint(); }});
  card.querySelectorAll('.cbar .flags button').forEach(b=>{{
    const f=b.dataset.f;
    if(st.flags&&st.flags[f]) b.classList.add('on');
    b.addEventListener('click',()=>{{
      st.flags=st.flags||{{}}; st.flags[f]=!st.flags[f];
      b.classList.toggle('on'); cmSave(); paint();
    }});
  }});
  paint();
}});
function refreshCount(){{
  let n=0;
  document.querySelectorAll('.card').forEach(c=>{{
    const s=CM[c.dataset.key]; if(!s) return;
    if((s.note&&s.note.trim())||Object.values(s.flags||{{}}).some(Boolean)) n++;
  }});
  const el=document.getElementById('cm-count'); if(el) el.textContent=n;
}}
function csvEsc(v){{ return '"'+String(v??'').replace(/"/g,'""')+'"'; }}
document.getElementById('exp-cm').addEventListener('click',()=>{{
  const out=[["vendor","cartella","depth","immagine","stato","mm_per_px","flag","commento"]];
  document.querySelectorAll('.card').forEach(c=>{{
    const s=CM[c.dataset.key]; if(!s) return;
    const flags=Object.keys(s.flags||{{}}).filter(f=>s.flags[f]).join('|');
    const note=(s.note||'').trim();
    if(!flags&&!note) return;
    out.push([c.dataset.vendor,c.dataset.folder,c.dataset.depth,c.dataset.name,
              c.dataset.status,c.dataset.mmpx,flags,note]);
  }});
  if(out.length===1){{ alert('Nessun commento da esportare.'); return; }}
  const csv=out.map(r=>r.map(csvEsc).join(',')).join('\\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8;'}}));
  a.download='commenti_scala.csv'; document.body.appendChild(a); a.click(); a.remove();
}});
refreshCount();
const all=document.getElementById('tgl-all');
all.addEventListener('click',()=>{{
  const on=all.dataset.on==='1'; all.dataset.on=on?'0':'1';
  all.textContent=on?'mostra elaborazioni su tutte':'nascondi elaborazioni su tutte';
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hideovl',on));
  document.querySelectorAll('.tgl').forEach(t=>{{t.dataset.on=on?'0':'1';
    t.textContent=on?'mostra':'nascondi';}});
}});
</script></body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folders-file", type=Path, required=True,
                    help="TSV: vendor<TAB>folder_name (relative to --root).")
    ap.add_argument("--root", default="/Volumes/SSD_esi1_n1")
    ap.add_argument("--per-vendor", type=int, default=3, help="Frames shown per vendor.")
    ap.add_argument("--max-images", type=int, default=14, help="Frames processed per vendor.")
    ap.add_argument("--high-recall", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    entries: List[Tuple[str, str]] = []
    for line in args.folders_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        entries.append((parts[0].strip(), parts[-1].strip()))

    results: List[dict] = []
    for i, (vendor, folder) in enumerate(entries, 1):
        path = os.path.join(args.root, folder)
        print(f"[{i}/{len(entries)}] {vendor} · {folder[:44]}", flush=True)
        hint = "" if vendor.endswith("-hard") else vendor
        r = run_folder(path, hint, args.per_vendor, args.max_images)
        if r:
            r["vendor"] = vendor
            results.append(r)
            print(f"      {r['status']} correzioni={r['fixes']}", flush=True)
        else:
            print("      nessuna immagine utilizzabile", flush=True)

    if not results:
        print("[error] nessun vendor elaborato")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = render(results, args.per_vendor)
    args.out.write_text(html, encoding="utf-8")
    print(f"[ok] {len(results)} vendor -> {args.out} ({len(html)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
