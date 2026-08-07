#!/usr/bin/env python3
"""One page listing every review produced by a sweep, so they can be judged side by side.

Reads the sweep's results JSON and links each study, showing at a glance where the scale zone
was found, how the frames were classified and how many numbers came out suspect.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "50_scale_study"

CSS = """
:root{--bg:#0b0d11;--panel:#14171e;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
      --good:#37d67a;--warn:#ffb020;--bad:#ff5470}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:19px} .muted{color:var(--mut)}
.wrap{padding:18px 24px 60px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}
tr:hover td{background:#12141a}
a{color:#6aa6ff;text-decoration:none} a:hover{text-decoration:underline}
.fam{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);
      font-size:11px;color:var(--mut);white-space:nowrap}
.ok{border-color:var(--good);color:var(--good)}
.wr{border-color:var(--warn);color:var(--warn)}
.bd{border-color:var(--bad);color:var(--bad)}
.sum{display:flex;gap:22px;flex-wrap:wrap;margin:12px 0 20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 15px}
.card b{display:block;font-size:21px;font-weight:700}
.nm{max-width:430px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chk{width:17px;height:17px;cursor:pointer;accent-color:#37d67a}
tr.done td{opacity:.42}
tr.done td:first-child, tr.done td:nth-child(2){opacity:.62}
.bar{display:flex;gap:12px;align-items:center;margin:0 0 12px;flex-wrap:wrap}
button{background:#232732;color:var(--ink);border:1px solid var(--line);border-radius:6px;
       padding:5px 11px;font-size:12px;cursor:pointer}
button.on{background:#1f6feb;border-color:#1f6feb}
.seen{color:var(--good);font-weight:700}
button.go{background:#1f6feb;border-color:#1f6feb;font-weight:600}
#rlog{position:fixed;right:16px;bottom:16px;z-index:50;width:min(580px,calc(100vw - 32px));
      background:#0c0e13;border:1px solid var(--line);border-radius:10px;box-shadow:0 12px 34px #000b;
      padding:12px 14px;max-height:56vh;overflow:auto;
      font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap}
#rlog .bar2{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;
            font-family:-apple-system,Segoe UI,Roboto,sans-serif}
"""

# Plain (non f-string) so the braces in this JavaScript need no escaping.
JS = """
// Same registry the review pages write, keyed by the study's file name: ticking here or there is
// the same tick, because both sides key on the name this page links to.
const DK='scale_study_done';
const rows=[...document.querySelectorAll('tr[data-out]')];
function all(){ try{return JSON.parse(localStorage.getItem(DK)||'{}');}catch(e){return {};} }
function paint(){
  const A=all(); let n=0;
  rows.forEach(tr=>{
    const d=!!(A[tr.dataset.out]||{}).done;
    if(d) n++;
    tr.classList.toggle('done',d);
    tr.querySelector('.chk').checked=d;
    const w=tr.querySelector('.when');
    if(w) w.textContent = d ? ((A[tr.dataset.out]||{}).at||'') : '';
  });
  document.getElementById('seen').textContent=n;
  filter();
}
function filter(){
  const only=document.getElementById('flt').classList.contains('on');
  rows.forEach(tr=>{ tr.style.display = (only && tr.classList.contains('done')) ? 'none' : ''; });
}
document.addEventListener('change',e=>{
  if(!e.target.classList.contains('chk')) return;
  const tr=e.target.closest('tr'), A=all();
  if(e.target.checked) A[tr.dataset.out]={done:true, at:new Date().toISOString().slice(0,16).replace('T',' ')};
  else delete A[tr.dataset.out];
  localStorage.setItem(DK,JSON.stringify(A));
  paint();
});
document.getElementById('flt').onclick=e=>{ e.currentTarget.classList.toggle('on'); filter(); };
document.getElementById('clr').onclick=()=>{
  if(!confirm('Togliere la spunta a tutte le cartelle?')) return;
  localStorage.removeItem(DK); paint();
};
// Another tab may have ticked a folder while this page was open.
window.addEventListener('storage',e=>{ if(e.key===DK) paint(); });
window.addEventListener('focus',paint);
paint();

// ---- how much feedback the server holds, and re-running with it ----
async function cstat(){
  try{
    const j=await (await fetch('/api/corrections')).json();
    document.getElementById('cstat').textContent = j.folders
      ? ('il server ha correzioni per ' + j.folders + ' cartelle, ' + j.frames + ' frame')
      : 'il server non ha ancora ricevuto correzioni: inviale da dentro una revisione';
  }catch(e){ document.getElementById('cstat').textContent =
    "questa pagina non e' servita dal tool, quindi non posso rielaborare da qui"; }
}
cstat();
document.getElementById('rerun').onclick=async()=>{
  const label=prompt('Nome della nuova run (per distinguerla dalle altre):',
    'run_' + new Date().toISOString().slice(0,10).replace(/-/g,'') + '_corretta');
  if(label===null) return;
  const L=document.getElementById('rlog');
  L.hidden=false;
  // Handler wired in code, not in an onclick attribute: quoting an id inside an attribute inside
  // a string inside Python is exactly how the last syntax error got in.
  L.innerHTML='<div class="bar2"><b>rielaborazione in corso…</b>'
    +'<button id="rx">chiudi</button></div><div id="rbody">avvio…</div>';
  document.getElementById('rx').onclick=()=>{ L.hidden=true; };
  const r=await (await fetch('/api/rerun',{method:'POST',
    body:new URLSearchParams({label})})).json();
  if(r.error){ document.getElementById('rbody').textContent='ERRORE: '+r.error; return; }
  poll2();
};
async function poll2(){
  const s=await (await fetch('/api/sweep_status')).json();
  const b=document.getElementById('rbody');
  if(b){ b.textContent=s.log.join('\\n'); b.scrollTop=b.scrollHeight; }
  document.getElementById('rlog').scrollTop=document.getElementById('rlog').scrollHeight;
  if(s.state==='running'){ setTimeout(poll2,2000); return; }
  if(s.state==='done' && s.index){
    b.innerHTML += '\\n\\nPronto. <a href="/study/'+encodeURIComponent(s.index)
      +'">' + "apri l'indice della nuova run" + '</a> e confronta con questo.';
  } else { b.textContent += '\\n\\nfinito con errori.'; }
}
"""


def pill(txt: str, cls: str = "") -> str:
    return f'<span class="pill {cls}">{html.escape(txt)}</span>'


def build(results: list, title: str) -> str:
    rows = []
    n_ok = sum(1 for r in results if r.get("ok"))
    zf = sum(r.get("zone_found", 0) for r in results)
    zt = sum(r.get("zone_total", 0) for r in results)
    acc = sum((r.get("stati") or {}).get("accepted", 0) for r in results)
    tot = sum(sum((r.get("stati") or {}).values()) for r in results)
    sus = sum(r.get("suspect", 0) for r in results)

    for r in sorted(results, key=lambda x: (x.get("family", ""), x.get("acquisition", ""))):
        st = r.get("stati") or {}
        n = sum(st.values())
        parts = []
        for k, cls in (("accepted", "ok"), ("review", "wr"), ("reject", "bd")):
            if st.get(k):
                parts.append(pill(f"{st[k]} {k}", cls))
        zone = ""
        if r.get("zone_total"):
            f, t = r["zone_found"], r["zone_total"]
            zone = pill(f"{f}/{t}", "ok" if f == t else ("wr" if f >= t * 0.6 else "bd"))
        ori = ", ".join(f"{k}:{v}" for k, v in (r.get("orient") or {}).items()) or "—"
        vend = html.escape(r.get("vendor") or "default")
        vsrc = html.escape(r.get("vendor_src") or "")
        name = html.escape(r.get("acquisition", ""))
        link = (f'<a href="{html.escape(r["out"])}" target="_blank">{name}</a>'
                if r.get("ok") else f'{name} <span class="pill bd">non riuscito</span>')
        s = r.get("suspect")
        rows.append(
            f'<tr data-out="{html.escape(r.get("out",""))}">'
            f'<td><input class="chk" type="checkbox" title="gia\' riguardata"></td>'
            f'<td class="fam">{html.escape(r.get("family",""))}</td>'
            f'<td class="nm">{link}<div class="muted">{r.get("n_images","?")} frame '
            f'nella cartella<span class="when muted"></span></div></td>'
            f'<td>{vend}<div class="muted">{vsrc}</div></td>'
            f'<td>{zone}</td><td>{" ".join(parts) or "—"}</td>'
            f'<td>{pill(f"{s}/{n}", "wr" if s else "ok") if s is not None else "—"}</td>'
            f'<td class="muted">{html.escape(ori)}</td>'
            f'<td class="muted">{r.get("seconds","")}s</td></tr>')

    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="muted">Ogni riga e' una cartella diversa. Clicca il nome per aprire la revisione
 con i cinque stadi, dove puoi correggere e commentare.</div></header>
<div class="wrap">
 <div class="sum">
  <div class="card"><b>{n_ok}/{len(results)}</b><span class="muted">studi riusciti</span></div>
  <div class="card"><b>{len({r.get('family') for r in results})}</b>
    <span class="muted">famiglie di ecografi</span></div>
  <div class="card"><b>{zf}/{zt}</b><span class="muted">frame con zona scala trovata</span></div>
  <div class="card"><b>{acc}/{tot}</b><span class="muted">frame accepted</span></div>
  <div class="card"><b>{sus}</b><span class="muted">frame con numeri sospetti</span></div>
 </div>
 <div class="bar">
  <span class="muted">riguardate: <span class="seen" id="seen">0</span> su {len(results)}</span>
  <button id="flt">nascondi quelle riguardate</button>
  <button id="clr">azzera le spunte</button>
  <span class="muted">la spunta vale anche dentro la revisione (tasto <b>d</b>)</span>
 </div>
 <div class="bar">
  <button id="rerun" class="go">rielabora tutto con le mie correzioni (nuova run)</button>
  <a href="/api/corrections.csv">scarica tutte le correzioni raccolte</a>
  <span class="muted" id="cstat">…</span>
 </div>
 <div id="rlog" hidden></div>
 <table><tr><th></th><th>famiglia</th><th>cartella</th><th>vendor</th><th>zona</th><th>stati</th>
   <th>sospetti</th><th>orientamento</th><th>tempo</th></tr>
 {"".join(rows)}</table>
</div><script>{JS}</script></body></html>"""


def check_js(js: str) -> list:
    """Catch the one mistake this file invites: a JS string closed by an Italian apostrophe.

    The JS lives inside a Python string, so a single-quoted JS literal containing an escaped
    apostrophe ends up containing a real one and the whole script dies with a SyntaxError - and
    a dead script looks exactly like a page that simply does nothing.
    """
    bad = []
    for n, line in enumerate(js.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Walk the line tracking the quote we are inside. A bare apostrophe cannot be told from a
        # closing quote by counting, but what follows gives it away: a real closing quote is
        # followed by an operator, never by a word.
        i, quote = 0, ""
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = ""
                    j = i + 1
                    while j < len(line) and line[j] == " ":
                        j += 1
                    if j < len(line) and (line[j].isalpha() or line[j] == "_"):
                        bad.append(f"riga {n}: stringa chiusa da un apostrofo -> {stripped[:78]}")
                        break
            elif ch in "\"'":
                quote = ch
            elif ch == "/" and line[i + 1:i + 2] == "/":
                break
            i += 1
    return bad


def main_with(results: str, out: str, title: str) -> int:
    """Callable entry point, so a sweep can build its own index without a subprocess."""
    problems = check_js(JS)
    if problems:
        print("[errore] il JavaScript della pagina e' rotto, non la scrivo:")
        for p in problems:
            print("  " + p)
        return 2
    data = json.loads(Path(results).read_text("utf-8"))
    Path(out).write_text(build(data, title), "utf-8")
    print(f"[ok] {len(data)} studi -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT_DIR / "v3_sweep.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "v3_indice.html"))
    ap.add_argument("--title", default="Studio della scala — volume 3")
    args = ap.parse_args()
    return main_with(args.results, args.out, args.title)


if __name__ == "__main__":
    raise SystemExit(main())
