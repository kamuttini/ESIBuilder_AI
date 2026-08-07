"""Local web app to pick a folder and study its scale, without touching the terminal.

An HTML page cannot run the study by itself, so this serves one: browse the volumes, click a
folder, watch the five stages go by, and land on the review tool for that folder. Same
pattern the repo already uses for the other workbenches, and standard library only.

  browse   one listdir per request, so navigating 272 folders costs nothing
  run      the study runs in a worker thread and its log is streamed to the page
  review   when it is done the page opens the usual five-stage review tool

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/scale_study_web.py
  # then open http://127.0.0.1:8790/
"""
from __future__ import annotations

import argparse
import glob
import html
import io
import json
import os
import re
import sys
import threading
import traceback
import urllib.parse
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "50_scale_study"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# one job at a time: the study is heavy and the operator looks at one folder at a time
JOB: Dict[str, object] = {"state": "idle", "log": [], "out": None, "folder": ""}
LOCK = threading.Lock()


def count_images(d: str, pattern: str) -> int:
    """Images directly in the folder, or in its image_samples, without walking deeper."""
    n = len(glob.glob(os.path.join(d, pattern)))
    if not n:
        n = len(glob.glob(os.path.join(d, "image_samples", pattern)))
    return n


def list_dir(path: str, pattern: str) -> dict:
    """Subdirectories of ``path`` with how many images each holds (shallow)."""
    entries = []
    try:
        with os.scandir(path) as it:
            for e in sorted(it, key=lambda e: e.name.lower()):
                if not e.is_dir(follow_symlinks=False) or e.name.startswith("."):
                    continue
                if e.name in ("$RECYCLE.BIN", "System Volume Information"):
                    continue
                entries.append({"name": e.name, "path": e.path,
                                "images": count_images(e.path, pattern)})
    except OSError as exc:
        return {"path": path, "error": str(exc), "entries": [], "self_images": 0}
    return {"path": path, "parent": os.path.dirname(path.rstrip("/")) or "/",
            "entries": entries, "self_images": count_images(path, pattern)}


def roots() -> List[dict]:
    out = []
    for p in sorted(glob.glob("/Volumes/*")):
        if os.path.isdir(p):
            out.append({"name": os.path.basename(p), "path": p})
    out.append({"name": "home", "path": str(Path.home())})
    return out


def run_study(folder: str, opts: dict) -> None:
    """Run the five-stage study in this thread, streaming its log into JOB."""
    import study_scale_folder as S  # same directory

    def log(msg: str) -> None:
        with LOCK:
            JOB["log"].append(msg)  # type: ignore[union-attr]

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(folder.rstrip("/")))[:60] or "studio"
    out = OUT_DIR / f"{safe}.html"
    argv = ["--folder", folder, "--out", str(out),
            "--pattern", opts.get("pattern") or "image_depth_value_setup_*.png",
            "--max-images", str(opts.get("max_images") or 14),
            "--python-bin", sys.executable]
    if opts.get("no_depth"):
        argv.append("--no-depth")
    if opts.get("high_recall"):
        argv.append("--high-recall")
    if opts.get("vendor"):
        argv += ["--vendor", str(opts["vendor"])]

    buf = io.StringIO()

    class Tee(io.TextIOBase):
        """Forward the study's prints to the page as they happen."""
        def write(self, s):  # type: ignore[override]
            buf.write(s)
            for line in s.splitlines():
                if line.strip():
                    log(line.rstrip())
            return len(s)

    try:
        with redirect_stdout(Tee()):
            rc = S.main(argv)
        with LOCK:
            JOB["state"] = "done" if rc == 0 else "error"
            JOB["out"] = out.name if rc == 0 and out.exists() else None
    except Exception:
        log("ERRORE: " + traceback.format_exc(limit=4))
        with LOCK:
            JOB["state"] = "error"


# Raw string: the JS below contains \n escapes, and letting Python interpret them
# turned them into real line breaks that split the JS string literals in half —
# a SyntaxError that killed the whole script, so nothing on the page worked.
HOME = r"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio scala — scegli la cartella</title>
<style>
 :root{--bg:#0b0d11;--panel:#14171e;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
       --good:#37d67a;--warn:#ffb020;--bad:#ff5470}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:13px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:16px 22px;border-bottom:1px solid var(--line);background:var(--panel)}
 h1{margin:0 0 4px;font-size:18px} .muted{color:var(--mut)}
 .wrap{max-width:1020px;margin:0 auto;padding:18px 22px 60px}
 button,select,input{background:#232732;color:var(--ink);border:1px solid var(--line);
                     border-radius:6px;padding:5px 10px;font-size:12px}
 button{cursor:pointer} button.go{background:#1f6feb;border-color:#1f6feb}
 button:disabled{opacity:.5;cursor:default}
 .crumb{margin:12px 0;font-size:12px;color:var(--mut);word-break:break-all}
 .row{display:flex;gap:8px;align-items:center;padding:6px 9px;border-bottom:1px solid var(--line)}
 .row:hover{background:#12141a}
 .row .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .pill{padding:1px 8px;border-radius:99px;border:1px solid var(--line);font-size:11px;color:var(--mut)}
 .pill.good{border-color:var(--good);color:var(--good)}
 .opts{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0;
       background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
 /* Fixed, not in the flow: with a few hundred folders listed the panel ended up ~16000px
    below the fold, so clicking "studia" looked like it did nothing at all. */
 #log{position:fixed;right:16px;bottom:16px;z-index:50;width:min(580px,calc(100vw - 32px));
      background:#0c0e13;border:1px solid var(--line);border-radius:10px;
      box-shadow:0 12px 34px #000b;padding:12px 14px 14px;max-height:56vh;overflow:auto}
 #logbody{font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap}
 #log .bar{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}
 a.open{display:inline-block;margin-top:10px;background:#1f6feb;color:#fff;text-decoration:none;
        padding:8px 15px;border-radius:6px;font-weight:600;font-size:13px}
 .done{color:var(--good);font-weight:700} .err{color:var(--bad);font-weight:700}
 #ready{margin:10px 0 0;font-size:12px} #ready a{color:#6aa6ff}
</style></head><body>
<header><h1>Studio della scala</h1>
  <div class="muted">Scegli qui sotto: <b>apri</b> per entrare in una cartella,
    <b>studia</b> per analizzarla. Il numero verde dice quante immagini contiene.</div></header>
<div class="wrap">
  <div class="opts">
    <label>immagini max <input id="maxi" type="number" value="14" min="2" max="60" style="width:70px"></label>
    <label>pattern <input id="pat" value="image_depth_value_setup_*.png" style="width:230px"></label>
    <label><input id="nodepth" type="checkbox"> salta lo stadio depth (piu' veloce)</label>
    <label><input id="hr" type="checkbox"> alta recall</label>
    <span class="muted">per il materiale grezzo usa pattern <code>*.png</code></span>
  </div>
  <div><span class="muted">volumi:</span> <span id="roots"></span></div>
  <div id="ready"></div>
  <div class="crumb">sei in: <span id="crumb"></span></div>
  <div id="list"></div>
</div>
<div id="log" hidden>
  <div class="bar"><b id="logttl">studio in corso…</b>
    <button id="logx">chiudi</button></div>
  <div id="logbody"></div>
  <div id="logact"></div>
</div>
<script>
let cur = null;
const $ = id => document.getElementById(id);
const pat = () => $('pat').value.trim() || 'image_depth_value_setup_*.png';
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                          .replace(/"/g,'&quot;').replace(/'/g,'&#39;');

// One listener per container and the path in a data attribute: putting a JSON path inside an
// inline onclick broke the attribute on any path containing a quote, and every click silently
// did nothing.
function rowsHtml(d){
  const out=[];
  if(d.parent && d.parent !== d.path)
    out.push(`<div class="row"><span class="nm">⤴ ..</span>
      <button data-act="open" data-path="${esc(d.parent)}">apri</button></div>`);
  if(d.self_images)
    out.push(`<div class="row"><span class="nm"><b>questa cartella</b></span>
      <span class="pill good">${d.self_images} immagini</span>
      <button class="go" data-act="run" data-path="${esc(d.path)}">studia</button></div>`);
  d.entries.forEach(e=>{
    out.push(`<div class="row"><span class="nm">${esc(e.name)}</span>
      ${e.images?`<span class="pill good">${e.images} img</span>`:'<span class="pill">—</span>'}
      <button data-act="open" data-path="${esc(e.path)}">apri</button>
      ${e.images?`<button class="go" data-act="run" data-path="${esc(e.path)}">studia</button>`:''}
      </div>`);
  });
  return out.join('') || '<div class="muted">nessuna sottocartella</div>';
}
async function loadRoots(){
  const r = await (await fetch('/api/roots')).json();
  $('roots').innerHTML = r.map(x =>
    `<button data-act="open" data-path="${esc(x.path)}">${esc(x.name)}</button>`).join(' ');
}
async function go(path){
  cur = path;
  $('list').innerHTML = '<div class="muted">apro…</div>';
  const d = await (await fetch('/api/list?pattern='+encodeURIComponent(pat())
                   +'&path='+encodeURIComponent(path))).json();
  $('crumb').textContent = d.path + (d.error ? '  —  ' + d.error : '');
  $('list').innerHTML = rowsHtml(d);
}
document.addEventListener('click', e=>{
  const b = e.target.closest('button[data-act]');
  if(!b) return;
  const path = b.getAttribute('data-path');
  if(b.getAttribute('data-act')==='open') go(path); else run(path);
});
$('pat').addEventListener('change', ()=>{ if(cur) go(cur); });
async function run(path){
  $('log').hidden = false;
  $('logttl').textContent = 'studio in corso…';
  $('logact').innerHTML = '';
  $('logbody').textContent = 'avvio dello studio su:\n' + path + '\n\n';
  const body = new URLSearchParams({folder:path, pattern:pat(),
    max_images:$('maxi').value, no_depth:$('nodepth').checked?'1':'',
    high_recall:$('hr').checked?'1':''});
  const r = await (await fetch('/api/run',{method:'POST',body})).json();
  if(r.error){ $('logbody').textContent += 'ERRORE: '+r.error; return; }
  poll();
}
async function poll(){
  const s = await (await fetch('/api/status')).json();
  $('logbody').textContent = s.log.join('\n');
  $('log').scrollTop = $('log').scrollHeight;
  if(s.state==='running'){ setTimeout(poll, 900); return; }
  if(s.state==='done' && s.out){
    // A link the user clicks, not window.open(): opening a tab from this async callback is
    // not a user gesture, so Chrome blocked it and the finished study stayed unreachable.
    $('logttl').innerHTML = '<span class="done">studio pronto</span>';
    $('logact').innerHTML = '<a class="open" href="/study/' + encodeURIComponent(s.out)
      + '" target="_blank" rel="noopener">apri la revisione delle immagini →</a>';
    loadReady();
  } else {
    $('logttl').innerHTML = '<span class="err">finito con errori</span>';
  }
}
$('logx').addEventListener('click', ()=>{ $('log').hidden = true; });
// Studies already on disk: without this a finished run was only reachable from the panel of
// the session that produced it.
async function loadReady(){
  const r = await (await fetch('/api/studies')).json();
  $('ready').innerHTML = r.length
    ? '<span class="muted">revisioni pronte:</span> ' + r.slice(0,12).map(x =>
        '<a href="/study/' + encodeURIComponent(x.name) + '" target="_blank" rel="noopener">'
        + esc(x.label) + '</a>').join(' &middot; ')
    : '';
}
// Open /Volumes straight away: showing only the volume buttons with an empty list below
// made the page look like it had nowhere to choose the folder.
(async ()=>{ await loadRoots(); await loadReady(); await go('/Volumes'); })();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ScaleStudy/1.0"

    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(200, HOME.encode("utf-8"))
        elif u.path == "/api/roots":
            self._json(roots())
        elif u.path == "/api/list":
            path = (q.get("path") or ["/Volumes"])[0]
            pattern = (q.get("pattern") or ["image_depth_value_setup_*.png"])[0]
            self._json(list_dir(path, pattern))
        elif u.path == "/api/studies":
            done = sorted(OUT_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
            self._json([{"name": p.name, "label": p.stem.replace("_", " ")} for p in done])
        elif u.path == "/api/status":
            with LOCK:
                self._json({"state": JOB["state"], "log": list(JOB["log"]),  # type: ignore[arg-type]
                            "out": JOB["out"], "folder": JOB["folder"]})
        elif u.path.startswith("/study/"):
            name = urllib.parse.unquote(u.path[len("/study/"):])
            p = (OUT_DIR / name).resolve()
            # stay inside the output directory
            if OUT_DIR.resolve() in p.parents and p.is_file():
                self._send(200, p.read_bytes())
            else:
                self._send(404, b"non trovato")
        else:
            self._send(404, b"non trovato")

    def do_POST(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/run":
            self._send(404, b"non trovato")
            return
        n = int(self.headers.get("Content-Length") or 0)
        form = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8"))
        folder = (form.get("folder") or [""])[0]
        if not folder or not os.path.isdir(folder):
            self._json({"error": "cartella non valida"})
            return
        with LOCK:
            if JOB["state"] == "running":
                self._json({"error": "uno studio e' gia' in corso"})
                return
            JOB.update({"state": "running", "log": [f"cartella: {folder}"],
                        "out": None, "folder": folder})
        opts = {
            "pattern": (form.get("pattern") or [""])[0],
            "max_images": (form.get("max_images") or ["14"])[0],
            "no_depth": bool((form.get("no_depth") or [""])[0]),
            "high_recall": bool((form.get("high_recall") or [""])[0]),
        }
        threading.Thread(target=run_study, args=(folder, opts), daemon=True).start()
        self._json({"ok": True})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--open", dest="open_after", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[ok] studio scala su {url}   (Ctrl+C per fermare)")
    if args.open_after and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", url], check=False)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
