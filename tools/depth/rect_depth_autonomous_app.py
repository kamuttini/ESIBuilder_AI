#!/usr/bin/env python3
"""Local folder launcher for autonomous RECT_DEPTH analysis and review."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

from predict_rect_depth_autonomous import _infer_probe_from_text, _infer_vendor_from_text


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "artifacts/24_rect_depth_hybrid/autonomous_interactive_runs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

JOBS: Dict[str, Dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


def _safe_slug(text: str, max_len: int = 90) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "run"


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _find_fss(folder: Path) -> List[str]:
    candidates: List[Path] = []
    db_setup = folder / "DB_setup"
    if db_setup.is_dir():
        candidates.extend(sorted(db_setup.glob("*.fss")))
    candidates.extend(sorted(folder.glob("*.fss")))
    seen: set[str] = set()
    out: List[str] = []
    for path in candidates:
        resolved = path.expanduser().resolve().as_posix()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _image_dir(folder: Path) -> Path:
    return folder / "image_samples" if (folder / "image_samples").is_dir() else folder


def _count_images(folder: Path, limit: int = 100000) -> int:
    image_dir = _image_dir(folder)
    if not image_dir.is_dir():
        return 0
    count = 0
    for item in image_dir.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            count += 1
            if count >= limit:
                break
    return count


def _folder_info(path_text: str) -> Dict[str, object]:
    path = Path(path_text).expanduser()
    exists = path.exists()
    is_dir = path.is_dir()
    resolved = path.resolve().as_posix() if exists else path.as_posix()
    dirs: List[Dict[str, object]] = []
    if is_dir:
        try:
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    dirs.append(
                        {
                            "name": child.name,
                            "path": child.resolve().as_posix(),
                            "has_image_samples": (child / "image_samples").is_dir(),
                            "has_fss": bool(_find_fss(child)),
                        }
                    )
        except Exception:
            dirs = []
    text_for_context = f"{resolved} {path.name}"
    return {
        "path": resolved,
        "parent": path.parent.resolve().as_posix() if path.parent.exists() else "",
        "exists": exists,
        "is_dir": is_dir,
        "image_dir": _image_dir(path).resolve().as_posix() if is_dir else "",
        "image_count": _count_images(path) if is_dir else 0,
        "fss": _find_fss(path) if is_dir else [],
        "inferred_vendor": _infer_vendor_from_text(text_for_context),
        "inferred_probe": _infer_probe_from_text(text_for_context),
        "dirs": dirs[:500],
        "dir_count": len(dirs),
    }


def _job_update(job_id: str, **patch: object) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(patch)


def _job_log(job_id: str, text: str) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        log = str(job.get("log") or "")
        job["log"] = (log + text)[-60000:]


def _run_and_log(job_id: str, cmd: List[str], cwd: Path) -> int:
    _job_log(job_id, "$ " + " ".join(cmd) + "\n")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _job_log(job_id, line)
    return int(proc.wait())


def _run_job(job_id: str, payload: Dict[str, object]) -> None:
    started = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(str(payload.get("folder") or "")).expanduser().resolve()
    run_slug = _safe_slug(str(payload.get("run_name") or folder.name))
    batch_dir = OUTPUT_ROOT / f"{started}_{run_slug}"
    run_dir = batch_dir / run_slug
    output_html = batch_dir / "autonomous_review.html"
    _job_update(
        job_id,
        status="running",
        started_at=started,
        folder=folder.as_posix(),
        batch_dir=batch_dir.as_posix(),
        output_dir=run_dir.as_posix(),
        review_url=f"/runs/{job_id}/autonomous_review.html?v={int(time.time())}",
        log="",
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            (SCRIPT_DIR / "predict_rect_depth_autonomous.py").as_posix(),
            "--folder",
            folder.as_posix(),
            "--output-dir",
            run_dir.as_posix(),
            "--max-images",
            str(int(payload.get("max_images") or 80)),
            "--ocr-timeout",
            str(float(payload.get("ocr_timeout") or 8.0)),
            "--ocr-max-side",
            str(int(payload.get("ocr_max_side") or 1800)),
            "--min-ocr-conf",
            str(float(payload.get("min_ocr_conf") or 18.0)),
            "--max-candidates-per-sample",
            str(int(payload.get("max_candidates_per_sample") or 24)),
        ]
        fss = str(payload.get("fss") or "").strip()
        if fss:
            cmd.extend(["--fss", str(Path(fss).expanduser().resolve())])
        vendor = str(payload.get("vendor") or "").strip()
        if vendor:
            cmd.extend(["--vendor", vendor])
        probe = str(payload.get("probe") or "").strip()
        if probe:
            cmd.extend(["--probe", probe])
        vendor_template = str(payload.get("vendor_template") or "").strip()
        if vendor_template:
            cmd.extend(["--vendor-template", vendor_template])
        rect_echo = str(payload.get("rect_echo") or "").strip()
        if rect_echo:
            cmd.extend(["--rect-echo", rect_echo])
        expected_depths = str(payload.get("expected_depths") or "").strip()
        if expected_depths:
            cmd.extend(["--expected-depths", expected_depths])
        scale_side = str(payload.get("scale_side_preference") or "").strip()
        if scale_side:
            cmd.extend(["--scale-side-preference", scale_side])
        if not bool(payload.get("ocr_roi_passes", True)):
            cmd.append("--no-ocr-roi-passes")

        rc = _run_and_log(job_id, cmd, REPO_ROOT)
        if rc != 0:
            _job_update(job_id, status="error", returncode=rc, finished_at=time.strftime("%Y%m%d_%H%M%S"))
            return

        build_cmd = [
            sys.executable,
            (SCRIPT_DIR / "build_rect_depth_autonomous_review_html.py").as_posix(),
            "--batch-dir",
            batch_dir.as_posix(),
            "--output-html",
            output_html.as_posix(),
            "--max-candidates",
            "8",
        ]
        rc = _run_and_log(job_id, build_cmd, REPO_ROOT)
        status = "done" if rc == 0 else "error"
        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        _job_update(
            job_id,
            status=status,
            returncode=rc,
            summary=summary,
            finished_at=time.strftime("%Y%m%d_%H%M%S"),
        )
    except Exception as exc:
        _job_log(job_id, f"\nERRORE: {exc}\n")
        _job_update(job_id, status="error", error=str(exc), finished_at=time.strftime("%Y%m%d_%H%M%S"))


INDEX_HTML = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RECT_DEPTH launcher</title>
<style>
:root{--bg:#0b0f12;--panel:#151b20;--line:#2b3640;--text:#edf4f7;--muted:#94a4af;--accent:#58b7ff;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(11,15,18,.96);border-bottom:1px solid var(--line);padding:14px 18px}h1{margin:0 0 8px;font-size:21px}
main{display:grid;grid-template-columns:minmax(320px,480px) 1fr;gap:16px;padding:16px}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.stack{display:grid;gap:10px}label span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}
input,button,select,textarea{background:#0d1216;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:inherit}button{cursor:pointer}button.primary{background:#11344a;border-color:#2d7199}button:disabled{opacity:.55;cursor:wait}
input.full{width:100%}.path-row{display:flex;gap:8px}.path-row input{flex:1}.hint{color:var(--muted);font-size:12px}.metric{background:#0d1216;border:1px solid #24303a;border-radius:7px;padding:7px 9px}.metric b{font-size:18px}
.browser{max-height:54vh;overflow:auto;border:1px solid #25313a;border-radius:7px;background:#0d1216}.dir{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;border-bottom:1px solid #202a33;padding:8px 10px}.dir:last-child{border-bottom:0}.dir button{padding:5px 8px}.bad{color:#ff9c9c}.ok{color:#8bf2aa}.warn{color:#ffd783}
.log{height:280px;overflow:auto;white-space:pre-wrap;background:#070a0d;border:1px solid #27313a;border-radius:7px;padding:10px;color:#dce8ef;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}.review-frame{width:100%;height:72vh;border:1px solid var(--line);border-radius:8px;background:#050708}.hidden{display:none}
@media(max-width:950px){main{grid-template-columns:1fr}.review-frame{height:62vh}}
</style>
</head>
<body>
<header>
  <h1>RECT_DEPTH autonomous launcher</h1>
  <div class="hint">Scegli una cartella di configurazione, lancia il riconoscimento e poi commenta i risultati nella review.</div>
</header>
<main>
  <section class="stack">
    <div class="path-row">
      <input id="path" class="full" value="/Volumes/SSD_esi1_n1" autocomplete="off">
      <button id="openPath">Apri</button>
    </div>
    <div class="row">
      <button id="parent">Su</button>
      <button id="selectFolder" class="primary">Usa questa cartella</button>
      <span id="pathState" class="hint"></span>
    </div>
    <input id="dirFilter" class="full" placeholder="Filtra cartelle, es. BK, Esaote, Hitachi, 18L5, TRT33">
    <div id="info" class="row"></div>
    <div id="browser" class="browser"></div>
  </section>
  <section class="stack">
    <div class="row">
      <div class="metric"><b id="selectedName">-</b><br><span class="hint">cartella selezionata</span></div>
      <div class="metric"><b id="imageCount">0</b><br><span class="hint">immagini</span></div>
      <div class="metric"><b id="fssCount">0</b><br><span class="hint">.fss</span></div>
    </div>
    <label><span>Cartella da analizzare</span><input id="folder" class="full" autocomplete="off"></label>
    <label><span>FSS da usare, se presente</span><select id="fss" class="full"></select></label>
    <div class="row">
      <label><span>Template vendor</span><select id="vendorTemplate"><option value="">auto/manuale</option><option>BK</option><option>Esaote</option><option>Hitachi</option><option>GE</option><option>Mindray</option><option>Philips</option><option>Canon</option><option>Alpinion</option><option>Siemens</option><option>Biopsee</option><option>Koelis</option><option>Terason</option><option>Sonostar</option><option>Vinno</option><option>ExactVu</option></select></label>
      <label><span>Vendor</span><input id="vendor" autocomplete="off" placeholder="auto"></label>
      <label><span>Sonda</span><input id="probe" autocomplete="off" placeholder="auto"></label>
      <label><span>Lato scala</span><select id="scaleSide"><option value="">profilo/auto</option><option value="right">destra</option><option value="left">sinistra</option><option value="auto">auto</option></select></label>
      <label><span>Max immagini</span><input id="maxImages" type="number" min="1" value="80"></label>
    </div>
    <div class="row">
      <label><span>Eco left</span><input id="echoLeft" type="number" min="0" step="1" placeholder="x1"></label>
      <label><span>Eco top</span><input id="echoTop" type="number" min="0" step="1" placeholder="y1"></label>
      <label><span>Eco right</span><input id="echoRight" type="number" min="0" step="1" placeholder="x2"></label>
      <label><span>Eco bottom</span><input id="echoBottom" type="number" min="0" step="1" placeholder="y2"></label>
      <label><span>Depth attese mm</span><input id="expectedDepths" autocomplete="off" placeholder="es. 20,30,40,50"></label>
    </div>
    <div class="row">
      <label><span>OCR timeout</span><input id="ocrTimeout" type="number" min="1" step="0.5" value="8"></label>
      <label><span>Max side OCR</span><input id="ocrMaxSide" type="number" min="600" step="100" value="1800"></label>
      <label><span>Min conf OCR</span><input id="minConf" type="number" min="0" step="1" value="18"></label>
      <label><span>Candidati</span><input id="maxCandidates" type="number" min="4" step="1" value="24"></label>
    </div>
    <div class="row">
      <button id="run" class="primary">Analizza cartella</button>
      <button id="openReview" disabled>Apri review</button>
      <span id="jobState" class="hint"></span>
    </div>
    <div id="log" class="log"></div>
    <iframe id="review" class="review-frame hidden"></iframe>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
let currentInfo=null, selectedInfo=null, jobId=null, pollTimer=null, allDirs=[];
function esc(s){return String(s??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]))}
async function api(path, opts={}){const res=await fetch(path, opts); if(!res.ok) throw new Error(await res.text()); return await res.json();}
function setLog(text){$('log').textContent=text||'';$('log').scrollTop=$('log').scrollHeight}
function renderBrowser(){
  const q=$('dirFilter').value.trim().toLowerCase();
  const dirs=(allDirs||[]).filter(d=>!q||[d.name,d.path].join(' ').toLowerCase().includes(q));
  $('browser').innerHTML=(currentInfo?.parent?`<div class="dir"><b>..</b><button data-path="${esc(currentInfo.parent)}">Apri</button></div>`:'')+dirs.map(d=>`<div class="dir"><div><b>${esc(d.name)}</b><div class="hint">${d.has_image_samples?'image_samples ':''}${d.has_fss?'· fss':''}</div></div><button data-path="${esc(d.path)}">Apri</button></div>`).join('');
  document.querySelectorAll('#browser button').forEach(b=>b.onclick=()=>openPath(b.dataset.path));
}
function renderInfo(info){
  currentInfo=info;$('path').value=info.path;$('pathState').textContent=info.exists&&info.is_dir?'cartella aperta':'non trovata';
  $('info').innerHTML=`<div class="metric"><b>${info.image_count||0}</b><br><span class="hint">immagini qui</span></div><div class="metric"><b>${info.fss?.length||0}</b><br><span class="hint">fss</span></div><div class="metric"><b>${esc(info.inferred_vendor||'-')}</b><br><span class="hint">vendor stimato</span></div><div class="metric"><b>${esc(info.inferred_probe||'-')}</b><br><span class="hint">sonda stimata</span></div>`;
  allDirs=info.dirs||[];
  renderBrowser();
}
async function openPath(path){try{renderInfo(await api('/api/list?path='+encodeURIComponent(path)));}catch(e){$('pathState').innerHTML='<span class="bad">'+esc(e.message)+'</span>';}}
function selectCurrent(){
  if(!currentInfo||!currentInfo.is_dir)return;
  selectedInfo=currentInfo;$('folder').value=currentInfo.path;$('selectedName').textContent=currentInfo.path.split('/').filter(Boolean).pop()||currentInfo.path;$('imageCount').textContent=currentInfo.image_count||0;$('fssCount').textContent=currentInfo.fss?.length||0;
  $('vendor').value=currentInfo.inferred_vendor||'';$('probe').value=currentInfo.inferred_probe||'';
  if ([...$('vendorTemplate').options].some(o=>o.value===currentInfo.inferred_vendor)) $('vendorTemplate').value=currentInfo.inferred_vendor||'';
  $('fss').innerHTML='<option value="">nessun fss</option>'+(currentInfo.fss||[]).map(p=>`<option value="${esc(p)}">${esc(p.split('/').pop())}</option>`).join('');
  if((currentInfo.fss||[]).length)$('fss').value=currentInfo.fss[0];
}
async function startRun(){
  const folder=$('folder').value.trim(); if(!folder){alert('Seleziona prima una cartella');return}
  const rectParts=[$('echoLeft').value,$('echoTop').value,$('echoRight').value,$('echoBottom').value].map(v=>String(v||'').trim());
  const rectEcho=rectParts.every(Boolean)?rectParts.join(','):'';
  $('run').disabled=true;$('openReview').disabled=true;$('review').classList.add('hidden');setLog('Avvio analisi...\\n');
  const payload={folder,fss:$('fss').value,vendor:$('vendor').value,probe:$('probe').value,vendor_template:$('vendorTemplate').value,scale_side_preference:$('scaleSide').value,rect_echo:rectEcho,expected_depths:$('expectedDepths').value,max_images:+$('maxImages').value||80,ocr_timeout:+$('ocrTimeout').value||8,ocr_max_side:+$('ocrMaxSide').value||1800,min_ocr_conf:+$('minConf').value||18,max_candidates_per_sample:+$('maxCandidates').value||24};
  const job=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  jobId=job.job_id;pollJob();
}
async function pollJob(){
  if(!jobId)return;
  const job=await api('/api/job?id='+encodeURIComponent(jobId));
  $('jobState').textContent=job.status||'';setLog(job.log||'');
  if(job.status==='done'){
    $('run').disabled=false;$('openReview').disabled=false;$('openReview').dataset.url=job.review_url;$('review').src=job.review_url;$('review').classList.remove('hidden');
    return;
  }
  if(job.status==='error'){
    $('run').disabled=false;$('openReview').disabled=true;return;
  }
  pollTimer=setTimeout(pollJob,1200);
}
$('openPath').onclick=()=>openPath($('path').value);$('path').addEventListener('keydown',e=>{if(e.key==='Enter')openPath($('path').value)});$('dirFilter').addEventListener('input',renderBrowser);$('parent').onclick=()=>currentInfo?.parent&&openPath(currentInfo.parent);$('selectFolder').onclick=selectCurrent;$('run').onclick=startRun;$('openReview').onclick=()=>{if($('openReview').dataset.url)location.href=$('openReview').dataset.url};
openPath($('path').value);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "RectDepthAutonomousApp/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8") or "{}")
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/list":
            query = urllib.parse.parse_qs(parsed.query)
            path = query.get("path", ["/Volumes"])[0]
            self._send_json(HTTPStatus.OK, _folder_info(path))
            return
        if parsed.path == "/api/job":
            query = urllib.parse.parse_qs(parsed.query)
            job_id = query.get("id", [""])[0]
            with JOBS_LOCK:
                payload = dict(JOBS.get(job_id) or {})
            self._send_json(HTTPStatus.OK, payload or {"status": "missing"})
            return
        if parsed.path.startswith("/runs/"):
            parts = parsed.path.split("/", 3)
            if len(parts) < 4:
                self._send(HTTPStatus.NOT_FOUND, b"missing file", "text/plain")
                return
            job_id = parts[2]
            rel = urllib.parse.unquote(parts[3])
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id) or {})
            batch_dir = Path(str(job.get("batch_dir") or ""))
            target = (batch_dir / rel).resolve()
            try:
                target.relative_to(batch_dir.resolve())
            except Exception:
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
            if not target.exists() or not target.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            content_type = mimetypes.guess_type(target.as_posix())[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/run":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            payload = self._read_json()
            folder = Path(str(payload.get("folder") or "")).expanduser()
            if not folder.exists() or not folder.is_dir():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "cartella non valida"})
                return
            job_id = f"job_{int(time.time())}_{len(JOBS) + 1}"
            _job_update(job_id, status="queued", log="")
            thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
            thread.start()
            self._send_json(HTTPStatus.OK, {"job_id": job_id, "status": "queued"})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local autonomous RECT_DEPTH launcher.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RECT_DEPTH autonomous app: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
