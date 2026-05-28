#!/usr/bin/env python3
"""Small local HTML reviewer for SU/GIU-driven LR marker manifests."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Rect = Tuple[int, int, int, int]


def _int(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _rect(row: Dict[str, str], prefix: str) -> Optional[Rect]:
    vals = [
        _int(row.get(f"{prefix}_top_crop", row.get(f"{prefix}_top_abs", ""))),
        _int(row.get(f"{prefix}_left_crop", row.get(f"{prefix}_left_abs", ""))),
        _int(row.get(f"{prefix}_bottom_crop", row.get(f"{prefix}_bottom_abs", ""))),
        _int(row.get(f"{prefix}_right_crop", row.get(f"{prefix}_right_abs", ""))),
    ]
    if any(v is None for v in vals):
        return None
    return tuple(int(v) for v in vals)  # type: ignore[return-value]


def _abs_rect(row: Dict[str, str], prefix: str) -> Optional[Rect]:
    vals = [
        _int(row.get(f"{prefix}_top_abs", "")),
        _int(row.get(f"{prefix}_left_abs", "")),
        _int(row.get(f"{prefix}_bottom_abs", "")),
        _int(row.get(f"{prefix}_right_abs", "")),
    ]
    if any(v is None for v in vals):
        return None
    return tuple(int(v) for v in vals)  # type: ignore[return-value]


def _box(rect: Optional[Rect]) -> Optional[Dict[str, int]]:
    if rect is None:
        return None
    top, left, bottom, right = rect
    return {"top": top, "left": left, "bottom": bottom, "right": right, "width": right - left + 1, "height": bottom - top + 1}


def _sample_id(row: Dict[str, str], idx: int) -> str:
    return row.get("sample_id", "") or f"row_{idx + 1:08d}"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _load_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = row.get("sample_id", "")
            if sid:
                out[sid] = dict(row)
    return out


def _write_decisions(path: Path, decisions: Dict[str, Dict[str, str]]) -> None:
    fields = ["sample_id", "decision", "note", "updated_at_epoch"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for sid in sorted(decisions):
            writer.writerow({k: decisions[sid].get(k, "") for k in fields})


def _counts(rows: List[Dict[str, str]], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = row.get(field, "") or "(empty)"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _decision_counts(rows: List[Dict[str, str]], decisions: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    out = {"missing": 0, "accepted": 0, "rejected": 0, "needs_fix": 0}
    for i, row in enumerate(rows):
        value = decisions.get(_sample_id(row, i), {}).get("decision", "")
        out[value or "missing"] = out.get(value or "missing", 0) + 1
    return out


def _matches(row: Dict[str, str], idx: int, decisions: Dict[str, Dict[str, str]], filters: Dict[str, str]) -> bool:
    mode = filters.get("mode", "review")
    decision = decisions.get(_sample_id(row, idx), {}).get("decision", "")
    if mode == "review" and row.get("label_kind", "") == "auto_candidate":
        return False
    if mode == "auto" and row.get("label_kind", "") != "auto_candidate":
        return False
    if mode == "decided" and not decision:
        return False
    for key, col in [("manufacturer", "manufacturer"), ("kind", "label_kind"), ("reason", "review_reason"), ("sugiu", "su_giu_pred"), ("lr", "lr_label")]:
        value = filters.get(key, "")
        if value and row.get(col, "") != value:
            return False
    q = filters.get("q", "").lower()
    if q:
        blob = " ".join(row.get(k, "") for k in ("sample_id", "image_name", "config_folder", "review_reason")).lower()
        if q not in blob:
            return False
    return True


def _crop_bytes(row: Dict[str, str]) -> bytes:
    path = Path(row.get("image_path", "")).expanduser().resolve()
    echo = _abs_rect(row, "echo_rect")
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if echo is not None:
            top, left, bottom, right = echo
            rgb = rgb.crop((left, top, right + 1, bottom + 1))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


HTML = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LR Marker SU/GIU Review</title>
<style>
body{margin:0;background:#f5f6f8;color:#182230;font:14px system-ui,-apple-system,Segoe UI,sans-serif}
header,.filters{background:#fff;border-bottom:1px solid #d0d5dd;padding:10px 14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0 16px 0 0}select,input,button,textarea{font:inherit;border:1px solid #cfd6e0;border-radius:6px;background:#fff;padding:7px}
button{cursor:pointer}.primary{background:#2563eb;border-color:#2563eb;color:#fff}.good{background:#16803c;border-color:#16803c;color:#fff}.bad{background:#c3322d;border-color:#c3322d;color:#fff}.warn{background:#b7791f;border-color:#b7791f;color:#fff}
main{display:grid;grid-template-columns:minmax(0,1fr)360px;gap:12px;padding:12px}
.panel{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:10px}.wrap{position:relative;background:#111827;overflow:auto;max-height:calc(100vh - 210px);border-radius:6px}
img{display:block;width:100%;height:auto}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.kv{display:grid;grid-template-columns:120px 1fr;gap:5px 8px}.kv div:nth-child(odd){color:#667085}.kv div:nth-child(even){font:12px ui-monospace,Menlo,monospace;overflow-wrap:anywhere}textarea{width:100%;min-height:70px;margin-top:10px}.muted{color:#667085}.legend{font-size:12px;color:#667085;margin-bottom:8px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>LR Marker SU/GIU Review</h1><span id="status" class="muted"></span></header>
<section class="filters">
<select id="mode"><option value="review" selected>Da rivedere</option><option value="auto">Auto candidati</option><option value="all">Tutti</option><option value="decided">Decisi</option></select>
<select id="manufacturer"></select><select id="kind"></select><select id="reason"></select><select id="sugiu"></select><select id="lr"></select><input id="q" placeholder="cerca">
</section>
<main><section class="panel"><div class="wrap"><img id="img"><canvas id="canvas"></canvas></div><div class="bar"><button id="prev">← Prec</button><button id="next" class="primary">Succ →</button><span id="pos" class="muted"></span></div></section>
<aside class="panel"><div class="legend">verde: metà LR · blu: ROI su/giu · rosso: marker rilevato</div><div id="meta" class="kv"></div><textarea id="note" placeholder="nota"></textarea><div class="bar"><button class="good" id="accept">Accetta</button><button class="warn" id="fix">Da correggere</button><button class="bad" id="reject">Scarta</button></div><div id="msg" class="muted"></div></aside></main>
<script>
let S={i:0,total:0,sample:null,meta:null};
const $=id=>document.getElementById(id);
async function J(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw Error(d.error||'errore');return d}
function qs(){return new URLSearchParams({mode:$('mode').value,manufacturer:$('manufacturer').value,kind:$('kind').value,reason:$('reason').value,sugiu:$('sugiu').value,lr:$('lr').value,q:$('q').value})}
function fill(id,label,vals){const e=$(id),cur=e.value;e.innerHTML='<option value="">'+label+'</option>';vals.forEach(v=>e.innerHTML+='<option>'+v+'</option>');if(vals.includes(cur))e.value=cur}
async function meta(){S.meta=await J('/api/meta');fill('manufacturer','Vendor',S.meta.manufacturers);fill('kind','Tipo',S.meta.kinds);fill('reason','Motivo',S.meta.reasons);fill('sugiu','SU/GIU',S.meta.sugiu);fill('lr','LR label',S.meta.lr);$('status').textContent=`${S.meta.total} righe · decisioni ${JSON.stringify(S.meta.decisions)}`}
function esc(v){return String(v||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
async function load(i){const p=qs();p.set('index',Math.max(0,i));const d=await J('/api/sample?'+p);S.i=d.index;S.total=d.total;S.sample=d.sample;$('pos').textContent=d.total?`${d.index+1}/${d.total}`:'nessun campione';if(!d.sample)return;const im=$('img');im.onload=draw;im.src=d.sample.image_url+'&t='+Date.now();renderMeta(d.sample)}
function renderMeta(s){let rows=[['sample',s.sample_id],['vendor',s.manufacturer],['kind',s.label_kind],['reason',s.review_reason],['su/giu',`${s.su_giu_pred} conf ${s.su_giu_conf}`],['filename vert',s.filename_vertical_label],['LR finale',s.lr_label_it||s.lr_label],['side',s.detected_marker_side],['score',s.match_score],['file LR hint',s.filename_lr_label],['mismatch file LR',s.filename_lr_mismatch],['config',s.config_folder],['file',s.image_name]];$('meta').innerHTML=rows.map(r=>`<div>${r[0]}</div><div>${esc(r[1])}</div>`).join('');$('note').value=s.decision?.note||''}
function rect(ctx,b,color,w,dash){if(!b)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=w;if(dash)ctx.setLineDash(dash);ctx.strokeRect(b.left+.5,b.top+.5,Math.max(1,b.right-b.left),Math.max(1,b.bottom-b.top));ctx.restore()}
function draw(){const s=S.sample,im=$('img'),c=$('canvas');if(!s||!im.naturalWidth)return;c.width=im.naturalWidth;c.height=im.naturalHeight;const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.strokeStyle='#22c55e';x.lineWidth=2;x.beginPath();x.moveTo(c.width/2,0);x.lineTo(c.width/2,c.height);x.stroke();rect(x,s.boxes.roi,'#3b82f6',3,[8,5]);rect(x,s.boxes.marker,'#ef4444',3,null)}
async function decide(v){if(!S.sample)return;const d=await J('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:S.sample.sample_id,decision:v,note:$('note').value})});$('msg').textContent=d.message;await meta();await load(S.i)}
['mode','manufacturer','kind','reason','sugiu','lr'].forEach(id=>$(id).onchange=()=>load(0));$('q').oninput=()=>{clearTimeout(S.t);S.t=setTimeout(()=>load(0),250)};$('prev').onclick=()=>load(S.i-1);$('next').onclick=()=>load(S.i+1);$('accept').onclick=()=>decide('accepted');$('fix').onclick=()=>decide('needs_fix');$('reject').onclick=()=>decide('rejected');window.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight')load(S.i+1);if(e.key==='ArrowLeft')load(S.i-1);if(e.key.toLowerCase()==='a')decide('accepted');if(e.key.toLowerCase()==='r')decide('rejected');if(e.key.toLowerCase()==='f')decide('needs_fix')};
meta().then(()=>load(0)).catch(e=>$('status').textContent=e.message);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    rows: List[Dict[str, str]]
    decisions: Dict[str, Dict[str, str]]
    decisions_path: Path

    def _json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/meta":
            self._json({
                "total": len(self.rows),
                "manufacturers": sorted({r.get("manufacturer", "") for r in self.rows if r.get("manufacturer", "")}),
                "kinds": sorted({r.get("label_kind", "") for r in self.rows if r.get("label_kind", "")}),
                "reasons": sorted({r.get("review_reason", "") for r in self.rows if r.get("review_reason", "")}),
                "sugiu": sorted({r.get("su_giu_pred", "") for r in self.rows if r.get("su_giu_pred", "")}),
                "lr": sorted({r.get("lr_label", "") for r in self.rows if r.get("lr_label", "")}),
                "decisions": _decision_counts(self.rows, self.decisions),
                "label_kind_counts": _counts(self.rows, "label_kind"),
                "reason_counts": _counts(self.rows, "review_reason"),
            })
            return
        if parsed.path == "/api/sample":
            filters = {k: (q.get(k, [""])[0] or "") for k in ("mode", "manufacturer", "kind", "reason", "sugiu", "lr", "q")}
            idxs = [i for i, r in enumerate(self.rows) if _matches(r, i, self.decisions, filters)]
            req = _int(q.get("index", ["0"])[0]) or 0
            if not idxs:
                self._json({"total": 0, "index": 0, "sample": None})
                return
            req = max(0, min(req, len(idxs) - 1))
            i = idxs[req]
            row = self.rows[i]
            sid = _sample_id(row, i)
            self._json({"total": len(idxs), "index": req, "sample": {
                **row,
                "sample_id": sid,
                "image_url": f"/api/crop?sample_id={quote(sid)}",
                "boxes": {"roi": _box(_rect(row, "roi")), "marker": _box(_rect(row, "marker"))},
                "decision": self.decisions.get(sid, {}),
            }})
            return
        if parsed.path == "/api/crop":
            sid = unquote(q.get("sample_id", [""])[0])
            found = next((r for i, r in enumerate(self.rows) if _sample_id(r, i) == sid), None)
            if found is None:
                self._json({"error": "sample not found"}, 404)
                return
            try:
                data = _crop_bytes(found)
            except Exception as exc:  # pylint: disable=broad-except
                self._json({"error": str(exc)}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        sid = str(payload.get("sample_id", "")).strip()
        decision = str(payload.get("decision", "")).strip()
        note = str(payload.get("note", "") or "").strip()
        if decision not in {"accepted", "rejected", "needs_fix"}:
            self._json({"error": "invalid decision"}, 400)
            return
        self.decisions[sid] = {"sample_id": sid, "decision": decision, "note": note, "updated_at_epoch": f"{time.time():.3f}"}
        _write_decisions(self.decisions_path, self.decisions)
        self._json({"ok": True, "message": f"salvato: {decision}"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve() if args.decisions else manifest.with_name("review_decisions.csv")
    Handler.rows = _load_csv(manifest)
    Handler.decisions_path = decisions_path
    Handler.decisions = _load_decisions(decisions_path)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Reviewer: http://{args.host}:{args.port}", flush=True)
    print(f"Manifest rows: {len(Handler.rows)}", flush=True)
    print(f"Decisions: {decisions_path}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
