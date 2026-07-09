#!/usr/bin/env python3
"""Review gallery (multi-page) for marker + envelope predictions on raw acquisition folders.

Reads the CSVs produced by `predict_marker_envelopes_batch.py` and builds:

- `index.html`: one row per folder (progress, marker-found rate, review rate,
  groups) -> click opens the folder page
- `folders/NNNN.html`: ALL processed images of that folder, each with the
  detected marker (red) and the folder-level envelopes (green, same
  coordinates for the whole folder). Toggleable boxes (B), fullscreen viewer
  (arrows/Esc), per-image comments autosaved in the browser + CSV export.

No image copies: pages reference the ORIGINAL files on the volume, so the
gallery is instant to (re)build while the batch runner is still going.

Example:
  python3 tools/orientation/build_marker_envelope_review_gallery.py \
    --run-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run2 \
    --dataset-root-href "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Rect = Tuple[int, int, int, int]
GROUPS = ("NF", "LR", "UD", "LRUD")

STYLE = """
body { font-family: -apple-system, sans-serif; margin: 20px; background: #16181d; color: #e8e8e8; }
h1 { font-size: 20px; } h2 { font-size: 15px; color: #9ecbff; }
a { color: #9ecbff; }
.count { color: #999; font-weight: normal; }
.legend span { margin-right: 18px; }
.toolbar { position: sticky; top: 0; z-index: 50; background: #16181d; padding: 8px 0; }
.toolbar button { background: #2f6fed; color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font-size: 14px; cursor: pointer; margin-right: 8px; }
.toolbar .hint { color: #999; font-size: 12px; margin-left: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #333; }
tr:hover td { background: #22252c; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 14px; }
.card { background: #22252c; border-radius: 8px; padding: 10px; }
.imgwrap { position: relative; cursor: zoom-in; }
.imgwrap img { width: 100%; display: block; border-radius: 4px; }
.title { font-size: 12px; margin-bottom: 6px; color: #9ecbff; }
.meta { font-size: 12px; margin-top: 6px; color: #ccc; }
.box { position: absolute; box-sizing: border-box; pointer-events: none; }
.box.legacy { border: 2px solid #00c800; }
.box.legacy .lbl { position: absolute; top: -18px; left: 0; color: #00e000; font-size: 12px; font-weight: bold; text-shadow: 0 0 3px #000; }
.box.marker { border: 2px solid #ff2828; min-width: 10px; min-height: 10px; }
.box.rect { border: 2px solid #3c78ff; }
.box.rect .lbl { position: absolute; bottom: -18px; left: 0; color: #7ea6ff; font-size: 11px; text-shadow: 0 0 3px #000; }
.box.excl { border: 2px dashed #ffd000; }
.box.excl .lbl { position: absolute; top: -18px; left: 0; color: #ffd000; font-size: 11px; text-shadow: 0 0 3px #000; }
.sugiu { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 11px; font-weight: bold; }
.sugiu.su { background: #1e3a8a; color: #bfdbfe; }
.sugiu.giu { background: #7c2d12; color: #fed7aa; }
body.noboxes .box { display: none; }
.comment { width: 100%; box-sizing: border-box; margin-top: 8px; background: #1a1d23; color: #ffd76e;
  border: 1px solid #444; border-radius: 6px; padding: 6px 8px; font-size: 13px; font-family: inherit; resize: vertical; }
.comment:focus { outline: 1px solid #2f6fed; }
.card.commented { outline: 2px solid #ffd76e33; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; }
.badge.ok { background: #14532d; color: #86efac; }
.badge.partial { background: #713f12; color: #fde68a; }
#viewer { display: none; position: fixed; inset: 0; z-index: 100; background: rgba(8,9,12,.97); }
#viewer.open { display: flex; flex-direction: column; }
#viewer .stage { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }
#viewer .frame { position: relative; }
#viewer .frame img { max-width: 96vw; max-height: 86vh; display: block; }
#viewer .bar { padding: 10px 16px; font-size: 13px; color: #ddd; display: flex; gap: 16px; align-items: center; }
#viewer .bar button { background: #333; color: #fff; border: 0; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
"""

SCRIPT = """
function renderBoxes(container, boxes) {
  container.querySelectorAll('.box').forEach(b => b.remove());
  boxes.forEach(b => {
    const div = document.createElement('div');
    div.className = 'box ' + b.kind;
    div.style.top = b.t + '%'; div.style.left = b.l + '%';
    div.style.height = b.h + '%'; div.style.width = b.w + '%';
    if (b.label) { const s = document.createElement('span'); s.className = 'lbl'; s.textContent = b.label; div.appendChild(s); }
    container.appendChild(div);
  });
}
const wraps = Array.from(document.querySelectorAll('.imgwrap'));
wraps.forEach(w => renderBoxes(w, JSON.parse(w.dataset.boxes)));

const STORE = 'marker_envelope_gallery_comments_v1';
let comments = {};
try { comments = JSON.parse(localStorage.getItem(STORE) || '{}'); } catch (e) {}
const cards = Array.from(document.querySelectorAll('.card'));
const vComment = document.getElementById('vComment');
function setComment(key, value) {
  if (value.trim()) comments[key] = value; else delete comments[key];
  localStorage.setItem(STORE, JSON.stringify(comments));
  const card = cards.find(c => c.dataset.key === key);
  if (card) {
    const ta = card.querySelector('.comment');
    if (ta.value !== value) ta.value = value;
    card.classList.toggle('commented', !!value.trim());
  }
}
cards.forEach(c => {
  const ta = c.querySelector('.comment');
  ta.value = comments[c.dataset.key] || '';
  c.classList.toggle('commented', !!ta.value.trim());
  ta.addEventListener('input', () => setComment(c.dataset.key, ta.value));
});
const exportBtn = document.getElementById('exportComments');
if (exportBtn) exportBtn.onclick = () => {
  const lines = [['folder', 'image_id', 'comment']];
  Object.keys(comments).forEach(k => {
    const v = comments[k].trim();
    if (!v) return;
    const i = k.indexOf('::');
    lines.push([k.slice(0, i), k.slice(i + 2), v]);
  });
  const csv = lines.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['\\ufeff' + csv], {type: 'text/csv;charset=utf-8'}));
  a.download = 'commenti_marker_envelope.csv';
  a.click();
};

const body = document.body;
const toggleBtn = document.getElementById('toggleBoxes');
function setBoxes(visible) {
  body.classList.toggle('noboxes', !visible);
  if (toggleBtn) toggleBtn.textContent = visible ? 'Nascondi box' : 'Mostra box';
}
if (toggleBtn) toggleBtn.onclick = () => setBoxes(body.classList.contains('noboxes'));

const viewer = document.getElementById('viewer');
const viewerImg = document.getElementById('viewerImg');
const frame = viewer ? viewer.querySelector('.frame') : null;
const vTitle = document.getElementById('vTitle');
let current = -1;
function openViewer(i) {
  current = (i + wraps.length) % wraps.length;
  const w = wraps[current];
  viewerImg.src = w.dataset.src;
  renderBoxes(frame, JSON.parse(w.dataset.boxes));
  const card = w.closest('.card');
  vTitle.textContent = (current + 1) + '/' + wraps.length + ' — ' + card.querySelector('.title').innerText;
  const vMeta = document.getElementById('vMeta');
  if (vMeta) vMeta.innerHTML = card.querySelector('.meta').innerHTML;
  vComment.value = comments[card.dataset.key] || '';
  viewer.classList.add('open');
}
function closeViewer() { viewer.classList.remove('open'); current = -1; }
wraps.forEach((w, i) => w.addEventListener('click', () => openViewer(i)));
if (viewer) {
  document.getElementById('vClose').onclick = closeViewer;
  document.getElementById('vPrev').onclick = () => openViewer(current - 1);
  document.getElementById('vNext').onclick = () => openViewer(current + 1);
  viewer.querySelector('.stage').addEventListener('click', e => { if (e.target === e.currentTarget) closeViewer(); });
  vComment.addEventListener('input', () => {
    if (current >= 0) setComment(wraps[current].closest('.card').dataset.key, vComment.value);
  });
}
document.addEventListener('keydown', e => {
  const typing = e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT';
  if (e.key === 'Escape' && current >= 0) { e.target.blur(); closeViewer(); return; }
  if (typing) return;
  if (e.key === 'b' || e.key === 'B') setBoxes(body.classList.contains('noboxes'));
  if (current < 0) return;
  if (e.key === 'ArrowLeft') openViewer(current - 1);
  if (e.key === 'ArrowRight') openViewer(current + 1);
});
"""


def _parse_box(text: str) -> Optional[Rect]:
    if not text:
        return None
    parts = [int(float(p)) for p in str(text).split("|")]
    return (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None


def _parse_box_prefix(text: str) -> Optional[Rect]:
    """First 4 integers of a pipe-separated .fss-style value (e.g. line #11/#13)."""
    if not text:
        return None
    values: List[int] = []
    for part in str(text).split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(float(part)))
        except ValueError:
            break
        if len(values) == 4:
            return (values[0], values[1], values[2], values[3])
    return None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def _file_href(root: str, folder: str, image_id: str) -> str:
    full = f"{root}/{folder}/{image_id}"
    return "file://" + urllib.parse.quote(full)


def _union(boxes: List[Rect]) -> Rect:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _page(title: str, toolbar: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{STYLE}</style></head><body>
<h1>{html.escape(title)}</h1>
{toolbar}
{content}
<div id="viewer">
  <div class="stage"><div class="frame"><img id="viewerImg"></div></div>
  <div class="bar">
    <button id="vClose">Chiudi (Esc)</button>
    <button id="vPrev">← Prec</button>
    <button id="vNext">Succ →</button>
    <span id="vTitle"></span>
    <input id="vComment" placeholder="Commento..." style="flex:1; background:#1a1d23; color:#ffd76e; border:1px solid #444; border-radius:6px; padding:6px 8px; font-size:13px;">
  </div>
  <div class="bar" style="padding-top:0"><span id="vMeta"></span></div>
</div>
<script>{SCRIPT}</script>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root-href", type=str, required=True,
                        help="Dataset root path AS SEEN BY THE BROWSER (e.g. /Volumes/SSD_esi1_n3/ACQUISITION ELABORATION).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/review_gallery")
    parser.add_argument("--official-run-dir", type=Path, default=None,
                        help="Output dir of predict_fss_head_from_acquisitions.py: merges per-image rect + "
                             "SU/GIU (su_giu_per_image_predictions.csv) and folder line #11/#13 boxes.")
    args = parser.parse_args()

    out = args.output_dir or (args.run_dir / "review_gallery")
    (out / "folders").mkdir(parents=True, exist_ok=True)
    image_rows = _load_csv(args.run_dir / "per_image_predictions.csv")
    envelope_rows = _load_csv(args.run_dir / "folder_envelopes.csv")
    folder_rows = {str(r["folder"]): r for r in _load_csv(args.run_dir / "folder_summary.csv")}

    rows_by_folder: Dict[str, List[Dict[str, str]]] = {}
    for row in image_rows:
        rows_by_folder.setdefault(str(row["folder"]), []).append(row)
    env_by_folder: Dict[str, List[Dict[str, str]]] = {}
    for row in envelope_rows:
        env_by_folder.setdefault(str(row["folder"]), []).append(row)

    # Optional merge with the official-stages outputs (official_stages_batch.py).
    sugiu_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    official_folder: Dict[str, Dict[str, str]] = {}
    if args.official_run_dir:
        for row in _load_csv(args.official_run_dir / "official_per_image.csv"):
            sugiu_by_key[(str(row.get("folder", "")), str(row.get("image_id", "")))] = row
        for row in _load_csv(args.official_run_dir / "official_folder.csv"):
            official_folder[str(row.get("folder", ""))] = row

    href_root = args.dataset_root_href.rstrip("/")
    index_lines: List[str] = []
    page_count = 0
    for fid, (folder, rows) in enumerate(sorted(rows_by_folder.items())):
        info = folder_rows.get(folder, {})
        complete = info.get("status") == "ok"
        vendor = rows[0].get("vendor", "")
        size_txt = str(rows[0].get("image_size", ""))
        try:
            width, height = (int(v) for v in size_txt.split("x"))
        except ValueError:
            width = height = 0

        # Envelopes: final ones if complete, otherwise live-computed from current rows.
        envelopes: List[Tuple[str, Rect]] = []
        if complete and folder in env_by_folder:
            for env in env_by_folder[folder]:
                box = _parse_box(env["envelope_box"])
                if box:
                    envelopes.append((str(env["group"]), box))
        else:
            markers_by_group: Dict[str, List[Rect]] = {}
            for r in rows:
                box = _parse_box(r.get("marker_box_abs", ""))
                if box and r.get("status") == "ok" and r.get("pred_group"):
                    markers_by_group.setdefault(str(r["pred_group"]), []).append(box)
            envelopes = [(g, _union(bs)) for g, bs in sorted(markers_by_group.items())]

        def norm(rect: Rect) -> Dict[str, float]:
            top, left, bottom, right = rect
            return {
                "t": round(100.0 * top / height, 3) if height else 0,
                "l": round(100.0 * left / width, 3) if width else 0,
                "h": round(100.0 * max(1, bottom - top + 1) / height, 3) if height else 0,
                "w": round(100.0 * max(1, right - left + 1) / width, 3) if width else 0,
            }

        env_boxes_payload = [{"kind": "legacy", "label": g, **norm(b)} for g, b in envelopes]

        # Folder-level official boxes (line #11 rect, line #13 vendor-name template).
        folder_official = official_folder.get(folder, {})
        line11_box = _parse_box_prefix(str(folder_official.get("line11_final", "")))
        line13_box = _parse_box_prefix(str(folder_official.get("line13_text", "")))
        if line11_box:
            env_boxes_payload.append({"kind": "rect", "label": "#11 " + str(folder_official.get("line11_method", "")), **norm(line11_box)})
        if line13_box:
            env_boxes_payload.append({"kind": "excl", "label": "#13 vendor", **norm(line13_box)})

        cards: List[str] = []
        n_marker = 0
        for row in rows:
            marker = _parse_box(row.get("marker_box_abs", ""))
            if marker:
                n_marker += 1
            boxes = list(env_boxes_payload)
            sugiu = sugiu_by_key.get((folder, str(row["image_id"])))
            sugiu_html = ""
            if sugiu:
                rect_img = _parse_box(str(sugiu.get("rect_box_abs", "")))
                if rect_img and rect_img != line11_box:
                    boxes.append({"kind": "rect", "label": "rect img", **norm(rect_img)})
                label = str(sugiu.get("sugiu_label", "")).lower()
                conf = str(sugiu.get("sugiu_conf", ""))[:4]
                if label in ("su", "giu"):
                    sugiu_html = f" | <span class='sugiu {label}'>{label.upper()} {conf}</span>"
            if marker:
                boxes.append({"kind": "marker", "label": "", **norm(marker)})
            href = _file_href(href_root, folder, str(row["image_id"]))
            meta = (
                f"pred: <b>{html.escape(row.get('pred_group') or '-')}</b> | "
                f"score: {html.escape((row.get('match_score') or '')[:6])} | "
                f"scope: {html.escape(row.get('search_scope') or '-')} | "
                f"status: {html.escape(row.get('status') or '-')}"
                + (f" | reason: {html.escape(row['review_reason'])}" if row.get("review_reason") else "")
                + sugiu_html
                + ("" if marker else " | <b style='color:#f87171'>NESSUN MARKER</b>")
            )
            comment_key = f"{folder}::{row['image_id']}"
            cards.append(
                f"<div class='card' data-key='{html.escape(comment_key, quote=True)}'>"
                f"<div class='title'>{html.escape(str(row['image_id']))}</div>"
                f"<div class='imgwrap' data-boxes='{html.escape(json.dumps(boxes), quote=True)}' data-src='{html.escape(href, quote=True)}'>"
                f"<img loading='lazy' src='{html.escape(href, quote=True)}'></div>"
                f"<div class='meta'>{meta}</div>"
                "<textarea class='comment' placeholder='Commento...' rows='2'></textarea>"
                "</div>"
            )

        env_txt = ", ".join(g for g, _ in envelopes) or "-"
        badge = (
            "<span class='badge ok'>completata</span>" if complete else "<span class='badge partial'>in corso</span>"
        )
        toolbar = (
            "<div class='toolbar'><a href='../index.html'>← Indice</a> "
            "<button id='toggleBoxes'>Nascondi box</button>"
            "<button id='exportComments'>Esporta commenti CSV</button>"
            "<span class='legend'><span style='color:#00c800'>■ envelope cartella</span>"
            "<span style='color:#ff2828'>■ marker trovato</span>"
            "<span style='color:#3c78ff'>■ rect ecografico (#11 / per-immagine)</span>"
            "<span style='color:#ffd000'>▨ template vendor #13</span></span>"
            "<span class='hint'>Click = tutto schermo · B = box on/off · ←/→ · Esc</span></div>"
            f"<p>Vendor: <b>{html.escape(str(vendor))}</b> · {len(rows)} immagini ({size_txt}) · "
            f"marker trovati: {n_marker}/{len(rows)} · envelope: {html.escape(env_txt)} · {badge}</p>"
        )
        page_name = f"folders/{fid:04d}.html"
        (out / page_name).write_text(
            _page(folder, toolbar, "<div class='grid'>" + "".join(cards) + "</div>"), encoding="utf-8"
        )
        page_count += 1

        index_lines.append(
            f"<tr><td><a href='{page_name}'>{html.escape(folder)}</a></td>"
            f"<td>{html.escape(str(vendor))}</td><td>{len(rows)}</td>"
            f"<td>{n_marker}/{len(rows)}</td>"
            f"<td>{html.escape(str(info.get('review_rate', '')))}</td>"
            f"<td>{html.escape(env_txt)}</td><td>{badge}</td></tr>"
        )

    index_content = (
        "<p>Click su una cartella per vedere tutte le sue immagini con marker ed envelope. "
        "Le cartelle 'in corso' mostrano i risultati parziali già disponibili.</p>"
        "<table><tr><th>Cartella</th><th>Vendor</th><th>Immagini</th><th>Marker trovati</th>"
        "<th>Review</th><th>Envelope</th><th>Stato</th></tr>"
        + "".join(index_lines)
        + "</table>"
    )
    (out / "index.html").write_text(
        _page("Review marker + envelope — indice cartelle", "", index_content), encoding="utf-8"
    )
    print(f"Gallery: {out / 'index.html'} ({page_count} cartelle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
