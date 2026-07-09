#!/usr/bin/env python3
"""Review gallery for marker + envelope predictions on raw acquisition folders.

Reads the CSVs produced by `predict_marker_envelopes_batch.py` and builds an
interactive HTML gallery, one section per folder:

- red box = detected marker (per image)
- green boxes = folder-level envelopes (same coordinates for the whole folder,
  one per orientation group, containing all accepted marker positions)
- toggleable boxes (button / key B), fullscreen viewer (arrows, Esc)
- per-image comments, autosaved in the browser, exportable as CSV

Rebuild it at any time while the batch runner is still running (--resume).

Example:
  python3 tools/orientation/build_marker_envelope_review_gallery.py \
    --run-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run1 \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

Rect = Tuple[int, int, int, int]


def _parse_box(text: str) -> Optional[Rect]:
    if not text:
        return None
    parts = [int(float(p)) for p in str(text).split("|")]
    return (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/review_gallery")
    parser.add_argument("--max-images-per-folder", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    args = parser.parse_args()

    out = args.output_dir or (args.run_dir / "review_gallery")
    image_rows = _load_csv(args.run_dir / "per_image_predictions.csv")
    envelope_rows = _load_csv(args.run_dir / "folder_envelopes.csv")
    folder_rows = _load_csv(args.run_dir / "folder_summary.csv")

    envelopes_by_folder: Dict[str, List[Dict[str, str]]] = {}
    for row in envelope_rows:
        envelopes_by_folder.setdefault(str(row["folder"]), []).append(row)
    rows_by_folder: Dict[str, List[Dict[str, str]]] = {}
    for row in image_rows:
        rows_by_folder.setdefault(str(row["folder"]), []).append(row)

    sections: List[str] = []
    total_cards = 0
    for folder_info in folder_rows:
        folder = str(folder_info["folder"])
        if folder_info.get("status") != "ok":
            continue
        rows = rows_by_folder.get(folder, [])
        # Prefer images with a detected marker; fill with review cases.
        with_marker = [r for r in rows if r.get("marker_box_abs")]
        without = [r for r in rows if not r.get("marker_box_abs")]
        selected = (with_marker + without)[: args.max_images_per_folder]

        env_boxes = []
        for env in envelopes_by_folder.get(folder, []):
            box = _parse_box(env["envelope_box"])
            if box:
                env_boxes.append((str(env["group"]), box, env.get("resolution", "")))

        cards: List[str] = []
        for row in selected:
            image_path = args.dataset_root / folder / str(row["image_id"])
            if not image_path.is_file():
                continue
            img_rel = Path("images") / f"{total_cards:05d}.jpg"
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    width, height = rgb.size
                    if rgb.width > args.max_width:
                        scale = args.max_width / rgb.width
                        rgb = rgb.resize((args.max_width, int(rgb.height * scale)))
                    (out / img_rel).parent.mkdir(parents=True, exist_ok=True)
                    rgb.save(out / img_rel, quality=args.jpeg_quality)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {image_path}: {exc}")
                continue
            total_cards += 1

            def norm(rect: Rect) -> Dict[str, float]:
                top, left, bottom, right = rect
                return {
                    "t": round(100.0 * top / height, 3),
                    "l": round(100.0 * left / width, 3),
                    "h": round(100.0 * max(1, bottom - top + 1) / height, 3),
                    "w": round(100.0 * max(1, right - left + 1) / width, 3),
                }

            boxes: List[Dict[str, object]] = []
            size_txt = f"{width}x{height}"
            for group, box, resolution in env_boxes:
                if resolution and resolution != size_txt:
                    continue  # envelope valid only on the dominant resolution
                boxes.append({"kind": "legacy", "label": group, **norm(box)})
            marker = _parse_box(row.get("marker_box_abs", ""))
            if marker:
                boxes.append({"kind": "marker", "label": "", **norm(marker)})

            meta = (
                f"pred: <b>{html.escape(row.get('pred_group') or '-')}</b> | "
                f"score: {html.escape((row.get('match_score') or '')[:6])} | "
                f"scope: {html.escape(row.get('search_scope') or '-')} | "
                f"status: {html.escape(row.get('status') or '-')}"
                + (f" | reason: {html.escape(row['review_reason'])}" if row.get("review_reason") else "")
            )
            comment_key = f"{folder}::{row['image_id']}"
            cards.append(
                f"<div class='card' data-key='{html.escape(comment_key, quote=True)}' data-category='{html.escape(folder, quote=True)}'>"
                f"<div class='title'>{html.escape(str(row['image_id']))}</div>"
                f"<div class='imgwrap' data-boxes='{html.escape(json.dumps(boxes), quote=True)}' data-src='{img_rel.as_posix()}'>"
                f"<img loading='lazy' src='{img_rel.as_posix()}'></div>"
                f"<div class='meta'>{meta}</div>"
                "<textarea class='comment' placeholder='Commento...' rows='2'></textarea>"
                "</div>"
            )

        env_txt = ", ".join(f"{g}" for g, _, _ in env_boxes) or "nessun envelope"
        sections.append(
            f"<h2>{html.escape(folder)} <span class='count'>[{html.escape(str(folder_info.get('vendor', '')))}] "
            f"— envelope: {html.escape(env_txt)} — review {html.escape(str(folder_info.get('review_rate', '')))}</span></h2>"
            + ("<div class='grid'>" + "".join(cards) + "</div>" if cards else "<p>Nessuna immagine.</p>")
        )

    page = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>Review marker + envelope (acquisizioni raw)</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #16181d; color: #e8e8e8; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 34px; border-bottom: 1px solid #444; padding-bottom: 6px; }}
.count {{ color: #999; font-weight: normal; }}
.legend span {{ margin-right: 18px; }}
.toolbar {{ position: sticky; top: 0; z-index: 50; background: #16181d; padding: 8px 0; }}
.toolbar button {{ background: #2f6fed; color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font-size: 14px; cursor: pointer; margin-right: 8px; }}
.toolbar .hint {{ color: #999; font-size: 12px; margin-left: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 14px; }}
.card {{ background: #22252c; border-radius: 8px; padding: 10px; }}
.imgwrap {{ position: relative; cursor: zoom-in; }}
.imgwrap img {{ width: 100%; display: block; border-radius: 4px; }}
.title {{ font-size: 12px; margin-bottom: 6px; color: #9ecbff; }}
.meta {{ font-size: 12px; margin-top: 6px; color: #ccc; }}
.box {{ position: absolute; box-sizing: border-box; pointer-events: none; }}
.box.legacy {{ border: 2px solid #00c800; }}
.box.legacy .lbl {{ position: absolute; top: -18px; left: 0; color: #00e000; font-size: 12px; font-weight: bold; text-shadow: 0 0 3px #000; }}
.box.marker {{ border: 2px solid #ff2828; min-width: 10px; min-height: 10px; }}
body.noboxes .box {{ display: none; }}
.comment {{ width: 100%; box-sizing: border-box; margin-top: 8px; background: #1a1d23; color: #ffd76e;
  border: 1px solid #444; border-radius: 6px; padding: 6px 8px; font-size: 13px; font-family: inherit; resize: vertical; }}
.comment:focus {{ outline: 1px solid #2f6fed; }}
.card.commented {{ outline: 2px solid #ffd76e33; }}
#viewer {{ display: none; position: fixed; inset: 0; z-index: 100; background: rgba(8,9,12,.97); }}
#viewer.open {{ display: flex; flex-direction: column; }}
#viewer .stage {{ flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
#viewer .frame {{ position: relative; }}
#viewer .frame img {{ max-width: 96vw; max-height: 88vh; display: block; }}
#viewer .bar {{ padding: 10px 16px; font-size: 13px; color: #ddd; display: flex; gap: 16px; align-items: center; }}
#viewer .bar button {{ background: #333; color: #fff; border: 0; border-radius: 6px; padding: 6px 12px; cursor: pointer; }}
</style></head><body>
<h1>Review marker + envelope — acquisizioni raw</h1>
<div class="toolbar">
<button id="toggleBoxes">Nascondi box</button>
<button id="exportComments">Esporta commenti CSV</button>
<span class="legend"><span style="color:#00c800">■ envelope cartella (per orientamento)</span>
<span style="color:#ff2828">■ marker trovato</span></span>
<span class="hint">Click = tutto schermo · B = box on/off · ←/→ = scorri · Esc = chiudi · I commenti si salvano da soli</span>
</div>
{''.join(sections)}
<div id="viewer">
  <div class="stage"><div class="frame"><img id="viewerImg"></div></div>
  <div class="bar">
    <button id="vClose">Chiudi (Esc)</button>
    <button id="vPrev">← Prec</button>
    <button id="vNext">Succ →</button>
    <span id="vTitle"></span>
    <input id="vComment" placeholder="Commento..." style="flex:1; background:#1a1d23; color:#ffd76e; border:1px solid #444; border-radius:6px; padding:6px 8px; font-size:13px;">
  </div>
</div>
<script>
function renderBoxes(container, boxes) {{
  container.querySelectorAll('.box').forEach(b => b.remove());
  boxes.forEach(b => {{
    const div = document.createElement('div');
    div.className = 'box ' + b.kind;
    div.style.top = b.t + '%'; div.style.left = b.l + '%';
    div.style.height = b.h + '%'; div.style.width = b.w + '%';
    if (b.label) {{ const s = document.createElement('span'); s.className = 'lbl'; s.textContent = b.label; div.appendChild(s); }}
    container.appendChild(div);
  }});
}}
const wraps = Array.from(document.querySelectorAll('.imgwrap'));
wraps.forEach(w => renderBoxes(w, JSON.parse(w.dataset.boxes)));

const STORE = 'marker_envelope_gallery_comments_v1';
let comments = {{}};
try {{ comments = JSON.parse(localStorage.getItem(STORE) || '{{}}'); }} catch (e) {{}}
const cards = Array.from(document.querySelectorAll('.card'));
const vComment = document.getElementById('vComment');
function setComment(key, value) {{
  if (value.trim()) comments[key] = value; else delete comments[key];
  localStorage.setItem(STORE, JSON.stringify(comments));
  const card = cards.find(c => c.dataset.key === key);
  if (card) {{
    const ta = card.querySelector('.comment');
    if (ta.value !== value) ta.value = value;
    card.classList.toggle('commented', !!value.trim());
  }}
}}
cards.forEach(c => {{
  const ta = c.querySelector('.comment');
  ta.value = comments[c.dataset.key] || '';
  c.classList.toggle('commented', !!ta.value.trim());
  ta.addEventListener('input', () => setComment(c.dataset.key, ta.value));
}});
document.getElementById('exportComments').onclick = () => {{
  const lines = [['folder', 'image_id', 'comment']];
  cards.forEach(c => {{
    const v = (comments[c.dataset.key] || '').trim();
    if (!v) return;
    const parts = c.dataset.key.split('::');
    lines.push([parts[0], parts[1], v]);
  }});
  const csv = lines.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['\\ufeff' + csv], {{type: 'text/csv;charset=utf-8'}}));
  a.download = 'commenti_marker_envelope.csv';
  a.click();
}};

const body = document.body;
const toggleBtn = document.getElementById('toggleBoxes');
function setBoxes(visible) {{
  body.classList.toggle('noboxes', !visible);
  toggleBtn.textContent = visible ? 'Nascondi box' : 'Mostra box';
}}
toggleBtn.onclick = () => setBoxes(body.classList.contains('noboxes'));

const viewer = document.getElementById('viewer');
const viewerImg = document.getElementById('viewerImg');
const frame = viewer.querySelector('.frame');
const vTitle = document.getElementById('vTitle');
let current = -1;
function openViewer(i) {{
  current = (i + wraps.length) % wraps.length;
  const w = wraps[current];
  viewerImg.src = w.dataset.src;
  renderBoxes(frame, JSON.parse(w.dataset.boxes));
  const card = w.closest('.card');
  vTitle.textContent = (current + 1) + '/' + wraps.length + ' — ' + card.dataset.category + ' / ' + card.querySelector('.title').innerText;
  vComment.value = comments[card.dataset.key] || '';
  viewer.classList.add('open');
}}
function closeViewer() {{ viewer.classList.remove('open'); current = -1; }}
wraps.forEach((w, i) => w.addEventListener('click', () => openViewer(i)));
document.getElementById('vClose').onclick = closeViewer;
document.getElementById('vPrev').onclick = () => openViewer(current - 1);
document.getElementById('vNext').onclick = () => openViewer(current + 1);
viewer.querySelector('.stage').addEventListener('click', e => {{ if (e.target === e.currentTarget) closeViewer(); }});
vComment.addEventListener('input', () => {{
  if (current >= 0) setComment(wraps[current].closest('.card').dataset.key, vComment.value);
}});
document.addEventListener('keydown', e => {{
  const typing = e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT';
  if (e.key === 'Escape' && current >= 0) {{ e.target.blur(); closeViewer(); return; }}
  if (typing) return;
  if (e.key === 'b' || e.key === 'B') setBoxes(body.classList.contains('noboxes'));
  if (current < 0) return;
  if (e.key === 'ArrowLeft') openViewer(current - 1);
  if (e.key === 'ArrowRight') openViewer(current + 1);
}});
</script>
</body></html>"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Gallery: {out / 'index.html'} ({total_cards} immagini, {len(sections)} cartelle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
