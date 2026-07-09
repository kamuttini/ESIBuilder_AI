#!/usr/bin/env python3
"""Build an HTML review gallery for marker-vs-legacy-line16 disagreements.

Reads `per_image_predictions.csv` produced by
`eval_marker_bundle_vs_legacy_line16.py` and builds an interactive gallery:

- clean scaled images (no baked-in drawing); boxes are rendered as HTML
  overlays so they can be toggled on/off (button or key `B`)
- fullscreen viewer (click a card; arrows = prev/next, Esc = close)
- overlay colors: green = legacy line-16 boxes, red = detected marker,
  blue = echo rect (line 11)

Cards are grouped by disagreement category:
- `outside_legacy_boxes`: marker accepted but its center is in no legacy box
- `group_mismatch`: predicted group != legacy box containing the marker
- `gt_mismatch`: predicted group != folder-level GT (SOLO NF/UD/... folders)
- `review`: rows the detector itself flagged
- `control_ok`: random sample of agreeing rows for visual comparison

Example:
  python3 tools/orientation/build_marker_eval_disagreement_gallery.py \
    --eval-dir artifacts/42_orientation_eval_vs_legacy_line16/pilot_v2 \
    --dataset-root /Volumes/SSD_esi1_n1
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "eval_marker_bundle_vs_legacy_line16", SCRIPT_DIR / "eval_marker_bundle_vs_legacy_line16.py"
)
_eval_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eval_mod
_spec.loader.exec_module(_eval_mod)  # type: ignore[union-attr]
_parse_fss = _eval_mod._parse_fss
GROUPS = _eval_mod.GROUPS

Rect = Tuple[int, int, int, int]  # top, left, bottom, right

CATEGORY_ORDER = ("outside_legacy_boxes", "group_mismatch", "gt_mismatch", "review", "control_ok")
CATEGORY_TITLES = {
    "outside_legacy_boxes": "Marker fuori da tutti i box legacy",
    "group_mismatch": "Gruppo predetto ≠ box legacy contenente il marker",
    "gt_mismatch": "Gruppo predetto ≠ GT cartella (SOLO NF/UD/...)",
    "review": "Casi flaggati review dal detector",
    "control_ok": "Controllo: casi in accordo (campione)",
}


def _parse_box(text: str) -> Optional[Rect]:
    if not text:
        return None
    parts = [int(float(p)) for p in text.split("|")]
    return (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None


def _load_folder_gt(dataset_root: Path, folder_name: str, cache: Dict[str, object]) -> Optional[object]:
    if folder_name in cache:
        return cache[folder_name]
    fss_candidates = sorted((dataset_root / folder_name / "DB_setup").glob("setup_*.fss"))
    result = None
    if fss_candidates:
        try:
            result = _parse_fss(fss_candidates[0])
        except ValueError:
            result = None
    cache[folder_name] = result
    return result


def _boxes_payload(
    width: int,
    height: int,
    marker: Optional[Rect],
    rect_echo: Optional[Rect],
    line16: Optional[Dict[str, Rect]],
    duplicated: Optional[Dict[str, bool]],
    name_rects: Optional[List[Rect]] = None,
) -> List[Dict[str, object]]:
    """Normalized (percent) boxes for the HTML overlay renderer."""

    def norm(rect: Rect) -> Dict[str, float]:
        top, left, bottom, right = rect
        return {
            "t": round(100.0 * top / height, 3),
            "l": round(100.0 * left / width, 3),
            "h": round(100.0 * max(1, bottom - top + 1) / height, 3),
            "w": round(100.0 * max(1, right - left + 1) / width, 3),
        }

    boxes: List[Dict[str, object]] = []
    if rect_echo:
        boxes.append({"kind": "rect", "label": "", **norm(rect_echo)})
    if line16:
        single_orientation = bool(duplicated) and all(duplicated.get(g) for g in GROUPS if g != "NF")
        for group in GROUPS:
            if duplicated and group != "NF" and duplicated.get(group):
                continue
            label = "CAL (unico stato)" if (single_orientation and group == "NF") else group
            boxes.append({"kind": "legacy", "label": label, **norm(line16[group])})
    for idx, rect in enumerate(name_rects or []):
        boxes.append({"kind": "excl", "label": f"#{13 + idx} escluso", **norm(rect)})
    if marker:
        boxes.append({"kind": "marker", "label": "", **norm(marker)})
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Output dir of the eval script.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <eval-dir>/disagreement_gallery")
    parser.add_argument("--max-cards-per-category", type=int, default=60)
    parser.add_argument("--control-samples", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.output_dir or (args.eval_dir / "disagreement_gallery")
    csv_path = args.eval_dir / "per_image_predictions.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]

    gt_cache_pre: Dict[str, object] = {}

    def _is_single_orientation(folder: str) -> bool:
        gt = _load_folder_gt(args.dataset_root, folder, gt_cache_pre)
        if not gt:
            return False
        duplicated = gt[2]
        return all(duplicated.get(g) for g in GROUPS if g != "NF")

    rng = random.Random(args.seed)
    by_category: Dict[str, List[Dict[str, str]]] = {c: [] for c in CATEGORY_ORDER}
    for row in rows:
        marker = _parse_box(row.get("marker_box_abs", ""))
        agree_ok = True
        if row.get("gt_group") and row.get("pred_group") and row["gt_group"] != row["pred_group"]:
            by_category["gt_mismatch"].append(row)
            agree_ok = False
        if row.get("box_agree", "") == "0":
            by_category["group_mismatch"].append(row)
            agree_ok = False
        if (
            marker
            and row.get("status") == "ok"
            and not row.get("gt_by_box")
            and not _is_single_orientation(str(row["folder"]))  # line16 not usable as GT there
        ):
            by_category["outside_legacy_boxes"].append(row)
            agree_ok = False
        if row.get("status") not in ("ok", ""):
            by_category["review"].append(row)
            agree_ok = False
        if agree_ok and marker and row.get("gt_by_box"):
            by_category["control_ok"].append(row)

    by_category["control_ok"] = rng.sample(
        by_category["control_ok"], min(args.control_samples, len(by_category["control_ok"]))
    )

    gt_cache: Dict[str, object] = {}
    sections: List[str] = []
    total_cards = 0
    for category in CATEGORY_ORDER:
        selected = by_category[category][: args.max_cards_per_category]
        cards: List[str] = []
        for row in selected:
            folder = str(row["folder"])
            image_path = args.dataset_root / folder / str(row["image_id"])
            if not image_path.is_file():
                continue
            gt = _load_folder_gt(args.dataset_root, folder, gt_cache)
            rect_echo, line16, duplicated, name_rects = gt if gt else (None, None, None, None)
            img_rel = Path("images") / category / f"{total_cards:04d}.jpg"
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
                print(f"[warn] image failed for {image_path}: {exc}")
                continue
            total_cards += 1
            boxes = _boxes_payload(
                width, height, _parse_box(row.get("marker_box_abs", "")), rect_echo, line16, duplicated, name_rects
            )
            meta = (
                f"pred: <b>{html.escape(row.get('pred_group') or '-')}</b> | "
                f"gt_box: {html.escape(row.get('gt_by_box') or '-')} | "
                f"gt_cartella: {html.escape(row.get('gt_group') or '-')} | "
                f"score: {html.escape((row.get('match_score') or '')[:6])} | "
                f"scope: {html.escape(row.get('search_scope') or '-')} | "
                f"status: {html.escape(row.get('status') or '-')}"
                + (f" | reason: {html.escape(row['review_reason'])}" if row.get("review_reason") else "")
            )
            comment_key = f"{folder}::{row['image_id']}"
            cards.append(
                f"<div class='card' data-key='{html.escape(comment_key, quote=True)}' data-category='{category}'>"
                f"<div class='title'>{html.escape(folder)}<br><small>{html.escape(str(row['image_id']))}</small></div>"
                f"<div class='imgwrap' data-boxes='{html.escape(json.dumps(boxes), quote=True)}' data-src='{img_rel.as_posix()}'>"
                f"<img loading='lazy' src='{img_rel.as_posix()}'></div>"
                f"<div class='meta'>{meta}</div>"
                "<textarea class='comment' placeholder='Commento...' rows='2'></textarea>"
                "</div>"
            )
        sections.append(
            f"<h2>{html.escape(CATEGORY_TITLES[category])} "
            f"<span class='count'>({len(by_category[category])} casi, mostrati {len(cards)})</span></h2>"
            + ("<div class='grid'>" + "".join(cards) + "</div>" if cards else "<p>Nessun caso.</p>")
        )

    page = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>Disaccordi marker vs riga 16 legacy</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #16181d; color: #e8e8e8; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #444; padding-bottom: 6px; }}
.count {{ color: #999; font-weight: normal; }}
.legend span {{ margin-right: 18px; }}
.toolbar {{ position: sticky; top: 0; z-index: 50; background: #16181d; padding: 8px 0; }}
.toolbar button {{ background: #2f6fed; color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font-size: 14px; cursor: pointer; }}
.toolbar .hint {{ color: #999; font-size: 12px; margin-left: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 14px; }}
.card {{ background: #22252c; border-radius: 8px; padding: 10px; }}
.imgwrap {{ position: relative; cursor: zoom-in; }}
.imgwrap img {{ width: 100%; display: block; border-radius: 4px; }}
.title {{ font-size: 12px; margin-bottom: 6px; color: #9ecbff; }}
.meta {{ font-size: 12px; margin-top: 6px; color: #ccc; }}
.box {{ position: absolute; box-sizing: border-box; pointer-events: none; }}
.box.rect {{ border: 2px solid #3c78ff; }}
.box.legacy {{ border: 2px solid #00c800; }}
.box.legacy .lbl {{ position: absolute; top: -18px; left: 0; color: #00e000; font-size: 12px; font-weight: bold; text-shadow: 0 0 3px #000; }}
.box.marker {{ border: 2px solid #ff2828; min-width: 10px; min-height: 10px; }}
.box.excl {{ border: 2px dashed #ffd000; }}
.box.excl .lbl {{ position: absolute; top: -18px; left: 0; color: #ffd000; font-size: 11px; text-shadow: 0 0 3px #000; }}
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
<h1>Disaccordi marker vs riga 16 legacy</h1>
<div class="toolbar">
<button id="toggleBoxes">Nascondi box</button>
<button id="exportComments">Esporta commenti CSV</button>
<span class="legend"><span style="color:#3c78ff">■ rect ecografico (riga 11 .fss)</span>
<span style="color:#00c800">■ box riga 16 .fss legacy</span>
<span style="color:#ffd000">▨ zona esclusa (nome ecografo/sonda #13/#14)</span>
<span style="color:#ff2828">■ marker trovato dal nuovo detector</span></span>
<span class="hint">Click su un'immagine = tutto schermo · B = mostra/nascondi box · ←/→ = scorri · Esc = chiudi · I commenti si salvano da soli nel browser</span>
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

const STORE = 'marker_gallery_comments_v1';
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
  const lines = [['folder', 'image_id', 'category', 'comment']];
  cards.forEach(c => {{
    const v = (comments[c.dataset.key] || '').trim();
    if (!v) return;
    const parts = c.dataset.key.split('::');
    lines.push([parts[0], parts[1], c.dataset.category, v]);
  }});
  const csv = lines.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['\\ufeff' + csv], {{type: 'text/csv;charset=utf-8'}}));
  a.download = 'commenti_disaccordi_marker.csv';
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
  vTitle.textContent = (current + 1) + '/' + wraps.length + ' — ' + card.querySelector('.title').innerText.replace('\\n', ' / ');
  vComment.value = comments[card.dataset.key] || '';
  viewer.classList.add('open');
}}
vComment.addEventListener('input', () => {{
  if (current >= 0) setComment(wraps[current].closest('.card').dataset.key, vComment.value);
}});
function closeViewer() {{ viewer.classList.remove('open'); current = -1; }}
wraps.forEach((w, i) => w.addEventListener('click', () => openViewer(i)));
document.getElementById('vClose').onclick = closeViewer;
document.getElementById('vPrev').onclick = () => openViewer(current - 1);
document.getElementById('vNext').onclick = () => openViewer(current + 1);
viewer.querySelector('.stage').addEventListener('click', e => {{ if (e.target === e.currentTarget) closeViewer(); }});
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
    print(f"Gallery: {out / 'index.html'} ({total_cards} immagini)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
