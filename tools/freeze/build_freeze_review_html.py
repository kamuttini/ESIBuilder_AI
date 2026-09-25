#!/usr/bin/env python3
"""Build a static HTML gallery to review the freeze/proibite ground truth.

For every `[Freeze_i]` entry the gallery shows two derived images:

    context/<id>.jpg   the full sample screenshot, downscaled, with the
                       rectTemplate drawn in magenta and rectFind in cyan
    crop/<id>.png      the template plus some surrounding context, upscaled
                       with nearest-neighbour so single pixels stay readable

The page is fully static: review verdicts live in the browser (localStorage)
and leave it only through the "Esporta CSV" button. Nothing is written back to
the dataset, so opening the gallery can never damage the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_io import Rect  # noqa: E402

CONTEXT_WIDTH = 520
CROP_PAD_PX = 48
CROP_SCALE = 3
CROP_MAX_WIDTH = 900


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _draw_rect(draw: ImageDraw.ImageDraw, rect: Rect, scale: float, color: str, width: int) -> None:
    draw.rectangle(
        [rect.x * scale, rect.y * scale, rect.x2 * scale, rect.y2 * scale],
        outline=color,
        width=width,
    )


def _render_context(image: Image.Image, template: Rect, find: Rect, out_path: Path) -> None:
    scale = CONTEXT_WIDTH / image.width
    canvas = image.convert("RGB").resize(
        (CONTEXT_WIDTH, max(1, round(image.height * scale))), Image.LANCZOS
    )
    draw = ImageDraw.Draw(canvas)
    _draw_rect(draw, find, scale, "#00e5ff", 1)
    _draw_rect(draw, template, scale, "#ff00c8", 2)
    canvas.save(out_path, format="JPEG", quality=78)


def _render_crop(image: Image.Image, template: Rect, out_path: Path) -> None:
    box = (
        max(0, template.x - CROP_PAD_PX),
        max(0, template.y - CROP_PAD_PX),
        min(image.width, template.x2 + CROP_PAD_PX),
        min(image.height, template.y2 + CROP_PAD_PX),
    )
    patch = image.convert("RGB").crop(box)
    scale = CROP_SCALE
    if patch.width * scale > CROP_MAX_WIDTH:
        scale = max(1, CROP_MAX_WIDTH // max(1, patch.width))
    patch = patch.resize((patch.width * scale, patch.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(patch)
    draw.rectangle(
        [
            (template.x - box[0]) * scale,
            (template.y - box[1]) * scale,
            (template.x2 - box[0]) * scale,
            (template.y2 - box[1]) * scale,
        ],
        outline="#ff00c8",
        width=2,
    )
    patch.save(out_path, format="PNG")


def _load_rows(manifest: Path, classes: Optional[Sequence[str]], limit: Optional[int]) -> List[Dict[str, str]]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if classes:
        wanted = {c.strip().lower() for c in classes}
        rows = [r for r in rows if r["class_label"].lower() in wanted]
    rows.sort(key=lambda r: (r["manufacturer"], r["workspace"], int(r["freeze_index"])))
    if limit:
        rows = rows[:limit]
    return rows


def _render_page(rows: List[Dict[str, object]], title: str, stats: Dict[str, object]) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    stats_html = " · ".join(
        f"<b>{html.escape(str(key))}</b> {html.escape(str(value))}" for key, value in stats.items()
    )
    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; --bg:#14161a; --card:#1d2027; --line:#2c313b; --fg:#e6e8ec; --mut:#9aa3b2; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  header {{ position:sticky; top:0; z-index:10; background:#11131782; backdrop-filter:blur(10px);
            border-bottom:1px solid var(--line); padding:12px 18px; }}
  h1 {{ margin:0 0 6px; font-size:16px; font-weight:650; }}
  .stats {{ color:var(--mut); font-size:12px; }}
  .bar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }}
  select, input, button {{ background:#22262e; color:var(--fg); border:1px solid var(--line);
                           border-radius:7px; padding:6px 9px; font-size:13px; }}
  button {{ cursor:pointer; }}
  button.primary {{ background:#2d6cdf; border-color:#2d6cdf; }}
  main {{ padding:16px; display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(540px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:11px; overflow:hidden; }}
  .card.ok {{ border-color:#2f7d4f; }}
  .card.no {{ border-color:#a23b3b; }}
  .card.dub {{ border-color:#b08514; }}
  .imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:8px; background:#0e1014; }}
  .imgs img {{ width:100%; height:auto; border-radius:5px; display:block; cursor:zoom-in; }}
  .meta {{ padding:9px 11px; border-top:1px solid var(--line); }}
  .row1 {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
  .cls {{ font-weight:650; }}
  .cls.freeze {{ color:#63d2a1; }}
  .cls.unknown {{ color:#e0b13c; }}
  .mut {{ color:var(--mut); font-size:12px; word-break:break-all; }}
  .verdict {{ display:flex; gap:6px; padding:8px 11px; border-top:1px solid var(--line); }}
  .verdict button {{ flex:1; }}
  .verdict button.sel {{ background:#2d6cdf; border-color:#2d6cdf; }}
  dialog {{ border:none; background:#000; padding:0; max-width:96vw; max-height:96vh; }}
  dialog img {{ max-width:96vw; max-height:92vh; display:block; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="stats">{stats_html}</div>
  <div class="bar">
    <select id="fClass"></select>
    <select id="fVendor"></select>
    <select id="fSource"></select>
    <select id="fVerdict">
      <option value="">verdetto: tutti</option>
      <option value="none">da rivedere</option>
      <option value="ok">confermati</option>
      <option value="no">sbagliati</option>
      <option value="dub">dubbi</option>
    </select>
    <input id="fText" type="search" placeholder="cerca workspace / file…" size="26">
    <span class="stats" id="count"></span>
    <button class="primary" id="export">Esporta CSV</button>
    <button id="reset">Azzera verdetti</button>
  </div>
</header>
<main id="grid"></main>
<dialog id="zoom"><img id="zoomImg" alt=""></dialog>
<script>
const ROWS = {payload};
const KEY = "freeze_review_verdicts_v1";
let verdicts = {{}};
try {{ verdicts = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ verdicts = {{}}; }}

const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(verdicts)); }} catch (e) {{}} }};
const grid = document.getElementById("grid");
const zoom = document.getElementById("zoom");
const zoomImg = document.getElementById("zoomImg");

function fillSelect(el, label, values) {{
  el.innerHTML = `<option value="">${{label}}: tutti</option>` +
    values.map(v => `<option value="${{v}}">${{label}}: ${{v}}</option>`).join("");
}}
const uniq = key => [...new Set(ROWS.map(r => r[key]))].filter(Boolean).sort();
fillSelect(document.getElementById("fClass"), "classe", uniq("class_label"));
fillSelect(document.getElementById("fVendor"), "vendor", uniq("manufacturer"));
fillSelect(document.getElementById("fSource"), "origine", uniq("class_source"));

function visible() {{
  const c = document.getElementById("fClass").value;
  const v = document.getElementById("fVendor").value;
  const s = document.getElementById("fSource").value;
  const w = document.getElementById("fVerdict").value;
  const t = document.getElementById("fText").value.trim().toLowerCase();
  return ROWS.filter(r => (!c || r.class_label === c)
    && (!v || r.manufacturer === v)
    && (!s || r.class_source === s)
    && (!w || (w === "none" ? !verdicts[r.entry_id] : verdicts[r.entry_id] === w))
    && (!t || (r.workspace + " " + r.original_name).toLowerCase().includes(t)));
}}

function card(r) {{
  const v = verdicts[r.entry_id] || "";
  const el = document.createElement("div");
  el.className = "card " + v;
  el.innerHTML = `
    <div class="imgs">
      <img loading="lazy" src="${{r.context_src}}" alt="contesto">
      <img loading="lazy" src="${{r.crop_src}}" alt="template">
    </div>
    <div class="meta">
      <div class="row1">
        <span class="cls ${{r.class_label}}">${{r.class_label}}</span>
        <span class="mut">${{r.manufacturer}} · ${{r.template_w}}×${{r.template_h}} px · ${{r.split}}${{r.is_screen_saver === "True" ? " · screensaver" : ""}}</span>
      </div>
      <div class="mut">${{r.workspace}} · setup ${{r.setup_id}} · Freeze_${{r.freeze_index}}</div>
      <div class="mut">${{r.original_name || "(nome originale non recuperato)"}} · ${{r.class_source}}</div>
    </div>
    <div class="verdict">
      <button data-v="ok" class="${{v === "ok" ? "sel" : ""}}">È freeze</button>
      <button data-v="no" class="${{v === "no" ? "sel" : ""}}">Non è freeze</button>
      <button data-v="dub" class="${{v === "dub" ? "sel" : ""}}">Dubbio</button>
    </div>`;
  el.querySelectorAll(".verdict button").forEach(b => b.addEventListener("click", () => {{
    const choice = b.dataset.v;
    if (verdicts[r.entry_id] === choice) delete verdicts[r.entry_id];
    else verdicts[r.entry_id] = choice;
    save(); render();
  }}));
  el.querySelectorAll(".imgs img").forEach(img => img.addEventListener("click", () => {{
    zoomImg.src = img.src; zoom.showModal();
  }}));
  return el;
}}

function render() {{
  const rows = visible();
  grid.replaceChildren(...rows.map(card));
  const done = ROWS.filter(r => verdicts[r.entry_id]).length;
  document.getElementById("count").textContent =
    `${{rows.length}} mostrate · ${{done}}/${{ROWS.length}} riviste`;
}}

["fClass", "fVendor", "fSource", "fVerdict", "fText"].forEach(id =>
  document.getElementById(id).addEventListener("input", render));
zoom.addEventListener("click", () => zoom.close());

document.getElementById("export").addEventListener("click", () => {{
  const head = ["entry_id", "workspace", "setup_id", "freeze_index", "manufacturer",
                "class_label", "class_source", "original_name", "verdict"];
  const esc = s => `"${{String(s ?? "").replace(/"/g, '""')}}"`;
  const lines = [head.join(",")].concat(ROWS.filter(r => verdicts[r.entry_id]).map(r =>
    head.map(k => esc(k === "verdict" ? verdicts[r.entry_id] : r[k])).join(",")));
  const blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "freeze_review_verdicts.csv";
  a.click();
}});

document.getElementById("reset").addEventListener("click", () => {{
  if (!confirm("Azzerare tutti i verdetti salvati nel browser?")) return;
  verdicts = {{}}; save(); render();
}});

render();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Folder with manifest_proibite.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder for the gallery.")
    parser.add_argument("--classes", nargs="*", default=None, help="Keep only these class labels.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of entries (debug).")
    parser.add_argument("--title", default="Review ground truth proibite / freeze")
    parser.add_argument("--skip-existing", action="store_true", help="Do not re-render images already on disk.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.dataset_dir / "manifest_proibite.csv"
    if not manifest.is_file():
        print(f"ERROR: missing manifest {manifest}")
        return 2

    rows = _load_rows(manifest, args.classes, args.limit)
    if not rows:
        print("ERROR: no rows selected")
        return 2

    output_dir = args.output_dir
    (output_dir / "context").mkdir(parents=True, exist_ok=True)
    (output_dir / "crop").mkdir(parents=True, exist_ok=True)

    cards: List[Dict[str, object]] = []
    skipped = 0
    for position, row in enumerate(rows, start=1):
        sample_path = Path(row["sample_path"]) if row["sample_path"] else None
        if sample_path is None or not sample_path.is_file():
            skipped += 1
            continue
        key = _safe_id(row["entry_id"])
        context_path = output_dir / "context" / f"{key}.jpg"
        crop_path = output_dir / "crop" / f"{key}.png"

        if not (args.skip_existing and context_path.exists() and crop_path.exists()):
            try:
                with Image.open(sample_path) as image:
                    image.load()
                    template = Rect(int(row["template_x"]), int(row["template_y"]),
                                    int(row["template_w"]), int(row["template_h"]))
                    find = Rect(int(row["find_x"]), int(row["find_y"]),
                                int(row["find_w"]), int(row["find_h"]))
                    _render_context(image, template, find, context_path)
                    _render_crop(image, template, crop_path)
            except Exception as error:  # unreadable or truncated source image
                print(f"WARNING: cannot render {sample_path}: {error}")
                skipped += 1
                continue

        card = {key: row[key] for key in (
            "entry_id", "workspace", "setup_id", "manufacturer", "freeze_index",
            "class_label", "class_source", "original_name", "split",
            "template_w", "template_h", "is_screen_saver")}
        card["context_src"] = f"context/{key}.jpg"
        card["crop_src"] = f"crop/{key}.png"
        cards.append(card)

        if position % 100 == 0:
            print(f"  rendered {position}/{len(rows)}")

    counts = Counter(card["class_label"] for card in cards)
    stats = {
        "entry": len(cards),
        "workspace": len({card["workspace"] for card in cards}),
        "freeze": counts.get("freeze", 0),
        "unknown": counts.get("unknown", 0),
        "senza immagine": skipped,
    }
    page = _render_page(cards, args.title, stats)
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")

    print(json.dumps({"cards": len(cards), "skipped": skipped, "index": str(index_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
