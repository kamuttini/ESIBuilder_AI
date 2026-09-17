#!/usr/bin/env python3
"""Build a self-contained review page for probe-template (line #14) ground truth.

For every config flagged by ``prepare_probe_template_dataset.py`` it shows a zoom of
the region with two boxes overlaid - the legacy ``.fss`` box (red) and the box
reconstructed from ``probe_name.png`` (green) - plus the reference crop, so the
disagreement can be judged by eye. Decisions are exported as CSV from the page
(no server, nothing is written to the dataset).
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PAD = 70
ZOOM = 2


def pick_frame(folder: Path) -> Optional[Path]:
    image_dir = folder / "image_samples"
    if not image_dir.is_dir():
        return None
    frames = [p for p in sorted(image_dir.iterdir())
              if p.is_file() and p.suffix.lower() == ".png" and not p.name.startswith("._")]
    return frames[len(frames) // 2] if frames else None


def to_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def render_zoom(frame: np.ndarray, boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str]]) -> str:
    h, w = frame.shape[:2]
    xs = [c for box, _, _ in boxes for c in (box[0], box[2])]
    ys = [c for box, _, _ in boxes for c in (box[1], box[3])]
    x0, x1 = max(0, min(xs) - PAD), min(w, max(xs) + PAD)
    y0, y1 = max(0, min(ys) - PAD), min(h, max(ys) + PAD)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = frame[y0:y1, x0:x1].copy()
    crop = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
    for (bx1, by1, bx2, by2), color, label in boxes:
        p1 = ((bx1 - x0) * ZOOM, (by1 - y0) * ZOOM)
        p2 = ((bx2 - x0) * ZOOM, (by2 - y0) * ZOOM)
        cv2.rectangle(crop, p1, p2, color, 1)
        cv2.putText(crop, label, (p1[0], max(10, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return to_b64(crop)


def build(folders_csv: Path, output: Path, statuses: List[str], limit: int) -> int:
    rows = [r for r in csv.DictReader(folders_csv.open(encoding="utf-8"))]
    flagged = [r for r in rows if r["gt_status"] in statuses]
    if limit > 0:
        flagged = flagged[:limit]
    cards: List[str] = []
    for row in flagged:
        folder = Path(row["folder_path"])
        frame_path = pick_frame(folder)
        frame = cv2.imread(str(frame_path)) if frame_path else None
        try:
            fss_box = (int(row["fss_left"]), int(row["fss_top"]), int(row["fss_right"]), int(row["fss_bottom"]))
        except ValueError:
            fss_box = None
        rec_box = None
        if row["median_box_xmin"] not in ("", None) and row["template_w"] not in ("", None):
            x1, y1 = int(row["median_box_xmin"]), int(row["median_box_ymin"])
            rec_box = (x1, y1, x1 + int(row["template_w"]) + 5, y1 + int(row["template_h"]) + 5)

        zoom_b64 = ""
        if frame is not None:
            boxes = []
            if fss_box:
                boxes.append((fss_box, (0, 0, 255), "fss"))
            if rec_box:
                boxes.append((rec_box, (0, 220, 0), "crop"))
            if boxes:
                zoom_b64 = render_zoom(frame, boxes)

        tpl_b64 = ""
        tpl_path = Path(row["template_path"]) if row["template_path"] else None
        if tpl_path and tpl_path.is_file():
            tpl = cv2.imread(str(tpl_path))
            if tpl is not None:
                tpl = cv2.resize(tpl, (tpl.shape[1] * ZOOM, tpl.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
                tpl_b64 = to_b64(tpl)

        cards.append(f"""
<div class="card" data-folder="{html.escape(row['folder'])}">
  <div class="head"><span class="status {html.escape(row['gt_status'])}">{html.escape(row['gt_status'])}</span>
    <b>{html.escape(row['folder'])}</b> <span class="muted">{html.escape(row['manufacturer'])} &middot; {html.escape(row['setup_name'])}</span></div>
  <div class="body">
    <div class="imgs">
      {'<img src="data:image/png;base64,' + zoom_b64 + '" alt="zoom">' if zoom_b64 else '<div class="missing">frame non disponibile</div>'}
      <div class="tpl">{'<img src="data:image/png;base64,' + tpl_b64 + '" alt="template">' if tpl_b64 else 'crop mancante'}<div class="muted">probe_name.png</div></div>
    </div>
    <div class="meta">
      <div><span class="k">riga #14 (.fss)</span> <span class="v red">{row['fss_left']},{row['fss_top']} &rarr; {row['fss_right']},{row['fss_bottom']}</span></div>
      <div><span class="k">da crop</span> <span class="v green">{rec_box[0] if rec_box else '-'},{rec_box[1] if rec_box else '-'} &rarr; {rec_box[2] if rec_box else '-'},{rec_box[3] if rec_box else '-'}</span></div>
      <div><span class="k">IoU</span> <span class="v">{row['median_iou_vs_fss']}</span>
           <span class="k">hit rate</span> <span class="v">{row['hit_rate']}</span>
           <span class="k">score</span> <span class="v">{row['median_score']}</span></div>
      <div><span class="k">frame</span> <span class="v">{row['n_hits']}/{row['n_images_scanned']}</span>
           <span class="k">gruppo</span> <span class="v">{row['group_id']}</span>
           <span class="k">split</span> <span class="v">{row['split']}</span></div>
      <div class="choices">
        <label><input type="radio" name="c_{html.escape(row['folder'])}" value="fss"> tiene la riga #14 legacy</label>
        <label><input type="radio" name="c_{html.escape(row['folder'])}" value="crop"> tiene il box dal crop</label>
        <label><input type="radio" name="c_{html.escape(row['folder'])}" value="escludere"> escludere dal dataset</label>
        <label><input type="radio" name="c_{html.escape(row['folder'])}" value="dubbio"> da rivedere</label>
      </div>
      <input class="note" type="text" placeholder="nota (opzionale)" data-folder="{html.escape(row['folder'])}">
    </div>
  </div>
</div>""")

    page = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><title>Review template sonda (riga #14)</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #14161a; color: #e6e8eb; }}
 header {{ position: sticky; top: 0; background: #1b1e24; padding: 12px 20px; border-bottom: 1px solid #2c3039; z-index: 5; }}
 h1 {{ font-size: 16px; margin: 0 0 6px; }}
 .muted {{ color: #9aa3ad; font-weight: normal; }}
 .card {{ background: #1b1e24; margin: 14px 20px; border: 1px solid #2c3039; border-radius: 8px; overflow: hidden; }}
 .head {{ padding: 8px 12px; border-bottom: 1px solid #2c3039; font-size: 13px; }}
 .body {{ display: flex; gap: 16px; padding: 12px; flex-wrap: wrap; }}
 .imgs {{ display: flex; gap: 12px; align-items: flex-start; }}
 .imgs img {{ image-rendering: pixelated; border: 1px solid #2c3039; border-radius: 4px; max-width: 760px; }}
 .tpl {{ text-align: center; font-size: 11px; }}
 .meta {{ font-size: 12px; line-height: 1.9; min-width: 320px; }}
 .k {{ color: #9aa3ad; margin-right: 6px; }}
 .v {{ font-family: ui-monospace, Menlo, monospace; margin-right: 14px; }}
 .red {{ color: #ff6b6b; }} .green {{ color: #51cf66; }}
 .status {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right: 8px; }}
 .fss_mismatch {{ background: #7a3b12; }} .low_hit_rate {{ background: #123b7a; }}
 .choices {{ margin-top: 8px; display: flex; flex-direction: column; gap: 2px; }}
 .note {{ margin-top: 6px; width: 320px; background: #14161a; color: #e6e8eb; border: 1px solid #2c3039; border-radius: 4px; padding: 4px 6px; }}
 button {{ background: #2f6feb; color: #fff; border: 0; padding: 6px 12px; border-radius: 6px; cursor: pointer; }}
 .missing {{ color: #9aa3ad; padding: 30px; }}
</style></head>
<body>
<header>
  <h1>Review ground truth template sonda &mdash; riga #14 RECT_NAME_PROBE</h1>
  <div class="muted">{len(flagged)} configurazioni segnalate &middot; <span class="red">rosso</span> = box scritto nel <code>.fss</code> legacy,
   <span class="green">verde</span> = box ricostruito dal crop <code>probe_name.png</code>.
   Le scelte restano nel browser finche non esporti.</div>
  <div style="margin-top:8px"><button onclick="exportCsv()">Esporta decisioni CSV</button></div>
</header>
{''.join(cards)}
<script>
function exportCsv() {{
  const rows = [["folder","decisione","nota"]];
  document.querySelectorAll('.card').forEach(card => {{
    const folder = card.dataset.folder;
    const checked = card.querySelector('input[type=radio]:checked');
    const note = card.querySelector('input.note').value || "";
    if (checked || note) rows.push([folder, checked ? checked.value : "", note]);
  }});
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = "probe_template_review_decisions.csv";
  a.click();
}}
</script>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"{len(flagged)} schede -> {output}  ({output.stat().st_size/1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folders-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statuses", type=str, default="fss_mismatch,low_hit_rate,template_missing,no_fss_box")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    return build(args.folders_csv, args.output, [s.strip() for s in args.statuses.split(",") if s.strip()], args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
