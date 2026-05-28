#!/usr/bin/env python3
"""Build prioritized GT-fix queue for scale line annotations.

Input:
- one or more predictions CSV produced by train_scale_line_image_model.py

Output:
- priority CSV (all rows + top-N)
- summary JSON
- HTML review with large overlays (full image + zoom)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


@dataclass
class Row:
    rank: int
    sample_id: str
    split: str
    config_folder: str
    image_path: str
    setup_id: str
    depth_index: int
    fss_path: str
    scale_line_raw: str
    x_gt: float
    y_top_gt: float
    y_bottom_gt: float
    x_pred: float
    y_top_pred: float
    y_bottom_pred: float
    x_err_px: float
    y_err_px: float
    score: float
    flagged_config: bool
    priority: str


def _f(text: str) -> float:
    return float(text.strip())


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:140]


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)_([0-9]+)$", sample_id.strip())
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def config_from_image_path(image_path: str) -> str:
    p = Path(image_path)
    parts = p.parts
    try:
        i = parts.index("SSD_esi1_n1")
        if i + 1 < len(parts):
            return parts[i + 1]
    except ValueError:
        pass
    if len(parts) >= 4:
        return parts[-4]
    return ""


def find_fss_for_row(image_path: str, setup_id: str, cache: Dict[Tuple[str, str], str]) -> str:
    p = Path(image_path)
    cache_key = (str(p.parent), setup_id)
    if cache_key in cache:
        return cache[cache_key]

    target_name = f"setup_{setup_id}.fss"
    # Try direct ancestors first.
    for anc in [p.parent, *p.parents]:
        cand = anc / "DB_setup" / target_name
        if cand.exists():
            out = cand.as_posix()
            cache[cache_key] = out
            return out

    # Fallback: limited recursive search around first useful ancestor.
    search_root: Optional[Path] = None
    for anc in [p.parent, *p.parents]:
        if anc.name == "SSD_esi1_n1":
            break
        search_root = anc
    if search_root is not None and search_root.exists():
        hits = sorted(search_root.glob(f"**/DB_setup/{target_name}"))
        if hits:
            out = hits[0].as_posix()
            cache[cache_key] = out
            return out

    cache[cache_key] = ""
    return ""


def read_scale_line_raw(fss_path: str, cache: Dict[str, str]) -> str:
    if not fss_path:
        return ""
    if fss_path in cache:
        return cache[fss_path]
    p = Path(fss_path)
    if not p.exists():
        cache[fss_path] = ""
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        # FSS line 21 is 1-based index 21 => zero-based 20
        raw = lines[20].strip() if len(lines) >= 21 else ""
    except Exception:
        raw = ""
    cache[fss_path] = raw
    return raw


def is_flagged_config(config_name: str) -> bool:
    t = config_name.upper()
    bad_tokens = [
        "SBAGLIATO",
        "NON USARE",
        "STRANA",
        "SBALLATA",
        "CAMBIALE",
    ]
    return any(tok in t for tok in bad_tokens)


def priority_from_errors(score: float, x_err: float, flagged_config: bool) -> str:
    adj = score + (70.0 if flagged_config else 0.0)
    if x_err >= 250.0 or adj >= 320.0:
        return "P0"
    if x_err >= 140.0 or adj >= 190.0:
        return "P1"
    if x_err >= 70.0 or adj >= 120.0:
        return "P2"
    return "P3"


def parse_row_dict(d: Dict[str, str]) -> Optional[Dict[str, float | str]]:
    try:
        return {
            "sample_id": d["sample_id"],
            "split": d["split"],
            "image_path": d["image_path"],
            "x_gt": _f(d["x_gt"]),
            "y_top_gt": _f(d["y_top_gt"]),
            "y_bottom_gt": _f(d["y_bottom_gt"]),
            "x_pred": _f(d["x_pred"]),
            "y_top_pred": _f(d["y_top_pred"]),
            "y_bottom_pred": _f(d["y_bottom_pred"]),
            "x_err_px": _f(d["x_err_px"]),
            "y_err_px": _f(d["y_err_px"]),
        }
    except Exception:
        return None


def load_prediction_rows(pred_csvs: Sequence[Path]) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for csv_path in pred_csvs:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for d in reader:
                row = parse_row_dict(d)
                if row is None:
                    continue
                rows.append(row)
    return rows


def draw_overlay_previews(
    row: Row,
    full_out: Path,
    zoom_out: Path,
) -> None:
    img = Image.open(row.image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # GT in green, prediction in magenta.
    draw.line((row.x_gt, row.y_top_gt, row.x_gt, row.y_bottom_gt), fill=(0, 255, 120, 255), width=6)
    draw.line((row.x_pred, row.y_top_pred, row.x_pred, row.y_bottom_pred), fill=(255, 0, 220, 255), width=5)

    r = 6
    for x, y, c in [
        (row.x_gt, row.y_top_gt, (0, 255, 120, 255)),
        (row.x_gt, row.y_bottom_gt, (0, 255, 120, 255)),
        (row.x_pred, row.y_top_pred, (255, 0, 220, 255)),
        (row.x_pred, row.y_bottom_pred, (255, 0, 220, 255)),
    ]:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(255, 255, 255, 255))

    full_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(full_out, quality=92)

    x_min = max(0, int(min(row.x_gt, row.x_pred) - 260))
    x_max = min(w, int(max(row.x_gt, row.x_pred) + 260))
    y_min = max(0, int(min(row.y_top_gt, row.y_top_pred) - 120))
    y_max = min(h, int(max(row.y_bottom_gt, row.y_bottom_pred) + 120))
    if x_max <= x_min or y_max <= y_min:
        crop = img
    else:
        crop = img.crop((x_min, y_min, x_max, y_max))
    zoom_out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(zoom_out, quality=92)


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "priority",
                "split",
                "sample_id",
                "config_folder",
                "setup_id",
                "depth_index",
                "score",
                "x_err_px",
                "y_err_px",
                "x_gt",
                "y_top_gt",
                "y_bottom_gt",
                "x_pred",
                "y_top_pred",
                "y_bottom_pred",
                "flagged_config",
                "image_path",
                "fss_path",
                "scale_line_raw",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rank,
                    r.priority,
                    r.split,
                    r.sample_id,
                    r.config_folder,
                    r.setup_id,
                    r.depth_index,
                    f"{r.score:.4f}",
                    f"{r.x_err_px:.4f}",
                    f"{r.y_err_px:.4f}",
                    f"{r.x_gt:.4f}",
                    f"{r.y_top_gt:.4f}",
                    f"{r.y_bottom_gt:.4f}",
                    f"{r.x_pred:.4f}",
                    f"{r.y_top_pred:.4f}",
                    f"{r.y_bottom_pred:.4f}",
                    1 if r.flagged_config else 0,
                    r.image_path,
                    r.fss_path,
                    r.scale_line_raw,
                ]
            )


def write_html(path: Path, rows: Sequence[Row], preview_dir: Path) -> None:
    def short_line21(text: str, max_len: int = 80) -> str:
        t = text.strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 3] + "..."

    trs: List[str] = []
    for r in rows:
        full_name = f"{r.rank:04d}_{_safe_slug(r.sample_id)}_full.jpg"
        zoom_name = f"{r.rank:04d}_{_safe_slug(r.sample_id)}_zoom.jpg"
        full_rel = f"{preview_dir.name}/{full_name}"
        zoom_rel = f"{preview_dir.name}/{zoom_name}"
        trs.append(
            "<tr>"
            f"<td>{r.rank}</td>"
            f"<td>{r.priority}</td>"
            f"<td>{r.split}</td>"
            f"<td>{r.sample_id}</td>"
            f"<td>{r.score:.1f}</td>"
            f"<td>{r.x_err_px:.1f}</td>"
            f"<td>{r.y_err_px:.1f}</td>"
            f"<td>{'yes' if r.flagged_config else 'no'}</td>"
            f"<td>{r.config_folder}</td>"
            f"<td>{r.setup_id}</td>"
            f"<td>{r.depth_index}</td>"
            f"<td><a href='{full_rel}' target='_blank'><img src='{full_rel}' loading='lazy' /></a></td>"
            f"<td><a href='{zoom_rel}' target='_blank'><img src='{zoom_rel}' loading='lazy' /></a></td>"
            f"<td>{Path(r.image_path).name}</td>"
            f"<td>{Path(r.fss_path).name if r.fss_path else ''}</td>"
            f"<td><code>{short_line21(r.scale_line_raw)}</code></td>"
            "</tr>"
        )

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale GT Fix Queue</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 860px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Scale GT Fix Queue</h1>
  <p>Overlay colors: <b>green=GT</b>, <b>magenta=prediction</b>. Prioritization by model disagreement and flagged config names.</p>
  <table>
    <thead>
      <tr>
        <th>rank</th><th>prio</th><th>split</th><th>sample_id</th><th>score</th><th>x_err</th><th>y_err</th>
        <th>flag</th><th>config</th><th>setup</th><th>depth</th><th>full</th><th>zoom</th><th>image</th><th>fss</th><th>line21</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_html_big(path: Path, rows: Sequence[Row], preview_dir: Path) -> None:
    ui_version = "onecard_v6_2026-04-22"
    cards: List[str] = []
    for r in rows:
        base = f"{r.rank:04d}_{_safe_slug(r.sample_id)}"
        full_name = f"{base}_full.jpg"
        zoom_name = f"{base}_zoom.jpg"
        full_rel = f"{preview_dir.name}/{full_name}"
        zoom_rel = f"{preview_dir.name}/{zoom_name}"
        item_key = f"{r.sample_id}||{r.image_path}"
        item_key_esc = html.escape(item_key, quote=True)
        line21_esc = html.escape(r.scale_line_raw, quote=True)
        sample_esc = html.escape(r.sample_id, quote=True)
        split_esc = html.escape(r.split, quote=True)
        prio_esc = html.escape(r.priority, quote=True)
        cfg_esc = html.escape(r.config_folder, quote=True)
        setup_esc = html.escape(r.setup_id, quote=True)
        depth_esc = html.escape(str(r.depth_index), quote=True)
        img_path_esc = html.escape(r.image_path, quote=True)
        fss_path_esc = html.escape(r.fss_path, quote=True)

        cards.append(
            "<section class='card'"
            f" data-item-key='{item_key_esc}'"
            f" data-rank='{r.rank}'"
            f" data-priority='{prio_esc}'"
            f" data-split='{split_esc}'"
            f" data-sample-id='{sample_esc}'"
            f" data-config-folder='{cfg_esc}'"
            f" data-setup-id='{setup_esc}'"
            f" data-depth-index='{depth_esc}'"
            f" data-image-path='{img_path_esc}'"
            f" data-fss-path='{fss_path_esc}'"
            f" data-line21='{line21_esc}'"
            f" data-x-gt='{r.x_gt:.4f}'"
            f" data-y-top-gt='{r.y_top_gt:.4f}'"
            f" data-y-bottom-gt='{r.y_bottom_gt:.4f}'"
            f" data-x-pred='{r.x_pred:.4f}'"
            f" data-y-top-pred='{r.y_top_pred:.4f}'"
            f" data-y-bottom-pred='{r.y_bottom_pred:.4f}'>"
            f"<h2>#{r.rank} [{r.priority}] {r.sample_id} ({r.split})</h2>"
            f"<p class='meta'>score={r.score:.1f} | x_err={r.x_err_px:.1f}px | y_err={r.y_err_px:.1f}px | setup={r.setup_id} depth={r.depth_index}</p>"
            f"<p class='meta'>config: {r.config_folder}</p>"
            f"<p class='meta'>GT: x={r.x_gt:.1f} | y_top={r.y_top_gt:.1f} | y_bottom={r.y_bottom_gt:.1f}</p>"
            f"<p class='meta'>Pred: x={r.x_pred:.1f} | y_top={r.y_top_pred:.1f} | y_bottom={r.y_bottom_pred:.1f}</p>"
            f"<p class='meta'>fss: <code>{r.fss_path}</code></p>"
            f"<p class='meta'>line21: <code>{r.scale_line_raw}</code></p>"
            "<div class='actions'>"
            f"<label class='exclude-label'><input type='checkbox' class='exclude-toggle' data-item-key='{item_key_esc}' /> Escludi dal training</label>"
            f"<input type='text' class='exclude-note' data-item-key='{item_key_esc}' placeholder='Motivo esclusione (opzionale)' />"
            "</div>"
            "<div class='gt-editor'>"
            "<div class='gt-row'>"
            "<label class='gt-field'>x GT nuovo"
            f"<input type='number' step='0.1' class='gt-input gt-x-new' data-item-key='{item_key_esc}' placeholder='{r.x_gt:.1f}' />"
            "</label>"
            "<label class='gt-field'>y top nuovo"
            f"<input type='number' step='0.1' class='gt-input gt-y-top-new' data-item-key='{item_key_esc}' placeholder='{r.y_top_gt:.1f}' />"
            "</label>"
            "<label class='gt-field'>y bottom nuovo"
            f"<input type='number' step='0.1' class='gt-input gt-y-bottom-new' data-item-key='{item_key_esc}' placeholder='{r.y_bottom_gt:.1f}' />"
            "</label>"
            "<div class='gt-buttons'>"
            f"<button type='button' class='gt-btn gt-use-pred' data-item-key='{item_key_esc}'>Usa Pred</button>"
            f"<button type='button' class='gt-btn gt-reset' data-item-key='{item_key_esc}'>Reset GT</button>"
            "</div>"
            "</div>"
            "</div>"
            "<div class='imgs'>"
            f"<a href='{full_rel}' target='_blank'><img src='{full_rel}' loading='lazy' alt='full overlay' /></a>"
            f"<a href='{zoom_rel}' target='_blank'><img src='{zoom_rel}' loading='lazy' alt='zoom overlay' /></a>"
            "</div>"
            "</section>"
        )

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale GT Fix Queue - BIG ({ui_version})</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --cardExcluded: #fff7ed;
      --text: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #b45309;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.35;
    }}
    .wrap {{
      max-width: 1900px;
      margin: 0 auto;
      padding: 16px 18px 36px 18px;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(248, 250, 252, 0.96);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      margin-bottom: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .toolbar button {{
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
    }}
    .toolbar .stat {{
      color: var(--muted);
      font-size: 14px;
      margin-right: 8px;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 30px;
    }}
    .legend {{
      margin: 0 0 18px 0;
      color: var(--muted);
      font-size: 17px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin: 0 0 18px 0;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      display: none;
    }}
    .card:first-of-type {{
      display: block;
    }}
    .card h2 {{
      margin: 0 0 8px 0;
      font-size: 24px;
    }}
    .card.excluded {{
      background: var(--cardExcluded);
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(180, 83, 9, 0.15);
    }}
    .meta {{
      margin: 2px 0;
      color: var(--muted);
      font-size: 15px;
      word-break: break-word;
    }}
    .actions {{
      margin-top: 10px;
      margin-bottom: 10px;
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 10px;
      align-items: center;
    }}
    .exclude-label {{
      font-size: 16px;
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .exclude-note {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      width: 100%;
      box-sizing: border-box;
      background: #ffffff;
    }}
    .gt-editor {{
      margin-top: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: #f8fafc;
    }}
    .gt-row {{
      display: grid;
      grid-template-columns: minmax(140px, 1fr) minmax(140px, 1fr) minmax(160px, 1fr) auto;
      gap: 10px;
      align-items: end;
    }}
    .gt-field {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .gt-input {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 9px;
      font-size: 14px;
      background: #ffffff;
      color: var(--text);
    }}
    .gt-buttons {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
    }}
    .gt-btn {{
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
      white-space: nowrap;
    }}
    .gt-btn:hover {{
      background: #eef2ff;
    }}
    .imgs {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      margin-top: 12px;
    }}
    img {{
      width: 100%;
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      display: block;
      background: #000;
    }}
    @media (min-width: 1700px) {{
      .imgs {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 1100px) {{
      .actions {{
        grid-template-columns: 1fr;
      }}
      .gt-row {{
        grid-template-columns: 1fr;
      }}
      .gt-buttons {{
        justify-content: flex-start;
      }}
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 5px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Scale GT Fix Queue - BIG Preview [{ui_version}]</h1>
    <p class="legend">Overlay: green=GT, magenta=prediction. Una card alla volta: usa frecce tastiera &larr; &rarr; (anche &uarr; &darr;).</p>
    <div class="toolbar">
      <span class="stat" id="statSummary">Esclusi: 0 / {len(rows)}</span>
      <span class="stat" id="cardPos">Card: 0 / 0</span>
      <button id="btnPrevCard" type="button">Card precedente</button>
      <button id="btnNextCard" type="button">Card successiva</button>
      <button id="btnShowAll" type="button">Mostra Tutti</button>
      <button id="btnShowExcluded" type="button">Mostra Solo Esclusi</button>
      <button id="btnClear" type="button">Azzera Flag</button>
      <button id="btnExportCsv" type="button">Esporta CSV Esclusioni</button>
      <button id="btnExportCorrCsv" type="button">Esporta CSV Correzioni GT</button>
      <button id="btnCopyPaths" type="button">Copia Path Immagini</button>
    </div>
    {''.join(cards)}
  </div>
  <script>
    (function() {{
      try {{
      const STORAGE_KEY = "scale_gt_exclusions_v1";
      const UI_VERSION = "{ui_version}";
      const cards = Array.from(document.querySelectorAll(".card"));
      const statSummary = document.getElementById("statSummary");
      const cardPos = document.getElementById("cardPos");
      const btnPrevCard = document.getElementById("btnPrevCard");
      const btnNextCard = document.getElementById("btnNextCard");
      const btnShowAll = document.getElementById("btnShowAll");
      const btnShowExcluded = document.getElementById("btnShowExcluded");
      const btnClear = document.getElementById("btnClear");
      const btnExportCsv = document.getElementById("btnExportCsv");
      const btnExportCorrCsv = document.getElementById("btnExportCorrCsv");
      const btnCopyPaths = document.getElementById("btnCopyPaths");
      let showOnlyExcluded = false;
      let currentCardIdx = 0;

      function mergeObjects(a, b) {{
        const out = {{}};
        let k = "";
        for (k in a) {{
          if (Object.prototype.hasOwnProperty.call(a, k)) out[k] = a[k];
        }}
        for (k in b) {{
          if (Object.prototype.hasOwnProperty.call(b, k)) out[k] = b[k];
        }}
        return out;
      }}

      function loadState() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) return {{}};
          const obj = JSON.parse(raw);
          if (obj && typeof obj === "object") return obj;
        }} catch (e) {{}}
        return {{}};
      }}

      function saveState(state) {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      }}

      function csvEscape(value) {{
        const s = String(value === undefined || value === null ? "" : value);
        if (s.indexOf(",") >= 0 || s.indexOf('"') >= 0 || s.indexOf("\\n") >= 0) {{
          return '"' + s.split('"').join('""') + '"';
        }}
        return s;
      }}

      function parseNumber(value) {{
        if (value === undefined || value === null) return null;
        const t = String(value).trim();
        if (t.length === 0) return null;
        const n = Number(t);
        return Number.isFinite(n) ? n : null;
      }}

      function getFilteredCards() {{
        return cards.filter((card) => card.dataset.filterVisible === "1");
      }}

      function isEditableTarget(el) {{
        if (!el) return false;
        const tag = String(el.tagName || "").toUpperCase();
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
        if (el.isContentEditable) return true;
        const parentEditable = el.closest ? el.closest("input, textarea, select, [contenteditable='true']") : null;
        return !!parentEditable;
      }}

      function renderCardSelection(shouldScroll) {{
        const filteredCards = getFilteredCards();
        cards.forEach((card) => {{
          card.style.display = "none";
        }});
        if (!filteredCards.length) {{
          cardPos.textContent = "Card: 0 / 0";
          return;
        }}

        if (currentCardIdx < 0) currentCardIdx = 0;
        if (currentCardIdx >= filteredCards.length) currentCardIdx = filteredCards.length - 1;
        const target = filteredCards[currentCardIdx];
        target.style.display = "block";
        cardPos.textContent = "Card: " + (currentCardIdx + 1) + " / " + filteredCards.length;
        if (shouldScroll) {{
          target.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
      }}

      function stepCard(direction) {{
        const filteredCards = getFilteredCards();
        if (!filteredCards.length) return;
        const nextIdx = Math.max(0, Math.min(filteredCards.length - 1, currentCardIdx + direction));
        if (nextIdx === currentCardIdx) return;
        currentCardIdx = nextIdx;
        renderCardSelection(true);
      }}

      const state = loadState();

      function hasValidCorrection(card, saved) {{
        const xOld = parseNumber(card.dataset.xGt);
        const yTopOld = parseNumber(card.dataset.yTopGt);
        const yBottomOld = parseNumber(card.dataset.yBottomGt);
        const xNew = parseNumber(saved.x_gt_new);
        const yTopNew = parseNumber(saved.y_top_gt_new);
        const yBottomNew = parseNumber(saved.y_bottom_gt_new);
        if (xOld === null || yTopOld === null || yBottomOld === null) return false;
        if (xNew === null || yTopNew === null || yBottomNew === null) return false;
        return (
          Math.abs(xNew - xOld) > 1e-6 ||
          Math.abs(yTopNew - yTopOld) > 1e-6 ||
          Math.abs(yBottomNew - yBottomOld) > 1e-6
        );
      }}

      function applyCardState(card) {{
        const key = card.dataset.itemKey || "";
        const toggle = card.querySelector(".exclude-toggle");
        const noteInput = card.querySelector(".exclude-note");
        const xNewInput = card.querySelector(".gt-x-new");
        const yTopNewInput = card.querySelector(".gt-y-top-new");
        const yBottomNewInput = card.querySelector(".gt-y-bottom-new");
        const saved = state[key] || {{}};
        const excluded = !!saved.exclude;
        toggle.checked = excluded;
        noteInput.value = saved.note || "";
        xNewInput.value = saved.x_gt_new === undefined || saved.x_gt_new === null ? "" : String(saved.x_gt_new);
        yTopNewInput.value =
          saved.y_top_gt_new === undefined || saved.y_top_gt_new === null ? "" : String(saved.y_top_gt_new);
        yBottomNewInput.value =
          saved.y_bottom_gt_new === undefined || saved.y_bottom_gt_new === null ? "" : String(saved.y_bottom_gt_new);
        card.classList.toggle("excluded", excluded);
        const isFilteredVisible = !(showOnlyExcluded && !excluded);
        card.dataset.filterVisible = isFilteredVisible ? "1" : "0";
      }}

      function refresh() {{
        cards.forEach(applyCardState);
        const excludedCount = cards.filter((card) => {{
          const key = card.dataset.itemKey || "";
          return !!(state[key] && state[key].exclude);
        }}).length;
        const correctionCount = cards.filter((card) => {{
          const key = card.dataset.itemKey || "";
          const saved = state[key] || {{}};
          return hasValidCorrection(card, saved);
        }}).length;
        const filteredCount = getFilteredCards().length;
        if (currentCardIdx >= filteredCount) {{
          currentCardIdx = Math.max(0, filteredCount - 1);
        }}
        renderCardSelection(false);
        statSummary.textContent =
          "Esclusi: " + excludedCount + " / " + cards.length + " | Correzioni GT: " + correctionCount + " | Card filtrate: " + filteredCount;
      }}

      cards.forEach((card) => {{
        const key = card.dataset.itemKey || "";
        const toggle = card.querySelector(".exclude-toggle");
        const noteInput = card.querySelector(".exclude-note");
        const xNewInput = card.querySelector(".gt-x-new");
        const yTopNewInput = card.querySelector(".gt-y-top-new");
        const yBottomNewInput = card.querySelector(".gt-y-bottom-new");
        const btnUsePred = card.querySelector(".gt-use-pred");
        const btnReset = card.querySelector(".gt-reset");

        function saveCardState() {{
          const current = state[key] || {{}};
          state[key] = mergeObjects(current, {{
            exclude: !!toggle.checked,
            note: noteInput.value || "",
            x_gt_new: xNewInput.value || "",
            y_top_gt_new: yTopNewInput.value || "",
            y_bottom_gt_new: yBottomNewInput.value || "",
            updated_at: new Date().toISOString(),
          }});
          saveState(state);
          refresh();
        }}

        toggle.addEventListener("change", saveCardState);
        noteInput.addEventListener("change", saveCardState);
        xNewInput.addEventListener("change", saveCardState);
        yTopNewInput.addEventListener("change", saveCardState);
        yBottomNewInput.addEventListener("change", saveCardState);

        btnUsePred.addEventListener("click", () => {{
          xNewInput.value = card.dataset.xPred || "";
          yTopNewInput.value = card.dataset.yTopPred || "";
          yBottomNewInput.value = card.dataset.yBottomPred || "";
          saveCardState();
        }});

        btnReset.addEventListener("click", () => {{
          xNewInput.value = "";
          yTopNewInput.value = "";
          yBottomNewInput.value = "";
          saveCardState();
        }});
      }});

      btnShowAll.addEventListener("click", () => {{
        showOnlyExcluded = false;
        currentCardIdx = 0;
        refresh();
      }});

      btnShowExcluded.addEventListener("click", () => {{
        showOnlyExcluded = true;
        currentCardIdx = 0;
        refresh();
      }});

      btnPrevCard.addEventListener("click", () => {{
        stepCard(-1);
      }});

      btnNextCard.addEventListener("click", () => {{
        stepCard(1);
      }});

      btnClear.addEventListener("click", () => {{
        if (!confirm("Azzero tutti i flag e tutte le correzioni GT salvate in questa pagina?")) return;
        cards.forEach((card) => {{
          const key = card.dataset.itemKey || "";
          state[key] = {{
            exclude: false,
            note: "",
            x_gt_new: "",
            y_top_gt_new: "",
            y_bottom_gt_new: "",
            updated_at: new Date().toISOString(),
          }};
        }});
        saveState(state);
        refresh();
      }});

      function getExcludedRows() {{
        return cards
          .map((card) => {{
            const key = card.dataset.itemKey || "";
            const saved = state[key] || {{}};
            if (!saved.exclude) return null;
            return {{
              rank: card.dataset.rank || "",
              priority: card.dataset.priority || "",
              split: card.dataset.split || "",
              sample_id: card.dataset.sampleId || "",
              config_folder: card.dataset.configFolder || "",
              setup_id: card.dataset.setupId || "",
              depth_index: card.dataset.depthIndex || "",
              image_path: card.dataset.imagePath || "",
              fss_path: card.dataset.fssPath || "",
              line21: card.dataset.line21 || "",
              note: saved.note || "",
            }};
          }})
          .filter(Boolean);
      }}

      function getCorrectionRows() {{
        return cards
          .map((card) => {{
            const key = card.dataset.itemKey || "";
            const saved = state[key] || {{}};
            if (!hasValidCorrection(card, saved)) return null;

            const xOld = parseNumber(card.dataset.xGt);
            const yTopOld = parseNumber(card.dataset.yTopGt);
            const yBottomOld = parseNumber(card.dataset.yBottomGt);
            const xNew = parseNumber(saved.x_gt_new);
            const yTopNew = parseNumber(saved.y_top_gt_new);
            const yBottomNew = parseNumber(saved.y_bottom_gt_new);
            if (xOld === null || yTopOld === null || yBottomOld === null) return null;
            if (xNew === null || yTopNew === null || yBottomNew === null) return null;

            return {{
              rank: card.dataset.rank || "",
              priority: card.dataset.priority || "",
              split: card.dataset.split || "",
              sample_id: card.dataset.sampleId || "",
              config_folder: card.dataset.configFolder || "",
              setup_id: card.dataset.setupId || "",
              depth_index: card.dataset.depthIndex || "",
              image_path: card.dataset.imagePath || "",
              fss_path: card.dataset.fssPath || "",
              line21: card.dataset.line21 || "",
              x_gt_old: xOld,
              y_top_gt_old: yTopOld,
              y_bottom_gt_old: yBottomOld,
              x_gt_new: xNew,
              y_top_gt_new: yTopNew,
              y_bottom_gt_new: yBottomNew,
              note: saved.note || "",
            }};
          }})
          .filter(Boolean);
      }}

      btnExportCsv.addEventListener("click", () => {{
        const rows = getExcludedRows();
        const header = [
          "rank",
          "priority",
          "split",
          "sample_id",
          "config_folder",
          "setup_id",
          "depth_index",
          "image_path",
          "fss_path",
          "line21",
          "note",
        ];
        const lines = [header.join(",")];
        rows.forEach((row) => {{
          lines.push(header.map((h) => csvEscape(row[h])).join(","));
        }});
        const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8;" }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        a.href = url;
        a.download = "scale_exclusions_" + ts + ".csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }});

      btnExportCorrCsv.addEventListener("click", () => {{
        const rows = getCorrectionRows();
        const header = [
          "rank",
          "priority",
          "split",
          "sample_id",
          "config_folder",
          "setup_id",
          "depth_index",
          "image_path",
          "fss_path",
          "line21",
          "x_gt_old",
          "y_top_gt_old",
          "y_bottom_gt_old",
          "x_gt_new",
          "y_top_gt_new",
          "y_bottom_gt_new",
          "note",
        ];
        const lines = [header.join(",")];
        rows.forEach((row) => {{
          lines.push(header.map((h) => csvEscape(row[h])).join(","));
        }});
        const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8;" }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        a.href = url;
        a.download = "scale_gt_corrections_" + ts + ".csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }});

      btnCopyPaths.addEventListener("click", async () => {{
        const rows = getExcludedRows();
        const text = rows.map((r) => r.image_path).join("\\n");
        try {{
          await navigator.clipboard.writeText(text);
          alert("Path immagini copiati negli appunti: " + rows.length);
        }} catch (e) {{
          alert("Impossibile copiare automaticamente. Usa 'Esporta CSV Esclusioni'.");
        }}
      }});

      function handleArrowNav(event) {{
        if (isEditableTarget(event.target)) return;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {{
          event.preventDefault();
          stepCard(1);
          return;
        }}
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") {{
          event.preventDefault();
          stepCard(-1);
        }}
      }}

      window.addEventListener("keydown", handleArrowNav, true);
      document.addEventListener("keydown", handleArrowNav);

      if (document.body) {{
        document.body.setAttribute("tabindex", "-1");
      }}
      document.addEventListener("click", (event) => {{
        if (isEditableTarget(event.target)) return;
        if (document.body) {{
          try {{
            document.body.focus();
          }} catch (e) {{}}
        }}
      }});

      console.log("Scale GT UI version:", UI_VERSION);

      refresh();
      }} catch (e) {{
        console.error("Scale GT UI fatal error", e);
        alert("Errore nella pagina di review: " + (e && e.message ? e.message : e));
      }}
    }})();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prioritized GT-fix queue for scale line.")
    p.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/32_scale_image_model_medium_cpu/train_predictions_best.csv"),
            Path("artifacts/32_scale_image_model_medium_cpu/val_predictions_best.csv"),
            Path("artifacts/32_scale_image_model_medium_cpu/test_predictions_best.csv"),
        ],
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/33_scale_gt_fix_queue"))
    p.add_argument("--top-n", type=int, default=220)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    pred_csvs = [p.expanduser().resolve() for p in args.predictions]
    raw_rows = load_prediction_rows(pred_csvs)
    if not raw_rows:
        raise RuntimeError("No prediction rows loaded.")

    fss_cache: Dict[Tuple[str, str], str] = {}
    raw_line_cache: Dict[str, str] = {}
    rows: List[Row] = []
    for d in raw_rows:
        sample_id = str(d["sample_id"])
        split = str(d["split"])
        image_path = str(d["image_path"])
        config_folder = config_from_image_path(image_path)
        setup_id, depth_index = parse_sample_id(sample_id)
        fss_path = find_fss_for_row(image_path, setup_id, fss_cache) if setup_id else ""
        scale_raw = read_scale_line_raw(fss_path, raw_line_cache)

        x_err = float(d["x_err_px"])
        y_err = float(d["y_err_px"])
        score = x_err + y_err
        flagged = is_flagged_config(config_folder)
        prio = priority_from_errors(score, x_err, flagged)
        rows.append(
            Row(
                rank=0,
                sample_id=sample_id,
                split=split,
                config_folder=config_folder,
                image_path=image_path,
                setup_id=setup_id,
                depth_index=depth_index,
                fss_path=fss_path,
                scale_line_raw=scale_raw,
                x_gt=float(d["x_gt"]),
                y_top_gt=float(d["y_top_gt"]),
                y_bottom_gt=float(d["y_bottom_gt"]),
                x_pred=float(d["x_pred"]),
                y_top_pred=float(d["y_top_pred"]),
                y_bottom_pred=float(d["y_bottom_pred"]),
                x_err_px=x_err,
                y_err_px=y_err,
                score=score,
                flagged_config=flagged,
                priority=prio,
            )
        )

    pri_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rows.sort(key=lambda r: (pri_order.get(r.priority, 9), -r.score, -r.x_err_px))
    for i, r in enumerate(rows, start=1):
        r.rank = i

    top_rows = rows[: max(0, args.top_n)]
    for r in top_rows:
        full_name = f"{r.rank:04d}_{_safe_slug(r.sample_id)}_full.jpg"
        zoom_name = f"{r.rank:04d}_{_safe_slug(r.sample_id)}_zoom.jpg"
        full_out = preview_dir / full_name
        zoom_out = preview_dir / zoom_name
        try:
            draw_overlay_previews(r, full_out, zoom_out)
        except Exception:
            continue

    write_csv(out_dir / "gt_fix_queue_all.csv", rows)
    write_csv(out_dir / "gt_fix_queue_top.csv", top_rows)
    write_html(out_dir / "gt_fix_queue_review.html", top_rows, preview_dir)
    write_html_big(out_dir / "gt_fix_queue_review_big.html", top_rows, preview_dir)
    write_html_big(out_dir / "gt_fix_queue_review_big_onecard.html", top_rows, preview_dir)

    by_fss: Dict[str, Dict[str, float | int | str]] = {}
    for r in rows:
        d = by_fss.setdefault(
            r.fss_path,
            {"fss_path": r.fss_path, "rows": 0, "p0": 0, "p1": 0, "sum_score": 0.0, "max_score": 0.0},
        )
        d["rows"] = int(d["rows"]) + 1
        d["sum_score"] = float(d["sum_score"]) + float(r.score)
        d["max_score"] = max(float(d["max_score"]), float(r.score))
        if r.priority == "P0":
            d["p0"] = int(d["p0"]) + 1
        if r.priority == "P1":
            d["p1"] = int(d["p1"]) + 1

    by_fss_rows = []
    for d in by_fss.values():
        rows_n = int(d["rows"])
        by_fss_rows.append(
            {
                "fss_path": str(d["fss_path"]),
                "rows": rows_n,
                "p0": int(d["p0"]),
                "p1": int(d["p1"]),
                "mean_score": float(d["sum_score"]) / max(1, rows_n),
                "max_score": float(d["max_score"]),
            }
        )
    by_fss_rows.sort(key=lambda x: (-x["p0"], -x["p1"], -x["mean_score"]))
    by_fss_csv = out_dir / "gt_fix_queue_by_fss.csv"
    with by_fss_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fss_path", "rows", "p0", "p1", "mean_score", "max_score"])
        for d in by_fss_rows:
            w.writerow(
                [
                    d["fss_path"],
                    d["rows"],
                    d["p0"],
                    d["p1"],
                    f"{d['mean_score']:.4f}",
                    f"{d['max_score']:.4f}",
                ]
            )

    by_priority: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    for r in rows:
        by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
        by_split[r.split] = by_split.get(r.split, 0) + 1

    # Worst configs by mean score (at least 5 rows).
    per_cfg: Dict[str, List[float]] = {}
    for r in rows:
        per_cfg.setdefault(r.config_folder, []).append(r.score)
    worst_cfg = []
    for cfg, vals in per_cfg.items():
        if len(vals) >= 5:
            worst_cfg.append({"config_folder": cfg, "rows": len(vals), "mean_score": float(sum(vals) / len(vals))})
    worst_cfg.sort(key=lambda d: d["mean_score"], reverse=True)

    summary = {
        "predictions": [p.as_posix() for p in pred_csvs],
        "rows_total": len(rows),
        "rows_top": len(top_rows),
        "by_priority": by_priority,
        "by_split": by_split,
        "missing_fss_path": sum(1 for r in rows if not r.fss_path),
        "missing_scale_line_raw": sum(1 for r in rows if not r.scale_line_raw),
        "top10_worst_configs": worst_cfg[:10],
        "outputs": {
            "queue_all_csv": (out_dir / "gt_fix_queue_all.csv").as_posix(),
            "queue_top_csv": (out_dir / "gt_fix_queue_top.csv").as_posix(),
            "queue_by_fss_csv": by_fss_csv.as_posix(),
            "review_html": (out_dir / "gt_fix_queue_review.html").as_posix(),
            "review_html_big": (out_dir / "gt_fix_queue_review_big.html").as_posix(),
            "review_html_big_onecard": (out_dir / "gt_fix_queue_review_big_onecard.html").as_posix(),
            "preview_dir": preview_dir.as_posix(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
