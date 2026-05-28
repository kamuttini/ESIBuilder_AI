#!/usr/bin/env python3
"""Build full HTML review package for ultrasound rectangle predictions.

Output:
- index.html grouped by vendor -> folder
- one HTML page per folder
Each folder page contains:
1) all images with predicted/global boxes and "flag bad" controls
2) correction section with only flagged images and editable rectangle values
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    image_path: Path
    image_uri: str
    rel_path: str
    group_name: str
    orientation_name: str
    vendor: str
    vendor_confidence: float
    vendor_margin: float
    vendor_vote_ratio: float
    width: int
    height: int
    pred_top: int
    pred_left: int
    pred_bottom: int
    pred_right: int
    global_top: int
    global_left: int
    global_bottom: int
    global_right: int
    rect_route_type: str = ""
    rect_route_vendor: str = ""
    rect_model_checkpoint: str = ""
    rect_model_image_size: int = 0


@dataclass(frozen=True)
class FolderPageInfo:
    vendor: str
    group_name: str
    filename: str
    num_images: int
    preview_image_uri: str
    preview_rel_path: str
    preview_width: int
    preview_height: int
    preview_pred_top: int
    preview_pred_left: int
    preview_pred_bottom: int
    preview_pred_right: int
    preview_global_top: int
    preview_global_left: int
    preview_global_bottom: int
    preview_global_right: int
    vendor_confidence: float
    vendor_margin: float
    vendor_vote_ratio: float


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def _infer_vendor(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    # Many folders start with a numeric setup prefix, e.g. "103.BK..." -> strip it.
    name = re.sub(r"^\d+\s*[\.\)]\s*", "", name)
    compact = f" {name} "
    if re.match(r"^bk(?:[\s,_-]|\d)", name):
        return "BK"
    if name.startswith("esaote"):
        return "Esaote"
    if name.startswith("hitachi"):
        return "Hitachi"
    if name.startswith("ge"):
        return "GE"
    if name.startswith("mindray"):
        return "Mindray"
    if name.startswith("canon") or name.startswith("toshiba"):
        return "Canon"
    if name.startswith("philips"):
        return "Philips"
    if name.startswith("siemens"):
        return "Siemens"
    if name.startswith("koelis"):
        return "Koelis"
    if name.startswith("biopsee"):
        return "Biopsee"
    if name.startswith("terason"):
        return "Terason"
    if name.startswith("sonostar"):
        return "Sonostar"
    if name.startswith("exactvu"):
        return "ExactVu"
    if name.startswith("alpinion"):
        return "Alpinion"
    if name.startswith("vinno"):
        return "Vinno"

    # Fallback: match common vendor/model keywords anywhere in folder name.
    contains_rules = [
        ("BK", ("bk5000", "bk3000", "bk ", "flexfocus", "specto", "profocus")),
        ("Esaote", ("esaote", "mylab", "echolaser")),
        ("Hitachi", ("fujifilm hitachi", "hitachi", "arietta")),
        ("GE", ("logiq", "voluson", " versana ", " ge ")),
        ("Mindray", ("mindray", "resona", "dc70", "mx7", " m6 ")),
        ("Canon", ("canon", "toshiba", "aplio", "alpio")),
        ("Philips", ("philips", "affiniti", "affinity", " cx50 ")),
        ("Siemens", ("siemens", "acuson")),
        ("Koelis", ("koelis",)),
        ("Biopsee", ("biopsee",)),
        ("Terason", ("terason",)),
        ("Sonostar", ("sonostar",)),
        ("ExactVu", ("exact vu", "exactvu", "edap")),
        ("Alpinion", ("alpinion", "ecube")),
        ("Vinno", ("vinno",)),
        ("Supersonic", ("supersonic", "aixplorer")),
    ]
    for vendor, keys in contains_rules:
        if any((f" {k} " in compact) or (k in name) for k in keys):
            return vendor

    token = re.split(r"[\s,_-]+", folder_name.strip())[0]
    return token if token else "UNKNOWN"


def _slugify(value: str) -> str:
    v = _normalize_name(value)
    v = re.sub(r"[^a-z0-9]+", "-", v).strip("-")
    return v or "item"


def _rel_src_from_output(rel_path: str, depth: int) -> str:
    normalized = rel_path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    prefix = "../" * max(0, int(depth))
    return f"{prefix}{normalized}"


def _to_int(value: str) -> int:
    return int(round(float(value)))


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _load_predictions(csv_path: Path) -> List[PredictionRow]:
    rows: List[PredictionRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "image_path",
            "rel_path",
            "group_name",
            "orientation_name",
            "width",
            "height",
            "pred_top",
            "pred_left",
            "pred_bottom",
            "pred_right",
            "global_top",
            "global_left",
            "global_bottom",
            "global_right",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Predictions CSV missing required columns: {sorted(missing)}")
        has_vendor_predicted = "vendor_predicted" in (reader.fieldnames or [])
        for row in reader:
            image_path = Path(row["image_path"]).expanduser().resolve()
            group_name = row["group_name"]
            vendor_pred = row.get("vendor_predicted", "").strip() if has_vendor_predicted else ""
            vendor = vendor_pred if vendor_pred else _infer_vendor(group_name)
            rows.append(
                PredictionRow(
                    sample_id=row["sample_id"],
                    image_path=image_path,
                    image_uri=image_path.as_uri(),
                    rel_path=row["rel_path"],
                    group_name=group_name,
                    orientation_name=row["orientation_name"],
                    vendor=vendor,
                    vendor_confidence=_to_float(row.get("vendor_top1_prob", "0")),
                    vendor_margin=_to_float(row.get("vendor_margin_top1_top2", "0")),
                    vendor_vote_ratio=_to_float(row.get("vendor_vote_ratio", "0")),
                    width=_to_int(row["width"]),
                    height=_to_int(row["height"]),
                    pred_top=_to_int(row["pred_top"]),
                    pred_left=_to_int(row["pred_left"]),
                    pred_bottom=_to_int(row["pred_bottom"]),
                    pred_right=_to_int(row["pred_right"]),
                    global_top=_to_int(row["global_top"]),
                    global_left=_to_int(row["global_left"]),
                    global_bottom=_to_int(row["global_bottom"]),
                    global_right=_to_int(row["global_right"]),
                    rect_route_type=str(row.get("rect_route_type", "")),
                    rect_route_vendor=str(row.get("rect_route_vendor", "")),
                    rect_model_checkpoint=str(row.get("rect_model_checkpoint", "")),
                    rect_model_image_size=_to_int(row.get("rect_model_image_size", 0)),
                )
            )
    return rows


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _build_folder_payload(rows: Sequence[PredictionRow], vendor: str, group_name: str) -> Dict[str, object]:
    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                "sample_id": row.sample_id,
                "image_uri": row.image_uri,
                "image_src_rel": _rel_src_from_output(row.rel_path, depth=3),
                "image_path": row.image_path.as_posix(),
                "rel_path": row.rel_path,
                "orientation_name": row.orientation_name,
                "width": row.width,
                "height": row.height,
                "pred": {
                    "top": row.pred_top,
                    "left": row.pred_left,
                    "bottom": row.pred_bottom,
                    "right": row.pred_right,
                },
                "global": {
                    "top": row.global_top,
                    "left": row.global_left,
                    "bottom": row.global_bottom,
                    "right": row.global_right,
                },
                "rect_route_type": row.rect_route_type,
                "rect_route_vendor": row.rect_route_vendor,
                "rect_model_checkpoint": row.rect_model_checkpoint,
                "rect_model_image_size": row.rect_model_image_size,
            }
        )
    return {
        "vendor": vendor,
        "group_name": group_name,
        "rows": payload_rows,
    }


def _folder_page_html(folder_payload: Dict[str, object]) -> str:
    payload_json = _json_for_script(folder_payload)
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Rect Review Folder</title>
  <style>
    :root {{
      --bg: #0f1722;
      --panel: #172333;
      --panel2: #121c2a;
      --border: #2b3a4f;
      --text: #e7edf5;
      --muted: #9fb0c3;
      --accent: #3b82f6;
      --danger: #ef4444;
      --ok: #22c55e;
      --pred: #ef4444;
      --glob: #22d3ee;
      --corr: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 14px; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    h2 {{ margin: 14px 0 8px; font-size: 18px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      color: #fff;
      background: var(--accent);
      font-weight: 600;
    }}
    button.alt {{ background: #4b5f7a; }}
    button.warn {{ background: var(--danger); }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 12px; margin-top: 8px; }}
    .chip {{ padding: 4px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--panel2); }}
    .chip i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; margin-top: 10px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    .card.is-excluded {{ border-color: #ef4444; box-shadow: 0 0 0 1px #ef444488 inset; }}
    .img-wrap {{ position: relative; width: 100%; background: #000; }}
    .img-wrap img {{ width: 100%; display: block; }}
    .box {{
      position: absolute;
      border: 2px solid;
      pointer-events: none;
    }}
    .box.pred {{ border-color: var(--pred); }}
    .box.glob {{ border-color: var(--glob); }}
    .box.corr {{ border-color: var(--corr); }}
    .meta {{ padding: 10px; font-size: 12px; line-height: 1.35; }}
    .row {{ margin-top: 6px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    input[type="text"], input[type="number"] {{
      background: #0d1520;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px;
    }}
    .nums {{ display: grid; grid-template-columns: repeat(4, minmax(76px, 1fr)); gap: 6px; max-width: 430px; }}
    .section {{
      margin-top: 14px;
      padding: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      font-size: 11px;
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--muted);
    }}
    .topline {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 4px; }}
    .toolbar label {{ font-size: 12px; color: var(--text); }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <a href="../index.html" style="color:#8ab4ff">← Torna all'indice</a>
    <h1 id="title"></h1>
    <div class="muted" id="subtitle"></div>
    <div class="topline">
      <span class="pill" id="stats-total"></span>
      <span class="pill" id="stats-flagged"></span>
      <span class="pill" id="stats-corrected"></span>
      <span class="pill" id="stats-excluded"></span>
    </div>

    <div class="toolbar">
      <button id="btn-export-flags">Export flags CSV</button>
      <button id="btn-export-corrections">Export corrections CSV</button>
      <button id="btn-export-exclusions">Export exclusions CSV</button>
      <button id="btn-refresh" class="alt">Aggiorna sezione correzioni</button>
      <button id="btn-clear" class="warn">Reset flags/correzioni (cartella)</button>
      <label><input id="folder-exclude-toggle" type="checkbox"> Escludi tutta la cartella dal training</label>
      <input id="folder-exclude-note" type="text" placeholder="nota esclusione cartella">
    </div>
    <div class="legend">
      <span class="chip"><i style="background:var(--pred)"></i>rosso = box predetto</span>
      <span class="chip"><i style="background:var(--glob)"></i>ciano = box globale usato</span>
      <span class="chip"><i style="background:var(--corr)"></i>arancio = box corretto manualmente</span>
    </div>

    <div class="section">
      <h2>1) Tutte le immagini (flag bbox + esclusione training)</h2>
      <div class="muted">Le immagini sono tutte quelle della cartella selezionata.</div>
      <div class="grid" id="gallery-grid"></div>
    </div>

    <div class="section">
      <h2>2) Correzione box solo immagini flaggate</h2>
      <div class="muted">Qui compaiono solo le immagini con flag attivo.</div>
      <div class="grid" id="flagged-grid"></div>
    </div>
  </div>

  <script>
    const PAYLOAD = {payload_json};
    const VENDOR = PAYLOAD.vendor;
    const GROUP_NAME = PAYLOAD.group_name;
    const ROWS = PAYLOAD.rows;
    const STORAGE_KEY = `ultra-rect-review::${{VENDOR}}::${{GROUP_NAME}}`;

    function loadState() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {{ items: {{}}, folder: {{ exclude_training: false, note: '' }} }};
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return {{ items: {{}}, folder: {{ exclude_training: false, note: '' }} }};
        if (!parsed.items || typeof parsed.items !== 'object') parsed.items = {{}};
        if (!parsed.folder || typeof parsed.folder !== 'object') parsed.folder = {{}};
        if (typeof parsed.folder.exclude_training !== 'boolean') parsed.folder.exclude_training = false;
        if (typeof parsed.folder.note !== 'string') parsed.folder.note = '';
        return parsed;
      }} catch (_err) {{
        return {{ items: {{}}, folder: {{ exclude_training: false, note: '' }} }};
      }}
    }}

    function saveState(state) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function getItemState(state, sampleId) {{
      if (!state.items[sampleId]) state.items[sampleId] = {{}};
      return state.items[sampleId];
    }}

    function clampBox(row, box) {{
      const w = Number(row.width);
      const h = Number(row.height);
      let top = Math.max(0, Math.min(h - 1, Number(box.top)));
      let left = Math.max(0, Math.min(w - 1, Number(box.left)));
      let bottom = Math.max(1, Math.min(h, Number(box.bottom)));
      let right = Math.max(1, Math.min(w, Number(box.right)));
      if (bottom <= top) bottom = Math.min(h, top + 1);
      if (right <= left) right = Math.min(w, left + 1);
      return {{ top, left, bottom, right }};
    }}

    function toPercentStyle(row, box) {{
      const w = Number(row.width);
      const h = Number(row.height);
      const bw = Math.max(1, box.right - box.left);
      const bh = Math.max(1, box.bottom - box.top);
      return [
        `left:${{(100 * box.left / w).toFixed(6)}}%`,
        `top:${{(100 * box.top / h).toFixed(6)}}%`,
        `width:${{(100 * bw / w).toFixed(6)}}%`,
        `height:${{(100 * bh / h).toFixed(6)}}%`,
      ].join(';');
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function csvEscape(value) {{
      return `"${{String(value ?? '').replaceAll('"', '""')}}"`;
    }}

    function downloadCsv(filename, rows) {{
      const csv = rows.map((r) => r.map(csvEscape).join(',')).join('\\n');
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function defaultCorrection(row) {{
      return {{
        top: Number(row.global.top),
        left: Number(row.global.left),
        bottom: Number(row.global.bottom),
        right: Number(row.global.right),
      }};
    }}

    function ensureCorrection(state, row) {{
      const item = getItemState(state, row.sample_id);
      if (!item.correction) item.correction = defaultCorrection(row);
      item.correction = clampBox(row, item.correction);
      return item.correction;
    }}

    function isRowExcluded(state, row) {{
      if (state.folder && state.folder.exclude_training) return true;
      const item = getItemState(state, row.sample_id);
      return !!item.exclude_training;
    }}

    function stats(state) {{
      let flagged = 0;
      let corrected = 0;
      let excluded = 0;
      ROWS.forEach((row) => {{
        const item = getItemState(state, row.sample_id);
        if (item.flagged) {{
          flagged += 1;
          if (item.correction) corrected += 1;
        }}
        if (isRowExcluded(state, row)) excluded += 1;
      }});
      return {{ total: ROWS.length, flagged, corrected, excluded }};
    }}

    function updateStats(state) {{
      const s = stats(state);
      document.getElementById('stats-total').textContent = `Totale immagini: ${{s.total}}`;
      document.getElementById('stats-flagged').textContent = `Flaggate: ${{s.flagged}}`;
      document.getElementById('stats-corrected').textContent = `Con box corretto: ${{s.corrected}}`;
      const folderMark = state.folder && state.folder.exclude_training ? ' (cartella intera)' : '';
      document.getElementById('stats-excluded').textContent = `Escluse training: ${{s.excluded}}${{folderMark}}`;
    }}

    function attachImageFallback(rootEl) {{
      rootEl.querySelectorAll('img[data-fallback-src]').forEach((img) => {{
        img.addEventListener('error', () => {{
          if (img.dataset.fallbackUsed === '1') return;
          const fallback = img.getAttribute('data-fallback-src') || '';
          if (!fallback) return;
          img.dataset.fallbackUsed = '1';
          img.setAttribute('src', fallback);
        }});
      }});
    }}

    function renderGallery(state) {{
      const grid = document.getElementById('gallery-grid');
      const out = [];
      for (const row of ROWS) {{
        const item = getItemState(state, row.sample_id);
        const isFlagged = !!item.flagged;
        const note = item.note || '';
        const isExcluded = isRowExcluded(state, row);
        const ownExcluded = !!item.exclude_training;
        const excludeNote = item.exclude_note || '';
        out.push(`
          <article class="card ${{isExcluded ? 'is-excluded' : ''}}" data-sample-id="${{escapeHtml(row.sample_id)}}">
            <div class="img-wrap">
              <img loading="lazy" src="${{escapeHtml(row.image_src_rel || row.image_uri)}}" data-fallback-src="${{escapeHtml(row.image_uri)}}" alt="${{escapeHtml(row.sample_id)}}">
              <div class="box pred" style="${{toPercentStyle(row, row.pred)}}"></div>
              <div class="box glob" style="${{toPercentStyle(row, row.global)}}"></div>
            </div>
            <div class="meta">
              <div><b>${{escapeHtml(row.sample_id)}}</b></div>
              <div>${{escapeHtml(row.rel_path)}}</div>
              <div class="muted">route: ${{escapeHtml(row.rect_route_type || '-')}} | vendor: ${{escapeHtml(row.rect_route_vendor || '-')}} | model: ${{escapeHtml(row.rect_model_checkpoint ? row.rect_model_checkpoint.split('/').pop() : '-')}} | size: ${{escapeHtml(String(row.rect_model_image_size || ''))}}</div>
              <div class="row">
                <label><input type="checkbox" class="flag-checkbox" data-sample-id="${{escapeHtml(row.sample_id)}}" ${{isFlagged ? 'checked' : ''}}> non va bene</label>
                <input type="text" class="flag-note" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{escapeHtml(note)}}" placeholder="nota">
              </div>
              <div class="row">
                <label><input type="checkbox" class="exclude-checkbox" data-sample-id="${{escapeHtml(row.sample_id)}}" ${{ownExcluded ? 'checked' : ''}}> escludi dal training</label>
                <input type="text" class="exclude-note" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{escapeHtml(excludeNote)}}" placeholder="motivo esclusione">
              </div>
            </div>
          </article>
        `);
      }}
      grid.innerHTML = out.join('');
      attachImageFallback(grid);

      grid.querySelectorAll('.flag-checkbox').forEach((el) => {{
        el.addEventListener('change', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const row = ROWS.find((r) => r.sample_id === sid);
          const item = getItemState(state, sid);
          item.flagged = !!evt.target.checked;
          if (item.flagged && !item.correction && row) item.correction = defaultCorrection(row);
          saveState(state);
          updateStats(state);
          renderFlagged(state);
        }});
      }});

      grid.querySelectorAll('.flag-note').forEach((el) => {{
        el.addEventListener('input', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const item = getItemState(state, sid);
          item.note = evt.target.value || '';
          saveState(state);
        }});
      }});

      grid.querySelectorAll('.exclude-checkbox').forEach((el) => {{
        el.addEventListener('change', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const item = getItemState(state, sid);
          item.exclude_training = !!evt.target.checked;
          saveState(state);
          updateStats(state);
          renderGallery(state);
        }});
      }});

      grid.querySelectorAll('.exclude-note').forEach((el) => {{
        el.addEventListener('input', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const item = getItemState(state, sid);
          item.exclude_note = evt.target.value || '';
          saveState(state);
        }});
      }});
    }}

    function renderFlagged(state) {{
      const grid = document.getElementById('flagged-grid');
      const flaggedRows = ROWS.filter((row) => !!getItemState(state, row.sample_id).flagged);
      if (flaggedRows.length === 0) {{
        grid.innerHTML = '<div class="muted">Nessuna immagine flaggata in questa cartella.</div>';
        return;
      }}

      const out = [];
      for (const row of flaggedRows) {{
        const item = getItemState(state, row.sample_id);
        const corr = ensureCorrection(state, row);
        const corrNote = item.correction_note || '';
        out.push(`
          <article class="card" data-sample-id="${{escapeHtml(row.sample_id)}}">
            <div class="img-wrap">
              <img loading="lazy" src="${{escapeHtml(row.image_src_rel || row.image_uri)}}" data-fallback-src="${{escapeHtml(row.image_uri)}}" alt="${{escapeHtml(row.sample_id)}}">
              <div class="box pred" style="${{toPercentStyle(row, row.pred)}}"></div>
              <div class="box glob" style="${{toPercentStyle(row, row.global)}}"></div>
              <div class="box corr" id="corr-box-${{escapeHtml(row.sample_id)}}" style="${{toPercentStyle(row, corr)}}"></div>
            </div>
            <div class="meta">
              <div><b>${{escapeHtml(row.sample_id)}}</b></div>
              <div>${{escapeHtml(row.rel_path)}}</div>
              <div class="muted">route: ${{escapeHtml(row.rect_route_type || '-')}} | vendor: ${{escapeHtml(row.rect_route_vendor || '-')}} | model: ${{escapeHtml(row.rect_model_checkpoint ? row.rect_model_checkpoint.split('/').pop() : '-')}} | size: ${{escapeHtml(String(row.rect_model_image_size || ''))}}</div>
              <div class="row nums">
                <input type="number" step="1" min="0" class="corr-top" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{corr.top}}" title="top">
                <input type="number" step="1" min="0" class="corr-left" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{corr.left}}" title="left">
                <input type="number" step="1" min="1" class="corr-bottom" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{corr.bottom}}" title="bottom">
                <input type="number" step="1" min="1" class="corr-right" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{corr.right}}" title="right">
              </div>
              <div class="row">
                <button class="set-global alt" data-sample-id="${{escapeHtml(row.sample_id)}}">Usa globale</button>
                <button class="set-pred alt" data-sample-id="${{escapeHtml(row.sample_id)}}">Usa predetto</button>
                <input type="text" class="corr-note" data-sample-id="${{escapeHtml(row.sample_id)}}" value="${{escapeHtml(corrNote)}}" placeholder="nota correzione">
              </div>
            </div>
          </article>
        `);
      }}
      grid.innerHTML = out.join('');
      attachImageFallback(grid);
      saveState(state);

      function updateOne(sampleId) {{
        const row = ROWS.find((r) => r.sample_id === sampleId);
        if (!row) return;
        const item = getItemState(state, sampleId);
        const topEl = grid.querySelector(`.corr-top[data-sample-id="${{CSS.escape(sampleId)}}"]`);
        const leftEl = grid.querySelector(`.corr-left[data-sample-id="${{CSS.escape(sampleId)}}"]`);
        const bottomEl = grid.querySelector(`.corr-bottom[data-sample-id="${{CSS.escape(sampleId)}}"]`);
        const rightEl = grid.querySelector(`.corr-right[data-sample-id="${{CSS.escape(sampleId)}}"]`);
        const boxEl = document.getElementById(`corr-box-${{sampleId}}`);
        const next = clampBox(row, {{
          top: Number(topEl.value),
          left: Number(leftEl.value),
          bottom: Number(bottomEl.value),
          right: Number(rightEl.value),
        }});
        topEl.value = String(next.top);
        leftEl.value = String(next.left);
        bottomEl.value = String(next.bottom);
        rightEl.value = String(next.right);
        item.correction = next;
        if (boxEl) boxEl.setAttribute('style', toPercentStyle(row, next));
        saveState(state);
        updateStats(state);
      }}

      ['corr-top', 'corr-left', 'corr-bottom', 'corr-right'].forEach((cls) => {{
        grid.querySelectorAll(`.${{cls}}`).forEach((el) => {{
          el.addEventListener('input', (evt) => {{
            const sid = evt.target.getAttribute('data-sample-id');
            updateOne(sid);
          }});
        }});
      }});

      grid.querySelectorAll('.set-global').forEach((el) => {{
        el.addEventListener('click', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const row = ROWS.find((r) => r.sample_id === sid);
          if (!row) return;
          const item = getItemState(state, sid);
          item.correction = defaultCorrection(row);
          renderFlagged(state);
        }});
      }});

      grid.querySelectorAll('.set-pred').forEach((el) => {{
        el.addEventListener('click', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const row = ROWS.find((r) => r.sample_id === sid);
          if (!row) return;
          const item = getItemState(state, sid);
          item.correction = clampBox(row, row.pred);
          renderFlagged(state);
        }});
      }});

      grid.querySelectorAll('.corr-note').forEach((el) => {{
        el.addEventListener('input', (evt) => {{
          const sid = evt.target.getAttribute('data-sample-id');
          const item = getItemState(state, sid);
          item.correction_note = evt.target.value || '';
          saveState(state);
        }});
      }});
    }}

    function exportFlags(state) {{
      const rows = [['sample_id', 'vendor', 'group_name', 'image_path', 'rel_path', 'flagged', 'note', 'pred_top', 'pred_left', 'pred_bottom', 'pred_right', 'global_top', 'global_left', 'global_bottom', 'global_right']];
      for (const row of ROWS) {{
        const item = getItemState(state, row.sample_id);
        if (!item.flagged) continue;
        rows.push([
          row.sample_id,
          VENDOR,
          GROUP_NAME,
          row.image_path,
          row.rel_path,
          '1',
          item.note || '',
          String(row.pred.top),
          String(row.pred.left),
          String(row.pred.bottom),
          String(row.pred.right),
          String(row.global.top),
          String(row.global.left),
          String(row.global.bottom),
          String(row.global.right),
        ]);
      }}
      downloadCsv(`flags_${{VENDOR}}_${{GROUP_NAME}}.csv`, rows);
    }}

    function exportCorrections(state) {{
      const rows = [[
        'sample_id',
        'vendor',
        'group_name',
        'image_path',
        'rel_path',
        'pred_top',
        'pred_left',
        'pred_bottom',
        'pred_right',
        'global_top',
        'global_left',
        'global_bottom',
        'global_right',
        'corr_top',
        'corr_left',
        'corr_bottom',
        'corr_right',
        'note'
      ]];
      for (const row of ROWS) {{
        const item = getItemState(state, row.sample_id);
        if (!item.flagged) continue;
        const corr = item.correction ? clampBox(row, item.correction) : defaultCorrection(row);
        rows.push([
          row.sample_id,
          VENDOR,
          GROUP_NAME,
          row.image_path,
          row.rel_path,
          String(row.pred.top),
          String(row.pred.left),
          String(row.pred.bottom),
          String(row.pred.right),
          String(row.global.top),
          String(row.global.left),
          String(row.global.bottom),
          String(row.global.right),
          String(corr.top),
          String(corr.left),
          String(corr.bottom),
          String(corr.right),
          item.correction_note || '',
        ]);
      }}
      downloadCsv(`corrections_${{VENDOR}}_${{GROUP_NAME}}.csv`, rows);
    }}

    function exportExclusions(state) {{
      const rows = [[
        'type',
        'vendor',
        'group_name',
        'sample_id',
        'image_path',
        'rel_path',
        'exclude_training',
        'note'
      ]];
      if (state.folder && state.folder.exclude_training) {{
        rows.push([
          'folder',
          VENDOR,
          GROUP_NAME,
          '',
          '',
          '',
          '1',
          state.folder.note || '',
        ]);
      }}
      for (const row of ROWS) {{
        const item = getItemState(state, row.sample_id);
        if (!item.exclude_training) continue;
        rows.push([
          'image',
          VENDOR,
          GROUP_NAME,
          row.sample_id,
          row.image_path,
          row.rel_path,
          '1',
          item.exclude_note || '',
        ]);
      }}
      downloadCsv(`exclusions_${{VENDOR}}_${{GROUP_NAME}}.csv`, rows);
    }}

    function bindFolderControls(state) {{
      const folder = state.folder || (state.folder = {{ exclude_training: false, note: '' }});
      const toggle = document.getElementById('folder-exclude-toggle');
      const note = document.getElementById('folder-exclude-note');
      toggle.checked = !!folder.exclude_training;
      note.value = folder.note || '';

      toggle.onchange = (evt) => {{
        folder.exclude_training = !!evt.target.checked;
        saveState(state);
        updateStats(state);
        renderGallery(state);
      }};

      note.oninput = (evt) => {{
        folder.note = evt.target.value || '';
        saveState(state);
      }};
    }}

    function boot() {{
      document.getElementById('title').textContent = `${{VENDOR}} -> ${{GROUP_NAME}}`;
      document.getElementById('subtitle').textContent = 'Workflow: flag immagini non corrette, correggi box se serve, e marca esclusioni training per immagine o cartella.';
      const state = loadState();
      bindFolderControls(state);
      renderGallery(state);
      renderFlagged(state);
      updateStats(state);

      document.getElementById('btn-export-flags').addEventListener('click', () => exportFlags(state));
      document.getElementById('btn-export-corrections').addEventListener('click', () => exportCorrections(state));
      document.getElementById('btn-export-exclusions').addEventListener('click', () => exportExclusions(state));
      document.getElementById('btn-refresh').addEventListener('click', () => renderFlagged(state));
      document.getElementById('btn-clear').addEventListener('click', () => {{
        if (!confirm('Confermi reset completo flags/correzioni/esclusioni di questa cartella?')) return;
        localStorage.removeItem(STORAGE_KEY);
        const fresh = {{ items: {{}}, folder: {{ exclude_training: false, note: '' }} }};
        bindFolderControls(fresh);
        saveState(fresh);
        renderGallery(fresh);
        renderFlagged(fresh);
        updateStats(fresh);
      }});
    }}

    boot();
  </script>
</body>
</html>
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    m = n // 2
    if n % 2 == 1:
        return float(ordered[m])
    return float((ordered[m - 1] + ordered[m]) / 2.0)


def _box_style_percent(
    *,
    top: int,
    left: int,
    bottom: int,
    right: int,
    width: int,
    height: int,
) -> str:
    w = max(1, int(width))
    h = max(1, int(height))
    t = max(0, min(h - 1, int(top)))
    l = max(0, min(w - 1, int(left)))
    b = max(1, min(h, int(bottom)))
    r = max(1, min(w, int(right)))
    if b <= t:
        b = min(h, t + 1)
    if r <= l:
        r = min(w, l + 1)
    bw = max(1, r - l)
    bh = max(1, b - t)
    return (
        f"left:{100.0 * l / w:.6f}%;"
        f"top:{100.0 * t / h:.6f}%;"
        f"width:{100.0 * bw / w:.6f}%;"
        f"height:{100.0 * bh / h:.6f}%;"
    )


def _preview_grid_html(package_name: str, total_images: int, folder_pages: Sequence[FolderPageInfo]) -> str:
    by_vendor: Dict[str, List[FolderPageInfo]] = defaultdict(list)
    for info in folder_pages:
        by_vendor[info.vendor].append(info)

    vendor_blocks: List[str] = []
    for vendor in sorted(by_vendor):
        infos = sorted(
            by_vendor[vendor],
            key=lambda i: (float(i.vendor_confidence), i.group_name.lower()),
        )
        cards: List[str] = []
        for info in infos:
            pred_style = _box_style_percent(
                top=info.preview_pred_top,
                left=info.preview_pred_left,
                bottom=info.preview_pred_bottom,
                right=info.preview_pred_right,
                width=info.preview_width,
                height=info.preview_height,
            )
            glob_style = _box_style_percent(
                top=info.preview_global_top,
                left=info.preview_global_left,
                bottom=info.preview_global_bottom,
                right=info.preview_global_right,
                width=info.preview_width,
                height=info.preview_height,
            )
            preview_rel_src = _rel_src_from_output(info.preview_rel_path, depth=2)
            cards.append(
                f"""<article class="card">
  <a class="thumb img-wrap" href="folders/{_escape(info.filename)}">
    <img loading="lazy" src="{_escape(preview_rel_src)}" data-fallback-src="{_escape(info.preview_image_uri)}" alt="{_escape(info.group_name)}">
    <div class="box pred" style="{_escape(pred_style)}"></div>
    <div class="box glob" style="{_escape(glob_style)}"></div>
  </a>
  <div class="meta">
    <div class="title"><a href="folders/{_escape(info.filename)}">{_escape(info.group_name)}</a></div>
    <div class="line">confidence: {info.vendor_confidence:.4f} | margin: {info.vendor_margin:.4f} | vote: {info.vendor_vote_ratio:.4f}</div>
    <div class="line">immagini: {info.num_images} | preview: {_escape(info.preview_rel_path)}</div>
    <div class="line">pred: T{info.preview_pred_top} L{info.preview_pred_left} B{info.preview_pred_bottom} R{info.preview_pred_right}</div>
    <div class="line">global: T{info.preview_global_top} L{info.preview_global_left} B{info.preview_global_bottom} R{info.preview_global_right}</div>
  </div>
</article>"""
            )

        vendor_blocks.append(
            f"""<details open>
  <summary><b>{_escape(vendor)}</b> - {len(infos)} cartelle (ordinate per confidence crescente)</summary>
  <div class="grid">
    {"".join(cards)}
  </div>
</details>"""
        )

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_escape(package_name)} - Preview by Vendor Confidence</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0f1722; color: #e7edf5; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0; }}
    .muted {{ color: #9fb0c3; margin: 6px 0 12px; }}
    details {{
      background: #172333;
      border: 1px solid #2b3a4f;
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 12px;
    }}
    summary {{ cursor: pointer; }}
    .top-links {{ margin: 8px 0 14px; }}
    a {{ color: #8ab4ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px; font-size: 12px; color: #c7d4e4; }}
    .chip {{ border: 1px solid #2b3a4f; border-radius: 999px; padding: 3px 8px; background: #121c2a; }}
    .chip i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
    .grid {{ margin-top: 10px; display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 10px; }}
    .card {{ background: #121c2a; border: 1px solid #2b3a4f; border-radius: 8px; overflow: hidden; }}
    .thumb {{ display: block; background: #000; }}
    .img-wrap {{ position: relative; }}
    .thumb img {{ width: 100%; display: block; }}
    .box {{ position: absolute; border: 2px solid; pointer-events: none; }}
    .box.pred {{ border-color: #ef4444; }}
    .box.glob {{ border-color: #22d3ee; }}
    .meta {{ padding: 8px 10px; font-size: 12px; line-height: 1.35; }}
    .title {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
    .line {{ color: #c7d4e4; }}
    @media (max-width: 1200px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(320px, 1fr)); }}
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{_escape(package_name)} - Preview Cartelle</h1>
    <div class="muted">Una preview per cartella, raggruppata per vendor e ordinata per confidence crescente.</div>
    <div class="top-links"><a href="index.html">Vai all'indice cartelle</a></div>
    <div class="legend">
      <span class="chip"><i style="background:#ef4444"></i>rosso = rect predetto</span>
      <span class="chip"><i style="background:#22d3ee"></i>ciano = rect globale cartella</span>
    </div>
    <div class="muted">Totale immagini: {total_images}. Totale cartelle: {len(folder_pages)}.</div>
    {"".join(vendor_blocks)}
  </div>
  <script>
    document.querySelectorAll('img[data-fallback-src]').forEach((img) => {{
      img.addEventListener('error', () => {{
        if (img.dataset.fallbackUsed === '1') return;
        const fallback = img.getAttribute('data-fallback-src') || '';
        if (!fallback) return;
        img.dataset.fallbackUsed = '1';
        img.setAttribute('src', fallback);
      }});
    }});
  </script>
</body>
</html>
"""


def _one_page_one_preview_html(package_name: str, total_images: int, folder_pages: Sequence[FolderPageInfo]) -> str:
    infos = sorted(
        folder_pages,
        key=lambda i: (i.vendor.lower(), float(i.vendor_confidence), i.group_name.lower()),
    )
    cards: List[str] = []
    for info in infos:
        pred_style = _box_style_percent(
            top=info.preview_pred_top,
            left=info.preview_pred_left,
            bottom=info.preview_pred_bottom,
            right=info.preview_pred_right,
            width=info.preview_width,
            height=info.preview_height,
        )
        glob_style = _box_style_percent(
            top=info.preview_global_top,
            left=info.preview_global_left,
            bottom=info.preview_global_bottom,
            right=info.preview_global_right,
            width=info.preview_width,
            height=info.preview_height,
        )
        preview_rel_src = _rel_src_from_output(info.preview_rel_path, depth=2)
        search_blob = f"{info.vendor} {info.group_name}".lower()
        cards.append(
            f"""<article class="card" data-search="{_escape(search_blob)}">
  <a class="thumb img-wrap" href="folders/{_escape(info.filename)}">
    <img loading="lazy" src="{_escape(preview_rel_src)}" data-fallback-src="{_escape(info.preview_image_uri)}" alt="{_escape(info.group_name)}">
    <div class="box pred" style="{_escape(pred_style)}"></div>
    <div class="box glob" style="{_escape(glob_style)}"></div>
  </a>
  <div class="meta">
    <div class="title"><a href="folders/{_escape(info.filename)}">{_escape(info.group_name)}</a></div>
    <div class="line">vendor: {_escape(info.vendor)} | confidence: {info.vendor_confidence:.4f}</div>
    <div class="line">immagini: {info.num_images} | preview: {_escape(info.preview_rel_path)}</div>
  </div>
</article>"""
        )

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_escape(package_name)} - 1 immagine per cartella</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0f1722; color: #e7edf5; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0; }}
    .muted {{ color: #9fb0c3; margin: 6px 0 12px; }}
    .top-links {{ margin: 8px 0 14px; }}
    a {{ color: #8ab4ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px; font-size: 12px; color: #c7d4e4; }}
    .chip {{ border: 1px solid #2b3a4f; border-radius: 999px; padding: 3px 8px; background: #121c2a; }}
    .chip i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
    .toolbar {{ margin: 10px 0 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .toolbar input {{
      width: min(420px, 100%);
      background: #121c2a;
      color: #e7edf5;
      border: 1px solid #2b3a4f;
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .grid {{ margin-top: 10px; display: grid; grid-template-columns: repeat(4, minmax(280px, 1fr)); gap: 10px; }}
    .card {{ background: #121c2a; border: 1px solid #2b3a4f; border-radius: 8px; overflow: hidden; }}
    .thumb {{ display: block; background: #000; }}
    .img-wrap {{ position: relative; }}
    .thumb img {{ width: 100%; display: block; }}
    .box {{ position: absolute; border: 2px solid; pointer-events: none; }}
    .box.pred {{ border-color: #ef4444; }}
    .box.glob {{ border-color: #22d3ee; }}
    .meta {{ padding: 8px 10px; font-size: 12px; line-height: 1.35; }}
    .title {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
    .line {{ color: #c7d4e4; }}
    @media (max-width: 1300px) {{
      .grid {{ grid-template-columns: repeat(3, minmax(280px, 1fr)); }}
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(260px, 1fr)); }}
    }}
    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{_escape(package_name)} - Una immagine per cartella</h1>
    <div class="muted">Totale immagini: {total_images}. Totale cartelle: {len(folder_pages)}.</div>
    <div class="top-links"><a href="index.html">Vai all'indice cartelle</a></div>
    <div class="legend">
      <span class="chip"><i style="background:#ef4444"></i>rosso = rect predetto</span>
      <span class="chip"><i style="background:#22d3ee"></i>ciano = rect globale cartella</span>
    </div>
    <div class="toolbar">
      <input id="search" type="text" placeholder="Filtra per vendor o nome cartella...">
    </div>
    <div class="grid" id="grid">
      {"".join(cards)}
    </div>
  </div>
  <script>
    document.querySelectorAll('img[data-fallback-src]').forEach((img) => {{
      img.addEventListener('error', () => {{
        if (img.dataset.fallbackUsed === '1') return;
        const fallback = img.getAttribute('data-fallback-src') || '';
        if (!fallback) return;
        img.dataset.fallbackUsed = '1';
        img.setAttribute('src', fallback);
      }});
    }});

    const searchInput = document.getElementById('search');
    const cards = Array.from(document.querySelectorAll('.card'));
    searchInput.addEventListener('input', () => {{
      const q = String(searchInput.value || '').trim().toLowerCase();
      for (const card of cards) {{
        const blob = String(card.getAttribute('data-search') || '');
        card.style.display = !q || blob.includes(q) ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""


def _index_html(
    package_name: str,
    total_images: int,
    folder_pages: Sequence[FolderPageInfo],
    preview_page_name: str,
    one_page_preview_name: str,
) -> str:
    by_vendor: Dict[str, List[FolderPageInfo]] = defaultdict(list)
    for info in folder_pages:
        by_vendor[info.vendor].append(info)

    vendor_blocks: List[str] = []
    for vendor in sorted(by_vendor):
        infos = sorted(by_vendor[vendor], key=lambda i: i.group_name.lower())
        count_images = sum(i.num_images for i in infos)
        links = []
        for info in infos:
            links.append(
                f'<li><a href="folders/{info.filename}">{info.group_name}</a> '
                f'<span class="count">({info.num_images} immagini)</span></li>'
            )
        vendor_blocks.append(
            f"""<details open>
  <summary><b>{vendor}</b> - {len(infos)} cartelle, {count_images} immagini</summary>
  <ul>
    {"".join(links)}
  </ul>
</details>"""
        )

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{package_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0f1722; color: #e7edf5; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0; }}
    .muted {{ color: #9fb0c3; margin: 6px 0 12px; }}
    details {{
      background: #172333;
      border: 1px solid #2b3a4f;
      border-radius: 8px;
      padding: 8px 10px;
      margin-bottom: 10px;
    }}
    summary {{ cursor: pointer; }}
    ul {{ margin: 8px 0 0 20px; }}
    li {{ margin: 4px 0; }}
    a {{ color: #8ab4ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .count {{ color: #9fb0c3; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{package_name}</h1>
    <div class="muted">Totale immagini: {total_images}. Apri una cartella per flaggare box e impostare esclusioni training (immagine o cartella intera).</div>
    <div class="muted"><a href="{_escape(preview_page_name)}">Apri pagina preview cartelle (vendor + confidence crescente)</a></div>
    <div class="muted"><a href="{_escape(one_page_preview_name)}">Apri pagina unica: 1 immagine per cartella</a></div>
    {"".join(vendor_blocks)}
  </div>
</body>
</html>
"""


def _write_folder_pages(output_dir: Path, rows: Sequence[PredictionRow]) -> List[FolderPageInfo]:
    folders_dir = output_dir / "folders"
    folders_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str], List[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.vendor, row.group_name)].append(row)

    infos: List[FolderPageInfo] = []
    for idx, (key, grows) in enumerate(sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
        vendor, group_name = key
        slug = _slugify(group_name)[:80]
        filename = f"folder_{idx:04d}_{slug}.html"
        payload = _build_folder_payload(grows, vendor=vendor, group_name=group_name)
        page_html = _folder_page_html(payload)
        (folders_dir / filename).write_text(page_html, encoding="utf-8")
        preview_sorted = sorted(grows, key=lambda r: (r.rel_path.lower(), r.sample_id))
        preview = preview_sorted[len(preview_sorted) // 2]
        conf_values = [float(r.vendor_confidence) for r in grows]
        margin_values = [float(r.vendor_margin) for r in grows]
        vote_values = [float(r.vendor_vote_ratio) for r in grows]
        infos.append(
            FolderPageInfo(
                vendor=vendor,
                group_name=group_name,
                filename=filename,
                num_images=len(grows),
                preview_image_uri=preview.image_uri,
                preview_rel_path=preview.rel_path,
                preview_width=preview.width,
                preview_height=preview.height,
                preview_pred_top=preview.pred_top,
                preview_pred_left=preview.pred_left,
                preview_pred_bottom=preview.pred_bottom,
                preview_pred_right=preview.pred_right,
                preview_global_top=preview.global_top,
                preview_global_left=preview.global_left,
                preview_global_bottom=preview.global_bottom,
                preview_global_right=preview.global_right,
                vendor_confidence=_median(conf_values),
                vendor_margin=_median(margin_values),
                vendor_vote_ratio=_median(vote_values),
            )
        )
    return infos


def _write_summary_json(output_dir: Path, rows: Sequence[PredictionRow], folder_pages: Sequence[FolderPageInfo]) -> None:
    vendor_counter = Counter(info.vendor for info in folder_pages)
    payload = {
        "num_images": len(rows),
        "num_folders": len(folder_pages),
        "num_vendors": len(vendor_counter),
        "folders_per_vendor": dict(sorted(vendor_counter.items())),
        "preview_page": (output_dir / "preview_by_vendor_confidence.html").as_posix(),
        "one_page_one_image_per_folder": (output_dir / "one_page_one_image_per_folder.html").as_posix(),
        "output_dir": output_dir.as_posix(),
    }
    (output_dir / "package_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera pacchetto HTML completo: indice vendor/cartelle + pagina per cartella "
            "con flag immagini, correzione box su flaggate ed esclusioni training."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/ultrasound_rect_inference_dataset_orientation/predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/ultrasound_rect_review_package_full"),
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default="Ultrasound Rect Full Review",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    rows = _load_predictions(predictions_csv)
    if not rows:
        raise RuntimeError("No rows found in predictions CSV.")

    folder_pages = _write_folder_pages(output_dir=output_dir, rows=rows)
    preview_filename = "preview_by_vendor_confidence.html"
    one_page_filename = "one_page_one_image_per_folder.html"
    preview_html = _preview_grid_html(
        package_name=args.package_name,
        total_images=len(rows),
        folder_pages=folder_pages,
    )
    (output_dir / preview_filename).write_text(preview_html, encoding="utf-8")
    one_page_html = _one_page_one_preview_html(
        package_name=args.package_name,
        total_images=len(rows),
        folder_pages=folder_pages,
    )
    (output_dir / one_page_filename).write_text(one_page_html, encoding="utf-8")
    index_html = _index_html(
        package_name=args.package_name,
        total_images=len(rows),
        folder_pages=folder_pages,
        preview_page_name=preview_filename,
        one_page_preview_name=one_page_filename,
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    _write_summary_json(output_dir=output_dir, rows=rows, folder_pages=folder_pages)

    print(f"Review package created: {output_dir}", flush=True)
    print(f"Index: {output_dir / 'index.html'}", flush=True)
    print(f"Preview by vendor confidence: {output_dir / preview_filename}", flush=True)
    print(f"One page (1 immagine/cartella): {output_dir / one_page_filename}", flush=True)
    print(f"Folder pages: {len(folder_pages)}", flush=True)
    print(f"Images: {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
