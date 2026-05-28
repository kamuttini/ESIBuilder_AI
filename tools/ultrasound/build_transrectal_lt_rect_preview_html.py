#!/usr/bin/env python3
"""Build interactive HTML preview for transrectal L/T dataset with rect GT/pred evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _safe_text(value: str, max_len: int = 260) -> str:
    out = (value or "").strip()
    if len(out) <= max_len:
        return out
    return out[: max_len - 1] + "..."


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_folder_images(folder_path: str, max_images: int) -> List[str]:
    base = Path(folder_path)
    if not base.is_dir():
        return []
    out: List[str] = []
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        out.append(path.as_posix())
        if max_images > 0 and len(out) >= max_images:
            break
    return out


def _load_rows(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = (row.get("label_lt") or "").strip().upper()
            if label not in {"L", "T"}:
                continue
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            gt_left_norm = _safe_float(row.get("gt_left_norm"), 0.0)
            gt_top_norm = _safe_float(row.get("gt_top_norm"), 0.0)
            gt_right_norm = _safe_float(row.get("gt_right_norm"), 0.0)
            gt_bottom_norm = _safe_float(row.get("gt_bottom_norm"), 0.0)
            pred_left_norm = _safe_float(row.get("pred_left_norm"), 0.0)
            pred_top_norm = _safe_float(row.get("pred_top_norm"), 0.0)
            pred_right_norm = _safe_float(row.get("pred_right_norm"), 0.0)
            pred_bottom_norm = _safe_float(row.get("pred_bottom_norm"), 0.0)
            iou = _safe_float(row.get("iou_pred_vs_orientation_rect"), -1.0)
            if iou < 0.0:
                iou = _safe_float(row.get("iou_pred_vs_gt"), -1.0)

            rows.append(
                {
                    "image_path": image_path,
                    "image_folder": (row.get("image_folder") or Path(image_path).parent.as_posix()).strip(),
                    "source_name": _safe_text(row.get("source_name") or Path(image_path).name, max_len=220),
                    "split": (row.get("split") or "").strip().lower(),
                    "label_lt": label,
                    "class_target": str(row.get("class_target") or "").strip(),
                    "manufacturer": _safe_text(row.get("manufacturer") or "UNKNOWN", max_len=120),
                    "model_name": _safe_text(row.get("model_name") or "", max_len=180),
                    "fss_probe_type": str(row.get("fss_probe_type") or "").strip(),
                    "fss_id_probe": str(row.get("fss_id_probe") or "").strip(),
                    "fss_id_echo": str(row.get("fss_id_echo") or "").strip(),
                    "dataset_folder": _safe_text(row.get("dataset_folder") or "", max_len=220),
                    "fss_path": _safe_text(row.get("fss_path") or "", max_len=260),
                    "label_evidence": _safe_text(row.get("label_evidence") or "", max_len=200),
                    "iou_pred_vs_gt": f"{iou:.6f}" if iou >= 0.0 else "",
                    "rect_gt_source": _safe_text(row.get("rect_gt_source") or "", max_len=120),
                    "rect_gt_is_ultrasound_bbox": str(row.get("rect_gt_is_ultrasound_bbox") or ""),
                    "gt_left_norm": f"{gt_left_norm:.6f}",
                    "gt_top_norm": f"{gt_top_norm:.6f}",
                    "gt_right_norm": f"{gt_right_norm:.6f}",
                    "gt_bottom_norm": f"{gt_bottom_norm:.6f}",
                    "pred_left_norm": f"{pred_left_norm:.6f}",
                    "pred_top_norm": f"{pred_top_norm:.6f}",
                    "pred_right_norm": f"{pred_right_norm:.6f}",
                    "pred_bottom_norm": f"{pred_bottom_norm:.6f}",
                    "image_width": str(row.get("image_width") or ""),
                    "image_height": str(row.get("image_height") or ""),
                }
            )
    return rows


def _stratified_sample(rows: List[Dict[str, str]], max_rows: int, seed: int) -> List[Dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('split', '')}|{row.get('label_lt', '')}"
        buckets[key].append(row)
    for key in list(buckets.keys()):
        rng.shuffle(buckets[key])

    keys = sorted(buckets.keys())
    selected: List[Dict[str, str]] = []
    base = max_rows // max(1, len(keys))
    for key in keys:
        selected.extend(buckets[key][:base])
        buckets[key] = buckets[key][base:]

    rest = max_rows - len(selected)
    tail: List[Dict[str, str]] = []
    for key in keys:
        tail.extend(buckets[key])
    rng.shuffle(tail)
    selected.extend(tail[:rest])
    rng.shuffle(selected)
    return selected


def _build_html(
    rows: List[Dict[str, str]],
    folder_images: Dict[str, List[str]],
    notes: List[str],
) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    folder_payload = json.dumps(folder_images, ensure_ascii=False)
    notes_text = " | ".join(notes) if notes else "all rows"

    style = """
:root {
  --bg: #f3f6fb;
  --panel: #ffffff;
  --line: #d2dae8;
  --text: #152238;
  --muted: #5a6782;
  --gt: #16a34a;
  --pred: #ef4444;
  --accent: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  background: radial-gradient(circle at 8% 0%, #e6ecfa, #f3f6fb 38%);
  color: var(--text);
}
.wrap { max-width: 1900px; margin: 0 auto; padding: 14px; display: grid; gap: 10px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
}
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 auto; }
.small { font-size: 12px; color: var(--muted); }
.select, .input, .textarea {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--text);
  padding: 6px 8px;
  font-size: 13px;
}
.textarea { width: 100%; min-height: 58px; resize: vertical; }
.button {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--text);
  padding: 6px 9px;
  font-size: 12px;
  cursor: pointer;
}
.button:hover { background: #eef3ff; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 10px;
}
.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
}
.card-corrected {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent) inset;
}
.img-wrap {
  width: 100%;
  height: 200px;
  border-radius: 8px;
  overflow: hidden;
  background: #000;
  position: relative;
}
.thumb {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.box {
  position: absolute;
  border: 2px solid #fff;
  pointer-events: none;
}
.box-pred { border-color: var(--pred); }
.legend {
  margin-top: 6px;
  display: flex;
  gap: 10px;
  font-size: 11px;
}
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
.meta { margin-top: 6px; font-size: 12px; line-height: 1.35; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-all;
}
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
}
.pill-l { background: #ddf6e8; color: #0f7a42; }
.pill-t { background: #ffe8ce; color: #965300; }
.pill-x { background: #ffe3e3; color: #9b1c1c; }
.modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.90);
  display: none;
  z-index: 50;
}
.modal-inner {
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 8px;
  padding: 12px;
}
.modal-top, .modal-bottom {
  color: #e6edf7;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.modal-main {
  display: grid;
  grid-template-columns: 120px 1fr 120px;
  gap: 10px;
  align-items: center;
}
.modal-btn {
  width: 100%;
  height: 100%;
  border: 1px solid #2f3c55;
  border-radius: 10px;
  background: #0f1624;
  color: #e6edf7;
  cursor: pointer;
}
.modal-img-wrap {
  width: 100%;
  height: 100%;
  min-height: 280px;
  max-height: calc(100vh - 210px);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.modal-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
"""

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transrectal L/T GT Preview</title>
  <style>{style}</style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="row">
        <div class="grow">
          <div><b>Transrectal L/T dataset review</b></div>
          <div class="small">sampling: {html.escape(notes_text)}</div>
        </div>
        <div class="small">rows loaded: <b id="rowsLoaded">{len(rows)}</b></div>
        <div class="small">rows visible: <b id="rowsVisible">0</b></div>
        <div class="small">rows corrette: <b id="rowsCorrected">0</b></div>
      </div>
      <div class="row" style="margin-top:8px;">
        <label class="small">Label GT</label>
        <select id="labelFilter" class="select">
          <option value="ALL">All</option>
          <option value="L">L</option>
          <option value="T">T</option>
        </select>
        <label class="small">Split</label>
        <select id="splitFilter" class="select">
          <option value="ALL">All</option>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
        <label class="small">Vendor</label>
        <select id="vendorFilter" class="select">
          <option value="ALL">All</option>
        </select>
        <label class="small">Stato immagine</label>
        <select id="excludedFilter" class="select">
          <option value="ALL">Tutte</option>
          <option value="ONLY_INCLUDED">Solo incluse</option>
          <option value="ONLY_EXCLUDED">Solo escluse</option>
        </select>
        <label class="small">Probe ID</label>
        <input id="probeIdFilter" class="input" placeholder="es. 37" style="width:90px;" />
        <label class="small">IoU >=</label>
        <input id="iouMinFilter" class="input" value="0.00" style="width:80px;" />
        <label class="small">Cerca</label>
        <input id="searchFilter" class="input" placeholder="path/folder/model/fss..." style="min-width:220px;" />
        <button id="exportCorrectionsBtn" class="button" type="button">Export correzioni CSV</button>
        <button id="resetCorrectionsBtn" class="button" type="button">Reset correzioni</button>
      </div>
      <div class="small" style="margin-top:8px;">
        Evidenze: `label_evidence` deriva da `fss_probe_type`; rettangolo rosso = predizione rete rect.
      </div>
    </div>
    <div id="grid" class="grid"></div>
  </div>

  <div id="viewerModal" class="modal">
    <div class="modal-inner">
      <div class="modal-top">
        <button id="modalCloseBtn" class="button" type="button">Chiudi (Esc)</button>
        <div id="modalTitle" class="small mono"></div>
      </div>
      <div class="modal-main">
        <button id="modalPrevBtn" class="modal-btn" type="button">Prev</button>
        <div class="modal-img-wrap">
          <img id="modalImg" class="modal-img" src="" alt="fullscreen preview" />
        </div>
        <button id="modalNextBtn" class="modal-btn" type="button">Next</button>
      </div>
      <div class="modal-bottom">
        <div id="modalCounter" class="small"></div>
        <div id="modalPath" class="small mono"></div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload};
    const FOLDER_IMAGES = {folder_payload};
    const CORR_KEY = "transrectal_lt_rect_gt_corrections_v1";

    const grid = document.getElementById("grid");
    const rowsVisible = document.getElementById("rowsVisible");
    const rowsCorrected = document.getElementById("rowsCorrected");

    const labelFilter = document.getElementById("labelFilter");
    const splitFilter = document.getElementById("splitFilter");
    const vendorFilter = document.getElementById("vendorFilter");
    const excludedFilter = document.getElementById("excludedFilter");
    const probeIdFilter = document.getElementById("probeIdFilter");
    const iouMinFilter = document.getElementById("iouMinFilter");
    const searchFilter = document.getElementById("searchFilter");
    const exportCorrectionsBtn = document.getElementById("exportCorrectionsBtn");
    const resetCorrectionsBtn = document.getElementById("resetCorrectionsBtn");

    const viewerModal = document.getElementById("viewerModal");
    const modalCloseBtn = document.getElementById("modalCloseBtn");
    const modalPrevBtn = document.getElementById("modalPrevBtn");
    const modalNextBtn = document.getElementById("modalNextBtn");
    const modalImg = document.getElementById("modalImg");
    const modalTitle = document.getElementById("modalTitle");
    const modalCounter = document.getElementById("modalCounter");
    const modalPath = document.getElementById("modalPath");

    let corrections = {{}};
    try {{
      corrections = JSON.parse(localStorage.getItem(CORR_KEY) || "{{}}");
    }} catch (_) {{
      corrections = {{}};
    }}

    let modalImages = [];
    let modalIndex = -1;

    function esc(text) {{
      return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function rowKey(row) {{
      return `${{row.image_path}}|${{row.label_lt}}`;
    }}

    function normalizeCorrection(raw) {{
      if (!raw) {{
        return {{ corrected_label: "AUTO", note: "", exclude_image: 0 }};
      }}
      if (typeof raw === "string") {{
        return {{
          corrected_label: raw || "AUTO",
          note: "",
          exclude_image: 0,
        }};
      }}
      const corrected = String(raw.corrected_label || "AUTO");
      const note = String(raw.note || "");
      const exclude = Number(raw.exclude_image || 0) === 1 ? 1 : 0;
      return {{
        corrected_label: corrected || "AUTO",
        note,
        exclude_image: exclude,
      }};
    }}

    function hasMeaningfulCorrection(corr) {{
      if (!corr) return false;
      if ((corr.corrected_label || "AUTO") !== "AUTO") return true;
      if (String(corr.note || "").trim().length > 0) return true;
      if (Number(corr.exclude_image || 0) === 1) return true;
      return false;
    }}

    function getCorrection(row) {{
      const key = rowKey(row);
      return normalizeCorrection(corrections[key]);
    }}

    function effectiveLabel(row) {{
      const corr = getCorrection(row);
      if (corr.corrected_label && corr.corrected_label !== "AUTO") {{
        return corr.corrected_label;
      }}
      return row.label_lt || "";
    }}

    function correctionCount() {{
      rowsCorrected.textContent = String(Object.keys(corrections).length);
    }}

    function toPct(value) {{
      const n = Number(value || 0);
      return `${{Math.max(0, Math.min(1, n)) * 100}}%`;
    }}

    function rowMatches(row) {{
      if (labelFilter.value !== "ALL" && row.label_lt !== labelFilter.value) return false;
      if (splitFilter.value !== "ALL" && row.split !== splitFilter.value) return false;
      if (vendorFilter.value !== "ALL" && row.manufacturer !== vendorFilter.value) return false;
      const excludedMode = excludedFilter.value || "ALL";
      const excluded = Number(getCorrection(row).exclude_image || 0) === 1;
      if (excludedMode === "ONLY_INCLUDED" && excluded) return false;
      if (excludedMode === "ONLY_EXCLUDED" && !excluded) return false;

      const probeNeedle = String(probeIdFilter.value || "").trim();
      if (probeNeedle && String(row.fss_id_probe || "") !== probeNeedle) return false;

      const iouMin = Number(iouMinFilter.value || "0");
      const rowIou = Number(row.iou_pred_vs_gt || "0");
      if (!Number.isNaN(iouMin) && rowIou < iouMin) return false;

      const q = String(searchFilter.value || "").trim().toLowerCase();
      if (!q) return true;
      const hay = `${{row.image_path}} ${{row.image_folder}} ${{row.model_name}} ${{row.dataset_folder}} ${{row.fss_path}} ${{row.manufacturer}} ${{row.fss_id_probe}}`.toLowerCase();
      return hay.includes(q);
    }}

    function openFolderViewer(folderPath, currentPath) {{
      modalImages = FOLDER_IMAGES[folderPath] || [];
      if (!modalImages.length) {{
        modalImages = [currentPath];
      }}
      modalIndex = Math.max(0, modalImages.indexOf(currentPath));
      if (modalIndex < 0) modalIndex = 0;
      viewerModal.style.display = "block";
      renderModal();
    }}

    function renderModal() {{
      if (!modalImages.length || modalIndex < 0) return;
      const path = modalImages[modalIndex];
      modalImg.src = path.startsWith("file://") ? path : `file://${{path}}`;
      modalCounter.textContent = `${{modalIndex + 1}} / ${{modalImages.length}}`;
      modalPath.textContent = path;
      modalTitle.textContent = "Preview cartella completa";
    }}

    function closeModal() {{
      viewerModal.style.display = "none";
      modalImages = [];
      modalIndex = -1;
      modalImg.src = "";
    }}

    function modalPrev() {{
      if (!modalImages.length) return;
      modalIndex = (modalIndex - 1 + modalImages.length) % modalImages.length;
      renderModal();
    }}

    function modalNext() {{
      if (!modalImages.length) return;
      modalIndex = (modalIndex + 1) % modalImages.length;
      renderModal();
    }}

    function refreshVendorOptions() {{
      const vendors = Array.from(new Set(DATA.map(r => String(r.manufacturer || "UNKNOWN")))).sort();
      const current = vendorFilter.value || "ALL";
      vendorFilter.innerHTML = "";
      const first = document.createElement("option");
      first.value = "ALL";
      first.textContent = "All";
      vendorFilter.appendChild(first);
      for (const v of vendors) {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        vendorFilter.appendChild(opt);
      }}
      if ([...vendorFilter.options].some(o => o.value === current)) {{
        vendorFilter.value = current;
      }} else {{
        vendorFilter.value = "ALL";
      }}
    }}

    function render() {{
      const rows = DATA.filter(rowMatches);
      rowsVisible.textContent = String(rows.length);

      const cardsHtml = rows.map(row => {{
        const key = rowKey(row);
        const corr = getCorrection(row);
        const correctedLabel = corr.corrected_label || "AUTO";
        const note = corr.note || "";
        const excluded = Number(corr.exclude_image || 0) === 1;
        const isCorrected = hasMeaningfulCorrection(corr);
        const gtLeft = Number(row.gt_left_norm || "0");
        const gtTop = Number(row.gt_top_norm || "0");
        const gtRight = Number(row.gt_right_norm || "0");
        const gtBottom = Number(row.gt_bottom_norm || "0");
        const predLeft = Number(row.pred_left_norm || "0");
        const predTop = Number(row.pred_top_norm || "0");
        const predRight = Number(row.pred_right_norm || "0");
        const predBottom = Number(row.pred_bottom_norm || "0");

        const gtWidth = Math.max(0, gtRight - gtLeft);
        const gtHeight = Math.max(0, gtBottom - gtTop);
        const predWidth = Math.max(0, predRight - predLeft);
        const predHeight = Math.max(0, predBottom - predTop);

        const imgSrc = row.image_path.startsWith("file://") ? row.image_path : `file://${{row.image_path}}`;
        const pillClass = row.label_lt === "L" ? "pill pill-l" : "pill pill-t";

        return `
          <div class="card ${{isCorrected ? "card-corrected" : ""}}">
            <div class="img-wrap">
              <img class="thumb" src="${{esc(imgSrc)}}" loading="lazy" alt="sample" />
              <div class="box box-pred" style="left:${{toPct(predLeft)}}; top:${{toPct(predTop)}}; width:${{toPct(predWidth)}}; height:${{toPct(predHeight)}};"></div>
            </div>
            <div class="legend">
              <span><span class="swatch" style="background:var(--pred);"></span>Pred</span>
            </div>
            <div class="meta"><span class="${{pillClass}}">${{esc(row.label_lt)}}</span> ${{excluded ? '<span class="pill pill-x">EXCLUDED</span>' : ''}} split=${{esc(row.split)}} | iou=${{esc(row.iou_pred_vs_gt || "-")}} | vendor=${{esc(row.manufacturer)}}</div>
            <div class="meta">probe_id=${{esc(row.fss_id_probe)}} | probe_type=${{esc(row.fss_probe_type)}} | class_target=${{esc(row.class_target)}}</div>
            <div class="meta">evidenza gruppo: label=${{esc(row.label_evidence)}} | fss_id_echo=${{esc(row.fss_id_echo)}}</div>
            <div class="meta">rect_gt_source=${{esc(row.rect_gt_source || "-")}} | rect_gt_is_ultrasound_bbox=${{esc(row.rect_gt_is_ultrasound_bbox || "0")}}</div>
            <div class="meta">model=${{esc(row.model_name)}}</div>
            <div class="meta">dataset_folder=${{esc(row.dataset_folder)}}</div>
            <div class="meta">source_name=${{esc(row.source_name)}}</div>
            <div class="meta mono">image_path=${{esc(row.image_path)}}</div>
            <div class="meta mono">fss_path=${{esc(row.fss_path)}}</div>
            <div class="row" style="margin-top:6px;">
              <span class="small">Correzione label</span>
              <select class="select corr-label" data-row-key="${{esc(key)}}" data-auto-label="${{esc(row.label_lt)}}">
                <option value="AUTO">AUTO (${{esc(row.label_lt)}})</option>
                <option value="L">L</option>
                <option value="T">T</option>
                <option value="SKIP">SKIP</option>
              </select>
              <label class="small"><input type="checkbox" class="corr-exclude" data-row-key="${{esc(key)}}" ${{excluded ? "checked" : ""}} /> Escludi immagine</label>
              <button class="button open-folder-btn" type="button" data-folder="${{esc(row.image_folder)}}" data-image="${{esc(row.image_path)}}">Fullscreen cartella</button>
            </div>
            <div style="margin-top:6px;">
              <textarea class="textarea corr-note" data-row-key="${{esc(key)}}" placeholder="Note/correzioni...">${{esc(note)}}</textarea>
            </div>
          </div>
        `;
      }}).join("");

      grid.innerHTML = cardsHtml;

      for (const sel of grid.querySelectorAll(".corr-label")) {{
        const key = sel.getAttribute("data-row-key") || "";
        const corr = normalizeCorrection(corrections[key]);
        sel.value = corr.corrected_label || "AUTO";
        sel.addEventListener("change", () => {{
          if (!key) return;
          const prev = normalizeCorrection(corrections[key]);
          const next = {{
            corrected_label: sel.value,
            note: prev.note || "",
            exclude_image: prev.exclude_image || 0,
          }};
          if (!hasMeaningfulCorrection(next)) {{
            delete corrections[key];
          }} else {{
            corrections[key] = next;
          }}
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          correctionCount();
          render();
        }});
      }}

      for (const ta of grid.querySelectorAll(".corr-note")) {{
        const key = ta.getAttribute("data-row-key") || "";
        ta.addEventListener("change", () => {{
          if (!key) return;
          const prev = normalizeCorrection(corrections[key]);
          const next = {{
            corrected_label: prev.corrected_label || "AUTO",
            note: ta.value || "",
            exclude_image: prev.exclude_image || 0,
          }};
          if (!hasMeaningfulCorrection(next)) {{
            delete corrections[key];
          }} else {{
            corrections[key] = next;
          }}
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          correctionCount();
        }});
      }}

      for (const ch of grid.querySelectorAll(".corr-exclude")) {{
        const key = ch.getAttribute("data-row-key") || "";
        ch.addEventListener("change", () => {{
          if (!key) return;
          const prev = normalizeCorrection(corrections[key]);
          const next = {{
            corrected_label: prev.corrected_label || "AUTO",
            note: prev.note || "",
            exclude_image: ch.checked ? 1 : 0,
          }};
          if (!hasMeaningfulCorrection(next)) {{
            delete corrections[key];
          }} else {{
            corrections[key] = next;
          }}
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          correctionCount();
          render();
        }});
      }}

      for (const btn of grid.querySelectorAll(".open-folder-btn")) {{
        btn.addEventListener("click", () => {{
          const folder = btn.getAttribute("data-folder") || "";
          const image = btn.getAttribute("data-image") || "";
          openFolderViewer(folder, image);
        }});
      }}

      correctionCount();
    }}

    function toCsvCell(value) {{
      const s = String(value ?? "");
      if (s.includes(",") || s.includes('"') || s.includes("\\n")) {{
        return `"${{s.replaceAll('"', '""')}}"`;
      }}
      return s;
    }}

    function exportCorrectionsCsv() {{
      const header = [
        "row_key",
        "image_path",
        "split",
        "label_gt",
        "corrected_label",
        "exclude_image",
        "final_action",
        "note",
        "manufacturer",
        "fss_id_probe",
        "fss_probe_type",
        "fss_id_echo",
        "iou_pred_vs_gt",
        "label_evidence",
        "exported_at",
      ];
      const lines = [header];
      for (const row of DATA) {{
        const key = rowKey(row);
        const corr = normalizeCorrection(corrections[key]);
        if (!hasMeaningfulCorrection(corr)) continue;
        let finalAction = "NOTE_ONLY";
        if (Number(corr.exclude_image || 0) === 1) {{
          finalAction = "EXCLUDE_IMAGE";
        }} else if ((corr.corrected_label || "AUTO") !== "AUTO") {{
          finalAction = "RELABEL";
        }}
        lines.push([
          key,
          row.image_path || "",
          row.split || "",
          row.label_lt || "",
          corr.corrected_label || "AUTO",
          String(Number(corr.exclude_image || 0) === 1 ? 1 : 0),
          finalAction,
          corr.note || "",
          row.manufacturer || "",
          row.fss_id_probe || "",
          row.fss_probe_type || "",
          row.fss_id_echo || "",
          row.iou_pred_vs_gt || "",
          row.label_evidence || "",
          new Date().toISOString(),
        ]);
      }}
      const csvText = lines.map(row => row.map(toCsvCell).join(",")).join("\\n");
      const blob = new Blob([csvText], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "transrectal_lt_rect_corrections.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    modalCloseBtn.addEventListener("click", closeModal);
    modalPrevBtn.addEventListener("click", modalPrev);
    modalNextBtn.addEventListener("click", modalNext);
    viewerModal.addEventListener("click", (evt) => {{
      if (evt.target === viewerModal) closeModal();
    }});
    document.addEventListener("keydown", (evt) => {{
      if (viewerModal.style.display !== "block") return;
      if (evt.key === "Escape") closeModal();
      if (evt.key === "ArrowLeft") modalPrev();
      if (evt.key === "ArrowRight") modalNext();
    }});

    for (const el of [labelFilter, splitFilter, vendorFilter, excludedFilter]) {{
      el.addEventListener("change", render);
    }}
    for (const el of [probeIdFilter, iouMinFilter, searchFilter]) {{
      el.addEventListener("input", render);
    }}
    exportCorrectionsBtn.addEventListener("click", exportCorrectionsCsv);
    resetCorrectionsBtn.addEventListener("click", () => {{
      corrections = {{}};
      localStorage.removeItem(CORR_KEY);
      correctionCount();
      render();
    }});

    refreshVendorOptions();
    correctionCount();
    render();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build HTML review for transrectal L/T manifest with GT/pred rect evidence."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/manifest_transrectal_lt_rect.csv"),
        help="Prepared dataset manifest.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/preview/transrectal_lt_rect_gt_review.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1200,
        help="0=all rows, >0 stratified sample by split/label.",
    )
    parser.add_argument(
        "--max-folder-images",
        type=int,
        default=0,
        help="0=all images in folder for fullscreen browsing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest non trovato: {manifest}")

    rows = _load_rows(manifest)
    if not rows:
        raise RuntimeError("Manifest vuoto o senza label L/T.")

    notes: List[str] = []
    if int(args.max_rows) > 0:
        notes.append(f"max_rows={int(args.max_rows)} stratified by split|label")
    sampled = _stratified_sample(rows, max_rows=int(args.max_rows), seed=int(args.seed))

    folder_images: Dict[str, List[str]] = {}
    for row in sampled:
        folder = row.get("image_folder") or ""
        if not folder:
            continue
        if folder in folder_images:
            continue
        folder_images[folder] = _iter_folder_images(folder, max_images=int(args.max_folder_images))
    if int(args.max_folder_images) > 0:
        notes.append(f"max_folder_images={int(args.max_folder_images)}")

    doc = _build_html(rows=sampled, folder_images=folder_images, notes=notes)
    output_html.write_text(doc, encoding="utf-8")

    print(f"Rows loaded: {len(rows)}", flush=True)
    print(f"Rows in HTML: {len(sampled)}", flush=True)
    print(f"Folders indexed: {len(folder_images)}", flush=True)
    print(f"Output HTML: {output_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
