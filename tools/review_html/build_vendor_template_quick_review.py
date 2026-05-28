#!/usr/bin/env python3
"""Build a fast single-page vendor review with per-image template crop preview."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(str(value).strip())))
    except Exception:  # noqa: BLE001
        return int(default)


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return float(default)


def _slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def _read_manifest(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"Manifest has no header: {path}")
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return fieldnames, rows


def _build_payload(rows: List[Dict[str, str]]) -> Dict[str, object]:
    by_vendor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        vendor = (row.get("manufacturer", "") or "UNKNOWN").strip() or "UNKNOWN"
        image_path = (row.get("image_path", "") or "").strip()
        if not image_path:
            continue
        left = _parse_int(row.get("bbox_left", row.get("bbox_xmin", "0")), 0)
        top = _parse_int(row.get("bbox_top", row.get("bbox_ymin", "0")), 0)
        right = _parse_int(row.get("bbox_right", row.get("bbox_xmax", "1")), 1)
        bottom = _parse_int(row.get("bbox_bottom", row.get("bbox_ymax", "1")), 1)
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1

        w = _parse_int(row.get("image_width", "0"), 0)
        h = _parse_int(row.get("image_height", "0"), 0)
        if w <= 0:
            w = max(1, right + 1)
        if h <= 0:
            h = max(1, bottom + 1)

        by_vendor[vendor].append(
            {
                "image_path": image_path,
                "split": (row.get("split", "") or "").strip(),
                "group_id": (row.get("group_id", "") or "").strip(),
                "dataset_folder": (row.get("dataset_folder", "") or "").strip(),
                "setup_name": (row.get("setup_name", "") or "").strip(),
                "bbox_source": (row.get("bbox_source", "") or "").strip(),
                "manual_folder_action": (row.get("manual_folder_action", "") or "").strip(),
                "manual_rect_updated": (row.get("manual_rect_updated", "") or "").strip(),
                "bbox_top": top,
                "bbox_left": left,
                "bbox_bottom": bottom,
                "bbox_right": right,
                "image_width": w,
                "image_height": h,
                "uniform_suspect": _parse_int(row.get("uniform_suspect", "0"), 0),
                "uniform_black_exact": _parse_int(row.get("uniform_black_exact", "0"), 0),
                "uniform_exact": _parse_int(row.get("uniform_exact", "0"), 0),
                "uniform_reason": (row.get("uniform_reason", "") or "").strip(),
                "gt_std": _parse_float(row.get("gt_std", "nan"), float("nan")),
                "gt_unique": _parse_int(row.get("gt_unique", "0"), 0),
                "gt_range": _parse_float(row.get("gt_range", "nan"), float("nan")),
            }
        )

    vendors = sorted(by_vendor.keys(), key=lambda x: x.lower())
    vendor_files: List[Dict[str, object]] = []
    for idx, vendor in enumerate(vendors, start=1):
        rows_vendor = by_vendor[vendor]
        rows_vendor.sort(
            key=lambda r: (
                str(r.get("dataset_folder", "")).lower(),
                str(r.get("setup_name", "")).lower(),
                str(r.get("image_path", "")).lower(),
            )
        )
        slug = _slugify(vendor)
        vendor_files.append(
            {
                "id": f"{slug}-{idx}",
                "vendor": vendor,
                "count": len(rows_vendor),
                "file": f"{slug}-{idx}.json",
            }
        )
    return {"vendors": vendor_files, "rows_by_vendor": by_vendor}


def _write_vendor_json(out_dir: Path, vendor_file: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    path = out_dir / str(vendor_file["file"])
    payload = {"vendor": vendor_file["vendor"], "count": len(rows), "rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _render_index_html(vendors: List[Dict[str, object]], rows_per_page: int) -> str:
    vendors_json = json.dumps(vendors, ensure_ascii=True)
    rows_per_page_json = json.dumps(int(max(1, rows_per_page)))
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vendor Template Quick Review</title>
  <style>
    :root {{
      --bg: #0f141c;
      --panel: #151e2b;
      --panel2: #0f1723;
      --border: #31445f;
      --txt: #eaf0fb;
      --muted: #9bb0cf;
      --accent: #5fa8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--txt);
      background: radial-gradient(900px 500px at 80% -10%, #1c2c46 0%, var(--bg) 60%);
    }}
    .wrap {{ max-width: 1900px; margin: 0 auto; padding: 12px; }}
    .top {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: linear-gradient(180deg, #1a273b, var(--panel));
      padding: 10px;
      display: grid;
      gap: 8px;
    }}
    .title {{ margin: 0; font-size: 22px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .bar button, .bar select, .bar input {{
      border: 1px solid #49638b;
      border-radius: 6px;
      background: #1c2d49;
      color: #eef4ff;
      padding: 6px 9px;
      font-size: 12px;
    }}
    .bar button {{ cursor: pointer; font-weight: 700; }}
    .bar button.active {{
      background: #2e6dc4;
      border-color: #8bb6ff;
    }}
    .bar button.warn {{
      background: #5a2230;
      border-color: #b36a7d;
    }}
    .bar button.alt {{
      background: #1b385a;
      border-color: #5f89ba;
    }}
    .layout {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: 290px 1fr;
      gap: 10px;
      min-height: calc(100vh - 180px);
    }}
    .vendors {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel2);
      overflow: auto;
      padding: 8px;
    }}
    .vendors .v {{
      width: 100%;
      text-align: left;
      padding: 7px 8px;
      margin: 0 0 6px 0;
      border: 1px solid #3d5375;
      border-radius: 7px;
      background: #1a2a41;
      color: #e6efff;
      cursor: pointer;
      font-size: 12px;
    }}
    .vendors .v.active {{
      background: #4177c4;
      border-color: #8bb6ff;
      color: #fff;
      font-weight: 700;
    }}
    .main {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel2);
      padding: 8px;
      overflow: auto;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(220px, 1fr));
      gap: 10px;
    }}
    .card {{
      border: 1px solid #3a4f6e;
      border-radius: 8px;
      background: #121d2d;
      overflow: hidden;
      cursor: pointer;
    }}
    .card.selected {{
      border-color: #7fb4ff;
      box-shadow: 0 0 0 2px rgba(95,168,255,0.4) inset;
    }}
    .card.excluded {{
      border-color: #d66767;
      box-shadow: 0 0 0 2px rgba(214,103,103,0.35) inset;
    }}
    .imgs {{
      padding: 8px;
    }}
    .crop {{
      position: relative;
      background: #000;
      border: 1px solid #2f405b;
      border-radius: 6px;
      overflow: hidden;
      margin: 0 auto;
    }}
    .crop img {{
      position: absolute;
      image-rendering: auto;
      pointer-events: none;
      user-select: none;
    }}
    .meta {{
      border-top: 1px solid #2f405b;
      padding: 7px 8px;
      font-size: 11px;
      color: #dce8fb;
      line-height: 1.35;
    }}
    .path {{
      color: #9bb0cf;
      word-break: break-all;
    }}
    .status {{
      color: #ffd7a1;
      font-weight: 700;
    }}
    .pill {{
      display: inline-block;
      margin-left: 6px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid #c26c6c;
      background: #5a2230;
      color: #ffd7d7;
      font-size: 10px;
      font-weight: 700;
    }}
    .empty {{
      color: #9bb0cf;
      font-size: 13px;
      padding: 18px;
      text-align: center;
    }}
    .grid.drag-mode {{
      user-select: none;
      -webkit-user-select: none;
    }}
    .lasso {{
      position: fixed;
      z-index: 9999;
      border: 1px dashed #8ec0ff;
      background: rgba(95,168,255,0.18);
      pointer-events: none;
      display: none;
    }}
    @media (max-width: 1500px) {{
      .grid {{ grid-template-columns: repeat(4, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 1300px) {{
      .grid {{ grid-template-columns: repeat(3, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top">
      <h1 class="title">Vendor Template Quick Review</h1>
      <div class="muted">
        Review veloce pre-training: immagini divise per vendor, con solo crop template.
        Shortcut: <b>↑/↓</b> vendor, <b>←/→</b> pagina, <b>X</b> escludi selezionate, <b>R</b> ripristina selezionate.
      </div>
      <div class="bar">
        <button id="btn-prev-vendor" type="button">Vendor precedente</button>
        <button id="btn-next-vendor" type="button">Vendor successivo</button>
        <button id="btn-prev-page" type="button">Pagina precedente</button>
        <button id="btn-next-page" type="button">Pagina successiva</button>
        <button id="btn-toggle-drag" class="active" type="button">Drag select: ON</button>
        <button id="btn-exclude-selected" class="warn" type="button">Escludi selezionate</button>
        <button id="btn-restore-selected" class="alt" type="button">Ripristina selezionate</button>
        <button id="btn-clear-selection" type="button">Azzera selezione</button>
        <button id="btn-export-excluded" type="button">Export escluse CSV</button>
        <label>Rows/page
          <select id="rows-per-page">
            <option>60</option>
            <option selected>100</option>
            <option>150</option>
            <option>200</option>
            <option>300</option>
          </select>
        </label>
        <label>Filtro
          <select id="filter-mode">
            <option value="all" selected>Tutte</option>
            <option value="suspect">Solo sospette</option>
            <option value="excluded">Solo escluse</option>
            <option value="active">Solo attive</option>
          </select>
        </label>
        <span id="status" class="status"></span>
        <span id="selection-info" class="muted"></span>
      </div>
    </section>
    <section class="layout">
      <aside class="vendors" id="vendors"></aside>
      <main class="main">
        <div id="grid" class="grid"></div>
      </main>
    </section>
  </div>
  <div id="lasso" class="lasso"></div>
  <script>
    const VENDORS = __VENDORS__;
    const DEFAULT_ROWS_PER_PAGE = __ROWS_PER_PAGE__;
    const STORAGE_EXCLUDED_KEY = "vendor_template_quick_review_exclusions::" + String(window.location.pathname || "");
    const cache = {{}};
    let vendorIdx = 0;
    let pageIdx = 0;
    let rowsPerPage = DEFAULT_ROWS_PER_PAGE;
    let dragEnabled = true;
    let selectedKeys = new Set();
    let exclusionMap = loadExcludedMap();
    let currentRowMap = new Map();
    let suppressCardClickUntil = 0;
    let lassoState = null;

    function encodePath(path) {{
      if (!path) return "";
      const clean = String(path).replace(/\\\\/g, "/");
      const parts = clean.split("/");
      const out = [];
      for (const p of parts) {{
        if (p === "") {{
          if (out.length === 0) out.push("");
          continue;
        }}
        out.push(encodeURIComponent(p));
      }}
      return out.join("/");
    }}

    function esc(value) {{
      return String(value == null ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function rowKey(row) {{
      return [
        String(row.image_path || ""),
        String(row.group_id || ""),
        String(row.setup_name || ""),
        String(row.split || ""),
      ].join("||");
    }}

    function loadExcludedMap() {{
      try {{
        const raw = window.localStorage.getItem(STORAGE_EXCLUDED_KEY);
        if (!raw) return {{}};
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") return parsed;
      }} catch (_err) {{
        return {{}};
      }}
      return {{}};
    }}

    function saveExcludedMap() {{
      try {{
        window.localStorage.setItem(STORAGE_EXCLUDED_KEY, JSON.stringify(exclusionMap));
      }} catch (_err) {{
        // ignore quota/private mode issues
      }}
    }}

    function currentVendor() {{
      if (!Array.isArray(VENDORS) || VENDORS.length === 0) return null;
      vendorIdx = Math.max(0, Math.min(VENDORS.length - 1, vendorIdx));
      return VENDORS[vendorIdx];
    }}

    async function loadVendorData(vendor) {{
      if (!vendor) return {{ rows: [] }};
      const key = String(vendor.id || "");
      if (cache[key]) return cache[key];
      const response = await fetch("vendors/" + vendor.file, {{ cache: "no-store" }});
      if (!response.ok) throw new Error("HTTP " + response.status);
      const payload = await response.json();
      cache[key] = payload;
      return payload;
    }}

    function cropStrictStyles(row, maxCropW, maxCropH) {{
      const w = Math.max(1, Number(row.image_width || 1));
      const h = Math.max(1, Number(row.image_height || 1));
      const l = Math.max(0, Number(row.bbox_left || 0));
      const t = Math.max(0, Number(row.bbox_top || 0));
      const r = Math.max(l + 1, Number(row.bbox_right || (l + 1)));
      const b = Math.max(t + 1, Number(row.bbox_bottom || (t + 1)));
      const bw = Math.max(1, r - l);
      const bh = Math.max(1, b - t);
      const scale = Math.max(0.01, Math.min(maxCropW / bw, maxCropH / bh));
      const cropW = bw * scale;
      const cropH = bh * scale;
      const fullW = w * scale;
      const fullH = h * scale;
      const x = -l * scale;
      const y = -t * scale;
      return {{
        box: [
          "width:" + cropW.toFixed(3) + "px",
          "height:" + cropH.toFixed(3) + "px"
        ].join(";"),
        img: [
          "left:" + x.toFixed(3) + "px",
          "top:" + y.toFixed(3) + "px",
          "width:" + fullW.toFixed(3) + "px",
          "height:" + fullH.toFixed(3) + "px"
        ].join(";")
      }};
    }}

    function renderVendors() {{
      const root = document.getElementById("vendors");
      if (!root) return;
      root.innerHTML = "";
      VENDORS.forEach((v, idx) => {{
        const b = document.createElement("button");
        b.type = "button";
        b.className = "v" + (idx === vendorIdx ? " active" : "");
        b.textContent = v.vendor + " (" + v.count + ")";
        b.addEventListener("click", () => {{
          vendorIdx = idx;
          pageIdx = 0;
          selectedKeys.clear();
          renderAll();
        }});
        root.appendChild(b);
      }});
    }}

    function setStatus(text) {{
      const el = document.getElementById("status");
      if (el) el.textContent = text || "";
    }}

    function setSelectionInfo() {{
      const el = document.getElementById("selection-info");
      if (!el) return;
      let visibleExcluded = 0;
      for (const key of currentRowMap.keys()) {{
        if (exclusionMap[key]) visibleExcluded += 1;
      }}
      const selectedVisible = Array.from(selectedKeys).filter((k) => currentRowMap.has(k)).length;
      const totalExcluded = Object.keys(exclusionMap).length;
      el.textContent = "selezionate: " + selectedVisible + " | escluse visibili: " + visibleExcluded + " | escluse totali: " + totalExcluded;
    }}

    function updateVisibleCardStates(extraKeys) {{
      const extra = extraKeys instanceof Set ? extraKeys : new Set();
      const cards = document.querySelectorAll("#grid .card[data-row-key]");
      cards.forEach((card) => {{
        const key = String(card.dataset.rowKey || "");
        const selected = selectedKeys.has(key) || extra.has(key);
        const excluded = !!exclusionMap[key];
        card.classList.toggle("selected", selected);
        card.classList.toggle("excluded", excluded);
      }});
    }}

    function toExcludedMeta(row, vendorName) {{
      return {{
        key: rowKey(row),
        image_path: String(row.image_path || ""),
        vendor: String(vendorName || ""),
        group_id: String(row.group_id || ""),
        dataset_folder: String(row.dataset_folder || ""),
        setup_name: String(row.setup_name || ""),
        split: String(row.split || ""),
        reason: "manual_quick_review_exclude",
      }};
    }}

    function setDragButtonState() {{
      const btn = document.getElementById("btn-toggle-drag");
      if (!btn) return;
      btn.classList.toggle("active", !!dragEnabled);
      btn.textContent = dragEnabled ? "Drag select: ON" : "Drag select: OFF";
    }}

    function clearSelection() {{
      selectedKeys.clear();
      updateVisibleCardStates();
      setSelectionInfo();
    }}

    function downloadText(name, text, mimeType) {{
      const blob = new Blob([text], {{ type: mimeType || "text/plain;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}

    function exportExcludedCsv() {{
      const rows = Object.values(exclusionMap || {{}});
      if (!rows.length) {{
        alert("Nessuna immagine esclusa da esportare.");
        return;
      }}
      rows.sort((a, b) => String(a.image_path || "").localeCompare(String(b.image_path || "")));
      const header = ["image_path", "vendor", "group_id", "dataset_folder", "setup_name", "split", "reason"];
      function csvCell(v) {{
        const s = String(v == null ? "" : v);
        if (s.includes('"') || s.includes(",") || s.includes("\\n")) return '"' + s.replace(/"/g, '""') + '"';
        return s;
      }}
      const lines = [header.join(",")];
      for (const row of rows) {{
        lines.push(header.map((h) => csvCell(row[h] || "")).join(","));
      }}
      downloadText("vendor_template_image_exclusions.csv", lines.join("\\n"), "text/csv;charset=utf-8");
    }}

    function applyExcludeToSelected(excludeFlag) {{
      const vendor = currentVendor();
      const vendorName = vendor ? String(vendor.vendor || "") : "";
      let changed = 0;
      for (const key of selectedKeys) {{
        const row = currentRowMap.get(key);
        if (!row) continue;
        if (excludeFlag) {{
          if (!exclusionMap[key]) {{
            exclusionMap[key] = toExcludedMeta(row, vendorName);
            changed += 1;
          }}
        }} else if (exclusionMap[key]) {{
          delete exclusionMap[key];
          changed += 1;
        }}
      }}
      if (changed > 0) saveExcludedMap();
      renderAll();
    }}

    function normalizeRect(a, b) {{
      const x1 = Math.min(a.x, b.x);
      const y1 = Math.min(a.y, b.y);
      const x2 = Math.max(a.x, b.x);
      const y2 = Math.max(a.y, b.y);
      return {{ left: x1, top: y1, right: x2, bottom: y2, width: x2 - x1, height: y2 - y1 }};
    }}

    function intersectsRect(a, b) {{
      return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
    }}

    function onGridMouseDown(ev) {{
      if (!dragEnabled || ev.button !== 0) return;
      const grid = document.getElementById("grid");
      if (!grid || !grid.contains(ev.target)) return;
      const additive = !!(ev.ctrlKey || ev.metaKey || ev.shiftKey);
      lassoState = {{
        startX: ev.clientX,
        startY: ev.clientY,
        pendingClear: !additive,
        moved: false,
        hitKeys: new Set(),
      }};
      const lasso = document.getElementById("lasso");
      if (lasso) {{
        lasso.style.display = "block";
        lasso.style.left = String(ev.clientX) + "px";
        lasso.style.top = String(ev.clientY) + "px";
        lasso.style.width = "0px";
        lasso.style.height = "0px";
      }}
      grid.classList.add("drag-mode");
      document.addEventListener("mousemove", onGridMouseMove);
      document.addEventListener("mouseup", onGridMouseUp, true);
      ev.preventDefault();
    }}

    function onGridMouseMove(ev) {{
      if (!lassoState) return;
      const rect = normalizeRect(
        {{ x: lassoState.startX, y: lassoState.startY }},
        {{ x: ev.clientX, y: ev.clientY }}
      );
      if (rect.width > 3 || rect.height > 3) lassoState.moved = true;
      const lasso = document.getElementById("lasso");
      if (lasso) {{
        lasso.style.display = "block";
        lasso.style.left = String(rect.left) + "px";
        lasso.style.top = String(rect.top) + "px";
        lasso.style.width = String(rect.width) + "px";
        lasso.style.height = String(rect.height) + "px";
      }}
      if (!lassoState.moved) return;
      const nextHit = new Set();
      const cards = document.querySelectorAll("#grid .card[data-row-key]");
      cards.forEach((card) => {{
        const key = String(card.dataset.rowKey || "");
        const r = card.getBoundingClientRect();
        const cRect = {{ left: r.left, top: r.top, right: r.right, bottom: r.bottom }};
        if (intersectsRect(rect, cRect)) nextHit.add(key);
      }});
      lassoState.hitKeys = nextHit;
      updateVisibleCardStates(nextHit);
    }}

    function onGridMouseUp(_ev) {{
      if (!lassoState) return;
      const grid = document.getElementById("grid");
      if (grid) grid.classList.remove("drag-mode");
      const lasso = document.getElementById("lasso");
      if (lasso) lasso.style.display = "none";
      document.removeEventListener("mousemove", onGridMouseMove);
      document.removeEventListener("mouseup", onGridMouseUp, true);

      const moved = !!lassoState.moved;
      const hit = lassoState.hitKeys || new Set();
      const pendingClear = !!lassoState.pendingClear;
      lassoState = null;

      if (!moved) {{
        updateVisibleCardStates();
        return;
      }}
      if (pendingClear) selectedKeys.clear();
      for (const key of hit) selectedKeys.add(key);
      suppressCardClickUntil = Date.now() + 180;
      updateVisibleCardStates();
      setSelectionInfo();
    }}

    function onGridClick(ev) {{
      const now = Date.now();
      if (now < suppressCardClickUntil) return;
      const card = ev.target && ev.target.closest ? ev.target.closest(".card[data-row-key]") : null;
      if (!card) return;
      const key = String(card.dataset.rowKey || "");
      if (!key) return;
      const multi = !!(ev.ctrlKey || ev.metaKey || ev.shiftKey);
      if (!multi) selectedKeys.clear();
      if (multi && selectedKeys.has(key)) {{
        selectedKeys.delete(key);
      }} else {{
        selectedKeys.add(key);
      }}
      updateVisibleCardStates();
      setSelectionInfo();
    }}

    async function renderAll() {{
      renderVendors();
      const grid = document.getElementById("grid");
      if (!grid) return;
      const vendor = currentVendor();
      if (!vendor) {{
        grid.innerHTML = '<div class="empty">Nessun vendor disponibile.</div>';
        setStatus("0/0");
        currentRowMap = new Map();
        selectedKeys.clear();
        setSelectionInfo();
        return;
      }}
      setStatus("loading " + vendor.vendor + " ...");
      let payload;
      try {{
        payload = await loadVendorData(vendor);
      }} catch (_err) {{
        grid.innerHTML = '<div class="empty">Errore caricamento dati vendor.</div>';
        setStatus("errore");
        currentRowMap = new Map();
        selectedKeys.clear();
        setSelectionInfo();
        return;
      }}

      const rows = Array.isArray(payload.rows) ? payload.rows : [];
      const total = rows.length;
      const pages = Math.max(1, Math.ceil(total / Math.max(1, rowsPerPage)));
      pageIdx = Math.max(0, Math.min(pages - 1, pageIdx));
      const start = pageIdx * rowsPerPage;
      const end = Math.min(total, start + rowsPerPage);
      const part = rows.slice(start, end);
      const filterMode = String(document.getElementById("filter-mode")?.value || "all");
      const visibleRows = part.filter((row) => {{
        const excluded = !!exclusionMap[rowKey(row)];
        const suspect = Number(row.uniform_suspect || 0) === 1;
        if (filterMode === "excluded") return excluded;
        if (filterMode === "active") return !excluded;
        if (filterMode === "suspect") return suspect;
        return true;
      }});

      currentRowMap = new Map();
      selectedKeys.clear();

      if (visibleRows.length <= 0) {{
        grid.innerHTML = '<div class="empty">Nessuna riga con il filtro corrente in questa pagina.</div>';
        let vendorExcluded = 0;
        let vendorSuspects = 0;
        for (const row of rows) {{
          if (exclusionMap[rowKey(row)]) vendorExcluded += 1;
          if (Number(row.uniform_suspect || 0) === 1) vendorSuspects += 1;
        }}
        setStatus(vendor.vendor + " | " + (start + 1) + "-" + end + "/" + total + " | pagina " + (pageIdx + 1) + "/" + pages + " | sospette vendor: " + vendorSuspects + " | escluse vendor: " + vendorExcluded);
        setSelectionInfo();
        return;
      }}

      const cropW = 220;
      const cropH = 140;
      const cards = visibleRows.map((row) => {{
        const src = encodePath(row.image_path || "");
        const cropStyles = cropStrictStyles(row, cropW, cropH);
        const title = esc((row.dataset_folder || "") + " | " + (row.setup_name || ""));
        const action = esc(row.manual_folder_action || "");
        const updated = String(row.manual_rect_updated || "") === "1" ? "modified" : "not_modified";
        const path = esc(row.image_path || "");
        const split = esc(row.split || "");
        const source = esc(row.bbox_source || "");
        const key = rowKey(row);
        const excluded = !!exclusionMap[key];
        const suspect = Number(row.uniform_suspect || 0) === 1;
        const uReason = esc(row.uniform_reason || "");
        const uStdRaw = Number(row.gt_std);
        const uRangeRaw = Number(row.gt_range);
        const uStd = Number.isFinite(uStdRaw) ? uStdRaw.toFixed(3) : "-";
        const uRange = Number.isFinite(uRangeRaw) ? uRangeRaw.toFixed(3) : "-";
        const uUnique = Number(row.gt_unique || 0);
        const uBlack = Number(row.uniform_black_exact || 0) === 1;
        const uExact = Number(row.uniform_exact || 0) === 1;
        currentRowMap.set(key, row);
        return (
          '<article class="card' + (excluded ? ' excluded' : '') + '" data-row-key="' + esc(key) + '">' +
            '<div class="imgs">' +
              '<div class="crop" style="' + cropStyles.box + '">' +
                '<img loading="lazy" src="' + esc(src) + '" style="' + cropStyles.img + '" alt="crop ' + title + '">' +
              '</div>' +
            '</div>' +
            '<div class="meta">' +
              '<div><b>' + title + '</b>' +
                (excluded ? '<span class="pill">ESCLUSA</span>' : '') +
                (suspect ? '<span class="pill">UNIFORME?</span>' : '') +
              '</div>' +
              '<div>split=' + split + ' | action=' + action + ' | rect=' + updated + ' | source=' + source + '</div>' +
              '<div>uniform: suspect=' + (suspect ? "1" : "0") + ' | std=' + uStd + ' | uniq=' + uUnique + ' | range=' + uRange + ' | exact=' + (uExact ? "1" : "0") + ' | black=' + (uBlack ? "1" : "0") + '</div>' +
              '<div>reason: ' + uReason + '</div>' +
              '<div>bbox: ' + row.bbox_top + ',' + row.bbox_left + ',' + row.bbox_bottom + ',' + row.bbox_right + '</div>' +
              '<div class="path">' + path + '</div>' +
            '</div>' +
          '</article>'
        );
      }});
      grid.innerHTML = cards.join("");
      let vendorExcluded = 0;
      let vendorSuspects = 0;
      for (const row of rows) {{
        if (exclusionMap[rowKey(row)]) vendorExcluded += 1;
        if (Number(row.uniform_suspect || 0) === 1) vendorSuspects += 1;
      }}
      setStatus(
        vendor.vendor + " | " + (start + 1) + "-" + end + "/" + total + " | visibili " + visibleRows.length +
        " | pagina " + (pageIdx + 1) + "/" + pages + " | sospette vendor: " + vendorSuspects + " | escluse vendor: " + vendorExcluded
      );
      updateVisibleCardStates();
      setSelectionInfo();
    }}

    function moveVendor(delta) {{
      if (!Array.isArray(VENDORS) || VENDORS.length <= 0) return;
      vendorIdx = Math.max(0, Math.min(VENDORS.length - 1, vendorIdx + delta));
      pageIdx = 0;
      selectedKeys.clear();
      renderAll();
    }}

    function movePage(delta) {{
      pageIdx = Math.max(0, pageIdx + delta);
      selectedKeys.clear();
      renderAll();
    }}

    document.getElementById("grid")?.addEventListener("mousedown", onGridMouseDown);
    document.getElementById("grid")?.addEventListener("click", onGridClick);

    document.getElementById("btn-prev-vendor")?.addEventListener("click", () => moveVendor(-1));
    document.getElementById("btn-next-vendor")?.addEventListener("click", () => moveVendor(1));
    document.getElementById("btn-prev-page")?.addEventListener("click", () => movePage(-1));
    document.getElementById("btn-next-page")?.addEventListener("click", () => movePage(1));
    document.getElementById("btn-toggle-drag")?.addEventListener("click", () => {{
      dragEnabled = !dragEnabled;
      setDragButtonState();
    }});
    document.getElementById("btn-exclude-selected")?.addEventListener("click", () => applyExcludeToSelected(true));
    document.getElementById("btn-restore-selected")?.addEventListener("click", () => applyExcludeToSelected(false));
    document.getElementById("btn-clear-selection")?.addEventListener("click", clearSelection);
    document.getElementById("btn-export-excluded")?.addEventListener("click", exportExcludedCsv);
    document.getElementById("rows-per-page")?.addEventListener("change", (ev) => {{
      const next = Number(ev.target.value || rowsPerPage);
      rowsPerPage = Math.max(1, Number.isFinite(next) ? next : rowsPerPage);
      pageIdx = 0;
      selectedKeys.clear();
      renderAll();
    }});
    document.getElementById("filter-mode")?.addEventListener("change", () => {{
      selectedKeys.clear();
      renderAll();
    }});

    document.addEventListener("keydown", (ev) => {{
      const t = ev.target;
      const tag = String(t && t.tagName ? t.tagName : "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (ev.key === "ArrowUp") {{
        ev.preventDefault();
        moveVendor(-1);
        return;
      }}
      if (ev.key === "ArrowDown") {{
        ev.preventDefault();
        moveVendor(1);
        return;
      }}
      if (ev.key === "ArrowLeft") {{
        ev.preventDefault();
        movePage(-1);
        return;
      }}
      if (ev.key === "ArrowRight") {{
        ev.preventDefault();
        movePage(1);
        return;
      }}
      if (ev.key === "x" || ev.key === "X") {{
        ev.preventDefault();
        applyExcludeToSelected(true);
        return;
      }}
      if (ev.key === "r" || ev.key === "R") {{
        ev.preventDefault();
        applyExcludeToSelected(false);
      }}
    }});

    document.getElementById("rows-per-page").value = String(DEFAULT_ROWS_PER_PAGE);
    setDragButtonState();
    setSelectionInfo();
    renderAll();
  </script>
</body>
</html>
""".replace("__VENDORS__", vendors_json).replace("__ROWS_PER_PAGE__", rows_per_page_json)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build one-page fast review package with per-image template crop, grouped by vendor."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Training-ready manifest CSV to review.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_template_quick_review"),
        help="Output directory for review package.",
    )
    p.add_argument(
        "--rows-per-page",
        type=int,
        default=100,
        help="Rows rendered per page in the browser (default: 100).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    vendors_dir = output_dir / "vendors"
    output_dir.mkdir(parents=True, exist_ok=True)
    vendors_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    _, rows = _read_manifest(manifest_path)
    payload = _build_payload(rows)
    vendors: List[Dict[str, object]] = payload["vendors"]  # type: ignore[assignment]
    rows_by_vendor: Dict[str, List[Dict[str, object]]] = payload["rows_by_vendor"]  # type: ignore[assignment]

    if not vendors:
        raise RuntimeError("No rows available for review after parsing manifest.")

    for v in vendors:
        vendor_name = str(v["vendor"])
        _write_vendor_json(vendors_dir, v, rows_by_vendor[vendor_name])

    index_path = output_dir / "index.html"
    index_path.write_text(
        _render_index_html(vendors=vendors, rows_per_page=max(1, int(args.rows_per_page))),
        encoding="utf-8",
    )

    for root_name in ("Volumes", "Users"):
        src = Path("/") / root_name
        dst = output_dir / root_name
        try:
            if src.exists() and src.is_dir() and not dst.exists() and not dst.is_symlink():
                dst.symlink_to(src)
        except OSError:
            pass

    summary = {
        "manifest": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "index_html": index_path.as_posix(),
        "vendors": len(vendors),
        "rows_total": sum(int(v.get("count", 0) or 0) for v in vendors),
        "rows_per_page_default": max(1, int(args.rows_per_page)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Index: {index_path}", flush=True)
    print(f"Vendors: {summary['vendors']} | Rows: {summary['rows_total']}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
