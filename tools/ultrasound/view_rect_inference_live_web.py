#!/usr/bin/env python3
"""Live web viewer for rectangle inference CSV (no saved overlays)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template_string, request, send_file


HTML_PAGE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rect Inference Live Viewer</title>
  <style>
    :root {
      --bg: #f2f4f8;
      --card: #ffffff;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --primary: #0369a1;
      --danger: #b91c1c;
      --ok: #15803d;
      --gt: #16a34a;
      --pred: #dc2626;
      --glob: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 16px;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 12px;
    }
    @media (max-width: 1100px) {
      .wrap { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
    }
    .muted { color: var(--muted); font-size: 13px; }
    .row { margin-top: 10px; }
    select, button {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      font-size: 14px;
    }
    .btn-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .btn-main {
      border-color: var(--primary);
      background: #e0f2fe;
      color: #0c4a6e;
      font-weight: 700;
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 13px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .dot {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid #11182733;
    }
    .gt { background: var(--gt); }
    .pred { background: var(--pred); }
    .glob { background: var(--glob); }
    .info {
      margin-top: 10px;
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fafafa;
      min-height: 120px;
    }
    .viewer {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .toolbar button {
      width: auto;
    }
    .checks {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
    }
    .audit-row {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap: 8px;
      align-items: center;
    }
    .audit-row button.active {
      background: #fee2e2;
      border-color: #dc2626;
      color: #7f1d1d;
      font-weight: 700;
    }
    .audit-note {
      width: 100%;
      min-height: 64px;
      resize: vertical;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: 13px;
    }
    .canvas-wrap {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0b1220;
      overflow: auto;
      min-height: 400px;
    }
    canvas {
      display: block;
      width: 100%;
      height: auto;
      image-rendering: auto;
    }
    .err { color: var(--danger); font-weight: 700; }
    .oktext { color: var(--ok); font-weight: 700; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Rect Inference Viewer</h1>
      <div class="muted">Live overlay su immagine originale. Tasti: ← → (immagine), ↑ ↓ (cartella).</div>

      <div class="row">
        <label class="muted">Cartella</label>
        <select id="folderSelect"></select>
      </div>

      <div class="btn-row">
        <button id="prevFolderBtn">Cartella precedente</button>
        <button id="nextFolderBtn">Cartella successiva</button>
      </div>

      <div class="row">
        <div class="legend">
          <span class="chip"><span class="dot gt"></span>GT</span>
          <span class="chip"><span class="dot pred"></span>Pred routed</span>
        </div>
      </div>

      <div class="row info" id="folderInfo"></div>
      <div class="row info" id="imageInfo"></div>
    </section>

    <section class="card viewer">
      <div class="toolbar">
        <button id="prevImgBtn" class="btn-main">← Immagine</button>
        <button id="nextImgBtn" class="btn-main">Immagine →</button>
        <button id="firstImgBtn">Prima</button>
        <button id="lastImgBtn">Ultima</button>
      </div>
      <div class="checks">
        <label><input type="checkbox" id="showGt" checked> GT</label>
        <label><input type="checkbox" id="showPred" checked> Pred routed</label>
      </div>
      <div class="audit-row">
        <button id="toggleGtWrongBtn">Segna GT errata (G)</button>
        <label><input type="checkbox" id="onlyFlagged"> Solo segnate</label>
        <button id="exportGtWrongBtn">Export GT errate CSV</button>
      </div>
      <textarea id="gtNote" class="audit-note" placeholder="Nota su GT errata (autosave)"></textarea>
      <div id="gtFlagInfo" class="muted"></div>
      <div class="canvas-wrap">
        <canvas id="cv"></canvas>
      </div>
      <div id="status" class="muted">Caricamento...</div>
    </section>
  </div>

  <script>
    const folderSelect = document.getElementById("folderSelect");
    const folderInfo = document.getElementById("folderInfo");
    const imageInfo = document.getElementById("imageInfo");
    const statusEl = document.getElementById("status");
    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    const showGt = document.getElementById("showGt");
    const showPred = document.getElementById("showPred");
    const toggleGtWrongBtn = document.getElementById("toggleGtWrongBtn");
    const exportGtWrongBtn = document.getElementById("exportGtWrongBtn");
    const onlyFlagged = document.getElementById("onlyFlagged");
    const gtNote = document.getElementById("gtNote");
    const gtFlagInfo = document.getElementById("gtFlagInfo");

    let folders = [];
    let currentFolderId = 0;
    let allRows = [];
    let rows = [];
    let rowIdx = 0;
    let currentImg = null;
    let currentRec = null;
    let suppressNoteSync = false;
    const GT_FLAG_STORAGE_KEY = "rect_inference_gt_flags::" + String(window.location.pathname || "");
    let gtFlagMap = loadGtFlagMap();

    function esc(s) {
      return String(s ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }

    function recKey(rec) {
      return String(rec?.image_path || "");
    }

    function loadGtFlagMap() {
      try {
        const raw = window.localStorage.getItem(GT_FLAG_STORAGE_KEY);
        if (!raw) return {};
        const obj = JSON.parse(raw);
        if (obj && typeof obj === "object") return obj;
      } catch (_err) {
        return {};
      }
      return {};
    }

    function saveGtFlagMap() {
      try {
        window.localStorage.setItem(GT_FLAG_STORAGE_KEY, JSON.stringify(gtFlagMap));
      } catch (_err) {
        // ignore
      }
    }

    function folderFlaggedCount() {
      if (!Array.isArray(allRows)) return 0;
      let n = 0;
      for (const rec of allRows) {
        if (gtFlagMap[recKey(rec)]) n += 1;
      }
      return n;
    }

    function applyRowFilter(keepCurrent=true) {
      const currentKey = keepCurrent && currentRec ? recKey(currentRec) : "";
      if (onlyFlagged && onlyFlagged.checked) {
        rows = allRows.filter((rec) => !!gtFlagMap[recKey(rec)]);
      } else {
        rows = [...allRows];
      }
      if (!rows.length) {
        rowIdx = 0;
        return;
      }
      if (currentKey) {
        const idx = rows.findIndex((r) => recKey(r) === currentKey);
        rowIdx = idx >= 0 ? idx : Math.min(rowIdx, rows.length - 1);
      } else {
        rowIdx = Math.min(rowIdx, rows.length - 1);
      }
    }

    function buildFlagPayload(rec, noteText="") {
      return {
        image_path: String(rec?.image_path || ""),
        folder_path: String(rec?.folder_path || ""),
        folder_name: String(rec?.folder_name || ""),
        true_vendor: String(rec?.true_vendor || ""),
        pred_vendor: String(rec?.pred_vendor || ""),
        pred_vendor_conf: Number(rec?.pred_vendor_conf || 0),
        route_type: String(rec?.route_type || ""),
        routed_iou: Number(rec?.routed_iou || 0),
        gt_x1: Number(rec?.gt?.x1 || 0),
        gt_y1: Number(rec?.gt?.y1 || 0),
        gt_x2: Number(rec?.gt?.x2 || 0),
        gt_y2: Number(rec?.gt?.y2 || 0),
        pred_x1: Number(rec?.pred?.x1 || 0),
        pred_y1: Number(rec?.pred?.y1 || 0),
        pred_x2: Number(rec?.pred?.x2 || 0),
        pred_y2: Number(rec?.pred?.y2 || 0),
        note: String(noteText || ""),
        flagged_at: new Date().toISOString(),
      };
    }

    function updateFlagUi() {
      const total = Object.keys(gtFlagMap || {}).length;
      const inFolder = folderFlaggedCount();
      if (gtFlagInfo) gtFlagInfo.textContent = `GT errate segnate: ${total} | in cartella: ${inFolder}`;
      if (!currentRec) {
        if (toggleGtWrongBtn) {
          toggleGtWrongBtn.classList.remove("active");
          toggleGtWrongBtn.textContent = "Segna GT errata (G)";
        }
        if (gtNote) {
          suppressNoteSync = true;
          gtNote.value = "";
          suppressNoteSync = false;
        }
        return;
      }
      const k = recKey(currentRec);
      const flagged = !!gtFlagMap[k];
      if (toggleGtWrongBtn) {
        toggleGtWrongBtn.classList.toggle("active", flagged);
        toggleGtWrongBtn.textContent = flagged ? "Rimuovi flag GT errata (G)" : "Segna GT errata (G)";
      }
      if (gtNote) {
        suppressNoteSync = true;
        gtNote.value = flagged ? String(gtFlagMap[k].note || "") : "";
        suppressNoteSync = false;
      }
    }

    function toggleCurrentFlag() {
      if (!currentRec) return;
      const k = recKey(currentRec);
      if (gtFlagMap[k]) {
        delete gtFlagMap[k];
      } else {
        const noteText = gtNote ? String(gtNote.value || "").trim() : "";
        gtFlagMap[k] = buildFlagPayload(currentRec, noteText);
      }
      saveGtFlagMap();
      applyRowFilter(true);
      if (!rows.length) {
        currentRec = null;
        currentImg = null;
        ctx.clearRect(0, 0, cv.width, cv.height);
        imageInfo.innerHTML = "Nessuna immagine con il filtro corrente.";
      } else {
        rowIdx = Math.min(rowIdx, rows.length - 1);
        loadCurrentImage();
      }
      updateFlagUi();
    }

    function csvEscape(v) {
      const s = String(v ?? "");
      if (s.includes('"') || s.includes(",") || s.includes("\\n")) return '"' + s.replaceAll('"', '""') + '"';
      return s;
    }

    function downloadText(name, text, mime="text/plain;charset=utf-8") {
      const blob = new Blob([text], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1200);
    }

    function exportFlaggedCsv() {
      const vals = Object.values(gtFlagMap || {});
      if (!vals.length) {
        alert("Nessuna GT errata segnata.");
        return;
      }
      vals.sort((a, b) => String(a.image_path || "").localeCompare(String(b.image_path || "")));
      const header = [
        "image_path","folder_path","folder_name","true_vendor","pred_vendor","pred_vendor_conf",
        "route_type","routed_iou","gt_x1","gt_y1","gt_x2","gt_y2","pred_x1","pred_y1","pred_x2","pred_y2",
        "note","flagged_at"
      ];
      const lines = [header.join(",")];
      for (const row of vals) {
        lines.push(header.map((h) => csvEscape(row[h] ?? "")).join(","));
      }
      downloadText("gt_errate_ge_review.csv", lines.join("\\n"), "text/csv;charset=utf-8");
    }

    function drawBox(box, color, label, dashed=false) {
      if (!box) return;
      const x = Number(box.x1), y = Number(box.y1), w = Number(box.x2) - Number(box.x1), h = Number(box.y2) - Number(box.y1);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, Math.round(cv.width / 450));
      if (dashed) ctx.setLineDash([10, 7]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      const fontSize = Math.max(14, Math.round(cv.width / 65));
      ctx.font = `700 ${fontSize}px Segoe UI, sans-serif`;
      const padX = 8, padY = 4;
      const tw = ctx.measureText(label).width;
      const tx = Math.max(0, x);
      const ty = Math.max(fontSize + 4, y);
      ctx.fillStyle = color;
      ctx.fillRect(tx, ty - fontSize - padY, tw + padX * 2, fontSize + padY + 2);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, tx + padX, ty);
      ctx.restore();
    }

    function redrawCanvas() {
      if (!currentImg || !currentRec) return;
      cv.width = currentImg.naturalWidth || currentImg.width;
      cv.height = currentImg.naturalHeight || currentImg.height;
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(currentImg, 0, 0, cv.width, cv.height);
      if (showGt.checked) drawBox(currentRec.gt, "#16a34a", "GT");
      if (showPred.checked) drawBox(currentRec.pred, "#dc2626", "PRED");
    }

    async function loadFolders() {
      const res = await fetch("/api/folders");
      if (!res.ok) throw new Error("Errore caricamento cartelle");
      const data = await res.json();
      folders = data.folders || [];
      folderSelect.innerHTML = "";
      folders.forEach((f) => {
        const opt = document.createElement("option");
        opt.value = String(f.id);
        opt.textContent = `[${String(f.mean_iou).slice(0,6)}] ${f.folder_name} (${f.n_images})`;
        folderSelect.appendChild(opt);
      });
      if (!folders.length) throw new Error("Nessuna cartella trovata nel CSV.");
    }

    async function loadFolder(folderId, resetRow=true) {
      currentFolderId = folderId;
      folderSelect.value = String(folderId);
      statusEl.textContent = "Carico cartella...";
      const res = await fetch(`/api/folder/${folderId}`);
      if (!res.ok) throw new Error("Errore caricamento immagini cartella");
      const data = await res.json();
      allRows = data.rows || [];
      const f = data.folder || {};
      if (resetRow || rowIdx >= allRows.length) rowIdx = 0;
      applyRowFilter(true);
      const flaggedInFolder = folderFlaggedCount();
      folderInfo.innerHTML = [
        `Folder: <b>${esc(f.folder_name || "-")}</b>`,
        `Vendor true/pred: <b>${esc(f.true_vendor_mode || "-")}</b> / <b>${esc(f.pred_vendor_mode || "-")}</b>`,
        `Immagini: <b>${esc(f.n_images || 0)}</b> (filtrate: <b>${rows.length}</b>)`,
        `GT errate segnate in cartella: <b>${flaggedInFolder}</b>`,
        `Mean IoU: <b>${esc(f.mean_iou || "-")}</b>`,
      ].join("\\n");
      await loadCurrentImage();
      updateFlagUi();
    }

    async function loadCurrentImage() {
      if (!rows.length) {
        statusEl.innerHTML = '<span class="err">Nessuna immagine per questa cartella con il filtro corrente.</span>';
        imageInfo.innerHTML = "Nessuna immagine disponibile.";
        currentRec = null;
        currentImg = null;
        ctx.clearRect(0, 0, cv.width, cv.height);
        updateFlagUi();
        return;
      }
      const rec = rows[rowIdx];
      currentRec = rec;
      statusEl.textContent = `Carico immagine ${rowIdx + 1}/${rows.length}...`;
      const url = `/api/image?path=${encodeURIComponent(rec.image_path)}`;
      const img = new Image();
      img.onload = () => {
        currentImg = img;
        redrawCanvas();
        statusEl.innerHTML = `<span class="oktext">Immagine ${rowIdx + 1}/${rows.length}</span>`;
      };
      img.onerror = () => {
        currentImg = null;
        ctx.clearRect(0, 0, cv.width, cv.height);
        statusEl.innerHTML = `<span class="err">Errore caricamento: ${esc(rec.image_path)}</span>`;
      };
      imageInfo.innerHTML = [
        `Path: <code>${esc(rec.image_path)}</code>`,
        `True/Pred vendor: <b>${esc(rec.true_vendor)}</b> / <b>${esc(rec.pred_vendor)}</b> (conf=${Number(rec.pred_vendor_conf).toFixed(4)})`,
        `IoU routed: <b>${Number(rec.routed_iou).toFixed(4)}</b>`,
        `Route: <b>${esc(rec.route_type)}</b>`,
        `GT errata segnata: <b>${gtFlagMap[recKey(rec)] ? "SI" : "NO"}</b>`,
      ].join("\\n");
      updateFlagUi();
      img.src = url;
    }

    function changeImage(step) {
      if (!rows.length) return;
      rowIdx = Math.max(0, Math.min(rows.length - 1, rowIdx + step));
      loadCurrentImage();
    }

    function changeFolder(step) {
      if (!folders.length) return;
      const ids = folders.map(f => Number(f.id));
      const idx = Math.max(0, ids.indexOf(Number(currentFolderId)));
      const next = Math.max(0, Math.min(ids.length - 1, idx + step));
      loadFolder(ids[next], true);
    }

    document.getElementById("prevImgBtn").onclick = () => changeImage(-1);
    document.getElementById("nextImgBtn").onclick = () => changeImage(+1);
    document.getElementById("firstImgBtn").onclick = () => { rowIdx = 0; loadCurrentImage(); };
    document.getElementById("lastImgBtn").onclick = () => { rowIdx = Math.max(0, rows.length - 1); loadCurrentImage(); };
    document.getElementById("prevFolderBtn").onclick = () => changeFolder(-1);
    document.getElementById("nextFolderBtn").onclick = () => changeFolder(+1);
    folderSelect.onchange = () => loadFolder(Number(folderSelect.value), true);
    showGt.onchange = redrawCanvas;
    showPred.onchange = redrawCanvas;
    if (toggleGtWrongBtn) toggleGtWrongBtn.onclick = () => toggleCurrentFlag();
    if (exportGtWrongBtn) exportGtWrongBtn.onclick = () => exportFlaggedCsv();
    if (onlyFlagged) onlyFlagged.onchange = () => {
      applyRowFilter(true);
      loadCurrentImage();
      updateFlagUi();
    };
    if (gtNote) {
      gtNote.addEventListener("input", () => {
        if (suppressNoteSync) return;
        if (!currentRec) return;
        const key = recKey(currentRec);
        const text = String(gtNote.value || "").trim();
        if (!gtFlagMap[key]) {
          gtFlagMap[key] = buildFlagPayload(currentRec, text);
        } else {
          gtFlagMap[key].note = text;
          gtFlagMap[key].flagged_at = new Date().toISOString();
        }
        saveGtFlagMap();
        updateFlagUi();
      });
    }

    window.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight") { ev.preventDefault(); changeImage(+1); }
      if (ev.key === "ArrowLeft") { ev.preventDefault(); changeImage(-1); }
      if (ev.key === "ArrowDown") { ev.preventDefault(); changeFolder(+1); }
      if (ev.key === "ArrowUp") { ev.preventDefault(); changeFolder(-1); }
      if (ev.key === "g" || ev.key === "G") { ev.preventDefault(); toggleCurrentFlag(); }
    });

    async function boot() {
      try {
        await loadFolders();
        await loadFolder(Number(folders[0].id), true);
      } catch (e) {
        statusEl.innerHTML = `<span class="err">${esc(e.message || e)}</span>`;
      }
    }
    boot();
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live viewer for rect inference predictions CSV.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "artifacts/40_outputs_eval/rect_inference_vendor_routing_live_20260330/"
            "perfolder20_thr090/predictions_routed_test.csv"
        ),
        help="CSV generated by infer_ultrasound_rect_vendor_routing.py",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--debug", action="store_true")
    return parser


def _to_float(row: Dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "") or 0.0)
        if not math.isfinite(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def _folder_from_image_path(image_path: str) -> str:
    marker = "/image_samples/"
    if marker in image_path:
        return image_path.split(marker, 1)[0]
    return str(Path(image_path).parent)


def _load_predictions(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    rows: List[Dict[str, Any]] = []
    by_folder: Dict[str, List[Dict[str, Any]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            image_path = str(r.get("image_path", "")).strip()
            if not image_path:
                continue
            folder_path = _folder_from_image_path(image_path)
            folder_name = Path(folder_path).name
            rec = {
                "image_path": image_path,
                "folder_path": folder_path,
                "folder_name": folder_name,
                "true_vendor": str(r.get("true_vendor", "")),
                "pred_vendor": str(r.get("pred_vendor", "")),
                "pred_vendor_conf": _to_float(r, "pred_vendor_conf"),
                "route_type": str(r.get("route_type", "")),
                "routed_iou": _to_float(r, "routed_iou"),
                "global_iou": _to_float(r, "global_iou"),
                "delta_iou": _to_float(r, "delta_iou"),
                "gt": {
                    "x1": _to_float(r, "gt_x1"),
                    "y1": _to_float(r, "gt_y1"),
                    "x2": _to_float(r, "gt_x2"),
                    "y2": _to_float(r, "gt_y2"),
                },
                "pred": {
                    "x1": _to_float(r, "pred_x1"),
                    "y1": _to_float(r, "pred_y1"),
                    "x2": _to_float(r, "pred_x2"),
                    "y2": _to_float(r, "pred_y2"),
                },
                "global": {
                    "x1": _to_float(r, "global_x1"),
                    "y1": _to_float(r, "global_y1"),
                    "x2": _to_float(r, "global_x2"),
                    "y2": _to_float(r, "global_y2"),
                },
            }
            rows.append(rec)
            by_folder.setdefault(folder_path, []).append(rec)

    folders: List[Dict[str, Any]] = []
    folder_rows_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for idx, folder_path in enumerate(sorted(by_folder)):
        rs = by_folder[folder_path]
        n = len(rs)
        routed_ious = [float(x["routed_iou"]) for x in rs if math.isfinite(float(x["routed_iou"]))]
        confs = [float(x["pred_vendor_conf"]) for x in rs if math.isfinite(float(x["pred_vendor_conf"]))]
        mean_iou = (sum(routed_ious) / len(routed_ious)) if routed_ious else 0.0
        mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        true_vendor_mode = max((x["true_vendor"] for x in rs), key=lambda v: sum(1 for z in rs if z["true_vendor"] == v))
        pred_vendor_mode = max((x["pred_vendor"] for x in rs), key=lambda v: sum(1 for z in rs if z["pred_vendor"] == v))
        folders.append(
            {
                "id": idx,
                "folder_path": folder_path,
                "folder_name": Path(folder_path).name,
                "n_images": n,
                "mean_iou": round(mean_iou, 6),
                "mean_conf": round(mean_conf, 6),
                "true_vendor_mode": true_vendor_mode,
                "pred_vendor_mode": pred_vendor_mode,
            }
        )
        folder_rows_by_id[idx] = rs

    folders.sort(key=lambda x: (float(x["mean_iou"]), x["folder_name"]))
    remap: Dict[int, int] = {}
    for new_id, f in enumerate(folders):
        old_id = int(f["id"])
        remap[new_id] = old_id
        f["id"] = new_id
    remapped_rows = {new_id: folder_rows_by_id[old_id] for new_id, old_id in remap.items()}
    return rows, folders, remapped_rows


def create_app(predictions_csv: Path) -> Flask:
    rows, folders, folder_rows = _load_predictions(predictions_csv)
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.get("/api/folders")
    def api_folders():
        return jsonify(
            {
                "predictions_csv": predictions_csv.as_posix(),
                "num_rows": len(rows),
                "num_folders": len(folders),
                "folders": folders,
            }
        )

    @app.get("/api/folder/<int:folder_id>")
    def api_folder(folder_id: int):
        if folder_id not in folder_rows:
            return jsonify({"error": f"folder_id non valido: {folder_id}"}), 404
        folder = next((f for f in folders if int(f["id"]) == folder_id), None)
        return jsonify({"folder": folder, "rows": folder_rows[folder_id]})

    @app.get("/api/image")
    def api_image():
        raw = str(request.args.get("path", "")).strip()
        if not raw:
            return jsonify({"error": "path mancante"}), 400
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            return jsonify({"error": f"file non trovato: {p.as_posix()}"}), 404
        return send_file(p)

    return app


def main() -> int:
    args = _build_parser().parse_args()
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be in 1..65535.")
    predictions_csv = args.predictions_csv.expanduser().resolve()
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV non trovato: {predictions_csv}")

    app = create_app(predictions_csv=predictions_csv)
    print(
        f"Rect live viewer running on http://{args.host}:{args.port} | csv={predictions_csv}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
