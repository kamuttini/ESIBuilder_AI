#!/usr/bin/env python3
"""Build an interactive HTML review for probe model -> probe type mapping."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _safe_text(value: str, max_len: int = 260) -> str:
    out = (value or "").strip()
    if len(out) <= max_len:
        return out
    return out[: max_len - 1] + "…"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _resolve_legacy_path(path_str: str, dataset_root: Optional[Path] = None) -> str:
    raw = (path_str or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add_candidate(candidate: Path) -> None:
        key = candidate.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    dataset_root_resolved: Optional[Path] = None
    if dataset_root is not None:
        dataset_root = dataset_root.expanduser()
        if dataset_root.exists():
            dataset_root_resolved = dataset_root.resolve()

    # If a dataset root is explicitly requested, force that source first.
    if dataset_root_resolved is not None:
        if p.exists() and str(p).startswith(dataset_root_resolved.as_posix()):
            return p.as_posix()
        if "/data/Dataset/" in raw:
            suffix = raw.split("/data/Dataset/", 1)[1]
            _add_candidate(dataset_root_resolved / suffix)
        if "/Dataset/" in raw:
            suffix = raw.split("/Dataset/", 1)[1]
            _add_candidate(dataset_root_resolved / suffix)
        _add_candidate(dataset_root_resolved / p.name)

        for cand in candidates:
            if cand.exists():
                return cand.as_posix()
        # Keep volume as single source even when an exact path is missing.
        if candidates:
            return candidates[0].as_posix()

    if p.exists():
        return p.as_posix()

    marker = "/Users/camilla/Documents/Develop/ESIBuilder_AI/Dataset/"
    if marker in raw:
        _add_candidate(Path(raw.replace(marker, "/Users/camilla/Documents/Develop/ESIBuilder_AI/data/Dataset/")))
    if "/data/Dataset/" in raw:
        suffix = raw.split("/data/Dataset/", 1)[1]
        _add_candidate(PROJECT_ROOT / "data" / "Dataset" / suffix)
    if "/Dataset/" in raw:
        suffix = raw.split("/Dataset/", 1)[1]
        _add_candidate(PROJECT_ROOT / "data" / "Dataset" / suffix)

    for cand in candidates:
        if cand.exists():
            return cand.as_posix()
    return raw


def _iter_folder_images(folder_path: str) -> List[str]:
    base = Path(folder_path)
    image_dir = base / "image_samples"
    if not image_dir.is_dir():
        return []
    out: List[str] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "negative" in path.name.lower():
            continue
        out.append(path.as_posix())
    return out


def _extract_storage_root(path_str: str) -> str:
    p = Path(path_str)
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        return Path(parts[0], parts[1], parts[2]).as_posix()
    if "Dataset" in parts:
        idx = parts.index("Dataset")
        return Path(*parts[: idx + 1]).as_posix()
    return p.parent.as_posix()


def _load_probe_type_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("probe_type_id") or "").strip()
            if not key:
                continue
            out[key] = _safe_text(row.get("probe_type_description") or "", 120)
    return out


def _load_probe_summary(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Probe summary CSV non trovato: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["probe_id"] = str(r.get("probe_id") or "").strip()
        r["probe_name"] = _safe_text(r.get("probe_name") or "", 120)
        r["type_status"] = (r.get("type_status") or "UNKNOWN").strip().upper() or "UNKNOWN"
        r["primary_probe_type_id"] = str(r.get("primary_probe_type_id") or "").strip()
        r["primary_probe_type_description"] = _safe_text(r.get("primary_probe_type_description") or "", 120)
        r["all_type_evidence"] = _safe_text(r.get("all_type_evidence") or "", 260)
    rows.sort(key=lambda x: (int(x["probe_id"]) if x["probe_id"].isdigit() else 9999, x["probe_name"]))
    return rows


def _load_probe_type_evidence(path: Path) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            probe_id = str(row.get("probe_id") or "").strip()
            if not probe_id:
                continue
            out[probe_id].append(
                {
                    "probe_id": probe_id,
                    "probe_name": str(row.get("probe_name") or "").strip(),
                    "probe_type_id": str(row.get("probe_type_id") or "").strip(),
                    "probe_type_description": str(row.get("probe_type_description") or "").strip(),
                    "samples": str(row.get("samples") or "").strip(),
                    "share_within_probe": str(row.get("share_within_probe") or "").strip(),
                }
            )

    for probe_id, rows in out.items():
        rows.sort(key=lambda x: _safe_float(x.get("share_within_probe"), 0.0), reverse=True)
        out[probe_id] = rows
    return out


def _load_folder_predictions(path: Path, dataset_root: Optional[Path]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            folder_path = _resolve_legacy_path((row.get("folder_path") or "").strip(), dataset_root=dataset_root)
            if not folder_path:
                continue
            out[folder_path] = {
                "folder_path": folder_path,
                "folder_name": _safe_text((row.get("folder_name") or Path(folder_path).name), 180),
                "predicted_probe_id": str(row.get("predicted_probe_id") or "").strip(),
                "folder_top1_prob": f"{_safe_float(row.get('folder_top1_prob'), 0.0):.4f}",
                "folder_vote_ratio": f"{_safe_float(row.get('folder_vote_ratio'), 0.0):.4f}",
                "folder_margin_top1_top2": f"{_safe_float(row.get('folder_margin_top1_top2'), 0.0):.4f}",
                "status": (row.get("status") or "").strip(),
                "topk": _safe_text(row.get("topk") or "", 220),
            }
    return out


def _load_per_image(path: Path, max_images_per_folder: int, dataset_root: Optional[Path]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            folder_path = _resolve_legacy_path((row.get("folder_path") or "").strip(), dataset_root=dataset_root)
            image_path = _resolve_legacy_path((row.get("image_path") or "").strip(), dataset_root=dataset_root)
            if not folder_path or not image_path:
                continue
            bucket = out[folder_path]
            if max_images_per_folder > 0 and len(bucket) >= max_images_per_folder:
                continue
            bucket.append(image_path)
    return out


def _build_probe_items(
    probe_summary: List[Dict[str, str]],
    folder_meta: Dict[str, Dict[str, str]],
    images_by_folder: Dict[str, List[str]],
    max_folders_per_probe: int,
    max_images_per_folder: int,
    seed: int,
) -> List[Dict[str, object]]:
    by_probe_folders: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for folder_path, meta in folder_meta.items():
        pid = meta.get("predicted_probe_id") or ""
        if not pid:
            continue
        entry = dict(meta)
        entry["images"] = images_by_folder.get(folder_path, [])
        by_probe_folders[pid].append(entry)

    rng = random.Random(seed)
    full_folder_images_cache: Dict[str, List[str]] = {}
    items: List[Dict[str, object]] = []
    for row in probe_summary:
        pid = row.get("probe_id") or ""
        folders = by_probe_folders.get(pid, [])
        folders = sorted(
            folders,
            key=lambda x: (
                _safe_float(x.get("folder_top1_prob"), 0.0),
                _safe_float(x.get("folder_vote_ratio"), 0.0),
                x.get("folder_name") or "",
            ),
            reverse=True,
        )
        if max_folders_per_probe > 0 and len(folders) > max_folders_per_probe:
            # Keep top strong evidence and randomize tail among equally useful candidates.
            head = folders[: max_folders_per_probe // 2]
            tail = folders[max_folders_per_probe // 2 :]
            rng.shuffle(tail)
            folders = (head + tail)[:max_folders_per_probe]

        normalized_folders: List[Dict[str, object]] = []
        for folder in folders:
            folder_path = str(folder.get("folder_path") or "")
            if folder_path not in full_folder_images_cache:
                full_folder_images_cache[folder_path] = _iter_folder_images(folder_path)
            all_images = full_folder_images_cache[folder_path]
            sample_images = list(folder.get("images", []))
            if not sample_images:
                if max_images_per_folder > 0:
                    sample_images = all_images[:max_images_per_folder]
                else:
                    sample_images = all_images[:6]
            normalized = dict(folder)
            normalized["images"] = sample_images
            normalized["all_images"] = all_images
            normalized_folders.append(normalized)

        item = {
            "probe_id": pid,
            "probe_name": row.get("probe_name") or "",
            "type_status": row.get("type_status") or "UNKNOWN",
            "primary_probe_type_id": row.get("primary_probe_type_id") or "",
            "primary_probe_type_description": row.get("primary_probe_type_description") or "",
            "all_type_evidence": row.get("all_type_evidence") or "",
            "folder_count_predicted": len(by_probe_folders.get(pid, [])),
            "preview_folders": normalized_folders,
        }
        items.append(item)
    return items


def _build_html(
    title: str,
    probe_items: List[Dict[str, object]],
    probe_type_options: List[Dict[str, str]],
    source_notes: List[str],
) -> str:
    payload = json.dumps(probe_items, ensure_ascii=False)
    type_payload = json.dumps(probe_type_options, ensure_ascii=False)
    notes = " | ".join(source_notes)

    style = """
:root {
  --bg: #f7f8fc;
  --panel: #ffffff;
  --line: #d7dfef;
  --txt: #1a2236;
  --muted: #5a6783;
  --ok: #0e6a3a;
  --warn: #8a4b00;
  --unk: #45506a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  color: var(--txt);
  background: radial-gradient(circle at 10% 0%, #e7eefb, var(--bg) 38%);
}
.wrap { max-width: 1880px; margin: 0 auto; padding: 14px; display: grid; gap: 10px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 auto; }
.small { color: var(--muted); font-size: 12px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.select, .input, .button {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 8px;
  font-size: 13px;
}
.button { cursor: pointer; }
.button:hover { background: #f2f6ff; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 10px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 9px; }
.card-corrected { border-color: #2b6ad8; box-shadow: 0 0 0 1px #2b6ad8 inset; }
.hdr { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.badge { border-radius: 999px; font-size: 11px; font-weight: 700; padding: 2px 8px; display: inline-block; }
.badge-ok { background: #d7f4e5; color: var(--ok); }
.badge-warn { background: #ffe7ca; color: var(--warn); }
.badge-unk { background: #e8ebf2; color: var(--unk); }
.meta { margin-top: 6px; font-size: 12px; line-height: 1.35; }
.folders { margin-top: 8px; border-top: 1px dashed var(--line); padding-top: 8px; display: grid; gap: 8px; }
.folder { border: 1px solid var(--line); border-radius: 8px; padding: 7px; }
.folder-actions { margin-top: 6px; }
.thumbs { margin-top: 6px; display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.thumb-item { border: 1px solid var(--line); border-radius: 8px; padding: 5px; background: #fbfcff; }
.thumb-btn { border: 0; margin: 0; padding: 0; width: 100%; background: transparent; cursor: zoom-in; }
.thumb { width: 100%; height: 110px; object-fit: contain; background: #000; border-radius: 6px; display: block; }
.thumb-name { margin-top: 4px; color: #233256; font-size: 11px; font-weight: 600; word-break: break-word; }
.thumb-path { color: #354565; font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: normal; overflow-wrap: anywhere; line-height: 1.2; margin-top: 2px; }
.path { color: #354565; font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: normal; overflow-wrap: anywhere; line-height: 1.2; }
.legend { margin-top: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; }

.viewer {
  position: fixed;
  inset: 0;
  background: rgba(5, 10, 20, 0.92);
  z-index: 9999;
  display: none;
  flex-direction: column;
}
.viewer.open { display: flex; }
.viewer-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}
.viewer-title { color: #f4f7ff; font-size: 13px; font-weight: 600; }
.viewer-counter { color: #d7e1ff; font-size: 12px; }
.viewer-stage {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
}
.viewer-img {
  max-width: calc(100vw - 20px);
  max-height: calc(100vh - 160px);
  object-fit: contain;
  border-radius: 8px;
  background: #000;
}
.viewer-nav {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 8px;
}
.viewer-meta {
  color: #d5ddf4;
  font-size: 11px;
  white-space: normal;
  overflow-wrap: anywhere;
  text-align: center;
  padding: 0 12px 12px;
}
.viewer-btn {
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.12);
  color: #fff;
  padding: 6px 10px;
  cursor: pointer;
}
.viewer-btn:hover { background: rgba(255, 255, 255, 0.2); }
body.no-scroll { overflow: hidden; }
"""

    return f"""<!doctype html>
<html lang=\"it\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>{style}</style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <div class=\"row\">
        <div class=\"grow\">
          <div><b>{html.escape(title)}</b></div>
          <div class=\"small\">{html.escape(notes)}</div>
        </div>
        <div class=\"small\">modelli totali: <b id=\"modelsTotal\">0</b></div>
        <div class=\"small\">modelli visibili: <b id=\"modelsVisible\">0</b></div>
        <div class=\"small\">correzioni: <b id=\"corrCount\">0</b></div>
      </div>
      <div class=\"row\" style=\"margin-top:8px;\">
        <label class=\"small\">Stato</label>
        <select id=\"statusFilter\" class=\"select\">
          <option value=\"ALL\">All</option>
          <option value=\"UNIVOCO\">UNIVOCO</option>
          <option value=\"AMBIGUO\">AMBIGUO</option>
          <option value=\"UNKNOWN\">UNKNOWN</option>
        </select>
        <label class=\"small\">Tipo auto</label>
        <select id=\"typeFilter\" class=\"select\"></select>
        <label class=\"small\">Cerca</label>
        <input id=\"searchFilter\" class=\"input\" placeholder=\"probe id/nome/evidenza/cartella...\" />
        <button id=\"exportBtn\" class=\"button\" type=\"button\">Export correzioni CSV</button>
        <button id=\"resetBtn\" class=\"button\" type=\"button\">Reset correzioni</button>
      </div>
      <div class=\"legend\">
        <table>
          <thead><tr><th>Type ID</th><th>Descrizione</th></tr></thead>
          <tbody id=\"typeLegendBody\"></tbody>
        </table>
      </div>
    </div>
    <div id=\"grid\" class=\"grid\"></div>
  </div>
  <div id=\"viewer\" class=\"viewer\" role=\"dialog\" aria-modal=\"true\" aria-hidden=\"true\">
    <div class=\"viewer-top\">
      <div id=\"viewerTitle\" class=\"viewer-title\"></div>
      <button id=\"viewerCloseTop\" type=\"button\" class=\"viewer-btn\">Chiudi</button>
    </div>
    <div class=\"viewer-stage\">
      <img id=\"viewerImg\" class=\"viewer-img\" src=\"\" alt=\"fullscreen preview\" />
    </div>
    <div class=\"viewer-nav\">
      <button id=\"viewerPrev\" type=\"button\" class=\"viewer-btn\">Precedente</button>
      <span id=\"viewerCounter\" class=\"viewer-counter\">0 / 0</span>
      <button id=\"viewerNext\" type=\"button\" class=\"viewer-btn\">Successiva</button>
    </div>
    <div id=\"viewerMeta\" class=\"viewer-meta mono\"></div>
  </div>

  <script>
    const DATA = {payload};
    const TYPE_OPTIONS = {type_payload};
    const CORR_KEY = \"probe_model_type_corrections_v1\";

    const modelsTotal = document.getElementById(\"modelsTotal\");
    const modelsVisible = document.getElementById(\"modelsVisible\");
    const corrCount = document.getElementById(\"corrCount\");
    const statusFilter = document.getElementById(\"statusFilter\");
    const typeFilter = document.getElementById(\"typeFilter\");
    const searchFilter = document.getElementById(\"searchFilter\");
    const exportBtn = document.getElementById(\"exportBtn\");
    const resetBtn = document.getElementById(\"resetBtn\");
    const grid = document.getElementById(\"grid\");
    const typeLegendBody = document.getElementById(\"typeLegendBody\");
    const viewer = document.getElementById(\"viewer\");
    const viewerTitle = document.getElementById(\"viewerTitle\");
    const viewerImg = document.getElementById(\"viewerImg\");
    const viewerMeta = document.getElementById(\"viewerMeta\");
    const viewerCounter = document.getElementById(\"viewerCounter\");
    const viewerPrev = document.getElementById(\"viewerPrev\");
    const viewerNext = document.getElementById(\"viewerNext\");
    const viewerCloseTop = document.getElementById(\"viewerCloseTop\");

    const viewerState = {{
      open: false,
      itemIdx: -1,
      folderIdx: -1,
      images: [],
      index: 0,
    }};

    let corrections = {{}};
    try {{ corrections = JSON.parse(localStorage.getItem(CORR_KEY) || \"{{}}\"); }} catch (_) {{ corrections = {{}}; }}

    function esc(text) {{
      return String(text || \"\")
        .replaceAll(\"&\", \"&amp;\")
        .replaceAll(\"<\", \"&lt;\")
        .replaceAll(\">\", \"&gt;\")
        .replaceAll('"', \"&quot;\");
    }}

    function toFileUrl(path) {{
      const p = String(path || \"\");
      if (!p) return \"\";
      return p.startsWith(\"file://\") ? p : `file://${{p}}`;
    }}

    function fileName(path) {{
      const p = String(path || \"\");
      if (!p) return \"\";
      const bits = p.split(\"/\");
      return bits[bits.length - 1] || p;
    }}

    function getFolder(itemIdx, folderIdx) {{
      const item = DATA[itemIdx];
      if (!item) return null;
      const folders = item.preview_folders || [];
      return folders[folderIdx] || null;
    }}

    function openViewer(itemIdx, folderIdx, startIndex) {{
      const item = DATA[itemIdx];
      const folder = getFolder(itemIdx, folderIdx);
      if (!item || !folder) return;
      const allImages = Array.isArray(folder.all_images) && folder.all_images.length
        ? folder.all_images
        : (Array.isArray(folder.images) ? folder.images : []);
      const images = allImages.map(x => String(x || \"\")).filter(Boolean);
      if (!images.length) return;

      viewerState.open = true;
      viewerState.itemIdx = itemIdx;
      viewerState.folderIdx = folderIdx;
      viewerState.images = images;
      viewerState.index = Math.max(0, Math.min(images.length - 1, startIndex || 0));

      document.body.classList.add(\"no-scroll\");
      viewer.classList.add(\"open\");
      viewer.setAttribute(\"aria-hidden\", \"false\");
      updateViewer();
    }}

    function closeViewer() {{
      viewerState.open = false;
      viewerState.images = [];
      viewerState.index = 0;
      viewerState.itemIdx = -1;
      viewerState.folderIdx = -1;
      viewer.classList.remove(\"open\");
      viewer.setAttribute(\"aria-hidden\", \"true\");
      viewerImg.src = \"\";
      viewerMeta.textContent = \"\";
      viewerCounter.textContent = \"0 / 0\";
      document.body.classList.remove(\"no-scroll\");
    }}

    function updateViewer() {{
      if (!viewerState.open || !viewerState.images.length) return;
      const item = DATA[viewerState.itemIdx] || {{}};
      const folder = getFolder(viewerState.itemIdx, viewerState.folderIdx) || {{}};
      const imagePath = viewerState.images[viewerState.index] || \"\";

      viewerImg.src = toFileUrl(imagePath);
      viewerCounter.textContent = `${{viewerState.index + 1}} / ${{viewerState.images.length}}`;
      viewerTitle.textContent = `#${{item.probe_id || \"\"}} ${{item.probe_name || \"\"}} | ${{folder.folder_name || \"\"}}`;
      viewerMeta.textContent = imagePath;
    }}

    function stepViewer(delta) {{
      if (!viewerState.open || !viewerState.images.length) return;
      const n = viewerState.images.length;
      viewerState.index = (viewerState.index + delta + n) % n;
      updateViewer();
    }}

    function intAttr(node, key) {{
      const raw = node.getAttribute(key) || \"-1\";
      const out = parseInt(raw, 10);
      return Number.isFinite(out) ? out : -1;
    }}

    function typeLabel(typeId, typeDesc) {{
      if (!typeId) return \"UNKNOWN\";
      return `${{typeId}} - ${{typeDesc || \"\"}}`;
    }}

    function statusBadgeClass(status) {{
      if (status === \"UNIVOCO\") return \"badge badge-ok\";
      if (status === \"AMBIGUO\") return \"badge badge-warn\";
      return \"badge badge-unk\";
    }}

    function currentCorr(probeId) {{
      return corrections[probeId] || {{ corrected_type_id: \"__AUTO__\", note: \"\" }};
    }}

    function effectiveTypeId(item) {{
      const c = currentCorr(item.probe_id);
      if (c.corrected_type_id && c.corrected_type_id !== \"__AUTO__\") return c.corrected_type_id;
      return item.primary_probe_type_id || \"UNKNOWN\";
    }}

    function setCorrectionCount() {{
      let n = 0;
      for (const [_, c] of Object.entries(corrections)) {{
        const t = String(c.corrected_type_id || \"__AUTO__\");
        const note = String(c.note || \"\").trim();
        if (t !== \"__AUTO__\" || note) n += 1;
      }}
      corrCount.textContent = String(n);
    }}

    function setTypeFilterOptions() {{
      const values = new Set([\"UNKNOWN\"]);
      for (const it of DATA) values.add(it.primary_probe_type_id || \"UNKNOWN\");
      const sorted = [...values].sort((a,b)=>a.localeCompare(b, undefined, {{numeric:true}}));

      typeFilter.innerHTML = \"\";
      const all = document.createElement(\"option\");
      all.value = \"ALL\";
      all.textContent = \"All\";
      typeFilter.appendChild(all);
      for (const id of sorted) {{
        const opt = document.createElement(\"option\");
        opt.value = id;
        if (id === \"UNKNOWN\") {{
          opt.textContent = \"UNKNOWN\";
        }} else {{
          const meta = TYPE_OPTIONS.find(t => String(t.id) === String(id));
          opt.textContent = `${{id}} - ${{meta ? meta.description : \"\"}}`;
        }}
        typeFilter.appendChild(opt);
      }}
    }}

    function matches(item) {{
      const sf = statusFilter.value;
      if (sf !== \"ALL\" && item.type_status !== sf) return false;

      const tf = typeFilter.value;
      if (tf !== \"ALL\" && String(effectiveTypeId(item)) !== String(tf)) return false;

      const q = String(searchFilter.value || \"\").trim().toLowerCase();
      if (!q) return true;
      const folderBits = (item.preview_folders || []).map(f => `${{f.folder_name}} ${{f.folder_path}}`).join(\" \");
      const hay = `${{item.probe_id}} ${{item.probe_name}} ${{item.type_status}} ${{item.primary_probe_type_description}} ${{item.all_type_evidence}} ${{folderBits}}`.toLowerCase();
      return hay.includes(q);
    }}

    function render() {{
      const rows = DATA.filter(matches);
      modelsVisible.textContent = String(rows.length);
      modelsTotal.textContent = String(DATA.length);

      grid.innerHTML = rows.map((item, itemIdx) => {{
        const corr = currentCorr(item.probe_id);
        const effType = effectiveTypeId(item);
        const hasCorr = (corr.corrected_type_id && corr.corrected_type_id !== \"__AUTO__\") || String(corr.note || \"\").trim().length > 0;
        const autoTypeLabel = typeLabel(item.primary_probe_type_id, item.primary_probe_type_description);

        const correctionOptions = [
          `<option value=\"__AUTO__\">AUTO (${{esc(autoTypeLabel)}})</option>`,
          ...TYPE_OPTIONS.map(t => `<option value=\"${{esc(String(t.id))}}\">${{esc(String(t.id))}} - ${{esc(t.description)}}</option>`),
          `<option value=\"UNKNOWN\">UNKNOWN</option>`,
        ].join(\"\");

        const foldersHtml = (item.preview_folders || []).map((f, folderIdx) => {{
          const allImages = Array.isArray(f.all_images) ? f.all_images : [];
          const thumbs = (f.images || []).map(ip => {{
            const ipText = String(ip || \"\");
            const src = toFileUrl(ipText);
            return `
              <div class=\"thumb-item\">
                <button class=\"thumb-btn open-image-viewer\" type=\"button\" data-item-idx=\"${{itemIdx}}\" data-folder-idx=\"${{folderIdx}}\" data-image-path=\"${{esc(ipText)}}\" title=\"${{esc(ipText)}}\">
                  <img class=\"thumb\" loading=\"lazy\" src=\"${{esc(src)}}\" alt=\"preview\" />
                </button>
                <div class=\"thumb-name\">${{esc(fileName(ipText))}}</div>
                <div class=\"thumb-path\" title=\"${{esc(ipText)}}\">${{esc(ipText)}}</div>
              </div>
            `;
          }}).join(\"\");
          return `
            <div class=\"folder\">
              <div class=\"meta\"><b>${{esc(f.folder_name || \"\")}}</b></div>
              <div class=\"small\">top1=${{esc(f.folder_top1_prob || \"0\")}} | vote=${{esc(f.folder_vote_ratio || \"0\")}} | margin=${{esc(f.folder_margin_top1_top2 || \"0\")}} | status=${{esc(f.status || \"\")}}</div>
              <div class=\"small\">topk=${{esc(f.topk || \"\")}}</div>
              <div class=\"path\" title=\"${{esc(f.folder_path || \"\")}}\">${{esc(f.folder_path || \"\")}}</div>
              <div class=\"folder-actions\">
                <button type=\"button\" class=\"button open-folder-viewer\" data-item-idx=\"${{itemIdx}}\" data-folder-idx=\"${{folderIdx}}\">Apri cartella fullscreen (${{esc(String(allImages.length))}} immagini)</button>
              </div>
              <div class=\"thumbs\">${{thumbs || \"<span class='small'>Nessuna immagine preview</span>\"}}</div>
            </div>
          `;
        }}).join(\"\");

        return `
          <div class=\"card ${{hasCorr ? \"card-corrected\" : \"\"}}\" data-probe-id=\"${{esc(item.probe_id)}}\">
            <div class=\"hdr\">
              <div><b>#${{esc(item.probe_id)}} ${{esc(item.probe_name)}}</b></div>
              <span class=\"${{statusBadgeClass(item.type_status)}}\">${{esc(item.type_status)}}</span>
            </div>
            <div class=\"meta\">Tipo auto: <b>${{esc(autoTypeLabel)}}</b> | tipo effettivo corrente: <b>${{esc(String(effType))}}</b></div>
            <div class=\"meta\">Evidenza: ${{esc(item.all_type_evidence || \"(nessuna)\")}}</div>
            <div class=\"meta small\">Cartelle previste con questo probe: <b>${{esc(String(item.folder_count_predicted || 0))}}</b></div>

            <div class=\"row\" style=\"margin-top:8px;\">
              <label class=\"small\">Correzione tipo</label>
              <select class=\"select corr-type\" data-probe-id=\"${{esc(item.probe_id)}}\">${{correctionOptions}}</select>
              <label class=\"small\">Nota</label>
              <input class=\"input corr-note\" data-probe-id=\"${{esc(item.probe_id)}}\" placeholder=\"Motivo correzione...\" />
            </div>

            <div class=\"folders\">${{foldersHtml || \"<div class='small'>Nessuna cartella preview disponibile per questo probe.</div>\"}}</div>
          </div>
        `;
      }}).join(\"\");

      for (const sel of grid.querySelectorAll(\".corr-type\")) {{
        const pid = sel.getAttribute(\"data-probe-id\") || \"\";
        const corr = currentCorr(pid);
        sel.value = corr.corrected_type_id || \"__AUTO__\";
        sel.addEventListener(\"change\", () => {{
          const c = currentCorr(pid);
          c.corrected_type_id = sel.value;
          corrections[pid] = c;
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          setCorrectionCount();
          render();
        }});
      }}

      for (const inp of grid.querySelectorAll(\".corr-note\")) {{
        const pid = inp.getAttribute(\"data-probe-id\") || \"\";
        const corr = currentCorr(pid);
        inp.value = corr.note || \"\";
        inp.addEventListener(\"input\", () => {{
          const c = currentCorr(pid);
          c.note = inp.value || \"\";
          corrections[pid] = c;
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          setCorrectionCount();
        }});
      }}

      for (const btn of grid.querySelectorAll(\".open-folder-viewer\")) {{
        btn.addEventListener(\"click\", () => {{
          const itemIdx = intAttr(btn, \"data-item-idx\");
          const folderIdx = intAttr(btn, \"data-folder-idx\");
          openViewer(itemIdx, folderIdx, 0);
        }});
      }}

      for (const btn of grid.querySelectorAll(\".open-image-viewer\")) {{
        btn.addEventListener(\"click\", () => {{
          const itemIdx = intAttr(btn, \"data-item-idx\");
          const folderIdx = intAttr(btn, \"data-folder-idx\");
          const imagePath = String(btn.getAttribute(\"data-image-path\") || \"\");
          const folder = getFolder(itemIdx, folderIdx);
          if (!folder) return;
          const allImages = Array.isArray(folder.all_images) && folder.all_images.length
            ? folder.all_images
            : (Array.isArray(folder.images) ? folder.images : []);
          let startIdx = allImages.findIndex(x => String(x || \"\") === imagePath);
          if (startIdx < 0) startIdx = 0;
          openViewer(itemIdx, folderIdx, startIdx);
        }});
      }}
    }}

    function toCsvCell(value) {{
      const s = String(value ?? \"\");
      if (s.includes(\",\") || s.includes('"') || s.includes("\\n")) return `"${{s.replaceAll('"', '""')}}"`;
      return s;
    }}

    function exportCorrections() {{
      const header = [
        \"probe_id\",\"probe_name\",\"type_status\",\"auto_primary_type_id\",\"auto_primary_type_description\",\"all_type_evidence\",\"corrected_type_id\",\"corrected_type_description\",\"note\",\"exported_at\"
      ];
      const typeMap = Object.fromEntries(TYPE_OPTIONS.map(t => [String(t.id), t.description]));
      const rows = [];
      const now = new Date().toISOString();
      for (const item of DATA) {{
        const c = currentCorr(item.probe_id);
        const t = String(c.corrected_type_id || \"__AUTO__\");
        const note = String(c.note || \"\").trim();
        if (t === \"__AUTO__\" && !note) continue;
        const tdesc = t === \"__AUTO__\" ? \"\" : (t === \"UNKNOWN\" ? \"UNKNOWN\" : (typeMap[t] || \"\"));
        rows.push([
          item.probe_id || \"\",
          item.probe_name || \"\",
          item.type_status || \"\",
          item.primary_probe_type_id || \"\",
          item.primary_probe_type_description || \"\",
          item.all_type_evidence || \"\",
          t,
          tdesc,
          note,
          now,
        ]);
      }}
      const lines = [header, ...rows].map(r => r.map(toCsvCell).join(\",\"));
      const csv = lines.join("\\n");
      const blob = new Blob([csv], {{ type: \"text/csv;charset=utf-8;\" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement(\"a\");
      a.href = url;
      a.download = \"probe_model_type_corrections.csv\";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    typeLegendBody.innerHTML = TYPE_OPTIONS.map(t => `
      <tr><td>${{esc(String(t.id))}}</td><td>${{esc(t.description || \"\")}}</td></tr>
    `).join(\"\");

    viewerCloseTop.addEventListener(\"click\", closeViewer);
    viewerPrev.addEventListener(\"click\", () => stepViewer(-1));
    viewerNext.addEventListener(\"click\", () => stepViewer(1));
    viewer.addEventListener(\"click\", (ev) => {{
      if (ev.target === viewer) closeViewer();
    }});
    viewer.addEventListener(\"wheel\", (ev) => {{
      if (!viewerState.open) return;
      ev.preventDefault();
      stepViewer(ev.deltaY > 0 ? 1 : -1);
    }}, {{ passive: false }});
    document.addEventListener(\"keydown\", (ev) => {{
      if (!viewerState.open) return;
      if (ev.key === \"Escape\") {{
        closeViewer();
      }} else if (ev.key === \"ArrowLeft\") {{
        stepViewer(-1);
      }} else if (ev.key === \"ArrowRight\") {{
        stepViewer(1);
      }}
    }});

    setTypeFilterOptions();
    setCorrectionCount();
    render();

    statusFilter.addEventListener(\"change\", render);
    typeFilter.addEventListener(\"change\", render);
    searchFilter.addEventListener(\"input\", render);
    exportBtn.addEventListener(\"click\", exportCorrections);
    resetBtn.addEventListener(\"click\", () => {{
      corrections = {{}};
      localStorage.removeItem(CORR_KEY);
      setCorrectionCount();
      render();
    }});
  </script>
</body>
</html>
"""


def _build_probe_type_map_html(
    title: str,
    probe_summary: List[Dict[str, str]],
    probe_type_evidence: Dict[str, List[Dict[str, str]]],
    probe_type_options: List[Dict[str, str]],
    source_notes: List[str],
) -> str:
    compact_rows: List[Dict[str, str]] = []
    for row in probe_summary:
        probe_id = str(row.get("probe_id") or "")
        probe_name = str(row.get("probe_name") or "")
        status = str(row.get("type_status") or "UNKNOWN")
        evidence_rows = probe_type_evidence.get(probe_id, [])

        if evidence_rows:
            for ev in evidence_rows:
                type_id = str(ev.get("probe_type_id") or "UNKNOWN")
                row_key = f"{probe_id}|{type_id}"
                compact_rows.append(
                    {
                        "row_key": row_key,
                        "probe_id": probe_id,
                        "probe_name": probe_name or str(ev.get("probe_name") or ""),
                        "type_status": status,
                        "type_id": type_id,
                        "type_description": str(ev.get("probe_type_description") or ""),
                        "samples": str(ev.get("samples") or ""),
                        "share_within_probe": str(ev.get("share_within_probe") or ""),
                        "all_type_evidence": str(row.get("all_type_evidence") or ""),
                    }
                )
        else:
            row_key = f"{probe_id}|UNKNOWN"
            compact_rows.append(
                {
                    "row_key": row_key,
                    "probe_id": probe_id,
                    "probe_name": probe_name,
                    "type_status": status,
                    "type_id": "UNKNOWN",
                    "type_description": str(row.get("primary_probe_type_description") or ""),
                    "samples": "",
                    "share_within_probe": "",
                    "all_type_evidence": str(row.get("all_type_evidence") or ""),
                }
            )

    payload = json.dumps(compact_rows, ensure_ascii=False)
    type_payload = json.dumps(probe_type_options, ensure_ascii=False)
    notes = " | ".join(source_notes)
    style = """
:root {
  --bg: #f8f9fd;
  --panel: #ffffff;
  --line: #d7dfef;
  --txt: #1a2236;
  --muted: #5a6783;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  color: var(--txt);
  background: radial-gradient(circle at 10% 0%, #e7eefb, var(--bg) 38%);
}
.wrap { max-width: 1800px; margin: 0 auto; padding: 14px; display: grid; gap: 10px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 auto; }
.small { color: var(--muted); font-size: 12px; }
.select, .input {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 8px;
  font-size: 13px;
}
.table-wrap { overflow: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.evidence { max-width: 680px; white-space: normal; overflow-wrap: anywhere; }
.note-cell { min-width: 260px; }
.note-input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}
"""

    return f"""<!doctype html>
<html lang=\"it\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>{style}</style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <div class=\"row\">
        <div class=\"grow\">
          <div><b>{html.escape(title)}</b></div>
          <div class=\"small\">{html.escape(notes)}</div>
        </div>
        <div class=\"small\">totali: <b id=\"totalCount\">0</b></div>
        <div class=\"small\">visibili: <b id=\"visibleCount\">0</b></div>
        <div class=\"small\">note: <b id=\"notesCount\">0</b></div>
      </div>
      <div class=\"row\" style=\"margin-top:8px;\">
        <label class=\"small\">Stato</label>
        <select id=\"statusFilter\" class=\"select\">
          <option value=\"ALL\">All</option>
          <option value=\"UNIVOCO\">UNIVOCO</option>
          <option value=\"AMBIGUO\">AMBIGUO</option>
          <option value=\"UNKNOWN\">UNKNOWN</option>
        </select>
        <label class=\"small\">Tipo</label>
        <select id=\"typeFilter\" class=\"select\"></select>
        <label class=\"small\">Cerca</label>
        <input id=\"searchFilter\" class=\"input\" placeholder=\"probe id/nome/tipo/evidenza...\" />
      </div>
    </div>

    <div class=\"panel table-wrap\">
      <table>
        <thead>
          <tr>
            <th>Probe ID</th>
            <th>Probe Name</th>
            <th>Status</th>
            <th>Type ID</th>
            <th>Type Description</th>
            <th>Samples</th>
            <th>Share</th>
            <th>Evidence</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody id=\"rowsBody\"></tbody>
      </table>
    </div>
  </div>

  <script>
    const DATA = {payload};
    const TYPE_OPTIONS = {type_payload};
    const NOTES_KEY = "probe_type_map_notes_v2";
    const totalCount = document.getElementById(\"totalCount\");
    const visibleCount = document.getElementById(\"visibleCount\");
    const notesCount = document.getElementById(\"notesCount\");
    const rowsBody = document.getElementById(\"rowsBody\");
    const statusFilter = document.getElementById(\"statusFilter\");
    const typeFilter = document.getElementById(\"typeFilter\");
    const searchFilter = document.getElementById(\"searchFilter\");

    let notesByRow = {{}};
    try {{ notesByRow = JSON.parse(localStorage.getItem(NOTES_KEY) || "{{}}"); }} catch (_) {{ notesByRow = {{}}; }}

    function esc(text) {{
      return String(text || \"\")
        .replaceAll(\"&\", \"&amp;\")
        .replaceAll(\"<\", \"&lt;\")
        .replaceAll(\">\", \"&gt;\")
        .replaceAll('"', \"&quot;\");
    }}

    function sortRows(rows) {{
      return [...rows].sort((a, b) => {{
        const ai = Number.parseInt(a.probe_id || \"\", 10);
        const bi = Number.parseInt(b.probe_id || \"\", 10);
        const aNum = Number.isFinite(ai) ? ai : 999999;
        const bNum = Number.isFinite(bi) ? bi : 999999;
        if (aNum !== bNum) return aNum - bNum;
        const as = Number.parseFloat(a.share_within_probe || "-1");
        const bs = Number.parseFloat(b.share_within_probe || "-1");
        if (Number.isFinite(as) && Number.isFinite(bs) && as !== bs) return bs - as;
        return String(a.probe_name || \"\").localeCompare(String(b.probe_name || \"\"));
      }});
    }}

    function currentNote(rowKey) {{
      return String(notesByRow[String(rowKey) || ""] || "");
    }}

    function setNotesCount() {{
      let n = 0;
      for (const [_, v] of Object.entries(notesByRow)) {{
        if (String(v || "").trim()) n += 1;
      }}
      notesCount.textContent = String(n);
    }}

    function setTypeFilterOptions() {{
      const values = new Set([\"UNKNOWN\"]);
      for (const it of DATA) values.add(it.type_id || \"UNKNOWN\");
      const sorted = [...values].sort((a, b) => a.localeCompare(b, undefined, {{ numeric: true }}));
      typeFilter.innerHTML = \"\";
      const optAll = document.createElement(\"option\");
      optAll.value = \"ALL\";
      optAll.textContent = \"All\";
      typeFilter.appendChild(optAll);
      for (const id of sorted) {{
        const opt = document.createElement(\"option\");
        opt.value = id;
        if (id === \"UNKNOWN\") {{
          opt.textContent = \"UNKNOWN\";
        }} else {{
          const meta = TYPE_OPTIONS.find(t => String(t.id) === String(id));
          opt.textContent = `${{id}} - ${{meta ? meta.description : \"\"}}`;
        }}
        typeFilter.appendChild(opt);
      }}
    }}

    function matches(row) {{
      const sf = statusFilter.value;
      if (sf !== \"ALL\" && row.type_status !== sf) return false;

      const tf = typeFilter.value;
      if (tf !== \"ALL\" && String(row.type_id || \"UNKNOWN\") !== String(tf)) return false;

      const q = String(searchFilter.value || \"\").trim().toLowerCase();
      if (!q) return true;
      const hay = `${{row.probe_id}} ${{row.probe_name}} ${{row.type_status}} ${{row.type_id}} ${{row.type_description}} ${{row.samples}} ${{row.share_within_probe}} ${{row.all_type_evidence}} ${{currentNote(row.row_key)}}`.toLowerCase();
      return hay.includes(q);
    }}

    function render() {{
      const rows = sortRows(DATA.filter(matches));
      totalCount.textContent = String(DATA.length);
      visibleCount.textContent = String(rows.length);
      rowsBody.innerHTML = rows.map(r => `
        <tr>
          <td class=\"mono\">${{esc(r.probe_id)}}</td>
          <td>${{esc(r.probe_name)}}</td>
          <td>${{esc(r.type_status)}}</td>
          <td class=\"mono\">${{esc(r.type_id || \"UNKNOWN\")}}</td>
          <td>${{esc(r.type_description || \"\")}}</td>
          <td class=\"mono\">${{esc(r.samples || \"\")}}</td>
          <td class=\"mono\">${{esc(r.share_within_probe || \"\")}}</td>
          <td class=\"evidence\">${{esc(r.all_type_evidence || \"\")}}</td>
          <td class=\"note-cell\"><input class=\"note-input\" data-row-key=\"${{esc(r.row_key)}}\" value=\"${{esc(currentNote(r.row_key))}}\" placeholder=\"Aggiungi nota...\" /></td>
        </tr>
      `).join(\"\");

      for (const inp of rowsBody.querySelectorAll(\".note-input\")) {{
        inp.addEventListener(\"input\", () => {{
          const rowKey = String(inp.getAttribute(\"data-row-key\") || \"\");
          notesByRow[rowKey] = inp.value || \"\";
          localStorage.setItem(NOTES_KEY, JSON.stringify(notesByRow));
          setNotesCount();
        }});
      }}
    }}

    setTypeFilterOptions();
    setNotesCount();
    render();
    statusFilter.addEventListener(\"change\", render);
    typeFilter.addEventListener(\"change\", render);
    searchFilter.addEventListener(\"input\", render);
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build HTML review for probe model -> type mapping.")
    p.add_argument(
        "--probe-summary-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_model_to_type_summary_2026-03-31.csv"),
        help="CSV con mappa modello probe -> tipo (UNIVOCO/AMBIGUO/UNKNOWN).",
    )
    p.add_argument(
        "--probe-type-list-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_type_list_from_encoding_struct_2026-03-30.csv"),
        help="CSV type_id -> descrizione.",
    )
    p.add_argument(
        "--probe-type-evidence-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv"),
        help="CSV con righe probe_id + probe_type_id + share.",
    )
    p.add_argument(
        "--folder-predictions-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv"),
        help="CSV folder-level con predicted_probe_id.",
    )
    p.add_argument(
        "--per-image-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv"),
        help="CSV per-image con folder_path/image_path.",
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/60_metadata/probe_model_type_review.html"),
        help="Output HTML.",
    )
    p.add_argument(
        "--output-map-html",
        type=Path,
        default=Path("artifacts/60_metadata/probe_model_type_map.html"),
        help="Output HTML tabellare probe -> tipo.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n1"),
        help="Root dataset dove risolvere i path legacy (/Dataset/...).",
    )
    p.add_argument("--max-folders-per-probe", type=int, default=4)
    p.add_argument("--max-images-per-folder", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()

    probe_summary_csv = args.probe_summary_csv.expanduser().resolve()
    probe_type_list_csv = args.probe_type_list_csv.expanduser().resolve()
    probe_type_evidence_csv = args.probe_type_evidence_csv.expanduser().resolve()
    folder_predictions_csv = args.folder_predictions_csv.expanduser().resolve()
    per_image_csv = args.per_image_csv.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser()
    output_html = args.output_html.expanduser().resolve()
    output_map_html = args.output_map_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_map_html.parent.mkdir(parents=True, exist_ok=True)

    probe_type_map = _load_probe_type_map(probe_type_list_csv)
    probe_type_options = [
        {"id": str(k), "description": v}
        for k, v in sorted(((int(k), v) for k, v in probe_type_map.items()), key=lambda x: x[0])
    ]

    summary_rows = _load_probe_summary(probe_summary_csv)
    probe_type_evidence = _load_probe_type_evidence(probe_type_evidence_csv)
    folder_meta = _load_folder_predictions(folder_predictions_csv, dataset_root=dataset_root)
    images_by_folder = _load_per_image(
        per_image_csv,
        max_images_per_folder=max(0, int(args.max_images_per_folder)),
        dataset_root=dataset_root,
    )
    probe_items = _build_probe_items(
        probe_summary=summary_rows,
        folder_meta=folder_meta,
        images_by_folder=images_by_folder,
        max_folders_per_probe=max(0, int(args.max_folders_per_probe)),
        max_images_per_folder=max(0, int(args.max_images_per_folder)),
        seed=int(args.seed),
    )

    storage_roots = sorted({_extract_storage_root(path) for path in folder_meta.keys() if path})
    missing_folders = sum(1 for path in folder_meta.keys() if path and not Path(path).exists())

    source_notes = [
        f"probe summary: {probe_summary_csv}",
        f"probe type evidence: {probe_type_evidence_csv}",
        f"folder predictions: {folder_predictions_csv}",
        f"per-image predictions: {per_image_csv}",
        f"dataset root (requested): {dataset_root}",
        f"storage roots (resolved): {', '.join(storage_roots) if storage_roots else 'n/a'}",
        f"folders missing on selected volume: {missing_folders}/{len(folder_meta)}",
        f"preview policy: max_folders_per_probe={int(args.max_folders_per_probe)}, max_images_per_folder={int(args.max_images_per_folder)}",
    ]
    doc = _build_html(
        title="Probe Model -> Type Review",
        probe_items=probe_items,
        probe_type_options=probe_type_options,
        source_notes=source_notes,
    )
    output_html.write_text(doc, encoding="utf-8")
    map_doc = _build_probe_type_map_html(
        title="Probe -> Type Map",
        probe_summary=summary_rows,
        probe_type_evidence=probe_type_evidence,
        probe_type_options=probe_type_options,
        source_notes=source_notes,
    )
    output_map_html.write_text(map_doc, encoding="utf-8")

    counts = defaultdict(int)
    for r in probe_items:
        counts[str(r.get("type_status") or "UNKNOWN")] += 1

    print(f"Models: {len(probe_items)}")
    print(f"Status counts: {dict(sorted(counts.items()))}")
    print(f"Output HTML: {output_html}")
    print(f"Output map HTML: {output_map_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
