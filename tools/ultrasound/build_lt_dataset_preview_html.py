#!/usr/bin/env python3
"""Build an interactive HTML preview for L/T data with switchable sources."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_text(value: str, max_len: int = 240) -> str:
    out = (value or "").strip()
    if len(out) <= max_len:
        return out
    return out[: max_len - 1] + "…"


def _resolve_legacy_path(path_str: str) -> str:
    raw = (path_str or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.exists():
        return p.as_posix()

    candidates: List[Path] = []

    marker = "/Users/camilla/Documents/Develop/ESIBuilder_AI/Dataset/"
    if marker in raw:
        candidates.append(Path(raw.replace(marker, "/Users/camilla/Documents/Develop/ESIBuilder_AI/data/Dataset/")))

    if "/Dataset/" in raw:
        suffix = raw.split("/Dataset/", 1)[1]
        candidates.append(PROJECT_ROOT / "data" / "Dataset" / suffix)

    for cand in candidates:
        if cand.exists():
            return cand.as_posix()
    return raw


def _path_prefix(path_text: str, depth: int = 2) -> str:
    parts = [p for p in (path_text or "").split("/") if p]
    if not parts:
        return ""
    return "/".join(parts[:depth])


def _infer_vendor_from_text(text: str) -> str:
    upper = (text or "").upper()
    rules = [
        ("ESAOTE", "Esaote"),
        ("BK", "BK"),
        ("HITACHI", "Hitachi"),
        ("ALOKA", "Hitachi"),
        ("GE", "GE"),
        ("LOGIQ", "GE"),
        ("MINDRAY", "Mindray"),
        ("KOELIS", "Koelis"),
        ("CANON", "Canon"),
        ("TOSHIBA", "Canon/Toshiba"),
        ("SIEMENS", "Siemens"),
        ("PHILIPS", "Philips"),
        ("BIOPSEE", "Biopsee"),
        ("ALPINION", "Alpinion"),
        ("SONOSTAR", "Sonostar"),
    ]
    for token, vendor in rules:
        if token in upper:
            return vendor
    return "UNKNOWN"


def _load_probe_names(probe_list_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not probe_list_csv.is_file():
        return out
    with probe_list_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = (row.get("probe_id") or "").strip()
            if not pid:
                continue
            out[pid] = _safe_text(row.get("probe_name") or "", max_len=120)
    return out


def _load_probe_type_descriptions(probe_type_list_csv: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not probe_type_list_csv.is_file():
        return out
    with probe_type_list_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                type_id = int(float((row.get("probe_type_id") or "").strip()))
            except ValueError:
                continue
            out[type_id] = _safe_text(row.get("probe_type_description") or "", max_len=120)
    return out


def _load_probe_evidence(
    probe_evidence_csv: Path,
    probe_type_descriptions: Dict[int, str],
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not probe_evidence_csv.is_file():
        return out
    by_probe: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"n_l": 0.0, "n_t": 0.0, "type_counts": defaultdict(float)}
    )
    with probe_evidence_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = (row.get("probe_id") or "").strip()
            if not pid:
                continue
            try:
                type_id = int(float((row.get("probe_type_id") or "").strip()))
            except ValueError:
                continue
            try:
                samples = float((row.get("samples") or "").strip())
            except ValueError:
                samples = 0.0

            type_counts = by_probe[pid]["type_counts"]
            assert isinstance(type_counts, defaultdict)
            type_counts[type_id] += samples

            if type_id == 3:
                by_probe[pid]["n_l"] = float(by_probe[pid]["n_l"]) + samples
            elif type_id == 4:
                by_probe[pid]["n_t"] = float(by_probe[pid]["n_t"]) + samples

    for pid, stats in by_probe.items():
        n_l = float(stats.get("n_l", 0.0))
        n_t = float(stats.get("n_t", 0.0))
        total = n_l + n_t
        p_l = 0.0 if total <= 0 else (n_l / total)
        p_t = 0.0 if total <= 0 else (n_t / total)

        type_counts_obj = stats.get("type_counts", defaultdict(float))
        assert isinstance(type_counts_obj, defaultdict)
        sorted_types = sorted(type_counts_obj.items(), key=lambda x: (-x[1], x[0]))
        if sorted_types:
            primary_type_id = int(sorted_types[0][0])
            primary_type_desc = probe_type_descriptions.get(primary_type_id, "UNKNOWN")
        else:
            primary_type_id = 0
            primary_type_desc = "UNKNOWN"

        type_ids_csv = ",".join(str(int(type_id)) for type_id, _ in sorted_types)
        parts: List[str] = []
        total_all_types = float(sum(v for _, v in sorted_types))
        for type_id, count in sorted_types:
            desc = probe_type_descriptions.get(int(type_id), f"type_{int(type_id)}")
            share = (count / total_all_types) if total_all_types > 0 else 0.0
            parts.append(f"{int(type_id)}:{desc} ({share:.2f})")

        out[pid] = {
            "p_l": f"{p_l:.6f}",
            "p_t": f"{p_t:.6f}",
            "probe_type_ids": type_ids_csv,
            "probe_type_primary_id": str(primary_type_id if primary_type_id > 0 else ""),
            "probe_type_primary_desc": _safe_text(primary_type_desc, max_len=120),
            "probe_type_summary": _safe_text(" | ".join(parts), max_len=220),
        }
    return out


def _prior_label_from_probe(p_l: float, p_t: float, min_confidence: float) -> str:
    if p_l >= min_confidence and p_l > p_t:
        return "L"
    if p_t >= min_confidence and p_t > p_l:
        return "T"
    if p_l > 0 or p_t > 0:
        return "MIXED"
    return "UNKNOWN"


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = (row.get("label_lt") or "").strip().upper()
            if label not in {"L", "T"}:
                continue
            image_path = (row.get("image_path") or "").strip()
            image_path = _resolve_legacy_path(image_path)
            if not image_path:
                continue
            group_id = _safe_text(row.get("group_id") or "", max_len=180)
            source_rel_path = _safe_text(row.get("source_rel_path") or "", max_len=220)
            prefix = _path_prefix(source_rel_path, depth=2)
            if prefix:
                group_evidence = f"manifest.group_id + source_rel_path prefix: {prefix}"
            else:
                group_evidence = f"manifest.group_id: {group_id or 'NA'}"
            rows.append(
                {
                    "source_id": "manifest",
                    "source_name": "Manifest label L/T",
                    "label": label,
                    "auto_label": label,
                    "split": (row.get("split") or "").strip().lower(),
                    "manufacturer": _safe_text(row.get("manufacturer") or "UNKNOWN", max_len=80),
                    "group_id": group_id,
                    "image_path": image_path,
                    "source_rel_path": source_rel_path,
                    "note": "label from manifest",
                    "group_evidence": _safe_text(group_evidence, max_len=240),
                    "probe_id": "",
                    "probe_name": "",
                    "probe_type_ids": "",
                    "probe_type_primary_id": "",
                    "probe_type_primary_desc": "",
                    "probe_type_summary": "",
                }
            )
    return rows


def _load_probe_rows(
    probe_folder_predictions: Path,
    probe_per_image_predictions: Path,
    probe_name_map: Dict[str, str],
    probe_evidence_map: Dict[str, Dict[str, str]],
    max_images_per_folder: int,
    prior_min_confidence: float,
) -> List[Dict[str, str]]:
    folder_meta: Dict[str, Dict[str, str]] = {}
    if not probe_folder_predictions.is_file():
        return []
    if not probe_per_image_predictions.is_file():
        return []

    with probe_folder_predictions.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder_path = (row.get("folder_path") or "").strip()
            folder_path = _resolve_legacy_path(folder_path)
            if not folder_path:
                continue
            folder_name = (row.get("folder_name") or "").strip() or Path(folder_path).name
            probe_id = (row.get("predicted_probe_id") or "").strip()
            probe_name = probe_name_map.get(probe_id, "")
            evidence = probe_evidence_map.get(
                probe_id,
                {
                    "p_l": "0.0",
                    "p_t": "0.0",
                    "probe_type_ids": "",
                    "probe_type_primary_id": "",
                    "probe_type_primary_desc": "",
                    "probe_type_summary": "",
                },
            )
            p_l = float(evidence.get("p_l", "0.0"))
            p_t = float(evidence.get("p_t", "0.0"))
            prior_label = _prior_label_from_probe(
                p_l=p_l,
                p_t=p_t,
                min_confidence=prior_min_confidence,
            )
            note = (
                f"probe={probe_id} ({probe_name or 'UNKNOWN'}) | "
                f"prior L={p_l:.2f} T={p_t:.2f} | "
                f"types={evidence.get('probe_type_summary', '') or 'NA'} | "
                f"folder_top1={row.get('folder_top1_prob') or 'NA'} | "
                f"vote={row.get('folder_vote_ratio') or 'NA'}"
            )
            folder_meta[folder_path] = {
                "group_id": _safe_text(folder_name, max_len=180),
                "manufacturer": _infer_vendor_from_text(folder_name),
                "label": prior_label,
                "note": _safe_text(note, max_len=240),
                "group_evidence": _safe_text(
                    f"from probe CSV folder_path: {folder_path}",
                    max_len=240,
                ),
                "probe_id": probe_id,
                "probe_name": _safe_text(probe_name, max_len=120),
                "probe_type_ids": evidence.get("probe_type_ids", ""),
                "probe_type_primary_id": evidence.get("probe_type_primary_id", ""),
                "probe_type_primary_desc": evidence.get("probe_type_primary_desc", ""),
                "probe_type_summary": evidence.get("probe_type_summary", ""),
            }

    images_by_folder: Dict[str, List[str]] = defaultdict(list)
    with probe_per_image_predictions.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder_path = (row.get("folder_path") or "").strip()
            folder_path = _resolve_legacy_path(folder_path)
            image_path = _resolve_legacy_path((row.get("image_path") or "").strip())
            if not folder_path or not image_path:
                continue
            bucket = images_by_folder[folder_path]
            if max_images_per_folder > 0 and len(bucket) >= max_images_per_folder:
                continue
            bucket.append(image_path)

    out_rows: List[Dict[str, str]] = []
    for folder_path in sorted(images_by_folder.keys()):
        meta = folder_meta.get(folder_path)
        if meta is None:
            folder_name = Path(folder_path).name
            meta = {
                "group_id": _safe_text(folder_name, max_len=180),
                "manufacturer": _infer_vendor_from_text(folder_name),
                "label": "UNKNOWN",
                "note": "probe metadata not found",
                "group_evidence": _safe_text(
                    f"from probe CSV folder_path: {folder_path}",
                    max_len=240,
                ),
                "probe_id": "",
                "probe_name": "",
                "probe_type_ids": "",
                "probe_type_primary_id": "",
                "probe_type_primary_desc": "",
                "probe_type_summary": "",
            }
        for image_path in images_by_folder[folder_path]:
            out_rows.append(
                {
                    "source_id": "probe_prior",
                    "source_name": "Probe model -> L/T prior",
                    "label": meta["label"],
                    "auto_label": meta["label"],
                    "split": "",
                    "manufacturer": meta["manufacturer"],
                    "group_id": meta["group_id"],
                    "image_path": image_path,
                    "source_rel_path": _safe_text(Path(image_path).name, max_len=220),
                    "note": meta["note"],
                    "group_evidence": meta["group_evidence"],
                    "probe_id": meta["probe_id"],
                    "probe_name": meta["probe_name"],
                    "probe_type_ids": meta["probe_type_ids"],
                    "probe_type_primary_id": meta["probe_type_primary_id"],
                    "probe_type_primary_desc": meta["probe_type_primary_desc"],
                    "probe_type_summary": meta["probe_type_summary"],
                }
            )
    return out_rows


def _stratified_sample(rows: List[Dict[str, str]], max_rows: int, seed: int) -> List[Dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("label") or "UNKNOWN"].append(row)

    labels = sorted(by_label.keys())
    for label in labels:
        rng.shuffle(by_label[label])

    sampled: List[Dict[str, str]] = []
    base = max_rows // max(1, len(labels))
    for label in labels:
        sampled.extend(by_label[label][:base])
        by_label[label] = by_label[label][base:]

    remain_slots = max_rows - len(sampled)
    leftovers: List[Dict[str, str]] = []
    for label in labels:
        leftovers.extend(by_label[label])
    rng.shuffle(leftovers)
    sampled.extend(leftovers[:remain_slots])
    rng.shuffle(sampled)
    return sampled


def _build_html(
    rows: List[Dict[str, str]],
    source_stats: List[Dict[str, str]],
    sampling_notes: List[str],
    probe_types: List[Dict[str, str]],
) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    source_payload = json.dumps(source_stats, ensure_ascii=False)
    probe_types_payload = json.dumps(probe_types, ensure_ascii=False)
    total = len(rows)
    notes_block = " | ".join(sampling_notes) if sampling_notes else "all rows"

    style = """
:root {
  --bg: #f7f9fc;
  --panel: #ffffff;
  --line: #d8deec;
  --txt: #172033;
  --muted: #5e6b85;
  --pill-l-bg: #d7f4e5;
  --pill-l-txt: #0e6a3a;
  --pill-t-bg: #ffe5c7;
  --pill-t-txt: #8a4b00;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  color: var(--txt);
  background: radial-gradient(circle at 6% 0%, #e8eefb, var(--bg) 40%);
}
.wrap {
  max-width: 1840px;
  margin: 0 auto;
  padding: 14px;
  display: grid;
  gap: 10px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.grow { flex: 1 1 auto; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.small { color: var(--muted); font-size: 12px; }
.select, .input {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 8px;
  font-size: 13px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 10px;
}
.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
}
.thumb {
  width: 100%;
  height: 146px;
  object-fit: contain;
  background: #000;
  border-radius: 7px;
}
.meta { margin-top: 6px; font-size: 12px; line-height: 1.35; }
.path {
  color: #33435f;
  font-size: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pill {
  display: inline-block;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 9px;
}
.pill-l { background: var(--pill-l-bg); color: var(--pill-l-txt); }
.pill-t { background: var(--pill-t-bg); color: var(--pill-t-txt); }
.pill-u { background: #e8ebf2; color: #495468; }
.button {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 9px;
  font-size: 12px;
  cursor: pointer;
}
.button:hover { background: #f2f5fb; }
.table-wrap { overflow-x: auto; margin-top: 8px; }
table.legend {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}
table.legend th, table.legend td {
  border: 1px solid var(--line);
  padding: 6px 8px;
  text-align: left;
}
.card-corrected {
  border-color: #3a74d4;
  box-shadow: 0 0 0 1px #3a74d4 inset;
}
.card-correction {
  margin-top: 7px;
  display: flex;
  align-items: center;
  gap: 6px;
}
"""

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LT Dataset Preview</title>
  <style>{style}</style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="row">
        <div class="grow">
          <div><b>Preview dataset L/T multi-source</b></div>
          <div class="small">sampling: {html.escape(notes_block)}</div>
        </div>
        <div class="small">rows loaded: <b id="rowsLoaded">{total}</b></div>
        <div class="small">rows source: <b id="rowsSource">0</b></div>
        <div class="small">rows visible: <b id="rowsVisible">0</b></div>
        <div class="small">rows corrette: <b id="rowsCorrected">0</b></div>
      </div>
      <div class="row" style="margin-top:8px;">
        <label class="small">Source</label>
        <select id="sourceFilter" class="select"></select>
        <label class="small">Label</label>
        <select id="labelFilter" class="select">
          <option value="ALL">All</option>
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
        <label class="small">Probe Type</label>
        <select id="probeTypeFilter" class="select">
          <option value="ALL">All</option>
        </select>
        <label class="small">Cerca</label>
        <input id="searchFilter" class="input" placeholder="group/path/vendor..." />
        <button id="exportCorrectionsBtn" class="button" type="button">Export correzioni CSV</button>
        <button id="resetCorrectionsBtn" class="button" type="button">Reset correzioni</button>
      </div>
      <div class="table-wrap">
        <table class="legend" id="probeTypeLegendTable">
          <thead>
            <tr><th>Probe Type ID</th><th>Descrizione</th></tr>
          </thead>
          <tbody id="probeTypeLegendBody"></tbody>
        </table>
      </div>
    </div>
    <div id="grid" class="grid"></div>
  </div>
  <script>
    const DATA = {payload};
    const SOURCES = {source_payload};
    const PROBE_TYPES = {probe_types_payload};
    const grid = document.getElementById("grid");
    const rowsVisible = document.getElementById("rowsVisible");
    const rowsSource = document.getElementById("rowsSource");
    const rowsCorrected = document.getElementById("rowsCorrected");
    const sourceFilter = document.getElementById("sourceFilter");
    const labelFilter = document.getElementById("labelFilter");
    const splitFilter = document.getElementById("splitFilter");
    const vendorFilter = document.getElementById("vendorFilter");
    const probeTypeFilter = document.getElementById("probeTypeFilter");
    const searchFilter = document.getElementById("searchFilter");
    const exportCorrectionsBtn = document.getElementById("exportCorrectionsBtn");
    const resetCorrectionsBtn = document.getElementById("resetCorrectionsBtn");
    const probeTypeLegendBody = document.getElementById("probeTypeLegendBody");
    const CORR_KEY = "lt_preview_corrections_v1";

    let corrections = {{}};
    try {{
      corrections = JSON.parse(localStorage.getItem(CORR_KEY) || "{{}}");
    }} catch (_) {{
      corrections = {{}};
    }}

    function rowKey(row) {{
      return `${{row.source_id}}|${{row.image_path}}`;
    }}

    function effectiveLabel(row) {{
      const key = rowKey(row);
      return corrections[key] || row.label || "UNKNOWN";
    }}

    function correctionCount() {{
      rowsCorrected.textContent = String(Object.keys(corrections).length);
    }}

    function setOptions(selectEl, values) {{
      const current = selectEl.value || "ALL";
      selectEl.innerHTML = "";
      const first = document.createElement("option");
      first.value = "ALL";
      first.textContent = "All";
      selectEl.appendChild(first);
      for (const v of values) {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        selectEl.appendChild(opt);
      }}
      if ([...selectEl.options].some(o => o.value === current)) {{
        selectEl.value = current;
      }} else {{
        selectEl.value = "ALL";
      }}
    }}

    function rowsInSelectedSource() {{
      const sf = sourceFilter.value || "ALL";
      if (sf === "ALL") return DATA;
      return DATA.filter(r => r.source_id === sf);
    }}

    function rowHasProbeType(row, selectedTypeId) {{
      if (!selectedTypeId || selectedTypeId === "ALL") return true;
      const ids = String(row.probe_type_ids || "")
        .split(",")
        .map(s => s.trim())
        .filter(Boolean);
      return ids.includes(selectedTypeId);
    }}

    function refreshDependentFilters() {{
      const sourceRows = rowsInSelectedSource();
      rowsSource.textContent = String(sourceRows.length);

      const labels = Array.from(new Set(sourceRows.map(r => String(effectiveLabel(r) || "UNKNOWN")))).sort();
      setOptions(labelFilter, labels);

      const splits = Array.from(
        new Set(sourceRows.map(r => String(r.split || "").trim()).filter(s => s.length > 0))
      ).sort();
      setOptions(splitFilter, splits);

      const vendors = Array.from(new Set(sourceRows.map(r => String(r.manufacturer || "UNKNOWN")))).sort();
      setOptions(vendorFilter, vendors);

      const typeIdsInSource = new Set();
      for (const r of sourceRows) {{
        for (const token of String(r.probe_type_ids || "").split(",")) {{
          const t = token.trim();
          if (t) typeIdsInSource.add(t);
        }}
      }}
      const currentType = probeTypeFilter.value || "ALL";
      probeTypeFilter.innerHTML = "";
      const firstType = document.createElement("option");
      firstType.value = "ALL";
      firstType.textContent = "All";
      probeTypeFilter.appendChild(firstType);
      for (const pt of PROBE_TYPES.filter(pt => typeIdsInSource.has(String(pt.id)))) {{
        const opt = document.createElement("option");
        opt.value = String(pt.id);
        opt.textContent = `${{pt.id}} - ${{pt.description}}`;
        probeTypeFilter.appendChild(opt);
      }}
      if ([...probeTypeFilter.options].some(o => o.value === currentType)) {{
        probeTypeFilter.value = currentType;
      }} else {{
        probeTypeFilter.value = "ALL";
      }}
    }}

    const srcAll = document.createElement("option");
    srcAll.value = "ALL";
    srcAll.textContent = "All sources";
    sourceFilter.appendChild(srcAll);

    for (const src of SOURCES) {{
      const opt = document.createElement("option");
      opt.value = src.id;
      opt.textContent = `${{src.name}} (${{src.rows}})`;
      sourceFilter.appendChild(opt);
    }}

    probeTypeLegendBody.innerHTML = PROBE_TYPES.map(pt => `
      <tr>
        <td>${{esc(String(pt.id))}}</td>
        <td>${{esc(String(pt.description || ""))}}</td>
      </tr>
    `).join("");

    function esc(text) {{
      return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function matchRow(row) {{
      const src = sourceFilter.value;
      if (src !== "ALL" && row.source_id !== src) return false;
      const lf = labelFilter.value;
      if (lf !== "ALL" && effectiveLabel(row) !== lf) return false;
      const sf = splitFilter.value;
      if (sf !== "ALL" && row.split !== sf) return false;
      const vf = vendorFilter.value;
      if (vf !== "ALL" && row.manufacturer !== vf) return false;
      const ptf = probeTypeFilter.value;
      if (!rowHasProbeType(row, ptf)) return false;

      const q = String(searchFilter.value || "").trim().toLowerCase();
      if (!q) return true;
      const hay = `${{row.group_id}} ${{row.group_evidence || ""}} ${{row.source_rel_path}} ${{row.manufacturer}} ${{row.source_name}} ${{row.note || ""}} ${{row.probe_id || ""}} ${{row.probe_name || ""}} ${{row.probe_type_summary || ""}}`.toLowerCase();
      return hay.includes(q);
    }}

    function pillClass(label) {{
      if (label === "L") return "pill pill-l";
      if (label === "T") return "pill pill-t";
      return "pill pill-u";
    }}

    function render() {{
      const rows = DATA.filter(matchRow);
      rowsVisible.textContent = String(rows.length);

      const htmlCards = rows.map(row => {{
        const path = String(row.image_path || "");
        const imgSrc = path.startsWith("file://") ? path : `file://${{path}}`;
        const sourceLabel = row.source_name || row.source_id || "";
        const autoLabel = row.label || "UNKNOWN";
        const effLabel = effectiveLabel(row);
        const isCorrected = effLabel !== autoLabel;
        const key = rowKey(row);
        const probeTypeText = row.probe_type_summary || row.probe_type_primary_desc || "-";
        return `
          <div class="card ${{isCorrected ? "card-corrected" : ""}}">
            <img class="thumb" loading="lazy" src="${{esc(imgSrc)}}" alt="sample" />
            <div class="meta"><span class="${{pillClass(effLabel)}}">${{esc(effLabel)}}</span> split=${{esc(row.split || "-")}} | vendor=${{esc(row.manufacturer)}} | source=${{esc(sourceLabel)}}</div>
            <div class="meta">group=${{esc(row.group_id)}}</div>
            <div class="meta small">group evidence=${{esc(row.group_evidence || "-")}}</div>
            <div class="meta small">probe=${{esc(row.probe_id || "-")}} (${{esc(row.probe_name || "-")}})</div>
            <div class="meta small">probe_type=${{esc(probeTypeText)}}</div>
            <div class="meta small">${{esc(row.note || "")}}</div>
            <div class="path" title="${{esc(path)}}">${{esc(path)}}</div>
            <div class="card-correction">
              <span class="small">Correzione</span>
              <select class="select correction-select" data-row-key="${{esc(key)}}" data-auto-label="${{esc(autoLabel)}}">
                <option value="__AUTO__">AUTO (${{esc(autoLabel)}})</option>
                <option value="L">L</option>
                <option value="T">T</option>
                <option value="MIXED">MIXED</option>
                <option value="UNKNOWN">UNKNOWN</option>
                <option value="SKIP">SKIP</option>
              </select>
            </div>
          </div>
        `;
      }}).join("");
      grid.innerHTML = htmlCards;

      const selects = grid.querySelectorAll(".correction-select");
      for (const sel of selects) {{
        const key = sel.getAttribute("data-row-key") || "";
        const autoLabel = sel.getAttribute("data-auto-label") || "UNKNOWN";
        const corr = corrections[key];
        sel.value = corr ? corr : "__AUTO__";
        sel.addEventListener("change", () => {{
          const v = sel.value;
          if (!key) return;
          if (v === "__AUTO__" || v === autoLabel) {{
            delete corrections[key];
          }} else {{
            corrections[key] = v;
          }}
          localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
          correctionCount();
          refreshDependentFilters();
          render();
        }});
      }}
      correctionCount();
    }}

    sourceFilter.addEventListener("change", () => {{
      refreshDependentFilters();
      render();
    }});
    for (const el of [labelFilter, splitFilter, vendorFilter, probeTypeFilter]) {{
      el.addEventListener("change", render);
    }}
    searchFilter.addEventListener("input", render);

    function toCsvCell(value) {{
      const s = String(value ?? "");
      if (s.includes(",") || s.includes('"') || s.includes("\\n")) {{
        return `"${{s.replaceAll('"', '""')}}"`;
      }}
      return s;
    }}

    function exportCorrectionsCsv() {{
      const rows = [];
      for (const row of DATA) {{
        const key = rowKey(row);
        const corrected = corrections[key];
        if (!corrected) continue;
        rows.push([
          key,
          row.source_id || "",
          row.image_path || "",
          row.group_id || "",
          row.group_evidence || "",
          row.manufacturer || "",
          row.probe_id || "",
          row.probe_name || "",
          row.probe_type_primary_id || "",
          row.probe_type_primary_desc || "",
          row.label || "",
          corrected,
          row.note || "",
          new Date().toISOString(),
        ]);
      }}
      const header = [
        "row_key",
        "source_id",
        "image_path",
        "group_id",
        "group_evidence",
        "manufacturer",
        "probe_id",
        "probe_name",
        "probe_type_primary_id",
        "probe_type_primary_desc",
        "original_label",
        "corrected_label",
        "note",
        "exported_at",
      ];
      const lines = [header, ...rows].map(r => r.map(toCsvCell).join(","));
      const csv = lines.join("\\n");
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "lt_probe_corrections.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    exportCorrectionsBtn.addEventListener("click", exportCorrectionsCsv);
    resetCorrectionsBtn.addEventListener("click", () => {{
      corrections = {{}};
      localStorage.removeItem(CORR_KEY);
      correctionCount();
      refreshDependentFilters();
      render();
    }});

    correctionCount();
    refreshDependentFilters();
    render();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build interactive preview HTML for L/T dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv"),
        help="Manifest CSV con colonne image_path,label_lt,split,manufacturer.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/preview/preview_gallery.html"),
        help="Output HTML.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0=tutte le righe sorgente manifest; >0 campione stratificato.",
    )
    parser.add_argument(
        "--include-probe-source",
        action="store_true",
        help="Aggiunge la sorgente probe-based (rete probe + prior L/T).",
    )
    parser.add_argument(
        "--probe-folder-predictions",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv"),
        help="CSV folder-level con predicted_probe_id.",
    )
    parser.add_argument(
        "--probe-per-image-predictions",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv"),
        help="CSV per-image con image_path e folder_path.",
    )
    parser.add_argument(
        "--probe-list-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_list_from_encoding_struct_2026-03-30.csv"),
        help="CSV probe_id -> probe_name.",
    )
    parser.add_argument(
        "--probe-evidence-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv"),
        help="CSV storico tipo sonda (probe_type_id 3/4) per prior L/T.",
    )
    parser.add_argument(
        "--probe-type-list-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_type_list_from_encoding_struct_2026-03-30.csv"),
        help="CSV legenda type_id -> descrizione.",
    )
    parser.add_argument(
        "--probe-max-images-per-folder",
        type=int,
        default=6,
        help="Numero massimo immagini per folder dalla sorgente probe-based (0=tutte).",
    )
    parser.add_argument(
        "--probe-max-rows",
        type=int,
        default=0,
        help="0=tutte le righe sorgente probe; >0 campione stratificato.",
    )
    parser.add_argument(
        "--probe-prior-min-confidence",
        type=float,
        default=0.60,
        help="Soglia per etichetta L/T da prior probe (altrimenti MIXED).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    source_stats: List[Dict[str, str]] = []
    sampling_notes: List[str] = []
    probe_types_for_html: List[Dict[str, str]] = []

    manifest_path = args.manifest.expanduser().resolve()
    if manifest_path.is_file():
        manifest_rows = _load_manifest_rows(manifest_path)
        manifest_rows = _stratified_sample(
            manifest_rows,
            max_rows=int(args.max_rows),
            seed=int(args.seed),
        )
        rows.extend(manifest_rows)
        source_stats.append(
            {"id": "manifest", "name": "Manifest label L/T", "rows": len(manifest_rows)}
        )
        if int(args.max_rows) > 0:
            sampling_notes.append(f"manifest max_rows={int(args.max_rows)}")
    else:
        print(f"[warn] Manifest non trovato: {manifest_path}", flush=True)

    if args.include_probe_source:
        probe_folder_predictions = args.probe_folder_predictions.expanduser().resolve()
        probe_per_image_predictions = args.probe_per_image_predictions.expanduser().resolve()
        probe_list_csv = args.probe_list_csv.expanduser().resolve()
        probe_evidence_csv = args.probe_evidence_csv.expanduser().resolve()
        probe_type_list_csv = args.probe_type_list_csv.expanduser().resolve()

        probe_names = _load_probe_names(probe_list_csv)
        probe_type_descriptions = _load_probe_type_descriptions(probe_type_list_csv)
        probe_evidence = _load_probe_evidence(
            probe_evidence_csv=probe_evidence_csv,
            probe_type_descriptions=probe_type_descriptions,
        )
        probe_types_for_html = [
            {"id": str(type_id), "description": desc}
            for type_id, desc in sorted(probe_type_descriptions.items())
        ]
        probe_rows = _load_probe_rows(
            probe_folder_predictions=probe_folder_predictions,
            probe_per_image_predictions=probe_per_image_predictions,
            probe_name_map=probe_names,
            probe_evidence_map=probe_evidence,
            max_images_per_folder=int(args.probe_max_images_per_folder),
            prior_min_confidence=float(args.probe_prior_min_confidence),
        )
        probe_rows = _stratified_sample(
            probe_rows,
            max_rows=int(args.probe_max_rows),
            seed=int(args.seed) + 1,
        )
        rows.extend(probe_rows)
        source_stats.append(
            {"id": "probe_prior", "name": "Probe model -> L/T prior", "rows": len(probe_rows)}
        )
        if int(args.probe_max_rows) > 0:
            sampling_notes.append(f"probe_prior max_rows={int(args.probe_max_rows)}")

    if not rows:
        raise RuntimeError("Nessuna riga disponibile da visualizzare.")

    doc = _build_html(
        rows=rows,
        source_stats=source_stats,
        sampling_notes=sampling_notes,
        probe_types=probe_types_for_html,
    )
    output_html.write_text(doc, encoding="utf-8")

    print(f"Sources: {source_stats}", flush=True)
    print(f"Rows in HTML: {len(rows)}", flush=True)
    print(f"Output HTML: {output_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
