#!/usr/bin/env python3
"""Prepare a binary orientation dataset (su vs giu) from image file names.

Rules implemented (case-insensitive, filename based):
- `su`: filename contains `nf`, `noflip`, or `lr`
- `giu`: filename contains `ud` or `lrud`

Priority rule:
- if both conditions appear, `giu` wins (e.g. `lrud`).

Inputs are scanned recursively from the configured dataset root.
The output dataset contains PNG copies, train/val/test splits, manifest/summary files,
and a local HTML checker.
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import os
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LABEL_NOFLIP = "su"
LABEL_UPSIDE_DOWN = "giu"
SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_RE = re.compile(r".+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE)
SKIP_DIR_NAMES = {
    ".spotlight-v100",
    "system volume information",
    "$recycle.bin",
    ".trashes",
    ".fseventsd",
}


@dataclass(frozen=True)
class SetupRecord:
    setup_folder: Path
    image_paths: Tuple[Path, ...]
    label: str


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    setup_folder: Path
    source_image_path: Path
    label: str


def _slugify(text: str, max_len: int = 64) -> str:
    ascii_txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_txt).strip("_").lower()
    if not slug:
        slug = "sample"
    return slug[:max_len]


def _label_from_filename(image_path: Path) -> Optional[str]:
    """Classify from filename using case-insensitive substring rules."""
    name_l = image_path.name.lower()

    # Vertical/down class wins when both patterns appear (e.g. "lrud").
    if "lrud" in name_l or "ud" in name_l:
        return LABEL_UPSIDE_DOWN
    if "noflip" in name_l or "nf" in name_l or "lr" in name_l:
        return LABEL_NOFLIP
    return None


def _collect_setups(dataset_root: Path) -> Tuple[List[SetupRecord], Dict[str, int], List[str]]:
    grouped: Dict[Tuple[Path, str], List[Path]] = {}
    warnings: List[str] = []
    stats = {
        "folders_scanned": 0,
        "images_scanned": 0,
        "images_matched_rules": 0,
        "images_unmatched_rules": 0,
    }

    for root_dir, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIR_NAMES]
        folder = Path(root_dir).resolve()
        stats["folders_scanned"] += 1

        for file_name in filenames:
            if not IMAGE_RE.match(file_name):
                continue
            stats["images_scanned"] += 1

            src = (folder / file_name).resolve()
            label = _label_from_filename(src)
            if label is None:
                stats["images_unmatched_rules"] += 1
                continue

            stats["images_matched_rules"] += 1
            grouped.setdefault((folder, label), []).append(src)

    records: List[SetupRecord] = []
    for (folder, label), paths in sorted(grouped.items(), key=lambda x: (x[0][0].as_posix(), x[0][1])):
        if not paths:
            continue
        records.append(
            SetupRecord(
                setup_folder=folder,
                image_paths=tuple(sorted(paths)),
                label=label,
            )
        )

    stats["folders_with_images"] = len({rec.setup_folder for rec in records})
    return records, stats, warnings


def _expand_samples(setups: Sequence[SetupRecord]) -> Dict[str, List[SampleRecord]]:
    out: Dict[str, List[SampleRecord]] = {
        LABEL_NOFLIP: [],
        LABEL_UPSIDE_DOWN: [],
    }
    running_idx = 0
    for setup in setups:
        for src in setup.image_paths:
            running_idx += 1
            out[setup.label].append(
                SampleRecord(
                    sample_id=f"S{running_idx:07d}",
                    setup_folder=setup.setup_folder,
                    source_image_path=src,
                    label=setup.label,
                )
            )
    return out


def _select_balanced(
    per_label: Dict[str, List[SampleRecord]],
    rng: random.Random,
    max_per_class: int,
) -> Dict[str, List[SampleRecord]]:
    selected: Dict[str, List[SampleRecord]] = {}
    for label in (LABEL_NOFLIP, LABEL_UPSIDE_DOWN):
        pool = list(per_label.get(label, []))
        if not pool:
            selected[label] = []
            continue
        rng.shuffle(pool)
        if max_per_class > 0:
            take = min(max_per_class, len(pool))
        else:
            take = len(pool)
        selected[label] = pool[:take]
    return selected


def _split_list(items: List[SampleRecord], split_ratios: Tuple[float, float, float]) -> Dict[str, List[SampleRecord]]:
    train_r, val_r, test_r = split_ratios
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    # Keep all samples while ensuring non-negative sizes.
    if n_test < 0:
        n_test = 0
    if n_train + n_val + n_test < n:
        n_test += n - (n_train + n_val + n_test)

    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val : n_train + n_val + n_test],
    }


def _write_manifest(manifest_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "sample_id",
        "split",
        "label",
        "class_target",
        "output_image_rel",
        "output_image_path",
        "source_image_path",
        "setup_folder",
        "source_name",
        "image_width",
        "image_height",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_review_html(review_path: Path, entries: Sequence[Dict[str, object]]) -> None:
    review_root = review_path.parent.resolve()
    chunk_size = 200
    label_order = {LABEL_NOFLIP: 0, LABEL_UPSIDE_DOWN: 1}

    sorted_entries = sorted(
        entries,
        key=lambda r: (
            label_order.get(str(r.get("expected_label", "")), 99),
            str(r.get("split", "")),
            str(r.get("image_rel", "")),
        ),
    )

    by_label: Dict[str, List[Dict[str, object]]] = {LABEL_NOFLIP: [], LABEL_UPSIDE_DOWN: []}
    for row in sorted_entries:
        lbl = str(row.get("expected_label", ""))
        if lbl in by_label:
            by_label[lbl].append(row)

    blocks: List[Dict[str, object]] = []
    for lbl in (LABEL_NOFLIP, LABEL_UPSIDE_DOWN):
        arr = by_label[lbl]
        for start in range(0, len(arr), chunk_size):
            blocks.append(
                {
                    "label": lbl,
                    "start": start + 1,
                    "end": min(start + chunk_size, len(arr)),
                    "total": len(arr),
                    "entries": arr[start : start + chunk_size],
                }
            )

    def _esc(value: object) -> str:
        return html_lib.escape(str(value if value is not None else ""), quote=True)

    total_count = len(entries)
    total_blocks = len(blocks)

    pages_dir = review_root / "check_orientation_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_files: List[str] = []
    for i, block in enumerate(blocks, start=1):
        page_files.append(
            f"block_{i:02d}_{block['label']}_{int(block['start']):03d}_{int(block['end']):03d}.html"
        )

    style = """
:root {
  --bg: #0f131b;
  --panel: #171c26;
  --line: #2c3547;
  --text: #ecf1fa;
  --muted: #99a7c0;
  --acc: #5da6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  color: var(--text);
  background: radial-gradient(circle at 10% 0%, #222c3e, var(--bg) 40%);
  min-height: 100vh;
}
.wrap { max-width: 1800px; margin: 0 auto; padding: 12px; display: grid; gap: 10px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 auto; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
.btn {
  display: inline-block;
  border: 1px solid #3a465d;
  background: #222b3b;
  color: var(--text);
  border-radius: 8px;
  padding: 7px 10px;
  font-size: 13px;
  text-decoration: none;
}
.btn:hover { border-color: var(--acc); }
.btn.disabled { opacity: 0.55; pointer-events: none; }
.pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; background: #293348; border: 1px solid #3a4761; }
.hint { color: var(--muted); font-size: 12px; }
.grid {
  padding: 6px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
  gap: 8px;
  max-height: 78vh;
  overflow: auto;
  border: 1px solid #2a3447;
  border-radius: 10px;
  background: #0f141e;
}
.tile {
  position: relative;
  border: 1px solid #2b374d;
  border-radius: 8px;
  background: #0f141e;
  padding: 2px;
}
.thumb { width: 100%; height: 66px; object-fit: cover; display: block; border-radius: 6px; background: #090d14; }
.mark { position: absolute; top: 5px; left: 5px; width: 16px; height: 16px; z-index: 3; }
button.btn { cursor: pointer; }
.chip {
  position: absolute; top: 4px; right: 4px;
  border-radius: 999px; font-size: 10px; padding: 1px 4px;
  color: #fff; border: 1px solid rgba(255,255,255,0.25); background: rgba(0,0,0,0.62);
}
.meta { font-size: 10px; line-height: 1.2; color: #d3dcee; padding: 3px 4px 0 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.meta2 { font-size: 10px; color: #9fb0cd; padding: 0 4px 2px 4px; }
.corr-row { padding: 0 4px 4px 4px; }
.action-select {
  width: 100%;
  background: #161e2c;
  border: 1px solid #3a465d;
  color: #ecf1fa;
  border-radius: 6px;
  font-size: 11px;
  padding: 4px 6px;
}
.action-select:disabled { opacity: 0.55; }
.tile.corrected-invert { border-color: #4db872; box-shadow: inset 0 0 0 1px rgba(77,184,114,0.45); }
.tile.corrected-exclude { border-color: #e46d6d; box-shadow: inset 0 0 0 1px rgba(228,109,109,0.45); }
.tile.corrected-exclude .thumb { opacity: 0.52; }
.source { display: none; }
.zoom-wrap {
  display: none;
  position: fixed;
  top: 108px;
  right: 16px;
  width: min(44vw, 620px);
  max-height: 78vh;
  background: #0a0f17;
  border: 1px solid #2f3c52;
  border-radius: 10px;
  padding: 8px;
  z-index: 9999;
  box-shadow: 0 8px 28px rgba(0,0,0,0.45);
}
.tile:hover .zoom-wrap, .tile:focus-within .zoom-wrap { display: block; }
.zoom { width: 100%; max-height: 68vh; object-fit: contain; display: block; background: #000; border-radius: 8px; }
.zoom-caption { margin-top: 6px; font-size: 11px; color: #c7d5ef; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.zoom-origin { margin-top: 4px; font-size: 11px; color: #e2ebff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.zoom-path { margin-top: 2px; font-size: 10px; color: #9fb0cd; word-break: break-all; white-space: normal; }
@media (max-width: 1100px) {
  .zoom-wrap { right: 8px; left: 8px; width: auto; }
}
"""

    def _nav_btn(label: str, href: str, disabled: bool) -> str:
        if disabled:
            return f'<span class="btn disabled">{_esc(label)}</span>'
        return f'<a class="btn" href="{_esc(href)}">{_esc(label)}</a>'

    script = """
(() => {
  "use strict";
  const STORAGE_KEY = "us_orientation_review_corrections_v2";
  const state = { version: 2, updated_at: "", items: {} };
  const canStorage = (() => {
    try {
      const k = "__us_orient_test__";
      localStorage.setItem(k, "1");
      localStorage.removeItem(k);
      return true;
    } catch (_err) {
      return false;
    }
  })();

  const qs = (sel, root = document) => root.querySelector(sel);
  const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function safeParse(raw) {
    try { return JSON.parse(raw); } catch (_err) { return null; }
  }

  function loadState() {
    if (!canStorage) return;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = safeParse(raw);
    if (!parsed || typeof parsed !== "object") return;
    const items = parsed.items;
    if (!items || typeof items !== "object") return;
    state.version = Number(parsed.version || 2);
    state.updated_at = String(parsed.updated_at || "");
    state.items = items;
  }

  function persist() {
    state.updated_at = new Date().toISOString();
    if (canStorage) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }
    updateSaveStatus();
    updateCounters();
  }

  function csvEscape(v) {
    const s = String(v ?? "");
    if (s.includes(",") || s.includes('"') || s.includes("\\n")) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  function recordFromTile(tile, action) {
    const expected = String(tile.dataset.expectedLabel || "");
    let corrected = "";
    if (action === "invert") {
      corrected = expected === "su" ? "giu" : (expected === "giu" ? "su" : "");
    }
    return {
      sample_id: String(tile.dataset.sampleId || ""),
      action: action,
      expected_label: expected,
      corrected_label: corrected,
      split: String(tile.dataset.split || ""),
      image_rel: String(tile.dataset.imageRel || ""),
      source_image: String(tile.dataset.sourceImage || ""),
      source_name: String(tile.dataset.sourceName || ""),
      saved_at: new Date().toISOString(),
    };
  }

  function applyVisual(tile, action) {
    tile.classList.remove("corrected-invert", "corrected-exclude");
    if (action === "invert") tile.classList.add("corrected-invert");
    if (action === "exclude") tile.classList.add("corrected-exclude");
  }

  function applyTileState(tile) {
    const id = String(tile.dataset.sampleId || "");
    const mark = qs(".mark", tile);
    const sel = qs(".action-select", tile);
    if (!id || !mark || !sel) return;

    const rec = state.items[id];
    if (rec && (rec.action === "invert" || rec.action === "exclude")) {
      mark.checked = true;
      sel.value = rec.action;
      sel.disabled = false;
      applyVisual(tile, rec.action);
      return;
    }
    mark.checked = false;
    sel.value = "";
    sel.disabled = true;
    applyVisual(tile, "");
  }

  function commitTile(tile) {
    const id = String(tile.dataset.sampleId || "");
    const mark = qs(".mark", tile);
    const sel = qs(".action-select", tile);
    if (!id || !mark || !sel) return;

    const action = String(sel.value || "");
    if (mark.checked && (action === "invert" || action === "exclude")) {
      state.items[id] = recordFromTile(tile, action);
      applyVisual(tile, action);
    } else {
      delete state.items[id];
      applyVisual(tile, "");
    }
  }

  function updateCounters() {
    const all = Object.values(state.items);
    const total = all.length;
    let invert = 0;
    let exclude = 0;
    for (const rec of all) {
      if (rec && rec.action === "invert") invert += 1;
      if (rec && rec.action === "exclude") exclude += 1;
    }
    const t = qs("#corr-total");
    const i = qs("#corr-invert");
    const e = qs("#corr-exclude");
    if (t) t.textContent = String(total);
    if (i) i.textContent = String(invert);
    if (e) e.textContent = String(exclude);
  }

  function updateSaveStatus() {
    const el = qs("#autosave-status");
    if (!el) return;
    if (!canStorage) {
      el.textContent = "Autosave: non disponibile (browser)";
      return;
    }
    const ts = state.updated_at ? state.updated_at.replace("T", " ").slice(0, 19) : "-";
    el.textContent = "Autosave: OK (" + ts + ")";
  }

  function download(filename, mime, text) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function getSortedItems() {
    return Object.values(state.items).sort((a, b) => {
      const as = String((a && a.sample_id) || "");
      const bs = String((b && b.sample_id) || "");
      return as.localeCompare(bs);
    });
  }

  function exportJson() {
    const items = getSortedItems();
    const payload = {
      version: 2,
      generated_at: new Date().toISOString(),
      total: items.length,
      counts: {
        invert: items.filter((x) => x.action === "invert").length,
        exclude: items.filter((x) => x.action === "exclude").length,
      },
      items: items,
    };
    download("correzioni_orientamento.json", "application/json;charset=utf-8", JSON.stringify(payload, null, 2));
  }

  function exportCsv() {
    const items = getSortedItems();
    const headers = [
      "sample_id",
      "action",
      "expected_label",
      "corrected_label",
      "split",
      "image_rel",
      "source_name",
      "source_image",
      "saved_at",
    ];
    const lines = [headers.join(",")];
    for (const rec of items) {
      const row = headers.map((h) => csvEscape(rec[h]));
      lines.push(row.join(","));
    }
    download("correzioni_orientamento.csv", "text/csv;charset=utf-8", lines.join("\\n"));
  }

  function bindTile(tile) {
    const mark = qs(".mark", tile);
    const sel = qs(".action-select", tile);
    if (!mark || !sel) return;

    mark.addEventListener("change", () => {
      if (mark.checked) {
        if (!sel.value) sel.value = "invert";
        sel.disabled = false;
      } else {
        sel.value = "";
        sel.disabled = true;
      }
      commitTile(tile);
      persist();
    });

    sel.addEventListener("change", () => {
      if (sel.value) {
        mark.checked = true;
        sel.disabled = false;
      } else {
        mark.checked = false;
        sel.disabled = true;
      }
      commitTile(tile);
      persist();
    });
  }

  function clearAll() {
    const ok = window.confirm("Vuoi cancellare tutte le correzioni salvate?");
    if (!ok) return;
    state.items = {};
    qsa(".tile").forEach(applyTileState);
    persist();
  }

  function init() {
    loadState();
    qsa(".tile").forEach((tile) => {
      bindTile(tile);
      applyTileState(tile);
    });
    updateCounters();
    updateSaveStatus();

    const btnJson = qs("#export-json");
    const btnCsv = qs("#export-csv");
    const btnReset = qs("#clear-corrections");
    if (btnJson) btnJson.addEventListener("click", exportJson);
    if (btnCsv) btnCsv.addEventListener("click", exportCsv);
    if (btnReset) btnReset.addEventListener("click", clearAll);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""

    def _render_block_page(block_idx: int, base_prefix: str, asset_prefix: str, include_index_link: bool) -> str:
        block = blocks[block_idx]
        b_label = str(block["label"])
        b_start = int(block["start"])
        b_end = int(block["end"])
        b_total = int(block["total"])
        rows = block["entries"]  # type: ignore[assignment]

        prev_href = f"{base_prefix}{page_files[block_idx - 1]}" if block_idx > 0 else ""
        next_href = f"{base_prefix}{page_files[block_idx + 1]}" if block_idx + 1 < total_blocks else ""
        index_href = f"{base_prefix}index.html"

        tile_parts: List[str] = []
        for i, row in enumerate(rows, start=1):  # type: ignore[arg-type]
            sample_id = str(row.get("sample_id", ""))
            split = str(row.get("split", ""))
            expected = str(row.get("expected_label", ""))
            image_rel = str(row.get("image_rel", ""))
            source_image = str(row.get("source_image", ""))
            original_name = Path(source_image).name if source_image else ""
            chip = "SU" if expected == LABEL_NOFLIP else "GIU"

            small_src = f"{asset_prefix}{image_rel}" if image_rel else ""
            zoom_src = small_src

            tile_parts.append(
                "\n".join(
                    [
                        (
                            f'<div class="tile" '
                            f'data-sample-id="{_esc(sample_id)}" '
                            f'data-split="{_esc(split)}" '
                            f'data-expected-label="{_esc(expected)}" '
                            f'data-image-rel="{_esc(image_rel)}" '
                            f'data-source-image="{_esc(source_image)}" '
                            f'data-source-name="{_esc(original_name)}">'
                        ),
                        f'  <input type="checkbox" class="mark" title="Segna correzione per {_esc(sample_id)}">',
                        f'  <img class="thumb" src="{_esc(small_src)}" alt="{_esc(sample_id)}">',
                        f'  <div class="chip">{chip}</div>',
                        f'  <div class="meta mono">{i}. {_esc(sample_id)}</div>',
                        f'  <div class="meta2 mono">{_esc(split)} | {_esc(expected)}</div>',
                        '  <div class="corr-row">',
                        '    <select class="action-select mono" title="Tipo correzione">',
                        '      <option value="">Conferma (nessuna correzione)</option>',
                        '      <option value="invert">Inverti classe</option>',
                        '      <option value="exclude">Escludi</option>',
                        "    </select>",
                        "  </div>",
                        '  <div class="zoom-wrap">',
                        f'    <img class="zoom" src="{_esc(zoom_src)}" alt="{_esc(sample_id)}">',
                        f'    <div class="zoom-caption mono">{_esc(sample_id)} | {_esc(expected)} | {_esc(image_rel)}</div>',
                        f'    <div class="zoom-origin mono">Nome originale: {_esc(original_name)}</div>',
                        f'    <div class="zoom-path mono">Percorso originale: {_esc(source_image)}</div>',
                        "  </div>",
                        f'  <div class="source mono">{_esc(source_image)}</div>',
                        "</div>",
                    ]
                )
            )

        index_btn = _nav_btn("Indice blocchi", index_href, not include_index_link)
        tiles_html = "\n".join(tile_parts)

        return (
            "<!doctype html>\n"
            "<html lang=\"it\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>US Orientation Review Blocks v7</title>\n"
            f"  <style>{style}</style>\n"
            "</head>\n"
            "<body>\n"
            "  <div class=\"wrap\">\n"
            "    <div class=\"panel\">\n"
            "      <div class=\"row\">\n"
            f"        {_nav_btn('Prev blocco', prev_href, block_idx <= 0)}\n"
            f"        {_nav_btn('Next blocco', next_href, block_idx >= total_blocks - 1)}\n"
            f"        <div class=\"pill\">Pagina: {block_idx + 1}/{total_blocks}</div>\n"
            f"        {index_btn}\n"
            "        <div class=\"grow\"></div>\n"
            f"        <div class=\"pill\">Totale campioni: {total_count}</div>\n"
            f"        <div class=\"pill\">Blocchi da 200: {total_blocks}</div>\n"
            f"        <div class=\"pill\">Blocco: {b_start}-{b_end}/{b_total} ({_esc(b_label)})</div>\n"
            "      </div>\n"
            "      <div class=\"row\" style=\"margin-top:8px\">\n"
            "        <button id=\"export-json\" class=\"btn\" type=\"button\">Esporta JSON</button>\n"
            "        <button id=\"export-csv\" class=\"btn\" type=\"button\">Esporta CSV</button>\n"
            "        <button id=\"clear-corrections\" class=\"btn\" type=\"button\">Reset correzioni</button>\n"
            "        <div class=\"pill\">Correzioni: <span id=\"corr-total\">0</span></div>\n"
            "        <div class=\"pill\">Inverti: <span id=\"corr-invert\">0</span></div>\n"
            "        <div class=\"pill\">Escludi: <span id=\"corr-exclude\">0</span></div>\n"
            "        <div class=\"pill\" id=\"autosave-status\">Autosave: ...</div>\n"
            "      </div>\n"
            "      <div class=\"row\" style=\"margin-top:8px\">\n"
            "        <div class=\"hint\">Passa con il mouse su una miniatura per vedere la preview grande.</div>\n"
            "        <div class=\"hint\">Spunta l'immagine e scegli: Inverti classe oppure Escludi.</div>\n"
            "        <div class=\"hint\">Le correzioni vengono salvate in autosave nel browser.</div>\n"
            "      </div>\n"
            "    </div>\n"
            f"    <div class=\"grid\">{tiles_html}</div>\n"
            f"    <script>{script}</script>\n"
            "  </div>\n"
            "</body>\n"
            "</html>\n"
        )

    if blocks:
        for i in range(total_blocks):
            page_html = _render_block_page(i, base_prefix="", asset_prefix="../", include_index_link=True)
            (pages_dir / page_files[i]).write_text(page_html, encoding="utf-8")

        links = []
        for i, block in enumerate(blocks):
            label = str(block["label"])
            start = int(block["start"])
            end = int(block["end"])
            total = int(block["total"])
            href = page_files[i]
            links.append(
                f'<li><a href="{_esc(href)}">Blocco {i+1}: {_esc(label)} [{start}-{end}/{total}]</a></li>'
            )

        index_html = (
            "<!doctype html>\n"
            "<html lang=\"it\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>US Orientation Review Index v7</title>\n"
            "  <style>body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; background: #0f131b; color: #ecf1fa; } a { color: #89bcff; } h1 { margin-top: 0; } li { margin: 8px 0; } .muted { color: #99a7c0; }</style>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>US Orientation Review - Index</h1>\n"
            f"  <p class=\"muted\">Totale campioni: {total_count} | Blocchi: {total_blocks} (200 per blocco)</p>\n"
            f"  <ul>{''.join(links)}</ul>\n"
            "</body>\n"
            "</html>\n"
        )
        (pages_dir / "index.html").write_text(index_html, encoding="utf-8")

        root_html = _render_block_page(0, base_prefix="check_orientation_pages/", asset_prefix="", include_index_link=True)
        review_path.write_text(root_html, encoding="utf-8")
    else:
        review_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>US Orientation Review v7</title></head><body><p>Nessun campione disponibile.</p></body></html>",
            encoding="utf-8",
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare binary US orientation dataset (su vs giu) from recursive filename rules."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"),
        help="Root scanned recursively for images to classify by filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320"),
        help="Destination folder for images, manifest, summary, and review HTML.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Max number of samples kept per class. 0 = keep all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and split shuffle.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum ~1.0).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root.as_posix()}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = tuple(float(v) for v in args.split_ratios)
    if len(ratios) != 3:
        raise ValueError("split-ratios must contain exactly 3 numbers")
    if any(v < 0 for v in ratios):
        raise ValueError("split-ratios must be >= 0")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split-ratios sum must be > 0")
    ratios = (ratios[0] / ratio_sum, ratios[1] / ratio_sum, ratios[2] / ratio_sum)

    rng = random.Random(args.seed)

    setups, scan_stats, warnings = _collect_setups(dataset_root)
    per_label_all = _expand_samples(setups)
    per_label_selected = _select_balanced(
        per_label=per_label_all,
        rng=rng,
        max_per_class=max(0, int(args.max_per_class)),
    )

    per_split: Dict[str, Dict[str, List[SampleRecord]]] = {
        split: {LABEL_NOFLIP: [], LABEL_UPSIDE_DOWN: []} for split in SPLITS
    }
    for label in (LABEL_NOFLIP, LABEL_UPSIDE_DOWN):
        chosen = list(per_label_selected[label])
        rng.shuffle(chosen)
        split_map = _split_list(chosen, ratios)
        for split in SPLITS:
            per_split[split][label] = split_map[split]

    images_root = output_dir / "images"
    for split in SPLITS:
        for label in (LABEL_NOFLIP, LABEL_UPSIDE_DOWN):
            (images_root / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    review_rows: List[Dict[str, object]] = []
    counters: Dict[Tuple[str, str], int] = {}

    for split in SPLITS:
        for label in (LABEL_NOFLIP, LABEL_UPSIDE_DOWN):
            samples = per_split[split][label]
            for sample in samples:
                key = (split, label)
                counters[key] = counters.get(key, 0) + 1
                idx = counters[key]

                setup_slug = _slugify(sample.setup_folder.name, max_len=48)
                src_slug = _slugify(sample.source_image_path.stem, max_len=40)
                out_ext = sample.source_image_path.suffix.lower() or ".png"
                out_name = f"{idx:05d}__{setup_slug}__{src_slug}{out_ext}"
                out_rel = Path("images") / split / label / out_name
                out_abs = output_dir / out_rel

                out_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.source_image_path, out_abs)
                iw, ih = "", ""

                class_target = 0 if label == LABEL_NOFLIP else 1
                manifest_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": split,
                        "label": label,
                        "class_target": class_target,
                        "output_image_rel": out_rel.as_posix(),
                        "output_image_path": out_abs.as_posix(),
                        "source_image_path": sample.source_image_path.as_posix(),
                        "setup_folder": sample.setup_folder.as_posix(),
                        "source_name": sample.source_image_path.name,
                        "image_width": iw,
                        "image_height": ih,
                    }
                )
                review_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": split,
                        "expected_label": label,
                        "image_rel": out_rel.as_posix(),
                        "source_image": sample.source_image_path.as_posix(),
                    }
                )

    manifest_rows.sort(key=lambda r: (str(r["split"]), str(r["label"]), str(r["output_image_rel"])))
    review_rows.sort(key=lambda r: (str(r["split"]), str(r["expected_label"]), str(r["image_rel"])))

    manifest_csv = output_dir / "manifest.csv"
    _write_manifest(manifest_csv, manifest_rows)

    review_json = output_dir / "review_manifest.json"
    review_json.write_text(json.dumps(review_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    review_html = output_dir / "check_orientation_binary.html"
    _build_review_html(review_html, review_rows)

    split_counts = {
        split: {
            LABEL_NOFLIP: len(per_split[split][LABEL_NOFLIP]),
            LABEL_UPSIDE_DOWN: len(per_split[split][LABEL_UPSIDE_DOWN]),
        }
        for split in SPLITS
    }
    summary = {
        "dataset_root": dataset_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "seed": args.seed,
        "max_per_class": int(args.max_per_class),
        "split_ratios_normalized": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "classification_rules": {
            LABEL_NOFLIP: ["nf", "noflip", "lr"],
            LABEL_UPSIDE_DOWN: ["ud", "lrud"],
            "priority": f"{LABEL_UPSIDE_DOWN} over {LABEL_NOFLIP}",
            "case_insensitive": True,
        },
        "scan_stats": scan_stats,
        "raw_counts": {
            LABEL_NOFLIP: len(per_label_all[LABEL_NOFLIP]),
            LABEL_UPSIDE_DOWN: len(per_label_all[LABEL_UPSIDE_DOWN]),
        },
        "selected_counts": {
            LABEL_NOFLIP: len(per_label_selected[LABEL_NOFLIP]),
            LABEL_UPSIDE_DOWN: len(per_label_selected[LABEL_UPSIDE_DOWN]),
        },
        "split_counts": split_counts,
        "total_samples_written": len(manifest_rows),
        "manifest_csv": manifest_csv.as_posix(),
        "review_manifest_json": review_json.as_posix(),
        "review_html": review_html.as_posix(),
        "warnings_count": len(warnings),
        "warnings_preview": warnings[:200],
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_txt = output_dir / "summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"Dataset root: {dataset_root.as_posix()}",
                f"Output dir: {output_dir.as_posix()}",
                (
                    "Rules -> "
                    f"{LABEL_NOFLIP}: [nf, noflip, lr], "
                    f"{LABEL_UPSIDE_DOWN}: [ud, lrud] (priority {LABEL_UPSIDE_DOWN})"
                ),
                f"Raw counts -> {LABEL_NOFLIP}: {summary['raw_counts'][LABEL_NOFLIP]}, {LABEL_UPSIDE_DOWN}: {summary['raw_counts'][LABEL_UPSIDE_DOWN]}",
                f"Selected counts -> {LABEL_NOFLIP}: {summary['selected_counts'][LABEL_NOFLIP]}, {LABEL_UPSIDE_DOWN}: {summary['selected_counts'][LABEL_UPSIDE_DOWN]}",
                (
                    "Split counts -> "
                    f"train({split_counts['train'][LABEL_NOFLIP]}/{split_counts['train'][LABEL_UPSIDE_DOWN]}), "
                    f"val({split_counts['val'][LABEL_NOFLIP]}/{split_counts['val'][LABEL_UPSIDE_DOWN]}), "
                    f"test({split_counts['test'][LABEL_NOFLIP]}/{split_counts['test'][LABEL_UPSIDE_DOWN]})"
                ),
                f"Total samples written: {len(manifest_rows)}",
                f"Review HTML: {review_html.as_posix()}",
                f"Warnings: {len(warnings)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Manifest: {manifest_csv.as_posix()}", flush=True)
    print(f"Summary: {summary_json.as_posix()}", flush=True)
    print(f"Review HTML: {review_html.as_posix()}", flush=True)
    print(
        "Done. "
        f"{LABEL_NOFLIP}={summary['selected_counts'][LABEL_NOFLIP]} | "
        f"{LABEL_UPSIDE_DOWN}={summary['selected_counts'][LABEL_UPSIDE_DOWN]} | "
        f"total={len(manifest_rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
