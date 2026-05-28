#!/usr/bin/env python3
"""Build paginated HTML review package for vendor-template training manifest.

Input:
- manifest_vendor_template_no_negative.csv (or compatible manifest)

Output:
- index.html
- pages/page_XXXX.html
- review_queue.csv
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReviewRow:
    review_order: int
    row_id: str
    image_path: Path
    image_uri: str
    template_path: Optional[Path]
    template_uri: str
    split: str
    vendor: str
    vendor_id: str
    group_id: str
    dataset_folder: str
    setup_name: str
    image_width: int
    image_height: int
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int
    bbox_left_pct: float
    bbox_top_pct: float
    bbox_width_pct: float
    bbox_height_pct: float
    bbox_within_image: str
    bbox_source: str
    fss_id_echo: str
    fss_id_probe: str


RECT_ECHO_RE = re.compile(r"^\s*(\d+)\|(\d+)\|(\d+)\|(\d+)\|\s*$")
RECT_NAME_PREFIX_RE = re.compile(r"^\s*(\d+)\|(\d+)\|(\d+)\|(\d+)\|")


def _parse_int(value: str, default: int = 0) -> int:
    cleaned = (value or "").strip()
    if cleaned == "":
        return int(default)
    try:
        return int(round(float(cleaned)))
    except Exception:  # noqa: BLE001
        return int(default)


def _parse_float(value: str, default: float = 0.0) -> float:
    cleaned = (value or "").strip()
    if cleaned == "":
        return float(default)
    try:
        return float(cleaned)
    except Exception:  # noqa: BLE001
        return float(default)


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _pick_evenly_spaced(rows: Sequence[Dict[str, str]], keep_n: int) -> List[Dict[str, str]]:
    if keep_n <= 0 or len(rows) <= keep_n:
        return list(rows)
    if keep_n == 1:
        return [rows[len(rows) // 2]]
    out: List[Dict[str, str]] = []
    last_idx = len(rows) - 1
    for i in range(keep_n):
        idx = int(round((i * last_idx) / float(keep_n - 1)))
        out.append(rows[idx])
    return out


def _normalize_bbox(
    row: Dict[str, str],
    width: int,
    height: int,
) -> Tuple[int, int, int, int, float, float, float, float]:
    xmin = _parse_int(row.get("bbox_xmin", ""), default=_parse_int(row.get("bbox_left", ""), 0))
    ymin = _parse_int(row.get("bbox_ymin", ""), default=_parse_int(row.get("bbox_top", ""), 0))
    xmax = _parse_int(row.get("bbox_xmax", ""), default=_parse_int(row.get("bbox_right", ""), 0))
    ymax = _parse_int(row.get("bbox_ymax", ""), default=_parse_int(row.get("bbox_bottom", ""), 0))

    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1

    left_pct = _parse_float(row.get("bbox_norm_xmin", ""))
    top_pct = _parse_float(row.get("bbox_norm_ymin", ""))
    right_pct = _parse_float(row.get("bbox_norm_xmax", ""))
    bottom_pct = _parse_float(row.get("bbox_norm_ymax", ""))
    has_norm = any((row.get("bbox_norm_xmin", "") or "").strip() for _ in [0])

    if not has_norm or right_pct <= left_pct or bottom_pct <= top_pct:
        width_safe = max(1, int(width))
        height_safe = max(1, int(height))
        left_pct = max(0.0, min(1.0, xmin / float(width_safe)))
        top_pct = max(0.0, min(1.0, ymin / float(height_safe)))
        right_pct = max(0.0, min(1.0, xmax / float(width_safe)))
        bottom_pct = max(0.0, min(1.0, ymax / float(height_safe)))

    box_w_pct = max(0.01, (right_pct - left_pct) * 100.0)
    box_h_pct = max(0.01, (bottom_pct - top_pct) * 100.0)
    return (
        xmin,
        ymin,
        xmax,
        ymax,
        left_pct * 100.0,
        top_pct * 100.0,
        box_w_pct,
        box_h_pct,
    )


def _safe_line(lines: Sequence[str], line_num_1based: int) -> Optional[str]:
    if 1 <= line_num_1based <= len(lines):
        return lines[line_num_1based - 1].strip()
    return None


def _extract_rect_echo_line_num(lines: Sequence[str]) -> Optional[int]:
    for candidate in (11, 10):
        value = _safe_line(lines, candidate)
        if value and RECT_ECHO_RE.match(value):
            return candidate
    for idx in range(1, min(len(lines), 20) + 1):
        value = _safe_line(lines, idx)
        if value and RECT_ECHO_RE.match(value):
            return idx
    return None


def _extract_line13_bbox_from_fss(
    fss_path: Path,
    cache: Dict[str, Optional[Tuple[int, int, int, int]]],
) -> Optional[Tuple[int, int, int, int]]:
    key = fss_path.as_posix()
    if key in cache:
        return cache[key]
    try:
        lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        cache[key] = None
        return None

    rect_line = _extract_rect_echo_line_num(lines)
    if rect_line is None:
        cache[key] = None
        return None
    offset = rect_line - 11
    line13 = _safe_line(lines, 13 + offset)
    if not line13:
        cache[key] = None
        return None
    match = RECT_NAME_PREFIX_RE.match(line13.strip())
    if not match:
        cache[key] = None
        return None
    top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
    cache[key] = (top, left, bottom, right)
    return cache[key]


def _to_review_row(
    raw: Dict[str, str],
    idx: int,
    bbox_source: str,
    line13_cache: Dict[str, Optional[Tuple[int, int, int, int]]],
) -> Optional[ReviewRow]:
    image_path_str = (raw.get("image_path", "") or "").strip()
    if not image_path_str:
        return None
    image_path = Path(image_path_str).expanduser()
    if not image_path.is_absolute():
        image_path = image_path.resolve()

    template_path_str = (raw.get("template_path", "") or "").strip()
    template_path: Optional[Path] = None
    template_uri = ""
    if template_path_str:
        template_path = Path(template_path_str).expanduser()
        if not template_path.is_absolute():
            template_path = template_path.resolve()
        template_uri = template_path.as_uri()

    width = _parse_int(raw.get("image_width", ""), 0)
    height = _parse_int(raw.get("image_height", ""), 0)
    if width <= 0:
        width = _parse_int(raw.get("fss_video_x", ""), 1)
    if height <= 0:
        height = _parse_int(raw.get("fss_video_y", ""), 1)
    width = max(1, width)
    height = max(1, height)

    source_used = "manifest"
    if bbox_source == "line13_fss":
        fss_path_str = (raw.get("fss_path", "") or "").strip()
        if fss_path_str:
            fss_path = Path(fss_path_str).expanduser()
            if not fss_path.is_absolute():
                fss_path = fss_path.resolve()
            line13 = _extract_line13_bbox_from_fss(fss_path, line13_cache)
            if line13 is not None:
                top, left, bottom, right = line13
                width_safe = max(1, int(width))
                height_safe = max(1, int(height))
                xmin = left
                ymin = top
                xmax = max(left + 1, right)
                ymax = max(top + 1, bottom)
                left_pct = max(0.0, min(100.0, (xmin / float(width_safe)) * 100.0))
                top_pct = max(0.0, min(100.0, (ymin / float(height_safe)) * 100.0))
                box_w_pct = max(0.01, ((xmax - xmin) / float(width_safe)) * 100.0)
                box_h_pct = max(0.01, ((ymax - ymin) / float(height_safe)) * 100.0)
                source_used = "line13_fss"
            else:
                xmin, ymin, xmax, ymax, left_pct, top_pct, box_w_pct, box_h_pct = _normalize_bbox(
                    raw,
                    width=width,
                    height=height,
                )
                source_used = "manifest_fallback_missing_line13"
        else:
            xmin, ymin, xmax, ymax, left_pct, top_pct, box_w_pct, box_h_pct = _normalize_bbox(
                raw,
                width=width,
                height=height,
            )
            source_used = "manifest_fallback_missing_fss_path"
    else:
        xmin, ymin, xmax, ymax, left_pct, top_pct, box_w_pct, box_h_pct = _normalize_bbox(
            raw,
            width=width,
            height=height,
        )
        source_used = "manifest"

    vendor = (raw.get("manufacturer", "") or "").strip() or "UNKNOWN"
    group_id = (raw.get("group_id", "") or "").strip()
    dataset_folder = (raw.get("dataset_folder", "") or "").strip()
    setup_name = (raw.get("setup_name", "") or "").strip()
    split = (raw.get("split", "") or "").strip()
    vendor_id = (raw.get("vendor_id", "") or "").strip()
    row_id = f"{group_id}::{image_path.name}::{idx:08d}"

    return ReviewRow(
        review_order=idx,
        row_id=row_id,
        image_path=image_path,
        image_uri=image_path.as_uri(),
        template_path=template_path,
        template_uri=template_uri,
        split=split,
        vendor=vendor,
        vendor_id=vendor_id,
        group_id=group_id,
        dataset_folder=dataset_folder,
        setup_name=setup_name,
        image_width=width,
        image_height=height,
        bbox_xmin=xmin,
        bbox_ymin=ymin,
        bbox_xmax=xmax,
        bbox_ymax=ymax,
        bbox_left_pct=left_pct,
        bbox_top_pct=top_pct,
        bbox_width_pct=box_w_pct,
        bbox_height_pct=box_h_pct,
        bbox_within_image=(raw.get("bbox_within_image", "") or "").strip(),
        bbox_source=source_used,
        fss_id_echo=(raw.get("fss_id_echo", "") or "").strip(),
        fss_id_probe=(raw.get("fss_id_probe", "") or "").strip(),
    )


def _folder_key(row: ReviewRow) -> str:
    gid = (row.group_id or "").strip()
    if gid:
        return gid
    return f"{row.vendor}::{row.dataset_folder}::{row.setup_name}"


def _write_review_queue(path: Path, rows: Sequence[ReviewRow]) -> None:
    fields = [
        "review_order",
        "row_id",
        "split",
        "vendor",
        "vendor_id",
        "group_id",
        "dataset_folder",
        "setup_name",
        "image_path",
        "template_path",
        "image_width",
        "image_height",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_within_image",
        "bbox_source",
        "fss_id_echo",
        "fss_id_probe",
        "manual_status",
        "manual_issue_type",
        "manual_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "review_order": row.review_order,
                    "row_id": row.row_id,
                    "split": row.split,
                    "vendor": row.vendor,
                    "vendor_id": row.vendor_id,
                    "group_id": row.group_id,
                    "dataset_folder": row.dataset_folder,
                    "setup_name": row.setup_name,
                    "image_path": row.image_path.as_posix(),
                    "template_path": row.template_path.as_posix() if row.template_path else "",
                    "image_width": row.image_width,
                    "image_height": row.image_height,
                    "bbox_xmin": row.bbox_xmin,
                    "bbox_ymin": row.bbox_ymin,
                    "bbox_xmax": row.bbox_xmax,
                    "bbox_ymax": row.bbox_ymax,
                    "bbox_within_image": row.bbox_within_image,
                    "bbox_source": row.bbox_source,
                    "fss_id_echo": row.fss_id_echo,
                    "fss_id_probe": row.fss_id_probe,
                    "manual_status": "",
                    "manual_issue_type": "",
                    "manual_note": "",
                }
            )


def _render_card(row: ReviewRow) -> str:
    template_html = (
        f'<img loading="lazy" src="{_html_escape(row.template_uri)}" alt="template">'
        if row.template_uri
        else '<div class="missing">template non disponibile</div>'
    )
    image_flag_options = (
        '<option value="">-- none --</option>'
        '<option value="wrong_bbox">wrong_bbox</option>'
        '<option value="wrong_template">wrong_template</option>'
        '<option value="wrong_vendor">wrong_vendor</option>'
        '<option value="corrupt_or_unreadable">corrupt_or_unreadable</option>'
        '<option value="other">other</option>'
    )
    folder_flag_options = (
        '<option value="">-- none --</option>'
        '<option value="exclude_folder">exclude_folder</option>'
        '<option value="mixed_vendor">mixed_vendor</option>'
        '<option value="wrong_setup">wrong_setup</option>'
        '<option value="template_incoherent">template_incoherent</option>'
        '<option value="other">other</option>'
    )
    row_id_attr = _html_escape(row.row_id)
    group_id_attr = _html_escape(row.group_id)
    image_path_attr = _html_escape(row.image_path.as_posix())
    vendor_attr = _html_escape(row.vendor)
    split_attr = _html_escape(row.split)
    folder_attr = _html_escape(row.dataset_folder)
    setup_attr = _html_escape(row.setup_name)
    return f"""
<article class="card"
  id="card-{row.review_order}"
  data-review-order="{row.review_order}"
  data-row-id="{row_id_attr}"
  data-group-id="{group_id_attr}"
  data-image-path="{image_path_attr}"
  data-vendor="{vendor_attr}"
  data-split="{split_attr}"
  data-dataset-folder="{folder_attr}"
  data-setup-name="{setup_attr}">
  <div class="card-head">
    <button class="quick-flag-folder" type="button">FLAG TUTTA CARTELLA</button>
    <span class="flag-badge folder-flag-badge" style="display:none;"></span>
    <span class="flag-badge image-flag-badge" style="display:none;"></span>
  </div>
  <div class="visuals">
    <div class="img-wrap">
      <img loading="lazy" src="{_html_escape(row.image_uri)}" alt="{_html_escape(row.image_path.name)}">
      <div class="bbox" style="left:{row.bbox_left_pct:.5f}%;top:{row.bbox_top_pct:.5f}%;width:{row.bbox_width_pct:.5f}%;height:{row.bbox_height_pct:.5f}%"></div>
    </div>
    <div class="template-wrap">
      {template_html}
    </div>
  </div>
  <div class="meta">
    <div><b>#{row.review_order}</b> | vendor={_html_escape(row.vendor)} ({_html_escape(row.vendor_id)}) | split={_html_escape(row.split)}</div>
    <div>group={_html_escape(row.group_id)} | setup={_html_escape(row.setup_name)} | folder={_html_escape(row.dataset_folder)}</div>
    <div>bbox=({_html_escape(str(row.bbox_xmin))}, {_html_escape(str(row.bbox_ymin))}, {_html_escape(str(row.bbox_xmax))}, {_html_escape(str(row.bbox_ymax))}) | source={_html_escape(row.bbox_source)} | within={_html_escape(row.bbox_within_image)}</div>
    <div>echo={_html_escape(row.fss_id_echo)} | probe={_html_escape(row.fss_id_probe)}</div>
    <div class="path">image: {_html_escape(row.image_path.as_posix())}</div>
    <div class="path">template: {_html_escape(row.template_path.as_posix() if row.template_path else "")}</div>
    <div class="rid">row_id: {_html_escape(row.row_id)}</div>
    <div class="flag-panel">
      <div class="flag-row">
        <label>Image flag
          <select class="image-flag">{image_flag_options}</select>
        </label>
        <label>Image note
          <input class="image-note" type="text" placeholder="nota immagine">
        </label>
      </div>
      <div class="flag-row">
        <label>Folder flag
          <select class="folder-flag">{folder_flag_options}</select>
        </label>
        <label>Folder note
          <input class="folder-note" type="text" placeholder="nota cartella (sincronizzata nel gruppo)">
        </label>
      </div>
    </div>
  </div>
</article>
""".strip()


def _write_page(
    page_path: Path,
    page_num: int,
    page_count: int,
    rows: Sequence[ReviewRow],
    storage_key_suffix: str,
) -> None:
    cards = "\n".join(_render_card(r) for r in rows)
    prev_link = f'page_{page_num - 1:04d}.html' if page_num > 1 else ""
    next_link = f'page_{page_num + 1:04d}.html' if page_num < page_count else ""

    nav_parts: List[str] = []
    if prev_link:
        nav_parts.append(f'<a href="{prev_link}">Prev</a>')
    nav_parts.append(f"<span>Page {page_num}/{page_count}</span>")
    if next_link:
        nav_parts.append(f'<a href="{next_link}">Next</a>')
    nav_parts.append('<a href="../index.html">Index</a>')

    storage_key_suffix_json = json.dumps(storage_key_suffix, ensure_ascii=False)
    script_block = (
        """
  <script>
    const STORAGE_KEY_SUFFIX = __STORAGE_KEY_SUFFIX__;
    const STORAGE_KEY = "vendor_template_training_review_flags::" + STORAGE_KEY_SUFFIX;
    const URL_PARAMS = new URLSearchParams(window.location.search);
    const VIEW_MODE = (URL_PARAMS.get("view") || "").trim();
    const GROUP_FILTER = (URL_PARAMS.get("group_id") || "").trim();
    let state = null;
    let saveTimer = null;

    function defaultState() {
      return { version: 1, source: STORAGE_KEY_SUFFIX, updated_at: "", folders: {}, images: {} };
    }

    function readState() {
      let raw = null;
      try {
        raw = localStorage.getItem(STORAGE_KEY);
      } catch (_err) {
        return defaultState();
      }
      if (!raw) return defaultState();
      try {
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return defaultState();
        if (!parsed.folders || typeof parsed.folders !== "object") parsed.folders = {};
        if (!parsed.images || typeof parsed.images !== "object") parsed.images = {};
        return parsed;
      } catch (_err) {
        return defaultState();
      }
    }

    function compactState() {
      const compactFolders = {};
      Object.entries(state.folders || {}).forEach(([key, item]) => {
        if (!item || typeof item !== "object") return;
        const flag = (item.flag || "").trim();
        const note = (item.note || "").trim();
        const reviewAction = (item.review_action || "").trim();
        const reviewNote = (item.review_note || "").trim();
        if (!flag && !note && !reviewAction && !reviewNote) return;
        compactFolders[key] = {
          folder_key: key,
          group_id: item.group_id || "",
          vendor: item.vendor || "",
          split: item.split || "",
          dataset_folder: item.dataset_folder || "",
          setup_name: item.setup_name || "",
          preview_image_path: item.preview_image_path || "",
          flag,
          note,
          review_action: reviewAction,
          review_note: reviewNote,
        };
      });
      state.folders = compactFolders;

      const compactImages = {};
      Object.entries(state.images || {}).forEach(([rowId, item]) => {
        if (!item || typeof item !== "object") return;
        const flag = (item.flag || "").trim();
        const note = (item.note || "").trim();
        if (!flag && !note) return;
        compactImages[rowId] = {
          row_id: rowId,
          folder_key: item.folder_key || "",
          group_id: item.group_id || "",
          vendor: item.vendor || "",
          split: item.split || "",
          dataset_folder: item.dataset_folder || "",
          setup_name: item.setup_name || "",
          image_path: item.image_path || "",
          flag,
          note,
        };
      });
      state.images = compactImages;
    }

    function setStatus(text) {
      const el = document.getElementById("save-status");
      if (el) el.textContent = text || "";
    }

    function cardFolderKey(card) {
      const gid = (card.dataset.groupId || "").trim();
      if (gid) return gid;
      const vendor = (card.dataset.vendor || "").trim();
      const folder = (card.dataset.datasetFolder || "").trim();
      const setup = (card.dataset.setupName || "").trim();
      return [vendor, folder, setup].join("::");
    }

    function ensureFolder(card, create = false) {
      const folderKey = cardFolderKey(card);
      if (!folderKey) return null;
      const base = {
        folder_key: folderKey,
        group_id: card.dataset.groupId || "",
        vendor: card.dataset.vendor || "",
        split: card.dataset.split || "",
        dataset_folder: card.dataset.datasetFolder || "",
        setup_name: card.dataset.setupName || "",
        preview_image_path: card.dataset.imagePath || "",
        flag: "",
        note: "",
        review_action: "",
        review_note: "",
      };
      if (!state.folders[folderKey]) {
        if (!create) return base;
        state.folders[folderKey] = {
          ...base,
          flag: "",
          note: "",
        };
      }
      const item = state.folders[folderKey];
      if (create) {
        if (!item.group_id) item.group_id = card.dataset.groupId || "";
        if (!item.vendor) item.vendor = card.dataset.vendor || "";
        if (!item.split) item.split = card.dataset.split || "";
        if (!item.dataset_folder) item.dataset_folder = card.dataset.datasetFolder || "";
        if (!item.setup_name) item.setup_name = card.dataset.setupName || "";
        if (!item.preview_image_path) item.preview_image_path = card.dataset.imagePath || "";
        if (typeof item.review_action !== "string") item.review_action = "";
        if (typeof item.review_note !== "string") item.review_note = "";
      }
      return item;
    }

    function ensureImage(card, create = false) {
      const rowId = (card.dataset.rowId || "").trim();
      if (!rowId) return null;
      const folder = ensureFolder(card, create);
      const base = {
        row_id: rowId,
        folder_key: folder ? folder.folder_key : cardFolderKey(card),
        group_id: card.dataset.groupId || "",
        vendor: card.dataset.vendor || "",
        split: card.dataset.split || "",
        dataset_folder: card.dataset.datasetFolder || "",
        setup_name: card.dataset.setupName || "",
        image_path: card.dataset.imagePath || "",
        flag: "",
        note: "",
      };
      if (!state.images[rowId]) {
        if (!create) return base;
        state.images[rowId] = {
          ...base,
          flag: "",
          note: "",
        };
      }
      const item = state.images[rowId];
      if (create) {
        if (!item.folder_key && folder) item.folder_key = folder.folder_key;
        if (!item.group_id) item.group_id = card.dataset.groupId || "";
        if (!item.vendor) item.vendor = card.dataset.vendor || "";
        if (!item.split) item.split = card.dataset.split || "";
        if (!item.dataset_folder) item.dataset_folder = card.dataset.datasetFolder || "";
        if (!item.setup_name) item.setup_name = card.dataset.setupName || "";
        if (!item.image_path) item.image_path = card.dataset.imagePath || "";
      }
      return item;
    }

    function cardControls(card) {
      return {
        imageFlag: card.querySelector(".image-flag"),
        imageNote: card.querySelector(".image-note"),
        folderFlag: card.querySelector(".folder-flag"),
        folderNote: card.querySelector(".folder-note")
      };
    }

    function updateCardCss(card) {
      const folder = ensureFolder(card);
      const image = ensureImage(card);
      const folderFlag = !!(folder && folder.flag);
      const imageFlag = !!(image && image.flag);
      card.classList.toggle("has-folder-flag", folderFlag);
      card.classList.toggle("has-image-flag", imageFlag);
      const folderBadge = card.querySelector(".folder-flag-badge");
      const imageBadge = card.querySelector(".image-flag-badge");
      if (folderBadge) {
        folderBadge.style.display = folderFlag ? "inline-block" : "none";
        folderBadge.textContent = folderFlag ? `FOLDER: ${folder.flag}` : "";
      }
      if (imageBadge) {
        imageBadge.style.display = imageFlag ? "inline-block" : "none";
        imageBadge.textContent = imageFlag ? `IMAGE: ${image.flag}` : "";
      }
    }

    function isImageFlagged(card) {
      const image = ensureImage(card);
      return !!(image && image.flag);
    }

    function isFolderFlagged(card) {
      const folder = ensureFolder(card);
      return !!(folder && folder.flag);
    }

    function isCardFlagged(card) {
      const image = ensureImage(card);
      const folder = ensureFolder(card);
      return !!((image && image.flag) || (folder && folder.flag));
    }

    function applyFilter() {
      const onlyFlagged = !!document.getElementById("only-flagged")?.checked;
      document.querySelectorAll(".card").forEach((card) => {
        let show = true;
        if (GROUP_FILTER && cardFolderKey(card) !== GROUP_FILTER) {
          show = false;
        }
        if (VIEW_MODE === "folder_flagged") {
          show = show && isFolderFlagged(card);
        } else if (onlyFlagged) {
          show = show && isImageFlagged(card);
        }
        card.classList.toggle("hidden-by-filter", !show);
      });
    }

    function renderFlaggedSection() {
      if (VIEW_MODE === "folder_flagged") {
        const panel = document.querySelector(".flagged-section");
        if (panel) panel.style.display = "none";
        return;
      }
      const container = document.getElementById("flagged-list");
      if (!container) return;
      const flaggedCards = Array.from(document.querySelectorAll(".card")).filter((card) => isImageFlagged(card));
      if (flaggedCards.length === 0) {
        container.innerHTML = '<div class="flagged-empty">Nessuna immagine flaggata in questa pagina.</div>';
        return;
      }
      const rows = flaggedCards
        .map((card) => {
          const image = ensureImage(card);
          const folder = ensureFolder(card);
          const title = `#${card.dataset.reviewOrder || ""} | ${card.dataset.vendor || ""} | group=${card.dataset.groupId || ""}`;
          const imageTxt = image && image.flag ? `image=${image.flag}` : "";
          const folderTxt = folder && folder.flag ? `folder=${folder.flag}` : "";
          const flagsText = [folderTxt, imageTxt].filter(Boolean).join(" | ");
          const cardId = card.id || "";
          return (
            '<div class="flagged-item">' +
            `<div class="txt"><b>${title}</b><br>${flagsText}</div>` +
            `<a href="#${cardId}">Vai alla card</a>` +
            "</div>"
          );
        })
        .join("");
      container.innerHTML = rows;
    }

    function applyViewModeUi() {
      const onlyFlagged = document.getElementById("only-flagged");
      const modeStatus = document.getElementById("mode-status");
      if (VIEW_MODE === "folder_flagged") {
        if (onlyFlagged) {
          onlyFlagged.checked = false;
          onlyFlagged.disabled = true;
        }
        if (modeStatus) {
          modeStatus.textContent = GROUP_FILTER
            ? `Modalita cartella flaggata: group=${GROUP_FILTER}`
            : "Modalita solo cartelle flaggate";
        }
      } else if (modeStatus) {
        modeStatus.textContent = "";
      }
    }

    function applyFolderToCards(folderKey, sourceCard) {
      document.querySelectorAll(".card").forEach((card) => {
        if (cardFolderKey(card) !== folderKey) return;
        const folder = ensureFolder(card);
        const controls = cardControls(card);
        if (!controls.folderFlag || !controls.folderNote || !folder) return;
        if (!sourceCard || sourceCard !== card) {
          controls.folderFlag.value = folder.flag || "";
          controls.folderNote.value = folder.note || "";
        }
        updateCardCss(card);
      });
    }

    function applyStateToUi() {
      document.querySelectorAll(".card").forEach((card) => {
        const folder = ensureFolder(card);
        const image = ensureImage(card);
        const controls = cardControls(card);
        if (controls.imageFlag && image) controls.imageFlag.value = image.flag || "";
        if (controls.imageNote && image) controls.imageNote.value = image.note || "";
        if (controls.folderFlag && folder) controls.folderFlag.value = folder.flag || "";
        if (controls.folderNote && folder) controls.folderNote.value = folder.note || "";
        updateCardCss(card);
      });
      updateCounters();
    }

    function persistState() {
      compactState();
      state.updated_at = new Date().toISOString();
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        setStatus("autosave: " + state.updated_at);
      } catch (_err) {
        setStatus("errore salvataggio localStorage");
      }
    }

    function queuePersist() {
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {
        persistState();
        updateCounters();
      }, 250);
    }

    function updateCounters() {
      const cards = Array.from(document.querySelectorAll(".card"));
      let imageFlagged = 0;
      cards.forEach((card) => {
        const item = ensureImage(card);
        if (item && item.flag) imageFlagged += 1;
      });
      const folderFlagged = Object.values(state.folders).filter((item) => item && item.flag).length;
      const text = `image flag: ${imageFlagged}/${cards.length} | folder flag: ${folderFlagged}`;
      const el = document.getElementById("counts");
      if (el) el.textContent = text;
      applyFilter();
      renderFlaggedSection();
    }

    function csvEscape(value) {
      const s = String(value ?? "");
      if (s.includes(",") || s.includes("\\"") || s.includes("\\n")) {
        return "\\"" + s.replaceAll("\\"", "\\"\\"") + "\\"";
      }
      return s;
    }

    function downloadText(filename, text, contentType) {
      try {
        const blob = new Blob([text], { type: contentType || "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          a.remove();
          URL.revokeObjectURL(url);
        }, 0);
        return true;
      } catch (_err) {
        return false;
      }
    }

    function showExportFallback(filename, text, mode) {
      const panel = document.getElementById("export-fallback");
      const nameEl = document.getElementById("export-fallback-filename");
      const area = document.getElementById("export-fallback-text");
      const hint = document.getElementById("export-fallback-hint");
      if (nameEl) nameEl.textContent = filename;
      if (area) area.value = text;
      if (hint) {
        hint.textContent =
          mode === "download_ok"
            ? "Download avviato. Se non trovi il file, usa Copia."
            : "Download bloccato in questa vista. Usa Copia.";
      }
      if (panel) panel.style.display = "block";
    }

    function hideExportFallback() {
      const panel = document.getElementById("export-fallback");
      if (panel) panel.style.display = "none";
    }

    async function copyFallbackText() {
      const area = document.getElementById("export-fallback-text");
      if (!area) return;
      const text = area.value || "";
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
          setStatus("contenuto export copiato negli appunti");
          return;
        }
      } catch (_err) {
      }
      area.focus();
      area.select();
      try {
        document.execCommand("copy");
        setStatus("contenuto export copiato negli appunti");
      } catch (_err) {
        setStatus("copia non riuscita, copiare manualmente");
      }
    }

    function exportJson() {
      persistState();
      const payload = JSON.stringify(state, null, 2);
      const ok = downloadText("vendor_template_flags.json", payload, "application/json;charset=utf-8");
      showExportFallback("vendor_template_flags.json", payload, ok ? "download_ok" : "download_blocked");
    }

    function exportCsv() {
      persistState();
      const header = [
        "record_type",
        "row_id",
        "group_id",
        "vendor",
        "split",
        "dataset_folder",
        "setup_name",
        "image_path",
        "image_flag",
        "image_note",
        "folder_flag",
        "folder_note",
        "review_action",
        "review_note"
      ];
      const rows = [header];

      Object.values(state.folders)
        .sort((a, b) => String(a.folder_key || "").localeCompare(String(b.folder_key || "")))
        .forEach((folder) => {
          if (!folder) return;
          const hasFolderIssue = !!(folder.flag || folder.note || folder.review_action || folder.review_note);
          if (!hasFolderIssue) return;
          rows.push([
            "folder",
            "",
            folder.group_id || "",
            folder.vendor || "",
            folder.split || "",
            folder.dataset_folder || "",
            folder.setup_name || "",
            "",
            "",
            "",
            folder.flag || "",
            folder.note || "",
            folder.review_action || "",
            folder.review_note || ""
          ]);
        });

      Object.values(state.images)
        .sort((a, b) => String(a.row_id || "").localeCompare(String(b.row_id || "")))
        .forEach((image) => {
          if (!image) return;
          const folder = state.folders[image.folder_key || ""] || {};
          const hasIssue = !!(image.flag || image.note || folder.flag || folder.note || folder.review_action || folder.review_note);
          if (!hasIssue) return;
          rows.push([
            "image",
            image.row_id || "",
            image.group_id || "",
            image.vendor || "",
            image.split || "",
            image.dataset_folder || "",
            image.setup_name || "",
            image.image_path || "",
            image.flag || "",
            image.note || "",
            folder.flag || "",
            folder.note || "",
            folder.review_action || "",
            folder.review_note || ""
          ]);
        });

      const payload = rows.map((row) => row.map(csvEscape).join(",")).join("\\n");
      const ok = downloadText("vendor_template_flags.csv", payload, "text/csv;charset=utf-8");
      showExportFallback("vendor_template_flags.csv", payload, ok ? "download_ok" : "download_blocked");
    }

    function clearFlags() {
      if (!confirm("Confermi reset completo dei flag locali?")) return;
      state = defaultState();
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (_err) {
      }
      applyStateToUi();
      setStatus("flag locali resettati");
    }

    function bindEvents() {
      document.querySelectorAll(".card").forEach((card) => {
        const controls = cardControls(card);
        const quickFolderBtn = card.querySelector(".quick-flag-folder");
        if (quickFolderBtn) {
          quickFolderBtn.addEventListener("click", () => {
            const folder = ensureFolder(card, true);
            if (!folder) return;
            folder.flag = "exclude_folder";
            applyFolderToCards(folder.folder_key, null);
            persistState();
            updateCounters();
          });
        }
        if (controls.imageFlag) {
          controls.imageFlag.addEventListener("change", () => {
            const image = ensureImage(card, true);
            if (!image) return;
            image.flag = controls.imageFlag.value || "";
            updateCardCss(card);
            persistState();
            updateCounters();
          });
        }
        if (controls.imageNote) {
          controls.imageNote.addEventListener("input", () => {
            const image = ensureImage(card, true);
            if (!image) return;
            image.note = controls.imageNote.value || "";
            queuePersist();
          });
        }
        if (controls.folderFlag) {
          controls.folderFlag.addEventListener("change", () => {
            const folder = ensureFolder(card, true);
            if (!folder) return;
            folder.flag = controls.folderFlag.value || "";
            applyFolderToCards(folder.folder_key, card);
            persistState();
            updateCounters();
          });
        }
        if (controls.folderNote) {
          controls.folderNote.addEventListener("input", () => {
            const folder = ensureFolder(card, true);
            if (!folder) return;
            folder.note = controls.folderNote.value || "";
            applyFolderToCards(folder.folder_key, card);
            queuePersist();
          });
        }
      });

      const exportJsonBtn = document.getElementById("btn-export-json");
      if (exportJsonBtn) exportJsonBtn.addEventListener("click", exportJson);

      const exportCsvBtn = document.getElementById("btn-export-csv");
      if (exportCsvBtn) exportCsvBtn.addEventListener("click", exportCsv);

      const clearBtn = document.getElementById("btn-clear-flags");
      if (clearBtn) clearBtn.addEventListener("click", clearFlags);

      const copyBtn = document.getElementById("export-fallback-copy-btn");
      if (copyBtn) copyBtn.addEventListener("click", copyFallbackText);

      const closeBtn = document.getElementById("export-fallback-close-btn");
      if (closeBtn) closeBtn.addEventListener("click", hideExportFallback);

      const onlyFlagged = document.getElementById("only-flagged");
      if (onlyFlagged) {
        onlyFlagged.addEventListener("change", () => {
          applyFilter();
        });
      }
    }

    state = readState();
    compactState();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_err) {
    }
    bindEvents();
    applyViewModeUi();
    applyStateToUi();
    setStatus("annotazioni pronte (autosave locale attivo)");
  </script>
"""
    ).replace("__STORAGE_KEY_SUFFIX__", storage_key_suffix_json)

    html_out = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Training Review Page {page_num}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 14px; background:#0f1218; color:#e8e8e8; }}
    a {{ color:#9dd0ff; text-decoration:none; }}
    .nav {{ display:flex; gap:12px; align-items:center; margin-bottom:12px; }}
    .cards {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    .card {{ border:1px solid #2e3646; border-radius:6px; background:#1a1f2a; overflow:hidden; }}
    .card-head {{ padding:8px; border-bottom:1px solid #2e3646; background:#23151a; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .quick-flag-folder {{ width:100%; border:1px solid #a22929; border-radius:6px; background:#c82f2f; color:#fff; font-weight:700; padding:9px 10px; font-size:13px; cursor:pointer; letter-spacing:0.2px; }}
    .quick-flag-folder:hover {{ background:#e03838; }}
    .flag-badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
    .folder-flag-badge {{ background:#ff5757; color:#fff; }}
    .image-flag-badge {{ background:#ff9f43; color:#111; }}
    .visuals {{ display:grid; grid-template-columns: minmax(380px, 2fr) minmax(180px, 1fr); gap:8px; padding:8px; }}
    .img-wrap {{ position:relative; background:#090c12; min-height:180px; }}
    .img-wrap img {{ width:100%; height:auto; display:block; }}
    .bbox {{ position:absolute; border:2px solid #ff5252; box-sizing:border-box; pointer-events:none; }}
    .template-wrap {{ background:#090c12; display:flex; align-items:center; justify-content:center; min-height:120px; }}
    .template-wrap img {{ max-width:100%; max-height:260px; display:block; }}
    .template-wrap .missing {{ color:#bbb; font-size:12px; padding:8px; }}
    .meta {{ padding:8px; font-size:12px; line-height:1.35; border-top:1px solid #2e3646; }}
    .meta .path {{ color:#c8c8c8; word-break:break-all; }}
    .meta .rid {{ color:#95a1b8; }}
    .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:0 0 12px 0; }}
    .toolbar button {{ border:1px solid #3d4f6a; border-radius:6px; background:#223249; color:#eef4ff; padding:6px 10px; cursor:pointer; }}
    .toolbar button:hover {{ background:#2f4562; }}
    .toolbar .warn {{ background:#4a2a2a; border-color:#6c3b3b; }}
    .toolbar .warn:hover {{ background:#603434; }}
    .toolbar .status {{ color:#a8c5e6; }}
    .toolbar .mode {{ color:#ffd59b; font-weight:700; }}
    .toolbar .counts {{ color:#d4e4f8; font-weight:600; }}
    .toolbar .filter-inline {{ display:flex; align-items:center; gap:6px; color:#d4e4f8; font-weight:600; }}
    .toolbar .filter-inline input {{ width:16px; height:16px; }}
    .flag-panel {{ margin-top:6px; padding-top:6px; border-top:1px dashed #384359; }}
    .flag-row {{ display:grid; grid-template-columns: minmax(180px, 0.8fr) minmax(260px, 1.2fr); gap:8px; margin-top:6px; }}
    .flag-row label {{ display:flex; flex-direction:column; gap:4px; }}
    .flag-row select, .flag-row input {{ border:1px solid #44526a; border-radius:4px; background:#0f1520; color:#e8e8e8; padding:5px 6px; font-size:12px; }}
    .card.has-image-flag {{ border-color:#ff8a3d; box-shadow:0 0 0 2px rgba(255,138,61,0.35) inset; background:linear-gradient(180deg, #231f17 0%, #1a1f2a 80%); }}
    .card.has-folder-flag {{ border-color:#ff4d4d; box-shadow:0 0 0 2px rgba(255,77,77,0.45) inset; background:linear-gradient(180deg, #2a1414 0%, #1a1f2a 80%); }}
    .card.hidden-by-filter {{ display:none; }}
    .flagged-section {{ margin:14px 0; border:1px solid #3a475f; border-radius:8px; background:#141d2c; padding:10px; }}
    .flagged-section h3 {{ margin:0 0 8px 0; }}
    .flagged-list {{ display:grid; grid-template-columns:1fr; gap:6px; }}
    .flagged-item {{ display:flex; align-items:center; justify-content:space-between; gap:10px; border:1px solid #314057; border-radius:6px; background:#101726; padding:7px 8px; }}
    .flagged-item .txt {{ font-size:12px; color:#dce8ff; }}
    .flagged-item a {{ font-size:12px; font-weight:700; }}
    .flagged-empty {{ color:#9fb0cc; font-size:12px; }}
    #export-fallback {{ margin-top:12px; border:1px solid #40506a; border-radius:8px; padding:10px; background:#162134; }}
    #export-fallback textarea {{ width:100%; min-height:120px; background:#0c111b; color:#e8e8e8; border:1px solid #40506a; border-radius:6px; padding:8px; }}
    #export-fallback .row {{ display:flex; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }}
    #export-fallback .hint {{ color:#a5b8d6; font-size:12px; }}
  </style>
</head>
<body>
  <div class="nav">{' '.join(nav_parts)}</div>
  <div class="toolbar">
    <button id="btn-export-json" type="button">Export JSON</button>
    <button id="btn-export-csv" type="button">Export CSV</button>
    <button id="btn-clear-flags" class="warn" type="button">Reset flags</button>
    <label class="filter-inline"><input id="only-flagged" type="checkbox"> Solo immagini flaggate</label>
    <span id="mode-status" class="mode"></span>
    <span id="counts" class="counts"></span>
    <span id="save-status" class="status"></span>
  </div>
  <div class="flagged-section">
    <h3>Sezione Solo Immagini Flaggate</h3>
    <div id="flagged-list" class="flagged-list"></div>
  </div>
  <div class="cards" id="all-cards">
    {cards}
  </div>
  <div class="nav" style="margin-top:12px;">{' '.join(nav_parts)}</div>
  <div id="export-fallback" style="display:none;">
    <div><b>Export:</b> <span id="export-fallback-filename"></span></div>
    <div id="export-fallback-hint" class="hint"></div>
    <textarea id="export-fallback-text" readonly></textarea>
    <div class="row">
      <button id="export-fallback-copy-btn" type="button">Copia</button>
      <button id="export-fallback-close-btn" type="button">Chiudi</button>
    </div>
  </div>
{script_block}
</body>
</html>
"""
    page_path.write_text(html_out, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build HTML review package for vendor-template training dataset."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv"),
        help="Input manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_template_training_review_no_negative"),
        help="Output folder for HTML package.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated splits to include (e.g. train,val,test).",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=0,
        help="Max images per group/folder (0=all).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Max total rows after filtering/sampling (0=all).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Rows per HTML page.",
    )
    parser.add_argument(
        "--bbox-source",
        type=str,
        default="line13_fss",
        choices=("line13_fss", "manifest"),
        help="Source for red bbox overlay.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pages_dir = output_dir / "pages"

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    include_splits = {
        token.strip() for token in args.splits.split(",") if token.strip()
    }
    if not include_splits:
        raise ValueError("--splits non valido: specificare almeno uno split.")

    raw_rows = _load_manifest_rows(manifest)
    filtered = [row for row in raw_rows if (row.get("split", "") or "").strip() in include_splits]
    if not filtered:
        raise RuntimeError("Nessuna riga dopo filtro split.")

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in filtered:
        group_id = (row.get("group_id", "") or "").strip()
        grouped[group_id].append(row)

    sampled: List[Dict[str, str]] = []
    for group_id in sorted(grouped):
        rows = sorted(grouped[group_id], key=lambda r: (r.get("image_path", "") or ""))
        picked = _pick_evenly_spaced(rows, max(0, int(args.max_per_group)))
        sampled.extend(picked)

    rng = random.Random(int(args.seed))
    sampled.sort(key=lambda r: ((r.get("manufacturer", "") or ""), (r.get("group_id", "") or ""), (r.get("image_path", "") or "")))
    if args.max_rows > 0 and len(sampled) > int(args.max_rows):
        idxs = list(range(len(sampled)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[: int(args.max_rows)])
        sampled = [sampled[i] for i in idxs]

    review_rows: List[ReviewRow] = []
    line13_cache: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    skipped = 0
    for idx, raw in enumerate(sampled, start=1):
        rr = _to_review_row(
            raw,
            idx,
            bbox_source=str(args.bbox_source),
            line13_cache=line13_cache,
        )
        if rr is None:
            skipped += 1
            continue
        review_rows.append(rr)

    if not review_rows:
        raise RuntimeError("Nessuna riga valida da esportare in review HTML.")

    page_size = max(1, int(args.page_size))
    page_count = int(math.ceil(len(review_rows) / float(page_size)))
    storage_key_suffix = f"{output_dir.as_posix()}::{manifest.as_posix()}"
    folder_first_page: Dict[str, int] = {}
    folder_meta: Dict[str, Dict[str, str]] = {}
    folder_preview_src: Dict[str, Path] = {}
    for row_idx, row in enumerate(review_rows):
        key = _folder_key(row)
        if key not in folder_first_page:
            folder_first_page[key] = int(row_idx // page_size) + 1
        if key not in folder_meta:
            folder_meta[key] = {
                "group_id": row.group_id,
                "vendor": row.vendor,
                "split": row.split,
                "dataset_folder": row.dataset_folder,
                "setup_name": row.setup_name,
                "preview_bbox_left_pct": float(row.bbox_left_pct),
                "preview_bbox_top_pct": float(row.bbox_top_pct),
                "preview_bbox_width_pct": float(row.bbox_width_pct),
                "preview_bbox_height_pct": float(row.bbox_height_pct),
            }
            folder_preview_src[key] = row.image_path

    previews_dir = output_dir / "folder_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    for key, src_path in folder_preview_src.items():
        src = Path(src_path)
        rel_preview = ""
        if src.exists() and src.is_file():
            ext = src.suffix.lower() if src.suffix else ".png"
            name = f"{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}{ext}"
            dst = previews_dir / name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass
            if dst.exists():
                rel_preview = f"folder_previews/{name}"
        folder_meta[key]["preview_rel"] = rel_preview
        folder_meta[key]["preview_src"] = src.as_posix()
    for page_idx in range(page_count):
        start = page_idx * page_size
        end = min(len(review_rows), start + page_size)
        page_rows = review_rows[start:end]
        page_path = pages_dir / f"page_{page_idx + 1:04d}.html"
        _write_page(
            page_path=page_path,
            page_num=page_idx + 1,
            page_count=page_count,
            rows=page_rows,
            storage_key_suffix=storage_key_suffix,
        )

    _write_review_queue(output_dir / "review_queue.csv", review_rows)

    vendor_counts = Counter(r.vendor for r in review_rows)
    split_counts = Counter(r.split for r in review_rows)
    group_counts = Counter(r.group_id for r in review_rows)

    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": output_dir.as_posix(),
        "filters": {
            "splits": sorted(include_splits),
            "max_per_group": int(args.max_per_group),
            "max_rows": int(args.max_rows),
            "page_size": page_size,
            "bbox_source": str(args.bbox_source),
        },
        "total_rows_selected": len(review_rows),
        "total_groups_selected": len(group_counts),
        "total_pages": page_count,
        "skipped_rows": skipped,
        "rows_by_vendor": dict(sorted(vendor_counts.items())),
        "rows_by_split": dict(sorted(split_counts.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vendor_items = "".join(
        f"<li>{_html_escape(vendor)}: {count}</li>"
        for vendor, count in sorted(vendor_counts.items())
    )
    split_items = "".join(
        f"<li>{_html_escape(split)}: {count}</li>"
        for split, count in sorted(split_counts.items())
    )
    links = "".join(
        f'<li><a href="pages/page_{i:04d}.html">Page {i}</a></li>'
        for i in range(1, page_count + 1)
    )
    storage_key_suffix_json = json.dumps(storage_key_suffix, ensure_ascii=False)
    folder_first_page_json = json.dumps(folder_first_page, ensure_ascii=False)
    folder_meta_json = json.dumps(folder_meta, ensure_ascii=False)
    index_script = (
        """
<script>
  const STORAGE_KEY_SUFFIX = __STORAGE_KEY_SUFFIX__;
  const STORAGE_KEY = "vendor_template_training_review_flags::" + STORAGE_KEY_SUFFIX;
  const FOLDER_FIRST_PAGE = __FOLDER_FIRST_PAGE__;
  const FOLDER_META = __FOLDER_META__;
  let reviewSaveTimer = null;

  function readState() {
    let raw = null;
    try {
      raw = localStorage.getItem(STORAGE_KEY);
    } catch (_err) {
      return {folders: {}, __storage_error: "localStorage non accessibile in questa pagina"};
    }
    if (!raw) return {folders: {}};
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return {folders: {}};
      if (!parsed.folders || typeof parsed.folders !== "object") parsed.folders = {};
      return parsed;
    } catch (_err) {
      return {folders: {}};
    }
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function writeState(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      return true;
    } catch (_err) {
      return false;
    }
  }

  function ensureFolderEntry(state, folderKey, meta) {
    if (!state.folders || typeof state.folders !== "object") state.folders = {};
    if (!state.folders[folderKey]) {
      state.folders[folderKey] = {
        folder_key: folderKey,
        group_id: meta.group_id || "",
        vendor: meta.vendor || "",
        split: meta.split || "",
        dataset_folder: meta.dataset_folder || "",
        setup_name: meta.setup_name || "",
        preview_image_path: meta.preview_src || "",
        flag: "",
        note: "",
        review_action: "",
        review_note: "",
      };
    }
    const item = state.folders[folderKey];
    if (!item.group_id) item.group_id = meta.group_id || "";
    if (!item.vendor) item.vendor = meta.vendor || "";
    if (!item.split) item.split = meta.split || "";
    if (!item.dataset_folder) item.dataset_folder = meta.dataset_folder || "";
    if (!item.setup_name) item.setup_name = meta.setup_name || "";
    if (!item.preview_image_path) item.preview_image_path = meta.preview_src || "";
    if (typeof item.review_action !== "string") item.review_action = "";
    if (typeof item.review_note !== "string") item.review_note = "";
    return item;
  }

  function findByFolderKey(container, className, folderKey) {
    const arr = Array.from(container.querySelectorAll(`.${className}`));
    return arr.find((el) => (el.dataset.folderKey || "") === folderKey) || null;
  }

  function autosaveFolderReview(list, summary, folderKey, immediate) {
    const persist = () => {
      if (!folderKey) return;
      const actionEl = findByFolderKey(list, "review-action", folderKey);
      const noteEl = findByFolderKey(list, "review-note", folderKey);
      const action = (actionEl && actionEl.value ? actionEl.value : "").trim();
      const note = (noteEl && noteEl.value ? noteEl.value : "").trim();
      const current = readState();
      if (current.__storage_error) {
        if (summary) summary.textContent = current.__storage_error;
        return;
      }
      const metaCurrent = FOLDER_META[folderKey] || {};
      const entry = ensureFolderEntry(current, folderKey, metaCurrent);
      entry.review_action = action;
      entry.review_note = note;
      if (writeState(current)) {
        if (summary) summary.textContent = `Autosave annotazioni: ${new Date().toLocaleTimeString()}`;
      } else if (summary) {
        summary.textContent = "Errore: salvataggio locale non riuscito";
      }
    };

    if (immediate) {
      if (reviewSaveTimer) clearTimeout(reviewSaveTimer);
      persist();
      return;
    }
    if (reviewSaveTimer) clearTimeout(reviewSaveTimer);
    reviewSaveTimer = setTimeout(persist, 250);
  }

  function renderFlaggedFolders() {
    const list = document.getElementById("flagged-folders-list");
    const summary = document.getElementById("flagged-folders-summary");
    if (!list) return;
    const state = readState();
    if (state.__storage_error) {
      if (summary) summary.textContent = state.__storage_error;
      list.innerHTML = '<div class="muted">Storage locale bloccato: apri la review via localhost (non file://).</div>';
      return;
    }
    const folders = state.folders || {};
    const keys = Object.keys(folders)
      .filter((k) => folders[k] && folders[k].flag)
      .sort((a, b) => a.localeCompare(b));

    if (summary) {
      summary.textContent = `Cartelle flaggate locali: ${keys.length}`;
    }
    if (keys.length === 0) {
      list.innerHTML = '<div class="muted">Nessuna cartella flaggata trovata nel browser corrente.</div>';
      return;
    }

    const items = keys.map((k) => {
      const local = folders[k] || {};
      const meta = FOLDER_META[k] || {};
      const folder = ensureFolderEntry(state, k, meta);
      const pageNum = Number(FOLDER_FIRST_PAGE[k] || 1);
      const pageName = `page_${String(pageNum).padStart(4, "0")}.html`;
      const href = `pages/${pageName}?view=folder_flagged&group_id=${encodeURIComponent(k)}`;
      const vendor = folder.vendor || "";
      const groupId = folder.group_id || k;
      const setup = folder.setup_name || "";
      const datasetFolder = folder.dataset_folder || "";
      const flag = folder.flag || "";
      const previewRel = meta.preview_rel || "";
      const previewSrc = previewRel ? previewRel : "";
      const bboxLeft = Number(meta.preview_bbox_left_pct || 0);
      const bboxTop = Number(meta.preview_bbox_top_pct || 0);
      const bboxWidth = Number(meta.preview_bbox_width_pct || 0);
      const bboxHeight = Number(meta.preview_bbox_height_pct || 0);
      const hasBbox = bboxWidth > 0 && bboxHeight > 0;
      const action = folder.review_action || "";
      const reviewNote = folder.review_note || "";
      const optionExclude = action === "escludere" ? " selected" : "";
      const optionRect = action === "modificare_rettangolo" ? " selected" : "";
      const optionOther = action === "altro" ? " selected" : "";
      return (
        `<div class="folder-card" data-folder-key="${escapeHtml(k)}">` +
        (previewSrc
          ? `<div class="folder-preview"><img src="${escapeHtml(previewSrc)}" alt="preview ${escapeHtml(groupId)}">` +
            (hasBbox
              ? `<div class="preview-bbox" style="left:${bboxLeft.toFixed(5)}%;top:${bboxTop.toFixed(5)}%;width:${bboxWidth.toFixed(5)}%;height:${bboxHeight.toFixed(5)}%"></div>`
              : "") +
            `</div>`
          : `<div class="folder-preview missing">preview non disponibile</div>`) +
        '<div class="folder-body">' +
        `<div class="txt"><b>${escapeHtml(vendor)}</b> | group=${escapeHtml(groupId)} | setup=${escapeHtml(setup)} | folder=${escapeHtml(datasetFolder)} | flag=${escapeHtml(flag)}</div>` +
        '<div class="review-row">' +
        '<label>Azione' +
        `<select class="review-action" data-folder-key="${escapeHtml(k)}">` +
        '<option value="">-- seleziona --</option>' +
        `<option value="escludere"${optionExclude}>escludere</option>` +
        `<option value="modificare_rettangolo"${optionRect}>modificare rettangolo</option>` +
        `<option value="altro"${optionOther}>altro</option>` +
        '</select>' +
        '</label>' +
        '</div>' +
        '<div class="review-row">' +
        '<label>Note (obbligatorie se scegli "altro")' +
        `<textarea class="review-note" data-folder-key="${escapeHtml(k)}" rows="3" placeholder="descrivi cosa non va">${escapeHtml(reviewNote)}</textarea>` +
        '</label>' +
        '</div>' +
        '<div class="review-actions">' +
        `<a href="${href}">Apri vista cartella flaggata</a>` +
        '</div>' +
        '</div>' +
        '</div>'
      );
    });
    list.innerHTML = items.join("");

    list.querySelectorAll(".review-action").forEach((el) => {
      el.addEventListener("change", () => {
        const folderKey = el.dataset.folderKey || "";
        autosaveFolderReview(list, summary, folderKey, true);
      });
    });
    list.querySelectorAll(".review-note").forEach((el) => {
      el.addEventListener("input", () => {
        const folderKey = el.dataset.folderKey || "";
        autosaveFolderReview(list, summary, folderKey, false);
      });
    });
  }

  document.getElementById("refresh-flagged-folders")?.addEventListener("click", renderFlaggedFolders);
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY) renderFlaggedFolders();
  });
  renderFlaggedFolders();
</script>
"""
    )
    index_script = index_script.replace("__STORAGE_KEY_SUFFIX__", storage_key_suffix_json)
    index_script = index_script.replace("__FOLDER_FIRST_PAGE__", folder_first_page_json)
    index_script = index_script.replace("__FOLDER_META__", folder_meta_json)

    index_html = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Vendor-Template Training Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; background:#0f1218; color:#e8e8e8; }}
    a {{ color:#9dd0ff; text-decoration:none; }}
    .panel {{ background:#1a1f2a; border:1px solid #2e3646; border-radius:8px; padding:12px; margin-bottom:14px; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap:10px; }}
    ul {{ margin:6px 0 0 18px; }}
    .pages {{ columns: 3; -webkit-columns:3; -moz-columns:3; }}
    .muted {{ color:#9eb0c8; font-size:12px; }}
    .panel button {{ border:1px solid #3d4f6a; border-radius:6px; background:#223249; color:#eef4ff; padding:6px 10px; cursor:pointer; }}
    .panel button:hover {{ background:#2f4562; }}
    .flagged-folders-list {{ margin:10px 0 0 0; display:grid; gap:12px; }}
    .folder-card {{ border:1px solid #35435a; border-radius:8px; background:#121a28; padding:10px; display:grid; grid-template-columns:minmax(340px,1fr) minmax(320px,1fr); gap:12px; }}
    .folder-preview {{ position:relative; background:#0c121d; border:1px solid #2d3a52; border-radius:6px; overflow:hidden; }}
    .folder-preview img {{ width:100%; height:auto; display:block; }}
    .folder-preview .preview-bbox {{ position:absolute; border:3px solid #ff4343; box-sizing:border-box; pointer-events:none; box-shadow:0 0 0 1px rgba(255,67,67,0.4) inset; }}
    .folder-preview.missing {{ min-height:220px; color:#98a8c2; font-size:12px; display:flex; align-items:center; justify-content:center; }}
    .folder-body {{ display:flex; flex-direction:column; gap:8px; }}
    .folder-body .txt {{ font-size:12px; color:#dce8ff; }}
    .review-row label {{ display:flex; flex-direction:column; gap:4px; font-size:12px; color:#dce8ff; }}
    .review-row select, .review-row textarea {{ border:1px solid #44526a; border-radius:4px; background:#0f1520; color:#e8e8e8; padding:6px; font-size:12px; }}
    .review-actions {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  </style>
</head>
<body>
  <h1>Vendor-Template Training Review</h1>
  <div class="panel">
    <div><b>Manifest:</b> {_html_escape(manifest.as_posix())}</div>
    <div><b>Rows selected:</b> {len(review_rows)}</div>
    <div><b>Groups selected:</b> {len(group_counts)}</div>
    <div><b>Pages:</b> {page_count}</div>
    <div><b>Splits:</b> {_html_escape(', '.join(sorted(include_splits)))}</div>
    <div><b>max_per_group:</b> {int(args.max_per_group)} | <b>max_rows:</b> {int(args.max_rows)} | <b>page_size:</b> {page_size}</div>
    <div><b>Review queue:</b> <a href="review_queue.csv">review_queue.csv</a></div>
    <div><b>Summary JSON:</b> <a href="summary.json">summary.json</a></div>
    <div><b>Flagging:</b> in ogni pagina puoi impostare image/folder flag, con autosave locale + export JSON/CSV.</div>
  </div>
  <div class="grid">
    <div class="panel">
      <h3>Rows by Vendor</h3>
      <ul>{vendor_items}</ul>
    </div>
    <div class="panel">
      <h3>Rows by Split</h3>
      <ul>{split_items}</ul>
    </div>
  </div>
  <div class="panel">
    <h3>Cartelle Flaggate</h3>
    <div class="muted">Questa sezione legge i folder-flag salvati nel browser locale, mostra una preview con rettangolo e salva automaticamente le annotazioni.</div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px;">
      <button id="refresh-flagged-folders" type="button">Aggiorna lista</button>
      <span id="flagged-folders-summary" class="muted"></span>
    </div>
    <div id="flagged-folders-list" class="flagged-folders-list"></div>
  </div>
  <div class="panel pages">
    <h3>Pages</h3>
    <ul>{links}</ul>
  </div>
{index_script}
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Output dir: {output_dir}")
    print(f"Index: {output_dir / 'index.html'}")
    print(f"Rows selected: {len(review_rows)} | Pages: {page_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
