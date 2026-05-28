#!/usr/bin/env python3
"""Build HTML review for per-folder OSD rotation actions.

Important note:
- If rotation was already applied in-place, original files are no longer available.
- "Before" previews are reconstructed by applying the inverse transform to current files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
CAPTURE_FILENAME_PATTERN_RE = re.compile(
    r"^.+_(vga|hdmi)_(\d{3,5})[xX](\d{3,5})(?:[_\-].*)?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ActionRow:
    folder_path: Path
    folder_name: str
    images_total: int
    sample_image: Path
    osd_rotate_deg: int
    osd_confidence: float
    decision: str
    applied: int
    rotated_images: int
    errors: int
    warning: str


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _slugify(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    out = out.strip("_")
    return out or "item"


def _collect_acquisition_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not CAPTURE_FILENAME_PATTERN_RE.match(path.stem):
            continue
        out.append(path)
    out.sort()
    return out


def _select_preview_images(
    all_images: Sequence[Path],
    sample_image: Path,
    max_images: int,
) -> List[Path]:
    if not all_images:
        return []
    max_count = max(1, int(max_images))
    out: List[Path] = []
    seen: set[Path] = set()

    def _push(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        out.append(p)

    if sample_image in all_images:
        _push(sample_image)

    candidate_idxs = [0, len(all_images) // 2, len(all_images) - 1]
    for idx in candidate_idxs:
        _push(all_images[idx])
        if len(out) >= max_count:
            return out[:max_count]

    for p in all_images:
        _push(p)
        if len(out) >= max_count:
            break

    return out[:max_count]


def _resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = float(max_side / float(longest))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return image.resize((nw, nh), resample=Image.Resampling.BILINEAR)


def _write_preview_pair(
    source_path: Path,
    rotate_deg_clockwise: int,
    before_out: Path,
    after_out: Path,
    max_side: int,
) -> None:
    with Image.open(source_path) as img:
        after = img.convert("RGB")
        before = after.rotate(int(rotate_deg_clockwise), expand=True)
        before = _resize_to_max_side(before, max_side=max_side)
        after = _resize_to_max_side(after, max_side=max_side)
        before.save(before_out, format="PNG")
        after.save(after_out, format="PNG")


def _read_actions(path: Path) -> List[ActionRow]:
    rows: List[ActionRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder_path = Path((row.get("folder_path") or "").strip())
            sample_image = Path((row.get("sample_image") or "").strip()) if (row.get("sample_image") or "").strip() else Path("")
            rows.append(
                ActionRow(
                    folder_path=folder_path,
                    folder_name=(row.get("folder_name") or folder_path.name or "").strip(),
                    images_total=_to_int(row.get("images_total", "0"), default=0),
                    sample_image=sample_image,
                    osd_rotate_deg=_to_int(row.get("osd_rotate_deg", "0"), default=0),
                    osd_confidence=_to_float(row.get("osd_confidence", "0"), default=0.0),
                    decision=(row.get("decision") or "").strip(),
                    applied=_to_int(row.get("applied", "0"), default=0),
                    rotated_images=_to_int(row.get("rotated_images", "0"), default=0),
                    errors=_to_int(row.get("errors", "0"), default=0),
                    warning=(row.get("warning") or "").strip(),
                )
            )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build HTML review for OSD rotation actions.")
    p.add_argument(
        "--actions-csv",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/runs/rotation_one_sample_by_folder/rotation_actions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/runs/rotation_one_sample_by_folder/review_html"),
    )
    p.add_argument(
        "--max-preview-images-per-folder",
        type=int,
        default=3,
        help="How many images per rotated folder to show (includes sample image).",
    )
    p.add_argument(
        "--preview-max-side",
        type=int,
        default=900,
        help="Resize preview PNG so longest side is at most this value (0=no resize).",
    )
    return p


def _build_options_html(options: Sequence[Sequence[str]]) -> str:
    chunks: List[str] = []
    for item in options:
        value = str(item[0])
        label = str(item[1])
        chunks.append(
            f"<option value=\"{html.escape(value, quote=True)}\">{html.escape(label)}</option>"
        )
    return "".join(chunks)


def main() -> int:
    args = _build_parser().parse_args()
    if args.max_preview_images_per_folder <= 0:
        raise ValueError("--max-preview-images-per-folder must be > 0.")
    if args.preview_max_side < 0:
        raise ValueError("--preview-max-side must be >= 0.")

    actions_csv = args.actions_csv.expanduser().resolve()
    if not actions_csv.exists():
        raise RuntimeError(f"Actions CSV not found: {actions_csv}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _read_actions(actions_csv)
    applied_rows = [r for r in all_rows if r.applied == 1 and r.osd_rotate_deg in (90, 180, 270)]
    skipped_rows = [r for r in all_rows if r.applied != 1]

    folder_flag_options = _build_options_html(
        [
            ("", "-- seleziona --"),
            ("correct", "corretta"),
            ("rotation_not_needed", "rotazione non necessaria"),
            ("wrong_angle", "angolo sbagliato"),
            ("wrong_rotation", "rotazione sbagliata"),
            ("partial", "parzialmente corretta"),
            ("uncertain", "incerta"),
            ("needs_reprocess", "da rieseguire"),
        ]
    )
    image_flag_options = _build_options_html(
        [
            ("", "-- seleziona --"),
            ("ok", "ok"),
            ("before_reconstruction_wrong", "before ricostruito non plausibile"),
            ("after_wrong", "after non corretto"),
            ("artifact_or_unreadable", "artifact / illeggibile"),
            ("other", "altro"),
        ]
    )

    sections: List[str] = []
    for idx, row in enumerate(applied_rows, start=1):
        folder_images = _collect_acquisition_images(row.folder_path)
        previews = _select_preview_images(
            all_images=folder_images,
            sample_image=row.sample_image,
            max_images=args.max_preview_images_per_folder,
        )

        folder_slug = f"{idx:03d}_{_slugify(row.folder_name)}"
        folder_assets = assets_dir / folder_slug
        folder_assets.mkdir(parents=True, exist_ok=True)

        folder_key = row.folder_path.as_posix()
        folder_key_attr = html.escape(folder_key, quote=True)
        folder_name_attr = html.escape(row.folder_name, quote=True)
        folder_path_attr = html.escape(row.folder_path.as_posix(), quote=True)

        preview_rows: List[str] = []
        for j, image_path in enumerate(previews, start=1):
            img_slug = _slugify(image_path.stem)
            before_name = f"{j:02d}_{img_slug}_before_reconstructed.png"
            after_name = f"{j:02d}_{img_slug}_after_current.png"
            before_path = folder_assets / before_name
            after_path = folder_assets / after_name
            _write_preview_pair(
                source_path=image_path,
                rotate_deg_clockwise=row.osd_rotate_deg,
                before_out=before_path,
                after_out=after_path,
                max_side=args.preview_max_side,
            )
            rel_before = before_path.relative_to(output_dir).as_posix()
            rel_after = after_path.relative_to(output_dir).as_posix()
            image_key = image_path.as_posix()
            image_key_attr = html.escape(image_key, quote=True)
            is_sample = (image_path == row.sample_image)
            preview_rows.append(
                "<tr class=\"preview-row\" "
                f"data-folder-key=\"{folder_key_attr}\" "
                f"data-folder-name=\"{folder_name_attr}\" "
                f"data-folder-path=\"{folder_path_attr}\" "
                f"data-image-path=\"{image_key_attr}\" "
                f"data-is-sample=\"{'1' if is_sample else '0'}\">"
                f"<td>{j}</td>"
                f"<td>{html.escape(image_key)}</td>"
                f"<td>{'YES' if is_sample else ''}</td>"
                f"<td><img src=\"{html.escape(rel_before)}\" loading=\"lazy\" /></td>"
                f"<td><img src=\"{html.escape(rel_after)}\" loading=\"lazy\" /></td>"
                f"<td><select class=\"image-flag\" data-folder-key=\"{folder_key_attr}\" "
                f"data-image-path=\"{image_key_attr}\">{image_flag_options}</select></td>"
                f"<td><input class=\"image-note\" type=\"text\" "
                f"data-folder-key=\"{folder_key_attr}\" data-image-path=\"{image_key_attr}\" "
                "placeholder=\"nota immagine\" /></td>"
                "</tr>"
            )

        sections.append(
            "<section class=\"card annot-section\" "
            f"data-folder-key=\"{folder_key_attr}\" "
            f"data-folder-name=\"{folder_name_attr}\" "
            f"data-folder-path=\"{folder_path_attr}\">"
            f"<h2>{idx}. {html.escape(row.folder_name)}</h2>"
            "<div class=\"meta\">"
            f"<div><b>Folder:</b> {html.escape(row.folder_path.as_posix())}</div>"
            f"<div><b>Decision:</b> {html.escape(row.decision)}</div>"
            f"<div><b>OSD rotate (CW):</b> {row.osd_rotate_deg}°</div>"
            f"<div><b>OSD confidence:</b> {row.osd_confidence:.4f}</div>"
            f"<div><b>Images total in folder:</b> {row.images_total}</div>"
            f"<div><b>Images rotated:</b> {row.rotated_images}</div>"
            f"<div><b>Errors:</b> {row.errors}</div>"
            f"<div><b>Sample used for decision:</b> {html.escape(row.sample_image.as_posix())}</div>"
            "</div>"
            "<div class=\"annot-box\">"
            "<div class=\"annot-row\">"
            "<label><b>Flag cartella</b></label>"
            f"<select class=\"folder-flag\" data-folder-key=\"{folder_key_attr}\" "
            f"data-folder-name=\"{folder_name_attr}\" data-folder-path=\"{folder_path_attr}\">{folder_flag_options}</select>"
            "</div>"
            "<div class=\"annot-row\">"
            "<label><b>Nota cartella</b></label>"
            f"<textarea class=\"folder-note\" data-folder-key=\"{folder_key_attr}\" "
            f"data-folder-name=\"{folder_name_attr}\" data-folder-path=\"{folder_path_attr}\" "
            "placeholder=\"annotazioni, problemi, ipotesi...\"></textarea>"
            "</div>"
            "</div>"
            "<table>"
            "<thead><tr>"
            "<th>#</th><th>Image path</th><th>Sample used</th>"
            "<th>Before (reconstructed)</th><th>After (current)</th>"
            "<th>Flag immagine</th><th>Nota immagine</th>"
            "</tr></thead>"
            f"<tbody>{''.join(preview_rows)}</tbody>"
            "</table>"
            "</section>"
        )

    skipped_list = "".join(
        "<tr>"
        f"<td>{html.escape(r.folder_name)}</td>"
        f"<td>{html.escape(r.decision)}</td>"
        f"<td>{r.osd_rotate_deg}</td>"
        f"<td>{r.osd_confidence:.4f}</td>"
        f"<td>{html.escape(r.warning)}</td>"
        "</tr>"
        for r in skipped_rows
    )

    actions_csv_json = json.dumps(actions_csv.as_posix())
    script_block = (
        """
<script>
const ACTIONS_CSV_PATH = __ACTIONS_CSV_PATH__;
const STORAGE_KEY = "rotation_osd_review_annotations::" + ACTIONS_CSV_PATH;
let state = null;
let saveTimer = null;

function defaultState() {
  return {version: 1, source_actions_csv: ACTIONS_CSV_PATH, updated_at: "", folders: {}};
}

function readState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultState();
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return defaultState();
    if (!parsed.folders || typeof parsed.folders !== "object") parsed.folders = {};
    if (!parsed.source_actions_csv) parsed.source_actions_csv = ACTIONS_CSV_PATH;
    return parsed;
  } catch (_err) {
    return defaultState();
  }
}

function writeStatus(msg) {
  const el = document.getElementById("save-status");
  if (el) el.textContent = msg;
}

function ensureFolder(folderKey, folderName, folderPath) {
  if (!state.folders[folderKey]) {
    state.folders[folderKey] = {
      folder_key: folderKey,
      folder_name: folderName || "",
      folder_path: folderPath || "",
      flag: "",
      note: "",
      images: {}
    };
  }
  const f = state.folders[folderKey];
  if (folderName) f.folder_name = folderName;
  if (folderPath) f.folder_path = folderPath;
  if (!f.images || typeof f.images !== "object") f.images = {};
  return f;
}

function ensureImage(folderObj, imagePath, isSample) {
  if (!folderObj.images[imagePath]) {
    folderObj.images[imagePath] = {image_path: imagePath, is_sample: !!isSample, flag: "", note: ""};
  } else {
    folderObj.images[imagePath].is_sample = !!isSample;
  }
  return folderObj.images[imagePath];
}

function persistState() {
  state.updated_at = new Date().toISOString();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  writeStatus("Salvato in locale: " + state.updated_at);
}

function schedulePersist() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    collectFromUi();
    persistState();
    updateSummary();
  }, 350);
}

function collectFromUi() {
  document.querySelectorAll(".annot-section").forEach((section) => {
    const folderKey = section.dataset.folderKey || "";
    const folderName = section.dataset.folderName || "";
    const folderPath = section.dataset.folderPath || "";
    if (!folderKey) return;
    const folderObj = ensureFolder(folderKey, folderName, folderPath);
    const flagEl = section.querySelector(".folder-flag");
    const noteEl = section.querySelector(".folder-note");
    folderObj.flag = (flagEl && flagEl.value) ? flagEl.value : "";
    folderObj.note = (noteEl && noteEl.value) ? noteEl.value : "";
  });

  document.querySelectorAll(".preview-row").forEach((row) => {
    const folderKey = row.dataset.folderKey || "";
    const folderName = row.dataset.folderName || "";
    const folderPath = row.dataset.folderPath || "";
    const imagePath = row.dataset.imagePath || "";
    const isSample = (row.dataset.isSample || "0") === "1";
    if (!folderKey || !imagePath) return;
    const folderObj = ensureFolder(folderKey, folderName, folderPath);
    const imageObj = ensureImage(folderObj, imagePath, isSample);
    const flagEl = row.querySelector(".image-flag");
    const noteEl = row.querySelector(".image-note");
    imageObj.flag = (flagEl && flagEl.value) ? flagEl.value : "";
    imageObj.note = (noteEl && noteEl.value) ? noteEl.value : "";
  });
}

function applyStateToUi() {
  document.querySelectorAll(".annot-section").forEach((section) => {
    const folderKey = section.dataset.folderKey || "";
    const folderName = section.dataset.folderName || "";
    const folderPath = section.dataset.folderPath || "";
    if (!folderKey) return;
    const folderObj = ensureFolder(folderKey, folderName, folderPath);
    const flagEl = section.querySelector(".folder-flag");
    const noteEl = section.querySelector(".folder-note");
    if (flagEl) flagEl.value = folderObj.flag || "";
    if (noteEl) noteEl.value = folderObj.note || "";
  });

  document.querySelectorAll(".preview-row").forEach((row) => {
    const folderKey = row.dataset.folderKey || "";
    const folderName = row.dataset.folderName || "";
    const folderPath = row.dataset.folderPath || "";
    const imagePath = row.dataset.imagePath || "";
    const isSample = (row.dataset.isSample || "0") === "1";
    if (!folderKey || !imagePath) return;
    const folderObj = ensureFolder(folderKey, folderName, folderPath);
    const imageObj = ensureImage(folderObj, imagePath, isSample);
    const flagEl = row.querySelector(".image-flag");
    const noteEl = row.querySelector(".image-note");
    if (flagEl) flagEl.value = imageObj.flag || "";
    if (noteEl) noteEl.value = imageObj.note || "";
  });
}

function updateSummary() {
  const sections = Array.from(document.querySelectorAll(".annot-section"));
  const total = sections.length;
  let flagged = 0;
  const byFlag = {};
  sections.forEach((section) => {
    const folderKey = section.dataset.folderKey || "";
    const folder = state.folders[folderKey];
    const flag = folder && folder.flag ? folder.flag : "";
    if (flag) {
      flagged += 1;
      byFlag[flag] = (byFlag[flag] || 0) + 1;
    }
  });

  let imageFlagged = 0;
  Object.values(state.folders).forEach((folder) => {
    if (!folder || !folder.images) return;
    Object.values(folder.images).forEach((img) => {
      if (img && img.flag) imageFlagged += 1;
    });
  });

  const rows = Object.keys(byFlag)
    .sort()
    .map((k) => `<span class="pill">${k}: ${byFlag[k]}</span>`)
    .join(" ");
  const el = document.getElementById("annotation-summary");
  if (el) {
    el.innerHTML =
      `<div><b>Cartelle annotate:</b> ${flagged}/${total}</div>` +
      `<div><b>Immagini con flag:</b> ${imageFlagged}</div>` +
      (rows ? `<div class="pill-wrap">${rows}</div>` : "");
  }
}

function applyFilter() {
  const onlyFlagged = !!document.getElementById("only-flagged")?.checked;
  document.querySelectorAll(".annot-section").forEach((section) => {
    const folderKey = section.dataset.folderKey || "";
    const folder = state.folders[folderKey];
    const show = !onlyFlagged || !!(folder && folder.flag);
    section.style.display = show ? "" : "none";
  });
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (s.includes(",") || s.includes("\"") || s.includes("\\n")) {
    return "\"" + s.split("\"").join("\"\"") + "\"";
  }
  return s;
}

function showExportFallback(filename, text, mode) {
  const panel = document.getElementById("export-fallback");
  const filenameEl = document.getElementById("export-fallback-filename");
  const hintEl = document.getElementById("export-fallback-hint");
  const area = document.getElementById("export-fallback-text");
  if (filenameEl) filenameEl.textContent = filename;
  if (area) area.value = text;
  if (hintEl) {
    hintEl.textContent =
      mode === "download_ok"
        ? "Download avviato. Se non trovi il file, usa Copia."
        : "Download non disponibile in questa vista. Usa Copia.";
  }
  if (panel) panel.style.display = "block";
}

function hideExportFallback() {
  const panel = document.getElementById("export-fallback");
  if (panel) panel.style.display = "none";
}

async function copyExportText() {
  const area = document.getElementById("export-fallback-text");
  if (!area) return;
  const text = area.value || "";
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      writeStatus("Contenuto export copiato negli appunti.");
      return;
    }
  } catch (_err) {
    // fallback below
  }
  area.focus();
  area.select();
  try {
    document.execCommand("copy");
    writeStatus("Contenuto export copiato negli appunti.");
  } catch (_err) {
    writeStatus("Copia non riuscita: seleziona e copia manualmente.");
  }
}

function downloadText(filename, text, contentType) {
  try {
    const blob = new Blob([text], {type: contentType || "text/plain;charset=utf-8"});
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

function exportJson() {
  collectFromUi();
  persistState();
  const payload = JSON.stringify(state, null, 2);
  const ok = downloadText("rotation_osd_annotations.json", payload, "application/json;charset=utf-8");
  showExportFallback("rotation_osd_annotations.json", payload, ok ? "download_ok" : "download_blocked");
}

function exportCsv() {
  collectFromUi();
  persistState();
  const lines = [];
  lines.push([
    "level",
    "folder_key",
    "folder_name",
    "folder_path",
    "folder_flag",
    "folder_note",
    "image_path",
    "is_sample",
    "image_flag",
    "image_note"
  ].join(","));

  document.querySelectorAll(".annot-section").forEach((section) => {
    const folderKey = section.dataset.folderKey || "";
    const folder = state.folders[folderKey] || {folder_name: "", folder_path: "", flag: "", note: "", images: {}};
    lines.push([
      "folder",
      csvEscape(folderKey),
      csvEscape(folder.folder_name || ""),
      csvEscape(folder.folder_path || ""),
      csvEscape(folder.flag || ""),
      csvEscape(folder.note || ""),
      "",
      "",
      "",
      ""
    ].join(","));

    section.querySelectorAll(".preview-row").forEach((row) => {
      const imagePath = row.dataset.imagePath || "";
      const isSample = (row.dataset.isSample || "0") === "1" ? "1" : "0";
      const img = (folder.images && folder.images[imagePath]) || {flag: "", note: ""};
      lines.push([
        "image",
        csvEscape(folderKey),
        csvEscape(folder.folder_name || ""),
        csvEscape(folder.folder_path || ""),
        csvEscape(folder.flag || ""),
        csvEscape(folder.note || ""),
        csvEscape(imagePath),
        isSample,
        csvEscape(img.flag || ""),
        csvEscape(img.note || "")
      ].join(","));
    });
  });

  const payload = lines.join("\\n");
  const ok = downloadText("rotation_osd_annotations.csv", payload, "text/csv;charset=utf-8");
  showExportFallback("rotation_osd_annotations.csv", payload, ok ? "download_ok" : "download_blocked");
}

function importJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(String(reader.result || ""));
      if (!parsed || typeof parsed !== "object" || !parsed.folders || typeof parsed.folders !== "object") {
        alert("File JSON non valido.");
        return;
      }
      state = parsed;
      if (!state.source_actions_csv) state.source_actions_csv = ACTIONS_CSV_PATH;
      if (!state.folders) state.folders = {};
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      applyStateToUi();
      updateSummary();
      applyFilter();
      writeStatus("Import completato.");
    } catch (_err) {
      alert("Errore parsing JSON.");
    }
  };
  reader.readAsText(file, "utf-8");
}

function clearAnnotations() {
  if (!confirm("Cancellare tutte le annotazioni locali?")) return;
  state = defaultState();
  localStorage.removeItem(STORAGE_KEY);
  applyStateToUi();
  updateSummary();
  applyFilter();
  writeStatus("Annotazioni cancellate.");
}

function bindEvents() {
  document.querySelectorAll(".folder-flag, .image-flag").forEach((el) => {
    el.addEventListener("change", () => {
      schedulePersist();
      applyFilter();
    });
  });
  document.querySelectorAll(".folder-note, .image-note").forEach((el) => {
    el.addEventListener("input", () => {
      schedulePersist();
    });
  });

  document.getElementById("save-btn")?.addEventListener("click", () => {
    collectFromUi();
    persistState();
    updateSummary();
    applyFilter();
  });
  document.getElementById("export-json-btn")?.addEventListener("click", exportJson);
  document.getElementById("export-csv-btn")?.addEventListener("click", exportCsv);
  document.getElementById("export-fallback-copy-btn")?.addEventListener("click", () => {
    copyExportText();
  });
  document.getElementById("export-fallback-close-btn")?.addEventListener("click", () => {
    hideExportFallback();
  });
  document.getElementById("clear-btn")?.addEventListener("click", clearAnnotations);
  document.getElementById("only-flagged")?.addEventListener("change", applyFilter);
  document.getElementById("import-json-input")?.addEventListener("change", (ev) => {
    const input = ev.target;
    const file = input && input.files && input.files[0] ? input.files[0] : null;
    if (file) importJsonFile(file);
    if (input) input.value = "";
  });
}

state = readState();
bindEvents();
applyStateToUi();
updateSummary();
applyFilter();
writeStatus("Annotazioni pronte (autosave locale attivo).");
</script>
"""
    ).replace("__ACTIONS_CSV_PATH__", actions_csv_json)

    html_out = output_dir / "index.html"
    page = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Rotation OSD Review</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px;color:#222;}"
        "h1{margin:0 0 12px 0;} .note{background:#fff3cd;border:1px solid #ffe69c;padding:10px;border-radius:8px;margin-bottom:16px;}"
        ".summary{background:#fff;border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:16px;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;margin-bottom:16px;}"
        ".toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;}"
        ".toolbar button,.toolbar label{font-size:13px;} .toolbar input[type=file]{display:none;}"
        ".toolbar .import-btn{display:inline-block;border:1px solid #bbb;border-radius:6px;padding:6px 8px;background:#fafafa;cursor:pointer;}"
        ".toolbar .status{font-size:12px;color:#666;}"
        ".export-fallback{display:none;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;}"
        ".export-fallback .top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px;}"
        ".export-fallback .hint{font-size:12px;color:#555;}"
        ".export-fallback textarea{width:100%;min-height:220px;resize:vertical;font-family:monospace;font-size:12px;}"
        ".meta{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-bottom:10px;font-size:14px;}"
        ".annot-box{border:1px solid #ddd;border-radius:8px;padding:8px;margin-bottom:10px;background:#fcfcfc;}"
        ".annot-row{display:flex;flex-direction:column;gap:4px;margin-bottom:8px;}"
        ".annot-row:last-child{margin-bottom:0;} .annot-row textarea{min-height:64px;resize:vertical;}"
        "select,input,textarea,button{font-family:inherit;}"
        "table{width:100%;border-collapse:collapse;background:#fff;} th,td{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:13px;}"
        "th{background:#f0f0f0;} img{max-width:100%;height:auto;display:block;border:1px solid #ccc;}"
        ".pill-wrap{margin-top:6px;} .pill{display:inline-block;border:1px solid #ccc;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px;background:#fafafa;}"
        "@media (max-width:1000px){.meta{grid-template-columns:1fr;}}"
        "</style></head><body>"
        "<h1>Review Rotazione OSD Per Cartella</h1>"
        "<div class=\"note\"><b>Nota importante:</b> le immagini originali sono state sovrascritte dalla rotazione in-place. "
        "La colonna <i>Before (reconstructed)</i> e ricostruita invertendo la rotazione applicata.</div>"
        "<div class=\"summary\">"
        f"<div><b>Actions CSV:</b> {html.escape(actions_csv.as_posix())}</div>"
        f"<div><b>Folders scanned:</b> {len(all_rows)}</div>"
        f"<div><b>Folders rotated (applied=1):</b> {len(applied_rows)}</div>"
        f"<div><b>Folders not rotated / skipped:</b> {len(skipped_rows)}</div>"
        "</div>"
        "<div class=\"toolbar\">"
        "<button id=\"save-btn\" type=\"button\">Salva Ora</button>"
        "<button id=\"export-json-btn\" type=\"button\">Export JSON</button>"
        "<button id=\"export-csv-btn\" type=\"button\">Export CSV</button>"
        "<label class=\"import-btn\" for=\"import-json-input\">Import JSON</label>"
        "<input id=\"import-json-input\" type=\"file\" accept=\"application/json,.json\" />"
        "<button id=\"clear-btn\" type=\"button\">Pulisci Annotazioni</button>"
        "<label><input id=\"only-flagged\" type=\"checkbox\" /> mostra solo cartelle flaggate</label>"
        "<span id=\"save-status\" class=\"status\"></span>"
        "</div>"
        "<div id=\"export-fallback\" class=\"export-fallback\">"
        "<div class=\"top\">"
        "<b>Export fallback</b>"
        "<span id=\"export-fallback-filename\"></span>"
        "<button id=\"export-fallback-copy-btn\" type=\"button\">Copia contenuto</button>"
        "<button id=\"export-fallback-close-btn\" type=\"button\">Chiudi</button>"
        "</div>"
        "<div id=\"export-fallback-hint\" class=\"hint\"></div>"
        "<textarea id=\"export-fallback-text\" spellcheck=\"false\"></textarea>"
        "</div>"
        "<div id=\"annotation-summary\" class=\"summary\"></div>"
        f"{''.join(sections) if sections else '<div class=\"card\">No rotated folders found.</div>'}"
        "<section class=\"card\"><h2>Skipped / Not Rotated</h2>"
        "<table><thead><tr><th>Folder</th><th>Decision</th><th>OSD rotate</th><th>OSD confidence</th><th>Warning</th></tr></thead>"
        f"<tbody>{skipped_list}</tbody></table></section>"
        f"{script_block}"
        "</body></html>"
    )
    html_out.write_text(page, encoding="utf-8")

    print(f"Review HTML: {html_out}", flush=True)
    print(f"Rotated folders in report: {len(applied_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
