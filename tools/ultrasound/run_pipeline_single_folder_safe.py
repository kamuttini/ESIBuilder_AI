#!/usr/bin/env python3
"""Run main FSS pipeline on one input folder using an operational copy.

Pipeline tested:
1) dedup
2) rotation
3) vendor
4) rect
5) L/T (per frame, crop rect)
6) probe

The original folder is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


RECT_ECHO_RE = r"^\s*\d+\|\d+\|\d+\|\d+\|\s*$"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
CAPTURE_FILENAME_PATTERN_RE = re.compile(
    r"^.+_(vga|hdmi)_(\d{3,5})[xX](\d{3,5})(?:[_\-].*)?$",
    flags=re.IGNORECASE,
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LR_MARKER_REVIEW_FILE = REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/lr_marker_vendor_template_library/review_decisions.json"
DEFAULT_ORIENTATION_MARKER_BUNDLE_ZIP = Path("/Users/Shared/41_orientation_marker_detector_bundle.zip")
DEFAULT_ORIENTATION_MARKER_BUNDLE_DIR = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle"
DEFAULT_ORIENTATION_MARKER_BUNDLE_LIBRARY_ROOT = (
    DEFAULT_ORIENTATION_MARKER_BUNDLE_DIR / "orientation_marker_detector" / "templates"
)


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_json_obj(value: str) -> Dict[str, Any]:
    txt = str(value or "").strip()
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value.strip())
    out = out.strip("_")
    return out or "folder"


def _emit_event(event: str, **payload: object) -> None:
    obj = {
        "event": event,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    obj.update(payload)
    print(f"##EVENT {json.dumps(obj, ensure_ascii=False)}", flush=True)


def _pick_python(default_python: str) -> str:
    candidate = REPO_ROOT / "OldSoftwareEsiBuilder/.venv-mps/bin/python"
    if candidate.is_file():
        return candidate.as_posix()
    return default_python


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except Exception:
        return False


def _load_excluded_image_rels(exclude_images_file: Optional[Path], input_folder: Path) -> List[str]:
    if exclude_images_file is None:
        return []
    path = exclude_images_file.expanduser().resolve()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = path.read_text(encoding="utf-8").splitlines()
    if isinstance(raw, dict):
        items = raw.get("excluded_images_rel", raw.get("excluded", []))
    else:
        items = raw
    if not isinstance(items, list):
        return []

    out: List[str] = []
    seen = set()
    for item in items:
        txt = str(item or "").strip()
        if not txt:
            continue
        rel_txt = txt
        try:
            p = Path(txt).expanduser()
            if p.is_absolute():
                rel_txt = p.resolve().relative_to(input_folder).as_posix()
        except Exception:
            rel_txt = txt
        rel_txt = rel_txt.replace("\\", "/").lstrip("/")
        if not rel_txt or rel_txt.startswith("../") or "/../" in rel_txt:
            continue
        if rel_txt not in seen:
            seen.add(rel_txt)
            out.append(rel_txt)
    return out


def _create_filtered_input_symlinks(input_folder: Path, input_ref_folder: Path, excluded_rels: Sequence[str]) -> Dict[str, Any]:
    excluded = {str(x or "").replace("\\", "/").lstrip("/") for x in excluded_rels if str(x or "").strip()}
    raw_images = _collect_acquisition_images(input_folder)
    included = []
    excluded_existing = []
    for path in raw_images:
        try:
            rel = path.relative_to(input_folder).as_posix()
        except Exception:
            rel = path.name
        if rel in excluded:
            excluded_existing.append(rel)
        else:
            included.append((path, rel))
    if not included:
        raise RuntimeError("Tutte le immagini della cartella risultano escluse: modifica la lista esclusioni.")

    input_ref_folder.mkdir(parents=True, exist_ok=False)
    for src, rel in included:
        dst = input_ref_folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
        except Exception:
            # Fallback conservativo se il filesystem non consente symlink su file.
            import shutil as _shutil

            _shutil.copy2(src, dst)

    return {
        "mode": "filtered_symlinks",
        "raw_total": int(len(raw_images)),
        "included": int(len(included)),
        "excluded_existing": int(len(excluded_existing)),
        "excluded_images_rel": sorted(excluded),
        "excluded_existing_rel": sorted(excluded_existing),
    }


def _load_first_prediction(csv_path: Path) -> Tuple[Optional[Dict[str, str]], int]:
    first: Optional[Dict[str, str]] = None
    rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            if first is None:
                first = {str(k): str(v) for k, v in row.items()}
    return first, rows


def _build_step_checks(
    row: Dict[str, str],
    vendor_min_conf: float,
    probe_min_conf: float,
    lt_min_conf: float,
    probe_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    raw = _safe_int(row.get("images_total_raw", "0"), 0)
    unique = _safe_int(row.get("images_total", "0"), 0)
    dup_removed = _safe_int(row.get("images_duplicates_removed", "0"), 0)

    rotation_source = (row.get("rotation_source", "") or "").strip()
    rotation_deg = _safe_int(row.get("rotation_deg_clockwise", "0"), 0)
    rotation_ratio = _safe_float(row.get("rotation_vote_ratio", "0"), 0.0)
    rotation_votes = _safe_int(row.get("rotation_votes_total", "0"), 0)
    rotation_samples = _safe_int(row.get("rotation_samples_checked", "0"), 0)
    rotation_reason = str(row.get("rotation_decision_reason", "") or "").strip()

    vendor_pred = (row.get("vendor_predicted", "") or "").strip()
    vendor_conf = _safe_float(row.get("vendor_confidence", "0"), 0.0)

    rect_line = (row.get("line_11_rect_echo", "") or "").strip()
    rect_ok = bool(rect_line) and bool(__import__("re").match(RECT_ECHO_RE, rect_line))
    rect_source = (row.get("line_11_source", "") or "").strip()
    rect_method = (row.get("line_11_method", "") or "").strip()
    rect_margin = _safe_float(row.get("line_11_rect_red_margin_pct", "0"), 0.0)
    rect_winner = (row.get("line_11_rect_red_winner_group", "") or "").strip()
    su_giu_images = _safe_int(row.get("su_giu_images_predicted", "0"), 0)
    su_giu_source = (row.get("su_giu_source", "") or "").strip()
    lr_marker_images = _safe_int(row.get("lr_marker_images_predicted", "0"), 0)
    lr_marker_source = (row.get("lr_marker_source", "") or "").strip()
    lr_marker_method = (row.get("lr_marker_method", "") or "").strip().lower()
    if not lr_marker_method:
        lr_marker_method = "bundle" if lr_marker_source.startswith("bundle") else "classical"
    lr_marker_majority = (row.get("lr_marker_majority_label", "") or "").strip()
    lr_marker_best_label = (row.get("lr_marker_best_label", "") or "").strip()
    lr_marker_best_score = _safe_float(row.get("lr_marker_best_score", "0"), 0.0)
    lr_marker_best_strategy = (row.get("lr_marker_best_search_strategy", "") or "").strip()
    rect_depth_status = (row.get("rect_depth_status", "") or "").strip() or "review"
    rect_depth_images = _safe_int(row.get("rect_depth_images_predicted", "0"), 0)
    rect_depth_accepted = _safe_int(row.get("rect_depth_accepted_count", "0"), 0)
    rect_depth_review = _safe_int(row.get("rect_depth_review_count", "0"), 0)
    rect_depth_reject = _safe_int(row.get("rect_depth_reject_count", "0"), 0)
    rect_depth_missing = _safe_int(row.get("rect_depth_missing_count", "0"), 0)
    rect_depth_ratio = _safe_float(row.get("rect_depth_acceptance_ratio", "0"), 0.0)
    rect_depth_mode = (row.get("rect_depth_majority_mode", "") or "").strip()
    rect_depth_depths = (row.get("rect_depth_unique_depths_json", "") or "").strip()
    rect_depth_source = (row.get("rect_depth_source", "") or "").strip()
    lt_images = _safe_int(row.get("lt_images_predicted", "0"), 0)
    lt_source = (row.get("lt_source", "") or "").strip()
    lt_majority = (row.get("lt_majority_label", "") or "").strip().upper()
    lt_mean_conf = _safe_float(row.get("lt_mean_confidence", "0"), 0.0)

    probe_id = (row.get("line_03_id_probe", "") or "").strip()
    probe_conf = _safe_float(row.get("line_03_probe_confidence", "0"), 0.0)
    probe_name = (probe_name_map or {}).get(probe_id, "")
    probe_type_value = (row.get("line_04_probe_type", "") or "").strip()
    probe_type_source = (row.get("line_04_probe_type_source", "") or "").strip()
    probe_type_strategy = (row.get("line_04_probe_type_strategy", "") or "").strip()
    probe_type_secondary = (row.get("line_04_probe_type_needs_secondary_model", "") or "").strip()
    probe_type_candidates = (row.get("line_04_probe_type_candidate_ints", "") or "").strip()

    status = (row.get("status", "") or "").strip()
    reasons = (row.get("review_reasons", "") or "").strip()

    steps: List[Dict[str, object]] = []
    steps.append(
        {
            "step": "deduplicazione",
            "status": "ok" if unique > 0 else "error",
            "raw_images": raw,
            "unique_images": unique,
            "duplicates_removed": dup_removed,
        }
    )
    steps.append(
        {
            "step": "rotazione",
            "status": (
                "review"
                if rotation_source in {"osd_unavailable", "osd_no_votes", "osd_low_support"}
                else "ok"
            ),
            "rotation_deg_clockwise": rotation_deg,
            "rotation_source": rotation_source,
            "vote_ratio": rotation_ratio,
            "votes_total": rotation_votes,
            "samples_checked": rotation_samples,
            "decision_reason": rotation_reason,
        }
    )
    steps.append(
        {
            "step": "vendor",
            "status": "ok" if vendor_pred and vendor_conf >= vendor_min_conf else "review",
            "vendor_predicted": vendor_pred,
            "vendor_confidence": vendor_conf,
            "threshold": float(vendor_min_conf),
        }
    )
    steps.append(
        {
            "step": "rect",
            "status": "ok" if rect_ok else "error",
            "line_11_rect_echo": rect_line,
            "line_11_source": rect_source,
            "line_11_method": rect_method,
            "line_11_rect_red_margin_pct": rect_margin,
            "line_11_rect_red_winner_group": rect_winner,
        }
    )
    steps.append(
        {
            "step": "orientamento_su_giu_per_frame",
            "status": (
                "ok"
                if su_giu_images > 0
                else ("ok" if su_giu_source == "disabled" else "review")
            ),
            "images_predicted": su_giu_images,
            "source": su_giu_source,
            "mode": "per_frame_only",
        }
    )
    steps.append(
        {
            "step": "orientamento_lr_marker_bundle" if lr_marker_method == "bundle" else "orientamento_lr_marker_classico",
            "status": (
                "ok"
                if lr_marker_images > 0
                else ("ok" if lr_marker_source == "disabled" else "review")
            ),
            "images_predicted": lr_marker_images,
            "majority_label": lr_marker_majority,
            "best_label": lr_marker_best_label,
            "best_score": lr_marker_best_score,
            "best_search_strategy": lr_marker_best_strategy,
            "source": lr_marker_source,
            "mode": (
                "bundle_vendor_template_official_rect_axes"
                if lr_marker_method == "bundle"
                else "classical_vendor_template_best_score"
            ),
        }
    )
    steps.append(
        {
            "step": "orientamento_lt_per_frame",
            "status": (
                "ok"
                if (lt_images > 0 and lt_mean_conf >= float(lt_min_conf))
                else ("ok" if lt_source == "disabled" else "review")
            ),
            "images_predicted": lt_images,
            "majority_label": lt_majority,
            "mean_confidence": lt_mean_conf,
            "threshold": float(lt_min_conf),
            "source": lt_source,
            "mode": "per_frame_only",
        }
    )
    steps.append(
        {
            "step": "rect_depth_autonomous",
            "status": rect_depth_status if rect_depth_source != "disabled" else "ok",
            "images_predicted": rect_depth_images,
            "accepted": rect_depth_accepted,
            "review": rect_depth_review,
            "reject": rect_depth_reject,
            "missing": rect_depth_missing,
            "accepted_ratio": rect_depth_ratio,
            "majority_mode": rect_depth_mode,
            "depths_mm": rect_depth_depths,
            "source": rect_depth_source,
            "mode": "ocr_classical_ranker_per_frame",
        }
    )
    steps.append(
        {
            "step": "probe",
            "status": "ok" if probe_id and probe_conf >= probe_min_conf else "review",
            "line_03_id_probe": probe_id,
            "line_03_probe_name": probe_name,
            "line_03_probe_confidence": probe_conf,
            "threshold": float(probe_min_conf),
        }
    )
    steps.append(
        {
            "step": "probe_type_router_line4",
            "status": (
                "ok"
                if probe_type_value and probe_type_value.upper() not in {"UNKNOWN", "3-4"}
                else ("ok" if probe_type_source == "router_disabled" else "review")
            ),
            "line_04_probe_type": probe_type_value,
            "line_04_probe_type_source": probe_type_source,
            "line_04_probe_type_strategy": probe_type_strategy,
            "line_04_probe_type_needs_secondary_model": probe_type_secondary,
            "line_04_probe_type_candidate_ints": probe_type_candidates,
        }
    )
    steps.append(
        {
            "step": "pipeline_status",
            "status": status or "review",
            "review_reasons": reasons,
        }
    )
    return steps


def _select_uniform_subset(paths: List[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]

    total = len(paths)
    selected_idx: List[int] = []
    seen = set()
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)
    if len(selected_idx) < limit:
        for idx in range(total):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= limit:
                break
    selected_idx.sort()
    return [paths[i] for i in selected_idx]


def _collect_preview_images(folder: Path, limit: int = 8) -> List[Path]:
    paths_all: List[Path] = []
    paths_pattern: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paths_all.append(path)
        if CAPTURE_FILENAME_PATTERN_RE.match(path.stem):
            paths_pattern.append(path)
    selected = paths_pattern if paths_pattern else paths_all
    selected.sort()
    return _select_uniform_subset(selected, limit=max(1, int(limit)))


def _collect_acquisition_images(folder: Path) -> List[Path]:
    paths_all: List[Path] = []
    paths_pattern: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paths_all.append(path)
        if CAPTURE_FILENAME_PATTERN_RE.match(path.stem):
            paths_pattern.append(path)
    selected = paths_pattern if paths_pattern else paths_all
    selected.sort()
    return selected


def _sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _analyze_duplicates(paths: List[Path]) -> Tuple[List[Path], List[Dict[str, Any]]]:
    seen: Dict[Tuple[int, str], Path] = {}
    unique_paths: List[Path] = []
    removed: List[Dict[str, Any]] = []
    for path in paths:
        try:
            size = int(path.stat().st_size)
            sha1 = _sha1_file(path)
        except OSError:
            unique_paths.append(path)
            continue
        key = (size, sha1)
        if key in seen:
            removed.append(
                {
                    "kept_path": seen[key],
                    "removed_path": path,
                    "sha1": sha1,
                    "size_bytes": size,
                }
            )
            continue
        seen[key] = path
        unique_paths.append(path)
    return unique_paths, removed


def _save_image_preview(
    *,
    source: Path,
    target: Path,
    rotate_deg_clockwise: int = 0,
    max_side: int = 1200,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.copy()
        if int(rotate_deg_clockwise) % 360 != 0:
            out = out.rotate(-int(rotate_deg_clockwise), expand=True)
        out = out.convert("RGB")
        if max(out.size) > int(max_side):
            out.thumbnail((int(max_side), int(max_side)))
        out.save(target, format="PNG")


def _parse_rect_coords(value: str) -> Optional[Tuple[int, int, int, int]]:
    txt = str(value or "").strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split("|") if p.strip()]
    if len(parts) < 4:
        return None
    try:
        top = int(float(parts[0]))
        left = int(float(parts[1]))
        bottom = int(float(parts[2]))
        right = int(float(parts[3]))
    except Exception:
        return None
    return (top, left, bottom, right)


def _save_rect_overlay_preview(
    *,
    source: Path,
    target: Path,
    rect_coords: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 32, 32),
    max_side: int = 1200,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.convert("RGB")
        w, h = out.size
        top, left, bottom, right = rect_coords
        top = max(0, min(int(top), h - 1))
        bottom = max(0, min(int(bottom), h - 1))
        left = max(0, min(int(left), w - 1))
        right = max(0, min(int(right), w - 1))
        if bottom < top:
            top, bottom = bottom, top
        if right < left:
            left, right = right, left

        scale = 1.0
        if max(out.size) > int(max_side):
            scale = float(max_side) / float(max(out.size))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out = out.resize((nw, nh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(out)
        x1 = int(round(left * scale))
        y1 = int(round(top * scale))
        x2 = int(round(right * scale))
        y2 = int(round(bottom * scale))
        width = max(2, int(round(max(out.size) / 250)))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)
        out.save(target, format="PNG")


def _clip_rect_coords_tlbr(
    rect_coords: Tuple[int, int, int, int],
    *,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    top, left, bottom, right = rect_coords
    top = max(0, min(int(top), image_h - 1))
    bottom = max(0, min(int(bottom), image_h - 1))
    left = max(0, min(int(left), image_w - 1))
    right = max(0, min(int(right), image_w - 1))
    if bottom < top:
        top, bottom = bottom, top
    if right < left:
        left, right = right, left
    return top, left, bottom, right


def _save_line13_compare_overlay_preview(
    *,
    source: Path,
    target: Path,
    final_rect_coords: Tuple[int, int, int, int],
    pre_dark_trim_rect_coords: Optional[Tuple[int, int, int, int]] = None,
    max_side: int = 1200,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.convert("RGB")
        w, h = out.size
        final_rect = _clip_rect_coords_tlbr(final_rect_coords, image_w=w, image_h=h)
        pre_rect = (
            _clip_rect_coords_tlbr(pre_dark_trim_rect_coords, image_w=w, image_h=h)
            if pre_dark_trim_rect_coords is not None
            else None
        )

        scale = 1.0
        if max(out.size) > int(max_side):
            scale = float(max_side) / float(max(out.size))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out = out.resize((nw, nh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(out)
        width_px = max(2, int(round(max(out.size) / 250)))

        if pre_rect is not None:
            px1 = int(round(pre_rect[1] * scale))
            py1 = int(round(pre_rect[0] * scale))
            px2 = int(round(pre_rect[3] * scale))
            py2 = int(round(pre_rect[2] * scale))
            draw.rectangle([(px1, py1), (px2, py2)], outline=(255, 219, 77), width=width_px)

        fx1 = int(round(final_rect[1] * scale))
        fy1 = int(round(final_rect[0] * scale))
        fx2 = int(round(final_rect[3] * scale))
        fy2 = int(round(final_rect[2] * scale))
        draw.rectangle([(fx1, fy1), (fx2, fy2)], outline=(32, 128, 255), width=width_px)

        legend_items: List[Tuple[Tuple[int, int, int], str]] = []
        if pre_rect is not None:
            legend_items.append(((255, 219, 77), "giallo = box prima del fine tuning"))
        legend_items.append(((32, 128, 255), "blu = box finale dopo fine tuning"))
        rows = [(c, str(t or "").strip()) for c, t in legend_items if str(t or "").strip()]
        if not rows:
            out.save(target, format="PNG")
            return

        legend_pad = 6
        legend_line_h = 15
        footer_pad = 10
        max_len = max(len(txt) for _, txt in rows)
        legend_w = max(220, 30 + max_len * 7)
        legend_h = legend_pad * 2 + legend_line_h * len(rows)
        canvas_w = max(out.size[0], legend_w + footer_pad * 2)
        canvas_h = out.size[1] + legend_h + footer_pad * 2
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(10, 14, 20))
        image_x = max(0, (canvas_w - out.size[0]) // 2)
        canvas.paste(out, (image_x, 0))

        draw_canvas = ImageDraw.Draw(canvas)
        legend_x = max(0, (canvas_w - legend_w) // 2)
        legend_y = out.size[1] + footer_pad
        _draw_overlay_legend(draw_canvas, items=rows, start_xy=(legend_x, legend_y))
        canvas.save(target, format="PNG")


def _save_rect_crop_preview(
    *,
    source: Path,
    target: Path,
    rect_coords: Tuple[int, int, int, int],
    pad_ratio: float = 0.03,
    min_side: int = 720,
    max_side: int = 1800,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.convert("RGB")
        w, h = out.size
        top, left, bottom, right = rect_coords
        top = max(0, min(int(top), h - 1))
        bottom = max(0, min(int(bottom), h - 1))
        left = max(0, min(int(left), w - 1))
        right = max(0, min(int(right), w - 1))
        if bottom < top:
            top, bottom = bottom, top
        if right < left:
            left, right = right, left
        if bottom == top:
            bottom = min(h - 1, top + 1)
        if right == left:
            right = min(w - 1, left + 1)

        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        pad_x = max(1, int(round(box_w * float(max(0.0, pad_ratio)))))
        pad_y = max(1, int(round(box_h * float(max(0.0, pad_ratio)))))
        x1 = max(0, left - pad_x)
        y1 = max(0, top - pad_y)
        x2 = min(w, right + pad_x)
        y2 = min(h, bottom + pad_y)
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)

        crop = out.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        max_dim = max(cw, ch)
        if max_dim > int(max_side):
            scale_down = float(max_side) / float(max_dim)
            nw = max(1, int(round(cw * scale_down)))
            nh = max(1, int(round(ch * scale_down)))
            crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)
            cw, ch = crop.size

        max_dim = max(cw, ch)
        if max_dim < int(min_side):
            scale_up = float(min_side) / float(max_dim)
            nw = max(1, int(round(cw * scale_up)))
            nh = max(1, int(round(ch * scale_up)))
            crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)

        crop.save(target, format="PNG")


def _clip_norm_rect(rect: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        x = float(rect.get("x", 0.0))
        y = float(rect.get("y", 0.0))
        w = float(rect.get("w", 0.0))
        h = float(rect.get("h", 0.0))
    except Exception:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))
    if (x + w) > 1.0:
        w = 1.0 - x
    if (y + h) > 1.0:
        h = 1.0 - y
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def _norm_rect_to_xyxy(
    rect_norm: Dict[str, Any],
    *,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    rect = _clip_norm_rect(rect_norm)
    if rect is None or image_w <= 1 or image_h <= 1:
        return None
    x1 = float(rect["x"]) * float(image_w)
    y1 = float(rect["y"]) * float(image_h)
    x2 = float(rect["x"] + rect["w"]) * float(image_w)
    y2 = float(rect["y"] + rect["h"]) * float(image_h)
    return _clip_box_xyxy((x1, y1, x2, y2), width=image_w, height=image_h)


def _safe_float_any(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _draw_overlay_legend(
    draw: ImageDraw.ImageDraw,
    *,
    items: Sequence[Tuple[Tuple[int, int, int], str]],
    start_xy: Tuple[int, int] = (10, 10),
) -> None:
    rows = [(c, str(t or "").strip()) for c, t in items if str(t or "").strip()]
    if not rows:
        return
    x0, y0 = int(start_xy[0]), int(start_xy[1])
    pad = 6
    line_h = 15
    sw = 10
    max_len = max(len(txt) for _, txt in rows)
    legend_w = max(220, 30 + max_len * 7)
    legend_h = pad * 2 + line_h * len(rows)
    draw.rectangle(
        [(x0, y0), (x0 + legend_w, y0 + legend_h)],
        fill=(18, 25, 38),
        outline=(210, 218, 230),
        width=1,
    )
    for idx, (color, text) in enumerate(rows):
        yy = y0 + pad + idx * line_h
        draw.rectangle([(x0 + pad, yy + 2), (x0 + pad + sw, yy + 2 + sw)], fill=color, outline=color, width=1)
        draw.text((x0 + pad + sw + 6, yy), text, fill=(245, 248, 252))


def _save_rect_red_overlay_preview(
    *,
    source: Path,
    target: Path,
    rect_base_norm: Optional[Dict[str, Any]],
    rect_red_norm: Optional[Dict[str, Any]],
    segment_norm: Optional[Dict[str, Any]],
    rect_single_norm: Optional[Dict[str, Any]],
    max_side: int = 1200,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.convert("RGB")
        w, h = out.size
        scale = 1.0
        if max(out.size) > int(max_side):
            scale = float(max_side) / float(max(out.size))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out = out.resize((nw, nh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(out)
        width_px = max(2, int(round(max(out.size) / 260)))

        def _draw_rect_norm(rect: Optional[Dict[str, Any]], color: Tuple[int, int, int], *, dashed: bool = False) -> None:
            if not rect:
                return
            box = _norm_rect_to_xyxy(rect, image_w=w, image_h=h)
            if box is None:
                return
            x1 = int(round(box[0] * scale))
            y1 = int(round(box[1] * scale))
            x2 = int(round(box[2] * scale))
            y2 = int(round(box[3] * scale))
            if dashed:
                dash = max(5, width_px * 2)
                gap = max(4, width_px + 1)
                for x in range(x1, x2, dash + gap):
                    draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width_px)
                    draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width_px)
                for y in range(y1, y2, dash + gap):
                    draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width_px)
                    draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width_px)
            else:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width_px)

        _draw_rect_norm(rect_base_norm, (66, 66, 66))
        _draw_rect_norm(rect_single_norm, (38, 130, 255), dashed=True)
        _draw_rect_norm(rect_red_norm, (220, 40, 40))

        base_box = _norm_rect_to_xyxy(rect_base_norm or {}, image_w=w, image_h=h) if rect_base_norm else None
        if base_box is not None:
            cx = int(round(((base_box[0] + base_box[2]) * 0.5) * scale))
            cy = int(round(((base_box[1] + base_box[3]) * 0.5) * scale))
            x1b = int(round(base_box[0] * scale))
            y1b = int(round(base_box[1] * scale))
            x2b = int(round(base_box[2] * scale))
            y2b = int(round(base_box[3] * scale))
            draw.line([(0, cy), (out.size[0] - 1, cy)], fill=(110, 110, 110), width=max(1, width_px - 1))
            draw.line([(cx, 0), (cx, out.size[1] - 1)], fill=(110, 110, 110), width=max(1, width_px - 1))
            draw.ellipse(
                [(cx - max(3, width_px), cy - max(3, width_px)), (cx + max(3, width_px), cy + max(3, width_px))],
                outline=(66, 66, 66),
                fill=(66, 66, 66),
                width=1,
            )

        if segment_norm and w > 1 and h > 1:
            x1n = _safe_float_any(segment_norm.get("x1"), 0.0)
            x2n = _safe_float_any(segment_norm.get("x2"), 0.0)
            yn = _safe_float_any(segment_norm.get("y"), 0.0)
            x1n = max(0.0, min(1.0, x1n))
            x2n = max(0.0, min(1.0, x2n))
            yn = max(0.0, min(1.0, yn))
            if x2n < x1n:
                x1n, x2n = x2n, x1n
            px1 = int(round((x1n * float(w)) * scale))
            px2 = int(round((x2n * float(w)) * scale))
            py = int(round((yn * float(h)) * scale))
            draw.line([(px1, py), (px2, py)], fill=(255, 219, 77), width=max(2, width_px + 1))
            draw.ellipse(
                [(px1 - 3, py - 3), (px1 + 3, py + 3)],
                outline=(255, 219, 77),
                fill=(255, 219, 77),
            )
            draw.ellipse(
                [(px2 - 3, py - 3), (px2 + 3, py + 3)],
                outline=(255, 219, 77),
                fill=(255, 219, 77),
            )
            pm = int(round((((x1n + x2n) * 0.5) * float(w)) * scale))
            draw.ellipse(
                [(pm - 4, py - 4), (pm + 4, py + 4)],
                outline=(34, 197, 94),
                fill=(34, 197, 94),
            )

        red_box = _norm_rect_to_xyxy(rect_red_norm or {}, image_w=w, image_h=h) if rect_red_norm else None
        if red_box is not None:
            rcx = int(round(((red_box[0] + red_box[2]) * 0.5) * scale))
            rcy = int(round(((red_box[1] + red_box[3]) * 0.5) * scale))
            draw.ellipse(
                [(rcx - max(3, width_px), rcy - max(3, width_px)), (rcx + max(3, width_px), rcy + max(3, width_px))],
                outline=(220, 40, 40),
                fill=(220, 40, 40),
                width=1,
            )

        single_box = _norm_rect_to_xyxy(rect_single_norm or {}, image_w=w, image_h=h) if rect_single_norm else None
        if single_box is not None:
            scx = int(round(((single_box[0] + single_box[2]) * 0.5) * scale))
            scy = int(round(((single_box[1] + single_box[3]) * 0.5) * scale))
            draw.ellipse(
                [(scx - max(3, width_px), scy - max(3, width_px)), (scx + max(3, width_px), scy + max(3, width_px))],
                outline=(38, 130, 255),
                fill=(38, 130, 255),
                width=1,
            )

        _draw_overlay_legend(
            draw,
            items=[
                ((66, 66, 66), "grigio = rect base mediana"),
                ((38, 130, 255), "blu tratteggiato = rect per immagine"),
                ((220, 40, 40), "rosso = rect da segmento"),
                ((255, 219, 77), "giallo = segmento top"),
                ((34, 197, 94), "verde = punto medio segmento"),
            ],
        )

        out.save(target, format="PNG")


def _normalize_rotation_deg_clockwise(value: int) -> int:
    value_i = int(value) % 360
    if value_i in {0, 90, 180, 270}:
        return value_i
    return 0


def _inverse_rotate_point_clockwise(
    x_rot: float,
    y_rot: float,
    original_width: int,
    original_height: int,
    rotate_deg_clockwise: int,
) -> Tuple[float, float]:
    rotate = _normalize_rotation_deg_clockwise(rotate_deg_clockwise)
    ow = float(original_width)
    oh = float(original_height)

    if rotate == 0:
        return x_rot, y_rot
    if rotate == 90:
        return y_rot, oh - x_rot
    if rotate == 180:
        return ow - x_rot, oh - y_rot
    if rotate == 270:
        return ow - y_rot, x_rot
    return x_rot, y_rot


def _rotate_box_back_to_original_coords(
    box_rotated: Tuple[float, float, float, float],
    original_width: int,
    original_height: int,
    rotate_deg_clockwise: int,
) -> Tuple[float, float, float, float]:
    rotate = _normalize_rotation_deg_clockwise(rotate_deg_clockwise)
    if rotate == 0:
        return box_rotated

    x1, y1, x2, y2 = box_rotated
    points_rot = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    points_original = [
        _inverse_rotate_point_clockwise(
            x_rot=px,
            y_rot=py,
            original_width=original_width,
            original_height=original_height,
            rotate_deg_clockwise=rotate,
        )
        for px, py in points_rot
    ]
    xs = [p[0] for p in points_original]
    ys = [p[1] for p in points_original]
    x1_o = min(xs)
    y1_o = min(ys)
    x2_o = max(xs)
    y2_o = max(ys)

    x1_o = max(0.0, min(x1_o, float(original_width) - 1.0))
    y1_o = max(0.0, min(y1_o, float(original_height) - 1.0))
    x2_o = max(x1_o + 1.0, min(x2_o, float(original_width)))
    y2_o = max(y1_o + 1.0, min(y2_o, float(original_height)))
    return x1_o, y1_o, x2_o, y2_o


def _load_rect_tensor_for_inference(
    path: Path,
    image_size: int,
    rotate_deg_clockwise: int,
):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    with Image.open(path) as img:
        image = img.convert("RGB")
        original_width, original_height = image.size
        rotate = _normalize_rotation_deg_clockwise(rotate_deg_clockwise)
        if rotate:
            image = image.rotate(-rotate, expand=True)
        width, height = image.size
        image = TF.resize(
            image,
            size=[int(image_size), int(image_size)],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    return tensor, int(width), int(height), int(original_width), int(original_height)


def _choose_torch_device():
    import torch

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_checkpoint_path(checkpoint_raw: str) -> Optional[Path]:
    txt = str(checkpoint_raw or "").strip()
    if not txt:
        return None
    direct = Path(txt).expanduser()
    if direct.is_file():
        return direct.resolve()
    repo_relative = (REPO_ROOT / direct).resolve()
    if repo_relative.is_file():
        return repo_relative
    return None


def _clip_box_xyxy(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w_f = float(max(1, int(width)))
    h_f = float(max(1, int(height)))
    x1 = max(0.0, min(float(x1), w_f - 1.0))
    y1 = max(0.0, min(float(y1), h_f - 1.0))
    x2 = max(x1 + 1.0, min(float(x2), w_f))
    y2 = max(y1 + 1.0, min(float(y2), h_f))
    return x1, y1, x2, y2


def _scale_global_rect_to_image_xyxy(
    rect_tlbr: Tuple[int, int, int, int],
    ref_w: int,
    ref_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    top, left, bottom, right = rect_tlbr
    x1 = float(left)
    y1 = float(top)
    x2 = float(right)
    y2 = float(bottom)
    if ref_w > 0 and ref_h > 0:
        sx = float(image_w) / float(ref_w)
        sy = float(image_h) / float(ref_h)
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
    return _clip_box_xyxy((x1, y1, x2, y2), width=image_w, height=image_h)


def _xyxy_to_tlbr_dict(box: Tuple[float, float, float, float]) -> Dict[str, int]:
    x1, y1, x2, y2 = box
    return {
        "top": int(round(y1)),
        "left": int(round(x1)),
        "bottom": int(round(y2)),
        "right": int(round(x2)),
    }


def _box_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = _box_area_xyxy(a) + _box_area_xyxy(b) - inter_area
    if union <= 1e-8:
        return 0.0
    return float(inter_area / union)


def _save_rect_dual_overlay_preview(
    *,
    source: Path,
    target: Path,
    single_rect_xyxy: Tuple[float, float, float, float],
    global_rect_xyxy: Tuple[float, float, float, float],
    max_side: int = 1200,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        out = im.convert("RGB")
        w, h = out.size
        single = _clip_box_xyxy(single_rect_xyxy, width=w, height=h)
        global_box = _clip_box_xyxy(global_rect_xyxy, width=w, height=h)

        scale = 1.0
        if max(out.size) > int(max_side):
            scale = float(max_side) / float(max(out.size))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out = out.resize((nw, nh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(out)
        width_px = max(2, int(round(max(out.size) / 250)))

        gx1 = int(round(global_box[0] * scale))
        gy1 = int(round(global_box[1] * scale))
        gx2 = int(round(global_box[2] * scale))
        gy2 = int(round(global_box[3] * scale))
        sx1 = int(round(single[0] * scale))
        sy1 = int(round(single[1] * scale))
        sx2 = int(round(single[2] * scale))
        sy2 = int(round(single[3] * scale))

        # Requested visual convention:
        # - yellow: per-image rect
        # - red: folder-global rect
        draw.rectangle([(sx1, sy1), (sx2, sy2)], outline=(255, 219, 77), width=width_px)
        draw.rectangle([(gx1, gy1), (gx2, gy2)], outline=(220, 40, 40), width=width_px)

        # Axes and median points for better interpretability.
        gcx = int(round(((global_box[0] + global_box[2]) * 0.5) * scale))
        gcy = int(round(((global_box[1] + global_box[3]) * 0.5) * scale))
        scx = int(round(((single[0] + single[2]) * 0.5) * scale))
        scy = int(round(((single[1] + single[3]) * 0.5) * scale))
        draw.line([(0, gcy), (out.size[0] - 1, gcy)], fill=(220, 40, 40), width=max(1, width_px - 1))
        draw.line([(gcx, 0), (gcx, out.size[1] - 1)], fill=(220, 40, 40), width=max(1, width_px - 1))
        r_big = max(3, width_px + 1)
        r_small = max(2, width_px)
        draw.ellipse([(gcx - r_big, gcy - r_big), (gcx + r_big, gcy + r_big)], outline=(220, 40, 40), fill=(220, 40, 40), width=1)
        draw.ellipse([(scx - r_small, scy - r_small), (scx + r_small, scy + r_small)], outline=(255, 219, 77), fill=(255, 219, 77), width=1)
        _draw_overlay_legend(
            draw,
            items=[
                ((255, 219, 77), "giallo = rect singolo"),
                ((220, 40, 40), "rosso = rect globale"),
                ((220, 40, 40), "assi rossi = centro rect globale"),
                ((255, 219, 77), "punto giallo = centro rect singolo"),
                ((220, 40, 40), "punto rosso = centro rect globale"),
            ],
        )
        out.save(target, format="PNG")


def _predict_rect_boxes_for_images(
    *,
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    batch_size: int,
    rotate_deg_clockwise: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    import torch

    from train_ultrasound_rect_net import RectRegressor, normalize_box_order

    device = _choose_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    image_size = int(checkpoint.get("args", {}).get("image_size", 320))
    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results: List[Dict[str, object]] = []
    effective_batch = max(1, int(batch_size))

    with torch.inference_mode():
        for start in range(0, len(image_paths), effective_batch):
            batch_paths = list(image_paths[start : start + effective_batch])
            tensors = []
            metas: List[Tuple[Path, int, int, int, int]] = []
            for path in batch_paths:
                try:
                    tensor, width, height, original_width, original_height = _load_rect_tensor_for_inference(
                        path=path,
                        image_size=image_size,
                        rotate_deg_clockwise=rotate_deg_clockwise,
                    )
                except Exception:
                    continue
                tensors.append(tensor)
                metas.append((path, width, height, original_width, original_height))
            if not tensors:
                continue

            images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            preds = normalize_box_order(model(images)).detach().cpu().tolist()

            for pred, meta in zip(preds, metas):
                path, width, height, original_width, original_height = meta
                x1 = float(pred[0]) * float(width)
                y1 = float(pred[1]) * float(height)
                x2 = float(pred[2]) * float(width)
                y2 = float(pred[3]) * float(height)
                x1, y1, x2, y2 = _clip_box_xyxy((x1, y1, x2, y2), width=width, height=height)
                x1, y1, x2, y2 = _rotate_box_back_to_original_coords(
                    box_rotated=(x1, y1, x2, y2),
                    original_width=original_width,
                    original_height=original_height,
                    rotate_deg_clockwise=rotate_deg_clockwise,
                )
                x1, y1, x2, y2 = _clip_box_xyxy((x1, y1, x2, y2), width=original_width, height=original_height)
                results.append(
                    {
                        "path": path,
                        "image_width": int(original_width),
                        "image_height": int(original_height),
                        "single_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

    return results, {"device": str(device), "image_size": int(image_size)}


def _run_rect_per_image_worker(
    *,
    images_json_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    rotate_deg_clockwise: int,
    output_json_path: Path,
) -> int:
    if not images_json_path.is_file():
        raise RuntimeError(f"rect worker images json non trovato: {images_json_path}")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"rect worker checkpoint non trovato: {checkpoint_path}")

    raw_items = json.loads(images_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise RuntimeError("rect worker images json non valido (attesa lista path)")
    image_paths: List[Path] = []
    for raw in raw_items:
        txt = str(raw or "").strip()
        if not txt:
            continue
        image_paths.append(Path(txt).expanduser().resolve())

    predictions, model_meta = _predict_rect_boxes_for_images(
        image_paths=image_paths,
        checkpoint_path=checkpoint_path,
        batch_size=max(1, int(batch_size)),
        rotate_deg_clockwise=int(rotate_deg_clockwise),
    )

    serializable: List[Dict[str, object]] = []
    for item in predictions:
        path_obj = item.get("path")
        path_s = path_obj.as_posix() if isinstance(path_obj, Path) else str(path_obj or "")
        serializable.append(
            {
                "path": path_s,
                "image_width": int(item.get("image_width", 0) or 0),
                "image_height": int(item.get("image_height", 0) or 0),
                "single_xyxy": [
                    float(item.get("single_xyxy", [0.0, 0.0, 1.0, 1.0])[0]),
                    float(item.get("single_xyxy", [0.0, 0.0, 1.0, 1.0])[1]),
                    float(item.get("single_xyxy", [0.0, 0.0, 1.0, 1.0])[2]),
                    float(item.get("single_xyxy", [0.0, 0.0, 1.0, 1.0])[3]),
                ],
            }
        )

    payload = {
        "predictions": serializable,
        "meta": {
            "device": str(model_meta.get("device", "") or ""),
            "image_size": int(model_meta.get("image_size", 0) or 0),
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"rect_worker_predictions={len(serializable)}", flush=True)
    return 0


def _predict_rect_boxes_for_images_via_worker(
    *,
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    batch_size: int,
    rotate_deg_clockwise: int,
    run_dir: Path,
    python_bin: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    worker_dir = run_dir / "tmp" / "rect_per_image_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    images_json_path = worker_dir / "images.json"
    output_json_path = worker_dir / "output.json"
    images_payload = [p.as_posix() for p in image_paths]
    images_json_path.write_text(json.dumps(images_payload, ensure_ascii=False), encoding="utf-8")

    cmd = [
        str(python_bin),
        Path(__file__).resolve().as_posix(),
        "--rect-per-image-worker",
        "--rect-worker-images-json",
        images_json_path.as_posix(),
        "--rect-worker-checkpoint",
        checkpoint_path.as_posix(),
        "--rect-worker-batch-size",
        str(max(1, int(batch_size))),
        "--rect-worker-rotation",
        str(int(rotate_deg_clockwise)),
        "--rect-worker-output-json",
        output_json_path.as_posix(),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"rect worker fallito (returncode={proc.returncode}) con python={python_bin}\n{tail}"
        )
    if not output_json_path.is_file():
        raise RuntimeError(f"rect worker output non trovato: {output_json_path}")

    payload = json.loads(output_json_path.read_text(encoding="utf-8"))
    preds_raw = payload.get("predictions", [])
    if not isinstance(preds_raw, list):
        raise RuntimeError("rect worker output non valido (predictions non lista)")

    predictions: List[Dict[str, object]] = []
    for item in preds_raw:
        if not isinstance(item, dict):
            continue
        path_txt = str(item.get("path", "") or "").strip()
        if not path_txt:
            continue
        single_raw = item.get("single_xyxy", [])
        if not isinstance(single_raw, list) or len(single_raw) != 4:
            continue
        predictions.append(
            {
                "path": Path(path_txt),
                "image_width": int(item.get("image_width", 0) or 0),
                "image_height": int(item.get("image_height", 0) or 0),
                "single_xyxy": [
                    float(single_raw[0]),
                    float(single_raw[1]),
                    float(single_raw[2]),
                    float(single_raw[3]),
                ],
            }
        )

    meta_raw = payload.get("meta", {})
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    model_meta = {
        "device": str(meta_raw.get("device", "") or ""),
        "image_size": int(meta_raw.get("image_size", 0) or 0),
    }
    return predictions, model_meta


def _build_rect_per_image_evidence(
    *,
    run_dir: Path,
    row: Dict[str, str],
    unique_images: Sequence[Path],
    acquisition_images: Sequence[Path],
    batch_size: int,
    rotation_deg_clockwise: int,
    python_bin: str,
    generate_images: bool = True,
) -> Dict[str, object]:
    rect_coords = _parse_rect_coords(str(row.get("line_11_rect_echo", "") or ""))
    ref_w = _safe_int(str(row.get("line_09_video_x_size", "") or "0"), 0)
    ref_h = _safe_int(str(row.get("line_10_video_y_size", "") or "0"), 0)
    checkpoint_raw = str(row.get("line_11_model_checkpoint", "") or "").strip()
    checkpoint_path = _resolve_checkpoint_path(checkpoint_raw)
    image_pool = list(unique_images) if unique_images else list(acquisition_images)

    out: Dict[str, object] = {
        "available": False,
        "error": "",
        "images_total": 0,
        "sort": "discrepancy_desc",
        "metric": "1_minus_iou_single_vs_global_scaled",
        "rotation_deg_clockwise": int(rotation_deg_clockwise),
        "model_checkpoint_used": checkpoint_raw,
        "model_checkpoint_resolved": checkpoint_path.as_posix() if checkpoint_path else "",
        "model_device": "",
        "model_image_size": 0,
        "global_rect_reference": {
            "top": int(rect_coords[0]) if rect_coords is not None else 0,
            "left": int(rect_coords[1]) if rect_coords is not None else 0,
            "bottom": int(rect_coords[2]) if rect_coords is not None else 0,
            "right": int(rect_coords[3]) if rect_coords is not None else 0,
            "ref_width": int(ref_w),
            "ref_height": int(ref_h),
        },
        "items": [],
    }

    if rect_coords is None:
        out["error"] = "line_11_rect_echo non disponibile"
        return out
    if not image_pool:
        out["error"] = "nessuna immagine disponibile per analisi rect"
        return out
    if checkpoint_path is None:
        out["error"] = "checkpoint rect usato non risolvibile su filesystem"
        return out

    try:
        predictions, model_meta = _predict_rect_boxes_for_images_via_worker(
            image_paths=image_pool,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            rotate_deg_clockwise=rotation_deg_clockwise,
            run_dir=run_dir,
            python_bin=python_bin,
        )
    except Exception as exc:
        out["error"] = str(exc)
        return out

    if not predictions:
        out["error"] = "inferenza rect per-image non ha prodotto risultati"
        return out

    out["model_device"] = str(model_meta.get("device", "") or "")
    out["model_image_size"] = int(model_meta.get("image_size", 0) or 0)

    overlay_dir = run_dir / "evidence" / "rect_per_image"
    items: List[Dict[str, object]] = []
    for idx, item in enumerate(predictions, start=1):
        path_obj = item.get("path")
        if not isinstance(path_obj, Path):
            continue
        image_w = int(item.get("image_width", 0) or 0)
        image_h = int(item.get("image_height", 0) or 0)
        single_raw = item.get("single_xyxy")
        if (
            not isinstance(single_raw, list)
            or len(single_raw) != 4
            or image_w <= 0
            or image_h <= 0
        ):
            continue
        single_xyxy = _clip_box_xyxy(
            (
                float(single_raw[0]),
                float(single_raw[1]),
                float(single_raw[2]),
                float(single_raw[3]),
            ),
            width=image_w,
            height=image_h,
        )
        global_xyxy = _scale_global_rect_to_image_xyxy(
            rect_tlbr=rect_coords,
            ref_w=ref_w,
            ref_h=ref_h,
            image_w=image_w,
            image_h=image_h,
        )

        iou = _box_iou_xyxy(single_xyxy, global_xyxy)
        discrepancy = max(0.0, 1.0 - iou)
        area_single = _box_area_xyxy(single_xyxy)
        area_global = _box_area_xyxy(global_xyxy)
        size_ratio = float(area_single / area_global) if area_global > 1e-8 else 0.0
        sx = 0.5 * (single_xyxy[0] + single_xyxy[2])
        sy = 0.5 * (single_xyxy[1] + single_xyxy[3])
        gx = 0.5 * (global_xyxy[0] + global_xyxy[2])
        gy = 0.5 * (global_xyxy[1] + global_xyxy[3])
        center_dist = ((sx - gx) ** 2 + (sy - gy) ** 2) ** 0.5
        diag = (float(image_w) ** 2 + float(image_h) ** 2) ** 0.5
        center_dist_norm = float(center_dist / diag) if diag > 1e-8 else 0.0

        overlay_rel = ""
        overlay_error = ""
        if generate_images:
            overlay_path = overlay_dir / f"rect_compare_{idx:04d}.png"
            try:
                _save_rect_dual_overlay_preview(
                    source=path_obj,
                    target=overlay_path,
                    single_rect_xyxy=single_xyxy,
                    global_rect_xyxy=global_xyxy,
                )
                overlay_rel = overlay_path.relative_to(run_dir).as_posix()
            except Exception as exc:
                overlay_error = str(exc)
        else:
            overlay_error = "generated_images_disabled"

        image_rel = ""
        try:
            image_rel = path_obj.relative_to(run_dir).as_posix()
        except Exception:
            image_rel = path_obj.as_posix()
        items.append(
            {
                "rank": 0,
                "image_rel": image_rel,
                "overlay_rel": overlay_rel,
                "overlay_error": overlay_error,
                "image_width": int(image_w),
                "image_height": int(image_h),
                "single_rect": _xyxy_to_tlbr_dict(single_xyxy),
                "global_rect_scaled": _xyxy_to_tlbr_dict(global_xyxy),
                "iou": float(iou),
                "discrepancy": float(discrepancy),
                "center_distance_norm": float(center_dist_norm),
                "area_ratio_single_vs_global": float(size_ratio),
            }
        )

    items = sorted(
        items,
        key=lambda x: (
            -float(x.get("discrepancy", 0.0)),
            float(x.get("iou", 0.0)),
            str(x.get("image_rel", "")),
        ),
    )
    for rank, item in enumerate(items, start=1):
        item["rank"] = int(rank)

    out["available"] = bool(items)
    out["images_total"] = len(items)
    out["items"] = items
    if not items:
        out["error"] = "nessun item rect valido dopo il post-processing"
    return out


def _load_rect_red_folder_payload(
    *,
    pipeline_output: Path,
    pipeline_summary: Dict[str, Any],
    row: Dict[str, str],
) -> Tuple[Dict[str, Any], str]:
    path_txt = str(pipeline_summary.get("rect_red_pipeline_json", "") or "").strip()
    candidate_paths: List[Path] = []
    if path_txt:
        p = Path(path_txt).expanduser()
        candidate_paths.append(p if p.is_absolute() else (pipeline_output / p))
    candidate_paths.append(pipeline_output / "rect_red_pipeline_by_folder.json")

    selected_path: Optional[Path] = None
    for cand in candidate_paths:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        if resolved.is_file():
            selected_path = resolved
            break
    if selected_path is None:
        return {}, ""

    try:
        raw_obj = json.loads(selected_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, selected_path.as_posix()
    if not isinstance(raw_obj, dict):
        return {}, selected_path.as_posix()

    folder_path = str(row.get("folder_path", "") or "").strip()
    folder_name = str(row.get("folder_name", "") or "").strip()
    if folder_path and folder_path in raw_obj and isinstance(raw_obj[folder_path], dict):
        return raw_obj[folder_path], selected_path.as_posix()

    if folder_name:
        for key, value in raw_obj.items():
            if not isinstance(value, dict):
                continue
            if str(value.get("folder_name", "") or "").strip() == folder_name:
                return value, selected_path.as_posix()
            if str(key).rstrip("/").endswith(f"/{folder_name}"):
                return value, selected_path.as_posix()

    if len(raw_obj) == 1:
        only_val = next(iter(raw_obj.values()))
        if isinstance(only_val, dict):
            return only_val, selected_path.as_posix()
    return {}, selected_path.as_posix()


def _norm_rect_iou(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    ar = _clip_norm_rect(a or {})
    br = _clip_norm_rect(b or {})
    if ar is None or br is None:
        return 0.0
    ax1 = float(ar["x"])
    ay1 = float(ar["y"])
    ax2 = float(ar["x"] + ar["w"])
    ay2 = float(ar["y"] + ar["h"])
    bx1 = float(br["x"])
    by1 = float(br["y"])
    bx2 = float(br["x"] + br["w"])
    by2 = float(br["y"] + br["h"])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-8:
        return 0.0
    return float(inter_area / union)


def _build_rect_red_evidence(
    *,
    run_dir: Path,
    pipeline_output: Path,
    pipeline_summary: Dict[str, Any],
    row: Dict[str, str],
    generate_images: bool = True,
) -> Dict[str, Any]:
    payload, src_path = _load_rect_red_folder_payload(
        pipeline_output=pipeline_output,
        pipeline_summary=pipeline_summary,
        row=row,
    )
    out: Dict[str, Any] = {
        "available": False,
        "error": "",
        "source_json_path": src_path,
        "line11_method": str(row.get("line_11_method", "") or ""),
        "line11_red_text": str(row.get("line_11_rect_echo", "") or ""),
        "margin_pct": _safe_float(str(row.get("line_11_rect_red_margin_pct", "0") or "0"), 0.0),
        "winner_group": str(row.get("line_11_rect_red_winner_group", "") or ""),
        "winner_reason": "",
        "group_counts": {"su": 0, "giu": 0, "unknown": 0},
        "top_segment_selected": {},
        "top_segment_su": {},
        "top_segment_giu": {},
        "top_segment_global": {},
        "line11_base": {},
        "line11_red": {},
        "records_total": 0,
        "records_with_segment": 0,
        "items": [],
        "records_source": [],
    }

    if not payload:
        out["error"] = "payload_rect_red_non_disponibile"
        return out
    if not bool(payload.get("available", False)):
        out["error"] = str(payload.get("error", "") or "rect_red_not_available")
        out["margin_pct"] = _safe_float(payload.get("margin_pct", out["margin_pct"]), out["margin_pct"])
        out["winner_group"] = str(payload.get("winner_group", out["winner_group"]) or "")
        return out

    base_obj = payload.get("line11_base", {})
    red_obj = payload.get("line11_red", {})
    base_norm = {}
    red_norm = {}
    if isinstance(base_obj, dict):
        out["line11_base"] = base_obj
        r0 = base_obj.get("rect_norm", {})
        if isinstance(r0, dict):
            base_norm = r0
    if isinstance(red_obj, dict):
        out["line11_red"] = red_obj
        r1 = red_obj.get("rect_norm", {})
        if isinstance(r1, dict):
            red_norm = r1
        txt = str(red_obj.get("text", "") or "").strip()
        if txt:
            out["line11_red_text"] = txt

    out["margin_pct"] = _safe_float(payload.get("margin_pct", out["margin_pct"]), out["margin_pct"])
    out["winner_group"] = str(payload.get("winner_group", out["winner_group"]) or "")
    out["winner_reason"] = str(payload.get("winner_reason", "") or "")
    gc = payload.get("group_counts", {})
    if isinstance(gc, dict):
        out["group_counts"] = {
            "su": int(gc.get("su", 0) or 0),
            "giu": int(gc.get("giu", 0) or 0),
            "unknown": int(gc.get("unknown", 0) or 0),
        }
    out["top_segment_selected"] = payload.get("top_segment_selected", {}) if isinstance(payload.get("top_segment_selected"), dict) else {}
    out["top_segment_su"] = payload.get("top_segment_su", {}) if isinstance(payload.get("top_segment_su"), dict) else {}
    out["top_segment_giu"] = payload.get("top_segment_giu", {}) if isinstance(payload.get("top_segment_giu"), dict) else {}
    out["top_segment_global"] = payload.get("top_segment_global", {}) if isinstance(payload.get("top_segment_global"), dict) else {}
    out["records_total"] = int(payload.get("records_total", 0) or 0)
    out["records_with_segment"] = int(payload.get("records_with_segment", 0) or 0)

    top_seg = out["top_segment_selected"].get("segment", {}) if isinstance(out["top_segment_selected"], dict) else {}
    seg_norm = top_seg if isinstance(top_seg, dict) else {}

    records_raw = payload.get("records", [])
    if not isinstance(records_raw, list):
        records_raw = []
    overlay_dir = run_dir / "evidence" / "rect_red"
    items: List[Dict[str, Any]] = []
    records_source: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records_raw, start=1):
        if not isinstance(rec, dict):
            continue
        image_path_s = str(rec.get("image_path", "") or "").strip()
        if not image_path_s:
            continue
        img_path = Path(image_path_s).expanduser()
        if not img_path.is_file():
            try:
                img_path = img_path.resolve()
            except Exception:
                pass
        if not img_path.is_file():
            continue

        image_rel = ""
        try:
            image_rel = img_path.resolve().relative_to(run_dir).as_posix()
        except Exception:
            image_rel = img_path.as_posix()

        pred_rect_norm = rec.get("pred_rect_norm", {})
        if not isinstance(pred_rect_norm, dict):
            pred_rect_norm = {}
        overlay_rel = ""
        overlay_error = ""
        if generate_images:
            overlay_path = overlay_dir / f"rect_red_{idx:04d}.png"
            try:
                _save_rect_red_overlay_preview(
                    source=img_path,
                    target=overlay_path,
                    rect_base_norm=base_norm if isinstance(base_norm, dict) else None,
                    rect_red_norm=red_norm if isinstance(red_norm, dict) else None,
                    segment_norm=seg_norm if isinstance(seg_norm, dict) else None,
                    rect_single_norm=pred_rect_norm if isinstance(pred_rect_norm, dict) else None,
                )
                overlay_rel = overlay_path.relative_to(run_dir).as_posix()
            except Exception as exc:
                overlay_error = str(exc)
        else:
            overlay_error = "generated_images_disabled"

        iou_single_vs_base = _norm_rect_iou(pred_rect_norm if isinstance(pred_rect_norm, dict) else None, base_norm)
        discrepancy = max(0.0, 1.0 - float(iou_single_vs_base))
        item = {
            "rank": 0,
            "image_rel": image_rel,
            "overlay_rel": overlay_rel,
            "overlay_error": overlay_error,
            "orientation_label": str(rec.get("orientation_label", "") or ""),
            "image_width": int(rec.get("image_width", 0) or 0),
            "image_height": int(rec.get("image_height", 0) or 0),
            "pred_rect_norm": pred_rect_norm,
            "pred_rect_tlbr": rec.get("pred_rect_tlbr", {}) if isinstance(rec.get("pred_rect_tlbr"), dict) else {},
            "auto_segment": rec.get("auto_segment", {}) if isinstance(rec.get("auto_segment"), dict) else {},
            "iou_single_vs_base": float(iou_single_vs_base),
            "discrepancy_vs_base": float(discrepancy),
        }
        items.append(item)
        records_source.append(
            {
                "image_rel": image_rel,
                "image_width": int(rec.get("image_width", 0) or 0),
                "image_height": int(rec.get("image_height", 0) or 0),
                "orientation_label": str(rec.get("orientation_label", "") or ""),
                "pred_rect_norm": pred_rect_norm,
            }
        )

    items.sort(
        key=lambda x: (
            -float(x.get("discrepancy_vs_base", 0.0)),
            str(x.get("image_rel", "")),
        )
    )
    for rank, item in enumerate(items, start=1):
        item["rank"] = int(rank)

    out["items"] = items
    out["records_source"] = records_source
    out["available"] = bool(items)
    if not items:
        out["error"] = "rect_red_records_non_disponibili"
    return out


def _build_su_giu_per_image_evidence(
    *,
    run_dir: Path,
    pipeline_output: Path,
) -> Dict[str, object]:
    csv_path = pipeline_output / "su_giu_per_image_predictions.csv"
    out: Dict[str, object] = {
        "available": False,
        "error": "",
        "csv_path": csv_path.as_posix(),
        "lr_marker_method": "",
        "images_total": 0,
        "label_counts": {"su": 0, "giu": 0, "other": 0},
        "items": [],
        "note": "Etichettatura per singolo frame (nessun giudizio globale di cartella).",
    }
    if not csv_path.is_file():
        out["error"] = "file su_giu_per_image_predictions.csv non trovato"
        return out

    items: List[Dict[str, object]] = []
    counts = {"su": 0, "giu": 0, "other": 0}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_path_s = str(row.get("image_path", "") or "").strip()
                pred_label = str(row.get("pred_label", "") or "").strip().lower()
                image_idx = _safe_int(str(row.get("image_index", "") or "0"), 0)
                conf = _safe_float(str(row.get("confidence", "") or "0"), 0.0)
                p_su = _safe_float(str(row.get("prob_su", "") or "0"), 0.0)
                p_giu = _safe_float(str(row.get("prob_giu", "") or "0"), 0.0)
                crop_top = _safe_int(str(row.get("crop_top", "") or "0"), 0)
                crop_left = _safe_int(str(row.get("crop_left", "") or "0"), 0)
                crop_bottom = _safe_int(str(row.get("crop_bottom", "") or "0"), 0)
                crop_right = _safe_int(str(row.get("crop_right", "") or "0"), 0)
                crop_source = str(row.get("crop_source", "") or "").strip()

                image_rel = image_path_s
                if image_path_s:
                    try:
                        # Keep symlinked run-local paths stable for UI preview routing.
                        image_rel = Path(image_path_s).expanduser().relative_to(run_dir).as_posix()
                    except Exception:
                        try:
                            image_rel = Path(image_path_s).expanduser().resolve().relative_to(run_dir).as_posix()
                        except Exception:
                            image_rel = image_path_s

                if pred_label == "su":
                    counts["su"] += 1
                elif pred_label == "giu":
                    counts["giu"] += 1
                else:
                    counts["other"] += 1

                items.append(
                    {
                        "image_index": int(image_idx),
                        "image_rel": image_rel,
                        "pred_label": pred_label,
                        "confidence": float(conf),
                        "prob_su": float(p_su),
                        "prob_giu": float(p_giu),
                        "crop_top": int(crop_top),
                        "crop_left": int(crop_left),
                        "crop_bottom": int(crop_bottom),
                        "crop_right": int(crop_right),
                        "crop_source": crop_source,
                    }
                )
    except Exception as exc:
        out["error"] = str(exc)
        return out

    items.sort(
        key=lambda x: (
            int(x.get("image_index", 0)),
            str(x.get("image_rel", "")),
        )
    )
    out["items"] = items
    out["images_total"] = len(items)
    out["label_counts"] = counts
    out["available"] = bool(items)
    if not items:
        out["error"] = "nessuna riga valida nel CSV su_giu_per_image_predictions.csv"
    return out


def _darkness_pct_for_image_region(image_path: Path, rect_tlbr: Tuple[int, int, int, int]) -> Optional[float]:
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            width, height = gray.size
            top, left, bottom, right = rect_tlbr
            top = max(0, min(height, int(top)))
            bottom = max(0, min(height, int(bottom)))
            left = max(0, min(width, int(left)))
            right = max(0, min(width, int(right)))
            if bottom <= top or right <= left:
                top, left, bottom, right = 0, 0, height, width
            region = gray.crop((left, top, right, bottom))
            total = int(region.size[0] * region.size[1])
            if total <= 0:
                return None
            hist = region.histogram()
            dark = int(sum(hist[:33]))
            return (float(dark) / float(total)) * 100.0
    except Exception:
        return None


def _load_lr_marker_vendor_library_preview(vendor: str) -> Dict[str, object]:
    root = REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/lr_marker_vendor_template_library"
    out: Dict[str, object] = {
        "available": False,
        "root": root.as_posix(),
        "vendor": vendor,
        "items": [],
        "error": "",
    }
    vendor_key = str(vendor or "").strip().lower()
    if not vendor_key:
        out["error"] = "missing_vendor"
        return out
    if not root.is_dir():
        out["error"] = "library_not_built"
        return out
    vendor_dir = next((p for p in root.iterdir() if p.is_dir() and p.name.lower() == vendor_key), None)
    if vendor_dir is None:
        out["error"] = "vendor_not_in_library"
        return out
    manifest_path = vendor_dir / "manifest.json"
    rows: List[Dict[str, object]] = []
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        rows.append(dict(row))
        except Exception:
            rows = []
    if not rows:
        for path in sorted(vendor_dir.glob("*.png")):
            rows.append({"template_path": path.as_posix(), "source_path": "", "vendor": vendor})
    out["items"] = rows
    out["available"] = bool(rows)
    if not rows:
        out["error"] = "empty_vendor_library"
    return out


def _lr_marker_orientation_group_from_quadrant(side: str, vertical: str) -> str:
    side_norm = str(side or "").strip().lower()
    vertical_norm = str(vertical or "").strip().lower()
    if side_norm == "left" and vertical_norm == "su":
        return "NF"
    if side_norm == "right" and vertical_norm == "su":
        return "LR"
    if side_norm == "left" and vertical_norm == "giu":
        return "UD"
    if side_norm == "right" and vertical_norm == "giu":
        return "LRUD"
    return ""


def _lr_marker_quadrant_fields_from_item(item: Dict[str, object]) -> Dict[str, object]:
    fields: Dict[str, object] = {
        "quadrant_valid": 0,
        "quadrant_status": "unknown",
        "quadrant_reason": "",
        "quadrant_expected": "",
        "quadrant_center": "",
        "quadrant_group": "",
        "quadrant_center_group": "",
        "echo_mid_x_abs": "",
        "echo_mid_y_abs": "",
    }
    side = str(item.get("detected_marker_side", "") or "").strip().lower()
    sugiu = str(item.get("su_giu_pred", "") or "").strip().lower()
    try:
        echo_top = int(item.get("echo_rect_top_abs", 0) or 0)
        echo_left = int(item.get("echo_rect_left_abs", 0) or 0)
        echo_bottom = int(item.get("echo_rect_bottom_abs", 0) or 0)
        echo_right = int(item.get("echo_rect_right_abs", 0) or 0)
        marker_top = int(item.get("marker_top_abs", 0) or 0)
        marker_left = int(item.get("marker_left_abs", 0) or 0)
        marker_bottom = int(item.get("marker_bottom_abs", 0) or 0)
        marker_right = int(item.get("marker_right_abs", 0) or 0)
    except Exception:
        fields["quadrant_reason"] = "invalid_quadrant_coordinates"
        return fields
    if echo_bottom <= echo_top or echo_right <= echo_left or marker_bottom <= marker_top or marker_right <= marker_left:
        fields["quadrant_reason"] = "invalid_quadrant_rect"
        return fields

    mid_x = (echo_left + echo_right) / 2.0
    mid_y = (echo_top + echo_bottom) / 2.0
    marker_cx = (marker_left + marker_right) / 2.0
    marker_cy = (marker_top + marker_bottom) / 2.0
    center_side = "left" if marker_cx < mid_x else "right"
    center_vertical = "su" if marker_cy < mid_y else "giu"
    fields.update(
        {
            "quadrant_expected": f"{side}_{sugiu}" if side in {"left", "right"} and sugiu in {"su", "giu"} else "",
            "quadrant_center": f"{center_side}_{center_vertical}",
            "quadrant_group": _lr_marker_orientation_group_from_quadrant(side, sugiu),
            "quadrant_center_group": _lr_marker_orientation_group_from_quadrant(center_side, center_vertical),
            "echo_mid_x_abs": float(mid_x),
            "echo_mid_y_abs": float(mid_y),
        }
    )

    reasons: List[str] = []
    if side not in {"left", "right"}:
        reasons.append("missing_marker_side")
    if sugiu not in {"su", "giu"}:
        reasons.append("missing_sugiu_pred")
    if side == "left" and marker_right > mid_x:
        reasons.append("marker_crosses_vertical_median")
    elif side == "right" and marker_left < mid_x:
        reasons.append("marker_crosses_vertical_median")
    if sugiu == "su" and marker_bottom > mid_y:
        reasons.append("marker_crosses_horizontal_median")
    elif sugiu == "giu" and marker_top < mid_y:
        reasons.append("marker_crosses_horizontal_median")

    if reasons:
        fields["quadrant_status"] = "invalid"
        fields["quadrant_valid"] = 0
        fields["quadrant_reason"] = ";".join(dict.fromkeys(reasons))
    else:
        fields["quadrant_status"] = "ok"
        fields["quadrant_valid"] = 1
        fields["quadrant_reason"] = ""
    return fields


def _build_lr_marker_per_image_evidence(
    *,
    run_dir: Path,
    pipeline_output: Path,
) -> Dict[str, object]:
    csv_path = pipeline_output / "lr_marker_per_image_predictions.csv"
    out: Dict[str, object] = {
        "available": False,
        "error": "",
        "csv_path": csv_path.as_posix(),
        "images_total": 0,
        "label_counts": {"not_lr_flipped": 0, "lr_flipped": 0, "other": 0},
        "status_counts": {"ok": 0, "review": 0, "other": 0},
        "quadrant_counts": {"ok": 0, "invalid": 0, "unknown": 0},
        "search_strategy_counts": {},
        "best": {},
        "items": [],
        "template_policy": "",
        "fixed_template_path": "",
        "fixed_template_selection_score": 0.0,
        "template_path_counts": {},
        "vendor_template_library": {},
        "dark_sample": {},
        "darkness_rank": [],
        "note": "LR marker classico: seleziona un solo template vendor per cartella e poi cerca solo quello in ogni frame.",
    }
    if not csv_path.is_file():
        out["error"] = "file lr_marker_per_image_predictions.csv non trovato"
        return out

    items: List[Dict[str, object]] = []
    label_counts = {"not_lr_flipped": 0, "lr_flipped": 0, "other": 0}
    status_counts = {"ok": 0, "review": 0, "other": 0}
    quadrant_counts = {"ok": 0, "invalid": 0, "unknown": 0}
    strategy_counts: Dict[str, int] = {}
    template_path_counts: Dict[str, int] = {}
    best_item: Dict[str, object] = {}
    best_score = -1.0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_path_s = str(row.get("image_path", "") or "").strip()
                image_path_obj = Path(image_path_s).expanduser() if image_path_s else Path()
                if image_path_s and not image_path_obj.is_absolute():
                    image_path_obj = run_dir / image_path_obj
                image_rel = image_path_s
                if image_path_s:
                    try:
                        image_rel = Path(image_path_s).expanduser().relative_to(run_dir).as_posix()
                    except Exception:
                        try:
                            image_rel = Path(image_path_s).expanduser().resolve().relative_to(run_dir).as_posix()
                        except Exception:
                            image_rel = image_path_s
                label = str(row.get("lr_label", "") or "").strip()
                method = str(row.get("lr_marker_method", "") or "classical").strip() or "classical"
                orientation_group = str(row.get("orientation_group", "") or "").strip().upper()
                status = str(row.get("status", "") or "").strip()
                strategy = str(row.get("search_strategy", "") or "").strip()
                score = _safe_float(str(row.get("match_score", "") or "0"), 0.0)
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts["other"] += 1
                if strategy:
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                template_path = str(row.get("template_path", "") or "")
                if template_path:
                    template_path_counts[template_path] = template_path_counts.get(template_path, 0) + 1
                rect_tlbr = (
                    _safe_int(str(row.get("echo_rect_top_abs", "") or "0"), 0),
                    _safe_int(str(row.get("echo_rect_left_abs", "") or "0"), 0),
                    _safe_int(str(row.get("echo_rect_bottom_abs", "") or "0"), 0),
                    _safe_int(str(row.get("echo_rect_right_abs", "") or "0"), 0),
                )
                image_width = _safe_int(str(row.get("image_width", "") or "0"), 0)
                image_height = _safe_int(str(row.get("image_height", "") or "0"), 0)
                if (image_width <= 0 or image_height <= 0) and image_path_s and image_path_obj.is_file():
                    try:
                        with Image.open(image_path_obj) as img_probe:
                            image_width, image_height = img_probe.size
                    except Exception:
                        image_width = 0
                        image_height = 0
                darkness_pct = _darkness_pct_for_image_region(image_path_obj, rect_tlbr) if image_path_s else None
                item = {
                    "image_index": _safe_int(str(row.get("image_index", "") or "0"), 0),
                    "image_rel": image_rel,
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "canonical_image_size": _safe_int(str(row.get("canonical_image_size", "") or "0"), 0),
                    "vendor": str(row.get("vendor", "") or ""),
                    "lr_marker_method": method,
                    "status": status,
                    "review_reason": str(row.get("review_reason", "") or ""),
                    "orientation_group": orientation_group,
                    "lr_label": label,
                    "lr_label_it": str(row.get("lr_label_it", "") or ""),
                    "detected_marker_side": str(row.get("detected_marker_side", "") or ""),
                    "match_score": float(score),
                    "match_patch_is_blank": _safe_int(str(row.get("match_patch_is_blank", "") or "0"), 0),
                    "match_patch_dark_pct": _safe_float(str(row.get("match_patch_dark_pct", "") or "0"), 0.0),
                    "match_patch_max_value": _safe_int(str(row.get("match_patch_max_value", "") or "0"), 0),
                    "match_patch_std": _safe_float(str(row.get("match_patch_std", "") or "0"), 0.0),
                    "search_strategy": strategy,
                    "search_scope": str(row.get("search_scope", "") or ""),
                    "search_margin_px": _safe_int(str(row.get("search_margin_px", "") or "0"), 0),
                    "template_policy": str(row.get("template_policy", "") or ""),
                    "template_policy_requested": str(row.get("template_policy_requested", "") or ""),
                    "template_policy_effective": str(row.get("template_policy_effective", "") or ""),
                    "template_fallback_reason": str(row.get("template_fallback_reason", "") or ""),
                    "folder_fixed_template_path": str(row.get("folder_fixed_template_path", "") or ""),
                    "folder_fixed_template_selection_score": _safe_float(
                        str(row.get("folder_fixed_template_selection_score", "") or "0"),
                        0.0,
                    ),
                    "folder_template_seed_image_path": str(row.get("folder_template_seed_image_path", "") or ""),
                    "folder_template_seed_darkness_pct": _safe_float(
                        str(row.get("folder_template_seed_darkness_pct", "") or "-1"),
                        -1.0,
                    ),
                    "folder_template_seed_db_template_path": str(row.get("folder_template_seed_db_template_path", "") or ""),
                    "folder_template_rank_json": str(row.get("folder_template_rank_json", "") or ""),
                    "template_path": template_path,
                    "su_giu_model_pred": str(row.get("su_giu_model_pred", "") or ""),
                    "su_giu_pred": str(row.get("su_giu_pred", "") or ""),
                    "su_giu_conf": _safe_float(str(row.get("su_giu_conf", "") or "0"), 0.0),
                    "prob_su": _safe_float(str(row.get("prob_su", "") or "0"), 0.0),
                    "prob_giu": _safe_float(str(row.get("prob_giu", "") or "0"), 0.0),
                    "dual_sugiu_search_used": _safe_int(str(row.get("dual_sugiu_search_used", "") or "0"), 0),
                    "opposite_sugiu_match_score": _safe_float(str(row.get("opposite_sugiu_match_score", "") or "0"), 0.0),
                    "echo_rect_top_abs": rect_tlbr[0],
                    "echo_rect_left_abs": rect_tlbr[1],
                    "echo_rect_bottom_abs": rect_tlbr[2],
                    "echo_rect_right_abs": rect_tlbr[3],
                    "marker_top_abs": _safe_int(str(row.get("marker_top_abs", "") or "0"), 0),
                    "marker_left_abs": _safe_int(str(row.get("marker_left_abs", "") or "0"), 0),
                    "marker_bottom_abs": _safe_int(str(row.get("marker_bottom_abs", "") or "0"), 0),
                    "marker_right_abs": _safe_int(str(row.get("marker_right_abs", "") or "0"), 0),
                    "marker_cx_crop_norm": _safe_float(str(row.get("marker_cx_crop_norm", "") or "0"), 0.0),
                    "marker_cy_crop_norm": _safe_float(str(row.get("marker_cy_crop_norm", "") or "0"), 0.0),
                }
                quadrant_fields = _lr_marker_quadrant_fields_from_item(item)
                if method == "bundle" and orientation_group in {"NF", "LR", "UD", "LRUD"}:
                    quadrant_fields["quadrant_group"] = orientation_group
                item.update(quadrant_fields)
                quadrant_status = str(item.get("quadrant_status", "") or "unknown").strip().lower() or "unknown"
                if quadrant_status in quadrant_counts:
                    quadrant_counts[quadrant_status] += 1
                else:
                    quadrant_counts["unknown"] += 1
                if quadrant_status == "invalid":
                    status = "review"
                    review_parts = [
                        part
                        for part in str(item.get("review_reason", "") or "").split(";")
                        if part
                    ]
                    review_parts.append("quadrant_logic_violation")
                    review_parts.extend(
                        part
                        for part in str(item.get("quadrant_reason", "") or "").split(";")
                        if part
                    )
                    item["review_reason"] = ";".join(dict.fromkeys(review_parts))
                    item["status"] = status
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts["other"] += 1
                if darkness_pct is not None:
                    item["darkness_pct"] = float(darkness_pct)
                items.append(item)
                if score > best_score:
                    best_score = score
                    best_item = dict(item)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    items.sort(key=lambda x: (int(x.get("image_index", 0)), str(x.get("image_rel", ""))))
    out["items"] = items
    out["images_total"] = len(items)
    out["label_counts"] = label_counts
    out["status_counts"] = status_counts
    out["quadrant_counts"] = quadrant_counts
    out["search_strategy_counts"] = dict(sorted(strategy_counts.items()))
    out["template_path_counts"] = dict(sorted(template_path_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))))
    darkness_rank = sorted(
        [
            {
                "image_index": int(item.get("image_index", 0) or 0),
                "image_rel": str(item.get("image_rel", "") or ""),
                "darkness_pct": float(item.get("darkness_pct", 0.0) or 0.0),
                "match_score": float(item.get("match_score", 0.0) or 0.0),
                "template_path": str(item.get("template_path", "") or ""),
            }
            for item in items
            if item.get("darkness_pct") is not None
        ],
        key=lambda x: (-float(x.get("darkness_pct", 0.0) or 0.0), int(x.get("image_index", 0) or 0)),
    )
    out["darkness_rank"] = darkness_rank
    if darkness_rank:
        sample_idx = int(darkness_rank[0].get("image_index", 0) or 0)
        sample_rel = str(darkness_rank[0].get("image_rel", "") or "")
        sample_item = next(
            (
                dict(item)
                for item in items
                if int(item.get("image_index", 0) or 0) == sample_idx
                and str(item.get("image_rel", "") or "") == sample_rel
            ),
            dict(darkness_rank[0]),
        )
        out["dark_sample"] = sample_item
    if items:
        first_item = items[0]
        out["lr_marker_method"] = str(first_item.get("lr_marker_method", "") or "")
        out["template_policy"] = str(first_item.get("template_policy", "") or "")
        out["template_policy_requested"] = str(first_item.get("template_policy_requested", "") or "")
        out["template_policy_effective"] = str(first_item.get("template_policy_effective", "") or "")
        out["template_fallback_reason"] = str(first_item.get("template_fallback_reason", "") or "")
        out["fixed_template_path"] = str(first_item.get("folder_fixed_template_path", "") or "")
        out["fixed_template_selection_score"] = float(first_item.get("folder_fixed_template_selection_score", 0.0) or 0.0)
        out["vendor_template_library"] = _load_lr_marker_vendor_library_preview(str(first_item.get("vendor", "") or ""))
        if out["lr_marker_method"] == "bundle":
            out["note"] = (
                "LR marker bundle: usa la libreria templates del bundle, il rettangolo ecografico ufficiale "
                "e gli assi mediani del rect per assegnare NF/LR/UD/LRUD."
            )
        elif out["template_policy"] == "fixed_historical_best_template":
            out["note"] = "LR marker classico: seleziona il template storico best-match e cerca solo quello in ogni frame."
        elif out["template_policy"] == "fixed_derived_folder_template":
            out["note"] = "LR marker classico: deriva un template dalla nuova cartella e cerca solo quello in ogni frame."
        if out["template_fallback_reason"]:
            out["note"] = "LR marker classico: historical_best non affidabile, applicato fallback derived_folder."
    out["best"] = best_item
    out["available"] = bool(items)
    if not items:
        out["error"] = "nessuna riga valida nel CSV lr_marker_per_image_predictions.csv"
    return out


def _build_su_giu_split_folders(
    *,
    run_dir: Path,
    su_giu_per_image_evidence: Dict[str, object],
) -> Dict[str, object]:
    root_dir = run_dir / "evidence" / "su_giu_split"
    su_dir = root_dir / "su"
    giu_dir = root_dir / "giu"
    other_dir = root_dir / "other"
    out: Dict[str, object] = {
        "available": False,
        "mode": "symlink",
        "root_dir_abs": root_dir.as_posix(),
        "root_dir_rel": "",
        "su_dir_abs": su_dir.as_posix(),
        "su_dir_rel": "",
        "giu_dir_abs": giu_dir.as_posix(),
        "giu_dir_rel": "",
        "other_dir_abs": other_dir.as_posix(),
        "other_dir_rel": "",
        "counts": {"su": 0, "giu": 0, "other": 0},
        "created_total": 0,
        "skipped_missing": 0,
        "items_by_label": {"su": [], "giu": [], "other": []},
        "error": "",
    }
    items_raw = su_giu_per_image_evidence.get("items", [])
    if not isinstance(items_raw, list) or not items_raw:
        out["error"] = "nessun item su_giu per creare cartelle separate"
        return out

    try:
        if root_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(root_dir, ignore_errors=False)
        su_dir.mkdir(parents=True, exist_ok=True)
        giu_dir.mkdir(parents=True, exist_ok=True)
        other_dir.mkdir(parents=True, exist_ok=True)
        out["root_dir_rel"] = root_dir.relative_to(run_dir).as_posix()
        out["su_dir_rel"] = su_dir.relative_to(run_dir).as_posix()
        out["giu_dir_rel"] = giu_dir.relative_to(run_dir).as_posix()
        out["other_dir_rel"] = other_dir.relative_to(run_dir).as_posix()
    except Exception as exc:
        out["error"] = f"errore creazione cartelle su/giu: {exc}"
        return out

    def _pick_dir(label: str) -> Tuple[str, Path]:
        key = str(label or "").strip().lower()
        if key == "su":
            return "su", su_dir
        if key == "giu":
            return "giu", giu_dir
        return "other", other_dir

    for idx, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue
        image_rel = str(item.get("image_rel", "") or "").strip()
        frame_idx = _safe_int(str(item.get("image_index", "0") or "0"), 0)
        label_raw = str(item.get("pred_label", "") or "").strip().lower()
        label_key, dst_dir = _pick_dir(label_raw)

        if not image_rel:
            out["skipped_missing"] = int(out.get("skipped_missing", 0) or 0) + 1
            continue
        src_path = Path(image_rel).expanduser()
        if not src_path.is_absolute():
            src_path = run_dir / src_path
        if not src_path.is_file():
            out["skipped_missing"] = int(out.get("skipped_missing", 0) or 0) + 1
            continue

        src_name = src_path.name or f"image_{idx:04d}.png"
        link_base = f"{int(frame_idx):04d}_{src_name}"
        link_path = dst_dir / link_base
        if link_path.exists():
            stem = Path(src_name).stem or f"image_{idx:04d}"
            suffix = Path(src_name).suffix
            n = 1
            while True:
                candidate = dst_dir / f"{int(frame_idx):04d}_{stem}_{n}{suffix}"
                if not candidate.exists():
                    link_path = candidate
                    break
                n += 1
                if n > 9999:
                    break

        try:
            link_path.symlink_to(src_path)
        except Exception as exc:
            out["error"] = f"errore creazione symlink su/giu: {exc}"
            continue

        try:
            link_rel = link_path.relative_to(run_dir).as_posix()
        except Exception:
            link_rel = link_path.as_posix()

        out["created_total"] = int(out.get("created_total", 0) or 0) + 1
        counts = out.get("counts", {})
        if isinstance(counts, dict):
            counts[label_key] = int(counts.get(label_key, 0) or 0) + 1
        items_by_label = out.get("items_by_label", {})
        if isinstance(items_by_label, dict):
            lst = items_by_label.get(label_key, [])
            if isinstance(lst, list):
                lst.append(
                    {
                        "image_index": int(frame_idx),
                        "pred_label": label_key,
                        "link_rel": link_rel,
                        "image_rel": image_rel,
                    }
                )
                items_by_label[label_key] = lst

    out["available"] = bool(int(out.get("created_total", 0) or 0) > 0)
    if not out["available"] and not str(out.get("error", "")).strip():
        out["error"] = "nessun symlink creato per cartelle su/giu"
    return out


def _build_lt_per_image_evidence(
    *,
    run_dir: Path,
    pipeline_output: Path,
) -> Dict[str, object]:
    csv_path = pipeline_output / "lt_per_image_predictions.csv"
    out: Dict[str, object] = {
        "available": False,
        "error": "",
        "csv_path": csv_path.as_posix(),
        "images_total": 0,
        "label_counts": {"l": 0, "t": 0, "other": 0},
        "items": [],
        "note": "Etichettatura L/T per singolo frame sul crop rettangolo ecografico.",
    }
    if not csv_path.is_file():
        out["error"] = "file lt_per_image_predictions.csv non trovato"
        return out

    items: List[Dict[str, object]] = []
    counts = {"l": 0, "t": 0, "other": 0}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_path_s = str(row.get("image_path", "") or "").strip()
                pred_label = str(row.get("pred_label", "") or "").strip().upper()
                image_idx = _safe_int(str(row.get("image_index", "") or "0"), 0)
                conf = _safe_float(str(row.get("confidence", "") or "0"), 0.0)
                p_l = _safe_float(str(row.get("prob_l", "") or "0"), 0.0)
                p_t = _safe_float(str(row.get("prob_t", "") or "0"), 0.0)
                crop_top = _safe_int(str(row.get("crop_top", "") or "0"), 0)
                crop_left = _safe_int(str(row.get("crop_left", "") or "0"), 0)
                crop_bottom = _safe_int(str(row.get("crop_bottom", "") or "0"), 0)
                crop_right = _safe_int(str(row.get("crop_right", "") or "0"), 0)
                crop_source = str(row.get("crop_source", "") or "").strip()

                image_rel = image_path_s
                if image_path_s:
                    try:
                        image_rel = Path(image_path_s).expanduser().relative_to(run_dir).as_posix()
                    except Exception:
                        try:
                            image_rel = Path(image_path_s).expanduser().resolve().relative_to(run_dir).as_posix()
                        except Exception:
                            image_rel = image_path_s

                if pred_label == "L":
                    counts["l"] += 1
                elif pred_label == "T":
                    counts["t"] += 1
                else:
                    counts["other"] += 1

                items.append(
                    {
                        "image_index": int(image_idx),
                        "image_rel": image_rel,
                        "pred_label": pred_label,
                        "confidence": float(conf),
                        "prob_l": float(p_l),
                        "prob_t": float(p_t),
                        "crop_top": int(crop_top),
                        "crop_left": int(crop_left),
                        "crop_bottom": int(crop_bottom),
                        "crop_right": int(crop_right),
                        "crop_source": crop_source,
                    }
                )
    except Exception as exc:
        out["error"] = f"errore lettura lt_per_image_predictions.csv: {exc}"
        return out

    items.sort(
        key=lambda x: (
            int(x.get("image_index", 0)),
            str(x.get("image_rel", "")),
        )
    )
    out["items"] = items
    out["images_total"] = len(items)
    out["label_counts"] = counts
    out["available"] = bool(items)
    if not items:
        out["error"] = "nessuna riga valida nel CSV lt_per_image_predictions.csv"
    return out


def _build_rect_depth_per_image_evidence(
    *,
    run_dir: Path,
    pipeline_output: Path,
) -> Dict[str, object]:
    csv_path = pipeline_output / "rect_depth_autonomous_predictions.csv"
    out: Dict[str, object] = {
        "available": False,
        "error": "",
        "csv_path": csv_path.as_posix(),
        "images_total": 0,
        "status_counts": {"accepted": 0, "review": 0, "reject": 0, "missing": 0},
        "mode_counts": {},
        "accepted_ratio": 0.0,
        "items": [],
        "note": "Riconoscimento RECT_DEPTH autonomo per singolo frame (OCR + regole classiche + ranker tabellare).",
    }
    if not csv_path.is_file():
        out["error"] = "file rect_depth_autonomous_predictions.csv non trovato"
        return out

    items: List[Dict[str, object]] = []
    status_counts: Dict[str, int] = {"accepted": 0, "review": 0, "reject": 0, "missing": 0}
    mode_counts: Dict[str, int] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_path_s = str(row.get("image_path", "") or "").strip()
                status = str(row.get("status", "") or "missing").strip().lower() or "missing"
                mode = str(row.get("mode", "") or "").strip()
                image_idx = _safe_int(str(row.get("image_index", "") or "0"), 0)
                depth_mm = _safe_float(str(row.get("depth_mm", "") or "0"), 0.0)
                score = _safe_float(str(row.get("score", "") or "0"), 0.0)
                left = _safe_int(str(row.get("left", "") or "0"), 0)
                top = _safe_int(str(row.get("top", "") or "0"), 0)
                right = _safe_int(str(row.get("right", "") or "0"), 0)
                bottom = _safe_int(str(row.get("bottom", "") or "0"), 0)

                image_rel = image_path_s
                if image_path_s:
                    try:
                        image_rel = Path(image_path_s).expanduser().relative_to(run_dir).as_posix()
                    except Exception:
                        try:
                            image_rel = Path(image_path_s).expanduser().resolve().relative_to(run_dir).as_posix()
                        except Exception:
                            image_rel = image_path_s

                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
                if mode:
                    mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
                items.append(
                    {
                        "image_index": int(image_idx),
                        "image_rel": image_rel,
                        "status": status,
                        "score": float(score),
                        "mode": mode,
                        "depth_mm": float(depth_mm),
                        "left": int(left),
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "ocr_text": str(row.get("ocr_text", "") or ""),
                        "reason": str(row.get("reason", "") or ""),
                        "candidates": _safe_int(str(row.get("candidates", "") or "0"), 0),
                        "run_dir": str(row.get("rect_depth_run_dir", "") or ""),
                    }
                )
    except Exception as exc:
        out["error"] = f"errore lettura rect_depth_autonomous_predictions.csv: {exc}"
        return out

    items.sort(
        key=lambda x: (
            int(x.get("image_index", 0)),
            str(x.get("image_rel", "")),
        )
    )
    accepted = int(status_counts.get("accepted", 0))
    out["items"] = items
    out["images_total"] = len(items)
    out["status_counts"] = status_counts
    out["mode_counts"] = mode_counts
    out["accepted_ratio"] = float(accepted / max(1, len(items)))
    out["available"] = bool(items)
    if not items:
        out["error"] = "nessuna riga valida nel CSV rect_depth_autonomous_predictions.csv"
    return out


def _build_lt_split_folders(
    *,
    run_dir: Path,
    lt_per_image_evidence: Dict[str, object],
) -> Dict[str, object]:
    root_dir = run_dir / "evidence" / "lt_split"
    l_dir = root_dir / "l"
    t_dir = root_dir / "t"
    other_dir = root_dir / "other"
    out: Dict[str, object] = {
        "available": False,
        "mode": "symlink",
        "root_dir_abs": root_dir.as_posix(),
        "root_dir_rel": "",
        "l_dir_abs": l_dir.as_posix(),
        "l_dir_rel": "",
        "t_dir_abs": t_dir.as_posix(),
        "t_dir_rel": "",
        "other_dir_abs": other_dir.as_posix(),
        "other_dir_rel": "",
        "counts": {"l": 0, "t": 0, "other": 0},
        "created_total": 0,
        "skipped_missing": 0,
        "items_by_label": {"l": [], "t": [], "other": []},
        "error": "",
    }
    items_raw = lt_per_image_evidence.get("items", [])
    if not isinstance(items_raw, list) or not items_raw:
        out["error"] = "nessun item L/T per creare cartelle separate"
        return out

    try:
        if root_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(root_dir, ignore_errors=False)
        l_dir.mkdir(parents=True, exist_ok=True)
        t_dir.mkdir(parents=True, exist_ok=True)
        other_dir.mkdir(parents=True, exist_ok=True)
        out["root_dir_rel"] = root_dir.relative_to(run_dir).as_posix()
        out["l_dir_rel"] = l_dir.relative_to(run_dir).as_posix()
        out["t_dir_rel"] = t_dir.relative_to(run_dir).as_posix()
        out["other_dir_rel"] = other_dir.relative_to(run_dir).as_posix()
    except Exception as exc:
        out["error"] = f"errore creazione cartelle L/T: {exc}"
        return out

    def _pick_dir(label: str) -> Tuple[str, Path]:
        key = str(label or "").strip().upper()
        if key == "L":
            return "l", l_dir
        if key == "T":
            return "t", t_dir
        return "other", other_dir

    for idx, item in enumerate(items_raw):
        if not isinstance(item, dict):
            continue
        image_rel = str(item.get("image_rel", "") or "").strip()
        frame_idx = _safe_int(str(item.get("image_index", "0") or "0"), 0)
        label_raw = str(item.get("pred_label", "") or "").strip().upper()
        label_key, dst_dir = _pick_dir(label_raw)

        if not image_rel:
            out["skipped_missing"] = int(out.get("skipped_missing", 0) or 0) + 1
            continue
        src_path = Path(image_rel).expanduser()
        if not src_path.is_absolute():
            src_path = run_dir / src_path
        if not src_path.is_file():
            out["skipped_missing"] = int(out.get("skipped_missing", 0) or 0) + 1
            continue

        src_name = src_path.name or f"image_{idx:04d}.png"
        link_base = f"{int(frame_idx):04d}_{src_name}"
        link_path = dst_dir / link_base
        if link_path.exists():
            stem = Path(src_name).stem or f"image_{idx:04d}"
            suffix = Path(src_name).suffix
            n = 1
            while True:
                candidate = dst_dir / f"{int(frame_idx):04d}_{stem}_{n}{suffix}"
                if not candidate.exists():
                    link_path = candidate
                    break
                n += 1
                if n > 9999:
                    break

        try:
            link_path.symlink_to(src_path)
        except Exception as exc:
            out["error"] = f"errore creazione symlink L/T: {exc}"
            continue

        try:
            link_rel = link_path.relative_to(run_dir).as_posix()
        except Exception:
            link_rel = link_path.as_posix()

        out["created_total"] = int(out.get("created_total", 0) or 0) + 1
        counts = out.get("counts", {})
        if isinstance(counts, dict):
            counts[label_key] = int(counts.get(label_key, 0) or 0) + 1
        items_by_label = out.get("items_by_label", {})
        if isinstance(items_by_label, dict):
            lst = items_by_label.get(label_key, [])
            if isinstance(lst, list):
                lst.append(
                    {
                        "image_index": int(frame_idx),
                        "pred_label": label_key.upper(),
                        "link_rel": link_rel,
                        "image_rel": image_rel,
                    }
                )
                items_by_label[label_key] = lst

    out["available"] = bool(int(out.get("created_total", 0) or 0) > 0)
    if not out["available"] and not str(out.get("error", "")).strip():
        out["error"] = "nessun symlink creato per cartelle L/T"
    return out


def _extract_prefixed_row_fields(row: Dict[str, str], prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    prefix_l = prefix.lower()
    for k, v in row.items():
        if str(k).lower().startswith(prefix_l):
            txt = str(v or "").strip()
            if txt:
                out[str(k)] = txt
    return out


def _parse_top_choices(raw_value: str) -> List[Dict[str, Any]]:
    txt = str(raw_value or "").strip()
    if not txt:
        return []

    out: List[Dict[str, Any]] = []

    def _append_choice(label_raw: Any, confidence_raw: Any) -> None:
        label = str(label_raw or "").strip()
        if not label:
            return
        conf = _safe_float(str(confidence_raw), float("nan"))
        item: Dict[str, Any] = {"label": label}
        if conf == conf:  # NaN check
            item["confidence"] = float(conf)
        out.append(item)

    try:
        payload = json.loads(txt)
    except Exception:
        payload = None

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                _append_choice(
                    item.get("label") or item.get("value") or item.get("id"),
                    item.get("confidence") or item.get("prob") or item.get("score"),
                )
            elif isinstance(item, (list, tuple)) and item:
                _append_choice(item[0], item[1] if len(item) > 1 else "")
            else:
                _append_choice(item, "")
        if out:
            return out[:3]

    # Fallback legacy format: "A:0.1234 | B:0.0567 | C:0.0012"
    tokens = [tok.strip() for tok in txt.split("|") if tok.strip()]
    for tok in tokens:
        if ":" in tok:
            label, conf = tok.rsplit(":", 1)
            _append_choice(label.strip(), conf.strip())
        else:
            _append_choice(tok, "")
    return out[:3]


def _source_kind_from_hint(source_hint: str, ocr_fields: Dict[str, str]) -> str:
    hint = str(source_hint or "").strip().lower()
    if "ocr" in hint:
        return "ocr"
    if "manual" in hint or "user_" in hint:
        return "manual"
    for k, v in ocr_fields.items():
        key_l = str(k or "").strip().lower()
        if not (key_l.endswith("_ocr_used") or key_l == "ocr_used"):
            continue
        val_l = str(v or "").strip().lower()
        if val_l in {"1", "true", "yes", "y"}:
            return "ocr"
        return "cnn"
    if ocr_fields:
        return "ocr"
    return "cnn"


def _default_decision_reason(
    *,
    source_kind: str,
    confidence: float,
    threshold: float,
    field_label: str,
) -> str:
    if source_kind == "manual":
        return f"Selezione manuale utente su {field_label}."
    if source_kind == "ocr":
        return f"Decisione {field_label} guidata da OCR."
    if confidence < threshold:
        return (
            f"CNN top-1 {field_label} sotto soglia "
            f"({confidence:.4f} < {threshold:.4f}), mantenuta in review."
        )
    return f"CNN top-1 {field_label} sopra soglia ({confidence:.4f} >= {threshold:.4f})."


def _find_encoding_struct_workbook() -> Optional[Path]:
    candidates = sorted(REPO_ROOT.glob("encoding_struct*.xlsx"))
    if not candidates:
        return None
    return candidates[-1]


def _normalize_probe_id(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def _extract_probe_name_map_from_probe_model_type_map_html(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    m = re.search(r"const\s+DATA\s*=\s*(\[[\s\S]*?\])\s*;\s*const\s+TYPE_OPTIONS", text)
    if not m:
        return {}
    payload = m.group(1).strip()
    try:
        rows = json.loads(payload)
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    out: Dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        pid = _normalize_probe_id(item.get("probe_id"))
        name = str(item.get("probe_name") or "").strip()
        if not pid or not name:
            continue
        if pid not in out:
            out[pid] = name
    return out


def _load_probe_name_map() -> Dict[str, str]:
    map_candidates = sorted(
        (REPO_ROOT / "artifacts/60_metadata").glob("probe_model_type_map*.html"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for html_map in map_candidates:
        out_html = _extract_probe_name_map_from_probe_model_type_map_html(html_map)
        if out_html:
            return out_html

    workbook = _find_encoding_struct_workbook()
    if workbook is None or not workbook.is_file():
        return {}
    try:
        import openpyxl
    except Exception:
        return {}
    out: Dict[str, str] = {}
    try:
        wb = openpyxl.load_workbook(workbook, data_only=True, read_only=True)
    except Exception:
        return {}
    try:
        if "PROBE" not in wb.sheetnames:
            return {}
        ws = wb["PROBE"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) < 2:
                continue
            name = str(row[0] or "").strip()
            pid_raw = row[1]
            pid = str(pid_raw).strip() if pid_raw is not None else ""
            if not name or not pid:
                continue
            out[pid] = name
    except Exception:
        return {}
    return out


def _build_decision_evidence(
    row: Dict[str, str],
    probe_name_map: Dict[str, str],
    vendor_min_conf: float,
    probe_min_conf: float,
) -> Dict[str, Any]:
    vendor_name = str(row.get("vendor_predicted", "") or "").strip()
    vendor_conf = _safe_float(row.get("vendor_confidence", "0"), 0.0)
    probe_id = str(row.get("line_03_id_probe", "") or "").strip()
    probe_conf = _safe_float(row.get("line_03_probe_confidence", "0"), 0.0)
    probe_name = probe_name_map.get(probe_id, "")

    vendor_threshold = _safe_float(
        row.get("vendor_threshold", str(vendor_min_conf)),
        float(vendor_min_conf),
    )
    probe_threshold = _safe_float(
        row.get("probe_threshold", str(probe_min_conf)),
        float(probe_min_conf),
    )

    vendor_source_hint = str(row.get("vendor_source", "") or "cnn_classifier").strip()
    probe_source_hint = str(row.get("probe_source", "") or "cnn_classifier").strip()

    vendor_ocr_fields = _extract_prefixed_row_fields(row, "vendor_ocr")
    probe_ocr_fields = _extract_prefixed_row_fields(row, "probe_ocr")
    vendor_source_kind = _source_kind_from_hint(vendor_source_hint, vendor_ocr_fields)
    probe_source_kind = _source_kind_from_hint(probe_source_hint, probe_ocr_fields)
    vendor_ocr_used = vendor_source_kind == "ocr"
    probe_ocr_used = probe_source_kind == "ocr"

    vendor_top_choices = (
        _parse_top_choices(str(row.get("vendor_top3_json", "") or ""))
        or _parse_top_choices(str(row.get("vendor_topk_json", "") or ""))
        or _parse_top_choices(str(row.get("vendor_topk", "") or ""))
    )
    if not vendor_top_choices and vendor_name:
        vendor_top_choices = [{"label": vendor_name, "confidence": float(vendor_conf)}]

    probe_top_choices = (
        _parse_top_choices(str(row.get("probe_top3_json", "") or ""))
        or _parse_top_choices(str(row.get("probe_topk_json", "") or ""))
        or _parse_top_choices(str(row.get("probe_topk", "") or ""))
    )
    for item in probe_top_choices:
        pid = str(item.get("label", "")).strip()
        if pid and pid in probe_name_map:
            item["name"] = probe_name_map[pid]
    if not probe_top_choices and probe_id:
        probe_top_choices = [
            {
                "label": probe_id,
                "name": probe_name_map.get(probe_id, ""),
                "confidence": float(probe_conf),
            }
        ]

    vendor_decision_reason = str(row.get("vendor_decision_reason", "") or "").strip()
    if not vendor_decision_reason:
        vendor_decision_reason = _default_decision_reason(
            source_kind=vendor_source_kind,
            confidence=float(vendor_conf),
            threshold=float(vendor_threshold),
            field_label="vendor",
        )

    probe_decision_reason = str(row.get("probe_decision_reason", "") or "").strip()
    if not probe_decision_reason:
        probe_decision_reason = _default_decision_reason(
            source_kind=probe_source_kind,
            confidence=float(probe_conf),
            threshold=float(probe_threshold),
            field_label="probe",
        )

    su_giu_images = _safe_int(row.get("su_giu_images_predicted", "0"), 0)
    su_giu_majority = str(row.get("su_giu_majority_label", "") or "").strip()
    su_giu_vote_ratio = _safe_float(row.get("su_giu_majority_vote_ratio", "0"), 0.0)
    su_giu_mean_conf = _safe_float(row.get("su_giu_mean_confidence", "0"), 0.0)
    su_giu_mean_prob_su = _safe_float(row.get("su_giu_mean_prob_su", "0"), 0.0)
    su_giu_mean_prob_giu = _safe_float(row.get("su_giu_mean_prob_giu", "0"), 0.0)
    su_giu_source = str(row.get("su_giu_source", "") or "").strip()
    su_giu_checkpoint = str(row.get("su_giu_checkpoint", "") or "").strip()

    return {
        "vendor": {
            "predicted_name": vendor_name,
            "confidence": float(vendor_conf),
            "threshold": float(vendor_threshold),
            "source_kind": vendor_source_kind,
            "source_hint": vendor_source_hint or "cnn_classifier",
            "ocr_used": bool(vendor_ocr_used),
            "ocr_elements": vendor_ocr_fields,
            "top_choices": vendor_top_choices,
            "decision_reason": vendor_decision_reason,
            "ocr_comment": (
                "OCR usato nella decisione vendor."
                if vendor_ocr_used
                else "Decisione vendor da classificatore immagini (senza OCR)."
            ),
        },
        "probe": {
            "predicted_id": probe_id,
            "predicted_name": probe_name,
            "confidence": float(probe_conf),
            "threshold": float(probe_threshold),
            "source_kind": probe_source_kind,
            "source_hint": probe_source_hint or "cnn_classifier",
            "ocr_used": bool(probe_ocr_used),
            "ocr_elements": probe_ocr_fields,
            "top_choices": probe_top_choices,
            "decision_reason": probe_decision_reason,
            "ocr_comment": (
                "OCR usato nella decisione probe."
                if probe_ocr_used
                else "Decisione probe da classificatore immagini (senza OCR)."
            ),
        },
        "su_giu": {
            "images_predicted": int(su_giu_images),
            "majority_label": su_giu_majority,
            "majority_vote_ratio": float(su_giu_vote_ratio),
            "mean_confidence": float(su_giu_mean_conf),
            "mean_prob_su": float(su_giu_mean_prob_su),
            "mean_prob_giu": float(su_giu_mean_prob_giu),
            "source": su_giu_source,
            "checkpoint": su_giu_checkpoint,
            "comment": (
                "Classificazione su/giu eseguita su crop del rettangolo per ogni frame "
                "(majority solo riepilogo, non giudizio globale di cartella)."
                if su_giu_images > 0
                else "Nessuna predizione su/giu disponibile su questa run."
            ),
        },
    }


def _json_script_blob(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _build_review_html(
    *,
    summary: Dict[str, object],
    checks: List[Dict[str, object]],
    run_dir: Path,
    preview_images: List[Path],
) -> str:
    run_dir_s = str(summary.get("run_dir", run_dir.as_posix()))
    original_folder_s = str(summary.get("input_original_folder", ""))
    copy_folder_s = str(summary.get("input_copy_folder", ""))
    pipeline_csv_s = str(summary.get("pipeline_csv", ""))

    step_sections: List[str] = []
    for idx, step in enumerate(checks, start=1):
        step_key = str(step.get("step", f"step_{idx}"))
        status = str(step.get("status", "review"))
        detail_rows: List[str] = []
        for k, v in step.items():
            if k in {"step", "status"}:
                continue
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(k))}</td>"
                f"<td>{html.escape(str(v))}</td>"
                "</tr>"
            )
        if not detail_rows:
            detail_rows.append("<tr><td colspan=\"2\">(nessun dettaglio)</td></tr>")

        step_sections.append(
            "<section class=\"card step-card\" "
            f"data-step-key=\"{html.escape(step_key, quote=True)}\" "
            f"data-step-name=\"{html.escape(step_key, quote=True)}\" "
            f"data-pipeline-status=\"{html.escape(status, quote=True)}\">"
            "<div class=\"step-head\">"
            f"<h2>{idx}. {html.escape(step_key)}</h2>"
            f"<span class=\"status-pill status-{html.escape(status, quote=True)}\">pipeline: {html.escape(status)}</span>"
            "</div>"
            "<table class=\"kv\">"
            "<thead><tr><th>Campo</th><th>Valore</th></tr></thead>"
            f"<tbody>{''.join(detail_rows)}</tbody>"
            "</table>"
            "<div class=\"annot-box\">"
            "<div class=\"annot-row\">"
            "<label><b>Flag QA</b></label>"
            f"<select class=\"step-flag\" data-step-key=\"{html.escape(step_key, quote=True)}\">"
            "<option value=\"\">-- seleziona --</option>"
            "<option value=\"ok\">ok</option>"
            "<option value=\"issue_minor\">problema minore</option>"
            "<option value=\"issue_major\">problema maggiore</option>"
            "<option value=\"wrong_output\">output errato</option>"
            "<option value=\"corrected\">corretto manualmente</option>"
            "<option value=\"needs_reprocess\">da rieseguire</option>"
            "</select>"
            "</div>"
            "<div class=\"annot-row\">"
            "<label><b>Commento</b></label>"
            f"<textarea class=\"step-comment\" data-step-key=\"{html.escape(step_key, quote=True)}\" "
            "placeholder=\"note, motivazione del flag, cosa non torna...\"></textarea>"
            "</div>"
            "<div class=\"annot-row\">"
            "<label><b>Correzione proposta</b></label>"
            f"<input class=\"step-correction\" type=\"text\" data-step-key=\"{html.escape(step_key, quote=True)}\" "
            "placeholder=\"es: vendor=BK | rotation=0 | probe=12 | rect=top|left|bottom|right|\" />"
            "</div>"
            "</div>"
            "</section>"
        )

    preview_html = ""
    if preview_images:
        tiles: List[str] = []
        for i, path in enumerate(preview_images, start=1):
            rel = path.relative_to(run_dir).as_posix()
            tiles.append(
                "<div class=\"img-tile\">"
                f"<div class=\"img-title\">frame {i}</div>"
                f"<img src=\"{html.escape(rel, quote=True)}\" loading=\"lazy\" />"
                f"<div class=\"img-path\">{html.escape(rel)}</div>"
                "</div>"
            )
        preview_html = (
            "<section class=\"card\">"
            "<h2>Preview Immagini</h2>"
            "<div class=\"img-grid\">"
            f"{''.join(tiles)}"
            "</div>"
            "</section>"
        )

    checks_blob = _json_script_blob(checks)
    page = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Pipeline Check Review</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;background:#f5f6f8;color:#222;margin:0;padding:20px;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;margin-bottom:14px;}"
        ".toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;}"
        ".toolbar .status{font-size:12px;color:#555;}"
        ".toolbar .import-btn{display:inline-block;border:1px solid #bbb;border-radius:6px;padding:6px 8px;background:#fafafa;cursor:pointer;}"
        ".toolbar input[type=file]{display:none;}"
        ".summary{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;}"
        ".step-head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;}"
        ".status-pill{border:1px solid #ccc;border-radius:999px;padding:2px 8px;font-size:12px;background:#fafafa;}"
        ".status-ok{border-color:#b7eb8f;background:#f6ffed;}"
        ".status-review{border-color:#ffd591;background:#fff7e6;}"
        ".status-error{border-color:#ffa39e;background:#fff1f0;}"
        ".kv{width:100%;border-collapse:collapse;margin-top:8px;}"
        ".kv th,.kv td{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top;}"
        ".kv th{background:#f2f2f2;text-align:left;}"
        ".annot-box{border:1px solid #ddd;border-radius:8px;padding:8px;margin-top:10px;background:#fcfcfc;}"
        ".annot-row{display:flex;flex-direction:column;gap:4px;margin-bottom:8px;}"
        ".annot-row:last-child{margin-bottom:0;}"
        "input,textarea,select,button{font-family:inherit;}"
        "textarea{min-height:64px;resize:vertical;}"
        ".img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;}"
        ".img-tile{border:1px solid #ddd;border-radius:8px;padding:8px;background:#fff;}"
        ".img-title{font-size:12px;color:#666;margin-bottom:4px;}"
        ".img-path{font-size:11px;color:#666;word-break:break-all;margin-top:4px;}"
        ".img-tile img{width:100%;height:auto;border:1px solid #ccc;border-radius:6px;display:block;}"
        ".pill-wrap{margin-top:4px;}"
        ".pill{display:inline-block;border:1px solid #ccc;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px;background:#fafafa;}"
        ".export-fallback{display:none;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;}"
        ".export-fallback .top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px;}"
        ".export-fallback .hint{font-size:12px;color:#555;}"
        ".export-fallback textarea{width:100%;min-height:220px;resize:vertical;font-family:monospace;font-size:12px;}"
        "</style></head><body>"
        "<h1>Review Check Pipeline (Single Folder)</h1>"
        "<div class=\"summary\">"
        f"<div><b>Run dir:</b> {html.escape(run_dir_s)}</div>"
        f"<div><b>Original:</b> {html.escape(original_folder_s)}</div>"
        f"<div><b>Riferimento input:</b> {html.escape(copy_folder_s)}</div>"
        f"<div><b>Pipeline CSV:</b> {html.escape(pipeline_csv_s)}</div>"
        "</div>"
        "<div class=\"toolbar\">"
        "<button id=\"save-btn\" type=\"button\">Salva Ora</button>"
        "<button id=\"export-json-btn\" type=\"button\">Export JSON</button>"
        "<button id=\"export-csv-btn\" type=\"button\">Export CSV</button>"
        "<label class=\"import-btn\" for=\"import-json-input\">Import JSON</label>"
        "<input id=\"import-json-input\" type=\"file\" accept=\"application/json,.json\" />"
        "<button id=\"clear-btn\" type=\"button\">Pulisci Annotazioni</button>"
        "<label><input id=\"only-flagged\" type=\"checkbox\" /> mostra solo step flaggati</label>"
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
        "<section class=\"card\">"
        "<h2>Annotazione Globale</h2>"
        "<div class=\"annot-box\">"
        "<div class=\"annot-row\">"
        "<label><b>Flag globale</b></label>"
        "<select id=\"run-flag\">"
        "<option value=\"\">-- seleziona --</option>"
        "<option value=\"ok\">ok</option>"
        "<option value=\"review\">review</option>"
        "<option value=\"blocked\">bloccato</option>"
        "<option value=\"needs_reprocess\">da rieseguire</option>"
        "</select>"
        "</div>"
        "<div class=\"annot-row\">"
        "<label><b>Nota globale</b></label>"
        "<textarea id=\"run-note\" placeholder=\"sintesi review, decisione finale, prossime azioni...\"></textarea>"
        "</div>"
        "</div>"
        "</section>"
        f"{preview_html}"
        f"{''.join(step_sections)}"
        "<script>"
        f"const BASE_CHECKS = {checks_blob};"
        f"const RUN_DIR = {json.dumps(run_dir_s)};"
        "const STORAGE_KEY = 'single_folder_pipeline_review::' + RUN_DIR;"
        "let state = null;"
        "let saveTimer = null;"
        "function defaultState(){"
        "  const steps = {};"
        "  BASE_CHECKS.forEach((s)=>{ const k=String(s.step||''); if(k){ steps[k]={flag:'',comment:'',correction:''}; }});"
        "  return {version:1,run_dir:RUN_DIR,updated_at:'',run_flag:'',run_note:'',steps:steps};"
        "}"
        "function readState(){"
        "  const raw=localStorage.getItem(STORAGE_KEY);"
        "  if(!raw) return defaultState();"
        "  try{"
        "    const p=JSON.parse(raw);"
        "    if(!p||typeof p!=='object') return defaultState();"
        "    if(!p.steps||typeof p.steps!=='object') p.steps={};"
        "    BASE_CHECKS.forEach((s)=>{ const k=String(s.step||''); if(k && !p.steps[k]) p.steps[k]={flag:'',comment:'',correction:''}; });"
        "    return p;"
        "  }catch(_e){ return defaultState(); }"
        "}"
        "function writeStatus(msg){ const el=document.getElementById('save-status'); if(el) el.textContent=msg; }"
        "function persist(){ state.updated_at=new Date().toISOString(); localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); writeStatus('Salvato in locale: '+state.updated_at); }"
        "function collectFromUi(){"
        "  const runFlag=document.getElementById('run-flag'); const runNote=document.getElementById('run-note');"
        "  state.run_flag = runFlag ? (runFlag.value||'') : '';"
        "  state.run_note = runNote ? (runNote.value||'') : '';"
        "  document.querySelectorAll('.step-card').forEach((sec)=>{"
        "    const k=sec.dataset.stepKey||''; if(!k) return;"
        "    if(!state.steps[k]) state.steps[k]={flag:'',comment:'',correction:''};"
        "    const flagEl=sec.querySelector('.step-flag');"
        "    const commentEl=sec.querySelector('.step-comment');"
        "    const corrEl=sec.querySelector('.step-correction');"
        "    state.steps[k].flag = flagEl ? (flagEl.value||'') : '';"
        "    state.steps[k].comment = commentEl ? (commentEl.value||'') : '';"
        "    state.steps[k].correction = corrEl ? (corrEl.value||'') : '';"
        "  });"
        "}"
        "function applyToUi(){"
        "  const runFlag=document.getElementById('run-flag'); const runNote=document.getElementById('run-note');"
        "  if(runFlag) runFlag.value = state.run_flag || '';"
        "  if(runNote) runNote.value = state.run_note || '';"
        "  document.querySelectorAll('.step-card').forEach((sec)=>{"
        "    const k=sec.dataset.stepKey||''; if(!k) return;"
        "    const st = state.steps[k] || {flag:'',comment:'',correction:''};"
        "    const flagEl=sec.querySelector('.step-flag');"
        "    const commentEl=sec.querySelector('.step-comment');"
        "    const corrEl=sec.querySelector('.step-correction');"
        "    if(flagEl) flagEl.value = st.flag || '';"
        "    if(commentEl) commentEl.value = st.comment || '';"
        "    if(corrEl) corrEl.value = st.correction || '';"
        "  });"
        "}"
        "function updateSummary(){"
        "  let flagged=0; const byFlag={};"
        "  Object.values(state.steps||{}).forEach((s)=>{ const f=(s&&s.flag)?s.flag:''; if(f){ flagged+=1; byFlag[f]=(byFlag[f]||0)+1; }});"
        "  const total = document.querySelectorAll('.step-card').length;"
        "  const pills = Object.keys(byFlag).sort().map((k)=>`<span class='pill'>${k}: ${byFlag[k]}</span>`).join(' ');"
        "  const el=document.getElementById('annotation-summary');"
        "  if(el){ el.innerHTML=`<div><b>Step annotati:</b> ${flagged}/${total}</div><div><b>Flag globale:</b> ${state.run_flag||'-'}</div>${pills?`<div class='pill-wrap'>${pills}</div>`:''}`; }"
        "}"
        "function applyFilter(){"
        "  const only = !!document.getElementById('only-flagged')?.checked;"
        "  document.querySelectorAll('.step-card').forEach((sec)=>{"
        "    const k=sec.dataset.stepKey||''; const s=(state.steps||{})[k]||{};"
        "    const show = !only || !!s.flag;"
        "    sec.style.display = show ? '' : 'none';"
        "  });"
        "}"
        "function schedulePersist(){ if(saveTimer) clearTimeout(saveTimer); saveTimer=setTimeout(()=>{ collectFromUi(); persist(); updateSummary(); applyFilter(); }, 300); }"
        "function csvEscape(v){ const s=String(v??''); if(s.includes(',')||s.includes('\"')||s.includes('\\n')) return '\"'+s.split('\"').join('\"\"')+'\"'; return s; }"
        "function showExportFallback(filename, text, mode){"
        "  const panel=document.getElementById('export-fallback');"
        "  const fileEl=document.getElementById('export-fallback-filename');"
        "  const hintEl=document.getElementById('export-fallback-hint');"
        "  const area=document.getElementById('export-fallback-text');"
        "  if(fileEl) fileEl.textContent=filename; if(area) area.value=text;"
        "  if(hintEl){ hintEl.textContent = mode==='download_ok' ? 'Download avviato. Se non trovi il file, usa Copia.' : 'Download non disponibile in questa vista. Usa Copia.'; }"
        "  if(panel) panel.style.display='block';"
        "}"
        "function hideExportFallback(){ const p=document.getElementById('export-fallback'); if(p) p.style.display='none'; }"
        "async function copyExportText(){"
        "  const area=document.getElementById('export-fallback-text'); if(!area) return; const text=area.value||'';"
        "  try{ if(navigator.clipboard && navigator.clipboard.writeText){ await navigator.clipboard.writeText(text); writeStatus('Contenuto export copiato negli appunti.'); return; } }catch(_e){}"
        "  area.focus(); area.select();"
        "  try{ document.execCommand('copy'); writeStatus('Contenuto export copiato negli appunti.'); }catch(_e){ writeStatus('Copia non riuscita: seleziona e copia manualmente.'); }"
        "}"
        "function downloadText(filename, text, contentType){"
        "  try{ const blob=new Blob([text],{type:contentType||'text/plain;charset=utf-8'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); setTimeout(()=>{a.remove(); URL.revokeObjectURL(url);},0); return true; }catch(_e){ return false; }"
        "}"
        "function exportJson(){ collectFromUi(); persist(); const payload=JSON.stringify(state, null, 2); const ok=downloadText('pipeline_step_annotations.json', payload, 'application/json;charset=utf-8'); showExportFallback('pipeline_step_annotations.json', payload, ok?'download_ok':'download_blocked'); }"
        "function exportCsv(){"
        "  collectFromUi(); persist();"
        "  const lines=[]; lines.push(['run_dir','run_flag','run_note','step_key','pipeline_status','qa_flag','qa_comment','qa_correction'].join(','));"
        "  document.querySelectorAll('.step-card').forEach((sec)=>{"
        "    const k=sec.dataset.stepKey||''; const pStatus=sec.dataset.pipelineStatus||''; const s=(state.steps||{})[k]||{};"
        "    lines.push([csvEscape(RUN_DIR),csvEscape(state.run_flag||''),csvEscape(state.run_note||''),csvEscape(k),csvEscape(pStatus),csvEscape(s.flag||''),csvEscape(s.comment||''),csvEscape(s.correction||'')].join(','));"
        "  });"
        "  const payload=lines.join('\\n'); const ok=downloadText('pipeline_step_annotations.csv', payload, 'text/csv;charset=utf-8'); showExportFallback('pipeline_step_annotations.csv', payload, ok?'download_ok':'download_blocked');"
        "}"
        "function importJsonFile(file){"
        "  const r=new FileReader();"
        "  r.onload=()=>{"
        "    try{ const p=JSON.parse(String(r.result||'')); if(!p||typeof p!=='object'||!p.steps||typeof p.steps!=='object'){ alert('File JSON non valido.'); return; } state=p; if(!state.steps) state.steps={}; BASE_CHECKS.forEach((s)=>{ const k=String(s.step||''); if(k && !state.steps[k]) state.steps[k]={flag:'',comment:'',correction:''}; }); localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); applyToUi(); updateSummary(); applyFilter(); writeStatus('Import completato.'); }catch(_e){ alert('Errore parsing JSON.'); }"
        "  };"
        "  r.readAsText(file,'utf-8');"
        "}"
        "function clearAnnotations(){ if(!confirm('Cancellare tutte le annotazioni locali?')) return; state=defaultState(); localStorage.removeItem(STORAGE_KEY); applyToUi(); updateSummary(); applyFilter(); writeStatus('Annotazioni cancellate.'); }"
        "function bind(){"
        "  document.querySelectorAll('.step-flag').forEach((el)=>el.addEventListener('change',()=>{ schedulePersist(); applyFilter(); }));"
        "  document.querySelectorAll('.step-comment,.step-correction').forEach((el)=>el.addEventListener('input',schedulePersist));"
        "  document.getElementById('run-flag')?.addEventListener('change',schedulePersist);"
        "  document.getElementById('run-note')?.addEventListener('input',schedulePersist);"
        "  document.getElementById('save-btn')?.addEventListener('click',()=>{ collectFromUi(); persist(); updateSummary(); applyFilter(); });"
        "  document.getElementById('export-json-btn')?.addEventListener('click',exportJson);"
        "  document.getElementById('export-csv-btn')?.addEventListener('click',exportCsv);"
        "  document.getElementById('export-fallback-copy-btn')?.addEventListener('click',()=>{ copyExportText(); });"
        "  document.getElementById('export-fallback-close-btn')?.addEventListener('click',()=>{ hideExportFallback(); });"
        "  document.getElementById('clear-btn')?.addEventListener('click',clearAnnotations);"
        "  document.getElementById('only-flagged')?.addEventListener('change',applyFilter);"
        "  document.getElementById('import-json-input')?.addEventListener('change',(ev)=>{ const input=ev.target; const file=input&&input.files&&input.files[0]?input.files[0]:null; if(file) importJsonFile(file); if(input) input.value=''; });"
        "}"
        "state=readState(); bind(); applyToUi(); updateSummary(); applyFilter(); writeStatus('Annotazioni pronte (autosave locale attivo).');"
        "</script>"
        "</body></html>"
    )
    return page


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run main pipeline safely on one input folder copy.")
    p.add_argument("--input-folder", type=Path, required=False, help="Cartella acquisizione sorgente (originale).")
    p.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/runs/operational_single_folder",
        help="Root area operativa dove creare la copia e gli output.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Nome run (cartella sotto work-root). Se vuoto usa timestamp+folder.",
    )
    p.add_argument("--python-bin", type=str, default=sys.executable, help="Python per lanciare la pipeline principale.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-per-folder", type=int, default=80)
    p.add_argument("--rotation-max-samples", type=int, default=24)
    p.add_argument(
        "--low-confidence-policy",
        type=str,
        default="review",
        choices=("review", "error", "ask_user"),
        help="Policy bassa confidenza per la pipeline.",
    )
    p.add_argument("--vendor-min-confidence", type=float, default=0.50)
    p.add_argument("--probe-min-confidence", type=float, default=0.50)
    p.add_argument(
        "--lr-marker-template-policy",
        type=str,
        choices=("historical_best_then_derived", "historical_best", "derived_folder"),
        default="historical_best_then_derived",
        help=(
            "Policy template LR marker: historical_best_then_derived usa prima lo storico "
            "e passa al template di cartella solo se serve; historical_best usa solo "
            "lo storico; derived_folder crea subito il template di cartella."
        ),
    )
    p.add_argument(
        "--lr-marker-method",
        type=str,
        choices=("bundle", "classical"),
        default="bundle",
        help=(
            "Metodo ufficiale LR marker. Default bundle: usa il detector del bundle "
            "con assi mediani del rettangolo ecografico ufficiale."
        ),
    )
    p.add_argument(
        "--lr-marker-bundle-dir",
        type=Path,
        default=DEFAULT_ORIENTATION_MARKER_BUNDLE_DIR,
        help="Cartella unpacked del bundle orientation_marker_detector.",
    )
    p.add_argument(
        "--lr-marker-bundle-zip",
        type=Path,
        default=DEFAULT_ORIENTATION_MARKER_BUNDLE_ZIP,
        help="Zip del bundle orientation_marker_detector.",
    )
    p.add_argument(
        "--lr-marker-bundle-library-root",
        type=Path,
        default=DEFAULT_ORIENTATION_MARKER_BUNDLE_LIBRARY_ROOT,
        help="Root templates del bundle usata per i marker vendor.",
    )
    p.add_argument(
        "--lr-marker-manual-seeds-file",
        type=Path,
        default=None,
        help="JSON opzionale con rettangoli marker corretti da usare come seed del template LR.",
    )
    p.add_argument(
        "--lr-marker-review-file",
        type=Path,
        default=DEFAULT_LR_MARKER_REVIEW_FILE,
        help="JSON opzionale con template LR marker storici accettati/scartati per vendor.",
    )
    p.add_argument(
        "--lt-min-confidence",
        type=float,
        default=0.55,
        help="Soglia confidenza media L/T sotto cui segnare review.",
    )
    p.add_argument(
        "--lt-rect-checkpoint",
        type=Path,
        default=Path("artifacts/30_models/lt_training_transrectal_rect_only_v2_trecall/best_model.pt"),
        help="Checkpoint classificatore L/T su crop rettangolo ecografico.",
    )
    p.add_argument(
        "--disable-lt-rect-classifier",
        action="store_true",
        help="Disattiva step classificazione L/T sui crop del rect.",
    )
    p.add_argument(
        "--disable-rect-depth-autonomous",
        action="store_true",
        help="Disattiva lo step ufficiale RECT_DEPTH autonomo OCR/classico/ranker.",
    )
    p.add_argument(
        "--rect-depth-max-images",
        type=int,
        default=0,
        help="Numero massimo immagini per RECT_DEPTH autonomo (0=tutta la cartella).",
    )
    p.add_argument(
        "--rect-depth-min-accepted-ratio",
        type=float,
        default=0.80,
        help="Quota minima accepted per considerare ok lo step RECT_DEPTH.",
    )
    p.add_argument(
        "--rect-depth-subprocess-timeout-sec",
        type=float,
        default=0.0,
        help="Timeout globale subprocess RECT_DEPTH (0=nessun timeout globale).",
    )
    p.add_argument(
        "--no-generated-images",
        action="store_true",
        help="Non genera PNG di preview/overlay nelle evidenze; usa solo riferimenti agli originali.",
    )
    p.add_argument(
        "--no-split-symlinks",
        action="store_true",
        help="Non crea cartelle SU/GIU e L/T con symlink per ridurre inode/spazio nel batch.",
    )
    p.add_argument(
        "--exclude-images-file",
        type=Path,
        default=None,
        help="JSON o testo con immagini relative alla cartella input da escludere dalla run.",
    )
    p.add_argument("--rect-per-image-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rect-worker-images-json", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--rect-worker-checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--rect-worker-batch-size", type=int, default=16, help=argparse.SUPPRESS)
    p.add_argument("--rect-worker-rotation", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--rect-worker-output-json", type=Path, default=None, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if bool(args.rect_per_image_worker):
        images_json = args.rect_worker_images_json.expanduser().resolve() if args.rect_worker_images_json else None
        checkpoint = args.rect_worker_checkpoint.expanduser().resolve() if args.rect_worker_checkpoint else None
        output_json = args.rect_worker_output_json.expanduser().resolve() if args.rect_worker_output_json else None
        if images_json is None or checkpoint is None or output_json is None:
            raise RuntimeError("rect worker args mancanti")
        return _run_rect_per_image_worker(
            images_json_path=images_json,
            checkpoint_path=checkpoint,
            batch_size=max(1, int(args.rect_worker_batch_size)),
            rotate_deg_clockwise=int(args.rect_worker_rotation),
            output_json_path=output_json,
        )

    if args.input_folder is None:
        raise RuntimeError("--input-folder è obbligatorio")
    input_folder = args.input_folder.expanduser().resolve()
    if not input_folder.is_dir():
        raise RuntimeError(f"Input folder non valida: {input_folder}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _slug(args.run_name) if str(args.run_name).strip() else f"{now}_{_slug(input_folder.name)}"
    work_root = args.work_root.expanduser().resolve()
    run_dir = work_root / run_name
    if _is_subpath(work_root, input_folder):
        raise RuntimeError(
            "Configurazione non valida: work_root e' dentro input_folder. "
            "Questo causa copia ricorsiva infinita. Usa una work_root esterna."
        )
    if _is_subpath(run_dir, input_folder):
        raise RuntimeError(
            "Configurazione non valida: run_dir e' dentro input_folder. "
            "Questo causa copia ricorsiva infinita."
        )
    if run_dir.exists():
        if not run_dir.is_dir():
            raise RuntimeError(f"Run path esistente ma non directory: {run_dir}")
        allowed_existing = {"run_state.json", "review_annotations.json", "manual_lr_marker_seeds.json"}
        existing_entries = {p.name for p in run_dir.iterdir()}
        disallowed = existing_entries.difference(allowed_existing)
        if disallowed:
            disallowed_txt = ", ".join(sorted(disallowed))
            raise RuntimeError(
                "Run dir già esistente e non vuota (voci non consentite): "
                f"{run_dir} -> {disallowed_txt}"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    input_ref_root = run_dir / "input_ref"
    input_ref_folder = input_ref_root / input_folder.name
    pipeline_output = run_dir / "pipeline_output"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    input_ref_root.mkdir(parents=True, exist_ok=True)

    _emit_event(
        "run_started",
        run_name=run_name,
        run_dir=run_dir.as_posix(),
        input_original=input_folder.as_posix(),
    )
    _emit_event(
        "copy_started",
        src=input_folder.as_posix(),
        dst=input_ref_folder.as_posix(),
        mode="reference_symlink",
    )
    if input_ref_folder.exists():
        raise RuntimeError(f"Riferimento input già esistente: {input_ref_folder}")
    excluded_images_rel = _load_excluded_image_rels(args.exclude_images_file, input_folder)
    input_reference_meta: Dict[str, Any] = {
        "mode": "reference_symlink",
        "raw_total": 0,
        "included": 0,
        "excluded_existing": 0,
        "excluded_images_rel": excluded_images_rel,
        "excluded_existing_rel": [],
    }
    if excluded_images_rel:
        input_reference_meta = _create_filtered_input_symlinks(
            input_folder=input_folder,
            input_ref_folder=input_ref_folder,
            excluded_rels=excluded_images_rel,
        )
    else:
        try:
            input_ref_folder.symlink_to(input_folder, target_is_directory=True)
        except Exception as exc:
            raise RuntimeError(
                "Impossibile creare il riferimento (symlink) alla cartella input. "
                "Verifica permessi filesystem."
            ) from exc
    _emit_event(
        "copy_completed",
        dst=input_ref_folder.as_posix(),
        mode=str(input_reference_meta.get("mode", "reference_symlink")),
        excluded_images_count=int(input_reference_meta.get("excluded_existing", 0) or 0),
    )
    raw_images_before_pipeline = _collect_acquisition_images(input_folder)
    _emit_event(
        "raw_images_ready_for_input_copy",
        count=len(_collect_acquisition_images(input_ref_folder)),
        original_count=len(raw_images_before_pipeline),
        excluded_images_count=int(input_reference_meta.get("excluded_existing", 0) or 0),
        input_original=input_folder.as_posix(),
        input_copy_folder=input_ref_folder.as_posix(),
        mode=str(input_reference_meta.get("mode", "reference_symlink")),
    )

    pipeline_script = (SCRIPT_DIR / "predict_fss_head_from_acquisitions.py").resolve()
    if not pipeline_script.is_file():
        raise RuntimeError(f"Script pipeline non trovato: {pipeline_script}")

    python_bin = _pick_python(args.python_bin)
    cmd = [
        python_bin,
        pipeline_script.as_posix(),
        "--dataset-root",
        input_ref_root.as_posix(),
        "--output-dir",
        pipeline_output.as_posix(),
        "--max-folders",
        "1",
        "--batch-size",
        str(int(args.batch_size)),
        "--sample-per-folder",
        str(int(args.sample_per_folder)),
        "--rotation-max-samples",
        str(int(args.rotation_max_samples)),
        "--low-confidence-policy",
        str(args.low_confidence_policy),
        "--vendor-min-confidence",
        str(float(args.vendor_min_confidence)),
        "--probe-min-confidence",
        str(float(args.probe_min_confidence)),
        "--lr-marker-template-policy",
        str(args.lr_marker_template_policy),
        "--lr-marker-method",
        str(args.lr_marker_method),
        "--lr-marker-bundle-dir",
        args.lr_marker_bundle_dir.expanduser().resolve().as_posix(),
        "--lr-marker-bundle-zip",
        args.lr_marker_bundle_zip.expanduser().resolve().as_posix(),
        "--lr-marker-bundle-library-root",
        args.lr_marker_bundle_library_root.expanduser().resolve().as_posix(),
        "--lr-marker-review-file",
        args.lr_marker_review_file.expanduser().resolve().as_posix(),
        "--lt-min-confidence",
        str(float(args.lt_min_confidence)),
        "--lt-rect-checkpoint",
        args.lt_rect_checkpoint.expanduser().resolve().as_posix(),
        "--rect-depth-max-images",
        str(int(args.rect_depth_max_images)),
        "--rect-depth-min-accepted-ratio",
        str(float(args.rect_depth_min_accepted_ratio)),
        "--rect-depth-subprocess-timeout-sec",
        str(float(args.rect_depth_subprocess_timeout_sec)),
    ]
    if args.lr_marker_manual_seeds_file is not None:
        cmd.extend([
            "--lr-marker-manual-seeds-file",
            args.lr_marker_manual_seeds_file.expanduser().resolve().as_posix(),
        ])
    if bool(args.disable_lt_rect_classifier):
        cmd.append("--disable-lt-rect-classifier")
    if bool(args.disable_rect_depth_autonomous):
        cmd.append("--disable-rect-depth-autonomous")

    _emit_event("pipeline_started", command=" ".join(cmd))
    stdout_lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        print(line.rstrip("\n"), flush=True)
    proc.wait()
    combined_stdout = "".join(stdout_lines)
    _emit_event("pipeline_finished", returncode=int(proc.returncode))
    (logs_dir / "pipeline_stdout.log").write_text(combined_stdout, encoding="utf-8")
    (logs_dir / "pipeline_stderr.log").write_text("", encoding="utf-8")
    (logs_dir / "pipeline_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    pipeline_summary: Dict[str, Any] = {}
    pipeline_summary_path = pipeline_output / "summary.json"
    if pipeline_summary_path.is_file():
        try:
            pipeline_summary = json.loads(pipeline_summary_path.read_text(encoding="utf-8"))
            if not isinstance(pipeline_summary, dict):
                pipeline_summary = {}
        except Exception:
            pipeline_summary = {}

    if proc.returncode != 0:
        _emit_event("run_failed", reason=f"pipeline_returncode_{proc.returncode}", logs_dir=logs_dir.as_posix())
        raise RuntimeError(
            "Pipeline fallita. Controlla i log in "
            f"{logs_dir} (returncode={proc.returncode})."
        )

    csv_path = pipeline_output / "folder_fss_head_predictions.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"Output CSV non trovato: {csv_path}")
    row, predictions_rows = _load_first_prediction(csv_path)
    if row is None:
        warnings_txt = ""
        summary_json_path = pipeline_output / "summary.json"
        if summary_json_path.is_file():
            try:
                summary_obj = json.loads(summary_json_path.read_text(encoding="utf-8"))
                warn_items = summary_obj.get("warnings", [])
                if isinstance(warn_items, list) and warn_items:
                    warnings_txt = " | ".join(str(x).strip() for x in warn_items if str(x).strip())
            except Exception:
                warnings_txt = ""
        reason_parts = [f"Nessuna riga prediction in {csv_path.name}"]
        if warnings_txt:
            reason_parts.append(warnings_txt)
        reason_txt = " ; ".join(reason_parts)
        row = {
            "status": "review",
            "review_reasons": reason_txt,
        }
        _emit_event(
            "pipeline_empty_predictions",
            csv_path=csv_path.as_posix(),
            predictions_rows=0,
            review_reasons=reason_txt,
        )
    _emit_event(
        "pipeline_row_decisions",
        vendor=str(row.get("vendor_pred", "") or row.get("vendor_predicted", "") or ""),
        vendor_confidence=float(_safe_float(row.get("vendor_conf", row.get("vendor_confidence", 0.0)), 0.0)),
        rotation_deg_clockwise=int(_safe_int(row.get("rotation_deg_clockwise", 0), 0)),
        rect_echo=str(row.get("line_11_rect_echo", "") or ""),
        probe_id=str(row.get("line_03_id_probe", "") or ""),
        probe_confidence=float(_safe_float(row.get("line_03_probe_conf", row.get("line_03_probe_confidence", 0.0)), 0.0)),
        line13=str(row.get("line_13_rect_name_echo", "") or ""),
        line14=str(row.get("line_14_rect_name_probe", "") or ""),
    )
    probe_name_map = _load_probe_name_map()
    probe_id_row = _normalize_probe_id(row.get("line_03_id_probe", ""))
    if probe_id_row and probe_id_row in probe_name_map:
        row["line_03_probe_name"] = probe_name_map.get(probe_id_row, "")
    checks = _build_step_checks(
        row=row,
        vendor_min_conf=float(args.vendor_min_confidence),
        probe_min_conf=float(args.probe_min_confidence),
        lt_min_conf=float(args.lt_min_confidence),
        probe_name_map=probe_name_map,
    )
    _emit_event("checks_built", checks_count=len(checks))
    for item in checks:
        _emit_event(
            "step_check",
            step=str(item.get("step", "")),
            status=str(item.get("status", "review")),
        )
    acquisition_images = _collect_acquisition_images(input_ref_folder)
    raw_gallery = _select_uniform_subset(acquisition_images, limit=24)
    unique_images, duplicate_pairs = _analyze_duplicates(acquisition_images)
    preview_images = _select_uniform_subset(raw_gallery, limit=8)
    generate_images = not bool(args.no_generated_images)
    decision_evidence = _build_decision_evidence(
        row=row,
        probe_name_map=probe_name_map,
        vendor_min_conf=float(args.vendor_min_confidence),
        probe_min_conf=float(args.probe_min_confidence),
    )

    vendor_probe_samples_rel: List[str] = []
    cls_sample_pool = unique_images if unique_images else acquisition_images
    cls_sample_paths = _select_uniform_subset(cls_sample_pool, limit=3)
    for idx, path in enumerate(cls_sample_paths, start=1):
        try:
            if generate_images:
                target = run_dir / "evidence" / f"vendor_probe_sample_{idx}.png"
                _save_image_preview(source=path, target=target, rotate_deg_clockwise=0)
                vendor_probe_samples_rel.append(target.relative_to(run_dir).as_posix())
            else:
                vendor_probe_samples_rel.append(path.relative_to(run_dir).as_posix())
        except Exception:
            continue

    duplicate_examples: List[Dict[str, Any]] = []
    for item in duplicate_pairs[:18]:
        kept_p = item["kept_path"]
        removed_p = item["removed_path"]
        try:
            kept_rel = kept_p.relative_to(run_dir).as_posix()
            removed_rel = removed_p.relative_to(run_dir).as_posix()
        except Exception:
            continue
        duplicate_examples.append(
            {
                "kept_rel": kept_rel,
                "removed_rel": removed_rel,
                "sha1_prefix": str(item.get("sha1", ""))[:12],
                "size_bytes": int(item.get("size_bytes", 0)),
            }
        )

    rotation_deg = _safe_int(row.get("rotation_deg_clockwise", "0"), 0)
    rotation_source = str(row.get("rotation_source", "") or "").strip()
    rotation_vote_ratio = _safe_float(row.get("rotation_vote_ratio", "0"), 0.0)
    rotation_votes_total = _safe_int(row.get("rotation_votes_total", "0"), 0)
    rotation_samples_checked = _safe_int(row.get("rotation_samples_checked", "0"), 0)
    rotation_decision_reason = str(row.get("rotation_decision_reason", "") or "").strip()
    rotation_osd_debug = _safe_json_obj(str(row.get("rotation_osd_debug_json", "") or ""))
    rotation_ocr_debug = _safe_json_obj(str(row.get("rotation_ocr_debug_json", "") or ""))
    rotation_sample_path: Optional[Path] = None
    sample_pool = unique_images if unique_images else acquisition_images
    if sample_pool:
        rotation_sample_path = _select_uniform_subset(sample_pool, limit=1)[0]
    rotation_before_rel = ""
    rotation_after_rel = ""
    rotation_sample_original_rel = ""
    rotation_error = ""
    if rotation_sample_path is not None:
        try:
            rotation_sample_original_rel = rotation_sample_path.relative_to(run_dir).as_posix()
            if generate_images:
                rot_before = run_dir / "evidence" / "rotation_sample_before.png"
                rot_after = run_dir / "evidence" / "rotation_sample_after.png"
                _save_image_preview(source=rotation_sample_path, target=rot_before, rotate_deg_clockwise=0)
                _save_image_preview(source=rotation_sample_path, target=rot_after, rotate_deg_clockwise=rotation_deg)
                rotation_before_rel = rot_before.relative_to(run_dir).as_posix()
                rotation_after_rel = rot_after.relative_to(run_dir).as_posix()
            else:
                rotation_before_rel = rotation_sample_original_rel
                rotation_after_rel = ""
                rotation_error = "generated_images_disabled"
        except Exception as exc:
            rotation_error = str(exc)

    rect_overlay_rel = ""
    rect_overlay_source_rel = ""
    rect_overlay_error = ""
    rect_coords = _parse_rect_coords(str(row.get("line_11_rect_echo", "") or ""))
    rect_sample_path: Optional[Path] = None
    if unique_images:
        rect_sample_path = _select_uniform_subset(unique_images, limit=1)[0]
    elif acquisition_images:
        rect_sample_path = _select_uniform_subset(acquisition_images, limit=1)[0]
    if rect_sample_path is not None and rect_coords is not None:
        try:
            rect_overlay_source_rel = rect_sample_path.relative_to(run_dir).as_posix()
            if generate_images:
                rect_overlay_path = run_dir / "evidence" / "rect_overlay_sample.png"
                _save_rect_overlay_preview(
                    source=rect_sample_path,
                    target=rect_overlay_path,
                    rect_coords=rect_coords,
                    color=(255, 32, 32),
                )
                rect_overlay_rel = rect_overlay_path.relative_to(run_dir).as_posix()
            else:
                rect_overlay_error = "generated_images_disabled"
        except Exception as exc:
            rect_overlay_error = str(exc)

    line13_overlay_rel = ""
    line14_overlay_rel = ""
    line13_crop_rel = ""
    line13_source_rel = ""
    line13_coords = _parse_rect_coords(str(row.get("line_13_rect_name_echo", "") or ""))
    line13_pre_dark_trim_coords = _parse_rect_coords(str(row.get("line_13_pre_dark_trim_rect_name_echo", "") or ""))
    line14_coords = _parse_rect_coords(str(row.get("line_14_rect_name_probe", "") or ""))
    if rect_sample_path is not None and line13_coords is not None:
        try:
            line13_source_rel = rect_sample_path.relative_to(run_dir).as_posix()
            if generate_images:
                p13 = run_dir / "evidence" / "line13_template_overlay.png"
                _save_line13_compare_overlay_preview(
                    source=rect_sample_path,
                    target=p13,
                    final_rect_coords=line13_coords,
                    pre_dark_trim_rect_coords=line13_pre_dark_trim_coords,
                )
                line13_overlay_rel = p13.relative_to(run_dir).as_posix()
                p13_crop = run_dir / "evidence" / "line13_template_crop.png"
                _save_rect_crop_preview(
                    source=rect_sample_path,
                    target=p13_crop,
                    rect_coords=line13_coords,
                    pad_ratio=0.025,
                    min_side=820,
                    max_side=1900,
                )
                line13_crop_rel = p13_crop.relative_to(run_dir).as_posix()
        except Exception:
            line13_overlay_rel = ""
            line13_crop_rel = ""
    if rect_sample_path is not None and line14_coords is not None:
        if generate_images:
            try:
                p14 = run_dir / "evidence" / "line14_template_overlay.png"
                _save_rect_overlay_preview(
                    source=rect_sample_path,
                    target=p14,
                    rect_coords=line14_coords,
                    color=(32, 180, 80),
                )
                line14_overlay_rel = p14.relative_to(run_dir).as_posix()
            except Exception:
                line14_overlay_rel = ""

    rect_red_evidence = _build_rect_red_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output,
        pipeline_summary=pipeline_summary,
        row=row,
        generate_images=generate_images,
    )
    if bool(rect_red_evidence.get("available", False)):
        _emit_event(
            "rect_red_built",
            items=int(len(rect_red_evidence.get("items", []))),
            winner=str(rect_red_evidence.get("winner_group", "") or ""),
            margin_pct=float(rect_red_evidence.get("margin_pct", 0.0) or 0.0),
        )
    else:
        _emit_event(
            "rect_red_skipped",
            reason=str(rect_red_evidence.get("error", "") or "unknown"),
        )

    rect_per_image_evidence = _build_rect_per_image_evidence(
        run_dir=run_dir,
        row=row,
        unique_images=unique_images,
        acquisition_images=acquisition_images,
        batch_size=max(1, int(args.batch_size)),
        rotation_deg_clockwise=int(rotation_deg),
        python_bin=str(python_bin),
        generate_images=generate_images,
    )
    if bool(rect_per_image_evidence.get("available", False)):
        _emit_event(
            "rect_per_image_built",
            items=int(rect_per_image_evidence.get("images_total", 0)),
            model_device=str(rect_per_image_evidence.get("model_device", "") or ""),
        )
    else:
        _emit_event(
            "rect_per_image_skipped",
            reason=str(rect_per_image_evidence.get("error", "") or "unknown"),
        )
    su_giu_per_image_evidence = _build_su_giu_per_image_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output,
    )
    if bool(su_giu_per_image_evidence.get("available", False)):
        _emit_event(
            "su_giu_per_image_built",
            items=int(su_giu_per_image_evidence.get("images_total", 0)),
            label_counts=su_giu_per_image_evidence.get("label_counts", {}),
        )
    else:
        _emit_event(
            "su_giu_per_image_skipped",
            reason=str(su_giu_per_image_evidence.get("error", "") or "unknown"),
        )
    if bool(args.no_split_symlinks):
        su_giu_split_folders = {
            "available": False,
            "error": "split_symlinks_disabled",
            "mode": "disabled",
            "root_dir_rel": "",
            "counts": {"su": 0, "giu": 0, "other": 0},
            "created_total": 0,
            "examples": [],
        }
    else:
        su_giu_split_folders = _build_su_giu_split_folders(
            run_dir=run_dir,
            su_giu_per_image_evidence=su_giu_per_image_evidence,
        )
    if bool(su_giu_split_folders.get("available", False)):
        _emit_event(
            "su_giu_split_folders_built",
            root=str(su_giu_split_folders.get("root_dir_rel", "") or ""),
            counts=su_giu_split_folders.get("counts", {}),
            created_total=int(su_giu_split_folders.get("created_total", 0) or 0),
        )
    else:
        _emit_event(
            "su_giu_split_folders_skipped",
            reason=str(su_giu_split_folders.get("error", "") or "unknown"),
        )
    lr_marker_per_image_evidence = _build_lr_marker_per_image_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output,
    )
    if bool(lr_marker_per_image_evidence.get("available", False)):
        _emit_event(
            "lr_marker_per_image_built",
            items=int(lr_marker_per_image_evidence.get("images_total", 0)),
            label_counts=lr_marker_per_image_evidence.get("label_counts", {}),
            best=lr_marker_per_image_evidence.get("best", {}),
        )
    else:
        _emit_event(
            "lr_marker_per_image_skipped",
            reason=str(lr_marker_per_image_evidence.get("error", "") or "unknown"),
        )
    lt_per_image_evidence = _build_lt_per_image_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output,
    )
    if bool(lt_per_image_evidence.get("available", False)):
        _emit_event(
            "lt_per_image_built",
            items=int(lt_per_image_evidence.get("images_total", 0)),
            label_counts=lt_per_image_evidence.get("label_counts", {}),
        )
    else:
        _emit_event(
            "lt_per_image_skipped",
            reason=str(lt_per_image_evidence.get("error", "") or "unknown"),
        )
    if bool(args.no_split_symlinks):
        lt_split_folders = {
            "available": False,
            "error": "split_symlinks_disabled",
            "mode": "disabled",
            "root_dir_rel": "",
            "counts": {"l": 0, "t": 0, "other": 0},
            "created_total": 0,
            "examples": [],
        }
    else:
        lt_split_folders = _build_lt_split_folders(
            run_dir=run_dir,
            lt_per_image_evidence=lt_per_image_evidence,
        )
    if bool(lt_split_folders.get("available", False)):
        _emit_event(
            "lt_split_folders_built",
            root=str(lt_split_folders.get("root_dir_rel", "") or ""),
            counts=lt_split_folders.get("counts", {}),
            created_total=int(lt_split_folders.get("created_total", 0) or 0),
        )
    else:
        _emit_event(
            "lt_split_folders_skipped",
            reason=str(lt_split_folders.get("error", "") or "unknown"),
        )
    rect_depth_per_image_evidence = _build_rect_depth_per_image_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output,
    )
    if bool(rect_depth_per_image_evidence.get("available", False)):
        _emit_event(
            "rect_depth_per_image_built",
            items=int(rect_depth_per_image_evidence.get("images_total", 0)),
            status_counts=rect_depth_per_image_evidence.get("status_counts", {}),
            accepted_ratio=float(rect_depth_per_image_evidence.get("accepted_ratio", 0.0) or 0.0),
        )
    else:
        _emit_event(
            "rect_depth_per_image_skipped",
            reason=str(rect_depth_per_image_evidence.get("error", "") or "unknown"),
        )

    model_evidence = {
        "vendor_checkpoint": str(pipeline_summary.get("vendor_checkpoint", "") or ""),
        "probe_checkpoint": str(pipeline_summary.get("probe_checkpoint", "") or ""),
        "rect_checkpoint_global": str(pipeline_summary.get("rect_checkpoint", "") or ""),
        "rect_checkpoint_used": str(row.get("line_11_model_checkpoint", "") or ""),
        "su_giu_checkpoint_global": str(pipeline_summary.get("su_giu_rect_checkpoint", "") or ""),
        "su_giu_checkpoint_used": str(row.get("su_giu_checkpoint", "") or pipeline_summary.get("su_giu_rect_checkpoint", "") or ""),
        "su_giu_enabled": bool(pipeline_summary.get("su_giu_rect_classifier_enabled", False)),
        "su_giu_image_size": int(pipeline_summary.get("su_giu_rect_image_size", 0) or 0),
        "su_giu_batch_size": int(pipeline_summary.get("su_giu_rect_batch_size", 0) or 0),
        "su_giu_class_names": (
            pipeline_summary.get("su_giu_rect_class_names")
            if isinstance(pipeline_summary.get("su_giu_rect_class_names"), list)
            else []
        ),
        "lr_marker_method": str(pipeline_summary.get("lr_marker_method", "") or row.get("lr_marker_method", "") or ""),
        "lr_marker_classical_enabled": bool(pipeline_summary.get("lr_marker_classical_enabled", False)),
        "lr_marker_bundle_enabled": bool(pipeline_summary.get("lr_marker_bundle_enabled", False)),
        "lr_marker_bundle_dir": str(pipeline_summary.get("lr_marker_bundle_dir", "") or ""),
        "lr_marker_bundle_zip": str(pipeline_summary.get("lr_marker_bundle_zip", "") or ""),
        "lr_marker_bundle_library_root": str(pipeline_summary.get("lr_marker_bundle_library_root", "") or ""),
        "lr_marker_template_roots": (
            pipeline_summary.get("lr_marker_template_roots")
            if isinstance(pipeline_summary.get("lr_marker_template_roots"), list)
            else []
        ),
        "lr_marker_template_cache_meta": (
            pipeline_summary.get("lr_marker_template_cache_meta")
            if isinstance(pipeline_summary.get("lr_marker_template_cache_meta"), dict)
            else {}
        ),
        "lr_marker_min_match_score": float(pipeline_summary.get("lr_marker_min_match_score", 0.56) or 0.56),
        "lr_marker_full_crop_fallback_threshold": float(
            pipeline_summary.get("lr_marker_full_crop_fallback_threshold", 0.55) or 0.55
        ),
        "lr_marker_expanded_search_threshold": float(
            pipeline_summary.get("lr_marker_expanded_search_threshold", 0.66) or 0.66
        ),
        "lr_marker_expanded_search_steps": (
            pipeline_summary.get("lr_marker_expanded_search_steps")
            if isinstance(pipeline_summary.get("lr_marker_expanded_search_steps"), list)
            else []
        ),
        "lr_marker_blank_template_max_value": int(pipeline_summary.get("lr_marker_blank_template_max_value", 3) or 3),
        "lr_marker_min_sugiu_confidence": float(pipeline_summary.get("lr_marker_min_sugiu_confidence", 0.80) or 0.80),
        "lr_marker_template_policy": str(pipeline_summary.get("lr_marker_template_policy", "") or ""),
        "lt_checkpoint_global": str(pipeline_summary.get("lt_rect_checkpoint", "") or ""),
        "lt_checkpoint_used": str(row.get("lt_checkpoint", "") or pipeline_summary.get("lt_rect_checkpoint", "") or ""),
        "lt_enabled": bool(pipeline_summary.get("lt_rect_classifier_enabled", False)),
        "lt_image_size": int(pipeline_summary.get("lt_rect_image_size", 0) or 0),
        "lt_batch_size": int(pipeline_summary.get("lt_rect_batch_size", 0) or 0),
        "lt_class_names": (
            pipeline_summary.get("lt_rect_class_names")
            if isinstance(pipeline_summary.get("lt_rect_class_names"), list)
            else []
        ),
        "rect_depth_autonomous_enabled": bool(pipeline_summary.get("rect_depth_autonomous_enabled", False)),
        "rect_depth_autonomous_max_images": int(pipeline_summary.get("rect_depth_autonomous_max_images", 0) or 0),
        "rect_depth_autonomous_min_accepted_ratio": float(
            pipeline_summary.get("rect_depth_autonomous_min_accepted_ratio", 0.0) or 0.0
        ),
        "pipeline_stage_rect_depth": str(pipeline_summary.get("pipeline_stage_rect_depth", "") or ""),
        "rect_vendor_map_path": str(pipeline_summary.get("rect_vendor_map_path", "") or ""),
        "rect_vendor_min_confidence": float(pipeline_summary.get("rect_vendor_min_confidence", 0.7) or 0.7),
        "rect_vendor_map_loaded": (
            pipeline_summary.get("rect_vendor_map_loaded")
            if isinstance(pipeline_summary.get("rect_vendor_map_loaded"), dict)
            else {}
        ),
        "pipeline_stage_line11_rect_echo": str(pipeline_summary.get("pipeline_stage_line11_rect_echo", "") or ""),
        "pipeline_stage_su_giu_rect": str(pipeline_summary.get("pipeline_stage_su_giu_rect", "") or ""),
        "pipeline_stage_lr_marker": str(pipeline_summary.get("pipeline_stage_lr_marker", "") or ""),
        "pipeline_stage_lt_rect": str(pipeline_summary.get("pipeline_stage_lt_rect", "") or ""),
        "pipeline_stage_probe": str(pipeline_summary.get("pipeline_stage_probe", "") or ""),
        "pipeline_stage_line13_rect_name_echo": str(pipeline_summary.get("pipeline_stage_line13_rect_name_echo", "") or ""),
        "pipeline_stage_line14_rect_name_probe": str(pipeline_summary.get("pipeline_stage_line14_rect_name_probe", "") or ""),
        "line13_model_enabled": bool(pipeline_summary.get("line13_model_enabled", False)),
        "line13_checkpoint_global": str(pipeline_summary.get("line13_checkpoint_global", "") or ""),
        "line13_checkpoint_used": str(
            row.get("line_13_model_checkpoint", "")
            or pipeline_summary.get("line13_checkpoint_global", "")
            or ""
        ),
        "line13_model_route": str(row.get("line_13_model_route", "") or ""),
        "line13_image_size": int(pipeline_summary.get("line13_image_size", 0) or 0),
        "line13_vendor_map_path": str(pipeline_summary.get("line13_vendor_map_path", "") or ""),
        "line13_vendor_map_loaded": (
            pipeline_summary.get("line13_vendor_map_loaded")
            if isinstance(pipeline_summary.get("line13_vendor_map_loaded"), dict)
            else {}
        ),
        "line13_vendor_min_confidence": float(
            pipeline_summary.get("line13_vendor_min_confidence", 0.7) or 0.7
        ),
        "line13_postprocess_enabled": bool(pipeline_summary.get("line13_postprocess_enabled", False)),
        "line13_postprocess_iou_threshold": float(
            pipeline_summary.get("line13_postprocess_iou_threshold", 0.35) or 0.35
        ),
        "line13_postprocess_min_keep": int(
            pipeline_summary.get("line13_postprocess_min_keep", 3) or 3
        ),
        "line13_template_postprocess_enabled": bool(
            pipeline_summary.get("line13_template_postprocess_enabled", False)
        ),
        "line13_template_max_images": int(
            pipeline_summary.get("line13_template_max_images", 3) or 3
        ),
        "line13_template_search_margin_pct": float(
            pipeline_summary.get("line13_template_search_margin_pct", 80.0) or 80.0
        ),
        "line13_template_min_std": float(
            pipeline_summary.get("line13_template_min_std", 60.0) or 60.0
        ),
        "line13_template_min_nonblack_ratio": float(
            pipeline_summary.get("line13_template_min_nonblack_ratio", 0.20) or 0.20
        ),
        "line13_template_min_score": float(
            pipeline_summary.get("line13_template_min_score", 0.60) or 0.60
        ),
        "line13_template_min_consensus_iou": float(
            pipeline_summary.get("line13_template_min_consensus_iou", 0.35) or 0.35
        ),
        "line13_template_min_iou_with_model": float(
            pipeline_summary.get("line13_template_min_iou_with_model", 0.05) or 0.05
        ),
        "line13_postprocess_mode_counts": (
            pipeline_summary.get("line13_postprocess_mode_counts")
            if isinstance(pipeline_summary.get("line13_postprocess_mode_counts"), dict)
            else {}
        ),
        "line13_postprocess_boxes_total": int(
            pipeline_summary.get("line13_postprocess_boxes_total", 0) or 0
        ),
        "line13_postprocess_boxes_kept_total": int(
            pipeline_summary.get("line13_postprocess_boxes_kept_total", 0) or 0
        ),
        "line13_postprocess_boxes_dropped_total": int(
            pipeline_summary.get("line13_postprocess_boxes_dropped_total", 0) or 0
        ),
        "line13_template_postprocess_mode_counts": (
            pipeline_summary.get("line13_template_postprocess_mode_counts")
            if isinstance(pipeline_summary.get("line13_template_postprocess_mode_counts"), dict)
            else {}
        ),
        "line13_template_postprocess_attempted_total": int(
            pipeline_summary.get("line13_template_postprocess_attempted_total", 0) or 0
        ),
        "line13_template_postprocess_applied_total": int(
            pipeline_summary.get("line13_template_postprocess_applied_total", 0) or 0
        ),
        "line13_tail_policy": str(pipeline_summary.get("line13_tail_policy", "") or ""),
    }
    recognition_evidence = {
        "generated_images_enabled": bool(generate_images),
        "split_symlinks_enabled": not bool(args.no_split_symlinks),
        "vendor_probe_samples_rel": vendor_probe_samples_rel,
        "rect_overlay": {
            "source_rel": rect_overlay_source_rel,
            "overlay_rel": rect_overlay_rel,
            "coords": (
                {"top": int(rect_coords[0]), "left": int(rect_coords[1]), "bottom": int(rect_coords[2]), "right": int(rect_coords[3])}
                if rect_coords is not None
                else {}
            ),
            "error": rect_overlay_error,
        },
        "line13_template": {
            "value": str(row.get("line_13_rect_name_echo", "") or ""),
            "pre_dark_trim_value": str(row.get("line_13_pre_dark_trim_rect_name_echo", "") or ""),
            "source": str(row.get("line_13_source", "") or ""),
            "support": str(row.get("line_13_support", "") or ""),
            "model_checkpoint": str(row.get("line_13_model_checkpoint", "") or ""),
            "model_route": str(row.get("line_13_model_route", "") or ""),
            "tail_policy": str(pipeline_summary.get("line13_tail_policy", "") or ""),
            "source_image_rel": line13_source_rel,
            "overlay_rel": line13_overlay_rel,
            "crop_rel": line13_crop_rel,
            "coords": (
                {
                    "top": int(line13_coords[0]),
                    "left": int(line13_coords[1]),
                    "bottom": int(line13_coords[2]),
                    "right": int(line13_coords[3]),
                }
                if line13_coords is not None
                else {}
            ),
            "pre_dark_trim_coords": (
                {
                    "top": int(line13_pre_dark_trim_coords[0]),
                    "left": int(line13_pre_dark_trim_coords[1]),
                    "bottom": int(line13_pre_dark_trim_coords[2]),
                    "right": int(line13_pre_dark_trim_coords[3]),
                }
                if line13_pre_dark_trim_coords is not None
                else {}
            ),
        },
        "line14_template": {
            "value": str(row.get("line_14_rect_name_probe", "") or ""),
            "source": str(row.get("line_14_source", "") or ""),
            "support": str(row.get("line_14_support", "") or ""),
            "overlay_rel": line14_overlay_rel,
        },
        "rect_red": rect_red_evidence,
        "rect_per_image": rect_per_image_evidence,
        "su_giu_per_image": su_giu_per_image_evidence,
        "su_giu_split_folders": su_giu_split_folders,
        "lr_marker_per_image": lr_marker_per_image_evidence,
        "lt_per_image": lt_per_image_evidence,
        "lt_split_folders": lt_split_folders,
        "rect_depth_per_image": rect_depth_per_image_evidence,
        "model_evidence": model_evidence,
    }

    review_html = run_dir / "step_checks_review.html"
    preview_images_rel = [p.relative_to(run_dir).as_posix() for p in preview_images]
    raw_images_rel = [p.relative_to(run_dir).as_posix() for p in raw_gallery]
    _emit_event(
        "evidence_built",
        raw_gallery=len(raw_images_rel),
        duplicate_examples=len(duplicate_examples),
        rotation_sample=bool(rotation_before_rel and rotation_after_rel),
        vendor_probe_samples=len(vendor_probe_samples_rel),
        rect_overlay=bool(rect_overlay_rel),
        rect_red_items=int(len(rect_red_evidence.get("items", []))),
        rect_per_image_items=int(rect_per_image_evidence.get("images_total", 0)),
        su_giu_per_image_items=int(su_giu_per_image_evidence.get("images_total", 0)),
        su_giu_split_items=int(su_giu_split_folders.get("created_total", 0)),
        lr_marker_per_image_items=int(lr_marker_per_image_evidence.get("images_total", 0)),
        lt_per_image_items=int(lt_per_image_evidence.get("images_total", 0)),
        lt_split_items=int(lt_split_folders.get("created_total", 0)),
        rect_depth_per_image_items=int(rect_depth_per_image_evidence.get("images_total", 0)),
    )

    summary = {
        "input_original_folder": input_folder.as_posix(),
        "input_copy_folder": input_ref_folder.as_posix(),
        "input_reference_mode": str(input_reference_meta.get("mode", "reference_symlink")),
        "excluded_images_count": int(input_reference_meta.get("excluded_existing", 0) or 0),
        "excluded_images_rel": list(input_reference_meta.get("excluded_images_rel", [])),
        "excluded_existing_images_rel": list(input_reference_meta.get("excluded_existing_rel", [])),
        "generated_images_enabled": bool(generate_images),
        "split_symlinks_enabled": not bool(args.no_split_symlinks),
        "run_dir": run_dir.as_posix(),
        "pipeline_output_dir": pipeline_output.as_posix(),
        "pipeline_csv": csv_path.as_posix(),
        "predictions_rows": int(predictions_rows),
        "review_html": review_html.as_posix(),
        "preview_images_count": len(preview_images),
        "preview_images_rel": preview_images_rel,
        "raw_images_count": len(acquisition_images),
        "raw_images_rel": raw_images_rel,
        "duplicates_removed_count_recomputed": len(duplicate_pairs),
        "duplicate_examples": duplicate_examples,
        "rotation_evidence": {
            "rotation_deg_clockwise": int(rotation_deg),
            "rotation_source": rotation_source,
            "rotation_vote_ratio": float(rotation_vote_ratio),
            "rotation_votes_total": int(rotation_votes_total),
            "rotation_samples_checked": int(rotation_samples_checked),
            "rotation_decision_reason": rotation_decision_reason,
            "ocr_validation_used": "ocr" in rotation_source.lower(),
            "osd_debug": rotation_osd_debug,
            "ocr_debug": rotation_ocr_debug,
            "sample_original_rel": rotation_sample_original_rel,
            "sample_before_rel": rotation_before_rel,
            "sample_after_rel": rotation_after_rel,
            "error": rotation_error,
        },
        "decision_evidence": decision_evidence,
        "recognition_evidence": recognition_evidence,
        "checks": checks,
        "pipeline_row": row,
    }

    checks_json = run_dir / "step_checks.json"
    checks_txt = run_dir / "step_checks.txt"
    checks_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_lines: List[str] = []
    txt_lines.append(f"Input original: {input_folder.as_posix()}")
    txt_lines.append(f"Input reference(symlink): {input_ref_folder.as_posix()}")
    txt_lines.append("")
    for item in checks:
        txt_lines.append(f"[{item.get('status', 'review')}] {item.get('step', 'step')}")
        for k, v in item.items():
            if k in {"step", "status"}:
                continue
            txt_lines.append(f"  - {k}: {v}")
        txt_lines.append("")
    checks_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    review_html.write_text(
        _build_review_html(
            summary=summary,
            checks=checks,
            run_dir=run_dir,
            preview_images=preview_images,
        ),
        encoding="utf-8",
    )
    _emit_event("review_html_built", path=review_html.as_posix())

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Input reference(symlink): {input_ref_folder}", flush=True)
    print(f"Pipeline CSV: {csv_path}", flush=True)
    print(f"Checks JSON: {checks_json}", flush=True)
    print(f"Checks TXT: {checks_txt}", flush=True)
    print(f"Checks HTML: {review_html}", flush=True)
    _emit_event("run_completed", run_dir=run_dir.as_posix(), checks_html=review_html.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
