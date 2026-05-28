#!/usr/bin/env python3
"""Prepare vendor-template retraining artifacts from HTML review exports.

Workflow supported by this script:
1) Read original manifest + review export (JSON/CSV).
2) Exclude folders marked as "escludere" (or folder flag "exclude_folder").
3) Keep unmarked folders as good data (flags ignored if no review action).
4) Build an interactive HTML tool for folders marked "modificare_rettangolo".
5) Optionally apply exported rectangle corrections and emit final training manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

RECT_ECHO_RE = re.compile(r"^\s*-?\d+\|-?\d+\|-?\d+\|-?\d+\|\s*$")
RECT_NAME_COORDS_PREFIX_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|")
RECT_TARGET_LINE13_ECHO = "line13_echo"
RECT_TARGET_LINE14_PROBE = "line14_probe"


@dataclass(frozen=True)
class FolderDecision:
    group_id: str
    folder_key: str
    resolved_action: str
    review_action_raw: str
    folder_flag_raw: str
    folder_note: str
    review_note: str
    vendor: str
    split: str
    dataset_folder: str
    setup_name: str
    source: str


@dataclass(frozen=True)
class RectFolderTarget:
    group_id: str
    vendor: str
    split: str
    dataset_folder: str
    setup_name: str
    review_note: str
    initial_action: str
    initial_changed: int
    image_width: int
    image_height: int
    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    sample_images: Tuple[str, ...]
    sample_paths: Tuple[str, ...]


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


def _extract_fss_line_with_rect_offset(fss_path: Path, target_line_1based: int) -> Optional[str]:
    try:
        lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    rect_line = _extract_rect_echo_line_num(lines)
    if rect_line is None:
        return None
    offset = rect_line - 11
    return _safe_line(lines, target_line_1based + offset)


def _parse_rect_name_coords(value: str) -> Optional[Tuple[int, int, int, int]]:
    match = RECT_NAME_COORDS_PREFIX_RE.match((value or "").strip())
    if not match:
        return None
    top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
    return top, left, bottom, right


def _normalize_token(value: str) -> str:
    return "_".join((value or "").strip().lower().replace("-", " ").split())


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(str(value).strip())))
    except Exception:  # noqa: BLE001
        return int(default)


def _truthy(value: object) -> bool:
    tok = _normalize_token(str(value))
    return tok in {"1", "true", "yes", "y", "on"}


def _pick_evenly_spaced(values: Sequence[str], keep_n: int) -> List[str]:
    uniq = list(dict.fromkeys(values))
    if keep_n <= 0 or len(uniq) <= keep_n:
        return uniq
    if keep_n == 1:
        return [uniq[len(uniq) // 2]]
    out: List[str] = []
    last = len(uniq) - 1
    for i in range(keep_n):
        idx = int(round((i * last) / float(keep_n - 1)))
        out.append(uniq[idx])
    return out


def _resolve_action(review_action_raw: str, folder_flag_raw: str) -> str:
    action = _normalize_token(review_action_raw)
    flag = _normalize_token(folder_flag_raw)

    if action in {"escludere", "exclude", "exclude_folder"}:
        return "exclude"
    if action in {"modificare_rettangolo", "modifica_rettangolo", "modify_rect", "modify_rectangle"}:
        return "modify_rect"
    if action in {"altro", "other"}:
        return "other"

    # Backward compatibility with quick folder flag.
    if flag in {"exclude", "exclude_folder", "escludere"}:
        return "exclude"

    # Requirement: folders without explicit review action must be treated as good.
    return "keep"


def _read_manifest(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return fieldnames, rows


def _manifest_group_composite(row: Dict[str, str]) -> str:
    vendor = (row.get("manufacturer", "") or "").strip()
    folder = (row.get("dataset_folder", "") or "").strip()
    setup = (row.get("setup_name", "") or "").strip()
    return "::".join([vendor, folder, setup]).strip()


def _load_review_export(review_export: Path) -> Dict[str, FolderDecision]:
    suffix = review_export.suffix.lower()
    if suffix == ".json":
        return _load_review_export_json(review_export)
    if suffix == ".csv":
        return _load_review_export_csv(review_export)
    raise ValueError(f"Unsupported review export format: {review_export}")


def _load_review_export_json(path: Path) -> Dict[str, FolderDecision]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid JSON export: root must be object")

    folders = payload.get("folders", {})
    if not isinstance(folders, dict):
        raise RuntimeError("Invalid JSON export: 'folders' must be an object")

    out: Dict[str, FolderDecision] = {}
    for key, raw in folders.items():
        if not isinstance(raw, dict):
            continue
        group_id = str(raw.get("group_id", "") or "").strip()
        folder_key = str(raw.get("folder_key", "") or key or "").strip()
        if not group_id:
            group_id = folder_key
        if not group_id:
            continue

        review_action_raw = str(raw.get("review_action", "") or "").strip()
        folder_flag_raw = str(raw.get("flag", "") or "").strip()
        decision = FolderDecision(
            group_id=group_id,
            folder_key=folder_key,
            resolved_action=_resolve_action(review_action_raw, folder_flag_raw),
            review_action_raw=review_action_raw,
            folder_flag_raw=folder_flag_raw,
            folder_note=str(raw.get("note", "") or "").strip(),
            review_note=str(raw.get("review_note", "") or "").strip(),
            vendor=str(raw.get("vendor", "") or "").strip(),
            split=str(raw.get("split", "") or "").strip(),
            dataset_folder=str(raw.get("dataset_folder", "") or "").strip(),
            setup_name=str(raw.get("setup_name", "") or "").strip(),
            source="json",
        )
        out[group_id] = decision
    return out


def _load_review_export_csv(path: Path) -> Dict[str, FolderDecision]:
    out: Dict[str, FolderDecision] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower() for f in (reader.fieldnames or [])}
        if "group_id" not in fields:
            raise RuntimeError("CSV export must contain 'group_id' column")

        for row in reader:
            record_type = (row.get("record_type", "") or "").strip().lower()
            if record_type and record_type != "folder":
                continue

            group_id = str(row.get("group_id", "") or "").strip()
            if not group_id:
                continue

            review_action_raw = str(row.get("review_action", "") or "").strip()
            folder_flag_raw = str(row.get("folder_flag", "") or "").strip()
            decision = FolderDecision(
                group_id=group_id,
                folder_key=group_id,
                resolved_action=_resolve_action(review_action_raw, folder_flag_raw),
                review_action_raw=review_action_raw,
                folder_flag_raw=folder_flag_raw,
                folder_note=str(row.get("folder_note", "") or "").strip(),
                review_note=str(row.get("review_note", "") or "").strip(),
                vendor=str(row.get("vendor", "") or "").strip(),
                split=str(row.get("split", "") or "").strip(),
                dataset_folder=str(row.get("dataset_folder", "") or "").strip(),
                setup_name=str(row.get("setup_name", "") or "").strip(),
                source="csv",
            )
            out[group_id] = decision
    return out


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_manifest_group_meta(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        gid = (row.get("group_id", "") or "").strip()
        if not gid:
            continue
        if gid in out:
            continue
        out[gid] = {
            "group_id": gid,
            "manufacturer": (row.get("manufacturer", "") or "").strip(),
            "split": (row.get("split", "") or "").strip(),
            "dataset_folder": (row.get("dataset_folder", "") or "").strip(),
            "setup_name": (row.get("setup_name", "") or "").strip(),
            "folder_key_fallback": _manifest_group_composite(row),
        }
    return out


def _resolve_decision_keys(
    decisions_raw: Dict[str, FolderDecision],
    group_meta: Dict[str, Dict[str, str]],
) -> Dict[str, FolderDecision]:
    composite_to_group: Dict[str, str] = {}
    for gid, meta in group_meta.items():
        composite = (meta.get("folder_key_fallback", "") or "").strip()
        if composite:
            composite_to_group[composite] = gid

    resolved: Dict[str, FolderDecision] = {}
    for _, decision in decisions_raw.items():
        gid = decision.group_id.strip()
        if gid not in group_meta:
            folder_key = decision.folder_key.strip()
            if folder_key in group_meta:
                gid = folder_key
            elif folder_key in composite_to_group:
                gid = composite_to_group[folder_key]

        if gid not in group_meta:
            continue

        resolved[gid] = FolderDecision(
            group_id=gid,
            folder_key=decision.folder_key,
            resolved_action=decision.resolved_action,
            review_action_raw=decision.review_action_raw,
            folder_flag_raw=decision.folder_flag_raw,
            folder_note=decision.folder_note,
            review_note=decision.review_note,
            vendor=decision.vendor or group_meta[gid].get("manufacturer", ""),
            split=decision.split or group_meta[gid].get("split", ""),
            dataset_folder=decision.dataset_folder or group_meta[gid].get("dataset_folder", ""),
            setup_name=decision.setup_name or group_meta[gid].get("setup_name", ""),
            source=decision.source,
        )
    return resolved


def _build_stage1_manifest_rows(
    manifest_rows: Sequence[Dict[str, str]],
    decisions: Dict[str, FolderDecision],
    keep_excluded_rows: bool = False,
) -> Tuple[List[Dict[str, str]], Counter]:
    counters: Counter = Counter()
    out: List[Dict[str, str]] = []
    for row in manifest_rows:
        gid = (row.get("group_id", "") or "").strip()
        decision = decisions.get(gid)
        action = decision.resolved_action if decision else "keep"

        if action == "exclude":
            counters["rows_excluded"] += 1
            if not keep_excluded_rows:
                continue

        next_row = dict(row)
        next_row["manual_folder_action"] = action
        next_row["manual_folder_note"] = decision.folder_note if decision else ""
        next_row["manual_review_note"] = decision.review_note if decision else ""
        next_row["manual_rect_note"] = ""
        next_row["manual_rect_updated"] = "0"
        out.append(next_row)

        counters["rows_kept"] += 1
        if action == "exclude":
            counters["rows_kept_excluded_for_review"] += 1
        if action == "modify_rect":
            counters["rows_marked_modify_rect"] += 1
        elif action == "other":
            counters["rows_marked_other"] += 1
        else:
            counters["rows_keep_default"] += 1
    return out, counters


def _normalize_exclude_tokens(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token not in out:
            out.append(token)
    return out


def _filter_rows_by_image_path_tokens(
    rows: Sequence[Dict[str, str]],
    tokens: Sequence[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Counter]:
    normalized = _normalize_exclude_tokens(tokens)
    if not normalized:
        return list(rows), [], Counter()

    kept: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    counts: Counter = Counter()
    for row in rows:
        image_path = str(row.get("image_path", "") or "").strip().lower()
        matched = [tok for tok in normalized if tok in image_path]
        if not matched:
            kept.append(dict(row))
            continue
        next_row = dict(row)
        next_row["excluded_by_image_name_tokens"] = "|".join(sorted(set(matched)))
        excluded.append(next_row)
        for tok in set(matched):
            counts[tok] += 1
    return kept, excluded, counts


def _build_direct_modify_decisions(
    group_meta: Dict[str, Dict[str, str]],
    source: str = "direct_rect_tool",
) -> Dict[str, FolderDecision]:
    out: Dict[str, FolderDecision] = {}
    for gid, meta in group_meta.items():
        out[gid] = FolderDecision(
            group_id=gid,
            folder_key=gid,
            resolved_action="modify_rect",
            review_action_raw="modify_rect",
            folder_flag_raw="",
            folder_note="",
            review_note="",
            vendor=meta.get("manufacturer", ""),
            split=meta.get("split", ""),
            dataset_folder=meta.get("dataset_folder", ""),
            setup_name=meta.get("setup_name", ""),
            source=source,
        )
    return out


def _apply_line14_probe_bbox_to_stage_rows(
    stage_rows: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    by_group = _group_rows_by_id(stage_rows)
    probe_by_group: Dict[str, Dict[str, object]] = {}
    missing_groups: List[str] = []
    missing_reasons: Dict[str, str] = {}

    for gid, rows in by_group.items():
        if not rows:
            missing_groups.append(gid)
            missing_reasons[gid] = "group_without_rows"
            continue

        first = rows[0]
        fss_path_str = (first.get("fss_path", "") or "").strip()
        if not fss_path_str:
            missing_groups.append(gid)
            missing_reasons[gid] = "missing_fss_path"
            continue

        fss_path = Path(fss_path_str).expanduser()
        if not fss_path.is_absolute():
            fss_path = fss_path.resolve()
        if not fss_path.exists():
            missing_groups.append(gid)
            missing_reasons[gid] = "fss_not_found"
            continue

        line14_raw = _extract_fss_line_with_rect_offset(
            fss_path=fss_path,
            target_line_1based=14,
        )
        if not line14_raw:
            missing_groups.append(gid)
            missing_reasons[gid] = "line14_not_found"
            continue

        coords = _parse_rect_name_coords(line14_raw)
        if coords is None:
            missing_groups.append(gid)
            missing_reasons[gid] = "line14_invalid_rect"
            continue

        top, left, bottom, right = coords
        if bottom <= top:
            bottom = top + 1
        if right <= left:
            right = left + 1
        probe_by_group[gid] = {
            "fss_path": fss_path.as_posix(),
            "line14_raw": line14_raw,
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
        }

    out_rows: List[Dict[str, str]] = []
    rows_updated = 0
    groups_updated: set[str] = set()
    for row in stage_rows:
        gid = (row.get("group_id", "") or "").strip()
        next_row = dict(row)

        probe = probe_by_group.get(gid)
        if not probe:
            next_row["probe_line14_status"] = missing_reasons.get(gid, "probe_line14_missing")
            next_row.setdefault("probe_line14_raw", "")
            next_row.setdefault("probe_line14_fss_path", (row.get("fss_path", "") or "").strip())
            out_rows.append(next_row)
            continue

        top, left, bottom, right, width, height = _clamp_bbox_to_row(
            row,
            top=int(probe["top"]),
            left=int(probe["left"]),
            bottom=int(probe["bottom"]),
            right=int(probe["right"]),
        )
        next_row["bbox_source"] = "fss_line14_probe"
        next_row["bbox_top"] = str(top)
        next_row["bbox_left"] = str(left)
        next_row["bbox_bottom"] = str(bottom)
        next_row["bbox_right"] = str(right)
        next_row["bbox_xmin"] = str(float(left))
        next_row["bbox_ymin"] = str(float(top))
        next_row["bbox_xmax"] = str(float(right))
        next_row["bbox_ymax"] = str(float(bottom))
        next_row["bbox_norm_xmin"] = f"{(left / float(max(1, width))):.8f}"
        next_row["bbox_norm_ymin"] = f"{(top / float(max(1, height))):.8f}"
        next_row["bbox_norm_xmax"] = f"{(right / float(max(1, width))):.8f}"
        next_row["bbox_norm_ymax"] = f"{(bottom / float(max(1, height))):.8f}"
        next_row["bbox_within_image"] = "1"
        next_row["rect_name_probe_top"] = str(top)
        next_row["rect_name_probe_left"] = str(left)
        next_row["rect_name_probe_bottom"] = str(bottom)
        next_row["rect_name_probe_right"] = str(right)
        next_row["probe_line14_raw"] = str(probe["line14_raw"])
        next_row["probe_line14_fss_path"] = str(probe["fss_path"])
        next_row["probe_line14_status"] = "ok"
        if "line_14_top" in next_row:
            next_row["line_14_top"] = str(top)
        if "line_14_left" in next_row:
            next_row["line_14_left"] = str(left)
        if "line_14_bottom" in next_row:
            next_row["line_14_bottom"] = str(bottom)
        if "line_14_right" in next_row:
            next_row["line_14_right"] = str(right)
        if "expected_bbox_top" in next_row:
            next_row["expected_bbox_top"] = str(top)
        if "expected_bbox_left" in next_row:
            next_row["expected_bbox_left"] = str(left)
        if "expected_bbox_bottom" in next_row:
            next_row["expected_bbox_bottom"] = str(bottom)
        if "expected_bbox_right" in next_row:
            next_row["expected_bbox_right"] = str(right)

        rows_updated += 1
        groups_updated.add(gid)
        out_rows.append(next_row)

    summary = {
        "groups_total": len(by_group),
        "groups_with_line14_probe_bbox": len(groups_updated),
        "groups_missing_line14_probe_bbox": missing_groups,
        "groups_missing_line14_probe_bbox_count": len(missing_groups),
        "missing_reason_counts": dict(Counter(missing_reasons.values())),
        "rows_with_line14_probe_bbox": rows_updated,
    }
    return out_rows, summary


def _group_rows_by_id(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        gid = (row.get("group_id", "") or "").strip()
        if not gid:
            continue
        out[gid].append(row)
    return out


def _safe_image_src(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if p.exists():
        # HTTP-friendly absolute path (served by local root HTTP server).
        return quote(p.as_posix(), safe="/")
    return ""


def _load_preview_fallback_map(index_html_path: Path) -> Dict[str, str]:
    text = index_html_path.read_text(encoding="utf-8")
    m = re.search(r"const\s+FOLDER_META\s*=\s*(\{.*?\});", text, flags=re.S)
    if not m:
        raise RuntimeError(f"FOLDER_META not found in index: {index_html_path}")
    raw = m.group(1)
    meta = json.loads(raw)
    if not isinstance(meta, dict):
        raise RuntimeError(f"FOLDER_META invalid JSON object in: {index_html_path}")

    base_dir = index_html_path.parent
    out: Dict[str, str] = {}
    for gid, info in meta.items():
        if not isinstance(info, dict):
            continue
        rel = str(info.get("preview_rel", "") or "").strip()
        if not rel:
            continue
        p = (base_dir / rel).resolve()
        if p.exists():
            out[str(gid)] = p.as_posix()
    return out


def _build_rect_targets(
    stage_rows: Sequence[Dict[str, str]],
    decisions: Dict[str, FolderDecision],
    max_samples: int,
    preview_fallbacks: Optional[Dict[str, str]] = None,
    include_all_groups: bool = False,
) -> List[RectFolderTarget]:
    preview_fallbacks = preview_fallbacks or {}
    by_group = _group_rows_by_id(stage_rows)
    targets: List[RectFolderTarget] = []
    for gid in sorted(by_group.keys()):
        decision = decisions.get(gid)
        action = decision.resolved_action if decision else "keep"
        if not include_all_groups and action != "modify_rect":
            continue

        rows = by_group[gid]
        if not rows:
            continue
        first = rows[0]

        image_paths = [(r.get("image_path", "") or "").strip() for r in rows if (r.get("image_path", "") or "").strip()]
        sampled = _pick_evenly_spaced(image_paths, max_samples)
        sample_paths_effective: List[str] = list(sampled)
        sample_uris = tuple(uri for uri in (_safe_image_src(p) for p in sample_paths_effective) if uri)
        if not sample_uris:
            fallback = (preview_fallbacks.get(gid, "") or "").strip()
            if fallback:
                uri = _safe_image_src(fallback)
                if uri:
                    sample_uris = (uri,)
                    sample_paths_effective = [fallback]
        if not sample_uris:
            continue

        width = _parse_int(first.get("image_width", "0"), 0)
        height = _parse_int(first.get("image_height", "0"), 0)
        if width <= 0 or height <= 0:
            img0 = Path(sampled[0]).expanduser().resolve()
            try:
                from PIL import Image  # imported lazily to keep dependencies local

                with Image.open(img0) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001
                width = max(1, width)
                height = max(1, height)

        bbox_top = _parse_int(first.get("bbox_ymin", first.get("bbox_top", "0")), 0)
        bbox_left = _parse_int(first.get("bbox_xmin", first.get("bbox_left", "0")), 0)
        bbox_bottom = _parse_int(first.get("bbox_ymax", first.get("bbox_bottom", "1")), 1)
        bbox_right = _parse_int(first.get("bbox_xmax", first.get("bbox_right", "1")), 1)
        if bbox_bottom <= bbox_top:
            bbox_bottom = bbox_top + 1
        if bbox_right <= bbox_left:
            bbox_right = bbox_left + 1

        target = RectFolderTarget(
            group_id=gid,
            vendor=(first.get("manufacturer", "") or "").strip(),
            split=(first.get("split", "") or "").strip(),
            dataset_folder=(first.get("dataset_folder", "") or "").strip(),
            setup_name=(first.get("setup_name", "") or "").strip(),
            review_note=decision.review_note if decision else "",
            initial_action="exclude" if action == "exclude" else "keep",
            initial_changed=1 if _truthy(first.get("manual_rect_updated", "0")) else 0,
            image_width=max(1, width),
            image_height=max(1, height),
            bbox_top=bbox_top,
            bbox_left=bbox_left,
            bbox_bottom=bbox_bottom,
            bbox_right=bbox_right,
            sample_images=sample_uris,
            sample_paths=tuple(sample_paths_effective),
        )
        targets.append(target)
    return targets


def _render_rect_editor_html(
    targets: Sequence[RectFolderTarget],
    storage_key_suffix: str,
    page_title: str,
    header_title: str,
    header_subtitle: str,
    storage_key_prefix: str,
    export_filename: str,
) -> str:
    storage_json = json.dumps(storage_key_suffix, ensure_ascii=True)
    embedded_payload = [
        {
            "group_id": t.group_id,
            "vendor": t.vendor,
            "split": t.split,
            "dataset_folder": t.dataset_folder,
            "setup_name": t.setup_name,
            "review_note": t.review_note,
            "initial_action": t.initial_action,
            "initial_changed": t.initial_changed,
            "image_width": t.image_width,
            "image_height": t.image_height,
            "bbox_top": t.bbox_top,
            "bbox_left": t.bbox_left,
            "bbox_bottom": t.bbox_bottom,
            "bbox_right": t.bbox_right,
            "sample_images": list(t.sample_images),
            "sample_paths": list(t.sample_paths),
        }
        for t in targets
    ]
    embedded_json = json.dumps(embedded_payload, ensure_ascii=True)
    storage_key_prefix_json = json.dumps(storage_key_prefix, ensure_ascii=True)
    export_filename_json = json.dumps(export_filename, ensure_ascii=True)

    html = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #0f1218;
      --panel: #182030;
      --panel2: #111825;
      --border: #31415d;
      --txt: #e7edf7;
      --muted: #a7b4ca;
      --accent: #5fa8ff;
      --danger: #f05a5a;
      --ok: #4ecb7a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--txt); }
    .wrap { max-width: 1700px; margin: 0 auto; padding: 14px; }
    h1 { margin: 0 0 6px 0; }
    .muted { color: var(--muted); font-size: 12px; }
    .toolbar {
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .nav {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 4px 8px;
      border: 1px solid #3a4f72;
      border-radius: 6px;
      background: #132035;
    }
    .kbd {
      font-family: monospace;
      font-size: 11px;
      padding: 1px 5px;
      border: 1px solid #526a90;
      border-radius: 4px;
      color: #d7e7ff;
      background: #0d1727;
    }
    button {
      border: 1px solid #4a5f82;
      border-radius: 6px;
      background: #22364f;
      color: #eef4ff;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 700;
    }
    button:hover { background: #2d4664; }
    button.warn { background: #5a2a2a; border-color: #784040; }
    button.warn:hover { background: #713737; }
    .counts { color: #d5e5ff; font-weight: 700; }
    .status { color: #9ec4f5; }
    .cards { margin-top: 12px; display: grid; grid-template-columns: 1fr; gap: 14px; }
    .card {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel2);
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(520px, 1.8fr) minmax(320px, 1fr);
      gap: 10px;
      padding: 10px;
    }
    .card.changed { border-color: var(--ok); box-shadow: 0 0 0 2px rgba(78,203,122,0.35) inset; }
    .card.excluded { border-color: #f3b26d; box-shadow: 0 0 0 2px rgba(243,178,109,0.35) inset; }
    .card.flagged { border-color: #8c6bff; box-shadow: 0 0 0 2px rgba(140,107,255,0.35) inset; }
    .viewer { position: relative; background: #000; border: 1px solid #2d3b53; border-radius: 6px; overflow: hidden; user-select: none; touch-action: none; cursor: crosshair; }
    .viewer.drawing { cursor: crosshair; }
    .viewer img { width: 100%; height: auto; display: block; pointer-events: none; user-select: none; -webkit-user-drag: none; }
    .bbox, .draft {
      position: absolute;
      border: 3px solid;
      pointer-events: none;
      box-sizing: border-box;
    }
    .bbox { border-color: var(--danger); box-shadow: 0 0 0 1px rgba(240,90,90,0.35) inset; }
    .draft { border-color: #ffd166; display: none; }
    .thumbs { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
    .thumb {
      border: 1px solid #42516c;
      border-radius: 5px;
      background: #172233;
      color: #d6e4fa;
      padding: 5px 7px;
      cursor: pointer;
      font-size: 11px;
      max-width: 340px;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .thumb.active { border-color: #8dbbff; background: #243852; }
    .meta { font-size: 12px; line-height: 1.35; display: flex; flex-direction: column; gap: 8px; }
    .meta .line { color: #d6e4fa; }
    .meta .line strong { color: #fff; }
    .nums { display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 8px; }
    .nums label, .note label { display: flex; flex-direction: column; gap: 4px; color: #d6e4fa; }
    .nums input, .note textarea, .note select {
      border: 1px solid #4a5f82;
      border-radius: 5px;
      background: #0f1520;
      color: #e8edf6;
      padding: 6px;
      font-size: 12px;
    }
    .toolbar select {
      border: 1px solid #4a5f82;
      border-radius: 5px;
      background: #0f1520;
      color: #e8edf6;
      padding: 5px 8px;
      font-size: 12px;
    }
    .action-line .action-tag {
      display: inline-block;
      margin-left: 6px;
      padding: 2px 6px;
      border-radius: 5px;
      border: 1px solid #4a5f82;
      background: #1a2a3f;
      color: #dce9ff;
      font-weight: 700;
    }
    .note textarea { min-height: 70px; resize: vertical; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .hint { color: #ffd9a6; font-size: 12px; }
    @media (max-width: 1100px) {
      .card { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>__HEADER_TITLE__</h1>
    <div class="muted">__HEADER_SUBTITLE__</div>
    <div class="toolbar">
      <button id="btn-export-csv" type="button">Export correzioni CSV</button>
      <button id="btn-reset-all" class="warn" type="button">Reset locale</button>
      <button id="btn-flag-current" type="button">Flag preview</button>
      <div class="nav">
        <button id="btn-prev" type="button" title="Previous (ArrowLeft)">◀ Prev</button>
        <button id="btn-next" type="button" title="Next (ArrowRight)">Next ▶</button>
        <span id="nav-status"></span>
      </div>
      <label class="muted">Filtro
        <select id="filter-mode">
          <option value="all">Tutte</option>
          <option value="flagged">Flaggate</option>
          <option value="exclude">Escluse</option>
          <option value="modified">Modificate</option>
        </select>
      </label>
      <span class="muted">Shortcuts: <span class="kbd">←</span>/<span class="kbd">→</span> nav, <span class="kbd">X</span> escludi, <span class="kbd">F</span> flag, <span class="kbd">A</span>/<span class="kbd">G</span>/<span class="kbd">E</span>/<span class="kbd">M</span> filtri, <span class="kbd">R</span> reset</span>
      <span id="counts" class="counts"></span>
      <span id="save-status" class="status"></span>
    </div>
    <div id="cards" class="cards"></div>
  </div>
  <script>
    let FOLDERS = [];
    const DATA_FILE = "folders_payload.json";
    const EMBEDDED_FOLDERS = __EMBEDDED_FOLDERS__;
    const STORAGE_KEY_SUFFIX = __STORAGE_SUFFIX__;
    const STORAGE_KEY = __STORAGE_PREFIX__ + "::" + STORAGE_KEY_SUFFIX;
    const cssEscape = (window.CSS && typeof window.CSS.escape === "function")
      ? window.CSS.escape
      : function (value) {
          return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
        };
    let state = null;
    let saveTimer = null;

    function normalizeFilterMode(value) {
      const tok = String(value == null ? "" : value).trim().toLowerCase();
      if (tok === "flagged" || tok === "exclude" || tok === "modified") return tok;
      return "all";
    }

    function defaultState() {
      return { version: 1, updated_at: "", current_idx: 0, filter_mode: "all", items: {} };
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
        if (!parsed.items || typeof parsed.items !== "object") parsed.items = {};
        if (!Number.isFinite(Number(parsed.current_idx))) parsed.current_idx = 0;
        parsed.filter_mode = normalizeFilterMode(parsed.filter_mode);
        return parsed;
      } catch (_err) {
        return defaultState();
      }
    }

    function setStatus(text) {
      const el = document.getElementById("save-status");
      if (el) el.textContent = text || "";
    }

    function writeState(immediate) {
      const persist = () => {
        state.updated_at = new Date().toISOString();
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
          setStatus("autosave: " + new Date().toLocaleTimeString());
        } catch (_err) {
          setStatus("errore salvataggio localStorage");
        }
        updateCounters();
      };
      if (immediate) {
        if (saveTimer) clearTimeout(saveTimer);
        persist();
        return;
      }
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(persist, 200);
    }

    function folderByGroup(groupId) {
      return FOLDERS.find((f) => String(f.group_id || "") === String(groupId || "")) || null;
    }

    async function loadFolders() {
      const root = document.getElementById("cards");
      if (Array.isArray(EMBEDDED_FOLDERS) && EMBEDDED_FOLDERS.length > 0) {
        FOLDERS = EMBEDDED_FOLDERS;
        renderAll();
        return;
      }
      try {
        const response = await fetch(DATA_FILE, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!Array.isArray(data)) {
          throw new Error("payload non valido (array atteso)");
        }
        FOLDERS = data;
        renderAll();
      } catch (err) {
        FOLDERS = [];
        if (root) {
          root.innerHTML = '<div class="muted">Errore caricamento cartelle da correggere. Verifica che <code>folders_payload.json</code> sia presente.</div>';
        }
        setStatus("errore caricamento cartelle");
      }
    }

    function clampRect(folder, rect) {
      const w = Math.max(1, Number(folder.image_width || 1));
      const h = Math.max(1, Number(folder.image_height || 1));
      let top = Math.max(0, Math.min(h - 1, Number(rect.top || 0)));
      let left = Math.max(0, Math.min(w - 1, Number(rect.left || 0)));
      let bottom = Math.max(1, Math.min(h, Number(rect.bottom || 1)));
      let right = Math.max(1, Math.min(w, Number(rect.right || 1)));
      if (bottom <= top) bottom = Math.min(h, top + 1);
      if (right <= left) right = Math.min(w, left + 1);
      return { top, left, bottom, right };
    }

    function toStyle(folder, rect) {
      const w = Math.max(1, Number(folder.image_width || 1));
      const h = Math.max(1, Number(folder.image_height || 1));
      const bw = Math.max(1, Number(rect.right) - Number(rect.left));
      const bh = Math.max(1, Number(rect.bottom) - Number(rect.top));
      return [
        `left:${(100.0 * Number(rect.left) / w).toFixed(6)}%`,
        `top:${(100.0 * Number(rect.top) / h).toFixed(6)}%`,
        `width:${(100.0 * bw / w).toFixed(6)}%`,
        `height:${(100.0 * bh / h).toFixed(6)}%`,
      ].join(";");
    }

    function normalizeAction(value) {
      const tok = String(value == null ? "" : value).trim().toLowerCase();
      if (tok === "exclude" || tok === "exclude_folder" || tok === "escludere") return "exclude";
      return "keep";
    }

    function filteredIndices() {
      const mode = normalizeFilterMode(state.filter_mode);
      const out = [];
      FOLDERS.forEach((folder, idx) => {
        const item = ensureItem(folder);
        const include = (
          mode === "all"
          || (mode === "flagged" && !!item.flagged)
          || (mode === "exclude" && normalizeAction(item.action) === "exclude")
          || (mode === "modified" && !!item.changed)
        );
        if (include) out.push(idx);
      });
      return out;
    }

    function currentIndex() {
      const n = Array.isArray(FOLDERS) ? FOLDERS.length : 0;
      if (n <= 0) return -1;
      const shown = filteredIndices();
      if (shown.length <= 0) return -1;
      let idx = Math.max(0, Math.min(n - 1, Number(state.current_idx || 0)));
      if (!shown.includes(idx)) idx = shown[0];
      state.current_idx = idx;
      return idx;
    }

    function currentFolder() {
      if (!Array.isArray(FOLDERS) || FOLDERS.length === 0) return null;
      const idx = currentIndex();
      if (idx < 0) return null;
      return FOLDERS[idx] || null;
    }

    function setCurrentIndex(nextIdx, forceRender) {
      if (!Array.isArray(FOLDERS) || FOLDERS.length === 0) return;
      const n = FOLDERS.length;
      const clamped = Math.max(0, Math.min(n - 1, Number(nextIdx || 0)));
      const changed = clamped !== Number(state.current_idx || 0);
      state.current_idx = clamped;
      if (changed || forceRender) renderAll();
      writeState(false);
    }

    function setFilterMode(mode, forceRender) {
      const next = normalizeFilterMode(mode);
      const changed = next !== normalizeFilterMode(state.filter_mode);
      state.filter_mode = next;
      if (changed || forceRender) renderAll();
      writeState(false);
    }

    function ensureItem(folder) {
      const gid = String(folder.group_id || "");
      if (!state.items[gid]) {
        state.items[gid] = {
          group_id: gid,
          top: Number(folder.bbox_top || 0),
          left: Number(folder.bbox_left || 0),
          bottom: Number(folder.bbox_bottom || 1),
          right: Number(folder.bbox_right || 1),
          note: "",
          action: normalizeAction(folder.initial_action || "keep"),
          flagged: 0,
          changed: Number(folder.initial_changed || 0) ? 1 : 0,
          active_sample_idx: 0,
        };
      }
      state.items[gid] = {
        ...state.items[gid],
        group_id: gid,
      };
      const item = state.items[gid];
      const clamped = clampRect(folder, item);
      item.top = clamped.top;
      item.left = clamped.left;
      item.bottom = clamped.bottom;
      item.right = clamped.right;
      if (typeof item.note !== "string") item.note = "";
      item.action = normalizeAction(item.action);
      if (typeof item.flagged !== "number") item.flagged = Number(item.flagged) ? 1 : 0;
      if (typeof item.changed !== "number") item.changed = Number(item.changed) ? 1 : 0;
      if (!Number.isFinite(Number(item.active_sample_idx))) item.active_sample_idx = 0;
      item.active_sample_idx = Math.max(0, Math.min((folder.sample_images || []).length - 1, Number(item.active_sample_idx || 0)));
      return item;
    }

    function escapeHtml(value) {
      return String(value == null ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function cardHtml(folder, item) {
      const gid = escapeHtml(folder.group_id || "");
      const vendor = escapeHtml(folder.vendor || "");
      const split = escapeHtml(folder.split || "");
      const setup = escapeHtml(folder.setup_name || "");
      const datasetFolder = escapeHtml(folder.dataset_folder || "");
      const reviewNote = escapeHtml(folder.review_note || "");
      const action = normalizeAction(item.action);
      const actionKeepSel = action === "keep" ? " selected" : "";
      const actionExcludeSel = action === "exclude" ? " selected" : "";
      const actionLabel = action === "exclude" ? "ESCLUDI CARTELLA" : "usa cartella";
      const flagged = !!item.flagged;
      const flaggedLabel = flagged ? "FLAGGATA" : "non flaggata";
      const flagBtnLabel = flagged ? "Rimuovi flag preview" : "Flagga preview";
      const idx = Math.max(0, Math.min((folder.sample_images || []).length - 1, Number(item.active_sample_idx || 0)));
      const activeImage = (folder.sample_images || [""])[idx] || "";
      const activePath = (folder.sample_paths || [""])[idx] || "";
      const thumbs = (folder.sample_images || []).map((uri, i) => {
        const label = escapeHtml((folder.sample_paths || [])[i] || `sample_${i + 1}`);
        const active = i === idx ? " active" : "";
        return `<button class="thumb${active}" data-kind="thumb" data-group-id="${gid}" data-sample-idx="${i}" title="${label}" type="button">${label}</button>`;
      }).join("");

      return (
        `<article class="card${item.changed ? " changed" : ""}${action === "exclude" ? " excluded" : ""}${flagged ? " flagged" : ""}" id="card-${gid}" data-group-id="${gid}">` +
        '<div>' +
        `<div class="viewer" data-kind="viewer" data-group-id="${gid}">` +
        `<img src="${escapeHtml(activeImage)}" alt="preview ${gid}" draggable="false" data-kind="main-image" data-group-id="${gid}" data-image-path="${escapeHtml(activePath)}">` +
        `<div class="bbox" data-kind="bbox" data-group-id="${gid}" style="${toStyle(folder, item)}"></div>` +
        `<div class="draft" data-kind="draft" data-group-id="${gid}"></div>` +
        '</div>' +
        `<div class="hint">Trascina per impostare il rettangolo della cartella. Risoluzione riferimento: ${folder.image_width}x${folder.image_height}</div>` +
        `<div class="thumbs">${thumbs}</div>` +
        '</div>' +
        '<div class="meta">' +
        `<div class="line"><strong>${vendor}</strong> | split=${split} | group=${gid}</div>` +
        `<div class="line">setup=${setup} | folder=${datasetFolder}</div>` +
        `<div class="line">Nota review: ${reviewNote || "-"}</div>` +
        `<div class="line action-line">Azione training: <span class="action-tag" data-kind="action-tag" data-group-id="${gid}">${actionLabel}</span></div>` +
        `<div class="line action-line">Flag preview: <span class="action-tag" data-kind="flag-label" data-group-id="${gid}">${flaggedLabel}</span></div>` +
        '<div class="nums">' +
        `<label>top<input data-kind="num" data-field="top" data-group-id="${gid}" type="number" min="0" step="1" value="${item.top}"></label>` +
        `<label>left<input data-kind="num" data-field="left" data-group-id="${gid}" type="number" min="0" step="1" value="${item.left}"></label>` +
        `<label>bottom<input data-kind="num" data-field="bottom" data-group-id="${gid}" type="number" min="1" step="1" value="${item.bottom}"></label>` +
        `<label>right<input data-kind="num" data-field="right" data-group-id="${gid}" type="number" min="1" step="1" value="${item.right}"></label>` +
        '</div>' +
        '<div class="note">' +
        `<label>Nota correzione<textarea data-kind="note" data-group-id="${gid}" placeholder="opzionale">${escapeHtml(item.note || "")}</textarea></label>` +
        `<label>Azione training<select data-kind="action" data-group-id="${gid}"><option value="keep"${actionKeepSel}>Usa cartella</option><option value="exclude"${actionExcludeSel}>Escludi cartella</option></select></label>` +
        '</div>' +
        '<div class="actions">' +
        `<button data-kind="flag" data-group-id="${gid}" type="button">${flagBtnLabel}</button>` +
        `<button data-kind="reset" data-group-id="${gid}" type="button">Reset box originale</button>` +
        '</div>' +
        '</div>' +
        '</article>'
      );
    }

    function updateOverlay(groupId) {
      const folder = folderByGroup(groupId);
      if (!folder) return;
      const item = ensureItem(folder);
      const box = document.querySelector(`[data-kind="bbox"][data-group-id="${cssEscape(groupId)}"]`);
      if (box) box.setAttribute("style", toStyle(folder, item));
      const card = document.getElementById(`card-${groupId}`);
      if (card) {
        card.classList.toggle("changed", !!item.changed);
        card.classList.toggle("excluded", normalizeAction(item.action) === "exclude");
        card.classList.toggle("flagged", !!item.flagged);
      }
      const actionTag = document.querySelector(`[data-kind="action-tag"][data-group-id="${cssEscape(groupId)}"]`);
      if (actionTag) {
        actionTag.textContent = normalizeAction(item.action) === "exclude" ? "ESCLUDI CARTELLA" : "usa cartella";
      }
      const actionSel = document.querySelector(`[data-kind="action"][data-group-id="${cssEscape(groupId)}"]`);
      if (actionSel) actionSel.value = normalizeAction(item.action);
      const flagLabel = document.querySelector(`[data-kind="flag-label"][data-group-id="${cssEscape(groupId)}"]`);
      if (flagLabel) flagLabel.textContent = item.flagged ? "FLAGGATA" : "non flaggata";
      const flagBtn = document.querySelector(`[data-kind="flag"][data-group-id="${cssEscape(groupId)}"]`);
      if (flagBtn) flagBtn.textContent = item.flagged ? "Rimuovi flag preview" : "Flagga preview";
    }

    function setItemFromRect(groupId, rect, markChanged) {
      const folder = folderByGroup(groupId);
      if (!folder) return;
      const item = ensureItem(folder);
      const next = clampRect(folder, rect);
      item.top = next.top;
      item.left = next.left;
      item.bottom = next.bottom;
      item.right = next.right;
      if (markChanged) item.changed = 1;
      ["top", "left", "bottom", "right"].forEach((field) => {
        const el = document.querySelector(`[data-kind="num"][data-field="${field}"][data-group-id="${cssEscape(groupId)}"]`);
        if (el) el.value = String(item[field]);
      });
      updateOverlay(groupId);
      writeState(false);
    }

    function activateSample(groupId, sampleIdx) {
      const folder = folderByGroup(groupId);
      if (!folder) return;
      const item = ensureItem(folder);
      const maxIdx = Math.max(0, (folder.sample_images || []).length - 1);
      item.active_sample_idx = Math.max(0, Math.min(maxIdx, Number(sampleIdx || 0)));
      const card = document.getElementById(`card-${groupId}`);
      if (card) {
        const imgEl = card.querySelector(`[data-kind="main-image"][data-group-id="${cssEscape(groupId)}"]`);
        if (imgEl) {
          imgEl.src = (folder.sample_images || [""])[item.active_sample_idx] || "";
          imgEl.setAttribute("data-image-path", (folder.sample_paths || [""])[item.active_sample_idx] || "");
        }
        card.querySelectorAll(`[data-kind="thumb"][data-group-id="${cssEscape(groupId)}"]`).forEach((el, idx) => {
          el.classList.toggle("active", idx === item.active_sample_idx);
        });
      }
      writeState(false);
    }

    function updateCounters() {
      const total = FOLDERS.length;
      let changed = 0;
      let excluded = 0;
      let flagged = 0;
      FOLDERS.forEach((folder) => {
        const item = ensureItem(folder);
        if (item.changed) changed += 1;
        if (normalizeAction(item.action) === "exclude") excluded += 1;
        if (item.flagged) flagged += 1;
      });
      const shown = filteredIndices();
      const mode = normalizeFilterMode(state.filter_mode);
      const el = document.getElementById("counts");
      if (el) {
        el.textContent = `totali: ${total} | modificate: ${changed} | escluse: ${excluded} | flaggate: ${flagged} | visibili: ${shown.length}`;
      }
      const nav = document.getElementById("nav-status");
      if (nav) {
        const idx = currentIndex();
        if (shown.length <= 0 || idx < 0) {
          nav.textContent = `nessuna cartella (${mode})`;
        } else {
          const pos = Math.max(0, shown.indexOf(idx));
          nav.textContent = `${mode} ${pos + 1}/${shown.length} (globale ${idx + 1}/${total})`;
        }
      }
      const filterSel = document.getElementById("filter-mode");
      if (filterSel) {
        const expected = normalizeFilterMode(state.filter_mode);
        if (filterSel.value !== expected) filterSel.value = expected;
      }
      const flagCurrentBtn = document.getElementById("btn-flag-current");
      if (flagCurrentBtn) {
        const folder = currentFolder();
        if (!folder) {
          flagCurrentBtn.disabled = true;
          flagCurrentBtn.textContent = "Flag preview";
        } else {
          const item = ensureItem(folder);
          flagCurrentBtn.disabled = false;
          flagCurrentBtn.textContent = item.flagged ? "Rimuovi flag preview" : "Flag preview";
        }
      }
    }

    function ensureCurrentVisibleForFilter() {
      const idx = currentIndex();
      const shown = filteredIndices();
      if (shown.length <= 0) {
        renderAll();
        return true;
      }
      if (idx < 0 || !shown.includes(idx)) {
        renderAll();
        return true;
      }
      return false;
    }

    function bindCardDraw(card) {
      const gid = card.getAttribute("data-group-id") || "";
      const folder = folderByGroup(gid);
      if (!folder) return;
      const viewer = card.querySelector(`[data-kind="viewer"][data-group-id="${cssEscape(gid)}"]`);
      const img = card.querySelector(`[data-kind="main-image"][data-group-id="${cssEscape(gid)}"]`);
      const draft = card.querySelector(`[data-kind="draft"][data-group-id="${cssEscape(gid)}"]`);
      if (!viewer || !img || !draft) return;

      let active = false;
      let startNorm = null;
      let lastNorm = null;
      let activePointerId = null;

      function normFromEvent(ev) {
        const rect = img.getBoundingClientRect();
        if (!rect || rect.width <= 1 || rect.height <= 1) return null;
        const nx = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
        const ny = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
        return { x: nx, y: ny };
      }

      function rectFromNorm(a, b) {
        const x1 = Math.min(a.x, b.x);
        const y1 = Math.min(a.y, b.y);
        const x2 = Math.max(a.x, b.x);
        const y2 = Math.max(a.y, b.y);
        const w = Math.max(1, Number(folder.image_width || 1));
        const h = Math.max(1, Number(folder.image_height || 1));
        const left = Math.round(x1 * w);
        const top = Math.round(y1 * h);
        const right = Math.max(left + 1, Math.round(x2 * w));
        const bottom = Math.max(top + 1, Math.round(y2 * h));
        return clampRect(folder, { top, left, bottom, right });
      }

      function drawDraft(a, b) {
        const x1 = Math.min(a.x, b.x);
        const y1 = Math.min(a.y, b.y);
        const x2 = Math.max(a.x, b.x);
        const y2 = Math.max(a.y, b.y);
        const ww = Math.max(0.001, x2 - x1) * 100.0;
        const hh = Math.max(0.001, y2 - y1) * 100.0;
        draft.style.display = "block";
        draft.style.left = `${(x1 * 100).toFixed(6)}%`;
        draft.style.top = `${(y1 * 100).toFixed(6)}%`;
        draft.style.width = `${ww.toFixed(6)}%`;
        draft.style.height = `${hh.toFixed(6)}%`;
      }

      function hideDraft() {
        draft.style.display = "none";
      }

      function clearActive() {
        active = false;
        startNorm = null;
        lastNorm = null;
        activePointerId = null;
        viewer.classList.remove("drawing");
        hideDraft();
      }

      function teardownListeners() {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onCancel);
      }

      function onMove(ev) {
        if (!active || !startNorm) return;
        if (activePointerId != null && ev.pointerId != null && ev.pointerId !== activePointerId) return;
        ev.preventDefault();
        const next = normFromEvent(ev);
        if (!next) return;
        lastNorm = next;
        drawDraft(startNorm, next);
      }

      function onUp(ev) {
        if (!active || !startNorm) return;
        if (activePointerId != null && ev.pointerId != null && ev.pointerId !== activePointerId) return;
        ev.preventDefault();
        const end = normFromEvent(ev) || lastNorm || startNorm;
        const rect = rectFromNorm(startNorm, end);
        const boxW = rect.right - rect.left;
        const boxH = rect.bottom - rect.top;
        clearActive();
        teardownListeners();
        if (boxW < 2 || boxH < 2) {
          setStatus("rettangolo troppo piccolo: trascina di piu");
          return;
        }
        setItemFromRect(gid, rect, true);
      }

      function onCancel(ev) {
        if (!active) return;
        if (activePointerId != null && ev && ev.pointerId != null && ev.pointerId !== activePointerId) return;
        clearActive();
        teardownListeners();
        setStatus("disegno annullato");
      }

      viewer.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        if (!img.complete || Number(img.naturalWidth || 0) <= 0) {
          setStatus("attendi caricamento immagine");
          return;
        }
        const start = normFromEvent(ev);
        if (!start) return;
        ev.preventDefault();
        active = true;
        startNorm = start;
        lastNorm = start;
        activePointerId = (ev.pointerId != null ? ev.pointerId : null);
        if (activePointerId != null && typeof viewer.setPointerCapture === "function") {
          try {
            viewer.setPointerCapture(activePointerId);
          } catch (_err) {
          }
        }
        viewer.classList.add("drawing");
        drawDraft(startNorm, startNorm);
        window.addEventListener("pointermove", onMove, { passive: false });
        window.addEventListener("pointerup", onUp, { passive: false });
        window.addEventListener("pointercancel", onCancel, { passive: false });
      });

      viewer.addEventListener("lostpointercapture", (ev) => {
        if (!active) return;
        if (activePointerId != null && ev.pointerId != null && ev.pointerId !== activePointerId) return;
        onUp(ev);
      });

      viewer.addEventListener("dragstart", (ev) => {
        ev.preventDefault();
      });
      img.addEventListener("dragstart", (ev) => {
        ev.preventDefault();
      });
    }

    function csvEscape(value) {
      const s = String(value == null ? "" : value);
      if (s.includes(",") || s.includes('"') || s.includes("\\n")) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    }

    function downloadText(filename, text, contentType) {
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
    }

    function exportCsv() {
      if (!Array.isArray(FOLDERS) || FOLDERS.length === 0) {
        setStatus("nessuna cartella caricata");
        return;
      }
      writeState(true);
      const header = [
        "group_id",
        "vendor",
        "split",
        "dataset_folder",
        "setup_name",
        "image_width",
        "image_height",
        "orig_top",
        "orig_left",
        "orig_bottom",
        "orig_right",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "changed",
        "flagged",
        "action",
        "note"
      ];
      const rows = [header];
      FOLDERS.forEach((folder) => {
        const item = ensureItem(folder);
        rows.push([
          folder.group_id,
          folder.vendor || "",
          folder.split || "",
          folder.dataset_folder || "",
          folder.setup_name || "",
          String(folder.image_width || ""),
          String(folder.image_height || ""),
          String(folder.bbox_top || ""),
          String(folder.bbox_left || ""),
          String(folder.bbox_bottom || ""),
          String(folder.bbox_right || ""),
          String(item.top),
          String(item.left),
          String(item.bottom),
          String(item.right),
          item.changed ? "1" : "0",
          item.flagged ? "1" : "0",
          normalizeAction(item.action),
          item.note || "",
        ]);
      });
      const payload = rows.map((r) => r.map(csvEscape).join(",")).join("\\n");
      downloadText(__EXPORT_FILENAME__, payload, "text/csv;charset=utf-8");
    }

    function resetAll() {
      if (!confirm("Confermi reset completo delle correzioni locali?")) return;
      state = defaultState();
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (_err) {
      }
      renderAll();
      setStatus("reset locale completato");
    }

    function bindCardInputs(card) {
      const gid = card.getAttribute("data-group-id") || "";
      const folder = folderByGroup(gid);
      if (!folder) return;

      card.querySelectorAll('[data-kind="num"]').forEach((el) => {
        el.addEventListener("input", () => {
          const item = ensureItem(folder);
          const field = el.getAttribute("data-field") || "";
          item[field] = Number(el.value || 0);
          const next = clampRect(folder, item);
          item.top = next.top;
          item.left = next.left;
          item.bottom = next.bottom;
          item.right = next.right;
          item.changed = 1;
          ["top", "left", "bottom", "right"].forEach((f) => {
            const input = card.querySelector(`[data-kind="num"][data-field="${f}"][data-group-id="${cssEscape(gid)}"]`);
            if (input) input.value = String(item[f]);
          });
          updateOverlay(gid);
          if (ensureCurrentVisibleForFilter()) return;
          writeState(false);
        });
      });

      const noteEl = card.querySelector('[data-kind="note"]');
      if (noteEl) {
        noteEl.addEventListener("input", () => {
          const item = ensureItem(folder);
          item.note = noteEl.value || "";
          writeState(false);
        });
      }

      const actionEl = card.querySelector('[data-kind="action"]');
      if (actionEl) {
        actionEl.addEventListener("change", () => {
          const item = ensureItem(folder);
          item.action = normalizeAction(actionEl.value);
          updateOverlay(gid);
          if (ensureCurrentVisibleForFilter()) return;
          writeState(false);
        });
      }

      const flagEl = card.querySelector('[data-kind="flag"]');
      if (flagEl) {
        flagEl.addEventListener("click", () => {
          const item = ensureItem(folder);
          item.flagged = item.flagged ? 0 : 1;
          updateOverlay(gid);
          if (ensureCurrentVisibleForFilter()) return;
          writeState(false);
        });
      }

      card.querySelectorAll('[data-kind="thumb"]').forEach((el) => {
        el.addEventListener("click", () => {
          const idx = Number(el.getAttribute("data-sample-idx") || "0");
          activateSample(gid, idx);
        });
      });

      const resetBtn = card.querySelector('[data-kind="reset"]');
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          const item = ensureItem(folder);
          item.top = Number(folder.bbox_top || 0);
          item.left = Number(folder.bbox_left || 0);
          item.bottom = Number(folder.bbox_bottom || 1);
          item.right = Number(folder.bbox_right || 1);
          item.note = "";
          item.action = normalizeAction(folder.initial_action || "keep");
          item.changed = 0;
          item.active_sample_idx = 0;
          card.querySelectorAll('[data-kind="num"]').forEach((input) => {
            const field = input.getAttribute("data-field") || "";
            input.value = String(item[field]);
          });
          const note = card.querySelector('[data-kind="note"]');
          if (note) note.value = "";
          const action = card.querySelector('[data-kind="action"]');
          if (action) action.value = normalizeAction(folder.initial_action || "keep");
          activateSample(gid, 0);
          updateOverlay(gid);
          if (ensureCurrentVisibleForFilter()) return;
          writeState(true);
        });
      }

      bindCardDraw(card);
    }

    function goPrev() {
      const shown = filteredIndices();
      if (shown.length <= 0) return;
      const cur = currentIndex();
      const pos = Math.max(0, shown.indexOf(cur));
      const nextPos = Math.max(0, pos - 1);
      setCurrentIndex(shown[nextPos], false);
    }

    function goNext() {
      const shown = filteredIndices();
      if (shown.length <= 0) return;
      const cur = currentIndex();
      const pos = Math.max(0, shown.indexOf(cur));
      const nextPos = Math.min(shown.length - 1, pos + 1);
      setCurrentIndex(shown[nextPos], false);
    }

    function toggleExcludeCurrent() {
      const folder = currentFolder();
      if (!folder) return;
      const item = ensureItem(folder);
      item.action = normalizeAction(item.action) === "exclude" ? "keep" : "exclude";
      updateOverlay(String(folder.group_id || ""));
      if (ensureCurrentVisibleForFilter()) return;
      writeState(false);
    }

    function toggleFlagCurrent() {
      const folder = currentFolder();
      if (!folder) return;
      const item = ensureItem(folder);
      item.flagged = item.flagged ? 0 : 1;
      updateOverlay(String(folder.group_id || ""));
      if (ensureCurrentVisibleForFilter()) return;
      writeState(false);
    }

    function resetCurrent() {
      const folder = currentFolder();
      if (!folder) return;
      const gid = String(folder.group_id || "");
      const card = document.getElementById(`card-${gid}`);
      if (!card) return;
      const btn = card.querySelector('[data-kind="reset"]');
      if (btn) btn.click();
    }

    function isTypingTarget(target) {
      if (!target || typeof target !== "object") return false;
      const tag = String(target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return true;
      return !!target.isContentEditable;
    }

    function renderAll() {
      const root = document.getElementById("cards");
      if (!root) return;
      if (!Array.isArray(FOLDERS) || FOLDERS.length === 0) {
        root.innerHTML = '<div class="muted">Nessuna cartella da correggere.</div>';
        updateCounters();
        return;
      }
      const shown = filteredIndices();
      if (shown.length <= 0) {
        root.innerHTML = '<div class="muted">Nessuna cartella nel filtro corrente. Passa a "Tutte" per vederle.</div>';
        updateCounters();
        writeState(true);
        return;
      }
      const folder = currentFolder();
      if (!folder) {
        root.innerHTML = '<div class="muted">Nessuna cartella da correggere.</div>';
        updateCounters();
        return;
      }
      const item = ensureItem(folder);
      root.innerHTML = cardHtml(folder, item);
      const card = root.querySelector(".card");
      if (card) bindCardInputs(card);
      updateOverlay(String(folder.group_id || ""));
      updateCounters();
      writeState(true);
    }

    const exportBtn = document.getElementById("btn-export-csv");
    if (exportBtn) exportBtn.addEventListener("click", exportCsv);
    const resetBtn = document.getElementById("btn-reset-all");
    if (resetBtn) resetBtn.addEventListener("click", resetAll);
    const prevBtn = document.getElementById("btn-prev");
    if (prevBtn) prevBtn.addEventListener("click", goPrev);
    const nextBtn = document.getElementById("btn-next");
    if (nextBtn) nextBtn.addEventListener("click", goNext);
    const flagCurrentBtn = document.getElementById("btn-flag-current");
    if (flagCurrentBtn) flagCurrentBtn.addEventListener("click", toggleFlagCurrent);
    const filterModeSel = document.getElementById("filter-mode");
    if (filterModeSel) {
      filterModeSel.addEventListener("change", () => {
        setFilterMode(filterModeSel.value, false);
      });
    }

    document.addEventListener("keydown", (ev) => {
      if (isTypingTarget(ev.target)) return;
      if (ev.key === "ArrowLeft" || ev.key === "PageUp") {
        ev.preventDefault();
        goPrev();
        return;
      }
      if (ev.key === "ArrowRight" || ev.key === "PageDown") {
        ev.preventDefault();
        goNext();
        return;
      }
      if (ev.key === "x" || ev.key === "X") {
        ev.preventDefault();
        toggleExcludeCurrent();
        return;
      }
      if (ev.key === "f" || ev.key === "F") {
        ev.preventDefault();
        toggleFlagCurrent();
        return;
      }
      if (ev.key === "a" || ev.key === "A") {
        ev.preventDefault();
        setFilterMode("all", false);
        return;
      }
      if (ev.key === "g" || ev.key === "G") {
        ev.preventDefault();
        setFilterMode("flagged", false);
        return;
      }
      if (ev.key === "e" || ev.key === "E") {
        ev.preventDefault();
        setFilterMode("exclude", false);
        return;
      }
      if (ev.key === "m" || ev.key === "M") {
        ev.preventDefault();
        setFilterMode("modified", false);
        return;
      }
      if (ev.key === "r" || ev.key === "R") {
        ev.preventDefault();
        resetCurrent();
      }
    });

    state = readState();
    loadFolders();
  </script>
</body>
</html>
"""
    html = html.replace("__EMBEDDED_FOLDERS__", embedded_json)
    html = html.replace("__STORAGE_SUFFIX__", storage_json)
    html = html.replace("__STORAGE_PREFIX__", storage_key_prefix_json)
    html = html.replace("__EXPORT_FILENAME__", export_filename_json)
    html = html.replace("__PAGE_TITLE__", page_title)
    html = html.replace("__HEADER_TITLE__", header_title)
    html = html.replace("__HEADER_SUBTITLE__", header_subtitle)
    return html


def _write_rect_tool(
    output_dir: Path,
    targets: Sequence[RectFolderTarget],
    storage_suffix: str,
    *,
    page_title: str,
    header_title: str,
    header_subtitle: str,
    storage_key_prefix: str,
    export_filename: str,
) -> Tuple[Path, Path]:
    tool_dir = output_dir / "rect_correction_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "group_id": t.group_id,
            "vendor": t.vendor,
            "split": t.split,
            "dataset_folder": t.dataset_folder,
            "setup_name": t.setup_name,
            "review_note": t.review_note,
            "initial_action": t.initial_action,
            "initial_changed": t.initial_changed,
            "image_width": t.image_width,
            "image_height": t.image_height,
            "bbox_top": t.bbox_top,
            "bbox_left": t.bbox_left,
            "bbox_bottom": t.bbox_bottom,
            "bbox_right": t.bbox_right,
            "sample_images": list(t.sample_images),
            "sample_paths": list(t.sample_paths),
        }
        for t in targets
    ]
    payload_json_path = tool_dir / "folders_payload.json"
    payload_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = tool_dir / "folders_for_rect_review.csv"
    rows = [
        {
            "group_id": t.group_id,
            "vendor": t.vendor,
            "split": t.split,
            "dataset_folder": t.dataset_folder,
            "setup_name": t.setup_name,
            "review_note": t.review_note,
            "initial_action": t.initial_action,
            "initial_changed": t.initial_changed,
            "image_width": t.image_width,
            "image_height": t.image_height,
            "bbox_top": t.bbox_top,
            "bbox_left": t.bbox_left,
            "bbox_bottom": t.bbox_bottom,
            "bbox_right": t.bbox_right,
            "sample_images": " | ".join(t.sample_paths),
        }
        for t in targets
    ]
    _write_csv(
        csv_path,
        rows,
        fieldnames=[
            "group_id",
            "vendor",
            "split",
            "dataset_folder",
            "setup_name",
            "review_note",
            "initial_action",
            "initial_changed",
            "image_width",
            "image_height",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "sample_images",
        ],
    )

    html_path = tool_dir / "index.html"
    html_path.write_text(
        _render_rect_editor_html(
            targets=targets,
            storage_key_suffix=storage_suffix,
            page_title=page_title,
            header_title=header_title,
            header_subtitle=header_subtitle,
            storage_key_prefix=storage_key_prefix,
            export_filename=export_filename,
        ),
        encoding="utf-8",
    )
    # Make absolute paths like /Volumes/... reachable when serving from tool_dir.
    for root_name in ("Volumes", "Users"):
        src = Path("/") / root_name
        dst = tool_dir / root_name
        try:
            if src.exists() and src.is_dir() and not dst.exists() and not dst.is_symlink():
                dst.symlink_to(src)
        except OSError:
            # Non-fatal: tool still works if caller serves from filesystem root.
            pass
    return tool_dir, html_path


def _load_rect_corrections_csv(path: Path) -> Dict[str, Dict[str, object]]:
    """Load corrections from a single CSV.

    Supported formats:
    - Custom tool export: group_id + bbox_top/left/bottom/right (+ changed)
    - Ultrasound rect review export: group_name + corr_top/left/bottom/right
    """
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower() for f in (reader.fieldnames or [])}
        has_group_id = "group_id" in fields
        has_group_name = "group_name" in fields
        if not has_group_id and not has_group_name:
            raise RuntimeError(
                f"{path.name}: missing group key column (expected 'group_id' or 'group_name')"
            )

        for row in reader:
            gid = str(row.get("group_id", "") or "").strip()
            if not gid:
                gid = str(row.get("group_name", "") or "").strip()
            if not gid:
                continue

            note = str(row.get("note", "") or "").strip()
            action_raw = str(
                row.get(
                    "action",
                    row.get("folder_action", row.get("manual_folder_action", "")),
                )
                or ""
            ).strip()
            action_resolved = _resolve_action(action_raw, "")
            if action_resolved == "exclude":
                out[gid] = {
                    "action": "exclude",
                    "note": note,
                }
                continue

            changed_raw = row.get("changed", "")
            has_changed_field = "changed" in (reader.fieldnames or [])
            if has_changed_field and not _truthy(changed_raw):
                continue

            flagged_raw = row.get("flagged", "")
            has_flagged_field = "flagged" in (reader.fieldnames or [])
            if has_flagged_field and (not has_changed_field) and not _truthy(flagged_raw):
                continue

            top_raw = row.get("bbox_top", row.get("corr_top", row.get("target_top", "")))
            left_raw = row.get("bbox_left", row.get("corr_left", row.get("target_left", "")))
            bottom_raw = row.get("bbox_bottom", row.get("corr_bottom", row.get("target_bottom", "")))
            right_raw = row.get("bbox_right", row.get("corr_right", row.get("target_right", "")))

            if str(top_raw).strip() == "" or str(left_raw).strip() == "" or str(bottom_raw).strip() == "" or str(right_raw).strip() == "":
                continue

            top = _parse_int(top_raw, 0)
            left = _parse_int(left_raw, 0)
            bottom = _parse_int(bottom_raw, 1)
            right = _parse_int(right_raw, 1)
            if bottom <= top:
                bottom = top + 1
            if right <= left:
                right = left + 1

            out[gid] = {
                "bbox_top": top,
                "bbox_left": left,
                "bbox_bottom": bottom,
                "bbox_right": right,
                "action": "modify_rect",
                "note": note,
            }
    return out


def _load_rect_corrections_dir(
    corrections_dir: Path,
    pattern: str = "corrections_*.csv",
) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """Load corrections from a directory (recursive), last file wins per group."""
    if not corrections_dir.is_dir():
        raise FileNotFoundError(f"Rect corrections dir not found: {corrections_dir}")

    files = sorted(corrections_dir.rglob(pattern))
    if not files:
        raise RuntimeError(
            f"No correction files found in {corrections_dir} with pattern '{pattern}'"
        )

    merged: Dict[str, Dict[str, object]] = {}
    loaded_files: List[str] = []
    for path in files:
        rows = _load_rect_corrections_csv(path)
        if not rows:
            continue
        merged.update(rows)
        loaded_files.append(path.as_posix())
    return merged, loaded_files


def _clamp_bbox_to_row(
    row: Dict[str, str],
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> Tuple[int, int, int, int, int, int]:
    width = _parse_int(row.get("image_width", "0"), 0)
    height = _parse_int(row.get("image_height", "0"), 0)

    if width <= 0 or height <= 0:
        image_path = (row.get("image_path", "") or "").strip()
        if image_path:
            try:
                from PIL import Image  # imported lazily

                with Image.open(Path(image_path).expanduser().resolve()) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001
                width = max(1, width)
                height = max(1, height)
        else:
            width = max(1, width)
            height = max(1, height)

    left = max(0, min(width - 1, int(left)))
    top = max(0, min(height - 1, int(top)))
    right = max(1, min(width, int(right)))
    bottom = max(1, min(height, int(bottom)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)

    return top, left, bottom, right, width, height


def _apply_rect_corrections(
    stage_rows: Sequence[Dict[str, str]],
    decisions: Dict[str, FolderDecision],
    corrections: Dict[str, Dict[str, object]],
    allow_missing_corrections: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    def _corr_action(corr: Dict[str, object]) -> str:
        return _resolve_action(str(corr.get("action", "") or ""), "")

    modify_groups = sorted(
        gid for gid, d in decisions.items() if d.resolved_action == "modify_rect"
    )
    corrected_groups = sorted(gid for gid in corrections.keys())
    missing_groups = [gid for gid in modify_groups if gid not in corrections]
    unknown_corrections = [gid for gid in corrected_groups if gid not in set(modify_groups)]

    if missing_groups and not allow_missing_corrections:
        raise RuntimeError(
            "Missing rectangle corrections for modify groups: "
            + ", ".join(missing_groups[:20])
            + (" ..." if len(missing_groups) > 20 else "")
        )

    out_rows: List[Dict[str, str]] = []
    rows_corrected = 0
    rows_excluded = 0
    groups_touched: set[str] = set()
    groups_excluded: set[str] = set()

    for row in stage_rows:
        gid = (row.get("group_id", "") or "").strip()
        next_row = dict(row)

        corr = corrections.get(gid)
        if corr and _corr_action(corr) == "exclude":
            rows_excluded += 1
            groups_excluded.add(gid)
            continue

        has_bbox = bool(
            corr
            and all(
                key in corr
                for key in ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right")
            )
        )
        if corr and has_bbox:
            top, left, bottom, right, width, height = _clamp_bbox_to_row(
                row,
                top=_parse_int(corr.get("bbox_top", 0), 0),
                left=_parse_int(corr.get("bbox_left", 0), 0),
                bottom=_parse_int(corr.get("bbox_bottom", 1), 1),
                right=_parse_int(corr.get("bbox_right", 1), 1),
            )
            next_row["bbox_source"] = "manual_review_rect"
            next_row["bbox_top"] = str(top)
            next_row["bbox_left"] = str(left)
            next_row["bbox_bottom"] = str(bottom)
            next_row["bbox_right"] = str(right)
            next_row["bbox_xmin"] = str(float(left))
            next_row["bbox_ymin"] = str(float(top))
            next_row["bbox_xmax"] = str(float(right))
            next_row["bbox_ymax"] = str(float(bottom))
            next_row["bbox_norm_xmin"] = f"{(left / float(max(1, width))):.8f}"
            next_row["bbox_norm_ymin"] = f"{(top / float(max(1, height))):.8f}"
            next_row["bbox_norm_xmax"] = f"{(right / float(max(1, width))):.8f}"
            next_row["bbox_norm_ymax"] = f"{(bottom / float(max(1, height))):.8f}"
            next_row["bbox_within_image"] = "1"

            # Keep historical columns aligned with effective training target.
            if "rect_name_echo_top" in next_row:
                next_row["rect_name_echo_top"] = str(top)
            if "rect_name_echo_left" in next_row:
                next_row["rect_name_echo_left"] = str(left)
            if "rect_name_echo_bottom" in next_row:
                next_row["rect_name_echo_bottom"] = str(bottom)
            if "rect_name_echo_right" in next_row:
                next_row["rect_name_echo_right"] = str(right)
            if "rect_name_probe_top" in next_row:
                next_row["rect_name_probe_top"] = str(top)
            if "rect_name_probe_left" in next_row:
                next_row["rect_name_probe_left"] = str(left)
            if "rect_name_probe_bottom" in next_row:
                next_row["rect_name_probe_bottom"] = str(bottom)
            if "rect_name_probe_right" in next_row:
                next_row["rect_name_probe_right"] = str(right)
            if "line_14_top" in next_row:
                next_row["line_14_top"] = str(top)
            if "line_14_left" in next_row:
                next_row["line_14_left"] = str(left)
            if "line_14_bottom" in next_row:
                next_row["line_14_bottom"] = str(bottom)
            if "line_14_right" in next_row:
                next_row["line_14_right"] = str(right)
            if "expected_bbox_top" in next_row:
                next_row["expected_bbox_top"] = str(top)
            if "expected_bbox_left" in next_row:
                next_row["expected_bbox_left"] = str(left)
            if "expected_bbox_bottom" in next_row:
                next_row["expected_bbox_bottom"] = str(bottom)
            if "expected_bbox_right" in next_row:
                next_row["expected_bbox_right"] = str(right)

            next_row["manual_rect_note"] = str(corr.get("note", "") or "")
            next_row["manual_rect_updated"] = "1"
            rows_corrected += 1
            groups_touched.add(gid)

        out_rows.append(next_row)

    summary = {
        "modify_groups_total": len(modify_groups),
        "modify_groups_with_corrections": len(groups_touched),
        "modify_groups_missing_corrections": missing_groups,
        "corrections_groups_unknown_to_modify": unknown_corrections,
        "rows_corrected": rows_corrected,
        "groups_excluded_by_correction": sorted(groups_excluded),
        "rows_excluded_by_correction": rows_excluded,
    }
    return out_rows, summary


def _write_folder_actions_csv(
    path: Path,
    group_meta: Dict[str, Dict[str, str]],
    decisions: Dict[str, FolderDecision],
    rows_by_group: Dict[str, List[Dict[str, str]]],
) -> None:
    rows: List[Dict[str, object]] = []
    for gid in sorted(group_meta.keys()):
        meta = group_meta[gid]
        decision = decisions.get(gid)
        rows.append(
            {
                "group_id": gid,
                "resolved_action": decision.resolved_action if decision else "keep",
                "review_action_raw": decision.review_action_raw if decision else "",
                "folder_flag_raw": decision.folder_flag_raw if decision else "",
                "folder_note": decision.folder_note if decision else "",
                "review_note": decision.review_note if decision else "",
                "found_in_review_export": 1 if decision else 0,
                "manufacturer": meta.get("manufacturer", ""),
                "split": meta.get("split", ""),
                "dataset_folder": meta.get("dataset_folder", ""),
                "setup_name": meta.get("setup_name", ""),
                "rows_in_manifest": len(rows_by_group.get(gid, [])),
            }
        )

    _write_csv(
        path,
        rows,
        fieldnames=[
            "group_id",
            "resolved_action",
            "review_action_raw",
            "folder_flag_raw",
            "folder_note",
            "review_note",
            "found_in_review_export",
            "manufacturer",
            "split",
            "dataset_folder",
            "setup_name",
            "rows_in_manifest",
        ],
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare vendor-template retraining manifest from review export. "
            "Supports folder exclusion and manual rectangle correction workflow, "
            "including direct no-flag editing mode."
        )
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv"),
        help="Input manifest CSV.",
    )
    p.add_argument(
        "--review-export",
        type=Path,
        default=None,
        help=(
            "Export file from review HTML (vendor_template_flags.json or .csv). "
            "Optional when using --rect-tool-all-folders."
        ),
    )
    p.add_argument(
        "--rect-tool-all-folders",
        action="store_true",
        help=(
            "Skip flag workflow and force direct rectangle editing on all manifest folders "
            "(manual_folder_action=modify_rect for every group)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_template_retraining_from_review"),
        help="Output directory for derived manifests and rect tool.",
    )
    p.add_argument(
        "--rect-corrections-csv",
        type=Path,
        default=None,
        help=(
            "Optional corrections CSV exported by rect_correction_tool. "
            "If provided, script produces final training manifest with corrected boxes."
        ),
    )
    p.add_argument(
        "--rect-corrections-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with corrections_*.csv exported from per-folder review pages. "
            "All matched files are merged (last file wins per group)."
        ),
    )
    p.add_argument(
        "--rect-corrections-pattern",
        type=str,
        default="corrections_*.csv",
        help="Glob pattern for --rect-corrections-dir (default: corrections_*.csv).",
    )
    p.add_argument(
        "--allow-missing-corrections",
        action="store_true",
        help="Allow missing corrections for groups marked modify_rect (keeps original bbox for missing groups).",
    )
    p.add_argument(
        "--rect-tool-max-samples",
        type=int,
        default=1,
        help="Number of sample images shown per folder in rect correction tool.",
    )
    p.add_argument(
        "--rect-tool-show-all-groups",
        action="store_true",
        help=(
            "Show all groups in rect tool (not only modify_rect). "
            "Useful to browse/filter all folders while keeping current actions."
        ),
    )
    p.add_argument(
        "--keep-excluded-rows-in-stage",
        action="store_true",
        help=(
            "Keep rows marked exclude inside stage manifest/tool payload "
            "(for review/filtering visibility). They can still be removed later in final training manifest."
        ),
    )
    p.add_argument(
        "--prefill-rect-tool-from-corrections",
        action="store_true",
        help=(
            "If corrections CSV/dir is provided, pre-apply bbox corrections to the tool payload "
            "so the UI opens already aligned to saved corrections."
        ),
    )
    p.add_argument(
        "--fallback-preview-index",
        type=Path,
        default=None,
        help=(
            "Optional index.html from prior review package; if original image_path is unavailable, "
            "uses local folder_previews mapped by group_id."
        ),
    )
    p.add_argument(
        "--rect-target",
        type=str,
        choices=[RECT_TARGET_LINE13_ECHO, RECT_TARGET_LINE14_PROBE],
        default=RECT_TARGET_LINE13_ECHO,
        help=(
            "Which rectangle target to edit/train. "
            "'line13_echo' keeps current bbox columns (vendor template). "
            "'line14_probe' replaces bbox from FSS line 14 RECT_NAME_PROBE."
        ),
    )
    p.add_argument(
        "--exclude-image-substring",
        action="append",
        default=None,
        help=(
            "Case-insensitive substring used to exclude rows by image_path. "
            "Repeatable. Default if omitted: negative, proibite."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    review_export: Optional[Path] = None
    if args.review_export:
        review_export = args.review_export.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if review_export and not review_export.exists():
        raise FileNotFoundError(f"Missing review export: {review_export}")
    if not review_export and not bool(args.rect_tool_all_folders):
        raise RuntimeError(
            "Missing review export. Pass --review-export or enable --rect-tool-all-folders."
        )

    manifest_fields, manifest_rows = _read_manifest(manifest_path)
    group_meta = _collect_manifest_group_meta(manifest_rows)
    rows_by_group = _group_rows_by_id(manifest_rows)

    if bool(args.rect_tool_all_folders):
        decisions = _build_direct_modify_decisions(group_meta)
    else:
        if review_export is None:
            raise RuntimeError("Internal error: review export missing")
        decisions_raw = _load_review_export(review_export)
        decisions = _resolve_decision_keys(decisions_raw, group_meta)

    rect_target = str(args.rect_target or RECT_TARGET_LINE13_ECHO).strip()
    exclude_tokens = _normalize_exclude_tokens(
        args.exclude_image_substring if args.exclude_image_substring else ["negative", "proibite"]
    )
    keep_excluded_rows = bool(args.keep_excluded_rows_in_stage or args.rect_tool_show_all_groups)
    stage_rows, stage_counters = _build_stage1_manifest_rows(
        manifest_rows,
        decisions,
        keep_excluded_rows=keep_excluded_rows,
    )
    stage_rows, excluded_image_rows, excluded_image_token_counts = _filter_rows_by_image_path_tokens(
        stage_rows,
        exclude_tokens,
    )
    stage_counters["rows_excluded_by_image_name_filter"] = len(excluded_image_rows)
    stage_counters["rows_after_image_name_filter"] = len(stage_rows)
    excluded_images_report_csv: Optional[Path] = None
    if excluded_image_rows:
        excluded_images_report_csv = output_dir / "excluded_images_by_name_filter.csv"
        excluded_fields: List[str] = []
        for row in excluded_image_rows:
            for key in row.keys():
                if key not in excluded_fields:
                    excluded_fields.append(key)
        _write_csv(excluded_images_report_csv, excluded_image_rows, excluded_fields)

    probe_line14_summary: Dict[str, object] = {}
    if rect_target == RECT_TARGET_LINE14_PROBE:
        stage_rows, probe_line14_summary = _apply_line14_probe_bbox_to_stage_rows(stage_rows)

    stage_fields = list(manifest_fields)
    for extra in (
        "manual_folder_action",
        "manual_folder_note",
        "manual_review_note",
        "manual_rect_note",
        "manual_rect_updated",
    ):
        if extra not in stage_fields:
            stage_fields.append(extra)
    if rect_target == RECT_TARGET_LINE14_PROBE:
        for extra in (
            "bbox_source",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "bbox_norm_xmin",
            "bbox_norm_ymin",
            "bbox_norm_xmax",
            "bbox_norm_ymax",
            "bbox_within_image",
            "rect_name_probe_top",
            "rect_name_probe_left",
            "rect_name_probe_bottom",
            "rect_name_probe_right",
            "probe_line14_raw",
            "probe_line14_fss_path",
            "probe_line14_status",
        ):
            if extra not in stage_fields:
                stage_fields.append(extra)

    manifest_stage1 = output_dir / "manifest_after_exclusions.csv"
    _write_csv(manifest_stage1, stage_rows, stage_fields)

    folder_actions_csv = output_dir / "folder_actions_resolved.csv"
    _write_folder_actions_csv(folder_actions_csv, group_meta, decisions, rows_by_group)

    preview_fallbacks: Dict[str, str] = {}
    fallback_index_path: Optional[Path] = None
    if args.fallback_preview_index:
        fallback_index_path = args.fallback_preview_index.expanduser().resolve()
        if not fallback_index_path.exists():
            raise FileNotFoundError(f"Missing fallback preview index: {fallback_index_path}")
        preview_fallbacks = _load_preview_fallback_map(fallback_index_path)

    final_manifest: Optional[Path] = None
    correction_summary: Dict[str, object] = {
        "modify_groups_total": sum(1 for d in decisions.values() if d.resolved_action == "modify_rect"),
        "modify_groups_with_corrections": 0,
        "modify_groups_missing_corrections": [],
        "corrections_groups_unknown_to_modify": [],
        "rows_corrected": 0,
    }
    corrections_csv_path: Optional[Path] = None
    corrections_dir_path: Optional[Path] = None
    corrections_loaded_files: List[str] = []
    corrections: Dict[str, Dict[str, object]] = {}

    if args.rect_corrections_dir:
        corrections_dir_path = args.rect_corrections_dir.expanduser().resolve()
        dir_corrections, loaded_files = _load_rect_corrections_dir(
            corrections_dir_path,
            pattern=str(args.rect_corrections_pattern),
        )
        corrections.update(dir_corrections)
        corrections_loaded_files.extend(loaded_files)

    if args.rect_corrections_csv:
        corrections_csv_path = args.rect_corrections_csv.expanduser().resolve()
        if not corrections_csv_path.exists():
            raise FileNotFoundError(f"Missing rect corrections CSV: {corrections_csv_path}")
        one_file_corrections = _load_rect_corrections_csv(corrections_csv_path)
        corrections.update(one_file_corrections)
        corrections_loaded_files.append(corrections_csv_path.as_posix())

    stage_rows_for_tool = stage_rows
    if corrections and bool(args.prefill_rect_tool_from_corrections):
        prefill_rows, _prefill_summary = _apply_rect_corrections(
            stage_rows=stage_rows,
            decisions=decisions,
            corrections=corrections,
            allow_missing_corrections=True,
        )
        stage_rows_for_tool = prefill_rows

    rect_targets = _build_rect_targets(
        stage_rows=stage_rows_for_tool,
        decisions=decisions,
        max_samples=max(1, int(args.rect_tool_max_samples)),
        preview_fallbacks=preview_fallbacks,
        include_all_groups=bool(args.rect_tool_show_all_groups),
    )

    rect_tool_dir: Optional[Path] = None
    rect_tool_index: Optional[Path] = None
    if rect_targets:
        if rect_target == RECT_TARGET_LINE14_PROBE:
            page_title = "Probe line14 - Modifica Rettangoli"
            header_title = "Tool Modifica Rettangolo Probe (#14)"
            header_subtitle = (
                "Trascina direttamente sull'immagine per impostare il box probe "
                "della cartella (linea 14 FSS). Autosave locale attivo."
            )
            storage_key_prefix = "probe_line14_rect_editor"
            export_filename = "probe_rect_line14_corrections.csv"
        else:
            page_title = "Vendor Template - Modifica Rettangoli"
            header_title = "Tool Modifica Rettangolo (diretto)"
            header_subtitle = (
                "Trascina direttamente sull'immagine per impostare il box cartella. "
                "Autosave locale attivo."
            )
            storage_key_prefix = "vendor_template_rect_editor"
            export_filename = "vendor_template_rect_corrections.csv"

        storage_review_key = review_export.as_posix() if review_export else "__direct_all_folders__"
        rect_tool_dir, rect_tool_index = _write_rect_tool(
            output_dir=output_dir,
            targets=rect_targets,
            storage_suffix=f"{manifest_path.as_posix()}::{storage_review_key}",
            page_title=page_title,
            header_title=header_title,
            header_subtitle=header_subtitle,
            storage_key_prefix=storage_key_prefix,
            export_filename=export_filename,
        )

    if corrections:
        allow_missing = bool(args.allow_missing_corrections or args.rect_tool_all_folders)
        final_rows, correction_summary = _apply_rect_corrections(
            stage_rows=stage_rows,
            decisions=decisions,
            corrections=corrections,
            allow_missing_corrections=allow_missing,
        )
        final_manifest = output_dir / "manifest_training_ready.csv"
        _write_csv(final_manifest, final_rows, stage_fields)

    action_counter = Counter(
        (decision.resolved_action if decision else "keep")
        for decision in (decisions.get(gid) for gid in group_meta.keys())
    )
    summary = {
        "manifest_input": manifest_path.as_posix(),
        "review_export": review_export.as_posix() if review_export else "",
        "rect_target": rect_target,
        "rect_tool_all_folders": bool(args.rect_tool_all_folders),
        "rect_tool_show_all_groups": bool(args.rect_tool_show_all_groups),
        "keep_excluded_rows_in_stage": bool(keep_excluded_rows),
        "prefill_rect_tool_from_corrections": bool(args.prefill_rect_tool_from_corrections),
        "output_dir": output_dir.as_posix(),
        "exclude_image_substrings": exclude_tokens,
        "rows_input": len(manifest_rows),
        "groups_input": len(group_meta),
        "decisions_loaded": len(decisions),
        "action_counts_on_manifest_groups": dict(sorted(action_counter.items())),
        "stage1_manifest": manifest_stage1.as_posix(),
        "folder_actions_csv": folder_actions_csv.as_posix(),
        "stage1_counters": dict(stage_counters),
        "rect_tool": {
            "folders_marked_modify_rect": len(rect_targets),
            "tool_dir": rect_tool_dir.as_posix() if rect_tool_dir else "",
            "index_html": rect_tool_index.as_posix() if rect_tool_index else "",
        },
        "fallback_preview_index": fallback_index_path.as_posix() if fallback_index_path else "",
        "fallback_preview_groups_available": len(preview_fallbacks),
        "excluded_images_report_csv": excluded_images_report_csv.as_posix() if excluded_images_report_csv else "",
        "excluded_images_token_counts": dict(sorted(excluded_image_token_counts.items())),
        "excluded_images_rows_count": len(excluded_image_rows),
        "probe_line14_summary": probe_line14_summary,
        "rect_corrections_csv": corrections_csv_path.as_posix() if corrections_csv_path else "",
        "rect_corrections_dir": corrections_dir_path.as_posix() if corrections_dir_path else "",
        "rect_corrections_loaded_files": corrections_loaded_files,
        "final_manifest": final_manifest.as_posix() if final_manifest else "",
        "correction_summary": correction_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Manifest input: {manifest_path}", flush=True)
    if review_export:
        print(f"Review export: {review_export}", flush=True)
    else:
        print("Review export: not used (direct all-folders mode)", flush=True)
    print(
        "Image-name exclusion tokens: "
        + (", ".join(exclude_tokens) if exclude_tokens else "(none)"),
        flush=True,
    )
    print(
        f"Rows excluded by image-name filter: {len(excluded_image_rows)}",
        flush=True,
    )
    if excluded_images_report_csv:
        print(f"Excluded images report: {excluded_images_report_csv}", flush=True)
    print(f"Stage-1 manifest (excluded folders removed): {manifest_stage1}", flush=True)
    print(f"Folder actions: {folder_actions_csv}", flush=True)
    if rect_tool_index:
        print(f"Rect correction tool: {rect_tool_index}", flush=True)
    else:
        print("Rect correction tool: no folders marked modify_rect", flush=True)
    if final_manifest:
        print(f"Final manifest (training ready): {final_manifest}", flush=True)
    else:
        print(
            "Final manifest: not generated "
            "(pass --rect-corrections-csv or --rect-corrections-dir after completing rect tool)",
            flush=True,
        )
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
