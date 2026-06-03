#!/usr/bin/env python3
"""Unified HTML workbench for single-folder pipeline operations + review + analysis."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
import zipfile
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, url_for
try:
    from openpyxl import load_workbook
except Exception:
    load_workbook = None
try:
    from rect_red_pipeline import compute_rect_red_pipeline
except ModuleNotFoundError:
    from tools.ultrasound.rect_red_pipeline import compute_rect_red_pipeline
try:
    from run_pipeline_single_folder_safe import (
        _analyze_duplicates as _runner_analyze_duplicates,
        _build_lr_marker_per_image_evidence as _runner_build_lr_marker_per_image_evidence,
        _build_rect_red_evidence as _runner_build_rect_red_evidence,
        _build_rect_per_image_evidence as _runner_build_rect_per_image_evidence,
        _build_su_giu_per_image_evidence as _runner_build_su_giu_per_image_evidence,
        _build_su_giu_split_folders as _runner_build_su_giu_split_folders,
        _build_lt_per_image_evidence as _runner_build_lt_per_image_evidence,
        _build_lt_split_folders as _runner_build_lt_split_folders,
        _collect_acquisition_images as _runner_collect_acquisition_images,
        _parse_rect_coords as _runner_parse_rect_coords,
        _save_rect_crop_preview as _runner_save_rect_crop_preview,
    )
except ModuleNotFoundError:
    from tools.ultrasound.run_pipeline_single_folder_safe import (
        _analyze_duplicates as _runner_analyze_duplicates,
        _build_lr_marker_per_image_evidence as _runner_build_lr_marker_per_image_evidence,
        _build_rect_red_evidence as _runner_build_rect_red_evidence,
        _build_rect_per_image_evidence as _runner_build_rect_per_image_evidence,
        _build_su_giu_per_image_evidence as _runner_build_su_giu_per_image_evidence,
        _build_su_giu_split_folders as _runner_build_su_giu_split_folders,
        _build_lt_per_image_evidence as _runner_build_lt_per_image_evidence,
        _build_lt_split_folders as _runner_build_lt_split_folders,
        _collect_acquisition_images as _runner_collect_acquisition_images,
        _parse_rect_coords as _runner_parse_rect_coords,
        _save_rect_crop_preview as _runner_save_rect_crop_preview,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_BROWSE_ROOT = (REPO_ROOT / "data").resolve()
DEFAULT_FSS_TEMPLATE_CANDIDATES = [
    REPO_ROOT / "OldSoftwareEsiBuilder/templates/DB_setup/setup_71.fss",
    REPO_ROOT / "OldSoftwareEsiBuilder/templates/DB_setup/setup_70.fss",
]
DEFAULT_MODELS_METRICS_CSV_CANDIDATES = [
    REPO_ROOT / "artifacts/60_metadata/reti_metriche_status_2026-04-01.csv",
]
DEFAULT_ENCODING_STRUCT_XLSX_CANDIDATES = [
    REPO_ROOT / "encoding_struct 2026 02 23.xlsx",
]
DEFAULT_VENDOR_MAPPING_CSV_CANDIDATES = [
    REPO_ROOT / "artifacts/60_metadata/vendor_mapping_final_from_encoding_struct_2026-03-11.csv",
]
DEFAULT_MODEL_MERGE_MAPPING_CSV_CANDIDATES = [
    REPO_ROOT / "artifacts/60_metadata/review_model_merge/model_merge_mapping_applied_20260410_110113.csv",
]
DEFAULT_PROBE_MERGE_MAPPING_CSV_CANDIDATES = [
    REPO_ROOT / "artifacts/60_metadata/review_probe_merge/probe_merge_mapping_applied_20260410_114231.csv",
]
DEFAULT_LR_MARKER_TEMPLATE_ROOTS = [
    REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/lr_marker_vendor_template_library",
    Path("/Volumes/SSD_esi1_n1"),
]
DEFAULT_BULK_ACQUISITION_ROOT = Path("/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION")


def _pick_fss_template_path() -> Optional[Path]:
    for path in DEFAULT_FSS_TEMPLATE_CANDIDATES:
        if path.is_file():
            return path
    templates_dir = REPO_ROOT / "OldSoftwareEsiBuilder/templates/DB_setup"
    if templates_dir.is_dir():
        files = sorted(templates_dir.glob("*.fss"))
        if files:
            return files[0]
    return None


def _load_fss_template_lines(path: Optional[Path], max_lines: int = 40) -> List[str]:
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    if max_lines <= 0:
        return []
    return [str(x) for x in lines[: int(max_lines)]]


def _resolve_repo_path(path_like: str) -> Optional[Path]:
    txt = str(path_like or "").strip()
    if not txt:
        return None
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        return p.resolve()
    except Exception:
        return p


def _pick_models_metrics_csv_path(preferred: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        pref = preferred.expanduser()
        if not pref.is_absolute():
            pref = REPO_ROOT / pref
        candidates.append(pref)
    candidates.extend(DEFAULT_MODELS_METRICS_CSV_CANDIDATES)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    metadata_dir = REPO_ROOT / "artifacts/60_metadata"
    if metadata_dir.is_dir():
        files = sorted(metadata_dir.glob("reti_metriche_status_*.csv"))
        if files:
            return files[-1].resolve()
    return None


def _pick_encoding_struct_xlsx_path(preferred: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        p = preferred.expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates.append(p)
    candidates.extend(DEFAULT_ENCODING_STRUCT_XLSX_CANDIDATES)
    for c in candidates:
        if c.is_file() and not c.name.startswith("~$"):
            return c.resolve()

    wildcard = sorted(
        [
            p
            for p in REPO_ROOT.glob("*encoding*struct*.xlsx")
            if p.is_file() and not p.name.startswith("~$")
        ],
        key=lambda p: (p.stat().st_mtime, p.name.lower()),
        reverse=True,
    )
    if wildcard:
        return wildcard[0].resolve()
    return None


def _pick_vendor_mapping_csv_path(preferred: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        p = preferred.expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates.append(p)
    candidates.extend(DEFAULT_VENDOR_MAPPING_CSV_CANDIDATES)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    wildcard = sorted(
        [p for p in (REPO_ROOT / "artifacts/60_metadata").glob("vendor_mapping*_encoding_struct*.csv") if p.is_file()],
        key=lambda p: (p.stat().st_mtime, p.name.lower()),
        reverse=True,
    )
    if wildcard:
        return wildcard[0].resolve()
    return None


def _pick_model_merge_mapping_csv_path(preferred: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        p = preferred.expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates.append(p)
    candidates.extend(DEFAULT_MODEL_MERGE_MAPPING_CSV_CANDIDATES)
    for c in candidates:
        if c.is_file():
            return c.resolve()

    review_dir = REPO_ROOT / "artifacts/60_metadata/review_model_merge"
    if review_dir.is_dir():
        wildcard = sorted(
            [p for p in review_dir.glob("model_merge_mapping_applied_*.csv") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name.lower()),
            reverse=True,
        )
        if wildcard:
            return wildcard[0].resolve()
    return None


def _pick_probe_merge_mapping_csv_path(preferred: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        p = preferred.expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates.append(p)
    candidates.extend(DEFAULT_PROBE_MERGE_MAPPING_CSV_CANDIDATES)
    for c in candidates:
        if c.is_file():
            return c.resolve()

    review_dir = REPO_ROOT / "artifacts/60_metadata/review_probe_merge"
    if review_dir.is_dir():
        wildcard = sorted(
            [p for p in review_dir.glob("probe_merge_mapping_applied_*.csv") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name.lower()),
            reverse=True,
        )
        if wildcard:
            return wildcard[0].resolve()
    return None


def _normalize_header_key(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    out = []
    for ch in txt:
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value).strip()


def _normalize_model_name(model_raw: str, probe_name: str) -> str:
    txt = str(model_raw or "").strip()
    if not txt:
        return ""
    # Model field often contains extra suffixes (" - L", " - T", etc.).
    # Keep left-most model portion and strip explicit probe suffixes when present.
    base = txt.split(" - ")[0].strip()

    probe = str(probe_name or "").strip()
    if probe:
        p = re.escape(probe)
        # Remove probe token when appended to vendor with separators.
        base = re.sub(rf"(?i)(?:\s*[,;/]\s*|\s+-\s*){p}\b", " ", base)
        base = re.sub(rf"(?i)\b{p}\b", " ", base)

    base = re.sub(r"\s+", " ", base).strip()
    base = re.sub(r"[\s,;:/_-]+$", "", base).strip()
    return base or txt


def _normalize_model_key(value: str) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _load_vendor_mapping(csv_path: Optional[Path]) -> Dict[str, Any]:
    out = {"exact": {}, "contains": []}
    if csv_path is None or not csv_path.is_file():
        return out
    exact: Dict[str, str] = {}
    contains: List[tuple[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str((row or {}).get("model_raw", "") or "").strip()
                vendor = str((row or {}).get("vendor_canonical", "") or "").strip()
                status = str((row or {}).get("status", "") or "").strip().lower()
                if not raw or not vendor:
                    continue
                if status and status not in {"matched", "manual", "review"}:
                    continue
                k = _normalize_model_key(raw)
                if not k:
                    continue
                if k not in exact:
                    exact[k] = vendor
                contains.append((k, vendor))
    except Exception:
        return out
    contains.sort(key=lambda x: len(x[0]), reverse=True)
    return {"exact": exact, "contains": contains}


def _infer_vendor_from_model(model_text: str) -> str:
    t = str(model_text or "").strip().lower()
    if not t:
        return ""
    rules = [
        ("bk", "BK"),
        ("biopsee", "Biopsee"),
        ("canon", "Canon"),
        ("esaote", "Esaote"),
        ("exactvu", "ExactVu"),
        ("voluson", "GE"),
        ("logiq", "GE"),
        ("versana", "GE"),
        ("hitachi", "Hitachi"),
        ("arietta", "Hitachi"),
        ("aloka", "Hitachi"),
        ("koelis", "Koelis"),
        ("mindray", "Mindray"),
        ("philips", "Philips"),
        ("affiniti", "Philips"),
        ("siemens", "Siemens"),
        ("acuson", "Siemens"),
        ("terason", "Terason"),
        ("toshiba", "Toshiba"),
        ("aplio", "Toshiba"),
        ("alpinion", "Alpinion"),
    ]
    for needle, vendor in rules:
        if needle in t:
            return vendor
    return ""


def _resolve_vendor_canonical(model_raw: str, model_norm: str, vendor_map: Dict[str, Any]) -> str:
    exact = vendor_map.get("exact", {}) if isinstance(vendor_map, dict) else {}
    contains = vendor_map.get("contains", []) if isinstance(vendor_map, dict) else []
    cands = [
        _normalize_model_key(model_raw),
        _normalize_model_key(model_norm),
    ]
    for k in cands:
        if k and isinstance(exact, dict) and k in exact:
            return str(exact.get(k, "") or "").strip()
    for k, v in contains if isinstance(contains, list) else []:
        if not k:
            continue
        for cand in cands:
            if cand and (cand == k or k in cand):
                return str(v or "").strip()
    return _infer_vendor_from_model(model_norm or model_raw)


def _load_model_merge_mapping(csv_path: Optional[Path]) -> Dict[str, Any]:
    out = {"csv_path": "", "exact": {}, "norm": {}, "rows": 0}
    if csv_path is None or not csv_path.is_file():
        return out

    exact: Dict[str, str] = {}
    norm: Dict[str, str] = {}
    rows = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str((row or {}).get("source_model", "") or "").strip()
                canon = str((row or {}).get("canonical_model", "") or "").strip()
                if not raw or not canon:
                    continue
                rows += 1
                exact[raw] = canon
                nk = _normalize_model_key(raw)
                if nk and nk not in norm:
                    norm[nk] = canon
    except Exception:
        return out

    return {
        "csv_path": csv_path.as_posix(),
        "exact": exact,
        "norm": norm,
        "rows": int(rows),
    }


def _resolve_model_canonical(model_name: str, model_merge_map: Dict[str, Any]) -> str:
    txt = str(model_name or "").strip()
    if not txt:
        return ""
    exact = model_merge_map.get("exact", {}) if isinstance(model_merge_map, dict) else {}
    norm = model_merge_map.get("norm", {}) if isinstance(model_merge_map, dict) else {}
    if isinstance(exact, dict) and txt in exact:
        return str(exact.get(txt, "") or "").strip() or txt
    k = _normalize_model_key(txt)
    if k and isinstance(norm, dict) and k in norm:
        return str(norm.get(k, "") or "").strip() or txt
    return txt


def _normalize_probe_id_key(value: str) -> str:
    txt = str(value or "").strip()
    if not txt:
        return "-"
    if txt in {"-", "--"}:
        return "-"
    return txt


def _normalize_probe_name_key(value: str) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return "-"
    txt = re.sub(r"[^a-z0-9]+", "", txt)
    return txt or "-"


def _probe_compound_key(probe_id: str, probe_name: str) -> str:
    return f"{_normalize_probe_id_key(probe_id)}||{_normalize_probe_name_key(probe_name)}"


def _load_probe_merge_mapping(csv_path: Optional[Path]) -> Dict[str, Any]:
    out = {"csv_path": "", "exact": {}, "by_name": {}, "rows": 0}
    if csv_path is None or not csv_path.is_file():
        return out

    exact: Dict[str, tuple[str, str]] = {}
    by_name_candidates: Dict[str, set[tuple[str, str]]] = {}
    rows = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spid = str((row or {}).get("source_probe_id", "") or "").strip() or "-"
                spname = str((row or {}).get("source_probe_name", "") or "").strip() or "-"
                cpid = str((row or {}).get("canonical_probe_id", "") or "").strip() or "-"
                cpname = str((row or {}).get("canonical_probe_name", "") or "").strip() or "-"
                if not spname or not cpname:
                    continue
                rows += 1
                key = _probe_compound_key(spid, spname)
                exact[key] = (cpid, cpname)

                nk = _normalize_probe_name_key(spname)
                if nk and nk != "-":
                    by_name_candidates.setdefault(nk, set()).add((cpid, cpname))
    except Exception:
        return out

    by_name: Dict[str, tuple[str, str]] = {}
    for nk, vals in by_name_candidates.items():
        if len(vals) == 1:
            by_name[nk] = list(vals)[0]

    return {
        "csv_path": csv_path.as_posix(),
        "exact": exact,
        "by_name": by_name,
        "rows": int(rows),
    }


def _resolve_probe_canonical(probe_id: str, probe_name: str, probe_merge_map: Dict[str, Any]) -> tuple[str, str]:
    pid = str(probe_id or "").strip() or "-"
    pname = str(probe_name or "").strip() or "-"
    if not pname or pname == "-":
        return (pid, pname)

    exact = probe_merge_map.get("exact", {}) if isinstance(probe_merge_map, dict) else {}
    by_name = probe_merge_map.get("by_name", {}) if isinstance(probe_merge_map, dict) else {}

    key = _probe_compound_key(pid, pname)
    if isinstance(exact, dict) and key in exact:
        cpid, cpname = exact.get(key, (pid, pname))
        return (str(cpid or "-"), str(cpname or "-"))

    nk = _normalize_probe_name_key(pname)
    if nk and nk != "-" and isinstance(by_name, dict) and nk in by_name:
        cpid, cpname = by_name.get(nk, (pid, pname))
        return (str(cpid or "-"), str(cpname or "-"))

    return (pid, pname)


def _xlsx_col_letters_to_index(letters: str) -> int:
    acc = 0
    for ch in str(letters or "").strip().upper():
        if not ("A" <= ch <= "Z"):
            break
        acc = acc * 26 + (ord(ch) - ord("A") + 1)
    return max(0, acc - 1)


def _xlsx_cell_ref_to_col_index(cell_ref: str) -> int:
    txt = str(cell_ref or "")
    letters = []
    for ch in txt:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    return _xlsx_col_letters_to_index("".join(letters))


def _xlsx_load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    names = set(zf.namelist())
    if "xl/sharedStrings.xml" not in names:
        return []
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except Exception:
        return []
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out: List[str] = []
    for si in root.findall(".//m:si", ns):
        parts = [str(t.text or "") for t in si.findall(".//m:t", ns)]
        out.append("".join(parts))
    return out


def _xlsx_find_fss_sheet(zf: zipfile.ZipFile) -> tuple[str, str]:
    workbook_xml = "xl/workbook.xml"
    rels_xml = "xl/_rels/workbook.xml.rels"
    names = set(zf.namelist())
    if workbook_xml not in names or rels_xml not in names:
        raise RuntimeError("File workbook.xml/workbook.xml.rels non trovato in xlsx.")

    wb_root = ET.fromstring(zf.read(workbook_xml))
    rels_root = ET.fromstring(zf.read(rels_xml))
    ns_wb = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rel_map: Dict[str, str] = {}
    for rel in rels_root.findall("r:Relationship", ns_rel):
        rid = str(rel.get("Id", "") or "").strip()
        target = str(rel.get("Target", "") or "").strip()
        if rid and target:
            rel_map[rid] = target

    selected_name = ""
    selected_target = ""
    sheet_candidates: List[tuple[str, str]] = []
    for sh in wb_root.findall(".//m:sheets/m:sheet", ns_wb):
        name = str(sh.get("name", "") or "").strip()
        rid = str(sh.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "") or "").strip()
        target = rel_map.get(rid, "")
        if name and target:
            sheet_candidates.append((name, target))

    for name, target in sheet_candidates:
        if name.lower() == "fss":
            selected_name = name
            selected_target = target
            break
    if not selected_name:
        for name, target in sheet_candidates:
            if "fss" in name.lower():
                selected_name = name
                selected_target = target
                break
    if not selected_name:
        raise RuntimeError("Foglio FSS non trovato nel file encoding_struct.")

    target_clean = selected_target.lstrip("/")
    sheet_path = f"xl/{target_clean}" if not target_clean.startswith("xl/") else target_clean
    if sheet_path not in names:
        raise RuntimeError(f"Foglio selezionato non trovato nel package xlsx: {sheet_path}")
    return selected_name, sheet_path


def _xlsx_sheet_rows_values(zf: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]) -> List[List[Any]]:
    root = ET.fromstring(zf.read(sheet_path))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows_out: List[List[Any]] = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        row_map: Dict[int, Any] = {}
        max_idx = -1
        for c in row.findall("m:c", ns):
            ref = str(c.get("r", "") or "").strip()
            idx = _xlsx_cell_ref_to_col_index(ref) if ref else len(row_map)
            t = str(c.get("t", "") or "").strip()
            val: Any = ""
            if t == "s":
                vtxt = c.findtext("m:v", default="", namespaces=ns).strip()
                try:
                    vindex = int(vtxt)
                    val = shared_strings[vindex] if 0 <= vindex < len(shared_strings) else ""
                except Exception:
                    val = ""
            elif t == "inlineStr":
                parts = [str(x.text or "") for x in c.findall(".//m:is/m:t", ns)]
                val = "".join(parts)
            elif t == "b":
                btxt = c.findtext("m:v", default="", namespaces=ns).strip()
                val = "1" if btxt == "1" else "0"
            else:
                val = c.findtext("m:v", default="", namespaces=ns).strip()
            row_map[idx] = val
            if idx > max_idx:
                max_idx = idx
        if max_idx < 0:
            rows_out.append([])
            continue
        row_vals: List[Any] = ["" for _ in range(max_idx + 1)]
        for i, v in row_map.items():
            if 0 <= i < len(row_vals):
                row_vals[i] = v
        rows_out.append(row_vals)
    return rows_out


def _load_encoding_struct_fss_catalog_zip(
    xlsx_path: Path,
    vendor_map: Dict[str, Any],
    model_merge_map: Dict[str, Any],
    probe_merge_map: Dict[str, Any],
) -> Dict[str, Any]:
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        sheet_name, sheet_path = _xlsx_find_fss_sheet(zf)
        shared = _xlsx_load_shared_strings(zf)
        rows = _xlsx_sheet_rows_values(zf, sheet_path, shared)
    if not rows:
        raise RuntimeError("Foglio FSS vuoto.")

    headers = rows[0]
    header_map: Dict[str, int] = {}
    for idx, h in enumerate(headers):
        k = _normalize_header_key(h)
        if k and k not in header_map:
            header_map[k] = idx
    vendor_idx = header_map.get("modelloecografo")
    probe_name_idx = header_map.get("modellosonda")
    probe_id_idx = header_map.get("idsonda")
    if vendor_idx is None or probe_name_idx is None or probe_id_idx is None:
        raise RuntimeError(
            "Colonne richieste non trovate nel foglio FSS "
            "(attese: Modello ecografo, Modello sonda, ID SONDA)."
        )

    vendor_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    raw_model_counts: Dict[str, int] = {}
    probe_counts: Dict[tuple[str, str], int] = {}
    model_merged_occurrences = 0
    raw_probe_counts: Dict[tuple[str, str], int] = {}
    probe_merged_occurrences = 0
    rows_scanned = 0
    for vals in rows[1:]:
        rows_scanned += 1
        model_name_raw = _cell_to_text(vals[vendor_idx] if vendor_idx < len(vals) else "")
        probe_name = _cell_to_text(vals[probe_name_idx] if probe_name_idx < len(vals) else "")
        probe_id = _cell_to_text(vals[probe_id_idx] if probe_id_idx < len(vals) else "")
        model_name = _normalize_model_name(model_name_raw, probe_name)
        vendor_name = _resolve_vendor_canonical(model_name_raw, model_name, vendor_map)
        if model_name:
            raw_model_counts[model_name] = int(raw_model_counts.get(model_name, 0)) + 1
            canonical_model = _resolve_model_canonical(model_name, model_merge_map)
            if canonical_model != model_name:
                model_merged_occurrences += 1
            model_counts[canonical_model] = int(model_counts.get(canonical_model, 0)) + 1
        if vendor_name:
            vendor_counts[vendor_name] = int(vendor_counts.get(vendor_name, 0)) + 1
        if probe_id or probe_name:
            raw_key = (probe_id or "-", probe_name or "-")
            raw_probe_counts[raw_key] = int(raw_probe_counts.get(raw_key, 0)) + 1
            canonical_probe_id, canonical_probe_name = _resolve_probe_canonical(probe_id, probe_name, probe_merge_map)
            if (canonical_probe_id, canonical_probe_name) != raw_key:
                probe_merged_occurrences += 1
            key = (canonical_probe_id or "-", canonical_probe_name or "-")
            probe_counts[key] = int(probe_counts.get(key, 0)) + 1

    vendor_rows = [
        {"vendor_name": name, "count": count}
        for name, count in sorted(vendor_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    ]
    probe_rows = [
        {"probe_id": pid, "probe_name": pname, "count": count}
        for (pid, pname), count in sorted(
            probe_counts.items(),
            key=lambda kv: (-kv[1], kv[0][1].lower(), kv[0][0].lower()),
        )
    ]
    model_rows = [
        {"model_name": name, "count": count}
        for name, count in sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    ]
    return {
        "ok": True,
        "error": "",
        "xlsx_path": xlsx_path.as_posix(),
        "sheet_name": str(sheet_name),
        "rows_scanned": int(rows_scanned),
        "vendors": vendor_rows,
        "models": model_rows,
        "probes": probe_rows,
        "summary": {
            "vendor_unique": int(len(vendor_rows)),
            "vendor_total_occurrences": int(sum(int(x.get("count", 0)) for x in vendor_rows)),
            "model_unique": int(len(model_rows)),
            "model_total_occurrences": int(sum(int(x.get("count", 0)) for x in model_rows)),
            "model_raw_unique": int(len(raw_model_counts)),
            "model_merged_occurrences": int(model_merged_occurrences),
            "model_mapping_rows": int(model_merge_map.get("rows", 0) if isinstance(model_merge_map, dict) else 0),
            "model_mapping_csv_path": str(model_merge_map.get("csv_path", "") if isinstance(model_merge_map, dict) else ""),
            "probe_unique": int(len(probe_rows)),
            "probe_total_occurrences": int(sum(int(x.get("count", 0)) for x in probe_rows)),
            "probe_raw_unique": int(len(raw_probe_counts)),
            "probe_merged_occurrences": int(probe_merged_occurrences),
            "probe_mapping_rows": int(probe_merge_map.get("rows", 0) if isinstance(probe_merge_map, dict) else 0),
            "probe_mapping_csv_path": str(probe_merge_map.get("csv_path", "") if isinstance(probe_merge_map, dict) else ""),
        },
    }


def _load_encoding_struct_fss_catalog(xlsx_path: Optional[Path]) -> Dict[str, Any]:
    vendor_map = _load_vendor_mapping(_pick_vendor_mapping_csv_path())
    model_merge_map = _load_model_merge_mapping(_pick_model_merge_mapping_csv_path())
    probe_merge_map = _load_probe_merge_mapping(_pick_probe_merge_mapping_csv_path())
    base = {
        "ok": False,
        "error": "",
        "xlsx_path": xlsx_path.as_posix() if xlsx_path else "",
        "sheet_name": "",
        "rows_scanned": 0,
        "vendors": [],
        "models": [],
        "probes": [],
        "summary": {
            "vendor_unique": 0,
            "vendor_total_occurrences": 0,
            "model_unique": 0,
            "model_total_occurrences": 0,
            "model_raw_unique": 0,
            "model_merged_occurrences": 0,
            "model_mapping_rows": int(model_merge_map.get("rows", 0) if isinstance(model_merge_map, dict) else 0),
            "model_mapping_csv_path": str(model_merge_map.get("csv_path", "") if isinstance(model_merge_map, dict) else ""),
            "probe_unique": 0,
            "probe_total_occurrences": 0,
            "probe_raw_unique": 0,
            "probe_merged_occurrences": 0,
            "probe_mapping_rows": int(probe_merge_map.get("rows", 0) if isinstance(probe_merge_map, dict) else 0),
            "probe_mapping_csv_path": str(probe_merge_map.get("csv_path", "") if isinstance(probe_merge_map, dict) else ""),
        },
    }
    if xlsx_path is None:
        base["error"] = "File encoding_struct non trovato."
        return base
    if not xlsx_path.is_file():
        base["error"] = f"File encoding_struct non trovato: {xlsx_path}"
        return base

    if load_workbook is None:
        try:
            return _load_encoding_struct_fss_catalog_zip(xlsx_path, vendor_map, model_merge_map, probe_merge_map)
        except Exception as exc:
            base["error"] = f"Errore parsing xlsx (fallback xml): {exc}"
            return base

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception as exc:
        base["error"] = f"Errore apertura encoding_struct: {exc}"
        return base

    try:
        sheet_name = ""
        for s in wb.sheetnames:
            if str(s).strip().lower() == "fss":
                sheet_name = s
                break
        if not sheet_name:
            for s in wb.sheetnames:
                if "fss" in str(s).strip().lower():
                    sheet_name = s
                    break
        if not sheet_name:
            base["error"] = "Foglio FSS non trovato nel file encoding_struct."
            return base

        ws = wb[sheet_name]
        header_values = [ws.cell(1, c).value for c in range(1, int(ws.max_column or 0) + 1)]
        header_map: Dict[str, int] = {}
        for idx, h in enumerate(header_values):
            k = _normalize_header_key(h)
            if k and k not in header_map:
                header_map[k] = idx

        vendor_idx = header_map.get("modelloecografo")
        probe_name_idx = header_map.get("modellosonda")
        probe_id_idx = header_map.get("idsonda")

        if vendor_idx is None or probe_name_idx is None or probe_id_idx is None:
            base["error"] = (
                "Colonne richieste non trovate nel foglio FSS "
                "(attese: Modello ecografo, Modello sonda, ID SONDA)."
            )
            return base

        vendor_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        raw_model_counts: Dict[str, int] = {}
        probe_counts: Dict[tuple[str, str], int] = {}
        model_merged_occurrences = 0
        raw_probe_counts: Dict[tuple[str, str], int] = {}
        probe_merged_occurrences = 0
        rows_scanned = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row is None:
                continue
            vals = list(row)
            rows_scanned += 1
            model_name_raw = _cell_to_text(vals[vendor_idx] if vendor_idx < len(vals) else "")
            probe_name = _cell_to_text(vals[probe_name_idx] if probe_name_idx < len(vals) else "")
            probe_id = _cell_to_text(vals[probe_id_idx] if probe_id_idx < len(vals) else "")
            model_name = _normalize_model_name(model_name_raw, probe_name)
            vendor_name = _resolve_vendor_canonical(model_name_raw, model_name, vendor_map)

            if model_name:
                raw_model_counts[model_name] = int(raw_model_counts.get(model_name, 0)) + 1
                canonical_model = _resolve_model_canonical(model_name, model_merge_map)
                if canonical_model != model_name:
                    model_merged_occurrences += 1
                model_counts[canonical_model] = int(model_counts.get(canonical_model, 0)) + 1
            if vendor_name:
                vendor_counts[vendor_name] = int(vendor_counts.get(vendor_name, 0)) + 1
            if probe_id or probe_name:
                raw_key = (probe_id or "-", probe_name or "-")
                raw_probe_counts[raw_key] = int(raw_probe_counts.get(raw_key, 0)) + 1
                canonical_probe_id, canonical_probe_name = _resolve_probe_canonical(probe_id, probe_name, probe_merge_map)
                if (canonical_probe_id, canonical_probe_name) != raw_key:
                    probe_merged_occurrences += 1
                key = (canonical_probe_id or "-", canonical_probe_name or "-")
                probe_counts[key] = int(probe_counts.get(key, 0)) + 1

        vendor_rows = [
            {"vendor_name": name, "count": count}
            for name, count in sorted(vendor_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        ]
        probe_rows = [
            {"probe_id": pid, "probe_name": pname, "count": count}
            for (pid, pname), count in sorted(
                probe_counts.items(),
                key=lambda kv: (-kv[1], kv[0][1].lower(), kv[0][0].lower()),
            )
        ]
        model_rows = [
            {"model_name": name, "count": count}
            for name, count in sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        ]

        return {
            "ok": True,
            "error": "",
            "xlsx_path": xlsx_path.as_posix(),
            "sheet_name": str(sheet_name),
            "rows_scanned": int(rows_scanned),
            "vendors": vendor_rows,
            "models": model_rows,
            "probes": probe_rows,
            "summary": {
                "vendor_unique": int(len(vendor_rows)),
                "vendor_total_occurrences": int(sum(int(x.get("count", 0)) for x in vendor_rows)),
                "model_unique": int(len(model_rows)),
                "model_total_occurrences": int(sum(int(x.get("count", 0)) for x in model_rows)),
                "model_raw_unique": int(len(raw_model_counts)),
                "model_merged_occurrences": int(model_merged_occurrences),
                "model_mapping_rows": int(model_merge_map.get("rows", 0) if isinstance(model_merge_map, dict) else 0),
                "model_mapping_csv_path": str(model_merge_map.get("csv_path", "") if isinstance(model_merge_map, dict) else ""),
                "probe_unique": int(len(probe_rows)),
                "probe_total_occurrences": int(sum(int(x.get("count", 0)) for x in probe_rows)),
                "probe_raw_unique": int(len(raw_probe_counts)),
                "probe_merged_occurrences": int(probe_merged_occurrences),
                "probe_mapping_rows": int(probe_merge_map.get("rows", 0) if isinstance(probe_merge_map, dict) else 0),
                "probe_mapping_csv_path": str(probe_merge_map.get("csv_path", "") if isinstance(probe_merge_map, dict) else ""),
            },
        }
    except Exception as exc:
        base["error"] = f"Errore parsing foglio FSS: {exc}"
        return base
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _safe_float_or_none(value: Any) -> Optional[float]:
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _status_priority_models(update_status: str) -> int:
    s = str(update_status or "").strip().upper()
    if s == "AGGIORNARE":
        return 0
    if s == "VERIFICARE":
        return 1
    if s == "MONITORARE":
        return 2
    if s == "OK":
        return 3
    return 4


def _load_models_metrics_registry(csv_path: Optional[Path]) -> Dict[str, Any]:
    if csv_path is None:
        return {
            "ok": False,
            "error": "Nessun CSV metriche disponibile",
            "csv_path": "",
            "rows": [],
            "summary": {},
            "columns": [],
        }
    if not csv_path.is_file():
        return {
            "ok": False,
            "error": f"CSV metriche non trovato: {csv_path}",
            "csv_path": csv_path.as_posix(),
            "rows": [],
            "summary": {},
            "columns": [],
        }

    rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    scope_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    with_metrics_count = 0
    checkpoint_only_count = 0
    missing_checkpoint_count = 0
    missing_metrics_count = 0
    small_sample_count = 0

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            for idx, raw_row in enumerate(reader, start=1):
                raw = raw_row if isinstance(raw_row, dict) else {}
                family = str(raw.get("family", "") or "").strip()
                scope = str(raw.get("scope", "") or "").strip()
                vendor = str(raw.get("vendor", "") or "").strip()
                run_name = str(raw.get("run_name", "") or "").strip()
                checkpoint_path = str(raw.get("checkpoint_path", "") or "").strip()
                metrics_path = str(raw.get("metrics_path", "") or "").strip()
                primary_metric = str(raw.get("primary_metric", "") or "").strip()
                secondary_metrics = str(raw.get("secondary_metrics", "") or "").strip()
                update_status = str(raw.get("update_status", "") or "").strip().upper() or "UNKNOWN"
                update_reason = str(raw.get("update_reason", "") or "").strip()
                notes = str(raw.get("notes", "") or "").strip()

                primary_value_num = _safe_float_or_none(raw.get("primary_value"))
                test_samples_num = _safe_float_or_none(raw.get("test_samples"))

                if primary_metric and primary_metric.lower() != "none" and primary_value_num is not None:
                    with_metrics_count += 1
                if family.lower() == "checkpoint_only" or primary_metric.lower() == "none":
                    checkpoint_only_count += 1
                if test_samples_num is not None and test_samples_num < 50:
                    small_sample_count += 1

                checkpoint_abs = _resolve_repo_path(checkpoint_path)
                metrics_abs = _resolve_repo_path(metrics_path)
                checkpoint_exists = bool(checkpoint_abs and checkpoint_abs.is_file())
                metrics_exists = bool(metrics_abs and metrics_abs.is_file())
                if checkpoint_abs and not checkpoint_exists:
                    missing_checkpoint_count += 1
                if metrics_abs and not metrics_exists:
                    missing_metrics_count += 1

                status_counts[update_status] = int(status_counts.get(update_status, 0)) + 1
                if scope:
                    scope_counts[scope] = int(scope_counts.get(scope, 0)) + 1
                if family:
                    family_counts[family] = int(family_counts.get(family, 0)) + 1

                rows.append(
                    {
                        "row_index": idx,
                        "family": family,
                        "scope": scope,
                        "vendor": vendor,
                        "run_name": run_name,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_abs": checkpoint_abs.as_posix() if checkpoint_abs else "",
                        "checkpoint_exists": checkpoint_exists,
                        "metrics_path": metrics_path,
                        "metrics_abs": metrics_abs.as_posix() if metrics_abs else "",
                        "metrics_exists": metrics_exists,
                        "test_samples": str(raw.get("test_samples", "") or "").strip(),
                        "test_samples_num": test_samples_num,
                        "primary_metric": primary_metric,
                        "primary_value": str(raw.get("primary_value", "") or "").strip(),
                        "primary_value_num": primary_value_num,
                        "secondary_metrics": secondary_metrics,
                        "update_status": update_status,
                        "update_reason": update_reason,
                        "notes": notes,
                    }
                )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Errore parsing CSV metriche: {exc}",
            "csv_path": csv_path.as_posix(),
            "rows": [],
            "summary": {},
            "columns": [],
        }

    rows.sort(
        key=lambda r: (
            _status_priority_models(str(r.get("update_status", ""))),
            str(r.get("family", "")).lower(),
            str(r.get("scope", "")).lower(),
            str(r.get("vendor", "")).lower(),
            str(r.get("run_name", "")).lower(),
        )
    )

    ranked_rows = [
        r
        for r in rows
        if r.get("primary_value_num") is not None and str(r.get("primary_metric", "")).lower() != "none"
    ]
    ranked_rows.sort(
        key=lambda r: (
            float(r.get("primary_value_num") or 0.0),
            float(r.get("test_samples_num") or 0.0),
        ),
        reverse=True,
    )
    top_models = [
        {
            "family": str(r.get("family", "")),
            "vendor": str(r.get("vendor", "")),
            "run_name": str(r.get("run_name", "")),
            "primary_metric": str(r.get("primary_metric", "")),
            "primary_value_num": r.get("primary_value_num"),
            "test_samples_num": r.get("test_samples_num"),
            "update_status": str(r.get("update_status", "")),
        }
        for r in ranked_rows[:8]
    ]

    summary = {
        "total_rows": len(rows),
        "with_metrics_count": int(with_metrics_count),
        "checkpoint_only_count": int(checkpoint_only_count),
        "small_sample_count": int(small_sample_count),
        "missing_checkpoint_count": int(missing_checkpoint_count),
        "missing_metrics_count": int(missing_metrics_count),
        "status_counts": status_counts,
        "scope_counts": scope_counts,
        "family_counts": family_counts,
        "top_models": top_models,
    }
    return {
        "ok": True,
        "error": "",
        "csv_path": csv_path.as_posix(),
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


HTML_PAGE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {% if auto_refresh_seconds and auto_refresh_seconds > 0 %}
  <meta http-equiv="refresh" content="{{ auto_refresh_seconds }}">
  {% endif %}
  <title>Pipeline AIESIBuilder</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f4f6fb;
      --bg-overlay: rgba(71, 85, 105, 0.14);
      --card: #ffffff;
      --ink: #101828;
      --muted: #667085;
      --line: #d0d5dd;
      --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.05);
      --shadow-md: 0 10px 24px rgba(16, 24, 40, 0.08);
      --primary: #0f766e;
      --primary-2: #115e59;
      --ok-bg: #ecfdf3;
      --ok-line: #75d29a;
      --warn-bg: #fffaeb;
      --warn-line: #f0c36b;
      --err-bg: #fff1f3;
      --err-line: #f39aa8;
      --monitor: #0ea5e9;
      --analysis: #22c55e;
      --evidence: #f59e0b;
      --vendor: #0f766e;
      --probe: #4f46e5;
      --rect: #dc2626;
      --rotation: #ea580c;
      --data: #0891b2;
      --model: #64748b;
      --evidence-img-max-h: min(46vh, 420px);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f172a;
        --bg-overlay: rgba(148, 163, 184, 0.18);
      }
    }
    html, body {
      height: 100%;
      min-height: 100%;
      width: 100%;
      max-width: 100%;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 14px;
      font-family: "Avenir Next", "Nunito Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background-color: var(--bg);
      background-image: linear-gradient(180deg, var(--bg-overlay) 0%, rgba(71, 85, 105, 0.04) 42%, rgba(71, 85, 105, 0.00) 70%);
      overflow-y: auto;
      overflow-x: hidden;
    }
    html[data-darkreader-scheme="dark"] body {
      background-color: #0f172a !important;
      background-image: linear-gradient(180deg, rgba(148, 163, 184, 0.18) 0%, rgba(148, 163, 184, 0.06) 45%, rgba(148, 163, 184, 0.00) 75%) !important;
    }
    .app-shell {
      max-width: 1820px;
      min-height: calc(100vh - 28px);
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow-x: auto;
    }
    .header {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow-sm);
      padding: 16px 18px;
      margin-bottom: 14px;
      flex: 0 0 auto;
    }
    h1 { margin: 0 0 4px; font-size: 24px; letter-spacing: 0.2px; }
    .header h1 a { color: var(--ink); text-decoration: none; }
    .header h1 a:hover { text-decoration: underline; }
    .muted { color: var(--muted); font-size: 13px; }
    .app-layout {
      flex: 1 1 auto;
      min-height: 0;
      min-width: 980px;
      align-items: start;
      display: grid;
      grid-template-columns: minmax(0, 3fr) minmax(320px, 1fr);
      gap: 14px;
    }
    .main-column,
    .workspace-shell {
      min-height: 0;
    }
    .main-column {
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding-right: 2px;
      overflow: visible;
    }
    .panel-core {
      width: 100%;
      border-width: 2px;
      box-shadow: var(--shadow-md);
      border-color: #9ed9d4;
    }
    .home-models {
      margin-top: 12px;
      border: 1px solid #cfd8e3;
      border-radius: 12px;
      background: #f8fafc;
      padding: 10px;
      box-shadow: var(--shadow-sm);
    }
    .home-encoding {
      margin-top: 12px;
      border: 1px solid #cfd8e3;
      border-radius: 12px;
      background: #f8fafc;
      padding: 10px;
      box-shadow: var(--shadow-sm);
    }
    .home-run-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(280px, 0.8fr);
      gap: 10px;
      align-items: stretch;
    }
    .run-mode-section,
    .rerun-box,
    .exclusion-box {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fbfdff;
      padding: 10px;
      min-width: 0;
    }
    .run-mode-section.bulk-mode {
      background: #f8fafc;
    }
    .run-mode-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .run-mode-title h3 {
      margin: 0;
      font-size: 14px;
      line-height: 1.2;
      color: #0f172a;
    }
    .input-browse.compact {
      margin-bottom: 8px;
    }
    .compact-params,
    .rerun-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(84px, 1fr));
      gap: 8px;
      align-items: end;
      min-width: 0;
    }
    .compact-params.single {
      grid-template-columns: minmax(220px, 1fr) auto auto;
      margin-top: 8px;
    }
    .compact-params input,
    .compact-params select,
    .rerun-grid input,
    .rerun-grid select {
      padding: 6px 8px;
      font-size: 12px;
    }
    .check-line {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin: 0;
      color: #334155;
      font-size: 12px;
      white-space: nowrap;
    }
    .check-line input {
      width: auto;
      margin: 0;
    }
    .exclusion-box,
    .rerun-box {
      margin-top: 10px;
    }
    .exclusion-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .mini-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }
    .image-exclusion-list {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #fff;
      max-height: 190px;
      overflow: auto;
      min-height: 38px;
    }
    .image-exclusion-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 6px 8px;
      border-bottom: 1px solid #eef2f7;
      font-size: 12px;
    }
    .image-exclusion-row:last-child {
      border-bottom: none;
    }
    .image-exclusion-row input {
      width: auto;
      margin: 0;
    }
    .image-exclusion-row .path {
      overflow-wrap: anywhere;
      word-break: break-word;
      color: #334155;
    }
    .image-exclusion-row .size {
      color: #64748b;
      font-size: 11px;
      white-space: nowrap;
    }
    .compact-actions {
      margin-top: 8px;
    }
    .rerun-box {
      background: #f0f9ff;
      border-color: #bae6fd;
    }
    .rerun-grid {
      grid-template-columns: minmax(140px, 0.9fr) minmax(160px, 1.1fr) minmax(96px, 0.7fr) minmax(96px, 0.7fr) auto auto;
    }
    .rerun-step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 6px;
      margin-top: 8px;
    }
    .rerun-step {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      background: #fff;
      padding: 7px 8px;
      min-width: 0;
      cursor: pointer;
    }
    .rerun-step input {
      width: auto;
      margin: 2px 0 0;
    }
    .rerun-step .name {
      font-size: 12px;
      font-weight: 750;
      color: #0f172a;
      line-height: 1.2;
    }
    .rerun-step .hint {
      font-size: 11px;
      color: #64748b;
      line-height: 1.25;
      margin-top: 2px;
    }
    .rerun-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      margin-top: 8px;
    }
    .workspace-rerun-panel {
      border: 1px solid #bae6fd;
      border-radius: 8px;
      background: #f0f9ff;
      margin: 8px 0 10px;
      padding: 8px 10px;
    }
    .workspace-rerun-panel summary {
      cursor: pointer;
      font-size: 13px;
      font-weight: 800;
      color: #075985;
    }
    .workspace-rerun-panel .rerun-step-grid {
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .compact-models {
      margin-top: 10px;
      max-height: min(36vh, 360px);
    }
    @media (max-width: 1280px) {
      .home-run-grid { grid-template-columns: 1fr; }
      .compact-params { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .rerun-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .input-browse.compact { grid-template-columns: minmax(0, 1fr) auto auto; }
      .compact-params { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .compact-params.single { grid-template-columns: minmax(0, 1fr) auto auto; }
      .rerun-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .exclusion-head {
        display: grid;
      }
      .check-line {
        white-space: normal;
      }
    }
    @media (max-width: 560px) {
      .input-browse.compact,
      .compact-params,
      .compact-params.single,
      .rerun-grid {
        grid-template-columns: 1fr;
      }
    }
    .home-encoding-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
      min-width: 0;
    }
    @media (max-width: 1380px) {
      .home-encoding-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 1050px) {
      .home-encoding-grid { grid-template-columns: 1fr; }
    }
    .home-encoding-box {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .home-encoding-box h3 {
      margin: 0 0 6px;
      font-size: 14px;
      color: #0f172a;
    }
    .home-encoding-list {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      max-height: min(30vh, 300px);
      overflow: auto;
      background: #fff;
    }
    .home-encoding-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
      padding: 7px 8px;
      border-bottom: 1px solid #eef2f7;
      font-size: 12px;
      color: #334155;
    }
    .home-encoding-item:last-child {
      border-bottom: none;
    }
    .home-encoding-item .label {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .home-encoding-count {
      border: 1px solid #99f6e4;
      border-radius: 999px;
      background: #ecfeff;
      color: #0f766e;
      font-weight: 700;
      font-size: 11px;
      padding: 2px 8px;
      white-space: nowrap;
    }
    .home-models-toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-end;
      margin-bottom: 8px;
    }
    .models-kpi-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
      margin-bottom: 8px;
    }
    @media (max-width: 1320px) {
      .models-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 820px) {
      .models-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .models-kpi {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
    }
    .models-kpi .k {
      font-size: 11px;
      color: #667085;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .models-kpi .v {
      margin-top: 4px;
      font-size: 22px;
      line-height: 1;
      font-weight: 800;
      color: #0f172a;
    }
    .models-table-wrap {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fff;
      overflow: auto;
      max-height: min(52vh, 560px);
    }
    .models-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1180px;
      font-size: 12px;
    }
    .models-table-title {
      position: sticky;
      left: 0;
      padding: 8px;
      font-weight: 800;
      color: #0f172a;
      background: #f8fafc;
      border-bottom: 1px solid #e5e7eb;
    }
    .models-table th,
    .models-table td {
      border-bottom: 1px solid #eceef3;
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .models-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f1f5f9;
      color: #334155;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      border-bottom: 1px solid #d8dee8;
    }
    .models-table tr:hover td { background: #f8fbff; }
    .models-metric {
      font-weight: 700;
      color: #0b4f49;
      white-space: nowrap;
    }
    .models-secondary {
      display: block;
      margin-top: 4px;
      color: #64748b;
      font-size: 11px;
      line-height: 1.25;
    }
    .models-path {
      display: block;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      line-height: 1.25;
      color: #475569;
      margin-top: 3px;
    }
    .models-top-list {
      margin-top: 8px;
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
    }
    .models-top-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      border-bottom: 1px solid #eef2f7;
      padding: 6px 0;
    }
    .models-top-row:last-child {
      border-bottom: none;
      padding-bottom: 0;
    }
    .models-top-row:first-child { padding-top: 0; }
    .status-unknown {
      background: #f8fafc;
      border-color: #cbd5e1;
      color: #334155;
    }
    .history-column {
      min-height: 0;
      padding-left: 2px;
      position: sticky;
      top: 8px;
      max-height: calc(100vh - 28px);
      align-self: start;
    }
    .history-column .panel {
      height: 100%;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: var(--shadow-sm);
    }
    .workspace-shell {
      display: flex;
      flex-direction: column;
      min-height: 0;
      width: 100%;
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 14px;
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }
    .workspace-meta {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn.back {
      border-color: #cbd5e1;
      background: #fff;
      color: #334155;
    }
    body.setup-mode .workspace-shell { display: none; }
    body.workspace-mode .panel-core { display: none; }
    body.workspace-mode .workspace-shell { display: flex; }
    body.workspace-mode .history-column { display: block; }
    body.workspace-mode .app-layout { grid-template-columns: minmax(0, 3fr) minmax(320px, 1fr); }
    body.setup-mode .history-panel-workspace { display: none !important; }
    body.workspace-mode .history-panel-setup { display: none !important; }
    body.workspace-mode .history-panel-workspace { display: flex !important; }
    .history-panel-workspace { display: none; }
    body.workspace-mode .monitor-layout,
    body.workspace-mode .analysis-layout,
    body.workspace-mode .evidence-layout {
      grid-template-columns: minmax(0, 1fr);
    }
    body.workspace-mode .monitor-side,
    body.workspace-mode .analysis-side,
    body.workspace-mode .evidence-side {
      display: none;
    }
    body.workspace-mode #predictionSummary,
    body.workspace-mode #analysisReviewSummary {
      display: none !important;
    }
    .workspace-rail {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 0;
      width: 100%;
    }
    .rail-block {
      display: none;
      min-height: 0;
    }
    .rail-block.active {
      display: block;
    }
    .workspace-rail .evidence-card {
      margin-top: 0;
    }
    .workspace-rail .decision-tree {
      max-height: calc(100vh - 240px);
      overflow: auto;
      padding-right: 4px;
    }
    .workspace-topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border-bottom: 1px solid #e4e7ec;
      padding: 10px 12px;
    }
    .folder-badge {
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: #334155;
      background: #fff;
      min-width: 0;
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .folder-path {
      font-weight: 700;
      color: #0f172a;
    }
    @media (max-width: 1200px) {
      .workspace-topbar { grid-template-columns: 1fr; }
      .folder-badge { max-width: 100%; }
    }
    .pane-nav {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      background: transparent;
      border: 0;
      padding: 0;
      flex: 0 0 auto;
    }
    .evidence-nav {
      margin-bottom: 6px;
    }
    .pane-btn {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #1f2937;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    }
    .pane-btn:hover { background: #f8fafc; border-color: #94a3b8; }
    .pane-btn.active {
      border-color: var(--primary);
      background: #dff7f4;
      color: var(--primary-2);
    }
    .pane-stack {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 12px;
    }
    .pane-panel {
      display: block;
      height: auto;
      margin-bottom: 0;
      overflow: visible;
    }
    .workspace-shell .pane-panel {
      border: none;
      box-shadow: none;
      border-radius: 0;
      background: transparent;
      padding: 0;
    }
    .pane-panel[hidden] { display: none; }
    .batch-tools,
    .study-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: end;
      margin-top: 10px;
    }
    .batch-tools input,
    .study-toolbar select {
      min-width: 220px;
    }
    .study-kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .study-kpi {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
      min-width: 0;
    }
    .study-kpi .k {
      font-size: 11px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
    }
    .study-kpi .v {
      font-size: 16px;
      font-weight: 800;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .study-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .study-frame {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .study-frame img {
      display: block;
      width: 100%;
      max-height: 190px;
      object-fit: contain;
      background: #f8fafc;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
    }
    @media (max-width: 900px) {
      .study-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .monitor-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.65fr);
      gap: 12px;
      align-items: start;
      min-width: 0;
    }
    @media (max-width: 1200px) {
      .monitor-layout { grid-template-columns: 1fr; }
    }
    .monitor-main,
    .monitor-side {
      min-width: 0;
    }
    .monitor-left,
    .monitor-right {
      border: 1px solid #e4e7ec;
      border-radius: 10px;
      background: #fcfdff;
      padding: 10px;
      min-width: 0;
    }
    .monitor-left pre.log {
      max-height: min(58vh, 620px);
    }
    .monitor-right .decision-tree {
      max-height: min(62vh, 680px);
      overflow: auto;
      padding-right: 4px;
    }
    .monitor-activity {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 6px;
      margin: 10px 0;
      padding: 10px 12px;
      border: 1px solid #bae6fd;
      border-left: 4px solid var(--monitor);
      border-radius: 8px;
      background: #f0f9ff;
      color: #075985;
      min-width: 0;
    }
    .monitor-activity.is-hot {
      border-color: #38bdf8;
      background: #e0f2fe;
      box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.12);
    }
    .monitor-activity .activity-title {
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .monitor-activity .activity-main {
      font-size: 15px;
      font-weight: 750;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .monitor-activity .activity-meta {
      font-size: 12px;
      color: #0369a1;
      overflow-wrap: anywhere;
    }
    .monitor-activity .activity-events {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .monitor-activity .activity-event {
      border: 1px solid #7dd3fc;
      background: #fff;
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 11px;
      color: #075985;
    }
    .analysis-tree-card {
      margin-top: 10px;
    }
    .analysis-tree-card .decision-tree {
      max-height: min(44vh, 460px);
      overflow: auto;
      padding-right: 4px;
    }
    .history-panel-workspace .decision-tree {
      max-height: calc(100vh - 210px);
      overflow: auto;
      padding: 0 4px 0 0;
    }
    .decision-tree {
      position: relative;
      padding-left: 0;
      margin-top: 8px;
    }
    .decision-tree::before {
      content: "";
      position: absolute;
      left: 7px;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #d9e2ec;
      border-radius: 2px;
    }
    .dt-node {
      position: relative;
      margin: 0 0 12px;
      padding-left: 22px;
    }
    .dt-node:last-child { margin-bottom: 0; }
    .dt-node::before {
      content: "";
      position: absolute;
      left: 3px;
      top: 3px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #cbd5e1;
      border: 2px solid #fff;
      box-shadow: 0 0 0 1px #cbd5e1;
    }
    .dt-node .dt-title {
      font-size: 13px;
      font-weight: 700;
      color: #1f2937;
    }
    .dt-node .dt-meta {
      margin-top: 2px;
      font-size: 11px;
      color: #64748b;
      word-break: break-word;
    }
    .dt-node.done::before {
      background: #22c55e;
      box-shadow: 0 0 0 1px #22c55e;
    }
    .dt-node.running::before {
      background: #0ea5e9;
      box-shadow: 0 0 0 1px #0ea5e9;
      animation: pulseDot 1.1s ease-in-out infinite;
    }
    .dt-node.error::before {
      background: #ef4444;
      box-shadow: 0 0 0 1px #ef4444;
    }
    .dt-node.pending::before {
      background: #cbd5e1;
      box-shadow: 0 0 0 1px #cbd5e1;
    }
    @keyframes pulseDot {
      0% { transform: scale(1); opacity: 0.9; }
      50% { transform: scale(1.25); opacity: 1; }
      100% { transform: scale(1); opacity: 0.9; }
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow-sm);
      padding: 14px;
      margin-bottom: 0;
      min-width: 0;
      overflow-x: hidden;
    }
    .panel h2 { margin: 0 0 10px; font-size: 17px; }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .panel-head h2 {
      margin: 0;
      min-width: 0;
    }
    .row { display: grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 8px; }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
    @media (max-width: 760px) {
      .row2 { grid-template-columns: 1fr; }
    }
    .input-browse {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: end;
    }
    @media (max-width: 900px) {
      .input-browse { grid-template-columns: 1fr; }
    }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    input, select, textarea, button {
      font-family: inherit;
      font-size: 14px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 70px; resize: vertical; }
    .btn {
      border: 1px solid var(--primary);
      background: var(--primary);
      color: #fff;
      border-radius: 8px;
      padding: 9px 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-block;
      text-decoration: none;
      text-align: center;
    }
    .btn.secondary {
      color: var(--primary-2);
      background: #e6fffb;
      border-color: #99f6e4;
    }
    .btn.danger {
      color: #8a1c2f;
      background: #fff1f3;
      border-color: #f5c2cb;
    }
    .btn.mini {
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 600;
    }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; }
    .btn.is-disabled {
      opacity: 0.55;
      cursor: not-allowed;
      pointer-events: none;
    }
    .run-list {
      overflow-y: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      min-height: 240px;
      flex: 1 1 auto;
    }
    .run-item {
      padding: 10px;
      border-bottom: 1px solid #eef1f4;
      cursor: pointer;
      border-left: 3px solid transparent;
      transition: background 0.15s ease, border-color 0.15s ease;
    }
    .run-item:last-child { border-bottom: none; }
    .run-item:hover { background: #f8fafc; }
    .run-item.active { background: #ecfeff; border-left-color: var(--primary); }
    .run-item.openable:hover { background: #f0f9ff; }
    .run-item .t { font-weight: 600; font-size: 13px; }
    .run-item .s {
      font-size: 12px;
      color: var(--muted);
      margin-top: 3px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .run-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 7px;
    }
    .inline-form { display: inline; margin: 0; }
    .status-pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      margin-right: 4px;
      background: #fafafa;
    }
    .status-ok { background: var(--ok-bg); border-color: var(--ok-line); }
    .status-review { background: var(--warn-bg); border-color: var(--warn-line); }
    .status-error { background: var(--err-bg); border-color: var(--err-line); }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    @media (max-width: 1000px) {
      .kpi-grid { grid-template-columns: 1fr; }
    }
    .kpi {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
    }
    .kpi .k { font-size: 12px; color: var(--muted); }
    .kpi .v {
      font-size: 20px;
      font-weight: 700;
      margin-top: 2px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .timeline {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }
    .stage {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      background: #fafafa;
    }
    .stage.done { background: #ecfdf3; border-color: #75d29a; }
    .stage.current { background: #ecfeff; border-color: #99f6e4; }
    pre.log {
      background: #0b1220;
      color: #d7e7ff;
      border-radius: 10px;
      padding: 10px;
      margin: 0;
      font-size: 12px;
      max-height: min(56vh, 560px);
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .checks {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    .check-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
    }
    .check-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .check-head .name { font-weight: 700; }
    .kv {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .kv th, .kv td {
      border: 1px solid #eceef1;
      padding: 5px 6px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .kv th { width: 35%; background: #f8fafc; color: #344054; }
    .annot-row { margin-bottom: 7px; }
    .annot-row:last-child { margin-bottom: 0; }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      display: flex;
      flex-direction: column;
    }
    .tile img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: var(--evidence-img-max-h);
      border: 1px solid #d1d5db;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .tile .p {
      font-size: 11px;
      color: var(--muted);
      margin-top: 4px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .raw-gallery {
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    }
    .raw-tile {
      gap: 6px;
    }
    .raw-tile.excluded {
      border-color: #fca5a5;
      background: #fff7f7;
    }
    .gallery-modal-item.excluded {
      border-color: #fca5a5;
      background: #fff7f7;
    }
    .raw-image-button {
      border: 0;
      background: transparent;
      padding: 0;
      cursor: zoom-in;
      min-height: 120px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .raw-exclude-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: #334155;
      line-height: 1.2;
    }
    .raw-exclude-toggle input {
      width: auto;
      margin: 0;
    }
    .evidence-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
      margin-top: 10px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    #predictionSummary { display: none; }
    .analysis-layout,
    .evidence-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.72fr);
      gap: 12px;
      align-items: start;
      margin-top: 8px;
    }
    .analysis-main,
    .analysis-side,
    .evidence-main,
    .evidence-side {
      min-width: 0;
    }
    .analysis-side,
    .evidence-side {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    #evidenceRectSideHost {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }
    .analysis-summary {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .analysis-review-list,
    .analysis-model-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 6px;
    }
    .analysis-review-item,
    .analysis-model-item {
      border: 1px solid #dbe4ef;
      border-radius: 8px;
      background: #fbfdff;
      padding: 8px;
      font-size: 12px;
      color: #334155;
    }
    .analysis-review-item .head,
    .analysis-model-item .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 4px;
    }
    .analysis-review-item .name,
    .analysis-model-item .name {
      font-weight: 700;
      color: #0f172a;
    }
    .analysis-model-item .path {
      font-size: 11px;
      color: #64748b;
      overflow-wrap: anywhere;
      word-break: break-word;
      margin-top: 2px;
    }
    .analysis-model-item .kvline {
      font-size: 12px;
      color: #334155;
      margin-top: 2px;
    }
    .summary-item {
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #f8fafc;
      padding: 8px;
      padding-bottom: 30px;
      border-left-width: 4px;
      border-left-color: #cbd5e1;
      position: relative;
      min-height: 90px;
    }
    .summary-item.sum-vendor { border-left-color: var(--vendor); }
    .summary-item.sum-probe { border-left-color: var(--probe); }
    .summary-item.sum-rect { border-left-color: var(--rect); }
    .summary-item.sum-rotation { border-left-color: var(--rotation); }
    .summary-item.sum-data { border-left-color: var(--data); }
    .summary-item.sum-status { border-left-color: #a16207; }
    .summary-item .k {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 3px;
    }
    .summary-item .v {
      font-size: 13px;
      font-weight: 600;
      word-break: break-word;
    }
    .summary-item.has-conf .v { padding-right: 96px; }
    .summary-item .conf-chip {
      position: absolute;
      right: 8px;
      bottom: 8px;
      font-size: 11px;
      font-weight: 500;
      color: #64748b;
      border: 1px solid #d5deea;
      background: #eef2f7;
      border-radius: 999px;
      padding: 2px 8px;
      letter-spacing: 0.2px;
    }
    .summary-item .conf-chip.empty {
      opacity: 0.4;
    }
    .evidence-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    details.toggle-block {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      margin-top: 10px;
      overflow: hidden;
    }
    details.toggle-block > summary {
      list-style: none;
      cursor: pointer;
      padding: 10px;
      background: #f8fafc;
      font-size: 13px;
      font-weight: 700;
      border-bottom: 1px solid #e4e7ec;
    }
    details.toggle-block > summary::-webkit-details-marker { display: none; }
    .toggle-content { padding: 8px; }
    .dup-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 6px;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 8px;
    }
    .pair-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    @media (max-width: 900px) {
      .pair-grid { grid-template-columns: 1fr; }
    }
    .pair-item {
      border: 1px solid #e4e7ec;
      border-radius: 8px;
      padding: 6px;
      background: #fff;
      display: flex;
      flex-direction: column;
    }
    .pair-item .label {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .pair-item img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: var(--evidence-img-max-h);
      border: 1px solid #d1d5db;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .pair-item .path {
      font-size: 10px;
      color: var(--muted);
      margin-top: 4px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .rect-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 10px;
    }
    .rect-focus-pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 8px;
    }
    @media (max-width: 1000px) {
      .rect-focus-pair { grid-template-columns: 1fr; }
    }
    .rect-focus-card {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .rect-focus-card img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: min(58vh, 620px);
      border: 1px solid #d1d5db;
      border-radius: 8px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .rect-focus-title {
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
    }
    .rect-focus-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .rect-focus-meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .rect-focus-meta .meta-item {
      border: 1px solid #dbe4ef;
      border-radius: 8px;
      background: #fbfdff;
      padding: 8px;
      font-size: 12px;
      color: #334155;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .rect-focus-meta .meta-item b {
      display: block;
      font-size: 11px;
      color: #64748b;
      margin-bottom: 2px;
      font-weight: 700;
    }
    .rect-segment-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 10px;
      margin-top: 8px;
      align-items: start;
    }
    .segment-main-card {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-width: 0;
    }
    .segment-main-card img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: min(68vh, 760px);
      border: 1px solid #d1d5db;
      border-radius: 8px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .segment-list-wrapper {
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .segment-list-wrapper.rect-side-ranking {
      margin-top: 0;
    }
    .segment-list-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .segment-scroll-controls {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-wrap: nowrap;
    }
    .segment-scroll-btn {
      min-width: 34px;
      height: 30px;
      padding: 0 8px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #0f172a;
      font-weight: 700;
      cursor: pointer;
    }
    .segment-scroll-btn:hover {
      background: #eff6ff;
      border-color: #93c5fd;
    }
    .segment-scroll-btn:focus-visible {
      outline: 2px solid #60a5fa;
      outline-offset: 1px;
    }
    .segment-list {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #f8fafc;
      padding: 8px;
      max-height: min(42vh, 420px);
      overflow: auto;
      scroll-behavior: smooth;
    }
    .segment-list:focus {
      outline: none;
      box-shadow: inset 0 0 0 2px #93c5fd;
    }
    .segment-row {
      border-bottom: 1px solid #eef2f7;
      padding: 6px 2px;
      font-size: 12px;
      color: #334155;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .segment-row:last-child { border-bottom: none; }
    .segment-row .name {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .segment-row .len {
      font-size: 11px;
      color: #0f766e;
      border: 1px solid #99f6e4;
      background: #ecfeff;
      border-radius: 999px;
      padding: 2px 8px;
      font-weight: 700;
      white-space: nowrap;
    }
    .segment-row.active {
      background: #f0fdf4;
      border-radius: 8px;
      border-bottom-color: transparent;
      padding: 6px 8px;
      margin: 2px 0;
    }
    .segment-row button {
      border: none;
      background: transparent;
      text-align: left;
      padding: 0;
      margin: 0;
      cursor: pointer;
      color: inherit;
      font: inherit;
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
    }
    .segment-row button:hover .name {
      text-decoration: underline;
    }
    .rect-item-card {
      border: 1px solid #e4e7ec;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .rect-item-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      flex-wrap: wrap;
    }
    .rect-item-rank {
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
    }
    .rect-item-score {
      font-size: 11px;
      color: #7f1d1d;
      background: #fee2e2;
      border: 1px solid #fca5a5;
      border-radius: 999px;
      padding: 2px 8px;
      font-weight: 600;
      white-space: nowrap;
    }
    .rect-item-img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: var(--evidence-img-max-h);
      border: 1px solid #d1d5db;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .rect-item-meta {
      display: grid;
      grid-template-columns: 1fr;
      gap: 4px;
      font-size: 11px;
      color: #334155;
      min-width: 0;
    }
    .rect-item-meta .path {
      font-size: 10px;
      color: var(--muted);
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .sugiu-folders {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    @media (max-width: 900px) {
      .sugiu-folders { grid-template-columns: 1fr; }
    }
    .sugiu-folder {
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      background: #fbfdff;
      padding: 8px;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sugiu-folder.sugiu-folder-su {
      border-color: #b9d7ff;
      background: #f6fbff;
      box-shadow: inset 4px 0 0 #2563eb;
    }
    .sugiu-folder.sugiu-folder-giu {
      border-color: #badbcc;
      background: #f7fff9;
      box-shadow: inset 4px 0 0 #0f766e;
    }
    .sugiu-folder-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    .sugiu-folder-title {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      margin: 0;
    }
    .sugiu-folder-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 8px;
    }
    .sugiu-folder-card {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .sugiu-folder-card img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: 180px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      background: #fff;
    }
    .sugiu-folder-card .meta {
      font-size: 10px;
      color: #64748b;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .lr-marker-visual-grid {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(110px, 0.8fr);
      gap: 8px;
      align-items: start;
    }
    .lr-marker-image-wrap {
      position: relative;
      display: inline-block;
      max-width: 100%;
      margin: 0 auto;
    }
    .lr-marker-image-wrap.manual-marker-mode {
      cursor: crosshair;
    }
    .lr-marker-image-wrap img {
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: 220px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      display: block;
      object-fit: contain;
      background: #fff;
    }
    .lr-marker-box {
      position: absolute;
      left: 0;
      top: 0;
      width: 0;
      height: 0;
      border: 2px solid #dc2626;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.9), 0 0 10px rgba(220, 38, 38, 0.45);
      pointer-events: none;
      z-index: 8;
    }
    .lr-marker-manual-box {
      position: absolute;
      left: 0;
      top: 0;
      width: 0;
      height: 0;
      border: 3px solid #16a34a;
      background: rgba(22, 163, 74, 0.08);
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.95), 0 0 10px rgba(22, 163, 74, 0.35);
      pointer-events: none;
      z-index: 10;
    }
    .lr-marker-envelope-box {
      position: absolute;
      left: 0;
      top: 0;
      width: 0;
      height: 0;
      border: 2px solid #2563eb;
      background: rgba(37, 99, 235, 0.06);
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.8);
      pointer-events: none;
      z-index: 4;
    }
    .lr-marker-envelope-box b {
      position: absolute;
      left: -2px;
      top: -20px;
      border-radius: 5px 5px 0 0;
      padding: 2px 6px;
      font-size: 11px;
      line-height: 1.1;
      color: #fff;
      background: #2563eb;
      white-space: nowrap;
    }
    .lr-marker-envelope-box.current {
      border-width: 3px;
      background: rgba(37, 99, 235, 0.10);
      z-index: 5;
    }
    .lr-marker-envelope-box.lr-env-nf {
      border-color: #2563eb;
      background: rgba(37, 99, 235, 0.06);
    }
    .lr-marker-envelope-box.lr-env-nf b { background: #2563eb; }
    .lr-marker-envelope-box.lr-env-lr {
      border-color: #f59e0b;
      background: rgba(245, 158, 11, 0.08);
    }
    .lr-marker-envelope-box.lr-env-lr b { background: #b45309; }
    .lr-marker-envelope-box.lr-env-ud {
      border-color: #0f766e;
      background: rgba(15, 118, 110, 0.08);
    }
    .lr-marker-envelope-box.lr-env-ud b { background: #0f766e; }
    .lr-marker-envelope-box.lr-env-lrud {
      border-color: #7c3aed;
      background: rgba(124, 58, 237, 0.08);
    }
    .lr-marker-envelope-box.lr-env-lrud b { background: #6d28d9; }
    .lr-marker-box::after {
      content: "marker";
      position: absolute;
      right: -2px;
      bottom: -20px;
      border-radius: 0 0 5px 5px;
      padding: 2px 6px;
      font-size: 11px;
      line-height: 1.1;
      color: #fff;
      background: #dc2626;
      white-space: nowrap;
    }
    .lr-marker-manual-box::after {
      content: "manuale";
      position: absolute;
      right: -2px;
      bottom: -20px;
      border-radius: 0 0 5px 5px;
      padding: 2px 6px;
      font-size: 11px;
      line-height: 1.1;
      color: #fff;
      background: #16a34a;
      white-space: nowrap;
    }
    .lr-template-preview {
      min-width: 0;
      display: grid;
      gap: 4px;
    }
    .lr-template-preview img {
      width: auto;
      max-width: 100%;
      max-height: 96px;
      image-rendering: pixelated;
      object-fit: contain;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: #fff;
      padding: 4px;
    }
    .lr-template-preview .label {
      font-size: 10px;
      color: #334155;
      font-weight: 700;
    }
    .orientation-hero {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: start;
      margin-top: 10px;
      padding: 10px 0 12px;
      border-bottom: 1px solid #e2e8f0;
    }
    .orientation-hero-visual {
      min-width: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      background: #f8fafc;
      border: 1px solid #d8e2ee;
      border-radius: 8px;
      padding: 8px;
    }
    .orientation-hero-visual .lr-marker-image-wrap {
      display: block;
      width: 100%;
      max-width: 1040px;
    }
    .orientation-hero-visual .lr-marker-image-wrap img {
      width: 100%;
      max-height: 560px;
      object-fit: contain;
    }
    .orientation-hero-visual .lr-marker-envelope-box {
      border-width: 3px;
      background: rgba(37, 99, 235, 0.04);
      box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.9), 0 0 12px rgba(15, 23, 42, 0.18);
    }
    .orientation-hero-panel {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .orientation-hero-title {
      font-size: 14px;
      font-weight: 850;
      color: #0f172a;
      line-height: 1.25;
    }
    .orientation-box-legend {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 6px;
    }
    .orientation-box-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #fff;
      padding: 7px 8px;
    }
    .orientation-box-chip {
      min-width: 48px;
      border-radius: 7px;
      color: #fff;
      font-size: 12px;
      font-weight: 850;
      text-align: center;
      padding: 4px 7px;
      line-height: 1;
    }
    .orientation-box-chip.nf { background: #2563eb; }
    .orientation-box-chip.lr { background: #b45309; }
    .orientation-box-chip.ud { background: #0f766e; }
    .orientation-box-chip.lrud { background: #6d28d9; }
    .orientation-box-row .meta {
      font-size: 11px;
      color: #475569;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    @media (max-width: 980px) {
      .orientation-hero { grid-template-columns: 1fr; }
      .orientation-hero-visual .lr-marker-image-wrap img { max-height: 420px; }
    }
    .lr-orientation-groups {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .lr-orientation-card {
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .lr-orientation-card .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .lr-orientation-card .title {
      font-size: 13px;
      font-weight: 800;
      color: #0f172a;
    }
    .lr-orientation-card .box {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
      color: #7f1d1d;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 6px;
      padding: 5px 6px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .lr-orientation-card .meta {
      font-size: 11px;
      color: #475569;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .orientation-decision-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .orientation-decision-card {
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .orientation-decision-card b {
      display: block;
      font-size: 12px;
      color: #0f172a;
      margin-bottom: 4px;
    }
    .orientation-decision-card .meta {
      font-size: 11px;
      color: #475569;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .orientation-mini-table {
      display: grid;
      grid-template-columns: minmax(62px, 0.7fr) repeat(4, minmax(48px, 1fr));
      gap: 1px;
      margin-top: 8px;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      overflow: hidden;
      background: #d9e2ec;
      font-size: 11px;
    }
    .orientation-mini-table > div {
      background: #fff;
      padding: 5px 6px;
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .orientation-mini-table .head {
      font-weight: 800;
      color: #0f172a;
      background: #f8fafc;
    }
    .orientation-summary-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .orientation-summary-strip .tag {
      font-size: 11px;
    }
    .lr-low-review-grid {
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    }
    .lr-low-review-card {
      border-color: #f6c6a8;
      background: #fffaf5;
    }
    .lr-low-review-card .lr-marker-image-wrap img {
      max-height: 260px;
    }
    .lr-low-review-controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 4px;
    }
    .lr-low-review-controls .wide {
      grid-column: 1 / -1;
    }
    .lr-low-review-controls label {
      margin-bottom: 2px;
      color: #7c2d12;
    }
    .lr-low-review-controls select,
    .lr-low-review-controls input,
    .lr-low-review-controls textarea {
      font-size: 12px;
      padding: 6px 7px;
      border-radius: 7px;
    }
    .lr-low-review-controls textarea {
      min-height: 56px;
    }
    @media (max-width: 700px) {
      .lr-low-review-controls { grid-template-columns: 1fr; }
      .lr-low-review-controls .wide { grid-column: auto; }
    }
    .sugiu-folder-path {
      font-size: 11px;
      color: #64748b;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .detail-list {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      .detail-list { grid-template-columns: 1fr; }
      .lr-marker-visual-grid { grid-template-columns: 1fr; }
    }
    .detail-list div {
      border: 1px solid #e4e7ec;
      border-radius: 8px;
      padding: 6px;
      background: #f8fafc;
    }
    .template-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.85fr);
      gap: 10px;
      margin-top: 8px;
      align-items: start;
    }
    .template-main-stack {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      min-width: 0;
      align-content: start;
    }
    .template-crop-card {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
    }
    .template-crop-title {
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 6px;
      color: #0f172a;
    }
    .template-crop-frame {
      border: 1px solid #dbe3ee;
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
      min-height: 260px;
      display: grid;
      place-items: center;
    }
    .template-crop-frame img {
      display: block;
      width: 100%;
      max-height: min(72vh, 760px);
      object-fit: contain;
      background: #fff;
    }
    .template-dual-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 6px;
    }
    .template-dual-item {
      border: 1px solid #dbe3ee;
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      min-width: 0;
      overflow: hidden;
    }
    .template-dual-item .label {
      font-size: 11px;
      font-weight: 700;
      color: #475569;
      margin-bottom: 4px;
    }
    .template-dual-item img {
      display: block;
      width: 100%;
      max-height: min(34vh, 340px);
      object-fit: contain;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #fff;
    }
    .template-overlay-large {
      margin-top: 6px;
    }
    .template-overlay-large img {
      display: block;
      width: 100%;
      max-height: min(62vh, 760px);
      object-fit: contain;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #fff;
    }
    .template-crop-row {
      grid-template-columns: 1fr;
    }
    .template-dual-item .path {
      display: block;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }
    .template-note {
      margin-top: 8px;
      padding: 8px 10px;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      background: #f8fafc;
      font-size: 12px;
      color: #334155;
      line-height: 1.35;
    }
    .template-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
      align-items: center;
    }
    .template-legend .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #d5dbe5;
      border-radius: 999px;
      background: #fff;
      color: #334155;
      font-size: 11px;
      padding: 2px 8px;
      white-space: nowrap;
    }
    .template-legend .swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.25);
      display: inline-block;
    }
    .template-legend .swatch.azzurro { background: #38bdf8; }
    .template-legend .swatch.giallo { background: #facc15; }
    @media (max-width: 1100px) {
      .template-dual-grid { grid-template-columns: 1fr; }
    }
    .template-side-stack {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      align-content: start;
      min-width: 0;
    }
    .template-mini-card {
      border: 1px solid #d5dbe5;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }
    .template-mini-card img {
      display: block;
      width: 100%;
      max-height: min(28vh, 260px);
      object-fit: contain;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #fff;
    }
    .template-mini-card .path {
      display: block;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }
    .help-tip {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      margin-left: 6px;
      border-radius: 999px;
      border: 1px solid #94a3b8;
      color: #334155;
      background: #f8fafc;
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
      cursor: help;
      vertical-align: middle;
      user-select: none;
    }
    .help-tip:focus-visible {
      outline: 2px solid #60a5fa;
      outline-offset: 1px;
    }
    .help-tip-bubble {
      position: absolute;
      left: -6px;
      top: calc(100% + 8px);
      z-index: 40;
      width: min(460px, 82vw);
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background: #0f172a;
      color: #e2e8f0;
      box-shadow: 0 12px 24px rgba(2, 6, 23, 0.35);
      font-size: 12px;
      line-height: 1.4;
      display: none;
      text-align: left;
      white-space: normal;
    }
    .help-tip-bubble::before {
      content: "";
      position: absolute;
      top: -6px;
      left: 12px;
      width: 10px;
      height: 10px;
      background: #0f172a;
      border-left: 1px solid #cbd5e1;
      border-top: 1px solid #cbd5e1;
      transform: rotate(45deg);
    }
    .help-tip:hover .help-tip-bubble,
    .help-tip:focus .help-tip-bubble,
    .help-tip:focus-within .help-tip-bubble {
      display: block;
    }
    .help-tip-title {
      display: block;
      margin-bottom: 6px;
      font-weight: 700;
      color: #f8fafc;
    }
    .help-tip-list {
      margin: 0;
      padding-left: 16px;
    }
    .help-tip-list li + li {
      margin-top: 4px;
    }
    @media (max-width: 1080px) {
      .template-layout { grid-template-columns: 1fr; }
    }
    .fss-meta {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
      margin-bottom: 8px;
    }
    @media (max-width: 900px) {
      .fss-meta { grid-template-columns: 1fr; }
    }
    .fss-legend {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
      color: #64748b;
    }
    .fss-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #d5deea;
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8fafc;
    }
    .fss-chip::before {
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
      border: 1px solid #cbd5e1;
      background: #f3f6fb;
    }
    .fss-chip.filled::before {
      background: #111827;
      border-color: #111827;
    }
    .fss-viewer {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #fbfdff;
      max-height: min(64vh, 760px);
      overflow: auto;
      padding: 8px;
      font-family: "Menlo", "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.4;
    }
    .fss-line {
      display: grid;
      grid-template-columns: 46px minmax(0, 1fr);
      gap: 8px;
      align-items: baseline;
      padding: 2px 4px;
      border-radius: 6px;
      cursor: help;
    }
    .fss-line:hover {
      background: #f1f5f9;
    }
    .fss-ln {
      color: #94a3b8;
      font-weight: 700;
      text-align: right;
      user-select: none;
    }
    .fss-line.fss-template .fss-content {
      color: #9aa4b2;
      font-weight: 400;
    }
    .fss-line.fss-filled .fss-content {
      color: #111827;
      font-weight: 700;
    }
    .fss-content {
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .fss-tooltip {
      position: fixed;
      z-index: 12000;
      max-width: min(520px, 72vw);
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.35;
      box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
      pointer-events: none;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 80ms ease, transform 80ms ease;
    }
    .fss-tooltip.visible {
      opacity: 1;
      transform: translateY(0);
    }
    .fss-reference {
      margin-top: 14px;
      border-top: 1px solid #d9e2ec;
      padding-top: 12px;
    }
    .fss-reference summary {
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #0f172a;
      font-size: 15px;
      font-weight: 700;
    }
    .fss-reference summary::-webkit-details-marker {
      display: none;
    }
    .fss-reference summary::before {
      content: "+";
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      border: 1px solid #b9c7d8;
      border-radius: 4px;
      color: #334155;
      background: #f8fafc;
      flex: 0 0 auto;
      font-size: 14px;
      line-height: 1;
    }
    .fss-reference[open] summary::before {
      content: "-";
    }
    .fss-reference-body {
      margin-top: 10px;
      color: #334155;
      font-size: 13px;
      line-height: 1.5;
    }
    .fss-reference-body p {
      margin: 0 0 8px;
    }
    .fss-reference-body code {
      background: #eef4fb;
      border: 1px solid #d7e3f1;
      border-radius: 4px;
      color: #0f172a;
      padding: 1px 4px;
    }
    .fss-doc-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 10px;
    }
    @media (max-width: 960px) {
      .fss-doc-grid { grid-template-columns: 1fr; }
    }
    .fss-doc-block h3 {
      margin: 0 0 6px;
      color: #0f172a;
      font-size: 13px;
      letter-spacing: 0;
    }
    .fss-doc-list {
      margin: 0;
      padding-left: 18px;
    }
    .fss-doc-list li + li {
      margin-top: 4px;
    }
    .fss-line-guide-wrap {
      margin-top: 12px;
      overflow: auto;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #fff;
    }
    .fss-line-guide {
      width: 100%;
      min-width: 920px;
      border-collapse: collapse;
      table-layout: fixed;
    }
    .fss-line-guide th,
    .fss-line-guide td {
      border-bottom: 1px solid #e2e8f0;
      padding: 8px;
      vertical-align: top;
      text-align: left;
      overflow-wrap: anywhere;
    }
    .fss-line-guide th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f5f8fc;
      color: #334155;
      font-size: 11px;
      text-transform: uppercase;
    }
    .fss-line-guide tr:last-child td {
      border-bottom: 0;
    }
    .fss-line-guide .line-col {
      width: 64px;
      color: #475569;
      font-weight: 700;
      white-space: nowrap;
    }
    .fss-line-guide .field-col {
      width: 190px;
      color: #0f172a;
      font-weight: 700;
    }
    .fss-note {
      margin-top: 10px;
      border-left: 3px solid #0f766e;
      padding: 7px 0 7px 10px;
      color: #1f2937;
      background: #f0fdfa;
    }
    .split-mock {
      display: inline-flex;
      margin-top: 6px;
      border: 1px dashed #cbd5e1;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 600;
      color: #64748b;
      background: #f8fafc;
      opacity: 0.45;
      pointer-events: none;
      user-select: none;
    }
    .rect-routing-alert {
      border: 1px solid #d8e1ec;
      border-left-width: 4px;
      border-radius: 10px;
      background: #f8fafc;
      padding: 8px 10px;
      margin-top: 8px;
    }
    .rect-routing-alert.warn {
      border-color: #f0c36b;
      border-left-color: #d97706;
      background: #fffbeb;
    }
    .rect-routing-alert.ok {
      border-color: #86efac;
      border-left-color: #16a34a;
      background: #f0fdf4;
    }
    .rect-routing-alert.info {
      border-color: #93c5fd;
      border-left-color: #2563eb;
      background: #eff6ff;
    }
    .rect-routing-title {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 4px;
    }
    .rect-routing-kv {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 6px;
      margin-top: 6px;
    }
    .rect-routing-kv .kv {
      border: 1px solid #dbe4ef;
      border-radius: 8px;
      background: #ffffff;
      padding: 6px 8px;
      font-size: 12px;
      min-width: 0;
    }
    .rect-routing-kv .k {
      color: #64748b;
      font-size: 11px;
      margin-bottom: 2px;
    }
    .rect-routing-kv .v {
      color: #0f172a;
      font-weight: 600;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .rect-routing-actions {
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .rect-compare-state {
      margin-top: 8px;
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      padding: 8px;
      background: #f8fafc;
    }
    .rect-compare-state.running {
      border-color: #93c5fd;
      background: #eff6ff;
    }
    .rect-compare-state.error {
      border-color: #fca5a5;
      background: #fff1f2;
    }
    .rect-compare-grid {
      margin-top: 8px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 8px;
    }
    .rect-compare-model {
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      background: #ffffff;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .rect-compare-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .rect-compare-title {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
    }
    .rect-compare-metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    .rect-compare-metrics .m {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #f8fafc;
      padding: 5px 6px;
      min-width: 0;
    }
    .rect-compare-metrics .mk {
      font-size: 10px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.2px;
    }
    .rect-compare-metrics .mv {
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .winner-chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid #86efac;
      border-radius: 999px;
      padding: 2px 10px;
      background: #f0fdf4;
      color: #166534;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .tag {
      display: inline-block;
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      vertical-align: middle;
      margin-left: 6px;
    }
    .panel h2 .tag {
      transform: translateY(-1px);
    }
    .tag-monitor { color: #075985; background: #e0f2fe; border-color: #7dd3fc; }
    .tag-analysis { color: #166534; background: #dcfce7; border-color: #86efac; }
    .tag-evidence { color: #92400e; background: #fef3c7; border-color: #fcd34d; }
    .tag-vendor { color: #134e4a; background: #ccfbf1; border-color: #5eead4; }
    .tag-probe { color: #312e81; background: #e0e7ff; border-color: #a5b4fc; }
    .tag-rect { color: #7f1d1d; background: #fee2e2; border-color: #fca5a5; }
    .tag-rotation { color: #9a3412; background: #ffedd5; border-color: #fdba74; }
    .tag-data { color: #164e63; background: #cffafe; border-color: #67e8f9; }
    .tag-ok { color: #166534; background: #dcfce7; border-color: #86efac; }
    .tag-warn { color: #92400e; background: #fef3c7; border-color: #fcd34d; }
    .tag-model { color: #334155; background: #e2e8f0; border-color: #cbd5e1; }
    .small { font-size: 12px; color: var(--muted); }
    .path,
    .small,
    #runInfo,
    #checksStatus,
    #evidenceStatus,
    #startStatus {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .modal {
      position: fixed;
      inset: 0;
      background: rgba(16, 24, 40, 0.48);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      padding: 16px;
    }
    .modal.hidden { display: none; }
    .modal-card {
      width: min(960px, 100%);
      max-height: 86vh;
      overflow: hidden;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 8px;
      padding: 12px;
    }
    .modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    .modal-head h3 {
      margin: 0;
      font-size: 16px;
    }
    .browse-top {
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 8px;
      align-items: center;
    }
    @media (max-width: 900px) {
      .browse-top { grid-template-columns: 1fr 1fr; }
    }
    .dir-list {
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: auto;
      background: #fff;
      min-height: 220px;
      max-height: 52vh;
    }
    .dir-item {
      padding: 10px;
      border-bottom: 1px solid #eef1f4;
      cursor: pointer;
      font-size: 13px;
    }
    .dir-item:last-child { border-bottom: none; }
    .dir-item:hover { background: #f8fafc; }
    .modal-foot {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .compare-card {
      width: min(1200px, 100%);
      max-height: 92vh;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      overflow: auto;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    .compare-pane {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #f8fafc;
      padding: 8px;
    }
    .compare-pane .label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .compare-pane img {
      width: 100%;
      height: auto;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      display: block;
      background: #fff;
      max-height: 72vh;
      object-fit: contain;
    }
    .compare-path {
      font-size: 11px;
      color: var(--muted);
      margin-top: 6px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .rect-gallery-card {
      width: min(1700px, 100%);
      height: min(94vh, 980px);
      max-height: 94vh;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto auto;
      gap: 10px;
      overflow: hidden;
    }
    .rect-gallery-card.mode-rect-net {
      border-color: #f59e0b;
      box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.28) inset;
    }
    .rect-gallery-card.mode-rect-segment {
      border-color: #dc2626;
      box-shadow: 0 0 0 1px rgba(220, 38, 38, 0.28) inset;
    }
    .rect-gallery-layout {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.38fr);
      gap: 10px;
      align-items: stretch;
      overflow: hidden;
    }
    .rect-gallery-main {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #f8fafc;
      padding: 8px;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 6px;
      overflow: hidden;
    }
    .rect-gallery-main .label {
      font-size: 12px;
      color: var(--muted);
    }
    .rect-gallery-image-wrap {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 0;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .rect-gallery-main img {
      width: 100%;
      max-width: 100%;
      height: 100%;
      max-height: 100%;
      object-fit: contain;
      border: none;
      border-radius: 0;
      background: #fff;
      display: block;
      margin: 0;
    }
    .rect-gallery-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .rect-gallery-side {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #ffffff;
      padding: 8px;
      min-height: 0;
      overflow: auto;
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      align-content: start;
    }
    .rect-gallery-kv {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #f8fafc;
      padding: 6px 8px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 2px;
      min-width: 0;
    }
    .rect-gallery-kv .k {
      font-size: 11px;
      color: #64748b;
    }
    .rect-gallery-kv .v {
      font-size: 12px;
      color: #0f172a;
      font-weight: 600;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .rect-gallery-nav {
      margin-right: auto;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .rect-gallery-thumbs {
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      background: #fbfdff;
      padding: 6px;
      display: flex;
      gap: 6px;
      overflow-x: auto;
      overflow-y: hidden;
      min-height: 82px;
    }
    .rect-thumb-btn {
      border: 2px solid transparent;
      border-radius: 8px;
      background: #fff;
      padding: 2px;
      cursor: pointer;
      min-width: 118px;
      max-width: 118px;
      height: 82px;
      flex: 0 0 auto;
    }
    .rect-thumb-btn.active {
      border-color: #0f766e;
      background: #ecfeff;
    }
    .rect-thumb-btn img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
      display: block;
      background: #fff;
    }
    .rect-gallery-counter {
      font-size: 12px;
      color: #475569;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 4px 10px;
      align-self: center;
      background: #f8fafc;
    }
    .rect-mode-chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      background: #f8fafc;
      color: #334155;
    }
    .rect-mode-chip.net {
      border-color: #f59e0b;
      background: #fffbeb;
      color: #92400e;
    }
    .rect-mode-chip.segment {
      border-color: #dc2626;
      background: #fef2f2;
      color: #7f1d1d;
    }
    @media (max-width: 1200px) {
      .rect-gallery-layout { grid-template-columns: 1fr; }
      .rect-gallery-side { max-height: 32vh; }
    }
    @media (max-width: 1000px) {
      .compare-grid { grid-template-columns: 1fr; }
      .rect-gallery-layout { grid-template-columns: 1fr; }
      .rect-gallery-card {
        height: min(96vh, 980px);
      }
      .rect-gallery-side {
        max-height: 28vh;
      }
    }
    .gallery-modal-card {
      width: min(1480px, 100%);
      max-height: 92vh;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 8px;
      overflow: hidden;
    }
    .gallery-modal-card.lr-marker-mode {
      width: min(1760px, 100%);
      max-height: 96vh;
    }
    .lr-marker-viewer {
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 8px;
    }
    .lr-marker-viewer-main {
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 0.32fr);
      gap: 10px;
      align-items: stretch;
    }
    .lr-marker-viewer-image {
      min-height: 0;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #f8fafc;
      padding: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
    }
    .lr-marker-viewer-image .lr-marker-image-wrap img {
      max-height: 70vh;
      max-width: 100%;
    }
    .lr-marker-viewer-side {
      min-height: 0;
      overflow: auto;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      display: grid;
      align-content: start;
      gap: 8px;
    }
    .lr-marker-viewer-side .lr-template-preview img {
      max-height: 160px;
    }
    .lr-marker-viewer-kv {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #f8fafc;
      padding: 6px 8px;
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    .lr-marker-viewer-kv .k {
      font-size: 11px;
      color: #64748b;
    }
    .lr-marker-viewer-kv .v {
      font-size: 12px;
      color: #0f172a;
      font-weight: 600;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .lr-marker-viewer-thumbs {
      border: 1px solid #dbe4ef;
      border-radius: 10px;
      background: #fbfdff;
      padding: 6px;
      display: flex;
      gap: 6px;
      overflow-x: auto;
      overflow-y: hidden;
      min-height: 90px;
    }
    .lr-marker-thumb-btn {
      border: 2px solid transparent;
      border-radius: 8px;
      background: #fff;
      padding: 2px;
      cursor: pointer;
      min-width: 124px;
      max-width: 124px;
      height: 84px;
      flex: 0 0 auto;
      position: relative;
    }
    .lr-marker-thumb-btn.active {
      border-color: #dc2626;
      background: #fef2f2;
    }
    .lr-marker-thumb-btn img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
      display: block;
      background: #fff;
    }
    .lr-marker-thumb-btn span {
      position: absolute;
      right: 4px;
      top: 4px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.78);
      color: #fff;
      font-size: 10px;
      padding: 1px 5px;
    }
    .lr-marker-gallery-nav {
      margin-right: auto;
      display: none;
      gap: 8px;
      flex-wrap: wrap;
    }
    .gallery-modal-card.lr-marker-mode .lr-marker-gallery-nav {
      display: flex;
    }
    .gallery-modal-grid {
      overflow: auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 10px;
      align-content: start;
      padding-right: 2px;
    }
    .gallery-modal-grid.sugiu-split-mode {
      grid-template-columns: 1fr;
      align-items: start;
    }
    .sugiu-gallery-sections {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .sugiu-gallery-section {
      border: 1px solid #dbe4ef;
      border-radius: 8px;
      padding: 10px;
      min-width: 0;
    }
    .sugiu-gallery-section.su {
      border-color: #b9d7ff;
      background: #f6fbff;
      box-shadow: inset 5px 0 0 #2563eb;
    }
    .sugiu-gallery-section.giu {
      border-color: #badbcc;
      background: #f7fff9;
      box-shadow: inset 5px 0 0 #0f766e;
    }
    .sugiu-gallery-section.other {
      border-color: #e5e7eb;
      background: #fafafa;
      box-shadow: inset 5px 0 0 #64748b;
    }
    .sugiu-gallery-section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .sugiu-gallery-section-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 15px;
      font-weight: 850;
      color: #0f172a;
    }
    .sugiu-gallery-badge {
      border-radius: 999px;
      color: #fff;
      font-size: 11px;
      font-weight: 850;
      padding: 4px 8px;
      text-transform: uppercase;
    }
    .sugiu-gallery-section.su .sugiu-gallery-badge { background: #2563eb; }
    .sugiu-gallery-section.giu .sugiu-gallery-badge { background: #0f766e; }
    .sugiu-gallery-section.other .sugiu-gallery-badge { background: #64748b; }
    .sugiu-gallery-section-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 8px;
    }
    @media (max-width: 980px) {
      .sugiu-gallery-sections { grid-template-columns: 1fr; }
    }
    .gallery-modal-item {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .gallery-modal-item .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .gallery-modal-item .head .t {
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
    }
    .gallery-modal-item .head .m {
      font-size: 11px;
      color: #64748b;
      background: #eef2f7;
      border: 1px solid #d5deea;
      border-radius: 999px;
      padding: 2px 8px;
      white-space: nowrap;
    }
    .gallery-modal-item .img-pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .gallery-modal-item .img-pair img,
    .gallery-modal-item .img-single img {
      width: 100%;
      height: auto;
      max-height: min(28vh, 260px);
      object-fit: contain;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    .gallery-modal-item .meta {
      font-size: 11px;
      color: #334155;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    @media (max-width: 1000px) {
      .gallery-modal-item .img-pair { grid-template-columns: 1fr; }
      .gallery-modal-grid { grid-template-columns: 1fr; }
      .lr-marker-viewer-main { grid-template-columns: 1fr; }
      .lr-marker-viewer-image .lr-marker-image-wrap img { max-height: 52vh; }
      .analysis-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      body { padding: 8px; overflow: auto; }
      :root { --evidence-img-max-h: min(34vh, 280px); }
      .header { padding: 10px; }
      .panel { padding: 10px; }
      .run-list { max-height: 360px; }
      .kpi .v { font-size: 18px; }
      .app-shell { max-width: 100%; min-height: 100%; }
      .app-layout { min-height: auto; }
      .history-column,
      .workspace-shell,
      .main-column { overflow: visible; }
      .pane-stack { min-height: auto; }
      .evidence-layout { grid-template-columns: 1fr; }
      .pane-panel { height: auto; min-height: auto; overflow: visible; }
      .pane-nav { padding: 0; }
      .history-column { max-height: none; }
    }
    @media (max-width: 760px) {
      body { overflow: auto; }
      .app-shell { min-height: 100%; }
      .app-layout { min-height: auto; }
      .history-column { max-height: none; padding-left: 0; }
      .history-column .panel { max-height: none; }
    }
    /* Elesta full skin: typography, spacing, palette and components */
    :root {
      color-scheme: light;
      --bg: #f4efe7;
      --bg-strong: #efe5d6;
      --card: rgba(255, 250, 243, 0.82);
      --card-strong: #fffdf8;
      --ink: #221a12;
      --muted: #635344;
      --line: rgba(65, 48, 26, 0.16);
      --shadow-sm: 0 12px 28px rgba(54, 37, 22, 0.10);
      --shadow-md: 0 18px 55px rgba(54, 37, 22, 0.12);
      --primary: #0f766e;
      --primary-2: #0b5b55;
      --ok-bg: rgba(15, 118, 110, 0.12);
      --ok-line: rgba(15, 118, 110, 0.35);
      --warn-bg: rgba(201, 110, 55, 0.14);
      --warn-line: rgba(201, 110, 55, 0.35);
      --err-bg: rgba(184, 77, 69, 0.12);
      --err-line: rgba(184, 77, 69, 0.35);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
      --font-body: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --font-display: "Iowan Old Style", "Palatino Linotype", serif;
      --evidence-img-max-h: min(46vh, 420px);
    }
    html {
      scroll-behavior: smooth;
      background: var(--bg);
    }
    body {
      margin: 0 !important;
      padding: 0 !important;
      min-height: 100vh;
      font-family: var(--font-body) !important;
      color: var(--ink) !important;
      background:
        radial-gradient(circle at top left, rgba(201, 110, 55, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.16), transparent 30%),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 45%, #eee4d3 100%) !important;
    }
    html[data-darkreader-scheme="dark"] body {
      background-color: #1f1a14 !important;
      background-image: linear-gradient(180deg, rgba(201, 110, 55, 0.18) 0%, rgba(15, 118, 110, 0.12) 45%, rgba(15, 118, 110, 0.04) 80%) !important;
    }
    h1,
    h2,
    h3,
    .panel h2,
    .panel-head h2 {
      margin: 0;
      font-family: var(--font-display) !important;
      letter-spacing: -0.03em;
      color: var(--ink);
    }
    label,
    .muted,
    .kpi .k,
    .summary-item .k,
    .tile .p,
    .pair-item .label,
    .pair-item .path,
    .rect-item-meta .path,
    .rect-focus-meta .meta-item b,
    .models-secondary,
    .models-path,
    .analysis-model-item .path {
      color: var(--muted) !important;
    }
    .app-shell {
      max-width: none !important;
      min-height: 100vh !important;
      margin: 0 !important;
      padding: 1.5rem !important;
      display: grid !important;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 1.25rem;
      overflow: visible !important;
    }
    .header {
      background: rgba(248, 240, 230, 0.76) !important;
      border: 1px solid var(--line) !important;
      border-radius: var(--radius-lg) !important;
      box-shadow: var(--shadow-md) !important;
      backdrop-filter: blur(18px);
      padding: 1rem 1.15rem !important;
      margin: 0 !important;
    }
    .header h1 {
      font-size: 2rem !important;
      line-height: 1 !important;
      letter-spacing: -0.02em !important;
    }
    .header h1 a {
      color: var(--ink) !important;
      text-decoration: none !important;
    }
    .app-layout {
      min-width: 980px;
      gap: 1.25rem !important;
      grid-template-columns: minmax(0, 3fr) minmax(320px, 1fr) !important;
      align-items: start;
    }
    .main-column {
      gap: 1.25rem !important;
      padding-right: 0 !important;
    }
    .history-column {
      position: sticky;
      top: 1.5rem !important;
      max-height: calc(100vh - 3rem) !important;
      padding-left: 0 !important;
    }
    .history-column .panel {
      height: 100%;
    }
    .panel-core {
      border-width: 1px !important;
      border-color: var(--line) !important;
      box-shadow: var(--shadow-md) !important;
    }
    .panel,
    .workspace-shell,
    .history-column .panel,
    .monitor-left,
    .monitor-right,
    .check-card,
    .evidence-card,
    .home-models,
    .home-encoding,
    .home-encoding-box,
    .models-kpi,
    .models-table-wrap,
    .models-top-list,
    .kpi,
    .summary-item,
    .pair-item,
    .rect-focus-card,
    .segment-main-card,
    .segment-list-wrapper,
    .rect-item-card,
    .analysis-review-item,
    .analysis-model-item,
    details.toggle-block {
      background: var(--card) !important;
      border: 1px solid var(--line) !important;
      border-radius: var(--radius-lg) !important;
      box-shadow: var(--shadow-sm) !important;
      backdrop-filter: blur(16px);
    }
    .panel,
    .monitor-left,
    .monitor-right,
    .check-card,
    .evidence-card,
    .kpi,
    .summary-item {
      padding: 1rem !important;
    }
    .workspace-topbar {
      background: rgba(248, 240, 230, 0.76) !important;
      border-bottom: 1px solid var(--line) !important;
      padding: 0.85rem 1rem !important;
    }
    .folder-badge,
    .status-pill,
    .stage,
    .home-encoding-count,
    .segment-row .len,
    .rect-item-score,
    .summary-item .conf-chip {
      border-radius: 999px !important;
      border: 1px solid rgba(15, 118, 110, 0.32) !important;
      background: rgba(15, 118, 110, 0.12) !important;
      color: var(--primary-2) !important;
      font-weight: 700;
    }
    .summary-item .conf-chip {
      font-weight: 500;
      border-color: rgba(65, 48, 26, 0.20) !important;
      background: rgba(255, 253, 248, 0.92) !important;
      color: var(--muted) !important;
    }
    input,
    select,
    textarea {
      border: 1px solid rgba(65, 48, 26, 0.16) !important;
      border-radius: var(--radius-md) !important;
      background: #fffdf9 !important;
      color: var(--ink) !important;
      padding: 0.85rem 0.95rem !important;
    }
    .btn,
    .pane-btn {
      border-radius: 999px !important;
      transition: transform 0.18s ease, background 0.18s ease, opacity 0.18s ease;
      font-weight: 700 !important;
    }
    .btn:hover,
    .pane-btn:hover {
      transform: translateY(-1px);
    }
    .btn {
      border: 1px solid var(--primary) !important;
      background: var(--primary) !important;
      color: #fff !important;
    }
    .btn.secondary,
    .btn.back,
    .pane-btn {
      border: 1px solid rgba(65, 48, 26, 0.20) !important;
      background: rgba(34, 26, 18, 0.08) !important;
      color: var(--ink) !important;
    }
    .btn.danger {
      border: 1px solid rgba(184, 77, 69, 0.24) !important;
      background: rgba(184, 77, 69, 0.12) !important;
      color: #8f2d2a !important;
    }
    .pane-btn.active,
    .run-item.active,
    .stage.current {
      border-color: rgba(15, 118, 110, 0.40) !important;
      background: rgba(15, 118, 110, 0.14) !important;
      color: var(--primary-2) !important;
    }
    .run-list,
    .home-encoding-list {
      border-radius: var(--radius-md) !important;
      border-color: var(--line) !important;
      background: rgba(255, 253, 248, 0.94) !important;
    }
    .run-item,
    .home-encoding-item,
    .models-table td,
    .kv td {
      border-bottom-color: rgba(65, 48, 26, 0.10) !important;
    }
    .run-item:hover,
    .models-table tr:hover td,
    .segment-scroll-btn:hover {
      background: rgba(201, 110, 55, 0.10) !important;
    }
    .models-table th,
    .kv th,
    details.toggle-block > summary {
      background: rgba(239, 229, 214, 0.90) !important;
      color: #43301f !important;
      border-color: rgba(65, 48, 26, 0.16) !important;
    }
    .summary-item {
      border-left-color: rgba(65, 48, 26, 0.20) !important;
    }
    .summary-item.sum-vendor { border-left-color: #0f766e !important; }
    .summary-item.sum-probe { border-left-color: #6b4f2a !important; }
    .summary-item.sum-rect { border-left-color: #b84d45 !important; }
    .summary-item.sum-rotation { border-left-color: #c96e37 !important; }
    .summary-item.sum-data { border-left-color: #0b5b55 !important; }
    .summary-item.sum-status { border-left-color: #9a5a2c !important; }
    .status-ok,
    .stage.done {
      background: var(--ok-bg) !important;
      border-color: var(--ok-line) !important;
      color: var(--primary-2) !important;
    }
    .status-review {
      background: var(--warn-bg) !important;
      border-color: var(--warn-line) !important;
      color: #8b4d23 !important;
    }
    .status-error {
      background: var(--err-bg) !important;
      border-color: var(--err-line) !important;
      color: #8f2d2a !important;
    }
    .segment-row.active {
      background: rgba(15, 118, 110, 0.10) !important;
    }
    .models-metric,
    .folder-path,
    .analysis-review-item .name,
    .analysis-model-item .name {
      color: #3c2b1a !important;
    }
    pre.log {
      border: 1px solid rgba(15, 118, 110, 0.28);
      border-radius: var(--radius-md);
      box-shadow: inset 0 0 0 1px rgba(201, 110, 55, 0.18);
    }
    /* Exact Elesta color correction */
    :root {
      --bg: #f4efe7 !important;
      --bg-strong: #efe5d6 !important;
      --surface: rgba(255, 250, 243, 0.82) !important;
      --surface-strong: #fffdf8 !important;
      --line: rgba(65, 48, 26, 0.16) !important;
      --text: #221a12 !important;
      --muted: #635344 !important;
      --accent: #0f766e !important;
      --accent-strong: #0b5b55 !important;
      --accent-soft: rgba(15, 118, 110, 0.12) !important;
      --signal: #c96e37 !important;
      --danger: #b84d45 !important;
      --shadow: 0 18px 55px rgba(54, 37, 22, 0.12) !important;
      --ink: var(--text) !important;
      --card: var(--surface) !important;
      --card-strong: var(--surface-strong) !important;
      --primary: var(--accent) !important;
      --primary-2: var(--accent-strong) !important;
      --ok-bg: var(--accent-soft) !important;
      --ok-line: rgba(15, 118, 110, 0.35) !important;
      --warn-bg: rgba(201, 110, 55, 0.14) !important;
      --warn-line: rgba(201, 110, 55, 0.35) !important;
      --err-bg: rgba(184, 77, 69, 0.12) !important;
      --err-line: rgba(184, 77, 69, 0.35) !important;
    }
    body {
      color: var(--text) !important;
      background:
        radial-gradient(circle at top left, rgba(201, 110, 55, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.16), transparent 30%),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 45%, #eee4d3 100%) !important;
    }
    .header,
    .panel,
    .workspace-shell,
    .history-column .panel,
    .monitor-left,
    .monitor-right,
    .check-card,
    .evidence-card,
    .home-models,
    .home-encoding,
    .home-encoding-box,
    .models-kpi,
    .models-table-wrap,
    .models-top-list,
    .kpi,
    .summary-item,
    .pair-item,
    .rect-focus-card,
    .segment-main-card,
    .segment-list-wrapper,
    .rect-item-card,
    .analysis-review-item,
    .analysis-model-item,
    details.toggle-block {
      background: var(--surface) !important;
      border-color: var(--line) !important;
      box-shadow: var(--shadow) !important;
    }
    .home-encoding-list,
    .run-list,
    .dir-list {
      background: var(--surface-strong) !important;
      border-color: var(--line) !important;
    }
    .models-table th,
    .kv th,
    details.toggle-block > summary {
      background: var(--bg-strong) !important;
      color: var(--text) !important;
      border-color: var(--line) !important;
    }
    .models-table tr:hover td,
    .run-item:hover,
    .dir-item:hover,
    .segment-scroll-btn:hover {
      background: rgba(201, 110, 55, 0.08) !important;
    }
    .pane-btn.active,
    .run-item.active,
    .stage.current {
      background: var(--accent-soft) !important;
      border-color: rgba(15, 118, 110, 0.35) !important;
      color: var(--accent-strong) !important;
    }
    .status-ok,
    .stage.done {
      background: var(--accent-soft) !important;
      border-color: rgba(15, 118, 110, 0.35) !important;
      color: var(--accent-strong) !important;
    }
    .status-review {
      background: rgba(201, 110, 55, 0.14) !important;
      border-color: rgba(201, 110, 55, 0.35) !important;
      color: #8b4d23 !important;
    }
    .status-error {
      background: rgba(184, 77, 69, 0.12) !important;
      border-color: rgba(184, 77, 69, 0.35) !important;
      color: var(--danger) !important;
    }
    .summary-item.sum-vendor { border-left-color: var(--accent) !important; }
    .summary-item.sum-probe { border-left-color: var(--signal) !important; }
    .summary-item.sum-rect { border-left-color: var(--danger) !important; }
    .summary-item.sum-rotation { border-left-color: var(--signal) !important; }
    .summary-item.sum-data { border-left-color: var(--accent-strong) !important; }
    .summary-item.sum-status { border-left-color: var(--signal) !important; }
    .home-encoding-count,
    .segment-row .len,
    .status-pill,
    .stage {
      background: var(--accent-soft) !important;
      border-color: rgba(15, 118, 110, 0.35) !important;
      color: var(--accent-strong) !important;
    }
    .btn.secondary,
    .btn.back,
    .pane-btn {
      background: rgba(34, 26, 18, 0.08) !important;
      border-color: rgba(65, 48, 26, 0.20) !important;
      color: var(--text) !important;
    }
    .btn.danger {
      background: rgba(184, 77, 69, 0.12) !important;
      border-color: rgba(184, 77, 69, 0.25) !important;
      color: var(--danger) !important;
    }
    .dt-node .dt-title {
      color: var(--text) !important;
    }
    .dt-node .dt-meta,
    .models-secondary,
    .models-path,
    .analysis-model-item .path,
    .muted,
    .summary-item .k,
    .tile .p,
    .pair-item .label,
    .pair-item .path,
    .rect-item-meta .path,
    .rect-focus-meta .meta-item b {
      color: var(--muted) !important;
    }
    @media (max-width: 760px) {
      body {
        padding: 8px !important;
        overflow-x: hidden !important;
      }
      .header h1 {
        font-size: 1.55rem !important;
        letter-spacing: 0 !important;
        overflow-wrap: anywhere;
      }
      .app-layout,
      body.workspace-mode .app-layout {
        min-width: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        grid-template-columns: 1fr !important;
      }
      .app-shell,
      .main-column,
      .workspace-shell,
      .workspace-topbar,
      .workspace-meta,
      .pane-stack,
      .pane-panel,
      .evidence-layout,
      .evidence-main,
      .evidence-card,
      .pair-grid,
      .pair-item,
      .history-column .panel,
      .workspace-rail,
      .lr-template-preview {
        min-width: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
      }
      .history-column {
        position: static !important;
        width: 100% !important;
        max-height: none !important;
      }
      .folder-badge {
        max-width: 100% !important;
        white-space: normal !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
      }
      .folder-path,
      .lr-template-preview .meta {
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .pair-grid {
        grid-template-columns: 1fr !important;
      }
      .lr-marker-visual-grid {
        grid-template-columns: 1fr !important;
      }
      .orientation-mini-table {
        grid-template-columns: minmax(48px, 0.8fr) repeat(4, minmax(38px, 1fr));
        font-size: 10px;
      }
      .orientation-mini-table > div {
        padding: 4px;
      }
    }
    .history-column {
      height: calc(100vh - 3rem) !important;
      max-height: calc(100vh - 3rem) !important;
      min-height: 0 !important;
      overflow: hidden !important;
    }
    .history-column > .panel,
    .history-panel-setup,
    .history-panel-workspace {
      min-height: 0 !important;
      max-height: 100% !important;
      overflow: hidden !important;
      flex-direction: column !important;
    }
    body.setup-mode .history-panel-setup {
      display: flex !important;
    }
    body.workspace-mode .history-panel-workspace {
      display: flex !important;
      overflow: auto !important;
      overscroll-behavior: contain;
    }
    .history-panel-setup[hidden],
    .history-panel-workspace[hidden],
    body.setup-mode .history-panel-workspace,
    body.workspace-mode .history-panel-setup {
      display: none !important;
    }
    #runsList.run-list {
      flex: 1 1 auto !important;
      min-height: 0 !important;
      max-height: none !important;
      overflow-y: auto !important;
      overflow-x: hidden !important;
      overscroll-behavior: contain;
      scrollbar-gutter: stable;
    }
    .history-panel-workspace .workspace-rail {
      min-height: 0 !important;
    }
    @media (max-width: 760px) {
      .history-column {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
      }
      .history-column > .panel,
      .history-panel-setup,
      .history-panel-workspace {
        height: auto !important;
        max-height: none !important;
      }
      #runsList.run-list {
        max-height: min(62vh, 520px) !important;
      }
    }
  </style>
</head>
<body class="{% if initial_selected_run %}workspace-mode{% else %}setup-mode{% endif %}">
  <div class="app-shell">
    <div class="header">
      <h1><a id="homeLink" href="/">Pipeline AIESIBuilder</a></h1>
    </div>

    <div class="app-layout">
      <main class="main-column">
        <div class="panel panel-core">
          <div class="panel-head">
            <h2 style="margin:0;">Home Run <span class="tag tag-data">setup</span></h2>
            <button id="refreshRunsBtn" type="button" class="btn secondary mini">Aggiorna Lista</button>
          </div>
          <form id="startForm" method="post" action="/runs/start_form">
            <input id="excludedImagesJson" name="excluded_images_rel_json" type="hidden" value="__use_saved__">
            <div class="home-run-grid">
              <section class="run-mode-section">
                <div class="run-mode-title">
                  <h3>Singola run</h3>
                  <span class="tag tag-monitor">manuale</span>
                </div>
                <div class="input-browse compact">
                  <input id="inputFolder" name="input_folder" type="text" placeholder="/percorso/cartella_acquisizione" value="{{ initial_input|e }}">
                  <button id="browseInputBtn" type="submit" formmethod="get" formaction="/pick-folder" class="btn secondary">Sfoglia...</button>
                  <a id="browseWebLink" href="/browse" class="btn secondary" title="Fallback se il popup di sistema non si apre">Browser web</a>
                </div>
                <div class="compact-params">
                  <div>
                    <label>Batch</label>
                    <input id="batchSize" name="batch_size" type="number" min="1" value="32">
                  </div>
                  <div>
                    <label>Sample</label>
                    <input id="samplePerFolder" name="sample_per_folder" type="number" min="1" value="80">
                  </div>
                  <div>
                    <label>Rot. sample</label>
                    <input id="rotationMaxSamples" name="rotation_max_samples" type="number" min="1" value="24">
                  </div>
                  <div>
                    <label>Low conf</label>
                    <select id="lowPolicy" name="low_confidence_policy">
                      <option value="review" selected>review</option>
                      <option value="error">error</option>
                      <option value="ask_user">ask_user</option>
                    </select>
                  </div>
                  <div>
                    <label>Vendor conf</label>
                    <input id="vendorMinConf" name="vendor_min_confidence" type="number" min="0" max="1" step="0.01" value="0.50">
                  </div>
                  <div>
                    <label>Probe conf</label>
                    <input id="probeMinConf" name="probe_min_confidence" type="number" min="0" max="1" step="0.01" value="0.50">
                  </div>
                </div>
                <div class="compact-params single">
                  <div>
                    <label>LR marker template</label>
                    <select id="lrMarkerTemplatePolicy" name="lr_marker_template_policy">
                      <option value="historical_best_then_derived" selected>storico poi derivato</option>
                      <option value="historical_best">solo storico</option>
                      <option value="derived_folder">derivato cartella</option>
                    </select>
                  </div>
                  <label class="check-line"><input id="noGeneratedImages" name="no_generated_images" type="checkbox" value="1"> evidenze leggere</label>
                  <label class="check-line"><input id="noSplitSymlinks" name="no_split_symlinks" type="checkbox" value="1"> senza split symlink</label>
                </div>
                <div class="exclusion-box">
                  <div class="exclusion-head">
                    <div>
                      <b>Esclusioni immagini</b>
                      <div id="exclusionStatus" class="small">Nessuna cartella caricata.</div>
                    </div>
                    <div class="mini-actions">
                      <button id="loadExclusionsBtn" type="button" class="btn secondary mini">Carica</button>
                      <button id="saveExclusionsBtn" type="button" class="btn secondary mini">Salva</button>
                      <button id="clearExclusionsBtn" type="button" class="btn secondary mini">Svuota</button>
                    </div>
                  </div>
                  <div id="imageExclusionList" class="image-exclusion-list"></div>
                </div>
                <div class="actions compact-actions">
                  <button id="startRunBtn" type="submit" class="btn">Avvia singola run</button>
                </div>
              </section>

              <section class="run-mode-section bulk-mode">
                <div class="run-mode-title">
                  <h3>Runna tutte le cartelle</h3>
                  <span class="tag tag-data">batch</span>
                </div>
                <label>Root acquisizioni</label>
                <input id="bulkAcquisitionRoot" type="text" value="/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION">
                <div class="actions compact-actions">
                  <button id="startBulkBtn" type="button" class="btn secondary">Reset + run tutte</button>
                  <button id="refreshBulkBtn" type="button" class="btn secondary">Stato batch</button>
                </div>
                <div id="bulkStatus" class="small">Batch SSD non avviato.</div>
              </section>
            </div>

            <section class="rerun-box">
              <div class="run-mode-title">
                <h3>Rifai solo i punti che ti servono</h3>
                <span class="tag tag-evidence">rerun</span>
              </div>
              <div class="rerun-grid">
                <div>
                  <label>Preset</label>
                  <select id="rerunPreset">
                    <option value="selected">solo checkbox</option>
                    <option value="orientation">orientamento</option>
                    <option value="rect">rettangolo</option>
                    <option value="all_evidence">tutte evidenze</option>
                  </select>
                </div>
                <div>
                  <label>Rect margin %</label>
                  <input id="rerunRectMarginPct" type="number" min="0" max="30" step="0.5" value="5">
                </div>
                <div>
                  <label>Bright thr</label>
                  <input id="rerunRectBrightThr" type="number" min="0" max="255" step="1" value="70">
                </div>
                <label class="check-line"><input id="rerunForceEvidence" type="checkbox" checked> ricalcola cache</label>
                <button id="rerunSelectedBtn" type="button" class="btn secondary">Esegui rerun</button>
              </div>
              <div class="rerun-step-grid" id="rerunStepGrid">
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="full_pipeline">
                  <span><span class="name">Run completa nuova</span><span class="hint">Rilancia tutta la pipeline sulla cartella input.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="rect_red">
                  <span><span class="name">Rect rosso linea 11</span><span class="hint">Ricalcola il rettangolo eco da segmenti rossi.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="rect_per_image">
                  <span><span class="name">Rect per immagine</span><span class="hint">Riesegue l'audit box frame-by-frame.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="rect_compare">
                  <span><span class="name">Confronto reti rect</span><span class="hint">Globale vs specifica vendor.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="su_giu">
                  <span><span class="name">SU/GIU</span><span class="hint">Rigenera evidenza e cartelle split.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="lr_marker">
                  <span><span class="name">LR marker + linea 16</span><span class="hint">Ricalcola quadranti, gruppi NF/LR/UD/LRUD e line16.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="lt">
                  <span><span class="name">L/T</span><span class="hint">Rigenera evidenza e cartelle split L/T.</span></span>
                </label>
                <label class="rerun-step">
                  <input type="checkbox" data-rerun-target="line13">
                  <span><span class="name">Linea 13 template</span><span class="hint">Rigenera crop/preview del rect_name_echo.</span></span>
                </label>
              </div>
              <div class="rerun-tools">
                <button id="rerunSelectAllBtn" type="button" class="btn secondary mini">Spunta tutto</button>
                <button id="rerunClearBtn" type="button" class="btn secondary mini">Svuota</button>
                <button id="rerunOnlyOrientationBtn" type="button" class="btn secondary mini">Solo orientamento</button>
                <button id="rerunOnlyRectBtn" type="button" class="btn secondary mini">Solo rettangolo</button>
              </div>
              <div id="rerunStatus" class="small">Spunta una o piu parti: se scegli "Run completa nuova" nasce una nuova run; gli altri punti si ricalcolano sulla run selezionata.</div>
            </section>
            <div id="startStatus" class="small" style="margin-top:8px;">{{ initial_notice|e }}</div>
          </form>

          <section id="encodingStructSection" class="home-encoding">
            <div class="panel-head">
              <h2 style="margin:0;">Catalogo Vendor/Probe Da Encoding Struct <span class="tag tag-data">FSS</span></h2>
            </div>
            {% if encoding_struct_catalog and encoding_struct_catalog.ok %}
            <div class="small">
              Fonte: {{ encoding_struct_catalog.xlsx_path|e }} | foglio: {{ encoding_struct_catalog.sheet_name|e }} | righe analizzate: {{ encoding_struct_catalog.rows_scanned }}
            </div>
            <div class="small">
              {% if (encoding_struct_catalog.summary.model_mapping_rows or 0) > 0 %}
              Accorpamento modelli: attivo | regole: {{ encoding_struct_catalog.summary.model_mapping_rows }} | occorrenze accorpate: {{ encoding_struct_catalog.summary.model_merged_occurrences }} | raw unici: {{ encoding_struct_catalog.summary.model_raw_unique }} | mapping: {{ encoding_struct_catalog.summary.model_mapping_csv_path|e }}
              {% else %}
              Accorpamento modelli: non attivo
              {% endif %}
            </div>
            <div class="small">
              {% if (encoding_struct_catalog.summary.probe_mapping_rows or 0) > 0 %}
              Accorpamento probe: attivo | regole: {{ encoding_struct_catalog.summary.probe_mapping_rows }} | occorrenze accorpate: {{ encoding_struct_catalog.summary.probe_merged_occurrences }} | raw uniche: {{ encoding_struct_catalog.summary.probe_raw_unique }} | mapping: {{ encoding_struct_catalog.summary.probe_mapping_csv_path|e }}
              {% else %}
              Accorpamento probe: non attivo
              {% endif %}
            </div>
            <div class="home-encoding-grid">
              <div class="home-encoding-box">
                <h3>Vendor Canonici ({{ encoding_struct_catalog.summary.vendor_unique }} unici, {{ encoding_struct_catalog.summary.vendor_total_occurrences }} occorrenze)</h3>
                <div class="home-encoding-list">
                  {% if encoding_struct_catalog.vendors and encoding_struct_catalog.vendors|length > 0 %}
                    {% for it in encoding_struct_catalog.vendors %}
                    <div class="home-encoding-item">
                      <div class="label">{{ it.vendor_name|e }}</div>
                      <div class="home-encoding-count">{{ it.count }}</div>
                    </div>
                    {% endfor %}
                  {% else %}
                    <div class="home-encoding-item"><div class="label">Nessun vendor trovato.</div><div class="home-encoding-count">0</div></div>
                  {% endif %}
                </div>
              </div>
              <div class="home-encoding-box">
                <h3>Modelli Ecografo ({{ encoding_struct_catalog.summary.model_unique }} unici, {{ encoding_struct_catalog.summary.model_total_occurrences }} occorrenze)</h3>
                <div class="home-encoding-list">
                  {% if encoding_struct_catalog.models and encoding_struct_catalog.models|length > 0 %}
                    {% for it in encoding_struct_catalog.models %}
                    <div class="home-encoding-item">
                      <div class="label">{{ it.model_name|e }}</div>
                      <div class="home-encoding-count">{{ it.count }}</div>
                    </div>
                    {% endfor %}
                  {% else %}
                    <div class="home-encoding-item"><div class="label">Nessun modello trovato.</div><div class="home-encoding-count">0</div></div>
                  {% endif %}
                </div>
              </div>
              <div class="home-encoding-box">
                <h3>Probe ({{ encoding_struct_catalog.summary.probe_unique }} uniche, {{ encoding_struct_catalog.summary.probe_total_occurrences }} occorrenze)</h3>
                <div class="home-encoding-list">
                  {% if encoding_struct_catalog.probes and encoding_struct_catalog.probes|length > 0 %}
                    {% for it in encoding_struct_catalog.probes %}
                    <div class="home-encoding-item">
                      <div class="label">{{ it.probe_id|e }} - {{ it.probe_name|e }}</div>
                      <div class="home-encoding-count">{{ it.count }}</div>
                    </div>
                    {% endfor %}
                  {% else %}
                    <div class="home-encoding-item"><div class="label">Nessuna probe trovata.</div><div class="home-encoding-count">0</div></div>
                  {% endif %}
                </div>
              </div>
            </div>
            {% else %}
            <div class="small">Catalogo non disponibile: {{ (encoding_struct_catalog.error if encoding_struct_catalog else 'errore sconosciuto')|e }}</div>
            {% endif %}
          </section>

          <section id="modelsRegistrySection" class="home-models">
            <div class="panel-head">
              <h2 style="margin:0;">Catalogo Reti <span class="tag tag-model">registry</span></h2>
              <div class="home-models-toolbar">
                <button id="refreshModelsBtn" type="button" class="btn secondary mini">Aggiorna Metriche</button>
                <button
                  id="openModelsCsvBtn"
                  type="button"
                  class="btn secondary mini{% if not models_metrics_csv_path %} is-disabled{% endif %}"
                  {% if not models_metrics_csv_path %}disabled{% endif %}
                  data-open-folder-path="{{ models_metrics_csv_path|e }}"
                >
                  Apri CSV
                </button>
              </div>
            </div>
            <div id="modelsRegistryStatus" class="small">Caricamento registro reti...</div>
            <div id="modelsRegistryKpis" class="models-kpi-grid"></div>
            <div id="modelsRegistryTop" class="models-top-list" hidden></div>
            <div id="modelsPipelineTableWrap" class="models-table-wrap compact-models"></div>
            <div id="modelsLegacyTableWrap" class="models-table-wrap compact-models"></div>
          </section>
        </div>

        <div class="workspace-shell">
        <div class="workspace-topbar">
          <div class="pane-nav" role="tablist" aria-label="Sezioni run">
            <button id="tabMonitor" type="button" class="pane-btn active" data-pane-target="monitor" role="tab" aria-selected="true" aria-controls="paneMonitor">Monitor</button>
            <button id="tabAnalysis" type="button" class="pane-btn" data-pane-target="analysis" role="tab" aria-selected="false" aria-controls="paneAnalysis">Analisi Step</button>
            <button id="tabEvidence" type="button" class="pane-btn" data-pane-target="evidence" role="tab" aria-selected="false" aria-controls="paneEvidence">Evidenze</button>
            <button id="tabStudy" type="button" class="pane-btn" data-pane-target="study" role="tab" aria-selected="false" aria-controls="paneStudy">Studi</button>
            <button id="tabFss" type="button" class="pane-btn" data-pane-target="fss" role="tab" aria-selected="false" aria-controls="paneFss">File .fss</button>
          </div>
          <div class="workspace-meta">
            <div id="activeFolderTitle" class="folder-badge" title="{{ initial_input|e }}">
              Cartella: <span id="activeFolderName" class="folder-path">{{ initial_input|e }}</span>
            </div>
            <button id="backToSetupBtn" type="button" class="btn back mini">Nuova Run + Storico</button>
          </div>
        </div>
        <details class="workspace-rerun-panel" id="workspaceRerunPanel" data-rerun-root>
          <summary>Rerun selettivo: spunta i punti da ricalcolare</summary>
          <div class="rerun-step-grid">
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="full_pipeline">
              <span><span class="name">Run completa nuova</span><span class="hint">Rilancia tutta la pipeline.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="rect_red">
              <span><span class="name">Rect rosso linea 11</span><span class="hint">Segmenti rossi e line11.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="rect_per_image">
              <span><span class="name">Rect per immagine</span><span class="hint">Audit box frame-by-frame.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="rect_compare">
              <span><span class="name">Confronto reti rect</span><span class="hint">Globale vs specifica vendor.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="su_giu">
              <span><span class="name">SU/GIU</span><span class="hint">Evidenza e split.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="lr_marker">
              <span><span class="name">LR marker + linea 16</span><span class="hint">Quadranti e NF/LR/UD/LRUD.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="lt">
              <span><span class="name">L/T</span><span class="hint">Evidenza e split L/T.</span></span>
            </label>
            <label class="rerun-step">
              <input type="checkbox" data-rerun-target="line13">
              <span><span class="name">Linea 13 template</span><span class="hint">Crop/preview rect_name_echo.</span></span>
            </label>
          </div>
          <div class="rerun-tools">
            <button type="button" class="btn secondary mini" data-rerun-action="all">Spunta tutto</button>
            <button type="button" class="btn secondary mini" data-rerun-action="clear">Svuota</button>
            <button type="button" class="btn secondary mini" data-rerun-action="orientation">Solo orientamento</button>
            <button type="button" class="btn secondary mini" data-rerun-action="rect">Solo rettangolo</button>
            <button type="button" class="btn secondary mini" data-rerun-action="execute">Esegui rerun</button>
            <span id="workspaceRerunStatus" class="small">Nessun punto selezionato.</span>
          </div>
        </details>
        <div class="pane-stack">
        <section class="panel pane-panel active" id="paneMonitor" data-pane="monitor" role="tabpanel" aria-labelledby="tabMonitor">
          <h2>Monitor <span class="tag tag-monitor">live</span></h2>
          <div class="monitor-layout">
            <div class="monitor-main">
              <div class="monitor-left">
                <div id="runInfo" class="small">{{ fallback_run_info|e }}</div>
                <div id="timeline" class="timeline"></div>
                <div class="kpi-grid">
                  <div class="kpi"><div class="k">Status</div><div id="kpiStatus" class="v">{{ fallback_kpi_status|e }}</div></div>
                  <div class="kpi"><div class="k">Stage</div><div id="kpiStage" class="v">{{ fallback_kpi_stage|e }}</div></div>
                  <div class="kpi"><div class="k">Ultimo Agg.</div><div id="kpiUpdated" class="v" style="font-size:15px;">{{ fallback_kpi_updated|e }}</div></div>
                </div>
                <div id="monitorActivity" class="monitor-activity" hidden></div>
                <pre id="runLog" class="log">{{ fallback_log_text|e }}</pre>
              </div>
            </div>
            <div class="monitor-side">
              <div class="monitor-right">
                <div class="evidence-title">Decision Tree Live <span class="tag tag-monitor">live</span></div>
                <div id="decisionTreeLiveMonitor" class="decision-tree decision-tree-live-target"></div>
              </div>
            </div>
          </div>
        </section>

        <section class="panel pane-panel" id="paneAnalysis" data-pane="analysis" role="tabpanel" aria-labelledby="tabAnalysis" hidden>
          <h2>Analisi Step <span class="tag tag-analysis">qa</span></h2>
          <div class="analysis-layout">
            <div class="analysis-main">
              <div id="checksActions" class="actions" style="display:none;">
                <button id="saveAnnotationsBtn" class="btn">Salva Annotazioni</button>
                <button id="exportAnnotationsBtn" class="btn secondary">Export Annotazioni JSON</button>
                <button id="openLegacyHtmlBtn" class="btn secondary">Apri HTML Check Generato</button>
              </div>
              <div id="checksStatus" class="small">{{ fallback_checks_status|e }}</div>
              <div id="checksContainer" class="checks">
                {% if fallback_checks and fallback_checks|length > 0 %}
                  {% for step in fallback_checks %}
                  <div class="check-card">
                    <div class="check-head">
                      <div class="name">{{ loop.index }}. {{ step.step or ('step_' ~ loop.index) }}</div>
                      <span class="status-pill {% if step.status in ['ok','completed'] %}status-ok{% elif step.status in ['error','failed'] %}status-error{% else %}status-review{% endif %}">{{ step.status or 'review' }}</span>
                    </div>
                  </div>
                  {% endfor %}
                {% endif %}
              </div>
            </div>
            <aside class="analysis-side">
              <div id="analysisReviewSummary" class="analysis-summary">
                <div class="evidence-card">
                  <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
                  <div class="small">Seleziona una run completata.</div>
                </div>
              </div>
            </aside>
          </div>
        </section>

        <section class="panel pane-panel" id="paneEvidence" data-pane="evidence" role="tabpanel" aria-labelledby="tabEvidence" hidden>
          <h2>Evidenze Operative <span class="tag tag-evidence">inspect</span></h2>
          <div class="evidence-layout">
          <div class="evidence-main">
          <div id="evidenceStatus" class="small">{{ fallback_evidence_status|e }}</div>
          <div class="pane-nav evidence-nav" role="tablist" aria-label="Sezioni evidenze" style="margin-top:8px;">
            <button type="button" class="pane-btn evidence-btn active" data-evidence-section="raw">Raw</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="duplicates">Duplicati</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="rotation">Rotazioni</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="rect">Rect</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="template">Template</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="orientation">Orientamento</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="lt">L/T</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="scale">Scala</button>
            <button type="button" class="pane-btn evidence-btn" data-evidence-section="depth">Depth</button>
          </div>
          <div id="predictionSummary" hidden></div>
          <div id="decisionEvidence"></div>
          <div id="templateEvidence"></div>
          <div id="rotationEvidence"></div>
          <div id="rectPerImageEvidence"></div>
          <div id="suGiuEvidence"></div>
          <div id="lrMarkerEvidence"></div>
          <div id="ltEvidence"></div>
          <div id="scaleEvidence"></div>
          <div id="depthEvidence"></div>
          <div id="dupEvidence"></div>
          <div id="rawEvidenceBlock">
            <div class="evidence-card raw-evidence-card">
              <div class="evidence-title">Raw <span class="tag tag-data">images</span></div>
              <div class="dup-actions">
                <button type="button" class="btn secondary mini" data-open-evidence-gallery="raw">Apri galleria</button>
                <button type="button" class="btn secondary mini" data-raw-exclusion-action="save">Salva esclusioni</button>
                <button type="button" class="btn secondary mini" data-raw-exclusion-action="clear">Deseleziona</button>
              </div>
              <div id="rawExclusionStatus" class="small" style="margin-top:6px;">Caricamento immagini...</div>
              <div id="gallery" class="gallery raw-gallery">
            {% if false and initial_selected_run and fallback_raw_images_rel and fallback_raw_images_rel|length > 0 %}
              {% for rel in fallback_raw_images_rel[:12] %}
              <div class="tile">
                <img src="/api/runs/{{ initial_selected_run|urlencode }}/preview?path={{ rel|urlencode }}" loading="lazy">
                <div class="p">{{ rel }}</div>
              </div>
              {% endfor %}
            {% endif %}
              </div>
            </div>
          </div>
          </div>
          <aside class="evidence-side">
            <div id="evidenceSummaryHost"></div>
            <div id="evidenceRectSideHost"></div>
          </aside>
          </div>
        </section>
        <section class="panel pane-panel" id="paneStudy" data-pane="study" role="tabpanel" aria-labelledby="tabStudy" hidden>
          <h2>Studi Trasversali <span class="tag tag-evidence">batch</span></h2>
          <div class="study-toolbar">
            <div>
              <label>Studio</label>
              <select id="studyTypeSelect">
                <option value="duplicates">Duplicazione</option>
                <option value="rotation">Rotazione</option>
                <option value="vendor">Vendor</option>
                <option value="probe">Probe</option>
                <option value="rect">Rettangolo ecografico</option>
                <option value="orientation">Orientamento</option>
                <option value="lt">L/T</option>
              </select>
            </div>
            <button id="studyPrevRunBtn" type="button" class="btn secondary mini">Precedente</button>
            <button id="studyNextRunBtn" type="button" class="btn secondary mini">Successiva</button>
            <div>
              <label>Cartella</label>
              <select id="studyRunPicker"></select>
            </div>
            <button id="studyRefreshBtn" type="button" class="btn secondary mini">Aggiorna</button>
          </div>
          <div id="studyStatus" class="small" style="margin-top:8px;">Seleziona una run completata.</div>
          <div id="studyContent" style="margin-top:10px;"></div>
        </section>
        <section class="panel pane-panel" id="paneFss" data-pane="fss" role="tabpanel" aria-labelledby="tabFss" hidden>
          <h2>File .fss <span class="tag tag-data">preview</span></h2>
          <div id="fssStatus" class="small">Nessun dato .fss disponibile.</div>
          <div id="fssFileViewer" class="fss-viewer" style="margin-top:8px;">
            <div class="small">Seleziona una run completata per vedere il fake del file .fss.</div>
          </div>
          <details class="fss-reference" open>
            <summary>Guida formato legacy .fss</summary>
            <div class="fss-reference-body">
              <p>
                Il file <code>.fss</code> legacy e' una sequenza posizionale: il significato dipende dal numero di riga.
                Le righe <code>#01-#12</code> descrivono setup e coordinate video, <code>#13-#17</code> i check
                di template matching, <code>#18-#23</code> depth e geometria della guida, <code>#24-#26</code>
                l'estensione per sonde transrettali biplane.
              </p>
              <div class="fss-doc-grid">
                <section class="fss-doc-block">
                  <h3>Sintassi dei check</h3>
                  <p><code>TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|</code></p>
                  <ul class="fss-doc-list">
                    <li><code>TOP LEFT BOTTOM RIGHT</code>: rettangolo di ricerca nel frame delle immagini campione <code>VIDEO_X_SIZE x VIDEO_Y_SIZE</code>.</li>
                    <li><code>B</code>: esito atteso del check. <code>1</code> richiede match del template; <code>0</code> richiede assenza del match.</li>
                    <li><code>CH</code>: canale usato dal match, <code>1=R</code>, <code>2=G</code>, <code>4=B</code>, <code>7=tutti</code>.</li>
                    <li><code>TH</code>: soglia; <code>P1..P5</code>: parametri del metodo. Nel legacy il metodo <code>6</code> usa soprattutto <code>P1</code> e <code>P2</code> per la dinamica threshold.</li>
                    <li><code>MM</code>: <code>0=TM_SQDIFF</code>, <code>5=correlazione dopo edge</code>, <code>6=threshold + SQDIFF</code>, <code>7=equalizzazione istogramma + SQDIFF</code>.</li>
                  </ul>
                </section>
                <section class="fss-doc-block">
                  <h3>Regole pratiche del legacy</h3>
                  <ul class="fss-doc-list">
                    <li>Il file salva rettangoli di ricerca e parametri di match. I crop template stanno in <code>DB_echo/setup_&lt;id&gt;</code> come <code>echo_name.png</code>, <code>probe_name.png</code>, <code>orientation_0.png</code>, <code>depth_3.png</code>.</li>
                    <li>I separatori finali contano: il parser misura <code>|</code>, <code>;</code> e <code>,</code> per validare cardinalita' e gruppi.</li>
                    <li>I quattro flip state hanno ordine fisso: <code>No flip</code>, <code>Flip LR</code>, <code>Flip UD</code>, <code>Flip LR+UD</code>.</li>
                    <li>Se l'orientation non usa il simbolo, il builder conserva comunque la riga <code>#16</code> con placeholder per rispettare il formato.</li>
                    <li>Nel builder legacy <code>PIXEL_RATIO_X</code> e <code>PIXEL_RATIO_Y</code> vengono spesso dalla stessa lista, anche se il formato le tiene separate.</li>
                  </ul>
                </section>
              </div>
              <div class="fss-note">
                <strong>Riga #17.</strong> Ogni depth contiene quattro check. Il record esteso
                <code>...|MM|BM|CH:TH:P1:P2:P3:P4:P5|</code> abilita con <code>BM=1</code>
                una seconda immagine <code>depth_t_&lt;depth&gt;.png</code> per lo stato di transizione.
              </div>
              <div class="fss-note">
                <strong>Riga #24.</strong> Nel codice legacy <code>RECT_TRANS</code> e' il check che riconosce
                il piano biplana L/T per flip state, con template <code>trans_type_image_&lt;flip&gt;.png</code>.
              </div>
              <div class="fss-line-guide-wrap">
                <table class="fss-line-guide">
                  <thead>
                    <tr>
                      <th class="line-col">Riga</th>
                      <th class="field-col">Campo</th>
                      <th>Composizione</th>
                      <th>Cosa rappresenta</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td class="line-col">#01</td><td class="field-col">Versione</td><td><code>major.minor</code>, esempio <code>3.2</code>.</td><td>Versione del formato. Il reader legacy pretende due parti separate da punto.</td></tr>
                    <tr><td class="line-col">#02</td><td class="field-col">ID_ECHO</td><td>Intero.</td><td>ID univoco dell'ecografo scelto nel setup.</td></tr>
                    <tr><td class="line-col">#03</td><td class="field-col">ID_PROBE</td><td>Intero.</td><td>ID univoco della sonda.</td></tr>
                    <tr><td class="line-col">#04</td><td class="field-col">PROBETYPE</td><td><code>0</code> undefined, <code>1</code> linear, <code>2</code> convex, <code>3</code> trans L, <code>4</code> trans T.</td><td>Tipo probe. I define legacy elencano anche <code>5</code> trans single, ma questo reader FSS valida solo i valori <code>0-4</code>.</td></tr>
                    <tr><td class="line-col">#05</td><td class="field-col">KITNEEDLEGUIDE</td><td>Intero ID.</td><td>Riferimento a <code>DB_setup/kit_needle_guide_&lt;ID&gt;.ndg</code>.</td></tr>
                    <tr><td class="line-col">#06</td><td class="field-col">VIDEO_INPUT</td><td><code>0</code> HDMI, <code>1</code> VGA.</td><td>Tipo di flusso video in ingresso.</td></tr>
                    <tr><td class="line-col">#07</td><td class="field-col">VIDEO_INPUT_SIZE_X</td><td>Intero pixel.</td><td>Larghezza del frame in ingresso da ecografo/grabber.</td></tr>
                    <tr><td class="line-col">#08</td><td class="field-col">VIDEO_INPUT_SIZE_Y</td><td>Intero pixel.</td><td>Altezza del frame in ingresso da ecografo/grabber.</td></tr>
                    <tr><td class="line-col">#09</td><td class="field-col">VIDEO_X_SIZE</td><td>Intero pixel.</td><td>Larghezza immagini campione usate per crop template e rettangoli di ricerca.</td></tr>
                    <tr><td class="line-col">#10</td><td class="field-col">VIDEO_Y_SIZE</td><td>Intero pixel.</td><td>Altezza immagini campione usate per crop template e rettangoli di ricerca.</td></tr>
                    <tr><td class="line-col">#11</td><td class="field-col">RECT_ECHO</td><td><code>TOP|LEFT|BOTTOM|RIGHT|</code>.</td><td>Area ecografica dentro il frame campione; e' il riferimento geometrico per la pianificazione.</td></tr>
                    <tr><td class="line-col">#12</td><td class="field-col">GROUP_ORIENTATION</td><td>ID gruppo check. Nel wizard: <code>4</code> symbol, <code>5</code> depth.</td><td>Stabilisce quale famiglia di check ricava l'orientation.</td></tr>
                    <tr><td class="line-col">#13</td><td class="field-col">RECT_NAME_ECHO</td><td>Un record <code>Check</code>.</td><td>Rettangolo e soglia per riconoscere il nome/modello ecografo. Il crop associato e' <code>echo_name.png</code>.</td></tr>
                    <tr><td class="line-col">#14</td><td class="field-col">RECT_NAME_PROBE</td><td>Un record <code>Check</code>.</td><td>Rettangolo e soglia per riconoscere il nome/modello probe. Il crop associato e' <code>probe_name.png</code>.</td></tr>
                    <tr><td class="line-col">#15</td><td class="field-col">PROIBITED_SCREEN</td><td>Gruppi separati da <code>,</code>; check interni separati da <code>;</code>.</td><td>Template delle schermate vietate. Il formato supporta piu' check per schermata; il builder corrente salva di solito <code>proibited_screen_&lt;screen&gt;_0.png</code>.</td></tr>
                    <tr><td class="line-col">#16</td><td class="field-col">RECT_ORIENTATION</td><td>Quattro <code>Check</code> separati da <code>;</code>.</td><td>Check di orientation per i quattro flip state in ordine fisso.</td></tr>
                    <tr><td class="line-col">#17</td><td class="field-col">RECT_DEPTH</td><td>Depth separate da <code>,</code>; ogni depth ha quattro record <code>...|MM|BM|PARAMS2|;</code>.</td><td>Riconoscimento della depth per flip state. Il numero di gruppi definisce la cardinalita' delle righe <code>#18-#22</code>.</td></tr>
                    <tr><td class="line-col">#18</td><td class="field-col">VECT_DEPTH</td><td><code>D0|D1|...|DN|</code>, float mm.</td><td>Depth disponibili nello stesso ordine di <code>depth_0.png</code>, <code>depth_1.png</code> e dei gruppi di <code>#17</code>.</td></tr>
                    <tr><td class="line-col">#19</td><td class="field-col">PIXEL_RATIO_X</td><td><code>P0|P1|...|PN|</code>, float mm/pixel.</td><td>Rapporto millimetri/pixel lungo X per depth.</td></tr>
                    <tr><td class="line-col">#20</td><td class="field-col">PIXEL_RATIO_Y</td><td><code>P0|P1|...|PN|</code>, float mm/pixel.</td><td>Rapporto millimetri/pixel lungo Y per depth.</td></tr>
                    <tr><td class="line-col">#21</td><td class="field-col">SCALE_LINE</td><td><code>x1|x2|y1|y2|length_mm|tick_mm|label_side|;</code> per depth.</td><td>Linea della scala video: coordinate, lunghezza fisica, passo label/tick e lato label (<code>-1</code> sinistra, <code>+1</code> destra).</td></tr>
                    <tr><td class="line-col">#22</td><td class="field-col">CENTRE_DISTANCE</td><td>Per depth un vettore angle separato da <code>|</code>; depth separate da <code>;</code>.</td><td>Distanza della prima linea guida per ogni depth e angle. Le altre linee vengono ricostruite con le distanze del file <code>.ndg</code>.</td></tr>
                    <tr><td class="line-col">#23</td><td class="field-col">ANGLE</td><td><code>A0|A1|...|AM|</code>, gradi <code>-180..180</code>.</td><td>Angoli reali usati per disegnare le linee guida, non necessariamente gli angoli nominali della guida.</td></tr>
                    <tr><td class="line-col">#24</td><td class="field-col">RECT_TRANS</td><td>Quattro <code>Check</code> separati da <code>;</code>.</td><td>Solo sonde transrettali biplane: riconoscimento del piano L/T per flip state.</td></tr>
                    <tr><td class="line-col">#25</td><td class="field-col">ID NEXT PROBE</td><td>Intero setup ID.</td><td>ID del file <code>setup_&lt;ID&gt;.fss</code> complementare della stessa sonda biplana; non e' l'ID probe di riga <code>#03</code>.</td></tr>
                    <tr><td class="line-col">#26</td><td class="field-col">BIPLANA RECOGNITION MODE</td><td><code>0</code> no sign/no depth, <code>1</code> by depth, <code>2</code> by sign.</td><td>Strategia usata per distinguere vista lineare e trasversa. Le righe <code>#24-#26</code> non vengono scritte per probe non biplane.</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        </section>
        </div>
      </main>
      <aside class="history-column">
        <div class="panel history-panel-setup">
          <div class="panel-head">
            <h2>Storico Run <span class="tag tag-data">history</span></h2>
            <button id="historyOpenFinderBtn" type="button" class="btn secondary mini is-disabled" disabled>Apri Finder</button>
          </div>
          <div id="historyFinderPath" class="small" style="margin-bottom:8px;">Percorso: {{ history_runs_root|e }}</div>
          <div id="runsList" class="run-list">
            {% if initial_runs and initial_runs|length > 0 %}
              {% for run in initial_runs %}
              <div
                class="run-item openable"
                onclick="window.location.href='/?selected_run={{ run.run_id|urlencode }}&input_folder={{ initial_input|urlencode }}';"
              >
                <div class="t">{{ run.run_id }}</div>
                <div class="s">{{ run.input_folder or "-" }}</div>
                <div class="s"><span class="status-pill {% if run.status in ['ok','completed'] %}status-ok{% elif run.status in ['error','failed'] %}status-error{% else %}status-review{% endif %}">{{ run.status or "-" }}</span> {{ run.created_at or "-" }}</div>
                <div class="s">tempo: {{ run.elapsed_text or "-" }}</div>
                <div class="run-actions">
                  <form class="inline-form" method="post" action="/runs/rerun_form" onclick="event.stopPropagation();" onsubmit="return confirm('Rieseguire la pipeline sulla stessa cartella di questa run?');">
                    <input type="hidden" name="run_id" value="{{ run.run_id }}">
                    <input type="hidden" name="input_folder" value="{{ run.input_folder or '' }}">
                    <button type="submit" class="btn secondary mini">Rerun</button>
                  </form>
                  <form class="inline-form" method="post" action="/runs/delete_form" onclick="event.stopPropagation();" onsubmit="return confirm('Eliminare la run {{ run.run_id }}?');">
                    <input type="hidden" name="run_id" value="{{ run.run_id }}">
                    <input type="hidden" name="input_folder" value="{{ initial_input|e }}">
                    <button type="submit" class="btn danger mini">Elimina</button>
                  </form>
                </div>
              </div>
              {% endfor %}
            {% else %}
              <div class='run-item'><div class='s'>Nessuna run disponibile.</div></div>
            {% endif %}
          </div>
        </div>
        <div class="panel history-panel-workspace" hidden>
          <div class="workspace-rail">
            <section id="rightRailMonitor" class="rail-block" data-rail-pane="monitor" hidden>
              <div class="evidence-title">Decision Tree Live <span class="tag tag-monitor">live</span></div>
              <div class="small">Flusso decisionale in tempo reale della run selezionata.</div>
              <div id="decisionTreeLive" class="decision-tree decision-tree-live-target"></div>
            </section>
            <section id="rightRailAnalysis" class="rail-block" data-rail-pane="analysis" hidden>
              <div id="analysisReviewSummaryRail" class="analysis-summary">
                <div class="evidence-card">
                  <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
                  <div class="small">Seleziona una run completata.</div>
                </div>
              </div>
            </section>
            <section id="rightRailEvidence" class="rail-block" data-rail-pane="evidence" hidden>
              <div id="predictionSummaryRail">
                <div class="evidence-card">
                  <div class="evidence-title">Riepilogo Predizioni <span class="tag tag-data">overview</span></div>
                  <div class="small">Seleziona una run completata.</div>
                </div>
              </div>
            </section>
            <section id="rightRailFss" class="rail-block" data-rail-pane="fss" hidden>
              <div class="evidence-card">
                <div class="evidence-title">File .fss <span class="tag tag-data">preview</span></div>
                <div class="small">Nessun pannello laterale dedicato per questa sezione.</div>
              </div>
            </section>
          </div>
        </div>
      </aside>
    </div>
  </div>

  <div id="folderBrowserModal" class="modal hidden">
    <div class="modal-card">
      <div class="modal-head">
        <h3>Seleziona Cartella Input</h3>
        <div id="browseStatus" class="small"></div>
      </div>
      <div class="browse-top">
        <input id="browsePathInput" type="text" readonly>
        <button id="browseUpBtn" type="button" class="btn secondary">Su</button>
        <button id="browseRefreshBtn" type="button" class="btn secondary">Aggiorna</button>
        <button id="browseSelectCurrentBtn" type="button" class="btn">Usa questa</button>
      </div>
      <div id="browseDirsList" class="dir-list"></div>
      <div class="modal-foot">
        <button id="browseCancelBtn" type="button" class="btn secondary">Chiudi</button>
      </div>
    </div>
  </div>
  <div id="dupCompareModal" class="modal hidden">
    <div class="compare-card">
      <div class="modal-head">
        <h3>Confronto Duplicati</h3>
        <div id="dupCompareMeta" class="small"></div>
      </div>
      <div class="compare-grid">
        <div class="compare-pane">
          <div class="label">Tenuta (kept)</div>
          <img id="dupCompareKeptImg" alt="kept image">
          <div id="dupCompareKeptPath" class="compare-path"></div>
        </div>
        <div class="compare-pane">
          <div class="label">Rimossa (removed)</div>
          <img id="dupCompareRemovedImg" alt="removed image">
          <div id="dupCompareRemovedPath" class="compare-path"></div>
        </div>
      </div>
      <div class="modal-foot" style="margin-top:10px;">
        <button id="dupCompareCloseBtn" type="button" class="btn secondary">Chiudi</button>
      </div>
    </div>
  </div>
  <div id="rectCompareModal" class="modal hidden">
    <div id="rectCompareCard" class="rect-gallery-card">
      <div class="modal-head">
        <h3 id="rectCompareTitle">Galleria Rect</h3>
        <span id="rectCompareModeChip" class="rect-mode-chip">Modalita</span>
        <div id="rectCompareMeta" class="small"></div>
      </div>
      <div class="rect-gallery-layout">
        <div class="rect-gallery-main">
          <div id="rectCompareImageLabel" class="label">Frame con overlay rect (singolo + globale)</div>
          <div class="rect-gallery-image-wrap">
            <img id="rectCompareImage" alt="rect gallery image">
            <canvas id="rectCompareCanvas" class="rect-gallery-canvas" aria-hidden="true"></canvas>
          </div>
          <div id="rectCompareOverlayPath" class="compare-path"></div>
        </div>
        <aside class="rect-gallery-side">
          <div class="rect-gallery-kv"><div class="k">Posizione</div><div id="rectCompareIndex" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Discrepanza</div><div id="rectCompareDiscrepancy" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">IoU</div><div id="rectCompareIou" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Center norm</div><div id="rectCompareCenterNorm" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Area ratio</div><div id="rectCompareAreaRatio" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Dimensione immagine</div><div id="rectCompareSize" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Rect singolo</div><div id="rectCompareSingleRect" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Rect globale (scalato)</div><div id="rectCompareGlobalRect" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Frame sorgente</div><div id="rectCompareImagePath" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Overlay error</div><div id="rectCompareOverlayError" class="v">-</div></div>
          <div class="rect-gallery-kv"><div class="k">Legenda Overlay</div><div id="rectCompareLegend" class="v">-</div></div>
        </aside>
      </div>
      <div id="rectCompareThumbs" class="rect-gallery-thumbs"></div>
      <div class="modal-foot">
        <div class="rect-gallery-nav">
          <button id="rectComparePrevBtn" type="button" class="btn secondary">Precedente</button>
          <button id="rectCompareNextBtn" type="button" class="btn secondary">Successiva</button>
        </div>
        <div id="rectCompareCounter" class="rect-gallery-counter">-/-</div>
        <button id="rectCompareCloseBtn" type="button" class="btn secondary">Chiudi</button>
      </div>
    </div>
  </div>
  <div id="evidenceGalleryModal" class="modal hidden">
    <div class="gallery-modal-card">
      <div class="modal-head">
        <h3 id="evidenceGalleryTitle">Galleria Evidenze</h3>
        <div id="evidenceGalleryMeta" class="small"></div>
      </div>
      <div id="evidenceGalleryStatus" class="small"></div>
      <div id="evidenceGalleryGrid" class="gallery-modal-grid"></div>
      <div class="modal-foot">
        <div class="lr-marker-gallery-nav">
          <button type="button" class="btn secondary" data-lr-marker-nav="-1">Precedente</button>
          <button type="button" class="btn secondary" data-lr-marker-nav="1">Successiva</button>
        </div>
        <button id="evidenceGalleryCloseBtn" type="button" class="btn secondary">Chiudi</button>
      </div>
    </div>
  </div>
  <div id="fssHoverTip" class="fss-tooltip" hidden></div>

  <script>
    const runsListEl = document.getElementById("runsList");
    const startStatusEl = document.getElementById("startStatus");
    const runInfoEl = document.getElementById("runInfo");
    const timelineEl = document.getElementById("timeline");
    const runLogEl = document.getElementById("runLog");
    const checksStatusEl = document.getElementById("checksStatus");
    const checksContainerEl = document.getElementById("checksContainer");
    const galleryEl = document.getElementById("gallery");
    const evidenceStatusEl = document.getElementById("evidenceStatus");
    const predictionSummaryEl = document.getElementById("predictionSummary");
    const predictionSummaryRailEl = document.getElementById("predictionSummaryRail");
    const analysisReviewSummaryEl = document.getElementById("analysisReviewSummary");
    const analysisReviewSummaryRailEl = document.getElementById("analysisReviewSummaryRail");
    const evidenceSummaryHostEl = document.getElementById("evidenceSummaryHost");
    const evidenceRectSideHostEl = document.getElementById("evidenceRectSideHost");
    const fssStatusEl = document.getElementById("fssStatus");
    const fssFileViewerEl = document.getElementById("fssFileViewer");
    const decisionEvidenceEl = document.getElementById("decisionEvidence");
    const templateEvidenceEl = document.getElementById("templateEvidence");
    const rotationEvidenceEl = document.getElementById("rotationEvidence");
    const rectPerImageEvidenceEl = document.getElementById("rectPerImageEvidence");
    const suGiuEvidenceEl = document.getElementById("suGiuEvidence");
    const lrMarkerEvidenceEl = document.getElementById("lrMarkerEvidence");
    const ltEvidenceEl = document.getElementById("ltEvidence");
    const scaleEvidenceEl = document.getElementById("scaleEvidence");
    const depthEvidenceEl = document.getElementById("depthEvidence");
    const dupEvidenceEl = document.getElementById("dupEvidence");
    const folderBrowserModalEl = document.getElementById("folderBrowserModal");
    const dupCompareModalEl = document.getElementById("dupCompareModal");
    const rectCompareModalEl = document.getElementById("rectCompareModal");
    const rectCompareCardEl = document.getElementById("rectCompareCard");
    const evidenceGalleryModalEl = document.getElementById("evidenceGalleryModal");
    const evidenceGalleryCardEl = evidenceGalleryModalEl ? evidenceGalleryModalEl.querySelector(".gallery-modal-card") : null;
    const evidenceGalleryGridEl = document.getElementById("evidenceGalleryGrid");
    const evidenceGalleryTitleEl = document.getElementById("evidenceGalleryTitle");
    const evidenceGalleryMetaEl = document.getElementById("evidenceGalleryMeta");
    const evidenceGalleryStatusEl = document.getElementById("evidenceGalleryStatus");
    const fssHoverTipEl = document.getElementById("fssHoverTip");
    const dupCompareMetaEl = document.getElementById("dupCompareMeta");
    const dupCompareKeptImgEl = document.getElementById("dupCompareKeptImg");
    const dupCompareRemovedImgEl = document.getElementById("dupCompareRemovedImg");
    const dupCompareKeptPathEl = document.getElementById("dupCompareKeptPath");
    const dupCompareRemovedPathEl = document.getElementById("dupCompareRemovedPath");
    const rectCompareMetaEl = document.getElementById("rectCompareMeta");
    const rectCompareModeChipEl = document.getElementById("rectCompareModeChip");
    const rectCompareTitleEl = document.getElementById("rectCompareTitle");
    const rectCompareImageLabelEl = document.getElementById("rectCompareImageLabel");
    const rectCompareImageEl = document.getElementById("rectCompareImage");
    const rectCompareCanvasEl = document.getElementById("rectCompareCanvas");
    const rectCompareOverlayPathEl = document.getElementById("rectCompareOverlayPath");
    const rectCompareIndexEl = document.getElementById("rectCompareIndex");
    const rectCompareDiscrepancyEl = document.getElementById("rectCompareDiscrepancy");
    const rectCompareIouEl = document.getElementById("rectCompareIou");
    const rectCompareCenterNormEl = document.getElementById("rectCompareCenterNorm");
    const rectCompareAreaRatioEl = document.getElementById("rectCompareAreaRatio");
    const rectCompareSizeEl = document.getElementById("rectCompareSize");
    const rectCompareSingleRectEl = document.getElementById("rectCompareSingleRect");
    const rectCompareGlobalRectEl = document.getElementById("rectCompareGlobalRect");
    const rectCompareImagePathEl = document.getElementById("rectCompareImagePath");
    const rectCompareOverlayErrorEl = document.getElementById("rectCompareOverlayError");
    const rectCompareLegendEl = document.getElementById("rectCompareLegend");
    const rectCompareThumbsEl = document.getElementById("rectCompareThumbs");
    const rectCompareCounterEl = document.getElementById("rectCompareCounter");
    const rectComparePrevBtnEl = document.getElementById("rectComparePrevBtn");
    const rectCompareNextBtnEl = document.getElementById("rectCompareNextBtn");
    const browseStatusEl = document.getElementById("browseStatus");
    const browsePathInputEl = document.getElementById("browsePathInput");
    const browseDirsListEl = document.getElementById("browseDirsList");
    const checksActionsEl = document.getElementById("checksActions");
    const kpiStatusEl = document.getElementById("kpiStatus");
    const kpiStageEl = document.getElementById("kpiStage");
    const kpiUpdatedEl = document.getElementById("kpiUpdated");
    const historyOpenFinderBtnEl = document.getElementById("historyOpenFinderBtn");
    const historyFinderPathEl = document.getElementById("historyFinderPath");
    const backToSetupBtnEl = document.getElementById("backToSetupBtn");
    const refreshModelsBtnEl = document.getElementById("refreshModelsBtn");
    const modelsRegistryStatusEl = document.getElementById("modelsRegistryStatus");
    const modelsRegistryKpisEl = document.getElementById("modelsRegistryKpis");
    const modelsRegistryTopEl = document.getElementById("modelsRegistryTop");
    const modelsRegistryTableWrapEl = document.getElementById("modelsRegistryTableWrap");
    const modelsPipelineTableWrapEl = document.getElementById("modelsPipelineTableWrap");
    const modelsLegacyTableWrapEl = document.getElementById("modelsLegacyTableWrap");
    const bulkAcquisitionRootEl = document.getElementById("bulkAcquisitionRoot");
    const startBulkBtnEl = document.getElementById("startBulkBtn");
    const refreshBulkBtnEl = document.getElementById("refreshBulkBtn");
    const bulkStatusEl = document.getElementById("bulkStatus");
    const excludedImagesJsonEl = document.getElementById("excludedImagesJson");
    const exclusionStatusEl = document.getElementById("exclusionStatus");
    const rawExclusionStatusEl = document.getElementById("rawExclusionStatus");
    const imageExclusionListEl = document.getElementById("imageExclusionList");
    const loadExclusionsBtnEl = document.getElementById("loadExclusionsBtn");
    const saveExclusionsBtnEl = document.getElementById("saveExclusionsBtn");
    const clearExclusionsBtnEl = document.getElementById("clearExclusionsBtn");
    const noGeneratedImagesEl = document.getElementById("noGeneratedImages");
    const noSplitSymlinksEl = document.getElementById("noSplitSymlinks");
    const rerunPresetEl = document.getElementById("rerunPreset");
    const rerunStepGridEl = document.getElementById("rerunStepGrid");
    const rerunSelectAllBtnEl = document.getElementById("rerunSelectAllBtn");
    const rerunClearBtnEl = document.getElementById("rerunClearBtn");
    const rerunOnlyOrientationBtnEl = document.getElementById("rerunOnlyOrientationBtn");
    const rerunOnlyRectBtnEl = document.getElementById("rerunOnlyRectBtn");
    const rerunRectMarginPctEl = document.getElementById("rerunRectMarginPct");
    const rerunRectBrightThrEl = document.getElementById("rerunRectBrightThr");
    const rerunForceEvidenceEl = document.getElementById("rerunForceEvidence");
    const rerunSelectedBtnEl = document.getElementById("rerunSelectedBtn");
    const rerunStatusEl = document.getElementById("rerunStatus");
    const workspaceRerunStatusEl = document.getElementById("workspaceRerunStatus");
    const studyTypeSelectEl = document.getElementById("studyTypeSelect");
    const studyRunPickerEl = document.getElementById("studyRunPicker");
    const studyPrevRunBtnEl = document.getElementById("studyPrevRunBtn");
    const studyNextRunBtnEl = document.getElementById("studyNextRunBtn");
    const studyRefreshBtnEl = document.getElementById("studyRefreshBtn");
    const studyStatusEl = document.getElementById("studyStatus");
    const studyContentEl = document.getElementById("studyContent");
    const decisionTreeLiveEls = Array.from(document.querySelectorAll(".decision-tree-live-target"));
    const monitorActivityEl = document.getElementById("monitorActivity");
    const workspaceShellEl = document.querySelector(".workspace-shell");
    const setupPanelEl = document.querySelector(".panel-core");
    const historyColumnEl = document.querySelector(".history-column");
    const historySetupPanelEl = document.querySelector(".history-panel-setup");
    const historyWorkspacePanelEl = document.querySelector(".history-panel-workspace");
    const rightRailBlocks = Array.from(document.querySelectorAll(".rail-block[data-rail-pane]"));
    const mainColumnEl = document.querySelector(".main-column");
    const activeFolderTitleEl = document.getElementById("activeFolderTitle");
    const activeFolderNameEl = document.getElementById("activeFolderName");
    const paneButtons = Array.from(document.querySelectorAll(".pane-btn[data-pane-target]"));
    const panePanels = Array.from(document.querySelectorAll(".pane-panel"));
    const evidenceSectionButtons = Array.from(document.querySelectorAll(".evidence-btn"));
    const rawEvidenceBlockEl = document.getElementById("rawEvidenceBlock");

    const STAGES_ORDER = [
      "queued",
      "run_started",
      "copy_started",
      "copy_completed",
      "raw_images_ready_for_input_copy",
      "pipeline_started",
      "pipeline_finished",
      "pipeline_row_decisions",
      "checks_built",
      "review_html_built",
      "run_completed"
    ];

    let runs = [];
    let selectedRunId = null;
    let pollTimer = null;
    let selectedRunStatus = null;
    let selectedRunData = null;
    let annotationsState = null;
    let rectGalleryItems = [];
    let rectGalleryIndex = 0;
    let rectGalleryRunId = "";
    let rectGalleryMode = "rect";
    let lrMarkerGalleryItems = [];
    let lrMarkerGalleryIndex = 0;
    let lrMarkerGalleryRunId = "";
    let lrMarkerGalleryGroup = "";
    let lrMarkerGalleryMode = "all";
    let lrMarkerGalleryOrientationGroups = [];
    let lrMarkerGallerySourceTotal = 0;
    let lrMarkerGalleryExcludedCount = 0;
    let lrMarkerGalleryReliableThreshold = 0.62;
    let lrMarkerGalleryTargetWidth = 0;
    let lrMarkerGalleryTargetHeight = 0;
    let lrMarkerManualPickMode = false;
    let lrMarkerManualDragStart = null;
    let lrMarkerManualDragSuppressClickUntil = 0;
    let activeModalGallery = "";
    let lastGalleryOpenSig = "";
    let lastGalleryOpenTsMs = 0;
    let selectedRunDataRevisionKey = "";
    let browseCurrentPath = "";
    let browseListing = null;
    let runsPollTimer = null;
    let activePane = "monitor";
    let activeEvidenceSection = "raw";
    let historyFinderPath = "";
    let rectVendorCompareStateByRun = {};
    let activeStudyType = "duplicates";
    let studyRunId = "";
    let studyDataCache = {};
    let inputImageRows = [];
    let rawExclusionRows = [];
    let rawEvidenceLoadToken = 0;
    const initialSelectedRunId = "{{ initial_selected_run|e }}";
    const historyRunsRoot = "{{ history_runs_root|e }}";
    const browseDefaultRoot = "{{ browse_default_root|e }}";
    const modelsMetricsCsvPathDefault = "{{ models_metrics_csv_path|e }}";
    const fssTemplatePathDefault = "{{ fss_template_path|e }}";
    const fssTemplateLinesDefault = {{ fss_template_lines|tojson }};

    function fmtTs(iso) {
      if (!iso) return "-";
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleString();
    }

    function fmtRunElapsed(run) {
      const r = (run && typeof run === "object") ? run : {};
      const txt = String(r.elapsed_text || "").trim();
      if (txt) return txt;
      const sec = Number(r.elapsed_seconds);
      if (!Number.isFinite(sec)) return "-";
      const total = Math.max(0, Math.round(sec));
      if (total < 60) return `${total}s`;
      const minutes = Math.floor(total / 60);
      const rem = total % 60;
      if (minutes < 60) return `${minutes}m ${String(rem).padStart(2, "0")}s`;
      const hours = Math.floor(minutes / 60);
      const min = minutes % 60;
      if (hours < 24) return `${hours}h ${String(min).padStart(2, "0")}m`;
      const days = Math.floor(hours / 24);
      return `${days}g ${String(hours % 24).padStart(2, "0")}h`;
    }

    function statusClass(status) {
      if (status === "ok" || status === "completed") return "status-ok";
      if (status === "error" || status === "failed") return "status-error";
      return "status-review";
    }

    function modelsStatusClass(updateStatus) {
      const s = String(updateStatus || "").trim().toUpperCase();
      if (s === "OK") return "status-ok";
      if (s === "AGGIORNARE") return "status-error";
      if (s === "MONITORARE" || s === "VERIFICARE") return "status-review";
      return "status-unknown";
    }

    function toNumMaybe(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function formatMetricValue(value) {
      const n = toNumMaybe(value);
      if (n === null) return "-";
      if (Math.abs(n) >= 1000) return n.toLocaleString("it-IT", {maximumFractionDigits: 2});
      return n.toFixed(4);
    }

    function formatSamplesValue(value) {
      const n = toNumMaybe(value);
      if (n === null) return "-";
      if (Math.abs(n - Math.round(n)) < 1e-6) return Math.round(n).toLocaleString("it-IT");
      return n.toLocaleString("it-IT", {maximumFractionDigits: 2});
    }

    function formatBytes(value) {
      const n = Number(value || 0);
      if (!Number.isFinite(n) || n <= 0) return "-";
      if (n < 1024) return `${Math.round(n)} B`;
      if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
      return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    }

    function selectedExcludedImages() {
      if (!imageExclusionListEl) return [];
      return Array.from(imageExclusionListEl.querySelectorAll('input[data-image-rel]:checked'))
        .map((el) => String(el.getAttribute("data-image-rel") || "").trim())
        .filter(Boolean);
    }

    function syncExcludedImagesHidden() {
      const excluded = selectedExcludedImages();
      const total = Array.isArray(inputImageRows) ? inputImageRows.length : 0;
      if (excludedImagesJsonEl) excludedImagesJsonEl.value = total ? JSON.stringify(excluded) : "__use_saved__";
      if (exclusionStatusEl) {
        exclusionStatusEl.textContent = total
          ? `${excluded.length} escluse su ${total} immagini. Memoria modificabile.`
          : "Nessuna cartella caricata.";
      }
      return excluded;
    }

    function renderImageExclusions(images) {
      inputImageRows = Array.isArray(images) ? images : [];
      if (!imageExclusionListEl) return;
      if (!inputImageRows.length) {
        imageExclusionListEl.innerHTML = "<div class='small' style='padding:8px;'>Nessuna immagine trovata nella cartella.</div>";
        syncExcludedImagesHidden();
        return;
      }
      imageExclusionListEl.innerHTML = inputImageRows.map((img) => {
        const rel = String(img.rel || "");
        return `
          <label class="image-exclusion-row">
            <input type="checkbox" data-image-rel="${esc(rel)}"${img.excluded ? " checked" : ""}>
            <span class="path">${esc(rel)}</span>
            <span class="size">${esc(formatBytes(img.size_bytes))}</span>
          </label>
        `;
      }).join("");
      imageExclusionListEl.querySelectorAll('input[data-image-rel]').forEach((el) => {
        el.addEventListener("change", syncExcludedImagesHidden);
      });
      syncExcludedImagesHidden();
    }

    async function loadInputExclusions() {
      const inputFolder = String((document.getElementById("inputFolder") || {}).value || "").trim();
      if (!inputFolder) {
        if (exclusionStatusEl) exclusionStatusEl.textContent = "Inserisci prima una cartella input.";
        return;
      }
      if (exclusionStatusEl) exclusionStatusEl.textContent = "Caricamento immagini...";
      const out = await apiPost("/api/input/images", {input_folder: inputFolder});
      renderImageExclusions(out.images || []);
      if (exclusionStatusEl) {
        const excluded = selectedExcludedImages().length;
        exclusionStatusEl.textContent = `${excluded} escluse su ${Number(out.count || 0)} immagini. Memoria: ${out.memory_path || "-"}`;
      }
    }

    async function saveInputExclusions() {
      const inputFolder = String((document.getElementById("inputFolder") || {}).value || "").trim();
      if (!inputFolder) {
        if (exclusionStatusEl) exclusionStatusEl.textContent = "Inserisci prima una cartella input.";
        return;
      }
      const excluded = syncExcludedImagesHidden();
      const out = await apiPost("/api/input/exclusions", {
        input_folder: inputFolder,
        excluded_images_rel: excluded,
      });
      if (exclusionStatusEl) {
        exclusionStatusEl.textContent = `${Number(out.excluded_count || excluded.length)} esclusioni salvate.`;
      }
    }

    function clearInputExclusions() {
      if (!imageExclusionListEl) return;
      imageExclusionListEl.querySelectorAll('input[data-image-rel]').forEach((el) => {
        el.checked = false;
      });
      syncExcludedImagesHidden();
    }

    function renderModelsRegistry(payload) {
      const primaryWrap = modelsPipelineTableWrapEl || modelsRegistryTableWrapEl;
      const legacyWrap = modelsLegacyTableWrapEl;
      if (!modelsRegistryStatusEl || !modelsRegistryKpisEl || !primaryWrap) return;
      if (!payload || typeof payload !== "object") {
        modelsRegistryStatusEl.textContent = "Registro reti non disponibile.";
        modelsRegistryKpisEl.innerHTML = "";
        primaryWrap.innerHTML = "<div class='small' style='padding:8px;'>Nessun dato.</div>";
        if (legacyWrap) legacyWrap.innerHTML = "";
        if (modelsRegistryTopEl) {
          modelsRegistryTopEl.innerHTML = "";
          modelsRegistryTopEl.hidden = true;
        }
        return;
      }
      if (!payload.ok) {
        modelsRegistryStatusEl.textContent = `Errore registro reti: ${String(payload.error || "errore sconosciuto")}`;
        modelsRegistryKpisEl.innerHTML = "";
        primaryWrap.innerHTML = "<div class='small' style='padding:8px;'>Impossibile leggere il CSV metriche.</div>";
        if (legacyWrap) legacyWrap.innerHTML = "";
        if (modelsRegistryTopEl) {
          modelsRegistryTopEl.innerHTML = "";
          modelsRegistryTopEl.hidden = true;
        }
        return;
      }

      const rows = Array.isArray(payload.rows) ? payload.rows : [];
      const sm = (payload.summary && typeof payload.summary === "object") ? payload.summary : {};
      const sc = (sm.status_counts && typeof sm.status_counts === "object") ? sm.status_counts : {};
      const csvPath = String(payload.csv_path || modelsMetricsCsvPathDefault || "").trim();
      const topModels = Array.isArray(sm.top_models) ? sm.top_models : [];
      const totalRows = Number(sm.total_rows || rows.length || 0);
      const withMetrics = Number(sm.with_metrics_count || 0);
      const checkpointOnly = Number(sm.checkpoint_only_count || 0);
      const missingCkpt = Number(sm.missing_checkpoint_count || 0);
      const missingMetrics = Number(sm.missing_metrics_count || 0);
      const smallSamples = Number(sm.small_sample_count || 0);

      modelsRegistryStatusEl.textContent = `CSV: ${csvPath || "-"} | reti censite: ${totalRows}`;
      modelsRegistryStatusEl.title = csvPath || "-";

      const kpiItems = [
        {k: "Reti totali", v: totalRows, cls: ""},
        {k: "Con metriche", v: withMetrics, cls: ""},
        {k: "Checkpoint only", v: checkpointOnly, cls: ""},
        {k: "OK", v: Number(sc.OK || 0), cls: "status-ok"},
        {k: "Monitorare", v: Number(sc.MONITORARE || 0), cls: "status-review"},
        {k: "Aggiornare", v: Number(sc.AGGIORNARE || 0), cls: "status-error"},
        {k: "Verificare", v: Number(sc.VERIFICARE || 0), cls: "status-review"},
        {k: "Test < 50", v: smallSamples, cls: "status-review"},
        {k: "Checkpoint mancanti", v: missingCkpt, cls: missingCkpt > 0 ? "status-error" : "status-ok"},
        {k: "Metriche mancanti", v: missingMetrics, cls: missingMetrics > 0 ? "status-review" : "status-ok"},
      ];
      modelsRegistryKpisEl.innerHTML = kpiItems.map((it) => `
        <div class="models-kpi">
          <div class="k">${esc(it.k)}</div>
          <div class="v">${esc(String(it.v))}</div>
          ${it.cls ? `<span class="status-pill ${esc(it.cls)}" style="margin-top:4px;">${esc(it.cls.replace("status-", ""))}</span>` : ""}
        </div>
      `).join("");

      if (modelsRegistryTopEl) {
        if (!topModels.length) {
          modelsRegistryTopEl.hidden = true;
          modelsRegistryTopEl.innerHTML = "";
        } else {
          modelsRegistryTopEl.hidden = false;
          modelsRegistryTopEl.innerHTML = `
            <div class="small" style="margin-bottom:4px;"><b>Top metriche primarie</b> (solo righe con valore numerico)</div>
            ${topModels.map((item) => {
              const runLabel = [String(item.family || ""), String(item.vendor || ""), String(item.run_name || "")]
                .filter(Boolean)
                .join(" · ");
              return `
                <div class="models-top-row">
                  <div>${esc(runLabel || "-")}</div>
                  <div class="models-metric">${esc(String(item.primary_metric || "-"))}: ${esc(formatMetricValue(item.primary_value_num))}</div>
                  <div><span class="status-pill ${modelsStatusClass(item.update_status)}">${esc(String(item.update_status || "-"))}</span></div>
                </div>
              `;
            }).join("")}
          `;
        }
      }

      if (!rows.length) {
        primaryWrap.innerHTML = "<div class='small' style='padding:8px;'>CSV letto ma senza righe dati.</div>";
        if (legacyWrap) legacyWrap.innerHTML = "";
        return;
      }

      function isPipelineModel(r) {
        const family = String(r.family || "").toLowerCase();
        const ckpt = String(r.checkpoint_path || "").toLowerCase();
        const runName = String(r.run_name || "").toLowerCase();
        if (ckpt.includes("artifacts/10_active_pipeline/pipeline_fss_head/models/")) return true;
        if (family === "orientation_su_giu_classifier" && runName.includes("model_su_giu_rect")) return true;
        if (family === "lt_classifier" && runName.includes("transrectal_rect_only")) return true;
        return false;
      }

      function renderModelsTable(title, tableRows) {
        if (!tableRows.length) {
          return `<div class="small" style="padding:8px;"><b>${esc(title)}</b>: nessuna rete.</div>`;
        }
        const tableRowsHtml = tableRows.map((r) => {
        const status = String(r.update_status || "UNKNOWN");
        const sampleNum = toNumMaybe(r.test_samples_num);
        const sampleIsSmall = sampleNum !== null && sampleNum < 50;
        const cpAbs = String(r.checkpoint_abs || "").trim();
        const mtAbs = String(r.metrics_abs || "").trim();
        const cpPath = String(r.checkpoint_path || "").trim();
        const mtPath = String(r.metrics_path || "").trim();
        const cpExists = !!r.checkpoint_exists;
        const mtExists = !!r.metrics_exists;
        const familyScopeVendor = [
          String(r.family || ""),
          String(r.scope || ""),
          String(r.vendor || ""),
        ].filter(Boolean).join(" / ");
        return `
          <tr>
            <td>
              <b>${esc(familyScopeVendor || "-")}</b>
            </td>
            <td>
              <b>${esc(String(r.run_name || "-"))}</b>
            </td>
            <td>
              <span class="models-metric">${esc(String(r.primary_metric || "-"))}: ${esc(formatMetricValue(r.primary_value_num ?? r.primary_value))}</span>
              <span class="models-secondary">${esc(String(r.secondary_metrics || "-"))}</span>
            </td>
            <td>
              ${esc(formatSamplesValue(r.test_samples_num ?? r.test_samples))}
              ${sampleIsSmall ? `<div style="margin-top:4px;"><span class="status-pill status-review">campione piccolo</span></div>` : ""}
            </td>
            <td>
              <span class="status-pill ${modelsStatusClass(status)}">${esc(status)}</span>
              <div class="models-secondary">${esc(String(r.update_reason || "-"))}</div>
              ${String(r.notes || "").trim() ? `<div class="models-secondary">${esc(String(r.notes || ""))}</div>` : ""}
            </td>
            <td>
              <div>
                <span class="status-pill ${cpExists ? "status-ok" : "status-error"}">ckpt ${cpExists ? "ok" : "missing"}</span>
                <span class="models-path">${esc(cpPath || "-")}</span>
                ${cpAbs ? `<button type="button" class="btn secondary mini" style="margin-top:5px;" data-open-folder-path="${esc(cpAbs)}">Apri</button>` : ""}
              </div>
              <div style="margin-top:6px;">
                <span class="status-pill ${mtExists ? "status-ok" : "status-review"}">metrics ${mtExists ? "ok" : "missing"}</span>
                <span class="models-path">${esc(mtPath || "-")}</span>
                ${mtAbs ? `<button type="button" class="btn secondary mini" style="margin-top:5px;" data-open-folder-path="${esc(mtAbs)}">Apri</button>` : ""}
              </div>
            </td>
          </tr>
        `;
        }).join("");

        return `
        <div class="models-table-title">${esc(title)} <span class="tag tag-model">${tableRows.length}</span></div>
        <table class="models-table">
          <thead>
            <tr>
              <th>Famiglia / Scope / Vendor</th>
              <th>Run</th>
              <th>Metrica</th>
              <th>Test samples</th>
              <th>Stato</th>
              <th>Artifacts</th>
            </tr>
          </thead>
          <tbody>${tableRowsHtml}</tbody>
        </table>
      `;
      }

      const pipelineRows = rows.filter(isPipelineModel);
      const legacyRows = rows.filter((r) => !isPipelineModel(r));
      primaryWrap.innerHTML = renderModelsTable("Reti in uso dalla pipeline", pipelineRows);
      if (legacyWrap) legacyWrap.innerHTML = renderModelsTable("Reti legacy / non attive", legacyRows);
    }

    async function loadModelsRegistry() {
      if (!modelsRegistryStatusEl) return;
      modelsRegistryStatusEl.textContent = "Caricamento registro reti...";
      try {
        const payload = await apiGet("/api/models/metrics");
        renderModelsRegistry(payload);
      } catch (err) {
        renderModelsRegistry({ok: false, error: err.message});
      }
    }

    function setActivePane(name) {
      const valid = ["monitor", "analysis", "evidence", "study", "fss"];
      const next = valid.includes(String(name || "")) ? String(name) : "monitor";
      activePane = next;
      paneButtons.forEach((btn) => {
        const target = String(btn.getAttribute("data-pane-target") || "");
        const isActive = target === next;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
        btn.tabIndex = isActive ? 0 : -1;
      });
      panePanels.forEach((panel) => {
        const paneName = String(panel.getAttribute("data-pane") || "");
        const isActive = paneName === next;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
        panel.setAttribute("aria-hidden", isActive ? "false" : "true");
      });
      try {
        window.localStorage.setItem("workbench.activePane", next);
      } catch (err) {
        // no-op
      }
      updateWorkspaceRightRail();
    }

    function updateWorkspaceRightRail() {
      const isWorkspace = document.body.classList.contains("workspace-mode");
      rightRailBlocks.forEach((block) => {
        const paneName = String(block.getAttribute("data-rail-pane") || "");
        const isActive = isWorkspace && paneName === activePane;
        block.hidden = !isActive;
        block.classList.toggle("active", isActive);
      });
    }

    function setViewMode(mode) {
      const isWorkspace = String(mode || "") === "workspace";
      document.body.classList.toggle("workspace-mode", isWorkspace);
      document.body.classList.toggle("setup-mode", !isWorkspace);
      if (backToSetupBtnEl) backToSetupBtnEl.hidden = !isWorkspace;
      if (workspaceShellEl) workspaceShellEl.setAttribute("aria-hidden", isWorkspace ? "false" : "true");
      if (setupPanelEl) setupPanelEl.setAttribute("aria-hidden", isWorkspace ? "true" : "false");
      if (historyColumnEl) {
        historyColumnEl.hidden = false;
        historyColumnEl.setAttribute("aria-hidden", "false");
      }
      if (historySetupPanelEl) historySetupPanelEl.hidden = isWorkspace;
      if (historyWorkspacePanelEl) historyWorkspacePanelEl.hidden = !isWorkspace;
      if (mainColumnEl) mainColumnEl.setAttribute("data-view", isWorkspace ? "workspace" : "setup");
      if (isWorkspace) ensureWorkspaceSidebars();
      updateWorkspaceRightRail();
    }

    function ensureWorkspaceSidebars() {
      if (predictionSummaryRailEl && predictionSummaryEl) {
        predictionSummaryRailEl.innerHTML = predictionSummaryEl.innerHTML || "<div class='small'>Nessun riepilogo predizioni disponibile.</div>";
      }
      if (analysisReviewSummaryRailEl && analysisReviewSummaryEl) {
        analysisReviewSummaryRailEl.innerHTML = analysisReviewSummaryEl.innerHTML || `
          <div class="evidence-card">
            <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
            <div class="small">Seleziona una run completata.</div>
          </div>
        `;
      }
    }

    function setActiveFolderTitle(pathValue) {
      if (!activeFolderTitleEl || !activeFolderNameEl) return;
      const full = String(pathValue || "").trim();
      let folderName = "-";
      if (full) {
        const parts = full.split(/[\\\\/]+/).filter(Boolean);
        folderName = parts.length ? parts[parts.length - 1] : full;
      }
      activeFolderNameEl.textContent = folderName;
      activeFolderTitleEl.title = full || "-";
    }

    function setHistoryFinderPath(pathValue) {
      const p = String(pathValue || "").trim();
      historyFinderPath = p;
      if (historyFinderPathEl) {
        historyFinderPathEl.textContent = `Percorso: ${p || "-"}`;
        historyFinderPathEl.title = p || "-";
      }
      if (historyOpenFinderBtnEl) {
        const disabled = !p;
        historyOpenFinderBtnEl.disabled = disabled;
        historyOpenFinderBtnEl.classList.toggle("is-disabled", disabled);
      }
    }

    function resolveHistoryFinderPath() {
      return String(historyRunsRoot || "").trim();
    }

    function statusEvents(statusObj) {
      const st = (statusObj && typeof statusObj === "object") ? statusObj : {};
      return Array.isArray(st.events) ? st.events.filter((ev) => ev && typeof ev === "object") : [];
    }

    function eventName(ev) {
      return String((ev && ev.event) || "").trim();
    }

    function latestEvent(statusObj) {
      const events = statusEvents(statusObj);
      return events.length ? events[events.length - 1] : null;
    }

    function eventByName(statusObj, name) {
      const target = String(name || "");
      const events = statusEvents(statusObj);
      for (let i = events.length - 1; i >= 0; i -= 1) {
        if (eventName(events[i]) === target) return events[i];
      }
      return null;
    }

    function hasEvent(statusObj, name) {
      return !!eventByName(statusObj, name);
    }

    function eventCountsText(obj) {
      if (!obj || typeof obj !== "object") return "";
      const parts = Object.entries(obj)
        .map(([k, v]) => `${k}=${cleanVal(v)}`)
        .filter(Boolean);
      return parts.join(" | ");
    }

    function renderMonitorActivity(statusObj) {
      if (!monitorActivityEl) return;
      const st = (statusObj && typeof statusObj === "object") ? statusObj : {};
      if (!statusObj) {
        monitorActivityEl.hidden = true;
        monitorActivityEl.innerHTML = "";
        monitorActivityEl.classList.remove("is-hot");
        return;
      }
      const events = statusEvents(st);
      const latest = latestEvent(st);
      const latestName = eventName(latest);
      const rawReady = eventByName(st, "raw_images_ready_for_input_copy");
      const copyStarted = eventByName(st, "copy_started");
      const copyDone = eventByName(st, "copy_completed");
      const pipelineStarted = hasEvent(st, "pipeline_started");
      const logLines = Array.isArray(st.logs_tail) ? st.logs_tail : [];
      const latestLog = String(logLines.length ? logLines[logLines.length - 1] : "").trim();
      let title = "Attivita live";
      let main = latestName ? latestName : cleanVal(st.stage || st.status);
      let meta = latestLog ? latestLog : `run=${cleanVal(st.run_id)}`;
      let hot = false;
      if (rawReady && !pipelineStarted) {
        title = "Raw images pronte per elaborazione";
        main = `${cleanVal(rawReady.count)} raw images stanno per essere elaborate nella copia/riferimento input`;
        meta = `cartella input copia: ${cleanVal(rawReady.input_copy_folder)} | modo: ${cleanVal(rawReady.mode)}`;
        hot = true;
      } else if (rawReady && pipelineStarted && String(st.status || "") === "running") {
        title = "Raw images in elaborazione";
        main = `${cleanVal(rawReady.count)} raw images in elaborazione dalla copia/riferimento input`;
        meta = `cartella input copia: ${cleanVal(rawReady.input_copy_folder)} | modo: ${cleanVal(rawReady.mode)}`;
        hot = true;
      } else if (copyStarted && !copyDone) {
        title = "Copia cartella input in corso";
        main = "Preparazione della cartella input per la pipeline";
        meta = `${cleanVal(copyStarted.src)} -> ${cleanVal(copyStarted.dst)}`;
        hot = true;
      } else if (latestName === "pipeline_started") {
        title = "Pipeline principale in corso";
        main = "Le immagini raw sono entrate nella pipeline";
        meta = latestLog || cleanVal(latest.command);
        hot = true;
      } else if (latestName) {
        const count = latest.items ?? latest.count ?? latest.checks_count ?? latest.created_total ?? "";
        const suffix = count !== "" ? ` | n=${cleanVal(count)}` : "";
        main = `${latestName}${suffix}`;
        if (latest.counts && typeof latest.counts === "object") meta = eventCountsText(latest.counts);
        else if (latest.reason) meta = `reason=${cleanVal(latest.reason)}`;
        else if (latest.path) meta = cleanVal(latest.path);
      }
      const recent = events.slice(-5).map((ev) => `<span class="activity-event">${esc(eventName(ev))}</span>`).join("");
      monitorActivityEl.hidden = false;
      monitorActivityEl.classList.toggle("is-hot", !!hot);
      monitorActivityEl.innerHTML = `
        <div class="activity-title">${esc(title)}</div>
        <div class="activity-main">${esc(main)}</div>
        <div class="activity-meta">${esc(meta || "-")}</div>
        ${recent ? `<div class="activity-events">${recent}</div>` : ""}
      `;
    }

    function inferDecisionTreeNodes(statusObj, summary) {
      const st = (statusObj && typeof statusObj === "object") ? statusObj : {};
      const sm = (summary && typeof summary === "object") ? summary : {};
      const row = (sm.pipeline_row && typeof sm.pipeline_row === "object") ? sm.pipeline_row : {};
      const de = (sm.decision_evidence && typeof sm.decision_evidence === "object") ? sm.decision_evidence : {};
      const rec = (sm.recognition_evidence && typeof sm.recognition_evidence === "object") ? sm.recognition_evidence : {};
      const vendorObj = (de.vendor && typeof de.vendor === "object") ? de.vendor : {};
      const probeObj = (de.probe && typeof de.probe === "object") ? de.probe : {};
      const suGiuPer = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object") ? rec.su_giu_per_image : {};
      const suGiuSplit = (rec.su_giu_split_folders && typeof rec.su_giu_split_folders === "object") ? rec.su_giu_split_folders : {};
      const lrMarker = (rec.lr_marker_per_image && typeof rec.lr_marker_per_image === "object") ? rec.lr_marker_per_image : {};
      const suGiuCountsObj = (suGiuSplit.counts && typeof suGiuSplit.counts === "object")
        ? suGiuSplit.counts
        : ((suGiuPer.label_counts && typeof suGiuPer.label_counts === "object") ? suGiuPer.label_counts : {});
      const ltPer = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
      const ltSplit = (rec.lt_split_folders && typeof rec.lt_split_folders === "object") ? rec.lt_split_folders : {};
      const ltCountsObj = (ltSplit.counts && typeof ltSplit.counts === "object")
        ? ltSplit.counts
        : ((ltPer.label_counts && typeof ltPer.label_counts === "object") ? ltPer.label_counts : {});
      const completed = new Set(Array.isArray(st.completed_stages) ? st.completed_stages : []);
      const stage = String(st.stage || "");
      const runStatus = String(st.status || "");
      const logs = Array.isArray(st.logs_tail) ? st.logs_tail.join("\\n").toLowerCase() : "";
      const latest = latestEvent(st);
      const latestName = eventName(latest);
      const rawReady = eventByName(st, "raw_images_ready_for_input_copy");
      const rowDecision = eventByName(st, "pipeline_row_decisions");

      const copyStarted = completed.has("copy_started") || stage === "copy_started" || !!eventByName(st, "copy_started");
      const pipelineStarted = completed.has("pipeline_started") || completed.has("pipeline_finished") || completed.has("checks_built") || stage === "pipeline_started" || stage === "pipeline_finished";
      const pipelineFinished = completed.has("pipeline_finished") || completed.has("checks_built") || completed.has("review_html_built") || completed.has("run_completed") || stage === "pipeline_finished";
      const dedupSeen = /dedup|duplicat/.test(logs) || Object.prototype.hasOwnProperty.call(sm, "duplicates_removed_count_recomputed");
      const rotationSeen = /rotaz|rotation/.test(logs) || !!(rowDecision && rowDecision.rotation_deg_clockwise !== undefined) || (sm.rotation_evidence && typeof sm.rotation_evidence === "object" && Object.keys(sm.rotation_evidence).length > 0);
      const vendorName = String(vendorObj.predicted_name || row.vendor_predicted || (rowDecision || {}).vendor || "").trim();
      const vendorConf = formatConfidence((vendorObj.confidence !== undefined && vendorObj.confidence !== null) ? vendorObj.confidence : (row.vendor_confidence !== undefined ? row.vendor_confidence : (rowDecision || {}).vendor_confidence));
      const rectName = String(row.line_11_rect_echo || (rowDecision || {}).rect_echo || "").trim();
      const suCount = Number(suGiuCountsObj.su || 0);
      const giuCount = Number(suGiuCountsObj.giu || 0);
      const otherCount = Number(suGiuCountsObj.other || 0);
      const suGiuFrames = Number(suGiuPer.images_total || row.su_giu_images_predicted || 0);
      const lrMarkerFrames = Number(lrMarker.images_total || row.lr_marker_images_predicted || 0);
      const lrMarkerMajority = String(row.lr_marker_majority_label || "").trim();
      const lrMarkerBest = String(row.lr_marker_best_label || "").trim();
      const lrMarkerBestScore = formatNum(row.lr_marker_best_score, 4);
      const lCount = Number(ltCountsObj.l || 0);
      const tCount = Number(ltCountsObj.t || 0);
      const ltOtherCount = Number(ltCountsObj.other || 0);
      const ltFrames = Number(ltPer.images_total || row.lt_images_predicted || 0);
      const probeLabel = [String(probeObj.predicted_id || row.line_03_id_probe || "").trim(), String(probeObj.predicted_name || row.line_03_probe_name || "").trim()]
        .filter(Boolean)
        .join(" - ");
      const probeLabelLive = probeLabel || String((rowDecision || {}).probe_id || "").trim();
      const probeConf = formatConfidence((probeObj.confidence !== undefined && probeObj.confidence !== null) ? probeObj.confidence : (row.line_03_probe_confidence !== undefined ? row.line_03_probe_confidence : (rowDecision || {}).probe_confidence));
      const l13 = String(row.line_13_rect_name_echo || (rowDecision || {}).line13 || "").trim();
      const l14 = String(row.line_14_rect_name_probe || (rowDecision || {}).line14 || "").trim();

      const copyDone = completed.has("copy_completed");
      const vendorDone = !!vendorName;
      const rectDone = !!rectName && rectName !== "-";
      const suGiuDone = suGiuFrames > 0 || (suCount + giuCount + otherCount) > 0;
      const lrMarkerDone = lrMarkerFrames > 0 || !!lrMarkerBest || !!lrMarkerMajority;
      const ltDone = ltFrames > 0 || (lCount + tCount + ltOtherCount) > 0;
      const probeDone = !!probeLabelLive && probeLabelLive !== "-";
      const line13Done = !!(l13 && l13 !== "-");
      const line14Done = !!(l14 && l14 !== "-");
      const exportDone = runStatus === "completed" || runStatus === "ok" || completed.has("run_completed");
      const hasFailed = runStatus === "failed" || runStatus === "error";

      let currentId = "copy";
      const liveEventNode = {
        copy_started: "copy",
        copy_completed: "copy",
        raw_images_ready_for_input_copy: "copy",
        pipeline_started: "dedup",
        pipeline_finished: "vendor",
        pipeline_row_decisions: "rect",
        checks_built: "export",
        rect_red_built: "rect",
        rect_red_skipped: "rect",
        rect_per_image_built: "rect",
        rect_per_image_skipped: "rect",
        su_giu_per_image_built: "su_giu",
        su_giu_per_image_skipped: "su_giu",
        su_giu_split_folders_built: "su_giu",
        su_giu_split_folders_skipped: "su_giu",
        lr_marker_per_image_built: "lr_marker",
        lr_marker_per_image_skipped: "lr_marker",
        lt_per_image_built: "lt",
        lt_per_image_skipped: "lt",
        lt_split_folders_built: "lt",
        lt_split_folders_skipped: "lt",
        review_html_built: "export",
        run_completed: "export",
      }[latestName] || "";
      if (liveEventNode) currentId = liveEventNode;
      else if (!copyDone) currentId = "copy";
      else if (!pipelineStarted) currentId = "dedup";
      else if (!dedupSeen) currentId = "dedup";
      else if (!rotationSeen) currentId = "rotation";
      else if (!vendorDone) currentId = "vendor";
      else if (!rectDone) currentId = "rect";
      else if (!suGiuDone) currentId = "su_giu";
      else if (!lrMarkerDone) currentId = "lr_marker";
      else if (!ltDone) currentId = "lt";
      else if (!probeDone) currentId = "probe";
      else if (!line13Done) currentId = "template13";
      else if (!line14Done) currentId = "template14";
      else currentId = "export";

      const nodes = [
        {
          id: "copy",
          title: "Copia Cartella Input + Raw Images",
          meta: rawReady
            ? `${cleanVal(rawReady.count)} raw images pronte in copia/riferimento input | ${cleanVal(rawReady.input_copy_folder)}`
            : (copyDone ? `OK (${cleanVal(st.input_folder)} -> ${cleanVal(st.run_dir)})` : (copyStarted ? "Preparazione riferimento input in corso" : "In attesa")),
        },
        {id: "dedup", title: "Deduplicazione Frame", meta: dedupSeen ? `Eseguita | rimossi=${cleanVal(sm.duplicates_removed_count_recomputed)}` : "In attesa / in corso"},
        {id: "rotation", title: "Valutazione Rotazione", meta: rotationSeen ? `Rotazione=${cleanVal((sm.rotation_evidence || {}).rotation_deg_clockwise)}°` : "In attesa / in corso"},
        {id: "vendor", title: "Decisione Vendor", meta: vendorDone ? `${vendorName}${vendorConf ? ` | conf=${vendorConf}` : ""}` : "In attesa / in corso"},
        {id: "rect", title: "Detection Rettangolo", meta: rectDone ? cleanVal(rectName) : "In attesa / in corso"},
        {id: "su_giu", title: "Smistamento frame SU/GIU", meta: suGiuDone ? `su=${cleanVal(suCount)} | giu=${cleanVal(giuCount)} | other=${cleanVal(otherCount)} | n=${cleanVal(suGiuFrames)}` : "In attesa / in corso"},
        {id: "lr_marker", title: "LR marker classico", meta: lrMarkerDone ? `majority=${cleanVal(lrMarkerMajority)} | best=${cleanVal(lrMarkerBest)} | score=${cleanVal(lrMarkerBestScore)} | n=${cleanVal(lrMarkerFrames)}` : "In attesa / in corso"},
        {id: "lt", title: "Classificazione frame L/T", meta: ltDone ? `L=${cleanVal(lCount)} | T=${cleanVal(tCount)} | other=${cleanVal(ltOtherCount)} | n=${cleanVal(ltFrames)}` : "In attesa / in corso"},
        {id: "probe", title: "Decisione Probe", meta: probeDone ? `${probeLabelLive}${probeConf ? ` | conf=${probeConf}` : ""}` : "In attesa / in corso"},
        {id: "template13", title: "Template linea 13", meta: line13Done ? `line13=${cleanVal(l13)}` : "In attesa / in corso"},
        {id: "template14", title: "Template linea 14", meta: line14Done ? `line14=${cleanVal(l14)}` : "In attesa / in corso"},
        {id: "export", title: "Export & QA", meta: exportDone ? `Completata (${cleanVal(runStatus)})` : (pipelineFinished ? "Post-processing" : "In attesa")},
      ];

      return nodes.map((node) => {
        const done = (
          (node.id === "copy" && copyDone) ||
          (node.id === "dedup" && pipelineStarted && dedupSeen) ||
          (node.id === "rotation" && pipelineStarted && rotationSeen) ||
          (node.id === "vendor" && vendorDone) ||
          (node.id === "rect" && rectDone) ||
          (node.id === "su_giu" && suGiuDone) ||
          (node.id === "lr_marker" && lrMarkerDone) ||
          (node.id === "lt" && ltDone) ||
          (node.id === "probe" && probeDone) ||
          (node.id === "template13" && line13Done) ||
          (node.id === "template14" && line14Done) ||
          (node.id === "export" && exportDone)
        );
        let state = "pending";
        if (done) state = "done";
        else if (node.id === currentId && !exportDone) state = "running";
        if (node.id === currentId && !exportDone && latestName && latestName !== "run_completed") state = "running";
        if (hasFailed && node.id === currentId) state = "error";
        return {...node, state};
      });
    }

    function renderDecisionTreeLive(statusObj, summary) {
      if (!decisionTreeLiveEls.length) return;
      const nodes = inferDecisionTreeNodes(statusObj, summary);
      const htmlTree = nodes
        .map((node, idx) => `
          <div class="dt-node ${esc(node.state)}">
            <div class="dt-title">${idx + 1}. ${esc(node.title)}</div>
            <div class="dt-meta">${esc(node.meta)}</div>
          </div>
        `)
        .join("");
      decisionTreeLiveEls.forEach((el) => {
        if (!el) return;
        el.innerHTML = htmlTree;
      });
    }

    function imgUrlForRun(runId, relPath) {
      const run = String(runId || "").trim();
      if (!run) return "";
      return `/api/runs/${encodeURIComponent(run)}/preview?path=${encodeURIComponent(relPath)}`;
    }

    function imgUrl(relPath) {
      return imgUrlForRun(selectedRunId, relPath);
    }

    function lrMarkerEnvelopeSpansHtml(orientationGroups, currentGroupKey) {
      const groups = Array.isArray(orientationGroups) ? orientationGroups : [];
      const current = String(currentGroupKey || "").trim().toUpperCase();
      return groups
        .filter((group) => group && group.envelope)
        .map((group) => {
          const key = String(group.key || "").trim().toUpperCase();
          const env = group.envelope || {};
          const currentClass = key && key === current ? " current" : "";
          const keyClass = key ? ` lr-env-${key.toLowerCase()}` : "";
          return `
            <span
              class="lr-marker-envelope-box${keyClass}${currentClass}"
              data-marker-top="${esc(cleanVal(env.top))}"
              data-marker-left="${esc(cleanVal(env.left))}"
              data-marker-bottom="${esc(cleanVal(env.bottom))}"
              data-marker-right="${esc(cleanVal(env.right))}"
            ><b>${esc(key || "-")}</b></span>
          `;
        })
        .join("");
    }

    function lrMarkerImageOverlayHtml(item, runIdValue, orientationGroups, options) {
      const it = (item && typeof item === "object") ? item : {};
      const opts = (options && typeof options === "object") ? options : {};
      const showCurrentMarker = opts.showCurrentMarker !== false;
      const wrapClass = String(opts.wrapClass || "").trim();
      const rel = String(it.image_rel || "");
      const run = String(runIdValue || selectedRunId || "");
      const groupKey = Object.prototype.hasOwnProperty.call(opts, "currentGroupKey")
        ? String(opts.currentGroupKey || "").trim().toUpperCase()
        : inferLrOrientationGroup(it);
      const reviewKey = String(opts.reviewKey || "").trim();
      const manualRect = parseLrMarkerRectText(opts.manualRect || "");
      const markerAttrs = `
        data-marker-top="${esc(cleanVal(it.marker_top_abs))}"
        data-marker-left="${esc(cleanVal(it.marker_left_abs))}"
        data-marker-bottom="${esc(cleanVal(it.marker_bottom_abs))}"
        data-marker-right="${esc(cleanVal(it.marker_right_abs))}"
      `;
      const manualAttrs = manualRect ? `
        data-marker-top="${esc(cleanVal(manualRect.top))}"
        data-marker-left="${esc(cleanVal(manualRect.left))}"
        data-marker-bottom="${esc(cleanVal(manualRect.bottom))}"
        data-marker-right="${esc(cleanVal(manualRect.right))}"
      ` : "";
      const envelopeHtml = lrMarkerEnvelopeSpansHtml(orientationGroups, groupKey);
      const imageHtml = rel
        ? `
          <div class="lr-marker-image-wrap${wrapClass ? ` ${esc(wrapClass)}` : ""}" ${reviewKey ? `data-lr-marker-review-key="${esc(reviewKey)}"` : ""}>
            <img src="${imgUrlForRun(run, rel)}" loading="eager" decoding="async" ${markerAttrs} onload="window.__fitLrMarkerBox && window.__fitLrMarkerBox(this)">
            ${envelopeHtml}
            ${showCurrentMarker ? `<span class="lr-marker-box" ${markerAttrs}></span>` : ""}
            ${manualAttrs ? `<span class="lr-marker-manual-box" ${manualAttrs}></span>` : ""}
          </div>
        `
        : `<div class="small">Preview non disponibile.</div>`;
      return imageHtml;
    }

    function lrMarkerTemplateHtml(item, runIdValue) {
      const it = (item && typeof item === "object") ? item : {};
      const templatePath = String(it.template_path || "");
      const run = String(runIdValue || selectedRunId || "");
      const templateHtml = templatePath
        ? `
          <div class="lr-template-preview">
            <div class="label">Template marker</div>
            <img src="${imgUrlForRun(run, templatePath)}" loading="eager" decoding="async">
            <div class="meta">${esc(templatePath)}</div>
          </div>
        `
        : `<div class="lr-template-preview"><div class="label">Template marker</div><div class="small">Template non disponibile.</div></div>`;
      return templateHtml;
    }

    function lrMarkerVisualHtml(item, runIdValue) {
      return `<div class="lr-marker-visual-grid">${lrMarkerImageOverlayHtml(item, runIdValue)}${lrMarkerTemplateHtml(item, runIdValue)}</div>`;
    }

    function orientationHeroBoxPreviewHtml(item, orientationGroups, runIdValue) {
      const it = (item && typeof item === "object") ? item : {};
      const groups = Array.isArray(orientationGroups) ? orientationGroups : [];
      const orderedKeys = ["NF", "LR", "UD", "LRUD"];
      const byKey = {};
      groups.forEach((group) => {
        const key = String((group && group.key) || "").trim().toUpperCase();
        if (key) byKey[key] = group;
      });
      const rel = String(it.image_rel || "").trim();
      const frameLabel = rel
        ? `${shortImageName(rel)} | frame=${cleanVal(it.image_index)} | score=${formatNum(it.match_score, 4)}`
        : "Nessun frame disponibile.";
      const availableBoxes = orderedKeys
        .map((key) => byKey[key])
        .filter((group) => group && group.envelope).length;
      const legendHtml = orderedKeys.map((key) => {
        const group = byKey[key] || {key, envelope: null};
        const env = group.envelope || null;
        const used = env ? cleanVal(env.boxes) : "0";
        const total = cleanVal(group.items && group.items.length);
        const meta = env
          ? `${formatLrEnvelope(env)} | usati=${used}/${total}`
          : `box non disponibile | frame=${total}`;
        return `
          <div class="orientation-box-row">
            <div class="orientation-box-chip ${esc(key.toLowerCase())}">${esc(key)}</div>
            <div class="meta">${esc(meta)}</div>
          </div>
        `;
      }).join("");
      const visualHtml = rel
        ? lrMarkerImageOverlayHtml(it, runIdValue, groups, {
            showCurrentMarker: false,
            currentGroupKey: "",
            wrapClass: "orientation-hero-image",
          })
        : `<div class="small">Nessuna immagine disponibile per la preview orientamento.</div>`;
      return `
        <div class="orientation-hero">
          <div class="orientation-hero-visual">${visualHtml}</div>
          <div class="orientation-hero-panel">
            <div class="orientation-hero-title">Box orientamento NF / LR / UD / LRUD</div>
            <div class="small">Preview su: <span class="path">${esc(frameLabel)}</span></div>
            <div class="small">Box visibili: ${esc(String(availableBoxes))}/4. I rettangoli mostrano le aree marker usate per costruire la riga orientation.</div>
            <div class="orientation-box-legend">${legendHtml}</div>
          </div>
        </div>
      `;
    }

    function lrMarkerEnvelopeLegendHtml(orientationGroups) {
      const groups = Array.isArray(orientationGroups) ? orientationGroups : [];
      if (!groups.length) return "";
      return `
        <div class="lr-marker-viewer-kv">
          <div class="k">Box gruppi orientamento</div>
          <div class="v">${groups.map((group) => {
            const key = String((group && group.key) || "-").trim().toUpperCase() || "-";
            return `${esc(key)}: ${esc(formatLrEnvelope(group && group.envelope))}`;
          }).join("<br>")}</div>
        </div>
      `;
    }

    window.__fitLrMarkerBox = function(img) {
      try {
        if (!img || !(img instanceof HTMLImageElement)) return;
        const wrap = img.closest(".lr-marker-image-wrap");
        const boxes = wrap ? Array.from(wrap.querySelectorAll(".lr-marker-box, .lr-marker-envelope-box, .lr-marker-manual-box")) : [];
        if (!wrap || !boxes.length) return;
        const w = Number(img.naturalWidth || 0);
        const h = Number(img.naturalHeight || 0);
        boxes.forEach((box) => {
          const source = box.dataset.markerTop ? box : img;
          const top = Number(source.dataset.markerTop || 0);
          const left = Number(source.dataset.markerLeft || 0);
          const bottom = Number(source.dataset.markerBottom || 0);
          const right = Number(source.dataset.markerRight || 0);
          if (!(w > 0 && h > 0 && bottom > top && right > left)) {
            box.style.display = "none";
            return;
          }
          box.style.display = "block";
          box.style.left = `${Math.max(0, Math.min(100, (left / w) * 100))}%`;
          box.style.top = `${Math.max(0, Math.min(100, (top / h) * 100))}%`;
          box.style.width = `${Math.max(0.2, Math.min(100, ((right - left) / w) * 100))}%`;
          box.style.height = `${Math.max(0.2, Math.min(100, ((bottom - top) / h) * 100))}%`;
        });
      } catch (err) {
        // no-op
      }
    };

    function esc(s) {
      return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;");
    }

    function helpTipHtml(title, items) {
      const titleTxt = String(title || "").trim();
      const rows = Array.isArray(items) ? items : [];
      const rowsHtml = rows
        .map((x) => String(x || "").trim())
        .filter(Boolean)
        .map((x) => `<li>${esc(x)}</li>`)
        .join("");
      if (!titleTxt && !rowsHtml) return "";
      const label = titleTxt || "Info";
      return `
        <span class="help-tip" tabindex="0" aria-label="${esc(label)}">
          ?
          <span class="help-tip-bubble">
            ${titleTxt ? `<span class="help-tip-title">${esc(titleTxt)}</span>` : ""}
            ${rowsHtml ? `<ul class="help-tip-list">${rowsHtml}</ul>` : ""}
          </span>
        </span>
      `;
    }

    function hideFssTooltip() {
      if (!fssHoverTipEl) return;
      fssHoverTipEl.classList.remove("visible");
      fssHoverTipEl.hidden = true;
    }

    function showFssTooltip(text, x, y) {
      if (!fssHoverTipEl) return;
      const msg = String(text || "").trim();
      if (!msg) {
        hideFssTooltip();
        return;
      }
      fssHoverTipEl.textContent = msg;
      fssHoverTipEl.hidden = false;
      fssHoverTipEl.classList.add("visible");
      moveFssTooltip(x, y);
    }

    function moveFssTooltip(x, y) {
      if (!fssHoverTipEl || fssHoverTipEl.hidden) return;
      const pad = 14;
      const rect = fssHoverTipEl.getBoundingClientRect();
      const vw = window.innerWidth || document.documentElement.clientWidth || 1200;
      const vh = window.innerHeight || document.documentElement.clientHeight || 800;
      let left = Number(x || 0) + 16;
      let top = Number(y || 0) + 16;
      if (left + rect.width + pad > vw) left = Math.max(pad, vw - rect.width - pad);
      if (top + rect.height + pad > vh) top = Math.max(pad, Number(y || 0) - rect.height - 12);
      fssHoverTipEl.style.left = `${Math.max(pad, left)}px`;
      fssHoverTipEl.style.top = `${Math.max(pad, top)}px`;
    }

    function cleanLineValue(v) {
      return String(v === undefined || v === null ? "" : v).trim();
    }

    function buildPendingLine16FromSummary(summary) {
      const payload = getLrMarkerPerImagePayload(summary);
      const groups = Array.isArray(payload.orientationGroups) ? payload.orientationGroups : [];
      const order = ["NF", "LR", "UD", "LRUD"];
      const byKey = {};
      groups.forEach((group) => {
        const key = String((group && group.key) || "").trim().toUpperCase();
        if (key) byKey[key] = group;
      });
      if (!order.every((key) => byKey[key] && byKey[key].envelope)) return "";
      return order.map((key) => {
        const env = byKey[key].envelope || {};
        return `${cleanVal(env.top)}|${cleanVal(env.left)}|${cleanVal(env.bottom)}|${cleanVal(env.right)}|1|0:0.000000:0:0:0:0:0|0|;`;
      }).join("");
    }

    function buildFssLineMap(row, summary) {
      const src = (row && typeof row === "object") ? row : {};
      const out = {};
      function put(lineNo, value) {
        const txt = cleanLineValue(value);
        if (!txt) return;
        out[String(lineNo)] = txt;
      }
      put(1, src.line_01_version);
      put(2, src.line_02_id_echo);
      put(3, src.line_03_id_probe);
      put(4, src.line_04_probe_type);
      put(6, src.line_06_video_input_code_0hdmi_1vga || src.line_06_video_input_code || src.line_06_video_input);
      put(7, src.line_07_video_input_size_x);
      put(8, src.line_08_video_input_size_y);
      put(9, src.line_09_video_x_size);
      put(10, src.line_10_video_y_size);
      put(11, src.line_11_rect_echo);
      put(12, src.line_12_group_orientation);
      put(13, src.line_13_rect_name_echo);
      put(14, src.line_14_rect_name_probe);
      put(16, buildPendingLine16FromSummary(summary) || src.line_16_rect_orientation);
      return out;
    }

    function fssLineDescription(lineNo) {
      const key = String(lineNo);
      const map = {
        "1": "Versione formato file .fss.",
        "2": "ID ecografo (echo ID).",
        "3": "ID sonda/probe.",
        "4": "Tipo probe (router da line #03; se 3-4 viene risolto via classificatore L/T: L->3, T->4).",
        "5": "ID del kit needle guide: riferimento al file kit_needle_guide_<ID>.ndg.",
        "6": "Tipo input video (codice HDMI/VGA).",
        "7": "Risoluzione frame video in ingresso: larghezza.",
        "8": "Risoluzione frame video in ingresso: altezza.",
        "9": "Larghezza immagini campione usate per template e rettangoli.",
        "10": "Altezza immagini campione usate per template e rettangoli.",
        "11": "Rettangolo ecografico principale top|left|bottom|right|.",
        "12": "ID gruppo che ricava orientation: tipicamente symbol=4 oppure depth=5.",
        "13": "Check rettangolo per riconoscimento nome/modello ecografo.",
        "14": "Check rettangolo per riconoscimento nome/modello probe.",
        "15": "Check delle schermate proibite, raggruppati per schermata.",
        "16": "Quattro check orientation in ordine No flip, Flip LR, Flip UD, Flip LR+UD.",
        "17": "Check depth per depth e flip state; puo' includere depth_t di transizione.",
        "18": "Vettore depth disponibili in mm.",
        "19": "Pixel ratio X in mm/pixel per depth.",
        "20": "Pixel ratio Y in mm/pixel per depth.",
        "21": "Linea scala per depth: coordinate, lunghezza mm, tick e lato label.",
        "22": "Centre distance della prima linea guida per depth e angle.",
        "23": "Angoli reali usati per disegnare le linee guida.",
        "24": "Check biplana L/T per i quattro flip state.",
        "25": "Setup ID del file .fss complementare per sonda biplana.",
        "26": "Modalita' riconoscimento biplana: 0 none, 1 by depth, 2 by sign.",
      };
      return map[key] || "Parametro avanzato del file .fss (non compilato direttamente dalla pipeline head).";
    }

    function renderFssPreview(summary) {
      if (!fssStatusEl || !fssFileViewerEl) return;
      if (!summary || typeof summary !== "object") {
        fssStatusEl.textContent = "Nessun dato .fss disponibile.";
        fssFileViewerEl.innerHTML = "<div class='small'>Seleziona una run completata per vedere il fake del file .fss.</div>";
        return;
      }
      const row = (summary.pipeline_row && typeof summary.pipeline_row === "object")
        ? summary.pipeline_row
        : {};
      const lineMap = buildFssLineMap(row, summary);
      const lineKeys = Object.keys(lineMap);

      const templateLines = Array.isArray(summary.fss_template_lines) && summary.fss_template_lines.length
        ? summary.fss_template_lines
        : (Array.isArray(fssTemplateLinesDefault) ? fssTemplateLinesDefault : []);
      const templatePath = cleanLineValue(summary.fss_template_path || fssTemplatePathDefault);
      const totalLines = Math.max(14, templateLines.length || 0);
      if (totalLines <= 0) {
        fssStatusEl.textContent = "Template .fss non disponibile.";
        fssFileViewerEl.innerHTML = "<div class='small'>Template .fss non trovato nel repository.</div>";
        return;
      }

      const rows = [];
      for (let idx = 1; idx <= totalLines; idx += 1) {
        const key = String(idx);
        const hasPipeline = Object.prototype.hasOwnProperty.call(lineMap, key);
        const baseLine = idx <= templateLines.length ? String(templateLines[idx - 1] || "") : "";
        const text = hasPipeline ? lineMap[key] : baseLine;
        const desc = `Riga ${String(idx).padStart(2, "0")} - ${fssLineDescription(idx)}`;
        rows.push(`
          <div class="fss-line ${hasPipeline ? "fss-filled" : "fss-template"}" title="${esc(desc)}" data-fss-desc="${esc(desc)}">
            <div class="fss-ln">${String(idx).padStart(2, "0")}</div>
            <div class="fss-content">${esc(text || "-")}</div>
          </div>
        `);
      }

      fssStatusEl.textContent = `Template: ${templatePath || "-"} | Righe valorizzate dalla pipeline: ${lineKeys.length}`;
      fssFileViewerEl.innerHTML = `
        <div class="fss-meta">
          <div class="small">Fake .fss: base template in chiaro, righe completate dalla pipeline in nero. Passa il mouse su una riga per vedere cosa rappresenta.</div>
          <div class="fss-legend">
            <span class="fss-chip">Template</span>
            <span class="fss-chip filled">Compilata pipeline</span>
          </div>
        </div>
        ${rows.join("")}
      `;
    }

    function buildLegacyDecisionEvidence(summary) {
      const row = (summary && typeof summary === "object" && summary.pipeline_row && typeof summary.pipeline_row === "object")
        ? summary.pipeline_row
        : {};
      const probeId = String(row.line_03_id_probe || "");
      const probeName = String(row.line_03_probe_name || "");
      const vendorSourceHint = String(row.vendor_source || "cnn_classifier");
      const probeSourceHint = String(row.probe_source || "cnn_classifier");
      const vendorThreshold = asNumber(row.vendor_threshold, asNumber(summary && summary.vendor_min_confidence, 0.5));
      const probeThreshold = asNumber(row.probe_threshold, asNumber(summary && summary.probe_min_confidence, 0.5));
      const vendorTopChoices = parseTopChoices(row.vendor_top3_json || row.vendor_topk_json || row.vendor_topk || "");
      const probeTopChoices = parseTopChoices(row.probe_top3_json || row.probe_topk_json || row.probe_topk || "");
      if (!vendorTopChoices.length && row.vendor_predicted) {
        vendorTopChoices.push({label: String(row.vendor_predicted), confidence: asNumber(row.vendor_confidence, NaN)});
      }
      if (!probeTopChoices.length && probeId) {
        probeTopChoices.push({
          label: probeId,
          name: probeName,
          confidence: asNumber(row.line_03_probe_confidence, NaN),
        });
      }
      const vendorSourceKind = normalizeDecisionSourceKind(vendorSourceHint, false);
      const probeSourceKind = normalizeDecisionSourceKind(probeSourceHint, false);
      return {
        vendor: {
          predicted_name: String(row.vendor_predicted || ""),
          confidence: Number(row.vendor_confidence || 0),
          threshold: vendorThreshold,
          source_kind: vendorSourceKind,
          source_hint: vendorSourceHint || "cnn_classifier",
          ocr_used: false,
          ocr_elements: {},
          ocr_comment: "Decisione vendor da classificatore immagini (senza OCR).",
          top_choices: vendorTopChoices,
          decision_reason: String(row.vendor_decision_reason || "").trim() || buildDecisionReasonFallback(vendorSourceKind, Number(row.vendor_confidence || 0), vendorThreshold, "vendor"),
        },
        probe: {
          predicted_id: probeId,
          predicted_name: probeName,
          confidence: Number(row.line_03_probe_confidence || 0),
          threshold: probeThreshold,
          source_kind: probeSourceKind,
          source_hint: probeSourceHint || "cnn_classifier",
          ocr_used: false,
          ocr_elements: {},
          ocr_comment: "Decisione probe da classificatore immagini (senza OCR).",
          top_choices: probeTopChoices,
          decision_reason: String(row.probe_decision_reason || "").trim() || buildDecisionReasonFallback(probeSourceKind, Number(row.line_03_probe_confidence || 0), probeThreshold, "probe"),
        },
      };
    }

    function cleanVal(v) {
      const s = String(v == null ? "" : v).trim();
      return s ? s : "-";
    }

    function asNumber(v, fallback = NaN) {
      const x = Number(v);
      return Number.isFinite(x) ? x : fallback;
    }

    function clamp01(x) {
      if (!Number.isFinite(x)) return 0;
      if (x < 0) return 0;
      if (x > 1) return 1;
      return x;
    }

    function formatNum(x, digits = 4) {
      const n = Number(x);
      if (!Number.isFinite(n)) return "-";
      return n.toFixed(digits);
    }

    function formatConfidence(confVal) {
      const n = asNumber(confVal, NaN);
      if (!Number.isFinite(n)) return "";
      if (n >= 0 && n <= 1) return `${(n * 100).toFixed(1)}%`;
      return formatNum(n, 4);
    }

    function normalizeDecisionSourceKind(sourceHint, ocrUsed) {
      if (ocrUsed) return "ocr";
      const txt = String(sourceHint || "").trim().toLowerCase();
      if (txt.includes("ocr")) return "ocr";
      if (txt.includes("manual") || txt.includes("user_")) return "manual";
      return "cnn";
    }

    function parseTopChoices(rawVal) {
      if (Array.isArray(rawVal)) {
        return rawVal.map((item) => {
          if (!item || typeof item !== "object") return null;
          const label = String(item.label || item.value || item.id || "").trim();
          if (!label) return null;
          const name = String(item.name || "").trim();
          const conf = asNumber(item.confidence, NaN);
          const out = {label};
          if (name) out.name = name;
          if (Number.isFinite(conf)) out.confidence = conf;
          return out;
        }).filter(Boolean).slice(0, 3);
      }

      const txt = String(rawVal || "").trim();
      if (!txt) return [];
      try {
        const parsed = JSON.parse(txt);
        if (Array.isArray(parsed)) return parseTopChoices(parsed);
      } catch (err) {
        // legacy plain-text format fallback
      }

      return txt
        .split("|")
        .map((tok) => tok.trim())
        .filter(Boolean)
        .map((tok) => {
          const sep = tok.lastIndexOf(":");
          if (sep > 0) {
            const label = tok.slice(0, sep).trim();
            const conf = asNumber(tok.slice(sep + 1).trim(), NaN);
            const out = {label};
            if (Number.isFinite(conf)) out.confidence = conf;
            return out;
          }
          return {label: tok};
        })
        .slice(0, 3);
    }

    function buildDecisionReasonFallback(sourceKind, confidence, threshold, label) {
      const field = String(label || "").trim() || "item";
      if (sourceKind === "manual") return `Selezione manuale utente su ${field}.`;
      if (sourceKind === "ocr") return `Decisione ${field} guidata da OCR.`;
      if (Number.isFinite(confidence) && Number.isFinite(threshold) && confidence < threshold) {
        return `CNN top-1 ${field} sotto soglia (${formatNum(confidence, 4)} < ${formatNum(threshold, 4)}).`;
      }
      if (Number.isFinite(confidence) && Number.isFinite(threshold)) {
        return `CNN top-1 ${field} sopra soglia (${formatNum(confidence, 4)} >= ${formatNum(threshold, 4)}).`;
      }
      return `Decisione ${field} da classificatore CNN.`;
    }

    function renderTopChoicesHtml(rawChoices, mode) {
      const choices = parseTopChoices(rawChoices);
      if (!choices.length) return "-";
      const kind = String(mode || "vendor");
      return choices.map((item, idx) => {
        const label = String(item.label || "-");
        const name = String(item.name || "").trim();
        const confTxt = formatConfidence(item.confidence);
        const value = (kind === "probe" && name) ? `${label} - ${name}` : label;
        return `${idx + 1}) ${esc(value)}${confTxt ? ` (${esc(confTxt)})` : ""}`;
      }).join("<br>");
    }

    function computeRectMetrics(row) {
      const top = asNumber(row.line_11_top, NaN);
      const left = asNumber(row.line_11_left, NaN);
      const bottom = asNumber(row.line_11_bottom, NaN);
      const right = asNumber(row.line_11_right, NaN);
      const frameW = asNumber(row.line_09_video_x_size, NaN);
      const frameH = asNumber(row.line_10_video_y_size, NaN);
      const imagesUsed = asNumber(row.images_used_rect, 0);
      const source = String(row.line_11_source || "").trim();
      const hasBox = [top, left, bottom, right].every((v) => Number.isFinite(v));
      const width = hasBox ? Math.max(0, right - left) : 0;
      const height = hasBox ? Math.max(0, bottom - top) : 0;
      const area = width * height;
      const frameArea = (Number.isFinite(frameW) && Number.isFinite(frameH) && frameW > 0 && frameH > 0)
        ? frameW * frameH
        : 0;
      const areaRatio = frameArea > 0 ? (area / frameArea) : NaN;
      const aspect = height > 0 ? (width / height) : NaN;

      const exportedConf = asNumber(row.line_11_confidence_proxy, NaN);
      const hasExported = Number.isFinite(exportedConf) && exportedConf >= 0 && exportedConf <= 1;
      if (hasExported) {
        return {
          confidence: clamp01(exportedConf),
          method: "model_exported",
          note: "confidence esportata dal runner line11",
          source,
          imagesUsed,
          width,
          height,
          areaRatio,
          aspect,
        };
      }

      // Heuristic proxy for old runs where rect confidence is not exported.
      const sourceScoreMap = {
        "vendor_specialized": 0.95,
        "global": 0.88,
        "global_low_vendor_conf": 0.74,
      };
      const sourceScore = sourceScoreMap[source] || 0.70;
      const sizeScore = clamp01(imagesUsed / 24.0);
      let areaScore = 0.40;
      if (Number.isFinite(areaRatio) && areaRatio > 0) {
        const target = 0.42;
        const tolerance = 0.40;
        areaScore = clamp01(1.0 - (Math.abs(areaRatio - target) / tolerance));
      }
      const proxy = clamp01(0.45 * sizeScore + 0.35 * areaScore + 0.20 * sourceScore);
      return {
        confidence: proxy,
        method: "heuristic_proxy",
        note: "proxy da source+area+frame usati (non probabilita del modello)",
        source,
        imagesUsed,
        width,
        height,
        areaRatio,
        aspect,
      };
    }

    function summaryClassForKey(key) {
      const t = String(key || "").toLowerCase();
      if (t.includes("vendor")) return "sum-vendor";
      if (t.includes("probe")) return "sum-probe";
      if (t.includes("rect")) return "sum-rect";
      if (t.includes("su/giu") || t.includes("orientamento")) return "sum-rotation";
      if (t.includes("rotazione") || t.includes("rotation")) return "sum-rotation";
      if (t.includes("status") || t.includes("reason")) return "sum-status";
      return "sum-data";
    }

    function renderSummaryCard(item) {
      const key = String(item && item.key ? item.key : "");
      const value = String(item && item.value ? item.value : "-");
      const hasConf = item && Object.prototype.hasOwnProperty.call(item, "conf");
      const conf = hasConf ? String(item.conf || "") : "";
      const confClass = conf ? "conf-chip" : "conf-chip empty";
      const cardClass = hasConf ? "summary-item has-conf" : "summary-item";
      const confHtml = hasConf ? `<div class="${confClass}">conf ${esc(conf || "-")}</div>` : "";
      return `<div class="${cardClass} ${summaryClassForKey(key)}"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div>${confHtml}</div>`;
    }

    function setPredictionSummaryHtml(html) {
      const safeHtml = String(html || "<div class='small'>Nessun riepilogo predizioni disponibile.</div>");
      if (predictionSummaryEl) {
        predictionSummaryEl.style.display = "none";
        predictionSummaryEl.innerHTML = "";
      }
      if (predictionSummaryRailEl) {
        predictionSummaryRailEl.innerHTML = safeHtml;
      }
    }

    function setAnalysisReviewSummaryHtml(html) {
      const safeHtml = String(html || `
        <div class="evidence-card">
          <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
          <div class="small">Seleziona una run completata.</div>
        </div>
      `);
      if (analysisReviewSummaryEl) {
        analysisReviewSummaryEl.innerHTML = safeHtml;
      }
      if (analysisReviewSummaryRailEl) {
        analysisReviewSummaryRailEl.innerHTML = safeHtml;
      }
    }

    function renderPredictionSummary(summary) {
      if (!predictionSummaryEl && !predictionSummaryRailEl) return;
      if (!summary || typeof summary !== "object") {
        setPredictionSummaryHtml("<div class='small'>Nessun riepilogo predizioni disponibile.</div>");
        return;
      }
      const row = (summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const de = (summary.decision_evidence && typeof summary.decision_evidence === "object")
        ? summary.decision_evidence
        : buildLegacyDecisionEvidence(summary);
      const vendor = (de.vendor && typeof de.vendor === "object") ? de.vendor : {};
      const probe = (de.probe && typeof de.probe === "object") ? de.probe : {};
      const rec = (summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const suGiuPer = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object") ? rec.su_giu_per_image : {};
      const suGiuSplit = (rec.su_giu_split_folders && typeof rec.su_giu_split_folders === "object") ? rec.su_giu_split_folders : {};
      const suGiuCounts = (suGiuSplit.counts && typeof suGiuSplit.counts === "object")
        ? suGiuSplit.counts
        : ((suGiuPer.label_counts && typeof suGiuPer.label_counts === "object") ? suGiuPer.label_counts : {});
      const ltPer = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
      const ltSplit = (rec.lt_split_folders && typeof rec.lt_split_folders === "object") ? rec.lt_split_folders : {};
      const ltCounts = (ltSplit.counts && typeof ltSplit.counts === "object")
        ? ltSplit.counts
        : ((ltPer.label_counts && typeof ltPer.label_counts === "object") ? ltPer.label_counts : {});
      const rot = (summary.rotation_evidence && typeof summary.rotation_evidence === "object")
        ? summary.rotation_evidence
        : {};
      const l13 = (rec.line13_template && typeof rec.line13_template === "object") ? rec.line13_template : {};
      const l14 = (rec.line14_template && typeof rec.line14_template === "object") ? rec.line14_template : {};
      const probeLabel = [String(probe.predicted_id || ""), String(probe.predicted_name || "")]
        .filter(Boolean)
        .join(" - ");
      const rectM = computeRectMetrics(row);
      const vendorConfValue = (vendor.confidence !== undefined && vendor.confidence !== null)
        ? vendor.confidence
        : row.vendor_confidence;
      const probeConfValue = (probe.confidence !== undefined && probe.confidence !== null)
        ? probe.confidence
        : row.line_03_probe_confidence;
      const cards = [
        {key: "Vendor", value: cleanVal(vendor.predicted_name || row.vendor_predicted), conf: formatConfidence(vendorConfValue)},
        {key: "Probe", value: cleanVal(probeLabel || row.line_03_id_probe), conf: formatConfidence(probeConfValue)},
        {key: "line04 probe_type", value: cleanVal(row.line_04_probe_type)},
        {key: "Rect line11", value: cleanVal(row.line_11_rect_echo), conf: formatConfidence(rectM.confidence)},
        {key: "SU frames", value: cleanVal(suGiuCounts.su)},
        {key: "GIU frames", value: cleanVal(suGiuCounts.giu)},
        {key: "Other frames", value: cleanVal(suGiuCounts.other)},
        {key: "SU/GIU frame etichettati", value: cleanVal(suGiuPer.images_total || row.su_giu_images_predicted)},
        {key: "LR marker majority", value: cleanVal(row.lr_marker_majority_label)},
        {key: "LR marker best", value: cleanVal(row.lr_marker_best_label), conf: formatConfidence(row.lr_marker_best_score)},
        {key: "LR marker strategia", value: cleanVal(row.lr_marker_best_search_strategy)},
        {key: "LR marker frame", value: cleanVal(row.lr_marker_images_predicted)},
        {key: "L frames", value: cleanVal(ltCounts.l)},
        {key: "T frames", value: cleanVal(ltCounts.t)},
        {key: "LT other frames", value: cleanVal(ltCounts.other)},
        {key: "L/T frame etichettati", value: cleanVal(ltPer.images_total || row.lt_images_predicted)},
        {key: "Rect conf method", value: cleanVal(rectM.method)},
        {key: "Rect source", value: cleanVal(rectM.source)},
        {key: "Rect frames usati", value: cleanVal(rectM.imagesUsed)},
        {key: "Rect size px (WxH)", value: `${cleanVal(rectM.width)} x ${cleanVal(rectM.height)}`},
        {key: "Rect area ratio", value: formatNum(rectM.areaRatio, 4)},
        {key: "Rect aspect ratio", value: formatNum(rectM.aspect, 4)},
        {key: "Rect confidence note", value: cleanVal(rectM.note)},
        {key: "line13 rect_name_echo", value: cleanVal(row.line_13_rect_name_echo || l13.value)},
        {key: "line14 rect_name_probe", value: cleanVal(row.line_14_rect_name_probe || l14.value)},
        {key: "Status predizione", value: cleanVal(row.line_90_status_prediction)},
        {key: "Reason", value: cleanVal(row.line_91_reason)},
        {key: "Rotazione (deg)", value: cleanVal(rot.rotation_deg_clockwise)},
        {key: "Raw images", value: cleanVal(summary.raw_images_count)},
        {key: "Duplicati rimossi", value: cleanVal(summary.duplicates_removed_count_recomputed)},
      ];
      setPredictionSummaryHtml(`
        <div class="evidence-card">
          <div class="evidence-title">Riepilogo Predizioni <span class="tag tag-data">overview</span></div>
          <div class="summary-grid">
            ${cards.map((item) => renderSummaryCard(item)).join("")}
          </div>
        </div>
      `);
    }

    function renderAnalysisReviewSummary(summary) {
      if (!analysisReviewSummaryEl && !analysisReviewSummaryRailEl) return;
      if (!summary || typeof summary !== "object") {
        setAnalysisReviewSummaryHtml(`
          <div class="evidence-card">
            <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
            <div class="small">Seleziona una run completata.</div>
          </div>
        `);
        return;
      }

      const row = (summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const checks = Array.isArray(summary.checks) ? summary.checks : [];
      const de = (summary.decision_evidence && typeof summary.decision_evidence === "object")
        ? summary.decision_evidence
        : buildLegacyDecisionEvidence(summary);
      const rec = (summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const vendor = (de.vendor && typeof de.vendor === "object") ? de.vendor : {};
      const probe = (de.probe && typeof de.probe === "object") ? de.probe : {};
      const suGiu = (de.su_giu && typeof de.su_giu === "object") ? de.su_giu : {};
      const suGiuPer = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object") ? rec.su_giu_per_image : {};
      const ltPer = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
      const rectM = computeRectMetrics(row);

      const checksNeedReview = checks.filter((step) => {
        const st = String((step && step.status) || "review").trim().toLowerCase();
        return !(st === "ok" || st === "completed");
      });
      const reviewItemsHtml = checksNeedReview.length
        ? checksNeedReview.map((step, idx) => {
            const stepName = String((step && step.step) || `step_${idx + 1}`);
            const status = String((step && step.status) || "review");
            const details = Object.entries(step || {})
              .filter(([k]) => k !== "step" && k !== "status")
              .slice(0, 3)
              .map(([k, v]) => `${k}: ${v}`)
              .join(" | ");
            return `
              <div class="analysis-review-item">
                <div class="head">
                  <span class="name">${esc(stepName)}</span>
                  <span class="status-pill ${statusClass(status)}">${esc(status)}</span>
                </div>
                <div class="small">${esc(details || "Nessun dettaglio extra.")}</div>
              </div>
            `;
          }).join("")
        : `
          <div class="analysis-review-item">
            <div class="head">
              <span class="name">Nessuno step in review</span>
              <span class="status-pill status-ok">ok</span>
            </div>
            <div class="small">Tutti gli step risultano ok/completed.</div>
          </div>
        `;

      const vendorConf = (vendor.confidence !== undefined && vendor.confidence !== null) ? vendor.confidence : row.vendor_confidence;
      const probeConf = (probe.confidence !== undefined && probe.confidence !== null) ? probe.confidence : row.line_03_probe_confidence;
      const probeNameResolved = String(probe.predicted_name || row.line_03_probe_name || "").trim();
      const suGiuConf = (suGiu.mean_confidence !== undefined && suGiu.mean_confidence !== null)
        ? suGiu.mean_confidence
        : ((suGiuPer.mean_confidence !== undefined && suGiuPer.mean_confidence !== null) ? suGiuPer.mean_confidence : suGiuPer.avg_confidence);
      const ltConf = (ltPer.mean_confidence !== undefined && ltPer.mean_confidence !== null)
        ? ltPer.mean_confidence
        : row.lt_mean_confidence;

      const modelItems = [
        {
          name: "Vendor",
          source: normalizeDecisionSourceKind(vendor.source_hint, !!vendor.ocr_used),
          confidence: formatConfidence(vendorConf),
          checkpoint: String(model.vendor_checkpoint || ""),
        },
        {
          name: "Rect line11",
          source: String(row.line_11_source || model.rect_checkpoint_source || "").trim() || "model",
          confidence: formatConfidence(rectM.confidence),
          checkpoint: String(model.rect_checkpoint_used || row.line_11_model_checkpoint || model.rect_checkpoint_global || ""),
        },
        {
          name: probeNameResolved ? `Probe (${probeNameResolved})` : "Probe",
          source: normalizeDecisionSourceKind(probe.source_hint, !!probe.ocr_used),
          confidence: formatConfidence(probeConf),
          checkpoint: String(model.probe_checkpoint || ""),
        },
        {
          name: "SU/GIU",
          source: String(model.su_giu_enabled ? "cnn" : "disabled"),
          confidence: formatConfidence(suGiuConf),
          checkpoint: String(model.su_giu_checkpoint_used || model.su_giu_checkpoint_global || ""),
        },
        {
          name: "LR marker",
          source: String(row.lr_marker_source || (model.lr_marker_classical_enabled ? "classical_vendor_template_library" : "disabled")),
          confidence: formatConfidence(row.lr_marker_best_score),
          checkpoint: String(row.lr_marker_best_template_path || ""),
        },
        {
          name: "L/T",
          source: String(model.lt_enabled ? "cnn" : "disabled"),
          confidence: formatConfidence(ltConf),
          checkpoint: String(model.lt_checkpoint_used || model.lt_checkpoint_global || ""),
        },
      ];

      const modelItemsHtml = modelItems.map((it) => {
        const cp = String(it.checkpoint || "").trim();
        const cpName = checkpointBaseName(cp) || "-";
        return `
          <div class="analysis-model-item">
            <div class="head">
              <span class="name">${esc(it.name)}</span>
              <span class="status-pill ${it.confidence ? "status-ok" : "status-review"}">${esc(it.confidence || "-")}</span>
            </div>
            <div class="kvline"><b>fonte:</b> ${esc(it.source || "-")}</div>
            <div class="kvline"><b>rete:</b> ${esc(cpName)}</div>
            <div class="path">${esc(cp || "-")}</div>
          </div>
        `;
      }).join("");

      setAnalysisReviewSummaryHtml(`
        <div class="evidence-card">
          <div class="evidence-title">Riepilogo Review <span class="tag tag-analysis">review</span></div>
          <div class="small">Step da rivedere: <b>${checksNeedReview.length}</b> / ${checks.length}</div>
          <div class="analysis-review-list">${reviewItemsHtml}</div>
        </div>
        <div class="evidence-card">
          <div class="evidence-title">Reti + Confidenza <span class="tag tag-model">models</span></div>
          <div class="analysis-model-list">${modelItemsHtml}</div>
        </div>
      `);
    }

    function renderDecisionEvidence(summary) {
      decisionEvidenceEl.innerHTML = "";
      const de = (summary && summary.decision_evidence && typeof summary.decision_evidence === "object")
        ? summary.decision_evidence
        : buildLegacyDecisionEvidence(summary);
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object")
        ? rec.model_evidence
        : {};
      const row = (summary && typeof summary === "object" && summary.pipeline_row && typeof summary.pipeline_row === "object")
        ? summary.pipeline_row
        : {};
      const vendor = (de.vendor && typeof de.vendor === "object") ? de.vendor : {};
      const probe = (de.probe && typeof de.probe === "object") ? de.probe : {};
      const suGiuPer = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object") ? rec.su_giu_per_image : {};
      const suGiuSplit = (rec.su_giu_split_folders && typeof rec.su_giu_split_folders === "object") ? rec.su_giu_split_folders : {};
      const lrMarker = (rec.lr_marker_per_image && typeof rec.lr_marker_per_image === "object") ? rec.lr_marker_per_image : {};
      const suGiuCounts = (suGiuSplit.counts && typeof suGiuSplit.counts === "object")
        ? suGiuSplit.counts
        : ((suGiuPer.label_counts && typeof suGiuPer.label_counts === "object") ? suGiuPer.label_counts : {});
      const ltPer = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
      const ltSplit = (rec.lt_split_folders && typeof rec.lt_split_folders === "object") ? rec.lt_split_folders : {};
      const ltCounts = (ltSplit.counts && typeof ltSplit.counts === "object")
        ? ltSplit.counts
        : ((ltPer.label_counts && typeof ltPer.label_counts === "object") ? ltPer.label_counts : {});
      const suFolderAbs = String(suGiuSplit.su_dir_abs || "").trim();
      const giuFolderAbs = String(suGiuSplit.giu_dir_abs || "").trim();
      const suFolderRel = String(suGiuSplit.su_dir_rel || "").trim();
      const giuFolderRel = String(suGiuSplit.giu_dir_rel || "").trim();
      const suGiuFramesTotal = Number(suGiuPer.images_total || 0);
      const lrMarkerBest = (lrMarker.best && typeof lrMarker.best === "object") ? lrMarker.best : {};
      const lrMarkerCounts = (lrMarker.label_counts && typeof lrMarker.label_counts === "object") ? lrMarker.label_counts : {};
      const lrMarkerStrategies = (lrMarker.search_strategy_counts && typeof lrMarker.search_strategy_counts === "object") ? lrMarker.search_strategy_counts : {};
      const lrMarkerTemplateEffective = String(lrMarker.template_policy_effective || lrMarkerBest.template_policy_effective || row.lr_marker_template_policy_effective || "").trim();
      const lrMarkerTemplateFallbackReason = String(lrMarker.template_fallback_reason || lrMarkerBest.template_fallback_reason || row.lr_marker_template_fallback_reason || "").trim();
      const lrMarkerFallbackHtml = lrMarkerTemplateFallbackReason
        ? `<br><span class="tag tag-warn">fallback attivo</span> effective=${esc(cleanVal(lrMarkerTemplateEffective || "derived_folder"))}<br>motivo: ${esc(lrMarkerTemplateFallbackReason)}`
        : (lrMarkerTemplateEffective ? `<br>effective=${esc(cleanVal(lrMarkerTemplateEffective))}` : "");
      const lrMarkerStrategiesTxt = Object.entries(lrMarkerStrategies)
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => `${k}:${v}`)
        .join(" | ");
      const lFolderAbs = String(ltSplit.l_dir_abs || "").trim();
      const tFolderAbs = String(ltSplit.t_dir_abs || "").trim();
      const lFolderRel = String(ltSplit.l_dir_rel || "").trim();
      const tFolderRel = String(ltSplit.t_dir_rel || "").trim();
      const ltFramesTotal = Number(ltPer.images_total || row.lt_images_predicted || 0);
      const vendorOcrElements = (vendor.ocr_elements && typeof vendor.ocr_elements === "object") ? vendor.ocr_elements : {};
      const probeOcrElements = (probe.ocr_elements && typeof probe.ocr_elements === "object") ? probe.ocr_elements : {};
      const vendorOcrTxt = Object.keys(vendorOcrElements).length
        ? Object.entries(vendorOcrElements).map(([k, v]) => `${k}: ${v}`).join(" | ")
        : "-";
      const probeOcrTxt = Object.keys(probeOcrElements).length
        ? Object.entries(probeOcrElements).map(([k, v]) => `${k}: ${v}`).join(" | ")
        : "-";
      const probeLabel = [String(probe.predicted_id || ""), String(probe.predicted_name || "")]
        .filter(Boolean)
        .join(" - ");
      const vendorSourceKind = normalizeDecisionSourceKind(vendor.source_hint, !!vendor.ocr_used);
      const probeSourceKind = normalizeDecisionSourceKind(probe.source_hint, !!probe.ocr_used);
      const vendorThreshold = asNumber(
        vendor.threshold,
        asNumber(row.vendor_threshold, asNumber(summary && summary.vendor_min_confidence, 0.5)),
      );
      const probeThreshold = asNumber(
        probe.threshold,
        asNumber(row.probe_threshold, asNumber(summary && summary.probe_min_confidence, 0.5)),
      );
      const vendorTopChoices = (Array.isArray(vendor.top_choices) && vendor.top_choices.length)
        ? vendor.top_choices
        : parseTopChoices(row.vendor_top3_json || row.vendor_topk_json || row.vendor_topk || "");
      const probeTopChoices = (Array.isArray(probe.top_choices) && probe.top_choices.length)
        ? probe.top_choices
        : parseTopChoices(row.probe_top3_json || row.probe_topk_json || row.probe_topk || "");
      if (!vendorTopChoices.length && String(vendor.predicted_name || "").trim()) {
        vendorTopChoices.push({
          label: String(vendor.predicted_name || "").trim(),
          confidence: asNumber(vendor.confidence, NaN),
        });
      }
      if (!probeTopChoices.length && String(probe.predicted_id || "").trim()) {
        probeTopChoices.push({
          label: String(probe.predicted_id || "").trim(),
          name: String(probe.predicted_name || "").trim(),
          confidence: asNumber(probe.confidence, NaN),
        });
      }
      const vendorDecisionReason = String(vendor.decision_reason || "").trim()
        || buildDecisionReasonFallback(vendorSourceKind, asNumber(vendor.confidence, NaN), vendorThreshold, "vendor");
      const probeDecisionReason = String(probe.decision_reason || "").trim()
        || buildDecisionReasonFallback(probeSourceKind, asNumber(probe.confidence, NaN), probeThreshold, "probe");
      const vendorSamples = Array.isArray(rec.vendor_probe_samples_rel) ? rec.vendor_probe_samples_rel : [];
      const vendorSamplesHtml = vendorSamples.length
        ? `<div class="mini-grid">${
            vendorSamples.slice(0, 6).map((rel, idx) => `
              <div class="pair-item">
                <div class="label">sample ${idx + 1}</div>
                <img src="${imgUrl(String(rel || ""))}" loading="lazy">
                <div class="path">${esc(String(rel || ""))}</div>
              </div>
            `).join("")
          }</div>`
        : `<div class="small">Nessun sample vendor/probe disponibile.</div>`;
      const rectOverlay = (rec.rect_overlay && typeof rec.rect_overlay === "object") ? rec.rect_overlay : {};
      const rectOverlayRel = String(rectOverlay.overlay_rel || "");
      const rectOverlayHtml = rectOverlayRel
        ? `
          <div class="pair-item">
            <div class="label">Rect line11 overlay</div>
            <img src="${imgUrl(rectOverlayRel)}" loading="lazy">
            <div class="path">${esc(rectOverlayRel)}</div>
          </div>
        `
        : `<div class="small">Nessun overlay rettangolo disponibile.</div>`;
      const l13 = (rec.line13_template && typeof rec.line13_template === "object") ? rec.line13_template : {};
      const l14 = (rec.line14_template && typeof rec.line14_template === "object") ? rec.line14_template : {};
      const l13OverlayRel = String(l13.overlay_rel || "");
      const l14OverlayRel = String(l14.overlay_rel || "");
      const line13TailPolicy = String(l13.tail_policy || model.line13_tail_policy || "").trim() || "disabled_after_coords";
      const line13VendorMapPath = String(model.line13_vendor_map_path || "").trim();
      const line13VendorMapCount = (model.line13_vendor_map_loaded && typeof model.line13_vendor_map_loaded === "object")
        ? Object.keys(model.line13_vendor_map_loaded).length
        : 0;
      const line13VendorMinConf = asNumber(model.line13_vendor_min_confidence, NaN);
      const line13PostEnabled = !!model.line13_postprocess_enabled;
      const line13PostIouThr = asNumber(model.line13_postprocess_iou_threshold, NaN);
      const line13PostMinKeep = Number.isFinite(asNumber(model.line13_postprocess_min_keep, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_min_keep, NaN)))
        : "-";
      const line13PostBoxesTotal = Number.isFinite(asNumber(model.line13_postprocess_boxes_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_total, NaN)))
        : "-";
      const line13PostBoxesKept = Number.isFinite(asNumber(model.line13_postprocess_boxes_kept_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_kept_total, NaN)))
        : "-";
      const line13PostBoxesDropped = Number.isFinite(asNumber(model.line13_postprocess_boxes_dropped_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_dropped_total, NaN)))
        : "-";
      const line13PostModeCountsObj = (model.line13_postprocess_mode_counts && typeof model.line13_postprocess_mode_counts === "object")
        ? model.line13_postprocess_mode_counts
        : {};
      const line13PostModeCountsTxt = Object.entries(line13PostModeCountsObj)
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => `${k}:${v}`)
        .join(" | ");
      const modelSplitMock = `<span class="split-mock" title="Mock non implementato">split dataset: mock disabilitato</span>`;
      const suGiuClassNamesTxt = Array.isArray(model.su_giu_class_names)
        ? model.su_giu_class_names.map((x) => String(x || "").trim()).filter(Boolean).join(", ")
        : "";
      const rectRouting = getRectRoutingDecision(summary);
      const rectRoutingLevelClass = rectRouting.level === "warn" ? "warn" : (rectRouting.level === "ok" ? "ok" : "info");
      const vendorProbeHelp = helpTipHtml(
        "Come leggere Vendor / Probe Decision",
        [
          "predetto/predetta: classe scelta dal classificatore (o da fallback OCR/manuale).",
          "conf: confidenza della scelta finale; confrontala con la soglia usata.",
          "fonte decisione: indica se arriva da CNN pura, OCR fallback o intervento utente.",
          "top-3 alternative: classi candidate più probabili utili per audit.",
          "motivo scelta: spiegazione sintetica del percorso decisionale.",
        ],
      );
      const suGiuHelp = helpTipHtml(
        "SU/GIU per frame",
        [
          "Qui non c'è un giudizio unico di cartella: è una classificazione frame-by-frame.",
          "I conteggi su/giu/other servono per verificare qualità e rumore della sequenza.",
          "Le cartelle operative SU/GIU mostrano dove sono stati salvati i frame smistati.",
        ],
      );
      const lrMarkerHelp = helpTipHtml(
        "LR marker classico",
        [
          "Step finale dopo vendor, rettangolo eco e SU/GIU.",
          "Cerca tutti i template DB_echo del vendor e tiene per ogni frame quello con score massimo.",
          "Se lo score è basso prova full-crop e poi espansioni progressive fuori crop.",
        ],
      );
      const ltHelp = helpTipHtml(
        "L/T per frame",
        [
          "Classificazione L/T eseguita sui crop del rettangolo ecografico.",
          "majority e mean_conf riassumono il comportamento medio dei frame della cartella.",
          "Se i conteggi sono molto misti, conviene rivedere frame e predizione probe.",
        ],
      );
      const rectRoutingHelp = helpTipHtml(
        "Routing rettangolo line11",
        [
          "Mostra se è stata usata la rete globale o quella specifica vendor.",
          "Vendor conf / soglia spiegano perché il routing ha scelto quella rete.",
          "Confronta checkpoint usato e checkpoint specifico per verificare il path atteso.",
        ],
      );
      const modelsHelp = helpTipHtml(
        "Modelli usati",
        [
          "checkpoint (globale) è il modello base disponibile.",
          "checkpoint (usato) è quello effettivamente applicato nella run.",
          "route e image_size sono utili per verificare il routing e la compatibilità con il training.",
        ],
      );
      const rectOverlayHelp = helpTipHtml(
        "Rettangolo Ecografico (linea 11)",
        [
          "Verde = GT/target quando disponibile in evidenza.",
          "Rosso o altre tracce = predizione/fasi di post-processing.",
          "Usa questo pannello per capire velocemente se il box operativo è plausibile.",
        ],
      );
      decisionEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Vendor / Probe Decision ${vendorProbeHelp} <span class="tag tag-vendor">vendor</span><span class="tag tag-probe">probe</span></div>
          <div class="detail-list">
            <div>
              <b>Vendor</b><br>
              predetto: ${esc(vendor.predicted_name || "-")}<br>
              conf: ${esc(((vendor.confidence !== undefined && vendor.confidence !== null) ? vendor.confidence : "-").toString())}<br>
              fonte decisione: <b>${esc(vendorSourceKind)}</b><br>
              soglia usata: ${esc(Number.isFinite(vendorThreshold) ? formatNum(vendorThreshold, 4) : "-")}<br>
              top-3 alternative:<br>${renderTopChoicesHtml(vendorTopChoices, "vendor")}<br>
              motivo scelta: ${esc(vendorDecisionReason)}<br>
              OCR usato: <b>${vendor.ocr_used ? "SI" : "NO"}</b><br>
              source: ${esc(vendor.source_hint || "-")}<br>
              commento: ${esc(vendor.ocr_comment || "-")}<br>
              elementi OCR: ${esc(vendorOcrTxt)}
            </div>
            <div>
              <b>Probe</b><br>
              predetta: ${esc(probeLabel || "-")}<br>
              conf: ${esc(((probe.confidence !== undefined && probe.confidence !== null) ? probe.confidence : "-").toString())}<br>
              fonte decisione: <b>${esc(probeSourceKind)}</b><br>
              soglia usata: ${esc(Number.isFinite(probeThreshold) ? formatNum(probeThreshold, 4) : "-")}<br>
              top-3 alternative:<br>${renderTopChoicesHtml(probeTopChoices, "probe")}<br>
              motivo scelta: ${esc(probeDecisionReason)}<br>
              OCR usato: <b>${probe.ocr_used ? "SI" : "NO"}</b><br>
              source: ${esc(probe.source_hint || "-")}<br>
              commento: ${esc(probe.ocr_comment || "-")}<br>
              elementi OCR: ${esc(probeOcrTxt)}
            </div>
          </div>
          <div class="evidence-title" style="margin-top:10px;">Orientamento SU/GIU per frame ${suGiuHelp} <span class="tag tag-rotation">su/giu</span></div>
          <div class="detail-list">
            <div>
              <b>Smistamento immagini (nessun giudizio globale)</b><br>
              frame etichettati: ${esc(String(suGiuFramesTotal || "-"))}<br>
              su: ${esc(String(cleanVal(suGiuCounts.su)))}<br>
              giu: ${esc(String(cleanVal(suGiuCounts.giu)))}<br>
              other: ${esc(String(cleanVal(suGiuCounts.other)))}<br>
              vai alla tab evidenze <b>SU/GIU</b> per vedere le immagini divise in due cartelle.
            </div>
            <div>
              <b>Cartelle operative</b><br>
              su: <span class="path">${esc(suFolderRel || suFolderAbs || "-")}</span><br>
              giu: <span class="path">${esc(giuFolderRel || giuFolderAbs || "-")}</span><br>
              ${suFolderAbs ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(suFolderAbs)}" style="margin-top:6px;">Apri Finder SU</button>` : ""}
              ${giuFolderAbs ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(giuFolderAbs)}" style="margin-top:6px; margin-left:6px;">Apri Finder GIU</button>` : ""}
            </div>
          </div>
          <div class="evidence-title" style="margin-top:10px;">LR marker classico ${lrMarkerHelp} <span class="tag tag-rotation">lr</span></div>
          <div class="detail-list">
            <div>
              <b>Risultato cartella</b><br>
              frame con marker: ${esc(String(cleanVal(row.lr_marker_images_predicted || lrMarker.images_total || "-")))}<br>
              majority: ${esc(String(cleanVal(row.lr_marker_majority_label || "-")))} | vote_ratio=${esc(String(cleanVal(row.lr_marker_majority_vote_ratio || "-")))}<br>
              best: ${esc(String(cleanVal(row.lr_marker_best_label || lrMarkerBest.lr_label || "-")))} | score=${esc(String(cleanVal(row.lr_marker_best_score || lrMarkerBest.match_score || "-")))}<br>
              strategia best: ${esc(String(cleanVal(row.lr_marker_best_search_strategy || lrMarkerBest.search_strategy || "-")))}<br>
              source: ${esc(String(cleanVal(row.lr_marker_source || "-")))}
            </div>
            <div>
              <b>Template e fallback</b><br>
              template best: <span class="path">${esc(String(row.lr_marker_best_template_path || lrMarkerBest.template_path || "-"))}</span><br>
              ${lrMarkerFallbackHtml ? `${lrMarkerFallbackHtml}<br>` : ""}
              counts: not_lr_flipped=${esc(String(cleanVal(lrMarkerCounts.not_lr_flipped)))} | lr_flipped=${esc(String(cleanVal(lrMarkerCounts.lr_flipped)))} | other=${esc(String(cleanVal(lrMarkerCounts.other)))}<br>
              strategie: ${esc(lrMarkerStrategiesTxt || "-")}<br>
              vai alla tab evidenze <b>LR marker</b> per vedere score/template frame-per-frame.
            </div>
          </div>
          <div class="evidence-title" style="margin-top:10px;">Classificazione L/T per frame ${ltHelp} <span class="tag tag-rotation">lt</span></div>
          <div class="detail-list">
            <div>
              <b>Classificazione su crop rettangolo</b><br>
              frame etichettati: ${esc(String(ltFramesTotal || "-"))}<br>
              L: ${esc(String(cleanVal(ltCounts.l)))}<br>
              T: ${esc(String(cleanVal(ltCounts.t)))}<br>
              other: ${esc(String(cleanVal(ltCounts.other)))}<br>
              majority: ${esc(String(cleanVal(row.lt_majority_label || "-")))} | mean_conf=${esc(String(cleanVal(row.lt_mean_confidence || "-")))}
            </div>
            <div>
              <b>Cartelle operative</b><br>
              L: <span class="path">${esc(lFolderRel || lFolderAbs || "-")}</span><br>
              T: <span class="path">${esc(tFolderRel || tFolderAbs || "-")}</span><br>
              ${lFolderAbs ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(lFolderAbs)}" style="margin-top:6px;">Apri Finder L</button>` : ""}
              ${tFolderAbs ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(tFolderAbs)}" style="margin-top:6px; margin-left:6px;">Apri Finder T</button>` : ""}
            </div>
          </div>
          <div class="evidence-title" style="margin-top:10px;">Modelli usati ${modelsHelp} <span class="tag tag-model">models</span></div>
          <div class="detail-list">
            <div><b>Vendor checkpoint</b><br><span class="path">${esc(String(model.vendor_checkpoint || "-"))}</span><br>${modelSplitMock}</div>
            <div><b>Probe checkpoint</b><br><span class="path">${esc(String(model.probe_checkpoint || "-"))}</span><br>${modelSplitMock}</div>
            <div><b>Rect checkpoint (globale)</b><br><span class="path">${esc(String(model.rect_checkpoint_global || "-"))}</span><br>${modelSplitMock}</div>
            <div><b>Rect checkpoint (usato)</b><br><span class="path">${esc(String(model.rect_checkpoint_used || ((summary && summary.pipeline_row && summary.pipeline_row.line_11_model_checkpoint) || "-")))}</span><br>${modelSplitMock}</div>
            <div><b>Line13 checkpoint (usato)</b><br><span class="path">${esc(String(model.line13_checkpoint_used || model.line13_checkpoint_global || "-"))}</span><br>enabled=${esc(String(model.line13_model_enabled ? "SI" : "NO"))} | route=${esc(cleanVal(model.line13_model_route || l13.model_route || "-"))} | image_size=${esc(cleanVal(model.line13_image_size || "-"))}</div>
            <div><b>SU/GIU checkpoint (usato)</b><br><span class="path">${esc(String(model.su_giu_checkpoint_used || model.su_giu_checkpoint_global || "-"))}</span><br>enabled=${esc(String(model.su_giu_enabled ? "SI" : "NO"))} | image_size=${esc(cleanVal(model.su_giu_image_size))} | batch=${esc(cleanVal(model.su_giu_batch_size))} | classi=${esc(suGiuClassNamesTxt || "-")}</div>
            <div><b>LR marker classico</b><br><span class="path">${esc(String(row.lr_marker_best_template_path || lrMarkerBest.template_path || "-"))}</span><br>enabled=${esc(String(model.lr_marker_classical_enabled ? "SI" : "NO"))} | template=${esc(cleanVal(model.lr_marker_template_policy || "-"))} | soglia_full_crop=${esc(cleanVal(model.lr_marker_full_crop_fallback_threshold || "-"))} | soglia_expand=${esc(cleanVal(model.lr_marker_expanded_search_threshold || "-"))}${lrMarkerFallbackHtml}</div>
            <div><b>L/T checkpoint (usato)</b><br><span class="path">${esc(String(model.lt_checkpoint_used || model.lt_checkpoint_global || "-"))}</span><br>enabled=${esc(String(model.lt_enabled ? "SI" : "NO"))} | image_size=${esc(cleanVal(model.lt_image_size))} | batch=${esc(cleanVal(model.lt_batch_size))} | classi=${esc(Array.isArray(model.lt_class_names) ? model.lt_class_names.join(", ") : "-")}</div>
          </div>
          <div class="small" style="margin-top:6px;">
            stage line11=${esc(String(model.pipeline_stage_line11_rect_echo || "-"))} |
            stage su_giu=${esc(String(model.pipeline_stage_su_giu_rect || "-"))} |
            stage lr_marker=${esc(String(model.pipeline_stage_lr_marker || "-"))} |
            stage lt=${esc(String(model.pipeline_stage_lt_rect || "-"))} |
            stage probe=${esc(String(model.pipeline_stage_probe || "-"))} |
            stage line13=${esc(String(model.pipeline_stage_line13_rect_name_echo || "-"))} |
            stage line14=${esc(String(model.pipeline_stage_line14_rect_name_probe || "-"))}
          </div>
          <div class="rect-routing-alert ${esc(rectRoutingLevelClass)}">
            <div class="rect-routing-title">Routing Rettangolo line11 ${rectRoutingHelp}</div>
            <div class="small">${esc(rectRouting.title)}</div>
            <div class="small">${esc(rectRouting.reason)}</div>
            <div class="rect-routing-kv" style="margin-top:6px;">
              <div class="kv"><div class="k">line11_source</div><div class="v">${esc(cleanVal(rectRouting.source))}</div></div>
              <div class="kv"><div class="k">Vendor conf / soglia</div><div class="v">${esc(Number.isFinite(rectRouting.vendor_confidence) ? formatNum(rectRouting.vendor_confidence, 4) : "-")} / ${esc(Number.isFinite(rectRouting.threshold) ? formatNum(rectRouting.threshold, 4) : "-")}</div></div>
              <div class="kv"><div class="k">Checkpoint usato</div><div class="v">${esc(checkpointBaseName(rectRouting.used_checkpoint) || cleanVal(rectRouting.used_checkpoint))}</div></div>
              <div class="kv"><div class="k">Checkpoint specifico vendor</div><div class="v">${esc(checkpointBaseName(rectRouting.specific_checkpoint) || cleanVal(rectRouting.specific_checkpoint))}</div></div>
            </div>
          </div>
          <div class="evidence-title" style="margin-top:10px;">Sample usati per Vendor/Probe <span class="tag tag-data">samples</span></div>
          ${vendorSamplesHtml}
          <div class="evidence-title" style="margin-top:10px;">Rettangolo Ecografico (linea 11) ${rectOverlayHelp} <span class="tag tag-rect">rect</span></div>
          ${rectOverlayHtml}
        </div>
      `;
    }

    function renderTemplateEvidence(summary) {
      if (!templateEvidenceEl) return;
      templateEvidenceEl.innerHTML = "";
      const sm = (summary && typeof summary === "object") ? summary : {};
      const row = (sm.pipeline_row && typeof sm.pipeline_row === "object") ? sm.pipeline_row : {};
      const rec = (sm.recognition_evidence && typeof sm.recognition_evidence === "object") ? sm.recognition_evidence : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const l13 = (rec.line13_template && typeof rec.line13_template === "object") ? rec.line13_template : {};
      const l14 = (rec.line14_template && typeof rec.line14_template === "object") ? rec.line14_template : {};

      const l13Value = String(l13.value || row.line_13_rect_name_echo || "-");
      const l13Source = String(l13.source || row.line_13_source || "-");
      const l13Support = String(l13.support || row.line_13_support || "-");
      const l13Route = String(l13.model_route || model.line13_model_route || "-");
      const l13Checkpoint = String(l13.model_checkpoint || model.line13_checkpoint_used || model.line13_checkpoint_global || "-");
      const l13TailPolicyRaw = String(l13.tail_policy || model.line13_tail_policy || "").trim();
      const l13TailPolicy = l13TailPolicyRaw === "disabled_after_coords" ? "DISABILITATO" : (l13TailPolicyRaw || "-");
      const l13SourceImageRel = String(l13.source_image_rel || "").trim();
      const l13OverlayRel = String(l13.overlay_rel || "").trim();
      const l13CropRel = String(l13.crop_rel || "").trim();
      const l13RawOverlayRel = String(
        l13.raw_overlay_rel || l13.model_overlay_rel || l13.pre_finetune_overlay_rel || ""
      ).trim();
      const l13RawCropRel = String(
        l13.raw_crop_rel || l13.model_crop_rel || l13.pre_finetune_crop_rel || ""
      ).trim();
      const l13RawOverlayDisplayRel = String(l13RawOverlayRel || l13OverlayRel || "").trim();
      const l13RawCropDisplayRel = String(l13RawCropRel || l13CropRel || "").trim();
      const l13RawOverlayFallback = !l13RawOverlayRel && !!l13RawOverlayDisplayRel;
      const l13RawCropFallback = !l13RawCropRel && !!l13RawCropDisplayRel;
      const l13FineTunedCropRel = String(
        l13.finetuned_crop_rel || l13.postprocess_crop_rel || l13.template_crop_rel || l13CropRel || ""
      ).trim();
      const l13FineTunedOverlayRel = String(
        l13.finetuned_overlay_rel || l13.postprocess_overlay_rel || l13.template_overlay_rel || l13OverlayRel || ""
      ).trim();

      const line13VendorMapPath = String(model.line13_vendor_map_path || "").trim();
      const line13VendorMapCount = (model.line13_vendor_map_loaded && typeof model.line13_vendor_map_loaded === "object")
        ? Object.keys(model.line13_vendor_map_loaded).length
        : 0;
      const line13VendorMinConf = asNumber(model.line13_vendor_min_confidence, NaN);
      const line13PostEnabled = !!model.line13_postprocess_enabled;
      const line13PostIouThr = asNumber(model.line13_postprocess_iou_threshold, NaN);
      const line13PostMinKeep = Number.isFinite(asNumber(model.line13_postprocess_min_keep, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_min_keep, NaN)))
        : "-";
      const line13PostBoxesTotal = Number.isFinite(asNumber(model.line13_postprocess_boxes_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_total, NaN)))
        : "-";
      const line13PostBoxesKept = Number.isFinite(asNumber(model.line13_postprocess_boxes_kept_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_kept_total, NaN)))
        : "-";
      const line13PostBoxesDropped = Number.isFinite(asNumber(model.line13_postprocess_boxes_dropped_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_postprocess_boxes_dropped_total, NaN)))
        : "-";
      const line13PostModeCountsObj = (model.line13_postprocess_mode_counts && typeof model.line13_postprocess_mode_counts === "object")
        ? model.line13_postprocess_mode_counts
        : {};
      const line13PostModeCountsTxt = Object.entries(line13PostModeCountsObj)
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => `${k}:${v}`)
        .join(" | ");

      const line13TplPostEnabled = !!model.line13_template_postprocess_enabled;
      const line13TplPostMaxImages = Number.isFinite(asNumber(model.line13_template_max_images, NaN))
        ? String(Math.trunc(asNumber(model.line13_template_max_images, NaN)))
        : "-";
      const line13TplPostSearchMargin = asNumber(model.line13_template_search_margin_pct, NaN);
      const line13TplPostMinStd = asNumber(model.line13_template_min_std, NaN);
      const line13TplPostMinNonblack = asNumber(model.line13_template_min_nonblack_ratio, NaN);
      const line13TplPostMinScore = asNumber(model.line13_template_min_score, NaN);
      const line13TplPostMinConsensusIou = asNumber(model.line13_template_min_consensus_iou, NaN);
      const line13TplPostMinIouWithModel = asNumber(model.line13_template_min_iou_with_model, NaN);
      const line13TplPostAttempted = Number.isFinite(asNumber(model.line13_template_postprocess_attempted_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_template_postprocess_attempted_total, NaN)))
        : "-";
      const line13TplPostApplied = Number.isFinite(asNumber(model.line13_template_postprocess_applied_total, NaN))
        ? String(Math.trunc(asNumber(model.line13_template_postprocess_applied_total, NaN)))
        : "-";
      const line13TplPostModeCountsObj = (model.line13_template_postprocess_mode_counts && typeof model.line13_template_postprocess_mode_counts === "object")
        ? model.line13_template_postprocess_mode_counts
        : {};
      const line13TplPostModeCountsTxt = Object.entries(line13TplPostModeCountsObj)
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => `${k}:${v}`)
        .join(" | ");
      const templateVendorHelp = helpTipHtml(
        "Template Vendor (linea 13)",
        [
          "value è il RECT_NAME_ECHO finale scritto in output.",
          "source indica quale percorso ha prodotto il box (rete, postprocess, fallback).",
          "route mostra se è stata usata la rete globale o vendor-specific.",
          "support vale 1.0 quando arriva dal modello; valori più bassi indicano fallback storico/manuale.",
        ],
      );
      const line13TemplateGateHelp = helpTipHtml(
        "Template post-processing gated",
        [
          "enabled: se NO, non prova il template matching e usa solo il box rete.",
          "attempted/applied: cartelle provate / cartelle in cui il template ha sostituito il box rete.",
          "max images: numero massimo frame usati per cercare il template per cartella.",
          "search margin %: ampiezza della zona di ricerca attorno al box rete (più alto = più tollerante).",
          "min std e min nonblack ratio: filtro qualità template (scarta template troppo uniformi/neri).",
          "min score: soglia minima di similarità NCC del match (più alto = più severo).",
          "min consensus iou: coerenza minima tra match su frame diversi della stessa cartella.",
          "min iou with model: distanza massima accettata dal box rete (guardrail anti-salto).",
          "mode counts: motivo finale per cartella (es. template_applied, template_rejected_gate, skip_template_missing).",
        ],
      );
      const templateProbeHelp = helpTipHtml(
        "Template Probe (linea 14)",
        [
          "value è il RECT_NAME_PROBE finale.",
          "source e support aiutano a capire se la scelta è robusta o da review.",
          "L'overlay mostra visivamente dove è stato letto il template probe.",
        ],
      );

      const l14Value = String(l14.value || row.line_14_rect_name_probe || "-");
      const l14Source = String(l14.source || row.line_14_source || "-");
      const l14Support = String(l14.support || row.line_14_support || "-");
      const l14OverlayRel = String(l14.overlay_rel || "").trim();

      templateEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Template <span class="tag tag-data">template</span></div>
          <div class="small">Sezione dedicata ai template (vendor/probe/orientation).</div>

          <div class="evidence-title" style="margin-top:10px;">Template Vendor (linea 13) ${templateVendorHelp} <span class="tag tag-model">line13</span></div>
          <div class="template-layout">
            <div class="template-main-stack">
              <div class="template-crop-card">
                <div class="template-crop-title">1) Trovato dalla rete (pre fine-tuning)</div>
                <div class="template-dual-item template-overlay-large">
                  <div class="label">Originale con overlay rete</div>
                  ${
                    l13RawOverlayDisplayRel
                      ? `<img src="${imgUrl(l13RawOverlayDisplayRel)}" loading="lazy" decoding="async">`
                      : `<div class="small">Overlay rete non disponibile.</div>`
                  }
                  <div class="path" style="margin-top:6px;">${esc(l13RawOverlayDisplayRel || "-")}</div>
                  ${
                    l13RawOverlayFallback
                      ? `<div class="small">Nota: overlay rete raw non salvato, uso overlay composito disponibile in run.</div>`
                      : ``
                  }
                </div>
                <div class="template-dual-grid template-crop-row">
                  <div class="template-dual-item">
                    <div class="label">Crop template (solo rete)</div>
                    ${
                      l13RawCropDisplayRel
                        ? `<img src="${imgUrl(l13RawCropDisplayRel)}" loading="lazy" decoding="async">`
                        : `<div class="small">Crop rete non disponibile.</div>`
                    }
                    <div class="path" style="margin-top:6px;">${esc(l13RawCropDisplayRel || "-")}</div>
                    ${
                      l13RawCropFallback
                        ? `<div class="small">Nota: crop rete raw non salvato, uso crop disponibile in run.</div>`
                        : ``
                    }
                  </div>
                </div>
                <div class="template-note">
                  Didascalia rete: rettangolo <b>azzurro</b> = box trovato dalla rete vendor.
                  <div class="template-legend">
                    <span class="chip"><span class="swatch azzurro"></span>Rete vendor (pre fine-tuning)</span>
                  </div>
                </div>
                <div class="path">source image: ${esc(l13SourceImageRel || "-")}</div>
              </div>
              <div class="template-crop-card">
                <div class="template-crop-title">2) Dopo fine-tuning / post-processing</div>
                <div class="template-dual-item template-overlay-large">
                  <div class="label">Originale con overlay post-processing</div>
                  ${
                    l13FineTunedOverlayRel
                      ? `<img src="${imgUrl(l13FineTunedOverlayRel)}" loading="lazy" decoding="async">`
                      : `<div class="small">Overlay post-processing non disponibile.</div>`
                  }
                  <div class="path" style="margin-top:6px;">${esc(l13FineTunedOverlayRel || "-")}</div>
                </div>
                <div class="template-dual-grid template-crop-row">
                  <div class="template-dual-item">
                    <div class="label">Crop template (post-processing)</div>
                    ${
                      l13FineTunedCropRel
                        ? `<img src="${imgUrl(l13FineTunedCropRel)}" loading="lazy" decoding="async">`
                        : `<div class="small">Crop post-processing non disponibile.</div>`
                    }
                    <div class="path" style="margin-top:6px;">${esc(l13FineTunedCropRel || "-")}</div>
                  </div>
                </div>
                <div class="template-note">
                  Didascalia post-processing: rettangolo <b>giallo</b> = box raffinato dopo fine-tuning/post-processing.
                  <div class="template-legend">
                    <span class="chip"><span class="swatch giallo"></span>Post-processing (fine-tuned)</span>
                    <span class="chip"><span class="swatch azzurro"></span>Rete iniziale (se visibile nello stesso overlay)</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="template-side-stack">
              <div class="template-mini-card">
                <b>Dati template vendor</b><br>
                value: ${esc(l13Value)}<br>
                source: ${esc(l13Source)}<br>
                support: ${esc(l13Support)}<br>
                route: ${esc(l13Route)}<br>
                checkpoint: <span class="path">${esc(l13Checkpoint)}</span><br>
                tail post-coordinate: <b>${esc(l13TailPolicy)}</b>
              </div>
              <div class="template-mini-card">
                <b>Routing vendor template</b><br>
                enabled: ${esc(String(model.line13_model_enabled ? "SI" : "NO"))}<br>
                soglia conf vendor: ${esc(Number.isFinite(line13VendorMinConf) ? formatNum(line13VendorMinConf, 4) : "-")}<br>
                map entries: ${esc(String(line13VendorMapCount))}<br>
                map path: <span class="path">${esc(line13VendorMapPath || "-")}</span><br>
                post-processing line13: ${esc(line13PostEnabled ? "SI" : "NO")}<br>
                pp iou threshold: ${esc(Number.isFinite(line13PostIouThr) ? formatNum(line13PostIouThr, 3) : "-")}<br>
                pp min keep: ${esc(line13PostMinKeep)}<br>
                pp boxes total/kept/dropped: ${esc(`${line13PostBoxesTotal}/${line13PostBoxesKept}/${line13PostBoxesDropped}`)}<br>
                pp mode counts: ${esc(line13PostModeCountsTxt || "-")}<br><br>
                <b>Template post-processing gated ${line13TemplateGateHelp}</b><br>
                enabled: ${esc(line13TplPostEnabled ? "SI" : "NO")}<br>
                attempted/applied: ${esc(`${line13TplPostAttempted}/${line13TplPostApplied}`)}<br>
                max images: ${esc(line13TplPostMaxImages)}<br>
                search margin %: ${esc(Number.isFinite(line13TplPostSearchMargin) ? formatNum(line13TplPostSearchMargin, 2) : "-")}<br>
                min std: ${esc(Number.isFinite(line13TplPostMinStd) ? formatNum(line13TplPostMinStd, 2) : "-")}<br>
                min nonblack ratio: ${esc(Number.isFinite(line13TplPostMinNonblack) ? formatNum(line13TplPostMinNonblack, 3) : "-")}<br>
                min score: ${esc(Number.isFinite(line13TplPostMinScore) ? formatNum(line13TplPostMinScore, 3) : "-")}<br>
                min consensus iou: ${esc(Number.isFinite(line13TplPostMinConsensusIou) ? formatNum(line13TplPostMinConsensusIou, 3) : "-")}<br>
                min iou with model: ${esc(Number.isFinite(line13TplPostMinIouWithModel) ? formatNum(line13TplPostMinIouWithModel, 3) : "-")}<br>
                mode counts: ${esc(line13TplPostModeCountsTxt || "-")}
              </div>
              <div class="template-mini-card">
                <b>Legenda Colori Overlay</b><br>
                azzurro: box rete vendor (pre fine-tuning)<br>
                giallo: box dopo post-processing/fine-tuning<br>
                note: in alcune run può essere visibile un solo colore, in base agli artifact salvati.
              </div>
            </div>
          </div>

          <div class="evidence-title" style="margin-top:10px;">Template Probe (linea 14) ${templateProbeHelp} <span class="tag tag-probe">line14</span></div>
          <div class="detail-list">
            <div>
              value: ${esc(l14Value)}<br>
              source: ${esc(l14Source)}<br>
              support: ${esc(l14Support)}
            </div>
            <div>
              ${
                l14OverlayRel
                  ? `<div class="pair-item"><img src="${imgUrl(l14OverlayRel)}" loading="lazy"><div class="path">${esc(l14OverlayRel)}</div></div>`
                  : `<div class="small">Placeholder: preview probe template non ancora estesa.</div>`
              }
            </div>
          </div>

          <div class="evidence-title" style="margin-top:10px;">Template Orientation <span class="tag tag-rotation">orientation</span></div>
          <div class="small">Placeholder: sezione pronta per integrazione template orientation.</div>
        </div>
      `;
    }

    function checkpointBaseName(pathValue) {
      const txt = String(pathValue || "").trim();
      if (!txt) return "";
      const parts = txt.split(/[\\\\/]+/).filter(Boolean);
      return parts.length ? parts[parts.length - 1] : txt;
    }

    function getRectRoutingDecision(summary) {
      const sm = (summary && typeof summary === "object") ? summary : {};
      const row = (sm.pipeline_row && typeof sm.pipeline_row === "object") ? sm.pipeline_row : {};
      const rec = (sm.recognition_evidence && typeof sm.recognition_evidence === "object") ? sm.recognition_evidence : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const rectMap = (model.rect_vendor_map_loaded && typeof model.rect_vendor_map_loaded === "object")
        ? model.rect_vendor_map_loaded
        : {};

      const vendorName = String(row.vendor_predicted || "").trim();
      const vendorConf = asNumber(row.vendor_confidence, NaN);
      const rectVendorThreshold = asNumber(
        model.rect_vendor_min_confidence,
        asNumber(sm.rect_vendor_min_confidence, 0.7),
      );
      const source = String(row.line_11_source || "").trim();
      const usedCheckpoint = String(model.rect_checkpoint_used || row.line_11_model_checkpoint || "").trim();
      const globalCheckpoint = String(model.rect_checkpoint_global || "").trim();
      let specificCheckpoint = "";
      if (vendorName && Object.prototype.hasOwnProperty.call(rectMap, vendorName)) {
        specificCheckpoint = String(rectMap[vendorName] || "").trim();
      }
      const specificAvailable = !!specificCheckpoint;
      const comparePossible = !!globalCheckpoint && !!specificCheckpoint;
      const underThreshold = Number.isFinite(vendorConf) && Number.isFinite(rectVendorThreshold)
        ? vendorConf < rectVendorThreshold
        : false;

      let level = "info";
      let title = "Routing rect line11 non deterministico.";
      if (source === "global_low_vendor_conf") {
        level = "warn";
        title = "Rete rect globale forzata: confidence vendor sotto soglia.";
      } else if (source === "vendor_specialized") {
        level = "ok";
        title = "Rete rect specifica vendor usata correttamente.";
      } else if (source === "global" && specificAvailable) {
        level = "warn";
        title = "Rete rect globale usata anche se esiste una specifica vendor.";
      } else if (source === "global") {
        level = "info";
        title = "Rete rect globale usata (nessuna specifica disponibile).";
      }

      const reasonParts = [];
      reasonParts.push(`vendor=${vendorName || "-"} conf=${Number.isFinite(vendorConf) ? formatNum(vendorConf, 4) : "-"}`);
      reasonParts.push(`soglia_specifica=${Number.isFinite(rectVendorThreshold) ? formatNum(rectVendorThreshold, 4) : "-"}`);
      reasonParts.push(`line11_source=${source || "-"}`);
      if (source === "global_low_vendor_conf") reasonParts.push("motivo: conf vendor inferiore alla soglia.");
      if (source === "vendor_specialized") reasonParts.push("motivo: conf vendor sopra soglia e specifica disponibile.");
      if (source === "global" && specificAvailable && underThreshold) reasonParts.push("nota: conf vendor sotto soglia.");
      if (source === "global" && specificAvailable && !underThreshold) reasonParts.push("nota: verifica logica routing/vendor.");

      return {
        vendor_name: vendorName,
        vendor_confidence: vendorConf,
        threshold: rectVendorThreshold,
        source,
        used_checkpoint: usedCheckpoint,
        global_checkpoint: globalCheckpoint,
        specific_checkpoint: specificCheckpoint,
        specific_available: specificAvailable,
        compare_possible: comparePossible,
        under_threshold: underThreshold,
        level,
        title,
        reason: reasonParts.join(" "),
      };
    }

    function getRectVendorCompareState(runId) {
      const key = String(runId || "").trim();
      if (!key) return {status: "idle", payload: null, error: "", updated_at: ""};
      const current = rectVendorCompareStateByRun[key];
      if (!current || typeof current !== "object") {
        return {status: "idle", payload: null, error: "", updated_at: ""};
      }
      return {
        status: String(current.status || "idle"),
        payload: (current.payload && typeof current.payload === "object") ? current.payload : null,
        error: String(current.error || ""),
        updated_at: String(current.updated_at || ""),
      };
    }

    function setRectVendorCompareState(runId, nextState) {
      const key = String(runId || "").trim();
      if (!key) return;
      const prev = getRectVendorCompareState(key);
      rectVendorCompareStateByRun[key] = {
        status: String((nextState && nextState.status) || prev.status || "idle"),
        payload: (nextState && nextState.payload && typeof nextState.payload === "object")
          ? nextState.payload
          : prev.payload,
        error: String((nextState && nextState.error) || ""),
        updated_at: String((nextState && nextState.updated_at) || new Date().toISOString()),
      };
    }

    function renderRectCompareModelCard(modelKey, modelData, runId) {
      const key = String(modelKey || "").trim();
      const title = key === "specific" ? "Rete Specifica Vendor" : "Rete Globale";
      const tag = key === "specific" ? "specific" : "global";
      if (!modelData || typeof modelData !== "object") {
        return `
          <div class="rect-compare-model">
            <div class="rect-compare-head">
              <div class="rect-compare-title">${esc(title)}</div>
              <span class="tag tag-model">${esc(tag)}</span>
            </div>
            <div class="small">Modello non disponibile nel confronto.</div>
          </div>
        `;
      }
      const metrics = (modelData.metrics && typeof modelData.metrics === "object") ? modelData.metrics : {};
      const rectData = (modelData.rect_per_image && typeof modelData.rect_per_image === "object") ? modelData.rect_per_image : {};
      const available = !!modelData.available;
      const canOpenGallery = available && !!rectData.available && Array.isArray(rectData.items) && rectData.items.length > 0;
      const checkpoint = String(modelData.checkpoint || "");
      const modelError = String(modelData.error || "");
      return `
        <div class="rect-compare-model">
          <div class="rect-compare-head">
            <div class="rect-compare-title">${esc(title)}</div>
            <span class="tag tag-model">${esc(tag)}</span>
          </div>
          <div class="small"><b>checkpoint</b>: <span class="path">${esc(checkpoint || "-")}</span></div>
          <div class="rect-compare-metrics">
            <div class="m"><div class="mk">n img</div><div class="mv">${esc(cleanVal(metrics.count))}</div></div>
            <div class="m"><div class="mk">mean IoU</div><div class="mv">${esc(formatNum(metrics.mean_iou, 4))}</div></div>
            <div class="m"><div class="mk">median IoU</div><div class="mv">${esc(formatNum(metrics.median_iou, 4))}</div></div>
            <div class="m"><div class="mk">p25 / p75</div><div class="mv">${esc(formatNum(metrics.p25_iou, 4))} / ${esc(formatNum(metrics.p75_iou, 4))}</div></div>
            <div class="m"><div class="mk">mean discrepancy</div><div class="mv">${esc(formatNum(metrics.mean_discrepancy, 4))}</div></div>
            <div class="m"><div class="mk">stato</div><div class="mv">${available ? "ok" : "errore"}</div></div>
          </div>
          ${modelError ? `<div class="small">Errore: ${esc(modelError)}</div>` : ""}
          <div class="rect-routing-actions">
            <button
              type="button"
              class="btn secondary mini ${canOpenGallery ? "" : "is-disabled"}"
              ${canOpenGallery ? "" : "disabled"}
              data-open-rect-compare-model-gallery="1"
              data-run-id="${esc(runId)}"
              data-model-key="${esc(key)}"
              onclick="return window.__openRectCompareModelGalleryFromBtn ? window.__openRectCompareModelGalleryFromBtn(this) : false;"
            >Apri galleria ${esc(tag)}</button>
          </div>
        </div>
      `;
    }

    function renderRectVendorCompareState(compareState, runId) {
      const state = (compareState && typeof compareState === "object")
        ? compareState
        : {status: "idle", payload: null, error: "", updated_at: ""};
      const status = String(state.status || "idle");
      if (status === "idle") {
        return `
          <div class="rect-compare-state">
            <div class="small">Confronto non ancora eseguito. Premi "Confronta globale vs specifica".</div>
          </div>
        `;
      }
      if (status === "running") {
        return `
          <div class="rect-compare-state running">
            <div class="small"><b>Confronto in esecuzione...</b> sto calcolando rect-per-image per rete globale e specifica.</div>
          </div>
        `;
      }
      if (status === "error") {
        return `
          <div class="rect-compare-state error">
            <div class="small"><b>Errore confronto:</b> ${esc(state.error || "-")}</div>
          </div>
        `;
      }

      const payload = (state.payload && typeof state.payload === "object") ? state.payload : {};
      const models = (payload.models && typeof payload.models === "object") ? payload.models : {};
      const compareAvailable = !!payload.compare_available;
      const winner = String(payload.winner || "").trim();
      const delta = asNumber(payload.delta_mean_iou_specific_minus_global, NaN);
      const winnerTxt = winner === "specific" ? "specifica" : (winner === "global" ? "globale" : "-");
      const winnerHtml = compareAvailable
        ? `<span class="winner-chip">migliore: ${esc(winnerTxt)} | Δ mean IoU (spec-global)=${esc(formatNum(delta, 4))}</span>`
        : `<span class="small">Confronto completo non disponibile (manca uno dei due modelli).</span>`;
      const routingMessage = cleanVal(payload.routing_message || "");
      const generatedAt = cleanVal(payload.generated_at_utc || state.updated_at);

      return `
        <div class="rect-compare-state">
          <div class="rect-compare-head">
            <div class="rect-compare-title">Risultato confronto globale vs specifica</div>
            ${winnerHtml}
          </div>
          <div class="small" style="margin-top:4px;">${esc(routingMessage)}</div>
          <div class="small">generated_at_utc=${esc(generatedAt)}</div>
          <div class="rect-compare-grid">
            ${renderRectCompareModelCard("global", models.global, runId)}
            ${renderRectCompareModelCard("specific", models.specific, runId)}
          </div>
          <div class="rect-routing-actions">
            <button
              type="button"
              class="btn secondary mini"
              data-run-id="${esc(runId)}"
              data-force="1"
              data-run-rect-vendor-compare="1"
              onclick="return window.__runRectVendorCompareFromBtn ? window.__runRectVendorCompareFromBtn(this) : false;"
            >Ricalcola confronto</button>
          </div>
        </div>
      `;
    }

    function renderRectRoutingAndCompare(summary, runId) {
      const routing = getRectRoutingDecision(summary);
      const state = getRectVendorCompareState(runId);
      const levelClass = routing.level === "warn" ? "warn" : (routing.level === "ok" ? "ok" : "info");
      const compareDisabled = !routing.compare_possible || state.status === "running";
      const compareBtnTitle = routing.compare_possible
        ? "Esegue rect-per-image sia con globale sia con specifica, poi mostra metriche comparative."
        : "Confronto non disponibile: manca checkpoint specifico o globale.";
      const checkpointUsedName = checkpointBaseName(routing.used_checkpoint);
      const globalName = checkpointBaseName(routing.global_checkpoint);
      const specificName = checkpointBaseName(routing.specific_checkpoint);
      const compareStateHtml = renderRectVendorCompareState(state, runId);
      return `
        <div class="rect-routing-alert ${esc(levelClass)}">
          <div class="rect-routing-title">${esc(routing.title)}</div>
          <div class="small">${esc(routing.reason)}</div>
          <div class="rect-routing-kv">
            <div class="kv"><div class="k">Vendor</div><div class="v">${esc(cleanVal(routing.vendor_name))}</div></div>
            <div class="kv"><div class="k">Vendor confidence</div><div class="v">${esc(Number.isFinite(routing.vendor_confidence) ? formatNum(routing.vendor_confidence, 4) : "-")}</div></div>
            <div class="kv"><div class="k">Soglia specifica</div><div class="v">${esc(Number.isFinite(routing.threshold) ? formatNum(routing.threshold, 4) : "-")}</div></div>
            <div class="kv"><div class="k">line11_source</div><div class="v">${esc(cleanVal(routing.source))}</div></div>
            <div class="kv"><div class="k">Checkpoint usato</div><div class="v">${esc(checkpointUsedName || cleanVal(routing.used_checkpoint))}</div></div>
            <div class="kv"><div class="k">Checkpoint globale</div><div class="v">${esc(globalName || cleanVal(routing.global_checkpoint))}</div></div>
            <div class="kv"><div class="k">Checkpoint specifico vendor</div><div class="v">${esc(specificName || cleanVal(routing.specific_checkpoint))}</div></div>
            <div class="kv"><div class="k">Confronto disponibile</div><div class="v">${routing.compare_possible ? "SI" : "NO"}</div></div>
          </div>
          <div class="rect-routing-actions">
            <button
              type="button"
              class="btn secondary mini ${compareDisabled ? "is-disabled" : ""}"
              ${compareDisabled ? "disabled" : ""}
              title="${esc(compareBtnTitle)}"
              data-run-id="${esc(runId)}"
              data-force="0"
              data-run-rect-vendor-compare="1"
              onclick="return window.__runRectVendorCompareFromBtn ? window.__runRectVendorCompareFromBtn(this) : false;"
            >Confronta globale vs specifica</button>
            ${!routing.compare_possible ? `<span class="small">Manca il checkpoint specifico per vendor=${esc(cleanVal(routing.vendor_name))} o quello globale.</span>` : ""}
          </div>
          ${compareStateHtml}
        </div>
      `;
    }

    async function runRectVendorCompare(runId, forceFlag) {
      const run = String(runId || "").trim();
      if (!run) return false;
      setRectVendorCompareState(run, {
        status: "running",
        payload: null,
        error: "",
        updated_at: new Date().toISOString(),
      });
      if (selectedRunId === run && selectedRunData && selectedRunData.summary) {
        renderActiveEvidenceSection(selectedRunData.summary);
      }
      try {
        const out = await apiPost(`/api/runs/${encodeURIComponent(run)}/rect_vendor_compare`, {
          force: !!forceFlag,
        });
        setRectVendorCompareState(run, {
          status: "done",
          payload: out,
          error: "",
          updated_at: new Date().toISOString(),
        });
      } catch (err) {
        setRectVendorCompareState(run, {
          status: "error",
          payload: null,
          error: String(err && err.message ? err.message : err),
          updated_at: new Date().toISOString(),
        });
      }
      if (selectedRunId === run && selectedRunData && selectedRunData.summary) {
        renderActiveEvidenceSection(selectedRunData.summary);
      }
      return false;
    }

    function openRectCompareModelGallery(modelKey, runId) {
      const run = String(runId || "").trim();
      const key = String(modelKey || "").trim();
      if (!run || !key) return false;
      const state = getRectVendorCompareState(run);
      const payload = (state.payload && typeof state.payload === "object") ? state.payload : {};
      const models = (payload.models && typeof payload.models === "object") ? payload.models : {};
      const modelObj = (models[key] && typeof models[key] === "object") ? models[key] : null;
      if (!modelObj) return false;
      const rectData = (modelObj.rect_per_image && typeof modelObj.rect_per_image === "object")
        ? modelObj.rect_per_image
        : null;
      if (!rectData || !rectData.available || !Array.isArray(rectData.items) || !rectData.items.length) {
        return false;
      }
      const summaryLike = {recognition_evidence: {rect_per_image: rectData}};
      return openRectGallery(0, summaryLike, run);
    }

    function renderRotationEvidence(summary) {
      rotationEvidenceEl.innerHTML = "";
      const rot = (summary && summary.rotation_evidence && typeof summary.rotation_evidence === "object")
        ? summary.rotation_evidence
        : {};
      const osdDebug = (rot.osd_debug && typeof rot.osd_debug === "object") ? rot.osd_debug : {};
      const ocrDebug = (rot.ocr_debug && typeof rot.ocr_debug === "object") ? rot.ocr_debug : {};
      const beforeRel = String(rot.sample_before_rel || "");
      const afterRel = String(rot.sample_after_rel || "");
      const originalRel = String(rot.sample_original_rel || "");
      const hasPair = !!beforeRel && !!afterRel;
      const decisionReason = cleanVal(rot.rotation_decision_reason || "");
      const osdVotesTxt = [0, 90, 180, 270]
        .map((deg) => `${deg}:${cleanVal(osdDebug && osdDebug.votes ? osdDebug.votes[String(deg)] : 0)}`)
        .join(" | ");
      const ocrScoresTxt = [0, 90, 180, 270]
        .map((deg) => `${deg}:${formatNum(ocrDebug && ocrDebug.scores ? ocrDebug.scores[String(deg)] : NaN, 2)}`)
        .join(" | ");
      const ocrWordsTxt = [0, 90, 180, 270]
        .map((deg) => `${deg}:${cleanVal(ocrDebug && ocrDebug.word_counts ? ocrDebug.word_counts[String(deg)] : 0)}`)
        .join(" | ");
      const pairHtml = hasPair ? `
        <div class="pair-grid">
          <div class="pair-item">
            <div class="label">Prima</div>
            <img src="${imgUrl(beforeRel)}" loading="lazy">
            <div class="path">${esc(beforeRel)}</div>
          </div>
          <div class="pair-item">
            <div class="label">Dopo (rotazione applicata)</div>
            <img src="${imgUrl(afterRel)}" loading="lazy">
            <div class="path">${esc(afterRel)}</div>
          </div>
        </div>
      ` : `<div class="small">Nessun esempio before/after disponibile.</div>`;
      rotationEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Rotazione (esempio su 1 frame) <span class="tag tag-rotation">rotation</span></div>
          <div class="small">
            deg_clockwise=${esc(rot.rotation_deg_clockwise)} |
            source=${esc(rot.rotation_source)} |
            ratio=${esc(rot.rotation_vote_ratio)} |
            votes=${esc(rot.rotation_votes_total)}/${esc(rot.rotation_samples_checked)} |
            OCR validazione usata=${rot.ocr_validation_used ? "SI" : "NO"}
          </div>
          <div class="small" style="margin-top:4px;"><b>Motivo decisione</b>: ${esc(decisionReason)}</div>
          <div class="detail-list" style="margin-top:8px;">
            <div>
              <b>OSD diagnostics</b><br>
              best_rotation: ${esc(cleanVal(osdDebug.best_rotation))}<br>
              best_count: ${esc(cleanVal(osdDebug.best_count))}<br>
              ratio: ${esc(formatNum(osdDebug.ratio, 4))}<br>
              min_conf: ${esc(formatNum(osdDebug.min_confidence, 2))}<br>
              votes: ${esc(osdVotesTxt)}
            </div>
            <div>
              <b>OCR diagnostics</b><br>
              candidate: ${esc(cleanVal(ocrDebug.candidate_rotation))}<br>
              selected: ${esc(cleanVal(ocrDebug.selected_rotation))}<br>
              delta_score: ${esc(formatNum(ocrDebug.delta_score, 3))}<br>
              min_delta: ${esc(formatNum(ocrDebug.min_score_delta, 3))}<br>
              scores: ${esc(ocrScoresTxt)}<br>
              words: ${esc(ocrWordsTxt)}
            </div>
          </div>
          <div class="small" style="margin-top:4px;">frame sorgente: ${esc(originalRel || "-")}</div>
          ${pairHtml}
        </div>
      `;
    }

    function formatRectTlbr(boxObj) {
      if (!boxObj || typeof boxObj !== "object") return "-";
      const top = cleanVal(boxObj.top);
      const left = cleanVal(boxObj.left);
      const bottom = cleanVal(boxObj.bottom);
      const right = cleanVal(boxObj.right);
      return `${top}|${left}|${bottom}|${right}|`;
    }

    function getRectRedPayload(summary) {
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const rr = (rec.rect_red && typeof rec.rect_red === "object") ? rec.rect_red : null;
      const itemsRaw = rr && Array.isArray(rr.items) ? rr.items : [];
      const items = itemsRaw
        .slice()
        .sort((a, b) => {
          const al = segmentLengthPxFromItem(a);
          const bl = segmentLengthPxFromItem(b);
          if (bl !== al) return bl - al;
          return Number(b && b.discrepancy_vs_base ? b.discrepancy_vs_base : 0) - Number(a && a.discrepancy_vs_base ? a.discrepancy_vs_base : 0);
        });
      return {rectRed: rr, items};
    }

    function segmentLengthPxFromItem(it) {
      if (!it || typeof it !== "object") return 0;
      const seg = (it.auto_segment && typeof it.auto_segment === "object") ? it.auto_segment : {};
      const lenPx = Number(seg.length_px);
      if (Number.isFinite(lenPx) && lenPx > 0) return lenPx;
      const lenNorm = Number(seg.length_norm);
      if (Number.isFinite(lenNorm) && lenNorm > 0) return lenNorm;
      return 0;
    }

    function medianNumber(valuesRaw) {
      const values = (Array.isArray(valuesRaw) ? valuesRaw : [])
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v))
        .sort((a, b) => a - b);
      if (!values.length) return NaN;
      const n = values.length;
      if (n % 2 === 1) return values[Math.floor(n / 2)];
      return (values[(n / 2) - 1] + values[n / 2]) / 2.0;
    }

    function computeRectRedSegmentMedian(items) {
      const rows = Array.isArray(items) ? items : [];
      const lenPx = [];
      const lenNorm = [];
      const x1 = [];
      const x2 = [];
      const y = [];
      rows.forEach((it) => {
        const seg = (it && it.auto_segment && typeof it.auto_segment === "object") ? it.auto_segment : {};
        const lpx = Number(seg.length_px);
        const lnm = Number(seg.length_norm);
        const sx1 = Number(seg.x1);
        const sx2 = Number(seg.x2);
        const sy = Number(seg.y);
        if (Number.isFinite(lpx)) lenPx.push(lpx);
        if (Number.isFinite(lnm)) lenNorm.push(lnm);
        if (Number.isFinite(sx1)) x1.push(sx1);
        if (Number.isFinite(sx2)) x2.push(sx2);
        if (Number.isFinite(sy)) y.push(sy);
      });
      return {
        len_px: medianNumber(lenPx),
        len_norm: medianNumber(lenNorm),
        x1: medianNumber(x1),
        x2: medianNumber(x2),
        y: medianNumber(y),
        count: rows.length,
      };
    }

    function computeMedianRectTlbrFromItems(items, keyName) {
      const rows = Array.isArray(items) ? items : [];
      const k = String(keyName || "pred_rect_tlbr");
      const top = [];
      const left = [];
      const bottom = [];
      const right = [];
      rows.forEach((it) => {
        const box = (it && it[k] && typeof it[k] === "object") ? it[k] : {};
        const t = Number(box.top);
        const l = Number(box.left);
        const b = Number(box.bottom);
        const r = Number(box.right);
        if (Number.isFinite(t)) top.push(t);
        if (Number.isFinite(l)) left.push(l);
        if (Number.isFinite(b)) bottom.push(b);
        if (Number.isFinite(r)) right.push(r);
      });
      const mt = medianNumber(top);
      const ml = medianNumber(left);
      const mb = medianNumber(bottom);
      const mr = medianNumber(right);
      if (![mt, ml, mb, mr].every((v) => Number.isFinite(v))) return "-";
      return `${Math.round(mt)}|${Math.round(ml)}|${Math.round(mb)}|${Math.round(mr)}|`;
    }

    function shortImageName(pathValue) {
      const txt = String(pathValue || "").trim();
      if (!txt) return "-";
      const parts = txt.split(/[\\/]+/).filter(Boolean);
      return parts.length ? parts[parts.length - 1] : txt;
    }

    function formatSegmentRef(segRef) {
      if (!segRef || typeof segRef !== "object") return "-";
      const imageId = cleanVal(segRef.image_id || "-");
      const lenPx = formatNum(segRef.len_px, 2);
      const lenNorm = formatNum(segRef.len_norm, 4);
      const seg = (segRef.segment && typeof segRef.segment === "object") ? segRef.segment : {};
      const x1 = formatNum(seg.x1, 4);
      const x2 = formatNum(seg.x2, 4);
      const y = formatNum(seg.y, 4);
      return `img=${imageId} | len_px=${lenPx} | len_norm=${lenNorm} | x1=${x1} x2=${x2} y=${y}`;
    }

    async function runRectRedRecompute(runId, marginPct, brightThr) {
      const run = String(runId || "").trim();
      if (!run) return false;
      const margin = Number(marginPct);
      const bright = Number(brightThr);
      if (!Number.isFinite(margin) || margin < 0 || margin > 30) {
        alert("margin_pct deve essere tra 0 e 30.");
        return false;
      }
      if (!Number.isFinite(bright) || bright < 0 || bright > 255) {
        alert("bright_thr deve essere tra 0 e 255.");
        return false;
      }
      try {
        const out = await apiPost(`/api/runs/${encodeURIComponent(run)}/rect_red_recompute`, {
          margin_pct: margin,
          bright_thr: bright,
        });
        if (selectedRunData && selectedRunData.summary && selectedRunId === run) {
          const summary = selectedRunData.summary;
          if (!summary.recognition_evidence || typeof summary.recognition_evidence !== "object") {
            summary.recognition_evidence = {};
          }
          if (out.rect_red_evidence && typeof out.rect_red_evidence === "object") {
            summary.recognition_evidence.rect_red = out.rect_red_evidence;
          }
          if (!summary.pipeline_row || typeof summary.pipeline_row !== "object") {
            summary.pipeline_row = {};
          }
          summary.pipeline_row.line_11_rect_echo = String(out.line11_text || summary.pipeline_row.line_11_rect_echo || "");
          summary.pipeline_row.line_11_top = String(out.line11_top || summary.pipeline_row.line_11_top || "");
          summary.pipeline_row.line_11_left = String(out.line11_left || summary.pipeline_row.line_11_left || "");
          summary.pipeline_row.line_11_bottom = String(out.line11_bottom || summary.pipeline_row.line_11_bottom || "");
          summary.pipeline_row.line_11_right = String(out.line11_right || summary.pipeline_row.line_11_right || "");
          summary.pipeline_row.line_11_method = "segment_top_red_rect";
          summary.pipeline_row.line_11_rect_red_margin_pct = String(formatNum(out.margin_pct, 4));
          summary.pipeline_row.line_11_rect_red_winner_group = String(out.line11_winner_group || "");
          renderPredictionSummary(summary);
          renderFssFromSummary(summary);
          renderActiveEvidenceSection(summary);
        }
      } catch (err) {
        alert(`Ricalcolo rect rosso fallito: ${String(err && err.message ? err.message : err)}`);
      }
      return false;
    }

    function getRectPerImagePayload(summary) {
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const rectData = (rec.rect_per_image && typeof rec.rect_per_image === "object")
        ? rec.rect_per_image
        : null;
      const itemsRaw = rectData && Array.isArray(rectData.items) ? rectData.items : [];
      const items = itemsRaw
        .slice()
        .sort((a, b) => Number(b && b.discrepancy ? b.discrepancy : 0) - Number(a && a.discrepancy ? a.discrepancy : 0));
      return {rectData, items};
    }

    function normalizeSuGiuPredictionLabel(value) {
      const txt = String(value || "")
        .trim()
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "");
      if (txt === "su" || txt === "up" || txt === "alto") return "su";
      if (txt === "giu" || txt === "down" || txt === "basso") return "giu";
      return "other";
    }

    function predictionScoreNumber(value, fallback = NaN) {
      const direct = Number(value);
      if (Number.isFinite(direct)) return direct;
      const parsed = Number.parseFloat(String(value || "").replace(",", "."));
      return Number.isFinite(parsed) ? parsed : fallback;
    }

    function suGiuPredictionConfidence(item) {
      const it = (item && typeof item === "object") ? item : {};
      const exported = predictionScoreNumber(it.confidence, NaN);
      if (Number.isFinite(exported)) return exported;
      const label = normalizeSuGiuPredictionLabel(it.pred_label || it.su_giu_pred);
      if (label === "su") return predictionScoreNumber(it.prob_su, -1);
      if (label === "giu") return predictionScoreNumber(it.prob_giu, -1);
      return Math.max(
        predictionScoreNumber(it.prob_su, -1),
        predictionScoreNumber(it.prob_giu, -1),
      );
    }

    function compareSuGiuByLabelAndConfidence(a, b) {
      const order = {su: 0, giu: 1, other: 2};
      const al = normalizeSuGiuPredictionLabel((a && a.pred_label) || (a && a.su_giu_pred));
      const bl = normalizeSuGiuPredictionLabel((b && b.pred_label) || (b && b.su_giu_pred));
      const ar = Object.prototype.hasOwnProperty.call(order, al) ? order[al] : order.other;
      const br = Object.prototype.hasOwnProperty.call(order, bl) ? order[bl] : order.other;
      if (ar !== br) return ar - br;
      const ac = suGiuPredictionConfidence(a);
      const bc = suGiuPredictionConfidence(b);
      if (ac !== bc) return bc - ac;
      const ai = Number((a && a.image_index) || 0);
      const bi = Number((b && b.image_index) || 0);
      if (Number.isFinite(ai) && Number.isFinite(bi) && ai !== bi) return ai - bi;
      const ap = String((a && a.image_rel) || "");
      const bp = String((b && b.image_rel) || "");
      return ap.localeCompare(bp);
    }

    function getSuGiuPerImagePayload(summary) {
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const sgData = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object")
        ? rec.su_giu_per_image
        : null;
      const itemsRaw = sgData && Array.isArray(sgData.items) ? sgData.items : [];
      const items = itemsRaw
        .slice()
        .sort(compareSuGiuByLabelAndConfidence);
      const labelCounts = (sgData && sgData.label_counts && typeof sgData.label_counts === "object")
        ? sgData.label_counts
        : {su: 0, giu: 0, other: 0};
      const splitFolders = (rec.su_giu_split_folders && typeof rec.su_giu_split_folders === "object")
        ? rec.su_giu_split_folders
        : {};
      const groups = {su: [], giu: [], other: []};
      items.forEach((it) => {
        const label = normalizeSuGiuPredictionLabel((it && it.pred_label) || (it && it.su_giu_pred));
        if (label === "su") groups.su.push(it);
        else if (label === "giu") groups.giu.push(it);
        else groups.other.push(it);
      });
      return {sgData, items, labelCounts, splitFolders, groups};
    }

    function inferLrOrientationGroup(item) {
      const it = (item && typeof item === "object") ? item : {};
      const qGroup = String(it.quadrant_group || "").trim().toUpperCase();
      const qStatus = String(it.quadrant_status || "").trim().toLowerCase();
      const qValidRaw = String(it.quadrant_valid ?? "").trim().toLowerCase();
      const quadrantUsable = qStatus === "ok" && !["0", "false", "no"].includes(qValidRaw);
      if (quadrantUsable && ["NF", "LR", "UD", "LRUD"].includes(qGroup)) return qGroup;
      const pathTxt = String(it.image_rel || it.image_path || "").trim();
      const parts = pathTxt.split(/[\\/]+/).filter(Boolean);
      const candidates = [];
      if (parts.length) candidates.push(parts[parts.length - 1]);
      if (parts.length > 1) candidates.push(parts[parts.length - 2]);
      candidates.push(pathTxt);
      const txt = candidates.join(" ").toUpperCase();
      const normalized = txt.replace(/[^A-Z0-9]+/g, "_");
      const padded = `_${normalized.replace(/^_+|_+$/g, "")}_`;
      if (/_FLIPLR_UD_/.test(padded) || /_FLIPUD_LR_/.test(padded)) return "LRUD";
      if (/_FLIPLRUD_/.test(padded) || /_FLIPUDLR_/.test(padded)) return "LRUD";
      if (/(^|_)LRUD(_|$)/.test(normalized)) return "LRUD";
      if (/_NOFLIP_/.test(padded)) return "NF";
      if (/_FLIPLR_/.test(padded)) return "LR";
      if (/_FLIPUD_/.test(padded)) return "UD";
      if (/(^|_)UD(_|$)/.test(normalized)) return "UD";
      if (/(^|_)LR(_|$)/.test(normalized)) return "LR";
      if (/(^|_)NF(_|$)/.test(normalized)) return "NF";
      const side = String(it.detected_marker_side || "").trim().toLowerCase();
      const cy = Number(it.marker_cy_crop_norm);
      if (Number.isFinite(cy)) {
        if (cy < 0.5 && side === "left") return "NF";
        if (cy < 0.5 && side === "right") return "LR";
        if (cy >= 0.5 && side === "left") return "UD";
        if (cy >= 0.5 && side === "right") return "LRUD";
      }
      return "OTHER";
    }

    const LR_MARKER_RELIABLE_MATCH_SCORE = 0.62;
    const LR_MARKER_DEFAULT_EXPANDED_STEPS = [0.03, 0.06, 0.10, 0.15, 0.20];

    function lrMarkerThresholds(model) {
      const m = (model && typeof model === "object") ? model : {};
      const minMatch = asNumber(m.lr_marker_min_match_score, LR_MARKER_RELIABLE_MATCH_SCORE);
      const stepsRaw = Array.isArray(m.lr_marker_expanded_search_steps) && m.lr_marker_expanded_search_steps.length
        ? m.lr_marker_expanded_search_steps
        : LR_MARKER_DEFAULT_EXPANDED_STEPS;
      return {
        minMatch,
        reliableBox: Math.max(LR_MARKER_RELIABLE_MATCH_SCORE, minMatch),
        fullCropFallback: asNumber(m.lr_marker_full_crop_fallback_threshold, 0.55),
        expandedSearch: asNumber(m.lr_marker_expanded_search_threshold, 0.66),
        expandedSteps: stepsRaw.map((x) => asNumber(x, NaN)).filter((x) => Number.isFinite(x)),
        minSuGiuConfidence: asNumber(m.lr_marker_min_sugiu_confidence, 0.80),
        blankTemplateMaxValue: asNumber(m.lr_marker_blank_template_max_value, 3),
      };
    }

    function formatLrStepPct(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return cleanVal(value);
      return `${Math.round(n * 100)}%`;
    }

    function lrMarkerHasValidBox(item) {
      const it = (item && typeof item === "object") ? item : {};
      const t = Number(it.marker_top_abs);
      const l = Number(it.marker_left_abs);
      const b = Number(it.marker_bottom_abs);
      const r = Number(it.marker_right_abs);
      return Number.isFinite(t) && Number.isFinite(l) && Number.isFinite(b) && Number.isFinite(r) && b > t && r > l;
    }

    function lrMarkerBoxContributionSummary(items, threshold, targetWidth, targetHeight) {
      const order = ["NF", "LR", "UD", "LRUD", "OTHER"];
      const blockKeys = new Set(["NF", "LR", "UD", "LRUD"]);
      const byGroup = {};
      order.forEach((key) => {
        byGroup[key] = {total: 0, used: 0, excluded: 0, low_score: 0, noncanonical: 0, blank: 0, forced: 0, quadrant: 0, no_box: 0, other_group: 0};
      });
      const reasons = {low_score: 0, noncanonical: 0, blank: 0, forced: 0, quadrant: 0, no_box: 0, other_group: 0};
      let total = 0;
      let used = 0;
      let excluded = 0;
      (Array.isArray(items) ? items : []).forEach((it) => {
        total += 1;
        const groupKeyRaw = inferLrOrientationGroup(it);
        const groupKey = byGroup[groupKeyRaw] ? groupKeyRaw : "OTHER";
        const rel = lrMarkerReliability(it, threshold, targetWidth, targetHeight);
        const validBox = lrMarkerHasValidBox(it);
        const inBlock = blockKeys.has(groupKey);
        byGroup[groupKey].total += 1;
        if (inBlock && rel.reliable && validBox) {
          used += 1;
          byGroup[groupKey].used += 1;
          return;
        }
        excluded += 1;
        byGroup[groupKey].excluded += 1;
        if (!inBlock) {
          reasons.other_group += 1;
          byGroup[groupKey].other_group += 1;
        }
        if (rel.lowScore) {
          reasons.low_score += 1;
          byGroup[groupKey].low_score += 1;
        }
        if (!rel.canonicalSize) {
          reasons.noncanonical += 1;
          byGroup[groupKey].noncanonical += 1;
        }
        if (rel.blankMatch) {
          reasons.blank += 1;
          byGroup[groupKey].blank += 1;
        }
        if (rel.forced) {
          reasons.forced += 1;
          byGroup[groupKey].forced += 1;
        }
        if (!rel.quadrantValid) {
          reasons.quadrant += 1;
          byGroup[groupKey].quadrant += 1;
        }
        if (!validBox) {
          reasons.no_box += 1;
          byGroup[groupKey].no_box += 1;
        }
      });
      return {total, used, excluded, reasons, byGroup, targetWidth, targetHeight, threshold};
    }

    function orientationTemplateDecisionHtml(payload) {
      const p = (payload && typeof payload === "object") ? payload : {};
      const requested = String(p.requested || "").trim() || "historical_best_then_derived";
      const effective = String(p.effective || "").trim();
      const fixedPolicy = String(p.fixedPolicy || "").trim();
      const fallbackReason = String(p.fallbackReason || "").trim();
      const thresholds = (p.thresholds && typeof p.thresholds === "object") ? p.thresholds : lrMarkerThresholds({});
      const effectiveLabel = effective === "derived_folder" || fixedPolicy === "fixed_derived_folder_template"
        ? "Derived"
        : "Best historical";
      let decision = "Usato il template storico best-match della libreria vendor.";
      if (requested === "derived_folder") {
        decision = "Policy richiesta: Derived. La pipeline crea subito un template dalla cartella corrente.";
      } else if (effective === "derived_folder" || fixedPolicy === "fixed_derived_folder_template") {
        decision = "Best historical provato, poi sostituito con Derived per questa cartella.";
      } else if (requested === "historical_best") {
        decision = "Policy richiesta: solo Best historical. Nessun fallback Derived.";
      } else if (requested === "historical_best_then_derived") {
        decision = "Best historical mantenuto: i controlli di affidabilita non hanno richiesto Derived.";
      }
      const fallbackRule = `Fallback a Derived se score storico < ${formatNum(thresholds.minMatch, 2)} oppure se i marker affidabili sono troppo pochi: con almeno 8 frame comparabili serve max(2, 25%), sotto 8 ne serve almeno 1.`;
      const stepTxt = thresholds.expandedSteps.length ? thresholds.expandedSteps.map(formatLrStepPct).join(", ") : "-";
      return `
        <div class="orientation-decision-grid">
          <div class="orientation-decision-card">
            <b>Decisione template: ${esc(effectiveLabel)}</b>
            <div class="meta">${esc(decision)}</div>
            <div class="meta">requested=${esc(requested)}${effective ? ` | effective=${esc(effective)}` : ""}${fixedPolicy ? ` | output=${esc(fixedPolicy)}` : ""}</div>
            ${fallbackReason ? `<div class="meta"><span class="tag tag-warn">fallback</span> ${esc(fallbackReason)}</div>` : ""}
          </div>
          <div class="orientation-decision-card">
            <b>Parametri considerati</b>
            <div class="meta">score match marker, size frame canonica, patch non nera, strategia di ricerca e coordinate marker valide.</div>
            <div class="meta">Derived usa un crop del marker dal miglior match storico su un frame campione, evitando patch blank/troppo piccole.</div>
          </div>
          <div class="orientation-decision-card">
            <b>Soglie operative</b>
            <div class="meta">min_match=${esc(formatNum(thresholds.minMatch, 2))} | box=${esc(formatNum(thresholds.reliableBox, 2))} | SU/GIU min=${esc(formatNum(thresholds.minSuGiuConfidence, 2))}</div>
            <div class="meta">full-crop se score &lt; ${esc(formatNum(thresholds.fullCropFallback, 2))}; search espansa se score &lt; ${esc(formatNum(thresholds.expandedSearch, 2))}; step=${esc(stepTxt)}.</div>
          </div>
          <div class="orientation-decision-card">
            <b>Regola fallback</b>
            <div class="meta">${esc(fallbackRule)}</div>
            <div class="meta">Template blank esclusi se max pixel &lt;= ${esc(cleanVal(thresholds.blankTemplateMaxValue))}.</div>
          </div>
        </div>
      `;
    }

    function orientationBoxSummaryHtml(summary) {
      const s = (summary && typeof summary === "object") ? summary : lrMarkerBoxContributionSummary([], LR_MARKER_RELIABLE_MATCH_SCORE, 0, 0);
      const reasons = (s.reasons && typeof s.reasons === "object") ? s.reasons : {};
      const byGroup = (s.byGroup && typeof s.byGroup === "object") ? s.byGroup : {};
      const groups = ["NF", "LR", "UD", "LRUD"];
      const reasonTxt = [
        ["score basso", reasons.low_score],
        ["size diversa", reasons.noncanonical],
        ["match nero", reasons.blank],
        ["consenso forzato", reasons.forced],
        ["quadrante", reasons.quadrant],
        ["box assente", reasons.no_box],
        ["OTHER", reasons.other_group],
      ].filter(([, v]) => Number(v || 0) > 0).map(([k, v]) => `${k}:${v}`).join(" | ") || "-";
      const tableRows = groups.map((key) => {
        const row = byGroup[key] ? byGroup[key] : {};
        return `
          <div>${esc(key)}</div>
          <div>${esc(cleanVal(row.total || 0))}</div>
          <div>${esc(cleanVal(row.used || 0))}</div>
          <div>${esc(cleanVal(row.excluded || 0))}</div>
          <div>${esc(cleanVal(row.low_score || 0))}</div>
        `;
      }).join("");
      const targetTxt = Number(s.targetWidth) > 0 && Number(s.targetHeight) > 0
        ? `${cleanVal(s.targetWidth)}x${cleanVal(s.targetHeight)}`
        : "-";
      return `
        <div class="orientation-summary-strip">
          <span class="tag tag-data">marker totali ${esc(cleanVal(s.total))}</span>
          <span class="tag tag-ok">usati nei box ${esc(cleanVal(s.used))}</span>
          <span class="tag tag-warn">esclusi ${esc(cleanVal(s.excluded))}</span>
          <span class="tag tag-data">size target ${esc(targetTxt)}</span>
        </div>
        <div class="small" style="margin-top:6px;">
          I marker esclusi non contribuiscono ai rettangoli NF/LR/UD/LRUD della riga #16; restano solo per review/audit.
          Motivi: ${esc(reasonTxt)}.
        </div>
        <div class="orientation-mini-table">
          <div class="head">Gruppo</div><div class="head">tot</div><div class="head">usati</div><div class="head">esclusi</div><div class="head">low</div>
          ${tableRows}
        </div>
      `;
    }

    function lrMarkerReliability(item, minMatchScoreRaw, targetWidthRaw, targetHeightRaw) {
      const it = (item && typeof item === "object") ? item : {};
      const minMatchScore = Math.max(
        LR_MARKER_RELIABLE_MATCH_SCORE,
        Number.isFinite(Number(minMatchScoreRaw)) ? Number(minMatchScoreRaw) : LR_MARKER_RELIABLE_MATCH_SCORE,
      );
      const targetWidth = Number.isFinite(Number(targetWidthRaw)) ? Number(targetWidthRaw) : 0;
      const targetHeight = Number.isFinite(Number(targetHeightRaw)) ? Number(targetHeightRaw) : 0;
      const score = Number(it.match_score);
      const strategy = String(it.search_strategy || "");
      const imageW = Number(it.image_width);
      const imageH = Number(it.image_height);
      const hasTarget = targetWidth > 0 && targetHeight > 0;
      const hasDims = Number.isFinite(imageW) && Number.isFinite(imageH) && imageW > 0 && imageH > 0;
      const canonicalSize = !hasTarget || !hasDims || (imageW === targetWidth && imageH === targetHeight);
      const reviewReason = String(it.review_reason || "");
      const blankMatch = Number(it.match_patch_is_blank) === 1 || reviewReason.split(";").includes("blank_marker_match");
      const lowScore = !(Number.isFinite(score) && score >= minMatchScore);
      const forced = strategy === "spatial_consensus_forced";
      const qStatus = String(it.quadrant_status || "").trim().toLowerCase();
      const qValidRaw = String(it.quadrant_valid ?? "").trim().toLowerCase();
      const quadrantValid = !(
        qStatus === "invalid" ||
        qValidRaw === "0" ||
        qValidRaw === "false" ||
        qValidRaw === "no"
      );
      return {
        reliable: !lowScore && !forced && canonicalSize && !blankMatch && quadrantValid,
        lowScore,
        forced,
        canonicalSize,
        blankMatch,
        quadrantValid,
      };
    }

    function buildLrOrientationGroups(items, minMatchScoreRaw, targetWidthRaw, targetHeightRaw) {
      const minMatchScore = Math.max(
        LR_MARKER_RELIABLE_MATCH_SCORE,
        Number.isFinite(Number(minMatchScoreRaw)) ? Number(minMatchScoreRaw) : LR_MARKER_RELIABLE_MATCH_SCORE,
      );
      const targetWidth = Number.isFinite(Number(targetWidthRaw)) ? Number(targetWidthRaw) : 0;
      const targetHeight = Number.isFinite(Number(targetHeightRaw)) ? Number(targetHeightRaw) : 0;
      const order = ["NF", "LR", "UD", "LRUD", "OTHER"];
      const groups = {};
      order.forEach((key) => {
        groups[key] = {
          key,
          items: [],
          envelope: null,
          min_score: null,
          max_score: null,
          avg_score: null,
          label_counts: {},
          strategy_counts: {},
          reliable_count: 0,
          unreliable_count: 0,
          low_score_count: 0,
          forced_count: 0,
          noncanonical_count: 0,
          blank_match_count: 0,
          quadrant_invalid_count: 0,
        };
      });
      (Array.isArray(items) ? items : []).forEach((it) => {
        const key = inferLrOrientationGroup(it);
        if (!groups[key]) groups[key] = {key, items: [], envelope: null, min_score: null, max_score: null, avg_score: null, label_counts: {}, strategy_counts: {}, reliable_count: 0, unreliable_count: 0, low_score_count: 0, forced_count: 0, noncanonical_count: 0, blank_match_count: 0, quadrant_invalid_count: 0};
        groups[key].items.push(it);
      });
      Object.values(groups).forEach((group) => {
        let top = Infinity;
        let left = Infinity;
        let bottom = -Infinity;
        let right = -Infinity;
        let nBox = 0;
        let scoreSum = 0;
        let scoreN = 0;
        group.items.forEach((it) => {
          const label = String((it && it.lr_label) || "other").trim() || "other";
          const strategy = String((it && it.search_strategy) || "unknown").trim() || "unknown";
          group.label_counts[label] = (group.label_counts[label] || 0) + 1;
          group.strategy_counts[strategy] = (group.strategy_counts[strategy] || 0) + 1;
          const score = Number(it && it.match_score);
          if (Number.isFinite(score)) {
            scoreSum += score;
            scoreN += 1;
            group.min_score = group.min_score === null ? score : Math.min(group.min_score, score);
            group.max_score = group.max_score === null ? score : Math.max(group.max_score, score);
          }
          const reliability = lrMarkerReliability(it, minMatchScore, targetWidth, targetHeight);
          const reliable = reliability.reliable;
          if (reliable) group.reliable_count = Number(group.reliable_count || 0) + 1;
          else {
            group.unreliable_count = Number(group.unreliable_count || 0) + 1;
            if (reliability.lowScore) group.low_score_count = Number(group.low_score_count || 0) + 1;
            if (reliability.forced) group.forced_count = Number(group.forced_count || 0) + 1;
            if (!reliability.canonicalSize) group.noncanonical_count = Number(group.noncanonical_count || 0) + 1;
            if (reliability.blankMatch) group.blank_match_count = Number(group.blank_match_count || 0) + 1;
            if (!reliability.quadrantValid) group.quadrant_invalid_count = Number(group.quadrant_invalid_count || 0) + 1;
          }
          const t = Number(it && it.marker_top_abs);
          const l = Number(it && it.marker_left_abs);
          const b = Number(it && it.marker_bottom_abs);
          const r = Number(it && it.marker_right_abs);
          if (reliable && Number.isFinite(t) && Number.isFinite(l) && Number.isFinite(b) && Number.isFinite(r) && b > t && r > l) {
            top = Math.min(top, t);
            left = Math.min(left, l);
            bottom = Math.max(bottom, b);
            right = Math.max(right, r);
            nBox += 1;
          }
        });
        if (nBox > 0) {
          group.envelope = {
            top: Math.trunc(top),
            left: Math.trunc(left),
            bottom: Math.trunc(bottom),
            right: Math.trunc(right),
            width: Math.trunc(right - left),
            height: Math.trunc(bottom - top),
            boxes: nBox,
          };
        }
        group.avg_score = scoreN > 0 ? scoreSum / scoreN : null;
      });
      return order
        .filter((key) => key !== "OTHER")
        .map((key) => groups[key])
        .filter((group) => !!group);
    }

    function formatLrEnvelope(envelope) {
      if (!envelope || typeof envelope !== "object") return "-";
      return `top=${cleanVal(envelope.top)} | left=${cleanVal(envelope.left)} | bottom=${cleanVal(envelope.bottom)} | right=${cleanVal(envelope.right)} | ${cleanVal(envelope.width)}x${cleanVal(envelope.height)} px`;
    }

    function lrCountsToText(obj) {
      const entries = Object.entries((obj && typeof obj === "object") ? obj : {})
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])));
      return entries.length ? entries.map(([k, v]) => `${k}:${v}`).join(" | ") : "-";
    }

    function optionSelected(value, current) {
      return String(value || "") === String(current || "") ? " selected" : "";
    }

    function lrMarkerReviewKey(item) {
      const it = (item && typeof item === "object") ? item : {};
      const rel = String(it.image_rel || it.image_path || "").trim();
      const idx = String(cleanVal(it.image_index || "")).trim();
      return `${idx || "?"}|${rel || "image"}`;
    }

    function lrMarkerReviewAnnotation(key) {
      const root = annotationsState && annotationsState.lr_marker_reviews && typeof annotationsState.lr_marker_reviews === "object"
        ? annotationsState.lr_marker_reviews
        : {};
      const ann = root[String(key || "")];
      return ann && typeof ann === "object" ? ann : {};
    }

    function parseLrMarkerRectText(text) {
      const vals = String(text || "")
        .split(/[|,;\\s]+/)
        .map((x) => Number(x))
        .filter((x) => Number.isFinite(x));
      if (vals.length < 4) return null;
      const top = Math.trunc(vals[0]);
      const left = Math.trunc(vals[1]);
      const bottom = Math.trunc(vals[2]);
      const right = Math.trunc(vals[3]);
      if (!(bottom > top && right > left)) return null;
      return {top, left, bottom, right};
    }

    function lrMarkerReviewItemByKey(key) {
      const summary = selectedRunData && selectedRunData.summary ? selectedRunData.summary : null;
      if (!summary) return null;
      const payload = getLrMarkerPerImagePayload(summary);
      const rows = Array.isArray(payload.items) ? payload.items : [];
      return rows.find((it) => lrMarkerReviewKey(it) === String(key || "")) || null;
    }

    function lrMarkerReviewContextForItem(item) {
      const it = (item && typeof item === "object") ? item : {};
      return {
        image_index: cleanVal(it.image_index),
        image_rel: String(it.image_rel || it.image_path || ""),
        match_score: cleanVal(it.match_score),
        inferred_group: inferLrOrientationGroup(it),
        lr_label: cleanVal(it.lr_label || ""),
        search_strategy: cleanVal(it.search_strategy || ""),
        review_reason: cleanVal(it.review_reason || ""),
        quadrant_status: cleanVal(it.quadrant_status || ""),
        quadrant_reason: cleanVal(it.quadrant_reason || ""),
        original_rect: `${cleanVal(it.marker_top_abs)},${cleanVal(it.marker_left_abs)},${cleanVal(it.marker_bottom_abs)},${cleanVal(it.marker_right_abs)}`,
      };
    }

    function lrMarkerReliabilityReasonText(item, reliability, threshold) {
      const it = (item && typeof item === "object") ? item : {};
      const rel = reliability && typeof reliability === "object" ? reliability : {};
      const reasons = [];
      if (rel.lowScore) reasons.push(`score < ${formatNum(threshold, 2)}`);
      if (!rel.canonicalSize) {
        const w = cleanVal(it.image_width);
        const h = cleanVal(it.image_height);
        reasons.push(`size non canonica ${w}x${h}`);
      }
      if (rel.blankMatch) reasons.push("match nero/quasi nero");
      if (rel.forced) reasons.push("forzato da consenso spaziale");
      if (rel.quadrantValid === false) {
        const qReason = String(it.quadrant_reason || "").trim();
        reasons.push(qReason ? `quadrante incoerente: ${qReason}` : "quadrante incoerente");
      }
      const reviewReason = String(it.review_reason || "").trim();
      if (reviewReason) reasons.push(reviewReason);
      return reasons.length ? reasons.join(" | ") : "da rivedere";
    }

    function lrMarkerQuadrantText(item) {
      const it = (item && typeof item === "object") ? item : {};
      const status = String(it.quadrant_status || "").trim().toLowerCase();
      const expected = String(it.quadrant_expected || "").trim();
      const center = String(it.quadrant_center || "").trim();
      const group = String(it.quadrant_group || "").trim();
      const reason = String(it.quadrant_reason || "").trim();
      const statusTxt = status || "unknown";
      const parts = [
        `quadrante=${statusTxt}`,
        expected ? `atteso=${expected}` : "",
        center ? `centro=${center}` : "",
        group ? `gruppo=${group}` : "",
        reason ? `motivo=${reason}` : "",
      ].filter(Boolean);
      return parts.join(" | ");
    }

    function lrMarkerQuadrantTagHtml(item) {
      const status = String((item && item.quadrant_status) || "").trim().toLowerCase();
      if (status === "invalid") return `<span class="tag tag-warn">quadrante incoerente</span>`;
      if (status === "ok") return `<span class="tag tag-ok">quadrante ok</span>`;
      return `<span class="tag tag-data">quadrante unknown</span>`;
    }

    function lrMarkerReviewControlsHtml(key) {
      const ann = lrMarkerReviewAnnotation(key);
      const flag = String(ann.flag || "");
      const correction = String(ann.correction || "");
      const rect = String(ann.rect || "");
      const comment = String(ann.comment || "");
      const keyEsc = esc(key);
      return `
        <div class="lr-low-review-controls">
          <div>
            <label>Stato review</label>
            <select data-lr-marker-review-input="1" data-lr-review-key="${keyEsc}" data-lr-review-kind="flag">
              <option value=""${optionSelected("", flag)}>da rivedere</option>
              <option value="ok"${optionSelected("ok", flag)}>ok</option>
              <option value="wrong_marker"${optionSelected("wrong_marker", flag)}>marker sbagliato</option>
              <option value="corrected"${optionSelected("corrected", flag)}>corretto</option>
              <option value="exclude"${optionSelected("exclude", flag)}>escludi</option>
            </select>
          </div>
          <div>
            <label>Correzione</label>
            <select data-lr-marker-review-input="1" data-lr-review-key="${keyEsc}" data-lr-review-kind="correction">
              <option value=""${optionSelected("", correction)}>--</option>
              <option value="NF"${optionSelected("NF", correction)}>NF</option>
              <option value="LR"${optionSelected("LR", correction)}>LR</option>
              <option value="UD"${optionSelected("UD", correction)}>UD</option>
              <option value="LRUD"${optionSelected("LRUD", correction)}>LRUD</option>
              <option value="template_bad"${optionSelected("template_bad", correction)}>template errato</option>
              <option value="exclude"${optionSelected("exclude", correction)}>escludi dal box</option>
            </select>
          </div>
          <div class="wide">
            <label>Rect marker corretto</label>
            <input data-lr-marker-review-input="1" data-lr-review-key="${keyEsc}" data-lr-review-kind="rect" value="${esc(rect)}" placeholder="top,left,bottom,right oppure clicca sull'immagine grande" />
          </div>
          <div class="wide">
            <label>Commento</label>
            <textarea data-lr-marker-review-input="1" data-lr-review-key="${keyEsc}" data-lr-review-kind="comment" placeholder="nota sul caso, motivo della correzione...">${esc(comment)}</textarea>
          </div>
        </div>
      `;
    }

    function renderLrLowScoreReviewSection(items, orientationGroups, threshold, targetWidth, targetHeight, options) {
      const opts = (options && typeof options === "object") ? options : {};
      const compact = !!opts.compact;
      const rows = (Array.isArray(items) ? items : [])
        .filter((it) => !lrMarkerReliability(it, threshold, targetWidth, targetHeight).reliable)
        .sort((a, b) => {
          const as = Number(a && a.match_score);
          const bs = Number(b && b.match_score);
          const av = Number.isFinite(as) ? as : -Infinity;
          const bv = Number.isFinite(bs) ? bs : -Infinity;
          if (av !== bv) return av - bv;
          const ai = Number(a && a.image_index);
          const bi = Number(b && b.image_index);
          if (Number.isFinite(ai) && Number.isFinite(bi) && ai !== bi) return ai - bi;
          return String((a && a.image_rel) || "").localeCompare(String((b && b.image_rel) || ""));
        });
      const savedCount = Object.values((annotationsState && annotationsState.lr_marker_reviews && typeof annotationsState.lr_marker_reviews === "object") ? annotationsState.lr_marker_reviews : {})
        .filter((ann) => ann && typeof ann === "object" && Object.values(ann).some((v) => String(v || "").trim()))
        .length;
      if (compact) {
        const summary = lrMarkerBoxContributionSummary(items, threshold, targetWidth, targetHeight);
        const compactSummaryHtml = opts.includeBoxSummary === false
          ? `<div class="small" style="margin-top:6px;">I marker esclusi sono conteggiati nel riepilogo blocchi sopra e non entrano nei rettangoli NF/LR/UD/LRUD.</div>`
          : orientationBoxSummaryHtml(summary);
        return `
          <div class="evidence-title" style="margin-top:12px;">Marker esclusi dai box <span class="tag tag-warn">${esc(cleanVal(summary.excluded))}</span></div>
          ${compactSummaryHtml}
          <div class="dup-actions" style="margin-top:6px;">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker_low">Apri galleria review marker esclusi</button>
            <button type="button" class="btn secondary mini" onclick="saveAnnotations(); return false;">Salva review marker</button>
            <span class="small">annotazioni gia compilate: ${esc(cleanVal(savedCount))}</span>
          </div>
        `;
      }
      const cardsHtml = rows.length
        ? rows.map((it) => {
            const reliability = lrMarkerReliability(it, threshold, targetWidth, targetHeight);
            const key = lrMarkerReviewKey(it);
            const reason = lrMarkerReliabilityReasonText(it, reliability, threshold);
            const marker = `${cleanVal(it && it.marker_top_abs)}|${cleanVal(it && it.marker_left_abs)}|${cleanVal(it && it.marker_bottom_abs)}|${cleanVal(it && it.marker_right_abs)}`;
            return `
              <div class="sugiu-folder-card lr-low-review-card">
                ${lrMarkerImageOverlayHtml(it, selectedRunId, orientationGroups)}
                <div class="meta"><b>frame=${esc(cleanVal(it && it.image_index))}</b> | score=${esc(formatNum(it && it.match_score, 4))} | gruppo=${esc(inferLrOrientationGroup(it))}</div>
                <div class="meta">motivo=${esc(reason)}</div>
                <div class="meta">${lrMarkerQuadrantTagHtml(it)} ${esc(lrMarkerQuadrantText(it))}</div>
                <div class="meta">marker_abs=${esc(marker)} | ${esc(cleanVal(it && it.search_strategy))}</div>
                <div class="meta">${esc(String((it && it.image_rel) || "-"))}</div>
                ${lrMarkerReviewControlsHtml(key)}
              </div>
            `;
          }).join("")
        : `<div class="small">Nessun caso sotto soglia o escluso dai box orientamento.</div>`;
      return `
        <div class="evidence-title" style="margin-top:12px;">Review marker LR esclusi <span class="tag tag-warn">${esc(cleanVal(rows.length))}</span></div>
        <div class="small">
          Casi esclusi dai box per score &lt; ${esc(formatNum(threshold, 2))}, size non canonica, match nero/quasi nero, consenso spaziale forzato o quadrante incoerente.
          Qui puoi commentare e correggere manualmente; salva con il pulsante sotto o da "Analisi Step".
        </div>
        <div class="dup-actions" style="margin-top:6px;">
          <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker_low">Apri galleria review grande</button>
          <button type="button" class="btn secondary mini" onclick="saveAnnotations(); return false;">Salva review marker</button>
          <span class="small">annotazioni gia compilate: ${esc(cleanVal(savedCount))}</span>
        </div>
        <div class="sugiu-folder-grid lr-low-review-grid" style="margin-top:10px;">${cardsHtml}</div>
      `;
    }

    function getLrMarkerPerImagePayload(summary) {
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const lrData = (rec.lr_marker_per_image && typeof rec.lr_marker_per_image === "object")
        ? rec.lr_marker_per_image
        : null;
      const itemsRaw = lrData && Array.isArray(lrData.items) ? lrData.items : [];
      const items = itemsRaw
        .slice()
        .sort((a, b) => {
          const as = Number(a && a.match_score ? a.match_score : 0);
          const bs = Number(b && b.match_score ? b.match_score : 0);
          if (as !== bs) return bs - as;
          const ai = Number(a && a.image_index ? a.image_index : 0);
          const bi = Number(b && b.image_index ? b.image_index : 0);
          if (ai !== bi) return ai - bi;
          const ap = String((a && a.image_rel) || "");
          const bp = String((b && b.image_rel) || "");
          return ap.localeCompare(bp);
        });
      const labelCounts = (lrData && lrData.label_counts && typeof lrData.label_counts === "object")
        ? lrData.label_counts
        : {not_lr_flipped: 0, lr_flipped: 0, other: 0};
      const statusCounts = (lrData && lrData.status_counts && typeof lrData.status_counts === "object")
        ? lrData.status_counts
        : {ok: 0, review: 0, other: 0};
      const quadrantCounts = (lrData && lrData.quadrant_counts && typeof lrData.quadrant_counts === "object")
        ? lrData.quadrant_counts
        : {ok: 0, invalid: 0, unknown: 0};
      const strategyCounts = (lrData && lrData.search_strategy_counts && typeof lrData.search_strategy_counts === "object")
        ? lrData.search_strategy_counts
        : {};
      const templatePathCounts = (lrData && lrData.template_path_counts && typeof lrData.template_path_counts === "object")
        ? {...lrData.template_path_counts}
        : {};
      if (!Object.keys(templatePathCounts).length) {
        items.forEach((it) => {
          const templatePath = String((it && it.template_path) || "").trim();
          if (templatePath) templatePathCounts[templatePath] = (templatePathCounts[templatePath] || 0) + 1;
        });
      }
      const best = (lrData && lrData.best && typeof lrData.best === "object") ? lrData.best : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object")
        ? rec.model_evidence
        : {};
      const row = (summary && summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const targetWidth = Number(row.line_09_video_x_size || row.line_07_video_input_size_x || 0) || 0;
      const targetHeight = Number(row.line_10_video_y_size || row.line_08_video_input_size_y || 0) || 0;
      const orientationGroups = buildLrOrientationGroups(items, model.lr_marker_min_match_score, targetWidth, targetHeight);
      return {lrData, items, labelCounts, statusCounts, quadrantCounts, strategyCounts, templatePathCounts, best, orientationGroups, targetWidth, targetHeight};
    }

    function markerOrientationGroupFromItem(item) {
      const it = (item && typeof item === "object") ? item : {};
      const side = String(it.detected_marker_side || "").trim().toLowerCase();
      const cy = Number(it.marker_cy_crop_norm);
      if (!Number.isFinite(cy) || (side !== "left" && side !== "right")) return "OTHER";
      if (cy < 0.5 && side === "left") return "NF";
      if (cy < 0.5 && side === "right") return "LR";
      if (cy >= 0.5 && side === "left") return "UD";
      if (cy >= 0.5 && side === "right") return "LRUD";
      return "OTHER";
    }

    function suGiuVerticalGroup(item) {
      const pred = String((item && item.su_giu_pred) || (item && item.pred_label) || "").trim().toLowerCase();
      if (pred === "su") return "NF";
      if (pred === "giu") return "UD";
      return "OTHER";
    }

    function orientationDiscrepancyFromMarker(item) {
      const markerGroup = markerOrientationGroupFromItem(item);
      const networkGroup = suGiuVerticalGroup(item);
      if (markerGroup === "OTHER" || networkGroup === "OTHER") return false;
      const markerVertical = markerGroup === "NF" || markerGroup === "LR" ? "NF" : "UD";
      return markerVertical !== networkGroup;
    }

    function renderOrientationEvidence(summary) {
      if (!suGiuEvidenceEl) return;
      suGiuEvidenceEl.innerHTML = "";
      const orientationHelp = helpTipHtml(
        "Orientamento",
        [
          "Step 1: rete SU/GIU separa la parte alta dalla parte bassa.",
          "Step 2: marker LR completa il flip: alto/sinistra=NF, alto/destra=LR, basso/sinistra=UD, basso/destra=LRUD.",
          "I box NF/LR/UD/LRUD usano solo marker affidabili: score sufficiente, size canonica, patch non nera e coordinate valide.",
        ],
      );
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const row = (summary && summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const sgPayload = getSuGiuPerImagePayload(summary);
      const lrPayload = getLrMarkerPerImagePayload(summary);
      const sgData = sgPayload.sgData;
      const lrData = lrPayload.lrData;
      const sgItems = Array.isArray(sgPayload.items) ? sgPayload.items : [];
      const lrItems = Array.isArray(lrPayload.items) ? lrPayload.items : [];
      const sgCounts = sgPayload.labelCounts || {};
      const lrCounts = lrPayload.labelCounts || {};
      const statusCounts = lrPayload.statusCounts || {};
      const quadrantCounts = lrPayload.quadrantCounts || {};
      const strategyCounts = lrPayload.strategyCounts || {};
      const orientationGroups = Array.isArray(lrPayload.orientationGroups) ? lrPayload.orientationGroups : [];
      const thresholds = lrMarkerThresholds(model);
      const lrReliableThreshold = thresholds.reliableBox;
      const fixedTemplatePath = String((lrData && lrData.fixed_template_path) || (lrPayload.best && lrPayload.best.folder_fixed_template_path) || row.lr_marker_best_template_path || "").trim();
      const templatePolicyRequested = String((lrData && lrData.template_policy_requested) || (lrPayload.best && lrPayload.best.template_policy_requested) || (model && model.lr_marker_template_policy) || "").trim();
      const fixedTemplatePolicy = String((lrData && lrData.template_policy) || (lrPayload.best && lrPayload.best.template_policy) || "").trim();
      const templatePolicyEffective = String((lrData && lrData.template_policy_effective) || (lrPayload.best && lrPayload.best.template_policy_effective) || row.lr_marker_template_policy_effective || "").trim();
      const templateFallbackReason = String((lrData && lrData.template_fallback_reason) || (lrPayload.best && lrPayload.best.template_fallback_reason) || row.lr_marker_template_fallback_reason || "").trim();
      const templateFallbackHtml = templateFallbackReason
        ? `<div class="small" style="margin-top:8px;color:#b45309;"><span class="tag tag-warn">fallback attivo</span> historical_best -> derived_folder<br>motivo: ${esc(templateFallbackReason)}</div>`
        : "";
      const templateSelectionScore = lrData
        ? (lrData.fixed_template_selection_score || (lrPayload.best && lrPayload.best.folder_fixed_template_selection_score) || row.lr_marker_best_score || "")
        : "";
      const templatePathsSeen = Object.keys((lrPayload.templatePathCounts && typeof lrPayload.templatePathCounts === "object") ? lrPayload.templatePathCounts : {}).filter(Boolean);
      const vendorLibrary = (lrData && lrData.vendor_template_library && typeof lrData.vendor_template_library === "object")
        ? lrData.vendor_template_library
        : {};
      const vendorLibraryItems = Array.isArray(vendorLibrary.items) ? vendorLibrary.items : [];
      const boxSummary = lrMarkerBoxContributionSummary(lrItems, lrReliableThreshold, lrPayload.targetWidth, lrPayload.targetHeight);
      const templateDecisionHtml = orientationTemplateDecisionHtml({
        requested: templatePolicyRequested,
        effective: templatePolicyEffective,
        fixedPolicy: fixedTemplatePolicy,
        fallbackReason: templateFallbackReason,
        thresholds,
      });

      if (!sgData && !lrData) {
        suGiuEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Orientamento ${orientationHelp} <span class="tag tag-rotation">orientation</span></div>
            <div class="small">Dati SU/GIU e LR marker non presenti in questa run.</div>
          </div>
        `;
        return;
      }

      const sgAvailable = !!(sgData && sgData.available && sgItems.length);
      const lrAvailable = !!(lrData && lrData.available && lrItems.length);
      const bestFrame = lrItems.length ? lrItems[0] : {};
      const worstFrame = lrItems.length ? lrItems[lrItems.length - 1] : {};
      const discrepancies = lrItems.filter((it) => (
        lrMarkerReliability(it, lrReliableThreshold, lrPayload.targetWidth, lrPayload.targetHeight).reliable
        && orientationDiscrepancyFromMarker(it)
      ));
      const darkSample = (lrData && lrData.dark_sample && typeof lrData.dark_sample === "object") ? lrData.dark_sample : null;
      const orientationHeroFrame = darkSample || bestFrame || (lrItems.length ? lrItems[0] : {});
      const orientationHeroHtml = orientationHeroBoxPreviewHtml(orientationHeroFrame, orientationGroups, selectedRunId);
      const darknessRank = (lrData && Array.isArray(lrData.darkness_rank)) ? lrData.darkness_rank : [];
      const darkRankRows = darknessRank.slice(0, 10).map((it, idx) => {
        const pct = formatNum(it && it.darkness_pct, 2);
        return `<div class="meta">${idx + 1}. frame=${esc(cleanVal(it && it.image_index))} | scurita=${esc(pct)}% | score=${esc(formatNum(it && it.match_score, 4))}</div>`;
      }).join("");
      let sampleTemplateRank = [];
      if (darkSample && darkSample.folder_template_rank_json) {
        try {
          const parsed = JSON.parse(String(darkSample.folder_template_rank_json || "[]"));
          if (Array.isArray(parsed)) sampleTemplateRank = parsed;
        } catch (err) {
          sampleTemplateRank = [];
        }
      }
      const templateRowsSource = sampleTemplateRank.length
        ? sampleTemplateRank.map((row) => ({
            path: String((row && row.template_path) || ""),
            score: row && row.score,
            meta: `${cleanVal(row && row.template_width)}x${cleanVal(row && row.template_height)}`,
          }))
        : vendorLibraryItems.map((row) => ({
            path: String((row && row.template_path) || ""),
            score: "",
            meta: String((row && row.source_path) || ""),
          }));
      const templateRows = templateRowsSource.length
        ? templateRowsSource.slice(0, 18).map((row) => {
            const path = String((row && row.path) || "");
            const score = row && row.score !== "" && row.score !== undefined ? formatNum(row.score, 4) : "-";
            return `
              <div class="lr-template-preview">
                <div class="label">score ${esc(score)}</div>
                <img src="${imgUrlForRun(selectedRunId, path)}" loading="lazy" decoding="async">
                <div class="meta">${esc(String((row && row.meta) || ""))}</div>
              </div>
            `;
          }).join("")
        : `<div class="small">Nessun template vendor disponibile. Ricostruisci la libreria dal volume 1.</div>`;
      const discrepancyHtml = discrepancies.length
        ? `
          <div class="evidence-title" style="margin-top:10px;">Discrepanze rete/marker <span class="tag tag-warn">${discrepancies.length}</span></div>
          <div class="small">Qui la posizione del marker corregge NF/UD: rete=${esc(suGiuVerticalGroup(discrepancies[0]))}, marker=${esc(markerOrientationGroupFromItem(discrepancies[0]))} sul primo caso.</div>
          <div class="sugiu-folder-grid" style="margin-top:8px;">
            ${discrepancies.slice(0, 8).map((it) => `
              <div class="sugiu-folder-card">
                ${lrMarkerImageOverlayHtml(it, selectedRunId, orientationGroups)}
                <div class="meta">frame=${esc(cleanVal(it.image_index))} | rete=${esc(suGiuVerticalGroup(it))} | corretto=${esc(markerOrientationGroupFromItem(it))}</div>
                <div class="meta">score=${esc(formatNum(it.match_score, 4))} | ${esc(cleanVal(it.search_strategy))}</div>
              </div>
            `).join("")}
          </div>
        `
        : `<div class="small" style="margin-top:8px;">Nessuna discrepanza verticale rete/marker rilevata nei frame LR disponibili.</div>`;
      const groupCardsHtml = orientationGroups.length
        ? orientationGroups.map((group) => `
            <div class="lr-orientation-card">
              <div class="head">
                <div class="title">${esc(group.key)}</div>
                <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker" data-lr-marker-group="${esc(group.key)}">Apri gruppo</button>
              </div>
              <div class="box">${esc(formatLrEnvelope(group.envelope))}</div>
              <div class="meta">usati nel box=${esc(cleanVal(group.envelope && group.envelope.boxes))} | tot=${esc(cleanVal(group.items.length))} | esclusi=${esc(cleanVal(group.unreliable_count || 0))}</div>
              <div class="meta">motivi esclusione: low=${esc(cleanVal(group.low_score_count || 0))}, size=${esc(cleanVal(group.noncanonical_count || 0))}, blank=${esc(cleanVal(group.blank_match_count || 0))}, forced=${esc(cleanVal(group.forced_count || 0))}, quadrante=${esc(cleanVal(group.quadrant_invalid_count || 0))}</div>
              <div class="meta">score min=${esc(formatNum(group.min_score, 4))} avg=${esc(formatNum(group.avg_score, 4))} max=${esc(formatNum(group.max_score, 4))}</div>
            </div>
          `).join("")
        : `<div class="small">Nessun gruppo NF/LR/UD/LRUD ricavabile dai nomi immagine.</div>`;
      const lowScoreReviewHtml = lrAvailable
        ? renderLrLowScoreReviewSection(lrItems, orientationGroups, lrReliableThreshold, lrPayload.targetWidth, lrPayload.targetHeight, {compact: true, includeBoxSummary: false})
        : "";

      suGiuEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Orientamento ${orientationHelp} <span class="tag tag-rotation">orientation</span></div>
          <div class="dup-actions">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="su_giu" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Galleria SU/GIU</button>
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Galleria marker</button>
          </div>
          ${orientationHeroHtml}
          <div class="detail-list" style="margin-top:8px;">
            <div>
              <b>1. Rete NF/UD</b><br>
              frame=${esc(cleanVal(sgItems.length))} | NF/su=${esc(cleanVal(sgCounts.su))} | UD/giu=${esc(cleanVal(sgCounts.giu))} | other=${esc(cleanVal(sgCounts.other))}<br>
              checkpoint: <span class="path">${esc(String(model.su_giu_checkpoint_used || model.su_giu_checkpoint_global || "-"))}</span>
            </div>
            <div>
              <b>2. Marker LR</b><br>
              frame=${esc(cleanVal(lrItems.length))} | non flipped=${esc(cleanVal(lrCounts.not_lr_flipped))} | LR flipped=${esc(cleanVal(lrCounts.lr_flipped))} | review=${esc(cleanVal(statusCounts.review))}<br>
              quadranti ok=${esc(cleanVal(quadrantCounts.ok))} | invalid=${esc(cleanVal(quadrantCounts.invalid))}<br>
              strategie=${esc(lrCountsToText(strategyCounts))}
            </div>
          </div>
          ${(!sgAvailable || !lrAvailable) ? `<div class="small" style="margin-top:8px;color:#b45309;">Run incompleta per questa vista: SU/GIU disponibile=${sgAvailable ? "SI" : "NO"}, LR marker disponibile=${lrAvailable ? "SI" : "NO"}.</div>` : ""}
          <div class="evidence-title" style="margin-top:10px;">Decisioni prese <span class="tag tag-data">best historical / derived</span></div>
          ${templateDecisionHtml}
          <div class="evidence-title" style="margin-top:10px;">Blocchi NF/LR/UD/LRUD <span class="tag tag-data">contributi</span></div>
          ${orientationBoxSummaryHtml(boxSummary)}
          <div class="evidence-title" style="margin-top:10px;">Campione scuro e template <span class="tag tag-data">sample</span></div>
          <div class="detail-list" style="margin-top:8px;">
            <div>
              <b>Frame campione</b><br>
              ${darkSample ? `frame=${esc(cleanVal(darkSample.image_index))} | scurita crop eco=${esc(formatNum(darkSample.darkness_pct, 2))}% | marker score=${esc(formatNum(darkSample.match_score, 4))}` : "Da calcolare al prossimo rerun."}<br>
              ${darkSample ? `<span class="path">${esc(String(darkSample.image_rel || "-"))}</span>` : ""}
            </div>
            <div>
              <b>Template marker usato</b><br>
              score=${esc(cleanVal(templateSelectionScore || "-"))}<br>
              policy=${esc(cleanVal(templatePolicyRequested || (model && model.lr_marker_template_policy) || "-"))}${templatePolicyEffective ? ` | effective=${esc(cleanVal(templatePolicyEffective))}` : ""}<br>
              <span class="path">${esc(fixedTemplatePath || "-")}</span>
              ${darkSample && darkSample.folder_template_seed_db_template_path ? `<br><span class="small">DB seed: <span class="path">${esc(String(darkSample.folder_template_seed_db_template_path || ""))}</span></span>` : ""}
              ${templateFallbackHtml}
            </div>
          </div>
          ${darkSample ? `<div class="sugiu-folder-card" style="margin-top:8px;">${lrMarkerVisualHtml(darkSample, selectedRunId)}</div>` : ""}
          <details class="toggle-block" style="margin-top:8px;">
            <summary>Template conosciuti vendor (${esc(sampleTemplateRank.length ? "rank campione" : "library")})</summary>
            <div class="toggle-content">
              <div class="small">Ordinati per sovrapposizione sul frame campione quando la run contiene il ranking; altrimenti mostra la libreria deduplicata del vendor.</div>
              <div class="lr-orientation-groups" style="margin-top:8px;">${templateRows}</div>
            </div>
          </details>
          ${darkRankRows ? `<details class="toggle-block" style="margin-top:8px;"><summary>Score scurita altri frame</summary><div class="toggle-content">${darkRankRows}</div></details>` : ""}
          <div class="evidence-title" style="margin-top:10px;">Box marker per orientamento <span class="tag tag-data">NF | LR | UD | LRUD</span></div>
          <div class="lr-orientation-groups">${groupCardsHtml}</div>
          ${lowScoreReviewHtml}
          <div class="evidence-title" style="margin-top:10px;">Audit rapido match marker <span class="tag tag-data">best/worst</span></div>
          <div class="pair-grid">
            <div class="pair-item">
              <div class="label">Migliore score ${esc(formatNum(bestFrame && bestFrame.match_score, 4))}</div>
              ${lrAvailable ? lrMarkerVisualHtml(bestFrame, selectedRunId) : `<div class="small">Non disponibile.</div>`}
            </div>
            <div class="pair-item">
              <div class="label">Peggiore score ${esc(formatNum(worstFrame && worstFrame.match_score, 4))}</div>
              ${lrAvailable ? lrMarkerVisualHtml(worstFrame, selectedRunId) : `<div class="small">Non disponibile.</div>`}
            </div>
          </div>
          ${discrepancyHtml}
        </div>
      `;
    }

    function getLtPerImagePayload(summary) {
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const ltData = (rec.lt_per_image && typeof rec.lt_per_image === "object")
        ? rec.lt_per_image
        : null;
      const itemsRaw = ltData && Array.isArray(ltData.items) ? ltData.items : [];
      const items = itemsRaw
        .slice()
        .sort((a, b) => {
          const ai = Number(a && a.image_index ? a.image_index : 0);
          const bi = Number(b && b.image_index ? b.image_index : 0);
          if (ai !== bi) return ai - bi;
          const ap = String((a && a.image_rel) || "");
          const bp = String((b && b.image_rel) || "");
          return ap.localeCompare(bp);
        });
      const labelCounts = (ltData && ltData.label_counts && typeof ltData.label_counts === "object")
        ? ltData.label_counts
        : {l: 0, t: 0, other: 0};
      const splitFolders = (rec.lt_split_folders && typeof rec.lt_split_folders === "object")
        ? rec.lt_split_folders
        : {};
      const groups = {l: [], t: [], other: []};
      items.forEach((it) => {
        const label = String((it && it.pred_label) || "").trim().toUpperCase();
        if (label === "L") groups.l.push(it);
        else if (label === "T") groups.t.push(it);
        else groups.other.push(it);
      });
      return {ltData, items, labelCounts, splitFolders, groups};
    }

    function buildRectGalleryItems(summary, runForPreview) {
      const runId = String(runForPreview || selectedRunId || "").trim();
      const payload = getRectPerImagePayload(summary);
      const rectData = payload.rectData;
      const items = payload.items;
      if (!runId || !rectData || !rectData.available || !items.length) {
        rectGalleryRunId = runId;
        rectGalleryItems = [];
        rectGalleryIndex = 0;
        rectGalleryMode = "rect";
        return [];
      }

      rectGalleryRunId = runId;
      rectGalleryMode = "rect";
      rectGalleryItems = items.map((it, idx) => {
        const rank = idx + 1;
        const imageRel = String((it && it.image_rel) || "");
        const overlayRel = String((it && it.overlay_rel) || "");
        const previewRel = overlayRel || imageRel;
        const singleRectRaw = (it && it.single_rect && typeof it.single_rect === "object") ? it.single_rect : {};
        const globalRectRaw = (it && it.global_rect_scaled && typeof it.global_rect_scaled === "object") ? it.global_rect_scaled : {};
        return {
          rank,
          discrepancy: formatNum(it && it.discrepancy, 4),
          iou: formatNum(it && it.iou, 4),
          centerNorm: formatNum(it && it.center_distance_norm, 4),
          areaRatio: formatNum(it && it.area_ratio_single_vs_global, 4),
          imageRel,
          overlayRel,
          previewRel,
          thumbRel: previewRel,
          overlayErr: String((it && it.overlay_error) || ""),
          dims: `${cleanVal(it && it.image_width)}x${cleanVal(it && it.image_height)}`,
          singleRect: formatRectTlbr(singleRectRaw),
          globalRectScaled: formatRectTlbr(globalRectRaw),
          singleRectRaw,
          globalRectRaw,
          imageWidth: Number(it && it.image_width) || 0,
          imageHeight: Number(it && it.image_height) || 0,
          imageLabel: shortImageName(imageRel),
        };
      });
      if (rectGalleryIndex < 0 || rectGalleryIndex >= rectGalleryItems.length) {
        rectGalleryIndex = 0;
      }
      return rectGalleryItems;
    }

    function buildRectRedGalleryItems(summary, runForPreview) {
      const runId = String(runForPreview || selectedRunId || "").trim();
      const payload = getRectRedPayload(summary);
      const rectRed = payload.rectRed;
      const itemsRaw = payload.items;
      if (!runId || !rectRed || !rectRed.available || !itemsRaw.length) {
        rectGalleryRunId = runId;
        rectGalleryItems = [];
        rectGalleryIndex = 0;
        rectGalleryMode = "rect_red";
        return [];
      }
      const items = itemsRaw
        .slice()
        .sort((a, b) => {
          const al = segmentLengthPxFromItem(a);
          const bl = segmentLengthPxFromItem(b);
          return bl - al;
        });
      const baseRectObj = (rectRed.line11_base && rectRed.line11_base.rect_tlbr && typeof rectRed.line11_base.rect_tlbr === "object")
        ? rectRed.line11_base.rect_tlbr
        : {};
      const redRectObj = (rectRed.line11_red && rectRed.line11_red.rect_tlbr && typeof rectRed.line11_red.rect_tlbr === "object")
        ? rectRed.line11_red.rect_tlbr
        : {};
      const topSegObj = (rectRed.top_segment_selected && rectRed.top_segment_selected.segment && typeof rectRed.top_segment_selected.segment === "object")
        ? rectRed.top_segment_selected.segment
        : {};
      const redRect = formatRectTlbr((rectRed.line11_red && rectRed.line11_red.rect_tlbr) || {});
      rectGalleryRunId = runId;
      rectGalleryMode = "rect_red";
      rectGalleryItems = items.map((it, idx) => {
        const rank = idx + 1;
        const imageRel = String((it && it.image_rel) || "");
        const overlayRel = String((it && it.overlay_rel) || "");
        const previewRel = overlayRel || imageRel;
        const seg = (it && it.auto_segment && typeof it.auto_segment === "object") ? it.auto_segment : {};
        const lenPx = formatNum(seg.length_px, 2);
        const lenNorm = formatNum(seg.length_norm, 4);
        const segMid = (Number(seg.x1 || 0) + Number(seg.x2 || 0)) / 2.0;
        const singleRectRaw = (it && it.pred_rect_tlbr && typeof it.pred_rect_tlbr === "object") ? it.pred_rect_tlbr : {};
        return {
          rank,
          discrepancy: `seg_len_px=${lenPx}`,
          iou: formatNum(it && it.iou_single_vs_base, 4),
          centerNorm: formatNum(segMid, 4),
          areaRatio: formatNum(it && it.discrepancy_vs_base, 4),
          imageRel,
          overlayRel,
          previewRel,
          thumbRel: previewRel,
          overlayErr: String((it && it.overlay_error) || ""),
          dims: `${cleanVal(it && it.image_width)}x${cleanVal(it && it.image_height)}`,
          singleRect: formatRectTlbr(singleRectRaw),
          globalRectScaled: redRect,
          singleRectRaw,
          redRectRaw: redRectObj,
          baseRectRaw: baseRectObj,
          segmentTopRaw: topSegObj,
          imageWidth: Number(it && it.image_width) || 0,
          imageHeight: Number(it && it.image_height) || 0,
          extraTitle: `segment_len_norm=${lenNorm}`,
          imageLabel: shortImageName(imageRel),
        };
      });
      if (rectGalleryIndex < 0 || rectGalleryIndex >= rectGalleryItems.length) {
        rectGalleryIndex = 0;
      }
      return rectGalleryItems;
    }

    function buildRectGalleryItemsByMode(summary, runForPreview, modeRaw) {
      const mode = String(modeRaw || "rect").trim().toLowerCase();
      if (mode === "rect_red") {
        return buildRectRedGalleryItems(summary, runForPreview);
      }
      return buildRectGalleryItems(summary, runForPreview);
    }

    function writeRectGalleryField(el, value) {
      if (!el) return;
      el.textContent = cleanVal(value);
    }

    function _num(v, fallback = NaN) {
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    }

    function _rectTlbrToPx(rectObj, srcW, srcH) {
      const r = (rectObj && typeof rectObj === "object") ? rectObj : {};
      const t = _num(r.top);
      const l = _num(r.left);
      const b = _num(r.bottom);
      const rr = _num(r.right);
      if (![t, l, b, rr].every((x) => Number.isFinite(x))) return null;
      const x1 = Math.max(0, Math.min(srcW - 1, l));
      const y1 = Math.max(0, Math.min(srcH - 1, t));
      const x2 = Math.max(0, Math.min(srcW - 1, rr));
      const y2 = Math.max(0, Math.min(srcH - 1, b));
      if (x2 <= x1 || y2 <= y1) return null;
      return {x1, y1, x2, y2};
    }

    function _computeContainRect(viewW, viewH, srcW, srcH) {
      const vw = Math.max(1, Number(viewW) || 1);
      const vh = Math.max(1, Number(viewH) || 1);
      const sw = Math.max(1, Number(srcW) || 1);
      const sh = Math.max(1, Number(srcH) || 1);
      const scale = Math.min(vw / sw, vh / sh);
      const drawW = sw * scale;
      const drawH = sh * scale;
      const drawX = (vw - drawW) * 0.5;
      const drawY = (vh - drawH) * 0.5;
      return {drawX, drawY, drawW, drawH, scale};
    }

    function _drawRectCanvas(ctx, box, sx, sy, color, lineWidth, dashed = false, ox = 0, oy = 0) {
      if (!box) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dashed ? [Math.max(5, lineWidth * 2), Math.max(4, lineWidth + 1)] : []);
      ctx.strokeRect(ox + (box.x1 * sx), oy + (box.y1 * sy), (box.x2 - box.x1) * sx, (box.y2 - box.y1) * sy);
      ctx.restore();
    }

    function _drawPointCanvas(ctx, x, y, color, radius = 4) {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(15,23,42,0.95)";
      ctx.lineWidth = 1;
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function _drawAxesCanvas(ctx, cx, cy, width, height, color, lineWidth = 1.5, ox = 0, oy = 0) {
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(ox, oy + cy);
      ctx.lineTo(ox + width, oy + cy);
      ctx.moveTo(ox + cx, oy);
      ctx.lineTo(ox + cx, oy + height);
      ctx.stroke();
      ctx.restore();
    }

    function _drawLegendCanvas(ctx, rows) {
      const entries = Array.isArray(rows) ? rows.filter((r) => r && r.label) : [];
      if (!entries.length) return;
      const pad = 8;
      const lineH = 16;
      const boxW = 12;
      const gap = 8;
      const x0 = 10;
      const y0 = 10;
      const maxText = entries.reduce((m, r) => Math.max(m, String(r.label).length), 0);
      const legendW = Math.max(220, 36 + (maxText * 7));
      const legendH = pad * 2 + (entries.length * lineH);
      ctx.save();
      ctx.fillStyle = "rgba(15,23,42,0.72)";
      ctx.fillRect(x0, y0, legendW, legendH);
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, legendW, legendH);
      ctx.font = "12px sans-serif";
      entries.forEach((row, idx) => {
        const yy = y0 + pad + (idx * lineH) + 6;
        ctx.fillStyle = row.color || "#fff";
        ctx.fillRect(x0 + pad, yy - 6, boxW, boxW);
        if (row.dashed) {
          ctx.strokeStyle = row.color || "#fff";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(x0 + pad, yy);
          ctx.lineTo(x0 + pad + boxW, yy);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(String(row.label), x0 + pad + boxW + gap, yy + 4);
      });
      ctx.restore();
    }

    function drawRectGalleryOverlay(item, modeRaw) {
      if (!rectCompareCanvasEl || !rectCompareImageEl || !item) return;
      const mode = String(modeRaw || "rect");
      const img = rectCompareImageEl;
      const canvas = rectCompareCanvasEl;
      const overlayRel = String((item && item.overlayRel) || "").trim();
      const previewRel = String((item && item.previewRel) || "").trim();
      const hasPreRenderedOverlay = !!overlayRel && previewRel === overlayRel;
      const viewW = Math.max(1, Math.round(img.clientWidth || 0));
      const viewH = Math.max(1, Math.round(img.clientHeight || 0));
      if (viewW <= 2 || viewH <= 2) return;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(viewW * dpr));
      canvas.height = Math.max(1, Math.round(viewH * dpr));
      canvas.style.width = `${viewW}px`;
      canvas.style.height = `${viewH}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, viewW, viewH);
      if (hasPreRenderedOverlay) {
        canvas.style.display = "none";
        return;
      }
      canvas.style.display = "block";

      const srcW = Math.max(1, _num(item.imageWidth, _num(img.naturalWidth, 1)));
      const srcH = Math.max(1, _num(item.imageHeight, _num(img.naturalHeight, 1)));
      const fit = _computeContainRect(viewW, viewH, srcW, srcH);
      const sx = fit.drawW / srcW;
      const sy = fit.drawH / srcH;
      const ox = fit.drawX;
      const oy = fit.drawY;
      const lineWidth = Math.max(2, Math.round(Math.min(fit.drawW, fit.drawH) / 260));

      if (mode === "rect_red") {
        const base = _rectTlbrToPx(item.baseRectRaw, srcW, srcH);
        const red = _rectTlbrToPx(item.redRectRaw, srcW, srcH);
        const single = _rectTlbrToPx(item.singleRectRaw, srcW, srcH);
        _drawRectCanvas(ctx, base, sx, sy, "rgb(66,66,66)", lineWidth, false, ox, oy);
        _drawRectCanvas(ctx, single, sx, sy, "rgb(38,130,255)", lineWidth, true, ox, oy);
        _drawRectCanvas(ctx, red, sx, sy, "rgb(220,40,40)", lineWidth, false, ox, oy);

        if (base) {
          const bcx = ((base.x1 + base.x2) * 0.5) * sx;
          const bcy = ((base.y1 + base.y2) * 0.5) * sy;
          _drawAxesCanvas(ctx, bcx, bcy, fit.drawW, fit.drawH, "rgba(82,82,82,0.95)", Math.max(1.5, lineWidth - 1), ox, oy);
          _drawPointCanvas(ctx, ox + bcx, oy + bcy, "rgb(66,66,66)", Math.max(4, lineWidth + 1));
        }
        if (red) {
          const rcx = ((red.x1 + red.x2) * 0.5) * sx + ox;
          const rcy = ((red.y1 + red.y2) * 0.5) * sy + oy;
          _drawPointCanvas(ctx, rcx, rcy, "rgb(220,40,40)", Math.max(4, lineWidth + 1));
        }
        if (single) {
          const scx = ((single.x1 + single.x2) * 0.5) * sx + ox;
          const scy = ((single.y1 + single.y2) * 0.5) * sy + oy;
          _drawPointCanvas(ctx, scx, scy, "rgb(38,130,255)", Math.max(4, lineWidth + 1));
        }
        const seg = (item.segmentTopRaw && typeof item.segmentTopRaw === "object") ? item.segmentTopRaw : {};
        let x1 = _num(seg.x1);
        let x2 = _num(seg.x2);
        let y = _num(seg.y);
        if ([x1, x2, y].every((v) => Number.isFinite(v))) {
          if (x1 > 1.2 || x2 > 1.2 || y > 1.2) {
            x1 = x1 / srcW;
            x2 = x2 / srcW;
            y = y / srcH;
          }
          x1 = Math.max(0, Math.min(1, x1));
          x2 = Math.max(0, Math.min(1, x2));
          y = Math.max(0, Math.min(1, y));
          if (x2 < x1) {
            const tmp = x1; x1 = x2; x2 = tmp;
          }
          const px1 = ox + (x1 * fit.drawW);
          const px2 = ox + (x2 * fit.drawW);
          const py = oy + (y * fit.drawH);
          ctx.save();
          ctx.strokeStyle = "rgb(255,219,77)";
          ctx.lineWidth = Math.max(2, lineWidth + 1);
          ctx.beginPath();
          ctx.moveTo(px1, py);
          ctx.lineTo(px2, py);
          ctx.stroke();
          ctx.restore();
          _drawPointCanvas(ctx, px1, py, "rgb(255,219,77)", Math.max(4, lineWidth + 1));
          _drawPointCanvas(ctx, px2, py, "rgb(255,219,77)", Math.max(4, lineWidth + 1));
          _drawPointCanvas(ctx, ((px1 + px2) * 0.5), py, "rgb(34,197,94)", Math.max(4, lineWidth + 1));
        }
        _drawLegendCanvas(ctx, [
          {color: "rgb(66,66,66)", label: "grigio: rect base mediana"},
          {color: "rgb(38,130,255)", label: "blu tratteggiato: rect per immagine", dashed: true},
          {color: "rgb(220,40,40)", label: "rosso: rect da segmento"},
          {color: "rgb(255,219,77)", label: "giallo: segmento top"},
          {color: "rgb(34,197,94)", label: "verde: punto medio segmento"},
        ]);
        return;
      }

      const single = _rectTlbrToPx(item.singleRectRaw, srcW, srcH);
      const global = _rectTlbrToPx(item.globalRectRaw, srcW, srcH);
      _drawRectCanvas(ctx, single, sx, sy, "rgb(255,219,77)", lineWidth, false, ox, oy);
      _drawRectCanvas(ctx, global, sx, sy, "rgb(220,40,40)", lineWidth, false, ox, oy);
      if (global) {
        const gcx = ((global.x1 + global.x2) * 0.5) * sx;
        const gcy = ((global.y1 + global.y2) * 0.5) * sy;
        _drawAxesCanvas(ctx, gcx, gcy, fit.drawW, fit.drawH, "rgba(220,40,40,0.85)", Math.max(1.5, lineWidth - 1), ox, oy);
        _drawPointCanvas(ctx, gcx + ox, gcy + oy, "rgb(220,40,40)", Math.max(4, lineWidth + 1));
      }
      if (single) {
        const scx = ((single.x1 + single.x2) * 0.5) * sx + ox;
        const scy = ((single.y1 + single.y2) * 0.5) * sy + oy;
        _drawPointCanvas(ctx, scx, scy, "rgb(255,219,77)", Math.max(4, lineWidth + 1));
      }
      _drawLegendCanvas(ctx, [
        {color: "rgb(255,219,77)", label: "giallo: rect singolo"},
        {color: "rgb(220,40,40)", label: "rosso: rect globale"},
        {color: "rgb(220,40,40)", label: "assi rossi: centro rect globale"},
        {color: "rgb(255,219,77)", label: "punto giallo: centro rect singolo"},
        {color: "rgb(220,40,40)", label: "punto rosso: centro rect globale"},
      ]);
    }

    function renderRectGalleryThumbs() {
      if (!rectCompareThumbsEl) return;
      rectCompareThumbsEl.innerHTML = "";
      if (!rectGalleryItems.length || !rectGalleryRunId) return;
      const frag = document.createDocumentFragment();
      rectGalleryItems.forEach((item, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rect-thumb-btn" + (idx === rectGalleryIndex ? " active" : "");
        btn.title = `#${idx + 1} ${item.imageLabel || item.imageRel || ""}`;
        btn.setAttribute("data-rect-thumb-index", String(idx));
        const srcRel = String(item.thumbRel || item.previewRel || item.imageRel || "").trim();
        if (srcRel) {
          const img = document.createElement("img");
          img.src = imgUrlForRun(rectGalleryRunId, srcRel);
          img.loading = "lazy";
          img.decoding = "async";
          btn.appendChild(img);
        } else {
          btn.textContent = `#${idx + 1}`;
        }
        frag.appendChild(btn);
      });
      rectCompareThumbsEl.appendChild(frag);
    }

    function renderRectGalleryCurrent() {
      if (!rectCompareModalEl) return;
      if (!rectGalleryItems.length || !rectGalleryRunId) {
        closeRectCompareModal();
        return;
      }
      rectGalleryIndex = Math.max(0, Math.min(rectGalleryItems.length - 1, rectGalleryIndex));
      const item = rectGalleryItems[rectGalleryIndex];
      const total = rectGalleryItems.length;
      const isRectRed = rectGalleryMode === "rect_red";
      if (rectCompareTitleEl) {
        rectCompareTitleEl.textContent = isRectRed
          ? "Galleria Post-Processing Segmento"
          : "Galleria Reti Rect";
      }
      if (rectCompareCardEl) {
        rectCompareCardEl.classList.toggle("mode-rect-segment", isRectRed);
        rectCompareCardEl.classList.toggle("mode-rect-net", !isRectRed);
      }
      if (rectCompareModeChipEl) {
        rectCompareModeChipEl.textContent = isRectRed ? "post-processing segmento" : "reti rect";
        rectCompareModeChipEl.classList.toggle("segment", isRectRed);
        rectCompareModeChipEl.classList.toggle("net", !isRectRed);
      }
      if (rectCompareImageLabelEl) {
        rectCompareImageLabelEl.textContent = isRectRed
          ? "Frame post-processing (segmento top + rect rosso)"
          : "Frame da rete rect (singolo/globale)";
      }
      if (rectCompareMetaEl) {
        rectCompareMetaEl.textContent = `run=${rectGalleryRunId} | tipo=${isRectRed ? "segmento_post_processing" : "reti_rect"}`;
      }
      if (rectCompareCounterEl) {
        rectCompareCounterEl.textContent = `${rectGalleryIndex + 1}/${total}`;
      }
      if (rectCompareImageEl) {
        const previewRel = String(item.previewRel || "").trim();
        const overlayRel = String(item.overlayRel || "").trim();
        const fallbackRel = String(item.imageRel || "").trim();
        rectCompareImageEl.onerror = null;
        if (previewRel && overlayRel && previewRel === overlayRel && fallbackRel && fallbackRel !== previewRel) {
          rectCompareImageEl.onerror = () => {
            rectCompareImageEl.onerror = null;
            item.previewRel = fallbackRel;
            rectCompareImageEl.src = imgUrlForRun(rectGalleryRunId, fallbackRel);
          };
        }
        rectCompareImageEl.src = previewRel ? imgUrlForRun(rectGalleryRunId, previewRel) : "";
      }
      writeRectGalleryField(rectCompareOverlayPathEl, item.overlayRel || "-");
      writeRectGalleryField(rectCompareIndexEl, `#${item.rank} di ${total}`);
      writeRectGalleryField(rectCompareDiscrepancyEl, item.discrepancy);
      writeRectGalleryField(rectCompareIouEl, item.iou);
      writeRectGalleryField(rectCompareCenterNormEl, item.centerNorm);
      writeRectGalleryField(rectCompareAreaRatioEl, item.areaRatio);
      writeRectGalleryField(rectCompareSizeEl, item.dims);
      writeRectGalleryField(rectCompareSingleRectEl, item.singleRect);
      writeRectGalleryField(rectCompareGlobalRectEl, item.globalRectScaled);
      writeRectGalleryField(rectCompareImagePathEl, item.imageRel || "-");
      writeRectGalleryField(rectCompareOverlayErrorEl, item.overlayErr || "-");
      writeRectGalleryField(
        rectCompareLegendEl,
        isRectRed
          ? "grigio=rect base mediana | blu tratteggiato=rect per immagine | rosso=rect da segmento | giallo=segmento top | verde=punto medio segmento"
          : "giallo=rect singolo | rosso=rect globale | assi rossi=centro globale | punti: giallo/rosso centri",
      );
      if (rectComparePrevBtnEl) rectComparePrevBtnEl.disabled = rectGalleryIndex <= 0;
      if (rectCompareNextBtnEl) rectCompareNextBtnEl.disabled = rectGalleryIndex >= (total - 1);
      renderRectGalleryThumbs();
      const drawOverlay = () => drawRectGalleryOverlay(item, isRectRed ? "rect_red" : "rect");
      if (rectCompareImageEl && rectCompareImageEl.complete && rectCompareImageEl.naturalWidth > 0) {
        requestAnimationFrame(drawOverlay);
      } else if (rectCompareImageEl) {
        const onLoad = () => {
          rectCompareImageEl.removeEventListener("load", onLoad);
          drawOverlay();
        };
        rectCompareImageEl.addEventListener("load", onLoad, {once: true});
      }
    }

    function openRectGallery(startIndexValue, summaryOverride, runIdOverride, modeOverride) {
      if (!rectCompareModalEl) return false;
      const summary = (summaryOverride && typeof summaryOverride === "object")
        ? summaryOverride
        : (selectedRunData && selectedRunData.summary ? selectedRunData.summary : null);
      const runForPreview = String(runIdOverride || selectedRunId || "").trim();
      const mode = String(modeOverride || "rect").trim().toLowerCase() === "rect_red" ? "rect_red" : "rect";
      if (!summary || !runForPreview) return false;
      const startNum = Number(startIndexValue);
      const startIdxRaw = Number.isFinite(startNum) ? Math.trunc(startNum) : 0;
      const openSig = `rect|${runForPreview}|${mode}|${startIdxRaw}`;
      if (!_allowGalleryOpen(openSig, 180)) return true;
      const items = buildRectGalleryItemsByMode(summary, runForPreview, mode);
      if (!items.length) return false;
      const startIdx = startIdxRaw;
      rectGalleryIndex = Math.max(0, Math.min(items.length - 1, startIdx));
      _closeOtherModals("rect");
      rectCompareModalEl.classList.remove("hidden");
      if (rectCompareCardEl && typeof rectCompareCardEl.focus === "function") {
        rectCompareCardEl.setAttribute("tabindex", "-1");
        rectCompareCardEl.focus({preventScroll: true});
      }
      renderRectGalleryCurrent();
      return true;
    }

    function moveRectGallery(delta) {
      if (!rectGalleryItems.length) return;
      const step = Number(delta);
      if (!Number.isFinite(step)) return;
      const target = rectGalleryIndex + Math.trunc(step);
      if (target < 0 || target >= rectGalleryItems.length) return;
      rectGalleryIndex = target;
      renderRectGalleryCurrent();
    }

    function _isModalVisible(el) {
      return !!(el && !el.classList.contains("hidden"));
    }

    function _closeOtherModals(except) {
      const keep = String(except || "").trim().toLowerCase();
      if (keep !== "rect" && rectCompareModalEl) rectCompareModalEl.classList.add("hidden");
      if (keep !== "evidence" && evidenceGalleryModalEl) evidenceGalleryModalEl.classList.add("hidden");
      if (keep !== "dup" && dupCompareModalEl) dupCompareModalEl.classList.add("hidden");
      activeModalGallery = keep || "";
    }

    function _allowGalleryOpen(signature, minGapMs) {
      const sig = String(signature || "").trim();
      const gapMs = Math.max(80, Number(minGapMs || 0) || 180);
      if (!sig) return true;
      const now = Date.now();
      if (lastGalleryOpenSig === sig && (now - lastGalleryOpenTsMs) < gapMs) {
        return false;
      }
      lastGalleryOpenSig = sig;
      lastGalleryOpenTsMs = now;
      return true;
    }

    function updateRectRedFocusFromRow(listEl, rowEl) {
      if (!listEl || !rowEl) return;
      const btn = rowEl.querySelector("button[data-segment-index]");
      if (!btn) return;
      const runId = String(listEl.getAttribute("data-focus-run-id") || selectedRunId || "").trim();
      const imgId = String(listEl.getAttribute("data-focus-img-id") || "").trim();
      const nameId = String(listEl.getAttribute("data-focus-name-id") || "").trim();
      const pathId = String(listEl.getAttribute("data-focus-path-id") || "").trim();
      const segId = String(listEl.getAttribute("data-focus-seg-id") || "").trim();
      const detailId = String(listEl.getAttribute("data-focus-detail-id") || "").trim();
      const openBtnId = String(listEl.getAttribute("data-focus-open-btn-id") || "").trim();

      const previewRel = String(btn.getAttribute("data-focus-preview-rel") || "").trim();
      const imageRel = String(btn.getAttribute("data-focus-image-rel") || "").trim();
      const segTxt = String(btn.getAttribute("data-focus-segment") || "").trim();
      const detailTxt = String(btn.getAttribute("data-focus-detail") || "").trim();
      const idxTxt = String(btn.getAttribute("data-segment-index") || "0");

      if (imgId) {
        const imgEl = document.getElementById(imgId);
        if (imgEl && imgEl.tagName === "IMG") {
          if (previewRel && runId) {
            imgEl.src = imgUrlForRun(runId, previewRel);
            imgEl.dataset.rel = previewRel;
          } else if (imageRel && runId) {
            imgEl.src = imgUrlForRun(runId, imageRel);
            imgEl.dataset.rel = imageRel;
          }
        }
      }
      if (nameId) {
        const el = document.getElementById(nameId);
        if (el) el.textContent = shortImageName(imageRel) || "-";
      }
      if (pathId) {
        const el = document.getElementById(pathId);
        if (el) el.textContent = imageRel || "-";
      }
      if (segId) {
        const el = document.getElementById(segId);
        if (el) el.textContent = segTxt || "-";
      }
      if (detailId) {
        const el = document.getElementById(detailId);
        if (el) el.textContent = detailTxt || "-";
      }
      if (openBtnId) {
        const openBtn = document.getElementById(openBtnId);
        if (openBtn) openBtn.setAttribute("data-start-index", idxTxt);
      }
    }

    function setSegmentListActiveIndex(listEl, idxRaw, options) {
      if (!listEl || !(listEl instanceof Element)) return -1;
      const opts = (options && typeof options === "object") ? options : {};
      const rows = Array.from(listEl.querySelectorAll(".segment-row"));
      if (!rows.length) return -1;
      const idxNum = Number(idxRaw);
      const idx = Number.isFinite(idxNum) ? Math.trunc(idxNum) : 0;
      const clamped = Math.max(0, Math.min(rows.length - 1, idx));
      listEl.setAttribute("data-segment-current-index", String(clamped));
      rows.forEach((row, i) => row.classList.toggle("active", i === clamped));
      const row = rows[clamped];
      if (row && opts.scrollIntoView !== false && typeof row.scrollIntoView === "function") {
        row.scrollIntoView({block: "nearest", behavior: opts.behavior || "smooth"});
      }
      if (row) updateRectRedFocusFromRow(listEl, row);
      return clamped;
    }

    function renderRectRedEvidence(summary, runId) {
      const payload = getRectRedPayload(summary);
      const rr = payload.rectRed;
      const items = payload.items;
      if (!rr) {
        return {
          mainHtml: `
            <div class="evidence-card">
              <div class="evidence-title">Rect Rosso (segmento top) <span class="tag tag-rect">rect_red</span></div>
              <div class="small">Dati rect rosso non presenti in questa run.</div>
            </div>
          `,
          sideHtml: `
            <div class="evidence-card">
              <div class="evidence-title">Classifica segmenti più lunghi <span class="tag tag-rect">rect_red</span></div>
              <div class="small">Dati non disponibili.</div>
            </div>
          `,
          sideListId: "",
        };
      }
      const available = !!rr.available;
      const reason = cleanVal(rr.error || "");
      const line11Red = formatRectTlbr((rr.line11_red && rr.line11_red.rect_tlbr) || {});
      const line11Base = formatRectTlbr((rr.line11_base && rr.line11_base.rect_tlbr) || {});
      const line11MedianPred = computeMedianRectTlbrFromItems(items, "pred_rect_tlbr");
      const marginPct = asNumber(rr.margin_pct, asNumber((summary && summary.pipeline_row && summary.pipeline_row.line_11_rect_red_margin_pct), 5.0));
      const brightThr = asNumber(rr.bright_thr, 70.0);
      const winnerGroup = cleanVal(rr.winner_group || (summary && summary.pipeline_row && summary.pipeline_row.line_11_rect_red_winner_group) || "-");
      const winnerReason = cleanVal(rr.winner_reason || "-");
      const groupCounts = (rr.group_counts && typeof rr.group_counts === "object") ? rr.group_counts : {};
      const countsTxt = `su=${cleanVal(groupCounts.su)} | giu=${cleanVal(groupCounts.giu)} | unknown=${cleanVal(groupCounts.unknown)}`;
      const segTopTxt = formatSegmentRef(rr.top_segment_selected);
      const runIdEsc = esc(runId);
      const marginInputId = `rectRedMargin_${runIdEsc}`;
      const brightInputId = `rectRedBright_${runIdEsc}`;
      const segMedian = computeRectRedSegmentMedian(items);
      const segMedianTxt = `len_px=${formatNum(segMedian.len_px, 2)} | len_norm=${formatNum(segMedian.len_norm, 4)} | x1=${formatNum(segMedian.x1, 4)} x2=${formatNum(segMedian.x2, 4)} y=${formatNum(segMedian.y, 4)} | n=${cleanVal(segMedian.count)}`;

      const recomputeControls = `
        <div class="rect-routing-actions" style="margin-top:4px;">
          <label style="display:flex;align-items:center;gap:6px;">
            <span class="small">Margin %</span>
            <input id="${marginInputId}" type="number" min="0" max="30" step="0.5" value="${esc(formatNum(marginPct, 2))}" style="width:88px;">
          </label>
          <label style="display:flex;align-items:center;gap:6px;">
            <span class="small">Bright Thr</span>
            <input id="${brightInputId}" type="number" min="0" max="255" step="1" value="${esc(formatNum(brightThr, 1))}" style="width:88px;">
          </label>
          <button
            type="button"
            class="btn secondary mini"
            data-run-id="${runIdEsc}"
            data-rect-red-recompute="1"
            data-margin-input-id="${marginInputId}"
            data-bright-input-id="${brightInputId}"
            onclick="return window.__runRectRedRecomputeFromBtn ? window.__runRectRedRecomputeFromBtn(this) : false;"
          >Ricalcola con margine</button>
          <button type="button" class="btn secondary mini" data-open-rect-gallery="1" data-open-rect-gallery-mode="rect_red" data-run-id="${runIdEsc}" data-start-index="0" onclick="return window.__openRectGalleryFromBtn ? window.__openRectGalleryFromBtn(this) : false;">Apri Galleria Post-Processing</button>
        </div>
      `;

      if (!available || !items.length) {
        return {
          mainHtml: `
            <div class="evidence-card">
              <div class="evidence-title">Rect Rosso (segmento top) <span class="tag tag-rect">rect_red</span></div>
              <div class="small">Analisi non disponibile: ${esc(reason || "-")}</div>
              ${recomputeControls}
            </div>
          `,
          sideHtml: `
            <div class="evidence-card">
              <div class="evidence-title">Classifica segmenti più lunghi <span class="tag tag-rect">rect_red</span></div>
              <div class="small">Analisi non disponibile: ${esc(reason || "-")}</div>
            </div>
          `,
          sideListId: "",
        };
      }

      const selectedImageId = String((rr.top_segment_selected && rr.top_segment_selected.image_id) || "").trim();
      let selectedIdx = 0;
      if (selectedImageId) {
        const byId = items.findIndex((it) => {
          const rel = String((it && it.image_rel) || "");
          const name = shortImageName(rel);
          return name === selectedImageId || rel.endsWith(`/${selectedImageId}`) || rel.endsWith(`\\${selectedImageId}`);
        });
        if (byId >= 0) selectedIdx = byId;
      }
      const selected = items[selectedIdx] || items[0];
      const selectedImageRel = String((selected && selected.image_rel) || "");
      const selectedOverlayRel = String((selected && selected.overlay_rel) || "");
      const selectedPreviewRel = selectedOverlayRel || selectedImageRel;
      const selectedLabel = cleanVal(selected && selected.orientation_label);
      const selectedLenPx = formatNum(segmentLengthPxFromItem(selected), 2);
      const selectedLenNorm = formatNum(selected && selected.auto_segment && selected.auto_segment.length_norm, 4);
      const selectedRect = formatRectTlbr(selected && selected.pred_rect_tlbr);
      const selectedSegTxt = formatSegmentRef(selected && selected.auto_segment);
      const selectedDetailTxt = `len_px=${selectedLenPx} | len_norm=${selectedLenNorm} | label=${selectedLabel} | rect=${selectedRect}`;
      const runIdDom = String(runId || "run").replace(/[^a-zA-Z0-9_-]+/g, "_");
      const segmentListId = `segmentList_${runIdDom}`;
      const focusImgId = `rectRedFocusImg_${runIdDom}`;
      const focusNameId = `rectRedFocusName_${runIdDom}`;
      const focusPathId = `rectRedFocusPath_${runIdDom}`;
      const focusSegId = `rectRedFocusSeg_${runIdDom}`;
      const focusDetailId = `rectRedFocusDetail_${runIdDom}`;
      const focusOpenBtnId = `rectRedOpenGallery_${runIdDom}`;

      const listRows = items.map((it, idx) => {
        const imageRel = String((it && it.image_rel) || "");
        const overlayRel = String((it && it.overlay_rel) || "");
        const previewRel = overlayRel || imageRel;
        const imageName = shortImageName(imageRel);
        const lenPx = formatNum(segmentLengthPxFromItem(it), 2);
        const label = cleanVal(it && it.orientation_label);
        const lenNorm = formatNum(it && it.auto_segment && it.auto_segment.length_norm, 4);
        const rectTxt = formatRectTlbr(it && it.pred_rect_tlbr);
        const segTxt = formatSegmentRef(it && it.auto_segment);
        const detailTxt = `len_px=${lenPx} | len_norm=${lenNorm} | label=${label} | rect=${rectTxt}`;
        const active = idx === selectedIdx ? " active" : "";
        return `
          <div class="segment-row${active}" data-segment-rank-index="${idx}">
            <button
              type="button"
              data-segment-list-id="${segmentListId}"
              data-segment-index="${idx}"
              data-focus-preview-rel="${esc(previewRel)}"
              data-focus-image-rel="${esc(imageRel)}"
              data-focus-segment="${esc(segTxt)}"
              data-focus-detail="${esc(detailTxt)}"
              onclick="return window.__setSegmentListIndex ? window.__setSegmentListIndex(this) : false;"
            >
              <span class="name">#${idx + 1} ${esc(imageName)} | ${esc(label)}</span>
              <span class="len">${esc(lenPx)} px</span>
            </button>
          </div>
        `;
      }).join("");

      const mainHtml = `
        <div class="evidence-card">
          <div class="evidence-title">Rettangolo Da Segmento Più Lungo <span class="tag tag-rect">rect_red</span></div>
          <div class="rect-segment-layout">
            <div class="segment-main-card">
              <div class="rect-focus-title">Immagine selezionata (segmento top)</div>
              ${selectedPreviewRel ? `<img id="${focusImgId}" src="${imgUrl(selectedPreviewRel)}" loading="lazy" decoding="async">` : `<div class="small">Preview non disponibile.</div>`}
              <div class="rect-focus-meta">
                <div class="meta-item"><b>Gruppo vincente</b>${esc(winnerGroup)}<br><span class="small">${esc(winnerReason)}</span></div>
                <div class="meta-item"><b>Immagine scelta</b><span id="${focusNameId}">${esc(shortImageName(selectedImageRel))}</span><br><span id="${focusPathId}" class="small">${esc(selectedImageRel || "-")}</span></div>
                <div class="meta-item"><b>Segmento scelto</b><span id="${focusSegId}">${esc(selectedSegTxt)}</span></div>
                <div class="meta-item"><b>Segmento mediano</b>${esc(segMedianTxt)}</div>
                <div class="meta-item"><b>Rect da segmento (rosso)</b>${esc(line11Red)}</div>
                <div class="meta-item"><b>Rect mediana base</b>${esc(line11Base)}</div>
                <div class="meta-item"><b>Rect mediana per-frame</b>${esc(line11MedianPred)}</div>
                <div class="meta-item"><b>Dettaglio frame</b><span id="${focusDetailId}">${esc(selectedDetailTxt)}</span></div>
                <div class="meta-item"><b>Gruppi</b>${esc(countsTxt)}</div>
              </div>
              <div class="rect-focus-actions">
                <button id="${focusOpenBtnId}" type="button" class="btn secondary mini" data-open-rect-gallery="1" data-open-rect-gallery-mode="rect_red" data-run-id="${runIdEsc}" data-start-index="${selectedIdx}" onclick="return window.__openRectGalleryFromBtn ? window.__openRectGalleryFromBtn(this) : false;">Apri Galleria Post-Processing</button>
              </div>
              ${recomputeControls}
              <div class="small">Overlay: grigio scuro=rect mediana base, giallo=segmento top, rosso=rect da segmento, blu tratteggiato=rect per immagine.</div>
            </div>
          </div>
        </div>
      `;
      const sideHtml = `
        <div class="segment-list-wrapper rect-side-ranking">
          <div class="segment-list-head">
            <div>
              <div class="rect-focus-title">Classifica segmenti più lunghi</div>
              <div class="small" style="margin:4px 0 0;">ordine decrescente. Usa frecce su/giu per aggiornare la preview a sinistra.</div>
            </div>
            <div class="segment-scroll-controls">
              <button type="button" class="segment-scroll-btn" data-segment-scroll="up" data-segment-list-id="${segmentListId}" onclick="return window.__scrollSegmentListFromBtn ? window.__scrollSegmentListFromBtn(this) : false;" title="Scorri verso l'alto">↑</button>
              <button type="button" class="segment-scroll-btn" data-segment-scroll="down" data-segment-list-id="${segmentListId}" onclick="return window.__scrollSegmentListFromBtn ? window.__scrollSegmentListFromBtn(this) : false;" title="Scorri verso il basso">↓</button>
            </div>
          </div>
          <div id="${segmentListId}" class="segment-list" tabindex="0" data-segment-current-index="${selectedIdx}" data-focus-run-id="${runIdEsc}" data-focus-img-id="${focusImgId}" data-focus-name-id="${focusNameId}" data-focus-path-id="${focusPathId}" data-focus-seg-id="${focusSegId}" data-focus-detail-id="${focusDetailId}" data-focus-open-btn-id="${focusOpenBtnId}">
            ${listRows}
          </div>
          <div class="small" style="margin-top:6px;">Seleziona una riga per cambiare preview; apri galleria dal pulsante nella sezione rect.</div>
        </div>
      `;
      return {mainHtml, sideHtml, sideListId: segmentListId, selectedIdx};
    }

    function renderRectPerImageEvidence(summary) {
      if (!rectPerImageEvidenceEl) return;
      rectPerImageEvidenceEl.innerHTML = "";
      const payload = getRectPerImagePayload(summary);
      const rectData = payload.rectData;
      const runId = String(selectedRunId || "").trim();
      const rectRedBundle = renderRectRedEvidence(summary, runId);
      const rectRedMainHtml = (rectRedBundle && typeof rectRedBundle.mainHtml === "string") ? rectRedBundle.mainHtml : "";
      const rectRedSideHtml = (rectRedBundle && typeof rectRedBundle.sideHtml === "string") ? rectRedBundle.sideHtml : "";
      const rectRedSideListId = (rectRedBundle && typeof rectRedBundle.sideListId === "string") ? rectRedBundle.sideListId : "";
      const rectRedSelectedIdx = (rectRedBundle && Number.isFinite(Number(rectRedBundle.selectedIdx)))
        ? Math.trunc(Number(rectRedBundle.selectedIdx))
        : 0;
      if (evidenceRectSideHostEl) evidenceRectSideHostEl.innerHTML = rectRedSideHtml || "";
      const keepOpenRectGallery =
        _isModalVisible(rectCompareModalEl) &&
        activeModalGallery === "rect" &&
        rectGalleryRunId === runId;
      if (!keepOpenRectGallery) {
        rectGalleryItems = [];
        rectGalleryIndex = 0;
        rectGalleryRunId = runId;
      }
      if (!rectData) {
        rectPerImageEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Rettangolo Globale Dalla Rete <span class="tag tag-rect">rect</span></div>
            <div class="small">Dati non presenti in questa run.</div>
          </div>
          ${rectRedMainHtml}
        `;
        if (rectRedSideListId) {
          const listEl = document.getElementById(rectRedSideListId);
          if (listEl) setSegmentListActiveIndex(listEl, rectRedSelectedIdx, {behavior: "auto", scrollIntoView: false});
        }
        normalizeRectSegmentRankingPlacement();
        return;
      }

      const available = !!rectData.available;
      const reason = cleanVal(rectData.error || "");
      const modelDevice = cleanVal(rectData.model_device);
      const modelSize = cleanVal(rectData.model_image_size);
      const rotation = cleanVal(rectData.rotation_deg_clockwise);
      const items = payload.items;

      if (!available || !items.length) {
        rectPerImageEvidenceEl.innerHTML = `
          ${rectRedMainHtml}
          <div class="evidence-card">
            <div class="evidence-title">Rettangolo Globale Dalla Rete <span class="tag tag-rect">rect</span></div>
            <div class="small">Analisi non disponibile: ${esc(reason || "-")}</div>
          </div>
        `;
        if (rectRedSideListId) {
          const listEl = document.getElementById(rectRedSideListId);
          if (listEl) setSegmentListActiveIndex(listEl, rectRedSelectedIdx, {behavior: "auto", scrollIntoView: false});
        }
        normalizeRectSegmentRankingPlacement();
        return;
      }

      buildRectGalleryItems(summary, runId);
      const maxItem = items[0];
      const minItem = items[items.length - 1];
      const maxIndex = 0;
      const minIndex = items.length - 1;

      const renderFocusCard = (title, it, idx) => {
        const discrepancy = formatNum(it && it.discrepancy, 4);
        const iou = formatNum(it && it.iou, 4);
        const centerNorm = formatNum(it && it.center_distance_norm, 4);
        const areaRatio = formatNum(it && it.area_ratio_single_vs_global, 4);
        const imageRel = String((it && it.image_rel) || "");
        const overlayRel = String((it && it.overlay_rel) || "");
        const overlayErr = String((it && it.overlay_error) || "");
        const previewRel = overlayRel || imageRel;
        const dims = `${cleanVal(it && it.image_width)}x${cleanVal(it && it.image_height)}`;
        return `
          <div class="rect-focus-card">
            <div class="rect-focus-title">${esc(title)}</div>
            ${previewRel ? `<img src="${imgUrl(previewRel)}" loading="lazy" decoding="async">` : `<div class="small">Preview non disponibile.</div>`}
            <div class="rect-focus-meta">
              <div class="meta-item"><b>Discrepanza</b>${esc(discrepancy)}</div>
              <div class="meta-item"><b>IoU</b>${esc(iou)}</div>
              <div class="meta-item"><b>Center norm</b>${esc(centerNorm)}</div>
              <div class="meta-item"><b>Area ratio</b>${esc(areaRatio)}</div>
              <div class="meta-item"><b>Rect singolo</b>${esc(formatRectTlbr(it && it.single_rect))}</div>
              <div class="meta-item"><b>Rect globale</b>${esc(formatRectTlbr(it && it.global_rect_scaled))}</div>
              <div class="meta-item"><b>Size</b>${esc(dims)}</div>
              <div class="meta-item"><b>Immagine</b>${esc(imageRel || "-")}</div>
            </div>
            <div class="rect-focus-actions">
              <button
                type="button"
                class="btn secondary mini"
                data-open-rect-gallery="1"
                data-open-rect-gallery-mode="rect"
                data-run-id="${esc(runId)}"
                data-start-index="${idx}"
                onclick="return window.__openRectGalleryFromBtn ? window.__openRectGalleryFromBtn(this) : false;"
              >Apri in galleria reti rect</button>
              ${overlayErr ? `<span class="small">overlay_error=${esc(overlayErr)}</span>` : ""}
            </div>
          </div>
        `;
      };

      const globalMedianRect = formatRectTlbr((rectData && rectData.global_rect_median) || {});

      rectPerImageEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Rettangolo Globale Dalla Rete <span class="tag tag-rect">rect</span></div>
          <div class="small">
            Overlay: <b>giallo</b>=rect singolo | <b>rosso</b>=rect globale |
            ordinamento: più discostante -> meno discostante | totale=${items.length}
          </div>
          <div class="rect-focus-meta">
            <div class="meta-item"><b>Device modello</b>${esc(modelDevice)}</div>
            <div class="meta-item"><b>Image size modello</b>${esc(modelSize)}</div>
            <div class="meta-item"><b>Rotazione</b>${esc(rotation)}°</div>
            <div class="meta-item"><b>Rect mediana globale</b>${esc(globalMedianRect)}</div>
          </div>
          <div class="rect-focus-pair">
            ${renderFocusCard("Discrepanza massima", maxItem, maxIndex)}
            ${renderFocusCard("Discrepanza minima", minItem, minIndex)}
          </div>
          <div class="rect-focus-actions">
            <button type="button" class="btn secondary mini" data-open-rect-gallery="1" data-open-rect-gallery-mode="rect" data-run-id="${esc(runId)}" data-start-index="0" onclick="return window.__openRectGalleryFromBtn ? window.__openRectGalleryFromBtn(this) : false;">Apri Galleria Reti Rect</button>
          </div>
        </div>
        ${rectRedMainHtml}
      `;
      if (rectRedSideListId) {
        const listEl = document.getElementById(rectRedSideListId);
        if (listEl) setSegmentListActiveIndex(listEl, rectRedSelectedIdx, {behavior: "auto", scrollIntoView: false});
      }
      normalizeRectSegmentRankingPlacement();
    }

    function normalizeRectSegmentRankingPlacement() {
      if (!rectPerImageEvidenceEl || !evidenceRectSideHostEl) return;
      const stray = rectPerImageEvidenceEl.querySelector(".segment-list-wrapper");
      if (!stray) return;
      stray.classList.add("rect-side-ranking");
      evidenceRectSideHostEl.appendChild(stray);
    }

    function renderSuGiuEvidence(summary) {
      if (!suGiuEvidenceEl) return;
      suGiuEvidenceEl.innerHTML = "";
      const suGiuEvidenceHelp = helpTipHtml(
        "Orientamento SU/GIU per frame",
        [
          "Classificazione eseguita su ogni frame, non una sola etichetta per cartella.",
          "I conteggi su/giu/other servono per valutare coerenza e rumore del set.",
          "Apri galleria o cartelle SU/GIU per ispezione visuale rapida.",
        ],
      );
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object")
        ? rec.model_evidence
        : {};
      const payload = getSuGiuPerImagePayload(summary);
      const sgData = payload.sgData;
      const items = payload.items;
      const counts = payload.labelCounts || {su: 0, giu: 0, other: 0};
      const splitFolders = (payload.splitFolders && typeof payload.splitFolders === "object") ? payload.splitFolders : {};
      const groups = (payload.groups && typeof payload.groups === "object")
        ? payload.groups
        : {su: [], giu: [], other: []};

      if (!sgData) {
        suGiuEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Orientamento SU/GIU per frame ${suGiuEvidenceHelp} <span class="tag tag-rotation">su/giu</span></div>
            <div class="small">Dati non presenti in questa run.</div>
          </div>
        `;
        return;
      }

      const available = !!sgData.available;
      const reason = cleanVal(sgData.error || "");
      const note = cleanVal(sgData.note || "Etichettatura per singolo frame.");
      if (!available || !items.length) {
        suGiuEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Orientamento SU/GIU per frame ${suGiuEvidenceHelp} <span class="tag tag-rotation">su/giu</span></div>
            <div class="small">${esc(note)}</div>
            <div class="small">Analisi non disponibile: ${esc(reason || "-")}</div>
          </div>
        `;
        return;
      }

      const renderFolder = (key, title, tagClass, absDir, relDir, itemsList) => {
        const rows = Array.isArray(itemsList) ? itemsList : [];
        const shown = rows.slice(0, 24);
        const cardsHtml = shown.length
          ? shown.map((it) => {
              const rel = String((it && it.image_rel) || "");
              const frameIdx = cleanVal(it && it.image_index);
              const conf = formatNum(it && it.confidence, 4);
              const pSu = formatNum(it && it.prob_su, 4);
              const pGiu = formatNum(it && it.prob_giu, 4);
              return `
                <div class="sugiu-folder-card">
                  ${rel ? `<img src="${imgUrl(rel)}" loading="lazy" decoding="async">` : `<div class="small">Preview non disponibile.</div>`}
                  <div class="meta">frame=${esc(String(frameIdx))} | conf=${esc(conf)}</div>
                  <div class="meta">p_su=${esc(pSu)} | p_giu=${esc(pGiu)}</div>
                  <div class="meta">${esc(rel || "-")}</div>
                </div>
              `;
            }).join("")
          : `<div class="small">Nessuna immagine in questa cartella.</div>`;
        const absPath = String(absDir || "").trim();
        const relPath = String(relDir || "").trim();
        const finderBtn = absPath
          ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(absPath)}">Apri Finder</button>`
          : "";
        const shownNote = rows.length > shown.length
          ? `<div class="small">Mostrate ${shown.length}/${rows.length} immagini. Usa la galleria per vederle tutte.</div>`
          : "";
        const keyClass = String(key || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "");
        return `
          <div class="sugiu-folder${keyClass ? ` sugiu-folder-${esc(keyClass)}` : ""}">
            <div class="sugiu-folder-head">
              <div class="sugiu-folder-title">${esc(title)} <span class="tag ${esc(tagClass)}">${esc(key)}</span></div>
              ${finderBtn}
            </div>
            <div class="sugiu-folder-path">Path: ${esc(relPath || absPath || "-")}</div>
            <div class="small">Totale immagini: ${rows.length}</div>
            <div class="sugiu-folder-grid">${cardsHtml}</div>
            ${shownNote}
          </div>
        `;
      };

      const suRows = Array.isArray(groups.su) ? groups.su : [];
      const giuRows = Array.isArray(groups.giu) ? groups.giu : [];
      const suFolderHtml = renderFolder(
        "su",
        "Cartella SU",
        "tag-rotation",
        splitFolders.su_dir_abs,
        splitFolders.su_dir_rel,
        suRows,
      );
      const giuFolderHtml = renderFolder(
        "giu",
        "Cartella GIU",
        "tag-rotation",
        splitFolders.giu_dir_abs,
        splitFolders.giu_dir_rel,
        giuRows,
      );
      const otherCount = Array.isArray(groups.other) ? groups.other.length : 0;
      const suGiuModelPath = String(model.su_giu_checkpoint_used || model.su_giu_checkpoint_global || "").trim();

      suGiuEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Orientamento SU/GIU per frame ${suGiuEvidenceHelp} <span class="tag tag-rotation">su/giu</span></div>
          <div class="dup-actions">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="su_giu" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Apri galleria</button>
          </div>
          <div class="small" style="margin-top:6px;">
            ${esc(note)} | totale=${items.length} | su=${cleanVal(counts.su)} | giu=${cleanVal(counts.giu)} | other=${cleanVal(counts.other)}
          </div>
          <div class="small">
            rete usata: <span class="path">${esc(suGiuModelPath || "-")}</span>
          </div>
          <div class="small">
            Visualizzazione a due cartelle separate (SU/GIU), ordinate per confidence decrescente. Nessun giudizio globale sulla cartella sorgente.
          </div>
          <div class="sugiu-folders">
            ${suFolderHtml}
            ${giuFolderHtml}
          </div>
          ${otherCount > 0 ? `<div class="small" style="margin-top:8px;">Frame con label diversa da su/giu: ${otherCount} (cartella "other" disponibile su filesystem).</div>` : ""}
        </div>
      `;
    }

    function renderLrMarkerEvidence(summary) {
      if (!lrMarkerEvidenceEl) return;
      lrMarkerEvidenceEl.innerHTML = "";
      const lrMarkerHelp = helpTipHtml(
        "LR marker classico",
        [
          "Usa i template orientation_* in DB_echo del vendor predetto.",
          "Prima cerca nella metà suggerita da SU/GIU, poi full-crop se score < 0.55.",
          "Se resta sotto 0.66 espande progressivamente fuori dal rettangolo eco.",
        ],
      );
      const row = (summary && summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const payload = getLrMarkerPerImagePayload(summary);
      const lrData = payload.lrData;
      const items = payload.items || [];
      const counts = payload.labelCounts || {};
      const statusCounts = payload.statusCounts || {};
      const quadrantCounts = payload.quadrantCounts || {};
      const strategyCounts = payload.strategyCounts || {};
      const best = payload.best || {};
      const orientationGroups = Array.isArray(payload.orientationGroups) ? payload.orientationGroups : [];
      const lrReliableThreshold = Math.max(
        LR_MARKER_RELIABLE_MATCH_SCORE,
        Number.isFinite(Number(model.lr_marker_min_match_score)) ? Number(model.lr_marker_min_match_score) : LR_MARKER_RELIABLE_MATCH_SCORE,
      );
      const templatePathCounts = (payload.templatePathCounts && typeof payload.templatePathCounts === "object")
        ? payload.templatePathCounts
        : {};

      if (!lrData) {
        lrMarkerEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">LR marker classico ${lrMarkerHelp} <span class="tag tag-rotation">lr</span></div>
            <div class="small">Dati non presenti in questa run.</div>
          </div>
        `;
        return;
      }

      const templatePathsSeen = Object.keys(templatePathCounts).filter(Boolean);
      const fixedTemplatePath = String(lrData.fixed_template_path || best.folder_fixed_template_path || row.lr_marker_best_template_path || best.template_path || "").trim();
      const templatePolicy = String(lrData.template_policy || best.template_policy || "").trim();
      const templatePolicyEffective = String(lrData.template_policy_effective || best.template_policy_effective || row.lr_marker_template_policy_effective || "").trim();
      const templateFallbackReason = String(lrData.template_fallback_reason || best.template_fallback_reason || row.lr_marker_template_fallback_reason || "").trim();
      const templateFallbackHtml = templateFallbackReason
        ? `<div class="small" style="margin-top:8px;color:#b45309;"><span class="tag tag-warn">fallback attivo</span> historical_best -> derived_folder<br>motivo: ${esc(templateFallbackReason)}</div>`
        : "";
      const templateSelectionScore = lrData.fixed_template_selection_score || best.folder_fixed_template_selection_score || row.lr_marker_best_score || "";
      const available = !!lrData.available;
      const reason = cleanVal(lrData.error || "");
      const note = cleanVal(lrData.note || "LR marker classico frame-by-frame.");
      if (!available || !items.length) {
        lrMarkerEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">LR marker classico ${lrMarkerHelp} <span class="tag tag-rotation">lr</span></div>
            <div class="small">${esc(note)}</div>
            <div class="small">Analisi non disponibile: ${esc(reason || "-")}</div>
            <div class="small">source=${esc(String(row.lr_marker_source || "-"))}</div>
          </div>
        `;
        return;
      }

      const strategyTxt = Object.entries(strategyCounts)
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => `${k}:${v}`)
        .join(" | ");
      const bestScore = best.match_score !== undefined ? best.match_score : row.lr_marker_best_score;
      const shown = items.slice(0, 32);
      const groupCardsHtml = orientationGroups.length
        ? orientationGroups.map((group) => {
            const env = group.envelope || null;
            const scoreTxt = [
              `min=${formatNum(group.min_score, 4)}`,
              `avg=${formatNum(group.avg_score, 4)}`,
              `max=${formatNum(group.max_score, 4)}`,
            ].join(" | ");
            return `
              <div class="lr-orientation-card">
                <div class="head">
                  <div class="title">${esc(group.key)}</div>
                  <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker" data-lr-marker-group="${esc(group.key)}">Apri gruppo</button>
                </div>
                <div class="box">${esc(formatLrEnvelope(env))}</div>
                <div class="meta">box da frame validi >= ${esc(formatNum(lrReliableThreshold, 2))}: ${esc(cleanVal(env && env.boxes))}/${esc(cleanVal(group.items.length))} | esclusi low-score=${esc(cleanVal(group.low_score_count || 0))}, non-canonici=${esc(cleanVal(group.noncanonical_count || 0))}, neri=${esc(cleanVal(group.blank_match_count || 0))}, quadrante=${esc(cleanVal(group.quadrant_invalid_count || 0))}</div>
                <div class="meta">score ${esc(scoreTxt)}</div>
                <div class="meta">label: ${esc(lrCountsToText(group.label_counts))}</div>
                <div class="meta">strategie: ${esc(lrCountsToText(group.strategy_counts))}</div>
              </div>
            `;
          }).join("")
        : `<div class="small">Nessun gruppo orientamento riconosciuto dai nomi immagine.</div>`;
      const cardsHtml = shown.map((it) => {
        const label = String((it && it.lr_label) || "-");
        const labelIt = String((it && it.lr_label_it) || "");
        const score = formatNum(it && it.match_score, 4);
        const strategy = String((it && it.search_strategy) || "-");
        const scope = String((it && it.search_scope) || "-");
        const side = String((it && it.detected_marker_side) || "-");
        const idx = cleanVal(it && it.image_index);
        const marker = `${cleanVal(it && it.marker_top_abs)}|${cleanVal(it && it.marker_left_abs)}|${cleanVal(it && it.marker_bottom_abs)}|${cleanVal(it && it.marker_right_abs)}|`;
        const quadrantTag = lrMarkerQuadrantTagHtml(it);
        const quadrantText = lrMarkerQuadrantText(it);
        return `
          <div class="sugiu-folder-card">
            ${lrMarkerVisualHtml(it, selectedRunId)}
            <div class="meta">frame=${esc(String(idx))} | score=${esc(score)} | ${esc(strategy)} | ${esc(scope)}</div>
            <div class="meta">label=${esc(label)}${labelIt ? ` (${esc(labelIt)})` : ""} | marker=${esc(side)}</div>
            <div class="meta">${quadrantTag} ${esc(quadrantText)}</div>
            <div class="meta">marker_abs=${esc(marker)}</div>
          </div>
        `;
      }).join("");
      const lowScoreReviewHtml = renderLrLowScoreReviewSection(items, orientationGroups, lrReliableThreshold, payload.targetWidth, payload.targetHeight);

      lrMarkerEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">LR marker classico ${lrMarkerHelp} <span class="tag tag-rotation">lr</span></div>
          <div class="dup-actions">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="lr_marker" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Apri galleria</button>
          </div>
          <div class="detail-list" style="margin-top:8px;">
            <div>
              <b>Aggregato batch</b><br>
              majority=${esc(cleanVal(row.lr_marker_majority_label || "-"))} | vote_ratio=${esc(cleanVal(row.lr_marker_majority_vote_ratio || "-"))}<br>
              best=${esc(cleanVal(row.lr_marker_best_label || best.lr_label || "-"))} | score=${esc(cleanVal(bestScore || "-"))}<br>
              frame=${esc(cleanVal(row.lr_marker_images_predicted || lrData.images_total || "-"))} | source=${esc(cleanVal(row.lr_marker_source || "-"))}
            </div>
            <div>
              <b>Template fisso cartella</b><br>
              <span class="path">${esc(fixedTemplatePath || "-")}</span><br>
              policy=${esc(templatePolicy || "-")}${templatePolicyEffective ? ` | effective=${esc(templatePolicyEffective)}` : ""} | selection_score=${esc(cleanVal(templateSelectionScore || "-"))}
              ${templateFallbackHtml}
            </div>
          </div>
          <div class="small" style="margin-top:8px;">
            ${esc(note)} | not_lr_flipped=${esc(cleanVal(counts.not_lr_flipped))} | lr_flipped=${esc(cleanVal(counts.lr_flipped))} | other=${esc(cleanVal(counts.other))}
          </div>
          <div class="small">
            Nota: la majority ha senso solo su una cartella/acquisizione omogenea. Se la cartella contiene NF/LR/UD/LRUD misti, questa sezione va letta come audit per-frame.
          </div>
          ${templatePathsSeen.length > 1 ? `<div class="small" style="color:#b45309;">Questa run usa ancora piu template (${templatePathsSeen.length}). Riesegui la pipeline per applicare la nuova regola: un solo template fisso per tutta la cartella.</div>` : ""}
          <div class="small">
            status ok=${esc(cleanVal(statusCounts.ok))} | review=${esc(cleanVal(statusCounts.review))} | quadranti ok=${esc(cleanVal(quadrantCounts.ok))} invalid=${esc(cleanVal(quadrantCounts.invalid))} | strategie=${esc(strategyTxt || "-")}
          </div>
          <div class="small">
            template mode: ${esc(cleanVal(model.lr_marker_template_policy || "-"))} | roots template: ${esc(Array.isArray(model.lr_marker_template_roots) ? model.lr_marker_template_roots.join(" | ") : "-")}
          </div>
          <div class="evidence-title" style="margin-top:10px;">Box marker per orientamento <span class="tag tag-data">envelope</span></div>
          <div class="small">Coordinate assolute del rettangolo che contiene tutte le posizioni marker trovate nel gruppo.</div>
          <div class="lr-orientation-groups">${groupCardsHtml}</div>
          ${lowScoreReviewHtml}
          <div class="evidence-title" style="margin-top:10px;">Miglior match visivo <span class="tag tag-data">best</span></div>
          <div class="sugiu-folder-card">${lrMarkerVisualHtml(best, selectedRunId)}</div>
          <div class="evidence-title" style="margin-top:10px;">Frame ordinati per score <span class="tag tag-data">audit</span></div>
          <div class="sugiu-folder-grid" style="margin-top:10px;">${cardsHtml}</div>
          ${items.length > shown.length ? `<div class="small" style="margin-top:8px;">Mostrati ${shown.length}/${items.length} frame. Usa la galleria per vederli tutti ordinati per score.</div>` : ""}
        </div>
      `;
    }

    function renderLtEvidence(summary) {
      if (!ltEvidenceEl) return;
      ltEvidenceEl.innerHTML = "";
      const ltEvidenceHelp = helpTipHtml(
        "Classificazione L/T per frame",
        [
          "Classificazione L/T sui crop del rettangolo per ogni frame.",
          "I conteggi L/T/other mostrano stabilità o ambiguità del contenuto.",
          "Usa galleria e cartelle L/T per verificare rapidamente i casi dubbi.",
        ],
      );
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object")
        ? rec.model_evidence
        : {};
      const ltPayload = getLtPerImagePayload(summary);
      const ltData = ltPayload.ltData;
      const ltItems = ltPayload.items || [];
      const ltCounts = ltPayload.labelCounts || {l: 0, t: 0, other: 0};
      const ltSplitFolders = (ltPayload.splitFolders && typeof ltPayload.splitFolders === "object") ? ltPayload.splitFolders : {};
      const ltGroups = (ltPayload.groups && typeof ltPayload.groups === "object")
        ? ltPayload.groups
        : {l: [], t: [], other: []};

      if (!ltData) {
        ltEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Classificazione L/T per frame ${ltEvidenceHelp} <span class="tag tag-rotation">lt</span></div>
            <div class="small">Dati non presenti in questa run.</div>
          </div>
        `;
        return;
      }

      const ltAvailable = !!(ltData.available && ltItems.length);
      const ltReason = cleanVal(ltData.error || "");
      const ltNote = cleanVal(ltData.note || "Etichettatura L/T per frame.");
      if (!ltAvailable) {
        ltEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Classificazione L/T per frame ${ltEvidenceHelp} <span class="tag tag-rotation">lt</span></div>
            <div class="small">${esc(ltNote)}</div>
            <div class="small">Analisi non disponibile: ${esc(ltReason || "-")}</div>
          </div>
        `;
        return;
      }

      const renderFolderLt = (key, title, tagClass, absDir, relDir, itemsList) => {
        const rows = Array.isArray(itemsList) ? itemsList : [];
        const shown = rows.slice(0, 24);
        const cardsHtml = shown.length
          ? shown.map((it) => {
              const rel = String((it && it.image_rel) || "");
              const frameIdx = cleanVal(it && it.image_index);
              const conf = formatNum(it && it.confidence, 4);
              const pL = formatNum(it && it.prob_l, 4);
              const pT = formatNum(it && it.prob_t, 4);
              return `
                <div class="sugiu-folder-card">
                  ${rel ? `<img src="${imgUrl(rel)}" loading="lazy" decoding="async">` : `<div class="small">Preview non disponibile.</div>`}
                  <div class="meta">frame=${esc(String(frameIdx))} | conf=${esc(conf)}</div>
                  <div class="meta">p_l=${esc(pL)} | p_t=${esc(pT)}</div>
                  <div class="meta">${esc(rel || "-")}</div>
                </div>
              `;
            }).join("")
          : `<div class="small">Nessuna immagine in questa cartella.</div>`;
        const absPath = String(absDir || "").trim();
        const relPath = String(relDir || "").trim();
        const finderBtn = absPath
          ? `<button type="button" class="btn secondary mini" data-open-folder-path="${esc(absPath)}">Apri Finder</button>`
          : "";
        const shownNote = rows.length > shown.length
          ? `<div class="small">Mostrate ${shown.length}/${rows.length} immagini. Usa la galleria per vederle tutte.</div>`
          : "";
        const keyClass = String(key || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "");
        return `
          <div class="sugiu-folder${keyClass ? ` sugiu-folder-${esc(keyClass)}` : ""}">
            <div class="sugiu-folder-head">
              <div class="sugiu-folder-title">${esc(title)} <span class="tag ${esc(tagClass)}">${esc(key)}</span></div>
              ${finderBtn}
            </div>
            <div class="sugiu-folder-path">Path: ${esc(relPath || absPath || "-")}</div>
            <div class="small">Totale immagini: ${rows.length}</div>
            <div class="sugiu-folder-grid">${cardsHtml}</div>
            ${shownNote}
          </div>
        `;
      };

      const lRows = Array.isArray(ltGroups.l) ? ltGroups.l : [];
      const tRows = Array.isArray(ltGroups.t) ? ltGroups.t : [];
      const lFolderHtml = renderFolderLt(
        "L",
        "Cartella L",
        "tag-rotation",
        ltSplitFolders.l_dir_abs,
        ltSplitFolders.l_dir_rel,
        lRows,
      );
      const tFolderHtml = renderFolderLt(
        "T",
        "Cartella T",
        "tag-rotation",
        ltSplitFolders.t_dir_abs,
        ltSplitFolders.t_dir_rel,
        tRows,
      );
      const ltOtherCount = Array.isArray(ltGroups.other) ? ltGroups.other.length : 0;
      const ltModelPath = String(model.lt_checkpoint_used || model.lt_checkpoint_global || "").trim();

      ltEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Classificazione L/T per frame ${ltEvidenceHelp} <span class="tag tag-rotation">lt</span></div>
          <div class="dup-actions">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="lt" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Apri galleria</button>
          </div>
          <div class="small" style="margin-top:6px;">
            ${esc(ltNote)} | totale=${ltItems.length} | L=${cleanVal(ltCounts.l)} | T=${cleanVal(ltCounts.t)} | other=${cleanVal(ltCounts.other)}
          </div>
          <div class="small">
            rete usata: <span class="path">${esc(ltModelPath || "-")}</span>
          </div>
          <div class="small">
            Visualizzazione a due cartelle separate (L/T). Nessun giudizio globale sulla cartella sorgente.
          </div>
          <div class="sugiu-folders">
            ${lFolderHtml}
            ${tFolderHtml}
          </div>
          ${ltOtherCount > 0 ? `<div class="small" style="margin-top:8px;">Frame con label diversa da L/T: ${ltOtherCount} (cartella "other" disponibile su filesystem).</div>` : ""}
        </div>
      `;
    }

    function renderDuplicateEvidence(summary) {
      dupEvidenceEl.innerHTML = "";
      const dups = (summary && Array.isArray(summary.duplicate_examples)) ? summary.duplicate_examples : [];
      const recCount = Number(summary && summary.duplicates_removed_count_recomputed ? summary.duplicates_removed_count_recomputed : 0);
      if (!dups.length) {
        dupEvidenceEl.innerHTML = `
          <div class="evidence-card">
            <div class="evidence-title">Duplicati rimossi <span class="tag tag-data">dedup</span></div>
            <div class="small">Nessun esempio duplicato disponibile. Totale ricalcolato: ${recCount}</div>
          </div>
        `;
        return;
      }
      const previewItems = dups.slice(0, 12);
      const cards = previewItems.map((d, i) => {
        const kept = String((d && d.kept_rel) || "");
        const removed = String((d && d.removed_rel) || "");
        const sha = String((d && d.sha1_prefix) || "");
        const sz = String((d && d.size_bytes) || "");
        return `
          <div class="evidence-card">
            <div class="evidence-title">Duplicato #${i + 1} (sha1 ${esc(sha)}, ${esc(sz)} bytes)</div>
            <div class="pair-grid">
              <div class="pair-item">
                <div class="label">Tenuta (kept)</div>
                <img src="${imgUrl(kept)}" loading="lazy">
                <div class="path">${esc(kept)}</div>
              </div>
              <div class="pair-item">
                <div class="label">Rimossa in dedup (removed)</div>
                <img src="${imgUrl(removed)}" loading="lazy">
                <div class="path">${esc(removed)}</div>
              </div>
            </div>
            <div class="dup-actions">
              <button
                type="button"
                class="btn secondary mini"
                data-open-dup-compare="1"
                data-run-id="${esc(String(selectedRunId || ""))}"
                data-kept="${esc(kept)}"
                data-removed="${esc(removed)}"
                data-sha="${esc(sha)}"
                data-size="${esc(sz)}"
                onclick="return window.__openDupCompareFromBtn ? window.__openDupCompareFromBtn(this) : false;"
              >Apri confronto grande</button>
            </div>
          </div>
        `;
      }).join("");
      dupEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Duplicati rimossi <span class="tag tag-data">dedup</span></div>
          <div class="dup-actions">
            <button type="button" class="btn secondary mini" data-open-evidence-gallery="duplicates" onclick="return window.__openEvidenceGalleryFromBtn ? window.__openEvidenceGalleryFromBtn(this) : false;">Apri Galleria Pro</button>
          </div>
          <div class="small" style="margin-top:6px;">Totale dedup ricalcolato: ${recCount || dups.length} | esempi disponibili: ${dups.length}</div>
          ${cards}
          ${dups.length > previewItems.length ? `<div class="small" style="margin-top:8px;">Mostrati ${previewItems.length}/${dups.length} esempi. Usa "Apri Galleria Pro" per vederli tutti.</div>` : ""}
        </div>
      `;
    }

    function currentSelectedRunRecord() {
      const rid = String(selectedRunId || "").trim();
      if (!rid) return {};
      return runs.find((run) => String(run && run.run_id || "") === rid) || {};
    }

    function rawInputFolderForSummary(summary) {
      const sm = (summary && typeof summary === "object") ? summary : {};
      const runRecord = currentSelectedRunRecord();
      const candidates = [
        selectedRunStatus && selectedRunStatus.input_folder,
        runRecord && runRecord.input_folder,
        sm.input_original_folder,
        sm.input_folder,
        (document.getElementById("inputFolder") || {}).value,
      ];
      for (const value of candidates) {
        const txt = String(value || "").trim();
        if (txt) return txt;
      }
      return "";
    }

    function joinPreviewPath(basePath, relPath) {
      const rel = String(relPath || "").trim();
      if (!rel) return "";
      if (new RegExp("^([A-Za-z]:[\\\\/]|[\\\\/])").test(rel)) return rel;
      const base = String(basePath || "").trim().replace(/[\\/]+$/g, "");
      const cleanRel = rel.replace(/^[\\/]+/g, "");
      return base ? `${base}/${cleanRel}` : cleanRel;
    }

    function rawRowsFromSummary(summary) {
      const sm = (summary && typeof summary === "object") ? summary : {};
      const imgList = Array.isArray(sm.raw_images_rel) && sm.raw_images_rel.length
        ? sm.raw_images_rel
        : (Array.isArray(sm.preview_images_rel) ? sm.preview_images_rel : []);
      return imgList
        .map((rel, idx) => ({
          idx,
          rel: String(rel || ""),
          displayRel: String(rel || ""),
          previewPath: String(rel || ""),
          sizeBytes: 0,
          excluded: false,
          canExclude: false,
        }))
        .filter((row) => row.rel);
    }

    function selectedRawExcludedImages() {
      return rawExclusionRows
        .filter((row) => row && row.canExclude !== false && !!row.excluded)
        .map((row) => String(row.rel || "").trim())
        .filter(Boolean);
    }

    function updateRawExclusionStatus(inputFolder, totalOverride) {
      if (!rawExclusionStatusEl) return;
      const total = Number.isFinite(Number(totalOverride)) ? Number(totalOverride) : rawExclusionRows.length;
      const excluded = selectedRawExcludedImages().length;
      const folderName = String(inputFolder || "").split(/[\\/]/).filter(Boolean).pop() || "-";
      rawExclusionStatusEl.textContent = `${excluded} escluse su ${total} immagini | ${folderName}`;
    }

    function setRawExcluded(relValue, checkedValue) {
      const rel = String(relValue || "").trim();
      if (!rel) return;
      const checked = !!checkedValue;
      rawExclusionRows.forEach((row) => {
        if (String(row && row.rel || "") === rel) row.excluded = checked;
      });
      document.querySelectorAll("input[data-raw-image-rel]").forEach((input) => {
        if (String(input.getAttribute("data-raw-image-rel") || "") === rel) {
          input.checked = checked;
          const tile = input.closest(".raw-tile, .gallery-modal-item");
          if (tile) tile.classList.toggle("excluded", checked);
        }
      });
      updateRawExclusionStatus(rawInputFolderForSummary(selectedRunData && selectedRunData.summary));
    }

    function bindRawExclusionInputs(rootEl) {
      const scope = (rootEl && typeof rootEl.querySelectorAll === "function") ? rootEl : document;
      scope.querySelectorAll("input[data-raw-image-rel]").forEach((input) => {
        if (input.dataset.rawExclusionBound === "1") return;
        input.dataset.rawExclusionBound = "1";
        input.addEventListener("change", () => {
          setRawExcluded(input.getAttribute("data-raw-image-rel"), input.checked);
        });
      });
    }

    function renderRawGalleryRows(rows, inputFolder) {
      const normalized = (Array.isArray(rows) ? rows : [])
        .map((row, idx) => {
          const rel = String(row && row.rel || row && row.image_rel || "").trim();
          const previewPath = String(row && row.previewPath || row && row.preview_path || "").trim()
            || joinPreviewPath(inputFolder, rel);
          return {
            idx: Number.isFinite(Number(row && row.idx)) ? Number(row.idx) : idx,
            rel,
            displayRel: String(row && row.displayRel || rel),
            previewPath,
            sizeBytes: Number(row && row.sizeBytes !== undefined ? row.sizeBytes : row && row.size_bytes) || 0,
            excluded: !!(row && row.excluded),
            canExclude: row && row.canExclude === false ? false : true,
          };
        })
        .filter((row) => row.rel && row.previewPath);

      rawExclusionRows = normalized;
      if (!galleryEl) return;
      if (!normalized.length) {
        galleryEl.innerHTML = "<div class='small'>Nessuna immagine raw disponibile.</div>";
        updateRawExclusionStatus(inputFolder, 0);
        return;
      }
      galleryEl.innerHTML = normalized.map((row, idx) => {
        const src = imgUrlForRun(selectedRunId, row.previewPath);
        const checked = row.excluded ? " checked" : "";
        const excludeHtml = row.canExclude
          ? `<label class="raw-exclude-toggle"><input type="checkbox" data-raw-image-rel="${esc(row.rel)}"${checked}> Escludi</label>`
          : "";
        return `
          <div class="tile raw-tile${row.excluded ? " excluded" : ""}">
            <button type="button" class="raw-image-button" data-open-evidence-gallery="raw" data-raw-gallery-index="${idx}">
              <img src="${src}" loading="lazy" decoding="async">
            </button>
            ${excludeHtml}
            <div class="p">${esc(row.displayRel || row.rel)}</div>
          </div>
        `;
      }).join("");
      bindRawExclusionInputs(galleryEl);
      updateRawExclusionStatus(inputFolder, normalized.length);
    }

    async function saveRawExclusions() {
      const summary = selectedRunData && selectedRunData.summary ? selectedRunData.summary : null;
      const inputFolder = rawInputFolderForSummary(summary);
      if (!inputFolder) {
        if (rawExclusionStatusEl) rawExclusionStatusEl.textContent = "Cartella input non disponibile.";
        return;
      }
      const excluded = selectedRawExcludedImages();
      const out = await apiPost("/api/input/exclusions", {
        input_folder: inputFolder,
        excluded_images_rel: excluded,
      });
      if (summary && typeof summary === "object") {
        summary.excluded_images_rel = excluded.slice();
        summary.excluded_images_count = excluded.length;
      }
      if (rawExclusionStatusEl) {
        rawExclusionStatusEl.textContent = `${Number(out.excluded_count || excluded.length)} esclusioni salvate.`;
      }
    }

    function clearRawExclusions() {
      rawExclusionRows.forEach((row) => {
        if (row && row.canExclude !== false) row.excluded = false;
      });
      document.querySelectorAll("input[data-raw-image-rel]").forEach((input) => {
        input.checked = false;
        const tile = input.closest(".raw-tile, .gallery-modal-item");
        if (tile) tile.classList.remove("excluded");
      });
      updateRawExclusionStatus(rawInputFolderForSummary(selectedRunData && selectedRunData.summary));
    }

    function renderRawEvidence(summary) {
      const inputFolder = rawInputFolderForSummary(summary);
      const fallbackRows = rawRowsFromSummary(summary);
      rawEvidenceLoadToken += 1;
      const token = rawEvidenceLoadToken;
      renderRawGalleryRows(fallbackRows, inputFolder);
      if (!inputFolder) return;
      if (rawExclusionStatusEl) rawExclusionStatusEl.textContent = "Caricamento immagini raw...";
      apiPost("/api/input/images", {input_folder: inputFolder})
        .then((out) => {
          if (token !== rawEvidenceLoadToken || String(activeEvidenceSection || "") !== "raw") return;
          const rows = Array.isArray(out.images) ? out.images.map((img, idx) => ({
            idx,
            rel: String(img && img.rel || ""),
            displayRel: String(img && img.rel || ""),
            previewPath: joinPreviewPath(inputFolder, String(img && img.rel || "")),
            sizeBytes: Number(img && img.size_bytes) || 0,
            excluded: !!(img && img.excluded),
            canExclude: true,
          })) : [];
          renderRawGalleryRows(rows, inputFolder);
        })
        .catch((err) => {
          if (token !== rawEvidenceLoadToken || String(activeEvidenceSection || "") !== "raw") return;
          if (rawExclusionStatusEl) rawExclusionStatusEl.textContent = `Errore caricamento raw: ${String(err && err.message ? err.message : err)}`;
        });
    }

    function renderScaleEvidence(summary) {
      if (!scaleEvidenceEl) return;
      scaleEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Scala <span class="tag tag-data">scale</span></div>
        </div>
      `;
    }

    function renderDepthEvidence(summary) {
      if (!depthEvidenceEl) return;
      depthEvidenceEl.innerHTML = `
        <div class="evidence-card">
          <div class="evidence-title">Depth <span class="tag tag-data">depth</span></div>
          <div class="small">Nessuna evidenza depth disponibile.</div>
        </div>
      `;
    }

    function bindImageErrorFallback(rootEl) {
      const scope = (rootEl && typeof rootEl.querySelectorAll === "function") ? rootEl : document;
      scope.querySelectorAll("img").forEach((img) => {
        if (!img || img.dataset.errBound === "1") return;
        img.dataset.errBound = "1";
        img.addEventListener("error", () => {
          if (img.dataset.errShown === "1") return;
          img.dataset.errShown = "1";
          const fallback = document.createElement("div");
          fallback.className = "small";
          fallback.textContent = "Immagine non disponibile.";
          fallback.style.padding = "8px 6px";
          fallback.style.border = "1px dashed #cbd5e1";
          fallback.style.borderRadius = "6px";
          fallback.style.background = "#f8fafc";
          fallback.style.color = "#64748b";
          if (img.parentNode) {
            img.parentNode.replaceChild(fallback, img);
          }
        });
      });
    }

    function setEvidenceStatusForSection(summary, sectionRaw) {
      if (!evidenceStatusEl) return;
      const sm = (summary && typeof summary === "object") ? summary : {};
      const rec = (sm.recognition_evidence && typeof sm.recognition_evidence === "object") ? sm.recognition_evidence : {};
      const section = String(sectionRaw || "raw");
      if (section === "raw") {
        evidenceStatusEl.textContent = `Immagini raw: ${cleanVal(sm.raw_images_count)}`;
      } else if (section === "duplicates") {
        evidenceStatusEl.textContent = `Duplicati: ${cleanVal(sm.duplicates_removed_count_recomputed)}`;
      } else if (section === "rotation") {
        const rot = (sm.rotation_evidence && typeof sm.rotation_evidence === "object") ? sm.rotation_evidence : {};
        evidenceStatusEl.textContent = `Rotazione: ${cleanVal(rot.rotation_deg_clockwise)} deg`;
      } else if (section === "rect") {
        const rect = (rec.rect_per_image && typeof rec.rect_per_image === "object") ? rec.rect_per_image : {};
        evidenceStatusEl.textContent = `Rect: ${cleanVal(rect.images_total || sm.raw_images_count)} frame`;
      } else if (section === "template") {
        evidenceStatusEl.textContent = "Template";
      } else if (section === "orientation") {
        const lr = (rec.lr_marker_per_image && typeof rec.lr_marker_per_image === "object") ? rec.lr_marker_per_image : {};
        evidenceStatusEl.textContent = `Orientamento: ${cleanVal(lr.images_total || sm.raw_images_count)} frame`;
      } else if (section === "lt") {
        const lt = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
        evidenceStatusEl.textContent = `L/T: ${cleanVal(lt.images_total || sm.raw_images_count)} frame`;
      } else if (section === "scale") {
        evidenceStatusEl.textContent = "Scala";
      } else if (section === "depth") {
        evidenceStatusEl.textContent = "Depth";
      } else {
        evidenceStatusEl.textContent = "Evidenze";
      }
    }

    function renderActiveEvidenceSection(summary) {
      decisionEvidenceEl.innerHTML = "";
      if (templateEvidenceEl) templateEvidenceEl.innerHTML = "";
      rotationEvidenceEl.innerHTML = "";
      if (rectPerImageEvidenceEl) rectPerImageEvidenceEl.innerHTML = "";
      if (suGiuEvidenceEl) suGiuEvidenceEl.innerHTML = "";
      if (lrMarkerEvidenceEl) lrMarkerEvidenceEl.innerHTML = "";
      if (ltEvidenceEl) ltEvidenceEl.innerHTML = "";
      if (scaleEvidenceEl) scaleEvidenceEl.innerHTML = "";
      if (depthEvidenceEl) depthEvidenceEl.innerHTML = "";
      dupEvidenceEl.innerHTML = "";
      galleryEl.innerHTML = "";
      if (rawEvidenceBlockEl) rawEvidenceBlockEl.hidden = true;
      if (evidenceRectSideHostEl) evidenceRectSideHostEl.innerHTML = "";

      const section = String(activeEvidenceSection || "raw");
      setEvidenceStatusForSection(summary, section);
      if (section === "raw") {
        if (rawEvidenceBlockEl) rawEvidenceBlockEl.hidden = false;
        renderRawEvidence(summary);
      } else if (section === "decision") {
        renderDecisionEvidence(summary);
      } else if (section === "duplicates") {
        renderDuplicateEvidence(summary);
      } else if (section === "rotation") {
        renderRotationEvidence(summary);
      } else if (section === "orientation") {
        renderOrientationEvidence(summary);
      } else if (section === "su_giu") {
        renderSuGiuEvidence(summary);
      } else if (section === "lr_marker") {
        renderLrMarkerEvidence(summary);
      } else if (section === "lt") {
        renderLtEvidence(summary);
      } else if (section === "scale") {
        renderScaleEvidence(summary);
      } else if (section === "depth") {
        renderDepthEvidence(summary);
      } else if (section === "rect") {
        renderRectPerImageEvidence(summary);
      } else if (section === "template") {
        renderTemplateEvidence(summary);
      }
      bindImageErrorFallback(document);
    }

    function setActiveEvidenceSection(name) {
      const requested = String(name || "");
      const normalized = (requested === "su_giu" || requested === "lr_marker") ? "orientation" : requested;
      const valid = ["raw", "duplicates", "rotation", "rect", "template", "orientation", "lt", "scale", "depth"];
      const next = valid.includes(normalized) ? normalized : "raw";
      activeEvidenceSection = next;
      evidenceSectionButtons.forEach((btn) => {
        const sec = String(btn.getAttribute("data-evidence-section") || "");
        btn.classList.toggle("active", sec === next);
      });
      try {
        window.localStorage.setItem("workbench.activeEvidenceSection", next);
      } catch (err) {
        // no-op
      }
      if (selectedRunData && selectedRunData.summary) {
        renderActiveEvidenceSection(selectedRunData.summary);
      }
    }

    function openDupCompareModal(keptRel, removedRel, shaPrefix, sizeBytes, runIdValue) {
      if (!dupCompareModalEl) return;
      const kept = String(keptRel || "").trim();
      const removed = String(removedRel || "").trim();
      const runForPreview = String(runIdValue || selectedRunId || "").trim();
      if (!runForPreview) return;
      if (!kept || !removed) return;
      const openSig = `dup|${runForPreview}|${kept}|${removed}|${String(shaPrefix || "")}`;
      if (!_allowGalleryOpen(openSig, 220)) return;
      _closeOtherModals("dup");
      if (dupCompareMetaEl) {
        const shaTxt = String(shaPrefix || "").trim() || "-";
        const sizeTxt = String(sizeBytes || "").trim() || "-";
        dupCompareMetaEl.textContent = `run=${runForPreview} | sha1=${shaTxt} | size=${sizeTxt} bytes`;
      }
      if (dupCompareKeptImgEl) dupCompareKeptImgEl.src = imgUrlForRun(runForPreview, kept);
      if (dupCompareRemovedImgEl) dupCompareRemovedImgEl.src = imgUrlForRun(runForPreview, removed);
      if (dupCompareKeptPathEl) dupCompareKeptPathEl.textContent = kept;
      if (dupCompareRemovedPathEl) dupCompareRemovedPathEl.textContent = removed;
      dupCompareModalEl.classList.remove("hidden");
    }

    function closeDupCompareModal() {
      if (!dupCompareModalEl) return;
      dupCompareModalEl.classList.add("hidden");
      if (activeModalGallery === "dup") activeModalGallery = "";
    }

    function closeRectCompareModal() {
      if (!rectCompareModalEl) return;
      rectCompareModalEl.classList.add("hidden");
      if (activeModalGallery === "rect") activeModalGallery = "";
    }

    function openLrMarkerGallery(summary, runForPreview, startIndex, groupKey, modeRaw) {
      if (!evidenceGalleryModalEl || !evidenceGalleryGridEl) return;
      const payload = getLrMarkerPerImagePayload(summary);
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object")
        ? summary.recognition_evidence
        : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const threshold = Math.max(
        LR_MARKER_RELIABLE_MATCH_SCORE,
        Number.isFinite(Number(model.lr_marker_min_match_score)) ? Number(model.lr_marker_min_match_score) : LR_MARKER_RELIABLE_MATCH_SCORE,
      );
      const groupFilter = String(groupKey || "").trim().toUpperCase();
      const mode = String(modeRaw || "").trim().toLowerCase() === "low_score" ? "low_score" : "all";
      const allRows = Array.isArray(payload.items) ? payload.items : [];
      const groupRows = groupFilter ? allRows.filter((item) => inferLrOrientationGroup(item) === groupFilter) : allRows;
      const rows = mode === "low_score"
        ? groupRows.filter((item) => !lrMarkerReliability(item, threshold, payload.targetWidth, payload.targetHeight).reliable)
        : (groupFilter
            ? groupRows.filter((item) => lrMarkerReliability(item, threshold, payload.targetWidth, payload.targetHeight).reliable)
            : groupRows);
      lrMarkerGalleryItems = rows;
      lrMarkerGalleryRunId = String(runForPreview || selectedRunId || "").trim();
      lrMarkerGalleryGroup = groupFilter;
      lrMarkerGalleryMode = mode;
      lrMarkerGallerySourceTotal = groupRows.length;
      lrMarkerGalleryExcludedCount = Math.max(0, groupRows.length - rows.length);
      lrMarkerGalleryReliableThreshold = threshold;
      lrMarkerGalleryTargetWidth = payload.targetWidth || 0;
      lrMarkerGalleryTargetHeight = payload.targetHeight || 0;
      lrMarkerGalleryOrientationGroups = Array.isArray(payload.orientationGroups)
        ? payload.orientationGroups
        : buildLrOrientationGroups(allRows);
      lrMarkerGalleryIndex = Math.max(0, Math.min(rows.length - 1, Math.trunc(Number(startIndex || 0))));
      _closeOtherModals("evidence");
      if (evidenceGalleryCardEl) evidenceGalleryCardEl.classList.add("lr-marker-mode");
      evidenceGalleryGridEl.className = "lr-marker-viewer";
      if (evidenceGalleryTitleEl) evidenceGalleryTitleEl.textContent = mode === "low_score"
        ? "Galleria review LR marker - esclusi dai box"
        : (groupFilter ? `Galleria LR marker - ${groupFilter}` : "Galleria LR marker - verifica grande");
      renderLrMarkerGalleryViewer();
      evidenceGalleryModalEl.classList.remove("hidden");
      activeModalGallery = "evidence";
      bindImageErrorFallback(evidenceGalleryModalEl);
    }

    function renderLrMarkerGalleryViewer() {
      if (!evidenceGalleryGridEl) return;
      const rows = Array.isArray(lrMarkerGalleryItems) ? lrMarkerGalleryItems : [];
      const total = rows.length;
      if (!total) {
        if (evidenceGalleryMetaEl) evidenceGalleryMetaEl.textContent = "nessun frame";
        if (evidenceGalleryStatusEl) evidenceGalleryStatusEl.textContent = "Nessun marker LR disponibile.";
        evidenceGalleryGridEl.innerHTML = "<div class='small'>Nessun frame LR marker disponibile.</div>";
        return;
      }
      lrMarkerGalleryIndex = Math.max(0, Math.min(total - 1, Math.trunc(Number(lrMarkerGalleryIndex || 0))));
      const it = rows[lrMarkerGalleryIndex] || {};
      const score = formatNum(it.match_score, 4);
      const frameIdx = cleanVal(it.image_index);
      const label = cleanVal(it.lr_label || "-");
      const labelIt = cleanVal(it.lr_label_it || "");
      const side = cleanVal(it.detected_marker_side || "-");
      const strategy = cleanVal(it.search_strategy || "-");
      const scope = cleanVal(it.search_scope || "-");
      const sug = cleanVal(it.su_giu_pred || "-");
      const sugConf = formatNum(it.su_giu_conf, 4);
      const marker = `${cleanVal(it.marker_top_abs)}|${cleanVal(it.marker_left_abs)}|${cleanVal(it.marker_bottom_abs)}|${cleanVal(it.marker_right_abs)}|`;
      const quadrantText = lrMarkerQuadrantText(it);
      const quadrantTag = lrMarkerQuadrantTagHtml(it);
      const imageRel = String(it.image_rel || "");
      const templatePath = String(it.template_path || "");
      const templatePolicyEffective = String(it.template_policy_effective || "").trim();
      const templateFallbackReason = String(it.template_fallback_reason || "").trim();
      const envelopeLegend = lrMarkerEnvelopeLegendHtml(lrMarkerGalleryOrientationGroups);
      const reliability = lrMarkerReliability(it, lrMarkerGalleryReliableThreshold, lrMarkerGalleryTargetWidth, lrMarkerGalleryTargetHeight);
      const reviewReason = lrMarkerReliabilityReasonText(it, reliability, lrMarkerGalleryReliableThreshold);
      const reviewKey = lrMarkerReviewKey(it);
      const reviewAnn = lrMarkerReviewAnnotation(reviewKey);
      const reviewControls = lrMarkerGalleryMode === "low_score"
        ? lrMarkerReviewControlsHtml(reviewKey)
        : "";
      if (evidenceGalleryMetaEl) {
        const groupPart = lrMarkerGalleryGroup ? ` | gruppo=${lrMarkerGalleryGroup}` : "";
        const reliabilityPart = lrMarkerGalleryMode === "low_score"
          ? ` | review=${total}/${lrMarkerGallerySourceTotal || total}`
          : (lrMarkerGalleryGroup ? ` | validi=${total}/${lrMarkerGallerySourceTotal || total} | esclusi=${lrMarkerGalleryExcludedCount || 0}` : "");
        evidenceGalleryMetaEl.textContent = `run=${lrMarkerGalleryRunId || "-"}${groupPart}${reliabilityPart} | ${lrMarkerGalleryIndex + 1}/${total} | frame=${frameIdx || "-"} | score=${score}`;
      }
      if (evidenceGalleryStatusEl) {
        evidenceGalleryStatusEl.textContent = lrMarkerGalleryMode === "low_score"
          ? `Modalita review: solo casi esclusi dai box, con score sotto ${formatNum(lrMarkerGalleryReliableThreshold, 2)}, quadrante incoerente o altri motivi di affidabilita. Correggi e salva direttamente da qui.`
          : (lrMarkerGalleryGroup
          ? `Box colorato = rettangolo unico del gruppo; box rosso = marker trovato nel frame. Mostro solo match validi (score >= ${formatNum(lrMarkerGalleryReliableThreshold, 2)}).`
          : "Box rosso = marker trovato nel frame; i box colorati sono i rettangoli unici dei gruppi.");
      }
      const thumbs = rows.slice(0, 240).map((row, idx) => {
        const rel = String((row && row.image_rel) || "");
        const active = idx === lrMarkerGalleryIndex ? " active" : "";
        const thumbScore = formatNum(row && row.match_score, 3);
        const qStatus = cleanVal(row && row.quadrant_status);
        return `
          <button type="button" class="lr-marker-thumb-btn${active}" data-lr-marker-index="${idx}" title="frame ${esc(cleanVal(row && row.image_index))} | score ${esc(thumbScore)} | quadrante ${esc(qStatus || "-")}">
            ${rel ? `<img src="${imgUrlForRun(lrMarkerGalleryRunId, rel)}" loading="lazy" decoding="async">` : ""}
            <span>${esc(thumbScore)}</span>
          </button>
        `;
      }).join("");
      evidenceGalleryGridEl.innerHTML = `
        <div class="lr-marker-viewer-main">
          <div class="lr-marker-viewer-image">
            ${lrMarkerImageOverlayHtml(it, lrMarkerGalleryRunId, lrMarkerGalleryOrientationGroups, {
              reviewKey: lrMarkerGalleryMode === "low_score" ? reviewKey : "",
              manualRect: lrMarkerGalleryMode === "low_score" ? String(reviewAnn.rect || "") : "",
            })}
          </div>
          <aside class="lr-marker-viewer-side">
            ${lrMarkerTemplateHtml(it, lrMarkerGalleryRunId)}
            ${envelopeLegend}
            <div class="lr-marker-viewer-kv"><div class="k">Decisione LR</div><div class="v">${esc(label)}${labelIt ? ` (${esc(labelIt)})` : ""}</div></div>
            <div class="lr-marker-viewer-kv"><div class="k">Score / strategia</div><div class="v">${esc(score)} | ${esc(strategy)} | ${esc(scope)}</div></div>
            ${lrMarkerGalleryMode === "low_score" ? `<div class="lr-marker-viewer-kv"><div class="k">Motivo review</div><div class="v">${esc(reviewReason)}</div></div>` : ""}
            <div class="lr-marker-viewer-kv"><div class="k">Marker trovato</div><div class="v">${esc(side)} | ${esc(marker)}</div></div>
            <div class="lr-marker-viewer-kv"><div class="k">SU/GIU usato</div><div class="v">${esc(sug)} | conf=${esc(sugConf)}</div></div>
            <div class="lr-marker-viewer-kv"><div class="k">Quadrante</div><div class="v">${quadrantTag}<br>${esc(quadrantText)}</div></div>
            <div class="lr-marker-viewer-kv"><div class="k">Template</div><div class="v">${esc(templatePath || "-")}</div></div>
            ${templateFallbackReason ? `<div class="lr-marker-viewer-kv"><div class="k">Fallback template</div><div class="v"><span class="tag tag-warn">attivo</span> ${esc(templatePolicyEffective || "derived_folder")}<br>${esc(templateFallbackReason)}</div></div>` : ""}
            <div class="lr-marker-viewer-kv"><div class="k">Immagine</div><div class="v">${esc(imageRel || "-")}</div></div>
            ${lrMarkerGalleryMode === "low_score" ? `
              <button type="button" class="btn secondary mini" data-lr-marker-pick-mode="1">
                ${lrMarkerManualPickMode ? "Modalita segna marker attiva" : "Segna marker corretto"}
              </button>
              <div class="small">Premi il pulsante e clicca al centro del marker corretto sull'immagine grande. Il rettangolo verde verra salvato nelle annotazioni.</div>
            ` : ""}
            ${reviewControls}
            ${lrMarkerGalleryMode === "low_score" ? `<button type="button" class="btn secondary mini" onclick="saveAnnotations(); return false;">Salva review marker</button>` : ""}
          </aside>
        </div>
        <div class="lr-marker-viewer-thumbs">${thumbs}</div>
      `;
      const selectedThumb = evidenceGalleryGridEl.querySelector(".lr-marker-thumb-btn.active");
      if (selectedThumb) selectedThumb.scrollIntoView({block: "nearest", inline: "center"});
      bindImageErrorFallback(evidenceGalleryModalEl);
      updateLrMarkerManualPickUi();
    }

    function moveLrMarkerGallery(delta) {
      const total = Array.isArray(lrMarkerGalleryItems) ? lrMarkerGalleryItems.length : 0;
      if (!total) return;
      const next = Math.max(0, Math.min(total - 1, lrMarkerGalleryIndex + Math.trunc(Number(delta || 0))));
      if (next === lrMarkerGalleryIndex) return;
      lrMarkerGalleryIndex = next;
      renderLrMarkerGalleryViewer();
    }

    function setLrMarkerManualPickMode(active) {
      lrMarkerManualPickMode = !!active;
      updateLrMarkerManualPickUi();
    }

    function updateLrMarkerManualPickUi() {
      if (!evidenceGalleryModalEl) return;
      evidenceGalleryModalEl.querySelectorAll(".lr-marker-image-wrap[data-lr-marker-review-key]").forEach((wrap) => {
        wrap.classList.toggle("manual-marker-mode", !!lrMarkerManualPickMode);
      });
      evidenceGalleryModalEl.querySelectorAll("[data-lr-marker-pick-mode]").forEach((btn) => {
        btn.textContent = lrMarkerManualPickMode ? "Modalita segna marker attiva" : "Segna marker corretto";
        btn.classList.toggle("danger", !!lrMarkerManualPickMode);
      });
    }

    function lrMarkerGroupFromPoint(x, y, width, height) {
      const upper = y < height / 2;
      const left = x < width / 2;
      if (upper && left) return "NF";
      if (upper && !left) return "LR";
      if (!upper && left) return "UD";
      return "LRUD";
    }

    function setLrMarkerReviewField(key, kind, value) {
      const selector = `[data-lr-marker-review-input][data-lr-review-key="${CSS.escape(String(key || ""))}"][data-lr-review-kind="${CSS.escape(String(kind || ""))}"]`;
      document.querySelectorAll(selector).forEach((el) => {
        el.value = String(value || "");
      });
    }

    function upsertLrManualMarkerBox(wrap, rect) {
      if (!wrap || !rect) return;
      let box = wrap.querySelector(".lr-marker-manual-box");
      if (!box) {
        box = document.createElement("span");
        box.className = "lr-marker-manual-box";
        wrap.appendChild(box);
      }
      box.dataset.markerTop = String(rect.top);
      box.dataset.markerLeft = String(rect.left);
      box.dataset.markerBottom = String(rect.bottom);
      box.dataset.markerRight = String(rect.right);
      const img = wrap.querySelector("img");
      if (img) window.__fitLrMarkerBox && window.__fitLrMarkerBox(img);
    }

    function lrMarkerImagePointFromEvent(ev, wrap) {
      const img = wrap.querySelector("img");
      if (!img) return null;
      const bounds = img.getBoundingClientRect();
      const naturalW = Number(img.naturalWidth || 0);
      const naturalH = Number(img.naturalHeight || 0);
      if (!(bounds.width > 0 && bounds.height > 0 && naturalW > 0 && naturalH > 0)) return null;
      const x = Math.max(0, Math.min(naturalW, ((ev.clientX - bounds.left) / bounds.width) * naturalW));
      const y = Math.max(0, Math.min(naturalH, ((ev.clientY - bounds.top) / bounds.height) * naturalH));
      return {x, y, naturalW, naturalH, img};
    }

    function applyLrMarkerManualRect(key, wrap, rect, naturalW, naturalH) {
      if (!key || !rect) return false;
      const top = Math.max(0, Math.min(Math.trunc(Number(rect.top || 0)), Math.trunc(naturalH || 0) - 1));
      const left = Math.max(0, Math.min(Math.trunc(Number(rect.left || 0)), Math.trunc(naturalW || 0) - 1));
      const bottom = Math.max(top + 1, Math.min(Math.trunc(Number(rect.bottom || 0)), Math.trunc(naturalH || 0)));
      const right = Math.max(left + 1, Math.min(Math.trunc(Number(rect.right || 0)), Math.trunc(naturalW || 0)));
      const rectClean = {top, left, bottom, right};
      const rectText = `${top},${left},${bottom},${right}`;
      const cx = (left + right) * 0.5;
      const cy = (top + bottom) * 0.5;
      setLrMarkerReviewField(key, "rect", rectText);
      setLrMarkerReviewField(key, "flag", "corrected");
      setLrMarkerReviewField(key, "correction", lrMarkerGroupFromPoint(cx, cy, naturalW, naturalH));
      upsertLrManualMarkerBox(wrap, rectClean);
      collectAnnotationsFromUi();
      setLrMarkerManualPickMode(false);
      if (checksStatusEl) checksStatusEl.textContent = "Correzione marker pronta: premi Salva review marker per scriverla su file.";
      return true;
    }

    function defaultLrMarkerRectAroundPoint(point) {
      const img = point && point.img;
      if (!img) return null;
      const detectedLeft = Number(img.dataset.markerLeft || 0);
      const detectedTop = Number(img.dataset.markerTop || 0);
      const detectedRight = Number(img.dataset.markerRight || 0);
      const detectedBottom = Number(img.dataset.markerBottom || 0);
      let markerW = detectedRight > detectedLeft ? detectedRight - detectedLeft : 0;
      let markerH = detectedBottom > detectedTop ? detectedBottom - detectedTop : 0;
      if (!(markerW > 0 && markerH > 0)) {
        markerW = Math.max(12, Math.round(point.naturalW * 0.025));
        markerH = Math.max(12, Math.round(point.naturalH * 0.025));
      }
      const left = Math.max(0, Math.min(point.naturalW - 1, Math.round(point.x - markerW / 2)));
      const top = Math.max(0, Math.min(point.naturalH - 1, Math.round(point.y - markerH / 2)));
      const right = Math.max(left + 1, Math.min(point.naturalW, Math.round(left + markerW)));
      const bottom = Math.max(top + 1, Math.min(point.naturalH, Math.round(top + markerH)));
      return {top, left, bottom, right};
    }

    function handleLrMarkerManualPick(ev, wrap) {
      if (!lrMarkerManualPickMode || lrMarkerGalleryMode !== "low_score") return false;
      if (Date.now() < lrMarkerManualDragSuppressClickUntil) return false;
      if (!wrap || !(wrap instanceof Element)) return false;
      const key = String(wrap.getAttribute("data-lr-marker-review-key") || "");
      if (!key) return false;
      const point = lrMarkerImagePointFromEvent(ev, wrap);
      if (!point) return false;
      return applyLrMarkerManualRect(key, wrap, defaultLrMarkerRectAroundPoint(point), point.naturalW, point.naturalH);
    }

    function beginLrMarkerManualDrag(ev, wrap) {
      if (!lrMarkerManualPickMode || lrMarkerGalleryMode !== "low_score") return false;
      if (!wrap || !(wrap instanceof Element)) return false;
      const key = String(wrap.getAttribute("data-lr-marker-review-key") || "");
      if (!key) return false;
      const point = lrMarkerImagePointFromEvent(ev, wrap);
      if (!point) return false;
      lrMarkerManualDragStart = {key, wrap, point};
      return true;
    }

    function finishLrMarkerManualDrag(ev) {
      const start = lrMarkerManualDragStart;
      lrMarkerManualDragStart = null;
      if (!start || !start.wrap) return false;
      const point = lrMarkerImagePointFromEvent(ev, start.wrap);
      if (!point) return false;
      const dx = Math.abs(point.x - start.point.x);
      const dy = Math.abs(point.y - start.point.y);
      if (dx < 4 && dy < 4) return false;
      const rect = {
        top: Math.min(start.point.y, point.y),
        left: Math.min(start.point.x, point.x),
        bottom: Math.max(start.point.y, point.y),
        right: Math.max(start.point.x, point.x),
      };
      lrMarkerManualDragSuppressClickUntil = Date.now() + 350;
      return applyLrMarkerManualRect(start.key, start.wrap, rect, point.naturalW, point.naturalH);
    }

    function openEvidenceGallery(kindRaw, lrMarkerGroupRaw) {
      const summary = selectedRunData && selectedRunData.summary ? selectedRunData.summary : null;
      const runForPreview = String(selectedRunId || "").trim();
      if (!summary || !runForPreview) return;
      const rawKind = String(kindRaw || "").trim();
      const kind = (rawKind === "raw" || rawKind === "rect" || rawKind === "rect_red" || rawKind === "duplicates" || rawKind === "su_giu" || rawKind === "lr_marker" || rawKind === "lr_marker_low" || rawKind === "lt")
        ? rawKind
        : "raw";
      const maxItems = 240;
      const lrMarkerGroup = String(lrMarkerGroupRaw || "").trim().toUpperCase();
      const openSig = `evidence|${runForPreview}|${kind}|${lrMarkerGroup}`;
      if (!_allowGalleryOpen(openSig, 180)) return;

      if (kind === "rect") {
        openRectGallery(0, summary, runForPreview, "rect");
        return;
      }
      if (kind === "rect_red") {
        openRectGallery(0, summary, runForPreview, "rect_red");
        return;
      }
      if (!evidenceGalleryModalEl || !evidenceGalleryGridEl) return;
      _closeOtherModals("evidence");
      if (evidenceGalleryCardEl) evidenceGalleryCardEl.classList.remove("lr-marker-mode");
      evidenceGalleryGridEl.className = "gallery-modal-grid";

      if (kind === "raw") {
        const inputFolder = rawInputFolderForSummary(summary);
        const rows = rawExclusionRows.length ? rawExclusionRows : rawRowsFromSummary(summary);
        const shown = rows.slice(0, maxItems);
        const excludedCount = rows.filter((row) => row && row.canExclude !== false && row.excluded).length;
        if (evidenceGalleryTitleEl) evidenceGalleryTitleEl.textContent = "Galleria Raw";
        if (evidenceGalleryMetaEl) {
          evidenceGalleryMetaEl.textContent = `run=${runForPreview} | totale=${rows.length} | escluse=${excludedCount}`;
        }
        if (evidenceGalleryStatusEl) {
          evidenceGalleryStatusEl.textContent = shown.length < rows.length
            ? `Mostrate ${shown.length}/${rows.length} immagini per performance.`
            : (rows.length ? "Immagini raw disponibili." : "Nessuna immagine raw disponibile.");
        }
        evidenceGalleryGridEl.innerHTML = shown.map((row, i) => {
          const rel = String((row && row.rel) || "");
          const displayRel = String((row && row.displayRel) || rel || "-");
          const previewPath = String((row && row.previewPath) || joinPreviewPath(inputFolder, rel));
          const checked = row && row.excluded ? " checked" : "";
          const excludeHtml = row && row.canExclude !== false
            ? `<label class="raw-exclude-toggle"><input type="checkbox" data-raw-image-rel="${esc(rel)}"${checked}> Escludi</label>`
            : "";
          return `
            <div class="gallery-modal-item${row && row.excluded ? " excluded" : ""}">
              <div class="head">
                <div class="t">Raw #${i + 1}</div>
                <div class="m">${esc(formatBytes(row && row.sizeBytes))}</div>
              </div>
              <div class="img-single">
                <img src="${imgUrlForRun(runForPreview, previewPath)}" loading="lazy" decoding="async">
              </div>
              ${excludeHtml}
              <div class="meta">${esc(displayRel)}</div>
            </div>
          `;
        }).join("");
        bindRawExclusionInputs(evidenceGalleryGridEl);
      } else if (kind === "duplicates") {
        const dups = Array.isArray(summary.duplicate_examples) ? summary.duplicate_examples : [];
        const shown = dups.slice(0, maxItems);
        if (evidenceGalleryTitleEl) evidenceGalleryTitleEl.textContent = "Galleria Duplicati";
        if (evidenceGalleryMetaEl) evidenceGalleryMetaEl.textContent = `run=${runForPreview} | totale=${dups.length}`;
        if (evidenceGalleryStatusEl) {
          evidenceGalleryStatusEl.textContent = shown.length < dups.length
            ? `Mostrati ${shown.length}/${dups.length} elementi per performance.`
            : (dups.length ? "Confronto visivo completo duplicati." : "Nessun duplicato disponibile.");
        }
        evidenceGalleryGridEl.innerHTML = shown.map((d, i) => {
          const kept = String((d && d.kept_rel) || "");
          const removed = String((d && d.removed_rel) || "");
          const sha = String((d && d.sha1_prefix) || "");
          const size = String((d && d.size_bytes) || "");
          return `
            <div class="gallery-modal-item">
              <div class="head">
                <div class="t">Duplicato #${i + 1}</div>
                <div class="m">sha1 ${esc(sha || "-")}</div>
              </div>
              <div class="img-pair">
                <img src="${imgUrlForRun(runForPreview, kept)}" loading="lazy" decoding="async">
                <img src="${imgUrlForRun(runForPreview, removed)}" loading="lazy" decoding="async">
              </div>
              <div class="meta">size=${esc(size || "-")} bytes</div>
              <div class="meta"><b>kept</b>: ${esc(kept || "-")}</div>
              <div class="meta"><b>removed</b>: ${esc(removed || "-")}</div>
              <div class="dup-actions">
                <button
                  type="button"
                  class="btn secondary mini"
                  data-open-dup-compare="1"
                  data-run-id="${esc(runForPreview)}"
                  data-kept="${esc(kept)}"
                  data-removed="${esc(removed)}"
                  data-sha="${esc(sha)}"
                  data-size="${esc(size)}"
                  onclick="return window.__openDupCompareFromBtn ? window.__openDupCompareFromBtn(this) : false;"
                >Apri confronto grande</button>
              </div>
            </div>
          `;
        }).join("");
      } else if (kind === "su_giu") {
        const payload = getSuGiuPerImagePayload(summary);
        const rows = Array.isArray(payload.items) ? payload.items : [];
        const groups = (payload.groups && typeof payload.groups === "object") ? payload.groups : {};
        const suRows = Array.isArray(groups.su) ? groups.su : rows.filter((it) => normalizeSuGiuPredictionLabel((it && it.pred_label) || (it && it.su_giu_pred)) === "su");
        const giuRows = Array.isArray(groups.giu) ? groups.giu : rows.filter((it) => normalizeSuGiuPredictionLabel((it && it.pred_label) || (it && it.su_giu_pred)) === "giu");
        const otherRows = Array.isArray(groups.other) ? groups.other : rows.filter((it) => normalizeSuGiuPredictionLabel((it && it.pred_label) || (it && it.su_giu_pred)) === "other");
        const primaryLimit = Math.max(1, Math.ceil(maxItems / 2));
        const suShown = suRows.slice(0, primaryLimit);
        const giuShown = giuRows.slice(0, primaryLimit);
        const otherShown = otherRows.slice(0, Math.max(0, maxItems - suShown.length - giuShown.length));
        const counts = payload.labelCounts || {};
        if (evidenceGalleryTitleEl) evidenceGalleryTitleEl.textContent = "Galleria Orientamento SU/GIU per frame";
        if (evidenceGalleryMetaEl) {
          evidenceGalleryMetaEl.textContent = `run=${runForPreview} | totale=${rows.length} | su=${counts.su || 0} | giu=${counts.giu || 0}`;
        }
        if (evidenceGalleryStatusEl) {
          const shownCount = suShown.length + giuShown.length + otherShown.length;
          evidenceGalleryStatusEl.textContent = shownCount < rows.length
            ? `Mostrati ${shownCount}/${rows.length} frame per performance. Sezioni separate: SU e GIU, confidence decrescente.`
            : (rows.length ? "Sezioni separate: SU e GIU, confidence decrescente." : "Nessun frame etichettato.");
        }
        evidenceGalleryGridEl.className = "gallery-modal-grid sugiu-split-mode";
        const renderSuGiuCard = (it, i, sectionLabel) => {
          const rel = String((it && it.image_rel) || "");
          const label = String((it && it.pred_label) || sectionLabel || "-");
          const conf = formatNum(suGiuPredictionConfidence(it), 4);
          const pSu = formatNum(it && it.prob_su, 4);
          const pGiu = formatNum(it && it.prob_giu, 4);
          const idx = cleanVal(it && it.image_index);
          const source = cleanVal(it && it.crop_source);
          const crop = `${cleanVal(it && it.crop_top)}|${cleanVal(it && it.crop_left)}|${cleanVal(it && it.crop_bottom)}|${cleanVal(it && it.crop_right)}|`;
          return `
            <div class="gallery-modal-item">
              <div class="head">
                <div class="t">${esc(String(sectionLabel || label).toUpperCase())} #${i + 1} · frame ${esc(String(idx || "-"))}</div>
                <div class="m">label=${esc(label)}</div>
              </div>
              <div class="img-pair" style="grid-template-columns:1fr;">
                <img src="${imgUrlForRun(runForPreview, rel)}" loading="lazy" decoding="async">
              </div>
              <div class="meta">conf=${esc(conf)} | p_su=${esc(pSu)} | p_giu=${esc(pGiu)}</div>
              <div class="meta">crop=${esc(crop)} | source=${esc(String(source))}</div>
              <div class="meta"><b>img</b>: ${esc(rel || "-")}</div>
            </div>
          `;
        };
        const renderSuGiuSection = (key, title, sectionRows, shownRows) => {
          const total = Array.isArray(sectionRows) ? sectionRows.length : 0;
          const shownLocal = Array.isArray(shownRows) ? shownRows : [];
          const omitted = Math.max(0, total - shownLocal.length);
          const cards = shownLocal.length
            ? shownLocal.map((it, i) => renderSuGiuCard(it, i, key)).join("")
            : `<div class="small">Nessun frame ${esc(title)}.</div>`;
          return `
            <section class="sugiu-gallery-section ${esc(key)}">
              <div class="sugiu-gallery-section-head">
                <div class="sugiu-gallery-section-title"><span class="sugiu-gallery-badge">${esc(key)}</span>${esc(title)}</div>
                <div class="small">${shownLocal.length}/${total} frame</div>
              </div>
              <div class="small" style="margin-bottom:8px;">Ordinati per confidence decrescente.</div>
              <div class="sugiu-gallery-section-grid">${cards}</div>
              ${omitted ? `<div class="small" style="margin-top:8px;">Non mostrati ${omitted} frame per performance.</div>` : ""}
            </section>
          `;
        };
        const otherHtml = otherRows.length
          ? `<div class="sugiu-gallery-sections" style="grid-template-columns:1fr;margin-top:12px;">${renderSuGiuSection("other", "Other", otherRows, otherShown)}</div>`
          : "";
        evidenceGalleryGridEl.innerHTML = `
          <div class="sugiu-gallery-sections">
            ${renderSuGiuSection("su", "SU", suRows, suShown)}
            ${renderSuGiuSection("giu", "GIU", giuRows, giuShown)}
          </div>
          ${otherHtml}
        `;
      } else if (kind === "lr_marker") {
        openLrMarkerGallery(summary, runForPreview, 0, lrMarkerGroup);
        return;
      } else if (kind === "lr_marker_low") {
        openLrMarkerGallery(summary, runForPreview, 0, "", "low_score");
        return;
      } else if (kind === "lt") {
        const payload = getLtPerImagePayload(summary);
        const rows = Array.isArray(payload.items) ? payload.items : [];
        const shown = rows.slice(0, maxItems);
        const counts = payload.labelCounts || {};
        if (evidenceGalleryTitleEl) evidenceGalleryTitleEl.textContent = "Galleria Classificazione L/T per frame";
        if (evidenceGalleryMetaEl) {
          evidenceGalleryMetaEl.textContent = `run=${runForPreview} | totale=${rows.length} | L=${counts.l || 0} | T=${counts.t || 0}`;
        }
        if (evidenceGalleryStatusEl) {
          evidenceGalleryStatusEl.textContent = shown.length < rows.length
            ? `Mostrati ${shown.length}/${rows.length} frame per performance.`
            : (rows.length ? "Classificazione frame-by-frame disponibile." : "Nessun frame etichettato.");
        }
        evidenceGalleryGridEl.innerHTML = shown.map((it, i) => {
          const rel = String((it && it.image_rel) || "");
          const label = String((it && it.pred_label) || "-");
          const conf = formatNum(it && it.confidence, 4);
          const pL = formatNum(it && it.prob_l, 4);
          const pT = formatNum(it && it.prob_t, 4);
          const idx = cleanVal(it && it.image_index);
          const source = cleanVal(it && it.crop_source);
          const crop = `${cleanVal(it && it.crop_top)}|${cleanVal(it && it.crop_left)}|${cleanVal(it && it.crop_bottom)}|${cleanVal(it && it.crop_right)}|`;
          return `
            <div class="gallery-modal-item">
              <div class="head">
                <div class="t">Frame #${esc(String(idx || (i + 1)))}</div>
                <div class="m">label=${esc(label)}</div>
              </div>
              <div class="img-pair" style="grid-template-columns:1fr;">
                <img src="${imgUrlForRun(runForPreview, rel)}" loading="lazy" decoding="async">
              </div>
              <div class="meta">conf=${esc(conf)} | p_l=${esc(pL)} | p_t=${esc(pT)}</div>
              <div class="meta">crop=${esc(crop)} | source=${esc(String(source))}</div>
              <div class="meta"><b>img</b>: ${esc(rel || "-")}</div>
            </div>
          `;
        }).join("");
      }

      evidenceGalleryModalEl.classList.remove("hidden");
      bindImageErrorFallback(evidenceGalleryModalEl);
    }

    function closeEvidenceGallery() {
      if (!evidenceGalleryModalEl) return;
      evidenceGalleryModalEl.classList.add("hidden");
      if (evidenceGalleryCardEl) evidenceGalleryCardEl.classList.remove("lr-marker-mode");
      if (evidenceGalleryGridEl) evidenceGalleryGridEl.className = "gallery-modal-grid";
      lrMarkerManualPickMode = false;
      if (activeModalGallery === "evidence") activeModalGallery = "";
    }

    window.__openEvidenceGalleryFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const kind = String(btn.getAttribute("data-open-evidence-gallery") || "duplicates");
        const lrGroup = String(btn.getAttribute("data-lr-marker-group") || "").trim();
        openEvidenceGallery(kind, lrGroup);
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__openDupCompareFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const kept = String(btn.getAttribute("data-kept") || "");
        const removed = String(btn.getAttribute("data-removed") || "");
        const sha = String(btn.getAttribute("data-sha") || "");
        const size = String(btn.getAttribute("data-size") || "");
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "");
        openDupCompareModal(kept, removed, sha, size, runId);
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__setSegmentListIndex = function(el) {
      try {
        if (!el || !(el instanceof Element)) return false;
        const listId = String(el.getAttribute("data-segment-list-id") || "").trim();
        if (!listId) return false;
        const idx = Number(el.getAttribute("data-segment-index") || "0");
        if (!Number.isFinite(idx)) return false;
        const listEl = document.getElementById(listId);
        if (!listEl) return false;
        setSegmentListActiveIndex(listEl, idx, {behavior: "auto"});
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__scrollSegmentListFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const listId = String(btn.getAttribute("data-segment-list-id") || "").trim();
        if (!listId) return false;
        const listEl = document.getElementById(listId);
        if (!listEl) return false;
        const dir = String(btn.getAttribute("data-segment-scroll") || "down").trim().toLowerCase();
        const delta = dir === "up" ? -1 : 1;
        const current = Number(listEl.getAttribute("data-segment-current-index") || "0");
        const base = Number.isFinite(current) ? Math.trunc(current) : 0;
        const nextIdx = setSegmentListActiveIndex(listEl, base + delta, {behavior: "smooth"});
        if (nextIdx < 0) return false;
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__openRectGalleryFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "");
        const startIndex = Number(btn.getAttribute("data-start-index") || "0");
        const mode = String(btn.getAttribute("data-open-rect-gallery-mode") || "rect").trim().toLowerCase();
        openRectGallery(
          Number.isFinite(startIndex) ? Math.trunc(startIndex) : 0,
          null,
          runId,
          mode === "rect_red" ? "rect_red" : "rect",
        );
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__openRectGalleryAt = function(indexValue) {
      try {
        const idx = Number(indexValue);
        if (!Number.isFinite(idx)) return false;
        rectGalleryIndex = Math.max(0, Math.min(rectGalleryItems.length - 1, Math.trunc(idx)));
        renderRectGalleryCurrent();
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__runRectVendorCompareFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "").trim();
        const force = String(btn.getAttribute("data-force") || "0") === "1";
        runRectVendorCompare(runId, force);
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__runRectRedRecomputeFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "").trim();
        const marginInputId = String(btn.getAttribute("data-margin-input-id") || "").trim();
        const brightInputId = String(btn.getAttribute("data-bright-input-id") || "").trim();
        const marginEl = marginInputId ? document.getElementById(marginInputId) : null;
        const brightEl = brightInputId ? document.getElementById(brightInputId) : null;
        const marginVal = marginEl ? Number(marginEl.value) : NaN;
        const brightVal = brightEl ? Number(brightEl.value) : NaN;
        runRectRedRecompute(runId, marginVal, brightVal);
      } catch (err) {
        // no-op
      }
      return false;
    };

    window.__openRectCompareModelGalleryFromBtn = function(btn) {
      try {
        if (!btn || !(btn instanceof Element)) return false;
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "").trim();
        const modelKey = String(btn.getAttribute("data-model-key") || "").trim();
        openRectCompareModelGallery(modelKey, runId);
      } catch (err) {
        // no-op
      }
      return false;
    };

    function findActionTarget(ev, selector) {
      if (!ev || !selector) return null;
      const direct = ev.target;
      if (direct && direct instanceof Element) {
        const near = direct.closest(selector);
        if (near) return near;
      }
      if (typeof ev.composedPath === "function") {
        const path = ev.composedPath();
        for (const node of path) {
          if (!node || !(node instanceof Element)) continue;
          if (typeof node.matches === "function" && node.matches(selector)) return node;
        }
      }
      return null;
    }

    async function apiGet(path) {
      const r = await fetch(path);
      const j = await r.json();
      if (!r.ok) throw new Error(j && j.error ? j.error : ("HTTP " + r.status));
      return j;
    }

    async function apiPost(path, payload) {
      const r = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {}),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j && j.error ? j.error : ("HTTP " + r.status));
      return j;
    }

    async function apiPostWithTimeout(path, payload, timeoutMs) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), Math.max(300, Number(timeoutMs || 0)));
      try {
        const r = await fetch(path, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload || {}),
          signal: controller.signal,
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j && j.error ? j.error : ("HTTP " + r.status));
        return j;
      } finally {
        clearTimeout(timer);
      }
    }

    async function deleteRun(runId) {
      if (!runId) return;
      if (!confirm(`Eliminare la run ${runId}?`)) return;
      try {
        await apiPost(`/api/runs/${encodeURIComponent(runId)}/delete`, {delete_files: true});
        if (selectedRunId === runId) {
          selectedRunId = null;
          selectedRunData = null;
          selectedRunStatus = null;
          if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
          renderStatus(null);
          renderChecksAndAnalysis();
          setViewMode("setup");
        }
        startStatusEl.textContent = "Run eliminata: " + runId;
        await loadRuns();
      } catch (err) {
        alert("Errore eliminazione run: " + err.message);
      }
    }

    function buildRunPayload(inputFolderOverride) {
      const inputOverride = String(inputFolderOverride || "").trim();
      const inputCurrent = String((document.getElementById("inputFolder") || {}).value || "").trim();
      const payload = {
        input_folder: inputOverride || inputCurrent,
        batch_size: Number(document.getElementById("batchSize").value || 32),
        sample_per_folder: Number(document.getElementById("samplePerFolder").value || 80),
        rotation_max_samples: Number(document.getElementById("rotationMaxSamples").value || 24),
        low_confidence_policy: document.getElementById("lowPolicy").value,
        vendor_min_confidence: Number(document.getElementById("vendorMinConf").value || 0.5),
        probe_min_confidence: Number(document.getElementById("probeMinConf").value || 0.5),
        lr_marker_template_policy: document.getElementById("lrMarkerTemplatePolicy").value,
        no_generated_images: !!(noGeneratedImagesEl && noGeneratedImagesEl.checked),
        no_split_symlinks: !!(noSplitSymlinksEl && noSplitSymlinksEl.checked),
        remember_exclusions: true,
      };
      if (Array.isArray(inputImageRows) && inputImageRows.length > 0) {
        payload.excluded_images_rel = selectedExcludedImages();
      }
      return payload;
    }

    async function startRunWithPayload(payload, contextLabel) {
      const ctx = String(contextLabel || "run");
      const inputFolder = String((payload && payload.input_folder) || "").trim();
      if (!inputFolder) {
        startStatusEl.textContent = "Inserisci una cartella input valida.";
        alert("Inserisci prima una cartella input valida.");
        return;
      }
      setActiveFolderTitle(inputFolder);
      setViewMode("workspace");
      setActivePane("monitor");
      renderDecisionTreeLive(
        {
          status: "queued",
          stage: "queued",
          input_folder: inputFolder,
          completed_stages: [],
          logs_tail: [],
        },
        null,
      );
      document.getElementById("startRunBtn").disabled = true;
      startStatusEl.textContent = `Avvio ${ctx}...`;
      try {
        const out = await apiPost("/api/runs/start", payload);
        startStatusEl.textContent = `${ctx} avviata: ${out.run_id}`;
        setViewMode("workspace");
        setActivePane("monitor");
        await loadRuns();
        await selectRun(out.run_id);
      } catch (err) {
        startStatusEl.textContent = `Errore avvio ${ctx}: ` + err.message;
        alert(`Errore avvio ${ctx}: ` + err.message);
      } finally {
        document.getElementById("startRunBtn").disabled = false;
      }
    }

    function renderBulkStatus(st) {
      if (!bulkStatusEl) return;
      const status = st && typeof st === "object" ? st : {};
      const total = Number(status.total || 0);
      const created = Number(status.created || 0);
      const completed = Number(status.completed || 0);
      const failed = Number(status.failed || 0);
      const skipped = Number(status.skipped || 0);
      const current = String(status.current || "").split(/[\\/]/).filter(Boolean).pop() || "";
      bulkStatusEl.textContent = `Batch ${cleanVal(status.status || "idle")} | create=${created}/${total} | completate=${completed} | fallite=${failed} | senza immagini=${skipped}${current ? " | corrente=" + current : ""}${status.last_error ? " | errore=" + status.last_error : ""}`;
      if (startBulkBtnEl) startBulkBtnEl.disabled = !!status.running;
    }

    async function refreshBulkStatus() {
      try {
        const st = await apiGet("/api/batch/acquisition_elaboration/status");
        renderBulkStatus(st);
      } catch (err) {
        if (bulkStatusEl) bulkStatusEl.textContent = "Errore stato batch: " + err.message;
      }
    }

    async function startBulkAcquisitionRun() {
      const root = String((bulkAcquisitionRootEl && bulkAcquisitionRootEl.value) || "").trim();
      if (!confirm("Eliminare tutte le run passate e generare una run leggera per ogni cartella SSD?")) return;
      const payload = buildRunPayload("");
      payload.root = root || "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION";
      payload.input_folder = payload.root;
      payload.lr_marker_template_policy = "historical_best";
      payload.no_generated_images = true;
      payload.no_split_symlinks = true;
      if (startBulkBtnEl) startBulkBtnEl.disabled = true;
      try {
        const out = await apiPost("/api/batch/acquisition_elaboration/start", payload);
        renderBulkStatus(out.status || out);
        await loadRuns();
      } catch (err) {
        if (bulkStatusEl) bulkStatusEl.textContent = "Errore avvio batch: " + err.message;
        if (startBulkBtnEl) startBulkBtnEl.disabled = false;
      }
    }

    async function rerunFromHistory(runObj) {
      const run = (runObj && typeof runObj === "object") ? runObj : {};
      const runId = String(run.run_id || "").trim();
      const inputFolder = String(run.input_folder || "").trim();
      if (!inputFolder) {
        alert("Impossibile fare rerun: input_folder non disponibile in questa run.");
        return;
      }
      if (!confirm(`Rieseguire la pipeline sulla stessa cartella?\n\nrun=${runId || "-"}\ninput=${inputFolder}`)) {
        return;
      }
      const payload = buildRunPayload(inputFolder);
      await startRunWithPayload(payload, "rerun");
    }

    function selectedRunRecord() {
      const rid = String(selectedRunId || "").trim();
      if (!rid) return null;
      return runs.find((run) => String(run.run_id || "") === rid) || null;
    }

    const RERUN_TARGET_PRESETS = {
      orientation: ["su_giu", "lr_marker"],
      rect: ["rect_red", "rect_per_image", "rect_compare"],
      all_evidence: ["rect_red", "rect_per_image", "rect_compare", "su_giu", "lr_marker", "lt", "line13"],
    };

    function activeRerunRoot() {
      const workspaceRoot = document.getElementById("workspaceRerunPanel");
      if (document.body.classList.contains("workspace-mode") && workspaceRoot) return workspaceRoot;
      return document.querySelector(".rerun-box") || document;
    }

    function rerunTargetCheckboxes() {
      const root = activeRerunRoot();
      return Array.from(root.querySelectorAll("[data-rerun-target]"));
    }

    function selectedRerunTargets() {
      return rerunTargetCheckboxes()
        .filter((el) => !!el.checked)
        .map((el) => String(el.getAttribute("data-rerun-target") || "").trim())
        .filter(Boolean);
    }

    function setRerunTargets(targets) {
      const wanted = new Set((Array.isArray(targets) ? targets : []).map((x) => String(x || "").trim()).filter(Boolean));
      rerunTargetCheckboxes().forEach((el) => {
        el.checked = wanted.has(String(el.getAttribute("data-rerun-target") || "").trim());
      });
    }

    function applyRerunPreset(name) {
      const key = String(name || "").trim();
      if (key === "selected") return;
      setRerunTargets(RERUN_TARGET_PRESETS[key] || []);
    }

    function setRerunStatusText(text) {
      const msg = String(text || "");
      if (rerunStatusEl) rerunStatusEl.textContent = msg;
      if (workspaceRerunStatusEl) workspaceRerunStatusEl.textContent = msg;
    }

    async function runSelectedRerunAction() {
      const targets = selectedRerunTargets();
      if (!targets.length) {
        setRerunStatusText("Spunta almeno un punto da ricalcolare.");
        alert("Spunta almeno un punto da ricalcolare.");
        return;
      }
      const run = selectedRunRecord();
      const selectedInputFolder = String((run && run.input_folder) || "").trim();
      if (targets.includes("full_pipeline")) {
        const inputFolder = selectedInputFolder || String((document.getElementById("inputFolder") || {}).value || "").trim();
        if (!inputFolder) {
          setRerunStatusText("Scegli una run dallo storico o inserisci una cartella input.");
          alert("Scegli una run dallo storico o inserisci una cartella input.");
          return;
        }
        if (!confirm(`Rifare una run completa con i parametri correnti?\n\ninput=${inputFolder}`)) return;
        setRerunStatusText("Avvio rerun completa...");
        await startRunWithPayload(buildRunPayload(inputFolder), "rerun completa");
        setRerunStatusText("Rerun completa avviata.");
        return;
      }

      if (!selectedRunId) {
        setRerunStatusText("Per ricalcolare punti specifici devi selezionare una run completata.");
        alert("Seleziona prima una run dallo storico.");
        return;
      }
      if (!confirm(`Ricalcolare sulla run selezionata questi punti?\n\n${targets.join(", ")}`)) return;
      setRerunStatusText(`Ricalcolo: ${targets.join(", ")}...`);
      const payload = {
        targets,
        force: !!(rerunForceEvidenceEl && rerunForceEvidenceEl.checked),
        margin_pct: Number((rerunRectMarginPctEl && rerunRectMarginPctEl.value) || 5),
        bright_thr: Number((rerunRectBrightThrEl && rerunRectBrightThrEl.value) || 70),
        batch_size: Number(document.getElementById("batchSize").value || 32),
        no_generated_images: !!(noGeneratedImagesEl && noGeneratedImagesEl.checked),
        no_split_symlinks: !!(noSplitSymlinksEl && noSplitSymlinksEl.checked),
      };
      const out = await apiPost(`/api/runs/${encodeURIComponent(selectedRunId)}/recompute`, payload);
      if (out.rect_vendor_compare && typeof out.rect_vendor_compare === "object" && Object.keys(out.rect_vendor_compare).length) {
        setRectVendorCompareState(selectedRunId, {
          status: "done",
          payload: out.rect_vendor_compare,
          error: "",
          updated_at: new Date().toISOString(),
        });
      }
      if (out.summary && selectedRunData && selectedRunId === out.run_id) {
        selectedRunData.summary = out.summary;
        renderPredictionSummary(out.summary);
        renderFssFromSummary(out.summary);
        renderChecksAndAnalysis();
        renderActiveEvidenceSection(out.summary);
        renderDecisionTreeLive(selectedRunStatus, out.summary);
      } else if (selectedRunId) {
        await refreshSelectedRun(true);
      }
      setRerunStatusText(`Ricalcolo completato: ${(out.completed_targets || targets).join(", ")}`);
    }

    function renderRunsList() {
      runsListEl.innerHTML = "";
      if (!runs.length) {
        runsListEl.innerHTML = "<div class='run-item'><div class='s'>Nessuna run disponibile.</div></div>";
        setHistoryFinderPath(resolveHistoryFinderPath());
        return;
      }
      runs.forEach((run) => {
        const div = document.createElement("div");
        div.className = "run-item openable" + (run.run_id === selectedRunId ? " active" : "");
        div.innerHTML = `
          <div class="t">${run.run_id}</div>
          <div class="s">${run.input_folder || "-"}</div>
          <div class="s"><span class="status-pill ${statusClass(run.status)}">${run.status || "-"}</span> ${fmtTs(run.created_at)}</div>
          <div class="s">tempo: ${fmtRunElapsed(run)}</div>
          <div class="run-actions">
            <button type="button" class="btn secondary mini" data-act="rerun">Rerun</button>
            <button type="button" class="btn danger mini" data-act="delete">Elimina</button>
          </div>
        `;
        div.onclick = () => selectRun(run.run_id);
        const rerunBtn = div.querySelector('[data-act="rerun"]');
        if (rerunBtn) {
          rerunBtn.onclick = async (ev) => {
            ev.stopPropagation();
            await rerunFromHistory(run);
          };
        }
        const delBtn = div.querySelector('[data-act="delete"]');
        if (delBtn) {
          delBtn.onclick = (ev) => {
            ev.stopPropagation();
            deleteRun(run.run_id);
          };
        }
        runsListEl.appendChild(div);
      });
      setHistoryFinderPath(resolveHistoryFinderPath());
      renderStudyRunPicker();
    }

    function studyRunsSorted() {
      return runs
        .filter((run) => String(run.status || "") === "completed")
        .slice()
        .sort((a, b) => String(a.input_folder || "").localeCompare(String(b.input_folder || ""), undefined, {numeric: true, sensitivity: "base"}));
    }

    function renderStudyRunPicker() {
      if (!studyRunPickerEl) return;
      const studyRuns = studyRunsSorted();
      if (!studyRuns.length) {
        studyRunPickerEl.innerHTML = `<option value="">Nessuna run completata</option>`;
        studyRunId = "";
        if (studyContentEl) studyContentEl.innerHTML = "";
        if (studyStatusEl) studyStatusEl.textContent = "Nessuna run completata disponibile per gli studi trasversali.";
        return;
      }
      if (!studyRunId || !studyRuns.some((run) => String(run.run_id || "") === studyRunId)) {
        studyRunId = String(studyRuns[0].run_id || "");
      }
      studyRunPickerEl.innerHTML = studyRuns.map((run) => {
        const rid = String(run.run_id || "");
        const folder = String(run.input_folder || "").split(/[\\/]/).filter(Boolean).pop() || rid;
        return `<option value="${esc(rid)}"${rid === studyRunId ? " selected" : ""}>${esc(folder)}</option>`;
      }).join("");
    }

    async function loadStudyRunData(runId) {
      const rid = String(runId || "").trim();
      if (!rid) return null;
      if (selectedRunId === rid && selectedRunData && selectedRunData.summary) return selectedRunData;
      if (studyDataCache[rid]) return studyDataCache[rid];
      const data = await apiGet(`/api/runs/${encodeURIComponent(rid)}/data`);
      studyDataCache[rid] = data;
      return data;
    }

    function studyKpiHtml(items) {
      return `<div class="study-kpis">${items.map((it) => `
        <div class="study-kpi"><div class="k">${esc(it.k)}</div><div class="v">${esc(cleanVal(it.v))}</div></div>
      `).join("")}</div>`;
    }

    function studyImageCardHtml(runId, rel, title, meta) {
      const relTxt = String(rel || "").trim();
      return `
        <div class="study-frame">
          ${relTxt ? `<img src="${imgUrlForRun(runId, relTxt)}" loading="lazy" decoding="async">` : `<div class="small">Immagine non disponibile.</div>`}
          <div class="meta"><b>${esc(title || shortImageName(relTxt))}</b></div>
          <div class="meta">${esc(meta || relTxt || "-")}</div>
        </div>
      `;
    }

    function renderStudySummary(runId, runObj, summary, studyType) {
      if (!studyContentEl) return;
      const row = (summary && summary.pipeline_row && typeof summary.pipeline_row === "object") ? summary.pipeline_row : {};
      const rec = (summary && summary.recognition_evidence && typeof summary.recognition_evidence === "object") ? summary.recognition_evidence : {};
      const model = (rec.model_evidence && typeof rec.model_evidence === "object") ? rec.model_evidence : {};
      const folder = String((runObj && runObj.input_folder) || summary.input_original_folder || "");
      const folderName = folder.split(/[\\/]/).filter(Boolean).pop() || String(runId || "-");
      const header = `
        <div class="evidence-card">
          <div class="evidence-title">${esc(folderName)} <span class="tag tag-data">${esc(studyType)}</span></div>
          <div class="small"><span class="path">${esc(folder)}</span></div>
          <div class="small">Modalita spazio: immagini generate=${summary.generated_images_enabled ? "SI" : "NO"} | split symlink=${summary.split_symlinks_enabled ? "SI" : "NO"}</div>
        </div>
      `;

      if (studyType === "duplicates") {
        const dups = Array.isArray(summary.duplicate_examples) ? summary.duplicate_examples : [];
        const cards = dups.slice(0, 12).map((it) => `
          <div class="study-frame">
            <div class="meta"><b>sha1 ${esc(cleanVal(it.sha1_prefix))}</b> | ${esc(cleanVal(it.size_bytes))} bytes</div>
            ${it.kept_rel ? `<img src="${imgUrlForRun(runId, it.kept_rel)}" loading="lazy" decoding="async">` : ""}
            <div class="meta">tenuta: ${esc(shortImageName(it.kept_rel || ""))}</div>
            ${it.removed_rel ? `<img src="${imgUrlForRun(runId, it.removed_rel)}" loading="lazy" decoding="async">` : ""}
            <div class="meta">rimossa: ${esc(shortImageName(it.removed_rel || ""))}</div>
          </div>
        `).join("");
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml([
          {k: "raw", v: summary.raw_images_count || 0},
          {k: "duplicati rimossi", v: summary.duplicates_removed_count_recomputed || 0},
          {k: "esempi", v: dups.length},
          {k: "riferimento", v: summary.input_reference_mode || "-"},
        ])}<div class="study-gallery">${cards || "<div class='small'>Nessun duplicato trovato.</div>"}</div></div>`;
        return;
      }

      if (studyType === "rotation") {
        const rot = (summary.rotation_evidence && typeof summary.rotation_evidence === "object") ? summary.rotation_evidence : {};
        const sampleRel = rot.sample_original_rel || rot.sample_before_rel || (Array.isArray(summary.raw_images_rel) ? summary.raw_images_rel[0] : "");
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml([
          {k: "rotazione", v: `${cleanVal(rot.rotation_deg_clockwise)} deg`},
          {k: "source", v: rot.rotation_source || "-"},
          {k: "vote ratio", v: formatNum(rot.rotation_vote_ratio, 4)},
          {k: "sample", v: rot.rotation_samples_checked || "-"},
        ])}<div class="study-gallery">${studyImageCardHtml(runId, sampleRel, "Frame riferimento originale", rot.rotation_decision_reason || rot.error || sampleRel)}</div></div>`;
        return;
      }

      if (studyType === "vendor" || studyType === "probe") {
        const samples = Array.isArray(rec.vendor_probe_samples_rel) ? rec.vendor_probe_samples_rel : [];
        const cards = samples.map((rel, idx) => studyImageCardHtml(runId, rel, `Sample ${idx + 1}`, rel)).join("");
        const isVendor = studyType === "vendor";
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml(isVendor ? [
          {k: "vendor", v: row.vendor_predicted || "-"},
          {k: "confidenza", v: formatNum(row.vendor_confidence, 4)},
          {k: "source", v: row.vendor_source || "-"},
          {k: "checkpoint", v: model.vendor_checkpoint || "-"},
        ] : [
          {k: "probe id", v: row.line_03_id_probe || "-"},
          {k: "probe nome", v: row.line_03_probe_name || "-"},
          {k: "confidenza", v: formatNum(row.line_03_probe_confidence, 4)},
          {k: "checkpoint", v: model.probe_checkpoint || "-"},
        ])}<div class="small">${esc(isVendor ? (row.vendor_decision_reason || "") : (row.probe_decision_reason || ""))}</div><div class="study-gallery">${cards || "<div class='small'>Sample non disponibili.</div>"}</div></div>`;
        return;
      }

      if (studyType === "rect") {
        const rect = (rec.rect_per_image && typeof rec.rect_per_image === "object") ? rec.rect_per_image : {};
        const rectRed = (rec.rect_red && typeof rec.rect_red === "object") ? rec.rect_red : {};
        const items = Array.isArray(rect.items) ? rect.items : [];
        const cards = items.slice(0, 10).map((it) => studyImageCardHtml(runId, it.image_rel, `#${cleanVal(it.rank)} IoU ${formatNum(it.iou, 3)}`, `disc=${formatNum(it.discrepancy, 4)} | singolo=${formatRectTlbr(it.single_rect)} | globale=${formatRectTlbr(it.global_rect_scaled)}`)).join("");
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml([
          {k: "rect globale", v: row.line_11_rect_echo || "-"},
          {k: "rect per-frame", v: rect.images_total || 0},
          {k: "rect rosso", v: rectRed.winner_group || "-"},
          {k: "checkpoint", v: model.rect_checkpoint_used || "-"},
        ])}<div class="small">Le immagini qui sono originali; i box restano dati numerici, non PNG generati.</div><div class="study-gallery">${cards || "<div class='small'>Dati rettangolo non disponibili.</div>"}</div></div>`;
        return;
      }

      if (studyType === "orientation") {
        const sg = (rec.su_giu_per_image && typeof rec.su_giu_per_image === "object") ? rec.su_giu_per_image : {};
        const lr = (rec.lr_marker_per_image && typeof rec.lr_marker_per_image === "object") ? rec.lr_marker_per_image : {};
        const sgCounts = (sg.label_counts && typeof sg.label_counts === "object") ? sg.label_counts : {};
        const lrCounts = (lr.label_counts && typeof lr.label_counts === "object") ? lr.label_counts : {};
        const lrItems = Array.isArray(lr.items) ? lr.items : [];
        const cards = lrItems.slice(0, 8).map((it) => `<div class="study-frame">${lrMarkerVisualHtml(it, runId)}<div class="meta">frame=${esc(cleanVal(it.image_index))} | score=${esc(formatNum(it.match_score, 4))} | ${esc(cleanVal(it.search_strategy))}</div></div>`).join("");
        const fallbackReason = String(lr.template_fallback_reason || (lr.best && lr.best.template_fallback_reason) || "").trim();
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml([
          {k: "SU", v: sgCounts.su || 0},
          {k: "GIU", v: sgCounts.giu || 0},
          {k: "LR non flipped", v: lrCounts.not_lr_flipped || 0},
          {k: "LR flipped", v: lrCounts.lr_flipped || 0},
        ])}${fallbackReason ? `<div class="small" style="color:#b45309;"><span class="tag tag-warn">fallback attivo</span> ${esc(fallbackReason)}</div>` : ""}<div class="study-gallery">${cards || "<div class='small'>Marker LR non disponibile.</div>"}</div></div>`;
        return;
      }

      if (studyType === "lt") {
        const lt = (rec.lt_per_image && typeof rec.lt_per_image === "object") ? rec.lt_per_image : {};
        const counts = (lt.label_counts && typeof lt.label_counts === "object") ? lt.label_counts : {};
        const items = Array.isArray(lt.items) ? lt.items : [];
        const cards = items.slice(0, 12).map((it) => studyImageCardHtml(runId, it.image_rel, `frame ${cleanVal(it.image_index)} | ${cleanVal(it.pred_label)}`, `conf=${formatNum(it.confidence, 4)} | crop=${cleanVal(it.crop_source)}`)).join("");
        studyContentEl.innerHTML = header + `<div class="evidence-card">${studyKpiHtml([
          {k: "majority", v: row.lt_majority_label || "-"},
          {k: "L", v: counts.l || 0},
          {k: "T", v: counts.t || 0},
          {k: "mean conf", v: formatNum(row.lt_mean_confidence, 4)},
        ])}<div class="study-gallery">${cards || "<div class='small'>Dati L/T non disponibili.</div>"}</div></div>`;
      }
    }

    async function renderStudyCurrent() {
      if (!studyContentEl || !studyRunPickerEl) return;
      const studyRuns = studyRunsSorted();
      if (!studyRuns.length) {
        renderStudyRunPicker();
        return;
      }
      if (!studyRunId) studyRunId = String(studyRuns[0].run_id || "");
      const runObj = studyRuns.find((run) => String(run.run_id || "") === studyRunId) || studyRuns[0];
      studyRunId = String(runObj.run_id || "");
      studyRunPickerEl.value = studyRunId;
      activeStudyType = String((studyTypeSelectEl && studyTypeSelectEl.value) || activeStudyType || "duplicates");
      if (studyStatusEl) studyStatusEl.textContent = "Carico studio...";
      try {
        const data = await loadStudyRunData(studyRunId);
        if (!data || !data.summary) throw new Error("summary non disponibile");
        renderStudySummary(studyRunId, runObj, data.summary, activeStudyType);
        const idx = studyRuns.findIndex((run) => String(run.run_id || "") === studyRunId);
        if (studyStatusEl) studyStatusEl.textContent = `Studio ${activeStudyType} | cartella ${idx + 1}/${studyRuns.length}`;
      } catch (err) {
        if (studyStatusEl) studyStatusEl.textContent = "Errore studio: " + err.message;
        studyContentEl.innerHTML = `<div class="evidence-card"><div class="small">Dati non disponibili per questa run.</div></div>`;
      }
    }

    function moveStudyRun(delta) {
      const studyRuns = studyRunsSorted();
      if (!studyRuns.length) return;
      let idx = studyRuns.findIndex((run) => String(run.run_id || "") === studyRunId);
      if (idx < 0) idx = 0;
      idx = (idx + Number(delta || 0) + studyRuns.length) % studyRuns.length;
      studyRunId = String(studyRuns[idx].run_id || "");
      renderStudyRunPicker();
      renderStudyCurrent();
    }

    function renderTimeline(statusObj) {
      timelineEl.innerHTML = "";
      const doneSet = new Set((statusObj && statusObj.completed_stages) ? statusObj.completed_stages : []);
      const current = statusObj && statusObj.stage ? String(statusObj.stage) : "";
      STAGES_ORDER.forEach((s) => {
        const span = document.createElement("span");
        span.className = "stage";
        if (doneSet.has(s)) span.classList.add("done");
        if (current === s) span.classList.add("current");
        span.textContent = s;
        timelineEl.appendChild(span);
      });
    }

    function ensureAnnotationsState(checks, existing) {
      const out = existing && typeof existing === "object" ? existing : {};
      if (!out.steps || typeof out.steps !== "object") out.steps = {};
      if (!out.lr_marker_reviews || typeof out.lr_marker_reviews !== "object") out.lr_marker_reviews = {};
      if (typeof out.run_flag !== "string") out.run_flag = "";
      if (typeof out.run_note !== "string") out.run_note = "";
      (checks || []).forEach((step) => {
        const k = String(step.step || "");
        if (!k) return;
        if (!out.steps[k] || typeof out.steps[k] !== "object") out.steps[k] = {};
        if (typeof out.steps[k].flag !== "string") out.steps[k].flag = "";
        if (typeof out.steps[k].comment !== "string") out.steps[k].comment = "";
        if (typeof out.steps[k].correction !== "string") out.steps[k].correction = "";
      });
      return out;
    }

    function renderChecksAndAnalysis() {
      ensureWorkspaceSidebars();
      checksContainerEl.innerHTML = "";
      galleryEl.innerHTML = "";
      if (predictionSummaryEl) predictionSummaryEl.innerHTML = "";
      if (analysisReviewSummaryEl) analysisReviewSummaryEl.innerHTML = "";
      if (fssFileViewerEl) fssFileViewerEl.innerHTML = "";
      decisionEvidenceEl.innerHTML = "";
      if (templateEvidenceEl) templateEvidenceEl.innerHTML = "";
      rotationEvidenceEl.innerHTML = "";
      if (rectPerImageEvidenceEl) rectPerImageEvidenceEl.innerHTML = "";
      if (suGiuEvidenceEl) suGiuEvidenceEl.innerHTML = "";
      if (lrMarkerEvidenceEl) lrMarkerEvidenceEl.innerHTML = "";
      if (ltEvidenceEl) ltEvidenceEl.innerHTML = "";
      if (scaleEvidenceEl) scaleEvidenceEl.innerHTML = "";
      if (depthEvidenceEl) depthEvidenceEl.innerHTML = "";
      dupEvidenceEl.innerHTML = "";
      if (evidenceRectSideHostEl) evidenceRectSideHostEl.innerHTML = "";
      if (rawEvidenceBlockEl) rawEvidenceBlockEl.hidden = true;
      checksActionsEl.style.display = "none";
      if (!selectedRunData || !selectedRunData.summary) {
        checksStatusEl.textContent = "Nessun dato check disponibile.";
        evidenceStatusEl.textContent = "Nessuna evidenza disponibile.";
        renderPredictionSummary(null);
        renderAnalysisReviewSummary(null);
        if (fssStatusEl) fssStatusEl.textContent = "Nessun dato .fss disponibile.";
        if (fssFileViewerEl) fssFileViewerEl.innerHTML = "<div class='small'>Seleziona una run completata per vedere il fake del file .fss.</div>";
        renderDecisionTreeLive(selectedRunStatus, null);
        return;
      }
      const summary = selectedRunData.summary;
      if (!summary.fss_template_path && selectedRunData.fss_template_path) {
        summary.fss_template_path = String(selectedRunData.fss_template_path || "");
      }
      if ((!Array.isArray(summary.fss_template_lines) || !summary.fss_template_lines.length) && Array.isArray(selectedRunData.fss_template_lines)) {
        summary.fss_template_lines = selectedRunData.fss_template_lines.slice();
      }
      const checks = Array.isArray(summary.checks) ? summary.checks : [];
      const annotations = ensureAnnotationsState(checks, selectedRunData.annotations || {});
      annotationsState = annotations;
      checksActionsEl.style.display = "flex";
      checksStatusEl.textContent = `Step disponibili: ${checks.length} | run_dir: ${summary.run_dir || "-"}`;
      const rawCount = Number(summary.raw_images_count || 0);
      evidenceStatusEl.textContent = `Immagini raw: ${rawCount}`;
      renderPredictionSummary(summary);
      renderAnalysisReviewSummary(summary);

      const globalCard = document.createElement("div");
      globalCard.className = "check-card";
      globalCard.innerHTML = `
        <div class="check-head"><div class="name">Annotazione Globale Run</div></div>
        <div class="annot-row">
          <label>Flag globale</label>
          <select id="runFlagSel">
            <option value="">-- seleziona --</option>
            <option value="ok">ok</option>
            <option value="review">review</option>
            <option value="blocked">bloccato</option>
            <option value="needs_reprocess">da rieseguire</option>
          </select>
        </div>
        <div class="annot-row">
          <label>Nota globale</label>
          <textarea id="runNoteTxt" placeholder="commenti globali sulla run..."></textarea>
        </div>
      `;
      checksContainerEl.appendChild(globalCard);
      document.getElementById("runFlagSel").value = annotations.run_flag || "";
      document.getElementById("runNoteTxt").value = annotations.run_note || "";

      checks.forEach((step, idx) => {
        const key = String(step.step || ("step_" + idx));
        const st = annotations.steps[key] || {flag: "", comment: "", correction: ""};
        const card = document.createElement("div");
        card.className = "check-card";
        const status = String(step.status || "review");

        const rows = Object.entries(step)
          .filter(([k]) => k !== "step" && k !== "status")
          .map(([k, v]) => `<tr><th>${k}</th><td>${String(v)}</td></tr>`)
          .join("");

        card.innerHTML = `
          <div class="check-head">
            <div class="name">${idx + 1}. ${key}</div>
            <span class="status-pill ${statusClass(status)}">${status}</span>
          </div>
          <table class="kv">
            <tbody>${rows || "<tr><td colspan='2'>(nessun dettaglio)</td></tr>"}</tbody>
          </table>
          <div class="annot-row">
            <label>Flag QA</label>
            <select data-kind="flag" data-step="${key}">
              <option value="">-- seleziona --</option>
              <option value="ok">ok</option>
              <option value="issue_minor">problema minore</option>
              <option value="issue_major">problema maggiore</option>
              <option value="wrong_output">output errato</option>
              <option value="corrected">corretto manualmente</option>
              <option value="needs_reprocess">da rieseguire</option>
            </select>
          </div>
          <div class="annot-row">
            <label>Commento</label>
            <textarea data-kind="comment" data-step="${key}" placeholder="note, problemi, motivazione..."></textarea>
          </div>
          <div class="annot-row">
            <label>Correzione proposta</label>
            <input data-kind="correction" data-step="${key}" placeholder="es: vendor=BK | rotation=0 | probe=12 ..." />
          </div>
        `;
        checksContainerEl.appendChild(card);
        card.querySelector('[data-kind="flag"]').value = st.flag || "";
        card.querySelector('[data-kind="comment"]').value = st.comment || "";
        card.querySelector('[data-kind="correction"]').value = st.correction || "";
      });

      renderActiveEvidenceSection(summary);
      renderFssPreview(summary);
      renderDecisionTreeLive(selectedRunStatus, summary);
    }

    function collectAnnotationsFromUi() {
      if (!annotationsState) annotationsState = {run_flag: "", run_note: "", steps: {}, lr_marker_reviews: {}};
      if (!annotationsState.steps || typeof annotationsState.steps !== "object") annotationsState.steps = {};
      if (!annotationsState.lr_marker_reviews || typeof annotationsState.lr_marker_reviews !== "object") annotationsState.lr_marker_reviews = {};
      const runFlagEl = document.getElementById("runFlagSel");
      const runNoteEl = document.getElementById("runNoteTxt");
      annotationsState.run_flag = runFlagEl ? (runFlagEl.value || "") : "";
      annotationsState.run_note = runNoteEl ? (runNoteEl.value || "") : "";
      document.querySelectorAll("[data-step]").forEach((el) => {
        const step = String(el.getAttribute("data-step") || "");
        const kind = String(el.getAttribute("data-kind") || "");
        if (!step || !kind) return;
        if (!annotationsState.steps[step]) annotationsState.steps[step] = {flag: "", comment: "", correction: ""};
        if (kind === "flag") annotationsState.steps[step].flag = el.value || "";
        if (kind === "comment") annotationsState.steps[step].comment = el.value || "";
        if (kind === "correction") annotationsState.steps[step].correction = el.value || "";
      });
      const lrDraft = {};
      document.querySelectorAll("[data-lr-marker-review-input]").forEach((el) => {
        const key = String(el.getAttribute("data-lr-review-key") || "");
        const kind = String(el.getAttribute("data-lr-review-kind") || "");
        if (!key || !kind) return;
        if (!lrDraft[key]) lrDraft[key] = {};
        lrDraft[key][kind] = el.value || "";
      });
      Object.entries(lrDraft).forEach(([key, rec]) => {
        const previous = annotationsState.lr_marker_reviews[key] && typeof annotationsState.lr_marker_reviews[key] === "object"
          ? annotationsState.lr_marker_reviews[key]
          : {};
        const item = lrMarkerReviewItemByKey(key);
        const context = item ? lrMarkerReviewContextForItem(item) : {};
        const next = {...previous, ...rec};
        const hasValue = Object.entries(next)
          .filter(([k]) => k !== "context")
          .some(([, v]) => String(v || "").trim());
        if (hasValue) {
          if (Object.keys(context).length) next.context = context;
          annotationsState.lr_marker_reviews[key] = next;
        }
        else delete annotationsState.lr_marker_reviews[key];
      });
      return annotationsState;
    }

    async function loadRuns() {
      try {
        const data = await apiGet("/api/runs");
        runs = Array.isArray(data.runs) ? data.runs : [];
        renderRunsList();
        if (activePane === "study") renderStudyCurrent();
      } catch (err) {
        startStatusEl.textContent = "Errore caricamento lista run: " + err.message;
      }
    }

    function startRunsPolling() {
      if (runsPollTimer) clearInterval(runsPollTimer);
      runsPollTimer = setInterval(() => {
        loadRuns();
        refreshBulkStatus();
      }, 3000);
    }

    async function selectRun(runId) {
      selectedRunId = runId;
      selectedRunDataRevisionKey = "";
      setViewMode("workspace");
      const selectedRow = runs.find((r) => String(r.run_id || "") === String(runId || ""));
      if (selectedRow && selectedRow.input_folder) {
        setActiveFolderTitle(selectedRow.input_folder);
      }
      setHistoryFinderPath(resolveHistoryFinderPath());
      renderRunsList();
      await refreshSelectedRun(true);
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(() => refreshSelectedRun(false), 2000);
    }

    function renderStatus(statusObj) {
      selectedRunStatus = statusObj || null;
      if (!statusObj) {
        runInfoEl.textContent = "Nessuna run selezionata.";
        runLogEl.textContent = "";
        setActiveFolderTitle(document.getElementById("inputFolder").value || "");
        setHistoryFinderPath(resolveHistoryFinderPath());
        renderTimeline(null);
        renderMonitorActivity(null);
        renderDecisionTreeLive(null, selectedRunData ? selectedRunData.summary : null);
        return;
      }
      runInfoEl.textContent = `run_id=${statusObj.run_id || "-"} | input=${statusObj.input_folder || "-"} | run_dir=${statusObj.run_dir || "-"}`;
      setActiveFolderTitle(statusObj.input_folder || "");
      renderTimeline(statusObj);
      const logLines = Array.isArray(statusObj.logs_tail) ? statusObj.logs_tail : [];
      runLogEl.textContent = logLines.join("\\n");
      kpiStatusEl.textContent = statusObj.status || "-";
      kpiStageEl.textContent = statusObj.stage || "-";
      kpiUpdatedEl.textContent = fmtTs(statusObj.updated_at);
      renderMonitorActivity(statusObj);
      renderDecisionTreeLive(statusObj, selectedRunData ? selectedRunData.summary : null);
    }

    async function refreshSelectedRun(forceDataReload) {
      if (!selectedRunId) return;
      try {
        const st = await apiGet(`/api/runs/${encodeURIComponent(selectedRunId)}/status`);
        renderStatus(st);
        const statusTxt = String(st && st.status ? st.status : "").trim();
        const stageTxt = String(st && st.stage ? st.stage : "").trim();
        const updatedTxt = String(st && st.updated_at ? st.updated_at : "").trim();
        const revisionKey = `${statusTxt}|${stageTxt}|${updatedTxt}`;
        const terminal = statusTxt === "completed" || statusTxt === "failed";
        const shouldLoadData = !!forceDataReload || (terminal && revisionKey !== selectedRunDataRevisionKey);
        if (shouldLoadData) {
          const data = await apiGet(`/api/runs/${encodeURIComponent(selectedRunId)}/data`);
          selectedRunData = data;
          selectedRunDataRevisionKey = revisionKey;
          renderChecksAndAnalysis();
        }
      } catch (err) {
        runLogEl.textContent = "Errore status run: " + err.message;
      }
    }

    async function startRun() {
      const payload = buildRunPayload("");
      await startRunWithPayload(payload, "run");
    }

    function openFolderBrowserModal() {
      if (!folderBrowserModalEl) return;
      folderBrowserModalEl.classList.remove("hidden");
    }

    function closeFolderBrowserModal() {
      if (!folderBrowserModalEl) return;
      folderBrowserModalEl.classList.add("hidden");
    }

    function renderFolderListing() {
      if (!browseDirsListEl) return;
      const dirs = (browseListing && Array.isArray(browseListing.dirs)) ? browseListing.dirs : [];
      const parent = browseListing && browseListing.parent ? String(browseListing.parent) : "";
      browseDirsListEl.innerHTML = "";

      if (parent) {
        const up = document.createElement("div");
        up.className = "dir-item";
        up.textContent = "..";
        up.onclick = () => loadFolderListing(parent);
        browseDirsListEl.appendChild(up);
      }

      if (!dirs.length) {
        const empty = document.createElement("div");
        empty.className = "dir-item";
        empty.textContent = "(nessuna sottocartella)";
        browseDirsListEl.appendChild(empty);
        return;
      }

      dirs.forEach((d) => {
        const p = String((d && d.path) || "");
        const name = String((d && d.name) || p);
        const row = document.createElement("div");
        row.className = "dir-item";
        row.innerHTML = `<b>${esc(name)}</b><br><span class="small">${esc(p)}</span>`;
        row.onclick = () => loadFolderListing(p);
        browseDirsListEl.appendChild(row);
      });
    }

    async function loadFolderListing(pathValue) {
      const payload = { path: String(pathValue || "").trim() };
      if (browseStatusEl) browseStatusEl.textContent = "Carico cartelle...";
      try {
        const out = await apiPost("/api/fs/list_dirs", payload);
        browseListing = out;
        browseCurrentPath = String(out.path || "");
        if (browsePathInputEl) browsePathInputEl.value = browseCurrentPath;
        const count = Array.isArray(out.dirs) ? out.dirs.length : 0;
        if (browseStatusEl) browseStatusEl.textContent = `Cartella corrente: ${browseCurrentPath} | sottocartelle: ${count}`;
        renderFolderListing();
      } catch (err) {
        if (browseStatusEl) browseStatusEl.textContent = "Errore lettura cartelle: " + err.message;
      }
    }

    async function openFolderBrowser(initialDir) {
      openFolderBrowserModal();
      await loadFolderListing(initialDir || String(browseDefaultRoot || "").trim());
    }

    async function browseInputFolder() {
      const inputEl = document.getElementById("inputFolder");
      const browseBtn = document.getElementById("browseInputBtn");
      const initialDir = inputEl.value.trim() || String(browseDefaultRoot || "").trim();
      browseBtn.disabled = true;
      startStatusEl.textContent = "Apro file manager di sistema...";
      try {
        const out = await apiPost(
          "/api/dialog/select_folder",
          {initial_dir: initialDir, timeout_sec: 600}
        );
        if (out && out.ok && out.path) {
          inputEl.value = out.path;
          startStatusEl.textContent = "Cartella selezionata (" + String(out.method || "native") + ").";
          return;
        }
        if (out && out.cancelled) {
          startStatusEl.textContent = "Selezione annullata.";
          return;
        }
        throw new Error(out && out.error ? out.error : "Selettore cartelle non disponibile");
      } catch (err) {
        startStatusEl.textContent = "Errore apertura file manager: " + err.message + " (usa 'Browser web' come fallback).";
        alert(
          "Il file manager classico non si è aperto. Dettaglio: " +
          err.message +
          ". Usa 'Browser web' come fallback."
        );
      } finally {
        browseBtn.disabled = false;
      }
    }

    async function saveAnnotations() {
      if (!selectedRunId) return;
      try {
        const payload = collectAnnotationsFromUi();
        await apiPost(`/api/runs/${encodeURIComponent(selectedRunId)}/annotations`, payload);
        checksStatusEl.textContent = "Annotazioni salvate.";
      } catch (err) {
        checksStatusEl.textContent = "Errore salvataggio annotazioni: " + err.message;
      }
    }

    function exportAnnotationsJson() {
      if (!annotationsState) return;
      const txt = JSON.stringify(collectAnnotationsFromUi(), null, 2);
      const blob = new Blob([txt], {type: "application/json;charset=utf-8"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `annotations_${selectedRunId || "run"}.json`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 0);
    }

    function openLegacyHtml() {
      if (!selectedRunData || !selectedRunData.summary || !selectedRunData.summary.review_html) return;
      const url = `/api/runs/${encodeURIComponent(selectedRunId)}/open_review_html`;
      window.open(url, "_blank");
    }

    async function openFinderPath(pathValueRaw) {
      const pathValue = String(pathValueRaw || "").trim();
      if (!pathValue) return;
      try {
        const out = await apiPost("/api/fs/open_in_finder", {path: pathValue});
        startStatusEl.textContent = `Finder aperto: ${String((out && out.path) || pathValue)}`;
      } catch (err) {
        alert("Errore apertura Finder: " + err.message);
        throw err;
      }
    }

    async function openHistoryFinder() {
      const pathValue = String(historyFinderPath || "").trim();
      if (!pathValue) return;
      if (historyOpenFinderBtnEl) historyOpenFinderBtnEl.disabled = true;
      try {
        await openFinderPath(pathValue);
      } finally {
        if (historyOpenFinderBtnEl) historyOpenFinderBtnEl.disabled = !String(historyFinderPath || "").trim();
      }
    }

    paneButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = String(btn.getAttribute("data-pane-target") || "");
        setActivePane(target);
        if (target === "study") {
          renderStudyRunPicker();
          renderStudyCurrent();
        }
      });
      btn.addEventListener("keydown", (ev) => {
        if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
        ev.preventDefault();
        if (!paneButtons.length) return;
        const idx = paneButtons.indexOf(btn);
        if (idx < 0) return;
        const delta = ev.key === "ArrowRight" ? 1 : -1;
        const nextIdx = (idx + delta + paneButtons.length) % paneButtons.length;
        const nextBtn = paneButtons[nextIdx];
        if (!nextBtn) return;
        const target = String(nextBtn.getAttribute("data-pane-target") || "");
        setActivePane(target);
        nextBtn.focus();
      });
    });
    evidenceSectionButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const section = String(btn.getAttribute("data-evidence-section") || "");
        setActiveEvidenceSection(section);
      });
    });
    if (studyTypeSelectEl) {
      studyTypeSelectEl.addEventListener("change", () => {
        activeStudyType = String(studyTypeSelectEl.value || "duplicates");
        renderStudyCurrent();
      });
    }
    if (studyRunPickerEl) {
      studyRunPickerEl.addEventListener("change", () => {
        studyRunId = String(studyRunPickerEl.value || "");
        renderStudyCurrent();
      });
    }
    if (studyPrevRunBtnEl) studyPrevRunBtnEl.addEventListener("click", () => moveStudyRun(-1));
    if (studyNextRunBtnEl) studyNextRunBtnEl.addEventListener("click", () => moveStudyRun(1));
    if (studyRefreshBtnEl) {
      studyRefreshBtnEl.addEventListener("click", async () => {
        studyDataCache = {};
        await loadRuns();
        await renderStudyCurrent();
      });
    }
    if (startBulkBtnEl) startBulkBtnEl.addEventListener("click", startBulkAcquisitionRun);
    if (refreshBulkBtnEl) refreshBulkBtnEl.addEventListener("click", refreshBulkStatus);
    const startFormEl = document.getElementById("startForm");
    if (startFormEl) {
      startFormEl.addEventListener("submit", () => {
        syncExcludedImagesHidden();
      });
    }
    if (loadExclusionsBtnEl) {
      loadExclusionsBtnEl.addEventListener("click", async () => {
        try {
          await loadInputExclusions();
        } catch (err) {
          if (exclusionStatusEl) exclusionStatusEl.textContent = "Errore caricamento esclusioni: " + err.message;
        }
      });
    }
    if (saveExclusionsBtnEl) {
      saveExclusionsBtnEl.addEventListener("click", async () => {
        try {
          await saveInputExclusions();
        } catch (err) {
          if (exclusionStatusEl) exclusionStatusEl.textContent = "Errore salvataggio esclusioni: " + err.message;
        }
      });
    }
    if (clearExclusionsBtnEl) clearExclusionsBtnEl.addEventListener("click", clearInputExclusions);
    if (rerunPresetEl) rerunPresetEl.addEventListener("change", () => applyRerunPreset(rerunPresetEl.value));
    if (rerunSelectAllBtnEl) {
      rerunSelectAllBtnEl.addEventListener("click", () => {
        setRerunTargets(rerunTargetCheckboxes().map((el) => el.getAttribute("data-rerun-target")));
      });
    }
    if (rerunClearBtnEl) rerunClearBtnEl.addEventListener("click", () => setRerunTargets([]));
    if (rerunOnlyOrientationBtnEl) rerunOnlyOrientationBtnEl.addEventListener("click", () => setRerunTargets(RERUN_TARGET_PRESETS.orientation));
    if (rerunOnlyRectBtnEl) rerunOnlyRectBtnEl.addEventListener("click", () => setRerunTargets(RERUN_TARGET_PRESETS.rect));
    document.querySelectorAll("[data-rerun-action]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const action = String(btn.getAttribute("data-rerun-action") || "");
        if (action === "all") {
          setRerunTargets(rerunTargetCheckboxes().map((el) => el.getAttribute("data-rerun-target")));
          return;
        }
        if (action === "clear") {
          setRerunTargets([]);
          return;
        }
        if (action === "orientation") {
          setRerunTargets(RERUN_TARGET_PRESETS.orientation);
          return;
        }
        if (action === "rect") {
          setRerunTargets(RERUN_TARGET_PRESETS.rect);
          return;
        }
        if (action === "execute") {
          try {
            await runSelectedRerunAction();
          } catch (err) {
            setRerunStatusText("Errore rerun: " + err.message);
          }
        }
      });
    });
    if (rerunSelectedBtnEl) {
      rerunSelectedBtnEl.addEventListener("click", async () => {
        try {
          await runSelectedRerunAction();
        } catch (err) {
          setRerunStatusText("Errore rerun: " + err.message);
        }
      });
    }
    const inputFolderEl = document.getElementById("inputFolder");
    if (inputFolderEl) {
      inputFolderEl.addEventListener("change", () => {
        setActiveFolderTitle(inputFolderEl.value || "");
        if (!selectedRunId) setHistoryFinderPath(resolveHistoryFinderPath());
        inputImageRows = [];
        if (imageExclusionListEl) imageExclusionListEl.innerHTML = "";
        syncExcludedImagesHidden();
      });
      inputFolderEl.addEventListener("blur", () => {
        setActiveFolderTitle(inputFolderEl.value || "");
        if (!selectedRunId) setHistoryFinderPath(resolveHistoryFinderPath());
      });
    }
    if (historyOpenFinderBtnEl) {
      historyOpenFinderBtnEl.addEventListener("click", openHistoryFinder);
    }
    if (backToSetupBtnEl) {
      backToSetupBtnEl.addEventListener("click", () => {
        selectedRunId = null;
        selectedRunData = null;
        selectedRunStatus = null;
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        renderRunsList();
        renderStatus(null);
        renderChecksAndAnalysis();
        setViewMode("setup");
        setActivePane("monitor");
      });
    }
    document.getElementById("refreshRunsBtn").addEventListener("click", loadRuns);
    if (refreshModelsBtnEl) {
      refreshModelsBtnEl.addEventListener("click", () => {
        loadModelsRegistry();
      });
    }
    document.getElementById("saveAnnotationsBtn").addEventListener("click", saveAnnotations);
    document.getElementById("exportAnnotationsBtn").addEventListener("click", exportAnnotationsJson);
    document.getElementById("openLegacyHtmlBtn").addEventListener("click", openLegacyHtml);
    document.getElementById("browseCancelBtn").addEventListener("click", closeFolderBrowserModal);
    document.getElementById("browseRefreshBtn").addEventListener("click", () => loadFolderListing(browseCurrentPath));
    document.getElementById("browseUpBtn").addEventListener("click", () => {
      if (!browseListing || !browseListing.parent) return;
      loadFolderListing(String(browseListing.parent));
    });
    document.getElementById("browseSelectCurrentBtn").addEventListener("click", () => {
      const p = String(browseCurrentPath || "");
      if (!p) return;
      document.getElementById("inputFolder").value = p;
      setActiveFolderTitle(p);
      if (!selectedRunId) setHistoryFinderPath(resolveHistoryFinderPath());
      closeFolderBrowserModal();
      startStatusEl.textContent = "Cartella selezionata (browser integrato).";
    });
    folderBrowserModalEl.addEventListener("click", (ev) => {
      if (ev.target === folderBrowserModalEl) closeFolderBrowserModal();
    });
    document.addEventListener("mouseover", (ev) => {
      const lineEl = findActionTarget(ev, ".fss-line[data-fss-desc]");
      if (!lineEl) return;
      const text = String(lineEl.getAttribute("data-fss-desc") || "");
      showFssTooltip(text, ev.clientX, ev.clientY);
    });
    document.addEventListener("mousemove", (ev) => {
      if (!fssHoverTipEl || fssHoverTipEl.hidden) return;
      moveFssTooltip(ev.clientX, ev.clientY);
    });
    document.addEventListener("mouseout", (ev) => {
      const fromEl = ev.target instanceof Element ? ev.target.closest(".fss-line[data-fss-desc]") : null;
      if (!fromEl) return;
      const toEl = ev.relatedTarget instanceof Element ? ev.relatedTarget.closest(".fss-line[data-fss-desc]") : null;
      if (toEl === fromEl) return;
      hideFssTooltip();
    });
    window.addEventListener("blur", hideFssTooltip);
    window.addEventListener("scroll", hideFssTooltip, true);
    document.addEventListener("click", (ev) => {
      const rawActionBtn = findActionTarget(ev, "[data-raw-exclusion-action]");
      if (rawActionBtn) {
        ev.preventDefault();
        const action = String(rawActionBtn.getAttribute("data-raw-exclusion-action") || "").trim();
        if (action === "save") {
          saveRawExclusions().catch((err) => {
            if (rawExclusionStatusEl) rawExclusionStatusEl.textContent = "Errore salvataggio esclusioni: " + err.message;
          });
        } else if (action === "clear") {
          clearRawExclusions();
        }
        return;
      }
      const galleryBtn = findActionTarget(ev, '[data-open-evidence-gallery]');
      if (galleryBtn) {
        ev.preventDefault();
        const kind = String(galleryBtn.getAttribute("data-open-evidence-gallery") || "duplicates");
        const lrGroup = String(galleryBtn.getAttribute("data-lr-marker-group") || "").trim();
        openEvidenceGallery(kind, lrGroup);
        return;
      }
      const lrNavBtn = findActionTarget(ev, "[data-lr-marker-nav]");
      if (lrNavBtn) {
        ev.preventDefault();
        moveLrMarkerGallery(Number(lrNavBtn.getAttribute("data-lr-marker-nav") || "0"));
        return;
      }
      const lrThumbBtn = findActionTarget(ev, "[data-lr-marker-index]");
      if (lrThumbBtn) {
        ev.preventDefault();
        const idx = Number(lrThumbBtn.getAttribute("data-lr-marker-index") || "0");
        if (Number.isFinite(idx)) {
          lrMarkerGalleryIndex = Math.trunc(idx);
          renderLrMarkerGalleryViewer();
        }
        return;
      }
      const lrPickModeBtn = findActionTarget(ev, "[data-lr-marker-pick-mode]");
      if (lrPickModeBtn) {
        ev.preventDefault();
        setLrMarkerManualPickMode(!lrMarkerManualPickMode);
        return;
      }
      const lrPickWrap = lrMarkerManualPickMode && ev.target instanceof Element
        ? ev.target.closest(".lr-marker-image-wrap[data-lr-marker-review-key]")
        : null;
      if (lrPickWrap) {
        ev.preventDefault();
        if (handleLrMarkerManualPick(ev, lrPickWrap)) return;
      }
      const btn = findActionTarget(ev, '[data-open-dup-compare="1"]');
      if (btn) {
        ev.preventDefault();
        const kept = String(btn.getAttribute("data-kept") || "");
        const removed = String(btn.getAttribute("data-removed") || "");
        const sha = String(btn.getAttribute("data-sha") || "");
        const size = String(btn.getAttribute("data-size") || "");
        const runId = String(btn.getAttribute("data-run-id") || selectedRunId || "");
        openDupCompareModal(kept, removed, sha, size, runId);
        return;
      }
      const rectBtn = findActionTarget(ev, '[data-open-rect-gallery="1"]');
      if (rectBtn) {
        ev.preventDefault();
        const runId = String(rectBtn.getAttribute("data-run-id") || selectedRunId || "");
        const startIndex = Number(rectBtn.getAttribute("data-start-index") || "0");
        const mode = String(rectBtn.getAttribute("data-open-rect-gallery-mode") || "rect").trim().toLowerCase();
        openRectGallery(
          Number.isFinite(startIndex) ? Math.trunc(startIndex) : 0,
          null,
          runId,
          mode === "rect_red" ? "rect_red" : "rect",
        );
        return;
      }
      const openFolderBtn = findActionTarget(ev, '[data-open-folder-path]');
      if (openFolderBtn) {
        ev.preventDefault();
        const folderPath = String(openFolderBtn.getAttribute("data-open-folder-path") || "");
        openFinderPath(folderPath).catch(() => {});
        return;
      }
    });
    document.addEventListener("mousedown", (ev) => {
      const lrPickWrap = lrMarkerManualPickMode && ev.target instanceof Element
        ? ev.target.closest(".lr-marker-image-wrap[data-lr-marker-review-key]")
        : null;
      if (!lrPickWrap) return;
      if (beginLrMarkerManualDrag(ev, lrPickWrap)) {
        ev.preventDefault();
      }
    });
    document.addEventListener("mouseup", (ev) => {
      if (!lrMarkerManualDragStart) return;
      if (finishLrMarkerManualDrag(ev)) {
        ev.preventDefault();
      }
    });
    document.addEventListener("keydown", (ev) => {
      if (!evidenceGalleryModalEl || evidenceGalleryModalEl.classList.contains("hidden")) return;
      if (!evidenceGalleryCardEl || !evidenceGalleryCardEl.classList.contains("lr-marker-mode")) return;
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        moveLrMarkerGallery(-1);
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        moveLrMarkerGallery(1);
      }
    });
    if (dupCompareModalEl) {
      dupCompareModalEl.addEventListener("click", (ev) => {
        if (ev.target === dupCompareModalEl) closeDupCompareModal();
      });
    }
    const dupCompareCloseBtn = document.getElementById("dupCompareCloseBtn");
    if (dupCompareCloseBtn) {
      dupCompareCloseBtn.addEventListener("click", closeDupCompareModal);
    }
    if (rectCompareModalEl) {
      rectCompareModalEl.addEventListener("click", (ev) => {
        if (ev.target === rectCompareModalEl) closeRectCompareModal();
      });
    }
    const rectCompareCloseBtn = document.getElementById("rectCompareCloseBtn");
    if (rectCompareCloseBtn) {
      rectCompareCloseBtn.addEventListener("click", closeRectCompareModal);
    }
    if (rectComparePrevBtnEl) {
      rectComparePrevBtnEl.addEventListener("click", () => moveRectGallery(-1));
    }
    if (rectCompareNextBtnEl) {
      rectCompareNextBtnEl.addEventListener("click", () => moveRectGallery(1));
    }
    if (rectCompareThumbsEl) {
      rectCompareThumbsEl.addEventListener("click", (ev) => {
        if (activeModalGallery !== "rect" || !_isModalVisible(rectCompareModalEl)) return;
        const btn = findActionTarget(ev, "[data-rect-thumb-index]");
        if (!btn) return;
        const idx = Number(btn.getAttribute("data-rect-thumb-index") || "0");
        if (!Number.isFinite(idx)) return;
        rectGalleryIndex = Math.max(0, Math.min(rectGalleryItems.length - 1, Math.trunc(idx)));
        renderRectGalleryCurrent();
      });
      rectCompareThumbsEl.addEventListener("wheel", (ev) => {
        if (activeModalGallery !== "rect" || !_isModalVisible(rectCompareModalEl)) return;
        if (Math.abs(ev.deltaY) <= Math.abs(ev.deltaX)) return;
        rectCompareThumbsEl.scrollLeft += ev.deltaY;
        ev.preventDefault();
      }, {passive: false});
    }
    document.addEventListener("keydown", (ev) => {
      const listTarget = ev.target instanceof Element
        ? ev.target.closest(".segment-list[data-segment-current-index]")
        : null;
      if (listTarget && (ev.key === "ArrowUp" || ev.key === "ArrowDown")) {
        ev.preventDefault();
        const delta = ev.key === "ArrowUp" ? -1 : 1;
        const current = Number(listTarget.getAttribute("data-segment-current-index") || "0");
        const base = Number.isFinite(current) ? Math.trunc(current) : 0;
        setSegmentListActiveIndex(listTarget, base + delta, {behavior: "smooth"});
        return;
      }
      if (activeModalGallery !== "rect") return;
      if (!rectCompareModalEl || rectCompareModalEl.classList.contains("hidden")) return;
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        moveRectGallery(-1);
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        moveRectGallery(1);
      }
    });
    window.addEventListener("resize", () => {
      if (!rectCompareModalEl || rectCompareModalEl.classList.contains("hidden")) return;
      renderRectGalleryCurrent();
    });
    if (evidenceGalleryModalEl) {
      evidenceGalleryModalEl.addEventListener("click", (ev) => {
        if (ev.target === evidenceGalleryModalEl) closeEvidenceGallery();
      });
    }
    const evidenceGalleryCloseBtn = document.getElementById("evidenceGalleryCloseBtn");
    if (evidenceGalleryCloseBtn) {
      evidenceGalleryCloseBtn.addEventListener("click", closeEvidenceGallery);
    }

    (async () => {
      setViewMode(initialSelectedRunId ? "workspace" : "setup");
      setActiveFolderTitle(document.getElementById("inputFolder").value || "");
      setHistoryFinderPath(resolveHistoryFinderPath());
      renderDecisionTreeLive(null, null);
      try {
        const storedPane = window.localStorage.getItem("workbench.activePane");
        setActivePane(storedPane || "monitor");
      } catch (err) {
        setActivePane("monitor");
      }
      try {
        const storedEvidence = window.localStorage.getItem("workbench.activeEvidenceSection");
        setActiveEvidenceSection(storedEvidence || "raw");
      } catch (err) {
        setActiveEvidenceSection("raw");
      }
      bindImageErrorFallback(document);
      await loadModelsRegistry();
      await refreshBulkStatus();
      await loadRuns();
      startRunsPolling();
      if (initialSelectedRunId) {
        await selectRun(initialSelectedRunId);
      }
    })();
  </script>
</body>
</html>
"""


BROWSE_PAGE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seleziona Cartella Input</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe7;
      --bg-overlay: rgba(201, 110, 55, 0.16);
      --card: rgba(255, 250, 243, 0.88);
      --ink: #221a12;
      --muted: #635344;
      --line: rgba(65, 48, 26, 0.16);
      --primary: #0f766e;
      --primary-2: #0b5b55;
      --radius-md: 14px;
      --font-body: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --font-display: "Iowan Old Style", "Palatino Linotype", serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 18px;
      font-family: var(--font-body);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 110, 55, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.16), transparent 30%),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 45%, #eee4d3 100%);
    }
    html[data-darkreader-scheme="dark"] body {
      background-color: #1f1a14 !important;
      background-image: linear-gradient(180deg, rgba(201, 110, 55, 0.18) 0%, rgba(15, 118, 110, 0.12) 45%, rgba(15, 118, 110, 0.04) 80%) !important;
    }
    .wrap { max-width: 980px; margin: 0 auto; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px;
      margin-bottom: 12px;
      box-shadow: 0 10px 28px rgba(54, 37, 22, 0.10);
      backdrop-filter: blur(14px);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 24px;
      font-family: var(--font-display);
      letter-spacing: -0.02em;
    }
    .muted { color: var(--muted); font-size: 12px; }
    .row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    @media (max-width: 760px) {
      .row { grid-template-columns: 1fr; }
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      font-size: 14px;
      background: #fffdf9;
      color: var(--ink);
    }
    .btn {
      border: 1px solid var(--primary);
      background: var(--primary);
      color: #fff;
      border-radius: 999px;
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      text-align: center;
      transition: transform 0.18s ease, background 0.18s ease;
    }
    .btn:hover {
      transform: translateY(-1px);
    }
    .btn.secondary {
      color: var(--ink);
      background: rgba(34, 26, 18, 0.08);
      border-color: rgba(65, 48, 26, 0.18);
    }
    .dir-list {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      overflow: auto;
      background: rgba(255, 253, 248, 0.94);
      max-height: 68vh;
    }
    .dir-item {
      display: block;
      padding: 10px;
      border-bottom: 1px solid #eef1f4;
      text-decoration: none;
      color: inherit;
    }
    .dir-item:last-child { border-bottom: none; }
    .dir-item:hover { background: rgba(201, 110, 55, 0.10); }
    .dir-name { font-weight: 600; }
    .dir-path { font-size: 11px; color: var(--muted); margin-top: 3px; word-break: break-all; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .err {
      border: 1px solid rgba(184, 77, 69, 0.35);
      border-radius: var(--radius-md);
      background: rgba(184, 77, 69, 0.12);
      color: #8f2d2a;
      padding: 8px 10px;
      margin-top: 8px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Seleziona Cartella Input</h1>
      <div class="muted">Fallback robusto: usa questa pagina se il modal JS non si apre.</div>
      <form method="get" action="/browse" class="row">
        <input type="text" name="path" value="{{ current_path|e }}" placeholder="/percorso/cartella">
        <button type="submit" class="btn secondary">Vai</button>
        <a href="/" class="btn secondary">Torna alla Pipeline</a>
      </form>
      <div class="actions">
        {% if parent_path %}
        <a class="btn secondary" href="/browse?path={{ parent_path|urlencode }}">Su</a>
        {% endif %}
        <a class="btn" href="/?input_folder={{ current_path|urlencode }}">Usa questa cartella</a>
      </div>
      {% if error_msg %}
      <div class="err">{{ error_msg }}</div>
      {% endif %}
    </div>
    <div class="card">
      <div class="muted">Cartella corrente: {{ current_path }}</div>
      <div class="dir-list">
        {% if dirs|length == 0 %}
        <div class="dir-item">
          <div class="dir-name">(nessuna sottocartella)</div>
        </div>
        {% endif %}
        {% for d in dirs %}
        <a class="dir-item" href="/browse?path={{ d.path|urlencode }}">
          <div class="dir-name">{{ d.name }}</div>
          <div class="dir-path">{{ d.path }}</div>
        </a>
        {% endfor %}
      </div>
    </div>
  </div>
</body>
</html>
"""


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}g {hours:02d}h"


def _run_duration_payload(*, row: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    status = str((state or {}).get("status") or (row or {}).get("status") or "").strip()
    started_at = (
        (state or {}).get("started_at")
        or (row or {}).get("started_at")
        or (row or {}).get("created_at")
    )
    finished_at = (
        (state or {}).get("finished_at")
        or (row or {}).get("finished_at")
    )
    start_dt = _parse_iso_dt(started_at)
    end_dt = _parse_iso_dt(finished_at)
    running = status in {"queued", "running"} and start_dt is not None and end_dt is None
    if start_dt is None:
        return {
            "elapsed_seconds": None,
            "elapsed_text": "-",
            "started_at": str(started_at or ""),
            "finished_at": str(finished_at or ""),
        }
    if end_dt is None:
        end_dt = datetime.now(timezone.utc) if running else _parse_iso_dt((state or {}).get("updated_at"))
    if end_dt is None:
        return {
            "elapsed_seconds": None,
            "elapsed_text": "-",
            "started_at": str(started_at or ""),
            "finished_at": str(finished_at or ""),
        }
    elapsed = max(0.0, (end_dt - start_dt).total_seconds())
    text = _format_elapsed(elapsed)
    if running:
        text = f"{text} in corso"
    return {
        "elapsed_seconds": float(elapsed),
        "elapsed_text": text,
        "started_at": str(started_at or ""),
        "finished_at": str(finished_at or ""),
    }


class WorkbenchStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.runs_root = self.data_root / "runs"
        self.index_path = self.data_root / "index.json"
        self.lock = threading.Lock()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"runs": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index_unsafe(self) -> Dict[str, Any]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"runs": []}

    def _save_index_unsafe(self, data: Dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_runs(self) -> List[Dict[str, Any]]:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                return []
            out: List[Dict[str, Any]] = []
            for row_raw in runs:
                row = dict(row_raw) if isinstance(row_raw, dict) else {}
                run_id = str(row.get("run_id", "") or "").strip()
                state = self.load_state(run_id) if run_id else {}
                duration = _run_duration_payload(row=row, state=state)
                row.update(duration)
                out.append(row)
            return out

    def create_run_record(self, record: Dict[str, Any]) -> None:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            runs.insert(0, record)
            data["runs"] = runs
            self._save_index_unsafe(data)

    def update_run_record(self, run_id: str, updates: Dict[str, Any]) -> None:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            changed = False
            for row in runs:
                if str(row.get("run_id", "")) == run_id:
                    row.update(updates)
                    changed = True
                    break
            if changed:
                data["runs"] = runs
                self._save_index_unsafe(data)

    def delete_run_record(self, run_id: str) -> bool:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            original_len = len(runs)
            runs = [row for row in runs if str(row.get("run_id", "")) != run_id]
            if len(runs) == original_len:
                return False
            data["runs"] = runs
            self._save_index_unsafe(data)
            return True

    def clear_all_runs(self, delete_files: bool = True) -> int:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            count = len(runs)
            data["runs"] = []
            self._save_index_unsafe(data)
        if delete_files and self.runs_root.exists():
            for child in list(self.runs_root.iterdir()):
                try:
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=False)
                    else:
                        child.unlink()
                except FileNotFoundError:
                    continue
        return count

    def prune_missing_run_records(self) -> int:
        with self.lock:
            data = self._load_index_unsafe()
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            kept: List[Dict[str, Any]] = []
            removed = 0
            for row in runs:
                rid = str((row or {}).get("run_id", "")).strip()
                if not rid:
                    removed += 1
                    continue
                run_dir = self.run_dir(rid)
                if run_dir.exists() and run_dir.is_dir():
                    kept.append(row)
                else:
                    removed += 1
            if removed > 0:
                data["runs"] = kept
                self._save_index_unsafe(data)
            return removed

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run_state.json"

    def annotations_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "review_annotations.json"

    def load_state(self, run_id: str) -> Dict[str, Any]:
        path = self.state_path(run_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        path = self.state_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_log(self, run_id: str, line: str, max_lines: int = 1000) -> Dict[str, Any]:
        state = self.load_state(run_id)
        logs = state.get("logs", [])
        if not isinstance(logs, list):
            logs = []
        logs.append(line.rstrip("\n"))
        if len(logs) > max_lines:
            logs = logs[-max_lines:]
        state["logs"] = logs
        self.save_state(run_id, state)
        return state

    def append_event(self, run_id: str, event_obj: Dict[str, Any], max_events: int = 400) -> Dict[str, Any]:
        state = self.load_state(run_id)
        events = state.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append(event_obj)
        if len(events) > max_events:
            events = events[-max_events:]
        state["events"] = events

        completed = state.get("completed_stages", [])
        if not isinstance(completed, list):
            completed = []
        event_name = str(event_obj.get("event", "")).strip()
        if event_name and event_name not in completed:
            completed.append(event_name)
        state["completed_stages"] = completed
        state["stage"] = event_name or state.get("stage", "running")
        self.save_state(run_id, state)
        return state


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rect_metrics_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    top = _safe_int(row.get("line_11_top"), 0)
    left = _safe_int(row.get("line_11_left"), 0)
    bottom = _safe_int(row.get("line_11_bottom"), 0)
    right = _safe_int(row.get("line_11_right"), 0)
    frame_w = _safe_int(row.get("line_09_video_x_size"), 0)
    frame_h = _safe_int(row.get("line_10_video_y_size"), 0)
    images_used = _safe_int(row.get("images_used_rect"), 0)
    source = str(row.get("line_11_source", "") or "").strip()
    has_box = (bottom > top) and (right > left)
    width = max(0, right - left) if has_box else 0
    height = max(0, bottom - top) if has_box else 0
    area = width * height
    frame_area = frame_w * frame_h if (frame_w > 0 and frame_h > 0) else 0
    area_ratio = float(area / frame_area) if frame_area > 0 else 0.0
    aspect_ratio = float(width / height) if height > 0 else 0.0

    exported = _safe_float(row.get("line_11_confidence_proxy"), -1.0)
    if 0.0 <= exported <= 1.0:
        confidence = exported
        method = "model_exported"
        note = "confidence esportata dal runner line11"
    else:
        source_score = {
            "vendor_specialized": 0.95,
            "global": 0.88,
            "global_low_vendor_conf": 0.74,
        }.get(source, 0.70)
        size_score = min(1.0, max(0.0, float(images_used) / 24.0))
        area_score = 0.40
        if area_ratio > 0.0:
            target = 0.42
            tolerance = 0.40
            area_score = max(0.0, min(1.0, 1.0 - (abs(area_ratio - target) / tolerance)))
        confidence = max(0.0, min(1.0, 0.45 * size_score + 0.35 * area_score + 0.20 * source_score))
        method = "heuristic_proxy"
        note = "proxy da source+area+frame usati (non probabilita del modello)"

    return {
        "confidence": confidence,
        "method": method,
        "note": note,
        "source": source,
        "images_used": images_used,
        "width": width,
        "height": height,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
    }


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    txt = str(value or "").strip()
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _rect_per_image_metrics(rect_data: Dict[str, Any]) -> Dict[str, Any]:
    items_raw = rect_data.get("items", [])
    if not isinstance(items_raw, list):
        items_raw = []
    ious: List[float] = []
    discrepancies: List[float] = []
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        iou = _safe_float(item.get("iou"), float("nan"))
        disc = _safe_float(item.get("discrepancy"), float("nan"))
        if iou == iou:
            ious.append(float(iou))
        if disc == disc:
            discrepancies.append(float(disc))
    if not ious:
        return {
            "count": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "p25_iou": 0.0,
            "p75_iou": 0.0,
            "mean_discrepancy": 0.0,
        }

    ious_sorted = sorted(ious)
    n = len(ious_sorted)

    def _q(q: float) -> float:
        if n <= 1:
            return float(ious_sorted[0])
        pos = max(0.0, min(float(n - 1), float(q) * float(n - 1)))
        lo = int(pos)
        hi = min(n - 1, lo + 1)
        alpha = pos - float(lo)
        return float((1.0 - alpha) * ious_sorted[lo] + alpha * ious_sorted[hi])

    mean_iou = float(sum(ious_sorted) / float(n))
    if n % 2 == 1:
        median_iou = float(ious_sorted[n // 2])
    else:
        median_iou = float((ious_sorted[(n // 2) - 1] + ious_sorted[n // 2]) / 2.0)
    mean_disc = float(sum(discrepancies) / float(len(discrepancies))) if discrepancies else 0.0
    return {
        "count": int(n),
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "p25_iou": _q(0.25),
        "p75_iou": _q(0.75),
        "mean_discrepancy": mean_disc,
    }


def _remap_rect_per_image_paths_to_run_root(
    *,
    run_root: Path,
    local_root: Path,
    rect_data: Dict[str, Any],
) -> Dict[str, Any]:
    out = json.loads(json.dumps(rect_data, ensure_ascii=False))
    rel_prefix = local_root.relative_to(run_root).as_posix()
    items_raw = out.get("items", [])
    if not isinstance(items_raw, list):
        return out
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        overlay_rel = str(item.get("overlay_rel", "") or "").strip()
        if overlay_rel:
            item["overlay_rel"] = f"{rel_prefix}/{overlay_rel}"
        image_rel = str(item.get("image_rel", "") or "").strip()
        if image_rel:
            image_path = Path(image_rel)
            if image_path.is_absolute():
                try:
                    item["image_rel"] = image_path.relative_to(run_root).as_posix()
                except Exception:
                    # Keep original if image path is outside run root.
                    item["image_rel"] = image_rel
    return out


def _resolve_allowed_preview_path(run_dir: Path, path_value: str) -> Optional[Path]:
    txt = str(path_value or "").strip()
    if not txt:
        return None
    p = Path(txt).expanduser()
    try:
        target = p.resolve() if p.is_absolute() else (run_dir / p).resolve()
    except Exception:
        return None
    allowed_roots = [run_dir.resolve(), DEFAULT_INPUT_BROWSE_ROOT.resolve()]
    allowed_roots.extend(root.resolve() for root in DEFAULT_LR_MARKER_TEMPLATE_ROOTS if root.exists())
    for root in allowed_roots:
        try:
            target.relative_to(root)
            return target
        except Exception:
            continue
    return None


def _preview_allowed_roots_for_run(run_dir: Path, state: Optional[Dict[str, Any]] = None) -> List[Path]:
    roots: List[Path] = [run_dir.resolve(), DEFAULT_INPUT_BROWSE_ROOT.resolve()]
    if isinstance(state, dict):
        for key in ("input_folder", "run_dir"):
            value = str(state.get(key, "") or "").strip()
            if not value:
                continue
            try:
                path = Path(value).expanduser().resolve()
            except Exception:
                continue
            if path.exists():
                roots.append(path)
    input_ref = run_dir / "input_ref"
    if input_ref.exists():
        try:
            roots.append(input_ref.resolve())
        except Exception:
            pass
    roots.extend(root.resolve() for root in DEFAULT_LR_MARKER_TEMPLATE_ROOTS if root.exists())
    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = root.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _coords_dict_to_tlbr(obj: Any) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(obj, dict):
        try:
            top = int(float(obj.get("top", 0)))
            left = int(float(obj.get("left", 0)))
            bottom = int(float(obj.get("bottom", 0)))
            right = int(float(obj.get("right", 0)))
        except Exception:
            return None
        if bottom <= top or right <= left:
            return None
        return (top, left, bottom, right)
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        try:
            top = int(float(obj[0]))
            left = int(float(obj[1]))
            bottom = int(float(obj[2]))
            right = int(float(obj[3]))
        except Exception:
            return None
        if bottom <= top or right <= left:
            return None
        return (top, left, bottom, right)
    return None


def _row_tlbr_from_prefix(row: Dict[str, Any], prefix: str) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(row, dict):
        return None
    try:
        top = int(float(row.get(f"{prefix}_top", 0)))
        left = int(float(row.get(f"{prefix}_left", 0)))
        bottom = int(float(row.get(f"{prefix}_bottom", 0)))
        right = int(float(row.get(f"{prefix}_right", 0)))
    except Exception:
        return None
    if bottom <= top or right <= left:
        return None
    return (top, left, bottom, right)


def _extract_overlay_rect_coords_by_color(
    overlay_path: Path, color_name: str
) -> Optional[Tuple[int, int, int, int]]:
    # Fallback best-effort from overlay colors.
    # Priority path should be explicit coords from step_checks (pre_dark_trim/final).
    try:
        from PIL import Image
    except Exception:
        return None

    cname = str(color_name or "").strip().lower()
    if cname == "blue":
        target_rgb = (32, 128, 255)
    elif cname == "yellow":
        target_rgb = (255, 219, 77)
    else:
        return None

    try:
        with Image.open(overlay_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            pix = rgb.load()
            # Ignore footer legend area to avoid catching color swatches.
            y_limit = max(1, int(round(h * 0.88)))
            tol = 36
            xs: List[int] = []
            ys: List[int] = []
            tr, tg, tb = target_rgb
            for y in range(0, y_limit):
                for x in range(0, w):
                    r, g, b = pix[x, y]
                    if abs(int(r) - tr) <= tol and abs(int(g) - tg) <= tol and abs(int(b) - tb) <= tol:
                        xs.append(x)
                        ys.append(y)
    except Exception:
        return None

    if len(xs) < 16 or len(ys) < 16:
        return None
    top = max(0, min(ys))
    left = max(0, min(xs))
    bottom = min(max(1, h - 1), max(ys) + 1)
    right = min(max(1, w - 1), max(xs) + 1)
    if bottom <= top or right <= left:
        return None
    return (int(top), int(left), int(bottom), int(right))


def _ensure_line13_template_crop(summary: Dict[str, Any], run_dir: Path) -> None:
    if not isinstance(summary, dict):
        return
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        return
    l13 = rec.get("line13_template", {})
    if not isinstance(l13, dict):
        return

    row = summary.get("pipeline_row", {})
    if not isinstance(row, dict):
        row = {}

    overlay_rel = str(l13.get("overlay_rel", "") or "").strip()
    overlay_path = _resolve_allowed_preview_path(run_dir, overlay_rel) if overlay_rel else None
    if overlay_path is None or (not overlay_path.is_file()):
        overlay_candidate = (run_dir / overlay_rel) if overlay_rel else None
        overlay_path = overlay_candidate if (overlay_candidate and overlay_candidate.is_file()) else None

    source_rel = str(l13.get("source_image_rel", "") or "").strip()
    if not source_rel:
        rect_overlay = rec.get("rect_overlay", {})
        if isinstance(rect_overlay, dict):
            source_rel = str(rect_overlay.get("source_rel", "") or "").strip()
    source_path = _resolve_allowed_preview_path(run_dir, source_rel)
    if source_path is None or (not source_path.is_file()):
        source_candidate = (run_dir / source_rel) if source_rel else None
        if source_candidate is not None and source_candidate.is_file():
            source_path = source_candidate
        elif overlay_path is not None and overlay_path.is_file():
            source_path = overlay_path
        else:
            return

    l13_text = str(l13.get("value", "") or row.get("line_13_rect_name_echo", "") or "").strip()
    final_coords = (
        _coords_dict_to_tlbr(l13.get("coords"))
        or _runner_parse_rect_coords(l13_text)
        or _row_tlbr_from_prefix(row, "line_13")
    )
    pre_dark_value = str(
        l13.get("pre_dark_trim_value", "") or row.get("line_13_pre_dark_trim_rect_name_echo", "") or ""
    ).strip()
    pre_dark_coords = (
        _coords_dict_to_tlbr(l13.get("pre_dark_trim_coords"))
        or _runner_parse_rect_coords(pre_dark_value)
    )

    # 1) Crop finale/post-processing (giallo quando disponibile)
    post_coords = final_coords
    if post_coords is None and overlay_path is not None:
        post_coords = _extract_overlay_rect_coords_by_color(overlay_path, "yellow")

    if post_coords is not None:
        target_post = run_dir / "evidence" / "line13_template_crop_post.png"
        try:
            _runner_save_rect_crop_preview(
                source=source_path,
                target=target_post,
                rect_coords=post_coords,
                pad_ratio=0.025,
                min_side=820,
                max_side=1900,
            )
            post_rel = target_post.relative_to(run_dir).as_posix()
            l13["crop_rel"] = post_rel
            if not str(l13.get("postprocess_crop_rel", "") or "").strip():
                l13["postprocess_crop_rel"] = post_rel
            if not str(l13.get("finetuned_crop_rel", "") or "").strip():
                l13["finetuned_crop_rel"] = post_rel
            if not str(l13.get("template_crop_rel", "") or "").strip():
                l13["template_crop_rel"] = post_rel
        except Exception:
            pass

    # 2) Crop rete raw (pre dark-trim se disponibile, altrimenti fallback colore overlay)
    raw_coords = pre_dark_coords
    if raw_coords is None and overlay_path is not None:
        raw_coords = _extract_overlay_rect_coords_by_color(overlay_path, "blue")
    if raw_coords is not None:
        target_raw = run_dir / "evidence" / "line13_template_crop_raw.png"
        try:
            _runner_save_rect_crop_preview(
                source=source_path,
                target=target_raw,
                rect_coords=raw_coords,
                pad_ratio=0.025,
                min_side=820,
                max_side=1900,
            )
            raw_rel = target_raw.relative_to(run_dir).as_posix()
            if not str(l13.get("raw_crop_rel", "") or "").strip():
                l13["raw_crop_rel"] = raw_rel
            if not str(l13.get("model_crop_rel", "") or "").strip():
                l13["model_crop_rel"] = raw_rel
            if not str(l13.get("pre_finetune_crop_rel", "") or "").strip():
                l13["pre_finetune_crop_rel"] = raw_rel
        except Exception:
            pass

    # Overlay alias coerenti per UI (non sovrascriviamo valori espliciti)
    if overlay_rel:
        if not str(l13.get("postprocess_overlay_rel", "") or "").strip():
            l13["postprocess_overlay_rel"] = overlay_rel
        if not str(l13.get("finetuned_overlay_rel", "") or "").strip():
            l13["finetuned_overlay_rel"] = overlay_rel
        if not str(l13.get("template_overlay_rel", "") or "").strip():
            l13["template_overlay_rel"] = overlay_rel
        if not str(l13.get("raw_overlay_rel", "") or "").strip():
            l13["raw_overlay_rel"] = overlay_rel
        if not str(l13.get("model_overlay_rel", "") or "").strip():
            l13["model_overlay_rel"] = overlay_rel
        if not str(l13.get("pre_finetune_overlay_rel", "") or "").strip():
            l13["pre_finetune_overlay_rel"] = overlay_rel

    try:
        if source_rel and (not str(l13.get("source_image_rel", "") or "").strip()):
            l13["source_image_rel"] = source_rel
        rec["line13_template"] = l13
        summary["recognition_evidence"] = rec
    except Exception:
        return


def _lr_marker_evidence_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _recompute_line16_from_lr_marker_evidence(summary: Dict[str, Any], evidence: Dict[str, Any]) -> None:
    items = evidence.get("items", [])
    if not isinstance(items, list):
        return
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        rec = {}
    model = rec.get("model_evidence", {})
    if not isinstance(model, dict):
        model = {}
    row = summary.get("pipeline_row", {})
    if not isinstance(row, dict):
        row = {}

    min_score = max(0.62, _safe_float(model.get("lr_marker_min_match_score"), 0.62))
    target_w = _safe_int(row.get("line_09_video_x_size") or row.get("line_07_video_input_size_x"), 0)
    target_h = _safe_int(row.get("line_10_video_y_size") or row.get("line_08_video_input_size_y"), 0)
    order = ("NF", "LR", "UD", "LRUD")
    groups: Dict[str, Dict[str, Any]] = {
        key: {"top": None, "left": None, "bottom": None, "right": None, "boxes": 0}
        for key in order
    }
    boxes_count = 0
    for item_raw in items:
        item = dict(item_raw) if isinstance(item_raw, dict) else {}
        group_key = str(item.get("quadrant_group", "") or "").strip().upper()
        if group_key not in groups:
            continue
        if str(item.get("quadrant_status", "") or "").strip().lower() == "invalid":
            continue
        if str(item.get("quadrant_valid", "") or "").strip() and not _lr_marker_evidence_truthy(item.get("quadrant_valid")):
            continue
        score = _safe_float(item.get("match_score"), 0.0)
        if score < min_score:
            continue
        if str(item.get("search_strategy", "") or "") == "spatial_consensus_forced":
            continue
        review_reason = str(item.get("review_reason", "") or "")
        if _lr_marker_evidence_truthy(item.get("match_patch_is_blank")) or "blank_marker_match" in review_reason.split(";"):
            continue
        image_w = _safe_int(item.get("image_width"), 0)
        image_h = _safe_int(item.get("image_height"), 0)
        if target_w > 0 and target_h > 0 and image_w > 0 and image_h > 0 and (image_w != target_w or image_h != target_h):
            continue
        top = _safe_int(item.get("marker_top_abs"), 0)
        left = _safe_int(item.get("marker_left_abs"), 0)
        bottom = _safe_int(item.get("marker_bottom_abs"), 0)
        right = _safe_int(item.get("marker_right_abs"), 0)
        if bottom <= top or right <= left:
            continue
        env = groups[group_key]
        env["top"] = top if env["top"] is None else min(int(env["top"]), top)
        env["left"] = left if env["left"] is None else min(int(env["left"]), left)
        env["bottom"] = bottom if env["bottom"] is None else max(int(env["bottom"]), bottom)
        env["right"] = right if env["right"] is None else max(int(env["right"]), right)
        env["boxes"] = int(env["boxes"]) + 1
        boxes_count += 1

    groups_json = json.dumps(
        {
            "min_match_score": float(min_score),
            "target_image_width": int(target_w or 0),
            "target_image_height": int(target_h or 0),
            "grouping": "quadrant_marker_sugiu",
            "groups": groups,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if any(int(groups[key]["boxes"]) <= 0 for key in order):
        line16 = ""
        source = "lr_marker_quadrant_envelope_missing_groups"
    else:
        line16 = "".join(
            f"{int(groups[key]['top'])}|{int(groups[key]['left'])}|{int(groups[key]['bottom'])}|{int(groups[key]['right'])}|"
            "1|0:0.000000:0:0:0:0:0|0|;"
            for key in order
        )
        source = "lr_marker_quadrant_orientation_envelopes_pending_thresholds"
    row["line_16_rect_orientation"] = line16
    row["line_16_source"] = source
    row["line_16_marker_boxes_count"] = int(boxes_count)
    row["line_16_groups_json"] = groups_json
    summary["pipeline_row"] = row


def _refresh_lr_marker_evidence_from_csv(summary: Dict[str, Any], run_dir: Path) -> None:
    if not isinstance(summary, dict):
        return
    csv_path = run_dir / "pipeline_output" / "lr_marker_per_image_predictions.csv"
    if not csv_path.is_file():
        return
    try:
        evidence = _runner_build_lr_marker_per_image_evidence(
            run_dir=run_dir,
            pipeline_output=run_dir / "pipeline_output",
        )
    except Exception:
        return
    if not isinstance(evidence, dict) or not bool(evidence.get("available", False)):
        return
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        rec = {}
    rec["lr_marker_per_image"] = evidence
    summary["recognition_evidence"] = rec
    _recompute_line16_from_lr_marker_evidence(summary, evidence)


def _pipeline_output_dir_for_summary(summary: Dict[str, Any], run_dir: Path) -> Path:
    path_txt = str(summary.get("pipeline_output_dir", "") or "").strip() if isinstance(summary, dict) else ""
    if path_txt:
        p = Path(path_txt).expanduser()
        if not p.is_absolute():
            p = run_dir / p
        if p.is_dir():
            return p
    return run_dir / "pipeline_output"


def _row_as_str_dict(summary: Dict[str, Any]) -> Dict[str, str]:
    row = summary.get("pipeline_row", {}) if isinstance(summary, dict) else {}
    if not isinstance(row, dict):
        row = {}
    return {str(k): str(v) for k, v in row.items()}


def _input_copy_folder_for_summary(summary: Dict[str, Any], run_dir: Path) -> Path:
    for key in ("input_copy_folder", "input_reference_folder", "input_folder"):
        txt = str(summary.get(key, "") or "").strip() if isinstance(summary, dict) else ""
        if not txt:
            continue
        p = Path(txt).expanduser()
        if not p.is_absolute():
            p = run_dir / p
        if p.exists():
            return p
    candidate = run_dir / "input"
    return candidate if candidate.exists() else run_dir


def _recompute_run_targets(
    *,
    run_dir: Path,
    summary: Dict[str, Any],
    targets: List[str],
    payload: Dict[str, Any],
    python_bin: str,
) -> Dict[str, Any]:
    allowed = {
        "rect_red",
        "rect_per_image",
        "rect_compare",
        "su_giu",
        "su_giu_split",
        "lr_marker",
        "line16",
        "lt",
        "lt_split",
        "line13",
    }
    normalized: List[str] = []
    for target in targets:
        t = str(target or "").strip().lower()
        if t == "orientation":
            for item in ("su_giu", "lr_marker"):
                if item not in normalized:
                    normalized.append(item)
            continue
        if t == "rect":
            for item in ("rect_red", "rect_per_image", "rect_compare"):
                if item not in normalized:
                    normalized.append(item)
            continue
        if t == "all_evidence":
            for item in ("rect_red", "rect_per_image", "rect_compare", "su_giu", "lr_marker", "lt", "line13"):
                if item not in normalized:
                    normalized.append(item)
            continue
        if t in allowed and t not in normalized:
            normalized.append(t)
    if not normalized:
        raise RuntimeError("nessun target ricalcolabile selezionato")

    pipeline_output = _pipeline_output_dir_for_summary(summary, run_dir)
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        rec = {}
    model = rec.get("model_evidence", {})
    if not isinstance(model, dict):
        model = {}
    row_obj = summary.get("pipeline_row", {})
    if not isinstance(row_obj, dict):
        row_obj = {}
    results: Dict[str, Any] = {}
    completed: List[str] = []
    errors: Dict[str, str] = {}
    generate_images = not bool(payload.get("no_generated_images", False))
    no_split_symlinks = bool(payload.get("no_split_symlinks", False))

    def _remember_error(target: str, exc: Exception) -> None:
        errors[target] = str(exc)

    if "rect_red" in normalized:
        try:
            out = _build_rect_red_recompute_payload(
                run_dir=run_dir,
                summary=summary,
                margin_pct=_safe_float(payload.get("margin_pct"), 5.0),
                bright_thr=_safe_float(payload.get("bright_thr"), 70.0),
            )
            if isinstance(out.get("rect_red_evidence"), dict):
                rec["rect_red"] = out["rect_red_evidence"]
            if out.get("line11_text"):
                row_obj["line_11_rect_echo"] = str(out.get("line11_text", ""))
            for key_out, key_row in (
                ("line11_top", "line_11_top"),
                ("line11_left", "line_11_left"),
                ("line11_bottom", "line_11_bottom"),
                ("line11_right", "line_11_right"),
            ):
                if str(out.get(key_out, "") or "").strip():
                    row_obj[key_row] = str(out.get(key_out, ""))
            row_obj["line_11_method"] = "segment_top_red_rect"
            row_obj["line_11_rect_red_margin_pct"] = f"{_safe_float(out.get('margin_pct'), 0.0):.4f}"
            row_obj["line_11_rect_red_winner_group"] = str(out.get("line11_winner_group", "") or "")
            results["rect_red"] = out
            completed.append("rect_red")
        except Exception as exc:
            _remember_error("rect_red", exc)

    if "rect_per_image" in normalized:
        try:
            image_root = _input_copy_folder_for_summary(summary, run_dir)
            images = _runner_collect_acquisition_images(image_root)
            rect_per = _runner_build_rect_per_image_evidence(
                run_dir=run_dir,
                row={str(k): str(v) for k, v in row_obj.items()},
                unique_images=images,
                acquisition_images=images,
                batch_size=_safe_int(payload.get("batch_size"), 32),
                rotation_deg_clockwise=_safe_int(row_obj.get("rotation_deg_clockwise"), 0),
                python_bin=python_bin,
                generate_images=generate_images,
            )
            rec["rect_per_image"] = rect_per
            results["rect_per_image"] = {"available": bool(rect_per.get("available", False)), "items": rect_per.get("images_total", 0)}
            completed.append("rect_per_image")
        except Exception as exc:
            _remember_error("rect_per_image", exc)

    if "rect_compare" in normalized:
        try:
            compare_json = run_dir / "analysis" / "rect_vendor_compare" / "rect_vendor_compare.json"
            if bool(payload.get("force", False)) and compare_json.is_file():
                compare_json.unlink()
            if compare_json.is_file() and not bool(payload.get("force", False)):
                compare = json.loads(compare_json.read_text(encoding="utf-8"))
                if isinstance(compare, dict):
                    compare["cached"] = True
                else:
                    compare = {}
            else:
                compare = _build_rect_vendor_compare_payload(
                    run_dir=run_dir,
                    summary=summary,
                    python_bin=python_bin,
                )
                compare["cached"] = False
                compare_json.parent.mkdir(parents=True, exist_ok=True)
                compare_json.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
            results["rect_compare"] = compare
            completed.append("rect_compare")
        except Exception as exc:
            _remember_error("rect_compare", exc)

    if "su_giu" in normalized or "su_giu_split" in normalized:
        try:
            su_giu = _runner_build_su_giu_per_image_evidence(run_dir=run_dir, pipeline_output=pipeline_output)
            rec["su_giu_per_image"] = su_giu
            if (not no_split_symlinks) and (bool(su_giu.get("available", False))):
                rec["su_giu_split_folders"] = _runner_build_su_giu_split_folders(
                    run_dir=run_dir,
                    su_giu_per_image_evidence=su_giu,
                )
            results["su_giu"] = {"available": bool(su_giu.get("available", False)), "items": su_giu.get("images_total", 0)}
            completed.append("su_giu")
        except Exception as exc:
            _remember_error("su_giu", exc)

    if "lr_marker" in normalized or "line16" in normalized:
        try:
            lr_marker = _runner_build_lr_marker_per_image_evidence(run_dir=run_dir, pipeline_output=pipeline_output)
            rec["lr_marker_per_image"] = lr_marker
            summary["recognition_evidence"] = rec
            summary["pipeline_row"] = row_obj
            _recompute_line16_from_lr_marker_evidence(summary, lr_marker)
            row_obj = summary.get("pipeline_row", row_obj) if isinstance(summary.get("pipeline_row"), dict) else row_obj
            results["lr_marker"] = {
                "available": bool(lr_marker.get("available", False)),
                "items": lr_marker.get("images_total", 0),
                "quadrant_counts": lr_marker.get("quadrant_counts", {}),
                "line16": row_obj.get("line_16_rect_orientation", ""),
            }
            completed.append("lr_marker")
        except Exception as exc:
            _remember_error("lr_marker", exc)

    if "lt" in normalized or "lt_split" in normalized:
        try:
            lt = _runner_build_lt_per_image_evidence(run_dir=run_dir, pipeline_output=pipeline_output)
            rec["lt_per_image"] = lt
            if (not no_split_symlinks) and bool(lt.get("available", False)):
                rec["lt_split_folders"] = _runner_build_lt_split_folders(
                    run_dir=run_dir,
                    lt_per_image_evidence=lt,
                )
            results["lt"] = {"available": bool(lt.get("available", False)), "items": lt.get("images_total", 0)}
            completed.append("lt")
        except Exception as exc:
            _remember_error("lt", exc)

    summary["recognition_evidence"] = rec
    summary["pipeline_row"] = row_obj
    if "line13" in normalized:
        try:
            _ensure_line13_template_crop(summary, run_dir)
            results["line13"] = {"available": True}
            completed.append("line13")
        except Exception as exc:
            _remember_error("line13", exc)

    recompute_log = summary.get("manual_recompute_log", [])
    if not isinstance(recompute_log, list):
        recompute_log = []
    recompute_log.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "targets": normalized,
            "completed": completed,
            "errors": errors,
        }
    )
    summary["manual_recompute_log"] = recompute_log[-50:]
    if errors and not completed:
        raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items()))
    return {"completed": completed, "errors": errors, "results": results}


def _build_rect_vendor_compare_payload(
    *,
    run_dir: Path,
    summary: Dict[str, Any],
    python_bin: str,
) -> Dict[str, Any]:
    row = summary.get("pipeline_row", {})
    if not isinstance(row, dict):
        row = {}
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        rec = {}
    model = rec.get("model_evidence", {})
    if not isinstance(model, dict):
        model = {}

    vendor_name = str(row.get("vendor_predicted", "") or "").strip()
    vendor_conf = _safe_float(row.get("vendor_confidence"), 0.0)
    rect_source = str(row.get("line_11_source", "") or "").strip()
    used_checkpoint = str(row.get("line_11_model_checkpoint", "") or "").strip()
    global_checkpoint = str(model.get("rect_checkpoint_global", "") or "").strip()
    rect_vendor_map = model.get("rect_vendor_map_loaded", {})
    if not isinstance(rect_vendor_map, dict):
        rect_vendor_map = {}
    specific_checkpoint = str(rect_vendor_map.get(vendor_name, "") or "").strip()
    rect_vendor_min_conf = _safe_float(
        model.get("rect_vendor_min_confidence"),
        _safe_float(summary.get("rect_vendor_min_confidence"), 0.7),
    )

    input_copy_folder = Path(str(summary.get("input_copy_folder", "") or "")).expanduser().resolve()
    if not input_copy_folder.is_dir():
        input_copy_folder = (run_dir / "input_ref").resolve()
    if not input_copy_folder.is_dir():
        raise RuntimeError("input_copy_folder non disponibile per confronto rect.")

    acquisition_images = _runner_collect_acquisition_images(input_copy_folder)
    unique_images, _ = _runner_analyze_duplicates(acquisition_images)
    if not acquisition_images:
        raise RuntimeError("Nessuna immagine disponibile per confronto rect.")

    rotation_deg = _safe_int(row.get("rotation_deg_clockwise"), 0)
    batch_size = 32
    compare_root = run_dir / "analysis" / "rect_vendor_compare"
    compare_root.mkdir(parents=True, exist_ok=True)

    models_out: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {
        "vendor_predicted": vendor_name,
        "vendor_confidence": float(vendor_conf),
        "rect_vendor_min_confidence": float(rect_vendor_min_conf),
        "line_11_source": rect_source,
        "used_checkpoint": used_checkpoint,
        "global_checkpoint": global_checkpoint,
        "specific_checkpoint": specific_checkpoint,
        "specific_available": bool(vendor_name and specific_checkpoint),
    }

    def _compute_for_model(model_key: str, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        ckpt = str(checkpoint_path or "").strip()
        if not ckpt:
            return None
        local_root = compare_root / model_key
        local_root.mkdir(parents=True, exist_ok=True)
        row_local = dict(row)
        row_local["line_11_model_checkpoint"] = ckpt
        rect_data = _runner_build_rect_per_image_evidence(
            run_dir=local_root,
            row=row_local,
            unique_images=unique_images,
            acquisition_images=acquisition_images,
            batch_size=batch_size,
            rotation_deg_clockwise=rotation_deg,
            python_bin=python_bin,
        )
        rect_data_remap = _remap_rect_per_image_paths_to_run_root(
            run_root=run_dir,
            local_root=local_root,
            rect_data=rect_data,
        )
        return {
            "checkpoint": ckpt,
            "available": bool(rect_data_remap.get("available", False)),
            "error": str(rect_data_remap.get("error", "") or ""),
            "metrics": _rect_per_image_metrics(rect_data_remap),
            "rect_per_image": rect_data_remap,
        }

    # Reuse existing rect_per_image only if it already used global checkpoint.
    rect_existing = rec.get("rect_per_image", {})
    if not isinstance(rect_existing, dict):
        rect_existing = {}
    existing_ckpt = str(rect_existing.get("model_checkpoint_resolved", "") or "")
    if global_checkpoint and existing_ckpt and Path(existing_ckpt).name == Path(global_checkpoint).name:
        models_out["global"] = {
            "checkpoint": global_checkpoint,
            "available": bool(rect_existing.get("available", False)),
            "error": str(rect_existing.get("error", "") or ""),
            "metrics": _rect_per_image_metrics(rect_existing),
            "rect_per_image": rect_existing,
        }
    elif global_checkpoint:
        computed = _compute_for_model("global", global_checkpoint)
        if computed is not None:
            models_out["global"] = computed

    if specific_checkpoint:
        computed = _compute_for_model("specific", specific_checkpoint)
        if computed is not None:
            models_out["specific"] = computed

    compare_available = "global" in models_out and "specific" in models_out
    if compare_available:
        g_mean = _safe_float(models_out["global"]["metrics"].get("mean_iou"), 0.0)
        s_mean = _safe_float(models_out["specific"]["metrics"].get("mean_iou"), 0.0)
        winner = "specific" if s_mean > g_mean else "global"
        delta = float(s_mean - g_mean)
    else:
        winner = ""
        delta = 0.0

    routing_message = (
        f"Vendor={vendor_name or '-'} conf={vendor_conf:.4f} soglia_specifica={rect_vendor_min_conf:.4f}. "
        f"Source attuale line11={rect_source or '-'}."
    )
    if rect_source == "global_low_vendor_conf":
        routing_message += " Routing su rete globale perché conf vendor sotto soglia."
    elif rect_source == "vendor_specialized":
        routing_message += " Routing su rete specifica vendor."
    elif rect_source == "global":
        routing_message += " Routing su rete globale (nessuna specifica richiesta)."

    return {
        "run_id": run_dir.name,
        "routing": diagnostics,
        "routing_message": routing_message,
        "compare_available": bool(compare_available),
        "winner": winner,
        "delta_mean_iou_specific_minus_global": float(delta),
        "models": models_out,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _build_rect_red_recompute_payload(
    *,
    run_dir: Path,
    summary: Dict[str, Any],
    margin_pct: float,
    bright_thr: float,
) -> Dict[str, Any]:
    row = summary.get("pipeline_row", {})
    if not isinstance(row, dict):
        row = {}
    rec = summary.get("recognition_evidence", {})
    if not isinstance(rec, dict):
        rec = {}
    rect_red_existing = rec.get("rect_red", {})
    if not isinstance(rect_red_existing, dict):
        rect_red_existing = {}

    source_records = rect_red_existing.get("records_source", [])
    if not isinstance(source_records, list) or not source_records:
        raise RuntimeError("records_source rect_red non disponibili nella run corrente.")

    output_w = _safe_int(row.get("line_09_video_x_size"), 0)
    output_h = _safe_int(row.get("line_10_video_y_size"), 0)

    records_for_compute: List[Dict[str, Any]] = []
    for item in source_records:
        if not isinstance(item, dict):
            continue
        image_rel = str(item.get("image_rel", "") or "").strip()
        if not image_rel:
            continue
        image_path = Path(image_rel).expanduser()
        if not image_path.is_absolute():
            image_path = (run_dir / image_rel).resolve()
        if not image_path.is_file():
            continue
        pred_rect_norm = item.get("pred_rect_norm", {})
        if not isinstance(pred_rect_norm, dict):
            pred_rect_norm = {}
        records_for_compute.append(
            {
                "image_id": Path(image_rel).name or image_rel,
                "image_path": image_path.as_posix(),
                "image_width": int(item.get("image_width", 0) or 0),
                "image_height": int(item.get("image_height", 0) or 0),
                "orientation_label": str(item.get("orientation_label", "") or ""),
                "pred_rect_norm": pred_rect_norm,
                "note_text": str(item.get("note_text", "") or ""),
            }
        )

    if not records_for_compute:
        raise RuntimeError("Nessun record valido per ricalcolo rect rosso.")

    compute_out = compute_rect_red_pipeline(
        records=records_for_compute,
        output_width=int(output_w),
        output_height=int(output_h),
        margin_pct=float(margin_pct),
        bright_thr=float(bright_thr),
        detect_segments=True,
    )
    if not bool(compute_out.get("available", False)):
        err = str(compute_out.get("error", "") or "rect_red_compute_failed")
        raise RuntimeError(err)

    folder_key = str(row.get("folder_path", "") or "").strip()
    if not folder_key:
        folder_key = str(summary.get("input_copy_folder", "") or "").strip() or run_dir.name
    folder_name = str(row.get("folder_name", "") or "").strip() or Path(folder_key).name

    wrapped = {
        folder_key: {
            **compute_out,
            "folder_path": folder_key,
            "folder_name": folder_name,
            "line11_method": "segment_top_red_rect",
            "line11_source": str(row.get("line_11_source", "") or ""),
            "line11_model_checkpoint": str(row.get("line_11_model_checkpoint", "") or ""),
            "line11_median_fallback": str(rect_red_existing.get("line11_base", {}).get("text", "") or ""),
            "line11_applied": str(compute_out.get("line11_red", {}).get("text", "") or ""),
        }
    }

    analysis_dir = run_dir / "analysis" / "rect_red_recompute"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    override_json = analysis_dir / f"rect_red_pipeline_margin_{int(round(float(margin_pct) * 100)):04d}.json"
    override_json.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2), encoding="utf-8")

    row_override = dict(row)
    red_obj = compute_out.get("line11_red", {})
    red_tlbr = red_obj.get("rect_tlbr", {}) if isinstance(red_obj, dict) else {}
    red_text = str(red_obj.get("text", "") or "").strip() if isinstance(red_obj, dict) else ""
    if red_text:
        row_override["line_11_rect_echo"] = red_text
    if isinstance(red_tlbr, dict):
        row_override["line_11_top"] = str(int(red_tlbr.get("top", 0) or 0))
        row_override["line_11_left"] = str(int(red_tlbr.get("left", 0) or 0))
        row_override["line_11_bottom"] = str(int(red_tlbr.get("bottom", 0) or 0))
        row_override["line_11_right"] = str(int(red_tlbr.get("right", 0) or 0))
    row_override["line_11_method"] = "segment_top_red_rect"
    row_override["line_11_rect_red_margin_pct"] = f"{float(margin_pct):.4f}"
    row_override["line_11_rect_red_winner_group"] = str(compute_out.get("winner_group", "") or "")

    pipeline_output_dir = Path(str(summary.get("pipeline_output_dir", "") or "")).expanduser()
    if not pipeline_output_dir.is_dir():
        pipeline_output_dir = run_dir / "pipeline_output"

    rect_red_evidence = _runner_build_rect_red_evidence(
        run_dir=run_dir,
        pipeline_output=pipeline_output_dir,
        pipeline_summary={"rect_red_pipeline_json": override_json.as_posix()},
        row={str(k): str(v) for k, v in row_override.items()},
    )

    return {
        "run_id": run_dir.name,
        "margin_pct": float(margin_pct),
        "bright_thr": float(bright_thr),
        "line11_text": str(red_text or row_override.get("line_11_rect_echo", "")),
        "line11_top": str(row_override.get("line_11_top", "")),
        "line11_left": str(row_override.get("line_11_left", "")),
        "line11_bottom": str(row_override.get("line_11_bottom", "")),
        "line11_right": str(row_override.get("line_11_right", "")),
        "line11_winner_group": str(row_override.get("line_11_rect_red_winner_group", "")),
        "rect_red_evidence": rect_red_evidence,
        "override_json": override_json.as_posix(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _default_annotations_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    checks = summary.get("checks", [])
    out: Dict[str, Any] = {
        "run_flag": "",
        "run_note": "",
        "steps": {},
        "lr_marker_reviews": {},
    }
    if isinstance(checks, list):
        for step in checks:
            key = str((step or {}).get("step", "")).strip()
            if not key:
                continue
            out["steps"][key] = {"flag": "", "comment": "", "correction": ""}
    return out


def _default_input_browse_root() -> Path:
    candidate = DEFAULT_INPUT_BROWSE_ROOT
    if candidate.is_dir():
        return candidate
    return Path.home().resolve()


def _pick_folder_native(initial_dir: Optional[Path] = None, timeout_sec: float = 8.0) -> Dict[str, Any]:
    start_dir = initial_dir.expanduser().resolve() if initial_dir else _default_input_browse_root()
    if not start_dir.is_dir():
        start_dir = _default_input_browse_root()

    if sys.platform == "darwin":
        try:
            safe_start = start_dir.as_posix().replace('"', '\\"')
            try:
                subprocess.run(
                    ["osascript", "-e", 'tell application "Finder" to activate'],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
            except Exception:
                pass
            script = (
                f'set dflt to POSIX file "{safe_start}"\n'
                'set chosenFolder to choose folder with prompt "Seleziona cartella input" default location dflt\n'
                "POSIX path of chosenFolder"
            )
            cp = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_sec)),
            )
            if cp.returncode == 0:
                selected = (cp.stdout or "").strip()
                if selected:
                    picked = Path(selected).expanduser().resolve()
                    if picked.is_dir():
                        return {"ok": True, "path": picked.as_posix(), "method": "osascript"}
                return {"ok": False, "cancelled": True, "error": "Selezione annullata."}
            stderr = (cp.stderr or "").strip()
            if ("-128" in stderr) or ("User canceled" in stderr):
                return {"ok": False, "cancelled": True, "error": "Selezione annullata."}
            return {
                "ok": False,
                "cancelled": False,
                "error": "Impossibile aprire il selettore cartelle.",
                "details": {"osascript": stderr or f"returncode={cp.returncode}"},
            }
        except Exception as exc:
            return {
                "ok": False,
                "cancelled": False,
                "error": "Impossibile aprire il selettore cartelle.",
                "details": {"osascript": str(exc)},
            }

    tkinter_error = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(
            title="Seleziona cartella input",
            initialdir=start_dir.as_posix(),
            mustexist=True,
        )
        try:
            root.destroy()
        except Exception:
            pass
        selected = str(selected or "").strip()
        if selected:
            picked = Path(selected).expanduser().resolve()
            if picked.is_dir():
                return {"ok": True, "path": picked.as_posix(), "method": "tkinter"}
        return {"ok": False, "cancelled": True, "error": "Selezione annullata."}
    except Exception as exc:
        tkinter_error = str(exc)

    return {
        "ok": False,
        "cancelled": False,
        "error": "Selettore cartelle non disponibile su questo sistema.",
        "details": {"tkinter": tkinter_error},
    }


def _list_directories(path: Optional[Path], max_entries: int = 400) -> Dict[str, Any]:
    target = path.expanduser().resolve() if path else _default_input_browse_root()
    if target.is_file():
        target = target.parent
    if not target.exists():
        target = _default_input_browse_root()
    if not target.is_dir():
        raise RuntimeError(f"Path non valida: {target}")

    parent = target.parent if target.parent != target else None
    dirs: List[Dict[str, str]] = []
    try:
        for entry in target.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            dirs.append({"name": name, "path": entry.resolve().as_posix()})
    except PermissionError:
        raise RuntimeError(f"Permesso negato: {target}")

    dirs.sort(key=lambda d: d["name"].lower())
    truncated = False
    if len(dirs) > int(max_entries):
        dirs = dirs[: int(max_entries)]
        truncated = True

    return {
        "path": target.as_posix(),
        "parent": parent.as_posix() if parent else "",
        "dirs": dirs,
        "truncated": truncated,
    }


def _open_path_in_file_manager(path: Path) -> Dict[str, Any]:
    target = path.expanduser().resolve()
    if target.is_file():
        target = target.parent
    if not target.exists() or not target.is_dir():
        return {"ok": False, "error": f"Path non valida: {target.as_posix()}"}
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", target.as_posix()])
            method = "open"
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(target)])
            method = "explorer"
        else:
            subprocess.Popen(["xdg-open", target.as_posix()])
            method = "xdg-open"
        return {"ok": True, "path": target.as_posix(), "method": method}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": target.as_posix()}


def _slug_for_run_id(value: str, max_len: int = 70) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "folder")[: max(8, int(max_len))]


def _folder_has_acquisition_images(folder: Path) -> bool:
    try:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                return True
    except Exception:
        return False
    return False


def _run_safe_runner(
    *,
    store: WorkbenchStore,
    run_id: str,
    input_folder: Path,
    python_bin: str,
    batch_size: int,
    sample_per_folder: int,
    rotation_max_samples: int,
    low_confidence_policy: str,
    vendor_min_confidence: float,
    probe_min_confidence: float,
    lr_marker_template_policy: str,
    no_generated_images: bool = False,
    no_split_symlinks: bool = False,
    excluded_images_rel: Optional[List[str]] = None,
) -> None:
    run_dir = store.run_dir(run_id)
    safe_runner = (SCRIPT_DIR / "run_pipeline_single_folder_safe.py").resolve()
    if not safe_runner.is_file():
        state = store.load_state(run_id)
        state["status"] = "failed"
        state["stage"] = "failed"
        state["error"] = f"safe_runner_not_found:{safe_runner.as_posix()}"
        store.save_state(run_id, state)
        store.update_run_record(run_id, {"status": "failed", "run_dir": run_dir.as_posix()})
        return
    cmd = [
        python_bin,
        safe_runner.as_posix(),
        "--input-folder",
        input_folder.as_posix(),
        "--work-root",
        store.runs_root.as_posix(),
        "--run-name",
        run_id,
        "--batch-size",
        str(batch_size),
        "--sample-per-folder",
        str(sample_per_folder),
        "--rotation-max-samples",
        str(rotation_max_samples),
        "--low-confidence-policy",
        low_confidence_policy,
        "--vendor-min-confidence",
        str(vendor_min_confidence),
        "--probe-min-confidence",
        str(probe_min_confidence),
        "--lr-marker-template-policy",
        str(lr_marker_template_policy),
    ]
    excluded_images = [str(x).strip() for x in (excluded_images_rel or []) if str(x).strip()]
    if excluded_images:
        exclude_path = run_dir / "excluded_images.json"
        exclude_path.write_text(
            json.dumps(
                {
                    "input_folder": input_folder.as_posix(),
                    "excluded_images_rel": excluded_images,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd.extend(["--exclude-images-file", exclude_path.as_posix()])
    if bool(no_generated_images):
        cmd.append("--no-generated-images")
    if bool(no_split_symlinks):
        cmd.append("--no-split-symlinks")

    state = store.load_state(run_id)
    state.update(
        {
            "status": "running",
            "stage": "queued",
            "run_id": run_id,
            "run_dir": run_dir.as_posix(),
            "input_folder": input_folder.as_posix(),
            "excluded_images_rel": excluded_images,
            "excluded_images_count": int(len(excluded_images)),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(cmd),
            "events": [],
            "logs": [],
            "completed_stages": [],
            "error": "",
        }
    )
    store.save_state(run_id, state)
    store.update_run_record(
        run_id,
        {"status": "running", "run_dir": run_dir.as_posix(), "started_at": str(state.get("started_at", "") or "")},
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT.as_posix(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        try:
            store.append_log(run_id, line)
        except Exception as exc:
            print(f"[workbench] append_log failed for {run_id}: {exc}", flush=True)
        if line.startswith("##EVENT "):
            payload_txt = line[len("##EVENT ") :].strip()
            try:
                event_obj = json.loads(payload_txt)
                if isinstance(event_obj, dict):
                    try:
                        store.append_event(run_id, event_obj)
                    except Exception as exc:
                        print(f"[workbench] append_event failed for {run_id}: {exc}", flush=True)
            except Exception:
                pass

    proc.wait()

    state = store.load_state(run_id)
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    if int(proc.returncode) == 0:
        state["status"] = "completed"
        state["stage"] = "run_completed"
        store.update_run_record(
            run_id,
            {"status": "completed", "run_dir": run_dir.as_posix(), "finished_at": str(state.get("finished_at", "") or "")},
        )
    else:
        state["status"] = "failed"
        state["stage"] = "failed"
        state["error"] = f"safe_runner_returncode_{proc.returncode}"
        store.update_run_record(
            run_id,
            {"status": "failed", "run_dir": run_dir.as_posix(), "finished_at": str(state.get("finished_at", "") or "")},
        )
    try:
        store.save_state(run_id, state)
    except Exception as exc:
        print(f"[workbench] final save_state failed for {run_id}: {exc}", flush=True)


def _mark_stale_running_runs_failed(store: WorkbenchStore) -> None:
    stale_statuses = {"running", "queued"}
    for rec in store.list_runs():
        run_id = str(rec.get("run_id", "")).strip()
        rec_status = str(rec.get("status", "")).strip()
        if not run_id or rec_status not in stale_statuses:
            continue
        state = store.load_state(run_id)
        if state:
            st_status = str(state.get("status", "")).strip()
            if st_status in stale_statuses:
                state["status"] = "failed"
                state["stage"] = "failed"
                if not str(state.get("error", "")).strip():
                    state["error"] = "interrupted_server_restart"
                store.save_state(run_id, state)
        store.update_run_record(run_id, {"status": "failed"})


def create_app(data_root: Path, python_bin: str, models_metrics_csv: Optional[Path] = None) -> Flask:
    store = WorkbenchStore(data_root=data_root)
    store.prune_missing_run_records()
    _mark_stale_running_runs_failed(store)
    app = Flask(__name__)
    batch_lock = threading.Lock()
    batch_state: Dict[str, Any] = {
        "running": False,
        "status": "idle",
        "root": "",
        "total": 0,
        "created": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current": "",
        "started_at": "",
        "finished_at": "",
        "last_error": "",
    }

    @app.after_request
    def _disable_cache(resp):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    fss_template_path_obj = _pick_fss_template_path()
    fss_template_path = fss_template_path_obj.as_posix() if fss_template_path_obj else ""
    fss_template_lines = _load_fss_template_lines(fss_template_path_obj, max_lines=40)
    models_metrics_csv_obj = _pick_models_metrics_csv_path(models_metrics_csv)
    encoding_struct_catalog = _load_encoding_struct_fss_catalog(_pick_encoding_struct_xlsx_path())
    exclusions_memory_path = data_root / "folder_image_exclusions.json"

    def _read_exclusions_memory() -> Dict[str, Any]:
        if not exclusions_memory_path.is_file():
            return {"version": 1, "folders": {}}
        try:
            payload = json.loads(exclusions_memory_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "folders": {}}
        if not isinstance(payload, dict):
            return {"version": 1, "folders": {}}
        folders = payload.get("folders", {})
        if not isinstance(folders, dict):
            folders = {}
        payload["folders"] = folders
        payload["version"] = int(payload.get("version", 1) or 1)
        return payload

    def _write_exclusions_memory(payload: Dict[str, Any]) -> None:
        exclusions_memory_path.parent.mkdir(parents=True, exist_ok=True)
        exclusions_memory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _folder_memory_key(folder: Path) -> str:
        try:
            return folder.expanduser().resolve().as_posix()
        except Exception:
            return folder.expanduser().as_posix()

    def _normalize_exclusion_items(raw_items: Any, input_folder: Path) -> List[str]:
        if isinstance(raw_items, str):
            txt = raw_items.strip()
            if not txt:
                items: Any = []
            else:
                try:
                    items = json.loads(txt)
                except Exception:
                    items = [line.strip() for line in txt.splitlines() if line.strip()]
        else:
            items = raw_items
        if isinstance(items, dict):
            items = items.get("excluded_images_rel", items.get("excluded", []))
        if not isinstance(items, list):
            return []
        out: List[str] = []
        seen = set()
        for item in items:
            rel_txt = str(item or "").strip()
            if not rel_txt:
                continue
            try:
                p = Path(rel_txt).expanduser()
                if p.is_absolute():
                    rel_txt = p.resolve().relative_to(input_folder).as_posix()
            except Exception:
                rel_txt = rel_txt.replace("\\", "/")
            rel_txt = rel_txt.replace("\\", "/").lstrip("/")
            if not rel_txt or rel_txt.startswith("../") or "/../" in rel_txt:
                continue
            if rel_txt not in seen:
                seen.add(rel_txt)
                out.append(rel_txt)
        return out

    def _load_saved_exclusions(input_folder: Path) -> List[str]:
        memory = _read_exclusions_memory()
        entry = memory.get("folders", {}).get(_folder_memory_key(input_folder), {})
        if not isinstance(entry, dict):
            return []
        return _normalize_exclusion_items(entry.get("excluded_images_rel", []), input_folder)

    def _save_exclusions(input_folder: Path, excluded_images_rel: List[str]) -> Dict[str, Any]:
        memory = _read_exclusions_memory()
        folders = memory.setdefault("folders", {})
        if not isinstance(folders, dict):
            folders = {}
            memory["folders"] = folders
        key = _folder_memory_key(input_folder)
        folders[key] = {
            "input_folder": key,
            "excluded_images_rel": _normalize_exclusion_items(excluded_images_rel, input_folder),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusions_memory(memory)
        return folders[key]

    def _start_run_common(payload: Dict[str, Any]) -> Dict[str, Any]:
        input_folder_raw = str(payload.get("input_folder", "")).strip()
        if not input_folder_raw:
            return {"ok": False, "error": "input_folder mancante"}
        input_folder = Path(input_folder_raw).expanduser().resolve()
        if not input_folder.is_dir():
            return {"ok": False, "error": f"input_folder non valida: {input_folder}"}
        lr_marker_template_policy = str(payload.get("lr_marker_template_policy", "historical_best_then_derived")).strip()
        if lr_marker_template_policy not in {"historical_best_then_derived", "historical_best", "derived_folder"}:
            lr_marker_template_policy = "historical_best_then_derived"
        excluded_json_raw = str(payload.get("excluded_images_rel_json", "") or "").strip()
        if "excluded_images_rel" in payload:
            excluded_images_rel = _normalize_exclusion_items(payload.get("excluded_images_rel"), input_folder)
        elif "excluded_images_rel_json" in payload and excluded_json_raw not in {"", "__use_saved__"}:
            excluded_images_rel = _normalize_exclusion_items(payload.get("excluded_images_rel_json"), input_folder)
        else:
            excluded_images_rel = _load_saved_exclusions(input_folder)
        if bool(payload.get("remember_exclusions", True)):
            _save_exclusions(input_folder, excluded_images_rel)

        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = store.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)

        record = {
            "run_id": run_id,
            "input_folder": input_folder.as_posix(),
            "excluded_images_count": int(len(excluded_images_rel)),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": run_dir.as_posix(),
        }
        store.create_run_record(record)

        init_state = {
            "run_id": run_id,
            "input_folder": input_folder.as_posix(),
            "excluded_images_rel": excluded_images_rel,
            "excluded_images_count": int(len(excluded_images_rel)),
            "run_dir": run_dir.as_posix(),
            "status": "queued",
            "stage": "queued",
            "created_at": record["created_at"],
            "logs": ["run queued"],
            "events": [],
            "completed_stages": [],
            "error": "",
        }
        store.save_state(run_id, init_state)

        t = threading.Thread(
            target=_run_safe_runner,
            kwargs={
                "store": store,
                "run_id": run_id,
                "input_folder": input_folder,
                "python_bin": python_bin,
                "batch_size": _safe_int(payload.get("batch_size"), 32),
                "sample_per_folder": _safe_int(payload.get("sample_per_folder"), 80),
                "rotation_max_samples": _safe_int(payload.get("rotation_max_samples"), 24),
                "low_confidence_policy": str(payload.get("low_confidence_policy", "review")),
                "vendor_min_confidence": _safe_float(payload.get("vendor_min_confidence"), 0.50),
                "probe_min_confidence": _safe_float(payload.get("probe_min_confidence"), 0.50),
                "lr_marker_template_policy": lr_marker_template_policy,
                "no_generated_images": bool(payload.get("no_generated_images", False)),
                "no_split_symlinks": bool(payload.get("no_split_symlinks", False)),
                "excluded_images_rel": excluded_images_rel,
            },
            daemon=True,
        )
        t.start()
        return {"ok": True, "run_id": run_id, "input_folder": input_folder.as_posix()}

    def _bulk_acquisition_worker(payload: Dict[str, Any]) -> None:
        root_raw = str(payload.get("root", "") or DEFAULT_BULK_ACQUISITION_ROOT.as_posix()).strip()
        root = Path(root_raw).expanduser().resolve()
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if not root.is_dir():
                raise RuntimeError(f"root batch non valida: {root.as_posix()}")
            with batch_lock:
                batch_state.update(
                    {
                        "running": True,
                        "status": "clearing",
                        "root": root.as_posix(),
                        "total": 0,
                        "created": 0,
                        "completed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "current": "",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "finished_at": "",
                        "last_error": "",
                    }
                )
            removed = store.clear_all_runs(delete_files=True)
            with batch_lock:
                batch_state["status"] = f"scanning | vecchie run eliminate={removed}"

            candidates = [p for p in sorted(root.iterdir(), key=lambda x: x.name.lower()) if p.is_dir()]
            folders: List[Path] = []
            skipped = 0
            for folder in candidates:
                if _folder_has_acquisition_images(folder):
                    folders.append(folder)
                else:
                    skipped += 1
                with batch_lock:
                    batch_state["total"] = len(folders)
                    batch_state["skipped"] = skipped
                    batch_state["current"] = folder.as_posix()

            with batch_lock:
                batch_state["status"] = "running"
                batch_state["total"] = len(folders)
                batch_state["skipped"] = skipped
                batch_state["current"] = ""

            for idx, folder in enumerate(folders, start=1):
                # Batch leggero: evita anche il template derivato di cartella, che sarebbe un nuovo PNG.
                lr_policy = "historical_best"
                run_id = f"batch_{batch_stamp}_{idx:03d}_{_slug_for_run_id(folder.name)}"
                run_dir = store.run_dir(run_id)
                run_dir.mkdir(parents=True, exist_ok=False)
                created_at = datetime.now(timezone.utc).isoformat()
                store.create_run_record(
                    {
                        "run_id": run_id,
                        "input_folder": folder.as_posix(),
                        "status": "queued",
                        "created_at": created_at,
                        "run_dir": run_dir.as_posix(),
                        "batch_root": root.as_posix(),
                    }
                )
                store.save_state(
                    run_id,
                    {
                        "run_id": run_id,
                        "input_folder": folder.as_posix(),
                        "run_dir": run_dir.as_posix(),
                        "status": "queued",
                        "stage": "queued",
                        "created_at": created_at,
                        "logs": [f"batch queued {idx}/{len(folders)}"],
                        "events": [],
                        "completed_stages": [],
                        "error": "",
                    },
                )
                with batch_lock:
                    batch_state["created"] = idx
                    batch_state["current"] = folder.as_posix()
                _run_safe_runner(
                    store=store,
                    run_id=run_id,
                    input_folder=folder,
                    python_bin=python_bin,
                    batch_size=_safe_int(payload.get("batch_size"), 32),
                    sample_per_folder=_safe_int(payload.get("sample_per_folder"), 80),
                    rotation_max_samples=_safe_int(payload.get("rotation_max_samples"), 24),
                    low_confidence_policy=str(payload.get("low_confidence_policy", "review")),
                    vendor_min_confidence=_safe_float(payload.get("vendor_min_confidence"), 0.50),
                    probe_min_confidence=_safe_float(payload.get("probe_min_confidence"), 0.50),
                    lr_marker_template_policy=lr_policy,
                    no_generated_images=True,
                    no_split_symlinks=True,
                    excluded_images_rel=_load_saved_exclusions(folder),
                )
                run_state = store.load_state(run_id)
                with batch_lock:
                    if str(run_state.get("status", "")) == "completed":
                        batch_state["completed"] = int(batch_state.get("completed", 0) or 0) + 1
                    else:
                        batch_state["failed"] = int(batch_state.get("failed", 0) or 0) + 1
            with batch_lock:
                batch_state["running"] = False
                batch_state["status"] = "completed"
                batch_state["current"] = ""
                batch_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            with batch_lock:
                batch_state["running"] = False
                batch_state["status"] = "failed"
                batch_state["last_error"] = str(exc)
                batch_state["finished_at"] = datetime.now(timezone.utc).isoformat()

    @app.get("/")
    def index():
        initial_input = str(request.args.get("input_folder", "")).strip()
        initial_notice = str(request.args.get("notice", "")).strip()
        initial_runs = store.list_runs()
        initial_selected_run = str(request.args.get("selected_run", "")).strip()
        fallback_run_info = "Nessuna run selezionata."
        fallback_kpi_status = "-"
        fallback_kpi_stage = "-"
        fallback_kpi_updated = "-"
        fallback_log_text = ""
        fallback_checks: List[Dict[str, Any]] = []
        fallback_checks_status = "Nessun dato check disponibile."
        fallback_evidence_status = "Nessuna evidenza disponibile."
        fallback_raw_images_rel: List[str] = []
        fallback_vendor_probe_samples: List[str] = []
        fallback_rect_overlay_rel = ""
        fallback_line13_overlay_rel = ""
        fallback_line14_overlay_rel = ""
        fallback_line13_value = ""
        fallback_line13_source = ""
        fallback_line13_support = ""
        fallback_line14_value = ""
        fallback_line14_source = ""
        fallback_line14_support = ""
        fallback_model_vendor_checkpoint = ""
        fallback_model_probe_checkpoint = ""
        fallback_model_rect_global = ""
        fallback_model_rect_used = ""
        fallback_vendor_predicted_name = ""
        fallback_vendor_confidence = ""
        fallback_vendor_source_hint = ""
        fallback_vendor_ocr_used = False
        fallback_vendor_ocr_comment = ""
        fallback_vendor_ocr_elements_text = ""
        fallback_probe_label = ""
        fallback_probe_confidence = ""
        fallback_probe_source_hint = ""
        fallback_probe_ocr_used = False
        fallback_probe_ocr_comment = ""
        fallback_probe_ocr_elements_text = ""
        fallback_duplicate_examples: List[Dict[str, str]] = []
        fallback_line11_rect = ""
        fallback_line90_status = ""
        fallback_line91_reason = ""
        fallback_rotation_deg_clockwise = ""
        fallback_raw_images_count = 0
        fallback_duplicates_removed_count = 0
        fallback_rect_confidence = ""
        fallback_rect_confidence_method = ""
        fallback_rect_confidence_note = ""
        fallback_rect_source = ""
        fallback_rect_images_used = ""
        fallback_rect_size_px = ""
        fallback_rect_area_ratio = ""
        fallback_rect_aspect_ratio = ""
        auto_refresh_seconds = 0
        if initial_selected_run:
            st = store.load_state(initial_selected_run)
            if st:
                fallback_run_info = (
                    f"run_id={st.get('run_id', '-')}"
                    f" | input={st.get('input_folder', '-')}"
                    f" | run_dir={st.get('run_dir', '-')}"
                )
                fallback_kpi_status = str(st.get("status", "-"))
                fallback_kpi_stage = str(st.get("stage", "-"))
                fallback_kpi_updated = str(st.get("updated_at", "-"))
                if fallback_kpi_status in {"running", "queued"}:
                    auto_refresh_seconds = 3
                logs = st.get("logs", [])
                if isinstance(logs, list):
                    fallback_log_text = "\n".join([str(x) for x in logs[-300:]])
            summary_path = store.run_dir(initial_selected_run) / "step_checks.json"
            if summary_path.is_file():
                try:
                    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary_obj = {}
                _ensure_line13_template_crop(summary_obj, store.run_dir(initial_selected_run))
                checks_obj = summary_obj.get("checks", [])
                if isinstance(checks_obj, list):
                    fallback_checks = [x for x in checks_obj if isinstance(x, dict)]
                if fallback_checks:
                    fallback_checks_status = (
                        f"Step disponibili: {len(fallback_checks)} | "
                        f"run_dir: {summary_obj.get('run_dir', '-')}"
                    )
                raw_count = int(summary_obj.get("raw_images_count", 0) or 0)
                dup_count = int(summary_obj.get("duplicates_removed_count_recomputed", 0) or 0)
                fallback_raw_images_count = raw_count
                fallback_duplicates_removed_count = dup_count
                fallback_evidence_status = f"Immagini raw: {raw_count}"
                row_obj = summary_obj.get("pipeline_row", {})
                if isinstance(row_obj, dict):
                    fallback_line11_rect = str(row_obj.get("line_11_rect_echo", "") or "")
                    fallback_line90_status = str(row_obj.get("line_90_status_prediction", "") or "")
                    fallback_line91_reason = str(row_obj.get("line_91_reason", "") or "")
                    rect_metrics = _rect_metrics_from_row(row_obj)
                    fallback_rect_confidence = f"{float(rect_metrics.get('confidence', 0.0)):.4f}"
                    fallback_rect_confidence_method = str(rect_metrics.get("method", "") or "")
                    fallback_rect_confidence_note = str(rect_metrics.get("note", "") or "")
                    fallback_rect_source = str(rect_metrics.get("source", "") or "")
                    fallback_rect_images_used = str(rect_metrics.get("images_used", "") or "")
                    fallback_rect_size_px = (
                        f"{int(rect_metrics.get('width', 0))} x {int(rect_metrics.get('height', 0))}"
                    )
                    fallback_rect_area_ratio = f"{float(rect_metrics.get('area_ratio', 0.0)):.4f}"
                    fallback_rect_aspect_ratio = f"{float(rect_metrics.get('aspect_ratio', 0.0)):.4f}"
                raw_rel_obj = summary_obj.get("raw_images_rel", [])
                if isinstance(raw_rel_obj, list):
                    fallback_raw_images_rel = [str(x) for x in raw_rel_obj if str(x)]
                dup_examples_obj = summary_obj.get("duplicate_examples", [])
                if isinstance(dup_examples_obj, list):
                    for item in dup_examples_obj[:12]:
                        if not isinstance(item, dict):
                            continue
                        kept_rel = str(item.get("kept_rel", "") or "").strip()
                        removed_rel = str(item.get("removed_rel", "") or "").strip()
                        if not kept_rel or not removed_rel:
                            continue
                        fallback_duplicate_examples.append(
                            {
                                "kept_rel": kept_rel,
                                "removed_rel": removed_rel,
                                "sha1_prefix": str(item.get("sha1_prefix", "") or "").strip(),
                                "size_bytes": str(item.get("size_bytes", "") or "").strip(),
                            }
                        )
                decision_obj = summary_obj.get("decision_evidence", {})
                if isinstance(decision_obj, dict):
                    vendor_obj = decision_obj.get("vendor", {})
                    if isinstance(vendor_obj, dict):
                        fallback_vendor_predicted_name = str(vendor_obj.get("predicted_name", "") or "")
                        fallback_vendor_confidence = str(vendor_obj.get("confidence", "") or "")
                        fallback_vendor_source_hint = str(vendor_obj.get("source_hint", "") or "")
                        fallback_vendor_ocr_used = bool(vendor_obj.get("ocr_used", False))
                        fallback_vendor_ocr_comment = str(vendor_obj.get("ocr_comment", "") or "")
                        vendor_ocr_elements = vendor_obj.get("ocr_elements", {})
                        if isinstance(vendor_ocr_elements, dict):
                            fallback_vendor_ocr_elements_text = " | ".join(
                                [
                                    f"{str(k)}: {str(v)}"
                                    for (k, v) in vendor_ocr_elements.items()
                                    if str(k) and str(v)
                                ]
                            )
                    probe_obj = decision_obj.get("probe", {})
                    if isinstance(probe_obj, dict):
                        probe_id = str(probe_obj.get("predicted_id", "") or "").strip()
                        probe_name = str(probe_obj.get("predicted_name", "") or "").strip()
                        fallback_probe_label = " - ".join([x for x in [probe_id, probe_name] if x])
                        fallback_probe_confidence = str(probe_obj.get("confidence", "") or "")
                        fallback_probe_source_hint = str(probe_obj.get("source_hint", "") or "")
                        fallback_probe_ocr_used = bool(probe_obj.get("ocr_used", False))
                        fallback_probe_ocr_comment = str(probe_obj.get("ocr_comment", "") or "")
                        probe_ocr_elements = probe_obj.get("ocr_elements", {})
                        if isinstance(probe_ocr_elements, dict):
                            fallback_probe_ocr_elements_text = " | ".join(
                                [
                                    f"{str(k)}: {str(v)}"
                                    for (k, v) in probe_ocr_elements.items()
                                    if str(k) and str(v)
                                ]
                            )
                rotation_obj = summary_obj.get("rotation_evidence", {})
                if isinstance(rotation_obj, dict):
                    fallback_rotation_deg_clockwise = str(rotation_obj.get("rotation_deg_clockwise", "") or "")
                rec_obj = summary_obj.get("recognition_evidence", {})
                if isinstance(rec_obj, dict):
                    vps = rec_obj.get("vendor_probe_samples_rel", [])
                    if isinstance(vps, list):
                        fallback_vendor_probe_samples = [str(x) for x in vps if str(x)]
                    ro = rec_obj.get("rect_overlay", {})
                    if isinstance(ro, dict):
                        fallback_rect_overlay_rel = str(ro.get("overlay_rel", "") or "")
                    l13 = rec_obj.get("line13_template", {})
                    if isinstance(l13, dict):
                        fallback_line13_overlay_rel = str(l13.get("overlay_rel", "") or "")
                        fallback_line13_value = str(l13.get("value", "") or "")
                        fallback_line13_source = str(l13.get("source", "") or "")
                        fallback_line13_support = str(l13.get("support", "") or "")
                    l14 = rec_obj.get("line14_template", {})
                    if isinstance(l14, dict):
                        fallback_line14_overlay_rel = str(l14.get("overlay_rel", "") or "")
                        fallback_line14_value = str(l14.get("value", "") or "")
                        fallback_line14_source = str(l14.get("source", "") or "")
                        fallback_line14_support = str(l14.get("support", "") or "")
                    me = rec_obj.get("model_evidence", {})
                    if isinstance(me, dict):
                        fallback_model_vendor_checkpoint = str(me.get("vendor_checkpoint", "") or "")
                        fallback_model_probe_checkpoint = str(me.get("probe_checkpoint", "") or "")
                        fallback_model_rect_global = str(me.get("rect_checkpoint_global", "") or "")
                        fallback_model_rect_used = str(me.get("rect_checkpoint_used", "") or "")
        return render_template_string(
            HTML_PAGE,
            initial_input=initial_input,
            initial_notice=initial_notice,
            initial_selected_run=initial_selected_run,
            initial_runs=initial_runs,
            fallback_run_info=fallback_run_info,
            fallback_kpi_status=fallback_kpi_status,
            fallback_kpi_stage=fallback_kpi_stage,
            fallback_kpi_updated=fallback_kpi_updated,
            fallback_log_text=fallback_log_text,
            fallback_checks=fallback_checks,
            fallback_checks_status=fallback_checks_status,
            fallback_evidence_status=fallback_evidence_status,
            fallback_raw_images_rel=fallback_raw_images_rel,
            fallback_vendor_probe_samples=fallback_vendor_probe_samples,
            fallback_rect_overlay_rel=fallback_rect_overlay_rel,
            fallback_line13_overlay_rel=fallback_line13_overlay_rel,
            fallback_line14_overlay_rel=fallback_line14_overlay_rel,
            fallback_line13_value=fallback_line13_value,
            fallback_line13_source=fallback_line13_source,
            fallback_line13_support=fallback_line13_support,
            fallback_line14_value=fallback_line14_value,
            fallback_line14_source=fallback_line14_source,
            fallback_line14_support=fallback_line14_support,
            fallback_model_vendor_checkpoint=fallback_model_vendor_checkpoint,
            fallback_model_probe_checkpoint=fallback_model_probe_checkpoint,
            fallback_model_rect_global=fallback_model_rect_global,
            fallback_model_rect_used=fallback_model_rect_used,
            fallback_vendor_predicted_name=fallback_vendor_predicted_name,
            fallback_vendor_confidence=fallback_vendor_confidence,
            fallback_vendor_source_hint=fallback_vendor_source_hint,
            fallback_vendor_ocr_used=fallback_vendor_ocr_used,
            fallback_vendor_ocr_comment=fallback_vendor_ocr_comment,
            fallback_vendor_ocr_elements_text=fallback_vendor_ocr_elements_text,
            fallback_probe_label=fallback_probe_label,
            fallback_probe_confidence=fallback_probe_confidence,
            fallback_probe_source_hint=fallback_probe_source_hint,
            fallback_probe_ocr_used=fallback_probe_ocr_used,
            fallback_probe_ocr_comment=fallback_probe_ocr_comment,
            fallback_probe_ocr_elements_text=fallback_probe_ocr_elements_text,
            fallback_duplicate_examples=fallback_duplicate_examples,
            fallback_line11_rect=fallback_line11_rect,
            fallback_line90_status=fallback_line90_status,
            fallback_line91_reason=fallback_line91_reason,
            fallback_rotation_deg_clockwise=fallback_rotation_deg_clockwise,
            fallback_raw_images_count=fallback_raw_images_count,
            fallback_duplicates_removed_count=fallback_duplicates_removed_count,
            fallback_rect_confidence=fallback_rect_confidence,
            fallback_rect_confidence_method=fallback_rect_confidence_method,
            fallback_rect_confidence_note=fallback_rect_confidence_note,
            fallback_rect_source=fallback_rect_source,
            fallback_rect_images_used=fallback_rect_images_used,
            fallback_rect_size_px=fallback_rect_size_px,
            fallback_rect_area_ratio=fallback_rect_area_ratio,
            fallback_rect_aspect_ratio=fallback_rect_aspect_ratio,
            history_runs_root=store.runs_root.as_posix(),
            browse_default_root=_default_input_browse_root().as_posix(),
            fss_template_path=fss_template_path,
            fss_template_lines=fss_template_lines,
            models_metrics_csv_path=models_metrics_csv_obj.as_posix() if models_metrics_csv_obj else "",
            encoding_struct_catalog=encoding_struct_catalog,
            auto_refresh_seconds=auto_refresh_seconds,
        )

    @app.get("/pick-folder")
    def pick_folder():
        initial_raw = str(request.args.get("input_folder", "")).strip()
        initial_dir = Path(initial_raw).expanduser() if initial_raw else None
        timeout_sec = _safe_float(request.args.get("timeout_sec"), 600.0)
        picked = _pick_folder_native(initial_dir=initial_dir, timeout_sec=max(1.0, timeout_sec))
        input_folder_out = initial_raw
        notice = ""
        if picked.get("ok") and picked.get("path"):
            input_folder_out = str(picked.get("path", "")).strip()
            method = str(picked.get("method", "native")).strip() or "native"
            notice = f"Cartella selezionata ({method})."
        elif picked.get("cancelled"):
            notice = "Selezione cartella annullata."
        else:
            err = str(picked.get("error", "Selettore cartelle non disponibile.")).strip()
            notice = f"Errore apertura file manager: {err}"
        return redirect(url_for("index", input_folder=input_folder_out, notice=notice))

    @app.get("/browse")
    def browse():
        path_raw = str(request.args.get("path", "")).strip()
        path_obj = Path(path_raw).expanduser() if path_raw else None
        error_msg = ""
        try:
            listing = _list_directories(path_obj)
        except Exception as exc:
            error_msg = str(exc)
            listing = _list_directories(None)
        return render_template_string(
            BROWSE_PAGE,
            current_path=str(listing.get("path", "")),
            parent_path=str(listing.get("parent", "")),
            dirs=listing.get("dirs", []),
            error_msg=error_msg,
        )

    @app.post("/api/dialog/select_folder")
    def api_select_folder():
        payload = request.get_json(silent=True) or {}
        initial_raw = str(payload.get("initial_dir", "")).strip()
        initial_dir = Path(initial_raw).expanduser() if initial_raw else None
        timeout_sec = _safe_float(payload.get("timeout_sec"), 8.0)
        picked = _pick_folder_native(initial_dir=initial_dir, timeout_sec=timeout_sec)
        if picked.get("ok"):
            return jsonify(picked)
        if picked.get("cancelled"):
            return jsonify(picked)
        return jsonify(picked), 500

    @app.post("/api/fs/list_dirs")
    def api_fs_list_dirs():
        payload = request.get_json(silent=True) or {}
        path_raw = str(payload.get("path", "")).strip()
        path_obj = Path(path_raw).expanduser() if path_raw else None
        try:
            listing = _list_directories(path_obj)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(listing)

    @app.post("/api/fs/open_in_finder")
    def api_fs_open_in_finder():
        payload = request.get_json(silent=True) or {}
        path_raw = str(payload.get("path", "")).strip()
        if not path_raw:
            return jsonify({"error": "path mancante"}), 400
        out = _open_path_in_file_manager(Path(path_raw))
        if not out.get("ok"):
            return jsonify({"error": str(out.get("error", "impossibile aprire il file manager"))}), 400
        return jsonify(out)

    @app.post("/api/input/images")
    def api_input_images():
        payload = request.get_json(silent=True) or {}
        folder_raw = str(payload.get("input_folder", "")).strip()
        if not folder_raw:
            return jsonify({"error": "input_folder mancante"}), 400
        folder = Path(folder_raw).expanduser().resolve()
        if not folder.is_dir():
            return jsonify({"error": f"input_folder non valida: {folder.as_posix()}"}), 400
        saved = set(_load_saved_exclusions(folder))
        images = []
        try:
            paths = _runner_collect_acquisition_images(folder)
        except Exception as exc:
            return jsonify({"error": f"errore lettura immagini: {exc}"}), 500
        for idx, path in enumerate(paths):
            try:
                rel = path.relative_to(folder).as_posix()
            except Exception:
                rel = path.name
            try:
                size = int(path.stat().st_size)
            except Exception:
                size = 0
            images.append(
                {
                    "idx": idx,
                    "rel": rel,
                    "name": path.name,
                    "size_bytes": size,
                    "excluded": rel in saved,
                }
            )
        return jsonify(
            {
                "ok": True,
                "input_folder": folder.as_posix(),
                "images": images,
                "count": int(len(images)),
                "excluded_images_rel": sorted(saved),
                "memory_path": exclusions_memory_path.as_posix(),
            }
        )

    @app.post("/api/input/exclusions")
    def api_input_exclusions_save():
        payload = request.get_json(silent=True) or {}
        folder_raw = str(payload.get("input_folder", "")).strip()
        if not folder_raw:
            return jsonify({"error": "input_folder mancante"}), 400
        folder = Path(folder_raw).expanduser().resolve()
        if not folder.is_dir():
            return jsonify({"error": f"input_folder non valida: {folder.as_posix()}"}), 400
        excluded = _normalize_exclusion_items(payload.get("excluded_images_rel", []), folder)
        saved = _save_exclusions(folder, excluded)
        return jsonify(
            {
                "ok": True,
                "input_folder": folder.as_posix(),
                "excluded_images_rel": saved.get("excluded_images_rel", []),
                "excluded_count": len(saved.get("excluded_images_rel", [])),
                "memory_path": exclusions_memory_path.as_posix(),
            }
        )

    @app.get("/api/runs")
    def api_runs():
        store.prune_missing_run_records()
        runs = store.list_runs()
        return jsonify({"runs": runs})

    @app.get("/api/batch/acquisition_elaboration/status")
    def api_batch_acquisition_status():
        with batch_lock:
            return jsonify(dict(batch_state))

    @app.post("/api/batch/acquisition_elaboration/start")
    def api_batch_acquisition_start():
        payload = request.get_json(silent=True) or {}
        with batch_lock:
            if bool(batch_state.get("running", False)):
                return jsonify({"error": "batch già in esecuzione", "status": dict(batch_state)}), 409
            batch_state.update({"running": True, "status": "queued", "last_error": ""})
        t = threading.Thread(target=_bulk_acquisition_worker, args=(payload,), daemon=True)
        t.start()
        with batch_lock:
            return jsonify({"ok": True, "status": dict(batch_state)})

    @app.get("/api/models/metrics")
    def api_models_metrics():
        payload = _load_models_metrics_registry(models_metrics_csv_obj)
        return jsonify(payload)

    @app.post("/api/runs/start")
    def api_run_start():
        payload = request.get_json(silent=True) or {}
        out = _start_run_common(payload)
        if not out.get("ok"):
            return jsonify({"error": out.get("error", "Errore avvio run")}), 400
        return jsonify({"ok": True, "run_id": out.get("run_id", "")})

    @app.post("/runs/start_form")
    def runs_start_form():
        payload = {k: v for k, v in request.form.items()}
        out = _start_run_common(payload)
        input_folder = str(payload.get("input_folder", "")).strip()
        if not out.get("ok"):
            notice = str(out.get("error", "Errore avvio run")).strip()
            return redirect(url_for("index", input_folder=input_folder, notice=notice))
        run_id = str(out.get("run_id", "")).strip()
        notice = f"Run avviata: {run_id}"
        return redirect(url_for("index", input_folder=input_folder, selected_run=run_id, notice=notice))

    @app.post("/runs/rerun_form")
    def runs_rerun_form():
        run_id_src = str(request.form.get("run_id", "")).strip()
        input_folder = str(request.form.get("input_folder", "")).strip()
        if not input_folder:
            notice = f"Rerun non avviata: input_folder mancante per run {run_id_src or '-'}."
            return redirect(url_for("index", notice=notice))
        payload = {"input_folder": input_folder}
        out = _start_run_common(payload)
        if not out.get("ok"):
            notice = f"Errore rerun ({run_id_src or '-'}) : {str(out.get('error', 'Errore avvio run')).strip()}"
            return redirect(url_for("index", input_folder=input_folder, notice=notice))
        run_id_new = str(out.get("run_id", "")).strip()
        notice = f"Rerun avviata da {run_id_src or '-'}: {run_id_new}"
        return redirect(url_for("index", input_folder=input_folder, selected_run=run_id_new, notice=notice))

    @app.post("/api/runs/<run_id>/delete")
    def api_run_delete(run_id: str):
        payload = request.get_json(silent=True) or {}
        delete_files = bool(payload.get("delete_files", True))
        run_dir = store.run_dir(run_id)
        removed = store.delete_run_record(run_id)
        if not removed:
            return jsonify({"error": f"run non trovata: {run_id}"}), 404
        files_deleted = False
        if delete_files and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=False)
            files_deleted = True
        return jsonify({"ok": True, "run_id": run_id, "files_deleted": files_deleted})

    @app.post("/runs/delete_form")
    def runs_delete_form():
        run_id = str(request.form.get("run_id", "")).strip()
        input_folder = str(request.form.get("input_folder", "")).strip()
        if not run_id:
            return redirect(url_for("index", input_folder=input_folder, notice="run_id mancante"))
        run_dir = store.run_dir(run_id)
        removed = store.delete_run_record(run_id)
        if not removed:
            return redirect(url_for("index", input_folder=input_folder, notice=f"Run non trovata: {run_id}"))
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=False)
        return redirect(url_for("index", input_folder=input_folder, notice=f"Run eliminata: {run_id}"))

    @app.get("/api/runs/<run_id>/status")
    def api_run_status(run_id: str):
        state = store.load_state(run_id)
        if not state:
            return jsonify({"error": f"run non trovata: {run_id}"}), 404
        logs = state.get("logs", [])
        if not isinstance(logs, list):
            logs = []
        state["logs_tail"] = logs[-300:]
        return jsonify(state)

    @app.get("/api/runs/<run_id>/data")
    def api_run_data(run_id: str):
        run_dir = store.run_dir(run_id)
        summary_path = run_dir / "step_checks.json"
        summary: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        _ensure_line13_template_crop(summary, run_dir)
        _refresh_lr_marker_evidence_from_csv(summary, run_dir)
        annotations_path = store.annotations_path(run_id)
        if annotations_path.exists():
            try:
                annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
            except Exception:
                annotations = _default_annotations_from_summary(summary)
        else:
            annotations = _default_annotations_from_summary(summary)
        return jsonify(
            {
                "run_id": run_id,
                "run_dir": run_dir.as_posix(),
                "summary": summary,
                "annotations": annotations,
                "fss_template_path": fss_template_path,
                "fss_template_lines": fss_template_lines,
            }
        )

    @app.post("/api/runs/<run_id>/recompute")
    def api_run_recompute(run_id: str):
        run_dir = store.run_dir(run_id)
        if not run_dir.exists():
            return jsonify({"error": f"run non trovata: {run_id}"}), 404
        summary_path = run_dir / "step_checks.json"
        if not summary_path.exists():
            return jsonify({"error": "step_checks.json non trovato"}), 404
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return jsonify({"error": f"step_checks.json non valido: {exc}"}), 500
        if not isinstance(summary, dict):
            return jsonify({"error": "summary non valido"}), 500

        payload = request.get_json(silent=True) or {}
        raw_targets = payload.get("targets", [])
        if isinstance(raw_targets, str):
            raw_targets = [part.strip() for part in raw_targets.split(",") if part.strip()]
        if not isinstance(raw_targets, list):
            return jsonify({"error": "targets deve essere una lista"}), 400
        targets = [str(x or "").strip() for x in raw_targets if str(x or "").strip()]
        if not targets:
            return jsonify({"error": "nessun target selezionato"}), 400

        backup_dir = run_dir / "analysis" / "manual_recompute"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"step_checks_before_{stamp}.json"
        try:
            if not backup_path.exists():
                backup_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        started = time.time()
        try:
            recompute = _recompute_run_targets(
                run_dir=run_dir,
                summary=summary,
                targets=targets,
                payload=payload if isinstance(payload, dict) else {},
                python_bin=python_bin,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        summary["manual_recompute_updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            return jsonify({"error": f"ricalcolo completato ma salvataggio summary fallito: {exc}"}), 500
        state = store.load_state(run_id)
        logs = state.get("logs", [])
        if not isinstance(logs, list):
            logs = []
        logs.append(
            f"[{datetime.now(timezone.utc).isoformat()}] manual_recompute targets={','.join(targets)} completed={','.join(recompute.get('completed', []))}"
        )
        state["logs"] = logs[-1200:]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            store.save_state(run_id, state)
        except Exception:
            pass
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "requested_targets": targets,
                "completed_targets": recompute.get("completed", []),
                "errors": recompute.get("errors", {}),
                "results": recompute.get("results", {}),
                "rect_vendor_compare": recompute.get("results", {}).get("rect_compare", {}),
                "elapsed_sec": float(max(0.0, time.time() - started)),
                "backup_path": backup_path.as_posix(),
                "summary": summary,
            }
        )

    @app.post("/api/runs/<run_id>/rect_vendor_compare")
    def api_run_rect_vendor_compare(run_id: str):
        run_dir = store.run_dir(run_id)
        if not run_dir.exists():
            return jsonify({"error": f"run non trovata: {run_id}"}), 404

        summary_path = run_dir / "step_checks.json"
        if not summary_path.exists():
            return jsonify({"error": "step_checks.json non trovato"}), 404
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return jsonify({"error": f"step_checks.json non valido: {exc}"}), 500

        payload = request.get_json(silent=True) or {}
        force = bool(payload.get("force", False))
        compare_json = run_dir / "analysis" / "rect_vendor_compare" / "rect_vendor_compare.json"

        if (not force) and compare_json.is_file():
            try:
                cached_obj = json.loads(compare_json.read_text(encoding="utf-8"))
                if isinstance(cached_obj, dict):
                    cached_obj["cached"] = True
                    return jsonify(cached_obj)
            except Exception:
                pass

        started = time.time()
        try:
            out = _build_rect_vendor_compare_payload(
                run_dir=run_dir,
                summary=summary if isinstance(summary, dict) else {},
                python_bin=python_bin,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        out["elapsed_sec"] = float(max(0.0, time.time() - started))
        out["cached"] = False

        compare_json.parent.mkdir(parents=True, exist_ok=True)
        compare_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify(out)

    @app.post("/api/runs/<run_id>/rect_red_recompute")
    def api_run_rect_red_recompute(run_id: str):
        run_dir = store.run_dir(run_id)
        if not run_dir.exists():
            return jsonify({"error": f"run non trovata: {run_id}"}), 404

        summary_path = run_dir / "step_checks.json"
        if not summary_path.exists():
            return jsonify({"error": "step_checks.json non trovato"}), 404
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return jsonify({"error": f"step_checks.json non valido: {exc}"}), 500
        if not isinstance(summary, dict):
            return jsonify({"error": "summary non valido"}), 500

        payload = request.get_json(silent=True) or {}
        margin_pct = _safe_float(payload.get("margin_pct"), 5.0)
        bright_thr = _safe_float(payload.get("bright_thr"), 70.0)
        if margin_pct < 0.0 or margin_pct > 30.0:
            return jsonify({"error": "margin_pct deve essere tra 0 e 30"}), 400
        if bright_thr < 0.0 or bright_thr > 255.0:
            return jsonify({"error": "bright_thr deve essere tra 0 e 255"}), 400

        try:
            out = _build_rect_red_recompute_payload(
                run_dir=run_dir,
                summary=summary,
                margin_pct=float(margin_pct),
                bright_thr=float(bright_thr),
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(out)

    @app.post("/api/runs/<run_id>/annotations")
    def api_run_annotations(run_id: str):
        run_dir = store.run_dir(run_id)
        if not run_dir.exists():
            return jsonify({"error": f"run non trovata: {run_id}"}), 404
        payload = request.get_json(silent=True) or {}
        annotations_path = store.annotations_path(run_id)
        annotations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "path": annotations_path.as_posix()})

    @app.get("/api/runs/<run_id>/preview")
    def api_run_preview(run_id: str):
        run_dir = store.run_dir(run_id).resolve()
        state = store.load_state(run_id)
        rel = str(request.args.get("path", "")).strip()
        if not rel:
            return jsonify({"error": "path mancante"}), 400
        p = Path(rel).expanduser()
        if p.is_absolute():
            target = p.resolve()
        else:
            target = (run_dir / p).resolve()

        allowed_roots = _preview_allowed_roots_for_run(run_dir, state)
        allowed = False
        for root in allowed_roots:
            try:
                target.relative_to(root)
                allowed = True
                break
            except Exception:
                continue
        if not allowed:
            return jsonify({"error": "path non valido"}), 403
        if not target.is_file():
            return jsonify({"error": "file non trovato"}), 404
        return send_file(target)

    @app.get("/api/runs/<run_id>/open_review_html")
    def api_open_review_html(run_id: str):
        run_dir = store.run_dir(run_id)
        summary_path = run_dir / "step_checks.json"
        if not summary_path.exists():
            return jsonify({"error": "summary non trovato"}), 404
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            html_path = Path(str(summary.get("review_html", "")))
        except Exception:
            return jsonify({"error": "summary non valido"}), 500
        if not html_path.is_file():
            return jsonify({"error": "review html non trovato"}), 404
        return send_file(html_path)

    return app


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified pipeline workbench web UI.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8780)
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/runs/workbench_ui",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default="python3",
        help="Python usato per lanciare run_pipeline_single_folder_safe.py",
    )
    p.add_argument(
        "--models-metrics-csv",
        type=Path,
        default=REPO_ROOT / "artifacts/60_metadata/reti_metriche_status_2026-04-01.csv",
        help="CSV con registro reti+metriche da mostrare in home.",
    )
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port deve essere in 1..65535")

    app = create_app(
        data_root=args.data_root.expanduser().resolve(),
        python_bin=args.python_bin,
        models_metrics_csv=args.models_metrics_csv,
    )
    print(
        f"Pipeline AIESIBuilder running on http://{args.host}:{args.port} | data_root={args.data_root}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
