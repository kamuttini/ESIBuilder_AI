#!/usr/bin/env python3
"""Side-by-side UX comparison tool for the pipeline workbench."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/runs/workbench_ui"
DEFAULT_FEEDBACK_ROOT = REPO_ROOT / "artifacts/70_ux_feedback/pipeline_compare_tool"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_workbench_web import (  # noqa: E402
    WorkbenchStore,
    _load_models_metrics_registry,
    create_app as create_workbench_app,
)


def _is_port_open(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _safe_run_id(run_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_id or "").strip())


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _as_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out


def _as_text(value: Any, fallback: str = "-") -> str:
    txt = str(value if value is not None else "").strip()
    return txt or fallback


def _format_conf(value: Any) -> str:
    n = _as_float(value)
    if n is None:
        return "-"
    if 0 <= n <= 1:
        return f"{n * 100:.1f}%"
    return f"{n:.4f}"


def _format_metric_value(value: Any) -> str:
    n = _as_float(value)
    if n is None:
        return "-"
    if abs(n) >= 1000:
        return f"{n:,.0f}".replace(",", ".")
    return f"{n:.6f}"


def _resolve_artifact_path(path_text: str) -> Optional[Path]:
    txt = str(path_text or "").strip()
    if not txt or txt == "-":
        return None
    path = Path(txt).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _model_artifact_date(metrics_path: str, checkpoint_path: str) -> Dict[str, Any]:
    for source_label, path_text in [("metriche", metrics_path), ("checkpoint", checkpoint_path)]:
        path = _resolve_artifact_path(path_text)
        if path is None or not path.exists():
            continue
        try:
            ts = path.stat().st_mtime
        except OSError:
            continue
        dt = datetime.fromtimestamp(ts).astimezone()
        return {
            "model_date": dt.strftime("%d/%m/%Y"),
            "model_datetime": dt.strftime("%d/%m/%Y %H:%M"),
            "model_date_source": source_label,
            "model_date_ts": ts,
        }
    return {
        "model_date": "-",
        "model_datetime": "",
        "model_date_source": "",
        "model_date_ts": None,
    }


def _status_kind(status: Any) -> str:
    txt = str(status or "").strip().lower()
    if txt in {"ok", "completed", "done"}:
        return "ok"
    if txt in {"error", "failed", "blocked"}:
        return "blocked"
    if txt in {"running", "queued"}:
        return "running"
    return "review"


def _count_statuses(checks: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"ok": 0, "review": 0, "blocked": 0, "running": 0}
    for check in checks:
        kind = _status_kind(check.get("status"))
        out[kind] = int(out.get(kind, 0)) + 1
    return out


def _step_result(step: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    key = str(step.get("step") or "").strip()
    status = str(step.get("status") or "review").strip()
    kind = _status_kind(status)
    title_map = {
        "deduplicazione": "Deduplicazione",
        "rotazione": "Rotazione",
        "vendor": "Vendor",
        "rect": "Rettangolo ecografico",
        "orientamento_su_giu_per_frame": "Orientamento SU/GIU",
        "orientamento_lr_marker_classico": "LR marker",
        "orientamento_lr_marker_bundle": "LR marker bundle",
        "orientamento_lt_per_frame": "Classificazione L/T",
        "probe": "Probe",
        "probe_type_router_line4": "Probe type riga 04",
        "pipeline_status": "Stato pipeline",
    }
    title = title_map.get(key, key.replace("_", " ").strip().title() or "Step")

    result = "-"
    confidence = ""
    source = ""
    reason = ""

    if key == "deduplicazione":
        raw = step.get("raw_images", row.get("raw_images_count"))
        unique = step.get("unique_images")
        removed = step.get("duplicates_removed")
        result = f"{_as_text(unique)} uniche / {_as_text(raw)} raw"
        confidence = f"rimossi {_as_text(removed, '0')}"
        source = "hash frame"
    elif key == "rotazione":
        result = f"{_as_text(step.get('rotation_deg_clockwise'))} deg"
        confidence = f"vote {_format_conf(step.get('vote_ratio'))}"
        source = _as_text(step.get("rotation_source"), "")
        reason = _as_text(step.get("decision_reason"), "")
    elif key == "vendor":
        result = _as_text(step.get("vendor_predicted", row.get("vendor_predicted")))
        confidence = _format_conf(step.get("vendor_confidence", row.get("vendor_confidence")))
        source = _as_text(row.get("vendor_source"), "cnn")
        reason = _as_text(row.get("vendor_decision_reason"), "")
    elif key == "rect":
        result = _as_text(step.get("line_11_rect_echo", row.get("line_11_rect_echo")))
        confidence = _format_conf(row.get("line_11_confidence_proxy"))
        source = _as_text(step.get("line_11_method", row.get("line_11_method")), "")
        reason = _as_text(row.get("line_11_source"), "")
    elif key == "orientamento_su_giu_per_frame":
        result = f"{_as_text(row.get('su_giu_majority_label'))} ({_as_text(step.get('images_predicted'))} frame)"
        confidence = _format_conf(row.get("su_giu_mean_confidence"))
        source = _as_text(step.get("source", row.get("su_giu_source")), "")
    elif key in {"orientamento_lr_marker_classico", "orientamento_lr_marker_bundle"}:
        result = _as_text(step.get("best_label", row.get("lr_marker_best_label")))
        confidence = _format_conf(step.get("best_score", row.get("lr_marker_best_score")))
        source = _as_text(step.get("source", row.get("lr_marker_source")), "")
    elif key == "orientamento_lt_per_frame":
        result = _as_text(step.get("majority_label", row.get("lt_majority_label")))
        confidence = _format_conf(step.get("mean_confidence", row.get("lt_mean_confidence")))
        source = _as_text(step.get("source", row.get("lt_source")), "")
    elif key == "probe":
        probe_id = _as_text(step.get("line_03_id_probe", row.get("line_03_id_probe")), "")
        probe_name = _as_text(step.get("line_03_probe_name", row.get("line_03_probe_name")), "")
        result = " - ".join(x for x in [probe_id, probe_name] if x) or "-"
        confidence = _format_conf(step.get("line_03_probe_confidence", row.get("line_03_probe_confidence")))
        source = _as_text(row.get("probe_source"), "cnn")
        reason = _as_text(row.get("probe_decision_reason"), "")
    elif key == "probe_type_router_line4":
        result = _as_text(step.get("line_04_probe_type", row.get("line_04_probe_type")))
        confidence = _as_text(step.get("line_04_probe_type_strategy", row.get("line_04_probe_type_strategy")), "")
        source = _as_text(step.get("line_04_probe_type_source", row.get("line_04_probe_type_source")), "")
    elif key == "pipeline_status":
        reasons = _as_text(step.get("review_reasons", row.get("review_reasons")), "")
        result = "Review richiesta" if reasons else ("Pronta" if kind == "ok" else status)
        confidence = reasons
        source = "quality gate"
        reason = reasons
    else:
        payload = {k: v for k, v in step.items() if k not in {"step", "status"}}
        result = " | ".join(f"{k}={v}" for k, v in list(payload.items())[:2]) or "-"

    return {
        "id": key,
        "title": title,
        "status": status or "review",
        "kind": kind,
        "result": result,
        "confidence": confidence,
        "source": source,
        "reason": reason,
        "raw": step,
    }


def _missing_fss_lines(row: Dict[str, Any]) -> List[Dict[str, str]]:
    required = [
        ("02", "ID ecografo", "line_02_id_echo"),
        ("03", "ID probe", "line_03_id_probe"),
        ("04", "Tipo probe", "line_04_probe_type"),
        ("11", "Rect echo", "line_11_rect_echo"),
        ("13", "Template nome ecografo", "line_13_rect_name_echo"),
        ("14", "Template nome probe", "line_14_rect_name_probe"),
        ("16", "Orientation checks", "line_16_rect_orientation"),
    ]
    out = []
    for line_no, label, key in required:
        value = str(row.get(key, "") or "").strip()
        out.append(
            {
                "line": line_no,
                "label": label,
                "key": key,
                "value": value or "-",
                "status": "ok" if value else "missing",
            }
        )
    return out


def _build_digest(store: WorkbenchStore, run_id: str) -> Dict[str, Any]:
    run_id = str(run_id or "").strip()
    run_dir = store.run_dir(run_id) if run_id else Path()
    summary = _read_json(run_dir / "step_checks.json", {}) if run_id else {}
    state = store.load_state(run_id) if run_id else {}
    run_record = {}
    for row in store.list_runs():
        if str(row.get("run_id", "")) == run_id:
            run_record = row
            break

    checks = summary.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    checks = [c for c in checks if isinstance(c, dict)]

    row = summary.get("pipeline_row", {})
    if not isinstance(row, dict):
        row = {}

    status_counts = _count_statuses(checks)
    review_reasons = str(row.get("review_reasons") or "").strip()
    row_status = str(row.get("status") or "").strip().lower()
    has_blockers = bool(status_counts.get("blocked", 0))
    has_review = bool(status_counts.get("review", 0)) or bool(review_reasons) or row_status == "review"
    verdict = "Bloccata" if has_blockers else ("Review richiesta" if has_review else "Pronta")
    verdict_kind = "blocked" if has_blockers else ("review" if has_review else "ok")

    action_queue: List[Dict[str, str]] = []
    for step in checks:
        if _status_kind(step.get("status")) != "ok":
            action_queue.append(
                {
                    "target": str(step.get("step") or ""),
                    "title": str(step.get("step") or "step").replace("_", " "),
                    "reason": str(step.get("review_reasons") or step.get("error") or step.get("status") or "da controllare"),
                }
            )
    for item in _missing_fss_lines(row):
        if item["status"] == "missing":
            action_queue.append(
                {
                    "target": item["key"],
                    "title": f"Riga {item['line']} .fss",
                    "reason": f"{item['label']} mancante",
                }
            )

    steps = [_step_result(step, row) for step in checks]
    fss_lines = _missing_fss_lines(row)
    run_label = str(run_record.get("input_folder") or summary.get("input_original_folder") or state.get("input_folder") or "")
    folder_name = Path(run_label).name if run_label else run_id

    return {
        "run_id": run_id,
        "run_dir": run_dir.as_posix() if run_id else "",
        "folder_name": folder_name,
        "input_folder": run_label,
        "status": str(run_record.get("status") or state.get("status") or row.get("status") or "-"),
        "stage": str(state.get("stage") or "-"),
        "updated_at": str(state.get("updated_at") or run_record.get("created_at") or ""),
        "verdict": verdict,
        "verdict_kind": verdict_kind,
        "review_reasons": review_reasons,
        "status_counts": status_counts,
        "action_queue": action_queue,
        "steps": steps,
        "fss_lines": fss_lines,
        "summary": {
            "vendor": _as_text(row.get("vendor_predicted")),
            "vendor_confidence": _format_conf(row.get("vendor_confidence")),
            "probe": " - ".join(
                x
                for x in [
                    _as_text(row.get("line_03_id_probe"), ""),
                    _as_text(row.get("line_03_probe_name"), ""),
                ]
                if x
            )
            or "-",
            "probe_confidence": _format_conf(row.get("line_03_probe_confidence")),
            "probe_type": _as_text(row.get("line_04_probe_type")),
            "rect": _as_text(row.get("line_11_rect_echo")),
            "rect_source": _as_text(row.get("line_11_method", row.get("line_11_source"))),
            "rotation": _as_text(row.get("rotation_deg_clockwise", (summary.get("rotation_evidence") or {}).get("rotation_deg_clockwise"))),
            "su_giu": _as_text(row.get("su_giu_majority_label")),
            "lt": _as_text(row.get("lt_majority_label")),
            "lr": _as_text(row.get("lr_marker_best_label")),
            "raw_images": _as_text(summary.get("raw_images_count")),
            "duplicates_removed": _as_text(summary.get("duplicates_removed_count_recomputed"), "0"),
        },
        "raw_summary": summary,
    }


def _feedback_path(feedback_root: Path, run_id: str) -> Path:
    return feedback_root / f"{_safe_run_id(run_id) or 'no_run'}_feedback.json"


def _load_review_tag(run_dir: Path) -> Dict[str, str]:
    annotations = _read_json(run_dir / "review_annotations.json", {})
    if not isinstance(annotations, dict):
        annotations = {}
    tag = str(annotations.get("run_flag") or "").strip() or "senza_tag"
    note = str(annotations.get("run_note") or "").strip()
    return {"tag": tag, "note": note}


def _tag_kind(tag: str) -> str:
    txt = str(tag or "").strip().lower()
    if txt in {"ok", "approved", "pronta", "validata"}:
        return "ok"
    if txt in {"blocked", "bloccato", "wrong_output"}:
        return "blocked"
    if txt in {"needs_reprocess", "da rieseguire"}:
        return "running"
    return "review" if txt and txt != "senza_tag" else "neutral"


def _build_home_data(store: WorkbenchStore) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    vendor_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    verdict_counts: Dict[str, int] = {}
    records = store.list_runs()

    for record in records:
        run_id = str((record or {}).get("run_id", "")).strip()
        if not run_id:
            continue
        run_dir = store.run_dir(run_id)
        digest = _build_digest(store, run_id)
        summary = digest.get("summary", {}) if isinstance(digest.get("summary"), dict) else {}
        review = _load_review_tag(run_dir)
        vendor = _as_text(summary.get("vendor"), "sconosciuto")
        tag = review["tag"]
        verdict = _as_text(digest.get("verdict"), "sconosciuto")
        vendor_counts[vendor] = int(vendor_counts.get(vendor, 0)) + 1
        tag_counts[tag] = int(tag_counts.get(tag, 0)) + 1
        verdict_counts[verdict] = int(verdict_counts.get(verdict, 0)) + 1

        items.append(
            {
                "run_id": run_id,
                "created_at": str((record or {}).get("created_at") or digest.get("updated_at") or ""),
                "status": str((record or {}).get("status") or digest.get("status") or "-"),
                "verdict": verdict,
                "verdict_kind": str(digest.get("verdict_kind") or "review"),
                "vendor": vendor,
                "tag": tag,
                "tag_kind": _tag_kind(tag),
                "review_note": review["note"],
                "folder_name": str(digest.get("folder_name") or run_id),
                "input_folder": str(digest.get("input_folder") or ""),
                "probe": _as_text(summary.get("probe")),
                "probe_type": _as_text(summary.get("probe_type")),
                "rect": _as_text(summary.get("rect")),
                "review_reasons": str(digest.get("review_reasons") or ""),
                "actions_count": len(digest.get("action_queue", []) if isinstance(digest.get("action_queue"), list) else []),
                "ok_steps": int((digest.get("status_counts") or {}).get("ok", 0)),
                "review_steps": int((digest.get("status_counts") or {}).get("review", 0)),
            }
        )

    def _count_list(counter: Dict[str, int]) -> List[Dict[str, Any]]:
        return [
            {"label": key, "count": counter[key]}
            for key in sorted(counter.keys(), key=lambda k: (-counter[k], k.lower()))
        ]

    return {
        "runs": items,
        "vendors": _count_list(vendor_counts),
        "tags": _count_list(tag_counts),
        "verdicts": _count_list(verdict_counts),
        "summary": {
            "total_runs": len(items),
            "vendors_total": len(vendor_counts),
            "tagged_runs": sum(1 for item in items if item.get("tag") != "senza_tag"),
            "review_runs": sum(1 for item in items if item.get("verdict_kind") == "review"),
            "ready_runs": sum(1 for item in items if item.get("verdict_kind") == "ok"),
            "blocked_runs": sum(1 for item in items if item.get("verdict_kind") == "blocked"),
        },
    }


def _build_models_home_data(models_metrics_csv: Optional[Path]) -> Dict[str, Any]:
    registry = _load_models_metrics_registry(models_metrics_csv)
    rows = registry.get("rows", []) if isinstance(registry, dict) else []
    slim_rows: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "").strip()
        run_name = str(row.get("run_name") or "").strip()
        primary_metric = str(row.get("primary_metric") or "").strip()
        update_reason = str(row.get("update_reason") or "")
        notes = str(row.get("notes") or "")
        legacy_haystack = " ".join([family, run_name, update_reason, notes]).lower()
        if family.lower() == "checkpoint_only":
            continue
        if primary_metric.lower() in {"", "none", "-"}:
            continue
        if row.get("primary_value_num") is None:
            continue
        if not bool(row.get("metrics_exists")):
            continue
        if any(token in legacy_haystack for token in ["legacy", "old", "deprecated", "smoke", "checkpoint_only"]):
            continue
        checkpoint_path = str(row.get("checkpoint_path") or "")
        metrics_path = str(row.get("metrics_path") or "")
        primary_value = row.get("primary_value_num")
        artifact_date = _model_artifact_date(metrics_path, checkpoint_path)
        slim_rows.append(
            {
                "family": family or "-",
                "scope": str(row.get("scope") or "-"),
                "vendor": str(row.get("vendor") or "-"),
                "run_name": run_name or "-",
                "primary_metric": primary_metric or "-",
                "primary_value": primary_value,
                "primary_value_label": _format_metric_value(primary_value),
                "secondary_metrics": str(row.get("secondary_metrics") or ""),
                "test_samples": row.get("test_samples_num"),
                "test_samples_label": _as_text(row.get("test_samples_num"), "-"),
                "update_status": str(row.get("update_status") or "UNKNOWN"),
                "update_reason": update_reason,
                "notes": notes,
                "checkpoint_exists": bool(row.get("checkpoint_exists")),
                "metrics_exists": bool(row.get("metrics_exists")),
                "checkpoint_path": checkpoint_path,
                "metrics_path": metrics_path,
                "checkpoint_name": Path(checkpoint_path).name if checkpoint_path else "-",
                "metrics_name": Path(metrics_path).name if metrics_path else "-",
                **artifact_date,
            }
        )
    return {
        "ok": bool(registry.get("ok")) if isinstance(registry, dict) else False,
        "error": str(registry.get("error") or "") if isinstance(registry, dict) else "registro non disponibile",
        "csv_path": str(registry.get("csv_path") or "") if isinstance(registry, dict) else "",
        "rows": slim_rows,
        "summary": {
            **(registry.get("summary", {}) if isinstance(registry, dict) and isinstance(registry.get("summary"), dict) else {}),
            "visible_rows": len(slim_rows),
            "source_rows": len(rows),
        },
    }


HTML_PAGE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pipeline UX Compare</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --text: #111827;
      --muted: #64748b;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-soft: #d9f3ee;
      --blue: #1d4ed8;
      --blue-soft: #dbeafe;
      --warn: #b45309;
      --warn-soft: #fef3c7;
      --danger: #b91c1c;
      --danger-soft: #fee2e2;
      --ok: #15803d;
      --ok-soft: #dcfce7;
      --shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    button,
    .nav-link {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
    }
    .nav-link {
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }
    .nav-link.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.ghost { background: transparent; }
    button.small { padding: 6px 8px; font-size: 12px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 9px 10px;
      outline: none;
    }
    textarea { resize: vertical; min-height: 88px; line-height: 1.45; }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto minmax(0, 1fr); }
    .topbar {
      display: grid;
      grid-template-columns: minmax(260px, 1.4fr) minmax(300px, 2fr) auto;
      gap: 12px;
      align-items: end;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 20;
      box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
    }
    .brand h1 { margin: 0; font-size: 18px; line-height: 1.1; }
    .brand p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .field label { display: block; color: var(--muted); font-size: 12px; margin: 0 0 4px; font-weight: 700; }
    body[data-view="home"] .run-picker { display: none; }
    .top-actions { display: flex; gap: 8px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .layout {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1180px);
      justify-content: center;
      gap: 12px;
      padding: 12px;
    }
    body[data-view="home"] .old-column,
    body[data-view="home"] .proposal-column,
    body[data-view="home"] .notes-column,
    body[data-view="home"] .old-only,
    body[data-view="home"] .proposal-only,
    body[data-view="old"] .home-column,
    body[data-view="old"] .proposal-column,
    body[data-view="old"] .notes-column,
    body[data-view="old"] .proposal-only {
      display: none;
    }
    body[data-view="proposal"] .home-column,
    body[data-view="proposal"] .old-column,
    body[data-view="proposal"] .old-only {
      display: none;
    }
    body[data-view="home"] .layout,
    body[data-view="old"] .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .column {
      min-width: 0;
      min-height: calc(100vh - 92px);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .column-head {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .column-title { min-width: 0; }
    .column-title h2 { margin: 0; font-size: 15px; }
    .column-title p { margin: 3px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .column-body { min-height: 0; overflow: auto; }
    .old-frame { width: 100%; height: 100%; border: 0; background: #fff; }
    body[data-view="old"] .old-column {
      min-height: calc(100vh - 106px);
    }
    body[data-view="old"] .old-column .column-body {
      min-height: calc(100vh - 178px);
    }
    body[data-view="proposal"] .proposal-column,
    body[data-view="proposal"] .notes-column {
      min-height: auto;
    }
    body[data-view="proposal"] .proposal-column .column-body,
    body[data-view="proposal"] .notes-column .column-body {
      overflow: visible;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #fff;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .badge.ok { color: var(--ok); border-color: #86efac; background: var(--ok-soft); }
    .badge.review { color: var(--warn); border-color: #fcd34d; background: var(--warn-soft); }
    .badge.blocked { color: var(--danger); border-color: #fecaca; background: var(--danger-soft); }
    .badge.running { color: var(--blue); border-color: #bfdbfe; background: var(--blue-soft); }
    .proposal { padding: 12px; display: grid; gap: 12px; }
    .hero {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: stretch;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .panel h3 { margin: 0 0 8px; font-size: 14px; }
    .home { padding: 12px; display: grid; gap: 12px; }
    .section-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .section-head h3 { margin: 0; }
    .section-head p { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
    .home-kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
    }
    .run-param-grid {
      display: grid;
      grid-template-columns: minmax(280px, 1.6fr) repeat(3, minmax(120px, 0.7fr));
      gap: 8px;
      align-items: end;
    }
    .run-param-grid.secondary {
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      margin-top: 8px;
    }
    .form-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 10px;
    }
    .filters {
      display: grid;
      grid-template-columns: minmax(220px, 1.4fr) repeat(3, minmax(140px, 0.65fr));
      gap: 8px;
      align-items: end;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
      background: #fff;
    }
    .runs-table {
      table-layout: fixed;
      min-width: 1120px;
    }
    .runs-table th:nth-child(1),
    .runs-table td:nth-child(1) { width: 27%; }
    .runs-table th:nth-child(2),
    .runs-table td:nth-child(2) { width: 9%; }
    .runs-table th:nth-child(3),
    .runs-table td:nth-child(3) { width: 11%; }
    .runs-table th:nth-child(4),
    .runs-table td:nth-child(4) { width: 13%; }
    .runs-table th:nth-child(5),
    .runs-table td:nth-child(5) { width: 16%; }
    .runs-table th:nth-child(6),
    .runs-table td:nth-child(6) { width: 17%; }
    .runs-table th:nth-child(7),
    .runs-table td:nth-child(7) { width: 7%; }
    .models-table {
      table-layout: fixed;
      min-width: 1240px;
    }
    .models-table th:nth-child(1),
    .models-table td:nth-child(1) { width: 21%; }
    .models-table th:nth-child(2),
    .models-table td:nth-child(2) { width: 10%; }
    .models-table th:nth-child(3),
    .models-table td:nth-child(3) { width: 11%; }
    .models-table th:nth-child(4),
    .models-table td:nth-child(4) { width: 14%; }
    .models-table th:nth-child(5),
    .models-table td:nth-child(5) { width: 21%; }
    .models-table th:nth-child(6),
    .models-table td:nth-child(6) { width: 12%; }
    .models-table th:nth-child(7),
    .models-table td:nth-child(7) { width: 11%; }
    .data-table th,
    .data-table td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .data-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef2f7;
      color: #334155;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .data-table tr:hover td { background: #f8fafc; }
    .run-name,
    .model-name {
      font-weight: 850;
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .path-lite {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .row-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }
    .technical-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .models-toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(160px, 0.45fr) minmax(160px, 0.45fr);
      gap: 8px;
      align-items: end;
      margin: 10px 0;
    }
    .verdict {
      border-left: 5px solid var(--accent);
      display: grid;
      gap: 8px;
    }
    .verdict.review { border-left-color: var(--warn); background: linear-gradient(90deg, #fffbeb, #fff); }
    .verdict.blocked { border-left-color: var(--danger); background: linear-gradient(90deg, #fff1f2, #fff); }
    .verdict.ok { border-left-color: var(--ok); background: linear-gradient(90deg, #f0fdf4, #fff); }
    .verdict-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .verdict-main .title { font-size: 22px; font-weight: 850; }
    .muted { color: var(--muted); }
    .tiny { color: var(--muted); font-size: 12px; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: var(--panel-2);
      min-width: 0;
    }
    .metric .k { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .metric .v {
      margin-top: 4px;
      font-weight: 850;
      overflow-wrap: break-word;
      word-break: normal;
    }
    .metric .s { margin-top: 3px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .queue { display: grid; gap: 6px; }
    .queue-item {
      border: 1px solid #fcd34d;
      background: #fffbeb;
      color: #78350f;
      border-radius: 8px;
      padding: 8px;
      font-size: 12px;
    }
    .step-list { display: grid; gap: 6px; }
    .step-card {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      cursor: pointer;
    }
    .step-card.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
    .step-card .num {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      font-weight: 850;
      font-size: 12px;
    }
    .step-title { font-weight: 850; margin-bottom: 3px; }
    .step-meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .detail-grid .metric { background: #fff; }
    .fss-lines {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
      gap: 6px;
    }
    .fss-line {
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 8px;
      background: var(--panel-2);
      min-height: 78px;
    }
    .fss-line.ok { border-color: #86efac; background: #f0fdf4; }
    .fss-line.missing { border-color: #fcd34d; background: #fffbeb; }
    .fss-line .ln { font-weight: 900; font-size: 13px; }
    .fss-line .label { color: var(--muted); font-size: 11px; margin: 2px 0; }
    .fss-line .value { font-size: 11px; overflow-wrap: anywhere; }
    .notes { padding: 12px; display: grid; gap: 10px; align-content: start; }
    .note-tabs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .note-chip { font-size: 12px; padding: 7px; }
    .note-chip.active { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }
    .note-list { display: grid; gap: 8px; }
    .note-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fff;
      display: grid;
      gap: 5px;
    }
    .note-card .head {
      display: flex;
      gap: 6px;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }
    .note-card .body { font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .old-placeholder {
      padding: 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .raw-box {
      max-height: 230px;
      overflow: auto;
      background: #0f172a;
      color: #dbeafe;
      border-radius: 8px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      white-space: pre-wrap;
    }
    @media (max-width: 1180px) {
      .topbar { grid-template-columns: 1fr; }
      .layout { grid-template-columns: 1fr; }
      .column { min-height: 720px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero { grid-template-columns: 1fr; }
      .run-param-grid,
      .run-param-grid.secondary,
      .filters,
      .models-toolbar { grid-template-columns: 1fr; }
      .fss-lines { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body data-view="{{ initial_view|e }}">
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>Pipeline UX Lab</h1>
        <p>Home operativa, storico run, reti generate e prototipi della nuova review.</p>
      </div>
      <div class="field run-picker">
        <label for="runSelect">Run da confrontare</label>
        <select id="runSelect"></select>
      </div>
      <div class="top-actions">
        <a id="homePageLink" class="nav-link small" href="/">Home</a>
        <a id="oldPageLink" class="nav-link small" href="/old">Vecchia UI</a>
        <a id="proposalPageLink" class="nav-link small" href="/proposal">Proposta + commenti</a>
        <button id="reloadOldBtn" class="small old-only">Ricarica vecchia UI</button>
        <button id="copyBriefBtn" class="small proposal-only">Copia brief</button>
        <button id="saveNotesBtn" class="primary small proposal-only">Salva note</button>
      </div>
    </header>

    <main class="layout">
      <section class="column home-column">
        <div class="column-head">
          <div class="column-title">
            <h2>Home pipeline</h2>
            <p>Storico filtrabile e stato tecnico delle reti disponibili.</p>
          </div>
          <span class="badge">home</span>
        </div>
        <div class="column-body">
          <div class="home">
            <section class="panel">
              <div class="section-head">
                <div>
                  <h3>1. Parametri nuova run</h3>
                  <p>Configurazione minima per lanciare una cartella nel runner corrente.</p>
                </div>
                <span class="badge">run setup</span>
              </div>
              <form id="homeRunForm" method="post" target="_blank">
                <div class="run-param-grid">
                  <div class="field">
                    <label for="homeInputFolder">Cartella input</label>
                    <input id="homeInputFolder" name="input_folder" type="text" placeholder="/percorso/cartella_acquisizione">
                  </div>
                  <div class="field">
                    <label for="homeBatchSize">Batch size</label>
                    <input id="homeBatchSize" name="batch_size" type="number" min="1" value="32">
                  </div>
                  <div class="field">
                    <label for="homeSamplePerFolder">Sample/folder</label>
                    <input id="homeSamplePerFolder" name="sample_per_folder" type="number" min="1" value="80">
                  </div>
                  <div class="field">
                    <label for="homeRotationMaxSamples">Rotation samples</label>
                    <input id="homeRotationMaxSamples" name="rotation_max_samples" type="number" min="1" value="24">
                  </div>
                </div>
                <div class="run-param-grid secondary">
                  <div class="field">
                    <label for="homeLowPolicy">Low confidence</label>
                    <select id="homeLowPolicy" name="low_confidence_policy">
                      <option value="review" selected>review</option>
                      <option value="error">error</option>
                      <option value="ask_user">ask_user</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="homeVendorMinConf">Vendor min conf</label>
                    <input id="homeVendorMinConf" name="vendor_min_confidence" type="number" min="0" max="1" step="0.01" value="0.50">
                  </div>
                  <div class="field">
                    <label for="homeProbeMinConf">Probe min conf</label>
                    <input id="homeProbeMinConf" name="probe_min_confidence" type="number" min="0" max="1" step="0.01" value="0.50">
                  </div>
                  <div class="field">
                    <label for="homeRunProfile">Profilo</label>
                    <select id="homeRunProfile">
                      <option value="standard" selected>standard</option>
                      <option value="fast_review">fast review</option>
                      <option value="strict">strict</option>
                    </select>
                  </div>
                </div>
                <div class="form-actions">
                  <button type="submit" class="primary">Avvia run</button>
                  <a id="homeOpenOldSetup" class="nav-link" href="/old">Apri workbench classica</a>
                  <span class="tiny">L'avvio usa il motore della workbench classica e apre il monitor in una nuova pagina.</span>
                </div>
              </form>
            </section>

            <section class="panel">
              <div class="section-head">
                <div>
                  <h3>2. Storico run</h3>
                  <p>Archivio filtrabile per vendor, tag assegnato dopo review ed esito della pipeline.</p>
                </div>
                <span class="badge">history</span>
              </div>
              <div id="homeKpis" class="home-kpis" style="margin-bottom:10px;"></div>
              <div class="filters">
                <div class="field">
                  <label for="runSearchInput">Cerca run/cartella/probe/reason</label>
                  <input id="runSearchInput" type="search" placeholder="es. Mindray, 6LB7S, line14, Pavia">
                </div>
                <div class="field">
                  <label for="vendorFilter">Vendor</label>
                  <select id="vendorFilter"></select>
                </div>
                <div class="field">
                  <label for="tagFilter">Tag review</label>
                  <select id="tagFilter"></select>
                </div>
                <div class="field">
                  <label for="verdictFilter">Esito</label>
                  <select id="verdictFilter"></select>
                </div>
              </div>
              <div id="runsTableStatus" class="tiny" style="margin:10px 0 6px;"></div>
              <div class="table-wrap">
                <table class="data-table runs-table">
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Vendor</th>
                      <th>Tag review</th>
                      <th>Esito</th>
                      <th>Output</th>
                      <th>Problemi</th>
                      <th>Azioni</th>
                    </tr>
                  </thead>
                  <tbody id="runsTableBody"></tbody>
                </table>
              </div>
            </section>

            <section class="panel">
              <div class="section-head">
                <div>
                  <h3>3. Riepilogo reti generate finora</h3>
                  <p>Solo reti utili con metriche reali: escluse legacy, smoke, checkpoint-only e righe senza metrica primaria.</p>
                </div>
                <span class="badge">models</span>
              </div>
              <div id="modelsKpis" class="technical-grid"></div>
              <div class="models-toolbar">
                <div class="field">
                  <label for="modelSearchInput">Cerca rete/famiglia/vendor</label>
                  <input id="modelSearchInput" type="search" placeholder="vendor classifier, rect, BK, line16...">
                </div>
                <div class="field">
                  <label for="modelStatusFilter">Stato</label>
                  <select id="modelStatusFilter"></select>
                </div>
                <div class="field">
                  <label for="modelFamilyFilter">Famiglia</label>
                  <select id="modelFamilyFilter"></select>
                </div>
              </div>
              <div id="modelsTableStatus" class="tiny" style="margin:0 0 6px;"></div>
              <div class="table-wrap">
                <table class="data-table models-table">
                  <thead>
                    <tr>
                      <th>Rete</th>
                      <th>Data</th>
                      <th>Scope</th>
                      <th>Metrica primaria</th>
                      <th>Secondarie</th>
                      <th>Stato tecnico</th>
                      <th>Artifacts</th>
                    </tr>
                  </thead>
                  <tbody id="modelsTableBody"></tbody>
                </table>
              </div>
            </section>
          </div>
        </div>
      </section>

      <section class="column old-column">
        <div class="column-head">
          <div class="column-title">
            <h2>Interfaccia attuale</h2>
            <p id="oldUrlLabel"></p>
          </div>
          <span class="badge">old</span>
        </div>
        <div class="column-body" id="oldFrameHost">
          <iframe id="oldFrame" class="old-frame" title="Workbench attuale"></iframe>
        </div>
      </section>

      <section class="column proposal-column">
        <div class="column-head">
          <div class="column-title">
            <h2>Nuova proposta</h2>
            <p id="proposalSubtitle"></p>
          </div>
          <span class="badge">prototype</span>
        </div>
        <div class="column-body">
          <div class="proposal">
            <section id="verdictPanel" class="panel verdict">
              <div class="verdict-main">
                <div>
                  <div class="tiny">Esito run</div>
                  <div id="verdictTitle" class="title">-</div>
                </div>
                <span id="verdictBadge" class="badge">-</span>
              </div>
              <div id="verdictReason" class="muted"></div>
            </section>

            <section class="hero">
              <div class="panel">
                <h3>Output chiave</h3>
                <div id="summaryGrid" class="summary-grid"></div>
              </div>
              <div class="panel">
                <h3>Prossime azioni</h3>
                <div id="actionQueue" class="queue"></div>
              </div>
            </section>

            <section class="panel">
              <h3>Pipeline decisionale</h3>
              <div id="stepList" class="step-list"></div>
            </section>

            <section class="panel">
              <h3>Dettaglio step selezionato</h3>
              <div id="stepDetail"></div>
            </section>

            <section class="panel">
              <h3>Validazione .fss</h3>
              <div id="fssLines" class="fss-lines"></div>
            </section>
          </div>
        </div>
      </section>

      <aside class="column notes-column">
        <div class="column-head">
          <div class="column-title">
            <h2>Commenti e dritte</h2>
            <p id="notesStatus">Note non salvate</p>
          </div>
          <span class="badge">feedback</span>
        </div>
        <div class="column-body">
          <div class="notes">
            <div class="field">
              <label for="globalNote">Direzione generale</label>
              <textarea id="globalNote" placeholder="Cosa deve diventare questa pagina? Quale informazione deve saltare subito all'occhio?"></textarea>
            </div>

            <div>
              <div class="tiny" style="margin-bottom:6px;">Tipo nota rapida</div>
              <div id="noteKinds" class="note-tabs">
                <button class="note-chip active" data-kind="tenere">Tenere</button>
                <button class="note-chip" data-kind="cambiare">Cambiare</button>
                <button class="note-chip" data-kind="manca">Manca</button>
                <button class="note-chip" data-kind="domanda">Domanda</button>
              </div>
            </div>

            <div class="field">
              <label for="noteTarget">Area commentata</label>
              <select id="noteTarget"></select>
            </div>

            <div class="field">
              <label for="noteText">Nota</label>
              <textarea id="noteText" placeholder="Scrivi una dritta specifica. Es: questa card e' utile ma vorrei vedere anche la soglia..."></textarea>
            </div>

            <button id="addNoteBtn" class="primary">Aggiungi nota</button>
            <button id="exportNotesBtn">Esporta JSON note</button>

            <div class="panel" style="padding:10px;">
              <h3 style="margin-bottom:6px;">Note raccolte</h3>
              <div id="noteList" class="note-list"></div>
            </div>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const RUNS = {{ runs_json|safe }};
    const INITIAL_DIGEST = {{ digest_json|safe }};
    const INITIAL_FEEDBACK = {{ feedback_json|safe }};
    const HOME_DATA = {{ home_json|safe }};
    const MODELS_DATA = {{ models_json|safe }};
    const OLD_BASE_URL = {{ old_base_url_json|safe }};
    const INITIAL_VIEW = {{ initial_view_json|safe }};

    let digest = INITIAL_DIGEST;
    let feedback = normalizeFeedback(INITIAL_FEEDBACK);
    let selectedStepId = "";
    let activeNoteKind = "tenere";
    let currentView = INITIAL_VIEW || "home";

    const runSelect = document.getElementById("runSelect");
    const oldFrame = document.getElementById("oldFrame");
    const oldUrlLabel = document.getElementById("oldUrlLabel");
    const proposalSubtitle = document.getElementById("proposalSubtitle");
    const verdictPanel = document.getElementById("verdictPanel");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictReason = document.getElementById("verdictReason");
    const summaryGrid = document.getElementById("summaryGrid");
    const actionQueue = document.getElementById("actionQueue");
    const stepList = document.getElementById("stepList");
    const stepDetail = document.getElementById("stepDetail");
    const fssLines = document.getElementById("fssLines");
    const globalNote = document.getElementById("globalNote");
    const noteTarget = document.getElementById("noteTarget");
    const noteText = document.getElementById("noteText");
    const noteList = document.getElementById("noteList");
    const notesStatus = document.getElementById("notesStatus");
    const homePageLink = document.getElementById("homePageLink");
    const oldPageLink = document.getElementById("oldPageLink");
    const proposalPageLink = document.getElementById("proposalPageLink");
    const homeKpis = document.getElementById("homeKpis");
    const runSearchInput = document.getElementById("runSearchInput");
    const vendorFilter = document.getElementById("vendorFilter");
    const tagFilter = document.getElementById("tagFilter");
    const verdictFilter = document.getElementById("verdictFilter");
    const runsTableStatus = document.getElementById("runsTableStatus");
    const runsTableBody = document.getElementById("runsTableBody");
    const modelsKpis = document.getElementById("modelsKpis");
    const modelSearchInput = document.getElementById("modelSearchInput");
    const modelStatusFilter = document.getElementById("modelStatusFilter");
    const modelFamilyFilter = document.getElementById("modelFamilyFilter");
    const modelsTableStatus = document.getElementById("modelsTableStatus");
    const modelsTableBody = document.getElementById("modelsTableBody");
    const homeRunForm = document.getElementById("homeRunForm");
    const homeOpenOldSetup = document.getElementById("homeOpenOldSetup");
    const homeRunProfile = document.getElementById("homeRunProfile");
    const homeBatchSize = document.getElementById("homeBatchSize");
    const homeSamplePerFolder = document.getElementById("homeSamplePerFolder");
    const homeRotationMaxSamples = document.getElementById("homeRotationMaxSamples");
    const homeLowPolicy = document.getElementById("homeLowPolicy");
    const homeVendorMinConf = document.getElementById("homeVendorMinConf");
    const homeProbeMinConf = document.getElementById("homeProbeMinConf");
    const homeInputFolder = document.getElementById("homeInputFolder");

    function esc(value) {
      return String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function normalizeFeedback(raw) {
      const out = raw && typeof raw === "object" ? raw : {};
      if (!Array.isArray(out.notes)) out.notes = [];
      if (typeof out.global_note !== "string") out.global_note = "";
      return out;
    }

    function kindClass(kind) {
      const k = String(kind || "").toLowerCase();
      if (k === "ok") return "ok";
      if (k === "blocked") return "blocked";
      if (k === "running") return "running";
      if (k === "neutral") return "";
      return "review";
    }

    function statusBadgeClass(status) {
      const s = String(status || "").trim().toUpperCase();
      if (s === "OK") return "ok";
      if (s === "AGGIORNARE") return "blocked";
      if (s === "MONITORARE" || s === "VERIFICARE") return "review";
      return "";
    }

    function oldUrlForRun(runId) {
      const id = encodeURIComponent(runId || "");
      return `${OLD_BASE_URL}/?selected_run=${id}`;
    }

    function updatePageLinks() {
      const id = encodeURIComponent(digest.run_id || "");
      homePageLink.href = `/`;
      oldPageLink.href = `/old?run=${id}`;
      proposalPageLink.href = `/proposal?run=${id}`;
      homePageLink.classList.toggle("active", currentView === "home");
      oldPageLink.classList.toggle("active", currentView === "old");
      proposalPageLink.classList.toggle("active", currentView === "proposal");
      document.body.setAttribute("data-view", currentView);
    }

    function optionHtml(value, label, count) {
      const suffix = count === undefined ? "" : ` (${count})`;
      return `<option value="${esc(value)}">${esc(label)}${esc(suffix)}</option>`;
    }

    function populateHomeFilters() {
      if (!vendorFilter || !tagFilter || !verdictFilter) return;
      const vendors = Array.isArray(HOME_DATA.vendors) ? HOME_DATA.vendors : [];
      const tags = Array.isArray(HOME_DATA.tags) ? HOME_DATA.tags : [];
      const verdicts = Array.isArray(HOME_DATA.verdicts) ? HOME_DATA.verdicts : [];
      vendorFilter.innerHTML = optionHtml("", "Tutti i vendor") + vendors.map((it) => optionHtml(it.label, it.label, it.count)).join("");
      tagFilter.innerHTML = optionHtml("", "Tutti i tag") + tags.map((it) => optionHtml(it.label, it.label, it.count)).join("");
      verdictFilter.innerHTML = optionHtml("", "Tutti gli esiti") + verdicts.map((it) => optionHtml(it.label, it.label, it.count)).join("");
    }

    function renderHomeKpis() {
      if (!homeKpis) return;
      const sm = (HOME_DATA && HOME_DATA.summary) || {};
      homeKpis.innerHTML = [
        metric("Run totali", sm.total_runs, "storico workbench"),
        metric("Vendor", sm.vendors_total, "classi trovate"),
        metric("Con tag", sm.tagged_runs, "da review"),
        metric("In review", sm.review_runs, "quality gate"),
        metric("Pronte", sm.ready_runs, "senza blocchi"),
        metric("Bloccate", sm.blocked_runs, "errori/blocker"),
      ].join("");
    }

    function selectedRunHref(path, runId) {
      return `${path}?run=${encodeURIComponent(runId || "")}`;
    }

    function filteredRuns() {
      const rows = Array.isArray(HOME_DATA.runs) ? HOME_DATA.runs : [];
      const q = String(runSearchInput ? runSearchInput.value : "").trim().toLowerCase();
      const vendor = String(vendorFilter ? vendorFilter.value : "");
      const tag = String(tagFilter ? tagFilter.value : "");
      const verdict = String(verdictFilter ? verdictFilter.value : "");
      return rows.filter((row) => {
        if (vendor && String(row.vendor || "") !== vendor) return false;
        if (tag && String(row.tag || "") !== tag) return false;
        if (verdict && String(row.verdict || "") !== verdict) return false;
        if (!q) return true;
        const hay = [
          row.run_id,
          row.folder_name,
          row.input_folder,
          row.vendor,
          row.tag,
          row.probe,
          row.probe_type,
          row.review_reasons,
          row.review_note,
        ].join(" ").toLowerCase();
        return hay.includes(q);
      });
    }

    function renderRunsTable() {
      if (!runsTableBody) return;
      const rows = filteredRuns();
      const total = Array.isArray(HOME_DATA.runs) ? HOME_DATA.runs.length : 0;
      runsTableStatus.textContent = `${rows.length}/${total} run visibili`;
      if (!rows.length) {
        runsTableBody.innerHTML = `<tr><td colspan="7" class="muted">Nessuna run corrisponde ai filtri.</td></tr>`;
        return;
      }
      runsTableBody.innerHTML = rows.map((row) => {
        const verdictKind = kindClass(row.verdict_kind);
        const tagKind = kindClass(row.tag_kind);
        const reasons = String(row.review_reasons || "").trim();
        return `
          <tr>
            <td>
              <div class="run-name">${esc(row.folder_name || row.run_id)}</div>
              <span class="path-lite">${esc(row.run_id || "-")}</span>
              <span class="path-lite">${esc(row.created_at || "")}</span>
            </td>
            <td><span class="badge">${esc(row.vendor || "-")}</span></td>
            <td>
              <span class="badge ${tagKind}">${esc(row.tag || "senza_tag")}</span>
              ${row.review_note ? `<span class="path-lite">${esc(row.review_note)}</span>` : ""}
            </td>
            <td>
              <span class="badge ${verdictKind}">${esc(row.verdict || "-")}</span>
              <span class="path-lite">step ok ${esc(row.ok_steps)} | review ${esc(row.review_steps)}</span>
            </td>
            <td>
              <div><b>Probe</b> ${esc(row.probe || "-")}</div>
              <div class="path-lite">type ${esc(row.probe_type || "-")} | rect ${esc(row.rect || "-")}</div>
            </td>
            <td>
              ${reasons ? esc(reasons) : "<span class='muted'>nessun motivo review</span>"}
              <span class="path-lite">azioni: ${esc(row.actions_count || 0)}</span>
            </td>
            <td>
              <div class="row-actions">
                <a class="nav-link small" href="${esc(selectedRunHref('/proposal', row.run_id))}">review</a>
                <a class="nav-link small" href="${esc(selectedRunHref('/old', row.run_id))}">vecchia</a>
              </div>
            </td>
          </tr>
        `;
      }).join("");
    }

    function populateModelFilters() {
      if (!modelStatusFilter || !modelFamilyFilter) return;
      const rows = Array.isArray(MODELS_DATA.rows) ? MODELS_DATA.rows : [];
      const statusCounts = {};
      const familyCounts = {};
      rows.forEach((row) => {
        const st = String(row.update_status || "UNKNOWN");
        const fam = String(row.family || "-");
        statusCounts[st] = (statusCounts[st] || 0) + 1;
        familyCounts[fam] = (familyCounts[fam] || 0) + 1;
      });
      const statuses = Object.keys(statusCounts).sort();
      const families = Object.keys(familyCounts).sort();
      modelStatusFilter.innerHTML = optionHtml("", "Tutti") + statuses.map((st) => optionHtml(st, st, statusCounts[st])).join("");
      modelFamilyFilter.innerHTML = optionHtml("", "Tutte") + families.map((fam) => optionHtml(fam, fam, familyCounts[fam])).join("");
    }

    function renderModelsKpis() {
      if (!modelsKpis) return;
      const sm = MODELS_DATA.summary || {};
      const rows = Array.isArray(MODELS_DATA.rows) ? MODELS_DATA.rows : [];
      const sc = {};
      let missingCheckpoint = 0;
      let missingMetrics = 0;
      rows.forEach((row) => {
        const st = String(row.update_status || "UNKNOWN");
        sc[st] = (sc[st] || 0) + 1;
        if (!row.checkpoint_exists) missingCheckpoint += 1;
        if (!row.metrics_exists) missingMetrics += 1;
      });
      modelsKpis.innerHTML = [
        metric("Reti utili", rows.length, `${sm.source_rows || sm.total_rows || 0} censite nel CSV`),
        metric("Con metriche", rows.length, "primary metric + file metrics"),
        metric("OK", sc.OK || 0, "pronte"),
        metric("Monitorare", sc.MONITORARE || 0, "da osservare"),
        metric("Verificare", sc.VERIFICARE || 0, "audit mirato"),
        metric("Artifact mancanti", `${missingCheckpoint} ckpt / ${missingMetrics} metrics`, "dopo filtro legacy"),
      ].join("");
    }

    function configureHomeRunForm() {
      if (homeRunForm) homeRunForm.action = `${OLD_BASE_URL}/runs/start_form`;
      if (homeOpenOldSetup) homeOpenOldSetup.href = `${OLD_BASE_URL}/`;
    }

    function applyRunProfile() {
      const profile = String(homeRunProfile ? homeRunProfile.value : "standard");
      if (profile === "fast_review") {
        if (homeBatchSize) homeBatchSize.value = "24";
        if (homeSamplePerFolder) homeSamplePerFolder.value = "40";
        if (homeRotationMaxSamples) homeRotationMaxSamples.value = "12";
        if (homeLowPolicy) homeLowPolicy.value = "review";
        if (homeVendorMinConf) homeVendorMinConf.value = "0.50";
        if (homeProbeMinConf) homeProbeMinConf.value = "0.50";
      } else if (profile === "strict") {
        if (homeBatchSize) homeBatchSize.value = "32";
        if (homeSamplePerFolder) homeSamplePerFolder.value = "120";
        if (homeRotationMaxSamples) homeRotationMaxSamples.value = "36";
        if (homeLowPolicy) homeLowPolicy.value = "error";
        if (homeVendorMinConf) homeVendorMinConf.value = "0.70";
        if (homeProbeMinConf) homeProbeMinConf.value = "0.70";
      } else {
        if (homeBatchSize) homeBatchSize.value = "32";
        if (homeSamplePerFolder) homeSamplePerFolder.value = "80";
        if (homeRotationMaxSamples) homeRotationMaxSamples.value = "24";
        if (homeLowPolicy) homeLowPolicy.value = "review";
        if (homeVendorMinConf) homeVendorMinConf.value = "0.50";
        if (homeProbeMinConf) homeProbeMinConf.value = "0.50";
      }
    }

    function filteredModels() {
      const rows = Array.isArray(MODELS_DATA.rows) ? MODELS_DATA.rows : [];
      const q = String(modelSearchInput ? modelSearchInput.value : "").trim().toLowerCase();
      const status = String(modelStatusFilter ? modelStatusFilter.value : "");
      const family = String(modelFamilyFilter ? modelFamilyFilter.value : "");
      return rows.filter((row) => {
        if (status && String(row.update_status || "") !== status) return false;
        if (family && String(row.family || "") !== family) return false;
        if (!q) return true;
        const hay = [row.family, row.scope, row.vendor, row.run_name, row.model_date, row.primary_metric, row.secondary_metrics, row.update_reason].join(" ").toLowerCase();
        return hay.includes(q);
      });
    }

    function renderModelsTable() {
      if (!modelsTableBody) return;
      const rows = filteredModels();
      const total = Array.isArray(MODELS_DATA.rows) ? MODELS_DATA.rows.length : 0;
      modelsTableStatus.textContent = MODELS_DATA.ok
        ? `${rows.length}/${total} reti visibili`
        : `Registro non disponibile: ${MODELS_DATA.error || "-"}`;
      if (!rows.length) {
        modelsTableBody.innerHTML = `<tr><td colspan="7" class="muted">Nessuna rete corrisponde ai filtri.</td></tr>`;
        return;
      }
      modelsTableBody.innerHTML = rows.map((row) => {
        const statusClass = statusBadgeClass(row.update_status);
        const sample = row.test_samples == null ? "-" : Number(row.test_samples).toLocaleString("it-IT", {maximumFractionDigits: 0});
        return `
          <tr>
            <td>
              <div class="model-name">${esc(row.run_name || "-")}</div>
              <span class="path-lite">${esc(row.family || "-")}</span>
            </td>
            <td title="${esc(row.model_datetime || "")}">
              <b>${esc(row.model_date || "-")}</b>
              <span class="path-lite">${row.model_date_source ? `da ${esc(row.model_date_source)}` : "-"}</span>
            </td>
            <td>
              <span class="badge">${esc(row.scope || "-")}</span>
              <span class="badge">${esc(row.vendor || "-")}</span>
            </td>
            <td>
              <b>${esc(row.primary_metric || "-")}</b>: ${esc(row.primary_value_label || "-")}
              <span class="path-lite">test n=${esc(sample)}</span>
            </td>
            <td>${esc(row.secondary_metrics || "-")}</td>
            <td>
              <span class="badge ${statusClass}">${esc(row.update_status || "UNKNOWN")}</span>
              <span class="path-lite">${esc(row.update_reason || row.notes || "-")}</span>
            </td>
            <td>
              <span class="badge ${row.checkpoint_exists ? "ok" : "blocked"}">ckpt ${row.checkpoint_exists ? "ok" : "missing"}</span>
              <span class="path-lite" title="${esc(row.checkpoint_path || "")}">${esc(row.checkpoint_name || "-")}</span>
              <span class="badge ${row.metrics_exists ? "ok" : "review"}">metrics ${row.metrics_exists ? "ok" : "missing"}</span>
              <span class="path-lite" title="${esc(row.metrics_path || "")}">${esc(row.metrics_name || "-")}</span>
            </td>
          </tr>
        `;
      }).join("");
    }

    function renderHome() {
      renderHomeKpis();
      renderRunsTable();
      renderModelsKpis();
      renderModelsTable();
    }

    function populateRuns() {
      runSelect.innerHTML = RUNS.map((run) => {
        const id = String(run.run_id || "");
        const label = `${id} | ${String(run.input_folder || "").split(/[\\\\/]+/).filter(Boolean).pop() || "-"}`;
        return `<option value="${esc(id)}">${esc(label)}</option>`;
      }).join("");
      if (digest && digest.run_id) runSelect.value = digest.run_id;
    }

    function renderOldFrame() {
      const url = oldUrlForRun(digest.run_id);
      oldUrlLabel.textContent = url;
      if (currentView === "old") oldFrame.src = url;
    }

    function renderVerdict() {
      proposalSubtitle.textContent = `${digest.run_id || "-"} | ${digest.folder_name || "-"}`;
      const kind = kindClass(digest.verdict_kind);
      verdictPanel.className = `panel verdict ${kind}`;
      verdictTitle.textContent = digest.verdict || "-";
      verdictBadge.className = `badge ${kind}`;
      verdictBadge.textContent = kind;
      const queueCount = Array.isArray(digest.action_queue) ? digest.action_queue.length : 0;
      const reasons = digest.review_reasons || (queueCount ? `${queueCount} punto/i da controllare` : "Nessun blocco evidente.");
      verdictReason.textContent = reasons;
    }

    function metric(label, value, sub) {
      const shown = (value === 0 || value === "0") ? "0" : (value || "-");
      return `
        <div class="metric">
          <div class="k">${esc(label)}</div>
          <div class="v">${esc(shown)}</div>
          ${sub ? `<div class="s">${esc(sub)}</div>` : ""}
        </div>
      `;
    }

    function renderSummary() {
      const sm = digest.summary || {};
      summaryGrid.innerHTML = [
        metric("Vendor", sm.vendor, sm.vendor_confidence ? `conf ${sm.vendor_confidence}` : ""),
        metric("Probe", sm.probe, sm.probe_confidence ? `conf ${sm.probe_confidence}` : ""),
        metric("Probe type", sm.probe_type, "riga 04"),
        metric("Rect", sm.rect, sm.rect_source),
        metric("Rotazione", sm.rotation, "deg clockwise"),
        metric("SU/GIU", sm.su_giu, "majority"),
        metric("L/T", sm.lt, "majority"),
        metric("LR marker", sm.lr, "best match"),
        metric("Raw", sm.raw_images, `duplicati rimossi ${sm.duplicates_removed}`),
      ].join("");
    }

    function renderActionQueue() {
      const items = Array.isArray(digest.action_queue) ? digest.action_queue : [];
      if (!items.length) {
        actionQueue.innerHTML = `<div class="queue-item" style="border-color:#86efac;background:#f0fdf4;color:#14532d;">Nessuna azione obbligatoria emersa dai dati.</div>`;
        return;
      }
      actionQueue.innerHTML = items.slice(0, 8).map((item) => `
        <button class="queue-item" data-target="${esc(item.target || "")}" title="Seleziona area collegata">
          <b>${esc(item.title || "Controllo")}</b><br>${esc(item.reason || "")}
        </button>
      `).join("");
    }

    function renderSteps() {
      const steps = Array.isArray(digest.steps) ? digest.steps : [];
      if (!selectedStepId && steps.length) selectedStepId = steps[0].id;
      stepList.innerHTML = steps.map((step, idx) => {
        const active = step.id === selectedStepId ? " active" : "";
        const kind = kindClass(step.kind);
        return `
          <div class="step-card${active}" data-step-id="${esc(step.id)}">
            <div class="num">${idx + 1}</div>
            <div>
              <div class="step-title">${esc(step.title)}</div>
              <div class="step-meta">Risultato: <b>${esc(step.result)}</b>${step.confidence ? ` | ${esc(step.confidence)}` : ""}</div>
              <div class="step-meta">Fonte: ${esc(step.source || "-")}${step.reason ? ` | ${esc(step.reason)}` : ""}</div>
            </div>
            <span class="badge ${kind}">${esc(step.status || kind)}</span>
          </div>
        `;
      }).join("");
      renderStepDetail();
      renderNoteTargets();
    }

    function renderStepDetail() {
      const steps = Array.isArray(digest.steps) ? digest.steps : [];
      const step = steps.find((item) => item.id === selectedStepId) || steps[0];
      if (!step) {
        stepDetail.innerHTML = `<div class="tiny">Nessuno step disponibile.</div>`;
        return;
      }
      const raw = JSON.stringify(step.raw || {}, null, 2);
      stepDetail.innerHTML = `
        <div class="detail-grid">
          ${metric("Step", step.title, step.id)}
          ${metric("Stato", step.status, step.kind)}
          ${metric("Risultato", step.result, step.confidence)}
          ${metric("Fonte", step.source || "-", step.reason || "")}
        </div>
        <div class="tiny" style="margin:10px 0 5px;">Dati raw dello step</div>
        <div class="raw-box">${esc(raw)}</div>
      `;
      if (!noteText.value.trim()) {
        noteTarget.value = step.id;
      }
    }

    function renderFssLines() {
      const lines = Array.isArray(digest.fss_lines) ? digest.fss_lines : [];
      fssLines.innerHTML = lines.map((line) => {
        const status = line.status === "ok" ? "ok" : "missing";
        return `
          <div class="fss-line ${status}">
            <div class="ln">#${esc(line.line)}</div>
            <div class="label">${esc(line.label)}</div>
            <div class="value">${esc(line.value || "-")}</div>
            <div style="margin-top:6px;"><span class="badge ${status === "ok" ? "ok" : "review"}">${status === "ok" ? "compilata" : "manca"}</span></div>
          </div>
        `;
      }).join("");
    }

    function renderNoteTargets() {
      const steps = Array.isArray(digest.steps) ? digest.steps : [];
      const fss = Array.isArray(digest.fss_lines) ? digest.fss_lines : [];
      const opts = [
        ["overview", "Vista generale"],
        ["old_ui", "Vecchia interfaccia"],
        ["new_ui", "Nuova proposta"],
        ...steps.map((s) => [s.id, `Step: ${s.title}`]),
        ...fss.map((l) => [l.key, `.fss #${l.line}: ${l.label}`]),
      ];
      const current = noteTarget.value || selectedStepId || "overview";
      noteTarget.innerHTML = opts.map(([value, label]) => `<option value="${esc(value)}">${esc(label)}</option>`).join("");
      noteTarget.value = opts.some(([value]) => value === current) ? current : "overview";
    }

    function renderNotes() {
      globalNote.value = feedback.global_note || "";
      const notes = Array.isArray(feedback.notes) ? feedback.notes : [];
      notesStatus.textContent = `${notes.length} note | ${feedback.saved_at ? "salvate" : "draft"}`;
      if (!notes.length) {
        noteList.innerHTML = `<div class="tiny">Ancora nessuna nota. Seleziona un'area, scrivi una dritta e aggiungila qui.</div>`;
        return;
      }
      noteList.innerHTML = notes.map((note, idx) => `
        <div class="note-card">
          <div class="head">
            <span>${esc(note.kind || "nota")} | ${esc(note.target || "overview")}</span>
            <button class="small ghost" data-delete-note="${idx}">elimina</button>
          </div>
          <div class="body">${esc(note.text || "")}</div>
        </div>
      `).join("");
    }

    function renderAll() {
      updatePageLinks();
      renderHome();
      renderOldFrame();
      renderVerdict();
      renderSummary();
      renderActionQueue();
      renderSteps();
      renderFssLines();
      renderNoteTargets();
      renderNotes();
    }

    async function loadRun(runId) {
      const res = await fetch(`/api/run/${encodeURIComponent(runId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      digest = await res.json();
      selectedStepId = "";
      const fb = await fetch(`/api/feedback/${encodeURIComponent(runId)}`);
      feedback = normalizeFeedback(fb.ok ? await fb.json() : {});
      renderAll();
      const url = new URL(window.location.href);
      url.searchParams.set("run", runId);
      window.history.replaceState({}, "", url);
    }

    async function saveNotes() {
      feedback.global_note = globalNote.value || "";
      feedback.updated_at = new Date().toISOString();
      const res = await fetch(`/api/feedback/${encodeURIComponent(digest.run_id)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(feedback, null, 2),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      feedback = normalizeFeedback(await res.json());
      renderNotes();
    }

    function buildBrief() {
      const sm = digest.summary || {};
      const notes = (feedback.notes || []).map((note) => `- [${note.kind}] ${note.target}: ${note.text}`).join("\\n");
      return [
        `# Feedback UX Pipeline - ${digest.run_id}`,
        ``,
        `Verdetto: ${digest.verdict}`,
        `Motivo: ${digest.review_reasons || "-"}`,
        `Vendor: ${sm.vendor || "-"} (${sm.vendor_confidence || "-"})`,
        `Probe: ${sm.probe || "-"} (${sm.probe_confidence || "-"})`,
        `Rect: ${sm.rect || "-"}`,
        ``,
        `## Direzione generale`,
        feedback.global_note || "-",
        ``,
        `## Note puntuali`,
        notes || "-",
      ].join("\\n");
    }

    document.getElementById("reloadOldBtn").addEventListener("click", () => renderOldFrame());
    if (homeRunProfile) homeRunProfile.addEventListener("change", applyRunProfile);
    if (homeRunForm) {
      homeRunForm.addEventListener("submit", (ev) => {
        const inputFolder = String(homeInputFolder ? homeInputFolder.value : "").trim();
        if (!inputFolder) {
          ev.preventDefault();
          alert("Inserisci la cartella input prima di avviare la run.");
        }
      });
    }
    document.getElementById("saveNotesBtn").addEventListener("click", async () => {
      try {
        await saveNotes();
      } catch (err) {
        notesStatus.textContent = `Errore salvataggio: ${err.message}`;
      }
    });
    document.getElementById("copyBriefBtn").addEventListener("click", async () => {
      feedback.global_note = globalNote.value || "";
      const brief = buildBrief();
      await navigator.clipboard.writeText(brief);
      notesStatus.textContent = "Brief copiato negli appunti";
    });
    document.getElementById("exportNotesBtn").addEventListener("click", () => {
      feedback.global_note = globalNote.value || "";
      const blob = new Blob([JSON.stringify({run_id: digest.run_id, feedback}, null, 2)], {type: "application/json"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${digest.run_id || "run"}_ux_feedback.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
    document.getElementById("addNoteBtn").addEventListener("click", () => {
      const text = noteText.value.trim();
      if (!text) return;
      feedback.global_note = globalNote.value || "";
      feedback.notes.push({
        kind: activeNoteKind,
        target: noteTarget.value || "overview",
        text,
        created_at: new Date().toISOString(),
      });
      noteText.value = "";
      feedback.saved_at = "";
      renderNotes();
    });
    document.getElementById("noteKinds").addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-kind]");
      if (!btn) return;
      activeNoteKind = btn.getAttribute("data-kind") || "tenere";
      document.querySelectorAll(".note-chip").forEach((el) => el.classList.toggle("active", el === btn));
    });
    noteList.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-delete-note]");
      if (!btn) return;
      const idx = Number(btn.getAttribute("data-delete-note"));
      if (Number.isFinite(idx)) feedback.notes.splice(idx, 1);
      feedback.saved_at = "";
      renderNotes();
    });
    stepList.addEventListener("click", (ev) => {
      const card = ev.target.closest("[data-step-id]");
      if (!card) return;
      selectedStepId = card.getAttribute("data-step-id") || "";
      renderSteps();
    });
    actionQueue.addEventListener("click", (ev) => {
      const item = ev.target.closest("[data-target]");
      if (!item) return;
      const target = item.getAttribute("data-target") || "";
      if ((digest.steps || []).some((step) => step.id === target)) {
        selectedStepId = target;
        renderSteps();
      }
      noteTarget.value = target || "overview";
    });
    runSelect.addEventListener("change", () => loadRun(runSelect.value));
    [runSearchInput, vendorFilter, tagFilter, verdictFilter].forEach((el) => {
      if (!el) return;
      el.addEventListener("input", renderRunsTable);
      el.addEventListener("change", renderRunsTable);
    });
    [modelSearchInput, modelStatusFilter, modelFamilyFilter].forEach((el) => {
      if (!el) return;
      el.addEventListener("input", renderModelsTable);
      el.addEventListener("change", renderModelsTable);
    });

    populateRuns();
    populateHomeFilters();
    populateModelFilters();
    configureHomeRunForm();
    renderAll();
  </script>
</body>
</html>
"""


def create_compare_app(
    *,
    data_root: Path,
    feedback_root: Path,
    old_base_url: str,
    models_metrics_csv: Optional[Path],
) -> Flask:
    app = Flask(__name__)
    store = WorkbenchStore(data_root)
    feedback_root.mkdir(parents=True, exist_ok=True)

    def _pick_run(run_id: str = "") -> str:
        wanted = str(run_id or "").strip()
        runs = store.list_runs()
        if wanted:
            return wanted
        if runs:
            return str(runs[0].get("run_id", ""))
        return ""

    def _load_feedback(run_id: str) -> Dict[str, Any]:
        path = _feedback_path(feedback_root, run_id)
        out = _read_json(path, {"global_note": "", "notes": []})
        return out if isinstance(out, dict) else {"global_note": "", "notes": []}

    def _render_page(view: str):
        raw_view = str(view or "").strip().lower()
        initial_view = raw_view if raw_view in {"home", "old", "proposal"} else "home"
        run_id = _pick_run(str(request.args.get("run", "")))
        runs = store.list_runs()
        digest = _build_digest(store, run_id)
        feedback = _load_feedback(run_id)
        home_data = _build_home_data(store)
        models_data = _build_models_home_data(models_metrics_csv)
        return render_template_string(
            HTML_PAGE,
            initial_view=initial_view,
            initial_view_json=json.dumps(initial_view, ensure_ascii=False),
            runs_json=json.dumps(runs, ensure_ascii=False),
            digest_json=json.dumps(digest, ensure_ascii=False),
            feedback_json=json.dumps(feedback, ensure_ascii=False),
            home_json=json.dumps(home_data, ensure_ascii=False),
            models_json=json.dumps(models_data, ensure_ascii=False),
            old_base_url_json=json.dumps(old_base_url, ensure_ascii=False),
        )

    @app.get("/")
    def home_page():
        return _render_page("home")

    @app.get("/proposal")
    def proposal_page():
        return _render_page("proposal")

    @app.get("/old")
    def old_page():
        return _render_page("old")

    @app.get("/api/run/<run_id>")
    def api_run(run_id: str):
        return jsonify(_build_digest(store, run_id))

    @app.get("/api/feedback/<run_id>")
    def api_feedback_get(run_id: str):
        return jsonify(_load_feedback(run_id))

    @app.post("/api/feedback/<run_id>")
    def api_feedback_save(run_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "payload JSON non valido"}), 400
        payload["run_id"] = run_id
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        path = _feedback_path(feedback_root, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify(payload)

    return app


def _start_old_workbench_if_needed(
    *,
    host: str,
    port: int,
    data_root: Path,
    python_bin: str,
    models_metrics_csv: Optional[Path],
) -> bool:
    if _is_port_open(host, port):
        return False

    old_app = create_workbench_app(
        data_root=data_root,
        python_bin=python_bin,
        models_metrics_csv=models_metrics_csv,
    )

    def _run() -> None:
        old_app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for _ in range(40):
        if _is_port_open(host, port):
            return True
        time.sleep(0.1)
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline UX side-by-side compare tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--old-host", default="127.0.0.1")
    parser.add_argument("--old-port", type=int, default=8780)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feedback-root", type=Path, default=DEFAULT_FEEDBACK_ROOT)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument(
        "--models-metrics-csv",
        type=Path,
        default=REPO_ROOT / "artifacts/60_metadata/reti_metriche_status_2026-04-01.csv",
    )
    parser.add_argument("--no-old-server", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    feedback_root = args.feedback_root.expanduser().resolve()
    models_metrics_csv = args.models_metrics_csv.expanduser().resolve() if args.models_metrics_csv else None
    old_base_url = f"http://{args.old_host}:{args.old_port}"

    old_started = False
    if not args.no_old_server:
        old_started = _start_old_workbench_if_needed(
            host=args.old_host,
            port=args.old_port,
            data_root=data_root,
            python_bin=args.python_bin,
            models_metrics_csv=models_metrics_csv,
        )

    app = create_compare_app(
        data_root=data_root,
        feedback_root=feedback_root,
        old_base_url=old_base_url,
        models_metrics_csv=models_metrics_csv,
    )
    print(
        f"Pipeline UX Compare running on http://{args.host}:{args.port} | "
        f"old_ui={old_base_url} ({'started' if old_started else 'already available or disabled'}) | "
        f"feedback_root={feedback_root}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
