"""Scan `artifacts/` and normalise every run into flat metric records.

The repository stores results in a dozen different shapes (trainer `metrics.json`,
eval `summary.json`, per-vendor CSV exports, pipeline prediction CSVs). This module
holds one adapter per shape and emits a single record type:

    {step_id, module, run_id, run_path, run_kind, date, source, split,
     vendor, metric, value, n, production}

`vendor` is `None` for the aggregate value of a run and a canonical vendor name for
the per-vendor breakdown. Records are appended to `metrics.jsonl`; the run index
goes to `runs.jsonl`.

Usage:
    python3 tools/monitor/collect_metrics.py --artifacts-root artifacts \
        --out-dir artifacts/71_monitor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_registry import (  # noqa: E402
    REVIEW_REASON_TO_STEP,
    STEP_BY_ID,
    VENDOR_CANONICAL,
    canonical_vendor,
)

# Directories that only ever contain images/checkpoints: never worth descending.
PRUNE_DIR_NAMES = {
    "__pycache__",
    ".git",
    "image_samples",
    "images",
    "crops",
    "thumbs",
    "assets",
    "tiles",
    "node_modules",
}
PRUNE_DIR_PATTERNS = (
    re.compile(r"gallery", re.I),
    re.compile(r"overlay", re.I),
    re.compile(r"preview_images", re.I),
    re.compile(r"^review_.*_images$", re.I),
)

INTERESTING_FILES = {
    "metrics.json",
    "summary.json",
    "test_metrics_from_best.json",
    "train_summary.json",
    "vendor_training_summary.json",
    "folder_summary.csv",
    "folder_fss_head_predictions.csv",
}

DATE_COMPACT = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
DATE_DASHED = re.compile(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    """Coerce to float, returning None for empty / non numeric / non finite."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _date_for(path: Path, payload: Optional[Dict[str, Any]] = None) -> str:
    """Best available date for a run: explicit field, then path, then mtime."""
    if payload:
        for key in ("date", "generated_at", "timestamp", "created_at", "run_date"):
            raw = payload.get(key)
            if isinstance(raw, str):
                m = DATE_DASHED.search(raw) or DATE_COMPACT.search(raw)
                if m:
                    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    text = str(path)
    m = DATE_DASHED.search(text) or DATE_COMPACT.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    try:
        ts = path.stat().st_mtime
    except OSError:
        return "1970-01-01"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


STEP_PATH_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"line13", re.I), "06_line13"),
    (re.compile(r"line14", re.I), "09_line14"),
    (re.compile(r"line16|line21|orientation_marker|lr_marker|symbol_detection|envelope", re.I), "12_line16"),
    (re.compile(r"orientation_hint|su_?giu|sugiu", re.I), "11_sugiu_lr_lt"),
    (re.compile(r"(^|[_/])lt[_/]|lt_training|transrectal_lt|piano_lt", re.I), "11_sugiu_lr_lt"),
    (re.compile(r"scale|ladder|heatmap|depth", re.I), "13_rect_depth"),
    (re.compile(r"rotation|osd", re.I), "03_rotation"),
    (re.compile(r"probe_training|probe_classifier|probe_model", re.I), "07_probe"),
    (re.compile(r"probe_type|probetype", re.I), "08_probetype"),
    (re.compile(r"vendor_training|vendor_classifier", re.I), "05_vendor"),
    (re.compile(r"rect_training|rect_net|rect_model|rect_dataset", re.I), "10_line11"),
    (re.compile(r"id_echo", re.I), "14_residual"),
]


def step_from_path(path: Path, fallback: Optional[str] = None) -> Optional[str]:
    text = str(path)
    for pattern, step_id in STEP_PATH_RULES:
        if pattern.search(text):
            return step_id
    return fallback


# Runs whose name marks them as throwaway: kept in the store, hidden by default
# in the dashboard so a smoke test on 3 folders never moves a KPI.
SMOKE_PATTERN = re.compile(r"smoke|dry_?run|(^|/)tmp_|_test($|/)|debug|sanity|_check($|/)", re.I)


def _run_identity(source: Path, artifacts_root: Path) -> Tuple[str, str, bool, str]:
    """(run_id, run_path relative to the repo, is_production, scope) for a metric file."""
    run_dir = source.parent
    try:
        rel = run_dir.relative_to(artifacts_root)
    except ValueError:
        rel = run_dir
    rel_str = os.path.join(artifacts_root.name, str(rel)) if str(rel) != "." else artifacts_root.name
    parts = [p for p in rel.parts if p]
    run_id = "/".join(parts) if parts else run_dir.name
    production = bool(parts) and parts[0] == "10_active_pipeline"
    scope = "smoke" if SMOKE_PATTERN.search(run_id) else "full"
    return run_id, rel_str, production, scope


def run_vendor_scope(source: Path) -> Optional[str]:
    """Vendor a run is dedicated to, when the run directory says so.

    `30_models/rect_training_vendor_.../vendor_models/mindray` and
    `40_scale_heatmap_models_clean/bk` are single-vendor runs: their aggregate
    figure must not be compared with a multi-vendor model, so it is attributed to
    that vendor instead of to the overall series.
    """
    for name in (source.parent.name, source.parent.parent.name):
        key = name.lower().replace(" ", "").replace("_", "-")
        if key in VENDOR_CANONICAL:
            return VENDOR_CANONICAL[key]
    return None


class Collector:
    def __init__(self, artifacts_root: Path, overrides: Dict[str, str]):
        self.artifacts_root = artifacts_root
        self.overrides = overrides
        self.records: List[Dict[str, Any]] = []
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.reasons: List[Dict[str, Any]] = []
        self.skipped: List[str] = []
        self._scope_vendor: Optional[str] = None

    # -- record plumbing --------------------------------------------------
    def add(
        self,
        *,
        source: Path,
        step_id: Optional[str],
        run_kind: str,
        metric: str,
        value: Any,
        vendor: Optional[str] = None,
        split: str = "test",
        n: Optional[int] = None,
        label: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        val = _num(value)
        if val is None or step_id is None or step_id not in STEP_BY_ID:
            return
        if vendor is None and self._scope_vendor:
            vendor = self._scope_vendor
        run_id, run_path, production, scope = _run_identity(source, self.artifacts_root)
        run_id = self.overrides.get(f"run:{run_id}", run_id)
        step_id = self.overrides.get(f"step:{run_id}", step_id)
        run_date = date or _date_for(source)
        self.records.append(
            {
                "step_id": step_id,
                "module": STEP_BY_ID[step_id]["module"],
                "run_id": run_id,
                "run_path": run_path,
                "run_kind": run_kind,
                "scope": scope,
                "date": run_date,
                "source": source.name,
                "split": split,
                "vendor": canonical_vendor(vendor),
                "metric": metric,
                "value": val,
                "n": int(n) if n is not None else None,
                "production": production,
            }
        )
        entry = self.runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "run_path": run_path,
                "date": run_date,
                "kind": run_kind,
                "scope": scope,
                "production": production,
                "group": run_id.split("/")[0],
                "label": label or "/".join(run_id.split("/")[1:]) or run_id,
                "steps": [],
            },
        )
        if step_id not in entry["steps"]:
            entry["steps"].append(step_id)
        # keep the earliest resolved date for a run seen through several files
        if run_date < entry["date"]:
            entry["date"] = run_date

    # -- adapters ---------------------------------------------------------
    def adapt_classifier(self, source: Path, data: Dict[str, Any], step_id: str, kind: str) -> None:
        """Trainer metrics for a classifier: vendor, probe, L/T, orientation hint."""
        test = data.get("test") or {}
        if not isinstance(test, dict):
            return
        date = _date_for(source, data)
        samples = test.get("samples")
        for key, metric in (
            ("accuracy", "accuracy"),
            ("acc", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("balanced_acc", "balanced_accuracy"),
        ):
            if key in test:
                self.add(
                    source=source, step_id=step_id, run_kind=kind, metric=metric,
                    value=test.get(key), n=samples, date=date,
                )
        if "best_val_macro_f1" in data:
            self.add(
                source=source, step_id=step_id, run_kind=kind, metric="macro_f1",
                value=data.get("best_val_macro_f1"), split="val", date=date,
            )
        for row in test.get("per_class") or []:
            if not isinstance(row, dict):
                continue
            vendor = canonical_vendor(row.get("manufacturer"))
            if not vendor:
                continue
            support = row.get("support")
            if not support:
                continue  # a class with no test support says nothing about the vendor
            for key in ("f1", "precision", "recall"):
                self.add(
                    source=source, step_id=step_id, run_kind=kind, metric=key,
                    value=row.get(key), vendor=vendor, n=support, date=date,
                )
            self.add(
                source=source, step_id=step_id, run_kind=kind, metric="support",
                value=support, vendor=vendor, date=date,
            )
        for row in data.get("per_manufacturer_test") or []:
            if not isinstance(row, dict):
                continue
            vendor = canonical_vendor(row.get("manufacturer"))
            if not vendor:
                continue
            for key, metric in (("accuracy", "accuracy"), ("macro_f1", "macro_f1")):
                self.add(
                    source=source, step_id=step_id, run_kind=kind, metric=metric,
                    value=row.get(key), vendor=vendor, n=row.get("samples"), date=date,
                )

    def adapt_rect(self, source: Path, data: Dict[str, Any], step_id: str) -> None:
        """Trainer metrics for a rect regressor (IoU based)."""
        test = data.get("test") or {}
        date = _date_for(source, data)
        self.add(
            source=source, step_id=step_id, run_kind="training", metric="mean_iou",
            value=test.get("mean_iou"), n=test.get("samples"), date=date,
        )
        self.add(
            source=source, step_id=step_id, run_kind="training", metric="best_val_iou",
            value=data.get("best_val_iou"), split="val", date=date,
        )
        for raw_vendor, iou in (test.get("per_manufacturer_iou") or {}).items():
            self.add(
                source=source, step_id=step_id, run_kind="training", metric="mean_iou",
                value=iou, vendor=raw_vendor, date=date,
            )

    def adapt_line16_params(self, source: Path, data: Dict[str, Any]) -> None:
        test = data.get("test") or {}
        date = _date_for(source, data)
        for key, metric in (("acc_mm", "accuracy"), ("balanced_acc_mm", "balanced_accuracy")):
            self.add(
                source=source, step_id="12_line16", run_kind="training", metric=metric,
                value=test.get(key), n=test.get("samples"), date=date,
            )

    def adapt_scale_ladder(self, source: Path, data: Dict[str, Any]) -> None:
        """`summary.json` of the classic scale/ladder eval: metrics.overall + by_vendor."""
        metrics = data.get("metrics") or {}
        date = _date_for(source, data)
        blocks: List[Tuple[Optional[str], Dict[str, Any]]] = []
        if isinstance(metrics.get("overall"), dict):
            blocks.append((None, metrics["overall"]))
        for raw_vendor, block in (metrics.get("by_vendor") or {}).items():
            if isinstance(block, dict):
                blocks.append((raw_vendor, block))
        for vendor, block in blocks:
            rows = block.get("rows")
            for key in ("strict_ok_pct", "calib_ok_pct", "direction_ok_pct"):
                self.add(
                    source=source, step_id="13_rect_depth", run_kind="eval", metric=key,
                    value=block.get(key), vendor=vendor, n=rows, date=date,
                )
            status = block.get("status") or {}
            total = sum(_num(v) or 0 for v in status.values())
            if total:
                self.add(
                    source=source, step_id="13_rect_depth", run_kind="eval", metric="accepted_pct",
                    value=100.0 * (_num(status.get("accepted")) or 0) / total,
                    vendor=vendor, n=int(total), date=date,
                )
                self.add(
                    source=source, step_id="13_rect_depth", run_kind="eval", metric="reject_pct",
                    value=100.0 * (_num(status.get("reject")) or 0) / total,
                    vendor=vendor, n=int(total), date=date,
                )
            for key, metric in (
                ("err_x_px", "err_x_px_median"),
                ("err_y_zero_px", "err_y_zero_px_median"),
                ("rel_err_mm_per_px", "rel_err_mm_per_px_median"),
            ):
                stats = block.get(key)
                if isinstance(stats, dict):
                    self.add(
                        source=source, step_id="13_rect_depth", run_kind="eval", metric=metric,
                        value=stats.get("median"), vendor=vendor, n=stats.get("n"), date=date,
                    )

    def adapt_scale_heatmap(self, source: Path, data: Dict[str, Any]) -> None:
        """`summary.json` of a scale heatmap model: vendor comes from the run directory."""
        test = data.get("test") or {}
        date = _date_for(source, data)
        vendor = canonical_vendor(source.parent.name)
        rows = test.get("rows")
        for key in (
            "direction_acc", "x_within_tol_pct", "y_within_tol_pct",
            "calib_within_tol_pct", "strict_ok_pct", "handoff_score",
        ):
            self.add(
                source=source, step_id="13_rect_depth", run_kind="training", metric=key,
                value=test.get(key), vendor=vendor, n=rows, date=date,
            )
        for key, metric in (
            ("err_x_px", "err_x_px_median"),
            ("err_y_zero_px", "err_y_zero_px_median"),
            ("rel_err_mm_per_px", "rel_err_mm_per_px_median"),
        ):
            stats = test.get(key)
            if isinstance(stats, dict):
                self.add(
                    source=source, step_id="13_rect_depth", run_kind="training", metric=metric,
                    value=stats.get("median"), vendor=vendor, n=rows, date=date,
                )

    def adapt_orientation_eval(self, source: Path, data: Dict[str, Any]) -> None:
        """`summary.json` of the LR-marker / line16 eval, plus its folder_summary.csv."""
        date = _date_for(source, data)
        step_id = step_from_path(source, "12_line16")
        folders = data.get("folders_total")
        for key, metric in (
            ("group_accuracy", "group_accuracy"),
            ("box_agree_rate", "box_agree_rate"),
            ("review_rate", "review_rate"),
            ("envelope_iou_median_excl_dup", "envelope_iou_median"),
            ("marker_in_gt_rate_overall", "marker_in_gt_rate"),
        ):
            self.add(
                source=source, step_id=step_id, run_kind="eval", metric=metric,
                value=data.get(key), n=folders, date=date,
            )
        folder_csv = source.parent / "folder_summary.csv"
        if folder_csv.exists():
            self.adapt_orientation_folder_csv(folder_csv, step_id, date, emit_overall=False)

    ORIENTATION_FOLDER_METRICS = ("accuracy", "box_agree_rate", "review_rate", "marker_found_rate")
    LINE16_GROUPS = ("NF", "LR", "UD", "LRUD")

    def adapt_orientation_folder_csv(
        self,
        source: Path,
        step_id: Optional[str],
        date: Optional[str] = None,
        emit_overall: bool = True,
    ) -> None:
        """Aggregate an orientation `folder_summary.csv`, overall and per vendor.

        `groups_complete_rate` is the share of folders where all four orientation
        groups were found: below 1.0 the pipeline cannot emit line #16 at all.
        """
        try:
            rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
        except OSError:
            return
        if not rows or "vendor" not in (rows[0] or {}):
            return
        step_id = step_id or step_from_path(source, "12_line16")
        date = date or _date_for(source)

        buckets: Dict[Optional[str], Dict[str, List[float]]] = {}

        def bucket_for(key: Optional[str]) -> Dict[str, List[float]]:
            return buckets.setdefault(
                key, {metric: [] for metric in self.ORIENTATION_FOLDER_METRICS + ("groups_complete_rate",)}
            )

        for row in rows:
            vendor = canonical_vendor(row.get("vendor")) or "sconosciuto"
            keys: List[Optional[str]] = [vendor] + ([None] if emit_overall else [])
            for key in keys:
                bucket = bucket_for(key)
                for metric in self.ORIENTATION_FOLDER_METRICS:
                    val = _num(row.get(metric))
                    if val is not None:
                        bucket[metric].append(val)
                if "groups_found" in row:
                    found = {g.strip().upper() for g in (row.get("groups_found") or "").split(",") if g.strip()}
                    bucket["groups_complete_rate"].append(1.0 if set(self.LINE16_GROUPS) <= found else 0.0)

        for vendor, bucket in buckets.items():
            for metric, values in bucket.items():
                if values:
                    self.add(
                        source=source, step_id=step_id, run_kind="eval", metric=metric,
                        value=sum(values) / len(values), vendor=vendor, n=len(values), date=date,
                    )

    def adapt_fss_head_run(self, source: Path, data: Dict[str, Any]) -> None:
        """A full pipeline run: gives every step a per-vendor health figure.

        The prediction CSV carries `vendor_predicted` and the comma separated
        `review_reasons`, so a folder can be charged to the step that flagged it.
        """
        date = _date_for(source, data)
        csv_path = source.parent / "folder_fss_head_predictions.csv"
        if not csv_path.exists():
            candidate = data.get("output_csv")
            csv_path = Path(candidate) if candidate else csv_path
        if not csv_path.exists():
            return
        try:
            rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        except OSError:
            return
        if not rows:
            return

        # totals[vendor] = folders, flagged[vendor][step] = folders flagged on that step
        totals: Dict[Optional[str], int] = {}
        ok_folders: Dict[Optional[str], int] = {}
        flagged: Dict[Optional[str], Dict[str, int]] = {}
        reason_counts: Dict[Tuple[Optional[str], str], int] = {}
        unmapped: Dict[str, int] = {}
        for row in rows:
            vendor = canonical_vendor(row.get("vendor_predicted")) or "sconosciuto"
            for key in (None, vendor):
                totals[key] = totals.get(key, 0) + 1
                if (row.get("status") or "").strip() == "ok":
                    ok_folders[key] = ok_folders.get(key, 0) + 1
            reasons = [r.strip() for r in (row.get("review_reasons") or "").split(",") if r.strip()]
            steps_hit = set()
            for reason in reasons:
                for key in (None, vendor):
                    reason_counts[(key, reason)] = reason_counts.get((key, reason), 0) + 1
                step_id = REVIEW_REASON_TO_STEP.get(reason)
                if step_id is None:
                    unmapped[reason] = unmapped.get(reason, 0) + 1
                    continue
                steps_hit.add(step_id)
            for step_id in steps_hit:
                for key in (None, vendor):
                    flagged.setdefault(key, {})
                    flagged[key][step_id] = flagged[key].get(step_id, 0) + 1

        label = None
        for vendor, total in totals.items():
            self.add(
                source=source, step_id="00_end_to_end", run_kind="pipeline", metric="folder_ok_rate",
                value=ok_folders.get(vendor, 0) / total, vendor=vendor, n=total,
                date=date, label=label,
            )
            self.add(
                source=source, step_id="00_end_to_end", run_kind="pipeline", metric="folders",
                value=total, vendor=vendor, date=date, label=label,
            )
            for step in STEP_BY_ID.values():
                if step["id"] in ("00_end_to_end", "15_writer"):
                    continue
                hit = flagged.get(vendor, {}).get(step["id"], 0)
                self.add(
                    source=source, step_id=step["id"], run_kind="pipeline", metric="step_ok_rate",
                    value=(total - hit) / total, vendor=vendor, n=total, date=date, label=label,
                )
                self.add(
                    source=source, step_id=step["id"], run_kind="pipeline", metric="folders",
                    value=total, vendor=vendor, date=date, label=label,
                )

        run_id, _, _, scope = _run_identity(source, self.artifacts_root)
        for (vendor, reason), count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            self.reasons.append(
                {
                    "run_id": run_id,
                    "scope": scope,
                    "date": date,
                    "step_id": REVIEW_REASON_TO_STEP.get(reason),
                    "vendor": vendor,
                    "reason": reason,
                    "folders": count,
                    "folders_total": totals.get(vendor, 0),
                }
            )
        for reason, count in unmapped.items():
            self.skipped.append(f"review reason non mappata ({count}x) in {source.parent.name}: {reason}")

    # -- dispatch ---------------------------------------------------------
    @staticmethod
    def _single_vendor_breakdown(data: Dict[str, Any]) -> Optional[str]:
        """The only vendor a run reports on, when there is exactly one."""
        test = data.get("test") if isinstance(data.get("test"), dict) else {}
        found = set()
        for raw in (test.get("per_manufacturer_iou") or {}):
            found.add(canonical_vendor(raw))
        for row in (test.get("per_class") or []):
            if isinstance(row, dict) and row.get("support") and row.get("manufacturer"):
                found.add(canonical_vendor(row["manufacturer"]))
        for row in (data.get("per_manufacturer_test") or []):
            if isinstance(row, dict) and row.get("manufacturer"):
                found.add(canonical_vendor(row["manufacturer"]))
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            for raw in (metrics.get("by_vendor") or {}):
                found.add(canonical_vendor(raw))
        found.discard(None)
        return next(iter(found)) if len(found) == 1 else None

    def handle_json(self, source: Path) -> None:
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._scope_vendor = run_vendor_scope(source) or self._single_vendor_breakdown(data)
        test = data.get("test") if isinstance(data.get("test"), dict) else {}

        # trainer shapes
        if "per_manufacturer_iou" in test or "best_val_iou" in data:
            self.adapt_rect(source, data, step_from_path(source, "10_line11"))
            return
        if "acc_mm" in test:
            self.adapt_line16_params(source, data)
            return
        if test and ({"accuracy", "acc", "macro_f1", "balanced_acc"} & set(test)):
            step_id = step_from_path(source)
            if step_id is None:
                per_class = test.get("per_class") or []
                has_vendor = any(isinstance(r, dict) and "manufacturer" in r for r in per_class)
                step_id = "05_vendor" if has_vendor else None
            if step_id:
                self.adapt_classifier(source, data, step_id, "training")
            else:
                self.skipped.append(f"step non riconosciuto per {source}")
            return

        # eval / run shapes
        if isinstance(data.get("metrics"), dict) and "by_vendor" in data["metrics"]:
            self.adapt_scale_ladder(source, data)
            return
        if "handoff_score" in test:
            self.adapt_scale_heatmap(source, data)
            return
        if "group_accuracy" in data or "box_agree_rate" in data:
            self.adapt_orientation_eval(source, data)
            return
        if "status_counts" in data and "folders_scanned" in data:
            self.adapt_fss_head_run(source, data)
            return

    def handle_csv(self, source: Path) -> None:
        self._scope_vendor = run_vendor_scope(source)
        if source.name == "folder_summary.csv":
            # only when it was not already pulled in by its summary.json sibling
            if not (source.parent / "summary.json").exists():
                self.adapt_orientation_folder_csv(source, None)
        elif source.name == "folder_fss_head_predictions.csv":
            if not (source.parent / "summary.json").exists():
                self.adapt_fss_head_run(source, {})

    def walk(self, max_depth: int) -> None:
        root_depth = len(self.artifacts_root.parts)
        for dirpath, dirnames, filenames in os.walk(self.artifacts_root):
            here = Path(dirpath)
            if len(here.parts) - root_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [
                d for d in dirnames
                if d not in PRUNE_DIR_NAMES and not any(p.search(d) for p in PRUNE_DIR_PATTERNS)
            ]
            for name in filenames:
                if name not in INTERESTING_FILES:
                    continue
                source = here / name
                if name.endswith(".json"):
                    self.handle_json(source)
                else:
                    self.handle_csv(source)


def collect(artifacts_root: Path, overrides: Dict[str, str], max_depth: int = 6) -> Collector:
    collector = Collector(artifacts_root, overrides)
    collector.walk(max_depth)
    collector.records.sort(key=lambda r: (r["step_id"], r["date"], r["run_id"], r["metric"], r["vendor"] or ""))
    return collector


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Raccoglie le metriche dei run in artifacts/ per il monitor di progetto")
    parser.add_argument("--artifacts-root", default="artifacts", help="cartella artifacts da scandire")
    parser.add_argument("--out-dir", default=None, help="default: <artifacts-root>/71_monitor")
    parser.add_argument("--overrides", default=None, help="JSON con override step:<run_id> -> step_id")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    artifacts_root = Path(args.artifacts_root).resolve()
    if not artifacts_root.is_dir():
        parser.error(f"artifacts root inesistente: {artifacts_root}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else artifacts_root / "71_monitor"

    overrides: Dict[str, str] = {}
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))

    collector = collect(artifacts_root, overrides, args.max_depth)
    n_metrics = write_jsonl(out_dir / "metrics.jsonl", collector.records)
    runs = sorted(collector.runs.values(), key=lambda r: (r["date"], r["run_id"]))
    n_runs = write_jsonl(out_dir / "runs.jsonl", runs)
    n_reasons = write_jsonl(out_dir / "review_reasons.jsonl", collector.reasons)
    (out_dir / "collect_report.json").write_text(
        json.dumps(
            {
                "artifacts_root": str(artifacts_root),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "records": n_metrics,
                "runs": n_runs,
                "review_reasons": n_reasons,
                "warnings": sorted(set(collector.skipped)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"metriche: {n_metrics}  run: {n_runs}  ->  {out_dir}")
        for warning in sorted(set(collector.skipped))[:10]:
            print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
