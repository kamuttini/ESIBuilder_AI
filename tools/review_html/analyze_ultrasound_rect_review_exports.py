#!/usr/bin/env python3
"""Analyze rect review exports (flags/corrections) against predictions.

Outputs:
- flagged_detailed.csv
- vendor_sample_by_group.csv
- summary.json
- review_analysis.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:  # noqa: BLE001
        return default


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    it = max(at, bt)
    il = max(al, bl)
    ib = min(ab, bb)
    ir = min(ar, br)
    ih = max(0, ib - it)
    iw = max(0, ir - il)
    inter = ih * iw
    area_a = max(0, ab - at) * max(0, ar - al)
    area_b = max(0, bb - bt) * max(0, br - bl)
    den = area_a + area_b - inter
    if den <= 0:
        return 0.0
    return float(inter / den)


def _select_subset(paths: Sequence[str], limit: int) -> List[str]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]
    total = len(paths)
    selected_idx: List[int] = []
    seen: set[int] = set()
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


def _load_predictions(predictions_csv: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with predictions_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            out[sample_id] = {
                "sample_id": sample_id,
                "image_path": str(row.get("image_path", "")),
                "rel_path": str(row.get("rel_path", "")),
                "group_name": str(row.get("group_name", "")),
                "orientation_name": str(row.get("orientation_name", "")),
                "vendor_predicted": str(row.get("vendor_predicted", "")),
                "vendor_top1_prob": _to_float(row.get("vendor_top1_prob", 0.0)),
                "vendor_margin_top1_top2": _to_float(row.get("vendor_margin_top1_top2", 0.0)),
                "vendor_vote_ratio": _to_float(row.get("vendor_vote_ratio", 0.0)),
                "global_iou": _to_float(row.get("global_iou", 0.0)),
                "width": _to_int(row.get("width", 0)),
                "height": _to_int(row.get("height", 0)),
                "pred_top": _to_int(row.get("pred_top", 0)),
                "pred_left": _to_int(row.get("pred_left", 0)),
                "pred_bottom": _to_int(row.get("pred_bottom", 0)),
                "pred_right": _to_int(row.get("pred_right", 0)),
                "global_top": _to_int(row.get("global_top", 0)),
                "global_left": _to_int(row.get("global_left", 0)),
                "global_bottom": _to_int(row.get("global_bottom", 0)),
                "global_right": _to_int(row.get("global_right", 0)),
            }
    return out


def _read_export_csv(path: Path) -> Tuple[str, List[Dict[str, str]]]:
    kind = "unknown"
    name = path.name.lower()
    if name.startswith("flags_"):
        kind = "flags"
    if name.startswith("corrections_"):
        kind = "corrections"
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return kind, [dict(r) for r in reader]


def _collect_exports(exports_dir: Path) -> Tuple[List[Path], List[Path]]:
    flags_files = sorted(exports_dir.rglob("flags_*.csv"))
    corr_files = sorted(exports_dir.rglob("corrections_*.csv"))
    return flags_files, corr_files


def _build_rows(
    predictions: Dict[str, Dict[str, object]],
    flags_files: Sequence[Path],
    corr_files: Sequence[Path],
) -> List[Dict[str, object]]:
    by_sample: Dict[str, Dict[str, object]] = {}

    for path in flags_files:
        _kind, rows = _read_export_csv(path)
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            current = by_sample.setdefault(sample_id, {})
            current["sample_id"] = sample_id
            current["flagged"] = 1
            current["flag_note"] = str(row.get("note", ""))
            current["source_flag_file"] = path.as_posix()
            # keep exported geometry if available
            for key in (
                "vendor",
                "group_name",
                "image_path",
                "rel_path",
                "pred_top",
                "pred_left",
                "pred_bottom",
                "pred_right",
                "global_top",
                "global_left",
                "global_bottom",
                "global_right",
            ):
                if key in row and row[key] != "":
                    current[key] = row[key]

    for path in corr_files:
        _kind, rows = _read_export_csv(path)
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            current = by_sample.setdefault(sample_id, {})
            current["sample_id"] = sample_id
            current["flagged"] = 1
            current["has_correction"] = 1
            current["correction_note"] = str(row.get("note", ""))
            current["source_correction_file"] = path.as_posix()
            for key in (
                "vendor",
                "group_name",
                "image_path",
                "rel_path",
                "pred_top",
                "pred_left",
                "pred_bottom",
                "pred_right",
                "global_top",
                "global_left",
                "global_bottom",
                "global_right",
            ):
                if key in row and row[key] != "":
                    current[key] = row[key]
            for key in ("corr_top", "corr_left", "corr_bottom", "corr_right"):
                if key in row and row[key] != "":
                    current[key] = row[key]

    out_rows: List[Dict[str, object]] = []
    for sample_id, exp in by_sample.items():
        pred = predictions.get(sample_id)
        if pred is None:
            row = dict(exp)
            row["in_predictions"] = 0
            out_rows.append(row)
            continue

        row: Dict[str, object] = dict(exp)
        row["in_predictions"] = 1
        for k, v in pred.items():
            row.setdefault(k, v)

        pred_box = (
            _to_int(row.get("pred_top", pred["pred_top"])),
            _to_int(row.get("pred_left", pred["pred_left"])),
            _to_int(row.get("pred_bottom", pred["pred_bottom"])),
            _to_int(row.get("pred_right", pred["pred_right"])),
        )
        glob_box = (
            _to_int(row.get("global_top", pred["global_top"])),
            _to_int(row.get("global_left", pred["global_left"])),
            _to_int(row.get("global_bottom", pred["global_bottom"])),
            _to_int(row.get("global_right", pred["global_right"])),
        )
        row["iou_pred_vs_global"] = _box_iou(pred_box, glob_box)

        if all(str(row.get(k, "")).strip() for k in ("corr_top", "corr_left", "corr_bottom", "corr_right")):
            corr_box = (
                _to_int(row.get("corr_top", 0)),
                _to_int(row.get("corr_left", 0)),
                _to_int(row.get("corr_bottom", 0)),
                _to_int(row.get("corr_right", 0)),
            )
            row["has_correction"] = 1
            row["iou_pred_vs_corr"] = _box_iou(pred_box, corr_box)
            row["iou_global_vs_corr"] = _box_iou(glob_box, corr_box)
            row["delta_pred_to_corr_abs_sum"] = (
                abs(pred_box[0] - corr_box[0])
                + abs(pred_box[1] - corr_box[1])
                + abs(pred_box[2] - corr_box[2])
                + abs(pred_box[3] - corr_box[3])
            )
        else:
            row["has_correction"] = int(row.get("has_correction", 0) or 0)
            row["iou_pred_vs_corr"] = None
            row["iou_global_vs_corr"] = None
            row["delta_pred_to_corr_abs_sum"] = None

        out_rows.append(row)

    out_rows.sort(
        key=lambda r: (
            str(r.get("vendor_predicted", "")),
            str(r.get("group_name", "")),
            str(r.get("sample_id", "")),
        )
    )
    return out_rows


def _build_vendor_samples(
    predictions: Dict[str, Dict[str, object]],
    group_names: Sequence[str],
    vendor_sample_per_group: int,
) -> Tuple[Dict[str, set[str]], List[Dict[str, object]]]:
    paths_by_group: Dict[str, List[str]] = defaultdict(list)
    for p in predictions.values():
        group_name = str(p.get("group_name", ""))
        image_path = str(p.get("image_path", ""))
        if not group_name or not image_path:
            continue
        paths_by_group[group_name].append(image_path)

    selected_sets: Dict[str, set[str]] = {}
    selected_rows: List[Dict[str, object]] = []
    for g in sorted(set(group_names)):
        unique_paths = sorted(set(paths_by_group.get(g, [])))
        selected = _select_subset(unique_paths, vendor_sample_per_group)
        selected_sets[g] = set(selected)
        for rank, path in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "group_name": g,
                    "sample_rank": rank,
                    "image_path": path,
                }
            )
    return selected_sets, selected_rows


def _write_csv(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _safe_stat(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(vals)
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _html_report(rows: Sequence[Dict[str, object]], summary: Dict[str, object], out_html: Path) -> None:
    body_rows: List[str] = []
    for row in rows:
        image_path = str(row.get("image_path", ""))
        image_uri = Path(image_path).expanduser().resolve().as_uri() if image_path else ""
        has_corr = int(row.get("has_correction", 0) or 0)
        sampled = int(row.get("sampled_for_vendor", 0) or 0)
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('sample_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('vendor_predicted', '')))}</td>"
            f"<td>{html.escape(str(row.get('group_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('orientation_name', '')))}</td>"
            f"<td>{_to_float(row.get('vendor_top1_prob', 0.0)):.3f}</td>"
            f"<td>{_to_float(row.get('vendor_margin_top1_top2', 0.0)):.3f}</td>"
            f"<td>{_to_float(row.get('vendor_vote_ratio', 0.0)):.3f}</td>"
            f"<td>{_to_float(row.get('global_iou', 0.0)):.3f}</td>"
            f"<td>{'' if row.get('iou_pred_vs_corr') is None else f'{_to_float(row.get('iou_pred_vs_corr')):.3f}'}</td>"
            f"<td>{'' if row.get('iou_global_vs_corr') is None else f'{_to_float(row.get('iou_global_vs_corr')):.3f}'}</td>"
            f"<td>{sampled}</td>"
            f"<td>{has_corr}</td>"
            f"<td>{html.escape(str(row.get('flag_note', '') or row.get('correction_note', '')))}</td>"
            f"<td><a href=\"{html.escape(image_uri)}\" target=\"_blank\">open</a></td>"
            f"<td>{f'<img src=\"{html.escape(image_uri)}\" loading=\"lazy\"/>' if image_uri else ''}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Rect Review Export Analysis</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: #f3f5f4;
      color: #102119;
    }}
    .wrap {{ padding: 16px; }}
    pre {{
      background: #0f241a;
      color: #d8f6e8;
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #d6e2db;
    }}
    th, td {{
      border: 1px solid #e5ece8;
      padding: 6px;
      font-size: 12px;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef3ef;
      z-index: 2;
    }}
    img {{
      width: 160px;
      border: 1px solid #d6e2db;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Analisi export review rettangoli</h1>
    <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
    <table>
      <thead>
        <tr>
          <th>sample_id</th>
          <th>vendor</th>
          <th>group</th>
          <th>orient</th>
          <th>vendor_p</th>
          <th>vendor_margin</th>
          <th>vendor_vote</th>
          <th>iou_pred_global</th>
          <th>iou_pred_corr</th>
          <th>iou_global_corr</th>
          <th>in_vendor_sample</th>
          <th>has_corr</th>
          <th>note</th>
          <th>img</th>
          <th>preview</th>
        </tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analizza export flags/corrections della review rect e produce report con confidenze e campioni vendor."
    )
    p.add_argument("--exports-dir", type=Path, required=True, help="Cartella con flags_*.csv e corrections_*.csv.")
    p.add_argument("--predictions-csv", type=Path, required=True, help="predictions.csv del run rect.")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/40_outputs_eval/rect_review_exports_analysis"))
    p.add_argument(
        "--vendor-sample-per-group",
        type=int,
        default=64,
        help="Valore usato in inferenza vendor (default 64).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    exports_dir = args.exports_dir.expanduser().resolve()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exports_dir.is_dir():
        raise FileNotFoundError(f"exports-dir not found: {exports_dir}")
    if not predictions_csv.exists():
        raise FileNotFoundError(f"predictions-csv not found: {predictions_csv}")

    flags_files, corr_files = _collect_exports(exports_dir)
    if not flags_files and not corr_files:
        raise RuntimeError(f"No export files found in {exports_dir} (flags_*.csv / corrections_*.csv).")

    predictions = _load_predictions(predictions_csv)
    rows = _build_rows(predictions=predictions, flags_files=flags_files, corr_files=corr_files)
    if not rows:
        raise RuntimeError("No rows parsed from exports.")

    groups = [str(r.get("group_name", "")) for r in rows if str(r.get("group_name", ""))]
    sampled_sets, sampled_rows = _build_vendor_samples(
        predictions=predictions,
        group_names=groups,
        vendor_sample_per_group=int(args.vendor_sample_per_group),
    )

    for row in rows:
        group_name = str(row.get("group_name", ""))
        image_path = str(row.get("image_path", ""))
        selected = sampled_sets.get(group_name, set())
        row["sampled_for_vendor"] = 1 if image_path in selected else 0

    vendor_counts = Counter(str(r.get("vendor_predicted", "")) for r in rows)
    group_counts = Counter(str(r.get("group_name", "")) for r in rows)
    vendor_p = [_to_float(r.get("vendor_top1_prob", 0.0)) for r in rows if r.get("in_predictions") == 1]
    vendor_margin = [_to_float(r.get("vendor_margin_top1_top2", 0.0)) for r in rows if r.get("in_predictions") == 1]
    vendor_vote = [_to_float(r.get("vendor_vote_ratio", 0.0)) for r in rows if r.get("in_predictions") == 1]
    iou_pg = [_to_float(r.get("iou_pred_vs_global", 0.0)) for r in rows if r.get("in_predictions") == 1]
    iou_pc = [_to_float(r.get("iou_pred_vs_corr", 0.0)) for r in rows if r.get("iou_pred_vs_corr") is not None]
    iou_gc = [_to_float(r.get("iou_global_vs_corr", 0.0)) for r in rows if r.get("iou_global_vs_corr") is not None]

    summary = {
        "exports_dir": exports_dir.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
        "num_flag_files": len(flags_files),
        "num_correction_files": len(corr_files),
        "num_rows_total": len(rows),
        "num_rows_with_prediction_match": sum(int(r.get("in_predictions", 0) or 0) for r in rows),
        "num_rows_with_correction": sum(int(r.get("has_correction", 0) or 0) for r in rows),
        "sampled_for_vendor_count": sum(int(r.get("sampled_for_vendor", 0) or 0) for r in rows),
        "vendors": dict(sorted(vendor_counts.items())),
        "groups_top20": [{"group_name": g, "count": c} for g, c in group_counts.most_common(20)],
        "stats_vendor_top1_prob": _safe_stat(vendor_p),
        "stats_vendor_margin": _safe_stat(vendor_margin),
        "stats_vendor_vote_ratio": _safe_stat(vendor_vote),
        "stats_iou_pred_vs_global": _safe_stat(iou_pg),
        "stats_iou_pred_vs_corr": _safe_stat(iou_pc),
        "stats_iou_global_vs_corr": _safe_stat(iou_gc),
        "notes": [
            "sampled_for_vendor=1 significa che l'immagine era nel subset usato per classificare il vendor della cartella.",
            "iou_pred_vs_corr basso indica che la rete rect era lontana dalla correzione manuale.",
            "iou_global_vs_corr basso indica che anche il box globale di cartella era lontano dalla correzione.",
        ],
    }

    detailed_csv = output_dir / "flagged_detailed.csv"
    sample_csv = output_dir / "vendor_sample_by_group.csv"
    summary_json = output_dir / "summary.json"
    report_html = output_dir / "review_analysis.html"

    _write_csv(
        rows=rows,
        path=detailed_csv,
        fieldnames=[
            "sample_id",
            "in_predictions",
            "flagged",
            "has_correction",
            "vendor_predicted",
            "group_name",
            "orientation_name",
            "image_path",
            "rel_path",
            "vendor_top1_prob",
            "vendor_margin_top1_top2",
            "vendor_vote_ratio",
            "global_iou",
            "pred_top",
            "pred_left",
            "pred_bottom",
            "pred_right",
            "global_top",
            "global_left",
            "global_bottom",
            "global_right",
            "corr_top",
            "corr_left",
            "corr_bottom",
            "corr_right",
            "iou_pred_vs_global",
            "iou_pred_vs_corr",
            "iou_global_vs_corr",
            "delta_pred_to_corr_abs_sum",
            "sampled_for_vendor",
            "flag_note",
            "correction_note",
            "source_flag_file",
            "source_correction_file",
        ],
    )
    _write_csv(rows=sampled_rows, path=sample_csv, fieldnames=["group_name", "sample_rank", "image_path"])
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _html_report(rows=rows, summary=summary, out_html=report_html)

    print(f"Detailed CSV: {detailed_csv}", flush=True)
    print(f"Vendor sample CSV: {sample_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    print(f"HTML report: {report_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
