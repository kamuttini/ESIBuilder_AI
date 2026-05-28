#!/usr/bin/env python3
"""Build rect-adjustment targets from max-zoom segment selection + rect predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _norm(v: object) -> str:
    return str(v or "").strip().replace("\\", "/").lower()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [{str(k): str(v or "") for k, v in row.items()} for row in reader]


def _num(v: object, dflt: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return dflt


def run(args: argparse.Namespace) -> int:
    max_zoom_csv = args.max_zoom_csv.expanduser().resolve()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    summary_json = args.summary_json.expanduser().resolve()

    if not max_zoom_csv.exists():
        raise FileNotFoundError(f"max zoom csv non trovato: {max_zoom_csv}")
    if not predictions_csv.exists():
        raise FileNotFoundError(f"predictions csv non trovato: {predictions_csv}")

    sel = _read_csv_rows(max_zoom_csv)
    preds = _read_csv_rows(predictions_csv)

    by_rel: Dict[str, Dict[str, str]] = {}
    by_name: Dict[str, List[Dict[str, str]]] = {}
    for r in preds:
        rel = _norm(r.get("rel_path"))
        name = _norm(Path(str(r.get("rel_path") or "")).name)
        if rel and rel not in by_rel:
            by_rel[rel] = r
        if name:
            by_name.setdefault(name, []).append(r)

    out_rows: List[Dict[str, str]] = []
    missing: List[str] = []

    for row in sel:
        image_id = str(row.get("max_zoom_image_id") or "").strip()
        rel = str(row.get("max_zoom_rel_path") or "").strip().replace("\\", "/")
        key_rel = _norm(rel or image_id)
        p = by_rel.get(key_rel)
        if p is None:
            bname = _norm(Path(rel or image_id).name)
            cand = by_name.get(bname, [])
            if len(cand) == 1:
                p = cand[0]
        if p is None:
            missing.append(image_id or rel)
            continue

        out_rows.append(
            {
                "group_key": str(row.get("group_key") or ""),
                "group_name": str(row.get("group_name") or ""),
                "label": str(row.get("label") or ""),
                "max_zoom_image_id": image_id,
                "max_zoom_rel_path": rel,
                "segment_length_px": str(row.get("segment_length_px") or ""),
                "segment_length_norm": str(row.get("segment_length_norm") or ""),
                "image_path": str(p.get("image_path") or ""),
                "sample_id": str(p.get("sample_id") or ""),
                "pred_top": str(p.get("pred_top") or ""),
                "pred_left": str(p.get("pred_left") or ""),
                "pred_bottom": str(p.get("pred_bottom") or ""),
                "pred_right": str(p.get("pred_right") or ""),
                "pred_top_norm": str(p.get("pred_top_norm") or ""),
                "pred_left_norm": str(p.get("pred_left_norm") or ""),
                "pred_bottom_norm": str(p.get("pred_bottom_norm") or ""),
                "pred_right_norm": str(p.get("pred_right_norm") or ""),
                "global_top": str(p.get("global_top") or ""),
                "global_left": str(p.get("global_left") or ""),
                "global_bottom": str(p.get("global_bottom") or ""),
                "global_right": str(p.get("global_right") or ""),
                "global_iou": str(p.get("global_iou") or ""),
                "rect_route_type": str(p.get("rect_route_type") or ""),
                "vendor_predicted": str(p.get("vendor_predicted") or ""),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        cols = [
            "group_key",
            "group_name",
            "label",
            "max_zoom_image_id",
            "max_zoom_rel_path",
            "segment_length_px",
            "segment_length_norm",
            "image_path",
            "sample_id",
            "pred_top",
            "pred_left",
            "pred_bottom",
            "pred_right",
            "pred_top_norm",
            "pred_left_norm",
            "pred_bottom_norm",
            "pred_right_norm",
            "global_top",
            "global_left",
            "global_bottom",
            "global_right",
            "global_iou",
            "rect_route_type",
            "vendor_predicted",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    summary = {
        "max_zoom_csv": max_zoom_csv.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
        "targets_csv": output_csv.as_posix(),
        "selected_rows_input": len(sel),
        "targets_written": len(out_rows),
        "missing_predictions": len(missing),
        "missing_preview": missing[:20],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Targets CSV: {output_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Rows in: {len(sel)} | out: {len(out_rows)} | missing: {len(missing)}", flush=True)
    if out_rows:
        seg_vals = [_num(r.get("segment_length_px"), 0.0) for r in out_rows]
        print(f"Segment px range: min={min(seg_vals):.3f} max={max(seg_vals):.3f}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build rect adjustment targets from max zoom selection.")
    p.add_argument(
        "--max-zoom-csv",
        type=Path,
        default=Path(
            "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/segment_zoom_selection/max_zoom_by_group.csv"
        ),
    )
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/predictions.csv"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/segment_zoom_selection/rect_adjustment_targets.csv"
        ),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/segment_zoom_selection/rect_adjustment_targets_summary.json"
        ),
    )
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

