#!/usr/bin/env python3
"""Aggregate per-frame symbol detections into per-orientation envelope rectangles.

Input options:
1) detections CSV (preferred):
   columns required:
   - setup_id
   - orientation_idx (0..3)
   - x1,y1,x2,y2
   optional: score

2) frame manifest fallback:
   uses target_rect_* from frames with orientation_hint as pseudo-detections

Output:
- aggregated_orientation_envelopes.csv
- optional line16_preview.csv with reconstructed line16 strings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ORIENTATION_NAMES = {0: "NF", 1: "LR", 2: "UD", 3: "LRUD"}


@dataclass
class Det:
    setup_id: str
    orientation_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class SetupOrientParam:
    setup_id: str
    split: str
    dataset_folder: str
    manufacturer: str
    model_name: str
    fss_path: str
    orientation_idx: int
    orientation_name: str
    orientation_available: int
    rect_top: int
    rect_left: int
    rect_bottom: int
    rect_right: int
    b_value: int
    channel: int
    threshold: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    mm: int
    fss_video_x: int
    fss_video_y: int


@dataclass
class AggOut:
    setup_id: str
    orientation_idx: int
    source: str
    num_boxes: int
    top: int
    left: int
    bottom: int
    right: int


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp_box(x1: float, y1: float, x2: float, y2: float, vx: int, vy: int) -> Tuple[int, int, int, int]:
    x1 = max(0.0, min(x1, vx - 1.0))
    y1 = max(0.0, min(y1, vy - 1.0))
    x2 = max(1.0, min(x2, float(vx)))
    y2 = max(1.0, min(y2, float(vy)))
    if x2 <= x1:
        x2 = min(float(vx), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(vy), y1 + 1.0)
    return int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))


def _load_setup_params(path: Path) -> Dict[Tuple[str, int], SetupOrientParam]:
    out: Dict[Tuple[str, int], SetupOrientParam] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            idx = _parse_int(row["orientation_idx"], -1)
            if idx < 0 or idx > 3:
                continue
            key = (row["setup_id"], idx)
            out[key] = SetupOrientParam(
                setup_id=row["setup_id"],
                split=row["split"],
                dataset_folder=row["dataset_folder"],
                manufacturer=row["manufacturer"],
                model_name=row["model_name"],
                fss_path=row["fss_path"],
                orientation_idx=idx,
                orientation_name=row["orientation_name"],
                orientation_available=_parse_int(row["orientation_available"], 1),
                rect_top=_parse_int(row.get("rect_top", "0"), 0),
                rect_left=_parse_int(row.get("rect_left", "0"), 0),
                rect_bottom=_parse_int(row.get("rect_bottom", "1"), 1),
                rect_right=_parse_int(row.get("rect_right", "1"), 1),
                b_value=_parse_int(row["b_value"], 1),
                channel=_parse_int(row["channel"], 7),
                threshold=_parse_float(row["threshold"], 100.0),
                p1=_parse_float(row["p1"], 20.0),
                p2=_parse_float(row["p2"], 120.0),
                p3=_parse_float(row["p3"], 0.0),
                p4=_parse_float(row["p4"], 0.0),
                p5=_parse_float(row["p5"], 0.0),
                mm=_parse_int(row["mm"], 6),
                fss_video_x=_parse_int(row["fss_video_x"], 1920),
                fss_video_y=_parse_int(row["fss_video_y"], 1080),
            )
    return out


def _load_detections_csv(path: Path, min_score: float) -> List[Det]:
    out: List[Det] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            score = _parse_float(row.get("score", "1"), 1.0)
            if score < min_score:
                continue
            idx = _parse_int(row.get("orientation_idx", "-1"), -1)
            if idx < 0 or idx > 3:
                continue
            out.append(
                Det(
                    setup_id=row["setup_id"],
                    orientation_idx=idx,
                    x1=_parse_float(row["x1"], 0.0),
                    y1=_parse_float(row["y1"], 0.0),
                    x2=_parse_float(row["x2"], 1.0),
                    y2=_parse_float(row["y2"], 1.0),
                    score=score,
                )
            )
    return out


def _load_detections_from_frame_manifest(path: Path) -> List[Det]:
    out: List[Det] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("orientation_hint", "") == "":
                continue
            idx = _parse_int(row["orientation_hint"], -1)
            if idx < 0 or idx > 3:
                continue
            if row.get("target_rect_left", "") == "":
                continue
            out.append(
                Det(
                    setup_id=row["setup_id"],
                    orientation_idx=idx,
                    x1=_parse_float(row["target_rect_left"], 0.0),
                    y1=_parse_float(row["target_rect_top"], 0.0),
                    x2=_parse_float(row["target_rect_right"], 1.0),
                    y2=_parse_float(row["target_rect_bottom"], 1.0),
                    score=1.0,
                )
            )
    return out


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def _aggregate_group(
    dets: Sequence[Det],
    vx: int,
    vy: int,
    q_low: float,
    q_high: float,
    margin_px: int,
) -> Tuple[int, int, int, int]:
    xs1 = [d.x1 for d in dets]
    ys1 = [d.y1 for d in dets]
    xs2 = [d.x2 for d in dets]
    ys2 = [d.y2 for d in dets]

    left = _quantile(xs1, q_low) - margin_px
    top = _quantile(ys1, q_low) - margin_px
    right = _quantile(xs2, q_high) + margin_px
    bottom = _quantile(ys2, q_high) + margin_px

    return _clamp_box(left, top, right, bottom, vx=vx, vy=vy)


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return f"{float(round(v)):.1f}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _build_line16(setup_id: str, params: Dict[Tuple[str, int], SetupOrientParam], envs: Dict[int, AggOut]) -> str:
    parts: List[str] = []
    for idx in range(4):
        p = params[(setup_id, idx)]
        e = envs[idx]
        p3 = 0.0
        p4 = 0.0
        p5 = 0.0
        part = (
            f"{e.top}|{e.left}|{e.bottom}|{e.right}|{p.b_value}|"
            f"{p.channel}:{_fmt(p.threshold)}:{_fmt(p.p1)}:{_fmt(p.p2)}:{_fmt(p3)}:{_fmt(p4)}:{_fmt(p5)}|{p.mm}|"
        )
        parts.append(part)
    return ";".join(parts) + ";"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-frame detections into orientation envelopes.")
    parser.add_argument(
        "--setup-orientation-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbolic_dataset/setup_orientation_manifest.csv"),
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help="Per-frame detections with setup_id/orientation_idx/x1/y1/x2/y2[,score]",
    )
    parser.add_argument(
        "--frame-manifest-fallback",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/99_misc/orientation_envelope_aggregation"),
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-boxes-per-group", type=int, default=3)
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--margin-px", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    setup_manifest = args.setup_orientation_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _load_setup_params(setup_manifest)
    setups = sorted({k[0] for k in params})

    if args.detections_csv is not None and args.detections_csv.exists():
        detections = _load_detections_csv(args.detections_csv.expanduser().resolve(), min_score=args.min_score)
        det_source = "detections_csv"
    else:
        detections = _load_detections_from_frame_manifest(args.frame_manifest_fallback.expanduser().resolve())
        det_source = "frame_manifest_fallback"

    grouped: Dict[Tuple[str, int], List[Det]] = {}
    for d in detections:
        grouped.setdefault((d.setup_id, d.orientation_idx), []).append(d)

    out_rows: List[Dict[str, object]] = []
    line16_rows: List[Dict[str, object]] = []

    for setup_id in setups:
        envs: Dict[int, AggOut] = {}
        for idx in range(4):
            p = params[(setup_id, idx)]
            dets = grouped.get((setup_id, idx), [])
            if len(dets) >= args.min_boxes_per_group:
                top, left, bottom, right = _aggregate_group(
                    dets,
                    vx=max(1, p.fss_video_x),
                    vy=max(1, p.fss_video_y),
                    q_low=max(0.0, min(1.0, args.q_low)),
                    q_high=max(0.0, min(1.0, args.q_high)),
                    margin_px=max(0, args.margin_px),
                )
                source = "aggregated"
                num_boxes = len(dets)
            else:
                # fallback to legacy envelope
                # if sparse detections exist use them; otherwise use legacy line16 rect.
                if dets:
                    # with sparse boxes use direct min/max
                    t = min(d.y1 for d in dets)
                    l = min(d.x1 for d in dets)
                    b = max(d.y2 for d in dets)
                    r = max(d.x2 for d in dets)
                    top, left, bottom, right = _clamp_box(l, t, r, b, vx=max(1, p.fss_video_x), vy=max(1, p.fss_video_y))
                    source = "sparse"
                    num_boxes = len(dets)
                else:
                    top, left, bottom, right = p.rect_top, p.rect_left, p.rect_bottom, p.rect_right
                    source = "legacy"
                    num_boxes = 0

            env = AggOut(
                setup_id=setup_id,
                orientation_idx=idx,
                source=source,
                num_boxes=num_boxes,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
            )
            envs[idx] = env

            out_rows.append(
                {
                    "setup_id": setup_id,
                    "split": p.split,
                    "dataset_folder": p.dataset_folder,
                    "manufacturer": p.manufacturer,
                    "model_name": p.model_name,
                    "orientation_idx": idx,
                    "orientation_name": ORIENTATION_NAMES[idx],
                    "orientation_available": p.orientation_available,
                    "source": source,
                    "num_boxes": num_boxes,
                    "rect_top": top,
                    "rect_left": left,
                    "rect_bottom": bottom,
                    "rect_right": right,
                    "fss_video_x": p.fss_video_x,
                    "fss_video_y": p.fss_video_y,
                    "fss_path": p.fss_path,
                }
            )

        line16_rows.append(
            {
                "setup_id": setup_id,
                "dataset_folder": params[(setup_id, 0)].dataset_folder,
                "manufacturer": params[(setup_id, 0)].manufacturer,
                "model_name": params[(setup_id, 0)].model_name,
                "fss_path": params[(setup_id, 0)].fss_path,
                "line16_aggregated": _build_line16(setup_id, params, envs),
            }
        )

    agg_csv = output_dir / "aggregated_orientation_envelopes.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "setup_id",
                "split",
                "dataset_folder",
                "manufacturer",
                "model_name",
                "orientation_idx",
                "orientation_name",
                "orientation_available",
                "source",
                "num_boxes",
                "rect_top",
                "rect_left",
                "rect_bottom",
                "rect_right",
                "fss_video_x",
                "fss_video_y",
                "fss_path",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    line16_csv = output_dir / "line16_preview.csv"
    with line16_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["setup_id", "dataset_folder", "manufacturer", "model_name", "fss_path", "line16_aggregated"],
        )
        writer.writeheader()
        writer.writerows(line16_rows)

    summary = {
        "detection_source": det_source,
        "num_setups": len(setups),
        "num_input_detections": len(detections),
        "num_rows_output": len(out_rows),
        "min_boxes_per_group": args.min_boxes_per_group,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "margin_px": args.margin_px,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Detection source: {det_source}", flush=True)
    print(f"Input detections: {len(detections)}", flush=True)
    print(f"Aggregated CSV: {agg_csv}", flush=True)
    print(f"Line16 preview CSV: {line16_csv}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
