#!/usr/bin/env python3
"""Baseline for line #14 RECT_NAME_PROBE: the historical resolver, on dataset splits.

Reproduces the key priority of ``RectNameProbeResolver`` in
``predict_fss_head_from_acquisitions.py`` (vendor+probe+video -> vendor+probe ->
probe+video -> probe, no blind fallback) using the train split as history and the
eval split as new machines. Since the splits are leak-free at layout level, this
measures exactly the case the detector is meant to cover: a machine never configured
before.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from prepare_probe_template_dataset import parse_fss

KEY_NAMES = ("vendor_probe_video", "vendor_probe", "probe_video", "probe")


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def enrich(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        meta = parse_fss(Path(row["fss_path"]))
        row["probe"] = str(meta["fss_id_probe"] or "")
        row["vx"] = meta["fss_video_x"]
        row["vy"] = meta["fss_video_y"]
        box = meta["box14"]
        row["box"] = (box[1], box[0], box[3], box[2]) if box else None  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folders-csv", type=Path, required=True)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--history-split", type=str, default="train")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.folders_csv.open(encoding="utf-8")))
    enrich(rows)
    history = [r for r in rows if r["split"] == args.history_split and r["box"]]
    evaluation = [r for r in rows if r["split"] == args.eval_split]

    tables: List[Dict[Tuple, collections.Counter]] = [collections.defaultdict(collections.Counter) for _ in KEY_NAMES]
    for r in history:
        tables[0][(r["manufacturer"], r["probe"], r["vx"], r["vy"])][r["box"]] += 1
        tables[1][(r["manufacturer"], r["probe"])][r["box"]] += 1
        tables[2][(r["probe"], r["vx"], r["vy"])][r["box"]] += 1
        tables[3][r["probe"]][r["box"]] += 1

    results = []
    sources: collections.Counter = collections.Counter()
    for r in evaluation:
        keys = [
            (r["manufacturer"], r["probe"], r["vx"], r["vy"]),
            (r["manufacturer"], r["probe"]),
            (r["probe"], r["vx"], r["vy"]),
            r["probe"],
        ]
        chosen = None
        for i, key in enumerate(keys):
            counter = tables[i].get(key)
            if counter:
                chosen = (counter.most_common(1)[0][0], KEY_NAMES[i])
                break
        if chosen is None:
            sources["empty_line"] += 1
            results.append({"folder": r["folder"], "vendor": r["manufacturer"], "source": "empty_line", "iou": 0.0})
        else:
            sources[chosen[1]] += 1
            results.append({
                "folder": r["folder"], "vendor": r["manufacturer"], "source": chosen[1],
                "iou": iou_xyxy(chosen[0], r["box"]),
                "exact": int(chosen[0] == r["box"]),
            })

    n = len(results)
    ious = [float(r["iou"]) for r in results]
    covered = [r for r in results if r["source"] != "empty_line"]
    summary = {
        "eval_split": args.eval_split,
        "folders": n,
        "coverage": len(covered) / n if n else 0.0,
        "sources": dict(sources),
        "iou_median": statistics.median(ious) if ious else 0.0,
        "iou_ge_0.5": sum(i >= 0.5 for i in ious) / n if n else 0.0,
        "iou_ge_0.75": sum(i >= 0.75 for i in ious) / n if n else 0.0,
        "exact_lines": sum(int(r.get("exact", 0)) for r in covered),
        "per_vendor": {},
    }
    by_vendor: Dict[str, List[float]] = collections.defaultdict(list)
    for r in results:
        by_vendor[str(r["vendor"])].append(float(r["iou"]))
    for vendor, values in sorted(by_vendor.items(), key=lambda kv: -len(kv[1])):
        summary["per_vendor"][vendor] = {
            "folders": len(values),
            "iou_median": statistics.median(values),
            "iou_ge_0.5": sum(i >= 0.5 for i in values) / len(values),
        }

    print(f"RESOLVER STORICO su split '{args.eval_split}' ({n} cartelle, storia = '{args.history_split}')")
    print(f"  copertura           : {summary['coverage']*100:.0f}%   fonti: {summary['sources']}")
    print(f"  IoU mediana         : {summary['iou_median']:.3f}   >=0.5 {summary['iou_ge_0.5']*100:.0f}%   "
          f">=0.75 {summary['iou_ge_0.75']*100:.0f}%")
    print(f"  righe identiche     : {summary['exact_lines']}/{len(covered)}")
    for vendor, stats in summary["per_vendor"].items():
        print(f"    {vendor:10s} n={stats['folders']:2d}  IoU mediana {stats['iou_median']:.3f}  "
              f">=0.5 {stats['iou_ge_0.5']*100:3.0f}%")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8")
        print(f"-> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
