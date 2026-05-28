#!/usr/bin/env python3
"""Build an HTML pre-training review for scale GT quality and split sanity."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _f(text: str) -> float:
    return float(text.strip())


def _i(text: str) -> int:
    return int(float(text.strip()))


def _esc(v: object) -> str:
    return html.escape(str(v))


def _table(headers: List[str], rows: List[List[object]], cls: str = "") -> str:
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>")
    return (
        f'<table class="{cls}"><thead><tr>{head_html}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build pre-training HTML review for scale dataset.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/23_scale_manifest_ssd_esi1_n1_leakfree/manifest_scale_train_val_test_leakfree.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/24_scale_pretrain_review"),
    )
    p.add_argument("--max-anomaly-rows", type=int, default=300)
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError("Manifest is empty")

    by_fss: Dict[str, List[dict]] = defaultdict(list)
    split_rows = Counter()
    split_groups: Dict[str, set] = defaultdict(set)
    setup_splits: Dict[str, set] = defaultdict(set)
    hash_splits: Dict[str, set] = defaultdict(set)
    path_splits: Dict[str, set] = defaultdict(set)
    side_dist = Counter()
    tick_dist = Counter()
    res_dist = Counter()

    for r in rows:
        sp = r["split"]
        path = r["fss_path"]
        setup = r.get("setup_id", "")
        c_hash = r.get("fss_content_hash", "")
        split_rows[sp] += 1
        split_groups[sp].add(path)
        path_splits[path].add(sp)
        if setup:
            setup_splits[setup].add(sp)
        if c_hash:
            hash_splits[c_hash].add(sp)
        by_fss[path].append(r)
        side_dist[r.get("label_side", "")] += 1
        tick_dist[r.get("tick_mm", "")] += 1
        res_dist[f"{r.get('video_x_size','?')}x{r.get('video_y_size','?')}"] += 1

    # Fallback content hash leakage check if column is missing
    if not hash_splits:
        for p, sps in path_splits.items():
            h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            hash_splits[h].update(sps)

    setup_leak = sum(1 for s in setup_splits.values() if len(s) > 1)
    hash_leak = sum(1 for s in hash_splits.values() if len(s) > 1)
    path_leak = sum(1 for s in path_splits.values() if len(s) > 1)

    # Row-level stats
    y_down = 0
    y_up = 0
    deltas: List[float] = []
    delta_rows: List[Tuple[float, dict]] = []
    x_outlier_rows: List[Tuple[float, dict, float, float]] = []
    spread_rows: List[Tuple[float, str, int, int]] = []
    candidate_reasons: Dict[str, set] = defaultdict(set)

    for r in rows:
        y1 = _f(r["y1"])
        y2 = _f(r["y2"])
        if y2 >= y1:
            y_down += 1
        else:
            y_up += 1
        delta = abs(_f(r["length_mm"]) - _f(r["depth_mm"]))
        deltas.append(delta)
        delta_rows.append((delta, r))

    # Group-level spread / robust x outlier
    for path, grp in by_fss.items():
        xc = [(_f(r["x1"]) + _f(r["x2"])) * 0.5 for r in grp]
        if len(xc) >= 2:
            spread = max(xc) - min(xc)
            spread_rows.append((spread, path, len(grp), len({r["depth_index"] for r in grp})))
            if spread > 120:
                for r in grp:
                    candidate_reasons[r["sample_id"]].add("GROUP_X_SPREAD_GT120")

        if len(xc) >= 5:
            med = statistics.median(xc)
            mad = statistics.median(abs(x - med) for x in xc)
            if mad < 1.0:
                mad = 1.0
            for r in grp:
                x = (_f(r["x1"]) + _f(r["x2"])) * 0.5
                dist = abs(x - med)
                if dist > (6.0 * mad) and dist > 20.0:
                    x_outlier_rows.append((dist, r, med, mad))
                    candidate_reasons[r["sample_id"]].add("X_OUTLIER_VS_GROUP_MEDIAN")

    # Delta candidates
    for d, r in delta_rows:
        if d > 20.0:
            candidate_reasons[r["sample_id"]].add("LEN_DEPTH_DELTA_GT20MM")
        if d > 50.0:
            candidate_reasons[r["sample_id"]].add("LEN_DEPTH_DELTA_GT50MM")

    # Sort tables
    delta_rows.sort(key=lambda x: x[0], reverse=True)
    x_outlier_rows.sort(key=lambda x: x[0], reverse=True)
    spread_rows.sort(key=lambda x: x[0], reverse=True)

    top_delta = delta_rows[: args.max_anomaly_rows]
    top_xout = x_outlier_rows[: args.max_anomaly_rows]
    top_spread = spread_rows[: min(args.max_anomaly_rows, 200)]

    # Build anomaly candidate CSV
    candidates_csv = out_dir / "scale_gt_anomaly_candidates.csv"
    with candidates_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sample_id",
                "split",
                "setup_id",
                "depth_index",
                "depth_mm",
                "length_mm",
                "x1",
                "x2",
                "y1",
                "y2",
                "fss_path",
                "reasons",
            ]
        )
        rows_by_id = {r["sample_id"]: r for r in rows}
        for sid, reasons in sorted(candidate_reasons.items()):
            r = rows_by_id.get(sid)
            if not r:
                continue
            w.writerow(
                [
                    sid,
                    r["split"],
                    r.get("setup_id", ""),
                    r["depth_index"],
                    r["depth_mm"],
                    r["length_mm"],
                    r["x1"],
                    r["x2"],
                    r["y1"],
                    r["y2"],
                    r["fss_path"],
                    ";".join(sorted(reasons)),
                ]
            )

    summary = {
        "manifest": manifest.as_posix(),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows_total": len(rows),
        "groups_total": len(path_splits),
        "split_rows": dict(split_rows),
        "split_groups": {k: len(v) for k, v in split_groups.items()},
        "leak_check": {
            "path_cross_split": path_leak,
            "setup_cross_split": setup_leak,
            "content_hash_cross_split": hash_leak,
        },
        "label_side_distribution": dict(side_dist),
        "tick_distribution": dict(tick_dist),
        "y_direction": {"y2_gte_y1": y_down, "y2_lt_y1": y_up},
        "length_depth_delta": {
            "median": statistics.median(deltas),
            "p95": sorted(deltas)[max(0, int(0.95 * len(deltas)) - 1)],
            "max": max(deltas),
            "count_gt_20": sum(1 for d in deltas if d > 20.0),
            "count_gt_50": sum(1 for d in deltas if d > 50.0),
        },
        "anomaly_candidates": {
            "sample_count": len(candidate_reasons),
            "csv": candidates_csv.as_posix(),
        },
    }
    (out_dir / "scale_pretrain_review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    split_rows_table = _table(
        ["Split", "Rows", "Unique fss_path"],
        [
            [k, split_rows.get(k, 0), len(split_groups.get(k, set()))]
            for k in ["train", "val", "test"]
        ],
    )

    leak_table = _table(
        ["Leak Check", "Count", "Gate"],
        [
            ["fss_path in multiple splits", path_leak, "MUST be 0"],
            ["setup_id in multiple splits", setup_leak, "MUST be 0"],
            ["content hash in multiple splits", hash_leak, "MUST be 0"],
        ],
    )

    dist_table = _table(
        ["Metric", "Value"],
        [
            ["rows_total", len(rows)],
            ["groups_total", len(path_splits)],
            ["label_side distribution", dict(side_dist)],
            ["tick_mm distribution", dict(tick_dist)],
            ["y2>=y1", y_down],
            ["y2<y1", y_up],
            ["|length_mm-depth_mm| median", round(summary["length_depth_delta"]["median"], 3)],
            ["|length_mm-depth_mm| p95", round(summary["length_depth_delta"]["p95"], 3)],
            ["|length_mm-depth_mm| max", round(summary["length_depth_delta"]["max"], 3)],
            ["delta>20mm rows", summary["length_depth_delta"]["count_gt_20"]],
            ["delta>50mm rows", summary["length_depth_delta"]["count_gt_50"]],
            ["anomaly candidate sample_id", len(candidate_reasons)],
        ],
    )

    res_table = _table(
        ["Resolution", "Count"],
        [[k, v] for k, v in res_dist.most_common(20)],
    )

    delta_table = _table(
        [
            "abs(length-depth)",
            "split",
            "sample_id",
            "setup_id",
            "depth_idx",
            "depth_mm",
            "length_mm",
            "x1",
            "x2",
            "y1",
            "y2",
            "fss_path",
        ],
        [
            [
                round(d, 3),
                r["split"],
                r["sample_id"],
                r.get("setup_id", ""),
                r["depth_index"],
                r["depth_mm"],
                r["length_mm"],
                r["x1"],
                r["x2"],
                r["y1"],
                r["y2"],
                r["fss_path"],
            ]
            for d, r in top_delta
        ],
        cls="wide",
    )

    xout_table = _table(
        [
            "x_dist_from_group_med",
            "split",
            "sample_id",
            "setup_id",
            "x_center",
            "group_med",
            "group_mad",
            "depth_idx",
            "fss_path",
        ],
        [
            [
                round(dist, 2),
                r["split"],
                r["sample_id"],
                r.get("setup_id", ""),
                round((_f(r["x1"]) + _f(r["x2"])) * 0.5, 2),
                round(med, 2),
                round(mad, 2),
                r["depth_index"],
                r["fss_path"],
            ]
            for dist, r, med, mad in top_xout
        ],
        cls="wide",
    )

    spread_table = _table(
        ["x_spread_px", "rows_in_group", "depth_count", "fss_path"],
        [[round(sp, 2), nrows, ndepth, path] for sp, path, nrows, ndepth in top_spread],
        cls="wide",
    )

    html_path = out_dir / "scale_pretrain_review.html"
    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale GT Pre-Training Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; color: #1f2937; }}
    h1, h2 {{ margin: 0.2em 0; }}
    .meta {{ color: #4b5563; margin-bottom: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff; }}
    .gate-ok {{ color: #065f46; font-weight: 700; }}
    .gate-bad {{ color: #991b1b; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 1; }}
    .wide td {{ max-width: 700px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .small {{ font-size: 12px; color: #4b5563; }}
    .pill {{ display:inline-block; border-radius: 999px; padding: 3px 8px; background:#eef2ff; margin-right: 8px; }}
  </style>
</head>
<body>
  <h1>Scale GT Pre-Training Review</h1>
  <div class="meta">
    Manifest: <code>{_esc(manifest.as_posix())}</code><br/>
    Generated: {_esc(summary["generated_at"])}<br/>
    Anomaly CSV: <code>{_esc(candidates_csv.as_posix())}</code>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Split Overview</h2>
      {split_rows_table}
    </div>
    <div class="card">
      <h2>Leakage Gates</h2>
      {leak_table}
      <p class="small">
        Status:
        <span class="{'gate-ok' if (path_leak==0 and setup_leak==0 and hash_leak==0) else 'gate-bad'}">
          {'PASS' if (path_leak==0 and setup_leak==0 and hash_leak==0) else 'FAIL'}
        </span>
      </p>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Core Distributions</h2>
      {dist_table}
    </div>
    <div class="card">
      <h2>Top Resolutions</h2>
      {res_table}
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h2>Review Checklist</h2>
    <p class="small">
      <span class="pill">1. Leakage = 0</span>
      <span class="pill">2. Inspect LEN/DEPTH delta outliers</span>
      <span class="pill">3. Inspect X outliers in-group</span>
      <span class="pill">4. Fix GT rows in anomaly CSV before final training</span>
    </p>
  </div>

  <details open>
    <summary>Top |length_mm - depth_mm| rows (candidate GT checks)</summary>
    <div class="small">Most suspicious rows first. High deltas are not always wrong, but deserve review.</div>
    {delta_table}
  </details>

  <details>
    <summary>Robust X outliers within same fss_path</summary>
    <div class="small">Rows where x_center is far from group median (MAD rule).</div>
    {xout_table}
  </details>

  <details>
    <summary>Groups with large X spread</summary>
    <div class="small">Useful to find setups with unstable scale placement.</div>
    {spread_table}
  </details>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML review: {html_path}")
    print(f"Summary JSON: {out_dir / 'scale_pretrain_review_summary.json'}")
    print(f"Anomaly CSV: {candidates_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
