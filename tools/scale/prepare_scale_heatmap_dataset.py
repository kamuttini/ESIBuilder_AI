#!/usr/bin/env python3
"""Build the dense-supervision dataset for the scale block, split leak-free per vendor.

Why heatmaps and not scalar regression
--------------------------------------
The earlier attempts (``32_``/``34_``/``36_``) regressed three numbers — x, y_top,
y_bottom — from a 1280x1024 frame. Two things doomed that:

* half the target was noise (``y_bottom`` is wherever the legacy operator stopped
  dragging, see ``docs/scala_strategia_per_vendor_2026-07-29.md``);
* three scalars give the network no signal about *where to look*, so nothing supervises
  the localisation that the task is actually about.

Here the target is dense instead. For every row we know the ruler column ``x`` and the
zero ``y_zero`` exactly, so each image supervises a full 1-D distribution along x and
another along y — hundreds of labelled bins per image instead of two numbers.

What is predicted
-----------------
========================  ==========================================================
``x`` heatmap             1-D distribution over image columns: the ruler column
``y_zero`` heatmap        1-D distribution over image rows: the zero tick
``log_span_mm``           ``log(mm_per_px * image_height)``, i.e. how many mm the frame
                          spans vertically — resolution independent, so it survives the
                          resize and is directly comparable across vendors
``direction``             2 classes: zero at top / zero at bottom
========================  ==========================================================

``y_far`` is deliberately **not** a target: it is the operator's arbitrary endpoint.
The tick positions are also not targets, because the *visible* tick step is not always
the 0.5 cm the .fss declares (measured: a Hitachi row whose printed ticks are 1 cm
apart), so a synthesised tick grid would be a partly wrong label.

Split
-----
Leak-free at ``config_folder`` level, as required by the project conventions: images of
one acquisition folder never straddle two splits. The split is a deterministic hash of
the folder name, so re-running adds new folders without reshuffling the old ones.

Usage
-----
    python3 tools/scale/prepare_scale_heatmap_dataset.py \
        --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
        --output-dir artifacts/39_scale_heatmap_dataset_20260729
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Vendors below this many rows cannot support their own model; they are still written to
# the combined manifest so the general model covers them.
MIN_ROWS_PER_VENDOR = 150

SPLIT_WEIGHTS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def split_for_folder(folder: str, salt: str = "scale-heatmap-v1") -> str:
    """Deterministic leak-free split from the folder name."""
    h = hashlib.sha256(f"{salt}|{folder}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, w in SPLIT_WEIGHTS:
        acc += w
        if frac < acc:
            return name
    return SPLIT_WEIGHTS[-1][0]


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


FIELDS = [
    "sample_id",
    "vendor",
    "config_folder",
    "setup_id",
    "depth_index",
    "split",
    "image_path",
    "image_w",
    "image_h",
    # targets
    "x",
    "x_norm",
    "y_zero",
    "y_zero_norm",
    "mm_per_px",
    "span_mm",
    "log_span_mm",
    "direction",
    # context the model may use as input or the caller as a prior
    "rect_x1",
    "rect_y1",
    "rect_x2",
    "rect_y2",
    "vect_depth_mm",
    "tick_pitch_px",
    # provenance / filters
    "flag_fusion",
    "flag_scale_inside_rect",
    "side_vs_rect",
]


def build_rows(gt_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    skipped: Counter = Counter()
    for r in gt_rows:
        img = r.get("image_path", "")
        w, h = _f(r, "video_x_size"), _f(r, "video_y_size")
        x, y_zero = _f(r, "x"), _f(r, "y_zero")
        mm = _f(r, "mm_per_px_from_scale")
        if not img:
            skipped["no_image"] += 1
            continue
        if not w or not h or w <= 0 or h <= 0:
            skipped["no_frame_size"] += 1
            continue
        if x is None or y_zero is None or not mm or mm <= 0:
            skipped["no_target"] += 1
            continue
        if not (0 <= x < w and 0 <= y_zero < h):
            skipped["target_outside_frame"] += 1
            continue

        span_mm = mm * h
        out.append(
            {
                "sample_id": r.get("sample_id", ""),
                "vendor": r.get("vendor", ""),
                "config_folder": r.get("config_folder", ""),
                "setup_id": r.get("setup_id", ""),
                "depth_index": r.get("depth_index", ""),
                "split": split_for_folder(r.get("config_folder", "")),
                "image_path": img,
                "image_w": int(w),
                "image_h": int(h),
                "x": round(x, 2),
                "x_norm": round(x / w, 6),
                "y_zero": round(y_zero, 2),
                "y_zero_norm": round(y_zero / h, 6),
                "mm_per_px": round(mm, 6),
                "span_mm": round(span_mm, 3),
                "log_span_mm": round(math.log(span_mm), 6),
                "direction": -1 if r.get("zero_at") == "bottom" else 1,
                "rect_x1": r.get("rect_x1", ""),
                "rect_y1": r.get("rect_y1", ""),
                "rect_x2": r.get("rect_x2", ""),
                "rect_y2": r.get("rect_y2", ""),
                "vect_depth_mm": r.get("vect_depth_mm", ""),
                "tick_pitch_px": r.get("tick_pitch_px", ""),
                "flag_fusion": r.get("flag_fusion", ""),
                "flag_scale_inside_rect": r.get("flag_scale_inside_rect", ""),
                "side_vs_rect": r.get("side_vs_rect", ""),
            }
        )
    if skipped:
        print(f"[prepare] skipped: {dict(skipped)}")
    return out


def _write(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _vendor_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_split = Counter(str(r["split"]) for r in rows)
    folders_by_split = defaultdict(set)
    for r in rows:
        folders_by_split[str(r["split"])].add(str(r["config_folder"]))
    spans = [float(r["span_mm"]) for r in rows]
    return {
        "rows": len(rows),
        "folders": len({str(r["config_folder"]) for r in rows}),
        "setups": len({(r["config_folder"], r["setup_id"]) for r in rows}),
        "rows_by_split": dict(by_split),
        "folders_by_split": {k: len(v) for k, v in sorted(folders_by_split.items())},
        "direction": dict(Counter(str(r["direction"]) for r in rows)),
        "resolutions": dict(
            Counter(f"{r['image_w']}x{r['image_h']}" for r in rows).most_common(5)
        ),
        "span_mm": {
            "min": round(min(spans), 1),
            "median": round(statistics.median(spans), 1),
            "max": round(max(spans), 1),
        },
        "flag_fusion": sum(1 for r in rows if str(r["flag_fusion"]) == "1"),
        "eligible_own_model": len(rows) >= MIN_ROWS_PER_VENDOR
        and all(by_split.get(s, 0) >= 20 for s, _ in SPLIT_WEIGHTS),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the scale heatmap dataset.")
    p.add_argument(
        "--gt",
        type=Path,
        default=Path("artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--drop-fusion",
        action="store_true",
        help="Exclude multi-panel fusion rows: they contain two rulers, so the single-peak "
        "target is ambiguous.",
    )
    p.add_argument(
        "--drop-biopsee",
        action="store_true",
        help="Exclude Biopsee acquisitions: the navigation system re-renders the ultrasound, "
        "so the overlay is a different object (sometimes without numbers at all).",
    )
    p.add_argument(
        "--drop-suspect",
        action="store_true",
        help="Exclude configurations the team marked as wrong in the folder name "
        "(SBAGLIATA, NON USARE, proibita, da rifare...).",
    )
    p.add_argument(
        "--exclude-folder-token",
        action="append",
        default=[],
        help="Drop rows whose config folder contains this token, case-insensitive "
        "(repeatable). Escape hatch for one-off exclusions.",
    )
    p.add_argument(
        "--only-folder-token",
        action="append",
        default=[],
        help="Inverse of --exclude-folder-token: keep ONLY rows whose folder contains one "
        "of these tokens. Used to build diagnostic sets, e.g. a fusion-only test set to "
        "measure what a model trained without fusion does when it meets one.",
    )
    p.add_argument(
        "--force-split",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help="Put every row in this split. Only for diagnostic sets built with "
        "--only-folder-token, where the notion of a train split is meaningless.",
    )
    return p


# Tokens behind the named filters, kept here so the exclusion is auditable rather than
# buried in an if-chain.
#
# Note on what is NOT here: "negative" and "proibite" are *image types*, not folders
# (``image_th_echo_negative_*``, ``image_th_proibite_0_negative_*`` — the negatives the
# other blocks train on, and the forbidden screens of .fss line 15). This dataset only
# ever loads ``image_depth_value_setup_N``, so all 5210 rows already avoid them and there
# is nothing to filter. Verified: zero rows point at a ``*_negative_*`` file.
DROP_TOKENS = {
    "fusion": ("fusion",),
    "biopsee": ("biops",),
    "suspect": (
        "proibite",
        "proibita",
        "proibiti",
        "proibito",
        "prohibited",
        "sbagliat",
        "non usare",
        "non_usare",
        "da rifare",
    ),
}


def apply_exclusions(
    rows: List[Dict[str, object]],
    tokens: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Drop rows whose folder name contains any token. Returns (kept, report)."""
    if not tokens:
        return rows, {"tokens": [], "rows_dropped": 0}
    low = [t.lower() for t in tokens]
    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for r in rows:
        folder = str(r["config_folder"]).lower()
        if any(t in folder for t in low):
            dropped.append(r)
        else:
            kept.append(r)
    return kept, {
        "tokens": list(tokens),
        "rows_dropped": len(dropped),
        "folders_dropped": sorted({str(r["config_folder"]) for r in dropped}),
        "rows_dropped_by_vendor": dict(Counter(str(r["vendor"]) for r in dropped).most_common()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gt_rows = list(csv.DictReader(args.gt.open(encoding="utf-8")))
    rows = build_rows(gt_rows)

    tokens: List[str] = []
    if args.drop_fusion:
        tokens += list(DROP_TOKENS["fusion"])
    if args.drop_biopsee:
        tokens += list(DROP_TOKENS["biopsee"])
    if args.drop_suspect:
        tokens += list(DROP_TOKENS["suspect"])
    tokens += list(args.exclude_folder_token)
    if args.only_folder_token:
        low = [t.lower() for t in args.only_folder_token]
        before = len(rows)
        rows = [r for r in rows if any(t in str(r["config_folder"]).lower() for t in low)]
        print(
            f"[prepare] kept only folders matching {args.only_folder_token}: "
            f"{len(rows)} rows of {before}"
        )

    rows, exclusion_report = apply_exclusions(rows, tokens)
    if tokens:
        print(
            f"[prepare] exclusion tokens {exclusion_report['tokens']}: "
            f"dropped {exclusion_report['rows_dropped']} rows "
            f"from {len(exclusion_report['folders_dropped'])} folders "
            f"({exclusion_report['rows_dropped_by_vendor']})"
        )
    if args.force_split:
        for r in rows:
            r["split"] = args.force_split
        print(f"[prepare] every row forced into split '{args.force_split}'")

    if not rows:
        print("[error] no usable rows")
        return 2

    out_dir: Path = args.output_dir
    manifest_dir = out_dir / "manifests"
    _write(manifest_dir / "manifest_scale_heatmap_all.csv", rows)

    by_vendor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_vendor[str(r["vendor"])].append(r)

    stats: Dict[str, object] = {}
    eligible: List[str] = []
    for vendor, vrows in sorted(by_vendor.items(), key=lambda kv: -len(kv[1])):
        slug = vendor.lower().replace(" ", "_") or "unknown"
        _write(manifest_dir / f"manifest_scale_heatmap_{slug}.csv", vrows)
        st = _vendor_stats(vrows)
        stats[vendor] = st
        if st["eligible_own_model"]:
            eligible.append(vendor)

    # Sanity check the split really is leak-free at folder level.
    folder_splits = defaultdict(set)
    for r in rows:
        folder_splits[str(r["config_folder"])].add(str(r["split"]))
    leaks = [f for f, s in folder_splits.items() if len(s) > 1]

    summary = {
        "gt": args.gt.as_posix(),
        "exclusions": exclusion_report,
        "rows": len(rows),
        "folders": len(folder_splits),
        "setups": len({(r["config_folder"], r["setup_id"]) for r in rows}),
        "rows_by_split": dict(Counter(str(r["split"]) for r in rows)),
        "split_leaks": leaks,
        "vendors_eligible_own_model": eligible,
        "min_rows_per_vendor": MIN_ROWS_PER_VENDOR,
        "targets": [
            "x_norm (heatmap over columns)",
            "y_zero_norm (heatmap over rows)",
            "log_span_mm = log(mm_per_px * image_height)",
            "direction (zero at top / bottom)",
        ],
        "per_vendor": stats,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "per_vendor"}, indent=2))
    print("\nvendor            righe  cartelle  train/val/test         modello proprio")
    for vendor, st in stats.items():
        s = st["rows_by_split"]
        print(
            f"  {vendor:15s} {st['rows']:5d}  {st['folders']:7d}  "
            f"{s.get('train',0):5d}/{s.get('val',0):4d}/{s.get('test',0):4d}"
            f"        {'si' if st['eligible_own_model'] else 'no'}"
        )
    if leaks:
        print(f"\n[error] {len(leaks)} cartelle presenti in piu' split: {leaks[:5]}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
