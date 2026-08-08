"""Turn human corrections from the review tool into ground truth and a regression set.

This closes the human-in-the-loop ring: every correction the operator makes in
``build_scale_correction_tool.py`` becomes (a) a GT row with the confirmed geometry and
calibration, (b) a regression set the eval harness can run directly, and (c) a summary of
*what* the detector got wrong, which is what tells us where to work next.

The correction export carries, in original frame pixels:
``corr_x, corr_y_zero, corr_y_far, zero_at``, the ``x_set/zero_set/far_set`` flags saying
which handles the operator actually placed, ``added_labels`` as ``y_px:cm`` pairs typed in by
hand, plus quick flags and a free comment.

``mm_per_px`` is derived only when the operator gave enough to determine it — two typed
numbers, or one number plus the zero — because that is the quantity the .fss really encodes
and guessing it would poison the very GT we are building.

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/ingest_scale_corrections.py \
    --corrections ~/Downloads/scale_corrections.csv \
    --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
    --out-dir artifacts/46_scale_human_gt
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Columns the eval harness needs from a GT row.
REQUIRED = ("sample_id", "config_folder", "vendor", "depth_index", "image_path",
            "x", "y_zero", "y_far", "zero_at", "mm_per_px_from_scale")


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "si", "sì")


def parse_added_labels(raw: str) -> List[Tuple[float, float]]:
    """``"486:2.5|670:3"`` -> ``[(486.0, 2.5), (670.0, 3.0)]`` (y in px, value in cm)."""
    out: List[Tuple[float, float]] = []
    for chunk in str(raw or "").split("|"):
        if ":" not in chunk:
            continue
        y_s, v_s = chunk.split(":", 1)
        y, v = _f(y_s), _f(v_s.replace(",", "."))
        if y is not None and v is not None:
            out.append((y, v))
    return out


def derive_mm_per_px(labels: List[Tuple[float, float]], y_zero: Optional[float]) -> Optional[float]:
    """mm per pixel from what the operator marked, or None when it is not determined.

    Two typed numbers fix the scale outright. One number plus the zero also fixes it, and the
    zero is reliable here precisely because the operator placed it. Anything less is left
    empty rather than guessed.
    """
    if len(labels) >= 2:
        labels = sorted(labels, key=lambda t: t[0])
        (y1, v1), (y2, v2) = labels[0], labels[-1]
        if abs(y2 - y1) >= 2 and abs(v2 - v1) > 1e-9:
            return abs((v2 - v1) * 10.0 / (y2 - y1))
        return None
    if len(labels) == 1 and y_zero is not None:
        y, v = labels[0]
        if abs(y - y_zero) >= 2 and v > 0:
            return abs(v * 10.0 / (y - y_zero))
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrections", type=Path, nargs="+", required=True,
                    help="One or more scale_corrections.csv exported from the review tool.")
    ap.add_argument("--gt", type=Path, default=None,
                    help="Audit GT to merge with: unset fields are carried over from it.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    gt_by_id: Dict[str, Dict[str, str]] = {}
    gt_by_path: Dict[str, Dict[str, str]] = {}
    if args.gt and args.gt.exists():
        for r in csv.DictReader(open(args.gt, encoding="utf-8")):
            gt_by_id[r.get("sample_id", "")] = r
            gt_by_path[r.get("image_path", "")] = r
        print(f"[gt] {len(gt_by_id)} righe di riferimento da {args.gt.name}")

    rows_out: List[Dict[str, str]] = []
    stats = collections.Counter()
    flag_counter: collections.Counter = collections.Counter()
    by_vendor: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    comments: List[Dict[str, str]] = []

    for path in args.corrections:
        if not path.exists():
            print(f"[skip] file assente: {path}")
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            stats["righe_esportate"] += 1
            vendor = r.get("vendor", "")
            flags = [f for f in str(r.get("flags", "")).split("|") if f]
            for f in flags:
                flag_counter[f] += 1
                by_vendor[vendor][f] += 1
            note = (r.get("comment") or "").strip()
            if note:
                comments.append({"vendor": vendor, "folder": r.get("config_folder", ""),
                                 "depth_index": r.get("depth_index", ""), "comment": note})

            # "no righello" is a negative label: valuable, but not a GT geometry row.
            if "noruler" in flags:
                stats["escluse_no_righello"] += 1
                continue

            base = gt_by_id.get(r.get("sample_id", "")) or gt_by_path.get(r.get("image_path", "")) or {}
            row: Dict[str, str] = dict(base)
            row["sample_id"] = r.get("sample_id") or base.get("sample_id", "")
            row["config_folder"] = r.get("config_folder") or base.get("config_folder", "")
            row["vendor"] = vendor or base.get("vendor", "")
            row["depth_index"] = r.get("depth_index") or base.get("depth_index", "")
            row["image_path"] = r.get("image_path") or base.get("image_path", "")

            # Only take a handle the operator actually placed; otherwise keep the GT value.
            if _truthy(r.get("x_set")) and _f(r.get("corr_x")) is not None:
                row["x"] = str(_f(r.get("corr_x")))
                stats["x_corretta"] += 1
            if _truthy(r.get("zero_set")) and _f(r.get("corr_y_zero")) is not None:
                row["y_zero"] = str(_f(r.get("corr_y_zero")))
                stats["zero_corretto"] += 1
            if _truthy(r.get("far_set")) and _f(r.get("corr_y_far")) is not None:
                row["y_far"] = str(_f(r.get("corr_y_far")))
                stats["estremo_corretto"] += 1
            if r.get("zero_at"):
                row["zero_at"] = "top" if r["zero_at"] in ("top", "alto") else "bottom"

            labels = parse_added_labels(r.get("added_labels", ""))
            mmpp = derive_mm_per_px(labels, _f(row.get("y_zero")))
            if mmpp:
                row["mm_per_px_from_scale"] = f"{mmpp:.6f}"
                row["tick_cm"] = row.get("tick_cm") or "0.5"
                stats["calibrazione_da_numeri_umani"] += 1
            if labels:
                stats["numeri_inseriti_a_mano"] += len(labels)

            row["human_reviewed"] = "1"
            row["human_flags"] = "|".join(flags)
            row["human_comment"] = note
            # Usable as a regression row only if the eval has what it needs.
            if all(row.get(k) not in (None, "") for k in REQUIRED):
                rows_out.append(row)
                stats["righe_regressione"] += 1
            else:
                stats["incomplete_non_usabili"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rows_out:
        fields: List[str] = []
        for row in rows_out:
            for k in row:
                if k not in fields:
                    fields.append(k)
        out_csv = args.out_dir / "gt_human_regression.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"[ok] set di regressione: {out_csv} ({len(rows_out)} righe)")
        print("     usalo con: tools/scale/eval_scale_detector.py --gt "
              f"{out_csv} --output-dir artifacts/46_scale_human_gt/eval --consensus")
    else:
        print("[attenzione] nessuna riga completa: servono almeno x, zero, estremo e "
              "due numeri (o un numero + zero) per determinare la calibrazione")

    summary = {
        "conteggi": dict(stats),
        "flag_piu_frequenti": flag_counter.most_common(),
        "flag_per_vendor": {v: dict(c) for v, c in by_vendor.items()},
        "commenti": comments[:200],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== cosa ha corretto la revisione umana ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    if flag_counter:
        print("  flag:", ", ".join(f"{k}={v}" for k, v in flag_counter.most_common()))
    if by_vendor:
        print("  per vendor:")
        for v, c in sorted(by_vendor.items()):
            if c:
                print(f"    {v}: {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
