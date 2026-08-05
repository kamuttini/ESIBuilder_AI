#!/usr/bin/env python3
"""Measure a scale detector against the audited .fss ground truth, per vendor.

The metric that matters is **calibration**, not endpoints: .fss lines 19/20 and the
usable part of line 21 are ``mm_per_px`` plus the zero point, while ``y_bottom`` is
whatever tick the legacy operator happened to drag to. So we report, per vendor:

* ``mm_per_px`` relative error  -> the real quality of the block
* ``x`` error (px)             -> did we pick the right ruler column
* ``y_zero`` error (px)        -> did we find the zero tick
* coverage of accepted / review / reject under the detector's own confidence policy
* ``strict_ok`` = all three within tolerance, which is the production criterion

An HTML gallery with prediction-vs-GT overlays is written next to the metrics so the
failures can be looked at instead of guessed about.

Usage:
    python3 tools/scale/eval_scale_detector.py \
        --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
        --output-dir artifacts/38_scale_ladder_eval_20260729/bk \
        --vendor BK --max-rows 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from consolidate_scale_setup import ScaleCandidate, consolidate_setup  # noqa: E402
from detect_scale_ladder import ScalePrediction, detect_scale, profile_for  # noqa: E402

# Production tolerances, deliberately explicit.
TOL_MM_PER_PX_REL = 0.02  # 2% on the calibration
TOL_X_PX = 6.0
TOL_Y_ZERO_PX = 8.0


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def remap_path(path: str, remap: Sequence[str]) -> str:
    """Apply ``--path-remap OLD=NEW`` rules (SSD mounted at a different point)."""
    for rule in remap:
        if "=" in rule:
            old, new = rule.split("=", 1)
            if path.startswith(old):
                return new + path[len(old) :]
    return path


def predict_row(
    row: Dict[str, str],
    remap: Sequence[str],
    registry=None,
) -> Optional[ScalePrediction]:
    """Run the per-image detector for one GT row, or None if the image is unreadable.

    With a heatmap ``registry`` the network first proposes the ruler column, the classical
    stage searches only there, and the network also stands in for whatever the classical
    stage could not produce: if no ladder was found at all, the prior itself becomes the
    prediction (marked ``heatmap_only``) so the setup consensus has something to work with
    instead of a hole.
    """
    img_path = remap_path(row.get("image_path", ""), remap)
    if not img_path or not Path(img_path).is_file():
        return None
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    rect = None
    rx1, ry1, rx2, ry2 = (_f(row, k) for k in ("rect_x1", "rect_y1", "rect_x2", "rect_y2"))
    if None not in (rx1, ry1, rx2, ry2):
        rect = (int(rx1), int(ry1), int(rx2), int(ry2))  # type: ignore[arg-type]

    vendor = row.get("vendor", "")
    prior = None
    if registry:
        try:
            prior = registry.predict(img, vendor)
        except Exception as exc:  # noqa: BLE001 - never let the model break the run
            print(f"[heatmap] prediction failed on {img_path}: {exc}")

    pred = detect_scale(
        img,
        rect=rect,
        vendor=vendor,
        prior_x=None if prior is None or prior.ambiguous_column else prior.x,
    )

    if prior is not None and not pred.ok:
        # Classical stage found nothing usable: emit the network's answer for review.
        return ScalePrediction(
            ok=True,
            status="review",
            reason="heatmap_only",
            x=prior.x,
            y_zero=prior.y_zero,
            mm_per_px=prior.mm_per_px,
            direction=prior.direction,
            confidence=min(0.5, 0.5 * prior.x_confidence + 0.5 * prior.y_confidence),
            debug={"source": "heatmap", "x_sharpness": prior.x_sharpness},
        )
    if prior is not None:
        pred.debug["heatmap_x"] = round(prior.x, 1)
        pred.debug["heatmap_dx"] = None if pred.x is None else round(pred.x - prior.x, 1)
    return pred


def score_row(
    row: Dict[str, str],
    remap: Sequence[str],
    pred: ScalePrediction,
    source: str = "detected",
    notes: str = "",
) -> Dict[str, object]:
    """Compare one prediction with the GT row it belongs to."""
    img_path = remap_path(row.get("image_path", ""), remap)
    gt_x = _f(row, "x")
    gt_y_zero = _f(row, "y_zero")
    gt_y_far = _f(row, "y_far")
    gt_zero_at = row.get("zero_at", "")
    gt_mm_per_px = _f(row, "mm_per_px_from_scale")
    gt_length_mm = _f(row, "length_mm")

    err_x = None if (pred.x is None or gt_x is None) else abs(pred.x - gt_x)
    err_y = None if (pred.y_zero is None or gt_y_zero is None) else abs(pred.y_zero - gt_y_zero)
    rel_mm = None
    if pred.mm_per_px and gt_mm_per_px:
        rel_mm = abs(pred.mm_per_px - gt_mm_per_px) / gt_mm_per_px

    pred_zero_at = "bottom" if pred.direction < 0 else "top"
    direction_ok = bool(pred.ok and gt_zero_at and pred_zero_at == gt_zero_at)

    strict_ok = bool(
        err_x is not None
        and err_y is not None
        and rel_mm is not None
        and err_x <= TOL_X_PX
        and err_y <= TOL_Y_ZERO_PX
        and rel_mm <= TOL_MM_PER_PX_REL
        and direction_ok
    )
    calib_ok = bool(rel_mm is not None and rel_mm <= TOL_MM_PER_PX_REL)

    seg = pred.fss_segment()
    return {
        "sample_id": row.get("sample_id", ""),
        "vendor": row.get("vendor", ""),
        "setup_key": f"{row.get('config_folder','')}|{row.get('setup_id','')}",
        "source": source,
        "notes": notes,
        "config_folder": row.get("config_folder", ""),
        "depth_index": row.get("depth_index", ""),
        "image_path": img_path,
        "status": pred.status,
        "reason": pred.reason,
        "confidence": round(pred.confidence, 4),
        "n_ticks": pred.n_ticks,
        "n_labels": len(pred.labels),
        "label_side": pred.label_side or "",
        "pred_x": None if pred.x is None else round(pred.x, 2),
        "pred_y_zero": None if pred.y_zero is None else round(pred.y_zero, 2),
        "pred_mm_per_px": None if pred.mm_per_px is None else round(pred.mm_per_px, 6),
        "pred_tick_pitch_px": None if pred.tick_pitch_px is None else round(pred.tick_pitch_px, 2),
        "pred_tick_step_mm": None if pred.tick_step_mm is None else round(pred.tick_step_mm, 3),
        "pred_length_mm": None if seg is None else seg["length_mm"],
        "pred_zero_at": pred_zero_at,
        "gt_x": gt_x,
        "gt_y_zero": gt_y_zero,
        "gt_y_far": gt_y_far,
        "gt_zero_at": gt_zero_at,
        "direction_ok": int(direction_ok),
        "gt_mm_per_px": gt_mm_per_px,
        "gt_length_mm": gt_length_mm,
        "gt_tick_pitch_px": _f(row, "tick_pitch_px"),
        "err_x_px": None if err_x is None else round(err_x, 2),
        "err_y_zero_px": None if err_y is None else round(err_y, 2),
        "rel_err_mm_per_px": None if rel_mm is None else round(rel_mm, 5),
        "calib_ok": int(calib_ok),
        "strict_ok": int(strict_ok),
        "flag_scale_inside_rect": row.get("flag_scale_inside_rect", ""),
        "flag_fusion": row.get("flag_fusion", ""),
    }


def evaluate_row(
    row: Dict[str, str], remap: Sequence[str], registry=None
) -> Optional[Dict[str, object]]:
    """Per-image evaluation: detect on one frame, score it against its GT row."""
    pred = predict_row(row, remap, registry)
    if pred is None:
        return None
    return score_row(row, remap, pred)


def evaluate_setup(
    rows: Sequence[Dict[str, str]],
    remap: Sequence[str],
    registry=None,
) -> List[Dict[str, object]]:
    """Per-setup evaluation: detect on every depth of a setup, then consolidate.

    This is the mode that matters for the product: the user configures a folder, not a
    frame, and a folder is what constrains the answer enough to tell a good calibration
    from a plausible-looking wrong one.
    """
    preds: Dict[int, ScalePrediction] = {}
    keep: Dict[int, Dict[str, str]] = {}
    for row in rows:
        try:
            idx = int(row.get("depth_index", ""))
        except ValueError:
            continue
        pred = predict_row(row, remap, registry)
        if pred is None:
            continue
        keep[idx] = row
        preds[idx] = pred

    if not keep:
        return []

    candidates = [
        ScaleCandidate(
            depth_index=idx,
            x=p.x,
            y_zero=p.y_zero,
            mm_per_px=p.mm_per_px,
            direction=p.direction,
            confidence=p.confidence,
            status=p.status,
            n_labels=len(p.labels),
        )
        for idx, p in sorted(preds.items())
    ]
    consensus = consolidate_setup(candidates, depth_indices=sorted(keep))

    out: List[Dict[str, object]] = []
    for c in consensus:
        row = keep.get(c.depth_index)
        if row is None:
            continue
        base = preds[c.depth_index]
        merged = ScalePrediction(
            ok=c.mm_per_px is not None,
            status=c.status,
            reason=";".join(c.notes),
            x=c.x,
            y_zero=c.y_zero,
            mm_per_px=c.mm_per_px,
            tick_pitch_px=base.tick_pitch_px,
            tick_step_mm=base.tick_step_mm,
            y_last_tick=base.y_last_tick,
            direction=c.direction,
            span_mm=base.span_mm,
            labels=base.labels,
            label_side=base.label_side,
            calib_residual_mm=base.calib_residual_mm,
            confidence=c.confidence,
            ladder_score=base.ladder_score,
            n_ticks=base.n_ticks,
        )
        out.append(score_row(row, remap, merged, source=c.source, notes=";".join(c.notes)))
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _agg(vals: Sequence[float]) -> Dict[str, object]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 4),
        "median": round(statistics.median(s), 4),
        "p90": round(s[min(len(s) - 1, int(0.9 * (len(s) - 1)))], 4),
        "max": round(s[-1], 4),
    }


def compute_metrics(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def block(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        if not rows:
            return {"rows": 0}
        acc = [r for r in rows if r["status"] == "accepted"]
        return {
            "rows": len(rows),
            "status": dict(Counter(str(r["status"]) for r in rows).most_common()),
            "reject_reasons": dict(
                Counter(str(r["reason"]) for r in rows if r["status"] != "accepted").most_common(6)
            ),
            "strict_ok_pct": round(100 * sum(int(r["strict_ok"]) for r in rows) / len(rows), 2),
            "calib_ok_pct": round(100 * sum(int(r["calib_ok"]) for r in rows) / len(rows), 2),
            "direction_ok_pct": round(
                100 * sum(int(r["direction_ok"]) for r in rows) / len(rows), 2
            ),
            "strict_ok_pct_on_accepted": (
                round(100 * sum(int(r["strict_ok"]) for r in acc) / len(acc), 2) if acc else None
            ),
            "calib_ok_pct_on_accepted": (
                round(100 * sum(int(r["calib_ok"]) for r in acc) / len(acc), 2) if acc else None
            ),
            "err_x_px": _agg([r["err_x_px"] for r in rows]),  # type: ignore[list-item]
            "err_y_zero_px": _agg([r["err_y_zero_px"] for r in rows]),  # type: ignore[list-item]
            "rel_err_mm_per_px": _agg([r["rel_err_mm_per_px"] for r in rows]),  # type: ignore[list-item]
        }

    by_vendor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in results:
        by_vendor[str(r["vendor"])].append(r)
    return {
        "overall": block(results),
        "by_vendor": {v: block(rows) for v, rows in sorted(by_vendor.items(), key=lambda kv: -len(kv[1]))},
    }


# --------------------------------------------------------------------------- #
# review gallery
# --------------------------------------------------------------------------- #


def render_overlay(result: Dict[str, object], out_path: Path, crop_pad: int = 120) -> bool:
    img = cv2.imread(str(result["image_path"]), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    gt_x, gt_y1, gt_y2 = result.get("gt_x"), result.get("gt_y_zero"), result.get("gt_y_far")
    px, py = result.get("pred_x"), result.get("pred_y_zero")

    if gt_x is not None and gt_y1 is not None and gt_y2 is not None:
        cv2.line(img, (int(gt_x), int(gt_y1)), (int(gt_x), int(gt_y2)), (0, 255, 0), 3)
        cv2.circle(img, (int(gt_x), int(gt_y1)), 9, (0, 255, 0), 3)
    if px is not None and py is not None:
        mm = result.get("pred_mm_per_px")
        sign = -1 if result.get("pred_zero_at") == "bottom" else 1
        y2 = int(py) + sign * (int((result.get("pred_length_mm") or 0) / mm) if mm else 60)
        cv2.line(img, (int(px), int(py)), (int(px), max(0, min(h - 1, y2))), (0, 0, 255), 3)
        cv2.circle(img, (int(px), int(py)), 9, (0, 0, 255), 3)

    xs = [v for v in (gt_x, px) if v is not None]
    ys = [v for v in (gt_y1, gt_y2, py) if v is not None]
    if xs and ys:
        x1 = max(0, int(min(xs)) - crop_pad)
        x2 = min(w, int(max(xs)) + crop_pad)
        y1 = max(0, int(min(ys)) - crop_pad)
        y2 = min(h, int(max(ys)) + crop_pad)
        if x2 - x1 > 20 and y2 - y1 > 20:
            img = img[y1:y2, x1:x2]
    scale = min(1.0, 520 / max(1, img.shape[1]), 900 / max(1, img.shape[0]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 82]))


def write_gallery(
    results: Sequence[Dict[str, object]],
    metrics: Dict[str, object],
    out_dir: Path,
    max_cards: int,
    seed: int,
) -> Path:
    prev_dir = out_dir / "previews"
    rng = random.Random(seed)

    failures = [r for r in results if not r["strict_ok"]]
    successes = [r for r in results if r["strict_ok"]]
    rng.shuffle(failures)
    rng.shuffle(successes)
    # Failures first: a gallery of things that already work teaches nothing.
    n_fail = min(len(failures), int(max_cards * 0.7))
    picked = failures[:n_fail] + successes[: max_cards - n_fail]

    cards = []
    for i, r in enumerate(picked):
        rel = f"previews/{i:04d}.jpg"
        if not render_overlay(r, prev_dir / f"{i:04d}.jpg"):
            continue
        badge = "OK" if r["strict_ok"] else "KO"
        colour = "#1f8b4c" if r["strict_ok"] else "#b3261e"
        cards.append(
            f"""
<div class="card">
  <img src="{rel}" loading="lazy">
  <div class="meta">
    <span class="badge" style="background:{colour}">{badge}</span>
    <span class="st">{escape(str(r['status']))}</span>
    <b>{escape(str(r['vendor']))}</b>
    <div class="folder">{escape(str(r['config_folder']))} · depth {escape(str(r['depth_index']))}</div>
    <table>
      <tr><td>err x</td><td>{r['err_x_px']}</td><td>err y0</td><td>{r['err_y_zero_px']}</td></tr>
      <tr><td>rel mm/px</td><td>{r['rel_err_mm_per_px']}</td><td>conf</td><td>{r['confidence']}</td></tr>
      <tr><td>mm/px pred</td><td>{r['pred_mm_per_px']}</td><td>gt</td><td>{r['gt_mm_per_px']}</td></tr>
      <tr><td>tacche</td><td>{r['n_ticks']}</td><td>label OCR</td><td>{r['n_labels']}</td></tr>
      <tr><td>pitch pred</td><td>{r['pred_tick_pitch_px']}</td><td>gt</td><td>{r['gt_tick_pitch_px']}</td></tr>
      <tr><td>zero pred</td><td>{escape(str(r['pred_zero_at']))}</td><td>gt</td><td>{escape(str(r['gt_zero_at']))}</td></tr>
    </table>
    <div class="reason">{escape(str(r['reason']))}</div>
  </div>
</div>"""
        )

    overall = metrics["overall"]
    rows_html = "".join(
        f"<tr><td>{escape(v)}</td><td>{m['rows']}</td><td>{m['strict_ok_pct']}%</td>"
        f"<td>{m['calib_ok_pct']}%</td><td>{m['err_x_px'].get('median')}</td>"
        f"<td>{m['err_y_zero_px'].get('median')}</td>"
        f"<td>{m['rel_err_mm_per_px'].get('median')}</td><td>{escape(str(m['status']))}</td></tr>"
        for v, m in metrics["by_vendor"].items()
    )

    html = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<title>Review scala — detector tacche+OCR</title>
<style>
 body{{font:13px/1.45 system-ui,sans-serif;margin:0;padding:18px;background:#14161a;color:#e8e8e8}}
 h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px}}
 .sum table{{border-collapse:collapse;margin:8px 0 4px}}
 .sum td,.sum th{{border:1px solid #333;padding:4px 8px}}
 .sum th{{background:#22252b;text-align:left}}
 .grid{{display:flex;flex-wrap:wrap;gap:12px}}
 .card{{background:#1c1f24;border:1px solid #2c3038;border-radius:8px;padding:8px;width:540px}}
 .card img{{max-width:100%;border-radius:5px;display:block;background:#000}}
 .meta{{margin-top:6px}} .badge{{color:#fff;padding:1px 7px;border-radius:9px;font-weight:600}}
 .st{{color:#9aa4b2;margin-left:6px}} .folder{{color:#8d97a5;margin:4px 0;font-size:12px}}
 .meta table{{border-collapse:collapse;font-size:12px}} .meta td{{padding:1px 6px 1px 0}}
 .meta td:nth-child(odd){{color:#8d97a5}}
 .reason{{color:#d99;font-size:12px;margin-top:3px}}
 .legend span{{margin-right:14px}}
</style></head><body>
<h1>Review scala — detector tacche + OCR</h1>
<p class="legend"><span style="color:#0f0">verde = GT (riga 21)</span>
<span style="color:#f44">rosso = predizione</span>
<span>tolleranze: x ≤ {TOL_X_PX}px, y0 ≤ {TOL_Y_ZERO_PX}px, mm/px ≤ {int(TOL_MM_PER_PX_REL*100)}%</span></p>
<div class="sum">
<p><b>{overall['rows']}</b> righe · strict OK <b>{overall['strict_ok_pct']}%</b> ·
calibrazione OK <b>{overall['calib_ok_pct']}%</b> · stati {escape(str(overall['status']))}</p>
<table><tr><th>Vendor</th><th>Righe</th><th>strict OK</th><th>calib OK</th>
<th>err x (med)</th><th>err y0 (med)</th><th>rel mm/px (med)</th><th>stati</th></tr>
{rows_html}</table>
</div>
<h2>Casi ({len(cards)} carte, fallimenti in testa)</h2>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    path = out_dir / "review_scale_eval.html"
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the scale detector against .fss GT.")
    p.add_argument("--gt", type=Path, required=True, help="scale_gt_rows_clean.csv from the audit")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vendor", type=str, default="", help="Comma separated vendor whitelist.")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    p.add_argument("--max-per-folder", type=int, default=0, help="Cap rows per config folder.")
    p.add_argument("--max-cards", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument(
        "--consensus",
        action="store_true",
        help="Evaluate per setup: run every depth of a folder, then consolidate. "
        "Sampling switches from rows to setups, and --max-per-folder is ignored.",
    )
    p.add_argument("--max-setups", type=int, default=0, help="Cap setups in --consensus mode.")
    p.add_argument(
        "--heatmap-models",
        type=Path,
        default=None,
        help="Root of the per-vendor heatmap checkpoints (artifacts/40_.../). When given, "
        "the network proposes the ruler column and stands in where the classical stage "
        "finds nothing. Omit to run the classical pipeline alone.",
    )
    p.add_argument("--heatmap-device", type=str, default="auto")
    p.add_argument(
        "--path-remap",
        action="append",
        default=[],
        help="OLD=NEW prefix rewrite for image paths (repeatable).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = list(csv.DictReader(args.gt.open(encoding="utf-8")))
    if args.vendor:
        allow = {v.strip().lower() for v in args.vendor.split(",") if v.strip()}
        rows = [r for r in rows if r.get("vendor", "").lower() in allow]
    rows = [r for r in rows if r.get("image_path")]

    registry = None
    if args.heatmap_models is not None:
        from predict_scale_heatmap import describe_registry, load_registry

        registry = load_registry(args.heatmap_models, args.heatmap_device)
        print(f"[heatmap] {describe_registry(registry)}")
        if not registry:
            registry = None

    rng = random.Random(args.seed)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    skipped = 0

    if args.consensus:
        setups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for r in rows:
            setups[f"{r.get('config_folder','')}|{r.get('setup_id','')}"].append(r)
        keys = sorted(setups)
        rng.shuffle(keys)
        if args.max_setups:
            keys = keys[: args.max_setups]
        if args.max_rows:  # keep whole setups, stop once the row budget is spent
            budget, trimmed = args.max_rows, []
            for k in keys:
                if budget <= 0:
                    break
                trimmed.append(k)
                budget -= len(setups[k])
            keys = trimmed
        if not keys:
            print("[error] no setups to evaluate")
            return 2
        for i, key in enumerate(keys, 1):
            batch = evaluate_setup(setups[key], args.path_remap, registry)
            if not batch:
                skipped += len(setups[key])
                continue
            results.extend(batch)
            if i % 5 == 0:
                ok = sum(int(r["strict_ok"]) for r in results)
                print(
                    f"[eval] setup {i}/{len(keys)} rows={len(results)} strict_ok={ok}",
                    flush=True,
                )
        rows = [r for k in keys for r in setups[k]]
    else:
        if args.max_per_folder:
            per: Counter = Counter()
            kept = []
            for r in rows:
                key = r.get("config_folder", "")
                if per[key] >= args.max_per_folder:
                    continue
                per[key] += 1
                kept.append(r)
            rows = kept
        rng.shuffle(rows)
        if args.max_rows:
            rows = rows[: args.max_rows]
        if not rows:
            print("[error] no rows to evaluate")
            return 2
        for i, row in enumerate(rows, 1):
            res = evaluate_row(row, args.path_remap, registry)
            if res is None:
                skipped += 1
                continue
            results.append(res)
            if i % 25 == 0:
                ok = sum(int(r["strict_ok"]) for r in results)
                print(
                    f"[eval] {i}/{len(rows)} evaluated={len(results)} strict_ok={ok}",
                    flush=True,
                )

    if not results:
        print("[error] no images could be read")
        return 3

    fields = list(results[0].keys())
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    metrics = compute_metrics(results)
    summary = {
        "gt": args.gt.as_posix(),
        "mode": "setup_consensus" if args.consensus else "per_image",
        "heatmap_models": None if args.heatmap_models is None else args.heatmap_models.as_posix(),
        "sources": dict(Counter(str(r["source"]) for r in results).most_common()),
        "rows_requested": len(rows),
        "rows_evaluated": len(results),
        "rows_skipped_no_image": skipped,
        "tolerances": {
            "mm_per_px_rel": TOL_MM_PER_PX_REL,
            "x_px": TOL_X_PX,
            "y_zero_px": TOL_Y_ZERO_PX,
        },
        "profiles_used": sorted({profile_for(str(r["vendor"])).name for r in results}),
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_gallery:
        gallery = write_gallery(results, metrics, out_dir, args.max_cards, args.seed)
        print(f"[ok] gallery {gallery}")

    print(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))
    for vendor, m in metrics["by_vendor"].items():
        print(
            f"  {vendor:10s} rows={m['rows']:4d} strict={m['strict_ok_pct']:5.1f}% "
            f"calib={m['calib_ok_pct']:5.1f}% "
            f"err_x_med={m['err_x_px'].get('median')} err_y0_med={m['err_y_zero_px'].get('median')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
