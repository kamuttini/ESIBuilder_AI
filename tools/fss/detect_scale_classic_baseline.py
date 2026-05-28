#!/usr/bin/env python3
"""Classical baseline detector for scale line on full-frame image_samples.

Outputs:
- per-row predictions (x1,y1,x2,y2,length_mm,tick_mm)
- split metrics vs GT
- lightweight HTML with worst errors for quick inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class Row:
    sample_id: str
    split: str
    fss_path: str
    setup_id: str
    depth_index: int
    depth_mm: float
    x1_gt: float
    y1_gt: float
    x2_gt: float
    y2_gt: float
    length_mm_gt: float
    tick_mm_gt: float
    video_x: int
    video_y: int


@dataclass
class Pred:
    ok: bool
    reason: str
    image_path: str
    x1_pred: float
    y1_pred: float
    x2_pred: float
    y2_pred: float
    length_mm_pred: float
    tick_mm_pred: float


@dataclass
class LineCandidate:
    x: float
    yt: float
    yb: float
    score: float
    side: str
    tick_density: float
    tick_periodicity: float

    def as_line(self) -> Tuple[float, float, float, float]:
        return (self.x, self.yt, self.x, self.yb)


@dataclass
class MaskDetectionResult:
    best: Optional[LineCandidate] = None
    best_by_side: Dict[str, LineCandidate] = field(default_factory=dict)


@dataclass
class DetectResult:
    line: Optional[Tuple[float, float, float, float]]
    reason: str


def _f(text: str) -> float:
    return float(text.strip())


def _i(text: str) -> int:
    return int(float(text.strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120]


def load_rows(manifest: Path) -> List[Row]:
    out: List[Row] = []
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "split",
            "fss_path",
            "setup_id",
            "depth_index",
            "depth_mm",
            "x1",
            "y1",
            "x2",
            "y2",
            "length_mm",
            "tick_mm",
            "video_x_size",
            "video_y_size",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise RuntimeError(f"Manifest missing columns: {missing}")

        for r in reader:
            sp = (r.get("split") or "").strip().lower()
            if sp not in {"train", "val", "test"}:
                continue
            out.append(
                Row(
                    sample_id=r["sample_id"],
                    split=sp,
                    fss_path=r["fss_path"],
                    setup_id=r.get("setup_id", ""),
                    depth_index=_i(r["depth_index"]),
                    depth_mm=_f(r["depth_mm"]),
                    x1_gt=_f(r["x1"]),
                    y1_gt=_f(r["y1"]),
                    x2_gt=_f(r["x2"]),
                    y2_gt=_f(r["y2"]),
                    length_mm_gt=_f(r["length_mm"]),
                    tick_mm_gt=_f(r["tick_mm"]),
                    video_x=_i(r["video_x_size"]),
                    video_y=_i(r["video_y_size"]),
                )
            )
    return out


def find_full_frame_image(fss_path: str, setup_id: str, depth_index: int) -> Optional[Path]:
    fss = Path(fss_path)
    root = fss.parent.parent
    sid = setup_id.strip() or fss.stem.replace("setup_", "")
    image_samples = root / "image_samples"
    if not image_samples.exists():
        return None

    idx0 = max(0, depth_index - 1)
    stems = [
        f"image_depth_value_setup_{idx0}",
        f"image_depth_find_flip_ud_setup_{idx0}",
        f"image_depth_value_setup_{depth_index}",
        f"image_depth_find_flip_ud_setup_{depth_index}",
        "image_orientation_setup_0",
        "image_th_echo_negative_0",
        "image_th_probe_negative_0",
    ]
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for stem in stems:
        for ext in exts:
            p = image_samples / f"{stem}{ext}"
            if p.exists():
                return p

    for ext in ("image_*.png", "image_*.jpg", "image_*.jpeg", "image_*.bmp", "image_*.tif", "image_*.tiff"):
        found = [p for p in sorted(image_samples.glob(ext)) if not p.name.startswith("._")]
        if found:
            return found[0]

    return None


def _longest_true_run(mask: np.ndarray) -> Tuple[int, int, int]:
    # returns (start, end, length), end inclusive
    best_s = 0
    best_e = -1
    best_l = 0
    cur_s = None
    n = int(mask.shape[0])
    for i in range(n):
        if bool(mask[i]):
            if cur_s is None:
                cur_s = i
        else:
            if cur_s is not None:
                l = i - cur_s
                if l > best_l:
                    best_s, best_e, best_l = cur_s, i - 1, l
                cur_s = None
    if cur_s is not None:
        l = n - cur_s
        if l > best_l:
            best_s, best_e, best_l = cur_s, n - 1, l
    return best_s, best_e, best_l


def _run_starts(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=np.int32)
    prev = np.zeros_like(mask, dtype=bool)
    prev[1:] = mask[:-1]
    return np.flatnonzero(mask & (~prev))


def _tick_pattern_features(neigh: np.ndarray, center_idx: int) -> Tuple[float, float]:
    if neigh.size == 0:
        return 0.0, 0.0
    h, w = neigh.shape
    if w < 5:
        return 0.0, 0.0

    c0 = max(0, center_idx - 1)
    c1 = min(w, center_idx + 2)
    tick_zone = neigh.copy()
    tick_zone[:, c0:c1] = False

    row_counts = tick_zone.sum(axis=1).astype(np.float32)
    tick_rows = row_counts >= 2.0
    if tick_rows.size >= 9:
        tick_rows = np.convolve(tick_rows.astype(np.int16), np.ones(5, dtype=np.int16), mode="same") >= 2
    tick_density = float(np.mean(tick_rows))

    starts = _run_starts(tick_rows)
    periodicity = 0.0
    if starts.size >= 5:
        diffs = np.diff(starts).astype(np.float32)
        med = float(np.median(diffs))
        if 2.0 <= med <= 40.0:
            mad = float(np.median(np.abs(diffs - med)))
            periodicity = float(1.0 / (1.0 + mad))

    return tick_density, periodicity


def _build_adaptive_gray_mask(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    blur = np.asarray(Image.fromarray(gray_u8, mode="L").filter(ImageFilter.BoxBlur(radius=10.0)), dtype=np.float32)
    local = gray - blur

    shift = max(4, int(round(0.006 * W)))
    left = np.empty_like(gray)
    right = np.empty_like(gray)
    left[:, :shift] = gray[:, :1]
    left[:, shift:] = gray[:, :-shift]
    right[:, -shift:] = gray[:, -1:]
    right[:, :-shift] = gray[:, shift:]
    ridge = gray - 0.5 * (left + right)

    combined = 0.7 * local + 0.6 * ridge
    hi = float(np.percentile(combined, 93.0))
    mid = float(np.percentile(combined, 80.0))
    thr = max(6.0, mid, 0.55 * hi)
    bright_floor = float(np.percentile(gray, 18.0))

    gx = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gx_thr = float(np.percentile(gx, 94.5))
    local_thr = float(np.percentile(local, 72.0))

    mask = ((combined > thr) | ((gx > gx_thr) & (local > local_thr))) & (gray > bright_floor)

    if H >= 11:
        # Close tiny vertical holes to keep a contiguous center line.
        mask = np.convolve(mask.astype(np.int16).ravel(), np.ones(3, dtype=np.int16), mode="same").reshape(mask.shape) > 0
    return mask


def _is_possible_double_scale(best_by_side: Dict[str, LineCandidate], H: int, W: int) -> bool:
    left = best_by_side.get("left")
    right = best_by_side.get("right")
    if left is None or right is None:
        return False

    hi = max(left.score, right.score)
    lo = min(left.score, right.score)
    if hi < max(55.0, 0.14 * H):
        return False
    if lo < 0.93 * hi:
        return False

    min_span = min(left.yb - left.yt, right.yb - right.yt)
    if min_span < max(44.0, 0.16 * H):
        return False
    if abs(right.x - left.x) < 0.30 * W:
        return False

    min_tick_density = min(left.tick_density, right.tick_density)
    min_tick_period = min(left.tick_periodicity, right.tick_periodicity)
    return bool((min_tick_density >= 0.06) or (min_tick_period >= 0.45))


def _detect_line_from_mask(mask: np.ndarray) -> MaskDetectionResult:
    H, W = mask.shape
    y0 = int(0.04 * H)
    y1 = int(0.96 * H)
    if y1 <= y0:
        return MaskDetectionResult()

    # Prefer right side, then keep left fallback.
    rois = [
        (0.55, 0.98, "right", 1.14),
        (0.02, 0.45, "left", 1.00),
        (0.45, 0.98, "right", 1.08),
        (0.02, 0.55, "left", 0.97),
    ]
    out = MaskDetectionResult()
    best_score = -1.0

    for rx0, rx1, side, roi_weight in rois:
        x0 = max(0, min(W - 1, int(rx0 * W)))
        x1 = max(1, min(W, int(rx1 * W)))
        roi = mask[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        col_energy = roi.sum(axis=0).astype(np.float32)
        if col_energy.max() < 20:
            continue

        # Smooth and evaluate top candidates instead of only argmax.
        k = 9
        if col_energy.size >= k:
            kernel = np.ones(k, dtype=np.float32) / k
            col_smooth = np.convolve(col_energy, kernel, mode="same")
        else:
            col_smooth = col_energy

        k_top = min(16, col_smooth.size)
        top_idx = np.argpartition(col_smooth, -k_top)[-k_top:]
        top_idx = top_idx[np.argsort(col_smooth[top_idx])[::-1]]

        for x_local in top_idx:
            x = x0 + int(x_local)
            xl = max(x0, x - 4)
            xr = min(x1, x + 5)
            col_block = mask[y0:y1, xl:xr]
            if col_block.size == 0:
                continue

            row_hits = col_block.any(axis=1)
            # Fill small gaps
            if row_hits.size >= 11:
                row_hits = np.convolve(row_hits.astype(np.int8), np.ones(11, dtype=np.int8), mode="same") > 0
            rs, re, rl = _longest_true_run(row_hits)
            if rl < max(28, int(0.06 * H)):
                continue

            yt = y0 + rs
            yb = y0 + re

            # x refinement on run only
            band = mask[yt : yb + 1, x0:x1]
            band_energy = band.sum(axis=0).astype(np.float32)
            if band_energy.max() >= 8:
                x_local2 = int(np.argmax(band_energy))
                x = x0 + x_local2

            # Tick-support score around line
            tx0 = max(0, x - 24)
            tx1 = min(W, x + 25)
            neigh = mask[yt : yb + 1, tx0:tx1]
            tick_support = float(neigh.sum())
            tick_density, tick_periodicity = _tick_pattern_features(neigh, center_idx=(x - tx0))
            score = (
                roi_weight
                * (
                    float(rl)
                    + 0.05 * float(col_smooth[x_local])
                    + 0.0017 * tick_support
                    + 18.0 * tick_density
                    + 12.0 * tick_periodicity
                )
            )
            # Bias right side even when both sides are valid.
            if x >= int(0.52 * W):
                score += 4.0
            else:
                score -= 1.0

            cand = LineCandidate(
                x=float(x),
                yt=float(yt),
                yb=float(yb),
                score=float(score),
                side=side,
                tick_density=tick_density,
                tick_periodicity=tick_periodicity,
            )
            prev_side = out.best_by_side.get(side)
            if prev_side is None or cand.score > prev_side.score:
                out.best_by_side[side] = cand
            if score > best_score:
                best_score = score
                out.best = cand

    return out


def detect_line_classic(img: Image.Image) -> DetectResult:
    arr_hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    h = arr_hsv[:, :, 0].astype(np.int16)
    s = arr_hsv[:, :, 1].astype(np.int16)
    v = arr_hsv[:, :, 2].astype(np.int16)

    # candidate color mask for yellow/cyan/bright ticks
    yellow = (h >= 16) & (h <= 72) & (s >= 55) & (v >= 95)
    cyan = (h >= 80) & (h <= 142) & (s >= 45) & (v >= 90)
    white = (s <= 35) & (v >= 215)
    mask = (yellow | cyan | white)
    color_res = _detect_line_from_mask(mask)

    # Fallback with local-adaptive contrast (no OCR dependency).
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mask2 = _build_adaptive_gray_mask(gray)
    gray_res = _detect_line_from_mask(mask2)

    side_best: Dict[str, LineCandidate] = {}
    all_best = []
    for res in (color_res, gray_res):
        if res.best is not None:
            all_best.append(res.best)
        for side, cand in res.best_by_side.items():
            prev = side_best.get(side)
            if prev is None or cand.score > prev.score:
                side_best[side] = cand

    if not all_best:
        return DetectResult(line=None, reason="line_not_found")

    if _is_possible_double_scale(side_best, H=gray.shape[0], W=gray.shape[1]):
        return DetectResult(line=None, reason="possible_double_scale")

    best = max(all_best, key=lambda c: c.score)
    return DetectResult(line=best.as_line(), reason="ok")


def fit_length_model(train_rows: Sequence[Row]) -> Dict[float, float]:
    grouped: Dict[float, List[float]] = defaultdict(list)
    for r in train_rows:
        key = round(r.depth_mm, 3)
        grouped[key].append(r.length_mm_gt)
    depth_to_len = {k: float(statistics.median(v)) for k, v in grouped.items()}
    return depth_to_len


def predict_length(depth_mm: float, depth_to_len: Dict[float, float]) -> float:
    if not depth_to_len:
        return max(1.0, depth_mm)
    key = round(depth_mm, 3)
    if key in depth_to_len:
        return depth_to_len[key]
    nearest = min(depth_to_len.keys(), key=lambda d: abs(d - key))
    return depth_to_len[nearest]


def y_error_orientation_invariant(y1_pred: float, y2_pred: float, y1_gt: float, y2_gt: float) -> Tuple[float, bool]:
    direct = abs(y1_pred - y1_gt) + abs(y2_pred - y2_gt)
    swap = abs(y1_pred - y2_gt) + abs(y2_pred - y1_gt)
    if swap < direct:
        return 0.5 * swap, True
    return 0.5 * direct, False


def _row_filter_text(row: Row) -> str:
    return f"{row.fss_path} {row.setup_id} {row.sample_id}".lower()


def row_matches_exclusion(row: Row, label: str) -> bool:
    text = _row_filter_text(row)
    if label == "fusion":
        return "fusion" in text
    if label == "negative":
        return "negative" in text
    if label == "proibite":
        return any(
            tok in text
            for tok in (
                "proibite",
                "proibita",
                "proibiti",
                "proibito",
                "prohibited",
                "sbagliat",
                "non usare",
                "non_usare",
            )
        )
    return False


def evaluate_rows(rows: Sequence[Row], preds: Dict[str, Pred]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r.split == split]
        if not subset:
            continue
        ok_rows = []
        x_err = []
        y_err = []
        len_err = []
        tick_err = []
        for r in subset:
            p = preds.get(r.sample_id)
            if p is None or not p.ok:
                continue
            ok_rows.append(r)
            x_gt = 0.5 * (r.x1_gt + r.x2_gt)
            x_pr = 0.5 * (p.x1_pred + p.x2_pred)
            x_err.append(abs(x_pr - x_gt))
            ye, _ = y_error_orientation_invariant(p.y1_pred, p.y2_pred, r.y1_gt, r.y2_gt)
            y_err.append(ye)
            len_err.append(abs(p.length_mm_pred - r.length_mm_gt))
            tick_err.append(abs(p.tick_mm_pred - r.tick_mm_gt))

        coverage = len(ok_rows) / max(1, len(subset))
        out[split] = {
            "rows": float(len(subset)),
            "rows_ok": float(len(ok_rows)),
            "coverage": float(coverage),
            "mae_x_px": float(np.mean(x_err)) if x_err else float("nan"),
            "mae_y_px": float(np.mean(y_err)) if y_err else float("nan"),
            "mae_length_mm": float(np.mean(len_err)) if len_err else float("nan"),
            "mae_tick_mm": float(np.mean(tick_err)) if tick_err else float("nan"),
        }
    return out


def draw_compare_preview(
    image_path: Path,
    out_path: Path,
    pred: Pred,
    row: Row,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # GT in green
    draw.line((row.x1_gt, row.y1_gt, row.x2_gt, row.y2_gt), fill=(0, 255, 100, 255), width=6)
    # Pred in magenta
    draw.line((pred.x1_pred, pred.y1_pred, pred.x2_pred, pred.y2_pred), fill=(255, 0, 220, 255), width=5)

    # Endpoints markers
    r = 5
    for x, y, c in [
        (row.x1_gt, row.y1_gt, (0, 255, 100, 255)),
        (row.x2_gt, row.y2_gt, (0, 255, 100, 255)),
        (pred.x1_pred, pred.y1_pred, (255, 0, 220, 255)),
        (pred.x2_pred, pred.y2_pred, (255, 0, 220, 255)),
    ]:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def build_html(
    rows: Sequence[Row],
    preds: Dict[str, Pred],
    out_dir: Path,
    max_rows: int,
) -> None:
    html_path = out_dir / "classic_scale_review.html"
    preview_dir = out_dir / "classic_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scored = []
    for r in rows:
        p = preds.get(r.sample_id)
        if p is None or not p.ok:
            continue
        x_gt = 0.5 * (r.x1_gt + r.x2_gt)
        x_pr = 0.5 * (p.x1_pred + p.x2_pred)
        xe = abs(x_pr - x_gt)
        ye, _ = y_error_orientation_invariant(p.y1_pred, p.y2_pred, r.y1_gt, r.y2_gt)
        le = abs(p.length_mm_pred - r.length_mm_gt)
        score = xe + ye + 2.0 * le
        scored.append((score, xe, ye, le, r, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    worst = scored[: max_rows]

    rows_html = []
    for i, (score, xe, ye, le, r, p) in enumerate(worst, start=1):
        img_src = Path(p.image_path)
        out_img = preview_dir / f"{i:04d}_{_safe_slug(r.sample_id)}.jpg"
        try:
            draw_compare_preview(img_src, out_img, p, r)
            img_tag = f'<a href="classic_previews/{out_img.name}" target="_blank"><img src="classic_previews/{out_img.name}" loading="lazy" /></a>'
        except Exception:
            img_tag = "<span>preview_error</span>"

        rows_html.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{r.sample_id}</td>"
            f"<td>{r.split}</td>"
            f"<td>{score:.2f}</td>"
            f"<td>{xe:.2f}</td>"
            f"<td>{ye:.2f}</td>"
            f"<td>{le:.2f}</td>"
            f"<td>{r.depth_mm:.1f}</td>"
            f"<td>{r.length_mm_gt:.1f} / {p.length_mm_pred:.1f}</td>"
            f"<td>{Path(p.image_path).name}</td>"
            "</tr>"
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Classic Scale Baseline Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 520px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Classic Scale Baseline - Worst Cases</h1>
  <p>Overlay: <b>green=GT</b>, <b>magenta=pred</b></p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Preview</th><th>sample_id</th><th>split</th>
        <th>score</th><th>x_err_px</th><th>y_err_px</th><th>len_err_mm</th>
        <th>depth_mm</th><th>len_gt / len_pred</th><th>source</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical baseline detector for scale line.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/23_scale_manifest_ssd_esi1_n1_leakfree/manifest_scale_train_val_test_leakfree.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/31_scale_classic_baseline"),
    )
    p.add_argument("--max-html-rows", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="Optional max rows for quick experiments.")
    p.add_argument(
        "--exclude-fusion",
        action="store_true",
        help="Exclude rows with 'fusion' in setup/fss path (double-scale prone).",
    )
    p.add_argument(
        "--exclude-negative",
        action="store_true",
        help="Exclude rows with 'negative' in setup/fss/sample identifiers.",
    )
    p.add_argument(
        "--exclude-proibite",
        action="store_true",
        help="Exclude rows with forbidden/proibite markers in setup/fss/sample identifiers.",
    )
    p.add_argument(
        "--exclude-prohibited",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows_loaded = load_rows(manifest)
    exclude_labels: List[str] = []
    if args.exclude_fusion:
        exclude_labels.append("fusion")
    if args.exclude_negative:
        exclude_labels.append("negative")
    if args.exclude_proibite or args.exclude_prohibited:
        exclude_labels.append("proibite")

    rows_excluded_by_filter: Counter[str] = Counter()
    rows_excluded_total = 0
    if exclude_labels:
        rows = []
        for r in rows_loaded:
            matched = [label for label in exclude_labels if row_matches_exclusion(r, label)]
            if matched:
                rows_excluded_total += 1
                for label in matched:
                    rows_excluded_by_filter[label] += 1
            else:
                rows.append(r)
    else:
        rows = list(rows_loaded)

    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows to process after filtering.")

    split_rows = defaultdict(list)
    for r in rows:
        split_rows[r.split].append(r)

    length_model = fit_length_model(split_rows["train"])

    preds: Dict[str, Pred] = {}
    reason_counts = Counter()
    for idx, r in enumerate(rows, start=1):
        img_path = find_full_frame_image(r.fss_path, r.setup_id, r.depth_index)
        if img_path is None:
            preds[r.sample_id] = Pred(
                ok=False,
                reason="missing_image",
                image_path="",
                x1_pred=float("nan"),
                y1_pred=float("nan"),
                x2_pred=float("nan"),
                y2_pred=float("nan"),
                length_mm_pred=float("nan"),
                tick_mm_pred=0.5,
            )
            reason_counts["missing_image"] += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            det = detect_line_classic(img)
        except Exception:
            det = DetectResult(line=None, reason="runtime_error")

        if det.line is None:
            preds[r.sample_id] = Pred(
                ok=False,
                reason=det.reason,
                image_path=img_path.as_posix(),
                x1_pred=float("nan"),
                y1_pred=float("nan"),
                x2_pred=float("nan"),
                y2_pred=float("nan"),
                length_mm_pred=float("nan"),
                tick_mm_pred=0.5,
            )
            reason_counts[det.reason] += 1
            continue

        x1, y1, x2, y2 = det.line
        length_pred = predict_length(r.depth_mm, length_model)
        preds[r.sample_id] = Pred(
            ok=True,
            reason="ok",
            image_path=img_path.as_posix(),
            x1_pred=x1,
            y1_pred=y1,
            x2_pred=x2,
            y2_pred=y2,
            length_mm_pred=length_pred,
            tick_mm_pred=0.5,
        )

        if idx % 500 == 0:
            print(f"processed {idx}/{len(rows)}", flush=True)

    metrics = evaluate_rows(rows, preds)

    pred_csv = out_dir / "classic_scale_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sample_id",
                "split",
                "status",
                "reason",
                "image_path",
                "x1_gt",
                "y1_gt",
                "x2_gt",
                "y2_gt",
                "x1_pred",
                "y1_pred",
                "x2_pred",
                "y2_pred",
                "length_mm_gt",
                "length_mm_pred",
                "tick_mm_gt",
                "tick_mm_pred",
                "x_err_px",
                "y_err_px",
                "length_err_mm",
            ]
        )
        for r in rows:
            p = preds[r.sample_id]
            if p.ok:
                x_gt = 0.5 * (r.x1_gt + r.x2_gt)
                x_pr = 0.5 * (p.x1_pred + p.x2_pred)
                xe = abs(x_pr - x_gt)
                ye, _ = y_error_orientation_invariant(p.y1_pred, p.y2_pred, r.y1_gt, r.y2_gt)
                le = abs(p.length_mm_pred - r.length_mm_gt)
            else:
                xe = float("nan")
                ye = float("nan")
                le = float("nan")

            w.writerow(
                [
                    r.sample_id,
                    r.split,
                    "ok" if p.ok else "fail",
                    p.reason,
                    p.image_path,
                    r.x1_gt,
                    r.y1_gt,
                    r.x2_gt,
                    r.y2_gt,
                    p.x1_pred,
                    p.y1_pred,
                    p.x2_pred,
                    p.y2_pred,
                    r.length_mm_gt,
                    p.length_mm_pred,
                    r.tick_mm_gt,
                    p.tick_mm_pred,
                    xe,
                    ye,
                    le,
                ]
            )

    summary = {
        "manifest": manifest.as_posix(),
        "rows_total_loaded": len(rows_loaded),
        "exclusion_filters_enabled": exclude_labels,
        "rows_excluded_total": rows_excluded_total,
        "rows_excluded_by_filter": dict(rows_excluded_by_filter),
        "rows_excluded_fusion": int(rows_excluded_by_filter.get("fusion", 0)),
        "rows_excluded_negative": int(rows_excluded_by_filter.get("negative", 0)),
        "rows_excluded_proibite": int(rows_excluded_by_filter.get("proibite", 0)),
        "rows_total": len(rows),
        "reasons": dict(reason_counts),
        "metrics": metrics,
        "predictions_csv": pred_csv.as_posix(),
    }
    (out_dir / "classic_scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    build_html(rows, preds, out_dir, max_rows=args.max_html_rows)

    print(json.dumps(summary, indent=2))
    print(f"HTML: {out_dir / 'classic_scale_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
