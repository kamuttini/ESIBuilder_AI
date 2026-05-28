#!/usr/bin/env python3
"""Refine scale line from bbox predictions using ticks + OCR monotonic numbers.

Approach:
1) Network predicts scale ROI bbox.
2) Post-processing inside predicted ROI finds tick column and tick rows.
3) OCR finds numeric labels, checks monotonic order (increasing/decreasing),
   and snaps top/bottom vertices to nearest ticks around first/last numbers.
4) Final line is placed just to the right (or left) of the tick column.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import io
import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class BBoxPredRow:
    sample_id: str
    split: str
    image_path: str
    x1_gt: float
    y1_gt: float
    x2_gt: float
    y2_gt: float
    x1_pred: float
    y1_pred: float
    x2_pred: float
    y2_pred: float
    x_err_px: float
    y_err_px: float
    iou: float


@dataclass(frozen=True)
class ManifestLine:
    sample_id: str
    split: str
    x_gt_video: float
    y_top_gt_video: float
    y_bottom_gt_video: float
    video_w: float
    video_h: float
    label_side: int


@dataclass(frozen=True)
class TickDet:
    x_local: float
    score: float
    quality: float
    runs: int
    coverage: float
    periodicity: float
    vertical_support: float
    tick_rows_local: Tuple[int, ...]
    tick_step: float
    orientation: str  # "ticks_left" or "ticks_right"


@dataclass(frozen=True)
class OCRToken:
    value: int
    conf: float
    x_center: float
    y_center: float
    x1: float
    y1: float
    x2: float
    y2: float
    text_raw: str


def _f(text: str) -> float:
    return float(str(text).strip())


def _i(text: str) -> int:
    return int(float(str(text).strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))[:120]


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    b = mask.astype(np.int8)
    starts = int(np.sum((b[1:] == 1) & (b[:-1] == 0)))
    if b[0] == 1:
        starts += 1
    return starts


def _run_starts(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=np.int32)
    prev = np.zeros_like(mask, dtype=bool)
    prev[1:] = mask[:-1]
    return np.flatnonzero(mask & (~prev))


def _run_centers(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=np.float32)
    out: List[float] = []
    in_run = False
    s = 0
    for i, v in enumerate(mask.astype(bool)):
        if v and not in_run:
            s = i
            in_run = True
        if (not v) and in_run:
            e = i - 1
            out.append(0.5 * (s + e))
            in_run = False
    if in_run:
        e = mask.size - 1
        out.append(0.5 * (s + e))
    return np.asarray(out, dtype=np.float32) if out else np.zeros(0, dtype=np.float32)


def _periodicity_score(mask: np.ndarray) -> float:
    starts = _run_starts(mask)
    if starts.size < 5:
        return 0.0
    diffs = np.diff(starts).astype(np.float32)
    med = float(np.median(diffs))
    if med < 2.0 or med > 55.0:
        return 0.0
    mad = float(np.median(np.abs(diffs - med)))
    return float(1.0 / (1.0 + mad))


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    xa = int(max(0, min(w - 1, math.floor(min(x1, x2)))))
    xb = int(max(0, min(w, math.ceil(max(x1, x2)))))
    ya = int(max(0, min(h - 1, math.floor(min(y1, y2)))))
    yb = int(max(0, min(h, math.ceil(max(y1, y2)))))
    if xb <= xa:
        xb = min(w, xa + 1)
    if yb <= ya:
        yb = min(h, ya + 1)
    return xa, ya, xb, yb


def load_bbox_rows(pred_csv: Path, split_name: str) -> List[BBoxPredRow]:
    out: List[BBoxPredRow] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                out.append(
                    BBoxPredRow(
                        sample_id=str(r["sample_id"]),
                        split=split_name,
                        image_path=str(r["image_path"]),
                        x1_gt=_f(r["x1_gt"]),
                        y1_gt=_f(r["y1_gt"]),
                        x2_gt=_f(r["x2_gt"]),
                        y2_gt=_f(r["y2_gt"]),
                        x1_pred=_f(r["x1_pred"]),
                        y1_pred=_f(r["y1_pred"]),
                        x2_pred=_f(r["x2_pred"]),
                        y2_pred=_f(r["y2_pred"]),
                        x_err_px=_f(r.get("x_err_px", "nan")),
                        y_err_px=_f(r.get("y_err_px", "nan")),
                        iou=_f(r.get("iou", "nan")),
                    )
                )
            except Exception:
                continue
    return out


def build_image_path_resolver(dataset_roots: Sequence[Path]) -> Callable[[str], str]:
    roots = [Path(p).expanduser().resolve() for p in dataset_roots if str(p).strip()]
    marker = "/SSD_esi1_n1/"
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def _norm_cfg(name: str) -> str:
        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    cfg_index: Dict[Tuple[Path, str], Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for d in root.iterdir():
            if d.is_dir():
                cfg_index[(root, _norm_cfg(d.name))] = d

    def _resolve(raw_path: str) -> str:
        p = Path(str(raw_path))
        if p.exists():
            return p.as_posix()

        s = str(raw_path).replace("\\", "/")
        rel: Optional[str] = None
        if marker in s:
            rel = s.split(marker, 1)[1]
        elif s.startswith("/Volumes/SSD_esi1_n1/"):
            rel = s[len("/Volumes/SSD_esi1_n1/") :]

        if rel is not None:
            for root in roots:
                cand = root / rel
                if cand.exists():
                    return cand.as_posix()

        # Alternative name patterns in local dataset (depth_value vs biplana, etc.).
        if rel is not None:
            rel_p = Path(rel)
            if len(rel_p.parts) >= 3:
                cfg_name = rel_p.parts[0]
                file_name = rel_p.name
                file_ext = rel_p.suffix.lower()
                cfg_norm = _norm_cfg(cfg_name)
                setup_match = re.search(r"setup_(\d+)", file_name)
                idx = None
                if setup_match:
                    try:
                        idx = int(setup_match.group(1))
                    except Exception:
                        idx = None

                for root in roots:
                    cfg_dir = root / cfg_name
                    if not cfg_dir.exists():
                        cfg_dir = cfg_index.get((root, cfg_norm), cfg_dir)
                    img_dir = cfg_dir / "image_samples"
                    if not img_dir.exists():
                        continue

                    # 1) same basename, different extension.
                    stem = rel_p.stem
                    ext_order = [file_ext] + [e for e in exts if e != file_ext]
                    for ext in ext_order:
                        cand = img_dir / f"{stem}{ext}"
                        if cand.exists():
                            return cand.as_posix()

                    # 2) known alternative stems for same setup index.
                    if idx is not None:
                        idx0 = max(0, idx - 1)
                        alt_stems = [
                            f"image_depth_value_setup_{idx}",
                            f"image_depth_find_flip_ud_setup_{idx}",
                            f"image_biplana_setup_{idx}",
                            f"image_orientation_setup_{idx}",
                            f"image_depth_value_setup_{idx0}",
                            f"image_depth_find_flip_ud_setup_{idx0}",
                            f"image_biplana_setup_{idx0}",
                            f"image_orientation_setup_{idx0}",
                        ]
                        for stem_alt in alt_stems:
                            for ext in exts:
                                cand = img_dir / f"{stem_alt}{ext}"
                                if cand.exists():
                                    return cand.as_posix()

                    # 3) last resort in folder: first image_* file.
                    fallback = sorted(
                        [x for x in img_dir.glob("image_*") if x.is_file() and not x.name.startswith("._")]
                    )
                    if fallback:
                        return fallback[0].as_posix()

        return str(raw_path)

    return _resolve


def load_manifest_lines(manifest_csv: Path) -> Dict[str, ManifestLine]:
    out: Dict[str, ManifestLine] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        required = {"sample_id", "split", "x1", "x2", "y1", "y2", "video_x_size", "video_y_size", "label_side"}
        if not required.issubset(set(rd.fieldnames or [])):
            missing = sorted(required.difference(set(rd.fieldnames or [])))
            raise RuntimeError(f"Manifest missing columns: {missing}")
        for r in rd:
            try:
                sid = str(r["sample_id"]).strip()
                x1 = _f(r["x1"])
                x2 = _f(r["x2"])
                y1 = _f(r["y1"])
                y2 = _f(r["y2"])
                out[sid] = ManifestLine(
                    sample_id=sid,
                    split=str(r["split"]).strip().lower(),
                    x_gt_video=0.5 * (x1 + x2),
                    y_top_gt_video=min(y1, y2),
                    y_bottom_gt_video=max(y1, y2),
                    video_w=float(_f(r["video_x_size"])),
                    video_h=float(_f(r["video_y_size"])),
                    label_side=-1 if _i(r["label_side"]) < 0 else 1,
                )
            except Exception:
                continue
    return out


def baseline_line_from_bbox(
    row: BBoxPredRow,
    side: int,
    line_half_px: float,
    pad_y_px: float,
    shift_y_px: float,
) -> Tuple[float, float, float]:
    edge_off = float(line_half_px) + 8.0
    if side < 0:
        x_line = float(row.x2_pred) - edge_off
    else:
        x_line = float(row.x1_pred) + edge_off
    y_top = float(row.y1_pred) + float(pad_y_px) - float(shift_y_px)
    y_bottom = float(row.y2_pred) - float(pad_y_px) - float(shift_y_px)
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top
    return x_line, y_top, y_bottom


def detect_tick_line_in_roi(
    gray_roi: np.ndarray,
    x_hint_local: float,
    y_top_hint_local: float,
    y_bottom_hint_local: float,
    search_radius_x: int,
) -> Optional[TickDet]:
    h, w = gray_roi.shape
    if h < 30 or w < 30:
        return None

    gray_u8 = np.clip(gray_roi, 0.0, 255.0).astype(np.uint8)
    blur = np.asarray(Image.fromarray(gray_u8, mode="L").filter(ImageFilter.BoxBlur(radius=2.0)), dtype=np.float32)
    hi = np.abs(gray_roi - blur)
    gx = np.zeros_like(hi, dtype=np.float32)
    gx[:, 1:-1] = np.abs(hi[:, 2:] - hi[:, :-2])
    gy = np.zeros_like(hi, dtype=np.float32)
    gy[1:-1, :] = np.abs(hi[2:, :] - hi[:-2, :])
    feat = 0.72 * gx + 0.28 * gy

    yp0 = int(max(0.0, min(float(y_top_hint_local), float(y_bottom_hint_local))))
    yp1 = int(min(float(h), max(float(y_top_hint_local), float(y_bottom_hint_local))))
    pad = int(max(10.0, 0.08 * h))
    y0 = max(0, yp0 - pad)
    y1 = min(h, yp1 + pad)
    if y1 - y0 < int(0.18 * h):
        y0 = int(0.05 * h)
        y1 = int(0.95 * h)
    if y1 <= y0:
        return None

    cx = int(round(float(x_hint_local)))
    if not math.isfinite(x_hint_local):
        cx = int(0.8 * w)
    x_min = max(2, cx - int(search_radius_x))
    x_max = min(w - 3, cx + int(search_radius_x))
    if x_max <= x_min:
        x_min = 2
        x_max = w - 3
    if x_max <= x_min:
        return None

    base_thr = float(np.percentile(feat[y0:y1, :], 82.0))

    best_score = -1e9
    best: Optional[TickDet] = None

    for x in range(x_min, x_max + 1):
        xl0 = max(0, x - 14)
        xl1 = max(xl0 + 1, x)
        xr0 = min(w - 1, x + 1)
        xr1 = min(w, x + 9)
        xc0 = max(0, x - 1)
        xc1 = min(w, x + 2)
        if xl1 <= xl0 or xr1 <= xr0 or xc1 <= xc0:
            continue

        left = feat[y0:y1, xl0:xl1]
        right = feat[y0:y1, xr0:xr1]
        center = feat[y0:y1, xc0:xc1]
        if left.size == 0 or right.size == 0 or center.size == 0:
            continue

        left_m = left.mean(axis=1)
        right_m = right.mean(axis=1)
        center_m = center.mean(axis=1)

        for orientation in ("ticks_left", "ticks_right"):
            if orientation == "ticks_left":
                main = left_m
                opp = right_m
                row_hits = (left_m > max(3.5, 0.75 * base_thr)) & (left_m > right_m * 1.18)
            else:
                main = right_m
                opp = left_m
                row_hits = (right_m > max(3.5, 0.75 * base_thr)) & (right_m > left_m * 1.18)

            if row_hits.size >= 9:
                row_hits = np.convolve(row_hits.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0

            coverage = float(np.mean(row_hits))
            if coverage < 0.02 or coverage > 0.80:
                continue

            runs = _count_runs(row_hits)
            if runs < 2:
                continue

            centers = _run_centers(row_hits)
            if centers.size < 2:
                continue

            periodicity = _periodicity_score(row_hits)
            center_diffs = np.diff(centers).astype(np.float32)
            tick_step = float(np.median(center_diffs)) if center_diffs.size else float("nan")
            vertical_support = float(np.mean(center_m > max(3.0, 0.68 * base_thr)))
            opp_penalty = float(np.mean(np.maximum(0.0, opp - 0.88 * main)))

            dist = abs(float(x) - float(cx))
            dist_term = max(0.0, 36.0 - 0.24 * dist)
            score = (
                210.0 * coverage
                + 5.2 * float(runs)
                + 42.0 * periodicity
                + 73.0 * vertical_support
                - 8.0 * opp_penalty
                + dist_term
            )

            run_pref = min(1.0, runs / 16.0)
            cov_pref = max(0.0, 1.0 - abs(coverage - 0.22) / 0.22)
            per_pref = min(1.0, periodicity / 0.60)
            vert_pref = min(1.0, vertical_support / 0.50)
            quality = float(np.clip(0.33 * run_pref + 0.25 * cov_pref + 0.22 * per_pref + 0.20 * vert_pref, 0.0, 1.0))

            if score > best_score:
                best_score = score
                tick_rows = tuple(int(round(y0 + float(c))) for c in centers.tolist())
                best = TickDet(
                    x_local=float(x),
                    score=float(score),
                    quality=quality,
                    runs=int(runs),
                    coverage=float(coverage),
                    periodicity=float(periodicity),
                    vertical_support=float(vertical_support),
                    tick_rows_local=tick_rows,
                    tick_step=float(tick_step),
                    orientation=orientation,
                )

    return best


def _run_tesseract_tsv(img: Image.Image, psm: int) -> List[Dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="scale_ocr_") as td:
        in_path = Path(td) / "crop.png"
        img.save(in_path)
        cmd = [
            "tesseract",
            str(in_path),
            "stdout",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "-l",
            "eng",
            "-c",
            "tessedit_char_whitelist=0123456789",
            "tsv",
        ]
        try:
            cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=14)
        except FileNotFoundError:
            return []
        except subprocess.TimeoutExpired:
            return []
        if cp.returncode != 0 or not cp.stdout.strip():
            return []
        rows = list(csv.DictReader(io.StringIO(cp.stdout), delimiter="\t"))
        return rows


def detect_numeric_tokens(
    gray_strip: np.ndarray,
    min_conf: float,
) -> List[OCRToken]:
    if gray_strip.size == 0:
        return []
    h, w = gray_strip.shape
    if h < 8 or w < 8:
        return []

    arr = gray_strip.astype(np.float32)
    p5 = float(np.percentile(arr, 5.0))
    p98 = float(np.percentile(arr, 98.0))
    if p98 <= p5 + 1.0:
        arr_n = np.clip(arr, 0.0, 255.0)
    else:
        arr_n = np.clip((arr - p5) * (255.0 / (p98 - p5)), 0.0, 255.0)
    arr_u8 = arr_n.astype(np.uint8)

    scale = 2
    im_up = Image.fromarray(arr_u8, mode="L").resize((w * scale, h * scale), Image.Resampling.BICUBIC)
    up_np = np.asarray(im_up, dtype=np.uint8)
    thr = float(np.percentile(up_np, 58.0))
    bin_np = np.where(up_np > thr, 255, 0).astype(np.uint8)
    im_bin = Image.fromarray(bin_np, mode="L")

    tokens: List[OCRToken] = []
    for variant in (im_bin, im_up):
        rows = _run_tesseract_tsv(variant, psm=6)
        for r in rows:
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            txt_digits = re.sub(r"[^0-9]", "", txt)
            if not txt_digits:
                continue
            if len(txt_digits) > 4:
                continue
            try:
                conf = float(str(r.get("conf", "-1")).strip())
            except Exception:
                conf = -1.0
            if conf < min_conf:
                continue
            try:
                left = float(r.get("left", "0")) / float(scale)
                top = float(r.get("top", "0")) / float(scale)
                width = float(r.get("width", "0")) / float(scale)
                height = float(r.get("height", "0")) / float(scale)
            except Exception:
                continue
            if width <= 1.0 or height <= 1.0:
                continue
            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height
            try:
                value = int(txt_digits)
            except Exception:
                continue
            tokens.append(
                OCRToken(
                    value=value,
                    conf=conf,
                    x_center=0.5 * (x1 + x2),
                    y_center=0.5 * (y1 + y2),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    text_raw=txt,
                )
            )

    # De-duplicate nearby tokens (keep highest confidence).
    tokens = sorted(tokens, key=lambda t: t.conf, reverse=True)
    kept: List[OCRToken] = []
    for t in tokens:
        duplicate = False
        for k in kept:
            if abs(t.x_center - k.x_center) <= 7.0 and abs(t.y_center - k.y_center) <= 7.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(t)
    kept.sort(key=lambda t: t.y_center)
    return kept


def monotonic_direction(tokens: Sequence[OCRToken]) -> Tuple[str, float]:
    if len(tokens) < 2:
        return "none", 0.0
    vals = [int(t.value) for t in sorted(tokens, key=lambda t: t.y_center)]
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    diffs_nz = [d for d in diffs if d != 0]
    if not diffs_nz:
        return "flat", 0.0
    pos = sum(1 for d in diffs_nz if d > 0)
    neg = sum(1 for d in diffs_nz if d < 0)
    if pos >= neg:
        direction = "increasing"
        match = sum(1 for d in diffs_nz if d > 0)
    else:
        direction = "decreasing"
        match = sum(1 for d in diffs_nz if d < 0)
    consistency = float(match / max(1, len(diffs_nz)))
    return direction, consistency


def nearest_tick_row(y_target: float, tick_rows: Sequence[float], max_dist: float) -> Optional[float]:
    if not tick_rows:
        return None
    best = min(tick_rows, key=lambda y: abs(float(y) - float(y_target)))
    if abs(float(best) - float(y_target)) <= float(max_dist):
        return float(best)
    return None


def _dedupe_tick_rows(tick_rows: Sequence[float], min_gap: float = 2.0) -> List[float]:
    vals = sorted(float(v) for v in tick_rows if math.isfinite(float(v)))
    if not vals:
        return []
    groups: List[List[float]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= float(min_gap):
            groups[-1].append(v)
        else:
            groups.append([v])
    out = [float(np.mean(g)) for g in groups if g]
    return out


def select_uniform_tick_sequence(
    tick_rows: Sequence[float],
    y_top_hint: float,
    y_bottom_hint: float,
    step_hint: float,
) -> Tuple[List[float], float]:
    vals = _dedupe_tick_rows(tick_rows=tick_rows, min_gap=2.0)
    if len(vals) < 3:
        return vals, float("nan")

    diffs = np.diff(np.asarray(vals, dtype=np.float32))
    valid_diffs = [float(d) for d in diffs if 3.0 <= float(d) <= 70.0]
    if math.isfinite(float(step_hint)) and 3.0 <= float(step_hint) <= 70.0:
        step = float(step_hint)
    elif valid_diffs:
        step = float(np.median(np.asarray(valid_diffs, dtype=np.float32)))
    else:
        return vals, float("nan")

    tol = max(2.0, 0.22 * step)
    best_seq: List[float] = []
    best_score = -1e9

    for a in vals:
        picked: List[Tuple[float, int]] = []
        last_k = -10**9
        for y in vals:
            k = int(round((float(y) - float(a)) / max(step, 1e-6)))
            if k < 0:
                continue
            y_exp = float(a) + float(k) * float(step)
            if abs(float(y) - y_exp) > tol:
                continue
            if k <= last_k:
                continue
            picked.append((float(y), int(k)))
            last_k = int(k)
        if len(picked) < 3:
            continue

        ys = [p[0] for p in picked]
        if len(ys) >= 2:
            reg = -float(np.mean(np.abs(np.diff(np.asarray(ys, dtype=np.float32)) - float(step))))
        else:
            reg = 0.0
        hint_pen = 0.015 * (abs(float(ys[0]) - float(y_top_hint)) + abs(float(ys[-1]) - float(y_bottom_hint)))
        score = 4.0 * float(len(ys)) + reg - hint_pen

        if score > best_score:
            best_score = score
            best_seq = ys

    if not best_seq:
        return vals, float(step)

    # Keep only a regular contiguous run to reject remote outliers.
    best_seq = sorted(best_seq)
    if len(best_seq) >= 4:
        keep = [best_seq[0]]
        for y in best_seq[1:]:
            d = float(y - keep[-1])
            if 0.55 * step <= d <= 1.85 * step:
                keep.append(float(y))
        if len(keep) >= 3:
            best_seq = keep

    return best_seq, float(step)


def y_error(y_top_pred: float, y_bottom_pred: float, y_top_gt: float, y_bottom_gt: float) -> float:
    return 0.5 * (abs(float(y_top_pred) - float(y_top_gt)) + abs(float(y_bottom_pred) - float(y_bottom_gt)))


def _is_scored_row(row: Dict[str, object]) -> bool:
    needed = {
        "x_err_before_px",
        "x_err_after_px",
        "y_err_before_px",
        "y_err_after_px",
        "score_before",
        "score_after",
    }
    return needed.issubset(set(row.keys()))


def refine_row(
    row: BBoxPredRow,
    gt: Optional[ManifestLine],
    line_half_px: float,
    pad_y_px: float,
    shift_y_px: float,
    line_offset_px: float,
    tick_search_radius_px: int,
    roi_pad_frac_x: float,
    roi_pad_frac_y: float,
    numbers_strip_px: int,
    ocr_min_conf: float,
    ocr_min_consistency: float,
    use_ocr: bool,
) -> Dict[str, object]:
    # Defaults.
    out: Dict[str, object] = {
        "sample_id": row.sample_id,
        "split": row.split,
        "image_path": row.image_path,
        "status": "ok",
        "anchor_source": "fallback_bbox",
        "tick_orientation": "",
        "tick_quality": float("nan"),
        "tick_runs": 0,
        "tick_coverage": float("nan"),
        "tick_periodicity": float("nan"),
        "tick_step_px": float("nan"),
        "ocr_count": 0,
        "ocr_direction": "none",
        "ocr_consistency": 0.0,
        "ocr_values": "",
        "ocr_y_image": "",
        "tick_seq_count": 0,
        "tick_seq_step_px": float("nan"),
        "zero_tick_image": float("nan"),
        "last_tick_image": float("nan"),
        "zero_tick_source": "none",
    }

    try:
        with Image.open(row.image_path) as im:
            gray_img = np.asarray(im.convert("L"), dtype=np.float32)
            w_img, h_img = im.size
    except Exception:
        out["status"] = "image_open_failed"
        return out

    if gt is None or gt.video_w <= 1.0 or gt.video_h <= 1.0:
        out["status"] = "missing_gt_manifest"
        return out

    sx_to_img = float(w_img) / float(gt.video_w)
    sy_to_img = float(h_img) / float(gt.video_h)
    sx_to_video = float(gt.video_w) / float(w_img)
    sy_to_video = float(gt.video_h) / float(h_img)

    x_gt_img = float(gt.x_gt_video) * sx_to_img
    y_top_gt_img = float(gt.y_top_gt_video) * sy_to_img
    y_bottom_gt_img = float(gt.y_bottom_gt_video) * sy_to_img

    side = int(gt.label_side)
    x_before, y_top_before, y_bottom_before = baseline_line_from_bbox(
        row=row,
        side=side,
        line_half_px=line_half_px,
        pad_y_px=pad_y_px,
        shift_y_px=shift_y_px,
    )
    x_before = float(np.clip(x_before, 0.0, max(0.0, w_img - 1.0)))
    y_top_before = float(np.clip(y_top_before, 0.0, max(0.0, h_img - 1.0)))
    y_bottom_before = float(np.clip(y_bottom_before, 0.0, max(0.0, h_img - 1.0)))
    if y_bottom_before < y_top_before:
        y_top_before, y_bottom_before = y_bottom_before, y_top_before

    bw = abs(float(row.x2_pred) - float(row.x1_pred))
    bh = abs(float(row.y2_pred) - float(row.y1_pred))
    pad_x = max(18.0, float(roi_pad_frac_x) * bw)
    pad_y = max(18.0, float(roi_pad_frac_y) * bh)
    rx0, ry0, rx1, ry1 = _clip_box(
        float(min(row.x1_pred, row.x2_pred)) - pad_x,
        float(min(row.y1_pred, row.y2_pred)) - pad_y,
        float(max(row.x1_pred, row.x2_pred)) + pad_x,
        float(max(row.y1_pred, row.y2_pred)) + pad_y,
        w=w_img,
        h=h_img,
    )
    roi = gray_img[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        out["status"] = "empty_roi"
        return out

    x_hint_local = float(x_before) - float(rx0)
    y_top_hint_local = float(y_top_before) - float(ry0)
    y_bottom_hint_local = float(y_bottom_before) - float(ry0)
    det = detect_tick_line_in_roi(
        gray_roi=roi,
        x_hint_local=x_hint_local,
        y_top_hint_local=y_top_hint_local,
        y_bottom_hint_local=y_bottom_hint_local,
        search_radius_x=int(tick_search_radius_px),
    )

    x_after_img = float(x_before)
    y_top_after_img = float(y_top_before)
    y_bottom_after_img = float(y_bottom_before)
    tick_rows_img: List[float] = []
    tick_rows_seq: List[float] = []

    if det is not None:
        x_tick_img = float(rx0) + float(det.x_local)
        if det.orientation == "ticks_left":
            x_after_img = x_tick_img + float(line_offset_px)
        else:
            x_after_img = x_tick_img - float(line_offset_px)
        x_after_img = float(np.clip(x_after_img, 0.0, max(0.0, w_img - 1.0)))

        tick_rows_img = [float(ry0) + float(y) for y in det.tick_rows_local]
        if tick_rows_img:
            # Build a regular sequence of equal ticks.
            tick_rows_seq, seq_step = select_uniform_tick_sequence(
                tick_rows=tick_rows_img,
                y_top_hint=float(y_top_before),
                y_bottom_hint=float(y_bottom_before),
                step_hint=float(det.tick_step),
            )
            out["tick_seq_count"] = int(len(tick_rows_seq))
            out["tick_seq_step_px"] = float(seq_step)

            # Base assumption: first visible regular tick is "0",
            # and last visible regular tick is the final endpoint.
            if tick_rows_seq:
                y0_seq = float(np.min(np.asarray(tick_rows_seq, dtype=np.float32)))
                y1_seq = float(np.max(np.asarray(tick_rows_seq, dtype=np.float32)))
                pred_span = max(8.0, float(y_bottom_before) - float(y_top_before))
                seq_span = max(1.0, y1_seq - y0_seq)
                span_ratio = float(seq_span / pred_span)
                center_shift = abs(0.5 * (y0_seq + y1_seq) - 0.5 * (y_top_before + y_bottom_before))
                plausible = (
                    len(tick_rows_seq) >= 5
                    and 0.45 <= span_ratio <= 1.70
                    and center_shift <= max(55.0, 0.36 * pred_span)
                )
                if plausible:
                    y_top_after_img = float(y0_seq)
                    y_bottom_after_img = float(y1_seq)
                    out["anchor_source"] = "ticks_uniform"
                    out["zero_tick_image"] = float(y_top_after_img)
                    out["last_tick_image"] = float(y_bottom_after_img)
                    out["zero_tick_source"] = "uniform_top"

        if str(out.get("anchor_source", "")) == "fallback_bbox":
            out["anchor_source"] = "ticks_only"
        out["tick_orientation"] = det.orientation
        out["tick_quality"] = float(det.quality)
        out["tick_runs"] = int(det.runs)
        out["tick_coverage"] = float(det.coverage)
        out["tick_periodicity"] = float(det.periodicity)
        out["tick_step_px"] = float(det.tick_step)

        if use_ocr and tick_rows_img:
            top_tick = float(np.min(tick_rows_img))
            bot_tick = float(np.max(tick_rows_img))
            if det.orientation == "ticks_left":
                sx0 = max(0, int(round(x_tick_img - float(numbers_strip_px))))
                sx1 = max(sx0 + 6, int(round(x_tick_img - 2.0)))
            else:
                sx0 = min(w_img - 6, int(round(x_tick_img + 2.0)))
                sx1 = min(w_img, int(round(x_tick_img + float(numbers_strip_px))))
            sy0 = max(0, int(round(top_tick - 20.0)))
            sy1 = min(h_img, int(round(bot_tick + 20.0)))
            if sx1 > sx0 and sy1 > sy0:
                strip = gray_img[sy0:sy1, sx0:sx1]
                tokens = detect_numeric_tokens(strip, min_conf=float(ocr_min_conf))
                if tokens:
                    tokens_img = [
                        OCRToken(
                            value=t.value,
                            conf=t.conf,
                            x_center=float(sx0) + t.x_center,
                            y_center=float(sy0) + t.y_center,
                            x1=float(sx0) + t.x1,
                            y1=float(sy0) + t.y1,
                            x2=float(sx0) + t.x2,
                            y2=float(sy0) + t.y2,
                            text_raw=t.text_raw,
                        )
                        for t in tokens
                    ]
                else:
                    tokens_img = []

                direction, consistency = monotonic_direction(tokens_img)
                out["ocr_count"] = int(len(tokens_img))
                out["ocr_direction"] = direction
                out["ocr_consistency"] = float(consistency)
                out["ocr_values"] = "|".join(str(t.value) for t in tokens_img[:18])
                out["ocr_y_image"] = "|".join(f"{t.y_center:.1f}" for t in tokens_img[:18])

                if (
                    len(tokens_img) >= 3
                    and consistency >= float(ocr_min_consistency)
                    and len(tick_rows_seq) >= 4
                ):
                    tick_step = float(out.get("tick_seq_step_px", float("nan")))
                    if not math.isfinite(tick_step) or tick_step < 2.0:
                        tick_step = float(det.tick_step) if math.isfinite(float(det.tick_step)) else float("nan")
                    tol = max(8.0, 0.55 * tick_step) if math.isfinite(tick_step) else 14.0

                    # Prefer explicit OCR "0" to identify start tick.
                    zero_tokens = [t for t in tokens_img if int(t.value) == 0]
                    if zero_tokens:
                        y_zero_num = float(np.median(np.asarray([t.y_center for t in zero_tokens], dtype=np.float32)))
                        y_zero_snap = nearest_tick_row(y_zero_num, tick_rows=tick_rows_seq, max_dist=tol)
                    else:
                        y_zero_snap = None

                    # Last tick comes from bottom of the regular sequence.
                    if tick_rows_seq:
                        y_last_seq = float(np.max(np.asarray(tick_rows_seq, dtype=np.float32)))
                    else:
                        y_last_seq = float("nan")

                    if y_zero_snap is not None and math.isfinite(y_last_seq) and y_last_seq > y_zero_snap + 1.0:
                        y_top_after_img = float(y_zero_snap)
                        y_bottom_after_img = float(y_last_seq)
                        out["anchor_source"] = "ticks+ocr_zero"
                        out["zero_tick_image"] = float(y_top_after_img)
                        out["last_tick_image"] = float(y_bottom_after_img)
                        out["zero_tick_source"] = "ocr_zero"
                    elif tick_rows_seq and out.get("anchor_source") == "ticks_uniform":
                        # Keep uniform tick endpoints when OCR is monotonic but no explicit zero.
                        out["zero_tick_source"] = "uniform_top_no_zero_ocr"

    y_top_after_img = float(np.clip(y_top_after_img, 0.0, max(0.0, h_img - 1.0)))
    y_bottom_after_img = float(np.clip(y_bottom_after_img, 0.0, max(0.0, h_img - 1.0)))
    if y_bottom_after_img < y_top_after_img:
        y_top_after_img, y_bottom_after_img = y_bottom_after_img, y_top_after_img

    x_err_before = float(x_before - x_gt_img)
    x_err_after = float(x_after_img - x_gt_img)
    y_err_before = y_error(y_top_before, y_bottom_before, y_top_gt_img, y_bottom_gt_img)
    y_err_after = y_error(y_top_after_img, y_bottom_after_img, y_top_gt_img, y_bottom_gt_img)

    out.update(
        {
            "x_gt_video": float(gt.x_gt_video),
            "y_top_gt_video": float(gt.y_top_gt_video),
            "y_bottom_gt_video": float(gt.y_bottom_gt_video),
            "x_gt_image": x_gt_img,
            "y_top_gt_image": y_top_gt_img,
            "y_bottom_gt_image": y_bottom_gt_img,
            "x_before_image": x_before,
            "y_top_before_image": y_top_before,
            "y_bottom_before_image": y_bottom_before,
            "x_after_image": x_after_img,
            "y_top_after_image": y_top_after_img,
            "y_bottom_after_image": y_bottom_after_img,
            "x_before_video": float(x_before * sx_to_video),
            "y_top_before_video": float(y_top_before * sy_to_video),
            "y_bottom_before_video": float(y_bottom_before * sy_to_video),
            "x_after_video": float(x_after_img * sx_to_video),
            "y_top_after_video": float(y_top_after_img * sy_to_video),
            "y_bottom_after_video": float(y_bottom_after_img * sy_to_video),
            "x_err_before_px": x_err_before,
            "x_err_after_px": x_err_after,
            "y_err_before_px": y_err_before,
            "y_err_after_px": y_err_after,
            "score_before": float(abs(x_err_before) + y_err_before),
            "score_after": float(abs(x_err_after) + y_err_after),
            "bbox_x1_pred": float(row.x1_pred),
            "bbox_y1_pred": float(row.y1_pred),
            "bbox_x2_pred": float(row.x2_pred),
            "bbox_y2_pred": float(row.y2_pred),
            "bbox_iou": float(row.iou),
            "roi_x0": int(rx0),
            "roi_y0": int(ry0),
            "roi_x1": int(rx1),
            "roi_y1": int(ry1),
            "tick_rows_image": "|".join(f"{y:.1f}" for y in tick_rows_img[:80]),
            "tick_seq_rows_image": "|".join(f"{y:.1f}" for y in tick_rows_seq[:80]),
        }
    )
    return out


def compute_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    rows = [r for r in rows if _is_scored_row(r)]
    if not rows:
        return {
            "rows": 0.0,
            "mae_x_before_px": float("nan"),
            "mae_x_after_px": float("nan"),
            "mae_y_before_px": float("nan"),
            "mae_y_after_px": float("nan"),
            "score_before": float("nan"),
            "score_after": float("nan"),
            "delta_score": float("nan"),
            "improved_rows_ratio": float("nan"),
        }
    x_before = np.asarray([abs(float(r["x_err_before_px"])) for r in rows], dtype=np.float64)
    x_after = np.asarray([abs(float(r["x_err_after_px"])) for r in rows], dtype=np.float64)
    y_before = np.asarray([float(r["y_err_before_px"]) for r in rows], dtype=np.float64)
    y_after = np.asarray([float(r["y_err_after_px"]) for r in rows], dtype=np.float64)
    s_before = x_before + y_before
    s_after = x_after + y_after
    improved = np.mean((s_after < s_before).astype(np.float64))
    return {
        "rows": float(len(rows)),
        "mae_x_before_px": float(np.mean(x_before)),
        "mae_x_after_px": float(np.mean(x_after)),
        "mae_y_before_px": float(np.mean(y_before)),
        "mae_y_after_px": float(np.mean(y_after)),
        "score_before": float(np.mean(s_before)),
        "score_after": float(np.mean(s_after)),
        "delta_score": float(np.mean(s_after) - np.mean(s_before)),
        "improved_rows_ratio": float(improved),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def draw_preview(row: Dict[str, object], out_path: Path) -> None:
    img = Image.open(str(row["image_path"])).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Pred bbox.
    draw.rectangle(
        (
            float(row["bbox_x1_pred"]),
            float(row["bbox_y1_pred"]),
            float(row["bbox_x2_pred"]),
            float(row["bbox_y2_pred"]),
        ),
        outline=(255, 0, 220, 255),
        width=4,
        fill=(255, 0, 220, 22),
    )

    # GT line (green), baseline line (orange), refined line (cyan).
    draw.line(
        (
            float(row["x_gt_image"]),
            float(row["y_top_gt_image"]),
            float(row["x_gt_image"]),
            float(row["y_bottom_gt_image"]),
        ),
        fill=(34, 197, 94, 255),
        width=5,
    )
    draw.line(
        (
            float(row["x_before_image"]),
            float(row["y_top_before_image"]),
            float(row["x_before_image"]),
            float(row["y_bottom_before_image"]),
        ),
        fill=(245, 158, 11, 220),
        width=3,
    )
    draw.line(
        (
            float(row["x_after_image"]),
            float(row["y_top_after_image"]),
            float(row["x_after_image"]),
            float(row["y_bottom_after_image"]),
        ),
        fill=(0, 230, 255, 255),
        width=3,
    )

    # Tick markers.
    x_a = float(row["x_after_image"])
    ticks_raw = str(row.get("tick_seq_rows_image", "")).strip()
    if not ticks_raw:
        ticks_raw = str(row.get("tick_rows_image", "")).strip()
    if ticks_raw:
        for tok in ticks_raw.split("|")[:120]:
            try:
                y = float(tok)
            except Exception:
                continue
            draw.line((x_a - 7.0, y, x_a + 7.0, y), fill=(255, 230, 0, 220), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def build_review_html(
    out_dir: Path,
    split_name: str,
    rows: Sequence[Dict[str, object]],
    max_rows: int,
) -> Optional[Path]:
    rows = [r for r in rows if _is_scored_row(r)]
    if not rows:
        return None
    scored = sorted(
        rows,
        key=lambda r: float(abs(float(r["x_err_after_px"]))) + float(r["y_err_after_px"]),
        reverse=True,
    )[: max(0, int(max_rows))]
    preview_dir = out_dir / f"review_{split_name}_post_previews"
    html_path = out_dir / f"review_{split_name}_post.html"
    trs: List[str] = []
    for i, r in enumerate(scored, start=1):
        sample_id = str(r.get("sample_id", f"row_{i}"))
        out_img = preview_dir / f"{i:04d}_{_safe_slug(sample_id)}.jpg"
        try:
            draw_preview(r, out_img)
            metrics = {
                "sample_id": sample_id,
                "score_before": f"{float(r['score_before']):.2f}",
                "score_after": f"{float(r['score_after']):.2f}",
                "x_before": f"{abs(float(r['x_err_before_px'])):.2f}",
                "x_after": f"{abs(float(r['x_err_after_px'])):.2f}",
                "y_before": f"{float(r['y_err_before_px']):.2f}",
                "y_after": f"{float(r['y_err_after_px']):.2f}",
                "source": str(r.get("anchor_source", "")),
                "tick_orient": str(r.get("tick_orientation", "")),
                "tick_q": f"{float(r.get('tick_quality', float('nan'))):.3f}",
                "tick_n": f"{int(r.get('tick_seq_count', 0))}",
                "ocr_n": f"{int(r.get('ocr_count', 0))}",
                "ocr_dir": str(r.get("ocr_direction", "")),
                "ocr_cons": f"{float(r.get('ocr_consistency', 0.0)):.2f}",
                "zero_src": str(r.get("zero_tick_source", "")),
            }
            metrics_json = escape(json.dumps(metrics, ensure_ascii=False), quote=True)
            img_tag = (
                f'<button type="button" class="preview-btn" data-index="{i - 1}" '
                f'data-src="{escape(f"{preview_dir.name}/{out_img.name}", quote=True)}" '
                f'data-metrics="{metrics_json}">'
                f'<img src="{preview_dir.name}/{out_img.name}" loading="lazy" />'
                "</button>"
            )
        except Exception:
            img_tag = "preview_error"

        trs.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{escape(sample_id)}</td>"
            f"<td>{float(r['score_before']):.2f}</td>"
            f"<td>{float(r['score_after']):.2f}</td>"
            f"<td>{abs(float(r['x_err_before_px'])):.2f}</td>"
            f"<td>{abs(float(r['x_err_after_px'])):.2f}</td>"
            f"<td>{float(r['y_err_before_px']):.2f}</td>"
            f"<td>{float(r['y_err_after_px']):.2f}</td>"
            f"<td>{str(r.get('anchor_source',''))}</td>"
            f"<td>{str(r.get('tick_orientation',''))}</td>"
            f"<td>{float(r.get('tick_quality', float('nan'))):.3f}</td>"
            f"<td>{int(r.get('tick_seq_count', 0))}</td>"
            f"<td>{int(r.get('ocr_count', 0))}</td>"
            f"<td>{str(r.get('ocr_direction',''))}</td>"
            f"<td>{float(r.get('ocr_consistency', 0.0)):.2f}</td>"
            f"<td>{str(r.get('zero_tick_source',''))}</td>"
            "</tr>"
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BBox -> Line Postprocess Review - {split_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 620px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
    .preview-btn {{ border: 0; background: transparent; padding: 0; cursor: zoom-in; }}
    .preview-btn:focus-visible {{ outline: 2px solid #0ea5e9; outline-offset: 2px; border-radius: 8px; }}
    .modal {{ position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(2, 6, 23, 0.75); z-index: 9999; padding: 18px; }}
    .modal.open {{ display: flex; }}
    .modal-panel {{ background: #ffffff; width: min(96vw, 1260px); max-height: 92vh; border-radius: 12px; box-shadow: 0 16px 48px rgba(2, 6, 23, 0.35); display: grid; grid-template-columns: minmax(360px, 1fr) 320px; gap: 14px; padding: 14px; }}
    .modal-image-wrap {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; min-height: 420px; display: flex; align-items: center; justify-content: center; overflow: auto; }}
    .modal-image-wrap img {{ width: auto; max-width: 100%; max-height: calc(92vh - 120px); border-radius: 8px; border: 1px solid #cbd5e1; }}
    .modal-side {{ overflow: auto; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; }}
    .modal-title {{ margin: 0 0 8px; font-size: 14px; }}
    .metric-row {{ display: flex; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 12px; }}
    .metric-row span {{ color: #475569; }}
    .metric-row b {{ color: #0f172a; font-weight: 600; }}
    .metric-row:last-child {{ border-bottom: 0; }}
    .modal-toolbar {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .modal-btn {{ border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; padding: 6px 10px; border-radius: 8px; cursor: pointer; font-size: 12px; }}
    .modal-btn:hover {{ background: #f8fafc; }}
    .modal-index {{ font-size: 12px; color: #334155; }}
    body.modal-open {{ overflow: hidden; }}
    @media (max-width: 1024px) {{
      .modal-panel {{ grid-template-columns: 1fr; width: 96vw; }}
      .modal-image-wrap {{ min-height: 320px; }}
    }}
  </style>
</head>
<body>
  <h1>BBox -> Line Postprocess Review - {split_name}</h1>
  <p>Overlay: <b>green=GT line</b>, <b>orange=baseline from bbox inversion</b>, <b>cyan=postprocessed line</b>, <b>magenta=pred bbox</b>, <b>yellow=tick rows</b></p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Preview</th><th>sample_id</th><th>score_before</th><th>score_after</th>
        <th>x_before</th><th>x_after</th><th>y_before</th><th>y_after</th>
        <th>source</th><th>tick_orient</th><th>tick_q</th><th>tick_n</th><th>ocr_n</th><th>ocr_dir</th><th>ocr_cons</th><th>zero_src</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
  <div id="previewModal" class="modal" aria-hidden="true">
    <div class="modal-panel" role="dialog" aria-modal="true" aria-label="Preview immagine grande">
      <div>
        <div class="modal-toolbar">
          <div>
            <button id="btnPrev" type="button" class="modal-btn" title="Precedente (freccia sinistra)">← Precedente</button>
            <button id="btnNext" type="button" class="modal-btn" title="Successiva (freccia destra)">Successiva →</button>
          </div>
          <div id="modalIndex" class="modal-index"></div>
        </div>
        <div class="modal-image-wrap">
          <img id="modalImage" src="" alt="Preview grande" />
        </div>
      </div>
      <div class="modal-side">
        <div class="modal-toolbar">
          <h3 class="modal-title">Metriche</h3>
          <button id="btnClose" type="button" class="modal-btn" title="Chiudi (Esc)">Chiudi</button>
        </div>
        <div id="modalMetrics"></div>
      </div>
    </div>
  </div>
  <script>
    (function() {{
      const items = Array.from(document.querySelectorAll('.preview-btn'));
      const modal = document.getElementById('previewModal');
      const modalImage = document.getElementById('modalImage');
      const modalMetrics = document.getElementById('modalMetrics');
      const modalIndex = document.getElementById('modalIndex');
      const btnPrev = document.getElementById('btnPrev');
      const btnNext = document.getElementById('btnNext');
      const btnClose = document.getElementById('btnClose');
      let current = -1;

      const metricLabels = [
        ['sample_id', 'sample_id'],
        ['score_before', 'score_before'],
        ['score_after', 'score_after'],
        ['x_before', 'x_before'],
        ['x_after', 'x_after'],
        ['y_before', 'y_before'],
        ['y_after', 'y_after'],
        ['source', 'source'],
        ['tick_orient', 'tick_orient'],
        ['tick_q', 'tick_q'],
        ['tick_n', 'tick_n'],
        ['ocr_n', 'ocr_n'],
        ['ocr_dir', 'ocr_dir'],
        ['ocr_cons', 'ocr_cons'],
        ['zero_src', 'zero_src']
      ];

      function render(index) {{
        if (index < 0 || index >= items.length) return;
        current = index;
        const el = items[index];
        modalImage.src = el.dataset.src || '';
        modalImage.alt = 'Preview ' + (index + 1);
        let metrics = {{}};
        try {{
          metrics = JSON.parse(el.dataset.metrics || '{{}}');
        }} catch (_err) {{
          metrics = {{}};
        }}
        const rows = metricLabels.map(([key, label]) => {{
          const v = (metrics[key] ?? '').toString();
          return `<div class="metric-row"><span>${{label}}</span><b>${{v}}</b></div>`;
        }}).join('');
        modalMetrics.innerHTML = rows;
        modalIndex.textContent = `${{index + 1}} / ${{items.length}}`;
      }}

      function openModal(index) {{
        render(index);
        modal.classList.add('open');
        modal.setAttribute('aria-hidden', 'false');
        document.body.classList.add('modal-open');
      }}

      function closeModal() {{
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('modal-open');
      }}

      function move(delta) {{
        if (!items.length) return;
        let next = current + delta;
        if (next < 0) next = items.length - 1;
        if (next >= items.length) next = 0;
        render(next);
      }}

      items.forEach((el, idx) => {{
        el.addEventListener('click', () => openModal(idx));
      }});
      btnPrev.addEventListener('click', () => move(-1));
      btnNext.addEventListener('click', () => move(1));
      btnClose.addEventListener('click', closeModal);

      modal.addEventListener('click', (ev) => {{
        if (ev.target === modal) closeModal();
      }});

      document.addEventListener('keydown', (ev) => {{
        if (!modal.classList.contains('open')) return;
        if (ev.key === 'ArrowLeft') {{
          ev.preventDefault();
          move(-1);
        }} else if (ev.key === 'ArrowRight') {{
          ev.preventDefault();
          move(1);
        }} else if (ev.key === 'Escape') {{
          ev.preventDefault();
          closeModal();
        }}
      }});
    }})();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine scale line from bbox predictions with ticks + OCR.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes"),
    )
    p.add_argument("--splits", type=str, default="val,test", help="Comma list among train,val,test")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--line-offset-px", type=float, default=1.0)
    p.add_argument("--tick-search-radius-px", type=int, default=170)
    p.add_argument("--roi-pad-frac-x", type=float, default=0.18)
    p.add_argument("--roi-pad-frac-y", type=float, default=0.12)
    p.add_argument("--numbers-strip-px", type=int, default=120)
    p.add_argument("--ocr-min-confidence", type=float, default=24.0)
    p.add_argument("--ocr-min-consistency", type=float, default=0.58)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--max-rows-per-split", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-review-rows", type=int, default=180)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root locale dataset (es. data/Dataset) per remap dei path /Volumes/SSD_esi1_n1.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest_path = Path(str(summary["manifest"])).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest_map = load_manifest_lines(manifest_path)

    bbox_params = dict(summary.get("bbox_params") or {})
    line_half_px = float(bbox_params.get("line_half_px", 10.0))
    pad_y_px = float(bbox_params.get("pad_y_px", 16.0))
    shift_y_px = float(bbox_params.get("shift_y_px", 14.0))

    split_tokens = [s.strip().lower() for s in str(args.splits).split(",") if s.strip()]
    splits = [s for s in split_tokens if s in {"train", "val", "test"}]
    if not splits:
        splits = ["val", "test"]

    repo_root = Path(__file__).resolve().parents[2]
    if args.dataset_root is not None:
        dataset_roots = [args.dataset_root.expanduser().resolve()]
    else:
        dataset_roots = [
            (repo_root / "data" / "Dataset").resolve(),
            (repo_root / "data" / "dataset").resolve(),
        ]
    path_resolver = build_image_path_resolver(dataset_roots=dataset_roots)

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else (run_dir / "refined_ticks_ocr")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "run_dir": run_dir.as_posix(),
        "manifest": manifest_path.as_posix(),
        "splits": {},
        "params": {
            "line_half_px": line_half_px,
            "pad_y_px": pad_y_px,
            "shift_y_px": shift_y_px,
            "line_offset_px": float(args.line_offset_px),
            "tick_search_radius_px": int(args.tick_search_radius_px),
            "roi_pad_frac_x": float(args.roi_pad_frac_x),
            "roi_pad_frac_y": float(args.roi_pad_frac_y),
            "numbers_strip_px": int(args.numbers_strip_px),
            "ocr_min_confidence": float(args.ocr_min_confidence),
            "ocr_min_consistency": float(args.ocr_min_consistency),
            "use_ocr": not bool(args.no_ocr),
            "dataset_roots": [p.as_posix() for p in dataset_roots],
        },
        "artifacts": {},
    }

    for split in splits:
        pred_csv = run_dir / f"{split}_predictions_best.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(f"Missing predictions CSV: {pred_csv}")

        rows = load_bbox_rows(pred_csv=pred_csv, split_name=split)
        rows = [replace(r, image_path=path_resolver(r.image_path)) for r in rows]
        if args.max_rows_per_split > 0:
            rows = rows[: int(args.max_rows_per_split)]
        if rows:
            probe_rows = rows[: min(40, len(rows))]
            reachable = sum(1 for r in probe_rows if Path(str(r.image_path)).exists())
            if reachable == 0:
                raise RuntimeError(
                    "Nessuna immagine raggiungibile dai path del CSV dopo il remap. "
                    "Verifica --dataset-root (es. data/Dataset) o il mount /Volumes/SSD_esi1_n1."
                )

        refined_rows: List[Dict[str, object]] = []
        for i, row in enumerate(rows, start=1):
            gt = manifest_map.get(row.sample_id)
            refined = refine_row(
                row=row,
                gt=gt,
                line_half_px=line_half_px,
                pad_y_px=pad_y_px,
                shift_y_px=shift_y_px,
                line_offset_px=float(args.line_offset_px),
                tick_search_radius_px=int(args.tick_search_radius_px),
                roi_pad_frac_x=float(args.roi_pad_frac_x),
                roi_pad_frac_y=float(args.roi_pad_frac_y),
                numbers_strip_px=int(args.numbers_strip_px),
                ocr_min_conf=float(args.ocr_min_confidence),
                ocr_min_consistency=float(args.ocr_min_consistency),
                use_ocr=not bool(args.no_ocr),
            )
            refined_rows.append(refined)
            if i % 40 == 0:
                print(f"[{split}] processed {i}/{len(rows)}", flush=True)

        csv_out = out_dir / f"{split}_line_postprocessed.csv"
        write_csv(csv_out, refined_rows)
        html_out = build_review_html(
            out_dir=out_dir,
            split_name=split,
            rows=refined_rows,
            max_rows=int(args.max_review_rows),
        )
        metrics = compute_metrics(refined_rows)
        src_counts = Counter(str(r.get("anchor_source", "")) for r in refined_rows)

        report["splits"][split] = {
            "rows": len(refined_rows),
            "metrics": metrics,
            "anchor_source_counts": dict(src_counts),
            "ocr_applied_rows": int(sum(1 for r in refined_rows if int(r.get("ocr_count", 0)) > 0)),
        }
        report["artifacts"][f"{split}_csv"] = csv_out.as_posix()
        if html_out is not None:
            report["artifacts"][f"{split}_review_html"] = html_out.as_posix()

    summary_out = out_dir / "refine_summary.json"
    summary_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
