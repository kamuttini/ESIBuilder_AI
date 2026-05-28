#!/usr/bin/env python3
"""Train an image model to predict scale bbox (axis + ticks + numbers) on full-frame ultrasound images.

Ground truth bbox is generated from SCALE_LINE row (line21) using configurable context parameters:
- numbers-span
- tick-span
- line half-width
- vertical padding
- optional left trim
- optional vertical shift
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    fss_path: str
    setup_id: str
    depth_index: int
    image_path: str
    image_w: int
    image_h: int
    video_w: int
    video_h: int
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float
    label_side: int


@dataclass(frozen=True)
class SegmentAnchor:
    x: float
    y_top: float
    y_bottom: float
    score: float
    upper_len: int
    lower_len: int


def _f(text: str) -> float:
    return float(str(text).strip())


def _i(text: str) -> int:
    return int(float(str(text).strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120]


def _config_folder_from_fss_path(fss_path: str) -> str:
    p = Path(fss_path)
    if p.parent.name == "DB_setup":
        return p.parent.parent.name
    return p.parent.name


def _run_ranges(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    cur_start: Optional[int] = None
    n = int(mask.shape[0])
    for i in range(n):
        if bool(mask[i]):
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                s = int(cur_start)
                e = int(i - 1)
                out.append((s, e, e - s + 1))
                cur_start = None
    if cur_start is not None:
        s = int(cur_start)
        e = int(n - 1)
        out.append((s, e, e - s + 1))
    return out


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
    return mask


def detect_segment_anchor(
    gray: np.ndarray,
    x_hint: float,
    search_radius_px: int,
    min_len_frac: float,
) -> Optional[SegmentAnchor]:
    H, W = gray.shape
    if H < 40 or W < 40:
        return None

    mask = _build_adaptive_gray_mask(gray)
    min_len = max(20, int(round(float(min_len_frac) * float(H))))

    x0 = max(2, int(round(x_hint)) - int(search_radius_px))
    x1 = min(W - 3, int(round(x_hint)) + int(search_radius_px))
    if x1 <= x0:
        return None

    best: Optional[SegmentAnchor] = None
    best_score = -1e9

    for x in range(x0, x1 + 1):
        xl = max(0, x - 2)
        xr = min(W, x + 3)
        col_block = mask[:, xl:xr]
        if col_block.size == 0:
            continue

        row_hits = col_block.any(axis=1)
        if row_hits.size >= 9:
            row_hits = np.convolve(row_hits.astype(np.int8), np.ones(5, dtype=np.int8), mode="same") > 0
        runs = [r for r in _run_ranges(row_hits) if r[2] >= min_len]
        if not runs:
            continue

        centers = [0.5 * (s + e) for s, e, _ in runs]
        upper_candidates = [runs[i] for i, c in enumerate(centers) if c <= 0.55 * H]
        lower_candidates = [runs[i] for i, c in enumerate(centers) if c >= 0.45 * H]
        upper = max(upper_candidates, key=lambda t: t[2]) if upper_candidates else max(runs, key=lambda t: t[2])
        lower = max(lower_candidates, key=lambda t: t[2]) if lower_candidates else max(runs, key=lambda t: t[2])

        y_top = float(min(upper[0], lower[0]))
        y_bottom = float(max(upper[1], lower[1]))
        span = y_bottom - y_top + 1.0
        if span < max(44.0, 0.15 * H):
            continue

        score = float(span + 0.75 * upper[2] + 0.75 * lower[2] - 0.08 * abs(float(x) - float(x_hint)))
        if score > best_score:
            best_score = score
            best = SegmentAnchor(
                x=float(x),
                y_top=y_top,
                y_bottom=y_bottom,
                score=score,
                upper_len=int(upper[2]),
                lower_len=int(lower[2]),
            )

    return best


def _row_filter_text(sample_id: str, fss_path: str, setup_id: str) -> str:
    return f"{sample_id} {fss_path} {setup_id}".lower()


def row_matches_exclusion(sample_id: str, fss_path: str, setup_id: str, label: str) -> bool:
    text = _row_filter_text(sample_id=sample_id, fss_path=fss_path, setup_id=setup_id)
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
        f"image_depth_value_setup_{sid}_{idx0}",
        f"image_depth_value_setup_{sid}_{depth_index}",
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


def context_bbox(
    x_line: float,
    y_top: float,
    y_bottom: float,
    side: int,
    w: int,
    h: int,
    numbers_span_px: float,
    tick_span_px: float,
    line_half_px: float,
    pad_y_px: float,
    trim_left_px: float,
    shift_y_px: float,
) -> Tuple[float, float, float, float]:
    y1 = (y_top - pad_y_px) + float(shift_y_px)
    y2 = (y_bottom + pad_y_px) + float(shift_y_px)
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))

    if side < 0:
        x1 = x_line - (numbers_span_px + tick_span_px + line_half_px)
        x2 = x_line + (line_half_px + 8.0)
    else:
        x1 = x_line - (line_half_px + 8.0)
        x2 = x_line + (numbers_span_px + tick_span_px + line_half_px)
    if trim_left_px > 0:
        x1 += float(trim_left_px)

    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    if x2 <= x1 + 2.0:
        x2 = min(float(w - 1), x1 + 3.0)
    if y2 <= y1 + 2.0:
        y2 = min(float(h - 1), y1 + 3.0)
    return x1, y1, x2, y2


def load_samples(
    manifest: Path,
    limit: int,
    progress_every: int,
    exclude_labels: Optional[Sequence[str]],
    numbers_span_px: float,
    tick_span_px: float,
    line_half_px: float,
    pad_y_px: float,
    trim_left_px: float,
    shift_y_px: float,
    folder_canonical_line: bool,
    use_segment_anchor: bool,
    segment_search_radius_px: int,
    segment_line_offset_px: float,
    segment_blend_x: float,
    segment_blend_y: float,
    segment_min_len_frac: float,
    segment_max_shift_x_px: float,
    segment_max_shift_y_px: float,
) -> Tuple[List[Sample], Dict[str, int], int, Dict[str, float]]:
    out: List[Sample] = []
    excluded_by_filter: Counter[str] = Counter()
    excluded_total = 0
    active_labels = list(exclude_labels or [])
    raw_rows: List[Dict[str, object]] = []

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "split",
            "fss_path",
            "setup_id",
            "depth_index",
            "x1",
            "x2",
            "y1",
            "y2",
            "video_x_size",
            "video_y_size",
            "label_side",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise RuntimeError(f"Manifest missing columns: {missing}")

        for row in reader:
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            sample_id = row["sample_id"]
            fss_path = row["fss_path"]
            setup_id = row.get("setup_id", "")
            matched = [label for label in active_labels if row_matches_exclusion(sample_id, fss_path, setup_id, label)]
            if matched:
                excluded_total += 1
                for label in matched:
                    excluded_by_filter[label] += 1
                continue

            try:
                depth_index = _i(row["depth_index"])
                vw = _i(row["video_x_size"])
                vh = _i(row["video_y_size"])
                if vw <= 0 or vh <= 0:
                    continue
                x1 = _f(row["x1"])
                x2 = _f(row["x2"])
                y1 = _f(row["y1"])
                y2 = _f(row["y2"])
                label_side = _i(row["label_side"])
            except Exception:
                continue

            raw_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "fss_path": fss_path,
                    "setup_id": setup_id,
                    "depth_index": depth_index,
                    "video_w": vw,
                    "video_h": vh,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "label_side": label_side,
                    "config_folder": _config_folder_from_fss_path(fss_path),
                }
            )

    canonical_by_folder: Dict[str, Tuple[float, float, float, int]] = {}
    if folder_canonical_line and raw_rows:
        grouped: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
        for r in raw_rows:
            vw = float(r["video_w"])
            vh = float(r["video_h"])
            x_line_norm = 0.5 * (float(r["x1"]) + float(r["x2"])) / max(1.0, vw)
            y_top_norm = min(float(r["y1"]), float(r["y2"])) / max(1.0, vh)
            y_bottom_norm = max(float(r["y1"]), float(r["y2"])) / max(1.0, vh)
            side = -1 if int(r["label_side"]) < 0 else 1
            grouped[str(r["config_folder"])].append((x_line_norm, y_top_norm, y_bottom_norm, side))
        for folder, vals in grouped.items():
            x_med = float(np.median([v[0] for v in vals]))
            y_top_med = float(np.median([v[1] for v in vals]))
            y_bottom_med = float(np.median([v[2] for v in vals]))
            sides = [v[3] for v in vals]
            side_major = -1 if sum(1 for s in sides if s < 0) >= (len(sides) / 2.0) else 1
            canonical_by_folder[folder] = (x_med, y_top_med, y_bottom_med, side_major)

    label_stats: Dict[str, float] = {
        "rows_manifest_kept_after_filter": float(len(raw_rows)),
        "folders_with_canonical_line": float(len(canonical_by_folder)),
        "rows_with_folder_canonical_line": 0.0,
        "rows_with_segment_anchor": 0.0,
        "rows_segment_anchor_not_found": 0.0,
    }

    for i, r in enumerate(raw_rows, start=1):
        depth_index = int(r["depth_index"])
        image_path = find_full_frame_image(str(r["fss_path"]), str(r["setup_id"]), depth_index)
        if image_path is None:
            continue

        try:
            with Image.open(image_path) as im:
                iw, ih = im.size
                gray = np.asarray(im.convert("L"), dtype=np.float32) if use_segment_anchor else None
        except Exception:
            continue
        if iw <= 2 or ih <= 2:
            continue

        if folder_canonical_line and str(r["config_folder"]) in canonical_by_folder:
            x_norm, y_top_norm, y_bottom_norm, side = canonical_by_folder[str(r["config_folder"])]
            x_line = x_norm * float(iw)
            y_top = y_top_norm * float(ih)
            y_bottom = y_bottom_norm * float(ih)
            label_side = side
            label_stats["rows_with_folder_canonical_line"] += 1.0
        else:
            sx = float(iw) / float(r["video_w"])
            sy = float(ih) / float(r["video_h"])
            x_line = 0.5 * (float(r["x1"]) + float(r["x2"])) * sx
            y_top = min(float(r["y1"]), float(r["y2"])) * sy
            y_bottom = max(float(r["y1"]), float(r["y2"])) * sy
            label_side = -1 if int(r["label_side"]) < 0 else 1

        if use_segment_anchor and gray is not None:
            anchor = detect_segment_anchor(
                gray=gray,
                x_hint=float(x_line),
                search_radius_px=int(segment_search_radius_px),
                min_len_frac=float(segment_min_len_frac),
            )
            if anchor is not None:
                x_target = float(anchor.x) + float(segment_line_offset_px)
                x_target = float(x_line) + float(
                    np.clip(
                        x_target - float(x_line),
                        -float(segment_max_shift_x_px),
                        float(segment_max_shift_x_px),
                    )
                )
                y_top_target = float(y_top) + float(
                    np.clip(
                        float(anchor.y_top) - float(y_top),
                        -float(segment_max_shift_y_px),
                        float(segment_max_shift_y_px),
                    )
                )
                y_bottom_target = float(y_bottom) + float(
                    np.clip(
                        float(anchor.y_bottom) - float(y_bottom),
                        -float(segment_max_shift_y_px),
                        float(segment_max_shift_y_px),
                    )
                )
                bx = float(np.clip(segment_blend_x, 0.0, 1.0))
                by = float(np.clip(segment_blend_y, 0.0, 1.0))
                x_line = (1.0 - bx) * float(x_line) + bx * x_target
                y_top = (1.0 - by) * float(y_top) + by * y_top_target
                y_bottom = (1.0 - by) * float(y_bottom) + by * y_bottom_target
                if y_bottom < y_top:
                    y_top, y_bottom = y_bottom, y_top
                label_stats["rows_with_segment_anchor"] += 1.0
            else:
                label_stats["rows_segment_anchor_not_found"] += 1.0

        side = -1 if int(label_side) < 0 else 1
        bx1, by1, bx2, by2 = context_bbox(
            x_line=float(x_line),
            y_top=float(y_top),
            y_bottom=float(y_bottom),
            side=side,
            w=iw,
            h=ih,
            numbers_span_px=numbers_span_px,
            tick_span_px=tick_span_px,
            line_half_px=line_half_px,
            pad_y_px=pad_y_px,
            trim_left_px=trim_left_px,
            shift_y_px=shift_y_px,
        )

        out.append(
            Sample(
                sample_id=str(r["sample_id"]),
                split=str(r["split"]),
                fss_path=str(r["fss_path"]),
                setup_id=str(r["setup_id"]),
                depth_index=depth_index,
                image_path=image_path.as_posix(),
                image_w=iw,
                image_h=ih,
                video_w=int(r["video_w"]),
                video_h=int(r["video_h"]),
                x1_norm=bx1 / float(iw),
                y1_norm=by1 / float(ih),
                x2_norm=bx2 / float(iw),
                y2_norm=by2 / float(ih),
                label_side=int(label_side),
            )
        )
        if progress_every > 0 and len(out) % progress_every == 0:
            print(f"[load] resolved images: {len(out)} samples", flush=True)
        if limit > 0 and len(out) >= limit:
            break
        if progress_every > 0 and i % max(1000, progress_every * 2) == 0 and len(out) == 0:
            print(f"[load] scanned rows: {i}, still 0 samples with images", flush=True)

    label_stats["rows_with_images"] = float(len(out))
    return out, dict(excluded_by_filter), excluded_total, label_stats


class ScaleBBoxDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: Tuple[int, int],
        train: bool,
        train_shift_frac_x: float,
        train_shift_frac_y: float,
        train_shift_prob: float,
    ) -> None:
        self.samples = list(samples)
        self.train = train
        self.train_shift_frac_x = max(0.0, float(train_shift_frac_x))
        self.train_shift_frac_y = max(0.0, float(train_shift_frac_y))
        self.train_shift_prob = float(np.clip(train_shift_prob, 0.0, 1.0))
        w, h = image_size
        self.out_w = int(w)
        self.out_h = int(h)
        self.resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_jitter = transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.08, hue=0.015)
        self.train_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception:
            # Keep training resilient to transient volume I/O failures.
            img = Image.new("RGB", (max(8, s.image_w), max(8, s.image_h)), color=(0, 0, 0))
        y = np.array([s.x1_norm, s.y1_norm, s.x2_norm, s.y2_norm], dtype=np.float32)
        img = self.resize(img)

        if self.train:
            if random.random() < 0.90:
                img = self.train_jitter(img)
            if random.random() < 0.15:
                img = self.train_blur(img)

        if self.train and self.train_shift_prob > 0.0:
            if random.random() < self.train_shift_prob:
                max_dx = int(round(self.train_shift_frac_x * self.out_w))
                max_dy = int(round(self.train_shift_frac_y * self.out_h))
                dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
                dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
                if dx != 0 or dy != 0:
                    img = TF.affine(
                        img,
                        angle=0.0,
                        translate=[dx, dy],
                        scale=1.0,
                        shear=[0.0, 0.0],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=0,
                    )
                    y[0] = float(np.clip(y[0] + float(dx) / float(self.out_w), 0.0, 1.0))
                    y[2] = float(np.clip(y[2] + float(dx) / float(self.out_w), 0.0, 1.0))
                    y[1] = float(np.clip(y[1] + float(dy) / float(self.out_h), 0.0, 1.0))
                    y[3] = float(np.clip(y[3] + float(dy) / float(self.out_h), 0.0, 1.0))
                    if y[2] < y[0]:
                        y[0], y[2] = y[2], y[0]
                    if y[3] < y[1]:
                        y[1], y[3] = y[3], y[1]

        x = self.norm(TF.to_tensor(img))
        y_t = torch.tensor(y, dtype=torch.float32)
        return x, y_t, idx


class SmallScaleBBoxRegressor(nn.Module):
    def __init__(self, head_type: str = "spatial", use_coord_channels: bool = True) -> None:
        super().__init__()
        self.head_type = head_type
        self.use_coord_channels = bool(use_coord_channels)
        in_ch = 3 + (2 if self.use_coord_channels else 0)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        if self.head_type == "pooled":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 4),
            )
        elif self.head_type == "spatial":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(320),
                nn.SiLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(320, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 4),
            )
        else:
            raise ValueError(f"Unsupported head_type={head_type!r}")

    def _coord_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_coord_channels:
            return x
        b, _, h, w = x.shape
        xs = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        ys = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._coord_maps(x)
        z = self.backbone(x)
        out = self.head(z)
        return torch.sigmoid(out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bbox_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    w_x: float,
    w_y: float,
    order_penalty_weight: float,
) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none")
    weights = pred.new_tensor([float(w_x), float(w_y), float(w_x), float(w_y)]).view(1, 4)
    base = (per * weights).sum(dim=1).mean() / max(1e-6, float(2.0 * (w_x + w_y)))
    order_penalty = (torch.relu(pred[:, 0] - pred[:, 2]) + torch.relu(pred[:, 1] - pred[:, 3])).mean()
    return base + float(order_penalty_weight) * order_penalty


def pick_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req == "auto":
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if req == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device("cpu")


def box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def compute_errors(samples: Sequence[Sample], preds_norm: np.ndarray) -> Dict[str, float]:
    x_err = []
    y_err = []
    ious = []
    for s, p in zip(samples, preds_norm):
        x1_gt = s.x1_norm * s.image_w
        y1_gt = s.y1_norm * s.image_h
        x2_gt = s.x2_norm * s.image_w
        y2_gt = s.y2_norm * s.image_h
        x1_pr = float(min(p[0], p[2])) * s.image_w
        y1_pr = float(min(p[1], p[3])) * s.image_h
        x2_pr = float(max(p[0], p[2])) * s.image_w
        y2_pr = float(max(p[1], p[3])) * s.image_h

        x_err.append(0.5 * (abs(x1_pr - x1_gt) + abs(x2_pr - x2_gt)))
        y_err.append(0.5 * (abs(y1_pr - y1_gt) + abs(y2_pr - y2_gt)))
        ious.append(box_iou_xyxy((x1_pr, y1_pr, x2_pr, y2_pr), (x1_gt, y1_gt, x2_gt, y2_gt)))
    mae_x = float(np.mean(x_err)) if x_err else float("nan")
    mae_y = float(np.mean(y_err)) if y_err else float("nan")
    iou_mean = float(np.mean(ious)) if ious else float("nan")
    return {
        "rows": float(len(samples)),
        "mae_x_px": mae_x,
        "mae_y_px": mae_y,
        "iou_mean": iou_mean,
        "score_xy": float(mae_x + mae_y),
    }


@torch.no_grad()
def infer_dataset(
    model: nn.Module,
    loader: DataLoader,
    samples: Sequence[Sample],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    model.eval()
    out = np.zeros((len(samples), 4), dtype=np.float32)
    for xb, _, idxb in loader:
        xb = xb.to(device)
        pb = model(xb).detach().cpu().numpy()
        out[idxb.numpy()] = pb
    return out, compute_errors(samples, out)


def save_predictions_csv(path: Path, samples: Sequence[Sample], preds_norm: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "sample_id",
                "split",
                "image_path",
                "x1_gt",
                "y1_gt",
                "x2_gt",
                "y2_gt",
                "x1_pred",
                "y1_pred",
                "x2_pred",
                "y2_pred",
                "x_err_px",
                "y_err_px",
                "iou",
            ]
        )
        for s, p in zip(samples, preds_norm):
            x1_gt = s.x1_norm * s.image_w
            y1_gt = s.y1_norm * s.image_h
            x2_gt = s.x2_norm * s.image_w
            y2_gt = s.y2_norm * s.image_h
            x1_pr = float(min(p[0], p[2])) * s.image_w
            y1_pr = float(min(p[1], p[3])) * s.image_h
            x2_pr = float(max(p[0], p[2])) * s.image_w
            y2_pr = float(max(p[1], p[3])) * s.image_h
            x_err = 0.5 * (abs(x1_pr - x1_gt) + abs(x2_pr - x2_gt))
            y_err = 0.5 * (abs(y1_pr - y1_gt) + abs(y2_pr - y2_gt))
            iou = box_iou_xyxy((x1_pr, y1_pr, x2_pr, y2_pr), (x1_gt, y1_gt, x2_gt, y2_gt))
            wr.writerow(
                [
                    s.sample_id,
                    s.split,
                    s.image_path,
                    f"{x1_gt:.4f}",
                    f"{y1_gt:.4f}",
                    f"{x2_gt:.4f}",
                    f"{y2_gt:.4f}",
                    f"{x1_pr:.4f}",
                    f"{y1_pr:.4f}",
                    f"{x2_pr:.4f}",
                    f"{y2_pr:.4f}",
                    f"{x_err:.4f}",
                    f"{y_err:.4f}",
                    f"{iou:.5f}",
                ]
            )


def draw_preview(sample: Sample, pred_norm: np.ndarray, out_path: Path) -> None:
    img = Image.open(sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    x1_gt = sample.x1_norm * sample.image_w
    y1_gt = sample.y1_norm * sample.image_h
    x2_gt = sample.x2_norm * sample.image_w
    y2_gt = sample.y2_norm * sample.image_h
    x1_pr = float(min(pred_norm[0], pred_norm[2])) * sample.image_w
    y1_pr = float(min(pred_norm[1], pred_norm[3])) * sample.image_h
    x2_pr = float(max(pred_norm[0], pred_norm[2])) * sample.image_w
    y2_pr = float(max(pred_norm[1], pred_norm[3])) * sample.image_h

    draw.rectangle((x1_gt, y1_gt, x2_gt, y2_gt), outline=(0, 255, 120, 255), width=5, fill=(0, 255, 120, 35))
    draw.rectangle((x1_pr, y1_pr, x2_pr, y2_pr), outline=(255, 0, 220, 255), width=4, fill=(255, 0, 220, 25))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def build_review_html(
    out_dir: Path,
    split_name: str,
    samples: Sequence[Sample],
    preds_norm: np.ndarray,
    max_rows: int,
) -> Path:
    rows = []
    for s, p in zip(samples, preds_norm):
        x1_gt = s.x1_norm * s.image_w
        y1_gt = s.y1_norm * s.image_h
        x2_gt = s.x2_norm * s.image_w
        y2_gt = s.y2_norm * s.image_h
        x1_pr = float(min(p[0], p[2])) * s.image_w
        y1_pr = float(min(p[1], p[3])) * s.image_h
        x2_pr = float(max(p[0], p[2])) * s.image_w
        y2_pr = float(max(p[1], p[3])) * s.image_h
        x_err = 0.5 * (abs(x1_pr - x1_gt) + abs(x2_pr - x2_gt))
        y_err = 0.5 * (abs(y1_pr - y1_gt) + abs(y2_pr - y2_gt))
        iou = box_iou_xyxy((x1_pr, y1_pr, x2_pr, y2_pr), (x1_gt, y1_gt, x2_gt, y2_gt))
        score = x_err + y_err + (1.0 - iou) * 40.0
        rows.append((score, x_err, y_err, iou, s, p))
    rows.sort(key=lambda t: t[0], reverse=True)
    worst = rows[: max(0, int(max_rows))]

    preview_dir = out_dir / f"review_{split_name}_previews"
    html_path = out_dir / f"review_{split_name}.html"
    trs: List[str] = []
    for i, (score, x_err, y_err, iou, s, p) in enumerate(worst, start=1):
        out_img = preview_dir / f"{i:04d}_{_safe_slug(s.sample_id)}.jpg"
        try:
            draw_preview(s, p, out_img)
            img_tag = f'<a href="{preview_dir.name}/{out_img.name}" target="_blank"><img src="{preview_dir.name}/{out_img.name}" loading="lazy" /></a>'
        except Exception:
            img_tag = "preview_error"
        trs.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{s.sample_id}</td>"
            f"<td>{score:.2f}</td>"
            f"<td>{x_err:.2f}</td>"
            f"<td>{y_err:.2f}</td>"
            f"<td>{iou:.4f}</td>"
            f"<td>{Path(s.image_path).name}</td>"
            "</tr>"
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale BBox Model Review - {split_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 520px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Scale BBox Model - Worst {split_name} Errors</h1>
  <p>Overlay: <b>green=GT bbox</b>, <b>magenta=pred bbox</b></p>
  <table>
    <thead>
      <tr><th>#</th><th>Preview</th><th>sample_id</th><th>score</th><th>x_err_px</th><th>y_err_px</th><th>IoU</th><th>source</th></tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train scale bbox image regressor.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=360)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-type", type=str, default="spatial", choices=["spatial", "pooled"])
    p.add_argument("--no-coord-channels", action="store_true")
    p.add_argument("--loss-beta", type=float, default=0.02)
    p.add_argument("--loss-wx", type=float, default=2.0)
    p.add_argument("--loss-wy", type=float, default=1.1)
    p.add_argument("--order-penalty-weight", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-review-rows", type=int, default=140)
    p.add_argument("--eval-train-every", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--train-shift-frac-x", type=float, default=0.10)
    p.add_argument("--train-shift-frac-y", type=float, default=0.06)
    p.add_argument("--train-shift-prob", type=float, default=0.70)
    p.add_argument("--numbers-span-px", type=float, default=69.0)
    p.add_argument("--tick-span-px", type=float, default=30.0)
    p.add_argument("--line-half-px", type=float, default=10.0)
    p.add_argument("--pad-y-px", type=float, default=16.0)
    p.add_argument("--trim-left-px", type=float, default=35.0)
    p.add_argument("--shift-y-px", type=float, default=14.0)
    p.add_argument(
        "--folder-canonical-line",
        action="store_true",
        help="Use one canonical line21 geometry per config folder (median of rows in folder).",
    )
    p.add_argument(
        "--segment-anchor",
        action="store_true",
        help="Refine line x/y with longest-segment anchors (upper/lower) from image mask.",
    )
    p.add_argument("--segment-search-radius-px", type=int, default=120)
    p.add_argument(
        "--segment-line-offset-px",
        type=float,
        default=1.0,
        help="Place line this many px to the right of detected segment.",
    )
    p.add_argument("--segment-blend-x", type=float, default=0.45)
    p.add_argument("--segment-blend-y", type=float, default=0.35)
    p.add_argument("--segment-min-len-frac", type=float, default=0.05)
    p.add_argument("--segment-max-shift-x-px", type=float, default=22.0)
    p.add_argument("--segment-max-shift-y-px", type=float, default=70.0)
    p.add_argument("--exclude-fusion", action="store_true")
    p.add_argument("--exclude-negative", action="store_true")
    p.add_argument("--exclude-proibite", action="store_true")
    p.add_argument("--exclude-prohibited", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_labels: List[str] = []
    if args.exclude_fusion:
        exclude_labels.append("fusion")
    if args.exclude_negative:
        exclude_labels.append("negative")
    if args.exclude_proibite or args.exclude_prohibited:
        exclude_labels.append("proibite")

    samples, excluded_by_filter, excluded_total, label_stats = load_samples(
        manifest=manifest,
        limit=args.limit,
        progress_every=500,
        exclude_labels=exclude_labels,
        numbers_span_px=args.numbers_span_px,
        tick_span_px=args.tick_span_px,
        line_half_px=args.line_half_px,
        pad_y_px=args.pad_y_px,
        trim_left_px=args.trim_left_px,
        shift_y_px=args.shift_y_px,
        folder_canonical_line=bool(args.folder_canonical_line),
        use_segment_anchor=bool(args.segment_anchor),
        segment_search_radius_px=int(args.segment_search_radius_px),
        segment_line_offset_px=float(args.segment_line_offset_px),
        segment_blend_x=float(args.segment_blend_x),
        segment_blend_y=float(args.segment_blend_y),
        segment_min_len_frac=float(args.segment_min_len_frac),
        segment_max_shift_x_px=float(args.segment_max_shift_x_px),
        segment_max_shift_y_px=float(args.segment_max_shift_y_px),
    )
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        splits[s.split].append(s)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(
            f"Not enough samples by split after image resolution. "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    image_size = (args.image_width, args.image_height)
    ds_train = ScaleBBoxDataset(
        splits["train"],
        image_size=image_size,
        train=True,
        train_shift_frac_x=args.train_shift_frac_x,
        train_shift_frac_y=args.train_shift_frac_y,
        train_shift_prob=args.train_shift_prob,
    )
    ds_val = ScaleBBoxDataset(
        splits["val"],
        image_size=image_size,
        train=False,
        train_shift_frac_x=0.0,
        train_shift_frac_y=0.0,
        train_shift_prob=0.0,
    )
    ds_test = ScaleBBoxDataset(
        splits["test"],
        image_size=image_size,
        train=False,
        train_shift_frac_x=0.0,
        train_shift_frac_y=0.0,
        train_shift_prob=0.0,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    device = pick_device(args.device)
    model = SmallScaleBBoxRegressor(
        head_type=args.head_type,
        use_coord_channels=not args.no_coord_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(2, args.epochs))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_score = float("inf")
    history: List[Dict[str, float]] = []

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: List[float] = []
            for xb, yb, _ in loader_train:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = bbox_loss(
                    pred,
                    yb,
                    beta=args.loss_beta,
                    w_x=args.loss_wx,
                    w_y=args.loss_wy,
                    order_penalty_weight=args.order_penalty_weight,
                )
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"Non-finite loss detected at epoch={epoch}.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()
                train_losses.append(float(loss.item()))

            sched.step()

            val_pred, val_metrics = infer_dataset(model, loader_val, splits["val"], device)
            train_mae_x = float("nan")
            train_mae_y = float("nan")
            if args.eval_train_every > 0 and (epoch % args.eval_train_every == 0):
                _, train_metrics_epoch = infer_dataset(model, loader_train, splits["train"], device)
                train_mae_x = train_metrics_epoch["mae_x_px"]
                train_mae_y = train_metrics_epoch["mae_y_px"]

            rec = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "lr": float(opt.param_groups[0]["lr"]),
                "train_mae_x_px": train_mae_x,
                "train_mae_y_px": train_mae_y,
                "val_mae_x_px": val_metrics["mae_x_px"],
                "val_mae_y_px": val_metrics["mae_y_px"],
                "val_iou_mean": val_metrics["iou_mean"],
                "val_score_xy": val_metrics["score_xy"],
            }
            history.append(rec)
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            print(
                f"[epoch {epoch:03d}] "
                f"loss={rec['train_loss']:.5f} "
                f"val(x,y,iou)=({rec['val_mae_x_px']:.2f}, {rec['val_mae_y_px']:.2f}, {rec['val_iou_mean']:.3f}) "
                f"val_score={rec['val_score_xy']:.2f}",
                flush=True,
            )

            if val_metrics["score_xy"] < best_val_score:
                best_val_score = val_metrics["score_xy"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "val_score_xy": best_val_score,
                        "image_size": image_size,
                    },
                    out_dir / "best_model.pt",
                )
                save_predictions_csv(out_dir / "val_predictions_best.csv", splits["val"], val_pred)

    if best_state is None:
        raise RuntimeError("Training did not produce any best checkpoint.")

    model.load_state_dict(best_state, strict=True)
    train_pred, train_metrics = infer_dataset(model, loader_train, splits["train"], device)
    val_pred, val_metrics = infer_dataset(model, loader_val, splits["val"], device)
    test_pred, test_metrics = infer_dataset(model, loader_test, splits["test"], device)

    save_predictions_csv(out_dir / "train_predictions_best.csv", splits["train"], train_pred)
    save_predictions_csv(out_dir / "val_predictions_best.csv", splits["val"], val_pred)
    save_predictions_csv(out_dir / "test_predictions_best.csv", splits["test"], test_pred)
    val_html = build_review_html(out_dir, "val", splits["val"], val_pred, args.max_review_rows)
    test_html = build_review_html(out_dir, "test", splits["test"], test_pred, args.max_review_rows)

    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": out_dir.as_posix(),
        "exclusion_filters_enabled": exclude_labels,
        "rows_excluded_total": int(excluded_total),
        "rows_excluded_by_filter": excluded_by_filter,
        "rows_excluded_fusion": int(excluded_by_filter.get("fusion", 0)),
        "rows_excluded_negative": int(excluded_by_filter.get("negative", 0)),
        "rows_excluded_proibite": int(excluded_by_filter.get("proibite", 0)),
        "rows_total_with_images": len(samples),
        "rows_by_split": {k: len(v) for k, v in splits.items()},
        "bbox_params": {
            "numbers_span_px": args.numbers_span_px,
            "tick_span_px": args.tick_span_px,
            "line_half_px": args.line_half_px,
            "pad_y_px": args.pad_y_px,
            "trim_left_px": args.trim_left_px,
            "shift_y_px": args.shift_y_px,
            "folder_canonical_line": bool(args.folder_canonical_line),
            "segment_anchor": bool(args.segment_anchor),
            "segment_search_radius_px": args.segment_search_radius_px,
            "segment_line_offset_px": args.segment_line_offset_px,
            "segment_blend_x": args.segment_blend_x,
            "segment_blend_y": args.segment_blend_y,
            "segment_min_len_frac": args.segment_min_len_frac,
            "segment_max_shift_x_px": args.segment_max_shift_x_px,
            "segment_max_shift_y_px": args.segment_max_shift_y_px,
        },
        "label_generation_stats": label_stats,
        "image_size": {"width": args.image_width, "height": args.image_height},
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "head_type": args.head_type,
        "use_coord_channels": bool(not args.no_coord_channels),
        "loss_beta": args.loss_beta,
        "loss_wx": args.loss_wx,
        "loss_wy": args.loss_wy,
        "order_penalty_weight": args.order_penalty_weight,
        "train_shift_frac_x": args.train_shift_frac_x,
        "train_shift_frac_y": args.train_shift_frac_y,
        "train_shift_prob": args.train_shift_prob,
        "eval_train_every": args.eval_train_every,
        "best_epoch": best_epoch,
        "best_val_score_xy": best_val_score,
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        "artifacts": {
            "best_model": (out_dir / "best_model.pt").as_posix(),
            "train_log_jsonl": (out_dir / "train_log.jsonl").as_posix(),
            "train_predictions_csv": (out_dir / "train_predictions_best.csv").as_posix(),
            "val_predictions_csv": (out_dir / "val_predictions_best.csv").as_posix(),
            "test_predictions_csv": (out_dir / "test_predictions_best.csv").as_posix(),
            "val_review_html": val_html.as_posix(),
            "test_review_html": test_html.as_posix(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
