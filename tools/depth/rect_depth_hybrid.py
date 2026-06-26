#!/usr/bin/env python3
"""Hybrid/classical utilities for legacy RECT_DEPTH recognition.

The module has two intentionally small entry points:

1. ``manifest`` parses legacy ``.fss`` files and exports the RECT_DEPTH
   ground truth plus optional crops from ``image_samples``.
2. ``predict`` runs a folder-level OCR baseline that looks for recurring
   depth-like numeric text, with an optional ``.fss`` only for expected-depth
   snapping/evaluation.

The neural part is deliberately left as a downstream consumer of the manifest:
the manifest gives clean crops, labels, flip state and source frame paths
without spending training time while the data root is still being checked.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - optional at runtime
    cv2 = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
FLIP_STATES = ("nf", "lr", "ud", "lrud")
DEPTH_IMAGE_STEMS = {
    "nf": (
        "image_depth_find_no_flip_setup_{idx0}",
        "image_depth_value_setup_{idx0}",
        "image_biplana_setup_{idx0}",
        "image_CalGrid_depth_{idx0}",
        "image_depth_find_no_flip_setup_{idx1}",
        "image_depth_value_setup_{idx1}",
    ),
    "lr": (
        "image_depth_find_flip_lr_setup_{idx0}",
        "image_depth_find_flip_lr_setup_{idx1}",
    ),
    "ud": (
        "image_depth_find_flip_ud_setup_{idx0}",
        "image_depth_find_flip_ud_setup_{idx1}",
    ),
    "lrud": (
        "image_depth_find_flip_lrud_setup_{idx0}",
        "image_depth_find_flip_lrud_setup_{idx1}",
    ),
}


@dataclass(frozen=True)
class MatchParams:
    channel: str = ""
    threshold: str = ""
    p1: str = ""
    p2: str = ""
    p3: str = ""
    p4: str = ""
    p5: str = ""
    raw: str = ""


@dataclass(frozen=True)
class RectDepthCheck:
    depth_index0: int
    depth_mm: float
    flip_state: str
    top: int
    left: int
    bottom: int
    right: int
    b: int
    params: MatchParams
    match_method: str
    bm: str
    params2: MatchParams
    raw: str

    @property
    def width(self) -> int:
        return max(0, self.right - self.left + 1)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top + 1)


@dataclass(frozen=True)
class SourceImage:
    path: Optional[Path]
    strategy: str


@dataclass(frozen=True)
class OCRWord:
    image_path: Path
    image_index: int
    text: str
    norm_text: str
    conf: float
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def x_center(self) -> float:
        return self.left + 0.5 * self.width

    @property
    def y_center(self) -> float:
        return self.top + 0.5 * self.height


@dataclass
class DepthToken:
    word: OCRWord
    numeric_value: float
    has_cm_hint: bool = False
    has_mm_hint: bool = False
    has_depth_hint: bool = False
    has_d_hint: bool = False
    has_p_hint: bool = False
    has_r_hint: bool = False
    has_scale_hint: bool = False
    has_fps_ips_hint: bool = False
    snapped_depth_mm: Optional[float] = None
    snap_mode: str = ""
    snap_error_mm: float = float("nan")


@dataclass
class DepthCluster:
    cluster_id: int
    tokens: List[DepthToken] = field(default_factory=list)
    score: float = 0.0
    cm_ratio: float = 0.0
    expected_ratio: float = 0.0
    plausible_ratio: float = 0.0
    support_ratio: float = 0.0
    mm_ratio: float = 0.0
    depth_hint_ratio: float = 0.0
    d_hint_ratio: float = 0.0
    p_hint_ratio: float = 0.0
    r_hint_ratio: float = 0.0
    scale_hint_ratio: float = 0.0
    fps_ips_ratio: float = 0.0
    side_score: float = 0.0
    accessory_score: float = 0.0
    echo_center_penalty: float = 0.0
    unique_values: int = 0
    reason: str = ""

    @property
    def images(self) -> set[int]:
        return {t.word.image_index for t in self.tokens}

    @property
    def x_center(self) -> float:
        if not self.tokens:
            return 0.0
        return float(np.median([t.word.x_center for t in self.tokens]))

    @property
    def y_center(self) -> float:
        if not self.tokens:
            return 0.0
        return float(np.median([t.word.y_center for t in self.tokens]))

    @property
    def left(self) -> float:
        if not self.tokens:
            return 0.0
        return float(min(t.word.left for t in self.tokens))

    @property
    def top(self) -> float:
        if not self.tokens:
            return 0.0
        return float(min(t.word.top for t in self.tokens))

    @property
    def right(self) -> float:
        if not self.tokens:
            return 0.0
        return float(max(t.word.right for t in self.tokens))

    @property
    def bottom(self) -> float:
        if not self.tokens:
            return 0.0
        return float(max(t.word.bottom for t in self.tokens))


def _safe_slug(text: str, max_len: int = 140) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _safe_float(text: str, default: float = float("nan")) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def _safe_int(text: str, default: int = 0) -> int:
    try:
        return int(float(str(text).strip()))
    except Exception:
        return default


def _read_fss_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")
    if len(lines) < 18:
        raise ValueError(f"{path} has {len(lines)} lines, expected at least 18.")
    return lines


def _pipe_float_vector(line: str) -> List[float]:
    values: List[float] = []
    for tok in line.split("|"):
        tok = tok.strip()
        if tok:
            values.append(_safe_float(tok))
    return values


def parse_rect_echo(fss_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return RECT_ECHO as (left, top, right, bottom) in image coordinates."""
    try:
        lines = _read_fss_lines(fss_path)
    except Exception:
        return None
    if len(lines) < 11:
        return None
    parts = [p.strip() for p in lines[10].split("|") if p.strip()]
    if len(parts) < 4:
        return None
    try:
        top = float(parts[0])
        left = float(parts[1])
        bottom = float(parts[2])
        right = float(parts[3])
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _parse_match_params(raw: str) -> MatchParams:
    pieces = [p.strip() for p in str(raw).split(":")]
    pieces += [""] * max(0, 7 - len(pieces))
    return MatchParams(
        channel=pieces[0],
        threshold=pieces[1],
        p1=pieces[2],
        p2=pieces[3],
        p3=pieces[4],
        p4=pieces[5],
        p5=pieces[6],
        raw=str(raw).strip(),
    )


def parse_rect_depth_checks(fss_path: Path) -> Tuple[List[RectDepthCheck], List[float]]:
    lines = _read_fss_lines(fss_path)
    rect_depth_line = lines[16].strip()
    depth_values = _pipe_float_vector(lines[17])

    checks: List[RectDepthCheck] = []
    depth_groups = [g.strip() for g in rect_depth_line.split(",") if g.strip()]
    for depth_index0, group in enumerate(depth_groups):
        depth_mm = depth_values[depth_index0] if depth_index0 < len(depth_values) else float("nan")
        raw_records = [r.strip() for r in group.split(";") if r.strip()]
        for flip_idx, raw in enumerate(raw_records[:4]):
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 7:
                continue
            flip = FLIP_STATES[flip_idx] if flip_idx < len(FLIP_STATES) else f"flip_{flip_idx}"
            checks.append(
                RectDepthCheck(
                    depth_index0=depth_index0,
                    depth_mm=depth_mm,
                    flip_state=flip,
                    top=_safe_int(parts[0]),
                    left=_safe_int(parts[1]),
                    bottom=_safe_int(parts[2]),
                    right=_safe_int(parts[3]),
                    b=_safe_int(parts[4]),
                    params=_parse_match_params(parts[5]),
                    match_method=parts[6],
                    bm=parts[7] if len(parts) > 7 else "",
                    params2=_parse_match_params(parts[8] if len(parts) > 8 else ""),
                    raw=raw,
                )
            )
    return checks, depth_values


def find_config_dirs(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        names = set(dirnames)
        if "DB_setup" in names and "image_samples" in names:
            yield path
            dirnames[:] = [d for d in dirnames if d not in {"DB_setup", "DB_echo", "image_samples"}]
            continue
        if path.name in {"DB_setup", "DB_echo", "image_samples", "__pycache__"}:
            dirnames[:] = []


def iter_images(image_dir: Path) -> List[Path]:
    if (image_dir / "image_samples").is_dir():
        image_dir = image_dir / "image_samples"
    out: List[Path] = []
    for ext in IMAGE_EXTS:
        out.extend(p for p in image_dir.glob(f"*{ext}") if p.is_file() and not p.name.startswith("._"))
    def priority(p: Path) -> Tuple[int, str]:
        name = p.name.lower()
        if name.startswith("image_depth_"):
            return (0, name)
        if name.startswith("image_biplana_"):
            return (1, name)
        if name.startswith("image_orientation_"):
            return (2, name)
        if "calibration" in name:
            return (8, name)
        return (5, name)

    return sorted(set(out), key=priority)


def _first_existing_image(image_samples: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = image_samples / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_source_image(config_dir: Path, setup_id: str, depth_index0: int, flip_state: str) -> SourceImage:
    image_samples = config_dir / "image_samples"
    idx0 = max(0, depth_index0)
    idx1 = idx0 + 1
    for stem_tmpl in DEPTH_IMAGE_STEMS.get(flip_state, ()):
        stem = stem_tmpl.format(idx0=idx0, idx1=idx1, setup_id=setup_id)
        found = _first_existing_image(image_samples, stem)
        if found:
            return SourceImage(path=found, strategy=stem)

    fallbacks = [
        f"image_depth_value_setup_{idx0}",
        f"image_depth_find_flip_ud_setup_{idx0}",
        f"image_depth_find_no_flip_setup_{idx0}",
        f"image_orientation_setup_{idx0}",
        "image_orientation_setup_0",
    ]
    for stem in fallbacks:
        found = _first_existing_image(image_samples, stem)
        if found:
            return SourceImage(path=found, strategy=f"fallback:{stem}")

    imgs = iter_images(image_samples)
    if imgs:
        return SourceImage(path=imgs[0], strategy="fallback:first_image")
    return SourceImage(path=None, strategy="missing")


def find_depth_template(config_dir: Path, setup_id: str, depth_index0: int, template_root: Optional[Path]) -> Optional[Path]:
    candidates = [
        config_dir / "DB_echo" / f"setup_{setup_id}" / f"depth_{depth_index0}.png",
        config_dir / "DB_echo" / f"setup_{setup_id}" / f"depth_{depth_index0}.jpg",
    ]
    if template_root:
        candidates.extend(
            [
                template_root / f"setup_{setup_id}" / f"depth_{depth_index0}.png",
                template_root / f"setup_{setup_id}" / f"depth_{depth_index0}.jpg",
                template_root / "DB_echo" / f"setup_{setup_id}" / f"depth_{depth_index0}.png",
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    return None


def crop_rect(image_path: Path, check: RectDepthCheck, out_path: Path) -> None:
    with Image.open(image_path) as im:
        w, h = im.size
        left = max(0, min(w - 1, check.left))
        top = max(0, min(h - 1, check.top))
        right = max(left + 1, min(w, check.right + 1))
        bottom = max(top + 1, min(h, check.bottom + 1))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.crop((left, top, right, bottom)).save(out_path)


def build_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    crops_dir = Path(args.crops_dir).expanduser().resolve() if args.crops_dir else None
    template_root = Path(args.template_root).expanduser().resolve() if args.template_root else None

    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    config_count = 0

    for config_dir in find_config_dirs(dataset_root):
        fss_files = sorted((config_dir / "DB_setup").glob("*.fss"))
        if not fss_files:
            warnings.append(f"{config_dir}: missing .fss in DB_setup")
            continue
        for fss_path in fss_files:
            setup_id = fss_path.stem.replace("setup_", "")
            try:
                checks, depth_values = parse_rect_depth_checks(fss_path)
            except Exception as exc:
                warnings.append(f"{fss_path}: parse failed: {exc}")
                continue
            for check in checks:
                source = find_source_image(config_dir, setup_id, check.depth_index0, check.flip_state)
                template = find_depth_template(config_dir, setup_id, check.depth_index0, template_root)
                crop_path = ""
                if crops_dir and source.path:
                    rel = config_dir.relative_to(dataset_root)
                    crop_name = (
                        f"{_safe_slug(str(rel))}_setup_{setup_id}_"
                        f"depth_{check.depth_index0:03d}_{check.flip_state}.png"
                    )
                    out_crop = crops_dir / crop_name
                    crop_rect(source.path, check, out_crop)
                    crop_path = out_crop.as_posix()

                rows.append(
                    {
                        "config_folder": config_dir.relative_to(dataset_root).as_posix(),
                        "fss_path": fss_path.as_posix(),
                        "setup_id": setup_id,
                        "depth_index0": check.depth_index0,
                        "depth_index1": check.depth_index0 + 1,
                        "depth_mm": check.depth_mm,
                        "depth_count": len(depth_values),
                        "flip_state": check.flip_state,
                        "top": check.top,
                        "left": check.left,
                        "bottom": check.bottom,
                        "right": check.right,
                        "width": check.width,
                        "height": check.height,
                        "b": check.b,
                        "channel": check.params.channel,
                        "threshold": check.params.threshold,
                        "p1": check.params.p1,
                        "p2": check.params.p2,
                        "p3": check.params.p3,
                        "p4": check.params.p4,
                        "p5": check.params.p5,
                        "match_method": check.match_method,
                        "bm": check.bm,
                        "params2": check.params2.raw,
                        "source_image": source.path.as_posix() if source.path else "",
                        "source_strategy": source.strategy,
                        "template_path": template.as_posix() if template else "",
                        "crop_path": crop_path,
                        "rect_depth_raw": check.raw,
                    }
                )
        config_count += 1
        if args.max_configs and config_count >= args.max_configs:
            break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_folder",
        "fss_path",
        "setup_id",
        "depth_index0",
        "depth_index1",
        "depth_mm",
        "depth_count",
        "flip_state",
        "top",
        "left",
        "bottom",
        "right",
        "width",
        "height",
        "b",
        "channel",
        "threshold",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "match_method",
        "bm",
        "params2",
        "source_image",
        "source_strategy",
        "template_path",
        "crop_path",
        "rect_depth_raw",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "output_csv": output_csv.as_posix(),
        "rows": len(rows),
        "configs_seen": config_count,
        "warnings": warnings[:50],
        "warnings_total": len(warnings),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _normalize_for_ocr(im: Image.Image, max_side: int) -> Image.Image:
    im = im.convert("L")
    w, h = im.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    elif max(w, h) < 1200:
        scale = min(2.0, 1200.0 / float(max(w, h)))
    if scale != 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC)
    im = ImageOps.autocontrast(im)
    return im


def _preprocess_ocr_image(im: Image.Image, max_side: int, variant: str) -> Image.Image:
    """Prepare small scale labels that overlap the ultrasound texture.

    The base OCR pass stays intentionally unchanged.  Enhanced passes are only
    used on compact endpoint ROIs and make bright text survive a textured or
    partly bright ultrasound background.
    """
    normalized = _normalize_for_ocr(im, max_side=max_side)
    if variant == "base" or cv2 is None:
        return normalized

    array = np.asarray(normalized, dtype=np.uint8)

    if variant == "scale_clahe":
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        enhanced = clahe.apply(array)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        result = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    elif variant == "scale_tophat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        bright = cv2.morphologyEx(array, cv2.MORPH_TOPHAT, kernel)
        bright = cv2.normalize(bright, None, 0, 255, cv2.NORM_MINMAX)
        binary = cv2.adaptiveThreshold(bright, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -3)
        result = 255 - binary
    elif variant == "scale_line_suppressed":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(array)
        kernel_width = max(17, min(61, int(round(0.22 * enhanced.shape[1]))))
        horizontal = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
        residual = cv2.subtract(enhanced, horizontal)
        _ret, binary = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = 255 - binary
    else:
        return normalized
    return Image.fromarray(result.astype(np.uint8), mode="L")


def _clamp_crop_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    if right - left < 40 or bottom - top < 28:
        return None
    return left, top, right, bottom


def _ocr_roi_boxes(
    width: int,
    height: int,
    rect_echo: Optional[Tuple[float, float, float, float]],
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []

    def add(box: Tuple[float, float, float, float]) -> None:
        clamped = _clamp_crop_box(box, width, height)
        if not clamped:
            return
        if clamped not in boxes:
            boxes.append(clamped)

    # Generic accessory zones: depth labels tend to sit around the image, not
    # in the middle of the ultrasound content.
    add((0, 0, 0.44 * width, height))
    add((0.56 * width, 0, width, height))
    add((0, 0, width, 0.36 * height))
    add((0, 0.62 * height, width, height))
    add((0, 0.70 * height, 0.58 * width, 0.94 * height))
    add((0.42 * width, 0.70 * height, width, 0.94 * height))
    add((0, 0.78 * height, 0.58 * width, height))
    add((0.42 * width, 0.78 * height, width, height))
    # Tight scale-value crops. On Philips and similar layouts the max scale
    # value is often a small right-bottom string (for example "3.5cm"); broad
    # bottom OCR passes can merge it with ultrasound texture and lose the
    # leading digit.
    add((0.60 * width, 0.90 * height, 0.84 * width, height))
    add((0.64 * width, 0.92 * height, 0.80 * width, height))
    add((0.58 * width, 0.905 * height, 0.755 * width, 0.995 * height))
    add((0.615 * width, 0.94 * height, 0.74 * width, 0.987 * height))
    add((0.72 * width, 0.86 * height, width, height))
    add((0.80 * width, 0.90 * height, width, height))
    # Rare fallback for left-side scales.
    add((0, 0.86 * height, 0.32 * width, height))
    add((0, 0, 0.58 * width, 0.24 * height))
    add((0.42 * width, 0, width, 0.24 * height))

    if rect_echo is not None:
        left, top, right, bottom = rect_echo
        x_margin = max(80.0, 0.12 * width)
        y_margin = max(55.0, 0.08 * height)
        add((0, top - y_margin, left + x_margin, bottom + y_margin))
        add((right - x_margin, top - y_margin, width, bottom + y_margin))
        add((left - x_margin, 0, right + x_margin, top + y_margin))
        add((left - x_margin, bottom - y_margin, right + x_margin, height))
        # Compact rows around the echo border. Hitachi/Fujifilm layouts often
        # put R/P/D depth values just below the echo, close to a side edge; a
        # full bottom-band OCR pass tends to miss those low-contrast strings.
        row_h = max(130.0, 0.12 * height)
        row_w = max(420.0, 0.28 * width)
        add((left - 120.0, bottom - 95.0, left + row_w, bottom + row_h))
        add((right - row_w, bottom - 95.0, right + 120.0, bottom + row_h))
        add((left - 120.0, top - row_h, left + row_w, top + 100.0))
        add((right - row_w, top - row_h, right + 120.0, top + 100.0))
        add((left - 100.0, bottom - 105.0, left + 0.52 * (right - left), bottom + row_h))
        add((left + 0.48 * (right - left), bottom - 105.0, right + 100.0, bottom + row_h))

    return boxes


def _scale_endpoint_roi_boxes(image_path: Path) -> List[Tuple[int, int, int, int]]:
    """Find compact OCR crops around the two ends of a vertical tick column."""
    if cv2 is None:
        return []
    frame = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return []
    height, width = frame.shape[:2]
    if width < 80 or height < 80:
        return []
    # Restrict the search to the imaging region.  The old global component
    # scan could lock onto tiny UI controls in the lower settings panel and
    # consequently OCR the wrong "endpoint".
    y0, y1 = int(0.12 * height), int(0.78 * height)
    _ret, binary = cv2.threshold(frame[y0:y1, :], 180, 255, cv2.THRESH_BINARY)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components: List[Tuple[float, float, float, float, float, float, float]] = []
    tick_points: List[Tuple[float, float]] = []
    for idx in range(1, count):
        x, y, comp_w, comp_h, area = stats[idx]
        if area < 5 or area > 1500 or comp_w < 2 or comp_h < 2:
            continue
        cx = float(centroids[idx][0])
        cy = float(y0 + centroids[idx][1])
        component = (float(x), float(y0 + y), float(comp_w), float(comp_h), float(area), cx, cy)
        components.append(component)
        if (
            3 <= comp_w <= 18
            and 2 <= comp_h <= 12
            and 5 <= area <= 180
            and 0.10 * width <= cx <= 0.90 * width
        ):
            tick_points.append((cx, cy))
    if not tick_points:
        return []

    groups: Dict[int, List[Tuple[float, float]]] = {}
    for x, y in tick_points:
        groups.setdefault(int(round(x / 12.0)), []).append((x, y))
    scored: List[Tuple[float, float, List[float]]] = []
    for group in groups.values():
        ys = [y for _x, y in group]
        distinct_y = len({int(round(y / 30.0)) for y in ys})
        span = max(ys) - min(ys) if ys else 0.0
        if distinct_y < 2 or span < 80.0:
            continue
        lane_x = float(np.median([x for x, _y in group]))
        # Values are normally just left of the tick column.  Require textual
        # components on those rows so decorative tick-like controls lose.
        label_ys: List[float] = []
        for tick_y in ys:
            if any(
                lane_x - 95.0 <= item[5] <= lane_x - 8.0
                and abs(item[6] - tick_y) <= 25.0
                and item[4] >= 5.0
                for item in components
            ):
                label_ys.append(tick_y)
        if len(label_ys) < 2:
            continue
        right_bias = 0.35 if lane_x >= 0.50 * width else 0.0
        scored.append((2.0 * len(label_ys) + distinct_y + min(5.0, span / 100.0) + right_bias, lane_x, label_ys))
    if not scored:
        return []

    _score, lane_x, label_ys = max(scored, key=lambda item: item[0])
    boxes: List[Tuple[int, int, int, int]] = []
    # Ignore unlabeled terminal ticks: the depth is the outermost *numeric*
    # scale label, not a decorative end-cap of the ultrasound rectangle.
    for y in (min(label_ys), max(label_ys)):
        box = _clamp_crop_box((lane_x - 135.0, y - 62.0, lane_x + 75.0, y + 62.0), width, height)
        if box and box not in boxes:
            boxes.append(box)
    return boxes


def run_tesseract_tsv_region(
    image_path: Path,
    timeout: float,
    max_side: int,
    crop_box: Optional[Tuple[float, float, float, float]] = None,
    psm: str = "11",
    preprocess: str = "base",
    char_whitelist: str = "0123456789.,:/-%aAcCmMdDeEpPrRtThHfFiIsSzZlL",
) -> Tuple[List[OCRWord], Tuple[float, float]]:
    try:
        with Image.open(image_path) as original:
            ow, oh = original.size
            if crop_box is not None:
                clamped = _clamp_crop_box(crop_box, ow, oh)
                if not clamped:
                    return [], (1.0, 1.0)
                crop_left, crop_top, crop_right, crop_bottom = clamped
                work = original.crop((crop_left, crop_top, crop_right, crop_bottom))
            else:
                crop_left, crop_top = 0, 0
                crop_right, crop_bottom = ow, oh
                work = original.copy()
            im = _preprocess_ocr_image(work, max_side=max_side, variant=preprocess)
    except Exception:
        return [], (1.0, 1.0)

    sw = im.size[0] / float(max(1, crop_right - crop_left))
    sh = im.size[1] / float(max(1, crop_bottom - crop_top))
    with tempfile.TemporaryDirectory(prefix="rect_depth_ocr_") as td:
        in_path = Path(td) / "frame.png"
        im.save(in_path)
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
            f"tessedit_char_whitelist={char_whitelist}",
            "tsv",
        ]
        try:
            cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return [], (sw, sh)
    if cp.returncode != 0 or not cp.stdout.strip():
        return [], (sw, sh)

    out: List[OCRWord] = []
    for row in csv.DictReader(io.StringIO(cp.stdout), delimiter="\t"):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        try:
            conf = float(str(row.get("conf", "-1")).strip())
            left = float(row.get("left", "0")) / sw + crop_left
            top = float(row.get("top", "0")) / sh + crop_top
            width = float(row.get("width", "0")) / sw
            height = float(row.get("height", "0")) / sh
        except Exception:
            continue
        if width <= 1.0 or height <= 1.0:
            continue
        norm = text.lower().replace(" ", "")
        out.append(
            OCRWord(
                image_path=image_path,
                image_index=-1,
                text=text,
                norm_text=norm,
                conf=conf,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
    return out, (sw, sh)


def run_tesseract_tsv(image_path: Path, timeout: float, max_side: int) -> Tuple[List[OCRWord], Tuple[float, float]]:
    return run_tesseract_tsv_region(image_path, timeout=timeout, max_side=max_side, crop_box=None, psm="11")


_VISION_OCR_BINARY: Optional[Path] = None
_VISION_OCR_UNAVAILABLE = False


def _vision_ocr_binary() -> Optional[Path]:
    """Compile the optional macOS Vision OCR helper once per local machine."""
    global _VISION_OCR_BINARY, _VISION_OCR_UNAVAILABLE
    if _VISION_OCR_UNAVAILABLE:
        return None
    source = Path(__file__).resolve().with_name("vision_text_ocr.swift")
    if not source.exists() or not shutil.which("swiftc"):
        _VISION_OCR_UNAVAILABLE = True
        return None
    binary = Path(tempfile.gettempdir()) / "esi_builder_vision_text_ocr"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        _VISION_OCR_BINARY = binary
        return binary
    try:
        compiled = subprocess.run(
            ["swiftc", source.as_posix(), "-O", "-o", binary.as_posix()],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _VISION_OCR_UNAVAILABLE = True
        return None
    if compiled.returncode != 0 or not binary.exists():
        _VISION_OCR_UNAVAILABLE = True
        return None
    _VISION_OCR_BINARY = binary
    return binary


def run_vision_ocr_region(
    image_path: Path,
    timeout: float,
    crop_box: Tuple[float, float, float, float],
) -> List[OCRWord]:
    """Use the local macOS Vision OCR engine on a supplied UI region."""
    binary = _vision_ocr_binary()
    if binary is None:
        return []
    try:
        with Image.open(image_path) as image:
            clamped = _clamp_crop_box(crop_box, image.size[0], image.size[1])
    except Exception:
        return []
    if not clamped:
        return []
    try:
        result = subprocess.run(
            [binary.as_posix(), image_path.as_posix(), *(str(value) for value in clamped)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    out: List[OCRWord] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        text = parts[0].strip()
        if not text:
            continue
        try:
            confidence = 100.0 * float(parts[1])
            left, top, width, height = (float(value) for value in parts[2:6])
        except Exception:
            continue
        if width <= 1.0 or height <= 1.0:
            continue
        out.append(
            OCRWord(
                image_path=image_path,
                image_index=-1,
                text=text,
                norm_text=text.lower().replace(" ", ""),
                conf=confidence,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
    return out


def _dedupe_ocr_words(words: Sequence[OCRWord]) -> List[OCRWord]:
    kept: List[OCRWord] = []
    for word in sorted(words, key=lambda w: (-w.conf, w.top, w.left)):
        duplicate = False
        for prev in kept:
            if word.norm_text != prev.norm_text:
                continue
            max_center_delta = max(5.0, 0.35 * max(word.height, prev.height))
            if abs(word.x_center - prev.x_center) > max_center_delta:
                continue
            if abs(word.y_center - prev.y_center) > max_center_delta:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(word)
    return sorted(kept, key=lambda w: (w.top, w.left, -w.conf))


def _numeric_value(text: str) -> Optional[float]:
    raw = text.strip().replace(",", ".")
    # Vision commonly returns a scale number with the adjacent tick, for
    # example ``2-``. It is still a clean numeric value when the dash is only
    # the final character, never a separator between two numbers.
    if re.fullmatch(r"\d+(?:\.\d+)?[-–]", raw):
        raw = raw[:-1]
    if any(ch in raw for ch in ":/-"):
        return None
    cleaned = re.sub(r"[^0-9.]", "", raw)
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _has_strong_embedded_depth_pattern(text: str) -> bool:
    low = text.lower().replace(" ", "").replace(",", ".")
    if not re.search(r"\d", low):
        return False
    if re.search(r"(depth|dep|dept|dpth)", low):
        return True
    if re.search(r"[dpr]\s*[:./-]?\s*\d", low):
        return True
    if re.search(r"\d+(?:\.\d+)?\s*(cm|mm)\b", low):
        return True
    # Tesseract sometimes drops the final "m" in scale labels such as
    # "3.0cm" -> "3.0c". Keep these short numeric fragments alive; the
    # autonomous postprocess will only trust them when geometry also looks like
    # a scale edge value.
    if re.fullmatch(r"\d+(?:\.\d+)?c\W*", low):
        return True
    # Tight right-scale crops can misread "cm" as "em" or "tm" on Philips.
    if re.fullmatch(r"(?:s0|\d+(?:\.\d+)?)\s*(?:em|tm)\W*", low):
        return True
    return False


def _numeric_values(text: str) -> List[float]:
    primary = _numeric_value(text)
    if primary is not None:
        return [primary]

    raw = text.strip().replace(",", ".")
    if not _has_strong_embedded_depth_pattern(raw):
        return []
    if re.search(r"(?i)(?<![a-z0-9])s\.\d+\s*(?:cm|c)(?![a-z])", raw):
        raw = re.sub(r"(?i)(?<![a-z0-9])s(?=\.\d+\s*(?:cm|c)(?![a-z]))", "3", raw)
    raw = re.sub(r"(?i)(?<![a-z0-9])s0(?=\s*(?:cm|c|em|tm)(?![a-z]))", "3.0", raw)
    raw = re.sub(r"(?i)(?<=\d)(?:em|tm)(?![a-z])", "cm", raw)

    values: List[float] = []
    for match in re.finditer(r"\d+(?:\.\d+)?", raw):
        try:
            value = float(match.group(0))
        except Exception:
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        if not _is_plausible_depth_number(value):
            continue
        if all(abs(value - old) > 1e-6 for old in values):
            values.append(value)
    return values


def _looks_like_cm(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    return "cm" in t or t == "c" or bool(re.search(r"(?i)\d\s*(?:em|tm)\b", text))


def _looks_like_mm(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    return "mm" in t


def _looks_like_depth_hint(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    if not t:
        return False
    return "depth" in t or t in {"dep", "deph", "dept", "dpth"}


def _looks_like_d_hint(text: str) -> bool:
    low = text.lower().replace(" ", "")
    t = re.sub(r"[^a-z]", "", low)
    return (t == "d" and not re.search(r"\d", low)) or bool(re.search(r"d\s*[:./-]?\s*\d", low))


def _looks_like_p_hint(text: str) -> bool:
    low = text.lower().replace(" ", "")
    t = re.sub(r"[^a-z]", "", low)
    return (t == "p" and not re.search(r"\d", low)) or bool(re.search(r"p\s*[:./-]?\s*\d", low))


def _looks_like_r_hint(text: str) -> bool:
    low = text.lower().replace(" ", "")
    t = re.sub(r"[^a-z]", "", low)
    return (t == "r" and not re.search(r"\d", low)) or bool(re.search(r"r\s*[:./-]?\s*\d", low))


def _looks_like_scale_hint(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    if not t:
        return False
    return "scale" in t or "scala" in t or t in {"scal", "cale", "sla", "sca"}


def _looks_like_fps_ips(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    return t in {"fps", "ips", "fp", "ip"} or "fps" in t or "ips" in t


def _has_non_depth_numeric_marker(text: str) -> bool:
    """Reject UI numbers that are explicitly unrelated to a depth."""
    low = str(text or "").lower()
    return bool(
        re.search(r"\bprint\b", low)
        or re.search(r"\b(?:m\s*hz|hz|d\s*b)\b", low)
        or re.search(r"\bfr\s*\d", low)
        # The depth marker must precede the number. A trailing D is an image
        # mode (for example `2D`), never a depth label.
        or re.search(r"\b\d+(?:\.\d+)?\s*d\b", low)
    )


def _near_same_label_line(a: OCRWord, b: OCRWord, max_dx: float = 150.0) -> bool:
    if abs(a.y_center - b.y_center) > max(24.0, 1.6 * max(a.height, b.height)):
        return False
    return abs(a.x_center - b.x_center) <= max_dx


def _near_value_context(label: OCRWord, value: OCRWord, max_dx: float = 190.0, max_dy: float = 70.0) -> bool:
    """Return True when an OCR label is close enough to describe a number.

    Depth annotations are often compact side UI elements: units may be on the
    right of the number, while D/P/Depth/Scale labels may be left, right, above,
    or just offset on the same side scale. This deliberately stays local so the
    generic header text does not explain unrelated numbers.
    """
    dx = abs(label.x_center - value.x_center)
    dy = abs(label.y_center - value.y_center)
    if dx <= max_dx and dy <= max(28.0, 1.8 * max(label.height, value.height)):
        return True
    return dx <= max(95.0, 1.4 * max(label.width, value.width)) and dy <= max_dy


def _near_left_label(label: OCRWord, value: OCRWord, max_dx: float = 160.0) -> bool:
    if not _near_same_label_line(label, value, max_dx=max_dx):
        return False
    return label.x_center <= value.x_center + max(12.0, value.width)


def _is_plausible_depth_number(value: float) -> bool:
    if value <= 0:
        return False
    if 1900 <= value <= 2100:
        return False
    return 0.5 <= value <= 350.0


def _nearest_expected(candidate: float, expected_depths_mm: Sequence[float]) -> Tuple[Optional[float], float]:
    vals = [v for v in expected_depths_mm if math.isfinite(v) and v > 0]
    if not vals:
        return None, float("nan")
    best = min(vals, key=lambda v: abs(v - candidate))
    return best, abs(best - candidate)


def snap_depth_value(
    value: float,
    has_cm_hint: bool,
    has_mm_hint: bool,
    expected_depths_mm: Sequence[float],
) -> Tuple[Optional[float], str, float]:
    candidates: List[Tuple[str, float]] = []
    if has_mm_hint:
        candidates.append(("mm", value))
    elif has_cm_hint:
        if value > 35.0:
            candidates.append(("cm_implicit_decimal", value))
        candidates.append(("cm_x10", value * 10.0))
    else:
        candidates.append(("raw_mm", value))
        if value <= 35.0:
            candidates.append(("guess_cm_x10", value * 10.0))
        if 35.0 < value <= 300.0:
            candidates.append(("raw_mm_or_cm_digits", value))

    if expected_depths_mm:
        scored: List[Tuple[float, str, float, float]] = []
        for mode, cand in candidates:
            snapped, err = _nearest_expected(cand, expected_depths_mm)
            if snapped is not None:
                scored.append((err, mode, snapped, cand))
        if scored:
            err, mode, snapped, _ = min(scored, key=lambda x: x[0])
            return snapped, mode, err

    mode, cand = candidates[0]
    return cand, mode, float("nan")


def collect_depth_tokens(
    images: Sequence[Path],
    timeout: float,
    max_side: int,
    min_conf: float,
    expected_depths_mm: Sequence[float],
    rect_echo: Optional[Tuple[float, float, float, float]] = None,
    roi_passes: bool = False,
) -> List[DepthToken]:
    all_words: List[OCRWord] = []
    for image_index, image_path in enumerate(images):
        image_words: List[OCRWord] = []
        words, _scale = run_tesseract_tsv(image_path, timeout=timeout, max_side=max_side)
        image_words.extend(words)
        if roi_passes:
            try:
                with Image.open(image_path) as im:
                    roi_boxes = _ocr_roi_boxes(im.size[0], im.size[1], rect_echo)
            except Exception:
                roi_boxes = []
            roi_timeout = max(2.0, min(timeout, timeout * 0.65))
            # Vision works best with the complete UI context; unlike Tesseract
            # it can still localize small scale words over the echo image.
            try:
                with Image.open(image_path) as im:
                    vision_box = (0.0, 0.0, float(im.size[0]), float(im.size[1]))
                image_words.extend(run_vision_ocr_region(image_path, timeout=roi_timeout, crop_box=vision_box))
            except Exception:
                pass
            for box in roi_boxes:
                psm_modes = ["6"]
                if (box[3] - box[1]) <= max(230, int(0.22 * max(1, box[3]))):
                    psm_modes.append("7")
                if box[1] >= 0.85 * max(1, im.size[1]) and (box[2] - box[0]) <= 460:
                    psm_modes.extend(["11", "13"])
                for psm in psm_modes:
                    roi_words, _roi_scale = run_tesseract_tsv_region(
                        image_path,
                        timeout=roi_timeout,
                        max_side=max_side,
                        crop_box=box,
                        psm=psm,
                    )
                    image_words.extend(roi_words)

            # Endpoint-only recovery OCR: scale labels can lie over the echo
            # texture, where a broad ROI loses `cm` or a leading digit. These
            # compact crops use a restricted alphabet and two contrast passes.
            scale_timeout = max(1.5, min(timeout, timeout * 0.40))
            for box in _scale_endpoint_roi_boxes(image_path):
                for preprocess in ("scale_clahe", "scale_line_suppressed"):
                    for psm in ("6", "11"):
                        scale_words, _scale_factor = run_tesseract_tsv_region(
                            image_path,
                            timeout=scale_timeout,
                            max_side=max_side,
                            crop_box=box,
                            psm=psm,
                            preprocess=preprocess,
                            char_whitelist="0123456789.,cmCM",
                        )
                        image_words.extend(scale_words)
        for word in _dedupe_ocr_words(image_words):
            all_words.append(
                OCRWord(
                    image_path=word.image_path,
                    image_index=image_index,
                    text=word.text,
                    norm_text=word.norm_text,
                    conf=word.conf,
                    left=word.left,
                    top=word.top,
                    width=word.width,
                    height=word.height,
                )
            )

    words_by_image: Dict[int, List[OCRWord]] = {}
    for word in all_words:
        words_by_image.setdefault(word.image_index, []).append(word)

    tokens: List[DepthToken] = []
    for word in all_words:
        if _has_non_depth_numeric_marker(word.text):
            continue
        strong_embedded = _has_strong_embedded_depth_pattern(word.text)
        values = _numeric_values(word.text)
        # The tiny white labels printed on a scale often sit over horizontal
        # echo texture.  Tesseract can locate a clean digit there but assign a
        # near-zero confidence.  Preserve it as a *provisional* token; the
        # predictor will only use it when it lies next to a real tick lane and
        # at a scale endpoint.  Other low-confidence OCR remains discarded.
        provisional_scale_digit = bool(values) and re.fullmatch(r"\d+(?:[.,]\d+)?[-–.]?", word.text.strip()) is not None
        if word.conf < min_conf and not strong_embedded and not provisional_scale_digit:
            continue
        if not values:
            continue
        local_words = words_by_image.get(word.image_index, [])
        cm_hint = _looks_like_cm(word.text)
        mm_hint = _looks_like_mm(word.text)
        depth_hint = _looks_like_depth_hint(word.text)
        d_hint = _looks_like_d_hint(word.text)
        p_hint = _looks_like_p_hint(word.text)
        r_hint = _looks_like_r_hint(word.text)
        scale_hint = _looks_like_scale_hint(word.text)
        fps_ips_hint = _looks_like_fps_ips(word.text)
        image_height = max((other.bottom for other in local_words), default=1.0)
        column_cm_hint = False
        column_mm_hint = False
        non_depth_suffix_hint = False
        for other in local_words:
            if other is word:
                continue
            # A settings panel can place FR/Hz on the row immediately above a
            # real D value. Treat it as a forbidden suffix only on the same
            # typographic baseline, not merely in the same vertical column.
            same_suffix_line = (
                abs(other.y_center - word.y_center) <= max(12.0, 0.70 * max(other.height, word.height))
                and abs(other.x_center - word.x_center) <= 150.0
            )
            if _has_non_depth_numeric_marker(other.text) and same_suffix_line:
                non_depth_suffix_hint = True
            if _looks_like_mm(other.text) and _near_same_label_line(other, word, max_dx=130.0):
                mm_hint = True
            if _looks_like_cm(other.text) and _near_same_label_line(other, word, max_dx=130.0):
                cm_hint = True
            # A scale commonly prints the unit once (for example `0 cm`) and
            # then shows only digits next to the remaining ticks.  Propagate
            # that unit solely within the same narrow vertical scale column.
            same_scale_column = (
                abs(other.x_center - word.x_center) <= 105.0
                and abs(other.y_center - word.y_center) <= max(220.0, 0.58 * image_height)
            )
            unit_fragment = re.sub(r"[^a-z]", "", other.text.lower())
            # The compact, line-suppressed crop can recognize the literal
            # ``cm`` perfectly while assigning it confidence 0.  Within the
            # same narrow tick column that exact token is still reliable;
            # fuzzier fragments keep the normal confidence threshold.
            exact_cm = unit_fragment == "cm"
            fuzzy_cm = len(unit_fragment) <= 3 and unit_fragment.endswith("cm")
            if same_scale_column and (exact_cm or (other.conf >= 50.0 and fuzzy_cm)):
                column_cm_hint = True
            if same_scale_column and other.conf >= 75.0 and unit_fragment == "mm":
                column_mm_hint = True
            if _looks_like_depth_hint(other.text) and (
                _near_left_label(other, word, max_dx=190.0)
                or (
                    other.y_center <= word.y_center + max(18.0, 0.8 * word.height)
                    and _near_value_context(other, word, max_dx=150.0, max_dy=120.0)
                )
            ):
                depth_hint = True
            if _looks_like_d_hint(other.text) and _near_left_label(other, word, max_dx=90.0):
                d_hint = True
            if _looks_like_p_hint(other.text) and _near_left_label(other, word, max_dx=90.0):
                p_hint = True
            if _looks_like_r_hint(other.text) and _near_left_label(other, word, max_dx=90.0):
                r_hint = True
            if _looks_like_scale_hint(other.text) and _near_value_context(other, word, max_dx=240.0, max_dy=120.0):
                scale_hint = True
            if _looks_like_fps_ips(other.text) and _near_value_context(other, word, max_dx=160.0, max_dy=75.0):
                fps_ips_hint = True
        if column_cm_hint:
            cm_hint = True
            if not _looks_like_mm(word.text):
                mm_hint = False
        elif column_mm_hint:
            mm_hint = True
        if non_depth_suffix_hint:
            continue
        if fps_ips_hint and not (cm_hint or mm_hint or depth_hint or d_hint or p_hint or r_hint or scale_hint):
            continue
        for value in values:
            if not _is_plausible_depth_number(value):
                continue
            snapped, mode, err = snap_depth_value(value, cm_hint, mm_hint, expected_depths_mm)
            tokens.append(
                DepthToken(
                    word=word,
                    numeric_value=value,
                    has_cm_hint=cm_hint,
                    has_mm_hint=mm_hint,
                    has_depth_hint=depth_hint,
                    has_d_hint=d_hint,
                    has_p_hint=p_hint,
                    has_r_hint=r_hint,
                    has_scale_hint=scale_hint,
                    has_fps_ips_hint=fps_ips_hint,
                    snapped_depth_mm=snapped,
                    snap_mode=mode,
                    snap_error_mm=err,
                )
            )
    return tokens


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cluster_image_size(cluster: DepthCluster) -> Tuple[float, float]:
    if not cluster.tokens:
        return 1.0, 1.0
    try:
        with Image.open(cluster.tokens[0].word.image_path) as im:
            return float(im.size[0]), float(im.size[1])
    except Exception:
        return 1.0, 1.0


def _cluster_geometry_scores(
    cluster: DepthCluster,
    rect_echo: Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float, float]:
    """Score the UI geometry prior: depth is an accessory, usually near a side."""
    w, h = _cluster_image_size(cluster)
    x = cluster.x_center
    y = cluster.y_center
    side_score = _clamp01(abs((x / max(1.0, w)) - 0.5) * 2.0)
    accessory_score = side_score
    center_penalty = 0.0

    if rect_echo is not None:
        left, top, right, bottom = rect_echo
        rw = max(1.0, right - left)
        rh = max(1.0, bottom - top)
        inside_x = left <= x <= right
        inside_y = top <= y <= bottom
        inside_echo = inside_x and inside_y
        echo_x_norm = _clamp01((x - left) / rw)
        echo_y_norm = _clamp01((y - top) / rh)
        echo_side_score = _clamp01(abs(echo_x_norm - 0.5) * 2.0)

        if inside_echo:
            accessory_score = max(side_score, 0.2 + 0.8 * echo_side_score)
            central_x = 0.22 <= echo_x_norm <= 0.78
            central_y = 0.12 <= echo_y_norm <= 0.88
            if central_x and central_y:
                center_penalty = 1.0 - echo_side_score
        else:
            # Outside the ultrasound rectangle is accessory space, but a top/bottom
            # header centered over the echo is weaker than a true side overlay.
            if not inside_x:
                outside_score = 1.0
            else:
                outside_score = 0.45 + 0.45 * echo_side_score + 0.10 * side_score
            accessory_score = max(side_score, _clamp01(outside_score))

    return float(side_score), float(accessory_score), float(center_penalty)


def cluster_tokens(
    tokens: Sequence[DepthToken],
    bin_px: float,
    image_count: int,
    expected_depths_mm: Sequence[float],
    rect_echo: Optional[Tuple[float, float, float, float]] = None,
) -> List[DepthCluster]:
    clusters: List[DepthCluster] = []
    for token in sorted(tokens, key=lambda t: (t.word.y_center, t.word.x_center)):
        best_idx = -1
        best_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            dx = token.word.x_center - cluster.x_center
            dy = token.word.y_center - cluster.y_center
            dist = math.hypot(dx, dy)
            if dist < best_dist and abs(dx) <= bin_px and abs(dy) <= bin_px:
                best_idx = idx
                best_dist = dist
        if best_idx < 0:
            clusters.append(DepthCluster(cluster_id=len(clusters), tokens=[token]))
        else:
            clusters[best_idx].tokens.append(token)

    scored: List[DepthCluster] = []
    for idx, cluster in enumerate(clusters):
        cluster.cluster_id = idx
        if not cluster.tokens:
            continue
        support = len(cluster.images) / float(max(1, image_count))
        cm_ratio = sum(1 for t in cluster.tokens if t.has_cm_hint) / float(len(cluster.tokens))
        mm_ratio = sum(1 for t in cluster.tokens if t.has_mm_hint) / float(len(cluster.tokens))
        depth_hint_ratio = sum(1 for t in cluster.tokens if t.has_depth_hint) / float(len(cluster.tokens))
        d_hint_ratio = sum(1 for t in cluster.tokens if t.has_d_hint) / float(len(cluster.tokens))
        p_hint_ratio = sum(1 for t in cluster.tokens if t.has_p_hint) / float(len(cluster.tokens))
        r_hint_ratio = sum(1 for t in cluster.tokens if t.has_r_hint) / float(len(cluster.tokens))
        scale_hint_ratio = sum(1 for t in cluster.tokens if t.has_scale_hint) / float(len(cluster.tokens))
        fps_ips_ratio = sum(1 for t in cluster.tokens if t.has_fps_ips_hint) / float(len(cluster.tokens))
        plausible = sum(1 for t in cluster.tokens if _is_plausible_depth_number(t.numeric_value)) / float(len(cluster.tokens))
        values = {round(float(t.snapped_depth_mm or t.numeric_value), 1) for t in cluster.tokens}
        unique_values = len(values)
        expected_coverage = 0.0
        if expected_depths_mm:
            expected_good = sum(
                1
                for t in cluster.tokens
                if math.isfinite(t.snap_error_mm) and t.snap_error_mm <= max(2.0, 0.08 * float(t.snapped_depth_mm or 1.0))
            )
            expected_ratio = expected_good / float(len(cluster.tokens))
        else:
            expected_ratio = 0.0

        side_score, accessory_score, center_penalty = _cluster_geometry_scores(cluster, rect_echo)
        variability = min(1.0, unique_values / 4.0)
        unit_ratio = max(cm_ratio, mm_ratio)
        text_hint_ratio = max(depth_hint_ratio, d_hint_ratio, 0.80 * p_hint_ratio, 0.80 * r_hint_ratio, 0.65 * scale_hint_ratio)
        ui_hint_ratio = max(unit_ratio, text_hint_ratio, scale_hint_ratio)
        if expected_depths_mm:
            expected_set = {round(float(v), 1) for v in expected_depths_mm if math.isfinite(float(v))}
            expected_seen = {
                round(float(t.snapped_depth_mm), 1)
                for t in cluster.tokens
                if t.snapped_depth_mm is not None and math.isfinite(float(t.snapped_depth_mm))
            }
            expected_coverage = len(expected_seen.intersection(expected_set)) / float(max(1, min(len(expected_set), image_count)))
            score = (
                2.25 * text_hint_ratio
                + 1.45 * unit_ratio
                + 1.45 * accessory_score
                + 1.75 * expected_coverage
                + 1.10 * expected_ratio
                + 0.55 * support
                + 0.35 * plausible
                + 0.35 * variability
            )
        else:
            score = (
                2.45 * text_hint_ratio
                + 1.85 * unit_ratio
                + 1.50 * accessory_score
                + 0.75 * support
                + 0.45 * plausible
                + 0.40 * variability
            )
        if ui_hint_ratio < 0.2:
            score -= 0.50
        score -= 2.25 * center_penalty
        if accessory_score < 0.35 and ui_hint_ratio < 0.55:
            score -= 0.75
        if expected_depths_mm and expected_ratio < 0.25 and expected_coverage < 0.25 and ui_hint_ratio < 0.55:
            score -= 0.60
        if len(cluster.tokens) == 1 and image_count > 1:
            score -= 0.8

        cluster.score = float(score)
        cluster.cm_ratio = float(cm_ratio)
        cluster.mm_ratio = float(mm_ratio)
        cluster.expected_ratio = float(expected_ratio)
        cluster.plausible_ratio = float(plausible)
        cluster.support_ratio = float(support)
        cluster.depth_hint_ratio = float(depth_hint_ratio)
        cluster.d_hint_ratio = float(d_hint_ratio)
        cluster.p_hint_ratio = float(p_hint_ratio)
        cluster.r_hint_ratio = float(r_hint_ratio)
        cluster.scale_hint_ratio = float(scale_hint_ratio)
        cluster.fps_ips_ratio = float(fps_ips_ratio)
        cluster.side_score = float(side_score)
        cluster.accessory_score = float(accessory_score)
        cluster.echo_center_penalty = float(center_penalty)
        cluster.unique_values = int(unique_values)
        cluster.reason = (
            f"support={support:.2f}; cm={cm_ratio:.2f}; mm={mm_ratio:.2f}; "
            f"depth={depth_hint_ratio:.2f}; D={d_hint_ratio:.2f}; P={p_hint_ratio:.2f}; R={r_hint_ratio:.2f}; "
            f"scale={scale_hint_ratio:.2f}; fps_ips={fps_ips_ratio:.2f}; "
            f"expected={expected_ratio:.2f}; "
            f"coverage={expected_coverage:.2f}; side={side_score:.2f}; "
            f"accessory={accessory_score:.2f}; center_penalty={center_penalty:.2f}; "
            f"unique_values={unique_values}"
        )
        scored.append(cluster)
    return sorted(scored, key=lambda c: c.score, reverse=True)


def _load_expected_depths_from_fss(path: Optional[str]) -> List[float]:
    if not path:
        return []
    fss_path = Path(path).expanduser().resolve()
    if not fss_path.exists():
        return []
    try:
        _checks, depths = parse_rect_depth_checks(fss_path)
        return depths
    except Exception:
        return []


def write_candidates_csv(clusters: Sequence[DepthCluster], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "cluster_id",
        "score",
        "left",
        "top",
        "right",
        "bottom",
        "x_center",
        "y_center",
        "token_count",
        "image_support",
        "unique_values",
        "cm_ratio",
        "mm_ratio",
        "depth_hint_ratio",
        "d_hint_ratio",
        "p_hint_ratio",
        "r_hint_ratio",
        "scale_hint_ratio",
        "side_score",
        "accessory_score",
        "echo_center_penalty",
        "expected_ratio",
        "plausible_ratio",
        "reason",
        "values_preview",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, cluster in enumerate(clusters, start=1):
            values = sorted({round(float(t.snapped_depth_mm or t.numeric_value), 1) for t in cluster.tokens})
            writer.writerow(
                {
                    "rank": rank,
                    "cluster_id": cluster.cluster_id,
                    "score": f"{cluster.score:.4f}",
                    "left": f"{cluster.left:.1f}",
                    "top": f"{cluster.top:.1f}",
                    "right": f"{cluster.right:.1f}",
                    "bottom": f"{cluster.bottom:.1f}",
                    "x_center": f"{cluster.x_center:.1f}",
                    "y_center": f"{cluster.y_center:.1f}",
                    "token_count": len(cluster.tokens),
                    "image_support": len(cluster.images),
                    "unique_values": cluster.unique_values,
                    "cm_ratio": f"{cluster.cm_ratio:.3f}",
                    "mm_ratio": f"{cluster.mm_ratio:.3f}",
                    "depth_hint_ratio": f"{cluster.depth_hint_ratio:.3f}",
                    "d_hint_ratio": f"{cluster.d_hint_ratio:.3f}",
                    "p_hint_ratio": f"{cluster.p_hint_ratio:.3f}",
                    "r_hint_ratio": f"{cluster.r_hint_ratio:.3f}",
                    "scale_hint_ratio": f"{cluster.scale_hint_ratio:.3f}",
                    "side_score": f"{cluster.side_score:.3f}",
                    "accessory_score": f"{cluster.accessory_score:.3f}",
                    "echo_center_penalty": f"{cluster.echo_center_penalty:.3f}",
                    "expected_ratio": f"{cluster.expected_ratio:.3f}",
                    "plausible_ratio": f"{cluster.plausible_ratio:.3f}",
                    "reason": cluster.reason,
                    "values_preview": "|".join(str(v) for v in values[:20]),
                }
            )


def write_predictions_csv(best: Optional[DepthCluster], images: Sequence[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "prediction_ok",
        "depth_mm",
        "raw_text",
        "raw_numeric_value",
        "snap_mode",
        "snap_error_mm",
        "ocr_conf",
        "has_cm_hint",
        "has_mm_hint",
        "has_depth_hint",
        "has_d_hint",
        "has_p_hint",
        "has_r_hint",
        "has_scale_hint",
        "left",
        "top",
        "right",
        "bottom",
        "cluster_id",
        "cluster_score",
    ]
    by_image: Dict[int, DepthToken] = {}
    if best:
        for token in sorted(best.tokens, key=lambda t: t.word.conf, reverse=True):
            by_image.setdefault(token.word.image_index, token)

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, image_path in enumerate(images):
            token = by_image.get(idx)
            if token is None:
                writer.writerow(
                    {
                        "image_path": image_path.as_posix(),
                        "prediction_ok": 0,
                        "depth_mm": "",
                        "raw_text": "",
                        "raw_numeric_value": "",
                        "snap_mode": "",
                        "snap_error_mm": "",
                        "ocr_conf": "",
                        "has_cm_hint": "",
                        "has_mm_hint": "",
                        "has_depth_hint": "",
                        "has_d_hint": "",
                        "has_p_hint": "",
                        "has_r_hint": "",
                        "has_scale_hint": "",
                        "left": "",
                        "top": "",
                        "right": "",
                        "bottom": "",
                        "cluster_id": best.cluster_id if best else "",
                        "cluster_score": f"{best.score:.4f}" if best else "",
                    }
                )
                continue
            writer.writerow(
                {
                    "image_path": image_path.as_posix(),
                    "prediction_ok": 1,
                    "depth_mm": f"{float(token.snapped_depth_mm or 0.0):.3f}",
                    "raw_text": token.word.text,
                    "raw_numeric_value": f"{token.numeric_value:.3f}",
                    "snap_mode": token.snap_mode,
                    "snap_error_mm": f"{token.snap_error_mm:.3f}" if math.isfinite(token.snap_error_mm) else "",
                    "ocr_conf": f"{token.word.conf:.2f}",
                    "has_cm_hint": int(token.has_cm_hint),
                    "has_mm_hint": int(token.has_mm_hint),
                    "has_depth_hint": int(token.has_depth_hint),
                    "has_d_hint": int(token.has_d_hint),
                    "has_p_hint": int(token.has_p_hint),
                    "has_r_hint": int(token.has_r_hint),
                    "has_scale_hint": int(token.has_scale_hint),
                    "left": f"{token.word.left:.1f}",
                    "top": f"{token.word.top:.1f}",
                    "right": f"{token.word.right:.1f}",
                    "bottom": f"{token.word.bottom:.1f}",
                    "cluster_id": best.cluster_id,
                    "cluster_score": f"{best.score:.4f}",
                }
            )


def predict_folder(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    image_dir = folder / "image_samples" if (folder / "image_samples").is_dir() else folder
    images = iter_images(image_dir)
    if args.max_images and len(images) > args.max_images:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    expected_depths = _load_expected_depths_from_fss(args.fss)
    fss_path = Path(args.fss).expanduser().resolve() if args.fss else None
    rect_echo = parse_rect_echo(fss_path) if fss_path and fss_path.exists() else None
    tokens = collect_depth_tokens(
        images=images,
        timeout=float(args.ocr_timeout),
        max_side=int(args.ocr_max_side),
        min_conf=float(args.min_ocr_conf),
        expected_depths_mm=expected_depths,
    )
    clusters = cluster_tokens(
        tokens=tokens,
        bin_px=float(args.cluster_bin_px),
        image_count=len(images),
        expected_depths_mm=expected_depths,
        rect_echo=rect_echo,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = output_dir / "rect_depth_candidates.csv"
    predictions_csv = output_dir / "rect_depth_predictions.csv"
    write_candidates_csv(clusters, candidates_csv)
    best = clusters[0] if clusters else None
    write_predictions_csv(best, images, predictions_csv)

    summary = {
        "folder": folder.as_posix(),
        "image_count": len(images),
        "token_count": len(tokens),
        "candidate_count": len(clusters),
        "expected_depths_mm": expected_depths,
        "rect_echo": list(rect_echo) if rect_echo else None,
        "best_cluster": None
        if best is None
        else {
            "cluster_id": best.cluster_id,
            "score": best.score,
            "rect": [best.left, best.top, best.right, best.bottom],
            "reason": best.reason,
            "cm_ratio": best.cm_ratio,
            "mm_ratio": best.mm_ratio,
            "depth_hint_ratio": best.depth_hint_ratio,
            "d_hint_ratio": best.d_hint_ratio,
            "p_hint_ratio": best.p_hint_ratio,
            "side_score": best.side_score,
            "accessory_score": best.accessory_score,
            "echo_center_penalty": best.echo_center_penalty,
        },
        "candidates_csv": candidates_csv.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run a targeted hybrid/classical RECT_DEPTH module."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Parse legacy .fss RECT_DEPTH rows into CSV.")
    p_manifest.add_argument("--dataset-root", required=True, help="Root containing config folders with DB_setup + image_samples.")
    p_manifest.add_argument("--output-csv", required=True, help="Output manifest CSV.")
    p_manifest.add_argument("--crops-dir", default="", help="Optional directory for cropped RECT_DEPTH examples.")
    p_manifest.add_argument("--template-root", default="", help="Optional fallback DB_echo root for depth_*.png templates.")
    p_manifest.add_argument("--max-configs", type=int, default=0, help="Optional cap for smoke tests.")
    p_manifest.set_defaults(func=build_manifest)

    p_predict = sub.add_parser("predict", help="Folder-level OCR baseline for depth rect/value candidates.")
    p_predict.add_argument("--folder", required=True, help="Config folder or image_samples folder.")
    p_predict.add_argument("--output-dir", required=True, help="Directory for candidates/predictions CSV.")
    p_predict.add_argument("--fss", default="", help="Optional legacy .fss to snap OCR values to known depths.")
    p_predict.add_argument("--max-images", type=int, default=80, help="Max images to OCR from the folder.")
    p_predict.add_argument("--ocr-timeout", type=float, default=12.0, help="Tesseract timeout per frame.")
    p_predict.add_argument("--ocr-max-side", type=int, default=1800, help="Resize OCR frame to this max side.")
    p_predict.add_argument("--min-ocr-conf", type=float, default=15.0, help="Minimum OCR token confidence.")
    p_predict.add_argument("--cluster-bin-px", type=float, default=42.0, help="Spatial radius for recurring-token clusters.")
    p_predict.set_defaults(func=predict_folder)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
