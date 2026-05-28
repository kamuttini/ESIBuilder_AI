#!/usr/bin/env python3
"""Build LR-marker localization datasets driven by the SU/GIU classifier.

This version intentionally does not assign semantic meaning to
DB_echo/setup_*/orientation_0..3.png. Those files are treated as equivalent
copies/variants of the same vendor/configuration marker.

Pipeline per image:
1. crop the ultrasound rectangle from the .fss RECT_ECHO line;
2. classify the crop with the trained SU/GIU network;
3. search the DB_echo orientation template in the predicted top/bottom half;
4. if the match is weak, search the whole echo crop to recover SU/GIU mistakes;
5. if the match is still weak, progressively expand the search area outside the
   echo rectangle to recover rare markers just outside the crop;
6. derive the LR side from the detected marker x position.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ORIENTATION_NAMES = {0: "NF", 1: "LR", 2: "UD", 3: "LRUD"}
LABELS = ("su", "giu")
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
INT_RE = re.compile(r"^-?\d+$")
TEMPLATE_RE = re.compile(r"^orientation_[0-3]\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE)

IMAGE_PATTERNS = {
    "depth_find_nf": re.compile(r"^image_depth_find_no_flip_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lr": re.compile(r"^image_depth_find_flip_lr_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_ud": re.compile(r"^image_depth_find_flip_ud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lrud": re.compile(r"^image_depth_find_flip_lrud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
}

ORIENTATION_DIR_PATTERNS: Tuple[Tuple[int, re.Pattern[str]], ...] = (
    (3, re.compile(r"\b(lrud|fliplrud|flip\s*lr\s*ud|fliplr\s*ud|lr\s*ud)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(lr|fliplr|flip\s*lr|left\s*right)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(ud|flipud|flip\s*ud|upside\s*down)\b", re.IGNORECASE)),
    (0, re.compile(r"\b(nf|noflip|no\s*flip|flip\s*no)\b", re.IGNORECASE)),
)

SKIP_DIRS = {
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    ".DocumentRevisions-V100",
    "__pycache__",
}


@dataclass(frozen=True)
class ConfigRecord:
    folder_path: Path
    folder_name: str
    dataset_root: Path
    manufacturer: str
    fss_path: Path
    setup_token: str
    echo_rect: Tuple[int, int, int, int]
    image_paths: Tuple[Path, ...]
    template_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class TemplateRow:
    path: Path
    width: int
    height: int
    mean: float
    std: float
    tensor_zero_mean: torch.Tensor
    tensor_ones: torch.Tensor
    tensor_norm: float


@dataclass(frozen=True)
class MatchResult:
    score: float
    x: int
    y: int
    template: TemplateRow


@dataclass(frozen=True)
class LocatedMatch:
    score: float
    template: TemplateRow
    marker_top_abs: int
    marker_left_abs: int
    marker_bottom_abs: int
    marker_right_abs: int
    marker_top_crop: int
    marker_left_crop: int
    marker_bottom_crop: int
    marker_right_crop: int
    search_strategy: str
    search_scope: str
    search_margin_px: int
    search_rect_abs: Tuple[int, int, int, int]


class BinaryOrientationClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def _choose_device(requested: Optional[str]) -> torch.device:
    if requested:
        value = requested.strip().lower()
        if value == "cpu":
            return torch.device("cpu")
        if value == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if value.startswith("cuda") and torch.cuda.is_available():
            return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def infer_manufacturer(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    name = re.sub(r"^\d+[.\s_-]*", "", name)
    name = re.sub(r"^[a-z][\s._-]+", "", name)
    if re.match(r"^bk(?:[\s,_-]|\d)", name):
        return "BK"
    for vendor in (
        "esaote",
        "hitachi",
        "ge",
        "mindray",
        "canon",
        "philips",
        "toshiba",
        "siemens",
        "koelis",
        "biopsee",
        "terason",
        "sonostar",
        "exactvu",
        "alpinion",
        "vinno",
    ):
        if name.startswith(vendor):
            return "ExactVu" if vendor == "exactvu" else vendor.capitalize()
    token = re.split(r"[\s,_-]+", folder_name.strip())[0]
    return token if token else "UNKNOWN"


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    cleaned = value.strip()
    if INT_RE.match(cleaned):
        return int(cleaned)
    try:
        return int(float(cleaned))
    except ValueError:
        return default


def _extract_rect_line(lines: Sequence[str]) -> int:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines) and RECT_RE.match(lines[candidate - 1].strip()):
            return candidate
    for idx in range(1, min(len(lines), 20) + 1):
        if RECT_RE.match(lines[idx - 1].strip()):
            return idx
    raise ValueError("RECT_ECHO line not found")


def _parse_rect_line(lines: Sequence[str], rect_line: int) -> Tuple[int, int, int, int]:
    match = RECT_RE.match(lines[rect_line - 1].strip())
    if not match:
        raise ValueError("RECT_ECHO line malformed")
    return tuple(int(match.group(i)) for i in range(1, 5))  # type: ignore[return-value]


def _clip_rect(rect: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    top, left, bottom, right = rect
    top = max(0, min(height - 1, top))
    left = max(0, min(width - 1, left))
    bottom = max(0, min(height - 1, bottom))
    right = max(0, min(width - 1, right))
    if bottom <= top or right <= left:
        return None
    return top, left, bottom, right


def _expand_rect(rect: Tuple[int, int, int, int], width: int, height: int, margin_px: int) -> Optional[Tuple[int, int, int, int]]:
    top, left, bottom, right = rect
    return _clip_rect(
        (top - margin_px, left - margin_px, bottom + margin_px, right + margin_px),
        width=width,
        height=height,
    )


def _match_image_name(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    for source_type, regex in IMAGE_PATTERNS.items():
        m = regex.match(name)
        if not m:
            continue
        sample_idx = int(m.group(1))
        if source_type == "depth_find_nf":
            return source_type, 0, sample_idx
        if source_type == "depth_find_lr":
            return source_type, 1, sample_idx
        if source_type == "depth_find_ud":
            return source_type, 2, sample_idx
        if source_type == "depth_find_lrud":
            return source_type, 3, sample_idx
    return "other", None, None


def _infer_orientation_from_component(component: str) -> Optional[int]:
    normalized = re.sub(r"[^a-z0-9]+", " ", _normalize_name(component)).strip()
    if not normalized:
        return None
    for orientation_idx, regex in ORIENTATION_DIR_PATTERNS:
        if regex.search(normalized):
            return orientation_idx
    return None


def _infer_orientation_from_path(path: Path) -> Optional[int]:
    for part in reversed(path.parts[:-1]):
        orientation_idx = _infer_orientation_from_component(part)
        if orientation_idx is not None:
            return orientation_idx
    return None


def _orientation_hint(image_path: Path, config_folder: Path) -> Tuple[str, Optional[int], Optional[int]]:
    source_type, orientation_idx, sample_idx = _match_image_name(image_path.name)
    if orientation_idx is None:
        try:
            rel = image_path.relative_to(config_folder)
        except ValueError:
            rel = image_path
        orientation_idx = _infer_orientation_from_path(rel)
        if orientation_idx is not None and source_type == "other":
            source_type = "path_orientation_folder"
    return source_type, orientation_idx, sample_idx


def _filename_lr_hint(orientation_idx: Optional[int]) -> Tuple[str, str, str]:
    if orientation_idx in (1, 3):
        return "right", "lr_flipped", "specchiata_a_destra"
    if orientation_idx in (0, 2):
        return "left", "not_lr_flipped", "normale"
    return "", "", ""


def _lr_from_detected_side(side: str) -> Tuple[str, str, int]:
    if side == "right":
        return "lr_flipped", "specchiata_a_destra", 1
    if side == "left":
        return "not_lr_flipped", "normale", 0
    return "", "", -1


def _filename_vertical_label(orientation_idx: Optional[int]) -> str:
    if orientation_idx in (0, 1):
        return "su"
    if orientation_idx in (2, 3):
        return "giu"
    return ""


def _iter_config_folders(dataset_roots: Sequence[Path], max_depth: int, max_configs: int) -> Tuple[List[Path], List[str]]:
    folders: List[Path] = []
    warnings: List[str] = []
    seen: set[str] = set()
    for root in dataset_roots:
        if not root.is_dir():
            warnings.append(f"Dataset root missing: {root}")
            continue

        def _onerror(exc: OSError) -> None:
            warnings.append(f"Cannot scan {getattr(exc, 'filename', root)}: {exc}")

        root_depth = len(root.resolve().parts)
        for dirpath, dirnames, _filenames in os.walk(root, onerror=_onerror):
            depth = max(0, len(Path(dirpath).resolve().parts) - root_depth)
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
            if max_depth >= 0 and depth > max_depth:
                dirnames[:] = []
                continue
            names = set(dirnames)
            if {"DB_setup", "DB_echo", "image_samples"}.issubset(names):
                folder = Path(dirpath).resolve()
                key = folder.as_posix()
                if key not in seen:
                    folders.append(folder)
                    seen.add(key)
                    if max_configs > 0 and len(folders) >= max_configs:
                        return sorted(folders, key=lambda p: p.as_posix().lower()), warnings
                dirnames[:] = [name for name in dirnames if name not in {"DB_setup", "DB_echo", "image_samples"}]
    return sorted(folders, key=lambda p: p.as_posix().lower()), warnings


def _find_dataset_root(folder: Path, roots: Sequence[Path]) -> Path:
    resolved = folder.resolve()
    for root in sorted((p.resolve() for p in roots), key=lambda p: len(p.parts), reverse=True):
        try:
            resolved.relative_to(root)
            return root
        except ValueError:
            continue
    return resolved.parent


def _collect_templates(folder: Path, setup_token: str) -> Tuple[Path, ...]:
    echo_dir = folder / "DB_echo"
    preferred = echo_dir / f"setup_{setup_token}"
    paths: List[Path] = []
    if preferred.is_dir():
        paths.extend(sorted(p.resolve() for p in preferred.iterdir() if p.is_file() and TEMPLATE_RE.match(p.name)))
    for path in sorted(echo_dir.rglob("*")):
        if path.is_file() and TEMPLATE_RE.match(path.name):
            resolved = path.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return tuple(paths)


def _collect_images(image_dir: Path, config_folder: Path) -> Tuple[Path, ...]:
    out: List[Path] = []
    for path in sorted(p for p in image_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        _source_type, orientation_idx, _sample_idx = _orientation_hint(path, config_folder)
        if orientation_idx in (0, 1, 2, 3):
            out.append(path.resolve())
    return tuple(out)


def _scan_configs(
    dataset_roots: Sequence[Path],
    vendor_filter: str,
    exclude_vendors: Sequence[str],
    max_depth: int,
    max_configs: int,
) -> Tuple[List[ConfigRecord], List[str]]:
    folders, warnings = _iter_config_folders(dataset_roots, max_depth=max_depth, max_configs=max_configs)
    records: List[ConfigRecord] = []
    vendor_filter_norm = vendor_filter.strip().lower()
    excluded_vendor_norms = {value.strip().lower() for value in exclude_vendors if value.strip()}
    for folder in folders:
        manufacturer = infer_manufacturer(folder.name)
        if vendor_filter_norm and manufacturer.lower() != vendor_filter_norm:
            continue
        if manufacturer.lower() in excluded_vendor_norms:
            warnings.append(f"{folder}: vendor {manufacturer} excluded by --exclude-vendors")
            continue
        setup_dir = folder / "DB_setup"
        fss_files = sorted(p for p in setup_dir.glob("*.fss") if p.is_file())
        if len(fss_files) != 1:
            warnings.append(f"{folder}: expected 1 .fss, found {len(fss_files)}")
            continue
        fss_path = fss_files[0]
        try:
            lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
            rect_line = _extract_rect_line(lines)
            echo_rect = _parse_rect_line(lines, rect_line)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"{folder}: cannot parse {fss_path.name}: {exc}")
            continue
        setup_token = fss_path.stem.split("_")[-1]
        templates = _collect_templates(folder, setup_token)
        if not templates:
            warnings.append(f"{folder}: no orientation templates in DB_echo")
            continue
        images = _collect_images(folder / "image_samples", folder)
        if not images:
            warnings.append(f"{folder}: no labeled image_samples")
            continue
        records.append(
            ConfigRecord(
                folder_path=folder.resolve(),
                folder_name=folder.name,
                dataset_root=_find_dataset_root(folder, dataset_roots),
                manufacturer=manufacturer,
                fss_path=fss_path.resolve(),
                setup_token=setup_token,
                echo_rect=echo_rect,
                image_paths=images,
                template_paths=templates,
            )
        )
    return records, warnings


def _load_sugiu_model(checkpoint: Path, device: torch.device) -> Tuple[BinaryOrientationClassifier, int]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = BinaryOrientationClassifier().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    image_size = int(ckpt.get("args", {}).get("image_size", 256))
    return model, image_size


def _preprocess_crop(crop: Image.Image, image_size: int) -> torch.Tensor:
    image = crop.convert("RGB")
    image = TF.resize(image, size=[image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    x = TF.to_tensor(image)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def _predict_sugiu(model: BinaryOrientationClassifier, crop: Image.Image, image_size: int, device: torch.device) -> Tuple[str, float, float, float]:
    x = _preprocess_crop(crop, image_size=image_size).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], float(probs[pred_idx]), float(probs[0]), float(probs[1])


def _load_template(path: Path, device: torch.device, blank_template_max_value: int) -> Optional[TemplateRow]:
    try:
        with Image.open(path) as img:
            gray_img = img.convert("L")
            extrema = gray_img.getextrema()
            if extrema is not None and int(extrema[1]) <= int(blank_template_max_value):
                return None
            arr = np.asarray(gray_img, dtype=np.float32) / 255.0
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if width < 4 or height < 4:
        return None
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return None
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    zero_mean = tensor - tensor.mean()
    norm = float(torch.sqrt(torch.clamp((zero_mean * zero_mean).sum(), min=1e-12)).item())
    if norm < 1e-6:
        return None
    return TemplateRow(
        path=path.resolve(),
        width=width,
        height=height,
        mean=mean,
        std=std,
        tensor_zero_mean=zero_mean,
        tensor_ones=torch.ones_like(zero_mean),
        tensor_norm=norm,
    )


def _ncc_best_match(search_img: torch.Tensor, template: TemplateRow) -> Optional[Tuple[float, int, int]]:
    height, width = int(search_img.shape[0]), int(search_img.shape[1])
    if height < template.height or width < template.width:
        return None
    x = search_img.unsqueeze(0).unsqueeze(0)
    n = float(template.width * template.height)
    numerator = F.conv2d(x, template.tensor_zero_mean)
    sum_x = F.conv2d(x, template.tensor_ones)
    sum_x2 = F.conv2d(x * x, template.tensor_ones)
    var_x = torch.clamp(sum_x2 - (sum_x * sum_x) / n, min=0.0)
    denominator = torch.sqrt(var_x) * template.tensor_norm
    valid = denominator > 1e-6
    ncc = torch.where(valid, numerator / (denominator + 1e-8), torch.full_like(numerator, -1.0))
    ncc = torch.nan_to_num(torch.clamp(ncc, min=-1.0, max=1.0), nan=-1.0, posinf=-1.0, neginf=-1.0)
    flat = ncc.view(-1)
    best_val, best_idx = torch.max(flat, dim=0)
    idx = int(best_idx.item())
    out_w = int(ncc.shape[-1])
    return float(best_val.item()), idx % out_w, idx // out_w


def _find_best_match(search_img: torch.Tensor, templates: Sequence[TemplateRow]) -> Optional[MatchResult]:
    best: Optional[MatchResult] = None
    for template in templates:
        match = _ncc_best_match(search_img, template)
        if match is None:
            continue
        score, x, y = match
        if best is None or score > best.score:
            best = MatchResult(score=score, x=x, y=y, template=template)
    return best


def _score_of(located: Optional[LocatedMatch]) -> float:
    return located.score if located is not None else -1.0


def _locate_match_in_abs_rect(
    gray_full: torch.Tensor,
    search_rect_abs: Tuple[int, int, int, int],
    echo_rect_abs: Tuple[int, int, int, int],
    templates: Sequence[TemplateRow],
    search_strategy: str,
    search_scope: str,
    search_margin_px: int,
) -> Optional[LocatedMatch]:
    search_top, search_left, search_bottom, search_right = search_rect_abs
    search_img = gray_full[search_top : search_bottom + 1, search_left : search_right + 1]
    best = _find_best_match(search_img, templates)
    if best is None:
        return None
    echo_top, echo_left, _echo_bottom, _echo_right = echo_rect_abs
    marker_top_abs = search_top + best.y
    marker_left_abs = search_left + best.x
    marker_bottom_abs = marker_top_abs + best.template.height - 1
    marker_right_abs = marker_left_abs + best.template.width - 1
    return LocatedMatch(
        score=best.score,
        template=best.template,
        marker_top_abs=marker_top_abs,
        marker_left_abs=marker_left_abs,
        marker_bottom_abs=marker_bottom_abs,
        marker_right_abs=marker_right_abs,
        marker_top_crop=marker_top_abs - echo_top,
        marker_left_crop=marker_left_abs - echo_left,
        marker_bottom_crop=marker_bottom_abs - echo_top,
        marker_right_crop=marker_right_abs - echo_left,
        search_strategy=search_strategy,
        search_scope=search_scope,
        search_margin_px=search_margin_px,
        search_rect_abs=search_rect_abs,
    )


def _parse_float_steps(value: str) -> Tuple[float, ...]:
    out: List[float] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            step = float(text)
        except ValueError:
            continue
        if step > 0:
            out.append(step)
    return tuple(out)


def _roi_from_sugiu(rect_width: int, rect_height: int, label: str) -> Tuple[int, int, int, int]:
    cut = rect_height // 2
    if label == "su":
        return 0, 0, max(0, cut - 1), rect_width - 1
    return min(rect_height - 1, cut), 0, rect_height - 1, rect_width - 1


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _split_by_config(records: Sequence[ConfigRecord], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    by_vendor: Dict[str, List[ConfigRecord]] = {}
    for record in records:
        by_vendor.setdefault(record.manufacturer, []).append(record)
    out: Dict[str, str] = {}
    for _vendor, group in sorted(by_vendor.items()):
        items = list(group)
        rng.shuffle(items)
        items.sort(key=lambda r: (-len(r.image_paths), r.folder_name.lower()))
        n = len(items)
        if n == 1:
            out[items[0].folder_path.as_posix()] = "train"
            continue
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15))) if n >= 3 else 1
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        idx = 0
        for _ in range(max(0, n_train)):
            out[items[idx].folder_path.as_posix()] = "train"
            idx += 1
        for _ in range(max(0, n_val)):
            if idx < n:
                out[items[idx].folder_path.as_posix()] = "val"
                idx += 1
        while idx < n:
            out[items[idx].folder_path.as_posix()] = "test"
            idx += 1
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: Iterable[Dict[str, object]], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, "") or "(empty)")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _write_by_vendor(output_dir: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> Dict[str, object]:
    by_vendor: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_vendor.setdefault(str(row.get("manufacturer", "") or "UNKNOWN"), []).append(row)
    vendor_summary: Dict[str, object] = {}
    base = output_dir / "by_vendor"
    for vendor, vendor_rows in sorted(by_vendor.items()):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", vendor).strip("_") or "UNKNOWN"
        vendor_dir = base / safe
        candidates = [r for r in vendor_rows if str(r.get("label_kind", "")) == "auto_candidate"]
        review = [r for r in vendor_rows if str(r.get("label_kind", "")) != "auto_candidate"]
        _write_csv(vendor_dir / "manifest.csv", vendor_rows, fields)
        _write_csv(vendor_dir / "auto_candidates.csv", candidates, fields)
        _write_csv(vendor_dir / "review_queue.csv", review, fields)
        summary = {
            "vendor": vendor,
            "rows_total": len(vendor_rows),
            "auto_candidates": len(candidates),
            "review_rows": len(review),
            "label_kind_counts": _counts(vendor_rows, "label_kind"),
            "su_giu_pred_counts": _counts(vendor_rows, "su_giu_pred"),
            "filename_vertical_counts": _counts(vendor_rows, "filename_vertical_label"),
            "lr_label_counts": _counts(vendor_rows, "lr_label"),
            "search_strategy_counts": _counts(vendor_rows, "search_strategy"),
            "split_counts": _counts(vendor_rows, "split"),
            "config_count": len({str(r.get("config_folder", "")) for r in vendor_rows}),
        }
        vendor_summary[vendor] = summary
        (vendor_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return vendor_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SU/GIU-driven LR marker dataset.")
    parser.add_argument("--dataset-roots", type=Path, nargs="+", default=[Path("/Volumes/SSD_esi1_n1")])
    parser.add_argument(
        "--sugiu-checkpoint",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_ssd_esi1_n1"),
    )
    parser.add_argument("--vendor-filter", type=str, default="")
    parser.add_argument(
        "--exclude-vendors",
        nargs="*",
        default=["Biopsee"],
        help="Vendors excluded by default from LR marker dataset preparation.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-config-depth", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--max-rows-per-sugiu-pred",
        type=int,
        default=0,
        help="Keep up to N rows for each predicted SU/GIU class; useful for balanced review batches.",
    )
    parser.add_argument(
        "--max-evaluated",
        type=int,
        default=0,
        help="Optional safety cap on evaluated images when balancing by predicted SU/GIU.",
    )
    parser.add_argument("--min-match-score", type=float, default=0.45)
    parser.add_argument(
        "--full-crop-fallback-threshold",
        type=float,
        default=0.55,
        help="If predicted-half template score is below this value, retry on the full echo crop.",
    )
    parser.add_argument(
        "--expanded-search-threshold",
        type=float,
        default=0.55,
        help="If the best score remains below this value, retry with progressively expanded search rectangles.",
    )
    parser.add_argument(
        "--expanded-search-steps",
        type=str,
        default="0.03,0.06,0.10,0.15,0.20",
        help="Comma-separated margins as fractions of max echo-rect dimension for outside-crop marker recovery.",
    )
    parser.add_argument(
        "--blank-template-max-value",
        type=int,
        default=3,
        help="8-bit max pixel threshold used to ignore black/near-black orientation templates.",
    )
    parser.add_argument("--min-sugiu-confidence", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.device)
    dataset_roots = [p.expanduser().resolve() for p in args.dataset_roots]
    records, warnings = _scan_configs(
        dataset_roots=dataset_roots,
        vendor_filter=str(args.vendor_filter),
        exclude_vendors=tuple(str(vendor) for vendor in args.exclude_vendors),
        max_depth=int(args.max_config_depth),
        max_configs=int(args.max_configs),
    )
    if not records:
        raise RuntimeError("No config records found.")

    model, sugiu_image_size = _load_sugiu_model(args.sugiu_checkpoint.expanduser().resolve(), device=device)
    split_map = _split_by_config(records, seed=int(args.seed))
    template_cache: Dict[str, Optional[TemplateRow]] = {}
    expanded_search_steps = _parse_float_steps(str(args.expanded_search_steps))

    fields = [
        "sample_id",
        "split",
        "label_kind",
        "review_reason",
        "manufacturer",
        "config_folder",
        "config_path",
        "fss_path",
        "image_path",
        "image_name",
        "source_type",
        "orientation_hint",
        "orientation_hint_name",
        "filename_vertical_label",
        "filename_lr_label",
        "filename_lr_label_it",
        "filename_expected_marker_side",
        "su_giu_pred",
        "su_giu_conf",
        "prob_su",
        "prob_giu",
        "filename_vertical_mismatch",
        "lr_label",
        "lr_label_it",
        "lr_binary",
        "detected_marker_side",
        "filename_lr_mismatch",
        "match_score",
        "initial_match_score",
        "full_crop_fallback_score",
        "search_strategy",
        "search_scope",
        "search_margin_px",
        "search_rect_top_abs",
        "search_rect_left_abs",
        "search_rect_bottom_abs",
        "search_rect_right_abs",
        "template_path",
        "template_width",
        "template_height",
        "echo_rect_top_abs",
        "echo_rect_left_abs",
        "echo_rect_bottom_abs",
        "echo_rect_right_abs",
        "echo_rect_width",
        "echo_rect_height",
        "roi_top_crop",
        "roi_left_crop",
        "roi_bottom_crop",
        "roi_right_crop",
        "marker_top_crop",
        "marker_left_crop",
        "marker_bottom_crop",
        "marker_right_crop",
        "marker_top_abs",
        "marker_left_abs",
        "marker_bottom_abs",
        "marker_right_abs",
        "marker_cx_crop_norm",
        "marker_cy_crop_norm",
    ]

    rows: List[Dict[str, object]] = []
    processed = 0
    evaluated = 0
    max_rows = int(args.max_rows)
    per_sugiu_cap = int(args.max_rows_per_sugiu_pred)
    max_evaluated = int(args.max_evaluated)
    selected_sugiu_counts: Dict[str, int] = {"su": 0, "giu": 0}

    def _done_collecting() -> bool:
        if per_sugiu_cap > 0:
            return all(selected_sugiu_counts[label] >= per_sugiu_cap for label in LABELS)
        return max_rows > 0 and processed >= max_rows

    for record in records:
        if _done_collecting():
            break
        templates: List[TemplateRow] = []
        for path in record.template_paths:
            key = path.as_posix()
            if key not in template_cache:
                template_cache[key] = _load_template(
                    path,
                    device=device,
                    blank_template_max_value=int(args.blank_template_max_value),
                )
            tpl = template_cache[key]
            if tpl is not None:
                templates.append(tpl)

        for image_path in record.image_paths:
            if _done_collecting():
                break
            if max_evaluated > 0 and evaluated >= max_evaluated:
                break
            source_type, orientation_idx, _sample_idx = _orientation_hint(image_path, record.folder_path)
            filename_side, filename_lr_label, filename_lr_label_it = _filename_lr_hint(orientation_idx)
            filename_vertical = _filename_vertical_label(orientation_idx)
            row: Dict[str, object] = {
                "sample_id": "",
                "split": split_map.get(record.folder_path.as_posix(), "train"),
                "label_kind": "review",
                "review_reason": "",
                "manufacturer": record.manufacturer,
                "config_folder": _safe_rel(record.folder_path, record.dataset_root),
                "config_path": record.folder_path.as_posix(),
                "fss_path": record.fss_path.as_posix(),
                "image_path": image_path.as_posix(),
                "image_name": image_path.name,
                "source_type": source_type,
                "orientation_hint": "" if orientation_idx is None else orientation_idx,
                "orientation_hint_name": "" if orientation_idx is None else ORIENTATION_NAMES[orientation_idx],
                "filename_vertical_label": filename_vertical,
                "filename_lr_label": filename_lr_label,
                "filename_lr_label_it": filename_lr_label_it,
                "filename_expected_marker_side": filename_side,
            }
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    width, height = rgb.size
                    rect = _clip_rect(record.echo_rect, width=width, height=height)
                    if rect is None:
                        raise ValueError("bad echo rect")
                    top, left, bottom, right = rect
                    crop = rgb.crop((left, top, right + 1, bottom + 1))
                    crop_width, crop_height = crop.size
                    sugiu_pred, sugiu_conf, prob_su, prob_giu = _predict_sugiu(
                        model=model,
                        crop=crop,
                        image_size=sugiu_image_size,
                        device=device,
                    )
                    evaluated += 1
                    roi = _roi_from_sugiu(crop_width, crop_height, sugiu_pred)
                    roi_top, roi_left, roi_bottom, roi_right = roi
                    gray_full = torch.from_numpy(np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    initial_search_rect_abs = (top + roi_top, left + roi_left, top + roi_bottom, left + roi_right)
                    initial_loc = _locate_match_in_abs_rect(
                        gray_full=gray_full,
                        search_rect_abs=initial_search_rect_abs,
                        echo_rect_abs=rect,
                        templates=templates,
                        search_strategy="sugiu_roi",
                        search_scope="predicted_half",
                        search_margin_px=0,
                    )
                    best_loc = initial_loc
                    full_crop_loc: Optional[LocatedMatch] = None
                    if _score_of(best_loc) < float(args.full_crop_fallback_threshold):
                        full_crop_loc = _locate_match_in_abs_rect(
                            gray_full=gray_full,
                            search_rect_abs=rect,
                            echo_rect_abs=rect,
                            templates=templates,
                            search_strategy="full_crop_low_score_fallback",
                            search_scope="full_echo_crop",
                            search_margin_px=0,
                        )
                        if _score_of(full_crop_loc) > _score_of(best_loc):
                            best_loc = full_crop_loc
                    if _score_of(best_loc) < float(args.expanded_search_threshold):
                        for step in expanded_search_steps:
                            margin_px = max(1, int(round(max(crop_width, crop_height) * step)))
                            expanded_rect = _expand_rect(initial_search_rect_abs, width=width, height=height, margin_px=margin_px)
                            if expanded_rect is None:
                                continue
                            expanded_loc = _locate_match_in_abs_rect(
                                gray_full=gray_full,
                                search_rect_abs=expanded_rect,
                                echo_rect_abs=rect,
                                templates=templates,
                                search_strategy="expanded_rect_fallback",
                                search_scope="expanded_predicted_half",
                                search_margin_px=margin_px,
                            )
                            if _score_of(expanded_loc) > _score_of(best_loc):
                                best_loc = expanded_loc
                            if _score_of(best_loc) >= float(args.expanded_search_threshold):
                                break
                    row.update(
                        {
                            "su_giu_pred": sugiu_pred,
                            "su_giu_conf": f"{sugiu_conf:.6f}",
                            "prob_su": f"{prob_su:.6f}",
                            "prob_giu": f"{prob_giu:.6f}",
                            "filename_vertical_mismatch": int(bool(filename_vertical and filename_vertical != sugiu_pred)),
                            "initial_match_score": "" if initial_loc is None else f"{initial_loc.score:.6f}",
                            "full_crop_fallback_score": "" if full_crop_loc is None else f"{full_crop_loc.score:.6f}",
                            "echo_rect_top_abs": top,
                            "echo_rect_left_abs": left,
                            "echo_rect_bottom_abs": bottom,
                            "echo_rect_right_abs": right,
                            "echo_rect_width": crop_width,
                            "echo_rect_height": crop_height,
                            "roi_top_crop": roi_top,
                            "roi_left_crop": roi_left,
                            "roi_bottom_crop": roi_bottom,
                            "roi_right_crop": roi_right,
                        }
                    )
                    if best_loc is None:
                        row["review_reason"] = "no_usable_template" if not templates else "no_template_match"
                    else:
                        marker_top = best_loc.marker_top_crop
                        marker_left = best_loc.marker_left_crop
                        marker_bottom = best_loc.marker_bottom_crop
                        marker_right = best_loc.marker_right_crop
                        marker_cx = ((marker_left + marker_right) / 2.0)
                        marker_cy = ((marker_top + marker_bottom) / 2.0)
                        detected_side = "left" if marker_cx < (crop_width / 2.0) else "right"
                        lr_label, lr_label_it, lr_binary = _lr_from_detected_side(detected_side)
                        filename_lr_mismatch = int(bool(filename_side and filename_side != detected_side))
                        reasons: List[str] = []
                        if sugiu_conf < float(args.min_sugiu_confidence):
                            reasons.append("low_sugiu_conf")
                        if best_loc.score < float(args.min_match_score):
                            reasons.append("low_template_score")
                        label_kind = "auto_candidate" if not reasons else "review"
                        row.update(
                            {
                                "label_kind": label_kind,
                                "review_reason": ";".join(reasons),
                                "lr_label": lr_label,
                                "lr_label_it": lr_label_it,
                                "lr_binary": lr_binary,
                                "detected_marker_side": detected_side,
                                "filename_lr_mismatch": filename_lr_mismatch,
                                "match_score": f"{best_loc.score:.6f}",
                                "search_strategy": best_loc.search_strategy,
                                "search_scope": best_loc.search_scope,
                                "search_margin_px": best_loc.search_margin_px,
                                "search_rect_top_abs": best_loc.search_rect_abs[0],
                                "search_rect_left_abs": best_loc.search_rect_abs[1],
                                "search_rect_bottom_abs": best_loc.search_rect_abs[2],
                                "search_rect_right_abs": best_loc.search_rect_abs[3],
                                "template_path": best_loc.template.path.as_posix(),
                                "template_width": best_loc.template.width,
                                "template_height": best_loc.template.height,
                                "marker_top_crop": marker_top,
                                "marker_left_crop": marker_left,
                                "marker_bottom_crop": marker_bottom,
                                "marker_right_crop": marker_right,
                                "marker_top_abs": best_loc.marker_top_abs,
                                "marker_left_abs": best_loc.marker_left_abs,
                                "marker_bottom_abs": best_loc.marker_bottom_abs,
                                "marker_right_abs": best_loc.marker_right_abs,
                                "marker_cx_crop_norm": f"{marker_cx / max(1, crop_width):.6f}",
                                "marker_cy_crop_norm": f"{marker_cy / max(1, crop_height):.6f}",
                            }
                        )
                keep_row = True
                if per_sugiu_cap > 0:
                    keep_row = selected_sugiu_counts.get(str(row.get("su_giu_pred", "")), 0) < per_sugiu_cap
                if keep_row:
                    processed += 1
                    row["sample_id"] = f"lrsg_{processed:08d}"
                    pred_key = str(row.get("su_giu_pred", ""))
                    if pred_key in selected_sugiu_counts:
                        selected_sugiu_counts[pred_key] += 1
                    rows.append(row)
            except Exception as exc:  # pylint: disable=broad-except
                evaluated += 1
                row["review_reason"] = f"open_or_process_error:{exc}"
                if per_sugiu_cap <= 0:
                    processed += 1
                    row["sample_id"] = f"lrsg_{processed:08d}"
                    rows.append(row)
            if int(args.progress_every) > 0 and evaluated % int(args.progress_every) == 0:
                print(
                    f"evaluated {evaluated} images | kept {processed} "
                    f"(su={selected_sugiu_counts['su']}, giu={selected_sugiu_counts['giu']})",
                    flush=True,
                )
        if _done_collecting() or (max_evaluated > 0 and evaluated >= max_evaluated):
            break

    manifest_path = output_dir / "manifest.csv"
    _write_csv(manifest_path, rows, fields)
    vendor_summary = _write_by_vendor(output_dir, rows, fields)
    scores = [float(r["match_score"]) for r in rows if str(r.get("match_score", ""))]
    summary = {
        "dataset_roots": [p.as_posix() for p in dataset_roots],
        "sugiu_checkpoint": args.sugiu_checkpoint.expanduser().resolve().as_posix(),
        "output_dir": output_dir.as_posix(),
        "device": str(device),
        "config_count": len(records),
        "images_evaluated": evaluated,
        "rows": len(rows),
        "label_kind_counts": _counts(rows, "label_kind"),
        "review_reason_counts": _counts(rows, "review_reason"),
        "manufacturer_counts": _counts(rows, "manufacturer"),
        "su_giu_pred_counts": _counts(rows, "su_giu_pred"),
        "filename_vertical_counts": _counts(rows, "filename_vertical_label"),
        "lr_label_counts": _counts(rows, "lr_label"),
        "search_strategy_counts": _counts(rows, "search_strategy"),
        "match_score_stats": {
            "mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "median": float(statistics.median(scores)) if scores else 0.0,
            "min": float(min(scores)) if scores else 0.0,
            "max": float(max(scores)) if scores else 0.0,
        },
        "vendor_summary": vendor_summary,
        "warnings_count": len(warnings),
        "warnings": warnings[:300],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Rows: {len(rows)} | Auto candidates: {_counts(rows, 'label_kind').get('auto_candidate', 0)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
