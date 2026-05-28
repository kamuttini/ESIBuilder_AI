#!/usr/bin/env python3
"""Build training manifests for automatic compilation of .fss line #16.

This script creates:
1) image-level manifest for orientation template rectangle detection
2) flip-level manifest for line #16 parameter prediction (B/CH/TH/P1..P5/MM)
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
INT_RE = re.compile(r"^-?\d+$")
FLOAT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
ORIENTATION_SETUP_RE = re.compile(
    r"^image_orientation_setup_\d+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$",
    re.IGNORECASE,
)
GROUP_ORIENTATION_SYMBOL = 4


@dataclass(frozen=True)
class OrientationEntry:
    flip_idx: int
    top: int
    left: int
    bottom: int
    right: int
    b_value: int
    channel: int
    threshold: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    mm: int
    legacy_threshold_only: int


@dataclass(frozen=True)
class FolderRecord:
    folder_id: str
    folder_path: Path
    folder_name: str
    manufacturer: str
    model_name: str
    fss_path: Path
    fss_id_echo: int
    fss_id_probe: int
    fss_probe_type: int
    fss_video_x: int
    fss_video_y: int
    fss_group_orientation: int
    orient_available: Tuple[int, int, int, int]
    entries: Tuple[OrientationEntry, OrientationEntry, OrientationEntry, OrientationEntry]
    canonical_rect: Tuple[int, int, int, int]
    image_paths: Tuple[Path, ...]
    sample_image_width: int
    sample_image_height: int

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def infer_manufacturer(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    if re.match(r"^bk(?:[\s,_-]|\d)", name):
        return "BK"
    if name.startswith("esaote"):
        return "Esaote"
    if name.startswith("hitachi"):
        return "Hitachi"
    if name.startswith("ge"):
        return "GE"
    if name.startswith("mindray"):
        return "Mindray"
    if name.startswith("canon"):
        return "Canon"
    if name.startswith("philips"):
        return "Philips"
    if name.startswith("toshiba"):
        return "Toshiba"
    if name.startswith("siemens"):
        return "Siemens"
    if name.startswith("koelis"):
        return "Koelis"
    if name.startswith("biopsee"):
        return "Biopsee"
    if name.startswith("terason"):
        return "Terason"
    if name.startswith("sonostar"):
        return "Sonostar"
    if name.startswith("exactvu"):
        return "ExactVu"
    if name.startswith("alpinion"):
        return "Alpinion"
    if name.startswith("vinno"):
        return "Vinno"
    token = re.split(r"[\s,_-]+", folder_name.strip())[0]
    return token if token else "UNKNOWN"


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    cleaned = value.strip()
    if INT_RE.match(cleaned):
        return int(cleaned)
    return default


def _parse_float(value: str) -> Optional[float]:
    cleaned = value.strip()
    if FLOAT_RE.match(cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_rect_line(lines: List[str]) -> Tuple[int, int, int, int, int]:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines):
            match = RECT_RE.match(lines[candidate - 1].strip())
            if match:
                top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
                return candidate, top, left, bottom, right
    for idx in range(1, min(len(lines), 20) + 1):
        match = RECT_RE.match(lines[idx - 1].strip())
        if match:
            top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
            return idx, top, left, bottom, right
    raise ValueError("RECT_ECHO line not found.")


def _parse_params_token(token: str) -> Tuple[int, float, float, float, float, float, float, int]:
    """Return CH,TH,P1,P2,P3,P4,P5 and legacy_flag."""
    pieces = token.strip().split(":")
    if len(pieces) == 7:
        try:
            return (
                int(pieces[0]),
                float(pieces[1]),
                float(pieces[2]),
                float(pieces[3]),
                float(pieces[4]),
                float(pieces[5]),
                float(pieces[6]),
                0,
            )
        except ValueError:
            pass

    # Legacy format had only TH in this field.
    th_value = _parse_float(token)
    if th_value is not None:
        return 7, th_value, 20.0, 120.0, 0.0, 0.0, 0.0, 1

    raise ValueError(f"Unable to parse params token: {token!r}")


def _parse_line16_entries(lines: List[str], rect_line: int) -> List[OrientationEntry]:
    orientation_line = rect_line + 5
    if orientation_line > len(lines):
        raise ValueError("Line #16 not present.")

    raw = lines[orientation_line - 1].strip()
    if not raw:
        raise ValueError("Empty line #16.")

    segments = [seg.strip().rstrip(",") for seg in raw.split(";") if seg.strip()]
    entries: List[OrientationEntry] = []
    for idx, segment in enumerate(segments):
        parts = segment.split("|")
        if len(parts) < 7:
            raise ValueError(f"Malformed line #16 segment: {segment!r}")

        top = int(parts[0].strip())
        left = int(parts[1].strip())
        bottom = int(parts[2].strip())
        right = int(parts[3].strip())
        b_value = int(parts[4].strip())
        channel, threshold, p1, p2, p3, p4, p5, legacy = _parse_params_token(parts[5])
        mm = int(parts[6].strip())

        entries.append(
            OrientationEntry(
                flip_idx=idx,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
                b_value=b_value,
                channel=channel,
                threshold=threshold,
                p1=p1,
                p2=p2,
                p3=p3,
                p4=p4,
                p5=p5,
                mm=mm,
                legacy_threshold_only=legacy,
            )
        )

    if not entries:
        raise ValueError("No segments parsed from line #16.")

    if len(entries) < 4:
        raise ValueError(f"Expected 4 flip entries in line #16, got {len(entries)}.")
    if len(entries) > 4:
        entries = entries[:4]

    # Force canonical flip indexes 0..3.
    out: List[OrientationEntry] = []
    for flip_idx in range(4):
        src = entries[flip_idx]
        out.append(
            OrientationEntry(
                flip_idx=flip_idx,
                top=src.top,
                left=src.left,
                bottom=src.bottom,
                right=src.right,
                b_value=src.b_value,
                channel=src.channel,
                threshold=src.threshold,
                p1=src.p1,
                p2=src.p2,
                p3=src.p3,
                p4=src.p4,
                p5=src.p5,
                mm=src.mm,
                legacy_threshold_only=src.legacy_threshold_only,
            )
        )
    return out


def _parse_orient_file(orient_path: Path) -> Tuple[int, int, int, int]:
    if not orient_path.exists():
        return (1, 1, 1, 1)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(orient_path, encoding="utf-8")
    section = "Orientations"
    if section not in parser:
        return (1, 1, 1, 1)

    def _get_bool(name: str) -> int:
        value = parser[section].get(name, "true").strip().lower()
        return 1 if value in {"1", "true", "yes", "on"} else 0

    return (
        _get_bool("FLIP_NO"),
        _get_bool("FLIP_LR"),
        _get_bool("FLIP_UD"),
        _get_bool("FLIP_LR_UD"),
    )


def _median_rect(entries: Sequence[OrientationEntry]) -> Tuple[int, int, int, int]:
    top = int(round(statistics.median([item.top for item in entries])))
    left = int(round(statistics.median([item.left for item in entries])))
    bottom = int(round(statistics.median([item.bottom for item in entries])))
    right = int(round(statistics.median([item.right for item in entries])))
    return top, left, bottom, right


def _collect_orientation_images(image_dir: Path, max_images_per_folder: int, seed: int) -> Tuple[Path, ...]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and ORIENTATION_SETUP_RE.match(path.name)
    ]
    if max_images_per_folder > 0 and len(images) > max_images_per_folder:
        rng = random.Random(seed)
        idx = list(range(len(images)))
        rng.shuffle(idx)
        keep = sorted(idx[:max_images_per_folder])
        images = [images[i] for i in keep]
    return tuple(images)


def _split_by_manufacturer(
    records: List[FolderRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_manufacturer: Dict[str, List[FolderRecord]] = {}
    for record in records:
        by_manufacturer.setdefault(record.manufacturer, []).append(record)

    assignments: Dict[str, str] = {}
    for manufacturer, group in sorted(by_manufacturer.items()):
        records_group = list(group)
        rng.shuffle(records_group)
        records_group.sort(key=lambda rec: (-rec.image_count, rec.folder_name.lower()))
        n = len(records_group)
        if n == 1:
            assignments[records_group[0].folder_id] = "train"
            continue
        if n == 2:
            assignments[records_group[0].folder_id] = "train"
            assignments[records_group[1].folder_id] = "val"
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)
        while n_train + n_val + n_test > n:
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test = max(1, n_test - 1)

        idx = 0
        for _ in range(n_train):
            assignments[records_group[idx].folder_id] = "train"
            idx += 1
        for _ in range(n_val):
            assignments[records_group[idx].folder_id] = "val"
            idx += 1
        while idx < n:
            assignments[records_group[idx].folder_id] = "test"
            idx += 1

    return assignments


def _scan_dataset(
    dataset_roots: Sequence[Path],
    include_non_symbol: bool,
    max_images_per_folder: int,
    seed: int,
) -> Tuple[List[FolderRecord], List[str]]:
    records: List[FolderRecord] = []
    warnings: List[str] = []

    seen_folder_ids: set[str] = set()
    for dataset_root in dataset_roots:
        if not dataset_root.is_dir():
            warnings.append(f"Dataset root missing: {dataset_root}")
            continue

        for folder in sorted(dataset_root.iterdir()):
            if not folder.is_dir():
                continue

            setup_dir = folder / "DB_setup"
            image_dir = folder / "image_samples"
            if not setup_dir.is_dir() or not image_dir.is_dir():
                continue

            fss_files = sorted(p for p in setup_dir.glob("*.fss") if p.is_file())
            if len(fss_files) != 1:
                warnings.append(
                    f"{folder.name}: expected 1 .fss in DB_setup, found {len(fss_files)}"
                )
                continue
            fss_path = fss_files[0]

            try:
                lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"{folder.name}: cannot read {fss_path.name}: {exc}")
                continue

            try:
                rect_line, _, _, _, _ = _extract_rect_line(lines)
                entries = _parse_line16_entries(lines, rect_line=rect_line)
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"{folder.name}: line16 parse failed: {exc}")
                continue

            offset = rect_line - 11
            fss_id_echo = _parse_int(lines[1 + offset] if (2 + offset) <= len(lines) else None, -1)
            fss_id_probe = _parse_int(lines[2 + offset] if (3 + offset) <= len(lines) else None, -1)
            fss_probe_type = _parse_int(lines[3 + offset] if (4 + offset) <= len(lines) else None, -1)
            fss_video_x = _parse_int(lines[8 + offset] if (9 + offset) <= len(lines) else None, 0)
            fss_video_y = _parse_int(lines[9 + offset] if (10 + offset) <= len(lines) else None, 0)
            fss_group_orientation = _parse_int(
                lines[rect_line] if (rect_line + 1) <= len(lines) else None,
                GROUP_ORIENTATION_SYMBOL,
            )

            if not include_non_symbol and fss_group_orientation != GROUP_ORIENTATION_SYMBOL:
                continue

            orient_path = setup_dir / f"setup_{fss_path.stem.split('_')[-1]}.orient"
            orient_available = _parse_orient_file(orient_path)

            image_paths = _collect_orientation_images(
                image_dir,
                max_images_per_folder=max_images_per_folder,
                seed=seed + len(records) * 17 + len(folder.name),
            )
            if not image_paths:
                warnings.append(f"{folder.name}: no image_orientation_setup_* files found.")
                continue

            sample_image = image_paths[0]
            try:
                with Image.open(sample_image) as img:
                    sample_w, sample_h = img.size
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"{folder.name}: cannot open sample image: {exc}")
                continue

            if fss_video_x <= 0:
                fss_video_x = sample_w
            if fss_video_y <= 0:
                fss_video_y = sample_h

            canonical_rect = _median_rect(entries)
            folder_id = folder.resolve().as_posix()
            if folder_id in seen_folder_ids:
                continue
            seen_folder_ids.add(folder_id)

            records.append(
                FolderRecord(
                    folder_id=folder_id,
                    folder_path=folder.resolve(),
                    folder_name=folder.name,
                    manufacturer=infer_manufacturer(folder.name),
                    model_name=folder.name,
                    fss_path=fss_path.resolve(),
                    fss_id_echo=fss_id_echo,
                    fss_id_probe=fss_id_probe,
                    fss_probe_type=fss_probe_type,
                    fss_video_x=fss_video_x,
                    fss_video_y=fss_video_y,
                    fss_group_orientation=fss_group_orientation,
                    orient_available=orient_available,
                    entries=tuple(entries),
                    canonical_rect=canonical_rect,
                    image_paths=image_paths,
                    sample_image_width=sample_w,
                    sample_image_height=sample_h,
                )
            )

    return records, warnings


def _write_rect_manifest(
    path: Path,
    records: Sequence[FolderRecord],
    assignments: Dict[str, str],
) -> None:
    fields = [
        "image_path",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "fss_id_echo",
        "fss_id_probe",
        "fss_probe_type",
        "fss_video_x",
        "fss_video_y",
        "fss_group_orientation",
        "orient_available_flip_no",
        "orient_available_flip_lr",
        "orient_available_flip_ud",
        "orient_available_flip_lrud",
        "rect0_top",
        "rect0_left",
        "rect0_bottom",
        "rect0_right",
        "rect1_top",
        "rect1_left",
        "rect1_bottom",
        "rect1_right",
        "rect2_top",
        "rect2_left",
        "rect2_bottom",
        "rect2_right",
        "rect3_top",
        "rect3_left",
        "rect3_bottom",
        "rect3_right",
        "rect_canonical_top",
        "rect_canonical_left",
        "rect_canonical_bottom",
        "rect_canonical_right",
        "sample_image_width",
        "sample_image_height",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.folder_name.lower()):
            split = assignments[record.folder_id]
            for image_path in record.image_paths:
                row = {
                    "image_path": image_path.as_posix(),
                    "split": split,
                    "dataset_folder": record.folder_name,
                    "manufacturer": record.manufacturer,
                    "model_name": record.model_name,
                    "fss_path": record.fss_path.as_posix(),
                    "fss_id_echo": record.fss_id_echo,
                    "fss_id_probe": record.fss_id_probe,
                    "fss_probe_type": record.fss_probe_type,
                    "fss_video_x": record.fss_video_x,
                    "fss_video_y": record.fss_video_y,
                    "fss_group_orientation": record.fss_group_orientation,
                    "orient_available_flip_no": record.orient_available[0],
                    "orient_available_flip_lr": record.orient_available[1],
                    "orient_available_flip_ud": record.orient_available[2],
                    "orient_available_flip_lrud": record.orient_available[3],
                    "rect_canonical_top": record.canonical_rect[0],
                    "rect_canonical_left": record.canonical_rect[1],
                    "rect_canonical_bottom": record.canonical_rect[2],
                    "rect_canonical_right": record.canonical_rect[3],
                    "sample_image_width": record.sample_image_width,
                    "sample_image_height": record.sample_image_height,
                }
                for idx, entry in enumerate(record.entries):
                    row[f"rect{idx}_top"] = entry.top
                    row[f"rect{idx}_left"] = entry.left
                    row[f"rect{idx}_bottom"] = entry.bottom
                    row[f"rect{idx}_right"] = entry.right
                writer.writerow(row)


def _write_params_manifest(
    path: Path,
    records: Sequence[FolderRecord],
    assignments: Dict[str, str],
) -> None:
    fields = [
        "sample_id",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "fss_id_echo",
        "fss_id_probe",
        "fss_probe_type",
        "fss_video_x",
        "fss_video_y",
        "fss_group_orientation",
        "flip_idx",
        "flip_available",
        "rect_top",
        "rect_left",
        "rect_bottom",
        "rect_right",
        "rect_width",
        "rect_height",
        "rect_cx_norm",
        "rect_cy_norm",
        "rect_area_norm",
        "b_value",
        "channel",
        "threshold",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "mm",
        "legacy_threshold_only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.folder_name.lower()):
            split = assignments[record.folder_id]
            for entry in record.entries:
                rect_width = float(entry.right - entry.left + 1)
                rect_height = float(entry.bottom - entry.top + 1)
                video_x = max(1.0, float(record.fss_video_x))
                video_y = max(1.0, float(record.fss_video_y))
                cx = (float(entry.left) + float(entry.right)) * 0.5
                cy = (float(entry.top) + float(entry.bottom)) * 0.5
                area_norm = (rect_width * rect_height) / (video_x * video_y)

                writer.writerow(
                    {
                        "sample_id": f"{record.folder_name}::flip{entry.flip_idx}",
                        "split": split,
                        "dataset_folder": record.folder_name,
                        "manufacturer": record.manufacturer,
                        "model_name": record.model_name,
                        "fss_path": record.fss_path.as_posix(),
                        "fss_id_echo": record.fss_id_echo,
                        "fss_id_probe": record.fss_id_probe,
                        "fss_probe_type": record.fss_probe_type,
                        "fss_video_x": record.fss_video_x,
                        "fss_video_y": record.fss_video_y,
                        "fss_group_orientation": record.fss_group_orientation,
                        "flip_idx": entry.flip_idx,
                        "flip_available": record.orient_available[entry.flip_idx],
                        "rect_top": entry.top,
                        "rect_left": entry.left,
                        "rect_bottom": entry.bottom,
                        "rect_right": entry.right,
                        "rect_width": rect_width,
                        "rect_height": rect_height,
                        "rect_cx_norm": cx / video_x,
                        "rect_cy_norm": cy / video_y,
                        "rect_area_norm": area_norm,
                        "b_value": entry.b_value,
                        "channel": entry.channel,
                        "threshold": entry.threshold,
                        "p1": entry.p1,
                        "p2": entry.p2,
                        "p3": entry.p3,
                        "p4": entry.p4,
                        "p5": entry.p5,
                        "mm": entry.mm,
                        "legacy_threshold_only": entry.legacy_threshold_only,
                    }
                )


def _write_folder_table(path: Path, records: Sequence[FolderRecord], assignments: Dict[str, str]) -> None:
    fields = [
        "dataset_folder",
        "split",
        "manufacturer",
        "model_name",
        "image_count",
        "fss_path",
        "fss_group_orientation",
        "fss_id_echo",
        "fss_id_probe",
        "fss_probe_type",
        "fss_video_x",
        "fss_video_y",
        "orient_available_flip_no",
        "orient_available_flip_lr",
        "orient_available_flip_ud",
        "orient_available_flip_lrud",
        "canonical_rect_top",
        "canonical_rect_left",
        "canonical_rect_bottom",
        "canonical_rect_right",
        "sample_image_width",
        "sample_image_height",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.folder_name.lower()):
            writer.writerow(
                {
                    "dataset_folder": record.folder_name,
                    "split": assignments[record.folder_id],
                    "manufacturer": record.manufacturer,
                    "model_name": record.model_name,
                    "image_count": record.image_count,
                    "fss_path": record.fss_path.as_posix(),
                    "fss_group_orientation": record.fss_group_orientation,
                    "fss_id_echo": record.fss_id_echo,
                    "fss_id_probe": record.fss_id_probe,
                    "fss_probe_type": record.fss_probe_type,
                    "fss_video_x": record.fss_video_x,
                    "fss_video_y": record.fss_video_y,
                    "orient_available_flip_no": record.orient_available[0],
                    "orient_available_flip_lr": record.orient_available[1],
                    "orient_available_flip_ud": record.orient_available[2],
                    "orient_available_flip_lrud": record.orient_available[3],
                    "canonical_rect_top": record.canonical_rect[0],
                    "canonical_rect_left": record.canonical_rect[1],
                    "canonical_rect_bottom": record.canonical_rect[2],
                    "canonical_rect_right": record.canonical_rect[3],
                    "sample_image_width": record.sample_image_width,
                    "sample_image_height": record.sample_image_height,
                }
            )


def _build_summary(records: Sequence[FolderRecord], assignments: Dict[str, str], warnings: Sequence[str]) -> Dict[str, object]:
    folders_per_split = {split: 0 for split in SPLITS}
    images_per_split = {split: 0 for split in SPLITS}
    manufacturers: Dict[str, Dict[str, int]] = {}
    mm_counter: Dict[int, int] = {}
    ch_counter: Dict[int, int] = {}
    b_counter: Dict[int, int] = {}

    for record in records:
        split = assignments[record.folder_id]
        folders_per_split[split] += 1
        images_per_split[split] += record.image_count

        man = manufacturers.setdefault(record.manufacturer, {"folders": 0, "images": 0})
        man["folders"] += 1
        man["images"] += record.image_count

        for entry in record.entries:
            mm_counter[entry.mm] = mm_counter.get(entry.mm, 0) + 1
            ch_counter[entry.channel] = ch_counter.get(entry.channel, 0) + 1
            b_counter[entry.b_value] = b_counter.get(entry.b_value, 0) + 1

    return {
        "total_folders": len(records),
        "total_images": sum(record.image_count for record in records),
        "folders_per_split": folders_per_split,
        "images_per_split": images_per_split,
        "mm_distribution": dict(sorted(mm_counter.items())),
        "channel_distribution": dict(sorted(ch_counter.items())),
        "b_distribution": dict(sorted(b_counter.items())),
        "manufacturers": dict(sorted(manufacturers.items())),
        "warnings_count": len(warnings),
        "warnings": list(warnings)[:200],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare datasets for orientation template detection and line16 parameter prediction."
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=["Dataset"],
        help="One or more dataset roots containing per-model folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset"),
        help="Output directory for manifests.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images-per-folder",
        type=int,
        default=0,
        help="0 = keep all image_orientation_setup files.",
    )
    parser.add_argument(
        "--include-non-symbol",
        action="store_true",
        help="Include folders with GROUP_ORIENTATION != 4.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = [Path(path).expanduser().resolve() for path in args.dataset_roots]
    records, warnings = _scan_dataset(
        dataset_roots=dataset_roots,
        include_non_symbol=bool(args.include_non_symbol),
        max_images_per_folder=int(args.max_images_per_folder),
        seed=int(args.seed),
    )
    if not records:
        raise RuntimeError("No valid folders found for orientation line16 dataset.")

    assignments = _split_by_manufacturer(
        records=records,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    rect_manifest = output_dir / "manifest_orientation_rect.csv"
    params_manifest = output_dir / "manifest_line16_params.csv"
    folder_table = output_dir / "folders_line16.csv"

    _write_rect_manifest(rect_manifest, records, assignments)
    _write_params_manifest(params_manifest, records, assignments)
    _write_folder_table(folder_table, records, assignments)

    summary = _build_summary(records, assignments, warnings)
    summary.update(
        {
            "dataset_roots": [path.as_posix() for path in dataset_roots],
            "output_dir": output_dir.as_posix(),
            "rect_manifest": rect_manifest.as_posix(),
            "params_manifest": params_manifest.as_posix(),
            "folder_table": folder_table.as_posix(),
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_txt = output_dir / "summary.txt"
    lines = [
        f"Folders: {summary['total_folders']}",
        f"Images: {summary['total_images']}",
        f"Split folders: {summary['folders_per_split']}",
        f"Split images: {summary['images_per_split']}",
        f"MM distribution: {summary['mm_distribution']}",
        f"CH distribution: {summary['channel_distribution']}",
        f"B distribution: {summary['b_distribution']}",
        f"Warnings: {summary['warnings_count']}",
        f"Rect manifest: {rect_manifest.as_posix()}",
        f"Params manifest: {params_manifest.as_posix()}",
        f"Folder table: {folder_table.as_posix()}",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Rect manifest: {rect_manifest}", flush=True)
    print(f"Params manifest: {params_manifest}", flush=True)
    print(f"Folder table: {folder_table}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(
        f"Folders={summary['total_folders']} Images={summary['total_images']} "
        f"Warnings={summary['warnings_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

