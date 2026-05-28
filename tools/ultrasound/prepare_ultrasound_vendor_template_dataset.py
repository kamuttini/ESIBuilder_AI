#!/usr/bin/env python3
"""Build vendor-template manifest (line13 RECT_NAME_ECHO bbox) with folder-level splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
RECT_NAME_PREFIX_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|")
INT_RE = re.compile(r"^-?\d+$")
VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
NEGATIVE_RE_DEFAULT = r"(?i)negative"

VENDOR_CLASS_ORDER: Tuple[str, ...] = (
    "Alpinion",
    "BK",
    "Biopsee",
    "Canon",
    "Esaote",
    "ExactVu",
    "GE",
    "Hitachi",
    "Koelis",
    "Mindray",
    "Philips",
    "Siemens",
    "Sonostar",
    "Terason",
    "Toshiba",
    "Vinno",
)
VENDOR_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(VENDOR_CLASS_ORDER)}


@dataclass(frozen=True)
class FolderRecord:
    folder_id: str
    folder_path: Path
    folder_name: str
    manufacturer: str
    vendor_id: int
    model_name: str
    fss_path: Path
    setup_name: str
    setup_numeric_id: Optional[int]
    fss_version: Optional[str]
    fss_rect_line: int
    fss_id_echo: Optional[int]
    fss_id_probe: Optional[int]
    fss_video_x: Optional[int]
    fss_video_y: Optional[int]
    rect_echo_top: int
    rect_echo_left: int
    rect_echo_bottom: int
    rect_echo_right: int
    bbox_source: str
    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    template_path: Optional[Path]
    template_exists: bool
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
    return "UNKNOWN"


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    cleaned = value.strip()
    if INT_RE.match(cleaned):
        return int(cleaned)
    return None


def _safe_line(lines: List[str], line_num: int) -> Optional[str]:
    if 1 <= line_num <= len(lines):
        return lines[line_num - 1].strip()
    return None


def _extract_rect_line(lines: List[str]) -> Tuple[int, int, int, int, int]:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines):
            match = RECT_RE.match(lines[candidate - 1].strip())
            if match:
                top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
                return candidate, top, left, bottom, right
    for idx in range(1, min(len(lines), 15) + 1):
        match = RECT_RE.match(lines[idx - 1].strip())
        if match:
            top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
            return idx, top, left, bottom, right
    raise ValueError("RECT_ECHO not found in first 15 lines")


def _parse_rect_name_echo_coords(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    match = RECT_NAME_PREFIX_RE.match(value.strip())
    if not match:
        return None
    top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
    return top, left, bottom, right


def parse_fss(path: Path) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rect_line, rect_echo_top, rect_echo_left, rect_echo_bottom, rect_echo_right = _extract_rect_line(lines)
    offset = rect_line - 11

    version_line = _safe_line(lines, 1 + offset)
    version = version_line if version_line and VERSION_RE.match(version_line) else None

    id_echo = _parse_int(_safe_line(lines, 2 + offset))
    id_probe = _parse_int(_safe_line(lines, 3 + offset))
    video_x = _parse_int(_safe_line(lines, 9 + offset))
    video_y = _parse_int(_safe_line(lines, 10 + offset))
    line13 = _safe_line(lines, 13 + offset)
    rect_name_echo = _parse_rect_name_echo_coords(line13)

    return {
        "fss_rect_line": rect_line,
        "fss_version": version,
        "fss_id_echo": id_echo,
        "fss_id_probe": id_probe,
        "fss_video_x": video_x,
        "fss_video_y": video_y,
        "rect_echo_top": rect_echo_top,
        "rect_echo_left": rect_echo_left,
        "rect_echo_bottom": rect_echo_bottom,
        "rect_echo_right": rect_echo_right,
        "rect_name_echo_top": rect_name_echo[0] if rect_name_echo else None,
        "rect_name_echo_left": rect_name_echo[1] if rect_name_echo else None,
        "rect_name_echo_bottom": rect_name_echo[2] if rect_name_echo else None,
        "rect_name_echo_right": rect_name_echo[3] if rect_name_echo else None,
    }


def _find_dataset_folders(dataset_root: Path, exclude_re: Optional[re.Pattern[str]]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for image_dir in sorted(dataset_root.rglob("image_samples")):
        if not image_dir.is_dir():
            continue
        folder = image_dir.parent
        if exclude_re and exclude_re.search(folder.name):
            continue
        setup_dir = folder / "DB_setup"
        if not setup_dir.is_dir():
            continue
        real_folder = folder.resolve()
        if real_folder in seen:
            continue
        seen.add(real_folder)
        out.append(folder)
    return out


def _collect_image_paths(image_dir: Path) -> Tuple[Path, ...]:
    paths = [
        p
        for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
    ]
    return tuple(paths)


def _find_db_echo_dir(folder: Path) -> Optional[Path]:
    for child in folder.iterdir():
        if child.is_dir() and child.name.lower() == "db_echo":
            return child
    return None


def _find_template_path(folder: Path, setup_name: str) -> Optional[Path]:
    db_echo_dir = _find_db_echo_dir(folder)
    if db_echo_dir is None:
        return None

    setup_dir = db_echo_dir / setup_name
    if setup_dir.is_dir():
        exact = setup_dir / "echo_name.png"
        if exact.is_file():
            return exact
        for p in sorted(setup_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.name.lower() == "echo_name.png":
                return p
        for p in sorted(setup_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "echo_name" in p.name.lower():
                return p

    # Fallback for folders where setup naming is not aligned.
    candidates: List[Path] = []
    for p in db_echo_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        low = p.name.lower()
        if low == "echo_name.png" or "echo_name" in low:
            candidates.append(p)
    if not candidates:
        return None

    # Prefer candidate inside a folder named like setup_name.
    for p in candidates:
        if p.parent.name.lower() == setup_name.lower():
            return p
    return sorted(candidates)[0]


def _parse_setup_numeric_id(setup_name: str) -> Optional[int]:
    m = re.match(r"^setup_(\d+)$", setup_name.lower())
    if not m:
        return None
    return int(m.group(1))


def scan_dataset(
    dataset_root: Path,
    exclude_folder_regex: Optional[str],
    strict_line13: bool,
) -> Tuple[List[FolderRecord], List[str]]:
    exclude_re = re.compile(exclude_folder_regex) if exclude_folder_regex else None
    warnings: List[str] = []
    records: List[FolderRecord] = []

    folders = _find_dataset_folders(dataset_root, exclude_re)
    for folder in folders:
        image_dir = folder / "image_samples"
        setup_dir = folder / "DB_setup"

        image_paths = _collect_image_paths(image_dir)
        if not image_paths:
            warnings.append(f"{folder.as_posix()}: no images in image_samples")
            continue

        with Image.open(image_paths[0]) as img0:
            sample_w, sample_h = img0.size

        fss_candidates = sorted(p for p in setup_dir.glob("*.fss") if not p.name.startswith("~$"))
        if len(fss_candidates) != 1:
            warnings.append(
                f"{folder.as_posix()}: expected 1 .fss in DB_setup, found {len(fss_candidates)}"
            )
            continue

        fss_path = fss_candidates[0]
        setup_name = fss_path.stem

        try:
            fss_data = parse_fss(fss_path)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"{folder.as_posix()}: failed parsing {fss_path.name}: {exc}")
            continue

        rect_name_top = fss_data["rect_name_echo_top"]
        rect_name_left = fss_data["rect_name_echo_left"]
        rect_name_bottom = fss_data["rect_name_echo_bottom"]
        rect_name_right = fss_data["rect_name_echo_right"]

        if (
            rect_name_top is None
            or rect_name_left is None
            or rect_name_bottom is None
            or rect_name_right is None
        ):
            if strict_line13:
                warnings.append(
                    f"{folder.as_posix()}: line13 RECT_NAME_ECHO missing/unparseable in {fss_path.name}"
                )
                continue
            bbox_source = "line11_fallback_missing_line13"
            bbox_top = int(fss_data["rect_echo_top"])
            bbox_left = int(fss_data["rect_echo_left"])
            bbox_bottom = int(fss_data["rect_echo_bottom"])
            bbox_right = int(fss_data["rect_echo_right"])
        else:
            bbox_source = "line13"
            bbox_top = int(rect_name_top)
            bbox_left = int(rect_name_left)
            bbox_bottom = int(rect_name_bottom)
            bbox_right = int(rect_name_right)

        manufacturer = infer_manufacturer(folder.name)
        vendor_id = VENDOR_TO_ID.get(manufacturer, -1)
        if vendor_id < 0:
            warnings.append(f"{folder.as_posix()}: unknown manufacturer inferred -> {manufacturer}")
            continue

        template_path = _find_template_path(folder, setup_name)
        if template_path is None:
            warnings.append(f"{folder.as_posix()}: missing template echo_name for {setup_name}")

        record = FolderRecord(
            folder_id=folder.relative_to(dataset_root).as_posix(),
            folder_path=folder,
            folder_name=folder.name,
            manufacturer=manufacturer,
            vendor_id=vendor_id,
            model_name=folder.name,
            fss_path=fss_path,
            setup_name=setup_name,
            setup_numeric_id=_parse_setup_numeric_id(setup_name),
            fss_version=fss_data["fss_version"],
            fss_rect_line=int(fss_data["fss_rect_line"]),
            fss_id_echo=fss_data["fss_id_echo"],
            fss_id_probe=fss_data["fss_id_probe"],
            fss_video_x=fss_data["fss_video_x"],
            fss_video_y=fss_data["fss_video_y"],
            rect_echo_top=int(fss_data["rect_echo_top"]),
            rect_echo_left=int(fss_data["rect_echo_left"]),
            rect_echo_bottom=int(fss_data["rect_echo_bottom"]),
            rect_echo_right=int(fss_data["rect_echo_right"]),
            bbox_source=bbox_source,
            bbox_top=bbox_top,
            bbox_left=bbox_left,
            bbox_bottom=bbox_bottom,
            bbox_right=bbox_right,
            template_path=template_path,
            template_exists=template_path is not None and template_path.is_file(),
            image_paths=image_paths,
            sample_image_width=sample_w,
            sample_image_height=sample_h,
        )
        records.append(record)

    return records, warnings


def _manufacturer_min_requirements(num_folders: int) -> Dict[str, int]:
    if num_folders <= 1:
        return {"train": 1, "val": 0, "test": 0}
    if num_folders == 2:
        return {"train": 1, "val": 1, "test": 0}
    return {"train": 1, "val": 1, "test": 1}


def _split_targets(total: int, ratios: Dict[str, float]) -> Dict[str, float]:
    return {split: total * ratios[split] for split in SPLITS}


def assign_splits(
    records: List[FolderRecord],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    by_vendor: Dict[str, List[FolderRecord]] = defaultdict(list)
    for record in records:
        by_vendor[record.manufacturer].append(record)

    assignments: Dict[str, str] = {}
    for vendor in sorted(by_vendor):
        group = list(by_vendor[vendor])
        rng.shuffle(group)
        group.sort(key=lambda rec: (-rec.image_count, rec.folder_name))

        min_req = _manufacturer_min_requirements(len(group))
        image_targets = _split_targets(sum(rec.image_count for rec in group), ratios)
        image_counts = {split: 0 for split in SPLITS}
        folder_counts = {split: 0 for split in SPLITS}

        first = group[0]
        assignments[first.folder_id] = "train"
        image_counts["train"] += first.image_count
        folder_counts["train"] += 1

        for idx, record in enumerate(group[1:], start=1):
            remaining = len(group) - idx
            missing_min = [s for s in SPLITS if folder_counts[s] < min_req[s]]
            if missing_min and remaining <= len(missing_min):
                candidate_splits = missing_min
            else:
                candidate_splits = list(SPLITS)

            best_split = None
            best_score = None
            for split in candidate_splits:
                deficit = image_targets[split] - image_counts[split]
                score = deficit
                if folder_counts[split] < min_req[split]:
                    score += 1_000_000_000.0
                tie_break = -image_counts[split]
                candidate = (score, tie_break)
                if best_score is None or candidate > best_score:
                    best_score = candidate
                    best_split = split

            assignments[record.folder_id] = str(best_split)
            image_counts[str(best_split)] += record.image_count
            folder_counts[str(best_split)] += 1

    return assignments


def _write_manifest(
    output_csv: Path,
    records: List[FolderRecord],
    assignments: Dict[str, str],
    negative_re: re.Pattern[str],
    per_image_size: bool,
) -> Dict[str, int]:
    fields = [
        "image_path",
        "split",
        "group_id",
        "dataset_folder",
        "dataset_folder_path",
        "manufacturer",
        "vendor_id",
        "model_name",
        "setup_name",
        "setup_numeric_id",
        "template_path",
        "template_exists",
        "has_template",
        "bbox_source",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_norm_xmin",
        "bbox_norm_ymin",
        "bbox_norm_xmax",
        "bbox_norm_ymax",
        "rect_name_echo_top",
        "rect_name_echo_left",
        "rect_name_echo_bottom",
        "rect_name_echo_right",
        "rect_echo_top",
        "rect_echo_left",
        "rect_echo_bottom",
        "rect_echo_right",
        "expected_bbox_top",
        "expected_bbox_left",
        "expected_bbox_bottom",
        "expected_bbox_right",
        "image_width",
        "image_height",
        "bbox_within_image",
        "fss_path",
        "fss_version",
        "fss_rect_line",
        "fss_id_echo",
        "fss_id_probe",
        "fss_video_x",
        "fss_video_y",
    ]

    counters = Counter()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for record in sorted(records, key=lambda rec: rec.folder_name.lower()):
            split = assignments[record.folder_id]
            for image_path in record.image_paths:
                if per_image_size:
                    with Image.open(image_path) as img:
                        width, height = img.size
                else:
                    width = record.sample_image_width
                    height = record.sample_image_height

                has_template = 0 if negative_re.search(image_path.name) else 1
                within = int(
                    0 <= record.bbox_left < record.bbox_right <= width
                    and 0 <= record.bbox_top < record.bbox_bottom <= height
                )

                # For negative images, bbox target is intentionally empty.
                if has_template == 1:
                    bbox_left = record.bbox_left
                    bbox_top = record.bbox_top
                    bbox_right = record.bbox_right
                    bbox_bottom = record.bbox_bottom
                    bbox_norm_xmin = bbox_left / width
                    bbox_norm_ymin = bbox_top / height
                    bbox_norm_xmax = bbox_right / width
                    bbox_norm_ymax = bbox_bottom / height
                    counters["positive_images"] += 1
                else:
                    bbox_left = ""
                    bbox_top = ""
                    bbox_right = ""
                    bbox_bottom = ""
                    bbox_norm_xmin = ""
                    bbox_norm_ymin = ""
                    bbox_norm_xmax = ""
                    bbox_norm_ymax = ""
                    counters["negative_images"] += 1

                counters[f"split_{split}"] += 1
                counters[f"vendor_{record.manufacturer}"] += 1

                writer.writerow(
                    {
                        "image_path": image_path.as_posix(),
                        "split": split,
                        "group_id": record.folder_id,
                        "dataset_folder": record.folder_name,
                        "dataset_folder_path": record.folder_path.as_posix(),
                        "manufacturer": record.manufacturer,
                        "vendor_id": record.vendor_id,
                        "model_name": record.model_name,
                        "setup_name": record.setup_name,
                        "setup_numeric_id": record.setup_numeric_id if record.setup_numeric_id is not None else "",
                        "template_path": record.template_path.as_posix() if record.template_path else "",
                        "template_exists": 1 if record.template_exists else 0,
                        "has_template": has_template,
                        "bbox_source": record.bbox_source,
                        "bbox_top": bbox_top,
                        "bbox_left": bbox_left,
                        "bbox_bottom": bbox_bottom,
                        "bbox_right": bbox_right,
                        "bbox_xmin": bbox_left,
                        "bbox_ymin": bbox_top,
                        "bbox_xmax": bbox_right,
                        "bbox_ymax": bbox_bottom,
                        "bbox_norm_xmin": bbox_norm_xmin,
                        "bbox_norm_ymin": bbox_norm_ymin,
                        "bbox_norm_xmax": bbox_norm_xmax,
                        "bbox_norm_ymax": bbox_norm_ymax,
                        "rect_name_echo_top": record.bbox_top,
                        "rect_name_echo_left": record.bbox_left,
                        "rect_name_echo_bottom": record.bbox_bottom,
                        "rect_name_echo_right": record.bbox_right,
                        "rect_echo_top": record.rect_echo_top,
                        "rect_echo_left": record.rect_echo_left,
                        "rect_echo_bottom": record.rect_echo_bottom,
                        "rect_echo_right": record.rect_echo_right,
                        "expected_bbox_top": record.bbox_top,
                        "expected_bbox_left": record.bbox_left,
                        "expected_bbox_bottom": record.bbox_bottom,
                        "expected_bbox_right": record.bbox_right,
                        "image_width": width,
                        "image_height": height,
                        "bbox_within_image": within,
                        "fss_path": record.fss_path.as_posix(),
                        "fss_version": record.fss_version or "",
                        "fss_rect_line": record.fss_rect_line,
                        "fss_id_echo": record.fss_id_echo if record.fss_id_echo is not None else "",
                        "fss_id_probe": record.fss_id_probe if record.fss_id_probe is not None else "",
                        "fss_video_x": record.fss_video_x if record.fss_video_x is not None else "",
                        "fss_video_y": record.fss_video_y if record.fss_video_y is not None else "",
                    }
                )

    return dict(counters)


def _write_folder_table(output_csv: Path, records: List[FolderRecord], assignments: Dict[str, str]) -> None:
    fields = [
        "dataset_folder",
        "dataset_folder_path",
        "group_id",
        "split",
        "manufacturer",
        "vendor_id",
        "model_name",
        "image_count",
        "setup_name",
        "setup_numeric_id",
        "template_path",
        "template_exists",
        "fss_path",
        "fss_rect_line",
        "fss_id_echo",
        "fss_id_probe",
        "bbox_source",
        "rect_name_echo_top",
        "rect_name_echo_left",
        "rect_name_echo_bottom",
        "rect_name_echo_right",
        "rect_echo_top",
        "rect_echo_left",
        "rect_echo_bottom",
        "rect_echo_right",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(records, key=lambda x: x.folder_name.lower()):
            writer.writerow(
                {
                    "dataset_folder": rec.folder_name,
                    "dataset_folder_path": rec.folder_path.as_posix(),
                    "group_id": rec.folder_id,
                    "split": assignments[rec.folder_id],
                    "manufacturer": rec.manufacturer,
                    "vendor_id": rec.vendor_id,
                    "model_name": rec.model_name,
                    "image_count": rec.image_count,
                    "setup_name": rec.setup_name,
                    "setup_numeric_id": rec.setup_numeric_id if rec.setup_numeric_id is not None else "",
                    "template_path": rec.template_path.as_posix() if rec.template_path else "",
                    "template_exists": 1 if rec.template_exists else 0,
                    "fss_path": rec.fss_path.as_posix(),
                    "fss_rect_line": rec.fss_rect_line,
                    "fss_id_echo": rec.fss_id_echo if rec.fss_id_echo is not None else "",
                    "fss_id_probe": rec.fss_id_probe if rec.fss_id_probe is not None else "",
                    "bbox_source": rec.bbox_source,
                    "rect_name_echo_top": rec.bbox_top,
                    "rect_name_echo_left": rec.bbox_left,
                    "rect_name_echo_bottom": rec.bbox_bottom,
                    "rect_name_echo_right": rec.bbox_right,
                    "rect_echo_top": rec.rect_echo_top,
                    "rect_echo_left": rec.rect_echo_left,
                    "rect_echo_bottom": rec.rect_echo_bottom,
                    "rect_echo_right": rec.rect_echo_right,
                    "bbox_top": rec.bbox_top,
                    "bbox_left": rec.bbox_left,
                    "bbox_bottom": rec.bbox_bottom,
                    "bbox_right": rec.bbox_right,
                }
            )


def _build_summary(
    records: List[FolderRecord],
    assignments: Dict[str, str],
    manifest_counters: Dict[str, int],
    warnings: List[str],
    ratios: Dict[str, float],
) -> Dict[str, object]:
    folders_per_split = {split: 0 for split in SPLITS}
    images_per_split = {split: 0 for split in SPLITS}
    vendor_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"folders": 0, "images": 0, "train": 0, "val": 0, "test": 0})

    template_missing_folders = 0
    unknown_vendor_folders = 0
    bbox_line11_fallback_folders = 0

    for rec in records:
        split = assignments[rec.folder_id]
        folders_per_split[split] += 1
        images_per_split[split] += rec.image_count
        vendor_stats[rec.manufacturer]["folders"] += 1
        vendor_stats[rec.manufacturer]["images"] += rec.image_count
        vendor_stats[rec.manufacturer][split] += 1

        if not rec.template_exists:
            template_missing_folders += 1
        if rec.vendor_id < 0:
            unknown_vendor_folders += 1
        if rec.bbox_source != "line13":
            bbox_line11_fallback_folders += 1

    vendors_missing_train = [
        vendor
        for vendor, stats in sorted(vendor_stats.items())
        if stats["train"] < 1
    ]

    total_images = sum(images_per_split.values())
    image_ratio_actual = {
        split: (images_per_split[split] / total_images if total_images else 0.0)
        for split in SPLITS
    }

    return {
        "total_folders": len(records),
        "total_images": total_images,
        "folders_per_split": folders_per_split,
        "images_per_split": images_per_split,
        "image_ratio_target": ratios,
        "image_ratio_actual": image_ratio_actual,
        "positive_images": manifest_counters.get("positive_images", 0),
        "negative_images": manifest_counters.get("negative_images", 0),
        "template_missing_folders": template_missing_folders,
        "unknown_vendor_folders": unknown_vendor_folders,
        "bbox_source_main": "line13",
        "bbox_line11_fallback_folders": bbox_line11_fallback_folders,
        "vendors": dict(sorted(vendor_stats.items())),
        "vendors_missing_train": vendors_missing_train,
        "warnings": warnings,
    }


def _write_summary(summary_json: Path, summary_txt: Path, summary: Dict[str, object]) -> None:
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"Folders: {summary['total_folders']}")
    lines.append(f"Images: {summary['total_images']}")
    lines.append(f"Positive images: {summary['positive_images']}")
    lines.append(f"Negative images: {summary['negative_images']}")
    lines.append(f"Missing template folders: {summary['template_missing_folders']}")
    lines.append(
        f"Main bbox source: {summary['bbox_source_main']} "
        f"(line11 fallback folders={summary['bbox_line11_fallback_folders']})"
    )
    lines.append("")
    lines.append("Split images:")
    for split in SPLITS:
        lines.append(
            f"- {split}: {summary['images_per_split'][split]} "
            f"(target={summary['image_ratio_target'][split]:.2%}, actual={summary['image_ratio_actual'][split]:.2%})"
        )
    lines.append("")
    lines.append("Split folders:")
    for split in SPLITS:
        lines.append(f"- {split}: {summary['folders_per_split'][split]}")
    lines.append("")
    lines.append("Vendors:")
    for vendor, stats in summary["vendors"].items():
        lines.append(
            f"- {vendor}: folders={stats['folders']} images={stats['images']} "
            f"(train/val/test folders={stats['train']}/{stats['val']}/{stats['test']})"
        )

    if summary["vendors_missing_train"]:
        lines.append("")
        lines.append("Vendors missing in train: " + ", ".join(summary["vendors_missing_train"]))

    warnings = summary.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append(f"Warnings ({len(warnings)}):")
        lines.extend(f"- {w}" for w in warnings[:200])
        if len(warnings) > 200:
            lines.append(f"... +{len(warnings) - 200} more")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build vendor-template manifest (bbox + template path) with folder-level split."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n1"),
        help="Dataset root containing folders with image_samples + DB_setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/vendor_template_dataset_v1"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.65, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio.")
    parser.add_argument(
        "--exclude-folder-regex",
        type=str,
        default=r"(?i)(non\s*usare|sbagliat|griglia\s*strana)",
        help="Regex to exclude folders by name.",
    )
    parser.add_argument(
        "--negative-image-regex",
        type=str,
        default=NEGATIVE_RE_DEFAULT,
        help="Regex to mark image as negative (has_template=0).",
    )
    parser.add_argument(
        "--allow-line11-fallback",
        action="store_true",
        help=(
            "If enabled, folders with missing/unparseable line13 are kept using line11 bbox. "
            "Default is strict line13 (skip those folders)."
        ),
    )
    parser.add_argument(
        "--per-image-size",
        action="store_true",
        help=(
            "If set, open each image to read true dimensions (slower). "
            "Default uses first image size per folder for speed."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.6f}")

    ratios = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
    }

    records, warnings = scan_dataset(
        dataset_root=dataset_root,
        exclude_folder_regex=args.exclude_folder_regex,
        strict_line13=not bool(args.allow_line11_fallback),
    )
    if not records:
        raise RuntimeError("No valid folders found.")

    assignments = assign_splits(records, ratios=ratios, seed=int(args.seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = output_dir / "manifest_vendor_template.csv"
    folders_csv = output_dir / "folders_vendor_template.csv"
    summary_json = output_dir / "split_summary.json"
    summary_txt = output_dir / "split_summary.txt"
    vendor_map_json = output_dir / "vendor_id_map.json"

    negative_re = re.compile(args.negative_image_regex)
    manifest_counters = _write_manifest(
        manifest_csv,
        records,
        assignments,
        negative_re=negative_re,
        per_image_size=bool(args.per_image_size),
    )
    _write_folder_table(folders_csv, records, assignments)

    summary = _build_summary(
        records=records,
        assignments=assignments,
        manifest_counters=manifest_counters,
        warnings=warnings,
        ratios=ratios,
    )
    _write_summary(summary_json, summary_txt, summary)
    vendor_map_json.write_text(json.dumps(VENDOR_TO_ID, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Manifest: {manifest_csv}")
    print(f"Folders:  {folders_csv}")
    print(f"Summary:  {summary_json}")
    print(f"Vendor map: {vendor_map_json}")
    print(
        "Images split train/val/test: "
        f"{summary['images_per_split']['train']}/"
        f"{summary['images_per_split']['val']}/"
        f"{summary['images_per_split']['test']}"
    )
    print(
        "Positives/negatives: "
        f"{summary['positive_images']}/{summary['negative_images']}"
    )

    if summary["vendors_missing_train"]:
        print("ERROR: vendors missing in train -> " + ", ".join(summary["vendors_missing_train"]))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
