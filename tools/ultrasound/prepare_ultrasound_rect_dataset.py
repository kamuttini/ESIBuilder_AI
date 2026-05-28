#!/usr/bin/env python3
"""Build manifest and balanced split for ultrasound rectangle detection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ACQUISITION_MEDIA_SUFFIXES = IMAGE_SUFFIXES | {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".dcm",
    ".bin",
    ".raw",
}
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
INT_RE = re.compile(r"^-?\d+$")
VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
CAPTURE_RES_TOKEN_RE = re.compile(r"^(\d{3,5})[xX](\d{3,5})$")
CAPTURE_RES_ANY_RE = re.compile(r"(\d{3,5})[xX](\d{3,5})")
CAPTURE_INPUT_TOKENS = {"hdmi", "vga"}


@dataclass(frozen=True)
class FolderRecord:
    folder_id: str
    folder_path: Path
    folder_name: str
    manufacturer: str
    model_name: str
    fss_path: Path
    fss_version: Optional[str]
    fss_rect_line: int
    fss_id_echo: Optional[int]
    fss_id_probe: Optional[int]
    fss_video_x: Optional[int]
    fss_video_y: Optional[int]
    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    image_paths: Tuple[Path, ...]
    sample_image_width: int
    sample_image_height: int
    capture_video_input: Optional[str]
    capture_video_x: Optional[int]
    capture_video_y: Optional[int]
    capture_support_count: int
    capture_parse_source: str

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


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    cleaned = value.strip()
    if INT_RE.match(cleaned):
        return int(cleaned)
    return None


def _video_input_code_0hdmi_1vga(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "hdmi":
        return 0
    if lowered == "vga":
        return 1
    return None


def _parse_capture_metadata_from_filename(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int], str]:
    stem = path.stem
    tokens = [tok for tok in re.split(r"[_\s-]+", stem) if tok]

    input_label: Optional[str] = None
    input_idx = -1
    for idx, tok in enumerate(tokens):
        lowered = tok.lower()
        if lowered in CAPTURE_INPUT_TOKENS:
            input_label = lowered
            input_idx = idx
            break

    width: Optional[int] = None
    height: Optional[int] = None
    source_parts: List[str] = []

    if input_label is not None:
        source_parts.append("input_token")
        if input_idx + 1 < len(tokens):
            match = CAPTURE_RES_TOKEN_RE.match(tokens[input_idx + 1])
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                source_parts.append("next_token_resolution")

    if width is None or height is None:
        match_any = CAPTURE_RES_ANY_RE.search(stem)
        if match_any:
            width = int(match_any.group(1))
            height = int(match_any.group(2))
            source_parts.append("fallback_resolution_search")

    source = "+".join(source_parts)
    return input_label, width, height, source


def _infer_folder_capture_metadata(
    candidate_paths: Iterable[Path],
) -> Tuple[Optional[str], Optional[int], Optional[int], int]:
    histogram: Dict[Tuple[str, int, int], int] = defaultdict(int)

    for path in candidate_paths:
        input_label, width, height, _ = _parse_capture_metadata_from_filename(path)
        if input_label is None and (width is None or height is None):
            continue
        key = (input_label or "", int(width or 0), int(height or 0))
        histogram[key] += 1

    if not histogram:
        return None, None, None, 0

    best_key, count = max(
        histogram.items(),
        key=lambda item: (item[1], item[0][0] == "hdmi", item[0][1] * item[0][2]),
    )
    input_label = best_key[0] if best_key[0] else None
    width = best_key[1] if best_key[1] > 0 else None
    height = best_key[2] if best_key[2] > 0 else None
    return input_label, width, height, count


def _collect_acquisition_candidate_paths(folder: Path) -> Tuple[Path, ...]:
    candidates: List[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ACQUISITION_MEDIA_SUFFIXES:
            continue
        rel = path.relative_to(folder)
        if rel.parts and rel.parts[0] == "image_samples":
            continue
        candidates.append(path)
    return tuple(candidates)


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
    raise ValueError("RECT_ECHO not found in first 15 lines.")


def _safe_line(lines: List[str], line_num: int) -> Optional[str]:
    if 1 <= line_num <= len(lines):
        return lines[line_num - 1].strip()
    return None


def parse_fss(path: Path) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rect_line, top, left, bottom, right = _extract_rect_line(lines)
    offset = rect_line - 11

    version_line = _safe_line(lines, 1 + offset)
    version = version_line if version_line and VERSION_RE.match(version_line) else None

    id_echo = _parse_int(_safe_line(lines, 2 + offset))
    id_probe = _parse_int(_safe_line(lines, 3 + offset))
    video_x = _parse_int(_safe_line(lines, 9 + offset))
    video_y = _parse_int(_safe_line(lines, 10 + offset))

    return {
        "fss_rect_line": rect_line,
        "fss_version": version,
        "fss_id_echo": id_echo,
        "fss_id_probe": id_probe,
        "fss_video_x": video_x,
        "fss_video_y": video_y,
        "bbox_top": top,
        "bbox_left": left,
        "bbox_bottom": bottom,
        "bbox_right": right,
    }


def _collect_image_paths(image_dir: Path, exclude_name_re: Optional[re.Pattern[str]]) -> Tuple[Tuple[Path, ...], int]:
    kept: List[Path] = []
    excluded = 0
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if exclude_name_re and exclude_name_re.search(path.name):
            excluded += 1
            continue
        kept.append(path)
    return tuple(kept), excluded


def scan_dataset(
    dataset_root: Path,
    exclude_regex: Optional[str],
    exclude_image_regex: Optional[str],
) -> Tuple[List[FolderRecord], List[str], int]:
    records: List[FolderRecord] = []
    warnings: List[str] = []
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    exclude_image_re = re.compile(exclude_image_regex) if exclude_image_regex else None
    excluded_images_total = 0

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        if exclude_re and exclude_re.search(folder.name):
            continue

        image_dir = folder / "image_samples"
        setup_dir = folder / "DB_setup"
        if not image_dir.is_dir() or not setup_dir.is_dir():
            continue

        images, excluded_in_folder = _collect_image_paths(image_dir, exclude_image_re)
        excluded_images_total += excluded_in_folder
        if not images:
            if excluded_in_folder > 0:
                warnings.append(
                    f"{folder.name}: nessuna immagine valida in image_samples "
                    f"(escluse {excluded_in_folder} per filtro nome)."
                )
            else:
                warnings.append(f"{folder.name}: nessuna immagine in image_samples.")
            continue

        fss_candidates = sorted(p for p in setup_dir.glob("*.fss") if not p.name.startswith("~$"))
        if len(fss_candidates) != 1:
            warnings.append(
                f"{folder.name}: atteso 1 file .fss in DB_setup, trovati {len(fss_candidates)}."
            )
            continue

        fss_path = fss_candidates[0]
        try:
            fss_data = parse_fss(fss_path)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"{folder.name}: errore parsing {fss_path.name}: {exc}")
            continue

        sample_image = images[0]
        with Image.open(sample_image) as img:
            sample_w, sample_h = img.size
        acquisition_candidates = _collect_acquisition_candidate_paths(folder)
        (
            capture_video_input,
            capture_video_x,
            capture_video_y,
            capture_support_count,
        ) = _infer_folder_capture_metadata(acquisition_candidates)
        if capture_support_count > 0:
            capture_parse_source = "acquisition_filename_majority_vote"
        else:
            (
                capture_video_input,
                capture_video_x,
                capture_video_y,
                capture_support_count,
            ) = _infer_folder_capture_metadata(images)
            capture_parse_source = (
                "image_filename_fallback_majority_vote" if capture_support_count > 0 else ""
            )

        record = FolderRecord(
            folder_id=folder.relative_to(dataset_root).as_posix(),
            folder_path=folder,
            folder_name=folder.name,
            manufacturer=infer_manufacturer(folder.name),
            model_name=folder.name,
            fss_path=fss_path,
            fss_version=fss_data["fss_version"],
            fss_rect_line=int(fss_data["fss_rect_line"]),
            fss_id_echo=fss_data["fss_id_echo"],
            fss_id_probe=fss_data["fss_id_probe"],
            fss_video_x=fss_data["fss_video_x"],
            fss_video_y=fss_data["fss_video_y"],
            bbox_top=int(fss_data["bbox_top"]),
            bbox_left=int(fss_data["bbox_left"]),
            bbox_bottom=int(fss_data["bbox_bottom"]),
            bbox_right=int(fss_data["bbox_right"]),
            image_paths=images,
            sample_image_width=sample_w,
            sample_image_height=sample_h,
            capture_video_input=capture_video_input,
            capture_video_x=capture_video_x,
            capture_video_y=capture_video_y,
            capture_support_count=capture_support_count,
            capture_parse_source=capture_parse_source,
        )
        records.append(record)

    return records, warnings, excluded_images_total


def _manufacturer_min_requirements(num_models: int) -> Dict[str, int]:
    if num_models <= 1:
        return {"train": 1, "val": 0, "test": 0}
    if num_models == 2:
        return {"train": 1, "val": 1, "test": 0}
    return {"train": 1, "val": 1, "test": 1}


def _split_targets(total: int, ratios: Dict[str, float]) -> Dict[str, float]:
    return {split: total * ratios[split] for split in SPLITS}


def assign_splits(
    records: List[FolderRecord],
    ratios: Dict[str, float],
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    by_manufacturer: Dict[str, List[FolderRecord]] = defaultdict(list)
    for record in records:
        by_manufacturer[record.manufacturer].append(record)

    assignments: Dict[str, str] = {}
    min_requirements_by_manufacturer: Dict[str, Dict[str, int]] = {}

    for manufacturer in sorted(by_manufacturer):
        group = list(by_manufacturer[manufacturer])
        rng.shuffle(group)
        group.sort(key=lambda rec: (-rec.image_count, rec.folder_name))

        min_requirements = _manufacturer_min_requirements(len(group))
        min_requirements_by_manufacturer[manufacturer] = min_requirements

        image_targets = _split_targets(sum(rec.image_count for rec in group), ratios)
        image_counts = {split: 0 for split in SPLITS}
        folder_counts = {split: 0 for split in SPLITS}

        lead = group[0]
        assignments[lead.folder_id] = "train"
        image_counts["train"] += lead.image_count
        folder_counts["train"] += 1

        for idx, record in enumerate(group[1:], start=1):
            remaining_including_current = len(group) - idx
            missing_min = [split for split in SPLITS if folder_counts[split] < min_requirements[split]]
            if missing_min and remaining_including_current <= len(missing_min):
                candidate_splits = missing_min
            else:
                candidate_splits = list(SPLITS)

            best_split = None
            best_score = None
            for split in candidate_splits:
                deficit = image_targets[split] - image_counts[split]
                score = deficit
                if folder_counts[split] < min_requirements[split]:
                    score += 1_000_000_000.0
                tie_break = -image_counts[split]
                candidate = (score, tie_break)
                if best_score is None or candidate > best_score:
                    best_score = candidate
                    best_split = split

            assert best_split is not None
            assignments[record.folder_id] = best_split
            image_counts[best_split] += record.image_count
            folder_counts[best_split] += 1

    _rebalance_global_ratios(records, assignments, ratios, min_requirements_by_manufacturer)
    return assignments, min_requirements_by_manufacturer


def _rebalance_global_ratios(
    records: List[FolderRecord],
    assignments: Dict[str, str],
    ratios: Dict[str, float],
    min_requirements_by_manufacturer: Dict[str, Dict[str, int]],
) -> None:
    by_id = {rec.folder_id: rec for rec in records}
    total_images = sum(rec.image_count for rec in records)
    global_targets = _split_targets(total_images, ratios)

    image_counts = {split: 0 for split in SPLITS}
    manufacturer_folder_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {split: 0 for split in SPLITS})
    split_to_folder_ids: Dict[str, List[str]] = {split: [] for split in SPLITS}

    for folder_id, split in assignments.items():
        record = by_id[folder_id]
        image_counts[split] += record.image_count
        manufacturer_folder_counts[record.manufacturer][split] += 1
        split_to_folder_ids[split].append(folder_id)

    for split in SPLITS:
        split_to_folder_ids[split].sort(key=lambda folder_id: by_id[folder_id].image_count)

    tolerance = max(1.0, total_images * 0.005)

    for _ in range(5000):
        deficits = {split: global_targets[split] - image_counts[split] for split in SPLITS}
        split_under = max(SPLITS, key=lambda split: deficits[split])
        split_over = min(SPLITS, key=lambda split: deficits[split])

        if deficits[split_under] <= tolerance and -deficits[split_over] <= tolerance:
            break
        if deficits[split_under] <= 0 or deficits[split_over] >= 0:
            break

        transfer_need = min(deficits[split_under], -deficits[split_over])
        candidates: List[FolderRecord] = []
        for folder_id in split_to_folder_ids[split_over]:
            rec = by_id[folder_id]
            manufacturer_counts = manufacturer_folder_counts[rec.manufacturer]
            if (
                manufacturer_counts[split_over] - 1
                < min_requirements_by_manufacturer[rec.manufacturer][split_over]
            ):
                continue
            candidates.append(rec)

        if not candidates:
            break

        move_rec = min(
            candidates,
            key=lambda rec: (
                abs(rec.image_count - transfer_need),
                rec.image_count,
                rec.folder_name,
            ),
        )
        folder_id = move_rec.folder_id

        assignments[folder_id] = split_under
        split_to_folder_ids[split_over].remove(folder_id)
        split_to_folder_ids[split_under].append(folder_id)
        split_to_folder_ids[split_under].sort(key=lambda fid: by_id[fid].image_count)

        image_counts[split_over] -= move_rec.image_count
        image_counts[split_under] += move_rec.image_count

        manufacturer_folder_counts[move_rec.manufacturer][split_over] -= 1
        manufacturer_folder_counts[move_rec.manufacturer][split_under] += 1


def _write_manifest(
    output_csv: Path,
    dataset_root: Path,
    records: List[FolderRecord],
    assignments: Dict[str, str],
) -> None:
    del dataset_root
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "fss_version",
        "fss_rect_line",
        "fss_id_echo",
        "fss_id_probe",
        "fss_video_x",
        "fss_video_y",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "capture_video_input",
        "capture_video_input_code_0hdmi_1vga",
        "capture_video_x",
        "capture_video_y",
        "capture_parse_source",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda rec: rec.folder_name.lower()):
            split = assignments[record.folder_id]
            for image_path in record.image_paths:
                if record.capture_parse_source.startswith("acquisition_filename"):
                    capture_input = record.capture_video_input
                    capture_video_x = record.capture_video_x
                    capture_video_y = record.capture_video_y
                    parse_source = record.capture_parse_source
                else:
                    (
                        img_input,
                        img_video_x,
                        img_video_y,
                        img_parse_source,
                    ) = _parse_capture_metadata_from_filename(image_path)
                    capture_input = img_input or record.capture_video_input
                    capture_video_x = img_video_x if img_video_x is not None else record.capture_video_x
                    capture_video_y = img_video_y if img_video_y is not None else record.capture_video_y
                    parse_source = img_parse_source or record.capture_parse_source
                writer.writerow(
                    {
                        "image_path": image_path.as_posix(),
                        "split": split,
                        "dataset_folder": record.folder_name,
                        "manufacturer": record.manufacturer,
                        "model_name": record.model_name,
                        "fss_path": record.fss_path.as_posix(),
                        "fss_version": record.fss_version or "",
                        "fss_rect_line": record.fss_rect_line,
                        "fss_id_echo": record.fss_id_echo if record.fss_id_echo is not None else "",
                        "fss_id_probe": record.fss_id_probe if record.fss_id_probe is not None else "",
                        "fss_video_x": record.fss_video_x if record.fss_video_x is not None else "",
                        "fss_video_y": record.fss_video_y if record.fss_video_y is not None else "",
                        "bbox_top": record.bbox_top,
                        "bbox_left": record.bbox_left,
                        "bbox_bottom": record.bbox_bottom,
                        "bbox_right": record.bbox_right,
                        "bbox_xmin": record.bbox_left,
                        "bbox_ymin": record.bbox_top,
                        "bbox_xmax": record.bbox_right,
                        "bbox_ymax": record.bbox_bottom,
                        "capture_video_input": capture_input or "",
                        "capture_video_input_code_0hdmi_1vga": (
                            _video_input_code_0hdmi_1vga(capture_input)
                            if capture_input is not None
                            else ""
                        ),
                        "capture_video_x": capture_video_x if capture_video_x is not None else "",
                        "capture_video_y": capture_video_y if capture_video_y is not None else "",
                        "capture_parse_source": parse_source,
                    }
                )


def _write_folder_table(
    output_csv: Path,
    records: List[FolderRecord],
    assignments: Dict[str, str],
) -> None:
    fields = [
        "dataset_folder",
        "split",
        "manufacturer",
        "model_name",
        "image_count",
        "fss_path",
        "fss_rect_line",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "fss_video_x",
        "fss_video_y",
        "capture_video_input",
        "capture_video_input_code_0hdmi_1vga",
        "capture_video_x",
        "capture_video_y",
        "capture_support_count",
        "capture_parse_source",
        "sample_image_width",
        "sample_image_height",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda rec: rec.folder_name.lower()):
            writer.writerow(
                {
                    "dataset_folder": record.folder_name,
                    "split": assignments[record.folder_id],
                    "manufacturer": record.manufacturer,
                    "model_name": record.model_name,
                    "image_count": record.image_count,
                    "fss_path": record.fss_path.as_posix(),
                    "fss_rect_line": record.fss_rect_line,
                    "bbox_top": record.bbox_top,
                    "bbox_left": record.bbox_left,
                    "bbox_bottom": record.bbox_bottom,
                    "bbox_right": record.bbox_right,
                    "fss_video_x": record.fss_video_x if record.fss_video_x is not None else "",
                    "fss_video_y": record.fss_video_y if record.fss_video_y is not None else "",
                    "capture_video_input": record.capture_video_input or "",
                    "capture_video_input_code_0hdmi_1vga": (
                        _video_input_code_0hdmi_1vga(record.capture_video_input)
                        if record.capture_video_input is not None
                        else ""
                    ),
                    "capture_video_x": record.capture_video_x if record.capture_video_x is not None else "",
                    "capture_video_y": record.capture_video_y if record.capture_video_y is not None else "",
                    "capture_support_count": record.capture_support_count,
                    "capture_parse_source": record.capture_parse_source,
                    "sample_image_width": record.sample_image_width,
                    "sample_image_height": record.sample_image_height,
                }
            )


def build_summary(
    records: List[FolderRecord],
    assignments: Dict[str, str],
    ratios: Dict[str, float],
    warnings: List[str],
    excluded_images_by_name_filter: int,
) -> Dict[str, object]:
    images_per_split = {split: 0 for split in SPLITS}
    folders_per_split = {split: 0 for split in SPLITS}
    by_manufacturer: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {
            "folders": {split: 0 for split in SPLITS},
            "images": {split: 0 for split in SPLITS},
            "total_folders": 0,
            "total_images": 0,
        }
    )

    total_images = 0
    bbox_out_of_sample_size = 0
    sample_size_mismatch = 0
    folders_with_capture_input = 0
    folders_with_capture_resolution = 0
    folders_with_capture_full_metadata = 0

    for rec in records:
        split = assignments[rec.folder_id]
        images_per_split[split] += rec.image_count
        folders_per_split[split] += 1
        total_images += rec.image_count

        manufacturer_stats = by_manufacturer[rec.manufacturer]
        manufacturer_stats["folders"][split] += 1
        manufacturer_stats["images"][split] += rec.image_count
        manufacturer_stats["total_folders"] += 1
        manufacturer_stats["total_images"] += rec.image_count

        if not (
            0 <= rec.bbox_left < rec.bbox_right <= rec.sample_image_width
            and 0 <= rec.bbox_top < rec.bbox_bottom <= rec.sample_image_height
        ):
            bbox_out_of_sample_size += 1

        if rec.fss_video_x is not None and rec.fss_video_y is not None:
            if rec.fss_video_x != rec.sample_image_width or rec.fss_video_y != rec.sample_image_height:
                sample_size_mismatch += 1

        has_input = rec.capture_video_input is not None
        has_resolution = rec.capture_video_x is not None and rec.capture_video_y is not None
        if has_input:
            folders_with_capture_input += 1
        if has_resolution:
            folders_with_capture_resolution += 1
        if has_input and has_resolution:
            folders_with_capture_full_metadata += 1

    image_ratios = {split: (images_per_split[split] / total_images if total_images else 0.0) for split in SPLITS}
    image_target = {split: ratios[split] for split in SPLITS}

    manufacturer_ordered: Dict[str, Dict[str, object]] = {}
    for name in sorted(by_manufacturer):
        stats = by_manufacturer[name]
        manufacturer_ordered[name] = {
            "total_folders": stats["total_folders"],
            "total_images": stats["total_images"],
            "folders": stats["folders"],
            "images": stats["images"],
        }

    manufacturers_missing_train = [
        name
        for name, stats in manufacturer_ordered.items()
        if stats["folders"]["train"] < 1
    ]

    return {
        "total_folders": len(records),
        "total_images": total_images,
        "images_excluded_by_name_filter": excluded_images_by_name_filter,
        "folders_per_split": folders_per_split,
        "images_per_split": images_per_split,
        "image_ratio_target": image_target,
        "image_ratio_actual": image_ratios,
        "manufacturers": manufacturer_ordered,
        "manufacturers_missing_train_model": manufacturers_missing_train,
        "folders_with_bbox_out_of_sample_size": bbox_out_of_sample_size,
        "folders_with_fss_image_size_mismatch": sample_size_mismatch,
        "folders_with_capture_input_from_filename": folders_with_capture_input,
        "folders_with_capture_resolution_from_filename": folders_with_capture_resolution,
        "folders_with_capture_full_metadata_from_filename": folders_with_capture_full_metadata,
        "warnings": warnings,
    }


def _write_summary(summary: Dict[str, object], output_json: Path, output_txt: Path) -> None:
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"Cartelle valide: {summary['total_folders']}")
    lines.append(f"Immagini totali: {summary['total_images']}")
    lines.append(f"Immagini escluse da filtro nome: {summary['images_excluded_by_name_filter']}")
    lines.append("")
    lines.append("Split (immagini):")
    for split in SPLITS:
        lines.append(
            f"- {split}: {summary['images_per_split'][split]} "
            f"(target {summary['image_ratio_target'][split]:.2%}, "
            f"actual {summary['image_ratio_actual'][split]:.2%})"
        )
    lines.append("")
    lines.append("Split (cartelle):")
    for split in SPLITS:
        lines.append(f"- {split}: {summary['folders_per_split'][split]}")

    lines.append("")
    lines.append("Produttori:")
    for manufacturer, stats in summary["manufacturers"].items():
        lines.append(
            f"- {manufacturer}: folders {stats['total_folders']}, images {stats['total_images']} "
            f"(train/val/test folders: {stats['folders']['train']}/"
            f"{stats['folders']['val']}/{stats['folders']['test']})"
        )

    lines.append("")
    lines.append(
        "Cartelle con bbox fuori dimensione sample: "
        f"{summary['folders_with_bbox_out_of_sample_size']}"
    )
    lines.append(
        "Cartelle con mismatch fss_video_size vs sample image size: "
        f"{summary['folders_with_fss_image_size_mismatch']}"
    )
    lines.append(
        "Cartelle con input video estratto da filename (vga/hdmi): "
        f"{summary['folders_with_capture_input_from_filename']}"
    )
    lines.append(
        "Cartelle con risoluzione estratta da filename (WxH): "
        f"{summary['folders_with_capture_resolution_from_filename']}"
    )
    lines.append(
        "Cartelle con metadata completi da filename (input+risoluzione): "
        f"{summary['folders_with_capture_full_metadata_from_filename']}"
    )
    if summary["manufacturers_missing_train_model"]:
        lines.append(
            "Produttori senza modello nel train: "
            + ", ".join(summary["manufacturers_missing_train_model"])
        )
    else:
        lines.append("Tutti i produttori hanno almeno un modello nel train.")

    warnings = summary.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append(f"Warning ({len(warnings)}):")
        lines.extend(f"- {msg}" for msg in warnings[:100])
        if len(warnings) > 100:
            lines.append(f"... altri {len(warnings) - 100} warning")

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera manifest e split bilanciato per detection rettangolo ecografico."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset"),
        help="Root cartella Dataset (default: Dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/rect_dataset"),
        help="Cartella output manifest/split (default: artifacts/20_datasets/rect_dataset).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed random.")
    parser.add_argument("--train-ratio", type=float, default=0.65, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio.")
    parser.add_argument(
        "--exclude-folder-regex",
        type=str,
        default=None,
        help="Regex opzionale per escludere cartelle dataset.",
    )
    parser.add_argument(
        "--exclude-image-regex",
        type=str,
        default="(?i)negative",
        help="Regex opzionale per escludere immagini per nome file (default: '(?i)negative').",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Le ratio devono sommare a 1.0, valore attuale: {ratio_sum:.6f}"
        )

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    records, warnings, excluded_images_total = scan_dataset(
        dataset_root,
        args.exclude_folder_regex,
        args.exclude_image_regex,
    )
    if not records:
        raise RuntimeError("Nessuna cartella valida trovata in Dataset.")

    assignments, _ = assign_splits(records, ratios, seed=args.seed)

    manifest_csv = output_dir / "manifest_rect_echo.csv"
    folders_csv = output_dir / "folders_rect_echo.csv"
    summary_json = output_dir / "split_summary.json"
    summary_txt = output_dir / "split_summary.txt"

    _write_manifest(manifest_csv, dataset_root, records, assignments)
    _write_folder_table(folders_csv, records, assignments)
    summary = build_summary(
        records,
        assignments,
        ratios,
        warnings,
        excluded_images_by_name_filter=excluded_images_total,
    )
    _write_summary(summary, summary_json, summary_txt)

    print(f"Manifest immagini: {manifest_csv}")
    print(f"Tabella cartelle: {folders_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary TXT: {summary_txt}")
    print(
        "Split immagini (train/val/test): "
        f"{summary['images_per_split']['train']}/"
        f"{summary['images_per_split']['val']}/"
        f"{summary['images_per_split']['test']}"
    )
    print(
        "Ratio immagini effettive: "
        f"{summary['image_ratio_actual']['train']:.4f}/"
        f"{summary['image_ratio_actual']['val']:.4f}/"
        f"{summary['image_ratio_actual']['test']:.4f}"
    )

    missing_train = summary["manufacturers_missing_train_model"]
    if missing_train:
        print("ATTENZIONE: produttori senza modello nel train:", ", ".join(missing_train))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
