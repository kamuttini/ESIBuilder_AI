#!/usr/bin/env python3
"""Prepare orientation-symbolic dataset aligned with legacy ESIBuilder workflow.

Outputs:
1) setup_orientation_manifest.csv
   - one row per setup x orientation (NF/LR/UD/LRUD)
   - includes availability and target envelope rect from line #16
2) frame_manifest.csv
   - one row per image frame (recursive scan)
   - includes source type and orientation hint from filename and/or path folders
3) symbol_prior_manifest.csv
   - one row per image_th_orientation_<k>_positive_* image (explicit symbol prior)

This dataset is designed to support a redesigned pipeline:
- orientation class/hint recognition
- per-orientation symbol localization
- envelope aggregation over depth/frames per orientation
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ORIENTATION_NAMES = {
    0: "NF",
    1: "LR",
    2: "UD",
    3: "LRUD",
}

RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
INT_RE = re.compile(r"^-?\d+$")
FLOAT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

IMAGE_PATTERNS = {
    "orientation_setup": re.compile(r"^image_orientation_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_nf": re.compile(r"^image_depth_find_no_flip_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lr": re.compile(r"^image_depth_find_flip_lr_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_ud": re.compile(r"^image_depth_find_flip_ud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lrud": re.compile(r"^image_depth_find_flip_lrud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "th_orientation_positive": re.compile(r"^image_th_orientation_(\d+)_positive_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
}

ORIENTATION_DIR_PATTERNS: Tuple[Tuple[int, re.Pattern[str]], ...] = (
    (3, re.compile(r"\b(lrud|fliplrud|flip\s*lr\s*ud|fliplr\s*ud|lr\s*ud)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(lr|fliplr|flip\s*lr|left\s*right)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(ud|flipud|flip\s*ud|upside\s*down)\b", re.IGNORECASE)),
    (0, re.compile(r"\b(nf|noflip|no\s*flip|flip\s*no)\b", re.IGNORECASE)),
)


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
    split: str
    fss_group_orientation: int
    fss_id_echo: int
    fss_id_probe: int
    fss_probe_type: int
    fss_video_x: int
    fss_video_y: int
    orient_available: Tuple[int, int, int, int]
    depth_orientation_map: Dict[int, int]
    depth_ycoord_map: Dict[int, int]
    entries: Tuple[OrientationEntry, OrientationEntry, OrientationEntry, OrientationEntry]


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def infer_manufacturer(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    name = re.sub(r"^\d+[.\s_-]*", "", name)
    name = re.sub(r"^[a-z][\s._-]+", "", name)
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
    try:
        return int(float(cleaned))
    except ValueError:
        return default


def _parse_float(value: str) -> Optional[float]:
    cleaned = value.strip()
    if FLOAT_RE.match(cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_rect_line(lines: List[str]) -> int:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines):
            if RECT_RE.match(lines[candidate - 1].strip()):
                return candidate
    for idx in range(1, min(len(lines), 20) + 1):
        if RECT_RE.match(lines[idx - 1].strip()):
            return idx
    raise ValueError("RECT_ECHO line not found")


def _parse_params_token(token: str) -> Tuple[int, float, float, float, float, float, float, int]:
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

    th_value = _parse_float(token)
    if th_value is not None:
        return 7, th_value, 20.0, 120.0, 0.0, 0.0, 0.0, 1

    raise ValueError(f"Cannot parse params token: {token!r}")


def _parse_line16_entries(lines: List[str], rect_line: int) -> List[OrientationEntry]:
    orientation_line = rect_line + 5
    if orientation_line > len(lines):
        raise ValueError("line #16 missing")

    raw = lines[orientation_line - 1].strip()
    if not raw:
        raise ValueError("empty line #16")

    segments = [seg.strip().rstrip(",") for seg in raw.split(";") if seg.strip()]
    entries: List[OrientationEntry] = []
    for idx, segment in enumerate(segments):
        parts = segment.split("|")
        if len(parts) < 7:
            raise ValueError(f"malformed segment: {segment!r}")

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

    if len(entries) < 4:
        raise ValueError(f"expected 4 entries, got {len(entries)}")
    return entries[:4]


def _parse_orient_file(orient_path: Path) -> Tuple[Tuple[int, int, int, int], Dict[int, int], Dict[int, int]]:
    if not orient_path.exists():
        return (1, 1, 1, 1), {}, {}

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(orient_path, encoding="utf-8")

    if "Orientations" in parser:
        sec = parser["Orientations"]

        def _get_bool(name: str) -> int:
            value = sec.get(name, "true").strip().lower()
            return 1 if value in {"1", "true", "yes", "on"} else 0

        flips = (
            _get_bool("FLIP_NO"),
            _get_bool("FLIP_LR"),
            _get_bool("FLIP_UD"),
            _get_bool("FLIP_LR_UD"),
        )
    else:
        flips = (1, 1, 1, 1)

    depth_orientation: Dict[int, int] = {}
    depth_y: Dict[int, int] = {}
    for sec_name in parser.sections():
        if not sec_name.startswith("DepthValue_"):
            continue
        try:
            depth_idx = int(sec_name.split("_", 1)[1])
        except ValueError:
            continue
        depth_orientation[depth_idx] = _parse_int(parser[sec_name].get("Orientation"), -1)
        depth_y[depth_idx] = _parse_int(parser[sec_name].get("Y_Coordinate"), -1)

    return flips, depth_orientation, depth_y


def _load_split_map(folders_line16_path: Optional[Path]) -> Dict[str, str]:
    if folders_line16_path is None:
        return {}
    path = folders_line16_path.expanduser().resolve()
    if not path.exists():
        return {}

    split_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder = row.get("dataset_folder", "").strip()
            split = row.get("split", "").strip().lower()
            if folder and split in {"train", "val", "test"}:
                split_map[folder] = split
    return split_map


def _scan_folders(dataset_roots: Sequence[Path], split_map: Dict[str, str]) -> Tuple[List[FolderRecord], List[str]]:
    records: List[FolderRecord] = []
    warnings: List[str] = []
    seen_folder_ids: set[str] = set()

    for root in dataset_roots:
        if not root.is_dir():
            warnings.append(f"Dataset root missing: {root}")
            continue

        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue

            setup_dir = folder / "DB_setup"
            image_dir = folder / "image_samples"
            if not setup_dir.is_dir() or not image_dir.is_dir():
                continue

            fss_files = sorted(p for p in setup_dir.glob("*.fss") if p.is_file())
            if len(fss_files) != 1:
                warnings.append(f"{folder.name}: expected 1 fss, found {len(fss_files)}")
                continue

            fss_path = fss_files[0]
            lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()

            try:
                rect_line = _extract_rect_line(lines)
                entries = tuple(_parse_line16_entries(lines, rect_line=rect_line))
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"{folder.name}: line16 parse failed: {exc}")
                continue

            offset = rect_line - 11
            fss_group_orientation = _parse_int(lines[rect_line] if rect_line < len(lines) else None, 4)
            fss_id_echo = _parse_int(lines[1 + offset] if (2 + offset) <= len(lines) else None, -1)
            fss_id_probe = _parse_int(lines[2 + offset] if (3 + offset) <= len(lines) else None, -1)
            fss_probe_type = _parse_int(lines[3 + offset] if (4 + offset) <= len(lines) else None, -1)
            fss_video_x = _parse_int(lines[8 + offset] if (9 + offset) <= len(lines) else None, 0)
            fss_video_y = _parse_int(lines[9 + offset] if (10 + offset) <= len(lines) else None, 0)

            orient_path = setup_dir / f"setup_{fss_path.stem.split('_')[-1]}.orient"
            flips, depth_orientation_map, depth_ycoord_map = _parse_orient_file(orient_path)

            split = split_map.get(folder.name, "train")
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
                    split=split,
                    fss_group_orientation=fss_group_orientation,
                    fss_id_echo=fss_id_echo,
                    fss_id_probe=fss_id_probe,
                    fss_probe_type=fss_probe_type,
                    fss_video_x=fss_video_x,
                    fss_video_y=fss_video_y,
                    orient_available=flips,
                    depth_orientation_map=depth_orientation_map,
                    depth_ycoord_map=depth_ycoord_map,
                    entries=entries,
                )
            )

    return records, warnings


def _match_image(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    for source, regex in IMAGE_PATTERNS.items():
        m = regex.match(name)
        if not m:
            continue
        if source == "th_orientation_positive":
            orientation_hint = int(m.group(1))
            sample_idx = int(m.group(2))
            return source, orientation_hint, sample_idx

        sample_idx = int(m.group(1))
        orientation_hint = None
        if source == "depth_find_nf":
            orientation_hint = 0
        elif source == "depth_find_lr":
            orientation_hint = 1
        elif source == "depth_find_ud":
            orientation_hint = 2
        elif source == "depth_find_lrud":
            orientation_hint = 3
        return source, orientation_hint, sample_idx

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


def _is_under_known_setup(image_path: Path, setup_roots: set[Path]) -> bool:
    for parent in image_path.parents:
        if parent in setup_roots:
            return True
    return False


def _write_setup_orientation_manifest(path: Path, records: Sequence[FolderRecord]) -> None:
    fields = [
        "setup_id",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "orientation_idx",
        "orientation_name",
        "orientation_available",
        "rect_top",
        "rect_left",
        "rect_bottom",
        "rect_right",
        "rect_width",
        "rect_height",
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
        "fss_group_orientation",
        "fss_id_echo",
        "fss_id_probe",
        "fss_probe_type",
        "fss_video_x",
        "fss_video_y",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for record in sorted(records, key=lambda r: r.folder_name.lower()):
            for idx in range(4):
                e = record.entries[idx]
                writer.writerow(
                    {
                        "setup_id": record.folder_id,
                        "split": record.split,
                        "dataset_folder": record.folder_name,
                        "manufacturer": record.manufacturer,
                        "model_name": record.model_name,
                        "fss_path": record.fss_path.as_posix(),
                        "orientation_idx": idx,
                        "orientation_name": ORIENTATION_NAMES[idx],
                        "orientation_available": record.orient_available[idx],
                        "rect_top": e.top,
                        "rect_left": e.left,
                        "rect_bottom": e.bottom,
                        "rect_right": e.right,
                        "rect_width": float(e.right - e.left + 1),
                        "rect_height": float(e.bottom - e.top + 1),
                        "b_value": e.b_value,
                        "channel": e.channel,
                        "threshold": e.threshold,
                        "p1": e.p1,
                        "p2": e.p2,
                        "p3": e.p3,
                        "p4": e.p4,
                        "p5": e.p5,
                        "mm": e.mm,
                        "legacy_threshold_only": e.legacy_threshold_only,
                        "fss_group_orientation": record.fss_group_orientation,
                        "fss_id_echo": record.fss_id_echo,
                        "fss_id_probe": record.fss_id_probe,
                        "fss_probe_type": record.fss_probe_type,
                        "fss_video_x": record.fss_video_x,
                        "fss_video_y": record.fss_video_y,
                    }
                )


def _write_frame_manifest(
    path: Path,
    records: Sequence[FolderRecord],
    dataset_roots: Sequence[Path],
    split_map: Dict[str, str],
    include_other: bool,
    include_path_orientation_frames: bool,
) -> Tuple[int, int, int, int]:
    fields = [
        "setup_id",
        "split",
        "dataset_folder",
        "manufacturer",
        "model_name",
        "fss_path",
        "image_path",
        "image_name",
        "source_type",
        "sample_idx",
        "orientation_hint",
        "orientation_hint_name",
        "orientation_available_for_hint",
        "target_rect_top",
        "target_rect_left",
        "target_rect_bottom",
        "target_rect_right",
        "target_mm",
        "depth_orient_value",
        "depth_y_coordinate",
    ]

    total_rows = 0
    supervised_rows = 0
    path_hint_rows = 0
    orphan_rows = 0
    seen_images: set[str] = set()
    setup_roots = {record.folder_path.resolve() for record in records}

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for record in sorted(records, key=lambda r: r.folder_name.lower()):
            image_dir = record.folder_path / "image_samples"
            if not image_dir.is_dir():
                continue

            for image_path in sorted(p for p in image_dir.rglob("*") if p.is_file()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                source_type, orientation_hint, sample_idx = _match_image(image_path.name)
                if orientation_hint is None and include_path_orientation_frames:
                    path_hint = _infer_orientation_from_path(image_path.relative_to(record.folder_path))
                    if path_hint is not None:
                        orientation_hint = path_hint
                        path_hint_rows += 1
                        if source_type == "other":
                            source_type = "path_orientation_folder"

                if not include_other and source_type == "other" and orientation_hint is None:
                    continue

                target_rect = (None, None, None, None)
                target_mm: Optional[int] = None
                hint_avail: Optional[int] = None
                if orientation_hint is not None and 0 <= orientation_hint <= 3:
                    entry = record.entries[orientation_hint]
                    target_rect = (entry.top, entry.left, entry.bottom, entry.right)
                    target_mm = entry.mm
                    hint_avail = record.orient_available[orientation_hint]
                    supervised_rows += 1

                depth_orient = None
                depth_y = None
                if sample_idx is not None:
                    depth_orient = record.depth_orientation_map.get(sample_idx)
                    depth_y = record.depth_ycoord_map.get(sample_idx)

                writer.writerow(
                    {
                        "setup_id": record.folder_id,
                        "split": record.split,
                        "dataset_folder": record.folder_name,
                        "manufacturer": record.manufacturer,
                        "model_name": record.model_name,
                        "fss_path": record.fss_path.as_posix(),
                        "image_path": image_path.resolve().as_posix(),
                        "image_name": image_path.name,
                        "source_type": source_type,
                        "sample_idx": sample_idx if sample_idx is not None else "",
                        "orientation_hint": orientation_hint if orientation_hint is not None else "",
                        "orientation_hint_name": ORIENTATION_NAMES.get(orientation_hint, "") if orientation_hint is not None else "",
                        "orientation_available_for_hint": hint_avail if hint_avail is not None else "",
                        "target_rect_top": target_rect[0] if target_rect[0] is not None else "",
                        "target_rect_left": target_rect[1] if target_rect[1] is not None else "",
                        "target_rect_bottom": target_rect[2] if target_rect[2] is not None else "",
                        "target_rect_right": target_rect[3] if target_rect[3] is not None else "",
                        "target_mm": target_mm if target_mm is not None else "",
                        "depth_orient_value": depth_orient if depth_orient is not None else "",
                        "depth_y_coordinate": depth_y if depth_y is not None else "",
                    }
                )
                seen_images.add(image_path.resolve().as_posix())
                total_rows += 1

        if include_path_orientation_frames:
            for root in dataset_roots:
                if not root.is_dir():
                    continue

                for image_path in root.rglob("*"):
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue

                    resolved_path = image_path.resolve()
                    image_key = resolved_path.as_posix()
                    if image_key in seen_images:
                        continue
                    if _is_under_known_setup(resolved_path, setup_roots):
                        continue

                    try:
                        rel_to_root = resolved_path.relative_to(root)
                    except ValueError:
                        continue

                    source_type, orientation_hint, sample_idx = _match_image(resolved_path.name)
                    if orientation_hint is None:
                        orientation_hint = _infer_orientation_from_path(rel_to_root)
                        if orientation_hint is not None:
                            path_hint_rows += 1
                            if source_type == "other":
                                source_type = "path_orientation_folder"

                    if not include_other and source_type == "other" and orientation_hint is None:
                        continue

                    if rel_to_root.parts:
                        dataset_folder = rel_to_root.parts[0]
                        setup_id = (root / dataset_folder).resolve().as_posix()
                    else:
                        dataset_folder = root.name
                        setup_id = root.resolve().as_posix()

                    split = split_map.get(dataset_folder, "train")
                    manufacturer = infer_manufacturer(dataset_folder)

                    writer.writerow(
                        {
                            "setup_id": setup_id,
                            "split": split,
                            "dataset_folder": dataset_folder,
                            "manufacturer": manufacturer,
                            "model_name": dataset_folder,
                            "fss_path": "",
                            "image_path": image_key,
                            "image_name": resolved_path.name,
                            "source_type": source_type,
                            "sample_idx": sample_idx if sample_idx is not None else "",
                            "orientation_hint": orientation_hint if orientation_hint is not None else "",
                            "orientation_hint_name": ORIENTATION_NAMES.get(orientation_hint, "") if orientation_hint is not None else "",
                            "orientation_available_for_hint": "",
                            "target_rect_top": "",
                            "target_rect_left": "",
                            "target_rect_bottom": "",
                            "target_rect_right": "",
                            "target_mm": "",
                            "depth_orient_value": "",
                            "depth_y_coordinate": "",
                        }
                    )
                    seen_images.add(image_key)
                    total_rows += 1
                    orphan_rows += 1
                    if orientation_hint is not None and 0 <= orientation_hint <= 3:
                        supervised_rows += 1

    return total_rows, supervised_rows, path_hint_rows, orphan_rows


def _write_symbol_prior_manifest(path: Path, frame_manifest_path: Path) -> int:
    rows_out: List[Dict[str, str]] = []
    with frame_manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("source_type", "") != "th_orientation_positive":
                continue
            if row.get("orientation_hint", "") == "":
                continue
            rows_out.append(
                {
                    "setup_id": row["setup_id"],
                    "split": row["split"],
                    "dataset_folder": row["dataset_folder"],
                    "manufacturer": row["manufacturer"],
                    "image_path": row["image_path"],
                    "orientation_class": row["orientation_hint"],
                    "orientation_class_name": row["orientation_hint_name"],
                    "sample_idx": row["sample_idx"],
                }
            )

    fields = [
        "setup_id",
        "split",
        "dataset_folder",
        "manufacturer",
        "image_path",
        "orientation_class",
        "orientation_class_name",
        "sample_idx",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    return len(rows_out)


def _build_summary(
    records: Sequence[FolderRecord],
    frame_rows: int,
    supervised_rows: int,
    symbol_rows: int,
    path_hint_rows: int,
    orphan_rows: int,
    warnings: Sequence[str],
) -> Dict[str, object]:
    by_split = {split: 0 for split in SPLITS}
    by_manufacturer: Dict[str, int] = {}
    avail_counter: Dict[str, int] = {}

    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_manufacturer[r.manufacturer] = by_manufacturer.get(r.manufacturer, 0) + 1
        key = f"{r.orient_available[0]}{r.orient_available[1]}{r.orient_available[2]}{r.orient_available[3]}"
        avail_counter[key] = avail_counter.get(key, 0) + 1

    return {
        "num_setups": len(records),
        "setups_per_split": by_split,
        "setups_per_manufacturer": dict(sorted(by_manufacturer.items())),
        "orient_availability_patterns": dict(sorted(avail_counter.items(), key=lambda x: x[0])),
        "frame_rows": frame_rows,
        "frame_rows_with_orientation_hint": supervised_rows,
        "frame_rows_with_path_orientation_hint": path_hint_rows,
        "frame_rows_from_orphan_scan": orphan_rows,
        "symbol_prior_rows": symbol_rows,
        "warnings_count": len(warnings),
        "warnings": list(warnings)[:200],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare symbolic orientation dataset for redesigned line16 pipeline.")
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[Path("Dataset"), Path("Dataset L_T")],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbolic_dataset"),
    )
    parser.add_argument(
        "--folders-line16",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/folders_line16.csv"),
        help="Optional split source. If missing, defaults to split=train.",
    )
    parser.add_argument(
        "--include-other-frames",
        action="store_true",
        help="Include frame rows with source_type=other. Default: only orientation-related sources.",
    )
    parser.add_argument(
        "--exclude-path-orientation-frames",
        action="store_true",
        help="Disable orientation hints inferred from folder names (e.g., NoFlip/LR/UD/LRUD).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = _load_split_map(args.folders_line16)
    dataset_roots = [p.expanduser().resolve() for p in args.dataset_roots]
    records, warnings = _scan_folders(dataset_roots, split_map)

    setup_manifest = output_dir / "setup_orientation_manifest.csv"
    frame_manifest = output_dir / "frame_manifest.csv"
    symbol_manifest = output_dir / "symbol_prior_manifest.csv"

    _write_setup_orientation_manifest(setup_manifest, records)
    frame_rows, supervised_rows, path_hint_rows, orphan_rows = _write_frame_manifest(
        frame_manifest,
        records,
        dataset_roots=dataset_roots,
        split_map=split_map,
        include_other=args.include_other_frames,
        include_path_orientation_frames=not args.exclude_path_orientation_frames,
    )
    symbol_rows = _write_symbol_prior_manifest(symbol_manifest, frame_manifest)

    summary = _build_summary(
        records,
        frame_rows,
        supervised_rows,
        symbol_rows,
        path_hint_rows,
        orphan_rows,
        warnings,
    )
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_txt.write_text(
        "\n".join(
            [
                f"Setups: {summary['num_setups']}",
                f"Frame rows: {summary['frame_rows']}",
                f"Frame rows with orientation hint: {summary['frame_rows_with_orientation_hint']}",
                f"Frame rows with path orientation hint: {summary['frame_rows_with_path_orientation_hint']}",
                f"Frame rows from orphan scan: {summary['frame_rows_from_orphan_scan']}",
                f"Symbol prior rows: {summary['symbol_prior_rows']}",
                f"Warnings: {summary['warnings_count']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Setup manifest: {setup_manifest}", flush=True)
    print(f"Frame manifest: {frame_manifest}", flush=True)
    print(f"Symbol prior manifest: {symbol_manifest}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    print(f"Setups: {summary['num_setups']} | Frames: {summary['frame_rows']} | Hinted frames: {summary['frame_rows_with_orientation_hint']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
