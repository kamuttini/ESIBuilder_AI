"""Step 1: import a folder of acquisitions and normalise what can be read for free.

Deliberately mirrors the behaviour of tools/ultrasound/predict_fss_head_from_acquisitions.py
so the app and the pipeline see the same frames:

* acquisition frames are the files matching ``*_(vga|hdmi)_WxH*``; if none match, every image
  in the folder is used;
* duplicates are removed exactly (file size bucket, then SHA-1 on collisions);
* #06..#08 come from the filename by majority vote, #09/#10 from the sample image.

Rotation (OSD) and the model stages are not run here: they belong to the pipeline and are
wired in as separate stages.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from constants import RECT_MAX_HEIGHT, RECT_MAX_WIDTH, VIDEO_HDMI, VIDEO_VGA

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# e.g. "case12_hdmi_1920x1080_003.png"
_FRAME_RE = re.compile(r"^.+_(vga|hdmi)_(\d{3,4})x(\d{3,4})", re.IGNORECASE)
_SIZE_ANYWHERE_RE = re.compile(r"(\d{3,4})x(\d{3,4})")

VIDEO_INPUT_CODES = {"hdmi": VIDEO_HDMI, "vga": VIDEO_VGA}


def scan_folder(folder: Path) -> List[Path]:
    folder = Path(folder)
    images = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    frames = [path for path in images if _FRAME_RE.match(path.name)]
    return frames or images


def deduplicate(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Exact duplicates only: bucket by file size, hash just the collisions."""
    by_size: Dict[int, List[Path]] = {}
    for path in paths:
        by_size.setdefault(path.stat().st_size, []).append(path)

    kept: List[Path] = []
    removed: List[Path] = []
    for size, group in by_size.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        seen: Dict[str, Path] = {}
        for path in group:
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            if digest in seen:
                removed.append(path)
            else:
                seen[digest] = path
                kept.append(path)
    return sorted(kept), sorted(removed)


def parse_capture_metadata(name: str) -> Optional[Tuple[str, int, int]]:
    match = _FRAME_RE.match(name)
    if match:
        return match.group(1).lower(), int(match.group(2)), int(match.group(3))
    size = _SIZE_ANYWHERE_RE.search(name)
    lowered = name.lower()
    for token in ("hdmi", "vga"):
        if token in lowered and size:
            return token, int(size.group(1)), int(size.group(2))
    return None


def majority_metadata(paths: List[Path]) -> Optional[Tuple[str, int, int]]:
    """Winner among the filename triplets; ties prefer hdmi, then the larger area."""
    votes = Counter()
    for path in paths:
        parsed = parse_capture_metadata(path.name)
        if parsed:
            votes[parsed] += 1
    if not votes:
        return None
    best = max(
        votes.items(),
        key=lambda item: (item[1], item[0][0] == "hdmi", item[0][1] * item[0][2]),
    )
    return best[0]


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def import_folder(folder: Path, max_listed: int = 60) -> Dict:
    """Everything step 1 can establish without a model."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))

    all_images = scan_folder(folder)
    kept, removed = deduplicate(all_images)
    warnings: List[str] = []

    metadata = majority_metadata(kept)
    native_size = _image_size(kept[0]) if kept else None

    if metadata:
        video_input = VIDEO_INPUT_CODES[metadata[0]]
        video_input_size = [metadata[1], metadata[2]]
    else:
        video_input = VIDEO_HDMI
        video_input_size = list(native_size) if native_size else [0, 0]
        warnings.append(
            "input video e risoluzione non leggibili dai nomi file: "
            "usata la dimensione reale dell'immagine, da confermare"
        )
    if not kept:
        warnings.append("nessuna immagine trovata nella cartella")
    if native_size and metadata and list(native_size) != video_input_size:
        warnings.append(
            f"il nome file dice {video_input_size[0]}x{video_input_size[1]} "
            f"ma l'immagine e' {native_size[0]}x{native_size[1]}"
        )

    sample_size = list(native_size) if native_size else list(video_input_size)
    kept_names = [str(path.relative_to(folder)) for path in kept]
    return {
        "folder": str(folder),
        "video_input": video_input,
        "video_input_size": video_input_size,
        "image_sample_size": sample_size,
        "native_size": sample_size,
        "images_total_raw": len(all_images),
        "images_total": len(kept),
        "duplicates_removed": len(removed),
        "rotation_applied": 0,
        "rotation_source": "not_run",
        "resize_factor": 1.0,
        "images": [
            {"name": name, "bytes": path.stat().st_size}
            for name, path in zip(kept_names[:max_listed], kept[:max_listed])
        ],
        "images_listed": min(len(kept), max_listed),
        # Full deduplicated list: every later stage works on this and only this.
        "kept_names": kept_names,
        "warnings": warnings,
    }


def resize_proposal(rect: Dict, sample_size: List[int]) -> Optional[Dict]:
    """Spec sezione 11: native resolution by default, resize proposed only if the rect overflows.

    Returns None when the rectangle already fits the ESI screen.
    """
    rect = {side: int(rect[side]) for side in ("top", "left", "bottom", "right")}
    width = rect["right"] - rect["left"]
    height = rect["bottom"] - rect["top"]
    if width <= RECT_MAX_WIDTH and height <= RECT_MAX_HEIGHT:
        return None
    factor = min(RECT_MAX_WIDTH / width, RECT_MAX_HEIGHT / height)
    scaled_sample = [max(1, int(round(value * factor))) for value in sample_size]
    scaled_rect = {key: int(round(int(value) * factor)) for key, value in rect.items()}
    return {
        "reason": (
            f"il rettangolo e' {width}x{height}, oltre il limite ESI "
            f"{RECT_MAX_WIDTH}x{RECT_MAX_HEIGHT}"
        ),
        "factor": round(factor, 6),
        "image_sample_size": scaled_sample,
        "rect_echo": scaled_rect,
        "rect_size_after": [
            scaled_rect["right"] - scaled_rect["left"],
            scaled_rect["bottom"] - scaled_rect["top"],
        ],
    }
