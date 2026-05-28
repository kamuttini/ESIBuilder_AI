#!/usr/bin/env python3
"""Build a vendor LR marker template library from DB_echo orientation_0 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIRS = {
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    ".DocumentRevisions-V100",
    "__pycache__",
}


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


def _safe_folder_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip() or "UNKNOWN").strip("_")
    return out or "UNKNOWN"


def _iter_config_folders(roots: Sequence[Path], max_depth: int) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        root_depth = len(root.parts)
        for dirpath, dirnames, _filenames in os.walk(root):
            path = Path(dirpath).resolve()
            depth = max(0, len(path.parts) - root_depth)
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
            if max_depth >= 0 and depth > max_depth:
                dirnames[:] = []
                continue
            names = set(dirnames)
            if {"DB_setup", "DB_echo", "image_samples"}.issubset(names):
                yield path
                dirnames[:] = [name for name in dirnames if name not in {"DB_setup", "DB_echo", "image_samples"}]


def _orientation_0_path(config_dir: Path) -> Optional[Path]:
    db_echo = config_dir / "DB_echo"
    if not db_echo.is_dir():
        return None
    candidates = [
        p for p in db_echo.rglob("*")
        if p.is_file() and p.stem.lower() == "orientation_0" and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(candidates)[0] if candidates else None


def _template_stats(path: Path, blank_template_max_value: int) -> Optional[Dict[str, object]]:
    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            extrema = gray.getextrema()
            if extrema is None or int(extrema[1]) <= int(blank_template_max_value):
                return None
            hist = gray.histogram()
            total = max(1, gray.size[0] * gray.size[1])
            mean = sum(i * n for i, n in enumerate(hist)) / float(total)
            var = sum(((i - mean) ** 2) * n for i, n in enumerate(hist)) / float(total)
            if var < 1e-6:
                return None
            payload = gray.tobytes() + str(gray.size).encode("ascii")
            thumb = gray.resize((32, 32))
            thumb_vals = list(thumb.getdata())
            thumb_mean = sum(thumb_vals) / float(len(thumb_vals))
            centered = [float(v) - thumb_mean for v in thumb_vals]
            norm = sum(v * v for v in centered) ** 0.5
            return {
                "width": int(gray.size[0]),
                "height": int(gray.size[1]),
                "mean": float(mean),
                "std": float(var ** 0.5),
                "digest": hashlib.sha1(payload).hexdigest(),
                "thumb_centered": centered,
                "thumb_norm": float(norm),
            }
    except Exception:
        return None


def _similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    av = a.get("thumb_centered")
    bv = b.get("thumb_centered")
    an = float(a.get("thumb_norm", 0.0) or 0.0)
    bn = float(b.get("thumb_norm", 0.0) or 0.0)
    if not isinstance(av, list) or not isinstance(bv, list) or an <= 1e-6 or bn <= 1e-6:
        return -1.0
    dot = sum(float(x) * float(y) for x, y in zip(av, bv))
    return dot / (an * bn)


def build_library(
    *,
    roots: Sequence[Path],
    output_dir: Path,
    max_depth: int,
    blank_template_max_value: int,
    similarity_threshold: float,
) -> Dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_vendor: Dict[str, List[Dict[str, object]]] = {}
    scanned = 0
    blank = 0
    duplicate = 0
    for config_dir in _iter_config_folders(roots, max_depth=max_depth):
        src = _orientation_0_path(config_dir)
        if src is None:
            continue
        scanned += 1
        stats = _template_stats(src, blank_template_max_value=blank_template_max_value)
        if stats is None:
            blank += 1
            continue
        vendor = infer_manufacturer(config_dir.name)
        rows = by_vendor.setdefault(vendor, [])
        if any(str(row.get("digest", "")) == str(stats["digest"]) for row in rows):
            duplicate += 1
            continue
        best_sim = max((_similarity(stats, row) for row in rows), default=-1.0)
        if best_sim >= float(similarity_threshold):
            duplicate += 1
            continue
        rows.append(
            {
                "vendor": vendor,
                "source_path": src.as_posix(),
                "config_dir": config_dir.as_posix(),
                "width": int(stats["width"]),
                "height": int(stats["height"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "digest": str(stats["digest"]),
                "best_similarity_to_kept": float(best_sim),
                "thumb_centered": stats["thumb_centered"],
                "thumb_norm": float(stats["thumb_norm"]),
            }
        )

    manifest_rows: List[Dict[str, object]] = []
    for vendor, rows in sorted(by_vendor.items(), key=lambda item: item[0].lower()):
        vendor_dir = output_dir / _safe_folder_name(vendor)
        vendor_dir.mkdir(parents=True, exist_ok=True)
        public_rows = []
        for idx, row in enumerate(rows, start=1):
            dst = vendor_dir / f"marker_{idx:03d}.png"
            shutil.copy2(Path(str(row["source_path"])), dst)
            public = {
                "vendor": vendor,
                "template_path": dst.as_posix(),
                "source_path": str(row["source_path"]),
                "config_dir": str(row["config_dir"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "digest": str(row["digest"]),
                "best_similarity_to_kept": float(row["best_similarity_to_kept"]),
            }
            public_rows.append(public)
            manifest_rows.append(public)
        (vendor_dir / "manifest.json").write_text(json.dumps(public_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "vendor",
                "template_path",
                "source_path",
                "config_dir",
                "width",
                "height",
                "mean",
                "std",
                "digest",
                "best_similarity_to_kept",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "roots": [p.expanduser().resolve().as_posix() for p in roots],
        "output_dir": output_dir.as_posix(),
        "configs_with_orientation_0_seen": int(scanned),
        "blank_or_unusable": int(blank),
        "duplicates_or_too_similar": int(duplicate),
        "vendors": {vendor: len(rows) for vendor, rows in sorted(by_vendor.items(), key=lambda item: item[0].lower())},
        "templates_total": int(len(manifest_rows)),
        "similarity_threshold": float(similarity_threshold),
        "blank_template_max_value": int(blank_template_max_value),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("/Volumes/SSD_esi1_n1")])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/lr_marker_vendor_template_library"),
    )
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--blank-template-max-value", type=int, default=12)
    parser.add_argument("--similarity-threshold", type=float, default=0.985)
    args = parser.parse_args(argv)
    summary = build_library(
        roots=args.roots,
        output_dir=args.output_dir,
        max_depth=int(args.max_depth),
        blank_template_max_value=int(args.blank_template_max_value),
        similarity_threshold=float(args.similarity_threshold),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
