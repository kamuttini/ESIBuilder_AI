#!/usr/bin/env python3
"""Build the probe-template dataset (line #14 RECT_NAME_PROBE) with per-image boxes.

Ground truth is recovered by template matching the reference crop
``DB_echo/<setup>/probe_name.png`` inside every acquisition frame, so each image
gets its own pixel-exact box instead of inheriting one box per folder.

The ``.fss`` box and the saved crop are related by a constant border, measured on
the whole SSD_esi1_n1 volume (368/390 configs):

    crop top-left  = (line14_left + 3, line14_top + 3)
    crop size      = (line14_width - 5, line14_height - 5)

so a match at ``(mx, my)`` reconstructs the legacy box as
``(mx - 3, my - 3, mx - 3 + tpl_w + 5, my - 3 + tpl_h + 5)``.

Splits are leak-free at *layout* level: folders are grouped into connected
components of "same UI layout" (vendor + frame size + line #13 box) and "same
normalized folder name", which keeps the L/T twins of a machine in one split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")
RECT_NAME_PREFIX_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|")
INT_RE = re.compile(r"^-?\d+$")

# Border between the .fss box and the saved crop (measured, see module docstring).
CROP_MARGIN_TOP_LEFT = 3
CROP_MARGIN_TOTAL = 5

VENDOR_CLASS_ORDER: Tuple[str, ...] = (
    "Alpinion", "BK", "Biopsee", "Canon", "Esaote", "ExactVu", "GE", "Hitachi",
    "Koelis", "Mindray", "Philips", "Siemens", "Sonostar", "Terason", "Toshiba", "Vinno",
)
VENDOR_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(VENDOR_CLASS_ORDER)}

NAME_BASE_DROP = {
    "l", "t", "rev", "revisione", "lineare", "linear", "trasversale", "transversal",
    "biplana", "biplane", "grid", "griglia",
}


# --------------------------------------------------------------------------- #
# .fss parsing
# --------------------------------------------------------------------------- #
def _normalize_name(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()


def infer_manufacturer(folder_name: str) -> str:
    name = _normalize_name(folder_name).strip()
    if re.match(r"^bk(?:[\s,_-]|\d)", name):
        return "BK"
    for prefix, vendor in (
        ("esaote", "Esaote"), ("hitachi", "Hitachi"), ("ge", "GE"), ("mindray", "Mindray"),
        ("canon", "Canon"), ("philips", "Philips"), ("toshiba", "Toshiba"), ("siemens", "Siemens"),
        ("koelis", "Koelis"), ("biopsee", "Biopsee"), ("terason", "Terason"),
        ("sonostar", "Sonostar"), ("exactvu", "ExactVu"), ("alpinion", "Alpinion"),
        ("vinno", "Vinno"),
    ):
        if name.startswith(prefix):
            return vendor
    return "UNKNOWN"


def folder_name_base(folder_name: str) -> str:
    """Normalized machine key: drops orientation / revision markers."""
    text = re.sub(r"[^a-z0-9]+", " ", _normalize_name(folder_name))
    tokens = [t for t in text.split() if t and t not in NAME_BASE_DROP and not re.fullmatch(r"rev\d*", t)]
    return " ".join(tokens)


def _safe_line(lines: Sequence[str], idx: int) -> Optional[str]:
    return lines[idx - 1].strip() if 1 <= idx <= len(lines) else None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    cleaned = value.strip()
    return int(cleaned) if INT_RE.match(cleaned) else None


def _parse_box(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Return (top, left, bottom, right) from the first segment of a .fss rect line."""
    if not value:
        return None
    match = RECT_NAME_PREFIX_RE.match(re.split(r"[;,]", value.strip())[0])
    if not match:
        return None
    return tuple(int(match.group(i)) for i in range(1, 5))  # type: ignore[return-value]


def parse_fss(path: Path) -> Dict[str, object]:
    """Parse a .fss, tolerating the legacy layout (one header line less)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rect_line = None
    for candidate in (11, 10):
        if _safe_line(lines, candidate) and RECT_RE.match(_safe_line(lines, candidate) or ""):
            rect_line = candidate
            break
    if rect_line is None:
        for idx in range(1, min(len(lines), 15) + 1):
            if RECT_RE.match(_safe_line(lines, idx) or ""):
                rect_line = idx
                break
    if rect_line is None:
        raise ValueError("RECT_ECHO (#11) not found in the first 15 lines")

    offset = rect_line - 11
    rect_echo = _parse_box(_safe_line(lines, rect_line))
    box13 = _parse_box(_safe_line(lines, 13 + offset))
    box14 = _parse_box(_safe_line(lines, 14 + offset))
    return {
        "fss_rect_line": rect_line,
        "fss_version": _safe_line(lines, 1 + offset) if offset == 0 else "",
        "fss_id_echo": _parse_int(_safe_line(lines, 2 + offset)),
        "fss_id_probe": _parse_int(_safe_line(lines, 3 + offset)),
        "fss_video_x": _parse_int(_safe_line(lines, 9 + offset)),
        "fss_video_y": _parse_int(_safe_line(lines, 10 + offset)),
        "rect_echo": rect_echo,
        "box13": box13,
        "box14": box14,
    }


# --------------------------------------------------------------------------- #
# Folder scan
# --------------------------------------------------------------------------- #
@dataclass
class FolderJob:
    folder: Path
    folder_id: str
    manufacturer: str
    setup_name: str
    fss_path: Path
    template_path: Optional[Path]
    box14: Optional[Tuple[int, int, int, int]]
    box13: Optional[Tuple[int, int, int, int]]
    rect_echo: Optional[Tuple[int, int, int, int]]
    fss_meta: Dict[str, object]
    image_paths: List[Path] = field(default_factory=list)


def collect_images(image_dir: Path) -> List[Path]:
    return [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
    ]


def find_template(folder: Path, setup_name: str) -> Optional[Path]:
    db_echo = next((c for c in folder.iterdir() if c.is_dir() and c.name.lower() == "db_echo"), None)
    if db_echo is None:
        return None
    setup_dir = db_echo / setup_name
    if setup_dir.is_dir():
        exact = setup_dir / "probe_name.png"
        if exact.is_file():
            return exact
    candidates = [
        p for p in db_echo.rglob("probe_name.png")
        if p.is_file() and not p.name.startswith("._")
    ]
    if not candidates:
        return None
    for p in candidates:
        if p.parent.name.lower() == setup_name.lower():
            return p
    return sorted(candidates)[0]


def scan_folders(dataset_root: Path, exclude_re: Optional[re.Pattern[str]]) -> Tuple[List[FolderJob], List[str]]:
    jobs: List[FolderJob] = []
    warnings: List[str] = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if folder.name.startswith(("$", ".")) or folder.name == "System Volume Information":
            continue
        if exclude_re and exclude_re.search(folder.name):
            continue
        image_dir, setup_dir = folder / "image_samples", folder / "DB_setup"
        if not setup_dir.is_dir():
            warnings.append(f"{folder.name}: no DB_setup")
            continue
        if not image_dir.is_dir():
            warnings.append(f"{folder.name}: no image_samples")
            continue
        images = collect_images(image_dir)
        if not images:
            warnings.append(f"{folder.name}: image_samples empty")
            continue
        fss_files = sorted(p for p in setup_dir.glob("*.fss") if not p.name.startswith(("~$", "._")))
        if len(fss_files) != 1:
            warnings.append(f"{folder.name}: expected 1 .fss, found {len(fss_files)}")
            continue
        fss_path = fss_files[0]
        try:
            meta = parse_fss(fss_path)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"{folder.name}: cannot parse {fss_path.name}: {exc}")
            continue
        manufacturer = infer_manufacturer(folder.name)
        if manufacturer == "UNKNOWN":
            warnings.append(f"{folder.name}: unknown manufacturer")
        setup_name = fss_path.stem
        template = find_template(folder, setup_name)
        if template is None:
            warnings.append(f"{folder.name}: probe_name.png missing")
        jobs.append(FolderJob(
            folder=folder,
            folder_id=folder.name,
            manufacturer=manufacturer,
            setup_name=setup_name,
            fss_path=fss_path,
            template_path=template,
            box14=meta["box14"],          # type: ignore[arg-type]
            box13=meta["box13"],          # type: ignore[arg-type]
            rect_echo=meta["rect_echo"],  # type: ignore[arg-type]
            fss_meta=meta,
            image_paths=images,
        ))
    return jobs, warnings


def subsample(paths: List[Path], max_images: int) -> List[Path]:
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
    return [paths[i] for i in sorted(set(idx.tolist()))]


# --------------------------------------------------------------------------- #
# Template matching
# --------------------------------------------------------------------------- #
def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_full(image: np.ndarray, tpl: np.ndarray) -> Tuple[float, int, int]:
    res = cv2.matchTemplate(image, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return float(score), int(loc[0]), int(loc[1])


def match_roi(image: np.ndarray, tpl: np.ndarray, anchor: Tuple[int, int], margin: int) -> Tuple[float, int, int]:
    h, w = image.shape[:2]
    th, tw = tpl.shape[:2]
    x0 = max(0, anchor[0] - margin)
    y0 = max(0, anchor[1] - margin)
    x1 = min(w, anchor[0] + tw + margin)
    y1 = min(h, anchor[1] + th + margin)
    if x1 - x0 < tw or y1 - y0 < th:
        return match_full(image, tpl)
    score, mx, my = match_full(image[y0:y1, x0:x1], tpl)
    return score, mx + x0, my + y0


def box_from_match(mx: int, my: int, tpl_w: int, tpl_h: int) -> Tuple[int, int, int, int]:
    """Reconstruct the legacy .fss box (xmin, ymin, xmax, ymax) from a crop match."""
    x1 = mx - CROP_MARGIN_TOP_LEFT
    y1 = my - CROP_MARGIN_TOP_LEFT
    return x1, y1, x1 + tpl_w + CROP_MARGIN_TOTAL, y1 + tpl_h + CROP_MARGIN_TOTAL


def process_folder(args: Tuple[FolderJob, argparse.Namespace]) -> Dict[str, object]:
    job, opts = args
    cv2.setNumThreads(1)
    out: Dict[str, object] = {
        "folder_id": job.folder_id, "rows": [], "notes": [],
        "manufacturer": job.manufacturer, "setup_name": job.setup_name,
    }
    if job.template_path is None:
        out["notes"].append("template_missing")
        return out
    tpl = cv2.imread(str(job.template_path), cv2.IMREAD_GRAYSCALE)
    if tpl is None or tpl.size == 0:
        out["notes"].append("template_unreadable")
        return out
    tpl_h, tpl_w = tpl.shape[:2]
    fss_box = None
    if job.box14 is not None:
        top, left, bottom, right = job.box14
        fss_box = (left, top, right, bottom)

    paths = subsample(job.image_paths, opts.max_images_per_folder)
    anchor: Optional[Tuple[int, int]] = None
    rows: List[Dict[str, object]] = []
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        ih, iw = image.shape[:2]
        if ih < tpl_h or iw < tpl_w:
            continue
        if anchor is None or i < opts.full_search_frames:
            score, mx, my = match_full(image, tpl)
        else:
            score, mx, my = match_roi(image, tpl, anchor, opts.roi_margin)
            if score < opts.min_score:
                score, mx, my = match_full(image, tpl)
        hit = score >= opts.min_score
        if hit and anchor is None:
            anchor = (mx, my)
        box = box_from_match(mx, my, tpl_w, tpl_h) if hit else None
        rows.append({
            "image_path": str(path),
            "image_name": path.name,
            "image_width": iw,
            "image_height": ih,
            "has_template": int(hit),
            "match_score": round(score, 4),
            "match_x": mx if hit else "",
            "match_y": my if hit else "",
            "bbox_xmin": box[0] if box else "",
            "bbox_ymin": box[1] if box else "",
            "bbox_xmax": box[2] if box else "",
            "bbox_ymax": box[3] if box else "",
            "iou_vs_fss": round(iou_xyxy(box, fss_box), 4) if (box and fss_box) else "",
        })
    out["rows"] = rows
    out["template_w"] = tpl_w
    out["template_h"] = tpl_h
    out["template_path"] = str(job.template_path)
    return out


# --------------------------------------------------------------------------- #
# Grouping and splits
# --------------------------------------------------------------------------- #
class DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_groups(folder_rows: List[Dict[str, object]]) -> Dict[str, str]:
    """Connected components over 'same UI layout' and 'same machine name'."""
    dsu = DisjointSet()
    by_layout: Dict[str, List[str]] = defaultdict(list)
    by_base: Dict[str, List[str]] = defaultdict(list)
    for row in folder_rows:
        fid = str(row["folder"])
        dsu.find(fid)
        by_layout[str(row["layout_key"])].append(fid)
        by_base[str(row["name_base"])].append(fid)
    for bucket in (by_layout, by_base):
        for members in bucket.values():
            for other in members[1:]:
                dsu.union(members[0], other)
    roots = {}
    for row in folder_rows:
        roots[str(row["folder"])] = dsu.find(str(row["folder"]))
    ordered = {root: f"grp_{i:03d}" for i, root in enumerate(sorted(set(roots.values())))}
    return {fid: ordered[root] for fid, root in roots.items()}


def assign_splits(
    group_stats: Dict[str, Dict[str, object]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, str]:
    """Greedy split assignment on groups, stratified per vendor, balanced on image counts."""
    import random

    rng = random.Random(seed)
    by_vendor: Dict[str, List[str]] = defaultdict(list)
    for gid, stats in group_stats.items():
        by_vendor[str(stats["vendor"])].append(gid)

    assignment: Dict[str, str] = {}
    for vendor in sorted(by_vendor):
        groups = by_vendor[vendor]
        rng.shuffle(groups)
        groups.sort(key=lambda g: -int(group_stats[g]["n_images"]))
        total = sum(int(group_stats[g]["n_images"]) for g in groups)
        targets = {s: total * ratios[s] for s in SPLITS}
        got = {s: 0 for s in SPLITS}
        n_groups = {s: 0 for s in SPLITS}
        if len(groups) == 1:
            min_req = {"train": 1, "val": 0, "test": 0}
        elif len(groups) == 2:
            min_req = {"train": 1, "val": 0, "test": 1}
        else:
            min_req = {"train": 1, "val": 1, "test": 1}
        for idx, gid in enumerate(groups):
            remaining = len(groups) - idx
            missing = [s for s in SPLITS if n_groups[s] < min_req[s]]
            candidates = missing if missing and remaining <= len(missing) else list(SPLITS)
            best, best_score = None, None
            for split in candidates:
                score = targets[split] - got[split]
                if n_groups[split] < min_req[split]:
                    score += 1e9
                cand = (score, -got[split])
                if best_score is None or cand > best_score:
                    best, best_score = split, cand
            assignment[gid] = str(best)
            got[str(best)] += int(group_stats[gid]["n_images"])
            n_groups[str(best)] += 1
    return assignment


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images-per-folder", type=int, default=250, help="0 = all frames")
    parser.add_argument("--min-score", type=float, default=0.90, help="TM_CCOEFF_NORMED threshold")
    parser.add_argument("--roi-margin", type=int, default=80, help="px around the anchor for the fast search")
    parser.add_argument("--full-search-frames", type=int, default=3, help="frames matched on the full image first")
    parser.add_argument("--min-hit-rate", type=float, default=0.50, help="below this a config is flagged for review")
    parser.add_argument("--min-iou-vs-fss", type=float, default=0.75, help="below this a config is flagged for review")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--exclude-folder-regex", type=str, default="")
    parser.add_argument("--limit-folders", type=int, default=0, help="smoke test: only the first N folders")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    opts = parse_args(argv)
    ratios = {"train": opts.train_ratio, "val": opts.val_ratio, "test": opts.test_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        print(f"ERROR: ratios must sum to 1 (got {sum(ratios.values())})", file=sys.stderr)
        return 2

    exclude_re = re.compile(opts.exclude_folder_regex) if opts.exclude_folder_regex else None
    jobs, warnings = scan_folders(opts.dataset_root, exclude_re)
    if opts.limit_folders > 0:
        jobs = jobs[: opts.limit_folders]
    print(f"Folders with .fss + images: {len(jobs)} (skipped: {len(warnings)})", flush=True)

    results: List[Dict[str, object]] = []
    payload = [(job, opts) for job in jobs]
    if opts.workers > 1:
        with Pool(opts.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_folder, payload), start=1):
                results.append(res)
                if i % 20 == 0 or i == len(payload):
                    print(f"  {i}/{len(payload)} folders", flush=True)
    else:
        for i, item in enumerate(payload, start=1):
            results.append(process_folder(item))
            if i % 20 == 0 or i == len(payload):
                print(f"  {i}/{len(payload)} folders", flush=True)

    by_id = {job.folder_id: job for job in jobs}
    results_by_id = {str(r["folder_id"]): r for r in results}

    # ---------------- per-folder rows ---------------- #
    folder_rows: List[Dict[str, object]] = []
    for job in jobs:
        res = results_by_id.get(job.folder_id, {})
        rows: List[Dict[str, object]] = list(res.get("rows", []))  # type: ignore[arg-type]
        hits = [r for r in rows if r["has_template"] == 1]
        scores = [float(r["match_score"]) for r in rows]
        ious = [float(r["iou_vs_fss"]) for r in hits if r["iou_vs_fss"] != ""]
        xs = [int(r["bbox_xmin"]) for r in hits]
        ys = [int(r["bbox_ymin"]) for r in hits]
        hit_rate = len(hits) / len(rows) if rows else 0.0
        med_iou = statistics.median(ious) if ious else 0.0
        notes = list(res.get("notes", []))  # type: ignore[arg-type]
        if "template_missing" in notes or "template_unreadable" in notes:
            status = "template_missing"
        elif job.box14 is None:
            status = "no_fss_box"
        elif not rows:
            status = "no_images"
        elif hit_rate < opts.min_hit_rate:
            status = "low_hit_rate"
        elif med_iou < opts.min_iou_vs_fss:
            status = "fss_mismatch"
        else:
            status = "ok"
        w = int(rows[0]["image_width"]) if rows else 0
        h = int(rows[0]["image_height"]) if rows else 0
        b13 = job.box13 or (-1, -1, -1, -1)
        folder_rows.append({
            "folder": job.folder_id,
            "folder_path": str(job.folder),
            "manufacturer": job.manufacturer,
            "vendor_id": VENDOR_TO_ID.get(job.manufacturer, -1),
            "setup_name": job.setup_name,
            "fss_path": str(job.fss_path),
            "template_path": res.get("template_path", ""),
            "template_w": res.get("template_w", ""),
            "template_h": res.get("template_h", ""),
            "image_width": w,
            "image_height": h,
            "layout_key": f"{job.manufacturer}|{w}x{h}|{b13[0]},{b13[1]},{b13[2]},{b13[3]}",
            "name_base": folder_name_base(job.folder_id),
            "n_images_total": len(job.image_paths),
            "n_images_scanned": len(rows),
            "n_hits": len(hits),
            "hit_rate": round(hit_rate, 4),
            "median_score": round(statistics.median(scores), 4) if scores else "",
            "median_iou_vs_fss": round(med_iou, 4),
            "box_x_range_px": (max(xs) - min(xs)) if xs else "",
            "box_y_range_px": (max(ys) - min(ys)) if ys else "",
            "fss_top": job.box14[0] if job.box14 else "",
            "fss_left": job.box14[1] if job.box14 else "",
            "fss_bottom": job.box14[2] if job.box14 else "",
            "fss_right": job.box14[3] if job.box14 else "",
            "median_box_xmin": int(statistics.median(xs)) if xs else "",
            "median_box_ymin": int(statistics.median(ys)) if ys else "",
            "gt_status": status,
            "notes": ";".join(notes),
        })

    groups = build_groups(folder_rows)
    for row in folder_rows:
        row["group_id"] = groups[str(row["folder"])]

    group_stats: Dict[str, Dict[str, object]] = {}
    for row in folder_rows:
        gid = str(row["group_id"])
        stats = group_stats.setdefault(gid, {"vendor": row["manufacturer"], "n_images": 0, "n_folders": 0})
        stats["n_images"] = int(stats["n_images"]) + int(row["n_hits"])
        stats["n_folders"] = int(stats["n_folders"]) + 1
    split_by_group = assign_splits(group_stats, ratios, opts.seed)
    for row in folder_rows:
        row["split"] = split_by_group[str(row["group_id"])]

    # ---------------- write outputs ---------------- #
    out_dir = opts.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_fields = list(folder_rows[0].keys()) if folder_rows else []
    with (out_dir / "folders_probe_template.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=folder_fields)
        writer.writeheader()
        writer.writerows(sorted(folder_rows, key=lambda r: str(r["folder"]).lower()))

    manifest_fields = [
        "image_path", "split", "group_id", "gt_status", "dataset_folder", "manufacturer",
        "vendor_id", "setup_name", "layout_key", "image_name", "image_width", "image_height",
        "has_template", "match_score", "match_x", "match_y",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "bbox_norm_xmin", "bbox_norm_ymin", "bbox_norm_xmax", "bbox_norm_ymax",
        "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
        "fss_top", "fss_left", "fss_bottom", "fss_right", "iou_vs_fss", "template_path",
    ]
    counters: Counter = Counter()
    folder_index = {str(r["folder"]): r for r in folder_rows}
    with (out_dir / "manifest_probe_template.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_fields)
        writer.writeheader()
        for job in sorted(jobs, key=lambda j: j.folder_id.lower()):
            meta = folder_index[job.folder_id]
            res = results_by_id.get(job.folder_id, {})
            for row in res.get("rows", []):  # type: ignore[union-attr]
                hit = int(row["has_template"]) == 1
                w, h = int(row["image_width"]), int(row["image_height"])
                counters["images"] += 1
                counters["positive" if hit else "negative"] += 1
                writer.writerow({
                    "image_path": row["image_path"],
                    "split": meta["split"],
                    "group_id": meta["group_id"],
                    "gt_status": meta["gt_status"],
                    "dataset_folder": job.folder_id,
                    "manufacturer": job.manufacturer,
                    "vendor_id": VENDOR_TO_ID.get(job.manufacturer, -1),
                    "setup_name": job.setup_name,
                    "layout_key": meta["layout_key"],
                    "image_name": row["image_name"],
                    "image_width": w,
                    "image_height": h,
                    "has_template": row["has_template"],
                    "match_score": row["match_score"],
                    "match_x": row["match_x"],
                    "match_y": row["match_y"],
                    "bbox_xmin": row["bbox_xmin"],
                    "bbox_ymin": row["bbox_ymin"],
                    "bbox_xmax": row["bbox_xmax"],
                    "bbox_ymax": row["bbox_ymax"],
                    "bbox_norm_xmin": round(int(row["bbox_xmin"]) / w, 6) if hit else "",
                    "bbox_norm_ymin": round(int(row["bbox_ymin"]) / h, 6) if hit else "",
                    "bbox_norm_xmax": round(int(row["bbox_xmax"]) / w, 6) if hit else "",
                    "bbox_norm_ymax": round(int(row["bbox_ymax"]) / h, 6) if hit else "",
                    "bbox_top": row["bbox_ymin"],
                    "bbox_left": row["bbox_xmin"],
                    "bbox_bottom": row["bbox_ymax"],
                    "bbox_right": row["bbox_xmax"],
                    "fss_top": meta["fss_top"],
                    "fss_left": meta["fss_left"],
                    "fss_bottom": meta["fss_bottom"],
                    "fss_right": meta["fss_right"],
                    "iou_vs_fss": row["iou_vs_fss"],
                    "template_path": meta["template_path"],
                })

    review = [r for r in folder_rows if r["gt_status"] != "ok"]
    with (out_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=folder_fields)
        writer.writeheader()
        writer.writerows(sorted(review, key=lambda r: (str(r["gt_status"]), str(r["folder"]).lower())))

    per_split = Counter(r["split"] for r in folder_rows)
    per_split_images = Counter()
    per_vendor_split = defaultdict(Counter)
    for r in folder_rows:
        per_split_images[r["split"]] += int(r["n_hits"])
        per_vendor_split[r["manufacturer"]][r["split"]] += 1
    summary = {
        "dataset_root": str(opts.dataset_root),
        "output_dir": str(out_dir),
        "params": {
            "max_images_per_folder": opts.max_images_per_folder,
            "min_score": opts.min_score,
            "roi_margin": opts.roi_margin,
            "min_hit_rate": opts.min_hit_rate,
            "min_iou_vs_fss": opts.min_iou_vs_fss,
            "ratios": ratios,
            "seed": opts.seed,
            "crop_margin_top_left": CROP_MARGIN_TOP_LEFT,
            "crop_margin_total": CROP_MARGIN_TOTAL,
        },
        "folders_scanned": len(jobs),
        "folders_skipped": len(warnings),
        "groups": len(set(groups.values())),
        "layouts": len(set(str(r["layout_key"]) for r in folder_rows)),
        "images_scanned": int(counters["images"]),
        "images_positive": int(counters["positive"]),
        "images_negative": int(counters["negative"]),
        "gt_status": dict(Counter(r["gt_status"] for r in folder_rows)),
        "folders_per_split": dict(per_split),
        "positive_images_per_split": dict(per_split_images),
        "folders_per_vendor_split": {v: dict(c) for v, c in sorted(per_vendor_split.items())},
        "skipped_folders": warnings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"dataset_root       : {opts.dataset_root}",
        f"cartelle            : {len(jobs)} (scartate {len(warnings)})",
        f"gruppi leak-free    : {summary['groups']}   layout UI: {summary['layouts']}",
        f"immagini analizzate : {summary['images_scanned']}  (con template {summary['images_positive']}, senza {summary['images_negative']})",
        f"stato GT            : {summary['gt_status']}",
        f"cartelle per split  : {dict(per_split)}",
        f"immagini per split  : {dict(per_split_images)}",
        "",
        "vendor        train  val  test",
    ]
    for vendor, counts in sorted(per_vendor_split.items()):
        lines.append(f"{vendor:12s} {counts['train']:6d} {counts['val']:4d} {counts['test']:5d}")
    (out_dir / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
