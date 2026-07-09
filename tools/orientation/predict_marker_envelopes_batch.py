#!/usr/bin/env python3
"""Production-like marker + orientation-envelope prediction on raw acquisition folders.

For each top-level configuration folder (no `.fss` available, e.g. volume
SSD_esi1_n3/ACQUISITION ELABORATION):

1. collect acquisition frames (recursive; excludes Thumbs/AppleDouble/PROIBITE)
2. vendor inferred from folder name -> historical template bank
3. marker detection (bundle detector, full-image search: no rect available)
4. per-image orientation group from marker quadrant (image median axes)
5. folder-level ENVELOPE per group: one box per orientation, THE SAME for the
   whole folder, containing all accepted marker positions of that group
   (computed on the dominant resolution; other resolutions are counted and
   skipped)

Outputs (append/resume friendly):
- per_image_predictions.csv
- folder_envelopes.csv (folder x group envelopes)
- summary.json

Example:
  python3 tools/orientation/predict_marker_envelopes_batch.py \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
    --output-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run1 \
    --max-images-per-folder 14 --resume --time-budget 600
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle"

Rect = Tuple[int, int, int, int]  # top, left, bottom, right
GROUPS = ("NF", "LR", "UD", "LRUD")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDE_FILE_RE = re.compile(r"Thumbs\.db|Software Release|System Info|proibite", re.IGNORECASE)


def _import_bundle(bundle_dir: Path):
    root = bundle_dir.expanduser().resolve()
    if not (root / "orientation_marker_detector" / "detector.py").is_file():
        raise FileNotFoundError(f"Bundle not found: {root}")
    sys.path.insert(0, str(root))
    from orientation_marker_detector import detector as omd  # type: ignore

    return omd


def _collect_images(folder: Path, include_re: Optional[re.Pattern], max_images: int) -> List[Path]:
    paths: List[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name.startswith("._"):
            continue
        rel = path.relative_to(folder).as_posix()
        if EXCLUDE_FILE_RE.search(rel):
            continue
        if include_re is not None and not include_re.search(rel):
            continue
        paths.append(path)
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    stride = len(paths) / max_images
    return [paths[int(i * stride)] for i in range(max_images)]


def _union(boxes: Sequence[Rect]) -> Rect:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--library-root", type=Path, default=None)
    parser.add_argument("--folder-regex", type=str, default="")
    parser.add_argument("--include-regex", type=str, default="", help="Filter on image relative paths.")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-images-per-folder", type=int, default=14)
    parser.add_argument("--selection-images", type=int, default=4)
    parser.add_argument("--min-match-score", type=float, default=0.55)
    parser.add_argument("--fallback-threshold", type=float, default=0.58)
    parser.add_argument("--expanded-threshold", type=float, default=0.62)
    parser.add_argument("--vertical-delta", type=float, default=0.05)
    parser.add_argument("--match-max-side", type=int, default=560)
    parser.add_argument("--min-markers-per-envelope", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=0.0)
    args = parser.parse_args()

    omd = _import_bundle(args.bundle_dir)
    params = omd.DetectionParams(
        min_match_score=args.min_match_score,
        vertical_delta=args.vertical_delta,
        fallback_threshold=args.fallback_threshold,
        expanded_threshold=args.expanded_threshold,
        match_max_side=args.match_max_side,
    )
    include_re = re.compile(args.include_regex, re.IGNORECASE) if args.include_regex else None
    folder_pattern = re.compile(args.folder_regex, re.IGNORECASE) if args.folder_regex else None

    folders = [
        p
        for p in sorted(args.dataset_root.iterdir())
        if p.is_dir() and not p.name.startswith(("$", ".")) and p.name != "System Volume Information"
        and (folder_pattern is None or folder_pattern.search(p.name))
    ]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]

    image_rows: List[Dict[str, object]] = []
    envelope_rows: List[Dict[str, object]] = []
    folder_rows: List[Dict[str, object]] = []
    if args.resume:
        image_rows = _load_csv(args.output_dir / "per_image_predictions.csv")
        envelope_rows = _load_csv(args.output_dir / "folder_envelopes.csv")
        folder_rows = _load_csv(args.output_dir / "folder_summary.csv")
        done = {str(r["folder"]) for r in folder_rows}
        folders = [f for f in folders if f.name not in done]
        print(f"[resume] {len(done)} folders already done, {len(folders)} remaining")

    run_started = time.time()
    for index, folder in enumerate(folders, start=1):
        if args.time_budget > 0 and time.time() - run_started > args.time_budget:
            print(f"[time-budget] stopping after {index - 1} folders; rerun with --resume")
            break
        started = time.time()
        vendor = omd.infer_vendor_from_text(folder.name)
        paths = _collect_images(folder, include_re, args.max_images_per_folder)
        if not paths:
            folder_rows.append({"folder": folder.name, "vendor": vendor, "status": "no_images"})
            print(f"[{index}/{len(folders)}] {folder.name}: no images")
            continue

        inputs = [
            omd.ImageInput(image_path=path, image_id=path.relative_to(folder).as_posix())
            for path in paths
        ]
        try:
            analysis = omd.analyze_images(
                inputs, vendor=vendor,
                library_root=args.library_root, params=params,
                selection_images=args.selection_images,
            )
        except RuntimeError as exc:
            folder_rows.append({"folder": folder.name, "vendor": vendor, "status": f"error:{exc}"})
            print(f"[{index}/{len(folders)}] {folder.name}: ERROR {exc}")
            continue

        # Dominant resolution (envelope coordinates only make sense within one resolution).
        sizes: Counter = Counter()
        size_by_id: Dict[str, Tuple[int, int]] = {}
        for path in paths:
            try:
                with Image.open(path) as img:
                    size_by_id[path.relative_to(folder).as_posix()] = img.size
                    sizes[img.size] += 1
            except Exception:  # noqa: BLE001
                continue
        dominant_size = sizes.most_common(1)[0][0] if sizes else None

        markers_by_group: Dict[str, List[Rect]] = {}
        n_review = 0
        for row in analysis.rows:
            in_dominant = dominant_size is not None and size_by_id.get(row.image_id) == dominant_size
            if row.status != "ok":
                n_review += 1
            if row.status == "ok" and row.marker_box_abs and row.orientation_group and in_dominant:
                markers_by_group.setdefault(row.orientation_group, []).append(tuple(row.marker_box_abs))
            image_rows.append(
                {
                    "folder": folder.name,
                    "vendor": vendor,
                    "image_id": row.image_id,
                    "pred_group": row.orientation_group,
                    "status": row.status,
                    "review_reason": row.review_reason,
                    "match_score": row.match_score,
                    "search_scope": row.search_scope,
                    "marker_box_abs": "" if not row.marker_box_abs else "|".join(map(str, row.marker_box_abs)),
                    "template_name": row.template_name,
                    "image_size": "" if row.image_id not in size_by_id else f"{size_by_id[row.image_id][0]}x{size_by_id[row.image_id][1]}",
                    "in_dominant_resolution": int(in_dominant),
                }
            )

        for group in GROUPS:
            markers = markers_by_group.get(group, [])
            if len(markers) < args.min_markers_per_envelope:
                continue
            envelope = _union(markers)
            envelope_rows.append(
                {
                    "folder": folder.name,
                    "vendor": vendor,
                    "group": group,
                    "n_markers": len(markers),
                    "envelope_box": "|".join(map(str, envelope)),
                    "resolution": f"{dominant_size[0]}x{dominant_size[1]}" if dominant_size else "",
                }
            )

        elapsed = time.time() - started
        groups_found = ",".join(sorted(markers_by_group)) or "-"
        folder_rows.append(
            {
                "folder": folder.name,
                "vendor": vendor,
                "status": "ok",
                "n_images": len(analysis.rows),
                "review_rate": round(n_review / len(analysis.rows), 4),
                "groups_found": groups_found,
                "template_selected": analysis.template.name,
                "dominant_resolution": f"{dominant_size[0]}x{dominant_size[1]}" if dominant_size else "",
                "n_other_resolution": sum(1 for s in size_by_id.values() if s != dominant_size),
                "seconds": round(elapsed, 1),
            }
        )
        print(f"[{index}/{len(folders)}] {folder.name} [{vendor}] groups={groups_found} review={n_review}/{len(analysis.rows)} ({elapsed:.0f}s)")

        # Incremental write so the review gallery can be rebuilt at any time.
        _write_csv(args.output_dir / "per_image_predictions.csv", image_rows)
        _write_csv(args.output_dir / "folder_envelopes.csv", envelope_rows)
        _write_csv(args.output_dir / "folder_summary.csv", folder_rows)

    ok_rows = [r for r in folder_rows if r.get("status") == "ok"]
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "folders_done": len(folder_rows),
        "folders_ok": len(ok_rows),
        "images_total": len(image_rows),
        "review_rate": round(sum(1 for r in image_rows if r["status"] != "ok") / len(image_rows), 4) if image_rows else None,
        "envelopes_total": len(envelope_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
