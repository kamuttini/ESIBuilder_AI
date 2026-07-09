#!/usr/bin/env python3
"""Production-like marker + orientation-envelope prediction on raw acquisition folders.

For each top-level configuration folder (no `.fss` available, e.g. volume
SSD_esi1_n3/ACQUISITION ELABORATION):

1. collect acquisition frames (recursive; excludes Thumbs/AppleDouble/PROIBITE)
2. keep ONLY the dominant resolution of the folder (mixed resolutions are
   counted and skipped)
3. vendor from folder name -> historical template bank; best template chosen
   once per folder (persisted in state.json)
4. marker detection on ALL kept images (image-level checkpoint: safe to stop
   and resume at any moment with --resume)
5. when a folder completes: one ENVELOPE box per orientation group (same
   coordinates for the whole folder, containing all accepted markers)

Outputs:
- per_image_predictions.csv (appended incrementally)
- folder_envelopes.csv, folder_summary.csv (appended at folder completion)
- state.json (per-folder template/resolution cache)

Example:
  python3 tools/orientation/predict_marker_envelopes_batch.py \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
    --output-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run2 \
    --resume --time-budget 0
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

IMAGE_FIELDS = [
    "folder", "vendor", "image_id", "pred_group", "status", "review_reason",
    "match_score", "search_scope", "marker_box_abs", "template_name", "image_size",
]


def _import_bundle(bundle_dir: Path):
    root = bundle_dir.expanduser().resolve()
    if not (root / "orientation_marker_detector" / "detector.py").is_file():
        raise FileNotFoundError(f"Bundle not found: {root}")
    sys.path.insert(0, str(root))
    from orientation_marker_detector import detector as omd  # type: ignore

    return omd


def _collect_images(folder: Path, include_re: Optional[re.Pattern]) -> List[Path]:
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
    return paths


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001
        return None


def _union(boxes: Sequence[Rect]) -> Rect:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _append_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or not path.stat().st_size
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--library-root", type=Path, default=None)
    parser.add_argument("--folder-regex", type=str, default="")
    parser.add_argument("--include-regex", type=str, default="")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-images-per-folder", type=int, default=0, help="0 = ALL dominant-resolution images.")
    parser.add_argument("--selection-images", type=int, default=4)
    parser.add_argument("--min-match-score", type=float, default=0.55)
    parser.add_argument("--fallback-threshold", type=float, default=0.58)
    parser.add_argument("--expanded-threshold", type=float, default=0.62)
    parser.add_argument("--vertical-delta", type=float, default=0.05)
    parser.add_argument("--match-max-side", type=int, default=560)
    parser.add_argument("--min-markers-per-envelope", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=0.0, help="Stop gracefully after N seconds (0 = no limit).")
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

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state: Dict[str, Dict[str, object]] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text())

    done_pairs: set = set()
    completed_folders: set = set()
    if args.resume:
        for row in _load_csv(out / "per_image_predictions.csv"):
            done_pairs.add((row["folder"], row["image_id"]))
        for row in _load_csv(out / "folder_summary.csv"):
            completed_folders.add(row["folder"])

    folders = [
        p
        for p in sorted(args.dataset_root.iterdir())
        if p.is_dir() and not p.name.startswith(("$", ".")) and p.name != "System Volume Information"
        and (folder_pattern is None or folder_pattern.search(p.name))
        and p.name not in completed_folders
    ]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]
    print(f"[start] {len(completed_folders)} folders completed, {len(folders)} to process")

    run_started = time.time()
    out_of_time = False
    for index, folder in enumerate(folders, start=1):
        if out_of_time:
            break
        vendor = omd.infer_vendor_from_text(folder.name)
        folder_state = state.get(folder.name, {})

        paths = _collect_images(folder, include_re)
        if not paths:
            _append_csv(out / "folder_summary.csv",
                        [{"folder": folder.name, "vendor": vendor, "status": "no_images"}], FOLDER_FIELDS)
            completed_folders.add(folder.name)
            print(f"[{index}/{len(folders)}] {folder.name}: no images")
            continue

        # Dominant resolution (cached in state to avoid re-reading all sizes on resume).
        if "dominant_resolution" in folder_state and "n_other_resolution" in folder_state:
            dom_w, dom_h = map(int, str(folder_state["dominant_resolution"]).split("x"))
            dominant = (dom_w, dom_h)
            kept = [p for p in paths if _image_size(p) == dominant]
            n_other = len(paths) - len(kept)
        else:
            sizes: Dict[Path, Optional[Tuple[int, int]]] = {p: _image_size(p) for p in paths}
            counter = Counter(s for s in sizes.values() if s)
            if not counter:
                _append_csv(out / "folder_summary.csv",
                            [{"folder": folder.name, "vendor": vendor, "status": "no_readable_images"}], FOLDER_FIELDS)
                completed_folders.add(folder.name)
                continue
            dominant = counter.most_common(1)[0][0]
            kept = [p for p, s in sizes.items() if s == dominant]
            n_other = len(paths) - len(kept)
            folder_state["dominant_resolution"] = f"{dominant[0]}x{dominant[1]}"
            folder_state["n_other_resolution"] = n_other
            state[folder.name] = folder_state
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))

        if args.max_images_per_folder > 0 and len(kept) > args.max_images_per_folder:
            stride = len(kept) / args.max_images_per_folder
            kept = [kept[int(i * stride)] for i in range(args.max_images_per_folder)]

        # Template: select once per folder, persist in state.
        template = None
        templates = omd.load_vendor_templates(args.library_root, vendor=vendor)
        if not templates:
            _append_csv(out / "folder_summary.csv",
                        [{"folder": folder.name, "vendor": vendor, "status": "error:no_templates"}], FOLDER_FIELDS)
            completed_folders.add(folder.name)
            print(f"[{index}/{len(folders)}] {folder.name}: ERROR no templates for {vendor}")
            continue
        if folder_state.get("template_path"):
            template = next((t for t in templates if str(t.path) == folder_state["template_path"]), None)
        if template is None:
            selection = [
                omd.ImageInput(image_path=p, image_id=p.relative_to(folder).as_posix())
                for p in kept[:: max(1, len(kept) // max(1, args.selection_images))][: args.selection_images]
            ]
            template, _rows, _rank = omd.select_best_template(selection, templates, params=params)
            folder_state["template_path"] = str(template.path)
            state[folder.name] = folder_state
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))

        pending = [p for p in kept if (folder.name, p.relative_to(folder).as_posix()) not in done_pairs]
        buffer: List[Dict[str, object]] = []
        processed_now = 0
        for path in pending:
            if args.time_budget > 0 and time.time() - run_started > args.time_budget:
                out_of_time = True
                break
            image_id = path.relative_to(folder).as_posix()
            row = omd.detect_marker(
                omd.ImageInput(image_path=path, image_id=image_id), template, params=params
            )
            buffer.append(
                {
                    "folder": folder.name,
                    "vendor": vendor,
                    "image_id": image_id,
                    "pred_group": row.orientation_group,
                    "status": row.status,
                    "review_reason": row.review_reason,
                    "match_score": row.match_score,
                    "search_scope": row.search_scope,
                    "marker_box_abs": "" if not row.marker_box_abs else "|".join(map(str, row.marker_box_abs)),
                    "template_name": row.template_name,
                    "image_size": f"{dominant[0]}x{dominant[1]}",
                }
            )
            done_pairs.add((folder.name, image_id))
            processed_now += 1
            if len(buffer) >= args.flush_every:
                _append_csv(out / "per_image_predictions.csv", buffer, IMAGE_FIELDS)
                buffer = []
        _append_csv(out / "per_image_predictions.csv", buffer, IMAGE_FIELDS)

        remaining = len(pending) - processed_now
        if remaining > 0:
            print(f"[{index}/{len(folders)}] {folder.name} [{vendor}] PARTIAL {processed_now} img, {remaining} left; --resume to continue")
            break

        # Folder complete -> envelopes + summary from all its rows.
        folder_img_rows = [r for r in _load_csv(out / "per_image_predictions.csv") if r["folder"] == folder.name]
        markers_by_group: Dict[str, List[Rect]] = {}
        n_review = 0
        for r in folder_img_rows:
            if r["status"] != "ok":
                n_review += 1
            if r["status"] == "ok" and r["marker_box_abs"] and r["pred_group"]:
                box = tuple(int(v) for v in r["marker_box_abs"].split("|"))
                markers_by_group.setdefault(r["pred_group"], []).append(box)
        env_rows = []
        for group in GROUPS:
            markers = markers_by_group.get(group, [])
            if len(markers) < args.min_markers_per_envelope:
                continue
            env_rows.append(
                {
                    "folder": folder.name,
                    "vendor": vendor,
                    "group": group,
                    "n_markers": len(markers),
                    "envelope_box": "|".join(map(str, _union(markers))),
                    "resolution": f"{dominant[0]}x{dominant[1]}",
                }
            )
        _append_csv(out / "folder_envelopes.csv", env_rows, ENVELOPE_FIELDS)
        groups_found = ",".join(sorted(markers_by_group)) or "-"
        _append_csv(
            out / "folder_summary.csv",
            [
                {
                    "folder": folder.name,
                    "vendor": vendor,
                    "status": "ok",
                    "n_images": len(folder_img_rows),
                    "n_other_resolution": folder_state.get("n_other_resolution", n_other),
                    "review_rate": round(n_review / max(1, len(folder_img_rows)), 4),
                    "marker_found_rate": round(
                        sum(1 for r in folder_img_rows if r["marker_box_abs"]) / max(1, len(folder_img_rows)), 4
                    ),
                    "groups_found": groups_found,
                    "template_selected": template.name,
                    "dominant_resolution": f"{dominant[0]}x{dominant[1]}",
                }
            ],
            FOLDER_FIELDS,
        )
        completed_folders.add(folder.name)
        print(f"[{index}/{len(folders)}] {folder.name} [{vendor}] DONE {len(folder_img_rows)} img, groups={groups_found}, review={n_review}")

    print(f"[end] completed folders: {len(completed_folders)}")
    return 0


FOLDER_FIELDS = [
    "folder", "vendor", "status", "n_images", "n_other_resolution", "review_rate",
    "marker_found_rate", "groups_found", "template_selected", "dominant_resolution",
]
ENVELOPE_FIELDS = ["folder", "vendor", "group", "n_markers", "envelope_box", "resolution"]


if __name__ == "__main__":
    raise SystemExit(main())
