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
    "match_score", "search_scope", "marker_box_abs", "template_name", "template_scale", "image_size",
    "vertical_correction", "vertical_source",
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
    if not new_file:
        # Resuming a run written by an older version: honor the existing header
        # so appended rows stay aligned with it.
        with path.open(newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), None)
        if existing:
            fields = existing
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
    parser.add_argument("--vendor", type=str, default="",
                        help="Force the vendor instead of inferring it from the folder name. "
                             "The app passes the vendor its classifier recognised, so the run "
                             "does not depend on the dataset folder-naming convention.")
    parser.add_argument("--folder-regex", type=str, default="")
    parser.add_argument("--include-regex", type=str, default="")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-images-per-folder", type=int, default=0, help="0 = ALL dominant-resolution images.")
    parser.add_argument("--selection-images", type=int, default=4)
    parser.add_argument("--pinned-templates", type=Path, default=None,
                        help="JSON map folder-name -> [vendor/marker_NNN.png, ...]: human-verified "
                             "templates harvested from that folder's review corrections. Evaluated "
                             "per image next to the auto-selected template; win only by margin.")
    parser.add_argument("--pinned-margin", type=float, default=0.03,
                        help="Score margin a pinned template must exceed to replace the auto match.")
    parser.add_argument("--pinned-min-score", type=float, default=0.90,
                        help="Minimum absolute score for a pinned template to take over a frame.")
    parser.add_argument("--template-scales", type=str, default="0.75,1.0,1.3,1.7,2.2",
                        help="Scales tried during folder template selection (marker size varies "
                             "between machines). The winning scale is stored in state.json.")
    parser.add_argument("--min-match-score", type=float, default=0.55)
    parser.add_argument("--fallback-threshold", type=float, default=0.58)
    parser.add_argument("--expanded-threshold", type=float, default=0.62)
    parser.add_argument("--vertical-delta", type=float, default=0.05)
    parser.add_argument("--match-max-side", type=int, default=560)
    parser.add_argument("--min-markers-per-envelope", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--official-stages-dir", type=Path, default=None,
                        help="CHAINED mode: consume official_stages_batch.py outputs. Per image: rect crop + "
                             "SU/GIU prior; per folder: #13 exclusion. Search follows the designed chain: "
                             "predicted half -> other half (flags sugiu_corrected_by_marker) -> expansion.")
    parser.add_argument("--exclusion-rect", type=str, default="",
                        help="Explicit already-expanded #13 exclusion as top|left|bottom|right.")
    parser.add_argument("--exclusion-margin-frac", type=float, default=0.75,
                        help="Expansion of the #13 exclusion rect (vendor logo often sits just outside it).")
    parser.add_argument("--vendor-conf-min", type=float, default=0.35,
                        help="Min CNN confidence for the official vendor to override folder-name "
                             "inference (when the name resolves to an existing template bank).")
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

    # CHAINED mode: load official stages outputs (rect/su-giu per image, #13 per folder).
    def _tlbr(text: str) -> Optional[Tuple[int, int, int, int]]:
        parts = [p for p in str(text or "").split("|") if p.strip()]
        if len(parts) < 4:
            return None
        try:
            return tuple(int(float(p)) for p in parts[:4])  # type: ignore[return-value]
        except ValueError:
            return None

    pinned_map: Dict[str, List[str]] = {}
    if args.pinned_templates and args.pinned_templates.is_file():
        raw = json.loads(args.pinned_templates.read_text(encoding="utf-8"))
        pinned_map = {str(k): [str(x) for x in v] for k, v in raw.items()
                      if isinstance(v, list) and not str(k).startswith("_")}
        print(f"[pinned] {len(pinned_map)} folders with human-pinned templates", flush=True)

    official_img: Dict[Tuple[str, str], Dict[str, str]] = {}
    official_fold: Dict[str, Dict[str, str]] = {}
    if args.official_stages_dir:
        for row in _load_csv(args.official_stages_dir / "official_per_image.csv"):
            official_img[(str(row["folder"]), str(row["image_id"]))] = row
        for row in _load_csv(args.official_stages_dir / "official_folder.csv"):
            official_fold[str(row["folder"])] = row
        print(f"[chained] official stages: {len(official_fold)} folders, {len(official_img)} images", flush=True)

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
    if args.official_stages_dir:
        folders = [p for p in folders if p.name in official_fold]  # chain requires stages 1-3 done
    if args.max_folders > 0:
        folders = folders[: args.max_folders]
    print(f"[start] {len(completed_folders)} folders completed, {len(folders)} to process")

    run_started = time.time()
    out_of_time = False
    for index, folder in enumerate(folders, start=1):
        if out_of_time:
            break
        vendor = (args.vendor or "").strip() or omd.infer_vendor_from_text(folder.name)
        folder_state = state.get(folder.name, {})

        chained = bool(args.official_stages_dir)
        if chained:
            # Prefer the official image-based vendor CNN over folder-name inference:
            # some folders are named after the probe/model only (e.g. "Arietta 65"),
            # with no vendor keyword in the text, which infer_vendor_from_text can't resolve.
            # BUT a low-confidence CNN prediction must not override a vendor that IS
            # spelled out in the folder name (seen: "165.Mindray TE7" -> ExactVu @0.14).
            official_row = official_fold.get(folder.name, {})
            official_vendor = str(official_row.get("vendor_pred") or "").strip()
            try:
                official_conf = float(official_row.get("vendor_conf") or 0.0)
            except ValueError:
                official_conf = 0.0
            name_has_bank = omd._vendor_dir(
                Path(args.library_root).expanduser().resolve() if args.library_root else omd.DEFAULT_LIBRARY_ROOT,
                vendor,
            ) is not None
            if official_vendor and (official_conf >= args.vendor_conf_min or not name_has_bank):
                vendor = official_vendor
        if chained:
            # Same exact image set of the official stages (already dominant-resolution only).
            kept = [folder / image_id for (fname, image_id) in official_img if fname == folder.name]
            kept.sort()
            n_other = 0
            if "dominant_resolution" in folder_state:
                dom_w, dom_h = map(int, str(folder_state["dominant_resolution"]).split("x"))
            else:
                first_size = _image_size(kept[0]) if kept else None
                dom_w, dom_h = first_size if first_size else (0, 0)
                folder_state["dominant_resolution"] = f"{dom_w}x{dom_h}"
                state[folder.name] = folder_state
                state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))
            dominant = (dom_w, dom_h)
            if not kept:
                _append_csv(out / "folder_summary.csv",
                            [{"folder": folder.name, "vendor": vendor, "status": "no_images"}], FOLDER_FIELDS)
                completed_folders.add(folder.name)
                continue
        else:
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

        # CHAINED step 1: exclusion zone from the vendor-template box (#13) + margin.
        exclusion_rects: Tuple[Tuple[int, int, int, int], ...] = ()
        explicit_exclusion = _tlbr(args.exclusion_rect)
        if explicit_exclusion:
            exclusion_rects = (explicit_exclusion,)
        elif chained:
            box13 = _tlbr(official_fold.get(folder.name, {}).get("line13_text", ""))
            if box13:
                top13, left13, bottom13, right13 = box13
                mh = int((bottom13 - top13 + 1) * args.exclusion_margin_frac)
                mw = int((right13 - left13 + 1) * args.exclusion_margin_frac)
                exclusion_rects = ((top13 - mh, left13 - mw, bottom13 + mh, right13 + mw),)

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
        sel_scales = tuple(float(s) for s in str(args.template_scales).split(",") if s.strip()) or (1.0,)
        if folder_state.get("template_path"):
            template = next((t for t in templates if str(t.path) == folder_state["template_path"]), None)
        template_scale = float(folder_state.get("template_scale", 1.0))
        candidates = [(template, template_scale)] if template is not None else []
        if template is None:
            selection = [
                omd.ImageInput(image_path=p, image_id=p.relative_to(folder).as_posix())
                for p in kept[:: max(1, len(kept) // max(1, args.selection_images))][: args.selection_images]
            ]
            try:
                template, _rows, rank = omd.select_best_template(selection, templates, params=params, scales=sel_scales)
                template_scale = float(rank[0].get("scale", 1.0)) if rank else 1.0
            except TypeError:  # older bundle without multi-scale support
                template, _rows, _rank = omd.select_best_template(selection, templates, params=params)
                template_scale = 1.0
            candidates = [(template, template_scale)]
            folder_state["template_path"] = str(template.path)
            folder_state["template_scale"] = template_scale
            state[folder.name] = folder_state
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))

        # Human-pinned templates (harvested from THIS folder's review corrections):
        # evaluated per image alongside the auto-selected one; they win only when
        # clearly better, so unpinned folders behave exactly as before.
        pinned: List[Tuple[object, float]] = []
        primary_path = str(candidates[0][0].path)
        for rel in pinned_map.get(folder.name, []):
            suffix = str(rel).replace("\\", "/").lower()
            cand = next((t for t in templates if str(t.path).replace("\\", "/").lower().endswith(suffix)), None)
            if cand is None:
                print(f"[pinned] WARN not resolved for {folder.name[:40]}: {rel}", flush=True)
            elif str(cand.path) != primary_path:
                pinned.append((cand, 1.0))

        pending = [p for p in kept if (folder.name, p.relative_to(folder).as_posix()) not in done_pairs]
        buffer: List[Dict[str, object]] = []
        processed_now = 0
        for path in pending:
            if args.time_budget > 0 and time.time() - run_started > args.time_budget:
                out_of_time = True
                break
            image_id = path.relative_to(folder).as_posix()
            crop_rect = None
            sugiu_pred = ""
            sugiu_conf = 0.0
            if chained:
                # CHAINED steps 2-3: per-image rect + SU/GIU prior from the official nets.
                info = official_img.get((folder.name, image_id), {})
                crop_rect = _tlbr(info.get("rect_box_abs", ""))
                label = str(info.get("sugiu_label", "")).strip().lower()
                if label in ("su", "giu"):
                    sugiu_pred = label
                    try:
                        sugiu_conf = float(info.get("sugiu_conf", 0.0) or 0.0)
                    except ValueError:
                        sugiu_conf = 0.0
            image_input = omd.ImageInput(
                image_path=path,
                image_id=image_id,
                crop_rect=crop_rect,
                crop_source="official_stages" if crop_rect else "",
                sugiu_pred=sugiu_pred,
                sugiu_conf=sugiu_conf,
                exclusion_rects=exclusion_rects,
            )
            # Folder-selected scale first; neighbor scales only when the score is
            # weak (marker size can change between frames of the same folder).
            row = omd.detect_marker(image_input, template, params=params, scales=(template_scale,))
            if (row.match_score or -1.0) < args.fallback_threshold:
                alt = omd.detect_marker(
                    image_input, template, params=params,
                    scales=(template_scale * 0.8, template_scale * 1.25),
                )
                if (alt.match_score or -1.0) > (row.match_score or -1.0):
                    row = alt
            # Human-pinned templates win only on near-perfect evidence: score at
            # least --pinned-min-score (a true machine template reaches ~1.0 on
            # its frames, static twin glyphs top out lower) AND clearly above
            # the auto-selected match. So they can never degrade a frame.
            for pin_tpl, pin_scale in pinned:
                pin_row = omd.detect_marker(image_input, pin_tpl, params=params, scales=(pin_scale,))
                pin_score = pin_row.match_score or -1.0
                if pin_score >= args.pinned_min_score and pin_score > (row.match_score or -1.0) + args.pinned_margin:
                    row = pin_row
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
                    "template_scale": getattr(row, "template_scale", 1.0),
                    "image_size": f"{dominant[0]}x{dominant[1]}",
                    "vertical_correction": row.vertical_correction,
                    "vertical_source": row.vertical_source,
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
        n_sugiu_corrected = sum(1 for r in folder_img_rows if r.get("vertical_correction") == "corrected")
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
                    "n_sugiu_corrected": n_sugiu_corrected,
                }
            ],
            FOLDER_FIELDS,
        )
        completed_folders.add(folder.name)
        print(f"[{index}/{len(folders)}] {folder.name} [{vendor}] DONE {len(folder_img_rows)} img, "
              f"groups={groups_found}, review={n_review}, sugiu_corrected={n_sugiu_corrected}")

    print(f"[end] completed folders: {len(completed_folders)}")
    return 0


FOLDER_FIELDS = [
    "folder", "vendor", "status", "n_images", "n_other_resolution", "review_rate",
    "marker_found_rate", "groups_found", "template_selected", "dominant_resolution",
    "n_sugiu_corrected",
]
ENVELOPE_FIELDS = ["folder", "vendor", "group", "n_markers", "envelope_box", "resolution"]


if __name__ == "__main__":
    raise SystemExit(main())
