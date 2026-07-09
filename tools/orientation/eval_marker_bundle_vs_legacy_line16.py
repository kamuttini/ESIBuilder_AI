#!/usr/bin/env python3
"""Evaluate the orientation marker bundle against legacy `.fss` line #16.

For each legacy configuration folder (containing `DB_setup/setup_<ID>.fss`):
1. Parse the legacy `.fss`: RECT_ECHO (line 11, fallback 10) used as crop rect,
   and the 4 line-16 boxes (NF/LR/UD/LRUD) used as ground-truth envelopes.
2. Collect full-frame sample images; the per-image GT orientation group is
   derived from filename tokens (`no_flip` -> NF, `flip_lr` -> LR, ...) with a
   folder-name fallback (e.g. "(SOLO UD)").
3. Run the marker detector in BLIND mode: the detector never sees the GT
   (no expected_group, so template selection and status are unbiased).
   `--vertical-prior gt` optionally feeds the GT vertical as a perfect
   SU/GIU-network simulation.
4. Report per-image group accuracy, review rate, and per-group envelope IoU
   (union of accepted marker boxes vs the legacy line-16 box).

Example:
  python3 tools/orientation/eval_marker_bundle_vs_legacy_line16.py \
    --dataset-root /Volumes/SSD_esi1_n1 \
    --output-dir artifacts/42_orientation_eval_vs_legacy_line16/pilot \
    --max-folders 15
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle"

Rect = Tuple[int, int, int, int]  # top, left, bottom, right (inclusive)

RECT_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|?")
GROUPS = ("NF", "LR", "UD", "LRUD")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {"DB_setup", "DB_echo", "temp", "$RECYCLE.BIN", ".Spotlight-V100", ".fseventsd"}
DEFAULT_INCLUDE_RE = r"image_depth_find|image_orientation_setup"
DEFAULT_EXCLUDE_RE = r"image_th_|proibite|calibration|Thumbs\.db"


def _import_bundle(bundle_dir: Path):
    root = bundle_dir.expanduser().resolve()
    if not (root / "orientation_marker_detector" / "detector.py").is_file():
        raise FileNotFoundError(f"Bundle not found: {root}")
    sys.path.insert(0, str(root))
    from orientation_marker_detector import detector as omd  # type: ignore

    return omd


@dataclass
class FolderGT:
    folder: Path
    fss_path: Path
    setup_id: str
    rect_echo: Rect
    line16_boxes: Dict[str, Rect]
    line16_duplicated: Dict[str, bool]
    name_rects: List[Rect] = field(default_factory=list)  # #13 echo name, #14 probe name
    warnings: List[str] = field(default_factory=list)


def _parse_name_rects(lines: List[str], rect_line_no: int) -> List[Rect]:
    """Boxes of line #13 (RECT_NAME_ECHO) and #14 (RECT_NAME_PROBE).

    Vendor/probe name templates can look like orientation markers (e.g. the
    Mindray "m" logo), so the marker search must exclude these regions.
    """
    rects: List[Rect] = []
    for offset in (2, 3):  # rect_line + 2 -> #13, rect_line + 3 -> #14
        line_no = rect_line_no + offset
        if line_no > len(lines):
            continue
        first_segment = lines[line_no - 1].strip().split(";")[0]
        parts = first_segment.split("|")
        if len(parts) >= 4:
            try:
                rects.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            except ValueError:
                continue
    return rects


def _parse_fss(fss_path: Path) -> Tuple[Rect, Dict[str, Rect], Dict[str, bool], List[Rect]]:
    lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rect_line_no: Optional[int] = None
    rect: Optional[Rect] = None
    for candidate in (11, 10):
        if candidate <= len(lines):
            match = RECT_RE.match(lines[candidate - 1].strip())
            if match:
                rect_line_no = candidate
                top, left, bottom, right = (int(match.group(i)) for i in range(1, 5))
                rect = (top, left, bottom, right)
                break
    if rect is None or rect_line_no is None:
        raise ValueError("RECT_ECHO line not found")

    line16_no = rect_line_no + 5
    if line16_no > len(lines):
        raise ValueError("line #16 missing")
    raw = lines[line16_no - 1].strip()
    segments = [seg.strip().rstrip(",") for seg in raw.split(";") if seg.strip()]
    if len(segments) < 4:
        raise ValueError(f"line #16 has {len(segments)} segments (expected 4)")

    boxes: Dict[str, Rect] = {}
    for group, segment in zip(GROUPS, segments[:4]):
        parts = segment.split("|")
        if len(parts) < 4:
            raise ValueError(f"malformed line #16 segment for {group}: {segment!r}")
        boxes[group] = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))

    duplicated = {g: (g != "NF" and boxes[g] == boxes["NF"]) for g in GROUPS}
    # All 4 entries identical -> legacy project calibrated for a SINGLE orientation
    # (old workaround for mirrored templates). Slot order does NOT mean NF/LR/UD/LRUD
    # there, so line-16 boxes cannot be used as per-orientation GT.
    return rect, boxes, duplicated, _parse_name_rects(lines, rect_line_no)


def _scan_folders(dataset_root: Path, folder_regex: str, max_folders: int) -> List[FolderGT]:
    pattern = re.compile(folder_regex, re.IGNORECASE) if folder_regex else None
    out: List[FolderGT] = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if folder.name in SKIP_DIR_NAMES or folder.name.startswith((".", "$")):
            continue
        if pattern and not pattern.search(folder.name):
            continue
        fss_candidates = sorted((folder / "DB_setup").glob("setup_*.fss"))
        if not fss_candidates:
            continue
        fss_path = fss_candidates[0]
        gt_warnings = []
        if len(fss_candidates) > 1:
            gt_warnings.append(f"multiple .fss, using {fss_path.name}")
        try:
            rect, boxes, duplicated, name_rects = _parse_fss(fss_path)
        except ValueError as exc:
            print(f"[skip] {folder.name}: {exc}")
            continue
        out.append(
            FolderGT(
                folder=folder,
                fss_path=fss_path,
                setup_id=fss_path.stem.replace("setup_", ""),
                rect_echo=rect,
                line16_boxes=boxes,
                line16_duplicated=duplicated,
                name_rects=name_rects,
                warnings=gt_warnings,
            )
        )
        if max_folders > 0 and len(out) >= max_folders:
            break
    return out


def _collect_images(
    folder: Path,
    include_re: re.Pattern,
    exclude_re: re.Pattern,
    max_images: int,
    group_from_text,
) -> List[Tuple[Path, str, str]]:
    """Return (path, gt_group, gt_source). Stratified uniform sampling per GT group."""
    # GT only from DIRECTORY names (root folder or subfolders like "SOLO UD", "NoFlip", "LR").
    # Filename tokens like "image_depth_find_flip_lr_setup" refer to the legacy calibration
    # branch, NOT the actual on-screen orientation (verified visually) -> never used as GT.
    folder_group = group_from_text(folder.name)
    by_group: Dict[str, List[Tuple[Path, str, str]]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name.startswith("._"):
            continue
        rel = path.relative_to(folder).as_posix()
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(folder).parts):
            continue
        if exclude_re.search(rel) or not include_re.search(rel):
            continue
        gt_group, gt_source = "", ""
        for part in reversed(path.relative_to(folder).parts[:-1]):  # deepest subdir wins
            part_group = group_from_text(part)
            if part_group:
                gt_group, gt_source = part_group, f"subdir:{part}"
                break
        if not gt_group and folder_group:
            gt_group, gt_source = folder_group, "folder_name"
        by_group.setdefault(gt_group, []).append((path, gt_group, gt_source))

    if max_images <= 0:
        return [item for items in by_group.values() for item in items]

    total = sum(len(v) for v in by_group.values())
    selected: List[Tuple[Path, str, str]] = []
    for items in by_group.values():
        quota = max(1, round(max_images * len(items) / max(1, total)))
        if len(items) <= quota:
            selected.extend(items)
        else:
            stride = len(items) / quota
            selected.extend(items[int(i * stride)] for i in range(quota))
    return selected[:max_images] if len(selected) > max_images else selected


def _iou(a: Rect, b: Rect) -> float:
    top = max(a[0], b[0])
    left = max(a[1], b[1])
    bottom = min(a[2], b[2])
    right = min(a[3], b[3])
    if bottom < top or right < left:
        return 0.0
    inter = (bottom - top + 1) * (right - left + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--library-root", type=Path, default=None, help="Template library (default: bundle templates).")
    parser.add_argument("--folder-regex", type=str, default="")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-images-per-folder", type=int, default=16)
    parser.add_argument("--selection-images", type=int, default=8)
    parser.add_argument("--include-regex", type=str, default=DEFAULT_INCLUDE_RE)
    parser.add_argument("--exclude-regex", type=str, default=DEFAULT_EXCLUDE_RE)
    parser.add_argument("--vertical-prior", choices=("none", "gt"), default="none",
                        help="'gt' simulates a perfect SU/GIU network via the GT group vertical.")
    parser.add_argument("--no-name-exclusion", action="store_true",
                        help="Do NOT exclude the legacy #13/#14 name rects from the marker search.")
    parser.add_argument("--exclusion-margin-frac", type=float, default=0.75,
                        help="Expand each #13/#14 exclusion rect by this fraction of its own size "
                             "(vendor logos often sit just outside the legacy name box, e.g. Mindray 'm').")
    parser.add_argument("--min-match-score", type=float, default=0.55)
    parser.add_argument("--fallback-threshold", type=float, default=0.58)
    parser.add_argument("--expanded-threshold", type=float, default=0.62)
    parser.add_argument("--vertical-delta", type=float, default=0.05)
    parser.add_argument("--match-max-side", type=int, default=720)
    parser.add_argument("--min-markers-per-envelope", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="Skip folders already present in <output-dir>/folder_summary.csv and append results.")
    parser.add_argument("--time-budget", type=float, default=0.0,
                        help="Stop gracefully after N seconds (resume later with --resume).")
    args = parser.parse_args()

    omd = _import_bundle(args.bundle_dir)
    params = omd.DetectionParams(
        min_match_score=args.min_match_score,
        vertical_delta=args.vertical_delta,
        fallback_threshold=args.fallback_threshold,
        expanded_threshold=args.expanded_threshold,
        match_max_side=args.match_max_side,
    )
    include_re = re.compile(args.include_regex, re.IGNORECASE)
    exclude_re = re.compile(args.exclude_regex, re.IGNORECASE)

    folders = _scan_folders(args.dataset_root, args.folder_regex, args.max_folders)
    if not folders:
        print("No folders with DB_setup/setup_*.fss found.")
        return 2

    image_rows: List[Dict[str, object]] = []
    folder_rows: List[Dict[str, object]] = []
    envelope_rows: List[Dict[str, object]] = []

    def _load_csv(path: Path) -> List[Dict[str, object]]:
        if not path.is_file() or not path.stat().st_size:
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if args.resume:
        folder_rows = _load_csv(args.output_dir / "folder_summary.csv")
        image_rows = _load_csv(args.output_dir / "per_image_predictions.csv")
        envelope_rows = _load_csv(args.output_dir / "envelope_iou.csv")
        done = {str(r["folder"]) for r in folder_rows}
        folders = [gt for gt in folders if gt.folder.name not in done]
        print(f"[resume] {len(done)} folders already done, {len(folders)} remaining")
        if not folders:
            print("[resume] nothing to do")

    run_started = time.time()
    for index, gt in enumerate(folders, start=1):
        if args.time_budget > 0 and time.time() - run_started > args.time_budget:
            print(f"[time-budget] stopping after {index - 1} folders; rerun with --resume")
            break
        started = time.time()
        vendor = omd.infer_vendor_from_text(gt.folder.name)
        samples = _collect_images(
            gt.folder, include_re, exclude_re, args.max_images_per_folder, omd._orientation_group_from_text
        )
        if not samples:
            folder_rows.append({"folder": gt.folder.name, "vendor": vendor, "status": "no_images"})
            print(f"[{index}/{len(folders)}] {gt.folder.name}: no images")
            continue

        if args.no_name_exclusion:
            exclusion = ()
        else:
            frac = max(0.0, args.exclusion_margin_frac)
            exclusion = tuple(
                (
                    int(top - (bottom - top + 1) * frac),
                    int(left - (right - left + 1) * frac),
                    int(bottom + (bottom - top + 1) * frac),
                    int(right + (right - left + 1) * frac),
                )
                for top, left, bottom, right in gt.name_rects
            )
        inputs = []
        for path, gt_group, _source in samples:
            sugiu = omd._vertical_from_group(gt_group) if (args.vertical_prior == "gt" and gt_group) else ""
            inputs.append(
                omd.ImageInput(
                    image_path=path,
                    image_id=path.relative_to(gt.folder).as_posix(),
                    crop_rect=gt.rect_echo,
                    crop_source="legacy_fss_line11",
                    sugiu_pred=sugiu,
                    sugiu_conf=1.0 if sugiu else 0.0,
                    exclusion_rects=exclusion,
                )
            )

        try:
            analysis = omd.analyze_images(
                inputs, vendor=vendor,
                library_root=args.library_root, params=params,
                selection_images=args.selection_images,
            )
        except RuntimeError as exc:
            folder_rows.append({"folder": gt.folder.name, "vendor": vendor, "status": f"error:{exc}"})
            print(f"[{index}/{len(folders)}] {gt.folder.name}: ERROR {exc}")
            continue

        # Groups whose legacy line-16 box is unique (not a copy of NF -> "not calibrated").
        single_orientation = all(gt.line16_duplicated[g] for g in GROUPS if g != "NF")
        unique_groups = (
            [] if single_orientation else [g for g in GROUPS if g == "NF" or not gt.line16_duplicated[g]]
        )
        gt_by_id = {path.relative_to(gt.folder).as_posix(): (group, source) for path, group, source in samples}
        n_eval = n_correct = n_review = 0
        n_box_eval = n_box_agree = 0
        markers_by_group: Dict[str, List[Rect]] = {}
        for row in analysis.rows:
            gt_group, gt_source = gt_by_id.get(row.image_id, ("", ""))
            correct: Optional[bool] = None
            if gt_group and row.orientation_group:
                n_eval += 1
                correct = row.orientation_group == gt_group
                n_correct += int(correct)
            # Positional GT: unique legacy box containing the marker center.
            gt_by_box = ""
            if row.marker_box_abs:
                m = row.marker_box_abs
                cx, cy = (m[1] + m[3]) / 2.0, (m[0] + m[2]) / 2.0
                containing = [
                    g
                    for g in unique_groups
                    if gt.line16_boxes[g][0] <= cy <= gt.line16_boxes[g][2]
                    and gt.line16_boxes[g][1] <= cx <= gt.line16_boxes[g][3]
                ]
                if len(containing) == 1:
                    gt_by_box = containing[0]
            if gt_by_box and row.orientation_group:
                n_box_eval += 1
                n_box_agree += int(gt_by_box == row.orientation_group)
            if row.status != "ok":
                n_review += 1
            if row.status == "ok" and row.marker_box_abs and row.orientation_group:
                markers_by_group.setdefault(row.orientation_group, []).append(tuple(row.marker_box_abs))
            image_rows.append(
                {
                    "folder": gt.folder.name,
                    "vendor": vendor,
                    "image_id": row.image_id,
                    "gt_group": gt_group,
                    "gt_source": gt_source,
                    "gt_by_box": gt_by_box,
                    "pred_group": row.orientation_group,
                    "correct": "" if correct is None else int(correct),
                    "box_agree": "" if not (gt_by_box and row.orientation_group) else int(gt_by_box == row.orientation_group),
                    "status": row.status,
                    "review_reason": row.review_reason,
                    "match_score": row.match_score,
                    "search_scope": row.search_scope,
                    "vertical_source": row.vertical_source,
                    "vertical_correction": row.vertical_correction,
                    "marker_box_abs": "" if not row.marker_box_abs else "|".join(map(str, row.marker_box_abs)),
                    "template_name": row.template_name,
                }
            )

        iou_by_group: Dict[str, Optional[float]] = {g: None for g in GROUPS}
        for group in GROUPS if not single_orientation else []:
            markers = markers_by_group.get(group, [])
            if len(markers) < args.min_markers_per_envelope:
                continue
            pred_box = _union(markers)
            gt_box = gt.line16_boxes[group]
            iou = _iou(pred_box, gt_box)
            iou_by_group[group] = iou
            inside = sum(
                1
                for m in markers
                if gt_box[0] <= (m[0] + m[2]) / 2.0 <= gt_box[2] and gt_box[1] <= (m[1] + m[3]) / 2.0 <= gt_box[3]
            )
            envelope_rows.append(
                {
                    "folder": gt.folder.name,
                    "vendor": vendor,
                    "group": group,
                    "n_markers": len(markers),
                    "n_markers_in_gt": inside,
                    "marker_in_gt_rate": round(inside / len(markers), 4),
                    "pred_box": "|".join(map(str, pred_box)),
                    "gt_box": "|".join(map(str, gt_box)),
                    "iou": round(iou, 4),
                    "gt_duplicated_from_nf": int(gt.line16_duplicated[group]),
                }
            )

        elapsed = time.time() - started
        folder_rows.append(
            {
                "folder": gt.folder.name,
                "vendor": vendor,
                "status": "ok",
                "setup_id": gt.setup_id,
                "n_images": len(analysis.rows),
                "n_gt_eval": n_eval,
                "n_correct": n_correct,
                "accuracy": round(n_correct / n_eval, 4) if n_eval else "",
                "n_box_eval": n_box_eval,
                "box_agree_rate": round(n_box_agree / n_box_eval, 4) if n_box_eval else "",
                "line16_single_orientation": int(single_orientation),
                "review_rate": round(n_review / len(analysis.rows), 4),
                "template_selected": analysis.template.name,
                **{f"iou_{g.lower()}": ("" if iou_by_group[g] is None else round(iou_by_group[g], 4)) for g in GROUPS},
                "warnings": ";".join(gt.warnings),
                "seconds": round(elapsed, 1),
            }
        )
        acc_txt = f"{n_correct}/{n_eval}" if n_eval else "n/a"
        print(f"[{index}/{len(folders)}] {gt.folder.name} [{vendor}] acc={acc_txt} review={n_review}/{len(analysis.rows)} ({elapsed:.0f}s)")

    out = args.output_dir
    _write_csv(out / "per_image_predictions.csv", image_rows)
    _write_csv(out / "folder_summary.csv", folder_rows)
    _write_csv(out / "envelope_iou.csv", envelope_rows)

    eval_rows = [r for r in image_rows if r["correct"] != ""]
    ok_folders = [r for r in folder_rows if r.get("status") == "ok"]
    ious = [float(r["iou"]) for r in envelope_rows if not int(r["gt_duplicated_from_nf"] or 0)]
    per_vendor: Dict[str, Dict[str, float]] = {}
    for row in eval_rows:
        bucket = per_vendor.setdefault(str(row["vendor"]), {"eval": 0, "correct": 0})
        bucket["eval"] += 1
        bucket["correct"] += int(row["correct"])
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "folders_total": len(folder_rows),
        "folders_ok": len(ok_folders),
        "images_total": len(image_rows),
        "images_with_gt": len(eval_rows),
        "group_accuracy": round(sum(int(r["correct"]) for r in eval_rows) / len(eval_rows), 4) if eval_rows else None,
        "box_agree_rate": (
            round(
                sum(int(r["box_agree"]) for r in image_rows if r.get("box_agree", "") != "")
                / max(1, sum(1 for r in image_rows if r.get("box_agree", "") != "")),
                4,
            )
            if any(r.get("box_agree", "") != "" for r in image_rows)
            else None
        ),
        "review_rate": round(sum(1 for r in image_rows if r["status"] != "ok") / len(image_rows), 4) if image_rows else None,
        "envelopes_evaluated": len(envelope_rows),
        "envelope_iou_median_excl_dup": round(sorted(ious)[len(ious) // 2], 4) if ious else None,
        "marker_in_gt_rate_overall": (
            round(
                sum(int(r["n_markers_in_gt"]) for r in envelope_rows) / sum(int(r["n_markers"]) for r in envelope_rows),
                4,
            )
            if envelope_rows
            else None
        ),
        "per_vendor_accuracy": {
            vendor: {"images": int(b["eval"]), "accuracy": round(b["correct"] / b["eval"], 4)}
            for vendor, b in sorted(per_vendor.items())
            if b["eval"]
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
