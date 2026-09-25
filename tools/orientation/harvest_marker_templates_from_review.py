#!/usr/bin/env python3
"""Harvest new marker templates from human-corrected boxes (review gallery CSV).

Reads `review_marker_correzioni.csv` exported by the review gallery
(`build_marker_envelope_review_gallery.py`): every row with a redrawn box
(corr_*_pct columns) is a human-verified marker location. Crops those regions
from the ORIGINAL images and adds them to the historical template bank of the
predicted vendor, deduplicating against both the harvest itself and the
already-accepted bank templates.

`review_decisions.json` is updated (accepted list + maintenance_log entry) so
`load_vendor_templates()` picks the new templates up immediately.

Example:
  python3 tools/orientation/harvest_marker_templates_from_review.py \
    --corrections-csv ~/Downloads/review_marker_correzioni.csv \
    --run-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run_full_chained \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
    --library-root artifacts/41_orientation_marker_detector_bundle/orientation_marker_detector/templates
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle/orientation_marker_detector/templates"
COMPARE_SIZE = (32, 32)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation between two grayscale crops (resized to a common size)."""
    if cv2 is not None:
        a_r = cv2.resize(a, COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float64)
        b_r = cv2.resize(b, COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float64)
    else:
        a_r = np.asarray(Image.fromarray(a).resize(COMPARE_SIZE), dtype=np.float64)
        b_r = np.asarray(Image.fromarray(b).resize(COMPARE_SIZE), dtype=np.float64)
    a_r -= a_r.mean()
    b_r -= b_r.mean()
    denom = np.sqrt((a_r ** 2).sum() * (b_r ** 2).sum())
    if denom < 1e-9:
        return 1.0  # flat crops: treat as duplicates so they are skipped
    return float((a_r * b_r).sum() / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrections-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Marker run dir (per_image_predictions.csv gives the vendor per folder).")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--min-side", type=int, default=6, help="Skip crops smaller than this on any side.")
    parser.add_argument("--dedupe-similarity", type=float, default=0.90,
                        help="Skip a crop whose correlation with a kept/bank template exceeds this.")
    parser.add_argument("--max-per-folder", type=int, default=4,
                        help="Keep at most N distinct crops per folder (one per orientation is enough).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vendor_by_folder: Dict[str, str] = {}
    for r in csv.DictReader((args.run_dir / "per_image_predictions.csv").open()):
        vendor_by_folder.setdefault(str(r["folder"]), str(r["vendor"]))

    rows = list(csv.DictReader(args.corrections_csv.open(encoding="utf-8-sig")))
    box_rows = [r for r in rows if str(r.get("corr_top_pct", "")).strip()]
    print(f"[load] {len(rows)} correction rows, {len(box_rows)} with a redrawn box")

    # Load existing accepted bank crops per vendor for dedupe.
    decisions_path = args.library_root / "review_decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    vendors_node = decisions.setdefault("vendors", {})

    bank_gray: Dict[str, List[np.ndarray]] = defaultdict(list)

    def _vendor_dir(vendor: str) -> Optional[Path]:
        if not vendor:
            return None
        for p in args.library_root.iterdir():
            if p.is_dir() and p.name.lower() == vendor.lower():
                return p
        return None

    def _load_bank(vendor_dirname: str) -> None:
        if vendor_dirname in bank_gray:
            return
        node = None
        for key, value in vendors_node.items():
            if key.lower() == vendor_dirname.lower():
                node = value
                break
        accepted = [str(x) for x in (node or {}).get("accepted", [])]
        if not accepted:
            rejected = {Path(x).name for x in (node or {}).get("rejected", [])}
            vdir = args.library_root / vendor_dirname
            accepted = [f"{vendor_dirname}/{p.name}" for p in sorted(vdir.glob("marker_*.png")) if p.name not in rejected]
        for rel in accepted:
            p = args.library_root / rel
            if p.is_file():
                bank_gray[vendor_dirname].append(np.asarray(Image.open(p).convert("L"), dtype=np.uint8))

    harvested: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = defaultdict(list)  # vendor -> (gray, rgb, source)
    skipped = {"no_vendor": 0, "missing_image": 0, "too_small": 0, "duplicate": 0, "folder_cap": 0}
    per_folder_kept: Dict[str, int] = defaultdict(int)

    for r in box_rows:
        folder, image_id = str(r["folder"]), str(r["image_id"])
        vendor = vendor_by_folder.get(folder, "")
        vdir = _vendor_dir(vendor)
        if vdir is None:
            skipped["no_vendor"] += 1
            print(f"  [skip] no vendor dir for '{vendor}' ({folder[:50]})")
            continue
        img_path = args.dataset_root / folder / image_id
        if not img_path.is_file():
            skipped["missing_image"] += 1
            continue
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            W, H = rgb.size
            t = round(float(r["corr_top_pct"]) / 100 * H)
            l = round(float(r["corr_left_pct"]) / 100 * W)
            b = round(float(r["corr_bottom_pct"]) / 100 * H)
            rr = round(float(r["corr_right_pct"]) / 100 * W)
            t, l = max(0, t), max(0, l)
            b, rr = min(H - 1, b), min(W - 1, rr)
            if b - t < args.min_side or rr - l < args.min_side:
                skipped["too_small"] += 1
                continue
            crop_rgb = rgb.crop((l, t, rr + 1, b + 1))
        gray = np.asarray(crop_rgb.convert("L"), dtype=np.uint8)

        if per_folder_kept[folder] >= args.max_per_folder:
            skipped["folder_cap"] += 1
            continue
        _load_bank(vdir.name)
        dup = any(_similarity(gray, kept) >= args.dedupe_similarity for kept in bank_gray[vdir.name])
        if not dup:
            dup = any(_similarity(gray, g) >= args.dedupe_similarity for g, _rgb, _src in harvested[vdir.name])
        if dup:
            skipped["duplicate"] += 1
            continue
        harvested[vdir.name].append((gray, np.asarray(crop_rgb), f"{folder}/{image_id}"))
        per_folder_kept[folder] += 1

    print(f"\n[dedupe] skipped: {skipped}")
    total_new = sum(len(v) for v in harvested.values())
    print(f"[harvest] {total_new} new templates across {len(harvested)} vendors")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for vendor_dirname, items in sorted(harvested.items()):
        vdir = args.library_root / vendor_dirname
        existing = [int(m.group(1)) for p in vdir.glob("marker_*.png")
                    if (m := re.match(r"marker_(\d+)\.png$", p.name))]
        next_idx = max(existing, default=0) + 1
        node = None
        for key in vendors_node:
            if key.lower() == vendor_dirname.lower():
                node = vendors_node[key]
                break
        if node is None:
            node = vendors_node.setdefault(vendor_dirname, {"accepted": [], "rejected": []})
        accepted = node.setdefault("accepted", [])
        if not accepted:
            # Bank used implicit accepted-list (all non-rejected): materialize it
            # before appending, so the new entries do not hide the old ones.
            rejected = {Path(x).name for x in node.get("rejected", [])}
            accepted.extend(f"{vendor_dirname}/{p.name}" for p in sorted(vdir.glob("marker_*.png"))
                            if p.name not in rejected)
        for gray, rgb_arr, source in items:
            name = f"marker_{next_idx:03d}.png"
            rel = f"{vendor_dirname}/{name}"
            print(f"  + {rel}  {gray.shape[1]}x{gray.shape[0]}  from {source[:70]}")
            if not args.dry_run:
                Image.fromarray(rgb_arr).save(vdir / name)
                accepted.append(rel)
            next_idx += 1
        if not args.dry_run:
            decisions.setdefault("maintenance_log", []).append({
                "at": stamp,
                "action": "harvest_from_review_corrections",
                "vendor": vendor_dirname,
                "added": len(items),
                "source_csv": str(args.corrections_csv),
            })

    if not args.dry_run and total_new:
        decisions["updated_at"] = stamp
        decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[write] {decisions_path} updated")
    elif args.dry_run:
        print("\n[dry-run] nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
