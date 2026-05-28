#!/usr/bin/env python3
"""Build a single-image-per-folder rect review package for vendor-template data.

This script converts `manifest_after_exclusions.csv` into a synthetic predictions CSV
containing exactly one image per folder (`group_id`) for folders marked
`manual_folder_action=modify_rect`, then reuses the proven
`build_ultrasound_rect_review_package.py` UI (autosave + export corrections).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(round(float((value or "").strip())))
    except Exception:  # noqa: BLE001
        return int(default)


def _pick_one(rows: Sequence[Dict[str, str]], mode: str, rng: random.Random) -> Dict[str, str]:
    ordered = sorted(rows, key=lambda r: (r.get("image_path", "") or ""))
    if not ordered:
        raise RuntimeError("Cannot pick from empty row list")
    if mode == "first":
        return ordered[0]
    if mode == "last":
        return ordered[-1]
    if mode == "random":
        return ordered[rng.randrange(len(ordered))]
    # default: middle
    return ordered[len(ordered) // 2]


def _resolve_bbox(row: Dict[str, str]) -> Dict[str, int]:
    top = _parse_int(row.get("bbox_top", row.get("bbox_ymin", "0")), 0)
    left = _parse_int(row.get("bbox_left", row.get("bbox_xmin", "0")), 0)
    bottom = _parse_int(row.get("bbox_bottom", row.get("bbox_ymax", "1")), 1)
    right = _parse_int(row.get("bbox_right", row.get("bbox_xmax", "1")), 1)
    if bottom <= top:
        bottom = top + 1
    if right <= left:
        right = left + 1
    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
    }


def _resolve_image_size(row: Dict[str, str], image_path: Path) -> Dict[str, int]:
    width = _parse_int(row.get("image_width", ""), 0)
    height = _parse_int(row.get("image_height", ""), 0)
    if width > 0 and height > 0:
        return {"width": width, "height": height}

    try:
        from PIL import Image  # imported lazily

        with Image.open(image_path) as im:
            w, h = im.size
            return {"width": int(w), "height": int(h)}
    except Exception:  # noqa: BLE001
        return {"width": max(1, width), "height": max(1, height)}


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def _write_predictions_csv(
    rows: Sequence[Dict[str, str]],
    output_csv: Path,
    selection: str,
    seed: int,
) -> Dict[str, object]:
    by_group: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        action = (row.get("manual_folder_action", "") or "").strip()
        if action != "modify_rect":
            continue
        gid = (row.get("group_id", "") or "").strip()
        image_path = (row.get("image_path", "") or "").strip()
        if not gid or not image_path:
            continue
        by_group[gid].append(row)

    if not by_group:
        raise RuntimeError("No rows found with manual_folder_action=modify_rect")

    fields = [
        "sample_id",
        "image_path",
        "rel_path",
        "group_name",
        "orientation_name",
        "vendor_predicted",
        "vendor_top1_prob",
        "vendor_margin_top1_top2",
        "vendor_vote_ratio",
        "width",
        "height",
        "pred_top",
        "pred_left",
        "pred_bottom",
        "pred_right",
        "global_top",
        "global_left",
        "global_bottom",
        "global_right",
    ]

    rng = random.Random(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    group_sizes: Dict[str, int] = {}
    vendor_counter: Counter[str] = Counter()
    written = 0
    missing_images = 0

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for gid in sorted(by_group.keys()):
            group_rows = by_group[gid]
            group_sizes[gid] = len(group_rows)
            chosen = _pick_one(group_rows, mode=selection, rng=rng)

            image_path_str = (chosen.get("image_path", "") or "").strip()
            image_path = Path(image_path_str).expanduser().resolve()
            if not image_path.exists():
                missing_images += 1
                continue

            bbox = _resolve_bbox(chosen)
            size = _resolve_image_size(chosen, image_path=image_path)
            vendor = (chosen.get("manufacturer", "") or "").strip()
            setup = (chosen.get("setup_name", "") or "").strip() or "setup"
            name = image_path.name
            sample_id = f"{gid}::{name}"

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "image_path": image_path.as_posix(),
                    "rel_path": f"{gid}/{name}",
                    "group_name": gid,
                    "orientation_name": setup,
                    "vendor_predicted": vendor,
                    "vendor_top1_prob": "1.0",
                    "vendor_margin_top1_top2": "1.0",
                    "vendor_vote_ratio": "1.0",
                    "width": size["width"],
                    "height": size["height"],
                    "pred_top": bbox["top"],
                    "pred_left": bbox["left"],
                    "pred_bottom": bbox["bottom"],
                    "pred_right": bbox["right"],
                    "global_top": bbox["top"],
                    "global_left": bbox["left"],
                    "global_bottom": bbox["bottom"],
                    "global_right": bbox["right"],
                }
            )
            vendor_counter[vendor] += 1
            written += 1

    if written == 0:
        raise RuntimeError("No valid rows written to predictions CSV")

    return {
        "groups_total": len(by_group),
        "groups_written": written,
        "missing_images": missing_images,
        "group_sizes": group_sizes,
        "groups_per_vendor": dict(sorted(vendor_counter.items())),
        "selection_mode": selection,
        "seed": seed,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a single-image-per-folder rect review package for vendor-template "
            "folders marked as modify_rect."
        )
    )
    p.add_argument(
        "--manifest-after-exclusions",
        type=Path,
        default=Path(
            "artifacts/40_outputs_eval/vendor_template_retraining_from_review_real/manifest_after_exclusions.csv"
        ),
        help="Input stage-1 manifest produced by prepare_vendor_template_retraining_from_review.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_template_rect_single_image_review"),
    )
    p.add_argument(
        "--selection",
        type=str,
        choices=("first", "middle", "last", "random"),
        default="middle",
        help="Which image to choose as representative for each folder.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (used when selection=random).")
    p.add_argument(
        "--package-name",
        type=str,
        default="Vendor Template Rect Review (1 immagine per cartella)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    manifest = args.manifest_after_exclusions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest-after-exclusions: {manifest}")

    rows = _load_manifest(manifest)
    pred_csv = output_dir / "predictions_single_image_per_folder.csv"
    stats = _write_predictions_csv(
        rows=rows,
        output_csv=pred_csv,
        selection=str(args.selection),
        seed=int(args.seed),
    )

    repo_root = Path(__file__).resolve().parents[2]
    builder_script = repo_root / "tools" / "review_html" / "build_ultrasound_rect_review_package.py"
    html_out = output_dir / "html"

    cmd = [
        sys.executable,
        str(builder_script),
        "--predictions-csv",
        str(pred_csv),
        "--output-dir",
        str(html_out),
        "--package-name",
        str(args.package_name),
    ]
    subprocess.run(cmd, check=True)

    summary = {
        "manifest_after_exclusions": manifest.as_posix(),
        "predictions_csv": pred_csv.as_posix(),
        "html_output_dir": html_out.as_posix(),
        "index_html": (html_out / "index.html").as_posix(),
        "preview_html": (html_out / "preview_by_vendor_confidence.html").as_posix(),
        "stats": stats,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Predictions CSV: {pred_csv}", flush=True)
    print(f"Review package index: {html_out / 'index.html'}", flush=True)
    print(f"Groups written: {stats['groups_written']} / {stats['groups_total']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

