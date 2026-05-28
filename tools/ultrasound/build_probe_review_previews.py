#!/usr/bin/env python3
"""Build visual previews for probe review folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_review_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_per_image_rows(path: Path) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            grouped[row["folder_path"]].append(row)
    return grouped


def _iter_folder_images(folder_path: Path) -> List[Path]:
    image_dir = folder_path / "image_samples"
    if not image_dir.is_dir():
        return []
    images = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "negative" in path.name.lower():
            continue
        images.append(path)
    return images


def _select_image_rows(rows: Sequence[Dict[str, str]], max_tiles: int) -> List[Dict[str, str]]:
    if not rows:
        return []
    mismatches = sorted(
        [r for r in rows if str(r.get("matches_folder_probe", "")).strip() in {"0", "False", "false"}],
        key=lambda r: _safe_float(r.get("image_pred_confidence"), 0.0),
        reverse=True,
    )
    matches = sorted(
        [r for r in rows if str(r.get("matches_folder_probe", "")).strip() in {"1", "True", "true"}],
        key=lambda r: _safe_float(r.get("image_pred_confidence"), 0.0),
        reverse=True,
    )

    chosen: List[Dict[str, str]] = []
    seen = set()
    mismatch_target = max(1, max_tiles // 2)

    for bucket, target in ((mismatches, mismatch_target), (matches, max_tiles)):
        for row in bucket:
            key = row.get("image_path", "")
            if not key or key in seen:
                continue
            chosen.append(row)
            seen.add(key)
            if len(chosen) >= target:
                break
        if len(chosen) >= max_tiles:
            return chosen[:max_tiles]

    for row in mismatches + matches:
        key = row.get("image_path", "")
        if not key or key in seen:
            continue
        chosen.append(row)
        seen.add(key)
        if len(chosen) >= max_tiles:
            break
    return chosen[:max_tiles]


def _wrap_lines(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def _load_tile_image(path: Path, tile_w: int, tile_h: int) -> Image.Image:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return ImageOps.fit(
                rgb,
                (tile_w, tile_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
    except Exception:
        fallback = Image.new("RGB", (tile_w, tile_h), (25, 25, 25))
        draw = ImageDraw.Draw(fallback)
        draw.rectangle((0, 0, tile_w - 1, tile_h - 1), outline=(240, 70, 70), width=3)
        draw.text((14, tile_h // 2 - 8), "image load error", fill=(240, 70, 70), font=ImageFont.load_default())
        return fallback


def _build_single_preview(
    idx: int,
    row: Dict[str, str],
    selected_rows: Sequence[Dict[str, str]],
    fallback_paths: Sequence[Path],
    output_path: Path,
    max_tiles: int,
) -> int:
    tile_w = 460
    tile_h = 290
    cols = 3
    rows = max(1, math.ceil(max_tiles / cols))
    gap = 10
    margin = 20
    header_h = 210

    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = margin * 2 + header_h + rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (18, 23, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = f"Probe review #{idx:02d}"
    draw.text((margin, margin), title, fill=(255, 225, 120), font=font)

    info_lines = [
        f"Folder: {row.get('folder_name', '')}",
        f"Pred probe: {row.get('predicted_probe_id', '?')} | top1={_safe_float(row.get('folder_top1_prob')):.4f} "
        f"| margin={_safe_float(row.get('folder_margin_top1_top2')):.4f} "
        f"| vote={_safe_float(row.get('folder_vote_ratio')):.4f}",
        f"Reasons: {row.get('review_reasons', '')}",
        f"TopK: {row.get('topk', '')}",
    ]
    yy = margin + 24
    for text_line in info_lines:
        for line in _wrap_lines(text_line, 135):
            draw.text((margin, yy), line, fill=(235, 235, 235), font=font)
            yy += 15
        yy += 2

    selected: List[Dict[str, str]] = list(selected_rows)
    if len(selected) < max_tiles:
        selected_paths = {Path(x.get("image_path", "")).as_posix() for x in selected if x.get("image_path")}
        for path in fallback_paths:
            if path.as_posix() in selected_paths:
                continue
            selected.append(
                {
                    "image_path": path.as_posix(),
                    "image_pred_probe_id": "",
                    "image_pred_confidence": "",
                    "matches_folder_probe": "",
                }
            )
            if len(selected) >= max_tiles:
                break

    selected = selected[:max_tiles]

    for tile_idx, img_row in enumerate(selected):
        col = tile_idx % cols
        row_idx = tile_idx // cols
        x0 = margin + col * (tile_w + gap)
        y0 = margin + header_h + row_idx * (tile_h + gap)

        image_path = Path(img_row.get("image_path", ""))
        tile_img = _load_tile_image(image_path, tile_w=tile_w, tile_h=tile_h)
        canvas.paste(tile_img, (x0, y0))

        pred = (img_row.get("image_pred_probe_id") or "?").strip()
        conf = _safe_float(img_row.get("image_pred_confidence"), 0.0)
        match_flag = img_row.get("matches_folder_probe", "")
        if str(match_flag).strip() in {"0", "False", "false"}:
            match_txt = "mismatch"
            color = (255, 95, 95)
        elif str(match_flag).strip() in {"1", "True", "true"}:
            match_txt = "match"
            color = (120, 255, 140)
        else:
            match_txt = "n/a"
            color = (220, 220, 220)

        label_h = 54
        draw.rectangle((x0, y0 + tile_h - label_h, x0 + tile_w, y0 + tile_h), fill=(0, 0, 0, 180))
        draw.text((x0 + 8, y0 + tile_h - 46), image_path.name[:62], fill=(245, 245, 245), font=font)
        draw.text(
            (x0 + 8, y0 + tile_h - 28),
            f"img_pred={pred} conf={conf:.3f} {match_txt}",
            fill=color,
            font=font,
        )

    canvas.save(output_path)
    return len(selected)


def _build_contact_sheet(preview_paths: Sequence[Path], output_path: Path) -> None:
    if not preview_paths:
        return
    thumb_w = 500
    thumb_h = 320
    cols = 3
    rows = math.ceil(len(preview_paths) / cols)
    gap = 12
    margin = 18
    title_h = 48
    width = margin * 2 + cols * thumb_w + (cols - 1) * gap
    height = margin * 2 + title_h + rows * thumb_h + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), "Probe review contact sheet", fill=(245, 245, 245), font=font)

    for idx, preview_path in enumerate(preview_paths):
        col = idx % cols
        row = idx // cols
        x0 = margin + col * (thumb_w + gap)
        y0 = margin + title_h + row * (thumb_h + gap)
        with Image.open(preview_path) as img:
            thumb = ImageOps.fit(
                img.convert("RGB"),
                (thumb_w, thumb_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        canvas.paste(thumb, (x0, y0))
        draw.rectangle((x0, y0, x0 + 72, y0 + 24), fill=(0, 0, 0))
        draw.text((x0 + 7, y0 + 6), f"#{idx + 1:02d}", fill=(255, 225, 120), font=font)

    canvas.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build visual previews for probe review folders."
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions_review.csv"),
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/review_previews"),
    )
    parser.add_argument("--max-tiles", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tiles <= 0:
        raise ValueError("--max-tiles must be > 0")

    review_csv = args.review_csv.expanduser().resolve()
    per_image_csv = args.per_image_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows = _read_review_rows(review_csv)
    if not review_rows:
        raise RuntimeError("No review rows found.")
    per_image_rows = _read_per_image_rows(per_image_csv)

    manifest_rows: List[Dict[str, object]] = []
    preview_paths: List[Path] = []
    for idx, row in enumerate(review_rows, start=1):
        folder_path = Path(row["folder_path"])
        folder_image_rows = per_image_rows.get(folder_path.as_posix(), [])
        selected_rows = _select_image_rows(folder_image_rows, max_tiles=args.max_tiles)
        fallback_paths = _iter_folder_images(folder_path)

        out_path = output_dir / f"review_{idx:02d}.png"
        tiles_used = _build_single_preview(
            idx=idx,
            row=row,
            selected_rows=selected_rows,
            fallback_paths=fallback_paths,
            output_path=out_path,
            max_tiles=args.max_tiles,
        )
        preview_paths.append(out_path)
        manifest_rows.append(
            {
                "idx": idx,
                "folder_name": row.get("folder_name", ""),
                "folder_path": row.get("folder_path", ""),
                "preview_path": out_path.as_posix(),
                "predicted_probe_id": row.get("predicted_probe_id", ""),
                "folder_top1_prob": row.get("folder_top1_prob", ""),
                "folder_margin_top1_top2": row.get("folder_margin_top1_top2", ""),
                "folder_vote_ratio": row.get("folder_vote_ratio", ""),
                "review_reasons": row.get("review_reasons", ""),
                "topk": row.get("topk", ""),
                "tiles_used": tiles_used,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "idx",
                "folder_name",
                "folder_path",
                "preview_path",
                "predicted_probe_id",
                "folder_top1_prob",
                "folder_margin_top1_top2",
                "folder_vote_ratio",
                "review_reasons",
                "topk",
                "tiles_used",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    contact_sheet = output_dir / "review_contact_sheet.png"
    _build_contact_sheet(preview_paths=preview_paths, output_path=contact_sheet)

    summary = {
        "review_rows": len(review_rows),
        "preview_dir": output_dir.as_posix(),
        "manifest": manifest_path.as_posix(),
        "contact_sheet": contact_sheet.as_posix(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Review rows: {len(review_rows)}")
    print(f"Preview dir: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheet: {contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
