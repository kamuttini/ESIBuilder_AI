#!/usr/bin/env python3
"""Build visual examples of a line21-driven bbox around scale + numbers.

For each selected manifest row:
- resolve the full-frame image
- map line21 GT (x1,x2,y1,y2) into image coordinates
- build a context bbox that includes scale axis, ticks, and nearby numbers
- save full-frame overlay + bbox crop
- generate an HTML page for quick review
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Row:
    sample_id: str
    split: str
    fss_path: str
    setup_id: str
    depth_index: int
    x1: float
    x2: float
    y1: float
    y2: float
    video_x_size: int
    video_y_size: int
    label_side: int


def _f(text: str) -> float:
    return float(str(text).strip())


def _i(text: str) -> int:
    return int(float(str(text).strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))[:140]


def _find_image(fss_path: str, setup_id: str, depth_index: int) -> Optional[Path]:
    fss = Path(fss_path)
    root = fss.parent.parent
    sid = setup_id.strip() or fss.stem.replace("setup_", "")
    image_samples = root / "image_samples"
    if not image_samples.exists():
        return None

    idx0 = max(0, depth_index - 1)
    stems = [
        f"image_depth_value_setup_{idx0}",
        f"image_depth_find_flip_ud_setup_{idx0}",
        f"image_depth_value_setup_{depth_index}",
        f"image_depth_find_flip_ud_setup_{depth_index}",
        f"image_depth_value_setup_{sid}_{idx0}",
        f"image_depth_value_setup_{sid}_{depth_index}",
        "image_orientation_setup_0",
    ]
    for stem in stems:
        for ext in IMG_EXTS:
            p = image_samples / f"{stem}{ext}"
            if p.exists() and not p.name.startswith("._"):
                return p

    for pat in ("image_*.png", "image_*.jpg", "image_*.jpeg", "image_*.bmp", "image_*.tif", "image_*.tiff"):
        for p in sorted(image_samples.glob(pat)):
            if not p.name.startswith("._"):
                return p
    return None


def _load_rows(manifest: Path, split: str, limit: int, sample_id_substr: str) -> List[Row]:
    out: List[Row] = []
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            sp = str(r.get("split", "")).strip().lower()
            if split and sp != split:
                continue
            sid = str(r.get("sample_id", "")).strip()
            if sample_id_substr and (sample_id_substr.lower() not in sid.lower()):
                continue
            try:
                out.append(
                    Row(
                        sample_id=sid,
                        split=sp,
                        fss_path=str(r["fss_path"]),
                        setup_id=str(r.get("setup_id", "")),
                        depth_index=_i(r["depth_index"]),
                        x1=_f(r["x1"]),
                        x2=_f(r["x2"]),
                        y1=_f(r["y1"]),
                        y2=_f(r["y2"]),
                        video_x_size=_i(r["video_x_size"]),
                        video_y_size=_i(r["video_y_size"]),
                        label_side=_i(r.get("label_side", 1)),
                    )
                )
            except Exception:
                continue
            if limit > 0 and len(out) >= limit:
                break
    return out


def _context_bbox(
    x_line: float,
    y_top: float,
    y_bottom: float,
    side: int,
    w: int,
    h: int,
    numbers_span_px: float,
    tick_span_px: float,
    line_half_px: float,
    pad_y_px: float,
    trim_left_px: float,
    shift_y_px: float,
) -> tuple[int, int, int, int]:
    y1 = (y_top - pad_y_px) + float(shift_y_px)
    y2 = (y_bottom + pad_y_px) + float(shift_y_px)
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))

    if side < 0:
        x1 = x_line - (numbers_span_px + tick_span_px + line_half_px)
        x2 = x_line + (line_half_px + 8.0)
    else:
        x1 = x_line - (line_half_px + 8.0)
        x2 = x_line + (numbers_span_px + tick_span_px + line_half_px)

    # Optional absolute trim on the left side of the bbox.
    if trim_left_px > 0:
        x1 += float(trim_left_px)

    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    if x2 <= x1 + 2:
        x2 = min(float(w - 1), x1 + 3.0)
    if y2 <= y1 + 2:
        y2 = min(float(h - 1), y1 + 3.0)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _draw_and_save(
    row: Row,
    image_path: Path,
    out_full: Path,
    out_crop: Path,
    numbers_span_px: float,
    tick_span_px: float,
    line_half_px: float,
    pad_y_px: float,
    trim_left_px: float,
    shift_y_px: float,
) -> dict:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    sx = float(w) / max(1.0, float(row.video_x_size))
    sy = float(h) / max(1.0, float(row.video_y_size))

    x_line = 0.5 * (row.x1 + row.x2) * sx
    y_top = min(row.y1, row.y2) * sy
    y_bottom = max(row.y1, row.y2) * sy

    bx1, by1, bx2, by2 = _context_bbox(
        x_line=x_line,
        y_top=y_top,
        y_bottom=y_bottom,
        side=-1 if row.label_side < 0 else 1,
        w=w,
        h=h,
        numbers_span_px=numbers_span_px,
        tick_span_px=tick_span_px,
        line_half_px=line_half_px,
        pad_y_px=pad_y_px,
        trim_left_px=trim_left_px,
        shift_y_px=shift_y_px,
    )

    draw = ImageDraw.Draw(img, "RGBA")
    draw.line((x_line, y_top, x_line, y_bottom), fill=(0, 255, 120, 255), width=5)
    draw.rectangle((bx1, by1, bx2, by2), outline=(255, 210, 0, 255), width=4, fill=(255, 210, 0, 42))
    legend = f"line21 bbox: sample={row.sample_id} side={'L' if row.label_side < 0 else 'R'}"
    draw.rectangle((10, 10, min(w - 10, 10 + 8 * len(legend) + 18), 34), fill=(0, 0, 0, 180))
    draw.text((16, 16), legend, fill=(255, 255, 255, 255))

    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_crop.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_full, quality=92)

    crop = Image.open(image_path).convert("RGB").crop((bx1, by1, bx2, by2))
    crop.save(out_crop, quality=95)

    return {
        "sample_id": row.sample_id,
        "split": row.split,
        "source_image": image_path.as_posix(),
        "preview_full": out_full.as_posix(),
        "preview_crop": out_crop.as_posix(),
        "x_line_image": float(x_line),
        "y_top_image": float(y_top),
        "y_bottom_image": float(y_bottom),
        "bbox_x1": int(bx1),
        "bbox_y1": int(by1),
        "bbox_x2": int(bx2),
        "bbox_y2": int(by2),
        "label_side": int(row.label_side),
        "video_x_size": int(row.video_x_size),
        "video_y_size": int(row.video_y_size),
    }


def _build_html(items: Sequence[dict], out_html: Path) -> None:
    rows: List[str] = []
    for i, it in enumerate(items, start=1):
        full_rel = Path(it["preview_full"]).name
        crop_rel = Path(it["preview_crop"]).name
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{it['sample_id']}</td>"
            f"<td>{it['split']}</td>"
            f"<td>{'L' if int(it['label_side']) < 0 else 'R'}</td>"
            f"<td><a href='previews/{full_rel}' target='_blank'><img src='previews/{full_rel}' loading='lazy' /></a></td>"
            f"<td><a href='crops/{crop_rel}' target='_blank'><img src='crops/{crop_rel}' loading='lazy' /></a></td>"
            f"<td>{it['bbox_x1']},{it['bbox_y1']},{it['bbox_x2']},{it['bbox_y2']}</td>"
            "</tr>"
        )

    out_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>line21 bbox examples</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 420px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
    .note {{ margin-bottom: 12px; padding: 10px 12px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; }}
  </style>
</head>
<body>
  <h1>line21 -> bbox (scala + numeri) examples</h1>
  <div class="note">
    verde = asse scala line21; giallo = bbox candidata per training rettangolo.
  </div>
  <table>
    <thead>
      <tr><th>#</th><th>sample_id</th><th>split</th><th>label side</th><th>full overlay</th><th>crop bbox</th><th>bbox (x1,y1,x2,y2)</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build line21 bbox examples for rectangle training.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", ""])
    p.add_argument("--max-examples", type=int, default=8)
    p.add_argument("--sample-id-substr", type=str, default="")
    p.add_argument("--numbers-span-px", type=float, default=138.0)
    p.add_argument("--tick-span-px", type=float, default=30.0)
    p.add_argument("--line-half-px", type=float, default=10.0)
    p.add_argument("--pad-y-px", type=float, default=16.0)
    p.add_argument("--trim-left-px", type=float, default=0.0)
    p.add_argument("--shift-y-px", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    preview_dir = out_dir / "previews"
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(
        manifest=manifest,
        split=args.split.strip().lower(),
        limit=max(0, int(args.max_examples)),
        sample_id_substr=args.sample_id_substr.strip(),
    )
    if not rows:
        raise RuntimeError("No rows selected from manifest with current filters.")

    items: List[dict] = []
    for i, row in enumerate(rows, start=1):
        image_path = _find_image(row.fss_path, row.setup_id, row.depth_index)
        if image_path is None:
            continue
        slug = _safe_slug(f"{i:03d}_{row.sample_id}")
        out_full = preview_dir / f"{slug}.jpg"
        out_crop = crop_dir / f"{slug}_crop.jpg"
        item = _draw_and_save(
            row=row,
            image_path=image_path,
            out_full=out_full,
            out_crop=out_crop,
            numbers_span_px=float(args.numbers_span_px),
            tick_span_px=float(args.tick_span_px),
            line_half_px=float(args.line_half_px),
            pad_y_px=float(args.pad_y_px),
            trim_left_px=float(args.trim_left_px),
            shift_y_px=float(args.shift_y_px),
        )
        items.append(item)

    if not items:
        raise RuntimeError("Rows selected but no images could be resolved.")

    out_html = out_dir / "line21_bbox_examples.html"
    _build_html(items, out_html)
    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": out_dir.as_posix(),
        "rows_requested": int(args.max_examples),
        "rows_rendered": len(items),
        "split": args.split,
        "params": {
            "numbers_span_px": args.numbers_span_px,
            "tick_span_px": args.tick_span_px,
            "line_half_px": args.line_half_px,
            "pad_y_px": args.pad_y_px,
            "trim_left_px": args.trim_left_px,
            "shift_y_px": args.shift_y_px,
        },
        "artifacts": {
            "html": out_html.as_posix(),
            "preview_dir": preview_dir.as_posix(),
            "crop_dir": crop_dir.as_posix(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
