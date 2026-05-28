#!/usr/bin/env python3
"""Build a full manual-review package for symbol detection labels.

Input:
- symbol_detection_manifest.csv (from prepare_orientation_symbol_detection_dataset.py)

Output:
- review_queue.csv (ordered rows + empty manual decision columns)
- overlays/*.jpg (one overlay per sample)
- pages/page_XXXX.html (paginated review pages)
- index.html (links to pages + quick stats)
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ReviewRow:
    review_order: int
    sample_id: str
    image_path: Path
    image_name: str
    overlay_relpath: str
    split: str
    manufacturer: str
    model_name: str
    dataset_folder: str
    source_type: str
    orientation_idx: str
    orientation_name: str
    label_kind: str
    auto_accept: str
    match_score: str
    coarse_rect: Optional[Tuple[int, int, int, int]]  # top,left,bottom,right
    symbol_rect: Optional[Tuple[int, int, int, int]]  # top,left,bottom,right


def _parse_int(value: str) -> Optional[int]:
    cleaned = (value or "").strip()
    if cleaned == "":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_float(value: str) -> Optional[float]:
    cleaned = (value or "").strip()
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_slug(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "sample"


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _sort_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    def key_fn(row: Dict[str, str]):
        auto_accept = 1 if (row.get("auto_accept", "") or "").strip() == "1" else 0
        score = _parse_float(row.get("match_score", "")) or -999.0
        # First review-needed rows, then low scores, then orientation/vendor for stable navigation.
        return (
            auto_accept,  # 0 first (review)
            score,
            row.get("orientation_name", ""),
            row.get("manufacturer", ""),
            row.get("dataset_folder", ""),
            row.get("sample_id", ""),
        )

    return sorted(rows, key=key_fn)


def _build_overlay(
    image_path: Path,
    coarse_rect: Optional[Tuple[int, int, int, int]],
    symbol_rect: Optional[Tuple[int, int, int, int]],
    out_path: Path,
    title: str,
    subtitle: str,
    max_width: int,
    jpeg_quality: int,
) -> bool:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(rgb)
    font = ImageFont.load_default()

    if coarse_rect is not None:
        t, l, b, r = coarse_rect
        draw.rectangle((l, t, r, b), outline=(255, 220, 0), width=3)
    if symbol_rect is not None:
        t, l, b, r = symbol_rect
        draw.rectangle((l, t, r, b), outline=(255, 64, 64), width=3)

    pad = 8
    text_h = 34
    draw.rectangle((0, 0, rgb.width, text_h), fill=(0, 0, 0))
    draw.text((pad, 4), title, fill=(255, 255, 255), font=font)
    draw.text((pad, 18), subtitle[:180], fill=(200, 200, 200), font=font)

    if max_width > 0 and rgb.width > max_width:
        scale = max_width / float(rgb.width)
        new_w = int(round(rgb.width * scale))
        new_h = int(round(rgb.height * scale))
        rgb = rgb.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, format="JPEG", quality=max(30, min(95, jpeg_quality)))
    return True


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_page(
    page_path: Path,
    page_num: int,
    page_count: int,
    rows: Sequence[ReviewRow],
    rel_prefix: str,
) -> None:
    cards: List[str] = []
    for row in rows:
        score = row.match_score or ""
        cls = "needs-review" if row.auto_accept != "1" else "auto-accept"
        cards.append(
            f"""
<article class="card {cls}">
  <img loading="lazy" src="{_html_escape(rel_prefix + row.overlay_relpath)}" alt="{_html_escape(row.sample_id)}">
  <div class="meta">
    <div><b>{_html_escape(row.sample_id)}</b></div>
    <div>kind={_html_escape(row.label_kind)} | auto={_html_escape(row.auto_accept)} | score={_html_escape(score)}</div>
    <div>{_html_escape(row.orientation_name)} | {_html_escape(row.manufacturer)} | {_html_escape(row.split)}</div>
    <div>{_html_escape(row.dataset_folder)}</div>
  </div>
</article>
""".strip()
        )

    nav_links = []
    if page_num > 1:
        nav_links.append(f'<a href="page_{page_num - 1:04d}.html">Prev</a>')
    nav_links.append(f"<span>Page {page_num}/{page_count}</span>")
    if page_num < page_count:
        nav_links.append(f'<a href="page_{page_num + 1:04d}.html">Next</a>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Symbol Review Page {page_num}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1218; color: #e6e6e6; }}
    a {{ color: #8ec5ff; }}
    .nav {{ display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; }}
    .card {{ background: #1a1f29; border: 1px solid #2d3646; border-radius: 6px; overflow: hidden; }}
    .card.needs-review {{ border-color: #b06666; }}
    .card.auto-accept {{ border-color: #5f8a64; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px; font-size: 12px; line-height: 1.35; }}
  </style>
</head>
<body>
  <div class="nav">
    {' '.join(nav_links)}
    <a href="../index.html">Index</a>
  </div>
  <div class="grid">
    {' '.join(cards)}
  </div>
  <div class="nav" style="margin-top:14px;">
    {' '.join(nav_links)}
    <a href="../index.html">Index</a>
  </div>
</body>
</html>
"""
    page_path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build full manual review package for symbol detection.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset_v2/symbol_detection_manifest.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset_v2/review_package"),
    )
    p.add_argument("--page-size", type=int, default=120)
    p.add_argument("--max-width", type=int, default=1100, help="Resize overlay preview max width (0 to keep original).")
    p.add_argument("--jpeg-quality", type=int, default=82)
    p.add_argument("--max-rows", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = output_dir / "overlays"
    pages_dir = output_dir / "pages"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _sort_rows(_read_manifest(manifest_path))
    if args.max_rows > 0:
        raw_rows = raw_rows[: args.max_rows]
    if not raw_rows:
        raise RuntimeError("No rows found in manifest.")

    review_rows: List[ReviewRow] = []
    missing_images = 0
    rendered = 0
    for idx, row in enumerate(raw_rows, start=1):
        image_path = Path((row.get("image_path", "") or "")).expanduser().resolve()
        if not image_path.exists():
            missing_images += 1
            continue

        coarse = (
            _parse_int(row.get("coarse_rect_top", "")),
            _parse_int(row.get("coarse_rect_left", "")),
            _parse_int(row.get("coarse_rect_bottom", "")),
            _parse_int(row.get("coarse_rect_right", "")),
        )
        coarse_rect = None
        if None not in coarse:
            coarse_rect = (int(coarse[0]), int(coarse[1]), int(coarse[2]), int(coarse[3]))  # type: ignore[arg-type]

        symbol = (
            _parse_int(row.get("symbol_rect_top", "")),
            _parse_int(row.get("symbol_rect_left", "")),
            _parse_int(row.get("symbol_rect_bottom", "")),
            _parse_int(row.get("symbol_rect_right", "")),
        )
        symbol_rect = None
        if None not in symbol:
            symbol_rect = (int(symbol[0]), int(symbol[1]), int(symbol[2]), int(symbol[3]))  # type: ignore[arg-type]

        sample_id = (row.get("sample_id", "") or f"sample_{idx:08d}").strip()
        file_base = f"{idx:06d}_{_safe_slug(sample_id)}"
        overlay_name = f"{file_base}.jpg"
        overlay_path = overlays_dir / overlay_name

        title = f"{sample_id} | {row.get('label_kind', '')} | auto={row.get('auto_accept', '')}"
        subtitle = (
            f"score={row.get('match_score', '')} | ori={row.get('orientation_name', '')} | "
            f"vendor={row.get('manufacturer', '')} | split={row.get('split', '')}"
        )
        ok = _build_overlay(
            image_path=image_path,
            coarse_rect=coarse_rect,
            symbol_rect=symbol_rect,
            out_path=overlay_path,
            title=title,
            subtitle=subtitle,
            max_width=max(0, int(args.max_width)),
            jpeg_quality=int(args.jpeg_quality),
        )
        if not ok:
            missing_images += 1
            continue
        rendered += 1

        review_rows.append(
            ReviewRow(
                review_order=idx,
                sample_id=sample_id,
                image_path=image_path,
                image_name=(row.get("image_name", "") or image_path.name),
                overlay_relpath=f"overlays/{overlay_name}",
                split=(row.get("split", "") or ""),
                manufacturer=(row.get("manufacturer", "") or ""),
                model_name=(row.get("model_name", "") or ""),
                dataset_folder=(row.get("dataset_folder", "") or ""),
                source_type=(row.get("source_type", "") or ""),
                orientation_idx=(row.get("orientation_idx", "") or ""),
                orientation_name=(row.get("orientation_name", "") or ""),
                label_kind=(row.get("label_kind", "") or ""),
                auto_accept=(row.get("auto_accept", "") or ""),
                match_score=(row.get("match_score", "") or ""),
                coarse_rect=coarse_rect,
                symbol_rect=symbol_rect,
            )
        )

        if idx % 500 == 0:
            print(f"rendered overlays: {idx}", flush=True)

    if not review_rows:
        raise RuntimeError("No valid rows rendered.")

    queue_csv = output_dir / "review_queue.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "review_order",
            "sample_id",
            "overlay_path",
            "image_path",
            "image_name",
            "split",
            "manufacturer",
            "model_name",
            "dataset_folder",
            "source_type",
            "orientation_idx",
            "orientation_name",
            "label_kind",
            "auto_accept",
            "match_score",
            "review_decision",
            "review_notes",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "review_order": row.review_order,
                    "sample_id": row.sample_id,
                    "overlay_path": (output_dir / row.overlay_relpath).as_posix(),
                    "image_path": row.image_path.as_posix(),
                    "image_name": row.image_name,
                    "split": row.split,
                    "manufacturer": row.manufacturer,
                    "model_name": row.model_name,
                    "dataset_folder": row.dataset_folder,
                    "source_type": row.source_type,
                    "orientation_idx": row.orientation_idx,
                    "orientation_name": row.orientation_name,
                    "label_kind": row.label_kind,
                    "auto_accept": row.auto_accept,
                    "match_score": row.match_score,
                    "review_decision": "",
                    "review_notes": "",
                }
            )

    page_size = max(1, int(args.page_size))
    page_count = int(math.ceil(len(review_rows) / float(page_size)))
    page_links: List[str] = []
    for page_num in range(1, page_count + 1):
        start = (page_num - 1) * page_size
        end = min(len(review_rows), start + page_size)
        page_rows = review_rows[start:end]
        page_name = f"page_{page_num:04d}.html"
        _write_page(
            page_path=pages_dir / page_name,
            page_num=page_num,
            page_count=page_count,
            rows=page_rows,
            rel_prefix="../",
        )
        page_links.append(
            f'<li><a href="pages/{page_name}">Page {page_num}</a> '
            f'({start + 1}-{end})</li>'
        )

    by_kind: Dict[str, int] = {}
    by_vendor: Dict[str, int] = {}
    for row in review_rows:
        by_kind[row.label_kind] = by_kind.get(row.label_kind, 0) + 1
        by_vendor[row.manufacturer] = by_vendor.get(row.manufacturer, 0) + 1

    index_html = output_dir / "index.html"
    index_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Symbol Detection Review Package</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #0f1218; color: #e6e6e6; }}
    a {{ color: #8ec5ff; }}
    code {{ color: #ffd479; }}
  </style>
</head>
<body>
  <h2>Symbol Detection Review Package</h2>
  <p>Total rows: <b>{len(review_rows)}</b></p>
  <p>Review queue CSV: <code>{queue_csv.as_posix()}</code></p>
  <p>Overlay folder: <code>{overlays_dir.as_posix()}</code></p>
  <h3>Counts by label_kind</h3>
  <pre>{json.dumps(dict(sorted(by_kind.items())), indent=2, ensure_ascii=False)}</pre>
  <h3>Counts by vendor</h3>
  <pre>{json.dumps(dict(sorted(by_vendor.items())), indent=2, ensure_ascii=False)}</pre>
  <h3>Pages</h3>
  <ul>
    {" ".join(page_links)}
  </ul>
</body>
</html>
""",
        encoding="utf-8",
    )

    summary = {
        "manifest_path": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "rows_in_manifest_after_filter": len(raw_rows),
        "rows_rendered": len(review_rows),
        "overlays_rendered": rendered,
        "missing_or_failed_images": missing_images,
        "page_size": page_size,
        "page_count": page_count,
        "queue_csv": queue_csv.as_posix(),
        "index_html": index_html.as_posix(),
        "counts_by_label_kind": dict(sorted(by_kind.items())),
        "counts_by_vendor": dict(sorted(by_vendor.items())),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Index HTML: {index_html}", flush=True)
    print(f"Review queue CSV: {queue_csv}", flush=True)
    print(f"Overlays dir: {overlays_dir}", flush=True)
    print(f"Rows rendered: {len(review_rows)} | Missing/failed: {missing_images}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

