#!/usr/bin/env python3
"""Build HTML visual previews for SCALE_LINE GT anomaly rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


def _esc(v: object) -> str:
    return html.escape(str(v))


def _f(text: str) -> float:
    return float(str(text).strip())


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")[:120]


def _read_video_size_from_fss(fss_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None, None
    if len(lines) < 10:
        return None, None
    try:
        vx = int(lines[8].strip().replace("\r", ""))
        vy = int(lines[9].strip().replace("\r", ""))
        return vx, vy
    except Exception:
        return None, None


def _find_preview_image(fss_path: Path, setup_id: str, depth_index: int) -> Optional[Path]:
    root = fss_path.parent.parent
    sid = setup_id.strip() or fss_path.stem.replace("setup_", "")
    image_samples = root / "image_samples"

    # We want full frames, not tiny templates from DB_echo.
    if image_samples.exists():
        idx0 = max(0, depth_index - 1)
        stems = [
            f"image_depth_value_setup_{idx0}",
            f"image_depth_find_flip_ud_setup_{idx0}",
            f"image_depth_value_setup_{depth_index}",
            f"image_depth_find_flip_ud_setup_{depth_index}",
            "image_orientation_setup_0",
            "image_th_echo_negative_0",
            "image_th_probe_negative_0",
        ]
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        for stem in stems:
            for ext in exts:
                c = image_samples / f"{stem}{ext}"
                if c.exists():
                    return c

        # Generic fallback: any non-hidden image in image_samples.
        for ext in ("image_*.png", "image_*.jpg", "image_*.jpeg", "image_*.bmp", "image_*.tif", "image_*.tiff"):
            found = [p for p in sorted(image_samples.glob(ext)) if not p.name.startswith("._")]
            if found:
                return found[0]

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            found = [
                p
                for p in sorted(image_samples.glob(ext))
                if not p.name.startswith("._") and p.name.lower().startswith("image_")
            ]
            if found:
                return found[0]

    # No robust full-frame found in image_samples for this setup.
    return None


def _draw_overlay_and_zoom(
    src_img: Path,
    out_img_full: Path,
    out_img_zoom: Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    video_x: Optional[int],
    video_y: Optional[int],
    label: str,
    length_mm: float,
    tick_mm: float,
    label_side: int,
) -> Tuple[int, int, int, int]:
    img = Image.open(src_img).convert("RGB")
    w, h = img.size

    sx = 1.0
    sy = 1.0
    if video_x and video_x > 0:
        sx = w / float(video_x)
    if video_y and video_y > 0:
        sy = h / float(video_y)

    xx1 = x1 * sx
    yy1 = y1 * sy
    xx2 = x2 * sx
    yy2 = y2 * sy

    draw = ImageDraw.Draw(img, "RGBA")
    # Main scale line from line #21 (high-contrast overlay).
    draw.line((xx1, yy1, xx2, yy2), fill=(0, 0, 0, 235), width=11)
    draw.line((xx1, yy1, xx2, yy2), fill=(255, 0, 220, 255), width=7)
    # Endpoints
    r = 6
    draw.ellipse((xx1 - r, yy1 - r, xx1 + r, yy1 + r), fill=(255, 0, 220, 255), outline=(255, 255, 255, 255))
    draw.ellipse((xx2 - r, yy2 - r, xx2 + r, yy2 + r), fill=(0, 240, 255, 255), outline=(255, 255, 255, 255))

    # Draw scale ticks + labels based on line #21 values.
    # We treat the endpoint with lower y as 0 mm (consistent with vertical "down" scale).
    if length_mm > 0 and tick_mm > 0:
        if yy1 <= yy2:
            zx, zy = xx1, yy1
            fx, fy = xx2, yy2
        else:
            zx, zy = xx2, yy2
            fx, fy = xx1, yy1

        dx = fx - zx
        dy = fy - zy
        line_len_px = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux = dx / line_len_px
        uy = dy / line_len_px
        # Perpendicular unit vector for tick direction.
        px = -uy
        py = ux

        intervals = max(1, int(round(length_mm / tick_mm)))
        step_px = line_len_px / intervals
        # Keep labels readable by reducing label density when ticks are too dense.
        label_every = max(1, int(round(16.0 / max(1.0, step_px))))
        label_every = min(label_every, 20)

        side = -1 if label_side < 0 else 1
        minor_len = 8.0
        major_len = 14.0
        text_off = 8.0

        for i in range(intervals + 1):
            tx = zx + ux * (i * step_px)
            ty = zy + uy * (i * step_px)
            is_major = (i % max(1, label_every) == 0) or i == intervals
            tlen = major_len if is_major else minor_len

            x_a = tx - 0.5 * tlen * px
            y_a = ty - 0.5 * tlen * py
            x_b = tx + 0.5 * tlen * px
            y_b = ty + 0.5 * tlen * py
            draw.line((x_a, y_a, x_b, y_b), fill=(0, 240, 255, 245), width=2)

            if is_major:
                val_mm = i * tick_mm
                txt = f"{val_mm:.1f}"
                lx = tx + side * (0.5 * tlen + text_off)
                ly = ty - 7
                # Draw tiny dark box behind text for readability.
                tw = 7 * len(txt) + 6
                th = 16
                if side < 0:
                    bx1 = lx - tw
                    bx2 = lx
                    tx_draw = lx - tw + 3
                else:
                    bx1 = lx
                    bx2 = lx + tw
                    tx_draw = lx + 3
                by1 = ly - 1
                by2 = ly + th - 1
                draw.rectangle((bx1, by1, bx2, by2), fill=(0, 0, 0, 185))
                draw.text((tx_draw, ly), txt, fill=(255, 120, 255, 255))

    # Label box
    tx = 10
    ty = 10
    pad = 6
    tw = min(w - 20, 12 * max(10, min(80, len(label))))
    th = 28
    draw.rectangle((tx, ty, tx + tw, ty + th), fill=(0, 0, 0, 180))
    draw.text((tx + pad, ty + 7), label, fill=(255, 255, 255, 255))

    out_img_full.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img_full, quality=92)

    # Build a zoom crop around the scale line to make manual review easier.
    min_x = min(xx1, xx2)
    max_x = max(xx1, xx2)
    min_y = min(yy1, yy2)
    max_y = max(yy1, yy2)
    line_len = ((xx2 - xx1) ** 2 + (yy2 - yy1) ** 2) ** 0.5
    pad = max(120.0, line_len * 0.8)
    zx1 = max(0, int(min_x - pad))
    zy1 = max(0, int(min_y - pad))
    zx2 = min(w, int(max_x + pad))
    zy2 = min(h, int(max_y + pad))

    # Guarantee a minimally readable crop.
    min_side = 260
    if zx2 - zx1 < min_side:
        cx = 0.5 * (zx1 + zx2)
        zx1 = max(0, int(cx - min_side / 2))
        zx2 = min(w, int(cx + min_side / 2))
    if zy2 - zy1 < min_side:
        cy = 0.5 * (zy1 + zy2)
        zy1 = max(0, int(cy - min_side / 2))
        zy2 = min(h, int(cy + min_side / 2))

    zoom = img.crop((zx1, zy1, zx2, zy2))
    out_img_zoom.parent.mkdir(parents=True, exist_ok=True)
    zoom.save(out_img_zoom, quality=92)
    return w, h, zoom.size[0], zoom.size[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build HTML with preview overlays for scale GT anomaly review.")
    p.add_argument(
        "--anomaly-csv",
        type=Path,
        default=Path("artifacts/24_scale_pretrain_review/scale_gt_anomaly_candidates.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/24_scale_pretrain_review"),
    )
    p.add_argument("--max-rows", type=int, default=250)
    return p


def main() -> int:
    args = build_parser().parse_args()
    anomaly_csv = args.anomaly_csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    if not anomaly_csv.exists():
        raise FileNotFoundError(f"Anomaly CSV not found: {anomaly_csv}")

    rows = list(csv.DictReader(anomaly_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError("Anomaly CSV is empty")

    def severity_key(row: Dict[str, str]) -> Tuple[float, int]:
        delta = abs(_f(row.get("length_mm", 0.0)) - _f(row.get("depth_mm", 0.0)))
        reasons = row.get("reasons", "")
        bonus = 0
        if "LEN_DEPTH_DELTA_GT50MM" in reasons:
            bonus += 100
        if "X_OUTLIER_VS_GROUP_MEDIAN" in reasons:
            bonus += 40
        if "GROUP_X_SPREAD_GT120" in reasons:
            bonus += 20
        return (delta + bonus, len(reasons))

    rows_sorted = sorted(rows, key=severity_key, reverse=True)[: max(1, args.max_rows)]

    output_rows: List[Dict[str, str]] = []
    ok_count = 0
    missing_count = 0
    fail_count = 0

    for idx, r in enumerate(rows_sorted, start=1):
        fss_path = Path(r["fss_path"])
        setup_id = str(r.get("setup_id", "")).strip()
        depth_index = int(float(r.get("depth_index", "1")))
        source_img = _find_preview_image(fss_path=fss_path, setup_id=setup_id, depth_index=depth_index)

        delta = abs(_f(r["length_mm"]) - _f(r["depth_mm"]))
        label = f"{r['sample_id']} | d={r['depth_mm']}mm len={r['length_mm']}mm tick={r.get('tick_mm','?')} | Δ={delta:.1f}mm"

        preview_rel = ""
        preview_zoom_rel = ""
        status = "missing_source_image"
        source_img_str = ""
        image_size = ""
        zoom_size = ""
        notes = ""

        if source_img is None:
            missing_count += 1
        else:
            source_img_str = source_img.as_posix()
            out_name = f"{idx:04d}_{_safe_slug(r['sample_id'])}"
            out_img_full = previews_dir / f"{out_name}_full.jpg"
            out_img_zoom = previews_dir / f"{out_name}_zoom.jpg"
            vx, vy = _read_video_size_from_fss(fss_path)
            try:
                w, h, zw, zh = _draw_overlay_and_zoom(
                    src_img=source_img,
                    out_img_full=out_img_full,
                    out_img_zoom=out_img_zoom,
                    x1=_f(r["x1"]),
                    y1=_f(r["y1"]),
                    x2=_f(r["x2"]),
                    y2=_f(r["y2"]),
                    video_x=vx,
                    video_y=vy,
                    label=label,
                    length_mm=_f(r.get("length_mm", "0")),
                    tick_mm=_f(r.get("tick_mm", "0.5")),
                    label_side=int(float(r.get("label_side", "-1"))),
                )
                preview_rel = f"previews/{out_img_full.name}"
                preview_zoom_rel = f"previews/{out_img_zoom.name}"
                image_size = f"{w}x{h}"
                zoom_size = f"{zw}x{zh}"
                status = "ok"
                ok_count += 1
                if vx and vy and (w != vx or h != vy):
                    notes = f"scaled_from_video_{vx}x{vy}"
            except Exception as exc:
                fail_count += 1
                status = "overlay_failed"
                notes = str(exc)

        output_rows.append(
            {
                "rank": str(idx),
                "status": status,
                "preview_rel": preview_rel,
                "preview_zoom_rel": preview_zoom_rel,
                "sample_id": r.get("sample_id", ""),
                "split": r.get("split", ""),
                "setup_id": r.get("setup_id", ""),
                "depth_index": r.get("depth_index", ""),
                "depth_mm": r.get("depth_mm", ""),
                "length_mm": r.get("length_mm", ""),
                "delta_mm": f"{delta:.3f}",
                "x1": r.get("x1", ""),
                "x2": r.get("x2", ""),
                "y1": r.get("y1", ""),
                "y2": r.get("y2", ""),
                "reasons": r.get("reasons", ""),
                "source_img": source_img_str,
                "fss_path": r.get("fss_path", ""),
                "image_size": image_size,
                "zoom_size": zoom_size,
                "notes": notes,
            }
        )

    preview_csv = out_dir / "scale_gt_preview_index.csv"
    with preview_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(output_rows[0].keys()))
        w.writeheader()
        w.writerows(output_rows)

    html_rows = []
    for r in output_rows:
        if r["preview_rel"]:
            img_full_cell = (
                f'<a href="{_esc(r["preview_rel"])}" target="_blank">'
                f'<img class="full" src="{_esc(r["preview_rel"])}" loading="lazy" /></a>'
            )
        else:
            img_full_cell = "<span class='muted'>No image</span>"

        if r["preview_zoom_rel"]:
            img_zoom_cell = (
                f'<a href="{_esc(r["preview_zoom_rel"])}" target="_blank">'
                f'<img class="zoom" src="{_esc(r["preview_zoom_rel"])}" loading="lazy" /></a>'
            )
        else:
            img_zoom_cell = "<span class='muted'>No zoom</span>"

        html_rows.append(
            "<tr>"
            f"<td>{_esc(r['rank'])}</td>"
            f"<td>{img_full_cell}</td>"
            f"<td>{img_zoom_cell}</td>"
            f"<td>{_esc(r['status'])}</td>"
            f"<td>{_esc(r['sample_id'])}</td>"
            f"<td>{_esc(r['split'])}</td>"
            f"<td>{_esc(r['depth_index'])}</td>"
            f"<td>{_esc(r['depth_mm'])}</td>"
            f"<td>{_esc(r['length_mm'])}</td>"
            f"<td>{_esc(r['delta_mm'])}</td>"
            f"<td>{_esc(r['x1'])},{_esc(r['y1'])}<br/>{_esc(r['x2'])},{_esc(r['y2'])}</td>"
            f"<td>{_esc(r['reasons'])}</td>"
            f"<td class='path'>{_esc(r['image_size'])}<br/>zoom {_esc(r['zoom_size'])}<br/>{_esc(r['source_img'])}</td>"
            f"<td class='path'>{_esc(r['fss_path'])}</td>"
            f"<td>{_esc(r['notes'])}</td>"
            "</tr>"
        )

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "anomaly_csv": anomaly_csv.as_posix(),
        "rows_input": len(rows),
        "rows_rendered": len(output_rows),
        "rows_ok": ok_count,
        "rows_missing_source": missing_count,
        "rows_overlay_failed": fail_count,
        "preview_csv": preview_csv.as_posix(),
        "preview_dir": previews_dir.as_posix(),
    }
    (out_dir / "scale_gt_preview_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html_out = out_dir / "scale_gt_preview_review.html"
    html_out.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale GT Preview Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 18px; color:#1f2937; }}
    h1, h2 {{ margin: 0.2em 0; }}
    .meta {{ color: #4b5563; margin-bottom: 12px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#eef2ff; margin-right:8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; z-index: 1; }}
    img.full {{ width: 520px; max-width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 6px; }}
    img.zoom {{ width: 360px; max-width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 6px; }}
    .muted {{ color: #6b7280; }}
    .path {{ max-width: 520px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    details {{ margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>Scale GT Preview Review</h1>
  <div class="meta">
    Generated: {_esc(summary["generated_at"])}<br/>
    Input anomaly CSV: <code>{_esc(anomaly_csv.as_posix())}</code><br/>
    Preview index CSV: <code>{_esc(preview_csv.as_posix())}</code>
  </div>
  <p>
    <span class="badge">rows_rendered={_esc(summary["rows_rendered"])}</span>
    <span class="badge">rows_ok={_esc(summary["rows_ok"])}</span>
    <span class="badge">missing_source={_esc(summary["rows_missing_source"])}</span>
    <span class="badge">overlay_failed={_esc(summary["rows_overlay_failed"])}</span>
  </p>

  <details open>
    <summary><strong>Review Rows (click image to open full size)</strong></summary>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Full Preview</th>
          <th>Zoom Preview</th>
          <th>Status</th>
          <th>Sample</th>
          <th>Split</th>
          <th>Depth idx</th>
          <th>Depth mm</th>
          <th>Length mm</th>
          <th>Delta mm</th>
          <th>Coords</th>
          <th>Reasons</th>
          <th>Source Img</th>
          <th>FSS Path</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
  </details>
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"Preview HTML: {html_out}")
    print(f"Preview CSV: {preview_csv}")
    print(f"Preview summary: {out_dir / 'scale_gt_preview_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
