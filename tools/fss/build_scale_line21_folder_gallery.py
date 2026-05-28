#!/usr/bin/env python3
"""Build one-image-per-folder gallery with line21 scale overlay.

Scans a volume recursively for config directories that contain:
- DB_setup/setup_*.fss
- image_samples/

For each valid config directory, picks one representative image,
extracts SCALE_LINE (line 21) from the .fss, overlays it, and builds
an HTML gallery navigable with keyboard arrows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LINE21_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|(-?\d+(?:\.\d+)?)\|"
)
SETUP_IDX_PATTERN = re.compile(r"_setup_(\d+)$")


@dataclass
class ScaleEntry:
    x1: float
    x2: float
    y1: float
    y2: float
    length_mm: float
    tick_mm: float
    label_side: int
    raw: str


@dataclass
class Item:
    index: int
    config_dir: str
    folder_name: str
    fss_path: str
    source_image: str
    preview_image: str
    setup_id: str
    depth_index_from_image: int
    depth_index_used: int
    entries_count: int
    video_x: Optional[int]
    video_y: Optional[int]
    entry_raw: str
    x1: float
    x2: float
    y1: float
    y2: float
    length_mm: float
    tick_mm: float
    label_side: int


def _safe_slug(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return out[:140] if out else "item"


def _read_fss_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _read_video_size(lines: List[str]) -> tuple[Optional[int], Optional[int]]:
    # Keep existing project convention: lines[8]=video_x and lines[9]=video_y (0-based).
    if len(lines) < 10:
        return None, None
    try:
        vx = int(lines[8].strip().replace("\r", ""))
        vy = int(lines[9].strip().replace("\r", ""))
        return vx, vy
    except Exception:
        return None, None


def _parse_line21(lines: List[str]) -> List[ScaleEntry]:
    raw21 = lines[20].strip() if len(lines) >= 21 else ""
    if not raw21:
        return []

    entries: List[ScaleEntry] = []
    for m in LINE21_PATTERN.finditer(raw21):
        try:
            entries.append(
                ScaleEntry(
                    x1=float(m.group(1)),
                    x2=float(m.group(2)),
                    y1=float(m.group(3)),
                    y2=float(m.group(4)),
                    length_mm=float(m.group(5)),
                    tick_mm=float(m.group(6)),
                    label_side=int(round(float(m.group(7)))),
                    raw=m.group(0),
                )
            )
        except Exception:
            continue
    return entries


def _depth_index_from_image_name(img_path: Path) -> int:
    m = SETUP_IDX_PATTERN.search(img_path.stem)
    if not m:
        return 1
    try:
        return int(m.group(1)) + 1
    except Exception:
        return 1


def _pick_representative_image(image_samples: Path) -> Optional[Path]:
    # Priority order for full-frame images.
    priorities = [
        "image_depth_value_setup_0",
        "image_depth_find_no_flip_setup_0",
        "image_depth_find_flip_ud_setup_0",
        "image_depth_find_flip_lr_setup_0",
        "image_depth_find_flip_lrud_setup_0",
        "image_biplana_setup_0",
        "image_calibration_0",
    ]

    for stem in priorities:
        for ext in IMG_EXTS:
            c = image_samples / f"{stem}{ext}"
            if c.exists() and not c.name.startswith("._"):
                return c

    # Prefer image_ prefix, skip AppleDouble files.
    for ext in IMG_EXTS:
        for p in sorted(image_samples.glob(f"image_*{ext}")):
            if not p.name.startswith("._"):
                return p

    # Last fallback: any supported image.
    for ext in IMG_EXTS:
        for p in sorted(image_samples.glob(f"*{ext}")):
            if not p.name.startswith("._"):
                return p
    return None


def _choose_entry(entries: List[ScaleEntry], depth_index: int) -> tuple[ScaleEntry, int]:
    if not entries:
        raise RuntimeError("No line21 entries")
    idx = max(1, depth_index)
    if idx <= len(entries):
        return entries[idx - 1], idx
    return entries[0], 1


def _draw_overlay(
    src_img: Path,
    out_img: Path,
    entry: ScaleEntry,
    video_x: Optional[int],
    video_y: Optional[int],
    legend: str,
) -> None:
    img = Image.open(src_img).convert("RGB")
    w, h = img.size

    sx = (w / float(video_x)) if (video_x and video_x > 0) else 1.0
    sy = (h / float(video_y)) if (video_y and video_y > 0) else 1.0

    x1 = entry.x1 * sx
    y1 = entry.y1 * sy
    x2 = entry.x2 * sx
    y2 = entry.y2 * sy

    draw = ImageDraw.Draw(img, "RGBA")

    # Main axis
    draw.line((x1, y1, x2, y2), fill=(0, 0, 0, 230), width=11)
    draw.line((x1, y1, x2, y2), fill=(255, 0, 210, 255), width=7)

    # Endpoints
    rr = 6
    draw.ellipse((x1 - rr, y1 - rr, x1 + rr, y1 + rr), fill=(255, 0, 210, 255), outline=(255, 255, 255, 255))
    draw.ellipse((x2 - rr, y2 - rr, x2 + rr, y2 + rr), fill=(0, 235, 255, 255), outline=(255, 255, 255, 255))

    # Sparse ticks and labels, so rendering stays fast even with tiny tick_mm.
    if entry.length_mm > 0 and entry.tick_mm > 0:
        if y1 <= y2:
            zx, zy, fx, fy = x1, y1, x2, y2
        else:
            zx, zy, fx, fy = x2, y2, x1, y1

        dx = fx - zx
        dy = fy - zy
        ll = math.hypot(dx, dy)
        if ll > 1e-6:
            ux, uy = dx / ll, dy / ll
            px, py = -uy, ux

            intervals = max(1, int(round(entry.length_mm / entry.tick_mm)))
            major_every = max(1, int(round(5.0 / max(entry.tick_mm, 1e-6))))  # 5 mm labels
            major_every = min(major_every, 200)

            side = -1 if entry.label_side < 0 else 1
            tick_len = 14.0
            text_off = 8.0

            for i in range(0, intervals + 1):
                if i not in (0, intervals) and (i % major_every != 0):
                    continue
                t = i / float(intervals)
                tx = zx + dx * t
                ty = zy + dy * t

                ax = tx - 0.5 * tick_len * px
                ay = ty - 0.5 * tick_len * py
                bx = tx + 0.5 * tick_len * px
                by = ty + 0.5 * tick_len * py
                draw.line((ax, ay, bx, by), fill=(0, 235, 255, 245), width=2)

                val = i * entry.tick_mm
                txt = f"{val:.1f}".rstrip("0").rstrip(".")
                lx = tx + side * (0.5 * tick_len + text_off)
                ly = ty - 7
                tw = 7 * len(txt) + 6
                th = 15
                if side < 0:
                    rx1, rx2 = lx - tw, lx
                    tx_draw = lx - tw + 3
                else:
                    rx1, rx2 = lx, lx + tw
                    tx_draw = lx + 3
                ry1, ry2 = ly - 1, ly + th - 1
                draw.rectangle((rx1, ry1, rx2, ry2), fill=(0, 0, 0, 185))
                draw.text((tx_draw, ly), txt, fill=(255, 120, 255, 255))

    # Legend
    bx1, by1 = 10, 10
    bw = min(w - 20, max(300, min(1400, 8 * len(legend) + 20)))
    bh = 26
    draw.rectangle((bx1, by1, bx1 + bw, by1 + bh), fill=(0, 0, 0, 180))
    draw.text((bx1 + 6, by1 + 6), legend, fill=(255, 255, 255, 255))

    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img, quality=92)


def _scan_config_dirs(volume_root: Path) -> List[Path]:
    out: List[Path] = []
    for db_setup in volume_root.rglob("DB_setup"):
        cfg_dir = db_setup.parent
        if cfg_dir.name.startswith("."):
            continue
        image_samples = cfg_dir / "image_samples"
        if not image_samples.exists():
            continue
        fss_files = sorted(db_setup.glob("setup_*.fss"))
        if not fss_files:
            continue
        out.append(cfg_dir)
    out = sorted(set(out), key=lambda p: p.as_posix().lower())
    return out


def _build_html(items: List[Item], html_path: Path, title: str) -> None:
    payload = [
        {
            "index": it.index,
            "folder_name": it.folder_name,
            "config_dir": it.config_dir,
            "preview_image": it.preview_image,
            "source_image": it.source_image,
            "fss_path": it.fss_path,
            "setup_id": it.setup_id,
            "depth_index_from_image": it.depth_index_from_image,
            "depth_index_used": it.depth_index_used,
            "entries_count": it.entries_count,
            "video_x": it.video_x,
            "video_y": it.video_y,
            "entry_raw": it.entry_raw,
            "x1": it.x1,
            "x2": it.x2,
            "y1": it.y1,
            "y2": it.y2,
            "length_mm": it.length_mm,
            "tick_mm": it.tick_mm,
            "label_side": it.label_side,
        }
        for it in items
    ]

    html = f"""<!doctype html>
<html lang=\"it\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #101a30;
      --ink: #e5eefc;
      --muted: #95a7c2;
      --line: #263a5d;
      --accent: #3b82f6;
      --accent2: #14b8a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--ink); }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 14px; display:grid; gap:12px; }}
    .top {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; display:grid; gap:10px; }}
    .title {{ font-size: 1.2rem; font-weight: 700; }}
    .row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    button {{ border:0; border-radius:10px; padding:8px 12px; cursor:pointer; font-weight:600; }}
    .b1 {{ background:linear-gradient(130deg,var(--accent2),var(--accent)); color:#fff; }}
    .b2 {{ background:#1d2b47; color:#d6e2f7; border:1px solid #2c4370; }}
    input[type=range] {{ width:min(520px, 90vw); }}
    .k {{ color:var(--muted); font-size:0.9rem; }}
    .main {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .imgbox {{ width:100%; display:grid; place-items:center; background:#050913; border-radius:10px; border:1px solid #1c2d4a; min-height:220px; }}
    img {{ max-width:100%; max-height:76vh; object-fit:contain; border-radius:8px; }}
    .meta {{ margin-top:10px; display:grid; gap:7px; }}
    .meta code {{ color:#c7dbff; background:#0b172b; padding:2px 5px; border-radius:6px; }}
    .path {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    @media (max-width: 900px) {{
      .wrap {{ padding:8px; }}
      .title {{ font-size:1rem; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"top\">
      <div class=\"title\">{title}</div>
      <div class=\"row\">
        <button class=\"b1\" id=\"prevBtn\">← Prec</button>
        <button class=\"b1\" id=\"nextBtn\">Succ →</button>
        <input type=\"range\" id=\"idx\" min=\"0\" max=\"{max(0, len(payload)-1)}\" value=\"0\" />
        <span id=\"counter\" class=\"k\"></span>
      </div>
      <div class=\"k\">Scorciatoie tastiera: <b>←</b> precedente, <b>→</b> successiva. (Shift + freccia = salto di 10)</div>
    </section>

    <section class=\"main\">
      <div class=\"imgbox"><img id=\"img\" alt=\"preview\" /></div>
      <div class=\"meta\" id=\"meta\"></div>
    </section>
  </div>

  <script id=\"payload\" type=\"application/json\">{json.dumps(payload, ensure_ascii=False)}</script>
  <script>
    const items = JSON.parse(document.getElementById('payload').textContent || '[]');
    const imgEl = document.getElementById('img');
    const metaEl = document.getElementById('meta');
    const counterEl = document.getElementById('counter');
    const idxEl = document.getElementById('idx');
    let i = 0;

    function esc(s) {{
      return String(s ?? '').replace(/[&<>\"]/g, (c) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
    }}

    function clamp(v) {{
      if (!items.length) return 0;
      return Math.max(0, Math.min(items.length - 1, v));
    }}

    function setIndex(v) {{
      i = clamp(v);
      const it = items[i];
      idxEl.value = String(i);
      counterEl.textContent = `${{i+1}} / ${{items.length}}`;
      imgEl.src = it.preview_image;
      metaEl.innerHTML = `
        <div><b>Folder:</b> <code>${{esc(it.folder_name)}}</code></div>
        <div class=\"path\"><b>Config dir:</b> <code>${{esc(it.config_dir)}}</code></div>
        <div class=\"path\"><b>Source image:</b> <code>${{esc(it.source_image)}}</code></div>
        <div class=\"path\"><b>FSS:</b> <code>${{esc(it.fss_path)}}</code></div>
        <div><b>Setup:</b> <code>${{esc(it.setup_id)}}</code> | <b>Depth from image:</b> <code>${{esc(it.depth_index_from_image)}}</code> | <b>Depth used:</b> <code>${{esc(it.depth_index_used)}}</code> / <code>${{esc(it.entries_count)}}</code></div>
        <div><b>Video size:</b> <code>${{esc(it.video_x)}} x ${{esc(it.video_y)}}</code> | <b>Coords:</b> <code>(${{esc(it.x1)}}, ${{esc(it.y1)}}) -> (${{esc(it.x2)}}, ${{esc(it.y2)}})</code></div>
        <div><b>Scale:</b> length <code>${{esc(it.length_mm)}} mm</code>, tick <code>${{esc(it.tick_mm)}}</code>, side <code>${{esc(it.label_side)}}</code></div>
        <details><summary>Entry raw (linea 21)</summary><code>${{esc(it.entry_raw)}}</code></details>
      `;
    }}

    document.getElementById('prevBtn').addEventListener('click', () => setIndex(i - 1));
    document.getElementById('nextBtn').addEventListener('click', () => setIndex(i + 1));
    idxEl.addEventListener('input', () => setIndex(parseInt(idxEl.value || '0', 10)));

    window.addEventListener('keydown', (e) => {{
      if (!items.length) return;
      if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        setIndex(i - (e.shiftKey ? 10 : 1));
      }} else if (e.key === 'ArrowRight') {{
        e.preventDefault();
        setIndex(i + (e.shiftKey ? 10 : 1));
      }}
    }});

    setIndex(0);
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def build_gallery(volume_root: Path, output_dir: Path, title: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    config_dirs = _scan_config_dirs(volume_root)
    items: List[Item] = []

    skip_no_image = 0
    skip_bad_fss = 0
    skip_no_line21 = 0

    for idx, cfg_dir in enumerate(config_dirs, start=1):
        db_setup = cfg_dir / "DB_setup"
        image_samples = cfg_dir / "image_samples"

        fss_files = sorted(db_setup.glob("setup_*.fss"))
        if not fss_files:
            skip_bad_fss += 1
            continue
        fss_path = fss_files[0]

        src_img = _pick_representative_image(image_samples)
        if src_img is None:
            skip_no_image += 1
            continue

        try:
            lines = _read_fss_lines(fss_path)
            video_x, video_y = _read_video_size(lines)
            entries = _parse_line21(lines)
        except Exception:
            skip_bad_fss += 1
            continue

        if not entries:
            skip_no_line21 += 1
            continue

        depth_from_image = _depth_index_from_image_name(src_img)
        entry, depth_used = _choose_entry(entries, depth_from_image)

        setup_id = fss_path.stem.replace("setup_", "")
        slug = _safe_slug(cfg_dir.as_posix().replace(volume_root.as_posix(), "").strip("/"))
        out_img = previews_dir / f"{idx:04d}_{slug}.jpg"
        legend = f"{cfg_dir.name} | setup {setup_id} | depth {depth_used}"

        try:
            _draw_overlay(src_img, out_img, entry, video_x, video_y, legend)
        except Exception:
            skip_bad_fss += 1
            continue

        items.append(
            Item(
                index=len(items) + 1,
                config_dir=cfg_dir.as_posix(),
                folder_name=cfg_dir.name,
                fss_path=fss_path.as_posix(),
                source_image=src_img.as_posix(),
                preview_image=out_img.relative_to(output_dir).as_posix(),
                setup_id=setup_id,
                depth_index_from_image=depth_from_image,
                depth_index_used=depth_used,
                entries_count=len(entries),
                video_x=video_x,
                video_y=video_y,
                entry_raw=entry.raw,
                x1=entry.x1,
                x2=entry.x2,
                y1=entry.y1,
                y2=entry.y2,
                length_mm=entry.length_mm,
                tick_mm=entry.tick_mm,
                label_side=entry.label_side,
            )
        )

    html_path = output_dir / "scale_line21_gallery.html"
    _build_html(items=items, html_path=html_path, title=title)

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "volume_root": volume_root.as_posix(),
        "config_dirs_found": len(config_dirs),
        "items_rendered": len(items),
        "skipped_no_image": skip_no_image,
        "skipped_bad_fss": skip_bad_fss,
        "skipped_no_line21": skip_no_line21,
        "html": html_path.as_posix(),
        "previews_dir": previews_dir.as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build line21 gallery with one image per config folder.")
    p.add_argument("--volume-root", type=Path, default=Path("/Volumes/SSD_esi1_n1"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/34_scale_gt_fix_focus_20260416/scale_line21_gallery"),
    )
    p.add_argument("--title", type=str, default="Gallery Scala Linea 21 - 1 immagine per cartella")
    return p


def main() -> int:
    args = build_parser().parse_args()
    volume_root = args.volume_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not volume_root.exists():
        raise FileNotFoundError(f"Volume root not found: {volume_root}")

    summary = build_gallery(volume_root=volume_root, output_dir=output_dir, title=args.title)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
