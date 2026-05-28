#!/usr/bin/env python3
"""Build an HTML gallery of DB_echo orientation marker templates by vendor."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

from prepare_lr_marker_dataset_sugiu_v2 import (
    SKIP_DIRS,
    TEMPLATE_RE,
    infer_manufacturer,
)


@dataclass
class TemplateVariant:
    vendor: str
    digest: str
    blank: bool
    extrema: Tuple[int, int]
    width: int
    height: int
    thumbnail: str
    occurrences: List[Dict[str, str]] = field(default_factory=list)


def iter_config_folders(dataset_roots: Sequence[Path], max_depth: int) -> Iterable[Path]:
    seen: set[str] = set()
    for root in dataset_roots:
        if not root.is_dir():
            continue
        root_depth = len(root.resolve().parts)
        for dirpath, dirnames, _filenames in os.walk(root):
            depth = max(0, len(Path(dirpath).resolve().parts) - root_depth)
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
            if max_depth >= 0 and depth > max_depth:
                dirnames[:] = []
                continue
            names = set(dirnames)
            if {"DB_setup", "DB_echo", "image_samples"}.issubset(names):
                folder = Path(dirpath).resolve()
                key = folder.as_posix()
                if key not in seen:
                    seen.add(key)
                    yield folder
                dirnames[:] = [name for name in dirnames if name not in {"DB_setup", "DB_echo", "image_samples"}]


def iter_template_paths(config_dir: Path) -> Iterable[Path]:
    echo_dir = config_dir / "DB_echo"
    if not echo_dir.is_dir():
        return
    for path in sorted(echo_dir.rglob("*")):
        if path.is_file() and TEMPLATE_RE.match(path.name):
            yield path.resolve()


def template_digest(image: Image.Image) -> str:
    gray = image.convert("L")
    payload = gray.tobytes() + str(gray.size).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:16]


def make_thumbnail(src: Image.Image, path: Path, blank: bool, max_side: int) -> None:
    gray = src.convert("L")
    scale = max(1, min(max_side // max(1, gray.width), max_side // max(1, gray.height)))
    scale = min(scale, 18)
    thumb = gray.resize((gray.width * scale, gray.height * scale), resample=Image.Resampling.NEAREST)
    thumb = ImageOps.autocontrast(thumb) if not blank else thumb
    canvas = Image.new("RGB", (max_side + 20, max_side + 38), "white")
    x = (max_side + 20 - thumb.width) // 2
    y = (max_side + 10 - thumb.height) // 2
    canvas.paste(thumb.convert("RGB"), (x, y))
    draw = ImageDraw.Draw(canvas)
    border = "#d61f1f" if blank else "#1f7a4d"
    draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=border, width=3)
    draw.text((8, max_side + 14), f"{gray.width}x{gray.height}", fill="#111111")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_gallery(args: argparse.Namespace) -> Dict[str, object]:
    roots = [p.expanduser().resolve() for p in args.dataset_roots]
    output_dir = args.output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[Tuple[str, str], TemplateVariant] = {}
    config_count_by_vendor: DefaultDict[str, int] = defaultdict(int)
    template_count_by_vendor: DefaultDict[str, int] = defaultdict(int)
    warnings: List[str] = []

    config_dirs = sorted(iter_config_folders(roots, max_depth=int(args.max_depth)), key=lambda p: p.as_posix().lower())
    for config_dir in config_dirs:
        vendor = infer_manufacturer(config_dir.name)
        config_count_by_vendor[vendor] += 1
        template_paths = list(iter_template_paths(config_dir))
        if not template_paths:
            warnings.append(f"{config_dir}: no orientation templates")
            continue
        for template_path in template_paths:
            try:
                with Image.open(template_path) as img:
                    gray = img.convert("L")
                    extrema = gray.getextrema() or (0, 0)
                    blank = int(extrema[1]) <= int(args.blank_template_max_value)
                    digest = template_digest(gray)
                    key = (vendor, digest)
                    if key not in variants:
                        thumb_name = f"{vendor.lower().replace(' ', '_')}_{digest}.png"
                        make_thumbnail(gray, assets_dir / thumb_name, blank=blank, max_side=int(args.thumbnail_size))
                        variants[key] = TemplateVariant(
                            vendor=vendor,
                            digest=digest,
                            blank=blank,
                            extrema=(int(extrema[0]), int(extrema[1])),
                            width=gray.width,
                            height=gray.height,
                            thumbnail=f"assets/{thumb_name}",
                        )
                    variants[key].occurrences.append(
                        {
                            "config": config_dir.as_posix(),
                            "config_name": config_dir.name,
                            "template": template_path.as_posix(),
                            "template_name": template_path.name,
                        }
                    )
                    template_count_by_vendor[vendor] += 1
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"{template_path}: {exc}")

    by_vendor: DefaultDict[str, List[TemplateVariant]] = defaultdict(list)
    for variant in variants.values():
        by_vendor[variant.vendor].append(variant)
    for vendor in by_vendor:
        by_vendor[vendor].sort(key=lambda v: (v.blank, v.width * v.height, v.digest))

    summary = {
        "dataset_roots": [p.as_posix() for p in roots],
        "config_dirs": len(config_dirs),
        "unique_templates": len(variants),
        "vendors": {
            vendor: {
                "configs": int(config_count_by_vendor[vendor]),
                "template_files": int(template_count_by_vendor[vendor]),
                "unique_templates": len(by_vendor.get(vendor, [])),
                "unique_blank_templates": sum(1 for v in by_vendor.get(vendor, []) if v.blank),
            }
            for vendor in sorted(config_count_by_vendor)
        },
        "warnings": warnings[:200],
    }
    write_html(output_dir / "index.html", summary=summary, by_vendor=by_vendor)
    (output_dir / "summary.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_html(path: Path, summary: Dict[str, object], by_vendor: Dict[str, List[TemplateVariant]]) -> None:
    vendor_options = "\n".join(
        f'<button class="vendor-tab" data-vendor="{html.escape(vendor)}">{html.escape(vendor)} <span>{len(items)}</span></button>'
        for vendor, items in sorted(by_vendor.items())
    )
    sections: List[str] = []
    for vendor, items in sorted(by_vendor.items()):
        cards: List[str] = []
        for variant in items:
            occurrence_count = len(variant.occurrences)
            examples = variant.occurrences[:8]
            more = occurrence_count - len(examples)
            examples_html = "".join(
                "<li>"
                f"<strong>{html.escape(item['template_name'])}</strong>"
                f"<span>{html.escape(item['config_name'])}</span>"
                "</li>"
                for item in examples
            )
            if more > 0:
                examples_html += f"<li><em>+{more} altre occorrenze</em></li>"
            badge = "black / excluded" if variant.blank else "usable"
            cards.append(
                f"""
                <article class="card {'blank' if variant.blank else 'usable'}">
                  <div class="thumb-wrap"><img src="{html.escape(variant.thumbnail)}" alt="template {html.escape(variant.digest)}"></div>
                  <div class="card-body">
                    <div class="badge">{badge}</div>
                    <h3>{html.escape(variant.digest)}</h3>
                    <p>{variant.width}x{variant.height} px · max {variant.extrema[1]} · {occurrence_count} file</p>
                    <ul>{examples_html}</ul>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="vendor-section" data-vendor="{html.escape(vendor)}">
              <header>
                <h2>{html.escape(vendor)}</h2>
                <p>{len(items)} template unici</p>
              </header>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    total_configs = summary["config_dirs"]
    total_unique = summary["unique_templates"]
    html_text = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Orientation marker templates by vendor</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --text: #14171f;
      --muted: #5c6472;
      --line: #d9dde5;
      --ok: #1f7a4d;
      --bad: #b82020;
      --panel: #ffffff;
      --accent: #1e5aa8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 14px;
    }}
    .vendor-tabs {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
      white-space: nowrap;
    }}
    button.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(30, 90, 168, 0.12);
    }}
    button span {{ color: var(--muted); margin-left: 4px; }}
    main {{ padding: 22px 24px 40px; }}
    .vendor-section {{ margin-bottom: 34px; }}
    .vendor-section.hidden {{ display: none; }}
    .vendor-section header {{
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 12px;
    }}
    h2 {{ font-size: 20px; margin: 0; letter-spacing: 0; }}
    .vendor-section header p {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr);
      gap: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 160px;
    }}
    .card.blank {{ border-color: rgba(184, 32, 32, 0.45); }}
    .thumb-wrap {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 130px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
    }}
    .thumb-wrap img {{
      max-width: 100%;
      height: auto;
      image-rendering: pixelated;
    }}
    .card-body {{ min-width: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid currentColor;
      color: var(--ok);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .blank .badge {{ color: var(--bad); }}
    h3 {{
      font-size: 14px;
      margin: 0 0 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      letter-spacing: 0;
    }}
    .card p {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 5px;
      font-size: 12px;
    }}
    li {{
      display: grid;
      gap: 1px;
      color: var(--muted);
      min-width: 0;
    }}
    li strong, li span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    li strong {{ color: var(--text); }}
    @media (max-width: 560px) {{
      .top, main {{ padding-left: 14px; padding-right: 14px; }}
      .card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="top">
    <h1>Orientation marker templates by vendor</h1>
    <div class="summary">
      <span>{total_configs} cartelle configurazione</span>
      <span>{total_unique} template unici</span>
      <span>template neri marcati come excluded</span>
    </div>
    <nav class="vendor-tabs">
      <button class="vendor-tab active" data-vendor="__all__">Tutti</button>
      {vendor_options}
    </nav>
  </div>
  <main>
    {''.join(sections)}
  </main>
  <script>
    const tabs = [...document.querySelectorAll('.vendor-tab')];
    const sections = [...document.querySelectorAll('.vendor-section')];
    for (const tab of tabs) {{
      tab.addEventListener('click', () => {{
        const vendor = tab.dataset.vendor;
        tabs.forEach(t => t.classList.toggle('active', t === tab));
        sections.forEach(section => {{
          section.classList.toggle('hidden', vendor !== '__all__' && section.dataset.vendor !== vendor);
        }});
      }});
    }}
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build orientation marker template gallery.")
    parser.add_argument("--dataset-roots", type=Path, nargs="+", default=[Path("/Volumes/SSD_esi1_n1")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/30_reports/orientation_template_gallery"))
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--blank-template-max-value", type=int, default=3)
    parser.add_argument("--thumbnail-size", type=int, default=96)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_gallery(args)
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
