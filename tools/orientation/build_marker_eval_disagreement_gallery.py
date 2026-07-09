#!/usr/bin/env python3
"""Build an HTML review gallery for marker-vs-legacy-line16 disagreements.

Reads `per_image_predictions.csv` produced by
`eval_marker_bundle_vs_legacy_line16.py`, draws overlays (legacy line-16 boxes
in green, detected marker in red, echo rect in blue) and groups the cards by
disagreement category:

- `outside_legacy_boxes`: marker accepted but its center is in no legacy box
- `group_mismatch`: predicted group != legacy box containing the marker
- `gt_mismatch`: predicted group != folder-level GT (SOLO NF/UD/... folders)
- `review`: rows the detector itself flagged
- `control_ok`: random sample of agreeing rows for visual comparison

Example:
  python3 tools/orientation/build_marker_eval_disagreement_gallery.py \
    --eval-dir artifacts/42_orientation_eval_vs_legacy_line16/pilot_v2 \
    --dataset-root /Volumes/SSD_esi1_n1
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "eval_marker_bundle_vs_legacy_line16", SCRIPT_DIR / "eval_marker_bundle_vs_legacy_line16.py"
)
_eval_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eval_mod
_spec.loader.exec_module(_eval_mod)  # type: ignore[union-attr]
_parse_fss = _eval_mod._parse_fss
GROUPS = _eval_mod.GROUPS

Rect = Tuple[int, int, int, int]  # top, left, bottom, right

CATEGORY_ORDER = ("outside_legacy_boxes", "group_mismatch", "gt_mismatch", "review", "control_ok")
CATEGORY_TITLES = {
    "outside_legacy_boxes": "Marker fuori da tutti i box legacy",
    "group_mismatch": "Gruppo predetto ≠ box legacy contenente il marker",
    "gt_mismatch": "Gruppo predetto ≠ GT cartella (SOLO NF/UD/...)",
    "review": "Casi flaggati review dal detector",
    "control_ok": "Controllo: casi in accordo (campione)",
}


def _parse_box(text: str) -> Optional[Rect]:
    if not text:
        return None
    parts = [int(float(p)) for p in text.split("|")]
    return (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None


def _load_folder_gt(dataset_root: Path, folder_name: str, cache: Dict[str, object]) -> Optional[object]:
    if folder_name in cache:
        return cache[folder_name]
    fss_candidates = sorted((dataset_root / folder_name / "DB_setup").glob("setup_*.fss"))
    result = None
    if fss_candidates:
        try:
            result = _parse_fss(fss_candidates[0])
        except ValueError:
            result = None
    cache[folder_name] = result
    return result


def _draw_overlay(
    image_path: Path,
    marker: Optional[Rect],
    rect_echo: Optional[Rect],
    line16: Optional[Dict[str, Rect]],
    duplicated: Optional[Dict[str, bool]],
    output_path: Path,
    max_width: int,
    jpeg_quality: int,
) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        draw = ImageDraw.Draw(rgb)

        def box_xy(rect: Rect):
            top, left, bottom, right = rect
            return [(left, top), (right, bottom)]

        if rect_echo:
            draw.rectangle(box_xy(rect_echo), outline=(60, 120, 255), width=2)
        if line16:
            for group in GROUPS:
                if duplicated and group != "NF" and duplicated.get(group):
                    continue
                rect = line16[group]
                draw.rectangle(box_xy(rect), outline=(0, 200, 0), width=3)
                draw.text((rect[1] + 4, rect[0] + 4), group, fill=(0, 200, 0))
        if marker:
            draw.rectangle(box_xy(marker), outline=(255, 40, 40), width=3)

        if rgb.width > max_width:
            scale = max_width / rgb.width
            rgb = rgb.resize((max_width, int(rgb.height * scale)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(output_path, quality=jpeg_quality)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Output dir of the eval script.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <eval-dir>/disagreement_gallery")
    parser.add_argument("--max-cards-per-category", type=int, default=60)
    parser.add_argument("--control-samples", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.output_dir or (args.eval_dir / "disagreement_gallery")
    csv_path = args.eval_dir / "per_image_predictions.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]

    rng = random.Random(args.seed)
    by_category: Dict[str, List[Dict[str, str]]] = {c: [] for c in CATEGORY_ORDER}
    for row in rows:
        marker = _parse_box(row.get("marker_box_abs", ""))
        agree_ok = True
        if row.get("gt_group") and row.get("pred_group") and row["gt_group"] != row["pred_group"]:
            by_category["gt_mismatch"].append(row)
            agree_ok = False
        if row.get("box_agree", "") == "0":
            by_category["group_mismatch"].append(row)
            agree_ok = False
        if marker and row.get("status") == "ok" and not row.get("gt_by_box"):
            by_category["outside_legacy_boxes"].append(row)
            agree_ok = False
        if row.get("status") not in ("ok", ""):
            by_category["review"].append(row)
            agree_ok = False
        if agree_ok and marker and row.get("gt_by_box"):
            by_category["control_ok"].append(row)

    by_category["control_ok"] = rng.sample(
        by_category["control_ok"], min(args.control_samples, len(by_category["control_ok"]))
    )

    gt_cache: Dict[str, object] = {}
    sections: List[str] = []
    total_cards = 0
    for category in CATEGORY_ORDER:
        selected = by_category[category][: args.max_cards_per_category]
        cards: List[str] = []
        for idx, row in enumerate(selected):
            folder = str(row["folder"])
            image_path = args.dataset_root / folder / str(row["image_id"])
            if not image_path.is_file():
                continue
            gt = _load_folder_gt(args.dataset_root, folder, gt_cache)
            rect_echo, line16, duplicated = gt if gt else (None, None, None)
            overlay_rel = Path("overlays") / category / f"{total_cards:04d}.jpg"
            try:
                _draw_overlay(
                    image_path,
                    _parse_box(row.get("marker_box_abs", "")),
                    rect_echo,
                    line16,
                    duplicated,
                    out / overlay_rel,
                    args.max_width,
                    args.jpeg_quality,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] overlay failed for {image_path}: {exc}")
                continue
            total_cards += 1
            meta = (
                f"pred: <b>{html.escape(row.get('pred_group') or '-')}</b> | "
                f"gt_box: {html.escape(row.get('gt_by_box') or '-')} | "
                f"gt_cartella: {html.escape(row.get('gt_group') or '-')} | "
                f"score: {html.escape((row.get('match_score') or '')[:6])} | "
                f"scope: {html.escape(row.get('search_scope') or '-')} | "
                f"status: {html.escape(row.get('status') or '-')}"
                + (f" | reason: {html.escape(row['review_reason'])}" if row.get("review_reason") else "")
            )
            cards.append(
                "<div class='card'>"
                f"<div class='title'>{html.escape(folder)}<br><small>{html.escape(str(row['image_id']))}</small></div>"
                f"<a href='{overlay_rel.as_posix()}' target='_blank'><img loading='lazy' src='{overlay_rel.as_posix()}'></a>"
                f"<div class='meta'>{meta}</div>"
                "</div>"
            )
        sections.append(
            f"<h2>{html.escape(CATEGORY_TITLES[category])} "
            f"<span class='count'>({len(by_category[category])} casi, mostrati {len(cards)})</span></h2>"
            + ("<div class='grid'>" + "".join(cards) + "</div>" if cards else "<p>Nessun caso.</p>")
        )

    page = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>Disaccordi marker vs riga 16 legacy</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #16181d; color: #e8e8e8; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #444; padding-bottom: 6px; }}
.count {{ color: #999; font-weight: normal; }}
.legend span {{ margin-right: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 14px; }}
.card {{ background: #22252c; border-radius: 8px; padding: 10px; }}
.card img {{ width: 100%; border-radius: 4px; }}
.title {{ font-size: 12px; margin-bottom: 6px; color: #9ecbff; }}
.meta {{ font-size: 12px; margin-top: 6px; color: #ccc; }}
</style></head><body>
<h1>Disaccordi marker vs riga 16 legacy</h1>
<p class="legend"><span style="color:#3c78ff">■ rect ecografico (riga 11)</span>
<span style="color:#00c800">■ box riga 16 legacy (NF/LR/UD/LRUD)</span>
<span style="color:#ff2828">■ marker trovato dal detector</span></p>
{''.join(sections)}
</body></html>"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Gallery: {out / 'index.html'} ({total_cards} overlay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
