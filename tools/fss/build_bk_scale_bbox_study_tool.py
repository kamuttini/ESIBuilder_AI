#!/usr/bin/env python3
"""Build an interactive BK study HTML for scale bbox recognition errors.

This tool reads bbox prediction CSVs (train/val/test), ranks rows by error, renders
overlay previews, and writes a one-card-at-a-time interactive HTML page with:
- keyboard navigation (left/right/up/down),
- split/error filters,
- per-sample notes,
- per-sample issue tags,
- optional "exclude from training" flag,
- CSV export of annotations.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


@dataclass
class ReviewRow:
    rank: int
    sample_id: str
    split: str
    image_path: str
    source_name: str
    config_folder: str
    setup_id: str
    depth_index: int
    x1_gt: float
    y1_gt: float
    x2_gt: float
    y2_gt: float
    x1_pred: float
    y1_pred: float
    x2_pred: float
    y2_pred: float
    x_err_px: float
    y_err_px: float
    iou: float
    dx_center_px: float
    dy_center_px: float
    w_err_px: float
    h_err_px: float
    score: float
    priority: str
    auto_tags: str
    fss_path: str = ""
    scale_line_raw: str = ""


def _f(text: str) -> float:
    return float(str(text).strip())


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:140]


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)_([0-9]+)$", sample_id.strip())
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def box_iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def config_from_image_path(image_path: str) -> str:
    p = Path(image_path)
    parts = p.parts
    needle = "SSD_esi1_n1"
    if needle in parts:
        i = parts.index(needle)
        if i + 1 < len(parts):
            return parts[i + 1]
    # fallback: keep parent folder as best effort
    if len(parts) >= 2:
        return parts[-2]
    return ""


def find_fss_for_row(image_path: str, setup_id: str, cache: Dict[Tuple[str, str], str]) -> str:
    p = Path(image_path)
    key = (str(p.parent), setup_id)
    if key in cache:
        return cache[key]

    target = f"setup_{setup_id}.fss"
    for anc in [p.parent, *p.parents]:
        cand = anc / "DB_setup" / target
        if cand.exists():
            out = cand.as_posix()
            cache[key] = out
            return out

    # limited fallback recursive lookup around current config root
    root: Optional[Path] = None
    for anc in [p.parent, *p.parents]:
        if anc.name == "SSD_esi1_n1":
            break
        root = anc
    if root is not None and root.exists():
        hits = sorted(root.glob(f"**/DB_setup/{target}"))
        if hits:
            out = hits[0].as_posix()
            cache[key] = out
            return out

    cache[key] = ""
    return ""


def read_scale_line21_raw(fss_path: str, cache: Dict[str, str]) -> str:
    if not fss_path:
        return ""
    if fss_path in cache:
        return cache[fss_path]
    p = Path(fss_path)
    if not p.exists():
        cache[fss_path] = ""
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        raw = lines[20].strip() if len(lines) >= 21 else ""
    except Exception:
        raw = ""
    cache[fss_path] = raw
    return raw


def auto_tags_for_row(dx: float, dy: float, w_err: float, h_err: float, iou: float) -> str:
    tags: List[str] = []
    if dx > 7.0:
        tags.append("x_destra")
    elif dx < -7.0:
        tags.append("x_sinistra")
    if dy > 7.0:
        tags.append("y_bassa")
    elif dy < -7.0:
        tags.append("y_alta")
    if w_err > 9.0:
        tags.append("troppo_larga")
    elif w_err < -9.0:
        tags.append("troppo_stretta")
    if h_err > 14.0:
        tags.append("troppo_alta")
    elif h_err < -14.0:
        tags.append("troppo_bassa")
    if iou < 0.40:
        tags.append("iou_bassa")
    return "|".join(tags)


def priority_from_metrics(dx: float, dy: float, iou: float) -> str:
    adx = abs(dx)
    ady = abs(dy)
    if iou < 0.22 or adx > 70.0 or ady > 80.0:
        return "P0"
    if iou < 0.40 or adx > 38.0 or ady > 45.0:
        return "P1"
    if iou < 0.62 or adx > 20.0 or ady > 24.0:
        return "P2"
    return "P3"


def score_row(dx: float, dy: float, w_err: float, h_err: float, iou: float) -> float:
    # Weighted score tuned for "misalignment first", then shape errors, then IoU.
    return (
        abs(dx) * 1.20
        + abs(dy) * 0.95
        + abs(w_err) * 0.55
        + abs(h_err) * 0.35
        + (1.0 - iou) * 65.0
    )


def parse_prediction_row(d: Dict[str, str]) -> Optional[ReviewRow]:
    try:
        sample_id = str(d.get("sample_id", "")).strip()
        split = str(d.get("split", "")).strip().lower()
        image_path = str(d.get("image_path", "")).strip()
        if not sample_id or split not in {"train", "val", "test"} or not image_path:
            return None

        x1_gt = _f(d["x1_gt"])
        y1_gt = _f(d["y1_gt"])
        x2_gt = _f(d["x2_gt"])
        y2_gt = _f(d["y2_gt"])
        x1_pred = _f(d["x1_pred"])
        y1_pred = _f(d["y1_pred"])
        x2_pred = _f(d["x2_pred"])
        y2_pred = _f(d["y2_pred"])
    except Exception:
        return None

    # normalize box order
    x1_gt, x2_gt = min(x1_gt, x2_gt), max(x1_gt, x2_gt)
    y1_gt, y2_gt = min(y1_gt, y2_gt), max(y1_gt, y2_gt)
    x1_pred, x2_pred = min(x1_pred, x2_pred), max(x1_pred, x2_pred)
    y1_pred, y2_pred = min(y1_pred, y2_pred), max(y1_pred, y2_pred)

    iou = box_iou_xyxy((x1_pred, y1_pred, x2_pred, y2_pred), (x1_gt, y1_gt, x2_gt, y2_gt))
    x_err = 0.5 * (abs(x1_pred - x1_gt) + abs(x2_pred - x2_gt))
    y_err = 0.5 * (abs(y1_pred - y1_gt) + abs(y2_pred - y2_gt))

    x_c_gt = 0.5 * (x1_gt + x2_gt)
    y_c_gt = 0.5 * (y1_gt + y2_gt)
    x_c_pr = 0.5 * (x1_pred + x2_pred)
    y_c_pr = 0.5 * (y1_pred + y2_pred)
    dx = x_c_pr - x_c_gt
    dy = y_c_pr - y_c_gt
    w_err = (x2_pred - x1_pred) - (x2_gt - x1_gt)
    h_err = (y2_pred - y1_pred) - (y2_gt - y1_gt)

    score = score_row(dx=dx, dy=dy, w_err=w_err, h_err=h_err, iou=iou)
    priority = priority_from_metrics(dx=dx, dy=dy, iou=iou)
    auto_tags = auto_tags_for_row(dx=dx, dy=dy, w_err=w_err, h_err=h_err, iou=iou)

    setup_id, depth_index = parse_sample_id(sample_id)
    return ReviewRow(
        rank=0,
        sample_id=sample_id,
        split=split,
        image_path=image_path,
        source_name=Path(image_path).name,
        config_folder=config_from_image_path(image_path),
        setup_id=setup_id,
        depth_index=depth_index,
        x1_gt=x1_gt,
        y1_gt=y1_gt,
        x2_gt=x2_gt,
        y2_gt=y2_gt,
        x1_pred=x1_pred,
        y1_pred=y1_pred,
        x2_pred=x2_pred,
        y2_pred=y2_pred,
        x_err_px=x_err,
        y_err_px=y_err,
        iou=iou,
        dx_center_px=dx,
        dy_center_px=dy,
        w_err_px=w_err,
        h_err_px=h_err,
        score=score,
        priority=priority,
        auto_tags=auto_tags,
    )


def load_rows(
    predictions: Sequence[Path],
    split_filter: Optional[set[str]],
    dedupe_image_path: bool,
) -> List[ReviewRow]:
    dedup: Dict[str, ReviewRow] = {}
    rows: List[ReviewRow] = []

    for p in predictions:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for d in reader:
                row = parse_prediction_row(d)
                if row is None:
                    continue
                if split_filter and row.split not in split_filter:
                    continue
                key = row.image_path if dedupe_image_path else f"{row.split}|{row.sample_id}|{row.image_path}"
                if dedupe_image_path:
                    old = dedup.get(key)
                    if old is None or row.score > old.score:
                        dedup[key] = row
                else:
                    rows.append(row)

    if dedupe_image_path:
        rows = list(dedup.values())
    return rows


def enrich_rows_with_fss(rows: Sequence[ReviewRow]) -> None:
    fss_cache: Dict[Tuple[str, str], str] = {}
    raw_cache: Dict[str, str] = {}
    for r in rows:
        if not r.setup_id:
            continue
        fss = find_fss_for_row(r.image_path, r.setup_id, fss_cache)
        r.fss_path = fss
        r.scale_line_raw = read_scale_line21_raw(fss, raw_cache)


def draw_overlay_previews(row: ReviewRow, full_out: Path, zoom_out: Path) -> None:
    img = Image.open(row.image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    gt_outline = (30, 214, 124, 255)
    gt_fill = (30, 214, 124, 40)
    pr_outline = (236, 72, 153, 255)
    pr_fill = (236, 72, 153, 35)
    ctr_gt = (34, 197, 94, 255)
    ctr_pr = (217, 70, 239, 255)

    draw.rectangle((row.x1_gt, row.y1_gt, row.x2_gt, row.y2_gt), outline=gt_outline, width=5, fill=gt_fill)
    draw.rectangle((row.x1_pred, row.y1_pred, row.x2_pred, row.y2_pred), outline=pr_outline, width=4, fill=pr_fill)

    gx = 0.5 * (row.x1_gt + row.x2_gt)
    px = 0.5 * (row.x1_pred + row.x2_pred)
    gy1, gy2 = row.y1_gt, row.y2_gt
    py1, py2 = row.y1_pred, row.y2_pred
    draw.line((gx, gy1, gx, gy2), fill=ctr_gt, width=3)
    draw.line((px, py1, px, py2), fill=ctr_pr, width=3)

    # visible corners
    r = 4
    corners = [
        (row.x1_gt, row.y1_gt, gt_outline),
        (row.x2_gt, row.y1_gt, gt_outline),
        (row.x1_gt, row.y2_gt, gt_outline),
        (row.x2_gt, row.y2_gt, gt_outline),
        (row.x1_pred, row.y1_pred, pr_outline),
        (row.x2_pred, row.y1_pred, pr_outline),
        (row.x1_pred, row.y2_pred, pr_outline),
        (row.x2_pred, row.y2_pred, pr_outline),
    ]
    for x, y, c in corners:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(255, 255, 255, 255))

    full_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(full_out, quality=92)

    w, h = img.size
    x_min = min(row.x1_gt, row.x1_pred)
    y_min = min(row.y1_gt, row.y1_pred)
    x_max = max(row.x2_gt, row.x2_pred)
    y_max = max(row.y2_gt, row.y2_pred)
    bw = max(20.0, x_max - x_min)
    bh = max(20.0, y_max - y_min)
    pad_x = max(120.0, 0.9 * bw)
    pad_y = max(90.0, 0.45 * bh)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    zx1 = int(max(0.0, cx - 0.5 * bw - pad_x))
    zy1 = int(max(0.0, cy - 0.5 * bh - pad_y))
    zx2 = int(min(float(w), cx + 0.5 * bw + pad_x))
    zy2 = int(min(float(h), cy + 0.5 * bh + pad_y))
    if zx2 <= zx1 or zy2 <= zy1:
        crop = img
    else:
        crop = img.crop((zx1, zy1, zx2, zy2))

    zoom_out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(zoom_out, quality=92)


def _dataset_key(rows: Sequence[ReviewRow]) -> str:
    if not rows:
        return "empty"
    prefix = "|".join(r.sample_id for r in rows[:5])
    return f"{len(rows)}_{_safe_slug(prefix)}"


def _cards_html(rows: Sequence[ReviewRow], preview_dir: Path) -> str:
    cards: List[str] = []
    for r in rows:
        base = f"{r.rank:04d}_{_safe_slug(r.sample_id)}"
        full_name = f"{base}_full.jpg"
        zoom_name = f"{base}_zoom.jpg"
        full_rel = f"{preview_dir.name}/{full_name}"
        zoom_rel = f"{preview_dir.name}/{zoom_name}"

        item_key = f"{r.sample_id}||{r.image_path}"
        item_key_esc = html.escape(item_key, quote=True)
        sample_esc = html.escape(r.sample_id, quote=True)
        split_esc = html.escape(r.split, quote=True)
        cfg_esc = html.escape(r.config_folder, quote=True)
        source_esc = html.escape(r.source_name, quote=True)
        setup_esc = html.escape(r.setup_id, quote=True)
        fss_esc = html.escape(r.fss_path, quote=True)
        line21_esc = html.escape(r.scale_line_raw, quote=True)
        auto_tags_esc = html.escape(r.auto_tags, quote=True)
        image_path_esc = html.escape(r.image_path, quote=True)

        cards.append(
            "<section class='card'"
            f" data-item-key='{item_key_esc}'"
            f" data-rank='{r.rank}'"
            f" data-split='{split_esc}'"
            f" data-sample-id='{sample_esc}'"
            f" data-config-folder='{cfg_esc}'"
            f" data-source-name='{source_esc}'"
            f" data-setup-id='{setup_esc}'"
            f" data-depth-index='{r.depth_index}'"
            f" data-image-path='{image_path_esc}'"
            f" data-fss-path='{fss_esc}'"
            f" data-line21='{line21_esc}'"
            f" data-iou='{r.iou:.6f}'"
            f" data-score='{r.score:.6f}'"
            f" data-priority='{r.priority}'"
            f" data-xerr='{r.x_err_px:.6f}'"
            f" data-yerr='{r.y_err_px:.6f}'"
            f" data-dx='{r.dx_center_px:.6f}'"
            f" data-dy='{r.dy_center_px:.6f}'"
            f" data-werr='{r.w_err_px:.6f}'"
            f" data-herr='{r.h_err_px:.6f}'"
            f" data-auto-tags='{auto_tags_esc}'>"
            f"<h2>#{r.rank} [{r.priority}] {sample_esc} ({split_esc})</h2>"
            f"<p class='meta'>score={r.score:.1f} | IoU={r.iou:.3f} | x_err={r.x_err_px:.1f}px | y_err={r.y_err_px:.1f}px</p>"
            f"<p class='meta'>dx(center)={r.dx_center_px:+.1f}px | dy(center)={r.dy_center_px:+.1f}px | dw={r.w_err_px:+.1f}px | dh={r.h_err_px:+.1f}px</p>"
            f"<p class='meta'>config={cfg_esc} | setup={setup_esc} depth={r.depth_index} | source={source_esc}</p>"
            f"<p class='meta'>auto-tag: <code>{auto_tags_esc or 'none'}</code></p>"
            f"<p class='meta'>fss: <code>{fss_esc}</code></p>"
            f"<p class='meta'>line21: <code>{line21_esc}</code></p>"
            "<div class='actions'>"
            f"<label class='exclude-label'><input type='checkbox' class='exclude-toggle' data-item-key='{item_key_esc}' /> Escludi dal training</label>"
            f"<textarea class='note-input' data-item-key='{item_key_esc}' rows='2' placeholder='Nota libera su questo caso'></textarea>"
            "</div>"
            "<div class='issues'>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='x_destra' /> x troppo a destra</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='x_sinistra' /> x troppo a sinistra</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='y_alta' /> y troppo alta</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='y_bassa' /> y troppo bassa</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='troppo_larga' /> bbox troppo larga</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='troppo_stretta' /> bbox troppo stretta</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='troppo_alta' /> bbox troppo alta</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='troppo_bassa' /> bbox troppo bassa</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='gt_dubbia' /> GT dubbia</label>"
            "<label><input type='checkbox' class='issue-toggle' data-issue='altro' /> altro</label>"
            "</div>"
            "<div class='imgs'>"
            f"<a href='{full_rel}' target='_blank'><img class='preview-img' data-src='{full_rel}' alt='full overlay' /></a>"
            f"<a href='{zoom_rel}' target='_blank'><img class='preview-img' data-src='{zoom_rel}' alt='zoom overlay' /></a>"
            "</div>"
            "</section>"
        )
    return "".join(cards)


def write_html(path: Path, rows: Sequence[ReviewRow], preview_dir: Path) -> None:
    ui_version = "bk_study_v1_2026-04-22"
    storage_key = f"bk_scale_bbox_study_{_dataset_key(rows)}"
    cards = _cards_html(rows=rows, preview_dir=preview_dir)
    template = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BK Scale BBox Study (__UI_VERSION__)</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #0369a1;
      --warn: #b45309;
      --ok: #15803d;
      --excluded: #fff7ed;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.35;
    }
    .wrap {
      max-width: 1900px;
      margin: 0 auto;
      padding: 16px 18px 36px 18px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 30px;
    }
    .legend {
      margin: 0 0 14px 0;
      color: var(--muted);
      font-size: 16px;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 6;
      background: rgba(248, 250, 252, 0.97);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      margin-bottom: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .toolbar button {
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
    }
    .toolbar button:hover { background: #eff6ff; }
    .toolbar button.active {
      background: #dbeafe;
      border-color: #60a5fa;
      color: #1e3a8a;
      font-weight: 600;
    }
    .toolbar .stat {
      color: var(--muted);
      font-size: 14px;
      margin-right: 8px;
    }
    .toolbar .group {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 4px;
      background: #ffffff;
    }
    .toolbar .group-title {
      color: var(--muted);
      font-size: 12px;
      margin-right: 4px;
      padding-left: 2px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin: 0 0 18px 0;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      display: none;
    }
    .card.excluded {
      background: var(--excluded);
      border-color: var(--warn);
      box-shadow: 0 0 0 2px rgba(180, 83, 9, 0.15);
    }
    .card h2 {
      margin: 0 0 8px 0;
      font-size: 24px;
    }
    .meta {
      margin: 2px 0;
      color: var(--muted);
      font-size: 14px;
      word-break: break-word;
    }
    .actions {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 10px;
      align-items: start;
    }
    .exclude-label {
      font-size: 15px;
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .note-input {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      width: 100%;
      box-sizing: border-box;
      background: #ffffff;
      color: var(--text);
      resize: vertical;
    }
    .issues {
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(5, minmax(180px, 1fr));
      gap: 6px 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: #f8fafc;
    }
    .issues label {
      font-size: 13px;
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .imgs {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      margin-top: 12px;
    }
    img {
      width: 100%;
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      display: block;
      background: #000;
      min-height: 120px;
    }
    code {
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 5px;
      font-size: 12px;
    }
    .hidden {
      display: none !important;
    }
    @media (min-width: 1650px) {
      .imgs { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 1300px) {
      .issues { grid-template-columns: repeat(3, minmax(180px, 1fr)); }
    }
    @media (max-width: 900px) {
      .actions { grid-template-columns: 1fr; }
      .issues { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .toolbar { position: static; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>BK Scale BBox Study (__UI_VERSION__)</h1>
    <p class="legend">Overlay: verde=GT, magenta=prediction. Navigazione: frecce tastiera &larr; &rarr; (anche &uarr; &darr;). Una card alla volta.</p>
    <div class="toolbar">
      <span class="stat" id="statSummary">Righe: __TOTAL_ROWS__</span>
      <span class="stat" id="cardPos">Card: 0 / 0</span>
      <button id="btnPrevCard" type="button">Card precedente</button>
      <button id="btnNextCard" type="button">Card successiva</button>
      <button id="btnOneCard" type="button">Modalità 1 card</button>
      <button id="btnShowAllCards" type="button">Mostra tutte</button>
      <button id="btnExportNotesCsv" type="button">Esporta note CSV</button>
      <button id="btnExportExcludedCsv" type="button">Esporta esclusi CSV</button>
      <button id="btnClearState" type="button">Azzera note/flag</button>

      <div class="group">
        <span class="group-title">split</span>
        <button type="button" data-split-filter="all">all</button>
        <button type="button" data-split-filter="train">train</button>
        <button type="button" data-split-filter="val">val</button>
        <button type="button" data-split-filter="test">test</button>
      </div>
      <div class="group">
        <span class="group-title">errore</span>
        <button type="button" data-quality-filter="all">all</button>
        <button type="button" data-quality-filter="iou_lt_05">IoU&lt;0.50</button>
        <button type="button" data-quality-filter="iou_lt_03">IoU&lt;0.30</button>
        <button type="button" data-quality-filter="dx_abs_gt_20">|dx|&gt;20</button>
        <button type="button" data-quality-filter="dx_right_gt_20">dx&gt;20</button>
        <button type="button" data-quality-filter="dx_left_lt_20">dx&lt;-20</button>
        <button type="button" data-quality-filter="prio_p0p1">P0/P1</button>
      </div>
      <div class="group">
        <span class="group-title">annotazioni</span>
        <button type="button" id="btnShowAllAnno">tutte</button>
        <button type="button" id="btnShowAnnotated">solo annotate</button>
      </div>
    </div>

__CARDS__
  </div>
  <script>
    (function() {
      try {
        const STORAGE_KEY = "__STORAGE_KEY__";
        const ISSUE_KEYS = [
          "x_destra",
          "x_sinistra",
          "y_alta",
          "y_bassa",
          "troppo_larga",
          "troppo_stretta",
          "troppo_alta",
          "troppo_bassa",
          "gt_dubbia",
          "altro"
        ];

        const cards = Array.from(document.querySelectorAll(".card"));
        const statSummary = document.getElementById("statSummary");
        const cardPos = document.getElementById("cardPos");
        const btnPrevCard = document.getElementById("btnPrevCard");
        const btnNextCard = document.getElementById("btnNextCard");
        const btnOneCard = document.getElementById("btnOneCard");
        const btnShowAllCards = document.getElementById("btnShowAllCards");
        const btnExportNotesCsv = document.getElementById("btnExportNotesCsv");
        const btnExportExcludedCsv = document.getElementById("btnExportExcludedCsv");
        const btnClearState = document.getElementById("btnClearState");
        const btnShowAllAnno = document.getElementById("btnShowAllAnno");
        const btnShowAnnotated = document.getElementById("btnShowAnnotated");
        const splitFilterButtons = Array.from(document.querySelectorAll("button[data-split-filter]"));
        const qualityFilterButtons = Array.from(document.querySelectorAll("button[data-quality-filter]"));

        let splitFilter = "all";
        let qualityFilter = "all";
        let onlyAnnotated = false;
        let oneCardMode = true;
        let currentCardIdx = 0;

        function mergeObjects(a, b) {
          const out = {};
          let k = "";
          for (k in a) {
            if (Object.prototype.hasOwnProperty.call(a, k)) out[k] = a[k];
          }
          for (k in b) {
            if (Object.prototype.hasOwnProperty.call(b, k)) out[k] = b[k];
          }
          return out;
        }

        function loadState() {
          try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return {};
            const obj = JSON.parse(raw);
            if (obj && typeof obj === "object") return obj;
          } catch (e) {}
          return {};
        }

        function saveState(state) {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        }

        function parseNumber(value) {
          if (value === undefined || value === null) return NaN;
          const n = Number(String(value).trim());
          return Number.isFinite(n) ? n : NaN;
        }

        function csvEscape(value) {
          const s = String(value === undefined || value === null ? "" : value);
          if (s.indexOf(",") >= 0 || s.indexOf('"') >= 0 || s.indexOf("\\n") >= 0) {
            return '"' + s.split('"').join('""') + '"';
          }
          return s;
        }

        function ensureCardImages(card) {
          if (!card) return;
          const imgs = card.querySelectorAll("img.preview-img[data-src]");
          imgs.forEach((img) => {
            if (!img.getAttribute("src")) {
              img.setAttribute("src", img.getAttribute("data-src"));
            }
          });
        }

        function isEditableTarget(el) {
          if (!el) return false;
          const tag = String(el.tagName || "").toUpperCase();
          if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
          if (el.isContentEditable) return true;
          const p = el.closest ? el.closest("input, textarea, select, [contenteditable='true']") : null;
          return !!p;
        }

        const state = loadState();

        function hasAnyIssue(saved) {
          if (!saved || typeof saved !== "object") return false;
          const issues = saved.issues || {};
          return ISSUE_KEYS.some((k) => !!issues[k]);
        }

        function hasAnyAnnotation(saved) {
          if (!saved || typeof saved !== "object") return false;
          const note = String(saved.note || "").trim();
          return !!saved.exclude || note.length > 0 || hasAnyIssue(saved);
        }

        function cardPassesFilters(card) {
          const split = String(card.dataset.split || "");
          if (splitFilter !== "all" && split !== splitFilter) return false;

          const iou = parseNumber(card.dataset.iou);
          const dx = parseNumber(card.dataset.dx);
          const prio = String(card.dataset.priority || "");

          if (qualityFilter === "iou_lt_05" && !(iou < 0.50)) return false;
          if (qualityFilter === "iou_lt_03" && !(iou < 0.30)) return false;
          if (qualityFilter === "dx_abs_gt_20" && !(Math.abs(dx) > 20.0)) return false;
          if (qualityFilter === "dx_right_gt_20" && !(dx > 20.0)) return false;
          if (qualityFilter === "dx_left_lt_20" && !(dx < -20.0)) return false;
          if (qualityFilter === "prio_p0p1" && !(prio === "P0" || prio === "P1")) return false;

          if (onlyAnnotated) {
            const key = String(card.dataset.itemKey || "");
            const saved = state[key] || {};
            if (!hasAnyAnnotation(saved)) return false;
          }
          return true;
        }

        function applyCardState(card) {
          const key = String(card.dataset.itemKey || "");
          const saved = state[key] || {};
          const excluded = !!saved.exclude;
          const note = String(saved.note || "");
          const issues = saved.issues || {};

          const toggle = card.querySelector(".exclude-toggle");
          const noteInput = card.querySelector(".note-input");
          if (toggle) toggle.checked = excluded;
          if (noteInput) noteInput.value = note;

          const issueChecks = card.querySelectorAll(".issue-toggle");
          issueChecks.forEach((el) => {
            const issue = String(el.getAttribute("data-issue") || "");
            el.checked = !!issues[issue];
          });

          card.classList.toggle("excluded", excluded);
          card.dataset.filterVisible = cardPassesFilters(card) ? "1" : "0";
        }

        function filteredCards() {
          return cards.filter((card) => card.dataset.filterVisible === "1");
        }

        function renderCards(shouldScroll) {
          cards.forEach((card) => {
            card.style.display = "none";
          });

          const visible = filteredCards();
          if (!visible.length) {
            cardPos.textContent = "Card: 0 / 0";
            return;
          }

          if (oneCardMode) {
            if (currentCardIdx < 0) currentCardIdx = 0;
            if (currentCardIdx >= visible.length) currentCardIdx = visible.length - 1;
            const target = visible[currentCardIdx];
            target.style.display = "block";
            ensureCardImages(target);
            const pre = visible[Math.max(0, currentCardIdx - 1)];
            const nxt = visible[Math.min(visible.length - 1, currentCardIdx + 1)];
            ensureCardImages(pre);
            ensureCardImages(nxt);
            cardPos.textContent = "Card: " + (currentCardIdx + 1) + " / " + visible.length;
            if (shouldScroll) {
              target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          } else {
            visible.forEach((card) => {
              card.style.display = "block";
              ensureCardImages(card);
            });
            cardPos.textContent = "Card visibili: " + visible.length;
          }
        }

        function refresh(shouldScroll) {
          cards.forEach(applyCardState);
          const visible = filteredCards();
          if (currentCardIdx >= visible.length) {
            currentCardIdx = Math.max(0, visible.length - 1);
          }
          renderCards(!!shouldScroll);

          const excludedCount = cards.filter((card) => {
            const key = String(card.dataset.itemKey || "");
            const saved = state[key] || {};
            return !!saved.exclude;
          }).length;

          const annotatedCount = cards.filter((card) => {
            const key = String(card.dataset.itemKey || "");
            const saved = state[key] || {};
            return hasAnyAnnotation(saved);
          }).length;

          statSummary.textContent =
            "Righe totali: " + cards.length +
            " | Filtrate: " + visible.length +
            " | Escluse: " + excludedCount +
            " | Annotate: " + annotatedCount;
        }

        function stepCard(direction) {
          const visible = filteredCards();
          if (!visible.length || !oneCardMode) return;
          const nxt = Math.max(0, Math.min(visible.length - 1, currentCardIdx + direction));
          if (nxt === currentCardIdx) return;
          currentCardIdx = nxt;
          renderCards(true);
        }

        function updateFilterButtons() {
          splitFilterButtons.forEach((btn) => {
            const val = String(btn.getAttribute("data-split-filter") || "all");
            btn.classList.toggle("active", val === splitFilter);
          });
          qualityFilterButtons.forEach((btn) => {
            const val = String(btn.getAttribute("data-quality-filter") || "all");
            btn.classList.toggle("active", val === qualityFilter);
          });
        }

        cards.forEach((card) => {
          const key = String(card.dataset.itemKey || "");
          const toggle = card.querySelector(".exclude-toggle");
          const noteInput = card.querySelector(".note-input");
          const issueChecks = card.querySelectorAll(".issue-toggle");

          function saveCardState() {
            const current = state[key] || {};
            const nextIssues = {};
            issueChecks.forEach((el) => {
              const issue = String(el.getAttribute("data-issue") || "");
              nextIssues[issue] = !!el.checked;
            });
            state[key] = mergeObjects(current, {
              exclude: !!(toggle && toggle.checked),
              note: noteInput ? noteInput.value || "" : "",
              issues: nextIssues,
              updated_at: new Date().toISOString()
            });
            saveState(state);
            refresh(false);
          }

          if (toggle) toggle.addEventListener("change", saveCardState);
          if (noteInput) noteInput.addEventListener("change", saveCardState);
          issueChecks.forEach((el) => el.addEventListener("change", saveCardState));
        });

        splitFilterButtons.forEach((btn) => {
          btn.addEventListener("click", () => {
            splitFilter = String(btn.getAttribute("data-split-filter") || "all");
            currentCardIdx = 0;
            updateFilterButtons();
            refresh(false);
          });
        });
        qualityFilterButtons.forEach((btn) => {
          btn.addEventListener("click", () => {
            qualityFilter = String(btn.getAttribute("data-quality-filter") || "all");
            currentCardIdx = 0;
            updateFilterButtons();
            refresh(false);
          });
        });

        btnShowAllAnno.addEventListener("click", () => {
          onlyAnnotated = false;
          currentCardIdx = 0;
          refresh(false);
        });
        btnShowAnnotated.addEventListener("click", () => {
          onlyAnnotated = true;
          currentCardIdx = 0;
          refresh(false);
        });

        btnPrevCard.addEventListener("click", () => stepCard(-1));
        btnNextCard.addEventListener("click", () => stepCard(1));
        btnOneCard.addEventListener("click", () => {
          oneCardMode = true;
          refresh(false);
        });
        btnShowAllCards.addEventListener("click", () => {
          oneCardMode = false;
          refresh(false);
        });

        function rowsForExport(onlyExcluded) {
          return cards.map((card) => {
            const key = String(card.dataset.itemKey || "");
            const saved = state[key] || {};
            const issues = saved.issues || {};
            const obj = {
              rank: card.dataset.rank || "",
              split: card.dataset.split || "",
              priority: card.dataset.priority || "",
              sample_id: card.dataset.sampleId || "",
              config_folder: card.dataset.configFolder || "",
              setup_id: card.dataset.setupId || "",
              depth_index: card.dataset.depthIndex || "",
              source_name: card.dataset.sourceName || "",
              image_path: card.dataset.imagePath || "",
              fss_path: card.dataset.fssPath || "",
              line21: card.dataset.line21 || "",
              score: card.dataset.score || "",
              iou: card.dataset.iou || "",
              x_err_px: card.dataset.xerr || "",
              y_err_px: card.dataset.yerr || "",
              dx_center_px: card.dataset.dx || "",
              dy_center_px: card.dataset.dy || "",
              w_err_px: card.dataset.werr || "",
              h_err_px: card.dataset.herr || "",
              auto_tags: card.dataset.autoTags || "",
              exclude: saved.exclude ? "1" : "0",
              note: saved.note || "",
              issue_x_destra: issues.x_destra ? "1" : "0",
              issue_x_sinistra: issues.x_sinistra ? "1" : "0",
              issue_y_alta: issues.y_alta ? "1" : "0",
              issue_y_bassa: issues.y_bassa ? "1" : "0",
              issue_troppo_larga: issues.troppo_larga ? "1" : "0",
              issue_troppo_stretta: issues.troppo_stretta ? "1" : "0",
              issue_troppo_alta: issues.troppo_alta ? "1" : "0",
              issue_troppo_bassa: issues.troppo_bassa ? "1" : "0",
              issue_gt_dubbia: issues.gt_dubbia ? "1" : "0",
              issue_altro: issues.altro ? "1" : "0"
            };
            const keep = onlyExcluded ? (obj.exclude === "1") : hasAnyAnnotation(saved);
            return keep ? obj : null;
          }).filter(Boolean);
        }

        function downloadRowsAsCsv(rows, filenamePrefix) {
          const header = [
            "rank","split","priority","sample_id","config_folder","setup_id","depth_index","source_name",
            "image_path","fss_path","line21","score","iou","x_err_px","y_err_px",
            "dx_center_px","dy_center_px","w_err_px","h_err_px","auto_tags","exclude","note",
            "issue_x_destra","issue_x_sinistra","issue_y_alta","issue_y_bassa",
            "issue_troppo_larga","issue_troppo_stretta","issue_troppo_alta","issue_troppo_bassa",
            "issue_gt_dubbia","issue_altro"
          ];
          const lines = [header.join(",")];
          rows.forEach((row) => {
            lines.push(header.map((h) => csvEscape(row[h])).join(","));
          });
          const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv;charset=utf-8;" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          const ts = new Date().toISOString().replace(/[:.]/g, "-");
          a.href = url;
          a.download = filenamePrefix + "_" + ts + ".csv";
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        }

        btnExportNotesCsv.addEventListener("click", () => {
          const rows = rowsForExport(false);
          downloadRowsAsCsv(rows, "bk_scale_bbox_notes");
        });

        btnExportExcludedCsv.addEventListener("click", () => {
          const rows = rowsForExport(true);
          downloadRowsAsCsv(rows, "bk_scale_bbox_excluded");
        });

        btnClearState.addEventListener("click", () => {
          if (!confirm("Azzero tutte le note, issue e flag esclusione per questa pagina?")) return;
          cards.forEach((card) => {
            const key = String(card.dataset.itemKey || "");
            state[key] = {
              exclude: false,
              note: "",
              issues: {},
              updated_at: new Date().toISOString()
            };
          });
          saveState(state);
          refresh(false);
        });

        function handleArrowNav(event) {
          if (!oneCardMode) return;
          if (isEditableTarget(event.target)) return;
          if (event.key === "ArrowRight" || event.key === "ArrowDown") {
            event.preventDefault();
            stepCard(1);
            return;
          }
          if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
            event.preventDefault();
            stepCard(-1);
          }
        }
        window.addEventListener("keydown", handleArrowNav, true);
        document.addEventListener("keydown", handleArrowNav);

        if (document.body) {
          document.body.setAttribute("tabindex", "-1");
        }
        document.addEventListener("click", (event) => {
          if (isEditableTarget(event.target)) return;
          if (document.body) {
            try { document.body.focus(); } catch (e) {}
          }
        });

        updateFilterButtons();
        refresh(false);
        console.log("BK study UI loaded:", "__UI_VERSION__", "storage=", STORAGE_KEY);
      } catch (e) {
        console.error("BK study UI fatal error", e);
        alert("Errore nella pagina di review: " + (e && e.message ? e.message : e));
      }
    })();
  </script>
</body>
</html>
"""

    content = (
        template.replace("__UI_VERSION__", ui_version)
        .replace("__TOTAL_ROWS__", str(len(rows)))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__CARDS__", cards)
    )
    path.write_text(content, encoding="utf-8")


def write_rows_csv(path: Path, rows: Sequence[ReviewRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "split",
                "priority",
                "sample_id",
                "config_folder",
                "setup_id",
                "depth_index",
                "source_name",
                "image_path",
                "fss_path",
                "line21",
                "score",
                "iou",
                "x_err_px",
                "y_err_px",
                "dx_center_px",
                "dy_center_px",
                "w_err_px",
                "h_err_px",
                "auto_tags",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rank,
                    r.split,
                    r.priority,
                    r.sample_id,
                    r.config_folder,
                    r.setup_id,
                    r.depth_index,
                    r.source_name,
                    r.image_path,
                    r.fss_path,
                    r.scale_line_raw,
                    f"{r.score:.6f}",
                    f"{r.iou:.6f}",
                    f"{r.x_err_px:.6f}",
                    f"{r.y_err_px:.6f}",
                    f"{r.dx_center_px:.6f}",
                    f"{r.dy_center_px:.6f}",
                    f"{r.w_err_px:.6f}",
                    f"{r.h_err_px:.6f}",
                    r.auto_tags,
                ]
            )


def summarize(rows: Sequence[ReviewRow]) -> Dict[str, object]:
    def mean(vals: Iterable[float]) -> float:
        vals = list(vals)
        return float(sum(vals) / len(vals)) if vals else 0.0

    by_split: Dict[str, int] = {}
    by_prio: Dict[str, int] = {}
    for r in rows:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_prio[r.priority] = by_prio.get(r.priority, 0) + 1

    out = {
        "rows": len(rows),
        "by_split": by_split,
        "by_priority": by_prio,
        "mean_iou": mean(r.iou for r in rows),
        "mean_x_err_px": mean(r.x_err_px for r in rows),
        "mean_y_err_px": mean(r.y_err_px for r in rows),
        "mean_dx_center_px": mean(r.dx_center_px for r in rows),
        "mean_dy_center_px": mean(r.dy_center_px for r in rows),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build interactive BK scale bbox study HTML.")
    p.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/train_predictions_best.csv"),
            Path("artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/val_predictions_best.csv"),
            Path("artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/test_predictions_best.csv"),
        ],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/study_interactive"),
    )
    p.add_argument("--top-n", type=int, default=260, help="Max ranked rows rendered in HTML.")
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma list among train,val,test (default all).",
    )
    p.add_argument("--dedupe-image-path", action="store_true", help="Keep one row per image path (worst score wins).")
    p.add_argument("--enrich-fss", action="store_true", help="Resolve FSS path + line21 for each selected row.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    split_tokens = [s.strip().lower() for s in str(args.splits).split(",") if s.strip()]
    split_filter = set(s for s in split_tokens if s in {"train", "val", "test"})
    if not split_filter:
        split_filter = None

    pred_paths = [p.expanduser().resolve() for p in args.predictions]
    rows = load_rows(
        predictions=pred_paths,
        split_filter=split_filter,
        dedupe_image_path=bool(args.dedupe_image_path),
    )
    if not rows:
        raise RuntimeError("No valid prediction rows loaded from provided CSV paths.")

    rows.sort(key=lambda r: r.score, reverse=True)
    if args.top_n > 0:
        rows = rows[: int(args.top_n)]
    for i, r in enumerate(rows, start=1):
        r.rank = i

    if args.enrich_fss:
        enrich_rows_with_fss(rows)

    for r in rows:
        base = f"{r.rank:04d}_{_safe_slug(r.sample_id)}"
        full_out = preview_dir / f"{base}_full.jpg"
        zoom_out = preview_dir / f"{base}_zoom.jpg"
        try:
            draw_overlay_previews(r, full_out=full_out, zoom_out=zoom_out)
        except Exception:
            # keep generation robust: card still appears with metadata
            continue

    html_path = out_dir / "bk_scale_bbox_study.html"
    csv_path = out_dir / "bk_scale_bbox_study_rows.csv"
    summary_path = out_dir / "summary.json"
    write_html(path=html_path, rows=rows, preview_dir=preview_dir)
    write_rows_csv(path=csv_path, rows=rows)
    summary = {
        "predictions": [p.as_posix() for p in pred_paths],
        "splits": sorted(list(split_filter)) if split_filter else ["train", "val", "test"],
        "top_n": int(args.top_n),
        "dedupe_image_path": bool(args.dedupe_image_path),
        "enrich_fss": bool(args.enrich_fss),
        "stats": summarize(rows),
        "outputs": {
            "html": html_path.as_posix(),
            "rows_csv": csv_path.as_posix(),
            "preview_dir": preview_dir.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
