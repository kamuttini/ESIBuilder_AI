#!/usr/bin/env python3
"""Build GT-only orientation debug HTML for scale line investigation.

Goals:
- show only GT overlay (no predictions),
- estimate SU/GIU orientation from sibling frames (flip_ud vs no_flip/value),
- allow toggling to opposite-orientation image from same folder,
- summarize correlation with label_side.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class ManifestLine:
    sample_id: str
    split: str
    config_folder: str
    setup_id: str
    depth_index: int
    x_gt_video: float
    y_top_gt_video: float
    y_bottom_gt_video: float
    video_w: float
    video_h: float
    label_side: int


@dataclass(frozen=True)
class InputRow:
    sample_id: str
    split: str
    image_path: str
    score_after: float
    orientation_name: str
    orientation_idx: int
    orientation_source_type: str


@dataclass(frozen=True)
class ManifestMatch:
    gt: ManifestLine
    source: str


@dataclass(frozen=True)
class OppositeInfo:
    current_path: Path
    opposite_path: Optional[Path]
    su_giu_pred: str
    basis: str
    mse_to_ud: float
    mse_to_nf: float
    confidence: float
    source_ud: Optional[Path]
    source_nf: Optional[Path]


def _f(v: str) -> float:
    return float(str(v).strip())


def _i(v: str) -> int:
    return int(float(str(v).strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))[:120]


def _exists_image(p: Path) -> bool:
    return p.exists() and p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")


def _replace_stem_with_idx(stem_base: str, idx0: int, parent: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        c = parent / f"{stem_base}{idx0}{ext}"
        if _exists_image(c):
            return c
    return None


def _load_gray(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:
        return None


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b_img = Image.fromarray(np.clip(b, 0.0, 255.0).astype(np.uint8), mode="L")
        b_img = b_img.resize((a.shape[1], a.shape[0]), resample=Image.Resampling.BILINEAR)
        b = np.asarray(b_img, dtype=np.float32)
    d = a - b
    return float(np.mean(d * d))


def _depth_idx0_from_sample(sample_id: str) -> Optional[int]:
    m = re.match(r"^.+_([0-9]+)$", sample_id.strip())
    if not m:
        return None
    try:
        return max(0, int(m.group(1)) - 1)
    except Exception:
        return None


def _parse_sample_id(sample_id: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)_([0-9]+)$", sample_id.strip())
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def _normalize_name(text: str) -> str:
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _config_from_image_path(image_path: Path) -> str:
    # .../Dataset/<config>/image_samples/<image>
    parts = list(image_path.parts)
    for needle in ("Dataset", "dataset", "SSD_esi1_n1"):
        if needle in parts:
            i = parts.index(needle)
            if i + 1 < len(parts):
                return parts[i + 1]
    if len(parts) >= 3:
        return parts[-3]
    return ""


def _remap_to_preferred_root(image_path: Path, preferred_root: Optional[Path]) -> Path:
    if preferred_root is None:
        return image_path
    if str(image_path).startswith(str(preferred_root)):
        return image_path
    cfg = _config_from_image_path(image_path)
    if not cfg:
        return image_path
    rel_parts = image_path.parts[-2:] if len(image_path.parts) >= 2 else (image_path.name,)
    candidate = preferred_root / cfg
    for part in rel_parts:
        candidate = candidate / part
    if candidate.exists():
        return candidate
    # Fallback robusto: stesso nome file in image_samples sotto config.
    fallback = preferred_root / cfg / "image_samples" / image_path.name
    if fallback.exists():
        return fallback
    return image_path


def _find_candidates_same_folder(current: Path, sample_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (ud_candidate, nf_candidate) for same setup index in same folder.

    ud_candidate  => image_depth_find_flip_ud_setup_{idx0}
    nf_candidate  => one of image_depth_find_no_flip_setup_{idx0}, image_depth_value_setup_{idx0}, image_biplana_setup_{idx0}
    """
    parent = current.parent
    idx0 = _depth_idx0_from_sample(sample_id)
    if idx0 is None:
        idx0 = 0

    ud = _replace_stem_with_idx("image_depth_find_flip_ud_setup_", idx0, parent)

    nf = (
        _replace_stem_with_idx("image_depth_find_no_flip_setup_", idx0, parent)
        or _replace_stem_with_idx("image_depth_value_setup_", idx0, parent)
        or _replace_stem_with_idx("image_biplana_setup_", idx0, parent)
    )

    # If current already is one candidate type, keep it as valid side for comparison.
    return ud, nf


def find_opposite_info(current: Path, sample_id: str) -> OppositeInfo:
    ud, nf = _find_candidates_same_folder(current=current, sample_id=sample_id)
    cur_gray = _load_gray(current)

    mse_ud = float("nan")
    mse_nf = float("nan")
    basis = "none"
    pred = "unknown"
    conf = 0.0
    opposite: Optional[Path] = None

    stem_l = current.stem.lower()

    if cur_gray is not None:
        if ud is not None:
            ud_gray = _load_gray(ud)
            if ud_gray is not None:
                mse_ud = _mse(cur_gray, ud_gray)
        if nf is not None:
            nf_gray = _load_gray(nf)
            if nf_gray is not None:
                mse_nf = _mse(cur_gray, nf_gray)

    has_ud = math.isfinite(mse_ud)
    has_nf = math.isfinite(mse_nf)

    if has_ud and has_nf:
        basis = "mse_ud_vs_nf"
        denom = max(1e-6, max(mse_ud, mse_nf))
        conf = float(abs(mse_ud - mse_nf) / denom)
        if mse_ud < mse_nf:
            pred = "giu"
            opposite = nf
        else:
            pred = "su"
            opposite = ud
    elif "flip_ud" in stem_l:
        pred = "giu"
        basis = "filename_hint_flip_ud"
        opposite = nf
    elif "no_flip" in stem_l or "depth_value" in stem_l or "biplana" in stem_l:
        pred = "su"
        basis = "filename_hint_no_flip_like"
        opposite = ud
    else:
        pred = "unknown"
        basis = "no_hint"
        opposite = ud or nf

    if opposite is not None and not _exists_image(opposite):
        opposite = None

    return OppositeInfo(
        current_path=current,
        opposite_path=opposite,
        su_giu_pred=pred,
        basis=basis,
        mse_to_ud=mse_ud,
        mse_to_nf=mse_nf,
        confidence=conf,
        source_ud=ud,
        source_nf=nf,
    )


def _find_opposite_by_official_orientation(
    current: Path,
    sample_id: str,
    orientation_name: str,
) -> Optional[Path]:
    """Find opposite orientation image in the same folder using official 4-way orientation.

    Pairing used:
    - NF <-> UD
    - LR <-> LRUD
    """
    ori = str(orientation_name or "").strip().upper()
    if ori not in {"NF", "UD", "LR", "LRUD"}:
        return None

    opposite_of = {"NF": "UD", "UD": "NF", "LR": "LRUD", "LRUD": "LR"}
    target = opposite_of.get(ori, "")
    if not target:
        return None

    idx0 = _depth_idx0_from_sample(sample_id)
    if idx0 is None:
        idx0 = 0
    parent = current.parent

    stem_by_target: Dict[str, Tuple[str, ...]] = {
        "NF": (
            "image_depth_find_no_flip_setup_",
            "image_depth_value_setup_",
            "image_biplana_setup_",
        ),
        "UD": ("image_depth_find_flip_ud_setup_",),
        "LR": ("image_depth_find_flip_lr_setup_",),
        "LRUD": ("image_depth_find_flip_lrud_setup_",),
    }
    stems = stem_by_target.get(target, ())
    for stem in stems:
        p = _replace_stem_with_idx(stem, idx0, parent)
        if p is None:
            continue
        try:
            if p.resolve() == current.resolve():
                continue
        except Exception:
            if p.as_posix() == current.as_posix():
                continue
        return p
    return None


def load_manifest(manifest_csv: Path) -> Dict[str, List[ManifestLine]]:
    out: Dict[str, List[ManifestLine]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            sid = str(r.get("sample_id", "")).strip()
            if not sid:
                continue
            try:
                x1 = _f(r["x1"])
                x2 = _f(r["x2"])
                y1 = _f(r["y1"])
                y2 = _f(r["y2"])
                gt = ManifestLine(
                    sample_id=sid,
                    split=str(r.get("split", "")).strip().lower(),
                    config_folder=str(r.get("config_folder", "")).strip(),
                    setup_id=str(r.get("setup_id", "")).strip(),
                    depth_index=_i(r.get("depth_index", "0")),
                    x_gt_video=0.5 * (x1 + x2),
                    y_top_gt_video=min(y1, y2),
                    y_bottom_gt_video=max(y1, y2),
                    video_w=_f(r["video_x_size"]),
                    video_h=_f(r["video_y_size"]),
                    label_side=-1 if _i(r.get("label_side", "-1")) < 0 else 1,
                )
                out.setdefault(sid, []).append(gt)
            except Exception:
                continue
    return out


def select_manifest_line(
    manifest_by_sample: Dict[str, List[ManifestLine]],
    sample_id: str,
    image_path: Path,
) -> Optional[ManifestMatch]:
    cands = manifest_by_sample.get(sample_id, [])
    if not cands:
        return None
    if len(cands) == 1:
        return ManifestMatch(gt=cands[0], source="sample_id_unique")

    setup_id, depth_idx = _parse_sample_id(sample_id)
    image_cfg = _config_from_image_path(image_path)
    image_cfg_n = _normalize_name(image_cfg)

    filtered = [c for c in cands if c.setup_id == setup_id and c.depth_index == depth_idx]
    if not filtered:
        filtered = cands

    # 1) exact config folder
    for c in filtered:
        if c.config_folder == image_cfg:
            return ManifestMatch(gt=c, source="exact_config")

    # 2) normalized config folder
    for c in filtered:
        if _normalize_name(c.config_folder) == image_cfg_n and image_cfg_n:
            return ManifestMatch(gt=c, source="normalized_config")

    # 3) fallback first filtered
    return ManifestMatch(gt=filtered[0], source="fallback_first")


def load_rows(rows_csv: Path, max_rows: int) -> List[InputRow]:
    rows: List[InputRow] = []
    with rows_csv.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                sid = str(r.get("sample_id", "")).strip()
                img = str(r.get("image_path", "")).strip()
                split = str(r.get("split", "")).strip().lower()
                if not sid or not img:
                    continue
                sc = float(r.get("score_after", "nan"))
                ori_name = str(r.get("orientation_name", "")).strip().upper()
                ori_src = str(r.get("orientation_source_type", "")).strip()
                try:
                    ori_idx_raw = str(r.get("orientation_idx", "")).strip()
                    ori_idx = int(ori_idx_raw) if ori_idx_raw != "" else -1
                except Exception:
                    ori_idx = -1
            except Exception:
                continue
            rows.append(
                InputRow(
                    sample_id=sid,
                    split=split,
                    image_path=img,
                    score_after=sc,
                    orientation_name=ori_name,
                    orientation_idx=ori_idx,
                    orientation_source_type=ori_src,
                )
            )

    # Sort by score_after descending when available, then dedupe by image path.
    rows.sort(key=lambda x: (0.0 if math.isnan(x.score_after) else x.score_after), reverse=True)
    dedup: Dict[str, InputRow] = {}
    for r in rows:
        if r.image_path not in dedup:
            dedup[r.image_path] = r
    rows = list(dedup.values())

    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def draw_gt_overlay(
    src: Path,
    out_path: Path,
    gt: ManifestLine,
    label: str,
    draw_overlay: bool = True,
) -> Tuple[int, int, float, float, float]:
    with Image.open(src) as im:
        img = im.convert("RGB")
    w, h = img.size

    sx = float(w) / float(gt.video_w) if gt.video_w > 0 else 1.0
    sy = float(h) / float(gt.video_h) if gt.video_h > 0 else 1.0

    x = float(gt.x_gt_video) * sx
    y1 = float(gt.y_top_gt_video) * sy
    y2 = float(gt.y_bottom_gt_video) * sy

    if draw_overlay:
        dr = ImageDraw.Draw(img, "RGBA")
        # GT only: thick dark support + green line.
        dr.line((x, y1, x, y2), fill=(0, 0, 0, 230), width=10)
        dr.line((x, y1, x, y2), fill=(16, 185, 129, 255), width=6)

        rr = 6
        dr.ellipse((x - rr, y1 - rr, x + rr, y1 + rr), fill=(16, 185, 129, 255), outline=(255, 255, 255, 255))
        dr.ellipse((x - rr, y2 - rr, x + rr, y2 + rr), fill=(16, 185, 129, 255), outline=(255, 255, 255, 255))

        # small label in top-left
        txt = label
        dr.rectangle((8, 8, min(w - 8, 8 + 8 * max(20, len(txt))), 34), fill=(0, 0, 0, 170))
        dr.text((14, 14), txt, fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)
    return w, h, x, y1, y2


def build_html(output_html: Path, rows_payload: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    payload_json = json.dumps(rows_payload, ensure_ascii=False, indent=2).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2).replace("</", "<\\/")
    def hesc(v: object) -> str:
        s = str(v)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    fallback_cards: List[str] = []
    for i, r in enumerate(rows_payload, start=1):
        sample_id = hesc(r.get("sample_id", ""))
        split = hesc(r.get("split", ""))
        label_side = hesc(r.get("label_side", ""))
        su_giu = hesc(r.get("su_giu_pred", ""))
        ori_name = hesc(r.get("orientation_name", "") or "UNKNOWN")
        src = hesc(r.get("orig_preview_rel", ""))
        fallback_cards.append(
            f"""
          <section class=\"card\" style=\"display:block\" data-idx=\"{i - 1}\">
            <h2>#{i} {sample_id} ({split})</h2>
            <p class=\"meta\"><b>Orientamento ufficiale:</b> <b>{ori_name}</b></p>
            <p class=\"meta\"><b>label_side:</b> {label_side} | <b>SU/GIU stimato:</b> <b>{su_giu}</b></p>
            <div class=\"img-wrap\">
              <img class=\"gt-img\" src=\"{src}\" alt=\"gt preview\" />
            </div>
          </section>
          """
        )
    fallback_cards_html = "\n".join(fallback_cards)

    html_text = f"""<!doctype html>
<html lang=\"it\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Scale GT Debug (solo GT + SU/GIU)</title>
  <style>
    :root {{
      --bg:#f8fafc; --panel:#ffffff; --ink:#0f172a; --muted:#475569; --line:#cbd5e1; --accent:#0284c7;
      --ok:#15803d; --warn:#b45309;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif; }}
    .wrap {{ max-width:1900px; margin:0 auto; padding:16px; }}
    h1 {{ margin:0 0 6px; font-size:30px; }}
    .legend {{ color:var(--muted); margin:0 0 12px; }}
    .toolbar {{ position:sticky; top:0; z-index:7; background:rgba(248,250,252,0.97); border:1px solid var(--line); border-radius:10px; padding:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:14px; }}
    .toolbar button {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:7px 10px; border-radius:8px; cursor:pointer; }}
    .toolbar button:hover {{ background:#eff6ff; }}
    .stat {{ color:var(--muted); font-size:14px; margin-right:8px; }}
    .chip {{ border:1px solid var(--line); border-radius:999px; padding:4px 8px; background:#fff; font-size:12px; color:#334155; }}

    .card {{ display:none; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
    .card h2 {{ margin:0 0 6px; font-size:24px; }}
    .meta {{ margin:3px 0; color:var(--muted); font-size:14px; word-break:break-word; }}
    .meta b {{ color:var(--ink); }}
    .meta code {{ background:#f1f5f9; border-radius:5px; padding:1px 5px; }}
    .img-actions {{ margin:10px 0; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .img-actions button {{ border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:8px; padding:7px 10px; cursor:pointer; }}
    .img-actions button.active {{ background:#dbeafe; border-color:#60a5fa; color:#1e3a8a; font-weight:600; }}
    .img-actions .btn-exclude {{ border-color:#fca5a5; color:#991b1b; background:#fff1f2; }}
    .img-actions .btn-exclude.active {{ border-color:#ef4444; background:#fee2e2; color:#7f1d1d; font-weight:700; }}
    .img-actions .btn-shift {{ min-width:64px; }}
    .img-actions .warn {{ color:var(--warn); font-size:13px; }}
    .excluded-note {{ color:#b91c1c; font-size:13px; font-weight:700; }}
    .shift-badge {{ font-size:12px; color:#1f2937; border:1px dashed #94a3b8; border-radius:999px; padding:4px 9px; background:#f8fafc; }}

    .card.excluded {{ border-color:#fca5a5; background:#fff1f2; }}
    .card.excluded .img-wrap img {{ opacity:0.42; filter:grayscale(65%); }}

    .img-wrap {{ margin-top:8px; }}
    .img-stage {{ position:relative; width:100%; }}
    .img-wrap img {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; background:#000; }}
    .scale-overlay {{ position:absolute; inset:0; pointer-events:none; }}
    .scale-line {{ position:absolute; top:0; height:0; width:2px; margin-left:-1px; background:#22c55e; box-shadow:0 0 0 1px rgba(0,0,0,0.55); }}
    .scale-dot {{ position:absolute; width:8px; height:8px; margin-left:-4px; margin-top:-4px; border-radius:999px; background:#22c55e; border:1px solid #fff; box-shadow:0 0 0 1px rgba(0,0,0,0.45); }}

    .summary-box {{ border:1px solid var(--line); border-radius:10px; background:#fff; padding:10px; margin:0 0 14px; }}
    .summary-box pre {{ margin:0; white-space:pre-wrap; word-break:break-word; color:#1e293b; font-size:13px; }}

    @media (max-width: 900px) {{ .toolbar {{ position:static; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Scale GT Debug (solo GT)</h1>
    <p class=\"legend\">Linea scala dinamica: spostala in X per ispezionare i dettagli sotto la GT. Frecce ←/→ per card, ↑/↓ per originale/opposta, X per escludere.</p>

    <div class=\"summary-box\"><pre id=\"summaryText\"></pre></div>

    <div class=\"toolbar\">
      <span class=\"stat\" id=\"statRows\"></span>
      <span class=\"stat\" id=\"statPos\"></span>
      <span class=\"stat\" id=\"statSaved\"></span>
      <button id=\"btnPrev\" type=\"button\">Card precedente</button>
      <button id=\"btnNext\" type=\"button\">Card successiva</button>
      <button id=\"btnOne\" type=\"button\">Modalità 1 card</button>
      <button id=\"btnAll\" type=\"button\">Mostra tutte</button>
      <button id=\"btnExport\" type=\"button\">Esporta correzioni CSV</button>
      <span class=\"chip\">↑/↓ = sostituisci immagine con orientamento opposto | A/D = sposta scala X | R = reset X | X = escludi/includi</span>
    </div>

    <div id=\"cardsRoot\">{fallback_cards_html}</div>
  </div>

  <script>
    (function() {{
      try {{
      const ROWS = {payload_json};
      const SUMMARY = {summary_json};
      const cardsRoot = document.getElementById('cardsRoot');
      const statRows = document.getElementById('statRows');
      const statPos = document.getElementById('statPos');
      const statSaved = document.getElementById('statSaved');
      const btnPrev = document.getElementById('btnPrev');
      const btnNext = document.getElementById('btnNext');
      const btnOne = document.getElementById('btnOne');
      const btnAll = document.getElementById('btnAll');
      const btnExport = document.getElementById('btnExport');
      const summaryText = document.getElementById('summaryText');
      const STORAGE_KEY = 'scale_gt_orientation_debug__' + String(SUMMARY.rows_csv || 'default');

      let oneCardMode = true;
      let currentIdx = 0;
      let state = loadState();

      function loadState() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) return {{ variants: {{}}, excluded: {{}}, x_offsets: {{}} }};
          const obj = JSON.parse(raw);
          if (!obj || typeof obj !== 'object') return {{ variants: {{}}, excluded: {{}}, x_offsets: {{}} }};
          if (!obj.variants || typeof obj.variants !== 'object') obj.variants = {{}};
          if (!obj.excluded || typeof obj.excluded !== 'object') obj.excluded = {{}};
          if (!obj.x_offsets || typeof obj.x_offsets !== 'object') obj.x_offsets = {{}};
          return obj;
        }} catch (_err) {{
          return {{ variants: {{}}, excluded: {{}}, x_offsets: {{}} }};
        }}
      }}

      function saveState() {{
        try {{
          localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        }} catch (_err) {{}}
      }}

      function rowKey(r) {{
        return String(r.sample_id || '') + '||' + String(r.image_path || '');
      }}

      function repAll(s, from, to) {{
        return String(s).split(from).join(to);
      }}

      function esc(v) {{
        let s = String(v);
        s = repAll(s, '&', '&amp;');
        s = repAll(s, '<', '&lt;');
        s = repAll(s, '>', '&gt;');
        s = repAll(s, '"', '&quot;');
        s = repAll(s, \"'\", '&#39;');
        return s;
      }}

      function makeCard(r, i) {{
        const hasOpp = Boolean(r.opp_preview_rel);
        const oppStatus = String(r.opposite_status || (hasOpp ? 'ok' : 'missing'));
        const hasOppUsable = hasOpp && oppStatus === 'ok';
        const itemKey = rowKey(r);
        let oppWarn = '';
        if (oppStatus === 'missing') {{
          oppWarn = '<span class="warn">Nessuna immagine opposta trovata nella stessa cartella</span>';
        }} else if (oppStatus === 'same_as_original') {{
          oppWarn = '<span class="warn">Immagine opposta non disponibile: coincide con l\\'originale</span>';
        }}
        return `
          <section class="card" data-idx="${{i}}" data-item-key="${{esc(itemKey)}}" data-variant="orig" data-excluded="0">
            <h2>#${{i + 1}} ${{esc(r.sample_id)}} (${{esc(r.split)}})</h2>
            <p class="meta"><b>Orientamento ufficiale:</b> <b>${{esc(r.orientation_name || 'UNKNOWN')}}</b> | source=<code>${{esc(r.orientation_source_type || '')}}</code> | idx=${{Number.isFinite(Number(r.orientation_idx)) ? Number(r.orientation_idx) : -1}}</p>
            <p class="meta"><b>label_side:</b> ${{esc(r.label_side)}} | <b>SU/GIU stimato:</b> <b>${{esc(r.su_giu_pred)}}</b> | basis=<code>${{esc(r.orientation_basis)}}</code> | conf=${{Number(r.orientation_conf || 0).toFixed(3)}}</p>
            <p class="meta"><b>mse_to_ud:</b> ${{Number.isFinite(Number(r.mse_to_ud)) ? Number(r.mse_to_ud).toFixed(2) : 'nan'}} | <b>mse_to_nf:</b> ${{Number.isFinite(Number(r.mse_to_nf)) ? Number(r.mse_to_nf).toFixed(2) : 'nan'}}</p>
            <p class="meta"><b>manifest match:</b> <code>${{esc(r.manifest_match_source || '')}}</code> | <b>manifest cfg:</b> <code>${{esc(r.manifest_config_folder || '')}}</code></p>
            <p class="meta"><b>image:</b> <code>${{esc(r.image_path)}}</code></p>
            <p class="meta"><b>opp_source:</b> <code>${{esc(r.opposite_source_path || '')}}</code></p>
            <div class="img-actions">
              <button type="button" class="btn-orig active">Originale</button>
              <button type="button" class="btn-opp" ${{hasOppUsable ? '' : 'disabled'}}>Opposta</button>
              <button type="button" class="btn-shift btn-left5">← 5px</button>
              <button type="button" class="btn-shift btn-left1">← 1px</button>
              <button type="button" class="btn-shift btn-resetx">Reset X</button>
              <button type="button" class="btn-shift btn-right1">1px →</button>
              <button type="button" class="btn-shift btn-right5">5px →</button>
              <span class="shift-badge">Δx=<span class="x-offset-val">0</span>px</span>
              <button type="button" class="btn-exclude">Escludi</button>
              <span class="excluded-note" style="display:none">Immagine esclusa dal dataset</span>
              ${{oppWarn}}
            </div>
            <div class="img-wrap">
              <div class="img-stage">
                <a class="img-link" href="${{esc(r.orig_preview_rel)}}" target="_blank" rel="noopener noreferrer">
                  <img class="gt-img" src="${{esc(r.orig_preview_rel)}}" alt="gt preview" />
                </a>
                <div class="scale-overlay">
                  <div class="scale-line"></div>
                  <div class="scale-dot scale-dot-top"></div>
                  <div class="scale-dot scale-dot-bottom"></div>
                </div>
              </div>
            </div>
          </section>
        `;
      }}

      let cards = [];

      function renderCards() {{
        cardsRoot.innerHTML = ROWS.map((r, i) => makeCard(r, i)).join('');
        const renderedCards = Array.from(document.querySelectorAll('.card'));
        renderedCards.forEach((card, i) => {{
          const btnOrig = card.querySelector('.btn-orig');
          const btnOpp = card.querySelector('.btn-opp');
          const btnLeft5 = card.querySelector('.btn-left5');
          const btnLeft1 = card.querySelector('.btn-left1');
          const btnResetX = card.querySelector('.btn-resetx');
          const btnRight1 = card.querySelector('.btn-right1');
          const btnRight5 = card.querySelector('.btn-right5');
          const xOffsetVal = card.querySelector('.x-offset-val');
          const btnExclude = card.querySelector('.btn-exclude');
          const excludedNote = card.querySelector('.excluded-note');
          const img = card.querySelector('.gt-img');
          const link = card.querySelector('.img-link');
          const scaleOverlay = card.querySelector('.scale-overlay');
          const scaleLine = card.querySelector('.scale-line');
          const scaleDotTop = card.querySelector('.scale-dot-top');
          const scaleDotBottom = card.querySelector('.scale-dot-bottom');
          const itemKey = card.dataset.itemKey || '';
          const row = ROWS[i];

          function getOffsetPx() {{
            const v = Number(state.x_offsets && state.x_offsets[itemKey]);
            return Number.isFinite(v) ? Math.round(v) : 0;
          }}

          function applyScaleOverlay() {{
            const imgW = Number(row.image_w || 0);
            const imgH = Number(row.image_h || 0);
            const rect = img.getBoundingClientRect();
            const dw = Number(rect.width || img.clientWidth || 0);
            const dh = Number(rect.height || img.clientHeight || 0);
            if (!(imgW > 0 && imgH > 0 && dw > 0 && dh > 0)) {{
              scaleOverlay.style.display = 'none';
              return;
            }}
            scaleOverlay.style.display = 'block';
            const sx = dw / imgW;
            const sy = dh / imgH;

            const xBase = Number(row.x_gt_image || 0) + getOffsetPx();
            const yTopBase = Number(row.y_top_gt_image || 0);
            const yBotBase = Number(row.y_bottom_gt_image || 0);
            const x = xBase * sx;
            const yTop = yTopBase * sy;
            const yBot = yBotBase * sy;
            const h = Math.max(1, yBot - yTop);

            scaleLine.style.left = `${{x}}px`;
            scaleLine.style.top = `${{yTop}}px`;
            scaleLine.style.height = `${{h}}px`;
            scaleDotTop.style.left = `${{x}}px`;
            scaleDotTop.style.top = `${{yTop}}px`;
            scaleDotBottom.style.left = `${{x}}px`;
            scaleDotBottom.style.top = `${{yBot}}px`;
          }}

          function setOffsetPx(v) {{
            const next = Math.max(-250, Math.min(250, Math.round(Number(v) || 0)));
            if (!state.x_offsets || typeof state.x_offsets !== 'object') state.x_offsets = {{}};
            state.x_offsets[itemKey] = next;
            if (xOffsetVal) xOffsetVal.textContent = String(next);
            saveState();
            applyScaleOverlay();
            updateToolbar();
          }}

          function shiftOffsetPx(delta) {{
            setOffsetPx(getOffsetPx() + Number(delta || 0));
          }}

          function setVariant(v) {{
            if (v === 'opp' && (!row.opp_preview_rel || String(row.opposite_status || 'ok') !== 'ok')) return;
            card.dataset.variant = v;
            const src = (v === 'opp' && row.opp_preview_rel) ? row.opp_preview_rel : row.orig_preview_rel;
            img.src = src;
            link.href = src;
            btnOrig.classList.toggle('active', v === 'orig');
            btnOpp.classList.toggle('active', v === 'opp');
            if (img.complete) applyScaleOverlay();
            if (itemKey) {{
              state.variants[itemKey] = v;
              saveState();
              updateToolbar();
            }}
          }}

          function setExcluded(flag) {{
            const on = Boolean(flag);
            card.dataset.excluded = on ? '1' : '0';
            card.classList.toggle('excluded', on);
            btnExclude.textContent = on ? 'Includi' : 'Escludi';
            btnExclude.classList.toggle('active', on);
            excludedNote.style.display = on ? 'inline' : 'none';
            if (itemKey) {{
              state.excluded[itemKey] = on;
              saveState();
              updateToolbar();
            }}
          }}

          btnOrig.addEventListener('click', () => setVariant('orig'));
          btnOpp.addEventListener('click', () => setVariant('opp'));
          btnLeft5.addEventListener('click', () => shiftOffsetPx(-5));
          btnLeft1.addEventListener('click', () => shiftOffsetPx(-1));
          btnResetX.addEventListener('click', () => setOffsetPx(0));
          btnRight1.addEventListener('click', () => shiftOffsetPx(1));
          btnRight5.addEventListener('click', () => shiftOffsetPx(5));
          btnExclude.addEventListener('click', () => setExcluded(card.dataset.excluded !== '1'));
          img.addEventListener('load', applyScaleOverlay);

          const savedVariant = String((state.variants && state.variants[itemKey]) || 'orig');
          if (savedVariant === 'opp' && row.opp_preview_rel && String(row.opposite_status || 'ok') === 'ok') setVariant('opp');
          else setVariant('orig');
          setOffsetPx(getOffsetPx());
          const savedExcluded = Boolean(state.excluded && state.excluded[itemKey]);
          setExcluded(savedExcluded);
          card._applyScaleOverlay = applyScaleOverlay;
        }});

        return renderedCards;
      }}

      cards = renderCards();

      function updateSummaryBox() {{
        const lines = [];
        lines.push(`rows: ${{SUMMARY.rows}}`);
        lines.push(`with_opposite_image: ${{SUMMARY.with_opposite_image}}`);
        lines.push('su_giu_counts: ' + JSON.stringify(SUMMARY.su_giu_counts));
        lines.push('label_side_counts: ' + JSON.stringify(SUMMARY.label_side_counts));
        lines.push('correlation(label_side x su_giu): ' + JSON.stringify(SUMMARY.correlation));
        lines.push('rows_with_manifest_dupe_candidates: ' + String(SUMMARY.rows_with_manifest_dupe_candidates || 0));
        lines.push('manifest_match_source_counts: ' + JSON.stringify(SUMMARY.manifest_match_source_counts || {{}}));
        lines.push('orientation_counts: ' + JSON.stringify(SUMMARY.orientation_counts || {{}}));
        lines.push('opposite_status_counts: ' + JSON.stringify(SUMMARY.opposite_status_counts || {{}}));
        lines.push('note: su=più simile a no_flip/depth_value/biplana, giu=più simile a flip_ud');
        summaryText.textContent = lines.join('\\n');
      }}

      function updateToolbar() {{
        statRows.textContent = `Righe: ${{cards.length}}`;
        let savedOpp = 0;
        let savedExcluded = 0;
        let movedX = 0;
        for (const r of ROWS) {{
          const k = rowKey(r);
          if (String((state.variants && state.variants[k]) || 'orig') === 'opp') savedOpp += 1;
          if (Boolean(state.excluded && state.excluded[k])) savedExcluded += 1;
          const xoff = Number(state.x_offsets && state.x_offsets[k]);
          if (Number.isFinite(xoff) && Math.round(xoff) !== 0) movedX += 1;
        }}
        statSaved.textContent = `Correzioni: opposta=${{savedOpp}} | escluse=${{savedExcluded}} | spostateX=${{movedX}}`;
        if (!oneCardMode) {{
          statPos.textContent = 'Vista: tutte le card';
          return;
        }}
        const visibleIdx = cards.length ? (currentIdx + 1) : 0;
        statPos.textContent = `Card: ${{visibleIdx}} / ${{cards.length}}`;
      }}

      function showCards() {{
        cards.forEach((c, i) => {{
          if (oneCardMode) c.style.display = (i === currentIdx) ? 'block' : 'none';
          else c.style.display = 'block';
        }});
        cards.forEach((c, i) => {{
          if (oneCardMode && i !== currentIdx) return;
          if (c && c._applyScaleOverlay) c._applyScaleOverlay();
        }});
        updateToolbar();
      }}

      function move(delta) {{
        if (!cards.length) return;
        currentIdx += delta;
        if (currentIdx < 0) currentIdx = cards.length - 1;
        if (currentIdx >= cards.length) currentIdx = 0;
        showCards();
        if (oneCardMode) cards[currentIdx].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}

      function toggleVariant(delta) {{
        if (!cards.length) return;
        const activeCard = (!oneCardMode && document.activeElement && document.activeElement.closest)
          ? document.activeElement.closest('.card')
          : null;
        const card = oneCardMode ? cards[currentIdx] : (activeCard || cards[currentIdx]);
        if (!card) return;
        const btnOrig = card.querySelector('.btn-orig');
        const btnOpp = card.querySelector('.btn-opp');
        const hasOpp = !btnOpp.disabled;
        if (!hasOpp) return;
        const isOrig = card.dataset.variant !== 'opp';
        if (delta > 0) {{
          if (isOrig) btnOpp.click();
          else btnOrig.click();
        }} else {{
          if (isOrig) btnOpp.click();
          else btnOrig.click();
        }}
      }}

      function toggleExcludedCurrent() {{
        if (!cards.length) return;
        const card = cards[currentIdx];
        if (!card) return;
        const btnExclude = card.querySelector('.btn-exclude');
        if (btnExclude) btnExclude.click();
      }}

      function nudgeScaleCurrent(delta) {{
        if (!cards.length) return;
        const card = cards[currentIdx];
        if (!card) return;
        if (delta === 0) {{
          const btnResetX = card.querySelector('.btn-resetx');
          if (btnResetX) btnResetX.click();
          return;
        }}
        const btn = delta > 0
          ? (Math.abs(delta) >= 5 ? card.querySelector('.btn-right5') : card.querySelector('.btn-right1'))
          : (Math.abs(delta) >= 5 ? card.querySelector('.btn-left5') : card.querySelector('.btn-left1'));
        if (btn) btn.click();
      }}

      btnPrev.addEventListener('click', () => move(-1));
      btnNext.addEventListener('click', () => move(1));
      btnOne.addEventListener('click', () => {{ oneCardMode = true; showCards(); }});
      btnAll.addEventListener('click', () => {{ oneCardMode = false; showCards(); }});
      btnExport.addEventListener('click', () => {{
        const header = [
          'sample_id',
          'split',
          'image_path',
          'selected_variant',
          'excluded',
          'x_offset_px',
          'opposite_source_path',
          'opposite_status',
          'su_giu_pred',
          'orientation_basis',
          'label_side',
          'manifest_match_source',
          'manifest_config_folder'
        ];
        const outRows = [];
        for (const r of ROWS) {{
          const k = rowKey(r);
          const sel = String((state.variants && state.variants[k]) || 'orig');
          const excluded = Boolean(state.excluded && state.excluded[k]);
          if (sel !== 'opp' && !excluded) continue;
          outRows.push([
            r.sample_id || '',
            r.split || '',
            r.image_path || '',
            sel,
            excluded ? 1 : 0,
            Number((state.x_offsets && state.x_offsets[k]) || 0),
            r.opposite_source_path || '',
            r.opposite_status || '',
            r.su_giu_pred || '',
            r.orientation_basis || '',
            r.label_side || '',
            r.manifest_match_source || '',
            r.manifest_config_folder || ''
          ]);
        }}
        if (!outRows.length) {{
          alert('Nessuna correzione da esportare. Seleziona \"Opposta\" o \"Escludi\" su almeno una card.');
          return;
        }}
        function csvEsc(v) {{
          const s = String(v == null ? '' : v);
          if (s.indexOf('\"') >= 0 || s.indexOf(',') >= 0 || s.indexOf('\\n') >= 0) return '\"' + repAll(s, '\"', '\"\"') + '\"';
          return s;
        }}
        let csv = header.map(csvEsc).join(',') + '\\n';
        for (const rr of outRows) csv += rr.map(csvEsc).join(',') + '\\n';
        const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'gt_orientation_corrections.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      }});

      document.addEventListener('keydown', (ev) => {{
        const tag = ev.target && ev.target.tagName ? ev.target.tagName.toUpperCase() : '';
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (ev.key === 'ArrowRight') {{ ev.preventDefault(); move(1); }}
        else if (ev.key === 'ArrowLeft') {{ ev.preventDefault(); move(-1); }}
        else if (ev.key === 'ArrowUp') {{ ev.preventDefault(); toggleVariant(-1); }}
        else if (ev.key === 'ArrowDown') {{ ev.preventDefault(); toggleVariant(1); }}
        else if (ev.key === 'x' || ev.key === 'X') {{ ev.preventDefault(); toggleExcludedCurrent(); }}
        else if (ev.key === 'a' || ev.key === 'A') {{ ev.preventDefault(); nudgeScaleCurrent(-1); }}
        else if (ev.key === 'd' || ev.key === 'D') {{ ev.preventDefault(); nudgeScaleCurrent(1); }}
        else if (ev.key === 'r' || ev.key === 'R') {{ ev.preventDefault(); nudgeScaleCurrent(0); }}
      }});

      window.addEventListener('resize', () => {{
        cards.forEach((c, i) => {{
          if (oneCardMode && i !== currentIdx) return;
          if (c && c._applyScaleOverlay) c._applyScaleOverlay();
        }});
      }});

      updateSummaryBox();
      showCards();
      }} catch (err) {{
        const root = document.getElementById('cardsRoot');
        const msg = (err && err.message) ? err.message : String(err);
        if (root) {{
          root.innerHTML = '<section class=\"card\" style=\"display:block\"><h2>Errore rendering JS</h2><p class=\"meta\">' + msg + '</p></section>';
        }}
        if (window && window.console && window.console.error) {{
          window.console.error(err);
        }}
      }}
    }})();
  </script>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build GT-only orientation debug HTML.")
    p.add_argument(
        "--rows-csv",
        type=Path,
        default=Path(
            "artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/refined_ticks_zero_last_v2_dataset/test_line_postprocessed.csv"
        ),
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("artifacts/32_scale_image_model_per_vendor_20260416/manifests/manifest_scale_bk.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/gt_orientation_debug"
        ),
    )
    p.add_argument("--max-rows", type=int, default=220)
    p.add_argument(
        "--prefer-image-root",
        type=Path,
        default=None,
        help="Root immagini preferito (es. /Volumes/SSD_esi1_n1). Se il file esiste lì, viene usato al posto del path nel CSV.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    rows_csv = args.rows_csv.expanduser().resolve()
    manifest_csv = args.manifest_csv.expanduser().resolve()
    prefer_root = args.prefer_image_root.expanduser().resolve() if args.prefer_image_root else None
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    manifest = load_manifest(manifest_csv)
    rows = load_rows(rows_csv, max_rows=int(args.max_rows))

    payload: List[Dict[str, object]] = []
    corr: Dict[str, Dict[str, int]] = {}
    su_giu_counts: Dict[str, int] = {"su": 0, "giu": 0, "unknown": 0}
    label_side_counts: Dict[str, int] = {}
    with_opp = 0
    opposite_status_counts: Dict[str, int] = {"ok": 0, "missing": 0, "same_as_original": 0}
    orientation_counts: Dict[str, int] = {"NF": 0, "LR": 0, "UD": 0, "LRUD": 0, "UNKNOWN": 0}
    match_source_counts: Dict[str, int] = {}
    rows_with_manifest_dupes = 0

    for i, row in enumerate(rows, start=1):
        img_path = _remap_to_preferred_root(Path(row.image_path), prefer_root)
        mm = select_manifest_line(manifest_by_sample=manifest, sample_id=row.sample_id, image_path=img_path)
        if mm is None:
            continue
        gt = mm.gt
        if len(manifest.get(row.sample_id, [])) > 1:
            rows_with_manifest_dupes += 1
        match_source_counts[mm.source] = int(match_source_counts.get(mm.source, 0) + 1)

        if not _exists_image(img_path):
            continue

        opp = find_opposite_info(current=img_path, sample_id=row.sample_id)
        opp_official = _find_opposite_by_official_orientation(
            current=img_path,
            sample_id=row.sample_id,
            orientation_name=row.orientation_name,
        )
        opp_for_toggle = opp_official or opp.opposite_path

        base = f"{i:04d}_{_safe_slug(row.sample_id)}"
        out_orig = preview_dir / f"{base}_orig_gt.jpg"
        label_txt = f"GT only | sample={row.sample_id} | label_side={gt.label_side} | sugiu={opp.su_giu_pred}"
        w, h, x_img, y1_img, y2_img = draw_gt_overlay(
            src=img_path,
            out_path=out_orig,
            gt=gt,
            label=label_txt,
            draw_overlay=False,
        )

        out_opp_rel: Optional[str] = None
        opp_src_path: Optional[str] = None
        opposite_status = "missing"
        if opp_for_toggle is not None and _exists_image(opp_for_toggle):
            same_as_original = False
            try:
                same_as_original = opp_for_toggle.resolve() == img_path.resolve()
            except Exception:
                same_as_original = opp_for_toggle.as_posix() == img_path.as_posix()

            if same_as_original:
                opposite_status = "same_as_original"
            else:
                opposite_status = "ok"
                with_opp += 1
                out_opp = preview_dir / f"{base}_opp_gt.jpg"
                draw_gt_overlay(
                    src=opp_for_toggle,
                    out_path=out_opp,
                    gt=gt,
                    label=(
                        "GT only (same coords) | "
                        f"sample={row.sample_id} | label_side={gt.label_side} | sugiu={opp.su_giu_pred}"
                    ),
                    draw_overlay=False,
                )
                out_opp_rel = f"{preview_dir.name}/{out_opp.name}"
                opp_src_path = opp_for_toggle.as_posix()

        opposite_status_counts[opposite_status] = int(opposite_status_counts.get(opposite_status, 0) + 1)

        label_side_s = "-1" if gt.label_side < 0 else "+1"
        label_side_counts[label_side_s] = int(label_side_counts.get(label_side_s, 0) + 1)

        s = opp.su_giu_pred if opp.su_giu_pred in su_giu_counts else "unknown"
        su_giu_counts[s] = int(su_giu_counts.get(s, 0) + 1)
        corr.setdefault(label_side_s, {})
        corr[label_side_s][s] = int(corr[label_side_s].get(s, 0) + 1)

        ori_name = row.orientation_name if row.orientation_name in {"NF", "LR", "UD", "LRUD"} else "UNKNOWN"
        orientation_counts[ori_name] = int(orientation_counts.get(ori_name, 0) + 1)

        payload.append(
            {
                "rank": i,
                "sample_id": row.sample_id,
                "split": row.split,
                "image_path": img_path.as_posix(),
                "score_after": row.score_after,
                "orientation_name": row.orientation_name or "",
                "orientation_idx": row.orientation_idx,
                "orientation_source_type": row.orientation_source_type or "",
                "label_side": gt.label_side,
                "manifest_match_source": mm.source,
                "manifest_config_folder": gt.config_folder,
                "x_gt_image": x_img,
                "y_top_gt_image": y1_img,
                "y_bottom_gt_image": y2_img,
                "image_w": w,
                "image_h": h,
                "su_giu_pred": opp.su_giu_pred,
                "orientation_basis": opp.basis,
                "orientation_conf": opp.confidence,
                "mse_to_ud": opp.mse_to_ud,
                "mse_to_nf": opp.mse_to_nf,
                "ud_source_path": opp.source_ud.as_posix() if opp.source_ud else "",
                "nf_source_path": opp.source_nf.as_posix() if opp.source_nf else "",
                "opposite_source_path": opp_src_path or "",
                "opposite_status": opposite_status,
                "orig_preview_rel": f"{preview_dir.name}/{out_orig.name}",
                "opp_preview_rel": out_opp_rel or "",
            }
        )

    out_html = out_dir / "review_gt_orientation_debug.html"
    summary = {
        "rows": len(payload),
        "with_opposite_image": with_opp,
        "su_giu_counts": su_giu_counts,
        "label_side_counts": label_side_counts,
        "correlation": corr,
        "rows_with_manifest_dupe_candidates": rows_with_manifest_dupes,
        "manifest_match_source_counts": match_source_counts,
        "orientation_counts": orientation_counts,
        "opposite_status_counts": opposite_status_counts,
        "rows_csv": rows_csv.as_posix(),
        "manifest_csv": manifest_csv.as_posix(),
    }
    build_html(output_html=out_html, rows_payload=payload, summary=summary)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"html": out_html.as_posix(), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
