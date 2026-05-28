#!/usr/bin/env python3
"""Infer vendor per acquisition folder and build an HTML review page."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import torch

# Reuse the same vendor model pipeline used by rect inference.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import infer_ultrasound_rect_folder_html as rect_infer


def _build_ocr_keyword_map(class_names: Sequence[str]) -> Dict[str, List[str]]:
    base_map: Dict[str, List[str]] = {
        "BK": ["bk", "profocus", "flexfocus", "specto", "bk5000", "bk3000"],
        "Esaote": ["esaote", "mylab"],
        "Hitachi": ["hitachi", "arietta"],
        "GE": ["logiq", "versana", "voluson"],
        "Mindray": ["mindray", "resona", "te7"],
        "Canon": ["canon", "aplio"],
        "Philips": ["philips", "affiniti", "epiq"],
        "Toshiba": ["toshiba", "aplio"],
        "Siemens": ["siemens", "acuson"],
        "Koelis": ["koelis"],
        "Biopsee": ["biopsee", "biojet"],
        "ExactVu": ["exactvu"],
        "Alpinion": ["alpinion", "ecube", "e-cube"],
        "Terason": ["terason"],
        "Sonostar": ["sonostar"],
        "Vinno": ["vinno"],
    }
    return {name: base_map.get(name, [name.lower()]) for name in class_names}


def _run_ocr_text(image_path: Path, lang: str, psm: int, timeout_sec: float) -> str:
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--oem",
                "1",
                "--psm",
                str(psm),
                "-l",
                lang,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception:
        return ""
    return (proc.stdout or "").lower()


def _ocr_vendor_hits(
    image_paths: Sequence[Path],
    keyword_map: Dict[str, Sequence[str]],
    lang: str,
    psm: int,
    timeout_sec: float,
) -> Counter[str]:
    hits: Counter[str] = Counter()
    for image_path in image_paths:
        text = _run_ocr_text(image_path=image_path, lang=lang, psm=psm, timeout_sec=timeout_sec)
        if not text:
            continue
        for vendor, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                hits[vendor] += 1
    return hits


def _build_selected_paths_by_group(
    records: Sequence[rect_infer.ImageRecord],
    sample_per_group: int,
) -> Dict[str, List[Path]]:
    by_group_paths: Dict[str, List[Path]] = {}
    for rec in records:
        by_group_paths.setdefault(rec.group_name, []).append(rec.image_path)
    out: Dict[str, List[Path]] = {}
    for group_name, paths in by_group_paths.items():
        unique_paths = sorted({p for p in paths})
        out[group_name] = rect_infer._select_subset(unique_paths, limit=sample_per_group)
    return out


def _apply_ocr_low_confidence_refine(
    vendor_map: Dict[str, Dict[str, float | str | int]],
    selected_paths_by_group: Dict[str, List[Path]],
    class_names: Sequence[str],
    enable_ocr: bool,
    low_conf_threshold: float,
    ocr_lang: str,
    ocr_psm: int,
    ocr_timeout_sec: float,
    ocr_max_images_per_group: int,
    ocr_min_hit_count: int,
    ocr_min_hit_ratio: float,
) -> Dict[str, object]:
    keyword_map = _build_ocr_keyword_map(class_names)
    low_conf_groups = 0
    ocr_applied_groups = 0
    ocr_groups_with_hits = 0
    ocr_override_groups = 0
    ocr_override_dist: Dict[str, int] = {}

    for group_name, info in vendor_map.items():
        cnn_vendor = str(info.get("vendor_predicted", "UNKNOWN"))
        top1_prob = float(info.get("vendor_top1_prob", 0.0))
        info["vendor_predicted_cnn"] = cnn_vendor
        info["ocr_applied"] = 0
        info["ocr_override"] = 0
        info["ocr_top_vendor"] = ""
        info["ocr_hit_count"] = 0
        info["ocr_hit_ratio"] = 0.0
        info["ocr_images_checked"] = 0
        info["low_confidence_triggered"] = 1 if top1_prob <= low_conf_threshold else 0

        if top1_prob > low_conf_threshold:
            continue
        low_conf_groups += 1
        if not enable_ocr:
            continue

        selected_paths = selected_paths_by_group.get(group_name, [])
        if not selected_paths:
            continue
        ocr_images = rect_infer._select_subset(selected_paths, limit=ocr_max_images_per_group)
        info["ocr_images_checked"] = int(len(ocr_images))
        if not ocr_images:
            continue
        ocr_applied_groups += 1
        info["ocr_applied"] = 1

        hits = _ocr_vendor_hits(
            image_paths=ocr_images,
            keyword_map=keyword_map,
            lang=ocr_lang,
            psm=ocr_psm,
            timeout_sec=ocr_timeout_sec,
        )
        if not hits:
            continue
        ocr_groups_with_hits += 1
        top_vendor, top_count = hits.most_common(1)[0]
        hit_ratio = float(top_count / max(1, len(ocr_images)))
        info["ocr_top_vendor"] = str(top_vendor)
        info["ocr_hit_count"] = int(top_count)
        info["ocr_hit_ratio"] = hit_ratio

        if top_count >= ocr_min_hit_count and hit_ratio >= ocr_min_hit_ratio:
            if top_vendor != cnn_vendor:
                info["vendor_predicted"] = str(top_vendor)
                info["ocr_override"] = 1
                ocr_override_groups += 1
                ocr_override_dist[str(top_vendor)] = ocr_override_dist.get(str(top_vendor), 0) + 1

    return {
        "enabled": bool(enable_ocr),
        "low_conf_threshold": float(low_conf_threshold),
        "low_conf_groups": int(low_conf_groups),
        "ocr_applied_groups": int(ocr_applied_groups),
        "ocr_groups_with_hits": int(ocr_groups_with_hits),
        "ocr_override_groups": int(ocr_override_groups),
        "ocr_override_distribution": dict(sorted(ocr_override_dist.items())),
        "ocr_lang": ocr_lang,
        "ocr_psm": int(ocr_psm),
        "ocr_timeout_sec": float(ocr_timeout_sec),
        "ocr_max_images_per_group": int(ocr_max_images_per_group),
        "ocr_min_hit_count": int(ocr_min_hit_count),
        "ocr_min_hit_ratio": float(ocr_min_hit_ratio),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vendor inference by acquisition folder + HTML review (pipeline network)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root dataset (group folders containing NF/LR/UD/LRUD).",
    )
    parser.add_argument(
        "--vendor-checkpoint",
        type=Path,
        default=Path(
            "artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_folder_review"),
    )
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--vendor-image-size", type=int, default=0)
    parser.add_argument("--vendor-batch-size", type=int, default=48)
    parser.add_argument("--vendor-sample-per-group", type=int, default=64)
    parser.add_argument("--vendor-log-interval", type=int, default=25)
    parser.add_argument(
        "--enable-ocr-low-confidence",
        action="store_true",
        help="Se attivo, applica OCR sui gruppi con confidence <= soglia e può fare override vendor.",
    )
    parser.add_argument(
        "--ocr-low-confidence-threshold",
        type=float,
        default=0.60,
        help="Soglia su vendor_top1_prob (<=) per attivare OCR.",
    )
    parser.add_argument("--ocr-lang", type=str, default="eng")
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--ocr-timeout-sec", type=float, default=5.0)
    parser.add_argument("--ocr-max-images-per-group", type=int, default=8)
    parser.add_argument("--ocr-min-hit-count", type=int, default=2)
    parser.add_argument("--ocr-min-hit-ratio", type=float, default=0.20)
    parser.add_argument("--device", type=str, default=None)
    return parser


def _write_csv(
    rows: List[Dict[str, object]],
    out_csv: Path,
) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "group_name",
                "num_images",
                "vendor_predicted",
                "vendor_predicted_cnn",
                "vendor_top1_prob",
                "vendor_margin_top1_top2",
                "vendor_vote_ratio",
                "vendor_images_used",
                "low_confidence_triggered",
                "ocr_applied",
                "ocr_override",
                "ocr_top_vendor",
                "ocr_hit_count",
                "ocr_hit_ratio",
                "ocr_images_checked",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_html(
    rows: List[Dict[str, object]],
    summary: Dict[str, object],
    classes: List[str],
    out_html: Path,
) -> None:
    vendors = sorted({str(r["vendor_predicted"]) for r in rows}, key=lambda x: x.lower())
    options = ["KEEP"] + classes + ["UNKNOWN"]
    options_html = "\n".join(
        f"<option value=\"{html.escape(v)}\">{html.escape(v)}</option>" for v in options
    )

    blocks: List[str] = []
    for vendor in vendors:
        vendor_rows = [r for r in rows if str(r["vendor_predicted"]) == vendor]
        # Prioritize manual review: lowest confidence first inside each vendor block.
        vendor_rows.sort(
            key=lambda r: (
                float(r.get("vendor_top1_prob", 0.0)),
                float(r.get("vendor_margin_top1_top2", 0.0)),
                float(r.get("vendor_vote_ratio", 0.0)),
                str(r["group_name"]).lower(),
            )
        )
        cards = []
        for row in vendor_rows:
            group_name_raw = str(row["group_name"])
            group_name = html.escape(group_name_raw)
            group_name_attr = html.escape(group_name_raw, quote=True)
            vendor_attr = html.escape(vendor, quote=True)
            card = f"""
            <div class="card" data-vendor="{vendor_attr}" data-vendor-pred="{vendor_attr}" data-group-name="{group_name_attr}">
              <div class="head">
                <label><input type="checkbox" class="flag"> flag</label>
                <select class="corr">{options_html}</select>
              </div>
              <div class="group">{group_name}</div>
              <div class="meta">
                imgs={int(row["num_images"])} | used={int(row["vendor_images_used"])} |
                cnn={html.escape(str(row.get("vendor_predicted_cnn", row["vendor_predicted"])))} |
                p={float(row["vendor_top1_prob"]):.3f} |
                margin={float(row["vendor_margin_top1_top2"]):.3f} |
                vote={float(row["vendor_vote_ratio"]):.3f} |
                ocr={int(row.get("ocr_applied", 0))} |
                ocr_top={html.escape(str(row.get("ocr_top_vendor", "")))} |
                ocr_hit={int(row.get("ocr_hit_count", 0))}/{int(row.get("ocr_images_checked", 0))}
              </div>
              <div class="note-row">
                <input type="text" class="note" placeholder="nota (perché errata o ambigua)">
              </div>
            </div>
            """
            cards.append(card)
        blocks.append(
            f"""
            <section>
              <h2>{html.escape(vendor)} <small>({len(vendor_rows)} folders)</small></h2>
              <div class="grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    html_doc = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vendor Folder Review</title>
  <style>
    :root {{
      --bg: #f5f7f6;
      --card: #ffffff;
      --ink: #102119;
      --muted: #486154;
      --accent: #0f8f64;
      --warn: #b6402f;
      --line: #d8e2dc;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 500px at 10% -10%, #dfeee5 0%, var(--bg) 60%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h2 small {{ color: var(--muted); font-weight: 500; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 247, 246, 0.94);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 14px;
      backdrop-filter: blur(6px);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--card);
      color: var(--ink);
      border-radius: 9px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 10px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }}
    .group {{
      font-weight: 700;
      line-height: 1.25;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    .meta {{
      font-size: 13px;
      color: var(--muted);
    }}
    .note-row {{
      margin-top: 8px;
    }}
    .note {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      font-size: 12px;
      color: var(--ink);
      background: #fdfefd;
    }}
    .card.bad {{
      border-color: var(--warn);
      box-shadow: 0 0 0 2px rgba(182, 64, 47, 0.15);
    }}
    pre {{
      background: #0f241a;
      color: #d8f6e8;
      border-radius: 12px;
      padding: 12px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Vendor Folder Review</h1>
    <div class="toolbar">
      <button id="toggleBad">Mostra solo flaggati</button>
      <button class="primary" id="exportFlags">Esporta flags CSV</button>
      <button class="primary" id="exportCorrections">Esporta corrections CSV</button>
      <button id="clearState">Reset stato locale</button>
      <span id="count"></span>
    </div>
    {''.join(blocks)}
    <h2>Summary</h2>
    <pre>{summary_json}</pre>
  </div>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const STORAGE_KEY = `vendor-review::${{window.location.pathname}}`;
    let onlyBad = false;

    function loadState() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {{ items: {{}} }};
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return {{ items: {{}} }};
        if (!parsed.items || typeof parsed.items !== 'object') parsed.items = {{}};
        return parsed;
      }} catch (_err) {{
        return {{ items: {{}} }};
      }}
    }}

    function saveState(state) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function getItem(state, groupName) {{
      if (!state.items[groupName]) {{
        state.items[groupName] = {{ flagged: 0, correction_vendor: '', note: '' }};
      }}
      return state.items[groupName];
    }}

    function applyItemToCard(card, item) {{
      card.querySelector('.flag').checked = !!item.flagged;
      const corrEl = card.querySelector('.corr');
      const corrValue = item.correction_vendor ? item.correction_vendor : 'KEEP';
      const options = Array.from(corrEl.options).map((o) => o.value);
      corrEl.value = options.includes(corrValue) ? corrValue : 'KEEP';
      card.querySelector('.note').value = item.note || '';
    }}

    function readCardToItem(card, item) {{
      item.flagged = card.querySelector('.flag').checked ? 1 : 0;
      const corr = card.querySelector('.corr').value;
      item.correction_vendor = corr === 'KEEP' ? '' : corr;
      item.note = card.querySelector('.note').value || '';
    }}

    function updateView() {{
      let badCount = 0;
      for (const c of cards) {{
        const flagged = c.querySelector('.flag').checked;
        c.classList.toggle('bad', flagged);
        if (flagged) badCount += 1;
        c.style.display = (!onlyBad || flagged) ? '' : 'none';
      }}
      document.getElementById('count').textContent = `flaggati: ${{badCount}} / ${{cards.length}}`;
    }}
    cards.forEach((c) => {{
      c.querySelector('.flag').addEventListener('change', updateView);
      c.querySelector('.corr').addEventListener('change', updateView);
    }});
    document.getElementById('toggleBad').addEventListener('click', () => {{
      onlyBad = !onlyBad;
      document.getElementById('toggleBad').textContent = onlyBad ? 'Mostra tutti' : 'Mostra solo flaggati';
      updateView();
    }});

    function downloadCsv(filename, rows) {{
      const csv = rows.map(r => r.map(v => `"${{String(v ?? '').replace(/"/g, '""')}}"`).join(',')).join('\\n');
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function rowData(card) {{
      const groupName = card.dataset.groupName || card.querySelector('.group').textContent.trim();
      const vendorPred = card.dataset.vendorPred || '';
      const corr = card.querySelector('.corr').value;
      const note = (card.querySelector('.note').value || '').trim();
      return {{
        group_name: groupName,
        vendor_predicted: vendorPred,
        flagged: card.querySelector('.flag').checked ? 1 : 0,
        correction_vendor: corr === 'KEEP' ? '' : corr,
        note,
      }};
    }}

    function bindEvents(state) {{
      for (const c of cards) {{
        c.querySelector('.flag').addEventListener('change', () => {{
          const item = getItem(state, c.dataset.groupName || '');
          readCardToItem(c, item);
          saveState(state);
          updateView();
        }});
        c.querySelector('.corr').addEventListener('change', () => {{
          const item = getItem(state, c.dataset.groupName || '');
          readCardToItem(c, item);
          saveState(state);
          updateView();
        }});
        c.querySelector('.note').addEventListener('input', () => {{
          const item = getItem(state, c.dataset.groupName || '');
          readCardToItem(c, item);
          saveState(state);
        }});
      }}
    }}

    document.getElementById('exportFlags').addEventListener('click', () => {{
      const rows = [['group_name', 'vendor_predicted', 'flagged', 'correction_vendor', 'note']];
      for (const c of cards) {{
        const d = rowData(c);
        if (d.flagged) rows.push([d.group_name, d.vendor_predicted, d.flagged, d.correction_vendor, d.note]);
      }}
      downloadCsv('vendor_flags.csv', rows);
    }});

    document.getElementById('exportCorrections').addEventListener('click', () => {{
      const rows = [['group_name', 'vendor_predicted', 'correction_vendor', 'note']];
      for (const c of cards) {{
        const d = rowData(c);
        if (d.correction_vendor) rows.push([d.group_name, d.vendor_predicted, d.correction_vendor, d.note]);
      }}
      downloadCsv('vendor_corrections.csv', rows);
    }});

    document.getElementById('clearState').addEventListener('click', () => {{
      if (!confirm('Confermi reset completo di flag/correzioni/note locali?')) return;
      localStorage.removeItem(STORAGE_KEY);
      window.location.reload();
    }});

    const state = loadState();
    cards.forEach((c) => {{
      const item = getItem(state, c.dataset.groupName || '');
      applyItemToCard(c, item);
    }});
    bindEvents(state);
    updateView();
  </script>
</body>
</html>
"""
    out_html.write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    vendor_checkpoint = args.vendor_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not vendor_checkpoint.exists():
        raise FileNotFoundError(f"Vendor checkpoint not found: {vendor_checkpoint}")

    records = rect_infer._find_images(input_dir=input_dir, limit_images=int(args.limit_images))
    if not records:
        raise RuntimeError(f"No images found under: {input_dir}")
    print(f"Images discovered: {len(records)}", flush=True)

    device = rect_infer.choose_device(args.device)
    print(f"Device: {device}", flush=True)

    vendor_map, vendor_summary = rect_infer._infer_vendor_by_group(
        records=records,
        checkpoint_path=vendor_checkpoint,
        image_size_override=int(args.vendor_image_size),
        batch_size=int(args.vendor_batch_size),
        sample_per_group=int(args.vendor_sample_per_group),
        device=device,
        log_interval=int(args.vendor_log_interval),
    )
    selected_paths_by_group = _build_selected_paths_by_group(
        records=records,
        sample_per_group=int(args.vendor_sample_per_group),
    )
    ocr_refine_summary = _apply_ocr_low_confidence_refine(
        vendor_map=vendor_map,
        selected_paths_by_group=selected_paths_by_group,
        class_names=[str(x) for x in vendor_summary.get("class_names", [])],
        enable_ocr=bool(args.enable_ocr_low_confidence),
        low_conf_threshold=float(args.ocr_low_confidence_threshold),
        ocr_lang=str(args.ocr_lang),
        ocr_psm=int(args.ocr_psm),
        ocr_timeout_sec=float(args.ocr_timeout_sec),
        ocr_max_images_per_group=int(args.ocr_max_images_per_group),
        ocr_min_hit_count=int(args.ocr_min_hit_count),
        ocr_min_hit_ratio=float(args.ocr_min_hit_ratio),
    )

    group_counts = Counter(rec.group_name for rec in records)
    rows: List[Dict[str, object]] = []
    for group_name in sorted(vendor_map.keys(), key=lambda x: x.lower()):
        info = vendor_map[group_name]
        rows.append(
            {
                "group_name": group_name,
                "num_images": int(group_counts.get(group_name, 0)),
                "vendor_predicted": str(info.get("vendor_predicted", "UNKNOWN")),
                "vendor_predicted_cnn": str(info.get("vendor_predicted_cnn", info.get("vendor_predicted", "UNKNOWN"))),
                "vendor_top1_prob": float(info.get("vendor_top1_prob", 0.0)),
                "vendor_margin_top1_top2": float(info.get("vendor_margin_top1_top2", 0.0)),
                "vendor_vote_ratio": float(info.get("vendor_vote_ratio", 0.0)),
                "vendor_images_used": int(info.get("vendor_images_used", 0)),
                "low_confidence_triggered": int(info.get("low_confidence_triggered", 0)),
                "ocr_applied": int(info.get("ocr_applied", 0)),
                "ocr_override": int(info.get("ocr_override", 0)),
                "ocr_top_vendor": str(info.get("ocr_top_vendor", "")),
                "ocr_hit_count": int(info.get("ocr_hit_count", 0)),
                "ocr_hit_ratio": float(info.get("ocr_hit_ratio", 0.0)),
                "ocr_images_checked": int(info.get("ocr_images_checked", 0)),
            }
        )

    csv_path = output_dir / "vendor_by_group.csv"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "review.html"
    _write_csv(rows=rows, out_csv=csv_path)

    ckpt = torch.load(vendor_checkpoint, map_location="cpu", weights_only=False)
    classes = [str(x) for x in ckpt.get("class_names", [])]
    summary = {
        "input_dir": input_dir.as_posix(),
        "vendor_checkpoint": vendor_checkpoint.as_posix(),
        "num_images": len(records),
        "num_groups": len(vendor_map),
        "vendor_dist_groups": dict(sorted(Counter(str(r["vendor_predicted"]) for r in rows).items())),
        "vendor_recognition": vendor_summary,
        "ocr_low_confidence_refine": ocr_refine_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _build_html(rows=rows, summary=summary, classes=classes, out_html=html_path)

    print(f"Vendor CSV: {csv_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Review HTML: {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
