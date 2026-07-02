#!/usr/bin/env python3
"""Build an index page for RECT_DEPTH experiment runs."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from build_rect_depth_autonomous_review_html import build_records as build_autonomous_review_records


PREFERRED_HTML = (
    "autonomous_review.html",
    "ranker_review.html",
    "depth_dataset_review_all.html",
    "postprocess_v3_compare.html",
    "visual_results_with_ocr.html",
    "retrain_vendor_report.html",
    "test_review.html",
)


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _max_mtime(path: Path) -> float:
    newest = path.stat().st_mtime if path.exists() else 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    newest = max(newest, (Path(dirpath) / filename).stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass
    return newest


def _short_path(path_text: object, max_len: int = 120) -> str:
    text = str(path_text or "")
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def _counts_text(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    parts = []
    for key, count in value.items():
        if str(key).strip():
            parts.append(f"{key}: {count}")
    return " · ".join(parts)


def _find_primary_html(run_dir: Path) -> Optional[Path]:
    for name in PREFERRED_HTML:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    htmls = sorted(run_dir.glob("*.html"), key=lambda item: item.name)
    return htmls[0] if htmls else None


def _ensure_autonomous_review_html(run_dir: Path, max_candidates: int) -> Optional[Path]:
    pred_csv = run_dir / "rect_depth_autonomous_predictions.csv"
    cand_csv = run_dir / "rect_depth_autonomous_candidates.csv"
    if not pred_csv.exists() or not cand_csv.exists():
        return None
    output_html = run_dir / "autonomous_review.html"
    if output_html.exists():
        return output_html
    try:
        build_autonomous_review_records(run_dir.resolve(), output_html.resolve(), max_candidates=max_candidates)
    except Exception:
        return None
    return output_html if output_html.exists() else None


def _link(target: Optional[Path], output_dir: Path, label: str, class_name: str = "") -> str:
    if not target or not target.exists():
        return ""
    rel = os.path.relpath(target, output_dir)
    cls = f' class="{html.escape(class_name)}"' if class_name else ""
    return f'<a{cls} href="{html.escape(rel)}">{html.escape(label)}</a>'


def _run_records(root: Path, output_html: Path, *, generate_missing_reviews: bool, max_candidates: int) -> List[Dict[str, object]]:
    output_dir = output_html.parent
    records: List[Dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        run_dir = summary_path.parent
        data = _read_json(summary_path)
        mtime = _max_mtime(run_dir)
        context = data.get("context") if isinstance(data.get("context"), dict) else {}
        profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
        vendor = str((context or {}).get("vendor") or (profile or {}).get("vendor") or "")
        probe = str((context or {}).get("probe") or (profile or {}).get("probe") or "")
        primary_html = _find_primary_html(run_dir)
        if primary_html is None and generate_missing_reviews:
            primary_html = _ensure_autonomous_review_html(run_dir, max_candidates=max_candidates)
        prediction_csv = Path(str(data.get("prediction_csv") or "")) if data.get("prediction_csv") else None
        candidate_csv = Path(str(data.get("candidate_csv") or "")) if data.get("candidate_csv") else None
        if prediction_csv and not prediction_csv.exists():
            prediction_csv = run_dir / "rect_depth_autonomous_predictions.csv"
        if candidate_csv and not candidate_csv.exists():
            candidate_csv = run_dir / "rect_depth_autonomous_candidates.csv"
        records.append(
            {
                "mtime": mtime,
                "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "run": run_dir.relative_to(root).as_posix(),
                "folder": str(data.get("folder") or ""),
                "vendor": vendor,
                "probe": probe,
                "images": data.get("image_count") or data.get("records") or "",
                "status": _counts_text(data.get("prediction_status") or data.get("status_counts")),
                "mode": _counts_text(data.get("prediction_modes") or data.get("mode_counts")),
                "strategy": _counts_text(data.get("folder_strategy")),
                "notes": str(data.get("notes") or ""),
                "review_link": _link(primary_html, output_dir, "Apri review", "review-btn"),
                "summary_link": _link(summary_path, output_dir, "summary"),
                "pred_link": _link(prediction_csv, output_dir, "predictions"),
                "cand_link": _link(candidate_csv, output_dir, "candidates"),
            }
        )
    records.sort(key=lambda item: float(item["mtime"]), reverse=True)
    return records


def _render(records: Sequence[Dict[str, object]], root: Path) -> str:
    rows = []
    for idx, item in enumerate(records, start=1):
        links = " ".join(
            part
            for part in (
                str(item["review_link"]),
                str(item["summary_link"]),
                str(item["pred_link"]),
                str(item["cand_link"]),
            )
            if part
        )
        search_blob = " ".join(
            str(item.get(key) or "")
            for key in ("run", "folder", "vendor", "probe", "status", "mode", "strategy", "notes")
        ).lower()
        rows.append(
            "<tr data-search=\"{search}\">"
            "<td class=\"idx\">{idx}</td>"
            "<td class=\"date\">{date}</td>"
            "<td><strong>{run}</strong><div class=\"muted\">{folder}</div></td>"
            "<td>{vendor}<div class=\"muted\">{probe}</div></td>"
            "<td class=\"num\">{images}</td>"
            "<td>{status}</td>"
            "<td>{mode}</td>"
            "<td>{strategy}</td>"
            "<td>{links}<div class=\"muted notes\">{notes}</div></td>"
            "</tr>".format(
                search=html.escape(search_blob, quote=True),
                idx=idx,
                date=html.escape(str(item["date"])),
                run=html.escape(str(item["run"])),
                folder=html.escape(_short_path(item["folder"])),
                vendor=html.escape(str(item["vendor"])),
                probe=html.escape(str(item["probe"])),
                images=html.escape(str(item["images"])),
                status=html.escape(str(item["status"])),
                mode=html.escape(str(item["mode"])),
                strategy=html.escape(str(item["strategy"])),
                links=links,
                notes=html.escape(_short_path(item["notes"], 180)),
            )
        )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    page = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RECT_DEPTH run index</title>
  <style>
    :root { color-scheme: dark; --bg:#101315; --panel:#171b1f; --line:#2a3238; --text:#e8edf1; --muted:#9aa7b1; --accent:#6ee7b7; --warn:#facc15; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { position:sticky; top:0; z-index:5; background:rgba(16,19,21,.96); border-bottom:1px solid var(--line); padding:14px 18px; display:grid; gap:10px; }
    h1 { margin:0; font-size:20px; font-weight:650; letter-spacing:0; }
    .meta { color:var(--muted); display:flex; gap:14px; flex-wrap:wrap; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    input, select { background:#0c0f11; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-height:36px; }
    input { min-width:min(520px, 100%); flex:1; }
    main { padding:16px 18px 28px; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
    th, td { border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top; text-align:left; }
    th { position:sticky; top:112px; background:#15191d; z-index:4; color:#cbd5dd; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    tr:hover td { background:#1b2126; }
    a { color:var(--accent); text-decoration:none; margin-right:10px; white-space:nowrap; }
    a:hover { text-decoration:underline; }
    .review-btn { display:inline-flex; align-items:center; min-height:30px; padding:5px 10px; margin:0 8px 7px 0; border-radius:6px; background:#12352d; border:1px solid #1f6f5a; color:#b8f7df; font-weight:650; }
    .review-btn:hover { background:#174638; text-decoration:none; }
    .idx, .num { text-align:right; color:var(--muted); white-space:nowrap; }
    .date { white-space:nowrap; color:#d5dde4; }
    .muted { color:var(--muted); font-size:12px; margin-top:3px; }
    .notes { max-width:360px; }
    .empty { display:none; padding:28px; color:var(--warn); }
    @media (max-width: 980px) {
      table, thead, tbody, tr, th, td { display:block; }
      thead { display:none; }
      tr { border-bottom:1px solid var(--line); padding:10px; }
      td { border:0; padding:4px 0; }
      .idx { text-align:left; }
      .date { white-space:normal; }
    }
  </style>
</head>
<body>
  <header>
    <h1>RECT_DEPTH run index</h1>
    <div class="meta">
      <span><strong id="shown">__COUNT__</strong> / __COUNT__ run</span>
      <span>Root: __ROOT__</span>
      <span>Generato: __GENERATED__</span>
    </div>
    <div class="controls">
      <input id="q" type="search" placeholder="Cerca per vendor, cartella, nome run, status...">
      <select id="statusFilter" aria-label="Filtro status">
        <option value="">Tutti gli status</option>
        <option value="reject">Con reject</option>
        <option value="review">Con review</option>
        <option value="accepted">Con accepted</option>
      </select>
    </div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Data</th>
          <th>Run / cartella</th>
          <th>Vendor</th>
          <th>Img</th>
          <th>Status</th>
          <th>Metodo</th>
          <th>Strategia</th>
          <th>Link</th>
        </tr>
      </thead>
      <tbody id="rows">
        __ROWS__
      </tbody>
    </table>
    <div id="empty" class="empty">Nessuna run corrisponde ai filtri.</div>
  </main>
  <script>
    const q = document.getElementById('q');
    const statusFilter = document.getElementById('statusFilter');
    const rows = Array.from(document.querySelectorAll('#rows tr'));
    const shown = document.getElementById('shown');
    const empty = document.getElementById('empty');
    function applyFilters() {
      const needle = q.value.trim().toLowerCase();
      const status = statusFilter.value;
      let visible = 0;
      rows.forEach(row => {
        const hay = row.dataset.search || '';
        const okText = !needle || hay.includes(needle);
        const okStatus = !status || hay.includes(status);
        const ok = okText && okStatus;
        row.style.display = ok ? '' : 'none';
        if (ok) visible += 1;
      });
      shown.textContent = String(visible);
      empty.style.display = visible ? 'none' : 'block';
    }
    q.addEventListener('input', applyFilters);
    statusFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    return (
        page.replace("__COUNT__", str(len(records)))
        .replace("__ROOT__", html.escape(root.as_posix()))
        .replace("__GENERATED__", html.escape(generated))
        .replace("__ROWS__", "\n".join(rows))
    )


def build_index(root: Path, output_html: Path, generate_missing_reviews: bool, max_candidates: int) -> Dict[str, object]:
    root = root.expanduser().resolve()
    output_html = output_html.expanduser().resolve()
    records = _run_records(
        root,
        output_html,
        generate_missing_reviews=generate_missing_reviews,
        max_candidates=max_candidates,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render(records, root), encoding="utf-8")
    return {"output_html": output_html.as_posix(), "runs": len(records), "root": root.as_posix()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an HTML index of RECT_DEPTH runs.")
    parser.add_argument("--root", type=Path, default=Path("artifacts/24_rect_depth_hybrid"))
    parser.add_argument("--output-html", type=Path, default=Path("artifacts/24_rect_depth_hybrid/run_index.html"))
    parser.add_argument("--max-candidates", type=int, default=8, help="Candidates shown when missing autonomous review pages are generated.")
    parser.add_argument("--no-generate-missing-reviews", action="store_false", dest="generate_missing_reviews", default=True)
    return parser


def main() -> None:
    print(json.dumps(build_index(**vars(build_parser().parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
