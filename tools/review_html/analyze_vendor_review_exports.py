#!/usr/bin/env python3
"""Analyze vendor review exports against vendor predictions."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:  # noqa: BLE001
        return default


def _select_subset(paths: Sequence[str], limit: int) -> List[str]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]
    total = len(paths)
    selected_idx: List[int] = []
    seen: set[int] = set()
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)
    if len(selected_idx) < limit:
        for idx in range(total):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= limit:
                break
    selected_idx.sort()
    return [paths[i] for i in selected_idx]


def _load_vendor_by_group(path: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            g = str(row.get("group_name", "")).strip()
            if not g:
                continue
            out[g] = {
                "group_name": g,
                "num_images": _to_int(row.get("num_images", 0)),
                "vendor_predicted": str(row.get("vendor_predicted", "")).strip(),
                "vendor_top1_prob": _to_float(row.get("vendor_top1_prob", 0.0)),
                "vendor_margin_top1_top2": _to_float(row.get("vendor_margin_top1_top2", 0.0)),
                "vendor_vote_ratio": _to_float(row.get("vendor_vote_ratio", 0.0)),
                "vendor_images_used": _to_int(row.get("vendor_images_used", 0)),
            }
    return out


def _load_group_corrections(flags_csv: Path, corrections_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with flags_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            g = str(row.get("group_name", "")).strip()
            if not g:
                continue
            out[g] = {
                "group_name": g,
                "vendor_predicted_export": str(row.get("vendor_predicted", "")).strip(),
                "correction_vendor": str(row.get("correction_vendor", "")).strip(),
                "note": str(row.get("note", "")).strip(),
            }

    if corrections_csv is not None and corrections_csv.exists():
        with corrections_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                g = str(row.get("group_name", "")).strip()
                if not g:
                    continue
                current = out.setdefault(
                    g,
                    {
                        "group_name": g,
                        "vendor_predicted_export": str(row.get("vendor_predicted", "")).strip(),
                        "correction_vendor": "",
                        "note": "",
                    },
                )
                corr = str(row.get("correction_vendor", "")).strip()
                if corr:
                    current["correction_vendor"] = corr
                corr_note = str(row.get("note", "")).strip()
                if corr_note:
                    current["note"] = corr_note
    return out


def _load_group_images_from_predictions(predictions_csv: Path) -> Dict[str, List[str]]:
    by_group: Dict[str, List[str]] = defaultdict(list)
    with predictions_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            g = str(row.get("group_name", "")).strip()
            p = str(row.get("image_path", "")).strip()
            if not g or not p:
                continue
            by_group[g].append(p)
    for g in list(by_group.keys()):
        by_group[g] = sorted(set(by_group[g]))
    return by_group


def _reason(prob: float, margin: float, vote: float, correction_vendor: str, pred_vendor: str) -> str:
    low_prob = prob < 0.50
    low_margin = margin < 0.10
    low_vote = vote < 0.60
    if correction_vendor == "UNKNOWN":
        return "ambiguous_or_unknown_vendor"
    if not correction_vendor:
        if low_prob or low_margin or low_vote:
            return "flagged_without_correction_low_conf"
        return "flagged_without_correction"
    if pred_vendor and correction_vendor == pred_vendor:
        if low_prob or low_margin or low_vote:
            return "correct_prediction_but_low_confidence"
        return "correct_prediction"
    if low_prob or low_margin or low_vote:
        return "low_confidence_prediction"
    return "high_confidence_misclassification"


def _safe_stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(vals)
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analizza export vendor_flags/vendor_corrections.")
    p.add_argument("--vendor-by-group-csv", type=Path, required=True)
    p.add_argument("--vendor-flags-csv", type=Path, required=True)
    p.add_argument("--vendor-corrections-csv", type=Path, default=None)
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--sample-per-group", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/40_outputs_eval/vendor_review_exports_analysis"))
    return p


def main() -> int:
    args = _build_parser().parse_args()
    vendor_by_group_csv = args.vendor_by_group_csv.expanduser().resolve()
    vendor_flags_csv = args.vendor_flags_csv.expanduser().resolve()
    vendor_corrections_csv = args.vendor_corrections_csv.expanduser().resolve() if args.vendor_corrections_csv else None
    predictions_csv = args.predictions_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not vendor_by_group_csv.exists():
        raise FileNotFoundError(f"Missing vendor-by-group csv: {vendor_by_group_csv}")
    if not vendor_flags_csv.exists():
        raise FileNotFoundError(f"Missing vendor-flags csv: {vendor_flags_csv}")
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions csv: {predictions_csv}")

    vbg = _load_vendor_by_group(vendor_by_group_csv)
    corrections = _load_group_corrections(vendor_flags_csv, vendor_corrections_csv)
    images_by_group = _load_group_images_from_predictions(predictions_csv)

    rows: List[Dict[str, object]] = []
    confusion: Counter[str] = Counter()
    correction_dist: Counter[str] = Counter()
    prob_vals: List[float] = []
    margin_vals: List[float] = []
    vote_vals: List[float] = []

    for g in sorted(corrections.keys(), key=lambda x: x.lower()):
        base = vbg.get(g, {})
        pred_vendor = str(base.get("vendor_predicted", corrections[g].get("vendor_predicted_export", ""))).strip()
        corr_vendor = str(corrections[g].get("correction_vendor", "")).strip()
        num_images = _to_int(base.get("num_images", 0))
        prob = _to_float(base.get("vendor_top1_prob", 0.0))
        margin = _to_float(base.get("vendor_margin_top1_top2", 0.0))
        vote = _to_float(base.get("vendor_vote_ratio", 0.0))
        used = _to_int(base.get("vendor_images_used", 0))

        sample_images = _select_subset(images_by_group.get(g, []), int(args.sample_per_group))
        reason = _reason(
            prob=prob,
            margin=margin,
            vote=vote,
            correction_vendor=corr_vendor,
            pred_vendor=pred_vendor,
        )

        row = {
            "group_name": g,
            "vendor_predicted": pred_vendor,
            "vendor_corrected": corr_vendor,
            "note": str(corrections[g].get("note", "")).strip(),
            "num_images": num_images,
            "vendor_images_used": used,
            "vendor_top1_prob": prob,
            "vendor_margin_top1_top2": margin,
            "vendor_vote_ratio": vote,
            "analysis_reason": reason,
            "sample_images_count": len(sample_images),
            "sample_images": " | ".join(sample_images),
        }
        rows.append(row)

        if corr_vendor:
            correction_dist[corr_vendor] += 1
        if pred_vendor and corr_vendor and corr_vendor != "UNKNOWN" and corr_vendor != pred_vendor:
            confusion[f"{pred_vendor} -> {corr_vendor}"] += 1

        prob_vals.append(prob)
        margin_vals.append(margin)
        vote_vals.append(vote)

    flagged_count = len(rows)
    reason_count = Counter(str(r["analysis_reason"]) for r in rows)
    summary = {
        "vendor_by_group_csv": vendor_by_group_csv.as_posix(),
        "vendor_flags_csv": vendor_flags_csv.as_posix(),
        "vendor_corrections_csv": vendor_corrections_csv.as_posix() if vendor_corrections_csv else "",
        "predictions_csv": predictions_csv.as_posix(),
        "flagged_groups": flagged_count,
        "reason_counts": dict(sorted(reason_count.items())),
        "correction_vendor_distribution": dict(sorted(correction_dist.items())),
        "top_confusions_pred_to_corrected": [
            {"pair": k, "count": c} for k, c in confusion.most_common(20)
        ],
        "stats_vendor_top1_prob_on_flagged": _safe_stats(prob_vals),
        "stats_vendor_margin_on_flagged": _safe_stats(margin_vals),
        "stats_vendor_vote_ratio_on_flagged": _safe_stats(vote_vals),
        "sample_per_group_reconstructed": int(args.sample_per_group),
    }

    out_csv = output_dir / "vendor_flagged_analysis.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "group_name",
                "vendor_predicted",
                "vendor_corrected",
                "note",
                "num_images",
                "vendor_images_used",
                "vendor_top1_prob",
                "vendor_margin_top1_top2",
                "vendor_vote_ratio",
                "analysis_reason",
                "sample_images_count",
                "sample_images",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    tr_rows: List[str] = []
    for row in rows:
        sample_html: List[str] = []
        for p in str(row["sample_images"]).split(" | "):
            p = p.strip()
            if not p:
                continue
            uri = Path(p).expanduser().resolve().as_uri()
            sample_html.append(
                f'<a href="{html.escape(uri)}" target="_blank"><img src="{html.escape(uri)}" loading="lazy"/></a>'
            )
        group_name = str(row["group_name"])
        vendor_pred = str(row["vendor_predicted"])
        vendor_corr = str(row["vendor_corrected"])
        note = str(row["note"])
        tr_rows.append(
            f'<tr data-group-name="{html.escape(group_name, quote=True)}" '
            f'data-vendor-predicted="{html.escape(vendor_pred, quote=True)}" '
            f'data-vendor-corrected="{html.escape(vendor_corr, quote=True)}">'
            f"<td>{html.escape(group_name)}</td>"
            f"<td>{html.escape(vendor_pred)}</td>"
            f"<td>{html.escape(vendor_corr)}</td>"
            f"<td><input type='text' class='note-input' value='{html.escape(note, quote=True)}' "
            f"placeholder='nota analisi (cause, pattern, azione)'/></td>"
            f"<td>{int(row['num_images'])}</td>"
            f"<td>{int(row['vendor_images_used'])}</td>"
            f"<td>{_to_float(row['vendor_top1_prob']):.3f}</td>"
            f"<td>{_to_float(row['vendor_margin_top1_top2']):.3f}</td>"
            f"<td>{_to_float(row['vendor_vote_ratio']):.3f}</td>"
            f"<td>{html.escape(str(row['analysis_reason']))}</td>"
            f"<td>{''.join(sample_html)}</td>"
            "</tr>"
        )

    out_html = output_dir / "vendor_review_analysis.html"
    out_html.write_text(
        f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Vendor Review Analysis</title>
  <style>
    body {{ font-family: "IBM Plex Sans","Segoe UI",sans-serif; margin:0; background:#f3f5f4; color:#102119; }}
    .wrap {{ padding:16px; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 8;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      padding: 10px;
      border: 1px solid #d6e2db;
      border-radius: 10px;
      background: rgba(243, 245, 244, 0.96);
      margin-bottom: 12px;
      backdrop-filter: blur(4px);
    }}
    button {{
      border: 1px solid #c7d5cd;
      background: #ffffff;
      color: #102119;
      border-radius: 8px;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12px;
    }}
    button.primary {{
      background: #0f8f64;
      border-color: #0f8f64;
      color: #fff;
    }}
    pre {{ background:#0f241a; color:#d8f6e8; border-radius:10px; padding:12px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #d6e2db; }}
    th,td {{ border:1px solid #e5ece8; padding:6px; font-size:12px; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#eef3ef; }}
    img {{ width:128px; margin:2px; border:1px solid #d6e2db; border-radius:6px; }}
    .note-input {{
      width: 260px;
      max-width: 100%;
      border: 1px solid #c7d5cd;
      border-radius: 7px;
      padding: 6px 8px;
      font-size: 12px;
      color: #102119;
      background: #fdfffe;
      box-sizing: border-box;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Analisi Errori Vendor (export review)</h1>
    <div class="toolbar">
      <button id="btn-export-notes" class="primary">Esporta note CSV</button>
      <button id="btn-clear-notes">Reset note locali</button>
      <span id="notes-status"></span>
    </div>
    <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
    <table>
      <thead>
        <tr>
          <th>group_name</th><th>pred</th><th>corr</th><th>note</th><th>#img</th><th>#used</th>
          <th>p</th><th>margin</th><th>vote</th><th>reason</th><th>sample images</th>
        </tr>
      </thead>
      <tbody>{''.join(tr_rows)}</tbody>
    </table>
  </div>
  <script>
    const STORAGE_KEY = `vendor-analysis-notes::${{window.location.pathname}}`;
    const rows = Array.from(document.querySelectorAll('tbody tr[data-group-name]'));

    function loadNotes() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {{}};
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return {{}};
        return parsed;
      }} catch (_err) {{
        return {{}};
      }}
    }}

    function saveNotes(map) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    }}

    function updateStatus(map) {{
      const used = Object.values(map).filter((v) => String(v || '').trim().length > 0).length;
      document.getElementById('notes-status').textContent = `note compilate: ${{used}} / ${{rows.length}}`;
    }}

    function csvEscape(value) {{
      return `"${{String(value ?? '').replaceAll('"', '""')}}"`;
    }}

    function downloadCsv(filename, rows2d) {{
      const csv = rows2d.map((r) => r.map(csvEscape).join(',')).join('\\n');
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

    const noteMap = loadNotes();
    for (const tr of rows) {{
      const groupName = tr.dataset.groupName || '';
      const input = tr.querySelector('.note-input');
      if (!input) continue;
      if (groupName in noteMap) {{
        input.value = String(noteMap[groupName] || '');
      }}
      input.addEventListener('input', () => {{
        noteMap[groupName] = input.value || '';
        saveNotes(noteMap);
        updateStatus(noteMap);
      }});
    }}
    updateStatus(noteMap);

    document.getElementById('btn-export-notes').addEventListener('click', () => {{
      const out = [['group_name','vendor_predicted','vendor_corrected','note']];
      for (const tr of rows) {{
        const groupName = tr.dataset.groupName || '';
        const pred = tr.dataset.vendorPredicted || '';
        const corr = tr.dataset.vendorCorrected || '';
        const input = tr.querySelector('.note-input');
        const note = (input ? input.value : '').trim();
        if (note) {{
          out.push([groupName, pred, corr, note]);
        }}
      }}
      downloadCsv('vendor_analysis_notes.csv', out);
    }});

    document.getElementById('btn-clear-notes').addEventListener('click', () => {{
      if (!confirm('Confermi reset completo note locali in questo report?')) return;
      localStorage.removeItem(STORAGE_KEY);
      rows.forEach((tr) => {{
        const input = tr.querySelector('.note-input');
        if (input) input.value = '';
      }});
      updateStatus({{}});
    }});
  </script>
</body>
</html>""",
        encoding="utf-8",
    )

    print(f"Analysis CSV: {out_csv}", flush=True)
    print(f"Summary JSON: {out_json}", flush=True)
    print(f"Analysis HTML: {out_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
