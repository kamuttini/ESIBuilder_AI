#!/usr/bin/env python3
"""Static HTML report of a needle-classifier run: metrics plus the images behind them.

Two jobs in one page. It shows how the model scores (metric tables, score
histogram, per-vendor breakdown) and it shows the actual crops, sorted so that
the most informative cases come first. Because the labels come from folder names,
the error sections double as a label-review queue: the top false positives are
usually needles filed outside an AGHI folder, and the top false negatives are
usually AGHI folders holding no needle at all.

Read-only by design: a single self-contained file with no endpoints and nothing
that saves on click, so it can be opened straight from disk with no side effects.
Thumbnails are embedded, so the page also works with the SSD unplugged.

Example:
  python3 tools/needle/build_needle_review_html.py \
    --predictions artifacts/91_needle_models/resnet18_256_cpu/predictions_test.csv \
    --crops-dir artifacts/90_needle_dataset/v1/crops \
    --output artifacts/91_needle_models/resnet18_256_cpu/report_test.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def thumb_data_uri(path: Path, size: int) -> str:
    try:
        with Image.open(path) as img:
            image = img.convert("RGB")
        image.thumbnail((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return ""


def metric_table(metrics: Dict[str, Dict[str, object]]) -> str:
    head = (
        "<tr><th></th><th>n</th><th>positivi</th><th>AP</th><th>AUC</th>"
        "<th>precision</th><th>recall</th><th>F1</th><th>accuracy</th></tr>"
    )
    body = []
    for name, m in metrics.items():
        body.append(
            f"<tr><td class='rowname'>{html.escape(name)}</td>"
            f"<td>{m['n']}</td><td>{m['n_pos']}</td>"
            f"<td class='num'>{m['ap']:.3f}</td><td class='num'>{m['auc']:.3f}</td>"
            f"<td class='num'>{m['precision']:.3f}</td><td class='num'>{m['recall']:.3f}</td>"
            f"<td class='num strong'>{m['f1']:.3f}</td><td class='num'>{m['accuracy']:.3f}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def histogram_svg(scores: np.ndarray, labels: np.ndarray, threshold: float,
                  bins: int = 40) -> str:
    """Overlaid score histograms, log-scaled: the classes differ by ~10x in size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    pos, _ = np.histogram(scores[labels == 1], bins=edges)
    neg, _ = np.histogram(scores[labels == 0], bins=edges)
    peak = max(1, int(max(pos.max(), neg.max())))
    width, height, pad = 720, 220, 28

    def bar_height(value: int) -> float:
        if value <= 0:
            return 0.0
        return (np.log10(value + 1) / np.log10(peak + 1)) * (height - 2 * pad)

    bar_w = (width - 2 * pad) / bins
    parts = []
    for i in range(bins):
        x = pad + i * bar_w
        for value, color, opacity in ((neg[i], "#5b8def", 0.75), (pos[i], "#ff8a3d", 0.75)):
            h = bar_height(int(value))
            if h > 0:
                parts.append(
                    f'<rect x="{x:.1f}" y="{height - pad - h:.1f}" width="{bar_w - 0.8:.1f}" '
                    f'height="{h:.1f}" fill="{color}" opacity="{opacity}"/>'
                )
    tx = pad + threshold * (width - 2 * pad)
    parts.append(
        f'<line x1="{tx:.1f}" y1="{pad - 8}" x2="{tx:.1f}" y2="{height - pad}" '
        f'stroke="#eee" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{tx + 5:.1f}" y="{pad - 12}" fill="#eee" font-size="11">soglia {threshold:.3f}</text>'
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = pad + frac * (width - 2 * pad)
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad + 15}" fill="#999" font-size="10" '
            f'text-anchor="middle">{frac:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Distribuzione dei punteggi per classe">'
        f'<rect width="{width}" height="{height}" fill="#191919" rx="8"/>{"".join(parts)}</svg>'
        f'<p class="legend"><span class="sw" style="background:#5b8def"></span> negativi '
        f'<span class="sw" style="background:#ff8a3d"></span> positivi '
        f'&middot; altezza in scala logaritmica</p>'
    )


def gallery(rows: Sequence[Dict[str, str]], crops_dir: Path, index: Dict[str, str],
            size: int) -> str:
    cards = []
    for row in rows:
        uri = thumb_data_uri(crops_dir / index[row["rel_path"]], size)
        label_txt = "AGHI" if row["label"] == "1" else "non-aghi"
        cards.append(
            f'<figure><img loading="lazy" src="{uri}" alt="">'
            f'<figcaption><b>{float(row["score"]):.3f}</b> &middot; etichetta '
            f'<span class="lab l{row["label"]}">{label_txt}</span> &middot; '
            f'{html.escape(row["vendor"])}'
            f'<br><span class="path">{html.escape(row["leaf_dir"])}</span></figcaption></figure>'
        )
    return f'<div class="grid">{"".join(cards)}</div>'


def section(title: str, note: str, rows: Sequence[Dict[str, str]], crops_dir: Path,
            index: Dict[str, str], size: int, total: Optional[int] = None) -> str:
    count = total if total is not None else len(rows)
    shown = "" if count == len(rows) else f" &mdash; mostrate le prime {len(rows)}"
    return (
        f'<h2>{html.escape(title)} <small>({count}{shown})</small></h2>'
        f'<p class="note">{html.escape(note)}</p>'
        + (gallery(rows, crops_dir, index, size) if rows else '<p class="note">Nessun caso.</p>')
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Default: metrics.json next to the predictions file.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top", type=int, default=24, help="Images shown per section.")
    parser.add_argument("--thumb-size", type=int, default=300)
    parser.add_argument("--sensitivity-exclude-regex", type=str, default="guid")
    args = parser.parse_args()

    metrics_path = args.metrics or (args.predictions.parent / "metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    threshold = args.threshold
    if threshold is None:
        threshold = float(metrics.get("threshold_from_val", 0.5))

    index: Dict[str, str] = {}
    with (args.crops_dir / "manifest_crops.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index[row["rel_path"]] = row["crop_path"]

    rows = [
        row for row in csv.DictReader(args.predictions.open(encoding="utf-8", newline=""))
        if row["rel_path"] in index
    ]
    if not rows:
        raise SystemExit("nessuna predizione utilizzabile")

    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_needle_classifier import binary_metrics  # noqa: E402

    split = args.predictions.stem.replace("predictions_", "")
    tables: Dict[str, Dict[str, object]] = {
        f"{split} \u2014 per immagine": binary_metrics(labels, scores, threshold)
    }
    pattern = re.compile(args.sensitivity_exclude_regex, re.IGNORECASE) if args.sensitivity_exclude_regex else None
    if pattern is not None:
        keep = np.array([i for i, r in enumerate(rows) if not pattern.search(r["leaf_dir"])])
        if keep.size and keep.size < len(rows):
            tables[f'{split} \u2014 senza cartelle "{args.sensitivity_exclude_regex}"'] = binary_metrics(
                labels[keep], scores[keep], threshold
            )

    # per-vendor, only where there is something to measure
    vendor_rows: Dict[str, Dict[str, object]] = {}
    for vendor in sorted({r["vendor"] for r in rows}):
        idx = np.array([i for i, r in enumerate(rows) if r["vendor"] == vendor])
        if idx.size >= 10 and labels[idx].sum() > 0:
            vendor_rows[vendor] = binary_metrics(labels[idx], scores[idx], threshold)

    false_pos = sorted(
        (r for r in rows if r["label"] == "0" and float(r["score"]) >= threshold),
        key=lambda r: -float(r["score"]),
    )
    false_neg = sorted(
        (r for r in rows if r["label"] == "1" and float(r["score"]) < threshold),
        key=lambda r: float(r["score"]),
    )
    true_pos = sorted(
        (r for r in rows if r["label"] == "1" and float(r["score"]) >= threshold),
        key=lambda r: -float(r["score"]),
    )
    true_neg = sorted(
        (r for r in rows if r["label"] == "0" and float(r["score"]) < threshold),
        key=lambda r: float(r["score"]),
    )
    borderline = sorted(rows, key=lambda r: abs(float(r["score"]) - threshold))

    policy = metrics.get("review_policy_from_val", {})
    policy_html = ""
    if policy:
        policy_html = (
            f'<p class="note">Policy a due soglie calibrata su validation: '
            f'accettato sopra {policy.get("accept_above", float("nan")):.3f}, '
            f'rifiutato sotto {policy.get("reject_below", float("nan")):.3f}, '
            f'da rivedere in mezzo ({policy.get("review_rate", float("nan")):.0%} delle immagini).</p>'
        )

    body = f"""
<h2>Metriche</h2>
{metric_table(tables)}
{policy_html}

<h2>Distribuzione dei punteggi</h2>
{histogram_svg(scores, labels, threshold)}

<h2>Per vendor</h2>
{metric_table(vendor_rows)}

{section("Falsi positivi", "Classificate come aghi ma fuori da una cartella AGHI, dalla piu' sicura. Spesso e' l'etichetta a sbagliare: sono acquisizioni con aghi archiviate altrove.", false_pos[: args.top], args.crops_dir, index, args.thumb_size, len(false_pos))}

{section("Falsi negativi", "Dentro una cartella AGHI ma classificate come normali, dalla piu' sicura. Spesso sono overlay di guida aghi, schermate di menu o frame senza ago visibile.", false_neg[: args.top], args.crops_dir, index, args.thumb_size, len(false_neg))}

{section("Casi al confine", "Le immagini con punteggio piu' vicino alla soglia: sono quelle che finirebbero in revisione manuale.", borderline[: args.top], args.crops_dir, index, args.thumb_size, len(rows))}

{section("Positivi riconosciuti", "Controprova: cosa il modello considera con sicurezza un'acquisizione con aghi.", true_pos[: args.top], args.crops_dir, index, args.thumb_size, len(true_pos))}

{section("Negativi riconosciuti", "Controprova: cosa il modello considera con sicurezza un'acquisizione normale.", true_neg[: args.top], args.crops_dir, index, args.thumb_size, len(true_neg))}
"""

    document = f"""<!doctype html>
<html lang="it"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classificatore aghi &mdash; {html.escape(split)}</title>
<style>
 body {{ font: 14px/1.6 -apple-system, system-ui, sans-serif; margin: 0 auto; padding: 28px;
        max-width: 1200px; background: #121212; color: #e9e9e9; }}
 h1 {{ font-size: 22px; margin-bottom: 4px; }}
 h2 {{ font-size: 17px; margin-top: 38px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
 h2 small {{ color: #888; font-weight: normal; }}
 .note {{ color: #a6a6a6; max-width: 78ch; }}
 .sub {{ color: #888; margin-top: 0; }}
 table {{ border-collapse: collapse; margin: 14px 0; font-variant-numeric: tabular-nums; }}
 th, td {{ padding: 5px 12px; border-bottom: 1px solid #2b2b2b; text-align: right; }}
 th {{ color: #999; font-weight: 500; font-size: 12px; }}
 td.rowname, th:first-child {{ text-align: left; }}
 td.strong {{ color: #fff; font-weight: 600; }}
 .legend {{ color: #999; font-size: 12px; }}
 .sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px;
        vertical-align: -1px; margin: 0 3px 0 10px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 14px;
          margin-top: 14px; }}
 figure {{ margin: 0; background: #1d1d1d; border-radius: 8px; padding: 8px; }}
 img {{ width: 100%; display: block; border-radius: 4px; background: #000; }}
 figcaption {{ font-size: 12px; color: #c8c8c8; margin-top: 7px; }}
 .lab {{ padding: 1px 6px; border-radius: 4px; font-size: 11px; }}
 .l1 {{ background: #6b3a16; color: #ffcda5; }}
 .l0 {{ background: #1f3a63; color: #bcd5ff; }}
 .path {{ color: #7d7d7d; word-break: break-all; font-size: 11px; }}
</style>
<h1>Classificatore acquisizioni con aghi</h1>
<p class="sub">Split <b>{html.escape(split)}</b> &middot; modello
{html.escape(str(metrics.get("arch", "?")))} @ {html.escape(str(metrics.get("image_size", "?")))}px
&middot; soglia {threshold:.3f} scelta su validation.
Le etichette derivano dal nome della cartella: le sezioni di errore servono anche a trovare
le etichette sbagliate, non solo a giudicare il modello.</p>
{body}
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1048576
    print(
        f"report: {args.output}  ({size_mb:.1f} MB, soglia {threshold:.3f}, "
        f"FP {len(false_pos)}, FN {len(false_neg)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
