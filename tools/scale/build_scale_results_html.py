"""Build an HTML results page for the scale block: metrics + label/unit evidence.

Two sections:
  1. the chain-eval numbers of the current run vs a baseline run;
  2. an evidence gallery of the scale column at several zoom levels, with the GT tick
     values overlaid and the OCR readings listed — this is what tells us whether a wrong
     calibration comes from the OCR reading the wrong glyphs, from the wrong unit, or from
     picking numbers that are not scale labels at all.

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/build_scale_results_html.py \
    --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
    --baseline artifacts/41_scale_chain_eval/senza_rete \
    --current  artifacts/41_scale_chain_eval/senza_rete_v7_monotonic \
    --vendor Esaote --n 12 \
    --out artifacts/43_scale_results/index.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scale.detect_scale_ladder import detect_scale, load_image  # noqa: E402


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _b64(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _svg_overlay(
    vw: int,
    vh: int,
    s: float,
    ox: float,
    oy: float,
    gx: float,
    gy: float,
    gt_marks: List[Tuple[float, float]],
    reads_y: List[float],
    last_tick_y: Optional[float],
    font: int = 11,
) -> str:
    """The annotation as an SVG layer over the *untouched* image.

    Drawing it as vector rather than burning it into the JPEG is what lets the page toggle
    the overlay off and show the original frame, and it keeps the marks crisp at any zoom.
    Coordinates come in original-frame pixels and are mapped with ``(v - o) * s``.
    """
    X = (gx - ox) * s
    p: List[str] = [f'<svg class="ovl" viewBox="0 0 {vw} {vh}" preserveAspectRatio="none">']
    p.append(f'<line x1="{X:.1f}" y1="0" x2="{X:.1f}" y2="{vh}" stroke="#00b4ff" '
             f'stroke-width="1" stroke-dasharray="4 5" opacity="0.75"/>')
    for y, v_cm in gt_marks:
        Y = (y - oy) * s
        p.append(f'<line x1="{X-9:.1f}" y1="{Y:.1f}" x2="{X+9:.1f}" y2="{Y:.1f}" '
                 f'stroke="#00dc00" stroke-width="1"/>')
        p.append(f'<text x="{X+12:.1f}" y="{Y+4:.1f}" fill="#00dc00" '
                 f'font-size="{font}">{v_cm:g}</text>')
    p.append(f'<circle cx="{X:.1f}" cy="{(gy-oy)*s:.1f}" r="6" fill="none" '
             f'stroke="#00dc00" stroke-width="2"/>')
    for y in reads_y:
        p.append(f'<circle cx="{X:.1f}" cy="{(y-oy)*s:.1f}" r="5" fill="none" '
                 f'stroke="#ff5000" stroke-width="2"/>')
    if last_tick_y is not None:
        p.append(f'<circle cx="{X:.1f}" cy="{(last_tick_y-oy)*s:.1f}" r="7" '
                 f'fill="#a000c8" stroke="#fff" stroke-width="1"/>')
    p.append("</svg>")
    return "".join(p)


def build_case(row: Dict[str, str], zoom: float = 2.0) -> Optional[dict]:
    """Crop the scale column, overlay the GT tick values, collect the OCR readings."""
    img = load_image(row["image_path"])
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    gx, gy = _f(row["x"]), _f(row["y_zero"])
    mmpp, tick_cm = _f(row["mm_per_px_from_scale"]), _f(row["tick_cm"]) or 0.5
    if gx is None or gy is None or not mmpp:
        return None

    pred = detect_scale(gray, rect=None, vendor=row["vendor"], prior_x=gx, prior_band_px=70)

    # Both images stay untouched; the annotation is a separate SVG layer that the page can
    # switch off to show the original frame.
    # Labels sit to the left of the ticks (legacy side = -1 on every row).
    x0 = max(0, int(gx) - 190)
    x1 = min(w, int(gx) + 70)
    crop = cv2.cvtColor(gray[:, x0:x1], cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)

    direction = 1 if str(row.get("zero_at", "top")) == "top" else -1
    gt_marks: List[Tuple[float, float]] = []
    k = 0
    while True:
        v_cm = k * tick_cm
        y = gy + direction * (v_cm * 10.0) / mmpp
        if y < 0 or y > h:
            break
        gt_marks.append((y, v_cm))
        k += 1
        if k > 60:
            break

    # Last tick found by the detector, at the *far* end of the ruler: with the zero at the
    # bottom (UD-flipped) the ruler runs upwards, so the far end is the top-most tick.
    last_tick = pred.y_last_tick if pred.direction >= 0 else pred.y_first_tick

    # OCR readings, and what the GT says should be printed at that y
    reads = []
    for y, v_mm in (pred.labels_all or []):
        expected_cm = abs(y - gy) * mmpp / 10.0
        reads.append({"y": round(y, 1), "read_cm": round(v_mm / 10.0, 2),
                      "expected_cm": round(expected_cm, 2)})
    reads_y = [y for y, _v in (pred.labels_all or [])]

    # full frame, downscaled for the page but still detailed enough to zoom into
    cap_w = 1500
    sf = min(1.0, cap_w / float(w))
    full = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if sf < 1.0:
        full = cv2.resize(full, (int(w * sf), int(h * sf)), interpolation=cv2.INTER_AREA)

    err = None
    if pred.ok and pred.mm_per_px:
        err = round(abs(pred.mm_per_px - mmpp) / mmpp, 3)
    return {
        "folder": row["config_folder"],
        "depth_index": row.get("depth_index", ""),
        "img": _b64(crop),
        "svg_crop": _svg_overlay(crop.shape[1], crop.shape[0], zoom, x0, 0.0,
                                 gx, gy, gt_marks, reads_y, last_tick, font=12),
        "img_full": _b64(full, quality=72),
        "svg_full": _svg_overlay(full.shape[1], full.shape[0], sf, 0.0, 0.0,
                                 gx, gy, gt_marks, reads_y, last_tick, font=11),
        "last_tick_y": round(last_tick, 1) if last_tick is not None else None,
        "n_ticks": pred.n_ticks,
        "gt_mm_per_px": round(mmpp, 4),
        "pred_mm_per_px": round(pred.mm_per_px, 4) if pred.mm_per_px else None,
        "rel_err": err,
        "span_cm": round((_f(row["length_mm"]) or 0) / 10.0, 1),
        "tick_cm": tick_cm,
        "n_gt_marks": len(gt_marks),
        "visible_labels": [f"{v:g}" for _y, v in gt_marks][:14],
        "status": pred.status,
        "reads": reads,
    }


def metrics_block(d: Path) -> Optional[dict]:
    p = d / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    m = s["metrics"]["overall"]
    n = s.get("rows_evaluated") or 1
    return {
        "name": d.name,
        "none_pct": round(100 * s["sources"].get("none", 0) / n, 1),
        "calib_on_acc": m.get("calib_ok_pct_on_accepted"),
        "strict_on_acc": m.get("strict_ok_pct_on_accepted"),
        "calib": m.get("calib_ok_pct"),
        "strict": m.get("strict_ok_pct"),
        "direction": m.get("direction_ok_pct"),
        "status": m.get("status"),
    }


def render(cases: List[dict], base: Optional[dict], cur: Optional[dict], vendor: str) -> str:
    def cmp_row(label, k, unit="%"):
        a = base.get(k) if base else None
        b = cur.get(k) if cur else None
        cls = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            cls = "up" if b > a else ("down" if b < a else "same")
        return (f"<tr><td>{label}</td><td>{a if a is not None else '-'}{unit if a is not None else ''}</td>"
                f"<td class='{cls}'>{b if b is not None else '-'}{unit if b is not None else ''}</td></tr>")

    cards = []
    for c in cases:
        reads = "".join(
            f"<tr><td>{r['y']}</td><td class='{'bad' if abs(r['read_cm']-r['expected_cm'])>0.3 else 'good'}'>"
            f"{r['read_cm']}</td><td>{r['expected_cm']}</td></tr>" for r in c["reads"]
        ) or "<tr><td colspan=3 class='muted'>nessuna lettura</td></tr>"
        badge = "bad" if (c["rel_err"] is not None and c["rel_err"] > 0.05) else "good"
        cards.append(f"""
        <article class="card">
          <div class="hd"><b>{c['folder'][:52]}</b> · d{c['depth_index']}
            <span class="pill {badge}">err {c['rel_err'] if c['rel_err'] is not None else 'n/d'}</span>
            <span class="pill">{c['status']}</span></div>
          <div class="viewer">
            <div class="stage">
              <img class="full" src="{c['img_full']}" alt="frame completo">
              {c['svg_full']}
            </div>
            <div class="zbar">
              <button data-z="out">−</button><span class="zlvl">100%</span><button data-z="in">+</button>
              <button data-z="reset">reset</button>
              <button class="tgl" data-on="1">nascondi elaborazioni</button>
              <span class="muted">rotella = zoom · trascina = sposta · doppio click = reset</span>
            </div>
          </div>
          <div class="body">
            <div class="cropwrap"><img class="cropimg" src="{c['img']}" alt="colonna del righello">{c['svg_crop']}</div>
            <div class="info">
              <p><b>mm/px</b> GT {c['gt_mm_per_px']} · pred {c['pred_mm_per_px']}</p>
              <p><b>span righello</b> {c['span_cm']} cm · tacche ogni {c['tick_cm']} cm</p>
              <p><b>ultima tacca rilevata</b> <span class="dotv"></span> y = {c['last_tick_y'] if c['last_tick_y'] is not None else 'n/d'}
                 · tacche trovate: {c['n_ticks']}</p>
              <p><b>etichette che il righello mostra</b> (da GT):<br>
                 <code>{', '.join(c['visible_labels']) or '-'}</code></p>
              <table class="reads"><tr><th>y</th><th>letto (cm)</th><th>atteso (cm)</th></tr>{reads}</table>
            </div>
          </div>
        </article>""")

    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blocco scala — risultati</title>
<style>
 :root{{--bg:#0f1115;--card:#171a21;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
        --good:#37d67a;--bad:#ff5470;--warn:#ffb020}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
 header{{padding:22px 26px;border-bottom:1px solid var(--line);background:#12141a}}
 h1{{margin:0 0 4px;font-size:20px}} h2{{font-size:16px;margin:26px 0 10px}}
 .wrap{{max-width:1180px;margin:0 auto;padding:0 26px 40px}}
 .muted{{color:var(--mut)}}
 table{{border-collapse:collapse;width:100%;max-width:620px;margin:8px 0 18px}}
 th,td{{border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;font-size:13px}}
 th{{color:var(--mut);font-weight:600}}
 td.up{{color:var(--good);font-weight:700}} td.down{{color:var(--bad);font-weight:700}}
 td.same{{color:var(--mut)}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        margin:14px 0;overflow:hidden}}
 .hd{{padding:9px 12px;border-bottom:1px solid var(--line);font-size:13px}}
 .body{{display:flex;gap:14px;padding:12px;flex-wrap:wrap}}
 .body img{{max-height:520px;border-radius:6px;background:#000}}
 .info{{flex:1;min-width:260px}} .info p{{margin:4px 0}}
 code{{background:#0c0e13;padding:2px 6px;border-radius:4px;color:var(--warn)}}
 .pill{{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);
        font-size:11px;margin-left:6px;color:var(--mut)}}
 .pill.good{{border-color:var(--good);color:var(--good)}}
 .pill.bad{{border-color:var(--bad);color:var(--bad)}}
 table.reads{{max-width:340px}}
 td.good{{color:var(--good)}} td.bad{{color:var(--bad);font-weight:700}}
 .viewer{{position:relative;overflow:hidden;background:#000;border-bottom:1px solid var(--line);
          max-height:76vh;cursor:grab}}
 .viewer.dragging{{cursor:grabbing}}
 .viewer .stage{{position:relative;transform-origin:0 0;transition:transform .05s linear}}
 .viewer img.full{{display:block;width:100%;height:auto}}
 svg.ovl{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
 .hideovl svg.ovl{{display:none}}
 .cropwrap{{position:relative;display:inline-block;line-height:0}}
 .zbar button.tgl{{border-color:#00dc00;color:#00dc00}}
 .zbar button.tgl[data-on="0"]{{border-color:var(--line);color:var(--mut)}}
 .zbar{{position:absolute;left:10px;bottom:10px;display:flex;gap:6px;align-items:center;
        background:rgba(10,12,17,.82);border:1px solid var(--line);border-radius:8px;
        padding:5px 9px;font-size:12px}}
 .zbar button{{background:#232732;color:var(--ink);border:1px solid var(--line);
               border-radius:6px;padding:2px 9px;cursor:pointer;font-size:13px}}
 .zbar .zlvl{{min-width:44px;text-align:center;color:var(--mut)}}
 .cropimg{{max-height:520px;border-radius:6px;background:#000}}
 .dotv{{display:inline-block;width:10px;height:10px;border-radius:50%;background:#a000c8;
        border:1px solid #fff;vertical-align:middle;margin:0 3px}}
 .legend span{{margin-right:14px}}
 .k{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px}}
</style></head><body>
<header>
  <h1>Blocco scala — risultati e indagine unità</h1>
  <div class="muted">Catena: rete heatmap → detector a tacche → OCR numeri → consenso di setup.
     Vendor in esame: <b>{vendor}</b>.</div>
  <div style="margin-top:10px">
    <button id="tgl-all" data-on="1">nascondi elaborazioni su tutte le immagini</button>
  </div>
</header>
<div class="wrap">

<h2>1. Effetto della selezione coerente delle etichette (chain eval, 425 righe)</h2>
<div class="muted">Sottoinsieme di letture OCR monotone in y e con passo tacche corroborato;
   le letture incoerenti vengono scartate.</div>
<table>
  <tr><th>metrica</th><th>baseline</th><th>attuale</th></tr>
  {cmp_row('calib_ok su accepted', 'calib_on_acc')}
  {cmp_row('strict_ok su accepted', 'strict_on_acc')}
  {cmp_row('calib_ok globale', 'calib')}
  {cmp_row('strict_ok globale', 'strict')}
  {cmp_row('direction_ok', 'direction')}
  {cmp_row('righe senza predizione (none)', 'none_pct')}
</table>
<div class="muted">Stato: baseline <code>{base['status'] if base else '-'}</code> →
   attuale <code>{cur['status'] if cur else '-'}</code></div>

<h2>2. Esito indagine unità (viste zoomate) — ipotesi smentita</h2>
<p class="muted">Ipotesi di partenza: sulle viste molto zoomate Esaote (mediana
  <code>mm_per_px</code> 0.11; 732/1727 righe sotto 0.10, righello di soli 1–5 cm) la scala
  potrebbe essere stampata in <b>unità diverse</b> (mm invece di cm). <b>Smentita:</b> le
  letture corrette cadono sui valori attesi in cm senza alcun fattore di conversione, e la
  troncatura di cifra (<code>10</code> letto <code>1</code>) pesa solo l'1%.</p>
<p class="muted">Modi di errore misurati su <b>135 letture / 60 frame Esaote</b>
  (confronto lettura vs valore atteso dalla GT in quel punto):</p>
<table>
  <tr><th>modo</th><th>quota</th></tr>
  <tr><td>lettura corretta (±0.3 cm)</td><td class="up">53%</td></tr>
  <tr><td>sottostima</td><td class="down">25%</td></tr>
  <tr><td>sovrastima</td><td class="down">17%</td></tr>
  <tr><td>cifra extra</td><td>4%</td></tr>
  <tr><td>troncata (10 → 1)</td><td>1%</td></tr>
</table>
<p class="muted"><b>Conclusione:</b> su Esaote circa <b>metà delle letture è sbagliata</b>,
  senza un pattern sistematico da correggere a tavolino — le etichette sono piccole e poco
  contrastate sopra il tessuto. Con sole ~2 letture per frame non c'è ridondanza per
  scartare quella sbagliata: è esattamente il motivo per cui i passaggi OCR ad alta recall
  peggiorano Esaote (aggiungono letture, in maggioranza errate) mentre migliorano Hitachi.
  La strada per Esaote è quindi la <b>precisione</b> dell'OCR (o più etichette corrette per
  frame), non altra recall.</p>

<h2>3. Evidenza: cosa legge l'OCR vs cosa il righello mostra</h2>
<div class="muted legend">
  <span><i class="k" style="background:#00dc00"></i>GT: zero + valore di ogni tacca</span>
  <span><i class="k" style="background:#ff5000"></i>posizione di una lettura OCR</span>
  <span><i class="k" style="background:#00b4ff"></i>colonna del righello</span>
  <span><i class="k" style="background:#a000c8"></i><b>ultima tacca rilevata</b></span>
</div>
<p class="muted">In rosso le letture che si scostano di oltre 0.3 cm dal valore atteso in quel
  punto: se molte letture sono rosse e i valori attesi sono piccoli (righello di 1–5 cm),
  l'OCR sta leggendo numeri che <b>non appartengono al righello</b>, oppure ne sbaglia l'unità.</p>
{''.join(cards)}
</div>
<script>
// Pan/zoom on each full frame: wheel zooms at the cursor, drag pans, double-click resets.
document.querySelectorAll('.viewer').forEach(v => {{
  const img = v.querySelector('.stage');   // image + overlay move together
  const lvl = v.querySelector('.zlvl');
  // toggle the annotation layer on both the frame and the crop of this card
  const card = v.closest('.card');
  const tgl = v.querySelector('.tgl');
  tgl.addEventListener('click', () => {{
    const on = tgl.dataset.on === '1';
    tgl.dataset.on = on ? '0' : '1';
    tgl.textContent = on ? 'mostra elaborazioni' : 'nascondi elaborazioni';
    card.classList.toggle('hideovl', on);
  }});
  let z = 1, tx = 0, ty = 0, drag = false, sx = 0, sy = 0;
  const apply = () => {{
    const r = v.getBoundingClientRect();
    // keep the image covering the frame so it cannot be dragged off-screen
    const maxX = Math.max(0, r.width * (z - 1)), maxY = Math.max(0, r.height * (z - 1));
    tx = Math.min(0, Math.max(-maxX, tx)); ty = Math.min(0, Math.max(-maxY, ty));
    img.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{z}})`;
    lvl.textContent = Math.round(z * 100) + '%';
  }};
  const zoomAt = (f, cx, cy) => {{
    const nz = Math.min(12, Math.max(1, z * f));
    // keep the point under the cursor fixed
    tx = cx - (cx - tx) * (nz / z); ty = cy - (cy - ty) * (nz / z);
    z = nz; if (z === 1) {{ tx = 0; ty = 0; }} apply();
  }};
  v.addEventListener('wheel', e => {{
    e.preventDefault();
    const r = v.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }}, {{ passive: false }});
  v.addEventListener('pointerdown', e => {{
    if (e.target.closest('.zbar')) return;
    drag = true; sx = e.clientX - tx; sy = e.clientY - ty;
    v.classList.add('dragging'); v.setPointerCapture(e.pointerId);
  }});
  v.addEventListener('pointermove', e => {{
    if (!drag) return; tx = e.clientX - sx; ty = e.clientY - sy; apply();
  }});
  v.addEventListener('pointerup', () => {{ drag = false; v.classList.remove('dragging'); }});
  v.addEventListener('dblclick', () => {{ z = 1; tx = 0; ty = 0; apply(); }});
  v.querySelectorAll('.zbar button').forEach(b => b.addEventListener('click', () => {{
    const r = v.getBoundingClientRect(), k = b.dataset.z;
    if (k === 'in') zoomAt(1.4, r.width / 2, r.height / 2);
    else if (k === 'out') zoomAt(1 / 1.4, r.width / 2, r.height / 2);
    else {{ z = 1; tx = 0; ty = 0; apply(); }}
  }}));
  apply();
}});

// header switch: same toggle applied to every card at once
const all = document.getElementById('tgl-all');
all.addEventListener('click', () => {{
  const on = all.dataset.on === '1';
  all.dataset.on = on ? '0' : '1';
  all.textContent = on ? 'mostra elaborazioni su tutte le immagini'
                       : 'nascondi elaborazioni su tutte le immagini';
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('hideovl', on));
  document.querySelectorAll('.tgl').forEach(t => {{
    t.dataset.on = on ? '0' : '1';
    t.textContent = on ? 'mostra elaborazioni' : 'nascondi elaborazioni';
  }});
}});
</script>
</body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--current", type=Path, default=None)
    ap.add_argument("--vendor", type=str, default="Esaote")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--high-recall", action="store_true", help="Run the detector in high-recall mode.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    rows = [r for r in csv.DictReader(open(args.gt, encoding="utf-8")) if r["vendor"] == args.vendor]
    rows = [r for r in rows if r.get("image_path")]
    # Spread the sample across zoom levels: that is where the unit question lives.
    rows.sort(key=lambda r: _f(r["mm_per_px_from_scale"]) or 0.0)
    rng = random.Random(args.seed)
    buckets = [rows[i::3] for i in range(3)]  # zoomed / medium / normal
    picked: List[Dict[str, str]] = []
    for b in buckets:
        rng.shuffle(b)
        picked.extend(b[: max(1, args.n // 3)])

    cases = []
    for r in picked[: args.n]:
        c = build_case(r)
        if c:
            cases.append(c)
    cases.sort(key=lambda c: -(c["rel_err"] or 0))

    html = render(cases, metrics_block(args.baseline) if args.baseline else None,
                  metrics_block(args.current) if args.current else None, args.vendor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[ok] {len(cases)} casi -> {args.out} ({len(html)/1024:.0f} KB)")
    bad = sum(1 for c in cases if (c["rel_err"] or 0) > 0.05)
    print(f"[info] casi con errore calibrazione >5%: {bad}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
