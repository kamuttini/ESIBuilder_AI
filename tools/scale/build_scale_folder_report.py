"""Run the scale detector over a whole configuration folder and report it as one HTML page.

This is the production-shaped view: the input is a folder of acquisitions from **one**
scanner+probe, where the images differ by depth (so the ruler carries more or fewer ticks)
and by orientation. No ground truth is needed or used — everything shown is what the
pipeline predicts, which is what the operator has to confirm or correct.

Per image it draws, on the untouched frame: every detected tick, the zero, the last tick,
and the value of each tick in cm. The whole folder is then consolidated with the setup
consensus (robust trend over the depths) and both answers are shown side by side.

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/build_scale_folder_report.py \
    --folder "/Volumes/SSD_esi1_n1/Esaote MyLab 80XPro F100200 - TLC3-13 - L lecce" \
    --out artifacts/44_scale_folder_reports/esaote_lecce.html

  # every orientation variant too, not just one image per depth:
  ... --pattern "image_depth_find_*_setup_*.png"
"""
from __future__ import annotations

import argparse
import base64
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scale.consolidate_scale_setup import (  # noqa: E402
    ScaleCandidate,
    _theil_sen_log,
    consolidate_setup,
)
from scale.detect_scale_ladder import PROFILES, ScalePrediction, detect_scale, load_image  # noqa: E402

DEPTH_RE = re.compile(r"setup_(\d+)", re.I)


def guess_vendor(folder: str) -> str:
    """Vendor from the folder name: the corpus names folders '<Vendor> <model> - <probe>'."""
    name = os.path.basename(folder.rstrip("/\\")).lower()
    for v in PROFILES:
        if v != "default" and v.lower() in name:
            return v
    # a couple of aliases seen in the corpus
    for alias, v in (("toshiba", "Canon"), ("bk ", "BK"), ("bk_", "BK")):
        if alias in name:
            return v
    return ""


def orientation_group(name: str) -> str:
    """Which orientation group an image belongs to.

    A left-right mirror does not move the zero end, an up-down one does. So ``no_flip`` and
    ``flip_lr`` share the zero side, and ``flip_ud`` and ``flip_lrud`` share the opposite
    one. Inside a group the zero side is a property of the group, never of the image.
    """
    n = name.lower()
    if "flip_lrud" in n or "flip_ud" in n:
        return "ud"
    if "flip_lr" in n or "no_flip" in n:
        return "nf"
    return "base"  # image_depth_value_*: one per depth, all the same orientation


PLAUSIBLE_STEPS_MM = (1.0, 2.0, 2.5, 5.0, 10.0, 20.0)


def _step_fit(step_mm: float) -> Tuple[float, float]:
    """Nearest printed step and its relative distance."""
    best, err = 0.0, 1e9
    for s in PLAUSIBLE_STEPS_MM:
        e = abs(step_mm - s) / s
        if e < err:
            best, err = s, e
    return best, err


def enforce_direction_groups(cases: List[dict]) -> None:
    """One zero side per orientation group, in place.

    Images of the same group must agree: a left-right mirror cannot move the zero end, so a
    minority disagreeing is a detection error, not a real flip. The side is decided by the
    best-supported evidence (confidence x number of read labels) and the losers get their
    zero moved to the other end of their own ruler.
    """
    # --- 1. zero side per orientation group ---------------------------------
    groups: Dict[str, List[dict]] = {}
    for c in cases:
        groups.setdefault(orientation_group(c["name"]), []).append(c)

    for gname, items in groups.items():
        score = {1: 0.0, -1: 0.0}
        for c in items:
            p = c["pred"]
            if p.x is None:
                continue
            # labels are the only direct evidence of which way the numbers grow
            w = (0.2 + p.confidence) * (1.0 + 2.0 * len(p.labels or []))
            score[1 if p.direction >= 0 else -1] += w
        if score[1] == 0.0 and score[-1] == 0.0:
            continue
        winner = 1 if score[1] >= score[-1] else -1
        for c in items:
            p = c["pred"]
            if p.x is None or p.direction == winner:
                continue
            ticks = sorted(p.ticks_y or [])
            if ticks:
                # the zero sits at the near end for the group's direction
                p.y_zero = ticks[0] if winner > 0 else ticks[-1]
            p.direction = winner
            # An image whose zero side contradicts its own orientation group has demonstrably
            # failed, and the failures are correlated: on the validated Esaote folder the same
            # three depths that got the side wrong also had mm_per_px at half the truth. So its
            # calibration must not anchor the folder — drop it and let the robust depth trend
            # interpolate the value instead. (Keeping it cost 5% -> 30% median error.)
            p.mm_per_px = None
            p.status = "review"
            p.confidence = min(p.confidence, 0.2)
            c["mm_per_px"] = None
            c["status"] = "review"
            c["y_zero"] = None if p.y_zero is None else round(p.y_zero, 1)
            c["zero_at"] = "alto" if winner >= 0 else "basso"
            c["fixes"] = c.get("fixes", []) + [
                f"verso forzato dal gruppo {gname}; calibrazione scartata e reinterpolata"
            ]

def trend_is_trustworthy(cases: List[dict], reference: Dict[int, float]) -> bool:
    """Is the folder trend solid enough to correct individual images against?

    Repairing against a bad trend is worse than leaving the image alone: measured over 16 GT
    folders, folders whose trend rested on few coherent images got *worse* (Canon 12% -> 115%,
    Hitachi 12% -> 68%) while folders with a solid trend got much better (Siemens 826% -> 17%).
    So the trend has to earn the right to overrule an image: enough supporting images, and
    those images actually close to it.
    """
    if not reference:
        return False
    res: List[float] = []
    for c in cases:
        p = c["pred"]
        ref = reference.get(c["depth_index"])
        if not (p.mm_per_px and p.tick_pitch_px and ref):
            continue
        if p.status == "accepted" and _step_fit(p.mm_per_px * p.tick_pitch_px)[1] <= 0.08:
            res.append(abs(p.mm_per_px / ref - 1.0))
    if len(res) < 4:
        return False
    return float(np.median(res)) <= 0.15


def repair_step_incoherence(
    cases: List[dict],
    reference: Dict[int, float],
    tol: float = 0.08,
) -> None:
    """Re-derive the calibration where the tick spacing disagrees with the values.

    ``mm_per_px * pitch`` is the printed step, so it has to be a step a scanner really uses.
    When it is not, the values and the spacing contradict each other and something has to
    move. The pitch is measured on the ticks to ~1 px, so it is the trustworthy part: we keep
    it and pick the plausible step whose implied ``mm_per_px = step / pitch`` best matches
    the strongest thing we know about the folder — its robust depth trend (``reference``),
    which is built from the images that *are* coherent. That is the "highest match" anchor,
    rather than the image's own wrong value.
    """
    if not trend_is_trustworthy(cases, reference):
        # Nothing solid to correct against: leave the images as they are and let the ordinary
        # consensus (isotonic + robust trend) do its usual job.
        for c in cases:
            c["trend_ok"] = False
        return
    for c in cases:
        c["trend_ok"] = True

    votes: Dict[float, float] = {}
    for c in cases:
        p = c["pred"]
        if not (p.mm_per_px and p.tick_pitch_px):
            continue
        step, err = _step_fit(p.mm_per_px * p.tick_pitch_px)
        if err <= tol:
            votes[step] = votes.get(step, 0.0) + (1.0 - err) * (1.0 + len(p.labels or []))

    for c in cases:
        p = c["pred"]
        if not (p.mm_per_px and p.tick_pitch_px):
            continue
        raw = p.mm_per_px * p.tick_pitch_px
        _step, err = _step_fit(raw)
        c["step_fit_err"] = round(err, 3)
        ref = reference.get(c["depth_index"])
        # Two ways to be wrong, and the second is the sneaky one:
        #  - the implied step is not a step any scanner prints: unambiguous, always repair;
        #  - the step *is* plausible but a half/double multiple of the real one, which is
        #    self-consistent and only shows as a disagreement with the folder trend.
        #
        # The trend must not be allowed to overrule an image that has strong direct evidence
        # of its own, and it must only intervene for the large (factor-2-ish) disagreements
        # that a wrong step multiple produces. Measured over 16 GT folders, repairing on a
        # mere 25% disagreement cut the catastrophic tail (mean 135% -> 51%) but broke folders
        # that were already right (Canon 12% -> 115%): the median got worse, 11.6% -> 16.6%.
        ratio = (p.mm_per_px / ref) if ref else 1.0
        wrong_multiple = bool(ref) and (ratio >= 1.8 or ratio <= 1 / 1.8)
        strong_evidence = (
            p.status == "accepted" and len(p.labels or []) >= 3 and err <= tol
        )
        if err <= tol and not (wrong_multiple and not strong_evidence):
            continue  # spacing, values and folder agree, or the image knows better
        best: Optional[Tuple[float, float, float]] = None  # (score, step, mm_per_px)
        for s in PLAUSIBLE_STEPS_MM:
            cand = s / p.tick_pitch_px
            # agreement with the folder trend dominates; steps the folder already uses break ties
            agree = 1.0 / (1.0 + abs(cand / ref - 1.0)) if ref else 0.0
            sc = 6.0 * agree + votes.get(s, 0.0)
            if best is None or sc > best[0]:
                best = (sc, s, cand)
        if best is not None and abs(best[2] / p.mm_per_px - 1.0) > 1e-6:
            old = p.mm_per_px
            p.mm_per_px = best[2]
            p.tick_step_mm = best[1]
            c["mm_per_px"] = round(p.mm_per_px, 4)
            c["step_mm"] = round(best[1], 2)
            why = "passo incoerente" if err > tol else "in disaccordo col trend della cartella"
            c["fixes"] = c.get("fixes", []) + [
                f"{why} (passo letto {raw:.2f}mm): calibrazione riallineata a "
                f"{best[1]:g}mm/tacca ({old:.4f}→{p.mm_per_px:.4f}"
                + (f", trend {ref:.4f}" if ref else "") + ")"
            ]


def _b64(img: np.ndarray, quality: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _svg(
    vw: int,
    vh: int,
    s: float,
    pred: ScalePrediction,
    label_every: int,
) -> str:
    """Overlay: ruler column, every tick with its value in cm, the zero, the last tick."""
    if pred.x is None:
        return f'<svg class="ovl" viewBox="0 0 {vw} {vh}"></svg>'
    X = pred.x * s
    p: List[str] = [f'<svg class="ovl" viewBox="0 0 {vw} {vh}" preserveAspectRatio="none">']
    p.append(f'<line x1="{X:.1f}" y1="0" x2="{X:.1f}" y2="{vh}" stroke="#00b4ff" '
             f'stroke-width="1" stroke-dasharray="4 6" opacity="0.55"/>')
    ticks = sorted(pred.ticks_y or [])
    far = pred.y_last_tick if pred.direction >= 0 else pred.y_first_tick
    for i, y in enumerate(ticks):
        Y = y * s
        p.append(f'<line x1="{X-8:.1f}" y1="{Y:.1f}" x2="{X+8:.1f}" y2="{Y:.1f}" '
                 f'stroke="#ffd24a" stroke-width="1.4" opacity="0.95"/>')
        # value of this tick, from the prediction itself (cm)
        if pred.mm_per_px and pred.y_zero is not None and i % label_every == 0:
            v_cm = pred.direction * (y - pred.y_zero) * pred.mm_per_px / 10.0
            p.append(f'<text x="{X+12:.1f}" y="{Y+4:.1f}" fill="#ffd24a" font-size="11">'
                     f'{v_cm:.1f}</text>')
    # OCR readings that fed the calibration
    for y, v_mm in (pred.labels or []):
        p.append(f'<circle cx="{X:.1f}" cy="{y*s:.1f}" r="5" fill="none" stroke="#ff5000" '
                 f'stroke-width="2"/>')
        p.append(f'<text x="{X-16:.1f}" y="{y*s+4:.1f}" fill="#ff8c46" font-size="11" '
                 f'text-anchor="end">{v_mm/10.0:g}</text>')
    if pred.y_zero is not None:
        p.append(f'<circle cx="{X:.1f}" cy="{pred.y_zero*s:.1f}" r="7" fill="none" '
                 f'stroke="#37d67a" stroke-width="2.5"/>')
        p.append(f'<text x="{X+14:.1f}" y="{pred.y_zero*s-8:.1f}" fill="#37d67a" '
                 f'font-size="12" font-weight="bold">0</text>')
    if far is not None:
        p.append(f'<circle cx="{X:.1f}" cy="{far*s:.1f}" r="7" fill="#a000c8" stroke="#fff" '
                 f'stroke-width="1"/>')
    p.append("</svg>")
    return "".join(p)


def process_image(path: str, vendor: str, cap_w: int = 1400) -> Optional[dict]:
    img = load_image(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    pred = detect_scale(gray, rect=None, vendor=vendor)

    s = min(1.0, cap_w / float(w))
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if s < 1.0:
        view = cv2.resize(view, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    n = len(pred.ticks_y or [])
    label_every = 1 if n <= 14 else (2 if n <= 28 else 4)
    m = DEPTH_RE.search(os.path.basename(path))
    span_cm = None
    if pred.span_mm:
        span_cm = round(pred.span_mm / 10.0, 1)
    return {
        "name": os.path.basename(path),
        "depth_index": int(m.group(1)) if m else -1,
        "img": _b64(view),
        "svg": _svg(view.shape[1], view.shape[0], s, pred, label_every),
        "w": w, "h": h,
        "pred": pred,
        "x": None if pred.x is None else round(pred.x, 1),
        "y_zero": None if pred.y_zero is None else round(pred.y_zero, 1),
        "y_far": None if (pred.y_last_tick is None and pred.y_first_tick is None) else round(
            (pred.y_last_tick if pred.direction >= 0 else pred.y_first_tick) or 0, 1),
        "mm_per_px": None if pred.mm_per_px is None else round(pred.mm_per_px, 4),
        "pitch": None if pred.tick_pitch_px is None else round(pred.tick_pitch_px, 1),
        "step_mm": None if not (pred.mm_per_px and pred.tick_pitch_px) else round(
            pred.mm_per_px * pred.tick_pitch_px, 2),
        "n_ticks": n,
        "n_labels": len(pred.labels or []),
        "span_cm": span_cm,
        "zero_at": "alto" if pred.direction >= 0 else "basso",
        "status": pred.status,
        "reason": pred.reason,
    }


def render(folder: str, vendor: str, cases: List[dict], cons: Dict[int, object]) -> str:
    rows = []
    for c in sorted(cases, key=lambda c: (c["depth_index"], c["name"])):
        k = cons.get(c["depth_index"])
        cons_mm = f"{k.mm_per_px:.4f}" if (k is not None and k.mm_per_px) else "-"
        cons_src = getattr(k, "source", "-") if k is not None else "-"
        badge = {"accepted": "good", "review": "warn", "reject": "bad"}.get(c["status"], "")
        rows.append(
            f"<tr><td>{c['depth_index']}</td><td class='{badge}'>{c['status']}</td>"
            f"<td>{c['mm_per_px'] or '-'}</td><td>{cons_mm}</td><td>{cons_src}</td>"
            f"<td>{c['n_ticks']}</td><td>{c['pitch'] or '-'}</td><td>{c['step_mm'] or '-'}</td>"
            f"<td>{c['n_labels']}</td><td>{c['span_cm'] or '-'}</td><td>{c['zero_at']}</td></tr>"
        )

    cards = []
    for c in sorted(cases, key=lambda c: (c["depth_index"], c["name"])):
        badge = {"accepted": "good", "review": "warn", "reject": "bad"}.get(c["status"], "")
        cards.append(f"""
        <article class="card">
          <div class="hd">depth <b>{c['depth_index']}</b> · <span class="muted">{c['name']}</span>
            <span class="pill {badge}">{c['status']}</span>
            <span class="pill">mm/px {c['mm_per_px'] or 'n/d'}</span>
            <span class="pill">tacche {c['n_ticks']}</span>
            <span class="pill">passo {c['step_mm'] or 'n/d'} mm</span></div>
          <div class="viewer">
            <div class="stage"><img class="full" src="{c['img']}" alt="">{c['svg']}</div>
            <div class="zbar">
              <button data-z="out">−</button><span class="zlvl">100%</span><button data-z="in">+</button>
              <button data-z="reset">reset</button>
              <button class="tgl" data-on="1">nascondi elaborazioni</button>
              <span class="muted">rotella = zoom · trascina = sposta</span>
            </div>
          </div>
          <div class="foot muted">
            colonna x={c['x'] or '-'} · zero y={c['y_zero'] or '-'} · ultima tacca y={c['y_far'] or '-'}
            · zero in {c['zero_at']} · gruppo orient. {orientation_group(c['name'])}
            · etichette lette {c['n_labels']}
            {(' · ' + c['reason']) if c['reason'] else ''}
          </div>
          {('<div class="fixes">⟳ ' + ' · '.join(c['fixes']) + '</div>') if c.get('fixes') else ''}
        </article>""")

    n_acc = sum(1 for c in cases if c["status"] == "accepted")
    n_rev = sum(1 for c in cases if c["status"] == "review")
    n_rej = sum(1 for c in cases if c["status"] == "reject")
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scala — {os.path.basename(folder)}</title>
<style>
 :root{{--bg:#0f1115;--card:#171a21;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
        --good:#37d67a;--bad:#ff5470;--warn:#ffb020}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
 header{{padding:18px 24px;border-bottom:1px solid var(--line);background:#12141a;
         position:sticky;top:0;z-index:5}}
 h1{{margin:0 0 4px;font-size:18px}}
 .wrap{{max-width:1240px;margin:0 auto;padding:0 24px 40px}}
 .muted{{color:var(--mut)}}
 button{{background:#232732;color:var(--ink);border:1px solid var(--line);border-radius:7px;
         padding:5px 10px;cursor:pointer;font-size:13px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0 22px;font-size:13px}}
 th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}}
 th{{color:var(--mut);font-weight:600;position:sticky;top:0;background:#12141a}}
 td.good{{color:var(--good)}} td.warn{{color:var(--warn)}} td.bad{{color:var(--bad)}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        margin:16px 0;overflow:hidden}}
 .hd{{padding:9px 12px;border-bottom:1px solid var(--line);font-size:13px}}
 .foot{{padding:8px 12px;border-top:1px solid var(--line);font-size:12px}}
 .fixes{{padding:7px 12px;border-top:1px solid var(--line);font-size:12px;
         background:rgba(255,176,32,.09);color:var(--warn)}}
 .pill{{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);
        font-size:11px;margin-left:6px;color:var(--mut)}}
 .pill.good{{border-color:var(--good);color:var(--good)}}
 .pill.warn{{border-color:var(--warn);color:var(--warn)}}
 .pill.bad{{border-color:var(--bad);color:var(--bad)}}
 .viewer{{position:relative;overflow:hidden;background:#000;max-height:78vh;cursor:grab}}
 .viewer.dragging{{cursor:grabbing}}
 .viewer .stage{{position:relative;transform-origin:0 0;transition:transform .05s linear}}
 .viewer img.full{{display:block;width:100%;height:auto}}
 svg.ovl{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
 .hideovl svg.ovl{{display:none}}
 .zbar{{position:absolute;left:10px;bottom:10px;display:flex;gap:6px;align-items:center;
        background:rgba(10,12,17,.85);border:1px solid var(--line);border-radius:8px;
        padding:5px 9px;font-size:12px}}
 .zbar .zlvl{{min-width:44px;text-align:center;color:var(--mut)}}
 .zbar button.tgl{{border-color:var(--good);color:var(--good)}}
 .zbar button.tgl[data-on="0"]{{border-color:var(--line);color:var(--mut)}}
 .legend span{{margin-right:14px}}
 .k{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;
     margin-right:4px}}
</style></head><body>
<header>
  <h1>Scala — {os.path.basename(folder)}</h1>
  <div class="muted">vendor rilevato: <b>{vendor or 'default'}</b> ·
    immagini elaborate: <b>{len(cases)}</b> ·
    accepted {n_acc} / review {n_rev} / reject {n_rej}</div>
  <div style="margin-top:9px" class="legend muted">
    <span><i class="k" style="background:#ffd24a"></i>tacche + valore (cm)</span>
    <span><i class="k" style="background:#37d67a"></i>zero</span>
    <span><i class="k" style="background:#a000c8"></i>ultima tacca</span>
    <span><i class="k" style="background:#ff5000"></i>numero letto dall'OCR</span>
    <span><i class="k" style="background:#00b4ff"></i>colonna</span>
    <button id="tgl-all" data-on="1">nascondi elaborazioni su tutte</button>
  </div>
</header>
<div class="wrap">
<h2>Riepilogo della cartella</h2>
<div class="muted">La colonna <b>mm/px consenso</b> è il valore dopo il consolidamento sul
  trend robusto delle depth: è quello che la cartella "crede" nel suo insieme, e corregge le
  singole immagini incoerenti.</div>
<table>
 <tr><th>depth</th><th>stato</th><th>mm/px immagine</th><th>mm/px consenso</th><th>fonte</th>
     <th>tacche</th><th>pitch px</th><th>passo mm</th><th>etich.</th><th>span cm</th>
     <th>zero in</th></tr>
 {''.join(rows)}
</table>
<h2>Immagini</h2>
{''.join(cards)}
</div>
<script>
document.querySelectorAll('.viewer').forEach(v => {{
  const st = v.querySelector('.stage'), lvl = v.querySelector('.zlvl');
  const card = v.closest('.card'), tgl = v.querySelector('.tgl');
  let z = 1, tx = 0, ty = 0, drag = false, sx = 0, sy = 0;
  const apply = () => {{
    const r = v.getBoundingClientRect();
    const mx = Math.max(0, r.width * (z - 1)), my = Math.max(0, r.height * (z - 1));
    tx = Math.min(0, Math.max(-mx, tx)); ty = Math.min(0, Math.max(-my, ty));
    st.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{z}})`;
    lvl.textContent = Math.round(z * 100) + '%';
  }};
  const zoomAt = (f, cx, cy) => {{
    const nz = Math.min(12, Math.max(1, z * f));
    tx = cx - (cx - tx) * (nz / z); ty = cy - (cy - ty) * (nz / z);
    z = nz; if (z === 1) {{ tx = 0; ty = 0; }} apply();
  }};
  v.addEventListener('wheel', e => {{
    e.preventDefault(); const r = v.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1/1.15, e.clientX - r.left, e.clientY - r.top);
  }}, {{passive:false}});
  v.addEventListener('pointerdown', e => {{
    if (e.target.closest('.zbar')) return;
    drag = true; sx = e.clientX - tx; sy = e.clientY - ty;
    v.classList.add('dragging'); v.setPointerCapture(e.pointerId);
  }});
  v.addEventListener('pointermove', e => {{ if (drag) {{ tx = e.clientX - sx; ty = e.clientY - sy; apply(); }} }});
  v.addEventListener('pointerup', () => {{ drag = false; v.classList.remove('dragging'); }});
  v.addEventListener('dblclick', () => {{ z = 1; tx = 0; ty = 0; apply(); }});
  v.querySelectorAll('.zbar button[data-z]').forEach(b => b.addEventListener('click', () => {{
    const r = v.getBoundingClientRect();
    if (b.dataset.z === 'in') zoomAt(1.4, r.width/2, r.height/2);
    else if (b.dataset.z === 'out') zoomAt(1/1.4, r.width/2, r.height/2);
    else {{ z = 1; tx = 0; ty = 0; apply(); }}
  }}));
  tgl.addEventListener('click', () => {{
    const on = tgl.dataset.on === '1';
    tgl.dataset.on = on ? '0' : '1';
    tgl.textContent = on ? 'mostra elaborazioni' : 'nascondi elaborazioni';
    card.classList.toggle('hideovl', on);
  }});
  apply();
}});
const all = document.getElementById('tgl-all');
all.addEventListener('click', () => {{
  const on = all.dataset.on === '1';
  all.dataset.on = on ? '0' : '1';
  all.textContent = on ? 'mostra elaborazioni su tutte' : 'nascondi elaborazioni su tutte';
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
    ap.add_argument("--folder", required=True,
                    help="Configuration folder (or its image_samples directory).")
    ap.add_argument("--pattern", default="image_depth_value_setup_*.png",
                    help="Which images to process. Use 'image_depth_find_*_setup_*.png' to "
                         "include every orientation variant.")
    ap.add_argument("--vendor", default="", help="Override the vendor profile.")
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--high-recall", action="store_true",
                    help="Extra tick/OCR passes: finds more rulers, less precise calibration.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    folder = str(args.folder).rstrip("/\\")
    base = folder if os.path.basename(folder) == "image_samples" else os.path.join(folder, "image_samples")
    if not os.path.isdir(base):
        base = folder  # a plain directory of images is fine too
    paths = sorted(glob.glob(os.path.join(base, args.pattern)))
    if not paths:
        print(f"[error] nessuna immagine con pattern {args.pattern} in {base}")
        return 2
    paths = paths[: args.max_images]
    vendor = args.vendor or guess_vendor(folder)
    print(f"[info] {len(paths)} immagini, vendor='{vendor or 'default'}'")

    cases: List[dict] = []
    for i, p in enumerate(paths, 1):
        c = process_image(p, vendor)
        if c:
            cases.append(c)
        if i % 5 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    # Folder-level coherence. The zero side is a group property, so it is settled first and
    # needs no trend. The step repair instead needs a reference to aim at, so the consensus
    # runs once to give the robust depth trend, the incoherent images are repaired against
    # it, and the consensus is then recomputed on the repaired set.
    # Direction first: it now *removes* the calibrations that contradict their orientation
    # group, so the trend below is fitted on what is left, which is the trustworthy part.
    enforce_direction_groups(cases)

    def _consolidate() -> Dict[int, object]:
        cands = [
            ScaleCandidate(depth_index=c["depth_index"], x=c["pred"].x, y_zero=c["pred"].y_zero,
                           mm_per_px=c["pred"].mm_per_px, direction=c["pred"].direction,
                           confidence=c["pred"].confidence, status=c["pred"].status,
                           n_labels=len(c["pred"].labels or []))
            for c in cases if c["depth_index"] >= 0
        ]
        out: Dict[int, object] = {}
        if cands:
            for k in consolidate_setup(cands, depth_indices=sorted({c.depth_index for c in cands})):
                out[k.depth_index] = k
        return out

    # The reference trend must come from the *trustworthy* images only. Measured on an
    # Esaote folder: three depths had mm_per_px at half the truth (step 5 mm read where the
    # scanner printed 10 mm). Letting them into the fit flattened the trend to a constant
    # and the folder error went from 5% to 30%; fitting on the coherent, accepted subset
    # keeps the trend right and lets those three be repaired against it.
    reliable = [
        c for c in cases
        if c["pred"].mm_per_px and c["pred"].tick_pitch_px and c["depth_index"] >= 0
        and c["pred"].status == "accepted"
        and _step_fit(c["pred"].mm_per_px * c["pred"].tick_pitch_px)[1] <= 0.08
    ]
    trend = _theil_sen_log([(c["depth_index"], float(c["pred"].mm_per_px)) for c in reliable])
    ref: Dict[int, float] = {}
    if trend is not None:
        ref = {c["depth_index"]: float(trend(c["depth_index"])) for c in cases if c["depth_index"] >= 0}
    elif reliable:  # too few to fit a trend: use the reliable values themselves
        ref = {c["depth_index"]: float(c["pred"].mm_per_px) for c in reliable}
    print(f"[trend] costruito su {len(reliable)}/{len(cases)} immagini affidabili"
          f"{' (fit ok)' if trend is not None else ' (fit non possibile)'}")

    repair_step_incoherence(cases, ref)
    cons = _consolidate()

    n_fix = sum(1 for c in cases if c.get("fixes"))
    print(f"[coerenza] immagini corrette: {n_fix}/{len(cases)}")
    for c in cases:
        for fx in c.get("fixes", []):
            print(f"   depth {c['depth_index']}: {fx}")

    # the overlay must reflect the corrected geometry
    for c in cases:
        s = min(1.0, 1400.0 / float(c["w"]))
        vw = int(c["w"] * s) if s < 1.0 else c["w"]
        vh = int(c["h"] * s) if s < 1.0 else c["h"]
        n = len(c["pred"].ticks_y or [])
        c["svg"] = _svg(vw, vh, s, c["pred"], 1 if n <= 14 else (2 if n <= 28 else 4))

    html = render(folder, vendor, cases, cons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[ok] {len(cases)} immagini -> {args.out} ({len(html)/1024:.0f} KB)")
    st: Dict[str, int] = {}
    for c in cases:
        st[c["status"]] = st.get(c["status"], 0) + 1
    print(f"[stato] {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
