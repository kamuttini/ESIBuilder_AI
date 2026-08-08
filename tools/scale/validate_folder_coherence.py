"""Measure the folder-coherence rules against ground truth, over many folders.

Runs each folder twice — the plain chain, and the chain plus the two folder-level rules
(one zero side per orientation group; tick spacing that has to agree with the values and
with the folder trend) — and compares both to the GT of that folder. Writes an HTML report
with the aggregate, a row per folder, the per-depth detail, and a picture of the worst case
so the numbers can be checked by eye.

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/validate_folder_coherence.py \
    --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
    --folders 15 --out artifacts/45_scale_coherence_validation/index.html
"""
from __future__ import annotations

import argparse
import base64
import collections
import copy
import csv
import glob
import importlib.util
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from scale.consolidate_scale_setup import ScaleCandidate, _theil_sen_log, consolidate_setup  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", REPO / "tools/scale/build_scale_folder_report.py")
rep = importlib.util.module_from_spec(_spec)
sys.modules["rep"] = rep
_spec.loader.exec_module(rep)


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def consolidate(cases: List[dict]) -> Dict[int, object]:
    cands = [
        ScaleCandidate(depth_index=c["depth_index"], x=c["pred"].x, y_zero=c["pred"].y_zero,
                       mm_per_px=c["pred"].mm_per_px, direction=c["pred"].direction,
                       confidence=c["pred"].confidence, status=c["pred"].status,
                       n_labels=len(c["pred"].labels or []))
        for c in cases if c["depth_index"] >= 0
    ]
    if not cands:
        return {}
    return {k.depth_index: k for k in
            consolidate_setup(cands, depth_indices=sorted({c.depth_index for c in cands}))}


def apply_rules(cases: List[dict]) -> None:
    """The two folder-level coherence rules, in the order that was validated."""
    rep.enforce_direction_groups(cases)
    reliable = [
        c for c in cases
        if c["pred"].mm_per_px and c["pred"].tick_pitch_px and c["depth_index"] >= 0
        and c["pred"].status == "accepted"
        and rep._step_fit(c["pred"].mm_per_px * c["pred"].tick_pitch_px)[1] <= 0.08
    ]
    trend = _theil_sen_log([(c["depth_index"], float(c["pred"].mm_per_px)) for c in reliable])
    ref: Dict[int, float] = {}
    if trend is not None:
        ref = {c["depth_index"]: float(trend(c["depth_index"])) for c in cases if c["depth_index"] >= 0}
    elif reliable:
        ref = {c["depth_index"]: float(c["pred"].mm_per_px) for c in reliable}
    trend_ok = rep.trend_is_trustworthy(cases, ref)
    rep.repair_step_incoherence(cases, ref)


def worst_thumb(cases: List[dict], gt: Dict[int, float], cons: Dict[int, object]) -> Tuple[str, str]:
    """Crop+overlay of the depth with the largest calibration error, for eye-checking."""
    worst, werr = None, -1.0
    for c in cases:
        v = getattr(cons.get(c["depth_index"]), "mm_per_px", None)
        g = gt.get(c["depth_index"])
        if v and g:
            e = abs(v - g) / g
            if e > werr:
                worst, werr = c, e
    if worst is None:
        return "", ""
    p = worst["pred"]
    img = cv2.imread(worst["path"], cv2.IMREAD_GRAYSCALE)
    if img is None or p.x is None:
        return "", ""
    h, w = img.shape
    x0, x1 = max(0, int(p.x) - 170), min(w, int(p.x) + 70)
    crop = cv2.cvtColor(img[:, x0:x1], cv2.COLOR_GRAY2BGR)
    z = 1.6
    crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    xg = int((p.x - x0) * z)
    for y in sorted(p.ticks_y or []):
        cv2.line(crop, (xg - 7, int(y * z)), (xg + 7, int(y * z)), (74, 210, 255), 1)
    if p.y_zero is not None:
        cv2.circle(crop, (xg, int(p.y_zero * z)), 6, (122, 214, 55), 2)
    far = p.y_last_tick if p.direction >= 0 else p.y_first_tick
    if far is not None:
        cv2.circle(crop, (xg, int(far * z)), 6, (200, 0, 160), -1)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    return uri, f"depth {worst['depth_index']} · errore {werr:.0%}"


def run_folder(folder: str, vendor: str, gt: Dict[int, float], gt_dir: Dict[int, str]) -> Optional[dict]:
    paths = sorted(glob.glob(os.path.join(folder, "image_samples", "image_depth_value_setup_*.png")))
    if len(paths) < 4:
        return None
    cases: List[dict] = []
    for p in paths:
        c = rep.process_image(p, vendor)
        if c:
            c["path"] = p
            cases.append(c)
    if len(cases) < 4:
        return None

    base_cons = consolidate(cases)
    base_snapshot = {c["depth_index"]: (
        getattr(base_cons.get(c["depth_index"]), "mm_per_px", None),
        getattr(base_cons.get(c["depth_index"]), "direction", None)) for c in cases}

    # the rules mutate the predictions, so work on a copy for the "after" arm
    cases_new = []
    for c in cases:
        d = dict(c)
        d["pred"] = copy.copy(c["pred"])
        d["pred"].labels = list(c["pred"].labels or [])
        d["pred"].ticks_y = list(c["pred"].ticks_y or [])
        d["fixes"] = []
        cases_new.append(d)
    apply_rules(cases_new)
    new_cons = consolidate(cases_new)

    rows = []
    eb, en = [], []
    dir_b = dir_n = dir_tot = 0
    for c in sorted(cases, key=lambda c: c["depth_index"]):
        i = c["depth_index"]
        g, gd = gt.get(i), gt_dir.get(i)
        bv, bd = base_snapshot.get(i, (None, None))
        nv = getattr(new_cons.get(i), "mm_per_px", None)
        nd = getattr(new_cons.get(i), "direction", None)
        e_b = abs(bv - g) / g if (bv and g) else None
        e_n = abs(nv - g) / g if (nv and g) else None
        if e_b is not None:
            eb.append(e_b)
        if e_n is not None:
            en.append(e_n)
        if gd:
            dir_tot += 1
            want = 1 if gd == "top" else -1
            dir_b += int(bd == want)
            dir_n += int(nd == want)
        fixes = next((d["fixes"] for d in cases_new if d["depth_index"] == i), [])
        rows.append({"depth": i, "gt": g, "base": bv, "new": nv, "e_b": e_b, "e_n": e_n,
                     "gt_dir": gd, "fixes": fixes})

    thumb, thumb_cap = worst_thumb(cases_new, gt, new_cons)
    trend_ok = all(d.get("trend_ok", False) for d in cases_new) if cases_new else False
    return {
        "folder": os.path.basename(folder), "vendor": vendor, "n": len(cases),
        "rows": rows,
        "dir_base": dir_b, "dir_new": dir_n, "dir_tot": dir_tot,
        "err_base": statistics.median(eb) if eb else None,
        "err_new": statistics.median(en) if en else None,
        "ok2_base": sum(1 for x in eb if x <= 0.02), "nb": len(eb),
        "ok2_new": sum(1 for x in en if x <= 0.02), "nn": len(en),
        "n_fixes": sum(1 for d in cases_new if d.get("fixes")),
        "trend_ok": trend_ok,
        "thumb": thumb, "thumb_cap": thumb_cap,
    }


def render(results: List[dict]) -> str:
    tot_dir_b = sum(r["dir_base"] for r in results)
    tot_dir_n = sum(r["dir_new"] for r in results)
    tot_dir = sum(r["dir_tot"] for r in results)
    eb = [r["err_base"] for r in results if r["err_base"] is not None]
    en = [r["err_new"] for r in results if r["err_new"] is not None]
    ok2b = sum(r["ok2_base"] for r in results)
    ok2n = sum(r["ok2_new"] for r in results)
    nb = sum(r["nb"] for r in results)
    nn = sum(r["nn"] for r in results)
    better = sum(1 for r in results if r["err_new"] is not None and r["err_base"] is not None
                 and r["err_new"] < r["err_base"] - 1e-9)
    worse = sum(1 for r in results if r["err_new"] is not None and r["err_base"] is not None
                and r["err_new"] > r["err_base"] + 1e-9)

    def cls(a, b, lower_better=True):
        if a is None or b is None:
            return ""
        if abs(a - b) < 1e-9:
            return "same"
        good = (b < a) if lower_better else (b > a)
        return "up" if good else "down"

    frows = []
    for r in sorted(results, key=lambda r: (r["vendor"], r["folder"])):
        dcls = cls(r["dir_base"], r["dir_new"], lower_better=False)
        ecls = cls(r["err_base"], r["err_new"])
        eb_s = "-" if r["err_base"] is None else f"{r['err_base']:.1%}"
        en_s = "-" if r["err_new"] is None else f"{r['err_new']:.1%}"
        frows.append(
            f"<tr><td>{r['vendor']}</td><td class='fold'>{r['folder'][:44]}</td><td>{r['n']}</td>"
            f"<td>{r['dir_base']}/{r['dir_tot']}</td><td class='{dcls}'>{r['dir_new']}/{r['dir_tot']}</td>"
            f"<td>{eb_s}</td><td class='{ecls}'>{en_s}</td>"
            f"<td>{r['ok2_base']}/{r['nb']}</td><td>{r['ok2_new']}/{r['nn']}</td>"
            f"<td>{r['n_fixes']}</td></tr>"
        )

    details = []
    for r in sorted(results, key=lambda r: (r["vendor"], r["folder"])):
        drows = []
        for x in r["rows"]:
            fx = "<br>".join(x["fixes"]) if x["fixes"] else ""
            g = "-" if x["gt"] is None else f"{x['gt']:.4f}"
            b = "-" if x["base"] is None else f"{x['base']:.4f}"
            n = "-" if x["new"] is None else f"{x['new']:.4f}"
            eb_ = "-" if x["e_b"] is None else f"{x['e_b']:.0%}"
            en_ = "-" if x["e_n"] is None else f"{x['e_n']:.0%}"
            ec = cls(x["e_b"], x["e_n"])
            drows.append(f"<tr><td>{x['depth']}</td><td>{g}</td><td>{x['gt_dir'] or '-'}</td>"
                         f"<td>{b}</td><td>{eb_}</td><td>{n}</td><td class='{ec}'>{en_}</td>"
                         f"<td class='fx'>{fx}</td></tr>")
        img = (f"<div class='thumb'><img src='{r['thumb']}' alt=''>"
               f"<div class='muted'>{r['thumb_cap']}</div></div>") if r["thumb"] else ""
        details.append(f"""
        <details><summary><b>{r['vendor']}</b> · {r['folder'][:56]}
           <span class="muted">— verso {r['dir_base']}/{r['dir_tot']} → {r['dir_new']}/{r['dir_tot']},
           errore {'' if r['err_base'] is None else f"{r['err_base']:.1%}"} →
           {'' if r['err_new'] is None else f"{r['err_new']:.1%}"}</span></summary>
          <div class="dwrap">
            <table><tr><th>depth</th><th>GT mm/px</th><th>GT zero</th><th>base</th><th>err</th>
              <th>con regole</th><th>err</th><th>correzioni applicate</th></tr>
              {''.join(drows)}</table>
            {img}
          </div>
        </details>""")

    med_b = f"{statistics.median(eb):.1%}" if eb else "-"
    med_n = f"{statistics.median(en):.1%}" if en else "-"
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validazione regole di coerenza di cartella</title>
<style>
 :root{{--bg:#0f1115;--card:#171a21;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
        --good:#37d67a;--bad:#ff5470;--warn:#ffb020}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
 header{{padding:20px 26px;border-bottom:1px solid var(--line);background:#12141a}}
 h1{{margin:0 0 6px;font-size:19px}} h2{{font-size:16px;margin:26px 0 8px}}
 .wrap{{max-width:1200px;margin:0 auto;padding:0 26px 50px}}
 .muted{{color:var(--mut)}}
 .big{{display:flex;gap:26px;flex-wrap:wrap;margin:14px 0 4px}}
 .kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;
       min-width:190px}}
 .kpi .v{{font-size:20px;font-weight:700;margin-top:3px}}
 .kpi .v small{{font-size:13px;color:var(--mut);font-weight:400}}
 table{{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px}}
 th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
 th{{color:var(--mut);font-weight:600}}
 td.up{{color:var(--good);font-weight:700}} td.down{{color:var(--bad);font-weight:700}}
 td.same{{color:var(--mut)}} td.fold{{font-size:12px}} td.fx{{color:var(--warn);font-size:12px}}
 details{{background:var(--card);border:1px solid var(--line);border-radius:9px;
          margin:9px 0;padding:9px 13px}}
 summary{{cursor:pointer;font-size:13px}}
 .dwrap{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}}
 .dwrap table{{flex:1;min-width:520px}}
 .thumb img{{max-height:420px;border-radius:6px;background:#000}}
</style></head><body>
<header>
  <h1>Regole di coerenza di cartella — validazione su GT</h1>
  <div class="muted">Due regole: <b>1)</b> un solo lato dello zero per gruppo di orientamento
    (nf/lr contro ud/lrud); <b>2)</b> le distanze fra tacche devono concordare con i valori e
    con il trend della cartella, altrimenti la calibrazione viene riallineata sul passo che
    combacia meglio. Confronto: catena semplice contro catena + regole, entrambe contro la GT.</div>
</header>
<div class="wrap">
  <div class="big">
    <div class="kpi"><div class="muted">verso corretto</div>
      <div class="v">{tot_dir_b}/{tot_dir} → <span style="color:var(--good)">{tot_dir_n}/{tot_dir}</span></div></div>
    <div class="kpi"><div class="muted">errore calibrazione (mediana delle cartelle)</div>
      <div class="v">{med_b} → {med_n}</div></div>
    <div class="kpi"><div class="muted">righe entro il 2%</div>
      <div class="v">{ok2b}/{nb} → {ok2n}/{nn}</div></div>
    <div class="kpi"><div class="muted">cartelle migliorate / peggiorate</div>
      <div class="v" style="color:var(--good)">{better}<small> / </small><span style="color:var(--bad)">{worse}</span></div></div>
  </div>
  <h2>Per cartella</h2>
  <table>
    <tr><th>vendor</th><th>cartella</th><th>depth</th><th>verso base</th><th>verso regole</th>
        <th>err base</th><th>err regole</th><th>≤2% base</th><th>≤2% regole</th><th>corr.</th></tr>
    {''.join(frows)}
  </table>
  <h2>Dettaglio per depth (apri una cartella)</h2>
  <div class="muted">L'immagine è la depth con l'errore più grande dopo le regole, con tacche,
    zero (verde) e ultima tacca (viola): serve a controllare a occhio se il numero ha senso.</div>
  {''.join(details)}
</div></body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--folders", type=int, default=15)
    ap.add_argument("--min-depths", type=int, default=6)
    ap.add_argument("--root", default="/Volumes/SSD_esi1_n1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    by: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
    for r in csv.DictReader(open(args.gt, encoding="utf-8")):
        by[(r["vendor"], r["config_folder"])].append(r)
    cand = [(v, f, rs) for (v, f), rs in by.items() if len(rs) >= args.min_depths]
    # spread across vendors, most-populated vendors first
    per_vendor: Dict[str, List] = collections.defaultdict(list)
    for v, f, rs in cand:
        per_vendor[v].append((v, f, rs))
    picked: List = []
    i = 0
    order = sorted(per_vendor, key=lambda v: -len(per_vendor[v]))
    while len(picked) < args.folders and any(per_vendor[v] for v in order):
        v = order[i % len(order)]
        if per_vendor[v]:
            picked.append(per_vendor[v].pop(0))
        i += 1

    results: List[dict] = []
    for n, (vendor, folder, rs) in enumerate(picked, 1):
        path = os.path.join(args.root, folder)
        if not os.path.isdir(path):
            print(f"[skip] cartella assente: {folder[:50]}")
            continue
        gt = {int(_f(r["depth_index"]) or -1): _f(r["mm_per_px_from_scale"]) for r in rs}
        gt_dir = {int(_f(r["depth_index"]) or -1): r["zero_at"] for r in rs}
        print(f"[{n}/{len(picked)}] {vendor} · {folder[:46]}", flush=True)
        res = run_folder(path, vendor, gt, gt_dir)
        if res:
            results.append(res)
            eb_s = "-" if res["err_base"] is None else f"{res['err_base']:.1%}"
            en_s = "-" if res["err_new"] is None else f"{res['err_new']:.1%}"
            print(f"      verso {res['dir_base']}/{res['dir_tot']} -> {res['dir_new']}/{res['dir_tot']}"
                  f" · err {eb_s} -> {en_s} · correzioni {res['n_fixes']}"
                  f" · trend {'affidabile' if res['trend_ok'] else 'NON affidabile'}", flush=True)

    if not results:
        print("[error] nessuna cartella elaborata")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results), encoding="utf-8")
    print(f"[ok] {len(results)} cartelle -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
