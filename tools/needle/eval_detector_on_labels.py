#!/usr/bin/env python3
"""Measure the needle detector against Camilla's traced needles.

The numbers that decide whether a change is an improvement were, until now, produced by
throwaway commands: they could not be repeated and could not be compared. This is the same
measurement, written down.

A frame counts as answered when the detector returns anything. The error is the angle between
its first choice and the traced needle closest to it in angle -- closest, because a frame can
hold two parallel needles and calling the second one an error would measure nothing.

The excluded frames are the other half of the test. Camilla marked them as not-probe-in-water,
so the right answer there is to return nothing at all; `--strict` is what makes that possible,
and the table prints how many of them the detector still answers.

  python3 tools/needle/eval_detector_on_labels.py \
      --labels artifacts/92_guides/etichette/aghi_tracciati_camilla_v3_varieta.json --strict
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from detect_needle import detect  # noqa: E402
from propose_guide_lines import order_by_probe  # noqa: E402
from refine_needles import refine  # noqa: E402


def _angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Degrees from horizontal, folded to [-90, 90]: a segment has no head or tail."""
    a = math.degrees(math.atan2(y2 - y1, x2 - x1))
    while a > 90.0:
        a -= 180.0
    while a <= -90.0:
        a += 180.0
    return a


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _load(paths: Sequence[Path]) -> Tuple[List[Dict], List[Dict]]:
    tracciati: List[Dict] = []
    escluse: List[Dict] = []
    for path in paths:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, list):
            tracciati.extend(blob)
            continue
        tracciati.extend(blob.get("annotazioni") or [])
        escluse.extend(blob.get("escluse") or [])
    return tracciati, escluse


def _rects_from_gallery(path: Path) -> Dict[str, List[int]]:
    """frame -> rect, letto dalla galleria: le escluse non hanno annotazione, quindi non
    portano con se' il rettangolo, ma la galleria che le mostrava lo sa."""
    testo = path.read_text(encoding="utf-8", errors="ignore")
    trovati: Dict[str, List[int]] = {}
    for frame, rect in re.findall(r'data-frame="([^"]*)"[^>]*data-rect="([^"]*)"', testo):
        parti = [int(float(v)) for v in rect.split(",") if v.strip()]
        if len(parti) == 4:
            trovati[frame] = parti
    return trovati


def _choices(frame: Path, rect: Sequence[int], strict: bool,
             ordine: str = "score", flipped: bool = False) -> List[Tuple[float, float, float, float]]:
    gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    box = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    needles = refine(detect(gray, box, top_k=6), gray, rect=box, max_needles=2,
                     fallback=not strict)
    if ordine == "probe":
        needles = order_by_probe(needles, box, flipped)
    return [(n.p1[0], n.p1[1], n.p2[0], n.p2[1]) for n in needles]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="a rule excludes instead of choosing: no needle rather than a wrong one")
    ap.add_argument("--match", choices=("first", "any"), default="first",
                    help="'first' scores the top choice only; 'any' scores the better of the two "
                         "it returns, which is what a reviewer confirming a pair would see")
    ap.add_argument("--order", choices=("score", "probe"), default="score",
                    help="'probe' mette per primo l'ago piu' vicino alla sonda, che e' il "
                         "principale; 'score' l'ordine del rilevatore (luminosita' per cresta)")
    ap.add_argument("--flipped", action="store_true",
                    help="orientamento UD o LRUD: la sonda e' in basso")
    ap.add_argument("--gallery", type=Path, default=None,
                    help="HTML gallery the labels came from, to recover the rect of excluded frames")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    tracciati, escluse = _load(args.labels)
    rect_di = _rects_from_gallery(args.gallery) if args.gallery else {}
    per_frame: Dict[str, Dict] = {}
    for voce in tracciati:
        frame = str(voce.get("frame") or "")
        if not frame:
            continue
        riga = per_frame.setdefault(frame, {"rect": voce.get("rect"), "lines": []})
        riga["lines"].extend(voce.get("lines") or [])

    errori: List[float] = []
    senza_risposta = 0
    mancanti = 0
    for frame, dati in sorted(per_frame.items()):
        percorso = Path(frame)
        if not percorso.is_file():
            mancanti += 1
            continue
        scelte = _choices(percorso, dati["rect"], args.strict, args.order, args.flipped)
        if not scelte:
            senza_risposta += 1
            continue
        veri = [_angle(*line[:4]) for line in dati["lines"] if len(line) >= 4]
        if not veri:
            continue
        if args.match == "first":
            scelte = scelte[:1]
        errori.append(min(_gap(_angle(*s), v) for s in scelte for v in veri))

    risposte_su_escluse = 0
    escluse_lette = 0
    for voce in escluse:
        percorso = Path(str(voce.get("frame") or ""))
        rect = (voce.get("rect") or per_frame.get(str(percorso), {}).get("rect")
                or rect_di.get(str(percorso)))
        if not percorso.is_file() or not rect:
            continue
        escluse_lette += 1
        risposte_su_escluse += bool(_choices(percorso, rect, args.strict, args.order, args.flipped))

    esito = {
        "modo": "severo" if args.strict else "con ripesca",
        "confronto": args.match,
        "ordine": args.order + (" (ribaltato)" if args.flipped else ""),
        "fotogrammi": len(per_frame),
        "non_trovati_su_disco": mancanti,
        "risposte": len(errori),
        "senza_risposta": senza_risposta,
        "errore_mediano": round(statistics.median(errori), 3) if errori else None,
        "entro_1_grado": round(100.0 * sum(1 for e in errori if e <= 1.0) / len(errori), 1) if errori else None,
        "entro_3_gradi": round(100.0 * sum(1 for e in errori if e <= 3.0) / len(errori), 1) if errori else None,
        "escluse_lette": escluse_lette,
        "risposte_su_escluse": risposte_su_escluse,
    }
    for chiave, valore in esito.items():
        print(f"{chiave:24s} {valore}")
    if args.json_out:
        args.json_out.write_text(json.dumps(esito, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
