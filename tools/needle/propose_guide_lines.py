#!/usr/bin/env python3
"""From an acquisition folder to the #22 and #23 the guides step needs.

The whole chain in one place: find the calibration frames, detect the needle in each, turn it
into the legacy pair (angle, centre distance) with the ported geometry, and group the results
into the shape the .fss expects -- one angle per guide-line family in #23, one centre distance
per depth per angle in #22.

What it does NOT do is decide. Every proposal carries how many frames agreed, how much they
disagreed and a verdict, because on measured accuracy today roughly a third of the proposals
land inside the one-degree tolerance and about half within three. That is useful as a starting
point a person corrects, and is not usable unattended. The guides step is a confirm step by
design, so this hands over its evidence rather than its conclusion.

Depth is read from the sub-folder or file name when the acquisition says it (DEPTH_7, 3_65,
AGHI/60); when it does not, the proposal covers the angle only and the depth column stays for
the user.

Example:
  python3 tools/needle/propose_guide_lines.py \
    --folder "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION/191. Esaote MyLab xpro80 .../Aghi" \
    --rect 364,159,1468,882 --pixel-ratio-y 0.0561 --output proposta.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle import detect  # noqa: E402
from guides_geometry import clip_to_rect  # noqa: E402
from refine_needles import refine  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# depth written into a folder or file name: DEPTH_7, 3_65, 2_65.png, AGHI/60, 4)60-10
DEPTH_IN_NAME = re.compile(r"(?:depth[_\s-]*)(\d{1,3})|(?:^|[_\s)\-])(\d{2,3})(?:[_\s.\-]|$)",
                           re.IGNORECASE)


def depth_from_path(path: Path, root: Path) -> Optional[int]:
    """Depth in millimetres, if the acquisition wrote it into the path."""
    try:
        parts = list(path.relative_to(root).parts)
    except ValueError:
        parts = [path.name]
    for piece in reversed(parts):
        for match in DEPTH_IN_NAME.finditer(piece):
            value = match.group(1) or match.group(2)
            if value and 10 <= int(value) <= 250:
                return int(value)
    return None


def frames_in(folder: Path, size: Optional[Tuple[int, int]], cap: int) -> List[Path]:
    out: List[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or path.name.startswith("._"):
            continue
        if size is not None:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or (image.shape[1], image.shape[0]) != size:
                continue
        out.append(path)
        if len(out) >= cap:
            break
    return out


def line_across_rect(p1: Sequence[float], p2: Sequence[float],
                     rect: Tuple[int, int, int, int]) -> Optional[Tuple[float, float, float, float]]:
    """L'ago come retta dentro il rettangolo, non come il pezzo che era piu' luminoso.

    Quello che il rilevatore trova e' un tratto: dove il riflesso era abbastanza forte da
    superare la soglia. L'ago, e la linea guida che se ne ricava, attraversano tutto il
    rettangolo ecografico -- nel vecchio ESIBuilder la linea si disegnava da un bordo
    all'altro. Qui il tratto si prolunga nella sua direzione e si taglia sui quattro lati con
    lo stesso taglio parametrico del legacy (`upDateLine`), cosi' gli estremi cadono sul bordo
    e non fuori.
    """
    left, top, right, bottom = rect
    x1, y1 = float(p1[0]) - left, float(p1[1]) - top
    x2, y2 = float(p2[0]) - left, float(p2[1]) - top
    dx, dy = x2 - x1, y2 - y1
    lunghezza = math.hypot(dx, dy)
    if lunghezza < 1e-6:
        return None
    # abbondante: la diagonale basta a uscire dal rettangolo da tutte e due le parti
    passo = float(right - left + bottom - top)
    ux, uy = dx / lunghezza, dy / lunghezza
    esteso = (x1 - ux * passo, y1 - uy * passo, x2 + ux * passo, y2 + uy * passo)
    tagliato = clip_to_rect(esteso, right - left + 1, bottom - top + 1)
    if tagliato is None:
        return None
    return (tagliato[0] + left, tagliato[1] + top, tagliato[2] + left, tagliato[3] + top)


def measure_from_line(p1: Sequence[float], p2: Sequence[float],
                      rect: Tuple[int, int, int, int], ratio_x: float, ratio_y: float,
                      size: Optional[Sequence[int]] = None) -> Dict:
    """Da due punti sul fotogramma ad angolo e distanza, la strada del legacy.

    E' il pezzo di `measure` che non guarda l'immagine: serve anche quando la retta la
    corregge una persona invece del rilevatore, e il numero che ne esce deve nascere dalla
    stessa formula -- se no la correzione e la misura non sarebbero confrontabili.
    """
    left, top, right, bottom = rect
    fx1, fy1 = float(p1[0]), float(p1[1])
    fx2, fy2 = float(p2[0]), float(p2[1])
    x1, y1, x2, y2 = fx1 - left, fy1 - top, fx2 - left, fy2 - top
    angle = math.degrees(math.atan2((y2 - y1) * ratio_y, (x2 - x1) * ratio_x))

    centre_x = (right - left) / 2.0
    y_at_centre = None
    if abs(x2 - x1) < 1e-6:
        distance = (x1 + 1) * ratio_x - centre_x * ratio_x
    else:
        y_at_centre = y1 + (y2 - y1) * (centre_x - x1) / (x2 - x1)
        distance = (y_at_centre + 1) * ratio_y
    retta = line_across_rect((fx1, fy1), (fx2, fy2), rect)
    return {"angle": angle, "distance": distance,
            "p1": [fx1, fy1], "p2": [fx2, fy2],
            "line": [round(v, 2) for v in retta] if retta else None,
            "crossing": ([float(centre_x + left), float(y_at_centre + top)]
                         if y_at_centre is not None else None),
            "size": [int(size[0]), int(size[1])] if size else None}


def measure(frame: Path, rect: Tuple[int, int, int, int], ratio_x: float, ratio_y: float,
            strict: bool = True) -> Optional[Dict[str, float]]:
    """Angle in degrees and centre distance in millimetres, the legacy way.

    `strict` is what separates measuring from reviewing. A gallery wants the best guess even
    on a hopeless frame; a proposal wants silence, because a calibration folder also holds
    frames with no needle in them at all, and there the fallback invents a flat line -- eleven
    of those, on one real project, formed the largest "family" of guide lines in the proposal.
    Measured on the 66 frames of the diversity gallery, refusing to answer on 24 of them takes
    the median error from 3.48 to 1.20 degrees and within-one-degree from 30% to 45%, and the
    three frames Camilla marked as not-probe-in-water go from three answers to none.
    """
    gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    needles = refine(detect(gray, rect, top_k=6), gray, rect=rect, max_needles=2,
                     fallback=not strict)
    if not needles:
        return None
    needle = needles[0]
    # I punti stanno in coordinate del fotogramma, non del ritaglio: servono a disegnare la
    # misura sopra l'immagine cosi' com'e', ed e' l'unico modo di controllarla guardandola.
    misura = measure_from_line(needle.p1, needle.p2, rect, ratio_x, ratio_y,
                               (gray.shape[1], gray.shape[0]))
    misura.update({
        "confidence": float(getattr(needle, "source_score", 0.0)),
        "needles": [[float(n.p1[0]), float(n.p1[1]), float(n.p2[0]), float(n.p2[1])]
                    for n in needles],
        "lines": [[round(v, 2) for v in r] for r in
                  (line_across_rect(n.p1, n.p2, rect) for n in needles) if r],
    })
    return misura


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--rect", type=str, required=True,
                        help="RECT_ECHO as left,top,right,bottom (line #11 order is top|left|bottom|right)")
    parser.add_argument("--pixel-ratio-y", type=float, required=True, help="mm per pixel, #20")
    parser.add_argument("--pixel-ratio-x", type=float, default=None, help="defaults to ratio Y")
    parser.add_argument("--size", type=str, default="", help="only frames of WxH")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--angle-group-tol", type=float, default=4.0,
                        help="Degrees within which two frames are calibrating the same guide line.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rect = tuple(int(v) for v in args.rect.split(","))
    if len(rect) != 4:
        raise SystemExit("--rect vuole left,top,right,bottom")
    ratio_y = args.pixel_ratio_y
    ratio_x = args.pixel_ratio_x if args.pixel_ratio_x else ratio_y
    size = None
    if args.size:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))

    frames = frames_in(args.folder, size, args.max_frames)
    if not frames:
        raise SystemExit("nessun fotogramma utilizzabile nella cartella")

    misure = []
    for frame in frames:
        found = measure(frame, rect, ratio_x, ratio_y)
        if found is None:
            continue
        found["frame"] = str(frame)
        found["depth"] = depth_from_path(frame, args.folder)
        misure.append(found)
    if not misure:
        raise SystemExit("nessun ago rilevato")

    # raggruppo per angolo: ogni gruppo e' una famiglia di linee guida, cioe' una voce di #23
    gruppi: List[List[Dict]] = []
    for m in sorted(misure, key=lambda m: m["angle"]):
        for gruppo in gruppi:
            if abs(m["angle"] - statistics.median(g["angle"] for g in gruppo)) <= args.angle_group_tol:
                gruppo.append(m)
                break
        else:
            gruppi.append([m])
    gruppi.sort(key=lambda g: -len(g))

    proposte = []
    for gruppo in gruppi:
        angoli = [g["angle"] for g in gruppo]
        spread = (max(angoli) - min(angoli)) if len(angoli) > 1 else 0.0
        per_depth: Dict[Optional[int], List[float]] = defaultdict(list)
        for g in gruppo:
            per_depth[g["depth"]].append(g["distance"])
        verdetto = ("concorde" if len(gruppo) >= 3 and spread <= 2.0
                    else "da verificare" if len(gruppo) >= 2
                    else "un solo fotogramma")
        proposte.append({
            "angolo_23": round(statistics.median(angoli), 3),
            "fotogrammi": len(gruppo),
            "dispersione_angolo": round(spread, 2),
            "verdetto": verdetto,
            "distanze_22": {
                (str(d) if d is not None else "depth ignota"):
                    round(statistics.median(v), 3) for d, v in sorted(
                        per_depth.items(), key=lambda kv: (kv[0] is None, kv[0]))
            },
        })

    esito = {
        "cartella": str(args.folder),
        "rect": list(rect),
        "pixel_ratio": {"x": ratio_x, "y": ratio_y},
        "fotogrammi_esaminati": len(frames),
        "aghi_misurati": len(misure),
        "proposte": proposte,
        "avvertenza": ("Misurata su dati etichettati a mano: circa un terzo delle proposte cade "
                       "entro 1 grado e circa meta' entro 3. Da confermare, non da usare al buio."),
    }
    text = json.dumps(esito, indent=1, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")

    print(f"fotogrammi esaminati {len(frames)}, aghi misurati {len(misure)}\n")
    for i, p in enumerate(proposte, 1):
        print(f"famiglia {i}: #23 = {p['angolo_23']:+.2f}°  "
              f"({p['fotogrammi']} fotogrammi, dispersione {p['dispersione_angolo']}°, {p['verdetto']})")
        for depth, distance in p["distanze_22"].items():
            print(f"    depth {depth:>12}  ->  #22 = {distance:.2f} mm")
    if args.output:
        print(f"\nscritto: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
