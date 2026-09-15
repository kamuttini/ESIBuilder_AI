#!/usr/bin/env python3
"""Test whether a proposed visual cue actually separates needles from the detector's mistakes.

Every hint Camilla gives -- brighter, with ridges, jagged rather than smooth, a more visible
tip -- is a real thing she sees. Whether it survives being turned into one number is a separate
question, and guessing wrong costs a day: the first attempt at "ridges" made precision worse
because it measured reverberation bands below the needle, which is not what she meant.

So each cue gets measured before it gets coded into the pipeline. Positives are her traced
needles; negatives are the candidates the detector returns on those same frames pointing
somewhere else -- its real mistakes, not random lines, which separate trivially and prove
nothing. Separation is the gap between the medians in units of the pooled spread: below about
0.5 a feature is not worth having, and no single feature here has ever passed 1.2.

Run it whenever a new cue comes up:
  python3 tools/needle/measure_needle_features.py \
    --annotations artifacts/92_guides/etichette/aghi_tracciati_camilla_v2_2026-09-15.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle import detect  # noqa: E402
from refine_needles import _angle_gap  # noqa: E402


def sample_along(gray: np.ndarray, p1, p2, count: int = 192, offset: float = 0.0
                 ) -> Optional[np.ndarray]:
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 16:
        return None
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    t = np.linspace(0.0, 1.0, count)
    xs = x1 + t * (x2 - x1) + nx * offset
    ys = y1 + t * (y2 - y1) + ny * offset
    h, w = gray.shape
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    return gray[yi, xi].astype(np.float32)


def features(gray: np.ndarray, p1, p2) -> Optional[Dict[str, float]]:
    on = sample_along(gray, p1, p2)
    if on is None:
        return None
    median = float(np.median(on))
    smooth = np.convolve(on, np.ones(11) / 11, mode="same")
    residual = on - smooth

    sides = [sample_along(gray, p1, p2, offset=o) for o in (-10.0, -5.0, 5.0, 10.0)]
    if any(s is None for s in sides):
        return None
    flanks = [float(np.median(s)) for s in sides]

    spectrum = np.abs(np.fft.rfft(residual * np.hanning(len(residual))))[1: len(residual) // 4]
    if spectrum.sum() <= 0:
        return None
    weights = spectrum / spectrum.sum()
    half = len(on) // 2
    sixth = len(on) // 6

    return {
        "luminosita": median,
        "cresta sottile": median - 0.5 * (flanks[0] + flanks[3]),
        "frastagliatura": float(np.std(residual)),
        "periodicita (picco spettrale)": float(spectrum.max() / (spectrum.mean() + 1e-9)),
        "periodicita (entropia)": float(-(weights * np.log(weights + 1e-12)).sum()
                                        / math.log(len(weights))),
        "punta (estremo piu' brillante)": float(max(np.max(on[:sixth]), np.max(on[-sixth:]))
                                                - median),
        "asimmetria verso la punta": float(abs(np.median(on[:half]) - np.median(on[half:]))
                                           / max(1.0, median)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--negative-angle-gap", type=float, default=12.0)
    parser.add_argument("--candidates", type=int, default=6)
    args = parser.parse_args()

    traced: Dict[str, Dict] = {}
    for path in args.annotations:
        for entry in json.loads(path.read_text(encoding="utf-8"))["annotazioni"]:
            record = traced.setdefault(entry["frame"], {"rect": entry["rect"], "lines": []})
            record["lines"].extend(entry["lines"])

    positives: List[Dict[str, float]] = []
    negatives: List[Dict[str, float]] = []
    for frame, record in traced.items():
        gray = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        angles = [math.degrees(math.atan2(l[3] - l[1], l[2] - l[0])) for l in record["lines"]]
        for line in record["lines"]:
            found = features(gray, (line[0], line[1]), (line[2], line[3]))
            if found:
                positives.append(found)
        for cand in detect(gray, tuple(record["rect"]), top_k=args.candidates):
            if all(_angle_gap(cand.angle_deg, a) > args.negative_angle_gap for a in angles):
                found = features(gray, cand.p1, cand.p2)
                if found:
                    negatives.append(found)

    if not positives or not negatives:
        raise SystemExit("servono sia aghi tracciati sia candidati sbagliati")

    print(f"aghi tracciati: {len(positives)}   errori del rilevatore: {len(negatives)}\n")
    print(f"{'caratteristica':34} {'ago':>9} {'non-ago':>9} {'separazione':>12}")
    rows = []
    for name in positives[0]:
        a = [p[name] for p in positives]
        b = [n[name] for n in negatives]
        spread = (statistics.pstdev(a) + statistics.pstdev(b)) / 2 or 1.0
        rows.append((abs(statistics.median(a) - statistics.median(b)) / spread,
                     name, statistics.median(a), statistics.median(b)))
    for separation, name, ma, mb in sorted(rows, reverse=True):
        print(f"{name:34} {ma:9.2f} {mb:9.2f} {separation:12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
