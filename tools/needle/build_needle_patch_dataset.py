#!/usr/bin/env python3
"""Build a patch dataset for ranking line candidates, from the needles Camilla traced.

The classical detector's weakness is choosing, not proposing: on half the frames the needle is
among its candidates and not first. A ranker needs examples of both, with the same geometry
description, which is what the tracings give.

Positives are the traced needles. Negatives are candidates the detector returns on those same
frames that point somewhere else -- so the negatives are the mistakes this detector actually
makes, not random lines: tissue bands, the skin line, reflections in the water.

Each sample is a strip sampled ALONG the candidate: the line laid flat in the middle of the
patch, with its surroundings above and below. That is the view the decision needs -- is this
a bright ridge with reverberation, or a band edge -- and it removes the angle from the problem,
so the ranker cannot learn "needles point down-right in this vendor".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle import detect  # noqa: E402
from refine_needles import _angle_gap  # noqa: E402

PATCH_W = 160
PATCH_H = 48


def strip_along(gray: np.ndarray, p1, p2, width: int = PATCH_W, height: int = PATCH_H,
                margin: float = 1.35) -> Optional[np.ndarray]:
    """The candidate line laid flat: sampled along it, with its neighbourhood above and below."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 16:
        return None
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -uy, ux
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    span = length * margin

    t = np.linspace(-0.5, 0.5, width) * span
    s = np.linspace(-0.5, 0.5, height) * height
    tt, ss = np.meshgrid(t, s)
    xs = cx + tt * ux + ss * nx
    ys = cy + tt * uy + ss * ny

    h, w = gray.shape
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    return gray[yi, xi]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, action="append", required=True,
                        help="JSON exported from the review gallery. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-angle-gap", type=float, default=12.0,
                        help="A candidate further than this from every traced needle is a negative.")
    parser.add_argument("--candidates-per-frame", type=int, default=6)
    parser.add_argument("--random-negatives", type=int, default=14,
                        help="Random lines per frame, away from every traced needle. The "
                             "detector alone yields too few negatives to balance the set, and "
                             "these cover what it never proposes: black areas, tissue bands, "
                             "anything that is simply not a needle.")
    parser.add_argument("--jitter", type=int, default=4,
                        help="Extra positives per needle, shifted and shortened slightly.")
    args = parser.parse_args()

    traced: dict[str, dict] = {}
    for path in args.annotations:
        for entry in json.loads(path.read_text(encoding="utf-8"))["annotazioni"]:
            frame = entry["frame"]
            record = traced.setdefault(frame, {"rect": entry["rect"], "lines": [],
                                               "config": entry["config"]})
            record["lines"].extend(entry["lines"])

    out = args.output_dir.expanduser().resolve()
    (out / "pos").mkdir(parents=True, exist_ok=True)
    (out / "neg").mkdir(parents=True, exist_ok=True)
    index: List[dict] = []
    counts = {"pos": 0, "neg": 0, "frames": 0, "no_image": 0}

    rng = np.random.default_rng(42)

    for frame, record in sorted(traced.items()):
        gray = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            counts["no_image"] += 1
            continue
        counts["frames"] += 1
        rect = tuple(record["rect"])
        angles = []

        for k, line in enumerate(record["lines"]):
            x1, y1, x2, y2 = line
            angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            variants = [((x1, y1), (x2, y2))]
            for _ in range(args.jitter):
                # the same needle traced slightly differently: a human hand would
                shift = rng.normal(0, 2.5, 4)
                shorten = rng.uniform(0.0, 0.18)
                dx, dy = (x2 - x1) * shorten * 0.5, (y2 - y1) * shorten * 0.5
                variants.append(((x1 + dx + shift[0], y1 + dy + shift[1]),
                                 (x2 - dx + shift[2], y2 - dy + shift[3])))
            for j, (a, b) in enumerate(variants):
                patch = strip_along(gray, a, b)
                if patch is None:
                    continue
                name = f"{counts['pos']:05d}.png"
                cv2.imwrite(str(out / "pos" / name), patch)
                index.append({"file": f"pos/{name}", "label": 1, "frame": frame,
                              "config": record["config"], "needle": k, "variant": j})
                counts["pos"] += 1

        # random lines inside the rectangle, with the positives' length distribution
        left, top, right, bottom = rect
        lengths = [math.hypot(l[2] - l[0], l[3] - l[1]) for l in record["lines"]] or [200.0]
        drawn = 0
        for _ in range(args.random_negatives * 6):
            if drawn >= args.random_negatives:
                break
            angle = float(rng.uniform(-90, 90))
            length = float(rng.choice(lengths)) * float(rng.uniform(0.6, 1.2))
            cx = float(rng.uniform(left + 0.15 * (right - left), right - 0.15 * (right - left)))
            cy = float(rng.uniform(top + 0.1 * (bottom - top), bottom - 0.1 * (bottom - top)))
            ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            a = (cx - ux * length / 2, cy - uy * length / 2)
            b = (cx + ux * length / 2, cy + uy * length / 2)
            if any(_angle_gap(angle, ang) <= args.negative_angle_gap for ang in angles):
                # same direction as a needle: only keep it if it is somewhere else entirely
                near = min(abs((a[1] + b[1]) / 2 - (l[1] + l[3]) / 2) for l in record["lines"])
                if near < 60:
                    continue
            patch = strip_along(gray, a, b)
            if patch is None:
                continue
            name = f"{counts['neg']:05d}.png"
            cv2.imwrite(str(out / "neg" / name), patch)
            index.append({"file": f"neg/{name}", "label": 0, "frame": frame,
                          "config": record["config"], "method": "random"})
            counts["neg"] += 1
            drawn += 1

        for cand in detect(gray, rect, top_k=args.candidates_per_frame):
            if any(_angle_gap(cand.angle_deg, a) <= args.negative_angle_gap for a in angles):
                continue
            patch = strip_along(gray, cand.p1, cand.p2)
            if patch is None:
                continue
            name = f"{counts['neg']:05d}.png"
            cv2.imwrite(str(out / "neg" / name), patch)
            index.append({"file": f"neg/{name}", "label": 0, "frame": frame,
                          "config": record["config"], "method": cand.method})
            counts["neg"] += 1

    (out / "index.json").write_text(json.dumps({"counts": counts, "samples": index},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"fotogrammi: {counts['frames']} (immagini mancanti: {counts['no_image']})")
    print(f"positivi: {counts['pos']}   negativi: {counts['neg']}")
    print(f"output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
