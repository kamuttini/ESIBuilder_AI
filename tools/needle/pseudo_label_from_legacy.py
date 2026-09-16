#!/usr/bin/env python3
"""Label needles automatically, by asking the legacy where they should be.

For a paired configuration we already know the answer: #22 and #23 say where every guide line
runs, at every depth and for every angle, and that geometry has been checked against hand-traced
needles to 0.94 degrees. So on a calibration frame the needle is not something to search for
freely -- it must be ONE of those N angles by M depths lines.

That turns detection into verification, which is far easier: score the ridge along each
hypothesis and see which one the image actually supports. When one wins clearly, that line is a
label, with coordinates, for free.

The point is not to use the legacy at inference -- on a new machine #23 is exactly what must be
produced. It is to get enough labelled scenes to train a detector that then works without it.

Two guards, because pseudo-labels inherit every upstream error:
- only probe-consistent pairings, since a configuration measured on another probe's needles
  would teach the model that probe's angles;
- only frames where the winning hypothesis beats the runner-up by a margin, so an ambiguous
  frame contributes nothing rather than something wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guides_geometry import Setup, line_in_frame, read_ndg, read_setup  # noqa: E402
from needle_frames import NeedleScorer, all_frames, frames_of_probe, probe_tokens, _normalise  # noqa: E402

CALIB = ("agh",)


def ridge_score(gray: np.ndarray, p1, p2, samples: int = 128, gap: int = 6) -> float:
    """How much brighter the line is than its flanks. The same measure used everywhere else."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 20:
        return -1e9
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    t = np.linspace(0.0, 1.0, samples)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    h, w = gray.shape

    def at(offset: float) -> np.ndarray:
        xi = np.clip(np.round(xs + nx * offset).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys + ny * offset).astype(int), 0, h - 1)
        return gray[yi, xi].astype(np.float32)

    on = at(0.0)
    flanks = max(float(np.median(at(o))) for o in (gap, -gap, 2 * gap, -2 * gap))
    return float(np.median(on) - flanks)


def hypotheses(setup: Setup, distances
               ) -> List[Tuple[int, int, int, Tuple[float, float, float, float]]]:
    """Every line the kit can draw: angle by depth by hole.

    Testing only the first line of each angle was the mistake that made this return almost
    nothing. The .ndg holds up to eight lines per angle -- the holes of the guide -- and the
    operator puts the needle in whichever hole the calibration calls for, so the visible needle
    is line k, not line 0.
    """
    out = []
    for angle_index in range(setup.n_angles):
        per_angle = distances[angle_index] if angle_index < len(distances) else ()
        lines = len(per_angle) + 1
        for depth_index in range(len(setup.depths)):
            for line_index in range(lines):
                segment = line_in_frame(setup, depth_index, angle_index, line_index, per_angle)
                if segment is not None:
                    out.append((angle_index, depth_index, line_index, segment))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None,
                        help="Needle classifier, to confirm a frame is calibration material.")
    parser.add_argument("--probe-types", type=str, default="1,2")
    parser.add_argument("--frames-per-config", type=int, default=8)
    parser.add_argument("--min-margin", type=float, default=6.0,
                        help="How far the winning hypothesis must beat the runner-up, in grey levels.")
    parser.add_argument("--min-score", type=float, default=10.0)
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    wanted = {int(v) for v in args.probe_types.split(",") if v.strip().isdigit()}
    rows = [r for r in csv.DictReader(args.pairs.open(encoding="utf-8"))
            if not wanted or int(r["probe_type"]) in wanted]
    scorer = NeedleScorer(args.model) if args.model and args.model.is_file() else None

    etichette: List[Dict] = []
    stats = {"config": 0, "frames": 0, "accepted": 0, "ambiguous": 0, "weak": 0,
             "no_probe_match": 0}

    for index, row in enumerate(rows, 1):
        if args.max_configs and stats["config"] >= args.max_configs:
            break
        cached = args.legacy_cache / Path(row["config_path"]).name / "DB_setup" / Path(row["setup_file"]).name
        setup = read_setup(cached if cached.is_file() else Path(row["setup_file"]))
        if setup is None or not setup.consistent():
            continue
        kit = cached.parent / f"kit_needle_guide_{setup.kit_id}.ndg"
        distances = read_ndg(kit) or [[] for _ in range(setup.n_angles)]

        acquisition = Path(row["acquisition"])
        if not acquisition.is_dir():
            continue
        pool = [p for p in all_frames(acquisition, tuple(setup.video_size), cap=300)
                if any(any(c in part.lower() for c in CALIB)
                       for part in os.path.relpath(p.parent, acquisition).split(os.sep))]
        if not pool:
            continue
        # la sonda deve essere quella della configurazione, o si misura su aghi di un'altra
        tokens = [_normalise(t) for t in probe_tokens(row["config"])]
        filtered = frames_of_probe(pool, acquisition, row["config"])
        if tokens and len(filtered) == len(pool) and any(
                tokens and not any(t in _normalise(str(p.parent.relative_to(acquisition)))
                                   for t in tokens) for p in pool[:1]):
            stats["no_probe_match"] += 1
        pool = filtered[: args.frames_per_config * 4]

        rect = (setup.rect_echo.left, setup.rect_echo.top,
                setup.rect_echo.right, setup.rect_echo.bottom)
        if scorer is not None and pool:
            punteggi = scorer.score(pool, rect)
            pool = [p for p, v in sorted(zip(pool, punteggi), key=lambda z: -z[1]) if v >= 0.6]
        pool = pool[: args.frames_per_config]
        if not pool:
            continue
        stats["config"] += 1

        ipotesi = hypotheses(setup, distances)
        if not ipotesi:
            continue
        for frame in pool:
            gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            stats["frames"] += 1
            punteggiate = [(ridge_score(gray, (s[0], s[1]), (s[2], s[3])), a, d, li, s)
                           for a, d, li, s in ipotesi]
            punteggiate.sort(reverse=True, key=lambda z: z[0])
            best = punteggiate[0]
            if best[0] < args.min_score:
                stats["weak"] += 1
                continue
            # il secondo migliore con un ANGOLO diverso: due depth dello stesso angolo sono
            # linee parallele vicine, non un'ambiguita' su quale ago sia
            rivale = next((p for p in punteggiate[1:] if p[1] != best[1]), None)
            margine = best[0] - (rivale[0] if rivale else 0.0)
            if rivale is not None and margine < args.min_margin:
                stats["ambiguous"] += 1
                continue
            stats["accepted"] += 1
            etichette.append({
                "frame": str(frame), "config": row["config"], "rect": list(rect),
                "line": [round(v, 1) for v in best[4]],
                "angle": round(setup.angles[best[1]], 3),
                "depth_mm": setup.depths[best[2]],
                "hole": best[3],
                "score": round(best[0], 2), "margin": round(margine, 2),
            })
        if index % 20 == 0:
            print(f"  {index}/{len(rows)} configurazioni, {stats['accepted']} etichette", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"stats": stats, "etichette": etichette},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nconfigurazioni usate {stats['config']}   fotogrammi {stats['frames']}")
    print(f"accettate {stats['accepted']}   ambigue {stats['ambiguous']}   deboli {stats['weak']}")
    print(f"scritto: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
