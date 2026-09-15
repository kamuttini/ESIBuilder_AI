#!/usr/bin/env python3
"""Draw the legacy guide lines on the acquisition frames, to see whether they sit on the needle.

The geometry port round-trips against itself, which proves the arithmetic and nothing about
the interpretation: if RECT_ECHO-relative coordinates or the sign of the centre distance were
read wrong, the round trip would still close. Only the image can say. So for each paired
configuration this renders the lines of #22/#23 over the calibration frames of its acquisition.

One line per depth, drawn in a colour ramp and labelled: the frame belongs to one depth and we
do not know which, so the right reading shows up as *one* of the lines landing on the needle
(or on the biopsy line, for biplane probes) while the others sit above and below it.

Only frames whose size matches the configuration's #09/#10 are used: on a frame of another
resolution the rectangle coordinates do not apply, and the drawing would be meaningless.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guides_geometry import Setup, line_in_frame, read_ndg, read_setup  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}
# Folders that hold calibration material, by the names the operators used.
CALIB_DIR = re.compile(r"agh|guid|biops", re.IGNORECASE)

RAMP = [
    (255, 80, 80), (255, 150, 60), (255, 220, 60), (150, 230, 80),
    (70, 220, 170), (70, 190, 255), (120, 140, 255), (200, 110, 255),
]


def calibration_frames(acquisition: Path, size: Tuple[int, int], limit: int) -> List[Path]:
    """Frames from calibration-looking sub-folders, at the configuration's own resolution."""
    hits: List[Path] = []
    fallback: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(acquisition):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        rel = os.path.relpath(dirpath, acquisition)
        is_calib = any(CALIB_DIR.search(part) for part in rel.split(os.sep))
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in IMAGE_EXTS or name.startswith("._"):
                continue
            path = Path(dirpath) / name
            target = hits if is_calib else fallback
            if len(target) >= limit * 4:
                continue
            try:
                with Image.open(path) as img:
                    if img.size != size:
                        continue
            except Exception:
                continue
            target.append(path)
        if len(hits) >= limit * 4:
            break
    chosen = hits or fallback
    if len(chosen) <= limit:
        return chosen
    step = len(chosen) / float(limit)
    return [chosen[min(len(chosen) - 1, int(round(i * step)))] for i in range(limit)]


def pick_depths(count: int, keep: int) -> List[int]:
    """A readable subset of the depths, spread over the range.

    Drawing every depth of every angle buries the frame under a hundred lines, and with that
    much ink something always falls near the needle: the check would prove nothing.
    """
    if keep <= 0 or count <= keep:
        return list(range(count))
    step = (count - 1) / float(keep - 1) if keep > 1 else 0
    return sorted({int(round(i * step)) for i in range(keep)})


def draw_lines(frame: Path, setup: Setup, distances: Sequence[Sequence[float]],
               max_width: int, depth_indexes: Sequence[int],
               angle_indexes: Sequence[int]) -> Tuple[str, List[str]]:
    with Image.open(frame) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    legend: List[str] = []

    for slot, depth_index in enumerate(depth_indexes):
        depth = setup.depths[depth_index]
        colour = RAMP[slot % len(RAMP)]
        drawn = False
        for angle_index in angle_indexes:
            per_angle = distances[angle_index] if angle_index < len(distances) else ()
            segment = line_in_frame(setup, depth_index, angle_index, 0, per_angle)
            if segment is None:
                continue
            draw.line(segment, fill=colour, width=2)
            drawn = True
        if drawn:
            legend.append(f'<span class="sw" style="background:rgb{colour}"></span>{depth:g} mm')

    rect = setup.rect_echo
    draw.rectangle([rect.left, rect.top, rect.right, rect.bottom], outline=(90, 90, 90), width=1)

    image.thumbnail((max_width, max_width), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), legend


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-configs", type=int, default=24)
    parser.add_argument("--frames-per-config", type=int, default=2)
    parser.add_argument("--min-confidence", type=str, default="alta")
    parser.add_argument("--probe-types", type=str, default="",
                        help="Comma separated #04 values to keep, e.g. \"1,2\" for linear+convex.")
    parser.add_argument("--image-width", type=int, default=900)
    parser.add_argument("--max-depths", type=int, default=6,
                        help="Depths drawn per frame; 0 for all. Few on purpose: see pick_depths.")
    parser.add_argument("--angle-index", type=int, default=-1,
                        help="Draw only this angle (-1 = all of them).")
    args = parser.parse_args()

    wanted = {int(v) for v in args.probe_types.split(",") if v.strip().isdigit()}
    rows = [
        row for row in csv.DictReader(args.pairs.open(encoding="utf-8"))
        if (args.min_confidence == "" or row["confidence"] == args.min_confidence)
        and (not wanted or int(row["probe_type"]) in wanted)
    ]
    print(f"coppie utilizzabili: {len(rows)}")

    blocks: List[str] = []
    used = 0
    for row in rows:
        if used >= args.max_configs:
            break
        setup = read_setup(Path(row["setup_file"]))
        if setup is None or not setup.consistent():
            continue
        kit = Path(row["config_path"]) / "DB_setup" / f"kit_needle_guide_{setup.kit_id}.ndg"
        distances = read_ndg(kit) or [[] for _ in range(setup.n_angles)]

        frames = calibration_frames(
            Path(row["acquisition"]), tuple(setup.video_size), args.frames_per_config
        )
        if not frames:
            continue

        depth_indexes = pick_depths(len(setup.depths), args.max_depths)
        angle_indexes = ([args.angle_index] if 0 <= args.angle_index < setup.n_angles
                         else list(range(setup.n_angles)))
        cards = []
        legend: List[str] = []
        for frame in frames:
            try:
                uri, legend = draw_lines(frame, setup, distances, args.image_width,
                                         depth_indexes, angle_indexes)
            except Exception as error:
                cards.append(f'<p class="err">{html.escape(str(error))}</p>')
                continue
            cards.append(
                f'<figure><img loading="lazy" src="{uri}" alt="">'
                f'<figcaption>{html.escape(str(frame.relative_to(Path(row["acquisition"]))))}'
                f'</figcaption></figure>'
            )
        blocks.append(
            f'<section><h2>{html.escape(row["config"])}</h2>'
            f'<p class="meta">tipo sonda {row["probe_type"]} &middot; kit {row["kit_id"]} &middot; '
            f'{row["n_angles"]} angolo/i &middot; {len(setup.depths)} depth &middot; '
            f'{row["video_w"]}&times;{row["video_h"]} &middot; '
            f'risoluzione combaciante {row["size_match_pct"]}%</p>'
            f'<p class="meta">acquisizione: {html.escape(Path(row["acquisition"]).name)}</p>'
            f'<p class="legend">{" ".join(legend)}</p>'
            f'{"".join(cards)}</section>'
        )
        used += 1
        print(f"  {used}/{args.max_configs}: {row['config'][:60]}", flush=True)

    document = f"""<!doctype html>
<html lang="it"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verifica linee guida legacy</title>
<style>
 body {{ font: 14px/1.6 -apple-system, system-ui, sans-serif; margin: 0 auto; padding: 26px;
        max-width: 1000px; background: #111; color: #eaeaea; }}
 h1 {{ font-size: 21px; }}
 h2 {{ font-size: 16px; margin: 0 0 4px; }}
 section {{ border-top: 1px solid #303030; padding-top: 18px; margin-top: 26px; }}
 .meta, .legend {{ color: #9a9a9a; font-size: 12px; margin: 2px 0; }}
 .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
        margin: 0 4px 0 10px; vertical-align: -1px; }}
 figure {{ margin: 12px 0; }}
 img {{ width: 100%; border-radius: 6px; background: #000; display: block; }}
 figcaption {{ color: #7d7d7d; font-size: 11px; margin-top: 4px; word-break: break-all; }}
 .err {{ color: #ff8a8a; }}
 .intro {{ color: #b5b5b5; max-width: 74ch; }}
</style>
<h1>Linee guida legacy disegnate sulle acquisizioni</h1>
<p class="intro">Ogni riga colorata &egrave; la prima linea guida di una depth, ricostruita dalle
righe #22 e #23 del <code>.fss</code> legacy e dalle distanze del <code>.ndg</code>. Il fotogramma
appartiene a una sola depth e non sappiamo quale: la lettura &egrave; corretta se <b>una</b> delle
linee cade sull'ago (o sulla linea di biopsia, per le biplane) e le altre le stanno sopra e sotto.
Il riquadro grigio &egrave; il RECT_ECHO.</p>
{"".join(blocks)}
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"\nreport: {args.output}  ({args.output.stat().st_size/1048576:.1f} MB, {used} configurazioni)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
