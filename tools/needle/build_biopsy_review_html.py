#!/usr/bin/env python3
"""Show what the biopsy-guide detector found on biplane acquisitions.

Biplane probes calibrate on the machine's own biopsy lines -- a row of coloured dots the screen
draws -- and not on needles in water, so nothing in the needle pipeline applies to them. This is
the first look at whether the dots can be found reliably.

Frames are taken from the calibration folders of biplane acquisitions, one acquisition at a time
for variety. Where the frame is at the resolution its configuration declares, the search is
limited to RECT_ECHO; where it is not, the whole frame is used, since the dots are identified by
colour and alignment rather than by position.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_biopsy_dots import detect  # noqa: E402
from guides_geometry import read_setup  # noqa: E402

CALIB = re.compile(r"agh|guid|biops", re.IGNORECASE)


def first_frames(acquisition: Path, limit: int) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(acquisition):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "$"))]
        rel = os.path.relpath(dirpath, acquisition)
        if not any(CALIB.search(part) for part in rel.split(os.sep)):
            continue
        names = sorted(n for n in filenames if Path(n).suffix.lower() == ".png")
        step = max(1, len(names) // max(1, limit))
        for name in names[::step][:limit]:
            out.append(Path(dirpath) / name)
        if len(out) >= limit:
            break
    return out[:limit]


def encode(image: np.ndarray, width: int) -> str:
    scale = width / float(image.shape[1])
    if scale < 1:
        image = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-acquisitions", type=int, default=40)
    parser.add_argument("--frames-per-acquisition", type=int, default=2)
    parser.add_argument("--width", type=int, default=760)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(args.pairs.open(encoding="utf-8"))
            if r["probe_type"] in ("3", "4")]
    migliori: Dict[str, dict] = {}
    for row in rows:
        key = row["acquisition"]
        if key not in migliori or float(row["size_match_pct"]) > float(migliori[key]["size_match_pct"]):
            migliori[key] = row
    rows = sorted(migliori.values(), key=lambda r: r["config"])

    cards: List[str] = []
    saltati: List[str] = []
    trovate = provate = 0
    for row in rows[: args.max_acquisitions]:
        cached = (args.legacy_cache / Path(row["config_path"]).name / "DB_setup"
                  / Path(row["setup_file"]).name)
        setup = read_setup(cached if cached.is_file() else Path(row["setup_file"]))
        if setup is None:
            continue
        acquisition = Path(row["acquisition"])
        if not acquisition.is_dir():
            continue
        for frame in first_frames(acquisition, args.frames_per_acquisition):
            image = cv2.imread(str(frame))
            if image is None:
                continue
            provate += 1
            # RECT_ECHO is not optional here. Run on the whole frame and the detector finds the
            # coloured text of the interface instead -- six detections on the first try, all of
            # them machine labels and menu buttons, none a guide. The rectangle is what separates
            # the drawing the machine makes inside the image from the drawing it makes around it.
            if (image.shape[1], image.shape[0]) != tuple(setup.video_size):
                saltati.append(f"{frame.name}: {image.shape[1]}x{image.shape[0]} invece di "
                               f"{setup.video_size[0]}x{setup.video_size[1]}")
                continue
            rect = (setup.rect_echo.left, setup.rect_echo.top,
                    setup.rect_echo.right, setup.rect_echo.bottom)
            nota = "dentro RECT_ECHO"
            guida = detect(image, rect)
            canvas = image.copy()
            if guida:
                trovate += 1
                for dot in guida.dots:
                    cv2.circle(canvas, (int(dot[0]), int(dot[1])), 7, (0, 0, 255), 2)
                cv2.line(canvas, (int(guida.p1[0]), int(guida.p1[1])),
                         (int(guida.p2[0]), int(guida.p2[1])), (255, 0, 255), 2)
                didascalia = (f"<b>{guida.count} pallini</b> &middot; angolo "
                              f"{guida.angle_deg:.1f}&deg; &middot; scostamento "
                              f"{guida.straightness:.2f} px")
            else:
                didascalia = '<span class="no">nessuna guida trovata</span>'
            cards.append(
                f'<figure><img loading="lazy" src="{encode(canvas, args.width)}" alt="">'
                f'<figcaption>{didascalia}<br>'
                f'<span class="path">{html.escape(row["config"][:54])}</span><br>'
                f'<span class="path">{html.escape(frame.name)} &middot; {nota}</span>'
                f'</figcaption></figure>'
            )

    document = f"""<!doctype html>
<html lang="it"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guida di biopsia &mdash; sonde biplana</title>
<style>
 body {{ font: 14px/1.6 -apple-system, system-ui, sans-serif; margin: 0 auto; padding: 24px;
        max-width: 1250px; background: #111; color: #eaeaea; }}
 h1 {{ font-size: 21px; margin-bottom: 2px; }}
 .intro {{ color: #b0b0b0; max-width: 78ch; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px;
          margin-top: 20px; }}
 figure {{ margin: 0; background: #1b1b1b; border-radius: 8px; padding: 9px; }}
 img {{ width: 100%; display: block; border-radius: 4px; background: #000; }}
 figcaption {{ font-size: 12px; color: #cfcfcf; margin-top: 7px; }}
 .path {{ color: #777; font-size: 11px; word-break: break-all; }}
 .no {{ color: #ffab6b; }}
</style>
<h1>Guida di biopsia &mdash; {trovate} trovate su {provate} fotogrammi</h1>
<p class="intro">Le sonde biplana non si calibrano sugli aghi in acqua: si accendono le linee di
biopsia dell'ecografo, che disegna la guida come una fila di pallini colorati. I cerchi rossi sono
i pallini trovati, la linea magenta la retta che li interpola. Il segnale sfruttato &egrave; che
l'ecografia &egrave; in scala di grigi e la guida no.</p>
<div class="grid">{"".join(cards)}</div>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"{trovate}/{provate} guide trovate  ->  {args.output} "
          f"({args.output.stat().st_size/1048576:.1f} MB)")
    if saltati:
        print(f"saltati per risoluzione diversa da quella dichiarata: {len(saltati)}")
        for riga in saltati[:5]:
            print("   ", riga)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
