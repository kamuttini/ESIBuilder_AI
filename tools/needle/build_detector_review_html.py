#!/usr/bin/env python3
"""Show what the needle detector found, frame by frame, so a human can judge it.

The detector is the measuring instrument for the whole guides chain: if it sits a few degrees
off the needle, every number downstream is off with it, and no later stage can notice. So the
point of this page is not the count of detections but whether each red segment lies on the
needle -- which only someone who knows what a needle looks like can say.

Frames come from the calibration folders of the acquisitions paired with the legacy
configurations, at the resolution the configuration declares, cropped to RECT_ECHO.
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
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle_line import candidates_in_frame  # noqa: E402
from guides_geometry import read_setup  # noqa: E402
from needle_frames import NeedleScorer, all_frames  # noqa: E402

CALIB_DIR = re.compile(r"agh|guid|biops", re.IGNORECASE)
# BGR, as OpenCV wants them. One colour and one letter per needle, because several are
# usually visible and a verdict on "the detection" is meaningless when there are four.
SEGMENT_COLOURS = [(0, 0, 255), (0, 220, 255), (0, 255, 120), (255, 180, 0), (255, 120, 255)]
SEGMENT_LETTERS = "abcde"
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}


def frames_of(acquisition: Path, size: Tuple[int, int], limit: int,
              scorer: Optional["NeedleScorer"] = None,
              rect: Optional[Tuple[int, int, int, int]] = None,
              threshold: float = 0.60, scan_cap: int = 80) -> List[Path]:
    """Frames to measure: inside a calibration folder AND confirmed by the classifier.

    Neither test alone is right. Folder names carry the intent -- these acquisitions are filed
    by angle and depth, which is exactly what the calibration needs -- but they also contain
    frames that are not calibration material at all. The classifier answers "is there a needle",
    which is a different question: on its own it happily picks a needle from a biopsy demo
    filed elsewhere. Measured over twenty configurations, agreement with the legacy angles was
    33% by folder alone and 39% for the two together, 40% against 49% counting the shortlist.
    """
    if scorer is not None and rect is not None:
        pool = [
            path for path in all_frames(acquisition, size, cap=400)
            if any(CALIB_DIR.search(part)
                   for part in os.path.relpath(path.parent, acquisition).split(os.sep))
        ][:scan_cap]
        if pool:
            scores = scorer.score(pool, rect)
            ranked = [p for p, v in sorted(zip(pool, scores), key=lambda z: -z[1]) if v >= threshold]
            if ranked:
                return ranked[:limit]
    return _frames_by_folder(acquisition, size, limit)


def _frames_by_folder(acquisition: Path, size: Tuple[int, int], limit: int) -> List[Path]:
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(acquisition):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        rel = os.path.relpath(dirpath, acquisition)
        if not any(CALIB_DIR.search(part) for part in rel.split(os.sep)):
            continue
        names = sorted(n for n in filenames if Path(n).suffix.lower() == ".png")
        step = max(1, len(names) // max(1, limit))
        for name in names[::step][: limit * 2]:
            found.append(Path(dirpath) / name)
            if len(found) >= limit:
                return found
    return found


def _voti_html(number: int, n_segments: int) -> str:
    """One row of buttons per needle: with several drawn, a single verdict says nothing."""
    if n_segments == 0:
        return ('<div class="vote"><button onclick="vota(\'%d\',\'manca\')">'
                'c\'era un ago</button></div>' % number)
    rows = []
    for order in range(n_segments):
        letter = SEGMENT_LETTERS[order % len(SEGMENT_LETTERS)]
        key = f"{number}{letter}"
        rows.append(
            f'<div class="vote"><span class="seg">{letter}</span>'
            f'<button onclick="vota(\'{key}\',\'ok\')">sull\'ago</button>'
            f'<button onclick="vota(\'{key}\',\'no\')">sbagliato</button>'
            f'<button class="ghost" onclick="vota(\'{key}\',\'\')">&times;</button></div>'
        )
    rows.append(f'<div class="vote"><button class="ghost wide" '
                f'onclick="vota(\'{number}m\',\'manca\')">ne manca uno</button></div>')
    return "".join(rows)


def encode(image: np.ndarray, width: int) -> str:
    scale = width / float(image.shape[1])
    if scale < 1:
        image = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-types", type=str, default="1,2")
    parser.add_argument("--max-configs", type=int, default=14)
    parser.add_argument("--frames-per-config", type=int, default=3)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--per-frame", type=int, default=4,
                        help="Needles drawn per frame. Several are usually visible, and the "
                             "brightest is not necessarily the one being calibrated.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Needle classifier used to confirm the frames. Without it, folder "
                             "names decide alone and non-calibration frames get through.")
    args = parser.parse_args()

    scorer = NeedleScorer(args.model) if args.model and args.model.is_file() else None
    if scorer is None:
        print("attenzione: nessun classificatore, i fotogrammi li scelgono solo i nomi delle cartelle")

    wanted = {int(v) for v in args.probe_types.split(",") if v.strip().isdigit()}
    rows = [
        r for r in csv.DictReader(args.pairs.open(encoding="utf-8"))
        if r["confidence"] == "alta" and (not wanted or int(r["probe_type"]) in wanted)
    ]

    cards: List[str] = []
    found = total = 0
    for row in rows[: args.max_configs]:
        setup = read_setup(Path(row["setup_file"]))
        if setup is None or not setup.consistent():
            continue
        size = tuple(setup.video_size)
        rect = (setup.rect_echo.left, setup.rect_echo.top,
                setup.rect_echo.right, setup.rect_echo.bottom)

        for frame in frames_of(Path(row["acquisition"]), size, args.frames_per_config,
                               scorer=scorer, rect=rect):
            gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if gray is None or (gray.shape[1], gray.shape[0]) != size:
                continue
            total += 1
            detections = candidates_in_frame(gray, rect, top_k=args.per_frame)
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for order, detection in enumerate(detections):
                colour = SEGMENT_COLOURS[order % len(SEGMENT_COLOURS)]
                cv2.line(canvas,
                         (int(detection.p1[0]), int(detection.p1[1])),
                         (int(detection.p2[0]), int(detection.p2[1])), colour, 2)
                cv2.putText(canvas, SEGMENT_LETTERS[order % len(SEGMENT_LETTERS)],
                            (int(detection.p2[0]) + 6, int(detection.p2[1]) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
            if detections:
                found += 1
            crop = canvas[rect[1]:rect[3], rect[0]:rect[2]]
            uri = encode(crop, args.width)
            if not detections:
                caption = '<span class="no">nessuna rilevazione</span>'
            else:
                parts = []
                for order, detection in enumerate(detections):
                    letter = SEGMENT_LETTERS[order % len(SEGMENT_LETTERS)]
                    rgb = SEGMENT_COLOURS[order % len(SEGMENT_COLOURS)]
                    parts.append(
                        f'<span class="seg" style="color:rgb({rgb[2]},{rgb[1]},{rgb[0]})">'
                        f'{letter}</span> {detection.angle_deg:.1f}&deg;')
                caption = " &nbsp; ".join(parts)
            number = len(cards) + 1
            cards.append(
                f'<figure id="c{number}" data-n="{number}">'
                f'<div class="head"><span class="num">{number}</span>'
                f'<span class="mark" id="m{number}"></span></div>'
                f'<img loading="lazy" src="{uri}" alt="">'
                f'<div class="vote">'
                f'<button onclick="vota({number},\'ok\')">sull\'ago</button>'
                f'<button onclick="vota({number},\'no\')">sbagliato</button>'
                f'<button class="ghost" onclick="vota({number},\'\')">annulla</button>'
                f'</div>'
                f'<figcaption>{caption}<br>'
                f'<span class="path">{html.escape(row["config"][:52])}</span><br>'
                f'<span class="path">{html.escape(frame.name)}</span></figcaption></figure>'
            )

    document = f"""<!doctype html>
<html lang="it"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rilevatore ago &mdash; verifica</title>
<style>
 body {{ font: 14px/1.6 -apple-system, system-ui, sans-serif; margin: 0 auto; padding: 24px;
        max-width: 1250px; background: #111; color: #eaeaea; }}
 h1 {{ font-size: 21px; margin-bottom: 2px; }}
 .intro {{ color: #b0b0b0; max-width: 76ch; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px;
          margin-top: 20px; }}
 figure {{ margin: 0; background: #1b1b1b; border-radius: 8px; padding: 9px; }}
 img {{ width: 100%; display: block; border-radius: 4px; background: #000; }}
 figcaption {{ font-size: 12px; color: #cfcfcf; margin-top: 7px; }}
 .path {{ color: #777; font-size: 11px; word-break: break-all; }}
 .no {{ color: #ffab6b; }}
 .head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
 .num {{ font-weight: 700; color: #8fb6ff; }}
 .mark {{ font-size: 12px; }}
 .vote {{ display: flex; gap: 6px; margin-top: 6px; align-items: center; }}
 .seg {{ font-weight: 700; width: 14px; display: inline-block; }}
 .vote .wide {{ flex: 1; }}
 .vote button {{ flex: 1; padding: 5px 4px; font-size: 12px; border-radius: 5px; cursor: pointer;
                 border: 1px solid #3a3a3a; background: #262626; color: #e8e8e8; }}
 .vote button:hover {{ background: #333; }}
 .vote .ghost {{ flex: 0 0 62px; color: #999; }}
 figure.ok {{ outline: 2px solid #35c88a; }}
 figure.no {{ outline: 2px solid #f2564d; }}
 #barra {{ position: sticky; top: 0; z-index: 5; background: #191919; border: 1px solid #333;
           border-radius: 8px; padding: 10px 14px; margin-top: 16px; }}
 #esito {{ width: 100%; min-height: 62px; margin-top: 8px; background: #101010; color: #ddd;
           border: 1px solid #333; border-radius: 6px; padding: 8px; font-family: ui-monospace, monospace;
           font-size: 12px; }}
</style>
<h1>Rilevatore dell'ago &mdash; {found} rilevazioni su {total} fotogrammi</h1>
<p class="intro">Il segmento rosso &egrave; quello che il programma ha preso per l'ago, dentro il
RECT_ECHO. Quello che conta non &egrave; quante ne trova, ma se ognuna sta <b>sull'ago</b>: questo
&egrave; lo strumento di misura di tutta la catena delle linee guida, e se sbaglia di qualche grado
sbagliano tutti i numeri a valle senza che nessuno se ne accorga.</p>
<div id="barra">
  <b>Come segnalarmele:</b> ogni ago disegnato ha la sua lettera
  (<b>a</b>, <b>b</b>, <b>c</b>&hellip;) e la sua riga di bottoni, quindi un riquadro pu&ograve;
  avere <i>7a</i> giusto e <i>7b</i> sbagliato. Se il programma ha perso un ago che si vede,
  premi <i>ne manca uno</i>. Il riepilogo qui sotto si aggiorna da solo: copialo e incollamelo.
  <span id="conta" class="path"></span>
  <textarea id="esito" readonly></textarea>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
const CHIAVE = 'rilevatore_ago_voti';
let voti = {{}};
try {{ voti = JSON.parse(localStorage.getItem(CHIAVE) || '{{}}'); }} catch (e) {{ voti = {{}}; }}

function vota(n, valore) {{
  if (valore) voti[n] = valore; else delete voti[n];
  try {{ localStorage.setItem(CHIAVE, JSON.stringify(voti)); }} catch (e) {{}}
  disegna();
}}

function disegna() {{
  document.querySelectorAll('figure[data-n]').forEach((fig) => {{
    const n = fig.dataset.n;
    const suoi = Object.keys(voti).filter((k) => parseInt(k, 10) === parseInt(n, 10));
    const buoni = suoi.filter((k) => voti[k] === 'ok').length;
    const cattivi = suoi.filter((k) => voti[k] === 'no').length;
    fig.classList.toggle('ok', buoni > 0 && cattivi === 0);
    fig.classList.toggle('no', cattivi > 0 && buoni === 0);
    const m = document.getElementById('m' + n);
    if (m) m.textContent = suoi.length ? `${{buoni}} ok / ${{cattivi}} no` : '';
  }});
  const ordina = (a, b) => (parseInt(a, 10) - parseInt(b, 10)) || a.localeCompare(b);
  const ok = Object.keys(voti).filter((k) => voti[k] === 'ok').sort(ordina);
  const no = Object.keys(voti).filter((k) => voti[k] === 'no').sort(ordina);
  const manca = Object.keys(voti).filter((k) => voti[k] === 'manca').sort(ordina);
  const visti = new Set(Object.keys(voti).map((k) => parseInt(k, 10))).size;
  document.getElementById('conta').textContent =
    ` \u2014 ${{ok.length}} segmenti sull\u2019ago, ${{no.length}} sbagliati, ` +
    `${{TOTALE - visti}} fotogrammi da guardare`;
  document.getElementById('esito').value =
    `sull'ago: ${{ok.join(', ') || '-'}}\nsbagliate: ${{no.join(', ') || '-'}}` +
    (manca.length ? `\nne manca uno: ${{manca.join(', ')}}` : '');
}}
const TOTALE = document.querySelectorAll('figure[data-n]').length;
disegna();
</script>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"{found}/{total} rilevazioni  ->  {args.output} "
          f"({args.output.stat().st_size/1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
