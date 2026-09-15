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
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle import detect as detect_needles  # noqa: E402
from refine_needles import refine  # noqa: E402
from guides_geometry import read_setup  # noqa: E402
from needle_frames import NeedleScorer, all_frames, frames_of_probe  # noqa: E402

# Linear and convex probes calibrate on needles in water, which live in AGHI folders. BIOPSIA
# and GUIDA folders hold the dotted biopsy-line frames, and those belong to biplane probes:
# Camilla excluded three of them by hand, all BK3000 BIOPSIA, from a linear gallery.
CALIB_DIR = re.compile(r"agh", re.IGNORECASE)
CALIB_DIR_BIPLANA = re.compile(r"agh|guid|biops", re.IGNORECASE)
# BGR, as OpenCV wants them. One colour and one letter per needle, because several are
# usually visible and a verdict on "the detection" is meaningless when there are four.
SEGMENT_COLOURS = [(0, 0, 255), (0, 220, 255), (0, 255, 120), (255, 180, 0), (255, 120, 255)]
SEGMENT_LETTERS = "abcde"
SEGMENT_HTML = ["#ffb020"] * 5
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}


def frames_of(acquisition: Path, size: Tuple[int, int], limit: int,
              scorer: Optional["NeedleScorer"] = None,
              rect: Optional[Tuple[int, int, int, int]] = None,
              threshold: float = 0.60, scan_cap: int = 80,
              config_name: str = "") -> List[Path]:
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
        ]
        # one acquisition often covers several probes in sub-folders; a configuration must be
        # measured on its own probe's needles, whose guide angles differ from the others
        if config_name:
            pool = frames_of_probe(pool, acquisition, config_name)
        pool = pool[:scan_cap]
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
                'c\'era un ago</button>'
                '<button class="ghost" onclick="cancellaLinea(%d)">cancella</button></div>'
                '<div class="vote"><button class="ghost wide" id="b%dxx" '
                'onclick="vota(\'%dx\',\'escludi\')">non e\' sonda in acqua</button></div>'
                '<div class="disegnate" id="d%d"></div>'
                % (number, number, number, number, number))
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
    # Quali segmenti sono in realta' lo stesso ago spezzato: lo dice l'occhio, e serve a
    # tarare l'unione automatica invece di indovinarne le tolleranze.
    if n_segments >= 2:
        coppie = []
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                a, b = SEGMENT_LETTERS[i], SEGMENT_LETTERS[j]
                coppie.append(
                    f'<button id="b{number}{a}{b}eq" class="ghost" '
                    f'onclick="vota(\'{number}{a}{b}\',\'stesso\')">{a}={b}</button>')
        rows.append(f'<div class="vote"><span class="seg">=</span>{"".join(coppie)}</div>')
    rows.append(f'<div class="vote"><button class="ghost wide" '
                f'onclick="vota(\'{number}m\',\'manca\')">ne manca uno</button>'
                f'<button class="ghost" onclick="cancellaLinea({number})">cancella</button></div>'
                f'<div class="vote"><button class="ghost wide" id="b{number}xx" '
                f'onclick="vota(\'{number}x\',\'escludi\')">non e\' sonda in acqua</button>'
                f'</div>'
                f'<div class="disegnate" id="d{number}"></div>')
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
    parser.add_argument("--width", type=int, default=900,
                        help="Encoded width. The cards shrink it with CSS; full screen uses it "
                             "whole, which is where a needle is actually judged.")
    parser.add_argument("--min-confidence", type=str, default="alta",
                        help='Pairing confidence to keep ("alta", "media", or "" for all).')
    parser.add_argument("--una-per-acquisizione", action="store_true",
                        help="One frame per acquisition, for annotation. Diversity of scenes is "
                             "what a detector needs: five needles on thirty frames of four "
                             "configurations taught the ranker to memorise them.")
    parser.add_argument("--grezzo", action="store_true",
                        help="Show the raw candidates instead of the refined needles.")
    parser.add_argument("--per-frame", type=int, default=2,
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
    global CALIB_DIR
    if wanted and wanted <= {3, 4, 5}:
        CALIB_DIR = CALIB_DIR_BIPLANA
    rows = [
        r for r in csv.DictReader(args.pairs.open(encoding="utf-8"))
        if (not args.min_confidence or r["confidence"] == args.min_confidence)
        and (not wanted or int(r["probe_type"]) in wanted)
    ]
    if args.una_per_acquisizione:
        # several configurations share one acquisition; keep the one whose declared resolution
        # is best supported, so each acquisition contributes once and with its best evidence
        migliori: Dict[str, dict] = {}
        for row in rows:
            key = row["acquisition"]
            if key not in migliori or float(row["size_match_pct"]) > float(migliori[key]["size_match_pct"]):
                migliori[key] = row
        rows = sorted(migliori.values(), key=lambda r: r["config"])
        print(f"una per acquisizione: {len(rows)} acquisizioni distinte")

    cards: List[str] = []
    frame_paths: List[str] = []
    found = total = 0
    for row in rows[: args.max_configs]:
        setup = read_setup(Path(row["setup_file"]))
        if setup is None or not setup.consistent():
            continue
        size = tuple(setup.video_size)
        rect = (setup.rect_echo.left, setup.rect_echo.top,
                setup.rect_echo.right, setup.rect_echo.bottom)

        for frame in frames_of(Path(row["acquisition"]), size, args.frames_per_config,
                               scorer=scorer, rect=rect, config_name=row["config"]):
            gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if gray is None or (gray.shape[1], gray.shape[0]) != size:
                continue
            total += 1
            raw = detect_needles(gray, rect, top_k=max(4, args.per_frame))
            # what the pipeline actually proposes: merged pieces, at most two, parallel,
            # bright enough and with ridges -- not the raw candidate list
            detections = (refine(raw, gray, max_needles=args.per_frame)
                          if not args.grezzo else raw)
            # The needles are NOT burned into the image: they go into the SVG overlay, so a
            # verdict can restyle the one it refers to. Baked in, "b is wrong" left the frame
            # looking exactly as before and the reviewer could not see what she had marked.
            if detections:
                found += 1
            crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)[rect[1]:rect[3], rect[0]:rect[2]]
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
            frame_paths.append(str(frame))
            crop_w, crop_h = rect[2] - rect[0], rect[3] - rect[1]
            drawn = json.dumps([
                {"x1": (d.p1[0] - rect[0]) / crop_w, "y1": (d.p1[1] - rect[1]) / crop_h,
                 "x2": (d.p2[0] - rect[0]) / crop_w, "y2": (d.p2[1] - rect[1]) / crop_h,
                 "a": round(d.angle_deg, 2)}
                for d in detections
            ])
            cards.append(
                f'<figure id="c{number}" data-n="{number}" '
                f'data-frame="{html.escape(str(frame))}" '
                f'data-config="{html.escape(row["config"])}" '
                f'data-rect="{rect[0]},{rect[1]},{rect[2]},{rect[3]}" '
                f'data-crop="{crop_w},{crop_h}" '
                f"data-det='{html.escape(drawn, quote=True)}'>"
                f'<div class="head"><span class="num">{number}</span>'
                f'<span class="mark" id="m{number}"></span></div>'
                f'<div class="canvas"><img loading="lazy" src="{uri}" alt="">'
                f'<svg class="overlay" id="s{number}"></svg></div>'
                + _voti_html(number, len(detections)) +
                f'<figcaption>{caption}<br>'
                f'<span class="path">{html.escape(row["config"][:52])}</span><br>'
                f'<span class="path">{html.escape(frame.name)}</span></figcaption></figure>'
            )

    gallery_id = hashlib.sha1(
        "|".join(sorted(frame_paths)).encode("utf-8")).hexdigest()[:12]

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
 .head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;
          gap: 8px; }}
 .lente {{ background: none; border: 1px solid #3a3a3a; border-radius: 5px; color: #cfcfcf;
           cursor: pointer; font-size: 14px; padding: 1px 8px; }}
 .lente:hover {{ background: #2c2c2c; }}
 #pieno {{ position: fixed; inset: 0; z-index: 50; background: #0b0b0b; display: flex;
           flex-direction: column; }}
 #pieno[hidden] {{ display: none; }}
 #pienoTesta, #pienoPie {{ padding: 8px 14px; display: flex; gap: 10px; align-items: center;
                           flex-wrap: wrap; background: #161616; }}
 #pienoTesta {{ border-bottom: 1px solid #2a2a2a; }}
 #pienoPie {{ border-top: 1px solid #2a2a2a; }}
 #pienoCanvas {{ flex: 1; position: relative; display: flex; align-items: center;
                 justify-content: center; overflow: hidden; touch-action: none;
                 cursor: crosshair; background: #000; }}
 #pienoCanvas img {{ max-width: 100%; max-height: 100%; object-fit: contain; display: block; }}
 #pienoCanvas svg {{ position: absolute; }}
 #pienoPie button, #pienoTesta button {{ padding: 6px 12px; border-radius: 6px; cursor: pointer;
   border: 1px solid #3a3a3a; background: #262626; color: #e8e8e8; }}
 #pienoNome {{ color: #8f8f8f; font-size: 12px; }}
 .num {{ font-weight: 700; color: #8fb6ff; }}
 .mark {{ font-size: 12px; }}
 .vote {{ display: flex; gap: 6px; margin-top: 6px; align-items: center; }}
 .seg {{ font-weight: 700; width: 14px; display: inline-block; }}
 .vote .wide {{ flex: 1; }}
 .canvas {{ position: relative; line-height: 0; touch-action: none; cursor: crosshair; }}
 .overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
 .overlay line.mia {{ stroke: #35ff9b; stroke-width: 3; stroke-linecap: round; }}
 .overlay line.tmp {{ stroke: #9bffd0; stroke-dasharray: 5 4; stroke-width: 2.5; }}
 .overlay line.det {{ stroke: #ffb020; stroke-width: 2.5; }}
 .overlay line.det.ok {{ stroke: #2ee6a8; stroke-width: 4; }}
 .overlay line.det.no {{ stroke: #ff2d20; stroke-width: 4; }}
 .overlay text {{ font: 700 15px system-ui; paint-order: stroke; stroke: #000; stroke-width: 3px; }}
 .overlay text.segno {{ font: 900 30px system-ui; stroke-width: 5px; }}
 .vote button.attivo {{ background: #1c6b50; border-color: #2ee6a8; color: #eafff6; }}
 .vote button.attivo.rosso {{ background: #7a1c14; border-color: #ff2d20; color: #ffecea; }}
 .disegnate {{ color: #35ff9b; font-size: 12px; margin-top: 4px; }}
 #azzera {{ margin-top: 8px; margin-left: 6px; padding: 6px 12px; border-radius: 6px;
            cursor: pointer; border: 1px solid #5a3a3a; background: #2a1a1a; color: #e8cfcf; }}
 #scarica {{ margin-top: 8px; padding: 6px 12px; border-radius: 6px; cursor: pointer;
             border: 1px solid #3a3a3a; background: #263; color: #eaffea; }}
 .vote button {{ flex: 1; padding: 5px 4px; font-size: 12px; border-radius: 5px; cursor: pointer;
                 border: 1px solid #3a3a3a; background: #262626; color: #e8e8e8; }}
 .vote button:hover {{ background: #333; }}
 .vote .ghost {{ flex: 0 0 62px; color: #999; }}
 figure.ok {{ outline: 2px solid #35c88a; }}
 figure.no {{ outline: 2px solid #ff2d20; }}
 figure.escluso {{ outline: 2px dashed #8a8a8a; opacity: .45; }}
 figure.escluso .canvas::after {{ content: "esclusa \\2014 non e\\2019 sonda in acqua";
   position: absolute; inset: auto 0 0 0; background: rgba(0,0,0,.72); color: #ddd;
   font-size: 12px; padding: 4px 6px; text-align: center; }}
 .vote button.attivo.grigio {{ background: #444; border-color: #888; color: #eee; }}
 #barra {{ position: sticky; top: 0; z-index: 5; background: #191919; border: 1px solid #333;
           border-radius: 8px; padding: 10px 14px; margin-top: 16px; }}
 #json {{ width: 100%; min-height: 120px; margin-top: 8px; background: #101010; color: #ddd;
          border: 1px solid #3a6; border-radius: 6px; padding: 8px;
          font-family: ui-monospace, monospace; font-size: 11px; }}
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
  <b>Come segnalarmele:</b> ogni ago trovato dal programma &egrave;
  <span style="color:#ffb020">arancione</span>; quando lo marchi gli compare sopra
  un segno: <span style="color:#2ee6a8">&#10003; verde</span> se &egrave; giusto,
  <span style="color:#ff2d20">&#10005; rosso</span> se &egrave; sbagliato. Cos&igrave;
  vedi a colpo d'occhio cosa hai segnato, anche quando in un fotogramma uno solo &egrave;
  sbagliato. Ogni ago ha la sua lettera
  (<b>a</b>, <b>b</b>, <b>c</b>&hellip;) e la sua riga di bottoni, quindi un riquadro pu&ograve;
  avere <i>7a</i> giusto e <i>7b</i> sbagliato. Se il programma ha perso un ago che si vede,
  premi <i>ne manca uno</i>. Se il
  fotogramma non &egrave; proprio materiale di calibrazione &mdash; niente sonda in acqua &mdash;
  premi <b>a=b</b> se due segmenti sono in realt&agrave; lo stesso ago spezzato, e
  <b>non &egrave; sonda in acqua</b>: il riquadro si spegne e finisce nell'esportazione
  come da escludere, cos&igrave; correggo la selezione e non solo il rilevatore. Il riepilogo qui sotto si aggiorna da solo: copialo e incollamelo.
  <br><b>Per indicarmi l'ago giusto:</b> trascina sull'immagine da un capo all'altro dell'ago.
  Puoi tracciarne pi&ugrave; di uno; <i>cancella</i> toglie l'ultimo di quel riquadro. Sono queste
  le annotazioni che servono ad addestrare il rilevatore: dicono dov'&egrave; l'ago, non solo che
  la rilevazione era sbagliata.
  <span id="conta" class="path"></span>
  <textarea id="esito" readonly></textarea>
  <button id="scarica">Scarica le annotazioni (JSON)</button>
  <button id="azzera">Azzera tutto</button>
  <span id="esitoScarica" class="path"></span>
  <textarea id="json" readonly style="display:none"></textarea>
</div>
<div id="pieno" hidden>
  <div id="pienoTesta">
    <b id="pienoNum"></b>
    <span id="pienoNome"></span>
    <span style="flex:1"></span>
    <button onclick="chiudiPieno()">chiudi (Esc)</button>
  </div>
  <div id="pienoCanvas"><img id="pienoImg" alt=""><svg id="pienoSvg"></svg></div>
  <div id="pienoPie">
    <button onclick="vaiPieno(-1)">&#8592; precedente</button>
    <button onclick="vaiPieno(1)">successivo &#8594;</button>
    <span id="pienoVoti" style="display:flex;gap:8px;flex-wrap:wrap"></span>
    <span style="flex:1"></span>
    <button onclick="cancellaLinea(pienoN); disegnaLinee();">cancella ultimo ago</button>
  </div>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
// La chiave porta l'identita' della galleria, calcolata sui fotogrammi che contiene. Senza,
// aprendo una galleria nuova il browser rimetteva addosso i voti della precedente: sono
// indicizzati per numero di riquadro, e il riquadro 3 di una galleria non e' il riquadro 3
// dell'altra. Apparivano come correzioni a caso su immagini mai viste.
const GALLERIA = '{gallery_id}';
const CHIAVE = 'rilevatore_ago_voti_' + GALLERIA;
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
    const escluso = voti[n + 'x'] === 'escludi';
    fig.classList.toggle('escluso', escluso);
    const suoi = Object.keys(voti).filter((k) => parseInt(k, 10) === parseInt(n, 10)
                                                 && !k.endsWith('x'));
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
  const esclusi = Object.keys(voti).filter((k) => voti[k] === 'escludi')
    .map((k) => k.slice(0, -1)).sort(ordina);
  const stessi = Object.keys(voti).filter((k) => voti[k] === 'stesso').sort(ordina);
  document.querySelectorAll('.vote button[id^="b"]').forEach((b) => {{
    const key = b.id.slice(1, -2);
    const tipo = b.id.slice(-2);
    if (tipo === 'eq') {{
      const chiave = b.id.slice(1, -2);
      b.classList.toggle('attivo', voti[chiave] === 'stesso');
      return;
    }}
    if (tipo === 'xx') {{
      b.classList.toggle('attivo', voti[key + 'x'] === 'escludi');
      b.classList.toggle('grigio', true);
      return;
    }}
    b.classList.toggle('attivo', voti[key] === tipo);
    b.classList.toggle('rosso', tipo === 'no');
  }});
  disegnaLinee();
  const visti = new Set(Object.keys(voti).map((k) => parseInt(k, 10))).size;
  document.getElementById('conta').textContent =
    ` \u2014 ${{ok.length}} segmenti sull\u2019ago, ${{no.length}} sbagliati, ` +
    `${{TOTALE - visti}} fotogrammi da guardare`;
  aggiornaEsito();
}}
const CHIAVE_LINEE = 'rilevatore_ago_linee_' + GALLERIA;
let linee = {{}};
try {{ linee = JSON.parse(localStorage.getItem(CHIAVE_LINEE) || '{{}}'); }} catch (e) {{ linee = {{}}; }}

function salvaLinee() {{
  try {{ localStorage.setItem(CHIAVE_LINEE, JSON.stringify(linee)); }} catch (e) {{}}
  disegnaLinee();
  aggiornaEsito();
}}

function cancellaLinea(n) {{
  if (linee[n] && linee[n].length) linee[n].pop();
  if (linee[n] && !linee[n].length) delete linee[n];
  salvaLinee();
}}

const LETTERE = 'abcde';
const COLORI = {{det: '#ffb020', ok: '#2ee6a8', no: '#ff2d20'}};

function disegnaLinee() {{
  document.querySelectorAll('figure[data-n]').forEach((fig) => {{
    const n = fig.dataset.n;
    const svg = document.getElementById('s' + n);
    if (!svg) return;
    svg.innerHTML = '';
    // le rilevazioni, ognuna con l'aspetto del proprio verdetto
    let det = [];
    try {{ det = JSON.parse(fig.dataset.det || '[]'); }} catch (e) {{ det = []; }}
    det.forEach((d, i) => {{
      const lettera = LETTERE[i] || '?';
      const stato = voti[n + lettera] || '';
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('class', 'det ' + stato);
      el.setAttribute('x1', d.x1 * 100 + '%'); el.setAttribute('y1', d.y1 * 100 + '%');
      el.setAttribute('x2', d.x2 * 100 + '%'); el.setAttribute('y2', d.y2 * 100 + '%');
      svg.appendChild(el);
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', (d.x2 * 100) + '%');
      t.setAttribute('y', (d.y2 * 100) + '%');
      t.setAttribute('fill', COLORI[stato] || COLORI.det);
      t.textContent = lettera;
      svg.appendChild(t);
      // Il segno sopra l'ago: una croce o una spunta, grande, al centro del segmento.
      // Colore e spessore da soli non bastavano -- si vedeva che qualcosa era cambiato
      // ma non *cosa* si era deciso.
      if (stato) {{
        const segno = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        segno.setAttribute('class', 'segno');
        segno.setAttribute('x', ((d.x1 + d.x2) / 2 * 100) + '%');
        segno.setAttribute('y', ((d.y1 + d.y2) / 2 * 100) + '%');
        segno.setAttribute('text-anchor', 'middle');
        segno.setAttribute('dominant-baseline', 'central');
        segno.setAttribute('fill', COLORI[stato]);
        segno.textContent = stato === 'no' ? '\u2715' : '\u2713';
        svg.appendChild(segno);
      }}
    }});
    for (const l of (linee[n] || [])) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('class', 'mia');
      el.setAttribute('x1', l[0] * 100 + '%'); el.setAttribute('y1', l[1] * 100 + '%');
      el.setAttribute('x2', l[2] * 100 + '%'); el.setAttribute('y2', l[3] * 100 + '%');
      svg.appendChild(el);
    }}
    const d = document.getElementById('d' + n);
    if (d) d.textContent = (linee[n] || []).length ? `${{linee[n].length}} ago/aghi tracciati` : '';
  }});

  // e lo stesso contenuto sullo schermo intero, se e' aperto
  const svgPieno = document.getElementById('pienoSvg');
  if (pienoN && svgPieno && !document.getElementById('pieno').hidden) {{
    const fig = document.querySelector(`figure[data-n="${{pienoN}}"]`);
    svgPieno.innerHTML = '';
    if (fig) {{
      let det = [];
      try {{ det = JSON.parse(fig.dataset.det || '[]'); }} catch (e) {{ det = []; }}
      det.forEach((d, i) => {{
        const lettera = LETTERE[i] || '?';
        const stato = voti[pienoN + lettera] || '';
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        el.setAttribute('class', 'det ' + stato);
        el.setAttribute('x1', d.x1 * 100 + '%'); el.setAttribute('y1', d.y1 * 100 + '%');
        el.setAttribute('x2', d.x2 * 100 + '%'); el.setAttribute('y2', d.y2 * 100 + '%');
        svgPieno.appendChild(el);
        if (stato) {{
          const segno = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          segno.setAttribute('class', 'segno');
          segno.setAttribute('x', ((d.x1 + d.x2) / 2 * 100) + '%');
          segno.setAttribute('y', ((d.y1 + d.y2) / 2 * 100) + '%');
          segno.setAttribute('text-anchor', 'middle');
          segno.setAttribute('dominant-baseline', 'central');
          segno.setAttribute('fill', COLORI[stato]);
          segno.textContent = stato === 'no' ? '\u2715' : '\u2713';
          svgPieno.appendChild(segno);
        }}
      }});
      for (const l of (linee[pienoN] || [])) {{
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        el.setAttribute('class', 'mia');
        el.setAttribute('x1', l[0] * 100 + '%'); el.setAttribute('y1', l[1] * 100 + '%');
        el.setAttribute('x2', l[2] * 100 + '%'); el.setAttribute('y2', l[3] * 100 + '%');
        svgPieno.appendChild(el);
      }}
    }}
  }}
}}

// Trascinamento: due capi dell'ago, in coordinate relative al ritaglio. La stessa funzione
// serve la scheda e lo schermo intero, cosi' un ago tracciato in grande e' identico a uno
// tracciato in piccolo -- e sullo schermo intero si vede davvero quello che si sta segnando.
function abilitaTracciamento(box, numeroDi, svgDi) {{
  let start = null;
  const rel = (ev) => {{
    const r = (svgDi() || box).getBoundingClientRect();
    return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
            Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
  }};
  box.addEventListener('pointerdown', (ev) => {{
    if (ev.target.tagName === 'BUTTON') return;
    start = rel(ev); box.setPointerCapture(ev.pointerId); ev.preventDefault();
  }});
  box.addEventListener('pointermove', (ev) => {{
    if (!start) return;
    const svg = svgDi();
    if (!svg) return;
    const p = rel(ev);
    let tmp = svg.querySelector('line.tmp');
    if (!tmp) {{
      tmp = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tmp.setAttribute('class', 'tmp'); svg.appendChild(tmp);
    }}
    tmp.setAttribute('x1', start[0] * 100 + '%'); tmp.setAttribute('y1', start[1] * 100 + '%');
    tmp.setAttribute('x2', p[0] * 100 + '%'); tmp.setAttribute('y2', p[1] * 100 + '%');
  }});
  box.addEventListener('pointerup', (ev) => {{
    if (!start) return;
    const p = rel(ev);
    const n = numeroDi();
    if (Math.hypot(p[0] - start[0], p[1] - start[1]) > 0.02 && n) {{
      if (!linee[n]) linee[n] = [];
      linee[n].push([start[0], start[1], p[0], p[1]]);
      salvaLinee();
    }} else {{
      disegnaLinee();
    }}
    start = null;
  }});
}}

document.querySelectorAll('figure[data-n] .canvas').forEach((box) => {{
  const n = box.closest('figure').dataset.n;
  abilitaTracciamento(box, () => n, () => document.getElementById('s' + n));
}});

// ---------------------------------------------------------------- schermo intero
let pienoN = null;

function apriPieno(n) {{
  const fig = document.querySelector(`figure[data-n="${{n}}"]`);
  if (!fig) return;
  pienoN = String(n);
  document.getElementById('pienoImg').src = fig.querySelector('img').src;
  document.getElementById('pienoNum').textContent = '#' + n;
  document.getElementById('pienoNome').textContent =
    (fig.dataset.config || '') + '  ·  ' + (fig.dataset.frame || '').split('/').pop();
  const voti_ = document.getElementById('pienoVoti');
  voti_.innerHTML = '';
  const det = JSON.parse(fig.dataset.det || '[]');
  det.forEach((d, i) => {{
    const lettera = LETTERE[i] || '?';
    const chiave = n + lettera;
    const gruppo = document.createElement('span');
    gruppo.style.cssText = 'display:flex;gap:4px;align-items:center';
    gruppo.innerHTML = `<b style="color:${{COLORI[voti[chiave]] || COLORI.det}}">${{lettera}}</b>`;
    for (const [etichetta, valore] of [['ago', 'ok'], ['no', 'no'], ['×', '']]) {{
      const b = document.createElement('button');
      b.textContent = etichetta;
      if (voti[chiave] === valore && valore) b.style.borderColor = COLORI[valore];
      b.onclick = () => {{ vota(chiave, valore); apriPieno(n); }};
      gruppo.appendChild(b);
    }}
    voti_.appendChild(gruppo);
  }});
  const escl = document.createElement('button');
  escl.textContent = voti[n + 'x'] === 'escludi' ? 'esclusa ✓' : 'non e\u2019 sonda in acqua';
  escl.onclick = () => {{ vota(n + 'x', voti[n + 'x'] === 'escludi' ? '' : 'escludi'); apriPieno(n); }};
  voti_.appendChild(escl);

  document.getElementById('pieno').hidden = false;
  setTimeout(adattaSvgPieno, 60);
}}

function adattaSvgPieno() {{
  // l'SVG deve stare esattamente sopra l'immagine, che e' centrata e ridimensionata
  const img = document.getElementById('pienoImg');
  const svg = document.getElementById('pienoSvg');
  const box = document.getElementById('pienoCanvas').getBoundingClientRect();
  const r = img.getBoundingClientRect();
  svg.style.left = (r.left - box.left) + 'px';
  svg.style.top = (r.top - box.top) + 'px';
  svg.style.width = r.width + 'px';
  svg.style.height = r.height + 'px';
  disegnaLinee();
}}

function chiudiPieno() {{ document.getElementById('pieno').hidden = true; pienoN = null; }}

function vaiPieno(passo) {{
  const numeri = [...document.querySelectorAll('figure[data-n]')].map((f) => f.dataset.n);
  const i = numeri.indexOf(String(pienoN));
  if (i < 0) return;
  const j = Math.min(numeri.length - 1, Math.max(0, i + passo));
  apriPieno(numeri[j]);
}}

document.addEventListener('keydown', (ev) => {{
  if (document.getElementById('pieno').hidden) return;
  if (ev.key === 'Escape') chiudiPieno();
  if (ev.key === 'ArrowLeft') vaiPieno(-1);
  if (ev.key === 'ArrowRight') vaiPieno(1);
}});
window.addEventListener('resize', () => {{
  if (!document.getElementById('pieno').hidden) adattaSvgPieno();
}});
document.getElementById('pienoImg').addEventListener('load', adattaSvgPieno);
abilitaTracciamento(document.getElementById('pienoCanvas'), () => pienoN,
                    () => document.getElementById('pienoSvg'));

function annotazioni() {{
  const out = [];
  document.querySelectorAll('figure[data-n]').forEach((fig) => {{
    const n = fig.dataset.n;
    if (!linee[n] || !linee[n].length) return;
    const [l, t] = fig.dataset.rect.split(',').map(Number);
    const [cw, ch] = fig.dataset.crop.split(',').map(Number);
    if (voti[n + 'x'] === 'escludi') return;   // esclusa: non deve finire fra le annotazioni
    out.push({{
      n: Number(n), frame: fig.dataset.frame, config: fig.dataset.config,
      rect: fig.dataset.rect.split(',').map(Number),
      // capi dell'ago in pixel del fotogramma intero, come li vuole la geometria
      lines: linee[n].map((v) => [
        Math.round(l + v[0] * cw), Math.round(t + v[1] * ch),
        Math.round(l + v[2] * cw), Math.round(t + v[3] * ch),
      ]),
    }});
  }});
  return out;
}}

function aggiornaEsito() {{
  const ordina = (a, b) => (parseInt(a, 10) - parseInt(b, 10)) || a.localeCompare(b);
  const ok = Object.keys(voti).filter((k) => voti[k] === 'ok').sort(ordina);
  const no = Object.keys(voti).filter((k) => voti[k] === 'no').sort(ordina);
  const manca = Object.keys(voti).filter((k) => voti[k] === 'manca').sort(ordina);
  const esclusi = Object.keys(voti).filter((k) => voti[k] === 'escludi')
    .map((k) => k.slice(0, -1)).sort(ordina);
  const stessi = Object.keys(voti).filter((k) => voti[k] === 'stesso').sort(ordina);
  const ann = annotazioni();
  const quanti = ann.reduce((s, a) => s + a.lines.length, 0);
  document.getElementById('esito').value =
    `sull'ago: ${{ok.join(', ') || '-'}}\nsbagliate: ${{no.join(', ') || '-'}}` +
    (manca.length ? `\nne manca uno: ${{manca.join(', ')}}` : '') +
    (esclusi.length ? `\nda escludere (non sonda in acqua): ${{esclusi.join(', ')}}` : '') +
    (stessi.length ? `\nstesso ago: ${{stessi.join(', ')}}` : '') +
    (quanti ? `\naghi tracciati a mano: ${{quanti}} su ${{ann.length}} fotogrammi (usa il bottone per il file)` : '');
}}

// Scaricare da una pagina aperta con file:// non e' affidabile: certi browser ignorano
// l'attributo download e navigano sul blob, lasciando la pagina bianca e nessun file. Quindi
// si prova a scaricare, ma il testo viene comunque messo in chiaro e negli appunti: le
// annotazioni non devono poter sparire per un dettaglio del browser.
function esclusi() {{
  const out = [];
  document.querySelectorAll('figure[data-n]').forEach((fig) => {{
    const n = fig.dataset.n;
    if (voti[n + 'x'] === 'escludi') {{
      out.push({{n: Number(n), frame: fig.dataset.frame, config: fig.dataset.config,
                motivo: "non e' sonda in acqua"}});
    }}
  }});
  return out;
}}

function stessoAgo() {{
  return Object.keys(voti).filter((k) => voti[k] === 'stesso').map((k) => {{
    const m = k.match(/^(\\d+)([a-e])([a-e])$/);
    return m ? {{n: Number(m[1]), segmenti: [m[2], m[3]]}} : null;
  }}).filter(Boolean);
}}

function testoAnnotazioni() {{
  // le escluse viaggiano insieme: dicono quali fotogrammi la selezione non avrebbe
  // dovuto proporre, che serve a correggere la selezione, non solo il rilevatore
  return JSON.stringify({{annotazioni: annotazioni(), escluse: esclusi(),
                          stesso_ago: stessoAgo()}}, null, 1);
}}

document.getElementById('azzera').addEventListener('click', () => {{
  if (!confirm('Cancello tutti i voti e gli aghi tracciati di questa galleria?')) return;
  voti = {{}}; linee = {{}};
  try {{ localStorage.removeItem(CHIAVE); localStorage.removeItem(CHIAVE_LINEE); }} catch (e) {{}}
  disegna(); salvaLinee();
}});

document.getElementById('scarica').addEventListener('click', async () => {{
  const testo = testoAnnotazioni();
  const area = document.getElementById('json');
  area.style.display = 'block';
  area.value = testo;
  area.select();
  let copiato = false;
  try {{
    await navigator.clipboard.writeText(testo);
    copiato = true;
  }} catch (e) {{
    try {{ copiato = document.execCommand('copy'); }} catch (e2) {{ copiato = false; }}
  }}
  try {{
    const blob = new Blob([testo], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'aghi_annotati.json';
    a.style.display = 'none';
    document.body.appendChild(a);      // certi browser non scaricano un anchor fuori dal DOM
    a.click();
    setTimeout(() => {{ document.body.removeChild(a); URL.revokeObjectURL(url); }}, 4000);
  }} catch (e) {{ /* resta il testo qui sotto */ }}
  document.getElementById('esitoScarica').textContent = copiato
    ? 'copiato negli appunti — e il testo e\u2019 qui sotto, se il file non e\u2019 arrivato'
    : 'se il file non e\u2019 arrivato, copia il testo qui sotto';
}});

const TOTALE = document.querySelectorAll('figure[data-n]').length;
disegna();
disegnaLinee();
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
