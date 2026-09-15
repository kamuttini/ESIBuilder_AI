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
import json
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
SEGMENT_HTML = ["#ffb020"] * 5
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
                'c\'era un ago</button>'
                '<button class="ghost" onclick="cancellaLinea(%d)">cancella</button></div>'
                '<div class="disegnate" id="d%d"></div>' % (number, number, number))
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
                f'onclick="vota(\'{number}m\',\'manca\')">ne manca uno</button>'
                f'<button class="ghost" onclick="cancellaLinea({number})">cancella</button></div>'
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
 .canvas {{ position: relative; line-height: 0; touch-action: none; cursor: crosshair; }}
 .overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
 .overlay line.mia {{ stroke: #35ff9b; stroke-width: 3; stroke-linecap: round; }}
 .overlay line.tmp {{ stroke: #9bffd0; stroke-dasharray: 5 4; stroke-width: 2.5; }}
 .overlay line.det {{ stroke: #ffb020; stroke-width: 2.5; }}
 .overlay line.det.ok {{ stroke: #35c8ff; stroke-width: 4; }}
 .overlay line.det.no {{ stroke: #f2564d; stroke-width: 2; stroke-dasharray: 6 5; opacity: .7; }}
 .overlay text {{ font: 700 15px system-ui; paint-order: stroke; stroke: #000; stroke-width: 3px; }}
 .vote button.attivo {{ background: #35507a; border-color: #4d7ab8; color: #fff; }}
 .vote button.attivo.rosso {{ background: #6b2a26; border-color: #a04a44; }}
 .disegnate {{ color: #35ff9b; font-size: 12px; margin-top: 4px; }}
 #scarica {{ margin-top: 8px; padding: 6px 12px; border-radius: 6px; cursor: pointer;
             border: 1px solid #3a3a3a; background: #263; color: #eaffea; }}
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
  <b>Come segnalarmele:</b> ogni ago trovato dal programma &egrave;
  <span style="color:#ffb020">arancione</span>; quando lo marchi diventa
  <span style="color:#35c8ff">azzurro e spesso</span> se &egrave; giusto,
  <span style="color:#f2564d">rosso tratteggiato e smorto</span> se &egrave; sbagliato. Cos&igrave;
  vedi a colpo d'occhio cosa hai segnato, anche quando in un fotogramma uno solo &egrave;
  sbagliato. Ogni ago ha la sua lettera
  (<b>a</b>, <b>b</b>, <b>c</b>&hellip;) e la sua riga di bottoni, quindi un riquadro pu&ograve;
  avere <i>7a</i> giusto e <i>7b</i> sbagliato. Se il programma ha perso un ago che si vede,
  premi <i>ne manca uno</i>. Il riepilogo qui sotto si aggiorna da solo: copialo e incollamelo.
  <br><b>Per indicarmi l'ago giusto:</b> trascina sull'immagine da un capo all'altro dell'ago.
  Puoi tracciarne pi&ugrave; di uno; <i>cancella</i> toglie l'ultimo di quel riquadro. Sono queste
  le annotazioni che servono ad addestrare il rilevatore: dicono dov'&egrave; l'ago, non solo che
  la rilevazione era sbagliata.
  <span id="conta" class="path"></span>
  <textarea id="esito" readonly></textarea>
  <button id="scarica">Scarica le annotazioni (JSON)</button>
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
  document.querySelectorAll('.vote button[id^="b"]').forEach((b) => {{
    const key = b.id.slice(1, -2);
    const tipo = b.id.slice(-2);
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
const CHIAVE_LINEE = 'rilevatore_ago_linee';
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
const COLORI = {{det: '#ffb020', ok: '#35c8ff', no: '#f2564d'}};

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
      t.setAttribute('opacity', stato === 'no' ? '.7' : '1');
      t.textContent = lettera;
      svg.appendChild(t);
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
}}

// trascinamento sull'immagine: due capi dell'ago, in coordinate relative al ritaglio
document.querySelectorAll('figure[data-n] .canvas').forEach((box) => {{
  const fig = box.closest('figure');
  const n = fig.dataset.n;
  const svg = box.querySelector('svg');
  let start = null;
  const rel = (ev) => {{
    const r = box.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
            Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
  }};
  box.addEventListener('pointerdown', (ev) => {{
    start = rel(ev); box.setPointerCapture(ev.pointerId); ev.preventDefault();
  }});
  box.addEventListener('pointermove', (ev) => {{
    if (!start) return;
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
    const dx = (p[0] - start[0]), dy = (p[1] - start[1]);
    // un clic per sbaglio non e' un ago
    if (Math.hypot(dx, dy) > 0.05) {{
      if (!linee[n]) linee[n] = [];
      linee[n].push([start[0], start[1], p[0], p[1]]);
      salvaLinee();
    }} else {{
      disegnaLinee();
    }}
    start = null;
  }});
}});

function annotazioni() {{
  const out = [];
  document.querySelectorAll('figure[data-n]').forEach((fig) => {{
    const n = fig.dataset.n;
    if (!linee[n] || !linee[n].length) return;
    const [l, t] = fig.dataset.rect.split(',').map(Number);
    const [cw, ch] = fig.dataset.crop.split(',').map(Number);
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
  const ann = annotazioni();
  const quanti = ann.reduce((s, a) => s + a.lines.length, 0);
  document.getElementById('esito').value =
    `sull'ago: ${{ok.join(', ') || '-'}}\nsbagliate: ${{no.join(', ') || '-'}}` +
    (manca.length ? `\nne manca uno: ${{manca.join(', ')}}` : '') +
    (quanti ? `\naghi tracciati a mano: ${{quanti}} su ${{ann.length}} fotogrammi (usa il bottone per il file)` : '');
}}

document.getElementById('scarica').addEventListener('click', () => {{
  const blob = new Blob([JSON.stringify({{annotazioni: annotazioni()}}, null, 1)],
                        {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'aghi_annotati.json';
  a.click();
  URL.revokeObjectURL(a.href);
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
