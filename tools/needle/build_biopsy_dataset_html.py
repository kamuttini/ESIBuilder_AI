#!/usr/bin/env python3
"""Galleria per etichettare le immagini di biopsia: ci sono i pallini o no?

Il nome della cartella non basta come etichetta. Nelle 56 cartelle «biopsia» dell'archivio ci
sono 831 immagini, e guardandone una per acquisizione se ne vedono schermate di Windows,
ecografie senza pallini e pannelli dell'interfaccia: addestrare su quel nome insegnerebbe alla
rete che cosa c'era scritto sulla cartella, non che cosa si vede.

Nemmeno un rilevatore scritto a mano basta: i pallini bianchi non si distinguono dal rumore del
tessuto con una regola geometrica (misurato -- il passo fra pallini vicini ha la stessa
dispersione, 0.6-0.8, su una matrice vera e su una ecografia qualsiasi).

Quindi si guarda. Qui le immagini si segnano a clic, la scelta resta nel browser e si esporta
in JSON: sono le etichette con cui addestrare.

  python3 tools/needle/build_biopsy_dataset_html.py \
      --root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
      --output artifacts/92_guides/etichette_biopsia.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2

SUFFISSI = {".png", ".jpg", ".jpeg"}


def trova(root: Path, per_acquisizione: int, solo_biopsia: bool) -> List[Path]:
    """I candidati: le immagini delle cartelle che si chiamano «biopsia», a campione."""
    per_acq: Dict[str, List[Path]] = {}
    for cartella in sorted(root.rglob("*")):
        if not cartella.is_dir():
            continue
        if solo_biopsia and "biops" not in cartella.name.lower():
            continue
        acquisizione = str(cartella.relative_to(root)).split("/")[0]
        dentro = sorted(f for f in cartella.iterdir() if f.suffix.lower() in SUFFISSI)
        if dentro:
            per_acq.setdefault(acquisizione, []).extend(dentro)
    fuori: List[Path] = []
    for acquisizione, files in sorted(per_acq.items()):
        passo = max(1, len(files) // per_acquisizione)
        fuori.extend(files[::passo][:per_acquisizione])
    return fuori


def miniatura(path: Path, larghezza: int) -> str:
    immagine = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if immagine is None:
        return ""
    h, w = immagine.shape[:2]
    scala = larghezza / max(1, w)
    piccola = cv2.resize(immagine, (larghezza, max(1, int(h * scala))))
    ok, buf = cv2.imencode(".jpg", piccola, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-acquisizione", type=int, default=6)
    ap.add_argument("--width", type=int, default=900,
                    help="una copia sola, grande: la griglia la rimpicciolisce con il CSS e "
                         "il pieno schermo la usa com'e'")
    ap.add_argument("--tutte-le-cartelle", action="store_true",
                    help="non solo quelle che si chiamano biopsia: serve per i negativi")
    args = ap.parse_args()

    files = trova(args.root, args.per_acquisizione, not args.tutte_le_cartelle)
    print(f"candidati: {len(files)}", flush=True)

    voci = []
    for i, f in enumerate(files, 1):
        dato = miniatura(f, args.width)
        if not dato:
            continue
        voci.append({"n": len(voci) + 1, "file": str(f),
                     "acq": str(f.relative_to(args.root)).split("/")[0],
                     "cartella": f.parent.name, "img": dato})
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    chiave = hashlib.sha1("|".join(v["file"] for v in voci).encode()).hexdigest()[:10]
    corpo = []
    for v in voci:
        corpo.append(
            f'<figure class="v" data-n="{v["n"]}" data-file="{html.escape(v["file"])}">'
            f'<button class="lente" title="a tutto schermo">⤢</button>'
            f'<img src="data:image/jpeg;base64,{v["img"]}" loading="lazy">'
            f'<figcaption>{v["n"]} · {html.escape(v["acq"][:38])}'
            f'<br><span class="c">{html.escape(v["cartella"][:30])}</span></figcaption>'
            f'</figure>'
        )

    pagina = TEMPLATE.replace("__CHIAVE__", chiave)
    pagina = pagina.replace("__TOTALE__", str(len(voci)))
    pagina = pagina.replace("__CORPO__", "\n".join(corpo))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pagina, encoding="utf-8")
    print(f"scritto {args.output} — {len(voci)} immagini")
    return 0


TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>Immagini di biopsia — etichettatura</title>
<style>
 body{background:#0d1117;color:#c9d1d9;font:14px system-ui;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px}
 .barra{position:sticky;top:0;background:#0d1117;padding:8px 0 12px;z-index:5;
        border-bottom:1px solid #30363d;margin-bottom:12px}
 button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;
        padding:6px 10px;cursor:pointer;margin-right:6px}
 button:hover{border-color:#58a6ff}
 .griglia{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
 figure{margin:0;background:#161b22;border:2px solid #30363d;border-radius:8px;padding:6px;
        cursor:pointer;position:relative}
 figure img{width:100%;display:block;border-radius:4px}
 .lente{position:absolute;top:10px;right:10px;opacity:0;font-size:16px;padding:2px 7px}
 figure:hover .lente{opacity:1}
 #pieno{position:fixed;inset:0;background:rgba(2,6,12,.96);z-index:20;display:none;
        flex-direction:column}
 #pieno.on{display:flex}
 #pienoTesta{display:flex;gap:8px;align-items:center;padding:10px 14px;
             border-bottom:1px solid #30363d;background:#161b22;flex-wrap:wrap}
 #pienoScena{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:10px}
 #pienoScena img{max-width:100%;max-height:100%;object-fit:contain}
 button.si{border-color:#3fb950;color:#3fb950}
 button.no{border-color:#f85149;color:#f85149}
 figcaption{font-size:11px;color:#8b949e;margin-top:4px}
 .c{color:#6e7681}
 figure.si{border-color:#3fb950}
 figure.no{border-color:#f85149;opacity:.55}
 textarea{width:100%;height:120px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;
          border-radius:6px;margin-top:8px;font:12px monospace}
</style>
<div class="barra">
  <h1>Ci sono i pallini della biopsia dentro l'immagine ecografica?</h1>
  <div>clic = sì · secondo clic = no · terzo clic = non deciso &nbsp;
    <b id="conto"></b></div>
  <div style="margin-top:8px">
    <button onclick="esporta()">Esporta le etichette</button>
    <button onclick="if(confirm('Azzerare tutto?')){localStorage.removeItem(CHIAVE);location.reload()}">Azzera</button>
  </div>
  <textarea id="uscita" placeholder="qui compare il JSON da copiare"></textarea>
</div>
<div class="griglia">__CORPO__</div>
<div id="pieno">
  <div id="pienoTesta">
    <button onclick="vai(-1)">‹</button>
    <button onclick="vai(1)">›</button>
    <span id="pienoInfo"></span>
    <button class="si" onclick="segnaPieno('si')">sì (S)</button>
    <button class="no" onclick="segnaPieno('no')">no (N)</button>
    <button onclick="segnaPieno(null)">non deciso (D)</button>
    <button onclick="chiudiPieno()">chiudi (Esc)</button>
  </div>
  <div id="pienoScena"><img id="pienoImg" alt=""></div>
</div>
<script>
const CHIAVE = 'biopsia___CHIAVE__';
const stato = JSON.parse(localStorage.getItem(CHIAVE) || '{}');
const conto = document.getElementById('conto');
function aggiorna(){
  let si=0,no=0;
  for(const v of Object.values(stato)){ if(v==='si')si++; else if(v==='no')no++; }
  conto.textContent = `${si} sì · ${no} no · ${__TOTALE__-si-no} da vedere`;
  scrivi();
}
for(const fig of document.querySelectorAll('figure')){
  const n = fig.dataset.n;
  if(stato[n]) fig.classList.add(stato[n]);
  fig.addEventListener('click', () => {
    const ora = stato[n];
    fig.classList.remove('si','no');
    if(ora === 'si'){ stato[n]='no'; fig.classList.add('no'); }
    else if(ora === 'no'){ delete stato[n]; }
    else { stato[n]='si'; fig.classList.add('si'); }
    localStorage.setItem(CHIAVE, JSON.stringify(stato));
    aggiorna();
  });
}
/* Il JSON si riscrive a ogni clic.

   Prima lo compilava solo il pulsante «Esporta», quindi dopo averlo premuto una volta il
   riquadro restava fermo su una fotografia vecchia mentre si continuava a etichettare: sembrava
   che le scelte non entrassero. Adesso quello che si vede e' sempre quello che c'e'. */
function testoEtichette(){
  const righe = [];
  for(const fig of document.querySelectorAll('figure')){
    const v = stato[fig.dataset.n];
    if(v) righe.push({file: fig.dataset.file, pallini: v === 'si'});
  }
  return JSON.stringify({chiave: CHIAVE, etichette: righe}, null, 1);
}
function scrivi(){ document.getElementById('uscita').value = testoEtichette(); }

/* A tutto schermo, con le stesse scelte sotto mano: i pallini sono piccoli e in un riquadro
   da trecento pixel non si vedono, ed e' proprio quello che bisogna guardare per rispondere. */
const figure = [...document.querySelectorAll('figure')];
let corrente = -1;
function apriPieno(i){
  corrente = Math.max(0, Math.min(i, figure.length - 1));
  const fig = figure[corrente];
  document.getElementById('pienoImg').src = fig.querySelector('img').src;
  const n = fig.dataset.n;
  const eti = stato[n] === 'si' ? 'sì' : stato[n] === 'no' ? 'no' : 'non deciso';
  document.getElementById('pienoInfo').textContent =
    `${corrente + 1} di ${figure.length} · ${fig.querySelector('figcaption').textContent.trim()} · ${eti}`;
  document.getElementById('pieno').classList.add('on');
}
function chiudiPieno(){ document.getElementById('pieno').classList.remove('on'); corrente = -1; }
function vai(d){ if(corrente >= 0) apriPieno((corrente + d + figure.length) % figure.length); }
function segnaPieno(v){
  if(corrente < 0) return;
  const fig = figure[corrente];
  const n = fig.dataset.n;
  fig.classList.remove('si','no');
  if(v){ stato[n] = v; fig.classList.add(v); } else { delete stato[n]; }
  localStorage.setItem(CHIAVE, JSON.stringify(stato));
  aggiorna(); apriPieno(corrente);
}
for(const [i, fig] of figure.entries()){
  fig.querySelector('.lente').addEventListener('click', (e) => { e.stopPropagation(); apriPieno(i); });
}
window.addEventListener('keydown', (e) => {
  if(!document.getElementById('pieno').classList.contains('on')) return;
  if(e.key === 'Escape'){ e.preventDefault(); chiudiPieno(); }
  else if(e.key === 'ArrowLeft'){ e.preventDefault(); vai(-1); }
  else if(e.key === 'ArrowRight'){ e.preventDefault(); vai(1); }
  else if(e.key === 's' || e.key === 'S'){ e.preventDefault(); segnaPieno('si'); vai(1); }
  else if(e.key === 'n' || e.key === 'N'){ e.preventDefault(); segnaPieno('no'); vai(1); }
  else if(e.key === 'd' || e.key === 'D'){ e.preventDefault(); segnaPieno(null); }
});

aggiorna();
function esporta(){
  const testo = testoEtichette();
  document.getElementById('uscita').value = testo;
  if(navigator.clipboard) navigator.clipboard.writeText(testo).catch(()=>{});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([testo], {type:'application/json'}));
  a.download = 'etichette_biopsia.json';
  document.body.appendChild(a); a.click();
  setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 4000);
}
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
