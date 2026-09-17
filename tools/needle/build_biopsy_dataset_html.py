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
    ap.add_argument("--width", type=int, default=420)
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
        cursor:pointer}
 figure img{width:100%;display:block;border-radius:4px}
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
<script>
const CHIAVE = 'biopsia___CHIAVE__';
const stato = JSON.parse(localStorage.getItem(CHIAVE) || '{}');
const conto = document.getElementById('conto');
function aggiorna(){
  let si=0,no=0;
  for(const v of Object.values(stato)){ if(v==='si')si++; else if(v==='no')no++; }
  conto.textContent = `${si} sì · ${no} no · ${__TOTALE__-si-no} da vedere`;
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
aggiorna();
function esporta(){
  const righe = [];
  for(const fig of document.querySelectorAll('figure')){
    const v = stato[fig.dataset.n];
    if(v) righe.push({file: fig.dataset.file, pallini: v === 'si'});
  }
  const testo = JSON.stringify({chiave: CHIAVE, etichette: righe}, null, 1);
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
