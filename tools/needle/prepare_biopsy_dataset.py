#!/usr/bin/env python3
"""Il manifest della rete «immagini di biopsia», dalle etichette guardate a mano.

Le etichette arrivano da `build_biopsy_dataset_html.py`: 146 immagini, sei per acquisizione,
segnate a clic. Non si addestra su 146 immagini, e non serve: **nelle 48 cartelle etichettate
non c'e' una sola etichetta mista** -- dove una immagine ha i pallini li hanno tutte, dove non
li ha non li ha nessuna. Quindi l'etichetta si propaga alla cartella, e le 146 decisioni
coprono tutte le immagini che ci stanno dentro.

E' una propagazione, non un dato: vale finche' regge quella uniformita', che e' verificata sui
campioni ma non su ogni immagine. Il manifest la segna (`origine`), cosi' quando un errore
salta fuori si sa da dove viene.

I negativi buoni non sono le ecografie qualsiasi: sono le cartelle «biopsia» **senza** pallini
(schermate dell'interfaccia, ecografie della stessa macchina senza la guida accesa). Quelle
entrano tutte. Agli altri fotogrammi delle stesse acquisizioni si da' una quota, perche' la
rete deve vedere anche il caso comune.

  python3 tools/needle/prepare_biopsy_dataset.py \
      --labels ~/Downloads/etichette_biopsia.json \
      --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
      --output-dir artifacts/93_biopsy_dataset/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
CAMPI = ["image_path", "rel_path", "group", "leaf_dir", "label", "split", "vendor_hint", "origine"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--altri-per-cartella", type=int, default=3,
                    help="quanti fotogrammi prendere dalle cartelle non biopsia")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    radice = args.dataset_root.expanduser().resolve()
    etichette = []
    for f in args.labels:
        etichette.extend(json.loads(f.read_text(encoding="utf-8"))["etichette"])

    # 1. l'etichetta della cartella, e il controllo che sia una sola.
    #
    # Si propaga **solo** dalle cartelle che si chiamano biopsia: li' dentro c'e' quella scena
    # e basta, alle profondita' diverse. Un file «BIOPSY.png» dentro una cartella qualsiasi no:
    # accanto ci sono fotogrammi di tutt'altro, e dargli la stessa etichetta sarebbe inventarla.
    per_cartella: Dict[str, List[bool]] = defaultdict(list)
    singole: Dict[str, bool] = {}
    for voce in etichette:
        rel = Path(voce["file"]).resolve().relative_to(radice)
        if "biops" in rel.parent.name.lower():
            per_cartella[str(rel.parent)].append(bool(voce["pallini"]))
        else:
            singole[str(rel)] = bool(voce["pallini"])
    miste = {k: v for k, v in per_cartella.items() if len(set(v)) > 1}
    if miste:
        print("ATTENZIONE: cartelle con etichette discordi, la propagazione non vale:")
        for k, v in miste.items():
            print(f"   {k}: {v}")
    verdetto = {k: v[0] for k, v in per_cartella.items() if len(set(v)) == 1}
    print(f"file etichettati uno per uno: {len(singole)}")
    print(f"cartelle etichettate: {len(verdetto)} "
          f"({sum(verdetto.values())} con pallini, {len(verdetto)-sum(verdetto.values())} senza)")

    righe: List[Dict] = []

    def aggiungi(path: Path, label: int, origine: str) -> None:
        rel = path.relative_to(radice)
        righe.append({"image_path": str(path), "rel_path": str(rel),
                      "group": str(rel).split("/")[0], "leaf_dir": str(rel.parent),
                      "label": label, "split": "", "vendor_hint": "", "origine": origine})

    # 2. tutte le immagini delle cartelle etichettate prendono il verdetto della cartella
    for cartella, pallini in sorted(verdetto.items()):
        piena = radice / cartella
        if not piena.is_dir():
            continue
        for f in sorted(piena.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                aggiungi(f, int(pallini), "propagata" )

    # 2b. i file etichettati uno per uno, fuori dalle cartelle biopsia: non propagano, ma
    # portano acquisizioni nuove, che e' quello che mancava -- i positivi venivano da nove sole
    # macchine e la validazione ballava di trenta punti fra una epoca e l'altra.
    for rel, pallini in sorted(singole.items()):
        piena = radice / rel
        if piena.is_file():
            aggiungi(piena, int(pallini), "guardata")

    etichettate = {r["rel_path"] for r in righe}
    acquisizioni_note = {r["group"] for r in righe}

    # 3. una quota dalle altre cartelle delle stesse acquisizioni: il caso comune
    rng = random.Random(args.seed)
    for acquisizione in sorted(acquisizioni_note):
        base = radice / acquisizione
        for cartella in sorted(p for p in base.rglob("*") if p.is_dir()):
            rel_cart = str(cartella.relative_to(radice))
            if rel_cart in verdetto or "biops" in cartella.name.lower():
                continue
            dentro = sorted(f for f in cartella.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS
                            and str(f.relative_to(radice)) not in etichettate)
            if not dentro:
                continue
            passo = max(1, len(dentro) // args.altri_per_cartella)
            for f in dentro[::passo][:args.altri_per_cartella]:
                aggiungi(f, 0, "altra cartella")

    # 4. split per acquisizione, tenendo insieme positive e negative in ogni parte
    per_gruppo = defaultdict(list)
    for r in righe:
        per_gruppo[r["group"]].append(r)
    positivi = sorted(g for g, rs in per_gruppo.items() if any(int(r["label"]) for r in rs))
    negativi = sorted(g for g in per_gruppo if g not in positivi)
    rng.shuffle(positivi)
    rng.shuffle(negativi)

    def spezza(elenco: List[str], conta) -> Dict[str, str]:
        """Le quote si contano in immagini, non in acquisizioni.

        Dividendo per numero di acquisizioni la validazione era arrivata a 29 positivi su 425,
        e con ventinove positivi la media delle precisioni salta da 0.82 a 0.59 fra una epoca e
        l'altra per lo spostamento di due o tre fotogrammi: su un numero cosi' non si sceglie il
        modello migliore, ci si tira a sorte.

        Le acquisizioni restano intere -- e' l'unico modo di non far passare lo stesso
        fotogramma da due parti -- e ognuna va dove manca di piu', dalla piu' grande alla piu'
        piccola. Riempire invece le parti una per volta svuota quella che si serve per ultima:
        provato, e il train restava con quarantatre positivi su trecentoquindici.
        """
        quote = {"train": 0.50, "val": 0.25, "test": 0.25}
        totale = sum(conta[g] for g in elenco)
        presi = {k: 0 for k in quote}
        fuori = {}
        for g in sorted(elenco, key=lambda x: -conta[x]):
            parte = max(quote, key=lambda k: quote[k] * totale - presi[k])
            fuori[g] = parte
            presi[parte] += conta[g]
        return fuori

    quante = {g: len(rs) for g, rs in per_gruppo.items()}
    dove = {**spezza(positivi, quante), **spezza(negativi, quante)}
    for r in righe:
        r["split"] = dove[r["group"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest_biopsy.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=CAMPI)
        w.writeheader()
        w.writerows(righe)

    conto = Counter((r["split"], r["label"]) for r in righe)
    gruppi = Counter((dove[g], "positiva" if g in positivi else "negativa") for g in per_gruppo)
    riepilogo = {
        "immagini": len(righe),
        "per_split": {f"{s}/{'pallini' if l else 'niente'}": n for (s, l), n in sorted(conto.items())},
        "acquisizioni": {f"{s}/{t}": n for (s, t), n in sorted(gruppi.items())},
        "cartelle_etichettate": len(verdetto),
        "file_singoli": len(singole),
        "cartelle_discordi": len(miste),
    }
    (args.output_dir / "riepilogo.json").write_text(
        json.dumps(riepilogo, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))
    print(f"scritto {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
