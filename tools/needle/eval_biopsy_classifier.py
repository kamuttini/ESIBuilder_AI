#!/usr/bin/env python3
"""Come e' andata la rete delle biopsie, guardata da dove conta.

Il numero complessivo dice poco: i positivi sono un quinto del test, e una rete che risponde
sempre «no» prende comunque il 65%. Qui si guardano tre cose separate:

1. **Riconosciute e inventate**, non mescolate in una accuratezza.
2. **Per acquisizione**: il test ha 6 acquisizioni positive e 2 negative mai viste. Sbagliare
   qualche fotogramma dentro una acquisizione giusta e sbagliare l'acquisizione intera sono due
   problemi diversi, e solo il secondo rompe il lavoro a valle -- li' basta una immagine buona.
3. **I negativi difficili**: le cartelle che si chiamano «biopsia» ma i pallini non ce li hanno.
   Se li prende per positivi, ha imparato «questa e' una schermata di biopsia» e non «ci sono i
   pallini», che e' un'altra cosa e si rompe sulla prima macchina nuova.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True, help="il CSV dei punteggi del trainer")
    ap.add_argument("--manifest", type=Path, required=True, help="manifest_biopsy.csv, per l'origine")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    origine = {}
    for r in csv.DictReader(args.manifest.open(encoding="utf-8")):
        origine[r["rel_path"]] = r.get("origine", "")

    righe = list(csv.DictReader(args.scores.open(encoding="utf-8")))
    tp = fp = fn = tn = 0
    per_gruppo = defaultdict(lambda: {"n": 0, "pos": 0, "sopra": 0, "veri": 0, "falsi": 0})
    difficili = {"n": 0, "sopra": 0}
    for r in righe:
        y = int(r["label"]); s = float(r["score"]); pred = s >= args.threshold
        tp += y and pred; fp += (not y) and pred; fn += y and (not pred); tn += (not y) and (not pred)
        g = per_gruppo[r["group"]]
        g["n"] += 1; g["pos"] += y
        g["sopra"] += pred
        g["veri"] += (y and pred)
        g["falsi"] += ((not y) and pred)
        # negativo difficile: sta in una cartella che si chiama biopsia ma e' etichettato no
        if not y and "biops" in r["leaf_dir"].lower():
            difficili["n"] += 1
            difficili["sopra"] += pred

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"soglia {args.threshold}")
    print(f"  riconosciute  {tp} su {tp+fn}  (richiamo {100*rec:.1f}%)")
    print(f"  inventate     {fp} su {fp+tn}  (precisione {100*prec:.1f}%)")
    print()
    print("per acquisizione (il test):")
    print(f"  {'esito':10s} {'acquisizione':52s} {'positivi trovati':>18s} {'falsi':>12s}")
    for nome, g in sorted(per_gruppo.items(), key=lambda kv: -kv[1]["pos"]):
        neg = g["n"] - g["pos"]
        if g["pos"]:
            # per calibrare basta un fotogramma buono: l'acquisizione e' presa se ne trova almeno uno
            esito = "ok" if g["veri"] else "PERSA"
            quota = f"{g['veri']:4d} su {g['pos']:4d}"
        else:
            esito = "ok" if not g["falsi"] else "FALSI"
            quota = "       —"
        print(f"  {esito:10s} {nome[:52]:52s} {quota:>18s} "
              f"{g['falsi']:6d} su {neg:5d}")
    print()
    print(f"negativi difficili (cartelle «biopsia» senza pallini): "
          f"{difficili['sopra']} presi per positivi su {difficili['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
