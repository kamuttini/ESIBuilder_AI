#!/usr/bin/env python3
"""Il reticolo di pallini della calibrazione biplana: quanto e' fitto e dove sta.

Nella sessione a griglia l'operatore segna il rettangolo della **griglia intera**, che nelle
immagini e' quasi tutta fuori dallo schermo: se ne vedono poche righe e poche colonne, e il
resto e' estrapolato. Sembra difficile da automatizzare, e invece c'e' un vincolo che regge
l'intera cosa.

Sui 351 casi dell'archivio, la larghezza del rettangolo diviso il numero di colonne del kit da'
la spaziatura dichiarata dal kit stesso, con 0.009 mm di errore mediano. Cioe': **la dimensione
del rettangolo non e' una misura, e' una conseguenza** -- bastano il passo dei pallini in pixel
e la matrice del kit, e la larghezza esce da se'. Dell'immagine serve solo il passo e un punto
d'aggancio.

Questo modulo fa quella parte: trova i pallini, ne ricava il passo di riga e di colonna, e dice
dove cade il reticolo. Chi lo chiama ci mette la matrice del kit e ne ricava il rettangolo.

  python3 tools/needle/detect_dot_grid.py --image <...>/image_CalGrid_depth_0.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Reticolo:
    punti: List[Tuple[float, float]]
    passo_x: float
    passo_y: float
    righe: int                    # quante righe di pallini si vedono
    colonne: int
    x0: float                     # la colonna piu' a sinistra vista, in pixel
    y0: float
    regolarita_x: float           # dispersione dei passi sul totale; bassa = reticolo vero
    regolarita_y: float


def _pallini(bgr: np.ndarray, rect: Optional[Sequence[int]], min_contrast: int,
             max_area: int, sfondo_max: int = 25) -> np.ndarray:
    grigio = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if rect:
        l, t, r, b = (int(v) for v in rect)
        maschera = np.zeros_like(grigio)
        maschera[max(0, t):b, max(0, l):r] = 255
        grigio = cv2.bitwise_and(grigio, maschera)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(grigio, cv2.MORPH_TOPHAT, elemento)
    binaria = (tophat >= min_contrast).astype(np.uint8) * 255
    n, _l, stats, cen = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    punti = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (1 <= area <= max_area):
            continue
        if max(w, h) > 3 * max(1, min(w, h)):     # i pallini sono tondi, non tratti
            continue
        # disegnati sul nero, non dentro il tessuto: e' quello che separa i pallini della
        # griglia dal rumore dell'arco della sonda, che e' altrettanto piccolo e chiaro
        x, y = int(round(cen[i][0])), int(round(cen[i][1]))
        finestra = grigio[max(0, y - 12):y + 13, max(0, x - 12):x + 13]
        if finestra.size and float(np.median(finestra)) > sfondo_max:
            continue
        punti.append(cen[i])
    return np.array(punti, dtype=np.float32) if punti else np.zeros((0, 2), np.float32)


def passo_dominante(valori: np.ndarray, minimo: float, massimo: float,
                    tolleranza: float) -> Tuple[float, int]:
    """Il passo del reticolo: quello che riempie i suoi posti.

    Contare le coppie non basta e nemmeno guardarne la coerenza: un passo vero T e ogni suo
    sottomultiplo spiegano le stesse distanze -- se i punti stanno sui multipli di T ci stanno
    anche su quelli di T/2 -- e ogni suo multiplo spiega un sottoinsieme perfetto. Contando le
    coppie si scivola sui sottomultipli (misurato: 41 pixel dove il vero era 120), prendendo
    il piu' grande si sfora sui multipli (23% delle celle entro il 5%).

    Quello che distingue T da T/2 e da 2T e' **quanti posti sono occupati**: sul passo giusto
    le colonne si susseguono senza buchi, su T/2 ne resta vuota una su due, su 2T se ne perde
    meta'. Si sceglie quindi il passo che massimizza il numero di posti occupati *consecutivi*,
    a parita' di quanto bene i punti ci cadono sopra.
    """
    if valori.size < 3:
        return 0.0, 0
    v = np.sort(np.unique(np.round(valori, 1)))
    if v.size < 3:
        return 0.0, 0
    d = np.abs(v[:, None] - v[None, :])
    d = d[np.triu_indices_from(d, k=1)]
    d = d[(d >= minimo) & (d <= massimo)]
    if d.size == 0:
        return 0.0, 0

    migliore, punteggio_migliore, quanti = 0.0, -1.0, 0
    for candidato in np.unique(np.round(d, 0)):
        if candidato < minimo:
            continue
        posti = np.round((v - v.min()) / candidato)
        residui = np.abs((v - v.min()) - posti * candidato)
        dentro = residui <= tolleranza
        if dentro.sum() < 3:
            continue
        occupati = np.unique(posti[dentro])
        if occupati.size < 3:
            continue
        # i posti devono essere consecutivi: un reticolo non salta meta' delle sue colonne
        estensione = occupati.max() - occupati.min() + 1
        riempimento = occupati.size / max(1.0, estensione)
        if riempimento < 0.6:
            continue
        punteggio = occupati.size * riempimento
        if punteggio > punteggio_migliore:
            migliore, punteggio_migliore, quanti = float(candidato), punteggio, int(occupati.size)
    return migliore, quanti


def _righe(valori: np.ndarray, tolleranza: float) -> List[float]:
    """I valori si addensano su poche coordinate: qui si trovano quelle, non i singoli punti."""
    if valori.size == 0:
        return []
    ordinati = np.sort(valori)
    gruppi, corrente = [], [ordinati[0]]
    for v in ordinati[1:]:
        if v - corrente[-1] <= tolleranza:
            corrente.append(v)
        else:
            gruppi.append(corrente)
            corrente = [v]
    gruppi.append(corrente)
    return [float(np.median(g)) for g in gruppi if len(g) >= 2]


def detect(bgr: np.ndarray, rect: Optional[Sequence[int]] = None, min_contrast: int = 18,
           max_area: int = 60, min_righe: int = 2, min_colonne: int = 3,
           sfondo_max: int = 25) -> Optional[Reticolo]:
    """Il reticolo, se c'e'.

    I pallini di una griglia stanno su poche righe e poche colonne: si cercano gli
    addensamenti delle x e delle y, non i punti uno per uno. Cosi' il rumore del tessuto, che
    produce altrettanti puntini ma sparsi, non forma ne' righe ne' colonne e resta fuori --
    ed e' quello che un punteggio sul singolo pallino non riusciva a distinguere.
    """
    punti = _pallini(bgr, rect, min_contrast, max_area, sfondo_max)
    if len(punti) < min_righe * min_colonne:
        return None

    lato = math.hypot(bgr.shape[1], bgr.shape[0])
    tolleranza = max(2.0, lato * 0.004)
    xs = _righe(punti[:, 0], tolleranza)
    ys = _righe(punti[:, 1], tolleranza)
    if len(xs) < min_colonne or len(ys) < min_righe:
        return None

    larghezza = float(bgr.shape[1] if rect is None else rect[2] - rect[0])
    altezza = float(bgr.shape[0] if rect is None else rect[3] - rect[1])
    px, nx = passo_dominante(punti[:, 0], larghezza * 0.02, larghezza * 0.6, tolleranza)
    py, ny = passo_dominante(punti[:, 1], altezza * 0.02, altezza * 0.6, tolleranza)
    if px <= 0 or py <= 0:
        return None
    rx = 1.0 / max(1, nx)
    ry = 1.0 / max(1, ny)
    return Reticolo(punti=[(float(p[0]), float(p[1])) for p in punti],
                    passo_x=px, passo_y=py, righe=len(ys), colonne=len(xs),
                    x0=min(xs), y0=min(ys), regolarita_x=rx, regolarita_y=ry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--min-contrast", type=int, default=18)
    args = ap.parse_args()
    im = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit("immagine illeggibile")
    r = detect(im, min_contrast=args.min_contrast)
    if not r:
        print("nessun reticolo")
        return 1
    print(f"pallini {len(r.punti)} · {r.righe} righe x {r.colonne} colonne")
    print(f"passo {r.passo_x:.2f} x {r.passo_y:.2f} px · "
          f"regolarita {r.regolarita_x:.3f} / {r.regolarita_y:.3f}")
    print(f"primo pallino a ({r.x0:.1f}, {r.y0:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


@dataclass
class ReticoloNoto:
    """Il reticolo trovato sapendo gia' quanto e' fitto."""
    passo: float                       # in pixel, quello dato
    fase_x: float                      # dove cade la prima colonna, in pixel assoluti
    fase_y: float
    colonne: List[float]               # le x occupate, in pixel assoluti
    righe: List[float]
    pallini_usati: int
    sostegno: float                    # quota dei pallini che cadono sul reticolo


def _fase(valori: np.ndarray, passo: float, tolleranza: float) -> Tuple[float, int]:
    """Dove mettere il reticolo, dato il passo: la fase su cui cade piu' roba."""
    if valori.size == 0 or passo <= 0:
        return 0.0, 0
    resti = np.mod(valori, passo)
    migliore, quanti = 0.0, -1
    for candidata in np.unique(np.round(resti, 1)):
        d = np.abs(resti - candidata)
        d = np.minimum(d, passo - d)            # la fase e' circolare
        n = int((d <= tolleranza).sum())
        if n > quanti:
            migliore, quanti = float(candidata), n
    return migliore, quanti


def trova_con_passo(bgr: np.ndarray, rect: Sequence[int], passo: float,
                    min_contrast: int = 35, max_area: int = 60, sfondo_max: int = 25,
                    tolleranza: Optional[float] = None) -> Optional[ReticoloNoto]:
    """Il reticolo, quando il passo si sa gia'.

    E' il modo giusto di porre il problema: il passo in pixel e' il passo del kit in
    millimetri diviso i millimetri per pixel, e sull'archivio quel conto azzecca il vero in 347
    celle su 351. Non c'e' niente da cercare, c'e' da **verificare dove cade** -- e una fase e'
    una sola incognita per asse, contro un passo che ha infiniti sottomultipli.
    """
    if passo <= 0:
        return None
    punti = _pallini(bgr, rect, min_contrast, max_area, sfondo_max)
    if len(punti) < 4:
        return None
    toll = tolleranza if tolleranza is not None else max(2.0, passo * 0.12)

    fx, nx = _fase(punti[:, 0], passo, toll)
    fy, ny = _fase(punti[:, 1], passo, toll)
    if nx < 3 or ny < 2:
        return None

    def sulla_griglia(valori: np.ndarray, fase: float) -> np.ndarray:
        d = np.abs(np.mod(valori - fase, passo))
        d = np.minimum(d, passo - d)
        return d <= toll

    tenuti = sulla_griglia(punti[:, 0], fx) & sulla_griglia(punti[:, 1], fy)
    if tenuti.sum() < 4:
        return None
    dentro = punti[tenuti]
    colonne = sorted({float(np.round((x - fx) / passo) * passo + fx) for x in dentro[:, 0]})
    righe = sorted({float(np.round((y - fy) / passo) * passo + fy) for y in dentro[:, 1]})
    return ReticoloNoto(passo=passo, fase_x=fx, fase_y=fy, colonne=colonne, righe=righe,
                        pallini_usati=int(tenuti.sum()),
                        sostegno=float(tenuti.sum()) / len(punti))
