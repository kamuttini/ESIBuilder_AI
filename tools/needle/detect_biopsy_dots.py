#!/usr/bin/env python3
"""Find the biopsy guide of a biplane probe: the chain of coloured dots the machine draws.

Biplane probes do not calibrate on needles in water. The operator switches on the machine's own
biopsy lines and the screen draws the guide as a row of small dots, which in ESIBuilder are
aligned to the first dot. So the thing to find is not a bright streak in the tissue but an
overlay the machine painted.

That makes it far easier than the needle, for one reason: an ultrasound image is grey, and the
overlay is not. Coloured pixels inside the echo rectangle are almost never anatomy -- they are
drawing. Saturation finds the dots where brightness never could, because the dots are often
dimmer than the tissue around them.

The dots are then required to line up: a handful of coloured specks scattered around is a logo
or a marker, while a guide is ten or more of them on a line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class BiopsyGuide:
    dots: List[Tuple[float, float]]          # in frame coordinates, ordered along the line
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    angle_deg: float
    straightness: float                      # mean distance of the dots from the fitted line, px
    colour: Tuple[int, int, int]             # median BGR of the dots
    span: float                              # length of the chain over the rectangle's diagonal
    regularity: float                        # spread of the gaps over their mean; low is even

    @property
    def count(self) -> int:
        return len(self.dots)


def coloured_mask(bgr: np.ndarray, min_saturation: int = 90, min_value: int = 70) -> np.ndarray:
    """Pixels that are not grey. The overlay is drawn in colour; the tissue never is."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    return ((saturation >= min_saturation) & (value >= min_value)).astype(np.uint8) * 255


def bright_dot_mask(bgr: np.ndarray, min_contrast: int = 22, max_dot: int = 9) -> np.ndarray:
    """I pallini chiari, quelli disegnati in bianco invece che a colori.

    Sulle 831 immagini delle cartelle «biopsia» dell'archivio la guida e' colorata solo in due
    acquisizioni su trentacinque: quasi ovunque i pallini sono bianchi o grigi, e la maschera
    a saturazione non li vede. Qui si cerca quello che li distingue dal tessuto: sono **piccoli
    e piu' chiari di quello che hanno intorno**. Il top-hat tiene esattamente questo -- un
    massimo locale piu' stretto dell'elemento strutturante -- e lascia fuori le bande larghe
    del tessuto, che a occhio sono altrettanto luminose ma non sono piccole.
    """
    grigio = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_dot, max_dot))
    tophat = cv2.morphologyEx(grigio, cv2.MORPH_TOPHAT, elemento)
    return (tophat >= min_contrast).astype(np.uint8) * 255


def _ransac_line(points: np.ndarray, tolerance: float, iterations: int = 400,
                 rng: Optional[np.random.Generator] = None):
    """The line most of the dots agree on.

    Fitting all the dots at once does not work: inside the rectangle there are usually more
    coloured specks from the ruler, the logo and the labels than there are guide dots, and a
    least-squares fit over all of them lands on none. RANSAC asks instead which line has the
    most dots close to it, which is the guide by construction.
    """
    rng = rng or np.random.default_rng(0)
    n = len(points)
    if n < 2:
        return None
    best_inliers = None
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        a, b = points[i], points[j]
        direction = b - a
        length = float(np.hypot(*direction))
        if length < 8:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        distances = np.abs((points - a) @ normal)
        inliers = distances <= tolerance
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    return best_inliers


def detect(bgr: np.ndarray, rect: Tuple[int, int, int, int], min_dots: int = 6,
           max_dot_area: int = 400, max_straightness: float = 4.0,
           tolerance: float = 4.0, min_span: float = 0.30,
           max_irregularity: float = 0.55, edge_margin: float = 0.15,
           min_background: float = 18.0) -> Optional[BiopsyGuide]:
    """Two constraints beyond colour and alignment, and both are needed.

    The rectangle does not keep the interface out: machine labels are drawn inside RECT_ECHO,
    and a word in colour is a row of small coloured blobs on a line -- which is exactly what a
    guide looks like to the tests above. On the first run every single detection was a caption.

    A guide crosses the image, so the chain must span a good fraction of the rectangle's
    diagonal; a label spans a few percent. And the dots of a guide are evenly spaced, being drawn
    at fixed intervals, while the letters of a word are not.

    Those two let the depth ruler through, which is the next thing that is coloured, regular and
    long -- six of the next seven detections were ruler ticks. Position alone does not separate
    them, because on some machines RECT_ECHO is wide enough that the ruler sits well inside it.
    What does separate them is what they are drawn on: the guide is painted over the tissue and
    the ruler over the black margin, so the neighbourhood of the dots has to be lit.
    """
    left, top, right, bottom = rect
    h, w = bgr.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    crop = bgr[top:bottom, left:right]
    if crop.size == 0:
        return None

    mask = coloured_mask(crop)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    punti, tinte = [], []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (2 <= area <= max_dot_area):
            continue
        if max(width, height) > 6 * max(1, min(width, height)):
            continue
        x, y = int(round(centroids[i][0])), int(round(centroids[i][1]))
        if not (0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]):
            continue
        punti.append((float(centroids[i][0]), float(centroids[i][1])))
        tinte.append(int(hsv[y, x, 0]))
    if len(punti) < min_dots:
        return None

    points = np.array(punti, dtype=np.float32)
    hues = np.array(tinte)

    # the guide is drawn in one colour; the ruler and the labels are in others
    migliore = None
    for centro in np.unique(hues // 10):
        stesso = np.abs(((hues // 10) - centro)) <= 1
        if stesso.sum() < min_dots:
            continue
        gruppo = points[stesso]
        inliers = _ransac_line(gruppo, tolerance)
        if inliers is None or inliers.sum() < min_dots:
            continue
        if migliore is None or inliers.sum() > migliore[0]:
            migliore = (int(inliers.sum()), gruppo[inliers], int(centro))
    if migliore is None:
        return None
    points = migliore[1]

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    straightness = float(np.mean(np.abs((points - np.array([x0, y0])) @ normal)))
    if straightness > max_straightness:
        return None

    proiezione = (points - np.array([x0, y0])) @ np.array([vx, vy])
    order = np.argsort(proiezione)
    points = points[order]
    proiezione = proiezione[order]
    a = np.array([x0, y0]) + proiezione[0] * np.array([vx, vy])
    b = np.array([x0, y0]) + proiezione[-1] * np.array([vx, vy])

    diagonale = math.hypot(crop.shape[1], crop.shape[0])
    span = float(proiezione[-1] - proiezione[0]) / max(1.0, diagonale)
    if span < min_span:
        return None
    gaps = np.diff(proiezione)
    regularity = float(np.std(gaps) / max(1e-6, float(np.mean(gaps)))) if len(gaps) > 1 else 1.0
    if regularity > max_irregularity:
        return None

    # the depth ruler hugs a side; the guide crosses the sector
    xs = points[:, 0]
    larghezza = crop.shape[1]
    if float(np.max(xs)) <= edge_margin * larghezza or \
            float(np.min(xs)) >= (1.0 - edge_margin) * larghezza:
        return None

    # drawn on tissue, not on the black margin: sample around each dot, away from the dot itself
    grigio = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sfondi = []
    for px, py in points:
        x, y = int(round(px)), int(round(py))
        finestra = grigio[max(0, y - 14):y + 15, max(0, x - 14):x + 15]
        if finestra.size:
            sfondi.append(float(np.median(finestra)))
    if sfondi and float(np.median(sfondi)) < min_background:
        return None

    colori = [crop[int(round(p[1])), int(round(p[0]))] for p in points]
    mediano = np.median(np.array(colori), axis=0).astype(int)

    return BiopsyGuide(
        dots=[(float(p[0]) + left, float(p[1]) + top) for p in points],
        p1=(float(a[0]) + left, float(a[1]) + top),
        p2=(float(b[0]) + left, float(b[1]) + top),
        angle_deg=math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))),
        straightness=straightness,
        colour=tuple(int(c) for c in mediano),
        span=round(span, 3),
        regularity=round(regularity, 3),
    )


def detect_bright(bgr: np.ndarray, rect: Tuple[int, int, int, int], min_dots: int = 6,
                  max_dot_area: int = 400, tolerance: float = 4.0, min_span: float = 0.30,
                  max_irregularity: float = 0.55, edge_margin: float = 0.15,
                  min_background: float = 18.0, min_contrast: int = 22
                  ) -> Optional[BiopsyGuide]:
    """La stessa guida, quando i pallini sono bianchi invece che colorati.

    Cambia solo da dove vengono i punti: non piu' i pixel saturi ma i massimi locali stretti
    (`bright_dot_mask`). I controlli restano quelli, e servono tutti: senza, il righello e le
    scritte dell'interfaccia passano per guide -- sono anche loro piccoli, chiari e allineati.
    Il raggruppamento per tinta qui non ha senso e sparisce: in grigio la tinta non distingue
    niente, e a separare la guida dal resto resta il RANSAC.
    """
    left, top, right, bottom = (int(v) for v in rect)
    h, w = bgr.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    crop = bgr[top:bottom, left:right]
    if crop.size == 0:
        return None

    mask = bright_dot_mask(crop, min_contrast=min_contrast)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    punti = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (2 <= area <= max_dot_area):
            continue
        if max(width, height) > 6 * max(1, min(width, height)):
            continue
        punti.append((float(centroids[i][0]), float(centroids[i][1])))
    if len(punti) < min_dots:
        return None

    points = np.array(punti, dtype=np.float32)
    inliers = _ransac_line(points, tolerance)
    if inliers is None or inliers.sum() < min_dots:
        return None
    points = points[inliers]

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    straightness = float(np.mean(np.abs((points - np.array([x0, y0])) @ normal)))
    direzione = np.array([vx, vy])
    t = (points - np.array([x0, y0])) @ direzione
    ordine = np.argsort(t)
    points, t = points[ordine], t[ordine]
    a, b = points[0], points[-1]

    diagonale = math.hypot(crop.shape[1], crop.shape[0])
    span = float(t[-1] - t[0]) / max(1.0, diagonale)
    if span < min_span:
        return None
    passi = np.diff(t)
    regularity = float(np.std(passi) / max(1e-6, float(np.mean(passi))))
    if regularity > max_irregularity:
        return None

    # non appoggiata al bordo: il righello sta sul margine, la guida attraversa il tessuto
    margine = edge_margin
    dentro = [p for p in points
              if margine * crop.shape[1] <= p[0] <= (1 - margine) * crop.shape[1]]
    if len(dentro) < min_dots:
        return None

    grigio = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sfondi = []
    for px, py in points:
        x, y = int(round(px)), int(round(py))
        finestra = grigio[max(0, y - 14):y + 15, max(0, x - 14):x + 15]
        if finestra.size:
            sfondi.append(float(np.median(finestra)))
    if sfondi and float(np.median(sfondi)) < min_background:
        return None

    colori = [crop[int(round(p[1])), int(round(p[0]))] for p in points]
    mediano = np.median(np.array(colori), axis=0).astype(int)
    return BiopsyGuide(
        dots=[(float(p[0]) + left, float(p[1]) + top) for p in points],
        p1=(float(a[0]) + left, float(a[1]) + top),
        p2=(float(b[0]) + left, float(b[1]) + top),
        angle_deg=math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))),
        straightness=straightness,
        colour=tuple(int(c) for c in mediano),
        span=round(span, 3),
        regularity=round(regularity, 3),
    )


def detect_any(bgr: np.ndarray, rect: Tuple[int, int, int, int], **kwargs) -> Optional[BiopsyGuide]:
    """Prima a colori, poi in chiaro: sono la stessa guida disegnata in due modi."""
    colorata = detect(bgr, rect, **{k: v for k, v in kwargs.items() if k != "min_contrast"})
    if colorata is not None:
        return colorata
    return detect_bright(bgr, rect, **kwargs)
