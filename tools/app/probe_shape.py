"""Probe type (#04) from the images: linear, convex, or both (transrectal biplane).

STATO: **non agganciato all'app**. La logica di decisione (entrambe le forme presenti =>
sonda transrettale biplana) e' corretta e riusabile; la *misura* della forma sul frame intero
non lo e'. Misurato su cartelle reali con verita' dall'anagrafica:

* profilo di ampiezza dell'area luminosa, banda dalla prima all'ultima riga: 3/5 cartelle;
* stessa misura sulla banda contigua piu' lunga: 1/6;
* maschera di speckle (varianza locale) al posto della luminosita': 1/6.

La causa e' sempre la stessa: sul frame intero la geometria e' dominata dalla grafica
dell'ecografo (barre, pannelli, menu a tutta larghezza), non dall'immagine ecografica. La
misura ha senso solo **dentro il crop del rettangolo ecografico**, che nel progetto esiste
gia' come modello in produzione (mean IoU 0.917) ed e' lo stesso schema usato dal
classificatore L/T, che lavora sui crop del rect.

Conseguenza di progetto: la proposta per #04 dall'analisi immagini non puo' arrivare al passo
0 (codici), perche' lo rect si conosce al passo 2. Sulla pagina codici #04 arriva
dall'anagrafica; l'analisi immagini la confermera' o correggera' piu' avanti.

Il modulo resta qui con la sua CLI come banco di misura per quando il crop sara' disponibile.

Come funziona la misura: un array lineare disegna un rettangolo (area larga uguale in alto e
in basso), un convex disegna un ventaglio (si allarga con la profondita'). Il discriminante e'
il rapporto fra l'ampiezza in basso e quella in alto.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from constants import (
    PROBE_TYPE_CONVEX,
    PROBE_TYPE_LINEAR,
    PROBE_TYPE_TRANS_L,
    PROBE_TYPE_TRANS_T,
)

LINEAR = "linear"
CONVEX = "convex"
UNCERTAIN = "uncertain"

# Tuned on real acquisitions: see the CLI at the bottom for the evaluation harness.
BRIGHT_THRESHOLD = 28  # a pixel counts as active above this grey level
BAND_FRACTION = 0.15  # rows kept: at least this share of the widest row
EDGE_FRACTION = 0.22  # share of the band used as "top" and "bottom" samples
MIN_BAND_ROWS = 40  # below this the band is UI chrome, not an ultrasound area
MIN_TOP_WIDTH = 0.05  # top sample must span at least this share of the image width
LINEAR_MAX_RATIO = 1.30
CONVEX_MIN_RATIO = 1.70
MIN_SHARE_FOR_BOTH = 0.20  # each shape must reach this share to call the probe biplane
MIN_IMAGES_TO_DECIDE = 8  # fewer analysable frames than this: no verdict


@dataclass
class ImageShape:
    name: str
    shape: str
    ratio: float
    band: tuple[int, int]

    def as_dict(self) -> Dict:
        return {"name": self.name, "shape": self.shape, "ratio": self.ratio, "band": list(self.band)}


def classify_image(path: Path, max_side: int = 480) -> Optional[ImageShape]:
    """Width profile of the bright area, bottom vs top."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None

    try:
        with Image.open(path) as image:
            grey = image.convert("L")
            grey.thumbnail((max_side, max_side))
            data = np.asarray(grey, dtype="int16")
    except Exception:
        return None

    if data.size == 0:
        return None

    widths = (data > BRIGHT_THRESHOLD).sum(axis=1)
    if widths.max() <= 0:
        return None

    # Longest *contiguous* run of active rows. Taking first-to-last instead would swallow
    # the UI bars at the top or bottom of the screen, with dark rows in between: the median
    # of the bottom window then lands on black and every image looks linear.
    active = widths >= BAND_FRACTION * widths.max()
    best_start, best_len, start = 0, 0, None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start > best_len:
                best_start, best_len = start, index - start
            start = None
    if start is not None and len(active) - start > best_len:
        best_start, best_len = start, len(active) - start
    if best_len < MIN_BAND_ROWS:
        return None

    top_row, bottom_row = best_start, best_start + best_len - 1
    edge = max(2, int(best_len * EDGE_FRACTION))

    top = float(np.median(widths[top_row : top_row + edge]))
    bottom = float(np.median(widths[bottom_row - edge + 1 : bottom_row + 1]))
    if top < MIN_TOP_WIDTH * data.shape[1] or bottom <= 0:
        return None

    ratio = round(bottom / top, 3)
    if ratio <= LINEAR_MAX_RATIO:
        shape = LINEAR
    elif ratio >= CONVEX_MIN_RATIO:
        shape = CONVEX
    else:
        shape = UNCERTAIN
    return ImageShape(name=path.name, shape=shape, ratio=ratio, band=(top_row, bottom_row))


def classify_folder(paths: List[Path], sample: int = 40) -> Dict:
    """Vote over a sample of frames and decide the probe type."""
    if not paths:
        return {"probe_type": None, "reason": "nessuna immagine", "votes": {}, "images": []}

    step = max(1, len(paths) // sample)
    picked = paths[::step][:sample]
    shapes = [result for result in (classify_image(path) for path in picked) if result]
    votes = Counter(item.shape for item in shapes)
    decided = votes[LINEAR] + votes[CONVEX]

    if decided < MIN_IMAGES_TO_DECIDE:
        return {
            "probe_type": None,
            "reason": (
                f"solo {decided} immagini analizzabili su {len(picked)}: troppo poche per decidere"
                if decided
                else "forma non riconoscibile su nessuna immagine"
            ),
            "votes": dict(votes),
            "images": [s.as_dict() for s in shapes],
        }

    linear_share = votes[LINEAR] / decided
    convex_share = votes[CONVEX] / decided

    if linear_share >= MIN_SHARE_FOR_BOTH and convex_share >= MIN_SHARE_FOR_BOTH:
        probe_type = None  # transrectal: which of 3/4 depends on the plane of this setup
        both = True
        reason = (
            f"presenti entrambe le forme ({votes[LINEAR]} lineari, {votes[CONVEX]} convex): "
            "sonda transrettale biplana, il tipo 3 (L) o 4 (T) dipende dal piano di questo setup"
        )
    else:
        both = False
        if linear_share > convex_share:
            probe_type = PROBE_TYPE_LINEAR
            reason = f"{votes[LINEAR]}/{decided} immagini con area ad ampiezza costante: lineare"
        else:
            probe_type = PROBE_TYPE_CONVEX
            reason = f"{votes[CONVEX]}/{decided} immagini con area che si allarga in profondita': convex"

    return {
        "probe_type": probe_type,
        "biplane": both,
        "candidates": [PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T] if both else [],
        "confidence": round(max(linear_share, convex_share), 3),
        "reason": reason,
        "votes": dict(votes),
        "images_analysed": len(shapes),
        "images": [s.as_dict() for s in shapes[:20]],
    }


def main(argv: List[str]) -> int:
    """Evaluation harness: probe_shape.py <cartella> [...] -> verdict per folder."""
    import sys

    from importer import scan_folder

    if not argv:
        print("uso: probe_shape.py <cartella> [...]")
        return 2
    for raw in argv:
        folder = Path(raw)
        images = scan_folder(folder)
        result = classify_folder(images)
        print(f"\n{folder.name}  ({len(images)} immagini)")
        print(f"   voti: {result['votes']}")
        print(f"   -> {result['reason']}")
        if result.get("probe_type") is not None:
            print(f"   tipo sonda proposto: {result['probe_type']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))


# --- misura dentro il crop del rettangolo ecografico ----------------------
# E' la versione che ha senso: fuori dal rect la geometria e' quella della grafica
# dell'ecografo. Qui il crop contiene solo l'immagine ecografica.
CROP_LINEAR_MAX_RATIO = 1.25
CROP_CONVEX_MIN_RATIO = 1.60


def classify_image_in_rect(
    path: Path, rect: Dict[str, int], max_side: int = 320
) -> Optional[ImageShape]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        with Image.open(path) as raw:
            image = raw.convert("L")
            width, height = image.size
            left = max(0, min(int(rect["left"]), width - 1))
            top = max(0, min(int(rect["top"]), height - 1))
            right = max(left + 1, min(int(rect["right"]), width))
            bottom = max(top + 1, min(int(rect["bottom"]), height))
            crop = image.crop((left, top, right, bottom))
            crop.thumbnail((max_side, max_side))
            data = np.asarray(crop, dtype="int16")
    except Exception:
        return None
    if data.size == 0 or data.shape[0] < 12:
        return None

    widths = (data > BRIGHT_THRESHOLD).sum(axis=1)
    if widths.max() <= 0:
        return None
    rows = data.shape[0]
    edge = max(2, int(rows * EDGE_FRACTION))
    top_width = float(np.median(widths[:edge]))
    bottom_width = float(np.median(widths[rows - edge :]))
    if top_width < MIN_TOP_WIDTH * data.shape[1]:
        return None

    ratio = round(bottom_width / top_width, 3) if top_width > 0 else 0.0
    if ratio <= CROP_LINEAR_MAX_RATIO:
        shape = LINEAR
    elif ratio >= CROP_CONVEX_MIN_RATIO:
        shape = CONVEX
    else:
        shape = UNCERTAIN
    return ImageShape(name=path.name, shape=shape, ratio=ratio, band=(0, rows - 1))


def classify_folder_in_rect(paths: List[Path], rect: Dict[str, int], sample: int = 40) -> Dict:
    """Same voting as classify_folder, measuring inside the rect crop."""
    if not paths or not rect:
        return {"probe_type": None, "reason": "servono immagini e rettangolo", "votes": {}}
    step = max(1, len(paths) // sample)
    picked = paths[::step][:sample]
    shapes = [s for s in (classify_image_in_rect(path, rect) for path in picked) if s]
    votes = Counter(item.shape for item in shapes)
    decided = votes[LINEAR] + votes[CONVEX]
    if decided < MIN_IMAGES_TO_DECIDE:
        return {
            "probe_type": None,
            "reason": f"solo {decided} crop analizzabili su {len(picked)}",
            "votes": dict(votes),
        }
    linear_share = votes[LINEAR] / decided
    convex_share = votes[CONVEX] / decided
    if linear_share >= MIN_SHARE_FOR_BOTH and convex_share >= MIN_SHARE_FOR_BOTH:
        return {
            "probe_type": None,
            "biplane": True,
            "candidates": [PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T],
            "confidence": round(min(linear_share, convex_share) * 2, 3),
            "reason": (
                f"entrambe le forme ({votes[LINEAR]} lineari, {votes[CONVEX]} convex): "
                "sonda transrettale biplana"
            ),
            "votes": dict(votes),
            "images_analysed": len(shapes),
        }
    probe_type = PROBE_TYPE_LINEAR if linear_share > convex_share else PROBE_TYPE_CONVEX
    return {
        "probe_type": probe_type,
        "biplane": False,
        "confidence": round(max(linear_share, convex_share), 3),
        "reason": (
            f"{votes[LINEAR]}/{decided} lineari" if probe_type == PROBE_TYPE_LINEAR
            else f"{votes[CONVEX]}/{decided} convex"
        ),
        "votes": dict(votes),
        "images_analysed": len(shapes),
    }
