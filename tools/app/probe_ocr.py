"""Il nome della sonda letto sullo schermo, abbinato alle sonde dell'anagrafica.

La rete sonda conosce solo le sonde viste in training (44 su 75), e davanti a una sonda mai
vista risponde con un'altra dello stesso vendor, spesso con confidenza alta: SI2C41 diventa
AC2541 a 0.95. Il nome scritto sullo schermo invece c'e' anche per le sonde nuove, e la lista
dei nomi viene dall'anagrafica, non dal training.

Le chiavi si costruiscono **solo** dai nomi dell'anagrafica (foglio PROBE e colonna "Modello
sonda" del foglio FSS), mai dai nomi delle cartelle: le acquisizioni non si rinominano, e una
parola chiave presa da una cartella di prova sbaglia in silenzio sulla prima cartella vera.

L'abbinamento e' per parola: un nome deve comparire come parola intera (o come parte di una
parola, se e' lungo almeno `MIN_SUBSTRING`), e dentro una parola vince il nome piu' lungo —
altrimenti `L441` scatterebbe dentro `CL4416R`.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# Lettere che l'OCR scambia con le cifre: si uniformano da entrambe le parti, cosi' "SI2C41"
# letto "S12C41" resta la stessa chiave. Solo fusioni, mai separazioni.
_CONFUSABLE = str.maketrans({"O": "0", "Q": "0", "I": "1", "S": "5"})
# Simboli che l'OCR mette al posto di una lettera, prima di togliere la punteggiatura:
# "SI2C41" sugli Esaote esce "$12C41".
_SYMBOLS = str.maketrans({"$": "S", "|": "I", "§": "S"})
# Un nome cosi' corto ("9L", "11L") vale solo come parola intera: come pezzo di parola
# comparirebbe ovunque.
MIN_SUBSTRING = 4
# Parole del nome che non identificano la sonda.
_NOISE = {"PROBE", "SONDA", "NUOVI", "TEMPLATE", "DEPTH", "MODALITA", "VISUALIZZAZIONE", "AGHI",
          "LINEARE", "CONVEX"}
VENDOR_PREFIXES = ("CANON/TOSHIBA", "CANON", "TOSHIBA", "ESAOTE", "GE", "BK", "HITACHI",
                   "MINDRAY", "PHILIPS", "SIEMENS", "SUPERSONIC", "TERASON", "KOELIS", "LA")


def normalize(text: str) -> str:
    return _plain(text).translate(_CONFUSABLE)


def _plain(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper().translate(_SYMBOLS))


def name_keys(name: str) -> Set[str]:
    """Le forme con cui un nome di sonda puo' comparire sullo schermo.

    "Canon/Toshiba PLT-1005BT (14L5)" -> {"PLT1005BT", "14L5"}; "GE, 12L-RS" -> {"12LRS"}.
    Il prefisso del vendor si toglie: sullo schermo compare il modello, non la marca.
    """
    raw = (name or "").upper()
    for prefix in VENDOR_PREFIXES:
        raw = re.sub(rf"^\s*{re.escape(prefix)}\b[\s,\-]*", "", raw)
    keys = set()
    for part in re.split(r"[()]|\s+-\s+", raw):
        # rumore e cifre si giudicano sul testo com'e', prima di uniformare O/0 e I/1
        words = [w for w in re.split(r"\s+", part.strip()) if _plain(w) and _plain(w) not in _NOISE]
        if not words or not any(ch.isdigit() for w in words for ch in w):
            continue
        joined = normalize("".join(words))
        if len(joined) >= 2:
            keys.add(joined)
        for word in words:
            # una parola sola di un nome composto vale se da sola e' gia' distintiva
            if len(_plain(word)) >= 4 and any(ch.isdigit() for ch in word):
                keys.add(normalize(word))
    return keys


Group = FrozenSet[int]


@dataclass
class ProbeKeys:
    """Chiave normalizzata -> sonde che la portano.

    Una chiave di piu' sonde vale per il **gruppo**: la 8848 e la sua variante «NUOVI TEMPLATE
    DEPTH» si leggono uguali, e la lettura dice «una delle due». Molte ambiguita' spariscono
    restringendo al vendor: "14L5" e' sia Canon sia Siemens.
    """

    owners: Dict[str, Group] = field(default_factory=dict)
    names: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(cls, probes: Dict[int, str], variants: Iterable[Tuple[int, str]] = ()) -> "ProbeKeys":
        """`probes`: ID -> nome del foglio PROBE. `variants`: (ID, nome) dal foglio FSS.

        Le varianti del foglio FSS sono i nomi con cui la sonda e' stata davvero configurata.
        """
        owners: Dict[str, Set[int]] = defaultdict(set)
        for probe_id, name in probes.items():
            for key in name_keys(name):
                owners[key].add(probe_id)
        for probe_id, name in variants:
            if probe_id not in probes:
                continue
            # Una variante che porta il nome di un'altra sonda la unisce al gruppo: l'ID 0
            # (LA332) e' configurato piu' volte col nome "LA332E", che e' l'ID 11, e sullo
            # schermo di quelle macchine c'e' scritto LA332E. Fra i due decide la rete.
            for key in name_keys(name):
                owners[key].add(probe_id)
        return cls(owners={k: frozenset(v) for k, v in owners.items()}, names=dict(probes))

    def restricted(self, allowed: Iterable[int]) -> "ProbeKeys":
        allowed = set(allowed)
        owners = {k: frozenset(v & allowed) for k, v in self.owners.items()}
        return ProbeKeys(owners={k: v for k, v in owners.items() if v},
                         names={k: v for k, v in self.names.items() if k in allowed})


def _tokens(text: str) -> List[Tuple[str, bool]]:
    """Le parole del testo, e le coppie e terne vicine unite (secondo elemento: unita)."""
    words = [normalize(w) for w in re.split(r"\s+", text or "")]
    words = [w for w in words if w]
    # l'OCR spezza "TLC3-13" in "TLC3 -13": si provano anche le coppie e le terne vicine
    out = [(w, False) for w in words]
    for size in (2, 3):
        out += [("".join(words[i:i + size]), True) for i in range(len(words) - size + 1)]
    return out


def match_text(text: str, keys: ProbeKeys, exact: bool = False) -> Set[Group]:
    """I gruppi di sonde il cui nome compare nel testo.

    Un nome corto vale solo come parola intera e singola: unendo parole vicine "cs, i}"
    diventa "C51", cioe' la Philips C5-1 su uno schermo Esaote. Le parole unite valgono solo
    come nome esatto. Con `exact` nemmeno una parola singola vale come contenitore: serve a
    dire che il vendor e' sbagliato, e "L3-112" (una L3-12D letta male) contiene L3-11.
    """
    found: Set[Group] = set()
    for token, joined in _tokens(text):
        if joined or exact:
            # un'unione di parole vicine vale solo se e' il nome esatto ("TLC3 -13"):
            # come contenitore mescola data, ora e sigle ("12L 12/30/21" contiene L123)
            hits = [k for k in keys.owners if k == token and len(k) >= MIN_SUBSTRING]
        else:
            hits = [k for k in keys.owners
                    if k == token or (len(k) >= MIN_SUBSTRING and k in token)]
        # dentro una parola vince il nome piu' lungo: L441 non conta dentro CL4416R
        maximal = [k for k in hits if not any(k != other and k in other for other in hits)]
        found.update(keys.owners[k] for k in maximal)
    return found


@dataclass
class FolderReading:
    """`probe_ids`: la sonda letta, o il gruppo di sonde che si leggono allo stesso modo."""

    probe_ids: List[int]
    hits: Dict[int, int]
    images_read: int
    strong: bool
    reason: str

    @property
    def probe_id(self) -> Optional[int]:
        return self.probe_ids[0] if len(self.probe_ids) == 1 else None

    def as_dict(self) -> Dict:
        return {"probe_id": self.probe_id, "probe_ids": list(self.probe_ids),
                "hits": {str(k): v for k, v in sorted(self.hits.items())},
                "images_read": self.images_read, "strong": self.strong, "reason": self.reason}


def aggregate(per_image: Sequence[Set[Group]], min_hits: int = 2, min_ratio: float = 0.25,
              min_margin: float = 2.0) -> FolderReading:
    """La lettura della cartella: una sonda sola, vista in abbastanza fotogrammi, con margine.

    Il nome della sonda e' lo stesso su tutti i fotogrammi di una acquisizione: una sonda che
    compare in un fotogramma solo e' piu' probabilmente una parola letta male. Ogni fotogramma
    conta una volta per sonda; le sonde di un gruppo salgono insieme e restano pari.
    """
    counts: Counter = Counter()
    for groups in per_image:
        counts.update(set().union(*groups) if groups else set())
    read = len(per_image)
    if not counts:
        return FolderReading([], {}, read, False, "nessun nome di sonda letto")
    top = max(counts.values())
    leaders = sorted(pid for pid, n in counts.items() if n == top)
    second = max((n for pid, n in counts.items() if n < top), default=0)
    if top < min_hits:
        return FolderReading(leaders, dict(counts), read, False, f"letto in {top} fotogramma")
    if read and top / read < min_ratio:
        return FolderReading(leaders, dict(counts), read, False,
                             f"letto in {top} fotogrammi su {read}")
    if second and top < min_margin * second:
        runner = sorted(pid for pid, n in counts.items() if n == second)
        return FolderReading(leaders, dict(counts), read, False,
                             f"piu' sonde lette: {leaders} ({top}) e {runner} ({second})")
    return FolderReading(leaders, dict(counts), read, True, "")


# -- OCR --------------------------------------------------------------------
def _tesseract(path: Path, psm: int, timeout: float = 30.0) -> str:
    try:
        result = subprocess.run(["tesseract", str(path), "stdout", "--psm", str(psm)],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def read_image(image_path: Path, box: Optional[Dict], rect: Optional[Dict]) -> Dict[str, str]:
    """I testi di un fotogramma: il box #14 ingrandito, e lo schermo col ventaglio oscurato."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    texts: Dict[str, str] = {}
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    with tempfile.TemporaryDirectory() as tmp:
        if box:
            mx = max(6, int((box["right"] - box["left"]) * 0.25))
            my = max(4, int((box["bottom"] - box["top"]) * 0.5))
            crop = image.crop((max(0, box["left"] - mx), max(0, box["top"] - my),
                               min(width, box["right"] + mx), min(height, box["bottom"] + my)))
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
            path = Path(tmp) / "box.png"
            crop.save(path)
            texts["box_psm7"] = _tesseract(path, 7)
            texts["box_psm6"] = _tesseract(path, 6)
        full = image.copy()
        if rect:
            ImageDraw.Draw(full).rectangle(
                [rect["left"], rect["top"], rect["right"], rect["bottom"]], fill=(0, 0, 0))
        path = Path(tmp) / "full.png"
        full.save(path)
        texts["full_psm11"] = _tesseract(path, 11)
    return texts
