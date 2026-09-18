#!/usr/bin/env python3
"""Legge `setup_<ID>.grid`, il risultato della calibrazione a griglia del vecchio ESIBuilder.

Per le biplane a griglia la calibrazione **non finisce nel `.fss`**: finisce qui, scritta da
`QFileFssGrid`. Per ogni depth e per ogni flip ci sono due rettangoli -- il cerchio della sonda
e la griglia -- in pixel e in millimetri, piu' quali righe e colonne della matrice si vedono
nell'immagine.

I rettangoli in millimetri sono `QVariant` serializzati dentro l'INI, e l'escaping non e' quello
del C: QSettings scrive `\\xNN` con **quante cifre servono** (`\\xe` vale 0x0e) e `\\0` come
ottale, quindi `codecs.escape_decode` si ferma sul primo `\\xe`. Qui si segue la regola di Qt
(`iniUnescapedString`), e i 29 file dell'archivio si leggono tutti.

  python3 tools/needle/read_fss_grid.py --file "<...>/DB_setup/setup_42.grid"
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEX = set("0123456789abcdefABCDEF")
OTT = set("01234567")


def qsettings_unescape(testo: str) -> bytes:
    """Da come QSettings scrive un binario dentro un INI ai byte veri."""
    fuori = bytearray()
    i = 0
    while i < len(testo):
        c = testo[i]
        if c != "\\":
            fuori.append(ord(c) & 0xFF)
            i += 1
            continue
        i += 1
        if i >= len(testo):
            break
        c = testo[i]
        if c in ("x", "X"):
            i += 1
            cifre = ""
            while i < len(testo) and testo[i] in HEX and len(cifre) < 4:
                cifre += testo[i]
                i += 1
            fuori.append(int(cifre or "0", 16) & 0xFF)
        elif c in OTT:
            cifre = ""
            while i < len(testo) and testo[i] in OTT and len(cifre) < 3:
                cifre += testo[i]
                i += 1
            fuori.append(int(cifre, 8) & 0xFF)
        else:
            speciali = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11}
            fuori.append(speciali.get(c, ord(c)) & 0xFF)
            i += 1
    return bytes(fuori)


def rect_mm(valore: str) -> Optional[Tuple[float, float, float, float]]:
    """Il QRectF dentro `@Variant(...)`: tipo 0x14, poi quattro double big endian."""
    m = re.search(r"@Variant\((.*)\)\s*$", valore.strip(), re.S)
    if not m:
        return None
    b = qsettings_unescape(m.group(1))
    if len(b) < 4 + 32:
        return None
    (tipo,) = struct.unpack_from(">I", b, 0)
    if tipo != 0x14:          # QMetaType::QRectF
        return None
    return struct.unpack_from(">4d", b, 4)


def rect_px(valore: str) -> Optional[Tuple[int, int, int, int]]:
    """`@Rect(x y w h)` -> (x, y, larghezza, altezza)."""
    m = re.search(r"@Rect\(([-\d\s]+)\)", valore or "")
    if not m:
        return None
    parti = [int(v) for v in m.group(1).split()]
    return tuple(parti[:4]) if len(parti) >= 4 else None


def leggi(path: Path) -> Dict:
    # `interpolation=None`: dentro i @Variant capitano dei `%` (sono byte, non segnaposto) e
    # configparser di suo prova a espanderli e si ferma.
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(path, encoding="utf-8")
    fuori: Dict[str, Dict] = {"file": str(path), "depth": {}}
    for sezione in cp.sections():
        if sezione == "Settings":
            fuori["immagini"] = [v.strip() for v in
                                 cp.get(sezione, "ListFile", fallback="").split(",") if v.strip()]
            continue
        m = re.fullmatch(r"(CircleRect|GridRect)(?:_(\d+))?", sezione)
        if not m:
            continue
        quale, indice = m.group(1), m.group(2)
        voce = {
            "px": rect_px(cp.get(sezione, "RectPX", fallback="")),
            "mm": rect_mm(cp.get(sezione, "RectMM", fallback="")),
            "depth_mm": cp.getfloat(sezione, "fDepthMM", fallback=None),
            "depth_idx": cp.getint(sezione, "nDepthIndex", fallback=None),
            "flip_idx": cp.getint(sezione, "nFlipIndex", fallback=None),
            "ratio_x": cp.getfloat(sezione, "fPixelRatioX", fallback=None),
            "ratio_y": cp.getfloat(sezione, "fPixelRatioY", fallback=None),
        }
        if cp.has_option(sezione, "bVisibleMatrix"):
            voce["matrice_visibile"] = [int(v) for v in
                                        cp.get(sezione, "bVisibleMatrix").split(",")]
        chiave = "corrente" if indice is None else indice
        fuori["depth"].setdefault(chiave, {})[quale] = voce
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path)
    ap.add_argument("--cache", type=Path, help="tutta la cache legacy, per un riepilogo")
    args = ap.parse_args()

    if args.file:
        print(json.dumps(leggi(args.file), indent=1, ensure_ascii=False))
        return 0

    if args.cache:
        letti = falliti = 0
        senza_mm = 0
        for f in sorted(args.cache.rglob("setup_*.grid")):
            dati = leggi(f)
            voci = [v for d in dati["depth"].values() for v in d.values()]
            if not voci:
                falliti += 1
                continue
            letti += 1
            senza_mm += sum(1 for v in voci if v["mm"] is None)
            print(f"{f.parent.parent.name[:52]:54s} depth {len(dati['depth']):3d} "
                  f"· rettangoli {len(voci):3d} · senza mm {sum(1 for v in voci if v['mm'] is None)}")
        print(f"\nletti {letti}, illeggibili {falliti}, rettangoli senza millimetri {senza_mm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
