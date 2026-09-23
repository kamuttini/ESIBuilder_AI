"""La lista delle sonde per vendor: `tools/app/sonde_per_vendor.csv`.

Regola di dominio: ogni sonda appartiene a un solo vendor, e il vendor e' **chi costruisce
l'ecografo**. Piu' ecografi dello stesso vendor condividono le sue sonde; un ecografo non usa
mai le sonde di un altro vendor. I sistemi montati sopra la macchina di un altro (Elesta su
Esaote, le piattaforme di fusione come Uronav, Biopsee, Koelis) prendono il vendor della
macchina che sta sotto.

Perche' un file a parte e non una colonna in `encoding_struct`: l'anagrafica arriva a versioni
datate, e una colonna aggiunta da noi andrebbe persa alla versione successiva. La lista e' nostra
e si **riallinea** a ogni anagrafica nuova: la chiave e' l'ID sonda, stabile fra le versioni
(nessuna sonda rinominata fra 2026-02-23 e 2026-08-31). Decisione provvisoria, da rivedere:
`docs/specifiche_app_esibuilder_ai_2026-08-26.md`, punti aperti.

Il file usa `;` come separatore: si apre in Excel italiano senza passare dall'importazione.

    python3 tools/app/sonde_per_vendor.py --riallinea   # cosa manca rispetto all'anagrafica
    python3 tools/app/sonde_per_vendor.py --semina      # ricostruisce la lista (sovrascrive)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anagrafica import COL_ECHO_ID, COL_ECHO_MODEL, COL_PROBE_ID, COL_PROBE_MODEL  # noqa: E402
from anagrafica import SHEET_ECHO, SHEET_PROBE, Anagrafica, default_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LIST_PATH = Path(__file__).resolve().parent / "sonde_per_vendor.csv"
COLUMNS = ("id_sonda", "modello_sonda", "vendor", "fonte", "note")

# Prima parola del modello ecografo -> vendor. Le classi della rete vendor dove esistono;
# Canon e Toshiba restano insieme come nell'anagrafica (stessa azienda, stesse sonde).
BRAND = {
    "esaote": "Esaote", "elesta": "Esaote",
    "ge": "GE", "bk": "BK", "bk3000": "BK", "bk5000": "BK",
    "hitachi": "Hitachi", "arietta": "Hitachi",
    "canon": "Canon/Toshiba", "canon/toshiba": "Canon/Toshiba", "toshiba": "Canon/Toshiba",
    "mindray": "Mindray", "philips": "Philips", "siemens": "Siemens",
    "supersonic": "SuperSonic", "terason": "Terason",
    "alpinion": "Alpinion", "alpion": "Alpinion",
    "exact": "ExactVu", "sonostar": "Sonostar", "dawei": "Dawei", "vinno": "Vinno",
    "siui": "SIUI",
}
# Piattaforme che ricevono il video da un altro ecografo: il loro nome non dice di chi sono
# le sonde, quindi non votano.
PLATFORMS = ("uronav", "biopsee", "koelis")


def echo_vendor(model: str) -> Optional[str]:
    """Il vendor di un modello ecografo, o None per una piattaforma o un nome sconosciuto."""
    first = re.split(r"[\s,]+", (model or "").strip())[0].lower()
    if first in PLATFORMS:
        return None
    return BRAND.get(first)


def _ids(rows: List[Dict[str, str]], id_col: str, value_col: str) -> Dict[int, str]:
    return {
        int(row[id_col]): row.get(value_col, "")
        for row in rows
        if row.get(id_col, "").lstrip("-").isdigit()
    }


def derive(anagrafica: Anagrafica) -> List[Dict[str, str]]:
    """Una riga per ogni sonda del foglio PROBE, con il vendor votato dalle configurazioni FSS."""
    echoes = _ids(anagrafica._data_rows(SHEET_ECHO), COL_ECHO_ID, COL_ECHO_MODEL)
    probes = _ids(anagrafica._data_rows(SHEET_PROBE), COL_PROBE_ID, COL_PROBE_MODEL)
    votes: Dict[int, Counter] = defaultdict(Counter)
    for row in anagrafica.fss_rows():
        raw_echo, raw_probe = row.get("id_echo", ""), row.get("id_probe", "")
        if not (raw_echo.isdigit() and raw_probe.isdigit()):
            continue
        vendor = echo_vendor(echoes.get(int(raw_echo), ""))
        if vendor:
            votes[int(raw_probe)][vendor] += 1

    stamp = anagrafica.path.stem.replace("encoding_struct", "").strip()
    out = []
    for probe_id in sorted(probes):
        seen = votes.get(probe_id) or Counter()
        vendor, note = "", ""
        if len(seen) == 1:
            vendor = next(iter(seen))
        elif len(seen) > 1:
            note = "vendor in conflitto: " + ", ".join(f"{v} ({n})" for v, n in seen.most_common())
        else:
            note = "nessuna configurazione di un ecografo riconoscibile: da assegnare"
        out.append({
            "id_sonda": str(probe_id),
            "modello_sonda": probes[probe_id],
            "vendor": vendor,
            "fonte": f"anagrafica {stamp}, {sum(seen.values())} configurazioni" if vendor else "",
            "note": note,
        })
    return out


def load(path: Path = LIST_PATH) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def save(rows: List[Dict[str, str]], path: Path = LIST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def add(probe_id: int, model: str, vendor: str, fonte: str, path: Path = LIST_PATH) -> Dict[str, str]:
    """Una sonda nuova nella lista. Un ID gia' presente non si sovrascrive: si segnala."""
    rows = load(path)
    for row in rows:
        if row.get("id_sonda") == str(probe_id):
            return {**row, "note": (row.get("note") or "") + " (gia' presente: non modificata)"}
    row = {
        "id_sonda": str(probe_id),
        "modello_sonda": model,
        "vendor": list_vendor(vendor) if vendor else "",
        "fonte": fonte,
        "note": "" if vendor else "vendor da assegnare",
    }
    save(rows + [row], path)
    return row


def list_vendor(net_vendor: str) -> str:
    """Il vendor della lista per una classe della rete vendor: Canon e Toshiba sono uno solo."""
    first = re.split(r"[\s,]+", (net_vendor or "").strip())[0].lower()
    return BRAND.get(first, net_vendor or "")


def probes_of(vendor: str, rows: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Le sonde di un vendor. Una sonda senza vendor non appartiene a nessuno."""
    rows = load() if rows is None else rows
    return [row for row in rows if row.get("vendor") == vendor]


def realign(anagrafica: Anagrafica, rows: List[Dict[str, str]]) -> Dict[str, List]:
    """Cosa e' cambiato nell'anagrafica rispetto alla lista. Non scrive niente."""
    probes = _ids(anagrafica._data_rows(SHEET_PROBE), COL_PROBE_ID, COL_PROBE_MODEL)
    listed = {int(row["id_sonda"]): row for row in rows if row.get("id_sonda", "").isdigit()}
    derived = {int(row["id_sonda"]): row for row in derive(anagrafica)}
    return {
        # sonde nuove: il vendor proposto dalle configurazioni, se c'e', da confermare
        "nuove": [derived[pid] for pid in sorted(probes) if pid not in listed],
        "rinominate": [
            {"id_sonda": pid, "lista": listed[pid]["modello_sonda"], "anagrafica": probes[pid]}
            for pid in sorted(probes)
            if pid in listed and listed[pid]["modello_sonda"] != probes[pid]
        ],
        "senza_vendor": [row for pid, row in sorted(listed.items()) if not row.get("vendor")],
        "sparite": [row for pid, row in sorted(listed.items()) if pid not in probes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoding-struct", help="anagrafica xlsx (default: la piu' recente)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--riallinea", action="store_true", help="confronta lista e anagrafica")
    mode.add_argument("--semina", action="store_true", help="ricostruisce la lista (sovrascrive)")
    args = parser.parse_args()

    path = Path(args.encoding_struct) if args.encoding_struct else default_path(REPO_ROOT)
    if not path or not path.is_file():
        print("anagrafica encoding_struct non trovata", file=sys.stderr)
        return 2
    anagrafica = Anagrafica(path)

    if args.semina:
        rows = derive(anagrafica)
        save(rows)
        print(f"{LIST_PATH}: {len(rows)} sonde, "
              f"{sum(1 for r in rows if not r['vendor'])} senza vendor (da {path.name})")
        return 0

    report = realign(anagrafica, load())
    for key, items in report.items():
        print(f"{key}: {len(items)}")
        for item in items:
            print(f"   {item}")
    return 1 if report["nuove"] or report["rinominate"] or report["sparite"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
