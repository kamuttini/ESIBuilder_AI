"""Quale sonda: la rete sonda e il nome letto sullo schermo si controllano a vicenda.

Misurato su 178 cartelle val+test (`artifacts/40_outputs_eval/probe_vendor_restriction_20260923`):
restringere la rete alle sonde del vendor non cambia niente (158 giuste in tutti e due i casi,
la rete non confonde mai i vendor). Gli errori sono dentro al vendor, e i piu' gravi sono le
sonde che la rete non ha mai visto: risponde con un'altra sonda, spesso a confidenza 0.95 e
oltre. La confidenza non li distingue; il nome scritto sullo schermo si'.

La regola:

- lo schermo dice una sonda della lista del vendor, e la rete dice la stessa → **accettata**;
- lo schermo dice una sonda, la rete un'altra → **revisione**, con la sonda letta come proposta:
  e' il caso della sonda sconosciuta alla rete;
- lo schermo dice una sonda di un **altro** vendor → **revisione**: o il vendor e' sbagliato o
  la lettura, e non si sceglie da soli;
- lo schermo non dice niente → resta la rete, come prima. Nessuna cartella che oggi e' giusta
  diventa sbagliata per colpa di una lettura mancata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import probe_ocr
import sonde_per_vendor


@dataclass
class ProbeCatalog:
    """La lista sonde-per-vendor con le chiavi di lettura dei nomi."""

    rows: List[Dict[str, str]]
    keys: probe_ocr.ProbeKeys

    @classmethod
    def load(cls, list_path: Optional[Path] = None, anagrafica=None) -> "ProbeCatalog":  # noqa: ANN001
        rows = sonde_per_vendor.load(list_path or sonde_per_vendor.LIST_PATH)
        probes = {int(r["id_sonda"]): r["modello_sonda"] for r in rows if r["id_sonda"].isdigit()}
        variants = []
        if anagrafica is None:
            path = sonde_per_vendor.default_path(sonde_per_vendor.REPO_ROOT)
            anagrafica = sonde_per_vendor.Anagrafica(path) if path else None
        if anagrafica is not None:
            variants = [
                (int(r["id_probe"]), r["probe_model"])
                for r in anagrafica.fss_rows()
                if r.get("id_probe", "").isdigit() and r.get("probe_model")
            ]
        return cls(rows=rows, keys=probe_ocr.ProbeKeys.build(probes, variants))

    def vendor_of(self, probe_id: int) -> str:
        for row in self.rows:
            if row["id_sonda"] == str(probe_id):
                return row.get("vendor") or ""
        return ""

    def probes_of(self, net_vendor: str) -> List[Dict[str, str]]:
        return sonde_per_vendor.probes_of(sonde_per_vendor.list_vendor(net_vendor), self.rows)

    def name(self, probe_id: Optional[int]) -> str:
        return self.keys.names.get(probe_id, "") if probe_id is not None else ""


def read_folder(catalog: ProbeCatalog, net_vendor: str, texts: Sequence[str]) -> Dict:
    """La lettura dello schermo, fra le sonde del vendor e fra tutte."""
    allowed = [int(r["id_sonda"]) for r in catalog.probes_of(net_vendor)]
    own_keys = catalog.keys.restricted(allowed) if allowed else catalog.keys
    own = probe_ocr.aggregate([probe_ocr.match_text(t, own_keys) for t in texts])
    # fra tutte le sonde solo nomi esatti: accusare il vendor chiede una prova piu' forte
    anywhere = probe_ocr.aggregate([probe_ocr.match_text(t, catalog.keys, exact=True) for t in texts])
    return {"own": own, "anywhere": anywhere, "vendor_has_list": bool(allowed)}


def decide(catalog: ProbeCatalog, net_vendor: str, probs: Dict[int, float],
           texts: Sequence[str]) -> Dict:
    """`probs`: probabilita' medie della rete sonda per ID. `texts`: i testi OCR per fotogramma."""
    ranked = sorted(probs.items(), key=lambda item: -item[1])
    net_id, net_conf = ranked[0] if ranked else (None, 0.0)
    reading = read_folder(catalog, net_vendor, texts)
    own, anywhere = reading["own"], reading["anywhere"]
    vendor_probes = catalog.probes_of(net_vendor)
    known = set(probs)
    out = {
        "net": {"probe_id": net_id, "confidence": round(float(net_conf), 4)},
        "ocr": own.as_dict(),
        "vendor": net_vendor,
        "vendor_probes": [
            {"probe_id": int(r["id_sonda"]), "model": r["modello_sonda"],
             "known_to_net": int(r["id_sonda"]) in known}
            for r in vendor_probes
        ],
    }

    if own.strong:
        if net_id in own.probe_ids:
            return {**out, "status": "accepted", "probe_id": net_id, "source": "rete + schermo",
                    "reason": f"la rete e il nome sullo schermo dicono {catalog.name(net_id)}"}
        # sonde che si leggono uguali (8848 e la sua variante): fra loro decide la rete, e se
        # la rete non ne conosce nessuna si propone la prima e lo si dice
        proposal = max(own.probe_ids, key=lambda pid: (probs.get(pid, -1.0), -pid))
        read = " / ".join(catalog.name(pid) for pid in own.probe_ids)
        return {**out, "status": "review", "probe_id": proposal, "source": "schermo",
                "reason": (f"sullo schermo c'e' {read}, "
                           f"la rete dice {catalog.name(net_id)} ({net_conf:.2f})")}

    if anywhere.strong and anywhere.probe_id is not None and \
            catalog.vendor_of(anywhere.probe_id) not in ("", sonde_per_vendor.list_vendor(net_vendor)):
        other = catalog.vendor_of(anywhere.probe_id)
        return {**out, "status": "review", "probe_id": None, "source": "",
                "reason": (f"sullo schermo c'e' {catalog.name(anywhere.probe_id)}, che e' una sonda "
                           f"{other}, ma il vendor riconosciuto e' {net_vendor}")}

    return {**out, "status": "accepted", "probe_id": net_id, "source": "rete",
            "reason": "nome della sonda non letto sullo schermo: resta la rete"
                      + (f" ({own.reason})" if own.reason else "")}
