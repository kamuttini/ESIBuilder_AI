"""Controllo: l'abbinamento dei nomi di sonda letti sullo schermo.

Ogni caso viene da una cartella vera della misura del 2026-09-23
(`artifacts/40_outputs_eval/probe_ocr_decision_20260923`): sono gli errori che le regole di
`probe_ocr.py` evitano. Legge l'anagrafica e la lista sonde-per-vendor, non scrive niente.

    python3 tools/app/selftest_probe_ocr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_decision  # noqa: E402
import probe_ocr  # noqa: E402

CASES = [
    # (testo letto, sonde attese, perche')
    ("PRS 12 $12C41 General", {35}, "SI2C41 sugli Esaote esce $12C41"),
    ("Elastography TLC3 -13 Thyroid", {12}, "l'OCR spezza TLC3-13 in due parole"),
    ("CL4416R1", {57}, "CL4416R1 non e' anche CL4416R ne' L441"),
    ("CL4416R", {34}, "CL4416R non e' anche L441"),
    ("Mixing insertions (wai cs, i} Zoom", set(), "cs, i} unite non sono la Philips C5-1"),
    ("M12 Tis0.4 12L 12/30/21 03:14:48PM", set(), "12L 12/30/21 unite non contengono la L12-3"),
    ("Tis 0.5 (L3-112 08/02/24", {51}, "come pezzo di parola vale (fra le sonde del vendor)"),
    ("L3-12H 0.00 MB", {40}, "nome esatto"),
    ("9L-RS 46Hz", set(), "un nome corto (9L) vale solo come parola intera"),
]
GROUPS = [
    ("LA332E", {0, 11}, "l'ID 0 e' configurato col nome LA332E, che e' l'ID 11"),
    ("BK 8848", {5, 14}, "la 8848 e la sua variante NUOVI TEMPLATE DEPTH"),
]


def main() -> int:
    catalog = probe_decision.ProbeCatalog.load()
    failures = 0
    for text, expected, why in CASES:
        got = set().union(*probe_ocr.match_text(text, catalog.keys)) if text else set()
        ok = got == expected
        failures += not ok
        print(f"{'ok ' if ok else 'NO '} {text!r:45} -> {sorted(got)} (atteso {sorted(expected)}): {why}")
    for text, expected, why in GROUPS:
        groups = probe_ocr.match_text(text, catalog.keys)
        ok = frozenset(expected) in groups
        failures += not ok
        print(f"{'ok ' if ok else 'NO '} {text!r:45} -> {[sorted(g) for g in groups]}: {why}")

    # exact: per accusare il vendor serve il nome esatto - su un GE, "L3-112" (una L3-12D
    # letta male) faceva dire che la sonda era la Esaote L3-11
    exact = probe_ocr.match_text("Tis 0.5 (L3-112 08/02/24", catalog.keys, exact=True)
    ok = not exact
    failures += not ok
    print(f"{'ok ' if ok else 'NO '} exact su 'L3-112' -> {[sorted(g) for g in exact]}")

    # decisione: sonda sconosciuta alla rete, letta sullo schermo
    texts = ["PRS 6 $12C41 General"] * 12
    decision = probe_decision.decide(catalog, "Esaote", {21: 0.92, 12: 0.05}, texts)
    ok = decision["status"] == "review" and decision["probe_id"] == 35
    failures += not ok
    print(f"{'ok ' if ok else 'NO '} decisione SI2C41 vs rete AC2541 -> {decision['status']} {decision['probe_id']}")
    decision = probe_decision.decide(catalog, "Esaote", {0: 0.96, 11: 0.01}, ["LA332E"] * 12)
    ok = decision["status"] == "accepted" and decision["probe_id"] == 0
    failures += not ok
    print(f"{'ok ' if ok else 'NO '} decisione LA332E con rete LA332 -> {decision['status']} {decision['probe_id']}")
    decision = probe_decision.decide(catalog, "Esaote", {12: 0.99}, [""] * 12)
    ok = decision["status"] == "accepted" and decision["probe_id"] == 12
    failures += not ok
    print(f"{'ok ' if ok else 'NO '} niente letto -> resta la rete: {decision['status']} {decision['probe_id']}")

    print("tutti i controlli passati" if not failures else f"{failures} controlli falliti")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
