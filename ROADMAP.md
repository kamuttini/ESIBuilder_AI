# Roadmap ESIBuilder AI

Aggiornata: 2026-07-08. Deliverable finale: nuova app ESIBuilder con AI (propone → l'utente conferma/corregge), per colleghi interni su Windows.

## Dove siamo

Fatto e operativo: vendor (acc 0.982), probe (con review assistita), rect (IoU 0.917), pipeline `fss_head` integrata con smoke test, modulo rect_depth autonomo pronto (checkpoint 2026-07-02), toolkit QA `.fss`.

Non ancora a livello produzione: orientation simbolico (trainer da riscrivere), scala (backlog correzioni GT P0/P1), piano L/T (la fase più importante per il collega), interfaccia utente, deploy Windows.

---

## Fase 0 — Import e messa in sicurezza (subito, ~1 giorno)

- [x] `CLAUDE.md` nel repo (contesto per Claude Code)
- [ ] Tag git dello stato attuale (`v0-checkpoint-codex`) come baseline
- [ ] Verifica ambiente: venv, Tesseract (fissare versione), volumi SSD raggiungibili
- [ ] Rilanciare gli smoke test esistenti per confermare che tutto gira ancora

## Fase 1 — Consolidamento orientation + scala (priorità scelta, ~2-4 settimane)

Scala:
- [ ] Smaltire coda correzioni GT P0 (34) e P1 (49) con i tool di review esistenti
- [ ] Retraining modello scala su GT corretto, eval su set completo leak-free
- [ ] Definire soglia di accettazione (es. % accepted per cartella) e criterio di uscita fase

Orientation:
- [ ] Riscrivere il trainer del detector template orientamento (era stato rimosso)
- [ ] Definire target IoU minimo di produzione e misurarlo su eval routed
- [ ] Review dedicata con GUI `review_symbol_rects_gui.py` sui casi peggiori

Criterio di chiusura fase: orientation e scala con metriche a livello degli altri blocchi e review rate accettabile.

## Fase 2 — Chiusura loop `.fss` end-to-end (~2-3 settimane)

- [ ] **Writer `.fss`**: la pipeline oggi produce solo CSV/preview — serve il modulo che assembla il file `.fss` vero dai valori predetti (vedi `docs/albero_decisionale_pipeline_fss_head_2026-07-08.md`)
- [ ] Validare rect_depth nella pipeline `fss_head` su batch reale (hook subprocess già presente nel codice, contrariamente ai report)
- [ ] Integrare orientation e scala consolidate in Fase 1
- [ ] Run su batch reale completo → validazione automatica con `compare_fss.py` e `audit_setup_outputs.py`
- [ ] Ridurre review rate vendor/probe sotto target (<3%) con tuning soglie
- [ ] Report di compatibilità: % .fss semanticamente compatibili con i legacy

Criterio di chiusura: batch reale con % compatibilità concordata e review rate sotto soglia.

## Fase 3 — Piano L/T (~2-3 settimane, parallelizzabile in parte con Fase 2)

- [ ] Consolidare dataset L/T da SSD (tool 16.x già pronti) + correzioni via preview HTML
- [ ] Training rete generale L/T, confronto per-vendor
- [ ] Integrare router `probe_id -> PROBETYPE` (#04) + split biplano 3-4
- [ ] Policy: stima sempre fornita, conferma utente sotto soglia di confidenza

## Fase 4 — App ESIBuilder AI (~4-8 settimane)

- [ ] Scelta stack UI (proposta: app web locale — backend Python FastAPI che riusa i moduli esistenti + frontend browser; i tool di review sono già HTML, si riusa il pattern)
- [ ] Workflow guidato: pagina codici configurazione (modificabili in corso, requisito collega) → import cartella → proposte AI per ogni step → conferma/correzione → generazione `.fss` + template
- [ ] Le correzioni utente alimentano il retraining (loop di apprendimento progressivo)
- [ ] Pilota con un collega su configurazioni reali

## Fase 5 — Windows e deploy (~2-3 settimane)

- [ ] Export modelli per inferenza CPU (ONNX) — su Windows non c'è MPS
- [ ] Fissare versioni Tesseract/Python/dipendenze per parità macOS-Windows (nota nel checkpoint rect_depth)
- [ ] Packaging (installer o cartella portabile) + test su macchina Windows reale
- [ ] Documentazione utente per i colleghi

---

## Rischi aperti

- Orientation, casi limite legacy (da pensarci per gli sviluppi futuri): alcuni ecografi hanno template di orientamento ribaltati/speculari che rompevano il match del vecchio software — quei progetti sono stati configurati per un solo orientamento (spesso UD) e la loro riga #16 contiene 4 copie dello stesso box (non NF/LR/UD/LRUD). Il nuovo detector dovrà gestirli esplicitamente. Inoltre il template può cambiare dimensione tra immagini della stessa cartella (serve matching multi-scala).
- Probe: classi rare/unseen con recall basso — mitigare con policy review + OCR refine
- Orientation: trainer da rifare, effort incerto
- Dipendenza da GT scala: la qualità del blocco dipende dal backlog correzioni
- Portabilità OCR: risultati Tesseract diversi tra OS se le versioni non sono fissate
- Training oggi legato a macOS/MPS: per retraining futuri su Windows serve strategia (retraining resta su Mac, deploy solo inferenza)
