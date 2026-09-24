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

Scala — **ripartita il 2026-07-29 su basi nuove**, vedi `docs/scala_strategia_per_vendor_2026-07-29.md`:
- [x] Audit completo della GT per vendor (`artifacts/37_scale_gt_audit_20260729/`): 5290 righe, **98.5% coerenti**
- [x] Semantica della riga 21 chiarita: `length_mm` è la lunghezza del segmento (non la depth), `y1` è **lo zero** e nel 21.7% dei casi sta in basso
- [x] Detector deterministico tacche + OCR (`tools/scale/detect_scale_ladder.py`) e harness di eval per vendor (`tools/scale/eval_scale_detector.py`)
- [x] Prima eval BK + multivendor con gallery (`artifacts/38_scale_ladder_eval_20260729/`): verso corretto 27/27 sugli accepted BK, `mm_per_px` mediano allo 0.34%
- [x] **Consenso a livello di setup** (`tools/scale/consolidate_scale_setup.py`): voto sul verso, trend robusto Theil-Sen su `mm_per_px`/`x`/`y_zero`, riempimento dei buchi. Chiude il bloccante della policy di confidenza — errore max su `mm_per_px` da **519% a 19.8%**, media da 40.9% a 1.79%, `strict_ok` sugli accepted da 33% a 62%, `err_y_zero` mediano da 28 px a 0.5 px
- [ ] **Copertura del righello — unico collo di bottiglia rimasto**: 35 righe su 95 (BK) e 50 su 91 (Esaote) sono setup interi senza alcun agganciamento
  - [x] Reti heatmap per vendor: dataset a supervisione densa (`artifacts/39_scale_heatmap_dataset_20260729`, 5210 righe, split leak-free, 6 vendor eleggibili), trainer (`tools/scale/train_scale_heatmap.py`), inferenza e aggancio al detector. Target: heatmap su colonna + zero, `log_span_mm`, verso — **non** i 3 scalari che avevano fallito
  - [ ] **Da eseguire sul Mac (serve MPS)**: vedi `docs/scala_training_reti_runbook.md`
  - [ ] Misurare la catena con e senza rete sullo stesso harness: il successo si vede nel crollo di `source = none`
  - [ ] Alternativa/complemento economico: top-hat orizzontale e `min_ticks` per vendor (su BK FlexFocus Template grid le tacche visibili sono 3)
- [ ] Estendere la misura del consenso oltre i 6-8 setup per vendor attuali
- [ ] Ancora sull'unità `cm` nell'OCR — utile ma **non** risolve BK Profocus/FlexFocus, dove l'etichetta accanto allo zero non esiste
- [ ] Layout multi-pannello (fusion/biplana): vincolare il righello al pannello del rect
- [ ] Offset sistematico di `x` per vendor, da misurare dall'eval e mettere nel profilo
- [ ] Criterio di uscita proposto: per vendor con ≥ 200 righe, `strict_ok` sugli accepted ≥ 95% e copertura accepted ≥ 70%
- [ ] Verifica indipendente su `SSD_esi1_n3` (`ACQUISITION ELABORATION`), senza GT, con gallery
- [x] ~~Smaltire coda correzioni GT P0 (34) e P1 (49)~~ — **derubricato**: l'audit mostra che la GT è al 98.5% coerente, il ritardo era nel modello
- [ ] ~~Retraining regressori `32_`/`34_`/`36_`~~ — **abbandonato**: metà del loro target (`y_bottom`) è rumore di etichettatura umana

Orientation:
- [x] Run completo su 272 cartelle SSD_esi1_n3 + gallery di review (2026-07-15)
- [x] Detector migliorato dalle correzioni umane: gate anti-nero, multi-scala, template pinnati, 51 template raccolti — 71→97/112 box, zero regressioni (2026-07-22, vedi `docs/orientation_marker_processo_2026-07-09.md`)
- [x] Integrazione nella pipeline `fss_head` dopo lo SU/GIU (`--lr-marker-scales`, `--lr-marker-pinned-templates`) — smoke test ok (2026-07-22)
- [ ] Vendor mancanti nella banca: Sonostar, ExactVu/Edap (2 cartelle su 272)
- [ ] Casi immagini ruotate (es. 234.BK Florida): serve rotazione OSD, non applicata dai runner marker
- [ ] Riscrivere il trainer del detector template orientamento (approccio a rete, alternativa al template-matching attuale)
- [ ] Definire target IoU minimo di produzione e misurarlo su eval routed

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

- **Rete L/T: sulle acquisizioni mai viste sbaglia le T** (misura del 2026-09-24,
  `artifacts/40_outputs_eval/lt_net_by_acquisition_20260924/`, 51 acquisizioni con L/ e T/ separate).
  Sulle acquisizioni viste in training e' quasi perfetta (L 100%, T 99%), ed e' per questo che
  sembrava funzionare benissimo; sulle 37 mai viste le L restano al 100% ma le T scendono all'83%, e
  solo 18 cartelle su 37 sono perfette. L'errore e' sempre lo stesso, T detta L: su Bologna X8,
  Koelis e Biopsee quasi nessuna T e' riconosciuta. Conseguenza nell'app: il lavoro sulle sole L non
  parte se la rete non trova T, e le T sbagliate entrano fra le L al lavoro finche' non si correggono
  a mano. Anche lo split del training non era per acquisizione (12 acquisizioni con la L e la T in
  split diversi). Rimedio: riaddestrare con split per acquisizione, aggiungendo le acquisizioni che
  hanno gia' L e T separate; fino ad allora la lista dei piani nell'import va guardata.

- Orientation, casi limite legacy (da pensarci per gli sviluppi futuri): alcuni ecografi hanno template di orientamento ribaltati/speculari che rompevano il match del vecchio software — quei progetti sono stati configurati per un solo orientamento (spesso UD) e la loro riga #16 contiene 4 copie dello stesso box (non NF/LR/UD/LRUD). Il nuovo detector dovrà gestirli esplicitamente. Inoltre il template può cambiare dimensione tra immagini della stessa cartella (serve matching multi-scala).
- Probe: classi rare/unseen con recall basso — mitigare con policy review + OCR refine
- Orientation: trainer da rifare, effort incerto
- Dipendenza da GT scala: la qualità del blocco dipende dal backlog correzioni
- Portabilità OCR: risultati Tesseract diversi tra OS se le versioni non sono fissate
- Training oggi legato a macOS/MPS: per retraining futuri su Windows serve strategia (retraining resta su Mac, deploy solo inferenza)
