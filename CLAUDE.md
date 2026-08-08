# ESIBuilder AI

Automazione della configurazione ecografi per ESI: riconoscimento automatico da immagini di acquisizione (vendor, sonda, rettangolo ecografico, depth/scala, orientamento, piano L/T) per generare file `.fss` compatibili con il sistema legacy. Obiettivo finale: nuova app ESIBuilder dove l'AI propone e l'utente conferma/corregge. Utenti finali: colleghi interni su **Windows**.

## Struttura

- `tools/` — codice sorgente (109 script Python)
  - `tools/ultrasound/` — dataset/training/inferenza vendor, probe, rect + pipeline head `.fss`
  - `tools/fss/` — compatibilità/audit `.fss` (`compare_fss.py`), pipeline scala
  - `tools/depth/` — modulo RECT_DEPTH autonomo (entrypoint: `predict_rect_depth_autonomous.py`)
  - `tools/orientation/` — orientamento simbolico, GUI review
  - `tools/line16/` — riga #16 (template rect + parametri)
  - `tools/scale/` — blocco scala (#18-#21), stadio della pipeline
  - `tools/review/` — tool di revisione snella (web): run, correzioni, feedback, orchestratore
  - `tools/review_html/` — gallerie HTML di revisione
  - Vedi `tools/README.md` per i comandi completi di ogni blocco
- `feedback/inbox.jsonl` — commenti e correzioni umane dal tool di revisione (versionato)
- `artifacts/` — dataset, modelli, run, eval (~31GB, NON committati, naming `NN_categoria`)
- `docs/` — report di stato e checkpoint (in italiano)
- `OldSoftwareEsiBuilder/` — vecchio ESIBuilder Qt/C++ (riferimento per formato `.fss` e workflow legacy)
- `preparazione al progetto/` — requisiti e brainstorming iniziali
- `encoding_struct 2026 02 23.xlsx` — anagrafica ecografi/sonde/codici

## Ambiente

- Python per training/inferenza: `OldSoftwareEsiBuilder/.venv-mps/bin/python` (macOS, device `mps`)
- OCR: Tesseract locale (versione da fissare per parità macOS/Windows)
- Dataset raw su volumi esterni: `/Volumes/SSD_esi1_n1`, `/Volumes/SSD_esi1_n3`
- Review HTML servite in locale (es. `http://127.0.0.1:8765/`)

## Stato blocchi (luglio 2026)

| Blocco | Stato | Metriche chiave |
|---|---|---|
| Vendor | Produzione | test acc 0.9824, macro-F1 0.9738; 414/426 cartelle ok |
| Probe | Operativo con review | test acc 0.8477; OCR refine sui review |
| Rect | Pronto produzione | mean IoU test 0.9174 |
| Rect_depth | Modulo autonomo pronto, da integrare | checkpoint `docs/rect_depth_autonomous_checkpoint_2026-07-02.md` |
| Scala | Workflow pronto, backlog GT da smaltire | coda P0=34, P1=49 |
| Orientation | NON a livello produzione | trainer da riscrivere (rimosso) |
| Piano L/T | Dataset tools pronti, rete da consolidare | priorità alta per il collega |
| Pipeline fss_head | Integrata (vendor+probe+rect+depth+scala), smoke ok | manca validazione batch reale |

## Feedback umano (tool di revisione)

`tools/review/app.py` è il tool con cui Camilla rivede le run e corregge. Ogni suo commento o
correzione diventa una riga in `feedback/inbox.jsonl` con area, tag, verdetto, predizione,
correzione, contesto (confidenze, soglie, checkpoint) e provenienza.

**A inizio sessione, se l'inbox non è vuota:**

```bash
python3 tools/review/feedback_cli.py triage          # cosa lavorare, in ordine, per area
python3 tools/review/feedback_cli.py show fb_...     # una voce completa
```

Quando una voce è risolta, marcarla con il commit:

```bash
python3 tools/review/feedback_cli.py resolve fb_... --commit <hash> --note "cosa è cambiato"
```

Le correzioni si portano nelle code dei moduli con `feedback_cli.py export --target
{scale_corrections,depth_review,lr_seeds,labels}`. I verdetti `ok` sono il set di regressione:
servono a dimostrare zero regressioni, non sono rumore.

## Convenzioni

- Documenti e report in italiano; codice/commenti in inglese
- Ogni run produce artifact in `artifacts/NN_*/nome_run/` con `summary.json`
- Policy confidenza: `accepted` automatici, `review` manuali, `reject` scartati — il sistema non forza mai una predizione incerta
- Split dataset leak-free a livello cartella/gruppo (mai immagini della stessa acquisizione in split diversi)
- Prima di modifiche grosse ai moduli: smoke test (vedi `artifacts/50_smoke_tests/`)
- Quality gate `.fss`: `python3 tools/fss/compare_fss.py legacy.fss nuovo.fss` (exit 0 = compatibile)

## Roadmap

Vedi `ROADMAP.md` per fasi e priorità correnti.
