# App ESIBuilder AI — versione base

Wizard locale che porta da una cartella di acquisizioni al file `.fss`. Struttura e flusso:
`docs/specifiche_app_esibuilder_ai_2026-08-26.md`.

## Avvio

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/app/server.py --port 8800
```

Poi apri `http://127.0.0.1:8800/`. I progetti finiscono in `artifacts/80_app_projects/<progetto>/`
(`--projects-root` per cambiare cartella).

## Cosa c'e' dentro

| File | Ruolo |
|---|---|
| `constants.py` | costanti legacy (probe type, video input, gruppi orientamento, limite rect ESI) e default di match |
| `fss_writer.py` | serializzatore delle 23/26 righe `.fss` + validazione |
| `project.py` | `project.json`, registro degli 11 step, grafo delle dipendenze e invalidazione |
| `importer.py` | scansione cartella, dedup esatta, metadati video dai nomi file, proposta di resize |
| `rotation.py` | rotazione OSD con tesseract (>=2 voti, >=60% di supporto), 4 processi in parallelo |
| `timestamp_detection.py` | proposta OCR dell'area data/ora su 8 immagini, con supporto spaziale e confidenza |
| `anagrafica.py` | lettura/scrittura di `encoding_struct*.xlsx`: ID da modelli, candidati per sonda, nuova riga con backup |
| `inference.py` | vendor, sonda, rettangolo (routing per vendor), template #13, piano L/T e su/giu dai checkpoint della pipeline attiva |
| `stages.py` | marker di orientamento, depth e scala come sottoprocessi con `pipeline_context.json`, piu' i parser dalle righe del modulo al valore dello step |
| `probe_shape.py` | misura geometrica lineare/convex — **non agganciata**, i numeri misurati sono nel docstring |
| `server.py` | API Flask + servizio della UI |
| — | il materiale per le linee guida si cerca dall'import: `POST /api/projects/<id>/import/needle` |
| `static/` | wizard (vanilla JS, nessuna build) |
| `static/box_editor.js` | editor dei box: trascinamento, maniglie, slider, schermo intero con scorrimento delle immagini |
| `static/orientation_view.js` | visualizzatore orientamento: i 4 envelope, il marker per immagine, il ritaglio, filtri per gruppo, correzione con click |
| `marker_refine.py` | dal click approssimativo alla posizione precisa (match del bundle in una finestra) + verifica del gruppo + ricalcolo degli envelope |
| `orientation_marker.py` | il marker della cartella: innesco dalla banca, taglio del ritaglio, scelta per copertura, validazione dentro gli envelope |
| `selftest_roundtrip.py` | controllo: rigenera un `.fss` legacy e verifica che sia identico |

## Materiale per le linee guida (dallo step Import e analisi)

Nel pannello dell'import, la piega **Materiale per le linee guida** cerca nelle sottocartelle
del progetto i fotogrammi che servono allo step `guides` (`PAGE_CALIBRATION`, righe #22-#23).

Nel vecchio ESIBuilder quella sessione (`WdgPageCalibration`) chiede **una immagine per ogni
angolo** del kit — si rifiuta di proseguire se il numero non torna — e la richiede da capo per
ogni profondita' e per ogni flip. L'operatore ricalca l'ago che vede nel fotogramma, e da quel
tratto escono angolo e distanza dal centro. Per questo le acquisizioni sono fatte in acqua:
serve vedere l'ago nitido. Qui si risponde solo alla domanda *dove sono*, non *quale usare per
quale angolo*: quella e' la scelta dello step delle linee guida.

Esempio su un progetto con tre sottocartelle (30 fotogrammi campionati, pochi secondi):

```
calibrazione   0.978   10/12 img   VERIFICA AGHI
no             0.035   10/10 img   PROIBITE
no             0.018   10/124 img  DEPTH
```

Il verdetto e' la media dei punteggi della sottocartella, non quello di un singolo fotogramma,
quindi usa la banda di `folder_policy.json` accanto al checkpoint (accetta >= 0.60, rifiuta
< 0.25) e non la soglia per-immagine del `metrics.json`, che mira al 98% di precision su un
fotogramma solo. Misurata a precision 0.94 / recall 0.94 per cartella sul test.

Il ritaglio usa `_needle_tensor`, non `_crop_tensor`: quest'ultimo schiaccia il ritaglio in un
quadrato, e schiacciare cambia l'inclinazione degli aghi, che e' il segno da riconoscere.

Il risultato si salva in `needle_scan` dentro lo step `import` **senza invalidare** nulla a
valle: e' una informazione in piu' sulla cartella, non un dato geometrico.

Il checkpoint sta in `artifacts/91_needle_models/`, fuori dalla pipeline attiva: se manca, la
piega mostra l'errore e il resto dell'app funziona come prima. Il blocco che lo addestra e'
`tools/needle/` (preparazione dataset, ritagli, training, valutazione, report HTML).

**Limite noto:** il modello copre la sessione *a linee* (ago in acqua) e manca quella *a
griglia* (`WdgPageGridCalibration`, cartelle `GUIDA AGHI` / `GUIDA BIOPSIA`), che classifica
come negativa.

## Controllo del writer

```bash
python3 tools/app/selftest_roundtrip.py OldSoftwareEsiBuilder/templates/DB_setup/*.fss
```

Deve stampare `tutti i N file legacy rigenerati identici`: e' la prova che separatori e formati
numerici combaciano con il dialetto legacy.

## Stato

Reali in questa versione: passo zero (dedup, rotazione, vendor, sonda, rettangolo, piano L/T) come
job unico con avanzamento; pagina codici precompilata con la provenienza di ogni campo e selettore
del modello macchina quando l'anagrafica e' ambigua; rettangolo proposto con anteprima e box
disegnato; stato per step con invalidazione mirata; writer `.fss` con validazione; quality gate via
`tools/fss/compare_fss.py`.

La catena lavora in due tempi: l'import si ferma prima dei moduli avanzati; dopo aver controllato
e, se necessario, separato L e T, il comando del singolo progetto lancia orientamento, depth e
scala. Il server impedisce di lanciarli su una cartella biplana non classificata o ancora mista.
Durante l'import data e ora vengono cercate automaticamente dopo la rotazione: una proposta
affidabile entra subito nella deduplicazione, ma il riquadro rosa resta visibile e correggibile.
La ricerca puo' essere rilanciata o disattivata esplicitamente per il singolo progetto.

`#14 RECT_NAME_PROBE` resta vuoto: il resolver storico legge i `.fss` legacy indicati in
`references/.../manifest_rect_echo.csv`, che puntano a `ESIBuilder_AI/Dataset/` — cartella oggi
assente (0 su 426 percorsi risolve).

Non ancora agganciati: soglie e linee guida — quegli step accettano il valore in JSON, nella
forma che il writer si aspetta. Orientamento, depth e scala girano dai rispettivi moduli
(un bottone per modulo: "Calcola l'orientamento", "Calcola la depth", "Calcola la scala"), con gli artefatti in `<progetto>/stages/`. Il rettangolo `#11` e il template `#13` si correggono trascinando sull'immagine (anche a schermo
intero, scorrendo la cartella); il tracciamento a mano da zero degli altri box non c'e' ancora.

I checkpoint vengono da `artifacts/10_active_pipeline/pipeline_fss_head/models` (e
`artifacts/30_models` per il piano L/T). In un worktree git quella cartella sta nel checkout
principale: il server la trova da solo, oppure si passa `--models-root`.
