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
| `anagrafica.py` | lettura/scrittura di `encoding_struct*.xlsx`: ID da modelli, candidati per sonda, nuova riga con backup |
| `inference.py` | vendor, sonda, rettangolo (routing per vendor), template #13, piano L/T e su/giu dai checkpoint della pipeline attiva |
| `stages.py` | marker di orientamento, depth e scala come sottoprocessi con `pipeline_context.json`, piu' i parser dalle righe del modulo al valore dello step |
| `probe_shape.py` | misura geometrica lineare/convex — **non agganciata**, i numeri misurati sono nel docstring |
| `server.py` | API Flask + servizio della UI |
| `static/` | wizard (vanilla JS, nessuna build) |
| `static/box_editor.js` | editor dei box: trascinamento, maniglie, slider, schermo intero con scorrimento delle immagini |
| `static/orientation_view.js` | visualizzatore orientamento: i 4 envelope, il marker per immagine, il ritaglio, filtri per gruppo, correzione con click |
| `marker_refine.py` | dal click approssimativo alla posizione precisa (match del bundle in una finestra) + verifica del gruppo + ricalcolo degli envelope |
| `orientation_marker.py` | il marker della cartella: innesco dalla banca, taglio del ritaglio, scelta per copertura, validazione dentro gli envelope |
| `selftest_roundtrip.py` | controllo: rigenera un `.fss` legacy e verifica che sia identico |

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

`#14 RECT_NAME_PROBE` resta vuoto: il resolver storico legge i `.fss` legacy indicati in
`references/.../manifest_rect_echo.csv`, che puntano a `ESIBuilder_AI/Dataset/` — cartella oggi
assente (0 su 426 percorsi risolve).

Non ancora agganciati: soglie e linee guida — quegli step accettano il valore in JSON, nella
forma che il writer si aspetta. Orientamento, depth e scala girano dai rispettivi moduli
(bottone "Calcola con i moduli" nei due step), con gli artefatti in `<progetto>/stages/`. Il rettangolo `#11` e il template `#13` si correggono trascinando sull'immagine (anche a schermo
intero, scorrendo la cartella); il tracciamento a mano da zero degli altri box non c'e' ancora.

I checkpoint vengono da `artifacts/10_active_pipeline/pipeline_fss_head/models` (e
`artifacts/30_models` per il piano L/T). In un worktree git quella cartella sta nel checkout
principale: il server la trova da solo, oppure si passa `--models-root`.
