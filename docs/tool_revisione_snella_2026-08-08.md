# Tool di revisione snella — impianto e perché

Aggiornato: 2026-08-08. Codice: `tools/review/`. Istruzioni d'uso: `tools/review/README.md`.

## Il problema

Il workbench esistente (`pipeline_workbench_web.py`, 17.6k righe) fa molto e chiede molto: per
capire se una cartella è a posto bisogna aprire tab, incrociare CSV e ricordarsi quale run era.
E i commenti non arrivano da nessuna parte: restano in un JSON per run.

Serviva l'opposto: **una schermata per decidere, una per guardare, e un commento che vale**.

## Le tre scelte di impianto

### 1. Consumatore sottile, non secondo motore

Il tool non ricalcola niente. Lancia `run_pipeline_single_folder_safe.py` e legge quello che la
pipeline scrive: `folder_fss_head_predictions.csv`, i cinque CSV per-immagine (su/giù, marker,
L/T, depth, scala) e `step_checks.json`. Il valore che aggiunge è il **join**: nessuno di quei
file, da solo, permette di giudicare una depth, perché per giudicarla servono il rect in cui è
stata letta, il verso che il righello ha assunto e la calibrazione che la scala ha ricavato —
sulla stessa immagine.

Conseguenza pratica: qualunque miglioramento della pipeline si vede nel tool senza toccare il
tool.

### 2. Una cartella per processo

La pipeline ufficiale scrive i CSV **quando l'ultima cartella ha finito**. Con una run da dieci
cartelle non si vedrebbe nulla per un'ora. Un processo per cartella costa il ricaricamento dei
modelli, ma rende la revisione della prima cartella disponibile mentre la seconda gira.

### 3. `--stage-events`: l'unica modifica alla pipeline

Dentro una cartella la pipeline non stampava quasi niente, quindi la fila di stadi che si accende
non poteva esistere. Aggiunto un flag opt-in che stampa una riga `##STAGE {json}` a fine di ogni
stadio (dedup, rotazione, vendor, sonda, rect, orientamento, depth, scala) più un
`folder_done` con l'intera `FolderPrediction`.

Senza il flag l'output è **identico** a prima: il diff è additivo e la regola zero-regressioni è
soddisfatta per costruzione, come per lo stadio scala. Verificato: con il flag assente
`_emit_stage_event` non stampa nulla e `args.stage_events` è `False` per default.

Gli eventi portano anche i valori, non solo lo stato, ed è per questo che la card della run può
mostrare vendor, sonda e `#11` **su un frame vero** (endpoint `.../sample`, che pesca il frame
di mezzo della cartella) mentre la cartella sta ancora girando.

## Il feedback: perché così meticoloso

Un commento libero non è contabile. "Il box è troppo largo" non si può raggruppare, contare per
vendor, né trasformare in una riga di training. Quindi ogni voce ha:

- **area** (il modulo) e **tag** da un catalogo per area (`areas.py`), con il file che dovrebbe
  cambiare per risolverla — il commento nasce già con un proprietario;
- **verdetto** (`ok / wrong / partial / uncertain`): gli `ok` sono le etichette positive del set
  di regressione, cioè il meccanismo che protegge ciò che già funziona;
- **prediction** (cosa diceva il modulo, verbatim) e **correction** (cosa dovrebbe dire), con le
  coordinate raccolte trascinando o cliccando sull'immagine, mai scritte a mano;
- **context** compilato dal server: vendor, confidenze, soglie, checkpoint, sorgente della
  decisione, incroci di quell'immagine e **aree vicine** (la depth arriva con la sua scala e il
  suo rect allegati);
- **provenance**: run dir, `step_checks.json`, CSV — ogni affermazione è ri-derivabile da disco.

Store: `feedback/inbox.jsonl`, append-only, versionato nel repo. Append-only perché una sessione
di revisione è un log: un "in realtà andava bene" è una voce nuova, e servono entrambe. L'unica
parte riscritta è `status`/`resolution`.

Chiusura del giro: `feedback_cli.py export` porta le correzioni nei formati che le code esistenti
già leggono (`ingest_scale_corrections.py`, `apply_rect_depth_review_export.py`, i manual seeds
del marker), più un CSV generico di etichette per le aree di classificazione.

## Gli incroci: dove nasce l'orchestratore

`study.py:coherence_checks` calcola per ogni immagine i controlli che nessun modulo può fare da
solo, e ognuno **nomina le due aree che non si accordano**:

| controllo | cosa cattura |
|---|---|
| `depth_scala_mismatch` | `|y_far − y_zero| × mm_per_px` contro `depth_mm` oltre il 10% |
| `depth_box_out_of_image` / `depth_box_inside_rect` | box letto fuori dall'immagine o dentro il settore |
| `sugiu_scala_disagree` | il verso del righello contro quello del marker |
| `rect_per_image_far` | IoU del rect di immagine contro il `#11` di cartella |
| `depth_outlier` | valore che nessun'altra immagine della cartella mostra |
| `scala_weak_evidence` | `accepted` con righello fuori banda, zona di cartella, calibrazione geometrica |
| `scala_zero_check` / `scala_depth_check` | le controprove che la scala già calcola, mostrate |

`orchestrator.py` li aggrega su tutte le run insieme ai feedback, per vendor: prior di geometria
(rect, colonna del righello, intervallo `mm_per_px`), strategia effettiva di ogni stadio, punti
debloi per tag, e le **coppie di aree che litigano insieme**. Da qui:

- un prior con supporto ≥8 e dispersione bassa diventa una proposta di banda di ricerca;
- lo stesso tag su più vendor = problema di modulo; su un solo vendor = problema di profilo;
- ≥5 immagini in cui due aree non si accordano = le due aree devono scambiarsi l'evidenza invece
  di decidere da sole.

Ogni suggerimento porta il supporto: non decide niente da solo, propone e l'umano approva.

## Cosa manca / prossimi passi

- Le correzioni non rientrano ancora **automaticamente** nella pipeline: l'export c'è, il
  riaggancio (`--scale-corrections`, `--lr-marker-manual-seeds-file`, retraining) resta manuale.
- Il tool mostra il rect di ogni immagine solo se `step_checks.json` è completo (fine cartella).
- Il campionamento della depth su cartelle molto grandi va tarato: `depth_max_images=0` studia
  tutte le immagini e su 2000 frame costa.
- Un pannello "diff fra due run sulla stessa cartella" sarebbe la verifica zero-regressioni
  dentro il tool, invece che a mano.
