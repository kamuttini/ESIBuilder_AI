# Revisione snella

Tool web locale per rivedere la pipeline `.fss` cartella per cartella, correggere a mano
qualsiasi studio e far arrivare ogni commento a Claude Code in forma utilizzabile.

Non è un secondo workbench: è un **consumatore sottile** della pipeline ufficiale. Lancia
`run_pipeline_single_folder_safe.py` (che lancia `predict_fss_head_from_acquisitions.py`), legge
i CSV che la pipeline scrive e li mette uno accanto all'altro. Nessuna predizione è ricalcolata
qui: se un numero è in questa pagina, l'ha prodotto un modulo della pipeline.

## Avvio

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/review/app.py --port 8790
```

Poi apri <http://127.0.0.1:8790>. Opzioni utili:

| opzione | a cosa serve |
|---|---|
| `--port` | porta (default 8790) |
| `--inbox` | file JSONL dei feedback (default `feedback/inbox.jsonl`) |
| `--allow-root` | radice extra da cui servire immagini (ripetibile) |
| `--python-bin` | Python con cui lanciare la pipeline (default: il venv `.venv-mps`, cercato anche nel checkout principale se lanci da un worktree) |

## Le tre schermate

**Home.** Storico delle run (una riga per run: stato, cartelle, feedback raccolti) e scelta delle
cartelle da far girare: si incolla la radice, si carica l'elenco, si spuntano le cartelle. Le
opzioni della run stanno in un pannello richiudibile (campionamento vendor/sonda, immagini per la
depth, frame per la scala, salta uno stadio).

**Run in corso.** Una card per cartella con la fila degli stadi che si accendono man mano:
dedup → rotazione → vendor → sonda → rettangolo → orientamento → depth → scala → testa `.fss`.
Ogni stadio mostra già il suo risultato (vendor con confidenza, `#11`, quante depth accettate).
Appena una cartella finisce si apre la sua revisione, mentre le altre continuano.

**Revisione.** Una cartella per volta:

- in testa: la frase della deduplicazione (`2063 immagini, 1863 duplicati esatti rimossi (90.3%)
  → 200 studiate`), le sei caselle di sintesi e le **incoerenze fra aree**;
- *immagine campione*: il frame grande con il `#11` di cartella e il rect di quella immagine
  sovrapposti, e accanto le carte di vendor, sonda, rettangolo, rotazione, dedup, `#13` e testa
  `.fss` con top-3, soglie e sorgente della decisione;
- *tutte le immagini*: la griglia con **depth e scala su ogni miniatura** (più verso su/giù,
  piano L/T, marker L/R), filtrabile per esito o per "solo con problemi";
- clic su una miniatura: la scheda completa dell'immagine, con overlay accendibili (rect, box
  depth, marker, righello), gli zoom sul box della depth e sul righello, e tutte le aree con
  ogni campo che il modulo ha prodotto — compreso il *perché*.

## Commenti e correzioni

Ogni carta ha il pulsante 💬. Si apre un pannello con:

1. **cosa dice il programma** (la predizione esatta, non da ricostruire);
2. **suggerimenti pronti** dell'area, ognuno con il suo tag macchina — è quello che rende un
   commento contabile e raggruppabile (`depth.box_includes_extra` invece di "il box è largo");
3. commento libero, verdetto (`ok / wrong / partial / uncertain`), gravità, e se vale per
   **questa immagine** o **tutta la cartella**;
4. **correzione manuale** con i widget giusti per l'area: rettangolo trascinato sull'immagine per
   rect/`#13`/box depth/marker, clic per la colonna del righello e per zero ed estremo, menu per
   vendor/sonda/verso/piano, numero per la depth. Nessuna coordinata si scrive a mano.

I verdetti `ok` valgono quanto quelli sbagliati: sono le etichette positive del set di
regressione, cioè quello che impedisce a una correzione futura di rompere ciò che già funzionava.

Tutto finisce in `feedback/inbox.jsonl`, una riga per voce, con il contesto compilato dal server
(vendor, confidenze, soglie, checkpoint, sorgenti, aree vicine, percorsi della run).

## Gli incroci fra aree

La parte che nessun modulo può vedere da solo. Per ogni immagine il tool controlla:

- il box della depth cade fuori dall'immagine, o dentro il rettangolo ecografico (sospetto);
- depth e scala descrivono la stessa distanza? (`|y_far - y_zero| × mm_per_px` contro `depth_mm`);
- il verso della scala concorda con il su/giù del marker;
- il rect di questa immagine si discosta dal `#11` di cartella (IoU);
- la depth è un valore che nessun'altra immagine della cartella mostra;
- la scala è `accepted` su evidenza debole (fuori banda, zona di cartella, calibrazione
  geometrica) — il modo di sbagliare che è tornato quattro volte su quattro;
- le controprove che i moduli calcolano già (`zero_check`, `depth_check`), mostrate invece di
  restare sepolte in un CSV.

Ogni incrocio nomina **le due aree che non si accordano**: è il seme dell'orchestratore.

## Orchestratore

`tools/review/orchestrator.py` aggrega tutte le run e tutti i feedback in una conoscenza per
vendor: geometria su cui le cartelle concordano (rect, colonna del righello, intervallo di
`mm_per_px`), strategia effettivamente usata da ogni stadio, punti in cui l'umano ha detto che
era sbagliato (per tag), e quali coppie di aree litigano insieme. Ogni numero porta il suo
supporto.

Da lì escono suggerimenti concreti (banda del righello per vendor, modalità depth attesa, gate
dell'override rosso da rivedere) e la distinzione fra **problema di modulo** (lo stesso tag su
più vendor) e **problema di profilo** (un solo vendor). Dalla scheda Orchestratore si ricalcola
e si salva in `artifacts/72_review_snella/orchestrator/vendor_knowledge.json`.

```bash
python3 tools/review/orchestrator.py     # ricalcola e scrive il file
```

## Il lato di Claude Code

```bash
python3 tools/review/feedback_cli.py triage                    # cosa lavorare, in ordine
python3 tools/review/feedback_cli.py list --area depth --status open
python3 tools/review/feedback_cli.py show fb_20260808_143201_a3
python3 tools/review/feedback_cli.py resolve fb_... --commit abc1234 --note "gate rifatto"
```

Le correzioni tornano nelle code già esistenti dei moduli:

| target | formato | consumato da |
|---|---|---|
| `scale_corrections` | CSV | `tools/scale/ingest_scale_corrections.py` |
| `depth_review` | CSV | `tools/depth/apply_rect_depth_review_export.py` |
| `lr_seeds` | JSON | `--lr-marker-manual-seeds-file` della pipeline |
| `labels` | CSV | generico: etichetta confermata/corretta + percorso, per la preparazione dataset e come set di regressione |

```bash
python3 tools/review/feedback_cli.py export --target scale_corrections --out /tmp/corr.csv
python3 tools/review/feedback_cli.py export --target labels --area vendor --out /tmp/vendor.csv
```

## Dove finiscono i file

```
artifacts/72_review_snella/
  runs/<run_id>/
    run.json            manifest della run (cartelle, opzioni, stato, stadi)
    events.jsonl         tutti gli eventi (la pagina ricaricata rivede la run)
    logs/<slug>.log      stdout della pipeline per cartella
    folders/<slug>/      run dir della pipeline: step_checks.json + pipeline_output/*.csv
  cache/                 miniature
  orchestrator/          vendor_knowledge.json
feedback/inbox.jsonl     i commenti e le correzioni (nel repo, versionato)
```

## Il canale verso la pipeline

L'unica modifica alla pipeline ufficiale è il flag **`--stage-events`**: senza il flag l'output
è identico a prima, con il flag stampa una riga `##STAGE {json}` a fine di ogni stadio di ogni
cartella. Serve perché la pipeline scrive i CSV solo quando l'ultima cartella ha finito: senza
gli eventi la revisione non potrebbe mostrare niente durante la run. Il flag è passato anche da
`run_pipeline_single_folder_safe.py`.

## Note

- Una run alla volta: i modelli vogliono il device tutto per sé.
- Una cartella per processo, di proposito: costa il ricaricamento dei modelli, ma la revisione
  della prima cartella si apre mentre la seconda gira ancora.
- Se lanci il tool da un git worktree, `artifacts/` deve essere raggiungibile (nel worktree di
  sviluppo è un symlink al checkout principale): i modelli e lo storico stanno lì.
