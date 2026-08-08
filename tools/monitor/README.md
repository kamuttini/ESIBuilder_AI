# Monitor stato progetto — pipeline `.fss`

Cruscotto HTML che raccoglie automaticamente le metriche sparse in `artifacts/` e le
organizza in **uno spazio dedicato per ogni step** della pipeline di generazione del
`.fss`, con il dettaglio **per vendor** e l'**andamento nel tempo** sulle run di
raffinamento.

Output: un unico file `artifacts/71_monitor/index.html`, autocontenuto (niente CDN,
niente server obbligatorio) — si apre anche con doppio click su Windows.

## Uso

```bash
python3 tools/monitor/monitor.py all
```

Da lanciare dopo ogni training o eval: rilegge `artifacts/`, aggiorna lo store e
riscrive la dashboard. Comandi separati:

```bash
python3 tools/monitor/monitor.py collect
```

```bash
python3 tools/monitor/monitor.py build
```

```bash
python3 tools/monitor/monitor.py serve --open
```

`serve` espone la dashboard su `http://127.0.0.1:8765/index.html` (stesso pattern
delle gallerie di review). Opzioni globali: `--artifacts-root`, `--store`.

## Cosa vedi

- **Panoramica** — i 16 spazi in ordine di pipeline (`∑` end-to-end + step 01→15),
  ciascuno con stato del blocco, metrica principale, confronto con l'obiettivo,
  delta rispetto alla run precedente e sparkline.
- **Spazio per step** — descrizione della logica e riferimento `file:riga`, tile con
  ultimo valore / delta ultima run / delta dalla prima run / migliore di sempre,
  grafico dell'andamento sulle run di raffinamento (con linea dell'obiettivo),
  small multiples e tabella per vendor, motivi di review osservati, elenco delle run.
- **Confronto vendor** — matrice vendor × step con l'ultimo valore della metrica
  principale, confrontato con l'obiettivo dello step.
- **Storico run** — tutte le run tracciate con data, categoria, tipo e step misurati.

L'asse x dei grafici è **ordinale sulla sequenza di run**, non temporale: molte run
di raffinamento condividono la stessa data e quello che conta è l'ordine dei
tentativi. Data ed `n` compaiono nel tooltip e nelle tabelle — controllali sempre
prima di leggere un miglioramento, perché run diverse coprono insiemi diversi di
cartelle.

Le run il cui nome contiene `smoke`, `tmp_`, `_test`, `debug`, `sanity`, `_check`
sono marcate come smoke ed **escluse di default**: la casella "includi run
smoke/tmp" in alto le rimette dentro.

## File

| File | Ruolo |
|---|---|
| `pipeline_registry.py` | Registro degli step, delle metriche e delle review reason. **È qui che si mette mano** per cambiare obiettivi, stato di un blocco o metrica principale. |
| `collect_metrics.py` | Scansione di `artifacts/` e normalizzazione: un adapter per ogni formato di risultato presente nel repo. |
| `build_dashboard.py` | Inietta lo store dentro il template e scrive l'HTML finale. |
| `dashboard_template.html` | Struttura, stile e rendering dei grafici (SVG scritto a mano, nessuna dipendenza). |
| `monitor.py` | Entrypoint CLI (`collect` / `build` / `all` / `serve`). |

Lo store intermedio finisce in `artifacts/71_monitor/`:
`metrics.jsonl` (una riga per misura), `runs.jsonl`, `review_reasons.jsonl`,
`collect_report.json` (conteggi + warning, per esempio review reason non mappate).

## Formati riconosciuti

Il collector cerca solo file con nomi noti e non scende nelle cartelle di immagini,
quindi la scansione dura meno di un secondo anche su `artifacts/` da ~31 GB.

| Sorgente | Da dove | Cosa produce |
|---|---|---|
| `metrics.json` con `per_manufacturer_iou` | training rect | `mean_iou` complessivo e per vendor |
| `metrics.json` / `test_metrics_from_best.json` con `per_class` | training vendor / probe / L-T | `accuracy`, `macro_f1`, `f1`/`precision`/`recall` per vendor |
| `metrics.json` con `per_manufacturer_test` | training L/T | `accuracy` e `macro_f1` per vendor |
| `metrics.json` con `acc_mm` | training parametri riga #16 | `accuracy` |
| `summary.json` con `metrics.by_vendor` | eval scala/ladder | `strict_ok_pct`, `calib_ok_pct`, `direction_ok_pct`, `accepted_pct`, errori mediani |
| `summary.json` con `handoff_score` | modelli heatmap scala | metriche di test per vendor |
| `summary.json` con `group_accuracy` | eval orientamento/line16 | `group_accuracy`, `box_agree_rate`, `review_rate`, IoU envelope |
| `folder_summary.csv` | run orientamento recenti | `review_rate`, `marker_found_rate`, `groups_complete_rate` per vendor |
| `summary.json` + `folder_fss_head_predictions.csv` | run pipeline `fss_head` | `folder_ok_rate` e, per **ogni** step, la quota di cartelle che lo attraversano senza review reason |

Le run dedicate a un solo vendor (cartella `vendor_models/mindray`, `.../bk`, oppure
una breakdown con un solo vendor) **non** alimentano la serie complessiva: il loro
valore viene attribuito a quel vendor, altrimenti un modello mono-vendor
sembrerebbe una regressione del modello globale.

## Come estenderlo

- **Cambiare un obiettivo o lo stato di un blocco**: `PIPELINE_STEPS` in
  `pipeline_registry.py` (`target`, `status`, `primary_metric`).
- **Nuova review reason**: aggiungila alla lista `review_reasons` dello step giusto;
  quelle non mappate finiscono nei warning di `collect_report.json`.
- **Nuovo formato di risultato**: un metodo `adapt_*` in `collect_metrics.py` più una
  riga in `handle_json` / `handle_csv`. Aggiungi il nome file a `INTERESTING_FILES`.
- **Assegnare una run a uno step diverso**: file JSON passato con `--overrides`, con
  chiavi `step:<run_id>` → `step_id`.
