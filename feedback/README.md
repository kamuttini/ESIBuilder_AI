# Feedback della revisione

`inbox.jsonl` — una riga JSON per commento o correzione, scritta dal tool
`tools/review/app.py` e letta da Claude Code con `tools/review/feedback_cli.py`.
Il file nasce alla prima voce salvata; se non c'è, non c'è ancora feedback.

Append-only: una sessione di revisione è un log, non uno stato. Un "in realtà andava bene"
è una voce nuova, e servono entrambe. L'unica parte riscritta è `status`/`resolution`.

Campi di una voce:

| campo | contenuto |
|---|---|
| `id`, `ts`, `author` | identità della voce |
| `kind` | `comment` / `correction` / `confirm` / `flag` |
| `area` | il modulo (`vendor`, `probe`, `rect`, `depth`, `scala`, `su_giu`, `lr_marker`, `lt`, `line13`, `dedup`, `rotazione`, `fss`, `orchestratore`, `ui`) |
| `scope` | `folder` / `image` / `run` |
| `tag` | voce del catalogo di `tools/review/areas.py` (es. `depth.box_includes_extra`) |
| `verdict` | `ok` / `wrong` / `partial` / `uncertain` — gli `ok` sono il set di regressione |
| `severity` | `info` / `minor` / `major` / `blocking` |
| `status` | `open` / `ack` / `done` / `wontfix` |
| `comment` | testo libero |
| `target` | run, cartella, immagine |
| `prediction` | cosa diceva il modulo, verbatim |
| `correction` | cosa dovrebbe dire (coordinate in pixel dell'immagine originale) |
| `context` | vendor, confidenze, soglie, opzioni della run, aree vicine, percorsi originali |
| `provenance` | run dir, `step_checks.json`, CSV: ogni affermazione è ri-derivabile |
| `resolution` | chi, quale commit, quale nota |

Triage: `python3 tools/review/feedback_cli.py triage`
