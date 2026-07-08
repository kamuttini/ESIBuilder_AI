# Proposta: layer di feedback universale nel workbench

Aggiornata: 2026-07-08. Obiettivo: poter commentare **qualsiasi cosa** dal workbench — risultati dei modelli, algoritmi, e l'interfaccia stessa — e far arrivare ogni commento a Claude Code in forma strutturata, così che diventi il canale di comunicazione asincrono tra Camilla e Claude.

## Cosa esiste già (da riusare, non rifare)

- `pipeline_workbench_web.py` (Flask, porta 8780): launcher run, monitor con albero decisionale live, tab **Analisi Step** con `flag / commento / correzione proposta` per step, gallerie Evidenze, storia run.
- Persistenza: `review_annotations.json` per run (schema: `run_flag`, `run_note`, `steps{flag,comment,correction}`, `lr_marker_reviews`) + endpoint `POST /api/runs/<id>/annotations` (:17505).
- `pipeline_ux_compare_tool.py`: store feedback esterno per run (`artifacts/70_ux_feedback/...`) + verdetto `Bloccata/Review/Pronta` — il pattern più vicino a quello che serve.
- `lr_marker_sugiu_review_web.py`: modello pulito accetta/correggi/scarta + nota → CSV.

## I 4 gap

1. Il feedback è **sparso** (un JSON per run, un altro store per il compare tool, localStorage nel viewer rect) — nessuna vista unica.
2. **Nessun canale verso Claude**: i commenti non arrivano mai a chi sviluppa; l'export è un download manuale JSON.
3. **Non si può commentare l'interfaccia** né un'immagine specifica di una galleria — solo run e step.
4. Il commento **non cattura il contesto**: per capirlo bisogna riaprire la run e ricostruire a mano soglie, confidenze, immagine.

## Design proposto

### A. Inbox unico append-only: `feedback/inbox.jsonl`

Nel repo (committato, così Claude Code lo vede in git), una riga JSON per commento:

```json
{
  "id": "fb_20260708_143201_a3",
  "ts": "2026-07-08T14:32:01",
  "type": "result | model | ui | bug | idea",
  "severity": "minor | major | blocking",
  "status": "open | ack | done | wontfix",
  "comment": "il box depth ingloba la B di BG anche qui",
  "target": {
    "run_id": "...", "folder": "...", "step": "rect_depth",
    "image": "path/relativo.png", "ui_element": null
  },
  "context": {
    "line_values": {"line_11": "..."}, "confidences": {"vendor": 0.92},
    "sources": {"line_11_source": "global"}, "thresholds_effective": {...}
  },
  "screenshot": "feedback/shots/fb_20260708_143201_a3.png",
  "resolution": {"by": null, "commit": null, "note": null}
}
```

`context` è compilato **automaticamente** dal server leggendo `step_checks.json` della run: Camilla scrive solo il commento, il resto viaggia da solo.

### B. UI: commentare ovunque con un click

1. **Bottone 💬 su ogni cosa**: ogni card step, ogni immagine di galleria, ogni run card riceve un'icona commento (si aggancia con un attributo `data-feedback-key` sugli elementi esistenti — iniezione leggera nel template, niente riscrittura del monolite).
2. **Modalità "Feedback UI"** (toggle in alto): attivata, il click su qualunque elemento della pagina apre il box commento con selettore/nome elemento e screenshot auto-catturati → entry `type=ui`. È così che commenti l'interfaccia stessa.
3. **Hotkey nelle gallerie**: mentre scorri le evidenze, `G`=ok, `B`=sbagliato, `N`=nota rapida sull'immagine corrente. Micro-feedback a costo quasi zero che diventa weak-label per il retraining.
4. **Badge inbox** nell'header: contatore commenti `open`, click → pannello inbox filtrabile (tipo/step/run/status).

### C. Canale verso Claude: il protocollo

1. **Convenzione in CLAUDE.md**: a inizio sessione Claude Code legge `feedback/inbox.jsonl` filtrando `status=open`, propone un piano di triage, e quando risolve un item lo marca `done` con hash del commit in `resolution.commit`.
2. **CLI di supporto** `tools/feedback/feedback_cli.py`:
   - `list --status open --type model` — triage rapido
   - `show fb_...` — entry completa con contesto
   - `resolve fb_... --commit abc123 --note "..."`
   - `export --target scale_gt|lr_seeds|review_csv` — converte i feedback `type=result` con correzione nei formati delle code GT esistenti (coda fix scala, manual seeds LR, review CSV): **è il pezzo che chiude il loop feedback → retraining**.
3. **Migrazione soft**: gli attuali `review_annotations.json` restano (il tab Analisi Step continua a funzionare); al salvataggio il server duplica le novità come entry nell'inbox. Nessuna rottura.

### D. Cosa NON fare adesso

- **Non riscrivere il workbench** (17.6k righe monolitiche): il layer si inietta nel template esistente. La riscrittura pulita è la Fase 4 della roadmap (app ESIBuilder vera), e a quel punto l'inbox sarà già il requisito di design centrale.
- Non serve database: JSONL append-only + lock, coerente con lo stile del progetto.

## Piano di implementazione (per Claude Code)

| Step | Contenuto | Stima |
|---|---|---|
| S1 | Modulo `tools/feedback/store.py` (schema, append, lock) + `feedback_cli.py` list/show/resolve | 0.5 g |
| S2 | Endpoint Flask `POST /api/feedback` + auto-context da step_checks.json | 0.5 g |
| S3 | Iniezione UI: bottone 💬 su step card / immagini gallerie / run card + pannello inbox + badge | 1-2 g |
| S4 | Modalità Feedback UI (click su elemento + screenshot html2canvas) | 0.5-1 g |
| S5 | Hotkey gallerie G/B/N | 0.5 g |
| S6 | `export --target ...` verso code GT + aggiornamento CLAUDE.md con protocollo di triage | 1 g |

Ordine consigliato: S1+S2+S3 danno subito il valore principale (commentare risultati e farli arrivare a Claude); S4-S6 dopo.

## Criterio di successo

Camilla naviga una run, vede un box depth sbagliato, preme 💬, scrive una riga, chiude. Alla sessione successiva Claude Code apre l'inbox, ha già run, immagine, soglie e confidenze nel contesto, corregge, marca `done` con il commit. Zero copia-incolla, zero "quale run era?".
