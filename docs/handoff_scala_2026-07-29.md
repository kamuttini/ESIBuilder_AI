# Handoff blocco scala — 2026-07-29

Messaggio da incollare in Claude Code aperto su `~/Documents/Develop/ESIBuilder_AI`.
Il contenuto sotto la riga è il prompt; il resto di questo file è lo stato di dettaglio.

---

Riprendiamo il blocco scala (riga 21 `.fss`). Leggi prima
`docs/scala_strategia_per_vendor_2026-07-29.md` (semantica, strategia, backlog) e
`docs/scala_training_reti_runbook.md` (comandi, bug già corretti, come leggere le metriche).
Il codice è in `tools/scale/`. Non rifare l'audit né i dataset: sono già generati.

**Stato.** La catena è: rete heatmap (quale colonna, dov'è lo zero, verso) → detector
classico a tacche (raffina dentro ±70 px) → OCR dei numeri (calibrazione `mm_per_px`) →
consenso a livello di setup (trend robusto sulle depth, scarta e corregge gli incoerenti).
Reti addestrate sul dataset `artifacts/39_scale_heatmap_dataset_20260729_clean` (fusion,
biopsee e cartelle marcate sbagliate escluse; 4790 righe, 335 cartelle, split leak-free per
cartella). In `artifacts/40_scale_heatmap_models_clean/`: **esaote** e **bk** completi,
**hitachi** ha `best_model.pt` ma non `summary.json` perché il run è morto all'epoca 29.

**Da fare, in ordine:**

1. Recupera le metriche di test di Hitachi dal checkpoint già salvato, con `--eval-only`
   (§4a del runbook). Non riaddestrarlo: all'epoca 29 era il migliore dei tre
   (x_band 100%, y_tol 81%, verso 100%).
2. Addestra i tre vendor mancanti: **ge, mindray, canon**, con `--epochs 15`. BK ed Esaote
   hanno scelto le epoche 7 e 11 su 40, quindi 40 sono tempo buttato.
3. Esegui i **due bracci dell'eval di catena** del §5 del runbook, con e senza
   `--heatmap-models`, sulla GT ristretta `artifacts/41_scale_chain_eval/gt_heatmap_test_only.csv`
   (425 righe / 31 setup: sono le sole righe che le reti non hanno visto in training).
   Confronta `sources.none` fra i due `summary.json`: è la metrica decisiva.

**Attenzione a queste tre cose:**

- I percorsi nei CSV sono assoluti su `/Volumes/SSD_esi1_n1`. Se il volume non è montato
  fallisce tutto con `image_unreadable`.
- Non misurare la catena sulla GT completa: la maggior parte delle cartelle è nel *train*
  delle reti e il risultato sarebbe gonfiato. Usa solo il file ristretto.
- La catena usa Tesseract: fissane la versione prima di confrontare risultati fra macchine
  (`CLAUDE.md` lo segnala già come rischio per il deploy Windows).

**Ipotesi aperta da verificare con l'eval di catena.** Su BK la rete ha `x_band` al 100% ma
`y_tol` solo al 59%: sospetto che i frame dove il detector classico non trova il righello
siano gli stessi dove la `y` della rete è debole, perché falliscono per la stessa causa —
righello grigio scuro con poche tacche, che il resize a 512 px cancella. Se è così, su BK la
rete non recupera i `source=none` e serve un altro approccio (top-hat orizzontale, oppure
raffinamento a risoluzione nativa dentro la banda). Su Esaote invece mi aspetto che funzioni.

---

## Dettaglio dello stato (per riferimento, non serve incollarlo)

### Artifact prodotti

| Cartella | Cosa |
|---|---|
| `37_scale_gt_audit_20260729/` | audit GT: 5290 righe, 98.5% coerenti, profili per vendor |
| `38_scale_ladder_eval_20260729/` | eval pipeline classica: `bk_consensus`, `esaote_consensus`, `multivendor` + gallery |
| `39_scale_heatmap_dataset_20260729/` | dataset completo (5210 righe) |
| `39_..._20260729_clean/` | **usato per il training**: fusion + biopsee + sospette escluse (4790 righe) |
| `39_..._20260729_nofusion/` | variante solo-senza-fusion (per l'A/B) |
| `39_..._FUSION_ONLY/` | diagnostico: 389 righe di sole fusion, tutte in `test` |
| `40_scale_heatmap_models_clean/` | modelli attuali: esaote, bk, hitachi (parziale) |
| `40_..._20260729_PRIMO_RUN_teste_cieche/` | primo run, da non usare (vedi README dentro) |
| `41_scale_chain_eval/` | GT ristretta per l'eval di catena |

### Metriche di riferimento

Pipeline classica con consenso (BK, 95 righe / 6 setup): 45 accepted, verso 45/45,
calibrazione 38/45, `strict_ok` 28/45, errore su `mm_per_px` mediana 0.17% / max 19.8%,
`err_y_zero` mediano 0.5 px. Righe senza predizione: 37% (BK), 55% (Esaote).

Reti (test, dataset clean):

| | Esaote | BK | Hitachi (ep29, val) |
|---|---|---|---|
| `x_band` (±70 px) | 98.6% | 100.0% | 100.0% |
| `y_tol` (≤8 px) | 96.5% | 59.4% | 81.1% |
| verso | 98.2% | 91.6% | 100.0% |
| calib | 5.3% | 5.6% | — |

Baseline classe maggioritaria del verso, per capire se la testa impara davvero:
Esaote val 91.7%, BK val 23.8%.

### Bug corretti in questa sessione, da non reintrodurre

1. **Testa del verso cieca**: leggeva `AdaptiveAvgPool2d(1)`, che dà output identico per
   un'immagine e la sua versione ribaltata. Ora riceve anche un profilo verticale a 16 bin.
2. **Selezione del checkpoint su `strict_ok`**, dominato dalla calibrazione al 2-5% e quindi
   quasi casuale. Ora si usa `handoff_score` (media di `x_band`, `y_tol`, verso).
3. **Flag booleani nei CSV** salvati come `True`/`False`, per cui ogni test `== "1"` a valle
   non scattava: `--drop-fusion` era un no-op.
4. **Percorsi assoluti non portabili**: l'audit ora ha `--canonical-root`.
5. **Una lettura di immagine fallita uccideva il training**: ora ritenta e degrada.

### Fatti sul formato che hanno cambiato l'impostazione

- Riga 21 = `x1|x2|y1|y2|length_mm|tick|side`. Il campo 5 è la **lunghezza del segmento in
  mm**, non la depth. `tick` vale 0.5 (cm) e `side` -1 su tutte e 5290 le righe.
- **`y1` è lo zero**, e nel 21.7% dei casi sta in basso (UD ribaltato). Il vecchio target
  faceva `min`/`max` e perdeva questa informazione su un quinto del dataset.
- **`y2` è arbitrario**: è dove l'operatore ha smesso di trascinare. Non va predetto, va
  generato per convenzione.
- L'unica grandezza oggettiva è `mm_per_px = length_mm / length_px`, uguale a
  `PIXEL_RATIO_Y` entro lo 0.6% su tutte le righe.
- `mm_per_px` varia di un fattore ~4.6 **dentro un setup**, quindi non esiste un valore di
  cartella: l'invariante è il trend (rapporto fra depth adiacenti in [0.85, 1.35] nel 99%).
