# Runbook — training delle reti scala per vendor

Da eseguire **sul Mac**, perché serve MPS: l'ambiente in cui lavora Claude è un sandbox
Linux separato e non vede la GPU Apple. Tutto il codice è già scritto e la pipeline dati è
già validata (vedi in fondo "cosa è già stato verificato").

Interprete: `OldSoftwareEsiBuilder/.venv-mps/bin/python`.
Da eseguire dalla radice del repo (`~/Documents/Develop/ESIBuilder_AI`).

---

## 0. Prerequisito: i volumi SSD montati

Il dataset punta alle immagini su `/Volumes/SSD_esi1_n1`. Se il volume non è montato il
training si fermerà subito con immagini illeggibili.

```bash
ls /Volumes/SSD_esi1_n1 > /dev/null && echo "volume ok"
```

## 1. (già fatto, rieseguire solo se cambia la GT)

```bash
python3 tools/scale/audit_scale_gt.py \
  --root /Volumes/SSD_esi1_n1 \
  --output-dir artifacts/37_scale_gt_audit_20260729

python3 tools/scale/prepare_scale_heatmap_dataset.py \
  --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
  --output-dir artifacts/39_scale_heatmap_dataset_20260729
```

**I percorsi nei CSV sono assoluti**, quindi chi genera l'audit da un mount diverso deve
riscriverli in forma canonica, altrimenti il dataset è inutilizzabile su altre macchine:

```bash
python3 tools/scale/audit_scale_gt.py \
  --root /altro/mount/SSD_esi1_n1 \
  --canonical-root /Volumes/SSD_esi1_n1 \
  --output-dir artifacts/37_scale_gt_audit_20260729
```

I manifest attuali sono già in forma `/Volumes/SSD_esi1_n1/...`.

Il dataset attuale: 5210 righe, 384 cartelle, split leak-free per cartella.
Vendor con abbastanza dati per un modello proprio:

| Vendor | Righe | Cartelle | train / val / test |
|---|---|---|---|
| Esaote | 1727 | 134 | 1184 / 254 / 289 |
| BK | 1284 | 95 | 961 / 168 / 155 |
| Hitachi | 917 | 63 | 673 / 98 / 146 |
| GE | 515 | 34 | 416 / 68 / 31 |
| Mindray | 241 | 15 | 164 / 41 / 36 |
| Canon | 228 | 11 | 182 / 25 / 21 |

Gli altri 8 vendor (67 righe o meno) non hanno un modello proprio: useranno il modello
registrato come `default`, oppure la sola pipeline classica.

## 2. Controllo rapido prima di lanciare (non serve torch)

Tutti e 6 i vendor in un colpo:

```bash
python3 tools/scale/train_scale_heatmap.py --check-data \
  --dataset-dir artifacts/39_scale_heatmap_dataset_20260729
```

Deve chiudere con `NESSUN PROBLEMA`. Per un singolo vendor, in alternativa:

```bash
python3 tools/scale/train_scale_heatmap.py --check-data \
  --manifest artifacts/39_scale_heatmap_dataset_20260729/manifests/manifest_scale_heatmap_bk.csv
```

Se compare `frame_size_mismatch` significa che la risoluzione dichiarata nel `.fss` non
corrisponde al file immagine, e **tutti i target di quelle righe sarebbero sbagliati**:
fermarsi e segnalarlo. Gli altri controlli sono `image_unreadable` (volume SSD non
montato, o percorsi da rimappare con `--path-remap /Volumes/=/altro/percorso/`),
`x_quantisation_over_tol` e i round-trip di `mm_per_px` e del verso.

## 3. Un vendor solo, per vedere se il metodo regge

Parti da BK: ha il comportamento peggiore per la pipeline classica (righelli grigio scuro,
spesso senza etichette), quindi è il test più severo.

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --manifest artifacts/39_scale_heatmap_dataset_20260729/manifests/manifest_scale_heatmap_bk.csv \
  --output-dir artifacts/40_scale_heatmap_models_20260729/bk \
  --epochs 40 --batch-size 16 --device auto
```

Stampa una riga per epoca con le metriche di validazione già nelle stesse tolleranze
dell'harness (`x` ≤ 6 px, `y_zero` ≤ 8 px, calibrazione ≤ 2%), quindi i numeri sono
confrontabili con la pipeline classica senza conversioni.

**Cosa guardare, in ordine di importanza:**

1. `x_band=` è la metrica che conta davvero: la rete deve dire *quale colonna*, e il
   detector classico raffina entro ±70 px arrivando a ~2 px. Chiedere alla rete 6 px è
   chiederle il lavoro dello stadio classico, e non può farlo perché il resize distrugge
   le tacche (a 1920 px di larghezza ridotti a 512, una tacca passa da 8×2 px a 2.1×0.9).
2. `dir=` deve **salire** durante il training. Se resta piatta, confrontala con la quota
   della classe maggioritaria del vendor: se coincide, la testa sta predicendo una
   costante e non sta imparando (vedi sotto).
3. `y_tol=` dovrebbe arrivare sopra il 90%.
4. `calib=` sarà il più basso e va bene: la calibrazione la dà l'OCR, questa testa è
   solo un ripiego e un controllo incrociato.

`handoff=` è il punteggio con cui viene scelto il checkpoint migliore: media di
`x_band`, `y_tol` e verso, cioè le tre cose di cui la rete è responsabile.

> **Non selezionare su `strict_ok`.** Nel primo giro il checkpoint veniva scelto così, ma
> `strict_ok` è dominato dalla calibrazione (2-5%) — cioè dal compito che la rete non deve
> fare — quindi la selezione era di fatto casuale: su BK ha scelto l'epoca 18 invece della
> 30. Corretto il 2026-07-29.

### Bug corretto il 2026-07-29: la testa del verso era cieca

Nel primo giro `direction_acc` coincideva **esattamente** con la quota della classe
maggioritaria di ogni split (Esaote val 91.73%, Esaote test 97.58%, BK val 23.81%): la
testa prediceva una costante. Causa: leggeva un vettore da `AdaptiveAvgPool2d(1)`, cioè
una media su tutto il frame, che cancella l'asimmetria verticale di cui "lo zero è in alto
o in basso" è fatto. Verificato in modo diretto: quella testa dava output identico per
un'immagine e la sua versione ribaltata.

Correzione: la testa globale ora riceve anche un **profilo verticale** (media sulla
larghezza, poi 16 bin sull'altezza, che preserva l'ordine sopra/sotto), e la loss del verso
è bilanciata sulle classi — necessario su Esaote, dove lo zero è in basso solo nel 5% dei
casi e una loss non pesata si minimizza ignorando la classe minoritaria.

**Esaote, BK e Hitachi vanno riaddestrati** con il codice corretto (~26 + 20 + 15 min).

Tempo atteso: qualche minuto per epoca su MPS con 961 immagini. Se è troppo lento,
`--image-width 384 --image-height 384` dimezza il costo.

## 4. Tutti i vendor eleggibili

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --dataset-dir artifacts/39_scale_heatmap_dataset_20260729 \
  --output-root artifacts/40_scale_heatmap_models_20260729 \
  --all-eligible --epochs 40
```

Scrive `artifacts/40_scale_heatmap_models_20260729/run_summary.json` con il test di ogni
vendor, più `<vendor>/best_model.pt`, `summary.json` e `train_log.jsonl`.

## 4a. Riaddestrare dopo la correzione del 2026-07-29

Il run del primo giro (`artifacts/40_scale_heatmap_models_20260729/`) è da buttare per le
teste `direction` e `log_span_mm`: erano cieche. Le heatmap `x`/`y` invece erano sane, e i
loro numeri restano il riferimento da battere:

| | Esaote test | BK test |
|---|---|---|
| `y_tol` (≤8 px) | 96.2% | 58.7% |
| err `y_zero` mediano | 1.27 px | 5.94 px |
| `x_tol` (≤6 px) | 35.3% | 60.0% |
| err `x` mediano | 7.50 px | 4.40 px |
| verso | 97.6% *(costante)* | 47.7% *(costante)* |

Nota che `x` va **meglio** su BK (4.4 px) che su Esaote (7.5 px): BK è 1024 px di larghezza
(1 px di input = 2.0 px reali) contro i 1920 di Esaote (3.75 px reali). È la conferma che il
limite in `x` è il resize, non il modello. Su BK invece `y` è peggio, e lì l'imputato è la
selezione del checkpoint sbagliata (epoca 18).

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --dataset-dir artifacts/39_scale_heatmap_dataset_20260729 \
  --output-root artifacts/40_scale_heatmap_models_20260730 \
  --all-eligible --epochs 40
```

Output in una cartella nuova, così il confronto prima/dopo resta possibile.

## 4b. A/B: le fusion aiutano o disturbano il training?

Domanda legittima, perché sulle acquisizioni fusion ci sono **due righelli** e il target ha
un solo picco. Ma l'etichetta non è sbagliata (`x` e `y_zero` sono esatti), è il compito a
essere più difficile — quindi va misurato, non deciso a intuito.

Contesto misurato sulla pipeline classica: sulle fusion **15 righe su 15 non producono
alcuna predizione**, con errore mediano in `x` di 981 px (aggancia il pannello sbagliato),
contro 21% strict e 41% senza predizione sul resto. Quota di fusion nei dati: BK 12.5%,
Hitachi 9.9%, Canon 8.3%, Esaote 4.8%.

Il dataset senza fusion è già pronto in
`artifacts/39_scale_heatmap_dataset_20260729_nofusion/` (389 righe rimosse; su BK il train
passa da 961 a 825).

**La regola dell'esperimento: entrambi i bracci vanno misurati sulle STESSE righe**, cioè
sul test *senza* fusion. Verificato che il test no-fusion di BK (143 righe) è un
sottoinsieme esatto del test completo (155), quindi il confronto è pulito.

```bash
# braccio B — training senza fusion (il braccio A è il run già fatto al punto 4)
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --manifest artifacts/39_scale_heatmap_dataset_20260729_nofusion/manifests/manifest_scale_heatmap_bk.csv \
  --output-dir artifacts/40_scale_heatmap_models_20260729_nofusion/bk \
  --epochs 40

# braccio A rimisurato sulle stesse righe non-fusion
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --eval-only artifacts/40_scale_heatmap_models_20260729/bk/best_model.pt \
  --manifest artifacts/39_scale_heatmap_dataset_20260729_nofusion/manifests/manifest_scale_heatmap_bk.csv \
  --eval-split test \
  --eval-out artifacts/40_scale_heatmap_models_20260729/bk/eval_on_nofusion_test.json

# braccio B sullo stesso test (per completezza: e' gia' il suo test)
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --eval-only artifacts/40_scale_heatmap_models_20260729_nofusion/bk/best_model.pt \
  --manifest artifacts/39_scale_heatmap_dataset_20260729_nofusion/manifests/manifest_scale_heatmap_bk.csv \
  --eval-split test \
  --eval-out artifacts/40_scale_heatmap_models_20260729_nofusion/bk/eval_on_nofusion_test.json
```

**Come leggere il risultato.** Confronta `x_within_tol_pct` (la metrica per cui la rete
esiste) fra i due `eval_on_nofusion_test.json`:

- se il braccio senza fusion è **migliore in modo netto**, le fusion disturbano davvero e
  conviene escluderle dal training — ma allora serve una strategia dedicata per loro,
  perché oggi non abbiamo nulla che le gestisca;
- se sono **equivalenti o il braccio con fusion è migliore**, tenerle è la scelta giusta:
  costa nulla e sono l'8-12% dei casi reali.

Utile anche la controprova opposta: valutare il modello del braccio B sul test **completo**
(che include le fusion) per vedere quanto perde su ciò che non ha mai visto.

## 4c. Quanto costa aver escluso le fusion (misura, non congettura)

Rilanciare il training con e senza fusion per confrontare `x_tol` non serve: la metrica che
conta per la rete, `x_band`, è già al 98.6% e non ha margine. La domanda utile è un'altra —
**un modello che non ha mai visto una fusion, cosa fa davanti a una fusion?**

Set diagnostico già pronto: `artifacts/39_scale_heatmap_dataset_FUSION_ONLY/`, 389 righe di
sole cartelle fusion, tutte forzate nello split `test` perché per un modello addestrato
senza fusion sono comunque tutte inedite (Esaote 83, BK 161, Hitachi 91, Canon 19).

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
  --eval-only artifacts/40_scale_heatmap_models_clean/esaote/best_model.pt \
  --manifest artifacts/39_scale_heatmap_dataset_FUSION_ONLY/manifests/manifest_scale_heatmap_esaote.csv \
  --eval-split test \
  --eval-out artifacts/40_scale_heatmap_models_clean/esaote/eval_on_fusion.json
```

Costa secondi. **Come leggerlo**, confrontando con `x_band` 98.6% e `y_tol` 96.5% del test
normale:

- `x_band` ancora alto (>85%) → l'esclusione è gratis: la rete generalizza alle fusion
  anche senza averle viste, e il resto lo fanno detector e consenso;
- `x_band` che crolla (<50%) → il modello non regge le fusion, e allora servono nel
  training oppure serve una strategia dedicata. Non lasciarle semplicemente fuori: sono
  il 7.5% dei dati e la pipeline classica su quelle è a zero.

Set diagnostici analoghi si costruiscono con `--only-folder-token <token> --force-split test`.

## 5. Misurare l'effetto vero: rete + geometria + OCR + consenso

Il numero che conta non è la metrica della rete da sola, ma quanto migliora la catena.

**Due regole, entrambe necessarie perché il confronto significhi qualcosa:**

1. **Solo righe che le reti non hanno visto.** L'harness campiona dalla GT completa, dove
   la maggior parte delle cartelle è finita nel *train* delle reti. Misurare lì darebbe un
   risultato gonfiato. Usare `artifacts/41_scale_chain_eval/gt_heatmap_test_only.csv`:
   425 righe / 31 setup (BK 143, Esaote 282), tutte dallo split test dei manifest usati per
   addestrare.
2. **Entrambi i bracci sulla stessa macchina.** La catena usa Tesseract, e la versione
   differisce fra macOS e Linux (parità già segnalata come rischio aperto in `CLAUDE.md`).
   Due bracci su macchine diverse misurerebbero anche la differenza di OCR.

```bash
# braccio A — solo pipeline classica
python3 tools/scale/eval_scale_detector.py \
  --gt artifacts/41_scale_chain_eval/gt_heatmap_test_only.csv \
  --output-dir artifacts/41_scale_chain_eval/senza_rete \
  --consensus

# braccio B — con le reti come prior
python3 tools/scale/eval_scale_detector.py \
  --gt artifacts/41_scale_chain_eval/gt_heatmap_test_only.csv \
  --output-dir artifacts/41_scale_chain_eval/con_rete \
  --consensus \
  --heatmap-models artifacts/40_scale_heatmap_models_clean
```

Circa 3-4 minuti per braccio. Il braccio B deve stampare in testa
`[heatmap] loaded bk ...`, `loaded esaote ...`: se dice "nessun modello heatmap caricato",
la cartella dei modelli è sbagliata e stai misurando due volte la stessa cosa.

**Cosa confrontare**, in `summary.json` → `metrics.overall`:

| Campo | Perché |
|---|---|
| `sources.none` | **la metrica decisiva**: righe senza alcuna predizione. Nel classico era 37% su BK e 55% su Esaote. Se la rete funziona, qui crolla |
| `calib_ok_pct` | la calibrazione, cioè la qualità vera del blocco |
| `strict_ok_pct` | criterio di produzione |
| `direction_ok_pct` | dovrebbe salire: la rete ha il verso a 91-100% per vendor |
| `status.accepted` | copertura della policy di confidenza |

La gallery `review_scale_eval.html` di entrambi i bracci mostra verde = GT, rosso =
predizione, coi fallimenti in testa: utile per capire *dove* la rete aiuta e dove no.



Il numero che conta non è la metrica della rete da sola, ma quanto migliora la catena
completa. Stesso harness, stesse righe, con e senza rete:

```bash
# senza rete (baseline attuale)
python3 tools/scale/eval_scale_detector.py \
  --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
  --output-dir artifacts/41_scale_chain_eval/bk_classico \
  --vendor BK --consensus --max-rows 200

# con rete
python3 tools/scale/eval_scale_detector.py \
  --gt artifacts/37_scale_gt_audit_20260729/scale_gt_rows_clean.csv \
  --output-dir artifacts/41_scale_chain_eval/bk_con_rete \
  --vendor BK --consensus --max-rows 200 \
  --heatmap-models artifacts/40_scale_heatmap_models_20260729
```

Confronta in `summary.json` → `metrics.overall`: `strict_ok_pct`, `calib_ok_pct` e
soprattutto `sources`, dove `none` sono le righe senza alcuna predizione. **Il successo
della rete si vede lì: `none` deve crollare.** Oggi è 35/95 su BK e 50/91 su Esaote.

La gallery `review_scale_eval.html` mostra verde = GT, rosso = predizione, coi fallimenti
in testa.

---

## Come si incastrano i pezzi

```
rete heatmap  ──► quale colonna è il righello, dov'è lo zero, che verso, mm/px di ripiego
                        │
geometria     ──► trova le singole tacche DENTRO quella colonna (qui è precisa: 1 px)
                        │
OCR           ──► legge i numeri → mm_per_px vero e verso (unica evidenza assoluta)
                        │
consenso      ──► trend robusto sulle depth del setup: scarta e CORREGGE gli incoerenti
```

La rete non sostituisce l'OCR: i numeri stampati restano l'unica prova diretta della scala
assoluta. La rete sostituisce la **soglia** nella fase di detection, che è dove la catena
si rompe.

Se un modello manca o torch non è installato, `--heatmap-models` stampa un avviso e la
pipeline gira classica: nessun percorso obbligatorio.

## Cosa è già stato verificato (senza torch)

- Round-trip encode/decode dei target su **tutte le 5210 righe** della GT: errore di
  quantizzazione mediano 0.22 px in x e 0.14 px in y, verso corretto al 100%,
  `mm_per_px` esatto a 5e-7. I bin di `x` sono 512 proprio perché a 256 il caso peggiore
  era 4 px, troppo vicino alla tolleranza di 6.
- `--check-data` su tutti e 6 i vendor: nessuna immagine illeggibile, nessuna risoluzione
  incoerente, nessun problema di coordinate.
- La catena classica non è cambiata: rieseguita dopo l'aggancio, dà gli stessi numeri
  (BK strict 29.5%, calib 44.2%).

## Attenzione

- L'augmentation **non fa flip**, né orizzontali né verticali: uno sposterebbe il righello
  dall'altro lato, l'altro invertirebbe il verso. Insegnerebbero il contrario dell'etichetta.
- Le tacche non sono un target: il passo *visibile* non è sempre lo 0.5 cm che il `.fss`
  dichiara (misurata una riga Hitachi con tacche a 1 cm), quindi una griglia sintetica
  sarebbe un'etichetta in parte falsa.
- `y_far` non è un target: è dove l'operatore ha smesso di trascinare.
- Per il deploy Windows servirà l'export ONNX (Fase 5 della roadmap): su Windows non c'è
  MPS e il training resta sul Mac.
