# RECT_DEPTH autonomous checkpoint 2026-07-02

Questo checkpoint fotografa lo stato del modulo prima dell'integrazione nella
pipeline completa.

## Stato modulo

Entrypoint principale:

```text
tools/depth/predict_rect_depth_autonomous.py
```

Tool di supporto:

- `tools/depth/build_rect_depth_autonomous_review_html.py`
- `tools/depth/rect_depth_autonomous_app.py`
- `tools/depth/build_rect_depth_runs_index_html.py`
- `tools/depth/rect_depth_hybrid.py`
- `tools/depth/build_rect_depth_candidate_dataset.py`
- `tools/depth/train_rect_depth_candidate_ranker.py`

Documentazione tecnica:

- `docs/rect_depth_autonomous_integration.md`
- `tools/depth/rect_depth_module_contract.json`
- `tools/depth/README.md`

## Ultimo miglioramento importante

Hitachi Arietta V60:

```text
/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION/93.Hitachi Arietta V60_L441_POLONIA_2020 10 19/DEPTH TUTTE/
```

Problema osservato:

- il modulo favoriva la scala;
- invece la cartella doveva usare la direct label `R:valore` nella riga bassa
  dell'interfaccia;
- OCR fondeva spesso il valore con testo successivo, per esempio
  `R:10.0BG:77BD:80`;
- il box iniziale inglobava lettere/numeri dopo la depth.

Soluzione implementata:

- strategia specifica Hitachi/Arietta `hitachi_bottom_direct_stable`;
- OCR mirato sulla riga bassa;
- voto tra passate OCR indipendenti per evitare letture isolate sbagliate;
- box calcolato sulla sottostringa `R:numero`;
- rifinitura visiva sui pixel dei caratteri per fermarsi prima della `B` quando
  non ci sono `cm/mm`.

Risultato:

- artifact: `artifacts/24_rect_depth_hybrid/hitachi_arietta_v60_l441_bottom_r_direct_v5`
- review: `http://127.0.0.1:8765/hitachi_arietta_v60_l441_bottom_r_direct_v5/autonomous_review.html`
- 28/28 `accepted`
- 28/28 `direct_label`
- 0 mismatch diagnostici
- box mediano circa 59 px

## Decisioni consolidate

- La regola direct-vs-scale e' di cartella.
- Direct label vince sulla scala se contiene un valore numerico valido e varia
  coerentemente nella cartella.
- `cm/mm` sono indizi fortissimi, ma non obbligatori.
- Se non ci sono `cm/mm`, il box finisce subito dopo il numero.
- Dopo il numero non sono ammesse lettere diverse da `cm/mm`.
- `0` non puo' mai essere depth.
- La scala e' fallback e sceglie il valore massimo a un endpoint.
- La scala e' quasi sempre a destra; sinistra e' fallback.
- Marker non-depth (`Hz`, `MHz`, `dB`, `FR`, `2D`, `Print`, `X3`) non possono
  diventare depth.

## Cosa passare dalla pipeline

Priorita' degli input:

1. cartella immagini/configurazione;
2. vendor;
3. sonda;
4. rettangolo ecografico, da `.fss` o manuale;
5. eventuale lista di depth attese;
6. lato scala preferito se noto.

Se la pipeline produce gia' un JSON di contesto, preferire `--context-json`.
Altrimenti passare `--vendor`, `--probe`, `--rect-echo`.

## Output da usare

Output primario:

```text
rect_depth_autonomous_predictions.csv
```

Output di audit:

```text
rect_depth_autonomous_candidates.csv
summary.json
```

La pipeline dovrebbe usare automaticamente solo righe `accepted`. Le righe
`review` vanno portate al tool HTML o a una fase manuale. Le righe `reject` non
devono produrre depth automatica.

## Comando produzione raccomandato

```bash
python3 tools/depth/predict_rect_depth_autonomous.py \
  --folder "/path/cartella" \
  --output-dir "artifacts/24_rect_depth_hybrid/pipeline_runs/NOME_RUN" \
  --context-json "/path/context_pipeline.json" \
  --max-images 0
```

Se non c'e' `.fss`:

```bash
python3 tools/depth/predict_rect_depth_autonomous.py \
  --folder "/path/cartella" \
  --output-dir "artifacts/24_rect_depth_hybrid/pipeline_runs/NOME_RUN" \
  --vendor Esaote \
  --probe "L4-15" \
  --rect-echo "left,top,right,bottom" \
  --max-images 0
```

## Prima dell'integrazione

- Verificare che Tesseract sia installato nella macchina target.
- Verificare che il ranker joblib esista o accettare il fallback rule-only.
- Decidere se la pipeline vuole processare tutte le immagini (`--max-images 0`) o
  un campione.
- Decidere dove salvare artifact e review.
- Stabilire una policy: `accepted` automatici, `review` manuali, `reject` non
  usati.

## Nota su macOS e Windows

Il cuore del modulo e' Python + Tesseract + OpenCV, quindi e' portabile. Le
prestazioni dipendono dalla versione di Tesseract e dalla disponibilita' di OCR
locale aggiuntivo. Per avere risultati uguali tra macOS e Windows, fissare:

- versione Tesseract;
- lingua/configurazione OCR;
- versione Python e dipendenze;
- modello ranker;
- parametri CLI.
