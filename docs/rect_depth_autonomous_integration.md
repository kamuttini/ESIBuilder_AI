# RECT_DEPTH autonomous integration guide

Checkpoint: 2026-07-02

Questo documento descrive come integrare il modulo autonomo RECT_DEPTH nella pipeline
completa. Il modulo usa OCR, regole classiche, coerenza di cartella e un ranker
tabellare come prior. La rete full-frame non e' il driver principale: le review
hanno mostrato che la logica classica/OCR e' molto piu' affidabile per questo
campo.

## Entry point

Script principale:

```bash
python3 tools/depth/predict_rect_depth_autonomous.py \
  --folder "/path/cartella_immagini_o_configurazione" \
  --output-dir artifacts/24_rect_depth_hybrid/run_name \
  --context-json "/path/context_pipeline.json"
```

Per analizzare tutte le immagini della cartella in pipeline, non solo un campione:

```bash
--max-images 0
```

Il default e' `80`, pensato per uso interattivo e review rapida.

## Input dalla pipeline

Input obbligatori:

- `--folder`: cartella di configurazione o cartella contenente le immagini. Se
  esiste `image_samples/`, viene usata quella.
- `--output-dir`: directory dove scrivere CSV e summary.

Input consigliati quando disponibili:

- `--context-json`: JSON prodotto dalla pipeline con vendor/sonda. Il modulo cerca
  campi come `vendor_predicted`, `manufacturer`, `predicted_probe_id`,
  `predicted_probe`, `probe_name`.
- `--vendor`: override esplicito del vendor.
- `--probe`: override esplicito della sonda.
- `--vendor-template`: profilo/regole da usare quando il nome vendor libero non
  coincide con il template desiderato.
- `--fss`: se presente, consente di leggere `RECT_ECHO` e depth attese legacy.
- `--rect-echo "left,top,right,bottom"`: rettangolo ecografico manuale per cartelle
  senza `.fss`.
- `--expected-depths "20,25,30,35,40,50,60"`: valori attesi in mm, opzionali.
- `--scale-side-preference right|left|auto`: di default il lato scala preferito
  deriva dal profilo vendor ed e' quasi sempre `right`.

## Output per la pipeline

Il file da consumare normalmente e':

```text
rect_depth_autonomous_predictions.csv
```

Campi principali:

- `image_path`: immagine sorgente.
- `status`: `accepted`, `review`, `reject`.
- `score`: confidenza finale dopo ranker e regole.
- `mode`: `direct_label`, `scale`, `numeric_accessory`.
- `depth_mm`: depth normalizzata in millimetri.
- `left`, `top`, `right`, `bottom`: box del valore depth selezionato.
- `ocr_text`: testo OCR normalizzato del candidato top.
- `reason`: spiegazione leggibile delle regole decisive.
- `best_direct_*`, `best_scale_*`: diagnostica dei migliori candidati alternativi.

File di audit/debug:

- `rect_depth_autonomous_candidates.csv`: tutti i candidati con score, OCR flags,
  box e `folder_strategy`.
- `summary.json`: conteggi, profilo vendor applicato, `rect_echo`,
  `expected_depths_mm`, path dei CSV.

## Interpretazione degli status

- `accepted`: candidato coerente con regole locali e strategia di cartella. La
  pipeline puo' usarlo automaticamente.
- `review`: candidato plausibile ma con incertezza esplicita, per esempio unita'
  attesa non letta, OCR sporco, box locale non del tutto affidabile. Va mostrato
  nel tool di review o gestito come fallback controllato.
- `reject`: nessun valore depth valido. Non usare `depth_mm` come risultato
  automatico.

Soglie attuali nel codice:

- `accepted`: `score >= 0.62`
- `review`: `0.35 <= score < 0.62`
- `reject`: score sotto `0.35` o candidato invalidato dalle regole.

## Regole cliniche/OCR consolidate

Regole generali:

- Senza numero non esiste depth.
- `0`, `0.0`, `0.0 cm/mm` non sono mai depth.
- I soli suffissi ammessi sono `cm`, `mm` oppure nessun suffisso.
- `dB`, `Hz`, `MHz`, `Print`, `FR`, `2D`, `X3` e simili sono non-depth.
- Se non ci sono `cm/mm`, il box deve fermarsi subito dopo il numero.
- Dopo il numero non sono ammesse altre lettere; l'unica eccezione e' `cm/mm`.
- `D`, `P`, `R` sono marker validi solo come lettere isolate a sinistra del
  valore. `Depth` puo' stare a sinistra o sopra il valore.
- Il valore direct dell'interfaccia batte la scala quando e' presente e varia in
  modo coerente nella cartella.
- Se un direct resta costante nella cartella mentre il massimo scala varia, e'
  sospetto: il metodo di cartella puo' passare alla scala.
- Se si usa la scala, va selezionato il valore massimo, sempre a una delle due
  estremita' della scala e nella meta' opposta rispetto allo `0`.
- La scala e' quasi sempre a destra; la sinistra resta fallback raro.
- La scelta del metodo e' di cartella: non devono convivere direct e scala per
  immagini della stessa configurazione salvo casi `review` espliciti.

Strategie di cartella implementate:

- `direct_label_unit_stable`: `D/P/R/Depth + valore + cm/mm` stabile e variabile.
- `direct_label_d_stable`: serie `D + valore` senza unita', stabile e variabile.
- `direct_left_context_stable`: lettera target separata a sinistra del numero.
- `direct_interface_unit_stable`: valore con `cm/mm` stabile in interfaccia anche
  se la lettera non e' nel crop.
- `hitachi_bottom_direct_stable`: profilo Hitachi/Arietta, label `R/P/D` nella
  riga bassa dell'interfaccia.
- `scale_from_variable_scale`: direct quasi costante e scala variabile.
- `scale_unit_required`: scala con unita' `cm/mm` ricorrente.
- `scale_endpoint_lane`: endpoint scala stabile, usato soprattutto per BK.

## Nota Hitachi del checkpoint

La run:

```text
/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION/93.Hitachi Arietta V60_L441_POLONIA_2020 10 19/DEPTH TUTTE/
```

ha consolidato una regola specifica Hitachi/Arietta:

- la depth e' nella riga bassa dell'interfaccia, come `R:2.50`, `R:10.0`, ecc.;
- la scala non deve vincere quando questa direct label e' presente;
- OCR spesso fonde testo come `R:10.0BG:77BD:80`;
- il valore viene letto come direct, ma il box viene ritagliato solo su
  `R:10.0`, usando anche i pixel dei caratteri per non includere la `B`.

Risultato checkpoint `hitachi_arietta_v60_l441_bottom_r_direct_v5`:

- 28 immagini
- 28 accepted
- 28 direct_label
- 0 mismatch rispetto ai valori noti dal nome file
- box mediano circa 59 px, senza includere `BG/BD`

Review locale:

```text
http://127.0.0.1:8765/hitachi_arietta_v60_l441_bottom_r_direct_v5/autonomous_review.html
```

## Tool di review

Per generare la pagina HTML da una run:

```bash
python3 tools/depth/build_rect_depth_autonomous_review_html.py \
  --batch-dir artifacts/24_rect_depth_hybrid/run_name \
  --output-html artifacts/24_rect_depth_hybrid/run_name/autonomous_review.html \
  --max-candidates 8
```

Per launcher interattivo:

```bash
python3 tools/depth/rect_depth_autonomous_app.py --host 127.0.0.1 --port 8770
```

Per indice di tutte le run:

```bash
python3 tools/depth/build_rect_depth_runs_index_html.py \
  --root artifacts/24_rect_depth_hybrid \
  --output-html artifacts/24_rect_depth_hybrid/run_index.html \
  --no-generate-missing-reviews
```

## Integrazione consigliata

1. La pipeline identifica o riceve vendor, sonda, rettangolo ecografico e cartella
   immagini.
2. La pipeline lancia `predict_rect_depth_autonomous.py` con `--max-images 0` per
   produzione completa, oppure con limite per review rapida.
3. La pipeline legge `summary.json` e `rect_depth_autonomous_predictions.csv`.
4. Se tutte o quasi tutte le immagini sono `accepted`, usa i box e i valori.
5. Se ci sono `review`/`reject`, mostra la review HTML o esporta le immagini per
   correzione.
6. Le correzioni/commenti manuali vanno conservati come JSON di review per
   migliorare regole e OCR, non come hint runtime per singola immagine.

## Dipendenze operative

- Python 3
- Tesseract CLI disponibile nel PATH
- Pillow
- OpenCV opzionale ma consigliato per rilevamento tacche/scale
- joblib/scikit-learn per il ranker tabellare, se il modello e' disponibile
- su macOS, il modulo puo' sfruttare passaggi OCR locali aggiuntivi; nessun dato
  viene inviato fuori macchina.

## Rischi aperti

- Alcuni vendor hanno direct label senza `cm/mm` e senza marker forte: servono
  altre review per consolidare la strategia di cartella.
- Scale sovrapposte all'immagine ecografica restano difficili quando OCR non
  separa numeri e tacche.
- Il default `--max-images 80` e' comodo per review, ma va cambiato a `0` in
  produzione se serve processare tutta la cartella.
- Il ranker e' utile come prior, ma le decisioni affidabili vengono dalle regole
  OCR/classiche: non considerarlo un modello autonomo sufficiente.
