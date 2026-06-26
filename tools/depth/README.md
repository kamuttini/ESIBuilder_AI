# RECT_DEPTH Hybrid Module

Modulo mirato per lavorare sulle depth legacy senza avviare subito training costosi.

## 1. Manifest da configurazioni legacy

Estrae la riga `#17 RECT_DEPTH` e la riga `#18 VECT_DEPTH` dai `.fss`, collega ogni
depth al frame corrispondente in `image_samples` e, se richiesto, salva i crop.

```bash
python3 tools/depth/rect_depth_hybrid.py manifest \
  --dataset-root /Volumes/SSD_esi3_n1 \
  --output-csv artifacts/24_rect_depth_hybrid/rect_depth_manifest.csv \
  --crops-dir artifacts/24_rect_depth_hybrid/crops
```

Output utile per training rete:

- `depth_mm`: label target in mm
- `flip_state`: `nf`, `lr`, `ud`, `lrud`
- `top,left,bottom,right`: rettangolo `RECT_DEPTH` legacy
- `source_image`: frame da cui e' stato estratto il crop
- `crop_path`: crop pronto per classificatore/regressore
- `template_path`: eventuale `DB_echo/setup_<id>/depth_<n>.png`

## 2. Predizione OCR classica a livello cartella

Usa Tesseract su piu' immagini della stessa cartella, raggruppa i testi numerici
ricorrenti per posizione e li ordina con segnali classici:

- presenza di unita' tipo `cm`
- presenza di unita' tipo `mm`
- vicinanza o prefisso testuale `Depth` / `D` / `P`
- posizione laterale o fuori dal `RECT_ECHO`
- penalita' se il numero cade nel centro del rettangolo ecografico
- supporto su piu' immagini
- variabilita' dei valori
- plausibilita' numerica
- se disponibile, snapping alle depth attese del `.fss`

```bash
python3 tools/depth/rect_depth_hybrid.py predict \
  --folder "data/Dataset/Esaote_MyLab Omega, F080104 - TLC3-13 - T - Rev.01" \
  --fss "data/Dataset/Esaote_MyLab Omega, F080104 - TLC3-13 - T - Rev.01/DB_setup/setup_42.fss" \
  --output-dir artifacts/24_rect_depth_hybrid/predict_esaote_42
```

Output:

- `rect_depth_candidates.csv`: rettangoli candidati ordinati per score
- `rect_depth_predictions.csv`: valore depth stimato per immagine
- `summary.json`: riepilogo run

Senza `--fss`, il tool non forza una decisione unica: espone i candidati. Questo e'
voluto, perche' in alcune configurazioni la dicitura `cm` e' un ottimo hint, mentre
in altre la zona legacy della depth non ha `cm/mm` chiarissimo e va risolta con il
modello neurale o review.

## 3. Smoke training bbox RECT_DEPTH

Trainer leggero per verificare se una rete full-frame riesce a regredire il
rettangolo `RECT_DEPTH` direttamente.

```bash
python3 tools/depth/train_rect_depth_bbox_model.py \
  --manifest artifacts/24_rect_depth_hybrid/rect_depth_manifest_ssd_esi1_n1.csv \
  --output-dir artifacts/24_rect_depth_hybrid/bbox_smoke_cpu \
  --epochs 2 \
  --max-per-config 4 \
  --batch-size 32 \
  --image-width 320 \
  --image-height 180 \
  --device cpu
```

La prima smoke su `/Volumes/SSD_esi1_n1` ha confermato che la pipeline di training
funziona, ma il modello full-frame puro non e' ancora abbastanza preciso: dopo 2
epoche su 1.576 campioni ha ottenuto circa `val_mae_x=119.8 px`,
`val_mae_y=120.1 px`, `val_iou=0.012`.

Indicazione pratica: usare la rete da sola sul frame intero e' troppo debole per
ora. La direzione migliore e' ibrida: OCR/candidati classici restringono la zona
probabile, poi una rete piccola fa ranking/correzione dei candidati o riconosce il
crop depth.

## 4. Review GT v4 e ranker OCR

Export manuale usato:

```bash
artifacts/24_rect_depth_hybrid/bbox_smoke_cpu/rect_depth_dataset_gt_review_20260618_export4.json
```

Applicazione delle correzioni al manifest legacy:

```bash
python3 tools/depth/apply_rect_depth_review_export.py \
  --manifest artifacts/24_rect_depth_hybrid/rect_depth_manifest_ssd_esi1_n1.csv \
  --review-json artifacts/24_rect_depth_hybrid/bbox_smoke_cpu/rect_depth_dataset_gt_review_20260618_export4.json \
  --output-csv artifacts/24_rect_depth_hybrid/rect_depth_manifest_ssd_esi1_n1_gt_review_20260618_v4_corrected.csv \
  --summary-json artifacts/24_rect_depth_hybrid/rect_depth_manifest_ssd_esi1_n1_gt_review_20260618_v4_corrected.summary.json \
  --only-reviewed \
  --require-corrected
```

Risultato: `1314` campioni corretti, `360` configurazioni, `49` esclusi rispettati.
I commenti confermano gli hint principali: `D`, `P`, `R`, `cm`, `mm`, `depth`,
`scala/scale`, e confermano che il valore e' quasi sempre accessorio/laterale, non
nel centro del rettangolo ecografico.

Dataset candidati OCR supervisionato:

```bash
python3 tools/depth/build_rect_depth_candidate_dataset.py \
  --review-manifest artifacts/24_rect_depth_hybrid/rect_depth_manifest_ssd_esi1_n1_gt_review_20260618_v4_corrected.csv \
  --output-csv artifacts/24_rect_depth_hybrid/candidate_ranker_v4_full/candidate_rows.csv \
  --summary-json artifacts/24_rect_depth_hybrid/candidate_ranker_v4_full/summary.json \
  --max-candidates-per-sample 12 \
  --verbose
```

Risultato: `7148` righe candidato, `2071` cluster OCR, `470/1314` campioni con
almeno un candidato corretto tra quelli generati. Quindi il ranker funziona solo se
prima miglioriamo il recall dei candidati, soprattutto su Hitachi/Biopsee.

Training ranker tabellare:

```bash
python3 tools/depth/train_rect_depth_candidate_ranker.py \
  --candidates-csv artifacts/24_rect_depth_hybrid/candidate_ranker_v4_full/candidate_rows.csv \
  --output-model artifacts/24_rect_depth_hybrid/candidate_ranker_v4_full/ranker.joblib \
  --summary-json artifacts/24_rect_depth_hybrid/candidate_ranker_v4_full/ranker_summary.json
```

Prima metrica: `val_auc=0.953`, `val_average_precision=0.693`, ma `val_top1=0.189`
perche' l'`oracle_recall` dei candidati in validation e' solo `0.211`.

Direzione successiva:

- aumentare i candidati OCR con ROI laterali e crop scala, non con una rete full-frame;
- aggiungere regole vendor-specific per Hitachi (`R`, basso/laterale, scala) e Biopsee;
- usare il ranker quando OCR produce candidati plausibili;
- usare una rete piccola solo come fallback/ROI proposer sui casi senza candidato OCR,
  non come regressore unico del rettangolo.

## 5. Predittore autonomo RECT_DEPTH

Modulo production-facing dopo le review manuali:

- non usa hint manuali per immagine;
- puo' ricevere vendor/sonda dalla pipeline (`--vendor`, `--probe`, `--context-json`);
- usa il ranker tabellare come prior, ma il postprocess OCR/classico puo' bocciare
  candidati con marker non-depth;
- preferisce valori informativi diretti: `Depth`, `D`, `P`, `R` prima del numero;
- accetta solo suffissi `cm`, `mm` o nessun suffisso;
- se non c'e' valore diretto affidabile, usa la scala come fallback e sceglie il
  valore massimo coerente;
- la scala va cercata prioritariamente a destra del rettangolo ecografico; la
  sinistra resta un fallback raro;
- anche se un candidato `D`/`P`/`R`/`Depth` sembra forte, se il valore resta
  costante nella cartella mentre il massimo scala varia, scatta una strategia di
  cartella: il direct viene trattato come non-depth e si usa la scala per tutte
  le immagini;
- quando la cartella e' gia' in strategia scala ma l'OCR non legge il numero su
  una singola immagine, il caso resta in review: non viene mai creato un valore
  artificiale dalla posizione o dal numero di tacche;
- nei valori scala, il suffisso `cm/mm` vicino al bordo destro batte numeri
  unitless letti sulle tacche; valori OCR troncati come `40cm` vengono letti
  come `4.0cm` quando il numero sarebbe altrimenti fuori scala, e casi tipo
  `s.5cm` vengono normalizzati a `3.5cm`; su Philips alcune letture OCR sporche
  della scala (`s0em`, `em`, `tm`) vengono trattate come deformazioni locali di
  `cm` solo nei crop scala;
- per le scritte piccole sovrapposte all'ecografia aggiunge due recovery pass
  locali: contrasto CLAHE e soppressione delle linee orizzontali, entrambi
  limitati alle due estremita' della colonna di tacche; su macOS aggiunge anche
  Vision OCR locale. Nessun passaggio invia immagini o testo fuori macchina;
- una cifra OCR a confidenza bassa viene conservata solo come candidato
  provvisorio di scala: deve poi essere nella corsia delle tacche, avere un
  bounding box proprio e stare a un'estremita'. Un `cm` esatto letto nello
  stesso crop/corsia puo' validare la cifra anche se Tesseract gli assegna
  confidenza zero;
- per Hitachi gestisce coppie OCR incollate tipo `R:9.00R:64`, scegliendo la prima
  coppia label-valore (`R:9.00` -> `90 mm`) e trattando la `R` successiva come nuova
  label, non come suffisso.

Esempio su una cartella con `.fss` disponibile:

```bash
python3 tools/depth/predict_rect_depth_autonomous.py \
  --folder "/Volumes/SSD_esi1_n1/Esaote MyLab Eight - 14.03.01 TRT33 - L" \
  --fss "/Volumes/SSD_esi1_n1/Esaote MyLab Eight - 14.03.01 TRT33 - L/DB_setup/setup_65.fss" \
  --vendor Esaote --probe TRT33 \
  --output-dir artifacts/24_rect_depth_hybrid/autonomous_esaote_trt33_l
```

Output:

- `rect_depth_autonomous_candidates.csv`: tutti i candidati con `ranker_score`,
  `autonomous_score`, `autonomous_status`, OCR flags e spiegazione;
- `rect_depth_autonomous_predictions.csv`: top per immagine;
- `summary.json`: contesto vendor/sonda, profilo applicato, conteggi e path output.

Report visuale/commentabile:

```bash
python3 tools/depth/build_rect_depth_autonomous_review_html.py \
  --batch-dir artifacts/24_rect_depth_hybrid/autonomous_batch_20260622 \
  --output-html artifacts/24_rect_depth_hybrid/autonomous_batch_20260622/autonomous_review.html
```

Launcher interattivo per scegliere la cartella dal browser e lanciare un nuovo
batch:

```bash
python3 tools/depth/rect_depth_autonomous_app.py --host 127.0.0.1 --port 8770
```

Poi aprire `http://127.0.0.1:8770/`. La pagina permette di navigare il volume,
selezionare una cartella, auto-compilare `.fss`, vendor e sonda quando deducibili,
lanciare il predittore e commentare ogni immagine nella review generata.

Per cartelle senza `.fss`, usare la modalita' manuale della stessa pagina:
selezionare il template vendor, indicare vendor/sonda, compilare il rettangolo
ecografico in pixel (`left, top, right, bottom`) e, se note, le depth attese in mm.
Gli stessi parametri sono disponibili anche da CLI:

```bash
python3 tools/depth/predict_rect_depth_autonomous.py \
  --folder "/Volumes/ALTRO_VOLUME/cartella_immagini" \
  --vendor BK --probe 18L5 --vendor-template BK \
  --rect-echo "205,148,984,798" \
  --expected-depths "20,25,30,35,40,45,50,55,60,70,80,90" \
  --output-dir artifacts/24_rect_depth_hybrid/manual_no_fss_run
```

Regole operative implementate:

- `direct_label` e' sempre preferito alla scala quando esiste un valore numerico
  valido nell'interfaccia;
- il valore direct puo' essere introdotto da `D`, `P`, `R` o `Depth`; `Depth` puo'
  stare anche sopra o appena sfalsato rispetto al numero;
- senza numero non esiste una depth valida;
- `0`, `0.0`, `0.0 cm/mm` non sono mai una depth valida: quando compaiono sulla
  scala va cercato il valore non-zero sull'estremo opposto, oppure il caso resta
  in review se l'OCR non separa quel valore;
- i soli suffissi ammessi sono `mm`, `cm` oppure nessun suffisso;
- `dB` non e' mai un suffisso/unità depth valida;
- `D`, `P` e `R` sono validi solo come singola lettera immediatamente a sinistra
  del numero; `Depth` puo' stare a sinistra o sopra. Dopo il numero sono ammessi
  solo `cm`, `mm` o nessuna lettera. Il crop deve quindi contenere una sola
  espressione depth pulita, senza testo UI estraneo;
- se in una cartella molti candidati scala affidabili sono accompagnati da `cm`
  o `mm`, la cartella passa in strategia unita' obbligatoria. Un numero nella
  stessa corsia/tacca con unita' OCR assente puo' comunque vincere soltanto se
  e' il massimo reale della scala: viene marcato `review` come "unita' attesa ma
  non letta", mai accettato ciecamente;
- i cluster OCR con testo tipo `5cm` ma senza token numerico agganciato all'immagine
  non sono accettati automaticamente: vengono convertiti in valore solo come review
  debole, cosi' il tool mostra un candidato numerico correggibile invece di un box
  vuoto;
- se l'OCR concatena testo sporco ma contiene una coppia forte, per esempio
  `D52mm`, viene scelto il valore della coppia e non il primo numero spurio;
- un valore `mm/cm` ricorrente nella stessa posizione della cartella viene promosso
  a informazione di interfaccia, anche se la lettera non e' nel crop;
- la scala e' fallback: si cerca soprattutto nella fascia destra del rettangolo
  ecografico, fra numeri/tacche allineati verticalmente e valori con `cm/mm`; la
  fascia sinistra e' fallback raro, e si seleziona il valore massimo del lato
  scala preferito;
- il lato scala preferito non e' hardcoded per singola cartella: deriva dal
  profilo vendor/cartella ed e' sovrascrivibile con `--scale-side-preference`
  (`right`, `left`, `auto`) o dal launcher con il campo `Lato scala`;
- la posizione direct deve essere stabile fra le immagini della stessa cartella,
  ma un direct stabile con valore costante diventa sospetto quando la scala varia
  nella stessa cartella; la scala puo' muoversi con profondita/scala
  dell'immagine.
- la corsia scala viene validata geometricamente con tacche verticali e numeri
  allineati; tale geometria esclude valori del pannello laterale, ma non inventa
  mai millimetri. Nel report i marker sono gialli e il rosso indica soltanto il
  maggiore valore OCR effettivamente associato a un'estremita' della scala: un
  valore interno, fuori corsia o minore di un altro endpoint leggibile non puo'
  mai essere selezionato come depth massima;
- un cluster OCR non puo' mai essere il top: per selezionare una depth servono
  un token numerico isolato e il suo bounding box. Se nessun endpoint ha OCR
  leggibile, il report evidenzia entrambe le estremita' come `MAX? OCR assente`
  senza produrre un valore;
- `0` non e' mai una depth. Quando compare a un estremo, la meta' opposta della
  scala diventa la zona di ricerca prioritaria del valore non-zero; se questo
  valore non e' letto, l'immagine resta in review.

Smoke su `/Volumes/SSD_esi1_n3/.../DEPTH` Esaote MyLab Omega L4-15:
`24/24 accepted`, tutti `direct_label`. Il caso OCR sporco
`15TELD52mmIM46/1` viene risolto come `52 mm`.

Smoke multi-vendor del 2026-06-22 su 24 immagini:

- BK 18L5: 2 accepted, 1 review, 5 reject; tutti fallback scala, utile per tarare
  meglio il caso "massimo scala";
- Esaote TRT33: 8 accepted su 8, tutti direct label `P...mm`;
- Hitachi CL4416R1: 4 accepted, 4 reject; migliorato con parsing coppie `R/P/D`,
  da reject generalizzati a direct label corretti sui casi `R:2.00`, `R:2.50`,
  `R:9.00`, `R:10.0`.
