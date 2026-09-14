# Blocco AGHI — classificatore acquisizioni con aghi

Riconosce se un'immagine (o una cartella di acquisizione) è una prova con aghi —
tipicamente fatta in acqua, quindi molto più chiara del normale, con gli aghi
visibili dentro il rettangolo ecografico.

## Come sono fatte le etichette

Le etichette vengono dal nome della cartella sul volume: qualsiasi immagine il cui
percorso contiene una cartella che matcha `agh` (AGHI, Verifica Aghi, aghi, ...)
è positiva, tutto il resto è negativo. Non c'è annotazione manuale.

Questo ha due conseguenze misurate, non teoriche (vedi la sezione Rumore di
etichetta più sotto): ci sono immagini con aghi fuori dalle cartelle AGHI, e
immagini dentro le cartelle AGHI che non mostrano nessun ago.

## Pipeline

Quattro passi, ognuno uno script indipendente.

### 1. Manifest e split

```bash
python3 tools/needle/prepare_needle_dataset.py \
  --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
  --output-dir artifacts/90_needle_dataset/v1 \
  --pos-cap-per-dir 60 --neg-cap-per-dir-pos-group 12 --neg-cap-per-dir-other 5
```

- Split leak-free a livello di **acquisizione** (cartella di primo livello): tutte
  le immagini di un'acquisizione finiscono nello stesso split. I frame consecutivi
  sono quasi identici, qualsiasi split più fine sarebbe una fuga di informazione.
- Cap per cartella-foglia con campionamento a passo uniforme sui file ordinati,
  così il sottoinsieme copre tutta l'acquisizione invece di una singola raffica.
- I negativi hanno un budget più alto dentro le acquisizioni che contengono anche
  positivi: sono i negativi difficili (stessa macchina, stessa sessione, spesso
  anche loro in acqua).

### 2. Crop del rettangolo ecografico

```bash
python3 tools/needle/extract_needle_crops.py \
  --manifest artifacts/90_needle_dataset/v1/manifest_needle.csv \
  --output-dir artifacts/90_needle_dataset/v1/crops \
  --device mps --resume
```

Riusa **i modelli ufficiali della pipeline** (`predict_fss_head_from_acquisitions`):
vendor CNN per il routing, regressore rect (con il checkpoint BK specializzato dalla
mappa ufficiale quando il vendor è BK con confidenza >= 0.70).

Perché croppare: a schermo intero l'immagine è dominata dalla UI del vendor (logo,
menu, banner paziente, righello) e dalla geometria del rettangolo. Una rete
addestrata sul frame intero impara "quale macchina" invece di "c'è un ago", e
l'etichetta è correlata con la macchina. Il crop toglie quella scorciatoia.

Perché fare cache: i frame originali sono PNG 1920x1080 su un volume USB esterno;
decodificarli a ogni epoca costa più del training. I crop sono piccoli e locali.

La rotazione OSD **non** è applicata (assunta 0), coerentemente con il marker runner
sullo stesso volume.

### 3. Training

```bash
python3 tools/needle/train_needle_classifier.py \
  --crops-dir artifacts/90_needle_dataset/v1/crops \
  --output-dir artifacts/91_needle_models/<nome_run> \
  --arch resnet18 --image-size 288 --epochs 16 --steps-per-epoch 4000 \
  --num-workers 0 --device cpu
```

Scelte imposte dai dati:

- **Campionamento**: i positivi sono concentrati in poche acquisizioni (una sola
  vale un terzo dei positivi di train). Il `WeightedRandomSampler` bilancia le
  classi e, dentro ogni classe, smorza il peso delle acquisizioni grandi.
- **Letterbox**: i crop hanno proporzioni molto diverse; schiacciarli in un
  quadrato cambierebbe l'angolo degli aghi, che è parte del segnale. Si riempie
  con nero invece di deformare.
- **Jitter fotometrico lieve**: "più chiara del normale" è segnale vero qui, una
  augmentation aggressiva su luminosità/contrasto cancellerebbe proprio la
  caratteristica che serve.

Le metriche sono riportate per immagine, per cartella-foglia (l'unità operativa) e
per vendor, più una policy a due soglie accepted / review / reject calibrata su
validation, nello stile del resto del progetto.

### 4. Inferenza e revisione

```bash
python3 tools/needle/predict_needle_on_folder.py \
  --model artifacts/91_needle_models/<nome_run>/best_model.pt \
  --folder "/percorso/acquisizione" --output-csv /tmp/pred.csv

python3 tools/needle/build_needle_review_html.py \
  --predictions artifacts/91_needle_models/<nome_run>/predictions_test.csv \
  --crops-dir artifacts/90_needle_dataset/v1/crops \
  --output artifacts/91_needle_models/<nome_run>/review_test.html
```

La galleria è HTML statico, senza endpoint e senza niente che salvi al clic: si
apre nel browser senza effetti collaterali.

## Rumore di etichetta (misurato)

Guardando gli errori del modello di riferimento ordinati per confidenza:

- **Falsi positivi che sono aghi veri.** Le prime immagini classificate positive
  fuori dalle cartelle AGHI mostrano aghi evidenti (es. `144. GE Versana Active/NF`,
  `193. GE Logiq P9/Tu...`, `168. Esaote MyLab Nine/L3-11`). Qui sbaglia
  l'etichetta, non il modello: la precision misurata è un limite inferiore.
- **Falsi negativi che non hanno aghi.** Le cartelle "guida aghi" contengono
  l'overlay della guida (griglia di puntini) senza nessun ago e senza acqua —
  concetto diverso da "aghi in acqua". Sono 107 positivi su 4722 (2.3%), ma 74
  cadono tutti nel gruppo BK5000 di test e da soli spiegano il crollo delle
  metriche BK.

Decisione aperta, da confermare con Francesca: se "guida aghi" debba essere una
classe separata o un negativo. Finché non è deciso, il regex `--positive-regex`
resta `agh` e le include.

## Limiti noti

- Il crop dipende dal modello rect: dove il rect sbaglia, il classificatore vede
  la cosa sbagliata. Non è stato misurato separatamente quanto pesa.
- Dentro RECT_ECHO restano la barra della scala di grigi e il watermark del vendor:
  una scorciatoia "quale vendor" resta possibile. Per questo le metriche sono
  riportate anche per vendor.
- `--trim-dark-border` esiste ma è disattivo di default: proprio quella barra e quel
  watermark stanno dentro il rettangolo e mandano a vuoto il test riga/colonna.
