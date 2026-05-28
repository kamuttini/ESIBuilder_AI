# Vendor Recognition Report (stato al 2026-03-07)

## 1) Obiettivo
Costruire una pipeline robusta per riconoscere il **vendor** da una cartella di acquisizioni HDMI dello stesso ecografo e ottenere **una sola decisione vendor per cartella** da usare nella generazione automatica del file `.fss` unico.

---

## 2) Pulizia dataset e split

### 2.1 Rimozione immagini `negative`
- Azione eseguita: rimozione fisica di tutte le immagini con `negative` nel nome dentro `Dataset`.
- File rimossi: **5775**
- Log rimozione: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/removed_negative_images_Dataset_20260306_173730.txt`
- Verifica finale: `0` immagini `negative` residue nel dataset.

### 2.2 Rigenerazione split
- Script usato: `/Users/camilla/Documents/Develop/ESIBuilder_AI/tools/prepare_ultrasound_rect_dataset.py`
- Modifica introdotta: filtro immagini per nome con default `--exclude-image-regex '(?i)negative'` (hardening permanente).
- Output split pulito: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/rect_dataset_no_negative_v2/`

Dati split (`split_summary.txt`):
- cartelle valide: **426**
- immagini totali: **28537**
- train/val/test immagini: **18456 / 4299 / 5782**
- train/val/test cartelle: **236 / 86 / 104**
- tutti i manufacturer presenti nel train.

---

## 3) Evoluzione training vendor

## 3.1 Script aggiornati

### `/Users/camilla/Documents/Develop/ESIBuilder_AI/tools/train_ultrasound_vendor_classifier.py`
Migliorie implementate:
- `RandomErasing` in augmentation train.
- `label_smoothing` configurabile (`--label-smoothing`, default 0.03).
- class weights con esponente controllabile (`--class-weight-power`, default 0.5) invece di inverso pieno.
- possibilità di disattivare class weights (`--disable-class-weights`).

### `/Users/camilla/Documents/Develop/ESIBuilder_AI/tools/prepare_ultrasound_rect_dataset.py`
Migliorie implementate:
- esclusione immagini per regex nome (default include `negative`).
- tracciamento in summary di `images_excluded_by_name_filter`.

### Documentazione aggiornata
- `/Users/camilla/Documents/Develop/ESIBuilder_AI/tools/README.md`

---

## 3.2 Run principali e metriche

### A) Baseline storica prima della pulizia (per confronto)
- Artifact: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_full_e1/metrics.json`
- `best_epoch`: 1
- `best_val_macro_f1`: 0.8234
- `test_acc`: 0.9860
- `test_macro_f1`: 0.9736
- test sample count: 7002

### B) Run “power” su dataset pulito (run finale scelto)
- Checkpoint: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_no_negative_v2_power/best_model.pt`
- Best checkpoint: **epoch 2**
- `best_val_macro_f1`: **0.9001**

Valutazione test del best checkpoint:
- File: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_no_negative_v2_power/test_metrics_from_best.json`
- `test_acc`: **0.9824**
- `test_macro_f1`: **0.9738**
- `test_loss`: 0.2982
- `test_samples`: 5782

Confusioni residue principali:
- Hitachi -> Mindray: 61
- Mindray -> ExactVu: 31
- Mindray -> Hitachi: 4

Per-classe più critiche:
- Hitachi recall: 0.9251
- Mindray f1: 0.7942

Nota: classi con supporto 0 in test (Alpinion, ExactVu, Sonostar, Vinno) non sono valutabili su quel test split.

---

## 4) Quick test manuale campionato

Test rapido su 120 immagini test random:
- CSV: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_no_negative_v2_power/quick_test_sample_predictions.csv`
- accuracy: **0.9750** (117/120)
- errori: 3

Overlay errori creati:
- cartella: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_no_negative_v2_power/quick_test_errors_overlay/`
- manifest: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_training_no_negative_v2_power/quick_test_errors_overlay/manifest.csv`

---

## 5) Decisione architetturale: predizione vendor a livello cartella

Requisito utente recepito:
> il `.fss` è unico per cartella, quindi il vendor deve essere unico per cartella.

### 5.1 Nuovo script operativo
Creato script dedicato:
- `/Users/camilla/Documents/Develop/ESIBuilder_AI/tools/predict_ultrasound_vendor_by_folder.py`

Funzione:
- predice vendor per immagine,
- aggrega per cartella (media probabilità + vote ratio),
- produce un solo vendor per cartella,
- marca `ok/review` con soglie conservative.

Soglie usate nel run:
- `min_folder_confidence = 0.55`
- `min_folder_margin = 0.10`
- `min_vote_ratio = 0.50`
- campionamento max immagini/cartella: `80`

### 5.2 Output folder-level (CNN)
Run output:
`/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/`

File principali:
- `folder_vendor_predictions.csv`
- `folder_vendor_predictions_review.csv`
- `folder_vendor_for_fss.csv`
- `folder_vendor_for_fss_recommended.csv`
- `per_image_predictions.csv`
- `summary.json`

Risultato:
- folders total: **426**
- `ok`: **414**
- `review`: **12**

---

## 6) Review visuali

Preview visive generate per tutte le 12 cartelle in review:
- cartella: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/review_previews/`
- contact sheet: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/review_previews/review_contact_sheet.png`
- manifest preview: `/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/review_previews/manifest.csv`

---

## 7) Rifinitura OCR sui soli casi review

Domanda verificata: “l’OCR avrebbe evitato gli errori?”

Verifica effettuata sui 12 review:
- review con almeno un hit OCR vendor: **7/12**
- review con hit OCR forte (soglia): **5/12**
- mismatch predizione-vs-name_hint: 6 casi, OCR utile solo in parte

### 7.1 Output OCR refine
- `folder_vendor_predictions_ocr_refined.csv`
- `folder_vendor_predictions_review_ocr_refined.csv`
- `folder_vendor_for_fss_recommended_ocr.csv`
- `summary_ocr_refine.json`

Percorso:
`/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/`

### 7.2 Policy finale raccomandata per `.fss`
File da usare adesso:
- **`/Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/vendor_folder_predictions_no_negative_v1/folder_vendor_for_fss_recommended_ocr.csv`**

Breakdown policy su 426 cartelle:
- `predicted_vendor`: 420
- `name_hint_override_on_review`: 5
- `ocr_and_name_hint_override_on_review`: 1

---

## 8) Significato operativo dei campi chiave

- `ok`: predizione vendor cartella affidabile.
- `review`: caso ambiguo (bassa confidenza, margine basso o voto inconsistente).
- `name_hint_vendor`: vendor inferito dal nome cartella (se riconoscibile).
- `recommended_*`: mapping finale conservativo da usare in produzione `.fss`.

---

## 9) Stato attuale (decisione)

La task “vendor recognition” è considerata **chiusa per lo scope attuale**, con deliverable operativo pronto:
- mapping per generazione `.fss` unico per cartella,
- casi review identificati e visualizzati,
- policy conservativa CNN + OCR + name-hint già applicata.

---

## 10) Prossimo step consigliato (non ancora eseguito)

Integrare questo blocco come primo stage di una pipeline più ampia:
1. vendor resolver folder-level (già fatto)
2. model/probe resolver per-vendor
3. estrazione campi `.fss`
4. builder `.fss` + validator compatibilità

