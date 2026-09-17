# Tools - Compatibilita ESI

Questi script servono per dimostrare compatibilita tra output legacy e nuovo flusso.

## Requisiti

- Python 3.9+ (solo librerie standard)

## Struttura cartelle

- `tools/fss/`: utilita' compatibilita' e audit `.fss`
- `tools/ultrasound/`: dataset/training/inferenza vendor-probe-rect + pipeline head `.fss`
- `tools/line16/`: pipeline riga `#16` (template rect + parametri)
- `tools/orientation/`: pipeline orientamento simbolico e review GUI
- `tools/review_html/`: script per generare pacchetti/gallerie HTML di revisione
- `tools/old/`: script legacy/non raccomandati

## 1) Confronto `.fss`

Confronta due file `.fss`:

```bash
python3 tools/fss/compare_fss.py /path/legacy/setup_70.fss /path/new/setup_70.fss
```

Output:

- byte-identical: confronto testuale puro
- semantic-compatible: confronto token/numeri con tolleranza
- dettaglio campi diversi (#01..#26)

Codice di uscita:

- `0` se semanticamente compatibile
- `1` se incompatibile

## 2) Audit struttura output progetto

Verifica che per ogni `setup_<ID>.fss` siano presenti i template minimi attesi in `DB_echo/setup_<ID>/`.

```bash
python3 tools/fss/audit_setup_outputs.py /path/project_root
```

`project_root` deve contenere:

- `DB_setup/`
- `DB_echo/`

Codice di uscita:

- `0` se tutti i setup passano
- `1` se ci sono setup con file mancanti
- `2` se mancano cartelle/input fondamentali

## Nota

Questi strumenti sono il primo layer di quality gate per il progetto.
Nel tempo puoi estenderli con controlli su `.ndg`, `.grid`, `.orient` e `.freeze`.

## 3) Preparazione dataset rettangolo ecografico

Costruisce manifest unico da `Dataset/` leggendo il `RECT_ECHO` dal `.fss`
(riga 11, con fallback automatico a riga 10 per i file legacy) e crea split
bilanciato `train/val/test` a livello cartella (modello):

```bash
python3 tools/ultrasound/prepare_ultrasound_rect_dataset.py \
  --dataset-root Dataset \
  --output-dir artifacts/20_datasets/rect_dataset \
  --seed 42
```

Output principali:

- `artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv` (1 riga per immagine)
- `artifacts/20_datasets/rect_dataset/folders_rect_echo.csv` (1 riga per modello/cartella)
- `artifacts/20_datasets/rect_dataset/split_summary.{txt,json}`

Note:
- il manifest include anche i campi estratti dal nome file acquisizione
  (`capture_video_input`, `capture_video_x`, `capture_video_y`) usando il pattern
  `..._<vga|hdmi>_<WxH>...` quando presente.
- la ricerca avviene prima sui file di acquisizione della cartella modello
  (tutti i file tranne `image_samples/*`) e solo in fallback sui frame in
  `image_samples`.
- `capture_video_input_code_0hdmi_1vga` e il mapping pronto per la riga `#06` del `.fss`.

## 4) Training rete unica (bbox regressor)

Script PyTorch con augmentation simmetriche `left-right` e `up-down` con
trasformazione bbox coerente.

Esempio smoke test:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_rect_net.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --output-dir artifacts/30_models/rect_training_smoke \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 0 \
  --image-size 320 \
  --max-train-samples 1024 \
  --max-val-samples 256 \
  --max-test-samples 256
```

Run completo (consigliato, senza limiti campioni):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_rect_net.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --output-dir artifacts/30_models/rect_training_full \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 2 \
  --image-size 384
```

## 4b) Inferenza rettangolo (attuale: routing vendor)

Inferenza con routing vendor (`vendor -> rect-specialized`, fallback su globale):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/infer_ultrasound_rect_vendor_routing.py \
  --checkpoint-global artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt \
  --checkpoint-vendor artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt \
  --vendor-rect-map /path/to/vendor_rect_map.json \
  --manifest artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv \
  --output-dir artifacts/40_outputs_eval/rect_inference_vendor_routing_test \
  --split test \
  --vendor-conf-threshold 0.70 \
  --enable-vendor-ocr-lowconf \
  --vendor-ocr-min-hit-count 1 \
  --batch-size 64 \
  --num-workers 2 \
  --device mps
```

Con `--enable-vendor-ocr-lowconf`, OCR viene usato solo quando la confidenza
vendor CNN e sotto soglia (`--vendor-conf-threshold`), per proporre override
conservativo del vendor prima del routing del detector rect.

Nota legacy:
- runner globale baseline spostato in `tools/old/ultrasound/infer_ultrasound_rect_net.py`

Esempio `vendor_rect_map.json`:

```json
{
  "Mindray": "artifacts/30_models/rect_training_vendor_mindray/best_model.pt",
  "Philips": "artifacts/30_models/rect_training_vendor_philips/best_model.pt",
  "BK": "artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_vendor_bk/best_model.pt"
}
```

Output principali:

- `predictions_routed_<split>.csv`
- `summary_<split>.json`
- `per_true_vendor_<split>.csv`
- `routing_threshold_sweep.csv` (se generato da analisi successiva)
- `overlays/` (best/worst)

## 5) Training classificatore vendor (manufacturer)

Baseline `vendor-first` su frame HDMI: classifica la casa produttrice usando
`manufacturer` dal manifest.

Output principali:

- `artifacts/30_models/vendor_training/best_model.pt`
- `artifacts/30_models/vendor_training/metrics.json` (accuracy + macro-F1 train/val/test)
- `artifacts/30_models/vendor_training/test_per_manufacturer_metrics.csv`
- `artifacts/30_models/vendor_training/test_confusion_matrix.csv`

Esempio smoke test:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_vendor_classifier.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --output-dir artifacts/30_models/vendor_training_smoke \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 0 \
  --image-size 320 \
  --pretrained \
  --max-train-samples 2048 \
  --max-val-samples 512 \
  --max-test-samples 512
```

Run completo (consigliato):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_vendor_classifier.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --output-dir artifacts/30_models/vendor_training_full \
  --epochs 25 \
  --batch-size 48 \
  --num-workers 2 \
  --image-size 320 \
  --pretrained
```

## 6) Valutazione ibrida vendor (OCR + rete)

Valuta e fonde:

- probabilita del classificatore CNN (`best_model.pt`)
- probabilita OCR (Tesseract TSV + Naive Bayes su token)

La fusione usa:

`p_fused = alpha * p_cnn + (1 - alpha) * p_ocr`

con `alpha` ottimizzato su validation (macro-F1) o fissato manualmente.

Con `--enable-ocr-hint` puoi attivare anche una regola a soglia OCR per classi
specifiche (es. `Hitachi,Esaote`) con keyword manuali + keyword auto-derivate
dal modello OCR (`--ocr-hint-auto-topk`).

Output principali:

- `artifacts/40_outputs_eval/vendor_hybrid_eval/metrics.json`
- `artifacts/40_outputs_eval/vendor_hybrid_eval/fusion_alpha_search.csv`
- `artifacts/40_outputs_eval/vendor_hybrid_eval/hard_negatives_<classi>.csv`
- `artifacts/40_outputs_eval/vendor_hybrid_eval/test_per_manufacturer_metrics_{cnn,ocr,fused}.csv`
- `artifacts/40_outputs_eval/vendor_hybrid_eval/test_confusion_matrix_fused.csv`
- `artifacts/40_outputs_eval/vendor_hybrid_eval/ocr_cache.jsonl`

Smoke test:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/eval_ultrasound_vendor_hybrid.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --checkpoint artifacts/30_models/vendor_training_smoke_check/best_model.pt \
  --output-dir artifacts/40_outputs_eval/vendor_hybrid_smoke \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --max-test-samples 32 \
  --batch-size 16 \
  --num-workers 0 \
  --enable-ocr-hint \
  --ocr-hint-classes Hitachi,Esaote \
  --hard-negatives-classes Hitachi,Esaote \
  --ocr-cache-key folder \
  --ocr-log-interval 16
```

Run completo:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/eval_ultrasound_vendor_hybrid.py \
  --manifest artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv \
  --checkpoint artifacts/30_models/vendor_training_full_e1/best_model.pt \
  --output-dir artifacts/40_outputs_eval/vendor_hybrid_eval \
  --batch-size 48 \
  --num-workers 2 \
  --enable-ocr-hint \
  --ocr-hint-classes Hitachi,Esaote \
  --hard-negatives-classes Hitachi,Esaote \
  --ocr-cache-key folder \
  --ocr-log-interval 500
```

## 7) Predizione vendor unica per cartella (per `.fss` unico)

Quando una cartella deve ricevere un solo `.fss`, conviene assegnare un solo
vendor a livello cartella aggregando le probabilita su tutte (o molte)
immagini della cartella.

Output principali:

- `artifacts/40_outputs_eval/vendor_folder_predictions/folder_vendor_predictions.csv`
- `artifacts/40_outputs_eval/vendor_folder_predictions/folder_vendor_predictions_review.csv`
- `artifacts/40_outputs_eval/vendor_folder_predictions/folder_vendor_for_fss.csv`
- `artifacts/40_outputs_eval/vendor_folder_predictions/folder_vendor_for_fss_recommended.csv`
- `artifacts/40_outputs_eval/vendor_folder_predictions/per_image_predictions.csv`
- `artifacts/40_outputs_eval/vendor_folder_predictions/summary.json`

Esempio:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/predict_ultrasound_vendor_by_folder.py \
  --dataset-root Dataset \
  --checkpoint artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt \
  --output-dir artifacts/40_outputs_eval/vendor_folder_predictions \
  --sample-per-folder 80 \
  --batch-size 48 \
  --min-folder-confidence 0.55 \
  --min-folder-margin 0.10 \
  --min-vote-ratio 0.50
```

`folder_vendor_for_fss.csv` contiene la mappatura diretta `folder_path -> predicted_vendor`.

`folder_vendor_for_fss_recommended.csv` applica una policy conservativa: se una cartella
e `review` e il nome cartella suggerisce un vendor diverso (`name_hint`), usa il `name_hint`.
Questo aiuta a correggere mismatch evidenti a bassa confidenza prima della generazione `.fss`.

## 8) Training classificatore sonda (`fss_id_probe`)

Allena un classificatore che predice `fss_id_probe` dalle immagini HDMI.

Nota: nel manifest possono esistere probe ID presenti in val/test ma assenti in train.
Con `--drop-unseen-val-test` quei campioni vengono esclusi dalla valutazione.

Output principali:

- `artifacts/30_models/probe_training/best_model.pt`
- `artifacts/30_models/probe_training/metrics.json`
- `artifacts/30_models/probe_training/test_per_probe_metrics.csv`
- `artifacts/30_models/probe_training/test_confusion_matrix.csv`

Esempio:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_probe_classifier.py \
  --manifest artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv \
  --output-dir artifacts/30_models/probe_training_no_negative_v1 \
  --epochs 18 \
  --batch-size 48 \
  --num-workers 0 \
  --image-size 320 \
  --pretrained \
  --drop-unseen-val-test
```

## 9) Predizione sonda unica per cartella (per `.fss` unico)

Genera un solo `predicted_probe_id` per cartella aggregando le probabilita su piu immagini.

Output principali:

- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions_review.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_for_fss.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/summary.json`

Esempio:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/predict_ultrasound_probe_by_folder.py \
  --dataset-root Dataset \
  --checkpoint artifacts/10_active_pipeline/pipeline_fss_head/models/probe_training_no_negative_v1/best_model.pt \
  --output-dir artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1 \
  --sample-per-folder 80 \
  --batch-size 48
```

## 10) OCR refine sonda sui soli casi `review`

Refine conservativo: usa OCR solo sulle cartelle `review` della sonda.
Override della predizione CNN solo se:

- OCR supera soglie minime (`hit_count`, `hit_ratio`)
- (guard attivo di default) il probe OCR e presente nella top-k CNN della cartella

Output principali:

- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions_ocr_refined.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions_review_ocr_refined.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_for_fss_recommended_ocr.csv`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/summary_probe_ocr_refine.json`

Esempio:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/refine_ultrasound_probe_ocr.py \
  --folder-csv artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv \
  --manifest artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv \
  --output-dir artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1 \
  --ocr-max-images-per-folder 24 \
  --ocr-review-min-hit-count 2 \
  --ocr-review-min-hit-ratio 0.20
```

## 11) Predizione righe `.fss` head da acquisizioni raw

Genera automaticamente le righe chiave del `.fss` partendo da cartelle con soli frame acquisiti
(`timestamp_<vga|hdmi>_<WxH>...`), senza dipendere da `image_samples`.

Pipeline (stato attuale, transitorio):

1. raccolta frame acquisizione per cartella modello
2. deduplicazione esatta dei frame (stesso contenuto file)
3. stima rotazione classica `0/90/180/270` con OCR-OSD su campione dinamico (nessuna assunzione su numero frame)
4. applicazione rotazione selezionata in-memory a tutto il flusso di inferenza (vendor/probe/rect), con riporto coordinate rect nel sistema originale
5. campionamento uniforme dei frame unici per vendor/probe
6. inferenza vendor
7. compilazione `#13 RECT_NAME_ECHO` con resolver storico (fallback interattivo, senza rete dedicata)
8. inferenza `#03 ID_PROBE` con classificatore attuale (fallback interattivo)
9. compilazione `#14 RECT_NAME_PROBE` con resolver storico (fallback interattivo, senza rete dedicata)
10. uso di tutti i frame unici per rect (default, configurabile) + inferenza `#11` con routing vendor-specific opzionale (BK) e fallback globale
11. resolver righe `.fss` rimanenti

Pipeline target (prossimo step):

1. sostituire lo step 5 con rete dedicata di riconoscimento `RECT_NAME_ECHO`
2. sostituire lo step 7 con rete dedicata di riconoscimento `RECT_NAME_PROBE`
3. introdurre flusso probe dedicato successivo al classificatore attuale

Gestione confidenza bassa (`vendor/probe`), `#13/#14` non affidabili e `ID_ECHO` non risolto:

- `--low-confidence-policy ask_user` (default): prompt interattivo con top-k ordinato per confidenza.
- `ask_user` richiede esecuzione da terminale interattivo (TTY).
- se scegli `0) Altro`, inserisci valore manuale.
- i valori manuali vengono registrati e aggiunti su `encoding struct` (`--encoding-struct-path`), con fallback su CSV locale in output.
- `--low-confidence-policy error`: interrompe subito.
- `--low-confidence-policy review`: mantiene il comportamento a flag `review`.
- se il vendor e nuovo/non presente nello storico `#13`, non viene applicato fallback automatico: si segue la policy sopra.
- se la probe e nuova/non presente nello storico `#14`, non viene applicato fallback automatico: si segue la policy sopra.
- se la stima rotazione OSD non e affidabile (`osd_no_votes`, `osd_low_support`, `osd_unavailable`) la cartella viene marcata `review` con reason `rotation_not_reliable`.

Fail-fast:

- se l'inferenza vendor non produce probabilita valide, lo script interrompe l'esecuzione con errore (non salta la cartella).

Righe target:

- `#01 VERSION` (default fissato a `4.0`)
- `#02 ID_ECHO` (resolver da vendor/probe/video su manifest storico)
- `#03 ID_PROBE` (classificatore probe)
- `#06..#10` (video input e dimensioni da filename acquisizione)
- `#11 RECT_ECHO` (detector rettangolo)
- `#12 GROUP_ORIENTATION` (forzato a `4` = `symbol`)
- `#13 RECT_NAME_ECHO` (resolver storico da vendor/video con support score)
- `#14 RECT_NAME_PROBE` (resolver storico da vendor/probe/video con support score)

Output principali:

- `artifacts/10_active_pipeline/pipeline_fss_head/runs/fss_head_from_acquisitions/folder_fss_head_predictions.csv`
- `artifacts/10_active_pipeline/pipeline_fss_head/runs/fss_head_from_acquisitions/folder_fss_head_preview.txt`
- `artifacts/10_active_pipeline/pipeline_fss_head/runs/fss_head_from_acquisitions/summary.json`

Nel CSV trovi anche metriche dedup per cartella:
- `images_total_raw`
- `images_total` (frame unici usati nella pipeline)
- `images_duplicates_removed`

Metriche rotazione per cartella:
- `rotation_deg_clockwise`
- `rotation_source`
- `rotation_vote_ratio`
- `rotation_votes_total`
- `rotation_samples_checked`

Per `#11` trovi anche:
- `line_11_source` (`vendor_specialized`, `global`, `global_low_vendor_conf`)
- `line_11_model_checkpoint` (checkpoint realmente usato per la cartella)

Esempio:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/predict_fss_head_from_acquisitions.py \
  --dataset-root "Dataset L_T" \
  --vendor-checkpoint artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt \
  --probe-checkpoint artifacts/10_active_pipeline/pipeline_fss_head/models/probe_training_no_negative_v1/best_model.pt \
  --rect-checkpoint artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt \
  --rect-vendor-map artifacts/10_active_pipeline/pipeline_fss_head/maps/vendor_rect_map_bk_only.json \
  --rect-vendor-min-confidence 0.70 \
  --reference-manifest artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv \
  --output-dir artifacts/10_active_pipeline/pipeline_fss_head/runs/fss_head_from_acquisitions \
  --fss-version 4.0 \
  --low-confidence-policy ask_user \
  --vendor-min-confidence 0.50 \
  --probe-min-confidence 0.40 \
  --line13-min-support 0.00 \
  --line14-min-support 0.00 \
  --interactive-topk 3 \
  --encoding-struct-path artifacts/10_active_pipeline/pipeline_fss_head/runs/encoding_struct_fss_updates.csv \
  --batch-size 48 \
  --sample-per-folder 80 \
  --rotation-max-samples 24 \
  --rotation-min-votes 2 \
  --rotation-min-ratio 0.60 \
  --rect-sample-per-folder 0
```

## 7) Preparazione dataset riga #16 (orientamento + parametri)

Genera i manifest per la pipeline completa della riga #16:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/prepare_orientation_line16_dataset.py \
  --dataset-roots Dataset \
  --output-dir artifacts/20_datasets/orientation_line16_dataset \
  --seed 42
```

Output principali:

- `artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv`
- `artifacts/20_datasets/orientation_line16_dataset/manifest_line16_params.csv`
- `artifacts/20_datasets/orientation_line16_dataset/folders_line16.csv`
- `artifacts/20_datasets/orientation_line16_dataset/summary.{txt,json}`

## 8) Training detector template orientamento (fase 1)

`train_orientation_template_detector.py` è stato rimosso perché da rifare completamente.

Stato attuale:

- inferenza/eval su checkpoint esistenti rimangono disponibili
- per il retraining serve una nuova implementazione del trainer

## 9) Training parametri riga #16 (fase 2)

Addestra il modello multi-task per prevedere:

- classificazione: `B`, `CH`, `MM`
- regressione: `TH`, `P1`, `P2`, `P3`, `P4`, `P5`

Run consigliata:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/train_line16_params_model.py \
  --manifest artifacts/20_datasets/orientation_line16_dataset/manifest_line16_params.csv \
  --output-dir artifacts/30_models/line16_params_training_v2_nobalance \
  --epochs 400 \
  --batch-size 128 \
  --device cpu \
  --deduplicate \
  --disable-class-weights \
  --early-stopping-patience 70
```

Output principali:

- `artifacts/30_models/line16_params_training_v2_nobalance/best_model.pt`
- `artifacts/30_models/line16_params_training_v2_nobalance/metrics.json`
- `artifacts/30_models/line16_params_training_v2_nobalance/test_predictions.csv`

## 10) Inferenza end-to-end riga #16

Genera direttamente la stringa completa della riga #16 da una cartella setup
(`DB_setup` + `image_samples`):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/predict_line16_from_folder.py \
  --folder "Dataset/Alpinion E-Cube 9 Diamond sw 4.19.398, L3-12H" \
  --orientation-checkpoint artifacts/30_models/orientation_rect_training_v1/best_model.pt \
  --params-checkpoint artifacts/30_models/line16_params_training_v2_nobalance/best_model.pt \
  --output-json artifacts/40_outputs_eval/line16_predictions/alpinion_example.json
```

Output:

- stampa la riga #16 su stdout
- salva dettagli in JSON (rettangoli e parametri predetti per i 4 flip)

## 8b) Training orientamento per vendor + routing selettivo

Esegue training separati per vendor partendo dal manifest globale e produce:

- confronto `vendor model` vs `global model`
- mappa JSON `vendor -> checkpoint` solo per vendor che migliorano davvero

Comando unico consigliato (dopo disponibilita' del nuovo trainer):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/train_orientation_template_detector_per_vendor.py \
  --manifest artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv \
  --output-dir artifacts/30_models/orientation_rect_training_vendor \
  --trainer-script /path/to/new_train_orientation_template_detector.py \
  --global-metrics artifacts/30_models/orientation_rect_training_v1/test_metrics_recomputed.json \
  --epochs 24 \
  --batch-size 40 \
  --num-workers 2 \
  --image-size 320 \
  --pretrained \
  --early-stopping-patience 6 \
  --min-train-samples 40 \
  --min-val-samples 8 \
  --min-test-samples 8 \
  --min-delta-iou 0.01 \
  --min-test-samples-for-selection 16
```

Nota: se lanci lo script con un Python diverso, puoi forzare l'interprete del trainer con `--python-bin /path/to/python`.

Per lanciare solo alcuni vendor:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/train_orientation_template_detector_per_vendor.py \
  --manifest artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv \
  --output-dir artifacts/30_models/orientation_rect_training_vendor \
  --trainer-script /path/to/new_train_orientation_template_detector.py \
  --vendor Philips,Terason,Hitachi \
  --pretrained
```

File utili generati:

- `artifacts/30_models/orientation_rect_training_vendor/vendor_vs_global_summary.csv`
- `artifacts/30_models/orientation_rect_training_vendor/vendor_rect_map_selected.json`

`vendor_rect_map_selected.json` contiene solo i vendor che migliorano davvero
sul test rispetto al globale (routing selettivo).

Per usare il routing in inferenza riga #16:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/line16/predict_line16_from_folder.py \
  --folder "Dataset/Koelis, Koelis Probe - L" \
  --orientation-checkpoint artifacts/30_models/orientation_rect_training_v1/best_model.pt \
  --vendor-rect-map artifacts/30_models/orientation_rect_training_vendor/vendor_rect_map_selected.json \
  --params-checkpoint artifacts/30_models/line16_params_training_v2_nobalance/best_model.pt
```

## 11) Nuovo dataset simbolico orientamento (redesign)

Prepara un dataset coerente con il workflow ESIBuilder:

- `setup_orientation_manifest.csv`: 1 riga per setup x orientamento (`NF/LR/UD/LRUD`) con rect envelope da riga #16
- `frame_manifest.csv`: frame-level con `source_type` e `orientation_hint` da filename e/o da path (cartelle `NoFlip/LR/UD/LRUD`, anche annidate in `Dataset L_T`)
- `symbol_prior_manifest.csv`: prior esplicito dai file `image_th_orientation_<k>_positive_*`

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/orientation/prepare_orientation_symbolic_dataset.py \
  --dataset-roots Dataset "Dataset L_T" \
  --output-dir artifacts/20_datasets/orientation_symbolic_dataset \
  --folders-line16 artifacts/20_datasets/orientation_line16_dataset/folders_line16.csv
```

Nota: per disattivare i weak-label da cartella orientamento usa `--exclude-path-orientation-frames`.

Output principali:

- `artifacts/20_datasets/orientation_symbolic_dataset/setup_orientation_manifest.csv`
- `artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv`
- `artifacts/20_datasets/orientation_symbolic_dataset/symbol_prior_manifest.csv`
- `artifacts/20_datasets/orientation_symbolic_dataset/summary.{json,txt}`

## 12) Aggregazione envelope per orientamento (frame -> rect)

Aggrega detections frame-level in un box di raccoglimento per orientamento.

- se passi `--detections-csv`, usa le detection del tuo detector/classifier
- senza `--detections-csv`, usa fallback da `frame_manifest.csv` (solo per bootstrap)

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/orientation/aggregate_orientation_envelopes.py \
  --setup-orientation-manifest artifacts/20_datasets/orientation_symbolic_dataset/setup_orientation_manifest.csv \
  --frame-manifest-fallback artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv \
  --output-dir artifacts/99_misc/orientation_envelope_aggregation_fallback \
  --min-boxes-per-group 5 \
  --q-low 0.1 \
  --q-high 0.9 \
  --margin-px 2
```

Output principali:

- `aggregated_orientation_envelopes.csv`
- `line16_preview.csv`
- `summary.json`

## 13) Dataset simbolo orientamento (box piccolo)

Genera un dataset per detection del solo simbolo (box piccolo), separato dal
rettangolo envelope della riga #16.

Il tool:

- costruisce una bank di template da file `orientation_0..3.png`
- usa il rect grande della riga #16 solo come area di ricerca
- fa template matching NCC nel search area
- produce pseudo-label affidabili (`pseudo_box`) e casi da review (`review_no_match`)

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/orientation/prepare_orientation_symbol_detection_dataset.py \
  --frame-manifest artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv \
  --setup-orientation-manifest artifacts/20_datasets/orientation_symbolic_dataset/setup_orientation_manifest.csv \
  --dataset-roots Dataset "Dataset L_T" \
  --output-dir artifacts/20_datasets/orientation_symbol_detection_dataset_v2 \
  --min-match-score 0.45 \
  --search-margin-px 24
```

Output principali:

- `symbol_template_bank.csv`
- `symbol_detection_manifest.csv`
- `summary.{json,txt}`
- `previews/` (best/worst auto-accepted)

## 14) Revisione manuale completa del dataset simbolo

Crea un pacchetto review completo su **tutti** i campioni:

- overlay per ogni sample (coarse box + symbol box)
- `review_queue.csv` con colonne vuote `review_decision`, `review_notes`
- gallery HTML paginata (`index.html` + `pages/page_XXXX.html`)

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/review_html/build_symbol_detection_review_package.py \
  --manifest artifacts/20_datasets/orientation_symbol_detection_dataset_v2/symbol_detection_manifest.csv \
  --output-dir artifacts/20_datasets/orientation_symbol_detection_dataset_v2/review_package \
  --page-size 120 \
  --max-width 1100 \
  --jpeg-quality 82
```

Output principali:

- `review_package/index.html`
- `review_package/review_queue.csv`
- `review_package/overlays/*.jpg`
- `review_package/pages/page_XXXX.html`
- `review_package/summary.json`

## 15) Correzione manuale rect + orientamento (GUI interattiva)

Tool desktop (`tkinter`) per correggere i bounding box sbagliati e, quando
serve, correggere anche l'orientamento associato al sample.

Nuove funzioni UX:

- barra progresso + stato operativo in basso
- campo `Vai #` per saltare rapidamente a un indice
- pannello batch con `Cartella target` (sottocartella) e filtri:
  - `Solo stesso orientamento`
  - `Solo pending`
  - `Sovrascrivi rect gia corretti`
  - `Includi sottocartelle`
- pulsanti batch:
  - `Applica rect [A]` (propaga il rect corrente a tutta la sottocartella)
  - `Applica orientamento [B]` (propaga orientamento corrente + proposta rect da `.fss`)
- quando imposti orientamento (`1/2/3/4`) la GUI prova subito a proporre il `rect`
  da `.fss` (linea #16, entry dell'orientamento scelto)
- modalità annotazione separate:
  - `Simbolo (rosso)`: box piccolo del simbolo orientamento (annotazione principale)
  - `Area ricerca orientamento`: envelope dove il simbolo può comparire nelle immagini della stessa acquisizione e orientamento

Per il batch `rect`, il box viene trasferito in coordinate normalizzate:
stessa posizione relativa anche su immagini con risoluzioni diverse.

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/orientation/review_symbol_rects_gui.py \
  --manifest artifacts/20_datasets/orientation_symbol_detection_dataset_v2/symbol_detection_manifest.csv \
  --output-dir artifacts/20_datasets/orientation_symbol_detection_dataset_v2/manual_review
```

Shortcut principali:

- `drag` mouse: disegna/sposta/ridimensiona rect corretto
- `K`: decisione `keep`
- `C`: decisione `corrected`
- `X`: decisione `reject`
- `1/2/3/4`: override orientamento `NF/LR/UD/LRUD`
- `0`: reset override orientamento
- `F`: applica manualmente area ricerca da `.fss` per l'orientamento corrente
- `N/P`: prossimo/precedente sample
- `Space`: prossimo sample ancora pending
- `Delete`: cancella rect corretto
- `G`: vai all'indice scritto in `Vai #`
- `A`: applica rect corrente allo scope batch
- `B`: applica orientamento corrente allo scope batch

Output esportati da GUI:

- `<manifest>_manual_review.csv` (tutte le righe + decisioni + rect/orientamento finali)
- `<manifest>_train_ready.csv` (subset pronto training, esclusi `reject`)
- `review_state.json` (stato sessione per riprendere la review)

Nel CSV manual review trovi sia:
- `manual_symbol_rect_*` / `final_symbol_rect_*` per il simbolo
- `search_rect_*` / `final_search_rect_*` per l'area di ricerca orientamento

## 16) Dataset + training classificazione ecografie `L` vs `T`

Nuovo flusso per costruire dataset binario `L/T` direttamente da una sorgente
raw (es. volume `SSD_esi1_n3`) e allenare una rete unica generale.

### 16.1 Preparazione dataset da volume raw

Lo script scansiona ricorsivamente la root, etichetta le immagini usando token
`L/T` nei nomi cartella (con fallback keyword `lineare/transverse`), costruisce
gruppi per evitare leakage tra split e genera manifest + report bilanciamento.

Comando consigliato:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/prepare_ultrasound_lt_dataset.py \
  --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
  --output-dir artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3 \
  --seed 42
```

Output principali:

- `artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv`
- `artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/groups_lt.csv`
- `artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/summary_lt.{json,txt}`

### 16.1b Preview HTML filtrabile (`L/T`)

Genera una galleria HTML interattiva che legge il manifest e mostra le immagini
dal path originale (nessuna copia dati), con filtri:

- `Label`: `All`, `L`, `T`
- `Split`: `train`, `val`, `test`
- `Vendor` + campo ricerca testo

Modalità multi-sorgente (stessa HTML, switch da tendina `Source`):

- `Manifest label L/T`: etichetta da `manifest_lt.csv`
- `Probe model -> L/T prior`: etichetta da rete probe + prior storico tipo sonda
- con `Probe model` vedi anche:
  - legenda `Probe Type` (1=lineare, 2=convex, 3=transrettale L, 4=transrettale T)
  - filtro `Probe Type`
  - correzione manuale per card (`AUTO/L/T/MIXED/UNKNOWN/SKIP`)
  - export CSV delle correzioni (`Export correzioni CSV`)

Versione completa:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/build_lt_dataset_preview_html.py \
  --manifest artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv \
  --output-html artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/preview/preview_gallery.html \
  --max-rows 0
```

Versione multi-sorgente (manifest + probe-based):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/build_lt_dataset_preview_html.py \
  --manifest artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv \
  --include-probe-source \
  --probe-folder-predictions artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv \
  --probe-per-image-predictions artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv \
  --probe-list-csv artifacts/60_metadata/probe_list_from_encoding_struct_2026-03-30.csv \
  --probe-evidence-csv artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv \
  --probe-type-list-csv artifacts/60_metadata/probe_type_list_from_encoding_struct_2026-03-30.csv \
  --max-rows 12000 \
  --probe-max-rows 6000 \
  --probe-max-images-per-folder 6 \
  --output-html artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/preview/preview_gallery_multisource.html
```

Versione leggera (campione bilanciato):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/build_lt_dataset_preview_html.py \
  --manifest artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv \
  --output-html artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/preview/preview_gallery_10000.html \
  --max-rows 10000
```

### 16.1c Review HTML `modello sonda -> tipo`

Genera una pagina dedicata alla review dei modelli sonda (`probe_id/probe_name`)
con:

- evidenza tipo (`UNIVOCO/AMBIGUO/UNKNOWN` + distribuzione quote L/T)
- preview cartelle e immagini per ogni modello (path completo file/cartella)
- viewer fullscreen con scorrimento immagini della cartella (click thumbnail o bottone cartella)
- correzione manuale tipo + nota
- export CSV correzioni (`Export correzioni CSV`)
- pagina tabellare separata `probe -> tipo` (`probe_model_type_map.html`), con **tutte** le coppie `probe_id + type_id` (non solo tipo primario), quote e note

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/build_probe_model_type_review_html.py \
  --probe-summary-csv artifacts/60_metadata/probe_model_to_type_summary_2026-03-31.csv \
  --probe-type-list-csv artifacts/60_metadata/probe_type_list_from_encoding_struct_2026-03-30.csv \
  --probe-type-evidence-csv artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv \
  --folder-predictions-csv artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv \
  --per-image-csv artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/per_image_probe_predictions.csv \
  --dataset-root /Volumes/SSD_esi1_n1 \
  --max-folders-per-probe 4 \
  --max-images-per-folder 2 \
  --output-html artifacts/60_metadata/probe_model_type_review.html \
  --output-map-html artifacts/60_metadata/probe_model_type_map.html
```

### 16.1d Router `probe_id -> PROBETYPE` (linea #04 FSS)

Modulo operativo per derivare `#04 PROBETYPE` partendo da `#03 ID_PROBE`:

- `1` o `2` restano univoci
- presenza di tipo biplano (`3`/`4`) produce sempre gruppo `3-4` (serve seconda rete per split finale)

Singolo probe:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/probe_type_router.py \
  --probe-summary-csv artifacts/60_metadata/probe_model_to_type_summary_2026-03-31.csv \
  --probe-type-evidence-csv artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv \
  --probe-id 1 \
  --print-json
```

Annotazione batch (es. output classificatore probe per cartella):

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/probe_type_router.py \
  --input-csv artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv \
  --input-probe-column predicted_probe_id \
  --output-csv artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions_with_probe_type_router.csv
```

### 16.2 Training rete generale `L/T`

Addestra un classificatore binario (`L`,`T`) su tutto il dataset multi-vendor.
Nel report finale include anche metriche test per vendor, utili per verificare
se una rete generale basta o se serve specializzazione per vendor.

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_lt_classifier.py \
  --manifest artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv \
  --output-dir artifacts/30_models/lt_training_general \
  --epochs 20 \
  --batch-size 48 \
  --num-workers 2 \
  --image-size 320 \
  --pretrained
```

Output principali:

- `artifacts/30_models/lt_training_general/best_model.pt`
- `artifacts/30_models/lt_training_general/metrics.json`
- `artifacts/30_models/lt_training_general/test_per_lt_metrics.csv`
- `artifacts/30_models/lt_training_general/test_per_manufacturer_metrics.csv`

### 16.3 Confronto opzionale per-vendor

Se vuoi confrontare con training dedicato vendor, filtra il trainer:

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_ultrasound_lt_classifier.py \
  --manifest artifacts/20_datasets/lt_dataset_from_ssd_esi1_n3/manifest_lt.csv \
  --output-dir artifacts/30_models/lt_training_bk \
  --manufacturer BK \
  --epochs 20 \
  --batch-size 48 \
  --num-workers 2 \
  --image-size 320 \
  --pretrained
```

## 17) Template sonda (riga #14 `RECT_NAME_PROBE`)

### 17.1 Dataset con ground truth per-frame

La GT non viene copiata dal `.fss` su tutti i frame: il crop di riferimento
`DB_echo/<setup>/probe_name.png` viene ritrovato in ogni acquisizione con template
matching, quindi ogni immagine ha il suo box esatto. Relazione misurata fra `.fss`
e crop (valida su 368/390 configurazioni del volume `SSD_esi1_n1`):

```
crop top-left = (riga14_left + 3, riga14_top + 3)
crop size     = (larghezza_riga14 - 5, altezza_riga14 - 5)
```

I frame in cui il nome sonda non compare diventano negativi (box vuoto), utili per
insegnare alla rete ad astenersi.

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/prepare_probe_template_dataset.py \
  --dataset-root /Volumes/SSD_esi1_n1 \
  --output-dir artifacts/20_datasets/probe_template_line14_20260917 \
  --max-images-per-folder 250 \
  --workers 10
```

Output principali:

- `manifest_probe_template.csv` (1 riga per immagine: box ricostruito, box `.fss`, score, IoU)
- `folders_probe_template.csv` (1 riga per configurazione, con `gt_status` e `group_id`)
- `review_queue.csv` (configurazioni da rivedere: `fss_mismatch`, `low_hit_rate`, ...)
- `summary.json`, `split_summary.txt`

Gli split sono leak-free a livello di **layout**, non di cartella: le cartelle sono
unite in componenti connesse per "stessa UI" (vendor + risoluzione + box riga #13) e
"stesso nome macchina normalizzato", cosi le coppie L/T della stessa macchina non
finiscono in split diversi.

### 17.2 Training detector (heatmap CenterNet-style)

Il box e minuscolo (mediana 74x25 px su 1920x1080, 0,09% del frame): un regressore
bbox con pooling globale non ha la risoluzione spaziale per piazzarlo. Qui la rete e
ResNet18 + decoder FPN a stride 4 con teste heatmap + offset + size; il picco della
heatmap fornisce anche la confidenza. Nessuna augmentation di flip: il testo
specchiato non esiste in una UI reale.

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/ultrasound/train_probe_template_net.py \
  --manifest artifacts/20_datasets/probe_template_line14_20260917/manifest_probe_template.csv \
  --output-dir artifacts/30_models/probe_template_line14_20260917/generic \
  --image-size 512 --batch-size 16 --epochs 10 \
  --max-images-per-folder 40 --num-workers 6
```

Aggiungendo `--vendors Esaote` (o un elenco separato da virgole) si addestra la
variante per-vendor sugli stessi split, per il confronto generica vs per-vendor.

Output: `best_model.pt`, `metrics.json`, `predictions_test.csv`,
`folder_predictions_test.csv` (box mediano per cartella, IoU contro la riga #14 legacy
ed errore massimo per coordinata).

### 17.3 Review della ground truth segnalata

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/review_html/build_probe_template_review_html.py \
  --folders-csv artifacts/20_datasets/probe_template_line14_20260917/folders_probe_template.csv \
  --output artifacts/40_outputs_eval/probe_template_line14_review_20260917/index.html
```

Pagina autoconsistente: per ogni configurazione segnalata mostra lo zoom con il box
legacy (rosso) e quello ricostruito dal crop (verde), il crop di riferimento e i
numeri. Le decisioni si esportano in CSV dal browser, non viene scritto nulla sul
dataset.

### 17.4 Baseline del resolver storico (confronto)

```bash
cd tools/ultrasound && OldSoftwareEsiBuilder/.venv-mps/bin/python eval_line14_resolver_baseline.py \
  --folders-csv ../../artifacts/20_datasets/probe_template_line14_20260917/folders_probe_template.csv \
  --output-json ../../artifacts/40_outputs_eval/probe_template_line14_review_20260917/resolver_baseline_test.json
```

Riproduce la priorita di chiavi di `RectNameProbeResolver` (vendor+probe+video ->
vendor+probe -> probe+video -> probe, nessun fallback cieco) usando lo split train
come storico: misura quanto varrebbe oggi la riga #14 su macchine mai configurate.
