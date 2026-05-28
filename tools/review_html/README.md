# Review HTML Tools

Script per generare output di revisione visuale (HTML + overlay) utili al miglioramento dei training.

- `infer_ultrasound_rect_folder_html.py`: inferenza rect su cartella + review HTML per gruppo acquisizione.
- `build_ultrasound_rect_review_package.py`: costruisce pacchetto review HTML partendo da `predictions.csv`.
- `build_symbol_detection_review_package.py`: pacchetto review HTML completo per dataset simboli orientamento.
- `build_vendor_template_training_review_html.py`: review HTML per dataset vendor-template (bbox template vendor da linea 13).
- `prepare_vendor_template_retraining_from_review.py`: applica decisioni review (`escludere`, `modificare rettangolo`, keep), crea tool rettangolo e genera manifest finale per training.
- `build_vendor_template_single_image_rect_review_package.py`: crea review package con **1 immagine per cartella** (solo cartelle `modificare_rettangolo`) riusando il tool rect con autosave/export.

## Workflow review vendor-template -> training

1) Esporta le annotazioni dal review HTML (`vendor_template_flags.json` o `.csv`).
2) Genera manifest stage-1 (esclusioni) + tool modifica rettangolo:

```bash
python3 tools/review_html/prepare_vendor_template_retraining_from_review.py \
  --manifest artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv \
  --review-export /PATH/vendor_template_flags.json \
  --fallback-preview-index artifacts/40_outputs_eval/vendor_template_training_review_no_negative_train_line13/index.html \
  --output-dir artifacts/40_outputs_eval/vendor_template_retraining_from_review
```

Output principali:
- `manifest_after_exclusions.csv`: cartelle `escludere` rimosse.
- `rect_correction_tool/index.html`: tool visuale per cartelle `modificare_rettangolo`.

3) Nel tool rettangolo, disegna i nuovi bbox e fai `Export correzioni CSV` (`vendor_template_rect_corrections.csv`).
4) Applica le correzioni e genera il manifest finale:

```bash
python3 tools/review_html/prepare_vendor_template_retraining_from_review.py \
  --manifest artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv \
  --review-export /PATH/vendor_template_flags.json \
  --rect-corrections-csv /PATH/vendor_template_rect_corrections.csv \
  --output-dir artifacts/40_outputs_eval/vendor_template_retraining_from_review_final
```

Output finale:
- `manifest_training_ready.csv` (usa bbox corretti, `bbox_source=manual_review_rect` dove applicato).

## Modalita diretta (senza flag)

Se vuoi correggere i rettangoli direttamente con mouse drag, una immagine per cartella:

```bash
python3 tools/review_html/prepare_vendor_template_retraining_from_review.py \
  --manifest artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv \
  --rect-tool-all-folders \
  --rect-tool-max-samples 1 \
  --output-dir artifacts/40_outputs_eval/vendor_template_rect_direct_all_folders
```

Output:
- `rect_correction_tool/index.html` con autosave locale e export CSV (`vendor_template_rect_corrections.csv`)

## Variante consigliata: 1 immagine per cartella (autosave)

Genera package review con una sola immagine per ogni cartella da correggere:

```bash
python3 tools/review_html/build_vendor_template_single_image_rect_review_package.py \
  --manifest-after-exclusions artifacts/40_outputs_eval/vendor_template_retraining_from_review_real/manifest_after_exclusions.csv \
  --output-dir artifacts/40_outputs_eval/vendor_template_rect_single_image_review_real
```

Nel package esporti i file `corrections_*.csv` (uno per cartella) e poi li applichi al manifest finale propagando il box a tutta la cartella:

```bash
python3 tools/review_html/prepare_vendor_template_retraining_from_review.py \
  --manifest artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv \
  --review-export artifacts/40_outputs_eval/vendor_template_training_review_no_negative_train_line13/vendor_template_flags.json \
  --rect-corrections-dir artifacts/40_outputs_eval/vendor_template_rect_single_image_review_real/exports \
  --output-dir artifacts/40_outputs_eval/vendor_template_retraining_from_review_final
```
