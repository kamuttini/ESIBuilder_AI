#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/camilla/Documents/Develop/ESIBuilder_AI"
cd "$ROOT"

MANIFEST="artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv"
RUN_ID="$(date +%Y%m%d_%H%M%S)_mps"
BASE_OUT="artifacts/30_models/full_training_no_negative_v1_${RUN_ID}"
mkdir -p "$BASE_OUT"

echo "[$(date '+%F %T')] Start full training run: $RUN_ID"
echo "Manifest: $MANIFEST"
echo "Output root: $BASE_OUT"

# 1) Vendor classifier
echo "[$(date '+%F %T')] Training vendor classifier..."
python3 -u tools/ultrasound/train_ultrasound_vendor_classifier.py \
  --manifest "$MANIFEST" \
  --device mps \
  --epochs 40 \
  --early-stopping-patience 10 \
  --batch-size 64 \
  --num-workers 6 \
  --pretrained \
  --log-interval 100 \
  --output-dir "$BASE_OUT/vendor_training"

# 2) Rect regressor
echo "[$(date '+%F %T')] Training rect regressor..."
python3 -u tools/ultrasound/train_ultrasound_rect_net.py \
  --manifest "$MANIFEST" \
  --device mps \
  --epochs 40 \
  --early-stopping-patience 10 \
  --batch-size 32 \
  --num-workers 6 \
  --pretrained \
  --log-interval 100 \
  --output-dir "$BASE_OUT/rect_training"

# 3) Probe classifier
echo "[$(date '+%F %T')] Training probe classifier..."
python3 -u tools/ultrasound/train_ultrasound_probe_classifier.py \
  --manifest "$MANIFEST" \
  --device mps \
  --epochs 40 \
  --early-stopping-patience 10 \
  --batch-size 64 \
  --num-workers 6 \
  --pretrained \
  --drop-unseen-val-test \
  --log-interval 100 \
  --output-dir "$BASE_OUT/probe_training"

echo "[$(date '+%F %T')] Full training run completed: $RUN_ID"
