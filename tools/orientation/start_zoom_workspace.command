#!/bin/zsh
set -euo pipefail

ROOT="/Users/camilla/Documents/Develop/ESIBuilder_AI"
PY="/opt/anaconda3/bin/python"
BACKEND="$ROOT/tools/orientation/zoom_workspace_backend.py"
HTML="$ROOT/artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/zoom_workspace_interactive/index.html"
RECT_CKPT="$ROOT/artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"
VENDOR_CKPT="$ROOT/artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"
RECT_VENDOR_MAP="$ROOT/artifacts/10_active_pipeline/pipeline_fss_head/maps/vendor_rect_map_bk_only.json"
ORIENT_CKPT="$ROOT/artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"
HOST="127.0.0.1"
PORT="8765"
URL="http://${HOST}:${PORT}"
LOG="/tmp/zoom_workspace_backend_8765.log"
PID_FILE="/tmp/zoom_workspace_backend_8765.pid"

OLD_PID="$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null || true)"
if [[ -n "${OLD_PID}" ]]; then
  echo "[zoom-workspace] chiudo backend precedente (pid=${OLD_PID})"
  kill "${OLD_PID}" >/dev/null 2>&1 || true
  sleep 1
fi

echo "[zoom-workspace] avvio backend su ${URL} ..."
nohup "$PY" "$BACKEND" \
  --host "$HOST" \
  --port "$PORT" \
  --html "$HTML" \
  --checkpoint "$RECT_CKPT" \
  --vendor-checkpoint "$VENDOR_CKPT" \
  --rect-vendor-map "$RECT_VENDOR_MAP" \
  --orientation-checkpoint "$ORIENT_CKPT" \
  >"$LOG" 2>&1 &
echo $! > "$PID_FILE"
sleep 2

if curl -s --max-time 5 "${URL}/health" >/dev/null 2>&1; then
  echo "[zoom-workspace] backend OK -> ${URL}"
  open "${URL}" || true
else
  echo "[zoom-workspace] backend non raggiungibile. Log:"
  tail -n 80 "$LOG" || true
  exit 1
fi
