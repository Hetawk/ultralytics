#!/bin/bash
set -euo pipefail

PROJECT_DIR="/data2/enoch/ekd_coding_env/ultralytics"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs/robust_handoff"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ccts_to_tbcr_handoff.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

CCTS_RUN_TAG="${CCTS_RUN_TAG:-ccts_robust_v1}"
TBCR_RUN_TAG="${TBCR_RUN_TAG:-tbcr_legacy_v2}"
CCTS_DEPTH="${CCTS_DEPTH:-base}"
TBCR_DEPTH="${TBCR_DEPTH:-small}"
WAIT_FOR_CCTS="${WAIT_FOR_CCTS:-true}"
POLL_SECONDS="${POLL_SECONDS:-30}"

VARIANTS=(full no_def no_freq no_patch no_cbam baseline)

ccts_best_path() {
  local variant="$1"
  echo "$PROJECT_DIR/runs/classify/train_${CCTS_RUN_TAG}/ccts/${variant}_${CCTS_DEPTH}/weights/best.pt"
}

ensure_ccts_state_from_artifacts() {
  local state_dir="$PROJECT_DIR/logs/multi_gpu_training_ccts/state"
  local completed_file="$state_dir/completed.txt"
  local failed_file="$state_dir/failed.txt"
  local queue_file="$state_dir/queue.txt"
  local found=0

  mkdir -p "$state_dir"
  : > "$completed_file"
  : > "$failed_file"
  : > "$queue_file"

  for variant in "${VARIANTS[@]}"; do
    local key="${variant}:ccts"
    local best
    best=$(ccts_best_path "$variant")
    if [[ -f "$best" ]]; then
      echo "$key" >> "$completed_file"
      ((found++)) || true
      log "CCTS artifact OK: $key"
    else
      echo "$key" >> "$queue_file"
      log "CCTS artifact missing: $key"
    fi
  done

  log "CCTS reconciled from artifacts: completed=${found}/6"
}

ccts_training_active() {
  pgrep -f "train_meddef.py --data /data2/enoch/ekd_coding_env/meddef_winlab/processed_data/ccts" >/dev/null 2>&1
}

start_tbcr_robust() {
  log "Starting TBCR robust trainer"
  nohup env \
    RUN_TAG="$TBCR_RUN_TAG" DATASETS=tbcr DEPTH="$TBCR_DEPTH" EPOCHS=140 \
    BATCH=16 NBS=16 LR0=0.001 LRF=0.01 COS_LR=false PATIENCE=120 \
    DROPOUT=0.00 MIXUP=0.00 CUTMIX=0.00 CLASS_WEIGHTS=true \
    "$PROJECT_DIR/run/smart_multi_gpu_trainer_tbcr.sh" --resume \
    > "$PROJECT_DIR/logs/multi_gpu_training_tbcr/robust_launcher.log" 2>&1 < /dev/null &
}

main() {
  log "Handoff start: CCTS -> TBCR"

  # Avoid parallel stage automation while doing deterministic handoff.
  pkill -f '/robust_dataset_gatekeeper.sh' || true

  ensure_ccts_state_from_artifacts

  if ccts_training_active; then
    if [[ "${WAIT_FOR_CCTS,,}" != "true" ]]; then
      log "CCTS training still active. Not starting TBCR yet."
      exit 0
    fi

    log "CCTS training still active. Waiting for completion before TBCR launch..."
    while ccts_training_active; do
      sleep "$POLL_SECONDS"
      ensure_ccts_state_from_artifacts
      log "Waiting... CCTS process still running"
    done
    log "CCTS process stopped. Continuing handoff."
  fi

  # Clean stale CCTS scheduler to prevent duplicate relaunches.
  "$PROJECT_DIR/run/smart_multi_gpu_trainer_ccts.sh" --stop >/dev/null 2>&1 || true
  pkill -f '/smart_multi_gpu_trainer_ccts.sh --resume' || true

  local ccts_completed
  ccts_completed=$(wc -l < "$PROJECT_DIR/logs/multi_gpu_training_ccts/state/completed.txt" 2>/dev/null | tr -d ' ')
  if [[ "$ccts_completed" -lt 6 ]]; then
    log "CCTS not fully complete by artifacts (${ccts_completed}/6). TBCR launch skipped."
    exit 0
  fi

  mkdir -p "$PROJECT_DIR/logs/multi_gpu_training_tbcr"
  start_tbcr_robust

  sleep 5
  log "TBCR status snapshot:"
  "$PROJECT_DIR/run/smart_multi_gpu_trainer_tbcr.sh" --status | sed -n '1,80p' | tee -a "$LOG_FILE"
  log "Handoff finished."
}

main "$@"
