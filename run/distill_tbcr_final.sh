#!/usr/bin/env bash
################################################################################
# MedDef2 — TBCR FINAL Stage-2 Defensive Distillation  [v2: T=50, α=0.9]
#
# Reuses the already-trained TBCR final checkpoints as BOTH:
#   1) student initialization (`--pretrained`)
#   2) frozen teacher checkpoint (`--teacher_model`)
#
# v1 weights (T=4, α=0.5) are preserved at distill_v1_T4_a0.5/ per variant.
# v2 writes to DISTILL_NAME (default: distill_v2) — nothing overwrites v1.
#
# Output layout for each variant:
#   runs/classify/train_tbcr_final/tbcr/<variant>_small/distill_v2/
#
# Features:
#   ✓ uses any free GPU from GPU_IDS
#   ✓ skips already-distilled variants
#   ✓ resumes if distill/weights/last.pt exists
#   ✓ survives terminal closure / SSH disconnect
#
# Usage:
#   bash run/distill_tbcr_final.sh
#   bash run/distill_tbcr_final.sh --status
#   bash run/distill_tbcr_final.sh --watch
#   bash run/distill_tbcr_final.sh --stop
#   bash run/distill_tbcr_final.sh --dry-run
################################################################################

set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
WHITE='\033[1;37m'; NC='\033[0m'; BOLD='\033[1m'

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MIN_MEMORY_MB="${MIN_MEMORY_MB:-5000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-72}"

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-224}"
DEPTH="${DEPTH:-small}"
LR0="${LR0:-0.0002}"
LRF="${LRF:-0.01}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1.0}"
WARMUP_BIAS_LR="${WARMUP_BIAS_LR:-0.1}"
# Disable early stopping by default for stage-2 distillation. With PATIENCE=20
# the trainer exits cleanly long before 100 epochs, and the scheduler used to
# re-queue those runs, causing confusing epoch "restarts".
PATIENCE="${PATIENCE:-0}"
DROPOUT="${DROPOUT:-0.0}"
ERASING="${ERASING:-0.2}"
MIXUP="${MIXUP:-0.0}"
CUTMIX="${CUTMIX:-0.0}"
COS_LR="${COS_LR:-false}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"
WORKERS="${WORKERS:-8}"

# Defensive distillation temperature and alpha
# v1 used T=4 α=0.5 (standard knowledge distillation effect)
# v2 uses T=50 α=0.9 (true gradient-masking defensive distillation)
DIST_TEMP="${DIST_TEMP:-50.0}"
DIST_ALPHA="${DIST_ALPHA:-0.9}"

DATA_ROOT="${DATA_ROOT:-/data2/enoch/ekd_coding_env/meddef_winlab/processed_data}"
TBCR_DATA="${TBCR_DATA:-${DATA_ROOT}/tbcr}"
RUN_NAME="${RUN_NAME:-train_tbcr_final}"
DISTILL_NAME="${DISTILL_NAME:-distill_v2}"

PYTHON="${PYTHON:-python}"
VENV="${VENV:-/data2/enoch/.virtualenvs/meddef_final/bin/activate}"

VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"

LOG_BASE="$PROJECT_DIR/logs/tbcr_final_${DISTILL_NAME}"
MODEL_LOG_DIR="$LOG_BASE/model_logs"
JOB_DIR="$LOG_BASE/jobs"
STATE_DIR="$LOG_BASE/state"
LOCK_DIR="$LOG_BASE/locks"
MASTER_LOG="$LOG_BASE/master.log"
QUEUE_FILE="$STATE_DIR/queue.txt"
COMPLETED_FILE="$STATE_DIR/completed.txt"
FAILED_FILE="$STATE_DIR/failed.txt"
PID_FILE="$LOG_BASE/scheduler.pid"
mkdir -p "$LOG_BASE" "$MODEL_LOG_DIR" "$JOB_DIR" "$STATE_DIR" "$LOCK_DIR"

_log() {
  local lvl=$1 col=$2; shift 2
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$lvl] $*"
  echo -e "${col}${msg}${NC}"
  echo "$msg" >> "$MASTER_LOG"
}
log_info() { _log "INFO" "$BLUE" "$@"; }
log_ok()   { _log "OK" "$GREEN" "$@"; }
log_warn() { _log "WARN" "$YELLOW" "$@"; }
log_err()  { _log "ERROR" "$RED" "$@"; }
log_gpu()  { local g=$1; shift; _log "GPU $g" "$MAGENTA" "$@"; }

get_free_memory()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' \t\n' || echo 0; }
get_process_count() { nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null | grep -c '[0-9]' 2>/dev/null || true; }

is_gpu_free() {
  # For distillation we only want truly idle GPUs.  This avoids launching a
  # second training job onto a GPU that already has an active train_meddef
  # process (especially after a scheduler restart) and prevents avoidable OOMs.
  local gpu=$1
  local free procs
  free=$(get_free_memory "$gpu")
  procs=$(get_process_count "$gpu")
  free=${free:-0}
  procs=${procs:-0}
  [[ "$free" =~ ^[0-9]+$ ]] || return 1
  [[ "$procs" =~ ^[0-9]+$ ]] || return 1
  [[ "$free" -ge "$MIN_MEMORY_MB" && "$procs" -eq 0 ]]
}

acquire_lock() { mkdir "$LOCK_DIR/gpu$1.lock" 2>/dev/null; }
release_lock() { rmdir "$LOCK_DIR/gpu$1.lock" 2>/dev/null || true; }

teacher_pt_for_variant() {
  local variant=$1
  echo "$PROJECT_DIR/runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}/weights/best.pt"
}

distill_dir_for_variant() {
  local variant=$1
  echo "$PROJECT_DIR/runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}/${DISTILL_NAME}"
}

last_pt_for_variant() {
  local variant=$1
  echo "$(distill_dir_for_variant "$variant")/weights/last.pt"
}

best_pt_for_variant() {
  local variant=$1
  echo "$(distill_dir_for_variant "$variant")/weights/best.pt"
}

variant_has_teacher() {
  local variant=$1
  [[ -f "$(teacher_pt_for_variant "$variant")" ]]
}

variant_has_checkpoint() {
  local variant=$1
  [[ -f "$(last_pt_for_variant "$variant")" ]]
}

variant_progress_epoch() {
  local variant=$1
  local last_pt results max_epoch ckpt_epoch
  last_pt="$(last_pt_for_variant "$variant")"

  if [[ -f "$last_pt" ]]; then
    ckpt_epoch=$(python3 - "$last_pt" <<'PY'
import sys
from pathlib import Path

try:
    import torch
except Exception:
    print(-1)
    raise SystemExit(0)

pt = Path(sys.argv[1])
try:
    ckpt = torch.load(pt, map_location="cpu")
    epoch = ckpt.get("epoch", -1)
    print(int(epoch) if epoch is not None else -1)
except Exception:
    print(-1)
PY
)
    if [[ "$ckpt_epoch" =~ ^-?[0-9]+$ ]] && [[ "$ckpt_epoch" -ge 0 ]]; then
      echo "$ckpt_epoch"
      return 0
    fi
  fi

  results="$(distill_dir_for_variant "$variant")/results.csv"
  [[ -f "$results" ]] || { echo -1; return 0; }
  max_epoch=$(awk -F',' 'NR > 1 && $1 ~ /^[0-9]+$/ { if ($1 > max) max = $1 } END { print (max == "" ? -1 : max) }' "$results" 2>/dev/null)
  echo "${max_epoch:--1}"
}

variant_is_done() {
  local variant=$1
  local best last max_epoch
  best="$(best_pt_for_variant "$variant")"
  last="$(last_pt_for_variant "$variant")"
  [[ -f "$best" || -f "$last" ]] || return 1
  max_epoch=$(variant_progress_epoch "$variant")
  [[ "$max_epoch" =~ ^-?[0-9]+$ ]] || return 1
  [[ "$max_epoch" -ge $((EPOCHS - 1)) ]]
}

variant_is_running() {
  local variant=$1
  pgrep -af "python train_meddef.py .*--variant ${variant}( |$).*--name ${DISTILL_NAME}( |$)" >/dev/null 2>&1
}

queue_contains() {
  grep -qxF "$1" "$QUEUE_FILE" 2>/dev/null
}

enqueue_unique() {
  local variant=$1
  variant_is_done "$variant" && return 0
  queue_contains "$variant" && return 0
  echo "$variant" >> "$QUEUE_FILE"
}

prune_queue_file() {
  [[ -f "$QUEUE_FILE" ]] || return 0
  awk 'NF && !seen[$0]++' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" 2>/dev/null && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
}

is_completed() { grep -qxF "$1" "$COMPLETED_FILE" 2>/dev/null; }
unmark_completed() {
  sed -i "/^$1$/d" "$COMPLETED_FILE" 2>/dev/null || true
}
mark_completed() {
  grep -qxF "$1" "$COMPLETED_FILE" 2>/dev/null || echo "$1" >> "$COMPLETED_FILE"
  sed -i "/^$1$/d" "$FAILED_FILE" 2>/dev/null || true
}
mark_failed() {
  grep -qxF "$1" "$FAILED_FILE" 2>/dev/null || echo "$1" >> "$FAILED_FILE"
}

declare -A GPU_PIDS
declare -A GPU_VARIANTS
declare -A GPU_START

launch_variant() {
  local variant=$1 gpu=$2
  local teacher_pt="$(teacher_pt_for_variant "$variant")"
  local variant_dir="runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}"
  local log_file="$MODEL_LOG_DIR/${variant}.log"
  local resume_flag=""
  local resume_note="fresh stage-2 distillation"

  if [[ ! -f "$teacher_pt" ]]; then
    log_err "Skipping $variant: teacher checkpoint missing at $teacher_pt"
    mark_failed "$variant"
    return 0
  fi

  local last_pt="$(last_pt_for_variant "$variant")"
  if variant_has_checkpoint "$variant"; then
    resume_flag="--resume $last_pt"
    resume_note="RESUMING stage-2 distill (from $last_pt)"
  fi

  log_gpu "$gpu" "Launching ${BOLD}${variant}${NC} (${resume_note})"

  local cmd=(
    "$PYTHON" train_meddef.py
    --data "$TBCR_DATA"
    --variant "$variant"
    --depth "$DEPTH"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --imgsz "$IMGSZ"
    --device "$gpu"
    --workers "$WORKERS"
    --lr0 "$LR0"
    --lrf "$LRF"
    --optimizer "$OPTIMIZER"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_epochs "$WARMUP_EPOCHS"
    --warmup_bias_lr "$WARMUP_BIAS_LR"
    --patience "$PATIENCE"
    --dropout "$DROPOUT"
    --erasing "$ERASING"
    --mixup "$MIXUP"
    --cutmix "$CUTMIX"
    --cos_lr "$COS_LR"
    --save_period "$SAVE_PERIOD"
    --project "$variant_dir"
    --name "$DISTILL_NAME"
    --pretrained "$teacher_pt"
    --teacher_model "$teacher_pt"
    --def_distill
    --dist_temp "$DIST_TEMP"
    --dist_alpha "$DIST_ALPHA"
    --exist_ok
    $resume_flag
  )

  if [[ "${_DRY_RUN:-0}" -eq 1 ]]; then
    log_info "[DRY-RUN] would run: ${cmd[*]}"
    (sleep 2 && echo "completed" > "$JOB_DIR/gpu${gpu}.status") &
  else
    (
      trap '' INT TERM HUP
      [[ -f "$VENV" ]] && source "$VENV"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${cmd[*]}" >> "$log_file"
      "${cmd[@]}" >> "$log_file" 2>&1
      rc=$?
      if [[ $rc -eq 0 ]]; then
        echo "completed" > "$JOB_DIR/gpu${gpu}.status"
      else
        echo "failed:$rc" > "$JOB_DIR/gpu${gpu}.status"
      fi
    ) &
  fi

  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" > "$JOB_DIR/gpu${gpu}.pid"
  echo "running" > "$JOB_DIR/gpu${gpu}.status"
  GPU_PIDS[$gpu]=$pid
  GPU_VARIANTS[$gpu]=$variant
  GPU_START[$gpu]=$(date +%s)
  log_gpu "$gpu" "PID=$pid  variant=$variant"
  release_lock "$gpu"
  sleep 2
}

harvest() {
  IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
  for gpu in "${GPU_ARR[@]}"; do
    [[ -z "${GPU_PIDS[$gpu]:-}" ]] && continue
    local pid="${GPU_PIDS[$gpu]}"
    kill -0 "$pid" 2>/dev/null && continue

    local variant="${GPU_VARIANTS[$gpu]:-unknown}"
    local status_file="$JOB_DIR/gpu${gpu}.status"
    local status; status=$(cat "$status_file" 2>/dev/null || echo "unknown")

    if [[ "$status" == "completed" ]]; then
      if [[ "${_DRY_RUN:-0}" -eq 1 ]] || variant_is_done "$variant"; then
        log_ok "GPU $gpu: ✓ $variant distillation complete"
        mark_completed "$variant"
      elif [[ -f "$(last_pt_for_variant "$variant")" || -f "$(best_pt_for_variant "$variant")" ]]; then
        local progress_epoch
        progress_epoch=$(variant_progress_epoch "$variant")
        unmark_completed "$variant"
        log_info "GPU $gpu: $variant exited cleanly at epoch $((progress_epoch + 1))/$EPOCHS; re-queuing to continue toward the target"
        enqueue_unique "$variant"
      else
        log_warn "GPU $gpu: $variant exited 0 but no distilled checkpoint found"
        mark_failed "$variant"
      fi
    else
      log_err "GPU $gpu: ✗ $variant failed (status=$status)"
      mark_failed "$variant"
    fi

    unset GPU_PIDS[$gpu] GPU_VARIANTS[$gpu] GPU_START[$gpu]
    rm -f "$JOB_DIR/gpu${gpu}.pid" "$JOB_DIR/gpu${gpu}.status"
  done
}

queue_remaining() {
  : > "$QUEUE_FILE"
  for variant in $VARIANTS; do
    if variant_is_done "$variant"; then
      mark_completed "$variant"
      continue
    fi
    unmark_completed "$variant"
    if variant_is_running "$variant"; then
      continue
    fi
    enqueue_unique "$variant"
  done
  prune_queue_file
}

show_status() {
  local total=0 done=0 failed=0 running=0 pending=0
  echo
  echo "Stage-2 TBCR distillation status"
  echo "RUN_NAME=$RUN_NAME  DEPTH=$DEPTH  DISTILL_NAME=$DISTILL_NAME"
  echo "GPU_IDS=$GPU_IDS"
  echo
  for variant in $VARIANTS; do
    total=$((total + 1))
    local progress_epoch
    progress_epoch=$(variant_progress_epoch "$variant")
    if variant_is_done "$variant"; then
      printf "  %-10s  %bDONE%b      epoch=%s/%s  %s\n" "$variant" "$GREEN" "$NC" "$((progress_epoch + 1))" "$EPOCHS" "$(best_pt_for_variant "$variant")"
      done=$((done + 1))
    elif variant_has_checkpoint "$variant"; then
      printf "  %-10s  %bRESUME%b    epoch=%s/%s  %s\n" "$variant" "$YELLOW" "$NC" "$((progress_epoch + 1))" "$EPOCHS" "$(last_pt_for_variant "$variant")"
      pending=$((pending + 1))
    elif variant_has_teacher "$variant"; then
      printf "  %-10s  %bREADY%b     teacher=%s\n" "$variant" "$CYAN" "$NC" "$(teacher_pt_for_variant "$variant")"
      pending=$((pending + 1))
    else
      printf "  %-10s  %bMISSING%b   teacher=%s\n" "$variant" "$RED" "$NC" "$(teacher_pt_for_variant "$variant")"
      failed=$((failed + 1))
    fi
  done
  echo
  echo "Summary: $done done / $failed missing-failed / $pending pending / $total total"
  echo
}

watch_status() {
  while true; do
    clear
    show_status
    echo "Watching... press Ctrl+C to stop"
    sleep 5
  done
}

stop_scheduler() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      log_ok "Stopped scheduler PID=$pid"
    fi
    rm -f "$PID_FILE"
  fi
}

# Modes
case "${1:-}" in
  --status|status) show_status; exit 0 ;;
  --watch|watch) watch_status; exit 0 ;;
  --stop|stop) stop_scheduler; exit 0 ;;
  --dry-run|dry-run) _DRY_RUN=1 ;;
esac

# Self-daemonize unless already in background or dry-run.
if [[ "${MEDDEF_DISTILL_DAEMON:-0}" != "1" && "${_DRY_RUN:-0}" != "1" ]]; then
  echo "[INFO] Starting distillation scheduler in background..."
  nohup env MEDDEF_DISTILL_DAEMON=1 bash "$SCRIPT_PATH" "$@" >> "$LOG_BASE/nohup.out" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[INFO] Scheduler PID: $(cat "$PID_FILE")"
  echo "[INFO] Check status with: bash run/distill_tbcr_final.sh --status"
  exit 0
fi

echo $$ > "$PID_FILE"
queue_remaining
log_info "Starting TBCR stage-2 distillation scheduler"

while true; do
  harvest

  any_running=0
  for pid in "${GPU_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      any_running=1
      break
    fi
  done

  if [[ ! -s "$QUEUE_FILE" && "$any_running" -eq 0 ]]; then
    log_ok "All eligible TBCR distillation jobs completed."
    break
  fi

  if [[ -s "$QUEUE_FILE" ]]; then
    prune_queue_file
    IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
    for gpu in "${GPU_ARR[@]}"; do
      [[ -n "${GPU_PIDS[$gpu]:-}" ]] && kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null && continue
      is_gpu_free "$gpu" || continue
      acquire_lock "$gpu" || continue

      next_variant=""
      while [[ -s "$QUEUE_FILE" ]]; do
        local_candidate=$(head -1 "$QUEUE_FILE" 2>/dev/null || true)
        sed -i '1d' "$QUEUE_FILE" 2>/dev/null || true
        [[ -n "$local_candidate" ]] || continue
        if variant_is_done "$local_candidate"; then
          mark_completed "$local_candidate"
          continue
        fi
        if variant_is_running "$local_candidate"; then
          continue
        fi
        next_variant="$local_candidate"
        break
      done

      [[ -n "$next_variant" ]] || { release_lock "$gpu"; continue; }
      launch_variant "$next_variant" "$gpu"
    done
  fi

  sleep "$CHECK_INTERVAL"
done

rm -f "$PID_FILE"
show_status
