#!/usr/bin/env bash
################################################################################
# MedDef2 — TBCR FINAL Training Run
# Best hyperparameters from all previous ablation experiments.
#
# Features:
#   ✓ Trains all 6 variants in parallel across all free GPUs
#   ✓ Auto-detects free GPUs — starts on any GPU that becomes available
#   ✓ Auto-resume: if last.pt exists, resumes from where it left off
#   ✓ save_period=10: intermediate checkpoints every 10 epochs
#   ✓ State persistence: survives terminal closure / SSH disconnect
#   ✓ Re-runnable: already-completed variants are skipped automatically
#   ✓ --install-cron: registers itself as a @reboot job so training
#       restarts after server reboot and continues on free GPUs
#   ✓ Dashboard, status, live log tail modes
#
# Usage:
#   bash run/train_tbcr_final.sh                  # start / resume all
#   bash run/train_tbcr_final.sh --status         # check status
#   bash run/train_tbcr_final.sh --watch          # live dashboard
#   bash run/train_tbcr_final.sh --live           # tail all training logs
#   bash run/train_tbcr_final.sh --stop           # stop the scheduler
#   bash run/train_tbcr_final.sh --clean          # wipe state and restart
#   bash run/train_tbcr_final.sh --install-cron   # auto-start on reboot
#   bash run/train_tbcr_final.sh --remove-cron    # remove reboot hook
#   bash run/train_tbcr_final.sh --dry-run        # simulate (no actual training)
#
# Override any param via env:
#   GPU_IDS=2,3 bash run/train_tbcr_final.sh
#   EPOCHS=200  bash run/train_tbcr_final.sh
################################################################################

set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
WHITE='\033[1;37m'; NC='\033[0m'; BOLD='\033[1m'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all overridable via environment)
# ─────────────────────────────────────────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0,1,2,3}"          # comma-separated GPU indices to use
MIN_MEMORY_MB="${MIN_MEMORY_MB:-5000}" # minimum free VRAM before launching a job
CHECK_INTERVAL="${CHECK_INTERVAL:-20}" # seconds between scheduler ticks
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-240}"

# ── Best hyperparameters (from ALL_RUNS_HYPERPARAMS_SUMMARY.txt) ─────────────
EPOCHS="${EPOCHS:-160}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-224}"
DEPTH="${DEPTH:-small}"
LR0="${LR0:-0.0008}"
LRF="${LRF:-0.01}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3.0}"
WARMUP_BIAS_LR="${WARMUP_BIAS_LR:-0.1}"
PATIENCE="${PATIENCE:-140}"
DROPOUT="${DROPOUT:-0.0}"
ERASING="${ERASING:-0.4}"
MIXUP="${MIXUP:-0.0}"
CUTMIX="${CUTMIX:-0.0}"
COS_LR="${COS_LR:-false}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"       # save epoch checkpoint every N epochs
WORKERS="${WORKERS:-8}"

# ── Dataset & output ──────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/data2/enoch/ekd_coding_env/meddef_winlab/processed_data}"
TBCR_DATA="${TBCR_DATA:-${DATA_ROOT}/tbcr}"
RUN_NAME="${RUN_NAME:-train_tbcr_final}"  # output → runs/classify/train_tbcr_final/tbcr/<variant>/

# ── Python / venv ──────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
VENV="${VENV:-/data2/enoch/.virtualenvs/meddef_final/bin/activate}"

# ── Variants to train ─────────────────────────────────────────────────────
VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
LOG_BASE="$PROJECT_DIR/logs/tbcr_final"
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

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
_log() {
  local lvl=$1 col=$2; shift 2
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$lvl] $*"
  echo -e "${col}${msg}${NC}"
  echo "$msg" >> "$MASTER_LOG"
}
log_info()    { _log "INFO"    "$BLUE"    "$@"; }
log_ok()      { _log "OK"      "$GREEN"   "$@"; }
log_warn()    { _log "WARN"    "$YELLOW"  "$@"; }
log_err()     { _log "ERROR"   "$RED"     "$@"; }
log_gpu()     { local g=$1; shift; _log "GPU $g" "$MAGENTA" "$@"; }

# ─────────────────────────────────────────────────────────────────────────────
# GPU HELPERS
# ─────────────────────────────────────────────────────────────────────────────
get_free_memory()    { nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' \t\n' || echo 0; }
get_gpu_util()       { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' \t\n' || echo 0; }
get_gpu_temp()       { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' \t\n' || echo 0; }
# wc -l always exits 0 — avoids the grep -c / || double-output bug
get_process_count()  { nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null | grep -c '[0-9]' 2>/dev/null; local rc=$?; [[ $rc -eq 0 || $rc -eq 1 ]] || echo 0; }

is_gpu_free() {
  local gpu=$1
  local free procs
  free=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' \t\n')
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | wc -l | tr -d ' \t\n')
  free=${free:-0}; procs=${procs:-0}
  [[ "$free" =~ ^[0-9]+$ ]] || return 1
  [[ "$procs" =~ ^[0-9]+$ ]] || return 1
  [[ "$free" -ge "$MIN_MEMORY_MB" && "$procs" -eq 0 ]]
}

acquire_lock() {
  local gpu=$1 lf="$LOCK_DIR/gpu${1}.lock"
  mkdir "$lf" 2>/dev/null
}
release_lock() { rmdir "$LOCK_DIR/gpu${1}.lock" 2>/dev/null || true; }

# ─────────────────────────────────────────────────────────────────────────────
# STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
is_completed() { grep -qxF "$1" "$COMPLETED_FILE" 2>/dev/null; }
is_failed()    { grep -qxF "$1" "$FAILED_FILE"    2>/dev/null; }

mark_completed() {
  grep -qxF "$1" "$COMPLETED_FILE" 2>/dev/null || echo "$1" >> "$COMPLETED_FILE"
  sed -i "/^$1$/d" "$FAILED_FILE"   2>/dev/null || true
  sed -i "/^$1$/d" "$QUEUE_FILE"    2>/dev/null || true
}
mark_failed() {
  grep -qxF "$1" "$FAILED_FILE" 2>/dev/null || echo "$1" >> "$FAILED_FILE"
  sed -i "/^$1$/d" "$QUEUE_FILE" 2>/dev/null || true
}

# Dequeue first item from queue file
dequeue_next() {
  # Sets global _NEXT_VARIANT
  _NEXT_VARIANT=""
  [[ -f "$QUEUE_FILE" ]] || return
  _NEXT_VARIANT=$(head -1 "$QUEUE_FILE" 2>/dev/null)
  [[ -n "$_NEXT_VARIANT" ]] && sed -i "1d" "$QUEUE_FILE" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT DETECTION — for auto-resume
# ─────────────────────────────────────────────────────────────────────────────
last_pt_for_variant() {
  local variant=$1
  echo "$PROJECT_DIR/runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}/weights/last.pt"
}

best_pt_for_variant() {
  local variant=$1
  echo "$PROJECT_DIR/runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}/weights/best.pt"
}

variant_has_checkpoint() {
  local variant=$1
  [[ -f "$(last_pt_for_variant "$variant")" ]]
}

variant_is_done() {
  # done = best.pt exists AND results.csv has >= EPOCHS lines (past header)
  local variant=$1
  local best; best="$(best_pt_for_variant "$variant")"
  local results; results="$PROJECT_DIR/runs/classify/${RUN_NAME}/tbcr/${variant}_${DEPTH}/results.csv"
  if [[ ! -f "$best" ]]; then return 1; fi
  if [[ ! -f "$results" ]]; then return 1; fi
  local nlines; nlines=$(wc -l < "$results" 2>/dev/null || echo 0)
  # header + at least EPOCHS rows → training completed normally
  [[ "$nlines" -gt "$EPOCHS" ]]
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
declare -A GPU_PIDS
declare -A GPU_VARIANTS
declare -A GPU_START

launch_variant() {
  local variant=$1 gpu=$2
  local log_file="$MODEL_LOG_DIR/${variant}.log"
  local resume_flag=""
  local resume_note="fresh start"

  if variant_has_checkpoint "$variant"; then
    resume_flag="--resume"
    resume_note="RESUMING from last.pt"
  fi

  log_gpu "$gpu" "Launching ${BOLD}${variant}${NC} (${resume_note})"

  cat > "$JOB_DIR/gpu${gpu}.model" <<< "$variant"

  # Build the training command
  local cmd=(
    "$PYTHON" train_meddef.py
    --data    "$TBCR_DATA"
    --variant "$variant"
    --depth   "$DEPTH"
    --epochs  "$EPOCHS"
    --batch   "$BATCH"
    --imgsz   "$IMGSZ"
    --device  "$gpu"
    --workers "$WORKERS"
    --lr0     "$LR0"
    --lrf     "$LRF"
    --optimizer "$OPTIMIZER"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_epochs "$WARMUP_EPOCHS"
    --warmup_bias_lr "$WARMUP_BIAS_LR"
    --patience "$PATIENCE"
    --dropout "$DROPOUT"
    --erasing "$ERASING"
    --mixup   "$MIXUP"
    --cutmix  "$CUTMIX"
    --cos_lr  "$COS_LR"
    --save_period "$SAVE_PERIOD"
    --project "runs/classify/${RUN_NAME}/tbcr"
    --name    "${variant}_${DEPTH}"
    --exist_ok
    $resume_flag
  )

  if [[ "${_DRY_RUN:-0}" -eq 1 ]]; then
    log_info "[DRY-RUN] would run: ${cmd[*]}"
    (sleep 3 && echo "completed" > "$JOB_DIR/gpu${gpu}.status") &
  else
    (
      # trap '' INT TERM HUP makes Python inherit SIG_IGN for these signals;
      # Python skips installing its KeyboardInterrupt handler when SIGINT==SIG_IGN.
      # This means Ctrl+C in ANY terminal cannot kill these training processes.
      trap '' INT TERM HUP
      [[ -f "$VENV" ]] && source "$VENV"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${cmd[*]}" >> "$log_file"
      "${cmd[@]}" >> "$log_file" 2>&1
      EXIT_CODE=$?
      if [[ $EXIT_CODE -eq 0 ]]; then
        echo "completed" > "$JOB_DIR/gpu${gpu}.status"
      else
        echo "failed:$EXIT_CODE" > "$JOB_DIR/gpu${gpu}.status"
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

# ─────────────────────────────────────────────────────────────────────────────
# HARVEST FINISHED JOBS
# ─────────────────────────────────────────────────────────────────────────────
harvest() {
  IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
  for gpu in "${GPU_ARR[@]}"; do
    [[ -z "${GPU_PIDS[$gpu]:-}" ]] && continue
    local pid="${GPU_PIDS[$gpu]}"
    kill -0 "$pid" 2>/dev/null && continue   # still running

    local variant="${GPU_VARIANTS[$gpu]:-unknown}"
    local status_file="$JOB_DIR/gpu${gpu}.status"
    local status; status=$(cat "$status_file" 2>/dev/null || echo "unknown")

    if [[ "$status" == "completed" ]]; then
      # Extra sanity: check best.pt was actually written
      if [[ "${_DRY_RUN:-0}" -eq 1 ]] || [[ -f "$(best_pt_for_variant "$variant")" ]]; then
        local elapsed=$(( $(date +%s) - ${GPU_START[$gpu]:-$(date +%s)} ))
        log_ok "GPU $gpu: ✓ $variant done in $(printf '%02dh%02dm' $((elapsed/3600)) $(( (elapsed%3600)/60 )))"
        mark_completed "$variant"
      else
        log_warn "GPU $gpu: $variant — process exited 0 but no best.pt found; marking failed"
        mark_failed "$variant"
      fi
    else
      log_err "GPU $gpu: ✗ $variant failed (status=$status)"
      # Re-add to queue for retry on next scheduler tick (once)
      if ! is_completed "$variant" && ! is_failed "$variant"; then
        mark_failed "$variant"
      fi
    fi

    unset "GPU_PIDS[$gpu]" "GPU_VARIANTS[$gpu]" "GPU_START[$gpu]"
    rm -f "$JOB_DIR/gpu${gpu}.pid" "$JOB_DIR/gpu${gpu}.status" 2>/dev/null || true
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# FILL FREE GPUs
# ─────────────────────────────────────────────────────────────────────────────
fill_gpus() {
  [[ ! -s "$QUEUE_FILE" ]] && return

  IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
  for gpu in "${GPU_ARR[@]}"; do
    [[ ! -s "$QUEUE_FILE" ]] && break

    # Skip if we already have an active job on this GPU
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]] && kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
      continue
    fi

    is_gpu_free "$gpu" || continue
    acquire_lock "$gpu" || continue

    # Re-verify after lock
    if is_gpu_free "$gpu" && [[ -s "$QUEUE_FILE" ]]; then
      dequeue_next
      if [[ -n "$_NEXT_VARIANT" ]]; then
        launch_variant "$_NEXT_VARIANT" "$gpu"
      else
        release_lock "$gpu"
      fi
    else
      release_lock "$gpu"
    fi
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# BUILD INITIAL QUEUE
# ─────────────────────────────────────────────────────────────────────────────
build_queue() {
  # Truncate queue and rebuild from scratch
  > "$QUEUE_FILE"
  local added=0
  for variant in $VARIANTS; do
    if is_completed "$variant"; then
      log_info "  Skipping $variant — already completed"
      continue
    fi
    if variant_is_done "$variant"; then
      log_info "  $variant — results look complete; marking done"
      mark_completed "$variant"
      continue
    fi
    echo "$variant" >> "$QUEUE_FILE"
    (( added++ )) || true
    if variant_has_checkpoint "$variant"; then
      log_info "  Queued $variant  [will RESUME from last.pt]"
    else
      log_info "  Queued $variant  [fresh start]"
    fi
  done
  log_info "Queue built: $added variants to train"
}

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
show_dashboard() {
  local elapsed=$1
  local h=$(( elapsed/3600 )) m=$(( (elapsed%3600)/60 )) s=$(( elapsed%60 ))
  local queued=0; [[ -f "$QUEUE_FILE" ]] && queued=$(wc -l < "$QUEUE_FILE")
  local done=0;   [[ -f "$COMPLETED_FILE" ]] && done=$(wc -l < "$COMPLETED_FILE")
  local failed=0; [[ -f "$FAILED_FILE"    ]] && failed=$(wc -l < "$FAILED_FILE")
  local total; total=$(echo "$VARIANTS" | wc -w)
  local active=0
  IFS=',' read -ra _GA <<< "$GPU_IDS"
  for _g in "${_GA[@]}"; do
    [[ -n "${GPU_PIDS[$_g]:-}" ]] && kill -0 "${GPU_PIDS[$_g]}" 2>/dev/null && (( active++ )) || true
  done
  local pct=0; [[ $total -gt 0 ]] && pct=$(( done * 100 / total ))

  [[ -t 1 ]] && { tput clear 2>/dev/null || clear 2>/dev/null || true; }

  echo ""
  echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
  printf  "${CYAN}║${NC}  ${WHITE}${BOLD}MedDef2 TBCR FINAL Run${NC}  %-37s${CYAN}║${NC}\n" "Runtime: $(printf '%02dh%02dm%02ds' $h $m $s)"
  echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo ""

  # Progress bar
  printf "  Progress: ["
  local filled=$(( pct * 40 / 100 ))
  for (( i=0; i<40; i++ )); do
    [[ $i -lt $filled ]] && printf "${GREEN}|${NC}" || printf "."
  done
  printf "] %3d%%\n" $pct
  printf "  Queue: ${BLUE}%d${NC}  Active: ${YELLOW}%d${NC}  Done: ${GREEN}%d${NC}  Failed: ${RED}%d${NC}  Total: %d\n\n" \
    "$queued" "$active" "$done" "$failed" "$total"

  # GPU table
  echo -e "${CYAN}┌───────┬──────────────┬──────────┬──────────┬───────────────────────────────┐${NC}"
  echo -e "${CYAN}│${NC}  GPU  ${CYAN}│${NC} Free Memory  ${CYAN}│${NC}   Util   ${CYAN}│${NC}  Temp    ${CYAN}│${NC}  Status                       ${CYAN}│${NC}"
  echo -e "${CYAN}├───────┼──────────────┼──────────┼──────────┼───────────────────────────────┤${NC}"
  IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
  for gpu in "${GPU_ARR[@]}"; do
    local free; free=$(get_free_memory "$gpu")
    local util; util=$(get_gpu_util   "$gpu")
    local temp; temp=$(get_gpu_temp   "$gpu")
    local status_str scol
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]] && kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
      local v="${GPU_VARIANTS[$gpu]:-?}"
      local start="${GPU_START[$gpu]:-$(date +%s)}"
      local rt=$(( $(date +%s) - start ))
      # Try to read epoch from log
      local lf="$MODEL_LOG_DIR/${v}.log"
      local ep=""
      [[ -f "$lf" ]] && ep=$(tail -30 "$lf" 2>/dev/null | grep -oE '[0-9]+/[0-9]+' | grep '/' | tail -1)
      status_str="ACTIVE:${v} ep${ep:-?} $(printf '%02dh%02dm' $((rt/3600)) $(( (rt%3600)/60 )))"
      scol="$YELLOW"
    elif [[ "$free" -ge "$MIN_MEMORY_MB" && "$(get_process_count "$gpu")" -eq 0 ]]; then
      status_str="FREE — waiting"; scol="$GREEN"
    else
      status_str="BUSY (external)"; scol="$RED"
    fi
    local mcol="$GREEN"; [[ "$free" -lt "$MIN_MEMORY_MB" ]] && mcol="$RED"
    local tcol="$GREEN"; [[ "$temp" -ge 70 ]] && tcol="$YELLOW"; [[ "$temp" -ge 80 ]] && tcol="$RED"
    printf "${CYAN}│${NC} %5s ${CYAN}│${NC} ${mcol}%9s MB${NC} ${CYAN}│${NC} %6s%% ${CYAN}│${NC} ${tcol}%5s C ${NC} ${CYAN}│${NC} ${scol}%-29s${NC} ${CYAN}│${NC}\n" \
      "$gpu" "$free" "$util" "$temp" "$status_str"
  done
  echo -e "${CYAN}└───────┴──────────────┴──────────┴──────────┴───────────────────────────────┘${NC}"

  # Variant status
  echo ""
  echo -e "  ${BOLD}Variant Status:${NC}"
  for variant in $VARIANTS; do
    local sym color note
    if is_completed "$variant"; then
      local best; best="$(best_pt_for_variant "$variant")"
      local sz=""; [[ -f "$best" ]] && sz=" ($(du -sh "$best" 2>/dev/null | cut -f1))"
      sym="✓"; color="$GREEN"; note="DONE${sz}"
    elif is_failed "$variant"; then
      sym="✗"; color="$RED"; note="FAILED — will retry"
    elif variant_has_checkpoint "$variant"; then
      sym="⟳"; color="$YELLOW"; note="has checkpoint — will resume"
    else
      sym="○"; color="$BLUE"; note="waiting"
    fi
    # check if actively running
    for g in "${!GPU_VARIANTS[@]}"; do
      [[ "${GPU_VARIANTS[$g]:-}" == "$variant" ]] && kill -0 "${GPU_PIDS[$g]:-0}" 2>/dev/null && \
        { sym="▶"; color="$CYAN"; note="RUNNING on GPU ${g}"; break; }
    done
    printf "    ${color}%s  %-12s${NC}  %s\n" "$sym" "$variant" "$note"
  done

  if [[ -s "$QUEUE_FILE" ]]; then
    echo ""
    echo -e "  ${BOLD}Queue:${NC} $(wc -l < "$QUEUE_FILE") variants remaining"
    head -4 "$QUEUE_FILE" | while IFS= read -r v; do echo "    • $v"; done
  fi

  echo ""
  echo -e "  ${BOLD}Best hyperparameters:${NC} lr0=${LR0}  lrf=${LRF}  epochs=${EPOCHS}  batch=${BATCH}  patience=${PATIENCE}  save_period=${SAVE_PERIOD}"
  echo -e "  Logs: ${LOG_BASE}  |  $(date '+%H:%M:%S')  |  Ctrl+C to detach"
  echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# STATUS COMMAND
# ─────────────────────────────────────────────────────────────────────────────
cmd_status() {
  local total; total=$(echo "$VARIANTS" | wc -w)
  local done=0; [[ -f "$COMPLETED_FILE" ]] && done=$(wc -l < "$COMPLETED_FILE")
  local failed=0; [[ -f "$FAILED_FILE"  ]] && failed=$(wc -l < "$FAILED_FILE")
  echo ""
  echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${BLUE}║     MedDef2 TBCR FINAL — STATUS                  ║${NC}"
  echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════╝${NC}"
  echo -e "  Run tag  : ${WHITE}${RUN_NAME}${NC}"
  echo -e "  Output   : runs/classify/${RUN_NAME}/tbcr/<variant>_${DEPTH}/"
  echo -e "  Logs     : ${LOG_BASE}"
  echo ""
  echo -e "  Hyperparameters:"
  echo -e "    lr0=${LR0}  lrf=${LRF}  epochs=${EPOCHS}  batch=${BATCH}"
  echo -e "    patience=${PATIENCE}  weight_decay=${WEIGHT_DECAY}  dropout=${DROPOUT}"
  echo -e "    warmup=${WARMUP_EPOCHS}ep  erasing=${ERASING}  save_period=${SAVE_PERIOD}"
  echo ""
  echo -e "  Progress:  ${GREEN}${done} done${NC} / ${RED}${failed} failed${NC} / ${total} total"
  echo ""
  echo -e "  ${BOLD}Per-variant:${NC}"
  for variant in $VARIANTS; do
    local status best_pt last_pt row_color
    best_pt="$(best_pt_for_variant "$variant")"
    last_pt="$(last_pt_for_variant "$variant")"
    if is_completed "$variant"; then
      local sz=""; [[ -f "$best_pt" ]] && sz=" (best.pt: $(du -sh "$best_pt" 2>/dev/null | cut -f1))"
      echo -e "    ${GREEN}✓${NC}  ${variant}  —  DONE${sz}"
    elif is_failed "$variant"; then
      echo -e "    ${RED}✗${NC}  ${variant}  —  FAILED"
    elif [[ -f "$last_pt" ]]; then
      echo -e "    ${YELLOW}⟳${NC}  ${variant}  —  IN PROGRESS / RESUMABLE (last.pt exists)"
    else
      echo -e "    ${BLUE}○${NC}  ${variant}  —  NOT STARTED"
    fi
  done

  echo ""
  if [[ -f "$PID_FILE" ]]; then
    local p; p=$(cat "$PID_FILE")
    kill -0 "$p" 2>/dev/null \
      && echo -e "  ${GREEN}Scheduler: running (PID $p)${NC}" \
      || echo -e "  ${YELLOW}Scheduler: NOT running (stale PID $p)${NC}"
  else
    echo -e "  ${YELLOW}Scheduler: NOT running${NC}"
  fi
  echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE LOG TAIL
# ─────────────────────────────────────────────────────────────────────────────
cmd_live() {
  echo -e "${CYAN}${BOLD}== LIVE TRAINING LOGS — TBCR FINAL ==${NC}"
  echo -e "${YELLOW}Press Ctrl+C to exit (training continues)${NC}"
  local logs=()
  for f in "$MODEL_LOG_DIR"/*.log; do
    [[ -f "$f" ]] && logs+=("$f")
  done
  if [[ ${#logs[@]} -eq 0 ]]; then
    echo "No training logs found yet at $MODEL_LOG_DIR"
    exit 0
  fi
  tail -f "${logs[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
# WATCH (file-based live dashboard — safe from new shell)
# ─────────────────────────────────────────────────────────────────────────────
cmd_watch() {
  local interval="${1:-5}"
  trap 'echo ""; echo "Exiting watch — training continues in background"; exit 0' INT TERM

  # Initialize empty arrays (in-memory state is empty when called standalone)
  declare -A GPU_PIDS; declare -A GPU_VARIANTS; declare -A GPU_START

  # Reload GPU_PIDS from pid files
  while true; do
    IFS=',' read -ra _GA <<< "$GPU_IDS"
    for _g in "${_GA[@]}"; do
      local _pfp="$JOB_DIR/gpu${_g}.pid"
      local _mfp="$JOB_DIR/gpu${_g}.model"
      if [[ -f "$_pfp" ]]; then
        local _p; _p=$(cat "$_pfp" 2>/dev/null)
        if kill -0 "$_p" 2>/dev/null; then
          GPU_PIDS[$_g]="$_p"
          [[ -f "$_mfp" ]] && GPU_VARIANTS[$_g]=$(cat "$_mfp" 2>/dev/null)
        else
          unset "GPU_PIDS[$_g]" "GPU_VARIANTS[$_g]"
        fi
      fi
    done
    show_dashboard 0
    sleep "$interval"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# STOP
# ─────────────────────────────────────────────────────────────────────────────
cmd_stop() {
  if [[ -f "$PID_FILE" ]]; then
    local p; p=$(cat "$PID_FILE")
    kill "$p" 2>/dev/null && echo -e "${GREEN}Scheduler stopped (PID $p)${NC}" \
                          || echo -e "${YELLOW}Scheduler not running${NC}"
    rm -f "$PID_FILE"
  fi
  # Optionally kill running training jobs
  IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
  for gpu in "${GPU_ARR[@]}"; do
    local pf="$JOB_DIR/gpu${gpu}.pid"
    [[ -f "$pf" ]] || continue
    local p; p=$(cat "$pf")
    kill "$p" 2>/dev/null && echo -e "${RED}Killed GPU $gpu job (PID $p)${NC}" || true
    rm -f "$pf"
  done
  echo "Done."
}

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────────────────────────────────────
cmd_clean() {
  cmd_stop
  rm -rf "$LOG_BASE"
  echo -e "${GREEN}State and logs wiped. Re-run the script to start fresh.${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# CRON INSTALL / REMOVE
# ─────────────────────────────────────────────────────────────────────────────
cmd_install_cron() {
  local cron_entry="@reboot sleep 60 && bash ${SCRIPT_PATH} >> ${LOG_BASE}/cron_reboot.log 2>&1"
  # Remove any existing entries for this script
  crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
  # Add new entry
  (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -
  echo -e "${GREEN}Cron installed:${NC} $cron_entry"
  echo "On every reboot the trainer will resume automatically in 60 seconds."
  crontab -l | grep "$SCRIPT_PATH"
}

cmd_remove_cron() {
  crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
  echo -e "${GREEN}Cron entry removed.${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE INSTANCE GUARD
# ─────────────────────────────────────────────────────────────────────────────
guard_single_instance() {
  if [[ -f "$PID_FILE" ]]; then
    local old; old=$(cat "$PID_FILE")
    if kill -0 "$old" 2>/dev/null; then
      echo -e "${YELLOW}Trainer already running (PID $old). Use --status / --watch to monitor.${NC}"
      echo -e "To force restart: bash $SCRIPT_PATH --stop && bash $SCRIPT_PATH"
      exit 0
    fi
  fi
  echo $$ > "$PID_FILE"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCHEDULER LOOP
# ─────────────────────────────────────────────────────────────────────────────
run_scheduler() {
  # Detach cleanly on SIGINT/SIGTERM — training processes stay alive
  trap 'echo ""; log_info "Scheduler detached (all training jobs continue)"; rm -f "$PID_FILE"; exit 0' INT TERM

  local start_ts; start_ts=$(date +%s)
  local max_secs=$(( MAX_RUNTIME_HOURS * 3600 ))

  declare -A GPU_PIDS; declare -A GPU_VARIANTS; declare -A GPU_START

  log_info "════════════════════════════════════════════════════════"
  log_info "MedDef2 TBCR FINAL — Scheduler started (PID $$)"
  log_info "Run tag    : ${RUN_NAME}"
  log_info "Dataset    : ${TBCR_DATA}"
  log_info "Variants   : ${VARIANTS}"
  log_info "Hyperparams: lr0=${LR0} epochs=${EPOCHS} batch=${BATCH} patience=${PATIENCE}"
  log_info "GPUs       : ${GPU_IDS}"
  log_info "Logs       : ${LOG_BASE}"
  log_info "════════════════════════════════════════════════════════"

  build_queue

  local tick=0
  while true; do
    local elapsed=$(( $(date +%s) - start_ts ))
    [[ $elapsed -ge $max_secs ]] && { log_warn "Max runtime reached; stopping scheduler."; break; }

    harvest
    fill_gpus

    # Check if all done
    local total; total=$(echo "$VARIANTS" | wc -w)
    local done=0; [[ -f "$COMPLETED_FILE" ]] && done=$(wc -l < "$COMPLETED_FILE")
    local failed=0; [[ -f "$FAILED_FILE"  ]] && failed=$(wc -l < "$FAILED_FILE")
    local active=0
    IFS=',' read -ra _GA <<< "$GPU_IDS"
    for _g in "${_GA[@]}"; do
      [[ -n "${GPU_PIDS[$_g]:-}" ]] && kill -0 "${GPU_PIDS[$_g]}" 2>/dev/null && (( active++ )) || true
    done

    if [[ $(( done + failed )) -ge $total && $active -eq 0 ]]; then
      log_ok "All $total variants finished. Done=$done Failed=$failed"
      break
    fi

    # Show dashboard every 4 ticks
    if (( tick % 4 == 0 )); then
      show_dashboard "$elapsed"
    fi

    (( tick++ )) || true
    sleep "$CHECK_INTERVAL"
  done

  local _done _failed
  _done=$(  [[ -f "$COMPLETED_FILE" ]] && wc -l < "$COMPLETED_FILE" || echo 0)
  _failed=$([[ -f "$FAILED_FILE"    ]] && wc -l < "$FAILED_FILE"    || echo 0)
  log_ok "Scheduler exiting. Final: Done=${_done} Failed=${_failed}"
  rm -f "$PID_FILE"
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
_DRY_RUN=0

case "${1:-}" in
  --status)       cmd_status; exit 0 ;;
  --watch)        cmd_watch "${2:-5}"; exit 0 ;;
  --live)         cmd_live; exit 0 ;;
  --stop)         cmd_stop; exit 0 ;;
  --clean)
    read -rp "This will wipe all state/logs for TBCR FINAL. Continue? [y/N] " ans
    [[ "${ans,,}" == "y" ]] && cmd_clean || echo "Aborted."
    exit 0
    ;;
  --install-cron) cmd_install_cron; exit 0 ;;
  --remove-cron)  cmd_remove_cron;  exit 0 ;;
  --dry-run)      _DRY_RUN=1; log_warn "DRY-RUN mode — no real training" ;;
  --resume)       log_info "Explicit --resume: will resume any variant with last.pt" ;;
  --help|-h)
    head -30 "$SCRIPT_PATH" | grep '^#'
    exit 0
    ;;
  "")  ;;  # default: run
  *)   log_err "Unknown argument: $1. Run $SCRIPT_PATH --help"; exit 1 ;;
esac

# ── Self-daemonize: re-launch in background if we're the foreground entry ────
# This means 'bash train_tbcr_final.sh' always returns the prompt immediately;
# training can never be killed by Ctrl+C in any terminal.
if [[ -z "${_TBCR_DAEMON:-}" && "${_DRY_RUN:-0}" -ne 1 ]]; then
  mkdir -p "$LOG_BASE"
  export _TBCR_DAEMON=1
  nohup bash "$SCRIPT_PATH" "${@}" >> "$LOG_BASE/nohup.log" 2>&1 &
  DAEMON_PID=$!
  disown "$DAEMON_PID" 2>/dev/null || true
  echo -e "${GREEN}Scheduler launched in background (PID ${DAEMON_PID})${NC}"
  echo    "  Monitor : bash $SCRIPT_PATH --watch"
  echo    "  Live log: bash $SCRIPT_PATH --live"
  echo    "  Status  : bash $SCRIPT_PATH --status"
  echo    "  Stop    : bash $SCRIPT_PATH --stop"
  exit 0
fi

# Activate venv if available
[[ -f "$VENV" ]] && source "$VENV" || log_warn "venv not found at $VENV — using system Python"

# Verify data exists
if [[ ! -d "$TBCR_DATA" ]] && [[ "${_DRY_RUN:-0}" -ne 1 ]]; then
  log_err "TBCR dataset not found at: $TBCR_DATA"
  log_err "Set TBCR_DATA=/path/to/dataset and re-run"
  exit 1
fi

guard_single_instance
run_scheduler
