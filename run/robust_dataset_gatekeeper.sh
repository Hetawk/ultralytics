#!/bin/bash
set -euo pipefail

PROJECT_DIR="/data2/enoch/ekd_coding_env/ultralytics"
cd "$PROJECT_DIR"

DATASETS="${DATASETS:-ccts tbcr dermnet multic}"
VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
# Strict stage mode by default: finish current dataset before starting next.
STOP_OTHERS="${STOP_OTHERS:-true}"
BACKFILL_FREE_GPUS="${BACKFILL_FREE_GPUS:-false}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
# Quality gate: move to next dataset only after full variant improves.
QUALITY_GATE_ENABLED="${QUALITY_GATE_ENABLED:-true}"
MAX_STAGE_RETRIES="${MAX_STAGE_RETRIES:-4}"
MIN_IMPROVEMENT_DELTA="${MIN_IMPROVEMENT_DELTA:-0.003}"
# Require near-historical full-variant quality, not just tiny local improvement.
HISTORICAL_TOLERANCE="${HISTORICAL_TOLERANCE:-0.010}"
# Absolute quality floor: require at least 95% top1 before advancing.
ABS_MIN_FULL_TOP1="${ABS_MIN_FULL_TOP1:-0.95}"
# Runaway protection: guard against duplicate trainer/python launches.
MAX_TRAINER_INSTANCES_PER_DATASET="${MAX_TRAINER_INSTANCES_PER_DATASET:-4}"
MAX_PYTHON_PROCS_PER_DATASET="${MAX_PYTHON_PROCS_PER_DATASET:-0}"

# Always force full variant to the head of the queue, then keep others in order.
FULL_FIRST_VARIANTS="full"
for _v in $VARIANTS; do
  [[ "$_v" == "full" ]] && continue
  FULL_FIRST_VARIANTS+=" $_v"
done

GK_LOG_DIR="$PROJECT_DIR/logs/robust_dataset_gatekeeper"
mkdir -p "$GK_LOG_DIR"
GK_LOG="$GK_LOG_DIR/gatekeeper.log"
GK_PID_FILE="$GK_LOG_DIR/gatekeeper.pid"

# Singleton guard: prevent duplicate gatekeepers from racing stage transitions.
if [[ -f "$GK_PID_FILE" ]]; then
  _old_pid=$(cat "$GK_PID_FILE" 2>/dev/null || true)
  if [[ -n "${_old_pid:-}" ]] && kill -0 "$_old_pid" 2>/dev/null; then
    echo "Gatekeeper already running (PID $_old_pid). Exiting." >> "$GK_LOG"
    exit 0
  fi
fi
echo $$ > "$GK_PID_FILE"
trap 'rm -f "$GK_PID_FILE"' EXIT

linecount() {
  if [[ -f "$1" ]]; then
    awk 'NF { print }' "$1" 2>/dev/null | sort -u | wc -l | tr -d " \n"
  else
    echo 0
  fi
}

total_variants() {
  local n=0
  for _v in $VARIANTS; do ((n++)) || true; done
  echo "$n"
}

total_gpus() {
  local n=0
  local _g
  IFS=',' read -ra _arr <<< "$GPU_IDS"
  for _g in "${_arr[@]}"; do
    [[ -n "${_g// }" ]] && ((n++)) || true
  done
  [[ "$n" -gt 0 ]] || n=1
  echo "$n"
}

active_jobs_for_dataset() {
  local ds="$1"
  local jobs_dir="$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs"
  local c=0
  if [[ -d "$jobs_dir" ]]; then
    for pf in "$jobs_dir"/gpu*.pid; do
      [[ -f "$pf" ]] || continue
      local p
      p=$(cat "$pf" 2>/dev/null || true)
      [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && ((c++)) || true
    done
  fi
  echo "$c"
}

trainer_running_for_dataset() {
  local ds="$1"
  local pid_file="$PROJECT_DIR/logs/multi_gpu_training_${ds}/trainer.pid"
  if [[ -f "$pid_file" ]]; then
    local tp
    tp=$(cat "$pid_file" 2>/dev/null || true)
    [[ -n "$tp" ]] && kill -0 "$tp" 2>/dev/null && return 0
  fi
  # Fallback when trainer.pid is stale/missing: detect live trainer shell.
  if pgrep -f "smart_multi_gpu_trainer_${ds}\.sh" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$GK_LOG"
}

float_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

float_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'
}

float_max() {
  awk -v a="$1" -v b="$2" 'BEGIN { if (a>b) printf "%.6f", a; else printf "%.6f", b }'
}

best_full_top1_for_run_tag() {
  local ds="$1"
  local run_tag="$2"
  local base="runs/classify/train_${run_tag}/${ds}"
  if [[ ! -d "$base" ]]; then
    echo "nan"
    return
  fi

  awk -F',' '
    FNR==1 {
      col=0
      for (i=1; i<=NF; i++) if ($i=="metrics/accuracy_top1") col=i
      next
    }
    col>0 && $col!="" {
      v=$col+0
      if (!seen || v>best) { best=v; seen=1 }
    }
    END {
      if (seen) printf "%.6f", best
      else printf "nan"
    }
  ' "$base"/full*/results.csv 2>/dev/null
}

best_full_top1_for_tag_family() {
  local ds="$1"
  local base_tag="$2"
  local root="runs/classify"
  awk -F',' '
    FNR==1 {
      col=0
      for (i=1; i<=NF; i++) if ($i=="metrics/accuracy_top1") col=i
      next
    }
    col>0 && $col!="" {
      v=$col+0
      if (!seen || v>best) { best=v; seen=1 }
    }
    END {
      if (seen) printf "%.6f", best
      else printf "nan"
    }
  ' "$root"/train_"$base_tag"*/"$ds"/full*/results.csv 2>/dev/null
}

best_full_top1_global() {
  local ds="$1"
  local root="runs/classify"
  awk -F',' '
    FNR==1 {
      col=0
      for (i=1; i<=NF; i++) if ($i=="metrics/accuracy_top1") col=i
      next
    }
    col>0 && $col!="" {
      v=$col+0
      if (!seen || v>best) { best=v; seen=1 }
    }
    END {
      if (seen) printf "%.6f", best
      else printf "nan"
    }
  ' "$root"/*/"$ds"/full*/results.csv 2>/dev/null
}

count_trainer_instances_for_dataset() {
  local ds="$1"
  local n
  n=$(pgrep -fc "smart_multi_gpu_trainer_${ds}\\.sh" 2>/dev/null || true)
  [[ -z "$n" ]] && n=0
  echo "$n"
}

count_python_jobs_for_dataset() {
  local ds="$1"
  local n
  n=$(pgrep -fc "python .*train_meddef\.py .*processed_data/${ds}.*--variant" 2>/dev/null || true)
  [[ -z "$n" ]] && n=0
  echo "$n"
}

cleanup_dataset_runtime() {
  local ds="$1"
  local script="$PROJECT_DIR/run/smart_multi_gpu_trainer_${ds}.sh"
  [[ -x "$script" ]] && "$script" --stop >> "$GK_LOG_DIR/${ds}.stop.log" 2>&1 || true
  pkill -f "smart_multi_gpu_trainer_${ds}\.sh" 2>/dev/null || true
  pkill -f "python .*train_meddef\.py .*processed_data/${ds}.*--variant" 2>/dev/null || true
  rm -f "$PROJECT_DIR/logs/multi_gpu_training_${ds}/trainer.pid" \
        "$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs"/gpu*.pid \
        "$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs"/gpu*.status \
        "$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs"/gpu*.model \
        "$PROJECT_DIR/logs/multi_gpu_training_${ds}/locks"/gpu*.lock 2>/dev/null || true
}

reconcile_dataset_job_trackers() {
  local ds="$1"
  local state_dir="$PROJECT_DIR/logs/multi_gpu_training_${ds}/state"
  local completed_file="$state_dir/completed.txt"
  local failed_file="$state_dir/failed.txt"
  local jobs_dir="$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs"

  mkdir -p "$state_dir"
  touch "$completed_file" "$failed_file"
  [[ -d "$jobs_dir" ]] || return 0

  local sf
  for sf in "$jobs_dir"/gpu*.status; do
    [[ -f "$sf" ]] || continue

    local stem
    stem="${sf%.status}"
    local pid_file="${stem}.pid"
    local model_file="${stem}.model"

    local status model pid
    status=$(cat "$sf" 2>/dev/null || echo "unknown")
    model=$(cat "$model_file" 2>/dev/null || echo "")
    pid=$(cat "$pid_file" 2>/dev/null || echo "")

    local pid_alive=0
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && pid_alive=1 || true
    [[ "$pid_alive" -eq 1 ]] && continue

    [[ -n "$model" ]] || {
      rm -f "$pid_file" "$sf" "$model_file" 2>/dev/null || true
      continue
    }

    case "$status" in
      completed)
        if ! grep -qxF "$model" "$completed_file" 2>/dev/null; then
          echo "$model" >> "$completed_file"
          log "$ds reconciled completed tracker: $model"
        fi
        ;;
      failed)
        if ! grep -qxF "$model" "$failed_file" 2>/dev/null; then
          echo "$model" >> "$failed_file"
          log "$ds reconciled failed tracker: $model"
        fi
        ;;
      running|*)
        if ! grep -qxF "$model" "$failed_file" 2>/dev/null; then
          echo "$model" >> "$failed_file"
          log "$ds reconciled dead running tracker as failed: $model"
        fi
        ;;
    esac

    rm -f "$pid_file" "$sf" "$model_file" 2>/dev/null || true
  done
}

dataset_profile() {
  local ds="$1"
  local attempt="$2"

  PROFILE_RUN_TAG=""
  PROFILE_DEPTH=""
  PROFILE_EPOCHS=""
  PROFILE_BATCH=""
  PROFILE_NBS=""
  PROFILE_LR0=""
  PROFILE_LRF=""
  PROFILE_COS_LR=""
  PROFILE_PATIENCE=""
  PROFILE_DROPOUT=""
  PROFILE_MIXUP=""
  PROFILE_CUTMIX=""
  PROFILE_CLASS_WEIGHTS=""
  PROFILE_BASE_TAG=""

  case "$ds" in
    ccts)
      PROFILE_BASE_TAG="ccts_robust_v1"
      PROFILE_DEPTH="base"; PROFILE_EPOCHS="120"; PROFILE_BATCH="16"; PROFILE_NBS="16"
      PROFILE_LR0="0.0007"; PROFILE_LRF="0.05"; PROFILE_COS_LR="true"; PROFILE_PATIENCE="35"
      PROFILE_DROPOUT="0.10"; PROFILE_MIXUP="0.15"; PROFILE_CUTMIX="0.05"; PROFILE_CLASS_WEIGHTS="true"
      ;;
    tbcr)
      PROFILE_BASE_TAG="tbcr_legacy_v2"
      PROFILE_DEPTH="small"; PROFILE_EPOCHS="140"; PROFILE_BATCH="16"; PROFILE_NBS="16"
      PROFILE_LR0="0.001"; PROFILE_LRF="0.01"; PROFILE_COS_LR="false"; PROFILE_PATIENCE="120"
      PROFILE_DROPOUT="0.00"; PROFILE_MIXUP="0.00"; PROFILE_CUTMIX="0.00"; PROFILE_CLASS_WEIGHTS="true"
      ;;
    dermnet)
      PROFILE_BASE_TAG="dermnet_cfg_v2"
      PROFILE_DEPTH="small"; PROFILE_EPOCHS="80"; PROFILE_BATCH="32"; PROFILE_NBS="32"
      PROFILE_LR0="0.0003"; PROFILE_LRF="0.01"; PROFILE_COS_LR="true"; PROFILE_PATIENCE="40"
      PROFILE_DROPOUT="0.10"; PROFILE_MIXUP="0.10"; PROFILE_CUTMIX="0.05"; PROFILE_CLASS_WEIGHTS="true"
      ;;
    multic)
      PROFILE_BASE_TAG="multic_robust_v1"
      PROFILE_DEPTH="base"; PROFILE_EPOCHS="110"; PROFILE_BATCH="16"; PROFILE_NBS="16"
      PROFILE_LR0="0.0007"; PROFILE_LRF="0.05"; PROFILE_COS_LR="true"; PROFILE_PATIENCE="30"
      PROFILE_DROPOUT="0.08"; PROFILE_MIXUP="0.10"; PROFILE_CUTMIX="0.05"; PROFILE_CLASS_WEIGHTS="false"
      ;;
    *)
      return 1
      ;;
  esac

  case "$attempt" in
    0)
      PROFILE_RUN_TAG="$PROFILE_BASE_TAG"
      ;;
    1)
      PROFILE_RUN_TAG="${PROFILE_BASE_TAG}_impr1"
      PROFILE_EPOCHS=$((PROFILE_EPOCHS + 20))
      PROFILE_PATIENCE=$((PROFILE_PATIENCE + 20))
      PROFILE_LR0="$(awk -v v="$PROFILE_LR0" 'BEGIN{printf "%.6f", v*0.8}')"
      ;;
    2)
      PROFILE_RUN_TAG="${PROFILE_BASE_TAG}_impr2"
      PROFILE_EPOCHS=$((PROFILE_EPOCHS + 30))
      PROFILE_PATIENCE=$((PROFILE_PATIENCE + 30))
      if [[ "$PROFILE_COS_LR" == "true" ]]; then PROFILE_COS_LR="false"; else PROFILE_COS_LR="true"; fi
      PROFILE_LR0="$(awk -v v="$PROFILE_LR0" 'BEGIN{printf "%.6f", v*1.15}')"
      ;;
    3)
      PROFILE_RUN_TAG="${PROFILE_BASE_TAG}_impr3"
      PROFILE_EPOCHS=$((PROFILE_EPOCHS + 40))
      PROFILE_PATIENCE=$((PROFILE_PATIENCE + 40))
      PROFILE_DROPOUT="$(awk -v v="$PROFILE_DROPOUT" 'BEGIN{nv=v-0.03; if (nv<0) nv=0; printf "%.2f", nv}')"
      PROFILE_MIXUP="$(awk -v v="$PROFILE_MIXUP" 'BEGIN{nv=v-0.05; if (nv<0) nv=0; printf "%.2f", nv}')"
      ;;
    *)
      PROFILE_RUN_TAG="${PROFILE_BASE_TAG}_impr${attempt}"
      PROFILE_EPOCHS=$((PROFILE_EPOCHS + 45))
      PROFILE_PATIENCE=$((PROFILE_PATIENCE + 45))
      PROFILE_LR0="$(awk -v v="$PROFILE_LR0" 'BEGIN{printf "%.6f", v*0.75}')"
      ;;
  esac
}

stop_other_trainers() {
  local target_ds="$1"
  [[ "$STOP_OTHERS" == "true" ]] || return 0
  local loop_ds
  for loop_ds in ccts tbcr dermnet multic; do
    [[ "$loop_ds" == "$target_ds" ]] && continue
    # Force-clean non-target dataset runtime so orphaned python jobs cannot
    # block GPUs and make the active dataset appear stuck.
    log "Stopping non-target trainer/runtime: $loop_ds"
    cleanup_dataset_runtime "$loop_ds"
  done
}

start_dataset() {
  local ds="$1"
  local attempt="${2:-0}"
  local script="$PROJECT_DIR/run/smart_multi_gpu_trainer_${ds}.sh"
  [[ -x "$script" ]] || { log "Missing trainer script for $ds: $script"; return 1; }

  dataset_profile "$ds" "$attempt" || { log "Unknown dataset profile: $ds"; return 1; }

  local RUN_TAG="$PROFILE_RUN_TAG"
  local DEPTH="$PROFILE_DEPTH"
  local EPOCHS="$PROFILE_EPOCHS"
  local LR0="$PROFILE_LR0"
  local LRF="$PROFILE_LRF"
  local COS_LR="$PROFILE_COS_LR"
  local PATIENCE="$PROFILE_PATIENCE"
  local DROPOUT="$PROFILE_DROPOUT"
  local MIXUP="$PROFILE_MIXUP"
  local CUTMIX="$PROFILE_CUTMIX"
  local CLASS_WEIGHTS="$PROFILE_CLASS_WEIGHTS"
  local BATCH="$PROFILE_BATCH"
  local NBS="$PROFILE_NBS"

  # One-time stage reset per dataset+run_tag to prevent stale completed/failed
  # state from previous sessions affecting queue construction.
  local stage_marker="$GK_LOG_DIR/.stage_init_${ds}_${RUN_TAG}"
  if [[ ! -f "$stage_marker" ]]; then
    log "Initialising fresh stage for $ds (RUN_TAG=$RUN_TAG)"
    rm -rf "$PROJECT_DIR/logs/multi_gpu_training_${ds}/jobs" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/state" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/locks" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/model_logs" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/master.log" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/launcher.log" \
           "$PROJECT_DIR/logs/multi_gpu_training_${ds}/trainer.pid"
    # Preserve prior model artifacts/checkpoints; only reset scheduler state.
    mkdir -p "$PROJECT_DIR/logs/multi_gpu_training_${ds}"
    touch "$stage_marker"
  fi

  stop_other_trainers "$ds"

  log "Starting robust trainer for $ds (RUN_TAG=$RUN_TAG attempt=$attempt variants='$FULL_FIRST_VARIANTS')"
  nohup env \
    RUN_TAG="$RUN_TAG" DATASETS="$ds" DEPTH="$DEPTH" EPOCHS="$EPOCHS" \
    BATCH="$BATCH" NBS="$NBS" LR0="$LR0" LRF="$LRF" COS_LR="$COS_LR" \
    PATIENCE="$PATIENCE" DROPOUT="$DROPOUT" MIXUP="$MIXUP" CUTMIX="$CUTMIX" \
    CLASS_WEIGHTS="$CLASS_WEIGHTS" VARIANTS="$FULL_FIRST_VARIANTS" \
    SCHEDULER_MODE="conservative" ADAPTIVE_BATCH_ON_OOM="false" \
    "$script" --resume >> "$GK_LOG_DIR/${ds}.launch.log" 2>&1 &
}

monitor_dataset_until_done() {
  local ds="$1"
  local total
  total=$(total_variants)
  local state_dir="$PROJECT_DIR/logs/multi_gpu_training_${ds}/state"
  local completed_file="$state_dir/completed.txt"
  local failed_file="$state_dir/failed.txt"

  mkdir -p "$state_dir"
  touch "$completed_file" "$failed_file"

  dataset_profile "$ds" 0 || return 1
  local base_tag="$PROFILE_BASE_TAG"
  local family_best global_best baseline_best
  family_best=$(best_full_top1_for_tag_family "$ds" "$base_tag")
  global_best=$(best_full_top1_global "$ds")
  [[ "$family_best" == "nan" ]] && family_best=0
  [[ "$global_best" == "nan" ]] && global_best=0
  baseline_best=$(float_max "$family_best" "$global_best")
  local target_best
  local target_improve target_hist
  target_improve=$(awk -v b="$baseline_best" -v d="$MIN_IMPROVEMENT_DELTA" 'BEGIN{printf "%.6f", b+d}')
  target_hist=$(awk -v g="$global_best" -v t="$HISTORICAL_TOLERANCE" 'BEGIN{v=g-t; if (v<0) v=0; printf "%.6f", v}')
  target_best=$(float_max "$target_improve" "$target_hist")
  target_best=$(float_max "$target_best" "$ABS_MIN_FULL_TOP1")
  local stage_best="$baseline_best"
  local attempt=0
  local py_threshold
  if [[ "$MAX_PYTHON_PROCS_PER_DATASET" -gt 0 ]]; then
    py_threshold="$MAX_PYTHON_PROCS_PER_DATASET"
  else
    py_threshold=$(( $(total_gpus) + 2 ))
  fi

  log "$ds quality gate: family_best=$family_best global_best=$global_best baseline=$baseline_best abs_min=$ABS_MIN_FULL_TOP1 target>=$target_best max_retries=$MAX_STAGE_RETRIES"

  while [[ "$attempt" -le "$MAX_STAGE_RETRIES" ]]; do
    dataset_profile "$ds" "$attempt" || return 1

    # For retries, always force a fresh launch for the new attempt profile.
    if [[ "$attempt" -gt 0 ]]; then
      cleanup_dataset_runtime "$ds"
      start_dataset "$ds" "$attempt"
      sleep 5
    elif ! trainer_running_for_dataset "$ds"; then
      start_dataset "$ds" "$attempt"
      sleep 5
    else
      log "$ds trainer already running; monitoring only."
    fi

    while true; do
      # If wrappers finish after trainer controller died, reconcile tracker files
      # so completed/failed counts still progress and gatekeeper can recover.
      reconcile_dataset_job_trackers "$ds"

      local completed failed done active
      completed=$(linecount "$completed_file")
      failed=$(linecount "$failed_file")
      done=$((completed + failed))
      active=$(active_jobs_for_dataset "$ds")
      local trainer_instances py_jobs
      trainer_instances=$(count_trainer_instances_for_dataset "$ds")
      py_jobs=$(count_python_jobs_for_dataset "$ds")

      if [[ "$trainer_instances" -gt "$MAX_TRAINER_INSTANCES_PER_DATASET" && "$active" -eq 0 ]]; then
        log "$ds runaway detected: trainer_instances=$trainer_instances with no active jobs. Cleaning and restarting attempt=$attempt"
        cleanup_dataset_runtime "$ds"
        sleep 3
        start_dataset "$ds" "$attempt"
        sleep 5
        continue
      fi

      log "$ds progress: done=$done/$total completed=$completed failed=$failed active=$active trainers=$trainer_instances py_jobs=$py_jobs attempt=$attempt"

      if [[ "$done" -ge "$total" && "$active" -eq 0 ]]; then
        break
      fi

      if ! trainer_running_for_dataset "$ds" && [[ "$active" -eq 0 ]] && [[ "$done" -lt "$total" ]]; then
        log "$ds trainer not running and work remains. Restarting."
        start_dataset "$ds" "$attempt"
        sleep 5
      fi

    # Optional backfill mode: if there are idle GPUs, start later datasets so
    # we keep hardware busy instead of waiting for strict stage completion.
    if [[ "${BACKFILL_FREE_GPUS,,}" == "true" ]]; then
      local total_slots total_active idle
      total_slots=$(total_gpus)
      total_active=0
      local bds
      for bds in $DATASETS; do
        total_active=$(( total_active + $(active_jobs_for_dataset "$bds") ))
      done
      idle=$(( total_slots - total_active ))

      if [[ "$idle" -gt 0 ]]; then
        local started=0
        local cand
        for cand in $DATASETS; do
          [[ "$cand" == "$ds" ]] && continue

          local c_state_dir c_completed_file c_failed_file c_completed c_failed c_done
          c_state_dir="$PROJECT_DIR/logs/multi_gpu_training_${cand}/state"
          c_completed_file="$c_state_dir/completed.txt"
          c_failed_file="$c_state_dir/failed.txt"
          mkdir -p "$c_state_dir"
          touch "$c_completed_file" "$c_failed_file"
          c_completed=$(linecount "$c_completed_file")
          c_failed=$(linecount "$c_failed_file")
          c_done=$((c_completed + c_failed))

          if [[ "$c_done" -lt "$total" ]] && ! trainer_running_for_dataset "$cand"; then
            log "Backfill: idle_gpus=$idle -> starting $cand"
            start_dataset "$cand"
            started=$((started + 1))
            idle=$((idle - 1))
            [[ "$idle" -le 0 ]] && break
          fi
        done

        [[ "$started" -gt 0 ]] && sleep 5
      fi
    fi

      sleep "$CHECK_INTERVAL"
    done

    local current_best
    current_best=$(best_full_top1_for_run_tag "$ds" "$PROFILE_RUN_TAG")
    if [[ "$current_best" != "nan" ]] && float_gt "$current_best" "$stage_best"; then
      stage_best="$current_best"
    fi

    if [[ "$QUALITY_GATE_ENABLED" != "true" ]]; then
      log "$ds COMPLETE. quality gate disabled."
      return 0
    fi

    if [[ "$current_best" != "nan" ]] && float_ge "$current_best" "$target_best"; then
      log "$ds COMPLETE with improvement: full_top1=$current_best baseline=$baseline_best target=$target_best"
      return 0
    fi

    log "$ds quality below target after attempt=$attempt: full_top1=$current_best target=$target_best. Retrying with stronger profile."
    attempt=$((attempt + 1))
  done

  log "$ds did not meet quality target after $MAX_STAGE_RETRIES retries. Holding pipeline on this dataset."
  return 1
}

main() {
  log "Gatekeeper start: datasets='$DATASETS' check_interval=${CHECK_INTERVAL}s"
  for ds in $DATASETS; do
    log "Entering dataset stage: $ds"
    if ! monitor_dataset_until_done "$ds"; then
      log "Pipeline paused: dataset '$ds' failed quality gate. Will not advance to next dataset."
      while true; do sleep "$CHECK_INTERVAL"; done
    fi
  done
  log "All dataset stages completed."
}

main
