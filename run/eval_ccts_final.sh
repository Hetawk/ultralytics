#!/usr/bin/env bash
# =============================================================================
#  eval_ccts_final.sh — CCTS post-training evaluation scheduler
#  Mirrors eval_tbcr_final.sh for the CCTS lung-cancer CT dataset.
#
#  Evaluates all 6 MedDef variants × 2 model stages (stage1 + distill_v2).
#  Runs continuously alongside distillation — picks up each distill job as
#  soon as its best.pt appears and a GPU is free.
#
#  Per job saves:
#    evaluation_metrics.json
#    results.csv
#    robustness/
#      robustness_results.json
#      robustness_per_attack.csv
#      robustness_epsilon_sweep.csv
#      robustness_summary.txt
#      plots/
#    visualizations/
#      confusion_matrix.png
#      class_performance.png
#      saliency/
#      tsne.png / pca.png
#
#  Attacks evaluated (8 total):
#    fgsm   pgd   bim   mim   cw   deepfool   apgd   square
#
#  Usage:
#    bash run/eval_ccts_final.sh              # start scheduler (daemonises)
#    bash run/eval_ccts_final.sh --dry-run    # check prerequisites, show plan
#    bash run/eval_ccts_final.sh --status     # job table + GPU memory
#    bash run/eval_ccts_final.sh --watch      # live status + log tails
#    bash run/eval_ccts_final.sh --stop       # kill all eval jobs
#    bash run/eval_ccts_final.sh --reset      # wipe state, start over
#    STAGE=stage1  bash run/eval_ccts_final.sh   # stage-1 weights only
#    STAGE=distill bash run/eval_ccts_final.sh   # distill weights only
#    VARIANTS="full baseline" bash run/eval_ccts_final.sh
#    GPU_IDS=2,3   bash run/eval_ccts_final.sh
# =============================================================================

set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Configuration (override via env) ─────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/data2/enoch/ekd_coding_env/meddef_winlab/processed_data}"
CCTS_DATA="${CCTS_DATA:-${DATA_ROOT}/ccts}"
RUN_NAME="${RUN_NAME:-train_ccts_final}"
EVAL_NAME="${EVAL_NAME:-train_ccts_final_eval_v2}"
# Which subdirectory holds the distilled weights inside each variant folder.
DISTILL_STAGE_NAME="${DISTILL_STAGE_NAME:-distill_v2}"
VENV="${VENV:-/data2/enoch/.virtualenvs/meddef_final/bin/activate}"
PYTHON="${PYTHON:-python}"

# GPU pool
GPU_IDS="${GPU_IDS:-0,1,2,3}"
# 13 GB threshold: ensures eval doesn't grab a GPU running distillation.
MIN_MEMORY_MB="${MIN_MEMORY_MB:-13000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
DEPTH="${DEPTH:-small}"

MIN_ACTIVE_DISTILL="${MIN_ACTIVE_DISTILL:-0}"

# Post-crash GPU cooldown (seconds)
GPU_COOLDOWN_SECS="${GPU_COOLDOWN_SECS:-90}"

# OOM retry limit
MAX_RETRIES="${MAX_RETRIES:-3}"

# How many epochs distillation is configured for (must match distill_ccts_final.sh).
DISTILL_EPOCHS="${DISTILL_EPOCHS:-50}"

# Stages: "both" | "stage1" | "distill"
STAGE="${STAGE:-both}"
VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"

# Attacks — same 8 as TBCR for direct comparison
ATTACKS="${ATTACKS:-fgsm pgd bim mim cw deepfool apgd square}"
# Same epsilon sweep for comparability
EPSILONS="${EPSILONS:-0.0 0.005 0.01 0.02 0.03 0.05 0.1 0.15 0.2 0.3}"

# Eval settings
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-224}"
WORKERS="${WORKERS:-4}"
N_SALIENCY="${N_SALIENCY:-8}"

# ── Derived paths ─────────────────────────────────────────────────────────────
LOG_BASE="$PROJECT_DIR/logs/ccts_final_eval_${DISTILL_STAGE_NAME}"
DISTILL_JOBS_DIR="$PROJECT_DIR/logs/ccts_final_${DISTILL_STAGE_NAME}/jobs"
TRAIN_BASE="$PROJECT_DIR/runs/classify/${RUN_NAME}/ccts"
EVAL_BASE="$PROJECT_DIR/runs/classify/${EVAL_NAME}/ccts"
STATE_DIR="$LOG_BASE/state"
PID_FILE="$LOG_BASE/scheduler.pid"
MASTER_LOG="$LOG_BASE/master.log"

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(_ts)] $*" | tee -a "$MASTER_LOG" 2>/dev/null || echo "[$(_ts)] $*"; }
info() { log "[INFO ] $*"; }
ok()   { log "[DONE ] $*"; }
warn() { log "[WARN ] $*"; }
err()  { log "[ERROR] $*" >&2; }

# ── GPU helpers ───────────────────────────────────────────────────────────────
gpu_list() { echo "$GPU_IDS" | tr ',' ' '; }

is_gpu_free() {
    local gpu_id="$1"
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           -i "$gpu_id" 2>/dev/null | tr -d ' ')
    [[ -n "$free" && "$free" -ge "$MIN_MEMORY_MB" ]]
}

gpus_in_use() {
    for rf in "$STATE_DIR"/*.running; do
        [[ -f "$rf" ]] || continue
        local entry; entry=$(cat "$rf" 2>/dev/null)
        if [[ "$entry" == *:* ]]; then
            local g pid
            g=${entry%%:*}; pid=${entry##*:}
            kill -0 "$pid" 2>/dev/null && echo "$g"
        fi
    done
}

first_free_gpu() {
    local busy; busy=$(gpus_in_use | tr '\n' ' ')
    for g in $(gpu_list); do
        [[ " $busy " == *" $g "* ]] && continue
        is_gpu_cooled_down "$g" || continue
        is_gpu_free "$g" && echo "$g" && return 0
    done
    return 1
}

set_gpu_cooldown() {
    local gpu="$1"
    [[ -z "${gpu:-}" || "$GPU_COOLDOWN_SECS" -le 0 ]] && return
    echo $(( $(date +%s) + GPU_COOLDOWN_SECS )) > "$STATE_DIR/gpu_${gpu}.cooldown"
    info "GPU $gpu cooling down for ${GPU_COOLDOWN_SECS}s after crash"
}

is_gpu_cooled_down() {
    local gpu="$1"
    local cf="$STATE_DIR/gpu_${gpu}.cooldown"
    [[ -f "$cf" ]] || return 0
    local until; until=$(cat "$cf" 2>/dev/null)
    if [[ $(date +%s) -ge "${until:-0}" ]]; then
        rm -f "$cf"
        return 0
    fi
    return 1
}

count_active_distill() {
    local count=0
    if [[ -d "$DISTILL_JOBS_DIR" ]]; then
        for pf in "$DISTILL_JOBS_DIR"/*.pid; do
            [[ -f "$pf" ]] || continue
            local pid; pid=$(cat "$pf" 2>/dev/null)
            [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && ((count++))
        done
    fi
    echo "$count"
}

distill_guard_ok() {
    [[ "$MIN_ACTIVE_DISTILL" -le 0 ]] && return 0
    local n; n=$(count_active_distill)
    [[ "$n" -ge "$MIN_ACTIVE_DISTILL" ]] && return 0
    for g in $(gpu_list); do
        is_gpu_free "$g" || return 0
    done
    return 0
}

# ── Weight path helpers ───────────────────────────────────────────────────────
stage1_weights()  { echo "$TRAIN_BASE/${1}_${DEPTH}/weights/best.pt"; }
distill_weights() { echo "$TRAIN_BASE/${1}_${DEPTH}/${DISTILL_STAGE_NAME}/weights/best.pt"; }
weights_for()     { [[ "$2" == "stage1" ]] && stage1_weights "$1" || distill_weights "$1"; }

# ── Job state helpers ─────────────────────────────────────────────────────────
job_key() { echo "${1}_${2}"; }

is_done()         { [[ -f "$STATE_DIR/$(job_key "$1" "$2").done"    ]]; }
is_failed()       { [[ -f "$STATE_DIR/$(job_key "$1" "$2").failed"  ]]; }
is_running_file() { [[ -f "$STATE_DIR/$(job_key "$1" "$2").running" ]]; }

mark_pending() { touch "$STATE_DIR/$(job_key "$1" "$2").pending"; }
mark_running() {
    rm -f "$STATE_DIR/$(job_key "$1" "$2").pending"
    echo "${4:-}:${3}" > "$STATE_DIR/$(job_key "$1" "$2").running"
}
mark_done() {
    rm -f "$STATE_DIR/$(job_key "$1" "$2").running"
    touch "$STATE_DIR/$(job_key "$1" "$2").done"
}
mark_failed() {
    rm -f "$STATE_DIR/$(job_key "$1" "$2").running"
    touch "$STATE_DIR/$(job_key "$1" "$2").failed"
}

job_pid_alive() {
    local rf="$STATE_DIR/$(job_key "$1" "$2").running"
    [[ -f "$rf" ]] || return 1
    local entry; entry=$(cat "$rf" 2>/dev/null)
    local pid
    [[ "$entry" == *:* ]] && pid=${entry##*:} || pid=$entry
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

distill_weights_ready() {
    local best; best="$(distill_weights "$1")"
    [[ -f "$best" ]] || return 1
    local last; last="${best%best.pt}last.pt"
    if [[ -f "$last" ]]; then
        local epoch
        epoch=$(python3 - "$last" <<'PY'
import sys, torch
try:
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    print(int(ckpt.get("epoch", -1)))
except Exception:
    print(-1)
PY
)
        [[ "$epoch" =~ ^[0-9]+$ ]] && [[ "$epoch" -ge $((DISTILL_EPOCHS - 1)) ]] && return 0
        return 1
    fi
    return 0
}

# ── Launch a single eval job ──────────────────────────────────────────────────
launch_eval() {
    local variant="$1" stage="$2" gpu="$3"
    local weights; weights=$(weights_for "$variant" "$stage")
    local out_dir="$EVAL_BASE/${variant}_${DEPTH}/${stage}"
    local log_file="$LOG_BASE/${variant}_${stage}.log"

    mkdir -p "$out_dir" "$LOG_BASE"

    local atk_args=(); for a in $ATTACKS;  do atk_args+=("$a"); done
    local eps_args=(); for e in $EPSILONS; do eps_args+=("$e"); done

    info "[GPU $gpu] START  variant=$variant  stage=$stage"
    info "           weights → $weights"
    info "           output  → $out_dir"

    (
        source "$VENV"
        cd "$PROJECT_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        $PYTHON evaluate.py \
            --model      "$weights" \
            --data       "$CCTS_DATA" \
            --split      val \
            --batch      "$BATCH" \
            --imgsz      "$IMGSZ" \
            --workers    "$WORKERS" \
            --device     0 \
            --output-dir "$out_dir" \
            --verbose \
            --visualize \
            --saliency \
            --n-saliency "$N_SALIENCY" \
            --robustness \
            --attacks    "${atk_args[@]}" \
            --epsilons   "${eps_args[@]}" \
            2>&1
        echo "EVAL_EXIT_CODE:$?" >> "$log_file"
    ) >> "$log_file" &

    local pid=$!
    mark_running "$variant" "$stage" "$pid" "$gpu"
    info "[GPU $gpu] PID=$pid  variant=$variant  stage=$stage"
}

# ── Harvest completed jobs ────────────────────────────────────────────────────
harvest() {
    for v in $VARIANTS; do
        for s in stage1 distill; do
            [[ "$STAGE" != "both" && "$STAGE" != "$s" ]] && continue
            is_running_file "$v" "$s" || continue
            job_pid_alive "$v" "$s"   && continue

            local lf="$LOG_BASE/${v}_${s}.log"
            local rf="$STATE_DIR/$(job_key "$v" "$s").running"
            local entry; entry=$(cat "$rf" 2>/dev/null)
            local failed_gpu; [[ "$entry" == *:* ]] && failed_gpu=${entry%%:*} || failed_gpu=""

            local exit_line; exit_line=$(grep "EVAL_EXIT_CODE:" "$lf" 2>/dev/null | tail -1)
            if [[ "$exit_line" == *":0" ]]; then
                mark_done "$v" "$s"
                ok "Completed: $v / $s"
            else
                rm -f "$rf"

                local is_oom=0
                grep -q "CUDA error: out of memory\|RuntimeError: CUDA error" \
                    "$lf" 2>/dev/null && is_oom=1

                local rcf="$STATE_DIR/$(job_key "$v" "$s").retry_count"
                local retries=0
                [[ -f "$rcf" ]] && retries=$(cat "$rcf" 2>/dev/null || echo 0)

                if [[ "$is_oom" -eq 1 && "$retries" -lt "$MAX_RETRIES" ]]; then
                    echo $(( retries + 1 )) > "$rcf"
                    warn "OOM — retry $((retries+1))/${MAX_RETRIES}: $v/$s (was on GPU ${failed_gpu:-?})"
                    [[ -n "${failed_gpu:-}" ]] && set_gpu_cooldown "$failed_gpu"
                else
                    touch "$STATE_DIR/$(job_key "$v" "$s").failed"
                    warn "Failed:    $v / $s  (check $lf)"
                    [[ -n "${failed_gpu:-}" ]] && set_gpu_cooldown "$failed_gpu"
                fi
            fi
        done
    done
}

# ── Build job list ────────────────────────────────────────────────────────────
build_job_list() {
    for v in $VARIANTS; do
        [[ "$STAGE" == "both" || "$STAGE" == "stage1"  ]] && echo "$v stage1"
        [[ "$STAGE" == "both" || "$STAGE" == "distill" ]] && echo "$v distill"
    done
}

# ── --dry-run ─────────────────────────────────────────────────────────────────
cmd_dry_run() {
    local pass=1
    local PASS="  [ OK ]" FAIL="  [FAIL]" WARN="  [WARN]" IINF="  [INFO]"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       CCTS EVALUATION PREREQUISITE CHECK  (--dry-run)       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    echo "── Python & Environment ─────────────────────────────────────────"
    if [[ -f "$VENV" ]]; then
        echo "$PASS  venv found: $VENV"
        local pyver; pyver=$(source "$VENV" && python --version 2>&1)
        echo "$IINF  $pyver"
    else
        echo "$FAIL  venv NOT found: $VENV"; pass=0
    fi

    echo ""
    echo "── Evaluation Script ────────────────────────────────────────────"
    if [[ -f "$PROJECT_DIR/evaluate.py" ]]; then
        echo "$PASS  evaluate.py found"
    else
        echo "$FAIL  evaluate.py NOT found at $PROJECT_DIR/evaluate.py"; pass=0
    fi

    echo ""
    echo "── Dataset ──────────────────────────────────────────────────────"
    if [[ -d "$CCTS_DATA" ]]; then
        echo "$PASS  CCTS data dir: $CCTS_DATA"
        for split in val test train; do
            if [[ -d "$CCTS_DATA/$split" ]]; then
                local n_cls; n_cls=$(find "$CCTS_DATA/$split" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
                echo "$IINF    $split/  — $n_cls class folders (4 expected)"
                break
            fi
        done
    else
        echo "$FAIL  CCTS data NOT found: $CCTS_DATA"; pass=0
    fi

    echo ""
    echo "── GPUs ─────────────────────────────────────────────────────────"
    if command -v nvidia-smi &>/dev/null; then
        echo "$PASS  nvidia-smi available"
        for g in $(gpu_list); do
            local free used
            free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
            if [[ -n "$free" ]]; then
                local gstatus="busy (${free}MB free < ${MIN_MEMORY_MB}MB threshold)"
                [[ "$free" -ge "$MIN_MEMORY_MB" ]] && gstatus="FREE — will use (${free}MB free)"
                echo "$IINF    GPU $g  used=${used}MB  free=${free}MB  → $gstatus"
            else
                echo "$WARN  GPU $g not accessible"
            fi
        done
    else
        echo "$FAIL  nvidia-smi NOT found"; pass=0
    fi

    echo ""
    echo "── Model Weights ────────────────────────────────────────────────"
    for v in $VARIANTS; do
        local w1; w1=$(stage1_weights "$v")
        if [[ -f "$w1" ]]; then echo "$PASS  stage1  $v"; else echo "$WARN  stage1  $v  NOT FOUND: $w1"; fi
        if distill_weights_ready "$v"; then echo "$PASS  distill $v  (training complete)"
        elif [[ -f "$(distill_weights "$v")" ]]; then echo "$WARN  distill $v  weights exist but training still in progress"
        else echo "$WARN  distill $v  not yet available — scheduler will wait"; fi
    done

    echo ""
    echo "── Planned Jobs ─────────────────────────────────────────────────"
    printf "  %-24s %-10s %-13s  %s\n" "VARIANT" "STAGE" "WEIGHTS" "ACTION"
    printf "  %-24s %-10s %-13s  %s\n" "-------" "-----" "-------" "------"
    for v in $VARIANTS; do
        for s in stage1 distill; do
            [[ "$STAGE" != "both" && "$STAGE" != "$s" ]] && continue
            local w; w=$(weights_for "$v" "$s")
            local wst="missing"
            if [[ "$s" == "distill" ]]; then
                distill_weights_ready "$v" && wst="ready"
            else
                [[ -f "$w" ]] && wst="ready"
            fi
            local action="will run"
            is_done   "$v" "$s" && action="already done (skip)"
            is_failed "$v" "$s" && action="previously failed (skip)"
            [[ "$wst" == "missing" ]] && action="wait for weights"
            printf "  %-24s %-10s %-13s  %s\n" "${v}_${DEPTH}" "$s" "$wst" "$action"
        done
    done

    echo ""
    if [[ "$pass" -eq 1 ]]; then
        echo "  RESULT: All hard prerequisites satisfied ✓"
        echo "  Safe to start: bash run/eval_ccts_final.sh"
    else
        echo "  RESULT: Prerequisites FAILED — fix issues marked [FAIL] first."
    fi
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    return $((1 - pass))
}

# ── --status ──────────────────────────────────────────────────────────────────
cmd_status() {
    mkdir -p "$STATE_DIR"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         CCTS Final Evaluation — Status                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    printf "  %-24s %-10s %-32s\n" "VARIANT" "STAGE" "STATUS"
    printf "  %-24s %-10s %-32s\n" "-------" "-----" "------"
    for v in $VARIANTS; do
        for s in stage1 distill; do
            [[ "$STAGE" != "both" && "$STAGE" != "$s" ]] && continue
            local key; key=$(job_key "$v" "$s")
            local status="pending"
            if   [[ -f "$STATE_DIR/${key}.done"    ]]; then status="DONE ✓"
            elif [[ -f "$STATE_DIR/${key}.failed"  ]]; then status="FAILED ✗"
            elif [[ -f "$STATE_DIR/${key}.running" ]]; then
                local entry; entry=$(cat "$STATE_DIR/${key}.running" 2>/dev/null)
                local pid; [[ "$entry" == *:* ]] && pid=${entry##*:} || pid=$entry
                if kill -0 "$pid" 2>/dev/null; then status="running (PID $pid, GPU ${entry%%:*})"
                else status="STALE (process gone)"; fi
            elif [[ "$s" == "distill" ]] && ! distill_weights_ready "$v"; then
                status="waiting — distill not finished"
            fi
            printf "  %-24s %-10s %-32s\n" "${v}_${DEPTH}" "$s" "$status"
        done
    done
    echo ""
    echo "  Scheduler: $(cat "$PID_FILE" 2>/dev/null | xargs -I{} sh -c 'kill -0 {} 2>/dev/null && echo "running (PID {})" || echo "not running"' || echo 'not running')"
    echo ""
    echo "  GPU Memory:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null \
        | awk -F',' '{printf "    GPU %s: used=%s  free=%s\n",$1,$2,$3}'
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# ── --watch ───────────────────────────────────────────────────────────────────
cmd_watch() {
    while true; do
        clear
        cmd_status
        echo ""
        echo "  Recent log tails (last 5 lines):"
        for f in "$LOG_BASE"/*.log; do
            [[ -f "$f" ]] || continue
            [[ "$(basename "$f")" == "master.log" ]] && continue
            printf "  ── %s ──\n" "$(basename "$f")"
            tail -5 "$f" 2>/dev/null | sed 's/^/    /'
            echo ""
        done
        printf "  [watching — Ctrl+C to exit — refreshes every 30s]\n"
        sleep 30
    done
}

# ── --stop ────────────────────────────────────────────────────────────────────
cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        local spid; spid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "${spid:-}" ]] && kill -0 "$spid" 2>/dev/null; then
            kill "$spid" && info "Killed scheduler PID=$spid"
        fi
        rm -f "$PID_FILE"
    fi
    for sf in "$STATE_DIR"/*.running; do
        [[ -f "$sf" ]] || continue
        local pid; pid=$(cat "$sf" 2>/dev/null)
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && info "Killed eval job PID=$pid"
        fi
        rm -f "$sf"
    done
    ok "All stopped."
}

# ── --reset ───────────────────────────────────────────────────────────────────
cmd_reset() {
    cmd_stop 2>/dev/null || true
    rm -rf "$STATE_DIR"
    mkdir -p "$STATE_DIR"
    ok "State cleared — run without arguments to start fresh."
}

# ── Main scheduler ────────────────────────────────────────────────────────────
cmd_run() {
    mkdir -p "$LOG_BASE" "$STATE_DIR" "$EVAL_BASE"
    echo $$ > "$PID_FILE"

    info "════════════════════════════════════════════════════"
    info " CCTS Final Evaluation Scheduler"
    info " Dataset  : CCTS (4-class CT lung cancer, 1000 imgs)"
    info " Variants : $VARIANTS"
    info " Stage    : $STAGE  |  Depth: $DEPTH"
    info " Attacks  : $ATTACKS"
    info " Epsilons : $EPSILONS"
    info " GPU pool : $GPU_IDS  (min free: ${MIN_MEMORY_MB}MB)"
    info " Output   : $EVAL_BASE"
    info "════════════════════════════════════════════════════"

    local queue_file="$STATE_DIR/queue.txt"
    {
        build_job_list | while IFS= read -r jobline; do
            read -r v s <<< "$jobline"
            { is_done "$v" "$s" || is_failed "$v" "$s"; } && continue
            echo "$jobline"
        done
    } > "$queue_file"

    local total; total=$(grep -c . "$queue_file" 2>/dev/null || echo 0)
    info "Jobs queued: $total"

    while true; do
        harvest

        local active=0 waiting_distill=0 done_count=0

        while IFS= read -r jobline; do
            [[ -z "$jobline" ]] && continue
            read -r v s <<< "$jobline"

            if is_done "$v" "$s" || is_failed "$v" "$s"; then
                ((done_count++)); continue
            fi

            if is_running_file "$v" "$s" && ! job_pid_alive "$v" "$s"; then
                local stale_rf="$STATE_DIR/$(job_key "$v" "$s").running"
                local stale_entry; stale_entry=$(cat "$stale_rf" 2>/dev/null)
                local stale_gpu; [[ "$stale_entry" == *:* ]] && stale_gpu=${stale_entry%%:*}
                warn "Stale entry $v/$s — re-queuing"
                rm -f "$stale_rf"
                [[ -n "${stale_gpu:-}" ]] && set_gpu_cooldown "$stale_gpu"
            fi

            if job_pid_alive "$v" "$s"; then
                ((active++)); continue
            fi

            if [[ "$s" == "distill" ]] && ! distill_weights_ready "$v"; then
                ((waiting_distill++))
                continue
            fi

            local gpu
            if distill_guard_ok && gpu=$(first_free_gpu); then
                launch_eval "$v" "$s" "$gpu"
                ((active++))
                break
            fi

        done < "$queue_file"

        local remaining=$(( total - done_count ))

        if [[ "$active" -eq 0 && "$waiting_distill" -eq 0 && "$remaining" -le 0 ]]; then
            ok "All evaluation jobs complete."
            break
        fi
        if [[ "$STAGE" == "stage1" && "$active" -eq 0 && "$remaining" -le 0 ]]; then
            ok "All stage-1 evaluations complete."
            break
        fi

        info "Progress: ${done_count}/${total} done  active=$active  waiting_distill=$waiting_distill  active_distill=$(count_active_distill)  sleep=${CHECK_INTERVAL}s"
        sleep "$CHECK_INTERVAL"
    done

    rm -f "$PID_FILE"
    info "════════════════════════════════════════════════════"
    info " CCTS evaluation finished.  Results → $EVAL_BASE"
    info "════════════════════════════════════════════════════"
    cmd_status
}

# ── Self-daemonise ────────────────────────────────────────────────────────────
if [[ "${MEDDEF_CCTS_EVAL_DAEMON:-0}" != "1" && "${1:-}" == "" ]]; then
    mkdir -p "$LOG_BASE"
    echo "[INFO] Starting CCTS evaluation scheduler in background..."
    nohup env MEDDEF_CCTS_EVAL_DAEMON=1 bash "$SCRIPT_PATH" >> "$LOG_BASE/nohup.out" 2>&1 &
    BGPID=$!
    echo "[INFO] Scheduler PID : $BGPID"
    echo "[INFO] nohup log     : $LOG_BASE/nohup.out"
    echo "[INFO] Status        : bash run/eval_ccts_final.sh --status"
    echo "[INFO] Watch         : bash run/eval_ccts_final.sh --watch"
    exit 0
fi

# ── Entry point ───────────────────────────────────────────────────────────────
case "${1:-}" in
    --dry-run) cmd_dry_run ;;
    --status)  cmd_status  ;;
    --watch)   cmd_watch   ;;
    --stop)    cmd_stop    ;;
    --reset)   cmd_reset   ;;
    "")        cmd_run     ;;
    *) err "Unknown argument: $1. Valid: --dry-run --status --watch --stop --reset"; exit 1 ;;
esac
