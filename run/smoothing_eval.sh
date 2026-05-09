#!/usr/bin/env bash
################################################################################
# MedDef-VISTA — Randomized Smoothing Evaluation Scheduler
#
# Runs randomized_smoothing_eval.py for all 6 ablation variants across
# all available GPUs.  Mirrors the eval_tbcr_final.sh dispatch pattern:
#   ✓ Finds free GPUs by VRAM threshold
#   ✓ One variant per GPU (memory-heavy: ~4-6 GB per job due to N*batch)
#   ✓ Skips already-completed variants
#   ✓ Auto-retries on crash (up to MAX_RETRIES)
#   ✓ Survives SSH disconnect (nohup background jobs)
#   ✓ GPU cooldown after crash
#   ✓ Status / watch / stop modes
#
# Usage:
#   bash run/smoothing_eval.sh              # start all
#   bash run/smoothing_eval.sh --status     # show progress table
#   bash run/smoothing_eval.sh --watch      # live dashboard (refresh every 15s)
#   bash run/smoothing_eval.sh --stop       # kill all jobs
#   bash run/smoothing_eval.sh --reset      # wipe state, start over
#   bash run/smoothing_eval.sh --dry-run    # show what would run
#
# Override config via env:
#   GPU_IDS=2,3 bash run/smoothing_eval.sh
#   SIGMA=0.25  bash run/smoothing_eval.sh      # single sigma
#   FAST=1      bash run/smoothing_eval.sh      # fast mode (100 samples)
################################################################################

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
WHITE='\033[1;37m'; NC='\033[0m'; BOLD='\033[1m'

# ── Configuration ─────────────────────────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MIN_MEMORY_MB="${MIN_MEMORY_MB:-5000}"    # free VRAM needed before launching
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"   # scheduler poll interval (seconds)
GPU_COOLDOWN_SECS="${GPU_COOLDOWN_SECS:-60}"
MAX_RETRIES="${MAX_RETRIES:-2}"

VENV="${VENV:-/data2/enoch/.virtualenvs/meddef_final/bin/activate}"
PYTHON="${PYTHON:-python}"

DATA_DIR="${DATA_DIR:-/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/tbcr}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/visualizations/smoothing}"
N_SMOOTH="${N_SMOOTH:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-420}"
FAST="${FAST:-0}"       # set FAST=1 for quick 100-sample mode

# Variants (space-separated — matches training/eval scripts)
VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"

# Sigma sweep (all 4 by default; override with SIGMA=0.25 for single run)
if [[ -n "${SIGMA:-}" ]]; then
    SIGMA_ARG="--sigma $SIGMA"
else
    SIGMA_ARG=""
fi

FAST_ARG=""
[[ "${FAST:-0}" == "1" ]] && FAST_ARG="--fast"

# ── Derived paths ─────────────────────────────────────────────────────────────
LOG_BASE="${PROJECT_DIR}/logs/smoothing_eval"
STATE_DIR="${LOG_BASE}/state"
PID_FILE="${LOG_BASE}/scheduler.pid"
MASTER_LOG="${LOG_BASE}/master.log"

mkdir -p "$LOG_BASE" "$STATE_DIR"

# ── Logging ───────────────────────────────────────────────────────────────────
_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(_ts)] $*" | tee -a "$MASTER_LOG"; }
info() { log "[INFO ] $*"; }
ok()   { log "${GREEN}[DONE ]${NC} $*"; }
warn() { log "${YELLOW}[WARN ]${NC} $*"; }
err()  { log "${RED}[ERROR]${NC} $*" >&2; }

# ── GPU helpers ───────────────────────────────────────────────────────────────
gpu_list() { echo "$GPU_IDS" | tr ',' ' '; }

is_gpu_free() {
    local g="$1"
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           -i "$g" 2>/dev/null | tr -d ' ')
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

set_gpu_cooldown() {
    local g="$1"
    [[ "$GPU_COOLDOWN_SECS" -le 0 ]] && return
    echo $(( $(date +%s) + GPU_COOLDOWN_SECS )) > "$STATE_DIR/gpu_${g}.cooldown"
    info "GPU $g cooling down for ${GPU_COOLDOWN_SECS}s"
}

is_gpu_cooled_down() {
    local g="$1"
    local cf="$STATE_DIR/gpu_${g}.cooldown"
    [[ -f "$cf" ]] || return 0
    local until; until=$(cat "$cf" 2>/dev/null)
    if [[ $(date +%s) -ge "${until:-0}" ]]; then rm -f "$cf"; return 0; fi
    return 1
}

first_free_gpu() {
    local busy; busy=$(gpus_in_use | tr '\n' ' ')
    for g in $(gpu_list); do
        [[ " $busy " == *" $g "* ]] && continue
        is_gpu_cooled_down "$g" || continue
        is_gpu_free "$g"        && echo "$g" && return 0
    done
    return 1
}

# ── State helpers ─────────────────────────────────────────────────────────────
job_key()      { echo "$1"; }   # variant name is the key
state_file()   { echo "$STATE_DIR/$(job_key "$1")"; }
running_file() { echo "$(state_file "$1").running"; }
done_file()    { echo "$(state_file "$1").done"; }
retry_file()   { echo "$(state_file "$1").retries"; }

is_done()    { [[ -f "$(done_file "$1")" ]]; }
is_running() { [[ -f "$(running_file "$1")" ]]; }

job_pid_alive() {
    local rf; rf=$(running_file "$1")
    [[ -f "$rf" ]] || return 1
    local entry; entry=$(cat "$rf" 2>/dev/null)
    local pid; [[ "$entry" == *:* ]] && pid=${entry##*:} || pid=$entry
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

mark_running() {
    local variant="$1" pid="$2" gpu="$3"
    echo "${gpu}:${pid}" > "$(running_file "$variant")"
}

mark_done() {
    local variant="$1"
    touch "$(done_file "$variant")"
    rm -f "$(running_file "$variant")"
}

get_retries() {
    local rf; rf=$(retry_file "$1")
    [[ -f "$rf" ]] && cat "$rf" || echo 0
}

inc_retries() {
    local rf; rf=$(retry_file "$1")
    local n; n=$(get_retries "$1")
    echo $((n + 1)) > "$rf"
}

# ── Launch a single variant ───────────────────────────────────────────────────
launch_job() {
    local variant="$1" gpu="$2"
    local log_file="${LOG_BASE}/${variant}.log"
    mkdir -p "$OUT_DIR"

    info "[GPU $gpu] START  variant=$variant"

    (
        # shellcheck disable=SC1090
        source "$VENV"
        cd "$PROJECT_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        $PYTHON randomized_smoothing_eval.py \
            --variant     "$variant" \
            --device      0 \
            --out-dir     "$OUT_DIR" \
            --data        "$DATA_DIR" \
            --n-smooth    "$N_SMOOTH" \
            --max-samples "$MAX_SAMPLES" \
            $SIGMA_ARG \
            $FAST_ARG \
            2>&1
        echo "SMOOTH_EXIT_CODE:$?"
    ) >> "$log_file" &

    local pid=$!
    mark_running "$variant" "$pid" "$gpu"
    info "[GPU $gpu] PID=$pid  variant=$variant  log=$log_file"
}

# ── Harvest completed/crashed jobs ───────────────────────────────────────────
harvest() {
    for v in $VARIANTS; do
        is_running "$v" || continue
        job_pid_alive "$v" && continue   # still alive

        local lf="${LOG_BASE}/${v}.log"
        local rf; rf=$(running_file "$v")
        local entry; entry=$(cat "$rf" 2>/dev/null)
        local failed_gpu; [[ "$entry" == *:* ]] && failed_gpu=${entry%%:*} || failed_gpu=""

        local exit_line; exit_line=$(grep "SMOOTH_EXIT_CODE:" "$lf" 2>/dev/null | tail -1)
        if [[ "$exit_line" == *":0" ]]; then
            mark_done "$v"
            ok "Completed: $v"
        else
            rm -f "$rf"
            local retries; retries=$(get_retries "$v")
            if [[ -n "$failed_gpu" ]]; then
                set_gpu_cooldown "$failed_gpu"
            fi
            if [[ "$retries" -lt "$MAX_RETRIES" ]]; then
                inc_retries "$v"
                warn "Failed: $v (attempt $((retries+1))/${MAX_RETRIES}) — will retry"
            else
                err "Giving up on $v after $MAX_RETRIES retries"
                touch "$(state_file "$v").failed"
            fi
        fi
    done
}

# ── Pending variants ──────────────────────────────────────────────────────────
pending_variants() {
    for v in $VARIANTS; do
        is_done "$v"                         && continue
        [[ -f "$(state_file "$v").failed" ]] && continue
        is_running "$v" && job_pid_alive "$v" && continue
        echo "$v"
    done
}

# ── Status display ────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo -e "${BOLD}=== Randomized Smoothing Eval — Status ===${NC}"
    printf "%-20s %-10s %-10s %-8s\n" "VARIANT" "STATUS" "RETRIES" "GPU"
    echo "------------------------------------------------------------"
    for v in $VARIANTS; do
        local status gpu retries
        retries=$(get_retries "$v")
        if is_done "$v"; then
            status="${GREEN}done${NC}"
            gpu="—"
        elif [[ -f "$(state_file "$v").failed" ]]; then
            status="${RED}failed${NC}"
            gpu="—"
        elif is_running "$v" && job_pid_alive "$v"; then
            local entry; entry=$(cat "$(running_file "$v")" 2>/dev/null)
            [[ "$entry" == *:* ]] && gpu=${entry%%:*} || gpu="?"
            status="${YELLOW}running${NC}"
        else
            status="${CYAN}pending${NC}"
            gpu="—"
        fi
        printf "%-20s %-20b %-10s %-8s\n" "$v" "$status" "$retries" "$gpu"
    done

    echo ""
    echo "GPU Memory:"
    for g in $(gpu_list); do
        local free total
        free=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        printf "  GPU %s: %s / %s MiB free\n" "$g" "${free:-?}" "${total:-?}"
    done
    echo ""
}

# ── Watch mode ────────────────────────────────────────────────────────────────
watch_mode() {
    while true; do
        clear
        show_status
        echo "--- Last log lines ---"
        for v in $VARIANTS; do
            local lf="${LOG_BASE}/${v}.log"
            [[ -f "$lf" ]] || continue
            echo -e "\n${BOLD}$v${NC}:"
            tail -4 "$lf"
        done
        sleep 15
    done
}

# ── Stop all jobs ─────────────────────────────────────────────────────────────
stop_all() {
    echo "Stopping all smoothing eval jobs..."
    for rf in "$STATE_DIR"/*.running; do
        [[ -f "$rf" ]] || continue
        local entry; entry=$(cat "$rf" 2>/dev/null)
        local pid; [[ "$entry" == *:* ]] && pid=${entry##*:} || pid=$entry
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && echo "  Killed PID $pid"
        rm -f "$rf"
    done
    # Kill scheduler if running
    [[ -f "$PID_FILE" ]] && kill "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
    echo "Done."
}

# ── Reset state ───────────────────────────────────────────────────────────────
reset_state() {
    stop_all 2>/dev/null || true
    rm -rf "$STATE_DIR"
    mkdir -p "$STATE_DIR"
    echo "State reset. Re-run without --reset to start fresh."
}

# ── Main scheduler loop ───────────────────────────────────────────────────────
run_scheduler() {
    echo $$ > "$PID_FILE"
    info "Scheduler started (PID $$)"
    info "Variants : $VARIANTS"
    info "GPU pool : $GPU_IDS"
    info "N smooth : $N_SMOOTH"
    info "Max samp : $MAX_SAMPLES"
    info "Out dir  : $OUT_DIR"

    while true; do
        harvest

        # Check if all done or failed
        local remaining=0
        for v in $VARIANTS; do
            is_done "$v" && continue
            [[ -f "$(state_file "$v").failed" ]] && continue
            remaining=$((remaining + 1))
        done
        if [[ "$remaining" -eq 0 ]]; then
            info "All variants completed (or failed after retries)."
            rm -f "$PID_FILE"
            break
        fi

        # Try to dispatch pending variants onto free GPUs
        for v in $(pending_variants); do
            local gpu
            gpu=$(first_free_gpu) || break   # no free GPU right now
            launch_job "$v" "$gpu"
            sleep 5   # brief pause so nvidia-smi sees the new process
        done

        sleep "$CHECK_INTERVAL"
    done

    show_status
    info "Scheduler finished."
}

# ── Dry run ───────────────────────────────────────────────────────────────────
dry_run() {
    echo "=== DRY RUN ==="
    echo "Variants : $VARIANTS"
    echo "GPU IDs  : $GPU_IDS"
    echo "N smooth : $N_SMOOTH  (FAST=${FAST})"
    echo "Out dir  : $OUT_DIR"
    echo ""
    echo "Weights check:"
    # shellcheck disable=SC1090
    source "$VENV" 2>/dev/null || true
    for v in $VARIANTS; do
        for sub in "distill_v2/weights/best.pt" "distill/weights/best.pt"; do
            local wp="${PROJECT_DIR}/runs/classify/train_tbcr_final/tbcr/${v}_small/${sub}"
            if [[ -f "$wp" ]]; then
                echo "  ✓ $v  →  $wp"
                break
            fi
        done
    done
    echo ""
    echo "Data check:"
    for split in test val; do
        if [[ -d "${DATA_DIR}/${split}" ]]; then
            echo "  ✓ $DATA_DIR/$split"
            break
        fi
    done
}

# ── Entry point ───────────────────────────────────────────────────────────────
case "${1:-}" in
    --status)   show_status ;;
    --watch)    watch_mode  ;;
    --stop)     stop_all    ;;
    --reset)    reset_state ;;
    --dry-run)  dry_run     ;;
    "")
        # Start (or resume) the scheduler in foreground
        # Use nohup + & externally if you want it to survive SSH disconnect:
        #   nohup bash run/smoothing_eval.sh > /tmp/smoothing_sched.log 2>&1 &
        run_scheduler
        ;;
    *)
        echo "Unknown option: ${1}"
        echo "Usage: bash run/smoothing_eval.sh [--status|--watch|--stop|--reset|--dry-run]"
        exit 1
        ;;
esac
