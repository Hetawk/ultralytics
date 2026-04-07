#!/usr/bin/env bash
# =============================================================================
# ONNX Export Script for MedDef-VISTA (MedDef2) Models
# =============================================================================
# Exports all trained MedDef2 variants to ONNX format for deployment.
#
# Usage (on server):
#   cd /data2/enoch/ekd_coding_env/ultralytics
#   source /data2/enoch/.virtualenvs/meddef_final/bin/activate
#   bash run/export_onnx.sh [--check-only] [--variant VARIANT] [--stage STAGE]
#
# Options:
#   --check-only    Only check which models are ready, don't export
#   --variant NAME  Export only this variant (full, no_def, no_freq, no_patch, no_cbam, baseline)
#   --stage NAME    Export only this stage (stage1, distill, distill_v2)
# =============================================================================

set -euo pipefail

TRAIN_ROOT="runs/classify/train_tbcr_final/tbcr"
VARIANTS=(full no_def no_freq no_patch no_cbam baseline)
STAGES=(stage1 distill distill_v2)
EXPORT_DIR="runs/onnx_exports/tbcr"
IMGSZ=224

# Parse arguments
CHECK_ONLY=false
FILTER_VARIANT=""
FILTER_STAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=true; shift ;;
        --variant) FILTER_VARIANT="$2"; shift 2 ;;
        --stage) FILTER_STAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo "  MedDef-VISTA ONNX Export"
echo "  $(date)"
echo "============================================="

mkdir -p "$EXPORT_DIR"

TOTAL=0
READY=0
EXPORTED=0
SKIPPED=0

for variant in "${VARIANTS[@]}"; do
    [[ -n "$FILTER_VARIANT" && "$variant" != "$FILTER_VARIANT" ]] && continue

    for stage in "${STAGES[@]}"; do
        [[ -n "$FILTER_STAGE" && "$stage" != "$FILTER_STAGE" ]] && continue

        TOTAL=$((TOTAL + 1))
        model_dir="${TRAIN_ROOT}/${variant}_small"

        # stage1 weights are in the root, distill/distill_v2 in subdirs
        if [[ "$stage" == "stage1" ]]; then
            weights="${model_dir}/weights/best.pt"
        else
            weights="${model_dir}/${stage}/weights/best.pt"
        fi

        # Output: mirror the training structure under onnx_exports
        # e.g. runs/onnx_exports/tbcr/full_small/stage1/best.onnx
        if [[ "$stage" == "stage1" ]]; then
            onnx_dir="${EXPORT_DIR}/${variant}_small/stage1"
        else
            onnx_dir="${EXPORT_DIR}/${variant}_small/${stage}"
        fi
        onnx_out="${onnx_dir}/best.onnx"

        # Check if weights exist
        if [[ ! -f "$weights" ]]; then
            echo "[ SKIP ] ${variant}/${stage} — weights not found: $weights"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Check if training is complete (results.csv has enough rows)
        if [[ "$stage" == "stage1" ]]; then
            results_csv="${model_dir}/results.csv"
            expected_epochs=160
        else
            results_csv="${model_dir}/${stage}/results.csv"
            expected_epochs=100
        fi

        if [[ -f "$results_csv" ]]; then
            rows=$(wc -l < "$results_csv")
            actual_epochs=$((rows - 1))  # subtract header
            if [[ $actual_epochs -lt $expected_epochs ]]; then
                echo "[ WAIT ] ${variant}/${stage} — training at epoch ${actual_epochs}/${expected_epochs}"
                continue
            fi
        fi

        READY=$((READY + 1))

        # Check if already exported
        if [[ -f "$onnx_out" ]]; then
            echo "[  OK  ] ${variant}/${stage} — already exported: $onnx_out"
            EXPORTED=$((EXPORTED + 1))
            continue
        fi

        if [[ "$CHECK_ONLY" == true ]]; then
            echo "[READY ] ${variant}/${stage} — ready for export"
            continue
        fi

        # Export to ONNX using ultralytics MedDef API
        echo "[EXPORT] ${variant}/${stage} — exporting..."
        mkdir -p "$onnx_dir"
        python -c "
import shutil
from ultralytics import MedDef

model = MedDef('${weights}')
exported = model.export(format='onnx', imgsz=${IMGSZ}, simplify=True, dynamic=False, half=False)
print(f'Ultralytics exported to: {exported}')
shutil.move(str(exported), '${onnx_out}')
print(f'Moved to: ${onnx_out}')
" 2>&1 | tail -5

        if [[ -f "$onnx_out" ]]; then
            size=$(du -h "$onnx_out" | cut -f1)
            echo "[  OK  ] ${variant}/${stage} — exported ($size)"
            EXPORTED=$((EXPORTED + 1))
        else
            echo "[ FAIL ] ${variant}/${stage} — export failed"
        fi
    done
done

echo ""
echo "============================================="
echo "  Summary: ${EXPORTED} exported, ${READY} ready, ${SKIPPED} skipped, ${TOTAL} total"
echo "============================================="

# List all exported models in tree structure
if [[ -d "$EXPORT_DIR" ]]; then
    echo ""
    echo "Exported models:"
    find "$EXPORT_DIR" -name '*.onnx' -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
    if [[ $EXPORTED -eq 0 ]] && [[ "$CHECK_ONLY" == false ]]; then
        echo "  (none yet)"
    fi
fi
