#!/usr/bin/env bash
# cleanup_duplicate_runs.sh
# Run on the server:  bash run/cleanup_duplicate_runs.sh [--dry-run]
# Deletes redundant/failed classify runs after you have confirmed the keeper runs are intact.
# By default it is a DRY RUN. Pass --apply to actually delete.
#
# SAFE: only removes the runs listed in the "delete" section of ALL_RUNS_HYPERPARAMS_SUMMARY.txt

set -euo pipefail

BASE="${HOME}/ekd_coding_env/ultralytics/runs/classify"
DRY_RUN=true

for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && DRY_RUN=false
done

RUNS_TO_DELETE=(
  # TBCR lower performers / superseded
  "train_tbcr_legacy_v2"
  "train_tbcr_robust_v1"
  "train_tbcr_legacy_v2_impr2"
  "train_run_013_tbcr_95push"
  "train_run_tbcr_smart_20260318_allgpu"
  "train_run_012_robust_ft_20260312_fix1"
  # SCISIC failed / sanity
  "train_run_018_scisic_targeted_20260312_121506"
  "train_run_018_scisic_targeted_shared_20260312_122736"
  "train_run_019_scisic_rescue6_20260312_124208"
  "train_run_020_scisic_rescue6_ckptfix_20260312_131129"
  "train_run_014_sanity_ablation_20260312"
  "train_run_015_ablation_medium_20260312"
  # CCTS failed
  "train_ccts_legacy_v2"
  "train_ccts_robust_v1_impr1"
)

RUNS_TO_KEEP=(
  "train_tbcr_legacy_v2_impr1"
  "train"
  "archive"
  "train_run_010_legacy_plus"
  "train_ccts_robust_v1"
  "train_multic_robust_v1"
  "train_dermnet_cfg_v2"
)

echo "========================================================"
echo "  Classify Run Cleanup Script"
echo "  DRY_RUN=${DRY_RUN}  (pass --apply to delete for real)"
echo "========================================================"

# Sanity check: verify keeper runs exist
echo ""
echo "--- Verifying KEEPER runs exist ---"
for run in "${RUNS_TO_KEEP[@]}"; do
  path="${BASE}/${run}"
  if [[ -d "$path" ]]; then
    echo "  ✅  $run"
  else
    echo "  ⚠️   MISSING: $run  (skipping check)"
  fi
done

# Verify keeper best.pt files
echo ""
echo "--- Checking best.pt in critical keeper variants ---"
critical_variants=(
  "train_tbcr_legacy_v2_impr1/tbcr/full_small/weights/best.pt"
  "train_tbcr_legacy_v2_impr1/tbcr/no_cbam_small/weights/best.pt"
  "train/tbcr/full/weights/best.pt"
  "train/multic/full/weights/best.pt"
  "train/ccts/no_freq/weights/best.pt"
)
for chk in "${critical_variants[@]}"; do
  path="${BASE}/${chk}"
  if [[ -f "$path" ]]; then
    sz=$(du -sh "$path" | cut -f1)
    echo "  ✅  ${chk}  (${sz})"
  else
    echo "  ❌  MISSING: ${chk}  — ABORTING to avoid data loss!"
    exit 1
  fi
done

echo ""
echo "--- Runs to be deleted ---"
TOTAL_SIZE=0
for run in "${RUNS_TO_DELETE[@]}"; do
  path="${BASE}/${run}"
  if [[ -d "$path" ]]; then
    sz=$(du -sh "$path" 2>/dev/null | cut -f1)
    echo "  🗑️   ${run}  (${sz})"
  else
    echo "  (already gone) ${run}"
  fi
done

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN complete. No files deleted."
  echo "Re-run with  --apply  to actually delete."
else
  echo "--- DELETING ---"
  for run in "${RUNS_TO_DELETE[@]}"; do
    path="${BASE}/${run}"
    if [[ -d "$path" ]]; then
      echo "  rm -rf $path"
      rm -rf "$path"
      echo "  ✅ deleted"
    fi
  done
  echo ""
  echo "Cleanup complete."
  echo "Remaining runs:"
  ls -1 "${BASE}/"
fi
