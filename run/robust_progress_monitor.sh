#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data2/enoch/ekd_coding_env/ultralytics}"
DATASETS="${DATASETS:-ccts tbcr dermnet multic}"
VARIANTS="${VARIANTS:-full no_def no_freq no_patch no_cbam baseline}"
REFRESH_SEC="${REFRESH_SEC:-60}"
ONCE="${ONCE:-false}"

cd "$PROJECT_DIR"

run_tag_for_dataset() {
  local ds="$1"
  local root="$PROJECT_DIR/runs/classify"
  local latest_tag=""
  local latest_mtime=0

  # Optional manual override per dataset, e.g. RUN_TAG_TBCR=tbcr_legacy_v2_impr1
  local ds_upper="${ds^^}"
  local override_var="RUN_TAG_${ds_upper}"
  local override="${!override_var:-}"
  if [[ -n "$override" && -d "$root/train_${override}/${ds}" ]]; then
    echo "$override"
    return 0
  fi

  # Auto-detect newest run tag that has results for this dataset.
  local d
  for d in "$root"/train_*; do
    [[ -d "$d/${ds}" ]] || continue
    local tag
    tag=$(basename "$d" | sed 's/^train_//')

    local newest_for_tag=0
    local rf
    while IFS= read -r rf; do
      local m
      m=$(stat -c %Y "$rf" 2>/dev/null || echo 0)
      [[ "$m" -gt "$newest_for_tag" ]] && newest_for_tag="$m"
    done < <(find "$d/${ds}" -type f -name results.csv 2>/dev/null)

    [[ "$newest_for_tag" -gt "$latest_mtime" ]] || continue
    latest_mtime="$newest_for_tag"
    latest_tag="$tag"
  done

  echo "$latest_tag"
}

print_header() {
  echo "----------------------------------------------------------------------------------------------"
  printf "%-8s %-18s %-9s %-14s %-8s %-8s %-20s %-8s\n" "dataset" "run_tag" "variant" "state" "epoch" "top1" "mtime" "file"
  echo "----------------------------------------------------------------------------------------------"
}

latest_row() {
  local f="$1"
  tail -n 1 "$f" 2>/dev/null || true
}

find_variant_results_csv() {
  local run_tag="$1"
  local ds="$2"
  local variant="$3"
  local base="$PROJECT_DIR/runs/classify/train_${run_tag}/${ds}"

  [[ -d "$base" ]] || { echo ""; return 0; }

  local best=""
  local best_mtime=0
  local rf
  while IFS= read -r rf; do
    local dirname
    dirname=$(basename "$(dirname "$rf")")
    [[ "$dirname" == "$variant" || "$dirname" == "${variant}_"* ]] || continue
    local m
    m=$(stat -c %Y "$rf" 2>/dev/null || echo 0)
    if [[ "$m" -gt "$best_mtime" ]]; then
      best_mtime="$m"
      best="$rf"
    fi
  done < <(find "$base" -type f -name results.csv 2>/dev/null)

  echo "$best"
}

is_numeric_epoch() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

show_once() {
  local now
  now=$(date '+%F %T')
  echo
  echo "[robust-monitor] $now  project=$PROJECT_DIR"
  print_header

  for ds in $DATASETS; do
    local run_tag
    run_tag=$(run_tag_for_dataset "$ds")

    for v in $VARIANTS; do
      local f
      f=$(find_variant_results_csv "$run_tag" "$ds" "$v")

      if [[ -z "$f" ]]; then
        printf "%-8s %-18s %-9s %-14s %-8s %-8s %-20s %-8s\n" "$ds" "${run_tag:--}" "$v" "not-started" "-" "-" "-" "-"
        continue
      fi

      local row
      row=$(latest_row "$f")
      if [[ -z "$row" ]]; then
        printf "%-8s %-18s %-9s %-14s %-8s %-8s %-20s %-8s\n" "$ds" "${run_tag:--}" "$v" "empty" "-" "-" "-" "csv"
        continue
      fi

      local ep top1 mtime state
      ep=$(echo "$row" | cut -d',' -f1 | tr -d '[:space:]')
      top1=$(echo "$row" | cut -d',' -f5 | tr -d '[:space:]')
      mtime=$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1)

      if is_numeric_epoch "$ep"; then
        state="ok"
      else
        state="header"
      fi

      local shortf
      shortf=$(basename "$(dirname "$f")")
      printf "%-8s %-18s %-9s %-14s %-8s %-8s %-20s %-8s\n" "$ds" "${run_tag:--}" "$v" "$state" "$ep" "$top1" "$mtime" "$shortf"
    done
  done
  echo "----------------------------------------------------------------------------------------------"
}

main() {
  if [[ "$ONCE" == "true" ]]; then
    show_once
    return 0
  fi

  while true; do
    show_once
    sleep "$REFRESH_SEC"
  done
}

main
