#!/bin/bash
# MedDef2 Complete Training & Ablation Study Example
# 
# This script demonstrates the complete workflow for training MedDef2 models
# with all 6 ablation variants on a dataset.
#
# Usage:
#   bash run_complete_ablation_study.sh /path/to/dataset [epochs] [batch] [device]
#
# Example:
#   bash run_complete_ablation_study.sh ~/data/skin_cancer 100 16 0
#   bash run_complete_ablation_study.sh ~/data/medical_images 50 32 0,1

set -e  # Exit on error

# Configuration
DATASET="${1:?Error: Please provide dataset path as first argument}"
EPOCHS="${2:-100}"
BATCH="${3:-16}"
DEVICE="${4:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define variants
VARIANTS=("full" "no_def" "no_freq" "no_patch" "no_cbam" "baseline")

# Validate dataset
if [ ! -d "$DATASET" ]; then
    echo -e "${RED}Error: Dataset directory does not exist: $DATASET${NC}"
    exit 1
fi

if [ ! -f "$DATASET/data.yaml" ]; then
    echo -e "${YELLOW}Warning: data.yaml not found in $DATASET${NC}"
    echo "Expected format:"
    echo "  path: /absolute/path/to/dataset"
    echo "  train: images/train"
    echo "  val: images/val"
    echo "  test: images/test"
    echo "  nc: 10"
    echo "  names: [...]"
fi

# Print configuration
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}MedDef2 Ablation Study${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset:     $DATASET"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH"
echo "  Device:      $DEVICE"
echo "  Variants:    ${VARIANTS[@]}"
echo -e "${BLUE}================================${NC}"
echo ""

# Change to ultralytics directory
cd /data2/enoch/ekd_coding_env/ultralytics

# Track results
declare -A results

# Function to train a variant
train_variant() {
    local variant=$1
    local start_time=$(date +%s)
    
    echo -e "${BLUE}>>> Training variant: ${GREEN}$variant${NC}"
    
    python train_meddef.py \
        --data "$DATASET" \
        --variant "$variant" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --device "$DEVICE" \
        --name "$variant"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Store result
    results[$variant]=$duration
    
    echo -e "${GREEN}✓ $variant completed in ${duration}s${NC}"
    echo ""
}

# Train all variants
total_start=$(date +%s)

for variant in "${VARIANTS[@]}"; do
    train_variant "$variant"
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Training Summary${NC}"
echo -e "${BLUE}================================${NC}"

for variant in "${VARIANTS[@]}"; do
    duration=${results[$variant]}
    echo "  $variant: ${duration}s"
done

echo -e "${GREEN}Total training time: ${total_duration}s (~$((total_duration / 60))m)${NC}"
echo ""

# Test on test split if available
if grep -q "^test:" "$DATASET/data.yaml" 2>/dev/null; then
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Testing on Test Set${NC}"
    echo -e "${BLUE}================================${NC}"
    
    for variant in "${VARIANTS[@]}"; do
        echo -e "${BLUE}Testing: ${GREEN}$variant${NC}"
        # Note: This would test on test split if implemented
        # For now, this is a placeholder
    done
fi

# Generate comparison summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Generating Comparison Report${NC}"
echo -e "${BLUE}================================${NC}"

python3 << 'PYTHON_SCRIPT'
import pandas as pd
import json
from pathlib import Path
import yaml

variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']
results_dir = Path('runs/classify/train')

# Find the dataset directory
dataset_dirs = list(results_dir.glob('*/'))
if dataset_dirs:
    dataset_name = dataset_dirs[0].name
    dataset_results = results_dir / dataset_name
    
    print(f"\n{'='*60}")
    print(f"Results Directory: {dataset_results}")
    print(f"{'='*60}\n")
    
    summary = {}
    
    for variant in variants:
        variant_dir = dataset_results / variant
        if variant_dir.exists():
            results_csv = variant_dir / 'results.csv'
            args_yaml = variant_dir / 'args.yaml'
            
            try:
                # Load results
                df = pd.read_csv(results_csv)
                final = df.iloc[-1]
                
                # Load args for reproducibility
                with open(args_yaml) as f:
                    args = yaml.safe_load(f)
                
                summary[variant] = {
                    'top1_accuracy': float(final.get('metrics/top1_acc', 0)),
                    'top5_accuracy': float(final.get('metrics/top5_acc', 0)),
                    'val_loss': float(final.get('val/loss', 0)),
                    'train_loss': float(final.get('train/loss', 0)),
                    'epochs_trained': args.get('epochs', 0),
                }
            except Exception as e:
                print(f"Could not load results for {variant}: {e}")
    
    # Print comparison table
    if summary:
        print(f"{'Variant':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Val Loss':<12} {'Train Loss':<12}")
        print(f"{'-'*60}")
        
        for variant in variants:
            if variant in summary:
                s = summary[variant]
                print(f"{variant:<15} {s['top1_accuracy']:<12.4f} {s['top5_accuracy']:<12.4f} "
                      f"{s['val_loss']:<12.4f} {s['train_loss']:<12.4f}")
        
        print(f"\n{'='*60}")
        print("Key Observations:")
        print(f"{'='*60}")
        
        # Find best and worst
        best_variant = max(summary, key=lambda x: summary[x]['top1_accuracy'])
        worst_variant = min(summary, key=lambda x: summary[x]['top1_accuracy'])
        
        print(f"\n✓ Best Top-1 Accuracy:  {best_variant} ({summary[best_variant]['top1_accuracy']:.4f})")
        print(f"✗ Worst Top-1 Accuracy: {worst_variant} ({summary[worst_variant]['top1_accuracy']:.4f})")
        
        # Ablation insights
        if 'full' in summary and 'baseline' in summary:
            improvement = summary['full']['top1_accuracy'] - summary['baseline']['top1_accuracy']
            print(f"\n→ Total Defense Impact: {improvement:+.4f}")
        
        if 'full' in summary and 'no_def' in summary:
            improvement = summary['full']['top1_accuracy'] - summary['no_def']['top1_accuracy']
            print(f"→ DefenseModule Impact: {improvement:+.4f}")
        
        # Save detailed results
        with open('ablation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n→ Detailed results saved to: ablation_results.json")

PYTHON_SCRIPT

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Ablation study complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Results saved to:"
echo "  - runs/classify/train/"
echo "  - ablation_results.json"
echo ""
echo "Next steps:"
echo "  1. Check confusion matrices: display runs/classify/train/*/*/confusion_matrix.png"
echo "  2. View training curves: display runs/classify/train/*/*/results.png"
echo "  3. Analyze results: python3 analyze_ablation.py"
echo ""
