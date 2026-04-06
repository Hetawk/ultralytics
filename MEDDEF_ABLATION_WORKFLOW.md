# MedDef2 Ablation Study Workflow

## Complete Training & Evaluation Pipeline

This document outlines the complete workflow for running the MedDef2 ablation study with 6 model variants.

## Quick Summary

**Default Configuration:**
- Depth: `small` (3.4M params) - **NO NEED TO SPECIFY**
- All variants use small depth by default
- Training time: ~2-3 hours per variant (100 epochs)

## Step 1: Prepare Your Dataset

Create a `data.yaml` file in your dataset directory:

```yaml
path: /absolute/path/to/skin_cancer
train: images/train
val: images/val
test: images/test  # optional
nc: 10
names: ['melanoma', 'nevus', 'basal_cell_carcinoma', ...]
```

Verify directory structure:
```
skin_cancer/
├── images/
│   ├── train/
│   │   ├── class1/ (contains *.jpg, *.png files)
│   │   ├── class2/
│   │   └── ...
│   ├── val/
│   │   ├── class1/
│   │   └── ...
│   └── test/
│       ├── class1/
│       └── ...
└── data.yaml
```

## Step 2: Train All 6 Variants

Navigate to the training directory:
```bash
cd /data2/enoch/ekd_coding_env/ultralytics
```

### Option A: Train Each Variant Manually

```bash
# 1. Full Model (All defenses: CBAM + Freq + Patch + DefenseModule)
python train_meddef.py --data path/to/skin_cancer --variant full --epochs 100 --batch 16 --device 0 --name full

# 2. No DefenseModule (Remove one defense component)
python train_meddef.py --data path/to/skin_cancer --variant no_def --epochs 100 --batch 16 --device 0 --name no_def

# 3. No FrequencyDefense
python train_meddef.py --data path/to/skin_cancer --variant no_freq --epochs 100 --batch 16 --device 0 --name no_freq

# 4. No PatchConsistency
python train_meddef.py --data path/to/skin_cancer --variant no_patch --epochs 100 --batch 16 --device 0 --name no_patch

# 5. No CBAM (Standard transformer blocks)
python train_meddef.py --data path/to/skin_cancer --variant no_cbam --epochs 100 --batch 16 --device 0 --name no_cbam

# 6. Baseline (No defenses - Pure ViT for control)
python train_meddef.py --data path/to/skin_cancer --variant baseline --epochs 100 --batch 16 --device 0 --name baseline
```

### Option B: Run All Variants in Sequence (Bash Script)

Create `run_ablation.sh`:

```bash
#!/bin/bash
cd /data2/enoch/ekd_coding_env/ultralytics
DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_ablation.sh path/to/dataset"
    exit 1
fi

VARIANTS=("full" "no_def" "no_freq" "no_patch" "no_cbam" "baseline")
EPOCHS=100
BATCH=16
DEVICE=0

for variant in "${VARIANTS[@]}"; do
    echo "=========================================="
    echo "Training variant: $variant"
    echo "=========================================="
    python train_meddef.py --data "$DATASET" --variant "$variant" --epochs $EPOCHS --batch $BATCH --device $DEVICE --name "$variant"
done

echo ""
echo "=========================================="
echo "All variants trained successfully!"
echo "=========================================="
```

Usage:
```bash
bash run_ablation.sh /path/to/skin_cancer
```

## Step 3: Monitor Training

Training progress is saved to `runs/classify/train/skin_cancer/{variant}/`:

```
runs/classify/train/skin_cancer/full/
├── args.yaml                          # Reproducible args
├── results.csv                        # Per-epoch metrics
├── results.png                        # Training curves
├── confusion_matrix.png               # Classification matrix
├── confusion_matrix_normalized.png
├── weights/
│   ├── best.pt                        # Best checkpoint
│   └── last.pt                        # Latest checkpoint
└── ...
```

Real-time monitoring:
```bash
# Watch results for a specific variant
watch "cat runs/classify/train/skin_cancer/full/results.csv | tail -5"

# View training curve
display runs/classify/train/skin_cancer/full/results.png  # or use any image viewer

# View confusion matrix
display runs/classify/train/skin_cancer/full/confusion_matrix.png
```

## Step 4: Evaluate on Test Set

After training all variants, test them on the held-out test split:

```bash
# Test full model
python train_meddef.py --data path/to/skin_cancer \
    --model runs/classify/train/skin_cancer/full/weights/best.pt \
    --variant full \
    --device 0 \
    split=test

# Test all variants
for variant in full no_def no_freq no_patch no_cbam baseline; do
    echo "Testing $variant..."
    python train_meddef.py --data path/to/skin_cancer \
        --model runs/classify/train/skin_cancer/$variant/weights/best.pt \
        --variant $variant \
        --device 0 \
        split=test
done
```

## Step 5: Compare Results

Analyze the results across variants:

### Create Comparison Table

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results from each variant
variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']
results_dir = Path('runs/classify/train/skin_cancer')

data = {}
for variant in variants:
    csv_path = results_dir / variant / 'results.csv'
    df = pd.read_csv(csv_path)
    
    # Extract final metrics (last row)
    final_row = df.iloc[-1]
    data[variant] = {
        'train_loss': final_row.get('train/loss'),
        'val_loss': final_row.get('val/loss'),
        'top1_acc': final_row.get('metrics/top1_acc'),
        'top5_acc': final_row.get('metrics/top5_acc'),
    }

# Create comparison table
comparison_df = pd.DataFrame(data).T
print(comparison_df)

# Plot comparison
comparison_df[['top1_acc', 'top5_acc']].plot(kind='bar')
plt.title('Accuracy Comparison Across Variants')
plt.ylabel('Accuracy')
plt.xlabel('Variant')
plt.legend(['Top-1', 'Top-5'])
plt.tight_layout()
plt.savefig('variant_comparison.png')
plt.show()
```

### Compare Defense Components

| Variant | CBAM | Freq | Patch | Defense | Parameters |
|---------|:----:|:----:|:-----:|:-------:|:----------:|
| full | ✓ | ✓ | ✓ | ✓ | 3.4M |
| no_def | ✓ | ✓ | ✓ | ✗ | 2.9M |
| no_freq | ✓ | ✗ | ✓ | ✓ | 3.4M |
| no_patch | ✓ | ✓ | ✗ | ✓ | 3.4M |
| no_cbam | ✗ | ✓ | ✓ | ✓ | 3.4M |
| baseline | ✗ | ✗ | ✗ | ✗ | 2.9M |

## Step 6: Analyze Ablation Results

Key comparisons:

```
1. Full vs Baseline
   → Measures total impact of all defense mechanisms
   → Expected: Full > Baseline (more robust)

2. Full vs No_Def
   → Measures DefenseModule contribution
   → Expected: Full > No_Def

3. Full vs No_Freq
   → Measures FrequencyDefense contribution
   → Expected: Impact on adversarial robustness

4. Full vs No_Patch
   → Measures PatchConsistency contribution
   → Expected: Impact on adversarial robustness

5. Full vs No_CBAM
   → Measures CBAM attention contribution
   → Expected: Impact on standard accuracy

6. No_CBAM vs Baseline
   → Measures FreqDefense + PatchConsistency + DefenseModule
   → Expected: Shows value of defenses in standard ViT
```

## Step 7: Generate Reports

### Automatic Summary

```python
import json
from pathlib import Path

results_dir = Path('runs/classify/train/skin_cancer')
variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']

summary = {}
for variant in variants:
    args_path = results_dir / variant / 'args.yaml'
    results_path = results_dir / variant / 'results.csv'
    
    # Load final metrics
    import yaml
    with open(args_path) as f:
        args = yaml.safe_load(f)
    
    df = pd.read_csv(results_path)
    final = df.iloc[-1]
    
    summary[variant] = {
        'epochs': args.get('epochs'),
        'batch_size': args.get('batch'),
        'best_top1_acc': float(final.get('metrics/top1_acc')),
        'best_top5_acc': float(final.get('metrics/top5_acc')),
        'final_val_loss': float(final.get('val/loss')),
    }

# Save summary
with open('ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
```

## Common Commands Reference

```bash
# Train a single variant
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --device 0

# Resume training (if interrupted)
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --device 0 --resume

# Train with specific experiment name
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --device 0 --name exp_v1

# Use multiple GPUs
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --device 0,1

# Reduce batch size (if out of memory)
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --batch 8 --device 0

# Custom learning rate
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --lr0 0.0005 --device 0
```

## Troubleshooting

**Q: Training is very slow**
- Reduce workers: `--workers 2` (default is 8)
- Check GPU usage: `nvidia-smi`

**Q: Out of memory error**
- Reduce batch size: `--batch 8` or `--batch 4`
- Reduce image size: `--imgsz 192`

**Q: Results not appearing in results.csv**
- Check that validation split has class folders
- Verify data.yaml is correct
- Check training output for errors

**Q: How do I reproduce a training run?**
- All args are saved in `runs/classify/train/.../args.yaml`
- Copy the args and rerun with same settings

**Q: Can I train with pretrained weights?**
- Yes: `python train_meddef.py --data ... --pretrained path/to/weights.pt`

## Next Steps After Ablation Study

1. **Statistical Analysis**: Run confidence intervals on results
2. **Adversarial Testing**: Evaluate robustness on adversarial examples
3. **Inference Speed**: Benchmark inference time per variant
4. **Deployment**: Export best model for production
5. **Documentation**: Write up findings in paper/report

---

**Remember**: All variants use the `small` depth by default. No need to specify depth unless you want a different size (tiny, base, or large).
