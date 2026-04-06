# MedDef2: Complete Training System

## Overview

MedDef2 is a **Vision Transformer-based medical image classification system** with built-in adversarial defense mechanisms. This implementation provides:

- ✅ **6 Ablation Variants**: Full, no_def, no_freq, no_patch, no_cbam, baseline
- ✅ **4 Model Depths**: tiny, small (default), base, large  
- ✅ **Easy CLI Interface**: Simple Python script for training
- ✅ **Complete Output Structure**: Automatic saving of args, metrics, weights, plots, confusion matrices
- ✅ **Batch Processing**: Scripts to train all variants automatically
- ✅ **Reproducibility**: All training arguments saved in args.yaml

## Quick Start (30 seconds)

```bash
# 1. Navigate to the ultralytics directory
cd /data2/enoch/ekd_coding_env/ultralytics

# 2. Prepare your dataset with data.yaml (see Dataset Preparation below)

# 3. Train the full MedDef2 model (uses small depth by default)
python train_meddef.py --data /path/to/skin_cancer --variant full --epochs 100 --batch 16 --device 0 --name my_experiment

# 4. Check results
ls runs/classify/train/skin_cancer/my_experiment/
```

**That's it!** Results automatically saved with confusion matrices, training curves, and best weights.

## Key Features

### 1. Default Configuration
- **Depth**: `small` (3.4M parameters) - **NO NEED TO SPECIFY**
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Batch Size**: 16
- All other standard training hyperparameters available via CLI

### 2. 6 Ablation Variants

Test the contribution of each defense component:

| Variant | CBAM | Freq | Patch | Defense | Use Case |
|---------|:----:|:----:|:-----:|:-------:|----------|
| **full** | ✓ | ✓ | ✓ | ✓ | Complete defense model |
| **no_def** | ✓ | ✓ | ✓ | ✗ | Remove DefenseModule |
| **no_freq** | ✓ | ✗ | ✓ | ✓ | Remove FrequencyDefense |
| **no_patch** | ✓ | ✓ | ✗ | ✓ | Remove PatchConsistency |
| **no_cbam** | ✗ | ✓ | ✓ | ✓ | Standard transformer blocks |
| **baseline** | ✗ | ✗ | ✗ | ✗ | Pure ViT (control) |

### 3. Automatic Outputs

Each training run saves:

```
runs/classify/train/{dataset}/{experiment_name}/
├── args.yaml                          # Reproducible configuration
├── results.csv                        # Per-epoch metrics
├── results.png                        # Training curves
├── confusion_matrix.png               # Classification matrix
├── confusion_matrix_normalized.png    # Normalized version
├── train_batch*.jpg                   # Training samples
├── val_batch*_labels.jpg              # Validation ground truth
├── val_batch*_pred.jpg                # Validation predictions
└── weights/
    ├── best.pt                        # Best checkpoint
    └── last.pt                        # Latest checkpoint
```

## Installation

All dependencies are pre-installed. Verify with:

```bash
python -c "from ultralytics.models.meddef import MedDefTrainer; print('MedDef ready!')"
```

## Dataset Preparation

### Format

Your dataset should have a `data.yaml` file:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 10  # number of classes
names:
  - class1
  - class2
  - ...
```

### Directory Structure

```
skin_cancer/
├── images/
│   ├── train/
│   │   ├── melanoma/ (contains *.jpg files)
│   │   ├── nevus/
│   │   └── ...
│   ├── val/
│   │   ├── melanoma/
│   │   └── ...
│   └── test/
│       ├── melanoma/
│       └── ...
└── data.yaml
```

## Training Guide

### Basic Training

```bash
cd /data2/enoch/ekd_coding_env/ultralytics

# Train full model on your dataset
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --batch 16 --device 0
```

### Training All Variants

```bash
# Option 1: Use automated script
bash run_complete_ablation_study.sh /path/to/dataset 100 16 0

# Option 2: Manual training
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --device 0
python train_meddef.py --data /path/to/dataset --variant no_def --epochs 100 --device 0
python train_meddef.py --data /path/to/dataset --variant no_freq --epochs 100 --device 0
python train_meddef.py --data /path/to/dataset --variant no_patch --epochs 100 --device 0
python train_meddef.py --data /path/to/dataset --variant no_cbam --epochs 100 --device 0
python train_meddef.py --data /path/to/dataset --variant baseline --epochs 100 --device 0
```

### Available Arguments

```
python train_meddef.py --help

  --data DATA                      Path to dataset (REQUIRED)
  --variant {full,no_def,...}      Model variant (default: full)
  --depth {tiny,small,base,large}  Model depth (default: small)
  --epochs EPOCHS                  Number of epochs (default: 100)
  --batch BATCH                    Batch size (default: 16)
  --imgsz IMGSZ                    Image size (default: 224)
  --device DEVICE                  CUDA device (default: 0)
  --workers WORKERS                Dataloader workers (default: 8)
  --name NAME                      Experiment name
  --project PROJECT                Project directory
  --lr0 LR0                         Learning rate (default: 0.001)
  --optimizer OPTIMIZER            Optimizer (default: AdamW)
  --resume                         Resume training flag
  --pretrained PRETRAINED          Path to pretrained weights
```

### Example Commands

```bash
# Simple training
python train_meddef.py --data ~/datasets/skin_cancer --variant full --epochs 50

# Full configuration
python train_meddef.py \
    --data ~/datasets/skin_cancer \
    --variant full \
    --epochs 100 \
    --batch 32 \
    --device 0 \
    --name exp_v1 \
    --lr0 0.0005

# Resume interrupted training
python train_meddef.py --data ~/datasets/skin_cancer --variant full --resume

# With pretrained weights
python train_meddef.py --data ~/datasets/skin_cancer --variant full --pretrained weights.pt

# Multiple GPUs
python train_meddef.py --data ~/datasets/skin_cancer --variant full --device 0,1
```

## Results Analysis

### View Results During Training

```bash
# Watch training progress
watch "tail -5 runs/classify/train/skin_cancer/full/results.csv"

# View training curves
display runs/classify/train/skin_cancer/full/results.png

# View confusion matrix
display runs/classify/train/skin_cancer/full/confusion_matrix.png
```

### Compare Variants

```python
import pandas as pd
from pathlib import Path

# Load results from all variants
variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']
results_dir = Path('runs/classify/train/skin_cancer')

for variant in variants:
    df = pd.read_csv(results_dir / variant / 'results.csv')
    final = df.iloc[-1]
    print(f"{variant:12} Top-1: {final['metrics/top1_acc']:.4f}, Loss: {final['val/loss']:.4f}")
```

### Generate Summary Report

```bash
python3 << 'EOF'
import json
import pandas as pd
from pathlib import Path
import yaml

variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']
results_dir = Path('runs/classify/train/skin_cancer')

summary = {}
for variant in variants:
    df = pd.read_csv(results_dir / variant / 'results.csv')
    final = df.iloc[-1]
    summary[variant] = {
        'top1_acc': float(final['metrics/top1_acc']),
        'top5_acc': float(final['metrics/top5_acc']),
        'val_loss': float(final['val/loss']),
    }

# Print table
print(f"{'Variant':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Val Loss':<12}")
print("-" * 50)
for v in variants:
    s = summary[v]
    print(f"{v:<12} {s['top1_acc']:.4f}      {s['top5_acc']:.4f}      {s['val_loss']:.4f}")

# Save
with open('ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
EOF
```

## Python API

Use MedDef2 directly in your code:

```python
import torch
from ultralytics.models.meddef.meddef2 import get_variant

# Create a model
model = get_variant('full', 'small', num_classes=10)
model.eval()

# Forward pass
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)  # shape: [1, 10]
    probabilities = torch.softmax(out, dim=1)

print(probabilities)
```

### Training with Python

```python
from ultralytics.models.meddef import MedDefTrainer

trainer = MedDefTrainer(overrides={
    'model': 'meddef2.yaml',
    'data': 'path/to/data.yaml',
    'epochs': 100,
    'batch': 16,
    'device': 0,
    'name': 'experiment1'
})

# Train
trainer.train()

# Validate
results = trainer.val()
print(f"Top-1 Accuracy: {results['top1_acc']:.4f}")
```

## File Structure

### Training Script
- `train_meddef.py` - Main training CLI

### Model Implementation
- `ultralytics/models/meddef/` - MedDef module
  - `model.py` - MedDefModel class
  - `train.py` - MedDefTrainer class
  - `val.py` - MedDefValidator class
  - `predict.py` - MedDefPredictor class
  - `meddef2/variants.py` - 6 ablation variants

### Configs
- `ultralytics/cfg/models/meddef/` - YAML configs for each variant
  - `meddef2.yaml` - Full model
  - `meddef2_no_def.yaml` - No DefenseModule
  - `meddef2_no_freq.yaml` - No FrequencyDefense
  - `meddef2_no_patch.yaml` - No PatchConsistency
  - `meddef2_no_cbam.yaml` - No CBAM
  - `meddef2_baseline.yaml` - Baseline

### Defense Modules
- `ultralytics/nn/modules/defense/` - Centralized defense implementations
  - `freq_defense.py` - Frequency domain defense
  - `patch_consistency.py` - Patch consistency regularization
  - `trades_loss.py` - TRADES adversarial training
  - `distillation.py` - Knowledge distillation

## Documentation

- [MEDDEF_QUICK_START.md](MEDDEF_QUICK_START.md) - Quick reference guide
- [MEDDEF_ABLATION_WORKFLOW.md](MEDDEF_ABLATION_WORKFLOW.md) - Complete ablation study workflow
- [COMMANDS.md](COMMANDS.md) - Section 6 - Detailed commands reference

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_meddef.py --data ~/data --variant full --batch 8 --device 0

# Reduce image size
python train_meddef.py --data ~/data --variant full --imgsz 192 --device 0
```

### Slow Training
```bash
# Reduce workers
python train_meddef.py --data ~/data --variant full --workers 2 --device 0

# Use smaller batch
python train_meddef.py --data ~/data --variant full --batch 8 --device 0
```

### No Results in CSV
- Ensure dataset has `train/` and `val/` folders with class subfolders
- Check `data.yaml` format
- Verify image files are in supported formats (.jpg, .png, etc.)

### Model Loading Error
- Ensure you're in the correct directory: `cd /data2/enoch/ekd_coding_env/ultralytics`
- Check that variant name is valid (full, no_def, etc.)

## Performance Reference

Training on single GPU (small depth, 100 epochs):

| Model | Batch=16 | Batch=32 |
|-------|----------|----------|
| Full | ~2.5 hours | ~2 hours |
| Baseline | ~2 hours | ~1.5 hours |

Memory usage (small depth):
- Batch size 16: ~8GB
- Batch size 32: ~14GB

## Next Steps

1. ✅ **Prepare dataset** with data.yaml
2. ✅ **Train variants** using `train_meddef.py` or `run_complete_ablation_study.sh`
3. ✅ **Monitor training** with results.png and confusion_matrix.png
4. ✅ **Compare results** across ablation variants
5. ✅ **Evaluate robustness** against adversarial attacks (optional)

## References

Related files in workspace:
- `/data2/enoch/ekd_coding_env/ultralytics/` - Main ultralytics framework
- `/data2/enoch/ekd_coding_env/art/` - Adversarial robustness toolkit
- `/data2/enoch/ekd_coding_env/agkd_bml/` - Related attack/defense code

## Questions?

- Check `train_meddef.py --help` for CLI options
- See MEDDEF_QUICK_START.md for common examples
- Review MEDDEF_ABLATION_WORKFLOW.md for complete workflow
- Check results in `runs/classify/train/` after training

---

**Remember**: All variants default to `small` depth (3.4M params). You don't need to specify depth unless you want tiny, base, or large.
