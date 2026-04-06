# MedDef2 Training System - Complete Index

## 📋 Documentation Files

### Quick Start (Start Here!)
- **[MEDDEF_README.md](MEDDEF_README.md)** - Overview and quick start (30 seconds)
- **[MEDDEF_QUICK_START.md](MEDDEF_QUICK_START.md)** - Detailed quick reference guide

### Training Workflows
- **[MEDDEF_ABLATION_WORKFLOW.md](MEDDEF_ABLATION_WORKFLOW.md)** - Complete ablation study workflow
- **[COMMANDS.md](COMMANDS.md) - Section 6** - Detailed commands reference

## 🚀 Running Training

### Single Variant
```bash
cd /data2/enoch/ekd_coding_env/ultralytics
python train_meddef.py --data /path/to/dataset --variant full --epochs 100 --batch 16 --device 0
```

### All 6 Variants (Automated)
```bash
bash run_complete_ablation_study.sh /path/to/dataset 100 16 0
```

### Manual Script
```bash
# Full model
python train_meddef.py --data dataset --variant full --epochs 100 --device 0

# No DefenseModule
python train_meddef.py --data dataset --variant no_def --epochs 100 --device 0

# No FrequencyDefense
python train_meddef.py --data dataset --variant no_freq --epochs 100 --device 0

# No PatchConsistency
python train_meddef.py --data dataset --variant no_patch --epochs 100 --device 0

# No CBAM
python train_meddef.py --data dataset --variant no_cbam --epochs 100 --device 0

# Baseline (pure ViT)
python train_meddef.py --data dataset --variant baseline --epochs 100 --device 0
```

## 🏗️ System Architecture

### Training Script
- **[train_meddef.py](train_meddef.py)** - Main CLI for training
  - Supports 6 variants: full, no_def, no_freq, no_patch, no_cbam, baseline
  - Supports 4 depths: tiny, small (default), base, large
  - Full argparse interface with all training parameters

### Model Variants
- **Location**: `ultralytics/models/meddef/meddef2/variants.py`
- **Classes**:
  - `MedDef2_T` - Full model (CBAM + Freq + Patch + Defense)
  - `MedDef2_T_NoDef` - No DefenseModule
  - `MedDef2_T_NoFreq` - No FrequencyDefense
  - `MedDef2_T_NoPatch` - No PatchConsistency
  - `MedDef2_T_NoCBAM` - Standard transformer blocks
  - `MedDef2_T_Baseline` - Pure ViT

### YAML Configs
- **Location**: `ultralytics/cfg/models/meddef/`
- **Files**:
  - `meddef2.yaml` - Full model config
  - `meddef2_no_def.yaml` - No DefenseModule config
  - `meddef2_no_freq.yaml` - No FrequencyDefense config
  - `meddef2_no_patch.yaml` - No PatchConsistency config
  - `meddef2_no_cbam.yaml` - No CBAM config
  - `meddef2_baseline.yaml` - Baseline config

### MedDef Module
- **Location**: `ultralytics/models/meddef/`
- **Components**:
  - `model.py` - MedDefModel class
  - `train.py` - MedDefTrainer class
  - `val.py` - MedDefValidator with confusion matrix plotting
  - `predict.py` - MedDefPredictor class

### Defense Modules
- **Location**: `ultralytics/nn/modules/defense/`
- **Components**:
  - `freq_defense.py` - Frequency domain defense
  - `patch_consistency.py` - Patch consistency regularization
  - `trades_loss.py` - TRADES adversarial training
  - `distillation.py` - Knowledge distillation

## 📊 Output Structure

Each training produces:
```
runs/classify/train/{dataset}/{experiment_name}/
├── args.yaml                          # Reproducible configuration
├── results.csv                        # Per-epoch metrics
├── results.png                        # Training curves
├── confusion_matrix.png               # Classification performance
├── confusion_matrix_normalized.png
├── train_batch*.jpg                   # Training visualization
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── weights/
    ├── best.pt                        # Best checkpoint
    └── last.pt                        # Latest checkpoint
```

## 🎯 Ablation Variants

| Variant | CBAM | Freq | Patch | Defense | Params | Use |
|---------|:----:|:----:|:-----:|:-------:|:------:|-----|
| **full** | ✓ | ✓ | ✓ | ✓ | 3.4M | Complete model |
| **no_def** | ✓ | ✓ | ✓ | ✗ | 2.9M | Remove Defense module |
| **no_freq** | ✓ | ✗ | ✓ | ✓ | 3.4M | Remove Frequency defense |
| **no_patch** | ✓ | ✓ | ✗ | ✓ | 3.4M | Remove Patch consistency |
| **no_cbam** | ✗ | ✓ | ✓ | ✓ | 3.4M | Standard transformer |
| **baseline** | ✗ | ✗ | ✗ | ✗ | 2.9M | Pure ViT (control) |

## ⚙️ Default Configuration

- **Depth**: `small` (3.4M params)
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Batch Size**: 16 (can be overridden)
- **Epochs**: 100 (can be overridden)
- **Image Size**: 224x224

## 📖 Usage Examples

### Basic Training
```bash
python train_meddef.py --data ~/data/skin_cancer --variant full --epochs 100 --batch 16 --device 0
```

### Ablation Study
```bash
# Automated script
bash run_complete_ablation_study.sh ~/data/skin_cancer 100 16 0

# Or run individually
for v in full no_def no_freq no_patch no_cbam baseline; do
    python train_meddef.py --data ~/data/skin_cancer --variant $v --epochs 100 --device 0
done
```

### Advanced Options
```bash
# Custom learning rate
python train_meddef.py --data ~/data --variant full --lr0 0.0005 --device 0

# Resume training
python train_meddef.py --data ~/data --variant full --resume --device 0

# With pretrained weights
python train_meddef.py --data ~/data --variant full --pretrained weights.pt --device 0

# Multiple GPUs
python train_meddef.py --data ~/data --variant full --device 0,1 --batch 32

# Reduce memory usage
python train_meddef.py --data ~/data --variant full --batch 8 --imgsz 192 --device 0
```

### Python API
```python
from ultralytics.models.meddef.meddef2 import get_variant
import torch

# Create model
model = get_variant('full', 'small', num_classes=10)
model.eval()

# Forward pass
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)
    print(out.shape)  # [1, 10]
```

## 🔍 Monitoring Training

### Real-time Monitoring
```bash
# Watch metrics in real-time
watch "tail -5 runs/classify/train/skin_cancer/full/results.csv"

# View training curves
display runs/classify/train/skin_cancer/full/results.png

# View confusion matrix
display runs/classify/train/skin_cancer/full/confusion_matrix.png
```

### Comparison Across Variants
```python
import pandas as pd
from pathlib import Path

variants = ['full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline']
for v in variants:
    df = pd.read_csv(f'runs/classify/train/skin_cancer/{v}/results.csv')
    final = df.iloc[-1]
    print(f"{v:12} Top-1: {final['metrics/top1_acc']:.4f}")
```

## 🛠️ Setup & Installation

All dependencies are pre-installed. Verify:
```bash
cd /data2/enoch/ekd_coding_env/ultralytics
python -c "from ultralytics.models.meddef import MedDefTrainer; print('MedDef ready!')"
```

## 📋 Dataset Requirements

Create `data.yaml`:
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 10
names: ['class1', 'class2', ...]
```

Directory structure:
```
dataset/
├── images/
│   ├── train/
│   │   ├── class1/ (contains images)
│   │   ├── class2/
│   │   └── ...
│   ├── val/
│   │   └── ... (same structure)
│   └── test/ (optional)
│       └── ... (same structure)
└── data.yaml
```

## 📞 Common Commands

```bash
# Train
python train_meddef.py --data dataset --variant full --epochs 100 --device 0

# Get help
python train_meddef.py --help

# View results
ls runs/classify/train/dataset/

# Quick test import
python -c "from ultralytics.models.meddef.meddef2 import MedDef2_T"

# Run ablation study
bash run_complete_ablation_study.sh dataset 100 16 0
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | `cd /data2/enoch/ekd_coding_env/ultralytics` |
| Out of memory | `--batch 8` or `--batch 4` |
| Slow training | `--workers 2` (default is 8) |
| No results saved | Check dataset has train/val with class folders |
| CUDA error | Verify `--device 0` or available GPU |

## 📚 Related Documentation

- **[MEDDEF_README.md](MEDDEF_README.md)** - Full readme
- **[MEDDEF_QUICK_START.md](MEDDEF_QUICK_START.md)** - Quick reference
- **[MEDDEF_ABLATION_WORKFLOW.md](MEDDEF_ABLATION_WORKFLOW.md)** - Complete workflow
- **[MEDDEF_SETUP_GUIDE.md](MEDDEF_SETUP_GUIDE.md)** - Setup details
- **[COMMANDS.md](COMMANDS.md)** - Detailed commands (Section 6)

## ✅ Checklist

- [ ] Read [MEDDEF_README.md](MEDDEF_README.md)
- [ ] Prepare dataset with `data.yaml`
- [ ] Run: `python train_meddef.py --help`
- [ ] Train single variant: `python train_meddef.py --data ... --variant full --epochs 50`
- [ ] Check results in `runs/classify/train/`
- [ ] Train all variants: `bash run_complete_ablation_study.sh ...`
- [ ] Compare results across variants
- [ ] Analyze confusion matrices and training curves

## 🎓 Next Steps

1. **Setup Dataset** - Create data.yaml and organize files
2. **Quick Test** - Train one variant for 10 epochs
3. **Full Training** - Run complete ablation study
4. **Analysis** - Compare metrics across variants
5. **Evaluation** - Test on held-out test set

---

**Remember**: Default depth is `small` (3.4M params). No need to specify depth unless you want different size!
