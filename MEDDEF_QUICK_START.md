# MedDef2 Training Quick Start Guide

## Overview
MedDef2 is a Vision Transformer-based medical image classification model with built-in adversarial defense mechanisms. This guide covers how to train and evaluate MedDef2 models with different defense configurations.

## Installation

All dependencies are already installed in your environment. You can verify with:

```bash
cd /data2/enoch/ekd_coding_env/ultralytics
python -c "from ultralytics.models.meddef import MedDefTrainer; print('MedDef installed successfully')"
```

## Dataset Preparation

Create a YAML file describing your dataset (`data.yaml`):

```yaml
path: /path/to/dataset         # dataset directory
train: train                    # relative path to training split
val: val                        # relative path to validation split
test: test                      # relative path to test split (optional)
nc: 10                          # number of classes
names: ['class1', 'class2', ...]  # class names
```

Your dataset should be organized as:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/  (optional)
    ├── class1/
    ├── class2/
    └── ...
```

## Basic Training Commands

### Train Full Model (Small Depth - Default)

The **default depth is `small`** (3.4M params). No need to specify it:

```bash
cd /data2/enoch/ekd_coding_env/ultralytics

# Train full model with all defenses
python train_meddef.py --data path/to/data --variant full --epochs 100 --batch 16 --device 0 --name exp1

# Train with 50 epochs on GPU 1
python train_meddef.py --data path/to/data --variant full --epochs 50 --device 1

# Train with custom learning rate
python train_meddef.py --data path/to/data --variant full --epochs 100 --lr0 0.0005
```

### Training All Ablation Variants

```bash
# Full model (CBAM + Freq + Patch + Defense)
python train_meddef.py --data dataset/skin_cancer --variant full --epochs 100 --batch 16 --device 0 --name full

# Without DefenseModule
python train_meddef.py --data dataset/skin_cancer --variant no_def --epochs 100 --batch 16 --device 0 --name no_def

# Without FrequencyDefense
python train_meddef.py --data dataset/skin_cancer --variant no_freq --epochs 100 --batch 16 --device 0 --name no_freq

# Without PatchConsistency
python train_meddef.py --data dataset/skin_cancer --variant no_patch --epochs 100 --batch 16 --device 0 --name no_patch

# Without CBAM (standard transformer blocks)
python train_meddef.py --data dataset/skin_cancer --variant no_cbam --epochs 100 --batch 16 --device 0 --name no_cbam

# Baseline (no defenses - pure ViT)
python train_meddef.py --data dataset/skin_cancer --variant baseline --epochs 100 --batch 16 --device 0 --name baseline
```

## Available Variants

All variants use **small depth** by default:

| Variant | Description | Params | Key Features |
|---------|-------------|--------|--------------|
| `full` | Complete MedDef2 | 3.4M | CBAM + Freq + Patch + Defense |
| `no_def` | Without defense module | 2.9M | CBAM + Freq + Patch |
| `no_freq` | Without frequency defense | 3.4M | CBAM + Patch + Defense |
| `no_patch` | Without patch consistency | 3.4M | CBAM + Freq + Defense |
| `no_cbam` | Standard transformer blocks | 3.4M | Freq + Patch + Defense |
| `baseline` | Pure ViT (control) | 2.9M | No defenses |

## Understanding Output Structure

Training results are saved to:
```
runs/classify/train/{dataset}/{experiment_name}/
```

Example for `python train_meddef.py --data dataset/skin_cancer --variant full --name exp1`:
```
runs/classify/train/skin_cancer/exp1/
├── args.yaml                          # All training arguments (reproducibility)
├── results.csv                        # Per-epoch metrics (loss, accuracy, etc.)
├── results.png                        # Training curves plot
├── confusion_matrix.png               # Classification confusion matrix
├── confusion_matrix_normalized.png    # Normalized confusion matrix
├── train_batch*.jpg                   # Sample training images
├── val_batch*_labels.jpg              # Validation ground truth
├── val_batch*_pred.jpg                # Validation predictions
└── weights/
    ├── best.pt                        # Best model checkpoint
    └── last.pt                        # Latest checkpoint
```

## Validation and Testing

After training, validate the model:

```bash
# Validate on validation set
python train_meddef.py --data dataset/skin_cancer \
    --model runs/classify/train/skin_cancer/exp1/weights/best.pt \
    --variant full \
    --device 0 \
    split=val

# Test on test set
python train_meddef.py --data dataset/skin_cancer \
    --model runs/classify/train/skin_cancer/exp1/weights/best.pt \
    --variant full \
    --device 0 \
    split=test
```

## Python API Usage

You can also use MedDef programmatically:

```python
from ultralytics.models.meddef import MedDefTrainer
from ultralytics.models.meddef.meddef2 import get_variant

# Option 1: Direct model creation
model = get_variant('full', 'small', num_classes=10)

# Forward pass
x = torch.randn(1, 3, 224, 224)
out = model(x)  # output shape: [1, 10]

# Option 2: Training with MedDefTrainer
trainer = MedDefTrainer(overrides={
    'model': 'meddef2.yaml',
    'data': 'dataset/skin_cancer/data.yaml',
    'epochs': 100,
    'batch': 16,
    'device': 0,
    'name': 'exp1'
})

# Train
trainer.train()

# Validate
results = trainer.val()

# Test on test split
results = trainer.val(split='test')

# Access results
print(results['top1_acc'])  # Top-1 accuracy
print(results['top5_acc'])  # Top-5 accuracy
```

## Command Line Help

View all available arguments:

```bash
python train_meddef.py --help
```

## Performance Reference (Small Depth)

Typical training times on a single GPU:
- **Full model**: ~2-3 hours per 100 epochs
- **Baseline**: ~1.5-2 hours per 100 epochs
- **Dataset**: Skin cancer (10 classes, ~10k images)

Memory requirements:
- **GPU**: ~8GB for batch_size=16 at 224x224
- **GPU**: ~16GB for batch_size=32

## Tips & Best Practices

1. **Start with small depth**: The default `small` depth provides a good balance between accuracy and training time.

2. **Batch size**: Increase batch size if you have GPU memory:
   ```bash
   python train_meddef.py --data dataset/skin_cancer --variant full --batch 32 --device 0
   ```

3. **Early stopping**: Training will auto-save the best checkpoint. Check `results.png` to see if the model is still improving.

4. **Reproducibility**: All training arguments are saved in `args.yaml`. Reproduce runs with:
   ```bash
   python train_meddef.py --data dataset/skin_cancer --variant full --epochs 50 --device 0 --name exp_reproduce
   ```

5. **Multiple GPUs**: Use comma-separated device IDs:
   ```bash
   python train_meddef.py --data dataset/skin_cancer --variant full --device 0,1
   ```

6. **Ablation study workflow**:
   ```bash
   # Train all variants sequentially
   for variant in full no_def no_freq no_patch no_cbam baseline; do
       python train_meddef.py --data dataset/skin_cancer --variant $variant --epochs 100 --device 0 --name $variant
   done
   ```

## Comparison with YOLO

MedDef2 follows the same workflow as YOLOv8 classification but uses custom transformer architectures:

- **YOLO**: `yolo classify train model=yolov8n-cls.pt data=...`
- **MedDef**: `python train_meddef.py --data ... --variant full`

Output structure is identical to YOLO's `runs/classify/train/` format.

## Next Steps

1. **Train full model** on your dataset
2. **Compare variants** using the ablation commands above
3. **Analyze results** in `results.png` and `confusion_matrix*.png`
4. **Evaluate on test set** to get final metrics
5. **Check `args.yaml`** to verify reproducibility

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'ultralytics.models.meddef'"
- **Solution**: Ensure you're in the correct directory: `cd /data2/enoch/ekd_coding_env/ultralytics`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size: `--batch 8` or `--batch 4`

**Issue**: Validation metrics not appearing
- **Solution**: Check that your dataset has a `val/` split with class folders

**Issue**: Model training very slowly
- **Solution**: Reduce workers: `--workers 2` (default is 8)

## Questions?

Refer to:
- Training script help: `python train_meddef.py --help`
- Detailed commands: [COMMANDS.md](COMMANDS.md) Section 6
- Variant architecture: `ultralytics/models/meddef/meddef2/variants.py`
