# Ultralytics YOLO Command Guide

Concise reference for running Ultralytics YOLO tasks from this repository checkout. Commands assume you are executing inside the repo root (`ultralytics/`).

## 1. Environment Setup

```bash
# create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# install project in editable mode with extras
pip install --upgrade pip
pip install -e .[dev]
```

> ⚠️ Requires Python >= 3.8 and PyTorch >= 1.8. For GPU acceleration, install the PyTorch build matching your CUDA version before installing Ultralytics.

## 2. Core CLI Workflows

### 2.1 Prediction / Inference

```bash
# run detection on image/dir/URL/video/stream
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'

# run segmentation model on a folder of images
yolo predict task=segment model=yolo11s-seg.pt source=assets/samples/

# run pose estimation on webcam (device index 0)
yolo predict task=pose model=yolo11n-pose.pt source=0
```

### 2.2 Training

```bash
# quick start on COCO8 sample dataset
yolo train model=yolo11s.pt data=coco8.yaml epochs=50 imgsz=640

# train on local FaceMask dataset cloned under dataset/facemask
yolo train model=yolo11s.pt data=dataset/facemask/data.yaml epochs=50 imgsz=640

# resume training from the last checkpoint
yolo train resume=True

# fine-tune segmentation model with mixed precision on GPU 0
yolo train task=segment model=yolo11m-seg.pt data=path/to/data.yaml epochs=100 device=0 amp=True
```

> ℹ️ By default, training/validation/predict runs now save under `runs/<task>/<mode>/<dataset>/<model>` (e.g., `runs/detect/train/cface/yolov8_resnet`). Supplying a custom `project=` or `name=` continues to override this layout.

### 2.3 Validation & Evaluation

```bash
# evaluate latest training run
yolo val model=runs/detect/train/weights/best.pt data=coco8.yaml

# validate FaceMask checkpoint on its validation split
yolo val model=runs/detect/train/weights/best.pt data=dataset/facemask/data.yaml split=val

# validate exported engine (e.g., TensorRT) at custom image size
yolo val model=weights/yolo11s.engine data=coco.yaml imgsz=1280 conf=0.25
```

### 2.4 Tracking

```bash
# byte track on video with detection weights
yolo track model=yolo11s.pt source=path/to/video.mp4 tracker=bytetrack.yaml

# deep sort tracking with custom YAML and half precision
yolo track model=yolo11m.pt source=rtsp://user:pass@ip stream=True tracker=deepsort.yaml half=True
```

### 2.5 Export

```bash
# export to ONNX for CPU inference
yolo export model=yolo11s.pt format=onnx

# export pose model to TensorRT FP16 on GPU 0
yolo export task=pose model=yolo11m-pose.pt format=engine device=0 half=True

# generate CoreML package for iOS deployment
yolo export model=yolo11n.pt format=coreml imgsz=640 optimize=True
```

## 3. Python API Patterns

```python
from ultralytics import YOLO

# load pretrained weights
model = YOLO("yolo11n.pt")

# train
model.train(data="coco8.yaml", epochs=50, imgsz=640, workers=8)

# validate
metrics = model.val()
print(metrics.box.map50)

# infer on list of sources
results = model(["bus.jpg", "truck.jpg"], conf=0.4)
for r in results:
    r.save(filename=f"pred_{r.path.name}")

# export
model.export(format="onnx", dynamic=True)
```

## 4. Useful Utilities

```bash
# list saved runs
tree runs -L 2

# FaceMask dataset lives under dataset/facemask with train/valid/test splits
tree dataset/facemask -L 2

# configure custom dataset template
cp ultralytics/cfg/datasets/coco8.yaml data/custom.yaml
$EDITOR data/custom.yaml

# clean caches if you want fresh downloads
rm -rf ~/.cache/ultralytics

# run unit tests (needs dev extras)
pytest -q
```

## 5. Helpful Flags

- `device=`: choose hardware (`cpu`, `0`, `0,1`).
- `imgsz=`: override default image size at train/predict/val/export time.
- `batch=`: set batch size; use `auto` for automatic tuning during training.
- `conf=` / `iou=`: adjust thresholds for prediction filtering.
- `save=True|False`: control artifact saving (images, labels, metrics).
- `project=` / `name=`: customize run output directory (defaults to `runs/<task>/<name>`).

For more options, run `yolo help` or consult the official docs at https://docs.ultralytics.com.

## 6. MedDef Classification Models (Ablation Studies)

MedDef (Medical Defense) models are Vision Transformer-based classification architectures with defense mechanisms against adversarial attacks. They support comprehensive ablation studies for understanding defense component contributions.

### 6.1 Quick Start - Training with Small Depth (Default)

```bash
# Train full MedDef2 model (small depth) on your dataset
python train_meddef.py --data dataset/your_dataset --variant full --epochs 100 --batch 16 --device 0

# Validate the trained model
python train_meddef.py --data dataset/your_dataset --model runs/classify/train/dataset/full_small/weights/best.pt --variant full --device 0 split=val

# Test on test split
python train_meddef.py --data dataset/your_dataset --model runs/classify/train/dataset/full_small/weights/best.pt --variant full --device 0 split=test
```

### 6.2 Ablation Study Commands (All Use Small Depth)

```bash
# Full Model: CBAM + Frequency + Patch + DefenseModule
python train_meddef.py --data dataset/skin_cancer --variant full --epochs 100 --batch 16 --device 0 --name full_small
# Validate
python train_meddef.py --data dataset/skin_cancer --model runs/classify/train/skin_cancer/full_small/weights/best.pt --variant full --device 0 split=val
# Test
python train_meddef.py --data dataset/skin_cancer --model runs/classify/train/skin_cancer/full_small/weights/best.pt --variant full --device 0 split=test

# No DefenseModule: CBAM + Frequency + Patch only
python train_meddef.py --data dataset/skin_cancer --variant no_def --epochs 100 --batch 16 --device 0 --name no_def_small
# Validate
python train_meddef.py --data dataset/skin_cancer --model runs/classify/train/skin_cancer/no_def_small/weights/best.pt --variant no_def --device 0 split=val

# No Frequency Defense: CBAM + Patch + DefenseModule only
python train_meddef.py --data dataset/skin_cancer --variant no_freq --epochs 100 --batch 16 --device 0 --name no_freq_small

# No Patch Consistency: CBAM + Frequency + DefenseModule only
python train_meddef.py --data dataset/skin_cancer --variant no_patch --epochs 100 --batch 16 --device 0 --name no_patch_small

# No CBAM: Frequency + Patch + DefenseModule (standard transformer blocks)
python train_meddef.py --data dataset/skin_cancer --variant no_cbam --epochs 100 --batch 16 --device 0 --name no_cbam_small

# Baseline: No defenses (standard ViT)
python train_meddef.py --data dataset/skin_cancer --variant baseline --epochs 100 --batch 16 --device 0 --name baseline_small
# Validate
python train_meddef.py --data dataset/skin_cancer --model runs/classify/train/skin_cancer/baseline_small/weights/best.pt --variant baseline --device 0 split=val
```

### 6.3 Available Variants

| Variant | CBAM | Freq | Patch | Defense | Use Case |
|---------|:----:|:----:|:-----:|:-------:|----------|
| `full` | ✓ | ✓ | ✓ | ✓ | Full defense model |
| `no_def` | ✓ | ✓ | ✓ | ✗ | Without DefenseModule |
| `no_freq` | ✓ | ✗ | ✓ | ✓ | Without FrequencyDefense |
| `no_patch` | ✓ | ✓ | ✗ | ✓ | Without PatchConsistency |
| `no_cbam` | ✗ | ✓ | ✓ | ✓ | Standard transformer blocks |
| `baseline` | ✗ | ✗ | ✗ | ✗ | Standard ViT (control) |

### 6.4 Output Structure

Results are saved to `runs/classify/train/{dataset}/{experiment_name}/`:

```
runs/classify/train/skin_cancer/full_small/
├── args.yaml                          # All training arguments
├── results.csv                        # Per-epoch metrics
├── results.png                        # Training curves
├── confusion_matrix.png               # Classification performance (Ultralytics)
├── confusion_matrix_normalized.png
├── evaluation_metrics.csv             # Comprehensive metrics (MCC, κ, ECE, etc.)
├── detailed_metrics.txt               # Human-readable metrics report
├── metrics.json                       # Machine-readable full metrics dump
├── train_batch*.jpg                   # Training samples
├── val_batch*_labels.jpg              # Validation ground truth
├── val_batch*_pred.jpg                # Validation predictions
├── visualizations/                    # Enhanced MedDef2 visualizations
│   ├── training_curves.png            # Loss + accuracy with best epoch annotation
│   ├── confusion_matrix.png           # Enhanced heatmap (raw + normalized)
│   ├── per_class_metrics.png          # Precision / Recall / F1 / Specificity per class
│   ├── roc_auc.png                    # ROC curves (per-class + macro-avg)
│   ├── metrics_summary.png            # 2×2 dashboard panel
│   ├── tsne_embeddings.png            # t-SNE scatter plot (post-eval)
│   ├── pca_embeddings.png             # PCA scatter plot (post-eval)
│   └── saliency/                      # Grad-CAM overlays (post-eval)
│       └── saliency_0_classname.png
└── weights/
    ├── best.pt                        # Best checkpoint
    └── last.pt                        # Latest checkpoint
```

### 6.5 Python API Usage

```python
from ultralytics.models.meddef import MedDefTrainer
from ultralytics.models.meddef.meddef2 import get_variant

# Create model directly
model = get_variant('no_cbam', 'small', num_classes=10)

# Or use trainer with YAML config
trainer = MedDefTrainer(overrides={
    'model': 'meddef2_no_cbam.yaml',
    'data': 'dataset/skin_cancer/data.yaml',
    'epochs': 100,
    'batch': 16,
    'device': 0,
    'name': 'no_cbam_small'
})
trainer.train()

# Validate
trainer.val()

# Test on test split
trainer.val(data='dataset/skin_cancer/data.yaml', split='test')
```

## 7. Custom Backbones & Baselines

### 6.1 YOLOv8 + TorchVision Hybrids

```bash
# ResNet-50 backbone feeding the YOLOv8 head (short sanity run on GPU 3)
# Training and evaluation for both cbody and cface datasets

# Train on `cbody`, validate and test on `cbody`
yolo detect train model=models/yolov8_resnet.yaml data=dataset/cbody/data.yaml epochs=100 imgsz=640 batch=8 device=3 project=runs name=yolov8_resnet_cbody
# Validate the ResNet run on the validation split (cbody)
yolo detect val model=runs/detect/train/cbody/yolov8_resnet/weights/best.pt data=dataset/cbody/data.yaml device=3
# Evaluate the ResNet run on the held-out test split (cbody)
yolo detect val model=runs/detect/train/cbody/yolov8_resnet/weights/best.pt data=dataset/cbody/data.yaml device=3 split=test

# Train on `cface`, validate and test on `cface`
yolo detect train model=models/yolov8_resnet.yaml data=dataset/cface/data.yaml epochs=100 imgsz=640 batch=8 device=3 project=runs name=yolov8_resnet_cface
# Validate the ResNet run on the validation split (cface)
yolo detect val model=runs/detect/train/cface/yolov8_resnet/weights/best.pt data=dataset/cface/data.yaml device=3
# Evaluate the ResNet run on the held-out test split (cface)
yolo detect val model=runs/detect/train/cface/yolov8_resnet/weights/best.pt data=dataset/cface/data.yaml device=3 split=test


# VGG16 backbone variant (longer run)
# Train on `cface`, validate and test on `cface`
yolo detect train model=models/yolov8_vgg16.yaml data=dataset/cface/data.yaml epochs=100 imgsz=640 batch=8 device=2 project=runs name=yolov8_vgg16_cface
# Validate VGG16 on `cface`
yolo detect val model=runs/detect/train/cface/yolov8_vgg16/weights/best.pt data=dataset/cface/data.yaml device=2
# Evaluate VGG16 on `cface` test split
yolo detect val model=runs/detect/train/cface/yolov8_vgg16/weights/best.pt data=dataset/cface/data.yaml device=2 split=test

# Train on `cbody`, validate and test on `cbody`
yolo detect train model=models/yolov8_vgg16.yaml data=dataset/cbody/data.yaml epochs=100 imgsz=640 batch=8 device=3 project=runs name=yolov8_vgg16_cbody
# Validate VGG16 on `cbody`
yolo detect val model=runs/detect/train/cbody/yolov8_vgg16/weights/best.pt data=dataset/cbody/data.yaml device=3
# Evaluate VGG16 on `cbody` test split
yolo detect val model=runs/detect/train/cbody/yolov8_vgg16/weights/best.pt data=dataset/cbody/data.yaml device=3 split=test

```

Both YAMLs live under `models/` and use the adapters defined in `models/custom_layers.py`; the commands above write results to `runs/<model_name>` so artifacts stay grouped by architecture.

### 6.2 TorchVision Baselines via Custom Trainer

```bash

# Train Faster R-CNN on `cface` (GPU 3)
CUDA_VISIBLE_DEVICES=3 python ultralytics/trainers/custom_trainer.py --model faster_rcnn --data dataset/cface/data.yaml --epochs 100 --batch 8 --device cuda

CUDA_VISIBLE_DEVICES=3 python ultralytics/trainers/custom_trainer.py --model faster_rcnn --data dataset/cface/data.yaml --epochs 100 --batch 8 --device cuda --project runs --name faster_rcnn_cface
# Need to avoid downloading TorchVision pretrained weights (e.g., offline cluster)? Append: --model-arg weights=None
# Evaluate validation split
python models/faster_rcnn_eval.py --data dataset/cface/data.yaml --weights runs/faster_rcnn_cface/weights/best.pt --split val --batch 8 --device cuda
# Evaluate test split
python models/faster_rcnn_eval.py --data dataset/cface/data.yaml --weights runs/faster_rcnn_cface/weights/best.pt --split test --batch 8 --device cuda

# Train Faster R-CNN on `cbody` with deeper backbone layers (GPU 2)
CUDA_VISIBLE_DEVICES=2 python ultralytics/trainers/custom_trainer.py \
    --model faster_rcnn --data dataset/cbody/data.yaml --epochs 100 --batch 8 --device cuda \
    --project runs --name faster_rcnn_cbody --model-arg trainable_backbone_layers=5
# To skip pretrained weights for this run:
#     --model-arg weights=None
# Evaluate validation split
python models/faster_rcnn_eval.py --data dataset/cbody/data.yaml --weights runs/faster_rcnn_cbody/weights/best.pt --split val --batch 8 --device cuda
# Evaluate test split
python models/faster_rcnn_eval.py --data dataset/cbody/data.yaml --weights runs/faster_rcnn_cbody/weights/best.pt --split test --batch 8 --device cuda
```

`models/faster_rcnn_eval.py` mirrors the custom trainer’s loss computation, so you can point it at any Ultralytics-format dataset YAML and checkpoint to obtain split-specific losses without embedding inline Python in your terminal history.

Checkpoints land under `runs/<name or model>/weights/` with `best.pt` tracking the lowest validation loss. Use `--model-arg key=value` to pass extra kwargs to any registered builder (defaults live in `ultralytics/model_loader.py`). Note: TorchVision Faster R-CNN still handles image resizing internally and ignores `imgsz`.

### 6.3 Grad-CAM Visualization 

```bash
# show available layer indices (no source images needed)
python models/gradcam.py --model runs/detect/train/cbody/yolov8_resnet/weights/best.pt --list-layers

# generate Grad-CAM overlays for a few validation images (Cattle class index 0)
python models/gradcam.py --model runs/detect/train/cbody/yolov8_resnet/weights/best.pt --source dataset/cbody/valid/images --layer 12 --cls 0 --conf 0.0 --topk 2 --max-images 6 --device 2

# alternate dataset/model example (Face)
python models/gradcam.py --model runs/detect/train/cface/yolov8_resnet/weights/best.pt --source dataset/cface/valid/images --layer 12 --cls 0 --conf 0.0 --topk 2 --max-images 6 --device 3

# keep every face, suppress duplicates with IoU 0.4
python models/gradcam.py --model runs/detect/train/cface/yolov8_resnet/weights/best.pt --source dataset/cface/valid/images --layer 12 --cls 0 --conf 0.05 --topk -1 --iou-filter 0.4 --max-images 4 --device 2

python models/gradcam.py --model runs/detect/train/cface/yolov8_resnet/weights/best.pt --source dataset/cface/valid/images --layer 12 --cls 0 --conf 0.05  --topk -1 --iou-filter 0.4 --max-images 4 --device 2
```

The script now saves a single combined overlay (JPG) plus aggregate heatmap (`.npy`) per image under `runs/<task>/<mode>/<dataset>/<model>/` (defaults to `runs/detect/gradcam/<dataset>/<model>/`). Add `--save-individual` if you still want per-detection assets (`*_gradcam_<idx>.jpg`). Override the hierarchy with `--project/--task/--mode/--dataset-name/--run-name` or pass a fully-qualified `--output`. `--topk <= 0` keeps every unique detection that clears `--conf`, while `--iou-filter` controls how aggressively near-duplicate anchors are merged. `--max-images` caps how many files are processed, `--cls` targets a specific class, and `--layer` selects which backbone block to probe (use the `--list-layers` flag to see indices). If overlays look like the original image, inspect the `.npy` arrays directly (e.g., threshold at `>0.3` before plotting) or adjust the blend factor with `--alpha`. Grad-CAM quality depends on the model’s feature separation—longer training or more varied data may be needed for sharper activations.

```bash
# status
 ./run/smart_multi_gpu_trainer.sh --status

# stop (kills trainer scheduler + all running training jobs)
./run/smart_multi_gpu_trainer.sh --stop
```

## 8. MedDef2 Multi-GPU Trainer — Monitoring & Control

All commands run from the repo root (`ultralytics/`). The trainer process is backgrounded via `nohup`; these flags attach to it without interrupting training.

### 8.1 Status Snapshot

```bash
# One-shot summary: GPU utilisation, queue depth, completed/failed counts
# Each GPU row includes "User: <name>" showing who owns the running process
./run/smart_multi_gpu_trainer.sh --status
```

### 8.2 Live Dashboard (Recommended)

A refreshing full-screen dashboard showing per-GPU progress, epoch bars, queue, and failures. Each GPU row shows the owning user (e.g. `no_cbam:scisic ep5/118 | enoch`) — external jobs show the actual OS user via `nvidia-smi pmon`. Safe to open from any new terminal — reads persisted state files only.

```bash
# Auto-refresh every 5 s (default)
./run/smart_multi_gpu_trainer.sh --watch

# Alias — identical behaviour
./run/smart_multi_gpu_trainer.sh --dashboard

# Slower refresh (e.g. 10 s) to reduce terminal flicker
./run/smart_multi_gpu_trainer.sh --watch --watch-interval 10

# Faster refresh (2 s)
./run/smart_multi_gpu_trainer.sh --watch --watch-interval 2
```

Press **Ctrl+C** to exit the dashboard. Training continues uninterrupted.

### 8.3 Live Log Tail

```bash
# Tail raw tqdm/loss output from all active training logs simultaneously
./run/smart_multi_gpu_trainer.sh --live

# Follow the log for whichever job is currently on GPU 2
./run/smart_multi_gpu_trainer.sh --log gpu=2

# Follow GPU 3
./run/smart_multi_gpu_trainer.sh --log gpu=3
```

> Tip: install `multitail` for a split-pane side-by-side view — `--live` uses it automatically when available (`sudo apt install multitail`).

### 8.4 Start / Resume / Stop

```bash
# Start fresh (clears queue and state, builds 12-job queue automatically)
./run/smart_multi_gpu_trainer.sh

# Resume after a crash or reboot (replays existing queue.txt)
./run/smart_multi_gpu_trainer.sh --resume

# Dry-run — full scheduling loop without launching any training
./run/smart_multi_gpu_trainer.sh --dry-run

# Stop everything — kills trainer scheduler + all GPU training jobs
# Training can be resumed later with --resume
./run/smart_multi_gpu_trainer.sh --stop
```

### 8.5 Manual State Control

```bash
# Wipe all state and logs (interactive confirmation)
./run/smart_multi_gpu_trainer.sh --clean

# Install auto-resume cron (@reboot — useful on shared servers)
./run/smart_multi_gpu_trainer.sh --install-cron
# Remove: crontab -e

# View master scheduler log live
tail -f logs/multi_gpu_training/master.log

# Check which jobs are on each GPU right now
for g in 0 1 2 3; do
  echo -n "GPU $g: "
  cat logs/multi_gpu_training/jobs/gpu${g}.model 2>/dev/null || echo "(idle)"
done
```

### 8.6 Custom Training Runs

```bash
# Only specific GPUs
./run/smart_multi_gpu_trainer.sh --gpu 2,3

# Only a subset of datasets / variants
./run/smart_multi_gpu_trainer.sh --datasets "tbcr" --variants "full baseline"

# Override training hyper-parameters
./run/smart_multi_gpu_trainer.sh --epochs 50 --batch 32 --depth small

# Environment-variable style (equivalent)
VARIANTS="full no_cbam" DATASETS="scisic" ./run/smart_multi_gpu_trainer.sh
```

### 8.7 Log Paths

| Path | Contents |
|------|----------|
| `logs/multi_gpu_training/master.log` | Scheduler decisions, launch/complete/fail events |
| `logs/multi_gpu_training/model_logs/<dataset>/<variant>.log` | Per-model tqdm + loss output |
| `logs/multi_gpu_training/state/completed.txt` | One `variant:dataset` key per completed job |
| `logs/multi_gpu_training/state/failed.txt` | One `variant:dataset` key per failed job |
| `logs/multi_gpu_training/state/queue.txt` | Remaining jobs (shrinks as jobs launch) |
| `logs/multi_gpu_training/jobs/gpu<N>.{pid,model,status}` | Per-GPU runtime state |
| `runs/classify/train/<dataset>/<variant>_<dataset>/weights/best.pt` | Best checkpoint |

## 9. MedDef2 Evaluation & Testing

Post-training evaluation with comprehensive metrics (MCC, Cohen's κ, ECE, ROC-AUC, per-class breakdown). Results saved as CSV + TXT + JSON.

### 9.1 Basic Evaluation

```bash
# Evaluate a trained model on its validation split
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --data /data2/enoch/tbcr --variant full

# Evaluate on test split
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --data /data2/enoch/tbcr --split test

# Evaluate with custom batch size / device
python evaluate_meddef.py --model runs/classify/train/scisic/full/weights/best.pt \
                          --data /data2/enoch/scisic --batch 64 --device 2
```

### 9.2 Evaluation with Visualizations

```bash
# Full evaluation + all visualizations (confusion matrix, ROC, per-class, t-SNE, PCA, Grad-CAM)
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --data /data2/enoch/tbcr --visualize --saliency

# Only t-SNE + PCA embedding plots
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --data /data2/enoch/tbcr --tsne --pca

# Only Grad-CAM saliency maps (16 samples)
python evaluate_meddef.py --model runs/classify/train/scisic/full/weights/best.pt \
                          --data /data2/enoch/scisic --saliency --n-saliency 16
```

### 9.3 Generated Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| Accuracy | `accuracy` | Overall correct predictions / total |
| Top-1 / Top-5 | `top1_acc`, `top5_acc` | Standard top-k accuracy |
| Precision | `precision_macro`, `precision_weighted` | Macro & weighted avg |
| Recall | `recall_macro`, `recall_weighted` | Macro & weighted avg |
| F1 Score | `f1_macro`, `f1_weighted` | Harmonic mean of precision & recall |
| Specificity | `specificity_macro` | True negative rate |
| Balanced Accuracy | `balanced_accuracy` | Average per-class recall |
| MCC | `mcc` | Matthews Correlation Coefficient |
| Cohen's Kappa | `cohen_kappa` | Agreement corrected for chance |
| ROC AUC | `roc_auc` | Area under ROC (one-vs-rest, macro) |
| Log Loss | `log_loss` | Cross-entropy loss on probabilities |
| Brier Score | `brier_score` | Mean squared error of probabilities |
| ECE | `ece` | Expected Calibration Error (10 bins) |


## 10. MedDef2 Visualization & Analysis

### 10.1 Generate Plots from Existing Results

```bash
# Re-generate visualizations without re-running inference
python evaluate_meddef.py --viz-only --results-dir runs/classify/train/tbcr/full
```

### 10.2 Ablation Comparison (Cross-Variant)

```bash
# Compare all 6 ablation variants for a dataset
python evaluate_meddef.py --compare --results-dir runs/classify/train/tbcr

# Compare specific variants only
python evaluate_meddef.py --compare --results-dir runs/classify/train/scisic \
                          --variants-to-compare full no_def no_freq baseline
```

This generates:
- `comparison/ablation_radar.png` — Polar radar chart overlaying all variants
- `comparison/ablation_heatmap.png` — Heatmap of (variant × metric)
- `comparison/ablation_comparison.csv` — Tabular comparison data

### 10.3 Available Plot Types

| Plot | Description | When Generated |
|------|-------------|----------------|
| Training Curves | Loss + accuracy with best-epoch annotation | After training / `--viz-only` |
| Confusion Matrix | Raw counts + normalized heatmap (side-by-side) | After evaluation |
| Per-Class Metrics | Grouped bars: Precision, Recall, F1, Specificity | After evaluation |
| ROC / AUC | Per-class + macro-average ROC curves | After evaluation |
| Metrics Summary | 2×2 dashboard: aggregate, per-class F1, calibration, highlights | After evaluation |
| Radar Chart | Polar comparison across variants | `--compare` mode |
| Ablation Heatmap | Variant × metric table | `--compare` mode |
| t-SNE Embeddings | 2D t-SNE scatter with class centroids | `--tsne` or `--visualize` |
| PCA Embeddings | 2D PCA with explained variance | `--pca` or `--visualize` |
| Grad-CAM Saliency | Original / heatmap / overlay triplet | `--saliency` |
| Epsilon Sensitivity | Metric vs ε line plot (for robustness testing) | Python API |
| ASR Heatmap | Defense × attack success rate matrix | Python API |

### 10.4 Python API for Custom Visualizations

```python
from ultralytics.utils.meddef_visualize import (
    MedDefVisualizer,
    plot_radar_chart,
    plot_epsilon_sensitivity,
    plot_asr_heatmap,
    plot_ablation_heatmap,
)

# Per-experiment visualizations
viz = MedDefVisualizer("runs/classify/train/tbcr/full")
viz.training_curves()
viz.metrics_summary(metrics_dict)

# Cross-experiment radar comparison
radar_data = {
    "full": {"accuracy": 0.95, "f1_macro": 0.94, "mcc": 0.93, ...},
    "no_def": {"accuracy": 0.91, "f1_macro": 0.89, "mcc": 0.87, ...},
}
plot_radar_chart(radar_data, "comparison/radar.png")

# Epsilon sensitivity (from robustness testing)
eps_results = {
    "full": {0.01: 0.94, 0.05: 0.88, 0.1: 0.75, 0.2: 0.52},
    "baseline": {0.01: 0.90, 0.05: 0.72, 0.1: 0.45, 0.2: 0.20},
}
plot_epsilon_sensitivity(eps_results, "robustness/epsilon.png")

# Attack success rate heatmap
asr_data = {
    "full": {"FGSM": 0.12, "PGD": 0.18, "BIM": 0.15, "JSMA": 0.22},
    "baseline": {"FGSM": 0.65, "PGD": 0.78, "BIM": 0.72, "JSMA": 0.80},
}
plot_asr_heatmap(asr_data, "robustness/asr.png")
```


## 11. MedDef2 Model Export & Deployment

Convert trained models to formats for mobile/desktop/edge deployment.

### 11.1 Export Commands

```bash
# Export to ONNX (CPU inference, web deployment)
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --export onnx

# Export to TorchScript (C++ deployment)
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --export torchscript

# Export to multiple formats at once
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --export onnx torchscript openvino

# Export with custom image size
python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \
                          --export onnx --imgsz 384

# Using Ultralytics CLI directly
yolo export model=runs/classify/train/tbcr/full/weights/best.pt format=onnx imgsz=224
```

### 11.2 Supported Export Formats

| Format | Flag | Use Case |
|--------|------|----------|
| ONNX | `onnx` | Cross-platform inference, web deployment |
| TorchScript | `torchscript` | C++ / mobile (PyTorch Mobile) |
| TFLite | `tflite` | Android / edge devices |
| CoreML | `coreml` | iOS / macOS deployment |
| OpenVINO | `openvino` | Intel CPU/GPU/VPU acceleration |
| TensorRT | `engine` | NVIDIA GPU optimized inference |


```bash

# Check progress

ssh -p 8822 enoch@ci2p 'cd /data2/enoch/ekd_coding_env/ultralytics && echo "=== Progress ===" && for v in full no_def no_freq no_patch no_cbam baseline; do f="runs/classify/train_tbcr_final/tbcr/${v}_small/distill_v2/results.csv"; if [ -f "$f" ]; then lines=$(wc -l < "$f"); last=$(tail -1 "$f" | cut -d, -f1); acc=$(tail -1 "$f" | cut -d, -f5); echo "$v: epoch=$last acc=$acc ($lines rows)"; else echo "$v: not started yet"; fi; done && echo "=== Scheduler ===" && kill -0 $(cat logs/tbcr_final_distill_v2/scheduler.pid 2>/dev/null) 2>/dev/null && echo "ALIVE" || echo "DEAD"'

```
