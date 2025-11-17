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

## 6. Custom Backbones & Baselines

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

yolo detect train model=models/yolov8_vgg16.yaml data=dataset/cbody/data.yaml epochs=100 imgsz=640 batch=8 device=3
# Validate the VGG16 run on the validation split
yolo detect val model=runs/detect/train/cface/yolov8_vgg16/weights/best.pt data=dataset/cface/data.yaml device=0

# Evaluate the VGG16 run on the held-out test split
yolo detect val model=runs/detect/train/cface/yolov8_vgg16/weights/best.pt data=dataset/cface/data.yaml device=2 split=test

```

Both YAMLs live under `models/` and use the adapters defined in `models/custom_layers.py`; the commands above write results to `runs/<model_name>` so artifacts stay grouped by architecture.

### 6.2 TorchVision Baselines via Custom Trainer

```bash

# Train Faster R-CNN on `cface` dataset for 100 epochs on GPU 3
CUDA_VISIBLE_DEVICES=3 python ultralytics/trainers/custom_trainer.py --model faster_rcnn --data dataset/cface/data.yaml --epochs 100 --batch 8 --device cuda --project runs --name faster_rcnn_cface

# The custom trainer will compute validation loss if the YAML defines a `val` split.

# After training, validate the Faster R-CNN run on the `cface` validation split (if present)
# (the trainer prints validation loss to console during training and after each epoch)

# Evaluate/test the Faster R-CNN run on the `cface` test split (if present) using a short Python snippet
CUDA_VISIBLE_DEVICES=3 python - <<'PY'
from pathlib import Path
import torch
from ultralytics.data.utils import check_det_dataset
from ultralytics.model_loader import get_model_entry
from ultralytics.models.faster_rcnn import YoloDetectionDataset, collate_fn, resolve_split
from torch.utils.data import DataLoader
from ultralytics.trainers.custom_trainer import evaluate

cfg = "dataset/cface/data.yaml"
data_cfg = check_det_dataset(cfg)
base = Path(data_cfg["path"])
test_pair = resolve_split(base, data_cfg.get("test"))
if test_pair is None:
    print("No test split defined for", cfg)
    raise SystemExit(1)
imgs, labels = test_pair
nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
model_entry = get_model_entry("faster_rcnn")
model = model_entry.build(num_classes=nc + 1)
model.load_state_dict(torch.load("./runs/faster_rcnn_cface/weights/best.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loader = DataLoader(YoloDetectionDataset(imgs, labels), batch_size=8, shuffle=False, collate_fn=collate_fn)
print("Test loss:", evaluate(model, loader, device))
PY


# Train Faster R-CNN on `cbody` dataset with custom output directory and backbone override
CUDA_VISIBLE_DEVICES=2 python ultralytics/trainers/custom_trainer.py \
    --model faster_rcnn --data dataset/cbody/data.yaml --epochs 100 --batch 8 --device cuda \
    --project runs --name faster_rcnn_cbody --model-arg trainable_backbone_layers=5

# Evaluate/test the Faster R-CNN run on the `cbody` test split (if present)
CUDA_VISIBLE_DEVICES=2 python - <<'PY'
from pathlib import Path
import torch
from ultralytics.data.utils import check_det_dataset
from ultralytics.model_loader import get_model_entry
from ultralytics.models.faster_rcnn import YoloDetectionDataset, collate_fn, resolve_split
from torch.utils.data import DataLoader
from ultralytics.trainers.custom_trainer import evaluate

cfg = "dataset/cbody/data.yaml"
data_cfg = check_det_dataset(cfg)
base = Path(data_cfg["path"])
test_pair = resolve_split(base, data_cfg.get("test"))
if test_pair is None:
    print("No test split defined for", cfg)
    raise SystemExit(1)
imgs, labels = test_pair
nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
model_entry = get_model_entry("faster_rcnn")
model = model_entry.build(num_classes=nc + 1)
model.load_state_dict(torch.load("./runs/faster_rcnn_cbody/weights/best.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loader = DataLoader(YoloDetectionDataset(imgs, labels), batch_size=8, shuffle=False, collate_fn=collate_fn)
print("Test loss:", evaluate(model, loader, device))
PY
```

Checkpoints land under `runs/<name or model>/weights/` with `best.pt` tracking the lowest validation loss. Use `--model-arg key=value` to pass extra kwargs to any registered builder (defaults live in `ultralytics/model_loader.py`). Note: TorchVision Faster R-CNN still handles image resizing internally and ignores `imgsz`.

### 6.3 Grad-CAM Visualization 

```bash
# show available layer indices (no source images needed)
python models/gradcam.py --model runs/yolov8_resnet/weights/best.pt --list-layers

# generate Grad-CAM overlays for a few validation images (Cattle class index 0)
python models/gradcam.py --model runs/yolov8_resnet/weights/best.pt --source dataset/cbody/valid/images \
    --layer 12 --cls 0 --conf 0.0 --topk 2 --max-images 6 --device 2 --output runs/gradcam_resnet
```

The script writes per-detection overlays (JPG) plus raw heatmaps (`.npy` arrays of normalized CAM values) to the specified output directory. `--max-images` caps how many files are processed, `--cls` targets a specific class, and `--layer` selects which backbone block to probe (use the `--list-layers` flag to see indices; positive indices refer to the backbone modules shown there). If overlays look like the original image, inspect the `.npy` arrays directly (e.g., threshold at `>0.3` before plotting) or adjust the blend factor with `--alpha`. Grad-CAM quality depends on the model’s feature separation—longer training or more varied data may be needed for sharper activations.
