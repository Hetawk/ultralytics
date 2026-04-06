# Custom Model Playbook (Faster R-CNN + YOLOv8 hybrids)

This guide shows how to stand up three detector options inside the official `ultralytics/` checkout while keeping the workflow modular:

- **faster_rcnn** – TorchVision Faster R-CNN wrapper for baseline comparisons
- **yolov8_resnet** – YOLOv8 head fed by a TorchVision ResNet backbone
- **yolov8_vgg16** – YOLOv8 head driven by a TorchVision VGG16 backbone with optional custom blocks

The goal is to reuse Ultralytics' training/validation/export tooling (`yolo` CLI, Python `YOLO` API) while slotting in custom backbones or full models.

---

## 1. Directory Layout

Inside the repository root (`ultralytics/`), add one lightweight namespace to hold custom model glue code and YAML definitions:

```
ultralytics/
├── models/
│   ├── __init__.py          # optional registry helper
│   ├── faster_rcnn.py       # TorchVision wrapper & entry script
│   ├── yolov8_resnet.yaml   # YOLOv8 + ResNet backbone definition
│   ├── yolov8_vgg16.yaml    # YOLOv8 + VGG16 backbone definition
│   └── README.md            # short recap / command examples (optional)
├── COMMANDS.md              # general command cheatsheet (already present)
├── CUSTOM_MODEL_GUIDE.md    # <-- this file
└── ...
```

The YAML files plug directly into the `yolo` CLI via `model=models/NAME.yaml`. Python wrappers (if you choose to keep a registry similar to `meek/project1`) can live in the same folder.

---

## 2. Environment Setup (recap)

```bash
cd ultralytics
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]  # editable install so custom modules are picked up
```

For GPU training, install the CUDA-matched `torch`/`torchvision` wheels before `pip install -e .`.

---

## 3. Faster R-CNN Wrapper

Ultralytics is YOLO-centric, so we wrap TorchVision’s Faster R-CNN in a thin training harness:

1. Create `models/faster_rcnn.py` with a class/function that:
   - Loads `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=3)`
   - Adapts the number of classes via `model.roi_heads.box_predictor.cls_score`
   - Implements simple train/val loops reading the same dataset YAML (parse paths, build PyTorch dataset/dataloader)
   - Logs checkpoints under `runs/faster_rcnn/<name>` to keep parity with YOLO runs

2. Provide a CLI entry point, e.g.

   ```python
   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--data", default="dataset/facemask/data.yaml")
       parser.add_argument("--epochs", type=int, default=30)
       parser.add_argument("--batch", type=int, default=4)
       parser.add_argument("--device", default="cuda")
       args = parser.parse_args()
       train_faster_rcnn(args)
   ```

3. Run via:

   ```bash
  python models/faster_rcnn.py --data dataset/facemask/data.yaml --epochs 40 --batch 8
   ```

Keep the wrapper modular so it can share augmentation/transforms with YOLO (e.g., reuse Albumentations pipelines if desired).

---

## 4. YOLOv8 + TorchVision Backbones (ResNet, VGG16)

Ultralytics already ships a flexible YAML-driven model builder. We can harness the existing `TorchVision` module (see `ultralytics/nn/modules/block.py`) to emit feature maps from VGG/ResNet and then hand them to a YOLO head.

### 4.1 Create YAML templates

#### `models/yolov8_resnet.yaml`

```yaml
# Ultralytics YOLOv8 with TorchVision ResNet-50 backbone
nc: 2  # override on CLI if needed
scales: [8, 16, 32]

backbone:
  # name, #repeats, module, args
  - [-1, 1, TorchVision, ["resnet50", "IMAGENET1K_V1", True, 2, True]]  # split=True yields multi-scale outputs
  - [[1, 2, 3], 1, YOLOv8BackboneAdapter, [256, 512, 1024]]  # adapter merges TorchVision outputs to YOLO channels

head:
  - [[0, 1, 2], 1, YOLOv8DetectHead, [nc]]
```

#### `models/yolov8_vgg16.yaml`

```yaml
# Ultralytics YOLOv8 with TorchVision VGG16 backbone + custom SPP/CSP block
nc: 2

backbone:
  - [-1, 1, TorchVision, ["vgg16", "IMAGENET1K_V1", True, 2, True]]
  - [[1, 2, 3], 1, YOLOv8VGGBackboneAdapter, [256, 256, 256]]

head:
  - [[0, 1, 2], 1, YOLOv8VGGDetect, [nc]]
```

> **Important:** `YOLOv8BackboneAdapter`, `YOLOv8VGGBackboneAdapter`, and `YOLOv8VGGDetect` are small glue modules you add in `models/__init__.py` (or spread across Python files). They should:
>
> - Accept TorchVision feature maps (lists) and project them to the channel dimensions the YOLO head expects (`256/512/1024` etc.).
> - Optionally inject custom blocks (SPP, CSP, attention) like you had in `project1`.
> - Ultimately return the `[P3, P4, P5]` tensors that the stock `Detect` head (or your variant) consumes.

A barebones adapter might stack `Conv` layers and leverage existing modules (`SPPF`, `C2f`, etc.) so you don’t rewrite everything.

### 4.2 Implement glue modules

Place the helper modules in Python so the YAML references resolve:

```python
# models/__init__.py
from ultralytics.nn.modules import Conv, SPPF, C2f, Detect
import torch.nn as nn

class YOLOv8BackboneAdapter(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.proj = nn.ModuleList([Conv(ci, co, k=1) for ci, co in zip(ch_in, ch_out)])

    def forward(self, x):
        p3, p4, p5 = x  # TorchVision split outputs
        return [proj(f) for proj, f in zip(self.proj, (p3, p4, p5))]

class YOLOv8DetectHead(Detect):
    def __init__(self, nc, ch=(256, 512, 1024)):
        super().__init__(nc=nc, ch=ch)
```

Extend these stubs with your custom SPP/attention/CSP logic (mirroring `project1` but using existing Ultralytics blocks). Make sure to export the classes in `models/__all__` or `ultralytics/nn/modules/__init__.py` if you want global access.
Extend these stubs with your custom SPP/attention/CSP logic (mirroring `project1` but using existing Ultralytics blocks). Make sure to export the classes in `models/__all__` or `ultralytics/nn/modules/__init__.py` if you want global access.

### 4.3 Train / Validate

Once YAML and modules are in place, the standard CLI works:

```bash
# ResNet-backed YOLOv8
yolo train model=models/yolov8_resnet.yaml data=dataset/facemask/data.yaml epochs=60 imgsz=640 project=runs/resnet

# VGG16 hybrid
yolo train model=models/yolov8_vgg16.yaml data=dataset/facemask/data.yaml epochs=80 imgsz=640 project=runs/vgg16

# Evaluate and predict
yolo val model=runs/resnet/train/weights/best.pt data=dataset/facemask/data.yaml
yolo predict model=runs/vgg16/train/weights/best.pt source=dataset/facemask/test/images save=True
```

Set `nc` dynamically on CLI if the dataset has a different class count: `yolo train ... nc=2`.

---

## 5. Optional Registry Layer

If you appreciate the `project1`-style registry, mirror a slim version:

```python
# models/registry.py
MODEL_REGISTRY = {
    "faster_rcnn": {
    "module": "ultralytics.models.faster_rcnn",
        "class": "FasterRCNNWrapper"
    },
    "yolov8_resnet": {
    "module": "ultralytics.models.builders",
        "class": "YOLOv8ResNetBuilder"
    },
  "yolov8_vgg16": {
    "module": "ultralytics.models.builders",
    "class": "YOLOv8VGGBuilder"
    },
}
```

Each builder would expose a unified API (`build(num_classes, config)`), instantiating the underlying YOLO/TorchVision model. Hook this registry into your experiment scripts if you want parity with `meek/project1`.

---

## 6. Testing & Maintenance

- Run `pytest tests/` after touching internals to ensure core YOLO functionality remains intact.
- Keep custom modules light and rely on existing Ultralytics blocks whenever possible.
- Store dataset configs and checkpoints under `dataset/` and `runs/` (already aligned with repo conventions).
- Document new commands in `COMMANDS.md` or `models/README.md` for quick reference.

---

## 7. Next Steps Checklist

1. [x] Scaffold `models/` directory with placeholders above.
2. [x] Implement adapter/detect modules using Ultralytics primitives.
3. [x] Wrap TorchVision Faster R-CNN with training loop.
4. Train each model on `dataset/facemask` and evaluate.
5. Iterate on architecture tweaks (SPP, attention, CSP depth) directly in YAML/Python modules.

This modular setup keeps the official Ultralytics tooling intact while letting you mix in external backbones and baseline detectors for comparison.
