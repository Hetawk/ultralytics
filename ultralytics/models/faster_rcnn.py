"""TorchVision Faster R-CNN helpers shared across Ultralytics utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from ultralytics.utils import YAML

__all__ = [
    "YoloDetectionDataset",
    "collate_fn",
    "load_data_config",
    "resolve_split",
    "build_model",
]


class YoloDetectionDataset(Dataset):
    """Map Ultralytics-formatted datasets (YOLO txt labels) to TorchVision detection tensors."""

    def __init__(self, images_dir: Path, labels_dir: Path, transforms=None) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.samples = sorted(p for p in images_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.samples:
            raise FileNotFoundError(f"No images found under {images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        boxes = []
        labels = []
        if label_path.exists():
            with label_path.open() as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    labels.append(int(cls))
                    boxes.append(self._yolo_to_pascal(cx, cy, bw, bh, width, height))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp(min=0)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        image_tensor = TF.to_tensor(image)
        if self.transforms:
            image_tensor = self.transforms(image_tensor)

        return image_tensor, target

    @staticmethod
    def _yolo_to_pascal(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> list[float]:
        x_c = cx * width
        y_c = cy * height
        half_w = (bw * width) / 2.0
        half_h = (bh * height) / 2.0
        x1 = max(x_c - half_w, 0.0)
        y1 = max(y_c - half_h, 0.0)
        x2 = min(x_c + half_w, width)
        y2 = min(y_c + half_h, height)
        return [x1, y1, x2, y2]


def collate_fn(batch: Iterable):
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_data_config(cfg_path: Path) -> dict:
    """Load dataset YAML configuration."""
    data = YAML.load(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML at {cfg_path}")
    return data


def resolve_split(base: Path, entry: str | None) -> Tuple[Path, Path] | None:
    if not entry:
        return None
    images_dir = (base / entry).resolve()
    if images_dir.is_dir() and images_dir.name != "images":
        candidate = images_dir / "images"
        if candidate.is_dir():
            images_dir = candidate
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected images/labels directories for split at {images_dir.parent}")
    return images_dir, labels_dir


def build_model(
    num_classes: int,
    trainable_backbone_layers: int = 3,
    weights: str | None = "DEFAULT",
    weights_backbone: str | None = None,
    pretrained: bool | None = None,
    pretrained_backbone: bool | None = None,
) -> torchvision.models.detection.FasterRCNN:
    """Build Faster R-CNN with flexible weight-loading for differing torchvision versions."""

    def _call_new_api():
        kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
        if weights is not None:
            kwargs["weights"] = weights
        else:
            kwargs["weights"] = None
        if weights_backbone is not None:
            kwargs["weights_backbone"] = weights_backbone
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)

    def _call_legacy_api():
        use_pretrained = pretrained if pretrained is not None else bool(weights not in (None, "DEFAULT"))
        use_pretrained_backbone = (
            pretrained_backbone if pretrained_backbone is not None else use_pretrained
        )
        kwargs = {
            "trainable_backbone_layers": trainable_backbone_layers,
            "pretrained": use_pretrained,
            "pretrained_backbone": use_pretrained_backbone,
        }
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)

    try:
        model = _call_new_api()
    except TypeError:
        model = _call_legacy_api()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
