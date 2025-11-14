"""Lightweight training harness for TorchVision Faster R-CNN using YOLO-style datasets."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


class YoloDetectionDataset(Dataset):
    """Map Ultralytics-formatted datasets (YOLO txt labels) to TorchVision detection tensors."""

    def __init__(self, images_dir: Path, labels_dir: Path, _class_count: int, transforms=None) -> None:
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
    with cfg_path.open() as cf:
        data = yaml.safe_load(cf)
    if not data:
        raise ValueError(f"Unable to parse dataset config: {cfg_path}")
    return data


def resolve_split(base: Path, entry: str | None) -> tuple[Path, Path] | None:
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


def build_model(num_classes: int, trainable_backbone_layers: int = 3) -> torchvision.models.detection.FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=trainable_backbone_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(model, data_loader, device) -> float:
    if not data_loader:
        return math.inf
    model.train()  # required for loss computation in torchvision detectors
    running = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            running += loss.item()
    return running / max(len(data_loader), 1)


def train(args: argparse.Namespace) -> None:
    cfg_path = Path(args.data).resolve()
    data_cfg = load_data_config(cfg_path)
    base = cfg_path.parent

    splits = {}
    for key in ("train", "val"):
        splits[key] = resolve_split(base, data_cfg.get(key))
    if splits["train"] is None:
        raise ValueError("Dataset YAML must define a train split")

    nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
    if nc is None or nc < 1:
        raise ValueError("Dataset configuration must specify 'nc' or provide class 'names'")
    num_classes = nc + 1  # include background class for Faster R-CNN

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_images, train_labels = splits["train"]
    val_pair = splits.get("val")
    val_loader = None
    train_dataset = YoloDetectionDataset(train_images, train_labels, class_count=nc)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    if val_pair:
        val_dataset = YoloDetectionDataset(*val_pair, class_count=nc)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    model = build_model(num_classes=num_classes, trainable_backbone_layers=args.trainable_backbone_layers)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = project_dir / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        start = time.time()
        for images, targets in progress:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            progress.set_postfix({"loss": f"{batch_loss:.3f}"})

        scheduler.step()
        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device) if val_loader else math.inf
        elapsed = time.time() - start
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={elapsed:.1f}s")

        torch.save(model.state_dict(), weights_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")

    print(f"Training complete. Checkpoints saved to {weights_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TorchVision Faster R-CNN on Ultralytics datasets")
    parser.add_argument("--data", type=str, required=True, help="Path to Ultralytics YAML dataset file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr-step", type=int, default=5, help="StepLR step size")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="StepLR gamma")
    parser.add_argument("--device", type=str, default="auto", help="Training device (cuda|cpu|auto)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--project", type=str, default="runs", help="Output parent directory")
    parser.add_argument("--name", type=str, default="faster_rcnn", help="Run name inside the project directory")
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=3,
        help="Number of trainable layers in the ResNet backbone (0-5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
