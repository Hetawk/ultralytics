"""Generic trainer for Ultralytics custom-model registry."""

from __future__ import annotations

import argparse
import ast
import math
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.data.utils import check_det_dataset
from ultralytics.model_loader import available_models, get_model_entry
from ultralytics.models.faster_rcnn import YoloDetectionDataset, collate_fn, resolve_split
from ultralytics.utils import LOGGER, colorstr


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


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device(device_str)


def parse_model_kwargs(arg_list: list[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for item in arg_list:
        if "=" not in item:
            raise ValueError(f"Model argument '{item}' must be formatted as key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid model argument '{item}' (empty key)")
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        parsed[key] = value
    return parsed


def train(args: argparse.Namespace) -> None:
    cfg_path = Path(args.data).resolve()
    LOGGER.info(colorstr("bold", "\nChecking dataset..."))
    data_cfg = check_det_dataset(str(cfg_path), autodownload=True)
    base = Path(data_cfg["path"])

    splits = {key: resolve_split(base, data_cfg.get(key)) for key in ("train", "val")}
    if splits["train"] is None:
        raise ValueError("Dataset YAML must define a train split")

    nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
    if not nc:
        raise ValueError("Dataset configuration must specify 'nc' or provide class 'names'")
    num_classes = nc + 1

    device = resolve_device(args.device)
    train_images, train_labels = splits["train"]
    val_pair = splits.get("val")

    train_dataset = YoloDetectionDataset(train_images, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = None
    if val_pair:
        val_dataset = YoloDetectionDataset(*val_pair)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    model_entry = get_model_entry(args.model)
    model_kwargs = parse_model_kwargs(args.model_args)
    model = model_entry.build(num_classes=num_classes, **model_kwargs)
    model.to(device)

    LOGGER.info(f"Training model: {colorstr('bold', args.model)} -> {model_entry.description or 'custom model'}")
    LOGGER.info(f"Classes: {nc} (+ background) | Device: {device}")
    if model_kwargs:
        LOGGER.info(f"Model overrides: {model_kwargs}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    project_dir = Path(args.project).resolve()
    run_name = args.name or args.model
    run_dir = project_dir / run_name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Artifacts -> {colorstr('bold', str(run_dir))}")

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
        LOGGER.info(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={elapsed:.1f}s")

        torch.save(model.state_dict(), weights_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")
            LOGGER.info(f"{colorstr('green', 'bold', 'New best model saved!')} val_loss={best_loss:.4f}")

    LOGGER.info(
        f"\n{colorstr('green', 'bold', 'Training complete!')} Checkpoints saved to {colorstr('bold', str(weights_dir))}"
    )
    LOGGER.info(f"Best validation loss: {best_loss:.4f}")


def parse_args() -> argparse.Namespace:
    registered = sorted(available_models().keys())
    parser = argparse.ArgumentParser(description="Train custom Ultralytics models via the registry")
    parser.add_argument("--model", type=str, default="faster_rcnn", choices=registered, help="Registered model key")
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
    parser.add_argument("--name", type=str, default=None, help="Run name inside the project directory (defaults to model)")
    parser.add_argument(
        "--model-arg",
        dest="model_args",
        action="append",
        default=[],
        help="Override model-specific kwargs using key=value pairs (can be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
