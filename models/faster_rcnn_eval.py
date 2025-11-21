"""Utility script to evaluate Faster R-CNN checkpoints on a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ultralytics.data.utils import check_det_dataset
from ultralytics.model_loader import get_model_entry
from ultralytics.models.faster_rcnn import YoloDetectionDataset, collate_fn, resolve_split
from ultralytics.trainers.custom_trainer import evaluate, parse_model_kwargs
from ultralytics.utils import LOGGER, colorstr


def resolve_device(name: str) -> torch.device:
    """Mirror custom_trainer auto device resolution for consistency."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda"):
        return torch.device(name)
    return torch.device(name)


def fetch_split(data_cfg: dict, split: str) -> Optional[tuple[Path, Path]]:
    base = Path(data_cfg["path"])
    entry = data_cfg.get(split)
    if entry is None:
        return None
    return resolve_split(base, entry)


def build_dataloader(images: Path, labels: Path, batch: int, workers: int) -> DataLoader:
    dataset = YoloDetectionDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN checkpoints on Ultralytics datasets")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (Ultralytics format)")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt checkpoint produced by custom_trainer")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--device", type=str, default="auto", help="Device to evaluate on (cuda|cpu|auto)")
    parser.add_argument("--model", type=str, default="faster_rcnn", help="Registered model key (default: faster_rcnn)")
    parser.add_argument(
        "--model-arg",
        dest="model_args",
        action="append",
        default=[],
        help="Optional model overrides (key=value). Repeat flag for multiple values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    LOGGER.info(colorstr("bold", "Checking dataset config"))
    data_cfg = check_det_dataset(args.data, autodownload=True)
    split_pair = fetch_split(data_cfg, args.split)
    if split_pair is None:
        raise ValueError(f"Dataset config does not define a '{args.split}' split")

    LOGGER.info(colorstr("bold", f"Loading split '{args.split}'"))
    loader = build_dataloader(split_pair[0], split_pair[1], args.batch, args.workers)

    nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
    if not nc:
        raise ValueError("Dataset config must provide 'nc' or 'names'")
    num_classes = nc + 1  # torchvision background class

    model_entry = get_model_entry(args.model)
    model_kwargs = parse_model_kwargs(args.model_args)
    model = model_entry.build(num_classes=num_classes, **model_kwargs)

    LOGGER.info(colorstr("bold", f"Loading weights from {args.weights}"))
    checkpoint = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        LOGGER.warning("State dict load warnings: missing=%s unexpected=%s", missing, unexpected)

    device = resolve_device(args.device)
    model.to(device)

    loss = evaluate(model, loader, device)
    LOGGER.info(colorstr("green", f"{args.model} {args.split} loss: {loss:.4f}"))


if __name__ == "__main__":
    main()
