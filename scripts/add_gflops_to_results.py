#!/usr/bin/env python3
"""
Add GFLOPs column to existing results.csv files from trained model checkpoints.

This script scans training run directories, loads the best.pt checkpoint,
computes GFLOPs, and updates the results.csv with a new 'model/GFLOPs' column.

Usage:
    python scripts/add_gflops_to_results.py                           # scan all runs
    python scripts/add_gflops_to_results.py --run-dir runs/detect/train/cface/yolov8_resnet
    python scripts/add_gflops_to_results.py --runs-root runs --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl


def get_model_gflops(weights_path: Path, imgsz: int = 640, device: str = "cpu") -> float | None:
    """Load a checkpoint and compute GFLOPs.

    Args:
        weights_path: Path to .pt checkpoint file.
        imgsz: Image size for FLOP calculation.
        device: Device to load model on.

    Returns:
        GFLOPs value or None if computation fails.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        # model.info() returns (layers, params, gradients, flops)
        info = model.info(verbose=False)
        if info and len(info) >= 4:
            return round(info[3], 3)  # GFLOPs
    except Exception as e:
        print(f"  ⚠ Could not compute GFLOPs for {weights_path}: {e}")
    return None


def find_run_directories(runs_root: Path) -> list[Path]:
    """Recursively find directories containing both results.csv and weights/best.pt."""
    run_dirs = []
    for results_csv in runs_root.rglob("results.csv"):
        run_dir = results_csv.parent
        weights_dir = run_dir / "weights"
        if weights_dir.is_dir() and (weights_dir / "best.pt").exists():
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def update_results_csv(run_dir: Path, gflops: float, dry_run: bool = False) -> bool:
    """Add GFLOPs column to results.csv.

    Args:
        run_dir: Directory containing results.csv.
        gflops: GFLOPs value to add.
        dry_run: If True, don't modify files.

    Returns:
        True if updated successfully.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return False

    try:
        df = pl.read_csv(csv_path, infer_schema_length=None)

        # Check if GFLOPs column already exists
        if "model/GFLOPs" in df.columns:
            existing_val = df["model/GFLOPs"][0]
            if existing_val is not None and existing_val > 0:
                print(f"  ✓ GFLOPs already present ({existing_val}), skipping")
                return True

        # Add GFLOPs column (same value for all rows since it's a model property)
        df = df.with_columns(pl.lit(gflops).alias("model/GFLOPs"))

        if dry_run:
            print(f"  [DRY-RUN] Would add GFLOPs={gflops} to {csv_path}")
            return True

        # Backup original
        backup_path = csv_path.with_suffix(".csv.bak")
        if not backup_path.exists():
            shutil.copy(csv_path, backup_path)

        # Write updated CSV
        df.write_csv(csv_path)
        print(f"  ✓ Added GFLOPs={gflops} to {csv_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to update {csv_path}: {e}")
        return False


def process_single_run(run_dir: Path, imgsz: int = 640, device: str = "cpu", dry_run: bool = False) -> bool:
    """Process a single run directory."""
    print(f"\nProcessing: {run_dir}")

    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = run_dir / "weights" / "last.pt"
    if not weights_path.exists():
        print(f"  ✗ No checkpoint found in {run_dir / 'weights'}")
        return False

    gflops = get_model_gflops(weights_path, imgsz=imgsz, device=device)
    if gflops is None:
        return False

    return update_results_csv(run_dir, gflops, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Add GFLOPs to existing results.csv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory to scan for training runs (default: runs)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Process a specific run directory instead of scanning",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for GFLOPs calculation (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading model (default: cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files",
    )
    args = parser.parse_args()

    if args.run_dir:
        # Process single directory
        run_dirs = [args.run_dir] if args.run_dir.exists() else []
    else:
        # Scan for all run directories
        if not args.runs_root.exists():
            print(f"Runs root not found: {args.runs_root}")
            return 1
        run_dirs = find_run_directories(args.runs_root)

    if not run_dirs:
        print("No training run directories found.")
        return 1

    print(f"Found {len(run_dirs)} run(s) to process")
    if args.dry_run:
        print("[DRY-RUN MODE - no files will be modified]")

    success_count = 0
    for run_dir in run_dirs:
        if process_single_run(run_dir, imgsz=args.imgsz, device=args.device, dry_run=args.dry_run):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"Processed {success_count}/{len(run_dirs)} runs successfully")
    return 0 if success_count == len(run_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
