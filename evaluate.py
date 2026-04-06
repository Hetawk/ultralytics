#!/usr/bin/env python3
"""
Classification Evaluation & Visualization CLI
===============================================

Model-agnostic post-training evaluation, metrics computation, visualization
generation, saliency maps, and model export.

Works with **any** Ultralytics-compatible classifier (YOLO, MedDef, custom).

All heavy logic lives in ``ultralytics.utils.eval_engine`` — this file is
only the CLI argument parser + thin dispatch.

Usage Examples
--------------
# Evaluate a trained model on its validation set
python evaluate.py --model runs/classify/train/tbcr/full/weights/best.pt \\
                   --data /data2/enoch/tbcr

# Evaluate with all visualizations + saliency maps
python evaluate.py --model runs/classify/train/tbcr/full/weights/best.pt \\
                   --data /data2/enoch/tbcr --visualize --saliency

# Generate visualizations from existing results (no inference)
python evaluate.py --results-dir runs/classify/train/tbcr/full --viz-only

# Compare ablation variants across a dataset
python evaluate.py --compare --results-dir runs/classify/train/tbcr

# Export model to ONNX / TorchScript
python evaluate.py --model runs/classify/train/tbcr/full/weights/best.pt \\
                   --export onnx torchscript
"""

from __future__ import annotations

import argparse
import sys

from ultralytics.utils.eval_engine import (
    EXPORT_FORMATS,
    run_evaluation,
    run_visualization,
    run_comparison,
    run_saliency,
    run_embeddings,
    run_robustness,
    run_export,
    resolve_save_dir,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classification Evaluation, Visualization & Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Model & Data ---
    g_model = parser.add_argument_group("Model & Data")
    g_model.add_argument("--model", type=str, default=None,
                         help="Path to trained weights (.pt)")
    g_model.add_argument("--data", type=str, default=None,
                         help="Path to dataset root (e.g., /data2/enoch/tbcr)")
    g_model.add_argument("--split", type=str, default="val",
                         choices=["val", "test", "train"],
                         help="Data split to evaluate (default: val)")

    # --- Evaluation ---
    g_eval = parser.add_argument_group("Evaluation")
    g_eval.add_argument("--batch", type=int, default=32, help="Batch size")
    g_eval.add_argument("--imgsz", type=int, default=224, help="Image size")
    g_eval.add_argument("--device", type=str, default="0", help="CUDA device")
    g_eval.add_argument("--workers", type=int, default=8, help="Dataloader workers")

    # --- Visualization ---
    g_viz = parser.add_argument_group("Visualization")
    g_viz.add_argument("--visualize", action="store_true",
                       help="Generate all visualization plots")
    g_viz.add_argument("--viz-only", action="store_true",
                       help="Only generate visualizations from existing metrics (no inference)")
    g_viz.add_argument("--results-dir", type=str, default=None,
                       help="Directory with existing results for --viz-only or --compare")
    g_viz.add_argument("--saliency", action="store_true",
                       help="Generate comprehensive saliency maps (Grad-CAM, Grad-CAM++, etc.)")
    g_viz.add_argument("--n-saliency", type=int, default=8,
                       help="Number of saliency map samples (default: 8)")
    g_viz.add_argument("--tsne", action="store_true",
                       help="Generate t-SNE embedding plot")
    g_viz.add_argument("--pca", action="store_true",
                       help="Generate PCA embedding plot")

    # --- Comparison ---
    g_compare = parser.add_argument_group("Comparison")
    g_compare.add_argument("--compare", action="store_true",
                           help="Compare all variants in --results-dir")
    g_compare.add_argument("--variants", nargs="+", default=None,
                           help="Variant names to compare (default: auto-detect)")

    # --- Robustness ---
    g_robust = parser.add_argument_group("Adversarial Robustness")
    g_robust.add_argument("--robustness", action="store_true",
                          help="Run adversarial robustness evaluation")
    g_robust.add_argument("--attacks", nargs="+",
                          default=["fgsm", "pgd", "bim", "mim", "cw",
                                   "deepfool", "apgd", "square"],
                          choices=["fgsm", "pgd", "cw", "bim", "mim",
                                   "deepfool", "apgd", "square"],
                          help="Adversarial attacks to evaluate "
                               "(default: all 8 — fgsm pgd bim mim cw deepfool apgd square)")
    g_robust.add_argument("--epsilons", nargs="+", type=float, default=None,
                          help="Perturbation budgets for epsilon sweep "
                               "(default: 0 0.005 0.01 0.02 0.03 0.05 0.1 0.15 0.2 0.3)")
    g_robust.add_argument("--certified", action="store_true",
                          help="Run randomized-smoothing certified robustness")
    g_robust.add_argument("--certified-sigma", type=float, default=0.25,
                          help="Noise σ for randomized smoothing (default: 0.25)")
    g_robust.add_argument("--certified-n", type=int, default=100,
                          help="Number of noise samples for smoothing (default: 100)")

    # --- Export ---
    g_export = parser.add_argument_group("Export")
    g_export.add_argument("--export", nargs="+", default=None,
                          choices=EXPORT_FORMATS,
                          help="Export model to specified formats")

    # --- Output ---
    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory (default: auto from model path)")
    g_out.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # Flush any stale CUDA errors left by previous OOM crashes on this device.
    # torch.cuda.synchronize() consumes a pending asynchronous OOM error so
    # subsequent CUDA allocations succeed on an otherwise-free GPU.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    # --- viz-only mode (no model needed) ---
    if args.viz_only:
        if not args.results_dir:
            print("ERROR: --results-dir required with --viz-only")
            sys.exit(1)
        run_visualization(args.results_dir)
        return

    # --- comparison mode (no model needed) ---
    if args.compare:
        if not args.results_dir:
            print("ERROR: --results-dir required with --compare")
            sys.exit(1)
        run_comparison(args.results_dir, variants=args.variants)
        return

    # --- full evaluation ---
    if not args.model:
        print("ERROR: --model is required for evaluation")
        sys.exit(1)
    if not args.data:
        print("ERROR: --data is required for evaluation")
        sys.exit(1)

    enhanced, validator, save_dir = run_evaluation(
        model_path=args.model,
        data=args.data,
        save_dir=args.output_dir,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        verbose=args.verbose,
    )

    # --- optional add-ons (each is independent) ---
    if args.visualize:
        run_visualization(save_dir, metrics=enhanced)

    if args.saliency:
        run_saliency(
            args.model, args.data, save_dir,
            n_samples=args.n_saliency, device=args.device,
            imgsz=args.imgsz,
        )

    if args.visualize or args.tsne or args.pca:
        run_embeddings(
            args.model, args.data, save_dir,
            device=args.device,
            do_tsne=args.tsne or args.visualize,
            do_pca=args.pca or args.visualize,
            imgsz=args.imgsz,
        )

    if args.export:
        run_export(args.model, args.export, imgsz=args.imgsz, device=args.device)

    if args.robustness:
        run_robustness(
            model_path=args.model,
            data_path=args.data,
            save_dir=save_dir,
            attacks=args.attacks,
            epsilons=args.epsilons,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            certified=args.certified,
            certified_sigma=args.certified_sigma,
            certified_n=args.certified_n,
            verbose=args.verbose,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
