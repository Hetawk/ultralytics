#!/usr/bin/env python3
"""
Randomized Smoothing Evaluation for MedDef-VISTA Ablation Variants
====================================================================
Wraps each existing distill_v2 model with Cohen et al. (2019) randomized
smoothing and computes:

  1. Certified accuracy  @ multiple L2 radii R
  2. Abstain rate        @ each sigma / n_samples setting
  3. Clean accuracy (smoothed)  — prediction without certification
  4. Empirical robust acc vs FGSM / PGD (for comparison with non-smoothed)

No retraining required — wraps existing best.pt checkpoints at inference.

Output per variant:
  --out-dir/<variant>/
      smoothing_results.json          full results
      certified_accuracy_curve.png    cert acc vs radius R
      sigma_sweep.csv                 cert acc / abstain @ multiple sigmas

Combined output:
  --out-dir/
      certified_accuracy_all.png      all-variant overlay
      smoothing_summary.csv           one row per variant

Usage:
  python randomized_smoothing_eval.py [--variant full] [--device 0]
  python randomized_smoothing_eval.py --variant all --device 0

References:
  Cohen et al. (2019) "Certified Adversarial Robustness via Randomized Smoothing"
  ICML 2019. https://arxiv.org/abs/1902.02918
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import norm as sp_norm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_BASE = PROJECT_DIR / "runs/classify/train_tbcr_final/tbcr"
DATA_DIR = Path(
    "/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/tbcr")

VARIANTS = ["full", "no_def", "no_freq", "no_patch", "no_cbam", "baseline"]
VARIANT_LABELS = {
    "full":     "MedDef-VISTA Full",
    "no_def":   "w/o DefenseModule",
    "no_freq":  "w/o FrequencyDefense",
    "no_patch": "w/o PatchConsistency",
    "no_cbam":  "w/o CBAM",
    "baseline": "Baseline ViT",
}

# Randomized smoothing parameters to sweep
SIGMA_VALUES = [0.12, 0.25, 0.50, 1.00]          # Gaussian noise std
N_SMOOTH = 1000   # samples per prediction for certification
N_SMOOTH_FAST = 100    # samples for clean/empirical accuracy (faster)
ALPHA = 0.001  # failure probability for certification
RADII = np.linspace(0.0, 2.0, 41)          # L2 radii to certify at
BATCH_SIZE = 64     # images per batch
MAX_SAMPLES = 420    # use full test set


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def find_weights(variant: str) -> Path:
    """Find distill_v2 best.pt, fall back to distill best.pt."""
    base = WEIGHTS_BASE / f"{variant}_small"
    for sub in ("distill_v2/weights/best.pt", "distill/weights/best.pt"):
        p = base / sub
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No weights found for variant '{variant}' under {base}")


def load_model(weights: Path, device: torch.device) -> nn.Module:
    """Load a MedDef2 checkpoint via ultralytics task loader."""
    sys.path.insert(0, str(PROJECT_DIR))
    from ultralytics.nn.tasks import load_checkpoint
    result = load_checkpoint(str(weights), device=device)
    # load_checkpoint returns (model, ckpt_dict)
    model = result[0] if isinstance(result, (tuple, list)) else result
    model.eval()
    # Unwrap DataParallel / DDP if present
    if hasattr(model, "module"):
        model = model.module
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def get_test_loader(data_dir: Path, imgsz: int = 224) -> DataLoader:
    # Ultralytics classifiers are trained on raw [0, 1] pixel values
    # (no ImageNet mean/std normalization) — do NOT add Normalize() here.
    tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    split_dir = data_dir / "test"
    if not split_dir.exists():
        split_dir = data_dir / "val"
    ds = datasets.ImageFolder(str(split_dir), transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(
        f"  Dataset: {split_dir}  ({len(ds)} images, {len(ds.classes)} classes: {ds.classes})")
    return loader, ds.classes


# ═════════════════════════════════════════════════════════════════════════════
# Randomized Smoothing
# ═════════════════════════════════════════════════════════════════════════════

class SmoothedClassifier:
    """
    Cohen et al. (2019) Randomized Smoothing wrapper.

    g(x) = argmax_c P[f(x + ε) = c]  where ε ~ N(0, σ²I)

    Certifies L2 robustness radius  R = σ/2 * (Φ⁻¹(p_A) - Φ⁻¹(p_B))
    where p_A ≥ p_B are the top-2 smoothed class probabilities, reduced
    to the one-sided bound  R = σ * Φ⁻¹(p̲_A) when only pA is bounded.
    """

    def __init__(self, model: nn.Module, sigma: float, device: torch.device,
                 n_classes: int = 2):
        self.model = model
        self.sigma = sigma
        self.device = device
        self.n_classes = n_classes

    def _count_samples(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Run n noisy forward passes for a single image x (C,H,W).
        Returns a class-count vector of shape (n_classes,).
        """
        counts = torch.zeros(self.n_classes, dtype=torch.long)
        remaining = n
        while remaining > 0:
            bs = min(BATCH_SIZE, remaining)
            noise = torch.randn(bs, *x.shape, device=self.device) * self.sigma
            batch = x.unsqueeze(0).expand(
                bs, -1, -1, -1).to(self.device) + noise
            with torch.no_grad():
                logits = self.model(batch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                preds = logits.argmax(dim=1).cpu()
            for p in preds:
                counts[p.item()] += 1
            remaining -= bs
        return counts

    def predict(self, x: torch.Tensor, n: int = N_SMOOTH_FAST) -> int:
        """Return the smoothed prediction (may abstain by returning -1)."""
        counts = self._count_samples(x, n)
        top_class = int(counts.argmax())
        return top_class

    def certify(self, x: torch.Tensor, n0: int = 100, n: int = N_SMOOTH,
                alpha: float = ALPHA) -> Tuple[int, float]:
        """
        Returns (predicted_class, certified_radius).
        predicted_class == -1 means ABSTAIN.
        certified_radius == 0.0 when abstaining.

        Two-stage procedure (Cohen et al.):
          1. n0 samples → select top class c_A
          2. n  samples → lower-confidence-bound p_A via Clopper-Pearson
          3. Certify R = σ * Φ⁻¹(p_A)  if p_A > 0.5 else ABSTAIN
        """
        # Stage 1: pick top class
        counts0 = self._count_samples(x, n0)
        c_A = int(counts0.argmax())

        # Stage 2: estimate p_A with high confidence
        counts = self._count_samples(x, n)
        k_A = counts[c_A].item()
        p_A_lower = self._lower_confidence_bound(k_A, n, alpha)

        if p_A_lower > 0.5:
            radius = float(self.sigma * sp_norm.ppf(p_A_lower))
            return c_A, radius
        return -1, 0.0  # ABSTAIN

    @staticmethod
    def _lower_confidence_bound(k: int, n: int, alpha: float) -> float:
        """Clopper-Pearson one-sided lower confidence bound on p.

        Given k successes in n Bernoulli(p) trials, returns the lower
        (1-alpha) confidence bound on p using the exact Beta distribution,
        i.e.  p_lower = Beta.ppf(alpha, k, n-k+1).
        This is the formula used in Cohen et al. 2019 (Appendix A).
        """
        from scipy.stats import beta
        if k == 0:
            return 0.0
        if k == n:
            # When all samples agree, clamp to a value < 1 for numerical stability
            return float(beta.ppf(alpha, k, 1))
        return float(beta.ppf(alpha, k, n - k + 1))


# ═════════════════════════════════════════════════════════════════════════════
# Certification evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_smoothing(model: nn.Module, loader: DataLoader, sigma: float,
                       device: torch.device, max_samples: int = MAX_SAMPLES,
                       fast: bool = False) -> Dict:
    """
    Evaluate a SmoothedClassifier on the test loader.

    Returns a dict with:
      correct_radii   : list of (label, pred, radius) for non-abstained samples
      abstain_count   : int
      total           : int
      certified_acc_at_radii : {str(R): float}  for R in RADII
    """
    smoother = SmoothedClassifier(model, sigma, device)
    n_cert = N_SMOOTH_FAST if fast else N_SMOOTH
    n_sel = 100 if fast else 100  # selection samples (n0)

    results = []  # (label, prediction, radius)
    abstain_count = 0
    processed = 0

    model.eval()
    for images, labels in loader:
        for i in range(images.shape[0]):
            if processed >= max_samples:
                break
            x = images[i]
            lbl = int(labels[i])

            pred, radius = smoother.certify(x, n0=n_sel, n=n_cert, alpha=ALPHA)

            if pred == -1:
                abstain_count += 1
            else:
                results.append((lbl, pred, radius))
            processed += 1
            if processed % 20 == 0:
                certified_so_far = sum(1 for l, p, r in results if l == p)
                total_non_abs = len(results)
                print(f"    [{processed}/{min(max_samples, processed+1)}] "
                      f"abstain={abstain_count} "
                      f"cert_acc_clean={certified_so_far}/{total_non_abs}"
                      f"  (σ={sigma})", flush=True)
        if processed >= max_samples:
            break

    total = processed
    # Certified accuracy at each radius R:
    #   fraction of ALL samples (including abstains) where model
    #   both predicted correctly AND radius >= R
    cert_acc = {}
    for R in RADII:
        R_str = f"{R:.3f}"
        n_cert_correct = sum(1 for l, p, r in results if l == p and r >= R)
        cert_acc[R_str] = n_cert_correct / total if total > 0 else 0.0

    # Clean smoothed accuracy (ignoring radius, just pred vs label on non-abstained)
    n_correct_nonabs = sum(1 for l, p, r in results if l == p)
    clean_acc_nonabs = n_correct_nonabs / len(results) if results else 0.0
    clean_acc_all = n_correct_nonabs / total if total > 0 else 0.0

    return {
        "sigma":              sigma,
        "total":              total,
        "abstain_count":      abstain_count,
        "abstain_rate":       abstain_count / total if total > 0 else 1.0,
        "clean_acc_nonabs":   clean_acc_nonabs,   # accuracy among non-abstained
        # accuracy over all (abstain = wrong)
        "clean_acc_all":      clean_acc_all,
        "certified_acc_at_radii": cert_acc,
        # raw (label, pred, radius) list
        "results":            results,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_certified_accuracy_curve(sigma_results: List[Dict], variant: str,
                                  out_path: Path):
    """Plot certified accuracy vs L2 radius R for all sigma values."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, res in enumerate(sigma_results):
        sigma = res["sigma"]
        radii = [float(k) for k in res["certified_acc_at_radii"].keys()]
        accs = [res["certified_acc_at_radii"][f"{r:.3f}"] * 100 for r in radii]
        lbl = f"σ={sigma:.2f}  (abstain={res['abstain_rate']*100:.1f}%)"
        ax.plot(radii, accs, color=colors[i %
                len(colors)], linewidth=2, label=lbl)

    ax.set_xlabel("L₂ Perturbation Radius  R", fontsize=12)
    ax.set_ylabel("Certified Accuracy (%)", fontsize=12)
    ax.set_title(f"Randomized Smoothing — {VARIANT_LABELS.get(variant, variant)}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(RADII))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_variants_overlay(all_results: Dict[str, List[Dict]],
                              target_sigma: float, out_path: Path):
    """
    Single plot: certified accuracy vs R for all variants at a fixed sigma,
    showing which variant is most certifiably robust.
    """
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = cm.get_cmap("tab10")

    for idx, (variant, sigma_results) in enumerate(all_results.items()):
        # find the matching sigma
        match = [r for r in sigma_results if abs(
            r["sigma"] - target_sigma) < 1e-6]
        if not match:
            continue
        res = match[0]
        radii = [float(k) for k in res["certified_acc_at_radii"].keys()]
        accs = [res["certified_acc_at_radii"][f"{r:.3f}"] * 100 for r in radii]
        label = VARIANT_LABELS.get(variant, variant)
        color = cmap(idx / max(len(all_results) - 1, 1))
        ax.plot(radii, accs, linewidth=2, label=label, color=color)

    ax.set_xlabel("L₂ Perturbation Radius  R", fontsize=12)
    ax.set_ylabel("Certified Accuracy (%)", fontsize=12)
    ax.set_title(f"Certified Accuracy — All Variants  (σ={target_sigma})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(RADII))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_clean_vs_certified(summary_rows: List[Dict], sigma: float,
                            out_path: Path):
    """
    Bar chart: clean_acc_all vs certified_acc_at_R=0.5 for each variant.
    Highlights the smoothing overhead vs clean model.
    """
    if not HAS_MPL:
        return
    variants = [r["variant"] for r in summary_rows]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    clean = [r["clean_acc_all"] * 100 for r in summary_rows]
    cert05 = [r.get("cert_R0.50", 0) * 100 for r in summary_rows]

    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, clean,  w, label="Smoothed clean acc", color="#4c72b0")
    ax.bar(x + w/2, cert05, w, label="Certified acc @ R=0.5", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Clean vs Certified Accuracy (σ={sigma})", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# CSV / JSON helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_sigma_sweep_csv(sigma_results: List[Dict], variant: str, out_dir: Path):
    import csv
    path = out_dir / "sigma_sweep.csv"
    representative_radii = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["variant", "sigma", "clean_acc_all_%", "abstain_rate_%"] + \
                 [f"cert_R{R:.2f}_%" for R in representative_radii]
        writer.writerow(header)
        for res in sigma_results:
            ca = res["certified_acc_at_radii"]
            row = [
                variant,
                res["sigma"],
                f"{res['clean_acc_all']*100:.2f}",
                f"{res['abstain_rate']*100:.2f}",
            ] + [f"{ca.get(f'{R:.3f}', 0)*100:.2f}" for R in representative_radii]
            writer.writerow(row)
    print(f"  Saved: {path}")


def save_summary_csv(all_summary: List[Dict], out_dir: Path, sigma: float):
    import csv
    path = out_dir / f"smoothing_summary_sigma{sigma:.2f}.csv"
    if not all_summary:
        return
    keys = list(all_summary[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_summary)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Per-variant evaluation
# ═════════════════════════════════════════════════════════════════════════════

def run_variant(variant: str, device: torch.device, out_dir: Path,
                fast: bool = False) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"  Variant: {VARIANT_LABELS.get(variant, variant)}")
    print(f"{'='*60}")

    weights = find_weights(variant)
    print(f"  Weights: {weights}")
    model = load_model(weights, device)

    loader, classes = get_test_loader(DATA_DIR)

    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    sigma_results = []
    for sigma in SIGMA_VALUES:
        print(f"\n  ── σ = {sigma} ──")
        res = evaluate_smoothing(model, loader, sigma, device,
                                 max_samples=MAX_SAMPLES, fast=fast)
        res["variant"] = variant
        sigma_results.append(res)

        # Progress summary
        print(f"    abstain rate : {res['abstain_rate']*100:.1f}%")
        print(
            f"    clean acc    : {res['clean_acc_all']*100:.2f}% (all incl. abstain)")
        for R in [0.25, 0.50, 1.00]:
            ca = res["certified_acc_at_radii"].get(f"{R:.3f}", 0)
            print(f"    cert @ R={R:.2f} : {ca*100:.2f}%")

    # Save raw results (excluding large results list for JSON size)
    save_data = []
    for r in sigma_results:
        d = {k: v for k, v in r.items() if k != "results"}
        save_data.append(d)
    json_path = variant_dir / "smoothing_results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Per-variant CSV
    save_sigma_sweep_csv(sigma_results, variant, variant_dir)

    # Plot: certified accuracy curve for this variant
    if HAS_MPL:
        plot_certified_accuracy_curve(sigma_results, variant,
                                      variant_dir / "certified_accuracy_curve.png")
        print(f"  Saved: {variant_dir / 'certified_accuracy_curve.png'}")

    del model
    torch.cuda.empty_cache()
    return sigma_results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Randomized Smoothing Eval for MedDef-VISTA")
    p.add_argument("--variant",  default="all",
                   help="Variant name or 'all'. Can be comma-separated, e.g. 'full,baseline'")
    p.add_argument("--device",   default="0",
                   help="CUDA device index (0-3). Will be overridden by CUDA_VISIBLE_DEVICES.")
    p.add_argument("--out-dir",  default=str(PROJECT_DIR / "runs/visualizations/smoothing"),
                   type=str, help="Output root directory")
    p.add_argument("--data",     default=str(DATA_DIR), type=str,
                   help="Path to TBCR processed_data directory")
    p.add_argument("--sigma",    default=None, type=float,
                   help="Run only this sigma value (default: sweep all)")
    p.add_argument("--n-smooth", default=N_SMOOTH, type=int,
                   help="Samples per certification (default 1000)")
    p.add_argument("--max-samples", default=MAX_SAMPLES, type=int,
                   help="Max test images to certify (default 420 = full set)")
    p.add_argument("--fast",     action="store_true",
                   help="Quick mode: 100 samples per cert (less accurate)")
    p.add_argument("--summary-sigma", default=0.25, type=float,
                   help="Sigma to use for cross-variant summary plots (default 0.25)")
    return p.parse_args()


def main():
    args = parse_args()

    # Override globals from CLI
    global DATA_DIR, N_SMOOTH, MAX_SAMPLES, SIGMA_VALUES
    DATA_DIR = Path(args.data)
    N_SMOOTH = args.n_smooth
    MAX_SAMPLES = args.max_samples
    if args.sigma is not None:
        SIGMA_VALUES = [args.sigma]

    # Device
    dev_str = args.device
    # When launched via CUDA_VISIBLE_DEVICES=N, physical GPU N appears as cuda:0
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        dev_str = "0"
    device = torch.device(
        f"cuda:{dev_str}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve variants
    if args.variant == "all":
        variants_to_run = VARIANTS
    else:
        variants_to_run = [v.strip() for v in args.variant.split(",")]

    print(f"\nVariants: {variants_to_run}")
    print(f"Sigmas  : {SIGMA_VALUES}")
    print(f"N smooth: {N_SMOOTH} per certification")
    print(f"Samples : {MAX_SAMPLES}")
    print(f"Output  : {out_dir}")

    all_results: Dict[str, List[Dict]] = {}
    failed_variants = []

    for variant in variants_to_run:
        try:
            sigma_results = run_variant(
                variant, device, out_dir, fast=args.fast)
            all_results[variant] = sigma_results
        except FileNotFoundError as e:
            print(f"  SKIP {variant}: {e}")
            failed_variants.append(variant)
        except Exception as e:
            print(f"  ERROR {variant}: {e}")
            import traceback
            traceback.print_exc()
            failed_variants.append(variant)

    if failed_variants and not all_results:
        print(f"\nAll variants failed: {failed_variants}")
        sys.exit(1)

    if len(all_results) < 2:
        print("\nOnly one variant — skipping cross-variant plots.")
        return

    # ── Cross-variant summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Cross-variant summary")
    print(f"{'='*60}")

    # Overlay plot at summary_sigma
    if HAS_MPL:
        overlay_path = out_dir / \
            f"certified_accuracy_all_sigma{args.summary_sigma:.2f}.png"
        plot_all_variants_overlay(
            all_results, args.summary_sigma, overlay_path)
        print(f"  Saved: {overlay_path}")

    # Summary CSV rows (one per variant × sigma)
    summary_rows = []
    for variant, sigma_results in all_results.items():
        for res in sigma_results:
            if abs(res["sigma"] - args.summary_sigma) > 1e-6:
                continue
            ca = res["certified_acc_at_radii"]
            row = {
                "variant":         variant,
                "label":           VARIANT_LABELS.get(variant, variant),
                "sigma":           res["sigma"],
                "clean_acc_all_%": round(res["clean_acc_all"] * 100, 2),
                "abstain_rate_%":  round(res["abstain_rate"] * 100, 2),
                "cert_R0.00_%":    round(ca.get("0.000", 0) * 100, 2),
                "cert_R0.25_%":    round(ca.get("0.250", ca.get("0.249", 0)) * 100, 2),
                "cert_R0.50_%":    round(ca.get("0.500", ca.get("0.499", 0)) * 100, 2),
                "cert_R0.75_%":    round(ca.get("0.750", ca.get("0.749", 0)) * 100, 2),
                "cert_R1.00_%":    round(ca.get("1.000", ca.get("0.999", 0)) * 100, 2),
            }
            row["cert_R0.50"] = ca.get("0.500", ca.get("0.499", 0))  # for plot
            summary_rows.append(row)

    save_summary_csv(summary_rows, out_dir, args.summary_sigma)

    if HAS_MPL and summary_rows:
        bar_path = out_dir / \
            f"clean_vs_certified_sigma{args.summary_sigma:.2f}.png"
        plot_clean_vs_certified(summary_rows, args.summary_sigma, bar_path)
        print(f"  Saved: {bar_path}")

    # Print table
    print(f"\n{'Variant':<28} {'Clean%':>8} {'Abstain%':>9} "
          f"{'R=0.25%':>8} {'R=0.50%':>8} {'R=1.00%':>8}")
    print("-" * 75)
    for row in summary_rows:
        print(f"  {row['label']:<26} {row['clean_acc_all_%']:>8.2f} "
              f"{row['abstain_rate_%']:>9.2f} "
              f"{row['cert_R0.25_%']:>8.2f} "
              f"{row['cert_R0.50_%']:>8.2f} "
              f"{row['cert_R1.00_%']:>8.2f}")

    print(f"\nAll done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
