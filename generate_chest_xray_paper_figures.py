#!/usr/bin/env python3
"""
Generate Chest X-ray paper figures aligned with the TBCR figure set names.

Outputs:
  - class_distribution.png
  - fig2_radar_ablation.png
  - fig4_stage_improvement.png
  - fig7_metric_comparison.png

Notes:
  - Uses Chest X-ray stage1 robustness artifacts from eval_v2.
  - Distill/distill_v2 artifacts are currently unavailable for Chest X-ray,
    so fig4 visualizes stage availability + stage1 ranking (explicitly).
"""

from __future__ import annotations
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

VARIANTS_SMALL = [
    "full_small",
    "no_def_small",
    "no_freq_small",
    "no_patch_small",
    "no_cbam_small",
    "baseline_small",
]

VARIANTS = ["full", "no_def", "no_freq", "no_patch", "no_cbam", "baseline"]

LABELS_SHORT = {
    "no_freq": "MedDef-VISTA",
    "full": "Full",
    "no_patch": "w/o Patch",
    "no_cbam": "w/o CBAM",
    "no_def": "w/o DefMod",
    "baseline": "Baseline",
}

COLORS = {
    "no_freq": "#1565C0",
    "full": "#2E7D32",
    "no_patch": "#E65100",
    "no_cbam": "#6A1B9A",
    "no_def": "#C62828",
    "baseline": "#37474F",
}

MARKERS = {
    "no_freq": "*",
    "full": "D",
    "no_patch": "^",
    "no_cbam": "s",
    "no_def": "v",
    "baseline": "o",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Chest X-ray paper figures")
    p.add_argument(
        "--runs-root",
        default="runs/classify/train_chest_xray_final_eval_v2/chest_xray",
        help="Root containing <variant>_small/stage1/robustness/robustness_results.json",
    )
    p.add_argument(
        "--dataset-root",
        default="/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/chest_xray",
        help="Chest X-ray processed dataset root with val/test splits",
    )
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--out", default="out/chest_xray_figures")
    return p.parse_args()


def save_fig(fig: plt.Figure, fname: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved {path}")


def legend_handles(order: List[str]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS[v],
            color=COLORS[v],
            markersize=8,
            linestyle="None",
            label=LABELS_SHORT[v],
        )
        for v in order
    ]


def load_stage1_metrics(runs_root: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for vs in VARIANTS_SMALL:
        v = vs.replace("_small", "")
        p = runs_root / vs / "stage1" / "robustness" / "robustness_results.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing robustness file: {p}")
        with p.open("r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        per_attack = data.get("per_attack", {})

        metrics[v] = {
            "clean_acc": float(summary.get("clean_accuracy", 0.0)),
            "mean_robust": float(summary.get("mean_robust_accuracy", 0.0)),
            "mean_asr": float(summary.get("mean_attack_success_rate", 0.0)),
            "robust_ratio": float(summary.get("robustness_ratio", 0.0)) * 100.0,
            "apgd_robust": float(per_attack.get("apgd", {}).get("robust_accuracy", 0.0)),
            "deepfool_robust": float(per_attack.get("deepfool", {}).get("robust_accuracy", 0.0)),
            "n_samples": float(per_attack.get("fgsm", {}).get("samples", 0.0)),
        }
    return metrics


def fig_class_distribution(dataset_root: Path, split: str, out_dir: Path) -> None:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    counts = [len([f for f in d.rglob("*") if f.is_file()])
              for d in class_dirs]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Use consistent palette: baseline (gray) for NORMAL, no_freq (blue) for PNEUMONIA
    color_map = {
        "NORMAL": COLORS["baseline"],
        "PNEUMONIA": COLORS["no_freq"],
    }
    colors = [color_map.get(name.upper(), "#2E7D32") for name in class_names]
    bars = ax.bar(class_names, counts, color=colors,
                  edgecolor="black", linewidth=1.0)

    max_count = max(counts) if counts else 0
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            c + max_count * 0.01,
            str(c),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Chest X-ray {split.capitalize()} Split Class Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "class_distribution.png", out_dir)
    plt.close(fig)


def fig_radar(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    # Metrics chosen from available stage1 chest_xray artifacts.
    metric_defs = {
        "Clean Acc (%)": {v: metrics[v]["clean_acc"] for v in VARIANTS},
        "Mean Robust (%)": {v: metrics[v]["mean_robust"] for v in VARIANTS},
        "Attack Resistance (%)": {v: 100.0 - metrics[v]["mean_asr"] for v in VARIANTS},
        "APGD Robust (%)": {v: metrics[v]["apgd_robust"] for v in VARIANTS},
        "DeepFool Robust (%)": {v: metrics[v]["deepfool_robust"] for v in VARIANTS},
        "Robust Ratio (x100)": {v: metrics[v]["robust_ratio"] for v in VARIANTS},
    }

    categories = list(metric_defs.keys())
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    norm = {}
    for cat in categories:
        vals = np.array([metric_defs[cat][v] for v in VARIANTS], dtype=float)
        # Use robust percentile scaling to prevent outliers from collapsing the scale.
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        if hi <= lo:
            norm[cat] = {v: 0.5 for v in VARIANTS}
        else:
            norm[cat] = {
                v: float(
                    np.clip((metric_defs[cat][v] - lo) / (hi - lo), 0.0, 1.2))
                for v in VARIANTS
            }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for v in VARIANTS:
        vals = [norm[c][v] for c in categories]
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            color=COLORS[v],
            linewidth=2.2,
            marker=MARKERS[v],
            ms=7,
            label=LABELS_SHORT[v],
        )
        ax.fill(angles, vals, alpha=0.06, color=COLORS[v])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.3)
    ax.legend(handles=legend_handles(VARIANTS), loc="upper right",
              bbox_to_anchor=(1.43, 1.14), framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, "fig2_radar_ablation.png", out_dir)
    plt.close(fig)


def fig_stage_improvement(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    # Chest X-ray: stage1 is complete; distill/distill_v2 artifacts are not available yet.
    # Show two meaningful stage1 comparisons instead of unavailable stages.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: stage1 clean accuracy ranking.
    ax = axes[0]
    order = sorted(
        VARIANTS, key=lambda v: metrics[v]["clean_acc"], reverse=True)
    vals = [metrics[v]["clean_acc"] for v in order]
    y = np.arange(len(order))
    bars = ax.barh(y, vals, color=[COLORS[v] for v in order], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS_SHORT[v] for v in order])
    ax.invert_yaxis()
    ax.set_xlabel("Clean Accuracy (%)")
    ax.set_title("Stage-1: Clean Accuracy Ranking")

    for b, val in zip(bars, vals):
        ax.text(val + 0.3, b.get_y() + b.get_height() /
                2, f"{val:.2f}", va="center", fontsize=9)

    # Right: stage1 APGD robustness ranking (strongest attack in protocol).
    ax2 = axes[1]
    order_r = sorted(
        VARIANTS, key=lambda v: metrics[v]["apgd_robust"], reverse=True)
    vals_r = [metrics[v]["apgd_robust"] for v in order_r]
    y2 = np.arange(len(order_r))
    bars2 = ax2.barh(y2, vals_r, color=[COLORS[v] for v in order_r], alpha=0.9)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([LABELS_SHORT[v] for v in order_r])
    ax2.invert_yaxis()
    ax2.set_xlabel("APGD Robust Accuracy (%)")
    ax2.set_title("Stage-1: APGD Robustness Ranking")

    for b, val in zip(bars2, vals_r):
        ax2.text(val + 0.2, b.get_y() + b.get_height() /
                 2, f"{val:.2f}", va="center", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Chest X-ray distill/distill_v2 training artifacts are not yet available; figure shows stage1 comparisons.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "fig4_stage_improvement.png", out_dir)
    plt.close(fig)


def fig_metric_comparison(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    metric_names = [
        "Clean Acc (%)",
        "Mean Robust (%)",
        "Attack Resistance (%)",
        "APGD Robust (%)",
        "DeepFool Robust (%)",
        "Robust Ratio (x100)",
    ]

    metric_map = {
        "Clean Acc (%)": lambda v: metrics[v]["clean_acc"],
        "Mean Robust (%)": lambda v: metrics[v]["mean_robust"],
        "Attack Resistance (%)": lambda v: 100.0 - metrics[v]["mean_asr"],
        "APGD Robust (%)": lambda v: metrics[v]["apgd_robust"],
        "DeepFool Robust (%)": lambda v: metrics[v]["deepfool_robust"],
        "Robust Ratio (x100)": lambda v: metrics[v]["robust_ratio"],
    }

    x = np.arange(len(metric_names))
    width = 0.11
    offsets = np.linspace(-(len(VARIANTS) - 1) / 2,
                          (len(VARIANTS) - 1) / 2, len(VARIANTS)) * width

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, v in enumerate(VARIANTS):
        vals = [metric_map[m](v) for m in metric_names]
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.92,
            color=COLORS[v],
            alpha=0.88,
            label=LABELS_SHORT[v],
            edgecolor="white",
            linewidth=0.5,
        )
        for b, val in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.15,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=COLORS[v],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Metric Value (%)")
    ax.set_ylim(0, 109)
    ax.legend(handles=legend_handles(VARIANTS), loc="upper right",
              fontsize=9.0, framealpha=0.92, ncol=2)

    plt.tight_layout()
    save_fig(fig, "fig7_metric_comparison.png", out_dir)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    runs_root = Path(args.runs_root)
    dataset_root = Path(args.dataset_root)

    metrics = load_stage1_metrics(runs_root)

    print("Generating Chest X-ray figures...")
    fig_class_distribution(dataset_root, args.split, out_dir)
    fig_radar(metrics, out_dir)
    fig_stage_improvement(metrics, out_dir)
    fig_metric_comparison(metrics, out_dir)
    print(f"Done. Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
