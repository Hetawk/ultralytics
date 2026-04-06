"""
Classification Visualization Engine
=====================================

Model-agnostic, publication-quality visualization for any image
classification project.  All plots are 300 DPI, return ``plt.Figure``,
and optionally save to disk.

This module is **not** tied to any specific model architecture.

Available plots:
    1.  Training curves (loss + accuracy)
    2.  Confusion matrix (raw + normalized)
    3.  ROC / AUC curves (binary + multiclass)
    4.  Per-class metrics bar chart
    5.  Radar chart (multi-model comparison)
    6.  t-SNE / PCA embedding scatter
    7.  Epsilon-sensitivity line plots
    8.  Saliency / Grad-CAM overlays
    9.  Ablation comparison heatmap
    10. Attack success rate heatmap
    11. Metrics summary multi-panel
    12. Calibration reliability diagram

Orchestrator:
    ``ClassificationVisualizer`` — high-level one-liner API.

Usage:
    from ultralytics.utils.classify_visualize import ClassificationVisualizer
    viz = ClassificationVisualizer(save_dir="runs/classify/train/tbcr/full")
    viz.plot_training_curves("results.csv")
    viz.plot_confusion_matrix(cm, class_names)
    viz.generate_all(metrics_dict, results_csv)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style defaults (colorblind-safe, publication-quality)
# ---------------------------------------------------------------------------

_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (10, 7),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
}

# Colorblind-safe palette
COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#9C27B0",  # purple
    "#FF9800",  # amber
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#607D8B",  # blue-grey
    "#795548",  # brown
    "#3F51B5",  # indigo
    "#CDDC39",  # lime
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

# ---------------------------------------------------------------------------
# Registry: allows extending with custom plot functions
# ---------------------------------------------------------------------------
_CUSTOM_PLOTS_REGISTRY: Dict[str, callable] = {}


def register_plot(name: str):
    """Decorator to register a custom plot function.

    Example::

        @register_plot("my_custom_chart")
        def my_chart(data, output_path=None, **kwargs):
            fig, ax = plt.subplots()
            ...
            return fig
    """
    def decorator(fn):
        _CUSTOM_PLOTS_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_style():
    """Apply publication-quality rcParams."""
    plt.rcParams.update(_RCPARAMS)


def _save_fig(fig: plt.Figure, path: Optional[Path], close: bool = True):
    """Save figure if path is provided."""
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved plot → {path}")
    if close:
        plt.close(fig)


# ============================================================================
# 1. Training curves
# ============================================================================

def plot_training_curves(
    csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training loss, validation loss, and accuracy from results.csv.

    Parameters
    ----------
    csv_path : path to Ultralytics results.csv
    output_path : optional save path
    metrics : column substrings to plot (default: loss + accuracy)
    title : figure title
    """
    _apply_style()
    import csv as _csv

    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"results.csv not found: {path}")
        return None

    with open(path) as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Clean headers (ultralytics pads with spaces)
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
    epochs = list(range(1, len(rows) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Loss ---
    ax = axes[0]
    loss_keys = [k for k in rows[0] if "loss" in k.lower()]
    for i, key in enumerate(loss_keys):
        vals = [float(r[key]) for r in rows if r.get(key)]
        label = (
            key.replace("train/", "Train ")
            .replace("val/", "Val ")
            .replace("_", " ")
            .title()
        )
        ax.plot(
            epochs[: len(vals)], vals,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(vals) // 20),
            linewidth=2, label=label,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend()

    # --- Panel 2: Accuracy ---
    ax = axes[1]
    acc_keys = [k for k in rows[0] if "acc" in k.lower() or "top" in k.lower()]
    for i, key in enumerate(acc_keys):
        vals = [float(r[key]) for r in rows if r.get(key)]
        label = (
            key.replace("metrics/", "")
            .replace("_", " ")
            .title()
        )
        ax.plot(
            epochs[: len(vals)], vals,
            color=COLORS[(i + 3) % len(COLORS)],
            marker=MARKERS[(i + 3) % len(MARKERS)],
            markevery=max(1, len(vals) // 20),
            linewidth=2, label=label,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 2. Confusion matrix
# ============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap with annotations.

    Parameters
    ----------
    cm : (C, C) confusion matrix
    class_names : list of class labels
    normalize : if True, show percentages
    """
    _apply_style()
    nc = len(class_names)
    if figsize is None:
        side = max(6, nc * 0.7)
        figsize = (side + 2, side)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(
            cm.astype(float), row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns
        sns.heatmap(
            cm_plot, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, vmin=0, vmax=vmax,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )
    except ImportError:
        im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(nc))
        ax.set_yticklabels(class_names)
        for i in range(nc):
            for j in range(nc):
                val = cm_plot[i, j]
                txt = f"{val:{fmt}}" if isinstance(val, float) else str(val)
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 3. ROC / AUC curves
# ============================================================================

def plot_roc_auc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot ROC curves (binary or multiclass one-vs-rest).

    Parameters
    ----------
    y_true : (N,) ground-truth class ids
    y_probs : (N, C) predicted probabilities
    class_names : list of class labels
    """
    _apply_style()
    from sklearn.metrics import roc_curve, auc as sk_auc
    from sklearn.preprocessing import label_binarize

    nc = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 8))

    if nc == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{class_names[1]} (AUC = {roc_auc:.3f})")
    else:
        y_onehot = label_binarize(y_true, classes=range(nc))
        for i in range(nc):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            if y_onehot[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(
                fpr, tpr, linewidth=2,
                color=COLORS[i % len(COLORS)],
                label=f"{name} (AUC = {roc_auc:.3f})",
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 4. Per-class metrics bar chart
# ============================================================================

def plot_per_class_metrics(
    per_class: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Per-Class Metrics",
) -> plt.Figure:
    """Grouped horizontal bar chart of per-class precision, recall, F1."""
    _apply_style()

    names = [c["class_name"][:20] for c in per_class]
    prec = [c["precision"] for c in per_class]
    rec = [c["recall"] for c in per_class]
    f1 = [c["f1"] for c in per_class]

    y = np.arange(len(names))
    h = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))
    ax.barh(y - h, prec, h, label="Precision", color=COLORS[0], alpha=0.85)
    ax.barh(y, rec, h, label="Recall", color=COLORS[1], alpha=0.85)
    ax.barh(y + h, f1, h, label="F1", color=COLORS[2], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 5. Radar chart (multi-model comparison)
# ============================================================================

def plot_radar_chart(
    models: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Multi-Metric Comparison",
    metric_keys: Optional[List[str]] = None,
) -> plt.Figure:
    """Create radar / spider chart comparing models across multiple metrics.

    Parameters
    ----------
    models : {model_name: {metric_name: value, ...}, ...}
    metric_keys : which metrics to plot (default: all common keys)
    """
    _apply_style()

    if metric_keys is None:
        # Use intersection of all models' keys
        all_keys = [set(v.keys()) for v in models.values()]
        metric_keys = list(sorted(set.intersection(*all_keys))) if all_keys else []

    if len(metric_keys) < 3:
        logger.warning("Radar chart needs ≥ 3 metrics, got %d", len(metric_keys))
        return None

    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for idx, (model_name, metrics) in enumerate(models.items()):
        values = [metrics.get(k, 0) for k in metric_keys]
        # Auto-scale to [0, 1] if all values are already ≤ 1
        values += values[:1]

        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    labels = [k.replace("_", " ").title()[:18] for k in metric_keys]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 6. t-SNE / PCA
# ============================================================================

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "t-SNE Feature Embedding",
) -> plt.Figure:
    """2-D t-SNE scatter plot with class centroids."""
    _apply_style()
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[i % len(COLORS)], label=name, alpha=0.6, s=20,
        )
        # Centroid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, c=COLORS[i % len(COLORS)], s=200,
                   marker="X", edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


def plot_pca(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_components: int = 2,
    title: str = "PCA Feature Embedding",
) -> plt.Figure:
    """2-D PCA scatter plot."""
    _apply_style()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(features)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[i % len(COLORS)], label=name, alpha=0.6, s=20,
        )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 7. Epsilon-sensitivity
# ============================================================================

def plot_epsilon_sensitivity(
    results: Dict[str, Dict[float, float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Epsilon Sensitivity Analysis",
    xlabel: str = "Perturbation ε",
    ylabel: str = "Accuracy (%)",
) -> plt.Figure:
    """Line plot showing how metrics change with perturbation epsilon.

    Parameters
    ----------
    results : {model_name: {epsilon: metric_value, ...}, ...}
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (model_name, model_results) in enumerate(results.items()):
        epsilons = sorted(model_results.keys())
        values = [model_results[eps] for eps in epsilons]
        if max(values) <= 1:
            values = [v * 100 for v in values]

        ax.plot(
            epsilons, values, "o-",
            linewidth=2.5, markersize=8,
            label=model_name,
            color=COLORS[idx % len(COLORS)],
            marker=MARKERS[idx % len(MARKERS)],
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 8. Saliency map overlay
# ============================================================================

def plot_saliency_map(
    image: np.ndarray,
    saliency: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Saliency Map",
    alpha: float = 0.5,
) -> plt.Figure:
    """Overlay saliency heatmap on original image.

    Parameters
    ----------
    image : (H, W, 3) RGB image in [0, 255] or [0, 1]
    saliency : (H, W) saliency scores
    """
    _apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img = image.astype(float)
    if img.max() > 1.0:
        img = img / 255.0

    sal = saliency.astype(float)
    sal = (sal - sal.min()) / max(sal.max() - sal.min(), 1e-8)

    axes[0].imshow(img)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sal, cmap="jet")
    axes[1].set_title("Saliency", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(sal, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 9. Ablation comparison heatmap
# ============================================================================

def plot_ablation_heatmap(
    data: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Ablation Study — Component Impact",
) -> plt.Figure:
    """Heatmap of (variant × metric) for ablation studies.

    Parameters
    ----------
    data : {variant_name: {metric_name: value, ...}, ...}
    """
    _apply_style()

    variants = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    matrix = np.array([[data[v].get(m, 0) for m in metrics] for v in variants])

    fig, ax = plt.subplots(
        figsize=(max(10, len(metrics) * 1.2), max(5, len(variants) * 0.8))
    )

    try:
        import seaborn as sns
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=metrics, yticklabels=variants, ax=ax,
            vmin=0, vmax=1, cbar_kws={"label": "Score"},
        )
    except ImportError:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        for i in range(len(variants)):
            for j in range(len(metrics)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="Score")

    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 10. Attack success rate heatmap
# ============================================================================

def plot_asr_heatmap(
    data: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Attack Success Rate Heatmap",
) -> plt.Figure:
    """Heatmap of (defense_variant × attack_type) ASR values.

    Parameters
    ----------
    data : {defense_variant: {attack: asr, ...}, ...}
    """
    _apply_style()

    defenses = list(data.keys())
    attacks = list(next(iter(data.values())).keys())
    matrix = np.array([[data[d].get(a, 0) for a in attacks] for d in defenses])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "asr", ["#2196F3", "#FFEB3B", "#FF5722"], N=256,
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(attacks) * 1.5), max(5, len(defenses) * 0.8))
    )

    try:
        import seaborn as sns
        sns.heatmap(
            matrix * 100, annot=True, fmt=".1f", cmap=cmap,
            xticklabels=attacks, yticklabels=defenses, ax=ax,
            vmin=0, vmax=100, cbar_kws={"label": "ASR (%)"},
        )
    except ImportError:
        im = ax.imshow(matrix * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, rotation=45, ha="right")
        ax.set_yticks(range(len(defenses)))
        ax.set_yticklabels(defenses)
        for i in range(len(defenses)):
            for j in range(len(attacks)):
                ax.text(j, i, f"{matrix[i,j]*100:.1f}%",
                        ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="ASR (%)")

    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 11. Metrics summary multi-panel (2×2)
# ============================================================================

def plot_metrics_summary(
    metrics: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Classification Metrics Summary",
) -> plt.Figure:
    """2×2 summary: macro scores bar, per-class F1, calibration, highlights."""
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Macro metrics bar chart
    ax = axes[0, 0]
    macro_keys = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "specificity_macro", "balanced_accuracy", "mcc", "cohen_kappa",
    ]
    vals = [metrics.get(k, 0) for k in macro_keys]
    labels = [k.replace("_macro", "").replace("_", "\n").title() for k in macro_keys]
    bars = ax.bar(
        range(len(vals)), vals,
        color=COLORS[: len(vals)], alpha=0.85, edgecolor="white",
    )
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title("Aggregate Metrics", fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.3f}", ha="center", fontsize=8, fontweight="bold",
        )

    # Panel 2: Per-class F1 horizontal bar
    ax = axes[0, 1]
    per_class = metrics.get("per_class", [])
    if per_class:
        names = [c["class_name"][:15] for c in per_class]
        f1s = [c["f1"] for c in per_class]
        y_pos = range(len(names))
        colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
        ax.barh(y_pos, f1s, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("F1 Score")
        for i, v in enumerate(f1s):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title("Per-Class F1", fontweight="bold")

    # Panel 3: Calibration metrics
    ax = axes[1, 0]
    cal_keys = ["roc_auc", "average_precision", "log_loss", "brier_score", "ece"]
    cal_labels = ["ROC AUC", "Avg Precision", "Log Loss", "Brier Score", "ECE"]
    cal_vals = [metrics.get(k) for k in cal_keys]
    cal_data = [(l, v) for l, v in zip(cal_labels, cal_vals) if v is not None]
    if cal_data:
        c_labels, c_vals = zip(*cal_data)
        bars = ax.bar(
            range(len(c_vals)), c_vals,
            color=[COLORS[(5 + i) % len(COLORS)] for i in range(len(c_vals))],
            alpha=0.85, edgecolor="white",
        )
        ax.set_xticks(range(len(c_vals)))
        ax.set_xticklabels(c_labels, fontsize=9, rotation=30, ha="right")
        for bar, v in zip(bars, c_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", fontsize=8,
            )
    ax.set_title("Calibration & Probability Metrics", fontweight="bold")

    # Panel 4: Text highlights
    ax = axes[1, 1]
    ax.axis("off")
    highlights = [
        f"Accuracy:  {metrics.get('accuracy', 0):.4f}",
        f"F1 Macro:  {metrics.get('f1_macro', 0):.4f}",
        f"MCC:       {metrics.get('mcc', 0):.4f}",
        f"Cohen k:   {metrics.get('cohen_kappa', 0):.4f}",
        f"Classes:   {metrics.get('num_classes', '?')}",
        f"Samples:   {metrics.get('num_samples', '?')}",
    ]
    if metrics.get("roc_auc") is not None:
        highlights.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    if metrics.get("ece") is not None:
        highlights.append(f"ECE:       {metrics['ece']:.4f}")

    text = "\n".join(highlights)
    ax.text(
        0.1, 0.9, text, transform=ax.transAxes, fontsize=13,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", alpha=0.8),
    )
    ax.set_title("Key Highlights", fontweight="bold")

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# 12. Calibration reliability diagram
# ============================================================================

def plot_calibration_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
    title: str = "Calibration Reliability Diagram",
) -> plt.Figure:
    """Plot reliability diagram (confidence vs. accuracy) with histogram.

    Parameters
    ----------
    y_true : (N,) ground-truth class ids
    y_probs : (N, C) predicted probabilities
    """
    _apply_style()

    max_probs = y_probs.max(axis=1)
    pred_classes = y_probs.argmax(axis=1)
    correct = (pred_classes == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(correct[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    if bin_centers:
        ax1.bar(
            bin_centers, bin_accs,
            width=1 / n_bins * 0.8, alpha=0.7,
            color=COLORS[0], edgecolor="white", label="Model",
        )
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.hist(
        max_probs, bins=bins, alpha=0.7,
        color=COLORS[3], edgecolor="white",
    )
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    _save_fig(fig, output_path, close=False)
    return fig


# ============================================================================
# Orchestrator class  (model-agnostic)
# ============================================================================

class ClassificationVisualizer:
    """High-level orchestrator for generating all classification visualizations.

    Works with any classifier — YOLO, ViT, ResNet, MedDef, custom CNN, etc.

    Parameters
    ----------
    save_dir : path to experiment directory
               (e.g. ``runs/classify/train/tbcr/full``)
    """

    def __init__(self, save_dir: Union[str, Path]):
        self.save_dir = Path(save_dir)
        self.viz_dir = self.save_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    # ------ individual plots ------ #

    def training_curves(
        self, csv_path: Optional[Union[str, Path]] = None,
        title: str = "Training Progress",
    ) -> Optional[plt.Figure]:
        csv_path = csv_path or self.save_dir / "results.csv"
        return plot_training_curves(
            csv_path, self.viz_dir / "training_curves.png", title=title,
        )

    def confusion_matrix(
        self, cm: np.ndarray, class_names: List[str],
        title: str = "Confusion Matrix", normalize: bool = False,
    ) -> plt.Figure:
        return plot_confusion_matrix(
            cm, class_names, self.viz_dir / "confusion_matrix.png",
            title=title, normalize=normalize,
        )

    def roc_auc(
        self, y_true: np.ndarray, y_probs: np.ndarray,
        class_names: List[str],
    ) -> plt.Figure:
        return plot_roc_auc(
            y_true, y_probs, class_names, self.viz_dir / "roc_auc.png",
        )

    def per_class_metrics(self, per_class: List[Dict]) -> plt.Figure:
        return plot_per_class_metrics(
            per_class, self.viz_dir / "per_class_metrics.png",
        )

    def radar_chart(
        self, models: Dict[str, Dict[str, float]],
        title: str = "Multi-Metric Comparison",
    ) -> plt.Figure:
        return plot_radar_chart(
            models, self.viz_dir / "radar_chart.png", title=title,
        )

    def tsne(
        self, features: np.ndarray, labels: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        return plot_tsne(
            features, labels, class_names, self.viz_dir / "tsne.png",
        )

    def pca(
        self, features: np.ndarray, labels: np.ndarray,
        class_names: Optional[List[str]] = None, n_components: int = 2,
    ) -> plt.Figure:
        return plot_pca(
            features, labels, class_names,
            self.viz_dir / "pca.png", n_components,
        )

    def epsilon_sensitivity(
        self, results: Dict[str, Dict[float, float]],
    ) -> plt.Figure:
        return plot_epsilon_sensitivity(
            results, self.viz_dir / "epsilon_sensitivity.png",
        )

    def saliency_map(
        self, image: np.ndarray, saliency: np.ndarray,
        title: str = "Saliency Map",
    ) -> plt.Figure:
        return plot_saliency_map(
            image, saliency, self.viz_dir / "saliency_map.png", title,
        )

    def ablation_heatmap(
        self, data: Dict[str, Dict[str, float]],
    ) -> plt.Figure:
        return plot_ablation_heatmap(
            data, self.viz_dir / "ablation_heatmap.png",
        )

    def asr_heatmap(
        self, data: Dict[str, Dict[str, float]],
    ) -> plt.Figure:
        return plot_asr_heatmap(
            data, self.viz_dir / "asr_heatmap.png",
        )

    def metrics_summary(self, metrics: Dict[str, Any]) -> plt.Figure:
        return plot_metrics_summary(
            metrics, self.viz_dir / "metrics_summary.png",
        )

    def calibration_diagram(
        self, y_true: np.ndarray, y_probs: np.ndarray,
    ) -> plt.Figure:
        return plot_calibration_diagram(
            y_true, y_probs, self.viz_dir / "calibration_diagram.png",
        )

    # ------ generate all ------ #

    def generate_all(
        self,
        metrics: Dict[str, Any],
        csv_path: Optional[Union[str, Path]] = None,
        y_true: Optional[np.ndarray] = None,
        y_probs: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """Generate all applicable visualizations.

        Returns dict of ``{plot_name: file_path}``.
        """
        generated: Dict[str, Path] = {}

        def _try(name: str, fn, *args, **kwargs):
            try:
                fig = fn(*args, **kwargs)
                if fig:
                    generated[name] = self.viz_dir / f"{name}.png"
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

        # 1. Training curves
        _try("training_curves", self.training_curves, csv_path)

        # 2. Confusion matrix
        cm = metrics.get("confusion_matrix")
        class_names = [c["class_name"] for c in metrics.get("per_class", [])]
        if cm is not None and class_names:
            _try("confusion_matrix", self.confusion_matrix, np.array(cm), class_names)
            # Also save normalized version
            _try(
                "confusion_matrix_norm", plot_confusion_matrix,
                np.array(cm), class_names,
                self.viz_dir / "confusion_matrix_norm.png",
                True, "Normalized Confusion Matrix",
            )

        # 3. Per-class metrics
        per_class = metrics.get("per_class", [])
        if per_class:
            _try("per_class_metrics", self.per_class_metrics, per_class)

        # 4. ROC/AUC
        if y_true is not None and y_probs is not None and class_names:
            _try("roc_auc", self.roc_auc, y_true, y_probs, class_names)

        # 5. t-SNE
        if features is not None and labels is not None:
            _try("tsne", self.tsne, features, labels, class_names or None)

        # 6. PCA
        if features is not None and labels is not None:
            _try("pca", self.pca, features, labels, class_names or None)

        # 7. Calibration diagram
        if y_true is not None and y_probs is not None:
            _try("calibration_diagram", self.calibration_diagram, y_true, y_probs)

        # 8. Metrics summary
        if metrics:
            _try("metrics_summary", self.metrics_summary, metrics)

        # 9. Run any registered custom plots
        for name, fn in _CUSTOM_PLOTS_REGISTRY.items():
            _try(name, fn, metrics, self.viz_dir / f"{name}.png")

        logger.info(f"Generated {len(generated)} visualizations → {self.viz_dir}")
        return generated


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
MedDefVisualizer = ClassificationVisualizer
