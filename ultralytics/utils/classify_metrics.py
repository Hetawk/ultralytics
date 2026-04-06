"""
Classification Metrics Engine
==============================

Model-agnostic, comprehensive classification metrics for any image
classification project. Computes standard + advanced metrics (MCC,
Cohen's Kappa, ECE, specificity, per-class stats, ROC-AUC, Brier
score, etc.) and saves results as CSV, TXT, and JSON.

This module is **not** tied to any specific model architecture.
It works with raw predictions (logits, probabilities, or class ids)
and ground-truth labels.

Usage:
    from ultralytics.utils.classify_metrics import ClassificationMetrics

    cm = ClassificationMetrics(class_names=["Normal", "Tuberculosis"])
    cm.update(preds_tensor, targets_tensor)        # call after each batch
    results = cm.compute()                         # aggregate
    cm.save(save_dir, epoch=100)                   # CSV + TXT + JSON

One-shot (no accumulation):
    from ultralytics.utils.classify_metrics import compute_metrics
    results = compute_metrics(y_true, y_pred, probs=probs, class_names=names)

Design:
    Follows the accumulator pattern — call ``update()`` per batch and
    ``compute()`` once at the end.  All heavy computation lives in
    ``compute()`` so that per-batch overhead is minimal.  No dependencies
    beyond numpy + torch.  sklearn is used only as an *optional* fallback
    for ROC-AUC and average precision.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert tensor / list / ndarray → numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_div(a, b, default: float = 0.0):
    """Element-wise a / b with fallback when b == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b != 0, a / b, default)
    return out


# ---------------------------------------------------------------------------
# Registry: allows extending with custom metric functions
# ---------------------------------------------------------------------------
# Users can register additional metric functions that take
# (y_true, y_pred, probs, class_names) and return a dict.
_CUSTOM_METRICS_REGISTRY: Dict[str, callable] = {}


def register_metric(name: str):
    """Decorator to register a custom metric function.

    Example::

        @register_metric("my_custom_metric")
        def my_metric(y_true, y_pred, probs=None, class_names=None):
            return {"my_score": float(...)}
    """
    def decorator(fn):
        _CUSTOM_METRICS_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Core metrics engine  (model-agnostic)
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """Comprehensive classification metrics accumulator.

    Works with **any** classifier — YOLO, ViT, ResNet, MedDef, custom CNN, etc.
    Accumulates predictions across batches, then computes all metrics at once.

    Parameters
    ----------
    class_names : list[str] | None
        Human-readable class names.  Auto-generated if ``None``.
    top_k : tuple[int, ...]
        Which top-k accuracies to track.  Default ``(1, 5)``.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        top_k: Tuple[int, ...] = (1, 5),
    ):
        self.class_names = list(class_names) if class_names else None
        self.top_k = top_k
        self._preds: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._probs: List[np.ndarray] = []
        self._top_k_correct: Dict[int, int] = {k: 0 for k in top_k}
        self._n_samples: int = 0

    # ------------------------------------------------------------------ #
    #  Accumulation
    # ------------------------------------------------------------------ #

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of predictions.

        Parameters
        ----------
        preds : Tensor
            Shape ``(B, C)`` logits / softmax **or** ``(B,)`` class ids.
        targets : Tensor
            Shape ``(B,)`` ground-truth class ids.
        probs : Tensor | None
            ``(B, C)`` softmax probabilities.  Computed from *preds* if
            ``None`` and *preds* is 2-D.
        """
        targets_np = _to_numpy(targets).astype(int).ravel()

        if preds.ndim == 2:
            # logits / probs → derive class ids + top-k
            if probs is None:
                probs_np = torch.softmax(preds.float(), dim=1).detach().cpu().numpy()
            else:
                probs_np = _to_numpy(probs)
            pred_ids = preds.detach().cpu().argsort(dim=1, descending=True)
            for k in self.top_k:
                topk = pred_ids[:, :k].numpy()
                self._top_k_correct[k] += int(
                    np.any(topk == targets_np[:, None], axis=1).sum()
                )
            preds_np = pred_ids[:, 0].numpy().ravel()
            self._probs.append(probs_np)
        else:
            preds_np = _to_numpy(preds).astype(int).ravel()

        self._preds.append(preds_np)
        self._targets.append(targets_np)
        self._n_samples += len(targets_np)

    def reset(self) -> None:
        """Clear accumulated data."""
        self._preds.clear()
        self._targets.clear()
        self._probs.clear()
        self._top_k_correct = {k: 0 for k in self.top_k}
        self._n_samples = 0

    # ------------------------------------------------------------------ #
    #  Computation
    # ------------------------------------------------------------------ #

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from accumulated data.

        Returns
        -------
        dict
            Keys include: accuracy, top1_acc, top5_acc, precision_macro,
            recall_macro, f1_macro, specificity, mcc, cohen_kappa,
            balanced_accuracy, confusion_matrix, per_class, roc_auc,
            log_loss, brier_score, ece, plus any registered custom metrics.
        """
        if not self._targets:
            return {}

        y_true = np.concatenate(self._targets)
        y_pred = np.concatenate(self._preds)

        classes = np.unique(np.concatenate([y_true, y_pred]))
        nc = int(classes.max()) + 1 if len(classes) else 0
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(nc)]

        # Confusion matrix
        cm = self._confusion_matrix(y_true, y_pred, nc)

        # Derived quantities
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        specificity = _safe_div(tn, tn + fp)
        support = (tp + fn).astype(int)

        # Macro / weighted averages
        weights = support / max(support.sum(), 1)
        acc = float(tp.sum() / max(len(y_true), 1))

        # Micro averages
        micro_precision = float(tp.sum() / max(tp.sum() + fp.sum(), 1))
        micro_recall = float(tp.sum() / max(tp.sum() + fn.sum(), 1))
        micro_f1 = float(
            2 * micro_precision * micro_recall
            / max(micro_precision + micro_recall, 1e-10)
        )

        results: Dict[str, Any] = {
            # Top-k
            **{
                f"top{k}_acc": self._top_k_correct[k] / max(self._n_samples, 1)
                for k in self.top_k
            },
            # Core
            "accuracy": acc,
            "precision_macro": float(precision.mean()),
            "precision_micro": micro_precision,
            "precision_weighted": float((precision * weights).sum()),
            "recall_macro": float(recall.mean()),
            "recall_micro": micro_recall,
            "recall_weighted": float((recall * weights).sum()),
            "f1_macro": float(f1.mean()),
            "f1_micro": micro_f1,
            "f1_weighted": float((f1 * weights).sum()),
            "specificity_macro": float(specificity.mean()),
            "balanced_accuracy": float(recall.mean()),
            # Advanced
            "mcc": self._matthews_corrcoef(cm),
            "cohen_kappa": self._cohen_kappa(cm, len(y_true)),
            # Structure
            "confusion_matrix": cm.tolist(),
            "num_classes": nc,
            "num_samples": int(len(y_true)),
        }

        # Per-class breakdown
        per_class = []
        for i in range(nc):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            total_i = int(tp[i] + tn[i] + fp[i] + fn[i])
            acc_i = float((tp[i] + tn[i]) / max(total_i, 1))
            per_class.append({
                "class_id": i,
                "class_name": name,
                "accuracy": acc_i,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
                "support": int(support[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
            })
        results["per_class"] = per_class

        # Probability-based metrics
        if self._probs:
            all_probs = np.concatenate(self._probs, axis=0)
            results.update(self._prob_metrics(y_true, all_probs, nc))

        # Run registered custom metrics
        for name, fn in _CUSTOM_METRICS_REGISTRY.items():
            try:
                probs_arr = np.concatenate(self._probs) if self._probs else None
                custom = fn(y_true, y_pred, probs=probs_arr, class_names=self.class_names)
                if isinstance(custom, dict):
                    results.update(custom)
            except Exception as e:
                logger.warning(f"Custom metric '{name}' failed: {e}")

        return results

    # ------------------------------------------------------------------ #
    #  Saving
    # ------------------------------------------------------------------ #

    def save(
        self,
        save_dir: Union[str, Path],
        epoch: Optional[int] = None,
        prefix: str = "",
        report_title: str = "Classification Evaluation Report",
    ) -> Tuple[Path, Path, Path]:
        """Save metrics to CSV, TXT, and JSON.

        Returns ``(csv_path, txt_path, json_path)``.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{prefix}_" if prefix else ""

        results = self.compute()
        if not results:
            return None, None, None

        csv_path = save_dir / f"{tag}evaluation_metrics.csv"
        txt_path = save_dir / f"{tag}detailed_metrics.txt"
        json_path = save_dir / f"{tag}metrics.json"

        self._save_csv(results, csv_path, epoch)
        self._save_txt(results, txt_path, epoch, report_title)
        self._save_json(results, json_path, epoch)

        logger.info(f"Metrics saved → {save_dir}")
        return csv_path, txt_path, json_path

    # ------------------------------------------------------------------ #
    #  Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, nc: int) -> np.ndarray:
        cm = np.zeros((nc, nc), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < nc and 0 <= p < nc:
                cm[int(t), int(p)] += 1
        return cm

    @staticmethod
    def _matthews_corrcoef(cm: np.ndarray) -> float:
        """Multi-class MCC (numerically stable)."""
        t_sum = cm.sum(axis=1).astype(float)
        p_sum = cm.sum(axis=0).astype(float)
        n = float(cm.sum())
        cov_yy = float(np.dot(t_sum, p_sum))
        cov_pp = float(np.dot(p_sum, p_sum))
        cov_tt = float(np.dot(t_sum, t_sum))
        trace = float(np.trace(cm))
        num = trace * n - cov_yy
        den = np.sqrt((n * n - cov_pp) * (n * n - cov_tt))
        return float(num / den) if den > 0 else 0.0

    @staticmethod
    def _cohen_kappa(cm: np.ndarray, n: int) -> float:
        """Cohen's Kappa from confusion matrix."""
        if n == 0:
            return 0.0
        p_o = float(np.trace(cm)) / n
        t_sum = cm.sum(axis=1).astype(float) / n
        p_sum = cm.sum(axis=0).astype(float) / n
        p_e = float(np.dot(t_sum, p_sum))
        denom = 1.0 - p_e
        return float((p_o - p_e) / denom) if denom > 0 else 0.0

    @staticmethod
    def _prob_metrics(y_true: np.ndarray, probs: np.ndarray, nc: int) -> dict:
        """Compute probability-based metrics: ROC-AUC, log-loss, Brier, ECE."""
        results = {}

        # One-hot encode
        y_onehot = np.zeros((len(y_true), nc), dtype=float)
        for i, cls in enumerate(y_true):
            if 0 <= cls < nc:
                y_onehot[i, cls] = 1.0

        # Log-loss
        eps = 1e-12
        clipped = np.clip(probs, eps, 1 - eps)
        results["log_loss"] = float(
            -np.mean(np.sum(y_onehot * np.log(clipped), axis=1))
        )

        # Brier score (multi-class extension)
        results["brier_score"] = float(
            np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        )

        # Expected Calibration Error (ECE) — 10 bins
        max_probs = probs.max(axis=1)
        pred_classes = probs.argmax(axis=1)
        correct = (pred_classes == y_true).astype(float)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            mask = (max_probs > lo) & (max_probs <= hi)
            if mask.sum() > 0:
                avg_conf = max_probs[mask].mean()
                avg_acc = correct[mask].mean()
                ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
        results["ece"] = float(ece)

        # ROC-AUC (one-vs-rest, macro)  — sklearn optional
        try:
            if nc == 2:
                from sklearn.metrics import roc_auc_score
                results["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            elif nc > 2:
                from sklearn.metrics import roc_auc_score
                results["roc_auc"] = float(
                    roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
                )
        except Exception:
            results["roc_auc"] = None

        # Average precision (macro)
        try:
            from sklearn.metrics import average_precision_score
            results["average_precision"] = float(
                average_precision_score(y_onehot, probs, average="macro")
            )
        except Exception:
            results["average_precision"] = None

        # Per-class ROC AUC
        per_class_auc = []
        try:
            from sklearn.metrics import roc_auc_score as _roc_auc
            for i in range(nc):
                if y_onehot[:, i].sum() > 0 and y_onehot[:, i].sum() < len(y_true):
                    per_class_auc.append(float(_roc_auc(y_onehot[:, i], probs[:, i])))
                else:
                    per_class_auc.append(None)
        except Exception:
            per_class_auc = [None] * nc
        results["per_class_roc_auc"] = per_class_auc

        return results

    # ------------------------------------------------------------------ #
    #  File writers
    # ------------------------------------------------------------------ #

    def _save_csv(self, results: dict, path: Path, epoch: Optional[int]) -> None:
        """Write flat CSV with one row of aggregate metrics.

        Column order is organised by clinical relevance for medical
        image classification tasks.
        """
        flat = {
            "epoch": epoch if epoch is not None else "",
            # Primary performance
            "accuracy": results["accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
            "f1_weighted": results["f1_weighted"],
            "f1_macro": results["f1_macro"],
            "f1_micro": results["f1_micro"],
            # Recall / Specificity (critical for medical)
            "recall_weighted": results["recall_weighted"],
            "recall_macro": results["recall_macro"],
            "recall_micro": results["recall_micro"],
            "specificity_macro": results["specificity_macro"],
            # Precision
            "precision_weighted": results["precision_weighted"],
            "precision_macro": results["precision_macro"],
            "precision_micro": results["precision_micro"],
            # Agreement / correlation
            "mcc": results["mcc"],
            "cohen_kappa": results["cohen_kappa"],
            # Probabilistic / calibration
            "roc_auc": results.get("roc_auc", ""),
            "average_precision": results.get("average_precision", ""),
            "log_loss": results.get("log_loss", ""),
            "brier_score": results.get("brier_score", ""),
            "ece": results.get("ece", ""),
            # Dataset info
            "num_classes": results["num_classes"],
            "num_samples": results["num_samples"],
        }
        header = ",".join(flat.keys())
        values = ",".join(str(v) for v in flat.values())
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(values + "\n")

    def _save_txt(
        self, results: dict, path: Path, epoch: Optional[int],
        title: str = "Classification Evaluation Report",
    ) -> None:
        """Human-readable detailed metrics report."""
        lines = []
        w = 30

        lines.append("=" * 70)
        lines.append(f"  {title}")
        if epoch is not None:
            lines.append(f"  Epoch: {epoch}")
        lines.append(
            f"  Samples: {results['num_samples']}  |  "
            f"Classes: {results['num_classes']}"
        )
        lines.append("=" * 70)

        lines.append("\n── Primary Performance ──")
        for key, label in [
            ("accuracy",           "Accuracy"),
            ("balanced_accuracy",  "Balanced Accuracy"),
            ("f1_weighted",        "F1 Score (weighted)"),
            ("f1_macro",           "F1 Score (macro)"),
            ("f1_micro",           "F1 Score (micro)"),
        ]:
            val = results.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, float):
                lines.append(f"  {label:<{w}} {val:.6f}")
            else:
                lines.append(f"  {label:<{w}} {val}")

        lines.append("\n── Recall & Specificity (clinical sensitivity) ──")
        for key, label in [
            ("recall_weighted",    "Recall (weighted)"),
            ("recall_macro",       "Recall (macro)"),
            ("recall_micro",       "Recall (micro)"),
            ("specificity_macro",  "Specificity (macro)"),
            ("precision_weighted", "Precision (weighted)"),
            ("precision_macro",    "Precision (macro)"),
            ("precision_micro",    "Precision (micro)"),
        ]:
            val = results.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, float):
                lines.append(f"  {label:<{w}} {val:.6f}")
            else:
                lines.append(f"  {label:<{w}} {val}")

        lines.append("\n── Agreement & Correlation ──")
        for key, label in [
            ("mcc",          "MCC"),
            ("cohen_kappa",  "Cohen's Kappa"),
        ]:
            val = results.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, float):
                lines.append(f"  {label:<{w}} {val:.6f}")
            else:
                lines.append(f"  {label:<{w}} {val}")

        lines.append("\n── Probabilistic & Calibration ──")
        for key, label in [
            ("roc_auc",            "ROC AUC"),
            ("average_precision",  "Average Precision"),
            ("log_loss",           "Log Loss"),
            ("brier_score",        "Brier Score"),
            ("ece",                "ECE (calibration)"),
        ]:
            val = results.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, float):
                lines.append(f"  {label:<{w}} {val:.6f}")
            else:
                lines.append(f"  {label:<{w}} {val}")

        lines.append("\n── Per-Class Breakdown ──")
        header_line = (
            f"  {'Class':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
            f"{'F1':>8} {'Spec':>8} {'Supp':>8}"
        )
        lines.append(header_line)
        lines.append("  " + "-" * 68)
        for c in results.get("per_class", []):
            name = c["class_name"][:20]
            lines.append(
                f"  {name:<20} {c['accuracy']:>8.4f} {c['precision']:>8.4f} {c['recall']:>8.4f} "
                f"{c['f1']:>8.4f} {c['specificity']:>8.4f} {c['support']:>8d}"
            )

        lines.append("\n── Confusion Matrix ──")
        cm = results.get("confusion_matrix", [])
        if cm:
            nc = len(cm)
            names = [
                self.class_names[i][:12]
                if i < len(self.class_names)
                else f"C{i}"
                for i in range(nc)
            ]
            lines.append(
                "  " + " " * 14 + "  ".join(f"{n:>12}" for n in names)
            )
            for i, row in enumerate(cm):
                line = (
                    f"  {names[i]:>12}  "
                    + "  ".join(f"{v:>12d}" for v in row)
                )
                lines.append(line)

        lines.append("\n" + "=" * 70)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _save_json(
        self, results: dict, path: Path, epoch: Optional[int]
    ) -> None:
        """Machine-readable JSON dump."""
        data = {**results}
        if epoch is not None:
            data["epoch"] = epoch
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Convenience: one-shot compute from raw arrays
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One-shot metric computation — no accumulation needed.

    Parameters
    ----------
    y_true : (N,) ground-truth class ids
    y_pred : (N,) predicted class ids
    probs : (N, C) predicted probabilities (optional)
    class_names : list of class labels (optional)
    """
    cm = ClassificationMetrics(class_names=class_names)
    if probs is not None:
        cm.update(
            preds=torch.from_numpy(probs),
            targets=torch.from_numpy(y_true.astype(np.int64)),
            probs=torch.from_numpy(probs),
        )
    else:
        cm.update(
            preds=torch.from_numpy(y_pred.astype(np.int64)),
            targets=torch.from_numpy(y_true.astype(np.int64)),
        )
    return cm.compute()


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
MedDefMetrics = ClassificationMetrics


# ---------------------------------------------------------------------------
# Robustness-specific metrics (registered via the registry system)
# ---------------------------------------------------------------------------

@register_metric("robustness_degradation")
def _robustness_degradation(y_true, y_pred, probs=None, class_names=None,
                            clean_preds=None, **kwargs):
    """Compute per-class accuracy degradation under adversarial attack.

    This metric is designed to be called with ``clean_preds`` injected via
    ``_CUSTOM_METRICS_REGISTRY`` after an adversarial evaluation run.
    When ``clean_preds`` is ``None`` it returns an empty dict (no-op during
    standard evaluation).

    Returns
    -------
    dict with keys:
        accuracy_drop : overall accuracy drop (clean − adversarial)
        per_class_accuracy_drop : list of per-class drops
        attack_success_rate : fraction of correctly-classified clean samples
            that became misclassified
        robustness_ratio : robust_accuracy / clean_accuracy
    """
    if clean_preds is None:
        return {}

    clean_preds = np.asarray(clean_preds).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    clean_correct = clean_preds == y_true
    adv_correct = y_pred == y_true

    clean_acc = clean_correct.mean()
    adv_acc = adv_correct.mean()

    # Samples that were correct under clean but wrong under attack
    flipped = clean_correct & ~adv_correct
    asr = float(flipped.sum() / max(clean_correct.sum(), 1))

    # Per-class breakdown
    n_classes = int(max(y_true.max(), y_pred.max(), clean_preds.max()) + 1)
    per_class_drop = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            per_class_drop.append(0.0)
            continue
        c_clean = float(clean_correct[mask].mean())
        c_adv = float(adv_correct[mask].mean())
        per_class_drop.append(c_clean - c_adv)

    return {
        "accuracy_drop": float(clean_acc - adv_acc),
        "attack_success_rate": asr,
        "robustness_ratio": float(adv_acc / max(clean_acc, 1e-8)),
        "per_class_accuracy_drop": per_class_drop,
    }


def compute_robustness_metrics(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    y_true: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Standalone robustness metric computation.

    Computes the full suite of adversarial robustness statistics without
    needing the full ``ClassificationMetrics`` pipeline.

    Parameters
    ----------
    clean_preds : (N,) predicted class ids on clean inputs
    adv_preds   : (N,) predicted class ids on adversarial inputs
    y_true      : (N,) ground-truth labels
    class_names : optional list of class labels

    Returns
    -------
    dict with comprehensive robustness metrics
    """
    clean_preds = np.asarray(clean_preds).ravel()
    adv_preds = np.asarray(adv_preds).ravel()
    y_true = np.asarray(y_true).ravel()

    clean_correct = clean_preds == y_true
    adv_correct = adv_preds == y_true

    clean_acc = float(clean_correct.mean())
    robust_acc = float(adv_correct.mean())

    # Attack success rate: fraction of correctly-classified clean that flipped
    flipped = clean_correct & ~adv_correct
    n_clean_correct = int(clean_correct.sum())
    asr = float(flipped.sum() / max(n_clean_correct, 1))

    # Per-class
    n_classes = int(max(y_true.max(), adv_preds.max(), clean_preds.max()) + 1)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    per_class = []
    for c in range(n_classes):
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            per_class.append({
                "class_name": class_names[c] if c < len(class_names) else f"class_{c}",
                "clean_accuracy": 0.0, "robust_accuracy": 0.0,
                "accuracy_drop": 0.0, "attack_success_rate": 0.0, "support": 0,
            })
            continue
        c_clean_acc = float(clean_correct[mask].mean())
        c_robust_acc = float(adv_correct[mask].mean())
        c_flipped = int((clean_correct[mask] & ~adv_correct[mask]).sum())
        c_clean_ok = int(clean_correct[mask].sum())
        per_class.append({
            "class_name": class_names[c] if c < len(class_names) else f"class_{c}",
            "clean_accuracy": c_clean_acc,
            "robust_accuracy": c_robust_acc,
            "accuracy_drop": c_clean_acc - c_robust_acc,
            "attack_success_rate": float(c_flipped / max(c_clean_ok, 1)),
            "support": n,
        })

    # Class-level vulnerability: which classes are most vulnerable?
    drops = [p["accuracy_drop"] for p in per_class if p["support"] > 0]
    most_vulnerable_idx = int(np.argmax(drops)) if drops else -1

    return {
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "accuracy_drop": clean_acc - robust_acc,
        "attack_success_rate": asr,
        "robustness_ratio": robust_acc / max(clean_acc, 1e-8),
        "num_samples": len(y_true),
        "num_flipped": int(flipped.sum()),
        "num_clean_correct": n_clean_correct,
        "per_class": per_class,
        "most_vulnerable_class": (
            class_names[most_vulnerable_idx]
            if 0 <= most_vulnerable_idx < len(class_names) else "N/A"
        ),
        "max_per_class_drop": float(max(drops)) if drops else 0.0,
    }
