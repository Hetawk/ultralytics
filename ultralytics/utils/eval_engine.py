"""
Evaluation Engine — Shared Task Functions
==========================================

Reusable building-blocks for classification model evaluation,
visualization, saliency-map generation, embedding extraction,
variant comparison, adversarial robustness testing, and model export.

**Not** tied to any specific model architecture — works with YOLO,
MedDef, ViT, ResNet, any ``nn.Module`` classifier.

Every public function is importable by any CLI or notebook.

Usage::

    from ultralytics.utils.eval_engine import (
        run_evaluation, run_visualization, run_comparison,
        run_saliency, run_embeddings, run_robustness,
        run_export, print_results,
    )
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. Path / directory utilities
# ══════════════════════════════════════════════════════════════════════

def resolve_save_dir(
    model_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Determine where to write evaluation artefacts.

    Priority: ``output_dir`` → parent of ``model_path``'s weights dir.
    """
    if output_dir:
        p = Path(output_dir)
    elif model_path:
        p = Path(model_path).parent.parent  # weights/best.pt → experiment/
    else:
        p = Path(".")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_metrics_json(results_dir: Path) -> dict:
    """Load the first ``*metrics.json`` found in *results_dir*."""
    for candidate in sorted(results_dir.glob("*metrics.json")):
        with open(candidate) as f:
            return json.load(f)
    return {}


def _get_class_names(model) -> List[str]:
    """Extract class names from model ``names`` attribute."""
    if not hasattr(model, "names"):
        return []
    names = model.names
    return list(names.values()) if isinstance(names, dict) else list(names)


def _find_val_dir(data_path: Union[str, Path], split: str = "val") -> Optional[Path]:
    """Return the first existing split directory under *data_path*."""
    data_dir = Path(data_path)
    for candidate_name in [split, "val", "test"]:
        d = data_dir / candidate_name
        if d.exists():
            return d
    return None


def _collect_image_paths(
    val_dir: Path,
    n: Optional[int] = None,
    per_class: int = 0,
) -> Tuple[List[Path], List[str]]:
    """Collect image paths from a directory of class folders.

    Returns ``(image_paths, class_names)``.
    """
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    paths: list = []
    for cls_dir in class_dirs:
        imgs = sorted(cls_dir.glob("*.jpg")) + \
            sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
        if per_class > 0:
            imgs = imgs[:per_class]
        paths.extend(imgs)
        if n and len(paths) >= n:
            break
    if n:
        paths = paths[:n]
    return paths, class_names


def _default_transform(imgsz: int = 224):
    """Return the standard ImageNet-normalised eval transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════
# 2. Core evaluation
# ══════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_path: str,
    data: str,
    save_dir: Optional[str] = None,
    split: str = "val",
    batch: int = 32,
    imgsz: int = 224,
    device: str = "0",
    workers: int = 8,
    verbose: bool = False,
) -> Tuple[dict, Any, Path]:
    """Run inference + comprehensive metrics on a classification model.

    Tries, in order:
        1. ``MedDefValidator`` (MedDef models)
        2. ``YOLO.val()`` (standard Ultralytics)
        3. Manual image-level inference (fallback)

    Returns
    -------
    (metrics_dict, validator_or_None, save_dir)
    """
    out = resolve_save_dir(model_path, save_dir)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=" * 60)
    print("Classification Model Evaluation")
    print("=" * 60)
    print(f"  Model:  {model_path}")
    print(f"  Data:   {data}")
    print(f"  Split:  {split}")
    print(f"  Output: {out}")
    print("=" * 60)

    enhanced = None
    validator = None

    # --- Strategy 1: MedDef validator ---
    try:
        from ultralytics.models.meddef.val import MedDefValidator
        from types import SimpleNamespace

        val_args = SimpleNamespace(
            task="classify", data=data, batch=batch,
            imgsz=imgsz, device=device, workers=workers,
            split=split, plots=True, half=False,
        )
        validator = MedDefValidator(save_dir=out, args=val_args)
        validator(model=str(model_path))

        print("\nComputing enhanced metrics...")
        enhanced = validator.compute_enhanced_metrics(epoch=None)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            # CUDA OOM on Strategy 1 — free memory and re-raise so caller can handle
            validator = None
            torch.cuda.empty_cache()
            raise
        if verbose:
            print(f"  MedDef validator (RuntimeError): {exc}")
        validator = None
        torch.cuda.empty_cache()
    except Exception as exc:
        if verbose:
            print(f"  MedDef validator: {exc}")
        # Free any GPU state the failed validator may have allocated
        validator = None
        torch.cuda.empty_cache()

    # --- Strategy 2: YOLO validator ---
    if enhanced is None:
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            model.val(
                data=data, batch=batch, imgsz=imgsz,
                device=device, workers=workers, split=split,
                plots=True, half=False,
            )
            print("\nComputing metrics from YOLO validation...")
            enhanced = _yolo_metrics(model, out)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise
            if verbose:
                print(f"  YOLO validator (RuntimeError): {exc}")
            torch.cuda.empty_cache()
        except Exception as exc:
            if verbose:
                print(f"  YOLO validator: {exc}")
            torch.cuda.empty_cache()

    # --- Strategy 3: manual inference ---
    if enhanced is None:
        print("\nFalling back to manual inference...")
        enhanced = _manual_evaluate(model_path, data, out,
                                    split=split, imgsz=imgsz, device=device)

    print_results(enhanced)
    return enhanced, validator, out


def _yolo_metrics(model, save_dir: Path) -> dict:
    """Extract metrics after ``model.val()`` via ClassificationMetrics."""
    from ultralytics.utils.classify_metrics import ClassificationMetrics
    cm = ClassificationMetrics(class_names=_get_class_names(model))
    results = cm.compute()
    cm.save(save_dir)
    return results


def _manual_evaluate(
    model_path: Path,
    data: str,
    save_dir: Path,
    split: str = "val",
    imgsz: int = 224,
    device: str = "0",
) -> dict:
    """Image-by-image inference — works with any ``nn.Module``."""
    from ultralytics.utils.classify_metrics import ClassificationMetrics
    from ultralytics.nn.tasks import load_checkpoint
    from PIL import Image

    dev = f"cuda:{device}"
    model, _ = load_checkpoint(str(model_path), device=dev)
    model.eval()

    val_dir = _find_val_dir(data, split)
    if val_dir is None:
        print(f"  WARNING: no {split}/val/test directory in {data}")
        return {}

    img_paths, class_names = _collect_image_paths(val_dir)
    if not img_paths:
        print("  WARNING: no images found")
        return {}

    transform = _default_transform(imgsz)
    cm = ClassificationMetrics(class_names=class_names)

    all_logits, all_targets = [], []
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    with torch.no_grad():
        for cls_idx, cls_dir in enumerate(class_dirs):
            imgs = (
                list(cls_dir.glob("*.jpg"))
                + list(cls_dir.glob("*.jpeg"))
                + list(cls_dir.glob("*.png"))
            )
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img).unsqueeze(0).to(dev)
                    out = model(x)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    all_logits.append(out.cpu())
                    all_targets.append(cls_idx)
                except Exception:
                    continue

    if not all_targets:
        return {}

    targets = torch.tensor(all_targets, dtype=torch.long)
    preds = torch.cat(all_logits)
    cm.update(preds=preds, targets=targets)
    results = cm.compute()
    cm.save(save_dir, report_title="Classification Evaluation Report")
    return results


# ══════════════════════════════════════════════════════════════════════
# 3. Visualization from existing results
# ══════════════════════════════════════════════════════════════════════

def run_visualization(
    results_dir: Union[str, Path],
    metrics: Optional[dict] = None,
    y_true: Optional[np.ndarray] = None,
    y_probs: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Generate all applicable visualizations in *results_dir*.

    If *metrics* is ``None``, loads ``*metrics.json`` from the directory.
    """
    from ultralytics.utils.classify_visualize import ClassificationVisualizer
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    if metrics is None:
        metrics = _load_metrics_json(results_dir)

    print("=" * 60)
    print("Generating Visualizations")
    print(f"  Source: {results_dir}")
    print("=" * 60)

    viz = ClassificationVisualizer(results_dir)
    csv_path = results_dir / \
        "results.csv" if (results_dir / "results.csv").exists() else None

    generated = viz.generate_all(
        metrics,
        csv_path=csv_path,
        y_true=y_true,
        y_probs=y_probs,
        features=features,
        labels=labels,
    )
    plt.close("all")
    print(f"\nVisualizations saved -> {viz.viz_dir}")
    return generated


# ══════════════════════════════════════════════════════════════════════
# 4. Variant comparison (ablation study)
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_COMPARE_KEYS = [
    "accuracy", "f1_macro", "precision_macro", "recall_macro",
    "mcc", "cohen_kappa", "specificity_macro", "balanced_accuracy",
]


def run_comparison(
    results_dir: Union[str, Path],
    variants: Optional[List[str]] = None,
    compare_keys: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Compare ablation variants side-by-side (radar + heatmap + CSV).

    *variants* are auto-detected from subdirectories if ``None``.

    Returns the collected per-variant metrics dict.
    """
    from ultralytics.utils.classify_visualize import (
        plot_ablation_heatmap,
        plot_radar_chart,
    )
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    dataset_name = results_dir.name
    compare_keys = compare_keys or _DEFAULT_COMPARE_KEYS

    print("=" * 60)
    print(f"Comparing Variants — {dataset_name}")
    print("=" * 60)

    # Auto-detect variants
    if variants is None:
        variants = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and any((d / n).exists() for n in [
                "metrics.json", "evaluation_metrics.json",
            ])
        ])

    # Collect metrics
    all_metrics: Dict[str, dict] = {}
    for variant in variants:
        for json_name in ["metrics.json", "evaluation_metrics.json"]:
            jp = results_dir / variant / json_name
            if jp.exists():
                with open(jp) as f:
                    all_metrics[variant] = json.load(f)
                print(f"  Loaded: {variant}")
                break

    if len(all_metrics) < 2:
        print(f"  Only {len(all_metrics)} variant(s) found — need >= 2.")
        return all_metrics

    radar_data = {v: {k: m.get(k, 0) for k in compare_keys}
                  for v, m in all_metrics.items()}

    compare_dir = results_dir / "comparison"
    compare_dir.mkdir(exist_ok=True)

    # Radar chart
    print("  Generating radar comparison...")
    fig = plot_radar_chart(
        radar_data, compare_dir / "ablation_radar.png",
        title=f"Variant Comparison — {dataset_name}",
    )
    if fig:
        plt.close(fig)

    # Heatmap
    print("  Generating ablation heatmap...")
    fig = plot_ablation_heatmap(
        radar_data, compare_dir / "ablation_heatmap.png",
        title=f"Ablation Study — {dataset_name}",
    )
    if fig:
        plt.close(fig)

    # CSV
    csv_path = compare_dir / "ablation_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("variant," + ",".join(compare_keys) + "\n")
        for variant, m in all_metrics.items():
            vals = ",".join(str(m.get(k, 0)) for k in compare_keys)
            f.write(f"{variant},{vals}\n")

    print(f"\nComparison saved -> {compare_dir}")
    return all_metrics


# ══════════════════════════════════════════════════════════════════════
# 5. Saliency maps
# ══════════════════════════════════════════════════════════════════════

def run_saliency(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    save_dir: Union[str, Path],
    n_samples: int = 8,
    device: str = "0",
    imgsz: int = 224,
) -> List[Path]:
    """Generate comprehensive saliency maps for sample images.

    Returns list of saved file paths.
    """
    from ultralytics.utils.saliency import SaliencyGenerator
    from ultralytics.nn.tasks import load_checkpoint

    print("\nGenerating comprehensive saliency maps...")

    dev = f"cuda:{device}"
    model, _ = load_checkpoint(str(model_path), device=dev)
    model.eval()

    val_dir = _find_val_dir(data_path)
    if val_dir is None:
        print("  WARNING: no val/test directory found")
        return []

    sample_paths, class_names = _collect_image_paths(
        val_dir, n=n_samples, per_class=1)
    if not sample_paths:
        print("  No images found")
        return []

    saliency_dir = Path(save_dir) / "visualizations" / "saliency"
    gen = SaliencyGenerator(model, device=dev, output_dir=saliency_dir)
    transform = _default_transform(imgsz)

    saved = gen.generate_batch(
        sample_paths, transform, class_names=class_names,
        save_prefix="saliency",
    )
    print(f"  {len(saved)} saliency maps saved -> {saliency_dir}")
    return saved


# ══════════════════════════════════════════════════════════════════════
# 5b. Adversarial robustness evaluation
# ══════════════════════════════════════════════════════════════════════

# Default perturbation budgets (ε values) for multi-epsilon sweeps
_DEFAULT_EPSILONS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]

# Attacks that don't need ART (pure PyTorch)
_NATIVE_ATTACKS = {"fgsm", "pgd", "cw", "bim", "mim"}
# Attacks that require ART
_ART_ATTACKS = {"deepfool", "apgd", "square"}


def run_robustness(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    save_dir: Union[str, Path],
    attacks: Optional[List[str]] = None,
    epsilons: Optional[List[float]] = None,
    device: str = "0",
    imgsz: int = 224,
    batch: int = 32,
    workers: int = 4,
    certified: bool = False,
    certified_sigma: float = 0.25,
    certified_n: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run comprehensive adversarial robustness evaluation.

    Wraps ``defense.RobustnessEvaluator`` and ``CertifiedDefense`` into
    the shared eval pipeline.  Results are saved to JSON and optionally
    visualized (epsilon-sensitivity curves + ASR heatmap).

    Parameters
    ----------
    model_path : path to trained weights (.pt)
    data_path  : root of classification dataset
    save_dir   : output directory for results
    attacks    : list of attack names (default: fgsm, pgd, bim)
    epsilons   : perturbation budgets for multi-epsilon sweep (default: 10-point)
    device     : CUDA device id
    imgsz      : input image size
    batch      : batch size for dataloader
    workers    : dataloader workers
    certified  : run randomized-smoothing certification
    certified_sigma : noise σ for randomized smoothing
    certified_n : number of noise samples per input
    verbose     : print extra debug info

    Returns
    -------
    dict with keys: per_attack, epsilon_sweep, certified (optional), summary
    """
    from ultralytics.utils.defense import RobustnessEvaluator, CertifiedDefense
    from ultralytics.nn.tasks import load_checkpoint
    from torch.utils.data import DataLoader

    save_dir = Path(save_dir)
    robustness_dir = save_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)

    attacks = attacks or ["fgsm", "pgd", "bim", "mim", "cw",
                          "deepfool", "apgd", "square"]
    epsilons = epsilons or _DEFAULT_EPSILONS

    print("\n" + "=" * 60)
    print("Adversarial Robustness Evaluation")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Attacks: {', '.join(attacks)}")
    print(f"  Epsilons: {epsilons}")
    print(f"  Device:  cuda:{device}")
    print("=" * 60)

    # --- Load model ---
    dev = torch.device(
        f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), device=str(dev))
    model.eval()

    # --- Build dataloader ---
    val_dir = _find_val_dir(data_path)
    if val_dir is None:
        print(f"  WARNING: no val/test directory in {data_path}")
        return {}

    loader = _build_robustness_loader(
        val_dir, imgsz=imgsz, batch=batch, workers=workers)
    if loader is None:
        return {}

    # Detect number of classes
    class_names = _get_class_names(model)
    nb_classes = len(
        class_names) if class_names else _detect_nb_classes(val_dir)

    evaluator = RobustnessEvaluator(model, device=dev)
    results: Dict[str, Any] = {"per_attack": {},
                               "epsilon_sweep": {}, "summary": {}}

    # ---- Per-attack evaluation (default epsilon) ----
    print("\n── Per-Attack Evaluation (default ε) ──")
    for atk_name in attacks:
        try:
            atk_kwargs = _attack_kwargs(
                atk_name, nb_classes=nb_classes, imgsz=imgsz)
            res = evaluator.evaluate(
                loader, attack_name=atk_name, attack_kwargs=atk_kwargs)
            results["per_attack"][atk_name] = res
            drop = res["clean_accuracy"] - res["robust_accuracy"]
            asr = 100.0 * drop / max(res["clean_accuracy"], 1e-8)
            res["accuracy_drop"] = drop
            res["attack_success_rate"] = asr
            print(f"  {atk_name:>10}: clean={res['clean_accuracy']:.2f}%  "
                  f"robust={res['robust_accuracy']:.2f}%  "
                  f"ASR={asr:.2f}%  drop={drop:.2f}pp")
        except Exception as e:
            print(f"  {atk_name:>10}: FAILED — {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    # ---- Multi-epsilon sweep (key robustness metric) ----
    print("\n── Epsilon Sensitivity Sweep ──")
    for atk_name in attacks:
        if atk_name in _ART_ATTACKS:
            # ART attacks are expensive; skip sweep unless few epsilons
            if len(epsilons) > 5:
                print(
                    f"  {atk_name}: skipping sweep (ART attack, too many ε values)")
                continue
        sweep_results = []
        for eps in epsilons:
            try:
                atk_kwargs = _attack_kwargs(atk_name, epsilon=eps,
                                            nb_classes=nb_classes, imgsz=imgsz)
                res = evaluator.evaluate(loader, attack_name=atk_name,
                                         attack_kwargs=atk_kwargs)
                entry = {
                    "epsilon": eps,
                    "clean_accuracy": res["clean_accuracy"],
                    "robust_accuracy": res["robust_accuracy"],
                    "accuracy_drop": res["clean_accuracy"] - res["robust_accuracy"],
                }
                sweep_results.append(entry)
                print(
                    f"  {atk_name} ε={eps:.4f}: robust={res['robust_accuracy']:.2f}%")
            except Exception as e:
                if verbose:
                    print(f"  {atk_name} ε={eps}: FAILED — {e}")
        if sweep_results:
            results["epsilon_sweep"][atk_name] = sweep_results

    # ---- Certified robustness (randomized smoothing) ----
    if certified:
        print("\n── Certified Robustness (Randomized Smoothing) ──")
        print(f"  σ={certified_sigma}, n_samples={certified_n}")
        cert_results = _run_certified(
            model, loader, dev, nb_classes,
            sigma=certified_sigma, n_samples=certified_n,
        )
        results["certified"] = cert_results
        if cert_results:
            print(
                f"  Mean certified radius: {cert_results['mean_certified_radius']:.4f}")
            print(
                f"  Certified accuracy:    {cert_results['certified_accuracy']:.2f}%")

    # ---- Summary statistics ----
    results["summary"] = _robustness_summary(results)
    _print_robustness_summary(results["summary"])

    # ---- Save JSON ----
    json_path = robustness_dir / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {json_path}")

    # ---- Save CSV (flat, re-plottable) ----
    csv_path = _save_robustness_csv(results, robustness_dir)
    print(f"  CSV saved   → {csv_path}")

    # ---- Save TXT (human-readable report) ----
    txt_path = _save_robustness_txt(results, robustness_dir)
    print(f"  TXT saved   → {txt_path}")

    # ---- Generate robustness visualizations ----
    _generate_robustness_plots(results, robustness_dir)

    return results


# ── Robustness CSV/TXT writers ────────────────────────────────────────

def _save_robustness_csv(results: dict, save_dir: Path) -> Path:
    """Save robustness results to CSV files for later re-plotting.

    Creates two CSV files:
        - ``robustness_per_attack.csv``  — one row per attack (default ε)
        - ``robustness_epsilon_sweep.csv`` — one row per (attack, ε) pair

    Returns the per-attack CSV path.
    """
    import csv

    # ── Per-attack summary table ──
    pa_path = save_dir / "robustness_per_attack.csv"
    per_attack = results.get("per_attack", {})
    if per_attack:
        fieldnames = [
            "attack", "clean_accuracy", "robust_accuracy",
            "accuracy_drop", "attack_success_rate", "samples",
        ]
        with open(pa_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for atk_name, atk_res in per_attack.items():
                writer.writerow({
                    "attack": atk_name,
                    "clean_accuracy": f"{atk_res.get('clean_accuracy', 0):.4f}",
                    "robust_accuracy": f"{atk_res.get('robust_accuracy', 0):.4f}",
                    "accuracy_drop": f"{atk_res.get('accuracy_drop', 0):.4f}",
                    "attack_success_rate": f"{atk_res.get('attack_success_rate', 0):.4f}",
                    "samples": atk_res.get("samples", 0),
                })

    # ── Epsilon sweep table (long-form — easy to pivot / plot) ──
    sweep = results.get("epsilon_sweep", {})
    if sweep:
        sweep_path = save_dir / "robustness_epsilon_sweep.csv"
        fieldnames_s = [
            "attack", "epsilon", "clean_accuracy",
            "robust_accuracy", "accuracy_drop",
        ]
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_s)
            writer.writeheader()
            for atk_name, entries in sweep.items():
                for entry in entries:
                    writer.writerow({
                        "attack": atk_name,
                        "epsilon": f"{entry['epsilon']:.6f}",
                        "clean_accuracy": f"{entry.get('clean_accuracy', 0):.4f}",
                        "robust_accuracy": f"{entry.get('robust_accuracy', 0):.4f}",
                        "accuracy_drop": f"{entry.get('accuracy_drop', 0):.4f}",
                    })

    # ── Summary row (single-line CSV — easy to merge across variants) ──
    summary = results.get("summary", {})
    if summary:
        summary_path = save_dir / "robustness_summary.csv"
        flat = {
            "clean_accuracy": summary.get("clean_accuracy", ""),
            "mean_robust_accuracy": summary.get("mean_robust_accuracy", ""),
            "min_robust_accuracy": summary.get("min_robust_accuracy", ""),
            "robustness_ratio": summary.get("robustness_ratio", ""),
            "mean_attack_success_rate": summary.get("mean_attack_success_rate", ""),
            "max_attack_success_rate": summary.get("max_attack_success_rate", ""),
            "mean_accuracy_drop": summary.get("mean_accuracy_drop", ""),
            "max_accuracy_drop": summary.get("max_accuracy_drop", ""),
            "mean_auac": summary.get("mean_auac", ""),
            "certified_accuracy": summary.get("certified_accuracy", ""),
            "mean_certified_radius": summary.get("mean_certified_radius", ""),
        }
        # Filter out empty
        flat = {k: v for k, v in flat.items() if v != ""}
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow({
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in flat.items()
            })

    return pa_path


def _save_robustness_txt(results: dict, save_dir: Path) -> Path:
    """Save a human-readable robustness report to TXT.

    This mirrors the terminal output so you can reproduce any figure
    manually from the numbers in this file.
    """
    txt_path = save_dir / "robustness_report.txt"
    lines = []
    w = 32  # label width

    lines.append("=" * 70)
    lines.append("  Adversarial Robustness Evaluation Report")
    lines.append("=" * 70)

    # ── Per-attack table ──
    per_attack = results.get("per_attack", {})
    if per_attack:
        lines.append("")
        lines.append("── Per-Attack Results (default ε) ──")
        lines.append(
            f"  {'Attack':<12} {'Clean%':>10} {'Robust%':>10} "
            f"{'Drop(pp)':>10} {'ASR%':>10} {'Samples':>8}"
        )
        lines.append("  " + "-" * 62)
        for atk_name, r in per_attack.items():
            lines.append(
                f"  {atk_name:<12} {r.get('clean_accuracy', 0):>10.2f} "
                f"{r.get('robust_accuracy', 0):>10.2f} "
                f"{r.get('accuracy_drop', 0):>10.2f} "
                f"{r.get('attack_success_rate', 0):>10.2f} "
                f"{r.get('samples', 0):>8d}"
            )

    # ── Epsilon sweep ──
    sweep = results.get("epsilon_sweep", {})
    if sweep:
        lines.append("")
        lines.append("── Epsilon Sensitivity Sweep ──")
        for atk_name, entries in sweep.items():
            lines.append(f"\n  Attack: {atk_name}")
            lines.append(
                f"  {'Epsilon':>10} {'Clean%':>10} {'Robust%':>10} {'Drop(pp)':>10}")
            lines.append("  " + "-" * 42)
            for e in entries:
                lines.append(
                    f"  {e['epsilon']:>10.6f} {e.get('clean_accuracy', 0):>10.2f} "
                    f"{e.get('robust_accuracy', 0):>10.2f} "
                    f"{e.get('accuracy_drop', 0):>10.2f}"
                )

    # ── Certified robustness ──
    cert = results.get("certified", {})
    if cert:
        lines.append("")
        lines.append("── Certified Robustness (Randomized Smoothing) ──")
        for key, label in [
            ("sigma", "Noise σ"),
            ("n_samples", "Noise samples"),
            ("total_samples", "Total samples"),
            ("certified_accuracy", "Certified Accuracy (%)"),
            ("smoothed_accuracy", "Smoothed Accuracy (%)"),
            ("mean_certified_radius", "Mean Certified Radius"),
            ("median_certified_radius", "Median Certified Radius"),
            ("max_certified_radius", "Max Certified Radius"),
            ("std_certified_radius", "Std Certified Radius"),
        ]:
            val = cert.get(key)
            if val is not None:
                if isinstance(val, float):
                    lines.append(f"  {label:<{w}} {val:.6f}")
                else:
                    lines.append(f"  {label:<{w}} {val}")

        hist = cert.get("radii_histogram", {})
        if hist:
            lines.append(f"\n  {'Radius Threshold':<{w}} Count")
            lines.append("  " + "-" * 40)
            for thresh, count in hist.items():
                lines.append(f"  {thresh:<{w}} {count}")

    # ── Summary ──
    summary = results.get("summary", {})
    if summary:
        lines.append("")
        lines.append("=" * 70)
        lines.append("  Robustness Summary")
        lines.append("=" * 70)
        for key, label in [
            ("clean_accuracy", "Clean Accuracy (%)"),
            ("mean_robust_accuracy", "Mean Robust Accuracy (%)"),
            ("min_robust_accuracy", "Min Robust Accuracy (%)"),
            ("robustness_ratio", "Robustness Ratio (worst/clean)"),
            ("mean_attack_success_rate", "Mean Attack Success Rate (%)"),
            ("max_attack_success_rate", "Max Attack Success Rate (%)"),
            ("mean_accuracy_drop", "Mean Accuracy Drop (pp)"),
            ("max_accuracy_drop", "Max Accuracy Drop (pp)"),
            ("mean_auac", "Mean AUAC (area under acc curve)"),
            ("certified_accuracy", "Certified Accuracy (%)"),
            ("mean_certified_radius", "Mean Certified Radius"),
        ]:
            val = summary.get(key)
            if val is not None:
                if isinstance(val, float):
                    lines.append(f"  {label:<{w}} {val:.6f}")
                else:
                    lines.append(f"  {label:<{w}} {val}")

        auac = summary.get("auac_per_attack", {})
        if auac:
            lines.append(f"\n  AUAC per attack:")
            for atk, val in auac.items():
                lines.append(f"    {atk:<12} {val:.6f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  End of Robustness Report")
    lines.append("=" * 70)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path


def _build_robustness_loader(val_dir: Path, imgsz: int = 224,
                             batch: int = 32, workers: int = 4):
    """Build a plain (image, label) DataLoader for robustness evaluation."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        # No normalization — adversarial attacks need [0, 1] pixel range
    ])

    try:
        dataset = datasets.ImageFolder(str(val_dir), transform=transform)
        return DataLoader(dataset, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)
    except Exception as e:
        print(f"  WARNING: could not build robustness dataloader: {e}")
        return None


def _detect_nb_classes(val_dir: Path) -> int:
    """Count number of class subdirectories."""
    return len([d for d in val_dir.iterdir() if d.is_dir()])


def _attack_kwargs(attack_name: str, epsilon: float = 8/255,
                   nb_classes: int = 2, imgsz: int = 224) -> dict:
    """Build attack-specific keyword arguments."""
    kwargs = {"epsilon": epsilon}

    if attack_name == "pgd":
        kwargs["alpha"] = epsilon / 4
        kwargs["num_iter"] = 20
    elif attack_name == "bim":
        kwargs["alpha"] = epsilon / 4
        kwargs["num_iter"] = 10
    elif attack_name == "mim":
        kwargs["alpha"] = epsilon / 4
        kwargs["num_iter"] = 10
        kwargs["decay"] = 1.0
    elif attack_name == "cw":
        kwargs["learning_rate"] = 0.01
        kwargs["max_iter"] = 100
    elif attack_name in ("deepfool", "apgd", "square"):
        kwargs["input_shape"] = (3, imgsz, imgsz)
        kwargs["nb_classes"] = nb_classes

    return kwargs


def _run_certified(model, loader, device, nb_classes,
                   sigma: float = 0.25, n_samples: int = 100) -> dict:
    """Run randomized smoothing certification over the dataloader."""
    from ultralytics.utils.defense import CertifiedDefense

    radii = []
    correct = 0
    total = 0
    certified_correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i].item()
            try:
                pred_cls, radius = CertifiedDefense.randomized_smoothing(
                    model, xi, num_samples=n_samples,
                    sigma=sigma, num_classes=nb_classes,
                )
                radii.append(radius)
                if pred_cls == yi:
                    correct += 1
                    if radius > 0:
                        certified_correct += 1
            except Exception:
                radii.append(0.0)
            total += 1

    if total == 0:
        return {}

    radii_arr = np.array(radii)
    return {
        "sigma": sigma,
        "n_samples": n_samples,
        "total_samples": total,
        "certified_accuracy": 100.0 * certified_correct / total,
        "smoothed_accuracy": 100.0 * correct / total,
        "mean_certified_radius": float(radii_arr.mean()),
        "median_certified_radius": float(np.median(radii_arr)),
        "max_certified_radius": float(radii_arr.max()),
        "min_certified_radius": float(radii_arr.min()),
        "std_certified_radius": float(radii_arr.std()),
        "radii_histogram": {
            f">{t:.2f}": int((radii_arr > t).sum())
            for t in [0.0, 0.1, 0.25, 0.5, 1.0]
        },
    }


def _robustness_summary(results: dict) -> dict:
    """Compute aggregate robustness summary statistics."""
    summary: Dict[str, Any] = {}

    per_attack = results.get("per_attack", {})
    if per_attack:
        clean_accs = [v["clean_accuracy"] for v in per_attack.values()]
        robust_accs = [v["robust_accuracy"] for v in per_attack.values()]
        asrs = [v.get("attack_success_rate", 0) for v in per_attack.values()]
        drops = [v.get("accuracy_drop", 0) for v in per_attack.values()]

        summary["clean_accuracy"] = clean_accs[0] if clean_accs else 0
        summary["mean_robust_accuracy"] = float(np.mean(robust_accs))
        summary["min_robust_accuracy"] = float(np.min(robust_accs))
        summary["max_attack_success_rate"] = float(np.max(asrs))
        summary["mean_attack_success_rate"] = float(np.mean(asrs))
        summary["max_accuracy_drop"] = float(np.max(drops))
        summary["mean_accuracy_drop"] = float(np.mean(drops))

        # Robustness score: ratio of worst-case robust acc to clean acc
        if summary["clean_accuracy"] > 0:
            summary["robustness_ratio"] = (
                summary["min_robust_accuracy"] / summary["clean_accuracy"]
            )
        else:
            summary["robustness_ratio"] = 0.0

    # Epsilon sweep: area-under-accuracy-curve (AUAC) per attack
    auac = {}
    for atk_name, sweep in results.get("epsilon_sweep", {}).items():
        if len(sweep) >= 2:
            eps_vals = [s["epsilon"] for s in sweep]
            rob_vals = [s["robust_accuracy"] for s in sweep]
            # Trapezoidal integration normalized by epsilon range
            area = float(np.trapz(rob_vals, eps_vals))
            eps_range = max(eps_vals) - min(eps_vals)
            auac[atk_name] = area / eps_range if eps_range > 0 else 0
    if auac:
        summary["auac_per_attack"] = auac
        summary["mean_auac"] = float(np.mean(list(auac.values())))

    # Certified
    cert = results.get("certified", {})
    if cert:
        summary["certified_accuracy"] = cert.get("certified_accuracy", 0)
        summary["mean_certified_radius"] = cert.get("mean_certified_radius", 0)

    return summary


def _print_robustness_summary(summary: dict) -> None:
    """Pretty-print robustness summary."""
    if not summary:
        return
    print("\n" + "=" * 60)
    print("Robustness Summary")
    print("=" * 60)
    for key in [
        "clean_accuracy", "mean_robust_accuracy", "min_robust_accuracy",
        "robustness_ratio", "mean_attack_success_rate",
        "max_attack_success_rate", "mean_accuracy_drop",
        "max_accuracy_drop", "mean_auac",
        "certified_accuracy", "mean_certified_radius",
    ]:
        val = summary.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                print(f"  {label:<30} {val:.4f}")
            else:
                print(f"  {label:<30} {val}")
    print("=" * 60)


def _generate_robustness_plots(results: dict, save_dir: Path) -> None:
    """Generate epsilon-sensitivity and ASR heatmap visualizations."""
    try:
        from ultralytics.utils.classify_visualize import (
            plot_epsilon_sensitivity, plot_asr_heatmap,
        )
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # ---- Epsilon sensitivity curves ----
    sweep = results.get("epsilon_sweep", {})
    if sweep:
        for atk_name, entries in sweep.items():
            epsilons = [e["epsilon"] for e in entries]
            metrics_at_eps = {"robust_accuracy": [
                e["robust_accuracy"] for e in entries]}
            if entries and "clean_accuracy" in entries[0]:
                metrics_at_eps["clean_accuracy"] = [
                    entries[0]["clean_accuracy"]] * len(entries)
            try:
                fig = plot_epsilon_sensitivity(
                    epsilons, metrics_at_eps,
                    save_path=save_dir / f"epsilon_sensitivity_{atk_name}.png",
                    title=f"Epsilon Sensitivity — {atk_name.upper()}",
                )
                if fig:
                    plt.close(fig)
            except Exception:
                pass

    # ---- ASR heatmap (if multiple attacks) ----
    per_attack = results.get("per_attack", {})
    if len(per_attack) >= 2:
        asr_matrix = {}
        for atk_name, atk_res in per_attack.items():
            asr_matrix[atk_name] = {
                "default_eps": atk_res.get("attack_success_rate", 0)
            }
        try:
            fig = plot_asr_heatmap(
                asr_matrix,
                save_path=save_dir / "asr_heatmap.png",
                title="Attack Success Rate",
            )
            if fig:
                plt.close(fig)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# 6. Feature embeddings (t-SNE / PCA)
# ══════════════════════════════════════════════════════════════════════

def run_embeddings(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    save_dir: Union[str, Path],
    device: str = "0",
    do_tsne: bool = True,
    do_pca: bool = True,
    max_samples: int = 2000,
    imgsz: int = 224,
) -> None:
    """Extract penultimate-layer features → t-SNE / PCA scatter plots."""
    from ultralytics.utils.classify_visualize import plot_tsne, plot_pca
    from ultralytics.nn.tasks import load_checkpoint
    from torch.utils.data import DataLoader
    from ultralytics.data import ClassificationDataset
    from types import SimpleNamespace
    import matplotlib.pyplot as plt

    print("\nExtracting feature embeddings...")

    dev = f"cuda:{device}"
    model, _ = load_checkpoint(str(model_path), device=dev)
    model.eval()

    data_dir = Path(data_path)
    val_path = data_dir / \
        "val" if (data_dir / "val").exists() else data_dir / "test"
    ds_args = SimpleNamespace(
        imgsz=imgsz, crop_fraction=1.0, augment=False, task="classify",
        cache=False, scale=0.5, fraction=1.0,
        fliplr=0.0, flipud=0.0, erasing=0.0, auto_augment=None,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )
    dataset = ClassificationDataset(
        root=str(val_path), args=ds_args, augment=False, prefix="val")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Hook to capture features
    features_out: list = []

    def hook_fn(module, inp, out):
        features_out.append(out.detach().cpu())

    hook = _register_feature_hook(model, hook_fn)

    all_features, all_labels = [], []
    n_collected = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(dev).float()
            labels = batch["cls"]
            _ = model(imgs)
            if features_out:
                feat = features_out[-1]
                if feat.ndim > 2:
                    feat = feat.flatten(1)
                all_features.append(feat.numpy())
                all_labels.append(labels.numpy().ravel())
                features_out.clear()
            n_collected += len(labels)
            if n_collected >= max_samples:
                break

    if hook:
        hook.remove()

    if not all_features:
        print("  WARNING: no features extracted")
        return

    features = np.concatenate(all_features)[:max_samples]
    labels = np.concatenate(all_labels)[:max_samples]
    class_names = _get_class_names(model) or None

    viz_dir = Path(save_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if do_tsne:
        print(f"  t-SNE ({len(features)} samples)...")
        fig = plot_tsne(features, labels, class_names,
                        viz_dir / "tsne_embeddings.png")
        if fig:
            plt.close(fig)

    if do_pca:
        print(f"  PCA ({len(features)} samples)...")
        fig = plot_pca(features, labels, class_names,
                       viz_dir / "pca_embeddings.png")
        if fig:
            plt.close(fig)

    print(f"  Embedding plots saved -> {viz_dir}")


def _register_feature_hook(model, hook_fn):
    """Register a forward hook on the penultimate layer of *model*."""
    named = list(model.model.named_modules()) if hasattr(
        model, "model") else list(model.named_modules())
    # Skip final Linear; hook on last pooling/norm layer
    for name, module in reversed(named):
        if isinstance(module, torch.nn.Linear):
            continue
        if isinstance(module, (
            torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d,
            torch.nn.LayerNorm,
        )):
            print(f"  Hooked at: {name}")
            return module.register_forward_hook(hook_fn)
    # Fallback: second-to-last module
    if len(named) > 2:
        _, last_mod = named[-2]
        return last_mod.register_forward_hook(hook_fn)
    return None


# ══════════════════════════════════════════════════════════════════════
# 7. Model export
# ══════════════════════════════════════════════════════════════════════

EXPORT_FORMATS = ["onnx", "torchscript",
                  "tflite", "coreml", "openvino", "engine"]


def run_export(
    model_path: Union[str, Path],
    formats: List[str],
    imgsz: int = 224,
    device: str = "0",
) -> Dict[str, str]:
    """Export model to one or more deployment formats.

    Returns ``{format: export_path}`` for successes.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Model Export")
    print("=" * 60)

    model = YOLO(str(model_path))
    results: Dict[str, str] = {}
    for fmt in formats:
        print(f"\n  Exporting to {fmt}...")
        try:
            export_path = model.export(format=fmt, imgsz=imgsz, device=device)
            print(f"  OK: {export_path}")
            results[fmt] = str(export_path)
        except Exception as e:
            print(f"  FAIL: {fmt} — {e}")
    return results


# ══════════════════════════════════════════════════════════════════════
# 8. Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════

def print_results(results: dict) -> None:
    """Print a formatted summary of evaluation metrics.

    Organised for medical image classification where recall (sensitivity),
    specificity, and calibration are critical.  Top-k accuracy is omitted
    because it adds no value for tasks with fewer than ~20 classes.
    """
    if not results:
        print("  (no results)")
        return

    nc = results.get("num_classes", 0)
    ns = results.get("num_samples", 0)

    print("\n" + "=" * 62)
    print(f"  Evaluation Results  ({nc} classes, {ns} samples)")
    print("=" * 62)

    # ── Primary performance ──────────────────────────────────────────
    _pr_section("Primary Performance")
    _pr_metric(results, "accuracy",           "Accuracy")
    _pr_metric(results, "balanced_accuracy",   "Balanced Accuracy")
    _pr_metric(results, "f1_weighted",         "F1 Score (weighted)")
    _pr_metric(results, "f1_macro",            "F1 Score (macro)")

    # ── Sensitivity / Specificity (critical for medical) ─────────────
    _pr_section("Recall & Specificity")
    _pr_metric(results, "recall_weighted",     "Recall (weighted)")
    _pr_metric(results, "recall_macro",        "Recall (macro)")
    _pr_metric(results, "specificity_macro",   "Specificity (macro)")
    _pr_metric(results, "precision_weighted",  "Precision (weighted)")
    _pr_metric(results, "precision_macro",     "Precision (macro)")

    # ── Agreement & correlation ──────────────────────────────────────
    _pr_section("Agreement & Correlation")
    _pr_metric(results, "mcc",                 "MCC")
    _pr_metric(results, "cohen_kappa",         "Cohen's Kappa")

    # ── Probabilistic / calibration (important for clinical trust) ───
    _pr_section("Probabilistic & Calibration")
    _pr_metric(results, "roc_auc",             "ROC AUC")
    _pr_metric(results, "average_precision",   "Average Precision")
    _pr_metric(results, "log_loss",            "Log Loss")
    _pr_metric(results, "brier_score",         "Brier Score")
    _pr_metric(results, "ece",                 "ECE (calibration)")

    # ── Per-class breakdown ───────────────────────────────────────────
    pc = results.get("per_class", [])
    if pc:
        _pr_section("Per-Class Breakdown")
        hdr = (f"  {'Class':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
               f"{'F1':>8} {'Spec':>8} {'Supp':>8}")
        print(hdr)
        print("  " + "-" * 68)
        for c in pc:
            name = c["class_name"][:20]
            print(f"  {name:<20} {c['accuracy']:>8.4f} {c['precision']:>8.4f} "
                  f"{c['recall']:>8.4f} {c['f1']:>8.4f} "
                  f"{c['specificity']:>8.4f} {c['support']:>8d}")

    print("=" * 62)


def _pr_section(title: str) -> None:
    """Print a section header inside the results block."""
    print(f"\n  ── {title} {'─' * max(1, 42 - len(title))}")


def _pr_metric(results: dict, key: str, label: str) -> None:
    """Print a single metric line if it exists and is not None/empty."""
    val = results.get(key)
    if val is None or val == "":
        return
    if isinstance(val, float):
        print(f"  {label:<26} {val:.4f}")
    else:
        print(f"  {label:<26} {val}")
