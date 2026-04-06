"""
Saliency Map Engine
====================

Model-agnostic saliency / attribution visualization module.
Works with **any** ``nn.Module`` classifier — ResNet, ViT, MedDef, YOLO, etc.

Supported methods:
    1. **Grad-CAM**           — class activation mapping via gradient weighting
    2. **Grad-CAM++**         — improved alpha-weighted variant
    3. **Vanilla Gradient**   — input sensitivity (backprop to input)
    4. **Integrated Gradients** — Sundararajan et al. baseline → input integral
    5. **SmoothGrad**         — noise-averaged vanilla gradients

All methods return ``(H, W)`` numpy arrays in ``[0, 1]``.

Comprehensive figure:
    ``create_comprehensive_figure()`` creates a publication-quality
    multi-panel figure with all methods + statistics in one image.

Usage:
    from ultralytics.utils.saliency import SaliencyGenerator

    gen = SaliencyGenerator(model, device="cuda:0")
    heatmap = gen.gradcam(input_tensor)
    gen.create_comprehensive_figure(input_tensor, original_image,
                                     class_name="Melanoma", confidence=0.95)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry: allows extending with custom saliency methods
# ---------------------------------------------------------------------------
_SALIENCY_REGISTRY: Dict[str, callable] = {}


def register_saliency(name: str):
    """Decorator to register a custom saliency method.

    The function must accept ``(model, input_tensor, target_class, **kw)``
    and return a ``(H, W)`` numpy heatmap in ``[0, 1]``.

    Example::

        @register_saliency("my_method")
        def my_saliency(model, x, target_class=None, **kw):
            ...
            return heatmap  # (H, W) ndarray
    """
    def decorator(fn):
        _SALIENCY_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SaliencyGenerator:
    """Model-agnostic saliency map generator.

    Parameters
    ----------
    model : nn.Module
        Any classification model (must accept (B, C, H, W) input).
    device : str
        Device string (``"cuda:0"``, ``"cpu"``, etc.).
    output_dir : str | Path | None
        Default directory for saving figures.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Auto-detect target layer
    # ------------------------------------------------------------------ #

    def _find_last_conv(self) -> Optional[nn.Module]:
        """Find the last Conv2d layer in the model (works for any arch)."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    # ------------------------------------------------------------------ #
    #  1. Grad-CAM
    # ------------------------------------------------------------------ #

    def gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : (1, C, H, W) tensor
        target_class : class index (argmax if None)
        target_layer : specific layer (auto-detects last Conv2d if None)

        Returns
        -------
        (H, W) heatmap in [0, 1]
        """
        if target_layer is None:
            target_layer = self._find_last_conv()
        if target_layer is None:
            logger.warning("No Conv2d layer found for Grad-CAM")
            return np.ones(input_tensor.shape[2:]) * 0.5

        activations, gradients = [], []

        def fwd_hook(m, inp, out):
            activations.append(out.detach())

        def bwd_hook(m, gi, go):
            gradients.append(go[0].detach())

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x = input_tensor.clone().detach().to(self.device).requires_grad_(False)
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot)

            act = activations[0]  # (1, C, h, w)
            grad = gradients[0]   # (1, C, h, w)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=input_tensor.shape[2:],
                mode="bilinear", align_corners=False,
            )
            cam = cam.squeeze().cpu().numpy()
            return self._normalize(cam)
        finally:
            h_fwd.remove()
            h_bwd.remove()

    # ------------------------------------------------------------------ #
    #  2. Grad-CAM++
    # ------------------------------------------------------------------ #

    def gradcam_plusplus(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """Grad-CAM++ with improved alpha weighting."""
        if target_layer is None:
            target_layer = self._find_last_conv()
        if target_layer is None:
            return np.ones(input_tensor.shape[2:]) * 0.5

        activations, gradients = [], []

        def fwd_hook(m, inp, out):
            activations.append(out.detach())

        def bwd_hook(m, gi, go):
            gradients.append(go[0].detach())

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x = input_tensor.clone().detach().to(self.device)
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

            act = activations[0]
            grad = gradients[0]

            grad_2 = grad ** 2
            grad_3 = grad ** 3
            sum_act = torch.sum(act, dim=(2, 3), keepdim=True)
            alpha_num = grad_2
            alpha_den = 2 * grad_2 + sum_act * grad_3 + 1e-8
            alpha = alpha_num / alpha_den
            weights = torch.sum(alpha * F.relu(grad), dim=(2, 3), keepdim=True)

            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=input_tensor.shape[2:],
                mode="bilinear", align_corners=False,
            )
            return self._normalize(cam.squeeze().cpu().numpy())
        finally:
            h_fwd.remove()
            h_bwd.remove()

    # ------------------------------------------------------------------ #
    #  3. Vanilla Gradient
    # ------------------------------------------------------------------ #

    def vanilla_gradient(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Vanilla gradient saliency (input sensitivity)."""
        x = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        output = self.model(x)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        saliency = torch.abs(x.grad.data)
        saliency = torch.max(saliency, dim=1)[0]  # max across channels
        return self._normalize(saliency.squeeze().cpu().numpy())

    # ------------------------------------------------------------------ #
    #  4. Integrated Gradients
    # ------------------------------------------------------------------ #

    def integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Integrated Gradients attribution (Sundararajan et al. 2017)."""
        x = input_tensor.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        else:
            baseline = baseline.to(self.device)

        # Determine target class from full input
        with torch.no_grad():
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        grads = []
        for i in range(steps + 1):
            alpha = float(i) / steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            out = self.model(interpolated)
            self.model.zero_grad()
            one_hot = torch.zeros_like(out)
            one_hot[0, target_class] = 1.0
            out.backward(gradient=one_hot)
            grads.append(interpolated.grad.data.clone())

        avg_grad = torch.stack(grads).mean(dim=0)
        ig = (x - baseline) * avg_grad
        ig = torch.abs(ig)
        ig = torch.max(ig, dim=1)[0]
        return self._normalize(ig.squeeze().cpu().numpy())

    # ------------------------------------------------------------------ #
    #  5. SmoothGrad
    # ------------------------------------------------------------------ #

    def smooth_grad(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """SmoothGrad — noise-averaged vanilla gradients."""
        x = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        stdev = noise_level * (x.max() - x.min()).item()
        all_grads = torch.zeros_like(x)

        for _ in range(n_samples):
            noisy = x + torch.randn_like(x) * stdev
            noisy = noisy.clone().detach().requires_grad_(True)
            out = self.model(noisy)
            self.model.zero_grad()
            one_hot = torch.zeros_like(out)
            one_hot[0, target_class] = 1.0
            out.backward(gradient=one_hot)
            all_grads += noisy.grad.data

        avg = all_grads / n_samples
        avg = torch.abs(avg)
        avg = torch.max(avg, dim=1)[0]
        return self._normalize(avg.squeeze().cpu().numpy())

    # ------------------------------------------------------------------ #
    #  Comprehensive figure
    # ------------------------------------------------------------------ #

    def create_comprehensive_figure(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_name: str = "Unknown",
        confidence: float = 0.0,
        save_name: str = "comprehensive_saliency",
        title: str = "Comprehensive Saliency Analysis",
        methods: Optional[List[str]] = None,
    ) -> str:
        """Create publication-quality multi-panel saliency figure.

        Parameters
        ----------
        input_tensor : (1, C, H, W) preprocessed tensor
        original_image : (H, W, 3) RGB image (uint8 or float)
        class_name : predicted class label
        confidence : model confidence
        save_name : filename stem for saved figure
        title : figure title
        methods : which methods to include (default: all)

        Returns
        -------
        str : path to saved figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        if methods is None:
            methods = ["gradcam", "gradcam_plusplus", "vanilla_gradient", "smooth_grad"]

        # Compute saliency maps
        heatmaps = {}
        with torch.enable_grad():
            for method_name in methods:
                fn = getattr(self, method_name, None)
                if fn is None:
                    # Check registry
                    fn = _SALIENCY_REGISTRY.get(method_name)
                    if fn:
                        heatmaps[method_name] = fn(
                            self.model, input_tensor, target_class=None,
                        )
                        continue
                    logger.warning(f"Unknown saliency method: {method_name}")
                    continue
                try:
                    heatmaps[method_name] = fn(input_tensor)
                except Exception as e:
                    logger.warning(f"Saliency method '{method_name}' failed: {e}")

        if not heatmaps:
            logger.warning("No saliency maps computed")
            return ""

        # Normalize image
        img = original_image.astype(float)
        if img.max() > 1.0:
            img = img / 255.0

        n_methods = len(heatmaps)
        n_cols = min(4, n_methods + 1)
        n_rows = 3

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows), dpi=300)
        fig.patch.set_facecolor("white")
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.15)

        # Custom colormaps
        from matplotlib.colors import LinearSegmentedColormap
        jet_clean = LinearSegmentedColormap.from_list(
            "jet_clean",
            ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000", "#800000"],
            N=256,
        )
        medical_cmap = LinearSegmentedColormap.from_list(
            "medical",
            ["#000428", "#004e92", "#0077b6", "#00b4d8", "#90e0ef"],
            N=256,
        )

        # --- Row 1: Original + each method ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(img)
        ax0.set_title(
            f"Original\n{class_name} ({confidence:.1%})",
            fontsize=11, fontweight="bold",
        )
        ax0.axis("off")

        method_names_display = {
            "gradcam": "Grad-CAM",
            "gradcam_plusplus": "Grad-CAM++",
            "vanilla_gradient": "Vanilla Gradient",
            "integrated_gradients": "Integrated Grad.",
            "smooth_grad": "SmoothGrad",
        }

        for idx, (method_key, heatmap) in enumerate(heatmaps.items()):
            col = (idx + 1) % n_cols
            row = (idx + 1) // n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.imshow(heatmap, cmap=jet_clean, alpha=0.6)
            display_name = method_names_display.get(method_key, method_key.replace("_", " ").title())
            ax.set_title(display_name, fontsize=11, fontweight="bold")
            ax.axis("off")

        # --- Row 2: Different colormaps for primary method ---
        primary = list(heatmaps.values())[0]
        cmaps = [medical_cmap, "viridis", "hot", "inferno"]
        cmap_names = ["Medical", "Viridis", "Hot", "Inferno"]
        row2_start = 1 if n_methods < n_cols else 2
        for i, (cmap, cname) in enumerate(zip(cmaps, cmap_names)):
            if i >= n_cols:
                break
            ax = fig.add_subplot(gs[row2_start, i])
            ax.imshow(img)
            ax.imshow(primary, cmap=cmap, alpha=0.65)
            ax.set_title(f"{cname} Colormap", fontsize=10, fontweight="bold")
            ax.axis("off")

        # --- Row 3: Analysis ---
        # Histogram
        ax_hist = fig.add_subplot(gs[n_rows - 1, 0])
        ax_hist.hist(
            primary.flatten(), bins=50, color="#3498DB", alpha=0.7,
            edgecolor="black", linewidth=0.5,
        )
        ax_hist.axvline(
            primary.mean(), color="red", linestyle="--", linewidth=2,
            label=f"Mean: {primary.mean():.3f}",
        )
        ax_hist.set_xlabel("Activation Value", fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.set_title("Saliency Distribution", fontsize=11, fontweight="bold")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(alpha=0.3)

        # Method comparison
        if len(heatmaps) >= 2:
            ax_cmp = fig.add_subplot(gs[n_rows - 1, 1])
            m_names = [
                method_names_display.get(k, k)[:12] for k in heatmaps
            ]
            m_means = [h.mean() for h in heatmaps.values()]
            m_stds = [h.std() for h in heatmaps.values()]
            bar_colors = ["#E74C3C", "#9B59B6", "#F39C12", "#2ECC71", "#3498DB"]
            ax_cmp.bar(
                m_names, m_means, yerr=m_stds, capsize=5,
                color=bar_colors[: len(m_names)],
                edgecolor="black", linewidth=1, alpha=0.85,
            )
            ax_cmp.set_ylabel("Mean Activation", fontsize=10)
            ax_cmp.set_title("Method Comparison", fontsize=11, fontweight="bold")
            ax_cmp.grid(axis="y", alpha=0.3)
            plt.setp(ax_cmp.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # Top activation regions
        if n_cols >= 3:
            ax_top = fig.add_subplot(gs[n_rows - 1, 2])
            threshold = np.percentile(primary, 90)
            high = primary >= threshold
            ax_top.imshow(img)
            ax_top.imshow(high.astype(float), cmap="Reds", alpha=0.5)
            ax_top.set_title("Top 10% Regions", fontsize=11, fontweight="bold")
            ax_top.axis("off")

        # Stats panel
        if n_cols >= 4:
            ax_stats = fig.add_subplot(gs[n_rows - 1, 3])
            ax_stats.axis("off")
            stats_text = (
                f"  Prediction:  {class_name}\n"
                f"  Confidence:  {confidence:.2%}\n\n"
                f"  Saliency Stats:\n"
                f"    Mean:     {primary.mean():.4f}\n"
                f"    Std:      {primary.std():.4f}\n"
                f"    Max:      {primary.max():.4f}\n\n"
                f"  Coverage (>50%): {(primary > 0.5).sum() / primary.size * 100:.1f}%\n"
                f"  Focus (>70%):    {(primary > 0.7).sum() / primary.size * 100:.1f}%"
            )
            ax_stats.text(
                0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                fontsize=10, family="monospace", va="center", ha="center",
                bbox=dict(
                    boxstyle="round", facecolor="#E8F4FD",
                    edgecolor="#2196F3", linewidth=2,
                ),
            )

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # Save
        if self.output_dir:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight", facecolor="white",
            )
            plt.close(fig)
            logger.info(f"Saliency figure saved → {save_path}")
            return str(save_path)

        plt.close(fig)
        return ""

    # ------------------------------------------------------------------ #
    #  Batch saliency for dataset samples
    # ------------------------------------------------------------------ #

    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        transform: callable,
        class_names: Optional[List[str]] = None,
        method: str = "gradcam",
        save_prefix: str = "saliency",
    ) -> List[str]:
        """Generate saliency maps for a batch of image files.

        Parameters
        ----------
        image_paths : list of image file paths
        transform : preprocessing transform (PIL Image → tensor)
        class_names : list of class labels
        method : saliency method name
        save_prefix : filename prefix

        Returns
        -------
        list of saved file paths
        """
        from PIL import Image

        saved = []
        fn = getattr(self, method, None)
        if fn is None:
            logger.error(f"Unknown method: {method}")
            return saved

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224))
                img_np = np.array(img_resized)
                input_tensor = transform(img).unsqueeze(0)

                with torch.enable_grad():
                    heatmap = fn(input_tensor)

                # Get prediction
                with torch.no_grad():
                    x = input_tensor.to(self.device)
                    out = self.model(x)
                    probs = torch.softmax(out, dim=1)
                    pred_cls = out.argmax(dim=1).item()
                    conf = probs[0, pred_cls].item()

                cls_name = (
                    class_names[pred_cls]
                    if class_names and pred_cls < len(class_names)
                    else f"class_{pred_cls}"
                )

                out_path = self.create_comprehensive_figure(
                    input_tensor, img_np,
                    class_name=cls_name, confidence=conf,
                    save_name=f"{save_prefix}_{i}_{cls_name}",
                )
                saved.append(out_path)
                logger.info(f"  [{i+1}/{len(image_paths)}] {cls_name} ({conf:.1%})")
            except Exception as e:
                logger.warning(f"Failed for {img_path}: {e}")

        return saved

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)


# ---------------------------------------------------------------------------
# Standalone convenience function (backward compat with compute_gradcam)
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: Optional[str] = None,
    target_class: Optional[int] = None,
    device: str = "cuda",
) -> np.ndarray:
    """Compute Grad-CAM heatmap — standalone convenience function.

    Parameters
    ----------
    model : nn.Module
    input_tensor : (1, C, H, W) tensor
    target_layer_name : module name for hook (auto-detects if None)
    target_class : class index (argmax if None)
    device : device string

    Returns
    -------
    (H, W) numpy array in [0, 1]
    """
    gen = SaliencyGenerator(model, device=device)

    target_layer = None
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

    return gen.gradcam(
        input_tensor, target_class=target_class, target_layer=target_layer,
    )
