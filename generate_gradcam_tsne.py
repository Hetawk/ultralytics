#!/usr/bin/env python3
"""
MedDef-VISTA: Grad-CAM + Attention Rollout + t-SNE Visualizations
===================================================================

Generates publication-quality visualizations for all 6 ablation variants:
  1. Grad-CAM / Attention Rollout heatmaps overlaid on sample X-rays
     - Both Normal and Tuberculosis samples, all 6 variants
  2. t-SNE feature embedding scatter plots
     - Per-variant and a combined 6-panel comparison figure

Results are saved to:
  runs/visualizations/gradcam_tsne/
    gradcam/       — per-variant Grad-CAM panels
    tsne/          — per-variant + combined t-SNE plots
    attention/     — attention rollout maps (ViT-native)

Usage (on server with GPU):
  cd /data2/enoch/ekd_coding_env/ultralytics
  python generate_gradcam_tsne.py

Optional overrides:
  python generate_gradcam_tsne.py \\
      --data /data2/enoch/ekd_coding_env/meddef_winlab/processed_data/tbcr \\
      --weights-base runs/classify/train_tbcr_final/tbcr \\
      --out runs/visualizations/gradcam_tsne \\
      --device 0 \\
      --n-samples 12
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = ["full", "no_def", "no_freq", "no_patch", "no_cbam", "baseline"]

VARIANT_LABELS = {
    "full":     "MedDef-VISTA Full",
    "no_def":   "w/o DefenseModule",
    "no_freq":  "w/o FreqDefense",
    "no_patch": "w/o PatchConsistency",
    "no_cbam":  "w/o CBAM",
    "baseline": "Baseline ViT",
}

CLASS_NAMES = ["Normal", "Tuberculosis"]
IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196 patches for 224×224


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: Path, device: str) -> nn.Module:
    """Load a MedDef2 model from a .pt checkpoint."""
    from ultralytics.nn.tasks import load_checkpoint
    model, _ = load_checkpoint(str(weights_path), device=device)
    model.eval()
    return model


def find_weights(weights_base: Path, variant: str) -> Optional[Path]:
    """Find distill_v2 best.pt for a variant, falling back to distill/best.pt."""
    candidate_v2 = weights_base / \
        f"{variant}_small" / "distill_v2" / "weights" / "best.pt"
    candidate_v1 = weights_base / \
        f"{variant}_small" / "distill" / "weights" / "best.pt"
    if candidate_v2.exists():
        return candidate_v2
    if candidate_v1.exists():
        print(f"  [WARN] distill_v2 not found for {variant}, using distill_v1")
        return candidate_v1
    print(f"  [WARN] No weights found for {variant} at {weights_base}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Image loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(imgsz: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: Path, imgsz: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Load image -> (tensor [1,3,H,W], original_rgb [H,W,3])."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    original = np.array(img)
    tensor = get_transform(imgsz)(img).unsqueeze(0)
    return tensor, original


def collect_samples(
    val_dir: Path,
    class_names: List[str],
    n_per_class: int = 4,
) -> Dict[str, List[Path]]:
    """Collect up to n_per_class sample images per class from val_dir."""
    samples: Dict[str, List[Path]] = {}
    for cls in class_names:
        cls_dir = val_dir / cls
        if not cls_dir.exists():
            # Try case-insensitive match
            matches = [d for d in val_dir.iterdir() if d.is_dir()
                       and d.name.lower() == cls.lower()]
            if not matches:
                print(f"  [WARN] Class directory not found: {cls_dir}")
                continue
            cls_dir = matches[0]
        paths = sorted(cls_dir.glob("*.jpg")) + \
            sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpeg"))
        samples[cls] = paths[:n_per_class]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Attention Rollout (ViT-native interpretability)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout for ViT models (Abnar & Zuidema, 2020).

    Rolls out attention weights from all transformer blocks to produce
    a spatial importance map for the CLS token.

    Works with MedDef2's transformer blocks which have .attn.qkv layer.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_maps: List[torch.Tensor] = []
        self._hooks: List = []

    def _hook_fn(self, module, input, output):
        """Hook to capture attention weights from MultiHeadAttention."""
        # output is the projected result; we re-compute attn weights
        # Instead, we hook the qkv layer and compute attention manually
        pass

    def _register_hooks(self):
        """Register hooks on all transformer blocks' attention modules."""
        self._attention_maps.clear()
        self._hooks.clear()

        # Find transformer blocks — either in .blocks (baseline ViT) or mixed blocks
        blocks = self._find_blocks()
        for block in blocks:
            attn = self._find_attn_module(block)
            if attn is not None:
                hook = attn.register_forward_hook(self._capture_attn_hook())
                self._hooks.append(hook)

    def _capture_attn_hook(self):
        """Return a hook fn that captures the attention weight matrix."""
        def hook(module, inp, out):
            # inp[0] is the normalized input (B, N, C)
            x = inp[0]
            B, N, C = x.shape
            # Re-compute QKV to get attention weights
            if hasattr(module, 'qkv'):
                qkv = module.qkv(x)
                qkv = qkv.reshape(B, N, 3, module.num_heads,
                                  C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)  # (B, H, N, head_dim)
                scale = (C // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)  # (B, H, N, N)
                self._attention_maps.append(attn.detach().cpu())
        return hook

    def _find_blocks(self):
        """Find all transformer blocks in the model."""
        blocks = []
        # Standard ViT: model.blocks
        if hasattr(self.model, 'blocks'):
            blocks = list(self.model.blocks)
        # MedDef wrapper: model.model.blocks
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            blocks = list(self.model.model.blocks)
        return blocks

    def _find_attn_module(self, block):
        """Find the attention module within a transformer block."""
        if hasattr(block, 'attn'):
            return block.attn
        return None

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute attention rollout map.

        Parameters
        ----------
        input_tensor : (1, 3, H, W) tensor

        Returns
        -------
        (H, W) heatmap in [0, 1], showing spatial attention from CLS token
        """
        self._register_hooks()
        self._attention_maps.clear()

        device = next(self.model.parameters()).device
        x = input_tensor.to(device)

        with torch.no_grad():
            _ = self.model(x)

        self._remove_hooks()

        if not self._attention_maps:
            print("  [WARN] No attention maps captured — returning uniform map")
            return np.ones((IMG_SIZE, IMG_SIZE)) * 0.5

        # Rollout: multiply attention matrices across layers
        # Each attn: (1, H, N, N) where N = num_patches+1
        result = torch.eye(self._attention_maps[0].shape[-1])  # (N, N)

        for attn in self._attention_maps:
            # Fuse heads
            attn = attn.squeeze(0)  # (H, N, N)
            if self.head_fusion == "mean":
                attn_fused = attn.mean(0)  # (N, N)
            elif self.head_fusion == "max":
                attn_fused = attn.max(0)[0]
            elif self.head_fusion == "min":
                attn_fused = attn.min(0)[0]
            else:
                attn_fused = attn.mean(0)

            # Discard low-attention tokens
            flat = attn_fused.flatten()
            threshold_idx = int(self.discard_ratio * flat.shape[0])
            threshold_val = flat.sort()[0][threshold_idx]
            attn_fused[attn_fused < threshold_val] = 0

            # Add residual connection (identity + attention) / 2
            I = torch.eye(attn_fused.shape[-1])
            a = (attn_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = a @ result

        # CLS token attention to all patches: result[0, 1:] (skip CLS token itself)
        cls_attn = result[0, 1:]  # (num_patches,)
        grid_size = int(cls_attn.shape[0] ** 0.5)
        heatmap = cls_attn.reshape(grid_size, grid_size).numpy()

        # Resize to image size
        heatmap = np.array(
            __import__('PIL').Image.fromarray(
                (heatmap * 255).astype(np.uint8)
            ).resize((IMG_SIZE, IMG_SIZE), __import__('PIL').Image.BILINEAR)
        ) / 255.0

        # Normalize
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)

        return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM on patch embedding conv (ViT-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class ViTGradCAM:
    """
    Grad-CAM for ViT using the patch embedding Conv2d projection as target layer.

    The patch embedding conv (16×16 stride 16) produces a feature map of shape
    (B, embed_dim, 14, 14) for a 224×224 input — this is the natural spatial
    feature map in a ViT and gives spatially interpretable Grad-CAM output.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations = []
        self._gradients = []

    def _find_patch_embed_conv(self) -> Optional[nn.Module]:
        """Find the patch embedding Conv2d in the model."""
        # Direct model
        for name, module in self.model.named_modules():
            if 'patch_embed' in name and isinstance(module, nn.Conv2d):
                return module
        return None

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap using the patch embedding projection.

        Returns (H, W) heatmap in [0, 1].
        """
        target_layer = self._find_patch_embed_conv()
        if target_layer is None:
            print("  [WARN] patch_embed Conv2d not found — using uniform heatmap")
            return np.ones((IMG_SIZE, IMG_SIZE)) * 0.5

        activations, gradients = [], []

        def fwd_hook(m, inp, out):
            activations.append(out.detach())

        def bwd_hook(m, gi, go):
            gradients.append(go[0].detach())

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        device = next(self.model.parameters()).device
        x = input_tensor.clone().to(device).requires_grad_(True)

        try:
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot)

            act = activations[0]   # (1, embed_dim, 14, 14)
            grad = gradients[0]    # (1, embed_dim, 14, 14)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
            )
            cam_np = cam.squeeze().cpu().detach().numpy()

            # Normalize
            cmin, cmax = cam_np.min(), cam_np.max()
            if cmax > cmin:
                cam_np = (cam_np - cmin) / (cmax - cmin)

            return cam_np

        except Exception as e:
            print(f"  [WARN] Grad-CAM failed: {e}")
            return np.ones((IMG_SIZE, IMG_SIZE)) * 0.5
        finally:
            h_fwd.remove()
            h_bwd.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a [0,1] heatmap on an RGB image using a colormap."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * original).astype(np.uint8)
    return blended


def add_prediction_bar(
    ax,
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: List[str],
):
    """Add predicted class + confidence as text below the image axis."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_label = class_names[pred_idx] if pred_idx < len(
        class_names) else str(pred_idx)
    conf = probs[pred_idx]
    ax.set_xlabel(f"Pred: {pred_label} ({conf:.1%})", fontsize=8, color="navy")


# ─────────────────────────────────────────────────────────────────────────────
# Main generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam_panel(
    models: Dict[str, nn.Module],
    samples: Dict[str, List[Path]],
    out_dir: Path,
    use_rollout: bool = True,
):
    """
    Create a per-class Grad-CAM / Attention Rollout panel.

    Layout: rows = variants, columns = sample images (per class)
    Produces one figure per class.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    out_dir.mkdir(parents=True, exist_ok=True)

    for cls_name, img_paths in samples.items():
        if not img_paths:
            continue

        n_samples = len(img_paths)
        n_variants = len(VARIANTS)
        # Columns: original | gradcam | rollout  (×n_samples)
        n_methods = 2  # GradCAM + Attention Rollout
        cols_per_sample = n_methods + 1  # original + 2 heatmaps
        total_cols = n_samples * cols_per_sample

        fig, axes = plt.subplots(
            n_variants, total_cols,
            figsize=(total_cols * 2.2, n_variants * 2.5),
            dpi=150,
        )

        if n_variants == 1:
            axes = axes[np.newaxis, :]
        if total_cols == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(
            f"MedDef-VISTA — Grad-CAM & Attention Rollout | Class: {cls_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for row_idx, variant in enumerate(VARIANTS):
            model = models.get(variant)
            if model is None:
                continue

            gradcam_gen = ViTGradCAM(model)
            rollout_gen = AttentionRollout(
                model, head_fusion="mean", discard_ratio=0.9)

            for si, img_path in enumerate(img_paths):
                tensor, original = load_image(img_path)
                col_base = si * cols_per_sample

                # --- original image ---
                ax_orig = axes[row_idx, col_base]
                ax_orig.imshow(original)
                ax_orig.axis("off")
                if row_idx == 0:
                    ax_orig.set_title(
                        f"Original\n{img_path.stem[:12]}", fontsize=7)
                if si == 0:
                    ax_orig.set_ylabel(VARIANT_LABELS[variant], fontsize=8, rotation=90,
                                       labelpad=4, va="center")

                # --- Grad-CAM ---
                ax_gc = axes[row_idx, col_base + 1]
                heatmap_gc = gradcam_gen(tensor)
                overlay_gc = overlay_heatmap(
                    original, heatmap_gc, alpha=0.45, colormap="jet")
                ax_gc.imshow(overlay_gc)
                ax_gc.axis("off")
                add_prediction_bar(ax_gc, model, tensor, CLASS_NAMES)
                if row_idx == 0:
                    ax_gc.set_title("Grad-CAM\n(Patch Proj)", fontsize=7)

                # --- Attention Rollout ---
                if use_rollout:
                    ax_ar = axes[row_idx, col_base + 2]
                    heatmap_ar = rollout_gen(tensor)
                    overlay_ar = overlay_heatmap(
                        original, heatmap_ar, alpha=0.45, colormap="inferno")
                    ax_ar.imshow(overlay_ar)
                    ax_ar.axis("off")
                    if row_idx == 0:
                        ax_ar.set_title(
                            "Attn Rollout\n(ViT-native)", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        out_path = out_dir / f"gradcam_panel_{cls_name.lower()}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def generate_per_variant_gradcam(
    models: Dict[str, nn.Module],
    samples: Dict[str, List[Path]],
    out_dir: Path,
):
    """
    Compact per-variant figure: 2 rows (Normal / TB), n_samples columns.
    One figure per variant with Grad-CAM + Attention Rollout side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = [(cls, p) for cls, paths in samples.items() for p in paths]
    n_imgs = len(all_paths)
    if n_imgs == 0:
        print("  [WARN] No sample images found")
        return

    for variant in VARIANTS:
        model = models.get(variant)
        if model is None:
            continue

        gradcam_gen = ViTGradCAM(model)
        rollout_gen = AttentionRollout(
            model, head_fusion="mean", discard_ratio=0.9)

        # Layout: 3 rows per image group (original, gradcam, rollout), n_imgs columns
        rows = 3
        fig, axes = plt.subplots(rows, n_imgs, figsize=(
            n_imgs * 2.5, rows * 2.5), dpi=150)

        if n_imgs == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(
            f"{VARIANT_LABELS[variant]} — Grad-CAM & Attention Rollout",
            fontsize=12, fontweight="bold",
        )

        row_labels = ["Original",
                      "Grad-CAM\n(Patch Conv)", "Attn Rollout\n(ViT-native)"]
        for r, lbl in enumerate(row_labels):
            axes[r, 0].set_ylabel(
                lbl, fontsize=9, rotation=90, labelpad=4, va="center")

        for col_idx, (cls_name, img_path) in enumerate(all_paths):
            tensor, original = load_image(img_path)

            # Original
            axes[0, col_idx].imshow(original)
            axes[0, col_idx].axis("off")
            axes[0, col_idx].set_title(
                f"{cls_name}\n{img_path.stem[:10]}", fontsize=7)

            # Grad-CAM
            hm_gc = gradcam_gen(tensor)
            ov_gc = overlay_heatmap(
                original, hm_gc, alpha=0.45, colormap="jet")
            axes[1, col_idx].imshow(ov_gc)
            axes[1, col_idx].axis("off")
            add_prediction_bar(axes[1, col_idx], model, tensor, CLASS_NAMES)

            # Attention Rollout
            hm_ar = rollout_gen(tensor)
            ov_ar = overlay_heatmap(
                original, hm_ar, alpha=0.45, colormap="inferno")
            axes[2, col_idx].imshow(ov_ar)
            axes[2, col_idx].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"gradcam_{variant}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def generate_tsne(
    models: Dict[str, nn.Module],
    data_dir: Path,
    out_dir: Path,
    device: str,
    max_samples: int = 1000,
):
    """
    Generate t-SNE plots for all 6 variants.
    - One figure per variant
    - One combined 2×3 panel comparing all variants
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    from sklearn.manifold import TSNE

    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = data_dir / "val"
    if not val_dir.exists():
        val_dir = data_dir / "test"
    if not val_dir.exists():
        print(f"  [WARN] No val/test dir at {data_dir}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    class_names_folder = dataset.classes  # from folder structure
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=4, drop_last=False)

    # Color palette for classes
    palette = ["#2196F3", "#F44336"]   # blue=Normal, red=TB

    # Store embeddings per variant for combined figure
    all_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def extract_features(model, loader, max_n) -> Tuple[np.ndarray, np.ndarray]:
        """Extract penultimate-layer features via hook on norm layer."""
        features_list, labels_list = [], []
        collected = 0

        feat_buf = []

        def hook_fn(module, inp, out):
            # CLS token feature: out[:, 0, :] for LayerNorm after all blocks
            if out.ndim == 3:
                feat_buf.append(out[:, 0, :].detach().cpu())
            else:
                feat_buf.append(out.detach().cpu())

        # Find the final norm layer (after all transformer blocks)
        target = _find_norm_layer(model)
        hook = None
        if target is not None:
            hook = target.register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                _ = model(imgs)
                if feat_buf:
                    feat = feat_buf[-1]
                    if feat.ndim > 2:
                        feat = feat.flatten(1)
                    features_list.append(feat.numpy())
                    labels_list.append(labels.numpy())
                    feat_buf.clear()
                    collected += feat.shape[0]
                    if collected >= max_n:
                        break

        if hook:
            hook.remove()

        if not features_list:
            return np.array([]), np.array([])

        features = np.concatenate(features_list)[:max_n]
        labels = np.concatenate(labels_list)[:max_n]
        return features, labels

    def _find_norm_layer(model):
        """Find LayerNorm after all transformer blocks."""
        # Try model.norm (direct ViT)
        if hasattr(model, 'norm') and isinstance(model.norm, nn.LayerNorm):
            return model.norm
        # Try model.model.norm (wrapped)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            return model.model.norm
        # Fallback: last LayerNorm in the model
        last_ln = None
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                last_ln = m
        return last_ln

    def plot_tsne_single(features, labels, title, out_path, class_names_local, palette_local):
        """Fit t-SNE and plot a single figure."""
        if len(features) < 10:
            print(f"  [WARN] Too few features for t-SNE: {len(features)}")
            return

        print(f"  Fitting t-SNE for {title} ({len(features)} samples)...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 4),
                    random_state=42, n_iter=1000, init="pca", learning_rate="auto")
        embedded = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for ci, cls in enumerate(class_names_local):
            mask = labels == ci
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=palette_local[ci % len(palette_local)],
                label=cls, alpha=0.65, s=18, edgecolors="none",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.legend(fontsize=9, markerscale=1.5)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_path}")
        return embedded

    # Per-variant t-SNE
    for variant in VARIANTS:
        model = models.get(variant)
        if model is None:
            continue
        print(f"  Extracting features: {variant}")
        features, labels = extract_features(model, loader, max_samples)
        if features.size == 0:
            print(f"  [WARN] No features for {variant}")
            continue
        all_embeddings[variant] = (features, labels)
        plot_tsne_single(
            features, labels,
            title=f"{VARIANT_LABELS[variant]}",
            out_path=out_dir / f"tsne_{variant}.png",
            class_names_local=class_names_folder,
            palette_local=palette,
        )

    # Combined 2×3 panel
    if len(all_embeddings) > 0:
        print("  Generating combined t-SNE panel...")
        from sklearn.manifold import TSNE

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
        axes = axes.flatten()
        fig.suptitle("MedDef-VISTA — Feature Embeddings (t-SNE) — All Variants",
                     fontsize=13, fontweight="bold")

        for ax_idx, variant in enumerate(VARIANTS):
            ax = axes[ax_idx]
            if variant not in all_embeddings:
                ax.set_visible(False)
                continue

            features, labels = all_embeddings[variant]
            print(f"    Fitting t-SNE for combined panel: {variant}")
            tsne = TSNE(n_components=2,
                        perplexity=min(30, len(features) // 4),
                        random_state=42, n_iter=1000, init="pca", learning_rate="auto")
            embedded = tsne.fit_transform(features)

            for ci, cls in enumerate(class_names_folder):
                mask = labels == ci
                ax.scatter(
                    embedded[mask, 0], embedded[mask, 1],
                    c=palette[ci % len(palette)],
                    label=cls, alpha=0.65, s=14, edgecolors="none",
                )

            ax.set_title(VARIANT_LABELS[variant],
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("dim 1", fontsize=8)
            ax.set_ylabel("dim 2", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8, markerscale=1.2)

            # Compute & annotate class separation (silhouette score)
            try:
                from sklearn.metrics import silhouette_score
                sil = silhouette_score(embedded, labels)
                ax.annotate(f"Silhouette: {sil:.3f}", xy=(0.02, 0.02),
                            xycoords="axes fraction", fontsize=7,
                            color="gray", style="italic")
            except Exception:
                pass

        plt.tight_layout()
        out_path = out_dir / "tsne_all_variants_combined.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MedDef-VISTA: Grad-CAM + t-SNE visualizations")
    p.add_argument(
        "--data",
        default="/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/tbcr",
        help="Path to TBCR dataset root (must have val/ or test/ subdirs)",
    )
    p.add_argument(
        "--weights-base",
        default="/data2/enoch/ekd_coding_env/ultralytics/runs/classify/train_tbcr_final/tbcr",
        help="Base directory containing {variant}_small/distill_v2/weights/best.pt",
    )
    p.add_argument(
        "--out",
        default="/data2/enoch/ekd_coding_env/ultralytics/runs/visualizations/gradcam_tsne",
        help="Output directory for all generated figures",
    )
    p.add_argument("--device", default="0", help="CUDA device index (e.g. 0)")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of images per class for Grad-CAM (default: 4)")
    p.add_argument("--max-tsne", type=int, default=800,
                   help="Max samples for t-SNE per variant (default: 800)")
    p.add_argument("--skip-gradcam", action="store_true",
                   help="Skip Grad-CAM generation")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE generation")
    p.add_argument("--variants", nargs="+", default=VARIANTS,
                   choices=VARIANTS, help="Which variants to process")
    return p.parse_args()


def main():
    args = parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if not torch.cuda.is_available():
        print("  [WARN] CUDA not available — running on CPU (will be slow)")

    data_dir = Path(args.data)
    weights_base = Path(args.weights_base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:         {data_dir}")
    print(f"Weights base: {weights_base}")
    print(f"Output:       {out_dir}")

    # ── Load all models ──────────────────────────────────────────────────────
    print("\nLoading models...")
    models: Dict[str, nn.Module] = {}
    for variant in args.variants:
        w = find_weights(weights_base, variant)
        if w is None:
            continue
        try:
            print(f"  Loading {variant}: {w}")
            m = load_model(w, device)
            models[variant] = m
            print(
                f"    OK — {sum(p.numel() for p in m.parameters()):,} params")
        except Exception as e:
            print(f"  [ERROR] Failed to load {variant}: {e}")

    if not models:
        print("[ERROR] No models loaded. Exiting.")
        sys.exit(1)

    # ── Collect sample images ────────────────────────────────────────────────
    val_dir = data_dir / "val"
    if not val_dir.exists():
        val_dir = data_dir / "test"
    if not val_dir.exists():
        print(f"[ERROR] No val/test directory at {data_dir}")
        sys.exit(1)

    samples = collect_samples(val_dir, CLASS_NAMES, n_per_class=args.n_samples)
    print(f"\nSamples collected:")
    for cls, paths in samples.items():
        print(f"  {cls}: {len(paths)} images")

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    if not args.skip_gradcam:
        print("\n=== Generating Grad-CAM & Attention Rollout ===")

        # Per-variant compact figures
        gc_dir = out_dir / "gradcam"
        generate_per_variant_gradcam(models, samples, gc_dir)

        # Cross-variant comparison panels (one per class)
        panel_dir = out_dir / "gradcam_panel"
        generate_gradcam_panel(models, samples, panel_dir, use_rollout=True)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    if not args.skip_tsne:
        print("\n=== Generating t-SNE embeddings ===")
        tsne_dir = out_dir / "tsne"
        generate_tsne(models, data_dir, tsne_dir,
                      device, max_samples=args.max_tsne)

    print(f"\nAll done. Results saved to: {out_dir}")
    print("To copy back locally:")
    print(f"  scp -P 8822 -r enoch@ci2p:{out_dir} ./latex/mdpi/fig/")


if __name__ == "__main__":
    main()
