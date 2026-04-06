# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Input-space Preprocessing Defenses for MedDef models.

Ported / inspired from the Adversarial Robustness Toolbox (ART, MIT License)
https://github.com/Trusted-AI/adversarial-robustness-toolbox

These are *pure PyTorch* implementations — no ART dependency required.
They can be composed freely as nn.Module layers and inserted at any point in
the forward pipeline, or used as inference-time transforms.

Classes
-------
SpatialSmoothing       – Median / mean filter (ART: SpatialSmoothing)
FeatureSqueezing       – Bit-depth reduction  (ART: FeatureSqueezing)
JpegCompression        – JPEG round-trip      (ART: JpegCompression)
GaussianAugmentation   – Additive Gaussian noise (ART: GaussianAugmentation)
PixelDefend            – Pixel defence via projection to clean manifold
VarianceMinimization   – Variance-minimisation smoothing (ART: VarianceMinimization)
InputTransformPipeline – Composable pipeline of the above

Effectiveness reference (matches ART DEFENSE_EFFECTIVENESS_MATRIX):
  SpatialSmoothing  : FGSM +++, PGD ++, Patch +++
  FeatureSqueezing  : FGSM +++, PGD ++, C&W ++
  JpegCompression   : FGSM ++,  PGD +,  Patch +
  GaussianAugmentation: Black-box +++, Transfer ++
"""

from __future__ import annotations

import io
import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "SpatialSmoothing",
    "FeatureSqueezing",
    "JpegCompression",
    "GaussianAugmentation",
    "PixelDefend",
    "VarianceMinimization",
    "InputTransformPipeline",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial Smoothing  (ART: SpatialSmoothing / spatial_smoothing.py)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialSmoothing(nn.Module):
    """Local median-filter smoothing defence.

    Perturbs adversarial high-frequency noise injected by FGSM / PGD by
    applying a sliding median window over every spatial neighbourhood.

    Reference: Xu et al. "Feature Squeezing" (2018) – §spatial smoothing.
    https://arxiv.org/abs/1704.01155

    Args:
        window_size (int): Median filter kernel size (odd). Default 3.
        mode (str): 'median' (default) or 'mean'.
        clip_min (float): Clamp output minimum.
        clip_max (float): Clamp output maximum.
        apply_train (bool): Apply during training forward pass.
        apply_eval  (bool): Apply during eval / inference.
    """

    def __init__(
        self,
        window_size: int = 3,
        mode: str = "median",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = False,
        apply_eval: bool = True,
    ):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        self.window_size = window_size
        self.mode = mode
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing to [B, C, H, W] input."""
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        pad = self.window_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        B, C, H, W = x.shape
        k = self.window_size

        # Unfold into patches: [B, C, H, W, k, k]
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)  # [B, C, H, W, k, k]
        patches = patches.contiguous().view(B, C, H, W, k * k)

        if self.mode == "median":
            result = patches.median(dim=-1).values
        else:
            result = patches.mean(dim=-1)

        return torch.clamp(result, self.clip_min, self.clip_max)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Squeezing  (ART: FeatureSqueezing / feature_squeezing.py)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSqueezing(nn.Module):
    """Bit-depth reduction defence.

    Quantises pixel intensities to a reduced number of bits, collapsing small
    adversarial perturbations below the quantisation step.

    Reference: Xu et al. "Feature Squeezing" (2018).
    https://arxiv.org/abs/1704.01155

    Args:
        bit_depth (int): Number of bits per channel (1–8). Default 4.
        clip_min (float): Input lower bound. Default 0.0.
        clip_max (float): Input upper bound. Default 1.0.
        apply_train (bool): Apply during training.
        apply_eval  (bool): Apply during inference.
    """

    def __init__(
        self,
        bit_depth: int = 4,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = False,
        apply_eval: bool = True,
    ):
        super().__init__()
        if not (1 <= bit_depth <= 8):
            raise ValueError("bit_depth must be between 1 and 8.")
        self.bit_depth = bit_depth
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_val = float(2 ** bit_depth - 1)
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        # Normalise to [0, 1]
        x_norm = (x - self.clip_min) / (self.clip_max - self.clip_min)
        # Quantise
        x_quant = torch.round(x_norm * self.max_val) / self.max_val
        # Scale back
        x_out = x_quant * (self.clip_max - self.clip_min) + self.clip_min
        return torch.clamp(x_out, self.clip_min, self.clip_max)


# ─────────────────────────────────────────────────────────────────────────────
# 3. JPEG Compression  (ART: JpegCompression / jpeg_compression.py)
# ─────────────────────────────────────────────────────────────────────────────

class JpegCompression(nn.Module):
    """JPEG compression round-trip defence.

    Applies JPEG encode → decode to each image, removing high-frequency
    adversarial noise that is invisible to the human eye.

    Reference: Dziugaite et al. (2016), Guo et al. (2018).
    https://arxiv.org/abs/1705.02900

    Args:
        quality (int): JPEG quality factor 1–95. Default 75.
        clip_min (float): Input lower bound. Default 0.0.
        clip_max (float): Input upper bound. Default 1.0.
        apply_train (bool): Apply during training.
        apply_eval  (bool): Apply during inference.
    """

    def __init__(
        self,
        quality: int = 75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = True,
        apply_eval: bool = True,
    ):
        super().__init__()
        self.quality = quality
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    def _compress_one(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """JPEG round-trip for a single [C, H, W] tensor."""
        try:
            from PIL import Image
            import numpy as np

            c, h, w = img_tensor.shape
            # Scale to [0, 255]
            arr = img_tensor.detach().cpu().float()
            arr = (arr - self.clip_min) / (self.clip_max - self.clip_min)
            arr = (arr * 255).clamp(0, 255).byte().numpy()

            if c == 3:
                pil = Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
            else:
                # Compress each channel independently
                channels = []
                for ci in range(c):
                    buf = io.BytesIO()
                    ch_img = Image.fromarray(arr[ci], mode="L")
                    ch_img.save(buf, format="JPEG", quality=self.quality)
                    buf.seek(0)
                    channels.append(np.array(Image.open(buf)))
                result = np.stack(channels, axis=0).astype(np.float32) / 255.0
                result = result * (self.clip_max - self.clip_min) + self.clip_min
                return torch.from_numpy(result).to(img_tensor.device).to(img_tensor.dtype)

            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            arr_back = np.array(Image.open(buf)).astype(np.float32) / 255.0
            arr_back = arr_back.transpose(2, 0, 1)  # HWC → CHW
            arr_back = arr_back * (self.clip_max - self.clip_min) + self.clip_min
            return torch.from_numpy(arr_back).to(img_tensor.device).to(img_tensor.dtype)

        except ImportError:
            logger.warning("JpegCompression: Pillow not available — skipping compression.")
            return img_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        return torch.stack([self._compress_one(img) for img in x])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gaussian Augmentation  (ART: GaussianAugmentation / gaussian_augmentation.py)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianAugmentation(nn.Module):
    """Additive Gaussian noise augmentation / defence.

    At *training* time: adds random noise to improve robustness against small
    perturbations a la Cohen et al. (certified smoothing).
    At *inference* time: optionally applies noise for stochastic smoothing.

    Reference: Cohen et al. "Certified Adversarial Robustness via Randomised Smoothing" (2019).

    Args:
        sigma (float): Standard deviation of Gaussian noise. Default 0.05.
        clip_min (float): Clamp output minimum. Default 0.0.
        clip_max (float): Clamp output maximum. Default 1.0.
        apply_train (bool): Add noise during training. Default True.
        apply_eval  (bool): Add noise during inference. Default False.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = True,
        apply_eval: bool = False,
    ):
        super().__init__()
        self.sigma = sigma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, self.clip_min, self.clip_max)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pixel Defend  (ART: PixelDefend / pixel_defend.py)
# ─────────────────────────────────────────────────────────────────────────────

class PixelDefend(nn.Module):
    """Pixel Defence — project adversarial images onto the clean data manifold.

    Implemented here as a lightweight total-variation denoising pass, which
    approximates the projection in Song et al. (2018) without requiring a
    trained PixelCNN. Suitable as a fast, differentiable approximation.

    Reference: Song et al. "PixelDefend: Leveraging Generative Models to Understand
    and Defend against Adversarial Examples" (2018). https://arxiv.org/abs/1710.10766

    Args:
        tv_weight (float): TV regularisation weight. Default 0.05.
        num_iters (int): Gradient descent iterations. Default 5.
        lr (float): Learning rate for projection. Default 0.1.
        clip_min (float): Clamp minimum. Default 0.0.
        clip_max (float): Clamp maximum. Default 1.0.
        apply_train (bool): Apply during training. Default False.
        apply_eval  (bool): Apply during inference. Default True.
    """

    def __init__(
        self,
        tv_weight: float = 0.05,
        num_iters: int = 5,
        lr: float = 0.1,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = False,
        apply_eval: bool = True,
    ):
        super().__init__()
        self.tv_weight = tv_weight
        self.num_iters = num_iters
        self.lr = lr
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        """Compute isotropic total variation of [B, C, H, W] tensor."""
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
        return diff_h.mean() + diff_w.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        x_def = x.detach().clone()
        x_def.requires_grad_(True)

        for _ in range(self.num_iters):
            tv = self._total_variation(x_def)
            recon = F.mse_loss(x_def, x.detach())
            loss = recon + self.tv_weight * tv
            grad = torch.autograd.grad(loss, x_def)[0]
            x_def = (x_def - self.lr * grad).clamp(self.clip_min, self.clip_max).detach()
            x_def.requires_grad_(True)

        return x_def.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Variance Minimisation  (ART: VarianceMinimization)
# ─────────────────────────────────────────────────────────────────────────────

class VarianceMinimization(nn.Module):
    """Smooth each patch to the mean of its local neighbourhood.

    Reduces local pixel variance introduced by adversarial noise, effective
    against high-frequency perturbations (FGSM, BIM).

    Reference: Guo et al. "Countering Adversarial Images using Input Transformations" (2018).

    Args:
        window_size (int): Neighbourhood size. Default 3.
        clip_min (float): Clamp minimum. Default 0.0.
        clip_max (float): Clamp maximum. Default 1.0.
        apply_train (bool): Apply during training. Default False.
        apply_eval  (bool): Apply during inference. Default True.
    """

    def __init__(
        self,
        window_size: int = 3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        apply_train: bool = False,
        apply_eval: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_train = apply_train
        self.apply_eval = apply_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and not self.apply_train) or (not self.training and not self.apply_eval):
            return x

        B, C, H, W = x.shape
        k = self.window_size
        pad = k // 2

        # Average pooling approximates local mean smoothing
        kernel = torch.ones(C, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
        smoothed = F.conv2d(
            F.pad(x, (pad, pad, pad, pad), mode="reflect"),
            kernel,
            groups=C,
        )
        return torch.clamp(smoothed, self.clip_min, self.clip_max)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Composable Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class InputTransformPipeline(nn.Module):
    """Compose multiple input-space defences as a sequential pipeline.

    Example — combine feature squeezing + JPEG at inference:
    ```python
    pipeline = InputTransformPipeline([
        FeatureSqueezing(bit_depth=5),
        JpegCompression(quality=70),
        SpatialSmoothing(window_size=3),
    ])
    model = nn.Sequential(pipeline, backbone)
    ```

    Args:
        transforms (list[nn.Module]): Ordered list of defence transforms.
    """

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    @classmethod
    def recommended_inference(
        cls,
        bit_depth: int = 5,
        jpeg_quality: int = 75,
        smooth_window: int = 3,
    ) -> "InputTransformPipeline":
        """Build the recommended inference-time preprocessing stack.

        Stacks: FeatureSqueezing → JpegCompression → SpatialSmoothing
        Effective against FGSM, PGD, and black-box transfer attacks.

        Args:
            bit_depth (int): Bit depth for feature squeezing.
            jpeg_quality (int): JPEG quality factor.
            smooth_window (int): Spatial smoothing window size.
        """
        return cls([
            FeatureSqueezing(bit_depth=bit_depth, apply_train=False, apply_eval=True),
            JpegCompression(quality=jpeg_quality, apply_train=False, apply_eval=True),
            SpatialSmoothing(window_size=smooth_window, apply_train=False, apply_eval=True),
        ])

    @classmethod
    def recommended_training(cls, sigma: float = 0.05) -> "InputTransformPipeline":
        """Build the recommended training-time augmentation stack.

        Stacks: GaussianAugmentation → FeatureSqueezing
        Improves generalisation against unseen perturbations.

        Args:
            sigma (float): Gaussian noise standard deviation.
        """
        return cls([
            GaussianAugmentation(sigma=sigma, apply_train=True, apply_eval=False),
            FeatureSqueezing(bit_depth=6, apply_train=True, apply_eval=False),
        ])
