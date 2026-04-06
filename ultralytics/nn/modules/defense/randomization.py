# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Randomization-based Defense Techniques."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class InputRandomization(nn.Module):
    """
    Input Randomization Defense
    
    Applies random transformations to inputs before feeding to model,
    making it harder for adversaries to craft effective perturbations.
    
    Effective against:
    - Transfer attacks
    - Query-based attacks
    - Static adversarial examples
    
    Reference: Xie et al. "Mitigating Adversarial Effects Through Randomization"
    """

    def __init__(self, 
                 resize_range: Tuple[float, float] = (0.9, 1.1),
                 padding_range: Tuple[int, int] = (0, 4),
                 jpeg_quality_range: Tuple[int, int] = (70, 100),
                 gaussian_blur_range: Tuple[float, float] = (0.1, 1.0),
                 use_resize: bool = True,
                 use_padding: bool = True,
                 use_jpeg: bool = False,
                 use_blur: bool = True):
        """
        Args:
            resize_range: Range for random resize scale
            padding_range: Range for random padding
            jpeg_quality_range: Range for JPEG compression quality
            gaussian_blur_range: Range for Gaussian blur sigma
            use_*: Whether to use each transformation
        """
        super().__init__()
        self.resize_range = resize_range
        self.padding_range = padding_range
        self.jpeg_quality_range = jpeg_quality_range
        self.gaussian_blur_range = gaussian_blur_range
        self.use_resize = use_resize
        self.use_padding = use_padding
        self.use_jpeg = use_jpeg
        self.use_blur = use_blur

    def _random_resize(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Apply random resize."""
        scale = np.random.uniform(self.resize_range[0], self.resize_range[1])
        new_h = int(target_size[0] * scale)
        new_w = int(target_size[1] * scale)
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        # Resize back to original
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x

    def _random_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random padding."""
        pad = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            # Random crop back to original size
            _, _, H, W = x.shape
            start_h = np.random.randint(0, 2 * pad + 1)
            start_w = np.random.randint(0, 2 * pad + 1)
            x = x[:, :, start_h:start_h + H - 2*pad, start_w:start_w + W - 2*pad]
        return x

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random Gaussian blur."""
        sigma = np.random.uniform(self.gaussian_blur_range[0], self.gaussian_blur_range[1])
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel = kernel.expand(x.shape[1], 1, -1, -1)
        
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random transformations.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Transformed tensor
        """
        _, _, H, W = x.shape
        
        if self.use_resize:
            x = self._random_resize(x, (H, W))
        
        if self.use_padding:
            x = self._random_padding(x)
        
        if self.use_blur and np.random.rand() < 0.5:
            x = self._gaussian_blur(x)
        
        return torch.clamp(x, 0, 1)


class FeatureNoiseInjection(nn.Module):
    """
    Feature-level Noise Injection Defense
    
    Injects noise into intermediate feature representations,
    making the model more robust to small input perturbations.
    
    Can be applied:
    - At input layer (input noise)
    - At intermediate layers (feature noise)
    - At output layer (logit smoothing)
    """

    def __init__(self, noise_type: str = 'gaussian', 
                 noise_level: float = 0.1,
                 trainable: bool = False):
        """
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'dropout')
            noise_level: Magnitude of noise
            trainable: Whether noise level is learnable
        """
        super().__init__()
        self.noise_type = noise_type
        if trainable:
            self.noise_level = nn.Parameter(torch.tensor(noise_level))
        else:
            self.register_buffer('noise_level', torch.tensor(noise_level))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inject noise into features.
        
        Args:
            x: Input features
            
        Returns:
            Noisy features (only during training)
        """
        if not self.training:
            return x
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_level
            return x + noise
        elif self.noise_type == 'dropout':
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.noise_level))
            return x * mask / (1 - self.noise_level + 1e-12)
        else:
            return x


class StochasticDepthDefense(nn.Module):
    """
    Stochastic Depth as a Defense
    
    Randomly drops layers during training, creating an implicit ensemble
    of networks with different depths.
    
    Reference: Huang et al. "Deep Networks with Stochastic Depth"
    """

    def __init__(self, layers: nn.ModuleList, survival_prob: float = 0.8):
        """
        Args:
            layers: List of layers to apply stochastic depth to
            survival_prob: Probability of keeping each layer
        """
        super().__init__()
        self.layers = layers
        self.survival_prob = survival_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth during forward pass."""
        for layer in self.layers:
            if self.training:
                if torch.rand(1).item() < self.survival_prob:
                    x = layer(x) / self.survival_prob
            else:
                x = layer(x)
        return x


__all__ = [
    'InputRandomization',
    'FeatureNoiseInjection',
    'StochasticDepthDefense',
]
