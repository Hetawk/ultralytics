# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Medical Image-Specific Data Augmentation Modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class MedicalDataAugmentation(nn.Module):
    """
    Medical Image-Specific Data Augmentation Module
    
    Implements augmentations specifically designed for medical imaging:
    - Elastic deformation (simulates tissue deformation)
    - Intensity transformations (simulates imaging variations)
    - Anatomical consistency preserving transforms
    
    These augmentations improve robustness to real-world variations
    while maintaining anatomical plausibility.
    """

    def __init__(self, 
                 elastic_alpha: float = 50.0,
                 elastic_sigma: float = 5.0,
                 intensity_shift_range: float = 0.1,
                 intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
                 gamma_range: Tuple[float, float] = (0.8, 1.2),
                 gaussian_noise_std: float = 0.02,
                 p: float = 0.5):
        """
        Args:
            elastic_alpha: Elasticity intensity
            elastic_sigma: Gaussian smoothing sigma for displacement
            intensity_shift_range: Range for additive intensity shift
            intensity_scale_range: Range for multiplicative intensity scaling
            gamma_range: Range for gamma correction
            gaussian_noise_std: Standard deviation of Gaussian noise
            p: Probability of applying each augmentation
        """
        super().__init__()
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.gamma_range = gamma_range
        self.gaussian_noise_std = gaussian_noise_std
        self.p = p

    def _elastic_deformation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply elastic deformation to simulate tissue deformation.
        Common in CT, MRI, and ultrasound imaging.
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Generate random displacement fields
        dx = torch.randn(B, 1, H, W, device=device) * self.elastic_alpha
        dy = torch.randn(B, 1, H, W, device=device) * self.elastic_alpha
        
        # Smooth displacement fields with Gaussian
        kernel_size = int(4 * self.elastic_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * self.elastic_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(1, 1, -1, -1)
        
        # Apply Gaussian smoothing
        padding = kernel_size // 2
        dx = F.conv2d(dx, kernel_2d, padding=padding)
        dy = F.conv2d(dy, kernel_2d, padding=padding)
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add displacement
        displacement = torch.cat([
            dx.permute(0, 2, 3, 1) / (W / 2),
            dy.permute(0, 2, 3, 1) / (H / 2)
        ], dim=-1)
        
        new_grid = grid + displacement
        
        # Apply grid sampling
        return F.grid_sample(x, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _intensity_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply intensity transformations to simulate imaging variations.
        Common in different scanner/protocol settings.
        """
        B = x.shape[0]
        device = x.device
        
        # Random intensity shift
        if torch.rand(1).item() < self.p:
            shift = torch.empty(B, 1, 1, 1, device=device).uniform_(
                -self.intensity_shift_range, self.intensity_shift_range
            )
            x = x + shift
        
        # Random intensity scaling
        if torch.rand(1).item() < self.p:
            scale = torch.empty(B, 1, 1, 1, device=device).uniform_(
                self.intensity_scale_range[0], self.intensity_scale_range[1]
            )
            x = x * scale
        
        return torch.clamp(x, 0, 1)

    def _gamma_correction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gamma correction to simulate display/sensor variations.
        """
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(
                self.gamma_range[0], self.gamma_range[1]
            ).item()
            x = torch.pow(x.clamp(min=1e-8), gamma)
        return x

    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor noise."""
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.gaussian_noise_std
            x = x + noise
        return torch.clamp(x, 0, 1)

    def _add_speckle_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add speckle noise (multiplicative) common in ultrasound."""
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * 0.1
            x = x * (1 + noise)
        return torch.clamp(x, 0, 1)

    def forward(self, x: torch.Tensor, augment_types: Optional[List[str]] = None) -> torch.Tensor:
        """
        Apply medical image augmentations.
        
        Args:
            x: Input images [B, C, H, W]
            augment_types: List of augmentation types to apply.
                          Options: 'elastic', 'intensity', 'gamma', 'gaussian', 'speckle', 'all'
                          Default: ['all']
        
        Returns:
            Augmented images
        """
        if augment_types is None:
            augment_types = ['all']
        
        if 'all' in augment_types:
            augment_types = ['elastic', 'intensity', 'gamma', 'gaussian']
        
        if 'elastic' in augment_types and torch.rand(1).item() < self.p:
            x = self._elastic_deformation(x)
        
        if 'intensity' in augment_types:
            x = self._intensity_transform(x)
        
        if 'gamma' in augment_types:
            x = self._gamma_correction(x)
        
        if 'gaussian' in augment_types:
            x = self._add_gaussian_noise(x)
        
        if 'speckle' in augment_types:
            x = self._add_speckle_noise(x)
        
        return x


class CutMixMedical(nn.Module):
    """
    CutMix augmentation adapted for medical images.
    
    Cuts and pastes patches between training images, mixing labels proportionally.
    Helps model learn local features and reduces overfitting.
    
    Reference: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers"
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying CutMix
        """
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for cutout."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B]
            
        Returns:
            Tuple of (mixed_x, mixed_y, lambda)
        """
        if np.random.rand() > self.prob:
            return x, y, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(x.size()[0])
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        
        x_mixed = x.clone()
        x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        # Return mixed labels for soft target training
        y_mixed = lam * F.one_hot(y, num_classes=-1).float() + \
                  (1 - lam) * F.one_hot(y[rand_index], num_classes=-1).float()
        
        return x_mixed, y_mixed, lam


class MixUpMedical(nn.Module):
    """
    MixUp augmentation for medical images.
    
    Linearly interpolates between training samples and their labels.
    
    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (smaller = less mixing)
            prob: Probability of applying MixUp
        """
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, x: torch.Tensor, y: torch.Tensor, 
                num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B]
            num_classes: Number of classes
            
        Returns:
            Tuple of (mixed_x, mixed_y, lambda)
        """
        if np.random.rand() > self.prob:
            return x, F.one_hot(y, num_classes).float(), 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(x.size()[0])
        
        x_mixed = lam * x + (1 - lam) * x[rand_index]
        y_mixed = lam * F.one_hot(y, num_classes).float() + \
                  (1 - lam) * F.one_hot(y[rand_index], num_classes).float()
        
        return x_mixed, y_mixed, lam


__all__ = [
    'MedicalDataAugmentation',
    'CutMixMedical',
    'MixUpMedical',
]
