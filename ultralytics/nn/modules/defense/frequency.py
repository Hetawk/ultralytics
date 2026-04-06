# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frequency Domain Defense Module."""

import torch
import torch.nn as nn
import torch.fft as fft


class FrequencyDefense(nn.Module):
    """
    Frequency domain defense mechanism to suppress high-frequency adversarial noise.
    Works by applying low-pass filtering in frequency domain to suppress adversarial perturbations.
    
    This defense is effective against GRADIENT-BASED (White-Box) attacks:
    - FGSM (FastGradientMethod): Single-step gradient perturbations
    - PGD (ProjectedGradientDescent): Iterative gradient attacks
    - BIM (BasicIterativeMethod): Basic iterative method
    - MIM (MomentumIterativeMethod): Momentum-enhanced attacks
    - C&W (CarliniL2Method, CarliniLInfMethod): Optimization-based attacks
    - DeepFool: Minimal perturbation attacks
    
    Also provides partial defense against:
    - SquareAttack: Score-based black-box attack
    - AutoAttack: Ensemble attack evaluation
    - UniversalPerturbation: Image-agnostic perturbations
    
    Note: Less effective against decision-based black-box attacks (BoundaryAttack, 
    HopSkipJump) which don't rely on high-frequency perturbations.
    
    Reference: Integrates with ART's preprocessing defenses (JpegCompression, etc.)
    
    Attributes:
        cutoff_ratio (float): Ratio of frequencies to keep (0 to 1). Lower values = more aggressive filtering.
        adaptive (bool): Whether to use adaptive frequency cutoff based on input statistics.
    """

    def __init__(self, cutoff_ratio: float = 0.5):
        """
        Args:
            cutoff_ratio (float): Proportion of low frequencies to keep. Default 0.5 keeps center 50% of frequencies.
        """
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply low-pass filtering in the frequency domain.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Filtered tensor of the same shape
        """
        # Handle sequence format [B, N, D] from transformers
        if x.ndim == 3:
            B, N, D = x.shape
            # For now, skip frequency filtering for sequences
            # Could be extended to apply on flattened spatial features
            return x
        
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # Compute FFT
        x_float = x.float()
        x_freq = fft.fftn(x_float, dim=(-2, -1))
        
        # Shift zero frequency to center
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        
        # Compute cutoff frequencies
        cutoff_h = int(H * self.cutoff_ratio)
        cutoff_w = int(W * self.cutoff_ratio)
        
        # Create a mask to keep low frequencies
        mask = torch.zeros_like(x_freq, dtype=torch.bool, device=device)
        center_h, center_w = H // 2, W // 2
        
        h_start = max(0, center_h - cutoff_h // 2)
        h_end = min(H, center_h + cutoff_h // 2)
        w_start = max(0, center_w - cutoff_w // 2)
        w_end = min(W, center_w + cutoff_w // 2)
        
        mask[..., h_start:h_end, w_start:w_end] = True
        
        # Apply mask
        x_freq_filtered = x_freq * mask.float()
        
        # Shift back
        x_freq_filtered = fft.ifftshift(x_freq_filtered, dim=(-2, -1))
        
        # Inverse FFT
        x_filtered = fft.ifftn(x_freq_filtered, dim=(-2, -1)).real
        
        return x_filtered.to(dtype=dtype)


__all__ = ['FrequencyDefense']
