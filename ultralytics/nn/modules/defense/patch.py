# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Patch Consistency Defense Module."""

import torch
import torch.nn as nn
from typing import Tuple


class PatchConsistency(nn.Module):
    """
    Patch consistency check to enforce smoothness across neighboring patches.
    Effective for Vision Transformers by ensuring patches have consistent representations.
    
    This defense works by:
    1. Computing differences between neighboring patches
    2. Identifying anomalous (adversarially perturbed) patches
    3. Smoothing them using neighboring patches
    
    Attributes:
        embed_dim (int): Embedding dimension
        grid_size (Tuple[int, int]): Spatial grid size (height, width)
        threshold (float): Anomaly detection threshold
        smooth_factor (float): Strength of smoothing (0=no smoothing, 1=full neighbor average)
    """

    def __init__(self, embed_dim: int, grid_size: Tuple[int, int], threshold: float = 1.0, smooth_factor: float = 0.5):
        """
        Args:
            embed_dim (int): Dimension of embeddings
            grid_size (Tuple[int, int]): Grid size as (height, width)
            threshold (float): Anomaly threshold for patch difference detection
            smooth_factor (float): How much to smooth anomalous patches
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.threshold = threshold
        self.register_buffer('smooth_factor', torch.tensor(smooth_factor, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch consistency check and smoothing.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, num_patches, embed_dim]
            
        Returns:
            torch.Tensor: Smoothed tensor of the same shape
        """
        # ONNX tracing cannot handle data-dependent control flow — pass through.
        # Defence is still active for PyTorch (.pt) inference and training.
        if torch.onnx.is_in_onnx_export():
            return x

        B, N, D = x.shape
        grid_h, grid_w = self.grid_size
        
        # Validate input
        if N != grid_h * grid_w:
            # If grid size doesn't match, try to infer it
            inferred_grid = int(N ** 0.5)
            if inferred_grid * inferred_grid == N:
                grid_h = grid_w = inferred_grid
            else:
                # Can't apply patch consistency, return as is
                return x
        
        # Reshape to [B, grid_h, grid_w, embed_dim]
        x_reshaped = x.view(B, grid_h, grid_w, D)
        x_smoothed = x_reshaped.clone()
        
        # Compute differences with neighboring patches
        diff_h = x_reshaped[:, :, 1:, :] - x_reshaped[:, :, :-1, :]  # Horizontal
        diff_v = x_reshaped[:, 1:, :, :] - x_reshaped[:, :-1, :, :]  # Vertical
        
        # Compute anomaly scores (L2 norm)
        anomaly_h = torch.norm(diff_h, dim=-1, keepdim=False)
        anomaly_v = torch.norm(diff_v, dim=-1, keepdim=False)
        
        # Smooth anomalous patches
        for i in range(1, grid_h - 1):
            for j in range(1, grid_w - 1):
                # Check if this patch or its neighbors are anomalous
                is_anomalous = False
                if j < anomaly_h.shape[2] and (anomaly_h[:, i, j-1] > self.threshold).any():
                    is_anomalous = True
                if j < anomaly_h.shape[2] and (anomaly_h[:, i, j] > self.threshold).any():
                    is_anomalous = True
                if i < anomaly_v.shape[1] and (anomaly_v[:, i-1, j] > self.threshold).any():
                    is_anomalous = True
                if i < anomaly_v.shape[1] and (anomaly_v[:, i, j] > self.threshold).any():
                    is_anomalous = True
                
                if is_anomalous:
                    # Average with neighbors
                    neighbors = torch.stack([
                        x_reshaped[:, i-1, j, :],  # Top
                        x_reshaped[:, i+1, j, :],  # Bottom
                        x_reshaped[:, i, j-1, :],  # Left
                        x_reshaped[:, i, j+1, :],  # Right
                    ], dim=0)
                    avg_neighbor = neighbors.mean(dim=0)
                    x_smoothed[:, i, j, :] = (1 - self.smooth_factor) * x_reshaped[:, i, j, :] + self.smooth_factor * avg_neighbor
        
        # Reshape back
        return x_smoothed.view(B, N, D)


__all__ = ['PatchConsistency']
