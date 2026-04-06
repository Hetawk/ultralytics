# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attention-based Defense Modules (CBAM variants)."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention mechanism (CBAM variant for transformers)."""

    def __init__(self, dim: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // max(1, reduction_ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // max(1, reduction_ratio), dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, N, C] or [B, C, N]
        """
        if x.ndim == 3 and x.shape[-1] > x.shape[1]:  # [B, N, C] format
            B, N, C = x.shape
            # Transpose to [B, C, N] for pooling
            x_t = x.transpose(1, 2)
            avg_out = self.fc(self.avg_pool(x_t).squeeze(-1))
            max_out = self.fc(self.max_pool(x_t).squeeze(-1))
            out = torch.sigmoid(avg_out + max_out)
            return x * out.unsqueeze(1)
        else:
            # Standard [B, C, ...] format
            B, C = x.shape[:2]
            avg_out = self.fc(self.avg_pool(x).view(B, C))
            max_out = self.fc(self.max_pool(x).view(B, C))
            return x * torch.sigmoid(avg_out + max_out).view(B, C, *([1] * (x.ndim - 2)))


class CBAMAttention(nn.Module):
    """CBAM (Convolutional Block Attention Module) for transformer blocks."""

    def __init__(self, dim: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(dim, reduction_ratio)
        # Spatial attention (simplified for sequences)
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM to input tensor."""
        # Channel attention
        x = self.channel_att(x)
        
        # Spatial attention (for [B, N, C] format)
        if x.ndim == 3:
            B, N, C = x.shape
            avg_out = torch.mean(x, dim=2, keepdim=True)  # [B, N, 1]
            max_out, _ = torch.max(x, dim=2, keepdim=True)  # [B, N, 1]
            cat = torch.cat([avg_out, max_out], dim=2)  # [B, N, 2]
            
            # Reshape for conv1d: [B, 2, N]
            cat = cat.transpose(1, 2)
            att = self.spatial_att(cat)  # [B, 1, N]
            att = att.transpose(1, 2)  # [B, N, 1]
            x = x * att
        
        return x


__all__ = ['ChannelAttention', 'CBAMAttention']
