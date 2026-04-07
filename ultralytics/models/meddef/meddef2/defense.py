# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Defense modules for MedDef2 models.

This module re-exports defense mechanisms from the centralized defense module
and provides MedDef2-specific extensions.

NOTE: Core defense classes are now maintained in ultralytics.nn.modules.defense
for better modularity and consistency across the codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import core defenses from centralized module
from ultralytics.nn.modules.defense import (
    # Core inference-time defenses
    FrequencyDefense,
    PatchConsistency,
    ChannelAttention,
    CBAMAttention,
    # Regularization (used in loss computation)
    LabelSmoothing,
    GradientPenalty,
    # Wrapper for unified defense
    RobustDefenseWrapper,
)
# NOTE: TRADESLoss / MARTLoss / RobustMinMaxLoss / AdversarialWeightPerturbation are
# deliberately NOT imported here.  MedDef training uses ONLY defensive distillation
# (DefensiveDistillationLoss).  Adversarial training is prohibited in this pipeline.


# =============================================================================
# MedDef2-Specific Modules
# =============================================================================

class MultiScaleFeatures(nn.Module):
    """Multi-scale feature extraction module for MedDef2.
    
    Extracts features at multiple scales and fuses them for robust representation.
    """

    def __init__(self, in_channels: int):
        """Initialize MultiScaleFeatures.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ) for size in [(2, 2), (4, 4), (8, 8)]
        ])

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale feature extraction."""
        features = [x]
        for scale in self.scales:
            scaled = scale(x)
            features.append(F.interpolate(
                scaled, size=x.shape[-2:], mode='bilinear', align_corners=True
            ))
        multi_scale = torch.cat(features, dim=1)
        return self.fusion(multi_scale)


class DefenseModule(nn.Module):
    """Complete defense module combining multiple defense mechanisms for MedDef2.
    
    This module provides:
    - Multi-scale feature extraction
    - Feature fusion
    - Channel attention
    - Residual connection for gradient flow
    """

    def __init__(self, in_channels: int):
        """Initialize DefenseModule.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        # Multi-scale features
        self.multi_scale = MultiScaleFeatures(in_channels)

        # Feature aggregation
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale defense."""
        # Bypass during ONNX export — AdaptiveAvgPool2d with non-factor output
        # sizes (4×4, 8×8 on 14×14 feature maps) is unsupported in ONNX.
        if torch.onnx.is_in_onnx_export():
            return x

        # Get multi-scale features
        ms_features = self.multi_scale(x)
        
        # Fuse features
        fused = self.feature_fusion(ms_features)

        # Apply channel attention
        attention = self.channel_attention(fused)
        out = fused * attention

        return out + x  # Residual connection


class CBAM(nn.Module):
    """CBAM (Convolutional Block Attention Module) for MedDef2 models.
    
    This is a 2D CBAM variant optimized for CNN-style feature maps.
    For transformer sequences, use CBAMAttention from the centralized module.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        """Initialize CBAM.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
        """
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with channel and spatial attention."""
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Re-exported from centralized module (inference-time defenses only)
    'FrequencyDefense',
    'PatchConsistency',
    'ChannelAttention',
    'CBAMAttention',
    # TRADESLoss / MARTLoss / RobustMinMaxLoss / AdversarialWeightPerturbation are
    # intentionally absent — adversarial training is NEVER used in MedDef.
    'LabelSmoothing',
    'GradientPenalty',
    'RobustDefenseWrapper',
    # MedDef2-specific
    'MultiScaleFeatures',
    'DefenseModule',
    'CBAM',
]
