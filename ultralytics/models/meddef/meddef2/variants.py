# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MedDef2 Ablation Variants for systematic evaluation of defense components.

This module provides model variants with different defense combinations:
- MedDef2_T: Full model (CBAM + Freq + Patch + Defense)
- MedDef2_T_NoDef: No DefenseModule (CBAM + Freq + Patch)
- MedDef2_T_NoFreq: No FrequencyDefense (CBAM + Patch + Defense)
- MedDef2_T_NoPatch: No PatchConsistency (CBAM + Freq + Defense)  
- MedDef2_T_NoCBAM: No CBAM attention (Freq + Patch + Defense)
- MedDef2_T_Baseline: No defenses (standard ViT)

All variants use the centralized defense modules from ultralytics.nn.modules.defense
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.defense import (
    FrequencyDefense,
    PatchConsistency,
)

from .defense import CBAM, DefenseModule
from .transformer import (
    DropPath,
    MLP,
    MultiHeadAttention,
    TransformerBlock,
    VisionTransformer,
)
from .meddef2_t import CBAMTransformerBlock


# =============================================================================
# Variant 1: No DefenseModule (keeps CBAM, Freq, Patch)
# =============================================================================

class MedDef2_T_NoDef(VisionTransformer):
    """MedDef2_T without DefenseModule - for ablation study.
    
    Components:
    - ✓ CBAM attention in transformer blocks
    - ✓ FrequencyDefense
    - ✓ PatchConsistency
    - ✗ DefenseModule (removed)
    """

    def __init__(self, *args, **kwargs):
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        act_layer = kwargs.get('act_layer', nn.GELU)

        super().__init__(*args, **kwargs)
        
        # CBAM-integrated transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            CBAMTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # Defense components (no DefenseModule)
        self.frequency_defense = FrequencyDefense(cutoff_ratio=0.5)
        self.patch_consistency = PatchConsistency(
            embed_dim=self.embed_dim,
            grid_size=self.patch_embed.grid_size,
            threshold=1.0
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        B = x.shape[0]
        x = self.patch_embed(x).float()
        
        grid_h, grid_w = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        # Apply frequency defense
        x = self.frequency_defense(x).float()
        
        # NO defense module
        
        # Back to sequence
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_consistency(x).float()
        
        # Class token + position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x).float()
        x = self.blocks(x).float()
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x).float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.forward_features(x)
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
        else:
            x = self.head(x)
        return x


# =============================================================================
# Variant 2: No FrequencyDefense (keeps CBAM, Patch, Defense)
# =============================================================================

class MedDef2_T_NoFreq(VisionTransformer):
    """MedDef2_T without FrequencyDefense - for ablation study.
    
    Components:
    - ✓ CBAM attention in transformer blocks
    - ✗ FrequencyDefense (removed)
    - ✓ PatchConsistency
    - ✓ DefenseModule
    """

    def __init__(self, *args, **kwargs):
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        act_layer = kwargs.get('act_layer', nn.GELU)

        super().__init__(*args, **kwargs)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            CBAMTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # Defense components (no FrequencyDefense)
        self.defense = DefenseModule(in_channels=self.embed_dim)
        self.patch_consistency = PatchConsistency(
            embed_dim=self.embed_dim,
            grid_size=self.patch_embed.grid_size,
            threshold=1.0
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        B = x.shape[0]
        x = self.patch_embed(x).float()
        
        grid_h, grid_w = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        # NO frequency defense
        
        # Apply defense module
        x = self.defense(x).float()
        
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_consistency(x).float()
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x).float()
        x = self.blocks(x).float()
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x).float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.forward_features(x)
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
        else:
            x = self.head(x)
        return x


# =============================================================================
# Variant 3: No PatchConsistency (keeps CBAM, Freq, Defense)
# =============================================================================

class MedDef2_T_NoPatch(VisionTransformer):
    """MedDef2_T without PatchConsistency - for ablation study.
    
    Components:
    - ✓ CBAM attention in transformer blocks
    - ✓ FrequencyDefense
    - ✗ PatchConsistency (removed)
    - ✓ DefenseModule
    """

    def __init__(self, *args, **kwargs):
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        act_layer = kwargs.get('act_layer', nn.GELU)

        super().__init__(*args, **kwargs)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            CBAMTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # Defense components (no PatchConsistency)
        self.defense = DefenseModule(in_channels=self.embed_dim)
        self.frequency_defense = FrequencyDefense(cutoff_ratio=0.5)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        B = x.shape[0]
        x = self.patch_embed(x).float()
        
        grid_h, grid_w = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        x = self.frequency_defense(x).float()
        x = self.defense(x).float()
        
        # NO patch consistency - just reshape
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x).float()
        x = self.blocks(x).float()
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x).float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.forward_features(x)
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
        else:
            x = self.head(x)
        return x


# =============================================================================
# Variant 4: No CBAM (keeps Freq, Patch, Defense with standard blocks)
# =============================================================================

class MedDef2_T_NoCBAM(VisionTransformer):
    """MedDef2_T without CBAM attention - for ablation study.
    
    Components:
    - ✗ CBAM attention (uses standard TransformerBlock)
    - ✓ FrequencyDefense
    - ✓ PatchConsistency
    - ✓ DefenseModule
    """

    def __init__(self, *args, **kwargs):
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        act_layer = kwargs.get('act_layer', nn.GELU)

        super().__init__(*args, **kwargs)
        
        # Standard TransformerBlock (no CBAM)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # All defense components
        self.defense = DefenseModule(in_channels=self.embed_dim)
        self.frequency_defense = FrequencyDefense(cutoff_ratio=0.5)
        self.patch_consistency = PatchConsistency(
            embed_dim=self.embed_dim,
            grid_size=self.patch_embed.grid_size,
            threshold=1.0
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        B = x.shape[0]
        x = self.patch_embed(x).float()
        
        grid_h, grid_w = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        x = self.frequency_defense(x).float()
        x = self.defense(x).float()
        
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_consistency(x).float()
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x).float()
        x = self.blocks(x).float()
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x).float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.forward_features(x)
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
        else:
            x = self.head(x)
        return x


# =============================================================================
# Variant 5: Baseline (no defenses - standard ViT)
# =============================================================================

class MedDef2_T_Baseline(VisionTransformer):
    """MedDef2_T Baseline - standard ViT without any defenses.
    
    Components:
    - ✗ CBAM attention (uses standard TransformerBlock)
    - ✗ FrequencyDefense
    - ✗ PatchConsistency
    - ✗ DefenseModule
    
    This is the baseline for ablation comparison.
    """

    def __init__(self, *args, **kwargs):
        # Just use parent VisionTransformer as-is
        super().__init__(*args, **kwargs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Standard ViT forward without defenses."""
        x = x.float()
        B = x.shape[0]
        x = self.patch_embed(x).float()
        
        # Standard ViT flow - no defenses
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x).float()
        x = self.blocks(x).float()
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x).float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.forward_features(x)
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
        else:
            x = self.head(x)
        return x


# =============================================================================
# Factory Functions for Ablation Variants
# =============================================================================

def get_variant(variant: str, depth_config: str = 'base', **kwargs):
    """Factory function to create MedDef2 ablation variants.
    
    Args:
        variant: One of 'full', 'no_def', 'no_freq', 'no_patch', 'no_cbam', 'baseline'
        depth_config: One of 'tiny', 'small', 'base', 'large'
        **kwargs: Additional model arguments (num_classes required)
        
    Returns:
        Model instance
        
    Examples:
        >>> model = get_variant('full', 'base', num_classes=10)
        >>> model = get_variant('no_cbam', 'tiny', num_classes=5)
    """
    from .meddef2_t import MedDef2_T
    
    # Depth configurations
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 6, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    }
    
    # Variant classes
    variants = {
        'full': MedDef2_T,
        'no_def': MedDef2_T_NoDef,
        'no_freq': MedDef2_T_NoFreq,
        'no_patch': MedDef2_T_NoPatch,
        'no_cbam': MedDef2_T_NoCBAM,
        'baseline': MedDef2_T_Baseline,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Use one of {list(variants.keys())}")
    
    if depth_config not in configs:
        raise ValueError(f"Unknown depth: {depth_config}. Use one of {list(configs.keys())}")
    
    # Merge configurations
    config = configs[depth_config].copy()
    config.update(kwargs)
    
    model_class = variants[variant]
    return model_class(**config)


# Convenience functions
def meddef2_full(**kwargs):
    """Full MedDef2_T with all defenses."""
    return get_variant('full', **kwargs)

def meddef2_no_def(**kwargs):
    """MedDef2_T without DefenseModule."""
    return get_variant('no_def', **kwargs)

def meddef2_no_freq(**kwargs):
    """MedDef2_T without FrequencyDefense."""
    return get_variant('no_freq', **kwargs)

def meddef2_no_patch(**kwargs):
    """MedDef2_T without PatchConsistency."""
    return get_variant('no_patch', **kwargs)

def meddef2_no_cbam(**kwargs):
    """MedDef2_T without CBAM attention."""
    return get_variant('no_cbam', **kwargs)

def meddef2_baseline(**kwargs):
    """Baseline ViT without any defenses."""
    return get_variant('baseline', **kwargs)
