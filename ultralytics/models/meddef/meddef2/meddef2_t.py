# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MedDef2_T: Vision Transformer-based model with comprehensive defense mechanisms.

This module provides the MedDef2_T architecture with:
- CBAM (Convolutional Block Attention Module) integration
- Frequency domain defense
- Multi-scale defense module
- Patch consistency checking
- Configurable depth for different model sizes (tiny, small, base, large)
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .defense import CBAM, DefenseModule, FrequencyDefense, PatchConsistency
from .transformer import DropPath, MLP, MultiHeadAttention, VisionTransformer


class CBAMTransformerBlock(nn.Module):
    """Transformer block with CBAM integration."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize CBAMTransformerBlock.
        
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Enable bias for QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.cbam = CBAM(dim, reduction_ratio=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        """Forward pass with CBAM integration.
        
        Args:
            x: Input tensor of shape [B, N, C]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        input_dtype = x.dtype
        
        # Self-attention
        norm1_out = self.norm1(x)
        attn_out = self.attn(norm1_out)
        attn_dtype = attn_out.dtype

        # CBAM processing
        B, N, C = attn_out.shape
        attn_out_float = attn_out.float()
        
        # Reshape to 4D for CBAM
        h = w = 0
        if N > 0:
            h = w = int(math.sqrt(N)) if int(math.sqrt(N))**2 == N else int(math.sqrt(N)) + 1
        
        attn_out_4d = torch.zeros(B, C, h, w, dtype=torch.float32, device=attn_out_float.device)
        
        if h * w > 0:
            if h * w != N:
                if h * w > N:
                    padding = h * w - N
                    attn_out_reshaped = F.pad(attn_out_float, (0, 0, 0, padding))
                else:
                    attn_out_reshaped = attn_out_float
            else:
                attn_out_reshaped = attn_out_float
                
            if attn_out_reshaped.shape[1] > 0:
                attn_out_4d = attn_out_reshaped.transpose(1, 2).reshape(B, C, h, w)
        
        # Apply CBAM
        attn_out_4d_cbam = self.cbam(attn_out_4d)
        
        # Reshape back to sequence
        attn_out_cbam_seq = attn_out_4d_cbam.flatten(2).transpose(1, 2)
        
        if h * w != N and h * w > N and N > 0:
            attn_out_cbam_seq = attn_out_cbam_seq[:, :N, :]
        elif N == 0:
            attn_out_cbam_seq = torch.zeros(B, 0, C, dtype=torch.float32, device=x.device)

        attn_out_processed = attn_out_cbam_seq.to(attn_dtype)
        dropped_attn_out = self.drop_path(attn_out_processed)
        x_after_attn = x.float() + dropped_attn_out.float()
        x_after_attn = x_after_attn.to(input_dtype)
        
        # MLP
        norm2_out = self.norm2(x_after_attn)
        mlp_out = self.mlp(norm2_out)
        dropped_mlp_out = self.drop_path(mlp_out)
        x_after_mlp = x_after_attn.float() + dropped_mlp_out.float()
        
        return x_after_mlp.to(input_dtype)


class MedDef2_T(VisionTransformer):
    """MedDef2_T: A robust Vision Transformer-based model for medical image analysis.
    
    This model extends the VisionTransformer with:
    - CBAM-integrated transformer blocks
    - Frequency domain defense
    - Multi-scale defense module
    - Patch consistency checking
    """

    def __init__(self, *args, **kwargs):
        """Initialize MedDef2_T.
        
        Accepts all VisionTransformer arguments plus:
            img_size: Input image size (default: 224)
            patch_size: Patch size (default: 16)
            in_channels: Input channels (default: 3)
            num_classes: Number of classes (default: 1000)
            embed_dim: Embedding dimension (default: 768)
            depth: Number of transformer blocks (default: 12)
            num_heads: Number of attention heads (default: 12)
            mlp_ratio: MLP ratio (default: 4.0)
            qkv_bias: Enable QKV bias (default: True)
            drop_rate: Dropout rate (default: 0.)
            attn_drop_rate: Attention dropout rate (default: 0.)
            drop_path_rate: Stochastic depth rate (default: 0.)
            robust_method: Optional robust method (default: None)
        """
        # Extract parameters
        img_size = kwargs.get('img_size', 224)
        patch_size = kwargs.get('patch_size', 16)
        in_channels = kwargs.get('in_channels', 3)
        num_classes = kwargs.get('num_classes', 1000)
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
        representation_size = kwargs.get('representation_size', None)
        pretrained = kwargs.get('pretrained', False)
        robust_method = kwargs.get('robust_method', None)

        super().__init__(*args, **kwargs)
        
        # Replace blocks with CBAM-integrated blocks
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
        
        # Add defense mechanisms
        self.defense = DefenseModule(in_channels=self.embed_dim)
        self.frequency_defense = FrequencyDefense(cutoff_ratio=0.5)
        self.patch_consistency = PatchConsistency(
            embed_dim=self.embed_dim,
            grid_size=self.patch_embed.grid_size,
            threshold=1.0
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with integrated defense mechanisms.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Feature tensor
        """
        # Ensure input is FP32
        x = x.float()

        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.float()
        
        grid_h, grid_w = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        # Apply frequency defense
        x = self.frequency_defense(x)
        x = x.float()
        
        # Apply multi-scale defense
        x = self.defense(x)
        x = x.float()
        
        # Back to sequence format
        x = x.flatten(2).transpose(1, 2)
        
        # Apply patch consistency
        x = self.patch_consistency(x)
        x = x.float()
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        if x.size(1) != self.pos_embed.size(1):
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        x = x.float()
        
        # Process through transformer blocks
        x = self.blocks(x)
        x = x.float()
        
        x = self.norm(x)
        x = x[:, 0]  # Extract class token
        x = self.pre_logits(x)
        x = x.float()
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional robust method.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor
        """
        # Ensure input is FP32
        x = x.float()
        x = self.forward_features(x)
        
        if self.robust_method:
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)
            x, _ = self.robust_method(x, x, x)
            x = x.view(B, -1)
            return x
        else:
            x = self.head(x)
            return x


def check_num_classes(func):
    """Decorator to ensure num_classes is specified."""
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper


@check_num_classes
def get_meddef2_t(depth: float, pretrained: bool = False, input_channels: int = 3,
                  num_classes: int = None, robust_method: Optional[object] = None) -> MedDef2_T:
    """Factory function to create a MedDef2_T model with specified depth.
    
    Args:
        depth: Model depth variant:
            - 2.0: Tiny (6 blocks, 192 dim, 3 heads)
            - 2.1: Small (12 blocks, 384 dim, 6 heads)
            - 2.2: Base (12 blocks, 768 dim, 12 heads)
            - 2.3: Large (24 blocks, 1024 dim, 16 heads)
        pretrained: Load pretrained weights
        input_channels: Number of input channels
        num_classes: Number of output classes (required)
        robust_method: Optional robust method for additional defense
        
    Returns:
        MedDef2_T model instance
        
    Examples:
        >>> model = get_meddef2_t(2.0, num_classes=10)  # Tiny variant
        >>> model = get_meddef2_t(2.2, num_classes=1000)  # Base variant
    """
    depth_to_config = {
        2.0: {'name': 'meddef2_t_tiny', 'patch_size': 16, 'embed_dim': 192, 'depth': 6, 'num_heads': 3, 'mlp_ratio': 4},
        2.1: {'name': 'meddef2_t_small', 'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4},
        2.2: {'name': 'meddef2_t_base', 'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
        2.3: {'name': 'meddef2_t_large', 'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    }
    
    if depth not in depth_to_config:
        raise ValueError(
            f"Unsupported MedDef2_T depth: {depth}. Use one of: {list(depth_to_config.keys())}"
        )
    
    config = depth_to_config[depth]
    logging.info(
        f"Creating {config['name']} model with {config['depth']} transformer blocks"
    )
    
    model = MedDef2_T(
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=True,
        in_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        robust_method=robust_method
    )
    
    return model


def meddef2_t_0(pretrained: bool = False, **kwargs):
    """Create MedDef2_T Tiny model (depth=2.0)."""
    return get_meddef2_t(2.0, pretrained=pretrained, **kwargs)


def meddef2_t_1(pretrained: bool = False, **kwargs):
    """Create MedDef2_T Small model (depth=2.1)."""
    return get_meddef2_t(2.1, pretrained=pretrained, **kwargs)


def meddef2_t_2(pretrained: bool = False, **kwargs):
    """Create MedDef2_T Base model (depth=2.2)."""
    return get_meddef2_t(2.2, pretrained=pretrained, **kwargs)


def meddef2_t_3(pretrained: bool = False, **kwargs):
    """Create MedDef2_T Large model (depth=2.3)."""
    return get_meddef2_t(2.3, pretrained=pretrained, **kwargs)
