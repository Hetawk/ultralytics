# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# MedDef Transformer - Robust Vision Transformer for Medical Imaging
# Includes all defense mechanisms integrated into the architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from ultralytics.nn.modules.defense import (
    FrequencyDefense, PatchConsistency, CBAMAttention, DefensiveDistillationLoss
)


class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) as described in `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/pdf/1603.09382.pdf>`_"""

    def __init__(self, drop_prob: float = 0., training: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        x = self.proj(x)
        # [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0., proj_drop: float = 0., qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MedDefTransformerBlock(nn.Module):
    """Transformer block with integrated defense mechanisms (Frequency, Patch Consistency, CBAM)"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 grid_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Integrated defense mechanisms
        self.patch_consistency = PatchConsistency(dim, grid_size=grid_size, threshold=1.0, smooth_factor=0.5)
        self.cbam = CBAMAttention(dim, reduction_ratio=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with defenses
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        attn_out = self.cbam(attn_out)  # Apply CBAM
        attn_out = self.patch_consistency(attn_out)  # Apply patch consistency
        x = x + self.drop_path(attn_out)
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        return x


class MedDefVisionTransformer(nn.Module):
    """
    Robust Vision Transformer for Medical Image Analysis with Defensive Distillation
    
    Integrates:
    - Frequency Domain Defense (low-pass filtering)
    - Patch Consistency (anomaly smoothing)
    - CBAM Attention (channel + spatial)
    - Defensive Distillation (training time defense)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, act_layer: nn.Module = nn.GELU):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Frequency defense for input
        self.frequency_defense = FrequencyDefense(cutoff_ratio=0.5)
        
        # Transformer blocks with defenses
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        grid_size = (int(img_size / patch_size), int(img_size / patch_size))
        
        self.blocks = nn.ModuleList([
            MedDefTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,
                grid_size=grid_size
            ) for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Distillation loss (used during training)
        self.distill_loss = DefensiveDistillationLoss(temperature=4.0, alpha=0.5)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor, teacher_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
            teacher_logits (torch.Tensor, optional): Teacher model output for distillation
            
        Returns:
            torch.Tensor: Classification logits [B, num_classes]
        """
        # Frequency defense on input
        if x.shape[2:] == x.shape[2:]:  # Has spatial dimensions
            x = self.frequency_defense(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take class token
        x = self.head(x)
        
        return x

    def forward_with_distillation(self, x: torch.Tensor, y: torch.Tensor, 
                                  teacher_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with distillation loss for training"""
        logits = self.forward(x, teacher_logits=None)
        
        if teacher_logits is not None:
            loss = self.distill_loss(logits, y, teacher_logits)
        else:
            loss = F.cross_entropy(logits, y)
        
        return loss
