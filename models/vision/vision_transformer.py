#!/usr/bin/env python3
"""
Vision Transformer (ViT) Implementation for Delayed Generalization Research

This module provides Vision Transformer models optimized for studying delayed
generalization phenomena. Includes support for different model sizes and
specialized configurations for various experiments.

Key features:
- Multiple ViT variants (Tiny, Small, Base, Large)
- Configurable patch sizes and embedding dimensions
- Integration with delayed generalization tracking
- Support for different input resolutions
- Attention analysis capabilities for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Patch embeddings [batch_size, n_patches, embed_dim]
        """
        x = self.projection(x)  # [batch_size, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention with attention weight tracking for analysis.
    """
    
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For attention analysis
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Store attention weights for analysis
        self.attention_weights = attn_weights.detach()
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        
        return out
    
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for analysis."""
        return self.attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for regularization.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for delayed generalization research.
    
    Supports multiple configurations and includes features for analyzing
    attention patterns and delayed generalization phenomena.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        representation_size: Optional[int] = None,
        distilled: bool = False
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.distilled = distilled
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if distilled else None
        
        num_tokens = 2 if distilled else 1
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
            self.head = nn.Linear(representation_size, num_classes)
            if distilled:
                self.head_dist = nn.Linear(representation_size, num_classes)
        else:
            self.pre_logits = nn.Identity()
            self.head = nn.Linear(embed_dim, num_classes)
            if distilled:
                self.head_dist = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extraction layers.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, num_tokens, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.dist_token is not None:
            dist_tokens = self.dist_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        x = self.forward_features(x)
        
        if self.distilled:
            x_cls, x_dist = x[:, 0], x[:, 1]
            x_cls = self.pre_logits(x_cls)
            x_dist = self.pre_logits(x_dist)
            x_cls = self.head(x_cls)
            x_dist = self.head_dist(x_dist)
            if self.training:
                return x_cls, x_dist
            else:
                # Average the predictions
                return (x_cls + x_dist) / 2
        else:
            x = x[:, 0]  # Class token
            x = self.pre_logits(x)
            x = self.head(x)
            return x
    
    def get_attention_maps(self, layer_idx: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Get attention maps from transformer blocks.
        
        Args:
            layer_idx: Specific layer to get attention from, or None for all layers
            
        Returns:
            Dictionary mapping layer indices to attention weights
        """
        attention_maps = {}
        
        if layer_idx is not None:
            if 0 <= layer_idx < len(self.blocks):
                attn_weights = self.blocks[layer_idx].attn.get_attention_maps()
                if attn_weights is not None:
                    attention_maps[layer_idx] = attn_weights
        else:
            for i, block in enumerate(self.blocks):
                attn_weights = block.attn.get_attention_maps()
                if attn_weights is not None:
                    attention_maps[i] = attn_weights
        
        return attention_maps
    
    def get_feature_representations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get feature representations at different stages for analysis.
        
        Args:
            x: Input image tensor
            
        Returns:
            Dictionary of feature representations
        """
        representations = {}
        
        # Patch embeddings
        patch_features = self.patch_embed(x)
        representations['patch_embeddings'] = patch_features
        
        # Add positional embeddings
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, patch_features], dim=1)
        features = features + self.pos_embed
        features = self.pos_dropout(features)
        
        # Layer-wise features
        for i, block in enumerate(self.blocks):
            features = block(features)
            representations[f'layer_{i}'] = features.clone()
        
        # Final normalized features
        final_features = self.norm(features)
        representations['final_features'] = final_features
        representations['cls_token_final'] = final_features[:, 0]
        
        return representations


def create_vit_model(
    model_size: str = 'base',
    img_size: int = 224,
    num_classes: int = 1000,
    patch_size: int = 16,
    **kwargs
) -> VisionTransformer:
    """
    Create a Vision Transformer model with predefined configurations.
    
    Args:
        model_size: Size variant ('tiny', 'small', 'base', 'large', 'huge')
        img_size: Input image size
        num_classes: Number of output classes
        patch_size: Patch size for patch embedding
        **kwargs: Additional arguments for VisionTransformer
        
    Returns:
        Configured VisionTransformer model
    """
    
    # Model configurations
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'n_heads': 3,
            'mlp_ratio': 4.0
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'n_heads': 6,
            'mlp_ratio': 4.0
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'n_heads': 12,
            'mlp_ratio': 4.0
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'n_heads': 16,
            'mlp_ratio': 4.0
        },
        'huge': {
            'embed_dim': 1280,
            'depth': 32,
            'n_heads': 16,
            'mlp_ratio': 4.0
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size '{model_size}' not supported. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        **config
    )


# Factory functions for specific use cases
def vit_tiny_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Tiny with 16x16 patches for 224x224 images."""
    return create_vit_model('tiny', img_size=224, num_classes=num_classes, patch_size=16, **kwargs)


def vit_small_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Small with 16x16 patches for 224x224 images."""
    return create_vit_model('small', img_size=224, num_classes=num_classes, patch_size=16, **kwargs)


def vit_base_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base with 16x16 patches for 224x224 images."""
    return create_vit_model('base', img_size=224, num_classes=num_classes, patch_size=16, **kwargs)


def vit_large_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large with 16x16 patches for 224x224 images."""
    return create_vit_model('large', img_size=224, num_classes=num_classes, patch_size=16, **kwargs)


# Specialized variants for smaller images (CIFAR, etc.)
def vit_small_patch4_32(num_classes: int = 10, **kwargs) -> VisionTransformer:
    """ViT-Small with 4x4 patches for 32x32 images (CIFAR)."""
    return create_vit_model('small', img_size=32, num_classes=num_classes, patch_size=4, **kwargs)


def vit_tiny_patch4_32(num_classes: int = 10, **kwargs) -> VisionTransformer:
    """ViT-Tiny with 4x4 patches for 32x32 images (CIFAR)."""
    return create_vit_model('tiny', img_size=32, num_classes=num_classes, patch_size=4, **kwargs)


if __name__ == "__main__":
    # Test the models
    print("Testing Vision Transformer models...")
    
    # Test standard ViT
    model = vit_base_patch16_224(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"ViT-Base output shape: {output.shape}")
    print(f"ViT-Base parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test CIFAR variant
    cifar_model = vit_small_patch4_32(num_classes=10)
    x_cifar = torch.randn(2, 3, 32, 32)
    output_cifar = cifar_model(x_cifar)
    print(f"ViT-Small-CIFAR output shape: {output_cifar.shape}")
    print(f"ViT-Small-CIFAR parameters: {sum(p.numel() for p in cifar_model.parameters()):,}")
    
    # Test attention analysis
    _ = model(x)  # Forward pass to generate attention
    attention_maps = model.get_attention_maps()
    print(f"Attention maps available for layers: {list(attention_maps.keys())}")
    if attention_maps:
        layer_0_attn = list(attention_maps.values())[0]
        print(f"Layer 0 attention shape: {layer_0_attn.shape}")
    
    print("Vision Transformer tests completed successfully!")