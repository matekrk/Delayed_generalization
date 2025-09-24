#!/usr/bin/env python3
"""
Advanced Regularization Techniques for Delayed Generalization

This module implements regularization strategies specifically designed to
control delayed generalization phenomena like grokking and simplicity bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Any, Union, Tuple
from abc import ABC, abstractmethod


class DelayedGeneralizationRegularizer(ABC):
    """Base class for delayed generalization regularizers."""
    
    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply regularization."""
        pass


class AdaptiveWeightDecay(DelayedGeneralizationRegularizer):
    """
    Adaptive weight decay that adjusts based on training dynamics.
    Increases weight decay when overfitting is detected.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_wd: float = 1e-2,
        adaptation_rate: float = 0.1,
        overfitting_threshold: float = 1.5,
        max_wd: float = 1e-1
    ):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.adaptation_rate = adaptation_rate
        self.overfitting_threshold = overfitting_threshold
        self.max_wd = max_wd
        
        # Set initial weight decay
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = initial_wd
    
    def apply(self, train_loss: float, val_loss: float):
        """Update weight decay based on overfitting ratio."""
        overfitting_ratio = val_loss / (train_loss + 1e-8)
        
        if overfitting_ratio > self.overfitting_threshold:
            # Increase weight decay
            for param_group in self.optimizer.param_groups:
                current_wd = param_group.get('weight_decay', 0)
                new_wd = min(current_wd * (1 + self.adaptation_rate), self.max_wd)
                param_group['weight_decay'] = new_wd
                
        return overfitting_ratio


class GradualDropout(nn.Module, DelayedGeneralizationRegularizer):
    """
    Dropout that gradually increases during training.
    Helps with late-stage regularization without hurting early learning.
    """
    
    def __init__(
        self,
        max_dropout: float = 0.5,
        warmup_epochs: int = 50,
        dropout_type: str = 'linear'
    ):
        super().__init__()
        self.max_dropout = max_dropout
        self.warmup_epochs = warmup_epochs
        self.dropout_type = dropout_type
        self.current_epoch = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.current_epoch <= 0:
            return x
            
        # Compute current dropout rate
        dropout_rate = self._compute_dropout_rate()
        return F.dropout(x, p=dropout_rate, training=True)
    
    def _compute_dropout_rate(self) -> float:
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        
        if self.dropout_type == 'linear':
            return self.max_dropout * progress
        elif self.dropout_type == 'cosine':
            return self.max_dropout * 0.5 * (1 - np.cos(np.pi * progress))
        elif self.dropout_type == 'exponential':
            return self.max_dropout * (progress ** 2)
        else:
            return self.max_dropout * progress
    
    def apply(self, epoch: int):
        """Update current epoch for dropout computation."""
        self.current_epoch = epoch


class AttentionDropout(nn.Module, DelayedGeneralizationRegularizer):
    """
    Specialized dropout for attention mechanisms.
    Helps prevent overfitting in transformer architectures.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(query, key, value, attn_mask=attn_mask)
        return self.dropout(attn_out), attn_weights
    
    def apply(self, *args, **kwargs):
        """For interface compliance."""
        pass


class BiasedFeatureDropout(nn.Module, DelayedGeneralizationRegularizer):
    """
    Dropout that specifically targets potentially biased features.
    Higher dropout rates for channels/features likely to contain spurious correlations.
    """
    
    def __init__(
        self,
        feature_channels: int,
        bias_channels: List[int],
        bias_dropout: float = 0.8,
        regular_dropout: float = 0.2
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.bias_channels = set(bias_channels)
        self.bias_dropout = bias_dropout
        self.regular_dropout = regular_dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
            
        # Create channel-specific dropout mask
        mask = torch.ones_like(x)
        
        for channel in range(x.size(-1) if len(x.shape) == 2 else x.size(1)):
            dropout_rate = self.bias_dropout if channel in self.bias_channels else self.regular_dropout
            channel_mask = torch.bernoulli(torch.full_like(
                mask[..., channel] if len(x.shape) == 2 else mask[:, channel],
                1 - dropout_rate
            ))
            
            if len(x.shape) == 2:
                mask[..., channel] = channel_mask
            else:
                mask[:, channel] = channel_mask
                
        return x * mask / (1 - self.regular_dropout)  # Scale to maintain expected value
    
    def apply(self, *args, **kwargs):
        """For interface compliance."""
        pass


class AntiBiasAugmentation(DelayedGeneralizationRegularizer):
    """
    Data augmentation specifically designed to break spurious correlations.
    """
    
    def __init__(
        self,
        bias_type: str = 'background',
        augmentation_strength: float = 0.5,
        progressive: bool = True
    ):
        self.bias_type = bias_type
        self.augmentation_strength = augmentation_strength
        self.progressive = progressive
        self.current_epoch = 0
        self.max_epochs = 100
        
    def apply(self, image: torch.Tensor, label: torch.Tensor, epoch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply anti-bias augmentation to image-label pair."""
        if epoch is not None:
            self.current_epoch = epoch
            
        strength = self._get_current_strength()
        
        if self.bias_type == 'background':
            return self._background_replacement(image, label, strength)
        elif self.bias_type == 'color':
            return self._color_randomization(image, label, strength)
        elif self.bias_type == 'texture':
            return self._texture_randomization(image, label, strength)
        else:
            return image, label
    
    def _get_current_strength(self) -> float:
        if not self.progressive:
            return self.augmentation_strength
            
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        return self.augmentation_strength * progress
    
    def _background_replacement(self, image: torch.Tensor, label: torch.Tensor, strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace background with random patterns."""
        if torch.rand(1) < strength:
            # Simple background replacement - replace with noise
            background_mask = self._detect_background(image)
            noise = torch.randn_like(image) * 0.1
            image = image * (1 - background_mask) + noise * background_mask
        return image, label
    
    def _color_randomization(self, image: torch.Tensor, label: torch.Tensor, strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomize color channels."""
        if torch.rand(1) < strength:
            # Randomly permute color channels
            perm = torch.randperm(image.size(0) if len(image.shape) == 3 else image.size(1))
            if len(image.shape) == 3:
                image = image[perm]
            else:
                image = image[:, perm]
        return image, label
    
    def _texture_randomization(self, image: torch.Tensor, label: torch.Tensor, strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add random texture patterns."""
        if torch.rand(1) < strength:
            texture = torch.randn_like(image) * 0.05
            image = image + texture
        return image, label
    
    def _detect_background(self, image: torch.Tensor) -> torch.Tensor:
        """Simple background detection - assume edges are foreground."""
        # This is a simplified version - in practice, you'd use more sophisticated methods
        if len(image.shape) == 3:  # CHW
            gray = image.mean(dim=0, keepdim=True)
        else:  # BCHW
            gray = image.mean(dim=1, keepdim=True)
            
        # Simple edge detection
        edges = torch.abs(F.conv2d(gray, torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float().to(image.device), padding=1))
        background_mask = (edges < 0.1).float()
        
        return background_mask


class ProgressiveAugmentation(DelayedGeneralizationRegularizer):
    """
    Augmentation that gradually increases in strength during training.
    """
    
    def __init__(
        self,
        max_strength: float = 1.0,
        warmup_epochs: int = 100,
        augmentation_types: List[str] = ['rotation', 'translation', 'scaling']
    ):
        self.max_strength = max_strength
        self.warmup_epochs = warmup_epochs
        self.augmentation_types = augmentation_types
        self.current_epoch = 0
        
    def apply(self, image: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        """Apply progressive augmentation."""
        if epoch is not None:
            self.current_epoch = epoch
            
        strength = self._get_augmentation_strength()
        return self._apply_augmentations(image, strength)
    
    def _get_augmentation_strength(self) -> float:
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        return self.max_strength * progress
    
    def _apply_augmentations(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply augmentations with given strength."""
        augmented = image.clone()
        
        for aug_type in self.augmentation_types:
            if torch.rand(1) < strength:
                if aug_type == 'rotation':
                    angle = (torch.rand(1) - 0.5) * 2 * 15 * strength  # Max 15 degrees
                    augmented = self._rotate(augmented, angle.item())
                elif aug_type == 'translation':
                    tx = (torch.rand(1) - 0.5) * 0.2 * strength  # Max 20% translation
                    ty = (torch.rand(1) - 0.5) * 0.2 * strength
                    augmented = self._translate(augmented, tx.item(), ty.item())
                elif aug_type == 'scaling':
                    scale = 1 + (torch.rand(1) - 0.5) * 0.2 * strength  # Max 20% scaling
                    augmented = self._scale(augmented, scale.item())
        
        return augmented
    
    def _rotate(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate image by given angle."""
        # Simplified rotation - in practice use torchvision transforms
        return image  # Placeholder
    
    def _translate(self, image: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
        """Translate image."""
        # Simplified translation
        return image  # Placeholder
    
    def _scale(self, image: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale image."""
        # Simplified scaling
        return image  # Placeholder


class SpectralRegularizer(nn.Module, DelayedGeneralizationRegularizer):
    """
    Spectral normalization regularizer for controlling Lipschitz constant.
    Helps with training stability and generalization.
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        # Apply spectral normalization
        self.module = torch.nn.utils.spectral_norm(
            module, name=name, n_power_iterations=n_power_iterations
        )
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def apply(self, *args, **kwargs):
        """For interface compliance."""
        pass


class InformationBottleneck(nn.Module, DelayedGeneralizationRegularizer):
    """
    Information bottleneck regularizer based on mutual information.
    Encourages learning minimal sufficient representations.
    """
    
    def __init__(self, beta: float = 1e-3, estimation_method: str = 'variance'):
        super().__init__()
        self.beta = beta
        self.estimation_method = estimation_method
        
    def forward(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute information bottleneck loss."""
        mi_loss = self._estimate_mutual_information(representations, targets)
        return self.beta * mi_loss
    
    def apply(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply information bottleneck regularization."""
        return self.forward(representations, targets)
    
    def _estimate_mutual_information(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information between representations and targets."""
        if self.estimation_method == 'variance':
            # Simple variance-based approximation
            return torch.var(representations)
        elif self.estimation_method == 'kl':
            # More sophisticated KL-based estimation
            return self._kl_estimation(representations, targets)
        else:
            return torch.var(representations)
    
    def _kl_estimation(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """KL divergence-based MI estimation."""
        # Simplified implementation - in practice, use more sophisticated estimators
        batch_size = representations.size(0)
        feature_dim = representations.size(-1)
        
        # Estimate entropy
        cov = torch.cov(representations.T)
        log_det = torch.logdet(cov + 1e-8 * torch.eye(feature_dim).to(representations.device))
        entropy = 0.5 * (feature_dim * np.log(2 * np.pi * np.e) + log_det)
        
        return entropy


def create_weight_decay_groups(
    model: nn.Module,
    wd_embedding: float = 1e-1,
    wd_attention: float = 1e-2,
    wd_mlp: float = 1e-3,
    no_decay_names: List[str] = ['bias', 'LayerNorm.weight']
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay rates.
    
    Args:
        model: PyTorch model
        wd_embedding: Weight decay for embedding layers
        wd_attention: Weight decay for attention layers
        wd_mlp: Weight decay for MLP/linear layers
        no_decay_names: Parameter names that shouldn't have weight decay
        
    Returns:
        List of parameter groups for optimizer
    """
    embedding_params = []
    attention_params = []
    mlp_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should not have weight decay
        if any(nd_name in name for nd_name in no_decay_names):
            no_decay_params.append(param)
        elif 'embedding' in name.lower():
            embedding_params.append(param)
        elif any(attn_name in name.lower() for attn_name in ['attention', 'attn', 'self_attn']):
            attention_params.append(param)
        else:
            mlp_params.append(param)
    
    param_groups = []
    
    if embedding_params:
        param_groups.append({'params': embedding_params, 'weight_decay': wd_embedding, 'name': 'embedding'})
    if attention_params:
        param_groups.append({'params': attention_params, 'weight_decay': wd_attention, 'name': 'attention'})
    if mlp_params:
        param_groups.append({'params': mlp_params, 'weight_decay': wd_mlp, 'name': 'mlp'})
    if no_decay_params:
        param_groups.append({'params': no_decay_params, 'weight_decay': 0.0, 'name': 'no_decay'})
    
    return param_groups


def create_regularization_config(phenomenon_type: str) -> Dict[str, Any]:
    """
    Create regularization configuration optimized for specific phenomena.
    
    Args:
        phenomenon_type: 'grokking', 'simplicity_bias', 'phase_transitions', etc.
        
    Returns:
        Dictionary with regularization configuration
    """
    
    if phenomenon_type == 'grokking':
        return {
            'weight_decay': 1e-2,  # Critical for grokking
            'dropout': 0.0,        # Often not needed
            'attention_dropout': 0.0,
            'adaptive_weight_decay': {
                'enabled': True,
                'initial_wd': 1e-2,
                'adaptation_rate': 0.1,
                'max_wd': 1e-1
            },
            'layer_specific_wd': {
                'enabled': False,  # Simple models usually don't need this
            },
            'spectral_norm': False,
            'information_bottleneck': False
        }
    
    elif phenomenon_type == 'simplicity_bias':
        return {
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'biased_feature_dropout': {
                'enabled': True,
                'bias_channels': [0, 1, 2],  # RGB channels
                'bias_dropout': 0.8,
                'regular_dropout': 0.2
            },
            'anti_bias_augmentation': {
                'enabled': True,
                'bias_type': 'background',
                'strength': 0.5,
                'progressive': True
            },
            'progressive_augmentation': {
                'enabled': True,
                'max_strength': 0.8,
                'warmup_epochs': 50
            }
        }
    
    elif phenomenon_type == 'phase_transitions':
        return {
            'layer_specific_wd': {
                'enabled': True,
                'wd_embedding': 1e-1,
                'wd_attention': 1e-2,
                'wd_mlp': 1e-3
            },
            'gradual_dropout': {
                'enabled': True,
                'max_dropout': 0.1,
                'warmup_epochs': 1000
            },
            'spectral_norm': True,
            'information_bottleneck': {
                'enabled': True,
                'beta': 1e-4
            }
        }
    
    else:
        # Default configuration
        return {
            'weight_decay': 1e-3,
            'dropout': 0.1,
            'adaptive_weight_decay': {
                'enabled': False
            }
        }


def track_regularization_metrics(
    model: nn.Module,
    train_loss: float,
    val_loss: float,
    epoch: int
) -> Dict[str, Any]:
    """
    Track key regularization metrics during training.
    
    Args:
        model: PyTorch model
        train_loss: Training loss
        val_loss: Validation loss
        epoch: Current epoch
        
    Returns:
        Dictionary with regularization metrics
    """
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'generalization_gap': val_loss - train_loss,
        'weight_norms': {},
        'gradient_norms': {}
    }
    
    # Calculate weight norms for each layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            metrics['weight_norms'][name] = param.data.norm().item()
            
            if param.grad is not None:
                metrics['gradient_norms'][name] = param.grad.norm().item()
    
    # Aggregate statistics
    weight_norms = list(metrics['weight_norms'].values())
    if weight_norms:
        metrics['avg_weight_norm'] = np.mean(weight_norms)
        metrics['max_weight_norm'] = np.max(weight_norms)
        metrics['weight_norm_std'] = np.std(weight_norms)
    
    gradient_norms = list(metrics['gradient_norms'].values())
    if gradient_norms:
        metrics['avg_gradient_norm'] = np.mean(gradient_norms)
        metrics['max_gradient_norm'] = np.max(gradient_norms)
        metrics['gradient_norm_std'] = np.std(gradient_norms)
    
    return metrics


if __name__ == "__main__":
    # Test regularization components
    print("Testing regularization components...")
    
    # Test weight decay groups
    model = nn.Sequential(
        nn.Embedding(1000, 128),
        nn.Linear(128, 64),
        nn.Linear(64, 10)
    )
    
    param_groups = create_weight_decay_groups(model)
    print(f"Created {len(param_groups)} parameter groups")
    
    # Test regularization configs
    for phenomenon in ['grokking', 'simplicity_bias', 'phase_transitions']:
        config = create_regularization_config(phenomenon)
        print(f"{phenomenon} config: {list(config.keys())}")
    
    print("Regularization tests completed!")