#!/usr/bin/env python3
"""
Regularization Techniques for Delayed Generalization Research

Various regularization methods to control overfitting and influence training dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from collections import deque
import math


class WeightDecayRegularizer:
    """
    Flexible weight decay regularizer with different decay schedules.
    
    Can apply different weight decay rates to different parameter groups
    and adapt the decay rate during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_weight_decay: float = 1e-4,
        layer_specific_decay: Optional[Dict[str, float]] = None,
        exclude_bias: bool = True,
        exclude_bn: bool = True
    ):
        """
        Args:
            model: Model to regularize
            base_weight_decay: Base weight decay rate
            layer_specific_decay: Dict mapping layer names to specific decay rates
            exclude_bias: Whether to exclude bias terms from weight decay
            exclude_bn: Whether to exclude batch norm parameters
        """
        self.model = model
        self.base_weight_decay = base_weight_decay
        self.layer_specific_decay = layer_specific_decay or {}
        self.exclude_bias = exclude_bias
        self.exclude_bn = exclude_bn
    
    def compute_penalty(self) -> torch.Tensor:
        """Compute weight decay penalty."""
        penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip bias terms if requested
            if self.exclude_bias and 'bias' in name:
                continue
            
            # Skip batch norm parameters if requested
            if self.exclude_bn and ('bn' in name or 'batch_norm' in name):
                continue
            
            # Get decay rate for this parameter
            decay_rate = self.base_weight_decay
            for layer_name, specific_decay in self.layer_specific_decay.items():
                if layer_name in name:
                    decay_rate = specific_decay
                    break
            
            penalty += decay_rate * torch.norm(param, p=2) ** 2
        
        return penalty
    
    def apply_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply weight decay regularization to loss."""
        return loss + 0.5 * self.compute_penalty()


class DropoutRegularizer:
    """
    Dynamic dropout regularizer that can adapt dropout rates during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_dropout: float = 0.1,
        adaptive: bool = False,
        schedule: str = 'constant'
    ):
        """
        Args:
            model: Model with dropout layers
            base_dropout: Base dropout rate
            adaptive: Whether to adapt dropout based on training metrics
            schedule: Dropout schedule ('constant', 'linear_decay', 'cosine_decay')
        """
        self.model = model
        self.base_dropout = base_dropout
        self.adaptive = adaptive
        self.schedule = schedule
        self.step_count = 0
        
        # Find all dropout layers
        self.dropout_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self.dropout_layers.append(module)
    
    def update_dropout_rate(self, epoch: int, total_epochs: int, metric: Optional[float] = None):
        """Update dropout rate based on schedule and metrics."""
        if self.schedule == 'constant':
            dropout_rate = self.base_dropout
        elif self.schedule == 'linear_decay':
            progress = epoch / total_epochs
            dropout_rate = self.base_dropout * (1.0 - progress)
        elif self.schedule == 'cosine_decay':
            progress = epoch / total_epochs
            dropout_rate = self.base_dropout * (1 + math.cos(math.pi * progress)) / 2
        else:
            dropout_rate = self.base_dropout
        
        # Adaptive adjustment based on metric (e.g., generalization gap)
        if self.adaptive and metric is not None:
            # Increase dropout if overfitting (high metric)
            adaptation_factor = 1.0 + 0.5 * min(1.0, max(0.0, metric))
            dropout_rate *= adaptation_factor
        
        # Apply to all dropout layers
        for layer in self.dropout_layers:
            layer.p = min(0.9, max(0.0, dropout_rate))


class SpectralNormRegularizer:
    """
    Spectral normalization regularizer to control Lipschitz constant.
    
    Helps with training stability and can influence delayed generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        spectral_norm_factor: float = 1.0,
        power_iterations: int = 1
    ):
        """
        Args:
            model: Model to apply spectral normalization
            spectral_norm_factor: Maximum spectral norm
            power_iterations: Number of power iterations for spectral norm estimation
        """
        self.model = model
        self.spectral_norm_factor = spectral_norm_factor
        self.power_iterations = power_iterations
        
        # Apply spectral normalization to linear and conv layers
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to appropriate layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.utils.spectral_norm(module, n_power_iterations=self.power_iterations)


class GradientPenaltyRegularizer:
    """
    Gradient penalty regularizer for improved training stability.
    
    Particularly useful for adversarial training and certain delayed generalization scenarios.
    """
    
    def __init__(
        self,
        model: nn.Module,
        penalty_weight: float = 10.0,
        target_norm: float = 1.0
    ):
        """
        Args:
            model: Model to regularize
            penalty_weight: Weight for gradient penalty
            target_norm: Target gradient norm
        """
        self.model = model
        self.penalty_weight = penalty_weight
        self.target_norm = target_norm
    
    def compute_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        loss_fn: Callable
    ) -> torch.Tensor:
        """Compute gradient penalty between real and fake data."""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Sample random interpolation weights
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)
        
        # Create interpolated data
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Forward pass
        output = self.model(interpolated)
        loss = loss_fn(output)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=interpolated,
            grad_outputs=torch.ones_like(loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - self.target_norm) ** 2).mean()
        
        return self.penalty_weight * penalty


class EarlyStopping:
    """
    Early stopping regularizer with various stopping criteria.
    
    Can stop based on validation loss, generalization gap, or custom metrics.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = 'min',
        baseline: Optional[float] = None,
        monitor_generalization_gap: bool = False
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' or 'max' for the monitored metric
            baseline: Baseline value for early stopping
            monitor_generalization_gap: Whether to monitor generalization gap
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.baseline = baseline
        self.monitor_generalization_gap = monitor_generalization_gap
        
        self.best_score = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def step(
        self,
        current_score: float,
        model: nn.Module,
        epoch: int,
        train_acc: Optional[float] = None,
        test_acc: Optional[float] = None
    ) -> bool:
        """
        Update early stopping based on current score.
        
        Returns:
            bool: Whether to stop training
        """
        # Monitor generalization gap if requested
        if self.monitor_generalization_gap and train_acc is not None and test_acc is not None:
            gap_score = train_acc - test_acc
            # Add gap score to the main score (penalize large gaps)
            current_score = current_score + 0.1 * gap_score
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        elif self.monitor_op(current_score, self.best_score - self.min_delta):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                
                # Restore best weights if requested
                if self.restore_best_weights and self.best_weights is not None:
                    for name, param in model.named_parameters():
                        param.data.copy_(self.best_weights[name])
        
        return self.should_stop


if __name__ == "__main__":
    # Test the regularizers
    print("Testing regularization techniques...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 50)
            self.dropout = nn.Dropout(0.5)
            self.linear2 = nn.Linear(50, 1)
    
        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            return self.linear2(x)
    
    model = TestModel()
    
    # Test WeightDecayRegularizer
    wd_reg = WeightDecayRegularizer(model, base_weight_decay=1e-4)
    penalty = wd_reg.compute_penalty()
    print(f"Weight decay penalty: {penalty.item():.6f}")
    
    # Test DropoutRegularizer
    dropout_reg = DropoutRegularizer(model, base_dropout=0.1, adaptive=True)
    dropout_reg.update_dropout_rate(epoch=50, total_epochs=100, metric=0.2)
    print(f"Updated dropout rate: {model.dropout.p:.3f}")
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=5, mode='min')
    
    # Simulate training
    for epoch in range(10):
        # Simulate improving then plateauing loss
        loss = 1.0 / (1 + epoch * 0.5) if epoch < 5 else 0.2 + 0.01 * np.random.random()
        should_stop = early_stop.step(loss, model, epoch)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    print("All regularization techniques implemented successfully!")