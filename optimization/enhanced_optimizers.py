#!/usr/bin/env python3
"""
Enhanced Optimizers with Additional Features for Delayed Generalization Research

This module provides enhanced versions of standard optimizers with additional features:
- Gradient norm tracking and clipping
- Adaptive learning rate schedules
- Weight decay scheduling
- Detailed logging capabilities
- Exponential Moving Average (EMA) support
- Moving Average (MA) support
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
import math


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    This can be used to maintain an exponentially decaying average of model parameters
    which often leads to better generalization and more stable training.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        """
        Initialize EMA.
        
        Args:
            model: The model whose parameters to track
            decay: Decay factor for the exponential moving average
            device: Device to store the shadow parameters
        """
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        
        # Create shadow parameters
        self.shadow_params = {}
        self.backup_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name] = (
                        self.decay * self.shadow_params[name] + 
                        (1.0 - self.decay) * param.data.to(self.device)
                    )
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model (backup original parameters first)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.backup_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name])
    
    def restore(self, model: nn.Module):
        """Restore original parameters from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup_params:
                    param.data.copy_(self.backup_params[name])
    
    def get_decay_schedule(self, step: int, warmup_steps: int = 1000):
        """Get adaptive decay schedule that starts slow and increases."""
        if step < warmup_steps:
            return min(self.decay, step / warmup_steps)
        return self.decay


class MovingAverage:
    """
    Simple Moving Average (MA) for model parameters.
    
    Maintains a moving average over the last N parameter updates.
    """
    
    def __init__(self, model: nn.Module, window_size: int = 100):
        """
        Initialize Moving Average.
        
        Args:
            model: The model whose parameters to track
            window_size: Number of recent updates to average over
        """
        self.window_size = window_size
        self.param_history = {}
        self.step_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_history[name] = []
    
    def update(self, model: nn.Module):
        """Update moving average."""
        self.step_count += 1
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.param_history:
                    # Add current parameters
                    self.param_history[name].append(param.data.clone())
                    
                    # Keep only last window_size updates
                    if len(self.param_history[name]) > self.window_size:
                        self.param_history[name] = self.param_history[name][-self.window_size:]
    
    def get_averaged_params(self) -> Dict[str, torch.Tensor]:
        """Get current moving average of parameters."""
        averaged_params = {}
        
        for name, param_list in self.param_history.items():
            if param_list:
                averaged_params[name] = torch.stack(param_list).mean(dim=0)
        
        return averaged_params
    
    def apply_averaged_params(self, model: nn.Module):
        """Apply averaged parameters to model."""
        averaged_params = self.get_averaged_params()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in averaged_params:
                    param.data.copy_(averaged_params[name])


class EnhancedAdamW(torch.optim.AdamW):
    """
    Enhanced AdamW optimizer with additional features for delayed generalization research.
    
    Additional features:
    - Gradient norm tracking
    - Adaptive weight decay
    - Learning rate warm-up
    - Gradient statistics logging
    - EMA and MA support
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        # Enhanced features
        grad_clip_norm: Optional[float] = None,
        adaptive_weight_decay: bool = False,
        warmup_steps: int = 0,
        log_grad_stats: bool = True,
        # EMA/MA support
        use_ema: bool = False,
        ema_decay: float = 0.999,
        use_ma: bool = False,
        ma_window_size: int = 100
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # Enhanced features
        self.grad_clip_norm = grad_clip_norm
        self.adaptive_weight_decay = adaptive_weight_decay
        self.warmup_steps = warmup_steps
        self.log_grad_stats = log_grad_stats
        
        # EMA/MA support
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_ma = use_ma
        self.ma_window_size = ma_window_size
        
        # Initialize EMA/MA (will be set when model is provided)
        self.ema = None
        self.ma = None
        
        # Tracking variables
        self.step_count = 0
        self.grad_norms = []
        self.effective_lrs = []
        self.weight_decay_history = []
    
    def setup_averaging(self, model: nn.Module):
        """Setup EMA/MA for the given model."""
        if self.use_ema:
            self.ema = ExponentialMovingAverage(model, decay=self.ema_decay)
        
        if self.use_ma:
            self.ma = MovingAverage(model, window_size=self.ma_window_size)
    
    def step(self, closure: Optional[Callable] = None, model: Optional[nn.Module] = None):
        """Enhanced step function with gradient tracking and averaging support."""
        
        # Track gradient statistics before clipping
        if self.log_grad_stats:
            self._log_gradient_stats()
        
        # Gradient clipping
        if self.grad_clip_norm is not None:
            total_norm = self._clip_gradients()
            self.grad_norms.append(total_norm)
        
        # Adaptive weight decay
        if self.adaptive_weight_decay:
            self._update_weight_decay()
        
        # Learning rate warmup
        if self.step_count < self.warmup_steps:
            self._apply_warmup()
        
        # Standard optimization step
        loss = super().step(closure)
        
        # Update EMA/MA if model is provided
        if model is not None:
            if self.ema is not None:
                self.ema.update(model)
            
            if self.ma is not None:
                self.ma.update(model)
        
        self.step_count += 1
        return loss
    
    def apply_ema(self, model: nn.Module):
        """Apply EMA parameters to model."""
        if self.ema is not None:
            self.ema.apply_shadow(model)
    
    def restore_original(self, model: nn.Module):
        """Restore original parameters (if EMA was applied)."""
        if self.ema is not None:
            self.ema.restore(model)
    
    def apply_ma(self, model: nn.Module):
        """Apply moving average parameters to model."""
        if self.ma is not None:
            self.ma.apply_averaged_params(model)
    
    def get_averaging_metrics(self) -> Dict[str, Any]:
        """Get metrics related to parameter averaging."""
        metrics = {}
        
        if self.ema is not None:
            metrics['ema_decay'] = self.ema.decay
            metrics['ema_enabled'] = True
        else:
            metrics['ema_enabled'] = False
        
        if self.ma is not None:
            metrics['ma_window_size'] = self.ma.window_size
            metrics['ma_step_count'] = self.ma.step_count
            metrics['ma_enabled'] = True
        else:
            metrics['ma_enabled'] = False
        
        return metrics
        
        # Warm-up learning rate
        if self.step_count < self.warmup_steps:
            self._apply_warmup()
        
        # Track effective learning rate
        self._track_effective_lr()
        
        # Standard AdamW step
        loss = super().step(closure)
        
        self.step_count += 1
        return loss
    
    def _clip_gradients(self) -> float:
        """Clip gradients and return total norm."""
        parameters = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parameters.append(p)
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.grad_clip_norm, norm_type=2.0
        )
        return total_norm.item()
    
    def _log_gradient_stats(self):
        """Log gradient statistics for analysis."""
        total_norm = 0
        param_count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.grad_norms.append(total_norm)
    
    def _update_weight_decay(self):
        """Adaptively update weight decay based on training progress."""
        # Simple adaptive scheme: reduce weight decay over time
        decay_factor = max(0.1, 1.0 - (self.step_count / 10000))
        
        for group in self.param_groups:
            original_wd = group.get('original_weight_decay', group['weight_decay'])
            group['weight_decay'] = original_wd * decay_factor
            
        self.weight_decay_history.append(group['weight_decay'])
    
    def _apply_warmup(self):
        """Apply learning rate warm-up."""
        warmup_factor = min(1.0, self.step_count / self.warmup_steps)
        
        for group in self.param_groups:
            base_lr = group.get('base_lr', group['lr'])
            group['lr'] = base_lr * warmup_factor
    
    def _track_effective_lr(self):
        """Track effective learning rate accounting for Adam's bias correction."""
        # Simple approximation of effective LR after bias correction
        beta1, beta2 = self.param_groups[0]['betas']
        bias_correction1 = 1 - beta1 ** (self.step_count + 1)
        bias_correction2 = 1 - beta2 ** (self.step_count + 1)
        
        effective_lr = self.param_groups[0]['lr'] * math.sqrt(bias_correction2) / bias_correction1
        self.effective_lrs.append(effective_lr)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for logging."""
        return {
            'step_count': self.step_count,
            'grad_norms': self.grad_norms,
            'effective_lrs': self.effective_lrs,
            'weight_decay_history': self.weight_decay_history,
            'current_lr': self.param_groups[0]['lr'],
            'current_weight_decay': self.param_groups[0]['weight_decay']
        }


class EnhancedSGD(torch.optim.SGD):
    """
    Enhanced SGD optimizer with additional features for delayed generalization research.
    
    Additional features:
    - Adaptive momentum
    - Learning rate scheduling integration
    - Gradient norm tracking
    - Nesterov momentum with adaptive coefficients
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        # Enhanced features
        adaptive_momentum: bool = False,
        grad_clip_norm: Optional[float] = None,
        momentum_decay: float = 0.99,
        log_grad_stats: bool = True
    ):
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        # Enhanced features
        self.adaptive_momentum = adaptive_momentum
        self.grad_clip_norm = grad_clip_norm
        self.momentum_decay = momentum_decay
        self.log_grad_stats = log_grad_stats
        
        # Tracking variables
        self.step_count = 0
        self.grad_norms = []
        self.momentum_history = []
        self.lr_history = []
        
        # Store original momentum for adaptive updates
        for group in self.param_groups:
            group['original_momentum'] = group['momentum']
    
    def step(self, closure: Optional[Callable] = None):
        """Enhanced step function with adaptive features."""
        
        # Track gradient statistics
        if self.log_grad_stats:
            self._log_gradient_stats()
        
        # Gradient clipping
        if self.grad_clip_norm is not None:
            total_norm = self._clip_gradients()
            self.grad_norms.append(total_norm)
        
        # Adaptive momentum
        if self.adaptive_momentum:
            self._update_momentum()
        
        # Track learning rate
        self.lr_history.append(self.param_groups[0]['lr'])
        
        # Standard SGD step
        loss = super().step(closure)
        
        self.step_count += 1
        return loss
    
    def _clip_gradients(self) -> float:
        """Clip gradients and return total norm."""
        parameters = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parameters.append(p)
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.grad_clip_norm, norm_type=2.0
        )
        return total_norm.item()
    
    def _log_gradient_stats(self):
        """Log gradient statistics for analysis."""
        total_norm = 0
        param_count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.grad_norms.append(total_norm)
    
    def _update_momentum(self):
        """Adaptively update momentum based on gradient consistency."""
        if len(self.grad_norms) < 2:
            return
        
        # Simple adaptive scheme: increase momentum when gradients are consistent
        recent_norms = self.grad_norms[-10:] if len(self.grad_norms) >= 10 else self.grad_norms
        norm_std = np.std(recent_norms)
        norm_mean = np.mean(recent_norms)
        
        # If gradients are consistent (low relative std), increase momentum
        if norm_mean > 0:
            consistency = 1.0 - min(1.0, norm_std / norm_mean)
            
            for group in self.param_groups:
                original_momentum = group['original_momentum']
                # Interpolate between original momentum and higher momentum based on consistency
                target_momentum = original_momentum + (0.95 - original_momentum) * consistency
                # Smooth update
                group['momentum'] = (self.momentum_decay * group['momentum'] + 
                                   (1 - self.momentum_decay) * target_momentum)
                
            self.momentum_history.append(group['momentum'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for logging."""
        return {
            'step_count': self.step_count,
            'grad_norms': self.grad_norms,
            'momentum_history': self.momentum_history,
            'lr_history': self.lr_history,
            'current_lr': self.param_groups[0]['lr'],
            'current_momentum': self.param_groups[0]['momentum']
        }


def create_optimizer_with_scheduler(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    scheduler_type: str = 'cosine',
    **kwargs
) -> tuple:
    """
    Create an optimizer and scheduler combination for delayed generalization research.
    
    Args:
        model: PyTorch model
        optimizer_type: 'adamw', 'sgd', 'enhanced_adamw', 'enhanced_sgd'
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        scheduler_type: 'cosine', 'linear', 'exponential', 'warmup_cosine'
        **kwargs: Additional arguments for optimizer and scheduler
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    
    # Create optimizer
    if optimizer_type.lower() == 'enhanced_adamw':
        optimizer = EnhancedAdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **{k: v for k, v in kwargs.items() if k in [
                'betas', 'eps', 'amsgrad', 'grad_clip_norm', 
                'adaptive_weight_decay', 'warmup_steps', 'log_grad_stats'
            ]}
        )
    elif optimizer_type.lower() == 'enhanced_sgd':
        optimizer = EnhancedSGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **{k: v for k, v in kwargs.items() if k in [
                'momentum', 'dampening', 'nesterov', 'adaptive_momentum',
                'grad_clip_norm', 'momentum_decay', 'log_grad_stats'
            ]}
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create scheduler
    total_steps = kwargs.get('total_steps', 1000)
    warmup_steps = kwargs.get('warmup_steps', 0)
    
    if scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    elif scheduler_type.lower() == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    elif scheduler_type.lower() == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type.lower() == 'warmup_cosine':
        # Custom warmup + cosine scheduler (would need to implement)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def get_optimizer_stats(optimizer) -> Dict[str, Any]:
    """Get statistics from enhanced optimizers."""
    if hasattr(optimizer, 'get_stats'):
        return optimizer.get_stats()
    else:
        # Basic stats for standard optimizers
        return {
            'step_count': getattr(optimizer, 'step_count', 0),
            'current_lr': optimizer.param_groups[0]['lr'],
            'optimizer_type': type(optimizer).__name__
        }