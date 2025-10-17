#!/usr/bin/env python3
"""
Enhanced SGD optimizer with additional features for delayed generalization research.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import math


class EnhancedSGD(torch.optim.SGD):
    """
    Enhanced SGD optimizer with additional features for delayed generalization research.
    
    When all enhanced features are disabled (default), this behaves exactly like 
    the vanilla PyTorch SGD optimizer.
    
    Additional features:
    - Adaptive momentum based on gradient consistency
    - Gradient norm tracking and clipping
    - Learning rate tracking
    - Gradient statistics logging
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        # Enhanced features (all disabled by default for vanilla behavior)
        adaptive_momentum: bool = False,
        grad_clip_norm: Optional[float] = None,
        momentum_decay: float = 0.99,
        log_grad_stats: bool = False
    ):
        # Initialize parent class with standard parameters
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        # Enhanced features configuration
        self.adaptive_momentum = adaptive_momentum
        self.grad_clip_norm = grad_clip_norm
        self.momentum_decay = momentum_decay
        self.log_grad_stats = log_grad_stats
        
        # Tracking variables (only used if features are enabled)
        self.step_count = 0
        self.grad_norms = []
        self.momentum_history = []
        self.lr_history = []
        
        # Store original parameters for adaptive features
        for group in self.param_groups:
            group['original_momentum'] = group['momentum']
    
    def step(self, closure: Optional[Callable] = None):
        """
        Enhanced step function with optional adaptive features.
        
        When no enhanced features are enabled, this behaves exactly like 
        the vanilla SGD step.
        """
        
        # Only apply enhanced features if they are enabled
        if self._has_enhanced_features():
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
        
        # Standard SGD step (unchanged from parent)
        loss = super().step(closure)
        
        if self._has_enhanced_features():
            self.step_count += 1
            
        return loss
    
    def _has_enhanced_features(self) -> bool:
        """Check if any enhanced features are enabled."""
        return (
            self.adaptive_momentum or
            self.grad_clip_norm is not None or
            self.log_grad_stats
        )
    
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
            total_norm = math.sqrt(total_norm)
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
            consistency = 1.0 - (norm_std / norm_mean)
            target_momentum = min(0.99, self.param_groups[0]['original_momentum'] * (1 + consistency))
            
            # Smoothly update momentum
            for group in self.param_groups:
                current_momentum = group['momentum']
                new_momentum = (self.momentum_decay * current_momentum + 
                              (1 - self.momentum_decay) * target_momentum)
                group['momentum'] = new_momentum
                
            self.momentum_history.append(self.param_groups[0]['momentum'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for logging."""
        return {
            'step_count': self.step_count,
            'current_lr': self.param_groups[0]['lr'],
            'current_momentum': self.param_groups[0]['momentum'],
            'grad_norms': self.grad_norms.copy(),
            'momentum_history': self.momentum_history.copy(),
            'lr_history': self.lr_history.copy(),
            'optimizer_type': 'EnhancedSGD',
            'has_enhanced_features': self._has_enhanced_features(),
            'enhanced_features': {
                'adaptive_momentum': self.adaptive_momentum,
                'grad_clip_norm': self.grad_clip_norm,
                'momentum_decay': self.momentum_decay,
                'log_grad_stats': self.log_grad_stats
            }
        }
    
    def reset_stats(self):
        """Reset all tracking statistics."""
        self.grad_norms.clear()
        self.momentum_history.clear()
        self.lr_history.clear()