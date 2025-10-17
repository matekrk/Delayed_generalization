#!/usr/bin/env python3
"""
Enhanced AdamW optimizer with additional features for delayed generalization research.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import math


class EnhancedAdamW(torch.optim.AdamW):
    """
    Enhanced AdamW optimizer with additional features for delayed generalization research.
    
    When all enhanced features are disabled (default), this behaves exactly like 
    the vanilla PyTorch AdamW optimizer.
    
    Additional features:
    - Gradient norm tracking and clipping
    - Adaptive weight decay scheduling
    - Learning rate warm-up
    - Gradient statistics logging
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        # Enhanced features (all disabled by default for vanilla behavior)
        grad_clip_norm: Optional[float] = None,
        adaptive_weight_decay: bool = False,
        warmup_steps: int = 0,
        log_grad_stats: bool = False
    ):
        # Initialize parent class with standard parameters
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # Enhanced features configuration
        self.grad_clip_norm = grad_clip_norm
        self.adaptive_weight_decay = adaptive_weight_decay
        self.warmup_steps = warmup_steps
        self.log_grad_stats = log_grad_stats
        
        # Tracking variables (only used if features are enabled)
        self.step_count = 0
        self.grad_norms = []
        self.effective_lrs = []
        self.weight_decay_history = []
        
        # Store original parameters for adaptive features
        for group in self.param_groups:
            group['original_lr'] = group['lr']
            group['original_weight_decay'] = group['weight_decay']
        
    def step(self, closure: Optional[Callable] = None):
        """
        Enhanced step function with optional gradient tracking.
        
        When no enhanced features are enabled, this behaves exactly like 
        the vanilla AdamW step.
        """
        
        # Only apply enhanced features if they are enabled
        if self._has_enhanced_features():
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
            
            # Warm-up learning rate
            if self.step_count < self.warmup_steps:
                self._apply_warmup()
            
            # Track effective learning rate
            self._track_effective_lr()
        
        # Standard AdamW step (unchanged from parent)
        loss = super().step(closure)
        
        if self._has_enhanced_features():
            self.step_count += 1
            
        return loss
    
    def _has_enhanced_features(self) -> bool:
        """Check if any enhanced features are enabled."""
        return (
            self.grad_clip_norm is not None or
            self.adaptive_weight_decay or
            self.warmup_steps > 0 or
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
    
    def _update_weight_decay(self):
        """Adaptively update weight decay based on training progress."""
        # Simple adaptive weight decay that decreases over time
        decay_factor = max(0.1, 1.0 - self.step_count / 10000)
        
        for group in self.param_groups:
            new_weight_decay = group['original_weight_decay'] * decay_factor
            group['weight_decay'] = new_weight_decay
            
        self.weight_decay_history.append(decay_factor)
    
    def _apply_warmup(self):
        """Apply learning rate warmup."""
        warmup_factor = min(1.0, self.step_count / self.warmup_steps)
        
        for group in self.param_groups:
            group['lr'] = group['original_lr'] * warmup_factor
    
    def _track_effective_lr(self):
        """Track the effective learning rate for logging."""
        effective_lr = self.param_groups[0]['lr']
        self.effective_lrs.append(effective_lr)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for logging."""
        return {
            'step_count': self.step_count,
            'current_lr': self.param_groups[0]['lr'],
            'grad_norms': self.grad_norms.copy(),
            'effective_lrs': self.effective_lrs.copy(),
            'weight_decay_history': self.weight_decay_history.copy(),
            'optimizer_type': 'EnhancedAdamW',
            'has_enhanced_features': self._has_enhanced_features(),
            'enhanced_features': {
                'grad_clip_norm': self.grad_clip_norm,
                'adaptive_weight_decay': self.adaptive_weight_decay,
                'warmup_steps': self.warmup_steps,
                'log_grad_stats': self.log_grad_stats
            }
        }
    
    def reset_stats(self):
        """Reset all tracking statistics."""
        self.grad_norms.clear()
        self.effective_lrs.clear()
        self.weight_decay_history.clear()