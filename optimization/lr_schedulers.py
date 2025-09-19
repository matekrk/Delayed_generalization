#!/usr/bin/env python3
"""
Advanced Learning Rate Schedulers for Delayed Generalization Research

This module provides specialized learning rate schedulers that can help with:
- Phase transition detection and adaptation
- Warm-up schedules for stable training
- Adaptive scheduling based on metrics
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import List, Optional, Dict, Any


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    This is particularly useful for delayed generalization phenomena where
    initial stability is important.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class PhaseTransitionScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler that detects phase transitions and adjusts LR accordingly.
    
    This scheduler monitors training metrics and can reduce LR when phase transitions
    are detected, or increase it to help trigger transitions.
    """
    
    def __init__(
        self,
        optimizer,
        patience: int = 100,
        factor: float = 0.5,
        min_lr: float = 1e-8,
        threshold: float = 1e-4,
        cooldown: int = 50,
        last_epoch: int = -1
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.cooldown = cooldown
        
        # State tracking
        self.metric_history = []
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.in_cooldown = False
        
        super().__init__(optimizer, last_epoch)
    
    def step(self, metric: float = None):
        """Step with optional metric for adaptive behavior."""
        if metric is not None:
            self.metric_history.append(metric)
            self._check_phase_transition(metric)
        
        super().step()
    
    def _check_phase_transition(self, metric: float):
        """Check if a phase transition has occurred and adjust accordingly."""
        if self.in_cooldown:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.in_cooldown = False
            return
        
        if self.best_metric is None:
            self.best_metric = metric
            return
        
        # Check for improvement
        if metric > self.best_metric + self.threshold:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Detect potential phase transition (sudden improvement after plateau)
        if len(self.metric_history) >= 20:
            recent_improvement = metric - np.mean(self.metric_history[-20:-1])
            if recent_improvement > 0.1:  # Significant improvement
                # Phase transition detected - temporarily increase LR
                self._boost_lr()
                self.in_cooldown = True
                self.cooldown_counter = self.cooldown
        
        # Standard plateau detection
        elif self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.in_cooldown = True
            self.cooldown_counter = self.cooldown
    
    def _boost_lr(self):
        """Temporarily boost learning rate when phase transition detected."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = min(old_lr * 2.0, self.base_lrs[0])  # Don't exceed base LR
            param_group['lr'] = new_lr
    
    def _reduce_lr(self):
        """Reduce learning rate when plateau detected."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
    
    def get_lr(self):
        # Return current LR (may have been modified by adaptive logic)
        return [group['lr'] for group in self.optimizer.param_groups]


class AdaptiveLRScheduler(_LRScheduler):
    """
    Advanced adaptive learning rate scheduler with multiple strategies.
    
    This scheduler can switch between different strategies based on training dynamics:
    - Cosine annealing for stable phases
    - Exponential decay for overfitting prevention
    - Cyclical LR for escaping local minima
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        base_strategy: str = 'cosine',  # 'cosine', 'exponential', 'cyclical'
        adaptation_window: int = 100,
        min_lr: float = 1e-8,
        max_lr_factor: float = 10.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.base_strategy = base_strategy
        self.adaptation_window = adaptation_window
        self.min_lr = min_lr
        self.max_lr_factor = max_lr_factor
        
        # Strategy state
        self.current_strategy = base_strategy
        self.loss_history = []
        self.strategy_changes = []
        
        super().__init__(optimizer, last_epoch)
    
    def step(self, loss: float = None):
        """Step with optional loss for adaptive strategy selection."""
        if loss is not None:
            self.loss_history.append(loss)
            self._adapt_strategy()
        
        super().step()
    
    def _adapt_strategy(self):
        """Adapt the learning rate strategy based on loss dynamics."""
        if len(self.loss_history) < self.adaptation_window:
            return
        
        recent_losses = self.loss_history[-self.adaptation_window:]
        
        # Detect different training phases
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        loss_variance = np.var(recent_losses)
        
        # Strategy selection logic
        if loss_variance < 1e-6 and abs(loss_trend) < 1e-6:
            # Plateau detected - switch to cyclical to escape
            new_strategy = 'cyclical'
        elif loss_trend > 0:
            # Loss increasing - switch to exponential decay
            new_strategy = 'exponential'
        else:
            # Normal training - use cosine
            new_strategy = 'cosine'
        
        if new_strategy != self.current_strategy:
            self.current_strategy = new_strategy
            self.strategy_changes.append((self.last_epoch, new_strategy))
    
    def get_lr(self):
        """Get learning rate based on current strategy."""
        progress = self.last_epoch / self.total_steps
        
        if self.current_strategy == 'cosine':
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
        elif self.current_strategy == 'exponential':
            decay_factor = 0.95 ** (self.last_epoch / 100)
            return [base_lr * decay_factor for base_lr in self.base_lrs]
        elif self.current_strategy == 'cyclical':
            # Simple cyclical LR
            cycle_length = 100
            cycle_progress = (self.last_epoch % cycle_length) / cycle_length
            lr_factor = 0.5 * (1 + math.sin(2 * math.pi * cycle_progress))
            return [
                base_lr * (0.1 + 0.9 * lr_factor) for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs
    
    def get_strategy_history(self) -> List[tuple]:
        """Get history of strategy changes for analysis."""
        return self.strategy_changes


class CyclicalLRWithWarmup(_LRScheduler):
    """
    Cyclical learning rate with warm-up, useful for finding optimal LR ranges
    and potentially triggering phase transitions.
    """
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode = mode
        self.warmup_steps = warmup_steps
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [self.base_lr * warmup_factor for _ in self.base_lrs]
        
        # Cyclical LR after warmup
        cycle_epoch = self.last_epoch - self.warmup_steps
        cycle = math.floor(1 + cycle_epoch / (self.step_size_up + self.step_size_down))
        x = abs(cycle_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr_factor = max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr_factor = max(0, (1 - x) / (2 ** (cycle - 1)))
        elif self.mode == 'exp_range':
            gamma = 0.99994
            lr_factor = max(0, (1 - x) * (gamma ** cycle_epoch))
        else:
            lr_factor = max(0, (1 - x))
        
        return [self.base_lr + (self.max_lr - self.base_lr) * lr_factor for _ in self.base_lrs]


def create_scheduler_for_phenomenon(
    optimizer,
    phenomenon_type: str,
    total_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    Create a learning rate scheduler optimized for specific delayed generalization phenomena.
    
    Args:
        optimizer: PyTorch optimizer
        phenomenon_type: 'grokking', 'simplicity_bias', 'phase_transitions'
        total_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured learning rate scheduler
    """
    
    if phenomenon_type == 'grokking':
        # Grokking often benefits from warm-up followed by stable LR
        warmup_steps = kwargs.get('warmup_steps', total_steps // 10)
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif phenomenon_type == 'simplicity_bias':
        # Bias mitigation might benefit from adaptive scheduling
        return PhaseTransitionScheduler(
            optimizer,
            patience=kwargs.get('patience', 50),
            factor=kwargs.get('factor', 0.8),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    elif phenomenon_type == 'phase_transitions':
        # Phase transitions benefit from adaptive strategies
        return AdaptiveLRScheduler(
            optimizer,
            total_steps=total_steps,
            base_strategy=kwargs.get('base_strategy', 'cosine'),
            adaptation_window=kwargs.get('adaptation_window', 100)
        )
    
    else:
        # Default: cosine annealing with warmup
        warmup_steps = kwargs.get('warmup_steps', total_steps // 20)
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=kwargs.get('min_lr', 1e-6)
        )