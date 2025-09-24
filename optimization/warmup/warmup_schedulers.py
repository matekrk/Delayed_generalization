#!/usr/bin/env python3
"""
Learning Rate Warmup Schedulers for Delayed Generalization

This module implements various warmup strategies designed to help with
delayed generalization phenomena like grokking and phase transitions.
"""

import math
import torch
from typing import Union, Optional, Callable
from abc import ABC, abstractmethod


class WarmupScheduler(ABC):
    """Base class for all warmup schedulers."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        target_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        self.step_count = 0
        
        # Store initial learning rates for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    @abstractmethod
    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current step."""
        pass
    
    def step(self):
        """Update learning rate for current step."""
        if self.step_count < self.warmup_steps:
            multiplier = self.get_lr_multiplier(self.step_count)
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * multiplier
        
        self.step_count += 1
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class LinearWarmup(WarmupScheduler):
    """
    Linear warmup scheduler.
    
    Learning rate increases linearly from 0 to target_lr over warmup_steps.
    Best for standard grokking experiments and transformer architectures.
    """
    
    def get_lr_multiplier(self, step: int) -> float:
        if step >= self.warmup_steps:
            return 1.0
        return step / self.warmup_steps


class CosineWarmup(WarmupScheduler):
    """
    Cosine warmup scheduler.
    
    Learning rate follows cosine curve from 0 to target_lr over warmup_steps.
    Provides smoother transitions, good for sensitive models.
    """
    
    def get_lr_multiplier(self, step: int) -> float:
        if step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (1 - step / self.warmup_steps)))


class ExponentialWarmup(WarmupScheduler):
    """
    Exponential warmup scheduler.
    
    Learning rate increases exponentially from small value to target_lr.
    Useful for particularly difficult delayed generalization scenarios.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        target_lr: Optional[float] = None,
        base: float = 2.0
    ):
        super().__init__(optimizer, warmup_steps, target_lr)
        self.base = base
    
    def get_lr_multiplier(self, step: int) -> float:
        if step >= self.warmup_steps:
            return 1.0
        return self.base ** (step / self.warmup_steps - 1)


class PolynomialWarmup(WarmupScheduler):
    """
    Polynomial warmup scheduler.
    
    Learning rate follows polynomial curve with configurable power.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        target_lr: Optional[float] = None,
        power: float = 2.0
    ):
        super().__init__(optimizer, warmup_steps, target_lr)
        self.power = power
    
    def get_lr_multiplier(self, step: int) -> float:
        if step >= self.warmup_steps:
            return 1.0
        return (step / self.warmup_steps) ** self.power


class AdaptiveWarmup(WarmupScheduler):
    """
    Adaptive warmup that adjusts based on training dynamics.
    
    Monitors loss and adapts warmup schedule accordingly.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_warmup_steps: int,
        target_lr: Optional[float] = None,
        adaptation_factor: float = 1.5,
        loss_threshold: float = 0.1
    ):
        super().__init__(optimizer, initial_warmup_steps, target_lr)
        self.initial_warmup_steps = initial_warmup_steps
        self.adaptation_factor = adaptation_factor
        self.loss_threshold = loss_threshold
        self.loss_history = []
        
    def update_loss(self, loss: float):
        """Update loss history for adaptive behavior."""
        self.loss_history.append(loss)
        
        # Adapt warmup steps based on loss behavior
        if len(self.loss_history) > 50:
            recent_losses = self.loss_history[-10:]
            if max(recent_losses) - min(recent_losses) > self.loss_threshold:
                # High loss variance - extend warmup
                self.warmup_steps = int(self.warmup_steps * self.adaptation_factor)
    
    def get_lr_multiplier(self, step: int) -> float:
        if step >= self.warmup_steps:
            return 1.0
        return step / self.warmup_steps


def create_warmup_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    target_lr: Optional[float] = None,
    **kwargs
) -> WarmupScheduler:
    """
    Factory function to create warmup schedulers.
    
    Args:
        scheduler_type: Type of warmup ('linear', 'cosine', 'exponential', 'polynomial', 'adaptive')
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for warmup
        target_lr: Target learning rate (defaults to optimizer's current LR)
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Initialized warmup scheduler
    """
    
    scheduler_map = {
        'linear': LinearWarmup,
        'cosine': CosineWarmup,
        'exponential': ExponentialWarmup,
        'polynomial': PolynomialWarmup,
        'adaptive': AdaptiveWarmup
    }
    
    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Available: {list(scheduler_map.keys())}")
    
    scheduler_class = scheduler_map[scheduler_type]
    return scheduler_class(optimizer, warmup_steps, target_lr, **kwargs)


class WarmupWrapper:
    """
    Wrapper that combines warmup with a post-warmup scheduler.
    
    Example:
        warmup = LinearWarmup(optimizer, 1000)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=9000)
        combined = WarmupWrapper(warmup, main_scheduler)
    """
    
    def __init__(
        self,
        warmup_scheduler: WarmupScheduler,
        main_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        
    def step(self, metric: Optional[float] = None):
        """Step both warmup and main scheduler as appropriate."""
        if self.warmup_scheduler.step_count < self.warmup_scheduler.warmup_steps:
            # Still in warmup phase
            self.warmup_scheduler.step()
        elif self.main_scheduler:
            # Post-warmup phase
            if hasattr(self.main_scheduler, 'step'):
                if metric is not None and 'ReduceLROnPlateau' in str(type(self.main_scheduler)):
                    self.main_scheduler.step(metric)
                else:
                    self.main_scheduler.step()
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.warmup_scheduler.optimizer.param_groups[0]['lr']


# Phenomenon-specific warmup configurations
def create_grokking_warmup(
    optimizer: torch.optim.Optimizer,
    total_epochs: int = 10000,
    warmup_fraction: float = 0.1
) -> WarmupWrapper:
    """
    Create optimal warmup configuration for grokking experiments.
    
    Args:
        optimizer: PyTorch optimizer (should use AdamW with weight_decay)
        total_epochs: Total training epochs
        warmup_fraction: Fraction of training for warmup
        
    Returns:
        Configured warmup + main scheduler
    """
    warmup_steps = int(total_epochs * warmup_fraction)
    main_steps = total_epochs - warmup_steps
    
    # Linear warmup for grokking
    warmup = LinearWarmup(optimizer, warmup_steps)
    
    # Cosine annealing for main training
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_steps, eta_min=1e-6
    )
    
    return WarmupWrapper(warmup, main_scheduler)


def create_bias_mitigation_warmup(
    optimizer: torch.optim.Optimizer,
    total_epochs: int = 300,
    warmup_epochs: int = 10
) -> WarmupWrapper:
    """
    Create warmup configuration for simplicity bias mitigation.
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        
    Returns:
        Configured warmup + scheduler
    """
    # Short linear warmup
    warmup = LinearWarmup(optimizer, warmup_epochs)
    
    # Step decay for robustness training
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[total_epochs // 3, 2 * total_epochs // 3],
        gamma=0.1
    )
    
    return WarmupWrapper(warmup, main_scheduler)


if __name__ == "__main__":
    # Test warmup schedulers
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test different warmup types
    warmup_types = ['linear', 'cosine', 'exponential']
    warmup_steps = 1000
    
    plt.figure(figsize=(15, 5))
    
    for i, warmup_type in enumerate(warmup_types):
        plt.subplot(1, 3, i+1)
        
        # Reset optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
        
        scheduler = create_warmup_scheduler(warmup_type, optimizer, warmup_steps)
        
        lrs = []
        steps = []
        
        for step in range(warmup_steps + 100):
            lrs.append(scheduler.get_current_lr())
            steps.append(step)
            scheduler.step()
        
        plt.plot(steps, lrs)
        plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title(f'{warmup_type.capitalize()} Warmup')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/warmup_comparison.png')
    plt.show()
    
    print("Warmup schedulers test completed!")