#!/usr/bin/env python3
"""
Warmup Schedulers for Delayed Generalization Research

Various warmup strategies to stabilize early training and improve delayed generalization.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import List, Optional, Union, Callable


class LinearWarmup(_LRScheduler):
    """
    Linear warmup scheduler that gradually increases learning rate from 0 to base_lr.
    
    This is particularly effective for delayed generalization phenomena where
    rapid initial learning can lead to memorization rather than generalization.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        start_factor: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of steps for warmup
            start_factor: Starting factor (0.0 = start from 0, 1.0 = start from base_lr)
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = self.start_factor + (1.0 - self.start_factor) * (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # After warmup, return base learning rates
            return self.base_lrs


class ExponentialWarmup(_LRScheduler):
    """
    Exponential warmup scheduler that exponentially increases learning rate.
    
    Useful for very sensitive models where gradual acceleration is needed.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        gamma: float = 2.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of steps for warmup
            gamma: Exponential growth factor
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Exponential warmup
            progress = (self.last_epoch + 1) / self.warmup_steps
            factor = progress ** (1.0 / self.gamma)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class PolynomialWarmup(_LRScheduler):
    """
    Polynomial warmup scheduler for flexible warmup curves.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        power: float = 2.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of steps for warmup
            power: Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Polynomial warmup
            progress = (self.last_epoch + 1) / self.warmup_steps
            factor = progress ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class WarmupWrapper:
    """
    Wrapper that combines warmup with another scheduler.
    
    This allows you to apply warmup followed by any other scheduling strategy.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: _LRScheduler,
        main_scheduler: Optional[_LRScheduler] = None,
        warmup_steps: int = 1000
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_scheduler: Warmup scheduler to use initially
            main_scheduler: Main scheduler to use after warmup (optional)
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self, epoch: Optional[int] = None):
        """Step the appropriate scheduler based on training progress."""
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step(epoch)
        elif self.main_scheduler is not None:
            self.main_scheduler.step(epoch)
        
        self.step_count += 1
    
    def get_last_lr(self):
        """Get the last learning rate."""
        if self.step_count < self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        elif self.main_scheduler is not None:
            return self.main_scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]


class AdaptiveWarmup(_LRScheduler):
    """
    Adaptive warmup that adjusts based on training metrics.
    
    This can extend or shorten warmup based on training stability metrics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_warmup_steps: int,
        metric_fn: Optional[Callable[[], float]] = None,
        stability_threshold: float = 0.1,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            base_warmup_steps: Base number of warmup steps
            metric_fn: Function that returns training stability metric
            stability_threshold: Threshold for considering training stable
            last_epoch: Last epoch number
        """
        self.base_warmup_steps = base_warmup_steps
        self.metric_fn = metric_fn
        self.stability_threshold = stability_threshold
        self.effective_warmup_steps = base_warmup_steps
        self._last_metric = None
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Update effective warmup steps based on training stability
        if self.metric_fn is not None:
            current_metric = self.metric_fn()
            if current_metric > self.stability_threshold:
                # Training is unstable, extend warmup
                self.effective_warmup_steps = max(
                    self.effective_warmup_steps,
                    self.last_epoch + int(self.base_warmup_steps * 0.5)
                )
            self._last_metric = current_metric
        
        if self.last_epoch < self.effective_warmup_steps:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.effective_warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def get_warmup_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create warmup schedulers.
    
    Args:
        scheduler_type: Type of warmup ('linear', 'exponential', 'polynomial', 'adaptive')
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Configured warmup scheduler
    """
    
    if scheduler_type == 'linear':
        return LinearWarmup(optimizer, warmup_steps, **kwargs)
    elif scheduler_type == 'exponential':
        return ExponentialWarmup(optimizer, warmup_steps, **kwargs)
    elif scheduler_type == 'polynomial':
        return PolynomialWarmup(optimizer, warmup_steps, **kwargs)
    elif scheduler_type == 'adaptive':
        return AdaptiveWarmup(optimizer, warmup_steps, **kwargs)
    else:
        raise ValueError(f"Unknown warmup scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test the warmup schedulers
    import matplotlib.pyplot as plt
    
    # Create a dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test different warmup schedulers
    schedulers = {
        'Linear': LinearWarmup(optimizer, warmup_steps=100),
        'Exponential': ExponentialWarmup(optimizer, warmup_steps=100, gamma=2.0),
        'Polynomial': PolynomialWarmup(optimizer, warmup_steps=100, power=2.0)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        for epoch in range(150):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        
        ax.plot(lrs, label=name, linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Warmup Scheduler Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    
    plt.tight_layout()
    plt.savefig('/tmp/warmup_schedulers.png', dpi=150, bbox_inches='tight')
    print("Warmup scheduler comparison saved to /tmp/warmup_schedulers.png")