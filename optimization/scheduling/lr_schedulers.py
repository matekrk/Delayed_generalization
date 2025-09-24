#!/usr/bin/env python3
"""
Advanced Learning Rate Schedulers for Delayed Generalization

This module implements specialized learning rate scheduling strategies
designed to optimize delayed generalization phenomena.
"""

import math
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod


class DelayedGeneralizationScheduler(ABC):
    """Base class for delayed generalization schedulers."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    @abstractmethod
    def step(self, *args, **kwargs):
        """Step the scheduler."""
        pass
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class CosineAnnealingScheduler(DelayedGeneralizationScheduler):
    """
    Cosine annealing scheduler with warmup support.
    Excellent for grokking and phase transition experiments.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        warmup_steps: int = 0
    ):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        
    def step(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.base_lrs[i] * self.step_count / self.warmup_steps
                param_group['lr'] = lr
        else:
            # Cosine annealing after warmup
            progress = (self.step_count - self.warmup_steps) / (self.T_max - self.warmup_steps)
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                     (1 + math.cos(math.pi * progress)) / 2
                param_group['lr'] = lr
        
        self.step_count += 1


class AdaptiveStepDecay(DelayedGeneralizationScheduler):
    """
    Step decay scheduler that adapts based on training dynamics.
    Good for detecting and responding to phase transitions.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_factor: float = 0.1,
        patience: int = 1000,
        min_improvement: float = 1e-4,
        cooldown: int = 500
    ):
        super().__init__(optimizer)
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_improvement = min_improvement
        self.cooldown = cooldown
        
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        
    def step(self, val_loss: float):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        if val_loss < self.best_loss - self.min_improvement:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # Decay learning rate
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * self.decay_factor
                print(f"Reducing LR: {old_lr:.6f} -> {param_group['lr']:.6f}")
            
            self.wait = 0
            self.cooldown_counter = self.cooldown


class CyclicLR(DelayedGeneralizationScheduler):
    """
    Cyclic learning rate scheduler for escaping local minima.
    Useful for difficult optimization landscapes in delayed generalization.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = 'triangular',
        gamma: float = 1.0
    ):
        super().__init__(optimizer)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up
        self.mode = mode
        self.gamma = gamma
        
    def step(self):
        cycle = math.floor(1 + self.step_count / (self.step_size_up + self.step_size_down))
        x = abs(self.step_count / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.step_count
        else:
            scale_factor = 1.0
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * \
             max(0, (1 - x)) * scale_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1


class CosineAnnealingWarmRestarts(DelayedGeneralizationScheduler):
    """
    Cosine annealing with warm restarts.
    Multiple opportunities for phase transitions.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0
    ):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        
    def step(self):
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            # Restart
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            self.restart_count += 1
        
        # Cosine annealing within current restart
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            param_group['lr'] = lr


class GrokkingScheduler(DelayedGeneralizationScheduler):
    """
    Specialized scheduler optimized for grokking experiments.
    Multi-phase schedule: warmup -> main training -> fine-tuning.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int = 10000,
        warmup_fraction: float = 0.1,
        main_fraction: float = 0.8,
        min_lr: float = 1e-6
    ):
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
        self.main_epochs = int(total_epochs * main_fraction)
        self.final_epochs = total_epochs - self.warmup_epochs - self.main_epochs
        self.min_lr = min_lr
        self.epoch = 0
        
    def step(self):
        progress = self.epoch / self.total_epochs
        
        if progress < 0.1:  # Warmup phase
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.base_lrs[i] * (progress / 0.1)
                param_group['lr'] = lr
                
        elif progress < 0.9:  # Main training phase
            adjusted_progress = (progress - 0.1) / 0.8
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * \
                     (1 + math.cos(math.pi * adjusted_progress)) / 2
                param_group['lr'] = lr
                
        else:  # Final phase - very low learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.min_lr
        
        self.epoch += 1


class BiasScheduler(DelayedGeneralizationScheduler):
    """
    Scheduler designed to combat simplicity bias.
    Aggressive start -> plateau -> fine-tuning.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int = 300,
        aggressive_fraction: float = 0.2,
        plateau_fraction: float = 0.3
    ):
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.aggressive_epochs = int(total_epochs * aggressive_fraction)
        self.plateau_epochs = int(total_epochs * plateau_fraction)
        self.finetune_epochs = total_epochs - self.aggressive_epochs - self.plateau_epochs
        self.epoch = 0
        
    def step(self):
        progress = self.epoch / self.total_epochs
        
        if progress < 0.2:  # Aggressive start
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * 10  # 10x learning rate
                
        elif progress < 0.5:  # Plateau
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]
                
        else:  # Fine-tuning with cosine decay
            adjusted_progress = (progress - 0.5) / 0.5
            for i, param_group in enumerate(self.optimizer.param_groups):
                end_lr = self.base_lrs[i] * 0.01  # 1% of original
                lr = end_lr + (self.base_lrs[i] - end_lr) * \
                     (1 + math.cos(math.pi * adjusted_progress)) / 2
                param_group['lr'] = lr
        
        self.epoch += 1


class PhaseTransitionScheduler(DelayedGeneralizationScheduler):
    """
    Scheduler that adapts to detected phase transitions.
    Boosts learning rate during transitions for better exploration.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        detection_window: int = 100,
        transition_lr_boost: float = 2.0,
        boost_duration: int = 50
    ):
        super().__init__(optimizer)
        self.detection_window = detection_window
        self.transition_lr_boost = transition_lr_boost
        self.boost_duration = boost_duration
        
        self.loss_history = []
        self.in_transition = False
        self.transition_timer = 0
        
    def detect_phase_transition(self, loss: float) -> bool:
        """Detect if we're in a phase transition based on loss dynamics."""
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.detection_window:
            return False
        
        # Keep only recent history
        self.loss_history = self.loss_history[-self.detection_window:]
        
        # Calculate loss variance (high variance = potential transition)
        recent_losses = self.loss_history[-20:]
        if len(recent_losses) >= 20:
            variance = np.var(recent_losses)
            mean_loss = np.mean(recent_losses)
            cv = variance / (mean_loss + 1e-8)  # Coefficient of variation
            
            return cv > 0.1  # Threshold for transition detection
        
        return False
    
    def step(self, current_loss: float):
        transition_detected = self.detect_phase_transition(current_loss)
        
        if transition_detected and not self.in_transition:
            # Start of transition - boost learning rate
            self.in_transition = True
            self.transition_timer = 0
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.base_lrs[i] * self.transition_lr_boost
                param_group['lr'] = new_lr
            
            print(f"Phase transition detected! Boosting LR to {new_lr:.6f}")
            
        elif self.in_transition:
            self.transition_timer += 1
            
            # Gradually reduce LR back to base
            if self.transition_timer > self.boost_duration:
                decay_factor = 0.95
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] *= decay_factor
                
                    if param_group['lr'] <= self.base_lrs[i]:
                        param_group['lr'] = self.base_lrs[i]
                        self.in_transition = False
                        print("Transition complete, returning to base LR")


class LRRangeFinder:
    """
    Learning rate range finder for optimal LR selection.
    Useful for setting up cyclic learning rates.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def find_lr_range(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100
    ):
        """Find optimal LR range by gradually increasing LR and monitoring loss."""
        model = self.model
        model.train()
        
        lrs = []
        losses = []
        
        # Save initial state
        initial_state = model.state_dict()
        
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        current_lr = start_lr
        
        for i, (data, target) in enumerate(train_loader):
            if i >= num_iter:
                break
                
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            # Training step
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Record
            lrs.append(current_lr)
            losses.append(loss.item())
            
            # Update learning rate
            current_lr *= lr_mult
            
            # Stop if loss explodes
            if i > 10 and loss.item() > 4 * min(losses):
                break
        
        # Restore initial state
        model.load_state_dict(initial_state)
        
        return lrs, losses
    
    def suggest_lr(self, lrs: List[float], losses: List[float]) -> float:
        """Suggest optimal learning rate based on gradient."""
        if len(losses) > 1:
            derivatives = np.gradient(losses)
            min_gradient_idx = np.argmin(derivatives)
            return lrs[min_gradient_idx]
        return lrs[0]


class AdaptiveMomentumScheduler:
    """
    Scheduler that coordinates learning rate and momentum.
    Higher LR -> Lower momentum, Lower LR -> Higher momentum.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        max_momentum: float = 0.95,
        min_momentum: float = 0.85
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        # Step the LR scheduler
        self.lr_scheduler.step()
        
        # Adjust momentum inversely to learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_ratio = current_lr / self.initial_lr
        
        # Higher LR -> Lower momentum, Lower LR -> Higher momentum
        momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * lr_ratio
        momentum = max(min(momentum, self.max_momentum), self.min_momentum)
        
        # Update momentum (for SGD optimizers)
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                param_group['momentum'] = momentum


def create_phenomenon_scheduler(
    phenomenon_type: str,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> DelayedGeneralizationScheduler:
    """
    Factory function to create schedulers optimized for specific phenomena.
    
    Args:
        phenomenon_type: 'grokking', 'simplicity_bias', 'phase_transitions', etc.
        optimizer: PyTorch optimizer
        **kwargs: Phenomenon-specific arguments
        
    Returns:
        Configured scheduler for the phenomenon
    """
    
    if phenomenon_type == 'grokking':
        return GrokkingScheduler(
            optimizer,
            total_epochs=kwargs.get('total_epochs', 10000),
            warmup_fraction=kwargs.get('warmup_fraction', 0.1),
            main_fraction=kwargs.get('main_fraction', 0.8),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif phenomenon_type == 'simplicity_bias':
        return BiasScheduler(
            optimizer,
            total_epochs=kwargs.get('total_epochs', 300),
            aggressive_fraction=kwargs.get('aggressive_fraction', 0.2),
            plateau_fraction=kwargs.get('plateau_fraction', 0.3)
        )
    
    elif phenomenon_type == 'phase_transitions':
        return PhaseTransitionScheduler(
            optimizer,
            detection_window=kwargs.get('detection_window', 100),
            transition_lr_boost=kwargs.get('transition_lr_boost', 2.0),
            boost_duration=kwargs.get('boost_duration', 50)
        )
    
    elif phenomenon_type == 'cosine':
        return CosineAnnealingScheduler(
            optimizer,
            T_max=kwargs.get('T_max', 1000),
            eta_min=kwargs.get('eta_min', 0),
            warmup_steps=kwargs.get('warmup_steps', 0)
        )
    
    elif phenomenon_type == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-4),
            max_lr=kwargs.get('max_lr', 1e-2),
            step_size_up=kwargs.get('step_size_up', 2000),
            mode=kwargs.get('mode', 'triangular')
        )
    
    else:
        raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test grokking scheduler
    scheduler = create_phenomenon_scheduler('grokking', optimizer, total_epochs=1000)
    
    lrs = []
    for epoch in range(1000):
        lrs.append(scheduler.get_current_lr())
        scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Grokking Scheduler')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/grokking_scheduler.png')
    plt.show()
    
    print("Scheduling test completed!")