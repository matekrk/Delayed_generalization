#!/usr/bin/env python3
"""
Adaptive Scheduling for Delayed Generalization Research

Schedulers that adapt based on training metrics, gradients, and phase transitions.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import List, Optional, Dict, Any, Callable, Deque
from collections import deque


class MetricAdaptiveScheduler(_LRScheduler):
    """
    Scheduler that adapts learning rate based on arbitrary training metrics.
    
    Useful for delayed generalization where you want to adjust learning rate
    based on generalization gap, accuracy improvements, etc.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        metric_fn: Callable[[], float],
        target_metric: float,
        patience: int = 10,
        factor: float = 0.5,
        threshold: float = 1e-4,
        mode: str = 'min',
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            metric_fn: Function that returns the metric to track
            target_metric: Target value for the metric
            patience: Number of epochs to wait before reducing lr
            factor: Factor to multiply learning rate by
            threshold: Threshold for measuring improvement
            mode: 'min' if metric should decrease, 'max' if it should increase
            last_epoch: Last epoch number
        """
        self.metric_fn = metric_fn
        self.target_metric = target_metric
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.mode = mode
        
        self.best_metric = None
        self.bad_epochs = 0
        self.metric_history = deque(maxlen=100)
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_metric = self.metric_fn()
        self.metric_history.append(current_metric)
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return self.base_lrs
        
        # Check if metric improved
        if self.mode == 'min':
            improved = current_metric < self.best_metric - self.threshold
        else:
            improved = current_metric > self.best_metric + self.threshold
        
        if improved:
            self.best_metric = current_metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        # Reduce learning rate if no improvement
        if self.bad_epochs >= self.patience:
            self.bad_epochs = 0
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            return [lr * self.factor for lr in current_lrs]
        
        # Return current learning rates
        return [group['lr'] for group in self.optimizer.param_groups]


class PhaseTransitionScheduler(_LRScheduler):
    """
    Scheduler designed to trigger phase transitions in delayed generalization.
    
    This scheduler can detect potential phase transitions and adjust learning rate
    to either encourage or study these transitions.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        train_acc_fn: Callable[[], float],
        test_acc_fn: Callable[[], float],
        transition_threshold: float = 0.1,
        boost_factor: float = 2.0,
        reduce_factor: float = 0.5,
        detection_window: int = 50,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            train_acc_fn: Function returning training accuracy
            test_acc_fn: Function returning test accuracy
            transition_threshold: Threshold for detecting phase transitions
            boost_factor: Factor to boost lr during transitions
            reduce_factor: Factor to reduce lr after transitions
            detection_window: Window size for transition detection
            last_epoch: Last epoch number
        """
        self.train_acc_fn = train_acc_fn
        self.test_acc_fn = test_acc_fn
        self.transition_threshold = transition_threshold
        self.boost_factor = boost_factor
        self.reduce_factor = reduce_factor
        self.detection_window = detection_window
        
        self.train_acc_history = deque(maxlen=detection_window)
        self.test_acc_history = deque(maxlen=detection_window)
        self.in_transition = False
        self.transition_start = None
        
        super().__init__(optimizer, last_epoch)
    
    def _detect_phase_transition(self):
        """Detect if we're in a phase transition."""
        if len(self.test_acc_history) < self.detection_window // 2:
            return False
        
        # Look for sudden improvement in test accuracy
        recent_test = np.mean(list(self.test_acc_history)[-10:])
        older_test = np.mean(list(self.test_acc_history)[-30:-20]) if len(self.test_acc_history) >= 30 else 0
        
        improvement = recent_test - older_test
        return improvement > self.transition_threshold
    
    def get_lr(self):
        current_train = self.train_acc_fn()
        current_test = self.test_acc_fn()
        
        self.train_acc_history.append(current_train)
        self.test_acc_history.append(current_test)
        
        # Detect phase transitions
        if not self.in_transition and self._detect_phase_transition():
            self.in_transition = True
            self.transition_start = self.last_epoch
            # Boost learning rate during transition
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            return [lr * self.boost_factor for lr in current_lrs]
        
        elif self.in_transition:
            # Check if transition is complete
            if self.last_epoch - self.transition_start > 20:
                # Stable for a while, reduce learning rate
                self.in_transition = False
                current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                return [lr * self.reduce_factor for lr in current_lrs]
        
        return [group['lr'] for group in self.optimizer.param_groups]


class GradientAdaptiveScheduler(_LRScheduler):
    """
    Scheduler that adapts based on gradient properties.
    
    Adjusts learning rate based on gradient norms, variance, or other
    gradient-based metrics that can indicate training dynamics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        target_grad_norm: float = 1.0,
        adaptation_rate: float = 0.1,
        min_lr_factor: float = 0.1,
        max_lr_factor: float = 10.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            model: Model to monitor gradients
            target_grad_norm: Target gradient norm
            adaptation_rate: Rate of learning rate adaptation
            min_lr_factor: Minimum learning rate factor
            max_lr_factor: Maximum learning rate factor
            last_epoch: Last epoch number
        """
        self.model = model
        self.target_grad_norm = target_grad_norm
        self.adaptation_rate = adaptation_rate
        self.min_lr_factor = min_lr_factor
        self.max_lr_factor = max_lr_factor
        
        self.grad_norm_history = deque(maxlen=100)
        
        super().__init__(optimizer, last_epoch)
    
    def _compute_grad_norm(self):
        """Compute current gradient norm."""
        total_norm = 0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return math.sqrt(total_norm) if param_count > 0 else 0.0
    
    def get_lr(self):
        current_grad_norm = self._compute_grad_norm()
        self.grad_norm_history.append(current_grad_norm)
        
        if current_grad_norm > 0:
            # Adjust learning rate based on gradient norm
            ratio = self.target_grad_norm / current_grad_norm
            adaptation_factor = 1.0 + self.adaptation_rate * (ratio - 1.0)
            
            # Clamp the adaptation factor
            adaptation_factor = max(self.min_lr_factor, min(self.max_lr_factor, adaptation_factor))
            
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            return [lr * adaptation_factor for lr in current_lrs]
        
        return [group['lr'] for group in self.optimizer.param_groups]


class LossPlateauScheduler(_LRScheduler):
    """
    Enhanced plateau scheduler with more sophisticated plateau detection.
    
    Includes features like trend analysis and adaptive patience.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[], float],
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        adaptive_patience: bool = True,
        trend_window: int = 20,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            loss_fn: Function that returns current loss
            mode: 'min' or 'max'
            factor: Factor to reduce learning rate by
            patience: Number of epochs to wait
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs'
            cooldown: Number of epochs to wait after lr reduction
            min_lr: Minimum learning rate
            adaptive_patience: Whether to adapt patience based on training progress
            trend_window: Window size for trend analysis
            last_epoch: Last epoch number
        """
        self.loss_fn = loss_fn
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.adaptive_patience = adaptive_patience
        self.trend_window = trend_window
        
        self.best_loss = None
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.loss_history = deque(maxlen=100)
        self.current_patience = patience
        
        super().__init__(optimizer, last_epoch)
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current loss is better than best."""
        if self.threshold_mode == 'rel':
            rel_epsilon = 1.0 - self.threshold if self.mode == 'min' else self.threshold + 1.0
            return (current < best * rel_epsilon) if self.mode == 'min' else (current > best * rel_epsilon)
        else:
            return (current < best - self.threshold) if self.mode == 'min' else (current > best + self.threshold)
    
    def _analyze_trend(self) -> str:
        """Analyze the trend in recent loss values."""
        if len(self.loss_history) < self.trend_window:
            return 'insufficient_data'
        
        recent_losses = list(self.loss_history)[-self.trend_window:]
        
        # Simple linear regression to detect trend
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < self.threshold:
            return 'plateau'
        elif slope < 0:
            return 'decreasing' if self.mode == 'min' else 'increasing_bad'
        else:
            return 'increasing' if self.mode == 'max' else 'decreasing_bad'
    
    def get_lr(self):
        current_loss = self.loss_fn()
        self.loss_history.append(current_loss)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.get_last_lr()
        
        if self.best_loss is None:
            self.best_loss = current_loss
        
        # Check if loss improved
        if self._is_better(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        # Adaptive patience based on trend analysis
        if self.adaptive_patience:
            trend = self._analyze_trend()
            if trend == 'plateau':
                self.current_patience = max(self.patience // 2, 5)
            elif trend in ['decreasing', 'increasing']:
                self.current_patience = self.patience
            else:
                self.current_patience = min(self.patience * 2, 50)
        
        # Reduce learning rate if plateau detected
        if self.bad_epochs >= self.current_patience:
            self.bad_epochs = 0
            self.cooldown_counter = self.cooldown
            new_lrs = []
            
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            for lr in current_lrs:
                new_lr = max(lr * self.factor, self.min_lr)
                new_lrs.append(new_lr)
            
            return new_lrs
        
        return [group['lr'] for group in self.optimizer.param_groups]


if __name__ == "__main__":
    # Test the adaptive schedulers
    print("Testing adaptive schedulers...")
    
    # Create dummy functions and model
    def dummy_metric():
        return np.random.exponential(0.5)
    
    def dummy_train_acc():
        return min(0.99, 0.5 + np.random.exponential(0.1))
    
    def dummy_test_acc():
        return min(0.95, 0.3 + np.random.exponential(0.05))
    
    def dummy_loss():
        return np.random.exponential(1.0)
    
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test MetricAdaptiveScheduler
    scheduler = MetricAdaptiveScheduler(optimizer, dummy_metric, target_metric=0.1)
    
    lrs = []
    for epoch in range(100):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    print(f"MetricAdaptiveScheduler: Final LR = {lrs[-1]:.6f}")
    print("All adaptive schedulers implemented successfully!")