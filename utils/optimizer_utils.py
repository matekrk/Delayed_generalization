#!/usr/bin/env python3
"""
Optimizer Utilities for Delayed Generalization Research

This module provides utilities for working with optimizers, including
effective learning rate computation for AdamW and other adaptive optimizers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import logging


def compute_effective_lr_adamw(optimizer: torch.optim.AdamW, step: Optional[int] = None) -> Dict[str, float]:
    """
    Compute the effective learning rate for AdamW optimizer.
    
    For AdamW, the effective learning rate depends on:
    1. Base learning rate
    2. Bias correction terms (beta1^step, beta2^step)
    3. Second moment estimates (running average of squared gradients)
    4. Weight decay (which affects the effective step size)
    
    Args:
        optimizer: The AdamW optimizer instance
        step: Current step (if None, uses the step from optimizer state)
        
    Returns:
        Dictionary containing effective learning rates for each parameter group
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This function is specifically for AdamW optimizer")
    
    results = {
        'global_effective_lr': 0.0,
        'parameter_group_lrs': [],
        'layer_wise_lrs': {},
        'statistics': {}
    }
    
    total_params = 0
    weighted_lr_sum = 0.0
    
    for group_idx, param_group in enumerate(optimizer.param_groups):
        group_lr = param_group['lr']
        beta1, beta2 = param_group['betas']
        eps = param_group['eps']
        weight_decay = param_group['weight_decay']
        
        group_effective_lrs = []
        group_param_count = 0
        
        for param in param_group['params']:
            if param.grad is None:
                continue
                
            state = optimizer.state[param]
            
            # Get or use provided step
            current_step = step if step is not None else state.get('step', 0)
            
            if current_step == 0:
                # No updates yet, effective LR is base LR
                effective_lr = group_lr
            else:
                # Bias correction terms
                bias_correction1 = 1 - beta1 ** current_step
                bias_correction2 = 1 - beta2 ** current_step
                
                # Get second moment estimate
                exp_avg_sq = state.get('exp_avg_sq', torch.zeros_like(param))
                
                # Bias-corrected second moment
                v_hat = exp_avg_sq / bias_correction2
                
                # Denominator term (with eps for numerical stability)
                denom = v_hat.sqrt().add_(eps)
                
                # Effective step size per parameter element
                step_size = group_lr / bias_correction1
                effective_step_per_element = step_size / denom
                
                # Average effective learning rate for this parameter
                effective_lr = effective_step_per_element.mean().item()
                
                # Adjust for weight decay (which acts like additional gradient)
                if weight_decay > 0:
                    # Weight decay effectively adds `weight_decay * param` to gradient
                    # This reduces the effective learning rate for the original gradient
                    effective_lr = effective_lr / (1 + weight_decay * group_lr)
            
            group_effective_lrs.append(effective_lr)
            param_count = param.numel()
            group_param_count += param_count
            weighted_lr_sum += effective_lr * param_count
            total_params += param_count
            
            # Store layer-wise information if parameter has a name
            if hasattr(param, '_name'):
                results['layer_wise_lrs'][param._name] = effective_lr
        
        # Group statistics
        if group_effective_lrs:
            group_avg_lr = np.mean(group_effective_lrs)
            results['parameter_group_lrs'].append({
                'group_index': group_idx,
                'base_lr': group_lr,
                'effective_lr': group_avg_lr,
                'lr_ratio': group_avg_lr / group_lr if group_lr > 0 else 0,
                'parameter_count': group_param_count,
                'lr_std': np.std(group_effective_lrs)
            })
    
    # Global effective learning rate (weighted by parameter count)
    if total_params > 0:
        results['global_effective_lr'] = weighted_lr_sum / total_params
    
    # Additional statistics
    if results['parameter_group_lrs']:
        base_lrs = [group['base_lr'] for group in results['parameter_group_lrs']]
        effective_lrs = [group['effective_lr'] for group in results['parameter_group_lrs']]
        
        results['statistics'] = {
            'mean_base_lr': np.mean(base_lrs),
            'mean_effective_lr': np.mean(effective_lrs),
            'lr_adaptation_ratio': np.mean(effective_lrs) / np.mean(base_lrs) if np.mean(base_lrs) > 0 else 0,
            'lr_variance_across_groups': np.var(effective_lrs),
            'total_parameters': total_params
        }
    
    return results


def compute_effective_lr_sgd(optimizer: torch.optim.SGD) -> Dict[str, float]:
    """
    Compute effective learning rate for SGD optimizer.
    
    For SGD with momentum, the effective learning rate can be different
    from the base learning rate due to momentum accumulation.
    
    Args:
        optimizer: SGD optimizer instance
        
    Returns:
        Dictionary containing effective learning rates
    """
    results = {
        'global_effective_lr': 0.0,
        'parameter_group_lrs': [],
        'statistics': {}
    }
    
    total_params = 0
    weighted_lr_sum = 0.0
    
    for group_idx, param_group in enumerate(optimizer.param_groups):
        group_lr = param_group['lr']
        momentum = param_group.get('momentum', 0)
        weight_decay = param_group.get('weight_decay', 0)
        
        group_param_count = 0
        
        for param in param_group['params']:
            if param.grad is None:
                continue
            
            # For SGD, effective LR is base LR adjusted for weight decay
            effective_lr = group_lr
            
            # Weight decay reduces effective learning rate for original gradients
            if weight_decay > 0:
                effective_lr = group_lr / (1 + weight_decay)
            
            # Momentum doesn't change the effective LR per se, but changes convergence dynamics
            # We could compute momentum-adjusted effective LR, but it's more complex
            
            param_count = param.numel()
            group_param_count += param_count
            weighted_lr_sum += effective_lr * param_count
            total_params += param_count
        
        results['parameter_group_lrs'].append({
            'group_index': group_idx,
            'base_lr': group_lr,
            'effective_lr': group_lr,  # For SGD, often same as base LR
            'momentum': momentum,
            'weight_decay': weight_decay,
            'parameter_count': group_param_count
        })
    
    if total_params > 0:
        results['global_effective_lr'] = weighted_lr_sum / total_params
    
    return results


def compute_effective_lr_adam(optimizer: torch.optim.Adam, step: Optional[int] = None) -> Dict[str, float]:
    """
    Compute effective learning rate for standard Adam optimizer.
    
    Similar to AdamW but without the weight decay adjustment.
    
    Args:
        optimizer: Adam optimizer instance
        step: Current step (if None, uses optimizer state)
        
    Returns:
        Dictionary containing effective learning rates
    """
    results = {
        'global_effective_lr': 0.0,
        'parameter_group_lrs': [],
        'statistics': {}
    }
    
    total_params = 0
    weighted_lr_sum = 0.0
    
    for group_idx, param_group in enumerate(optimizer.param_groups):
        group_lr = param_group['lr']
        beta1, beta2 = param_group['betas']
        eps = param_group['eps']
        
        group_effective_lrs = []
        group_param_count = 0
        
        for param in param_group['params']:
            if param.grad is None:
                continue
                
            state = optimizer.state[param]
            current_step = step if step is not None else state.get('step', 0)
            
            if current_step == 0:
                effective_lr = group_lr
            else:
                bias_correction1 = 1 - beta1 ** current_step
                bias_correction2 = 1 - beta2 ** current_step
                
                exp_avg_sq = state.get('exp_avg_sq', torch.zeros_like(param))
                v_hat = exp_avg_sq / bias_correction2
                denom = v_hat.sqrt().add_(eps)
                
                step_size = group_lr / bias_correction1
                effective_step_per_element = step_size / denom
                effective_lr = effective_step_per_element.mean().item()
            
            group_effective_lrs.append(effective_lr)
            param_count = param.numel()
            group_param_count += param_count
            weighted_lr_sum += effective_lr * param_count
            total_params += param_count
        
        if group_effective_lrs:
            results['parameter_group_lrs'].append({
                'group_index': group_idx,
                'base_lr': group_lr,
                'effective_lr': np.mean(group_effective_lrs),
                'parameter_count': group_param_count,
                'lr_std': np.std(group_effective_lrs)
            })
    
    if total_params > 0:
        results['global_effective_lr'] = weighted_lr_sum / total_params
    
    return results


def compute_effective_lr(optimizer: torch.optim.Optimizer, step: Optional[int] = None) -> Dict[str, float]:
    """
    Compute effective learning rate for any optimizer.
    
    Args:
        optimizer: Optimizer instance
        step: Current step (optional)
        
    Returns:
        Dictionary containing effective learning rate information
    """
    if isinstance(optimizer, torch.optim.AdamW):
        return compute_effective_lr_adamw(optimizer, step)
    elif isinstance(optimizer, torch.optim.Adam):
        return compute_effective_lr_adam(optimizer, step)
    elif isinstance(optimizer, torch.optim.SGD):
        return compute_effective_lr_sgd(optimizer)
    else:
        # For other optimizers, return base learning rates
        results = {
            'global_effective_lr': 0.0,
            'parameter_group_lrs': [],
            'optimizer_type': type(optimizer).__name__,
            'note': 'Effective LR computation not implemented for this optimizer type'
        }
        
        total_params = 0
        weighted_lr_sum = 0.0
        
        for group_idx, param_group in enumerate(optimizer.param_groups):
            group_lr = param_group['lr']
            group_param_count = sum(p.numel() for p in param_group['params'] if p.grad is not None)
            
            results['parameter_group_lrs'].append({
                'group_index': group_idx,
                'base_lr': group_lr,
                'effective_lr': group_lr,  # Assume base LR = effective LR
                'parameter_count': group_param_count
            })
            
            weighted_lr_sum += group_lr * group_param_count
            total_params += group_param_count
        
        if total_params > 0:
            results['global_effective_lr'] = weighted_lr_sum / total_params
        
        return results


def track_lr_evolution(
    optimizer: torch.optim.Optimizer,
    steps: List[int],
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Track how effective learning rate evolves over training steps.
    
    Args:
        optimizer: Optimizer instance
        steps: List of training steps to evaluate
        save_path: Path to save the evolution plot
        
    Returns:
        Dictionary with LR evolution data
    """
    evolution_data = {
        'steps': steps,
        'global_effective_lrs': [],
        'base_lrs': [],
        'lr_ratios': []
    }
    
    for step in steps:
        lr_info = compute_effective_lr(optimizer, step)
        
        evolution_data['global_effective_lrs'].append(lr_info['global_effective_lr'])
        
        # Get base LR (from first parameter group)
        if lr_info['parameter_group_lrs']:
            base_lr = lr_info['parameter_group_lrs'][0]['base_lr']
            evolution_data['base_lrs'].append(base_lr)
            
            # Compute ratio
            effective_lr = lr_info['global_effective_lr']
            ratio = effective_lr / base_lr if base_lr > 0 else 1.0
            evolution_data['lr_ratios'].append(ratio)
        else:
            evolution_data['base_lrs'].append(0.0)
            evolution_data['lr_ratios'].append(1.0)
    
    # Create visualization
    if save_path or len(steps) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Effective vs Base LR
        axes[0].plot(steps, evolution_data['base_lrs'], label='Base LR', linestyle='--')
        axes[0].plot(steps, evolution_data['global_effective_lrs'], label='Effective LR')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Learning Rate')
        axes[0].set_title('Learning Rate Evolution')
        axes[0].legend()
        axes[0].set_yscale('log')
        
        # LR Ratio
        axes[1].plot(steps, evolution_data['lr_ratios'])
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Effective LR / Base LR')
        axes[1].set_title('Learning Rate Adaptation Ratio')
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No adaptation')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LR evolution plot saved to {save_path}")
        else:
            plt.show()
    
    return evolution_data


class LearningRateMonitor:
    """
    Monitor effective learning rate during training.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, log_frequency: int = 100):
        """
        Initialize LR monitor.
        
        Args:
            optimizer: Optimizer to monitor
            log_frequency: How often to compute and log LR (in steps)
        """
        self.optimizer = optimizer
        self.log_frequency = log_frequency
        self.lr_history = []
        self.step_history = []
        
    def step(self, current_step: int) -> Optional[Dict[str, float]]:
        """
        Monitor step - compute LR if it's time.
        
        Args:
            current_step: Current training step
            
        Returns:
            LR info if computed, None otherwise
        """
        if current_step % self.log_frequency == 0:
            lr_info = compute_effective_lr(self.optimizer, current_step)
            
            self.lr_history.append(lr_info)
            self.step_history.append(current_step)
            
            return lr_info
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of LR evolution."""
        if not self.lr_history:
            return {'error': 'No LR data collected yet'}
        
        global_lrs = [info['global_effective_lr'] for info in self.lr_history]
        
        return {
            'total_steps_monitored': len(self.lr_history),
            'step_range': (min(self.step_history), max(self.step_history)),
            'lr_range': (min(global_lrs), max(global_lrs)),
            'final_effective_lr': global_lrs[-1],
            'lr_trend': 'increasing' if global_lrs[-1] > global_lrs[0] else 'decreasing',
            'avg_effective_lr': np.mean(global_lrs)
        }
    
    def save_history(self, path: str):
        """Save LR monitoring history."""
        import json
        
        data = {
            'steps': self.step_history,
            'lr_history': self.lr_history,
            'summary': self.get_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"LR monitoring history saved to {path}")


# Example usage and testing
if __name__ == "__main__":
    # Create a simple model and optimizer for testing
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Simulate some training steps
    for step in range(100):
        # Dummy forward pass
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Monitor LR every 25 steps
        if step % 25 == 0:
            lr_info = compute_effective_lr(optimizer, step)
            print(f"Step {step}: Effective LR = {lr_info['global_effective_lr']:.6f}")
    
    # Test LR evolution tracking
    steps = list(range(0, 100, 10))
    evolution = track_lr_evolution(optimizer, steps)
    print(f"LR evolution over {len(steps)} steps completed")