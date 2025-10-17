#!/usr/bin/env python3
"""
Factory functions for creating optimizers with flexible configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

from .enhanced_adamw import EnhancedAdamW
from .enhanced_sgd import EnhancedSGD


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    use_enhanced: bool = True,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer with flexible configuration.
    
    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ('adamw', 'sgd', 'enhanced_adamw', 'enhanced_sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_enhanced: Whether to use enhanced versions by default
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Configured optimizer instance
    """
    
    # Normalize optimizer type
    opt_type = optimizer_type.lower().replace('_', '').replace('-', '')
    
    # AdamW variants
    if opt_type in ['adamw', 'enhancedadamw']:
        if opt_type == 'enhancedadamw' or (opt_type == 'adamw' and use_enhanced):
            # Use enhanced version
            return EnhancedAdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                amsgrad=kwargs.get('amsgrad', False),
                # Enhanced features
                grad_clip_norm=kwargs.get('grad_clip_norm', None),
                adaptive_weight_decay=kwargs.get('adaptive_weight_decay', False),
                warmup_steps=kwargs.get('warmup_steps', 0),
                log_grad_stats=kwargs.get('log_grad_stats', False)
            )
        else:
            # Use vanilla PyTorch version
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                amsgrad=kwargs.get('amsgrad', False)
            )
    
    # SGD variants
    elif opt_type in ['sgd', 'enhancedsgd']:
        if opt_type == 'enhancedsgd' or (opt_type == 'sgd' and use_enhanced):
            # Use enhanced version
            return EnhancedSGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                dampening=kwargs.get('dampening', 0),
                nesterov=kwargs.get('nesterov', False),
                # Enhanced features
                adaptive_momentum=kwargs.get('adaptive_momentum', False),
                grad_clip_norm=kwargs.get('grad_clip_norm', None),
                momentum_decay=kwargs.get('momentum_decay', 0.99),
                log_grad_stats=kwargs.get('log_grad_stats', False)
            )
        else:
            # Use vanilla PyTorch version
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                dampening=kwargs.get('dampening', 0),
                nesterov=kwargs.get('nesterov', False)
            )
    
    # Adam (vanilla)
    elif opt_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            amsgrad=kwargs.get('amsgrad', False)
        )
    
    # RMSprop
    elif opt_type == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            momentum=kwargs.get('momentum', 0),
            centered=kwargs.get('centered', False)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_default_optimizer(
    model: nn.Module,
    phenomenon_type: str = 'general',
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get default optimizer configuration for different phenomena.
    
    Args:
        model: The model to optimize
        phenomenon_type: Type of phenomenon ('grokking', 'simplicity_bias', 'robustness', 'general')
        **kwargs: Override default parameters
        
    Returns:
        Configured optimizer with phenomenon-specific defaults
    """
    
    # Default configurations for different phenomena
    configs = {
        'grokking': {
            'optimizer_type': 'enhanced_adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-2,  # High weight decay is crucial for grokking
            'grad_clip_norm': 1.0,
            'log_grad_stats': True
        },
        'simplicity_bias': {
            'optimizer_type': 'enhanced_adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,  # Lower weight decay for bias studies
            'adaptive_weight_decay': True,
            'log_grad_stats': True
        },
        'robustness': {
            'optimizer_type': 'enhanced_sgd',
            'learning_rate': 1e-2,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'adaptive_momentum': True,
            'log_grad_stats': True
        },
        'general': {
            'optimizer_type': 'enhanced_adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'log_grad_stats': True
        }
    }
    
    # Get base config and update with user overrides
    config = configs.get(phenomenon_type, configs['general']).copy()
    config.update(kwargs)
    
    return create_optimizer(model, **config)


def get_optimizer_stats(optimizer) -> Dict[str, Any]:
    """
    Get statistics from any optimizer.
    
    Args:
        optimizer: The optimizer instance
        
    Returns:
        Dictionary of optimizer statistics
    """
    if hasattr(optimizer, 'get_stats'):
        # Enhanced optimizer with built-in stats
        return optimizer.get_stats()
    else:
        # Basic stats for standard optimizers
        return {
            'step_count': getattr(optimizer, 'step_count', 0),
            'current_lr': optimizer.param_groups[0]['lr'],
            'optimizer_type': type(optimizer).__name__,
            'has_enhanced_features': False,
            'param_groups': len(optimizer.param_groups)
        }


def create_optimizer_with_scheduler(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    scheduler_type: str = 'cosine',
    **kwargs
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Create optimizer and scheduler together (backward compatibility).
    
    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        scheduler_type: Type of scheduler
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **{k: v for k, v in kwargs.items() if k not in ['total_steps', 'warmup_steps', 'gamma']}
    )
    
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
        # For now, just use cosine (full warmup scheduler would be in schedulers module)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    elif scheduler_type.lower() == 'none' or scheduler_type is None:
        scheduler = None
    else:
        scheduler = None
    
    return optimizer, scheduler