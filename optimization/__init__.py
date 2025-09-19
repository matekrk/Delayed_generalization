#!/usr/bin/env python3
"""
Enhanced Optimizers for Delayed Generalization Research

This module provides optimizers with additional features for studying delayed generalization:
- Enhanced AdamW with configurable weight decay schedules
- SGD with adaptive momentum and learning rate schedules  
- Custom LR schedulers for phase transition detection
- Gradient analysis and logging utilities
"""

from .enhanced_optimizers import (
    EnhancedAdamW,
    EnhancedSGD,
    create_optimizer_with_scheduler
)

from .lr_schedulers import (
    WarmupCosineScheduler,
    PhaseTransitionScheduler,
    AdaptiveLRScheduler
)

__all__ = [
    'EnhancedAdamW',
    'EnhancedSGD', 
    'create_optimizer_with_scheduler',
    'WarmupCosineScheduler',
    'PhaseTransitionScheduler',
    'AdaptiveLRScheduler'
]