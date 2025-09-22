#!/usr/bin/env python3
"""
Enhanced Optimizers for Delayed Generalization Research

This module provides optimizers with additional features for studying delayed generalization:
- Enhanced AdamW with configurable weight decay schedules
- SGD with adaptive momentum and learning rate schedules  
- Custom LR schedulers for phase transition detection
- Gradient analysis and logging utilities
- Factory functions for easy optimizer creation
"""

# Import from new organized structure
from .optimizers import (
    EnhancedAdamW,
    EnhancedSGD,
    create_optimizer,
    get_default_optimizer,
    get_optimizer_stats,
    create_optimizer_with_scheduler
)

# Import from existing lr_schedulers (maintain backward compatibility)
try:
    from .lr_schedulers import (
        WarmupCosineScheduler,
        PhaseTransitionScheduler,
        AdaptiveLRScheduler
    )
except ImportError:
    # If lr_schedulers module doesn't exist yet, provide placeholders
    WarmupCosineScheduler = None
    PhaseTransitionScheduler = None
    AdaptiveLRScheduler = None

# Backward compatibility - import from old enhanced_optimizers
try:
    from .enhanced_optimizers import get_optimizer_stats as _legacy_get_stats
except ImportError:
    _legacy_get_stats = get_optimizer_stats

__all__ = [
    'EnhancedAdamW',
    'EnhancedSGD', 
    'create_optimizer',
    'get_default_optimizer',
    'get_optimizer_stats',
    'create_optimizer_with_scheduler',
    'WarmupCosineScheduler',
    'PhaseTransitionScheduler',
    'AdaptiveLRScheduler'
]