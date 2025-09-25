#!/usr/bin/env python3
"""
Enhanced Optimizers for Delayed Generalization Research

This module provides comprehensive optimization techniques for studying delayed generalization:
- Enhanced optimizers with configurable weight decay schedules
- Warmup scheduling strategies for stable training
- Adaptive scheduling based on metrics and phase transitions
- Regularization techniques to control overfitting
- Factory functions for easy optimizer creation
"""

# Import from existing modules (maintain backward compatibility)
try:
    from .optimizers import (
        EnhancedAdamW,
        EnhancedSGD,
        create_optimizer,
        get_default_optimizer,
        get_optimizer_stats,
        create_optimizer_with_scheduler
    )
except ImportError:
    # Fallback imports if new structure not available
    try:
        from .enhanced_optimizers import get_optimizer_stats
        get_default_optimizer = None
        create_optimizer = None
        create_optimizer_with_scheduler = None
        EnhancedAdamW = None
        EnhancedSGD = None
    except ImportError:
        get_optimizer_stats = None
        get_default_optimizer = None
        create_optimizer = None
        create_optimizer_with_scheduler = None
        EnhancedAdamW = None
        EnhancedSGD = None

# Import from existing lr_schedulers (maintain backward compatibility)
try:
    from .lr_schedulers import (
        WarmupCosineScheduler,
        PhaseTransitionScheduler,
        AdaptiveLRScheduler
    )
except ImportError:
    WarmupCosineScheduler = None
    PhaseTransitionScheduler = None
    AdaptiveLRScheduler = None

# Import new warmup scheduling techniques
try:
    from .warmup import (
        LinearWarmup,
        ExponentialWarmup,
        PolynomialWarmup,
        WarmupWrapper,
        get_warmup_scheduler
    )
except ImportError:
    LinearWarmup = None
    ExponentialWarmup = None
    PolynomialWarmup = None
    WarmupWrapper = None
    get_warmup_scheduler = None

# Import new adaptive scheduling techniques
try:
    from .scheduling import (
        MetricAdaptiveScheduler,
        PhaseTransitionScheduler as NewPhaseTransitionScheduler,
        GradientAdaptiveScheduler,
        LossPlateauScheduler
    )
except ImportError:
    MetricAdaptiveScheduler = None
    NewPhaseTransitionScheduler = None
    GradientAdaptiveScheduler = None
    LossPlateauScheduler = None

# Import new regularization techniques
try:
    from .regularization import (
        WeightDecayRegularizer,
        DropoutRegularizer,
        SpectralNormRegularizer,
        GradientPenaltyRegularizer,
        EarlyStopping
    )
except ImportError:
    WeightDecayRegularizer = None
    DropoutRegularizer = None
    SpectralNormRegularizer = None
    GradientPenaltyRegularizer = None
    EarlyStopping = None

__all__ = [
    # Legacy optimizers
    'EnhancedAdamW',
    'EnhancedSGD', 
    'create_optimizer',
    'get_default_optimizer',
    'get_optimizer_stats',
    'create_optimizer_with_scheduler',
    # Legacy schedulers
    'WarmupCosineScheduler',
    'PhaseTransitionScheduler',
    'AdaptiveLRScheduler',
    # New warmup techniques
    'LinearWarmup',
    'ExponentialWarmup',
    'PolynomialWarmup',
    'WarmupWrapper',
    'get_warmup_scheduler',
    # New adaptive scheduling
    'MetricAdaptiveScheduler',
    'NewPhaseTransitionScheduler',
    'GradientAdaptiveScheduler', 
    'LossPlateauScheduler',
    # New regularization
    'WeightDecayRegularizer',
    'DropoutRegularizer',
    'SpectralNormRegularizer',
    'GradientPenaltyRegularizer',
    'EarlyStopping'
]