#!/usr/bin/env python3
"""
Optimization Package for Delayed Generalization Research

This package provides enhanced optimization techniques that significantly impact
delayed generalization phenomena like grokking, simplicity bias, and phase transitions.

Features:
- Enhanced optimizers with advanced features
- Learning rate schedulers optimized for delayed generalization
- Warmup strategies for stable training initialization
- Regularization techniques for controlling memorization vs generalization
- Momentum and batch strategies for different training scenarios

Usage:
    from optimization.warmup import create_warmup_scheduler
    from optimization.scheduling import create_phenomenon_scheduler
    from optimization.regularization import create_regularization_config
"""

# Import warmup strategies
try:
    from .warmup import (
        LinearWarmup,
        CosineWarmup,
        ExponentialWarmup,
        WarmupScheduler,
        create_warmup_scheduler,
        create_grokking_warmup,
        create_bias_mitigation_warmup
    )
    WARMUP_AVAILABLE = True
except ImportError:
    WARMUP_AVAILABLE = False

# Import advanced scheduling
try:
    from .scheduling import (
        CosineAnnealingScheduler,
        AdaptiveStepDecay,
        CyclicLR,
        CosineAnnealingWarmRestarts,
        GrokkingScheduler,
        BiasScheduler,
        PhaseTransitionScheduler,
        LRRangeFinder,
        AdaptiveMomentumScheduler,
        create_phenomenon_scheduler
    )
    SCHEDULING_AVAILABLE = True
except ImportError:
    SCHEDULING_AVAILABLE = False

# Import regularization techniques
try:
    from .regularization import (
        AdaptiveWeightDecay,
        GradualDropout,
        AttentionDropout,
        BiasedFeatureDropout,
        AntiBiasAugmentation,
        ProgressiveAugmentation,
        SpectralRegularizer,
        InformationBottleneck,
        create_regularization_config,
        track_regularization_metrics
    )
    REGULARIZATION_AVAILABLE = True
except ImportError:
    REGULARIZATION_AVAILABLE = False

# Legacy imports for backward compatibility
try:
    from .enhanced_optimizers import (
        create_enhanced_optimizer,
        AdamWWithLookAhead,
        SGDWithNesterovMomentumScheduler,
        RMSpropWithGradientCentralization,
        LionOptimizer,
        AdaBelief,
        configure_optimizer_for_phenomenon
    )
    ENHANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ENHANCED_OPTIMIZERS_AVAILABLE = False

try:
    from .lr_schedulers import (
        DelayedGeneralizationScheduler,
        GradualWarmupScheduler,
        CosineAnnealingWithRestarts,
        ExponentialDecayWithPlateau,
        CyclicLRWithMomentum,
        AdaptiveLRScheduler
    )
    LR_SCHEDULERS_AVAILABLE = True
except ImportError:
    LR_SCHEDULERS_AVAILABLE = False

try:
    from .batch_strategies import (
        AdaptiveBatchSize,
        GradientNormBatchScheduler,
        CurriculumBatchScheduler
    )
    BATCH_STRATEGIES_AVAILABLE = True
except ImportError:
    BATCH_STRATEGIES_AVAILABLE = False

# Build __all__ dynamically based on available modules
__all__ = []

if WARMUP_AVAILABLE:
    __all__.extend([
        'LinearWarmup',
        'CosineWarmup',
        'ExponentialWarmup',
        'WarmupScheduler',
        'create_warmup_scheduler',
        'create_grokking_warmup',
        'create_bias_mitigation_warmup'
    ])

if SCHEDULING_AVAILABLE:
    __all__.extend([
        'CosineAnnealingScheduler',
        'AdaptiveStepDecay',
        'CyclicLR',
        'CosineAnnealingWarmRestarts',
        'GrokkingScheduler',
        'BiasScheduler',
        'PhaseTransitionScheduler',
        'LRRangeFinder',
        'AdaptiveMomentumScheduler',
        'create_phenomenon_scheduler'
    ])

if REGULARIZATION_AVAILABLE:
    __all__.extend([
        'AdaptiveWeightDecay',
        'GradualDropout',
        'AttentionDropout',
        'BiasedFeatureDropout',
        'AntiBiasAugmentation',
        'ProgressiveAugmentation',
        'SpectralRegularizer',
        'InformationBottleneck',
        'create_regularization_config',
        'track_regularization_metrics'
    ])

if ENHANCED_OPTIMIZERS_AVAILABLE:
    __all__.extend([
        'create_enhanced_optimizer',
        'AdamWWithLookAhead',
        'SGDWithNesterovMomentumScheduler',
        'RMSpropWithGradientCentralization',
        'LionOptimizer',
        'AdaBelief',
        'configure_optimizer_for_phenomenon'
    ])

if LR_SCHEDULERS_AVAILABLE:
    __all__.extend([
        'DelayedGeneralizationScheduler',
        'GradualWarmupScheduler',
        'CosineAnnealingWithRestarts',
        'ExponentialDecayWithPlateau',
        'CyclicLRWithMomentum',
        'AdaptiveLRScheduler'
    ])

if BATCH_STRATEGIES_AVAILABLE:
    __all__.extend([
        'AdaptiveBatchSize',
        'GradientNormBatchScheduler',
        'CurriculumBatchScheduler'
    ])

# Convenience functions for complete training setup
def create_complete_training_setup(
    model,
    phenomenon_type: str = 'grokking',
    optimizer_type: str = 'adamw',
    total_epochs: int = 10000,
    **kwargs
):
    """
    Create a complete training setup for delayed generalization experiments.
    
    Args:
        model: PyTorch model
        phenomenon_type: 'grokking', 'simplicity_bias', 'phase_transitions'
        optimizer_type: 'adamw', 'sgd', 'lion', etc.
        total_epochs: Total training epochs
        **kwargs: Additional configuration
        
    Returns:
        Dictionary with optimizer, scheduler, regularization config
    """
    setup = {
        'phenomenon_type': phenomenon_type,
        'total_epochs': total_epochs
    }
    
    # Create scheduler if available
    if SCHEDULING_AVAILABLE:
        try:
            scheduler = create_phenomenon_scheduler(
                phenomenon_type,
                model.parameters() if hasattr(model, 'parameters') else None,
                total_epochs=total_epochs,
                **kwargs
            )
            setup['scheduler'] = scheduler
        except Exception as e:
            print(f"Warning: Could not create scheduler: {e}")
    
    # Create regularization config if available
    if REGULARIZATION_AVAILABLE:
        try:
            reg_config = create_regularization_config(phenomenon_type)
            setup['regularization_config'] = reg_config
        except Exception as e:
            print(f"Warning: Could not create regularization config: {e}")
    
    # Create warmup if available
    if WARMUP_AVAILABLE:
        try:
            if phenomenon_type == 'grokking':
                warmup = create_grokking_warmup(None, total_epochs)
                setup['warmup'] = warmup
            elif phenomenon_type == 'simplicity_bias':
                warmup = create_bias_mitigation_warmup(None, total_epochs)
                setup['warmup'] = warmup
        except Exception as e:
            print(f"Warning: Could not create warmup: {e}")
    
    return setup

# Add convenience function to __all__
__all__.append('create_complete_training_setup')

# Version info
__version__ = "0.2.0"
__author__ = "Delayed Generalization Research Team"

def get_available_modules():
    """Get information about which optimization modules are available."""
    return {
        'warmup': WARMUP_AVAILABLE,
        'scheduling': SCHEDULING_AVAILABLE,
        'regularization': REGULARIZATION_AVAILABLE,
        'enhanced_optimizers': ENHANCED_OPTIMIZERS_AVAILABLE,
        'lr_schedulers': LR_SCHEDULERS_AVAILABLE,
        'batch_strategies': BATCH_STRATEGIES_AVAILABLE
    }