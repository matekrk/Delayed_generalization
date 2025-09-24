#!/usr/bin/env python3
"""
Regularization Techniques for Delayed Generalization Research

This module provides various regularization methods that can influence
delayed generalization patterns and training dynamics.
"""

from .regularizers import (
    WeightDecayRegularizer,
    DropoutRegularizer,
    SpectralNormRegularizer,
    GradientPenaltyRegularizer,
    EarlyStopping
)

from .adaptive_regularization import (
    AdaptiveWeightDecay,
    AdaptiveDropout,
    ProgressiveRegularization,
    MetricBasedRegularization
)

__all__ = [
    # Basic regularizers
    'WeightDecayRegularizer',
    'DropoutRegularizer',
    'SpectralNormRegularizer', 
    'GradientPenaltyRegularizer',
    'EarlyStopping',
    # Adaptive regularizers
    'AdaptiveWeightDecay',
    'AdaptiveDropout',
    'ProgressiveRegularization',
    'MetricBasedRegularization'
]