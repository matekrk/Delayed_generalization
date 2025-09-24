#!/usr/bin/env python3
"""
Regularization Techniques for Delayed Generalization Research

This module provides advanced regularization implementations designed to
control and enhance delayed generalization phenomena.
"""

from .regularizers import (
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

__all__ = [
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
]