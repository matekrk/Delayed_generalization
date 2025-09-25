#!/usr/bin/env python3
"""
Advanced Scheduling Techniques for Delayed Generalization Research

This module provides sophisticated scheduling strategies that can help
trigger and study delayed generalization phenomena.
"""

from .adaptive_schedulers import (
    MetricAdaptiveScheduler,
    PhaseTransitionScheduler,
    GradientAdaptiveScheduler,
    LossPlateauScheduler
)

from .specialized_schedulers import (
    GrokkingScheduler,
    SimplicityBiasScheduler,
    ContinualLearningScheduler,
    MultiPhaseScheduler
)

__all__ = [
    # Adaptive schedulers
    'MetricAdaptiveScheduler',
    'PhaseTransitionScheduler', 
    'GradientAdaptiveScheduler',
    'LossPlateauScheduler',
    # Specialized schedulers
    'GrokkingScheduler',
    'SimplicityBiasScheduler',
    'ContinualLearningScheduler',
    'MultiPhaseScheduler'
]