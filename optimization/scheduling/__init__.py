#!/usr/bin/env python3
"""
Learning Rate Scheduling for Delayed Generalization Research

This module provides advanced learning rate scheduling implementations that
help control delayed generalization phenomena across different training scenarios.
"""

from .lr_schedulers import (
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

__all__ = [
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
]