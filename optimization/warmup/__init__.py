#!/usr/bin/env python3
"""
Warmup Scheduling for Delayed Generalization Research

This module provides learning rate warmup implementations that help with
delayed generalization phenomena, particularly grokking and phase transitions.
"""

from .warmup_schedulers import (
    LinearWarmup,
    CosineWarmup,
    ExponentialWarmup,
    WarmupScheduler,
    create_warmup_scheduler
)

__all__ = [
    'LinearWarmup',
    'CosineWarmup', 
    'ExponentialWarmup',
    'WarmupScheduler',
    'create_warmup_scheduler'
]