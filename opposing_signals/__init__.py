#!/usr/bin/env python3
"""
Opposing Signals Analysis for Delayed Generalization Research

This module provides tools for analyzing opposing gradient signals during training,
based on the work of Rosenfeld & Risteski (2023): "Outliers with Opposing Signals".
"""

from .gradient_tracker import GradientTracker
from .signal_detector import SignalDetector, OpposingSignalsAnalysis
from .visualization import OpposingSignalsVisualizer

__all__ = [
    'GradientTracker',
    'SignalDetector',
    'OpposingSignalsAnalysis',
    'OpposingSignalsVisualizer'
]

__version__ = "0.1.0"
__author__ = "Delayed Generalization Research Team"