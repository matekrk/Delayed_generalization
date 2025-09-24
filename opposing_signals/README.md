# Opposing Signals Analysis

This directory contains tools for analyzing opposing gradient signals during training, based on the work of Rosenfeld & Risteski (2023): "Outliers with Opposing Signals".

## Overview

During training, some examples may have opposing signal gradients - gradients that point in opposite directions from the majority of training examples. These examples can:

1. **Slow down convergence** by creating conflicting gradient directions
2. **Indicate distribution shift** or spurious correlations
3. **Help understand delayed generalization** patterns

## Contents

- `opposing_signals_analysis.ipynb` - Unified notebook for gradient analysis with visualizations
- `gradient_tracker.py` - Tools for tracking gradient directions and magnitudes
- `signal_detector.py` - Detection algorithms for opposing signal examples
- `visualization.py` - Plotting utilities for gradient and loss dynamics

## Key Features

### Gradient Direction Analysis
- Track gradient directions for individual examples
- Detect examples with opposing gradients
- Visualize gradient flow patterns

### Loss Dynamics Tracking
- Monitor examples with significant loss increases/decreases
- Identify training instabilities
- Track delayed generalization patterns

### Interactive Visualizations
- Real-time gradient direction plots
- Loss trajectory animations
- Example-specific analysis

## Usage

```python
from opposing_signals import GradientTracker, SignalDetector

# Initialize tracker
tracker = GradientTracker(model, data_loader)

# Track gradients during training
for epoch in range(num_epochs):
    # ... training code ...
    tracker.track_epoch(epoch, model, data_loader)
    
    # Detect opposing signals
    opposing_examples = tracker.detect_opposing_signals()
    
    # Analyze results
    tracker.plot_gradient_dynamics()
```

## References

- Rosenfeld & Risteski (2023). "Outliers with Opposing Signals"
- Fort et al. (2019). "Deep learning versus kernel learning: an empirical study of loss landscape evolution during training"