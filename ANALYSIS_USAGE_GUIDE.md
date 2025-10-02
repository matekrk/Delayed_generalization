# Advanced Analysis Usage Guide

This guide provides comprehensive instructions for using the advanced analysis features in the Delayed Generalization repository.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [During-Training Analysis](#during-training-analysis)
4. [Post-Hoc Analysis](#post-hoc-analysis)
5. [Multi-Phenomenon Comparison](#multi-phenomenon-comparison)
6. [Integration with Existing Code](#integration-with-existing-code)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn
```

## Quick Start

### Minimal Example

```python
from data_attribution.analysis import LearningDynamicsAnalyzer
import torch.nn as nn

# Initialize analyzer
analyzer = LearningDynamicsAnalyzer(num_classes=10)

# In your training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Your training code here
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Track learning dynamics
        batch_indices = torch.arange(
            batch_idx * batch_size,
            min((batch_idx + 1) * batch_size, len(train_dataset))
        )
        analyzer.track_batch(model, inputs, targets, batch_indices, epoch)
    
    # Compute statistics at end of epoch
    analyzer.compute_epoch_statistics(train_loader, model, epoch)

# Analyze and visualize
analysis = analyzer.analyze_learning_patterns()
analyzer.plot_learning_dynamics(save_dir='./results')
print(f"Hard examples: {len(analysis['hard_examples'])}")
```

## During-Training Analysis

For real-time monitoring during training, use multiple analyzers together:

### Complete Example

```python
from data_attribution.analysis import (
    LearningDynamicsAnalyzer,
    GradientFlowAnalyzer,
    MemorizationDetector
)

# Initialize analyzers
dynamics = LearningDynamicsAnalyzer(
    num_classes=10,
    track_examples=1000,
    device=device
)

gradients = GradientFlowAnalyzer(
    model=model,
    track_layers=None  # Track all layers
)

memorization = MemorizationDetector(
    num_classes=10,
    device=device
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradients BEFORE optimizer.step()
        if batch_idx % 10 == 0:
            gradients.track_batch_gradients(loss, epoch, batch_idx)
        
        optimizer.step()
        
        # Track learning dynamics
        batch_indices = torch.arange(...)
        dynamics.track_batch(model, inputs, targets, batch_indices, epoch)
    
    # End-of-epoch analysis
    dynamics.compute_epoch_statistics(train_loader, model, epoch)
    gradients.compute_epoch_statistics(epoch)
    
    if epoch % 5 == 0:
        mem_metrics = memorization.compute_memorization_scores(
            model, train_loader, test_loader, epoch
        )
        print(f"Epoch {epoch}: Memorization={mem_metrics['memorization_score']:.4f}")

# Generate reports
dynamics.plot_learning_dynamics(save_dir='./analysis')
gradients.plot_gradient_flow(save_dir='./analysis')
memorization.plot_memorization(save_dir='./analysis')
```

### When to Use Each Analyzer

- **LearningDynamicsAnalyzer**: Always use for example-level insights
- **GradientFlowAnalyzer**: Use when debugging training instabilities
- **MemorizationDetector**: Use when studying overfitting/generalization
- **FeatureEvolutionTracker**: Use periodically (every N epochs) to save memory

## Post-Hoc Analysis

For analyzing saved checkpoints after training:

### Loading Checkpoints

```python
from data_attribution.analysis import (
    PhaseTransitionAttributor,
    FeatureEvolutionTracker,
    MemorizationDetector
)

# Load checkpoints
checkpoint_dir = './checkpoints'
checkpoints = {}

for epoch in [0, 10, 20, 30, 40, 50]:
    model = YourModel()
    model.load_state_dict(torch.load(f'{checkpoint_dir}/model_epoch_{epoch}.pth'))
    model.eval()
    checkpoints[epoch] = model

# Analyze memorization evolution
mem_detector = MemorizationDetector(num_classes=10)

for epoch, model in checkpoints.items():
    metrics = mem_detector.compute_memorization_scores(
        model, train_loader, test_loader, epoch
    )
    print(f"Epoch {epoch}: Train={metrics['train_accuracy']:.4f}, "
          f"Test={metrics['test_accuracy']:.4f}")

# Analyze memorization patterns
mem_analysis = mem_detector.analyze_memorization_evolution()
print(f"Transition at epoch: {mem_analysis['transition_epoch']}")

mem_detector.plot_memorization(save_dir='./analysis')
```

### Feature Evolution Analysis

```python
# Track feature evolution across checkpoints
feature_tracker = FeatureEvolutionTracker(
    model=YourModel(),  # Fresh model for tracking
    layer_names=['layer3', 'layer4'],
    device=device
)

for epoch, checkpoint_model in checkpoints.items():
    # Load weights into tracker's model
    feature_tracker.model.load_state_dict(checkpoint_model.state_dict())
    
    # Track features
    feature_tracker.track_epoch(train_loader, epoch, sample_limit=500)

# Analyze evolution
analysis = feature_tracker.analyze_evolution()

# Detect transitions
transitions = feature_tracker.detect_phase_transitions(
    'layer4',
    metric='silhouette_score'
)
print(f"Phase transitions at epochs: {transitions}")

# Visualize
feature_tracker.visualize_evolution('layer4', save_dir='./analysis', method='pca')
feature_tracker.plot_metrics_evolution('layer4', save_dir='./analysis')
```

### Phase Transition Attribution

```python
# Identify which examples drive transitions
pt_attributor = PhaseTransitionAttributor(
    model_checkpoints={
        10: checkpoints[10],  # Before transition
        40: checkpoints[40]   # After transition
    },
    train_data=train_dataset,
    test_data=test_dataset
)

# Compute attributions
attributions = pt_attributor.compute_transition_attributions(
    pre_transition_epoch=10,
    post_transition_epoch=40
)

# Find critical examples
critical = pt_attributor.find_critical_examples(attributions, top_k=100)
analysis = pt_attributor.analyze_critical_examples(critical)

print(f"Critical examples: {len(critical)}")
print(f"Label distribution: {analysis['label_distribution']}")
```

## Multi-Phenomenon Comparison

To compare different phenomena side-by-side:

```python
from data_attribution.analysis import LearningDynamicsAnalyzer

# Train multiple models
models = {
    'grokking': train_grokking_model(),
    'bias': train_biased_model(),
    'normal': train_normal_model()
}

# Analyze each
analyses = {}

for name, (model, train_loader, test_loader) in models.items():
    analyzer = LearningDynamicsAnalyzer(num_classes=10)
    
    # Assume models are already trained with tracking
    # Load saved analyzer state
    analyzer = LearningDynamicsAnalyzer.load(f'./saved/{name}_analyzer.pkl')
    
    analyses[name] = analyzer.analyze_learning_patterns()

# Compare
for name, analysis in analyses.items():
    print(f"\n{name}:")
    print(f"  Hard examples: {len(analysis['hard_examples'])}")
    print(f"  Forgetting events: {analysis['forgetting_statistics']['examples_with_forgetting']}")
```

See `examples/analysis_multi_phenomenon.py` for a complete example.

## Integration with Existing Code

### Minimal Changes Required

To add analysis to existing training code, you only need to:

1. **Initialize analyzers** before training
2. **Add tracking calls** in your training loop
3. **Generate visualizations** after training

### Example Integration

```python
# BEFORE: Your existing training code
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# AFTER: With analysis integrated
from data_attribution.analysis import LearningDynamicsAnalyzer

analyzer = LearningDynamicsAnalyzer(num_classes=10)  # ADD THIS

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # ADD THIS
        batch_indices = torch.arange(
            batch_idx * batch_size,
            min((batch_idx + 1) * batch_size, len(train_dataset))
        )
        analyzer.track_batch(model, inputs, targets, batch_indices, epoch)
    
    analyzer.compute_epoch_statistics(train_loader, model, epoch)  # ADD THIS

# ADD THIS
analyzer.plot_learning_dynamics(save_dir='./results')
```

## Best Practices

### 1. Memory Management

When tracking large numbers of examples:

```python
# Limit tracked examples
analyzer = LearningDynamicsAnalyzer(
    num_classes=10,
    track_examples=1000  # Only track first 1000 examples
)

# For feature tracking, use sample_limit
feature_tracker.track_epoch(train_loader, epoch, sample_limit=500)
```

### 2. Periodic Tracking

For expensive analyses, track periodically:

```python
for epoch in range(num_epochs):
    # ... training ...
    
    # Track every epoch
    dynamics.compute_epoch_statistics(train_loader, model, epoch)
    
    # Track every 5 epochs
    if epoch % 5 == 0:
        memorization.compute_memorization_scores(...)
    
    # Track every 10 epochs
    if epoch % 10 == 0:
        feature_tracker.track_epoch(...)
```

### 3. Save Analysis State

Save analyzer state to resume or share analysis:

```python
# During training
if epoch % 10 == 0:
    analyzer.save(f'./checkpoints/analyzer_epoch_{epoch}.pkl')

# Later, load and continue
analyzer = LearningDynamicsAnalyzer.load('./checkpoints/analyzer_epoch_50.pkl')
```

### 4. Clean Up

Remove hooks when done with feature tracking:

```python
feature_tracker = FeatureEvolutionTracker(...)
# ... use tracker ...
feature_tracker.remove_hooks()  # Clean up before deleting
```

## Troubleshooting

### Out of Memory

**Problem**: OOM when tracking many examples

**Solution**:
```python
# Reduce tracked examples
analyzer = LearningDynamicsAnalyzer(track_examples=500)  # Instead of 1000

# Track gradients less frequently
if batch_idx % 20 == 0:  # Instead of every batch
    gradients.track_batch_gradients(...)
```

### Slow Training

**Problem**: Training slows down significantly

**Solution**:
```python
# Track only what you need
# Skip expensive analyses during training
# Do comprehensive analysis post-hoc instead

# Use eval mode for tracking
model.eval()
analyzer.track_batch(...)
model.train()  # Return to training mode
```

### Import Errors

**Problem**: `ModuleNotFoundError` for analysis modules

**Solution**:
```python
import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data_attribution.analysis import LearningDynamicsAnalyzer
```

### Hook Conflicts

**Problem**: Multiple feature trackers interfere

**Solution**:
```python
# Remove hooks before creating new tracker
tracker1.remove_hooks()

# Or reuse the same tracker
tracker = FeatureEvolutionTracker(model, ...)
# ... use it ...
# No need for multiple trackers on same model
```

## Examples Directory

See complete, runnable examples:

- `examples/analysis_during_training.py` - Real-time monitoring
- `examples/analysis_post_hoc.py` - Checkpoint analysis
- `examples/analysis_multi_phenomenon.py` - Comparative analysis

Run them with:
```bash
python examples/analysis_during_training.py
```

## Further Reading

- [Analysis Module README](data_attribution/analysis/README.md)
- [Data Attribution README](data_attribution/README.md)
- Individual analyzer docstrings in source files
