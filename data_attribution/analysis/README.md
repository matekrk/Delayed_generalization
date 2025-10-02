# Analysis Module for Delayed Generalization Research

This module provides comprehensive tools for analyzing delayed generalization phenomena during and after training.

## 📋 Overview

The analysis module includes six powerful tools that work together to provide deep insights into how models learn:

1. **LearningDynamicsAnalyzer** - Track example-level learning patterns
2. **FeatureEvolutionTracker** - Monitor representation quality evolution
3. **GradientFlowAnalyzer** - Analyze gradient dynamics
4. **MemorizationDetector** - Distinguish memorization from generalization
5. **PhaseTransitionAttributor** - Identify critical examples for transitions
6. **BiasAttributor** - Detect spurious correlation learning

## 🚀 Quick Start

### Installation

The analysis tools are part of the main repository. Ensure you have the required dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Basic Usage

```python
from data_attribution.analysis import LearningDynamicsAnalyzer

# Initialize
analyzer = LearningDynamicsAnalyzer(num_classes=10, device='cuda')

# Track during training
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ... training step ...
        analyzer.track_batch(model, inputs, targets, batch_indices, epoch)
    
    analyzer.compute_epoch_statistics(train_loader, model, epoch)

# Analyze and visualize
analysis = analyzer.analyze_learning_patterns()
analyzer.plot_learning_dynamics(save_dir='./results')
```

## 📊 Analysis Tools

### 1. Learning Dynamics Analyzer

**Purpose**: Track how individual examples are learned over time

**Key Features**:
- Prediction confidence evolution
- Example difficulty scoring
- Forgetting event detection
- Learning speed analysis
- Class-wise performance tracking

**Use Cases**:
- Identifying hard-to-learn examples
- Detecting curriculum learning opportunities
- Understanding forgetting patterns in continual learning
- Analyzing class imbalances

**Outputs**:
- Easy/medium/hard example categorization
- Forgetting statistics
- Learning speed distribution
- Comprehensive visualizations

### 2. Feature Evolution Tracker

**Purpose**: Monitor how learned representations change during training

**Key Features**:
- Representation quality metrics (silhouette score, separability)
- Effective rank computation
- Phase transition detection
- PCA/t-SNE visualization
- Layer-wise analysis

**Use Cases**:
- Understanding when features become useful
- Detecting representation collapse
- Identifying critical training phases
- Analyzing layer-specific learning

**Outputs**:
- Feature quality metrics over time
- Phase transition points
- Feature space visualizations
- Layer-wise evolution trends

### 3. Gradient Flow Analyzer

**Purpose**: Analyze gradient dynamics throughout training

**Key Features**:
- Gradient magnitude tracking
- Direction consistency measurement
- Layer-wise gradient statistics
- Vanishing/exploding gradient detection

**Use Cases**:
- Debugging training instabilities
- Optimizing learning rates
- Understanding layer-specific learning rates
- Detecting optimization issues

**Outputs**:
- Gradient norm evolution
- Stability metrics
- Layer-wise gradient flow
- Training dynamics visualizations

### 4. Memorization Detector

**Purpose**: Distinguish memorization from true generalization

**Key Features**:
- Memorization score computation
- Train-test gap tracking
- Confidence gap analysis
- Transition point detection

**Use Cases**:
- Detecting overfitting early
- Understanding grokking phenomena
- Analyzing simplicity bias
- Model selection and early stopping

**Outputs**:
- Memorization scores over time
- Transition epoch identification
- Memorized vs. generalized example lists
- Gap evolution visualizations

### 5. Phase Transition Attributor

**Purpose**: Identify which training examples drive phase transitions

**Key Features**:
- Gradient-based attribution
- Cross-checkpoint comparison
- Critical example identification
- Label distribution analysis

**Use Cases**:
- Understanding grokking triggers
- Data curation for efficient training
- Identifying influential examples
- Analyzing dataset properties

**Outputs**:
- Attribution scores per example
- Critical example indices
- Label distribution of critical examples

### 6. Bias Attributor

**Purpose**: Identify examples promoting spurious correlations

**Key Features**:
- Spurious example detection
- Bias strength quantification
- Temporal bias evolution
- Biased vs. unbiased comparison

**Use Cases**:
- Detecting simplicity bias
- Data cleaning and debiasing
- Understanding bias learning dynamics
- Fairness analysis

**Outputs**:
- Spurious example indices
- Bias strength scores
- Bias development timeline

## 💡 Usage Patterns

### Pattern 1: During-Training Monitoring

Best for: Real-time insights and adaptive training strategies

```python
# Initialize analyzers at training start
dynamics = LearningDynamicsAnalyzer(num_classes=10)
gradients = GradientFlowAnalyzer(model)

# Track during training
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training step
        loss = train_step(model, batch)
        
        # Analysis
        dynamics.track_batch(...)
        gradients.track_batch_gradients(loss, epoch, batch_idx)
    
    # Epoch-level analysis
    dynamics.compute_epoch_statistics(...)
    
    # Optional: early stopping based on analysis
    if should_stop(dynamics.epoch_stats):
        break
```

### Pattern 2: Post-Hoc Checkpoint Analysis

Best for: Comprehensive analysis after training

```python
# Load saved checkpoints
checkpoints = {
    epoch: load_checkpoint(f'checkpoint_{epoch}.pth')
    for epoch in [0, 10, 20, 30, 40, 50]
}

# Analyze evolution across checkpoints
memorization = MemorizationDetector(num_classes=10)
feature_tracker = FeatureEvolutionTracker(model, layer_names=['layer3', 'layer4'])

for epoch, model in checkpoints.items():
    memorization.compute_memorization_scores(model, train_loader, test_loader, epoch)
    feature_tracker.track_epoch(train_loader, epoch)

# Phase transition analysis
pt_attributor = PhaseTransitionAttributor(
    model_checkpoints={10: checkpoints[10], 50: checkpoints[50]},
    train_data=train_dataset,
    test_data=test_dataset
)

attributions = pt_attributor.compute_transition_attributions(
    pre_transition_epoch=10,
    post_transition_epoch=50
)
```

### Pattern 3: Comparative Phenomenon Analysis

Best for: Understanding different delayed generalization types

```python
# Train multiple models on different phenomena
phenomena = {
    'grokking': train_grokking_model(),
    'simplicity_bias': train_biased_model(),
    'normal': train_normal_model()
}

# Analyze each with same tools
results = {}
for name, (model, data) in phenomena.items():
    analyzer = LearningDynamicsAnalyzer(num_classes=10)
    # ... analyze ...
    results[name] = analyzer.analyze_learning_patterns()

# Compare
plot_comparative_analysis(results)
```

## 📁 Saving and Loading

All analyzers support state persistence:

```python
# Save
analyzer.save('analyzer_state.pkl')

# Load
analyzer = LearningDynamicsAnalyzer.load('analyzer_state.pkl')
```

This is useful for:
- Resuming analysis after interruption
- Sharing analysis results
- Post-hoc visualization generation

## 🎨 Visualization

Each analyzer provides built-in visualization methods:

```python
# Learning Dynamics
dynamics.plot_learning_dynamics(save_dir='./figs')
# Generates: learning_dynamics.png

# Feature Evolution
feature_tracker.plot_metrics_evolution('layer4', save_dir='./figs')
feature_tracker.visualize_evolution('layer4', save_dir='./figs', method='pca')
# Generates: metrics_evolution_layer4.png, feature_evolution_layer4.png

# Gradient Flow
gradients.plot_gradient_flow(save_dir='./figs')
# Generates: gradient_flow.png

# Memorization
memorization.plot_memorization(save_dir='./figs')
# Generates: memorization_analysis.png
```

## 🔬 Integration with Phenomena

### Grokking Analysis

Focus on: Phase transitions, memorization, feature evolution

```python
dynamics = LearningDynamicsAnalyzer(num_classes=output_dim)
memorization = MemorizationDetector(num_classes=output_dim)
feature_tracker = FeatureEvolutionTracker(model, layer_names=['final_layer'])

# Track until grokking occurs
# Identify transition point
# Analyze feature space changes
```

### Simplicity Bias Analysis

Focus on: Bias attribution, learning dynamics

```python
bias_attributor = BiasAttributor(model, biased_data, unbiased_data)
dynamics = LearningDynamicsAnalyzer(num_classes=num_classes)

# Identify spurious examples
# Track when bias learning dominates
# Analyze transition to core features
```

### Robustness Analysis

Focus on: Feature evolution, memorization

```python
feature_tracker = FeatureEvolutionTracker(model, layer_names=['conv_layers'])
memorization = MemorizationDetector(num_classes=num_classes)

# Track representation robustness
# Identify when robust features emerge
```

## 📚 References

The analysis tools are inspired by and build upon:

1. **Dataset Cartography**: Swayamdipta et al. (2020) - Example difficulty and learning dynamics
2. **Grokking Analysis**: Power et al. (2022) - Phase transition understanding
3. **Feature Learning**: Olah et al. (2020) - Feature visualization and analysis
4. **Memorization Studies**: Feldman & Zhang (2020) - Memorization vs. generalization

## 🤝 Contributing

To add new analysis methods:

1. Create a new analyzer class following the existing patterns
2. Implement `save()` and `load()` methods
3. Add visualization methods
4. Update `__init__.py`
5. Add documentation and examples

## 📄 License

This module is part of the Delayed Generalization repository and is licensed under the MIT License.
