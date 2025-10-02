# Data Attribution for Delayed Generalization Research

## 📋 Overview

Data attribution methods help identify which training examples most influenced a model's predictions, which is particularly valuable for understanding delayed generalization phenomena. This directory provides implementations of TRAK (Tracing with Randomly-projected After Kernels) and other attribution methods.

## 🔬 Why Data Attribution for Delayed Generalization?

### Understanding Phase Transitions
- **Critical Examples**: Identify which training examples trigger phase transitions
- **Memorization vs Generalization**: Distinguish examples that promote memorization vs generalization
- **Feature Development**: Track how influence patterns change as models develop new capabilities

### Debugging Delayed Generalization
- **Grokking Analysis**: Find examples that enable sudden generalization in algorithmic tasks
- **Bias Detection**: Identify training examples that promote spurious correlations
- **Robustness**: Understand which examples contribute to robust vs brittle features

## 🛠️ Available Methods

### [TRAK](./trak/)
- **Method**: Tracing with Randomly-projected After Kernels
- **Strengths**: Efficient, works well with large models
- **Use Cases**: Understanding which examples influence specific predictions
- **Timeline**: Can be computed during or after training

### [Influence Functions](./influence_functions/)
- **Method**: First-order approximation to leave-one-out retraining
- **Strengths**: Theoretically grounded
- **Use Cases**: Finding training examples similar to test examples
- **Limitations**: Computationally expensive for large models

### [Utilities](./utils/)
- **Gradient Similarity**: Simple cosine similarity between gradients
- **k-NN in Feature Space**: Nearest neighbors in learned representations
- **Training Dynamics**: Track example-level loss and accuracy over time

## 🚀 Quick Start

### Basic TRAK Usage
```python
from data_attribution.trak import TRAKAttributor

# Initialize TRAK
attributor = TRAKAttributor(
    model=model,
    task='classification',
    proj_dim=512,  # Projection dimension
    device=device
)

# Compute features for training data
train_features = attributor.compute_train_features(
    dataloader=train_loader,
    num_samples=len(train_dataset)
)

# Compute attributions for test examples
test_features = attributor.compute_test_features(
    dataloader=test_loader
)

attributions = attributor.compute_attributions(
    train_features=train_features,
    test_features=test_features
)

# attributions[i, j] = influence of train example j on test example i
```

### Integration with Training
```python
from data_attribution.utils import TrainingDynamicsTracker

# Track training dynamics during training
tracker = TrainingDynamicsTracker(
    dataset=train_dataset,
    save_interval=100  # Save every 100 epochs
)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # ... training step ...
        
        # Track example-level metrics
        tracker.update(
            epoch=epoch,
            batch_idx=batch_idx,
            data=data,
            targets=targets,
            predictions=predictions,
            losses=losses
        )

# Analyze training dynamics
dynamics_analysis = tracker.analyze()
```

## 📊 Analysis Tools

The `data_attribution.analysis` module provides comprehensive analysis tools for understanding delayed generalization:

### 1. Learning Dynamics Analyzer
Track example-level learning patterns throughout training:

```python
from data_attribution.analysis import LearningDynamicsAnalyzer

# Initialize analyzer
dynamics_analyzer = LearningDynamicsAnalyzer(
    num_classes=10,
    track_examples=1000,  # Number of examples to track in detail
    device=device
)

# During training: track batch dynamics
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ... training code ...
        
        dynamics_analyzer.track_batch(
            model=model,
            inputs=inputs,
            targets=targets,
            batch_indices=batch_indices,
            epoch=epoch
        )
    
    # Compute epoch statistics
    dynamics_analyzer.compute_epoch_statistics(train_loader, model, epoch)

# Analyze patterns
analysis = dynamics_analyzer.analyze_learning_patterns()
print(f"Easy examples: {len(analysis['easy_examples'])}")
print(f"Examples with forgetting: {analysis['forgetting_statistics']['examples_with_forgetting']}")

# Visualize
dynamics_analyzer.plot_learning_dynamics(save_dir='./analysis')
```

**Features:**
- Tracks prediction confidence, loss, and correctness over time
- Identifies easy/medium/hard examples
- Detects forgetting events
- Computes difficulty scores
- Analyzes learning speed distribution

### 2. Feature Evolution Tracker
Monitor how learned representations evolve:

```python
from data_attribution.analysis import FeatureEvolutionTracker

# Initialize tracker
feature_tracker = FeatureEvolutionTracker(
    model=model,
    layer_names=['layer3', 'layer4'],  # Layers to track
    device=device,
    max_samples_store=500
)

# Track features at specific epochs
for epoch in checkpoints:
    model.load_state_dict(checkpoints[epoch])
    feature_tracker.track_epoch(train_loader, epoch)

# Analyze evolution
analysis = feature_tracker.analyze_evolution()
for layer_name in feature_tracker.layer_names:
    stats = analysis[layer_name]['statistics'][-1]
    print(f"{layer_name}: Silhouette={stats['silhouette_score']:.4f}")

# Detect phase transitions
transitions = feature_tracker.detect_phase_transitions('layer4', metric='silhouette_score')
print(f"Transitions at epochs: {transitions}")

# Visualize
feature_tracker.visualize_evolution('layer4', save_dir='./analysis', method='pca')
feature_tracker.plot_metrics_evolution('layer4', save_dir='./analysis')
```

**Features:**
- Tracks representation quality metrics (silhouette score, separability)
- Computes effective rank and feature variance
- Detects phase transitions in feature space
- Visualizes feature space evolution with PCA/t-SNE

### 3. Gradient Flow Analyzer
Analyze gradient dynamics:

```python
from data_attribution.analysis import GradientFlowAnalyzer

# Initialize analyzer
gradient_analyzer = GradientFlowAnalyzer(
    model=model,
    track_layers=None  # Track all layers, or specify list
)

# During training: track gradients
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Track before optimizer.step()
        gradient_analyzer.track_batch_gradients(loss, epoch, batch_idx)
        
        optimizer.step()
    
    gradient_analyzer.compute_epoch_statistics(epoch)

# Analyze
analysis = gradient_analyzer.analyze_gradient_flow()
print(f"Final gradient stability: {analysis['overall_stability'][-1]:.4f}")

# Visualize
gradient_analyzer.plot_gradient_flow(save_dir='./analysis')
```

**Features:**
- Tracks gradient magnitudes and norms
- Measures gradient direction consistency (stability)
- Layer-wise gradient analysis
- Detects vanishing/exploding gradients

### 4. Memorization Detector
Distinguish memorization from generalization:

```python
from data_attribution.analysis import MemorizationDetector

# Initialize detector
mem_detector = MemorizationDetector(
    num_classes=10,
    device=device
)

# Compute memorization scores at checkpoints
for epoch in checkpoints:
    model.load_state_dict(checkpoints[epoch])
    metrics = mem_detector.compute_memorization_scores(
        model, train_loader, test_loader, epoch
    )
    print(f"Epoch {epoch}: Memorization={metrics['memorization_score']:.4f}")

# Identify memorized examples
memorized_idx, generalized_idx = mem_detector.identify_memorized_examples(
    model, train_loader, threshold=0.8
)

# Analyze evolution
analysis = mem_detector.analyze_memorization_evolution()
print(f"Transition epoch: {analysis['transition_epoch']}")
print(f"Memorization reduction: {analysis['memorization_reduction']:.4f}")

# Visualize
mem_detector.plot_memorization(save_dir='./analysis')
```

**Features:**
- Computes memorization score (train-test accuracy gap)
- Tracks confidence gap evolution
- Identifies memorized vs. generalized examples
- Detects transition from memorization to generalization

### 5. Phase Transition Attribution
Identify critical examples for phase transitions:

```python
from data_attribution.analysis import PhaseTransitionAttributor

# Analyze which examples contribute to phase transitions
pt_attributor = PhaseTransitionAttributor(
    model_checkpoints={
        early_epoch: model_early,
        late_epoch: model_late
    },
    train_data=train_data,
    test_data=test_data
)

# Compute attributions
attributions = pt_attributor.compute_transition_attributions(
    pre_transition_epoch=early_epoch,
    post_transition_epoch=late_epoch,
    method='gradient_similarity'
)

# Find critical examples
critical_examples = pt_attributor.find_critical_examples(
    attributions,
    top_k=100
)

# Analyze properties
analysis = pt_attributor.analyze_critical_examples(critical_examples)
print(f"Label distribution: {analysis['label_distribution']}")
```

**Features:**
- Gradient-based attribution across transitions
- Identifies examples with consistent influence
- Analyzes properties of critical examples

### 6. Bias Attribution Analysis
Identify examples promoting spurious correlations:

```python
from data_attribution.analysis import BiasAttributor

# Analyze which examples promote spurious correlations
bias_attributor = BiasAttributor(
    model=model,
    biased_data=biased_train_data,
    unbiased_data=unbiased_test_data
)

# Find spurious examples
spurious_examples = bias_attributor.find_spurious_examples(
    attribution_method='gradient_based',
    top_k=200
)

# Analyze bias development over training
bias_timeline = bias_attributor.analyze_bias_development(
    model_checkpoints=checkpoints
)

# Compute current bias strength
bias_strength = bias_attributor.compute_bias_strength()
print(f"Bias strength: {bias_strength:.4f}")
```

**Features:**
- Identifies examples promoting spurious features
- Tracks bias development over training
- Compares biased vs. unbiased performance
- Quantifies bias strength

## 🔗 Integration Examples

See the individual method directories and example scripts for detailed documentation and usage:

- **[TRAK Implementation](./trak/README.md)**: Efficient attribution for large models
- **[Influence Functions](./influence_functions/README.md)**: Classical attribution methods
- **[Analysis Module](./analysis/)**: Comprehensive analysis tools for delayed generalization

### Example Scripts

We provide complete, runnable examples demonstrating various analysis workflows:

#### 1. During-Training Analysis (`examples/analysis_during_training.py`)
Integrate analysis tools directly into your training loop for real-time insights:

```bash
python examples/analysis_during_training.py
```

This example demonstrates:
- Real-time learning dynamics tracking
- Feature evolution monitoring
- Gradient flow analysis
- Memorization detection during training
- Automatic visualization generation

#### 2. Post-Hoc Analysis (`examples/analysis_post_hoc.py`)
Analyze saved model checkpoints after training:

```bash
python examples/analysis_post_hoc.py
```

This example demonstrates:
- Loading and analyzing model checkpoints
- Phase transition attribution
- Memorization evolution analysis
- Feature space evolution across epochs
- Comparative checkpoint analysis

#### 3. Multi-Phenomenon Comparison (`examples/analysis_multi_phenomenon.py`)
Compare delayed generalization patterns across different phenomena:

```bash
python examples/analysis_multi_phenomenon.py
```

This example demonstrates:
- Comparing grokking, simplicity bias, and normal learning
- Side-by-side analysis of different phenomena
- Unified visualization of comparative results
- Phenomenon-specific insights

### Quick Integration Template

Here's a minimal template to add analysis to your training code:

```python
from data_attribution.analysis import (
    LearningDynamicsAnalyzer,
    MemorizationDetector
)

# Initialize analyzers
dynamics = LearningDynamicsAnalyzer(num_classes=10, device=device)
memorization = MemorizationDetector(num_classes=10, device=device)

# In your training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ... your training code ...
        
        # Track dynamics
        dynamics.track_batch(model, inputs, targets, batch_indices, epoch)
    
    # Epoch-level tracking
    dynamics.compute_epoch_statistics(train_loader, model, epoch)
    memorization.compute_memorization_scores(model, train_loader, test_loader, epoch)

# Generate analysis and visualizations
analysis = dynamics.analyze_learning_patterns()
dynamics.plot_learning_dynamics(save_dir='./results')
memorization.plot_memorization(save_dir='./results')
```

## 📚 References

1. Park et al. (2023). "TRAK: Attributing Model Behavior at Scale"
2. Koh & Liang (2017). "Understanding Black-box Predictions via Influence Functions"
3. Feldman & Zhang (2020). "What Neural Networks Memorize and Why"
4. Swayamdipta et al. (2020). "Dataset Cartography: Mapping and Diagnosing Datasets"