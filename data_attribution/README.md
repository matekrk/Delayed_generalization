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

### Phase Transition Attribution
```python
from data_attribution.analysis import PhaseTransitionAttributor

# Analyze which examples contribute to phase transitions
pt_attributor = PhaseTransitionAttributor(
    model_checkpoints=checkpoints,  # Models before/after transition
    train_data=train_data,
    test_data=test_data
)

# Find examples that most influence the transition
transition_influences = pt_attributor.compute_transition_attributions()

# Identify critical examples for generalization
critical_examples = pt_attributor.find_critical_examples(
    transition_influences,
    top_k=100
)
```

### Bias Attribution Analysis
```python
from data_attribution.analysis import BiasAttributor

# Analyze which examples promote spurious correlations
bias_attributor = BiasAttributor(
    model=model,
    biased_data=biased_train_data,
    unbiased_data=unbiased_test_data
)

# Find examples that promote spurious features
spurious_influences = bias_attributor.find_spurious_examples(
    attribution_method='trak',
    top_k=200
)

# Analyze bias development over training
bias_timeline = bias_attributor.analyze_bias_development(
    model_checkpoints=checkpoints
)
```

## 🔗 Integration Examples

See the individual method directories for detailed documentation and examples:

- **[TRAK Implementation](./trak/README.md)**: Efficient attribution for large models
- **[Influence Functions](./influence_functions/README.md)**: Classical attribution methods
- **[Analysis Utilities](./utils/README.md)**: Tools for analyzing attributions

## 📚 References

1. Park et al. (2023). "TRAK: Attributing Model Behavior at Scale"
2. Koh & Liang (2017). "Understanding Black-box Predictions via Influence Functions"
3. Feldman & Zhang (2020). "What Neural Networks Memorize and Why"
4. Swayamdipta et al. (2020). "Dataset Cartography: Mapping and Diagnosing Datasets"