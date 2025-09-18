# Optimization Techniques for Delayed Generalization

This directory contains documentation and examples of optimization techniques that influence delayed generalization patterns in machine learning models.

## 📋 Overview

Optimization choices significantly impact when and how models transition from memorization to generalization. Different techniques can:
- **Accelerate** the transition to good generalization
- **Delay** generalization while improving final performance  
- **Control** the timing and smoothness of phase transitions
- **Enable** generalization that wouldn't occur otherwise

## 🔧 Key Techniques

### [Warmup Schedules](./warmup/)
Learning rate warmup can dramatically affect delayed generalization patterns
- **Linear Warmup**: Gradual increase from 0 to target LR
- **Cosine Warmup**: Smooth transitions that help with phase changes
- **Effect**: Often delays initial memorization but improves final generalization

### [Regularization](./regularization/)  
Various regularization techniques influence memorization vs generalization timing
- **Weight Decay**: Critical for grokking, affects simplicity bias
- **Dropout**: Can delay overfitting and promote robust features
- **Data Augmentation**: Breaks spurious correlations early in training

### [Learning Rate Schedules](./scheduling/)
LR scheduling affects the speed and quality of different learning phases
- **Cosine Annealing**: Smooth transitions between learning phases
- **Step Decay**: Can trigger sudden improvements in generalization  
- **Cyclic Learning**: Multiple opportunities for phase transitions

## 🎯 Optimization for Specific Phenomena

### Grokking
- **Essential**: Weight decay (1e-3 to 1e-1)
- **Helpful**: Lower learning rates, smaller batch sizes
- **Timeline**: Can extend training requirements to 10,000+ epochs

### Simplicity Bias
- **Group DRO**: Minimize worst-group loss instead of average
- **Environment-aware**: Training across multiple domains
- **Robust Optimization**: Adversarial training techniques

### Phase Transitions
- **Learning Rate Schedules**: Coordinate with natural transition points
- **Batch Size Scaling**: Can affect transition sharpness
- **Optimizer Choice**: Adam vs SGD show different transition patterns

## 📊 Hyperparameter Recommendations

### For Delayed Generalization Research
```python
# Learning Rate Warmup
warmup_epochs = 100  # Or 5-10% of total training
warmup_schedule = "linear"  # or "cosine"
target_lr = 1e-3

# Weight Decay (crucial for grokking)
weight_decay = 1e-2  # Start here, tune from 1e-3 to 1e-1

# Batch Size
batch_size = 512  # Smaller often better for generalization

# Scheduling
lr_schedule = "cosine"  # Smooth transitions
min_lr = 1e-6  # Don't go to zero
```

## 🔗 References

See individual technique directories for detailed documentation and references.