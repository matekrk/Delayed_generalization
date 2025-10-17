# Learning Rate Warmup and Delayed Generalization

## 📋 Overview

Learning rate warmup is a training technique where the learning rate starts from a small value and gradually increases to a target value over a specified number of training steps. This technique significantly influences delayed generalization patterns across various ML scenarios.

## 🔬 Impact on Delayed Generalization

### Grokking
- **Effect**: Can delay the initial memorization phase but often leads to more robust generalization
- **Mechanism**: Slower initial learning allows better feature development
- **Timeline**: May extend total training time but improves generalization quality

### Simplicity Bias
- **Effect**: Can help models avoid getting stuck in spurious correlations
- **Mechanism**: Slower learning gives time for multiple features to develop
- **Best Practice**: Combine with other debiasing techniques

### Phase Transitions
- **Effect**: Can make transitions smoother and more predictable
- **Mechanism**: Gradual learning rate increase mirrors natural learning phases
- **Observation**: Often reduces the sharpness of phase transitions

## 🛠️ Warmup Schedules

### 1. Linear Warmup
```python
def linear_warmup(step, warmup_steps, target_lr):
    """Linear increase from 0 to target_lr over warmup_steps"""
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr
```

**Use Cases**: 
- Standard choice for most delayed generalization experiments
- Works well with transformer architectures
- Good baseline for grokking experiments

### 2. Cosine Warmup  
```python
import math

def cosine_warmup(step, warmup_steps, target_lr):
    """Cosine increase from 0 to target_lr over warmup_steps"""
    if step < warmup_steps:
        return target_lr * 0.5 * (1 + math.cos(math.pi * (1 - step / warmup_steps)))
    return target_lr
```

**Use Cases**:
- Smoother transitions, good for sensitive models
- Often used in large language model training
- Can help with training stability

### 3. Exponential Warmup
```python
def exponential_warmup(step, warmup_steps, target_lr, base=2):
    """Exponential increase to target_lr over warmup_steps"""
    if step < warmup_steps:
        return target_lr * (base ** (step / warmup_steps - 1))
    return target_lr
```

**Use Cases**:
- When you want slow initial learning then rapid acceleration
- Can be useful for particularly difficult delayed generalization scenarios

## 📊 Experimental Configurations

### For Grokking (Modular Arithmetic)
```python
# Standard setup with warmup
config = {
    "target_lr": 1e-3,
    "warmup_steps": 1000,  # 10% of total training typically
    "warmup_type": "linear",
    "total_steps": 10000,
    "weight_decay": 1e-2,  # Still crucial!
    "batch_size": 512
}

# Post-warmup schedule
"post_warmup_schedule": "cosine"  # Often combined with cosine annealing
"min_lr": 1e-6
```

### For Simplicity Bias (Waterbirds)
```python
config = {
    "target_lr": 1e-3,
    "warmup_epochs": 10,  # Shorter warmup often sufficient
    "warmup_type": "linear", 
    "total_epochs": 300,
    "scheduler": "cosine",
    "optimizer": "SGD",  # Often works better than Adam for robustness
    "momentum": 0.9
}
```

### For Large Models (Phase Transitions)
```python
config = {
    "target_lr": 1e-4,  # Lower LR for large models
    "warmup_steps": 2000,  # Longer warmup for stability
    "warmup_type": "cosine",
    "total_steps": 100000,
    "weight_decay": 0.1,
    "batch_size": 1024  # Can be larger for big models
}
```

## 📈 Monitoring and Analysis

### Key Metrics During Warmup
1. **Learning Rate Trajectory**: Verify warmup schedule is working
2. **Loss Smoothness**: Warmup should reduce training instability
3. **Gradient Norms**: Track gradient explosion/vanishing
4. **Feature Development**: Monitor when different features emerge

### Visualization Tools
```python
import matplotlib.pyplot as plt

def plot_warmup_schedule(warmup_fn, warmup_steps, target_lr):
    steps = range(warmup_steps * 2)  # Show beyond warmup
    lrs = [warmup_fn(step, warmup_steps, target_lr) for step in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.axvline(warmup_steps, color='red', linestyle='--', label='Warmup End')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Schedule')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 🎯 Best Practices

### Warmup Duration
- **Rule of Thumb**: 5-10% of total training time
- **Grokking**: Can use longer warmup (up to 20%)
- **Large Models**: Often need longer warmup for stability
- **Small Models**: Shorter warmup usually sufficient

### Combining with Other Techniques
- **Weight Decay**: Always include, especially for grokking
- **Gradient Clipping**: Can complement warmup for stability
- **Batch Size Scaling**: Consider warmup with batch size changes

### Common Mistakes
1. **Too Short Warmup**: Doesn't provide benefits
2. **Too Long Warmup**: Wastes training time
3. **Wrong Post-Warmup Schedule**: Sudden LR changes can hurt
4. **Ignoring Other Hyperparameters**: Warmup doesn't fix bad configs

## 🧪 Experimental Examples

### Grokking with Different Warmup Schedules
```python
# Experiment: Compare warmup effects on grokking timeline
warmup_configs = [
    {"warmup_steps": 0, "name": "No Warmup"},
    {"warmup_steps": 500, "name": "Short Warmup"},  
    {"warmup_steps": 1000, "name": "Medium Warmup"},
    {"warmup_steps": 2000, "name": "Long Warmup"}
]

# Expected results:
# - No warmup: Fastest initial memorization, may hurt generalization
# - Short warmup: Good balance
# - Medium warmup: Often optimal for grokking
# - Long warmup: May delay everything but improve final performance
```

### Waterbirds with Warmup + Group DRO
```python
# Combining warmup with robust optimization
config = {
    "warmup_epochs": 10,
    "target_lr": 1e-3,
    "total_epochs": 300,
    "group_dro": True,  # Focus on worst-group performance
    "warmup_type": "linear"
}

# Expected: Slower initial learning but better final robustness
```

## 🔗 Implementation Resources

### PyTorch Implementation
```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr, warmup_type='linear'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.warmup_type = warmup_type
        self.step_count = 0
        
    def step(self):
        if self.step_count < self.warmup_steps:
            if self.warmup_type == 'linear':
                lr = self.target_lr * self.step_count / self.warmup_steps
            elif self.warmup_type == 'cosine':
                lr = self.target_lr * 0.5 * (1 + math.cos(
                    math.pi * (1 - self.step_count / self.warmup_steps)))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        self.step_count += 1
```

## 📚 References

1. Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
2. He et al. (2019). "Bag of Tricks for Image Classification with Convolutional Neural Networks"  
3. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
4. Smith et al. (2017). "Cyclical Learning Rates for Training Neural Networks"
5. Huang et al. (2020). "Improving Transformer Optimization Through Better Initialization"