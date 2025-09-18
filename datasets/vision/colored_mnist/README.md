# Colored MNIST Dataset

A dataset for studying simplicity bias where color correlates with digit class in training but not test data.

## 📋 Overview

The Colored MNIST dataset demonstrates delayed generalization through simplicity bias. Models initially learn to rely on color as a spurious feature before eventually learning the underlying digit shape.

## 🔬 Phenomenon Details

### Simplicity Bias Pattern
1. **Initial Learning (0-50 epochs)**: Model quickly learns color-digit correlation
2. **Plateau Phase (50-200 epochs)**: Performance stagnates when relying on color
3. **Transition Phase (200-400 epochs)**: Model begins learning shape features  
4. **Generalization (400+ epochs)**: Robust performance on color-uncorrelated test data

### Key Characteristics
- **Training Correlation**: High (e.g., 90% of 0s are red, 90% of 1s are blue)
- **Test Correlation**: Low (e.g., 10% correlation, effectively random)
- **Challenge**: Model must learn to ignore salient but spurious color feature

## 🛠️ Dataset Generation

### Basic Usage
```bash
python generate_colored_mnist.py \
    --train_correlation 0.9 \
    --test_correlation 0.1 \
    --output_dir ./colored_mnist_data \
    --color_variance 0.02
```

### Advanced Options
```bash
python generate_colored_mnist.py \
    --train_correlation 0.95 \
    --test_correlation 0.05 \
    --num_colors 10 \
    --background_color \
    --output_dir ./colored_mnist_advanced \
    --color_variance 0.01 \
    --seed 42
```

### Synthetic Version
```bash
python generate_synthetic_colored_digits.py \
    --num_samples 10000 \
    --correlation 0.9 \
    --complexity simple \
    --output_dir ./synthetic_colored_digits
```

## 📊 Dataset Configurations

### Standard Bias Configuration
```python
config = {
    "train_correlation": 0.9,
    "test_correlation": 0.1,
    "color_variance": 0.02,
    "num_colors": 10,
    "background_color": False
}
```

### Strong Bias Configuration  
```python
config = {
    "train_correlation": 0.95,
    "test_correlation": 0.05,
    "color_variance": 0.01,
    "num_colors": 2,  # Binary coloring
    "background_color": True
}
```

### Weak Bias Configuration
```python
config = {
    "train_correlation": 0.7,
    "test_correlation": 0.3,
    "color_variance": 0.05,
    "num_colors": 10,
    "background_color": False
}
```

## 📈 Training Protocols

### Standard Training
```python
# Model architecture - simple CNN
model = SimpleCNN(
    input_channels=3,  # RGB
    num_classes=10,
    hidden_dims=[32, 64],
    dropout=0.1
)

# Training configuration
config = {
    "epochs": 500,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    
    # Monitoring
    "eval_interval": 10,
    "log_group_performance": True
}
```

### Bias Mitigation Training
```python
# Enhanced model with bias mitigation
model = RobustCNN(
    input_channels=3,
    num_classes=10,
    hidden_dims=[32, 64],
    dropout=0.3,  # Higher dropout
    bias_mitigation=True
)

# Group DRO configuration
config = {
    "method": "group_dro",
    "group_dro_step_size": 0.01,
    "epochs": 300,
    "batch_size": 64,  # Smaller batches
    "learning_rate": 1e-3,
    "data_augmentation": True
}
```

## 📊 Evaluation Metrics

### Core Metrics
- **Overall Accuracy**: Standard classification accuracy
- **Worst Group Accuracy**: Performance on hardest color-class combination
- **Color Robustness**: Performance when colors are randomized
- **Shape Reliance**: Gradient-based analysis of shape vs color features

### Group Analysis
```python
# Evaluate by color-digit groups
groups = {
    "color_0_digit_0": subset of samples,
    "color_0_digit_1": subset of samples,
    # ... all combinations
}

for group_name, group_data in groups.items():
    accuracy = evaluate_group(model, group_data)
    print(f"{group_name}: {accuracy:.3f}")
```

### Temporal Analysis
```python
# Track learning progression
metrics_over_time = {
    "epoch": [],
    "train_acc": [],
    "test_acc": [],
    "worst_group_acc": [],
    "color_reliance": [],  # Gradient analysis
    "shape_reliance": []
}
```

## 🎯 Expected Results

### Typical Learning Curve
1. **Epochs 0-50**: Rapid increase in training accuracy (learning color)
2. **Epochs 50-200**: Training accuracy high, test accuracy plateau (overfitting to color)
3. **Epochs 200-400**: Gradual improvement in test accuracy (learning shape)
4. **Epochs 400+**: Convergence to robust performance

### Performance Targets
- **Final Test Accuracy**: >95% (similar to standard MNIST)
- **Worst Group Accuracy**: >90% (robust across all color-digit combinations)  
- **Color Robustness**: >95% (performance when colors randomized)

## 🔬 Research Applications

### Bias Mitigation Methods
- **Group DRO**: Minimize worst-group loss
- **IRM**: Learn invariant representations
- **CORAL**: Domain adaptation techniques
- **Data Augmentation**: Color randomization, mixup

### Architecture Studies
- **CNN vs Transformer**: Different biases toward different features
- **Depth Effects**: How model capacity affects bias learning
- **Attention Analysis**: What features get attention over time

### Optimization Studies  
- **Learning Rate**: Effect on bias vs shape learning timeline
- **Batch Size**: Impact on exploration of robust features
- **Regularization**: Weight decay, dropout effects on generalization

## 🔗 Implementation Example

```python
from datasets.vision.colored_mnist.generate_colored_mnist import ColoredMNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Load base MNIST
mnist_train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)

# Create colored version
colored_train = ColoredMNISTDataset(
    mnist_data=mnist_train,
    correlation=0.9,
    color_variance=0.02,
    num_colors=10
)

# DataLoader
train_loader = DataLoader(
    colored_train,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# Training loop with group monitoring
for epoch in range(500):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_acc, group_accs = evaluate_with_groups(model, test_loader)
    
    print(f"Epoch {epoch}: Train {train_acc:.3f}, Test {test_acc:.3f}, "
          f"Worst Group {min(group_accs.values()):.3f}")
```

## 📚 References

1. Arjovsky et al. (2019). "Invariant Risk Minimization"
2. Sagawa et al. (2020). "Distributionally Robust Neural Networks"  
3. Kim et al. (2019). "Learning Not to Learn"
4. Shah et al. (2020). "The Pitfalls of Simplicity Bias"