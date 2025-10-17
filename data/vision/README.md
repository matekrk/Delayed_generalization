# Vision Datasets for Delayed Generalization Research

This directory contains vision datasets commonly used to study delayed generalization phenomena, particularly simplicity bias and robustness patterns.

## 📊 Available Datasets

### [CelebA](./celeba/)
- **Phenomenon**: Simplicity bias in attribute prediction
- **Bias Type**: Background vs demographic features
- **Timeline**: 50-200 epochs for bias emergence
- **Use Case**: Studying spurious correlations in face recognition

### [Colored MNIST](./colored_mnist/)  
- **Phenomenon**: Color vs shape learning
- **Bias Type**: Color correlation with digit class
- **Timeline**: 50-500 epochs for generalization
- **Use Case**: Classic simplicity bias demonstration

### [Waterbirds](./waterbirds/)
- **Phenomenon**: Background vs bird type classification
- **Bias Type**: Background correlation with bird species
- **Timeline**: 100-300 epochs
- **Use Case**: Standard benchmark for group robustness

### [CIFAR-10-C](./cifar10c/)
- **Phenomenon**: Robustness vs content learning
- **Bias Type**: Clean vs corrupted image features
- **Timeline**: Variable based on corruption type
- **Use Case**: Studying delayed generalization in robustness

## 🔗 Quick Start

Each dataset directory contains:
- Generation scripts for synthetic versions
- Documentation of experimental protocols
- Example training configurations
- Analysis tools and utilities

## 📈 Common Experimental Patterns

### Simplicity Bias Studies
1. **Training Phase**: Model learns spurious correlations quickly
2. **Plateau Phase**: Performance stagnates on correlated features  
3. **Generalization Phase**: Model eventually learns invariant features
4. **Evaluation**: Test on data where correlations are broken

### Robustness Studies  
1. **Clean Training**: Initial learning on uncorrupted data
2. **Corruption Introduction**: Gradual introduction of corruptions
3. **Adaptation Phase**: Model learns to handle corruptions
4. **Evaluation**: Test on various corruption types and severities

## 🛠️ Usage Guidelines

### Data Generation
```bash
# Generate Colored MNIST with strong bias
cd colored_mnist/
python generate_colored_mnist.py --train_correlation 0.9 --test_correlation 0.1

# Generate synthetic CelebA-like data
cd celeba/
python generate_synthetic_celeba.py --train_bias 0.8 --test_bias 0.2

# Generate CIFAR-10-C corruptions
cd cifar10c/
python generate_synthetic_cifar10c.py --corruption_types noise blur
```

### Integration with Training
```python
# Example integration
from data.vision.colored_mnist.generate_colored_mnist import ColoredMNISTDataset
from data.vision.celeba.generate_synthetic_celeba import SyntheticCelebADataset

# Create biased datasets
train_dataset = ColoredMNISTDataset(
    mnist_data=mnist_train,
    correlation=0.9,
    color_variance=0.02
)

test_dataset = ColoredMNISTDataset(
    mnist_data=mnist_test,
    correlation=0.1,  # Broken correlation
    color_variance=0.02
)
```

## 📊 Evaluation Protocols

### Group-wise Analysis
- Track performance on different subgroups
- Monitor worst-group performance over time
- Identify when bias learning vs true learning occurs

### Timing Analysis
- Log when spurious vs invariant features are learned
- Track phase transitions in learning dynamics
- Measure time to robust generalization

## 🔗 References

1. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
2. Arjovsky et al. (2019). "Invariant Risk Minimization"
3. Beery et al. (2018). "Recognition in Terra Incognita"
4. Hendrycks & Dietterich (2019). "Benchmarking Neural Network Robustness"