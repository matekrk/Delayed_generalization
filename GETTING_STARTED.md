# Getting Started with Delayed Generalization Research

This section provides quick start guides for running the implemented delayed generalization phenomena.

## 🚀 Quick Start Examples

### Grokking (Modular Arithmetic)

Generate data and train a transformer model:

```bash
# Generate modular arithmetic dataset
python data/algorithmic/modular_arithmetic/generate_data.py \
    --prime 97 --operation addition --output_dir ./grok_data

# Train transformer (may take hours for grokking to occur)
python phenomena/grokking/training/train_modular.py \
    --data_dir ./grok_data --epochs 10000 --weight_decay 1e-2
```

### Waterbirds (Simplicity Bias)

Study background bias in bird classification:

```bash
# Generate synthetic waterbirds dataset
python phenomena/simplicity_bias/waterbirds/data/generate_synthetic_waterbirds.py \
    --train_correlation 0.9 --test_correlation 0.1 --output_dir ./waterbirds_data

# Train with standard ERM
python phenomena/simplicity_bias/waterbirds/training/train_waterbirds.py \
    --data_dir ./waterbirds_data --method erm --epochs 100

# Train with Group DRO (bias mitigation)
python phenomena/simplicity_bias/waterbirds/training/train_waterbirds.py \
    --data_dir ./waterbirds_data --method group_dro --epochs 100
```

### Colored MNIST (Simplicity Bias)

Study color vs shape learning:

```bash
# Generate colored MNIST dataset
python phenomena/simplicity_bias/colored_mnist/data/generate_colored_mnist.py \
    --train_correlation 0.9 --test_correlation 0.1 --output_dir ./colored_mnist_data

# Train CNN model
python phenomena/simplicity_bias/colored_mnist/training/train_colored_mnist.py \
    --data_dir ./colored_mnist_data --epochs 100
```

### CelebA (Gender Bias)

Study background bias in gender classification:

```bash
# Generate synthetic CelebA dataset
python phenomena/simplicity_bias/celeba/data/generate_synthetic_celeba.py \
    --train_bias 0.8 --test_bias 0.2 --output_dir ./celeba_data

# Train CNN model
python phenomena/simplicity_bias/celeba/training/train_celeba.py \
    --data_dir ./celeba_data --epochs 100
```

### CIFAR-10-C (Robustness)

Study clean vs corrupted image classification:

```bash
# Generate synthetic CIFAR-10-C dataset
python phenomena/robustness/cifar10c/data/generate_synthetic_cifar10c.py \
    --train_corruptions gaussian_noise motion_blur \
    --test_corruptions fog brightness --output_dir ./cifar10c_data

# Train CNN model
python phenomena/robustness/cifar10c/training/train_cifar10c.py \
    --data_dir ./cifar10c_data --epochs 100
```

## 📊 Expected Phenomena

### Grokking Timeline
- **Epochs 0-1000**: Memorization phase (high train accuracy, low test accuracy)
- **Epochs 1000-5000**: Transition period (gradual test accuracy improvement)
- **Epochs 5000+**: Generalization phase (high test accuracy achieved)

### Simplicity Bias Timeline
- **Early Training**: Model learns spurious correlations (background, color)
- **Mid Training**: Spurious accuracy high, robust accuracy low
- **Late Training**: Gradual shift to learning true features (bird type, digit shape)

### Robustness Development
- **Clean Images**: High accuracy achieved quickly
- **Corrupted Images**: Initially poor, gradual improvement
- **Robustness Gap**: Decreases over training as model learns invariant features

## 🔍 Key Metrics to Monitor

- **Grokking**: Train vs test accuracy gap sudden closure
- **Bias Research**: Worst-group accuracy, spurious correlation strength
- **Robustness**: Clean vs corrupted accuracy gap

## 💡 Research Applications

This framework enables studying:
- **Delayed Generalization**: When and why models suddenly generalize
- **Simplicity Bias**: How models prefer simple features over complex ones
- **Robustness vs Accuracy**: Tradeoffs between clean and robust performance
- **Intervention Effects**: How different training methods affect bias learning