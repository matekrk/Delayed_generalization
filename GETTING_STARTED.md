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
python data/vision/waterbirds/generate_waterbirds.py \
    --train_correlation 0.9 --test_correlation 0.5 --output_dir ./waterbirds_data

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
python data/vision/colored_mnist/generate_colored_mnist.py \
    --train_correlation 0.9 --test_correlation 0.1 --output_dir ./colored_mnist_data

# Train CNN model
python phenomena/simplicity_bias/colored_mnist/training/train_colored_mnist.py \
    --data_dir ./colored_mnist_data --epochs 100
```

### CelebA (Attribute Bias)

Study attribute bias in face classification with real or synthetic data:

#### Real CelebA with Attribute Bias (Recommended)

```bash
# Generate real biased CelebA dataset with Male vs Blond_Hair bias
python data/vision/celeba/generate_bias_celeba.py \
    --attr1 Male --attr2 Blond_Hair \
    --train_bias 0.8 --test_bias 0.2 \
    --train_size 10000 --test_size 2000 \
    --output_dir ./bias_celeba_data

# Train CNN model with bias analysis
python phenomena/simplicity_bias/celeba/training/train_bias_celeba.py \
    --data_dir ./bias_celeba_data/real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20 \
    --epochs 100 --use_wandb

# Alternative attribute combinations
# Young vs Heavy_Makeup bias
python data/vision/celeba/generate_bias_celeba.py \
    --attr1 Young --attr2 Heavy_Makeup \
    --train_bias 0.9 --test_bias 0.1 --output_dir ./bias_celeba_young_makeup

# Attractive vs Eyeglasses bias  
python data/vision/celeba/generate_bias_celeba.py \
    --attr1 Attractive --attr2 Eyeglasses \
    --train_bias 0.85 --test_bias 0.15 --output_dir ./bias_celeba_attractive_glasses
```

#### Synthetic CelebA (Legacy)

```bash
# Generate synthetic CelebA dataset
python data/vision/generate_synthetic_celeba.py \
    --train_bias 0.8 --test_bias 0.5 --output_dir ./celeba_data

# Train CNN model
python phenomena/simplicity_bias/celeba/training/train_celeba.py \
    --data_dir ./celeba_data --epochs 100
```

### CIFAR-10-C (Robustness)

Study clean vs corrupted image classification:

```bash
# Generate synthetic CIFAR-10-C dataset
python data/vision/cifar10c/generate_synthetic_cifar10c.py \
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
- **Early Training**: Model learns spurious correlations (background, color, spurious attributes)
- **Mid Training**: Spurious accuracy high, robust accuracy low
- **Late Training**: Gradual shift to learning true features (bird type, digit shape, core attributes)

### CelebA Attribute Bias Timeline
- **Phase 1 (Epochs 0-20)**: Rapid learning of spurious correlations (e.g., Male → Non-Blonde)
- **Phase 2 (Epochs 20-60)**: High bias-conforming accuracy, poor bias-conflicting accuracy
- **Phase 3 (Epochs 60+)**: Potential delayed generalization to true attribute relationships
- **Key Transition**: Watch for bias gap reduction indicating robust feature learning

### Robustness Development
- **Clean Images**: High accuracy achieved quickly
- **Corrupted Images**: Initially poor, gradual improvement
- **Robustness Gap**: Decreases over training as model learns invariant features

## 🔍 Key Metrics to Monitor

### Grokking (Modular Arithmetic)
- **Train vs test accuracy gap**: Sudden closure indicating generalization
- **Loss transition**: Sharp drop in test loss after long plateau

### Simplicity Bias (Colored MNIST, CelebA)
- **Bias gap**: Difference between bias-conforming and bias-conflicting accuracy
- **Spurious correlation strength**: How much model relies on biased features
- **Worst-group accuracy**: Performance on hardest subgroup (e.g., blonde males)
- **Attribute accuracy**: Per-attribute classification performance

### CelebA-Specific Metrics
- **Bias conforming accuracy**: Performance when spurious feature matches target
- **Bias conflicting accuracy**: Performance when spurious feature conflicts with target
- **Attribute correlation**: Correlation between predicted and spurious attributes
- **Fairness metrics**: Equal opportunity across demographic groups

### Robustness (CIFAR-10-C)
- **Clean vs corrupted accuracy gap**: Robustness measure
- **Corruption-specific performance**: Per-corruption type accuracy

## 💡 Research Applications

This framework enables studying:
- **Delayed Generalization**: When and why models suddenly generalize
- **Simplicity Bias**: How models prefer simple features over complex ones
- **Robustness vs Accuracy**: Tradeoffs between clean and robust performance
- **Intervention Effects**: How different training methods affect bias learning