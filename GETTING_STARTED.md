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

# Multi-attribute bias (NEW): Multiple biased pairs + additional attributes
python data/vision/celeba/generate_bias_celeba.py \
    --multi_attribute \
    --bias_pairs "Male,Blond_Hair" "Young,Heavy_Makeup" \
    --additional_attrs "Attractive" "Eyeglasses" "Smiling" \
    --train_bias 0.8 --test_bias 0.2 --output_dir ./multi_attr_celeba
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

### Sentiment Bias NLP (NEW)

Study sentiment classification with topic bias:

```bash
# Generate sentiment bias dataset (technology bias)
python data/nlp/sentiment/generate_sentiment_bias.py \
    --bias_topic technology --train_bias 0.9 --test_bias 0.1 \
    --train_size 5000 --test_size 1000 --output_dir ./sentiment_bias_data

# Train sentiment classifier
python phenomena/nlp/sentiment_bias/training/train_sentiment_bias.py \
    --data_dir ./sentiment_bias_data --epochs 100 --use_wandb

# Alternative bias topics
python data/nlp/sentiment/generate_sentiment_bias.py \
    --bias_topic politics --neutral_topics sports entertainment science \
    --output_dir ./sentiment_politics_bias
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

### Sentiment Bias NLP Timeline (NEW)
- **Phase 1 (Epochs 0-30)**: Model learns topic-sentiment spurious correlations (e.g., technology → positive)
- **Phase 2 (Epochs 30-70)**: High bias-conforming accuracy, poor bias-conflicting accuracy  
- **Phase 3 (Epochs 70+)**: Potential delayed generalization to true sentiment patterns
- **Key Transition**: Watch for bias gap reduction indicating robust sentiment learning

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

### Sentiment Bias (NLP)
- **Bias gap**: Difference between bias-conforming and bias-conflicting accuracy
- **Topic-sentiment correlation**: How much model relies on topic for sentiment prediction
- **Cross-topic generalization**: Performance on sentiment classification across different topics

### Robustness (CIFAR-10-C)
- **Clean vs corrupted accuracy gap**: Robustness measure
- **Corruption-specific performance**: Per-corruption type accuracy

## 🌊 Phase Transition Detection

### What Are Phase Transitions?
Phase transitions in delayed generalization are sudden, dramatic changes in model behavior during training. They represent shifts from one learning regime to another:

- **Grokking**: Sudden transition from memorization to generalization
- **Bias Breaking**: Sudden shift from spurious to robust feature learning
- **Robustness Emergence**: Sudden improvement in performance on corrupted/adversarial examples

### Detecting Phase Transitions

#### 1. Grokking Phase Transitions
Monitor these signals for sudden generalization:

```python
# Key metrics to track
grokking_signals = {
    'test_accuracy_derivative': 'Sharp increase in test accuracy slope',
    'train_test_gap': 'Sudden closure of train-test accuracy gap', 
    'loss_transition': 'Sharp drop in test loss after plateau',
    'gradient_norm_change': 'Sudden change in gradient magnitude patterns'
}

# Detection criteria
def detect_grokking(test_accuracies, window=50, threshold=10.0):
    """Detect grokking transition based on test accuracy jump."""
    if len(test_accuracies) < window * 2:
        return False, -1
    
    # Calculate sliding window derivatives
    derivatives = []
    for i in range(window, len(test_accuracies)):
        recent_slope = (test_accuracies[i] - test_accuracies[i-window]) / window
        derivatives.append(recent_slope)
    
    # Find sudden jumps
    for i, slope in enumerate(derivatives):
        if slope > threshold:  # Significant accuracy jump
            return True, i + window
    
    return False, -1
```

#### 2. Simplicity Bias Phase Transitions
Monitor bias gap reduction:

```python
# Bias gap monitoring
def detect_bias_breaking(bias_conforming_acc, bias_conflicting_acc, threshold=5.0):
    """Detect when model starts learning robust features."""
    bias_gap = [conf - conf for conf, conf in zip(bias_conforming_acc, bias_conflicting_acc)]
    
    # Look for sustained gap reduction
    if len(bias_gap) < 20:
        return False, -1
        
    recent_trend = np.polyfit(range(len(bias_gap[-20:])), bias_gap[-20:], 1)[0]
    
    if recent_trend < -threshold:  # Gap closing significantly
        return True, len(bias_gap) - 20
    
    return False, -1
```

#### 3. Universal Phase Transition Indicators

**Statistical Signatures:**
- **Variance Explosion**: Sudden increase in metric variance before transition
- **Critical Slowing Down**: Metrics become "sticky" near transition point
- **Correlation Changes**: Sudden changes in correlation patterns between metrics

**Implementation Example:**
```python
def detect_phase_transition(metric_history, window=10):
    """General phase transition detector using variance and trend analysis."""
    if len(metric_history) < window * 3:
        return False, -1
    
    # Calculate rolling statistics
    variances = []
    trends = []
    
    for i in range(window, len(metric_history) - window):
        segment = metric_history[i-window:i+window]
        variances.append(np.var(segment))
        trends.append(np.polyfit(range(len(segment)), segment, 1)[0])
    
    # Look for variance spikes followed by trend changes
    for i in range(1, len(variances)-1):
        variance_spike = variances[i] > 2 * np.mean(variances[:i]) if i > 5 else False
        trend_change = abs(trends[i+1] - trends[i-1]) > np.std(trends[:i+1]) if i > 5 else False
        
        if variance_spike and trend_change:
            return True, i + window
    
    return False, -1
```

### Expected Transition Timings

#### Grokking (Modular Arithmetic)
- **Problem Complexity**: Harder operations (multiplication) transition later than easier ones (addition)
- **Model Size**: Larger models may grok earlier but with more dramatic transitions
- **Weight Decay**: Higher weight decay leads to earlier but sharper transitions
- **Typical Range**: 2,000-8,000 epochs for standard setups

#### Simplicity Bias (Visual Tasks)
- **Bias Strength**: Higher training bias leads to later robust learning transitions
- **Dataset Size**: Smaller datasets may never transition; larger datasets transition earlier
- **Architecture**: CNNs may transition faster than MLPs due to inductive biases
- **Typical Range**: 50-200 epochs for visual tasks

#### Sentiment Bias (NLP Tasks)
- **Vocabulary Size**: Larger vocabularies lead to later transitions
- **Bias Topic Frequency**: More frequent bias topics lead to stronger spurious learning
- **Model Architecture**: Transformers may transition faster than RNNs
- **Typical Range**: 30-100 epochs for simple sentiment tasks

### Monitoring and Visualization

#### Real-time Monitoring
```bash
# Use wandb to monitor phase transitions in real-time
python train_script.py --use_wandb --project phase_transitions

# Key plots to watch:
# 1. Test accuracy vs epochs (look for sudden jumps)
# 2. Train-test gap vs epochs (look for sudden closure)  
# 3. Bias gap vs epochs (look for sustained reduction)
# 4. Loss derivatives vs epochs (look for discontinuities)
```

#### Post-training Analysis
```python
# Analyze training curves for phase transitions
def analyze_phase_transitions(results_file):
    """Comprehensive phase transition analysis."""
    with open(results_file) as f:
        results = json.load(f)
    
    # Detect different types of transitions
    grokking_detected, grok_epoch = detect_grokking(results['test_accuracies'])
    bias_detected, bias_epoch = detect_bias_breaking(
        results['bias_conforming_accuracies'], 
        results['bias_conflicting_accuracies']
    )
    
    print(f"Grokking transition: {'Yes' if grokking_detected else 'No'}")
    if grokking_detected:
        print(f"  Occurred at epoch: {grok_epoch}")
    
    print(f"Bias breaking transition: {'Yes' if bias_detected else 'No'}")  
    if bias_detected:
        print(f"  Occurred at epoch: {bias_epoch}")
    
    return {
        'grokking': (grokking_detected, grok_epoch),
        'bias_breaking': (bias_detected, bias_epoch)
    }
```

## 💡 Research Applications

This framework enables studying:
- **Delayed Generalization**: When and why models suddenly generalize
- **Simplicity Bias**: How models prefer simple features over complex ones
- **Robustness vs Accuracy**: Tradeoffs between clean and robust performance
- **Intervention Effects**: How different training methods affect bias learning
- **Phase Transitions**: Detection and analysis of sudden behavioral changes during training

## 🔧 Advanced Usage

### Dataset Fraction Control
All training scripts support using fractions of datasets for faster experimentation:

```bash
# Use 10% of the dataset for quick prototyping
python phenomena/grokking/training/train_modular.py \
    --data_dir ./grok_data --data_fraction 0.1 --epochs 1000

# Use 50% for intermediate experiments  
python phenomena/simplicity_bias/celeba/training/train_bias_celeba.py \
    --data_dir ./celeba_data --data_fraction 0.5 --epochs 50

# Use full dataset (default)
python phenomena/nlp/sentiment_bias/training/train_sentiment_bias.py \
    --data_dir ./sentiment_data --data_fraction 1.0
```

### Enhanced Optimizer Usage
All scripts automatically use enhanced optimizers with phenomenon-specific defaults:

```bash
# Grokking: Uses EnhancedAdamW with high weight decay and gradient clipping
python phenomena/grokking/training/train_modular.py --data_dir ./data

# Simplicity Bias: Uses EnhancedAdamW with adaptive weight decay
python phenomena/simplicity_bias/*/training/train_*.py --data_dir ./data

# Robustness: Uses EnhancedSGD with adaptive momentum
python phenomena/robustness/*/train_*.py --data_dir ./data
```

### Experiment Organization
Results are automatically organized with descriptive names:

```bash
# CelebA experiments include bias characteristics and size
./real_celeba_results/real_celeba_Male_Blond_Hair_tb0.8_testb0.2_size5000/

# Grokking experiments include model and hyperparameter info
./grokking_results/grokking_d128_h4_l2_lr0.001_wd0.01/

# NLP experiments include bias topic and dataset characteristics
./sentiment_bias_results/sentiment_bias_technology_tb0.9_testb0.1_size5000/
```