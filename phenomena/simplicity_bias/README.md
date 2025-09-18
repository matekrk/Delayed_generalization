# Simplicity Bias: Learning Spurious Correlations Before True Features

## 📋 Overview

**Simplicity Bias** refers to the tendency of machine learning models to learn simpler patterns or spurious correlations before learning the true underlying features that enable robust generalization. This leads to a delayed generalization pattern where models appear to perform well on training data but fail on test data until they eventually learn the correct features.

## 🔬 Key Characteristics

- **Phase 1**: Rapid learning of spurious correlations (easy patterns)
- **Phase 2**: High training accuracy but poor test accuracy due to spurious features
- **Phase 3**: Gradual or sudden transition to learning true features
- **Phase 4**: Improved generalization as true features dominate

## 🐦 Waterbirds: The Classic Example

### Problem Setup
- **Task**: Classify birds into water birds vs land birds
- **Spurious Correlation**: Water birds often photographed on water backgrounds, land birds on land backgrounds
- **True Feature**: Bird morphology and characteristics
- **Challenge**: Models learn background instead of bird features

### Dataset Details
- **Source**: Caltech-UCSD Birds (CUB) + Places backgrounds
- **Classes**: 2 (water birds, land birds)  
- **Spurious Feature**: Background (water vs land)
- **Group Structure**:
  - Water birds on water backgrounds (majority)
  - Water birds on land backgrounds (minority)
  - Land birds on land backgrounds (majority)  
  - Land birds on water backgrounds (minority)

### Typical Results
- **Standard Training**: High average accuracy, poor worst-group accuracy
- **Delayed Generalization**: Models eventually learn bird features with proper techniques
- **Timeline**: Can take hundreds of epochs to overcome background bias

## 🎨 Colored MNIST: Synthetic Simplicity Bias

### Problem Setup  
- **Task**: Classify MNIST digits (0-9)
- **Spurious Correlation**: Digit color correlates with class in training
- **True Feature**: Digit shape/form
- **Challenge**: Models learn color instead of shape

### Experimental Design
```python
# Training set: Color correlates with label (e.g., 90% correlation)
# Test set: Color anti-correlates or is randomized
# Goal: Learn shape features for robust generalization
```

### Timeline
- **Initial Learning**: Color features learned in ~10-50 epochs
- **Shape Learning**: Requires 100-500 epochs typically
- **Intervention Effects**: Various techniques can accelerate shape learning

## ⚙️ Factors Influencing Simplicity Bias

### Data Characteristics
- **Correlation Strength**: Stronger spurious correlations = more bias
- **Feature Complexity**: Simpler spurious features learned first
- **Sample Size**: Smaller datasets may exacerbate bias
- **Group Balance**: Minority groups reveal the bias

### Model Architecture
- **Inductive Biases**: CNN biases toward texture, transformers toward global patterns
- **Capacity**: Over-parameterized models more prone to memorization
- **Regularization**: Can help or hurt depending on implementation

### Training Dynamics
- **Learning Rate**: Faster learning may entrench spurious patterns
- **Batch Size**: Larger batches may reinforce majority patterns
- **Optimization Algorithm**: SGD vs Adam show different bias patterns

## 🛠️ Mitigation Strategies

### 1. Group Distributionally Robust Optimization (Group DRO)
- **Idea**: Minimize worst-group loss instead of average loss
- **Effect**: Forces learning of robust features
- **Timeline**: Can accelerate transition to true features

### 2. Environment-Based Training
- **Idea**: Train across multiple environments/domains
- **Examples**: Invariant Risk Minimization (IRM), Domain Adaptation
- **Effect**: Discovers invariant predictive features

### 3. Data Augmentation
- **Background Augmentation**: For Waterbirds, mix backgrounds
- **Color Augmentation**: For Colored MNIST, randomize colors
- **Effect**: Breaks spurious correlations during training

### 4. Regularization Techniques
- **Weight Decay**: Can help prevent overfitting to spurious features
- **Dropout**: May reduce memorization of simple patterns
- **Early Stopping**: Prevent over-optimization on spurious features

## 📊 Experimental Setups

### Waterbirds Standard Setup
```python
# Dataset splits
train_split = 0.6  # With spurious correlation
val_split = 0.2    # Balanced validation
test_split = 0.2   # Test generalization

# Model: ResNet-50 (pretrained)
# Optimizer: SGD with momentum
# Learning rate: 1e-3 with cosine decay
# Batch size: 128
# Regularization: Weight decay 1e-4

# Metrics to track:
# - Average accuracy
# - Worst-group accuracy  
# - Per-group accuracy breakdown
```

### Colored MNIST Setup
```python
# Correlation strength: 0.9 (training), 0.1 (test)
# Colors: Red/Green or custom color schemes
# Architecture: Simple CNN or MLP
# Training: Standard cross-entropy loss

# Key metrics:
# - Train accuracy (color-based)
# - Test accuracy (shape-based)
# - Feature attribution analysis
```

## 🔍 Analysis Techniques

### Feature Attribution
- **Grad-CAM**: Visualize what regions models focus on
- **Integrated Gradients**: Understand feature importance
- **Saliency Maps**: Track attention shifts over training

### Group-wise Analysis
- **Confusion Matrices**: Per-group performance breakdown
- **Loss Curves**: Track majority vs minority group losses
- **Representation Analysis**: t-SNE/UMAP of learned features

### Training Dynamics
- **Learning Speed**: Track when different features are acquired
- **Forgetting Patterns**: Monitor if spurious features are unlearned
- **Phase Transitions**: Identify sudden shifts in feature usage

## 📈 Reproduction Guidelines

### For Waterbirds
1. **Data Preparation**: Download CUB + Places, create group splits
2. **Baseline Training**: Standard ERM to observe bias
3. **Robust Training**: Implement Group DRO or other techniques
4. **Evaluation**: Focus on worst-group accuracy as key metric

### For Colored MNIST
1. **Dataset Creation**: Generate color-correlated MNIST
2. **Baseline**: Train standard CNN/MLP
3. **Intervention**: Apply debiasing techniques
4. **Analysis**: Track color vs shape feature learning

## 🧪 Extensions and Variations

### Other Datasets with Simplicity Bias
- **CelebA**: Gender classification (background bias)
- **CIFAR-10-C**: Corruption vs content learning
- **NLP Datasets**: Lexical vs semantic biases

### Multi-Step Bias
- **Hierarchical Biases**: Multiple levels of spurious features
- **Sequential Learning**: Different biases learned at different stages
- **Compositional Bias**: Combinations of spurious features

## 🔗 References

1. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
2. Arjovsky et al. (2019). "Invariant Risk Minimization" 
3. Geirhos et al. (2020). "Shortcut Learning in Deep Neural Networks"
4. Shah et al. (2020). "The Pitfalls of Simplicity Bias in Neural Networks"
5. Kirichenko et al. (2022). "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations"