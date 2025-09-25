# Getting Started with Delayed Generalization Research

Welcome to the Delayed Generalization research framework! This guide will help you quickly get started with studying delayed generalization phenomena using our comprehensive toolkit.

## 🚀 Quick Setup

### Prerequisites

```bash
# Install required dependencies
pip install torch torchvision wandb numpy matplotlib seaborn pandas transformers scikit-learn
```

### Environment Setup

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## 📚 Overview of Delayed Generalization Phenomena

This framework supports studying several key delayed generalization phenomena:

### 1. **Grokking** 🧠
- **What**: Sudden generalization after extended memorization
- **When**: Typically after 1000-10000 epochs in algorithmic tasks
- **Example**: Learning modular arithmetic patterns

### 2. **Simplicity Bias** 🎯
- **What**: Models initially learn simple spurious features before true patterns
- **When**: 50-500 epochs depending on bias strength
- **Example**: Learning background colors before object shapes

### 3. **Robustness vs Accuracy** 🛡️
- **What**: Trade-offs between clean accuracy and corruption robustness
- **When**: Throughout training, with possible phase transitions
- **Example**: CIFAR-10-C corruption evaluation

### 4. **Continual Learning** 🔄
- **What**: Learning new tasks while retaining old knowledge
- **When**: Across multiple sequential tasks
- **Example**: CIFAR-100 divided into 10 tasks

### 5. **Phase Transitions** ⚡
- **What**: Sudden emergence of new capabilities during scaling
- **When**: At critical model sizes or data amounts
- **Example**: Language model emergent abilities

## 🔬 Your First Experiment: Simplicity Bias on Colored MNIST

Let's start with a classic delayed generalization experiment: studying simplicity bias on colored MNIST.

### Step 1: Import Required Modules

```python
# Core framework imports
from models.vision import create_model_for_phenomenon, ModelFactory
from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
from phenomena.simplicity_bias.colored_mnist.train_colored_mnist import ColoredMNISTTrainer
from phenomena.simplicity_bias.colored_mnist.data.generate_colored_mnist import create_colored_mnist_dataset

# Standard imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
```

### Step 2: Create Colored MNIST Dataset

```python
# Generate biased colored MNIST dataset
print("Generating colored MNIST dataset...")

train_dataset, test_dataset, metadata = create_colored_mnist_dataset(
    data_dir='./data/colored_mnist',
    num_train_samples=10000,
    num_test_samples=2000,
    color_correlation=0.9,  # Strong bias: 90% correlation between color and label
    noise_level=0.1
)

print(f"Dataset created:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Color correlation: {metadata['color_correlation']}")
print(f"  Classes: {metadata['num_classes']}")
```

### Step 3: Create Model

```python
# Create model optimized for simplicity bias research
model = create_model_for_phenomenon(
    phenomenon='simplicity_bias',
    model_type='mobilenet',  # Efficient for quick experiments
    efficiency='light',
    num_classes=10
)

print(f"Model created:")
print(f"  Architecture: MobileNet")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.2f} MB")
```

### Step 4: Setup WandB Logging (Optional)

```python
# Initialize sophisticated experiment tracking
wandb_logger = DelayedGeneralizationLogger(
    project_name="delayed-generalization-tutorial",
    experiment_name="colored_mnist_simplicity_bias",
    config={
        'dataset': 'colored_mnist',
        'phenomenon': 'simplicity_bias',
        'model': 'mobilenet_light',
        'color_correlation': 0.9,
        'num_classes': 10,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    },
    phenomenon_type='simplicity_bias',
    tags=['tutorial', 'colored_mnist', 'simplicity_bias'],
    notes="First experiment following the getting started guide"
)

print("WandB logging initialized!")
print(f"Experiment: {wandb_logger.run.name}")
```

### Step 5: Create Data Loaders

```python
# Create data loaders for training
train_loader = DataLoader(
    train_dataset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=4
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=128, 
    shuffle=False, 
    num_workers=4
)

print(f"Data loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
```

### Step 6: Setup Training

```python
# Create trainer with bias analysis capabilities
trainer = ColoredMNISTTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    learning_rate=1e-3,
    weight_decay=1e-4,
    wandb_logger=wandb_logger
)

print("Trainer initialized with bias analysis capabilities!")
```

### Step 7: Train and Monitor Delayed Generalization

```python
# Train model and study delayed generalization patterns
print("Starting training...")
print("=" * 50)

results = trainer.train(
    epochs=200,  # Enough epochs to observe delayed generalization
    log_interval=10,
    save_dir='./results/tutorial_colored_mnist'
)

print("=" * 50)
print("Training completed!")

# Print key results
print(f"\nFinal Results:")
print(f"  Test Accuracy: {results['final_test_acc']:.2f}%")
print(f"  Shape Accuracy: {results['final_shape_acc']:.2f}%")
print(f"  Color Accuracy: {results['final_color_acc']:.2f}%")
print(f"  Bias Score: {results['final_bias_score']:.3f}")

# Check for delayed generalization
if results.get('bias_reduction_detected'):
    print(f"  🎉 Delayed generalization detected!")
    print(f"     Bias reduction: {results['bias_reduction']:.3f}")
else:
    print(f"  ⏳ No clear delayed generalization pattern detected")
```

### Step 8: Analyze Results

```python
# Analyze training dynamics
print("\nAnalyzing training dynamics...")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
epochs = range(1, len(results['training_history']['test_accuracies']) + 1)

# Overall accuracy
axes[0, 0].plot(epochs, results['training_history']['test_accuracies'], label='Test Accuracy')
axes[0, 0].plot(epochs, results['training_history']['train_accuracies'], label='Train Accuracy')
axes[0, 0].set_title('Overall Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Bias metrics
if 'shape_accuracies' in results['training_history']:
    axes[0, 1].plot(epochs, results['training_history']['shape_accuracies'], label='Shape Accuracy')
    axes[0, 1].plot(epochs, results['training_history']['color_accuracies'], label='Color Accuracy')
    axes[0, 1].set_title('Feature-Specific Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Bias score evolution
if 'bias_scores' in results['training_history']:
    axes[1, 0].plot(epochs, results['training_history']['bias_scores'], color='red')
    axes[1, 0].set_title('Bias Score Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Bias Score')
    axes[1, 0].grid(True, alpha=0.3)

# Loss curves
axes[1, 1].plot(epochs, results['training_history']['test_losses'], label='Test Loss')
axes[1, 1].plot(epochs, results['training_history']['train_losses'], label='Train Loss')
axes[1, 1].set_title('Loss Curves')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/tutorial_colored_mnist/analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Check './results/tutorial_colored_mnist/' for detailed results.")
```

## 🎯 Next Steps: Exploring Other Phenomena

### Grokking on Modular Arithmetic

```python
# Quick setup for grokking experiments
from phenomena.grokking.training.train_modular import GrokkingTrainer
from phenomena.grokking.models.simple_transformer import create_grokking_model

# Create transformer for algorithmic task
grokking_model = create_grokking_model(
    vocab_size=97,  # Modular arithmetic mod 97
    d_model=128,
    n_heads=4,
    n_layers=2,
    num_classes=97
)

print("Grokking model ready for modular arithmetic!")
print(f"Parameters: {sum(p.numel() for p in grokking_model.parameters()):,}")
```

### Robustness on CIFAR-100-C

```python
# Quick setup for robustness experiments
from phenomena.robustness.cifar100c.train_cifar100c import CIFAR100CTrainer, CIFAR100CModel

# Create robust model for corruption evaluation
robust_model = CIFAR100CModel(num_classes=100, model_size='medium')

print("Robustness model ready for CIFAR-100-C!")
print(f"Parameters: {sum(p.numel() for p in robust_model.parameters()):,}")
```

### Continual Learning on CIFAR-100

```python
# Quick setup for continual learning
from phenomena.continual_learning.cifar100_10tasks.train_continual_cifar100 import ContinualLearningTrainer

# Create continual learning model
cl_model = create_model_for_phenomenon(
    phenomenon='continual_learning',
    efficiency='medium',
    num_classes=100
)

print("Continual learning model ready!")
print("Will learn 10 tasks sequentially with 10 classes each")
```

## 🛠️ Advanced Features

### 1. Enhanced WandB Analysis

```python
# Advanced metrics tracking
wandb_logger.analyze_mutual_information(
    epoch=100,
    model=model,
    data_loader=test_loader
)

# Cross-seed meta-analysis
seed_results = {
    42: {'final_test_acc': 85.2, 'bias_score': 0.15},
    123: {'final_test_acc': 84.8, 'bias_score': 0.18},
    456: {'final_test_acc': 85.0, 'bias_score': 0.16}
}

meta_analysis = DelayedGeneralizationLogger.perform_meta_analysis(
    project_name="delayed-generalization-tutorial",
    phenomenon_type="simplicity_bias",
    seed_results=seed_results
)
```

### 2. Model Architecture Comparison

```python
# Compare different architectures
architectures = ['mobilenet', 'vit']
efficiencies = ['light', 'medium']

results_comparison = {}
for arch in architectures:
    for eff in efficiencies:
        model = create_model_for_phenomenon(
            phenomenon='simplicity_bias',
            model_type=arch,
            efficiency=eff,
            num_classes=10
        )
        
        analysis = ModelFactory.analyze_model(model)
        results_comparison[f"{arch}_{eff}"] = analysis

print("Architecture comparison:")
for name, analysis in results_comparison.items():
    print(f"  {name}: {analysis['total_parameters']:,} params, {analysis['efficiency_class']}")
```

### 3. Hyperparameter Sweeps

```python
# Create automated hyperparameter sweeps
from utils.wandb_integration.delayed_generalization_logger import create_advanced_optimizer_sweep

sweep_config = create_advanced_optimizer_sweep(
    phenomenon_type='simplicity_bias',
    focus_optimizers=['adam', 'adamw', 'sgd'],
    include_scheduling=True,
    include_regularization=True
)

print("Sweep configuration created!")
print(f"Parameters to optimize: {list(sweep_config['parameters'].keys())}")
```

## 📖 Additional Resources

### Experiment Templates
- `examples/grokking_basic.py` - Basic grokking experiment
- `examples/simplicity_bias_advanced.py` - Advanced bias analysis
- `examples/continual_learning_demo.py` - Continual learning demo
- `examples/robustness_evaluation.py` - Robustness evaluation

### Documentation
- `IMPLEMENTATION_STATUS.md` - Current implementation status
- `ENHANCED_FEATURES.md` - Advanced feature documentation
- `CONTRIBUTING.md` - How to contribute new phenomena

### Research Papers
- Original grokking paper: "Grokking: Generalization Beyond Overfitting"
- Simplicity bias: "Learning Rule vs Implementation"
- Phase transitions: "Phase Transitions in Deep Learning"

## 🎉 Congratulations!

You've successfully run your first delayed generalization experiment! Here's what you accomplished:

✅ **Generated a biased dataset** with controlled spurious correlations
✅ **Created an efficient model** optimized for your research
✅ **Set up sophisticated logging** with WandB integration  
✅ **Trained and analyzed** delayed generalization patterns
✅ **Visualized training dynamics** and bias evolution

### What's Next?

1. **Experiment with different phenomena** - Try grokking, robustness, or continual learning
2. **Compare architectures** - Test Vision Transformers vs MobileNets
3. **Run parameter sweeps** - Find optimal hyperparameters automatically
4. **Analyze across seeds** - Study statistical significance with meta-analysis
5. **Contribute new phenomena** - Add your own delayed generalization tasks

## 💡 Tips for Success

- **Start small**: Use light models and small datasets for quick iteration
- **Monitor closely**: Watch for phase transitions and sudden improvements
- **Compare baselines**: Always compare against simple baselines
- **Statistical rigor**: Run multiple seeds and compute confidence intervals
- **Share findings**: Use WandB to share and collaborate on results

Happy researching! 🚀🔬

---

## 🚀 Legacy Quick Start Examples

For reference, here are the original command-line examples:

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

### Colored MNIST (Simplicity Bias)

Study color bias in digit classification:

```bash
# Generate colored MNIST with strong color-digit correlation
python data/vision/colored_mnist/generate_colored_mnist.py \
    --num_train_samples 10000 --num_test_samples 2000 \
    --color_correlation 0.9 --output_dir ./colored_mnist_data

# Train CNN model to study bias learning dynamics
python phenomena/simplicity_bias/colored_mnist/train_colored_mnist.py \
    --data_dir ./colored_mnist_data --epochs 200 --use_wandb
```

### CIFAR-10-C (Robustness)

Study clean vs corrupted image classification:

```bash
# Generate synthetic CIFAR-10-C dataset
python data/vision/cifar10c/generate_synthetic_cifar10c.py \
    --train_corruptions gaussian_noise motion_blur \
    --test_corruptions fog brightness --output_dir ./cifar10c_data

# Train CNN model
python phenomena/robustness/cifar10c/train_cifar10c.py \
    --data_dir ./cifar10c_data --epochs 100
```

### CIFAR-100-C (Advanced Robustness)

Study robustness with more classes and complexity:

```bash
# Train on CIFAR-100 with corruption robustness evaluation
python phenomena/robustness/cifar100c/train_cifar100c.py \
    --data_dir ./cifar100_data --epochs 200 --use_wandb \
    --model_size medium --evaluate_corruptions
```

### Continual Learning (CIFAR-100 → 10 Tasks)

Study catastrophic forgetting and delayed generalization across sequential tasks:

```bash
# Train continual learning on CIFAR-100 divided into 10 tasks
python phenomena/continual_learning/cifar100_10tasks/train_continual_cifar100.py \
    --data_dir ./cifar100_continual --num_tasks 10 --epochs_per_task 50 \
    --memory_size 1000 --use_wandb
```

### Sentiment Bias NLP (Improved Difficulty)

Study sentiment classification with topic bias (now more challenging):

```bash
# Generate sentiment bias dataset (technology bias)
python data/nlp/sentiment/generate_sentiment_bias.py \
    --bias_topic technology --train_bias 0.9 --test_bias 0.1 \
    --train_size 5000 --test_size 1000 --output_dir ./sentiment_bias_data

# Train sentiment classifier with configurable difficulty
python phenomena/nlp/sentiment_bias/training/train_sentiment_bias.py \
    --data_dir ./sentiment_bias_data --epochs 100 --use_wandb \
    --embed_dim 64 --hidden_dim 128 --dropout 0.3 --learning_rate 5e-4

# For even more challenging settings (smaller model, slower learning)
python phenomena/nlp/sentiment_bias/training/train_sentiment_bias.py \
    --data_dir ./sentiment_bias_data --epochs 200 --use_wandb \
    --embed_dim 32 --hidden_dim 64 --dropout 0.5 --learning_rate 1e-4

# Alternative bias topics
python data/nlp/sentiment/generate_sentiment_bias.py \
    --bias_topic politics --neutral_topics sports entertainment science \
    --output_dir ./sentiment_politics_bias
```

## 🔍 Data Attribution and Analysis Methods

### TRAK (Training Data Attribution)

Understand which training examples most influence model predictions:

```bash
# Run TRAK analysis on a trained model
python data_attribution/trak/run_trak_analysis.py \
    --model_path ./results/trained_model.pth \
    --data_dir ./dataset --target_examples ./test_samples \
    --output_dir ./trak_results

# Analyze TRAK scores and visualize influential examples
python data_attribution/trak/analyze_trak_scores.py \
    --trak_results ./trak_results --save_dir ./analysis_output
```

### GradCAM (Gradient-based Attribution)

Visualize what parts of input the model focuses on:

```bash
# Generate GradCAM visualizations for image classification
python data_attribution/gradcam/run_gradcam.py \
    --model_path ./results/trained_model.pth \
    --data_dir ./dataset --target_layer layer4 \
    --output_dir ./gradcam_visualizations

# Create comparison plots of attention across training phases
python data_attribution/gradcam/compare_attention_phases.py \
    --model_checkpoints ./checkpoints/ --data_dir ./dataset
```

### Post-Training Analysis

Comprehensive analysis of training dynamics and delayed generalization patterns:

```bash
# Analyze bias evolution over training
python visualization/bias_analysis.py \
    --results_file ./results/training_history.json \
    --save_dir ./bias_analysis

# Detect phase transitions in training curves
python utils/phase_transition_detector.py \
    --metrics_file ./results/metrics.json \
    --output_dir ./phase_analysis

# Color bias analysis for vision tasks
python utils/image_analysis.py \
    --dataset_path ./data/vision/dataset \
    --analysis_type color_bias --save_results ./color_analysis
```

### Example Analysis Workflow

Complete analysis pipeline for understanding delayed generalization:

```bash
# 1. Train model with detailed logging
python phenomena/simplicity_bias/colored_mnist/train_colored_mnist.py \
    --data_dir ./colored_mnist --epochs 200 --save_checkpoints \
    --checkpoint_interval 20 --use_wandb

# 2. Run TRAK attribution analysis
python data_attribution/trak/run_trak_analysis.py \
    --model_path ./results/final_model.pth \
    --data_dir ./colored_mnist --output_dir ./trak_colored_mnist

# 3. Generate GradCAM visualizations
python data_attribution/gradcam/run_gradcam.py \
    --model_path ./results/final_model.pth \
    --data_dir ./colored_mnist --output_dir ./gradcam_colored_mnist

# 4. Analyze training dynamics and phase transitions
python visualization/bias_analysis.py \
    --results_file ./results/training_history.json \
    --save_dir ./dynamics_analysis

# 5. Create comprehensive report
python utils/generate_analysis_report.py \
    --experiment_dir ./results --trak_dir ./trak_colored_mnist \
    --gradcam_dir ./gradcam_colored_mnist --output ./final_report.html
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

### Sentiment Bias NLP Timeline (Improved Difficulty)
- **Phase 1 (Epochs 0-50)**: Model learns topic-sentiment spurious correlations (e.g., technology → positive)
- **Phase 2 (Epochs 50-120)**: High bias-conforming accuracy, poor bias-conflicting accuracy  
- **Phase 3 (Epochs 120+)**: Potential delayed generalization to true sentiment patterns
- **Key Transition**: Watch for bias gap reduction indicating robust sentiment learning
- **With Smaller Models**: Delayed generalization may occur much later (epochs 150-300+)

### Colored MNIST Timeline
- **Phase 1 (Epochs 0-30)**: Rapid learning of color-digit correlations
- **Phase 2 (Epochs 30-100)**: High color-based accuracy, poor shape-based accuracy
- **Phase 3 (Epochs 100+)**: Gradual shift to shape-based digit recognition
- **Key Transition**: Color accuracy plateaus while shape accuracy increases

### Continual Learning Timeline
- **Task 1**: Normal learning curve, high accuracy achieved
- **Task 2-5**: Catastrophic forgetting of previous tasks, new task learning
- **Task 6+**: Potential delayed consolidation and improved retention
- **Final Phases**: Possible emergence of meta-learning capabilities

### CIFAR-100-C Robustness Timeline
- **Clean CIFAR-100**: Slower convergence than CIFAR-10 due to more classes
- **Corrupted Images**: Very poor initial performance, gradual improvement
- **Class-specific Robustness**: Some classes robust earlier than others
- **Late Training**: Potential emergence of corruption-invariant features

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

### Robustness (CIFAR-10-C, CIFAR-100-C)
- **Clean vs corrupted accuracy gap**: Robustness measure
- **Corruption-specific performance**: Per-corruption type accuracy
- **Robustness progression**: How corruption resistance develops over training

### Continual Learning (CIFAR-100 → 10 Tasks)
- **Average accuracy**: Performance across all learned tasks
- **Forgetting measure**: Accuracy drop on previous tasks when learning new ones
- **Forward transfer**: How well previous learning helps with new tasks
- **Backward transfer**: How new learning affects previous task performance
- **Task-specific accuracy**: Individual performance on each of the 10 tasks

### Colored MNIST (Simplicity Bias)
- **Shape accuracy**: Performance based on digit shape (robust feature)
- **Color accuracy**: Performance based on digit color (spurious feature)
- **Color-shape correlation**: Strength of spurious correlation in predictions
- **Robust generalization**: Performance on color-conflicting test examples

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