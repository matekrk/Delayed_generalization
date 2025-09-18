# Enhanced Delayed Generalization Repository - New Features

This document outlines the comprehensive enhancements made to the Delayed Generalization repository as requested in issue #9.

## 🎯 Summary of Additions

### 1. Color Analysis for Bias Detection (`utils/image_analysis.py`)

**Purpose**: Detect simplicity bias patterns in vision datasets, particularly sky correlation in outdoor scenes.

**Key Features**:
- Predominant color extraction using K-means clustering
- Regional analysis (top/middle/bottom, center/edges, full image)
- Multiple color space support (RGB, HSV, LAB)
- Bias detection algorithms for sky correlation and color separability
- Caching system to avoid recomputation
- Visualization tools for color analysis results

**Usage**:
```python
from utils.image_analysis import PredominantColorAnalyzer, analyze_image_colors

# Quick analysis
results = analyze_image_colors(dataset, cache_dir="./cache", sample_size=1000)

# Detailed analysis
analyzer = PredominantColorAnalyzer(n_colors=5, region_analysis=True)
analysis = analyzer.analyze_dataset(dataset, save_results=True)
bias_patterns = analyzer._detect_simplicity_bias(analysis['class_summaries'])
```

### 2. CIFAR-100 and TinyImageNet Support with Color Analysis

**CIFAR-100 Analysis** (`datasets/vision/cifar100_analysis.py`):
- Extended CIFAR-100 dataset class with color analysis
- Automatic detection of sky correlation in outdoor classes
- Superclass-aware bias analysis
- Comprehensive bias reporting with severity levels

**TinyImageNet Analysis** (`datasets/vision/tinyimagenet_analysis.py`):
- Complete TinyImageNet dataset implementation with download support
- Color bias detection for 200 classes
- Analysis of outdoor scenes, vehicles, and object classes
- Risk level assessment and recommendations

**Usage**:
```python
from datasets.vision.cifar100_analysis import analyze_cifar100_colors
from datasets.vision.tinyimagenet_analysis import analyze_tinyimagenet_colors

# Analyze CIFAR-100
cifar_results = analyze_cifar100_colors(root="./data", download=True)
print(f"Train biases: {len(cifar_results['train']['bias_report']['potential_biases'])}")

# Analyze TinyImageNet
tiny_results = analyze_tinyimagenet_colors(root="./data", download=True)
print(f"Risk level: {tiny_results['combined_analysis']['overall_risk_level']}")
```

### 3. GradCAM Implementation (`data_attribution/gradcam/`)

**Purpose**: Visualize model attention patterns to understand what features drive predictions during delayed generalization.

**Key Features**:
- Standard GradCAM implementation for CNN models
- Guided GradCAM and guided backpropagation
- Multi-layer analysis and comparison
- Spatial analysis of attention patterns (regional focus, center of mass)
- Integration with bias detection workflows

**Usage**:
```python
from data_attribution.gradcam.gradcam_attributor import GradCAM, compare_gradcam_layers

# Single layer analysis
gradcam = GradCAM(model, target_layer='layer4')
cam = gradcam.generate_cam(input_tensor, target_class=5)
analysis = gradcam.analyze_image(input_tensor, target_class=5)

# Multi-layer comparison
comparison = compare_gradcam_layers(
    model, input_tensor, 
    layer_names=['layer2', 'layer3', 'layer4'],
    save_path="layer_comparison.png"
)
```

### 4. Effective Learning Rate Computation (`utils/optimizer_utils.py`)

**Purpose**: Monitor how effective learning rates change during training, especially important for understanding AdamW dynamics in delayed generalization.

**Key Features**:
- Precise computation for AdamW, Adam, and SGD optimizers
- Accounts for bias correction, momentum, and weight decay
- Parameter-group-wise and layer-wise analysis
- Real-time monitoring during training
- Evolution tracking and visualization

**Usage**:
```python
from utils.optimizer_utils import compute_effective_lr, LearningRateMonitor, track_lr_evolution

# Compute current effective LR
lr_info = compute_effective_lr(optimizer, step=current_step)
print(f"Effective LR: {lr_info['global_effective_lr']:.6f}")

# Monitor during training
lr_monitor = LearningRateMonitor(optimizer, log_frequency=100)
for step in range(training_steps):
    # ... training code ...
    lr_info = lr_monitor.step(step)
    if lr_info:
        wandb.log({"effective_lr": lr_info['global_effective_lr']})

# Track evolution
evolution = track_lr_evolution(optimizer, steps=range(0, 1000, 100), save_path="lr_evolution.png")
```

### 5. Adversarial Robustness Evaluation (`phenomena/robustness/adversarial/`)

**Purpose**: Evaluate model robustness beyond CIFAR-10-C using adversarial attacks, important for understanding delayed robustness phenomena.

**Key Features**:
- FGSM, PGD, and Carlini & Wagner (C&W) attack implementations
- Multiple attack strengths (weak/medium/strong)
- Comprehensive robustness metrics and analysis
- Visualization tools for robustness results
- Integration with delayed generalization analysis

**Usage**:
```python
from phenomena.robustness.adversarial.adversarial_evaluator import evaluate_model_robustness

# Comprehensive evaluation
results = evaluate_model_robustness(
    model, test_loader, 
    save_dir="./robustness_results"
)

print(f"Robustness level: {results['analysis']['robustness_level']}")
print(f"Vulnerabilities: {len(results['analysis']['vulnerabilities'])}")

# Custom attack configuration
attack_configs = {
    'fgsm_strong': {'epsilon': 0.031},
    'pgd_strong': {'epsilon': 0.031, 'alpha': 0.005, 'num_steps': 40}
}
custom_results = evaluate_model_robustness(model, test_loader, attack_configs)
```

### 6. SLURM Cluster Scripts (`slurm_scripts/`)

**Purpose**: Ready-to-use scripts for running experiments on SLURM clusters with DGX partition.

**Scripts Provided**:
- `run_color_analysis.sh`: Dataset color analysis
- `run_grokking_experiment.sh`: Grokking experiments with modular arithmetic
- `run_simplicity_bias.sh`: Simplicity bias experiments
- `run_robustness_evaluation.sh`: Comprehensive robustness testing
- `run_data_attribution.sh`: TRAK and GradCAM analysis
- `run_comprehensive_analysis.sh`: Full pipeline with dependencies

**Usage**:
```bash
# Submit individual jobs
sbatch slurm_scripts/run_color_analysis.sh
sbatch slurm_scripts/run_grokking_experiment.sh

# Submit comprehensive pipeline
sbatch slurm_scripts/run_comprehensive_analysis.sh

# Monitor progress
squeue -u $USER
```

### 7. Repository Organization Improvements

**Data Script Migration**:
- Moved data generation scripts from `phenomena/*/data/` to `datasets/`
- Improved script organization and discoverability
- Fixed import paths and dependencies

**Compatibility Fixes**:
- Made OpenCV dependency optional with graceful fallbacks
- Fixed relative imports to work across different execution contexts
- Enhanced error handling and user feedback
- Updated requirements.txt with new dependencies

**Documentation Updates**:
- Enhanced README.md with new features and examples
- Added comprehensive usage examples
- Updated repository structure documentation

## 🔬 Integration with Delayed Generalization Research

### Bias Detection Workflow
1. **Dataset Analysis**: Use color analysis to detect potential biases
2. **Training Monitoring**: Track effective learning rates and training dynamics
3. **Model Understanding**: Apply GradCAM to verify attention patterns
4. **Robustness Evaluation**: Test delayed robustness with adversarial attacks
5. **Comprehensive Analysis**: Use SLURM scripts for full experimental pipelines

### Key Research Questions Addressed
- **Sky Bias**: Does the model rely on sky color for outdoor scene classification?
- **Color Separability**: Are classes too easily separable by color?
- **Attention Patterns**: Where does the model focus during delayed generalization?
- **Learning Rate Dynamics**: How do effective learning rates change during grokking?
- **Robustness Development**: When does robustness emerge during training?

## 🛠️ Technical Details

### Dependencies Added
- `opencv-python>=4.5.0` (optional, for advanced color space analysis)
- `requests>=2.25.0` (for dataset downloads)
- `pillow>=8.0.0` (for image processing)
- `scipy>=1.7.0` (for scientific computing)

### Caching Strategy
- Color analysis results are cached using pickle format
- Cache keys based on dataset hash and analysis parameters
- Configurable cache directories for different experiments
- Automatic cache invalidation for parameter changes

### Performance Considerations
- Color analysis uses sampling for large datasets (configurable)
- GradCAM supports batch processing where applicable
- Effective LR computation is lightweight and suitable for frequent monitoring
- Adversarial attacks are optimized for GPU execution

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Color Analysis**:
   ```python
   from datasets.vision.cifar100_analysis import analyze_cifar100_colors
   results = analyze_cifar100_colors(download=True)
   ```

3. **Monitor Training**:
   ```python
   from utils.optimizer_utils import LearningRateMonitor
   lr_monitor = LearningRateMonitor(optimizer)
   # ... integrate into training loop
   ```

4. **Evaluate Robustness**:
   ```python
   from phenomena.robustness.adversarial.adversarial_evaluator import evaluate_model_robustness
   evaluate_model_robustness(model, test_loader)
   ```

5. **Run on Cluster**:
   ```bash
   sbatch slurm_scripts/run_comprehensive_analysis.sh
   ```

This comprehensive enhancement provides researchers with powerful tools to investigate delayed generalization phenomena across multiple dimensions: bias detection, model interpretability, learning dynamics, and robustness evaluation.