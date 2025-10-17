# Delayed Generalization in Machine Learning

A comprehensive collection of scenarios, datasets, models, and optimization techniques where machine learning models exhibit **delayed generalization** - phenomena where performance on test data significantly lags behind performance on training data, often showing sudden improvements after extended training periods.

## 🎯 What is Delayed Generalization?

Delayed generalization refers to scenarios where:
- Training accuracy increases rapidly early in training
- Test/validation accuracy remains poor or improves slowly  
- Eventually, test performance "catches up" often showing sudden jumps
- The gap between train and test performance persists for extended periods

## 📚 Notable Phenomena

### 🧠 [Grokking](./phenomena/grokking/)
- Sudden generalization on algorithmic tasks after extensive memorization
- Originally observed in modular arithmetic and other mathematical operations
- Characterized by phase transitions from memorization to generalization

### 🐦 [Simplicity Bias](./phenomena/simplicity_bias/)
- Models learning spurious correlations before true patterns
- Classic example: Waterbirds dataset (background vs. bird type)
- Often requires specialized techniques to overcome

### ⚡ [Phase Transitions](./phenomena/phase_transitions/)
- Sharp transitions between different learning regimes
- Sudden improvements in generalization capabilities
- Often associated with scaling laws and emergent capabilities

### 🛡️ [Robustness](./phenomena/robustness/)
- Models showing delayed robustness to corruptions and adversarial attacks
- Gradual improvement in robustness during extended training
- Evaluation using CIFAR-10-C and adversarial attack methods

## 🗂️ Repository Structure

```
delayed_generalization/
├── phenomena/           # Different types of delayed generalization
│   ├── grokking/
│   ├── simplicity_bias/
│   ├── phase_transitions/
│   └── robustness/     # Robustness evaluation including adversarial attacks
├── data/           # Datasets known for delayed generalization
│   ├── algorithmic/
│   ├── vision/         # Includes CIFAR-100 and TinyImageNet analysis
│   ├── nlp/
│   └── phase_transitions/
├── models/            # Model architectures prone to delayed generalization  
│   ├── transformers/
│   ├── cnns/
│   └── mlps/
├── optimization/      # Training techniques and tricks
│   ├── warmup/
│   ├── regularization/
│   └── scheduling/
├── data_attribution/  # TRAK and GradCAM for understanding model behavior
│   ├── trak/
│   └── gradcam/
├── utils/            # Analysis utilities including color analysis
├── slurm_scripts/    # Scripts for running on SLURM clusters
├── experiments/      # Reproducible experimental setups
└── tools/           # Analysis and visualization tools
```

## 🚀 Quick Start

Browse the [catalog of scenarios](./CATALOG.md) or explore specific categories:

- **[Phenomena](./phenomena/)**: Types of delayed generalization
- **[Data](./data/)**: Datasets exhibiting these phenomena  
- **[Models](./models/)**: Architectures prone to delayed generalization
- **[Optimization](./optimization/)**: Training techniques and hyperparameters
- **[Data Attribution](./data_attribution/)**: TRAK and GradCAM for model understanding
- **[Utils](./utils/)**: Analysis utilities including color bias detection
- **[SLURM Scripts](./slurm_scripts/)**: Ready-to-use cluster scripts
- **[Experiments](./experiments/)**: Reproducible setups and results

## 🛠️ Key Capabilities

### Color Analysis for Bias Detection
```python
from utils.image_analysis import analyze_image_colors
from data.vision.cifar100_analysis import analyze_cifar100_colors
from data.vision.tinyimagenet_analysis import analyze_tinyimagenet_colors

# Analyze CIFAR-100 for color bias
results = analyze_cifar100_colors(root="./data", download=True)
print(f"Potential biases found: {len(results['train']['bias_report']['potential_biases'])}")

# Analyze TinyImageNet 
results = analyze_tinyimagenet_colors(root="./data", download=True)
```

### GradCAM for Model Understanding
```python
from data_attribution.gradcam.gradcam_attributor import GradCAM

# Initialize GradCAM
gradcam = GradCAM(model, target_layer='layer4')

# Generate attention maps
cam = gradcam.generate_cam(input_tensor, target_class=5)
gradcam.visualize_cam(input_tensor, cam, save_path="attention_map.png")
```

### Effective Learning Rate Computation
```python
from utils.optimizer_utils import compute_effective_lr, LearningRateMonitor

# Monitor effective learning rate during training
lr_monitor = LearningRateMonitor(optimizer)
lr_info = lr_monitor.step(current_step)
if lr_info:
    print(f"Effective LR: {lr_info['global_effective_lr']:.6f}")
```

### Robustness Evaluation
```python
from phenomena.robustness.adversarial.adversarial_evaluator import evaluate_model_robustness

# Comprehensive robustness evaluation
results = evaluate_model_robustness(model, test_loader, save_dir="./robustness_results")
print(f"Robustness level: {results['analysis']['robustness_level']}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for:
- How to add new scenarios
- Documentation standards
- Code and experiment submission guidelines
- Citation requirements

## 📖 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{delayed_generalization_repo,
  title={Delayed Generalization in Machine Learning: A Comprehensive Collection},
  author={Community Contributors},
  year={2024},
  url={https://github.com/matekrk/Delayed_generalization}
}
```

## 📄 License

This repository is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
