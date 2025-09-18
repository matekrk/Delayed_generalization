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

## 🗂️ Repository Structure

```
delayed_generalization/
├── phenomena/           # Different types of delayed generalization
│   ├── grokking/
│   ├── simplicity_bias/
│   └── phase_transitions/
├── datasets/           # Datasets known for delayed generalization
│   ├── algorithmic/
│   ├── vision/
│   └── nlp/
├── models/            # Model architectures prone to delayed generalization  
│   ├── transformers/
│   ├── cnns/
│   └── mlps/
├── optimization/      # Training techniques and tricks
│   ├── warmup/
│   ├── regularization/
│   └── scheduling/
├── experiments/       # Reproducible experimental setups
└── tools/            # Analysis and visualization tools
```

## 🚀 Quick Start

Browse the [catalog of scenarios](./CATALOG.md) or explore specific categories:

- **[Phenomena](./phenomena/)**: Types of delayed generalization
- **[Datasets](./datasets/)**: Datasets exhibiting these phenomena  
- **[Models](./models/)**: Architectures prone to delayed generalization
- **[Optimization](./optimization/)**: Training techniques and hyperparameters
- **[Experiments](./experiments/)**: Reproducible setups and results

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
