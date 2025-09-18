# Contributing to Delayed Generalization Repository

Thank you for your interest in contributing to this collection of delayed generalization scenarios! This repository aims to be a comprehensive resource for researchers studying phenomena where ML models show delayed improvement in test performance relative to training performance.

## 🎯 What We're Looking For

### New Phenomena
- Novel types of delayed generalization not yet documented
- Variations of known phenomena (e.g., multi-step grokking)
- Cross-domain examples of similar patterns

### Datasets
- New datasets exhibiting delayed generalization
- Synthetic datasets designed to study specific aspects
- Real-world datasets with documented delayed patterns

### Models & Architectures
- Architectures particularly prone to delayed generalization
- Model modifications that influence timing/quality of generalization
- Architectural innovations that help or hinder the phenomena

### Optimization Techniques
- Training methods that affect delayed generalization
- Hyperparameter configurations that reliably produce phenomena
- Novel optimization approaches

### Experimental Results
- Reproduction studies with detailed methodology
- Hyperparameter sensitivity analyses
- Scaling studies (model size, dataset size effects)

## 📋 Contribution Guidelines

### 1. Choose Your Contribution Type

#### 📄 Documentation Contribution
- Add new phenomenon description
- Improve existing documentation
- Add missing references or experimental details

#### 🧪 Experimental Contribution  
- Share experimental results and configurations
- Provide reproduction scripts and data
- Submit hyperparameter studies

#### 💻 Code Contribution
- Data generation scripts
- Model implementations
- Analysis and visualization tools

#### 📊 Dataset Contribution
- New datasets for delayed generalization research
- Processed versions of existing datasets
- Benchmark configurations and splits

### 2. Documentation Standards

#### Phenomenon Documentation
Each new phenomenon should include:

```markdown
# Phenomenon Name

## 📋 Overview
- Clear definition and key characteristics
- How it relates to delayed generalization

## 🔬 Key Characteristics  
- Detailed description of the pattern
- Timeline and phases of the phenomenon
- Distinguishing features from other phenomena

## 📊 Experimental Evidence
- Original paper(s) where discovered/studied
- Key experimental results
- Reproduction requirements

## ⚙️ Factors Influencing the Phenomenon
- Hyperparameters, architectures, datasets that affect it
- Known techniques to encourage/discourage the phenomenon

## 🛠️ Experimental Setup
- Standard configurations for reproducing
- Recommended hyperparameters
- Monitoring and evaluation protocols

## 🔗 References
- Academic papers, blog posts, implementations
```

#### Dataset Documentation
Each dataset should include:

```markdown
# Dataset Name

## 📋 Overview
- Task description and delayed generalization pattern
- Why this dataset is useful for research

## 📊 Dataset Details
- Size, splits, format
- Download/generation instructions
- Licensing information

## 🎯 Experimental Configurations
- Standard train/test splits
- Recommended evaluation metrics
- Baseline results

## 🔧 Usage Examples
- Code snippets for loading/using
- Integration with common frameworks

## 📚 References
- Original papers, related work
```

### 3. Code Standards

#### File Organization
```
phenomena/phenomenon_name/
├── README.md                 # Main documentation
├── experiments/             # Experimental configurations
│   ├── config_basic.yaml
│   └── config_advanced.yaml  
├── data/                    # Data generation/processing
│   └── generate_data.py
├── models/                  # Model implementations
│   └── model.py
└── results/                 # Benchmark results
    └── benchmark_results.md
```

#### Code Quality
- **Documentation**: All functions should have docstrings
- **Reproducibility**: Include random seeds and exact dependencies
- **Dependencies**: Minimize and clearly specify requirements
- **Style**: Follow PEP 8 for Python, appropriate standards for other languages

#### Example Implementation Template
```python
"""
Title: [Phenomenon/Dataset/Model Name]
Description: [Brief description]
Author: [Your name/organization]
Date: [Date]
License: [License if different from repository]
"""

import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class DelayedGeneralizationModel:
    """
    Model implementation for studying delayed generalization
    
    Args:
        config (dict): Configuration parameters
        
    Example:
        >>> config = {"lr": 1e-3, "weight_decay": 1e-2}
        >>> model = DelayedGeneralizationModel(config)
        >>> results = model.train(train_data, test_data)
    """
    
    def __init__(self, config):
        self.config = config
        set_seed(config.get('seed', 42))
        
    def train(self, train_data, test_data):
        """
        Train model and return results showing delayed generalization
        
        Returns:
            dict: Training metrics including timing of generalization
        """
        # Implementation here
        pass
```

### 4. Experimental Standards

#### Reproducibility Requirements
- **Exact Dependencies**: Pin package versions in requirements.txt
- **Random Seeds**: Set and document all random seeds
- **Hardware Info**: Document GPU/CPU used for timing-sensitive results
- **Multiple Runs**: Report statistics across multiple seeds

#### Experimental Reporting
```python
# Standard experimental configuration
config = {
    "seed": 42,
    "model": {
        "architecture": "transformer",
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 128
    },
    "training": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 512,
        "max_epochs": 10000
    },
    "data": {
        "dataset": "modular_arithmetic",
        "prime": 97,
        "operation": "addition",
        "train_fraction": 0.5
    }
}

# Results reporting format
results = {
    "generalization_epoch": 3421,  # When test accuracy > threshold
    "final_train_accuracy": 1.0,
    "final_test_accuracy": 0.98,
    "runtime_hours": 2.5,
    "hardware": "NVIDIA RTX 3080"
}
```

### 5. Submission Process

#### Step 1: Fork and Branch
1. Fork the repository
2. Create a feature branch: `git checkout -b add-new-phenomenon`
3. Make your changes

#### Step 2: Documentation First
1. Start with comprehensive documentation
2. Follow the templates provided above
3. Include all necessary references

#### Step 3: Implementation
1. Add code following the standards above
2. Include examples and usage instructions
3. Test your code thoroughly

#### Step 4: Testing and Validation
1. Verify your contributions work as documented
2. Test on multiple environments if possible
3. Run any existing tests to ensure no breaking changes

#### Step 5: Pull Request
1. Commit with clear, descriptive messages
2. Open a pull request with:
   - Clear title and description
   - Summary of what you're adding
   - How it fits into the repository structure
   - Any testing you've performed

## ✅ Review Process

### What We Look For
- **Scientific Rigor**: Proper experimental methodology
- **Reproducibility**: Can others reproduce your results?
- **Documentation Quality**: Clear, comprehensive documentation
- **Fit with Repository**: Does it advance our understanding of delayed generalization?

### Review Timeline
- Initial review within 1 week
- Detailed feedback within 2 weeks
- Collaborative iteration to address feedback
- Merge once requirements are met

## 🏷️ Recognition

Contributors will be:
- Listed in the repository contributors
- Credited in any publications using the repository
- Acknowledged in release notes for significant contributions

## 📞 Questions?

- **Issues**: Use GitHub issues for questions about contributing
- **Discussions**: Use GitHub discussions for broader topics
- **Contact**: Reach out to repository maintainers for specific questions

## 📚 Resources

### Getting Started with Git
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

### Scientific Writing
- [Ten Simple Rules for Mathematical Writing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004961)
- [The Science of Scientific Writing](https://www.americanscientist.org/blog/the-long-view/the-science-of-scientific-writing)

### Machine Learning Best Practices
- [Papers with Code](https://paperswithcode.com/) - For finding relevant papers and code
- [Reproducible Research](https://www.coursera.org/learn/reproducible-research) - Best practices course

Thank you for contributing to advancing our understanding of delayed generalization in machine learning!