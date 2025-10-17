# Catalog of Delayed Generalization Scenarios

This catalog provides a comprehensive overview of all documented scenarios where delayed generalization occurs in machine learning.

## 📊 Quick Reference Table

| Phenomenon | Dataset | Model | Key Metric | Delay Period | Reference |
|------------|---------|-------|------------|--------------|-----------|
| Grokking | Modular Addition | Transformer | Validation Accuracy | 1000-10000 epochs | [Power et al., 2022](https://arxiv.org/abs/2201.02177) |
| Grokking | Permutation Groups | Transformer | Test Accuracy | 5000+ epochs | [Power et al., 2022](https://arxiv.org/abs/2201.02177) |
| Simplicity Bias | Waterbirds | ResNet | Worst-group Accuracy | Variable | [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731) |
| Simplicity Bias | ColoredMNIST | CNN/MLP | Test Accuracy | 100-500 epochs | [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893) |
| Phase Transition | Language Modeling | GPT-style | Emergent Abilities | Scale-dependent | Various |

## 🔍 Detailed Scenarios

### Grokking Phenomena

#### 1. Modular Arithmetic
- **Dataset**: Modular addition/multiplication (p=97, p=113)
- **Model**: Small transformers (1-2 layers)
- **Observation**: Perfect memorization → sudden generalization
- **Timeline**: Generalization after 1000-10000 epochs
- **Key Factors**: Weight decay, learning rate
- **Location**: [`phenomena/grokking/modular_arithmetic/`](./phenomena/grokking/modular_arithmetic/)

#### 2. Permutation Groups
- **Dataset**: Symmetric group operations
- **Model**: Transformer architectures
- **Observation**: Extended memorization phase before generalization
- **Timeline**: 5000+ epochs typical
- **Location**: [`phenomena/grokking/permutation_groups/`](./phenomena/grokking/permutation_groups/)

### Simplicity Bias

#### 1. Waterbirds Dataset
- **Dataset**: Birds on land vs. water backgrounds
- **Model**: ResNet, DenseNet
- **Observation**: Models learn background correlation before bird features
- **Mitigation**: Group DRO, domain adaptation techniques
- **Location**: [`phenomena/simplicity_bias/waterbirds/`](./phenomena/simplicity_bias/waterbirds/)

#### 2. Colored MNIST
- **Dataset**: MNIST digits with color bias
- **Model**: CNN, MLP
- **Observation**: Color learned before shape
- **Timeline**: Shape generalization after 100-500 epochs
- **Location**: [`phenomena/simplicity_bias/colored_mnist/`](./phenomena/simplicity_bias/colored_mnist/)

### Phase Transitions

#### 1. Emergent Abilities in Language Models
- **Dataset**: Various language tasks
- **Model**: GPT-family models
- **Observation**: Sudden capability emergence at scale
- **Factors**: Model size, training data, task complexity
- **Location**: [`phenomena/phase_transitions/emergent_abilities/`](./phenomena/phase_transitions/emergent_abilities/)

## 📈 Contributing New Scenarios

To add a new scenario to this catalog:

1. Create detailed documentation in the appropriate category
2. Add entry to the Quick Reference Table above
3. Include experimental setup and reproducible results
4. Follow the [contribution guidelines](./CONTRIBUTING.md)

## 🏷️ Tags and Filters

### By Phenomenon Type
- `grokking`: Sudden generalization after memorization
- `simplicity-bias`: Learning spurious before true features  
- `phase-transition`: Sharp transitions in capabilities
- `scaling`: Scale-dependent emergence

### By Domain
- `vision`: Computer vision tasks
- `nlp`: Natural language processing
- `algorithmic`: Mathematical/algorithmic tasks
- `tabular`: Structured data

### By Model Type
- `transformer`: Attention-based models
- `cnn`: Convolutional networks
- `mlp`: Multi-layer perceptrons
- `rnn`: Recurrent networks

### By Intervention
- `weight-decay`: Regularization effects
- `learning-rate`: Schedule impacts
- `architecture`: Model design choices
- `data-augmentation`: Training data modifications