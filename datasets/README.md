# Datasets for Delayed Generalization Research

This directory contains documentation and resources for datasets commonly used to study delayed generalization phenomena.

## 📊 Dataset Categories

### [Algorithmic](./algorithmic/)
Mathematical and algorithmic tasks where delayed generalization patterns are well-studied
- **Modular Arithmetic**: Addition, multiplication, division modulo prime numbers
- **Group Operations**: Permutation groups, symmetry operations
- **Sequence Tasks**: Copying, reversal, sorting algorithms
- **Graph Problems**: Shortest path, connectivity, coloring

### [Vision](./vision/)  
Computer vision datasets exhibiting delayed generalization phenomena
- **Waterbirds**: Background vs bird type classification
- **Colored MNIST**: Color vs shape learning
- **CelebA**: Demographic bias in attribute prediction
- **CIFAR-10-C**: Corruption robustness vs content learning

### [NLP](./nlp/)
Natural language processing tasks with delayed generalization patterns
- **Sentiment Analysis**: Surface vs deep linguistic features
- **Natural Language Inference**: Lexical vs semantic reasoning  
- **Question Answering**: Memorization vs understanding
- **Language Modeling**: Token vs structural learning

## 🎯 Selection Criteria

When choosing datasets for delayed generalization research, consider:

### Clear Delayed Patterns
- Observable gap between train and test performance
- Identifiable transition points or phase changes
- Reproducible timing of generalization onset

### Controlled Variables
- Known spurious correlations or biases
- Tunable difficulty or complexity
- Well-understood underlying patterns

### Research Community Adoption
- Standard benchmarks for comparison
- Existing baselines and reproduction guidelines
- Active research community

## 📈 Common Experimental Setups

### Train/Test Splits
- **Algorithmic**: Often 50% train, 50% test from all possible examples
- **Vision**: Standard splits with careful attention to group balance
- **NLP**: Domain shifts, compositional splits, temporal splits

### Evaluation Metrics
- **Accuracy**: Both average and group-specific
- **Loss Curves**: Training vs validation trajectories  
- **Timing**: Epochs to generalization onset
- **Robustness**: Performance on distribution shifts

### Monitoring Protocols
- **Frequent Evaluation**: Every epoch or multiple times per epoch
- **Long Training**: Patience for delayed generalization (1000+ epochs)
- **Multiple Seeds**: Account for timing variability
- **Detailed Logging**: Capture transition dynamics

## 🔗 Quick Reference

| Dataset | Domain | Phenomenon | Train/Test Split | Typical Timeline |
|---------|--------|------------|------------------|------------------|
| Modular Addition | Algorithmic | Grokking | 50/50 | 1000-10000 epochs |
| Waterbirds | Vision | Simplicity Bias | Standard + group | 100-300 epochs |
| Colored MNIST | Vision | Simplicity Bias | Custom color split | 50-500 epochs |
| Permutation Groups | Algorithmic | Grokking | 50/50 | 5000+ epochs |
| CelebA (biased) | Vision | Spurious Correlation | Attribute-specific | 50-200 epochs |

See individual category directories for detailed dataset documentation, download instructions, and experimental protocols.