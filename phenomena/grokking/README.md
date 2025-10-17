# Grokking: Delayed Generalization After Memorization

## 📋 Overview

**Grokking** refers to the phenomenon where neural networks suddenly transition from memorization to generalization, often after thousands of training steps where the model has achieved perfect training accuracy but poor test performance.

## 🔬 Key Characteristics

- **Phase 1**: Rapid memorization of training data (high train accuracy, poor test accuracy)
- **Phase 2**: Extended plateau with perfect training but poor generalization  
- **Phase 3**: Sudden "aha moment" - rapid improvement in test accuracy
- **Phase 4**: Perfect or near-perfect generalization

## 📊 Original Discovery

**Paper**: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"  
**Authors**: Alethea Power, Yuri Burda, et al.  
**Year**: 2022  
**ArXiv**: https://arxiv.org/abs/2201.02177

### Key Findings
- Observed in small transformer models on algorithmic tasks
- Weight decay is crucial for triggering grokking
- Occurs across multiple mathematical operations
- Timeline varies dramatically based on hyperparameters

## 🧮 Common Grokking Scenarios

### 1. Modular Arithmetic
- **Tasks**: Addition, subtraction, multiplication modulo prime p
- **Typical Setup**: p=97 or p=113, train/test split
- **Model**: 1-2 layer transformers with attention
- **Timeline**: 1,000 - 10,000 epochs for generalization

### 2. Permutation Groups  
- **Tasks**: Group composition operations
- **Typical Setup**: Symmetric groups S_n
- **Model**: Transformer architectures
- **Timeline**: 5,000+ epochs typical

### 3. Polynomial Evaluation
- **Tasks**: Evaluating polynomials over finite fields
- **Model**: Small transformers
- **Observation**: Similar memorization → generalization pattern

## ⚙️ Critical Factors

### Hyperparameters
- **Weight Decay**: Essential for grokking (typical values: 1e-3 to 1e-1)
- **Learning Rate**: Lower rates often delay but improve grokking
- **Batch Size**: Smaller batches tend to promote grokking
- **Model Size**: Sweet spot exists - too small/large may prevent grokking

### Architecture Choices
- **Attention Layers**: Critical for algorithmic reasoning
- **Layer Normalization**: Standard placement typically used
- **Positional Encoding**: Important for sequence tasks

## 📈 Experimental Setups

### Standard Modular Addition Setup
```python
# Dataset parameters
prime = 97
train_fraction = 0.5  # 50% of all possible equations
operation = "addition"  # or "multiplication", "subtraction"

# Model parameters  
n_layers = 2
n_heads = 4
d_model = 128
d_ff = 512

# Training parameters
learning_rate = 1e-3
weight_decay = 1e-2  # Critical!
batch_size = 512
max_epochs = 10000
```

### Monitoring Metrics
- **Train Accuracy**: Should reach 100% quickly (< 1000 epochs)
- **Test Accuracy**: Key metric - watch for sudden jumps
- **Loss Curves**: Both train and test loss evolution
- **Weight Norms**: Often show interesting dynamics during grokking

## 🔧 Implementation Examples

### Datasets
- [`modular_arithmetic/`](./modular_arithmetic/) - Standard grokking datasets
- [`permutation_groups/`](./permutation_groups/) - Group theory tasks
- [`polynomial_evaluation/`](./polynomial_evaluation/) - Finite field operations

### Models
- [`simple_transformer.py`](./models/simple_transformer.py) - Basic grokking architecture
- [`attention_analysis.py`](./models/attention_analysis.py) - Attention pattern visualization

### Training Scripts
- [`train_modular.py`](./training/train_modular.py) - Modular arithmetic training
- [`hyperparameter_sweep.py`](./training/hyperparameter_sweep.py) - Parameter exploration

## 📚 Extended Research

### Theoretical Understanding
- **Circuit Formation**: Gradual development of algorithmic circuits
- **Feature Learning**: Transition from memorization to rule extraction
- **Double Descent**: Related to but distinct from double descent phenomenon

### Variations and Extensions
- **Multi-step Grokking**: Multiple phase transitions
- **Compositional Grokking**: Learning hierarchical rules
- **Scale Dependencies**: How model size affects grokking timeline

## 🎯 Reproduction Tips

1. **Start Simple**: Begin with modular addition, p=97
2. **Weight Decay is Key**: Don't skip regularization
3. **Be Patient**: Grokking can take 10,000+ epochs  
4. **Monitor Closely**: Use wandb or similar for long runs
5. **Try Multiple Seeds**: Grokking timing can vary significantly

## 📖 Related Phenomena

- **Lottery Ticket Hypothesis**: Sparse subnetworks that enable grokking
- **Phase Transitions**: Similar sudden improvements in other domains
- **Memorization vs Generalization**: Fundamental ML trade-offs

## 🔗 References

1. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
2. Liu et al. (2022). "Towards Understanding Grokking: An Effective Theory of Representation Learning"
3. Nanda et al. (2023). "Progress measures for grokking via mechanistic interpretability"
4. Varma et al. (2023). "Explaining grokking through circuit efficiency"