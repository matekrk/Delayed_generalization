# Waterbirds Dataset: Simplicity Bias in Action

## 📋 Overview

The Waterbirds dataset is a seminal example of **simplicity bias** in machine learning, where models learn spurious correlations (background features) before learning the true predictive features (bird characteristics). This leads to delayed generalization where models appear to perform well initially but fail to generalize robustly.

## 🐦 Dataset Description

### Task
Binary classification: Distinguish between **waterbirds** and **landbirds**

### Spurious Correlation
- **Waterbirds** are predominantly photographed against **water backgrounds**  
- **Landbirds** are predominantly photographed against **land backgrounds**
- Models tend to learn background features instead of bird characteristics

### Ground Truth Features
True predictive features include:
- Bill shape and size (longer, flatter bills for waterbirds)
- Leg length and webbing (webbed feet for waterbirds)
- Body proportions and posture
- Feather patterns and coloration

## 📊 Dataset Statistics

### Original CUB-200-2011 + Places Backgrounds
- **Total Images**: ~12,000 images
- **Classes**: 2 (waterbirds vs landbirds from 200 CUB species)
- **Backgrounds**: Water vs land environments from Places dataset

### Group Breakdown
| Group | Description | Typical Size | Performance Impact |
|-------|-------------|--------------|-------------------|
| Waterbirds on Water | Majority waterbird group | ~45% | High accuracy |
| Landbirds on Land | Majority landbird group | ~45% | High accuracy |
| Waterbirds on Land | Minority waterbird group | ~5% | Poor accuracy |  
| Landbirds on Water | Minority landbird group | ~5% | Poor accuracy |

## 🔍 Delayed Generalization Pattern

### Phase 1: Rapid Background Learning (Epochs 1-20)
- Model quickly learns to associate water backgrounds with waterbirds
- High average accuracy achieved rapidly (>85%)
- Poor worst-group accuracy (<60%)

### Phase 2: Background Overfitting (Epochs 20-100)  
- Model becomes increasingly confident in background features
- Average accuracy plateaus or slightly improves
- Worst-group accuracy may actually decrease
- Large gap between majority and minority group performance

### Phase 3: Feature Competition (Epochs 100-200)
- Without intervention: Model remains stuck on background features
- With techniques like Group DRO: Gradual learning of bird features begins
- Slow improvement in worst-group accuracy

### Phase 4: True Feature Learning (Epochs 200+)
- With proper techniques: Models learn robust bird characteristics
- Worst-group accuracy improves significantly  
- More balanced performance across all groups

## ⚙️ Experimental Setup

### Standard Training Configuration
```python
# Model
model = "ResNet-50"  # Pretrained on ImageNet
freeze_backbone = False  # Fine-tune all layers

# Training  
optimizer = "SGD"
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-4
batch_size = 128
epochs = 300

# Data
image_size = 224
augmentation = "standard"  # Random crops, flips
```

### Group Distributionally Robust Optimization (Group DRO)
```python
# Instead of minimizing average loss, minimize worst-group loss
loss_type = "group_dro"
group_weights_step_size = 0.01  # How fast to upweight worst groups
adjust_epochs = 1  # How often to reweight groups

# This encourages learning features that work for ALL groups
```

### Evaluation Metrics
- **Average Accuracy**: Standard metric across all examples
- **Worst-Group Accuracy**: Minimum accuracy across the 4 groups  
- **Group-wise Breakdown**: Individual accuracy for each group
- **Robust Accuracy**: Harmonic mean of group accuracies

## 📈 Typical Results Timeline

### Baseline ERM (Empirical Risk Minimization)
```
Epoch   | Avg Acc | Worst-Group | Water/Water | Land/Land | Water/Land | Land/Water
--------|---------|-------------|-------------|-----------|------------|----------
10      | 87.2%   | 52.1%      | 95.3%       | 94.8%     | 52.1%      | 58.7%
50      | 91.4%   | 47.8%      | 97.1%       | 96.9%     | 47.8%      | 51.2%  
100     | 92.1%   | 45.6%      | 97.8%       | 97.4%     | 45.6%      | 48.9%
200     | 92.3%   | 44.2%      | 98.1%       | 97.6%     | 44.2%      | 47.3%
```

### With Group DRO
```
Epoch   | Avg Acc | Worst-Group | Water/Water | Land/Land | Water/Land | Land/Water  
--------|---------|-------------|-------------|-----------|------------|----------
10      | 82.1%   | 61.4%      | 89.2%       | 88.7%     | 61.4%      | 65.8%
50      | 84.7%   | 69.2%      | 90.1%       | 89.8%     | 69.2%      | 72.1%
100     | 86.9%   | 75.8%      | 91.3%       | 90.7%     | 75.8%      | 78.4%
200     | 88.1%   | 82.3%      | 92.1%       | 91.6%     | 82.3%      | 84.7%
```

## 🛠️ Implementation Resources

### Data Preparation
```python
# Download CUB-200-2011 dataset
# Download Places365 dataset  
# Create waterbird/landbird labels from CUB species
# Composite bird images with background images
# Create group labels for each example
```

### Model Training
```python
import torch
import torch.nn as nn
from torchvision import models, transforms

class WaterbirdsClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, 2)  # Binary classification
        
    def forward(self, x):
        return self.backbone(x)

# Group DRO loss function
def group_dro_loss(outputs, targets, groups, group_weights):
    """Compute group-wise losses and weighted combination"""
    group_losses = []
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            group_loss = F.cross_entropy(outputs[mask], targets[mask])
            group_losses.append(group_loss)
    
    return torch.stack(group_losses) @ group_weights
```

### Evaluation and Analysis
```python
def evaluate_groups(model, dataloader):
    """Evaluate model performance per group"""
    group_correct = torch.zeros(4)
    group_total = torch.zeros(4)
    
    with torch.no_grad():
        for inputs, targets, groups in dataloader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            for g in range(4):
                mask = (groups == g)
                group_correct[g] += (preds[mask] == targets[mask]).sum()
                group_total[g] += mask.sum()
    
    group_accuracies = group_correct / group_total
    return {
        "worst_group_acc": group_accuracies.min().item(),
        "avg_acc": group_accuracies.mean().item(),
        "group_accs": group_accuracies.tolist()
    }
```

## 🎯 Key Research Questions

1. **Timing**: How long does it take for models to transition from background to bird features?

2. **Interventions**: Which techniques most effectively accelerate robust feature learning?

3. **Architecture**: Do different architectures show different susceptibility to simplicity bias?

4. **Data**: How does the strength of spurious correlation affect the delay timeline?

5. **Scaling**: Do larger models overcome simplicity bias faster or slower?

## 🔗 Extensions and Variations

### Related Datasets
- **CelebA**: Gender classification with background/makeup spurious correlations
- **MNIST-C**: Color vs shape in digit classification
- **CIFAR-10-C**: Natural vs corrupted features
- **MultiNLI**: Lexical vs semantic reasoning in NLP

### Experimental Variations
- **Correlation Strength**: Vary the percentage of spurious correlation
- **Group Balance**: Change relative sizes of majority/minority groups
- **Feature Complexity**: Modify difficulty of true vs spurious features
- **Multi-step Bias**: Hierarchical spurious correlations

## 📚 References

1. **Original Paper**: Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts: Hardening the Accuracy vs. Robustness Trade-off"

2. **Group DRO**: Sagawa et al. (2019). "Distributionally Robust Neural Networks"

3. **Simplicity Bias**: Shah et al. (2020). "The Pitfalls of Simplicity Bias in Neural Networks"

4. **Background**: Beery et al. (2018). "Recognition in Terra Incognita" 

5. **Analysis**: Kirichenko et al. (2022). "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations"