# Regularization Techniques for Delayed Generalization

## 📋 Overview

Regularization techniques play a crucial role in delayed generalization phenomena. Different regularization methods can either encourage or inhibit delayed generalization patterns, making them essential tools for controlling when and how models transition from memorization to generalization.

## 🔬 Impact on Delayed Generalization

### Weight Decay (L2 Regularization)
- **Critical for Grokking**: Often the difference between memorization and generalization
- **Mechanism**: Prevents weights from growing too large, encouraging simpler solutions
- **Timeline**: Can dramatically extend training time but enables generalization
- **Sweet Spot**: Usually 1e-3 to 1e-1 for grokking scenarios

### Dropout
- **Effect on Simplicity Bias**: Helps break reliance on spurious features
- **Mechanism**: Forces model to learn robust representations
- **Best Practice**: Often combined with other debiasing techniques
- **Timing**: Can delay initial memorization but improves final robustness

### Data Augmentation
- **Spurious Correlation Breaking**: Effective against background bias
- **Mechanism**: Increases data diversity, reduces memorization
- **Application**: Particularly useful for vision tasks with bias
- **Trade-off**: May slow initial learning but improves generalization

## 🛠️ Weight Decay Strategies

### 1. Standard L2 Weight Decay
```python
# PyTorch optimizers with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2  # Critical for grokking
)

# Or with SGD
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,  # Usually smaller for SGD
    momentum=0.9
)
```

**Grokking Guidelines**:
- Start with 1e-2, tune from 1e-3 to 1e-1
- Stronger weight decay = longer training but better generalization
- Essential for transformer architectures on algorithmic tasks

### 2. Adaptive Weight Decay
```python
class AdaptiveWeightDecay:
    def __init__(self, optimizer, initial_wd=1e-2, adaptation_rate=0.1):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.adaptation_rate = adaptation_rate
        
    def update_weight_decay(self, train_loss, val_loss):
        """Increase weight decay if overfitting detected"""
        overfitting_ratio = val_loss / train_loss
        if overfitting_ratio > 1.5:  # Significant overfitting
            new_wd = self.initial_wd * (1 + self.adaptation_rate)
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = new_wd
```

### 3. Layer-specific Weight Decay
```python
def create_weight_decay_groups(model, wd_embedding=1e-1, wd_attention=1e-2, wd_mlp=1e-3):
    """Different weight decay for different components"""
    embedding_params = []
    attention_params = []
    mlp_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embedding_params.append(param)
        elif 'attention' in name or 'attn' in name:
            attention_params.append(param)
        else:
            mlp_params.append(param)
    
    return [
        {'params': embedding_params, 'weight_decay': wd_embedding},
        {'params': attention_params, 'weight_decay': wd_attention},
        {'params': mlp_params, 'weight_decay': wd_mlp}
    ]
```

## 🎭 Dropout Strategies

### 1. Standard Dropout
```python
class GradualDropout(nn.Module):
    def __init__(self, max_dropout=0.5, warmup_epochs=50):
        super().__init__()
        self.max_dropout = max_dropout
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def forward(self, x):
        if self.training and self.current_epoch > 0:
            # Gradually increase dropout rate
            dropout_rate = min(
                self.max_dropout * self.current_epoch / self.warmup_epochs,
                self.max_dropout
            )
            return F.dropout(x, p=dropout_rate, training=True)
        return x
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
```

### 2. Attention Dropout
```python
class AttentionWithDropout(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply dropout to both attention weights and outputs
        attn_out, _ = self.attention(x, x, x)
        return self.dropout(attn_out)
```

### 3. Targeted Dropout for Bias Mitigation
```python
class BiasedFeatureDropout(nn.Module):
    """Dropout that specifically targets potentially biased features"""
    def __init__(self, feature_channels, bias_channels, bias_dropout=0.8):
        super().__init__()
        self.bias_channels = bias_channels
        self.bias_dropout = bias_dropout
        
    def forward(self, x):
        if self.training:
            # Higher dropout for bias-prone channels
            mask = torch.ones_like(x)
            mask[:, self.bias_channels] = torch.bernoulli(
                torch.full_like(mask[:, self.bias_channels], 1 - self.bias_dropout)
            )
            return x * mask
        return x
```

## 🖼️ Data Augmentation

### 1. Anti-Bias Augmentation
```python
class AntiBiasAugmentation:
    """Augmentation specifically designed to break spurious correlations"""
    
    def __init__(self, bias_type='background'):
        self.bias_type = bias_type
        
    def __call__(self, image, label):
        if self.bias_type == 'background':
            # Replace background with random textures
            return self.background_replacement(image, label)
        elif self.bias_type == 'color':
            # Randomize problematic color channels
            return self.color_randomization(image, label)
    
    def background_replacement(self, image, label):
        # Implementation for background bias (like Waterbirds)
        # Generate random background textures
        pass
    
    def color_randomization(self, image, label):
        # Implementation for color bias (like Colored MNIST)
        # Randomize color distributions
        pass
```

### 2. Progressive Augmentation
```python
class ProgressiveAugmentation:
    """Gradually increase augmentation strength during training"""
    
    def __init__(self, max_strength=1.0, warmup_epochs=100):
        self.max_strength = max_strength
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def get_augmentation_strength(self):
        return min(
            self.max_strength * self.current_epoch / self.warmup_epochs,
            self.max_strength
        )
    
    def __call__(self, image):
        strength = self.get_augmentation_strength()
        # Apply augmentations with current strength
        return self.apply_augmentations(image, strength)
```

## 📊 Experimental Configurations

### For Grokking (Modular Arithmetic)
```python
grokking_config = {
    # Weight decay is absolutely critical
    "weight_decay": 1e-2,  # Start here, tune 1e-3 to 1e-1
    "dropout": 0.0,  # Often not needed or can hurt
    "attention_dropout": 0.0,
    "optimizer": "AdamW",  # Better weight decay implementation
    
    # Advanced options
    "adaptive_weight_decay": False,  # Can help but adds complexity
    "layer_specific_wd": False,  # For very large models
}
```

### For Simplicity Bias (Waterbirds)
```python
waterbirds_config = {
    # Moderate weight decay
    "weight_decay": 1e-4,
    "dropout": 0.3,  # Helps with spurious features
    "data_augmentation": {
        "background_replacement": True,
        "color_jitter": True,
        "random_crop": True,
        "progressive": True,  # Start light, increase strength
    },
    
    # Bias-specific techniques
    "targeted_dropout": True,
    "bias_channels": [0, 1, 2],  # RGB channels for background bias
    "bias_dropout_rate": 0.5,
}
```

### For Large Models (Phase Transitions)
```python
large_model_config = {
    # Layer-specific weight decay
    "weight_decay_embedding": 1e-1,
    "weight_decay_attention": 1e-2, 
    "weight_decay_mlp": 1e-3,
    
    # Gradual dropout
    "dropout_schedule": "gradual",
    "max_dropout": 0.1,
    "dropout_warmup": 1000,  # Steps
    
    # Advanced regularization
    "gradient_clipping": 1.0,
    "spectral_normalization": False,  # For very large models
}
```

## 📈 Monitoring Regularization Effects

### Key Metrics
1. **Weight Norms**: Track magnitude of weights across layers
2. **Gradient Norms**: Monitor gradient flow and clipping frequency
3. **Effective Learning Rate**: Actual learning rate after weight decay
4. **Generalization Gap**: Train vs test performance over time

### Visualization Tools
```python
def plot_weight_norms(model, epoch_weights, save_path=None):
    """Plot evolution of weight norms across training"""
    import matplotlib.pyplot as plt
    
    epochs = list(epoch_weights.keys())
    layer_norms = {name: [] for name in epoch_weights[epochs[0]].keys()}
    
    for epoch in epochs:
        for name, norm in epoch_weights[epoch].items():
            layer_norms[name].append(norm)
    
    plt.figure(figsize=(12, 8))
    for name, norms in layer_norms.items():
        plt.plot(epochs, norms, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Weight Norm')
    plt.title('Weight Norm Evolution During Training')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def track_regularization_metrics(model, train_loss, val_loss, epoch):
    """Track key regularization metrics"""
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'generalization_gap': val_loss - train_loss,
        'weight_norms': {}
    }
    
    # Calculate weight norms for each layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            metrics['weight_norms'][name] = param.data.norm().item()
    
    return metrics
```

## 🎯 Best Practices

### Weight Decay Guidelines
1. **Start Strong**: Begin with 1e-2 for grokking, tune downward if needed
2. **Monitor Closely**: Track when generalization begins vs memorization
3. **Be Patient**: Strong weight decay extends training time significantly
4. **Combine Wisely**: Works well with warmup and proper scheduling

### Dropout Guidelines  
1. **Task-Specific**: Higher dropout for bias-prone tasks
2. **Gradual Introduction**: Can help with training stability
3. **Layer-Specific**: Different dropout rates for different components
4. **Monitor Impact**: Too much dropout can prevent any learning

### Data Augmentation Guidelines
1. **Target the Bias**: Design augmentations to break specific correlations
2. **Progressive Strength**: Start light, increase during training
3. **Balance Carefully**: Too much can prevent learning the task
4. **Combine with Other Methods**: Most effective as part of broader strategy

## 🧪 Advanced Techniques

### Spectral Regularization
```python
def spectral_norm_conv(module, name='weight', n_power_iterations=1):
    """Apply spectral normalization to convolutional layers"""
    return torch.nn.utils.spectral_norm(module, name, n_power_iterations)

class SpectralRegularizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = self.apply_spectral_norm(base_model)
    
    def apply_spectral_norm(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                spectral_norm_conv(module)
        return model
```

### Information Bottleneck Regularization
```python
class InformationBottleneck(nn.Module):
    """Regularization based on information bottleneck principle"""
    
    def __init__(self, beta=1e-3):
        super().__init__()
        self.beta = beta
        
    def forward(self, representations, targets):
        # Encourage minimal information while preserving task performance
        mi_loss = self.mutual_information_loss(representations, targets)
        return self.beta * mi_loss
    
    def mutual_information_loss(self, representations, targets):
        # Simplified MI estimation
        # In practice, use more sophisticated estimators
        return torch.var(representations)
```

## 🔗 References

1. Krogh & Hertz (1992). "A Simple Weight Decay Can Improve Generalization"
2. Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
3. Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization"
4. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
5. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
6. Loshchilov & Hutter (2017). "Decoupled Weight Decay Regularization"