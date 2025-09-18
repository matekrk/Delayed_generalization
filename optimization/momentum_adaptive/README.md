# Momentum and Adaptive Optimizers for Delayed Generalization

## 📋 Overview

The choice of optimizer and its momentum/adaptation characteristics significantly influence delayed generalization patterns. Different optimizers can accelerate or delay the transition from memorization to generalization, affect the sharpness of phase transitions, and determine the final quality of learned representations.

## 🔬 Optimizer Impact on Delayed Generalization

### SGD with Momentum
- **Grokking**: Often slower to start but can lead to more robust generalization
- **Simplicity Bias**: Better at avoiding spurious correlations than adaptive methods
- **Phase Transitions**: Creates sharper, more distinct transitions
- **Timeline**: May require longer training but achieves better final performance

### Adam/AdamW
- **Grokking**: Faster initial learning but can get stuck in memorization
- **Weight Decay**: AdamW's decoupled weight decay crucial for grokking
- **Phase Transitions**: Smoother transitions, sometimes less dramatic improvements
- **Timeline**: Faster initial progress but may plateau earlier

### Adaptive Methods (RMSprop, Adagrad)
- **Memory Effects**: Can remember spurious correlations too well
- **Generalization**: Often achieve lower final test performance
- **Best Use**: Initial exploration, then switch to SGD for final training

## 🛠️ SGD Variants and Momentum Strategies

### 1. Classical SGD with Momentum
```python
class MomentumSGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Update velocity
            self.velocity[i].mul_(self.momentum).add_(grad, alpha=-self.lr)
            
            # Update parameters
            param.data.add_(self.velocity[i])
```

**Grokking Configuration**:
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-3,  # Lighter than AdamW
    nesterov=True  # Often helps with grokking
)
```

### 2. Nesterov Accelerated Gradient
```python
class NesterovSGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Nesterov update
            self.velocity[i].mul_(self.momentum).add_(grad, alpha=-self.lr)
            
            # Look ahead update
            param.data.add_(grad, alpha=-self.lr).add_(self.velocity[i], alpha=self.momentum)
```

**Benefits for Delayed Generalization**:
- Better exploration of loss landscape
- Can escape shallow local minima
- Often leads to sharper phase transitions

### 3. Adaptive Momentum
```python
class AdaptiveMomentumSGD:
    """SGD with momentum that adapts to training phase"""
    
    def __init__(self, params, lr=0.01, momentum_range=(0.5, 0.95), 
                 adaptation_window=100):
        self.params = list(params)
        self.lr = lr
        self.min_momentum, self.max_momentum = momentum_range
        self.adaptation_window = adaptation_window
        
        self.velocity = [torch.zeros_like(p) for p in self.params]
        self.loss_history = []
        self.current_momentum = self.min_momentum
        
    def update_momentum(self, current_loss):
        """Adapt momentum based on training dynamics"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) > self.adaptation_window:
            self.loss_history = self.loss_history[-self.adaptation_window:]
            
            # Calculate loss stability (low variance = stable training)
            recent_losses = self.loss_history[-20:]
            if len(recent_losses) >= 20:
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                stability = 1.0 / (1.0 + loss_std / loss_mean)
                
                # Higher stability -> higher momentum
                self.current_momentum = self.min_momentum + \
                    (self.max_momentum - self.min_momentum) * stability
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Update velocity with current momentum
            self.velocity[i].mul_(self.current_momentum).add_(grad, alpha=-self.lr)
            
            # Update parameters
            param.data.add_(self.velocity[i])
```

## 🧠 Adaptive Optimizers

### 1. AdamW with Proper Weight Decay
```python
class AdamWForGrokking:
    """AdamW variant optimized for grokking phenomenon"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # State initialization
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        if amsgrad:
            self.v_max = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Update biased second raw moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            if self.amsgrad:
                # Maintain max of second moment
                torch.max(self.v_max[i], self.v[i], out=self.v_max[i])
                denom = (self.v_max[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            else:
                denom = (self.v[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            
            step_size = self.lr / bias_correction1
            
            # Weight decay (decoupled)
            param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Apply update
            param.data.addcdiv_(self.m[i], denom, value=-step_size)
```

**Grokking-Specific Configuration**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),  # Lower beta2 for grokking
    eps=1e-8,
    weight_decay=1e-2,  # Strong weight decay crucial
    amsgrad=False  # Usually not needed
)
```

### 2. AdaBound: Adaptive to SGD Transition
```python
class AdaBound:
    """Adaptive optimizer that transitions to SGD"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), 
                 final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.final_lr = final_lr
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            if self.weight_decay > 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Update moments
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            # Compute bounds
            final_lr = self.final_lr * self.lr / self.lr  # Relative to base LR
            lower_bound = final_lr * (1 - 1 / (self.gamma * self.step_count + 1))
            upper_bound = final_lr * (1 + 1 / (self.gamma * self.step_count))
            
            # Compute step size
            step_size = self.lr / bias_correction1
            v_corrected = self.v[i] / bias_correction2
            step_size = step_size / (v_corrected.sqrt().add_(self.eps))
            
            # Apply bounds
            step_size = torch.clamp(step_size, lower_bound, upper_bound)
            
            # Update parameters
            param.data.add_(self.m[i], alpha=-step_size)
```

### 3. Lookahead Optimizer Wrapper
```python
class Lookahead:
    """Lookahead wrapper that works with any base optimizer"""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = []
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights.append(p.data.clone())
                
    def step(self):
        self.base_optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Lookahead step
            param_idx = 0
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Interpolate between fast and slow weights
                    self.slow_weights[param_idx].add_(
                        p.data - self.slow_weights[param_idx], 
                        alpha=self.alpha
                    )
                    
                    # Update fast weights to slow weights
                    p.data.copy_(self.slow_weights[param_idx])
                    param_idx += 1

# Usage with delayed generalization
base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

## 📊 Optimizer Selection Guidelines

### For Grokking Phenomena
```python
def get_grokking_optimizer(model, optimizer_type='adamw'):
    """Get optimizer configuration for grokking"""
    
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),  # Lower beta2
            weight_decay=1e-2,  # Strong weight decay
            eps=1e-8
        )
    
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-3,  # Lighter for SGD
            nesterov=True
        )
    
    elif optimizer_type == 'adabound':
        return AdaBound(
            model.parameters(),
            lr=1e-3,
            final_lr=0.1,
            gamma=1e-3,
            weight_decay=1e-2
        )
```

### For Simplicity Bias Mitigation
```python
def get_bias_mitigation_optimizer(model, method='group_dro'):
    """Optimizer configurations for bias mitigation"""
    
    if method == 'erm':
        # Standard empirical risk minimization
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-4
        )
    
    elif method == 'group_dro':
        # Group distributionally robust optimization
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-3,  # Lower LR for stability
            momentum=0.9,
            weight_decay=1e-4
        )
    
    elif method == 'irm':
        # Invariant risk minimization
        return torch.optim.Adam(
            model.parameters(),
            lr=1e-4,  # Very conservative
            weight_decay=1e-4
        )
```

### For Phase Transitions
```python
def get_phase_transition_optimizer(model, model_size='small'):
    """Optimizers that work well with phase transitions"""
    
    if model_size == 'small':
        # Small models: SGD with momentum
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
    
    elif model_size == 'medium':
        # Medium models: AdamW
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-3
        )
    
    elif model_size == 'large':
        # Large models: AdamW with Lookahead
        base_opt = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )
        return Lookahead(base_opt, k=5, alpha=0.5)
```

## 📈 Advanced Optimizer Techniques

### 1. Progressive Resizing with Optimizer Reset
```python
class ProgressiveOptimizer:
    """Optimizer that resets during progressive training phases"""
    
    def __init__(self, model, optimizer_configs):
        self.model = model
        self.optimizer_configs = optimizer_configs
        self.current_phase = 0
        self.current_optimizer = None
        self.switch_to_phase(0)
        
    def switch_to_phase(self, phase):
        """Switch to a different optimizer configuration"""
        if phase < len(self.optimizer_configs):
            config = self.optimizer_configs[phase]
            self.current_optimizer = config['optimizer_class'](
                self.model.parameters(),
                **config['params']
            )
            self.current_phase = phase
            print(f"Switched to phase {phase}: {config['name']}")
    
    def step(self):
        self.current_optimizer.step()
    
    def zero_grad(self):
        self.current_optimizer.zero_grad()

# Example usage for grokking
grokking_phases = [
    {
        'name': 'Exploration',
        'optimizer_class': torch.optim.Adam,
        'params': {'lr': 1e-3, 'weight_decay': 1e-4},
        'epochs': 2000
    },
    {
        'name': 'Consolidation',
        'optimizer_class': torch.optim.AdamW,
        'params': {'lr': 1e-3, 'weight_decay': 1e-2},
        'epochs': 5000
    },
    {
        'name': 'Fine-tuning',
        'optimizer_class': torch.optim.SGD,
        'params': {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 1e-3},
        'epochs': 3000
    }
]
```

### 2. Gradient Centralization
```python
class GradientCentralization:
    """Wrapper to add gradient centralization to any optimizer"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def centralize_gradient(self, grad):
        """Centralize gradient by subtracting its mean"""
        if len(grad.shape) > 1:
            # For conv and linear layers
            grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
        return grad
    
    def step(self):
        # Apply gradient centralization before optimizer step
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data = self.centralize_gradient(p.grad.data)
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# Usage
base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = GradientCentralization(base_optimizer)
```

### 3. Warm Restarts with Optimizer State Reset
```python
class OptimizerWithWarmRestarts:
    """Optimizer with periodic warm restarts and state reset"""
    
    def __init__(self, model, optimizer_class, optimizer_params, 
                 restart_period=1000, restart_mult=2):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        
        self.current_optimizer = optimizer_class(model.parameters(), **optimizer_params)
        self.step_count = 0
        self.restart_count = 0
        
    def step(self):
        self.current_optimizer.step()
        self.step_count += 1
        
        # Check for restart
        if self.step_count >= self.restart_period:
            print(f"Warm restart #{self.restart_count + 1}")
            
            # Reset optimizer state
            self.current_optimizer = self.optimizer_class(
                self.model.parameters(), 
                **self.optimizer_params
            )
            
            # Update restart period
            self.restart_period = int(self.restart_period * self.restart_mult)
            self.step_count = 0
            self.restart_count += 1
    
    def zero_grad(self):
        self.current_optimizer.zero_grad()
```

## 🎯 Best Practices

### General Guidelines
1. **Start Conservative**: Begin with well-established configurations
2. **Monitor Carefully**: Track both training dynamics and final performance
3. **Be Patient**: Good optimizers for delayed generalization may look worse initially
4. **Combine Techniques**: Use multiple strategies (warmup, scheduling, etc.)

### Hyperparameter Tuning
1. **Learning Rate**: Most critical hyperparameter, tune first
2. **Weight Decay**: Especially important for grokking
3. **Momentum**: Higher values for stable training phases
4. **Betas (for Adam)**: Lower beta2 often better for grokking

### Common Mistakes
1. **Too High Learning Rate**: Prevents careful feature development
2. **Insufficient Weight Decay**: Allows memorization without generalization
3. **Wrong Optimizer for Task**: Adam for everything isn't always best
4. **Ignoring Momentum**: Fixed momentum when adaptive would help

## 🔗 References

1. Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
2. Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
3. Zhang et al. (2019). "Lookahead Optimizer: k steps forward, 1 step back"
4. Luo et al. (2019). "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
5. Yong et al. (2020). "Gradient Centralization: A New Optimization Technique"
6. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"