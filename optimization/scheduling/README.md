# Learning Rate Scheduling for Delayed Generalization

## 📋 Overview

Learning rate scheduling plays a pivotal role in delayed generalization phenomena. The right schedule can trigger phase transitions, smooth training dynamics, or extend the time to generalization. Different scheduling strategies can dramatically affect when and how models transition from memorization to true understanding.

## 🔬 Impact on Delayed Generalization

### Cosine Annealing
- **Effect**: Creates smooth transitions that often improve generalization
- **Mechanism**: Gradual learning rate reduction allows fine-tuning of features
- **Timeline**: Can delay generalization but improves final performance
- **Best for**: Grokking, phase transitions, long training runs

### Step Decay
- **Effect**: Can trigger sudden improvements in generalization
- **Mechanism**: Discrete LR drops force the model to find new minima
- **Timeline**: Often causes sudden jumps in test performance
- **Best for**: When you know approximate timing of phase transitions

### Cyclic Learning Rates
- **Effect**: Multiple opportunities for phase transitions
- **Mechanism**: Cycling allows escaping local minima repeatedly
- **Timeline**: Can accelerate discovery of generalizable solutions
- **Best for**: Difficult optimization landscapes

## 🛠️ Core Scheduling Strategies

### 1. Cosine Annealing
```python
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

class CosineAnnealingScheduler:
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_count = 0
        
    def step(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.get_base_lr() * self.step_count / self.warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (self.step_count - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.eta_min + (self.get_base_lr() - self.eta_min) * \
                 (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1
    
    def get_base_lr(self):
        return self.optimizer.param_groups[0]['lr']
```

**Use Cases**:
- **Grokking**: Excellent for long training runs (10k+ epochs)
- **Phase Transitions**: Smooth transitions reduce instability
- **Fine-tuning**: Good for final performance optimization

### 2. Step Decay with Adaptive Timing
```python
class AdaptiveStepDecay:
    def __init__(self, optimizer, decay_factor=0.1, patience=1000, 
                 min_improvement=1e-4, cooldown=500):
        self.optimizer = optimizer
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_improvement = min_improvement
        self.cooldown = cooldown
        
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        
    def step(self, val_loss):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        if val_loss < self.best_loss - self.min_improvement:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # Decay learning rate
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * self.decay_factor
                print(f"Reducing LR: {old_lr:.6f} -> {param_group['lr']:.6f}")
            
            self.wait = 0
            self.cooldown_counter = self.cooldown
```

### 3. Cyclic Learning Rates
```python
class CyclicLR:
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, 
                 mode='triangular', gamma=1.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up  # Symmetric cycles
        self.mode = mode
        self.gamma = gamma
        self.step_count = 0
        
    def step(self):
        cycle = math.floor(1 + self.step_count / (self.step_size_up + self.step_size_down))
        x = abs(self.step_count / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.step_count
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * \
             max(0, (1 - x)) * scale_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1
```

### 4. Exponential Decay with Restarts
```python
class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        
    def step(self):
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            # Restart
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            self.restart_count += 1
        
        # Cosine annealing within current restart
        lr = self.eta_min + (self.get_base_lr() - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_base_lr(self):
        return self.optimizer.param_groups[0]['lr']
```

## 📊 Scheduling for Specific Phenomena

### Grokking-Optimized Schedules
```python
def create_grokking_scheduler(optimizer, total_epochs=10000):
    """Schedule optimized for grokking phenomenon"""
    return {
        "warmup": {
            "type": "linear",
            "steps": int(0.1 * total_epochs),  # 10% warmup
            "target_lr": 1e-3
        },
        "main": {
            "type": "cosine",
            "T_max": int(0.8 * total_epochs),  # 80% main training
            "eta_min": 1e-6
        },
        "final": {
            "type": "constant",
            "steps": int(0.1 * total_epochs),  # 10% at min LR
            "lr": 1e-6
        }
    }

# Implementation
class GrokkingScheduler:
    def __init__(self, optimizer, total_epochs=10000):
        self.config = create_grokking_scheduler(optimizer, total_epochs)
        self.optimizer = optimizer
        self.epoch = 0
        self.total_epochs = total_epochs
        
    def step(self):
        progress = self.epoch / self.total_epochs
        
        if progress < 0.1:  # Warmup phase
            lr = self.config["warmup"]["target_lr"] * (progress / 0.1)
        elif progress < 0.9:  # Main training
            adjusted_progress = (progress - 0.1) / 0.8
            lr = self.config["main"]["eta_min"] + \
                (self.config["warmup"]["target_lr"] - self.config["main"]["eta_min"]) * \
                (1 + math.cos(math.pi * adjusted_progress)) / 2
        else:  # Final phase
            lr = self.config["final"]["lr"]
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
```

### Simplicity Bias Schedules
```python
def create_bias_mitigation_scheduler(optimizer, total_epochs=300):
    """Schedule designed to combat simplicity bias"""
    return {
        "aggressive_start": {
            "epochs": int(0.2 * total_epochs),  # 20% aggressive learning
            "lr": 1e-2,
            "description": "Learn core features quickly"
        },
        "plateau": {
            "epochs": int(0.3 * total_epochs),  # 30% plateau
            "lr": 1e-3,
            "description": "Allow robust feature development"
        },
        "fine_tune": {
            "epochs": int(0.5 * total_epochs),  # 50% fine-tuning
            "type": "cosine",
            "start_lr": 1e-3,
            "end_lr": 1e-5,
            "description": "Fine-tune without spurious features"
        }
    }

class BiasScheduler:
    def __init__(self, optimizer, total_epochs=300):
        self.config = create_bias_mitigation_scheduler(optimizer, total_epochs)
        self.optimizer = optimizer
        self.epoch = 0
        self.total_epochs = total_epochs
        
    def step(self):
        progress = self.epoch / self.total_epochs
        
        if progress < 0.2:
            lr = self.config["aggressive_start"]["lr"]
        elif progress < 0.5:
            lr = self.config["plateau"]["lr"]
        else:
            # Cosine decay for fine-tuning
            adjusted_progress = (progress - 0.5) / 0.5
            start_lr = self.config["fine_tune"]["start_lr"]
            end_lr = self.config["fine_tune"]["end_lr"]
            lr = end_lr + (start_lr - end_lr) * \
                (1 + math.cos(math.pi * adjusted_progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
```

### Phase Transition Schedules
```python
class PhaseTransitionScheduler:
    """Schedule that adapts to detected phase transitions"""
    
    def __init__(self, optimizer, detection_window=100, transition_lr_boost=2.0):
        self.optimizer = optimizer
        self.detection_window = detection_window
        self.transition_lr_boost = transition_lr_boost
        
        self.loss_history = []
        self.base_lr = optimizer.param_groups[0]['lr']
        self.in_transition = False
        self.transition_timer = 0
        
    def detect_phase_transition(self, loss):
        """Detect if we're in a phase transition based on loss dynamics"""
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.detection_window:
            return False
        
        # Keep only recent history
        self.loss_history = self.loss_history[-self.detection_window:]
        
        # Calculate loss variance (high variance = potential transition)
        recent_losses = self.loss_history[-20:]
        if len(recent_losses) >= 20:
            variance = np.var(recent_losses)
            mean_loss = np.mean(recent_losses)
            cv = variance / (mean_loss + 1e-8)  # Coefficient of variation
            
            return cv > 0.1  # Threshold for transition detection
        
        return False
    
    def step(self, current_loss):
        transition_detected = self.detect_phase_transition(current_loss)
        
        if transition_detected and not self.in_transition:
            # Start of transition - boost learning rate
            self.in_transition = True
            self.transition_timer = 0
            new_lr = self.base_lr * self.transition_lr_boost
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            print(f"Phase transition detected! Boosting LR to {new_lr:.6f}")
            
        elif self.in_transition:
            self.transition_timer += 1
            
            # Gradually reduce LR back to base
            if self.transition_timer > 50:  # 50 epochs of boost
                decay_factor = 0.95
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= decay_factor
                
                if param_group['lr'] <= self.base_lr:
                    param_group['lr'] = self.base_lr
                    self.in_transition = False
                    print("Transition complete, returning to base LR")
```

## 📈 Advanced Scheduling Techniques

### 1. Learning Rate Range Test
```python
class LRRangeFinder:
    """Find optimal learning rate range for cyclic LR"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def find_lr_range(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Find LR range by gradually increasing LR and monitoring loss"""
        model = self.model
        model.train()
        
        lrs = []
        losses = []
        
        # Save initial state
        initial_state = model.state_dict()
        
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        current_lr = start_lr
        
        for i, (data, target) in enumerate(train_loader):
            if i >= num_iter:
                break
                
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            # Training step
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Record
            lrs.append(current_lr)
            losses.append(loss.item())
            
            # Update learning rate
            current_lr *= lr_mult
            
            # Stop if loss explodes
            if i > 10 and loss.item() > 4 * min(losses):
                break
        
        # Restore initial state
        model.load_state_dict(initial_state)
        
        return lrs, losses
    
    def plot_lr_range(self, lrs, losses):
        """Plot LR range test results"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Plot derivative to find steepest descent
        if len(losses) > 1:
            derivatives = np.gradient(losses)
            plt.plot(lrs[1:], derivatives[1:])
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss Gradient')
            plt.title('Loss Gradient (Find Steepest)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Suggest optimal LR
        min_gradient_idx = np.argmin(derivatives)
        suggested_lr = lrs[min_gradient_idx]
        print(f"Suggested learning rate: {suggested_lr:.6f}")
        return suggested_lr
```

### 2. Adaptive Momentum Scheduling
```python
class AdaptiveMomentumScheduler:
    """Coordinate learning rate and momentum schedules"""
    
    def __init__(self, optimizer, lr_scheduler, max_momentum=0.95, min_momentum=0.85):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        # Step the LR scheduler
        self.lr_scheduler.step()
        
        # Adjust momentum inversely to learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_ratio = current_lr / self.initial_lr
        
        # Higher LR -> Lower momentum, Lower LR -> Higher momentum
        momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * lr_ratio
        momentum = max(min(momentum, self.max_momentum), self.min_momentum)
        
        # Update momentum (for SGD optimizers)
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                param_group['momentum'] = momentum
```

## 📊 Experimental Configurations

### Complete Grokking Setup
```python
def setup_grokking_training(model, total_epochs=10000):
    """Complete training setup optimized for grokking"""
    
    # Optimizer with strong weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
        betas=(0.9, 0.98)
    )
    
    # Multi-phase scheduler
    scheduler = GrokkingScheduler(optimizer, total_epochs)
    
    # Training configuration
    config = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epochs": total_epochs,
        "log_interval": 100,
        "eval_interval": 100,
        "patience": 2000,  # For early stopping (optional)
        
        # Monitoring
        "track_weight_norms": True,
        "track_gradient_norms": True,
        "save_checkpoints": True,
        "checkpoint_interval": 1000
    }
    
    return config
```

### Bias Mitigation Setup
```python
def setup_bias_mitigation_training(model, total_epochs=300):
    """Training setup for combating simplicity bias"""
    
    # SGD often better for robustness
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Bias-specific scheduler
    scheduler = BiasScheduler(optimizer, total_epochs)
    
    config = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epochs": total_epochs,
        "log_interval": 10,
        "eval_interval": 10,
        
        # Group evaluation for bias tracking
        "track_group_performance": True,
        "worst_group_patience": 50,
        
        # Data augmentation schedule
        "augmentation_schedule": "progressive",
        "max_augmentation_epoch": 100
    }
    
    return config
```

## 🎯 Best Practices

### Schedule Selection Guidelines
1. **Grokking**: Cosine annealing with long warmup
2. **Simplicity Bias**: Multi-phase with aggressive start
3. **Phase Transitions**: Adaptive schedules that respond to dynamics
4. **Large Models**: Conservative schedules with warmup restarts

### Monitoring and Adjustment
1. **Track Multiple Metrics**: Loss, accuracy, weight norms, gradient norms
2. **Visualize Schedules**: Plot LR trajectories before training
3. **Be Patient**: Good schedules may look worse initially
4. **Combine Techniques**: Warmup + main schedule + fine-tuning

### Common Mistakes
1. **Too Aggressive**: High LR without proper warmup
2. **Too Conservative**: Missing opportunities for faster convergence
3. **Fixed Schedules**: Not adapting to actual training dynamics
4. **Ignoring Momentum**: Not coordinating LR and momentum changes

## 🔗 References

1. Smith (2017). "Cyclical Learning Rates for Training Neural Networks"
2. Loshchilov & Hutter (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts"
3. He et al. (2019). "Bag of Tricks for Image Classification"
4. Power et al. (2022). "Grokking: Generalization Beyond Overfitting"
5. Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
6. Smith & Topin (2019). "Super-Convergence: Very Fast Training of Neural Networks"