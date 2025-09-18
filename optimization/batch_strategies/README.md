# Batch Size and Training Strategies for Delayed Generalization

## 📋 Overview

Batch size significantly influences delayed generalization phenomena, affecting everything from the sharpness of phase transitions to the final quality of generalization. Understanding how batch size interacts with different delayed generalization scenarios is crucial for optimal training strategies.

## 🔬 Batch Size Impact on Delayed Generalization

### Small Batch Sizes (8-64)
- **Grokking**: Often accelerates the transition to generalization
- **Mechanism**: Higher gradient noise helps escape memorization basins
- **Trade-off**: Slower training, more variance, but better final performance
- **Timeline**: May reduce total epochs needed for generalization

### Medium Batch Sizes (128-512)
- **Balance**: Good compromise between speed and generalization quality
- **Stability**: More stable training while preserving exploration
- **Most Common**: Default choice for many delayed generalization experiments

### Large Batch Sizes (1024+)
- **Risk**: Can get stuck in memorization phase longer
- **Requires**: Careful learning rate scaling and warmup
- **Benefits**: Faster training, more stable gradients
- **Challenge**: May need special techniques to achieve generalization

## 🛠️ Batch Size Strategies

### 1. Progressive Batch Size Scaling
```python
class ProgressiveBatchSize:
    """Gradually increase batch size during training"""
    
    def __init__(self, initial_batch_size=32, final_batch_size=512, 
                 growth_factor=1.5, growth_interval=1000):
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.growth_factor = growth_factor
        self.growth_interval = growth_interval
        
        self.current_batch_size = initial_batch_size
        self.step_count = 0
        
    def should_increase_batch_size(self):
        return (self.step_count % self.growth_interval == 0 and 
                self.step_count > 0 and 
                self.current_batch_size < self.final_batch_size)
    
    def get_current_batch_size(self):
        if self.should_increase_batch_size():
            new_size = min(
                int(self.current_batch_size * self.growth_factor),
                self.final_batch_size
            )
            if new_size != self.current_batch_size:
                print(f"Increasing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        
        self.step_count += 1
        return self.current_batch_size

def create_progressive_dataloader(dataset, batch_scheduler):
    """Create dataloader that adapts batch size"""
    while True:
        batch_size = batch_scheduler.get_current_batch_size()
        yield DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
```

**Benefits for Delayed Generalization**:
- Start small for exploration, increase for efficiency
- Helps avoid getting stuck in memorization
- Balances training speed with generalization quality

### 2. Adaptive Batch Size Based on Loss
```python
class AdaptiveBatchSize:
    """Adjust batch size based on training dynamics"""
    
    def __init__(self, initial_batch_size=128, min_batch_size=32, 
                 max_batch_size=1024, adaptation_window=100):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adaptation_window = adaptation_window
        
        self.current_batch_size = initial_batch_size
        self.loss_history = []
        
    def update_batch_size(self, train_loss, val_loss):
        """Adapt batch size based on overfitting detection"""
        self.loss_history.append({'train': train_loss, 'val': val_loss})
        
        if len(self.loss_history) < self.adaptation_window:
            return self.current_batch_size
        
        # Keep only recent history
        self.loss_history = self.loss_history[-self.adaptation_window:]
        
        # Calculate recent overfitting trend
        recent_data = self.loss_history[-20:]
        if len(recent_data) >= 20:
            avg_gap = np.mean([d['val'] - d['train'] for d in recent_data])
            gap_trend = np.polyfit(range(len(recent_data)), 
                                 [d['val'] - d['train'] for d in recent_data], 1)[0]
            
            # If gap is increasing rapidly, decrease batch size for more exploration
            if gap_trend > 0.01 and avg_gap > 0.5:
                new_size = max(self.current_batch_size // 2, self.min_batch_size)
                if new_size != self.current_batch_size:
                    print(f"Overfitting detected, reducing batch size: {self.current_batch_size} -> {new_size}")
                    self.current_batch_size = new_size
            
            # If training is stable, can increase batch size for efficiency
            elif gap_trend < 0.001 and avg_gap < 0.1:
                new_size = min(self.current_batch_size * 2, self.max_batch_size)
                if new_size != self.current_batch_size:
                    print(f"Stable training, increasing batch size: {self.current_batch_size} -> {new_size}")
                    self.current_batch_size = new_size
        
        return self.current_batch_size
```

### 3. Phase-Specific Batch Sizes
```python
class PhaseBatchStrategy:
    """Different batch sizes for different training phases"""
    
    def __init__(self, phases):
        """
        phases: list of dicts with 'epochs', 'batch_size', 'description'
        """
        self.phases = phases
        self.current_phase = 0
        self.epoch_count = 0
        
    def get_batch_size_for_epoch(self, epoch):
        """Get batch size for current epoch"""
        cumulative_epochs = 0
        
        for i, phase in enumerate(self.phases):
            cumulative_epochs += phase['epochs']
            if epoch < cumulative_epochs:
                if i != self.current_phase:
                    print(f"Entering phase {i}: {phase['description']}")
                    print(f"Batch size: {phase['batch_size']}")
                    self.current_phase = i
                return phase['batch_size']
        
        # Return last phase batch size if beyond all phases
        return self.phases[-1]['batch_size']

# Example for grokking
grokking_batch_strategy = PhaseBatchStrategy([
    {
        'epochs': 2000,
        'batch_size': 64,
        'description': 'Initial exploration with small batches'
    },
    {
        'epochs': 5000,
        'batch_size': 256,
        'description': 'Main training with medium batches'
    },
    {
        'epochs': 3000,
        'batch_size': 512,
        'description': 'Final convergence with larger batches'
    }
])
```

## 📊 Learning Rate Scaling with Batch Size

### 1. Linear Scaling Rule
```python
def linear_lr_scaling(base_lr, base_batch_size, current_batch_size):
    """Linear scaling: LR ∝ batch size"""
    return base_lr * (current_batch_size / base_batch_size)

# Example usage
base_lr = 1e-3
base_batch_size = 128
current_batch_size = 512

scaled_lr = linear_lr_scaling(base_lr, base_batch_size, current_batch_size)
print(f"Scaled LR: {scaled_lr}")  # 4e-3
```

### 2. Square Root Scaling
```python
def sqrt_lr_scaling(base_lr, base_batch_size, current_batch_size):
    """Square root scaling: LR ∝ √(batch size)"""
    import math
    return base_lr * math.sqrt(current_batch_size / base_batch_size)

# Often better for large batch training
scaled_lr = sqrt_lr_scaling(1e-3, 128, 512)
print(f"Sqrt scaled LR: {scaled_lr}")  # 2e-3
```

### 3. Adaptive LR Scaling for Delayed Generalization
```python
class DelayedGeneralizationLRScaling:
    """LR scaling that considers delayed generalization patterns"""
    
    def __init__(self, base_lr=1e-3, base_batch_size=128, 
                 conservative_factor=0.5):
        self.base_lr = base_lr
        self.base_batch_size = base_batch_size
        self.conservative_factor = conservative_factor
        
    def scale_lr(self, current_batch_size, training_phase='exploration'):
        """Scale LR based on batch size and training phase"""
        
        if training_phase == 'exploration':
            # More conservative scaling during exploration
            scaling_factor = (current_batch_size / self.base_batch_size) ** 0.5
            return self.base_lr * scaling_factor * self.conservative_factor
        
        elif training_phase == 'memorization':
            # Standard linear scaling during memorization
            scaling_factor = current_batch_size / self.base_batch_size
            return self.base_lr * scaling_factor
            
        elif training_phase == 'generalization':
            # Very conservative during generalization transition
            scaling_factor = (current_batch_size / self.base_batch_size) ** 0.25
            return self.base_lr * scaling_factor * self.conservative_factor
        
        else:
            # Default to sqrt scaling
            scaling_factor = (current_batch_size / self.base_batch_size) ** 0.5
            return self.base_lr * scaling_factor
```

## 🎯 Batch Strategies for Specific Phenomena

### Grokking-Optimized Batch Strategy
```python
class GrokkingBatchStrategy:
    """Batch strategy specifically for grokking experiments"""
    
    def __init__(self, dataset_size, target_epochs=10000):
        self.dataset_size = dataset_size
        self.target_epochs = target_epochs
        
        # Calculate optimal batch progression
        self.phases = self._calculate_grokking_phases()
        
    def _calculate_grokking_phases(self):
        """Calculate batch size phases for grokking"""
        return [
            {
                'epochs': int(0.2 * self.target_epochs),  # 20% exploration
                'batch_size': min(64, self.dataset_size // 100),
                'description': 'Small batch exploration',
                'lr_scaling': 'conservative'
            },
            {
                'epochs': int(0.6 * self.target_epochs),  # 60% main training
                'batch_size': min(256, self.dataset_size // 50),
                'description': 'Medium batch main training',
                'lr_scaling': 'sqrt'
            },
            {
                'epochs': int(0.2 * self.target_epochs),  # 20% convergence
                'batch_size': min(512, self.dataset_size // 20),
                'description': 'Large batch convergence',
                'lr_scaling': 'linear'
            }
        ]
    
    def get_config_for_epoch(self, epoch):
        """Get batch configuration for given epoch"""
        cumulative = 0
        for phase in self.phases:
            cumulative += phase['epochs']
            if epoch < cumulative:
                return phase
        return self.phases[-1]  # Return last phase if beyond

# Usage example
grokking_strategy = GrokkingBatchStrategy(dataset_size=10000, target_epochs=10000)
```

### Simplicity Bias Mitigation Strategy
```python
class BiasMinimizationBatchStrategy:
    """Batch strategy to combat simplicity bias"""
    
    def __init__(self, total_epochs=300, bias_strength='medium'):
        self.total_epochs = total_epochs
        self.bias_strength = bias_strength
        
    def get_batch_schedule(self):
        """Get batch schedule for bias mitigation"""
        
        if self.bias_strength == 'strong':
            # Very strong bias requires very small batches initially
            return [
                {
                    'epochs': int(0.3 * self.total_epochs),
                    'batch_size': 16,  # Very small for exploration
                    'description': 'Micro-batch exploration'
                },
                {
                    'epochs': int(0.4 * self.total_epochs),
                    'batch_size': 64,
                    'description': 'Small batch feature development'
                },
                {
                    'epochs': int(0.3 * self.total_epochs),
                    'batch_size': 128,
                    'description': 'Standard batch convergence'
                }
            ]
        
        else:  # medium or weak bias
            return [
                {
                    'epochs': int(0.4 * self.total_epochs),
                    'batch_size': 32,
                    'description': 'Small batch exploration'
                },
                {
                    'epochs': int(0.6 * self.total_epochs),
                    'batch_size': 128,
                    'description': 'Standard batch training'
                }
            ]
```

### Large Model Phase Transition Strategy
```python
class LargeModelBatchStrategy:
    """Batch strategy for large models with phase transitions"""
    
    def __init__(self, model_params, gpu_memory_gb=16):
        self.model_params = model_params
        self.gpu_memory_gb = gpu_memory_gb
        self.max_batch_size = self._estimate_max_batch_size()
        
    def _estimate_max_batch_size(self):
        """Estimate maximum batch size based on model size and GPU memory"""
        # Rough estimation: 4 bytes per parameter for float32
        model_memory_gb = (self.model_params * 4) / (1024**3)
        
        # Conservative estimate: use 50% of GPU memory for batch
        available_memory_gb = self.gpu_memory_gb * 0.5
        
        # Very rough batch size estimation
        estimated_max = int(available_memory_gb / model_memory_gb * 100)
        return min(max(estimated_max, 32), 2048)  # Reasonable bounds
    
    def get_phase_transition_schedule(self):
        """Get batch schedule optimized for phase transitions"""
        return [
            {
                'epochs': 1000,
                'batch_size': max(32, self.max_batch_size // 8),
                'description': 'Initial small batch exploration'
            },
            {
                'epochs': 5000,
                'batch_size': max(64, self.max_batch_size // 4),
                'description': 'Medium batch development'
            },
            {
                'epochs': 10000,
                'batch_size': max(128, self.max_batch_size // 2),
                'description': 'Large batch consolidation'
            }
        ]
```

## 📈 Advanced Batch Techniques

### 1. Gradient Accumulation for Effective Large Batches
```python
class GradientAccumulationTrainer:
    """Simulate large batch sizes with gradient accumulation"""
    
    def __init__(self, model, optimizer, effective_batch_size, 
                 actual_batch_size, device):
        self.model = model
        self.optimizer = optimizer
        self.effective_batch_size = effective_batch_size
        self.actual_batch_size = actual_batch_size
        self.device = device
        
        # Calculate accumulation steps
        self.accumulation_steps = effective_batch_size // actual_batch_size
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        
    def train_step(self, dataloader):
        """Training step with gradient accumulation"""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        data_iter = iter(dataloader)
        
        for step in range(self.accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step after accumulating gradients
        self.optimizer.step()
        
        return total_loss
```

### 2. Dynamic Batch Size with Memory Management
```python
class MemoryAwareBatchSize:
    """Dynamically adjust batch size based on available GPU memory"""
    
    def __init__(self, model, initial_batch_size=32, max_batch_size=1024):
        self.model = model
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
    def find_optimal_batch_size(self, sample_data, device):
        """Binary search for maximum batch size that fits in memory"""
        low, high = self.initial_batch_size, self.max_batch_size
        optimal_size = self.initial_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test if this batch size fits
                test_batch = sample_data[:mid].to(device)
                with torch.no_grad():
                    _ = self.model(test_batch)
                
                optimal_size = mid
                low = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    high = mid - 1
                else:
                    raise e
        
        # Use 90% of maximum for safety margin
        self.current_batch_size = int(optimal_size * 0.9)
        print(f"Optimal batch size found: {self.current_batch_size}")
        return self.current_batch_size
```

### 3. Curriculum Batch Scheduling
```python
class CurriculumBatchScheduler:
    """Batch size scheduling based on curriculum learning principles"""
    
    def __init__(self, difficulty_schedule, batch_size_mapping):
        """
        difficulty_schedule: list of (epoch, difficulty_level)
        batch_size_mapping: dict mapping difficulty to batch size
        """
        self.difficulty_schedule = difficulty_schedule
        self.batch_size_mapping = batch_size_mapping
        
    def get_batch_size_for_epoch(self, epoch):
        """Get batch size based on curriculum difficulty"""
        current_difficulty = 'easy'  # default
        
        for epoch_threshold, difficulty in self.difficulty_schedule:
            if epoch >= epoch_threshold:
                current_difficulty = difficulty
            else:
                break
        
        return self.batch_size_mapping.get(current_difficulty, 128)

# Example for delayed generalization curriculum
curriculum_schedule = [
    (0, 'easy'),      # Start with easy examples
    (1000, 'medium'), # Introduce medium difficulty
    (5000, 'hard'),   # Full difficulty
    (8000, 'mixed')   # Mixed difficulty for robustness
]

batch_mapping = {
    'easy': 64,    # Smaller batches for careful learning
    'medium': 128, # Standard batches
    'hard': 256,   # Larger batches for efficiency
    'mixed': 128   # Back to standard
}

scheduler = CurriculumBatchScheduler(curriculum_schedule, batch_mapping)
```

## 🎯 Best Practices

### Batch Size Selection Guidelines
1. **Start Small**: Begin with smaller batches for better exploration
2. **Monitor Carefully**: Track both efficiency and generalization quality
3. **Scale Gradually**: Progressive increases often work better than sudden changes
4. **Consider Memory**: Don't max out GPU memory, leave room for optimization

### Learning Rate Coordination
1. **Scale Learning Rate**: Adjust LR when changing batch size
2. **Use Warmup**: Especially important with large batch training
3. **Monitor Gradients**: Check gradient norms with different batch sizes
4. **Be Conservative**: Err on the side of smaller LR with larger batches

### Common Mistakes
1. **Too Large Too Early**: Large batches can prevent initial exploration
2. **Ignoring LR Scaling**: Not adjusting LR with batch size changes
3. **Memory Overflow**: Not leaving enough GPU memory for optimization
4. **Fixed Strategies**: Not adapting to actual training dynamics

## 🔗 References

1. Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
2. Smith et al. (2018). "Don't Decay the Learning Rate, Increase the Batch Size"
3. Masters & Luschi (2018). "Revisiting Small Batch Training for Deep Neural Networks"
4. Shallue et al. (2019). "Measuring the Effects of Data Parallelism on Neural Network Training"
5. McCandlish et al. (2018). "An Empirical Model of Large-Batch Training"
6. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"