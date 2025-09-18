# Phase Transitions Dataset

Synthetic datasets for studying emergent abilities and phase transitions in machine learning models.

## 📋 Overview

This dataset collection focuses on tasks where models exhibit sharp phase transitions - sudden improvements in performance that occur after extended training, often associated with emergent abilities in large language models and other architectures.

## 🔬 Phenomenon Details

### Phase Transition Characteristics
1. **Pre-transition Phase (0-N epochs)**: Near-random performance despite training
2. **Transition Point (epoch N)**: Sudden, sharp improvement in performance  
3. **Post-transition Phase (N+ epochs)**: High-quality task performance
4. **Scaling Behavior**: Transition point depends on model size, data size, compute

### Types of Phase Transitions
- **Emergent Abilities**: New capabilities that appear suddenly
- **Grokking**: Sudden generalization after memorization
- **In-context Learning**: Ability to learn from examples in context
- **Compositional Understanding**: Combining learned elements in novel ways

## 🛠️ Dataset Generation

### Basic Emergent Tasks
```bash
python generate_emergent_tasks.py \
    --task_types arithmetic reasoning memory \
    --train_size 5000 \
    --test_size 1000 \
    --output_dir ./emergent_tasks_data
```

### Arithmetic Tasks
```bash
# Simple arithmetic that shows phase transitions
python generate_emergent_tasks.py \
    --task_types arithmetic \
    --arithmetic_complexity multi_step \
    --max_numbers 100 \
    --train_size 10000 \
    --output_dir ./arithmetic_emergent
```

### Reasoning Tasks
```bash
# Logic and reasoning tasks
python generate_emergent_tasks.py \
    --task_types reasoning \
    --reasoning_types logical_inference pattern_completion \
    --complexity_levels simple medium hard \
    --output_dir ./reasoning_emergent
```

### Memory Tasks
```bash
# Memory and recall tasks
python generate_emergent_tasks.py \
    --task_types memory \
    --memory_length_range 5 50 \
    --memory_types sequence_recall associative_memory \
    --output_dir ./memory_emergent
```

## 📊 Task Categories

### 1. Arithmetic Tasks
```python
arithmetic_tasks = {
    "simple_addition": "Basic addition problems",
    "multi_step": "Multi-step arithmetic problems", 
    "word_problems": "Arithmetic word problems",
    "modular_arithmetic": "Arithmetic modulo prime numbers",
    "sequence_arithmetic": "Arithmetic sequences and patterns"
}
```

### 2. Reasoning Tasks  
```python
reasoning_tasks = {
    "logical_inference": "IF-THEN logical reasoning",
    "pattern_completion": "Complete number/symbol patterns",
    "analogical_reasoning": "A:B as C:? style problems",
    "causal_reasoning": "Cause and effect relationships",
    "compositional_reasoning": "Combine multiple reasoning steps"
}
```

### 3. Memory Tasks
```python
memory_tasks = {
    "sequence_recall": "Remember and recall sequences",
    "associative_memory": "Key-value pair recall",
    "context_memory": "Remember information from context",
    "working_memory": "Hold and manipulate information",
    "episodic_memory": "Remember specific episodes/events"
}
```

### 4. Language Tasks
```python
language_tasks = {
    "grammar_induction": "Learn grammar rules from examples",
    "semantic_understanding": "Understand word meanings",
    "compositional_language": "Combine words/phrases meaningfully",
    "translation": "Simple language translation tasks",
    "question_answering": "Answer questions about text"
}
```

## 📈 Training Protocols

### Standard Phase Transition Training
```python
model = TransformerModel(
    vocab_size=1000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=128
)

config = {
    "epochs": 10000,  # Long training for phase transitions
    "batch_size": 64,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    
    # Phase transition monitoring
    "eval_interval": 100,
    "log_detailed_metrics": True,
    "save_checkpoints": True,
    "checkpoint_interval": 500
}
```

### Emergent Ability Training
```python
class EmergentAbilityTrainer:
    def __init__(self, model, tasks):
        self.model = model
        self.tasks = tasks
        self.ability_emergence = {}
        
    def train_epoch(self, epoch):
        for task_name, task_data in self.tasks.items():
            # Train on task
            loss = self.train_task(task_data)
            
            # Evaluate ability emergence
            ability_score = self.evaluate_ability(task_name)
            
            # Track emergence
            self.track_emergence(task_name, epoch, ability_score)
            
    def track_emergence(self, task_name, epoch, score):
        """Detect when ability suddenly emerges"""
        if task_name not in self.ability_emergence:
            self.ability_emergence[task_name] = {
                "scores": [],
                "epochs": [],
                "emerged": False,
                "emergence_epoch": None
            }
        
        history = self.ability_emergence[task_name]
        history["scores"].append(score)
        history["epochs"].append(epoch)
        
        # Detect sharp improvement (emergence)
        if len(history["scores"]) >= 20:
            recent_improvement = (
                np.mean(history["scores"][-5:]) - 
                np.mean(history["scores"][-20:-15])
            )
            
            if recent_improvement > 0.3 and not history["emerged"]:
                history["emerged"] = True
                history["emergence_epoch"] = epoch
                print(f"Ability '{task_name}' emerged at epoch {epoch}!")
```

### Multi-Scale Training
```python
def multi_scale_training(tasks, model_sizes):
    """Study how phase transitions scale with model size"""
    results = {}
    
    for model_size in model_sizes:
        model = create_model(size=model_size)
        trainer = EmergentAbilityTrainer(model, tasks)
        
        # Train and track emergence
        for epoch in range(10000):
            trainer.train_epoch(epoch)
            
        results[model_size] = trainer.ability_emergence
    
    # Analyze scaling laws
    analyze_emergence_scaling(results)
```

## 📊 Evaluation Metrics

### Phase Transition Detection
```python
def detect_phase_transition(performance_history, window_size=100):
    """Detect sharp phase transitions in performance"""
    if len(performance_history) < window_size * 2:
        return None
    
    transitions = []
    
    for i in range(window_size, len(performance_history) - window_size):
        # Compare before and after windows
        before = np.mean(performance_history[i-window_size:i])
        after = np.mean(performance_history[i:i+window_size])
        
        # Detect sharp improvement
        if after - before > 0.2:  # 20% improvement threshold
            transitions.append({
                "epoch": i,
                "before_performance": before,
                "after_performance": after,
                "improvement": after - before
            })
    
    return transitions

def measure_transition_sharpness(performance_history, transition_epoch):
    """Measure how sharp the transition is"""
    # Look at performance change rate around transition
    window = 50
    start = max(0, transition_epoch - window)
    end = min(len(performance_history), transition_epoch + window)
    
    transition_window = performance_history[start:end]
    
    # Fit linear model to measure slope
    x = np.arange(len(transition_window))
    slope, _, r_squared, _, _ = stats.linregress(x, transition_window)
    
    return {
        "slope": slope,
        "r_squared": r_squared,
        "sharpness_score": slope * r_squared  # Sharp = high slope + good fit
    }
```

### Emergent Ability Metrics
```python
def evaluate_emergent_abilities(model, task_suite):
    """Comprehensive evaluation of emergent abilities"""
    
    results = {
        "task_performances": {},
        "ability_scores": {},
        "emergence_indicators": {}
    }
    
    for task_name, task_data in task_suite.items():
        # Basic task performance
        accuracy = evaluate_task_accuracy(model, task_data)
        results["task_performances"][task_name] = accuracy
        
        # Ability-specific metrics
        if task_name in ["arithmetic", "multi_step"]:
            # For arithmetic: measure step-by-step reasoning
            step_accuracy = evaluate_reasoning_steps(model, task_data)
            results["ability_scores"][f"{task_name}_reasoning"] = step_accuracy
            
        elif task_name in ["logical_inference", "analogical_reasoning"]:
            # For reasoning: measure generalization to new patterns
            generalization = evaluate_reasoning_generalization(model, task_data)
            results["ability_scores"][f"{task_name}_generalization"] = generalization
            
        elif task_name in ["sequence_recall", "associative_memory"]:
            # For memory: measure capacity and fidelity
            capacity = evaluate_memory_capacity(model, task_data)
            fidelity = evaluate_memory_fidelity(model, task_data)
            results["ability_scores"][f"{task_name}_capacity"] = capacity
            results["ability_scores"][f"{task_name}_fidelity"] = fidelity
        
        # Emergence indicators
        consistency = evaluate_performance_consistency(model, task_data)
        confidence = evaluate_prediction_confidence(model, task_data)
        results["emergence_indicators"][task_name] = {
            "consistency": consistency,
            "confidence": confidence
        }
    
    return results
```

## 🎯 Expected Results

### Phase Transition Timeline
1. **Random Phase (0-N epochs)**: Performance at chance level
2. **Transition Window (N to N+ΔN)**: Rapid improvement over ~100-500 epochs
3. **High Performance Phase (N+ΔN+)**: Stable high performance
4. **Scaling**: Larger models transition earlier (smaller N)

### Emergent Ability Patterns
- **Sudden Onset**: Abilities appear quickly once threshold is reached
- **Model Size Dependence**: Larger models develop abilities at earlier training
- **Task Hierarchy**: Some abilities emerge before others (prerequisites)
- **Compositional Emergence**: Complex abilities built from simpler ones

## 🔬 Research Applications

### Scaling Law Studies
```python
def study_emergence_scaling_laws(task_suite):
    """Study how emergence scales with model size, data, compute"""
    
    # Model size scaling
    model_sizes = [1e6, 5e6, 1e7, 5e7, 1e8]  # parameters
    emergence_vs_size = {}
    
    for size in model_sizes:
        model = create_model(size)
        emergence_epoch = train_until_emergence(model, task_suite)
        emergence_vs_size[size] = emergence_epoch
    
    # Data size scaling  
    data_sizes = [1e3, 5e3, 1e4, 5e4, 1e5]  # samples
    emergence_vs_data = {}
    
    for data_size in data_sizes:
        limited_tasks = limit_data_size(task_suite, data_size)
        emergence_epoch = train_until_emergence(model, limited_tasks)
        emergence_vs_data[data_size] = emergence_epoch
    
    # Analyze scaling relationships
    analyze_scaling_laws(emergence_vs_size, emergence_vs_data)
```

### Mechanistic Interpretability
```python
def analyze_emergence_mechanisms(model, task_data, emergence_epoch):
    """Understand what changes in the model during emergence"""
    
    # Compare model states before/after emergence
    before_epoch = emergence_epoch - 500
    after_epoch = emergence_epoch + 500
    
    before_state = load_checkpoint(before_epoch)
    after_state = load_checkpoint(after_epoch)
    
    # Analyze changes
    weight_changes = analyze_weight_changes(before_state, after_state)
    activation_changes = analyze_activation_patterns(
        model, task_data, before_state, after_state
    )
    attention_changes = analyze_attention_patterns(
        model, task_data, before_state, after_state
    )
    
    return {
        "weight_changes": weight_changes,
        "activation_changes": activation_changes, 
        "attention_changes": attention_changes
    }
```

## 🔗 Integration Example

```python
from datasets.phase_transitions.generate_emergent_tasks import create_emergent_datasets

# Create multi-task suite for phase transition study
task_suite = create_emergent_datasets(
    task_types=['arithmetic', 'reasoning', 'memory'],
    train_size=5000,
    test_size=1000,
    seed=42
)

# Model architecture prone to phase transitions
model = TransformerModel(
    vocab_size=task_suite.metadata['vocab_size'],
    d_model=512,
    n_heads=8,
    n_layers=8
)

# Training with emergence monitoring
trainer = EmergentAbilityTrainer(model, task_suite)
results = trainer.train(
    epochs=10000,
    monitor_emergence=True,
    save_emergence_analysis=True
)

# Analyze emergence patterns
emergence_analysis = analyze_emergence_patterns(results)
print("Emerged abilities:", emergence_analysis['emerged_abilities'])
print("Emergence timeline:", emergence_analysis['emergence_epochs'])
```

## 📚 References

1. Wei et al. (2022). "Emergent Abilities of Large Language Models"
2. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
3. Brown et al. (2020). "Language Models are Few-Shot Learners"
4. Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
5. Ganguli et al. (2022). "Predictability and Surprise in Large Generative Models"