# TRAK (Tracing with Randomly-projected After Kernels) Implementation

## 📋 Overview

TRAK is an efficient method for computing data attribution - identifying which training examples most influence a model's predictions on test examples. This implementation is optimized for studying delayed generalization phenomena.

## 🔬 How TRAK Works

### Algorithm Steps
1. **Gradient Computation**: Compute gradients of loss w.r.t. model parameters for each example
2. **Random Projection**: Project high-dimensional gradients to lower dimension
3. **Kernel Similarity**: Compute similarities between projected train/test gradients
4. **Attribution Scores**: Higher similarity = higher influence

### Advantages for Delayed Generalization
- **Efficient**: Works with large models and datasets
- **Temporal Analysis**: Can track attribution changes across training epochs
- **Phase Transitions**: Identify examples that trigger generalization
- **Memory Efficient**: Random projection reduces storage requirements

## 🛠️ Usage

### Basic Attribution Analysis
```python
from data_attribution.trak.trak_attributor import TRAKAttributor

# Initialize TRAK
attributor = TRAKAttributor(
    model=model,
    task='classification',
    proj_dim=512,
    device=device
)

# Compute training features
train_features = attributor.compute_train_features(
    dataloader=train_loader,
    num_samples=len(train_dataset)
)

# Compute test features
test_features = attributor.compute_test_features(
    dataloader=test_loader
)

# Compute attribution matrix
# attributions[i, j] = influence of train example j on test example i
attributions = attributor.compute_attributions(
    train_features=train_features,
    test_features=test_features
)

# Find most influential training examples for a test example
test_idx = 0
top_train_indices, top_scores = attributor.find_top_attributions(
    attributions, test_idx, top_k=10, return_scores=True
)

print(f"Most influential training examples for test example {test_idx}:")
for i, (train_idx, score) in enumerate(zip(top_train_indices, top_scores)):
    print(f"  {i+1}. Train example {train_idx}: score {score:.4f}")
```

### Delayed Generalization Analysis
```python
from data_attribution.trak.trak_attributor import DelayedGeneralizationTRAK

# Extended TRAK for tracking attribution evolution
dg_trak = DelayedGeneralizationTRAK(
    model=model,
    task='classification',
    proj_dim=512,
    device=device
)

# Compute features across multiple training epochs
epochs_to_analyze = [0, 1000, 2000, 5000, 8000, 10000]

for epoch in epochs_to_analyze:
    # Load model checkpoint for this epoch
    model.load_state_dict(torch.load(f'checkpoints/model_epoch_{epoch}.pt'))
    
    # Compute features
    dg_trak.compute_epoch_features(
        dataloader=train_loader,
        epoch=epoch,
        split='train'
    )
    dg_trak.compute_epoch_features(
        dataloader=test_loader,
        epoch=epoch,
        split='test'
    )

# Analyze how attributions evolve for a specific test example
test_idx = 42
attribution_evolution = dg_trak.analyze_attribution_evolution(
    test_idx=test_idx,
    epochs=epochs_to_analyze,
    top_k=20
)

# Find examples that change influence during phase transition
phase_transition_examples = dg_trak.find_phase_transition_examples(
    pre_transition_epoch=2000,
    post_transition_epoch=8000,
    top_k=50
)

print("Examples that gained influence during phase transition:")
for idx, score in zip(
    phase_transition_examples['gained_influence_indices'][:10],
    phase_transition_examples['gained_influence_scores'][:10]
):
    print(f"  Train example {idx}: gained {score:.4f}")
```

### Grokking Attribution Analysis
```python
# Analyze which examples enable grokking in algorithmic tasks
def analyze_grokking_attributions(
    model_checkpoints,
    train_loader,
    test_loader,
    grokking_epoch
):
    """
    Analyze which training examples contribute to grokking.
    
    Args:
        model_checkpoints: Dict mapping epochs to model states
        train_loader: Training data loader
        test_loader: Test data loader
        grokking_epoch: Epoch when grokking occurred
    """
    
    # Initialize TRAK
    trak = DelayedGeneralizationTRAK(
        model=model,
        task='classification',
        proj_dim=256,  # Smaller for algorithmic tasks
        device=device
    )
    
    # Compute features before and after grokking
    pre_grok_epoch = grokking_epoch - 500
    post_grok_epoch = grokking_epoch + 500
    
    for epoch in [pre_grok_epoch, post_grok_epoch]:
        model.load_state_dict(model_checkpoints[epoch])
        
        trak.compute_epoch_features(train_loader, epoch, 'train')
        trak.compute_epoch_features(test_loader, epoch, 'test')
    
    # Find examples that enable generalization
    transition_examples = trak.find_phase_transition_examples(
        pre_transition_epoch=pre_grok_epoch,
        post_transition_epoch=post_grok_epoch,
        top_k=100
    )
    
    # Analyze these examples
    analyze_grokking_examples(
        transition_examples['gained_influence_indices'],
        train_dataset
    )

def analyze_grokking_examples(example_indices, train_dataset):
    """Analyze what makes certain examples important for grokking."""
    
    # For modular arithmetic, analyze mathematical properties
    examples = [train_dataset[idx] for idx in example_indices]
    
    # Extract mathematical properties
    properties = {
        'operand_ranges': [],
        'result_ranges': [],
        'operation_types': [],
        'difficulty_levels': []
    }
    
    for example in examples:
        # Parse arithmetic expression (e.g., "5 + 7 = 12")
        parts = example['text'].split()
        a, op, b, eq, result = parts
        
        properties['operand_ranges'].append((int(a), int(b)))
        properties['result_ranges'].append(int(result))
        properties['operation_types'].append(op)
        
        # Simple difficulty metric
        difficulty = abs(int(a) - int(b)) + int(result)
        properties['difficulty_levels'].append(difficulty)
    
    # Analyze patterns
    print("Properties of examples that enable grokking:")
    print(f"  Average operand 1: {np.mean([r[0] for r in properties['operand_ranges']]):.2f}")
    print(f"  Average operand 2: {np.mean([r[1] for r in properties['operand_ranges']]):.2f}")
    print(f"  Average result: {np.mean(properties['result_ranges']):.2f}")
    print(f"  Average difficulty: {np.mean(properties['difficulty_levels']):.2f}")
    
    return properties
```

### Simplicity Bias Attribution
```python
def analyze_bias_attributions(
    model,
    biased_train_loader,
    unbiased_test_loader,
    device
):
    """
    Analyze which training examples promote spurious correlations.
    """
    
    trak = TRAKAttributor(
        model=model,
        task='classification',
        proj_dim=512,
        device=device
    )
    
    # Compute features
    train_features = trak.compute_train_features(biased_train_loader)
    test_features = trak.compute_test_features(unbiased_test_loader)
    
    # Compute attributions
    attributions = trak.compute_attributions(train_features, test_features)
    
    # Analyze attribution patterns
    train_labels = torch.tensor([example[1] for example in biased_train_loader.dataset])
    test_labels = torch.tensor([example[1] for example in unbiased_test_loader.dataset])
    
    bias_analysis = trak.analyze_attribution_patterns(
        attributions, train_labels, test_labels
    )
    
    print("Bias attribution analysis:")
    print(f"  Same-class attribution bias: {bias_analysis.get('same_class_bias', 0):.4f}")
    print(f"  Attribution concentration: {bias_analysis.get('attribution_concentration', 0):.4f}")
    
    # Find examples that promote spurious correlations
    # High attribution to wrong-class examples suggests spurious correlation learning
    spurious_examples = []
    
    for test_idx in range(attributions.size(0)):
        test_label = test_labels[test_idx]
        test_attrs = attributions[test_idx]
        
        # Find training examples with different labels but high attribution
        diff_class_mask = (train_labels != test_label)
        if diff_class_mask.sum() > 0:
            diff_class_attrs = test_attrs[diff_class_mask]
            
            # Top attributed examples from different class
            top_k = min(5, diff_class_attrs.size(0))
            top_scores, top_local_indices = torch.topk(diff_class_attrs, top_k)
            
            # Convert local indices to global indices
            diff_class_indices = torch.where(diff_class_mask)[0]
            top_global_indices = diff_class_indices[top_local_indices]
            
            for global_idx, score in zip(top_global_indices, top_scores):
                spurious_examples.append({
                    'test_idx': test_idx,
                    'train_idx': global_idx.item(),
                    'attribution_score': score.item(),
                    'test_label': test_label.item(),
                    'train_label': train_labels[global_idx].item()
                })
    
    # Sort by attribution score
    spurious_examples.sort(key=lambda x: x['attribution_score'], reverse=True)
    
    print(f"\nTop examples promoting spurious correlations:")
    for i, example in enumerate(spurious_examples[:10]):
        print(f"  {i+1}. Test {example['test_idx']} (label {example['test_label']}) "
              f"<- Train {example['train_idx']} (label {example['train_label']}) "
              f"score: {example['attribution_score']:.4f}")
    
    return spurious_examples
```

## 📊 Evaluation and Validation

### Attribution Quality Metrics
```python
def evaluate_attribution_quality(
    attributor,
    train_loader,
    test_loader,
    model,
    device
):
    """
    Evaluate quality of TRAK attributions using various metrics.
    """
    
    # Compute attributions
    train_features = attributor.compute_train_features(train_loader)
    test_features = attributor.compute_test_features(test_loader)
    attributions = attributor.compute_attributions(train_features, test_features)
    
    metrics = {}
    
    # 1. Linear Datamodeling Score (LDS)
    # Measures how well attributions predict model behavior
    lds_scores = []
    
    for test_idx in range(min(100, attributions.size(0))):  # Sample for efficiency
        test_attrs = attributions[test_idx]
        
        # Get model predictions for training examples
        train_predictions = []
        model.eval()
        
        with torch.no_grad():
            for batch in train_loader:
                data, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(data)
                predictions = torch.softmax(outputs, dim=1)
                train_predictions.append(predictions)
        
        train_predictions = torch.cat(train_predictions, dim=0)
        
        # Predict test example's class probabilities using attribution-weighted average
        weighted_probs = torch.zeros(train_predictions.size(1), device=device)
        
        for train_idx in range(len(test_attrs)):
            weight = test_attrs[train_idx]
            weighted_probs += weight * train_predictions[train_idx]
        
        # Normalize
        if test_attrs.sum() > 0:
            weighted_probs /= test_attrs.sum()
        
        # Compare with actual model prediction on test example
        test_data, test_target = test_loader.dataset[test_idx]
        test_data = test_data.unsqueeze(0).to(device)
        
        with torch.no_grad():
            actual_output = model(test_data)
            actual_probs = torch.softmax(actual_output, dim=1)[0]
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([weighted_probs, actual_probs]))[0, 1]
        if not torch.isnan(correlation):
            lds_scores.append(correlation.item())
    
    metrics['linear_datamodeling_score'] = np.mean(lds_scores) if lds_scores else 0
    
    # 2. Stability across multiple runs
    from data_attribution.trak.trak_attributor import compute_attribution_stability
    stability_metrics = compute_attribution_stability(
        attributor, train_loader, num_runs=3
    )
    metrics.update(stability_metrics)
    
    # 3. Same-class bias
    train_labels = torch.tensor([example[1] for example in train_loader.dataset])
    test_labels = torch.tensor([example[1] for example in test_loader.dataset])
    
    bias_metrics = attributor.analyze_attribution_patterns(
        attributions, train_labels, test_labels
    )
    metrics.update(bias_metrics)
    
    return metrics
```

### Computational Efficiency Analysis
```python
def benchmark_trak_efficiency(
    model,
    train_sizes=[1000, 5000, 10000],
    test_sizes=[100, 500, 1000],
    proj_dims=[128, 256, 512, 1024],
    device=torch.device('cuda')
):
    """
    Benchmark TRAK computational efficiency.
    """
    import time
    
    results = []
    
    for train_size in train_sizes:
        for test_size in test_sizes:
            for proj_dim in proj_dims:
                
                print(f"Benchmarking: train={train_size}, test={test_size}, proj_dim={proj_dim}")
                
                # Create dummy data loaders
                train_loader = create_dummy_loader(train_size, batch_size=32)
                test_loader = create_dummy_loader(test_size, batch_size=32)
                
                # Initialize TRAK
                attributor = TRAKAttributor(
                    model=model,
                    proj_dim=proj_dim,
                    device=device
                )
                
                # Time training feature computation
                start_time = time.time()
                train_features = attributor.compute_train_features(train_loader)
                train_time = time.time() - start_time
                
                # Time test feature computation
                start_time = time.time()
                test_features = attributor.compute_test_features(test_loader)
                test_time = time.time() - start_time
                
                # Time attribution computation
                start_time = time.time()
                attributions = attributor.compute_attributions(train_features, test_features)
                attribution_time = time.time() - start_time
                
                total_time = train_time + test_time + attribution_time
                
                results.append({
                    'train_size': train_size,
                    'test_size': test_size,
                    'proj_dim': proj_dim,
                    'train_time': train_time,
                    'test_time': test_time,
                    'attribution_time': attribution_time,
                    'total_time': total_time,
                    'memory_used': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None
                })
    
    return results

def create_dummy_loader(size, batch_size=32):
    """Create dummy data loader for benchmarking."""
    from torch.utils.data import TensorDataset, DataLoader
    
    data = torch.randn(size, 3, 32, 32)  # CIFAR-like data
    targets = torch.randint(0, 10, (size,))
    dataset = TensorDataset(data, targets)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
```

## 🎯 Best Practices

### Projection Dimension Selection
- **Small models (< 1M params)**: proj_dim = 128-256
- **Medium models (1M-10M params)**: proj_dim = 256-512  
- **Large models (> 10M params)**: proj_dim = 512-1024
- **Trade-off**: Higher proj_dim = more accurate but slower/more memory

### Memory Management
```python
# For large datasets, process in chunks
def compute_features_chunked(
    attributor,
    dataloader,
    chunk_size=1000,
    save_intermediate=True
):
    """Compute features in chunks to manage memory."""
    
    all_features = []
    processed = 0
    
    for chunk_start in range(0, len(dataloader.dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataloader.dataset))
        chunk_size_actual = chunk_end - chunk_start
        
        print(f"Processing chunk {chunk_start}-{chunk_end}")
        
        # Create chunk dataloader
        chunk_indices = list(range(chunk_start, chunk_end))
        chunk_dataset = torch.utils.data.Subset(dataloader.dataset, chunk_indices)
        chunk_loader = DataLoader(chunk_dataset, batch_size=32, shuffle=False)
        
        # Compute features for chunk
        chunk_features = attributor.compute_train_features(
            chunk_loader, 
            num_samples=chunk_size_actual
        )
        
        all_features.append(chunk_features.cpu())
        
        # Save intermediate results
        if save_intermediate:
            torch.save(chunk_features.cpu(), f'features_chunk_{chunk_start}_{chunk_end}.pt')
        
        # Clear GPU memory
        del chunk_features
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    return torch.cat(all_features, dim=0)
```

### Validation and Sanity Checks
```python
def validate_trak_setup(attributor, train_loader, test_loader):
    """Validate TRAK setup with sanity checks."""
    
    print("Running TRAK validation...")
    
    # 1. Check projection matrix properties
    proj_matrix = attributor.projection_matrix
    print(f"Projection matrix shape: {proj_matrix.shape}")
    print(f"Projection matrix norm: {proj_matrix.norm().item():.4f}")
    
    # 2. Test gradient computation
    try:
        sample_batch = next(iter(train_loader))
        data, targets = sample_batch[0][:1], sample_batch[1][:1]
        data, targets = data.to(attributor.device), targets.to(attributor.device)
        
        outputs = attributor.model(data)
        loss = F.cross_entropy(outputs, targets)
        grad_vector = attributor._compute_gradient_vector(loss)
        
        print(f"Gradient vector shape: {grad_vector.shape}")
        print(f"Expected shape: {attributor.total_params}")
        assert grad_vector.shape[0] == attributor.total_params
        
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        return False
    
    # 3. Test projection
    try:
        projected = attributor._project_gradient(grad_vector)
        print(f"Projected gradient shape: {projected.shape}")
        print(f"Expected shape: {attributor.proj_dim}")
        assert projected.shape[0] == attributor.proj_dim
        
    except Exception as e:
        print(f"Projection failed: {e}")
        return False
    
    # 4. Test small-scale attribution
    try:
        small_train_features = attributor.compute_train_features(
            train_loader, num_samples=10
        )
        small_test_features = attributor.compute_test_features(
            test_loader, num_samples=5
        )
        
        attributions = attributor.compute_attributions(
            small_train_features, small_test_features
        )
        
        print(f"Attribution matrix shape: {attributions.shape}")
        print(f"Expected shape: (5, 10)")
        assert attributions.shape == (5, 10)
        
    except Exception as e:
        print(f"Attribution computation failed: {e}")
        return False
    
    print("✓ All TRAK validation checks passed!")
    return True
```

## 📚 References

1. Park et al. (2023). "TRAK: Attributing Model Behavior at Scale"
2. Koh & Liang (2017). "Understanding Black-box Predictions via Influence Functions"
3. Grosse et al. (2023). "Studying Large Language Model Generalization with Influence Functions"
4. Feldman & Zhang (2020). "What Neural Networks Memorize and Why"