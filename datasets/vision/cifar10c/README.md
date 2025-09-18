# CIFAR-10-C Synthetic Dataset

A synthetic corruption benchmark for studying delayed generalization in robustness scenarios.

## 📋 Overview

This synthetic CIFAR-10-C dataset creates corrupted versions of CIFAR-10-like images to study how models develop robustness through delayed generalization patterns.

## 🔬 Phenomenon Details

### Robustness Delayed Generalization
1. **Clean Learning (0-50 epochs)**: Model learns on uncorrupted images
2. **Corruption Introduction (50-150 epochs)**: Gradual corruption exposure
3. **Adaptation Phase (150-300 epochs)**: Model learns corruption-invariant features
4. **Robust Generalization (300+ epochs)**: Strong performance across corruption types

### Key Characteristics
- **Corruption Types**: Noise, blur, weather effects, digital corruptions
- **Severity Levels**: 1-5 scale (mild to severe)
- **Progressive Training**: Gradual introduction of corruptions
- **Evaluation**: Performance across different corruption types and severities

## 🛠️ Dataset Generation

### Basic Usage
```bash
python generate_synthetic_cifar10c.py \
    --corruption_types noise blur weather \
    --severity_levels 1 2 3 4 5 \
    --num_samples 10000 \
    --output_dir ./cifar10c_data
```

### Specific Corruption Types
```bash
# Noise corruptions
python generate_synthetic_cifar10c.py \
    --corruption_types gaussian_noise shot_noise impulse_noise \
    --severity_levels 1 3 5 \
    --output_dir ./noise_corruptions

# Blur corruptions  
python generate_synthetic_cifar10c.py \
    --corruption_types motion_blur zoom_blur defocus_blur \
    --severity_levels 2 4 \
    --output_dir ./blur_corruptions

# Weather corruptions
python generate_synthetic_cifar10c.py \
    --corruption_types snow frost fog brightness \
    --severity_levels 1 2 3 4 5 \
    --output_dir ./weather_corruptions
```

### Progressive Corruption Training
```bash
python generate_synthetic_cifar10c.py \
    --progressive_training \
    --start_clean_epochs 50 \
    --corruption_ramp_epochs 100 \
    --final_corruption_mix 0.5 \
    --output_dir ./progressive_cifar10c
```

## 📊 Corruption Categories

### 1. Noise Corruptions
```python
noise_corruptions = {
    "gaussian_noise": "Additive Gaussian noise",
    "shot_noise": "Poisson noise simulation", 
    "impulse_noise": "Salt and pepper noise",
    "speckle_noise": "Multiplicative noise"
}
```

### 2. Blur Corruptions
```python
blur_corruptions = {
    "defocus_blur": "Out-of-focus blur",
    "motion_blur": "Camera motion blur",
    "zoom_blur": "Zoom-induced blur",
    "gaussian_blur": "Standard Gaussian blur"
}
```

### 3. Weather Corruptions
```python
weather_corruptions = {
    "snow": "Snow overlay effect",
    "frost": "Frost pattern overlay",
    "fog": "Fog/haze effect", 
    "brightness": "Brightness variations",
    "contrast": "Contrast modifications"
}
```

### 4. Digital Corruptions
```python
digital_corruptions = {
    "jpeg_compression": "JPEG compression artifacts",
    "pixelate": "Pixelation effect",
    "elastic_transform": "Elastic deformations",
    "saturate": "Color saturation changes"
}
```

## 📈 Training Protocols

### Standard Robustness Training
```python
model = RobustCNN(
    input_channels=3,
    num_classes=10,
    backbone='resnet18',
    dropout=0.1
)

config = {
    "epochs": 400,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    
    # Robustness-specific
    "corruption_schedule": "progressive",
    "max_corruption_ratio": 0.5
}
```

### Progressive Robustness Training
```python
class ProgressiveCorruptionTrainer:
    def __init__(self, model, clean_data, corrupted_data):
        self.model = model
        self.clean_data = clean_data
        self.corrupted_data = corrupted_data
        
    def get_epoch_data(self, epoch, total_epochs):
        # Start with clean data, gradually add corruptions
        if epoch < total_epochs * 0.2:
            return self.clean_data
        else:
            corruption_ratio = min(
                (epoch - total_epochs * 0.2) / (total_epochs * 0.6),
                1.0
            )
            return self.mix_data(corruption_ratio)
```

### Adversarial Robustness Training
```python
# Combine corruption robustness with adversarial training
config = {
    "corruption_training": True,
    "adversarial_training": True,
    "adversarial_eps": 8/255,
    "corruption_severity": [1, 2, 3],
    "training_mix": {
        "clean": 0.4,
        "corrupted": 0.4, 
        "adversarial": 0.2
    }
}
```

## 📊 Evaluation Protocols

### Corruption Robustness Evaluation
```python
def evaluate_corruption_robustness(model, test_data):
    results = {}
    
    for corruption_type in corruption_types:
        for severity in [1, 2, 3, 4, 5]:
            corrupted_data = apply_corruption(test_data, corruption_type, severity)
            accuracy = evaluate(model, corrupted_data)
            results[f"{corruption_type}_severity_{severity}"] = accuracy
    
    # Mean corruption error (mCE)
    mean_ce = calculate_mean_corruption_error(results)
    return results, mean_ce

def calculate_mean_corruption_error(results, baseline_results=None):
    """Calculate mCE relative to baseline model"""
    if baseline_results is None:
        # Use AlexNet as baseline (standard practice)
        baseline_results = load_alexnet_cifar10c_results()
    
    ces = []
    for corruption_severity, accuracy in results.items():
        baseline_acc = baseline_results[corruption_severity]
        ce = (1 - accuracy) / (1 - baseline_acc)
        ces.append(ce)
    
    return np.mean(ces)
```

### Progressive Evaluation
```python
def evaluate_robustness_development(model_checkpoints, test_data):
    """Track robustness development over training"""
    results = []
    
    for epoch, checkpoint in model_checkpoints.items():
        model.load_state_dict(checkpoint)
        
        # Evaluate on different corruption severities
        clean_acc = evaluate(model, clean_test_data)
        mild_corruption_acc = evaluate(model, mild_corrupted_data)
        severe_corruption_acc = evaluate(model, severe_corrupted_data)
        
        results.append({
            "epoch": epoch,
            "clean_accuracy": clean_acc,
            "mild_corruption_accuracy": mild_corruption_acc,
            "severe_corruption_accuracy": severe_corruption_acc,
            "robustness_gap": clean_acc - severe_corruption_acc
        })
    
    return results
```

## 🎯 Expected Results

### Learning Progression
1. **Phase 1 (0-50 epochs)**: High clean accuracy, poor corruption robustness
2. **Phase 2 (50-150 epochs)**: Gradual adaptation to mild corruptions
3. **Phase 3 (150-300 epochs)**: Developing robust features
4. **Phase 4 (300+ epochs)**: Strong performance across corruption types

### Performance Targets
- **Clean Accuracy**: >90% (comparable to standard CIFAR-10)
- **mCE (Mean Corruption Error)**: <0.8 (better than baseline)
- **Severe Corruption Accuracy**: >60% (robust to heavy corruptions)
- **Corruption Consistency**: <15% variance across corruption types

## 🔬 Research Applications

### Robustness Analysis
```python
# Analyze which features contribute to robustness
def analyze_robust_features(model, clean_images, corrupted_images):
    # Feature importance analysis
    clean_features = model.extract_features(clean_images)
    corrupted_features = model.extract_features(corrupted_images)
    
    # Measure feature stability across corruptions
    feature_stability = compute_feature_stability(clean_features, corrupted_features)
    
    # Identify corruption-invariant features
    invariant_features = identify_invariant_features(feature_stability)
    
    return feature_stability, invariant_features

# Study corruption transfer
def study_corruption_transfer(model, source_corruption, target_corruption):
    """Does robustness to one corruption transfer to others?"""
    
    # Train on source corruption
    source_model = train_on_corruption(model, source_corruption)
    
    # Evaluate on target corruption
    target_performance = evaluate(source_model, target_corruption)
    
    return target_performance
```

### Architecture Impact Studies
```python
# Compare different architectures' robustness development
architectures = ['resnet18', 'vgg16', 'mobilenet', 'efficientnet']

for arch in architectures:
    model = create_model(arch, num_classes=10)
    
    # Train with progressive corruption
    trainer = ProgressiveCorruptionTrainer(model)
    results = trainer.train(epochs=400)
    
    # Analyze robustness development timeline
    analyze_robustness_timeline(results, arch)
```

## 📊 Comparison with Real CIFAR-10-C

### Advantages of Synthetic Version
- **Controlled Corruption**: Precise control over corruption parameters
- **Scalability**: Generate unlimited corrupted samples
- **Reproducibility**: Consistent corruption application
- **Cost**: No need to download large corruption datasets

### Synthetic vs Real Corruptions
```python
# Validate synthetic corruptions against real ones
def validate_synthetic_corruptions(synthetic_data, real_data):
    # Compare corruption effects
    synthetic_difficulty = measure_corruption_difficulty(synthetic_data)
    real_difficulty = measure_corruption_difficulty(real_data)
    
    # Correlation analysis
    correlation = np.corrcoef(synthetic_difficulty, real_difficulty)[0, 1]
    
    return correlation
```

## 🔗 Integration Example

```python
from datasets.vision.cifar10c.generate_synthetic_cifar10c import SyntheticCIFAR10C

# Create progressive corruption dataset
dataset = SyntheticCIFAR10C(
    base_dataset='cifar10',
    corruption_types=['noise', 'blur', 'weather'],
    severity_range=(1, 5),
    progressive_training=True
)

# Training with robustness monitoring
trainer = RobustnessTrainer(model, dataset)
results = trainer.train(
    epochs=400,
    monitor_robustness=True,
    save_corruption_analysis=True
)

# Evaluate final robustness
final_mce = evaluate_corruption_robustness(model, test_dataset)
print(f"Final mCE: {final_mce:.3f}")
```

## 📚 References

1. Hendrycks & Dietterich (2019). "Benchmarking Neural Network Robustness to Common Corruptions"
2. Rusak et al. (2020). "A Simple Way to Make Neural Networks Robust Against Diverse Image Corruptions"
3. Schneider et al. (2020). "Improving robustness against common corruptions by covariate shift adaptation"
4. Geirhos et al. (2020). "Shortcut learning in deep neural networks"