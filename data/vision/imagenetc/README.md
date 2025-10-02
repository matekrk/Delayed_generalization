# ImageNet-C Dataset

A corruption benchmark for studying delayed generalization in robustness scenarios on ImageNet scale.

## 📋 Overview

This ImageNet-C dataset creates corrupted versions of ImageNet images to study how models develop robustness through delayed generalization patterns at larger scale.

## 🔬 Phenomenon Details

### Robustness Delayed Generalization
1. **Clean Learning (0-20 epochs)**: Model learns on uncorrupted images
2. **Corruption Introduction (20-60 epochs)**: Gradual corruption exposure
3. **Adaptation Phase (60-120 epochs)**: Model learns corruption-invariant features
4. **Robust Generalization (120+ epochs)**: Strong performance across corruption types

### Key Characteristics
- **Scale**: ImageNet-scale (1000 classes, 224x224 images)
- **Corruption Types**: Noise, blur, weather effects, digital corruptions
- **Severity Levels**: 1-5 scale (mild to severe)
- **Progressive Training**: Gradual introduction of corruptions
- **Evaluation**: Performance across different corruption types and severities

## 🛠️ Dataset Generation

### Basic Usage
```bash
python generate_imagenetc.py \
    --data_dir ./imagenet \
    --corruptions gaussian_noise defocus_blur brightness \
    --severity 3 \
    --output_dir ./imagenetc_data \
    --split val
```

### All Corruption Types
```bash
python generate_imagenetc.py \
    --data_dir ./imagenet \
    --corruptions gaussian_noise shot_noise impulse_noise \
                  defocus_blur glass_blur motion_blur zoom_blur \
                  snow frost fog brightness contrast \
                  elastic_transform pixelate jpeg_compression \
    --severity 3 \
    --output_dir ./imagenetc_full \
    --split val
```

### Different Severity Levels
```bash
# Light corruptions
python generate_imagenetc.py \
    --data_dir ./imagenet \
    --corruptions gaussian_noise defocus_blur \
    --severity 1 \
    --output_dir ./imagenetc_light

# Heavy corruptions
python generate_imagenetc.py \
    --data_dir ./imagenet \
    --corruptions gaussian_noise defocus_blur \
    --severity 5 \
    --output_dir ./imagenetc_heavy
```

## 📊 Corruption Categories

### 1. Noise Corruptions
```python
noise_corruptions = {
    "gaussian_noise": "Additive Gaussian noise",
    "shot_noise": "Poisson noise simulation", 
    "impulse_noise": "Salt and pepper noise"
}
```

### 2. Blur Corruptions
```python
blur_corruptions = {
    "defocus_blur": "Out-of-focus blur",
    "glass_blur": "Glass distortion blur",
    "motion_blur": "Camera motion blur",
    "zoom_blur": "Zoom-induced blur"
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
    "elastic_transform": "Elastic distortion",
    "pixelate": "Pixelation effect",
    "jpeg_compression": "JPEG compression artifacts"
}
```

## 📈 Training Protocols

### Standard Robustness Training
```python
from phenomena.robustness.imagenetc.train_imagenetc import train_imagenetc

model = create_imagenet_robustness_model(
    model_type='resnet50',
    num_classes=1000
)

config = {
    "epochs": 120,
    "batch_size": 256,
    "learning_rate": 0.1,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    
    # Robustness-specific
    "corruption_schedule": "progressive",
    "max_corruption_ratio": 0.5
}
```

### Progressive Corruption Training
```python
# Start with clean data, gradually introduce corruptions
scheduler = ProgressiveCorruptionScheduler(
    start_epoch=20,
    end_epoch=60,
    final_ratio=0.5
)

for epoch in range(epochs):
    corruption_ratio = scheduler.get_ratio(epoch)
    train_epoch(model, train_loader, corruption_ratio)
```

## 📊 Evaluation Protocols

### Corruption Robustness Evaluation
```python
def evaluate_corruption_error(model, test_data, baseline_acc=0.0):
    """Calculate Corruption Error (CE)"""
    ces = []
    
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            test_loader = load_corrupted_test(corruption, severity)
            accuracy = evaluate(model, test_loader)
            ce = (1 - accuracy) / (1 - baseline_acc)
            ces.append(ce)
    
    return np.mean(ces)
```

### Mean Corruption Error (mCE)
```python
# Calculate mCE across all corruptions
mce = calculate_mean_corruption_error(
    model, 
    corruption_types=ALL_CORRUPTIONS,
    severities=[1, 2, 3, 4, 5],
    baseline_model='resnet50'
)
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

## 🔗 Integration Example

```python
from data.vision.imagenetc.generate_imagenetc import load_imagenetc_dataset

# Load corrupted dataset
dataset, metadata = load_imagenetc_dataset('./imagenetc_data/gaussian_noise_severity_3')

# Create data loader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

# Training with robustness monitoring
from phenomena.robustness.imagenetc.train_imagenetc import ImageNetCTrainer

trainer = ImageNetCTrainer(model, train_loader, test_loader, device)
results = trainer.train(
    epochs=120,
    monitor_robustness=True,
    save_corruption_analysis=True
)

# Evaluate final robustness
final_mce = evaluate_corruption_robustness(model, test_dataset)
print(f"Final mCE: {final_mce:.3f}")
```

## 📦 Dataset Structure

```
imagenetc_data/
├── imagenetc/
│   ├── gaussian_noise_severity_3/
│   │   └── dataset.pt
│   ├── defocus_blur_severity_3/
│   │   └── dataset.pt
│   ├── brightness_severity_3/
│   │   └── dataset.pt
│   ├── metadata.json
│   └── corruptions.png
```

## 🎯 Expected Performance Patterns

### Clean vs Corrupted Accuracy
- **Early Training (0-20 epochs)**: High clean accuracy, low corrupted accuracy
- **Mid Training (20-60 epochs)**: Accuracy gap narrows
- **Late Training (60+ epochs)**: Robust performance on both clean and corrupted

### Robustness Gap
```
Robustness Gap = Clean Accuracy - Average Corrupted Accuracy

Expected progression:
Epoch 0-20:   ~40% gap (model memorizes clean data)
Epoch 20-60:  ~20% gap (adaptation phase)
Epoch 60+:    ~5-10% gap (robust generalization)
```

## 📝 Notes

- ImageNet must be manually downloaded from http://www.image-net.org/
- Dataset generation can be memory intensive for large corruption sets
- Recommend using validation set for initial experiments
- Training set generation requires significant storage (~150GB per corruption type)

## 🔍 Research Applications

1. **Robustness Analysis**: Study how models develop corruption-invariant features
2. **Transfer Learning**: Analyze robustness transfer across domains
3. **Architecture Comparison**: Compare robustness of different architectures at scale
4. **Training Dynamics**: Study delayed generalization patterns in robustness

## 📚 References

- Hendrycks & Dietterich (2019): "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"
- Original CIFAR-10-C and CIFAR-100-C papers and implementations
- ImageNet-C benchmark documentation
