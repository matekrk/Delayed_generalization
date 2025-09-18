# Waterbirds Dataset

The Waterbirds dataset for studying background bias and delayed generalization in vision models.

## 📋 Overview

The Waterbirds dataset is a classic benchmark for studying simplicity bias, where models initially learn spurious correlations between bird species and background environments before eventually learning to classify based on actual bird features.

## 🔬 Phenomenon Details

### Background Bias Pattern
1. **Initial Learning (0-30 epochs)**: Model learns background-species correlation
2. **Memorization Phase (30-100 epochs)**: High accuracy on correlated examples
3. **Plateau Phase (100-200 epochs)**: Poor performance on minority groups
4. **Generalization Phase (200+ epochs)**: Model learns bird-specific features

### Dataset Structure
- **Bird Species**: Waterbirds (e.g., ducks, geese) vs Landbirds (e.g., sparrows, cardinals)
- **Backgrounds**: Water environments vs Land environments  
- **Training Bias**: 95% of waterbirds on water backgrounds, 95% of landbirds on land
- **Test Balance**: Balanced across all bird-background combinations

## 🛠️ Dataset Access

### Standard Waterbirds
The standard Waterbirds dataset can be obtained from:
- **Source**: [Official Waterbirds Dataset](https://github.com/kohpangwei/group_DRO)
- **Download**: Follow instructions in the group_DRO repository
- **License**: Check original dataset licensing terms

### Dataset Setup
```bash
# Method 1: Use our compatible generation script
# First, download the required datasets:
# - CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# - CUB-200-2011 Segmentations: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html  
# - Places365: http://places2.csail.mit.edu/download.html

# Generate Waterbirds dataset with our script
python data/vision/waterbirds/generate_waterbirds.py \
    --cub_dir ./CUB-200-2011 \
    --cub_seg_dir ./CUB-200-2011_segmentations \
    --places_dir ./places365_standard \
    --output_dir ./waterbirds_data \
    --confounder_strength 0.95

# Method 2: Download pre-generated Waterbirds dataset
git clone https://github.com/kohpangwei/group_DRO.git
cd group_DRO
python scripts/download_waterbirds.py --root_dir ./data

# The dataset will be organized as:
# waterbirds_data/
# └── waterbird_complete95_forest2water2/
#     ├── train/
#     ├── val/  
#     ├── test/
#     └── metadata.csv
```

## 📊 Dataset Statistics

### Training Set Composition
```python
training_stats = {
    "waterbird_on_water": "3498 images (56.7%)",
    "waterbird_on_land": "184 images (3.0%)", 
    "landbird_on_land": "2255 images (36.6%)",
    "landbird_on_water": "228 images (3.7%)"
}
```

### Test Set Composition  
```python
test_stats = {
    "waterbird_on_water": "1057 images (22.8%)",
    "waterbird_on_land": "1057 images (22.8%)",
    "landbird_on_land": "1057 images (22.8%)", 
    "landbird_on_water": "1057 images (22.8%)"
}
```

## 📈 Training Protocols

### Standard ERM Training
```python
model = ResNet50(
    pretrained=True,
    num_classes=2,  # Waterbird vs Landbird
    fine_tune_layers=['layer4', 'fc']
)

config = {
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "scheduler": "cosine"
}
```

### Group DRO Training
```python
# Group Distributionally Robust Optimization
from training.group_dro import GroupDROTrainer

trainer = GroupDROTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    
    # Group DRO specific parameters
    group_dro_step_size=0.01,
    robust_step_size=0.01,
    btl=False,  # Use standard Group DRO
    gamma=0.1
)

results = trainer.train(epochs=300)
```

### Other Robustness Methods
```python
# IRM (Invariant Risk Minimization)
irm_config = {
    "method": "irm",
    "irm_lambda": 1e2,
    "irm_penalty_anneal_iters": 500
}

# CORAL (Deep CORAL)
coral_config = {
    "method": "coral", 
    "coral_lambda": 1.0
}

# Mixup
mixup_config = {
    "method": "mixup",
    "mixup_alpha": 0.2
}
```

## 📊 Evaluation Metrics

### Core Metrics
- **Average Accuracy**: Overall classification accuracy
- **Worst Group Accuracy**: Performance on hardest subgroup (usually landbird on water)
- **Group Robustness**: Minimal performance across all four groups
- **Spurious Correlation**: Measure of background reliance

### Group-wise Evaluation
```python
def evaluate_groups(model, dataloader, device):
    """Evaluate performance on each bird-background group"""
    
    groups = {
        (0, 0): "waterbird_on_water",
        (0, 1): "waterbird_on_land", 
        (1, 0): "landbird_on_water",
        (1, 1): "landbird_on_land"
    }
    
    group_counts = {group: 0 for group in groups}
    group_correct = {group: 0 for group in groups}
    
    model.eval()
    with torch.no_grad():
        for data, targets, groups_batch in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            
            for i in range(len(targets)):
                group = tuple(groups_batch[i].tolist())
                group_counts[group] += 1
                if predictions[i] == targets[i]:
                    group_correct[group] += 1
    
    # Calculate group accuracies
    group_accuracies = {}
    for group, count in group_counts.items():
        if count > 0:
            group_accuracies[groups[group]] = group_correct[group] / count
    
    return group_accuracies
```

### Spurious Correlation Analysis
```python
def analyze_spurious_correlation(model, dataset):
    """Measure how much model relies on background vs bird features"""
    
    # Original performance
    original_acc = evaluate(model, dataset)
    
    # Performance with backgrounds masked
    masked_bg_dataset = mask_backgrounds(dataset)
    masked_bg_acc = evaluate(model, masked_bg_dataset)
    
    # Performance with birds masked (background only)
    masked_bird_dataset = mask_birds(dataset)
    masked_bird_acc = evaluate(model, masked_bird_dataset)
    
    # Spurious correlation score
    background_reliance = masked_bird_acc / original_acc
    bird_reliance = masked_bg_acc / original_acc
    
    return {
        "background_reliance": background_reliance,
        "bird_reliance": bird_reliance,
        "spurious_score": background_reliance - bird_reliance
    }
```

## 🎯 Expected Results

### Standard ERM Results
- **Average Accuracy**: ~85-90%
- **Worst Group Accuracy**: ~60-70% (landbird on water)
- **Group Gap**: 20-30% difference between best and worst groups

### Group DRO Results  
- **Average Accuracy**: ~80-85% (slight decrease)
- **Worst Group Accuracy**: ~75-80% (significant improvement)
- **Group Gap**: <10% difference between groups

### Learning Timeline
1. **Epochs 0-50**: Rapid learning of background correlations
2. **Epochs 50-150**: High majority group performance, poor minority groups
3. **Epochs 150-250**: Gradual improvement on minority groups (with robust methods)
4. **Epochs 250+**: Convergence to final performance levels

## 🔬 Research Applications

### Bias Mitigation Studies
```python
# Compare different debiasing methods
methods = ['erm', 'group_dro', 'irm', 'coral', 'mixup']
results = {}

for method in methods:
    model = create_model()
    trainer = create_trainer(method)
    result = trainer.train(epochs=300)
    results[method] = result

# Analyze effectiveness
analyze_bias_mitigation_effectiveness(results)
```

### Architecture Robustness Studies
```python
# Study how different architectures handle bias
architectures = ['resnet50', 'vit_base', 'efficientnet_b3', 'densenet121']

for arch in architectures:
    model = create_model(arch)
    # Train with group DRO
    results = train_robust(model, waterbirds_data)
    analyze_architecture_bias_susceptibility(arch, results)
```

### Data Augmentation Studies
```python
# Study effect of different augmentations on bias
augmentations = {
    "standard": [RandomCrop(), RandomHorizontalFlip()],
    "background_aug": [BackgroundRandomization(), ColorJitter()],
    "bird_focused": [CenterCrop(), BirdSpecificAug()],
    "mixup": [Mixup(alpha=0.2)],
    "cutmix": [CutMix(alpha=1.0)]
}

for aug_name, aug_list in augmentations.items():
    dataset_aug = apply_augmentations(waterbirds_data, aug_list)
    results = train_and_evaluate(model, dataset_aug)
    analyze_augmentation_bias_effect(aug_name, results)
```

## 🔗 Integration with Training Scripts

The training scripts in `phenomena/simplicity_bias/waterbirds/training/` are designed to work with this dataset structure:

```python
# Example integration
from phenomena.simplicity_bias.waterbirds.training.train_waterbirds import WaterbirdsTrainer

# Load dataset
train_loader, test_loader = create_waterbirds_loaders(
    data_dir="./data/waterbirds",
    batch_size=32
)

# Create trainer
trainer = WaterbirdsTrainer(
    model=model,
    train_loader=train_loader, 
    test_loader=test_loader,
    device=device,
    method="group_dro"  # or "erm"
)

# Train with bias monitoring
results = trainer.train(epochs=300, save_dir="./waterbirds_results")
```

## 📚 References

1. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts: Waterbirds Dataset"
2. Wah et al. (2011). "The Caltech-UCSD Birds-200-2011 Dataset"  
3. Beery et al. (2018). "Recognition in Terra Incognita" 
4. Arjovsky et al. (2019). "Invariant Risk Minimization"
5. Sun & Saenko (2016). "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"