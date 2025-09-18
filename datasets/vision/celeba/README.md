# Synthetic CelebA Dataset  

A synthetic face dataset for studying gender bias and simplicity bias in attribute prediction.

## 📋 Overview

This synthetic CelebA-like dataset creates face images where background features correlate with gender in training but not test data, demonstrating delayed generalization through simplicity bias.

## 🔬 Phenomenon Details

### Simplicity Bias in Demographics
1. **Initial Learning (0-20 epochs)**: Model learns background-gender correlation
2. **Memorization Phase (20-100 epochs)**: High training accuracy, poor test performance
3. **Transition Phase (100-150 epochs)**: Model begins learning facial features
4. **Generalization (150+ epochs)**: Robust gender prediction based on facial features

### Key Characteristics
- **Training Bias**: Strong correlation between background and gender (e.g., 80%)
- **Test Bias**: Weak or reversed correlation (e.g., 20%)
- **Challenge**: Learn facial features while ignoring background spurious correlations

## 🛠️ Dataset Generation

### Basic Usage
```bash
python generate_synthetic_celeba.py \
    --train_bias 0.8 \
    --test_bias 0.2 \
    --num_train 10000 \
    --num_test 2000 \
    --output_dir ./synthetic_celeba_data
```

### High Bias Configuration
```bash
python generate_synthetic_celeba.py \
    --train_bias 0.95 \
    --test_bias 0.05 \
    --background_complexity high \
    --facial_features detailed \
    --image_size 128 \
    --output_dir ./high_bias_celeba
```

### Balanced Configuration  
```bash
python generate_synthetic_celeba.py \
    --train_bias 0.6 \
    --test_bias 0.4 \
    --background_complexity medium \
    --noise_level 0.1 \
    --output_dir ./balanced_celeba
```

## 📊 Dataset Structure

### Image Features
- **Face Shape**: Geometric face outlines (oval for female, square for male)
- **Facial Features**: Eyes, nose, mouth with gender-specific variations
- **Background**: Patterns that correlate with gender in training
- **Size**: Configurable (default 64x64, supports up to 128x128)

### Bias Configurations
```python
# Strong bias configuration
strong_bias = {
    "train_bias": 0.9,        # 90% correlation in training
    "test_bias": 0.1,         # 10% correlation in test  
    "background_types": ["floral", "geometric", "textured"],
    "background_complexity": "high"
}

# Medium bias configuration  
medium_bias = {
    "train_bias": 0.7,
    "test_bias": 0.3,
    "background_types": ["simple", "pattern"],
    "background_complexity": "medium"
}

# Weak bias configuration
weak_bias = {
    "train_bias": 0.6,
    "test_bias": 0.4,
    "background_types": ["minimal"],
    "background_complexity": "low"
}
```

## 📈 Training Protocols

### Standard Training
```python
model = FaceClassifier(
    input_size=64,
    num_classes=2,  # Male/Female
    backbone='resnet18',
    pretrained=False
)

config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "scheduler": "cosine"
}
```

### Bias-Aware Training
```python
model = RobustFaceClassifier(
    input_size=64,
    num_classes=2,
    backbone='resnet18',
    dropout=0.3,
    attention_mechanism=True  # Focus on facial features
)

config = {
    "method": "group_dro",
    "epochs": 300,
    "batch_size": 16,  # Smaller for robustness
    "learning_rate": 5e-4,
    "data_augmentation": {
        "background_randomization": True,
        "color_jitter": True,
        "random_crop": True
    }
}
```

## 📊 Evaluation Metrics

### Core Metrics
- **Overall Accuracy**: Binary gender classification accuracy
- **Background Robustness**: Performance when backgrounds randomized
- **Group Fairness**: Performance across gender-background combinations
- **Feature Attribution**: Grad-CAM analysis of important regions

### Group Analysis
```python
groups = {
    "male_floral_bg": subset,
    "male_geometric_bg": subset,
    "female_floral_bg": subset,
    "female_geometric_bg": subset
}

# Track worst-group performance
worst_group_acc = min([
    evaluate_group(model, group_data) 
    for group_data in groups.values()
])
```

### Bias Detection
```python
# Measure reliance on background vs facial features
def measure_bias_reliance(model, dataset):
    # Original images
    orig_acc = evaluate(model, dataset)
    
    # Background-randomized images  
    randomized_dataset = randomize_backgrounds(dataset)
    random_bg_acc = evaluate(model, randomized_dataset)
    
    # Bias reliance score (higher = more biased)
    bias_score = (orig_acc - random_bg_acc) / orig_acc
    return bias_score
```

## 🎯 Expected Results

### Learning Progression
1. **Epochs 0-30**: Rapid learning of background-gender correlation
2. **Epochs 30-80**: Plateau in test performance despite high training accuracy
3. **Epochs 80-150**: Gradual learning of facial features
4. **Epochs 150+**: Robust performance independent of background

### Performance Targets
- **Final Test Accuracy**: >90% on unbiased test set
- **Background Robustness**: <5% drop when backgrounds randomized
- **Group Fairness**: <10% gap between best and worst performing groups

## 🔬 Research Applications

### Bias Mitigation Techniques
```python
# Background augmentation
class BackgroundAugmentation:
    def __call__(self, image):
        # Replace background with random texture
        face_mask = detect_face_region(image)
        background = generate_random_background()
        return composite_image(image, background, face_mask)

# Adversarial debiasing
class AdversarialDebiasing:
    def __init__(self, main_classifier, bias_classifier):
        self.main_clf = main_classifier
        self.bias_clf = bias_classifier
    
    def forward(self, x):
        features = self.main_clf.extract_features(x)
        gender_pred = self.main_clf.classify(features)
        
        # Adversarial loss to prevent background bias
        bias_pred = self.bias_clf(features)
        return gender_pred, bias_pred
```

### Attention Analysis
```python
# Analyze what regions the model focuses on
def analyze_attention(model, images):
    grad_cam = GradCAM(model, target_layer='layer4')
    
    attention_maps = []
    for image in images:
        cam = grad_cam(image)
        attention_maps.append(cam)
    
    # Measure attention on face vs background regions
    face_attention = measure_face_attention(attention_maps, images)
    bg_attention = measure_background_attention(attention_maps, images)
    
    return face_attention, bg_attention
```

## 📊 Comparison with Real CelebA

### Advantages of Synthetic Version
- **Controlled Bias**: Precise control over correlation strength
- **Ground Truth**: Known facial features and background elements
- **Scalability**: Generate unlimited samples with different bias levels
- **Privacy**: No real faces, avoiding privacy concerns

### Limitations
- **Simplicity**: Less complex than real faces
- **Domain Gap**: May not transfer to real-world scenarios
- **Feature Richness**: Limited facial feature diversity

## 🔗 Integration Example

```python
from datasets.vision.celeba.generate_synthetic_celeba import SyntheticCelebADataset

# Create dataset with different bias levels
train_dataset = SyntheticCelebADataset(
    num_samples=10000,
    bias_strength=0.8,
    image_size=64,
    background_complexity='high'
)

test_dataset = SyntheticCelebADataset(
    num_samples=2000,
    bias_strength=0.2,  # Reduced bias
    image_size=64,
    background_complexity='high',
    seed=train_dataset.seed + 1000  # Different but reproducible
)

# Training with bias monitoring
trainer = BiasAwareTrainer(model, train_dataset, test_dataset)
results = trainer.train(
    epochs=200,
    monitor_bias=True,
    save_attention_maps=True
)
```

## 📚 References

1. Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
2. Kim et al. (2019). "Learning Not to Learn: Training DNNs with Biased Data"
3. Wang et al. (2020). "Towards Fairness in Visual Recognition"
4. Hendricks et al. (2018). "Women also Snowboard: Overcoming Bias in Captioning Models"