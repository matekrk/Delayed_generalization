# CelebA Attribute Bias Research

This directory contains implementations for studying attribute bias and delayed generalization using both real and synthetic CelebA datasets.

## 📋 Overview

The CelebA implementations allow you to study simplicity bias using face images with configurable attribute correlations. We provide both real CelebA (recommended) and synthetic CelebA implementations for different research needs.

## 🔬 Available Implementations

### 1. Real CelebA with Attribute Bias (⭐ Recommended)

Uses the actual CelebA dataset with configurable bias between any two facial attributes.

**Key Features:**
- Downloads and processes real CelebA dataset automatically  
- Configurable bias between any pair of 40 facial attributes
- Hierarchical directory structure for organized experiments
- Comprehensive bias analysis during training
- More realistic for real-world bias research

**Usage:**
```bash
# Generate dataset with Male vs Blond_Hair bias
python generate_real_celeba.py \
    --attr1 Male --attr2 Blond_Hair \
    --train_bias 0.8 --test_bias 0.2 \
    --output_dir ./real_celeba_data

# Train model with bias analysis
python ../training/train_real_celeba.py \
    --data_dir ./real_celeba_data/real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20 \
    --epochs 100 --use_wandb
```

### 2. Biased CelebA-Style (Alternative Synthetic)

Creates synthetic face-like images with controllable two-feature bias.

**Usage:**
```bash
# Generate dataset with gender vs hair_color bias
python generate_bias_celeba.py \
    --feature1 gender --feature2 hair_color \
    --train_bias 0.8 --test_bias 0.2 \
    --output_dir ./bias_celeba_data
```

### 3. Synthetic CelebA (Legacy)

Generates synthetic faces with background-gender correlation.

**Usage:**
```bash
# Generate synthetic dataset
python generate_synthetic_celeba.py \
    --train_bias 0.8 --test_bias 0.5 \
    --output_dir ./synthetic_celeba_data

# Train model
python ../training/train_celeba.py \
    --data_dir ./synthetic_celeba_data --epochs 100
```

## 🎯 Real CelebA: Attribute Combinations

### Popular Research Combinations

1. **Gender vs Hair Color** (`Male` vs `Blond_Hair`)
   - Studies gender bias in hiring, criminal justice
   - Strong real-world correlation patterns

2. **Age vs Makeup** (`Young` vs `Heavy_Makeup`)  
   - Examines age-related appearance bias
   - Relevant for age discrimination research

3. **Attractiveness vs Accessories** (`Attractive` vs `Eyeglasses`)
   - Studies beauty standards and intellectual stereotypes

### Available Attributes

**Primary Attributes (target labels):**
- `Male` - Gender classification
- `Young` - Age classification
- `Attractive` - Attractiveness rating
- `Smiling` - Facial expression

**Secondary Attributes (spurious features):**
- `Blond_Hair`, `Brown_Hair`, `Black_Hair` - Hair color
- `Heavy_Makeup`, `Wearing_Lipstick` - Makeup/cosmetics  
- `Eyeglasses`, `Wearing_Hat` - Accessories
- `High_Cheekbones`, `Pointy_Nose` - Facial features

## 📊 Dataset Structure

### Real CelebA Structure
```
output_dir/
└── real_celeba_{attr1}_{attr2}_trainbias_{X.XX}_testbias_{Y.YY}/
    ├── train_dataset.pt          # PyTorch training dataset
    ├── test_dataset.pt           # PyTorch test dataset  
    ├── metadata.json             # Dataset configuration & statistics
    ├── dataset_summary.json      # Quick summary
    ├── train_samples.png         # Sample visualizations
    └── test_samples.png          # Sample visualizations
```

## 🔍 Key Metrics & Phenomena

### Bias Analysis Metrics

1. **Overall Accuracy**: Standard classification accuracy
2. **Bias Conforming Accuracy**: Performance when spurious feature matches target
3. **Bias Conflicting Accuracy**: Performance when spurious feature conflicts  
4. **Bias Gap**: Difference between conforming and conflicting accuracy
5. **Attribute-Specific Accuracy**: Per-attribute performance breakdown

### Expected Learning Timeline

**Real CelebA Pattern:**
1. **Phase 1 (Epochs 0-20)**: Rapid spurious correlation learning
   - High bias-conforming accuracy (>80%)
   - Poor bias-conflicting accuracy (<40%)

2. **Phase 2 (Epochs 20-60)**: Spurious feature reliance  
   - Large bias gap (20-40 percentage points)
   - Overall accuracy plateaus

3. **Phase 3 (Epochs 60+)**: Potential delayed generalization
   - Gradual bias gap reduction  
   - True attribute relationship learning

### Success Indicators
- **Delayed Generalization**: Sudden bias-conflicting accuracy improvement
- **Robust Learning**: Bias gap reduces to <10 percentage points
- **Fair Performance**: Similar accuracy across attribute subgroups

## 🛠️ Advanced Usage

### Multiple Attribute Experiments
```bash
# Run systematic bias strength comparison
for bias in 0.95 0.85 0.75 0.65; do
    python generate_real_celeba.py \
        --attr1 Male --attr2 Blond_Hair \
        --train_bias $bias --test_bias 0.2 \
        --output_dir ./bias_study_$bias
done
```

### Attribute Combination Study
```bash
# Compare different attribute pairs
python generate_real_celeba.py --attr1 Male --attr2 Blond_Hair --output_dir ./male_blonde
python generate_real_celeba.py --attr1 Young --attr2 Heavy_Makeup --output_dir ./young_makeup  
python generate_real_celeba.py --attr1 Attractive --attr2 Eyeglasses --output_dir ./attractive_glasses
```

## 🔬 Research Applications

### Fairness and Bias Research
- Study demographic bias in face recognition systems
- Develop bias mitigation techniques  
- Evaluate fairness across protected attributes
- Test algorithmic discrimination detection

### Delayed Generalization Studies
- Identify when models overcome spurious correlations
- Study role of dataset size and bias strength
- Examine architectural influences on bias learning
- Compare with other delayed generalization phenomena

### Real-World Relevance  
- More realistic than purely synthetic datasets
- Connects to actual fairness problems in AI systems
- Validates findings on real image distributions
- Bridges lab research and practical applications

## 📈 Comparison: Real vs Synthetic

### Real CelebA Advantages
- **Realistic Images**: Actual face photos with natural variation
- **Complex Features**: Rich facial attribute relationships  
- **Research Validity**: Direct relevance to real-world bias
- **Attribute Diversity**: 40 different facial attributes available

### Synthetic CelebA Advantages  
- **Controlled Environment**: Precise bias control
- **Computational Efficiency**: Faster generation and training
- **Privacy**: No real faces, avoiding ethical concerns
- **Educational**: Clear geometric patterns for understanding

## 📚 Files Overview

- `generate_real_celeba.py` - **Main implementation** for real CelebA with bias
- `generate_bias_celeba.py` - Alternative synthetic with two-feature bias
- `generate_synthetic_celeba.py` - Legacy synthetic with background bias  
- `training/train_real_celeba.py` - Training script for real CelebA
- `training/train_celeba.py` - Training script for synthetic CelebA

## ⚙️ Requirements

- PyTorch >= 1.12.0
- torchvision >= 0.13.0  
- Sufficient disk space for CelebA dataset (~1.3GB)
- CUDA-compatible GPU recommended for training
- Internet connection for initial CelebA download

## 💡 Quick Start Tips

1. **Start Small**: Use `--train_size 1000 --test_size 200` for quick testing
2. **Monitor Bias Gap**: Key metric for delayed generalization research  
3. **Try Multiple Attributes**: Different combinations show different patterns
4. **Use Wandb**: Add `--use_wandb` for comprehensive experiment tracking
5. **Save Experiments**: Hierarchical structure organizes multiple runs

## 🔗 Integration Example

```python
from generate_real_celeba import load_real_celeba_dataset

# Load pre-generated dataset
train_dataset, test_dataset, metadata = load_real_celeba_dataset(
    "./real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20"
)

# Access bias information
for image, label, metadata in train_dataset:
    attr1_value = metadata['attr1']  # Male (0/1)  
    attr2_value = metadata['attr2']  # Blond_Hair (0/1)
    bias_followed = metadata['bias_followed']  # True/False
    break
```