# CIFAR-100-C Corrupted Dataset Generation

This directory contains scripts for generating CIFAR-100-C (corrupted) datasets to study robustness and delayed generalization patterns with 100 classes.

## Scripts

### `generate_cifar100c.py`
Generates corrupted versions of the real CIFAR-100 dataset by applying various corruption types and severity levels.

**Features:**
- Applies 15 different corruption types (noise, blur, weather, digital, etc.)
- 5 severity levels for each corruption (1=light, 5=severe)
- Works with the real CIFAR-100 dataset (100 classes, 20 superclasses)
- Supports both training and test set corruption
- Generates visualizations and metadata

**Usage:**
```bash
# Generate with default corruptions
python generate_cifar100c.py --output_dir ./cifar100c_data

# Custom corruptions and severities
python generate_cifar100c.py \
    --corruptions gaussian_noise motion_blur snow brightness contrast \
    --severities 1 3 5 \
    --output_dir ./cifar100c_data
```

### `generate_synthetic_cifar100c.py`
Creates synthetic CIFAR-100-like images with corruptions for fast experimentation and prototyping.

**Features:**
- Generates distinctive synthetic objects for all 100 CIFAR-100 classes
- Organized by 20 superclasses with 5 subclasses each
- Applies corruptions during generation
- Configurable train/test splits with different corruption types
- Faster than real CIFAR-100 corruption for rapid iteration

**Usage:**
```bash
# Generate synthetic dataset
python generate_synthetic_cifar100c.py \
    --train_corruptions gaussian_noise motion_blur \
    --test_corruptions snow brightness contrast \
    --train_size 50000 --test_size 10000 \
    --output_dir ./synthetic_cifar100c_data
```

## Corruption Types

### Noise
- `gaussian_noise`: Additive Gaussian noise
- `shot_noise`: Poisson noise
- `impulse_noise`: Salt-and-pepper noise

### Blur
- `defocus_blur`: Defocus blur (simulates camera defocus)
- `glass_blur`: Glass-like distortion
- `motion_blur`: Motion blur (camera shake)
- `zoom_blur`: Zoom blur effect

### Weather
- `snow`: Snow effect
- `frost`: Frost on lens
- `fog`: Fog/haze effect

### Digital
- `brightness`: Brightness adjustment
- `contrast`: Contrast adjustment
- `elastic_transform`: Elastic deformation
- `pixelate`: Pixelation effect
- `jpeg_compression`: JPEG compression artifacts

## Integration with Training

The generated corrupted datasets are automatically used by `phenomena/robustness/cifar100c/train_cifar100c.py` when available. The training script will:

1. Load clean CIFAR-100 for training
2. Automatically detect and load corrupted test sets for evaluation
3. Evaluate robustness across all corruption types and severities
4. Track clean vs. corrupted accuracy gaps over training

## Output Structure

```
output_dir/
├── metadata.json           # Dataset metadata and corruption info
├── corruption_samples.png  # Visualization of corruption effects
└── (corrupted datasets stored in memory during training)
```

For synthetic datasets:
```
output_dir/
├── metadata.json          # Dataset metadata
├── train_samples.png      # Training samples visualization
├── test_samples.png       # Test samples visualization
├── train_dataset.pt       # Serialized training dataset
└── test_dataset.pt        # Serialized test dataset
```

## CIFAR-100 Class Structure

The CIFAR-100 dataset has 100 classes organized into 20 superclasses:

1. **Aquatic mammals**: beaver, dolphin, otter, seal, whale
2. **Fish**: aquarium fish, flatfish, ray, shark, trout
3. **Flowers**: orchid, poppy, rose, sunflower, tulip
4. **Food containers**: bottle, bowl, can, cup, plate
5. **Fruit and vegetables**: apple, mushroom, orange, pear, sweet pepper
6. **Household electrical devices**: clock, keyboard, lamp, telephone, television
7. **Household furniture**: bed, chair, couch, table, wardrobe
8. **Insects**: bee, beetle, butterfly, caterpillar, cockroach
9. **Large carnivores**: bear, leopard, lion, tiger, wolf
10. **Large man-made outdoor things**: bridge, castle, house, road, skyscraper
11. **Large natural outdoor scenes**: cloud, forest, mountain, plain, sea
12. **Large omnivores and herbivores**: camel, cattle, chimpanzee, elephant, kangaroo
13. **Medium-sized mammals**: fox, porcupine, possum, raccoon, skunk
14. **Non-insect invertebrates**: crab, lobster, snail, spider, worm
15. **People**: baby, boy, girl, man, woman
16. **Reptiles**: crocodile, dinosaur, lizard, snake, turtle
17. **Small mammals**: hamster, mouse, rabbit, shrew, squirrel
18. **Trees**: maple, oak, palm, pine, willow
19. **Vehicles 1**: bicycle, bus, motorcycle, pickup truck, train
20. **Vehicles 2**: lawn mower, rocket, streetcar, tank, tractor

This class structure allows for studying both fine-grained robustness (100 classes) and coarse-grained robustness (20 superclasses).