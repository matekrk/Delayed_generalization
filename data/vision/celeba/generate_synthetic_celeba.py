#!/usr/bin/env python3
"""
Synthetic CelebA-like Dataset Generator for Simplicity Bias Research

This script generates synthetic face-like images with gender bias similar to CelebA,
where background features correlate with gender in training but not in test data.

Usage:
    python generate_synthetic_celeba.py --train_bias 0.8 --test_bias 0.2
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def generate_face_features(gender: int, size: int = 64) -> np.ndarray:
    """Generate basic face features (simplified geometric shapes)"""
    image = np.zeros((size, size))
    center_x, center_y = size // 2, size // 2
    
    # Face outline (oval)
    y, x = np.ogrid[:size, :size]
    face_mask = ((x - center_x) / (size * 0.35))**2 + ((y - center_y) / (size * 0.4))**2 <= 1
    image[face_mask] = 0.8
    
    # Eyes
    eye_y = center_y - size // 6
    left_eye_x = center_x - size // 6
    right_eye_x = center_x + size // 6
    eye_size = size // 12
    
    # Left eye
    eye_mask = (x - left_eye_x)**2 + (y - eye_y)**2 <= eye_size**2
    image[eye_mask] = 0.3
    
    # Right eye
    eye_mask = (x - right_eye_x)**2 + (y - eye_y)**2 <= eye_size**2
    image[eye_mask] = 0.3
    
    # Nose
    nose_y = center_y + size // 12
    nose_mask = (x - center_x)**2 + (y - nose_y)**2 <= (size // 20)**2
    image[nose_mask] = 0.6
    
    # Mouth
    mouth_y = center_y + size // 4
    mouth_width = size // 8
    mouth_mask = ((x - center_x)**2 / mouth_width**2 + (y - mouth_y)**2 / (size // 20)**2) <= 1
    image[mouth_mask] = 0.4
    
    # Gender-specific features
    if gender == 1:  # Female
        # Longer hair outline
        hair_mask = ((x - center_x) / (size * 0.45))**2 + ((y - center_y + size * 0.1) / (size * 0.5))**2 <= 1
        image[hair_mask] = np.maximum(image[hair_mask], 0.7)
        
        # Earrings (small bright spots)
        if np.random.random() > 0.3:
            left_ear_x, right_ear_x = center_x - size // 3, center_x + size // 3
            ear_y = center_y
            image[ear_y-1:ear_y+2, left_ear_x-1:left_ear_x+2] = 1.0
            image[ear_y-1:ear_y+2, right_ear_x-1:right_ear_x+2] = 1.0
    else:  # Male
        # Shorter hair
        hair_mask = ((x - center_x) / (size * 0.38))**2 + ((y - center_y + size * 0.05) / (size * 0.35))**2 <= 1
        image[hair_mask] = np.maximum(image[hair_mask], 0.5)
    
    return image


def generate_background(bg_type: str, size: int = 64) -> np.ndarray:
    """Generate different background types"""
    background = np.zeros((size, size))
    
    if bg_type == 'indoor':
        # Indoor background - vertical and horizontal lines (like walls, furniture)
        # Vertical lines
        for i in range(0, size, size//8):
            background[:, i:i+2] = 0.3
        # Horizontal lines
        for i in range(0, size, size//6):
            background[i:i+2, :] = 0.2
            
    elif bg_type == 'outdoor':
        # Outdoor background - more organic patterns (like trees, sky)
        # Random organic shapes
        for _ in range(20):
            center_x = np.random.randint(size//4, 3*size//4)
            center_y = np.random.randint(size//4, 3*size//4)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            background[mask] = np.random.uniform(0.1, 0.4)
    
    return background


class SyntheticCelebADataset(Dataset):
    """Dataset that creates synthetic face images with gender-background bias"""
    
    def __init__(
        self,
        num_samples: int,
        bias_strength: float,
        seed: int = 42,
        size: int = 64
    ):
        self.num_samples = num_samples
        self.bias_strength = bias_strength
        self.size = size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic face data with background bias"""
        self.images = []
        self.labels = []  # 0: male, 1: female
        self.background_types = []  # 0: indoor, 1: outdoor
        self.bias_followed = []
        
        for _ in range(self.num_samples):
            # Random gender
            gender = np.random.randint(0, 2)
            
            # Determine background based on bias
            if np.random.random() < self.bias_strength:
                # Biased: male -> indoor, female -> outdoor
                bg_type_idx = gender
                bias_followed = True
            else:
                # Unbiased: random background
                bg_type_idx = np.random.randint(0, 2)
                bias_followed = False
            
            bg_type = 'indoor' if bg_type_idx == 0 else 'outdoor'
            
            # Generate face and background
            face = generate_face_features(gender, self.size)
            background = generate_background(bg_type, self.size)
            
            # Combine face and background
            # Face overrides background where face intensity > 0.5
            combined = background.copy()
            face_mask = face > 0.5
            combined[face_mask] = face[face_mask]
            
            # Add some noise
            noise = np.random.normal(0, 0.05, (self.size, self.size))
            combined = np.clip(combined + noise, 0, 1)
            
            # Convert to 3-channel image
            rgb_image = torch.stack([
                torch.tensor(combined, dtype=torch.float32),
                torch.tensor(combined, dtype=torch.float32),
                torch.tensor(combined, dtype=torch.float32)
            ])
            
            self.images.append(rgb_image)
            self.labels.append(gender)
            self.background_types.append(bg_type_idx)
            self.bias_followed.append(bias_followed)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Metadata for analysis
        metadata = {
            'background_type': self.background_types[idx],
            'bias_followed': self.bias_followed[idx],
            'original_label': label
        }
        
        return image, label, metadata


def create_synthetic_celeba_datasets(
    train_bias: float = 0.8,
    test_bias: float = 0.2,
    train_size: int = 8000,
    test_size: int = 2000,
    seed: int = 42
) -> Tuple[SyntheticCelebADataset, SyntheticCelebADataset, Dict]:
    """Create synthetic CelebA-like train and test datasets"""
    
    train_dataset = SyntheticCelebADataset(
        train_size, train_bias, seed
    )
    
    test_dataset = SyntheticCelebADataset(
        test_size, test_bias, seed + 1
    )
    
    # Metadata
    metadata = {
        'train_bias': train_bias,
        'test_bias': test_bias,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'num_classes': 2,  # male, female
        'image_shape': [3, 64, 64],
        'seed': seed,
        'dataset_type': 'synthetic_celeba',
        'task': 'gender_classification',
        'spurious_feature': 'background_type'
    }
    
    return train_dataset, test_dataset, metadata


def analyze_celeba_bias(dataset: SyntheticCelebADataset, name: str) -> Dict:
    """Analyze the bias in a synthetic CelebA dataset"""
    bg_counts = {0: {}, 1: {}}  # bg_type -> {gender: count}
    bias_counts = {'followed': 0, 'violated': 0}
    
    for i in range(len(dataset)):
        _, label, metadata = dataset[i]
        bg_type = metadata['background_type']
        bias_followed = metadata['bias_followed']
        
        # Count background-gender combinations
        if label not in bg_counts[bg_type]:
            bg_counts[bg_type][label] = 0
        bg_counts[bg_type][label] += 1
        
        # Count bias following
        if bias_followed:
            bias_counts['followed'] += 1
        else:
            bias_counts['violated'] += 1
    
    # Calculate actual bias
    total_samples = len(dataset)
    actual_bias = bias_counts['followed'] / total_samples
    
    print(f"\n{name} Dataset Analysis:")
    print(f"  Total samples: {total_samples}")
    print(f"  Actual bias strength: {actual_bias:.3f}")
    print(f"  Indoor background distribution:")
    for gender, count in sorted(bg_counts[0].items()):
        gender_name = 'Male' if gender == 0 else 'Female'
        print(f"    {gender_name}: {count} samples")
    print(f"  Outdoor background distribution:")
    for gender, count in sorted(bg_counts[1].items()):
        gender_name = 'Male' if gender == 0 else 'Female'
        print(f"    {gender_name}: {count} samples")
    
    return {
        'actual_bias': actual_bias,
        'background_counts': bg_counts,
        'bias_counts': bias_counts
    }


def visualize_synthetic_celeba(dataset: SyntheticCelebADataset, num_samples: int = 16, save_path: str = None):
    """Visualize synthetic CelebA samples"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # Sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label, metadata = dataset[idx]
        
        # Convert tensor to numpy and transpose for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)
        
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(image_np, cmap='gray')
        gender = 'F' if label == 1 else 'M'
        bg = 'In' if metadata['background_type'] == 0 else 'Out'
        bias = '✓' if metadata['bias_followed'] else '✗'
        axes[row, col].set_title(f'{gender}, {bg}, {bias}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_synthetic_celeba_dataset(
    train_dataset: SyntheticCelebADataset,
    test_dataset: SyntheticCelebADataset,
    metadata: Dict,
    output_dir: str
):
    """Save synthetic CelebA dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as tensors and lists
    torch.save({
        'images': train_dataset.images,
        'labels': train_dataset.labels,
        'background_types': train_dataset.background_types,
        'bias_followed': train_dataset.bias_followed,
        'num_samples': train_dataset.num_samples,
        'bias_strength': train_dataset.bias_strength,
        'size': train_dataset.size
    }, output_path / "train_data.pt", pickle_protocol=4)
    
    torch.save({
        'images': test_dataset.images,
        'labels': test_dataset.labels,
        'background_types': test_dataset.background_types,
        'bias_followed': test_dataset.bias_followed,
        'num_samples': test_dataset.num_samples,
        'bias_strength': test_dataset.bias_strength,
        'size': test_dataset.size
    }, output_path / "test_data.pt", pickle_protocol=4)
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_synthetic_celeba(train_dataset, 16, str(output_path / "train_samples.png"))
    visualize_synthetic_celeba(test_dataset, 16, str(output_path / "test_samples.png"))
    
    print(f"Synthetic CelebA dataset saved to {output_path}")


def load_synthetic_celeba_dataset(data_dir: str) -> Tuple[SyntheticCelebADataset, SyntheticCelebADataset, Dict]:
    """Load synthetic CelebA dataset from files"""
    data_path = Path(data_dir)
    
    # Load data dictionaries
    train_data = torch.load(data_path / "train_data.pt", weights_only=False)
    test_data = torch.load(data_path / "test_data.pt", weights_only=False)
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Reconstruct datasets
    train_dataset = SyntheticCelebADataset.__new__(SyntheticCelebADataset)
    train_dataset.images = train_data['images']
    train_dataset.labels = train_data['labels']
    train_dataset.background_types = train_data['background_types']
    train_dataset.bias_followed = train_data['bias_followed']
    train_dataset.num_samples = train_data['num_samples']
    train_dataset.bias_strength = train_data['bias_strength']
    train_dataset.size = train_data['size']
    
    test_dataset = SyntheticCelebADataset.__new__(SyntheticCelebADataset)
    test_dataset.images = test_data['images']
    test_dataset.labels = test_data['labels']
    test_dataset.background_types = test_data['background_types']
    test_dataset.bias_followed = test_data['bias_followed']
    test_dataset.num_samples = test_data['num_samples']
    test_dataset.bias_strength = test_data['bias_strength']
    test_dataset.size = test_data['size']
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CelebA-like dataset for bias research")
    parser.add_argument("--train_bias", type=float, default=0.8,
                       help="Background bias strength in training set")
    parser.add_argument("--test_bias", type=float, default=0.2,
                       help="Background bias strength in test set")
    parser.add_argument("--train_size", type=int, default=8000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./synthetic_celeba_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic CelebA-like Dataset")
    print("=" * 40)
    print(f"Train bias: {args.train_bias}")
    print(f"Test bias: {args.test_bias}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_synthetic_celeba_datasets(
        train_bias=args.train_bias,
        test_bias=args.test_bias,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Analyze bias
    train_analysis = analyze_celeba_bias(train_dataset, "Training")
    test_analysis = analyze_celeba_bias(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_synthetic_celeba_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()