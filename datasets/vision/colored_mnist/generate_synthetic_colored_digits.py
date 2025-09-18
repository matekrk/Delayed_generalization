#!/usr/bin/env python3
"""
Synthetic Colored MNIST Dataset Generator for Simplicity Bias Research

This script generates synthetic colored digit-like datasets where color correlates 
with class label in training data but not in test data, to study simplicity bias.
Since we don't have internet access, this creates synthetic digit-like patterns.

Usage:
    python generate_synthetic_colored_digits.py --train_correlation 0.9 --test_correlation 0.1
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def create_synthetic_digit(digit_class: int, size: int = 28) -> np.ndarray:
    """Create a synthetic digit-like pattern"""
    image = np.zeros((size, size))
    center = size // 2
    
    if digit_class == 0:  # Circle
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 3) ** 2
        image[mask] = 1.0
        # Add inner circle (hollow)
        inner_mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 5) ** 2
        image[inner_mask] = 0.0
        
    elif digit_class == 1:  # Vertical line
        image[:, center-2:center+3] = 1.0
        
    elif digit_class == 2:  # Horizontal lines (steps)
        image[size//4:size//4+3, :] = 1.0
        image[center:center+3, :] = 1.0
        image[3*size//4:3*size//4+3, :] = 1.0
        
    elif digit_class == 3:  # Right curves
        image[:size//2, center:] = 1.0
        image[center:, center:] = 1.0
        image[size//4:3*size//4, center+size//4:] = 0.0
        
    elif digit_class == 4:  # Y shape
        image[:center, center-2:center+3] = 1.0
        image[center:, :center] = 1.0
        image[center:, center:] = 1.0
        
    elif digit_class == 5:  # S shape
        image[:size//3, :2*size//3] = 1.0
        image[size//3:2*size//3, size//3:] = 1.0
        image[2*size//3:, size//3:] = 1.0
        
    elif digit_class == 6:  # P shape
        image[:, :size//2] = 1.0
        image[:size//2, :] = 1.0
        image[size//4:size//2, size//2:3*size//4] = 1.0
        
    elif digit_class == 7:  # Triangle
        for i in range(size):
            width = int((i / size) * size)
            if width > 0:
                start = center - width // 2
                end = center + width // 2
                image[i, start:end] = 1.0
        
    elif digit_class == 8:  # Double circle
        y, x = np.ogrid[:size, :size]
        # Upper circle
        mask1 = (x - center) ** 2 + (y - center//2) ** 2 <= (size // 5) ** 2
        # Lower circle
        mask2 = (x - center) ** 2 + (y - 3*center//2) ** 2 <= (size // 5) ** 2
        image[mask1 | mask2] = 1.0
        
    elif digit_class == 9:  # Spiral
        for angle in np.linspace(0, 4*np.pi, 200):
            r = angle * size // (8*np.pi)
            x = int(center + r * np.cos(angle))
            y = int(center + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                image[y, x] = 1.0
    
    # Add some noise
    noise = np.random.normal(0, 0.1, (size, size))
    image = np.clip(image + noise, 0, 1)
    
    return image


class SyntheticColoredDataset(Dataset):
    """Dataset that creates synthetic colored digits with correlation"""
    
    def __init__(
        self,
        num_samples: int,
        correlation: float,
        colors: List[Tuple[float, float, float]] = None,
        seed: int = 42,
        size: int = 28
    ):
        self.num_samples = num_samples
        self.correlation = correlation
        self.size = size
        
        # Default color scheme: red and green
        if colors is None:
            self.colors = [
                (1.0, 0.0, 0.0),  # Red
                (0.0, 1.0, 0.0),  # Green
            ]
        else:
            self.colors = colors
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic colored digit data"""
        self.images = []
        self.labels = []
        self.color_assignments = []
        self.correlation_followed = []
        
        for _ in range(self.num_samples):
            # Random digit class
            label = np.random.randint(0, 10)
            
            # Generate synthetic digit
            digit_image = create_synthetic_digit(label, self.size)
            
            # Determine color based on correlation
            if np.random.random() < self.correlation:
                # Correlated: even digits -> color 0, odd digits -> color 1
                color_idx = label % 2
                corr_followed = True
            else:
                # Anti-correlated: random color assignment
                color_idx = np.random.randint(0, len(self.colors))
                corr_followed = False
            
            # Apply color to image
            color = torch.tensor(self.colors[color_idx], dtype=torch.float32)
            
            # Convert to RGB and apply color
            rgb_image = torch.stack([
                torch.tensor(digit_image, dtype=torch.float32) * color[0],
                torch.tensor(digit_image, dtype=torch.float32) * color[1],
                torch.tensor(digit_image, dtype=torch.float32) * color[2]
            ])
            
            self.images.append(rgb_image)
            self.labels.append(label)
            self.color_assignments.append(color_idx)
            self.correlation_followed.append(corr_followed)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Metadata for analysis
        metadata = {
            'color_idx': self.color_assignments[idx],
            'correlation_followed': self.correlation_followed[idx],
            'original_label': label
        }
        
        return image, label, metadata


def create_synthetic_colored_datasets(
    train_correlation: float = 0.9,
    test_correlation: float = 0.1,
    train_size: int = 12000,
    test_size: int = 2000,
    colors: List[Tuple[float, float, float]] = None,
    seed: int = 42
) -> Tuple[SyntheticColoredDataset, SyntheticColoredDataset, Dict]:
    """
    Create synthetic colored digit train and test datasets
    """
    
    train_dataset = SyntheticColoredDataset(
        train_size, train_correlation, colors, seed
    )
    
    test_dataset = SyntheticColoredDataset(
        test_size, test_correlation, colors, seed + 1
    )
    
    # Metadata
    metadata = {
        'train_correlation': train_correlation,
        'test_correlation': test_correlation,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'colors': colors or [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        'num_classes': 10,
        'image_shape': [3, 28, 28],
        'seed': seed,
        'dataset_type': 'synthetic'
    }
    
    return train_dataset, test_dataset, metadata


def analyze_dataset_bias(dataset: SyntheticColoredDataset, name: str) -> Dict:
    """Analyze the bias in a synthetic colored dataset"""
    color_counts = {0: {}, 1: {}}
    correlation_counts = {'followed': 0, 'violated': 0}
    
    for i in range(len(dataset)):
        _, label, metadata = dataset[i]
        color_idx = metadata['color_idx']
        correlation_followed = metadata['correlation_followed']
        
        # Count color-label combinations
        if label not in color_counts[color_idx]:
            color_counts[color_idx][label] = 0
        color_counts[color_idx][label] += 1
        
        # Count correlation following
        if correlation_followed:
            correlation_counts['followed'] += 1
        else:
            correlation_counts['violated'] += 1
    
    # Calculate actual correlation
    total_samples = len(dataset)
    actual_correlation = correlation_counts['followed'] / total_samples
    
    print(f"\n{name} Dataset Analysis:")
    print(f"  Total samples: {total_samples}")
    print(f"  Actual correlation: {actual_correlation:.3f}")
    print(f"  Color 0 (Red) distribution:")
    for label, count in sorted(color_counts[0].items()):
        print(f"    Digit {label}: {count} samples")
    print(f"  Color 1 (Green) distribution:")
    for label, count in sorted(color_counts[1].items()):
        print(f"    Digit {label}: {count} samples")
    
    return {
        'actual_correlation': actual_correlation,
        'color_counts': color_counts,
        'correlation_counts': correlation_counts
    }


def visualize_synthetic_dataset(dataset: SyntheticColoredDataset, num_samples: int = 20, save_path: str = None):
    """Visualize synthetic colored digit samples"""
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    
    # Sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label, metadata = dataset[idx]
        
        # Convert tensor to numpy and transpose for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)
        
        row = i // 10
        col = i % 10
        
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'L:{label}, C:{metadata["color_idx"]}', fontsize=8)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_synthetic_dataset(
    train_dataset: SyntheticColoredDataset,
    test_dataset: SyntheticColoredDataset,
    metadata: Dict,
    output_dir: str
):
    """Save synthetic colored dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as tensors and lists instead of pickle
    torch.save({
        'images': train_dataset.images,
        'labels': train_dataset.labels,
        'color_assignments': train_dataset.color_assignments,
        'correlation_followed': train_dataset.correlation_followed,
        'num_samples': train_dataset.num_samples,
        'correlation': train_dataset.correlation,
        'colors': train_dataset.colors,
        'size': train_dataset.size
    }, output_path / "train_data.pt", pickle_protocol=4)
    
    torch.save({
        'images': test_dataset.images,
        'labels': test_dataset.labels,
        'color_assignments': test_dataset.color_assignments,
        'correlation_followed': test_dataset.correlation_followed,
        'num_samples': test_dataset.num_samples,
        'correlation': test_dataset.correlation,
        'colors': test_dataset.colors,
        'size': test_dataset.size
    }, output_path / "test_data.pt", pickle_protocol=4)
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_synthetic_dataset(train_dataset, 20, str(output_path / "train_samples.png"))
    visualize_synthetic_dataset(test_dataset, 20, str(output_path / "test_samples.png"))
    
    print(f"Synthetic colored dataset saved to {output_path}")


def load_synthetic_dataset(data_dir: str) -> Tuple[SyntheticColoredDataset, SyntheticColoredDataset, Dict]:
    """Load synthetic colored dataset from files"""
    data_path = Path(data_dir)
    
    # Load data dictionaries
    train_data = torch.load(data_path / "train_data.pt", weights_only=False)
    test_data = torch.load(data_path / "test_data.pt", weights_only=False)
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Reconstruct datasets
    train_dataset = SyntheticColoredDataset.__new__(SyntheticColoredDataset)
    train_dataset.images = train_data['images']
    train_dataset.labels = train_data['labels']
    train_dataset.color_assignments = train_data['color_assignments']
    train_dataset.correlation_followed = train_data['correlation_followed']
    train_dataset.num_samples = train_data['num_samples']
    train_dataset.correlation = train_data['correlation']
    train_dataset.colors = train_data['colors']
    train_dataset.size = train_data['size']
    
    test_dataset = SyntheticColoredDataset.__new__(SyntheticColoredDataset)
    test_dataset.images = test_data['images']
    test_dataset.labels = test_data['labels']
    test_dataset.color_assignments = test_data['color_assignments']
    test_dataset.correlation_followed = test_data['correlation_followed']
    test_dataset.num_samples = test_data['num_samples']
    test_dataset.correlation = test_data['correlation']
    test_dataset.colors = test_data['colors']
    test_dataset.size = test_data['size']
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic colored digits for simplicity bias research")
    parser.add_argument("--train_correlation", type=float, default=0.9,
                       help="Correlation between color and label in training set")
    parser.add_argument("--test_correlation", type=float, default=0.1,
                       help="Correlation between color and label in test set")
    parser.add_argument("--train_size", type=int, default=12000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./synthetic_colored_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic Colored Digits Dataset")
    print("=" * 45)
    print(f"Train correlation: {args.train_correlation}")
    print(f"Test correlation: {args.test_correlation}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_synthetic_colored_datasets(
        train_correlation=args.train_correlation,
        test_correlation=args.test_correlation,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Analyze bias
    train_analysis = analyze_dataset_bias(train_dataset, "Training")
    test_analysis = analyze_dataset_bias(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_synthetic_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()