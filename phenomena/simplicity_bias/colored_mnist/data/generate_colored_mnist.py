#!/usr/bin/env python3
"""
Colored MNIST Dataset Generator for Simplicity Bias Research

This script generates colored MNIST datasets where digit color correlates with
class label in training data but not in test data, to study simplicity bias.

Usage:
    python generate_colored_mnist.py --train_correlation 0.9 --test_correlation 0.1
"""

import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


class ColoredMNISTDataset(Dataset):
    """Dataset that adds color correlation to MNIST digits"""
    
    def __init__(
        self,
        mnist_data: torch.utils.data.Dataset,
        correlation: float,
        colors: List[Tuple[float, float, float]] = None,
        seed: int = 42
    ):
        self.mnist_data = mnist_data
        self.correlation = correlation
        
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
        
        # Pre-compute color assignments
        self._create_color_assignments()
    
    def _create_color_assignments(self):
        """Create color assignments for each sample"""
        self.color_assignments = []
        
        for idx in range(len(self.mnist_data)):
            _, label = self.mnist_data[idx]
            
            # Determine if this sample follows the correlation
            if np.random.random() < self.correlation:
                # Correlated: even digits -> color 0, odd digits -> color 1
                color_idx = label % 2
            else:
                # Anti-correlated: random color assignment
                color_idx = np.random.randint(0, len(self.colors))
            
            self.color_assignments.append(color_idx)
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        
        # Convert grayscale to RGB
        if image.shape[0] == 1:  # [1, H, W]
            image = image.repeat(3, 1, 1)  # [3, H, W]
        
        # Apply color
        color_idx = self.color_assignments[idx]
        color = torch.tensor(self.colors[color_idx], dtype=torch.float32).view(3, 1, 1)
        
        # Colorize the image
        colored_image = image * color
        
        # Metadata for analysis
        metadata = {
            'color_idx': color_idx,
            'correlation_followed': (label % 2) == color_idx,
            'original_label': label
        }
        
        return colored_image, label, metadata


def create_colored_mnist_datasets(
    train_correlation: float = 0.9,
    test_correlation: float = 0.1,
    data_dir: str = "./data",
    colors: List[Tuple[float, float, float]] = None,
    seed: int = 42
) -> Tuple[ColoredMNISTDataset, ColoredMNISTDataset, Dict]:
    """
    Create colored MNIST train and test datasets
    
    Args:
        train_correlation: Correlation between color and label in training set
        test_correlation: Correlation between color and label in test set
        data_dir: Directory to store MNIST data
        colors: List of RGB color tuples
        seed: Random seed
        
    Returns:
        train_dataset, test_dataset, metadata
    """
    
    # Download and load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    mnist_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    mnist_test = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Create colored versions
    train_dataset = ColoredMNISTDataset(
        mnist_train, train_correlation, colors, seed
    )
    
    test_dataset = ColoredMNISTDataset(
        mnist_test, test_correlation, colors, seed + 1
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
        'seed': seed
    }
    
    return train_dataset, test_dataset, metadata


def analyze_dataset_bias(dataset: ColoredMNISTDataset, name: str) -> Dict:
    """Analyze the bias in a colored MNIST dataset"""
    color_counts = {0: {}, 1: {}}  # color_idx -> {label: count}
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


def visualize_colored_mnist(dataset: ColoredMNISTDataset, num_samples: int = 20, save_path: str = None):
    """Visualize colored MNIST samples"""
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    
    # Sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label, metadata = dataset[idx]
        
        # Convert tensor to numpy and transpose for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)  # Ensure valid range
        
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


def save_colored_mnist_dataset(
    train_dataset: ColoredMNISTDataset,
    test_dataset: ColoredMNISTDataset,
    metadata: Dict,
    output_dir: str
):
    """Save colored MNIST dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset with subdir
    subdir = f"traincorr_{metadata['train_correlation']}_testcorr_{metadata['test_correlation']}"
    output_path = output_path / subdir
    
    # Save datasets using torch.save for efficiency
    torch.save(train_dataset, output_path / "train_dataset.pt")
    torch.save(test_dataset, output_path / "test_dataset.pt")
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_colored_mnist(train_dataset, 20, str(output_path / "train_samples.png"))
    visualize_colored_mnist(test_dataset, 20, str(output_path / "test_samples.png"))
    
    print(f"Colored MNIST dataset saved to {output_path}")


def load_colored_mnist_dataset(data_dir: str) -> Tuple[ColoredMNISTDataset, ColoredMNISTDataset, Dict]:
    """Load colored MNIST dataset from files"""
    data_path = Path(data_dir)
    
    train_dataset = torch.load(data_path / "train_dataset.pt")
    test_dataset = torch.load(data_path / "test_dataset.pt")
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate colored MNIST dataset for simplicity bias research")
    parser.add_argument("--train_correlation", type=float, default=0.9, 
                       help="Correlation between color and label in training set")
    parser.add_argument("--test_correlation", type=float, default=0.1,
                       help="Correlation between color and label in test set")
    parser.add_argument("--data_dir", type=str, default="./mnist_data",
                       help="Directory to store original MNIST data")
    parser.add_argument("--output_dir", type=str, default="./colored_mnist_data",
                       help="Output directory for colored MNIST dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Create sample visualizations")
    
    args = parser.parse_args()
    
    print("Generating Colored MNIST Dataset")
    print("=" * 40)
    print(f"Train correlation: {args.train_correlation}")
    print(f"Test correlation: {args.test_correlation}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_colored_mnist_datasets(
        train_correlation=args.train_correlation,
        test_correlation=args.test_correlation,
        data_dir=args.data_dir,
        seed=args.seed
    )
    
    # Analyze bias
    train_analysis = analyze_dataset_bias(train_dataset, "Training")
    test_analysis = analyze_dataset_bias(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_colored_mnist_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()