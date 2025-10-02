#!/usr/bin/env python3
"""
Synthetic ImageNet-C Dataset Generator for Robustness Research

This script generates synthetic ImageNet-like images with various corruptions
to study delayed generalization in robustness at ImageNet scale.

Usage:
    python generate_synthetic_imagenetc.py --corruption_types noise blur --output_dir ./imagenetc_data
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def generate_synthetic_imagenet_sample(class_id: int, size: int = 224) -> np.ndarray:
    """
    Generate simple synthetic ImageNet-like images.
    For 1000 classes, we use procedural generation with class-specific patterns.
    """
    image = np.zeros((size, size, 3))
    
    # Use class_id to generate unique but consistent patterns
    np.random.seed(class_id)
    
    # Base color from class id
    hue = (class_id * 137) % 360  # Prime number for good distribution
    r = 0.5 + 0.3 * np.cos(hue * np.pi / 180)
    g = 0.5 + 0.3 * np.cos((hue + 120) * np.pi / 180)
    b = 0.5 + 0.3 * np.cos((hue + 240) * np.pi / 180)
    
    # Create base color
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    
    # Add class-specific geometric patterns
    center_x, center_y = size // 2, size // 2
    pattern_type = class_id % 10
    
    if pattern_type == 0:  # Circle
        y, x = np.ogrid[:size, :size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (size // 3)**2
        image[mask] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 1:  # Rectangle
        h, w = size // 3, size // 2
        image[center_y-h:center_y+h, center_x-w:center_x+w] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 2:  # Horizontal stripes
        stripe_width = size // 10
        for i in range(0, size, stripe_width * 2):
            image[i:i+stripe_width, :] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 3:  # Vertical stripes
        stripe_width = size // 10
        for i in range(0, size, stripe_width * 2):
            image[:, i:i+stripe_width] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 4:  # Diagonal pattern
        for i in range(size):
            for j in range(size):
                if (i + j) % 20 < 10:
                    image[i, j] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 5:  # Grid pattern
        grid_size = size // 8
        for i in range(0, size, grid_size):
            image[i:i+2, :] = [r * 1.3, g * 1.3, b * 1.3]
            image[:, i:i+2] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 6:  # Gradient
        for i in range(size):
            factor = i / size
            image[i, :] = [r * (1 + 0.5 * factor), g * (1 + 0.5 * factor), b * (1 + 0.5 * factor)]
    
    elif pattern_type == 7:  # Checkerboard
        checker_size = size // 8
        for i in range(0, size, checker_size):
            for j in range(0, size, checker_size):
                if (i // checker_size + j // checker_size) % 2:
                    image[i:i+checker_size, j:j+checker_size] = [r * 1.3, g * 1.3, b * 1.3]
    
    elif pattern_type == 8:  # Concentric circles
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        ring_width = size // 10
        mask = (dist % (ring_width * 2)) < ring_width
        image[mask] = [r * 1.3, g * 1.3, b * 1.3]
    
    else:  # Random noise pattern
        noise = np.random.random((size, size, 3)) * 0.3
        image = image + noise
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image


def apply_corruption(image: np.ndarray, corruption_type: str, severity: int = 3) -> np.ndarray:
    """Apply corruption to image"""
    corrupted = image.copy()
    
    if corruption_type == 'gaussian_noise':
        noise_std = severity * 0.03
        noise = np.random.normal(0, noise_std, image.shape)
        corrupted = np.clip(image + noise, 0, 1)
        
    elif corruption_type == 'motion_blur':
        # Simple motion blur approximation
        kernel_size = severity * 3 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        for c in range(3):
            temp = np.zeros_like(corrupted[:, :, c])
            for i in range(kernel_size):
                shift = i - kernel_size//2
                if shift >= 0:
                    temp[:, shift:] += corrupted[:, :-shift if shift > 0 else None, c] * kernel[0, i]
                else:
                    temp[:, :shift] += corrupted[:, -shift:, c] * kernel[0, i]
            corrupted[:, :, c] = temp
            
    elif corruption_type == 'fog':
        fog_strength = severity * 0.1
        fog = np.random.uniform(0.3, 0.7, image.shape)
        corrupted = (1 - fog_strength) * image + fog_strength * fog
        
    elif corruption_type == 'brightness':
        brightness_factor = 1 + (severity - 3) * 0.2
        corrupted = np.clip(image * brightness_factor, 0, 1)
        
    elif corruption_type == 'contrast':
        contrast_factor = 1 + (severity - 3) * 0.3
        mean_val = np.mean(image)
        corrupted = np.clip((image - mean_val) * contrast_factor + mean_val, 0, 1)
    
    return corrupted


class SyntheticImageNetCDataset(Dataset):
    """Dataset that creates synthetic ImageNet-like images with corruptions"""
    
    def __init__(
        self,
        num_samples: int = 10000,
        num_classes: int = 1000,
        corruption_types: List[str] = None,
        severity_range: Tuple[int, int] = (1, 5),
        image_size: int = 224,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes (1000 for ImageNet)
            corruption_types: List of corruption types
            severity_range: Range of corruption severities (min, max)
            image_size: Size of generated images
            seed: Random seed
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        if corruption_types is None:
            corruption_types = ['gaussian_noise', 'motion_blur', 'fog', 'brightness', 'contrast']
        self.corruption_types = corruption_types
        
        self.severity_range = severity_range
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Pre-generate data
        print(f"Generating {num_samples} synthetic ImageNet-C samples...")
        self.images = []
        self.labels = []
        self.corruption_info = []
        
        for i in range(num_samples):
            # Generate class label
            label = np.random.randint(0, num_classes)
            
            # Generate base image
            base_image = generate_synthetic_imagenet_sample(label, image_size)
            
            # Apply corruption
            corruption_type = np.random.choice(corruption_types)
            severity = np.random.randint(severity_range[0], severity_range[1] + 1)
            corrupted_image = apply_corruption(base_image, corruption_type, severity)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(corrupted_image.transpose(2, 0, 1)).float()
            
            self.images.append(image_tensor)
            self.labels.append(label)
            self.corruption_info.append({
                'corruption_type': corruption_type,
                'severity': severity
            })
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")
        
        print("Dataset generation complete!")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        corruption_info = self.corruption_info[idx]
        
        return image, label, corruption_info


def visualize_synthetic_imagenetc(dataset: SyntheticImageNetCDataset, save_path: str = None):
    """Visualize synthetic ImageNet-C samples"""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(20):
        image, label, corruption_info = dataset[i]
        
        # Convert tensor to numpy for display
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Class {label}\n{corruption_info['corruption_type']}\nSeverity {corruption_info['severity']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic ImageNet-C dataset")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to generate")
    parser.add_argument("--num_classes", type=int, default=1000,
                       help="Number of classes")
    parser.add_argument("--corruption_types", type=str, nargs="+",
                       default=['gaussian_noise', 'motion_blur', 'fog', 'brightness', 'contrast'],
                       help="Corruption types to include")
    parser.add_argument("--severity_min", type=int, default=1,
                       help="Minimum corruption severity")
    parser.add_argument("--severity_max", type=int, default=5,
                       help="Maximum corruption severity")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Size of generated images")
    parser.add_argument("--output_dir", type=str, default="./synthetic_imagenetc_data",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic ImageNet-C Dataset")
    print("=" * 50)
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Corruptions: {args.corruption_types}")
    print(f"Severity range: {args.severity_min}-{args.severity_max}")
    
    # Create dataset
    dataset = SyntheticImageNetCDataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        corruption_types=args.corruption_types,
        severity_range=(args.severity_min, args.severity_max),
        image_size=args.image_size,
        seed=args.seed
    )
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    torch.save(dataset, output_path / "synthetic_imagenetc_dataset.pt")
    
    # Save metadata
    metadata = {
        'num_samples': args.num_samples,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'corruption_types': args.corruption_types,
        'severity_range': (args.severity_min, args.severity_max),
        'seed': args.seed
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Visualize samples
    print("\nCreating visualization...")
    visualize_synthetic_imagenetc(dataset, str(output_path / "samples.png"))
    
    print("\n✓ Dataset generation complete!")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
