#!/usr/bin/env python3
"""
Synthetic CIFAR-10-C Dataset Generator for Robustness Research

This script generates synthetic CIFAR-10-like images with various corruptions
to study delayed generalization in robustness.

Usage:
    python generate_synthetic_cifar10c.py --corruption_types noise blur --output_dir ./cifar10c_data
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def generate_synthetic_object(class_id: int, size: int = 32) -> np.ndarray:
    """Generate simple synthetic objects for different classes"""
    image = np.zeros((size, size, 3))
    center = size // 2
    
    if class_id == 0:  # Airplane
        # Wing
        image[center-2:center+3, :, :] = [0.7, 0.7, 0.7]
        # Body
        image[:, center-2:center+3, :] = [0.8, 0.8, 0.8]
        
    elif class_id == 1:  # Automobile
        # Body (rectangle)
        image[center-6:center+7, center-10:center+11, :] = [0.8, 0.2, 0.2]
        # Wheels
        image[center+4:center+8, center-6:center-2, :] = [0.1, 0.1, 0.1]
        image[center+4:center+8, center+3:center+7, :] = [0.1, 0.1, 0.1]
        
    elif class_id == 2:  # Bird
        # Body (oval)
        y, x = np.ogrid[:size, :size]
        mask = ((x - center) / 8)**2 + ((y - center) / 6)**2 <= 1
        image[mask] = [0.6, 0.4, 0.2]
        # Wing
        wing_mask = ((x - center + 4) / 6)**2 + ((y - center - 2) / 4)**2 <= 1
        image[wing_mask] = [0.4, 0.3, 0.1]
        
    elif class_id == 3:  # Cat
        # Head (circle)
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= 36
        image[mask] = [0.7, 0.5, 0.3]
        # Ears (triangles)
        image[center-8:center-3, center-4:center, :] = [0.6, 0.4, 0.2]
        image[center-8:center-3, center+1:center+5, :] = [0.6, 0.4, 0.2]
        
    elif class_id == 4:  # Deer
        # Body
        image[center:center+8, center-6:center+7, :] = [0.6, 0.4, 0.2]
        # Head
        image[center-8:center+2, center-3:center+4, :] = [0.6, 0.4, 0.2]
        # Antlers
        image[center-12:center-6, center-2:center+3, :] = [0.4, 0.3, 0.2]
        
    elif class_id == 5:  # Dog
        # Similar to cat but different proportions
        y, x = np.ogrid[:size, :size]
        mask = ((x - center) / 7)**2 + ((y - center) / 9)**2 <= 1
        image[mask] = [0.5, 0.3, 0.1]
        # Ears (floppy)
        image[center-6:center, center-6:center-2, :] = [0.4, 0.2, 0.0]
        image[center-6:center, center+3:center+7, :] = [0.4, 0.2, 0.0]
        
    elif class_id == 6:  # Frog
        # Body (circle, green)
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= 49
        image[mask] = [0.2, 0.7, 0.2]
        # Eyes
        image[center-4:center, center-2:center+3, :] = [0.1, 0.8, 0.1]
        
    elif class_id == 7:  # Horse
        # Body (elongated)
        image[center-2:center+8, center-8:center+9, :] = [0.6, 0.4, 0.2]
        # Head
        image[center-8:center-2, center-4:center+5, :] = [0.6, 0.4, 0.2]
        # Legs
        image[center+6:center+12, center-6:center-4, :] = [0.5, 0.3, 0.1]
        image[center+6:center+12, center+5:center+7, :] = [0.5, 0.3, 0.1]
        
    elif class_id == 8:  # Ship
        # Hull
        image[center+2:center+8, center-8:center+9, :] = [0.3, 0.3, 0.8]
        # Mast
        image[center-8:center+2, center-1:center+2, :] = [0.6, 0.4, 0.2]
        # Sail
        image[center-6:center, center-6:center-1, :] = [1.0, 1.0, 1.0]
        
    elif class_id == 9:  # Truck
        # Similar to car but larger
        image[center-8:center+9, center-12:center+13, :] = [0.7, 0.7, 0.1]
        # Wheels
        image[center+6:center+10, center-8:center-4, :] = [0.1, 0.1, 0.1]
        image[center+6:center+10, center+5:center+9, :] = [0.1, 0.1, 0.1]
    
    # Clip values
    image = np.clip(image, 0, 1)
    return image


def apply_corruption(image: np.ndarray, corruption_type: str, severity: int = 3) -> np.ndarray:
    """Apply corruption to image"""
    corrupted = image.copy()
    
    if corruption_type == 'gaussian_noise':
        noise_std = severity * 0.05
        noise = np.random.normal(0, noise_std, image.shape)
        corrupted = np.clip(image + noise, 0, 1)
        
    elif corruption_type == 'motion_blur':
        # Simple motion blur approximation
        kernel_size = severity * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        for c in range(3):
            # Simple convolution approximation
            temp = np.zeros_like(corrupted[:, :, c])
            for i in range(kernel_size):
                shift = i - kernel_size//2
                if shift >= 0:
                    temp[:, shift:] += corrupted[:, :-shift if shift > 0 else None, c] * kernel[0, i]
                else:
                    temp[:, :shift] += corrupted[:, -shift:, c] * kernel[0, i]
            corrupted[:, :, c] = temp
            
    elif corruption_type == 'fog':
        # Add fog effect
        fog_strength = severity * 0.15
        fog = np.random.uniform(0.3, 0.7, image.shape)
        corrupted = (1 - fog_strength) * image + fog_strength * fog
        
    elif corruption_type == 'brightness':
        # Brightness change
        brightness_factor = 1 + (severity - 3) * 0.3
        corrupted = np.clip(image * brightness_factor, 0, 1)
        
    elif corruption_type == 'contrast':
        # Contrast change
        contrast_factor = 1 + (severity - 3) * 0.4
        mean_val = np.mean(image)
        corrupted = np.clip((image - mean_val) * contrast_factor + mean_val, 0, 1)
    
    return corrupted


class SyntheticCIFAR10CDataset(Dataset):
    """Dataset that creates synthetic CIFAR-10-like images with corruptions"""
    
    def __init__(
        self,
        num_samples: int,
        corruption_types: List[str],
        severity_range: Tuple[int, int] = (1, 5),
        seed: int = 42,
        size: int = 32
    ):
        self.num_samples = num_samples
        self.corruption_types = corruption_types
        self.severity_range = severity_range
        self.size = size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic CIFAR-10-C data"""
        self.images = []
        self.labels = []
        self.corruption_info = []
        
        for _ in range(self.num_samples):
            # Random class
            class_id = np.random.randint(0, 10)
            
            # Generate clean image
            clean_image = generate_synthetic_object(class_id, self.size)
            
            # Apply random corruption
            corruption_type = np.random.choice(self.corruption_types)
            severity = np.random.randint(self.severity_range[0], self.severity_range[1] + 1)
            corrupted_image = apply_corruption(clean_image, corruption_type, severity)
            
            # Convert to tensor (CHW format)
            image_tensor = torch.tensor(corrupted_image.transpose(2, 0, 1), dtype=torch.float32)
            
            self.images.append(image_tensor)
            self.labels.append(class_id)
            self.corruption_info.append({
                'corruption_type': corruption_type,
                'severity': severity
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        corruption_info = self.corruption_info[idx]
        
        return image, label, corruption_info


def create_synthetic_cifar10c_datasets(
    train_corruptions: List[str],
    test_corruptions: List[str],
    train_size: int = 10000,
    test_size: int = 2000,
    severity_range: Tuple[int, int] = (1, 5),
    seed: int = 42
) -> Tuple[SyntheticCIFAR10CDataset, SyntheticCIFAR10CDataset, Dict]:
    """Create synthetic CIFAR-10-C train and test datasets"""
    
    train_dataset = SyntheticCIFAR10CDataset(
        train_size, train_corruptions, severity_range, seed
    )
    
    test_dataset = SyntheticCIFAR10CDataset(
        test_size, test_corruptions, severity_range, seed + 1
    )
    
    # Metadata
    metadata = {
        'train_corruptions': train_corruptions,
        'test_corruptions': test_corruptions,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'num_classes': 10,
        'image_shape': [3, 32, 32],
        'severity_range': severity_range,
        'seed': seed,
        'dataset_type': 'synthetic_cifar10c'
    }
    
    return train_dataset, test_dataset, metadata


def save_synthetic_cifar10c_dataset(
    train_dataset: SyntheticCIFAR10CDataset,
    test_dataset: SyntheticCIFAR10CDataset,
    metadata: Dict,
    output_dir: str
):
    """Save synthetic CIFAR-10-C dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as tensors and lists
    torch.save({
        'images': train_dataset.images,
        'labels': train_dataset.labels,
        'corruption_info': train_dataset.corruption_info,
        'num_samples': train_dataset.num_samples,
        'corruption_types': train_dataset.corruption_types,
        'severity_range': train_dataset.severity_range,
        'size': train_dataset.size
    }, output_path / "train_data.pt", pickle_protocol=4)
    
    torch.save({
        'images': test_dataset.images,
        'labels': test_dataset.labels,
        'corruption_info': test_dataset.corruption_info,
        'num_samples': test_dataset.num_samples,
        'corruption_types': test_dataset.corruption_types,
        'severity_range': test_dataset.severity_range,
        'size': test_dataset.size
    }, output_path / "test_data.pt", pickle_protocol=4)
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Synthetic CIFAR-10-C dataset saved to {output_path}")


def load_synthetic_cifar10c_dataset(data_dir: str) -> Tuple[SyntheticCIFAR10CDataset, SyntheticCIFAR10CDataset, Dict]:
    """Load synthetic CIFAR-10-C dataset from files"""
    data_path = Path(data_dir)
    
    # Load data dictionaries
    train_data = torch.load(data_path / "train_data.pt", weights_only=False)
    test_data = torch.load(data_path / "test_data.pt", weights_only=False)
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Reconstruct datasets
    train_dataset = SyntheticCIFAR10CDataset.__new__(SyntheticCIFAR10CDataset)
    train_dataset.images = train_data['images']
    train_dataset.labels = train_data['labels']
    train_dataset.corruption_info = train_data['corruption_info']
    train_dataset.num_samples = train_data['num_samples']
    train_dataset.corruption_types = train_data['corruption_types']
    train_dataset.severity_range = train_data['severity_range']
    train_dataset.size = train_data['size']
    
    test_dataset = SyntheticCIFAR10CDataset.__new__(SyntheticCIFAR10CDataset)
    test_dataset.images = test_data['images']
    test_dataset.labels = test_data['labels']
    test_dataset.corruption_info = test_data['corruption_info']
    test_dataset.num_samples = test_data['num_samples']
    test_dataset.corruption_types = test_data['corruption_types']
    test_dataset.severity_range = test_data['severity_range']
    test_dataset.size = test_data['size']
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CIFAR-10-C dataset")
    parser.add_argument("--train_corruptions", nargs='+', 
                       default=['gaussian_noise', 'motion_blur'],
                       help="Corruption types for training")
    parser.add_argument("--test_corruptions", nargs='+',
                       default=['fog', 'brightness', 'contrast'],
                       help="Corruption types for testing")
    parser.add_argument("--train_size", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./synthetic_cifar10c_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic CIFAR-10-C Dataset")
    print("=" * 38)
    print(f"Train corruptions: {args.train_corruptions}")
    print(f"Test corruptions: {args.test_corruptions}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_synthetic_cifar10c_datasets(
        train_corruptions=args.train_corruptions,
        test_corruptions=args.test_corruptions,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Save dataset
    save_synthetic_cifar10c_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()