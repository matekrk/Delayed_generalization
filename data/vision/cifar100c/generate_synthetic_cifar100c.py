#!/usr/bin/env python3
"""
Synthetic CIFAR-100-C Dataset Generator for Robustness Research

This script generates synthetic CIFAR-100-like images with various corruptions
to study delayed generalization in robustness with 100 classes.

Usage:
    python generate_synthetic_cifar100c.py --train_corruptions noise blur --test_corruptions fog brightness --output_dir ./cifar100c_data
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
    """Generate simple synthetic objects for different CIFAR-100 classes"""
    image = np.zeros((size, size, 3))
    center = size // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    
    # We'll create distinctive patterns for each of the 100 classes
    # Organized by the 20 superclasses with 5 subclasses each
    
    superclass = class_id // 5
    subclass = class_id % 5
    
    if superclass == 0:  # Aquatic mammals
        # Base aquatic shape
        mask = ((x - center) / 12)**2 + ((y - center) / 8)**2 <= 1
        base_color = [0.3 + subclass * 0.1, 0.5, 0.7]
        image[mask] = base_color
        # Add distinctive features for each subclass
        if subclass == 0:  # beaver - add tail
            tail_mask = ((x - center - 8) / 4)**2 + ((y - center) / 3)**2 <= 1
            image[tail_mask] = [0.4, 0.3, 0.2]
        elif subclass == 1:  # dolphin - add fin
            fin_mask = ((x - center) / 3)**2 + ((y - center - 6) / 4)**2 <= 1
            image[fin_mask] = base_color
            
    elif superclass == 1:  # Fish
        # Fish-like shape
        body_mask = ((x - center) / 10)**2 + ((y - center) / 6)**2 <= 1
        base_color = [0.2 + subclass * 0.15, 0.6, 0.4 + subclass * 0.1]
        image[body_mask] = base_color
        # Tail
        tail_mask = ((x - center - 8) / 3)**2 + ((y - center) / 8)**2 <= 1
        image[tail_mask] = base_color
        
    elif superclass == 2:  # Flowers
        # Petal pattern
        for angle in range(0, 360, 60):
            rad = np.radians(angle + subclass * 10)
            px = center + int(8 * np.cos(rad))
            py = center + int(8 * np.sin(rad))
            if 0 <= px < size and 0 <= py < size:
                petal_y, petal_x = np.ogrid[:size, :size]
                petal_mask = ((petal_x - px) / 4)**2 + ((petal_y - py) / 6)**2 <= 1
                petal_color = [0.8, 0.2 + subclass * 0.15, 0.3 + subclass * 0.1]
                image[petal_mask] = petal_color
        # Center
        center_mask = ((x - center) / 3)**2 + ((y - center) / 3)**2 <= 1
        image[center_mask] = [0.9, 0.8, 0.1]
        
    elif superclass == 3:  # Food containers
        # Container shapes
        if subclass == 0:  # bottle - tall rectangle
            image[center-12:center+12, center-3:center+4, :] = [0.2, 0.7, 0.3]
            image[center-14:center-10, center-4:center+5, :] = [0.2, 0.7, 0.3]  # neck
        elif subclass == 1:  # bowl - semicircle
            bowl_mask = ((x - center) / 8)**2 + ((y - center + 4) / 6)**2 <= 1
            bowl_mask &= y >= center
            image[bowl_mask] = [0.8, 0.6, 0.4]
        else:  # can, cup, plate - variations of cylinders
            image[center-8:center+8, center-6:center+7, :] = [0.7, 0.7, 0.8]
            
    elif superclass == 4:  # Fruit and vegetables
        # Round fruit shapes with different colors
        fruit_mask = ((x - center) / 8)**2 + ((y - center) / 8)**2 <= 1
        colors = [[0.8, 0.2, 0.2], [0.6, 0.4, 0.2], [0.9, 0.5, 0.1], 
                  [0.3, 0.7, 0.2], [0.8, 0.3, 0.1]]
        image[fruit_mask] = colors[subclass]
        # Add stem
        image[center-10:center-6, center-1:center+2, :] = [0.2, 0.5, 0.1]
        
    elif superclass == 5:  # Household electrical devices
        # Rectangular devices with different aspect ratios
        if subclass == 0:  # clock - square with lines
            image[center-8:center+8, center-8:center+8, :] = [0.9, 0.9, 0.9]
            image[center-1:center+1, center-6:center+6, :] = [0.1, 0.1, 0.1]  # horizontal
            image[center-6:center+6, center-1:center+1, :] = [0.1, 0.1, 0.1]  # vertical
        elif subclass == 1:  # keyboard - wide rectangle
            image[center-4:center+4, center-12:center+12, :] = [0.2, 0.2, 0.2]
        else:  # lamp, phone, tv - vertical rectangles
            image[center-10:center+10, center-6:center+6, :] = [0.3, 0.3, 0.3]
            
    elif superclass == 6:  # Household furniture
        # Large rectangular shapes
        furniture_colors = [[0.6, 0.4, 0.2], [0.8, 0.6, 0.4], [0.5, 0.3, 0.2], 
                           [0.7, 0.5, 0.3], [0.4, 0.2, 0.1]]
        image[center-10:center+10, center-8:center+8, :] = furniture_colors[subclass]
        
    elif superclass == 7:  # Insects
        # Small body with wings/legs
        body_mask = ((x - center) / 3)**2 + ((y - center) / 8)**2 <= 1
        insect_colors = [[0.9, 0.8, 0.1], [0.2, 0.1, 0.1], [0.8, 0.4, 0.1], 
                        [0.3, 0.6, 0.2], [0.4, 0.2, 0.1]]
        image[body_mask] = insect_colors[subclass]
        # Wings
        if subclass in [1, 2]:  # beetle, butterfly
            wing_mask1 = ((x - center - 4) / 6)**2 + ((y - center) / 4)**2 <= 1
            wing_mask2 = ((x - center + 4) / 6)**2 + ((y - center) / 4)**2 <= 1
            image[wing_mask1] = insect_colors[subclass]
            image[wing_mask2] = insect_colors[subclass]
            
    elif superclass == 8:  # Large carnivores
        # Large body shapes
        body_mask = ((x - center) / 12)**2 + ((y - center) / 8)**2 <= 1
        carnivore_colors = [[0.6, 0.4, 0.2], [0.8, 0.7, 0.3], [0.7, 0.5, 0.2], 
                           [0.9, 0.6, 0.2], [0.5, 0.5, 0.5]]
        image[body_mask] = carnivore_colors[subclass]
        # Head
        head_mask = ((x - center - 8) / 6)**2 + ((y - center) / 6)**2 <= 1
        image[head_mask] = carnivore_colors[subclass]
        
    elif superclass == 9:  # Large man-made outdoor things
        # Architectural shapes
        if subclass == 0:  # bridge - horizontal structure
            image[center-2:center+2, :, :] = [0.6, 0.6, 0.6]
            image[center-8:center+8, center-2:center+2, :] = [0.6, 0.6, 0.6]
            image[center-8:center+8, center+10:center+12, :] = [0.6, 0.6, 0.6]
        elif subclass == 1:  # castle - castle-like structure
            image[center-10:center+10, center-8:center+8, :] = [0.7, 0.7, 0.7]
            image[center-12:center-8, center-6:center-2, :] = [0.7, 0.7, 0.7]
            image[center-12:center-8, center+2:center+6, :] = [0.7, 0.7, 0.7]
        else:  # house, road, skyscraper
            building_colors = [[0.8, 0.6, 0.4], [0.3, 0.3, 0.3], [0.6, 0.6, 0.7]]
            image[center-12:center+8, center-6:center+6, :] = building_colors[subclass-2]
            
    elif superclass == 10:  # Large natural outdoor scenes
        # Landscape-like patterns
        if subclass == 0:  # cloud - fluffy white shapes
            cloud_mask1 = ((x - center - 4) / 6)**2 + ((y - center - 2) / 4)**2 <= 1
            cloud_mask2 = ((x - center) / 8)**2 + ((y - center) / 5)**2 <= 1
            cloud_mask3 = ((x - center + 4) / 6)**2 + ((y - center + 1) / 4)**2 <= 1
            image[cloud_mask1 | cloud_mask2 | cloud_mask3] = [0.9, 0.9, 0.9]
        elif subclass == 1:  # forest - vertical green lines
            for i in range(0, size, 4):
                image[:, i:i+2, :] = [0.2, 0.6, 0.2]
        elif subclass == 2:  # mountain - triangular shape
            for y in range(size):
                width = max(0, size//2 - abs(y - center//2))
                if width > 0:
                    image[y, center-width:center+width, :] = [0.5, 0.4, 0.3]
        else:  # plain, sea - horizontal bands
            colors = [[0.3, 0.7, 0.2], [0.2, 0.4, 0.8]]
            image[:center, :, :] = [0.6, 0.8, 1.0]  # sky
            image[center:, :, :] = colors[subclass-3]
            
    elif superclass == 11:  # Large omnivores and herbivores
        # Large animal shapes
        body_mask = ((x - center) / 14)**2 + ((y - center) / 10)**2 <= 1
        animal_colors = [[0.8, 0.6, 0.4], [0.6, 0.4, 0.2], [0.5, 0.3, 0.2], 
                        [0.7, 0.7, 0.7], [0.8, 0.5, 0.3]]
        image[body_mask] = animal_colors[subclass]
        # Legs
        image[center+6:center+12, center-8:center-6, :] = animal_colors[subclass]
        image[center+6:center+12, center-2:center, :] = animal_colors[subclass]
        image[center+6:center+12, center+2:center+4, :] = animal_colors[subclass]
        image[center+6:center+12, center+6:center+8, :] = animal_colors[subclass]
        
    elif superclass == 12:  # Medium-sized mammals
        # Medium animal shapes
        body_mask = ((x - center) / 10)**2 + ((y - center) / 6)**2 <= 1
        mammal_colors = [[0.8, 0.4, 0.1], [0.6, 0.5, 0.4], [0.5, 0.5, 0.5], 
                        [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]
        image[body_mask] = mammal_colors[subclass]
        # Tail
        tail_mask = ((x - center - 8) / 6)**2 + ((y - center) / 3)**2 <= 1
        image[tail_mask] = mammal_colors[subclass]
        
    elif superclass == 13:  # Non-insect invertebrates
        # Various invertebrate shapes
        if subclass == 0:  # crab - wide body with claws
            body_mask = ((x - center) / 8)**2 + ((y - center) / 6)**2 <= 1
            image[body_mask] = [0.8, 0.3, 0.2]
            # Claws
            image[center-3:center+3, center-12:center-8, :] = [0.8, 0.3, 0.2]
            image[center-3:center+3, center+8:center+12, :] = [0.8, 0.3, 0.2]
        elif subclass == 1:  # lobster - elongated body
            image[center-4:center+4, center-12:center+12, :] = [0.7, 0.2, 0.1]
        else:  # snail, spider, worm - various small shapes
            colors = [[0.6, 0.5, 0.3], [0.2, 0.2, 0.2], [0.8, 0.4, 0.3]]
            if subclass == 2:  # snail - spiral
                spiral_mask = ((x - center) / 6)**2 + ((y - center) / 6)**2 <= 1
                image[spiral_mask] = colors[0]
            else:
                image[center-4:center+4, center-6:center+6, :] = colors[subclass-2]
                
    elif superclass == 14:  # People
        # Human-like shapes
        # Head
        head_mask = ((x - center) / 4)**2 + ((y - center - 8) / 5)**2 <= 1
        skin_colors = [[0.9, 0.7, 0.6], [0.8, 0.6, 0.5], [0.9, 0.8, 0.7], 
                      [0.7, 0.5, 0.4], [0.8, 0.7, 0.6]]
        image[head_mask] = skin_colors[subclass]
        # Body
        image[center-2:center+8, center-3:center+3, :] = [0.3, 0.4, 0.8]  # shirt
        # Arms and legs
        image[center-2:center+2, center-8:center-3, :] = skin_colors[subclass]  # left arm
        image[center-2:center+2, center+3:center+8, :] = skin_colors[subclass]   # right arm
        image[center+8:center+14, center-2:center, :] = [0.2, 0.2, 0.8]         # left leg
        image[center+8:center+14, center:center+2, :] = [0.2, 0.2, 0.8]         # right leg
        
    elif superclass == 15:  # Reptiles
        # Reptilian shapes
        body_mask = ((x - center) / 12)**2 + ((y - center) / 4)**2 <= 1
        reptile_colors = [[0.3, 0.6, 0.2], [0.5, 0.4, 0.2], [0.4, 0.6, 0.3], 
                         [0.2, 0.5, 0.1], [0.4, 0.5, 0.3]]
        image[body_mask] = reptile_colors[subclass]
        # Head
        head_mask = ((x - center - 10) / 4)**2 + ((y - center) / 3)**2 <= 1
        image[head_mask] = reptile_colors[subclass]
        # Tail
        tail_mask = ((x - center + 10) / 8)**2 + ((y - center) / 2)**2 <= 1
        image[tail_mask] = reptile_colors[subclass]
        
    elif superclass == 16:  # Small mammals
        # Small animal shapes
        body_mask = ((x - center) / 6)**2 + ((y - center) / 4)**2 <= 1
        small_colors = [[0.8, 0.6, 0.4], [0.5, 0.5, 0.5], [0.7, 0.5, 0.3], 
                       [0.4, 0.3, 0.2], [0.6, 0.4, 0.2]]
        image[body_mask] = small_colors[subclass]
        # Ears
        ear1_mask = ((x - center - 3) / 2)**2 + ((y - center - 4) / 3)**2 <= 1
        ear2_mask = ((x - center + 3) / 2)**2 + ((y - center - 4) / 3)**2 <= 1
        image[ear1_mask | ear2_mask] = small_colors[subclass]
        
    elif superclass == 17:  # Trees
        # Tree shapes
        # Trunk
        image[center+4:center+12, center-2:center+2, :] = [0.6, 0.4, 0.2]
        # Foliage
        foliage_colors = [[0.8, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.8, 0.3], 
                         [0.1, 0.6, 0.1], [0.6, 0.8, 0.2]]
        if subclass == 0:  # maple - star-like leaves
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                lx = center + int(8 * np.cos(rad))
                ly = center - 4 + int(6 * np.sin(rad))
                if 0 <= lx < size and 0 <= ly < size:
                    leaf_mask = ((x - lx) / 4)**2 + ((y - ly) / 4)**2 <= 1
                    image[leaf_mask] = foliage_colors[subclass]
        else:  # other trees - circular foliage
            foliage_mask = ((x - center) / 10)**2 + ((y - center - 4) / 8)**2 <= 1
            image[foliage_mask] = foliage_colors[subclass]
            
    elif superclass == 18:  # Vehicles 1
        # Ground vehicles
        if subclass == 0:  # bicycle - two circles connected
            wheel1_mask = ((x - center - 6) / 4)**2 + ((y - center + 4) / 4)**2 <= 1
            wheel2_mask = ((x - center + 6) / 4)**2 + ((y - center + 4) / 4)**2 <= 1
            image[wheel1_mask | wheel2_mask] = [0.2, 0.2, 0.2]
            # Frame
            image[center-2:center+2, center-6:center+6, :] = [0.8, 0.2, 0.2]
        elif subclass == 1:  # bus - large rectangle
            image[center-6:center+6, center-12:center+12, :] = [0.9, 0.8, 0.1]
            # Windows
            image[center-4:center-2, center-8:center+8, :] = [0.6, 0.8, 1.0]
        else:  # motorcycle, pickup, train
            vehicle_colors = [[0.2, 0.2, 0.8], [0.8, 0.2, 0.2], [0.3, 0.3, 0.3]]
            image[center-4:center+4, center-8:center+8, :] = vehicle_colors[subclass-2]
            
    else:  # Vehicles 2 (superclass == 19)
        # Various vehicles
        vehicle2_colors = [[0.3, 0.8, 0.2], [0.8, 0.3, 0.1], [0.9, 0.9, 0.1], 
                          [0.4, 0.4, 0.4], [0.2, 0.6, 0.2]]
        if subclass == 1:  # rocket - tall thin shape
            image[center-12:center+4, center-2:center+2, :] = vehicle2_colors[subclass]
            # Nose cone
            image[center-14:center-12, center-1:center+1, :] = vehicle2_colors[subclass]
        else:  # other vehicles
            image[center-6:center+6, center-8:center+8, :] = vehicle2_colors[subclass]
    
    return image


def apply_corruption(image: np.ndarray, corruption_type: str, severity: int = 3) -> np.ndarray:
    """Apply corruption to image"""
    severity = max(1, min(5, severity))
    
    if corruption_type == 'gaussian_noise':
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        noise = np.random.normal(size=image.shape) * c
        return np.clip(image + noise, 0, 1)
        
    elif corruption_type == 'motion_blur':
        from scipy import ndimage
        c = [3, 5, 7, 9, 11][severity - 1]
        kernel = np.zeros((c, c))
        kernel[c//2, :] = 1 / c
        
        blurred = np.zeros_like(image)
        for i in range(3):
            blurred[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return blurred
        
    elif corruption_type == 'fog':
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        fog = np.random.normal(size=image.shape[:2]) * 0.1 + 0.9
        fog = np.clip(fog, 0, 1)
        fog = fog[:, :, np.newaxis]
        return (1 - c) * image + c * fog
        
    elif corruption_type == 'brightness':
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        return np.clip(image + c, 0, 1)
        
    elif corruption_type == 'contrast':
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        return np.clip(image * c, 0, 1)
        
    elif corruption_type == 'snow':
        c = [0.1, 0.15, 0.2, 0.25, 0.3][severity - 1]
        snow = np.random.random(image.shape[:2]) < c
        snow = snow[:, :, np.newaxis]
        return np.where(snow, 1.0, image)
        
    else:
        return image


class SyntheticCIFAR100CDataset(Dataset):
    """Dataset that creates synthetic CIFAR-100-like images with corruptions"""
    
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
        """Generate synthetic CIFAR-100-C data"""
        self.images = []
        self.labels = []
        self.corruption_info = []
        
        for i in range(self.num_samples):
            # Random class (0-99)
            class_id = np.random.randint(0, 100)
            
            # Generate base synthetic object
            base_image = generate_synthetic_object(class_id, self.size)
            
            # Random corruption
            corruption_type = np.random.choice(self.corruption_types)
            severity = np.random.randint(self.severity_range[0], self.severity_range[1] + 1)
            
            # Apply corruption
            corrupted_image = apply_corruption(base_image, corruption_type, severity)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(corrupted_image).permute(2, 0, 1).float()
            
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


def create_synthetic_cifar100c_datasets(
    train_corruptions: List[str],
    test_corruptions: List[str],
    train_size: int = 50000,
    test_size: int = 10000,
    severity_range: Tuple[int, int] = (1, 5),
    seed: int = 42
) -> Tuple[SyntheticCIFAR100CDataset, SyntheticCIFAR100CDataset, Dict]:
    """Create synthetic CIFAR-100-C train and test datasets"""
    
    train_dataset = SyntheticCIFAR100CDataset(
        train_size, train_corruptions, severity_range, seed
    )
    
    test_dataset = SyntheticCIFAR100CDataset(
        test_size, test_corruptions, severity_range, seed + 1
    )
    
    # Metadata
    metadata = {
        'train_corruptions': train_corruptions,
        'test_corruptions': test_corruptions,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'num_classes': 100,
        'severity_range': severity_range,
        'image_size': 32,
        'seed': seed
    }
    
    return train_dataset, test_dataset, metadata


def visualize_samples(dataset: SyntheticCIFAR100CDataset, save_path: str, num_samples: int = 20):
    """Visualize samples from the dataset"""
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        image, label, corruption_info = dataset[i]
        
        # Convert tensor to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        
        axes[i].imshow(image_np)
        axes[i].set_title(f"Class: {label}\n{corruption_info['corruption_type']}\nSev: {corruption_info['severity']}", 
                         fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CIFAR-100-C dataset")
    parser.add_argument("--train_corruptions", nargs='+', 
                       default=['gaussian_noise', 'motion_blur'],
                       help="Corruption types for training")
    parser.add_argument("--test_corruptions", nargs='+',
                       default=['fog', 'brightness', 'contrast'],
                       help="Corruption types for testing")
    parser.add_argument("--train_size", type=int, default=50000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=10000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./synthetic_cifar100c_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic CIFAR-100-C Dataset")
    print("=" * 40)
    print(f"Train corruptions: {args.train_corruptions}")
    print(f"Test corruptions: {args.test_corruptions}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    train_dataset, test_dataset, metadata = create_synthetic_cifar100c_datasets(
        args.train_corruptions,
        args.test_corruptions,
        args.train_size,
        args.test_size,
        seed=args.seed
    )
    
    print(f"Generated train dataset: {len(train_dataset)} samples")
    print(f"Generated test dataset: {len(test_dataset)} samples")
    
    # Visualize samples
    print("Creating visualizations...")
    visualize_samples(train_dataset, output_dir / "train_samples.png")
    visualize_samples(test_dataset, output_dir / "test_samples.png")
    
    # Save datasets
    print("Saving datasets...")
    torch.save(train_dataset, output_dir / "train_dataset.pt")
    torch.save(test_dataset, output_dir / "test_dataset.pt")
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDatasets saved to: {output_dir}")
    print(f"Train samples visualization: {output_dir / 'train_samples.png'}")
    print(f"Test samples visualization: {output_dir / 'test_samples.png'}")
    print("Synthetic CIFAR-100-C dataset generation complete!")


if __name__ == "__main__":
    main()