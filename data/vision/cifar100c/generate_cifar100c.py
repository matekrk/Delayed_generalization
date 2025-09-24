#!/usr/bin/env python3
"""
CIFAR-100-C Data Generation and Loading

This module handles the creation and loading of CIFAR-100-C datasets with various
corruption types for studying robustness vs accuracy tradeoffs.
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Any
from torch.utils.data import DataLoader, Dataset, Subset


class CIFAR100CDataset(Dataset):
    """
    Dataset wrapper for CIFAR-100-C with corruption handling.
    
    This dataset supports both clean CIFAR-100 and corrupted versions
    for studying robustness patterns in delayed generalization.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        corruption_type: Optional[str] = None,
        corruption_level: int = 3,
        transform: Optional[transforms.Compose] = None,
        download: bool = True
    ):
        self.root = root
        self.train = train
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.transform = transform
        
        # Load base CIFAR-100 dataset
        self.cifar100 = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll apply transforms later
        )
        
        # Load corrupted data if specified
        if corruption_type and not train:  # Corruptions typically only for test
            self.corrupted_data = self._load_corrupted_data()
        else:
            self.corrupted_data = None
    
    def _load_corrupted_data(self) -> Optional[np.ndarray]:
        """Load corrupted CIFAR-100-C data if available."""
        corruption_file = os.path.join(
            self.root, 
            f'CIFAR-100-C',
            f'{self.corruption_type}_{self.corruption_level}.npy'
        )
        
        if os.path.exists(corruption_file):
            return np.load(corruption_file)
        else:
            print(f"Warning: Corruption file {corruption_file} not found. Using clean data.")
            return None
    
    def __len__(self) -> int:
        return len(self.cifar100)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Get base image and label
        if self.corrupted_data is not None:
            # Use corrupted image
            image = self.corrupted_data[index]
            # Convert from numpy to PIL for transform compatibility
            from PIL import Image
            image = Image.fromarray(image.astype(np.uint8))
            _, label = self.cifar100[index]
        else:
            # Use clean image
            image, label = self.cifar100[index]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_cifar100c_data_loaders(
    data_dir: str, 
    batch_size: int = 128, 
    num_workers: int = 4,
    corruption_types: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create data loaders for CIFAR-100-C experiments.
    
    Args:
        data_dir: Directory containing CIFAR-100 data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        corruption_types: List of corruption types to test
        
    Returns:
        Tuple of (train_loader, clean_test_loader, corruption_test_loaders)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 stats
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Create training dataset and loader
    train_dataset = CIFAR100CDataset(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create clean test dataset and loader
    clean_test_dataset = CIFAR100CDataset(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )
    
    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create corrupted test loaders
    corruption_loaders = {}
    
    if corruption_types is None:
        # Default corruption types for CIFAR-100-C
        corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    
    for corruption_type in corruption_types:
        try:
            corrupted_dataset = CIFAR100CDataset(
                root=data_dir,
                train=False,
                corruption_type=corruption_type,
                corruption_level=3,  # Medium corruption level
                transform=test_transform,
                download=False
            )
            
            corrupted_loader = DataLoader(
                corrupted_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            corruption_loaders[corruption_type] = corrupted_loader
            
        except Exception as e:
            print(f"Warning: Could not load corruption type {corruption_type}: {e}")
    
    return train_loader, clean_test_loader, corruption_loaders


def load_cifar100c_dataset(
    data_dir: str,
    corruption_type: Optional[str] = None,
    corruption_level: int = 3,
    train: bool = True
) -> CIFAR100CDataset:
    """
    Load CIFAR-100-C dataset with specified corruption.
    
    Args:
        data_dir: Data directory
        corruption_type: Type of corruption (None for clean data)
        corruption_level: Corruption severity level (1-5)
        train: Whether to load training or test set
        
    Returns:
        CIFAR100CDataset instance
    """
    
    # Choose appropriate transform
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    
    return CIFAR100CDataset(
        root=data_dir,
        train=train,
        corruption_type=corruption_type,
        corruption_level=corruption_level,
        transform=transform,
        download=True
    )


def get_cifar100_class_names() -> List[str]:
    """Get CIFAR-100 class names."""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]


def get_corruption_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available corruption types."""
    return {
        'noise': {
            'types': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
            'description': 'Noise-based corruptions'
        },
        'blur': {
            'types': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
            'description': 'Blur-based corruptions'
        },
        'weather': {
            'types': ['snow', 'frost', 'fog', 'brightness'],
            'description': 'Weather and lighting corruptions'
        },
        'digital': {
            'types': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
            'description': 'Digital processing corruptions'
        }
    }


if __name__ == "__main__":
    # Test the data loading functionality
    import tempfile
    
    print("Testing CIFAR-100-C data loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test basic data loading
        try:
            train_loader, test_loader, corruption_loaders = create_cifar100c_data_loaders(
                temp_dir, batch_size=32
            )
            
            print(f"Created train loader with {len(train_loader.dataset)} samples")
            print(f"Created test loader with {len(test_loader.dataset)} samples")
            print(f"Created {len(corruption_loaders)} corruption loaders")
            
            # Test a batch
            for images, labels in train_loader:
                print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
                break
                
            print("CIFAR-100-C data loading test successful!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("Note: Full functionality requires CIFAR-100-C corruption files")