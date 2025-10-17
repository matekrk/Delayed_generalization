#!/usr/bin/env python3
"""
CIFAR-100-C Dataset Generator for Robustness Research

This script creates CIFAR-100-C datasets by applying various corruptions
to the real CIFAR-100 dataset to study robustness and delayed generalization.

Usage:
    python generate_cifar100c.py --corruptions noise blur --severity 3 --output_dir ./cifar100c_data
"""

import argparse
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from PIL import Image, ImageFilter, ImageEnhance

# Try to import cv2, if not available use alternative implementations
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available, some corruption types will use simplified implementations")

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.vision.cifar100c.generate_synthetic_cifar100c import SyntheticCIFAR100CDataset


class CIFAR100CDataset(Dataset):
    """Dataset that applies corruptions to CIFAR-100 images"""
    
    def __init__(
        self,
        cifar100_data: torch.utils.data.Dataset,
        corruption_type: str,
        severity: int = 3,
        seed: int = 42
    ):
        self.cifar100_data = cifar100_data
        self.corruption_type = corruption_type
        self.severity = severity
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define CIFAR-100 class names (20 superclasses with 5 subclasses each)
        self.class_names = [
            # Aquatic mammals
            'beaver', 'dolphin', 'otter', 'seal', 'whale',
            # Fish
            'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            # Flowers
            'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            # Food containers
            'bottle', 'bowl', 'can', 'cup', 'plate',
            # Fruit and vegetables
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            # Household electrical devices
            'clock', 'computer_keyboard', 'lamp', 'telephone', 'television',
            # Household furniture
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            # Insects
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            # Large carnivores
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            # Large man-made outdoor things
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            # Large natural outdoor scenes
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            # Large omnivores and herbivores
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            # Medium-sized mammals
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            # Non-insect invertebrates
            'crab', 'lobster', 'snail', 'spider', 'worm',
            # People
            'baby', 'boy', 'girl', 'man', 'woman',
            # Reptiles
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            # Small mammals
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            # Trees
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            # Vehicles 1
            'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
            # Vehicles 2
            'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
        ]
    
    def __len__(self):
        return len(self.cifar100_data)
    
    def __getitem__(self, idx):
        image, label = self.cifar100_data[idx]
        
        # Convert tensor to PIL if necessary
        if isinstance(image, torch.Tensor):
            # Denormalize if normalized
            if image.min() < 0:  # Likely normalized
                image = image * 0.5 + 0.5  # Reverse common normalization
            image = transforms.ToPILImage()(image)
        
        # Apply corruption
        corrupted_image = self.apply_corruption(image, self.corruption_type, self.severity)
        
        # Convert back to tensor
        corrupted_tensor = transforms.ToTensor()(corrupted_image)
        
        metadata = {
            'corruption_type': self.corruption_type,
            'severity': self.severity,
            'class_name': self.class_names[label],
            'original_label': label
        }
        
        return corrupted_tensor, label, metadata
    
    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
        """Apply corruption to image based on type and severity"""
        severity = max(1, min(5, severity))  # Ensure severity is 1-5
        
        if corruption_type == 'gaussian_noise':
            return self._gaussian_noise(image, severity)
        elif corruption_type == 'shot_noise':
            return self._shot_noise(image, severity)
        elif corruption_type == 'impulse_noise':
            return self._impulse_noise(image, severity)
        elif corruption_type == 'defocus_blur':
            return self._defocus_blur(image, severity)
        elif corruption_type == 'glass_blur':
            return self._glass_blur(image, severity)
        elif corruption_type == 'motion_blur':
            return self._motion_blur(image, severity)
        elif corruption_type == 'zoom_blur':
            return self._zoom_blur(image, severity)
        elif corruption_type == 'snow':
            return self._snow(image, severity)
        elif corruption_type == 'frost':
            return self._frost(image, severity)
        elif corruption_type == 'fog':
            return self._fog(image, severity)
        elif corruption_type == 'brightness':
            return self._brightness(image, severity)
        elif corruption_type == 'contrast':
            return self._contrast(image, severity)
        elif corruption_type == 'elastic_transform':
            return self._elastic_transform(image, severity)
        elif corruption_type == 'pixelate':
            return self._pixelate(image, severity)
        elif corruption_type == 'jpeg_compression':
            return self._jpeg_compression(image, severity)
        else:
            return image
    
    def _gaussian_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add Gaussian noise"""
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        
        np_image = np.array(image) / 255.0
        noise = np.random.normal(size=np_image.shape) * c
        np_image = np.clip(np_image + noise, 0, 1)
        
        return Image.fromarray((np_image * 255).astype(np.uint8))
    
    def _shot_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add shot noise"""
        c = [60, 25, 12, 5, 3][severity - 1]
        
        np_image = np.array(image) / 255.0
        np_image = np.clip(np.random.poisson(np_image * c) / c, 0, 1)
        
        return Image.fromarray((np_image * 255).astype(np.uint8))
    
    def _impulse_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add impulse noise (salt and pepper)"""
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        
        np_image = np.array(image) / 255.0
        
        # Salt noise
        salt = np.random.random(np_image.shape) < c / 2
        np_image[salt] = 1
        
        # Pepper noise
        pepper = np.random.random(np_image.shape) < c / 2
        np_image[pepper] = 0
        
        return Image.fromarray((np_image * 255).astype(np.uint8))
    
    def _defocus_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply defocus blur"""
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        
        if CV2_AVAILABLE:
            np_image = np.array(image)
            kernel = self._disk(radius=c[0], alias_blur=c[1])
            
            channels = []
            for i in range(3):
                channels.append(cv2.filter2D(np_image[:, :, i], -1, kernel))
            
            return Image.fromarray(np.stack(channels, axis=2))
        else:
            # Fallback: use PIL blur
            radius = c[0] / 3.0  # Convert to PIL blur radius
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _glass_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply glass blur effect"""
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        
        np_image = np.array(image)
        h, w = np_image.shape[:2]
        
        # Generate glass effect
        for i in range(c[2]):
            for h_i in range(h - c[1], c[1], -1):
                for w_i in range(w - c[1], c[1], -1):
                    if np.random.random() < c[0]:
                        dx, dy = np.random.randint(-c[1], c[1], 2)
                        h_prime, w_prime = h_i + dy, w_i + dx
                        # Swap pixels
                        if 0 <= h_prime < h and 0 <= w_prime < w:
                            np_image[h_i, w_i], np_image[h_prime, w_prime] = \
                                np_image[h_prime, w_prime], np_image[h_i, w_i].copy()
        
        return Image.fromarray(np_image)
    
    def _motion_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply motion blur"""
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        
        if CV2_AVAILABLE:
            angle = np.random.uniform(-45, 45)
            kernel = self._motion_blur_kernel(c[0], angle)
            
            np_image = np.array(image)
            channels = []
            for i in range(3):
                channels.append(cv2.filter2D(np_image[:, :, i], -1, kernel))
            
            return Image.fromarray(np.stack(channels, axis=2))
        else:
            # Fallback: use PIL motion blur (simplified)
            radius = c[0] / 5.0
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _zoom_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply zoom blur"""
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]
        
        np_image = np.array(image) / 255.0
        h, w = np_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        out = np.zeros_like(np_image)
        for zoom_factor in c:
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            y1, x1 = max(0, center_y - new_h // 2), max(0, center_x - new_w // 2)
            y2, x2 = min(h, y1 + new_h), min(w, x1 + new_w)
            
            if CV2_AVAILABLE:
                zoomed = cv2.resize(np_image[y1:y2, x1:x2], (w, h))
            else:
                # Fallback: use PIL resize
                cropped = Image.fromarray((np_image[y1:y2, x1:x2] * 255).astype(np.uint8))
                zoomed_pil = cropped.resize((w, h), Image.BILINEAR)
                zoomed = np.array(zoomed_pil) / 255.0
            out += zoomed
        
        out /= len(c)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    
    def _snow(self, image: Image.Image, severity: int) -> Image.Image:
        """Add snow effect"""
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
        
        np_image = np.array(image) / 255.0
        snow_layer = np.random.normal(size=np_image.shape[:2]) * c[0] + c[1]
        snow_layer = np.clip(snow_layer, 0, 1)
        snow_layer = snow_layer[..., np.newaxis]
        
        np_image = c[6] * np_image + (1 - c[6]) * np.maximum(np_image, snow_layer)
        return Image.fromarray((np.clip(np_image, 0, 1) * 255).astype(np.uint8))
    
    def _frost(self, image: Image.Image, severity: int) -> Image.Image:
        """Add frost effect"""
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        
        np_image = np.array(image) / 255.0
        h, w = np_image.shape[:2]
        
        # Generate frost pattern
        frost = np.random.random((h, w))
        frost = (frost < c[0]).astype(float)
        
        if CV2_AVAILABLE:
            frost = cv2.GaussianBlur(frost, (9, 9), 0) * c[1]
        else:
            # Simplified frost without blur
            frost = frost * c[1]
            
        frost = frost[..., np.newaxis]
        
        np_image = np_image * (1 - frost) + frost
        return Image.fromarray((np.clip(np_image, 0, 1) * 255).astype(np.uint8))
        return Image.fromarray((np.clip(np_image, 0, 1) * 255).astype(np.uint8))
    
    def _fog(self, image: Image.Image, severity: int) -> Image.Image:
        """Add fog effect"""
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        
        np_image = np.array(image) / 255.0
        h, w = np_image.shape[:2]
        
        # Generate fog
        fog = np.random.normal(size=(h, w)) * 0.12 + 0.95
        fog = np.clip(fog, 0, 1)
        fog = fog[..., np.newaxis]
        
        np_image = c[1] * np_image + (1 - c[1]) * fog
        return Image.fromarray((np.clip(np_image, 0, 1) * 255).astype(np.uint8))
    
    def _brightness(self, image: Image.Image, severity: int) -> Image.Image:
        """Adjust brightness"""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1 + c)
    
    def _contrast(self, image: Image.Image, severity: int) -> Image.Image:
        """Adjust contrast"""
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(c)
    
    def _elastic_transform(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply elastic transformation"""
        if not CV2_AVAILABLE:
            # Fallback: return original image
            return image
            
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
        
        np_image = np.array(image)
        shape = np_image.shape
        
        dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (17, 17), c[0]) * c[2]
        dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (17, 17), c[1]) * c[2]
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        distorted_image = np.zeros_like(np_image)
        for i in range(shape[2]):
            distorted_image[:, :, i] = cv2.remap(np_image[:, :, i], 
                                                indices[1].astype(np.float32), 
                                                indices[0].astype(np.float32),
                                                cv2.INTER_LINEAR)
        
        return Image.fromarray(distorted_image)
    
    def _pixelate(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply pixelation effect"""
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        
        w, h = image.size
        image = image.resize((int(w * c), int(h * c)), Image.BOX)
        image = image.resize((w, h), Image.BOX)
        
        return image
    
    def _jpeg_compression(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply JPEG compression"""
        c = [25, 18, 15, 10, 7][severity - 1]
        
        import io
        output = io.BytesIO()
        image.save(output, 'JPEG', quality=c)
        output.seek(0)
        
        return Image.open(output)
    
    def _disk(self, radius: int, alias_blur: float = 0.1) -> np.ndarray:
        """Create a disk kernel for defocus blur"""
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = 3
        else:
            L = np.arange(-radius, radius + 1)
            ksize = 3
        
        X, Y = np.meshgrid(L, L)
        mask = X**2 + Y**2 <= radius**2
        kernel = mask.astype(float)
        
        if alias_blur > 0 and CV2_AVAILABLE:
            kernel = cv2.GaussianBlur(kernel, (ksize, ksize), alias_blur)
        
        return kernel / kernel.sum() if kernel.sum() > 0 else kernel
    
    def _motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Create line kernel
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_val)
            y = int(center + offset * sin_val)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        return kernel / kernel.sum() if kernel.sum() > 0 else kernel


def create_cifar100c_datasets(
    corruptions: List[str],
    severities: List[int],
    data_dir: str = "./data/cifar100c",
    train: bool = True,
    seed: int = 42
) -> Dict[str, CIFAR100CDataset]:
    """Create CIFAR-100-C datasets for specified corruptions and severities"""
    
    # Download CIFAR-100 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=train, download=True, transform=transform
    )
    
    datasets = {}
    
    for corruption in corruptions:
        for severity in severities:
            key = f"{corruption}_severity_{severity}"
            datasets[key] = CIFAR100CDataset(
                cifar100_dataset, corruption, severity, seed
            )
    
    return datasets


def load_cifar100c_dataset(data_dir: str):
    """Load processed CIFAR-100-C dataset from directory"""
    try:
        # Try to load metadata
        metadata_path = Path(data_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            print(f"No metadata found at {metadata_path}")
            return None
    except Exception as e:
        print(f"Error loading CIFAR-100-C dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR-100-C corrupted dataset")
    parser.add_argument("--corruptions", nargs='+', 
                       default=['gaussian_noise', 'motion_blur', 'snow', 'brightness'],
                       help="Corruption types to apply")
    parser.add_argument("--severities", nargs='+', type=int,
                       default=[1, 2, 3, 4, 5],
                       help="Severity levels (1-5)")
    parser.add_argument("--output_dir", type=str, default="./cifar100c_data",
                       help="Output directory for corrupted dataset")
    parser.add_argument("--train", action='store_true', default=False,
                       help="Use training set (default: test set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    print("Generating CIFAR-100-C Dataset")
    print("=" * 30)
    print(f"Corruptions: {args.corruptions}")
    print(f"Severities: {args.severities}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {'training' if args.train else 'test'} set")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate corrupted datasets
    datasets = create_cifar100c_datasets(
        args.corruptions, args.severities, args.output_dir, args.train, args.seed
    )
    
    print(f"Generated {len(datasets)} corrupted datasets")
    
    # Create visualizations
    plt.figure(figsize=(20, 4 * len(args.corruptions)))
    
    for i, corruption in enumerate(args.corruptions):
        for j, severity in enumerate(args.severities):
            key = f"{corruption}_severity_{severity}"
            if key in datasets:
                dataset = datasets[key]
                
                # Get a sample
                sample_idx = 0
                image, label, metadata = dataset[sample_idx]
                
                # Convert to numpy for visualization
                image_np = image.permute(1, 2, 0).numpy()
                
                plt.subplot(len(args.corruptions), len(args.severities), 
                           i * len(args.severities) + j + 1)
                plt.imshow(image_np)
                plt.title(f"{corruption}\nSeverity {severity}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "corruption_samples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metadata
    metadata = {
        'dataset': 'CIFAR-100-C',
        'corruptions': args.corruptions,
        'severities': args.severities,
        'num_classes': 100,
        'num_datasets': len(datasets),
        'train_set': args.train,
        'seed': args.seed,
        'class_names': datasets[list(datasets.keys())[0]].class_names
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Visualization saved to: {output_dir / 'corruption_samples.png'}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")
    print("CIFAR-100-C dataset generation complete!")


if __name__ == "__main__":
    main()