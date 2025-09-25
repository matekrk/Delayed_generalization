#!/usr/bin/env python3
"""
CIFAR-10-C Dataset Generator for Robustness Research

This script creates CIFAR-10-C datasets by applying various corruptions
to the real CIFAR-10 dataset to study robustness and delayed generalization.

Usage:
    python generate_cifar10c.py --corruptions noise blur --severity 3 --output_dir ./cifar10c_data
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
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.vision.cifar10c.generate_synthetic_cifar10c import SyntheticCIFAR10CDataset


class CIFAR10CDataset(Dataset):
    """Dataset that applies corruptions to CIFAR-10 images"""
    
    def __init__(
        self,
        cifar10_data: torch.utils.data.Dataset,
        corruption_type: str,
        severity: int = 3,
        seed: int = 42
    ):
        self.cifar10_data = cifar10_data
        self.corruption_type = corruption_type
        self.severity = severity
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def __len__(self):
        return len(self.cifar10_data)
    
    def __getitem__(self, idx):
        image, label = self.cifar10_data[idx]
        
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
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    def _gaussian_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add Gaussian noise"""
        noise_levels = [0.08, 0.12, 0.18, 0.26, 0.38]
        noise_level = noise_levels[severity - 1]
        
        img_array = np.array(image) / 255.0
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))
    
    def _shot_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add shot noise"""
        noise_levels = [60, 25, 12, 5, 3]
        lam = noise_levels[severity - 1]
        
        img_array = np.array(image)
        noisy_img = np.random.poisson(img_array / 255.0 * lam) / lam * 255
        return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))
    
    def _impulse_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add impulse (salt-and-pepper) noise"""
        noise_levels = [0.03, 0.06, 0.09, 0.17, 0.27]
        noise_level = noise_levels[severity - 1]
        
        img_array = np.array(image)
        mask = np.random.random(img_array.shape[:2]) < noise_level
        img_array[mask] = np.random.choice([0, 255])
        return Image.fromarray(img_array)
    
    def _defocus_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply defocus blur"""
        blur_levels = [0.3, 0.4, 0.6, 0.8, 1.0]
        radius = blur_levels[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _glass_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply glass blur effect"""
        blur_levels = [0.7, 0.9, 1.1, 1.3, 1.6]
        sigma = blur_levels[severity - 1]
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create random displacement field
        dx = np.random.uniform(-sigma, sigma, (h, w)).astype(np.float32)
        dy = np.random.uniform(-sigma, sigma, (h, w)).astype(np.float32)
        
        # Create coordinate matrices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply remapping
        result = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR)
        return Image.fromarray(result)
    
    def _motion_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply motion blur"""
        blur_levels = [6, 8, 10, 12, 15]
        kernel_size = blur_levels[severity - 1]
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        img_array = np.array(image)
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred)
    
    def _zoom_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply zoom blur"""
        zoom_levels = [1.01, 1.02, 1.03, 1.04, 1.06]
        zoom_factor = zoom_levels[severity - 1]
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create zoom effect by averaging multiple scaled versions
        result = img_array.astype(np.float32)
        for i in range(1, 10):
            scale = 1 + (zoom_factor - 1) * i / 10
            new_h, new_w = int(h / scale), int(w / scale)
            resized = cv2.resize(img_array, (new_w, new_h))
            # Pad to original size
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            padded = np.pad(resized, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)), 
                          mode='edge')
            result += padded.astype(np.float32)
        
        result = result / 10
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _snow(self, image: Image.Image, severity: int) -> Image.Image:
        """Add snow effect"""
        snow_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
        snow_amount = snow_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create snow flakes
        snow_mask = np.random.random((h, w)) < snow_amount
        snow_img = img_array.copy()
        snow_img[snow_mask] = 255
        
        # Blend with original
        alpha = 0.7
        result = alpha * img_array + (1 - alpha) * snow_img
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _frost(self, image: Image.Image, severity: int) -> Image.Image:
        """Add frost effect"""
        frost_levels = [0.4, 0.5, 0.6, 0.7, 0.8]
        frost_amount = frost_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create frost pattern
        frost = np.random.random((h, w, 3)) * 255
        frost = cv2.GaussianBlur(frost, (3, 3), 0)
        
        # Blend with original
        result = (1 - frost_amount) * img_array + frost_amount * frost
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _fog(self, image: Image.Image, severity: int) -> Image.Image:
        """Add fog effect"""
        fog_levels = [1.5, 2.0, 2.5, 3.0, 3.5]
        fog_strength = fog_levels[severity - 1]
        
        enhancer = ImageEnhance.Brightness(image)
        fogged = enhancer.enhance(fog_strength)
        
        # Blend with original
        alpha = 0.7
        result = Image.blend(image, fogged, alpha)
        return result
    
    def _brightness(self, image: Image.Image, severity: int) -> Image.Image:
        """Adjust brightness"""
        bright_levels = [0.1, 0.2, 0.3, 0.4, 0.6]
        brightness = bright_levels[severity - 1]
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness)
    
    def _contrast(self, image: Image.Image, severity: int) -> Image.Image:
        """Adjust contrast"""
        contrast_levels = [0.4, 0.3, 0.2, 0.1, 0.05]
        contrast = contrast_levels[severity - 1]
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast)
    
    def _elastic_transform(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply elastic transformation"""
        alpha_levels = [244, 306, 408, 510, 612]
        sigma_levels = [4.0, 3.5, 3.0, 2.5, 2.0]
        
        alpha = alpha_levels[severity - 1]
        sigma = sigma_levels[severity - 1]
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create coordinate matrices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply transformation
        result = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR)
        return Image.fromarray(result)
    
    def _pixelate(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply pixelation"""
        pixel_levels = [0.6, 0.5, 0.4, 0.3, 0.25]
        pixel_factor = pixel_levels[severity - 1]
        
        w, h = image.size
        small_w, small_h = int(w * pixel_factor), int(h * pixel_factor)
        
        # Downscale then upscale
        small_img = image.resize((small_w, small_h), Image.NEAREST)
        pixelated = small_img.resize((w, h), Image.NEAREST)
        return pixelated
    
    def _jpeg_compression(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply JPEG compression"""
        quality_levels = [25, 18, 15, 10, 7]
        quality = quality_levels[severity - 1]
        
        # Save to bytes buffer and reload to simulate compression
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        return compressed.convert('RGB')


def create_cifar10c_datasets(
    data_dir: str = "./data",
    corruption_types: List[str] = None,
    severity: int = 3,
    download: bool = True,
    seed: int = 42
) -> Tuple[Dict[str, CIFAR10CDataset], Dict[str, CIFAR10CDataset], Dict]:
    """
    Create CIFAR-10-C train and test datasets for multiple corruption types
    
    Args:
        data_dir: Directory to store CIFAR-10 data
        corruption_types: List of corruption types to apply
        severity: Corruption severity (1-5)
        download: Whether to download CIFAR-10
        seed: Random seed
        
    Returns:
        train_datasets, test_datasets, metadata
    """
    
    if corruption_types is None:
        corruption_types = ['gaussian_noise', 'defocus_blur', 'brightness']
    
    # Download and load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Use minimal transform for corruption application
    minimal_transform = transforms.ToTensor()
    
    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=minimal_transform
    )
    
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=minimal_transform
    )
    
    # Create corrupted versions
    train_datasets = {}
    test_datasets = {}
    
    for corruption_type in corruption_types:
        train_datasets[corruption_type] = CIFAR10CDataset(
            cifar10_train, corruption_type, severity, seed
        )
        
        test_datasets[corruption_type] = CIFAR10CDataset(
            cifar10_test, corruption_type, severity, seed + 1
        )
    
    # Metadata
    metadata = {
        'corruption_types': corruption_types,
        'severity': severity,
        'train_size': len(cifar10_train),
        'test_size': len(cifar10_test),
        'num_classes': 10,
        'class_names': train_datasets[corruption_types[0]].class_names,
        'image_shape': [3, 32, 32],
        'seed': seed
    }
    
    return train_datasets, test_datasets, metadata


def analyze_cifar10c_dataset(dataset: CIFAR10CDataset, name: str) -> Dict:
    """Analyze a CIFAR-10-C dataset"""
    class_counts = {}
    
    for i in range(min(1000, len(dataset))):  # Sample for efficiency
        _, label, metadata = dataset[i]
        class_name = metadata['class_name']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print(f"\n{name} Dataset Analysis:")
    print(f"  Corruption: {dataset.corruption_type}")
    print(f"  Severity: {dataset.severity}")
    print(f"  Sampled {min(1000, len(dataset))} examples")
    print(f"  Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"    {class_name}: {count} samples")
    
    return {
        'corruption_type': dataset.corruption_type,
        'severity': dataset.severity,
        'class_counts': class_counts
    }


def visualize_cifar10c(datasets: Dict[str, CIFAR10CDataset], num_samples: int = 5, save_path: str = None):
    """Create a visualization comparing corruptions"""
    corruption_types = list(datasets.keys())
    fig, axes = plt.subplots(len(corruption_types), num_samples, figsize=(15, 3 * len(corruption_types)))
    
    if len(corruption_types) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('CIFAR-10-C Corruption Examples', fontsize=16)
    
    for row, corruption_type in enumerate(corruption_types):
        dataset = datasets[corruption_type]
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for col, idx in enumerate(indices):
            image, label, metadata = dataset[idx]
            
            # Convert tensor to numpy
            img_np = image.permute(1, 2, 0).numpy()
            # Denormalize if needed
            if img_np.min() < 0:
                img_np = img_np * 0.5 + 0.5
            img_np = np.clip(img_np, 0, 1)
            
            axes[row, col].imshow(img_np)
            if col == 0:
                axes[row, col].set_ylabel(f'{corruption_type}\n(severity {metadata["severity"]})', 
                                        rotation=0, ha='right', va='center')
            axes[row, col].set_title(f'{metadata["class_name"]}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_cifar10c_datasets(
    train_datasets: Dict[str, CIFAR10CDataset],
    test_datasets: Dict[str, CIFAR10CDataset],
    metadata: Dict,
    output_dir: str
):
    """Save CIFAR-10-C datasets to files"""
    output_path = Path(output_dir)
    corrupted_path = output_path / "cifar10c"
    corrupted_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets with informative subdir
    # corruption_str = "_".join(metadata['corruption_types'])
    # severity_str = f"severity_{metadata['severity']}"
    # subdir = f"cifar10c_{corruption_str}_{severity_str}"
    # output_path = output_path / subdir
    # output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets for each corruption type
    for corruption_type in metadata['corruption_types']:
        severity_str = f"severity_{metadata['severity']}"
        subdir = f"{corruption_type}_{severity_str}"
        output_path = corrupted_path / subdir
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(train_datasets[corruption_type], output_path / "train_dataset.pt")
        torch.save(test_datasets[corruption_type], output_path / "test_dataset.pt")
    
    # Save metadata
    with open(corrupted_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_cifar10c(train_datasets, 5, str(corrupted_path / "train_corruptions.png"))
    visualize_cifar10c(test_datasets, 5, str(corrupted_path / "test_corruptions.png"))
    
    print(f"CIFAR-10-C datasets saved to {corrupted_path} with each corruption type in its own subdirectory.")

def load_cifar10c_dataset(data_dir: str):
    """Load CIFAR-10-C datasets saved by save_cifar10c_datasets.

    Supports:
      - current format where each corruption subdir contains pickled CIFAR10CDataset
        objects saved with torch.save(...)
      - legacy dict-based dumps that contain precomputed 'images' and 'labels'

    Returns:
      train_dataset, test_dataset, metadata
    """
    data_path = Path(data_dir)

    # If user passed a file, use its parent dir
    if data_path.is_file():
        data_path = data_path.parent

    train_path = data_path / "train_dataset.pt"
    test_path = data_path / "test_dataset.pt"
    metadata_path = data_path / "metadata.json"

    # If not found at this level, try to find a single corruption subdir under it
    if not (train_path.exists() and test_path.exists()):
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            # use the only subdir
            data_path = subdirs[0]
            train_path = data_path / "train_dataset.pt"
            test_path = data_path / "test_dataset.pt"
            metadata_path = data_path.parent / "metadata.json"
        else:
            raise FileNotFoundError(
                f"Could not find train_dataset.pt/test_dataset.pt in {data_dir} "
                "and could not infer a unique corruption subdirectory."
            )

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing dataset files in {data_path}: "
                                f"{'train_dataset.pt missing' if not train_path.exists() else ''} "
                                f"{'test_dataset.pt missing' if not test_path.exists() else ''}")

    # Load metadata (try provided location, then parent)
    metadata = {}
    if (metadata_path).exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        parent_meta = data_path.parent / "metadata.json"
        if parent_meta.exists():
            with open(parent_meta, "r") as f:
                metadata = json.load(f)
        else:
            # not fatal; continue but warn user
            metadata = {}

    # Helper to turn legacy dict dumps into a Dataset-like object
    class _LoadedPrecomputedDataset(Dataset):
        def __init__(self, data_dict):
            # expected keys: 'images' (N x H x W x C or list), 'labels' (N,)
            self.images = data_dict.get("images")
            self.labels = data_dict.get("labels")
            # normalize data types
            if isinstance(self.images, list):
                self.images = np.array(self.images)
            # ensure numpy array with shape (N, H, W, C)
            if isinstance(self.images, np.ndarray):
                # convert to uint8 if floats in 0-1
                if self.images.dtype == np.float32 or self.images.dtype == np.float64:
                    if self.images.max() <= 1.0:
                        self.images = (self.images * 255).astype(np.uint8)
                # ensure shape consistency
            else:
                raise TypeError("Unsupported images type in saved dict")
            self.labels = np.array(self.labels, dtype=np.int64)

        def __len__(self):
            return int(self.images.shape[0])

        def __getitem__(self, idx):
            img = self.images[int(idx)]
            # convert HWC numpy uint8 to tensor CxHxW float in [0,1]
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            else:
                # fallback to PIL->tensor
                img_tensor = transforms.ToTensor()(img)
            label = int(self.labels[int(idx)])
            # keep minimal metadata for compatibility
            metadata_local = {"source": "precomputed"}
            return img_tensor, label, metadata_local

    def _wrap_loaded(obj):
        # If it's already a CIFAR10CDataset (pickled), return as-is
        if isinstance(obj, CIFAR10CDataset):
            return obj
        # If it's a generic Dataset (saved directly), return as-is
        if isinstance(obj, Dataset):
            return obj
        # If it's a dict produced by older save routines, convert
        if isinstance(obj, dict):
            return _LoadedPrecomputedDataset(obj)
        # If torch.load returned something unexpected, raise
        raise TypeError(f"Loaded object type {type(obj)} not supported")

    # Load with torch.load (no extra args)
    train_obj = torch.load(train_path)
    test_obj = torch.load(test_path)

    train_dataset = _wrap_loaded(train_obj)
    test_dataset = _wrap_loaded(test_obj)

    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10-C dataset with corruptions")
    parser.add_argument("--data_dir", type=str, default="./cifar10_data",
                       help="Directory to store original CIFAR-10 data")
    parser.add_argument("--output_dir", type=str, default="./cifar10c_data",
                       help="Output directory for CIFAR-10-C dataset")
    parser.add_argument("--corruptions", type=str, nargs="+", 
                       default=["gaussian_noise", "defocus_blur", "brightness"],
                       choices=["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
                       help="Corruption types to apply")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help="Corruption severity (1=light, 5=heavy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_download", action="store_true", help="Don't download CIFAR-10")
    
    args = parser.parse_args()
    
    print("Generating CIFAR-10-C Dataset")
    print("=" * 30)
    print(f"Corruptions: {args.corruptions}")
    print(f"Severity: {args.severity}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_datasets, test_datasets, metadata = create_cifar10c_datasets(
        data_dir=args.data_dir,
        corruption_types=args.corruptions,
        severity=args.severity,
        download=not args.no_download,
        seed=args.seed
    )
    
    # Analyze datasets
    for corruption_type in args.corruptions:
        train_analysis = analyze_cifar10c_dataset(train_datasets[corruption_type], f"Training ({corruption_type})")
        test_analysis = analyze_cifar10c_dataset(test_datasets[corruption_type], f"Test ({corruption_type})")
    
    # Save datasets
    save_cifar10c_datasets(train_datasets, test_datasets, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()