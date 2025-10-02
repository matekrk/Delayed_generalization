#!/usr/bin/env python3
"""
ImageNet-C Dataset Generator for Robustness Research

This script creates ImageNet-C datasets by applying various corruptions
to the real ImageNet dataset to study robustness and delayed generalization.

Usage:
    python generate_imagenetc.py --corruptions noise blur --severity 3 --output_dir ./imagenetc_data
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


class ImageNetCDataset(Dataset):
    """Dataset that applies corruptions to ImageNet images"""
    
    def __init__(
        self,
        imagenet_data: torch.utils.data.Dataset,
        corruption_type: str,
        severity: int = 3,
        seed: int = 42
    ):
        self.imagenet_data = imagenet_data
        self.corruption_type = corruption_type
        self.severity = severity
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def __len__(self):
        return len(self.imagenet_data)
    
    def __getitem__(self, idx):
        image, label = self.imagenet_data[idx]
        
        # Convert tensor to PIL if necessary
        if isinstance(image, torch.Tensor):
            # Denormalize if normalized
            if image.min() < 0:  # Likely normalized
                # Reverse ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = torch.clamp(image, 0, 1)
            image = transforms.ToPILImage()(image)
        
        # Apply corruption
        corrupted_image = self.apply_corruption(image, self.corruption_type, self.severity)
        
        # Convert back to tensor
        corrupted_tensor = transforms.ToTensor()(corrupted_image)
        
        metadata = {
            'corruption_type': self.corruption_type,
            'severity': self.severity,
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
        noise_levels = [0.04, 0.06, 0.08, 0.09, 0.10]
        noise_std = noise_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    def _shot_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add shot noise (Poisson noise)"""
        noise_levels = [60, 25, 12, 5, 3]
        lam = noise_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        noisy = np.random.poisson(img_array / 255.0 * lam) / lam * 255.0
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    
    def _impulse_noise(self, image: Image.Image, severity: int) -> Image.Image:
        """Add impulse noise (salt and pepper)"""
        noise_levels = [0.03, 0.06, 0.09, 0.17, 0.27]
        noise_amount = noise_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        salt = np.random.random(img_array.shape[:2]) < noise_amount / 2
        pepper = np.random.random(img_array.shape[:2]) < noise_amount / 2
        
        img_array[salt] = 255
        img_array[pepper] = 0
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _defocus_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply defocus blur"""
        blur_levels = [3, 4, 6, 8, 10]
        radius = blur_levels[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    def _glass_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply glass blur effect"""
        blur_levels = [0.7, 0.9, 1.0, 1.1, 1.5]
        sigma = blur_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Glass distortion
        for _ in range(2):
            dx = np.random.randint(-2, 3, (h, w))
            dy = np.random.randint(-2, 3, (h, w))
            
            x = np.clip(np.arange(w) + dx, 0, w - 1)
            y = np.clip(np.arange(h).reshape(-1, 1) + dy, 0, h - 1)
            
            img_array = img_array[y, x]
        
        # Additional Gaussian blur
        img_array = cv2.GaussianBlur(img_array, (0, 0), sigma)
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _motion_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply motion blur"""
        blur_levels = [10, 15, 15, 15, 20]
        kernel_size = blur_levels[severity - 1]
        
        img_array = np.array(image)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred)
    
    def _zoom_blur(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply zoom blur"""
        zoom_factors = [1.11, 1.16, 1.21, 1.26, 1.31]
        zoom_factor = zoom_factors[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        result = np.zeros_like(img_array)
        for i in range(10):
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
        snow_levels = [0.1, 0.2, 0.55, 0.7, 0.75]
        snow_amount = snow_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create snow flakes with appropriate scale
        snow_mask = np.random.random((h, w)) < snow_amount * 0.5
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
        frost = cv2.GaussianBlur(frost, (5, 5), 0)
        
        # Blend with original
        result = (1 - frost_amount) * img_array + frost_amount * frost
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _fog(self, image: Image.Image, severity: int) -> Image.Image:
        """Add fog effect"""
        fog_levels = [1.5, 2.0, 2.5, 3.0, 3.5]
        fog_strength = fog_levels[severity - 1]
        
        img_array = np.array(image).astype(np.float32) / 255.0
        h, w = img_array.shape[:2]
        
        # Create fog map
        max_val = img_array.max()
        fog = np.random.uniform(0.5, 1.0, (h, w, 3))
        
        # Apply fog with distance-based intensity
        result = img_array + fog_strength * 0.1 * fog
        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))
    
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
        # Scale appropriately for ImageNet (224x224 typical size)
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
        # Scale for ImageNet images
        pixel_levels = [0.6, 0.5, 0.4, 0.3, 0.25]
        scale = pixel_levels[severity - 1]
        
        w, h = image.size
        small_w, small_h = int(w * scale), int(h * scale)
        
        pixelated = image.resize((small_w, small_h), Image.NEAREST)
        return pixelated.resize((w, h), Image.NEAREST)
    
    def _jpeg_compression(self, image: Image.Image, severity: int) -> Image.Image:
        """Apply JPEG compression"""
        quality_levels = [25, 18, 15, 10, 7]
        quality = quality_levels[severity - 1]
        
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        return compressed.convert('RGB')


def create_imagenetc_datasets(
    data_dir: str = "./data",
    corruption_types: List[str] = None,
    severity: int = 3,
    download: bool = False,
    seed: int = 42,
    split: str = 'val'
) -> Tuple[Dict[str, ImageNetCDataset], Dict]:
    """
    Create ImageNet-C datasets for multiple corruption types
    
    Args:
        data_dir: Directory containing ImageNet data
        corruption_types: List of corruption types to apply
        severity: Corruption severity (1-5)
        download: Whether to download ImageNet (Note: ImageNet requires manual download)
        seed: Random seed
        split: Which split to use ('train' or 'val')
        
    Returns:
        datasets, metadata
    """
    
    if corruption_types is None:
        corruption_types = ['gaussian_noise', 'defocus_blur', 'brightness']
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # Use minimal transform for corruption application
    minimal_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Load ImageNet
    # Note: ImageNet requires manual download and setup
    try:
        imagenet_data = torchvision.datasets.ImageNet(
            root=data_dir, split=split, transform=minimal_transform
        )
    except:
        print(f"Warning: Could not load ImageNet from {data_dir}. Please ensure ImageNet is properly downloaded.")
        print("ImageNet must be manually downloaded from http://www.image-net.org/")
        raise RuntimeError("ImageNet dataset not found. Please download and place in correct directory.")
    
    # Create corrupted versions
    datasets = {}
    
    for corruption_type in corruption_types:
        datasets[corruption_type] = ImageNetCDataset(
            imagenet_data, corruption_type, severity, seed
        )
    
    # Metadata
    metadata = {
        'corruption_types': corruption_types,
        'severity': severity,
        'dataset_size': len(imagenet_data),
        'num_classes': 1000,
        'image_shape': [3, 224, 224],
        'seed': seed,
        'split': split
    }
    
    return datasets, metadata


def visualize_imagenetc(datasets: Dict[str, ImageNetCDataset], num_samples: int = 5, save_path: str = None):
    """Visualize corrupted ImageNet samples"""
    corruption_types = list(datasets.keys())
    
    fig, axes = plt.subplots(num_samples, len(corruption_types), 
                            figsize=(4 * len(corruption_types), 4 * num_samples))
    
    if len(corruption_types) == 1:
        axes = axes.reshape(-1, 1)
    
    for col, corruption_type in enumerate(corruption_types):
        dataset = datasets[corruption_type]
        
        for row in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, label, metadata = dataset[idx]
            
            # Convert tensor to numpy for display
            img_np = image.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[row, col].imshow(img_np)
            if col == 0:
                axes[row, col].set_ylabel(f'Sample {row+1}', rotation=0, ha='right', va='center')
            if row == 0:
                axes[row, col].set_title(f'{corruption_type}\n(severity {metadata["severity"]})')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_imagenetc_datasets(
    datasets: Dict[str, ImageNetCDataset],
    metadata: Dict,
    output_dir: str,
    wandb_logger: Optional[object] = None
):
    """Save ImageNet-C datasets to files"""
    output_path = Path(output_dir)
    corrupted_path = output_path / "imagenetc"
    corrupted_path.mkdir(parents=True, exist_ok=True)
    
    # Log dataset metadata to wandb if logger is provided
    if wandb_logger:
        try:
            import wandb
            enhanced_metadata = {
                **metadata,
                'output_path': str(corrupted_path),
                'corruption_count': len(metadata['corruption_types']),
                'total_samples': len(metadata['corruption_types']) * metadata['dataset_size'],
                'dataset_type': 'ImageNet-C'
            }
            
            wandb_logger.run.config.update({
                'dataset_metadata': enhanced_metadata
            })
            
            artifact = wandb.Artifact(
                name=f"imagenetc_metadata_severity_{metadata['severity']}",
                type="dataset_metadata",
                description=f"ImageNet-C dataset metadata for severity {metadata['severity']} with corruptions: {', '.join(metadata['corruption_types'])}"
            )
            
        except ImportError:
            print("Warning: wandb not available for metadata logging")
        except Exception as e:
            print(f"Warning: Failed to log metadata to wandb: {e}")
    
    # Save datasets for each corruption type
    for corruption_type in metadata['corruption_types']:
        severity_str = f"severity_{metadata['severity']}"
        subdir = f"{corruption_type}_{severity_str}"
        output_path = corrupted_path / subdir
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(datasets[corruption_type], output_path / "dataset.pt")
    
    # Save metadata
    with open(corrupted_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Complete wandb logging
    if wandb_logger:
        try:
            import wandb
            artifact.add_file(str(corrupted_path / "metadata.json"))
            wandb_logger.run.log_artifact(artifact)
            print(f"Dataset metadata logged to wandb project: {wandb_logger.run.project}")
        except Exception as e:
            print(f"Warning: Failed to complete wandb artifact logging: {e}")
    
    # Create visualizations
    visualize_imagenetc(datasets, 5, str(corrupted_path / "corruptions.png"))
    
    print(f"\nDatasets saved to: {corrupted_path}")
    print(f"Corruption types: {metadata['corruption_types']}")
    print(f"Severity: {metadata['severity']}")


def load_imagenetc_dataset(data_dir: str):
    """Load ImageNet-C datasets saved by save_imagenetc_datasets
    
    Returns:
        dataset, metadata
    """
    data_path = Path(data_dir)
    
    if data_path.is_file():
        data_path = data_path.parent
    
    dataset_path = data_path / "dataset.pt"
    metadata_path = data_path / "metadata.json"
    
    # If not found at this level, try to find a single corruption subdir
    if not dataset_path.exists():
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            data_path = subdirs[0]
            dataset_path = data_path / "dataset.pt"
            metadata_path = data_path.parent / "metadata.json"
        else:
            raise FileNotFoundError(
                f"Could not find dataset.pt in {data_dir} "
                "and could not infer a unique corruption subdirectory."
            )
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
    
    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        parent_meta = data_path.parent / "metadata.json"
        if parent_meta.exists():
            with open(parent_meta, "r") as f:
                metadata = json.load(f)
    
    # Load dataset
    dataset = torch.load(dataset_path)
    
    return dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate ImageNet-C dataset with corruptions")
    parser.add_argument("--data_dir", type=str, default="./imagenet_data",
                       help="Directory containing ImageNet data")
    parser.add_argument("--output_dir", type=str, default="./imagenetc_data",
                       help="Output directory for ImageNet-C dataset")
    parser.add_argument("--corruptions", type=str, nargs="+", 
                       default=["gaussian_noise", "defocus_blur", "brightness"],
                       choices=["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
                       help="Corruption types to apply")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help="Corruption severity (1=light, 5=heavy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                       help="ImageNet split to use")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name for logging dataset metadata")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs='*', default=None,
                       help="WandB tags")
    
    args = parser.parse_args()
    
    print("Generating ImageNet-C Dataset")
    print("=" * 30)
    print(f"Corruptions: {args.corruptions}")
    print(f"Severity: {args.severity}")
    print(f"Seed: {args.seed}")
    print(f"Split: {args.split}")
    
    # Setup WandB if requested
    wandb_logger = None
    if args.wandb_project:
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent.parent))
            from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
            
            config = {
                'corruption_types': args.corruptions,
                'severity': args.severity,
                'seed': args.seed,
                'split': args.split
            }
            
            experiment_name = args.wandb_name or f"imagenetc_generation_severity{args.severity}"
            
            wandb_logger = DelayedGeneralizationLogger(
                project_name=args.wandb_project,
                experiment_name=experiment_name,
                config=config,
                phenomenon_type='robustness',
                tags=(args.wandb_tags or []) + ['imagenetc', 'dataset_generation'],
                notes="ImageNet-C dataset generation for robustness research"
            )
            print(f"WandB logging enabled: {args.wandb_project}/{wandb_logger.run.name}")
        except Exception as e:
            print(f"Warning: Could not initialize WandB: {e}")
    
    # Create datasets
    print("\nCreating corrupted datasets...")
    datasets, metadata = create_imagenetc_datasets(
        data_dir=args.data_dir,
        corruption_types=args.corruptions,
        severity=args.severity,
        seed=args.seed,
        split=args.split
    )
    
    print(f"Created {len(datasets)} corruption variants")
    
    # Save datasets
    print("\nSaving datasets...")
    save_imagenetc_datasets(datasets, metadata, args.output_dir, wandb_logger)
    
    print("\nDataset generation complete!")
    
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
