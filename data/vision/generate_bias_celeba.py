#!/usr/bin/env python3
"""
Biased CelebA-style Dataset Generator for Multi-Feature Bias Research

This script generates biased face-like datasets with configurable two-feature bias,
where any two features can correlate with labels in training but not in test data.
This allows studying bias in more complex scenarios like CelebA.

Usage:
    python generate_bias_celeba.py --feature1 gender --feature2 hair_color --train_bias 0.8 --test_bias 0.2
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union


def generate_face_features(
    feature1_value: int,
    feature2_value: int, 
    feature1_name: str,
    feature2_name: str,
    size: int = 64
) -> np.ndarray:
    """Generate basic face features with configurable attributes"""
    image = np.zeros((size, size))
    center_x, center_y = size // 2, size // 2
    
    # Base face outline (oval)
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
    
    # Apply feature1 modifications
    if feature1_name == "gender":
        if feature1_value == 1:  # Female
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
    
    elif feature1_name == "age":
        if feature1_value == 1:  # Young
            # Smoother features (higher base intensity)
            image[face_mask] = 0.9
        else:  # Old
            # Add wrinkles (lines around eyes)
            for offset in [-3, -2, 2, 3]:
                wrinkle_mask = (x - left_eye_x)**2 + (y - (eye_y + offset))**2 <= (eye_size//3)**2
                image[wrinkle_mask] = 0.5
                wrinkle_mask = (x - right_eye_x)**2 + (y - (eye_y + offset))**2 <= (eye_size//3)**2
                image[wrinkle_mask] = 0.5
    
    elif feature1_name == "hair_color":
        # Hair color affects the hair intensity
        base_hair_intensity = 0.7 if feature1_value == 1 else 0.3  # 1=dark, 0=light
        hair_mask = ((x - center_x) / (size * 0.42))**2 + ((y - center_y + size * 0.08) / (size * 0.45))**2 <= 1
        image[hair_mask] = np.maximum(image[hair_mask], base_hair_intensity)
    
    # Apply feature2 modifications
    if feature2_name == "hair_color" and feature1_name != "hair_color":
        base_hair_intensity = 0.7 if feature2_value == 1 else 0.3
        hair_mask = ((x - center_x) / (size * 0.42))**2 + ((y - center_y + size * 0.08) / (size * 0.45))**2 <= 1
        image[hair_mask] = np.maximum(image[hair_mask], base_hair_intensity)
    
    elif feature2_name == "smiling":
        if feature2_value == 1:  # Smiling
            # Curved mouth
            mouth_y = center_y + size // 4
            for offset in range(-2, 3):
                curve_y = mouth_y + abs(offset) // 2
                mouth_mask = (x - (center_x + offset * 2))**2 + (y - curve_y)**2 <= (size // 25)**2
                image[mouth_mask] = 0.2
        # No specific modification for non-smiling (default mouth is neutral)
    
    elif feature2_name == "glasses":
        if feature2_value == 1:  # Wearing glasses
            # Glasses frames around eyes
            glasses_thickness = 2
            # Left lens
            for r in range(eye_size, eye_size + glasses_thickness):
                glasses_mask = (x - left_eye_x)**2 + (y - eye_y)**2 <= r**2
                border_mask = (x - left_eye_x)**2 + (y - eye_y)**2 <= (r - glasses_thickness)**2
                image[glasses_mask & ~border_mask] = 0.1
            
            # Right lens
            for r in range(eye_size, eye_size + glasses_thickness):
                glasses_mask = (x - right_eye_x)**2 + (y - eye_y)**2 <= r**2
                border_mask = (x - right_eye_x)**2 + (y - eye_y)**2 <= (r - glasses_thickness)**2
                image[glasses_mask & ~border_mask] = 0.1
            
            # Bridge
            image[eye_y-1:eye_y+2, left_eye_x+eye_size:right_eye_x-eye_size] = 0.1
    
    return image


def generate_background(bg_type: str, size: int = 64) -> np.ndarray:
    """Generate different background types"""
    background = np.zeros((size, size))
    
    if bg_type == 'indoor':
        # Indoor background - vertical and horizontal lines (like walls, furniture)
        for i in range(0, size, size//8):
            background[:, i:i+2] = 0.3
        for i in range(0, size, size//6):
            background[i:i+2, :] = 0.2
            
    elif bg_type == 'outdoor':
        # Outdoor background - more organic patterns (like trees, sky)
        for _ in range(20):
            center_x = np.random.randint(size//4, 3*size//4)
            center_y = np.random.randint(size//4, 3*size//4)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            background[mask] = np.random.uniform(0.1, 0.4)
            
    elif bg_type == 'plain':
        # Plain background
        background.fill(0.1)
        
    elif bg_type == 'textured':
        # Textured background
        background = np.random.uniform(0.1, 0.3, (size, size))
    
    return background


class BiasedCelebADataset(Dataset):
    """Dataset that creates biased face images with two configurable features"""
    
    def __init__(
        self,
        num_samples: int,
        bias_strength: float,
        feature1_name: str = "gender",
        feature2_name: str = "hair_color",
        feature1_values: List[str] = None,
        feature2_values: List[str] = None,
        seed: int = 42,
        size: int = 64
    ):
        self.num_samples = num_samples
        self.bias_strength = bias_strength
        self.feature1_name = feature1_name
        self.feature2_name = feature2_name
        self.size = size
        
        # Default feature values
        if feature1_values is None:
            feature1_values = ["male", "female"] if feature1_name == "gender" else ["no", "yes"]
        if feature2_values is None:
            feature2_values = ["light", "dark"] if feature2_name == "hair_color" else ["no", "yes"]
        
        self.feature1_values = feature1_values
        self.feature2_values = feature2_values
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate biased face data with two features"""
        self.images = []
        self.feature1_labels = []
        self.feature2_labels = []
        self.bias_followed = []
        
        for _ in range(self.num_samples):
            # Random feature values
            feature1_val = np.random.randint(0, 2)
            feature2_val = np.random.randint(0, 2)
            
            # Determine background based on bias
            # Bias pattern: feature1=0 & feature2=0 -> indoor (bias_strength prob)
            #               feature1=1 & feature2=1 -> outdoor (bias_strength prob)
            #               Others -> random
            
            if (feature1_val == 0 and feature2_val == 0):
                # Expected: indoor background
                if np.random.random() < self.bias_strength:
                    bg_type = 'indoor'
                    bias_followed = True
                else:
                    bg_type = 'outdoor'
                    bias_followed = False
            elif (feature1_val == 1 and feature2_val == 1):
                # Expected: outdoor background  
                if np.random.random() < self.bias_strength:
                    bg_type = 'outdoor'
                    bias_followed = True
                else:
                    bg_type = 'indoor'
                    bias_followed = False
            else:
                # Mixed cases - random background
                bg_type = np.random.choice(['indoor', 'outdoor'])
                bias_followed = False
            
            # Generate face and background
            face = generate_face_features(
                feature1_val, feature2_val, 
                self.feature1_name, self.feature2_name, 
                self.size
            )
            background = generate_background(bg_type, self.size)
            
            # Combine (face is foreground, background where face is empty)
            combined = np.maximum(face, background * (face < 0.1))
            
            # Convert to RGB
            image_rgb = np.stack([combined, combined, combined], axis=0)
            
            self.images.append(torch.tensor(image_rgb, dtype=torch.float32))
            self.feature1_labels.append(feature1_val)
            self.feature2_labels.append(feature2_val)
            self.bias_followed.append(bias_followed)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], {
            'feature1': self.feature1_labels[idx],
            'feature2': self.feature2_labels[idx],
            'bias_followed': self.bias_followed[idx],
            'feature1_name': self.feature1_name,
            'feature2_name': self.feature2_name
        }


def create_biased_celeba_datasets(
    train_bias: float = 0.8,
    test_bias: float = 0.2,
    train_size: int = 8000,
    test_size: int = 2000,
    feature1_name: str = "gender",
    feature2_name: str = "hair_color",
    feature1_values: List[str] = None,
    feature2_values: List[str] = None,
    seed: int = 42
) -> Tuple[BiasedCelebADataset, BiasedCelebADataset, Dict]:
    """Create biased CelebA-style train and test datasets"""
    
    train_dataset = BiasedCelebADataset(
        train_size, train_bias, feature1_name, feature2_name,
        feature1_values, feature2_values, seed
    )
    
    test_dataset = BiasedCelebADataset(
        test_size, test_bias, feature1_name, feature2_name,
        feature1_values, feature2_values, seed + 1
    )
    
    # Metadata
    metadata = {
        'train_bias': train_bias,
        'test_bias': test_bias,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'feature1_name': feature1_name,
        'feature2_name': feature2_name,
        'feature1_values': train_dataset.feature1_values,
        'feature2_values': train_dataset.feature2_values,
        'image_shape': [3, 64, 64],
        'seed': seed,
        'bias_pattern': f"{feature1_name}={feature1_values[0]}&{feature2_name}={feature2_values[0]} -> indoor, {feature1_name}={feature1_values[1]}&{feature2_name}={feature2_values[1]} -> outdoor"
    }
    
    return train_dataset, test_dataset, metadata


def analyze_dataset_bias(dataset: BiasedCelebADataset, name: str) -> Dict:
    """Analyze the bias in a biased CelebA dataset"""
    feature_counts = {}
    bias_counts = {'followed': 0, 'violated': 0}
    
    for i in range(len(dataset)):
        _, metadata = dataset[i]
        feature1_val = metadata['feature1']
        feature2_val = metadata['feature2']
        bias_followed = metadata['bias_followed']
        
        # Count feature combinations
        key = (feature1_val, feature2_val)
        if key not in feature_counts:
            feature_counts[key] = 0
        feature_counts[key] += 1
        
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
    print(f"  Feature combination distribution:")
    
    f1_name = dataset.feature1_name
    f2_name = dataset.feature2_name
    f1_vals = dataset.feature1_values
    f2_vals = dataset.feature2_values
    
    for (f1, f2), count in feature_counts.items():
        f1_str = f1_vals[f1]
        f2_str = f2_vals[f2]
        print(f"    {f1_name}={f1_str}, {f2_name}={f2_str}: {count} samples")
    
    return {
        'actual_bias': actual_bias,
        'feature_counts': feature_counts,
        'bias_counts': bias_counts
    }


def visualize_biased_celeba(dataset: BiasedCelebADataset, num_samples: int = 16, save_path: str = None):
    """Create a visualization of the biased CelebA dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Biased CelebA-style Dataset ({dataset.feature1_name} vs {dataset.feature2_name})', fontsize=16)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            image, metadata = dataset[indices[idx]]
            
            # Convert to numpy and transpose for matplotlib
            img_np = image.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np, cmap='gray')
            
            f1_val = dataset.feature1_values[metadata['feature1']]
            f2_val = dataset.feature2_values[metadata['feature2']]
            bias_str = "✓" if metadata['bias_followed'] else "✗"
            
            ax.set_title(f'{dataset.feature1_name}: {f1_val}\n{dataset.feature2_name}: {f2_val}\nBias: {bias_str}', 
                        fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_biased_celeba_dataset(
    train_dataset: BiasedCelebADataset,
    test_dataset: BiasedCelebADataset,
    metadata: Dict,
    output_dir: str
):
    """Save biased CelebA dataset to files"""
    output_path = Path(output_dir)
    
    # Save dataset with informative subdir
    f1_name = metadata['feature1_name']
    f2_name = metadata['feature2_name']
    bias_str = f"trainbias_{metadata['train_bias']}_testbias_{metadata['test_bias']}"
    subdir = f"{f1_name}_{f2_name}_{bias_str}"
    output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    torch.save(train_dataset, output_path / "train_dataset.pt")
    torch.save(test_dataset, output_path / "test_dataset.pt")
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_biased_celeba(train_dataset, 16, str(output_path / "train_samples.png"))
    visualize_biased_celeba(test_dataset, 16, str(output_path / "test_samples.png"))
    
    print(f"Biased CelebA dataset saved to {output_path}")


def load_biased_celeba_dataset(data_dir: str) -> Tuple[BiasedCelebADataset, BiasedCelebADataset, Dict]:
    """Load biased CelebA dataset from files"""
    data_path = Path(data_dir)
    
    train_dataset = torch.load(data_path / "train_dataset.pt")
    test_dataset = torch.load(data_path / "test_dataset.pt")
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate biased CelebA-style dataset for multi-feature bias research")
    parser.add_argument("--train_bias", type=float, default=0.8,
                       help="Bias strength in training set")
    parser.add_argument("--test_bias", type=float, default=0.2,
                       help="Bias strength in test set")
    parser.add_argument("--train_size", type=int, default=8000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Number of test samples")
    parser.add_argument("--feature1", type=str, default="gender",
                       choices=["gender", "age", "hair_color"],
                       help="First feature for bias")
    parser.add_argument("--feature2", type=str, default="hair_color", 
                       choices=["hair_color", "smiling", "glasses"],
                       help="Second feature for bias")
    parser.add_argument("--output_dir", type=str, default="./bias_celeba_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate feature combination
    if args.feature1 == args.feature2:
        raise ValueError("feature1 and feature2 must be different")
    
    print("Generating Biased CelebA-style Dataset")
    print("=" * 45)
    print(f"Train bias: {args.train_bias}")
    print(f"Test bias: {args.test_bias}")
    print(f"Feature 1: {args.feature1}")
    print(f"Feature 2: {args.feature2}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_biased_celeba_datasets(
        train_bias=args.train_bias,
        test_bias=args.test_bias,
        train_size=args.train_size,
        test_size=args.test_size,
        feature1_name=args.feature1,
        feature2_name=args.feature2,
        seed=args.seed
    )
    
    # Analyze bias
    train_analysis = analyze_dataset_bias(train_dataset, "Training")
    test_analysis = analyze_dataset_bias(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_biased_celeba_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()