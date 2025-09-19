#!/usr/bin/env python3
"""
Real CelebA Dataset Generator with Biased Features for Simplicity Bias Research

This script downloads and processes the real CelebA dataset with configurable bias
between any two attributes, similar to the colored MNIST approach. It creates
biased training and test sets where attribute correlations differ.

Usage:
    python generate_real_celeba.py --attr1 Male --attr2 Blond_Hair --train_bias 0.8 --test_bias 0.2
"""

import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import pandas as pd
from PIL import Image


class BiasedRealCelebADataset(Dataset):
    """Dataset that creates biased real CelebA with configurable two-attribute bias"""
    
    def __init__(
        self,
        celeba_dataset: torchvision.datasets.CelebA,
        indices: List[int],
        attr1_name: str = "Male",
        attr2_name: str = "Blond_Hair",
        bias_strength: float = 0.8,
        transform: Optional[transforms.Compose] = None,
        dataset_type: str = "train"
    ):
        self.celeba_dataset = celeba_dataset
        self.indices = indices
        self.attr1_name = attr1_name
        self.attr2_name = attr2_name
        self.bias_strength = bias_strength
        self.transform = transform
        self.dataset_type = dataset_type
        
        # Map attribute names to indices
        self.attr_names = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
            "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]
        
        self.attr1_idx = self.attr_names.index(attr1_name)
        self.attr2_idx = self.attr_names.index(attr2_name)
        
        # Prepare biased dataset
        self._prepare_biased_data()
    
    def _prepare_biased_data(self):
        """Prepare data with specified bias between two attributes"""
        # Get attributes for all samples
        all_attrs = []
        for idx in self.indices:
            _, attrs = self.celeba_dataset[idx]
            all_attrs.append(attrs)
        all_attrs = torch.stack(all_attrs)
        
        # Extract the two attributes of interest
        attr1_values = all_attrs[:, self.attr1_idx]  # 0 or 1
        attr2_values = all_attrs[:, self.attr2_idx]  # 0 or 1
        
        # Create label from first attribute (target we want to predict)
        labels = attr1_values
        
        # Calculate current correlation
        pos_label_mask = (labels == 1)
        neg_label_mask = (labels == 0)
        
        pos_attr2_corr = attr2_values[pos_label_mask].float().mean()
        neg_attr2_corr = attr2_values[neg_label_mask].float().mean()
        
        print(f"\n{self.dataset_type.capitalize()} set statistics:")
        print(f"Original {self.attr1_name}=1 & {self.attr2_name}=1 correlation: {pos_attr2_corr:.3f}")
        print(f"Original {self.attr1_name}=0 & {self.attr2_name}=1 correlation: {neg_attr2_corr:.3f}")
        
        # Create biased subset
        self.biased_indices = []
        self.bias_followed = []
        
        for i, idx in enumerate(self.indices):
            label = labels[i].item()
            attr2_val = attr2_values[i].item()
            
            # Determine if this sample follows the bias
            if label == 1:  # Positive label
                follows_bias = (attr2_val == 1)
            else:  # Negative label
                follows_bias = (attr2_val == 0)
            
            # Include sample based on bias strength
            if follows_bias:
                # Always include bias-following samples up to bias_strength
                if np.random.random() < self.bias_strength:
                    self.biased_indices.append(idx)
                    self.bias_followed.append(True)
            else:
                # Include bias-violating samples up to (1 - bias_strength)
                if np.random.random() < (1 - self.bias_strength):
                    self.biased_indices.append(idx)
                    self.bias_followed.append(False)
        
        print(f"Selected {len(self.biased_indices)} samples from {len(self.indices)} available")
        
        # Calculate final bias statistics
        final_labels = []
        final_attr2 = []
        for idx in self.biased_indices:
            _, attrs = self.celeba_dataset[idx]
            final_labels.append(attrs[self.attr1_idx].item())
            final_attr2.append(attrs[self.attr2_idx].item())
        
        final_labels = np.array(final_labels)
        final_attr2 = np.array(final_attr2)
        
        pos_mask = (final_labels == 1)
        neg_mask = (final_labels == 0)
        
        if pos_mask.sum() > 0:
            final_pos_corr = final_attr2[pos_mask].mean()
        else:
            final_pos_corr = 0
            
        if neg_mask.sum() > 0:
            final_neg_corr = final_attr2[neg_mask].mean()
        else:
            final_neg_corr = 0
        
        print(f"Final {self.attr1_name}=1 & {self.attr2_name}=1 correlation: {final_pos_corr:.3f}")
        print(f"Final {self.attr1_name}=0 & {self.attr2_name}=1 correlation: {final_neg_corr:.3f}")
    
    def __len__(self):
        return len(self.biased_indices)
    
    def __getitem__(self, idx):
        real_idx = self.biased_indices[idx]
        image, attrs = self.celeba_dataset[real_idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Target label is the first attribute
        label = attrs[self.attr1_idx].item()
        
        metadata = {
            'attr1': attrs[self.attr1_idx].item(),
            'attr2': attrs[self.attr2_idx].item(), 
            'bias_followed': self.bias_followed[idx],
            'attr1_name': self.attr1_name,
            'attr2_name': self.attr2_name,
            'original_index': real_idx
        }
        
        return image, label, metadata


def create_biased_real_celeba_datasets(
    data_root: str = "./celeba_data",
    attr1_name: str = "Male",
    attr2_name: str = "Blond_Hair", 
    train_bias: float = 0.8,
    test_bias: float = 0.2,
    train_size: int = 10000,
    test_size: int = 2000,
    image_size: int = 64,
    seed: int = 42
) -> Tuple[BiasedRealCelebADataset, BiasedRealCelebADataset, Dict]:
    """Create biased training and test datasets from real CelebA"""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    print(f"Loading CelebA dataset...")
    print(f"Data will be downloaded to: {data_root}")
    
    # Load full CelebA dataset
    try:
        full_dataset = torchvision.datasets.CelebA(
            root=data_root,
            split='all',  # Use all data, we'll split ourselves
            download=True,
            transform=None  # We'll apply transform in our custom dataset
        )
    except Exception as e:
        print(f"Error loading CelebA: {e}")
        print("Please ensure you have enough disk space and internet connection.")
        raise
    
    print(f"Loaded {len(full_dataset)} images from CelebA")
    
    # Filter for samples that have both attributes we're interested in
    valid_indices = []
    attr_stats = {f"{attr1_name}=0": 0, f"{attr1_name}=1": 0, 
                 f"{attr2_name}=0": 0, f"{attr2_name}=1": 0}
    
    attr_names = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
        "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]
    
    attr1_idx = attr_names.index(attr1_name)
    attr2_idx = attr_names.index(attr2_name)
    
    # Sample a subset for efficiency if the dataset is too large
    total_samples = min(len(full_dataset), 50000)  # Limit to reasonable size
    sample_indices = np.random.choice(len(full_dataset), total_samples, replace=False)
    
    for i in sample_indices:
        _, attrs = full_dataset[i]
        attr1_val = attrs[attr1_idx].item()
        attr2_val = attrs[attr2_idx].item()
        
        valid_indices.append(i)
        attr_stats[f"{attr1_name}={attr1_val}"] += 1
        attr_stats[f"{attr2_name}={attr2_val}"] += 1
    
    print(f"\nAttribute distribution in sampled data:")
    for key, count in attr_stats.items():
        print(f"  {key}: {count} ({count/len(valid_indices)*100:.1f}%)")
    
    # Split into train and test
    np.random.shuffle(valid_indices)
    
    # Ensure we have enough data
    total_needed = train_size + test_size
    if len(valid_indices) < total_needed:
        print(f"Warning: Only {len(valid_indices)} samples available, need {total_needed}")
        print(f"Reducing dataset sizes proportionally...")
        ratio = len(valid_indices) / total_needed
        train_size = int(train_size * ratio)
        test_size = len(valid_indices) - train_size
    
    train_indices = valid_indices[:train_size]
    test_indices = valid_indices[train_size:train_size + test_size]
    
    print(f"\nCreating datasets:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    # Create biased datasets
    train_dataset = BiasedRealCelebADataset(
        celeba_dataset=full_dataset,
        indices=train_indices,
        attr1_name=attr1_name,
        attr2_name=attr2_name,
        bias_strength=train_bias,
        transform=transform,
        dataset_type="train"
    )
    
    test_dataset = BiasedRealCelebADataset(
        celeba_dataset=full_dataset,
        indices=test_indices,
        attr1_name=attr1_name,
        attr2_name=attr2_name,
        bias_strength=test_bias,
        transform=transform,
        dataset_type="test"
    )
    
    # Create metadata
    metadata = {
        'attr1_name': attr1_name,
        'attr2_name': attr2_name,
        'train_bias': train_bias,
        'test_bias': test_bias,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'image_size': image_size,
        'seed': seed,
        'attribute_stats': attr_stats,
        'dataset_type': 'real_celeba'
    }
    
    return train_dataset, test_dataset, metadata


def analyze_dataset_bias(dataset: BiasedRealCelebADataset, dataset_name: str) -> Dict:
    """Analyze bias statistics in the dataset"""
    labels = []
    attr2_values = []
    bias_followed = []
    
    for i in range(len(dataset)):
        _, label, metadata = dataset[i]
        labels.append(label)
        attr2_values.append(metadata['attr2'])
        bias_followed.append(metadata['bias_followed'])
    
    labels = np.array(labels)
    attr2_values = np.array(attr2_values)
    bias_followed = np.array(bias_followed)
    
    # Calculate statistics
    total_samples = len(labels)
    pos_samples = (labels == 1).sum()
    neg_samples = (labels == 0).sum()
    
    # Correlation between attr1 and attr2
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    
    pos_attr2_mean = attr2_values[pos_mask].mean() if pos_mask.sum() > 0 else 0
    neg_attr2_mean = attr2_values[neg_mask].mean() if neg_mask.sum() > 0 else 0
    
    bias_following_rate = bias_followed.mean()
    
    analysis = {
        'total_samples': total_samples,
        'positive_samples': int(pos_samples),
        'negative_samples': int(neg_samples),
        'positive_ratio': pos_samples / total_samples,
        f'{dataset.attr1_name}=1_to_{dataset.attr2_name}=1_correlation': pos_attr2_mean,
        f'{dataset.attr1_name}=0_to_{dataset.attr2_name}=1_correlation': neg_attr2_mean,
        'bias_following_rate': bias_following_rate,
    }
    
    print(f"\n{dataset_name} Dataset Analysis:")
    print(f"  Total samples: {total_samples}")
    print(f"  {dataset.attr1_name}=1: {pos_samples} ({pos_samples/total_samples*100:.1f}%)")
    print(f"  {dataset.attr1_name}=0: {neg_samples} ({neg_samples/total_samples*100:.1f}%)")
    print(f"  {dataset.attr1_name}=1 → {dataset.attr2_name}=1 correlation: {pos_attr2_mean:.3f}")
    print(f"  {dataset.attr1_name}=0 → {dataset.attr2_name}=1 correlation: {neg_attr2_mean:.3f}")
    print(f"  Bias following rate: {bias_following_rate:.3f}")
    
    return analysis


def visualize_real_celeba(dataset: BiasedRealCelebADataset, num_samples: int = 16, save_path: str = None):
    """Create a visualization of the real CelebA dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Real CelebA Dataset ({dataset.attr1_name} vs {dataset.attr2_name})', fontsize=16)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            image, label, metadata = dataset[indices[idx]]
            
            # Convert tensor to numpy and denormalize
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            
            attr1_val = "1" if metadata['attr1'] else "0"
            attr2_val = "1" if metadata['attr2'] else "0"
            bias_str = "✓" if metadata['bias_followed'] else "✗"
            
            title = f"{dataset.attr1_name}={attr1_val}, {dataset.attr2_name}={attr2_val}\nBias: {bias_str}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def save_real_celeba_dataset(
    train_dataset: BiasedRealCelebADataset,
    test_dataset: BiasedRealCelebADataset,
    metadata: Dict,
    output_dir: str
):
    """Save real CelebA dataset to files with hierarchical structure"""
    output_path = Path(output_dir)
    
    # Create hierarchical directory structure
    attr1_name = metadata['attr1_name']
    attr2_name = metadata['attr2_name']
    bias_str = f"trainbias_{metadata['train_bias']:.2f}_testbias_{metadata['test_bias']:.2f}"
    subdir = f"real_celeba_{attr1_name}_{attr2_name}_{bias_str}"
    output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    torch.save(train_dataset, output_path / "train_dataset.pt")
    torch.save(test_dataset, output_path / "test_dataset.pt")
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualizations
    visualize_real_celeba(train_dataset, 16, str(output_path / "train_samples.png"))
    visualize_real_celeba(test_dataset, 16, str(output_path / "test_samples.png"))
    
    print(f"\nReal CelebA dataset saved to {output_path}")
    
    # Save data summary
    summary = {
        'dataset_type': 'real_celeba_biased',
        'attributes': [attr1_name, attr2_name],
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'bias_configuration': {
            'train_bias': metadata['train_bias'],
            'test_bias': metadata['test_bias']
        },
        'path': str(output_path)
    }
    
    with open(output_path / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_real_celeba_dataset(data_dir: str) -> Tuple[BiasedRealCelebADataset, BiasedRealCelebADataset, Dict]:
    """Load real CelebA dataset from files"""
    data_path = Path(data_dir)
    
    # Load datasets
    train_dataset = torch.load(data_path / "train_dataset.pt", weights_only=False)
    test_dataset = torch.load(data_path / "test_dataset.pt", weights_only=False)
    
    # Load metadata
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate biased real CelebA dataset for bias research")
    parser.add_argument("--attr1", type=str, default="Male",
                       choices=["Male", "Young", "Smiling", "Blond_Hair", "Brown_Hair", "Black_Hair", 
                               "Eyeglasses", "Heavy_Makeup", "Attractive", "High_Cheekbones"],
                       help="First attribute for bias (target to predict)")
    parser.add_argument("--attr2", type=str, default="Blond_Hair",
                       choices=["Blond_Hair", "Brown_Hair", "Black_Hair", "Smiling", "Eyeglasses", 
                               "Heavy_Makeup", "Young", "Attractive", "High_Cheekbones", "Wearing_Lipstick"],
                       help="Second attribute for bias (spurious correlation)")
    parser.add_argument("--train_bias", type=float, default=0.8,
                       help="Bias strength in training set")
    parser.add_argument("--test_bias", type=float, default=0.2,
                       help="Bias strength in test set") 
    parser.add_argument("--train_size", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Number of test samples")
    parser.add_argument("--image_size", type=int, default=64,
                       help="Image size (will be resized to this)")
    parser.add_argument("--data_root", type=str, default="./celeba_data",
                       help="Root directory for CelebA data download")
    parser.add_argument("--output_dir", type=str, default="./real_celeba_bias_data",
                       help="Output directory for biased dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate attribute combination
    if args.attr1 == args.attr2:
        raise ValueError("attr1 and attr2 must be different")
    
    print("Generating Biased Real CelebA Dataset")
    print("=" * 45)
    print(f"Target attribute (attr1): {args.attr1}")
    print(f"Biasing attribute (attr2): {args.attr2}")
    print(f"Train bias: {args.train_bias}")
    print(f"Test bias: {args.test_bias}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Image size: {args.image_size}")
    print(f"Seed: {args.seed}")
    print(f"CelebA data root: {args.data_root}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_biased_real_celeba_datasets(
        data_root=args.data_root,
        attr1_name=args.attr1,
        attr2_name=args.attr2,
        train_bias=args.train_bias,
        test_bias=args.test_bias,
        train_size=args.train_size,
        test_size=args.test_size,
        image_size=args.image_size,
        seed=args.seed
    )
    
    # Analyze bias
    train_analysis = analyze_dataset_bias(train_dataset, "Training")
    test_analysis = analyze_dataset_bias(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_real_celeba_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()