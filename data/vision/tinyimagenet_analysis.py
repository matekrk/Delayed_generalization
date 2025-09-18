#!/usr/bin/env python3
"""
TinyImageNet Dataset Generator with Predominant Color Analysis

This module provides support for TinyImageNet dataset with color analysis 
capabilities for detecting simplicity bias patterns.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import json
import requests
import zipfile
import os
from PIL import Image

# Try to import the image analysis utility
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_analysis import PredominantColorAnalyzer
except ImportError:
    # Fallback if import fails
    print("Warning: Could not import PredominantColorAnalyzer. Color analysis will be disabled.")
    PredominantColorAnalyzer = None


class TinyImageNetDataset(Dataset):
    """
    TinyImageNet dataset with color analysis capabilities.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
        color_analysis: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize TinyImageNet dataset.
        
        Args:
            root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            download: Whether to download dataset
            color_analysis: Whether to perform color analysis
            cache_dir: Directory to cache color analysis results
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.color_analysis = color_analysis
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "color_analysis"
        
        # Download and setup dataset
        if download:
            self._download_dataset()
        
        self._setup_dataset()
        
        if self.color_analysis and PredominantColorAnalyzer is not None:
            self.color_analyzer = PredominantColorAnalyzer(
                cache_dir=str(self.cache_dir),
                region_analysis=True
            )
            self._perform_color_analysis()
        elif self.color_analysis:
            print("Warning: Color analysis requested but PredominantColorAnalyzer not available")
    
    def _download_dataset(self):
        """Download TinyImageNet dataset."""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        dataset_dir = self.root / "tiny-imagenet-200"
        
        if dataset_dir.exists():
            print("TinyImageNet already downloaded.")
            return
        
        print("Downloading TinyImageNet dataset...")
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download zip file
        zip_path = self.root / "tiny-imagenet-200.zip"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Clean up zip file
        zip_path.unlink()
        print("TinyImageNet download completed.")
    
    def _setup_dataset(self):
        """Setup dataset structure and load metadata."""
        dataset_dir = self.root / "tiny-imagenet-200"
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"TinyImageNet not found at {dataset_dir}. Set download=True.")
        
        # Load class names
        wnids_file = dataset_dir / "wnids.txt"
        with open(wnids_file, 'r') as f:
            self.class_ids = [line.strip() for line in f.readlines()]
        
        # Load class names mapping
        words_file = dataset_dir / "words.txt"
        self.id_to_name = {}
        if words_file.exists():
            with open(words_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.id_to_name[parts[0]] = parts[1]
        
        self.class_names = [self.id_to_name.get(class_id, class_id) for class_id in self.class_ids]
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        
        # Setup file paths based on split
        self.samples = []
        
        if self.split == 'train':
            train_dir = dataset_dir / "train"
            for class_id in self.class_ids:
                class_dir = train_dir / class_id / "images"
                if class_dir.exists():
                    for img_file in class_dir.glob("*.JPEG"):
                        self.samples.append((str(img_file), self.class_to_idx[class_id]))
        
        elif self.split == 'val':
            val_dir = dataset_dir / "val"
            val_annotations = val_dir / "val_annotations.txt"
            
            if val_annotations.exists():
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_id = parts[1]
                            if class_id in self.class_to_idx:
                                img_path = val_dir / "images" / img_name
                                if img_path.exists():
                                    self.samples.append((str(img_path), self.class_to_idx[class_id]))
        
        elif self.split == 'test':
            test_dir = dataset_dir / "test" / "images"
            if test_dir.exists():
                for img_file in test_dir.glob("*.JPEG"):
                    # Test set has no labels
                    self.samples.append((str(img_file), -1))
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _perform_color_analysis(self):
        """Perform color analysis on the dataset."""
        if hasattr(self, '_color_analysis_results') or self.split == 'test':
            return
        
        print(f"Performing color analysis on TinyImageNet {self.split} set...")
        
        # Create temporary dataset without transforms for color analysis
        temp_dataset = TinyImageNetDataset(
            self.root,
            self.split,
            transform=transforms.ToTensor(),
            download=False,
            color_analysis=False
        )
        
        # Perform analysis on subset for efficiency
        sample_size = min(2000, len(temp_dataset))
        
        self._color_analysis_results = self.color_analyzer.analyze_dataset(
            temp_dataset,
            class_names=self.class_names,
            sample_size=sample_size,
            save_results=True
        )
        
        print("Color analysis completed.")
    
    def get_color_analysis(self) -> Dict[str, Any]:
        """Get color analysis results."""
        if not hasattr(self, '_color_analysis_results'):
            self._perform_color_analysis()
        return self._color_analysis_results
    
    def get_simplicity_bias_report(self) -> Dict[str, Any]:
        """
        Generate a report on potential simplicity bias patterns.
        
        Returns:
            Dictionary containing bias analysis
        """
        if self.split == 'test':
            return {'error': 'Cannot perform bias analysis on test set without labels'}
        
        analysis = self.get_color_analysis()
        
        report = {
            'dataset': 'TinyImageNet',
            'split': self.split,
            'potential_biases': [],
            'class_analysis': {},
            'recommendations': []
        }
        
        # Identify classes that might have strong color biases
        outdoor_keywords = ['beach', 'desert', 'mountain', 'forest', 'meadow', 'valley', 'cliff']
        sky_related_keywords = ['beach', 'desert', 'mountain', 'cliff', 'valley', 'lakeside']
        vehicle_keywords = ['car', 'truck', 'bus', 'taxi', 'police car', 'limousine']
        
        class_summaries = analysis.get('class_summaries', {})
        
        # Check for sky correlation in outdoor scenes
        for class_name, class_data in class_summaries.items():
            # Check if class is likely outdoor scene
            is_outdoor = any(keyword in class_name.lower() for keyword in outdoor_keywords)
            is_sky_related = any(keyword in class_name.lower() for keyword in sky_related_keywords)
            
            if is_sky_related and 'top' in class_data:
                top_region = class_data['top']
                colors = np.array(top_region['representative_colors'])
                
                # Check for blue dominance (sky bias)
                if len(colors) > 0:
                    # Calculate blue dominance
                    blue_scores = []
                    for color in colors:
                        # RGB values, check if blue is dominant
                        r, g, b = color[:3]
                        blue_dominance = b > max(r, g) and b > 100  # Blue is strongest and reasonably bright
                        blue_scores.append(blue_dominance)
                    
                    if np.mean(blue_scores) > 0.6:
                        report['potential_biases'].append({
                            'type': 'sky_correlation',
                            'class': class_name,
                            'description': f'Strong blue dominance in top region of {class_name} images',
                            'severity': 'high',
                            'blue_dominance_rate': np.mean(blue_scores)
                        })
            
            # Check for vehicle color patterns
            is_vehicle = any(keyword in class_name.lower() for keyword in vehicle_keywords)
            if is_vehicle and 'full' in class_data:
                full_region = class_data['full']
                colors = np.array(full_region['representative_colors'])
                
                if len(colors) > 0:
                    # Check for unnatural color concentration
                    color_variance = np.var(colors, axis=0)
                    if np.mean(color_variance) < 200:  # Low variance indicates color concentration
                        report['potential_biases'].append({
                            'type': 'color_concentration',
                            'class': class_name,
                            'description': f'Low color variance in {class_name} may indicate simplicity bias',
                            'severity': 'medium',
                            'color_variance': np.mean(color_variance)
                        })
        
        # Check for overall color separability between classes
        if len(class_summaries) > 1:
            separability_scores = []
            for i, (class1, data1) in enumerate(class_summaries.items()):
                for j, (class2, data2) in enumerate(list(class_summaries.items())[i+1:], i+1):
                    if 'full' in data1 and 'full' in data2:
                        colors1 = np.array(data1['full']['representative_colors'])
                        colors2 = np.array(data2['full']['representative_colors'])
                        
                        if len(colors1) > 0 and len(colors2) > 0:
                            # Calculate minimum color distance
                            min_dist = float('inf')
                            for c1 in colors1:
                                for c2 in colors2:
                                    dist = np.linalg.norm(c1 - c2)
                                    min_dist = min(min_dist, dist)
                            separability_scores.append(min_dist)
            
            if separability_scores:
                mean_separability = np.mean(separability_scores)
                if mean_separability > 80:  # High color separability
                    report['potential_biases'].append({
                        'type': 'high_color_separability',
                        'description': 'Classes are highly separable by color, indicating potential simplicity bias',
                        'severity': 'medium',
                        'mean_separability': mean_separability
                    })
        
        # Generate recommendations
        if len(report['potential_biases']) > 0:
            report['recommendations'] = [
                "Use strong color augmentations during training",
                "Monitor model attention using GradCAM throughout training",
                "Consider background replacement augmentation for outdoor scenes",
                "Evaluate model on grayscale versions of images",
                "Use adversarial training to reduce color dependency"
            ]
        else:
            report['recommendations'] = [
                "Continue monitoring for color bias during training",
                "Periodically validate model attention patterns"
            ]
        
        return report


def analyze_tinyimagenet_colors(
    root: str = "./data",
    cache_dir: str = "./tinyimagenet_color_analysis",
    download: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze TinyImageNet colors.
    
    Args:
        root: Dataset root directory
        cache_dir: Cache directory for results
        download: Whether to download dataset
        
    Returns:
        Combined analysis results for train and validation sets
    """
    results = {
        'train': {},
        'val': {},
        'combined_analysis': {}
    }
    
    # Analyze train set
    print("Analyzing TinyImageNet training set...")
    train_dataset = TinyImageNetDataset(
        root=root,
        split='train',
        download=download,
        cache_dir=cache_dir
    )
    results['train'] = {
        'color_analysis': train_dataset.get_color_analysis(),
        'bias_report': train_dataset.get_simplicity_bias_report()
    }
    
    # Analyze validation set
    print("Analyzing TinyImageNet validation set...")
    val_dataset = TinyImageNetDataset(
        root=root,
        split='val',
        download=False,
        cache_dir=cache_dir
    )
    results['val'] = {
        'color_analysis': val_dataset.get_color_analysis(),
        'bias_report': val_dataset.get_simplicity_bias_report()
    }
    
    # Combined analysis
    train_biases = results['train']['bias_report']['potential_biases']
    val_biases = results['val']['bias_report']['potential_biases']
    
    # Find consistent patterns
    consistent_biases = []
    for train_bias in train_biases:
        for val_bias in val_biases:
            if (train_bias['type'] == val_bias['type'] and
                train_bias.get('class') == val_bias.get('class')):
                consistent_biases.append({
                    'type': train_bias['type'],
                    'class': train_bias.get('class'),
                    'description': f"Consistent across train/val: {train_bias['description']}",
                    'severity': 'high'
                })
    
    results['combined_analysis'] = {
        'consistent_biases': consistent_biases,
        'total_train_biases': len(train_biases),
        'total_val_biases': len(val_biases),
        'overall_risk_level': _assess_risk_level(train_biases, val_biases),
        'recommendations': _generate_tinyimagenet_recommendations(train_biases, val_biases)
    }
    
    # Save results
    output_path = Path(cache_dir) / "tinyimagenet_full_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = _convert_for_json(results)
        json.dump(json_results, f, indent=2)
    
    print(f"Full analysis saved to {output_path}")
    
    return results


def _assess_risk_level(train_biases: List[Dict], val_biases: List[Dict]) -> str:
    """Assess overall risk level for simplicity bias."""
    total_biases = len(train_biases) + len(val_biases)
    high_severity = sum(1 for bias in train_biases + val_biases if bias.get('severity') == 'high')
    
    if high_severity > 3 or total_biases > 8:
        return 'high'
    elif high_severity > 1 or total_biases > 4:
        return 'medium'
    else:
        return 'low'


def _generate_tinyimagenet_recommendations(train_biases: List[Dict], val_biases: List[Dict]) -> List[str]:
    """Generate specific recommendations for TinyImageNet."""
    recommendations = []
    
    # Check for specific bias types
    has_sky_bias = any(bias['type'] == 'sky_correlation' for bias in train_biases + val_biases)
    has_color_concentration = any(bias['type'] == 'color_concentration' for bias in train_biases + val_biases)
    has_high_separability = any(bias['type'] == 'high_color_separability' for bias in train_biases + val_biases)
    
    if has_sky_bias:
        recommendations.extend([
            "Implement sky replacement augmentation for outdoor scene classes",
            "Use GradCAM to verify model focuses on objects rather than sky regions"
        ])
    
    if has_color_concentration:
        recommendations.extend([
            "Apply strong color jittering and saturation changes",
            "Consider grayscale training phases to reduce color dependency"
        ])
    
    if has_high_separability:
        recommendations.extend([
            "Use color-invariant preprocessing techniques",
            "Implement adversarial training with color-modified examples"
        ])
    
    # General recommendations
    recommendations.extend([
        "Monitor validation accuracy on color-modified images throughout training",
        "Use diverse data augmentation including color space transformations",
        "Consider ensemble methods to reduce individual model biases"
    ])
    
    return recommendations


def _convert_for_json(obj):
    """Convert numpy arrays and other non-JSON-serializable objects for JSON output."""
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


if __name__ == "__main__":
    # Example usage
    results = analyze_tinyimagenet_colors(
        root="./data/tinyimagenet",
        cache_dir="./tinyimagenet_color_analysis"
    )
    
    print("\n=== TinyImageNet Color Analysis Summary ===")
    print(f"Train biases found: {len(results['train']['bias_report']['potential_biases'])}")
    print(f"Validation biases found: {len(results['val']['bias_report']['potential_biases'])}")
    print(f"Consistent biases: {len(results['combined_analysis']['consistent_biases'])}")
    print(f"Overall risk level: {results['combined_analysis']['overall_risk_level']}")