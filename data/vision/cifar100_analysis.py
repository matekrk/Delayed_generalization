#!/usr/bin/env python3
"""
CIFAR-100 Dataset Generator with Predominant Color Analysis

This module extends CIFAR-100 dataset with color analysis capabilities 
for detecting simplicity bias patterns. 
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import json

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


class CIFAR100WithColorAnalysis(torchvision.datasets.CIFAR100):
    """
    Extended CIFAR-100 dataset with color analysis capabilities.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: bool = False,
        color_analysis: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CIFAR-100 with color analysis.
        
        Args:
            root: Root directory of dataset
            train: Whether to load training or test set
            transform: Image transformations
            target_transform: Target transformations
            download: Whether to download dataset
            color_analysis: Whether to perform color analysis
            cache_dir: Directory to cache color analysis results
        """
        super().__init__(root, train, transform, target_transform, download)
        
        self.color_analysis = color_analysis
        self.cache_dir = Path(cache_dir) if cache_dir else Path(root) / "color_analysis"
        
        # CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        # Superclass information for CIFAR-100
        self.superclass_names = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 
            'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
            'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]
        
        self.class_to_superclass = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10,
            7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14,
            17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15,
            6, 0, 17, 8, 14, 13
        ]
        
        if self.color_analysis and PredominantColorAnalyzer is not None:
            self.color_analyzer = PredominantColorAnalyzer(
                cache_dir=str(self.cache_dir),
                region_analysis=True
            )
            self._perform_color_analysis()
        elif self.color_analysis:
            print("Warning: Color analysis requested but PredominantColorAnalyzer not available")
    
    def _perform_color_analysis(self):
        """Perform color analysis on the dataset."""
        if hasattr(self, '_color_analysis_results'):
            return
        
        print(f"Performing color analysis on CIFAR-100 {'train' if self.train else 'test'} set...")
        
        # Create temporary dataset without transforms for color analysis
        temp_dataset = torchvision.datasets.CIFAR100(
            self.root, self.train, transform=transforms.ToTensor(), download=False
        )
        
        # Perform analysis
        self._color_analysis_results = self.color_analyzer.analyze_dataset(
            temp_dataset,
            class_names=self.class_names,
            sample_size=min(1000, len(temp_dataset)),  # Analyze subset for efficiency
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
        analysis = self.get_color_analysis()
        
        report = {
            'dataset': 'CIFAR-100',
            'split': 'train' if self.train else 'test',
            'potential_biases': [],
            'class_analysis': {},
            'recommendations': []
        }
        
        # Analyze outdoor vs indoor classes
        outdoor_classes = ['forest', 'mountain', 'plain', 'sea', 'cloud', 'road']
        sky_related_classes = ['cloud', 'plain', 'mountain', 'sea']
        
        class_summaries = analysis.get('class_summaries', {})
        
        # Check for sky correlation in outdoor scenes
        for class_name in sky_related_classes:
            if class_name in class_summaries and 'top' in class_summaries[class_name]:
                top_region = class_summaries[class_name]['top']
                colors = np.array(top_region['representative_colors'])
                
                # Check for blue dominance (sky bias)
                if len(colors) > 0:
                    blue_dominance = np.mean(colors[:, 2] > colors[:, 0]) > 0.7  # Blue > Red
                    if blue_dominance:
                        report['potential_biases'].append({
                            'type': 'sky_correlation',
                            'class': class_name,
                            'description': f'Strong blue dominance in top region of {class_name} images',
                            'severity': 'high'
                        })
        
        # Check for color separability between similar classes
        similar_class_groups = [
            ['apple', 'orange', 'pear'],  # Fruits
            ['rose', 'tulip', 'poppy', 'sunflower'],  # Flowers
            ['tiger', 'lion', 'leopard'],  # Big cats
            ['oak_tree', 'pine_tree', 'willow_tree', 'maple_tree', 'palm_tree']  # Trees
        ]
        
        for group in similar_class_groups:
            group_colors = {}
            for class_name in group:
                if class_name in class_summaries and 'full' in class_summaries[class_name]:
                    group_colors[class_name] = class_summaries[class_name]['full']['representative_colors']
            
            if len(group_colors) > 1:
                # Calculate color distances between classes in group
                separabilities = []
                class_pairs = []
                for i, (class1, colors1) in enumerate(group_colors.items()):
                    for j, (class2, colors2) in enumerate(list(group_colors.items())[i+1:], i+1):
                        if colors1 and colors2:
                            min_dist = float('inf')
                            for c1 in colors1:
                                for c2 in colors2:
                                    dist = np.linalg.norm(np.array(c1) - np.array(c2))
                                    min_dist = min(min_dist, dist)
                            separabilities.append(min_dist)
                            class_pairs.append((class1, class2))
                
                if separabilities and np.mean(separabilities) > 60:
                    report['potential_biases'].append({
                        'type': 'color_separability',
                        'classes': group,
                        'description': f'High color separability between similar classes: {group}',
                        'severity': 'medium',
                        'mean_separability': np.mean(separabilities)
                    })
        
        # Generate recommendations
        if len(report['potential_biases']) > 0:
            report['recommendations'] = [
                "Consider using color augmentations to reduce color bias",
                "Monitor model attention using GradCAM to verify focus on object features",
                "Use background-invariant training techniques",
                "Evaluate model performance on color-modified versions of test images"
            ]
        
        return report


def analyze_cifar100_colors(
    root: str = "./data",
    cache_dir: str = "./cifar100_color_analysis",
    download: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze CIFAR-100 colors.
    
    Args:
        root: Dataset root directory
        cache_dir: Cache directory for results
        download: Whether to download dataset
        
    Returns:
        Combined analysis results for train and test sets
    """
    results = {
        'train': {},
        'test': {},
        'combined_analysis': {}
    }
    
    # Analyze train set
    print("Analyzing CIFAR-100 training set...")
    train_dataset = CIFAR100WithColorAnalysis(
        root=root,
        train=True,
        download=download,
        cache_dir=cache_dir
    )
    results['train'] = {
        'color_analysis': train_dataset.get_color_analysis(),
        'bias_report': train_dataset.get_simplicity_bias_report()
    }
    
    # Analyze test set
    print("Analyzing CIFAR-100 test set...")
    test_dataset = CIFAR100WithColorAnalysis(
        root=root,
        train=False,
        download=False,
        cache_dir=cache_dir
    )
    results['test'] = {
        'color_analysis': test_dataset.get_color_analysis(),
        'bias_report': test_dataset.get_simplicity_bias_report()
    }
    
    # Combined analysis
    train_biases = results['train']['bias_report']['potential_biases']
    test_biases = results['test']['bias_report']['potential_biases']
    
    results['combined_analysis'] = {
        'consistent_biases': _find_consistent_biases(train_biases, test_biases),
        'split_specific_biases': _find_split_specific_biases(train_biases, test_biases),
        'overall_recommendations': _generate_overall_recommendations(train_biases, test_biases)
    }
    
    # Save results
    output_path = Path(cache_dir) / "cifar100_full_analysis.json"
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = _convert_for_json(results)
        json.dump(json_results, f, indent=2)
    
    print(f"Full analysis saved to {output_path}")
    
    return results


def _find_consistent_biases(train_biases: List[Dict], test_biases: List[Dict]) -> List[Dict]:
    """Find biases that appear in both train and test sets."""
    consistent = []
    
    for train_bias in train_biases:
        for test_bias in test_biases:
            if (train_bias['type'] == test_bias['type'] and
                train_bias.get('class') == test_bias.get('class')):
                consistent.append({
                    'type': train_bias['type'],
                    'class': train_bias.get('class'),
                    'description': f"Consistent across train/test: {train_bias['description']}",
                    'severity': 'high'
                })
    
    return consistent


def _find_split_specific_biases(train_biases: List[Dict], test_biases: List[Dict]) -> Dict[str, List[Dict]]:
    """Find biases specific to train or test sets."""
    train_only = []
    test_only = []
    
    # Find train-only biases
    for train_bias in train_biases:
        found_in_test = any(
            train_bias['type'] == test_bias['type'] and
            train_bias.get('class') == test_bias.get('class')
            for test_bias in test_biases
        )
        if not found_in_test:
            train_only.append(train_bias)
    
    # Find test-only biases
    for test_bias in test_biases:
        found_in_train = any(
            test_bias['type'] == train_bias['type'] and
            test_bias.get('class') == train_bias.get('class')
            for train_bias in train_biases
        )
        if not found_in_train:
            test_only.append(test_bias)
    
    return {
        'train_only': train_only,
        'test_only': test_only
    }


def _generate_overall_recommendations(train_biases: List[Dict], test_biases: List[Dict]) -> List[str]:
    """Generate overall recommendations based on bias analysis."""
    recommendations = []
    
    total_biases = len(train_biases) + len(test_biases)
    
    if total_biases > 5:
        recommendations.append("High number of potential biases detected. Consider comprehensive bias mitigation.")
    
    if any(bias['type'] == 'sky_correlation' for bias in train_biases + test_biases):
        recommendations.append("Sky correlation detected. Use GradCAM to verify model focuses on objects, not sky.")
    
    if any(bias['severity'] == 'high' for bias in train_biases + test_biases):
        recommendations.append("High-severity biases detected. Implement color augmentation and bias-aware training.")
    
    recommendations.extend([
        "Regularly monitor model attention patterns during training",
        "Consider using domain adaptation techniques if deploying to different environments",
        "Validate model robustness using color-modified test images"
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
    results = analyze_cifar100_colors(
        root="./data/cifar100",
        cache_dir="./cifar100_color_analysis"
    )
    
    print("\n=== CIFAR-100 Color Analysis Summary ===")
    print(f"Train biases found: {len(results['train']['bias_report']['potential_biases'])}")
    print(f"Test biases found: {len(results['test']['bias_report']['potential_biases'])}")
    print(f"Consistent biases: {len(results['combined_analysis']['consistent_biases'])}")