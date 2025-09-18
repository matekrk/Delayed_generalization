#!/usr/bin/env python3
"""
Image Analysis Utilities for Delayed Generalization Research

This module provides utilities for analyzing images in datasets,
including predominant color analysis for detecting simplicity bias
(e.g., sky correlation in classification tasks).
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pickle
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

# Optional import for OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Some color space conversions will be limited.")

class PredominantColorAnalyzer:
    """
    Analyzes predominant colors in images to detect simplicity bias patterns.
    
    This is particularly useful for detecting when models might be learning
    based on background colors (e.g., sky correlation) rather than actual
    object features.
    """
    
    def __init__(
        self,
        n_colors: int = 5,
        color_space: str = 'RGB',
        cache_dir: Optional[str] = None,
        region_analysis: bool = True
    ):
        """
        Initialize the predominant color analyzer.
        
        Args:
            n_colors: Number of dominant colors to extract
            color_space: Color space for analysis ('RGB', 'HSV', 'LAB')
            cache_dir: Directory to cache results (optional)
            region_analysis: Whether to analyze different image regions separately
        """
        self.n_colors = n_colors
        self.color_space = color_space
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.region_analysis = region_analysis
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Color space conversion functions
        if CV2_AVAILABLE:
            self.color_converters = {
                'HSV': cv2.COLOR_RGB2HSV,
                'LAB': cv2.COLOR_RGB2LAB,
                'RGB': None
            }
        else:
            self.color_converters = {
                'RGB': None,
                'HSV': None,
                'LAB': None
            }
            if color_space != 'RGB':
                warnings.warn(f"OpenCV not available. Color space {color_space} not supported, using RGB.")
        
    def _convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """Convert image to specified color space."""
        if self.color_space == 'RGB':
            return image
        elif self.color_space in self.color_converters and CV2_AVAILABLE:
            if self.color_converters[self.color_space] is not None:
                return cv2.cvtColor(image, self.color_converters[self.color_space])
            else:
                return image
        else:
            warnings.warn(f"Color space {self.color_space} not available. Using RGB.")
            return image
    
    def _extract_image_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract different regions of the image for separate analysis.
        
        Returns:
            Dictionary with region names as keys and image patches as values
        """
        h, w = image.shape[:2]
        regions = {}
        
        # Whole image
        regions['full'] = image
        
        if self.region_analysis:
            # Top region (potentially sky)
            regions['top'] = image[:h//3, :]
            
            # Middle region (main objects)
            regions['middle'] = image[h//3:2*h//3, :]
            
            # Bottom region (ground/water)
            regions['bottom'] = image[2*h//3:, :]
            
            # Center region (main object focus)
            center_h, center_w = h//4, w//4
            regions['center'] = image[center_h:3*center_h, center_w:3*center_w]
            
            # Edges (background)
            edge_size = min(h, w) // 10
            regions['edges'] = np.concatenate([
                image[:edge_size, :].flatten(),
                image[-edge_size:, :].flatten(),
                image[:, :edge_size].flatten(),
                image[:, -edge_size:].flatten()
            ]).reshape(-1, 3)
        
        return regions
    
    def analyze_image(self, image: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze predominant colors in a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) or torch tensor
            
        Returns:
            Dictionary containing color analysis results
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Ensure values are in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert color space
        image_converted = self._convert_color_space(image)
        
        # Extract regions
        regions = self._extract_image_regions(image_converted)
        
        results = {}
        
        for region_name, region_pixels in regions.items():
            if region_name == 'edges':
                pixels = region_pixels
            else:
                pixels = region_pixels.reshape(-1, 3)
            
            # Remove any NaN or invalid pixels
            valid_mask = np.all(np.isfinite(pixels), axis=1)
            pixels = pixels[valid_mask]
            
            if len(pixels) == 0:
                continue
                
            # Apply K-means clustering to find dominant colors
            try:
                kmeans = KMeans(n_clusters=min(self.n_colors, len(pixels)), 
                              random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster centers (dominant colors) and their frequencies
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_
                
                # Calculate color frequencies
                unique_labels, counts = np.unique(labels, return_counts=True)
                frequencies = counts / len(labels)
                
                # Sort by frequency
                sorted_indices = np.argsort(frequencies)[::-1]
                
                region_results = {
                    'dominant_colors': colors[sorted_indices].tolist(),
                    'frequencies': frequencies[sorted_indices].tolist(),
                    'total_pixels': len(pixels)
                }
                
                # Add color statistics
                region_results['color_stats'] = {
                    'mean_color': np.mean(pixels, axis=0).tolist(),
                    'std_color': np.std(pixels, axis=0).tolist(),
                    'brightness': np.mean(np.sum(pixels, axis=1)),
                    'contrast': np.std(np.sum(pixels, axis=1))
                }
                
                results[region_name] = region_results
                
            except Exception as e:
                warnings.warn(f"Could not analyze region {region_name}: {e}")
                continue
        
        return results
    
    def analyze_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        labels: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze predominant colors across an entire dataset.
        
        Args:
            dataset: PyTorch dataset
            labels: Labels for each sample (if not provided, tries to get from dataset)
            class_names: Names for each class
            sample_size: Number of samples to analyze (None for all)
            save_results: Whether to save results to cache
            
        Returns:
            Dictionary containing dataset-wide color analysis
        """
        # Generate cache key
        cache_key = f"color_analysis_{hash(str(dataset))}_{self.color_space}_{self.n_colors}"
        cache_file = self.cache_dir / f"{cache_key}.pkl" if self.cache_dir else None
        
        # Try to load from cache
        if cache_file and cache_file.exists():
            print(f"Loading cached color analysis from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Analyzing dataset colors...")
        
        # Determine sample indices
        total_samples = len(dataset)
        if sample_size is None:
            sample_size = total_samples
        else:
            sample_size = min(sample_size, total_samples)
        
        indices = np.random.choice(total_samples, sample_size, replace=False)
        
        # Initialize results storage
        class_colors = defaultdict(lambda: defaultdict(list))
        all_results = []
        
        # Analyze each sample
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"Processed {i}/{sample_size} images")
            
            try:
                # Get sample
                if hasattr(dataset, '__getitem__'):
                    sample = dataset[idx]
                    if isinstance(sample, tuple):
                        image = sample[0]
                        label = sample[1] if len(sample) > 1 else None
                    else:
                        image = sample
                        label = labels[idx] if labels else None
                else:
                    continue
                
                # Analyze image
                analysis = self.analyze_image(image)
                analysis['sample_idx'] = idx
                analysis['label'] = label
                
                all_results.append(analysis)
                
                # Aggregate by class
                if label is not None:
                    for region_name, region_data in analysis.items():
                        if isinstance(region_data, dict) and 'dominant_colors' in region_data:
                            class_colors[label][region_name].append(region_data)
                            
            except Exception as e:
                warnings.warn(f"Could not analyze sample {idx}: {e}")
                continue
        
        # Compute aggregate statistics
        results = {
            'individual_analyses': all_results,
            'class_summaries': {},
            'dataset_summary': {},
            'color_correlations': {},
            'metadata': {
                'n_colors': self.n_colors,
                'color_space': self.color_space,
                'sample_size': sample_size,
                'total_samples': total_samples,
                'class_names': class_names
            }
        }
        
        # Analyze class-wise patterns
        for class_label, regions_data in class_colors.items():
            class_name = class_names[class_label] if class_names and class_label < len(class_names) else f"class_{class_label}"
            results['class_summaries'][class_name] = {}
            
            for region_name, region_analyses in regions_data.items():
                if not region_analyses:
                    continue
                    
                # Aggregate dominant colors across samples
                all_colors = []
                all_frequencies = []
                
                for analysis in region_analyses:
                    colors = np.array(analysis['dominant_colors'])
                    freqs = np.array(analysis['frequencies'])
                    
                    # Weight colors by their frequency within the image
                    for color, freq in zip(colors, freqs):
                        all_colors.append(color)
                        all_frequencies.append(freq)
                
                if all_colors:
                    # Cluster similar colors across the class
                    all_colors = np.array(all_colors)
                    all_frequencies = np.array(all_frequencies)
                    
                    # Find representative colors for this class-region combination
                    n_clusters = min(self.n_colors, len(all_colors))
                    if n_clusters > 0:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(all_colors)
                        
                        # Compute weighted cluster frequencies
                        cluster_freqs = []
                        for cluster_id in range(n_clusters):
                            mask = cluster_labels == cluster_id
                            cluster_freq = np.sum(all_frequencies[mask])
                            cluster_freqs.append(cluster_freq)
                        
                        cluster_freqs = np.array(cluster_freqs)
                        cluster_freqs = cluster_freqs / np.sum(cluster_freqs)
                        
                        # Sort by frequency
                        sorted_indices = np.argsort(cluster_freqs)[::-1]
                        
                        results['class_summaries'][class_name][region_name] = {
                            'representative_colors': kmeans.cluster_centers_[sorted_indices].tolist(),
                            'frequencies': cluster_freqs[sorted_indices].tolist(),
                            'n_samples': len(region_analyses)
                        }
        
        # Detect potential simplicity bias patterns
        results['bias_analysis'] = self._detect_simplicity_bias(results['class_summaries'])
        
        # Save results if requested
        if save_results and cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved color analysis to {cache_file}")
        
        return results
    
    def _detect_simplicity_bias(self, class_summaries: Dict) -> Dict[str, Any]:
        """
        Detect potential simplicity bias patterns based on color analysis.
        
        Returns:
            Dictionary containing bias analysis results
        """
        bias_analysis = {
            'sky_correlation': {},
            'background_correlation': {},
            'color_separability': {},
            'warnings': []
        }
        
        # Define sky-like colors (blue tones in different color spaces)
        if self.color_space == 'RGB':
            sky_color_ranges = [(100, 150, 200), (135, 206, 235)]  # Light blue ranges
        elif self.color_space == 'HSV':
            sky_color_ranges = [(100, 100, 100), (120, 255, 255)]  # Blue hue ranges
        else:
            sky_color_ranges = []
        
        # Analyze each region for potential bias
        regions_to_check = ['top', 'full', 'edges']
        
        for region in regions_to_check:
            region_colors_by_class = {}
            
            # Collect colors for each class in this region
            for class_name, class_data in class_summaries.items():
                if region in class_data:
                    region_colors_by_class[class_name] = class_data[region]['representative_colors']
            
            if len(region_colors_by_class) < 2:
                continue
            
            # Check for color separability between classes
            separability_scores = []
            class_pairs = []
            
            class_names = list(region_colors_by_class.keys())
            for i in range(len(class_names)):
                for j in range(i + 1, len(class_names)):
                    class1, class2 = class_names[i], class_names[j]
                    colors1 = np.array(region_colors_by_class[class1])
                    colors2 = np.array(region_colors_by_class[class2])
                    
                    # Compute minimum distance between color sets
                    min_distance = float('inf')
                    for c1 in colors1:
                        for c2 in colors2:
                            distance = np.linalg.norm(c1 - c2)
                            min_distance = min(min_distance, distance)
                    
                    separability_scores.append(min_distance)
                    class_pairs.append((class1, class2))
            
            if separability_scores:
                bias_analysis['color_separability'][region] = {
                    'mean_separability': np.mean(separability_scores),
                    'min_separability': np.min(separability_scores),
                    'max_separability': np.max(separability_scores),
                    'class_pairs': class_pairs
                }
                
                # Flag potential bias if colors are very separable
                if np.mean(separability_scores) > 50:  # Threshold for "too separable"
                    bias_analysis['warnings'].append(
                        f"High color separability in {region} region may indicate simplicity bias"
                    )
        
        return bias_analysis
    
    def visualize_results(
        self,
        analysis_results: Dict,
        save_path: Optional[str] = None,
        max_classes: int = 10
    ) -> None:
        """
        Create visualizations of the color analysis results.
        
        Args:
            analysis_results: Results from analyze_dataset
            save_path: Path to save the visualization
            max_classes: Maximum number of classes to visualize
        """
        class_summaries = analysis_results['class_summaries']
        class_names = list(class_summaries.keys())[:max_classes]
        
        if not class_names:
            print("No class data to visualize")
            return
        
        # Create figure with subplots
        n_regions = len(set().union(*[class_data.keys() for class_data in class_summaries.values()]))
        fig, axes = plt.subplots(n_regions, 1, figsize=(12, 3 * n_regions))
        if n_regions == 1:
            axes = [axes]
        
        region_names = []
        for class_data in class_summaries.values():
            region_names.extend(class_data.keys())
        region_names = list(set(region_names))
        
        for i, region in enumerate(region_names):
            ax = axes[i]
            
            # Create color palette for each class
            y_pos = 0
            for class_name in class_names:
                if region in class_summaries[class_name]:
                    colors = class_summaries[class_name][region]['representative_colors']
                    freqs = class_summaries[class_name][region]['frequencies']
                    
                    # Draw color bars
                    x_start = 0
                    for color, freq in zip(colors, freqs):
                        width = freq * 10  # Scale for visibility
                        
                        # Normalize color to [0, 1] range
                        if self.color_space == 'RGB':
                            normalized_color = np.array(color) / 255.0
                        else:
                            # Convert back to RGB for display
                            if CV2_AVAILABLE and self.color_space == 'HSV':
                                color_rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0, 0]
                                normalized_color = color_rgb / 255.0
                            elif CV2_AVAILABLE and self.color_space == 'LAB':
                                color_rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0, 0]
                                normalized_color = color_rgb / 255.0
                            else:
                                # Fallback to direct normalization
                                normalized_color = np.array(color) / 255.0
                        
                        ax.barh(y_pos, width, left=x_start, height=0.8, 
                               color=normalized_color, edgecolor='black', linewidth=0.5)
                        x_start += width
                    
                    ax.text(-0.5, y_pos, class_name, ha='right', va='center')
                    y_pos += 1
            
            ax.set_title(f'Dominant Colors by Class - {region.title()} Region')
            ax.set_xlabel('Relative Frequency')
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.5, len(class_names) - 0.5)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


def compute_effective_lr(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    """
    Compute the effective learning rate for AdamW optimizer.
    
    For AdamW, the effective learning rate depends on the bias correction terms
    and the accumulated gradient moments.
    
    Args:
        optimizer: The AdamW optimizer instance
        epoch: Current epoch (used for bias correction)
        
    Returns:
        Effective learning rate
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        # For non-AdamW optimizers, return the base learning rate
        return optimizer.param_groups[0]['lr']
    
    # Get first parameter group (assuming all groups have similar behavior)
    param_group = optimizer.param_groups[0]
    
    # Get optimizer hyperparameters
    lr = param_group['lr']
    beta1, beta2 = param_group['betas']
    eps = param_group['eps']
    
    # Get optimizer state for first parameter (representative)
    states = []
    for param in param_group['params']:
        if param.grad is not None and param in optimizer.state:
            states.append(optimizer.state[param])
    
    if not states:
        return lr  # No state available, return base learning rate
    
    # Compute effective learning rate based on bias correction and gradient moments
    effective_lrs = []
    
    for state in states:
        step = state.get('step', 0)
        if step == 0:
            continue
            
        # Bias correction terms
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Get gradient moments
        exp_avg = state.get('exp_avg', None)
        exp_avg_sq = state.get('exp_avg_sq', None)
        
        if exp_avg is None or exp_avg_sq is None:
            continue
        
        # Compute bias-corrected moments
        bias_corrected_exp_avg = exp_avg / bias_correction1
        bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
        
        # Effective learning rate per parameter
        denom = bias_corrected_exp_avg_sq.sqrt().add_(eps)
        effective_lr_per_param = lr / denom.mean().item()
        
        effective_lrs.append(effective_lr_per_param)
    
    if effective_lrs:
        return np.mean(effective_lrs)
    else:
        return lr


# Convenience function for quick color analysis
def analyze_image_colors(
    dataset: torch.utils.data.Dataset,
    cache_dir: str = "./color_analysis_cache",
    sample_size: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to analyze predominant colors in a dataset.
    
    Args:
        dataset: PyTorch dataset to analyze
        cache_dir: Directory to cache results
        sample_size: Number of samples to analyze
        **kwargs: Additional arguments for PredominantColorAnalyzer
        
    Returns:
        Color analysis results
    """
    analyzer = PredominantColorAnalyzer(cache_dir=cache_dir, **kwargs)
    return analyzer.analyze_dataset(dataset, sample_size=sample_size)