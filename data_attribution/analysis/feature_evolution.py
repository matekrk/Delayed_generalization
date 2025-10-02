#!/usr/bin/env python3
"""
Feature Evolution Tracker for Delayed Generalization Research

Tracks how learned representations evolve over training, identifying:
- Representation quality changes
- Feature space organization
- Clustering patterns
- Representation stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path
import pickle


class FeatureEvolutionTracker:
    """
    Track and analyze how learned representations evolve during training.
    
    Useful for understanding when models transition from memorization to
    generalization by analyzing representation quality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: torch.device = None,
        max_samples_store: int = 1000
    ):
        """
        Initialize feature evolution tracker.
        
        Args:
            model: PyTorch model to analyze
            layer_names: Names of layers to track (e.g., ['layer3', 'layer4'])
            device: Device to run computations on
            max_samples_store: Maximum samples to store features for
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_samples_store = max_samples_store
        
        # Storage for features over time
        self.feature_history = defaultdict(lambda: defaultdict(list))  # layer -> epoch -> features
        self.label_history = defaultdict(list)  # epoch -> labels
        
        # Hooks for extracting features
        self.hooks = []
        self.features = {}
        self._register_hooks()
        
        # Analysis results
        self.analysis_history = []
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name in self.layer_names:
            # Navigate to the layer by name
            layer = self.model
            for part in name.split('.'):
                layer = getattr(layer, part)
            
            handle = layer.register_forward_hook(get_hook(name))
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def track_epoch(
        self,
        dataloader,
        epoch: int,
        sample_limit: Optional[int] = None
    ):
        """
        Extract and store features for an epoch.
        
        Args:
            dataloader: DataLoader for the dataset
            epoch: Current epoch number
            sample_limit: Maximum number of samples to process (optional)
        """
        self.model.eval()
        
        epoch_features = {name: [] for name in self.layer_names}
        epoch_labels = []
        
        samples_processed = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                if sample_limit and samples_processed >= sample_limit:
                    break
                
                inputs = inputs.to(self.device)
                
                # Forward pass to trigger hooks
                _ = self.model(inputs)
                
                # Collect features from hooks
                for name in self.layer_names:
                    features = self.features[name]
                    # Flatten spatial dimensions if present
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                    epoch_features[name].append(features.cpu())
                
                epoch_labels.append(targets)
                samples_processed += inputs.size(0)
                
                if samples_processed >= self.max_samples_store:
                    break
        
        # Concatenate and store
        for name in self.layer_names:
            if epoch_features[name]:
                features_tensor = torch.cat(epoch_features[name], dim=0)
                self.feature_history[name][epoch] = features_tensor
        
        if epoch_labels:
            labels_tensor = torch.cat(epoch_labels, dim=0)
            self.label_history[epoch] = labels_tensor
    
    def compute_feature_statistics(self, epoch: int, layer_name: str) -> Dict[str, float]:
        """
        Compute statistics about features at a specific epoch.
        
        Args:
            epoch: Epoch number
            layer_name: Name of the layer
            
        Returns:
            Dictionary of statistics
        """
        if epoch not in self.feature_history[layer_name]:
            return {}
        
        features = self.feature_history[layer_name][epoch].numpy()
        labels = self.label_history[epoch].numpy()
        
        # Basic statistics
        feature_norms = np.linalg.norm(features, axis=1)
        
        # Dimensionality (effective rank)
        _, s, _ = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
        effective_rank = np.sum(s) ** 2 / np.sum(s ** 2)
        
        # Clustering quality (silhouette score)
        if len(np.unique(labels)) > 1 and features.shape[0] > len(np.unique(labels)):
            try:
                silhouette = silhouette_score(features, labels, metric='euclidean')
            except:
                silhouette = 0.0
        else:
            silhouette = 0.0
        
        # Feature variance
        feature_variance = np.var(features, axis=0).mean()
        
        # Inter-class distance vs intra-class distance
        class_centers = []
        class_variances = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_features = features[labels == label]
            if len(class_features) > 0:
                center = class_features.mean(axis=0)
                class_centers.append(center)
                class_variances.append(np.var(class_features))
        
        if len(class_centers) > 1:
            class_centers = np.array(class_centers)
            inter_class_dist = np.mean([
                np.linalg.norm(class_centers[i] - class_centers[j])
                for i in range(len(class_centers))
                for j in range(i + 1, len(class_centers))
            ])
            intra_class_var = np.mean(class_variances)
            separability = inter_class_dist / (intra_class_var + 1e-8)
        else:
            separability = 0.0
        
        return {
            'mean_norm': float(np.mean(feature_norms)),
            'std_norm': float(np.std(feature_norms)),
            'effective_rank': float(effective_rank),
            'silhouette_score': float(silhouette),
            'feature_variance': float(feature_variance),
            'separability': float(separability)
        }
    
    def analyze_evolution(self) -> Dict[str, Any]:
        """
        Analyze how features evolved across all tracked epochs.
        
        Returns:
            Dictionary containing evolution metrics and trends
        """
        analysis = {}
        
        for layer_name in self.layer_names:
            epochs = sorted(self.feature_history[layer_name].keys())
            if not epochs:
                continue
            
            layer_analysis = {
                'epochs': epochs,
                'statistics': []
            }
            
            for epoch in epochs:
                stats = self.compute_feature_statistics(epoch, layer_name)
                layer_analysis['statistics'].append(stats)
            
            # Compute trends
            if len(epochs) > 1:
                metrics = ['mean_norm', 'effective_rank', 'silhouette_score', 'separability']
                trends = {}
                
                for metric in metrics:
                    values = [s[metric] for s in layer_analysis['statistics'] if metric in s]
                    if len(values) > 1:
                        # Compute linear trend
                        x = np.arange(len(values))
                        y = np.array(values)
                        trend = np.polyfit(x, y, 1)[0]  # Slope
                        trends[f'{metric}_trend'] = float(trend)
                
                layer_analysis['trends'] = trends
            
            analysis[layer_name] = layer_analysis
        
        self.analysis_history.append(analysis)
        return analysis
    
    def detect_phase_transitions(self, layer_name: str, metric: str = 'silhouette_score') -> List[int]:
        """
        Detect potential phase transitions based on feature evolution.
        
        Args:
            layer_name: Name of layer to analyze
            metric: Metric to use for detection ('silhouette_score', 'separability', etc.)
            
        Returns:
            List of epochs where transitions were detected
        """
        epochs = sorted(self.feature_history[layer_name].keys())
        if len(epochs) < 3:
            return []
        
        values = []
        for epoch in epochs:
            stats = self.compute_feature_statistics(epoch, layer_name)
            if metric in stats:
                values.append(stats[metric])
        
        if len(values) < 3:
            return []
        
        values = np.array(values)
        
        # Compute rate of change
        rate_of_change = np.diff(values)
        
        # Find points where rate of change changes significantly
        transitions = []
        threshold = np.std(rate_of_change) * 1.5
        
        for i in range(1, len(rate_of_change) - 1):
            # Look for acceleration (change in rate of change)
            acceleration = rate_of_change[i] - rate_of_change[i-1]
            if abs(acceleration) > threshold:
                transitions.append(epochs[i])
        
        return transitions
    
    def visualize_evolution(
        self,
        layer_name: str,
        save_dir: Optional[str] = None,
        method: str = 'pca'
    ) -> plt.Figure:
        """
        Visualize feature evolution using dimensionality reduction.
        
        Args:
            layer_name: Name of layer to visualize
            save_dir: Directory to save plots
            method: Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            Matplotlib figure
        """
        epochs = sorted(self.feature_history[layer_name].keys())
        if not epochs:
            raise ValueError(f"No features tracked for layer {layer_name}")
        
        # Select a subset of epochs for visualization
        if len(epochs) > 6:
            epoch_indices = np.linspace(0, len(epochs) - 1, 6, dtype=int)
            epochs_to_plot = [epochs[i] for i in epoch_indices]
        else:
            epochs_to_plot = epochs
        
        n_epochs = len(epochs_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, epoch in enumerate(epochs_to_plot):
            if idx >= len(axes):
                break
            
            features = self.feature_history[layer_name][epoch].numpy()
            labels = self.label_history[epoch].numpy()
            
            # Dimensionality reduction
            if method == 'pca':
                reducer = PCA(n_components=2)
                features_2d = reducer.fit_transform(features)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                features_2d = reducer.fit_transform(features)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Plot
            ax = axes[idx]
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.6, s=10)
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax)
        
        # Hide unused subplots
        for idx in range(n_epochs, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Feature Space Evolution - {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / f'feature_evolution_{layer_name}.png',
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_evolution(
        self,
        layer_name: str,
        save_dir: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how feature quality metrics evolve over training.
        
        Args:
            layer_name: Name of layer to analyze
            save_dir: Directory to save plots
            
        Returns:
            Matplotlib figure
        """
        epochs = sorted(self.feature_history[layer_name].keys())
        if not epochs:
            raise ValueError(f"No features tracked for layer {layer_name}")
        
        # Compute statistics for all epochs
        stats_history = []
        for epoch in epochs:
            stats = self.compute_feature_statistics(epoch, layer_name)
            stats['epoch'] = epoch
            stats_history.append(stats)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Feature norms
        ax = axes[0, 0]
        ax.plot(epochs, [s['mean_norm'] for s in stats_history], 'b-', linewidth=2)
        ax.fill_between(
            epochs,
            [s['mean_norm'] - s['std_norm'] for s in stats_history],
            [s['mean_norm'] + s['std_norm'] for s in stats_history],
            alpha=0.3
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Norm')
        ax.set_title('Feature Magnitude Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Effective rank
        ax = axes[0, 1]
        ax.plot(epochs, [s['effective_rank'] for s in stats_history], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Feature Dimensionality')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Silhouette score
        ax = axes[1, 0]
        ax.plot(epochs, [s['silhouette_score'] for s in stats_history], 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Class separability
        ax = axes[1, 1]
        ax.plot(epochs, [s['separability'] for s in stats_history], 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Separability Ratio')
        ax.set_title('Class Separability')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Feature Quality Metrics - {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / f'metrics_evolution_{layer_name}.png',
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def save(self, filepath: str):
        """Save tracker state to file."""
        # Remove hooks before saving
        self.remove_hooks()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_history': dict(self.feature_history),
                'label_history': dict(self.label_history),
                'analysis_history': self.analysis_history,
                'layer_names': self.layer_names,
                'max_samples_store': self.max_samples_store
            }, f)
    
    @classmethod
    def load(cls, filepath: str, model: nn.Module, device: torch.device = None) -> 'FeatureEvolutionTracker':
        """Load tracker state from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tracker = cls(
            model=model,
            layer_names=data['layer_names'],
            device=device,
            max_samples_store=data['max_samples_store']
        )
        
        tracker.feature_history = defaultdict(lambda: defaultdict(list))
        tracker.feature_history.update(data['feature_history'])
        tracker.label_history = defaultdict(list)
        tracker.label_history.update(data['label_history'])
        tracker.analysis_history = data['analysis_history']
        
        return tracker
