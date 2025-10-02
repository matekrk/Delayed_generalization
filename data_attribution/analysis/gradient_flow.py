#!/usr/bin/env python3
"""
Gradient Flow Analyzer for Delayed Generalization Research

Analyzes gradient dynamics including:
- Gradient magnitudes over time
- Gradient direction consistency
- Layer-wise gradient statistics
- Gradient signal-to-noise ratio
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class GradientFlowAnalyzer:
    """
    Analyze gradient flow patterns during training.
    
    Helps identify when gradients become more stable/consistent,
    indicating transitions from memorization to generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        track_layers: Optional[List[str]] = None
    ):
        """
        Initialize gradient flow analyzer.
        
        Args:
            model: PyTorch model to analyze
            track_layers: Specific layers to track (None = track all)
        """
        self.model = model
        self.track_layers = track_layers
        
        # Storage for gradient statistics
        self.gradient_history = defaultdict(lambda: {
            'magnitudes': [],
            'norms': [],
            'variances': [],
            'directions': []
        })
        
        self.epoch_stats = []
        self.previous_gradients = {}
    
    def track_batch_gradients(
        self,
        loss: torch.Tensor,
        epoch: int,
        batch_idx: int
    ):
        """
        Track gradients for a training batch.
        
        Args:
            loss: Loss tensor (should have grad_fn)
            epoch: Current epoch
            batch_idx: Batch index
        """
        # Compute gradients
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # Track gradients for each layer
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            if self.track_layers and name not in self.track_layers:
                continue
            
            grad = param.grad.detach()
            
            # Compute statistics
            grad_norm = torch.norm(grad).item()
            grad_magnitude = torch.abs(grad).mean().item()
            grad_variance = torch.var(grad).item()
            
            self.gradient_history[name]['norms'].append(grad_norm)
            self.gradient_history[name]['magnitudes'].append(grad_magnitude)
            self.gradient_history[name]['variances'].append(grad_variance)
            
            # Track direction consistency
            if name in self.previous_gradients:
                prev_grad = self.previous_gradients[name]
                cosine_sim = torch.nn.functional.cosine_similarity(
                    grad.flatten(), prev_grad.flatten(), dim=0
                ).item()
                self.gradient_history[name]['directions'].append(cosine_sim)
            
            self.previous_gradients[name] = grad.clone()
    
    def compute_epoch_statistics(self, epoch: int):
        """
        Compute aggregate gradient statistics for an epoch.
        
        Args:
            epoch: Epoch number
        """
        epoch_stat = {
            'epoch': epoch,
            'layer_stats': {}
        }
        
        for name, history in self.gradient_history.items():
            if not history['norms']:
                continue
            
            # Get recent gradients (last 100 batches or all if less)
            recent_norms = history['norms'][-100:]
            recent_mags = history['magnitudes'][-100:]
            recent_dirs = history['directions'][-100:] if history['directions'] else []
            
            layer_stat = {
                'mean_norm': np.mean(recent_norms),
                'std_norm': np.std(recent_norms),
                'mean_magnitude': np.mean(recent_mags),
                'gradient_stability': np.mean(recent_dirs) if recent_dirs else 0.0,
                'direction_variance': np.var(recent_dirs) if recent_dirs else 0.0
            }
            
            epoch_stat['layer_stats'][name] = layer_stat
        
        # Compute overall statistics
        all_norms = [s['mean_norm'] for s in epoch_stat['layer_stats'].values()]
        all_stabilities = [s['gradient_stability'] for s in epoch_stat['layer_stats'].values() 
                          if s['gradient_stability'] != 0.0]
        
        epoch_stat['overall'] = {
            'mean_gradient_norm': np.mean(all_norms) if all_norms else 0.0,
            'mean_stability': np.mean(all_stabilities) if all_stabilities else 0.0
        }
        
        self.epoch_stats.append(epoch_stat)
    
    def analyze_gradient_flow(self) -> Dict[str, Any]:
        """
        Analyze gradient flow patterns across training.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.epoch_stats:
            return {}
        
        analysis = {
            'epochs': [s['epoch'] for s in self.epoch_stats],
            'overall_norms': [s['overall']['mean_gradient_norm'] for s in self.epoch_stats],
            'overall_stability': [s['overall']['mean_stability'] for s in self.epoch_stats],
            'layer_analyses': {}
        }
        
        # Per-layer analysis
        for name in self.gradient_history.keys():
            layer_norms = []
            layer_stabilities = []
            
            for stat in self.epoch_stats:
                if name in stat['layer_stats']:
                    layer_norms.append(stat['layer_stats'][name]['mean_norm'])
                    layer_stabilities.append(stat['layer_stats'][name]['gradient_stability'])
            
            if layer_norms:
                analysis['layer_analyses'][name] = {
                    'norm_trend': np.polyfit(range(len(layer_norms)), layer_norms, 1)[0] 
                                 if len(layer_norms) > 1 else 0.0,
                    'stability_trend': np.polyfit(range(len(layer_stabilities)), layer_stabilities, 1)[0]
                                      if len(layer_stabilities) > 1 else 0.0,
                    'final_norm': layer_norms[-1],
                    'final_stability': layer_stabilities[-1]
                }
        
        return analysis
    
    def plot_gradient_flow(self, save_dir: Optional[str] = None) -> plt.Figure:
        """
        Visualize gradient flow patterns.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Matplotlib figure
        """
        if not self.epoch_stats:
            raise ValueError("No epoch statistics available")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = [s['epoch'] for s in self.epoch_stats]
        
        # 1. Overall gradient norm
        ax = axes[0, 0]
        overall_norms = [s['overall']['mean_gradient_norm'] for s in self.epoch_stats]
        ax.plot(epochs, overall_norms, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Gradient Norm')
        ax.set_title('Overall Gradient Magnitude')
        ax.grid(True, alpha=0.3)
        
        # 2. Overall stability
        ax = axes[0, 1]
        overall_stability = [s['overall']['mean_stability'] for s in self.epoch_stats]
        ax.plot(epochs, overall_stability, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Direction Consistency')
        ax.set_title('Gradient Direction Stability')
        ax.grid(True, alpha=0.3)
        
        # 3. Layer-wise gradient norms
        ax = axes[1, 0]
        for name in list(self.gradient_history.keys())[:5]:  # Plot up to 5 layers
            layer_norms = []
            for stat in self.epoch_stats:
                if name in stat['layer_stats']:
                    layer_norms.append(stat['layer_stats'][name]['mean_norm'])
            if layer_norms:
                ax.plot(epochs[:len(layer_norms)], layer_norms, label=name, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Layer-wise Gradient Norms')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Layer-wise stability
        ax = axes[1, 1]
        for name in list(self.gradient_history.keys())[:5]:
            layer_stabilities = []
            for stat in self.epoch_stats:
                if name in stat['layer_stats']:
                    layer_stabilities.append(stat['layer_stats'][name]['gradient_stability'])
            if layer_stabilities:
                ax.plot(epochs[:len(layer_stabilities)], layer_stabilities, label=name, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Direction Consistency')
        ax.set_title('Layer-wise Gradient Stability')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / 'gradient_flow.png', 
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def save(self, filepath: str):
        """Save analyzer state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'gradient_history': dict(self.gradient_history),
                'epoch_stats': self.epoch_stats,
                'track_layers': self.track_layers
            }, f)
    
    @classmethod
    def load(cls, filepath: str, model: nn.Module) -> 'GradientFlowAnalyzer':
        """Load analyzer state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        analyzer = cls(model=model, track_layers=data['track_layers'])
        analyzer.gradient_history = defaultdict(lambda: {
            'magnitudes': [], 'norms': [], 'variances': [], 'directions': []
        })
        analyzer.gradient_history.update(data['gradient_history'])
        analyzer.epoch_stats = data['epoch_stats']
        
        return analyzer
