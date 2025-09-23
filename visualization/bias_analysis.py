#!/usr/bin/env python3
"""
Bias Analysis Visualization Module

Centralized plotting functions for bias analysis that were previously
scattered across simplicity bias training scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Any, Tuple


class BiasAnalysisPlotter:
    """Centralized plotter for bias analysis across all phenomena"""
    
    def __init__(self, save_dir: str, dpi: int = 150):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_simplicity_bias_curves(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float], 
        train_accuracies: List[float],
        test_accuracies: List[float],
        bias_accuracies: List[float],
        bias_type: str = "spurious",
        save_name: str = "bias_analysis.png"
    ) -> plt.Figure:
        """Plot bias analysis curves for simplicity bias experiments"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train', linewidth=2)
        ax1.plot(epochs, test_losses, label='Test', linewidth=2)
        ax1.set_title('Loss Curves', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall accuracy curves
        ax2.plot(epochs, train_accuracies, label='Train', linewidth=2)
        ax2.plot(epochs, test_accuracies, label='Test', linewidth=2)
        ax2.set_title('Overall Accuracy Curves', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bias analysis
        ax3.plot(epochs, bias_accuracies, label=f'{bias_type.title()} Correlation', 
                color='red', linewidth=2)
        ax3.set_title(f'Spurious Correlation Learning ({bias_type})', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Bias Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Test vs Bias accuracy comparison
        ax4.plot(epochs, test_accuracies, label='Test Accuracy', linewidth=2)
        ax4.plot(epochs, bias_accuracies, label=f'{bias_type.title()} Accuracy', 
                color='red', linewidth=2)
        ax4.set_title('True vs Spurious Performance', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_color_shape_bias(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float], 
        test_accuracies: List[float],
        color_accuracies: List[float],
        shape_accuracies: List[float],
        save_name: str = "color_shape_bias.png"
    ) -> plt.Figure:
        """Plot color vs shape bias analysis for colored MNIST"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train', color='blue')
        ax1.plot(epochs, test_losses, label='Test', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Overall accuracy curves
        ax2.plot(epochs, train_accuracies, label='Train', color='blue')
        ax2.plot(epochs, test_accuracies, label='Test', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Bias analysis: Color vs Shape
        ax3.plot(epochs, color_accuracies, label='Color Bias', color='orange', linewidth=2)
        ax3.plot(epochs, shape_accuracies, label='Shape Bias', color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Color vs Shape Bias Analysis')
        ax3.legend()
        ax3.grid(True)
        
        # Combined view
        ax4.plot(epochs, test_accuracies, label='Overall Test', color='red')
        ax4.plot(epochs, color_accuracies, label='Color Bias', color='orange')
        ax4.plot(epochs, shape_accuracies, label='Shape Bias', color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Simplicity Bias Analysis')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_celeba_bias_analysis(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float],
        bias_conforming_accuracies: List[float],
        bias_conflicting_accuracies: List[float],
        save_name: str = "celeba_bias_analysis.png"
    ) -> plt.Figure:
        """Plot CelebA bias analysis with conforming vs conflicting"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(epochs, test_losses, label='Test', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Overall accuracy curves
        axes[0, 1].plot(epochs, train_accuracies, label='Train', linewidth=2)
        axes[0, 1].plot(epochs, test_accuracies, label='Test', linewidth=2)
        axes[0, 1].set_title('Overall Accuracy Curves', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bias analysis
        axes[0, 2].plot(epochs, bias_conforming_accuracies, label='Bias Conforming', 
                       color='green', linewidth=2)
        axes[0, 2].plot(epochs, bias_conflicting_accuracies, label='Bias Conflicting', 
                       color='red', linewidth=2)
        axes[0, 2].set_title('Bias Conforming vs Conflicting', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bias gap (difference between conforming and conflicting)
        bias_gap = np.array(bias_conforming_accuracies) - np.array(bias_conflicting_accuracies)
        axes[1, 0].plot(epochs, bias_gap, color='purple', linewidth=2)
        axes[1, 0].set_title('Bias Gap (Conforming - Conflicting)', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Difference (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Three-way comparison
        axes[1, 1].plot(epochs, test_accuracies, label='Overall Test', color='blue', linewidth=2)
        axes[1, 1].plot(epochs, bias_conforming_accuracies, label='Bias Conforming', 
                       color='green', linewidth=2)
        axes[1, 1].plot(epochs, bias_conflicting_accuracies, label='Bias Conflicting', 
                       color='red', linewidth=2)
        axes[1, 1].set_title('Comprehensive Bias Analysis', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Bias learning dynamics (rate of change)
        if len(epochs) > 1:
            bias_conf_rate = np.diff(bias_conforming_accuracies)
            bias_conf_rate = np.concatenate([[0], bias_conf_rate])  # Pad with 0 for first epoch
            
            bias_conf_rate = np.diff(bias_conflicting_accuracies)
            bias_conf_rate = np.concatenate([[0], bias_conf_rate])
            
            axes[1, 2].plot(epochs, bias_conf_rate, label='Conforming Rate', color='green', alpha=0.7)
            axes[1, 2].plot(epochs, bias_conf_rate, label='Conflicting Rate', color='red', alpha=0.7)
            axes[1, 2].set_title('Bias Learning Rate', fontsize=14)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy Change (%/epoch)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_bias_summary_statistics(
        self,
        bias_metrics: Dict[str, float],
        save_name: str = "bias_summary.png"
    ) -> plt.Figure:
        """Plot summary statistics for bias analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bias strength metrics
        bias_strength_metrics = {k: v for k, v in bias_metrics.items() 
                               if 'gap' in k.lower() or 'bias' in k.lower()}
        
        if bias_strength_metrics:
            ax1.bar(bias_strength_metrics.keys(), bias_strength_metrics.values(), 
                   color='skyblue', alpha=0.7)
            ax1.set_title('Bias Strength Metrics')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        performance_metrics = {k: v for k, v in bias_metrics.items() 
                             if 'acc' in k.lower() and 'gap' not in k.lower()}
        
        if performance_metrics:
            ax2.bar(performance_metrics.keys(), performance_metrics.values(), 
                   color='lightcoral', alpha=0.7)
            ax2.set_title('Performance Metrics')
            ax2.set_ylabel('Accuracy (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig