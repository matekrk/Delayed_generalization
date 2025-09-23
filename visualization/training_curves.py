#!/usr/bin/env python3
"""
Training Curve Visualization Module

Centralized plotting functions for training curves that were previously
scattered across individual training scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Any, Tuple
import json


class TrainingCurvePlotter:
    """Centralized plotter for training curves across all phenomena"""
    
    def __init__(self, save_dir: str, dpi: int = 150):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_basic_training_curves(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float],
        title: str = "Training Curves",
        save_name: str = "training_curves.png"
    ) -> plt.Figure:
        """Plot basic training and test curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train', color='blue')
        ax1.plot(epochs, test_losses, label='Test', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, label='Train', color='blue')
        ax2.plot(epochs, test_accuracies, label='Test', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Log scale loss
        ax3.semilogy(epochs, train_losses, label='Train', color='blue')
        ax3.semilogy(epochs, test_losses, label='Test', color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (log scale)')
        ax3.set_title('Loss (Log Scale)')
        ax3.legend()
        ax3.grid(True)
        
        # Test accuracy zoom (useful for grokking detection)
        ax4.plot(epochs, test_accuracies, color='red', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Test Accuracy (Grokking Detection)')
        ax4.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_grokking_curves(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float],
        save_name: str = "grokking_curves.png"
    ) -> plt.Figure:
        """Plot training curves with grokking-specific analysis"""
        fig = self.plot_basic_training_curves(
            epochs, train_losses, test_losses, train_accuracies, test_accuracies,
            title="Grokking Training Curves", save_name=save_name
        )
        
        # Add grokking detection annotations
        ax4 = fig.axes[3]  # Test accuracy plot
        
        # Detect potential grokking point (rapid accuracy increase)
        test_acc = np.array(test_accuracies)
        if len(test_acc) > 10:
            # Look for rapid improvement
            diff = np.diff(test_acc)
            smooth_diff = np.convolve(diff, np.ones(5)/5, mode='valid')
            if len(smooth_diff) > 0:
                max_improvement_idx = np.argmax(smooth_diff) + 5
                if max_improvement_idx < len(epochs) and smooth_diff[max_improvement_idx-5] > 0.1:
                    ax4.axvline(x=epochs[max_improvement_idx], color='green', 
                              linestyle='--', alpha=0.7, linewidth=2, label='Potential Grokking')
                    ax4.legend()
        
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_robustness_curves(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float],
        corruption_accuracies: Dict[str, List[float]],
        save_name: str = "robustness_curves.png"
    ) -> plt.Figure:
        """Plot training curves with robustness analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Basic curves
        axes[0, 0].plot(epochs, train_losses, label='Train', color='blue')
        axes[0, 0].plot(epochs, test_losses, label='Test', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, train_accuracies, label='Train', color='blue')
        axes[0, 1].plot(epochs, test_accuracies, label='Test', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Corruption robustness
        axes[0, 2].plot(epochs, test_accuracies, label='Clean Test', color='green', linewidth=2)
        colors = ['orange', 'purple', 'brown', 'pink', 'gray']
        for i, (corruption, accs) in enumerate(corruption_accuracies.items()):
            color = colors[i % len(colors)]
            axes[0, 2].plot(epochs, accs, label=corruption, color=color, alpha=0.7)
        axes[0, 2].set_title('Robustness Analysis')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Robustness gap analysis
        clean_acc = np.array(test_accuracies)
        for i, (corruption, accs) in enumerate(corruption_accuracies.items()):
            gap = clean_acc - np.array(accs)
            color = colors[i % len(colors)]
            axes[1, 0].plot(epochs, gap, label=f'{corruption} gap', color=color)
        axes[1, 0].set_title('Robustness Gap')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Clean - Corrupted Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Average corruption performance
        if corruption_accuracies:
            avg_corruption = np.mean([accs for accs in corruption_accuracies.values()], axis=0)
            axes[1, 1].plot(epochs, test_accuracies, label='Clean', color='green', linewidth=2)
            axes[1, 1].plot(epochs, avg_corruption, label='Avg Corruption', color='red', linewidth=2)
            axes[1, 1].set_title('Clean vs Average Corruption')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Final corruption analysis
        if len(epochs) > 0 and corruption_accuracies:
            final_clean = test_accuracies[-1]
            final_corruptions = [accs[-1] for accs in corruption_accuracies.values()]
            corruption_names = list(corruption_accuracies.keys())
            
            axes[1, 2].bar(corruption_names, final_corruptions, alpha=0.7)
            axes[1, 2].axhline(y=final_clean, color='green', linestyle='--', 
                              label=f'Clean: {final_clean:.2f}%')
            axes[1, 2].set_title('Final Corruption Performance')
            axes[1, 2].set_ylabel('Final Accuracy')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = "training_metrics.json"):
        """Save training metrics to JSON file"""
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_metrics[key] = [v.tolist() for v in value]
                else:
                    serializable_metrics[key] = value
            
            json.dump(serializable_metrics, f, indent=2)