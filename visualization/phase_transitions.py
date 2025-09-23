#!/usr/bin/env python3
"""
Phase Transition Visualization Module

Specialized plotting functions for phase transition phenomena.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Any, Tuple


class PhaseTransitionPlotter:
    """Specialized plotter for phase transition analysis"""
    
    def __init__(self, save_dir: str, dpi: int = 150):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_emergence_curves(
        self,
        epochs: List[int],
        capability_metrics: Dict[str, List[float]],
        threshold: float = 0.5,
        save_name: str = "emergence_curves.png"
    ) -> plt.Figure:
        """Plot emergence of different capabilities"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Individual capability curves
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (capability, values) in enumerate(capability_metrics.items()):
            color = colors[i % len(colors)]
            ax1.plot(epochs, values, label=capability, color=color, linewidth=2)
            
            # Mark emergence point (first time crossing threshold)
            values_array = np.array(values)
            emergence_idx = np.where(values_array >= threshold)[0]
            if len(emergence_idx) > 0:
                emergence_epoch = epochs[emergence_idx[0]]
                ax1.axvline(x=emergence_epoch, color=color, linestyle='--', alpha=0.5)
                ax1.scatter([emergence_epoch], [threshold], color=color, s=100, zorder=5)
        
        ax1.axhline(y=threshold, color='black', linestyle='-', alpha=0.3, label='Threshold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Capability Score')
        ax1.set_title('Capability Emergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Emergence timing analysis
        ax2 = axes[0, 1]
        emergence_times = []
        capability_names = []
        
        for capability, values in capability_metrics.items():
            values_array = np.array(values)
            emergence_idx = np.where(values_array >= threshold)[0]
            if len(emergence_idx) > 0:
                emergence_times.append(epochs[emergence_idx[0]])
                capability_names.append(capability)
        
        if emergence_times:
            ax2.bar(capability_names, emergence_times, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Capability')
            ax2.set_ylabel('Emergence Epoch')
            ax2.set_title('Emergence Timing')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Rate of change analysis
        ax3 = axes[1, 0]
        for i, (capability, values) in enumerate(capability_metrics.items()):
            if len(values) > 1:
                color = colors[i % len(colors)]
                rate_of_change = np.diff(values)
                # Smooth the rate of change
                if len(rate_of_change) > 5:
                    window_size = min(5, len(rate_of_change))
                    rate_of_change = np.convolve(rate_of_change, np.ones(window_size)/window_size, mode='valid')
                    epochs_subset = epochs[window_size//2:len(epochs)-window_size//2+1][:len(rate_of_change)]
                else:
                    epochs_subset = epochs[1:len(rate_of_change)+1]
                    
                ax3.plot(epochs_subset, rate_of_change, label=capability, color=color, alpha=0.7)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Rate of Change')
        ax3.set_title('Capability Development Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Phase transition detection
        ax4 = axes[1, 1]
        
        # Aggregate capability score
        if capability_metrics:
            aggregate_scores = []
            for epoch_idx in range(len(epochs)):
                epoch_scores = [values[epoch_idx] for values in capability_metrics.values() 
                              if epoch_idx < len(values)]
                if epoch_scores:
                    aggregate_scores.append(np.mean(epoch_scores))
                else:
                    aggregate_scores.append(0)
            
            ax4.plot(epochs, aggregate_scores, color='darkblue', linewidth=3, label='Aggregate Capability')
            
            # Detect phase transitions using second derivative
            if len(aggregate_scores) > 5:
                first_deriv = np.diff(aggregate_scores)
                second_deriv = np.diff(first_deriv)
                
                # Find peaks in second derivative (acceleration points)
                if len(second_deriv) > 3:
                    # Simple peak detection
                    peaks = []
                    for i in range(1, len(second_deriv)-1):
                        if (second_deriv[i] > second_deriv[i-1] and 
                            second_deriv[i] > second_deriv[i+1] and 
                            second_deriv[i] > 0.01):  # Threshold for significance
                            peaks.append(i + 2)  # Adjust for derivative offset
                    
                    for peak in peaks:
                        if peak < len(epochs):
                            ax4.axvline(x=epochs[peak], color='red', linestyle=':', 
                                      alpha=0.7, linewidth=2, label='Phase Transition' if peak == peaks[0] else '')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Aggregate Capability')
            ax4.set_title('Phase Transition Detection')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_scaling_analysis(
        self,
        model_sizes: List[int],
        final_capabilities: Dict[str, List[float]],
        save_name: str = "scaling_analysis.png"
    ) -> plt.Figure:
        """Plot how capabilities scale with model size"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (capability, values) in enumerate(final_capabilities.items()):
            color = colors[i % len(colors)]
            ax1.plot(model_sizes, values, 'o-', label=capability, color=color, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Model Size (Parameters)')
        ax1.set_ylabel('Final Capability Score')
        ax1.set_title('Capability vs Model Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale for model size
        for i, (capability, values) in enumerate(final_capabilities.items()):
            color = colors[i % len(colors)]
            ax2.semilogx(model_sizes, values, 'o-', label=capability, color=color, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Model Size (Parameters, log scale)')
        ax2.set_ylabel('Final Capability Score')
        ax2.set_title('Capability vs Model Size (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        return fig