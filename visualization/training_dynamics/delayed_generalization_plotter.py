#!/usr/bin/env python3
"""
Training Dynamics Visualization for Delayed Generalization

This module provides comprehensive visualization tools for training dynamics
in delayed generalization scenarios, including automatic phase transition
detection and phenomenon-specific analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DelayedGeneralizationPlotter:
    """
    Comprehensive plotter for delayed generalization training dynamics.
    
    Features:
    - Automatic phase transition detection
    - Phenomenon-specific visualizations
    - Statistical analysis of training patterns
    - Export capabilities for publications
    """
    
    def __init__(
        self,
        phenomenon_type: str = 'general',
        style: str = 'publication',
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Initialize the plotter.
        
        Args:
            phenomenon_type: Type of delayed generalization ('grokking', 'simplicity_bias', 'phase_transitions')
            style: Plotting style ('publication', 'presentation', 'interactive')
            figsize: Figure size for plots
        """
        self.phenomenon_type = phenomenon_type
        self.style = style
        self.figsize = figsize
        
        # Data storage
        self.training_data = {
            'epochs': [],
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'learning_rate': [],
            'weight_norms': [],
            'gradient_norms': []
        }
        
        # Phenomenon-specific data
        if phenomenon_type == 'grokking':
            self.training_data.update({
                'memorization_score': [],
                'generalization_score': [],
                'circuit_formation': []
            })
        elif phenomenon_type == 'simplicity_bias':
            self.training_data.update({
                'worst_group_acc': [],
                'bias_score': [],
                'group_accuracies': []
            })
        elif phenomenon_type == 'phase_transitions':
            self.training_data.update({
                'emergent_abilities': [],
                'capability_scores': [],
                'transition_sharpness': []
            })
        
        # Detected transitions
        self.phase_transitions = []
        
        # Style configuration
        self._configure_style()
    
    def _configure_style(self):
        """Configure plotting style based on use case."""
        
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'axes.linewidth': 1.2,
                'lines.linewidth': 2,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'font.family': 'serif'
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.linewidth': 2,
                'lines.linewidth': 3,
                'figure.dpi': 150,
                'savefig.dpi': 150,
                'font.family': 'sans-serif'
            })
    
    def add_epoch_data(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """
        Add training data for a single epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            test_loss: Test/validation loss
            train_acc: Training accuracy
            test_acc: Test/validation accuracy
            learning_rate: Current learning rate
            **kwargs: Additional metrics specific to phenomenon
        """
        
        self.training_data['epochs'].append(epoch)
        self.training_data['train_loss'].append(train_loss)
        self.training_data['test_loss'].append(test_loss)
        self.training_data['train_acc'].append(train_acc)
        self.training_data['test_acc'].append(test_acc)
        
        if learning_rate is not None:
            self.training_data['learning_rate'].append(learning_rate)
        
        # Add phenomenon-specific metrics
        for key, value in kwargs.items():
            if key in self.training_data:
                self.training_data[key].append(value)
    
    def detect_phase_transitions(
        self,
        metric: str = 'test_acc',
        threshold: float = 0.2,
        window_size: int = 100,
        min_improvement: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Automatically detect phase transitions in training.
        
        Args:
            metric: Metric to analyze for transitions ('test_acc', 'test_loss', etc.)
            threshold: Minimum relative improvement to detect transition
            window_size: Window size for comparing before/after performance
            min_improvement: Minimum absolute improvement required
            
        Returns:
            List of detected transitions with metadata
        """
        
        if len(self.training_data[metric]) < window_size * 2:
            return []
        
        data = np.array(self.training_data[metric])
        epochs = np.array(self.training_data['epochs'])
        
        transitions = []
        
        for i in range(window_size, len(data) - window_size):
            # Compare before and after windows
            before_window = data[i-window_size:i]
            after_window = data[i:i+window_size]
            
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            
            # For accuracy metrics, look for improvements
            if metric in ['test_acc', 'train_acc']:
                improvement = after_mean - before_mean
                relative_improvement = improvement / (before_mean + 1e-8)
            else:  # For loss metrics, look for decreases
                improvement = before_mean - after_mean
                relative_improvement = improvement / (before_mean + 1e-8)
            
            # Detect significant improvement
            if improvement > min_improvement and relative_improvement > threshold:
                
                # Compute transition sharpness
                sharpness = self._compute_transition_sharpness(data, i, window_size//2)
                
                transition = {
                    'epoch': epochs[i],
                    'epoch_idx': i,
                    'metric': metric,
                    'improvement': improvement,
                    'relative_improvement': relative_improvement,
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'sharpness': sharpness,
                    'window_size': window_size
                }
                
                transitions.append(transition)
        
        # Remove overlapping transitions (keep the strongest)
        transitions = self._remove_overlapping_transitions(transitions, window_size)
        
        self.phase_transitions = transitions
        return transitions
    
    def _compute_transition_sharpness(
        self,
        data: np.ndarray,
        transition_idx: int,
        window: int
    ) -> float:
        """Compute how sharp a transition is using sigmoid fitting."""
        
        start_idx = max(0, transition_idx - window)
        end_idx = min(len(data), transition_idx + window)
        
        x = np.arange(end_idx - start_idx)
        y = data[start_idx:end_idx]
        
        # Try to fit sigmoid
        try:
            def sigmoid(x, a, b, c, d):
                return a / (1 + np.exp(-b * (x - c))) + d
            
            # Initial parameter guess
            p0 = [np.max(y) - np.min(y), 1, len(x)/2, np.min(y)]
            
            popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=1000)
            
            # Sharpness is the steepness parameter
            sharpness = abs(popt[1])
            
        except:
            # Fallback: use linear slope
            slope, _, r_squared, _, _ = stats.linregress(x, y)
            sharpness = abs(slope) * r_squared
        
        return sharpness
    
    def _remove_overlapping_transitions(
        self,
        transitions: List[Dict],
        window_size: int
    ) -> List[Dict]:
        """Remove overlapping transitions, keeping the strongest."""
        
        if len(transitions) <= 1:
            return transitions
        
        # Sort by epoch
        transitions.sort(key=lambda x: x['epoch'])
        
        filtered = [transitions[0]]
        
        for transition in transitions[1:]:
            last_transition = filtered[-1]
            
            # Check if overlapping
            if transition['epoch'] - last_transition['epoch'] < window_size:
                # Keep the one with larger improvement
                if transition['improvement'] > last_transition['improvement']:
                    filtered[-1] = transition
            else:
                filtered.append(transition)
        
        return filtered
    
    def create_training_dynamics_plot(
        self,
        save_path: Optional[str] = None,
        show_transitions: bool = True,
        include_secondary_metrics: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive training dynamics plot.
        
        Args:
            save_path: Path to save the plot
            show_transitions: Whether to mark detected phase transitions
            include_secondary_metrics: Whether to include additional metrics
            
        Returns:
            Matplotlib figure
        """
        
        # Determine subplot layout
        if include_secondary_metrics:
            if self.phenomenon_type == 'grokking':
                fig, axes = plt.subplots(2, 3, figsize=self.figsize)
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = [axes]  # Make it consistent
        
        fig.suptitle(f'Training Dynamics - {self.phenomenon_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        epochs = self.training_data['epochs']
        
        # Plot 1: Loss curves
        ax = axes[0] if len(axes.shape) == 1 else axes[0, 0]
        ax.plot(epochs, self.training_data['train_loss'], 
               label='Training Loss', alpha=0.8, linewidth=2)
        ax.plot(epochs, self.training_data['test_loss'], 
               label='Test Loss', alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax = axes[1] if len(axes.shape) == 1 else axes[0, 1]
        ax.plot(epochs, self.training_data['train_acc'], 
               label='Training Accuracy', alpha=0.8, linewidth=2)
        ax.plot(epochs, self.training_data['test_acc'], 
               label='Test Accuracy', alpha=0.8, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if include_secondary_metrics and len(axes.shape) == 2:
            # Plot 3: Generalization gap
            ax = axes[1, 0]
            gen_gap = np.array(self.training_data['train_acc']) - np.array(self.training_data['test_acc'])
            ax.plot(epochs, gen_gap, color='red', alpha=0.8, linewidth=2, label='Train - Test Acc')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Generalization Gap')
            ax.set_title('Generalization Gap')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Phenomenon-specific
            ax = axes[1, 1]
            self._add_phenomenon_specific_plot(ax, epochs)
            
            # Additional plots for grokking
            if self.phenomenon_type == 'grokking' and axes.shape[1] > 2:
                # Learning rate
                ax = axes[0, 2]
                if self.training_data['learning_rate']:
                    ax.plot(epochs[:len(self.training_data['learning_rate'])], 
                           self.training_data['learning_rate'], 
                           color='purple', alpha=0.8, linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.set_title('Learning Rate Schedule')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                
                # Weight norms
                ax = axes[1, 2]
                if self.training_data['weight_norms']:
                    weight_norm_means = [np.mean(list(wn.values())) if isinstance(wn, dict) else wn 
                                       for wn in self.training_data['weight_norms']]
                    ax.plot(epochs[:len(weight_norm_means)], weight_norm_means, 
                           color='orange', alpha=0.8, linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Mean Weight Norm')
                    ax.set_title('Weight Norm Evolution')
                    ax.grid(True, alpha=0.3)
        
        # Mark phase transitions
        if show_transitions and self.phase_transitions:
            for transition in self.phase_transitions:
                for ax in axes.flat:
                    ax.axvline(x=transition['epoch'], color='red', linestyle=':', 
                             alpha=0.7, linewidth=2, label='Phase Transition' if ax == axes.flat[0] else '')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_phenomenon_specific_plot(self, ax: plt.Axes, epochs: List[int]):
        """Add phenomenon-specific plot to the figure."""
        
        if self.phenomenon_type == 'grokking':
            # Plot memorization vs generalization scores
            if self.training_data.get('memorization_score') and self.training_data.get('generalization_score'):
                ax.plot(epochs[:len(self.training_data['memorization_score'])], 
                       self.training_data['memorization_score'], 
                       label='Memorization Score', alpha=0.8, linewidth=2)
                ax.plot(epochs[:len(self.training_data['generalization_score'])], 
                       self.training_data['generalization_score'], 
                       label='Generalization Score', alpha=0.8, linewidth=2)
                ax.set_title('Memorization vs Generalization')
            else:
                # Fallback: show test accuracy with log scale if available
                ax.semilogy(epochs, np.maximum(1e-6, 1 - np.array(self.training_data['test_acc'])), 
                           label='Test Error', alpha=0.8, linewidth=2)
                ax.set_title('Test Error (Log Scale)')
            
        elif self.phenomenon_type == 'simplicity_bias':
            # Plot worst group accuracy
            if self.training_data.get('worst_group_acc'):
                ax.plot(epochs[:len(self.training_data['worst_group_acc'])], 
                       self.training_data['worst_group_acc'], 
                       color='orange', alpha=0.8, linewidth=2, label='Worst Group Acc')
                ax.set_title('Worst Group Performance')
            else:
                # Fallback: show bias score if available
                if self.training_data.get('bias_score'):
                    ax.plot(epochs[:len(self.training_data['bias_score'])], 
                           self.training_data['bias_score'], 
                           color='red', alpha=0.8, linewidth=2, label='Bias Score')
                    ax.set_title('Bias Score Evolution')
                else:
                    ax.text(0.5, 0.5, 'Worst Group Accuracy\nwould appear here', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title('Group Performance Analysis')
            
        elif self.phenomenon_type == 'phase_transitions':
            # Plot capability scores or emergent abilities
            if self.training_data.get('capability_scores'):
                cap_scores = self.training_data['capability_scores']
                if cap_scores and isinstance(cap_scores[0], dict):
                    # Multiple capabilities
                    for cap_name in cap_scores[0].keys():
                        scores = [cs[cap_name] for cs in cap_scores]
                        ax.plot(epochs[:len(scores)], scores, 
                               label=f'Capability: {cap_name}', alpha=0.8, linewidth=2)
                    ax.set_title('Capability Development')
                else:
                    # Single capability score
                    ax.plot(epochs[:len(cap_scores)], cap_scores, 
                           label='Capability Score', alpha=0.8, linewidth=2)
                    ax.set_title('Overall Capability Score')
            else:
                ax.text(0.5, 0.5, 'Emergent Abilities\nwould appear here', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Emergent Abilities')
        
        else:
            # General case: show learning rate if available
            if self.training_data['learning_rate']:
                ax.plot(epochs[:len(self.training_data['learning_rate'])], 
                       self.training_data['learning_rate'], 
                       color='purple', alpha=0.8, linewidth=2, label='Learning Rate')
                ax.set_yscale('log')
                ax.set_title('Learning Rate Schedule')
            else:
                ax.text(0.5, 0.5, 'Additional metrics\nwould appear here', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Additional Metrics')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')
    
    def create_phase_transition_analysis(
        self,
        transition_idx: int = 0,
        window_size: int = 200,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create detailed analysis of a specific phase transition.
        
        Args:
            transition_idx: Index of transition to analyze
            window_size: Window size around transition
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        if not self.phase_transitions or transition_idx >= len(self.phase_transitions):
            raise ValueError("No phase transitions detected or invalid index")
        
        transition = self.phase_transitions[transition_idx]
        transition_epoch = transition['epoch']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Phase Transition Analysis - Epoch {transition_epoch}', fontsize=16)
        
        epochs = np.array(self.training_data['epochs'])
        
        # Find window around transition
        transition_epoch_idx = np.where(epochs == transition_epoch)[0][0]
        start_idx = max(0, transition_epoch_idx - window_size)
        end_idx = min(len(epochs), transition_epoch_idx + window_size)
        
        window_epochs = epochs[start_idx:end_idx]
        
        # Plot 1: Accuracy around transition
        ax = axes[0, 0]
        ax.plot(window_epochs, self.training_data['test_acc'][start_idx:end_idx], 
               'o-', label='Test Accuracy', markersize=3)
        ax.axvline(x=transition_epoch, color='red', linestyle='--', alpha=0.7, label='Transition')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Test Accuracy Around Transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Loss around transition
        ax = axes[0, 1]
        ax.plot(window_epochs, self.training_data['test_loss'][start_idx:end_idx], 
               'o-', label='Test Loss', markersize=3, color='orange')
        ax.axvline(x=transition_epoch, color='red', linestyle='--', alpha=0.7, label='Transition')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Loss')
        ax.set_title('Test Loss Around Transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Generalization gap
        ax = axes[1, 0]
        gen_gap = (np.array(self.training_data['train_acc']) - 
                  np.array(self.training_data['test_acc']))[start_idx:end_idx]
        ax.plot(window_epochs, gen_gap, 'o-', label='Generalization Gap', markersize=3, color='red')
        ax.axvline(x=transition_epoch, color='red', linestyle='--', alpha=0.7, label='Transition')
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train - Test Accuracy')
        ax.set_title('Generalization Gap Around Transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Transition sharpness visualization
        ax = axes[1, 1]
        
        # Fit sigmoid to transition
        x_fit = np.arange(len(window_epochs))
        y_fit = self.training_data['test_acc'][start_idx:end_idx]
        
        try:
            def sigmoid(x, a, b, c, d):
                return a / (1 + np.exp(-b * (x - c))) + d
            
            p0 = [np.max(y_fit) - np.min(y_fit), 0.1, len(x_fit)/2, np.min(y_fit)]
            popt, _ = curve_fit(sigmoid, x_fit, y_fit, p0=p0, maxfev=1000)
            
            x_smooth = np.linspace(0, len(x_fit)-1, 200)
            y_smooth = sigmoid(x_smooth, *popt)
            
            # Convert back to epoch scale
            epoch_smooth = np.interp(x_smooth, x_fit, window_epochs)
            
            ax.plot(window_epochs, y_fit, 'o', label='Data', markersize=4, alpha=0.7)
            ax.plot(epoch_smooth, y_smooth, '-', label=f'Sigmoid Fit (sharpness={popt[1]:.2f})', linewidth=2)
            
        except:
            ax.plot(window_epochs, y_fit, 'o-', label='Test Accuracy', markersize=3)
        
        ax.axvline(x=transition_epoch, color='red', linestyle='--', alpha=0.7, label='Transition')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Transition Sharpness Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparative_plot(
        self,
        other_plotters: List['DelayedGeneralizationPlotter'],
        labels: List[str],
        metric: str = 'test_acc',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparative plot across multiple experiments.
        
        Args:
            other_plotters: List of other plotter instances
            labels: Labels for each experiment
            metric: Metric to compare
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        all_plotters = [self] + other_plotters
        all_labels = ['Current'] + labels if len(labels) == len(other_plotters) else [f'Exp {i}' for i in range(len(all_plotters))]
        
        for plotter, label in zip(all_plotters, all_labels):
            epochs = plotter.training_data['epochs']
            values = plotter.training_data[metric]
            
            ax.plot(epochs, values, label=label, alpha=0.8, linewidth=2)
            
            # Mark transitions for each experiment
            for transition in plotter.phase_transitions:
                ax.axvline(x=transition['epoch'], alpha=0.5, linestyle=':', 
                          label=f'{label} Transition' if transition == plotter.phase_transitions[0] else '')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Comparative Analysis - {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_data(self, save_path: str):
        """Export training data to CSV for external analysis."""
        
        # Convert to DataFrame
        max_length = max(len(v) for v in self.training_data.values() if isinstance(v, list))
        
        export_data = {}
        for key, values in self.training_data.items():
            if isinstance(values, list):
                # Pad shorter lists with NaN
                padded_values = values + [np.nan] * (max_length - len(values))
                export_data[key] = padded_values[:max_length]
        
        df = pd.DataFrame(export_data)
        
        # Add transition information
        transition_epochs = [t['epoch'] for t in self.phase_transitions]
        df['is_transition_epoch'] = df['epochs'].isin(transition_epochs)
        
        df.to_csv(save_path, index=False)
        print(f"Training data exported to {save_path}")
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """Get summary statistics of detected phase transitions."""
        
        if not self.phase_transitions:
            return {'num_transitions': 0}
        
        summary = {
            'num_transitions': len(self.phase_transitions),
            'transition_epochs': [t['epoch'] for t in self.phase_transitions],
            'improvements': [t['improvement'] for t in self.phase_transitions],
            'sharpness_scores': [t['sharpness'] for t in self.phase_transitions],
            'avg_improvement': np.mean([t['improvement'] for t in self.phase_transitions]),
            'avg_sharpness': np.mean([t['sharpness'] for t in self.phase_transitions])
        }
        
        return summary