#!/usr/bin/env python3
"""
Weights & Biases Integration for Delayed Generalization Research

This module provides utilities for tracking delayed generalization experiments
with wandb, including specialized metrics and visualizations for phenomena
like grokking, simplicity bias, and phase transitions.
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import logging


class DelayedGeneralizationLogger:
    """
    Enhanced wandb logger for delayed generalization experiments.
    
    Features:
    - Automatic detection of phase transitions
    - Specialized metrics for different phenomena
    - Custom visualizations for training dynamics
    - Integration with existing training scripts
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        phenomenon_type: str = 'general',
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        save_code: bool = True
    ):
        """
        Initialize wandb logger for delayed generalization experiments.
        
        Args:
            project_name: WandB project name
            experiment_name: Name for this specific experiment
            config: Experiment configuration dictionary
            phenomenon_type: Type of delayed generalization ('grokking', 'simplicity_bias', 'phase_transitions')
            tags: Additional tags for the run
            notes: Experiment notes
            save_code: Whether to save code to wandb
        """
        self.phenomenon_type = phenomenon_type
        
        # Initialize wandb run
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=tags or [phenomenon_type],
            notes=notes,
            save_code=save_code
        )
        
        # Metrics storage for analysis
        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'epochs': []
        }
        
        # Phenomenon-specific metrics
        if phenomenon_type == 'grokking':
            self.metrics_history.update({
                'generalization_gap': [],
                'weight_norms': [],
                'gradient_norms': []
            })
        elif phenomenon_type == 'simplicity_bias':
            self.metrics_history.update({
                'worst_group_acc': [],
                'group_accuracies': [],
                'bias_score': []
            })
        elif phenomenon_type == 'phase_transitions':
            self.metrics_history.update({
                'emergent_abilities': [],
                'transition_sharpness': [],
                'capability_scores': []
            })
        
        # Phase transition detection
        self.phase_transitions = []
        self.last_transition_check = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized wandb logger for {phenomenon_type} experiment: {experiment_name}")
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
        **kwargs
    ):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            test_loss: Test/validation loss
            train_acc: Training accuracy
            test_acc: Test/validation accuracy
            **kwargs: Additional metrics specific to the phenomenon
        """
        
        # Core metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'generalization_gap': train_acc - test_acc
        }
        
        # Store in history
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['test_acc'].append(test_acc)
        self.metrics_history['generalization_gap'].append(train_acc - test_acc)
        
        # Add phenomenon-specific metrics
        if self.phenomenon_type == 'grokking':
            self._log_grokking_metrics(metrics, kwargs)
        elif self.phenomenon_type == 'simplicity_bias':
            self._log_bias_metrics(metrics, kwargs)
        elif self.phenomenon_type == 'phase_transitions':
            self._log_transition_metrics(metrics, kwargs)
        
        # Add any additional metrics
        metrics.update(kwargs)
        
        # Log to wandb
        wandb.log(metrics, step=epoch)
        
        # Check for phase transitions
        if epoch - self.last_transition_check >= 100:  # Check every 100 epochs
            self._detect_phase_transitions(epoch)
            self.last_transition_check = epoch
    
    def _log_grokking_metrics(self, metrics: Dict, additional: Dict):
        """Log grokking-specific metrics."""
        
        # Weight and gradient norms
        if 'weight_norms' in additional:
            weight_norms = additional['weight_norms']
            metrics['weight_norm_mean'] = np.mean(list(weight_norms.values()))
            metrics['weight_norm_std'] = np.std(list(weight_norms.values()))
            self.metrics_history['weight_norms'].append(weight_norms)
            
            # Log individual layer norms
            for layer_name, norm in weight_norms.items():
                metrics[f'weight_norm_{layer_name}'] = norm
        
        if 'gradient_norms' in additional:
            grad_norms = additional['gradient_norms']
            metrics['gradient_norm_mean'] = np.mean(list(grad_norms.values()))
            metrics['gradient_norm_std'] = np.std(list(grad_norms.values()))
            self.metrics_history['gradient_norms'].append(grad_norms)
            
            # Log individual layer gradient norms
            for layer_name, norm in grad_norms.items():
                metrics[f'grad_norm_{layer_name}'] = norm
        
        # Grokking detection
        train_acc = metrics['train_acc']
        test_acc = metrics['test_acc']
        
        # Simple grokking detection: high train acc, suddenly improved test acc
        if (train_acc > 0.99 and test_acc > 0.95 and 
            len(self.metrics_history['test_acc']) > 100):
            
            # Check if this is a sudden improvement
            recent_test_acc = self.metrics_history['test_acc'][-100:-1]
            if np.mean(recent_test_acc) < 0.7:  # Was previously poor
                metrics['grokking_detected'] = 1
                self.logger.info(f"Grokking detected at epoch {metrics['epoch']}!")
            else:
                metrics['grokking_detected'] = 0
        else:
            metrics['grokking_detected'] = 0
    
    def _log_bias_metrics(self, metrics: Dict, additional: Dict):
        """Log simplicity bias specific metrics."""
        
        if 'group_accuracies' in additional:
            group_accs = additional['group_accuracies']
            
            # Worst group accuracy
            worst_acc = min(group_accs.values())
            metrics['worst_group_acc'] = worst_acc
            self.metrics_history['worst_group_acc'].append(worst_acc)
            
            # Group accuracy gap
            best_acc = max(group_accs.values())
            metrics['group_acc_gap'] = best_acc - worst_acc
            
            # Log individual group accuracies
            for group_name, acc in group_accs.items():
                metrics[f'group_acc_{group_name}'] = acc
            
            self.metrics_history['group_accuracies'].append(group_accs)
        
        if 'bias_score' in additional:
            bias_score = additional['bias_score']
            metrics['bias_score'] = bias_score
            self.metrics_history['bias_score'].append(bias_score)
            
            # Bias reduction detection
            if len(self.metrics_history['bias_score']) > 50:
                recent_bias = np.mean(self.metrics_history['bias_score'][-20:])
                early_bias = np.mean(self.metrics_history['bias_score'][:20])
                
                bias_reduction = early_bias - recent_bias
                metrics['bias_reduction'] = bias_reduction
                
                if bias_reduction > 0.2:  # Significant bias reduction
                    metrics['bias_mitigation_detected'] = 1
                else:
                    metrics['bias_mitigation_detected'] = 0
    
    def _log_transition_metrics(self, metrics: Dict, additional: Dict):
        """Log phase transition specific metrics."""
        
        if 'emergent_abilities' in additional:
            abilities = additional['emergent_abilities']
            
            # Count emerged abilities
            emerged_count = sum(1 for emerged in abilities.values() if emerged)
            metrics['emerged_abilities_count'] = emerged_count
            
            # Log individual abilities
            for ability_name, emerged in abilities.items():
                metrics[f'ability_{ability_name}'] = 1 if emerged else 0
            
            self.metrics_history['emergent_abilities'].append(abilities)
        
        if 'capability_scores' in additional:
            cap_scores = additional['capability_scores']
            
            # Average capability score
            avg_capability = np.mean(list(cap_scores.values()))
            metrics['avg_capability_score'] = avg_capability
            
            # Log individual capability scores
            for cap_name, score in cap_scores.items():
                metrics[f'capability_{cap_name}'] = score
            
            self.metrics_history['capability_scores'].append(cap_scores)
    
    def _detect_phase_transitions(self, current_epoch: int):
        """Detect phase transitions in training dynamics."""
        
        if len(self.metrics_history['test_acc']) < 200:  # Need sufficient history
            return
        
        # Look for sudden improvements in test accuracy
        recent_window = 50
        comparison_window = 100
        
        if len(self.metrics_history['test_acc']) < comparison_window + recent_window:
            return
        
        # Recent performance
        recent_acc = np.mean(self.metrics_history['test_acc'][-recent_window:])
        
        # Earlier performance
        earlier_acc = np.mean(
            self.metrics_history['test_acc'][-comparison_window-recent_window:-recent_window]
        )
        
        # Detect significant improvement
        improvement = recent_acc - earlier_acc
        
        if improvement > 0.2:  # 20% improvement threshold
            transition = {
                'epoch': current_epoch,
                'improvement': improvement,
                'recent_acc': recent_acc,
                'earlier_acc': earlier_acc,
                'type': 'test_accuracy_jump'
            }
            
            self.phase_transitions.append(transition)
            
            # Log transition
            wandb.log({
                'phase_transition_detected': 1,
                'transition_epoch': current_epoch,
                'transition_improvement': improvement,
                'transition_type': 'test_accuracy_jump'
            }, step=current_epoch)
            
            self.logger.info(f"Phase transition detected at epoch {current_epoch}: "
                           f"{improvement:.3f} improvement in test accuracy")
    
    def log_model_analysis(
        self,
        epoch: int,
        model: torch.nn.Module,
        analyze_weights: bool = True,
        analyze_gradients: bool = True,
        analyze_activations: bool = False
    ):
        """
        Log detailed model analysis metrics.
        
        Args:
            epoch: Current epoch
            model: PyTorch model to analyze
            analyze_weights: Whether to analyze weight distributions
            analyze_gradients: Whether to analyze gradient norms
            analyze_activations: Whether to analyze activation patterns
        """
        
        analysis_metrics = {}
        
        if analyze_weights:
            weight_analysis = self._analyze_weights(model)
            analysis_metrics.update(weight_analysis)
        
        if analyze_gradients:
            gradient_analysis = self._analyze_gradients(model)
            analysis_metrics.update(gradient_analysis)
        
        if analyze_activations:
            # This would require forward hooks and sample data
            pass  # Implement if needed
        
        # Log analysis metrics
        if analysis_metrics:
            wandb.log(analysis_metrics, step=epoch)
    
    def _analyze_weights(self, model: torch.nn.Module) -> Dict[str, float]:
        """Analyze weight distributions and norms."""
        
        weight_metrics = {}
        layer_norms = {}
        
        total_params = 0
        total_norm = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Layer-specific metrics
                layer_norm = param.data.norm().item()
                layer_norms[name] = layer_norm
                
                # Global metrics
                num_params = param.numel()
                total_params += num_params
                total_norm += layer_norm ** 2
                
                # Distribution metrics
                weight_metrics[f'{name}_norm'] = layer_norm
                weight_metrics[f'{name}_mean'] = param.data.mean().item()
                weight_metrics[f'{name}_std'] = param.data.std().item()
                weight_metrics[f'{name}_max'] = param.data.max().item()
                weight_metrics[f'{name}_min'] = param.data.min().item()
        
        # Global weight metrics
        weight_metrics['total_weight_norm'] = np.sqrt(total_norm)
        weight_metrics['avg_layer_norm'] = np.mean(list(layer_norms.values()))
        weight_metrics['std_layer_norm'] = np.std(list(layer_norms.values()))
        
        return weight_metrics
    
    def _analyze_gradients(self, model: torch.nn.Module) -> Dict[str, float]:
        """Analyze gradient norms and distributions."""
        
        gradient_metrics = {}
        layer_grad_norms = {}
        
        total_grad_norm = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Layer-specific gradient metrics
                grad_norm = param.grad.norm().item()
                layer_grad_norms[name] = grad_norm
                
                total_grad_norm += grad_norm ** 2
                
                # Distribution metrics
                gradient_metrics[f'{name}_grad_norm'] = grad_norm
                gradient_metrics[f'{name}_grad_mean'] = param.grad.mean().item()
                gradient_metrics[f'{name}_grad_std'] = param.grad.std().item()
        
        # Global gradient metrics
        gradient_metrics['total_grad_norm'] = np.sqrt(total_grad_norm)
        if layer_grad_norms:
            gradient_metrics['avg_grad_norm'] = np.mean(list(layer_grad_norms.values()))
            gradient_metrics['std_grad_norm'] = np.std(list(layer_grad_norms.values()))
        
        return gradient_metrics
    
    def create_training_dynamics_plot(self, save_to_wandb: bool = True) -> plt.Figure:
        """Create comprehensive training dynamics plot."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Dynamics - {self.phenomenon_type.title()}', fontsize=16)
        
        epochs = self.metrics_history['epochs']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, self.metrics_history['test_loss'], label='Test Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.metrics_history['train_acc'], label='Train Acc', alpha=0.8)
        axes[0, 1].plot(epochs, self.metrics_history['test_acc'], label='Test Acc', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Generalization gap
        axes[1, 0].plot(epochs, self.metrics_history['generalization_gap'], 
                       color='red', alpha=0.8, label='Train - Test Acc')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Generalization Gap')
        axes[1, 0].set_title('Generalization Gap')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phenomenon-specific plot
        if self.phenomenon_type == 'grokking' and self.metrics_history['weight_norms']:
            # Weight norm evolution
            weight_norm_means = [np.mean(list(wn.values())) for wn in self.metrics_history['weight_norms']]
            axes[1, 1].plot(epochs[:len(weight_norm_means)], weight_norm_means, 
                           color='purple', alpha=0.8, label='Avg Weight Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Weight Norm')
            axes[1, 1].set_title('Weight Norm Evolution')
            
        elif self.phenomenon_type == 'simplicity_bias' and self.metrics_history['worst_group_acc']:
            # Worst group accuracy
            axes[1, 1].plot(epochs[:len(self.metrics_history['worst_group_acc'])], 
                           self.metrics_history['worst_group_acc'], 
                           color='orange', alpha=0.8, label='Worst Group Acc')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Worst Group Accuracy')
            axes[1, 1].set_title('Worst Group Performance')
            
        else:
            # Default: learning rate or other metric
            axes[1, 1].text(0.5, 0.5, 'Additional metrics\nwould appear here', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Additional Metrics')
        
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark phase transitions
        for transition in self.phase_transitions:
            for ax in axes.flat:
                ax.axvline(x=transition['epoch'], color='red', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        if save_to_wandb:
            wandb.log({"training_dynamics": wandb.Image(fig)})
        
        return fig
    
    def save_experiment_summary(self, final_metrics: Dict[str, Any]):
        """Save comprehensive experiment summary."""
        
        summary = {
            'experiment_config': dict(wandb.config),
            'phenomenon_type': self.phenomenon_type,
            'final_metrics': final_metrics,
            'phase_transitions': self.phase_transitions,
            'total_epochs': len(self.metrics_history['epochs']),
            'best_test_acc': max(self.metrics_history['test_acc']) if self.metrics_history['test_acc'] else 0,
            'final_test_acc': self.metrics_history['test_acc'][-1] if self.metrics_history['test_acc'] else 0,
            'training_stability': self._compute_training_stability()
        }
        
        # Phenomenon-specific summary
        if self.phenomenon_type == 'grokking':
            summary.update(self._grokking_summary())
        elif self.phenomenon_type == 'simplicity_bias':
            summary.update(self._bias_summary())
        elif self.phenomenon_type == 'phase_transitions':
            summary.update(self._transition_summary())
        
        # Save to wandb
        wandb.summary.update(summary)
        
        # Save detailed history
        history_artifact = wandb.Artifact(
            name=f"training_history_{wandb.run.id}",
            type="training_data"
        )
        
        # Save metrics history as JSON
        with open('metrics_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_compatible_history = {}
            for key, value in self.metrics_history.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        json_compatible_history[key] = value
                    else:
                        json_compatible_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                else:
                    json_compatible_history[key] = value
            
            json.dump(json_compatible_history, f, indent=2)
        
        history_artifact.add_file('metrics_history.json')
        wandb.log_artifact(history_artifact)
        
        self.logger.info(f"Experiment summary saved. Best test accuracy: {summary['best_test_acc']:.4f}")
        
        return summary
    
    def _compute_training_stability(self) -> Dict[str, float]:
        """Compute training stability metrics."""
        
        if len(self.metrics_history['test_acc']) < 100:
            return {}
        
        test_accs = np.array(self.metrics_history['test_acc'])
        
        # Compute stability over different windows
        stability = {}
        
        # Overall variance
        stability['test_acc_variance'] = np.var(test_accs)
        stability['test_acc_std'] = np.std(test_accs)
        
        # Late training stability (last 25% of training)
        late_start = len(test_accs) * 3 // 4
        late_accs = test_accs[late_start:]
        stability['late_training_stability'] = 1.0 / (1.0 + np.std(late_accs))
        
        # Trend analysis
        x = np.arange(len(test_accs))
        slope, intercept = np.polyfit(x, test_accs, 1)
        stability['test_acc_trend'] = slope
        
        return stability
    
    def _grokking_summary(self) -> Dict[str, Any]:
        """Generate grokking-specific summary."""
        
        summary = {}
        
        # Detect grokking epoch
        grokking_epoch = None
        test_accs = self.metrics_history['test_acc']
        
        for i in range(100, len(test_accs)):
            if test_accs[i] > 0.95 and np.mean(test_accs[max(0, i-100):i]) < 0.7:
                grokking_epoch = self.metrics_history['epochs'][i]
                break
        
        summary['grokking_epoch'] = grokking_epoch
        summary['grokking_detected'] = grokking_epoch is not None
        
        if grokking_epoch:
            # Compute grokking sharpness
            grok_idx = self.metrics_history['epochs'].index(grokking_epoch)
            window = 50
            
            if grok_idx >= window and grok_idx + window < len(test_accs):
                pre_grok = np.mean(test_accs[grok_idx-window:grok_idx])
                post_grok = np.mean(test_accs[grok_idx:grok_idx+window])
                summary['grokking_sharpness'] = post_grok - pre_grok
        
        return summary
    
    def _bias_summary(self) -> Dict[str, Any]:
        """Generate bias-specific summary."""
        
        summary = {}
        
        if self.metrics_history['worst_group_acc']:
            worst_group_accs = self.metrics_history['worst_group_acc']
            
            summary['final_worst_group_acc'] = worst_group_accs[-1]
            summary['best_worst_group_acc'] = max(worst_group_accs)
            summary['worst_group_improvement'] = worst_group_accs[-1] - worst_group_accs[0]
        
        if self.metrics_history['bias_score']:
            bias_scores = self.metrics_history['bias_score']
            
            summary['final_bias_score'] = bias_scores[-1]
            summary['initial_bias_score'] = bias_scores[0]
            summary['bias_reduction'] = bias_scores[0] - bias_scores[-1]
            summary['bias_mitigation_success'] = summary['bias_reduction'] > 0.2
        
        return summary
    
    def _transition_summary(self) -> Dict[str, Any]:
        """Generate phase transition summary."""
        
        summary = {}
        
        summary['num_phase_transitions'] = len(self.phase_transitions)
        summary['phase_transition_epochs'] = [t['epoch'] for t in self.phase_transitions]
        
        if self.metrics_history['emergent_abilities']:
            # Count final emerged abilities
            final_abilities = self.metrics_history['emergent_abilities'][-1]
            summary['final_emerged_abilities'] = sum(1 for emerged in final_abilities.values() if emerged)
            summary['total_abilities_tracked'] = len(final_abilities)
        
        return summary
    
    def finish(self):
        """Finish the wandb run."""
        
        # Create final visualization
        self.create_training_dynamics_plot(save_to_wandb=True)
        
        # Save experiment summary
        final_metrics = {
            'final_train_acc': self.metrics_history['train_acc'][-1] if self.metrics_history['train_acc'] else 0,
            'final_test_acc': self.metrics_history['test_acc'][-1] if self.metrics_history['test_acc'] else 0,
            'best_test_acc': max(self.metrics_history['test_acc']) if self.metrics_history['test_acc'] else 0,
            'total_epochs': len(self.metrics_history['epochs'])
        }
        
        self.save_experiment_summary(final_metrics)
        
        # Finish wandb run
        wandb.finish()
        
        self.logger.info("WandB logging completed and run finished.")


# Utility functions for wandb integration

def setup_wandb_for_phenomenon(
    phenomenon_type: str,
    project_name: str,
    config: Dict[str, Any]
) -> DelayedGeneralizationLogger:
    """
    Quick setup function for different delayed generalization phenomena.
    
    Args:
        phenomenon_type: 'grokking', 'simplicity_bias', or 'phase_transitions'
        project_name: WandB project name
        config: Experiment configuration
        
    Returns:
        Configured DelayedGeneralizationLogger
    """
    
    # Phenomenon-specific configurations
    if phenomenon_type == 'grokking':
        experiment_name = f"grokking_{config.get('dataset', 'unknown')}_{config.get('model', 'transformer')}"
        tags = ['grokking', 'algorithmic', 'phase_transition']
        notes = "Grokking experiment - studying sudden generalization after memorization"
        
    elif phenomenon_type == 'simplicity_bias':
        experiment_name = f"bias_{config.get('dataset', 'unknown')}_{config.get('bias_strength', '0.8')}"
        tags = ['simplicity_bias', 'robustness', 'spurious_correlation']
        notes = "Simplicity bias experiment - studying spurious correlation learning"
        
    elif phenomenon_type == 'phase_transitions':
        experiment_name = f"transitions_{config.get('model_size', 'unknown')}_{config.get('capabilities', 'multi')}"
        tags = ['phase_transitions', 'emergent_abilities', 'scaling']
        notes = "Phase transition experiment - studying emergent capabilities"
        
    else:
        experiment_name = f"delayed_gen_{phenomenon_type}"
        tags = ['delayed_generalization']
        notes = "General delayed generalization experiment"
    
    return DelayedGeneralizationLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config,
        phenomenon_type=phenomenon_type,
        tags=tags,
        notes=notes
    )


def integrate_with_existing_trainer(trainer_class):
    """
    Decorator to integrate wandb logging with existing trainer classes.
    
    Usage:
        @integrate_with_existing_trainer
        class MyTrainer:
            def train_epoch(self):
                # ... training code ...
                return train_loss, train_acc
    """
    
    class WandbIntegratedTrainer(trainer_class):
        def __init__(self, *args, **kwargs):
            # Extract wandb config if provided
            self.wandb_logger = kwargs.pop('wandb_logger', None)
            super().__init__(*args, **kwargs)
        
        def train(self, *args, **kwargs):
            # Wrap the original train method
            if hasattr(super(), 'train'):
                return super().train(*args, **kwargs)
            else:
                raise NotImplementedError("Original trainer must implement train method")
        
        def _log_epoch_if_wandb(self, epoch, train_loss, test_loss, train_acc, test_acc, **kwargs):
            """Helper method to log epoch metrics if wandb logger is available."""
            if self.wandb_logger:
                self.wandb_logger.log_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    **kwargs
                )
    
    return WandbIntegratedTrainer


# Example integration with common training patterns

def create_wandb_config_from_args(args) -> Dict[str, Any]:
    """Convert argparse Namespace to wandb config dict."""
    
    if hasattr(args, '__dict__'):
        config = vars(args)
    else:
        config = args
    
    # Add system info
    import torch
    import platform
    
    config.update({
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'python_version': platform.python_version(),
        'platform': platform.platform()
    })
    
    return config


def log_dataset_info(logger: DelayedGeneralizationLogger, dataset_info: Dict[str, Any]):
    """Log dataset-specific information to wandb."""
    
    wandb.log({
        'dataset_size_train': dataset_info.get('train_size', 0),
        'dataset_size_test': dataset_info.get('test_size', 0),
        'dataset_num_classes': dataset_info.get('num_classes', 0),
        'dataset_input_shape': str(dataset_info.get('input_shape', 'unknown')),
        'dataset_type': dataset_info.get('type', 'unknown')
    })
    
    # Log dataset-specific metrics
    if 'bias_strength' in dataset_info:
        wandb.log({'dataset_bias_strength': dataset_info['bias_strength']})
    
    if 'correlation_strength' in dataset_info:
        wandb.log({'dataset_correlation_strength': dataset_info['correlation_strength']})


def create_hyperparameter_sweep_config(phenomenon_type: str) -> Dict[str, Any]:
    """Create wandb sweep configuration for different phenomena."""
    
    if phenomenon_type == 'grokking':
        return {
            'method': 'grid',
            'parameters': {
                'learning_rate': {'values': [1e-4, 1e-3, 1e-2]},
                'weight_decay': {'values': [1e-3, 1e-2, 1e-1]},
                'batch_size': {'values': [128, 256, 512]},
                'model_size': {'values': [64, 128, 256]},
                'optimizer': {'values': ['adam', 'adamw', 'sgd']}
            },
            'metric': {
                'name': 'best_test_acc',
                'goal': 'maximize'
            }
        }
    
    elif phenomenon_type == 'simplicity_bias':
        return {
            'method': 'bayes',
            'parameters': {
                'learning_rate': {'min': 1e-5, 'max': 1e-1},
                'weight_decay': {'min': 1e-5, 'max': 1e-1},
                'dropout': {'min': 0.0, 'max': 0.5},
                'bias_strength': {'values': [0.7, 0.8, 0.9, 0.95]},
                'training_method': {'values': ['erm', 'group_dro', 'irm']}
            },
            'metric': {
                'name': 'worst_group_acc',
                'goal': 'maximize'
            }
        }
    
    elif phenomenon_type == 'phase_transitions':
        return {
            'method': 'random',
            'parameters': {
                'model_size': {'values': [1e6, 5e6, 1e7, 5e7]},
                'learning_rate': {'min': 1e-5, 'max': 1e-2},
                'batch_size': {'values': [32, 64, 128, 256]},
                'data_size': {'values': [1e3, 5e3, 1e4, 5e4]},
                'warmup_steps': {'min': 0, 'max': 2000}
            },
            'metric': {
                'name': 'emerged_abilities_count',
                'goal': 'maximize'
            }
        }
    
    else:
        return {
            'method': 'grid',
            'parameters': {
                'learning_rate': {'values': [1e-4, 1e-3, 1e-2]},
                'batch_size': {'values': [32, 64, 128]},
                'weight_decay': {'values': [1e-4, 1e-3, 1e-2]}
            }
        }