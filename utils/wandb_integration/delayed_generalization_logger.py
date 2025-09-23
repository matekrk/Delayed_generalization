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
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from collections import defaultdict
import torch.nn.functional as F


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
            save_code=save_code,
            dir=config.get('save_dir', './wandb')
        )
        
        # Metrics storage for analysis
        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'epochs': [],
            'generalization_gap': [],  # Add this to base metrics
            'gradient_directions': [],  # For gradient direction analysis
            'mutual_information': [],  # For mutual information tracking
            'statistical_significance': []  # For significance testing
        }
        
        # Enhanced tracking for sophisticated analysis
        self.gradient_history = []
        self.feature_representations = []
        self.seed_results = {}  # For meta-analysis across seeds
        
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
        """Analyze gradient norms and distributions with enhanced direction analysis."""
        
        gradient_metrics = {}
        layer_grad_norms = {}
        
        total_grad_norm = 0
        all_gradients = []
        layer_gradients = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Layer-specific gradient metrics
                grad_norm = param.grad.norm().item()
                layer_grad_norms[name] = grad_norm
                
                total_grad_norm += grad_norm ** 2
                
                # Store gradients for direction analysis
                grad_flat = param.grad.detach().flatten()
                all_gradients.append(grad_flat)
                layer_gradients[name] = grad_flat
                
                # Distribution metrics
                gradient_metrics[f'{name}_grad_norm'] = grad_norm
                gradient_metrics[f'{name}_grad_mean'] = param.grad.mean().item()
                gradient_metrics[f'{name}_grad_std'] = param.grad.std().item()
        
        # Global gradient metrics
        gradient_metrics['total_grad_norm'] = np.sqrt(total_grad_norm)
        if layer_grad_norms:
            gradient_metrics['avg_grad_norm'] = np.mean(list(layer_grad_norms.values()))
            gradient_metrics['std_grad_norm'] = np.std(list(layer_grad_norms.values()))
        
        # Enhanced: Gradient direction analysis
        if len(all_gradients) > 0:
            all_grads_tensor = torch.cat(all_gradients)
            self.gradient_history.append(all_grads_tensor.cpu().numpy())
            
            # Gradient direction consistency
            if len(self.gradient_history) > 1:
                current_grad = self.gradient_history[-1]
                prev_grad = self.gradient_history[-2]
                
                # Cosine similarity between consecutive gradient updates
                cos_sim = np.dot(current_grad, prev_grad) / (
                    np.linalg.norm(current_grad) * np.linalg.norm(prev_grad) + 1e-8
                )
                gradient_metrics['gradient_direction_consistency'] = cos_sim
                
                # Gradient direction variance over recent steps
                if len(self.gradient_history) > 10:
                    recent_grads = np.array(self.gradient_history[-10:])
                    pairwise_similarities = []
                    for i in range(len(recent_grads)):
                        for j in range(i+1, len(recent_grads)):
                            sim = np.dot(recent_grads[i], recent_grads[j]) / (
                                np.linalg.norm(recent_grads[i]) * np.linalg.norm(recent_grads[j]) + 1e-8
                            )
                            pairwise_similarities.append(sim)
                    
                    gradient_metrics['gradient_direction_variance'] = np.var(pairwise_similarities)
                    gradient_metrics['gradient_direction_stability'] = np.mean(pairwise_similarities)
        
        return gradient_metrics
    
    def analyze_mutual_information(
        self,
        epoch: int,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        max_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Analyze mutual information between features and targets.
        
        Args:
            epoch: Current epoch
            model: The model to analyze
            data_loader: Data loader for analysis
            max_samples: Maximum number of samples to use for efficiency
            
        Returns:
            Dictionary of mutual information metrics
        """
        
        model.eval()
        features_list = []
        targets_list = []
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                if i * data_loader.batch_size >= max_samples:
                    break
                    
                if hasattr(model, 'features'):
                    # For models with explicit feature extractor
                    features = model.features(inputs)
                    features = features.view(features.size(0), -1)
                else:
                    # Generic approach: use penultimate layer
                    features = self._extract_penultimate_features(model, inputs)
                
                features_list.append(features.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        if not features_list:
            return {}
            
        # Concatenate all features and targets
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        
        mi_metrics = {}
        
        # Compute mutual information for different feature subsets
        if all_features.shape[1] > 1:
            # Subsample features for computational efficiency
            n_features = min(all_features.shape[1], 100)
            feature_indices = np.random.choice(all_features.shape[1], n_features, replace=False)
            selected_features = all_features[:, feature_indices]
            
            try:
                # Mutual information between features and targets
                mi_scores = mutual_info_regression(selected_features, all_targets, random_state=42)
                
                mi_metrics['mutual_info_mean'] = np.mean(mi_scores)
                mi_metrics['mutual_info_std'] = np.std(mi_scores)
                mi_metrics['mutual_info_max'] = np.max(mi_scores)
                mi_metrics['mutual_info_min'] = np.min(mi_scores)
                
                # Store for trend analysis
                self.metrics_history['mutual_information'].append({
                    'epoch': epoch,
                    'mi_mean': np.mean(mi_scores),
                    'mi_scores': mi_scores.tolist()
                })
                
                # Analyze trends if we have enough history
                if len(self.metrics_history['mutual_information']) > 5:
                    recent_mi = [entry['mi_mean'] for entry in self.metrics_history['mutual_information'][-5:]]
                    mi_trend = np.polyfit(range(len(recent_mi)), recent_mi, 1)[0]
                    mi_metrics['mutual_info_trend'] = mi_trend
                    
            except Exception as e:
                logging.warning(f"Failed to compute mutual information: {e}")
        
        model.train()
        return mi_metrics
    
    def _extract_penultimate_features(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from the penultimate layer of the model."""
        
        features = inputs
        modules = list(model.children())
        
        # Forward through all but the last layer
        for module in modules[:-1]:
            features = module(features)
            
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
            
        return features
    
    def compute_statistical_significance(
        self,
        epoch: int,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        min_history_length: int = 50
    ) -> Dict[str, float]:
        """
        Compute statistical significance of current performance vs baseline/history.
        
        Args:
            epoch: Current epoch
            current_metrics: Current performance metrics
            baseline_metrics: Optional baseline to compare against
            min_history_length: Minimum history length for statistical tests
            
        Returns:
            Dictionary of statistical significance metrics
        """
        
        sig_metrics = {}
        
        # We need sufficient history for meaningful statistical tests
        if len(self.metrics_history['test_acc']) < min_history_length:
            return sig_metrics
        
        # Test statistical significance of current performance vs recent average
        recent_window = min(20, len(self.metrics_history['test_acc']) // 4)
        recent_accs = self.metrics_history['test_acc'][-recent_window:]
        
        if 'test_acc' in current_metrics:
            current_acc = current_metrics['test_acc']
            
            # One-sample t-test against recent performance
            if len(recent_accs) > 3:
                t_stat, p_value = stats.ttest_1samp(recent_accs, current_acc)
                sig_metrics['test_acc_t_stat'] = t_stat
                sig_metrics['test_acc_p_value'] = p_value
                sig_metrics['test_acc_significant'] = 1 if p_value < 0.05 else 0
        
        # Test for trend significance
        if len(self.metrics_history['test_acc']) > 30:
            epochs_subset = range(len(self.metrics_history['test_acc']))
            trend_slope, _, _, p_value, _ = stats.linregress(
                epochs_subset, self.metrics_history['test_acc']
            )
            sig_metrics['accuracy_trend_slope'] = trend_slope
            sig_metrics['accuracy_trend_p_value'] = p_value
            sig_metrics['accuracy_trend_significant'] = 1 if p_value < 0.05 else 0
        
        # Compare against baseline if provided
        if baseline_metrics and 'test_acc' in baseline_metrics:
            baseline_acc = baseline_metrics['test_acc']
            recent_accs_array = np.array(recent_accs)
            
            # Test if recent performance is significantly different from baseline
            t_stat, p_value = stats.ttest_1samp(recent_accs_array, baseline_acc)
            sig_metrics['baseline_comparison_t_stat'] = t_stat
            sig_metrics['baseline_comparison_p_value'] = p_value
            sig_metrics['baseline_comparison_significant'] = 1 if p_value < 0.05 else 0
            
            # Effect size (Cohen's d)
            pooled_std = np.std(recent_accs_array)
            if pooled_std > 0:
                cohens_d = (np.mean(recent_accs_array) - baseline_acc) / pooled_std
                sig_metrics['baseline_comparison_effect_size'] = cohens_d
        
        # Store statistical significance history
        self.metrics_history['statistical_significance'].append({
            'epoch': epoch,
            'significance_metrics': sig_metrics
        })
        
        return sig_metrics
    
    def log_seed_result(self, seed: int, final_metrics: Dict[str, Any]):
        """
        Log results for a specific seed for meta-analysis.
        
        Args:
            seed: Random seed used for this run
            final_metrics: Final metrics achieved with this seed
        """
        
        self.seed_results[seed] = {
            'final_metrics': final_metrics,
            'metrics_history': {
                key: list(value) for key, value in self.metrics_history.items()
                if isinstance(value, list) and len(value) > 0
            },
            'phase_transitions': list(self.phase_transitions)
        }
        
        # Log seed-specific metrics to wandb
        wandb.log({
            f'seed_{seed}_final_test_acc': final_metrics.get('test_acc', 0),
            f'seed_{seed}_final_train_acc': final_metrics.get('train_acc', 0),
            f'seed_{seed}_num_transitions': len(self.phase_transitions)
        })
    
    @classmethod
    def perform_meta_analysis(
        cls, 
        project_name: str, 
        phenomenon_type: str,
        seed_results: Dict[int, Dict[str, Any]],
        save_to_wandb: bool = True
    ) -> Dict[str, Any]:
        """
        Perform meta-analysis across multiple seed runs.
        
        Args:
            project_name: WandB project name
            phenomenon_type: Type of phenomenon studied
            seed_results: Dictionary mapping seeds to their results
            save_to_wandb: Whether to save results to wandb
            
        Returns:
            Meta-analysis results
        """
        
        if len(seed_results) < 2:
            logging.warning("Need at least 2 seed results for meaningful meta-analysis")
            return {}
        
        meta_results = {
            'num_seeds': len(seed_results),
            'phenomenon_type': phenomenon_type
        }
        
        # Aggregate final performance metrics
        final_test_accs = []
        final_train_accs = []
        num_transitions = []
        
        for seed, results in seed_results.items():
            final_metrics = results.get('final_metrics', {})
            final_test_accs.append(final_metrics.get('test_acc', 0))
            final_train_accs.append(final_metrics.get('train_acc', 0))
            num_transitions.append(len(results.get('phase_transitions', [])))
        
        # Statistical analysis of final performance
        if final_test_accs:
            meta_results.update({
                'test_acc_mean': np.mean(final_test_accs),
                'test_acc_std': np.std(final_test_accs),
                'test_acc_min': np.min(final_test_accs),
                'test_acc_max': np.max(final_test_accs),
                'test_acc_median': np.median(final_test_accs),
                'test_acc_ci_95': cls._compute_confidence_interval(final_test_accs, 0.95)
            })
        
        if final_train_accs:
            meta_results.update({
                'train_acc_mean': np.mean(final_train_accs),
                'train_acc_std': np.std(final_train_accs),
                'train_acc_min': np.min(final_train_accs),
                'train_acc_max': np.max(final_train_accs)
            })
        
        # Analyze phase transitions
        if num_transitions:
            meta_results.update({
                'transitions_mean': np.mean(num_transitions),
                'transitions_std': np.std(num_transitions),
                'transitions_mode': stats.mode(num_transitions)[0][0] if len(num_transitions) > 1 else num_transitions[0]
            })
        
        # Analyze convergence consistency
        # Look at the variance in final epochs across seeds
        final_epochs = []
        for results in seed_results.values():
            history = results.get('metrics_history', {})
            if 'epochs' in history and history['epochs']:
                final_epochs.append(history['epochs'][-1])
        
        if final_epochs:
            meta_results['convergence_consistency'] = 1.0 / (1.0 + np.std(final_epochs))
        
        # Phenomenon-specific meta-analysis
        if phenomenon_type == 'grokking':
            meta_results.update(cls._grokking_meta_analysis(seed_results))
        elif phenomenon_type == 'simplicity_bias':
            meta_results.update(cls._bias_meta_analysis(seed_results))
        elif phenomenon_type == 'phase_transitions':
            meta_results.update(cls._transition_meta_analysis(seed_results))
        
        # Statistical robustness metrics
        if len(final_test_accs) >= 3:
            # Test for outliers using IQR method
            q75, q25 = np.percentile(final_test_accs, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = [acc for acc in final_test_accs if acc < lower_bound or acc > upper_bound]
            meta_results['outlier_fraction'] = len(outliers) / len(final_test_accs)
            meta_results['result_stability'] = 1.0 - meta_results['outlier_fraction']
        
        if save_to_wandb:
            # Create a new wandb run for meta-analysis
            meta_run = wandb.init(
                project=project_name,
                name=f"meta_analysis_{phenomenon_type}",
                job_type="meta_analysis",
                tags=['meta_analysis', phenomenon_type]
            )
            
            wandb.log(meta_results)
            
            # Create visualization of seed results
            cls._create_meta_analysis_plots(seed_results, meta_results)
            
            wandb.finish()
        
        return meta_results
    
    @staticmethod
    def _compute_confidence_interval(data: List[float], confidence: float = 0.95) -> List[float]:
        """Compute confidence interval for the mean."""
        
        if len(data) < 2:
            return [np.mean(data), np.mean(data)]
        
        alpha = 1 - confidence
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Use t-distribution for small samples
        t_val = stats.t.ppf(1 - alpha/2, n - 1)
        margin = t_val * std / np.sqrt(n)
        
        return [mean - margin, mean + margin]
    
    @staticmethod
    def _grokking_meta_analysis(seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform grokking-specific meta-analysis."""
        
        meta = {}
        
        # Analyze grokking epochs across seeds
        grokking_epochs = []
        grokking_detected = []
        
        for results in seed_results.values():
            final_metrics = results.get('final_metrics', {})
            grokking_epochs.append(final_metrics.get('grokking_epoch', None))
            grokking_detected.append(final_metrics.get('grokking_detected', False))
        
        # Filter out None values
        valid_grokking_epochs = [e for e in grokking_epochs if e is not None]
        
        if valid_grokking_epochs:
            meta.update({
                'grokking_epoch_mean': np.mean(valid_grokking_epochs),
                'grokking_epoch_std': np.std(valid_grokking_epochs),
                'grokking_success_rate': sum(grokking_detected) / len(grokking_detected)
            })
        
        return meta
    
    @staticmethod
    def _bias_meta_analysis(seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform bias-specific meta-analysis."""
        
        meta = {}
        
        # Analyze worst group accuracies
        worst_group_accs = []
        bias_scores = []
        
        for results in seed_results.values():
            final_metrics = results.get('final_metrics', {})
            if 'worst_group_acc' in final_metrics:
                worst_group_accs.append(final_metrics['worst_group_acc'])
            if 'bias_score' in final_metrics:
                bias_scores.append(final_metrics['bias_score'])
        
        if worst_group_accs:
            meta.update({
                'worst_group_acc_mean': np.mean(worst_group_accs),
                'worst_group_acc_std': np.std(worst_group_accs),
                'bias_mitigation_consistency': 1.0 / (1.0 + np.std(worst_group_accs))
            })
        
        return meta
    
    @staticmethod
    def _transition_meta_analysis(seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform phase transition specific meta-analysis."""
        
        meta = {}
        
        # Analyze emergent abilities
        emerged_abilities = []
        transition_epochs = []
        
        for results in seed_results.values():
            final_metrics = results.get('final_metrics', {})
            if 'emerged_abilities_count' in final_metrics:
                emerged_abilities.append(final_metrics['emerged_abilities_count'])
            
            # Extract transition epochs
            transitions = results.get('phase_transitions', [])
            if transitions:
                transition_epochs.extend([t['epoch'] for t in transitions])
        
        if emerged_abilities:
            meta.update({
                'emergent_abilities_mean': np.mean(emerged_abilities),
                'emergent_abilities_std': np.std(emerged_abilities),
                'emergence_consistency': 1.0 / (1.0 + np.std(emerged_abilities))
            })
        
        if transition_epochs:
            meta.update({
                'transition_epochs_mean': np.mean(transition_epochs),
                'transition_epochs_std': np.std(transition_epochs)
            })
        
        return meta
    
    @staticmethod
    def _create_meta_analysis_plots(seed_results: Dict[int, Dict[str, Any]], meta_results: Dict[str, Any]):
        """Create visualization plots for meta-analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Meta-Analysis Across Seeds', fontsize=16)
        
        # Plot 1: Final test accuracy distribution
        final_test_accs = []
        for results in seed_results.values():
            final_metrics = results.get('final_metrics', {})
            final_test_accs.append(final_metrics.get('test_acc', 0))
        
        if final_test_accs:
            axes[0, 0].hist(final_test_accs, bins=min(10, len(final_test_accs)), alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(np.mean(final_test_accs), color='red', linestyle='--', label=f'Mean: {np.mean(final_test_accs):.3f}')
            axes[0, 0].set_xlabel('Final Test Accuracy')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Final Test Accuracy Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training curves for all seeds
        for seed, results in seed_results.items():
            history = results.get('metrics_history', {})
            if 'epochs' in history and 'test_acc' in history:
                epochs = history['epochs']
                test_acc = history['test_acc']
                axes[0, 1].plot(epochs, test_acc, alpha=0.6, label=f'Seed {seed}')
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Training Curves (All Seeds)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Phase transitions count
        num_transitions = [len(results.get('phase_transitions', [])) for results in seed_results.values()]
        if num_transitions:
            unique_counts, counts = np.unique(num_transitions, return_counts=True)
            axes[1, 0].bar(unique_counts, counts, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Number of Phase Transitions')
            axes[1, 0].set_ylabel('Number of Seeds')
            axes[1, 0].set_title('Phase Transitions Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Convergence consistency
        final_epochs = []
        for results in seed_results.values():
            history = results.get('metrics_history', {})
            if 'epochs' in history and history['epochs']:
                final_epochs.append(history['epochs'][-1])
        
        if final_epochs and len(set(final_epochs)) > 1:
            axes[1, 1].hist(final_epochs, bins=min(10, len(final_epochs)), alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(final_epochs), color='red', linestyle='--', label=f'Mean: {np.mean(final_epochs):.0f}')
            axes[1, 1].set_xlabel('Final Epoch')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Convergence Epoch Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'All seeds converged\nat same epoch', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Convergence Consistency')
        
        plt.tight_layout()
        
        # Log the plot to wandb
        wandb.log({"meta_analysis_visualization": wandb.Image(fig)})
        plt.close(fig)
    
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
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb. This is a general-purpose logging method.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        wandb.log(metrics, step=step)
    
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


def create_hyperparameter_sweep_config(phenomenon_type: str, advanced: bool = False) -> Dict[str, Any]:
    """Create wandb sweep configuration for different phenomena with enhanced optimizer support."""
    
    if phenomenon_type == 'grokking':
        base_config = {
            'method': 'grid',
            'parameters': {
                'learning_rate': {'values': [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
                'weight_decay': {'values': [1e-4, 1e-3, 1e-2, 1e-1, 0.5]},
                'batch_size': {'values': [64, 128, 256, 512]},
                'model_size': {'values': [64, 128, 256, 512]},
                'optimizer': {'values': ['adam', 'adamw', 'sgd', 'rmsprop']}
            },
            'metric': {
                'name': 'best_test_acc',
                'goal': 'maximize'
            }
        }
        
        if advanced:
            base_config['parameters'].update({
                'optimizer': {'values': ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion']},
                'lr_scheduler': {'values': ['cosine', 'step', 'exponential', 'plateau', 'cyclic']},
                'warmup_steps': {'values': [0, 100, 500, 1000]},
                'gradient_clip': {'values': [0.0, 0.5, 1.0, 5.0]},
                'beta1': {'values': [0.8, 0.9, 0.95, 0.99]},
                'beta2': {'values': [0.95, 0.99, 0.999, 0.9999]},
                'epsilon': {'values': [1e-8, 1e-7, 1e-6]},
                'use_ema': {'values': [True, False]},
                'ema_decay': {'values': [0.99, 0.999, 0.9999]}
            })
            base_config['method'] = 'bayes'  # Use Bayesian optimization for advanced configs
            
        return base_config
    
    elif phenomenon_type == 'simplicity_bias':
        base_config = {
            'method': 'bayes',
            'parameters': {
                'learning_rate': {'min': 1e-5, 'max': 1e-1},
                'weight_decay': {'min': 1e-5, 'max': 1e-1},
                'dropout': {'min': 0.0, 'max': 0.5},
                'bias_strength': {'values': [0.7, 0.8, 0.9, 0.95, 0.99]},
                'training_method': {'values': ['erm', 'group_dro', 'irm', 'mixup']},
                'optimizer': {'values': ['adam', 'adamw', 'sgd']}
            },
            'metric': {
                'name': 'worst_group_acc',
                'goal': 'maximize'
            }
        }
        
        if advanced:
            base_config['parameters'].update({
                'optimizer': {'values': ['adam', 'adamw', 'sgd', 'rmsprop', 'lion']},
                'lr_scheduler': {'values': ['cosine', 'step', 'exponential', 'plateau']},
                'gradient_penalty': {'min': 0.0, 'max': 10.0},
                'label_smoothing': {'min': 0.0, 'max': 0.2},
                'mixup_alpha': {'min': 0.0, 'max': 2.0},
                'sam_rho': {'min': 0.0, 'max': 0.5},  # Sharpness-Aware Minimization
                'use_ema': {'values': [True, False]}
            })
            
        return base_config
    
    elif phenomenon_type == 'phase_transitions':
        base_config = {
            'method': 'random',
            'parameters': {
                'model_size': {'values': [1e6, 5e6, 1e7, 5e7]},
                'learning_rate': {'min': 1e-5, 'max': 1e-2},
                'batch_size': {'values': [32, 64, 128, 256, 512]},
                'data_size': {'values': [1e3, 5e3, 1e4, 5e4]},
                'warmup_steps': {'min': 0, 'max': 2000},
                'optimizer': {'values': ['adam', 'adamw', 'sgd']}
            },
            'metric': {
                'name': 'emerged_abilities_count',
                'goal': 'maximize'
            }
        }
        
        if advanced:
            base_config['parameters'].update({
                'optimizer': {'values': ['adam', 'adamw', 'sgd', 'rmsprop', 'lion']},
                'lr_scheduler': {'values': ['cosine', 'linear', 'polynomial', 'constant']},
                'temperature_scaling': {'min': 0.5, 'max': 2.0},
                'knowledge_distillation': {'values': [True, False]},
                'progressive_resizing': {'values': [True, False]},
                'curriculum_learning': {'values': [True, False]}
            })
            
        return base_config
    
    # Enhanced continual learning configuration
    elif phenomenon_type == 'continual_learning':
        return {
            'method': 'bayes',
            'parameters': {
                'learning_rate': {'min': 1e-5, 'max': 1e-2},
                'weight_decay': {'min': 1e-5, 'max': 1e-1},
                'optimizer': {'values': ['adam', 'adamw', 'sgd', 'rmsprop']},
                'memory_size': {'values': [100, 500, 1000, 2000]},
                'regularization_strength': {'min': 0.0, 'max': 10.0},
                'plasticity_factor': {'min': 0.1, 'max': 1.0},
                'consolidation_method': {'values': ['ewc', 'l2', 'dropout', 'packnet']},
                'task_order': {'values': ['sequential', 'interleaved', 'random']},
                'replay_strategy': {'values': ['none', 'random', 'herding', 'gradient_based']}
            },
            'metric': {
                'name': 'average_accuracy_after_all_tasks',
                'goal': 'maximize'
            }
        }
    
    else:
        return {
            'method': 'grid',
            'parameters': {
                'learning_rate': {'values': [1e-4, 1e-3, 1e-2]},
                'batch_size': {'values': [32, 64, 128]},
                'weight_decay': {'values': [1e-4, 1e-3, 1e-2]},
                'optimizer': {'values': ['adam', 'adamw', 'sgd']}
            }
        }


def create_advanced_optimizer_sweep(
    phenomenon_type: str,
    focus_optimizers: Optional[List[str]] = None,
    include_scheduling: bool = True,
    include_regularization: bool = True
) -> Dict[str, Any]:
    """
    Create advanced optimizer-focused sweep configuration.
    
    Args:
        phenomenon_type: Type of phenomenon ('grokking', 'simplicity_bias', etc.)
        focus_optimizers: List of optimizers to focus on, or None for all
        include_scheduling: Whether to include learning rate scheduling
        include_regularization: Whether to include advanced regularization techniques
        
    Returns:
        Advanced sweep configuration
    """
    
    if focus_optimizers is None:
        focus_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop', 'lion', 'adagrad']
    
    base_params = {
        'optimizer': {'values': focus_optimizers},
        'learning_rate': {'min': 1e-5, 'max': 1e-1},
        'batch_size': {'values': [32, 64, 128, 256, 512]},
        'weight_decay': {'min': 1e-6, 'max': 1e-1}
    }
    
    # Optimizer-specific parameters
    if 'adam' in focus_optimizers or 'adamw' in focus_optimizers:
        base_params.update({
            'beta1': {'min': 0.8, 'max': 0.99},
            'beta2': {'min': 0.9, 'max': 0.9999},
            'epsilon': {'values': [1e-8, 1e-7, 1e-6]}
        })
    
    if 'sgd' in focus_optimizers:
        base_params.update({
            'momentum': {'min': 0.0, 'max': 0.99},
            'nesterov': {'values': [True, False]}
        })
    
    if 'lion' in focus_optimizers:
        base_params.update({
            'lion_beta1': {'min': 0.9, 'max': 0.99},
            'lion_beta2': {'min': 0.99, 'max': 0.999}
        })
    
    # Learning rate scheduling
    if include_scheduling:
        base_params.update({
            'lr_scheduler': {'values': ['cosine', 'step', 'exponential', 'plateau', 'cyclic', 'onecycle']},
            'warmup_steps': {'values': [0, 100, 500, 1000, 2000]},
            'cosine_restarts': {'values': [True, False]},
            'step_size': {'values': [10, 50, 100, 200]},
            'gamma': {'min': 0.1, 'max': 0.9}
        })
    
    # Advanced regularization techniques
    if include_regularization:
        base_params.update({
            'gradient_clip': {'values': [0.0, 0.5, 1.0, 5.0, 10.0]},
            'use_ema': {'values': [True, False]},
            'ema_decay': {'values': [0.99, 0.999, 0.9999]},
            'label_smoothing': {'min': 0.0, 'max': 0.2},
            'dropout': {'min': 0.0, 'max': 0.5},
            'stochastic_weight_averaging': {'values': [True, False]},
            'sam_rho': {'min': 0.0, 'max': 0.1}  # Sharpness-Aware Minimization
        })
    
    # Phenomenon-specific metrics
    if phenomenon_type == 'grokking':
        metric = {'name': 'grokking_success_rate', 'goal': 'maximize'}
    elif phenomenon_type == 'simplicity_bias':
        metric = {'name': 'worst_group_acc', 'goal': 'maximize'}
    elif phenomenon_type == 'phase_transitions':
        metric = {'name': 'emerged_abilities_count', 'goal': 'maximize'}
    else:
        metric = {'name': 'best_test_acc', 'goal': 'maximize'}
    
    return {
        'method': 'bayes',
        'parameters': base_params,
        'metric': metric,
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 100,
            'eta': 2
        }
    }


def create_multi_seed_sweep(
    base_config: Dict[str, Any],
    num_seeds: int = 5,
    seed_range: tuple = (0, 10000)
) -> Dict[str, Any]:
    """
    Create a sweep configuration that includes multiple random seeds for statistical robustness.
    
    Args:
        base_config: Base sweep configuration
        num_seeds: Number of different seeds to test for each parameter combination
        seed_range: Range for random seed generation
        
    Returns:
        Enhanced sweep configuration with multiple seeds
    """
    
    enhanced_config = base_config.copy()
    
    # Add seed parameter
    seeds = [np.random.randint(seed_range[0], seed_range[1]) for _ in range(num_seeds)]
    enhanced_config['parameters']['seed'] = {'values': seeds}
    
    # Add meta-analysis tracking
    enhanced_config['parameters']['enable_meta_analysis'] = {'value': True}
    
    return enhanced_config