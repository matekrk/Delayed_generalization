#!/usr/bin/env python3
"""
Gradient Tracking for Opposing Signals Analysis

This module implements comprehensive gradient tracking capabilities for analyzing
opposing gradient signals during neural network training.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class GradientTracker:
    """
    Comprehensive gradient tracker for opposing signals analysis.
    
    This class monitors gradient directions, magnitudes, and patterns throughout
    training to identify examples with opposing signals that may impact delayed
    generalization phenomena.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        track_individual_examples: bool = True,
        memory_efficient: bool = False,
        max_history_size: int = 1000
    ):
        """
        Initialize gradient tracker.
        
        Args:
            model: PyTorch model to track
            track_individual_examples: Whether to track individual example gradients
            memory_efficient: Use memory-efficient tracking (aggregated statistics only)
            max_history_size: Maximum number of gradients to keep in memory
        """
        self.model = model
        self.track_individual_examples = track_individual_examples
        self.memory_efficient = memory_efficient
        self.max_history_size = max_history_size
        
        # Gradient storage
        self.gradient_history = defaultdict(list)  # epoch -> [gradients]
        self.loss_history = defaultdict(list)      # epoch -> [losses]
        
        # Individual example tracking
        if track_individual_examples and not memory_efficient:
            self.example_gradients = defaultdict(lambda: defaultdict(list))  # example_id -> epoch -> gradient
            self.example_losses = defaultdict(lambda: defaultdict(float))    # example_id -> epoch -> loss
        else:
            self.example_gradients = None
            self.example_losses = None
        
        # Opposing signal detection
        self.opposing_examples = defaultdict(set)  # epoch -> {example_ids}
        self.gradient_statistics = defaultdict(dict)  # epoch -> stats
        
        # Configuration
        self.opposing_threshold = -0.1  # Cosine similarity threshold for opposing signals
        self.loss_change_threshold = 0.5  # Threshold for significant loss changes
        
        # Memory management
        self.gradient_buffer = deque(maxlen=max_history_size)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Initialized GradientTracker for model with {total_params:,} parameters")
    
    def track_batch(
        self,
        epoch: int,
        batch_idx: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        individual_losses: torch.Tensor,
        example_ids: Optional[List[int]] = None
    ):
        """
        Track gradients for a single batch.
        
        Args:
            epoch: Current epoch number
            batch_idx: Batch index within epoch
            inputs: Input tensor
            targets: Target tensor
            individual_losses: Per-example losses (no reduction)
            example_ids: Optional example IDs for tracking
        """
        
        if example_ids is None:
            example_ids = list(range(batch_idx * inputs.size(0), (batch_idx + 1) * inputs.size(0)))
        
        batch_gradients = []
        
        # Compute per-example gradients
        for i, (example_id, loss) in enumerate(zip(example_ids, individual_losses)):
            # Zero gradients
            self.model.zero_grad()
            
            # Compute gradient for this example
            loss.backward(retain_graph=True)
            
            # Collect gradients
            example_grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    example_grad.append(param.grad.clone().flatten())
            
            if example_grad:
                grad_vector = torch.cat(example_grad)
                
                if self.memory_efficient:
                    # Store only essential statistics
                    grad_norm = grad_vector.norm().item()
                    self.gradient_buffer.append({
                        'epoch': epoch,
                        'example_id': example_id,
                        'grad_norm': grad_norm,
                        'loss': loss.item()
                    })
                else:
                    batch_gradients.append(grad_vector)
                    
                    # Store individual example data if enabled
                    if self.track_individual_examples and self.example_gradients is not None:
                        self.example_gradients[example_id][epoch] = grad_vector.cpu().numpy()
                        self.example_losses[example_id][epoch] = loss.item()
        
        # Store batch-level data (if not memory efficient)
        if not self.memory_efficient and batch_gradients:
            self.gradient_history[epoch].extend([g.cpu().numpy() for g in batch_gradients])
            self.loss_history[epoch].extend(individual_losses.cpu().numpy().tolist())
    
    def detect_opposing_signals(
        self, 
        epoch: int, 
        similarity_threshold: Optional[float] = None
    ) -> List[int]:
        """
        Detect examples with opposing gradient signals for a given epoch.
        
        Args:
            epoch: Epoch to analyze
            similarity_threshold: Cosine similarity threshold (default: self.opposing_threshold)
            
        Returns:
            List of indices of examples with opposing signals
        """
        
        if similarity_threshold is None:
            similarity_threshold = self.opposing_threshold
        
        if self.memory_efficient:
            return self._detect_opposing_memory_efficient(epoch, similarity_threshold)
        
        if epoch not in self.gradient_history or len(self.gradient_history[epoch]) == 0:
            return []
        
        gradients = np.array(self.gradient_history[epoch])
        
        # Compute mean gradient direction
        mean_gradient = np.mean(gradients, axis=0)
        mean_gradient_norm = np.linalg.norm(mean_gradient)
        
        if mean_gradient_norm < 1e-8:
            return []  # No meaningful gradient direction
        
        mean_gradient = mean_gradient / mean_gradient_norm
        
        # Find opposing examples
        opposing_indices = []
        similarities = []
        
        for i, grad in enumerate(gradients):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                continue
                
            grad_normalized = grad / grad_norm
            similarity = np.dot(mean_gradient, grad_normalized)
            similarities.append(similarity)
            
            if similarity < similarity_threshold:
                opposing_indices.append(i)
        
        # Store results
        self.opposing_examples[epoch] = set(opposing_indices)
        
        # Store statistics
        self.gradient_statistics[epoch] = {
            'mean_similarity': np.mean(similarities) if similarities else 0,
            'std_similarity': np.std(similarities) if similarities else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'max_similarity': np.max(similarities) if similarities else 0,
            'opposing_count': len(opposing_indices),
            'total_examples': len(gradients)
        }
        
        return opposing_indices
    
    def _detect_opposing_memory_efficient(self, epoch: int, similarity_threshold: float) -> List[int]:
        """Memory-efficient opposing signal detection using buffered data."""
        
        # Filter buffer for current epoch
        epoch_data = [item for item in self.gradient_buffer if item['epoch'] == epoch]
        
        if len(epoch_data) < 2:
            return []
        
        # Use gradient norms and loss patterns as proxy for opposing signals
        grad_norms = [item['grad_norm'] for item in epoch_data]
        losses = [item['loss'] for item in epoch_data]
        
        # Simple heuristic: examples with very different gradient norms and high losses
        # are likely opposing signals
        mean_grad_norm = np.mean(grad_norms)
        mean_loss = np.mean(losses)
        
        opposing_indices = []
        for i, item in enumerate(epoch_data):
            # Check if gradient norm and loss are outliers
            grad_outlier = abs(item['grad_norm'] - mean_grad_norm) > 2 * np.std(grad_norms)
            loss_outlier = item['loss'] > mean_loss + np.std(losses)
            
            if grad_outlier and loss_outlier:
                opposing_indices.append(i)
        
        return opposing_indices
    
    def analyze_loss_dynamics(self, window_size: int = 10) -> Dict[str, List[int]]:
        """
        Analyze examples with significant loss changes over time.
        
        Args:
            window_size: Size of window for computing loss trends
            
        Returns:
            Dictionary categorizing examples by loss dynamics
        """
        
        if self.example_losses is None:
            self.logger.warning("Individual example tracking not enabled")\n            return {}\n        \n        results = {\n            'increasing_loss': [],\n            'decreasing_loss': [],\n            'volatile_loss': [],\n            'stable_loss': []\n        }\n        \n        for example_id, loss_dict in self.example_losses.items():\n            epochs = sorted(loss_dict.keys())\n            if len(epochs) < window_size:\n                continue\n            \n            losses = [loss_dict[epoch] for epoch in epochs]\n            \n            # Compute loss statistics\n            recent_avg = np.mean(losses[-window_size:])\n            early_avg = np.mean(losses[:window_size])\n            overall_std = np.std(losses)\n            overall_mean = np.mean(losses)\n            \n            # Compute change metrics\n            change_ratio = (recent_avg - early_avg) / (early_avg + 1e-8)\n            volatility_ratio = overall_std / (overall_mean + 1e-8)\n            \n            # Categorize based on patterns\n            if change_ratio > self.loss_change_threshold:\n                results['increasing_loss'].append(example_id)\n            elif change_ratio < -self.loss_change_threshold:\n                results['decreasing_loss'].append(example_id)\n            elif volatility_ratio > 1.0:  # High volatility\n                results['volatile_loss'].append(example_id)\n            else:\n                results['stable_loss'].append(example_id)\n        \n        return results\n    \n    def compute_gradient_similarity_matrix(self, epoch: int) -> Optional[np.ndarray]:\n        """Compute pairwise gradient similarity matrix for an epoch."""\n        \n        if self.memory_efficient or epoch not in self.gradient_history:\n            return None\n        \n        gradients = np.array(self.gradient_history[epoch])\n        if len(gradients) == 0:\n            return None\n        \n        # Normalize gradients\n        norms = np.linalg.norm(gradients, axis=1, keepdims=True)\n        normalized_gradients = gradients / (norms + 1e-8)\n        \n        # Compute similarity matrix\n        similarity_matrix = np.dot(normalized_gradients, normalized_gradients.T)\n        \n        return similarity_matrix\n    \n    def get_gradient_flow_statistics(self, start_epoch: int, end_epoch: int) -> Dict[str, Any]:\n        """Get statistics about gradient flow over a range of epochs."""\n        \n        stats = {\n            'epoch_range': (start_epoch, end_epoch),\n            'opposing_signals_timeline': [],\n            'gradient_norm_evolution': [],\n            'loss_distribution_evolution': [],\n            'similarity_statistics': []\n        }\n        \n        for epoch in range(start_epoch, end_epoch + 1):\n            # Opposing signals count\n            opposing_count = len(self.opposing_examples.get(epoch, set()))\n            stats['opposing_signals_timeline'].append(opposing_count)\n            \n            # Gradient statistics\n            if epoch in self.gradient_history:\n                gradients = self.gradient_history[epoch]\n                if gradients:\n                    grad_norms = [np.linalg.norm(grad) for grad in gradients]\n                    stats['gradient_norm_evolution'].append({\n                        'epoch': epoch,\n                        'mean_norm': np.mean(grad_norms),\n                        'std_norm': np.std(grad_norms),\n                        'max_norm': np.max(grad_norms)\n                    })\n            \n            # Loss statistics\n            if epoch in self.loss_history:\n                losses = self.loss_history[epoch]\n                if losses:\n                    stats['loss_distribution_evolution'].append({\n                        'epoch': epoch,\n                        'mean_loss': np.mean(losses),\n                        'std_loss': np.std(losses),\n                        'max_loss': np.max(losses),\n                        'min_loss': np.min(losses)\n                    })\n            \n            # Similarity statistics\n            if epoch in self.gradient_statistics:\n                stats['similarity_statistics'].append(self.gradient_statistics[epoch])\n        \n        return stats\n    \n    def get_memory_usage_info(self) -> Dict[str, Any]:\n        """Get information about memory usage."""\n        \n        info = {\n            'memory_efficient_mode': self.memory_efficient,\n            'track_individual_examples': self.track_individual_examples,\n            'max_history_size': self.max_history_size,\n            'current_buffer_size': len(self.gradient_buffer) if hasattr(self, 'gradient_buffer') else 0\n        }\n        \n        if not self.memory_efficient:\n            info.update({\n                'epochs_in_gradient_history': len(self.gradient_history),\n                'total_gradients_stored': sum(len(grads) for grads in self.gradient_history.values()),\n                'examples_tracked': len(self.example_gradients) if self.example_gradients else 0\n            })\n        \n        return info\n    \n    def clear_old_data(self, keep_last_n_epochs: int = 10):\n        """Clear old data to manage memory usage."""\n        \n        if self.memory_efficient:\n            return  # Buffer automatically manages size\n        \n        # Get epochs to keep\n        all_epochs = sorted(self.gradient_history.keys())\n        if len(all_epochs) <= keep_last_n_epochs:\n            return\n        \n        epochs_to_remove = all_epochs[:-keep_last_n_epochs]\n        \n        # Clear gradient history\n        for epoch in epochs_to_remove:\n            if epoch in self.gradient_history:\n                del self.gradient_history[epoch]\n            if epoch in self.loss_history:\n                del self.loss_history[epoch]\n            if epoch in self.opposing_examples:\n                del self.opposing_examples[epoch]\n            if epoch in self.gradient_statistics:\n                del self.gradient_statistics[epoch]\n        \n        # Clear individual example data\n        if self.example_gradients is not None:\n            for example_id in list(self.example_gradients.keys()):\n                epochs_to_clear = [e for e in self.example_gradients[example_id] if e in epochs_to_remove]\n                for epoch in epochs_to_clear:\n                    del self.example_gradients[example_id][epoch]\n                    if epoch in self.example_losses[example_id]:\n                        del self.example_losses[example_id][epoch]\n        \n        self.logger.info(f"Cleared data for {len(epochs_to_remove)} old epochs")\n    \n    def save_state(self, filepath: str):\n        """Save tracker state to file."""\n        import pickle\n        \n        state = {\n            'gradient_history': dict(self.gradient_history) if not self.memory_efficient else None,\n            'loss_history': dict(self.loss_history) if not self.memory_efficient else None,\n            'example_gradients': dict(self.example_gradients) if self.example_gradients else None,\n            'example_losses': dict(self.example_losses) if self.example_losses else None,\n            'opposing_examples': dict(self.opposing_examples),\n            'gradient_statistics': dict(self.gradient_statistics),\n            'gradient_buffer': list(self.gradient_buffer) if hasattr(self, 'gradient_buffer') else None,\n            'config': {\n                'track_individual_examples': self.track_individual_examples,\n                'memory_efficient': self.memory_efficient,\n                'max_history_size': self.max_history_size,\n                'opposing_threshold': self.opposing_threshold,\n                'loss_change_threshold': self.loss_change_threshold\n            }\n        }\n        \n        with open(filepath, 'wb') as f:\n            pickle.dump(state, f)\n        \n        self.logger.info(f"Saved tracker state to {filepath}")\n    \n    def load_state(self, filepath: str):\n        """Load tracker state from file."""\n        import pickle\n        \n        with open(filepath, 'rb') as f:\n            state = pickle.load(f)\n        \n        # Restore data\n        if state['gradient_history'] is not None:\n            self.gradient_history = defaultdict(list, state['gradient_history'])\n        if state['loss_history'] is not None:\n            self.loss_history = defaultdict(list, state['loss_history'])\n        if state['example_gradients'] is not None:\n            self.example_gradients = defaultdict(lambda: defaultdict(list), state['example_gradients'])\n        if state['example_losses'] is not None:\n            self.example_losses = defaultdict(lambda: defaultdict(float), state['example_losses'])\n        \n        self.opposing_examples = defaultdict(set, state['opposing_examples'])\n        self.gradient_statistics = defaultdict(dict, state['gradient_statistics'])\n        \n        if state['gradient_buffer'] is not None:\n            self.gradient_buffer = deque(state['gradient_buffer'], maxlen=self.max_history_size)\n        \n        # Restore config\n        config = state.get('config', {})\n        for key, value in config.items():\n            if hasattr(self, key):\n                setattr(self, key, value)\n        \n        self.logger.info(f"Loaded tracker state from {filepath}")\n    \n    def get_comprehensive_statistics(self) -> Dict[str, Any]:\n        """Get comprehensive statistics about all tracked data."""\n        \n        stats = {\n            'tracking_config': {\n                'memory_efficient': self.memory_efficient,\n                'track_individual_examples': self.track_individual_examples,\n                'opposing_threshold': self.opposing_threshold,\n                'loss_change_threshold': self.loss_change_threshold\n            },\n            'data_summary': {\n                'total_epochs_tracked': len(self.gradient_history) if not self.memory_efficient else len(set(item['epoch'] for item in self.gradient_buffer)),\n                'total_examples_tracked': len(self.example_losses) if self.example_losses else 0,\n                'memory_usage': self.get_memory_usage_info()\n            },\n            'opposing_signals_summary': {\n                'total_opposing_detections': sum(len(examples) for examples in self.opposing_examples.values()),\n                'epochs_with_opposing': len([epoch for epoch, examples in self.opposing_examples.items() if len(examples) > 0]),\n                'average_opposing_per_epoch': np.mean([len(examples) for examples in self.opposing_examples.values()]) if self.opposing_examples else 0\n            }\n        }\n        \n        # Add loss dynamics if available\n        if self.example_losses is not None:\n            loss_dynamics = self.analyze_loss_dynamics()\n            stats['loss_dynamics_summary'] = {\n                key: len(examples) for key, examples in loss_dynamics.items()\n            }\n        \n        return stats


if __name__ == "__main__":
    # Simple test\n    import torch\n    import torch.nn as nn\n    \n    # Create simple model and data\n    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))\n    tracker = GradientTracker(model)\n    \n    # Simulate some tracking\n    inputs = torch.randn(32, 10)\n    targets = torch.randn(32, 1)\n    outputs = model(inputs)\n    losses = nn.MSELoss(reduction='none')(outputs.squeeze(), targets.squeeze())\n    \n    # Track gradients\n    tracker.track_batch(0, 0, inputs, targets, losses)\n    \n    # Detect opposing signals\n    opposing = tracker.detect_opposing_signals(0)\n    \n    print(f"Tracked {len(losses)} examples")\n    print(f"Detected {len(opposing)} opposing signals")\n    print("GradientTracker test completed!")