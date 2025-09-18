#!/usr/bin/env python3
"""
TRAK (Tracing with Randomly-projected After Kernels) Implementation
for Data Attribution in Delayed Generalization Research

Based on "TRAK: Attributing Model Behavior at Scale" by Park et al. (2023)
Adapted for studying delayed generalization phenomena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pickle


class TRAKAttributor:
    """
    TRAK implementation for efficient data attribution.
    
    TRAK computes influence scores by:
    1. Computing gradients for each training example
    2. Projecting gradients to lower dimension using random projections
    3. Computing kernel similarities between train/test gradients
    """
    
    def __init__(
        self,
        model: nn.Module,
        task: str = 'classification',
        proj_dim: int = 512,
        device: torch.device = None,
        seed: int = 42
    ):
        """
        Initialize TRAK attributor.
        
        Args:
            model: PyTorch model to analyze
            task: Task type ('classification' or 'regression')
            proj_dim: Dimension for random projection
            device: Device to run computations on
            seed: Random seed for reproducibility
        """
        self.model = model
        self.task = task
        self.proj_dim = proj_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize projection matrices
        self.projection_matrices = {}
        self._initialize_projections()
        
        # Storage for computed features
        self.train_features = None
        self.test_features = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_projections(self):
        """Initialize random projection matrices for each parameter group."""
        self.model.eval()
        
        # Get parameter dimensions
        param_dims = {}
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_dims[name] = param.numel()
                total_params += param.numel()
        
        self.logger.info(f"Total parameters: {total_params:,}")
        
        # Create projection matrix
        # Use random Gaussian projection (Johnson-Lindenstrauss)
        self.projection_matrix = torch.randn(
            self.proj_dim, total_params,
            device=self.device,
            dtype=torch.float32
        ) / np.sqrt(self.proj_dim)
        
        self.param_dims = param_dims
        self.total_params = total_params
    
    def _compute_gradient_vector(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute flattened gradient vector for the loss."""
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=self.model.parameters(),
            retain_graph=False,
            create_graph=False
        )
        
        # Flatten and concatenate gradients
        grad_vector = torch.cat([g.flatten() for g in gradients])
        return grad_vector
    
    def _project_gradient(self, grad_vector: torch.Tensor) -> torch.Tensor:
        """Project gradient vector to lower dimension."""
        return torch.matmul(self.projection_matrix, grad_vector)
    
    def compute_train_features(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute TRAK features for training data.
        
        Args:
            dataloader: DataLoader for training data
            num_samples: Number of samples to process (None for all)
            save_path: Path to save computed features
            
        Returns:
            Projected gradient features [num_samples, proj_dim]
        """
        self.model.eval()
        
        if num_samples is None:
            num_samples = len(dataloader.dataset)
        
        features = torch.zeros(num_samples, self.proj_dim, device=self.device)
        sample_idx = 0
        
        self.logger.info(f"Computing TRAK features for {num_samples} training samples")
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
            else:
                data, targets = batch['data'], batch['targets']
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            batch_size = data.size(0)
            
            # Check if we've processed enough samples
            if sample_idx >= num_samples:
                break
            
            # Process each example individually for gradient computation
            for i in range(batch_size):
                if sample_idx >= num_samples:
                    break
                
                # Single example
                x = data[i:i+1]
                y = targets[i:i+1]
                
                # Forward pass
                outputs = self.model(x)
                
                # Compute loss
                if self.task == 'classification':
                    loss = F.cross_entropy(outputs, y)
                else:
                    loss = F.mse_loss(outputs, y)
                
                # Compute and project gradient
                grad_vector = self._compute_gradient_vector(loss)
                projected_grad = self._project_gradient(grad_vector)
                
                features[sample_idx] = projected_grad.detach()
                sample_idx += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Processed batch {batch_idx}, samples: {sample_idx}/{num_samples}")
        
        # Trim to actual number of processed samples
        features = features[:sample_idx]
        
        # Save features if path provided
        if save_path:
            torch.save(features, save_path)
            self.logger.info(f"Saved train features to {save_path}")
        
        self.train_features = features
        return features
    
    def compute_test_features(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute TRAK features for test data.
        
        Args:
            dataloader: DataLoader for test data
            num_samples: Number of samples to process (None for all)
            save_path: Path to save computed features
            
        Returns:
            Projected gradient features [num_samples, proj_dim]
        """
        self.model.eval()
        
        if num_samples is None:
            num_samples = len(dataloader.dataset)
        
        features = torch.zeros(num_samples, self.proj_dim, device=self.device)
        sample_idx = 0
        
        self.logger.info(f"Computing TRAK features for {num_samples} test samples")
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
            else:
                data, targets = batch['data'], batch['targets']
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            batch_size = data.size(0)
            
            # Check if we've processed enough samples
            if sample_idx >= num_samples:
                break
            
            # Process each example individually for gradient computation
            for i in range(batch_size):
                if sample_idx >= num_samples:
                    break
                
                # Single example
                x = data[i:i+1]
                y = targets[i:i+1]
                
                # Forward pass
                outputs = self.model(x)
                
                # Compute loss
                if self.task == 'classification':
                    loss = F.cross_entropy(outputs, y)
                else:
                    loss = F.mse_loss(outputs, y)
                
                # Compute and project gradient
                grad_vector = self._compute_gradient_vector(loss)
                projected_grad = self._project_gradient(grad_vector)
                
                features[sample_idx] = projected_grad.detach()
                sample_idx += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Processed batch {batch_idx}, samples: {sample_idx}/{num_samples}")
        
        # Trim to actual number of processed samples
        features = features[:sample_idx]
        
        # Save features if path provided
        if save_path:
            torch.save(features, save_path)
            self.logger.info(f"Saved test features to {save_path}")
        
        self.test_features = features
        return features
    
    def compute_attributions(
        self,
        train_features: Optional[torch.Tensor] = None,
        test_features: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute attribution scores between test and train examples.
        
        Args:
            train_features: Training features [num_train, proj_dim]
            test_features: Test features [num_test, proj_dim]
            normalize: Whether to normalize features before computing similarities
            
        Returns:
            Attribution matrix [num_test, num_train]
        """
        if train_features is None:
            train_features = self.train_features
        if test_features is None:
            test_features = self.test_features
        
        if train_features is None or test_features is None:
            raise ValueError("Must compute or provide train and test features")
        
        self.logger.info(f"Computing attributions: {test_features.size(0)} test x {train_features.size(0)} train")
        
        if normalize:
            # L2 normalize features
            train_features = F.normalize(train_features, p=2, dim=1)
            test_features = F.normalize(test_features, p=2, dim=1)
        
        # Compute cosine similarity matrix
        # attributions[i, j] = similarity between test_i and train_j
        attributions = torch.matmul(test_features, train_features.T)
        
        return attributions
    
    def find_top_attributions(
        self,
        attributions: torch.Tensor,
        test_idx: int,
        top_k: int = 10,
        return_scores: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Find top-k training examples that most influence a test example.
        
        Args:
            attributions: Attribution matrix [num_test, num_train]
            test_idx: Index of test example to analyze
            top_k: Number of top training examples to return
            return_scores: Whether to return attribution scores
            
        Returns:
            Top-k training indices, optionally with scores
        """
        test_attributions = attributions[test_idx]
        top_scores, top_indices = torch.topk(test_attributions, top_k)
        
        if return_scores:
            return top_indices, top_scores
        else:
            return top_indices
    
    def analyze_attribution_patterns(
        self,
        attributions: torch.Tensor,
        train_labels: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze patterns in attributions (e.g., same-class bias).
        
        Args:
            attributions: Attribution matrix [num_test, num_train]
            train_labels: Training labels [num_train]
            test_labels: Test labels [num_test]
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Same-class attribution bias
        same_class_attrs = []
        diff_class_attrs = []
        
        for test_idx in range(attributions.size(0)):
            test_label = test_labels[test_idx]
            test_attrs = attributions[test_idx]
            
            # Split by class
            same_class_mask = (train_labels == test_label)
            diff_class_mask = ~same_class_mask
            
            if same_class_mask.sum() > 0:
                same_class_attrs.extend(test_attrs[same_class_mask].tolist())
            if diff_class_mask.sum() > 0:
                diff_class_attrs.extend(test_attrs[diff_class_mask].tolist())
        
        if same_class_attrs and diff_class_attrs:
            results['same_class_mean'] = np.mean(same_class_attrs)
            results['diff_class_mean'] = np.mean(diff_class_attrs)
            results['same_class_bias'] = results['same_class_mean'] - results['diff_class_mean']
        
        # Attribution concentration (how concentrated are high attributions?)
        top_k_concentration = []
        for test_idx in range(attributions.size(0)):
            test_attrs = attributions[test_idx]
            sorted_attrs, _ = torch.sort(test_attrs, descending=True)
            
            # Concentration: fraction of total attribution in top 10%
            top_k = max(1, len(sorted_attrs) // 10)
            top_sum = sorted_attrs[:top_k].sum()
            total_sum = sorted_attrs.sum()
            
            if total_sum > 0:
                concentration = (top_sum / total_sum).item()
                top_k_concentration.append(concentration)
        
        if top_k_concentration:
            results['attribution_concentration'] = np.mean(top_k_concentration)
        
        return results
    
    def save_state(self, save_path: str):
        """Save TRAK attributor state."""
        state = {
            'proj_dim': self.proj_dim,
            'task': self.task,
            'seed': self.seed,
            'projection_matrix': self.projection_matrix.cpu(),
            'param_dims': self.param_dims,
            'total_params': self.total_params,
            'train_features': self.train_features.cpu() if self.train_features is not None else None,
            'test_features': self.test_features.cpu() if self.test_features is not None else None
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Saved TRAK state to {save_path}")
    
    def load_state(self, load_path: str):
        """Load TRAK attributor state."""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.proj_dim = state['proj_dim']
        self.task = state['task']
        self.seed = state['seed']
        self.projection_matrix = state['projection_matrix'].to(self.device)
        self.param_dims = state['param_dims']
        self.total_params = state['total_params']
        
        if state['train_features'] is not None:
            self.train_features = state['train_features'].to(self.device)
        if state['test_features'] is not None:
            self.test_features = state['test_features'].to(self.device)
        
        self.logger.info(f"Loaded TRAK state from {load_path}")


class DelayedGeneralizationTRAK(TRAKAttributor):
    """
    Extended TRAK implementation for delayed generalization analysis.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_features = {}  # Store features across training epochs
        
    def compute_epoch_features(
        self,
        dataloader: DataLoader,
        epoch: int,
        split: str = 'train',
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Compute features for a specific training epoch."""
        if split == 'train':
            features = self.compute_train_features(dataloader, num_samples)
        else:
            features = self.compute_test_features(dataloader, num_samples)
        
        if epoch not in self.epoch_features:
            self.epoch_features[epoch] = {}
        
        self.epoch_features[epoch][split] = features
        return features
    
    def analyze_attribution_evolution(
        self,
        test_idx: int,
        epochs: List[int],
        top_k: int = 20
    ) -> Dict[int, Dict]:
        """
        Analyze how attributions for a test example evolve during training.
        
        Args:
            test_idx: Test example to analyze
            epochs: List of epochs to compare
            top_k: Number of top attributions to track
            
        Returns:
            Dictionary mapping epochs to attribution analysis
        """
        results = {}
        
        for epoch in epochs:
            if epoch not in self.epoch_features:
                continue
            
            epoch_data = self.epoch_features[epoch]
            if 'train' not in epoch_data or 'test' not in epoch_data:
                continue
            
            # Compute attributions for this epoch
            attributions = self.compute_attributions(
                train_features=epoch_data['train'],
                test_features=epoch_data['test']
            )
            
            # Get top attributions for test example
            top_indices, top_scores = self.find_top_attributions(
                attributions, test_idx, top_k, return_scores=True
            )
            
            results[epoch] = {
                'top_indices': top_indices.cpu().numpy(),
                'top_scores': top_scores.cpu().numpy(),
                'mean_attribution': attributions[test_idx].mean().item(),
                'max_attribution': attributions[test_idx].max().item(),
                'attribution_std': attributions[test_idx].std().item()
            }
        
        return results
    
    def find_phase_transition_examples(
        self,
        pre_transition_epoch: int,
        post_transition_epoch: int,
        top_k: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Find training examples whose influence changes most during phase transition.
        
        Args:
            pre_transition_epoch: Epoch before phase transition
            post_transition_epoch: Epoch after phase transition
            top_k: Number of examples to return
            
        Returns:
            Dictionary with examples that gained/lost influence
        """
        if (pre_transition_epoch not in self.epoch_features or 
            post_transition_epoch not in self.epoch_features):
            raise ValueError("Required epochs not available in stored features")
        
        # Get features for both epochs
        pre_train = self.epoch_features[pre_transition_epoch]['train']
        pre_test = self.epoch_features[pre_transition_epoch]['test']
        post_train = self.epoch_features[post_transition_epoch]['train']
        post_test = self.epoch_features[post_transition_epoch]['test']
        
        # Compute attributions for both epochs
        pre_attributions = self.compute_attributions(pre_train, pre_test)
        post_attributions = self.compute_attributions(post_train, post_test)
        
        # Compute attribution changes
        attribution_changes = post_attributions - pre_attributions
        
        # Find examples with largest changes (averaged across test examples)
        mean_changes = attribution_changes.mean(dim=0)  # Average across test examples
        
        # Top examples that gained influence
        top_gains = torch.topk(mean_changes, top_k)
        
        # Top examples that lost influence  
        top_losses = torch.topk(-mean_changes, top_k)
        
        return {
            'gained_influence_indices': top_gains.indices,
            'gained_influence_scores': top_gains.values,
            'lost_influence_indices': top_losses.indices,
            'lost_influence_scores': -top_losses.values
        }


# Utility functions for TRAK analysis

def compute_attribution_stability(
    trak_attributor: TRAKAttributor,
    dataloader: DataLoader,
    num_runs: int = 5,
    test_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute stability of TRAK attributions across multiple runs.
    
    Args:
        trak_attributor: TRAK attributor instance
        dataloader: DataLoader for computing features
        num_runs: Number of runs to average over
        test_indices: Specific test indices to analyze
        
    Returns:
        Stability metrics
    """
    if test_indices is None:
        test_indices = list(range(min(100, len(dataloader.dataset))))
    
    all_attributions = []
    
    for run in range(num_runs):
        # Re-initialize projections for each run
        trak_attributor._initialize_projections()
        
        # Compute features
        train_features = trak_attributor.compute_train_features(dataloader)
        test_features = trak_attributor.compute_test_features(dataloader)
        
        # Compute attributions
        attributions = trak_attributor.compute_attributions(train_features, test_features)
        all_attributions.append(attributions[test_indices])
    
    # Compute stability metrics
    attributions_tensor = torch.stack(all_attributions)  # [num_runs, num_test, num_train]
    
    # Pairwise correlations between runs
    correlations = []
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            for test_idx in range(len(test_indices)):
                corr = torch.corrcoef(torch.stack([
                    attributions_tensor[i, test_idx],
                    attributions_tensor[j, test_idx]
                ]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())
    
    # Top-k stability (how often are the same examples in top-k?)
    top_k_stability = []
    for test_idx in range(len(test_indices)):
        top_k_sets = []
        for run in range(num_runs):
            _, top_indices = torch.topk(attributions_tensor[run, test_idx], 10)
            top_k_sets.append(set(top_indices.tolist()))
        
        # Compute average pairwise Jaccard similarity
        jaccard_sims = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                intersection = len(top_k_sets[i].intersection(top_k_sets[j]))
                union = len(top_k_sets[i].union(top_k_sets[j]))
                jaccard_sims.append(intersection / union if union > 0 else 0)
        
        if jaccard_sims:
            top_k_stability.append(np.mean(jaccard_sims))
    
    return {
        'mean_correlation': np.mean(correlations) if correlations else 0,
        'std_correlation': np.std(correlations) if correlations else 0,
        'top_k_stability': np.mean(top_k_stability) if top_k_stability else 0,
        'num_runs': num_runs,
        'num_test_examples': len(test_indices)
    }