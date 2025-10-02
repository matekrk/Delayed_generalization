#!/usr/bin/env python3
"""
Phase Transition Attributor for Delayed Generalization Research

Identifies which training examples most influence phase transitions
from memorization to generalization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle


class PhaseTransitionAttributor:
    """
    Attribute phase transitions to specific training examples.
    
    Uses gradient-based methods to identify which examples contribute
    most to sudden improvements in generalization.
    """
    
    def __init__(
        self,
        model_checkpoints: Dict[int, nn.Module],
        train_data,
        test_data,
        device: torch.device = None
    ):
        """
        Initialize phase transition attributor.
        
        Args:
            model_checkpoints: Dict mapping epoch -> model state
            train_data: Training dataset
            test_data: Test dataset
            device: Device for computations
        """
        self.model_checkpoints = model_checkpoints
        self.train_data = train_data
        self.test_data = test_data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.attribution_cache = {}
    
    def compute_transition_attributions(
        self,
        pre_transition_epoch: int,
        post_transition_epoch: int,
        method: str = 'gradient_similarity'
    ) -> Dict[int, float]:
        """
        Compute attribution scores for training examples across transition.
        
        Args:
            pre_transition_epoch: Epoch before transition
            post_transition_epoch: Epoch after transition
            method: Attribution method to use
            
        Returns:
            Dictionary mapping train example index -> attribution score
        """
        if pre_transition_epoch not in self.model_checkpoints:
            raise ValueError(f"No checkpoint for epoch {pre_transition_epoch}")
        if post_transition_epoch not in self.model_checkpoints:
            raise ValueError(f"No checkpoint for epoch {post_transition_epoch}")
        
        model_pre = self.model_checkpoints[pre_transition_epoch]
        model_post = self.model_checkpoints[post_transition_epoch]
        
        if method == 'gradient_similarity':
            return self._compute_gradient_similarity_attributions(model_pre, model_post)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
    
    def _compute_gradient_similarity_attributions(
        self,
        model_pre: nn.Module,
        model_post: nn.Module
    ) -> Dict[int, float]:
        """
        Compute attributions based on gradient similarity.
        
        Examples that have similar gradients before and after transition
        are likely important for the transition.
        """
        attributions = {}
        
        # Simple implementation: compute gradient magnitude change
        # More sophisticated: use influence functions or TRAK
        
        model_pre.eval()
        model_post.eval()
        
        for idx in range(len(self.train_data)):
            # Get example
            x, y = self.train_data[idx]
            x = x.unsqueeze(0).to(self.device)
            y = torch.tensor([y]).to(self.device)
            
            # Compute loss and gradients for pre-transition model
            model_pre.zero_grad()
            output_pre = model_pre(x)
            loss_pre = nn.functional.cross_entropy(output_pre, y)
            loss_pre.backward()
            
            grad_pre = torch.cat([p.grad.flatten() for p in model_pre.parameters() 
                                 if p.grad is not None])
            
            # Compute for post-transition model
            model_post.zero_grad()
            output_post = model_post(x)
            loss_post = nn.functional.cross_entropy(output_post, y)
            loss_post.backward()
            
            grad_post = torch.cat([p.grad.flatten() for p in model_post.parameters()
                                  if p.grad is not None])
            
            # Compute similarity (cosine similarity)
            similarity = nn.functional.cosine_similarity(
                grad_pre.unsqueeze(0), 
                grad_post.unsqueeze(0)
            ).item()
            
            # Attribution score: examples with consistent gradients are important
            attributions[idx] = similarity
        
        return attributions
    
    def find_critical_examples(
        self,
        attributions: Dict[int, float],
        top_k: int = 100,
        method: str = 'highest'
    ) -> List[int]:
        """
        Find the most critical examples for the transition.
        
        Args:
            attributions: Attribution scores
            top_k: Number of examples to return
            method: 'highest' or 'lowest' attribution scores
            
        Returns:
            List of example indices
        """
        sorted_items = sorted(attributions.items(), 
                            key=lambda x: x[1],
                            reverse=(method == 'highest'))
        
        return [idx for idx, _ in sorted_items[:top_k]]
    
    def analyze_critical_examples(
        self,
        critical_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze properties of critical examples.
        
        Args:
            critical_indices: Indices of critical examples
            
        Returns:
            Analysis dictionary
        """
        # Extract labels
        labels = [self.train_data[idx][1] for idx in critical_indices]
        
        # Compute label distribution
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        analysis = {
            'num_examples': len(critical_indices),
            'label_distribution': label_counts,
            'most_common_label': max(label_counts.items(), key=lambda x: x[1])[0] 
                                 if label_counts else None,
            'label_diversity': len(label_counts) / len(set(labels)) if labels else 0.0
        }
        
        return analysis
    
    def save(self, filepath: str):
        """Save attributor state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Note: Cannot save model checkpoints directly, only attribution results
        with open(filepath, 'wb') as f:
            pickle.dump({
                'attribution_cache': self.attribution_cache
            }, f)
