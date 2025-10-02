#!/usr/bin/env python3
"""
Bias Attributor for Delayed Generalization Research

Identifies which training examples promote spurious correlations
and bias learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle


class BiasAttributor:
    """
    Attribute bias learning to specific training examples.
    
    Identifies which examples promote spurious features vs. core features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        biased_data,
        unbiased_data,
        device: torch.device = None
    ):
        """
        Initialize bias attributor.
        
        Args:
            model: PyTorch model to analyze
            biased_data: Training data with spurious correlations
            unbiased_data: Test data without spurious correlations
            device: Device for computations
        """
        self.model = model
        self.biased_data = biased_data
        self.unbiased_data = unbiased_data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.attribution_results = {}
    
    def find_spurious_examples(
        self,
        attribution_method: str = 'gradient_based',
        top_k: int = 200
    ) -> List[int]:
        """
        Find training examples that most promote spurious correlations.
        
        Args:
            attribution_method: Method to use ('gradient_based' or 'loss_based')
            top_k: Number of examples to return
            
        Returns:
            List of example indices that promote spurious features
        """
        self.model.eval()
        
        if attribution_method == 'gradient_based':
            return self._find_spurious_via_gradients(top_k)
        elif attribution_method == 'loss_based':
            return self._find_spurious_via_loss(top_k)
        else:
            raise ValueError(f"Unknown method: {attribution_method}")
    
    def _find_spurious_via_gradients(self, top_k: int) -> List[int]:
        """
        Find spurious examples by analyzing gradient alignment with biased features.
        """
        spurious_scores = {}
        
        # Evaluate each training example
        for idx in range(len(self.biased_data)):
            x, y = self.biased_data[idx]
            x = x.unsqueeze(0).to(self.device)
            y = torch.tensor([y]).to(self.device)
            
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Compute gradient norm (examples with large gradients are influential)
            grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() 
                           if p.grad is not None)
            
            # Check if model is confident and correct (likely using spurious feature)
            with torch.no_grad():
                prob = F.softmax(output, dim=1)
                confidence = prob.max().item()
                correct = (output.argmax() == y).item()
            
            # Score: high confidence + correct + strong gradient = likely spurious
            if correct and confidence > 0.9:
                spurious_scores[idx] = grad_norm * confidence
            else:
                spurious_scores[idx] = 0.0
        
        # Return top-k
        sorted_items = sorted(spurious_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_items[:top_k]]
    
    def _find_spurious_via_loss(self, top_k: int) -> List[int]:
        """
        Find spurious examples by comparing loss on biased vs unbiased data.
        """
        spurious_scores = {}
        
        self.model.eval()
        
        with torch.no_grad():
            # Compute model performance on biased training data
            for idx in range(len(self.biased_data)):
                x, y = self.biased_data[idx]
                x = x.unsqueeze(0).to(self.device)
                y = torch.tensor([y]).to(self.device)
                
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                prob = F.softmax(output, dim=1)
                confidence = prob.max().item()
                
                # Examples with low loss but high confidence are likely using spurious features
                spurious_scores[idx] = confidence / (loss.item() + 1e-6)
        
        sorted_items = sorted(spurious_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_items[:top_k]]
    
    def analyze_bias_development(
        self,
        model_checkpoints: Dict[int, nn.Module]
    ) -> Dict[str, Any]:
        """
        Analyze how bias develops over training checkpoints.
        
        Args:
            model_checkpoints: Dict mapping epoch -> model state
            
        Returns:
            Timeline of bias metrics
        """
        bias_timeline = {
            'epochs': [],
            'biased_accuracy': [],
            'unbiased_accuracy': [],
            'bias_gap': []
        }
        
        for epoch in sorted(model_checkpoints.keys()):
            model = model_checkpoints[epoch]
            model.eval()
            
            # Evaluate on biased data
            biased_correct = 0
            biased_total = 0
            
            with torch.no_grad():
                for x, y in self.biased_data:
                    x = x.unsqueeze(0).to(self.device)
                    y = torch.tensor([y]).to(self.device)
                    
                    output = model(x)
                    pred = output.argmax(dim=1)
                    biased_correct += (pred == y).item()
                    biased_total += 1
            
            # Evaluate on unbiased data
            unbiased_correct = 0
            unbiased_total = 0
            
            with torch.no_grad():
                for x, y in self.unbiased_data:
                    x = x.unsqueeze(0).to(self.device)
                    y = torch.tensor([y]).to(self.device)
                    
                    output = model(x)
                    pred = output.argmax(dim=1)
                    unbiased_correct += (pred == y).item()
                    unbiased_total += 1
            
            biased_acc = biased_correct / biased_total if biased_total > 0 else 0.0
            unbiased_acc = unbiased_correct / unbiased_total if unbiased_total > 0 else 0.0
            
            bias_timeline['epochs'].append(epoch)
            bias_timeline['biased_accuracy'].append(biased_acc)
            bias_timeline['unbiased_accuracy'].append(unbiased_acc)
            bias_timeline['bias_gap'].append(biased_acc - unbiased_acc)
        
        self.attribution_results['bias_timeline'] = bias_timeline
        return bias_timeline
    
    def compute_bias_strength(self) -> float:
        """
        Compute overall bias strength of current model.
        
        Returns:
            Bias strength score (higher = more biased)
        """
        self.model.eval()
        
        # Compare performance on biased vs unbiased data
        with torch.no_grad():
            biased_losses = []
            for x, y in self.biased_data:
                x = x.unsqueeze(0).to(self.device)
                y = torch.tensor([y]).to(self.device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                biased_losses.append(loss.item())
            
            unbiased_losses = []
            for x, y in self.unbiased_data:
                x = x.unsqueeze(0).to(self.device)
                y = torch.tensor([y]).to(self.device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                unbiased_losses.append(loss.item())
        
        # Bias strength: difference in average loss
        bias_strength = np.mean(unbiased_losses) - np.mean(biased_losses)
        return float(bias_strength)
    
    def save(self, filepath: str):
        """Save attributor state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'attribution_results': self.attribution_results
            }, f)
