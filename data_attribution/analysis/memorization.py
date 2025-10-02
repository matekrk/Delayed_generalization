#!/usr/bin/env python3
"""
Memorization Detector for Delayed Generalization Research

Detects and analyzes memorization patterns:
- Identifies memorized vs. generalized examples
- Tracks memorization score over training
- Analyzes atypical example learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class MemorizationDetector:
    """
    Detect memorization patterns in neural network training.
    
    Useful for understanding when models transition from pure
    memorization to true generalization.
    """
    
    def __init__(
        self,
        num_classes: int,
        device: torch.device = None
    ):
        """
        Initialize memorization detector.
        
        Args:
            num_classes: Number of classes
            device: Device for computations
        """
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track example-level metrics
        self.example_scores = defaultdict(lambda: {
            'train_correct': [],
            'test_correct': [],  # For examples also in test set
            'confidence': [],
            'loss': [],
            'memorization_score': []
        })
        
        self.epoch_memorization = []
    
    def compute_memorization_scores(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Compute memorization scores for current epoch.
        
        High memorization = good train performance but poor test performance
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            epoch: Current epoch
            
        Returns:
            Dictionary with memorization metrics
        """
        model.eval()
        
        # Evaluate on train set
        train_correct = 0
        train_total = 0
        train_confidence = []
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                logits = outputs if outputs.dim() == 2 else outputs.logits
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                train_confidence.extend(confidences.cpu().numpy())
        
        train_acc = train_correct / train_total
        train_conf = np.mean(train_confidence)
        
        # Evaluate on test set
        test_correct = 0
        test_total = 0
        test_confidence = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                logits = outputs if outputs.dim() == 2 else outputs.logits
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)
                test_confidence.extend(confidences.cpu().numpy())
        
        test_acc = test_correct / test_total
        test_conf = np.mean(test_confidence)
        
        # Compute memorization score
        # High score = high train acc but low test acc (memorization)
        # Low score = similar train and test acc (generalization)
        memorization_score = train_acc - test_acc
        confidence_gap = train_conf - test_conf
        
        metrics = {
            'epoch': epoch,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_confidence': train_conf,
            'test_confidence': test_conf,
            'memorization_score': memorization_score,
            'confidence_gap': confidence_gap,
            'generalization_gap': memorization_score  # Alias
        }
        
        self.epoch_memorization.append(metrics)
        
        return metrics
    
    def identify_memorized_examples(
        self,
        model: nn.Module,
        train_loader,
        threshold: float = 0.8
    ) -> Tuple[List[int], List[int]]:
        """
        Identify likely memorized vs. generalized examples.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            threshold: Confidence threshold for memorization
            
        Returns:
            (memorized_indices, generalized_indices)
        """
        model.eval()
        
        example_metrics = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                logits = outputs if outputs.dim() == 2 else outputs.logits
                
                losses = F.cross_entropy(logits, targets, reduction='none')
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                correct = (preds == targets)
                
                for i in range(inputs.size(0)):
                    idx = batch_idx * train_loader.batch_size + i
                    
                    # Memorized: high confidence but high loss variation
                    # or correct but with unusual patterns
                    metric = {
                        'idx': idx,
                        'confidence': confidences[i].item(),
                        'loss': losses[i].item(),
                        'correct': correct[i].item()
                    }
                    example_metrics.append(metric)
        
        # Simple heuristic: memorized = high confidence + correct
        # In practice, would need more sophisticated analysis
        memorized = [m['idx'] for m in example_metrics 
                    if m['correct'] and m['confidence'] > threshold]
        generalized = [m['idx'] for m in example_metrics 
                      if m['correct'] and m['confidence'] <= threshold]
        
        return memorized, generalized
    
    def analyze_memorization_evolution(self) -> Dict[str, Any]:
        """
        Analyze how memorization changes over training.
        
        Returns:
            Dictionary with evolution metrics
        """
        if not self.epoch_memorization:
            return {}
        
        epochs = [m['epoch'] for m in self.epoch_memorization]
        mem_scores = [m['memorization_score'] for m in self.epoch_memorization]
        
        # Detect transition point (when memorization score decreases)
        transition_epoch = None
        if len(mem_scores) > 1:
            max_mem = max(mem_scores)
            max_idx = mem_scores.index(max_mem)
            
            # Look for sustained decrease after peak
            for i in range(max_idx + 1, len(mem_scores)):
                if mem_scores[i] < max_mem * 0.8:  # 20% reduction
                    transition_epoch = epochs[i]
                    break
        
        analysis = {
            'transition_epoch': transition_epoch,
            'max_memorization': max(mem_scores) if mem_scores else 0.0,
            'final_memorization': mem_scores[-1] if mem_scores else 0.0,
            'memorization_reduction': (max(mem_scores) - mem_scores[-1]) if mem_scores else 0.0,
            'epochs_tracked': len(epochs)
        }
        
        return analysis
    
    def plot_memorization(self, save_dir: Optional[str] = None) -> plt.Figure:
        """
        Visualize memorization patterns.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Matplotlib figure
        """
        if not self.epoch_memorization:
            raise ValueError("No memorization data available")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = [m['epoch'] for m in self.epoch_memorization]
        
        # 1. Train vs Test Accuracy
        ax = axes[0, 0]
        ax.plot(epochs, [m['train_accuracy'] for m in self.epoch_memorization],
               'b-', label='Train', linewidth=2)
        ax.plot(epochs, [m['test_accuracy'] for m in self.epoch_memorization],
               'r-', label='Test', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Train vs Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Memorization Score
        ax = axes[0, 1]
        mem_scores = [m['memorization_score'] for m in self.epoch_memorization]
        ax.plot(epochs, mem_scores, 'g-', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Memorization Score')
        ax.set_title('Memorization Score (Train Acc - Test Acc)')
        ax.grid(True, alpha=0.3)
        
        # 3. Confidence Gap
        ax = axes[1, 0]
        ax.plot(epochs, [m['train_confidence'] for m in self.epoch_memorization],
               'b-', label='Train', linewidth=2)
        ax.plot(epochs, [m['test_confidence'] for m in self.epoch_memorization],
               'r-', label='Test', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Confidence Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Generalization Gap
        ax = axes[1, 1]
        ax.plot(epochs, mem_scores, 'purple', linewidth=2)
        ax.fill_between(epochs, 0, mem_scores, 
                        where=np.array(mem_scores) > 0, 
                        alpha=0.3, color='red', label='Overfitting')
        ax.fill_between(epochs, 0, mem_scores,
                        where=np.array(mem_scores) <= 0,
                        alpha=0.3, color='green', label='Generalization')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Generalization Gap')
        ax.set_title('Overfitting vs Generalization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / 'memorization_analysis.png',
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def save(self, filepath: str):
        """Save detector state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'example_scores': dict(self.example_scores),
                'epoch_memorization': self.epoch_memorization,
                'num_classes': self.num_classes
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MemorizationDetector':
        """Load detector state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(num_classes=data['num_classes'])
        detector.example_scores = defaultdict(lambda: {
            'train_correct': [], 'test_correct': [], 'confidence': [],
            'loss': [], 'memorization_score': []
        })
        detector.example_scores.update(data['example_scores'])
        detector.epoch_memorization = data['epoch_memorization']
        
        return detector
