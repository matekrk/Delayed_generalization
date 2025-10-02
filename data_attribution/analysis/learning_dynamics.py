#!/usr/bin/env python3
"""
Learning Dynamics Analyzer for Delayed Generalization Research

Tracks and analyzes example-level learning dynamics including:
- Prediction variance over time
- Confidence evolution
- Example difficulty scores
- Forgetting events
- Learning transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle


class LearningDynamicsAnalyzer:
    """
    Comprehensive analyzer for example-level learning dynamics.
    
    Tracks how individual examples are learned over training, detecting
    patterns relevant to delayed generalization phenomena.
    """
    
    def __init__(
        self,
        num_classes: int,
        track_examples: int = 1000,
        device: torch.device = None
    ):
        """
        Initialize learning dynamics analyzer.
        
        Args:
            num_classes: Number of classes in the dataset
            track_examples: Maximum number of examples to track in detail
            device: Device to run computations on
        """
        self.num_classes = num_classes
        self.track_examples = track_examples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Storage for dynamics
        self.example_dynamics = defaultdict(lambda: {
            'losses': [],
            'predictions': [],
            'confidences': [],
            'correct': [],
            'logits_history': [],
            'difficulty_scores': [],
            'forgetting_events': 0,
            'first_learned_epoch': None,
            'last_forgotten_epoch': None
        })
        
        # Aggregate statistics
        self.epoch_stats = []
        self.class_dynamics = {i: {'accuracy': [], 'avg_confidence': [], 'avg_loss': []} 
                               for i in range(num_classes)}
        
        self.current_epoch = 0
    
    def track_batch(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_indices: torch.Tensor,
        epoch: int,
        criterion: Optional[nn.Module] = None
    ):
        """
        Track learning dynamics for a batch of examples.
        
        Args:
            model: PyTorch model
            inputs: Input batch tensor
            targets: Target labels
            batch_indices: Global indices of examples in the dataset
            epoch: Current epoch number
            criterion: Loss criterion (defaults to CrossEntropyLoss)
        """
        self.current_epoch = epoch
        model.eval()
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs if outputs.dim() == 2 else outputs.logits
            
            # Compute metrics
            losses = criterion(logits, targets)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            correct = (predictions == targets).cpu().numpy()
            
            # Track each example
            for i, idx in enumerate(batch_indices.cpu().numpy()):
                if idx >= self.track_examples:
                    continue
                
                dynamics = self.example_dynamics[idx]
                
                # Store current state
                dynamics['losses'].append(losses[i].item())
                dynamics['predictions'].append(predictions[i].item())
                dynamics['confidences'].append(confidences[i].item())
                dynamics['correct'].append(bool(correct[i]))
                dynamics['logits_history'].append(logits[i].cpu().numpy())
                
                # Compute difficulty score (higher = more difficult)
                # Based on loss magnitude and prediction consistency
                if len(dynamics['losses']) > 1:
                    loss_variance = np.var(dynamics['losses'][-min(10, len(dynamics['losses'])):])
                    prediction_changes = sum(
                        1 for j in range(len(dynamics['predictions'])-1) 
                        if dynamics['predictions'][j] != dynamics['predictions'][j+1]
                    )
                    difficulty = dynamics['losses'][-1] + 0.1 * loss_variance + 0.05 * prediction_changes
                else:
                    difficulty = dynamics['losses'][-1]
                
                dynamics['difficulty_scores'].append(difficulty)
                
                # Track forgetting events
                if len(dynamics['correct']) > 1:
                    if dynamics['correct'][-2] and not dynamics['correct'][-1]:
                        dynamics['forgetting_events'] += 1
                        dynamics['last_forgotten_epoch'] = epoch
                    elif not dynamics['correct'][-2] and dynamics['correct'][-1]:
                        if dynamics['first_learned_epoch'] is None:
                            dynamics['first_learned_epoch'] = epoch
    
    def compute_epoch_statistics(self, dataloader, model: nn.Module, epoch: int):
        """
        Compute aggregate statistics for an epoch.
        
        Args:
            dataloader: DataLoader for the dataset
            model: PyTorch model
            epoch: Current epoch number
        """
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_correct = {i: 0 for i in range(self.num_classes)}
        class_total = {i: 0 for i in range(self.num_classes)}
        class_confidences = {i: [] for i in range(self.num_classes)}
        class_losses = {i: [] for i in range(self.num_classes)}
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                logits = outputs if outputs.dim() == 2 else outputs.logits
                
                losses = criterion(logits, targets)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                total_loss += losses.sum().item()
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
                # Per-class statistics
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    class_losses[label].append(losses[i].item())
                    class_confidences[label].append(confidences[i].item())
                    if predictions[i] == targets[i]:
                        class_correct[label] += 1
        
        # Store epoch statistics
        epoch_stat = {
            'epoch': epoch,
            'avg_loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'class_accuracies': {k: class_correct[k] / max(class_total[k], 1) 
                                for k in range(self.num_classes)},
            'class_avg_confidences': {k: np.mean(class_confidences[k]) if class_confidences[k] else 0.0
                                      for k in range(self.num_classes)},
            'class_avg_losses': {k: np.mean(class_losses[k]) if class_losses[k] else 0.0
                                for k in range(self.num_classes)}
        }
        
        self.epoch_stats.append(epoch_stat)
        
        # Update class dynamics
        for cls in range(self.num_classes):
            self.class_dynamics[cls]['accuracy'].append(epoch_stat['class_accuracies'][cls])
            self.class_dynamics[cls]['avg_confidence'].append(epoch_stat['class_avg_confidences'][cls])
            self.class_dynamics[cls]['avg_loss'].append(epoch_stat['class_avg_losses'][cls])
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """
        Analyze learning patterns across tracked examples.
        
        Returns:
            Dictionary containing:
            - Easy/medium/hard example categories
            - Forgetting statistics
            - Learning transition points
            - Confidence evolution patterns
        """
        if not self.example_dynamics:
            return {}
        
        # Categorize examples by difficulty
        example_difficulties = {}
        forgetting_counts = []
        learning_speeds = []
        final_confidences = []
        
        for idx, dynamics in self.example_dynamics.items():
            if not dynamics['difficulty_scores']:
                continue
            
            avg_difficulty = np.mean(dynamics['difficulty_scores'])
            example_difficulties[idx] = avg_difficulty
            forgetting_counts.append(dynamics['forgetting_events'])
            
            if dynamics['first_learned_epoch'] is not None:
                learning_speeds.append(dynamics['first_learned_epoch'])
            
            if dynamics['confidences']:
                final_confidences.append(dynamics['confidences'][-1])
        
        # Sort by difficulty
        sorted_examples = sorted(example_difficulties.items(), key=lambda x: x[1])
        n_examples = len(sorted_examples)
        
        easy_threshold = n_examples // 3
        hard_threshold = 2 * n_examples // 3
        
        analysis = {
            'easy_examples': [idx for idx, _ in sorted_examples[:easy_threshold]],
            'medium_examples': [idx for idx, _ in sorted_examples[easy_threshold:hard_threshold]],
            'hard_examples': [idx for idx, _ in sorted_examples[hard_threshold:]],
            'forgetting_statistics': {
                'mean': np.mean(forgetting_counts) if forgetting_counts else 0.0,
                'std': np.std(forgetting_counts) if forgetting_counts else 0.0,
                'max': max(forgetting_counts) if forgetting_counts else 0,
                'examples_with_forgetting': sum(1 for c in forgetting_counts if c > 0)
            },
            'learning_speed_statistics': {
                'mean_first_learned_epoch': np.mean(learning_speeds) if learning_speeds else None,
                'std_first_learned_epoch': np.std(learning_speeds) if learning_speeds else None,
                'distribution': np.histogram(learning_speeds, bins=10)[0].tolist() if learning_speeds else []
            },
            'confidence_statistics': {
                'mean_final_confidence': np.mean(final_confidences) if final_confidences else 0.0,
                'std_final_confidence': np.std(final_confidences) if final_confidences else 0.0
            },
            'total_examples_tracked': len(self.example_dynamics)
        }
        
        return analysis
    
    def plot_learning_dynamics(self, save_dir: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of learning dynamics.
        
        Args:
            save_dir: Directory to save plots (optional)
            
        Returns:
            Matplotlib figure
        """
        if not self.epoch_stats:
            raise ValueError("No epoch statistics available. Call compute_epoch_statistics first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = [stat['epoch'] for stat in self.epoch_stats]
        
        # 1. Overall accuracy and loss
        ax = axes[0, 0]
        ax2 = ax.twinx()
        ax.plot(epochs, [s['accuracy'] for s in self.epoch_stats], 
                'b-', label='Accuracy', linewidth=2)
        ax2.plot(epochs, [s['avg_loss'] for s in self.epoch_stats], 
                 'r-', label='Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy', color='b')
        ax2.set_ylabel('Loss', color='r')
        ax.set_title('Overall Learning Progress')
        ax.grid(True, alpha=0.3)
        
        # 2. Class-wise accuracy
        ax = axes[0, 1]
        for cls in range(min(self.num_classes, 10)):  # Plot up to 10 classes
            ax.plot(epochs, self.class_dynamics[cls]['accuracy'], 
                   label=f'Class {cls}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Class-wise Accuracy Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Confidence evolution
        ax = axes[0, 2]
        for cls in range(min(self.num_classes, 10)):
            ax.plot(epochs, self.class_dynamics[cls]['avg_confidence'], 
                   label=f'Class {cls}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Confidence Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Difficulty distribution over time
        ax = axes[1, 0]
        if self.example_dynamics:
            sample_examples = list(self.example_dynamics.keys())[:min(100, len(self.example_dynamics))]
            for idx in sample_examples:
                if self.example_dynamics[idx]['difficulty_scores']:
                    ax.plot(self.example_dynamics[idx]['difficulty_scores'], 
                           alpha=0.1, color='blue')
        ax.set_xlabel('Tracking Step')
        ax.set_ylabel('Difficulty Score')
        ax.set_title('Example Difficulty Evolution')
        ax.grid(True, alpha=0.3)
        
        # 5. Forgetting events
        ax = axes[1, 1]
        forgetting_counts = [d['forgetting_events'] for d in self.example_dynamics.values()]
        if forgetting_counts:
            ax.hist(forgetting_counts, bins=max(forgetting_counts) + 1 if forgetting_counts else 10, 
                   alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Forgetting Events')
        ax.set_ylabel('Number of Examples')
        ax.set_title('Distribution of Forgetting Events')
        ax.grid(True, alpha=0.3)
        
        # 6. Learning speed distribution
        ax = axes[1, 2]
        learning_epochs = [d['first_learned_epoch'] for d in self.example_dynamics.values() 
                          if d['first_learned_epoch'] is not None]
        if learning_epochs:
            ax.hist(learning_epochs, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Epoch First Learned')
        ax.set_ylabel('Number of Examples')
        ax.set_title('Learning Speed Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / 'learning_dynamics.png', 
                       dpi=150, bbox_inches='tight')
        
        return fig
    
    def save(self, filepath: str):
        """Save analyzer state to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'example_dynamics': dict(self.example_dynamics),
                'epoch_stats': self.epoch_stats,
                'class_dynamics': self.class_dynamics,
                'num_classes': self.num_classes,
                'current_epoch': self.current_epoch
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'LearningDynamicsAnalyzer':
        """Load analyzer state from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        analyzer = cls(num_classes=data['num_classes'])
        analyzer.example_dynamics = defaultdict(lambda: {
            'losses': [], 'predictions': [], 'confidences': [], 'correct': [],
            'logits_history': [], 'difficulty_scores': [], 'forgetting_events': 0,
            'first_learned_epoch': None, 'last_forgotten_epoch': None
        })
        analyzer.example_dynamics.update(data['example_dynamics'])
        analyzer.epoch_stats = data['epoch_stats']
        analyzer.class_dynamics = data['class_dynamics']
        analyzer.current_epoch = data['current_epoch']
        
        return analyzer
