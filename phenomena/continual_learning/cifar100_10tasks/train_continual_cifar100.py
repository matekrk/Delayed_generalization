#!/usr/bin/env python3
"""
Continual Learning on CIFAR-100: 10 Tasks with 10 Classes Each

This implements a continual learning regime that studies delayed generalization
in the context of sequential task learning. CIFAR-100's 100 classes are divided
into 10 tasks with 10 classes each, allowing us to study:

1. Catastrophic forgetting patterns
2. Delayed generalization across tasks
3. Knowledge transfer and interference
4. Phase transitions in continual learning

Usage:
    python train_continual_cifar100.py --data_dir ./data --save_dir ./results/continual
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict
import copy

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: WandB integration not available")
    WANDB_AVAILABLE = False


# CIFAR-100 class groupings for continual learning tasks
CIFAR100_TASK_CLASSES = {
    0: [4, 30, 55, 72, 95, 1, 32, 67, 73, 91],  # Aquatic mammals, Fish
    1: [54, 62, 70, 82, 92, 9, 10, 16, 28, 61],  # Flowers
    2: [0, 51, 53, 57, 83, 22, 39, 40, 86, 87],  # Food containers
    3: [5, 20, 25, 84, 94, 6, 7, 14, 18, 24],   # Fruit and vegetables
    4: [3, 42, 43, 88, 97, 12, 17, 37, 68, 76], # Household electrical devices
    5: [23, 33, 49, 60, 71, 15, 19, 21, 31, 38], # Household furniture
    6: [34, 63, 64, 66, 75, 26, 45, 77, 79, 99], # Insects
    7: [2, 11, 35, 46, 98, 27, 29, 44, 78, 93],  # Large carnivores, omnivores
    8: [36, 50, 65, 74, 80, 47, 52, 56, 59, 96], # Large man-made outdoor things
    9: [8, 13, 48, 58, 90, 41, 69, 81, 85, 89]   # People, reptiles, small mammals
}


class ContinualCIFAR100Model(nn.Module):
    """Model for continual learning with optional task-specific heads"""
    
    def __init__(self, num_classes: int = 100, num_tasks: int = 10, use_task_heads: bool = False):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.use_task_heads = use_task_heads
        self.classes_per_task = num_classes // num_tasks
        
        # Shared feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.5),
        )
        
        # Feature dimension
        self.feature_dim = 256 * 4 * 4
        
        if use_task_heads:
            # Separate heads for each task
            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, self.classes_per_task)
                ) for _ in range(num_tasks)
            ])
        else:
            # Single shared head
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x, task_id: Optional[int] = None):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        if self.use_task_heads and task_id is not None:
            return self.task_heads[task_id](features)
        else:
            return self.classifier(features)
    
    def get_features(self, x):
        """Extract features for analysis"""
        with torch.no_grad():
            features = self.features(x)
            return features.view(features.size(0), -1)


class EWCRegularizer:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, model: nn.Module, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self.fisher_information = {}
        self.optimal_weights = {}
        
    def compute_fisher_information(self, data_loader: DataLoader, device: torch.device):
        """Compute Fisher Information Matrix for current task"""
        
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        # Compute Fisher information
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
        
        # Normalize by dataset size
        for name in fisher_info:
            fisher_info[name] /= len(data_loader.dataset)
        
        self.fisher_information = fisher_info
        
        # Store optimal weights
        for name, param in self.model.named_parameters():
            self.optimal_weights[name] = param.data.clone()
    
    def get_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        
        if not self.fisher_information:
            return torch.tensor(0.0)
        
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_weights[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.importance * ewc_loss


class ContinualLearningTrainer:
    """Trainer for continual learning experiments on CIFAR-100"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        regularization_method: str = 'none',
        regularization_strength: float = 1000.0,
        wandb_logger: Optional[object] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.regularization_method = regularization_method
        self.wandb_logger = wandb_logger
        
        # Initialize regularizer
        if regularization_method == 'ewc':
            self.regularizer = EWCRegularizer(model, regularization_strength)
        else:
            self.regularizer = None
        
        # Tracking variables
        self.task_accuracies = defaultdict(list)  # Task -> [accuracies over time]
        self.all_task_accuracies = []  # Average accuracy on all seen tasks
        self.forgetting_measures = defaultdict(list)
        self.transfer_measures = []
        
        # Task-specific tracking
        self.completed_tasks = 0
        self.task_histories = {}
        
    def create_task_data_loaders(self, task_id: int, data_dir: str, batch_size: int = 128):
        """Create data loaders for a specific task"""
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load full CIFAR-100 dataset
        full_train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        full_test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        # Get classes for this task
        task_classes = CIFAR100_TASK_CLASSES[task_id]
        
        # Filter dataset for task classes
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label in task_classes]
        
        task_train_dataset = Subset(full_train_dataset, train_indices)
        task_test_dataset = Subset(full_test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            task_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            task_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, test_loader, task_classes
    
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 50,
        log_interval: int = 10
    ):
        """Train on a single task"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING TASK {task_id + 1}/10")
        print(f"Classes: {CIFAR100_TASK_CLASSES[task_id]}")
        print(f"{'='*60}")
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        task_train_losses = []
        task_train_accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Map global labels to task-local labels if using task heads
                if self.model.use_task_heads:
                    task_classes = CIFAR100_TASK_CLASSES[task_id]
                    local_target = torch.tensor([task_classes.index(t.item()) for t in target]).to(self.device)
                    output = self.model(data, task_id)
                    ce_loss = criterion(output, local_target)
                else:
                    output = self.model(data)
                    ce_loss = criterion(output, target)
                
                # Add regularization loss
                reg_loss = 0
                if self.regularizer is not None:
                    reg_loss = self.regularizer.get_ewc_loss()
                
                total_loss_val = ce_loss + reg_loss
                
                optimizer.zero_grad()
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
                pred = output.argmax(dim=1, keepdim=True)
                
                if self.model.use_task_heads:
                    correct += pred.eq(local_target.view_as(pred)).sum().item()
                else:
                    correct += pred.eq(target.view_as(pred)).sum().item()
                
                total += target.size(0)
            
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            task_train_losses.append(avg_loss)
            task_train_accuracies.append(accuracy)
            
            if (epoch + 1) % log_interval == 0:
                print(f'  Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        # Store task history
        self.task_histories[task_id] = {
            'train_losses': task_train_losses,
            'train_accuracies': task_train_accuracies
        }
        
        print(f"Task {task_id + 1} training completed. Final accuracy: {accuracy:.2f}%")
    
    def evaluate_all_tasks(self, data_dir: str, batch_size: int = 128) -> Dict[int, float]:
        """Evaluate model on all completed tasks"""
        
        task_accuracies = {}
        
        for task_id in range(self.completed_tasks + 1):
            _, test_loader, task_classes = self.create_task_data_loaders(
                task_id, data_dir, batch_size
            )
            
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.model.use_task_heads:
                        local_target = torch.tensor([task_classes.index(t.item()) for t in target]).to(self.device)
                        output = self.model(data, task_id)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(local_target.view_as(pred)).sum().item()
                    else:
                        output = self.model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    
                    total += target.size(0)
            
            accuracy = 100. * correct / total
            task_accuracies[task_id] = accuracy
            
            # Update task accuracy history
            self.task_accuracies[task_id].append(accuracy)
        
        return task_accuracies
    
    def compute_continual_learning_metrics(self, current_task_accuracies: Dict[int, float]) -> Dict[str, float]:
        """Compute continual learning specific metrics"""
        
        metrics = {}
        
        # Average accuracy on all seen tasks
        if current_task_accuracies:
            avg_accuracy = np.mean(list(current_task_accuracies.values()))
            metrics['average_accuracy'] = avg_accuracy
            self.all_task_accuracies.append(avg_accuracy)
        
        # Forgetting measure for each task (except current)
        forgetting_values = []
        for task_id in range(self.completed_tasks):
            if len(self.task_accuracies[task_id]) > 1:
                # Best accuracy achieved on this task
                best_acc = max(self.task_accuracies[task_id])
                current_acc = self.task_accuracies[task_id][-1]
                forgetting = best_acc - current_acc
                forgetting_values.append(forgetting)
                self.forgetting_measures[task_id].append(forgetting)
        
        if forgetting_values:
            metrics['average_forgetting'] = np.mean(forgetting_values)
            metrics['max_forgetting'] = np.max(forgetting_values)
        
        # Forward transfer (if not first task)
        if self.completed_tasks > 0:
            # Compare first epoch accuracy on current task to random baseline
            current_task_initial_acc = self.task_histories[self.completed_tasks]['train_accuracies'][0]
            random_baseline = 100.0 / 10  # 10 classes per task
            forward_transfer = current_task_initial_acc - random_baseline
            metrics['forward_transfer'] = forward_transfer
            self.transfer_measures.append(forward_transfer)
        
        # Learning progress (accuracy improvement on current task)
        current_task_id = self.completed_tasks
        if current_task_id in self.task_histories:
            initial_acc = self.task_histories[current_task_id]['train_accuracies'][0]
            final_acc = self.task_histories[current_task_id]['train_accuracies'][-1]
            learning_progress = final_acc - initial_acc
            metrics['learning_progress'] = learning_progress
        
        # Stability-plasticity analysis
        if len(self.all_task_accuracies) > 1:
            # Stability: how much average accuracy changes
            acc_changes = np.diff(self.all_task_accuracies)
            metrics['stability'] = 1.0 / (1.0 + np.std(acc_changes))
            
            # Plasticity: ability to learn new tasks
            if self.transfer_measures:
                metrics['plasticity'] = np.mean(self.transfer_measures)
        
        return metrics
    
    def detect_delayed_generalization(self, task_id: int) -> Dict[str, Any]:
        """Detect delayed generalization patterns in continual learning"""
        
        detection_results = {}
        
        # Check for sudden improvements in old tasks
        for old_task_id in range(task_id):
            if len(self.task_accuracies[old_task_id]) >= 2:
                recent_accs = self.task_accuracies[old_task_id][-2:]
                improvement = recent_accs[-1] - recent_accs[-2]
                
                if improvement > 5.0:  # 5% improvement threshold
                    detection_results[f'delayed_improvement_task_{old_task_id}'] = improvement
        
        # Check for knowledge transfer patterns
        if task_id > 0:
            current_performance = self.task_accuracies[task_id][-1]
            avg_previous_performance = np.mean([
                self.task_accuracies[tid][-1] for tid in range(task_id)
            ])
            
            if current_performance > avg_previous_performance + 10.0:  # 10% better
                detection_results['positive_transfer_detected'] = True
                detection_results['transfer_magnitude'] = current_performance - avg_previous_performance
        
        return detection_results
    
    def continual_learning_experiment(
        self,
        data_dir: str,
        epochs_per_task: int = 50,
        batch_size: int = 128,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete continual learning experiment"""
        
        print("Starting CIFAR-100 Continual Learning Experiment")
        print("10 Tasks × 10 Classes Each")
        print(f"Regularization: {self.regularization_method}")
        
        all_results = {
            'task_accuracies_over_time': {},
            'continual_learning_metrics': [],
            'delayed_generalization_events': {},
            'final_task_accuracies': {},
            'task_training_histories': {}
        }
        
        # Train each task sequentially
        for task_id in range(10):
            # Create data loaders for current task
            train_loader, test_loader, task_classes = self.create_task_data_loaders(
                task_id, data_dir, batch_size
            )
            
            # Train on current task
            self.train_task(task_id, train_loader, epochs_per_task)
            
            # Update completed tasks counter
            self.completed_tasks = task_id + 1
            
            # Compute Fisher Information for EWC (after training current task)
            if self.regularizer is not None and hasattr(self.regularizer, 'compute_fisher_information'):
                print(f"Computing Fisher Information for task {task_id + 1}...")
                self.regularizer.compute_fisher_information(train_loader, self.device)
            
            # Evaluate on all seen tasks
            current_task_accuracies = self.evaluate_all_tasks(data_dir, batch_size)
            all_results['task_accuracies_over_time'][task_id] = current_task_accuracies.copy()
            
            # Compute continual learning metrics
            cl_metrics = self.compute_continual_learning_metrics(current_task_accuracies)
            all_results['continual_learning_metrics'].append(cl_metrics)
            
            # Detect delayed generalization
            delayed_gen_results = self.detect_delayed_generalization(task_id)
            if delayed_gen_results:
                all_results['delayed_generalization_events'][task_id] = delayed_gen_results
            
            # Logging
            print(f"\nAfter Task {task_id + 1}:")
            for tid, acc in current_task_accuracies.items():
                print(f"  Task {tid + 1} Accuracy: {acc:.2f}%")
            print(f"  Average Accuracy: {cl_metrics.get('average_accuracy', 0):.2f}%")
            if 'average_forgetting' in cl_metrics:
                print(f"  Average Forgetting: {cl_metrics['average_forgetting']:.2f}%")
            
            # WandB logging
            if self.wandb_logger:
                log_metrics = {
                    'completed_tasks': task_id + 1,
                    **{f'task_{tid}_accuracy': acc for tid, acc in current_task_accuracies.items()},
                    **{f'cl_{key}': value for key, value in cl_metrics.items()}
                }
                
                # Log delayed generalization events
                for event_key, event_value in delayed_gen_results.items():
                    log_metrics[f'delayed_gen_{event_key}'] = event_value
                
                self.wandb_logger.log_epoch_metrics(
                    epoch=task_id,
                    train_loss=0,  # Will be filled from task history
                    test_loss=0,   # Will be filled from task history  
                    train_acc=cl_metrics.get('average_accuracy', 0),
                    test_acc=cl_metrics.get('average_accuracy', 0),
                    **log_metrics
                )
        
        # Final results
        all_results['final_task_accuracies'] = current_task_accuracies
        all_results['task_training_histories'] = self.task_histories
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save detailed results
            with open(os.path.join(save_dir, 'continual_learning_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Create visualization
            self.plot_continual_learning_results(all_results, save_dir)
        
        return all_results
    
    def plot_continual_learning_results(self, results: Dict[str, Any], save_dir: str):
        """Create comprehensive plots for continual learning results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continual Learning Results: CIFAR-100 (10 Tasks)', fontsize=16)
        
        # Plot 1: Task accuracies over time
        for task_id in range(10):
            task_accs = [results['task_accuracies_over_time'][completed_task].get(task_id, 0) 
                        for completed_task in range(task_id, 10)]
            if task_accs:
                axes[0, 0].plot(range(task_id + 1, 11), task_accs, 
                               label=f'Task {task_id + 1}', marker='o', alpha=0.7)
        
        axes[0, 0].set_xlabel('Tasks Completed')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Task Accuracies Over Time')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average accuracy and forgetting
        avg_accs = [metrics.get('average_accuracy', 0) for metrics in results['continual_learning_metrics']]
        avg_forgetting = [metrics.get('average_forgetting', 0) for metrics in results['continual_learning_metrics']]
        
        ax2_twin = axes[0, 1].twinx()
        
        line1 = axes[0, 1].plot(range(1, 11), avg_accs, 'b-', marker='o', label='Avg Accuracy')
        line2 = ax2_twin.plot(range(2, 11), avg_forgetting, 'r-', marker='s', label='Avg Forgetting', alpha=0.7)
        
        axes[0, 1].set_xlabel('Tasks Completed')
        axes[0, 1].set_ylabel('Average Accuracy (%)', color='b')
        ax2_twin.set_ylabel('Average Forgetting (%)', color='r')
        axes[0, 1].set_title('Average Performance and Forgetting')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0, 1].legend(lines, labels, loc='center right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Forward transfer
        forward_transfers = [metrics.get('forward_transfer', 0) for metrics in results['continual_learning_metrics']]
        if any(ft != 0 for ft in forward_transfers):
            axes[1, 0].plot(range(2, 11), forward_transfers[1:], 'g-', marker='d', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Task Number')
            axes[1, 0].set_ylabel('Forward Transfer (%)')
            axes[1, 0].set_title('Forward Transfer (Initial Performance - Random)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final task accuracies
        final_accs = list(results['final_task_accuracies'].values())
        task_names = [f'Task {i+1}' for i in range(len(final_accs))]
        
        bars = axes[1, 1].bar(task_names, final_accs, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1, 1].set_ylabel('Final Accuracy (%)')
        axes[1, 1].set_title('Final Accuracy on All Tasks')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'continual_learning_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Continual Learning (10 Tasks)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store CIFAR-100 data')
    parser.add_argument('--save_dir', type=str, default='./results/continual_learning',
                       help='Directory to save results')
    parser.add_argument('--epochs_per_task', type=int, default=50,
                       help='Number of epochs to train each task')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--regularization', type=str, default='none',
                       choices=['none', 'ewc', 'l2'],
                       help='Regularization method for continual learning')
    parser.add_argument('--regularization_strength', type=float, default=1000.0,
                       help='Strength of regularization')
    parser.add_argument('--use_task_heads', action='store_true',
                       help='Use separate heads for each task')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='WandB run name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                       help='WandB tags')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    model = ContinualCIFAR100Model(
        num_classes=100,
        num_tasks=10,
        use_task_heads=args.use_task_heads
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using {'task-specific' if args.use_task_heads else 'shared'} heads")
    
    # Setup WandB if requested
    wandb_logger = None
    if args.wandb_project and WANDB_AVAILABLE:
        config = {
            'epochs_per_task': args.epochs_per_task,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'regularization': args.regularization,
            'regularization_strength': args.regularization_strength,
            'use_task_heads': args.use_task_heads,
            'seed': args.seed,
            'num_tasks': 10,
            'classes_per_task': 10,
            'dataset': 'CIFAR-100'
        }
        
        wandb_logger = DelayedGeneralizationLogger(
            project_name=args.wandb_project,
            experiment_name=args.wandb_name or f"continual_cifar100_{args.regularization}_{args.seed}",
            config=config,
            phenomenon_type='continual_learning',
            tags=(args.wandb_tags or []) + ['continual_learning', 'cifar100', '10tasks'],
            notes="CIFAR-100 continual learning experiment studying delayed generalization"
        )
        
        print(f"WandB logging enabled: {args.wandb_project}/{wandb_logger.run.name}")
    
    # Create trainer
    trainer = ContinualLearningTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regularization_method=args.regularization,
        regularization_strength=args.regularization_strength,
        wandb_logger=wandb_logger
    )
    
    # Run continual learning experiment
    print("\n" + "="*60)
    print("STARTING CIFAR-100 CONTINUAL LEARNING EXPERIMENT")
    print("="*60)
    
    results = trainer.continual_learning_experiment(
        data_dir=args.data_dir,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Final summary
    print("\n" + "="*60)
    print("CONTINUAL LEARNING EXPERIMENT COMPLETED")
    print("="*60)
    
    final_accs = results['final_task_accuracies']
    avg_final_acc = np.mean(list(final_accs.values()))
    
    print(f"Final Average Accuracy: {avg_final_acc:.2f}%")
    print(f"Final Accuracy per Task:")
    for task_id, acc in final_accs.items():
        print(f"  Task {task_id + 1}: {acc:.2f}%")
    
    # Forgetting analysis
    final_metrics = results['continual_learning_metrics'][-1]
    if 'average_forgetting' in final_metrics:
        print(f"Average Forgetting: {final_metrics['average_forgetting']:.2f}%")
    
    # Delayed generalization events
    delayed_events = results['delayed_generalization_events']
    if delayed_events:
        print(f"Delayed Generalization Events Detected: {len(delayed_events)}")
        for task_id, events in delayed_events.items():
            print(f"  After Task {task_id + 1}: {events}")
    
    print(f"Results saved to: {args.save_dir}")
    
    # Final WandB logging
    if wandb_logger:
        wandb_logger.save_experiment_summary({
            'final_average_accuracy': avg_final_acc,
            'final_task_accuracies': final_accs,
            'total_delayed_generalization_events': len(delayed_events),
            **final_metrics
        })


if __name__ == '__main__':
    main()