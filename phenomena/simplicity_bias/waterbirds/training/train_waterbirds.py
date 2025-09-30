#!/usr/bin/env python3
"""
Training Script for Waterbirds Simplicity Bias Research

This script trains models on the synthetic waterbirds dataset to study simplicity bias
and delayed generalization. It implements both standard ERM and Group DRO training.

Usage:
    python train_waterbirds.py --data_dir ./synthetic_waterbirds_data --method erm
    python train_waterbirds.py --data_dir ./synthetic_waterbirds_data --method group_dro
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import time

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.generate_synthetic_waterbirds import load_synthetic_waterbirds_dataset


class WaterbirdsModel(nn.Module):
    """ResNet-based model for waterbirds classification"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # Use ResNet-50 as backbone (standard for waterbirds)
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class GroupDROLoss(nn.Module):
    """Group Distributionally Robust Optimization Loss"""
    
    def __init__(self, num_groups: int, step_size: float = 0.01):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        self.group_weights = torch.ones(num_groups) / num_groups
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets, group_labels):
        losses = self.criterion(outputs, targets)
        
        # Compute group losses
        group_losses = torch.zeros(self.num_groups)
        group_counts = torch.zeros(self.num_groups)
        
        for g in range(self.num_groups):
            group_mask = (group_labels == g)
            if group_mask.sum() > 0:
                group_losses[g] = losses[group_mask].mean()
                group_counts[g] = group_mask.sum()
        
        # Update group weights (upweight groups with higher loss)
        with torch.no_grad():
            for g in range(self.num_groups):
                if group_counts[g] > 0:
                    self.group_weights[g] *= torch.exp(self.step_size * group_losses[g])
            
            # Normalize weights
            self.group_weights = self.group_weights / self.group_weights.sum()
        
        # Compute weighted loss
        weighted_loss = torch.sum(self.group_weights * group_losses)
        
        return weighted_loss, group_losses, self.group_weights


class WaterbirdsTrainer:
    """Trainer for waterbirds experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        method: str = "erm",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        group_dro_step_size: float = 0.01
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.method = method
        
        # Setup optimizer using enhanced optimizers
        try:
            # Try to use enhanced optimizer from our reorganized structure
            from optimization import get_default_optimizer
            self.optimizer = get_default_optimizer(
                model, 
                phenomenon_type='simplicity_bias',
                optimizer_type='enhanced_sgd',
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
            print("Using enhanced SGD optimizer for Waterbirds experiments")
        except ImportError:
            # Fallback to standard SGD
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
            print("Using standard SGD optimizer")
        
        # Setup loss function
        if method == "group_dro":
            self.criterion = GroupDROLoss(num_groups=4, step_size=group_dro_step_size)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.group_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            bird_types = batch['bird_types'].to(self.device)
            group_labels = batch['group_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            if self.method == "group_dro":
                loss, group_losses, group_weights = self.criterion(outputs, bird_types, group_labels)
            else:
                loss = self.criterion(outputs, bird_types)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(bird_types).sum().item()
            total += bird_types.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, name="Test"):
        """Evaluate on given data loader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Group-wise tracking
        group_correct = torch.zeros(4)
        group_total = torch.zeros(4)
        
        with torch.no_grad():
            for batch in loader:
                images = batch['images'].to(self.device)
                bird_types = batch['bird_types'].to(self.device)
                group_labels = batch['group_labels'].to(self.device)
                
                outputs = self.model(images)
                
                if self.method == "group_dro":
                    loss, _, _ = self.criterion(outputs, bird_types, group_labels)
                else:
                    loss = self.criterion(outputs, bird_types)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(bird_types).sum().item()
                total += bird_types.size(0)
                
                # Group-wise accuracy
                for g in range(4):
                    group_mask = (group_labels == g)
                    if group_mask.sum() > 0:
                        group_correct[g] += pred[group_mask].eq(bird_types[group_mask]).sum().item()
                        group_total[g] += group_mask.sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        # Group accuracies
        group_accs = {}
        group_names = ['landbird_water', 'landbird_land', 'waterbird_water', 'waterbird_land']
        for i, name_g in enumerate(group_names):
            if group_total[i] > 0:
                group_accs[name_g] = group_correct[i] / group_total[i]
            else:
                group_accs[name_g] = 0.0
        
        # Worst group accuracy (key metric for bias research)
        worst_group_acc = min(group_accs.values())
        
        return avg_loss, accuracy, group_accs, worst_group_acc
    
    def train(self, epochs: int, save_dir: str) -> Dict:
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training with method: {self.method}")
        print(f"Device: {self.device}")
        
        best_worst_group_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Evaluation
            test_loss, test_acc, test_group_accs, worst_group_acc = self.evaluate(self.test_loader, "Test")
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.group_accuracies.append(test_group_accs)
            
            # Save best model based on worst group accuracy
            if worst_group_acc > best_worst_group_acc:
                best_worst_group_acc = worst_group_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
                      f"Test Acc: {test_acc*100:.2f}%, Worst Group Acc: {worst_group_acc*100:.2f}%")
                for group, acc in test_group_accs.items():
                    print(f"  {group}: {acc*100:.2f}%")
        
        print(f"\nBest worst group accuracy: {best_worst_group_acc*100:.2f}% at epoch {best_epoch}")
        
        # Save final results
        results = {
            'method': self.method,
            'best_worst_group_acc': best_worst_group_acc,
            'best_epoch': best_epoch,
            'final_test_acc': self.test_accuracies[-1],
            'final_group_accs': self.group_accuracies[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'group_accuracies': self.group_accuracies
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self._plot_results(save_dir)
        
        return results
    
    def _plot_results(self, save_dir: str):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.test_losses, label='Test')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train')
        axes[0, 1].plot(self.test_accuracies, label='Test')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Group accuracies over time
        group_names = ['landbird_water', 'landbird_land', 'waterbird_water', 'waterbird_land']
        for group_name in group_names:
            group_acc_history = [ga[group_name] for ga in self.group_accuracies]
            axes[1, 0].plot(group_acc_history, label=group_name)
        
        axes[1, 0].set_title('Group Accuracies Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Final group accuracies
        final_accs = self.group_accuracies[-1]
        axes[1, 1].bar(group_names, [final_accs[g] for g in group_names])
        axes[1, 1].set_title('Final Group Accuracies')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 32, data_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders from saved dataset with optional data fraction"""
    # Load dataset
    train_dataset, test_dataset, metadata = load_synthetic_waterbirds_dataset(data_dir)
    
    original_train_size = len(train_dataset)
    original_test_size = len(test_dataset)
    
    # Apply data fraction if specified
    if data_fraction < 1.0:
        print(f"Using {data_fraction:.2%} of the dataset")
        
        # Sample fraction of training data
        train_size = int(original_train_size * data_fraction)
        train_indices = torch.randperm(original_train_size)[:train_size]
        train_dataset = [train_dataset[i] for i in train_indices]
        
        # Sample fraction of test data
        test_size = int(original_test_size * data_fraction)
        test_indices = torch.randperm(original_test_size)[:test_size]
        test_dataset = [test_dataset[i] for i in test_indices]
        
        print(f"Reduced train size: {len(train_dataset)} (from {original_train_size})")
        print(f"Reduced test size: {len(test_dataset)} (from {original_test_size})")
    
    print(f"Final train size: {len(train_dataset)}")
    print(f"Final test size: {len(test_dataset)}")
    
    # Convert to tensor datasets
    def dataset_to_tensors(dataset):
        return {
            'images': torch.stack([sample['image'] for sample in dataset]),
            'bird_types': torch.tensor([sample['bird_type'] for sample in dataset]),
            'background_types': torch.tensor([sample['background_type'] for sample in dataset]),
            'group_labels': torch.tensor([sample['group_label'] for sample in dataset])
        }
    
    train_data = dataset_to_tensors(train_dataset)
    test_data = dataset_to_tensors(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_data['images'], train_data['bird_types'], 
                      train_data['background_types'], train_data['group_labels']),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(test_data['images'], test_data['bird_types'],
                      test_data['background_types'], test_data['group_labels']),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Wrap to return dictionaries
    def collate_fn(batch):
        images, bird_types, background_types, group_labels = zip(*batch)
        return {
            'images': torch.stack(images),
            'bird_types': torch.stack(bird_types),
            'background_types': torch.stack(background_types),
            'group_labels': torch.stack(group_labels)
        }
    
    train_loader = DataLoader(
        TensorDataset(train_data['images'], train_data['bird_types'], 
                      train_data['background_types'], train_data['group_labels']),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        TensorDataset(test_data['images'], test_data['bird_types'],
                      test_data['background_types'], test_data['group_labels']),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train model on waterbirds dataset for simplicity bias research")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--method", type=str, default="erm", choices=["erm", "group_dro"],
                       help="Training method")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--group_dro_step_size", type=float, default=0.01, help="Group DRO step size")
    parser.add_argument("--save_dir", type=str, default="./waterbirds_results", help="Directory to save results")
    parser.add_argument("--results_dir", type=str, default=None, 
                       help="Alternative name for save_dir (same functionality)")
    parser.add_argument("--data_fraction", type=float, default=1.0, 
                       help="Fraction of dataset to use (0.0-1.0, default: 1.0 for full dataset)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Handle alternative results_dir argument
    if args.results_dir is not None:
        args.save_dir = args.results_dir
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Waterbirds Simplicity Bias Training")
    print("=" * 40)
    print(f"Method: {args.method}")
    print(f"Data directory: {args.data_dir}")
    print(f"Data fraction: {args.data_fraction:.2%}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Use wandb: {args.use_wandb}")
    if args.method == "group_dro":
        print(f"Group DRO step size: {args.group_dro_step_size}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.data_fraction)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = WaterbirdsModel(num_classes=2, pretrained=True)
    
    # Create trainer
    trainer = WaterbirdsTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        method=args.method,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        group_dro_step_size=args.group_dro_step_size
    )
    
    # Train model
    print("\nStarting training...")
    results = trainer.train(args.epochs, args.save_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"Best worst group accuracy: {results['best_worst_group_acc']*100:.2f}%")


if __name__ == "__main__":
    main()