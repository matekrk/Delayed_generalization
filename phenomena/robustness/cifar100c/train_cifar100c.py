#!/usr/bin/env python3
"""
Training Script for CIFAR-100-C Robustness Research

This script trains models on the CIFAR-100-C dataset to study 
robustness vs accuracy tradeoffs and delayed generalization patterns.
Extends the CIFAR-10-C approach to the more challenging 100-class setting.

Usage:
    python train_cifar100c.py --data_dir ./cifar100c_data --epochs 200
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: WandB integration not available")
    WANDB_AVAILABLE = False

from models.vision.cifar_robustness_models import CIFARModel, create_cifar_robustness_model


class CIFAR100CTrainer:
    """Trainer for CIFAR-100-C experiments with robustness analysis"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        corruption_test_loaders: Optional[Dict[str, DataLoader]] = None,
        wandb_logger: Optional[object] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.corruption_test_loaders = corruption_test_loaders or {}
        self.device = device
        self.wandb_logger = wandb_logger
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
        
        # Tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.corruption_accuracies = {}
        
        # Initialize corruption tracking
        for corruption_name in self.corruption_test_loaders.keys():
            self.corruption_accuracies[corruption_name] = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def test(self, test_loader: DataLoader = None):
        """Test the model"""
        if test_loader is None:
            test_loader = self.test_loader
            
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate_corruptions(self) -> Dict[str, float]:
        """Evaluate model on all corruption types"""
        corruption_results = {}
        
        for corruption_name, corruption_loader in self.corruption_test_loaders.items():
            _, accuracy = self.test(corruption_loader)
            corruption_results[corruption_name] = accuracy
            
            # Track over time
            if corruption_name not in self.corruption_accuracies:
                self.corruption_accuracies[corruption_name] = []
            self.corruption_accuracies[corruption_name].append(accuracy)
        
        return corruption_results
    
    def compute_robustness_metrics(self, clean_acc: float, corruption_results: Dict[str, float]) -> Dict[str, float]:
        """Compute various robustness metrics"""
        metrics = {}
        
        if corruption_results:
            corruption_accs = list(corruption_results.values())
            
            # Basic robustness metrics
            metrics['mean_corruption_acc'] = np.mean(corruption_accs)
            metrics['min_corruption_acc'] = np.min(corruption_accs)
            metrics['max_corruption_acc'] = np.max(corruption_accs)
            metrics['std_corruption_acc'] = np.std(corruption_accs)
            
            # Robustness gap
            metrics['robustness_gap'] = clean_acc - metrics['mean_corruption_acc']
            metrics['worst_case_gap'] = clean_acc - metrics['min_corruption_acc']
            
            # Relative robustness
            if clean_acc > 0:
                metrics['relative_robustness'] = metrics['mean_corruption_acc'] / clean_acc
                metrics['worst_case_relative_robustness'] = metrics['min_corruption_acc'] / clean_acc
            
            # Corruption consistency (1 - coefficient of variation)
            if metrics['mean_corruption_acc'] > 0:
                cv = metrics['std_corruption_acc'] / metrics['mean_corruption_acc']
                metrics['corruption_consistency'] = 1.0 / (1.0 + cv)
        
        return metrics
    
    def train(self, epochs: int = 200, log_interval: int = 10, save_dir: Optional[str] = None):
        """Main training loop"""
        
        print(f"Training CIFAR-100-C model for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_test_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Testing
            test_loss, test_acc = self.test()
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Evaluate corruptions every few epochs
            corruption_results = {}
            robustness_metrics = {}
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                corruption_results = self.evaluate_corruptions()
                robustness_metrics = self.compute_robustness_metrics(test_acc, corruption_results)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = self.model.state_dict().copy()
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
                print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')
                print(f'  LR: {self.scheduler.get_last_lr()[0]:.6f}')
                
                if corruption_results:
                    print(f'  Mean Corruption Acc: {robustness_metrics.get("mean_corruption_acc", 0)*100:.2f}%')
                    print(f'  Robustness Gap: {robustness_metrics.get("robustness_gap", 0)*100:.2f}%')
            
            # WandB logging
            if self.wandb_logger:
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                
                # Add corruption results
                for corruption_name, acc in corruption_results.items():
                    metrics[f'corruption_{corruption_name}_acc'] = acc
                
                # Add robustness metrics
                metrics.update(robustness_metrics)
                
                self.wandb_logger.log_epoch_metrics(**metrics)
        
        # Save final results
        results = {
            'final_train_acc': self.train_accuracies[-1],
            'final_test_acc': self.test_accuracies[-1],
            'best_test_acc': best_test_acc,
            'final_robustness_metrics': robustness_metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'test_losses': self.test_losses,
                'train_accuracies': self.train_accuracies,
                'test_accuracies': self.test_accuracies,
                'corruption_accuracies': self.corruption_accuracies
            }
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            
            # Save results
            with open(os.path.join(save_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save training curves
            self.plot_training_curves(save_dir)
        
        return results
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, self.test_losses, label='Test Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Test Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, label='Train Acc', alpha=0.8)
        axes[0, 1].plot(epochs, self.test_accuracies, label='Test Acc', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Test Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Corruption accuracies
        if self.corruption_accuracies:
            for corruption_name, accs in self.corruption_accuracies.items():
                if accs:  # Only plot if we have data
                    corruption_epochs = range(5, len(accs) * 5 + 1, 5)  # Every 5 epochs
                    axes[1, 0].plot(corruption_epochs, accs, label=corruption_name, alpha=0.8)
            
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Corruption Robustness Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Robustness gap over time
        if self.corruption_accuracies:
            robustness_gaps = []
            gap_epochs = []
            
            for i in range(0, len(self.test_accuracies), 5):
                if i // 5 < len(list(self.corruption_accuracies.values())[0]):
                    test_acc = self.test_accuracies[i]
                    corruption_accs = [accs[i // 5] for accs in self.corruption_accuracies.values() if i // 5 < len(accs)]
                    if corruption_accs:
                        mean_corruption_acc = np.mean(corruption_accs)
                        robustness_gaps.append(test_acc - mean_corruption_acc)
                        gap_epochs.append(i + 1)
            
            if robustness_gaps:
                axes[1, 1].plot(gap_epochs, robustness_gaps, 'r-', alpha=0.8, linewidth=2)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Robustness Gap (%)')
                axes[1, 1].set_title('Clean-Corruption Accuracy Gap')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_cifar100c_data_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 4):
    """Create CIFAR-100-C data loaders"""
    
    # Standard CIFAR-100 transforms
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
    
    # Load clean CIFAR-100
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Create CIFAR-100-C corruption loaders
    corruption_loaders = {}
    
    # Import CIFAR-100-C dataset classes
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from data.vision.cifar100c.generate_cifar100c import CIFAR100CDataset, create_cifar100c_datasets
        
        # Default corruptions to evaluate
        corruptions = ['gaussian_noise', 'motion_blur', 'snow', 'brightness', 'contrast']
        severities = [1, 3, 5]  # Light, medium, severe
        
        # Create corruption datasets
        corruption_datasets = create_cifar100c_datasets(
            corruptions=corruptions,
            severities=severities,
            data_dir=data_dir,
            train=False,  # Use test set for evaluation
            seed=42
        )
        
        # Create data loaders for each corruption
        for corruption_key, corruption_dataset in corruption_datasets.items():
            corruption_loaders[corruption_key] = DataLoader(
                corruption_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
        
        print(f"Created {len(corruption_loaders)} corruption test loaders")
        
    except ImportError as e:
        print(f"Warning: Could not import CIFAR-100-C corruption datasets: {e}")
        print("Using clean test set as fallback for corruption evaluation")
        corruption_loaders = {
            'clean': test_loader
        }
    except Exception as e:
        print(f"Warning: Error creating corruption loaders: {e}")
        print("Using clean test set as fallback for corruption evaluation")
        corruption_loaders = {
            'clean': test_loader
        }
    
    return train_loader, test_loader, corruption_loaders


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100-C Robustness Training')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100c',
                       help='Directory to store CIFAR-100 data')
    parser.add_argument('--save_dir', type=str, default='./results/cifar100c',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size variant')
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
    
    # Create data loaders
    print("Loading CIFAR-100-C data...")
    train_loader, test_loader, corruption_loaders = create_cifar100c_data_loaders(
        args.data_dir, args.batch_size
    )
    
    # Create model
    print(f"Creating {args.model_size} CIFAR-100-C model...")
    model = create_cifar_robustness_model(
        model_type='cifar100c',
        num_classes=100,
        model_size=args.model_size
    )
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Setup WandB if requested
    wandb_logger = None
    if args.wandb_project and WANDB_AVAILABLE:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'model_size': args.model_size,
            'seed': args.seed,
            'num_classes': 100,
            'dataset': 'CIFAR-100-C'
        }
        
        wandb_logger = DelayedGeneralizationLogger(
            project_name=args.wandb_project,
            experiment_name=args.wandb_name or f"cifar100c_{args.model_size}_{args.seed}",
            config=config,
            phenomenon_type='robustness',
            tags=(args.wandb_tags or []) + ['cifar100c', 'robustness', 'corruption'],
            notes="CIFAR-100-C robustness experiment studying delayed generalization patterns"
        )
        
        print(f"WandB logging enabled: {args.wandb_project}/{wandb_logger.run.name}")
    
    # Create trainer
    trainer = CIFAR100CTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        corruption_test_loaders=corruption_loaders,
        wandb_logger=wandb_logger
    )
    
    # Train model
    print("\n" + "="*60)
    print("STARTING CIFAR-100-C ROBUSTNESS EXPERIMENT")
    print("="*60)
    
    results = trainer.train(epochs=args.epochs, save_dir=args.save_dir)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Final Test Accuracy: {results['final_test_acc']*100:.2f}%")
    print(f"Best Test Accuracy: {results['best_test_acc']*100:.2f}%")
    
    if results['final_robustness_metrics']:
        print(f"Mean Corruption Accuracy: {results['final_robustness_metrics'].get('mean_corruption_acc', 0)*100:.2f}%")
        print(f"Robustness Gap: {results['final_robustness_metrics'].get('robustness_gap', 0)*100:.2f}%")
    
    print(f"Results saved to: {args.save_dir}")
    
    # Final WandB logging
    if wandb_logger:
        wandb_logger.save_experiment_summary(results)


if __name__ == '__main__':
    main()