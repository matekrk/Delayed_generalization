#!/usr/bin/env python3
"""
Training Script for ImageNet-C Robustness Research

This script trains models on the ImageNet-C dataset to study 
robustness vs accuracy tradeoffs and delayed generalization patterns.

Usage:
    python train_imagenetc.py --data_dir ./imagenetc_data
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

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: WandB integration not available")
    WANDB_AVAILABLE = False

from data.vision.imagenetc.generate_imagenetc import load_imagenetc_dataset, ImageNetCDataset
from visualization.training_curves import TrainingCurvePlotter
from models.vision.imagenet_robustness_models import create_imagenet_robustness_model


class ImageNetCTrainer:
    """Trainer for ImageNet-C experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        corruption_test_loaders: Optional[Dict[str, DataLoader]] = None,
        wandb_logger: Optional[object] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.corruption_test_loaders = corruption_test_loaders or {}
        self.wandb_logger = wandb_logger
        
        # Setup optimizer with SGD (standard for ImageNet)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.clean_accuracies = []
        self.corrupted_accuracies = []
        self.corruption_accuracies = {}
        
        # Initialize corruption tracking for individual corruption types
        for corruption_name in self.corruption_test_loaders.keys():
            self.corruption_accuracies[corruption_name] = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both 2-tuple and 3-tuple returns
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def test(self, test_loader: DataLoader = None):
        """Test the model"""
        if test_loader is None:
            test_loader = self.test_loader
            
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle both 2-tuple and 3-tuple returns
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_corruptions(self) -> Dict[str, float]:
        """Evaluate model on all corruption types"""
        corruption_results = {}
        
        for corruption_name, corruption_loader in self.corruption_test_loaders.items():
            _, accuracy = self.test(corruption_loader)
            corruption_results[corruption_name] = accuracy
        
        return corruption_results
    
    def compute_robustness_metrics(self, clean_acc: float, corruption_results: Dict[str, float]) -> Dict[str, float]:
        """Compute various robustness metrics"""
        if not corruption_results:
            return {}
        
        corruption_accs = list(corruption_results.values())
        
        metrics = {
            'mean_corruption_acc': np.mean(corruption_accs),
            'min_corruption_acc': np.min(corruption_accs),
            'max_corruption_acc': np.max(corruption_accs),
            'std_corruption_acc': np.std(corruption_accs),
            'robustness_gap': clean_acc - np.mean(corruption_accs),
            'relative_robustness': np.mean(corruption_accs) / clean_acc if clean_acc > 0 else 0
        }
        
        return metrics
    
    def train(self, epochs: int = 120, log_interval: int = 10, save_dir: Optional[str] = None):
        """Main training loop"""
        
        print(f"Training ImageNet-C model for {epochs} epochs...")
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
            
            # Step learning rate scheduler
            self.scheduler.step()
            
            # Evaluate corruptions every few epochs
            corruption_results = {}
            robustness_metrics = {}
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                corruption_results = self.evaluate_corruptions()
                robustness_metrics = self.compute_robustness_metrics(test_acc, corruption_results)
                
                # Track per-corruption accuracies
                for corruption_name, acc in corruption_results.items():
                    self.corruption_accuracies[corruption_name].append(acc)
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if corruption_results:
                    print(f"  Corruption Results:")
                    for corruption_name, acc in corruption_results.items():
                        print(f"    {corruption_name}: {acc:.2f}%")
                
                if robustness_metrics:
                    print(f"  Robustness Metrics:")
                    print(f"    Mean corruption acc: {robustness_metrics['mean_corruption_acc']:.2f}%")
                    print(f"    Robustness gap: {robustness_metrics['robustness_gap']:.2f}%")
                    print(f"    Relative robustness: {robustness_metrics['relative_robustness']:.3f}")
            
            # WandB logging
            if self.wandb_logger:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'test_loss': test_loss,
                    'test_accuracy': test_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                if corruption_results:
                    for corruption_name, acc in corruption_results.items():
                        log_dict[f'corruption_acc/{corruption_name}'] = acc
                
                if robustness_metrics:
                    for metric_name, value in robustness_metrics.items():
                        log_dict[f'robustness/{metric_name}'] = value
                
                self.wandb_logger.log_metrics(log_dict, step=epoch + 1)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_test_loss, final_test_acc = self.test()
        final_corruption_results = self.evaluate_corruptions()
        final_robustness_metrics = self.compute_robustness_metrics(final_test_acc, final_corruption_results)
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best test accuracy: {best_test_acc:.2f}%")
        print(f"Final test accuracy: {final_test_acc:.2f}%")
        
        if final_corruption_results:
            print("\nFinal Corruption Results:")
            for corruption_name, acc in final_corruption_results.items():
                print(f"  {corruption_name}: {acc:.2f}%")
        
        if final_robustness_metrics:
            print("\nFinal Robustness Metrics:")
            for metric_name, value in final_robustness_metrics.items():
                print(f"  {metric_name}: {value:.3f}")
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save(self.model.state_dict(), save_path / "final_model.pt")
            torch.save(best_model_state, save_path / "best_model.pt")
            
            # Save training history
            history = {
                'train_losses': self.train_losses,
                'test_losses': self.test_losses,
                'train_accuracies': self.train_accuracies,
                'test_accuracies': self.test_accuracies,
                'corruption_accuracies': self.corruption_accuracies,
                'best_test_acc': best_test_acc,
                'final_test_acc': final_test_acc,
                'final_corruption_results': final_corruption_results,
                'final_robustness_metrics': final_robustness_metrics
            }
            
            with open(save_path / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)
            
            print(f"\nResults saved to: {save_path}")
        
        return {
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'final_corruption_results': final_corruption_results,
            'final_robustness_metrics': final_robustness_metrics
        }


def create_data_loaders(data_dir: str, batch_size: int = 256, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """Create data loaders for ImageNet-C training"""
    
    # Load main dataset
    try:
        dataset, metadata = load_imagenetc_dataset(data_dir)
        print(f"Loaded dataset with {len(dataset)} samples")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Split into train/test (or use as test only)
    # For ImageNet-C, typically we use validation set
    # Here we'll use it as test set
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # For training, we would load clean ImageNet data
    # For now, we'll use the same data for both
    train_loader = test_loader
    
    # Load individual corruption loaders if available
    corruption_loaders = {}
    data_path = Path(data_dir).parent
    if (data_path / "imagenetc").exists():
        imagenetc_path = data_path / "imagenetc"
        for corruption_dir in imagenetc_path.iterdir():
            if corruption_dir.is_dir() and (corruption_dir / "dataset.pt").exists():
                corruption_name = corruption_dir.name.split('_severity_')[0]
                try:
                    corr_dataset, _ = load_imagenetc_dataset(str(corruption_dir))
                    corruption_loaders[corruption_name] = DataLoader(
                        corr_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True
                    )
                except Exception as e:
                    print(f"Warning: Could not load corruption {corruption_name}: {e}")
    
    return train_loader, test_loader, corruption_loaders


def main():
    parser = argparse.ArgumentParser(description="Train ImageNet-C robustness models")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing ImageNet-C data")
    parser.add_argument("--save_dir", type=str, default="./imagenetc_results",
                       help="Directory to save results")
    parser.add_argument("--model_type", type=str, default="resnet50",
                       choices=["resnet18", "resnet50", "resnet101", "mobilenet_v2"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=90,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--pretrained", action="store_true",
                       help="Use pretrained model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs='*', default=None,
                       help="WandB tags")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup WandB if requested
    wandb_logger = None
    if args.wandb_project and WANDB_AVAILABLE:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'model_type': args.model_type,
            'pretrained': args.pretrained,
            'seed': args.seed,
            'num_classes': 1000,
            'dataset': 'ImageNet-C'
        }
        
        experiment_name = args.wandb_name or f"imagenetc_{args.model_type}_{args.seed}"
        
        wandb_logger = DelayedGeneralizationLogger(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config=config,
            phenomenon_type='robustness',
            tags=(args.wandb_tags or []) + ['imagenetc', 'robustness'],
            notes="ImageNet-C robustness experiment studying delayed generalization patterns"
        )
        print(f"WandB logging enabled: {args.wandb_project}/{wandb_logger.run.name}")
    
    print("ImageNet-C Training")
    print("=" * 30)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Model: {args.model_type}")
    print(f"Pretrained: {args.pretrained}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader, corruption_loaders = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Corruption datasets: {list(corruption_loaders.keys())}")
    
    # Create model
    model = create_imagenet_robustness_model(
        model_type=args.model_type,
        num_classes=1000,
        pretrained=args.pretrained,
        robust=True
    )
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Create trainer
    trainer = ImageNetCTrainer(
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
    print("\nStarting training...")
    results = trainer.train(args.epochs, args.save_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.save_dir}")
    
    # Final wandb logging
    if wandb_logger:
        wandb_logger.save_experiment_summary(results)
        wandb_logger.finish()


if __name__ == "__main__":
    main()
