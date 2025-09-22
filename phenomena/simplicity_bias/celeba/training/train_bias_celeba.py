#!/usr/bin/env python3
"""
Training Script for Real CelebA Simplicity Bias Research

This script trains models on the real CelebA dataset with bias to study 
attribute bias and delayed generalization patterns.

Usage:
    python train_real_celeba.py --data_dir ./real_celeba_bias_data/real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import wandb

# Add parent directories to path for imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(repo_root))

from data.vision.celeba.generate_bias_celeba import load_real_celeba_dataset, BiasedRealCelebADataset


class RealCelebAModel(nn.Module):
    """CNN model for real CelebA classification"""
    
    def __init__(self, num_classes: int = 2, input_size: int = 64, dropout_rate: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block for better real image handling
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers
        conv_output_size = input_size // (2 ** 4)  # 4 max pool operations
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * conv_output_size * conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RealCelebATrainer:
    """Trainer for real CelebA experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_wandb: bool = False,
        experiment_name: str = "real_celeba_bias"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_wandb = use_wandb
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.bias_conforming_accuracies = []
        self.bias_conflicting_accuracies = []
        self.epochs_logged = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="delayed-generalization-celeba",
                name=experiment_name,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "model": "RealCelebAModel"
                }
            )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, name="Test"):
        """Evaluate on given data loader with detailed bias analysis"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Detailed bias analysis
        bias_conforming_correct = 0
        bias_conforming_total = 0
        bias_conflicting_correct = 0
        bias_conflicting_total = 0
        
        attr1_correct = {}
        attr1_total = {}
        attr2_correct = {}
        attr2_total = {}
        
        with torch.no_grad():
            for images, labels, metadata_list in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                # Bias analysis
                for i, metadata in enumerate(metadata_list):
                    if isinstance(metadata, dict):
                        bias_followed = metadata.get('bias_followed', False)
                        attr1_val = metadata.get('attr1', -1)
                        attr2_val = metadata.get('attr2', -1)
                        
                        is_correct = pred[i].item() == labels[i].item()
                        
                        if bias_followed:
                            bias_conforming_correct += is_correct
                            bias_conforming_total += 1
                        else:
                            bias_conflicting_correct += is_correct
                            bias_conflicting_total += 1
                        
                        # Detailed attribute accuracy
                        attr1_key = f"attr1={attr1_val}"
                        attr2_key = f"attr2={attr2_val}"
                        
                        if attr1_key not in attr1_correct:
                            attr1_correct[attr1_key] = 0
                            attr1_total[attr1_key] = 0
                        if attr2_key not in attr2_correct:
                            attr2_correct[attr2_key] = 0
                            attr2_total[attr2_key] = 0
                        
                        attr1_correct[attr1_key] += is_correct
                        attr1_total[attr1_key] += 1
                        attr2_correct[attr2_key] += is_correct
                        attr2_total[attr2_key] += 1
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        bias_conforming_acc = 100. * bias_conforming_correct / bias_conforming_total if bias_conforming_total > 0 else 0
        bias_conflicting_acc = 100. * bias_conflicting_correct / bias_conflicting_total if bias_conflicting_total > 0 else 0
        
        # Calculate attribute-specific accuracies
        attr_accuracies = {}
        for attr_key in attr1_correct:
            if attr1_total[attr_key] > 0:
                attr_accuracies[attr_key] = 100. * attr1_correct[attr_key] / attr1_total[attr_key]
        for attr_key in attr2_correct:
            if attr2_total[attr_key] > 0:
                attr_accuracies[attr_key] = 100. * attr2_correct[attr_key] / attr2_total[attr_key]
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bias_conforming_acc': bias_conforming_acc,
            'bias_conflicting_acc': bias_conflicting_acc,
            'bias_conforming_total': bias_conforming_total,
            'bias_conflicting_total': bias_conflicting_total,
            'attr_accuracies': attr_accuracies
        }
        
        return results
    
    def train(self, epochs: int, save_dir: str, log_interval: int = 10) -> Dict:
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        
        best_test_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Evaluation
            train_results = self.evaluate(self.train_loader, "Train")
            test_results = self.evaluate(self.test_loader, "Test")
            
            # Update learning rate
            self.scheduler.step(test_results['accuracy'])
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_results['loss'])
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_results['accuracy'])
            self.bias_conforming_accuracies.append(test_results['bias_conforming_acc'])
            self.bias_conflicting_accuracies.append(test_results['bias_conflicting_acc'])
            self.epochs_logged.append(epoch)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'test_loss': test_results['loss'],
                    'test_accuracy': test_results['accuracy'],
                    'bias_conforming_accuracy': test_results['bias_conforming_acc'],
                    'bias_conflicting_accuracy': test_results['bias_conflicting_acc'],
                    'bias_gap': test_results['bias_conforming_acc'] - test_results['bias_conflicting_acc'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if test_results['accuracy'] > best_test_acc:
                best_test_acc = test_results['accuracy']
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            # Logging
            if epoch % log_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Acc: {test_results['accuracy']:.2f}%, "
                      f"Bias Conform: {test_results['bias_conforming_acc']:.2f}%, "
                      f"Bias Conflict: {test_results['bias_conflicting_acc']:.2f}%")
        
        print(f"\nBest test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
        
        # Final evaluation with detailed analysis
        print("\nFinal detailed evaluation:")
        train_final = self.evaluate(self.train_loader, "Train")
        test_final = self.evaluate(self.test_loader, "Test")
        
        print(f"Train - Bias conforming: {train_final['bias_conforming_acc']:.2f}% "
              f"({train_final['bias_conforming_total']} samples)")
        print(f"Train - Bias conflicting: {train_final['bias_conflicting_acc']:.2f}% "
              f"({train_final['bias_conflicting_total']} samples)")
        print(f"Test - Bias conforming: {test_final['bias_conforming_acc']:.2f}% "
              f"({test_final['bias_conforming_total']} samples)")
        print(f"Test - Bias conflicting: {test_final['bias_conflicting_acc']:.2f}% "
              f"({test_final['bias_conflicting_total']} samples)")
        
        # Save final results
        results = {
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_acc': self.test_accuracies[-1],
            'final_bias_conforming_acc': self.bias_conforming_accuracies[-1],
            'final_bias_conflicting_acc': self.bias_conflicting_accuracies[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'bias_conforming_accuracies': self.bias_conforming_accuracies,
            'bias_conflicting_accuracies': self.bias_conflicting_accuracies,
            'epochs': self.epochs_logged,
            'final_train_results': train_final,
            'final_test_results': test_final
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self._plot_results(save_dir)
        
        return results
    
    def _plot_results(self, save_dir: str):
        """Plot training results with bias analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = self.epochs_logged
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.test_losses, label='Test', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.test_accuracies, label='Test', linewidth=2)
        axes[0, 1].set_title('Overall Accuracy Curves', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bias analysis
        axes[0, 2].plot(epochs, self.bias_conforming_accuracies, label='Bias Conforming', 
                       color='green', linewidth=2)
        axes[0, 2].plot(epochs, self.bias_conflicting_accuracies, label='Bias Conflicting', 
                       color='red', linewidth=2)
        axes[0, 2].set_title('Bias Analysis', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bias gap over time
        bias_gap = np.array(self.bias_conforming_accuracies) - np.array(self.bias_conflicting_accuracies)
        axes[1, 0].plot(epochs, bias_gap, color='purple', linewidth=2)
        axes[1, 0].set_title('Bias Gap (Conforming - Conflicting)', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Gap (%)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test accuracy vs bias gap
        axes[1, 1].scatter(self.test_accuracies, bias_gap, alpha=0.6, s=30)
        axes[1, 1].set_title('Test Accuracy vs Bias Gap', fontsize=14)
        axes[1, 1].set_xlabel('Test Accuracy (%)')
        axes[1, 1].set_ylabel('Bias Gap (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final comparison
        final_metrics = ['Test Acc', 'Bias Conform', 'Bias Conflict']
        final_values = [self.test_accuracies[-1], self.bias_conforming_accuracies[-1], 
                       self.bias_conflicting_accuracies[-1]]
        colors = ['blue', 'green', 'red']
        
        bars = axes[1, 2].bar(final_metrics, final_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Final Performance Comparison', fontsize=14)
        axes[1, 2].set_ylabel('Accuracy (%)')
        axes[1, 2].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 32, data_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create data loaders from saved real CelebA dataset with optional data fraction"""
    # Load dataset
    train_dataset, test_dataset, metadata = load_real_celeba_dataset(data_dir)
    
    original_train_size = len(train_dataset)
    original_test_size = len(test_dataset)
    
    # Apply data fraction if specified
    if data_fraction < 1.0:
        print(f"Using {data_fraction:.2%} of the dataset")
        
        # Create subset of training data
        train_size = int(original_train_size * data_fraction)
        indices = torch.randperm(original_train_size)[:train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        # Create subset of test data
        test_size = int(original_test_size * data_fraction)
        indices = torch.randperm(original_test_size)[:test_size]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        print(f"Reduced train size: {len(train_dataset)} (from {original_train_size})")
        print(f"Reduced test size: {len(test_dataset)} (from {original_test_size})")
    
    print(f"Dataset loaded: {metadata['attr1_name']} vs {metadata['attr2_name']}")
    print(f"Final train size: {len(train_dataset)}, Final test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Update metadata with actual sizes used
    metadata = metadata.copy()
    metadata['actual_train_size'] = len(train_dataset)
    metadata['actual_test_size'] = len(test_dataset)
    metadata['data_fraction'] = data_fraction
    
    return train_loader, test_loader, metadata


def main():
    parser = argparse.ArgumentParser(description="Train model on real CelebA dataset with bias")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to dataset directory (should contain train_dataset.pt, test_dataset.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--save_dir", type=str, default="./real_celeba_results", 
                       help="Directory to save results")
    parser.add_argument("--results_dir", type=str, default=None, 
                       help="Alternative name for save_dir (same functionality)")
    parser.add_argument("--data_fraction", type=float, default=1.0, 
                       help="Fraction of dataset to use (0.0-1.0, default: 1.0 for full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    args = parser.parse_args()
    
    # Handle alternative results_dir argument
    if args.results_dir is not None:
        args.save_dir = args.results_dir
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("Real CelebA Bias Training")
    print("=" * 30)
    print(f"Data directory: {args.data_dir}")
    print(f"Data fraction: {args.data_fraction:.2%}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Use wandb: {args.use_wandb}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader, metadata = create_data_loaders(args.data_dir, args.batch_size, args.data_fraction)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create experiment name
    attr1 = metadata['attr1_name']
    attr2 = metadata['attr2_name']
    train_bias = metadata['train_bias']
    test_bias = metadata['test_bias']
    train_size = metadata['actual_train_size']
    
    # Include size and fraction in experiment name
    if args.data_fraction < 1.0:
        experiment_name = f"real_celeba_{attr1}_{attr2}_tb{train_bias}_testb{test_bias}_frac{args.data_fraction:.2f}_size{train_size}"
    else:
        experiment_name = f"real_celeba_{attr1}_{attr2}_tb{train_bias}_testb{test_bias}_size{train_size}"
    
    # Create model
    model = RealCelebAModel(
        num_classes=2, 
        input_size=metadata['image_size'],
        dropout_rate=args.dropout_rate
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = RealCelebATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        experiment_name=experiment_name
    )
    
    # Train model
    print("\nStarting training...")
    results = trainer.train(args.epochs, args.save_dir, args.log_interval)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"Best test accuracy: {results['best_test_acc']:.2f}%")
    print(f"Final bias gap: {results['final_bias_conforming_acc'] - results['final_bias_conflicting_acc']:.2f}%")
    
    # Save metadata with results
    final_metadata = {
        **metadata,
        'training_args': vars(args),
        'final_results': {
            'best_test_acc': results['best_test_acc'],
            'final_bias_gap': results['final_bias_conforming_acc'] - results['final_bias_conflicting_acc']
        }
    }
    
    with open(os.path.join(args.save_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(final_metadata, f, indent=2)


if __name__ == "__main__":
    main()