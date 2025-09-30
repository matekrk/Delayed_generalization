#!/usr/bin/env python3
"""
Training Script for Synthetic CelebA Simplicity Bias Research

This script trains models on the synthetic CelebA dataset to study gender bias
and delayed generalization patterns.

Usage:
    python train_celeba.py --data_dir ./synthetic_celeba_data
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

from data.generate_synthetic_celeba import load_synthetic_celeba_dataset
from visualization.bias_analysis import BiasAnalysisPlotter


class CelebAModel(nn.Module):
    """CNN model for synthetic CelebA classification"""
    
    def __init__(self, num_classes: int = 2, input_size: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers
        conv_output_size = input_size // (2 ** 3)  # 3 max pool operations
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * conv_output_size * conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CelebATrainer:
    """Trainer for synthetic CelebA experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.bias_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, backgrounds) in enumerate(self.train_loader):
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
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, name="Test"):
        """Evaluate on given data loader with bias analysis"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Bias analysis: accuracy when prediction matches spurious feature
        spurious_correct = 0
        spurious_total = 0
        
        with torch.no_grad():
            for images, labels, backgrounds in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                backgrounds = backgrounds.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                # Check if prediction matches background (spurious correlation)
                spurious_pred = backgrounds  # Model using background to predict gender
                spurious_correct += pred.eq(spurious_pred).sum().item()
                spurious_total += labels.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        bias_accuracy = spurious_correct / spurious_total if spurious_total > 0 else 0
        
        return avg_loss, accuracy, bias_accuracy
    
    def train(self, epochs: int, save_dir: str) -> Dict:
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        
        best_test_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Evaluation
            test_loss, test_acc, bias_acc = self.evaluate(self.test_loader, "Test")
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.bias_accuracies.append(bias_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
                      f"Test Acc: {test_acc*100:.2f}%, Bias Acc: {bias_acc*100:.2f}%")
        
        print(f"\nBest test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")
        
        # Save final results
        results = {
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_acc': self.test_accuracies[-1],
            'final_bias_acc': self.bias_accuracies[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'bias_accuracies': self.bias_accuracies
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self._plot_results(save_dir)
        
        return results
    
    def _plot_results(self, save_dir: str):
        """Plot training results using centralized visualization"""
        plotter = BiasAnalysisPlotter(save_dir)
        
        epochs = list(range(len(self.train_losses)))
        
        # Use centralized simplicity bias plotting
        plotter.plot_simplicity_bias_curves(
            epochs=epochs,
            train_losses=self.train_losses,
            test_losses=self.test_losses,
            train_accuracies=self.train_accuracies,
            test_accuracies=self.test_accuracies,
            bias_accuracies=self.bias_accuracies,
            bias_type="background",
            save_name='bias_analysis.png'
        )
        
        # Create summary statistics
        bias_metrics = {
            'final_test_acc': self.test_accuracies[-1] if self.test_accuracies else 0,
            'final_bias_acc': self.bias_accuracies[-1] if self.bias_accuracies else 0,
            'bias_gap': (self.bias_accuracies[-1] - self.test_accuracies[-1]) if (self.bias_accuracies and self.test_accuracies) else 0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_test_loss': self.test_losses[-1] if self.test_losses else 0
        }
        plotter.plot_bias_summary_statistics(bias_metrics, 'bias_summary.png')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders from saved dataset"""
    # Load dataset
    train_dataset, test_dataset, metadata = load_synthetic_celeba_dataset(data_dir)
    
    # Convert to tensors
    def dataset_to_tensors(dataset):
        images = torch.stack([sample['image'] for sample in dataset])
        labels = torch.tensor([sample['label'] for sample in dataset])
        backgrounds = torch.tensor([sample['background_type'] for sample in dataset])
        return images, labels, backgrounds
    
    train_images, train_labels, train_backgrounds = dataset_to_tensors(train_dataset)
    test_images, test_labels, test_backgrounds = dataset_to_tensors(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels, train_backgrounds),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels, test_backgrounds),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train model on synthetic CelebA dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./celeba_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Synthetic CelebA Training")
    print("=" * 25)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = CelebAModel(num_classes=2, input_size=64)
    
    # Create trainer
    trainer = CelebATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("\nStarting training...")
    results = trainer.train(args.epochs, args.save_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"Best test accuracy: {results['best_test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()