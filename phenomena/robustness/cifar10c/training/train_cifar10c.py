#!/usr/bin/env python3
"""
Training Script for Synthetic CIFAR-10-C Robustness Research

This script trains models on the synthetic CIFAR-10-C dataset to study 
robustness vs accuracy tradeoffs and delayed generalization patterns.

Usage:
    python train_cifar10c.py --data_dir ./synthetic_cifar10c_data
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

from data.generate_synthetic_cifar10c import load_synthetic_cifar10c_dataset


class CIFAR10CModel(nn.Module):
    """CNN model for synthetic CIFAR-10-C classification"""
    
    def __init__(self, num_classes: int = 10, input_size: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers
        conv_output_size = input_size // (2 ** 3)  # 3 max pool operations
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * conv_output_size * conv_output_size, 512),
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


class CIFAR10CTrainer:
    """Trainer for synthetic CIFAR-10-C experiments"""
    
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
        self.clean_accuracies = []
        self.corrupted_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, corruption_types) in enumerate(self.train_loader):
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
        """Evaluate on given data loader with corruption analysis"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Corruption analysis
        clean_correct = 0
        clean_total = 0
        corrupted_correct = 0
        corrupted_total = 0
        
        with torch.no_grad():
            for images, labels, corruption_types in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                corruption_types = corruption_types.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                # Separate clean vs corrupted performance
                clean_mask = (corruption_types == 0)  # Assume 0 means clean
                corrupted_mask = ~clean_mask
                
                if clean_mask.sum() > 0:
                    clean_correct += pred[clean_mask].eq(labels[clean_mask]).sum().item()
                    clean_total += clean_mask.sum().item()
                
                if corrupted_mask.sum() > 0:
                    corrupted_correct += pred[corrupted_mask].eq(labels[corrupted_mask]).sum().item()
                    corrupted_total += corrupted_mask.sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        clean_acc = 100. * clean_correct / clean_total if clean_total > 0 else 0
        corrupted_acc = 100. * corrupted_correct / corrupted_total if corrupted_total > 0 else 0
        
        return avg_loss, accuracy, clean_acc, corrupted_acc
    
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
            test_loss, test_acc, clean_acc, corrupted_acc = self.evaluate(self.test_loader, "Test")
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.clean_accuracies.append(clean_acc)
            self.corrupted_accuracies.append(corrupted_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%, Clean: {clean_acc:.2f}%, Corrupted: {corrupted_acc:.2f}%")
        
        print(f"\nBest test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
        
        # Save final results
        results = {
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_acc': self.test_accuracies[-1],
            'final_clean_acc': self.clean_accuracies[-1],
            'final_corrupted_acc': self.corrupted_accuracies[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'clean_accuracies': self.clean_accuracies,
            'corrupted_accuracies': self.corrupted_accuracies
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
        
        # Clean vs Corrupted accuracy
        axes[1, 0].plot(self.clean_accuracies, label='Clean', color='blue')
        axes[1, 0].plot(self.corrupted_accuracies, label='Corrupted', color='red')
        axes[1, 0].set_title('Clean vs Corrupted Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Robustness gap
        robustness_gap = np.array(self.clean_accuracies) - np.array(self.corrupted_accuracies)
        axes[1, 1].plot(robustness_gap, label='Robustness Gap', color='orange')
        axes[1, 1].set_title('Robustness Gap (Clean - Corrupted)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders from saved dataset"""
    # Load dataset
    train_dataset, test_dataset, metadata = load_synthetic_cifar10c_dataset(data_dir)
    
    # Convert to tensors
    def dataset_to_tensors(dataset):
        images = torch.stack([sample['image'] for sample in dataset])
        labels = torch.tensor([sample['label'] for sample in dataset])
        corruption_types = torch.tensor([sample['corruption_type'] for sample in dataset])
        return images, labels, corruption_types
    
    train_images, train_labels, train_corruptions = dataset_to_tensors(train_dataset)
    test_images, test_labels, test_corruptions = dataset_to_tensors(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels, train_corruptions),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels, test_corruptions),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train model on synthetic CIFAR-10-C dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./cifar10c_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Synthetic CIFAR-10-C Training")
    print("=" * 30)
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
    model = CIFAR10CModel(num_classes=10, input_size=32)
    
    # Create trainer
    trainer = CIFAR10CTrainer(
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
    print(f"Best test accuracy: {results['best_test_acc']:.2f}%")
    print(f"Final clean accuracy: {results['final_clean_acc']:.2f}%")
    print(f"Final corrupted accuracy: {results['final_corrupted_acc']:.2f}%")


if __name__ == "__main__":
    main()