#!/usr/bin/env python3
"""
Training Script for Colored MNIST Simplicity Bias Experiments

This script trains CNN models on colored MNIST to observe simplicity bias,
where models initially learn color features before shape features.

Usage:
    python train_colored_mnist.py --data_dir ./colored_mnist_data --epochs 500
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import time

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from models.cnn_models import create_colored_mnist_model
from data.generate_colored_mnist import load_colored_mnist_dataset, ColoredMNISTDataset
from data.generate_synthetic_colored_digits import load_synthetic_dataset, SyntheticColoredDataset


class SimplicitybIasTrainer:
    """Trainer for colored MNIST simplicity bias experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.color_accuracies = []  # Accuracy when predicting based on color
        self.shape_accuracies = []  # Accuracy when predicting based on shape
        self.epochs_logged = []
        
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        return correct.mean().item()
    
    def compute_bias_metrics(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Compute metrics to assess color vs shape bias
        
        Returns:
            overall_accuracy, color_bias_accuracy, shape_bias_accuracy
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_color_indices = []
        all_correlation_followed = []
        
        with torch.no_grad():
            for inputs, labels, metadata in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Get predictions
                logits = self.model(inputs)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_color_indices.extend([m['color_idx'] for m in metadata])
                all_correlation_followed.extend([m['correlation_followed'] for m in metadata])
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_color_indices = np.array(all_color_indices)
        all_correlation_followed = np.array(all_correlation_followed)
        
        # Overall accuracy
        overall_acc = (all_predictions == all_labels).mean()
        
        # Color bias: accuracy on samples that follow color correlation
        color_corr_mask = all_correlation_followed == True
        color_bias_acc = (all_predictions[color_corr_mask] == all_labels[color_corr_mask]).mean() if color_corr_mask.any() else 0.0
        
        # Shape bias: accuracy on samples that violate color correlation
        shape_corr_mask = all_correlation_followed == False
        shape_bias_acc = (all_predictions[shape_corr_mask] == all_labels[shape_corr_mask]).mean() if shape_corr_mask.any() else 0.0
        
        return overall_acc, color_bias_acc, shape_bias_acc
    
    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """Evaluate model on test set with bias metrics"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                accuracy = self.compute_accuracy(logits, labels)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Compute bias metrics
        _, color_acc, shape_acc = self.compute_bias_metrics(self.test_loader)
        
        return avg_loss, avg_accuracy, color_acc, shape_acc
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for inputs, labels, _ in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            accuracy = self.compute_accuracy(logits, labels)
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, epochs: int, save_dir: Optional[str] = None) -> Dict:
        """Train the model and track bias metrics"""
        print(f"Training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        shape_learning_epoch = None
        best_shape_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate on test set
            test_loss, test_acc, color_acc, shape_acc = self.evaluate()
            
            # Check for shape learning (when shape accuracy > color accuracy)
            if shape_acc > color_acc and shape_acc > best_shape_acc and shape_learning_epoch is None:
                shape_learning_epoch = epoch
                print(f"\n🎯 SHAPE LEARNING DETECTED AT EPOCH {epoch}!")
                print(f"   Shape accuracy ({shape_acc:.3f}) > Color accuracy ({color_acc:.3f})")
            
            best_shape_acc = max(best_shape_acc, shape_acc)
            
            # Log metrics
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f} | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"Test Acc: {test_acc:.3f} | "
                      f"Color Acc: {color_acc:.3f} | "
                      f"Shape Acc: {shape_acc:.3f} | "
                      f"Time: {elapsed:.2f}s")
                
                # Store metrics
                self.epochs_logged.append(epoch)
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
                self.color_accuracies.append(color_acc)
                self.shape_accuracies.append(shape_acc)
        
        # Final results
        results = {
            'shape_learning_epoch': shape_learning_epoch,
            'final_train_acc': train_acc,
            'final_test_acc': test_acc,
            'final_color_acc': color_acc,
            'final_shape_acc': shape_acc,
            'best_shape_acc': best_shape_acc,
            'epochs': self.epochs_logged,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'color_accuracies': self.color_accuracies,
            'shape_accuracies': self.shape_accuracies
        }
        
        if save_dir:
            # Save model
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
            
            # Save metrics
            with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Plot training curves
            self.plot_training_curves(save_dir)
        
        return results
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves with bias analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.epochs_logged
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, label='Train', color='blue')
        ax1.plot(epochs, self.test_losses, label='Test', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Overall accuracy curves
        ax2.plot(epochs, self.train_accuracies, label='Train', color='blue')
        ax2.plot(epochs, self.test_accuracies, label='Test', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Bias analysis: Color vs Shape
        ax3.plot(epochs, self.color_accuracies, label='Color Bias', color='orange', linewidth=2)
        ax3.plot(epochs, self.shape_accuracies, label='Shape Bias', color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Color vs Shape Bias Analysis')
        ax3.legend()
        ax3.grid(True)
        
        # Combined view
        ax4.plot(epochs, self.test_accuracies, label='Overall Test', color='red')
        ax4.plot(epochs, self.color_accuracies, label='Color Bias', color='orange')
        ax4.plot(epochs, self.shape_accuracies, label='Shape Bias', color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Simplicity Bias Analysis')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bias_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and test data loaders for colored MNIST"""
    print(f"Loading colored MNIST from {data_dir}")
    
    # Try loading synthetic dataset first, then fall back to real MNIST
    try:
        train_dataset, test_dataset, metadata = load_synthetic_dataset(data_dir)
        print("Loaded synthetic colored dataset")
    except:
        train_dataset, test_dataset, metadata = load_colored_mnist_dataset(data_dir)
        print("Loaded real colored MNIST dataset")
    
    # Custom collate function to handle metadata properly
    def collate_fn(batch):
        images, labels, metadata_list = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        return images, labels, metadata_list
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Train correlation: {metadata['train_correlation']}")
    print(f"Test correlation: {metadata['test_correlation']}")
    
    return train_loader, test_loader, metadata


def main():
    parser = argparse.ArgumentParser(description="Train CNN on colored MNIST for simplicity bias analysis")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to colored MNIST dataset")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model_type", type=str, choices=['simple', 'invariant', 'feature_extractor'], 
                       default='simple', help="Type of model to use")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--save_dir", type=str, default="./colored_mnist_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader, metadata = create_data_loaders(args.data_dir, args.batch_size)
    
    # Create model
    model_kwargs = {'dropout': args.dropout}
    if args.model_type == 'simple':
        model_kwargs['use_batch_norm'] = True
    
    model = create_colored_mnist_model(args.model_type, **model_kwargs)
    
    # Create trainer
    trainer = SimplicitybIasTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval
    )
    
    # Train model
    print("\n" + "="*70)
    print("STARTING COLORED MNIST SIMPLICITY BIAS EXPERIMENT")
    print("="*70)
    print(f"Configuration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Model: {args.model_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Train correlation: {metadata['train_correlation']}")
    print(f"  Test correlation: {metadata['test_correlation']}")
    print("="*70)
    
    results = trainer.train(args.epochs, args.save_dir)
    
    # Print final results
    print("\n" + "="*70)
    print("SIMPLICITY BIAS EXPERIMENT COMPLETED")
    print("="*70)
    print(f"Shape learning epoch: {results['shape_learning_epoch']}")
    print(f"Final test accuracy: {results['final_test_acc']:.3f}")
    print(f"Final color accuracy: {results['final_color_acc']:.3f}")
    print(f"Final shape accuracy: {results['final_shape_acc']:.3f}")
    print(f"Best shape accuracy: {results['best_shape_acc']:.3f}")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()