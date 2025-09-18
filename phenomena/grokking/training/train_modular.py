#!/usr/bin/env python3
"""
Training Script for Grokking in Modular Arithmetic

This script trains a transformer model on modular arithmetic tasks to observe
the grokking phenomenon - sudden generalization after a period of memorization.

Usage:
    python train_modular.py --data_dir ./modular_arithmetic_data --epochs 10000
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
import time

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from models.simple_transformer import create_grokking_model
# Robust import for the dataset loader: try the expected package path first,
# then ensure the repository root is on sys.path and try again, then fall back
# to an alternative nested package name used in some layouts.
try:
    from data.algorithmic.modular_arithmetic.generate_data import load_dataset
except Exception:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        sys.path.append(str(repo_root))
        from data.algorithmic.modular_arithmetic.generate_data import load_dataset
    except Exception:
        from data.grokking.datasets.algorithmic.modular_arithmetic.generate_data import load_dataset


class GrokkingTrainer:
    """Trainer class for grokking experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        log_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        
        # Optimizer - weight decay is crucial for grokking!
        self.optimizer = optim.AdamW(
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
        self.epochs_logged = []
        
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy for next-token prediction"""
        # We only care about predicting the last token (the result)
        predictions = torch.argmax(logits[:, -1, :], dim=-1)
        target_tokens = targets[:, -1]  # Last token is the result
        correct = (predictions == target_tokens).float()
        return correct.mean().item()
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass - predict all tokens but only evaluate on the last one
                logits = self.model(inputs)
                
                # Loss only on the last token (the result)
                loss = self.criterion(logits[:, -1, :], targets[:, -1])
                
                # Accuracy
                accuracy = self.compute_accuracy(logits, targets)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(inputs)
            
            # Loss only on the last token (the result)
            loss = self.criterion(logits[:, -1, :], targets[:, -1])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            accuracy = self.compute_accuracy(logits, targets)
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, epochs: int, save_dir: Optional[str] = None) -> Dict:
        """Train the model and track grokking metrics"""
        print(f"Training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        best_test_acc = 0.0
        grokking_epoch = None
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate on test set
            test_loss, test_acc = self.evaluate()
            
            # Check for grokking (sudden jump in test accuracy)
            if test_acc > best_test_acc + 0.1 and test_acc > 0.8 and grokking_epoch is None:
                grokking_epoch = epoch
                print(f"\n🎯 GROKKING DETECTED AT EPOCH {epoch}!")
                print(f"   Test accuracy jumped to {test_acc:.3f}")
            
            best_test_acc = max(best_test_acc, test_acc)
            
            # Log metrics
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f} | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"Test Acc: {test_acc:.3f} | "
                      f"Time: {elapsed:.2f}s")
                
                # Store metrics
                self.epochs_logged.append(epoch)
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
        
        # Final results
        results = {
            'grokking_epoch': grokking_epoch,
            'final_train_acc': train_acc,
            'final_test_acc': test_acc,
            'best_test_acc': best_test_acc,
            'epochs': self.epochs_logged,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
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
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = self.epochs_logged
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, label='Train', color='blue')
        ax1.plot(epochs, self.test_losses, label='Test', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, label='Train', color='blue')
        ax2.plot(epochs, self.test_accuracies, label='Test', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Log scale loss
        ax3.semilogy(epochs, self.train_losses, label='Train', color='blue')
        ax3.semilogy(epochs, self.test_losses, label='Test', color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (log scale)')
        ax3.set_title('Loss (Log Scale)')
        ax3.legend()
        ax3.grid(True)
        
        # Test accuracy zoom
        ax4.plot(epochs, self.test_accuracies, color='red', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Test Accuracy (Grokking Detection)')
        ax4.grid(True)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 512) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and test data loaders"""
    print(f"Loading dataset from {data_dir}")
    
    train_inputs, train_targets, test_inputs, test_targets, metadata = load_dataset(data_dir)
    
    # Convert to tensors
    train_inputs = torch.from_numpy(train_inputs).long()
    train_targets = torch.from_numpy(train_targets).long()
    test_inputs = torch.from_numpy(test_inputs).long()
    test_targets = torch.from_numpy(test_targets).long()
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    
    return train_loader, test_loader, metadata


def main():
    parser = argparse.ArgumentParser(description="Train transformer on modular arithmetic for grokking")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (crucial for grokking!)")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_dir", type=str, default="./grokking_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    
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
    model = create_grokking_model(
        vocab_size=metadata['vocab_size'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = GrokkingTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval
    )
    
    # Train model
    print("\n" + "="*60)
    print("STARTING GROKKING EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Model: {args.d_model}d, {args.n_heads}h, {args.n_layers}L")
    print("="*60)
    
    results = trainer.train(args.epochs, args.save_dir)
    
    # Print final results
    print("\n" + "="*60)
    print("GROKKING EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Grokking epoch: {results['grokking_epoch']}")
    print(f"Final train accuracy: {results['final_train_acc']:.3f}")
    print(f"Final test accuracy: {results['final_test_acc']:.3f}")
    print(f"Best test accuracy: {results['best_test_acc']:.3f}")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()