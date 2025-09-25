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

from models.grokking.simple_transformer import create_grokking_model
from utils.wandb_integration.delayed_generalization_logger import setup_wandb_for_phenomenon, create_wandb_config_from_args
from visualization.training_curves import TrainingCurvePlotter
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
        from data.algorithmic.modular_arithmetic.generate_data import load_dataset # FIXIT.


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
        log_interval: int = 100,
        wandb_logger = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        self.wandb_logger = wandb_logger
        
        # Optimizer - Enhanced AdamW with better features for grokking
        try:
            # Try to use enhanced optimizer
            import sys
            import os
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            
            from optimization.enhanced_optimizers import EnhancedAdamW
            self.optimizer = EnhancedAdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                grad_clip_norm=1.0,  # Gradient clipping for stability
                adaptive_weight_decay=False,  # Keep weight decay constant for grokking
                log_grad_stats=True  # Enable gradient statistics
            )
            print("Using Enhanced AdamW optimizer for grokking")
        except ImportError:
            # Fallback to standard AdamW - weight decay is crucial for grokking!
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            print("Using standard AdamW optimizer")
        
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
                
                # Log to wandb if available
                if self.wandb_logger:
                    metrics = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'test_loss': test_loss,
                        'test_accuracy': test_acc,
                        'best_test_accuracy': best_test_acc,
                        'grokking_detected': grokking_epoch is not None,
                        'time_per_epoch': elapsed
                    }
                    if grokking_epoch is not None:
                        metrics['grokking_epoch'] = grokking_epoch
                    
                    self.wandb_logger.log_metrics(metrics, step=epoch)
        
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
        """Plot and save training curves using centralized visualization"""
        plotter = TrainingCurvePlotter(save_dir)
        
        # Use centralized grokking-specific plotting
        plotter.plot_grokking_curves(
            epochs=self.epochs_logged,
            train_losses=self.train_losses,
            test_losses=self.test_losses,
            train_accuracies=self.train_accuracies,
            test_accuracies=self.test_accuracies,
            save_name='training_curves.png'
        )
        
        # Save metrics to JSON
        metrics = {
            'epochs': self.epochs_logged,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_test_loss': self.test_losses[-1] if self.test_losses else 0,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_test_acc': self.test_accuracies[-1] if self.test_accuracies else 0
        }
        plotter.save_metrics_json(metrics, 'grokking_metrics.json')


def create_data_loaders(data_dir: str, batch_size: int = 512, data_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and test data loaders with optional data fraction"""
    print(f"Loading dataset from {data_dir}")
    
    train_inputs, train_targets, test_inputs, test_targets, metadata = load_dataset(data_dir)
    
    # Apply data fraction if specified
    if data_fraction < 1.0:
        print(f"Using {data_fraction:.2%} of the dataset")
        
        # Sample fraction of training data
        train_size = len(train_inputs)
        new_train_size = int(train_size * data_fraction)
        indices = np.random.choice(train_size, new_train_size, replace=False)
        train_inputs = train_inputs[indices]
        train_targets = train_targets[indices]
        
        # Sample fraction of test data
        test_size = len(test_inputs)
        new_test_size = int(test_size * data_fraction)
        indices = np.random.choice(test_size, new_test_size, replace=False)
        test_inputs = test_inputs[indices]
        test_targets = test_targets[indices]
        
        print(f"Reduced train size: {len(train_inputs)} (from {train_size})")
        print(f"Reduced test size: {len(test_inputs)} (from {test_size})")
    
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
    
    print(f"Final train size: {len(train_dataset)}")
    print(f"Final test size: {len(test_dataset)}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    
    return train_loader, test_loader, metadata


def create_experiment_save_dir(base_save_dir: str, args) -> str:
    """Create experiment-specific save directory"""
    if not base_save_dir:
        return None
    
    # Generate experiment name based on key parameters
    exp_name = f"grokking_d{args.d_model}_h{args.n_heads}_l{args.n_layers}_lr{args.learning_rate}_wd{args.weight_decay}"
    if args.wandb_name:
        exp_name = args.wandb_name
    
    # Create full experiment directory path
    experiment_dir = os.path.join(base_save_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


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
    parser.add_argument("--results_dir", type=str, default=None, 
                       help="Alternative name for save_dir (same functionality)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--data_fraction", type=float, default=1.0, 
                       help="Fraction of dataset to use (0.0-1.0, default: 1.0 for full dataset)")
    
    # Wandb logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="delayed_generalization", 
                       help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, 
                       help="Wandb run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                       help="Wandb tags for the run")
    
    args = parser.parse_args()
    
    # Handle alternative results_dir argument
    if args.results_dir is not None:
        args.save_dir = args.results_dir
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader, metadata = create_data_loaders(args.data_dir, args.batch_size, args.data_fraction)
    
    # Create model
    model = create_grokking_model(
        vocab_size=metadata['vocab_size'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    
    # Setup wandb logging if requested
    wandb_logger = None
    if args.use_wandb:
        try:
            # Prepare configuration for wandb
            config = create_wandb_config_from_args(args)
            config.update({
                'dataset_size': len(train_loader.dataset),
                'test_size': len(test_loader.dataset),
                'vocab_size': metadata['vocab_size'],
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(device)
            })
            
            # Generate run name if not provided
            if not args.wandb_name:
                data_spec = args.data_dir.split('/')[-1]
                args.wandb_name = f"grokking_{data_spec}_{args.data_fraction}_d{args.d_model}_wd{args.weight_decay}_lr{args.learning_rate}"

            # Setup wandb logger
            wandb_logger = setup_wandb_for_phenomenon(
                phenomenon_type='grokking',
                project_name=args.wandb_project,
                config=config
            )
            wandb_logger.run.name = args.wandb_name
            if args.wandb_tags:
                wandb_logger.run.tags = args.wandb_tags
            
            print(f"  Wandb tracking: {args.wandb_project}/{args.wandb_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            wandb_logger = None
    
    # Create trainer
    trainer = GrokkingTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        wandb_logger=wandb_logger
    )
    
    # Train model
    print("\n" + "="*60)
    print("STARTING GROKKING EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Data fraction: {args.data_fraction:.2%}")
    print(f"  Train size: {len(train_loader.dataset)}")
    print(f"  Test size: {len(test_loader.dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Model: {args.d_model}d, {args.n_heads}h, {args.n_layers}L")
    if args.use_wandb and wandb_logger:
        print(f"  Wandb: {args.wandb_project}/{args.wandb_name}")
    print("="*60)
    
    # Create experiment-specific save directory
    save_dir = create_experiment_save_dir(args.save_dir, args)
    if save_dir:
        print(f"Experiment results will be saved to: {save_dir}")
    
    results = trainer.train(args.epochs, save_dir)
    
    # Log final results to wandb
    if wandb_logger:
        final_metrics = {
            'final/grokking_epoch': results['grokking_epoch'] if results['grokking_epoch'] else -1,
            'final/train_accuracy': results['final_train_acc'],
            'final/test_accuracy': results['final_test_acc'],
            'final/best_test_accuracy': results['best_test_acc'],
            'final/total_epochs': args.epochs
        }
        wandb_logger.log_metrics(final_metrics)
        
        # Log training curves as plots
        if len(results['epochs']) > 0:
            wandb_logger.log_grokking_curves(
                epochs=results['epochs'],
                train_losses=results['train_losses'],
                test_losses=results['test_losses'],
                train_accuracies=results['train_accuracies'],
                test_accuracies=results['test_accuracies'],
                grokking_epoch=results['grokking_epoch']
            )
        
        # Finish wandb run
        wandb_logger.finish()
    
    # Print final results
    print("\n" + "="*60)
    print("GROKKING EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Grokking epoch: {results['grokking_epoch']}")
    print(f"Final train accuracy: {results['final_train_acc']:.3f}")
    print(f"Final test accuracy: {results['final_test_acc']:.3f}")
    print(f"Best test accuracy: {results['best_test_acc']:.3f}")
    print(f"Results saved to: {save_dir}")
    if wandb_logger:
        print(f"Wandb run: {wandb_logger.run.url}")
    print("="*60)


if __name__ == "__main__":
    main()