#!/usr/bin/env python3
"""
Training Script for Sentiment Bias NLP Delayed Generalization Research

This script trains models on sentiment classification with topic bias to study
delayed generalization patterns in NLP tasks.

Usage:
    python train_sentiment_bias.py --data_dir ./sentiment_bias_data --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from transformers import AutoTokenizer, AutoModel
import wandb

# Add parent directories to path for imports
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(repo_root))

from optimization import get_default_optimizer, get_optimizer_stats
from utils.wandb_integration.delayed_generalization_logger import setup_wandb_for_phenomenon, create_wandb_config_from_args
from models.nlp.sentiment_models import SentimentBiasModel, create_sentiment_model


class SentimentBiasTrainer:
    """Trainer for sentiment bias experiments"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_wandb: bool = False,
        wandb_project: str = "delayed-generalization-nlp",
        wandb_tags: Optional[List[str]] = None,
        experiment_name: str = "sentiment_bias"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_wandb = use_wandb
        self.wandb_tags = wandb_tags
        self.experiment_name = experiment_name
        
        # Setup optimizer using enhanced optimizers
        try:
            self.optimizer = get_default_optimizer(
                model, 
                phenomenon_type='simplicity_bias',
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            print("Using enhanced optimizer for sentiment bias experiments")
        except ImportError:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            print("Using standard Adam optimizer")
            
        self.criterion = nn.CrossEntropyLoss()
        
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
                project=wandb_project,
                name=experiment_name,
                config={
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'model_type': model.__class__.__name__,
                    'optimizer': type(self.optimizer).__name__
                },
                tags=self.wandb_tags
            )
    
    def tokenize_batch(self, texts: List[str]):
        """Tokenize text for the model."""
        if self.model.model_type == "simple":
            # Simple tokenization (would need proper vocab in real implementation)
            # For demo, just use character-level encoding
            max_len = 100
            vocab = {chr(i): i for i in range(ord('a'), ord('z') + 1)}
            vocab.update({' ': 26, '.': 27, ',': 28, '!': 29, '?': 30})
            
            tokens = []
            for text in texts:
                text_tokens = [vocab.get(c.lower(), 0) for c in text[:max_len]]
                text_tokens += [0] * (max_len - len(text_tokens))  # padding
                tokens.append(text_tokens)
            
            return torch.tensor(tokens).to(self.device)
        else:
            # Transformer tokenization
            return self.model.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            texts = batch['text']
            labels = torch.tensor(batch['sentiment']).to(self.device)
            
            # Tokenize inputs
            inputs = self.tokenize_batch(texts)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, name="Test"):
        """Evaluate on given data loader with bias analysis"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Bias analysis
        bias_conforming_correct = 0
        bias_conforming_total = 0
        bias_conflicting_correct = 0
        bias_conflicting_total = 0
        
        with torch.no_grad():
            for batch in loader:
                texts = batch['text']
                labels = torch.tensor(batch['sentiment']).to(self.device)
                bias_conforming = batch['bias_conforming']
                
                # Tokenize inputs
                inputs = self.tokenize_batch(texts)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Bias analysis
                for i, is_bias_conf in enumerate(bias_conforming):
                    pred_correct = (predicted[i] == labels[i]).item()
                    if is_bias_conf:
                        bias_conforming_correct += pred_correct
                        bias_conforming_total += 1
                    else:
                        bias_conflicting_correct += pred_correct
                        bias_conflicting_total += 1
        
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        
        bias_conf_acc = 100 * bias_conforming_correct / bias_conforming_total if bias_conforming_total > 0 else 0
        bias_conf_acc = 100 * bias_conflicting_correct / bias_conflicting_total if bias_conflicting_total > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bias_conforming_acc': bias_conf_acc,
            'bias_conflicting_acc': bias_conf_acc,
            'bias_conforming_total': bias_conforming_total,
            'bias_conflicting_total': bias_conflicting_total
        }
    
    def train(self, epochs: int, save_dir: str, log_interval: int = 10) -> Dict:
        """Train the model"""
        print("Starting sentiment bias training...")
        
        best_test_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            test_results = self.evaluate(self.test_loader, "Test")
            
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
            'epochs': self.epochs_logged
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self._plot_results(save_dir)
        
        return results
    
    def _plot_results(self, save_dir: str):
        """Plot training results with bias analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = self.epochs_logged
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.test_losses, label='Test', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.test_accuracies, label='Test', linewidth=2)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Bias analysis
        axes[1, 0].plot(epochs, self.bias_conforming_accuracies, label='Bias Conforming', linewidth=2)
        axes[1, 0].plot(epochs, self.bias_conflicting_accuracies, label='Bias Conflicting', linewidth=2)
        axes[1, 0].set_title('Bias Analysis')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Bias gap
        bias_gap = [conf - conf for conf, conf in zip(self.bias_conforming_accuracies, self.bias_conflicting_accuracies)]
        axes[1, 1].plot(epochs, bias_gap, 'r-', linewidth=2)
        axes[1, 1].set_title('Bias Gap (Conforming - Conflicting)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_data_loaders(data_dir: str, batch_size: int = 32, data_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create data loaders from saved sentiment bias dataset"""
    # Load datasets
    train_data = torch.load(os.path.join(data_dir, "train_dataset.pt"))
    test_data = torch.load(os.path.join(data_dir, "test_dataset.pt"))
    
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    original_train_size = len(train_data['texts'])
    original_test_size = len(test_data['texts'])
    
    # Apply data fraction if specified
    if data_fraction < 1.0:
        print(f"Using {data_fraction:.2%} of the dataset")
        
        # Sample fraction of training data
        train_size = int(original_train_size * data_fraction)
        train_indices = torch.randperm(original_train_size)[:train_size]
        
        for key in train_data:
            train_data[key] = [train_data[key][i] for i in train_indices]
        
        # Sample fraction of test data
        test_size = int(original_test_size * data_fraction)
        test_indices = torch.randperm(original_test_size)[:test_size]
        
        for key in test_data:
            test_data[key] = [test_data[key][i] for i in test_indices]
        
        print(f"Reduced train size: {len(train_data['texts'])} (from {original_train_size})")
        print(f"Reduced test size: {len(test_data['texts'])} (from {original_test_size})")
    
    print(f"Final train size: {len(train_data['texts'])}")
    print(f"Final test size: {len(test_data['texts'])}")
    
    # Create data loaders
    def collate_fn(batch):
        return {
            'text': [item['text'] for item in batch],
            'sentiment': [item['sentiment'] for item in batch],
            'topic': [item['topic'] for item in batch],
            'bias_conforming': [item['bias_conforming'] for item in batch]
        }
    
    # Convert to datasets
    from torch.utils.data import Dataset
    
    class SentimentDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data['texts'])
            
        def __getitem__(self, idx):
            return {
                'text': self.data['texts'][idx],
                'sentiment': self.data['sentiments'][idx],
                'topic': self.data['topics'][idx],
                'bias_conforming': self.data['bias_conforming'][idx]
            }
    
    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Update metadata with actual sizes used
    metadata['actual_train_size'] = len(train_data['texts'])
    metadata['actual_test_size'] = len(test_data['texts'])
    metadata['data_fraction'] = data_fraction
    
    return train_loader, test_loader, metadata


def main():
    parser = argparse.ArgumentParser(description="Train model on sentiment bias dataset")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                       help="Learning rate (default: 5e-4, lower = slower learning)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model_type", type=str, choices=['simple', 'transformer'], 
                       default='simple', help="Type of model to use")
    parser.add_argument("--embed_dim", type=int, default=64, 
                       help="Embedding dimension (default: 64, smaller = harder)")
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="Hidden dimension (default: 128, smaller = harder)")
    parser.add_argument("--dropout", type=float, default=0.3, 
                       help="Dropout rate (default: 0.3, higher = harder)")
    parser.add_argument("--save_dir", type=str, default="./sentiment_bias_results", 
                       help="Directory to save results")
    parser.add_argument("--results_dir", type=str, default=None, 
                       help="Alternative name for save_dir (same functionality)")
    parser.add_argument("--data_fraction", type=float, default=1.0, 
                       help="Fraction of dataset to use (0.0-1.0, default: 1.0 for full dataset)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="delayed-generalization-nlp", 
                       help="WandB project name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                       help="Wandb tags for the run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    
    print("Sentiment Bias NLP Training")
    print("=" * 30)
    print(f"Data directory: {args.data_dir}")
    print(f"Data fraction: {args.data_fraction:.2%}")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Use wandb: {args.use_wandb}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader, metadata = create_data_loaders(args.data_dir, args.batch_size, args.data_fraction)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create experiment name
    bias_topic = metadata['bias_topic']
    train_bias = metadata['train_bias']
    test_bias = metadata['test_bias']
    train_size = metadata['actual_train_size']
    
    # Include size and fraction in experiment name
    if args.data_fraction < 1.0:
        experiment_name = f"sentiment_bias_{bias_topic}_tb{train_bias}_testb{test_bias}_frac{args.data_fraction:.2f}_size{train_size}"
    else:
        experiment_name = f"sentiment_bias_{bias_topic}_tb{train_bias}_testb{test_bias}_size{train_size}"
    
    # Create save directory
    full_save_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # Create model (configurable complexity to control difficulty)
    model = create_sentiment_model(
        model_type=args.model_type,
        vocab_size=5000,  # Fixed smaller vocab to increase difficulty
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = SentimentBiasTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        experiment_name=experiment_name
    )
    
    # Train model
    print(f"\nStarting training...")
    print(f"Results will be saved to: {full_save_dir}")
    
    results = trainer.train(args.epochs, full_save_dir, args.log_interval)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Best test accuracy: {results['best_test_acc']:.2f}%")
    print(f"Final bias conforming accuracy: {results['final_bias_conforming_acc']:.2f}%")
    print(f"Final bias conflicting accuracy: {results['final_bias_conflicting_acc']:.2f}%")
    print(f"Results saved to: {full_save_dir}")


if __name__ == "__main__":
    main()