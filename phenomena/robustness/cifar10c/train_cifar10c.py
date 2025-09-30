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

try:
    from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: WandB integration not available")
    WANDB_AVAILABLE = False

# from data.vision.cifar10c.generate_synthetic_cifar10c import load_synthetic_cifar10c_dataset
from data.vision.cifar10c.generate_cifar10c import load_cifar10c_dataset
from data.vision.cifar10c.generate_cifar10c import CIFAR10CDataset
from visualization.training_curves import TrainingCurvePlotter
from models.vision.cifar_robustness_models import CIFARModel, create_cifar_robustness_model


class CIFAR10CTrainer:
    """Trainer for synthetic CIFAR-10-C experiments"""
    
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
        self.device = device
        self.corruption_test_loaders = corruption_test_loaders or {}
        self.wandb_logger = wandb_logger
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
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
        accuracy = correct / total
        
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
        accuracy = correct / total
        clean_acc = clean_correct / clean_total if clean_total > 0 else 0
        corrupted_acc = corrupted_correct / corrupted_total if corrupted_total > 0 else 0
        
        return avg_loss, accuracy, clean_acc, corrupted_acc
    
    def evaluate_corruptions(self) -> Dict[str, float]:
        """Evaluate on all corruption types"""
        corruption_results = {}
        
        for corruption_name, loader in self.corruption_test_loaders.items():
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels, _ in loader:  # Ignoring metadata for now
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    pred = outputs.argmax(dim=1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100. * correct / total
            corruption_results[corruption_name] = accuracy
        
        return corruption_results
    
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
            
            # Evaluate individual corruptions
            corruption_results = self.evaluate_corruptions()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.clean_accuracies.append(clean_acc)
            self.corrupted_accuracies.append(corrupted_acc)
            
            # Track individual corruption results
            for corruption_name, acc in corruption_results.items():
                self.corruption_accuracies[corruption_name].append(acc)
            
            # Calculate robustness metrics
            mean_corruption_acc = np.mean(list(corruption_results.values())) if corruption_results else corrupted_acc
            robustness_gap = clean_acc - mean_corruption_acc
            
            robustness_metrics = {
                'mean_corruption_acc': mean_corruption_acc,
                'robustness_gap': robustness_gap
            }
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
                      f"Test Acc: {test_acc*100:.2f}%, Clean: {clean_acc*100:.2f}%, Corrupted: {corrupted_acc*100:.2f}%")
                if corruption_results:
                    print(f"  Mean Corruption Acc: {mean_corruption_acc:.2f}%, Robustness Gap: {robustness_gap:.2f}%")
            
            # WandB logging
            if self.wandb_logger:
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'clean_acc': clean_acc,
                    'corrupted_acc': corrupted_acc
                }
                
                # Add individual corruption results
                for corruption_name, acc in corruption_results.items():
                    metrics[f'corruption_{corruption_name}_acc'] = acc
                
                # Add robustness metrics
                metrics.update(robustness_metrics)
                
                self.wandb_logger.log_epoch_metrics(**metrics)
        
        print(f"\nBest test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")
        
        # Save final results
        results = {
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_acc': self.test_accuracies[-1],
            'final_clean_acc': self.clean_accuracies[-1],
            'final_corrupted_acc': self.corrupted_accuracies[-1],
            'final_robustness_metrics': robustness_metrics,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'clean_accuracies': self.clean_accuracies,
            'corrupted_accuracies': self.corrupted_accuracies,
            'corruption_accuracies': self.corruption_accuracies
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self._plot_results(save_dir)
        
        return results
    
    def _plot_results(self, save_dir: str):
        """Plot training results using centralized visualization"""
        plotter = TrainingCurvePlotter(save_dir)
        
        epochs = list(range(len(self.train_losses)))
        
        # Create corruption accuracies dict for robustness plotting
        corruption_accuracies = {
            'corrupted': self.corrupted_accuracies
        }
        
        # Use centralized robustness plotting
        plotter.plot_robustness_curves(
            epochs=epochs,
            train_losses=self.train_losses,
            test_losses=self.test_losses,
            train_accuracies=self.train_accuracies,
            test_accuracies=self.clean_accuracies,  # Use clean accuracies as test
            corruption_accuracies=corruption_accuracies,
            save_name='robustness_curves.png'
        )
        
        # Save robustness metrics to JSON
        metrics = {
            'epochs': epochs,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'clean_accuracies': self.clean_accuracies,
            'corrupted_accuracies': self.corrupted_accuracies,
            'final_robustness_gap': (self.clean_accuracies[-1] - self.corrupted_accuracies[-1]) if (self.clean_accuracies and self.corrupted_accuracies) else 0,
            'final_clean_acc': self.clean_accuracies[-1] if self.clean_accuracies else 0,
            'final_corrupted_acc': self.corrupted_accuracies[-1] if self.corrupted_accuracies else 0
        }
        plotter.save_metrics_json(metrics, 'robustness_metrics.json')


def create_data_loaders(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """Create data loaders from saved dataset including individual corruption loaders"""
    
    def load_individual_corruption_loaders(base_dir: str, batch_size: int) -> Dict[str, DataLoader]:
        """Load individual corruption datasets as separate data loaders"""
        corruption_loaders = {}
        data_path = Path(base_dir)
        
        # Look for cifar10c subdirectory
        cifar10c_path = data_path / "cifar10c"
        if not cifar10c_path.exists():
            print(f"Warning: cifar10c subdirectory not found in {base_dir}")
            return corruption_loaders
        
        # Load metadata to get corruption types
        metadata_path = cifar10c_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            corruption_types = metadata.get('corruption_types', [])
            severity = metadata.get('severity', 3)
            
            for corruption_type in corruption_types:
                severity_str = f"severity_{severity}"
                subdir = f"{corruption_type}_{severity_str}"
                corruption_subdir = cifar10c_path / subdir
                
                test_dataset_path = corruption_subdir / "test_dataset.pt"
                if test_dataset_path.exists():
                    try:
                        # Load the individual corruption dataset
                        test_dataset = torch.load(test_dataset_path)
                        
                        # Convert to tensors if it's a CIFAR10CDataset
                        if hasattr(test_dataset, '__len__') and hasattr(test_dataset, '__getitem__'):
                            images_list = []
                            labels_list = []
                            corruption_list = []
                            
                            # Sample to determine structure (use subset for efficiency)
                            sample_size = min(100, len(test_dataset))
                            for i in range(sample_size):
                                sample = test_dataset[i]
                                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                                    img, lbl = sample[0], sample[1]
                                    corr = sample[2] if len(sample) > 2 else corruption_type
                                    images_list.append(img)
                                    labels_list.append(lbl)
                                    corruption_list.append(corr)
                            
                            if images_list:
                                # Create a simple tensor dataset for now
                                images = torch.stack([torch.as_tensor(img) for img in images_list])
                                labels = torch.tensor(labels_list)
                                corruptions = torch.tensor([0] * len(labels_list))  # Simplified
                                
                                # Apply normalization
                                cifar10_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                                cifar10_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                                
                                if images.max() > 1.0:
                                    images = images / 255.0
                                images = (images - cifar10_mean) / cifar10_std
                                
                                corruption_dataset = TensorDataset(images.float(), labels, corruptions)
                                corruption_loaders[corruption_type] = DataLoader(
                                    corruption_dataset,
                                    batch_size=batch_size,
                                    shuffle=False
                                )
                                print(f"Loaded corruption dataset: {corruption_type} with {len(corruption_dataset)} samples")
                    except Exception as e:
                        print(f"Warning: Failed to load corruption dataset {corruption_type}: {e}")
        
        return corruption_loaders
    
    # Load main dataset (keeping existing logic)
    train_dataset, test_dataset, metadata = load_cifar10c_dataset(data_dir)
    
    # Convert to tensors
    def dataset_to_tensors(dataset):
        # Support datasets that yield dict-like samples or tuples/lists
        images_list = []
        labels_list = []
        corruption_list = []
        for sample in dataset:
            if isinstance(sample, dict):
                img = sample['image']
                lbl = sample['label']
                corr = sample.get('corruption_type', 0)
            elif isinstance(sample, (tuple, list)):
                # expect (image, label, corruption_type) or (image, label)
                if len(sample) == 3:
                    img, lbl, corr = sample
                elif len(sample) == 2:
                    img, lbl = sample
                    corr = 0
                else:
                    raise ValueError(f"Unexpected sample structure with length {len(sample)}")
            else:
                # try attribute access as a fallback
                try:
                    img = sample.image
                    lbl = sample.label
                    corr = getattr(sample, 'corruption_type', 0)
                except Exception:
                    raise TypeError("Unsupported sample type: must be dict, tuple/list, or object with attributes")
            images_list.append(img)
            labels_list.append(lbl)
            corruption_list.append(corr)
        images = torch.stack([torch.as_tensor(img) for img in images_list])
        labels = torch.tensor(labels_list)

        # Normalize/convert corruption types to integer indices
        def _parse_corruption(corr):
            # Handle dict-like corruption info
            if isinstance(corr, dict):
                for key in ('corruption_type', 'type', 'id', 'idx', 'index', 'severity', 'name'):
                    if key in corr:
                        corr = corr[key]
                        break
                else:
                    # fallback to 0 if dict structure is unexpected
                    return 0
            # If it's a string try to convert to int, otherwise fallback to 0
            if isinstance(corr, str):
                try:
                    return int(corr)
                except Exception:
                    return 0
            # Numeric types
            try:
                return int(corr)
            except Exception:
                return 0

        corruption_types = torch.tensor([_parse_corruption(c) for c in corruption_list], dtype=torch.long)
        return images, labels, corruption_types
    
    train_images, train_labels, train_corruptions = dataset_to_tensors(train_dataset)
    test_images, test_labels, test_corruptions = dataset_to_tensors(test_dataset)
    
    # Apply CIFAR-10 normalization to improve training
    # CIFAR-10 normalization statistics
    cifar10_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    cifar10_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    # Normalize images to [0, 1] first if they're not already
    if train_images.max() > 1.0:
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    # Apply standard CIFAR-10 normalization
    train_images = (train_images - cifar10_mean) / cifar10_std
    test_images = (test_images - cifar10_mean) / cifar10_std
    
    print(f"Data normalization applied - Train images shape: {train_images.shape}")
    print(f"Train images range: [{train_images.min():.3f}, {train_images.max():.3f}]")

    train_images = train_images.float()
    test_images = test_images.float()

    print(f"Train images shape: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"Test images shape: {test_images.shape}, dtype: {test_images.dtype}")
    
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
    
    # Create individual corruption loaders
    corruption_loaders = load_individual_corruption_loaders(data_dir, batch_size)
    
    # Fallback: if no individual corruption loaders found, use main test loader as placeholder
    if not corruption_loaders and hasattr(metadata, 'corruption_types') or 'corruption_types' in metadata:
        corruption_types = metadata.get('corruption_types', []) if isinstance(metadata, dict) else getattr(metadata, 'corruption_types', [])
        for corruption_type in corruption_types:
            corruption_loaders[corruption_type] = test_loader  # Placeholder fallback
    
    return train_loader, test_loader, corruption_loaders


def main():
    parser = argparse.ArgumentParser(description="Train model on synthetic CIFAR-10-C dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_size", type=str, default="small", choices=['very_small', 'small', 'medium', 'large'],
                        help="Model size to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate (default: 1e-4, lower due to normalization)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./cifar10c_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="delayed-generalization", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs='*', default=None, help="Wandb tags")
    
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
    train_loader, test_loader, corruption_loaders = create_data_loaders(args.data_dir, args.batch_size)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Corruption datasets: {list(corruption_loaders.keys())}")
    
    # Create model
    model = create_cifar_robustness_model(model_type='cifar10c', num_classes=10, model_size=args.model_size)
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
            'num_classes': 10,
            'dataset': 'CIFAR-10-C',
            'corruption_types': list(corruption_loaders.keys())
        }
        
        wandb_logger = DelayedGeneralizationLogger(
            project_name=args.wandb_project,
            experiment_name=args.wandb_name or f"cifar10c_{args.model_size}_{args.seed}",
            config=config,
            phenomenon_type='robustness',
            tags=(args.wandb_tags or []) + ['cifar10c', 'robustness', 'corruption'],
            notes="CIFAR-10-C robustness experiment studying delayed generalization patterns"
        )
        
        print(f"WandB logging enabled: {args.wandb_project}/{wandb_logger.run.name}")

    # Create trainer
    trainer = CIFAR10CTrainer(
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
    print(f"Best test accuracy: {results['best_test_acc']*100:.2f}%")
    print(f"Final clean accuracy: {results['final_clean_acc']*100:.2f}%")
    print(f"Final corrupted accuracy: {results['final_corrupted_acc']*100:.2f}%")
    
    # Print corruption results if available
    if 'corruption_accuracies' in results:
        print("\nFinal corruption-specific accuracies:")
        for corruption_type, accuracies in results['corruption_accuracies'].items():
            if accuracies:
                print(f"  {corruption_type}: {accuracies[-1]:.2f}%")
    
    # Save experiment summary to WandB
    if wandb_logger:
        wandb_logger.save_experiment_summary(results)


if __name__ == "__main__":
    main()