#!/usr/bin/env python3
"""
Enhanced Analysis Features for Neural Network Training Dynamics

This module provides additional analysis features that were missing from the unified notebook:
1. Difficulty over time tracking
2. Separate accuracy evolution (train/test)
3. Forgetting events analysis with color-coded plots
4. Distribution of forgetting events
5. Class-wise accuracy and loss evolution
6. Highlight opposing gradient pairs during training
7. Loss change highlighting (green for positive, red for negative)
8. More informative .gif animations with bar plots

Usage:
    from enhanced_analysis_features import EnhancedTrainingAnalyzer
    analyzer = EnhancedTrainingAnalyzer()
    # Use with existing training loop...
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime


class EnhancedTrainingAnalyzer:
    """Enhanced analyzer for neural network training dynamics"""
    
    def __init__(self, num_classes: int = 10, max_examples_track: int = 1000):
        self.num_classes = num_classes
        self.max_examples_track = max_examples_track
        
        # Core tracking
        self.epoch_data = []
        self.example_histories = defaultdict(lambda: {
            'losses': [], 
            'predictions': [], 
            'correct': [], 
            'difficulty_score': [],
            'forgetting_events': 0,
            'learned_epochs': [],
            'forgotten_epochs': []
        })
        
        # Class-wise tracking
        self.class_accuracies = {i: [] for i in range(num_classes)}
        self.class_losses = {i: [] for i in range(num_classes)}
        self.class_difficulties = {i: [] for i in range(num_classes)}
        
        # Opposing signals tracking
        self.opposing_pairs = []
        self.gradient_similarities = []
        self.loss_changes = []
        
        # Animation data
        self.animation_frames = []
        self.current_epoch = 0
        
    def track_epoch(self, 
                   model: nn.Module, 
                   train_loader, 
                   test_loader, 
                   criterion,
                   epoch: int,
                   optimizer=None):
        """Track comprehensive training dynamics for one epoch"""
        
        self.current_epoch = epoch
        model.eval()
        
        # Track train and test metrics separately
        train_metrics = self._compute_metrics(model, train_loader, criterion, 'train')
        test_metrics = self._compute_metrics(model, test_loader, criterion, 'test')
        
        # Combine metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'test_acc': test_metrics['accuracy'],
            'class_train_acc': train_metrics['class_accuracies'],
            'class_test_acc': test_metrics['class_accuracies'],
            'class_train_loss': train_metrics['class_losses'],
            'class_test_loss': test_metrics['class_losses']
        }
        
        self.epoch_data.append(epoch_metrics)
        
        # Track class-wise evolution
        for cls in range(self.num_classes):
            self.class_accuracies[cls].append(train_metrics['class_accuracies'][cls])
            self.class_losses[cls].append(train_metrics['class_losses'][cls])
            self.class_difficulties[cls].append(train_metrics['class_difficulties'][cls])
        
        # Track example-level dynamics if we have access to individual examples
        if len(self.example_histories) < self.max_examples_track:
            self._track_example_dynamics(model, train_loader, criterion, epoch)
        
        # Detect opposing signals
        if optimizer is not None:
            opposing_pairs = self._detect_opposing_signals(model, train_loader, criterion)
            self.opposing_pairs.append(opposing_pairs)
        
        # Store frame for animation
        self._store_animation_frame(epoch_metrics)
        
    def _compute_metrics(self, model, loader, criterion, split_name):
        """Compute comprehensive metrics for a data split"""
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        class_losses = [[] for _ in range(self.num_classes)]
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(model.device if hasattr(model, 'device') else 'cpu'), \
                              target.to(model.device if hasattr(model, 'device') else 'cpu')
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Class-wise metrics
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    class_losses[label].append(loss.item())
                    
                    if pred[i].item() == label:
                        class_correct[label] += 1
        
        # Calculate class accuracies and average losses
        class_accuracies = [
            100. * class_correct[i] / max(class_total[i], 1) 
            for i in range(self.num_classes)
        ]
        
        class_avg_losses = [
            np.mean(class_losses[i]) if class_losses[i] else 0
            for i in range(self.num_classes)
        ]
        
        class_difficulties = [
            np.std(class_losses[i]) if len(class_losses[i]) > 1 else 0
            for i in range(self.num_classes)
        ]
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': 100. * correct / total,
            'class_accuracies': class_accuracies,
            'class_losses': class_avg_losses,
            'class_difficulties': class_difficulties
        }
    
    def _track_example_dynamics(self, model, loader, criterion, epoch):
        """Track individual example learning dynamics"""
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if len(self.example_histories) >= self.max_examples_track:
                    break
                    
                data, target = data.to(model.device if hasattr(model, 'device') else 'cpu'), \
                              target.to(model.device if hasattr(model, 'device') else 'cpu')
                
                output = model(data)
                losses = torch.nn.functional.cross_entropy(output, target, reduction='none')
                predictions = output.argmax(dim=1)
                correct = predictions.eq(target)
                
                for i in range(data.size(0)):
                    example_idx = batch_idx * loader.batch_size + i
                    if example_idx >= self.max_examples_track:
                        break
                    
                    # Track example metrics
                    history = self.example_histories[example_idx]
                    history['losses'].append(losses[i].item())
                    history['predictions'].append(predictions[i].item())
                    history['correct'].append(correct[i].item())
                    
                    # Calculate difficulty score (moving average of loss)
                    recent_losses = history['losses'][-10:]  # Last 10 epochs
                    difficulty = np.mean(recent_losses) if recent_losses else losses[i].item()
                    history['difficulty_score'].append(difficulty)
                    
                    # Detect forgetting events
                    if len(history['correct']) >= 2:
                        if history['correct'][-2] and not history['correct'][-1]:
                            history['forgetting_events'] += 1
                            history['forgotten_epochs'].append(epoch)
                        elif not history['correct'][-2] and history['correct'][-1]:
                            history['learned_epochs'].append(epoch)
    
    def _detect_opposing_signals(self, model, loader, criterion, threshold=0.1):
        """Detect pairs of examples with opposing gradient signals"""
        model.train()
        
        # Sample a batch for gradient analysis
        data_iter = iter(loader)
        data, target = next(data_iter)
        data, target = data.to(model.device if hasattr(model, 'device') else 'cpu'), \
                      target.to(model.device if hasattr(model, 'device') else 'cpu')
        
        batch_size = min(32, data.size(0))  # Limit for efficiency
        data, target = data[:batch_size], target[:batch_size]
        
        # Compute gradients for each example
        example_gradients = []
        
        for i in range(batch_size):
            model.zero_grad()
            output = model(data[i:i+1])
            loss = criterion(output, target[i:i+1])
            loss.backward()
            
            # Collect gradients
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.flatten())
            
            if grad_vec:
                example_gradients.append(torch.cat(grad_vec))
        
        # Find opposing pairs
        opposing_pairs = []
        if len(example_gradients) > 1:
            for i, grad_i in enumerate(example_gradients):
                for j, grad_j in enumerate(example_gradients[i+1:], i+1):
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0))
                    
                    # If similarity is negative and magnitude is above threshold
                    if similarity.item() < -threshold:
                        opposing_pairs.append({
                            'example_i': i,
                            'example_j': j,
                            'similarity': similarity.item(),
                            'grad_norm_i': torch.norm(grad_i).item(),
                            'grad_norm_j': torch.norm(grad_j).item()
                        })
        
        return opposing_pairs
    
    def _store_animation_frame(self, metrics):
        """Store data for animation creation"""
        frame_data = {
            'epoch': metrics['epoch'],
            'train_acc': metrics['train_acc'],
            'test_acc': metrics['test_acc'],
            'class_accuracies': metrics['class_train_acc'],
            'opposing_pairs_count': len(self.opposing_pairs[-1]) if self.opposing_pairs else 0,
            'loss_changes': self._compute_loss_changes()
        }
        self.animation_frames.append(frame_data)
    
    def _compute_loss_changes(self):
        """Compute loss changes for highlighting"""
        if len(self.epoch_data) < 2:
            return []
        
        prev_loss = self.epoch_data[-2]['train_loss']
        curr_loss = self.epoch_data[-1]['train_loss']
        change = curr_loss - prev_loss
        
        return {
            'change': change,
            'color': 'green' if change < 0 else 'red',
            'magnitude': abs(change)
        }
    
    def plot_difficulty_evolution(self, save_path=None):
        """Plot difficulty evolution over time"""
        if not self.example_histories:
            print("No example histories to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Average difficulty over time
        epochs = list(range(len(self.epoch_data)))
        avg_difficulties = []
        
        for epoch in epochs:
            epoch_difficulties = [
                history['difficulty_score'][epoch] if len(history['difficulty_score']) > epoch else 0
                for history in self.example_histories.values()
            ]
            avg_difficulties.append(np.mean(epoch_difficulties))
        
        axes[0, 0].plot(epochs, avg_difficulties, 'b-', linewidth=2)
        axes[0, 0].set_title('Average Example Difficulty Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Average Difficulty Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class-wise difficulty evolution
        for cls in range(self.num_classes):
            axes[0, 1].plot(epochs, self.class_difficulties[cls], 
                           label=f'Class {cls}', alpha=0.7)
        
        axes[0, 1].set_title('Class-wise Difficulty Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Difficulty Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Forgetting events distribution
        forgetting_counts = [history['forgetting_events'] for history in self.example_histories.values()]
        axes[1, 0].hist(forgetting_counts, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution of Forgetting Events')
        axes[1, 0].set_xlabel('Number of Forgetting Events')
        axes[1, 0].set_ylabel('Number of Examples')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Forgetting events over time (color-coded)
        forgetting_timeline = defaultdict(int)
        for history in self.example_histories.values():
            for epoch in history['forgotten_epochs']:
                forgetting_timeline[epoch] += 1
        
        if forgetting_timeline:
            epochs_with_forgetting = sorted(forgetting_timeline.keys())
            forgetting_counts = [forgetting_timeline[e] for e in epochs_with_forgetting]
            
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(epochs_with_forgetting)))
            axes[1, 1].bar(epochs_with_forgetting, forgetting_counts, color=colors)
            axes[1, 1].set_title('Forgetting Events Over Time (Color-coded by Intensity)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Number of Forgetting Events')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Difficulty evolution plot saved to {save_path}")
        
        plt.show()
        
    def plot_accuracy_evolution(self, save_path=None):
        """Plot separate train/test accuracy evolution with class-wise breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = [d['epoch'] for d in self.epoch_data]
        train_accs = [d['train_acc'] for d in self.epoch_data]
        test_accs = [d['test_acc'] for d in self.epoch_data]
        
        # 1. Overall train vs test accuracy
        axes[0, 0].plot(epochs, train_accs, 'b-', linewidth=2, label='Train Accuracy')
        axes[0, 0].plot(epochs, test_accs, 'r-', linewidth=2, label='Test Accuracy')
        axes[0, 0].fill_between(epochs, train_accs, test_accs, alpha=0.3, color='gray')
        axes[0, 0].set_title('Train vs Test Accuracy Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class-wise train accuracy
        for cls in range(self.num_classes):
            class_train_accs = [d['class_train_acc'][cls] for d in self.epoch_data]
            axes[0, 1].plot(epochs, class_train_accs, label=f'Class {cls}', alpha=0.7)
        
        axes[0, 1].set_title('Class-wise Train Accuracy Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Class-wise test accuracy
        for cls in range(self.num_classes):
            class_test_accs = [d['class_test_acc'][cls] for d in self.epoch_data]
            axes[1, 0].plot(epochs, class_test_accs, label=f'Class {cls}', alpha=0.7)
        
        axes[1, 0].set_title('Class-wise Test Accuracy Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Accuracy gap (train - test) by class
        for cls in range(self.num_classes):
            class_train_accs = [d['class_train_acc'][cls] for d in self.epoch_data]
            class_test_accs = [d['class_test_acc'][cls] for d in self.epoch_data]
            gap = [t - te for t, te in zip(class_train_accs, class_test_accs)]
            axes[1, 1].plot(epochs, gap, label=f'Class {cls}', alpha=0.7)
        
        axes[1, 1].set_title('Class-wise Accuracy Gap (Train - Test)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy evolution plot saved to {save_path}")
        
        plt.show()
    
    def plot_opposing_signals(self, save_path=None):
        """Plot opposing signals analysis"""
        if not self.opposing_pairs:
            print("No opposing signals data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Opposing pairs count over time
        epochs = list(range(len(self.opposing_pairs)))
        pair_counts = [len(pairs) for pairs in self.opposing_pairs]
        
        axes[0, 0].plot(epochs, pair_counts, 'r-', linewidth=2, marker='o')
        axes[0, 0].set_title('Opposing Signal Pairs Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Number of Opposing Pairs')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Similarity distribution of opposing pairs
        all_similarities = []
        for pairs in self.opposing_pairs:
            all_similarities.extend([pair['similarity'] for pair in pairs])
        
        if all_similarities:
            axes[0, 1].hist(all_similarities, bins=30, alpha=0.7, color='red')
            axes[0, 1].set_title('Distribution of Gradient Similarities (Opposing Pairs)')
            axes[0, 1].set_xlabel('Cosine Similarity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss changes over time with color coding
        epochs = [d['epoch'] for d in self.epoch_data[1:]]  # Skip first epoch
        loss_changes = []
        colors = []
        
        for i in range(1, len(self.epoch_data)):
            change = self.epoch_data[i]['train_loss'] - self.epoch_data[i-1]['train_loss']
            loss_changes.append(change)
            colors.append('green' if change < 0 else 'red')
        
        axes[1, 0].bar(epochs, loss_changes, color=colors, alpha=0.7)
        axes[1, 0].set_title('Loss Changes Over Time (Green=Decrease, Red=Increase)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Change')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Gradient norm distribution for opposing pairs
        all_grad_norms = []
        for pairs in self.opposing_pairs:
            for pair in pairs:
                all_grad_norms.extend([pair['grad_norm_i'], pair['grad_norm_j']])
        
        if all_grad_norms:
            axes[1, 1].hist(all_grad_norms, bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title('Gradient Norm Distribution (Opposing Examples)')
            axes[1, 1].set_xlabel('Gradient Norm')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Opposing signals plot saved to {save_path}")
        
        plt.show()
    
    def create_enhanced_animation(self, save_path=None, fps=10):
        """Create enhanced animation with bar plots and dynamic information"""
        if not self.animation_frames:
            print("No animation data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        def animate(frame_idx):
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            frame = self.animation_frames[frame_idx]
            epoch = frame['epoch']
            
            # 1. Train vs Test accuracy bars
            accuracies = [frame['train_acc'], frame['test_acc']]
            labels = ['Train', 'Test']
            colors = ['blue', 'red']
            
            bars1 = axes[0, 0].bar(labels, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_title(f'Train vs Test Accuracy - Epoch {epoch}')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{acc:.1f}%', ha='center', va='bottom')
            
            # 2. Class-wise accuracy bars
            class_accs = frame['class_accuracies']
            class_indices = list(range(len(class_accs)))
            colors_class = plt.cm.tab10(np.linspace(0, 1, len(class_accs)))
            
            bars2 = axes[0, 1].bar(class_indices, class_accs, color=colors_class, alpha=0.7)
            axes[0, 1].set_title(f'Class-wise Accuracy - Epoch {epoch}')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].set_xticks(class_indices)
            
            # 3. Training dynamics timeline
            epochs_so_far = [f['epoch'] for f in self.animation_frames[:frame_idx+1]]
            train_accs_so_far = [f['train_acc'] for f in self.animation_frames[:frame_idx+1]]
            test_accs_so_far = [f['test_acc'] for f in self.animation_frames[:frame_idx+1]]
            
            axes[1, 0].plot(epochs_so_far, train_accs_so_far, 'b-', linewidth=2, label='Train')
            axes[1, 0].plot(epochs_so_far, test_accs_so_far, 'r-', linewidth=2, label='Test')
            axes[1, 0].scatter([epoch], [frame['train_acc']], color='blue', s=100, zorder=5)
            axes[1, 0].scatter([epoch], [frame['test_acc']], color='red', s=100, zorder=5)
            axes[1, 0].set_title('Accuracy Evolution Timeline')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Training statistics panel
            axes[1, 1].text(0.1, 0.8, f'Epoch: {epoch}', fontsize=16, fontweight='bold')
            axes[1, 1].text(0.1, 0.7, f'Train Accuracy: {frame["train_acc"]:.2f}%', fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Test Accuracy: {frame["test_acc"]:.2f}%', fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Opposing Pairs: {frame["opposing_pairs_count"]}', fontsize=12)
            
            # Loss change indicator
            if frame['loss_changes']:
                change = frame['loss_changes']['change']
                color = frame['loss_changes']['color']
                axes[1, 1].text(0.1, 0.4, f'Loss Change: {change:.4f}', 
                               fontsize=12, color=color, fontweight='bold')
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Training Statistics')
            axes[1, 1].axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.animation_frames),
                                     interval=1000//fps, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Enhanced animation saved to {save_path}")
        
        return anim
    
    def generate_comprehensive_report(self, save_dir="./analysis_results"):
        """Generate comprehensive analysis report with all visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Generating comprehensive training dynamics report...")
        
        # Generate all plots
        self.plot_difficulty_evolution(f"{save_dir}/difficulty_evolution_{timestamp}.png")
        self.plot_accuracy_evolution(f"{save_dir}/accuracy_evolution_{timestamp}.png")
        self.plot_opposing_signals(f"{save_dir}/opposing_signals_{timestamp}.png")
        
        # Create animation
        anim = self.create_enhanced_animation(f"{save_dir}/training_animation_{timestamp}.gif")
        
        # Generate summary statistics
        summary_stats = self._compute_summary_statistics()
        
        # Save summary to file
        with open(f"{save_dir}/summary_statistics_{timestamp}.txt", 'w') as f:
            f.write("Enhanced Training Dynamics Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Comprehensive report generated in {save_dir}/")
        return save_dir
    
    def _compute_summary_statistics(self):
        """Compute summary statistics for the analysis"""
        stats = {}
        
        if self.epoch_data:
            final_epoch = self.epoch_data[-1]
            stats['Final Train Accuracy'] = f"{final_epoch['train_acc']:.2f}%"
            stats['Final Test Accuracy'] = f"{final_epoch['test_acc']:.2f}%"
            stats['Total Epochs'] = len(self.epoch_data)
        
        if self.example_histories:
            forgetting_events = [h['forgetting_events'] for h in self.example_histories.values()]
            stats['Average Forgetting Events'] = f"{np.mean(forgetting_events):.2f}"
            stats['Max Forgetting Events'] = f"{max(forgetting_events)}"
            stats['Examples Tracked'] = len(self.example_histories)
        
        if self.opposing_pairs:
            total_opposing_pairs = sum(len(pairs) for pairs in self.opposing_pairs)
            stats['Total Opposing Pairs Detected'] = total_opposing_pairs
            stats['Average Opposing Pairs per Epoch'] = f"{total_opposing_pairs / len(self.opposing_pairs):.2f}"
        
        return stats


# Example usage function
def demonstrate_enhanced_analysis():
    """Demonstrate the enhanced analysis features"""
    
    print("Enhanced Training Dynamics Analysis - Demo")
    print("=" * 50)
    
    # This would be integrated into existing training loops
    analyzer = EnhancedTrainingAnalyzer(num_classes=10, max_examples_track=500)
    
    print("\nTo use with your training loop:")
    print("""
    # Initialize analyzer
    analyzer = EnhancedTrainingAnalyzer(num_classes=10)
    
    # In your training loop:
    for epoch in range(num_epochs):
        # ... training code ...
        
        # Track enhanced dynamics
        analyzer.track_epoch(model, train_loader, test_loader, criterion, epoch, optimizer)
        
        # Generate plots periodically
        if epoch % 10 == 0:
            analyzer.plot_accuracy_evolution()
            analyzer.plot_difficulty_evolution()
            analyzer.plot_opposing_signals()
    
    # Generate final comprehensive report
    analyzer.generate_comprehensive_report("./enhanced_analysis_results")
    """)


if __name__ == "__main__":
    demonstrate_enhanced_analysis()