#!/usr/bin/env python3
"""
Example: During-Training Analysis with Advanced Features

This example demonstrates how to integrate advanced analysis tools
during the training loop for real-time insights into delayed generalization.
"""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import analysis modules
from data_attribution.analysis import (
    LearningDynamicsAnalyzer,
    FeatureEvolutionTracker,
    GradientFlowAnalyzer,
    MemorizationDetector
)

# Simple CNN for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    print("="*70)
    print("During-Training Analysis Example")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    save_dir = './analysis_results/during_training'
    
    print(f"\nDevice: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Save directory: {save_dir}")
    
    # Load CIFAR-10 dataset
    print("\n[1/7] Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    print("\n[2/7] Creating model...")
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize analysis tools
    print("\n[3/7] Initializing analysis tools...")
    
    # 1. Learning Dynamics Analyzer
    dynamics_analyzer = LearningDynamicsAnalyzer(
        num_classes=10,
        track_examples=1000,  # Track first 1000 examples
        device=device
    )
    print("  ✓ Learning Dynamics Analyzer")
    
    # 2. Feature Evolution Tracker
    feature_tracker = FeatureEvolutionTracker(
        model=model,
        layer_names=['layer2.0', 'layer3.0'],  # Track conv layers
        device=device,
        max_samples_store=500
    )
    print("  ✓ Feature Evolution Tracker")
    
    # 3. Gradient Flow Analyzer
    gradient_analyzer = GradientFlowAnalyzer(
        model=model,
        track_layers=None  # Track all layers
    )
    print("  ✓ Gradient Flow Analyzer")
    
    # 4. Memorization Detector
    memorization_detector = MemorizationDetector(
        num_classes=10,
        device=device
    )
    print("  ✓ Memorization Detector")
    
    # Training loop with integrated analysis
    print("\n[4/7] Starting training with real-time analysis...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradients (before optimizer step)
            if batch_idx % 10 == 0:  # Track every 10 batches to save time
                gradient_analyzer.track_batch_gradients(
                    loss=loss,
                    epoch=epoch,
                    batch_idx=batch_idx
                )
            
            optimizer.step()
            
            # Track learning dynamics
            batch_indices = torch.arange(
                batch_idx * batch_size,
                min((batch_idx + 1) * batch_size, len(train_dataset))
            )
            dynamics_analyzer.track_batch(
                model=model,
                inputs=inputs,
                targets=targets,
                batch_indices=batch_indices,
                epoch=epoch,
                criterion=criterion
            )
            
            # Compute statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")
        
        # End of epoch analysis
        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"    Train Acc: {100.*correct/total:.2f}%")
        
        # Compute epoch-level statistics
        dynamics_analyzer.compute_epoch_statistics(train_loader, model, epoch)
        gradient_analyzer.compute_epoch_statistics(epoch)
        
        # Track feature evolution
        if epoch % 2 == 0:  # Track every 2 epochs to save time
            print("    Tracking feature evolution...")
            feature_tracker.track_epoch(train_loader, epoch, sample_limit=500)
        
        # Compute memorization scores
        if epoch % 3 == 0:  # Every 3 epochs
            print("    Computing memorization scores...")
            mem_metrics = memorization_detector.compute_memorization_scores(
                model, train_loader, test_loader, epoch
            )
            print(f"    Memorization Score: {mem_metrics['memorization_score']:.4f}")
            print(f"    Test Accuracy: {mem_metrics['test_accuracy']:.4f}")
    
    # Post-training analysis
    print("\n[5/7] Performing post-training analysis...")
    
    # Analyze learning patterns
    print("  Analyzing learning patterns...")
    learning_analysis = dynamics_analyzer.analyze_learning_patterns()
    print(f"    Easy examples: {len(learning_analysis['easy_examples'])}")
    print(f"    Hard examples: {len(learning_analysis['hard_examples'])}")
    print(f"    Examples with forgetting: "
          f"{learning_analysis['forgetting_statistics']['examples_with_forgetting']}")
    
    # Analyze gradient flow
    print("  Analyzing gradient flow...")
    gradient_analysis = gradient_analyzer.analyze_gradient_flow()
    print(f"    Final gradient stability: "
          f"{gradient_analysis['overall_stability'][-1]:.4f}")
    
    # Analyze feature evolution
    print("  Analyzing feature evolution...")
    feature_analysis = feature_tracker.analyze_evolution()
    for layer_name in feature_tracker.layer_names:
        if layer_name in feature_analysis:
            stats = feature_analysis[layer_name]['statistics'][-1]
            print(f"    {layer_name}: Silhouette={stats['silhouette_score']:.4f}, "
                  f"Separability={stats['separability']:.4f}")
    
    # Analyze memorization evolution
    print("  Analyzing memorization patterns...")
    mem_analysis = memorization_detector.analyze_memorization_evolution()
    print(f"    Transition epoch: {mem_analysis['transition_epoch']}")
    print(f"    Final memorization: {mem_analysis['final_memorization']:.4f}")
    
    # Generate visualizations
    print("\n[6/7] Generating visualizations...")
    
    print("  Creating learning dynamics plots...")
    dynamics_analyzer.plot_learning_dynamics(save_dir=save_dir)
    
    print("  Creating gradient flow plots...")
    gradient_analyzer.plot_gradient_flow(save_dir=save_dir)
    
    print("  Creating memorization plots...")
    memorization_detector.plot_memorization(save_dir=save_dir)
    
    print("  Creating feature evolution plots...")
    for layer_name in feature_tracker.layer_names:
        try:
            feature_tracker.plot_metrics_evolution(layer_name, save_dir=save_dir)
            feature_tracker.visualize_evolution(layer_name, save_dir=save_dir, method='pca')
        except Exception as e:
            print(f"    Warning: Could not plot {layer_name}: {e}")
    
    # Save analysis results
    print("\n[7/7] Saving analysis results...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    dynamics_analyzer.save(f'{save_dir}/learning_dynamics.pkl')
    gradient_analyzer.save(f'{save_dir}/gradient_flow.pkl')
    memorization_detector.save(f'{save_dir}/memorization.pkl')
    feature_tracker.save(f'{save_dir}/feature_evolution.pkl')
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    print("\nGenerated files:")
    print("  - learning_dynamics.png")
    print("  - gradient_flow.png")
    print("  - memorization_analysis.png")
    print("  - metrics_evolution_*.png")
    print("  - feature_evolution_*.png")
    print("  - *.pkl (analysis data for post-hoc analysis)")
    print("\nKey Insights:")
    print(f"  • {learning_analysis['forgetting_statistics']['examples_with_forgetting']} "
          f"examples experienced forgetting")
    print(f"  • Gradient stability improved to {gradient_analysis['overall_stability'][-1]:.4f}")
    print(f"  • Memorization score reduced by {mem_analysis['memorization_reduction']:.4f}")
    
    # Clean up
    feature_tracker.remove_hooks()
    
    print("\n✓ Example completed successfully!")


if __name__ == '__main__':
    main()
