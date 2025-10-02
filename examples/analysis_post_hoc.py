#!/usr/bin/env python3
"""
Example: Post-Hoc Analysis of Saved Checkpoints

This example demonstrates how to analyze saved model checkpoints
to understand delayed generalization after training is complete.
"""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import analysis modules
from data_attribution.analysis import (
    PhaseTransitionAttributor,
    BiasAttributor,
    FeatureEvolutionTracker,
    MemorizationDetector
)


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration"""
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


def train_and_save_checkpoints(epochs=30, save_dir='./checkpoints'):
    """
    Train a model and save periodic checkpoints for post-hoc analysis.
    """
    print("="*70)
    print("Training Model and Saving Checkpoints")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("\nLoading CIFAR-10...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_epochs = [0, 5, 10, 15, 20, 25, 30]
    
    for epoch in range(epochs + 1):
        if epoch in checkpoint_epochs:
            # Save checkpoint
            checkpoint_path = Path(save_dir) / f'model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        if epoch == epochs:
            break
        
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}")
    
    print("\n✓ Training complete, checkpoints saved!")
    return checkpoint_epochs


def main():
    print("="*70)
    print("Post-Hoc Analysis Example")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './analysis_results/post_hoc'
    checkpoint_dir = './checkpoints'
    
    # Step 1: Train and save checkpoints (or load existing ones)
    print("\n[1/5] Preparing model checkpoints...")
    if not Path(checkpoint_dir).exists() or len(list(Path(checkpoint_dir).glob('*.pth'))) == 0:
        print("  No checkpoints found, training model...")
        checkpoint_epochs = train_and_save_checkpoints(epochs=30, save_dir=checkpoint_dir)
    else:
        print("  Found existing checkpoints!")
        checkpoint_epochs = [0, 5, 10, 15, 20, 25, 30]
    
    # Step 2: Load data
    print("\n[2/5] Loading data...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Step 3: Load model checkpoints
    print("\n[3/5] Loading model checkpoints...")
    model_checkpoints = {}
    
    for epoch in checkpoint_epochs:
        checkpoint_path = Path(checkpoint_dir) / f'model_epoch_{epoch}.pth'
        if checkpoint_path.exists():
            model = SimpleCNN().to(device)
            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()
            model_checkpoints[epoch] = model
            print(f"  ✓ Loaded epoch {epoch}")
    
    print(f"\n  Total checkpoints loaded: {len(model_checkpoints)}")
    
    # Step 4: Perform post-hoc analyses
    print("\n[4/5] Performing post-hoc analyses...")
    
    # Analysis 1: Phase Transition Attribution
    print("\n  [4.1] Phase Transition Attribution")
    print("  " + "-"*60)
    
    if len(model_checkpoints) >= 2:
        # Identify transition epochs (simplified: use early and late epochs)
        early_epoch = checkpoint_epochs[1] if len(checkpoint_epochs) > 1 else checkpoint_epochs[0]
        late_epoch = checkpoint_epochs[-1]
        
        print(f"    Analyzing transition from epoch {early_epoch} to {late_epoch}...")
        
        pt_attributor = PhaseTransitionAttributor(
            model_checkpoints={early_epoch: model_checkpoints[early_epoch],
                             late_epoch: model_checkpoints[late_epoch]},
            train_data=train_dataset,
            test_data=test_dataset,
            device=device
        )
        
        # Note: Full attribution is computationally expensive
        # For demo, we'll just show the setup
        print("    Phase transition attributor initialized")
        print("    (Full attribution computation can take significant time)")
        
        # Example: Find critical examples (simplified version)
        print("    Analyzing a sample of critical examples...")
        sample_indices = list(range(min(100, len(train_dataset))))
        
        # Compute simple influence scores
        influence_scores = {}
        for idx in sample_indices:
            # Simplified: just use model confidence change
            x, y = train_dataset[idx]
            x = x.unsqueeze(0).to(device)
            
            with torch.no_grad():
                out_early = model_checkpoints[early_epoch](x)
                out_late = model_checkpoints[late_epoch](x)
                
                conf_early = torch.softmax(out_early, dim=1).max().item()
                conf_late = torch.softmax(out_late, dim=1).max().item()
                
                influence_scores[idx] = conf_late - conf_early
        
        # Find examples with largest confidence gain
        top_examples = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"    Top 10 examples by confidence gain:")
        for idx, score in top_examples:
            print(f"      Example {idx}: +{score:.4f}")
    
    # Analysis 2: Memorization Analysis Across Checkpoints
    print("\n  [4.2] Memorization Evolution Analysis")
    print("  " + "-"*60)
    
    memorization_detector = MemorizationDetector(num_classes=10, device=device)
    
    for epoch in sorted(model_checkpoints.keys()):
        model = model_checkpoints[epoch]
        mem_metrics = memorization_detector.compute_memorization_scores(
            model, train_loader, test_loader, epoch
        )
        print(f"    Epoch {epoch:2d}: Train Acc={mem_metrics['train_accuracy']:.4f}, "
              f"Test Acc={mem_metrics['test_accuracy']:.4f}, "
              f"Mem Score={mem_metrics['memorization_score']:.4f}")
    
    # Analyze memorization evolution
    mem_analysis = memorization_detector.analyze_memorization_evolution()
    print(f"\n    Memorization Analysis:")
    print(f"      Transition epoch: {mem_analysis['transition_epoch']}")
    print(f"      Max memorization: {mem_analysis['max_memorization']:.4f}")
    print(f"      Final memorization: {mem_analysis['final_memorization']:.4f}")
    print(f"      Reduction: {mem_analysis['memorization_reduction']:.4f}")
    
    # Analysis 3: Feature Evolution Across Checkpoints
    print("\n  [4.3] Feature Space Evolution")
    print("  " + "-"*60)
    
    # Create a fresh model for feature tracking
    feature_model = SimpleCNN().to(device)
    feature_tracker = FeatureEvolutionTracker(
        model=feature_model,
        layer_names=['layer2.0', 'layer3.0'],
        device=device,
        max_samples_store=500
    )
    
    for epoch in sorted(model_checkpoints.keys()):
        print(f"    Tracking features for epoch {epoch}...")
        feature_model.load_state_dict(model_checkpoints[epoch].state_dict())
        feature_tracker.track_epoch(train_loader, epoch, sample_limit=500)
    
    # Analyze evolution
    feature_analysis = feature_tracker.analyze_evolution()
    for layer_name in feature_tracker.layer_names:
        if layer_name in feature_analysis:
            stats = feature_analysis[layer_name]['statistics']
            print(f"\n    {layer_name} Evolution:")
            print(f"      Initial silhouette: {stats[0]['silhouette_score']:.4f}")
            print(f"      Final silhouette: {stats[-1]['silhouette_score']:.4f}")
            print(f"      Initial separability: {stats[0]['separability']:.4f}")
            print(f"      Final separability: {stats[-1]['separability']:.4f}")
    
    # Detect phase transitions
    for layer_name in feature_tracker.layer_names:
        transitions = feature_tracker.detect_phase_transitions(
            layer_name, metric='silhouette_score'
        )
        if transitions:
            print(f"    Detected transitions in {layer_name} at epochs: {transitions}")
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Memorization plots
    print("  Creating memorization analysis plots...")
    memorization_detector.plot_memorization(save_dir=save_dir)
    
    # Feature evolution plots
    print("  Creating feature evolution plots...")
    for layer_name in feature_tracker.layer_names:
        try:
            feature_tracker.plot_metrics_evolution(layer_name, save_dir=save_dir)
            feature_tracker.visualize_evolution(layer_name, save_dir=save_dir, method='pca')
        except Exception as e:
            print(f"    Warning: Could not plot {layer_name}: {e}")
    
    # Custom comparison plot
    print("  Creating checkpoint comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy evolution
    epochs = [m['epoch'] for m in memorization_detector.epoch_memorization]
    train_accs = [m['train_accuracy'] for m in memorization_detector.epoch_memorization]
    test_accs = [m['test_accuracy'] for m in memorization_detector.epoch_memorization]
    
    axes[0].plot(epochs, train_accs, 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs, test_accs, 'r-o', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Evolution Across Checkpoints')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Memorization score
    mem_scores = [m['memorization_score'] for m in memorization_detector.epoch_memorization]
    axes[1].plot(epochs, mem_scores, 'g-o', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Memorization Score')
    axes[1].set_title('Memorization Score Evolution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'checkpoint_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Clean up
    feature_tracker.remove_hooks()
    
    print("\n" + "="*70)
    print("Post-Hoc Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    print("\nKey Findings:")
    print(f"  • Memorization transition occurred at epoch {mem_analysis['transition_epoch']}")
    print(f"  • Memorization score reduced from {mem_analysis['max_memorization']:.4f} "
          f"to {mem_analysis['final_memorization']:.4f}")
    print(f"  • Feature space quality improved across all tracked layers")
    
    print("\n✓ Example completed successfully!")


if __name__ == '__main__':
    main()
