#!/usr/bin/env python3
"""
Example: Multi-Phenomenon Analysis Comparison

This example demonstrates how to use the analysis tools to compare
different delayed generalization phenomena (grokking, simplicity bias, etc.).
"""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Import analysis modules
from data_attribution.analysis import (
    LearningDynamicsAnalyzer,
    FeatureEvolutionTracker,
    MemorizationDetector
)


def create_grokking_dataset(n_samples=1000, modulo=97):
    """
    Create a modular arithmetic dataset (for grokking phenomenon).
    Task: (a + b) mod modulo
    """
    np.random.seed(42)
    
    # Generate all possible pairs
    a = np.random.randint(0, modulo, n_samples)
    b = np.random.randint(0, modulo, n_samples)
    y = (a + b) % modulo
    
    # One-hot encode inputs
    X = np.zeros((n_samples, modulo * 2))
    for i in range(n_samples):
        X[i, a[i]] = 1
        X[i, modulo + b[i]] = 1
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def create_simplicity_bias_dataset(n_samples=1000, bias_strength=0.9):
    """
    Create a dataset with spurious correlations (for simplicity bias).
    Task: Classify based on core feature, but spurious feature is easier.
    """
    np.random.seed(42)
    
    # Core features (harder to learn)
    core_features = np.random.randn(n_samples, 10)
    labels = (core_features[:, 0] + core_features[:, 1] > 0).astype(int)
    
    # Spurious features (easier to learn, correlated with label)
    spurious_features = np.zeros((n_samples, 5))
    for i in range(n_samples):
        if np.random.rand() < bias_strength:
            # Spurious feature correlates with label
            spurious_features[i] = labels[i] * 2 + np.random.randn(5) * 0.1
        else:
            # Random (breaks correlation)
            spurious_features[i] = np.random.randn(5)
    
    X = np.concatenate([core_features, spurious_features], axis=1)
    
    return torch.FloatTensor(X), torch.LongTensor(labels)


def create_normal_dataset(n_samples=1000):
    """
    Create a normal dataset without delayed generalization.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    return torch.FloatTensor(X), torch.LongTensor(y)


class SimpleMLLP(nn.Module):
    """Simple MLP for tabular data"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_and_analyze_phenomenon(
    phenomenon_name: str,
    train_data,
    test_data,
    input_dim: int,
    output_dim: int,
    epochs: int,
    learning_rate: float,
    device: torch.device
):
    """
    Train a model and analyze it for a specific phenomenon.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {phenomenon_name}")
    print(f"{'='*70}")
    
    # Create data loaders
    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleMLLP(input_dim, hidden_dim=128, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize analyzers
    dynamics_analyzer = LearningDynamicsAnalyzer(
        num_classes=output_dim,
        track_examples=min(500, len(train_dataset)),
        device=device
    )
    
    memorization_detector = MemorizationDetector(
        num_classes=output_dim,
        device=device
    )
    
    # Training loop
    results = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)
            
            # Track dynamics
            batch_indices = torch.arange(
                batch_idx * train_loader.batch_size,
                min((batch_idx + 1) * train_loader.batch_size, len(train_dataset))
            )
            dynamics_analyzer.track_batch(
                model, inputs, targets, batch_indices, epoch, criterion
            )
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == targets).sum().item()
                test_total += targets.size(0)
        
        # Record results
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_loss'].append(train_loss / len(train_loader))
        results['test_loss'].append(test_loss / len(test_loader))
        
        # Compute epoch statistics
        dynamics_analyzer.compute_epoch_statistics(train_loader, model, epoch)
        
        # Compute memorization scores every 10 epochs
        if epoch % 10 == 0:
            mem_metrics = memorization_detector.compute_memorization_scores(
                model, train_loader, test_loader, epoch
            )
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: "
                  f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
                  f"Loss={train_loss/len(train_loader):.4f}")
    
    # Final analysis
    learning_analysis = dynamics_analyzer.analyze_learning_patterns()
    mem_analysis = memorization_detector.analyze_memorization_evolution()
    
    return {
        'results': results,
        'dynamics_analyzer': dynamics_analyzer,
        'memorization_detector': memorization_detector,
        'learning_analysis': learning_analysis,
        'mem_analysis': mem_analysis,
        'model': model
    }


def create_comparison_plots(analyses, save_dir):
    """
    Create comparison plots across all phenomena.
    """
    print(f"\nCreating comparison plots...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    phenomena_names = list(analyses.keys())
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot 1: Training Accuracy
    ax = axes[0, 0]
    for i, (name, data) in enumerate(analyses.items()):
        epochs = range(len(data['results']['train_acc']))
        ax.plot(epochs, data['results']['train_acc'], 
               label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for i, (name, data) in enumerate(analyses.items()):
        epochs = range(len(data['results']['test_acc']))
        ax.plot(epochs, data['results']['test_acc'],
               label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Generalization Gap (Train - Test Accuracy)
    ax = axes[0, 2]
    for i, (name, data) in enumerate(analyses.items()):
        train_acc = np.array(data['results']['train_acc'])
        test_acc = np.array(data['results']['test_acc'])
        gap = train_acc - test_acc
        epochs = range(len(gap))
        ax.plot(epochs, gap, label=name, color=colors[i], linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap')
    ax.set_title('Generalization Gap (Train - Test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memorization Scores
    ax = axes[1, 0]
    for i, (name, data) in enumerate(analyses.items()):
        mem_detector = data['memorization_detector']
        if mem_detector.epoch_memorization:
            epochs = [m['epoch'] for m in mem_detector.epoch_memorization]
            scores = [m['memorization_score'] for m in mem_detector.epoch_memorization]
            ax.plot(epochs, scores, label=name, color=colors[i], 
                   linewidth=2, marker='o')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Memorization Score')
    ax.set_title('Memorization Score Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Forgetting Statistics
    ax = axes[1, 1]
    forgetting_counts = []
    for name in phenomena_names:
        stats = analyses[name]['learning_analysis']['forgetting_statistics']
        forgetting_counts.append(stats['examples_with_forgetting'])
    
    ax.bar(range(len(phenomena_names)), forgetting_counts, color=colors[:len(phenomena_names)])
    ax.set_xticks(range(len(phenomena_names)))
    ax.set_xticklabels(phenomena_names, rotation=45, ha='right')
    ax.set_ylabel('Examples with Forgetting')
    ax.set_title('Forgetting Events Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Transition Epochs
    ax = axes[1, 2]
    transition_epochs = []
    for name in phenomena_names:
        mem_analysis = analyses[name]['mem_analysis']
        epoch = mem_analysis.get('transition_epoch', 0)
        transition_epochs.append(epoch if epoch is not None else 0)
    
    ax.bar(range(len(phenomena_names)), transition_epochs, color=colors[:len(phenomena_names)])
    ax.set_xticks(range(len(phenomena_names)))
    ax.set_xticklabels(phenomena_names, rotation=45, ha='right')
    ax.set_ylabel('Transition Epoch')
    ax.set_title('Memorization Transition Points')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'phenomenon_comparison.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved comparison plots to {save_dir}")


def main():
    print("="*70)
    print("Multi-Phenomenon Analysis Comparison")
    print("="*70)
    print("\nThis example compares delayed generalization patterns across:")
    print("  1. Grokking (modular arithmetic)")
    print("  2. Simplicity Bias (spurious correlations)")
    print("  3. Normal Learning (baseline)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './analysis_results/multi_phenomenon'
    
    # Configuration
    n_train = 1000
    n_test = 200
    epochs = 100
    
    # Prepare datasets for each phenomenon
    print("\n[1/4] Preparing datasets...")
    
    # 1. Grokking dataset
    print("  Creating grokking dataset (modular arithmetic)...")
    grokking_train = create_grokking_dataset(n_train, modulo=97)
    grokking_test = create_grokking_dataset(n_test, modulo=97)
    
    # 2. Simplicity bias dataset
    print("  Creating simplicity bias dataset...")
    bias_train = create_simplicity_bias_dataset(n_train, bias_strength=0.9)
    bias_test = create_simplicity_bias_dataset(n_test, bias_strength=0.0)  # No bias in test
    
    # 3. Normal dataset
    print("  Creating normal dataset...")
    normal_train = create_normal_dataset(n_train)
    normal_test = create_normal_dataset(n_test)
    
    # Train and analyze each phenomenon
    print("\n[2/4] Training and analyzing models...")
    
    analyses = {}
    
    # Analyze grokking
    analyses['Grokking'] = train_and_analyze_phenomenon(
        phenomenon_name='Grokking',
        train_data=grokking_train,
        test_data=grokking_test,
        input_dim=97*2,
        output_dim=97,
        epochs=epochs,
        learning_rate=0.001,
        device=device
    )
    
    # Analyze simplicity bias
    analyses['Simplicity Bias'] = train_and_analyze_phenomenon(
        phenomenon_name='Simplicity Bias',
        train_data=bias_train,
        test_data=bias_test,
        input_dim=15,
        output_dim=2,
        epochs=epochs,
        learning_rate=0.001,
        device=device
    )
    
    # Analyze normal learning
    analyses['Normal'] = train_and_analyze_phenomenon(
        phenomenon_name='Normal Learning',
        train_data=normal_train,
        test_data=normal_test,
        input_dim=10,
        output_dim=2,
        epochs=epochs,
        learning_rate=0.001,
        device=device
    )
    
    # Compare results
    print("\n[3/4] Comparing phenomena...")
    print("\nSummary:")
    print("-" * 70)
    
    for name, data in analyses.items():
        print(f"\n{name}:")
        print(f"  Final Train Acc: {data['results']['train_acc'][-1]:.4f}")
        print(f"  Final Test Acc: {data['results']['test_acc'][-1]:.4f}")
        print(f"  Final Gap: {data['results']['train_acc'][-1] - data['results']['test_acc'][-1]:.4f}")
        print(f"  Transition Epoch: {data['mem_analysis'].get('transition_epoch', 'N/A')}")
        print(f"  Forgetting Events: {data['learning_analysis']['forgetting_statistics']['examples_with_forgetting']}")
        print(f"  Final Memorization: {data['mem_analysis'].get('final_memorization', 'N/A'):.4f}")
    
    # Create comparison visualizations
    print("\n[4/4] Creating visualizations...")
    create_comparison_plots(analyses, save_dir)
    
    # Save individual analyses
    for name, data in analyses.items():
        phenomenon_dir = Path(save_dir) / name.lower().replace(' ', '_')
        phenomenon_dir.mkdir(parents=True, exist_ok=True)
        
        data['dynamics_analyzer'].plot_learning_dynamics(save_dir=str(phenomenon_dir))
        data['memorization_detector'].plot_memorization(save_dir=str(phenomenon_dir))
    
    print("\n" + "="*70)
    print("Multi-Phenomenon Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    print("\nKey Observations:")
    print("  • Grokking shows delayed but eventual strong generalization")
    print("  • Simplicity Bias shows early overfitting to spurious features")
    print("  • Normal learning shows smooth convergence without delays")
    print("\n✓ Example completed successfully!")


if __name__ == '__main__':
    main()
