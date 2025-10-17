#!/usr/bin/env python3
"""
Integration Example: Enhanced Features with Unified Notebook

This script demonstrates how to integrate the enhanced analysis features
with the existing neural_network_training_dynamics_unified.ipynb workflow.

It provides all the missing features requested:
1. Difficulty over time tracking
2. Separate accuracy evolution (train/test) 
3. Forgetting events analysis with color-coded plots
4. Distribution of forgetting events
5. Class-wise accuracy and loss evolution
6. Highlighting opposing gradient pairs during training
7. Loss change highlighting (green for positive, red for negative)
8. More informative .gif animations with bar plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from enhanced_analysis_features import EnhancedTrainingAnalyzer

# Configuration matching the unified notebook
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = './data'  # Adjust as needed
SAVE_ANIMATIONS = True
DENSE_SAMPLING = True
N_EXAMPLES_TRACK = 500
ANIMATION_FPS = 10

class SimpleCNN(nn.Module):
    """Simple CNN matching the unified notebook architecture"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.device = DEVICE
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        
        # Fully connected layers
        features = F.relu(self.fc1(x))
        features = self.dropout(features)
        output = self.fc2(features)
        
        if return_features:
            return output, features
        return output


def load_cifar10_data(batch_size=64, num_workers=2):
    """Load CIFAR-10 data with transforms"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform_test
    )
    
    # Use subset for faster demonstration
    train_subset = torch.utils.data.Subset(trainset, range(10000))
    test_subset = torch.utils.data.Subset(testset, range(2000))
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def enhanced_training_loop():
    """Enhanced training loop with comprehensive analysis"""
    
    print("Starting Enhanced Training Dynamics Analysis")
    print("=" * 60)
    
    # Load data
    print("Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10_data(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    model = SimpleCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedTrainingAnalyzer(
        num_classes=10, 
        max_examples_track=N_EXAMPLES_TRACK
    )
    
    print(f"Enhanced analyzer initialized (tracking {N_EXAMPLES_TRACK} examples)")
    
    # Training loop with enhanced tracking
    num_epochs = 20  # Reduced for demonstration
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss {loss.item():.4f}, '
                      f'Acc {100.*correct/total:.2f}%')
        
        # Enhanced dynamics tracking
        print("  Tracking enhanced dynamics...")
        analyzer.track_epoch(
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer
        )
        
        # Generate visualizations periodically
        if epoch % 5 == 4:  # Every 5 epochs
            print("  Generating intermediate visualizations...")
            analyzer.plot_accuracy_evolution()
            analyzer.plot_difficulty_evolution()
            if analyzer.opposing_pairs:
                analyzer.plot_opposing_signals()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED - Generating Comprehensive Report")
    print("=" * 60)
    
    # Generate final comprehensive analysis
    report_dir = analyzer.generate_comprehensive_report("./enhanced_analysis_results")
    
    # Create final enhanced animation
    if SAVE_ANIMATIONS:
        print("\nCreating enhanced animation...")
        anim = analyzer.create_enhanced_animation(
            save_path="./enhanced_analysis_results/comprehensive_training_dynamics.gif",
            fps=ANIMATION_FPS
        )
        
        print("Enhanced animation created with:")
        print("  ✓ Bar plots for each epoch evaluation")
        print("  ✓ Class-wise accuracy breakdown")
        print("  ✓ Dynamic training timeline")
        print("  ✓ Loss change highlighting")
        print("  ✓ Opposing pairs detection")
    
    print(f"\nAll enhanced analysis results saved to: {report_dir}")
    print("\nEnhanced features implemented:")
    print("  ✓ Difficulty over time tracking")
    print("  ✓ Separate accuracy evolution (train/test)")
    print("  ✓ Forgetting events analysis with color-coded plots")
    print("  ✓ Distribution of forgetting events")
    print("  ✓ Class-wise accuracy and loss evolution")
    print("  ✓ Opposing gradient pairs highlighting")
    print("  ✓ Loss change highlighting (green/red)")
    print("  ✓ Informative .gif animations with bar plots")
    
    return analyzer, report_dir


def integration_with_existing_notebook():
    """Show how to integrate with existing notebook code"""
    
    print("\nIntegration with Existing Unified Notebook:")
    print("=" * 50)
    
    integration_code = '''
# Add this to your existing unified notebook:

# 1. Import the enhanced analyzer
from enhanced_analysis_features import EnhancedTrainingAnalyzer

# 2. Initialize in your setup cell
analyzer = EnhancedTrainingAnalyzer(num_classes=10, max_examples_track=1000)

# 3. Add to your training loop (replace existing tracking):
for epoch in range(num_epochs):
    # ... your existing training code ...
    
    # Replace/enhance your existing tracking with:
    analyzer.track_epoch(model, train_loader, test_loader, criterion, epoch, optimizer)
    
    # Generate enhanced visualizations
    if epoch % 10 == 9:
        analyzer.plot_accuracy_evolution()
        analyzer.plot_difficulty_evolution() 
        analyzer.plot_opposing_signals()

# 4. Generate final comprehensive report
report_dir = analyzer.generate_comprehensive_report("./enhanced_results")

# 5. Create enhanced animation
anim = analyzer.create_enhanced_animation("./enhanced_training.gif", fps=10)
'''
    
    print(integration_code)
    
    print("\nKey Enhancements Added:")
    print("  • Difficulty tracking with color-coded forgetting events")
    print("  • Separate train/test accuracy plots with class breakdown")
    print("  • Opposing gradient pair detection and visualization")
    print("  • Loss change highlighting (green=decrease, red=increase)")
    print("  • Enhanced animations with bar plots per epoch")
    print("  • Comprehensive analysis reports")
    
    print("\nAll features from enhanced_neural_network_training_dynamics.ipynb")
    print("are now available in the unified notebook workflow!")


if __name__ == "__main__":
    # Demonstrate the enhanced features
    analyzer, report_dir = enhanced_training_loop()
    
    # Show integration instructions
    integration_with_existing_notebook()
    
    print(f"\n🎉 Enhanced analysis complete! Check {report_dir} for all results.")