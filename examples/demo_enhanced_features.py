#!/usr/bin/env python3
"""
Example Usage of Enhanced Delayed Generalization Features

This script demonstrates how to use the enhanced features that were added to fix
the reported issues:

1. DelayedGeneralizationLogger with log_metrics method
2. Enhanced optimizers with additional features  
3. Advanced learning rate schedulers
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def example_logger_usage():
    """Example of using the enhanced DelayedGeneralizationLogger."""
    print("=== DelayedGeneralizationLogger Example ===")
    
    # Mock wandb for demonstration
    class MockWandB:
        @staticmethod
        def init(**kwargs): return None
        @staticmethod 
        def log(metrics, step=None): 
            print(f"Logged: {list(metrics.keys())}")
        @staticmethod
        def finish(): pass
    
    # Import and setup logger
    from utils.wandb_integration import delayed_generalization_logger
    delayed_generalization_logger.wandb = MockWandB()
    
    from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
    
    # Create logger for simplicity bias experiment
    logger = DelayedGeneralizationLogger(
        project_name="delayed-gen-demo",
        experiment_name="bias-example", 
        config={"lr": 1e-3, "epochs": 100},
        phenomenon_type="simplicity_bias"
    )
    
    # Use the fixed log_metrics method
    logger.log_metrics({
        "custom_metric": 0.85,
        "bias_strength": 0.3,
        "model_complexity": 1000
    }, step=50)
    
    # Use standard epoch logging
    logger.log_epoch_metrics(
        epoch=50,
        train_loss=0.2,
        test_loss=0.3,
        train_acc=0.95,
        test_acc=0.85,
        # Additional phenomenon-specific metrics
        group_accuracies={"majority": 0.95, "minority": 0.75},
        bias_score=0.3
    )
    
    print("✓ Logger demonstrates all fixed functionality")


def example_enhanced_optimizers():
    """Example of using enhanced optimizers."""
    print("\n=== Enhanced Optimizers Example ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Softmax(dim=1)
    )
    
    # Use EnhancedAdamW with additional features
    from optimization.enhanced_optimizers import EnhancedAdamW
    
    optimizer = EnhancedAdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
        grad_clip_norm=1.0,          # Gradient clipping
        adaptive_weight_decay=True,   # Adaptive weight decay
        warmup_steps=100,            # Learning rate warmup
        log_grad_stats=True          # Gradient statistics
    )
    
    print("✓ EnhancedAdamW created with additional features")
    
    # Use factory function for easy setup
    from optimization.enhanced_optimizers import create_optimizer_with_scheduler
    
    opt, scheduler = create_optimizer_with_scheduler(
        model,
        optimizer_type='enhanced_adamw',
        learning_rate=1e-3,
        scheduler_type='warmup_cosine',
        total_steps=1000,
        warmup_steps=100
    )
    
    print("✓ Optimizer + scheduler created via factory function")
    
    # Demonstrate a training step with enhanced features
    dummy_input = torch.randn(32, 100)
    dummy_target = torch.randint(0, 10, (32,))
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_target)
    loss.backward()
    optimizer.step()  # This will use gradient clipping and logging
    
    # Get optimizer statistics
    stats = optimizer.get_stats()
    print(f"✓ Optimizer stats: {len(stats['grad_norms'])} gradient norms recorded")


def example_advanced_schedulers():
    """Example of using advanced learning rate schedulers."""
    print("\n=== Advanced LR Schedulers Example ===")
    
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Phase transition scheduler that adapts to training dynamics
    from optimization.lr_schedulers import PhaseTransitionScheduler
    
    phase_scheduler = PhaseTransitionScheduler(
        optimizer,
        patience=50,      # Wait 50 epochs before reducing LR
        factor=0.5,       # Reduce LR by half
        threshold=1e-4    # Improvement threshold
    )
    
    print("✓ PhaseTransitionScheduler created")
    
    # Adaptive scheduler that switches strategies
    from optimization.lr_schedulers import AdaptiveLRScheduler
    
    adaptive_scheduler = AdaptiveLRScheduler(
        optimizer,
        total_steps=1000,
        base_strategy='cosine',
        adaptation_window=100
    )
    
    print("✓ AdaptiveLRScheduler created")
    
    # Create phenomenon-specific scheduler
    from optimization.lr_schedulers import create_scheduler_for_phenomenon
    
    grokking_scheduler = create_scheduler_for_phenomenon(
        optimizer,
        phenomenon_type='grokking',
        total_steps=5000,
        warmup_steps=500
    )
    
    print("✓ Phenomenon-specific scheduler created for grokking")


def example_training_integration():
    """Example of using enhanced features in training."""
    print("\n=== Training Integration Example ===")
    
    # Create dummy data
    X = torch.randn(100, 3, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Create model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    )
    
    # Use enhanced trainer (will automatically use enhanced optimizers)
    from phenomena.simplicity_bias.colored_mnist.train_colored_mnist import SimplicityBiasTrainer
    
    trainer = SimplicityBiasTrainer(
        model=model,
        train_loader=dataloader,
        test_loader=dataloader,
        device=torch.device('cpu'),
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    print("✓ Training script uses enhanced optimizers automatically")
    print(f"   Optimizer type: {type(trainer.optimizer).__name__}")


def main():
    """Run all examples."""
    print("ENHANCED DELAYED GENERALIZATION FEATURES DEMO")
    print("=" * 60)
    print("This demonstrates the fixes for the reported issues:")
    print("1. DelayedGeneralizationLogger.log_metrics method")
    print("2. Enhanced optimizers in optimization directory")
    print("3. Robust import handling in training scripts")
    print("=" * 60)
    
    try:
        example_logger_usage()
        example_enhanced_optimizers()
        example_advanced_schedulers()
        example_training_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("All reported issues have been fixed and enhanced.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()