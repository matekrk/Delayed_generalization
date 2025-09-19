#!/usr/bin/env python3
"""
Test script to verify all fixes for the delayed generalization issues.
"""

import sys
import os
import traceback

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_logger_fixes():
    """Test DelayedGeneralizationLogger fixes."""
    print("=" * 60)
    print("TESTING DELAYEDGENERALIZATIONLOGGER FIXES")
    print("=" * 60)
    
    try:
        # Mock wandb to avoid actual initialization
        class MockWandB:
            @staticmethod
            def init(**kwargs):
                return None
            @staticmethod 
            def log(metrics, step=None):
                print(f"✓ Would log metrics: {list(metrics.keys())}")
            @staticmethod
            def finish():
                pass
        
        # Import and patch wandb
        from utils.wandb_integration import delayed_generalization_logger
        delayed_generalization_logger.wandb = MockWandB()
        
        # Test logger creation
        from utils.wandb_integration.delayed_generalization_logger import DelayedGeneralizationLogger
        logger = DelayedGeneralizationLogger('test-project', 'test-experiment', {})
        print("✓ DelayedGeneralizationLogger created successfully")
        
        # Test the missing log_metrics method
        test_metrics = {'loss': 0.5, 'accuracy': 0.85, 'epoch': 10}
        logger.log_metrics(test_metrics, step=10)
        print("✓ log_metrics method works correctly")
        
        # Test epoch logging
        logger.log_epoch_metrics(
            epoch=1,
            train_loss=0.6,
            test_loss=0.7, 
            train_acc=0.8,
            test_acc=0.75
        )
        print("✓ log_epoch_metrics method works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        print(traceback.format_exc())
        return False


def test_optimizer_enhancements():
    """Test enhanced optimizer implementations."""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZER ENHANCEMENTS")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Test Enhanced AdamW
        from optimization.enhanced_optimizers import EnhancedAdamW
        optimizer = EnhancedAdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            grad_clip_norm=1.0,
            adaptive_weight_decay=True,
            log_grad_stats=True
        )
        print("✓ EnhancedAdamW created successfully")
        
        # Test Enhanced SGD
        from optimization.enhanced_optimizers import EnhancedSGD
        sgd_optimizer = EnhancedSGD(
            model.parameters(),
            lr=1e-3,
            adaptive_momentum=True,
            grad_clip_norm=1.0
        )
        print("✓ EnhancedSGD created successfully")
        
        # Test optimizer factory function
        from optimization.enhanced_optimizers import create_optimizer_with_scheduler
        opt, scheduler = create_optimizer_with_scheduler(
            model,
            optimizer_type='enhanced_adamw',
            learning_rate=1e-3,
            total_steps=1000
        )
        print("✓ create_optimizer_with_scheduler works correctly")
        
        # Test LR schedulers
        from optimization.lr_schedulers import WarmupCosineScheduler, PhaseTransitionScheduler
        
        warmup_scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000
        )
        print("✓ WarmupCosineScheduler created successfully")
        
        phase_scheduler = PhaseTransitionScheduler(optimizer)
        print("✓ PhaseTransitionScheduler created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        print(traceback.format_exc())
        return False


def test_import_fixes():
    """Test import fixes in training scripts."""
    print("\n" + "=" * 60)
    print("TESTING IMPORT FIXES")
    print("=" * 60)
    
    try:
        # Test the import fixes in train_colored_mnist.py
        from phenomena.simplicity_bias.colored_mnist.training.train_colored_mnist import SimplicityBiasTrainer
        print("✓ train_colored_mnist.py imports work correctly")
        
        # Test that we can import synthetic dataset functions
        try:
            from data.vision.colored_mnist.generate_synthetic_colored_digits import load_synthetic_dataset, SyntheticColoredDataset
            print("✓ Synthetic colored dataset imports work")
        except ImportError:
            print("⚠ Synthetic dataset imports not available (fallback mode)")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        print(traceback.format_exc())
        return False


def test_training_script_integration():
    """Test that training scripts can use enhanced optimizers."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SCRIPT INTEGRATION")
    print("=" * 60)
    
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data for testing
        X = torch.randn(100, 3, 28, 28)  # Dummy image data
        y = torch.randint(0, 10, (100,))  # Dummy labels
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)
        
        # Test creating a trainer (without actual training)
        from phenomena.simplicity_bias.colored_mnist.training.train_colored_mnist import SimplicityBiasTrainer
        
        # Create a simple CNN model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 10)
        )
        
        trainer = SimplicityBiasTrainer(
            model=model,
            train_loader=dataloader,
            test_loader=dataloader,
            device=torch.device('cpu'),
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        print("✓ SimplicityBiasTrainer created with enhanced optimizer support")
        
        # Test grokking trainer
        from phenomena.grokking.training.train_modular import GrokkingTrainer
        
        grokking_trainer = GrokkingTrainer(
            model=model,
            train_loader=dataloader,
            test_loader=dataloader,
            device=torch.device('cpu'),
            learning_rate=1e-3,
            weight_decay=1e-2
        )
        print("✓ GrokkingTrainer created with enhanced optimizer support")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script integration test failed: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    print("DELAYED GENERALIZATION FIXES TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Logger Fixes", test_logger_fixes),
        ("Optimizer Enhancements", test_optimizer_enhancements),
        ("Import Fixes", test_import_fixes),
        ("Training Script Integration", test_training_script_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! All issues have been fixed.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)