#!/usr/bin/env python3
"""
Example: End-to-End Parentheses Grokking Experiment

This script demonstrates a complete grokking experiment on the parentheses dataset.
It generates data, trains a model, and shows the delayed generalization phenomenon.

Usage:
    python example_parentheses_grokking.py
"""

import sys
import os
from pathlib import Path

# Add repository to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.algorithmic.parentheses.generate_data import (
    generate_parentheses_dataset,
    save_dataset
)


def run_example():
    """Run a complete parentheses grokking experiment"""
    
    print("=" * 60)
    print("Parentheses Grokking Example")
    print("=" * 60)
    
    # Step 1: Generate dataset
    print("\n1. Generating dataset...")
    print("   Task: nested")
    print("   Sequence length: 8")
    print("   Bracket types: 1")
    print("   Samples: 2000")
    
    train_inputs, train_targets, test_inputs, test_targets = generate_parentheses_dataset(
        task='nested',
        length=8,
        n_types=1,
        n_samples=2000,
        train_fraction=0.5,
        seed=42
    )
    
    print(f"   ✓ Generated {len(train_inputs)} training samples")
    print(f"   ✓ Generated {len(test_inputs)} test samples")
    
    # Step 2: Save dataset
    output_dir = "/tmp/example_parentheses"
    config = {
        'task': 'nested',
        'length': 8,
        'n_types': 1,
        'n_samples': 2000,
        'train_fraction': 0.5,
        'seed': 42
    }
    
    print(f"\n2. Saving dataset to {output_dir}...")
    save_dataset(train_inputs, train_targets, test_inputs, test_targets, config, output_dir)
    print("   ✓ Dataset saved")
    
    # Step 3: Show examples
    print("\n3. Sample sequences:")
    
    # Convert token IDs back to brackets
    brackets = ['(', ')']
    
    for i in range(3):
        seq = ''.join([brackets[tid] for tid in train_inputs[i]])
        label = "valid" if train_targets[i] == 1 else "invalid"
        print(f"   Example {i+1}: {seq:12s} -> {label}")
    
    # Step 4: Training instructions
    print("\n4. To train a model on this dataset:")
    print(f"\n   python phenomena/grokking/parentheses/training/train_parentheses.py \\")
    print(f"       --data_dir {output_dir}/nested_len8_types1_trainfrac_0.5 \\")
    print(f"       --epochs 10000 \\")
    print(f"       --batch_size 512 \\")
    print(f"       --learning_rate 1e-3 \\")
    print(f"       --weight_decay 1e-2")
    
    # Step 5: Expected behavior
    print("\n5. Expected grokking behavior:")
    print("   - Epochs 0-2000: Train accuracy increases, test stays ~50%")
    print("   - Epochs 2000-3000: Sudden jump in test accuracy (grokking!)")
    print("   - Epochs 3000+: Both accuracies reach 90%+")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the training command above")
    print("2. Monitor training curves for grokking")
    print("3. Try different tasks (equal_count) and bracket types (n_types=2,3,4)")
    print("4. Experiment with hyperparameters (weight_decay is crucial!)")


if __name__ == "__main__":
    run_example()
