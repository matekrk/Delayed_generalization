#!/usr/bin/env python3
"""
Modular Arithmetic Dataset Generator for Delayed Generalization Research

This script generates datasets for studying grokking in modular arithmetic tasks.
Based on the experiments from "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
(Power et al., 2022)

Usage:
    python generate_data.py --prime 97 --operation addition --train_fraction 0.5
"""

import argparse
import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def generate_modular_arithmetic_data(
    prime: int, 
    operation: str, 
    train_fraction: float = 0.5,
    seed: int = 42
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """
    Generate modular arithmetic dataset for grokking experiments.
    
    Args:
        prime: Prime number for modular arithmetic (typically 97 or 113)
        operation: Arithmetic operation ('addition', 'subtraction', 'multiplication')
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_inputs, train_targets, test_inputs, test_targets)
        Each input is [a, op_token, b, equals_token]
        Each target is [a, op_token, b, equals_token, result]
    """
    set_seed(seed)
    
    # Define tokens
    EQUALS_TOKEN = prime
    if operation == 'addition':
        OP_TOKEN = prime + 1
    elif operation == 'subtraction':
        OP_TOKEN = prime + 2
    elif operation == 'multiplication':
        OP_TOKEN = prime + 3
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    # Generate all possible pairs
    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    
    # Compute results based on operation
    if operation == 'addition':
        results = [(a + b) % prime for a, b in all_pairs]
    elif operation == 'subtraction':
        results = [(a - b) % prime for a, b in all_pairs]
    elif operation == 'multiplication':
        results = [(a * b) % prime for a, b in all_pairs]
    
    # Create input and target sequences
    inputs = []
    targets = []
    
    for (a, b), result in zip(all_pairs, results):
        input_seq = [a, OP_TOKEN, b, EQUALS_TOKEN]
        target_seq = [a, OP_TOKEN, b, EQUALS_TOKEN, result]
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Split into train and test
    n_total = len(inputs)
    n_train = int(n_total * train_fraction)
    
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_inputs = [inputs[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    test_inputs = [inputs[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
    
    return train_inputs, train_targets, test_inputs, test_targets

def save_dataset(
    train_inputs: List[List[int]], 
    train_targets: List[List[int]], 
    test_inputs: List[List[int]], 
    test_targets: List[List[int]], 
    config: Dict[str, Any], 
    output_dir: str
):
    """Save dataset to files with metadata"""
    output_path = Path(output_dir)
    # create a subdirectory named by prime and operation (e.g. "./modular_arithmetic_data/prime_97_addition")
    subdir = f"prime_{config['prime']}_{config['operation']}_trainfrac_{config['train_fraction']}"
    output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save data
    np.save(output_path / "train_inputs.npy", np.array(train_inputs))
    np.save(output_path / "train_targets.npy", np.array(train_targets))
    np.save(output_path / "test_inputs.npy", np.array(test_inputs))
    np.save(output_path / "test_targets.npy", np.array(test_targets))
    
    # Save configuration and metadata
    metadata = {
        "config": config,
        "vocab_size": config["prime"] + 4,  # numbers + 3 special tokens
        "train_size": len(train_inputs),
        "test_size": len(test_inputs),
        "input_length": 4,
        "target_length": 5,
        "token_mapping": {
            "numbers": f"0 to {config['prime']-1}",
            "equals": config["prime"],
            "addition": config["prime"] + 1,
            "subtraction": config["prime"] + 2,
            "multiplication": config["prime"] + 3
        }
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Train size: {len(train_inputs)}")
    print(f"Test size: {len(test_inputs)}")
    print(f"Vocabulary size: {metadata['vocab_size']}")

def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load dataset from saved files"""
    data_path = Path(data_dir)
    
    train_inputs = np.load(data_path / "train_inputs.npy")
    train_targets = np.load(data_path / "train_targets.npy")
    test_inputs = np.load(data_path / "test_inputs.npy")
    test_targets = np.load(data_path / "test_targets.npy")
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return train_inputs, train_targets, test_inputs, test_targets, metadata

def main():
    parser = argparse.ArgumentParser(description="Generate modular arithmetic dataset for grokking")
    parser.add_argument("--prime", type=int, default=97, help="Prime number for modular arithmetic")
    parser.add_argument("--operation", choices=["addition", "subtraction", "multiplication"], 
                       default="addition", help="Arithmetic operation")
    parser.add_argument("--train_fraction", type=float, default=0.5, 
                       help="Fraction of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./modular_arithmetic_data", 
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    config = {
        "prime": args.prime,
        "operation": args.operation,
        "train_fraction": args.train_fraction,
        "seed": args.seed
    }
    
    print(f"Generating modular arithmetic dataset:")
    print(f"  Prime: {args.prime}")
    print(f"  Operation: {args.operation}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Seed: {args.seed}")
    
    # Generate dataset
    train_inputs, train_targets, test_inputs, test_targets = generate_modular_arithmetic_data(
        args.prime, args.operation, args.train_fraction, args.seed
    )
    
    # Save dataset
    save_dataset(train_inputs, train_targets, test_inputs, test_targets, config, args.output_dir)
    
    # Example usage demonstration
    print("\nExample sequences:")
    print("Input format: [a, operation_token, b, equals_token]")
    print("Target format: [a, operation_token, b, equals_token, result]")
    print(f"First training example:")
    print(f"  Input:  {train_inputs[0]}")
    print(f"  Target: {train_targets[0]}")
    
    print(f"\nToken meanings:")
    print(f"  Numbers: 0 to {args.prime-1}")
    print(f"  Equals token: {args.prime}")
    print(f"  Operation token ({args.operation}): {args.prime + 1}")

if __name__ == "__main__":
    main()