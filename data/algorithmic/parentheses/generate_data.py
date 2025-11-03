#!/usr/bin/env python3
"""
Parentheses Dataset Generator for Delayed Generalization Research

This script generates datasets for studying grokking and delayed generalization 
in parentheses matching tasks as described in the paper:
"Grokking Transitions in Language Models" (arXiv:2507.06445)

Two main tasks:
1. Equal Count: Sequences with the same number of open and closed brackets (balanced count)
2. Nested: Properly nested/matched parentheses sequences (valid Dyck language)

Usage:
    python generate_data.py --task nested --length 10 --n_types 2 --train_fraction 0.5
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


def get_bracket_pairs(n_types: int) -> List[Tuple[str, str]]:
    """
    Get bracket pairs based on n_types.
    
    Args:
        n_types: Number of bracket types (1-4)
            1: ()
            2: (), []
            3: (), [], {}
            4: (), [], {}, <>
    
    Returns:
        List of (open, close) bracket pairs
    """
    all_pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    if n_types < 1 or n_types > 4:
        raise ValueError(f"n_types must be between 1 and 4, got {n_types}")
    return all_pairs[:n_types]


def is_valid_nested(sequence: List[str], bracket_pairs: List[Tuple[str, str]]) -> bool:
    """
    Check if a sequence is properly nested (valid Dyck language).
    
    Args:
        sequence: List of bracket characters
        bracket_pairs: List of (open, close) bracket pairs
    
    Returns:
        True if sequence is properly nested
    """
    stack = []
    open_to_close = {open_br: close_br for open_br, close_br in bracket_pairs}
    close_to_open = {close_br: open_br for open_br, close_br in bracket_pairs}
    
    for char in sequence:
        if char in open_to_close:
            stack.append(char)
        elif char in close_to_open:
            if not stack or stack[-1] != close_to_open[char]:
                return False
            stack.pop()
        else:
            return False  # Invalid character
    
    return len(stack) == 0


def has_equal_count(sequence: List[str], bracket_pairs: List[Tuple[str, str]]) -> bool:
    """
    Check if a sequence has equal count of open and closed brackets for each type.
    
    Args:
        sequence: List of bracket characters
        bracket_pairs: List of (open, close) bracket pairs
    
    Returns:
        True if sequence has equal counts
    """
    for open_br, close_br in bracket_pairs:
        if sequence.count(open_br) != sequence.count(close_br):
            return False
    return True


def generate_random_sequence(length: int, bracket_pairs: List[Tuple[str, str]]) -> List[str]:
    """Generate a random sequence of brackets"""
    all_brackets = []
    for open_br, close_br in bracket_pairs:
        all_brackets.extend([open_br, close_br])
    return [random.choice(all_brackets) for _ in range(length)]


def generate_equal_count_sequence(length: int, bracket_pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Generate a sequence with equal count of open and closed brackets.
    For simplicity, generates equal pairs for each bracket type.
    """
    if length % (2 * len(bracket_pairs)) != 0:
        # Adjust length to be divisible
        length = (length // (2 * len(bracket_pairs))) * (2 * len(bracket_pairs))
    
    pairs_per_type = length // (2 * len(bracket_pairs))
    sequence = []
    
    for open_br, close_br in bracket_pairs:
        sequence.extend([open_br] * pairs_per_type)
        sequence.extend([close_br] * pairs_per_type)
    
    random.shuffle(sequence)
    return sequence


def generate_nested_sequence(length: int, bracket_pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Generate a properly nested sequence using recursive construction.
    """
    if length == 0:
        return []
    if length % 2 != 0:
        length -= 1  # Make even
    
    sequence = []
    remaining = length
    
    while remaining > 0:
        if remaining < 2:
            break
        
        # Choose a random bracket type
        open_br, close_br = random.choice(bracket_pairs)
        
        # Decide how many pairs to nest
        max_pairs = remaining // 2
        if max_pairs > 1:
            # Could nest inside
            inner_length = random.randint(0, min(max_pairs - 1, 10) * 2)
            if inner_length % 2 != 0:
                inner_length -= 1
        else:
            inner_length = 0
        
        # Add opening bracket
        sequence.append(open_br)
        
        # Add nested content
        if inner_length > 0:
            nested = generate_nested_sequence(inner_length, bracket_pairs)
            sequence.extend(nested)
        
        # Add closing bracket
        sequence.append(close_br)
        
        remaining -= (2 + inner_length)
    
    return sequence


def generate_parentheses_dataset(
    task: str,
    length: int,
    n_types: int = 1,
    n_samples: int = 10000,
    train_fraction: float = 0.5,
    seed: int = 42
) -> Tuple[List[List[int]], List[int], List[List[int]], List[int]]:
    """
    Generate parentheses dataset for grokking experiments.
    
    Args:
        task: Task type ('equal_count' or 'nested')
        length: Length of sequences
        n_types: Number of bracket types (1-4)
        n_samples: Total number of samples to generate
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_inputs, train_targets, test_inputs, test_targets)
        Each input is a sequence of token ids
        Each target is 0 (negative) or 1 (positive)
    """
    set_seed(seed)
    
    bracket_pairs = get_bracket_pairs(n_types)
    
    # Create token mapping
    all_brackets = []
    for open_br, close_br in bracket_pairs:
        all_brackets.extend([open_br, close_br])
    
    char_to_token = {char: idx for idx, char in enumerate(all_brackets)}
    
    # Generate dataset
    inputs = []
    targets = []
    
    positive_samples = n_samples // 2
    negative_samples = n_samples - positive_samples
    
    # Generate positive samples
    for _ in range(positive_samples):
        if task == 'equal_count':
            sequence = generate_equal_count_sequence(length, bracket_pairs)
        elif task == 'nested':
            sequence = generate_nested_sequence(length, bracket_pairs)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Convert to token ids
        token_ids = [char_to_token[char] for char in sequence]
        inputs.append(token_ids)
        targets.append(1)  # Positive
    
    # Generate negative samples
    attempts = 0
    max_attempts = negative_samples * 10
    
    while len([t for t in targets if t == 0]) < negative_samples and attempts < max_attempts:
        sequence = generate_random_sequence(length, bracket_pairs)
        
        # Check if it's actually negative
        is_positive = False
        if task == 'equal_count':
            is_positive = has_equal_count(sequence, bracket_pairs)
        elif task == 'nested':
            is_positive = is_valid_nested(sequence, bracket_pairs)
        
        if not is_positive:
            token_ids = [char_to_token[char] for char in sequence]
            inputs.append(token_ids)
            targets.append(0)  # Negative
        
        attempts += 1
    
    # If we couldn't generate enough negative samples, warn
    actual_negative = len([t for t in targets if t == 0])
    if actual_negative < negative_samples:
        print(f"Warning: Could only generate {actual_negative} negative samples out of {negative_samples} requested")
    
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
    train_targets: List[int],
    test_inputs: List[List[int]],
    test_targets: List[int],
    config: Dict[str, Any],
    output_dir: str
):
    """Save dataset to files with metadata"""
    output_path = Path(output_dir)
    
    # Create subdirectory based on configuration
    subdir = f"{config['task']}_len{config['length']}_types{config['n_types']}_trainfrac_{config['train_fraction']}"
    output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save data
    np.save(output_path / "train_inputs.npy", np.array(train_inputs))
    np.save(output_path / "train_targets.npy", np.array(train_targets))
    np.save(output_path / "test_inputs.npy", np.array(test_inputs))
    np.save(output_path / "test_targets.npy", np.array(test_targets))
    
    # Create token mapping
    bracket_pairs = get_bracket_pairs(config['n_types'])
    all_brackets = []
    for open_br, close_br in bracket_pairs:
        all_brackets.extend([open_br, close_br])
    
    token_mapping = {idx: char for idx, char in enumerate(all_brackets)}
    
    # Save configuration and metadata
    metadata = {
        "config": config,
        "vocab_size": len(all_brackets),
        "train_size": len(train_inputs),
        "test_size": len(test_inputs),
        "sequence_length": config['length'],
        "n_classes": 2,
        "class_names": ["invalid", "valid"],
        "token_mapping": token_mapping,
        "bracket_pairs": [f"{o}{c}" for o, c in bracket_pairs],
        "task_description": {
            "equal_count": "Sequences with equal number of open and closed brackets",
            "nested": "Properly nested/matched bracket sequences (Dyck language)"
        }[config['task']]
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Train size: {len(train_inputs)}")
    print(f"Test size: {len(test_inputs)}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Sequence length: {config['length']}")
    print(f"Train positive samples: {sum(train_targets)}/{len(train_targets)}")
    print(f"Test positive samples: {sum(test_targets)}/{len(test_targets)}")


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
    parser = argparse.ArgumentParser(description="Generate parentheses dataset for grokking")
    parser.add_argument("--task", choices=["equal_count", "nested"], 
                       default="nested", help="Task type")
    parser.add_argument("--length", type=int, default=10, 
                       help="Sequence length")
    parser.add_argument("--n_types", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Number of bracket types (1=(), 2=()[], 3=()[]{}, 4=()[]{}<>)")
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--train_fraction", type=float, default=0.5,
                       help="Fraction of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./parentheses_data",
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    config = {
        "task": args.task,
        "length": args.length,
        "n_types": args.n_types,
        "n_samples": args.n_samples,
        "train_fraction": args.train_fraction,
        "seed": args.seed
    }
    
    print(f"Generating parentheses dataset:")
    print(f"  Task: {args.task}")
    print(f"  Sequence length: {args.length}")
    print(f"  Bracket types: {args.n_types}")
    print(f"  Total samples: {args.n_samples}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Seed: {args.seed}")
    
    # Generate dataset
    train_inputs, train_targets, test_inputs, test_targets = generate_parentheses_dataset(
        args.task, args.length, args.n_types, args.n_samples, args.train_fraction, args.seed
    )
    
    # Save dataset
    save_dataset(train_inputs, train_targets, test_inputs, test_targets, config, args.output_dir)
    
    # Example sequences
    print("\nExample sequences:")
    bracket_pairs = get_bracket_pairs(args.n_types)
    all_brackets = []
    for open_br, close_br in bracket_pairs:
        all_brackets.extend([open_br, close_br])
    
    print(f"First training example (class {train_targets[0]}):")
    print(f"  Token IDs: {train_inputs[0]}")
    print(f"  Brackets: {''.join([all_brackets[tid] for tid in train_inputs[0]])}")
    
    if len(train_inputs) > 1:
        print(f"Second training example (class {train_targets[1]}):")
        print(f"  Token IDs: {train_inputs[1]}")
        print(f"  Brackets: {''.join([all_brackets[tid] for tid in train_inputs[1]])}")


if __name__ == "__main__":
    main()
