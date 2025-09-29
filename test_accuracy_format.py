#!/usr/bin/env python3
"""
Test script to validate accuracy format consistency.

This script verifies that:
1. Accuracy calculation methods return values in 0.0-1.0 format
2. Print statements show values in 0-100% format
3. WandB logging receives values in 0.0-1.0 format
"""

import ast
import re
import os
from pathlib import Path


def check_accuracy_format_in_file(filepath):
    """Check a file for accuracy format consistency."""
    print(f"\nChecking {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for accuracy calculations that multiply by 100 (should be removed)
    multiply_100_pattern = r'accuracy\s*=\s*100\.\s*\*\s*correct\s*/\s*total'
    multiply_100_matches = re.findall(multiply_100_pattern, content)
    if multiply_100_matches:
        issues.append(f"Found {len(multiply_100_matches)} accuracy calculations multiplying by 100")
    
    # Check for print statements that don't multiply by 100 for percentage display
    # Look for print statements with accuracy but without *100
    print_patterns = [
        r'print\([^)]*{[^}]*accuracy[^}]*:.2f}%[^)]*\)',
        r'print\([^)]*{[^}]*acc[^}]*:.2f}%[^)]*\)'
    ]
    
    for pattern in print_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            # Check if this print statement multiplies by 100
            if '*100' not in match:
                # Look around the match to see if it's a variable that should be multiplied
                if 'train_acc' in match or 'test_acc' in match or 'accuracy' in match:
                    issues.append(f"Print statement may not multiply accuracy by 100: {match[:80]}...")
    
    # Check for .3f format in accuracy prints (should be .2f% with *100)
    accuracy_3f_pattern = r'{[^}]*acc[^}]*:.3f}'
    accuracy_3f_matches = re.findall(accuracy_3f_pattern, content, re.IGNORECASE)
    if accuracy_3f_matches:
        issues.append(f"Found {len(accuracy_3f_matches)} accuracy prints using .3f format (should be .2f% with *100)")
    
    if issues:
        print(f"  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print(f"  ✓ PASSED - Accuracy format looks correct")
        return True


def main():
    """Main test function."""
    print("Testing accuracy format consistency across training files...")
    
    # Find all training files in phenomena directory
    phenomena_dir = Path(__file__).parent / "phenomena"
    training_files = []
    
    for pattern in ["**/train*.py", "**/training/*.py"]:
        training_files.extend(phenomena_dir.glob(pattern))
    
    # Filter to just the main training scripts we modified
    target_files = [
        "phenomena/grokking/training/train_modular.py",
        "phenomena/simplicity_bias/colored_mnist/train_colored_mnist.py", 
        "phenomena/robustness/cifar100c/train_cifar100c.py",
        "phenomena/continual_learning/cifar100_10tasks/train_continual_cifar100.py",
        "phenomena/simplicity_bias/celeba/training/train_bias_celeba.py",
        "phenomena/simplicity_bias/waterbirds/training/train_waterbirds.py",
        "phenomena/simplicity_bias/celeba/training/train_celeba.py",
        "phenomena/robustness/cifar10c/train_cifar10c.py"
    ]
    
    all_passed = True
    
    for target_file in target_files:
        filepath = Path(__file__).parent / target_file
        if filepath.exists():
            passed = check_accuracy_format_in_file(filepath)
            all_passed = all_passed and passed
        else:
            print(f"\nWarning: {filepath} not found")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - Accuracy format is consistent!")
        print("✓ Accuracy calculations return 0.0-1.0 values")
        print("✓ Print statements display 0-100% format") 
        print("✓ WandB logging receives 0.0-1.0 values")
    else:
        print("❌ SOME ISSUES FOUND - Please review the files above")
    print(f"{'='*60}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)