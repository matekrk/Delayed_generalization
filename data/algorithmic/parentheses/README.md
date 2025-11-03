# Parentheses Matching Dataset

A synthetic dataset for studying grokking and delayed generalization in parentheses matching tasks.

## Overview

This dataset implements two related classification tasks based on bracket sequences, designed to exhibit strong grokking phenomena during training. The tasks are inspired by research on delayed generalization in language models.

## Tasks

### 1. Equal Count
Classify whether a sequence has the same number of open and closed brackets for each bracket type.

**Example (n_types=1):**
- ✅ Valid: `())(()` - 3 open `(` and 3 closed `)`
- ❌ Invalid: `()()(` - 4 open `(` and 2 closed `)`

### 2. Nested (Dyck Language)
Classify whether a sequence forms properly nested/matched brackets (valid Dyck language).

**Example (n_types=1):**
- ✅ Valid: `(()())` - properly nested
- ❌ Invalid: `())(()` - not properly nested

The nested task is significantly harder and shows more pronounced grokking behavior.

## Bracket Types

Control complexity with the `n_types` parameter:

| n_types | Brackets | Vocabulary Size |
|---------|----------|-----------------|
| 1 | `()` | 2 |
| 2 | `()` `[]` | 4 |
| 3 | `()` `[]` `{}` | 6 |
| 4 | `()` `[]` `{}` `<>` | 8 |

## Quick Start

### Generate Dataset

```bash
python generate_data.py \
    --task nested \
    --length 10 \
    --n_types 1 \
    --n_samples 10000 \
    --train_fraction 0.5 \
    --output_dir ./parentheses_data
```

### Train Model

```bash
python ../../../phenomena/grokking/parentheses/training/train_parentheses.py \
    --data_dir ./parentheses_data/nested_len10_types1_trainfrac_0.5 \
    --epochs 10000 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2
```

### Or Use Run Script

```bash
cd ../../../phenomena/grokking/parentheses
./run_parentheses_experiment.sh nested 1 10 10000
```

## Dataset Structure

Generated datasets have the following structure:

```
parentheses_data/
└── nested_len10_types1_trainfrac_0.5/
    ├── train_inputs.npy      # Training sequences [N, seq_len]
    ├── train_targets.npy     # Training labels [N] (0=invalid, 1=valid)
    ├── test_inputs.npy       # Test sequences
    ├── test_targets.npy      # Test labels
    └── metadata.json         # Configuration and statistics
```

## Parameters

### Data Generation

- `--task`: `equal_count` or `nested`
- `--length`: Sequence length (even numbers recommended)
- `--n_types`: Number of bracket types (1-4)
- `--n_samples`: Total samples to generate (split by train_fraction)
- `--train_fraction`: Training data fraction (default: 0.5)
- `--seed`: Random seed (default: 42)

### Training

See [training documentation](../../../phenomena/grokking/parentheses/README.md) for model architecture and hyperparameters.

## Grokking Behavior

Both tasks exhibit classic grokking:

1. **Memorization (0-3000 epochs)**: High train accuracy, low test accuracy
2. **Transition (sudden)**: Test accuracy jumps from ~50% to 80%+
3. **Generalization (3000+ epochs)**: Both accuracies reach 90%+

The nested task typically shows:
- Later grokking onset (more epochs needed)
- Sharper transition (more sudden jump)
- Higher sensitivity to weight decay

## Recommended Configurations

### Easy (Quick Debugging)
```bash
--task equal_count --length 8 --n_types 1 --n_samples 5000 --epochs 5000
```

### Standard (Paper Replication)
```bash
--task nested --length 10 --n_types 1 --n_samples 10000 --epochs 10000
```

### Hard (Complex Structure)
```bash
--task nested --length 16 --n_types 3 --n_samples 15000 --epochs 15000
```

## Related

- **Training Scripts**: `../../../phenomena/grokking/parentheses/training/`
- **Documentation**: `../../../phenomena/grokking/parentheses/README.md`
- **Modular Arithmetic**: `../modular_arithmetic/` (similar grokking task)

## Citation

```bibtex
@article{grokking_transitions_2025,
  title={Grokking Transitions in Language Models},
  journal={arXiv preprint arXiv:2507.06445},
  year={2025}
}
```
