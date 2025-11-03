# Parentheses Matching Dataset

This dataset is designed to study grokking and delayed generalization phenomena in parentheses matching tasks, based on the research described in "Grokking Transitions in Language Models" (arXiv:2507.06445).

## Tasks

The dataset supports two related but distinct tasks:

### 1. Equal Count Task
Sequences are classified as positive if they have the same number of open and closed brackets for each bracket type, regardless of whether they are properly nested.

**Examples:**
- Positive: `())(()` - 3 open, 3 closed parentheses
- Negative: `()()(` - 4 open, 2 closed parentheses

### 2. Nested Task (Dyck Language)
Sequences are classified as positive if they form properly nested/matched bracket sequences (valid Dyck language).

**Examples:**
- Positive: `(()())` - properly nested
- Negative: `())(()` - not properly nested (even though counts are equal)

## Bracket Types

The dataset supports multiple bracket types controlled by the `n_types` parameter:

- `n_types=1`: Only `()` parentheses
- `n_types=2`: `()` and `[]` brackets
- `n_types=3`: `()`, `[]`, and `{}` brackets
- `n_types=4`: `()`, `[]`, `{}`, and `<>` brackets

## Delayed Generalization Phenomenon

These tasks exhibit interesting delayed generalization behavior:

1. **Memorization Phase**: Models initially memorize training examples, achieving high training accuracy while test accuracy remains low
2. **Grokking Phase**: After extended training, test accuracy suddenly jumps to high values
3. **Generalization Phase**: Model learns the underlying rule structure

The **nested task** is particularly challenging as it requires understanding recursive structure, making the grokking phenomenon more pronounced.

## Dataset Generation

Generate a dataset using:

```bash
python generate_data.py \
    --task nested \
    --length 10 \
    --n_types 1 \
    --n_samples 10000 \
    --train_fraction 0.5 \
    --output_dir ./parentheses_data
```

### Parameters

- `--task`: Task type (`equal_count` or `nested`)
- `--length`: Sequence length (number of brackets)
- `--n_types`: Number of bracket types (1-4)
- `--n_samples`: Total number of samples to generate
- `--train_fraction`: Fraction of data for training (rest for testing)
- `--seed`: Random seed for reproducibility
- `--output_dir`: Output directory for dataset files

### Output Structure

The script creates a directory with the following structure:

```
parentheses_data/
└── nested_len10_types1_trainfrac_0.5/
    ├── train_inputs.npy      # Training sequences (token IDs)
    ├── train_targets.npy     # Training labels (0=invalid, 1=valid)
    ├── test_inputs.npy       # Test sequences
    ├── test_targets.npy      # Test labels
    └── metadata.json         # Dataset configuration and statistics
```

## Training

Train a transformer model on the dataset:

```bash
python phenomena/grokking/parentheses/training/train_parentheses.py \
    --data_dir ./parentheses_data/nested_len10_types1_trainfrac_0.5 \
    --epochs 10000 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --save_dir ./results
```

### Training Parameters

- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (10000+ recommended for grokking)
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--weight_decay`: Weight decay (crucial for grokking!)
- `--d_model`: Model dimension
- `--n_heads`: Number of attention heads
- `--n_layers`: Number of transformer layers
- `--data_fraction`: Fraction of dataset to use (for ablation studies)

### Wandb Integration

Enable experiment tracking with Weights & Biases:

```bash
python phenomena/grokking/parentheses/training/train_parentheses.py \
    --data_dir ./parentheses_data/nested_len10_types1_trainfrac_0.5 \
    --epochs 10000 \
    --use_wandb \
    --wandb_project delayed_generalization \
    --wandb_name parentheses_nested_experiment
```

## Expected Behavior

### Nested Task (More Challenging)
- **Memorization phase**: ~1000-3000 epochs
- **Grokking point**: Sudden test accuracy jump from ~50% to ~90%+
- **Final performance**: Near-perfect accuracy on both train and test

### Equal Count Task (Easier)
- **Memorization phase**: ~500-1500 epochs
- **Grokking point**: Earlier than nested task
- **Final performance**: Perfect or near-perfect accuracy

## Key Differences from Modular Arithmetic

Unlike modular arithmetic (next-token prediction):
- **Classification task**: Binary classification instead of sequence generation
- **Structural understanding**: Requires understanding of nested/recursive structure
- **Multiple bracket types**: Complexity increases with more bracket types

## Research Applications

This dataset is useful for studying:

1. **Grokking mechanisms**: Understanding sudden generalization
2. **Structural learning**: How models learn recursive patterns
3. **Weight decay effects**: Impact on delayed generalization
4. **Scaling laws**: Effect of model size and data size on grokking timing
5. **Multi-task learning**: Comparing equal_count vs nested task learning dynamics

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{grokking_transitions_2025,
  title={Grokking Transitions in Language Models},
  author={...},
  journal={arXiv preprint arXiv:2507.06445},
  year={2025}
}
```

## Related Work

- Power et al. (2022): "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- Nanda et al. (2023): "Progress measures for grokking via mechanistic interpretability"
- Liu et al. (2022): "Omnigrok: Grokking Beyond Algorithmic Data"
