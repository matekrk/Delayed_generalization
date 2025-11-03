#!/bin/bash
# Run script for parentheses grokking experiments
#
# This script generates a parentheses dataset and trains a model to observe grokking.
# 
# Usage:
#   ./run_parentheses_experiment.sh [task] [n_types] [length] [epochs]
#
# Example:
#   ./run_parentheses_experiment.sh nested 1 10 10000

set -e

# Default parameters
TASK=${1:-nested}
N_TYPES=${2:-1}
LENGTH=${3:-10}
EPOCHS=${4:-10000}
TRAIN_FRACTION=${5:-0.5}
N_SAMPLES=${6:-10000}

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/algorithmic/parentheses"
PHENOMENA_DIR="${REPO_ROOT}/phenomena/grokking/parentheses"
OUTPUT_DIR="${REPO_ROOT}/parentheses_data"
RESULTS_DIR="${REPO_ROOT}/parentheses_results"

echo "=================================================="
echo "Parentheses Grokking Experiment"
echo "=================================================="
echo "Task: ${TASK}"
echo "Bracket types: ${N_TYPES}"
echo "Sequence length: ${LENGTH}"
echo "Training epochs: ${EPOCHS}"
echo "Train fraction: ${TRAIN_FRACTION}"
echo "Total samples: ${N_SAMPLES}"
echo "=================================================="
echo ""

# Step 1: Generate dataset
echo "Step 1: Generating dataset..."
python "${DATA_DIR}/generate_data.py" \
    --task "${TASK}" \
    --n_types "${N_TYPES}" \
    --length "${LENGTH}" \
    --n_samples "${N_SAMPLES}" \
    --train_fraction "${TRAIN_FRACTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

# Determine dataset directory name
DATASET_SUBDIR="${TASK}_len${LENGTH}_types${N_TYPES}_trainfrac_${TRAIN_FRACTION}"
DATASET_PATH="${OUTPUT_DIR}/${DATASET_SUBDIR}"

echo ""
echo "Dataset generated at: ${DATASET_PATH}"
echo ""

# Step 2: Train model
echo "Step 2: Training model..."
python "${PHENOMENA_DIR}/training/train_parentheses.py" \
    --data_dir "${DATASET_PATH}" \
    --epochs "${EPOCHS}" \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --save_dir "${RESULTS_DIR}" \
    --log_interval 100 \
    --seed 42

echo ""
echo "=================================================="
echo "Experiment completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=================================================="
