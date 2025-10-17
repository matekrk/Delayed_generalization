#!/bin/bash
#SBATCH --job-name=robustness_eval
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/robustness_%j.out
#SBATCH --error=logs/robustness_%j.err

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Set up environment
source activate delayed_gen

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create output directory
mkdir -p results/robustness_evaluation

# Evaluate model robustness (assuming model checkpoint exists)
echo "Evaluating model robustness..."
python -m phenomena.robustness.adversarial.adversarial_evaluator \
    --model_path results/trained_models/best_model.pth \
    --test_data_path data/cifar10 \
    --save_dir results/robustness_evaluation \
    --attacks fgsm pgd cw \
    --attack_strengths weak medium strong

# Also run CIFAR-10-C evaluation
echo "Evaluating CIFAR-10-C robustness..."
python -m phenomena.robustness.cifar10c.training.train_cifar10c \
    --evaluate_only \
    --model_path results/trained_models/best_model.pth \
    --data_dir data/cifar10c \
    --save_dir results/robustness_evaluation/cifar10c

echo "Robustness evaluation completed!"