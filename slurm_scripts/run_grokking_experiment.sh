#!/bin/bash
#SBATCH --job-name=grokking_experiment
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grokking_%j.out
#SBATCH --error=logs/grokking_%j.err

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Set up environment
source activate delayed_gen

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create output directory
mkdir -p results/grokking_experiments

# Run grokking experiment with modular arithmetic
echo "Running grokking experiment..."
python -m phenomena.grokking.training.train_modular \
    --operation addition \
    --prime 97 \
    --train_fraction 0.5 \
    --model_type transformer \
    --n_layers 2 \
    --n_heads 4 \
    --d_model 128 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --batch_size 512 \
    --max_epochs 10000 \
    --patience 1000 \
    --save_dir results/grokking_experiments \
    --wandb_project "delayed_generalization" \
    --wandb_name "grokking_addition_${SLURM_JOB_ID}"

echo "Grokking experiment completed!"