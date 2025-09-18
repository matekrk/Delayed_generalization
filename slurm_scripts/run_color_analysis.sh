#!/bin/bash
#SBATCH --job-name=color_analysis
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/color_analysis_%j.out
#SBATCH --error=logs/color_analysis_%j.err

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Set up environment
source activate delayed_gen  # Activate your conda environment

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create output directory
mkdir -p results/color_analysis

# Run CIFAR-100 color analysis
echo "Running CIFAR-100 color analysis..."
python -m datasets.vision.cifar100_analysis \
    --root data/cifar100 \
    --cache_dir results/color_analysis/cifar100 \
    --download

# Run TinyImageNet color analysis
echo "Running TinyImageNet color analysis..."
python -m datasets.vision.tinyimagenet_analysis \
    --root data/tinyimagenet \
    --cache_dir results/color_analysis/tinyimagenet \
    --download

echo "Color analysis completed!"