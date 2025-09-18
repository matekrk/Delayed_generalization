#!/bin/bash
#SBATCH --job-name=simplicity_bias
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/simplicity_bias_%j.out
#SBATCH --error=logs/simplicity_bias_%j.err

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Set up environment
source activate delayed_gen

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create output directory
mkdir -p results/simplicity_bias_experiments

# Generate colored MNIST dataset
echo "Generating colored MNIST dataset..."
python -m datasets.vision.generate_colored_mnist \
    --train_correlation 0.9 \
    --test_correlation 0.1 \
    --output_dir data/colored_mnist \
    --data_dir data/mnist

# Train model with simplicity bias analysis
echo "Training model on colored MNIST..."
python -m phenomena.simplicity_bias.colored_mnist.training.train_colored_mnist \
    --data_dir data/colored_mnist \
    --model_type cnn \
    --epochs 200 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --save_dir results/simplicity_bias_experiments \
    --color_analysis \
    --gradcam_analysis \
    --wandb_project "delayed_generalization" \
    --wandb_name "simplicity_bias_${SLURM_JOB_ID}"

echo "Simplicity bias experiment completed!"