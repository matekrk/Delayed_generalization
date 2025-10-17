#!/bin/bash
#SBATCH --job-name=data_attribution
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/data_attribution_%j.out
#SBATCH --error=logs/data_attribution_%j.err

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Set up environment
source activate delayed_gen

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create output directory
mkdir -p results/data_attribution

# Run TRAK data attribution analysis
echo "Running TRAK data attribution analysis..."
python -m data_attribution.trak.trak_attributor \
    --model_path results/trained_models/best_model.pth \
    --train_data_path data/cifar10 \
    --test_data_path data/cifar10 \
    --save_dir results/data_attribution/trak \
    --num_train_samples 10000 \
    --num_test_samples 1000

# Run GradCAM analysis
echo "Running GradCAM analysis..."
python -m data_attribution.gradcam.gradcam_attributor \
    --model_path results/trained_models/best_model.pth \
    --test_data_path data/cifar10 \
    --target_layers "layer4" "layer3" "layer2" \
    --save_dir results/data_attribution/gradcam \
    --num_samples 500

echo "Data attribution analysis completed!"