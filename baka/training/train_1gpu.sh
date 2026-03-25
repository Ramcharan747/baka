#!/bin/bash
#SBATCH --job-name=baka-300m-1gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:45:00
#SBATCH --output=logs/baka_1gpu_%j.out
#SBATCH --error=logs/baka_1gpu_%j.err

# Create log directory
mkdir -p logs

# Activate environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set CUDA
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

cd $SLURM_SUBMIT_DIR

python baka/training/pretrain_1gpu.py \
    --data pretrain_final.jsonl \
    --ckpt_dir checkpoints/300m \
    --total_steps 76000 \
    --resume
