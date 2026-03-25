#!/bin/bash
#SBATCH --job-name=baka-300m-2gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:45:00
#SBATCH --output=logs/baka_2gpu_%j.out
#SBATCH --error=logs/baka_2gpu_%j.err

# Create log directory
mkdir -p logs

# Activate environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# NCCL settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

cd $SLURM_SUBMIT_DIR

torchrun --nproc_per_node=2 \
    baka/training/pretrain_2gpu.py \
    --data pretrain_final.jsonl \
    --ckpt_dir checkpoints/300m \
    --total_steps 38000 \
    --resume
