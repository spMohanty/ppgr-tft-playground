#!/bin/bash
#SBATCH --job-name=ppgr_tft_sweep-food-only
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Configuration variables
WANDB_ENTITY="mfr-ppgr-2025"
WANDB_PROJECT="tft-ppgr-2025-ablation-v0"
WANDB_SWEEP_ID="u6s3pker"

NUM_AGENTS=2 # 3 for h100 and 1 for l40s

cd /home/mohanty/food/pytorch-forecasting-playground

# Activate environment (uncomment if needed)
source ~/.bashrc
micromamba activate ppgr

for i in $(seq 1 $NUM_AGENTS); do
    CUDA_VISIBLE_DEVICES=0 wandb agent $WANDB_SWEEP_ID -p $WANDB_PROJECT -e $WANDB_ENTITY &
done

wait


