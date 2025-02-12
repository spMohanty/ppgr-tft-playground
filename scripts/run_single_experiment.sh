#!/bin/bash
#SBATCH --job-name=ppgr_tft_single-experiment
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Configuration variables

cd /home/mohanty/food/pytorch-forecasting-playground

# Activate environment (uncomment if needed)
source ~/.bashrc
micromamba activate ppgr


time python run.py --max_encoder_length 96

