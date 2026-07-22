#!/bin/bash
# Gate 1: prove the GPU/CUDA/torch stack works on a compute node before trusting
# anything else. A login node has no GPU, so checking there proves nothing.
#
# Usage (from the repo root): sbatch src/slurm/check_gpu.slurm

#SBATCH --job-name=check_gpu
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

python src/scripts/check_gpu.py
