#!/bin/bash
# Gate 2: end-to-end plumbing check (check_env + train 3k steps + save/load
# round-trip + rollout). ~2 minutes, not a performance test -- see
# src/tests/smoke_test.py's docstring. Run this after check_gpu.slurm passes
# and before submitting any real training job.
#
# Usage (from the repo root): sbatch src/slurm/smoke_test.slurm

#SBATCH --job-name=smoke_test
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

python src/tests/smoke_test.py --steps 3000
