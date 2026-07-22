#!/bin/bash
# Stage 3, all objects: roll every trained lift_<object> run under RUNS_DIR out
# and print a per-object success-rate table (scripts/rollout_all_objects.py).
#
# Usage (from the repo root):
#   sbatch src/slurm/rollout_all_objects.slurm
#   sbatch --export=ALL,RUNS_DIR=/scratch/$USER/lang2grasp_runs,EPISODES=20 \
#       src/slurm/rollout_all_objects.slurm

#SBATCH --job-name=lift_rollout_all
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

RUNS_DIR="${RUNS_DIR:-/scratch/${USER}/lang2grasp_runs}"
EPISODES="${EPISODES:-20}"

python src/scripts/rollout_all_objects.py --runs-dir "$RUNS_DIR" --episodes "$EPISODES"
