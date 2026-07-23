#!/bin/bash
# Stage 3, comparison: roll the generic baseline policy (BASELINE_RUN_DIR, trained
# with no --object) out against every lift_<object> run's own physics, alongside that
# object's own dedicated policy, and write a side-by-side comparison to src/results/
# (scripts/compare_policies.py).
#
# Usage (from the repo root):
#   sbatch --export=ALL,BASELINE_RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_baseline \
#       src/slurm/compare_policies.slurm
#   sbatch --export=ALL,BASELINE_RUN_DIR=...,RUNS_DIR=...,EPISODES=20,PLOT=1 \
#       src/slurm/compare_policies.slurm

#SBATCH --job-name=lift_compare
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

if [ -z "${BASELINE_RUN_DIR:-}" ]; then
  echo "BASELINE_RUN_DIR is required, e.g.:" >&2
  echo "  sbatch --export=ALL,BASELINE_RUN_DIR=/scratch/\$USER/lang2grasp_runs/lift_baseline \\" >&2
  echo "      src/slurm/compare_policies.slurm" >&2
  exit 1
fi

RUNS_DIR="${RUNS_DIR:-/scratch/${USER}/lang2grasp_runs}"
EPISODES="${EPISODES:-20}"

PLOT_FLAG=()
if [ -n "${PLOT:-}" ]; then
  PLOT_FLAG=(--plot)
fi

python src/scripts/compare_policies.py --baseline-run-dir "$BASELINE_RUN_DIR" \
    --runs-dir "$RUNS_DIR" --episodes "$EPISODES" "${PLOT_FLAG[@]}"
