#!/bin/bash
# Stage 3, all objects: roll every trained lift_<object> run under RUNS_DIR out,
# print a per-object success-rate table, and write results/plot/video to
# src/results/ (scripts/rollout_all_objects.py).
#
# Usage (from the repo root):
#   sbatch src/slurm/rollout_all_objects.slurm
#   sbatch --export=ALL,RUNS_DIR=/scratch/$USER/lang2grasp_runs,EPISODES=20 \
#       src/slurm/rollout_all_objects.slurm
#   sbatch --export=ALL,PLOT=1,VIDEO=1 src/slurm/rollout_all_objects.slurm

#SBATCH --job-name=lift_rollout_all
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --time=01:00:00               # <-- EDIT: bump this if VIDEO=1 renders many objects
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

RUNS_DIR="${RUNS_DIR:-/scratch/${USER}/lang2grasp_runs}"
EPISODES="${EPISODES:-20}"

PLOT_FLAG=()
if [ -n "${PLOT:-}" ]; then
  PLOT_FLAG=(--plot)
fi

VIDEO_FLAG=()
if [ -n "${VIDEO:-}" ]; then
  # Offscreen rendering needs a working GL context (EGL/OSMesa). If this fails
  # headless, check what CSF3's GPU nodes provide and set MUJOCO_GL accordingly,
  # e.g.: export MUJOCO_GL=egl
  VIDEO_FLAG=(--video)
fi

python src/scripts/rollout_all_objects.py --runs-dir "$RUNS_DIR" --episodes "$EPISODES" \
    "${PLOT_FLAG[@]}" "${VIDEO_FLAG[@]}"
