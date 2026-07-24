#!/bin/bash
# Stage 3: load one saved policy and roll it out -- the save/load round-trip
# sanity check, or a quick success-rate readout. Rollout is cheap; this does not
# need the same resources as training.
#
# Usage (from the repo root):
#   sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_123_s0 \
#       src/slurm/rollout.slurm
#   sbatch --export=ALL,RUN_DIR=...,EPISODES=20,VIDEO=out.mp4 src/slurm/rollout.slurm

#SBATCH --job-name=lift_rollout
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

if [ -z "${RUN_DIR:-}" ]; then
  echo "RUN_DIR is required, e.g.:" >&2
  echo "  sbatch --export=ALL,RUN_DIR=/scratch/\$USER/lang2grasp_runs/lift_123_s0 src/slurm/rollout.slurm" >&2
  exit 1
fi

EPISODES="${EPISODES:-10}"
VIDEO_FLAG=()
if [ -n "${VIDEO:-}" ]; then
  # Offscreen rendering needs a working GL context (EGL/OSMesa). If this fails
  # headless, check what CSF3's GPU nodes provide and set MUJOCO_GL accordingly,
  # e.g.: export MUJOCO_GL=egl
  VIDEO_FLAG=(--video "$VIDEO")
fi

python -m rl.rollout --run-dir "$RUN_DIR" --episodes "$EPISODES" "${VIDEO_FLAG[@]}"
