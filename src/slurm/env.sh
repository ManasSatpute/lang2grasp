#!/bin/bash
# Shared setup, sourced by every slurm/*.slurm script -- edit the two CSF3-specific
# placeholders below once, and every job picks them up.
#
# CSF3 has historically run Grid Engine (`qsub`), not Slurm. Before relying on any
# of these scripts, confirm `sbatch`/`squeue`/`scontrol` actually exist on your
# allocation (e.g. `which sbatch`) -- if not, these need Grid Engine equivalents
# (`qsub`, `#$ -l ...`) instead, ask if you want that version.

set -uo pipefail

# --- EDIT ME: modules -------------------------------------------------------
# Find the exact names with `module avail anaconda` / `module avail cuda` on CSF3.
ANACONDA_MODULE="apps/binapps/anaconda3/2023.09"   # <-- EDIT
CUDA_MODULE=""                                     # <-- EDIT (GPU jobs only), e.g. "libs/cuda/12.1.0"

module purge
module load "$ANACONDA_MODULE"
if [ -n "$CUDA_MODULE" ]; then
  module load "$CUDA_MODULE"   # must match whatever `pip install torch` was built against
fi

# --- Conda env ---------------------------------------------------------------
# Created once, not part of any job -- see src/slurm/README.md.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lang2grasp

# --- Repo paths ----------------------------------------------------------------
# All sbatch commands in src/slurm/README.md are submitted from the repo root, so
# SLURM_SUBMIT_DIR is that root.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
cd "${SLURM_SUBMIT_DIR}"

# --- Threading (see README "Sizing for 12 cores + 1 GPU") ---------------------
# Every SubprocVecEnv worker links BLAS independently; an unpinned OpenMP pool per
# worker thrashes a shared-node cgroup and runs slower than one environment.
export OMP_NUM_THREADS=1
