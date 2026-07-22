#!/bin/bash
# Shared setup, sourced by every slurm/*.slurm script. Module names below are
# confirmed for this CSF3 account (`module avail` doesn't list an "anaconda"
# module here -- conda comes from miniforge3 instead).
#
# CSF3 has historically run Grid Engine (`qsub`), not Slurm. Before relying on any
# of these scripts, confirm `sbatch`/`squeue`/`scontrol` actually exist on your
# allocation (e.g. `which sbatch`) -- if not, these need Grid Engine equivalents
# (`qsub`, `#$ -l ...`) instead, ask if you want that version.

set -uo pipefail

CONDA_MODULE="apps/binapps/conda/miniforge3/25.9.1"
CUDA_MODULE="cuda/12.6.2"   # must match the `pip install torch --index-url .../cu126` used when
                            # the conda env was created -- see the root README's "Setup" section

module purge
module load "$CONDA_MODULE"
module load "$CUDA_MODULE"

# --- Conda env ---------------------------------------------------------------
# Created once, not part of any job -- see the root README's "Setup" section.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lang2grasp

# --- Repo paths ----------------------------------------------------------------
# All sbatch commands in the root README are submitted from the repo root, so
# SLURM_SUBMIT_DIR is that root.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
cd "${SLURM_SUBMIT_DIR}"

# --- Threading (see README "Sizing for 12 cores + 1 GPU") ---------------------
# Every SubprocVecEnv worker links BLAS independently; an unpinned OpenMP pool per
# worker thrashes a shared-node cgroup and runs slower than one environment.
export OMP_NUM_THREADS=1
