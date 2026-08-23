#!/bin/bash
# Gate 3: fingertip force sensor + crush penalty/termination + width_mass_set check
# (src/tests/force_sensor_test.py). ~1 minute, CPU-only. Run this after smoke_test.slurm
# passes and before submitting any real training job.
#
# Usage (from the repo root): sbatch src/slurm/force_sensor_test.slurm

#SBATCH --job-name=force_sensor_test
#SBATCH --partition=serial            # CSF3: single-core CPU queue, no GPU needed for this gate
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1             # serial only allows 1 core per job on this account
#SBATCH --mem=4G                      # serial caps memory at ~5345 MB per core on this account
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

python src/tests/force_sensor_test.py
