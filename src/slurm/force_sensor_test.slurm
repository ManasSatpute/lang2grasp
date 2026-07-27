#!/bin/bash
# Gate 3: fingertip force sensor + crush penalty/termination + width_mass_set check
# (src/tests/force_sensor_test.py). ~1 minute, CPU-only -- no torch/SB3/GPU involved,
# just robosuite/mujoco/numpy. Run this after smoke_test.slurm passes and before
# submitting any real training job -- it's the one gate that actually exercises
# objects/force_gripper.py's derived-XML gripper against whatever robosuite version
# this cluster's conda env installed, which may differ from the version this was
# developed against (robosuite 1.5.2 -- see requirements.txt's `robosuite<1.6` pin and
# force_gripper.py's module docstring). If the installed version's panda_gripper.xml
# has a different body layout, this is where that shows up, as a clear RuntimeError
# rather than a silent wrong result later.
#
# Usage (from the repo root): sbatch src/slurm/force_sensor_test.slurm

#SBATCH --job-name=force_sensor_test
#SBATCH --partition=serial            # CSF3: single-core CPU queue, no GPU needed for this gate
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1             # serial only allows 1 core per job on this account
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

python src/tests/force_sensor_test.py
