#!/bin/bash
# Stage 1.5: how accurate was the LLM's extraction against the golden (ground-truth)
# dataset? Writes CSVs (+ optionally a chart) to src/results/ -- independent of any
# training run, so this can run right after extraction. CPU-only, no GPU needed. See
# evaluate_extraction_accuracy.py's own module docstring for details.
#
# Usage (from the repo root), after extract_object_params.slurm has produced the
# snapshots this reads:
#   sbatch src/slurm/evaluate_extraction_accuracy.slurm
#   sbatch --export=ALL,PLOT=1 src/slurm/evaluate_extraction_accuracy.slurm

#SBATCH --job-name=extraction_accuracy
#SBATCH --partition=serial            # CSF3: single-core CPU queue, no GPU needed for this stage
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

PLOT_FLAG=()
if [ -n "${PLOT:-}" ]; then
  PLOT_FLAG=(--plot)
fi

python src/scripts/evaluate_extraction_accuracy.py "${PLOT_FLAG[@]}"
