#!/bin/bash
# Stage 1 (+ 1.5): text prompt -> LLM -> ObjectParams, snapshotted to
# src/configs/objects/<name>.json, immediately followed by the accuracy check
# against the golden dataset (writes src/results/extraction_accuracy_*.csv,
# +chart with PLOT=1). CPU-only -- no GPU requested. See
# extract_object_params.py's own module docstring for the --no-evaluate escape
# hatch if you want stage 1 without stage 1.5.
#
# Every backend (anthropic/openai/groq) needs outbound internet AND an API key --
# many HPC compute nodes block outbound internet, so check with
# `curl -sI https://api.anthropic.com` first, or run this stage on the login node
# instead. Training/rollout never call an LLM, so only this stage is network-sensitive.
#
# API key: put it in a `.env` file at the repo root (copy `.env.example`, gitignored)
# -- `extraction/llm_backends.py` loads it automatically. Passing it via --export
# (below) also works and takes precedence over `.env` if both are set.
#
# BACKEND is required -- there's no offline default.
#
# SAMPLES (default 1) draws that many independent extractions per object and takes
# the median/majority vote across them -- meaningfully less noisy than a single call
# (see extraction/param_extraction.py's module docstring) at the cost of SAMPLES x
# the backend calls/runtime; bump --time accordingly for a large SAMPLES.
#
# Usage (from the repo root):
#   sbatch --export=ALL,BACKEND=anthropic,MODEL=claude-haiku-4-5 src/slurm/extract_object_params.slurm
#   sbatch --export=ALL,BACKEND=openai,MODEL=gpt-4o-mini src/slurm/extract_object_params.slurm
#   sbatch --export=ALL,BACKEND=groq,MODEL=openai/gpt-oss-120b,PLOT=1 src/slurm/extract_object_params.slurm
#   sbatch --export=ALL,BACKEND=groq,SAMPLES=5 --time=00:20:00 src/slurm/extract_object_params.slurm

#SBATCH --job-name=extract_params
#SBATCH --partition=serial            # CSF3: single-core CPU queue, no GPU needed for this stage
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

if [ -z "${BACKEND:-}" ]; then
  echo "BACKEND is required, e.g. --export=ALL,BACKEND=groq" >&2
  exit 1
fi
MODEL_FLAG=()
if [ -n "${MODEL:-}" ]; then
  MODEL_FLAG=(--model "$MODEL")
fi
PLOT_FLAG=()
if [ -n "${PLOT:-}" ]; then
  PLOT_FLAG=(--plot)
fi
SAMPLES_FLAG=()
if [ -n "${SAMPLES:-}" ]; then
  SAMPLES_FLAG=(--samples "$SAMPLES")
fi

python src/scripts/extract_object_params.py --backend "$BACKEND" "${MODEL_FLAG[@]}" "${PLOT_FLAG[@]}" "${SAMPLES_FLAG[@]}"
