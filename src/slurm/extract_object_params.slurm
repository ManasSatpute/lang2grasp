#!/bin/bash
# Stage 1: text prompt -> LLM -> ObjectParams, snapshotted to
# src/configs/objects/<name>.json. CPU-only -- no GPU requested.
#
# --backend mock (the default here) is fully offline/deterministic. Real backends
# (anthropic/openai/groq) need outbound internet AND an API key -- many HPC compute
# nodes block outbound internet, so check with `curl -sI https://api.anthropic.com`
# first, or run this stage on the login node instead. Training/rollout never call an
# LLM, so only this stage is network-sensitive.
#
# API key: put it in a `.env` file at the repo root (copy `.env.example`, gitignored)
# -- `extraction/llm_backends.py` loads it automatically. Passing it via --export
# (below) also works and takes precedence over `.env` if both are set.
#
# Usage (from the repo root):
#   sbatch src/slurm/extract_object_params.slurm                                   # --backend mock
#   sbatch --export=ALL,BACKEND=anthropic,MODEL=claude-haiku-4-5 src/slurm/extract_object_params.slurm
#   sbatch --export=ALL,BACKEND=groq,MODEL=llama-3.3-70b-versatile src/slurm/extract_object_params.slurm

#SBATCH --job-name=extract_params
#SBATCH --partition=serial            # CSF3: single-core CPU queue, no GPU needed for this stage
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

BACKEND="${BACKEND:-mock}"
MODEL_FLAG=()
if [ -n "${MODEL:-}" ]; then
  MODEL_FLAG=(--model "$MODEL")
fi

python src/scripts/extract_object_params.py --backend "$BACKEND" "${MODEL_FLAG[@]}"
