#!/bin/bash
# Stage 1: text prompt -> LLM -> ObjectParams, snapshotted to
# src/configs/objects/<name>.json. CPU-only -- no GPU requested.
#
# --backend mock (the default here) is fully offline/deterministic and always
# works as a batch job. Real backends (anthropic/openai/groq) need outbound
# internet AND an API key in the environment -- many HPC compute nodes block
# outbound internet entirely. Check this works before relying on it:
#   srun --partition=serial --pty curl -sI https://api.anthropic.com
# If that fails, run this stage on the login node instead (still just
# `PYTHONPATH=src python src/scripts/extract_object_params.py --backend ...`,
# no sbatch needed) and commit the resulting JSON snapshots -- training and
# rollout never call an LLM, so only this one stage is network-sensitive.
#
# Usage (from the repo root):
#   sbatch src/slurm/extract_object_params.slurm                                   # --backend mock
#   sbatch --export=ALL,BACKEND=anthropic,MODEL=claude-haiku-4-5,\
#     ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY src/slurm/extract_object_params.slurm

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
