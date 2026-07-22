#!/bin/bash
# Stage 2: train one SAC policy. Survives the wall-clock limit -- see the top-level
# README's "Training that survives the wall clock" for the full mechanism; this
# script implements its two SLURM-side halves:
#   1. `--signal=B:USR1@300` + the trap below forward the pre-emption warning to
#      python (the `B:` prefix targets the batch *shell*, since these jobs don't
#      use `srun`, and the trap is what actually relays it to the child process).
#   2. Exit code 42 means "GracefulExitCallback checkpointed cleanly, work
#      remains" -> requeue; anything else is a real success/failure and is left
#      alone.
#
# Without OBJECT set, this trains the un-parameterised baseline (stock Lift cube)
# via `rl.train`. With OBJECT set, it trains that LLM-described object instead via
# `scripts/train_object.py` -- run_name is then derived from the object's name (not
# the job ID) so repeated submissions/resumes for the same object share one run dir.
#
# Usage (from the repo root):
#   JOB=$(sbatch --parsable src/slurm/train.slurm)                    # baseline cube
#   sbatch --export=ALL,OBJECT=src/configs/objects/raw_egg.json src/slurm/train.slurm
#   sbatch --export=ALL,CONFIG=src/configs/policy/sac.json,SEED=1 src/slurm/train.slurm

#SBATCH --job-name=lift_train
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00               # <-- EDIT: keep TrainConfig.max_hours ~15 min below this
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# No `-e`: we need $? from `wait` below, which `-e` would abort past.
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

RUNS_DIR="/scratch/${USER}/lang2grasp_runs"   # <-- must be shared storage; node-local /tmp
                                               # loses checkpoints if a requeue lands elsewhere.
CONFIG="${CONFIG:-src/configs/policy/sac.json}"
SEED="${SEED:-0}"

if [ -n "${OBJECT:-}" ]; then
  RUN_NAME="lift_$(basename "${OBJECT%.json}")"
  CMD=(python src/scripts/train_object.py --object "$OBJECT" --base-config "$CONFIG"
       --log-dir "$RUNS_DIR" --run-name "$RUN_NAME" --seed "$SEED" --resume)
else
  RUN_NAME="lift_${SLURM_JOB_ID}_s${SEED}"
  CMD=(python -m rl.train --config "$CONFIG"
       --log-dir "$RUNS_DIR" --run-name "$RUN_NAME" --seed "$SEED" --resume)
fi
echo "Run dir: ${RUNS_DIR}/${RUN_NAME}"
echo "Command: ${CMD[*]}"

_forward_signal() {
  echo "Caught pre-emption signal; forwarding SIGUSR1 to python (pid ${PID})."
  kill -USR1 "$PID" 2>/dev/null || true
}
trap _forward_signal USR1

"${CMD[@]}" &
PID=$!
wait "$PID"
EXIT_CODE=$?
echo "Training process exited with code ${EXIT_CODE}."

if [ "$EXIT_CODE" -eq 42 ]; then
  echo "Checkpointed cleanly, work remains -- requeuing ${SLURM_JOB_ID}."
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$EXIT_CODE"
