#!/bin/bash
# Stage 2, paradigm switch, cluster form: one SLURM array task per variant (pi_blind /
# pi_blind+hist / pi_param), launched together instead of three separate
# train_paradigm.slurm submissions. Same requeue mechanism as train.slurm/
# train_objects_array.slurm -- see train.slurm's header comment.
#
# --array=0-2 (fixed): index 0=blind, 1=blind_hist, 2=param.
#
# Usage (from the repo root):
#   sbatch src/slurm/train_paradigm_array.slurm
#   sbatch --export=ALL,CONFIG=src/configs/policy/sac.json,SEED=1 \
#       src/slurm/train_paradigm_array.slurm
#   sbatch --export=ALL,HISTORY_LEN=32,Z_NOISE_STD=0.2 src/slurm/train_paradigm_array.slurm

#SBATCH --job-name=lift_paradigm_array
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00               # <-- EDIT: keep TrainConfig.max_hours ~15 min below this.
                                       # Domain-randomized episodes recompile MuJoCo every reset
                                       # (hard_reset=True) -- budget more wall-clock than
                                       # train_objects_array.slurm.
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# No `-e`: we need $? from `wait` below, which `-e` would abort past.
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

VARIANTS=(blind blind_hist param)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]:-}"
if [ -z "$VARIANT" ]; then
  echo "No variant at array index ${SLURM_ARRAY_TASK_ID} (expected 0-2: blind, blind_hist, param)." >&2
  exit 1
fi

RUNS_DIR="/scratch/${USER}/lang2grasp_runs"   # <-- must be shared storage, see train.slurm
CONFIG="${CONFIG:-src/configs/policy/sac.json}"
SEED="${SEED:-0}"
RUN_NAME="paradigm_${VARIANT}_s${SEED}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: ${VARIANT} -> ${RUNS_DIR}/${RUN_NAME}"

CMD=(python src/scripts/train_paradigm.py --variant "$VARIANT" --base-config "$CONFIG"
     --log-dir "$RUNS_DIR" --run-name "$RUN_NAME" --seed "$SEED" --resume)

[ -n "${TOTAL_TIMESTEPS:-}" ] && CMD+=(--total-timesteps "$TOTAL_TIMESTEPS")
# HISTORY_LEN/GRU_HIDDEN only apply to the blind_hist task; Z_NOISE_STD only to param.
# train_paradigm.py itself would ignore either on the wrong variant, but skipping them
# here keeps each task's logged command free of irrelevant flags.
if [ "$VARIANT" = "blind_hist" ]; then
  [ -n "${HISTORY_LEN:-}" ] && CMD+=(--history-len "$HISTORY_LEN")
  [ -n "${GRU_HIDDEN:-}" ]  && CMD+=(--gru-hidden "$GRU_HIDDEN")
fi
if [ "$VARIANT" = "param" ]; then
  [ -n "${Z_NOISE_STD:-}" ] && CMD+=(--z-noise-std "$Z_NOISE_STD")
fi
[ -n "${GRIP_FORCE_SHAPING:-}" ] && CMD+=(--grip-force-shaping)
[ -n "${SHAPES:-}" ] && CMD+=(--shapes $SHAPES)  # unquoted on purpose, see train_paradigm.slurm

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
  echo "Checkpointed cleanly, work remains -- requeuing array task ${SLURM_ARRAY_TASK_ID}."
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
exit "$EXIT_CODE"
