#!/bin/bash
# Stage 2, cluster form: one SLURM array task per LLM-parameterised object,
# running in parallel instead of scripts/train_all_objects.py's local sequential
# loop. Same requeue mechanism as train.slurm -- see that file's header comment.
#
# --array=0-5 assumes exactly 6 object snapshots in src/configs/objects/ (the
# repo's default set). If you add/remove objects, update the --array range to
# match (count them: `ls src/configs/objects/*.json | grep -v prompts.json | wc -l`).
#
# Usage (from the repo root):
#   sbatch src/slurm/train_objects_array.slurm
#   sbatch --export=ALL,CONFIG=src/configs/policy/sac.json src/slurm/train_objects_array.slurm

#SBATCH --job-name=lift_train_array
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00               # <-- EDIT: keep TrainConfig.max_hours ~15 min below this
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# No `-e`: we need $? from `wait` below, which `-e` would abort past.
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

OBJECTS_DIR="src/configs/objects"
mapfile -t OBJECT_FILES < <(find "$OBJECTS_DIR" -maxdepth 1 -name '*.json' ! -name 'prompts.json' | sort)

OBJECT="${OBJECT_FILES[$SLURM_ARRAY_TASK_ID]:-}"
if [ -z "$OBJECT" ]; then
  echo "No object snapshot at array index ${SLURM_ARRAY_TASK_ID} (found ${#OBJECT_FILES[@]}" \
       "under ${OBJECTS_DIR}). Does --array's range match the object count?" >&2
  exit 1
fi

RUNS_DIR="/scratch/${USER}/lang2grasp_runs"   # <-- must be shared storage, see train.slurm
CONFIG="${CONFIG:-src/configs/policy/sac.json}"
RUN_NAME="lift_$(basename "${OBJECT%.json}")"
echo "Array task ${SLURM_ARRAY_TASK_ID}: ${OBJECT} -> ${RUNS_DIR}/${RUN_NAME}"

_forward_signal() {
  echo "Caught pre-emption signal; forwarding SIGUSR1 to python (pid ${PID})."
  kill -USR1 "$PID" 2>/dev/null || true
}
trap _forward_signal USR1

python src/scripts/train_object.py --object "$OBJECT" --base-config "$CONFIG" \
  --log-dir "$RUNS_DIR" --run-name "$RUN_NAME" --resume &
PID=$!
wait "$PID"
EXIT_CODE=$?
echo "Training process exited with code ${EXIT_CODE}."

if [ "$EXIT_CODE" -eq 42 ]; then
  echo "Checkpointed cleanly, work remains -- requeuing array task ${SLURM_ARRAY_TASK_ID}."
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
exit "$EXIT_CODE"
