#!/bin/bash
# Stage 2, paradigm switch: train one of pi_blind / pi_blind+hist / pi_param against a
# continuous distribution of objects (scripts/train_paradigm.py) -- see the root
# README's "Paradigm switch: domain randomization vs. per-object specialists". Same
# requeue mechanism as train.slurm -- see that file's header comment for the two
# SLURM-side halves (signal forwarding, exit-code-42 requeue).
#
# VARIANT is required (blind | blind_hist | param). RUN_NAME is derived from
# VARIANT+SEED, not the job ID, so a resubmission (e.g. after a wall-clock cutoff
# beyond what the in-job requeue already handles) resumes the same run dir via
# --resume, same as train.slurm's OBJECT-set branch.
#
# TOTAL_TIMESTEPS is for smoke-checking this pipeline on real cluster hardware before
# committing to a full run -- pair it with a short --time on the sbatch command line
# (see Usage). There's no dedicated tests/train_paradigm smoke script the way
# tests/smoke_test.py covers the baseline: this *is* the smoke check for this path.
#
# Usage (from the repo root):
#   # Quick smoke check on real hardware before a long run (~2 min):
#   sbatch --time=00:15:00 --export=ALL,VARIANT=blind,TOTAL_TIMESTEPS=2000 \
#       src/slurm/train_paradigm.slurm
#   # Full runs, one submission per variant:
#   sbatch --export=ALL,VARIANT=blind      src/slurm/train_paradigm.slurm
#   sbatch --export=ALL,VARIANT=blind_hist src/slurm/train_paradigm.slurm
#   sbatch --export=ALL,VARIANT=param      src/slurm/train_paradigm.slurm
#   sbatch --export=ALL,VARIANT=blind_hist,HISTORY_LEN=32 src/slurm/train_paradigm.slurm
#   sbatch --export=ALL,VARIANT=param,Z_NOISE_STD=0.2,SEED=1 src/slurm/train_paradigm.slurm
#   # To launch all three at once instead, see train_paradigm_array.slurm.

#SBATCH --job-name=lift_paradigm
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00               # <-- EDIT: keep TrainConfig.max_hours ~15 min below this.
                                       # Domain-randomized episodes recompile MuJoCo every reset
                                       # (hard_reset=True) -- budget more wall-clock than train.slurm.
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# No `-e`: we need $? from `wait` below, which `-e` would abort past.
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

if [ -z "${VARIANT:-}" ]; then
  echo "VARIANT is required (blind | blind_hist | param), e.g.:" >&2
  echo "  sbatch --export=ALL,VARIANT=blind src/slurm/train_paradigm.slurm" >&2
  exit 1
fi

RUNS_DIR="/scratch/${USER}/lang2grasp_runs"   # <-- must be shared storage, see train.slurm
CONFIG="${CONFIG:-src/configs/policy/sac.json}"
SEED="${SEED:-0}"
RUN_NAME="paradigm_${VARIANT}_s${SEED}"

CMD=(python src/scripts/train_paradigm.py --variant "$VARIANT" --base-config "$CONFIG"
     --log-dir "$RUNS_DIR" --run-name "$RUN_NAME" --seed "$SEED" --resume)

[ -n "${TOTAL_TIMESTEPS:-}" ] && CMD+=(--total-timesteps "$TOTAL_TIMESTEPS")
[ -n "${HISTORY_LEN:-}" ]     && CMD+=(--history-len "$HISTORY_LEN")
[ -n "${GRU_HIDDEN:-}" ]      && CMD+=(--gru-hidden "$GRU_HIDDEN")
[ -n "${Z_NOISE_STD:-}" ]     && CMD+=(--z-noise-std "$Z_NOISE_STD")
[ -n "${GRIP_FORCE_SHAPING:-}" ] && CMD+=(--grip-force-shaping)
# SHAPES="box cylinder" -> --shapes box cylinder (unquoted on purpose: word-split
# into separate argv entries).
[ -n "${SHAPES:-}" ] && CMD+=(--shapes $SHAPES)

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
