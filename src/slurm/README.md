# Cluster job scripts (CSF3)

Batch scripts for the two pipelines in the top-level README: LLM object-parameter
extraction, and SAC training/rollout. All of them source [`env.sh`](env.sh) for
the conda env, `PYTHONPATH`, and thread pinning, so most of what's cluster-specific
lives in that one file.

> **Scheduler check first.** These are Slurm scripts (`#SBATCH`, `sbatch`,
> `scontrol`). CSF3 has historically run Grid Engine (`qsub`, `#$ -l ...`)
> instead. Run `which sbatch` on a CSF3 login node before submitting anything --
> if it's not found, these need Grid Engine equivalents, not these files as-is.

## One-time setup: the conda environment

```bash
module avail anaconda            # find the exact module name for your CSF3
module load <ANACONDA_MODULE>    # e.g. apps/binapps/anaconda3/2023.09

conda create -n lang2grasp python=3.10 -y
conda activate lang2grasp

module avail cuda                # find the CUDA version CSF3's GPU nodes provide
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match it
pip install -r requirements.txt  # from the repo root; robosuite/mujoco/sb3/etc.

# Optional, only for the standalone DeliGrasp benchmark (src/extraction/deligrasp,
# scripts/run_experiment.py) -- not needed for the two pipelines below:
pip install -r src/extraction/requirements.txt
```

Python 3.10 is a good choice here: `requirements.txt` pins `numpy<2.0` for
robosuite 1.4.x/1.5.x's ABI, which 3.10 supports cleanly.

Re-run `module load`/`conda activate lang2grasp` in every new shell (or job --
`env.sh` does this for you inside a submitted script).

## Edit before submitting anything

1. **`env.sh`**: fill in `<ANACONDA_MODULE>` (and `<CUDA_MODULE>` if your torch
   build needs a matching module loaded at runtime).
2. **Every `.slurm` file**: `#SBATCH --partition=gpuL` is a placeholder -- set it
   to your allocation's actual GPU partition. `extract_object_params.slurm` uses
   `<CPU_PARTITION>` instead since that stage needs no GPU.
3. **`train.slurm` / `train_objects_array.slurm`**: `RUNS_DIR=/scratch/${USER}/lang2grasp_runs`
   must be shared storage (Lustre/NFS), not node-local `/tmp` -- a requeued job
   can land on a different node and has to find its own checkpoints there.
4. If your account needs an `--account`/`--qos` line, add it to each file.

## Usage (from the repo root)

```bash
# Gates -- run once, in order, before trusting anything below.
sbatch src/slurm/check_gpu.slurm
sbatch src/slurm/smoke_test.slurm

# Stage 1: prompt -> LLM -> ObjectParams JSON. See extract_object_params.slurm's
# own header for the network-access caveat with real (non-mock) backends.
sbatch src/slurm/extract_object_params.slurm

# Stage 2: train. Baseline (stock Lift cube):
JOB=$(sbatch --parsable src/slurm/train.slurm)
tail -f logs/lift_train_${JOB}.out
# One LLM-described object:
sbatch --export=ALL,OBJECT=src/configs/objects/raw_egg.json src/slurm/train.slurm
# All 6 objects, one array task each:
sbatch src/slurm/train_objects_array.slurm

# Stage 3: rollout.
sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0 \
    src/slurm/rollout.slurm
sbatch src/slurm/rollout_all_objects.slurm   # every lift_<object> run under RUNS_DIR

# Cancel / watch, same as any Slurm job:
scancel $JOB
squeue --me
tensorboard --logdir /scratch/$USER/lang2grasp_runs
```

**A `train*` job vanishing from `squeue` and reappearing with the same ID is the
requeue mechanism working, not a crash** -- see the top-level README's "Training
that survives the wall clock" for the full explanation of the exit-code-42 /
`SIGUSR1` protocol these scripts implement.

## Files

```
env.sh                       # sourced by every script below: conda env, PYTHONPATH, threads
check_gpu.slurm              # gate 1: GPU/CUDA/torch sanity
smoke_test.slurm             # gate 2: check_env + 3k-step train + save/load round-trip + rollout
extract_object_params.slurm  # stage 1: prompt -> LLM -> configs/objects/<name>.json
train.slurm                  # stage 2: one run -- baseline cube, or OBJECT=<snapshot.json>
train_objects_array.slurm    # stage 2: all 6 objects as parallel array tasks
rollout.slurm                # stage 3: roll out one run dir
rollout_all_objects.slurm    # stage 3: roll out every lift_<object> run, success-rate table
```
