# lang2grasp

Stock Stable-Baselines3 **SAC** on **robosuite `Lift`** with a **Franka Emika Panda**
arm, through a native **Gymnasium** interface, plus an LLM-driven pipeline that turns a
text description of an object into the physical parameters (`ObjectParams`) that
parameterise the sim. Built for a SLURM cluster with 1 GPU and 12 CPU cores per job.

No custom rewards, no custom policies — a baseline you can trust before changing things.

## Contents

- [Layout](#layout)
- [Setup](#setup)
- [Pipeline: prompt → SAC policy → Panda rollout](#pipeline-prompt--sac-policy--panda-rollout)
- [Running on a Slurm cluster (CSF3)](#running-on-a-slurm-cluster-csf3)
- [Training that survives the wall clock](#training-that-survives-the-wall-clock)
- [Sizing for 12 cores + 1 GPU](#sizing-for-12-cores--1-gpu)
- [Things that will silently ruin a run](#things-that-will-silently-ruin-a-run)
- [Expectations](#expectations)

## Layout

```
lang2grasp/
├── requirements.txt
├── logs/                        # SBATCH output/error files land here (tracked via logs/.gitkeep)
├── runs/                        # default local training output dir (gitignored, created at runtime)
├── src/
│   ├── common/
│   │   └── utils.py             # logging, seeding, device, threads, run dirs
│   ├── configs/
│   │   ├── policy/
│   │   │   └── sac.json         # 4 env workers, gradient_steps=4 (1:1 replay ratio)
│   │   └── objects/
│   │       ├── prompts.json     # 6 named text prompts, one per object
│   │       └── <name>.json      # ObjectParams snapshots written by extract_object_params.py
│   ├── objects/
│   │   ├── object_params.py     # ObjectParams: shape/size/mass/friction, validated + clamped
│   │   └── lift_object_task.py  # ParamLift: Lift with the cube replaced by an ObjectParams object
│   ├── extraction/
│   │   ├── llm_backends.py      # Mock / Anthropic / OpenAI / Groq extraction backends
│   │   ├── param_prompts.py     # extraction prompt, JSON schema, offline priors
│   │   └── param_extraction.py  # prompt -> validated ObjectParams, with one retry
│   ├── rl/
│   │   ├── env.py               # robosuite -> gymnasium.Env wrapper
│   │   ├── vec_env.py           # VecEnv + VecNormalize construction (shared)
│   │   ├── config.py            # typed config, JSON load/snapshot
│   │   ├── callbacks.py         # checkpoint / eval / graceful pre-emption
│   │   ├── train.py             # training entrypoint, resume-aware
│   │   └── rollout.py           # load-and-roll-out entrypoint
│   ├── scripts/
│   │   ├── check_gpu.py             # GPU hello-world
│   │   ├── extract_object_params.py # stage 1: prompt -> LLM -> ObjectParams JSON
│   │   ├── train_object.py          # stage 2: train one object's SAC policy
│   │   ├── train_all_objects.py     # stage 2: local sequential driver, all 6 objects
│   │   ├── rollout_all_objects.py   # stage 3: roll out + results/plot/video, all 6 objects
│   │   ├── plot_rollout_results.py  # stage 3: success-rate/return plots (per-object + comparison)
│   │   └── compare_policies.py      # stage 3: generic baseline vs. per-object policies
│   ├── tests/
│   │   └── smoke_test.py        # end-to-end: check_env + train + save/load round-trip + rollout
│   ├── slurm/                   # CSF3 job scripts -- see "Running on a Slurm cluster" below
│   └── results/                 # rollout_all_objects.py / compare_policies.py output
```

No install step beyond the one-time environment setup below. Every entry point is
written to be run **from the repo root** with `PYTHONPATH=src`, which makes `common`,
`rl`, `objects`, `extraction`, `scripts` and `tests` importable as top-level packages.

## Setup

```bash
# 1. Conda, Python 3.10. On CSF3, conda comes from a miniforge3 module (no "anaconda"
#    module exists there) -- confirm names for your account with `module avail conda`
#    and `module avail cuda` if they differ from below:
module load apps/binapps/conda/miniforge3/25.9.1
module load cuda/12.6.2

conda create -n lang2grasp python=3.10 -y
conda activate lang2grasp

# 2. Torch first, matched to the CUDA module above, so pip doesn't silently replace it:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. Everything else -- robosuite/mujoco/gymnasium/stable-baselines3/tensorboard,
#    plus matplotlib for rollout_all_objects.py's --plot. anthropic/openai/groq
#    (for real LLM extraction backends) are commented out in requirements.txt --
#    uncomment only the one(s) you use.
pip install -r requirements.txt
```

Python 3.10 is a good choice here: `requirements.txt` pins `numpy<2.0` for robosuite
1.4.x/1.5.x's ABI, which 3.10 supports cleanly.

Re-run both `module load` lines and `conda activate lang2grasp` in every new shell (a
submitted Slurm job does this via `src/slurm/env.sh` automatically). If a CSF3 software
update changes a module's version string, `module avail conda`/`module avail cuda` will
show the new name -- update `src/slurm/env.sh`'s `CONDA_MODULE`/`CUDA_MODULE` (and
rebuild the env against the new CUDA version if it changed) to match.

None of this requires a Slurm cluster -- everything below also runs with plain
`python` on a laptop with a GPU (or CPU, just slower).

## Pipeline: prompt → SAC policy → Panda rollout

Three stages, each a separate script so they can run independently (extraction needs
an LLM/network; training and rollout never do):

```bash
# 1. Prompt -> LLM -> physical parameters, snapshotted to src/configs/objects/<name>.json.
#    --backend mock is offline/deterministic (no API key); anthropic/openai/groq call a real LLM.
PYTHONPATH=src python src/scripts/extract_object_params.py --backend mock

# 2. Train one SAC policy per object. Locally, sequentially:
PYTHONPATH=src python src/scripts/train_all_objects.py --base-config src/configs/policy/sac.json
#    ...or one object at a time:
PYTHONPATH=src python src/scripts/train_object.py \
    --object src/configs/objects/raw_egg.json --base-config src/configs/policy/sac.json
#    ...or on a Slurm cluster, one array task per object -- see "Running on a Slurm
#    cluster" below:
sbatch src/slurm/train_objects_array.slurm

# 3. Roll every trained policy out against the Panda arm in robosuite/MuJoCo.
#    Always writes src/results/rollout_results.csv (all objects, full metric set).
PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
#    --plot adds a success-rate/return chart; --video adds a short rollout video per
#    object (src/results/videos/<object>.mp4, --video-episodes controls the length).
#    Both opt-in: rendering is slow and --video additionally needs a working offscreen
#    GL context (EGL/OSMesa/GLFW -- set MUJOCO_GL if the default doesn't work headless).
PYTHONPATH=src python src/scripts/rollout_all_objects.py --plot --video
```

**Does extraction actually help?** `compare_policies.py` answers that by rolling a
single *generic* policy (trained once, with `env.object` unset -- so neither its
physics nor its reward ever sees an extracted parameter) out against each object's
*real* physics, alongside that object's own dedicated policy from stage 2:

```bash
# Train the generic baseline once (this is just rl.train with no --object):
PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json --run-name lift_baseline

# Compare it against the per-object policies trained above:
PYTHONPATH=src python src/scripts/compare_policies.py \
    --baseline-run-dir runs/lift_baseline --plot
```

Writes `src/results/policy_comparison.csv` (always) and, with `--plot`,
`policy_comparison.png` -- a grouped success-rate/return chart, generic vs.
object-aware, per object. Per-object training can also optionally turn on
grip-force-aware reward shaping for a 3-way comparison:
`train_object.py --grip-force-shaping` (off by default, same as `EnvConfig`'s own
default -- see "How it fits together" below).

**The 6 default objects** (`src/configs/objects/prompts.json`) span the axes that matter
for grasping, not just geometry — fragile vs. rugged, light vs. heavy, slick vs.
grippy:

| object | shape | fragile | grip force (N) | notes |
|---|---|---|---|---|
| `glass_bottle` | cylinder | yes | 2–8 | thin-walled, slippery |
| `steel_bolt` | cylinder | no | 10–60 | small but dense, grips well |
| `ceramic_mug` | cylinder | yes | 3–12 | |
| `rice_bag` | box | no | 8–50 | rugged, doesn't care about grip force |
| `raw_egg` | ball | yes | 1–4 | narrow safe force window |
| `brick` | box | no | 15–80 | heavy |

**How it fits together.** `ObjectParams` (`src/objects/object_params.py`) holds
simulation fields (`shape`, `size`, `density`, `friction`) that map directly onto
robosuite's primitive objects; descriptive fields (`mass_class`, `fragile`) carried
through the pipeline as metadata; and force fields (`grip_force_min_N`/`max_N`,
`spring_Npm`, `crush_force_N`) that drive this training pipeline's **optional**
grip-force-aware reward shaping (`EnvConfig.grip_force_shaping`, off by default —
see `rl/env.py`'s module docstring for how contact force is estimated from gripper
aperture). `ParamLift`
(`src/objects/lift_object_task.py`) is a `robosuite.Lift` subclass whose `_load_model`
builds the object from `shape`/`size`/`density`/`friction` instead of the stock red
cube — every other `Lift` method (`reward`, `_check_success`, ...) references
`self.cube` generically and needs no changes. `EnvConfig.object` (in `rl/env.py`) is
`None` by default, so every existing config/test is byte-for-byte unaffected; setting
it switches `RobosuiteLiftEnv` from `suite.make("Lift", ...)` to
`suite.make("ParamLift", object_params=..., ...)`.

Extraction is deliberately decoupled from training: `extract_object_params.py` writes
a plain JSON snapshot of `ObjectParams`, and everything downstream — including a
SLURM node with no internet — reads that snapshot. No training run ever calls an LLM.

## Running on a Slurm cluster (CSF3)

> **Scheduler check first.** The job scripts below are Slurm (`#SBATCH`, `sbatch`,
> `scontrol`). CSF3 has historically run Grid Engine (`qsub`, `#$ -l ...`) instead.
> Run `which sbatch` on a CSF3 login node before submitting anything — if it's not
> found, these need Grid Engine equivalents, not these files as-is.

All scripts under `src/slurm/` source `src/slurm/env.sh` first, which handles the
conda env, `PYTHONPATH`, and thread pinning from [Setup](#setup) above — most of
what's cluster-specific lives in that one file.

### Edit before submitting anything

1. **`src/slurm/env.sh`**: `CONDA_MODULE`/`CUDA_MODULE` are already set to the values
   from Setup. Only touch these if CSF3's module names change.
2. **Every `.slurm` file**: `#SBATCH --partition=gpuL` is a placeholder — set it to
   your allocation's actual GPU partition. `extract_object_params.slurm` uses
   `<CPU_PARTITION>` instead, since that stage needs no GPU.
3. **`train.slurm` / `train_objects_array.slurm`**: set
   ```bash
   RUNS_DIR="/scratch/${USER}/lang2grasp_runs"      # <-- must be shared storage
   ```
   Shared storage matters: a requeued job may land on a different node and has to
   find its own checkpoints. Lustre and NFS are fine; node-local `/tmp` is not.
4. If your account needs an `--account`/`--qos` line, add it to each file.

### Usage (from the repo root)

```bash
# Gates -- run once, in order, before trusting anything below.
sbatch src/slurm/check_gpu.slurm     # gate 1: "hello world from cuda:0 ... sum = 27.0"
sbatch src/slurm/smoke_test.slurm    # gate 2: "SMOKE TEST PASSED"

# Stage 1: prompt -> LLM -> ObjectParams JSON. See extract_object_params.slurm's own
# header for the network-access caveat with real (non-mock) backends.
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
sbatch --export=ALL,PLOT=1,VIDEO=1 src/slurm/rollout_all_objects.slurm   # + plot + per-object video

# Stage 3, comparison: the "Baseline (stock Lift cube)" job above is the generic
# policy -- compare it against the per-object runs from train_objects_array.slurm:
sbatch --export=ALL,BASELINE_RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0,PLOT=1 \
    src/slurm/compare_policies.slurm

# Cancel / watch, same as any Slurm job:
scancel $JOB
squeue --me
tensorboard --logdir /scratch/$USER/lang2grasp_runs
```

**A `train*` job vanishing from `squeue` and reappearing with the same ID is the
requeue mechanism working, not a crash** — see
[Training that survives the wall clock](#training-that-survives-the-wall-clock) for
the full exit-code-42 / `SIGUSR1` protocol these scripts implement.

Without a SLURM cluster, run the same steps directly:

```bash
PYTHONPATH=src python src/scripts/check_gpu.py
PYTHONPATH=src python src/tests/smoke_test.py --steps 3000

PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json
PYTHONPATH=src python -m rl.rollout --run-dir runs/SAC_local --episodes 10
```

### Job script files

```
src/slurm/
  env.sh                       # sourced by every script below: conda env, PYTHONPATH, threads
  check_gpu.slurm              # gate 1: GPU/CUDA/torch sanity
  smoke_test.slurm             # gate 2: check_env + 3k-step train + save/load round-trip + rollout
  extract_object_params.slurm  # stage 1: prompt -> LLM -> configs/objects/<name>.json
  train.slurm                  # stage 2: one run -- baseline cube, or OBJECT=<snapshot.json>
  train_objects_array.slurm    # stage 2: all 6 objects as parallel array tasks
  rollout.slurm                # stage 3: roll out one run dir
  rollout_all_objects.slurm    # stage 3: roll out every lift_<object> run, results/plot/video
  compare_policies.slurm       # stage 3: generic baseline vs. per-object policies
```

## Training that survives the wall clock

A 1M-step SAC run does not finish in one 4-hour slot.

- `#SBATCH --signal=B:USR1@300` warns the batch shell 300s before the kill. The `B:`
  prefix matters — without it the signal goes to job *steps*, and these scripts don't
  use `srun`, so the trap would never fire.
- The script traps `USR1`, forwards it to python. `GracefulExitCallback` sets a flag,
  `learn()` unwinds, and `train()`'s `finally` writes model + replay buffer + VecNormalize.
- `train.py` exits **42** = "checkpointed cleanly, work remains" → `scontrol requeue`.
  Any other non-zero exit is a real failure and is *not* requeued.
- The requeued job keeps the same `$SLURM_JOB_ID`, so the run name is stable, so
  `--resume` finds the checkpoints.

Keep `max_hours` in the JSON config ~15 min below `#SBATCH --time`.

Artifacts, per run (under `runs/<run_name>/`):

```
config.json            # written once; a resumed job keeps its original config
final_model.zip
replay_buffer.pkl      # without it a resume restarts the critic cold
vecnormalize.pkl       # only present if normalize_obs/normalize_reward is on
checkpoints/model_50000_steps.zip ...
eval/best_model.zip, evaluations.npz
tb/tb_1/events.out.tfevents...
```

## Sizing for 12 cores + 1 GPU

|     | `n_envs` | `n_threads` | why |
|-----|---------:|------------:|-----|
| SAC | 8        | 4           | gradient-bound; `gradient_steps=4` keeps the replay ratio at 1:1 |

Each `.slurm` script should export `OMP_NUM_THREADS=1`. Not cosmetic: every
`SubprocVecEnv` worker links BLAS, and several workers × an unpinned OpenMP pool each will
thrash a 12-core cgroup and run **slower than a single environment**.

> **MuJoCo physics is CPU-only.** The GPU accelerates the policy network, nothing else.
> SAC is gradient-bound and benefits from it; benchmark `--device cpu` before assuming
> the GPU actually helps your particular `net_arch`.

## Things that will silently ruin a run

**`terminated` vs `truncated`.** robosuite raises `done` at the horizon. That is truncation.
Report it as termination and SB3 bootstraps a zero value at every cut-off, biasing the value
function on every episode. `rl/env.py` computes both flags itself. There's a regression test.

**The sparse default reward.** `Lift` defaults to `reward_shaping=False`. Random exploration
on a 7-DoF arm essentially never lifts the cube, so the gradient is zero and the loss curve
looks "stable" while nothing learns. The configs default to `reward_shaping: true`.

**Resuming without the replay buffer.** SAC reloaded with an empty buffer unlearns its critic;
the return curve craters at every requeue boundary.

**`reset_num_timesteps=False` is additive.** SB3's `_setup_learn` does
`total_timesteps += self.num_timesteps`. Passing the global budget on a resume trains for
`budget + already_done` steps. `train.py` passes the *remaining* budget.

**`VecNormalize` + off-policy replay.** A replay buffer holds observations normalised under
statistics that keep moving, so `normalize_obs`/`normalize_reward` default to `false`. If you
turn them on, `rollout.py` reloads `vecnormalize.pkl` with `training=False, norm_reward=False` —
a policy fed raw (unnormalised) observations after training with normalisation on performs at
chance and looks like a training failure.

**`SubprocVecEnv` + `fork`.** MuJoCo GL contexts do not survive `fork()`. `vec_env.py` forces
`start_method="spawn"`.

## Expectations

`Lift` is the easy robosuite task, not a solved-in-ten-minutes one. With the shaped reward,
SAC typically shows a rising `eval/success_rate` in the low hundreds of thousands of steps.
If `rollout/ep_rew_mean` climbs while `eval/success_rate` stays at 0, the policy is farming
the shaping term (usually hovering near the cube) — a reward-hacking signal, not a bug in
this code.

Single-seed RL results aren't evidence. Run three seeds before believing a curve.

