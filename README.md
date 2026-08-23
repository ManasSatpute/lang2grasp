# lang2grasp

Stock Stable-Baselines3 **SAC** on **robosuite `Lift`** with a **Franka Emika Panda**
arm, through a native **Gymnasium** interface, plus an LLM-driven pipeline that turns a
text description of an object into the physical parameters (`ObjectParams`) that
parameterise the sim. Built for a SLURM cluster with 1 GPU and 12 CPU cores per job.

No custom policies in the base pipeline — a baseline you can trust before changing
things. Reward shaping is limited to a genuine fingertip-force-sensor-driven crush
penalty/termination for LLM-described objects (see "How it fits together" below); the
stock cube baseline is otherwise untouched robosuite `Lift`. Two custom feature
extractors do exist for the paradigm-switch experiment track (see below) — they're
opt-in, off the baseline pipeline entirely.

## Contents

- [Layout](#layout)
- [Setup](#setup)
- [Pipeline: prompt → SAC policy → Panda rollout](#pipeline-prompt--sac-policy--panda-rollout)
- [Paradigm switch: domain randomization vs. per-object specialists](#paradigm-switch-domain-randomization-vs-per-object-specialists)
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
│   │       ├── <name>.json      # ObjectParams snapshots written by extract_object_params.py
│   │       └── width_mass_set/  # 5 objects isolating width vs. mass, see generate_width_mass_objects.py
│   ├── objects/
│   │   ├── object_params.py     # ObjectParams: shape/size/mass/friction, validated + clamped
│   │   ├── lift_object_task.py  # ParamLift: Lift with the cube replaced by an ObjectParams object
│   │   └── force_gripper.py     # PandaGripperForce: adds a real per-fingertip MuJoCo force sensor
│   ├── extraction/
│   │   ├── llm_backends.py      # Anthropic / OpenAI / Groq extraction backends
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
│   │   ├── extract_object_params.py # stage 1 (+1.5): prompt -> LLM -> ObjectParams JSON, then accuracy check
│   │   ├── evaluate_extraction_accuracy.py # stage 1.5, standalone: re-check/re-plot without re-extracting
│   │   ├── generate_width_mass_objects.py # stage 1 (analytic): the width/mass-matched 5-object set
│   │   ├── train_object.py          # stage 2: train one object's SAC policy
│   │   ├── train_all_objects.py     # stage 2: local sequential driver, all objects in a dir
│   │   ├── rollout_all_objects.py   # stage 3: roll out + results/plot/video, all objects
│   │   ├── plot_rollout_results.py  # stage 3: success-rate/return plots (per-object + comparison)
│   │   └── compare_policies.py      # stage 3: generic baseline vs. per-object policies
│   ├── tests/
│   │   ├── smoke_test.py        # end-to-end: check_env + train + save/load round-trip + rollout
│   │   └── force_sensor_test.py # fingertip force sensor + crush penalty/termination + object-set check
│   ├── slurm/                   # CSF3 job scripts -- see "Running on a Slurm cluster" below
│   └── results/                 # evaluate_extraction_accuracy.py / rollout_all_objects.py / compare_policies.py output
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

**LLM API keys** (`extract_object_params.py --backend` is required -- `anthropic`,
`openai`, and `groq` all need a key):

```bash
cp .env.example .env
# then edit .env and fill in the key(s) for the backend(s) you use, e.g.:
#   GROQ_API_KEY=gsk_...
```

`.env` is gitignored -- real keys never get committed. `extraction/llm_backends.py`
loads it automatically (via `python-dotenv`, in `requirements.txt`) whenever it's
imported, so no `export`ing or `sbatch --export=...`ing keys by hand. A real
environment variable, if one is already set, always wins over `.env`.

## Pipeline: prompt → SAC policy → Panda rollout

Three stages, each a separate script so they can run independently (extraction needs
an LLM/network; training and rollout never do). Stage 1 folds in an accuracy check
against the golden dataset (formerly a separate "stage 1.5") right after extraction,
since it only needs stage 1's own output and there's no reason to wait on training/
rollout to see it:

```bash
# 1. Prompt -> LLM -> physical parameters, snapshotted to src/configs/objects/<name>.json,
#    immediately followed by an accuracy check against the golden (ground-truth) dataset.
#    --backend is required: anthropic/openai/groq all call a real LLM. Always writes
#    src/results/extraction_accuracy_{detail,summary}.csv; --plot adds
#    src/results/extraction_accuracy.png. --no-evaluate skips the accuracy check.
PYTHONPATH=src python src/scripts/extract_object_params.py --backend groq --plot

#    --samples N draws N independent extractions per object and takes the median
#    (numeric fields) / majority vote (categorical fields) across them instead of
#    trusting a single call -- even at temperature 0, a single call has real sample-
#    to-sample noise on the harder-to-calibrate fields (grip/crush force especially).
#    Costs N backend calls per object; logs which fields actually disagreed and by
#    how much. See extraction/param_extraction.py's module docstring for details.
PYTHONPATH=src python src/scripts/extract_object_params.py --backend groq --samples 5

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
object-aware, per object. Per-object training can also optionally turn on the
grip-force safe-hold bonus for a 3-way comparison: `train_object.py
--grip-force-shaping` (off by default, same as `EnvConfig`'s own default -- see "How
it fits together" below). This is a bonus term only -- crush penalty/termination are
unconditional whenever `--object` is set and aren't affected by this flag.

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

These 6 confound geometry and mass — `rest_width_mm` predicts rollout success much
more strongly than `mass_g` does across this set, and nothing here holds one axis
fixed while varying the other. See "Width-matched / mass-matched object set" below
for a set that does.

**How it fits together.** `ObjectParams` (`src/objects/object_params.py`) holds
simulation fields (`shape`, `size`, `density`, `friction`) that map directly onto
robosuite's primitive objects; descriptive fields (`mass_class`, `fragile`) carried
through the pipeline as metadata; and force fields (`grip_force_min_N`/`max_N`,
`crush_force_N`) that drive this training pipeline's reward, keyed off a **genuine
MuJoCo fingertip force sensor** (`objects/force_gripper.py`'s `PandaGripperForce`, a
3-axis `<force>` sensor on each finger pad) — not an estimate, and its 6-dim reading
(`fingertip_force`) is part of the observation for *every* config, including the
generic baseline with no `ObjectParams` at all. Crush penalty/termination
(`EnvConfig.crush_penalty_coeff`/`terminate_on_crush`) are unconditional whenever
`env.object` is set; `EnvConfig.grip_force_shaping` (off by default) only adds an
extra bonus for staying within `grip_force_min_N`/`max_N` — see `rl/env.py`'s module
docstring for the full mechanism.

**Golden physics vs. extracted perception.** `EnvConfig.object` is what's actually
built into the MuJoCo scene — the real/"golden" object. `train_object.py`/
`train_all_objects.py` load it from `extraction.param_prompts.golden_object_params`
(the `PRIORS` table) when a golden entry exists for that object's name, and set the
*extracted* snapshot (`scripts/extract_object_params.py`'s output — potentially an
imperfect LLM guess) as a separate `EnvConfig.extracted_object` instead. Crush
penalty/termination, the grip-force bonus, and `include_object_z`'s z-vector are all
computed from `extracted_object` when it's set, falling back to `object` otherwise —
i.e. the policy is trained against its (possibly wrong) *belief* about the object,
while what it's actually lifting (and how much it actually masses/how it actually
slides) is the golden object. Objects with no golden entry (e.g. `width_mass_set`,
generated analytically rather than extracted) keep the old coupled behaviour
(`extracted_object=None`, so perception falls back to `object`). `ParamLift`
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

### Width-matched / mass-matched object set

The 6 objects above vary shape, mass *and* size together, so a result that looks like
"heavier objects are harder to lift" can't be told apart from "wider objects are
harder to grasp." `generate_width_mass_objects.py` generates 5 cylinders that hold one
axis fixed at a time:

| object | width | mass |
|---|---|---|
| `width40_mass050g` | 40mm | 50g |
| `width40_mass200g` | 40mm | 200g *(shared anchor)* |
| `width40_mass500g` | 40mm | 500g |
| `width25_mass200g` | 25mm | 200g |
| `width55_mass200g` | 55mm | 200g |

`{width40_mass050g, width40_mass200g, width40_mass500g}` isolates mass (width
constant); `{width25_mass200g, width40_mass200g, width55_mass200g}` isolates width
(mass constant). Every other field (friction, fragile, grip force window,
`crush_force_N`) is held constant across all 5, so width/mass are the only things
that differ between objects in this set. Deterministic and analytic (radius/
half-height/density solved in closed form from the target width/mass), not
LLM-extracted — these are specified physical points, not free-text descriptions.

```bash
PYTHONPATH=src python src/scripts/generate_width_mass_objects.py
#   -> src/configs/objects/width_mass_set/{width40_mass050g,width40_mass200g,...}.json

# Same pipeline as the 6 narrative objects, just pointed at this directory:
PYTHONPATH=src python src/scripts/train_all_objects.py \
    --objects-dir src/configs/objects/width_mass_set --base-config src/configs/policy/sac.json
PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
```

## Paradigm switch: domain randomization vs. per-object specialists

Everything above trains one SAC *specialist* per object (`train_object.py`/
`train_all_objects.py`): a fixed `ObjectParams` baked into the env for that whole run.
That's an oracle topline, not something that scales — a specialist has no way to
handle an object it wasn't trained on. `scripts/train_paradigm.py` trains three
policies against a *continuous distribution* of objects instead (a fresh
`ObjectParams` sampled every episode — shape/size/density/friction — via
`EnvConfig.randomize_object`/`objects.object_params.sample_object_params`), differing
only in what each is allowed to see:

| variant | sees | feature extractor | question it answers |
|---|---|---|---|
| `blind` | proprioception + object pose + fingertip force (same as the baseline obs) | stock `MlpPolicy` | memoryless floor — no way to tell objects apart within an episode |
| `blind_hist` | `blind`'s obs + a GRU over the last `--history-len` (default 16) steps of proprioception + force | `rl.policies.HistoryGRUExtractor` | the honest ceiling — implicit system identification from how the arm's own sensors responded, no cheating via a ground-truth parameter |
| `param` | `blind`'s obs + a (possibly noisy) object-parameter vector `z` | `rl.policies.FiLMExtractor` | the informed upper bound — told approximately what it's holding |

`z` (`objects.object_params.object_params_to_z`/`Z_DIM`) is a fixed-width, shape-agnostic
encoding: one-hot shape, size zero-padded to the widest shape's dimensionality, density,
friction. `mass_kg` is deliberately **not** in `z` — it's `density * volume_m3`, a
deterministic function of two dims already in `z`, so including it too would just hand
the FiLM layer a redundant, perfectly-collinear input. `param`'s `z` is noised
(`--z-noise-std`, default 0.1 relative) on its continuous dims only — it's meant to model
a noisy sysID-style estimate, not ground truth.

```bash
PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind \
    --base-config src/configs/policy/sac.json
PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind_hist \
    --base-config src/configs/policy/sac.json --history-len 16
PYTHONPATH=src python src/scripts/train_paradigm.py --variant param \
    --base-config src/configs/policy/sac.json --z-noise-std 0.1
```

A domain-randomized episode needs a MuJoCo recompile (shape/size can change), so these
runs use `hard_reset=True` internally — markedly slower per reset than the fixed-object
specialist path (`hard_reset=False`, reuses the compiled model). Budget accordingly.

Evaluating a trained `blind`/`blind_hist`/`param` policy against one specific real
object (rather than the training distribution) reuses the existing rollout machinery:
`rl.rollout.rollout(..., object_override=params)`, with `randomize_object` turned back
off on the override (`EnvConfig.object` and `EnvConfig.randomize_object` are mutually
exclusive) — there isn't yet a dedicated 3-way comparison script analogous to
`compare_policies.py` for this track.

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
   your allocation's actual GPU partition. `extract_object_params.slurm` and
   `evaluate_extraction_accuracy.slurm` use `<CPU_PARTITION>` instead, since neither
   stage needs a GPU.
3. **`train.slurm` / `train_objects_array.slurm` / `train_paradigm.slurm` /
   `train_paradigm_array.slurm`**: set
   ```bash
   RUNS_DIR="/scratch/${USER}/lang2grasp_runs"      # <-- must be shared storage
   ```
   Shared storage matters: a requeued job may land on a different node and has to
   find its own checkpoints. Lustre and NFS are fine; node-local `/tmp` is not.
4. If your account needs an `--account`/`--qos` line, add it to each file.

### Usage (from the repo root)

```bash
# Gates -- run once, in order, before trusting anything below.
sbatch src/slurm/check_gpu.slurm         # gate 1: "hello world from cuda:0 ... sum = 27.0"
sbatch src/slurm/smoke_test.slurm        # gate 2: "SMOKE TEST PASSED"
sbatch src/slurm/force_sensor_test.slurm # gate 3: "FORCE SENSOR TEST PASSED"

# Stage 1 (+ 1.5): prompt -> LLM -> ObjectParams JSON, immediately followed by the
# accuracy check against the golden dataset. BACKEND is required -- see
# extract_object_params.slurm's own header for the network-access caveat.
sbatch --export=ALL,BACKEND=groq src/slurm/extract_object_params.slurm
sbatch --export=ALL,BACKEND=groq,PLOT=1 src/slurm/extract_object_params.slurm   # + accuracy chart
# Width/mass-matched set (analytic, no LLM/network involved -- cheap enough to just
# run directly on the login node instead of via sbatch):
PYTHONPATH=src python src/scripts/generate_width_mass_objects.py

# Stage 1.5 standalone: re-check/re-plot existing snapshots without calling an LLM
# again (extract_object_params.slurm above already runs this once automatically).
sbatch src/slurm/evaluate_extraction_accuracy.slurm
sbatch --export=ALL,PLOT=1 src/slurm/evaluate_extraction_accuracy.slurm   # + accuracy chart

# Stage 2: train. Baseline (stock Lift cube):
JOB=$(sbatch --parsable src/slurm/train.slurm)
tail -f logs/lift_train_${JOB}.out
# One LLM-described object:
sbatch --export=ALL,OBJECT=src/configs/objects/raw_egg.json src/slurm/train.slurm
# All 6 objects, one array task each:
sbatch src/slurm/train_objects_array.slurm
# The width/mass-matched set instead (5 objects -- note --array=0-4, one less than
# the default):
sbatch --array=0-4 --export=ALL,OBJECTS_DIR=src/configs/objects/width_mass_set \
    src/slurm/train_objects_array.slurm

# Stage 2, paradigm switch (see "Paradigm switch" above): smoke-check on real
# hardware first (~2 min; there's no dedicated tests/ script for this path --
# a tiny TOTAL_TIMESTEPS run is the check):
sbatch --time=00:15:00 --export=ALL,VARIANT=blind,TOTAL_TIMESTEPS=2000 \
    src/slurm/train_paradigm.slurm
# One variant at a time:
sbatch --export=ALL,VARIANT=blind      src/slurm/train_paradigm.slurm
sbatch --export=ALL,VARIANT=blind_hist src/slurm/train_paradigm.slurm
sbatch --export=ALL,VARIANT=param      src/slurm/train_paradigm.slurm
# ...or all three as one array job (index 0/1/2 = blind/blind_hist/param):
sbatch src/slurm/train_paradigm_array.slurm

# Stage 3: rollout.
sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0 \
    src/slurm/rollout.slurm
sbatch src/slurm/rollout_all_objects.slurm   # every lift_<object> run under RUNS_DIR
sbatch --export=ALL,PLOT=1,VIDEO=1 src/slurm/rollout_all_objects.slurm   # + plot + per-object video
# rollout.slurm is generic -- point it at a paradigm run dir the same way:
sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/paradigm_blind_s0,EPISODES=20 \
    src/slurm/rollout.slurm

# Stage 3, live view: watch the MuJoCo viewer from your own machine while it runs
# on a headless CSF3 node, via a VNC session -- see "Watching a rollout live" below.
sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0 \
    src/slurm/rollout_vnc.slurm

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
PYTHONPATH=src python src/tests/force_sensor_test.py

PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json
PYTHONPATH=src python -m rl.rollout --run-dir runs/SAC_local --episodes 10

# Paradigm switch (see "Paradigm switch" above):
PYTHONPATH=src python src/tests/paradigm_test.py
PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind --total-timesteps 2000  # smoke check
PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind
PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind_hist --history-len 16
PYTHONPATH=src python src/scripts/train_paradigm.py --variant param --z-noise-std 0.1
PYTHONPATH=src python -m rl.rollout --run-dir runs/paradigm_blind --episodes 20
```

### Job script files

```
src/slurm/
  env.sh                       # sourced by every script below: conda env, PYTHONPATH, threads
  check_gpu.slurm              # gate 1: GPU/CUDA/torch sanity
  smoke_test.slurm             # gate 2: check_env + 3k-step train + save/load round-trip + rollout
  force_sensor_test.slurm      # gate 3: fingertip force sensor + crush + width_mass_set (CPU-only)
  extract_object_params.slurm  # stage 1 (+1.5): prompt -> LLM -> configs/objects/<name>.json, then accuracy check
  evaluate_extraction_accuracy.slurm # stage 1.5, standalone: re-check/re-plot into results/ without re-extracting
  train.slurm                  # stage 2: one run -- baseline cube, or OBJECT=<snapshot.json>
  train_objects_array.slurm    # stage 2: all objects in OBJECTS_DIR as parallel array tasks
  train_paradigm.slurm         # stage 2, paradigm switch: one VARIANT=blind|blind_hist|param run
  train_paradigm_array.slurm   # stage 2, paradigm switch: all 3 variants as parallel array tasks
  rollout.slurm                # stage 3: roll out one run dir (paradigm runs too -- it's generic)
  rollout_all_objects.slurm    # stage 3: roll out every lift_<object> run, results/plot/video
  rollout_vnc.slurm            # stage 3: live MuJoCo viewer over VNC, see below
  compare_policies.slurm       # stage 3: generic baseline vs. per-object policies
```

### Watching a rollout live

`rollout.slurm`/`rollout_all_objects.slurm` render *offscreen* (`--video`) — a saved
`.mp4` you `scp` down afterwards. To watch the actual MuJoCo viewer window update in
real time while the job runs on a headless CSF3 node:

```bash
sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0 \
    src/slurm/rollout_vnc.slurm
tail -f logs/lift_rollout_vnc_<jobid>.out   # prints the node, VNC port, and password
```

Then, from your own machine, tunnel through the login node to the compute node the
job landed on and connect a VNC viewer (e.g. TigerVNC Viewer) to `localhost:<port>`:

```bash
ssh -L <port>:localhost:<port> -J <you>@<csf3-login-host> <you>@<compute-node>
```

`rollout_vnc.slurm` tries a `turbovnc`/`tigervnc` module first, then falls back to
`Xvfb`+`x11vnc` if both are on `$PATH`. **This is unverified against your actual CSF3
allocation** — module names and whether compute nodes accept a second `ssh` hop are
cluster-specific. If it fails, check CSF3's own docs for a "VNC"/"remote desktop"
session (most HPC centres provide one) and run
`python -m rl.rollout --run-dir "$RUN_DIR" --render` (see `rollout.py`'s new
`--render` flag) inside that session instead.

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

**Every checkpoint trained before the fingertip force sensor is now incompatible.**
`fingertip_force` (6 dims) was added to `DEFAULT_OBS_KEYS` for every config, baseline
included, so `obs_dim` grew for every run -- an old `final_model.zip`'s policy network
has the wrong input shape for the current env and won't load/reload against it.
Retrain the baseline and every object. Old `config.json` snapshots referencing the
former `EnvConfig.crush_penalty` field also won't reload as-is (renamed to
`crush_penalty_coeff` with different units -- flat penalty vs. per-Newton
coefficient); this only matters if you're hand-editing an old snapshot rather than
generating a fresh one.

**`terminated` vs `truncated`.** robosuite raises `done` at the horizon. That is truncation.
Report it as termination and SB3 bootstraps a zero value at every cut-off, biasing the value
function on every episode. `rl/env.py` computes both flags itself. There's a regression test.

**The sparse default reward.** `Lift` defaults to `reward_shaping=False`. Random exploration
on a 7-DoF arm essentially never lifts the cube, so the gradient is zero and the loss curve
looks "stable" while nothing learns. The configs default to `reward_shaping: true`. If you do
train with `reward_shaping: false`, `train.py` calls `rl/sparse_seed.py` on fresh (non-resumed)
runs: a scripted, non-learned reach/descend/grasp/lift heuristic rolls out against a throwaway
env to find real successes and inserts their transitions directly into SAC's replay buffer,
so the critic has non-zero reward to bootstrap from instead of waiting on luck. It only runs
for `reward_shaping=False`; the default shaped path never touches it. Skipped (with a log
warning) if `normalize_obs`/`normalize_reward` is on, since seeded transitions would need
`VecNormalize` stats applied too.

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

**The derived force-sensor gripper XML is version-qualified, not path-qualified.**
`objects/force_gripper.py` builds `PandaGripperForce`'s XML once per installed
robosuite version and caches it at
`src/objects/_generated/panda_gripper_force_<version>.xml`; every process (every
`SubprocVecEnv` worker, every `train_objects_array.slurm` array task, possibly on
different compute nodes sharing the same NFS/GPFS-mounted repo checkout) checks for
that exact file before rebuilding it, so in the steady state only the first caller on
the whole cluster actually writes it. If you ever run two different robosuite
versions against the same checkout (e.g. mid-upgrade), each gets its own cache file
instead of one silently clobbering the other. This directory is gitignored and safe
to delete — it's rebuilt on demand.

**`force_sensor_test.slurm` (gate 3) matters more on a cluster than it looks.** It was
developed and verified against robosuite 1.5.2; `requirements.txt` allows
`>=1.4.1,<1.6`. `objects/force_gripper.py` assumes specific body names
(`finger_joint1_tip`/`finger_joint2_tip`) inside the installed `panda_gripper.xml` —
if CSF's conda env resolves to a version whose gripper XML differs, this gate fails
loudly with a clear `RuntimeError` naming the missing body, rather than a real
training job silently getting a broken (or all-zero) `fingertip_force`. Run
`pip show robosuite` in the activated env to see what actually resolved, and run this
gate before trusting anything downstream of it.

## Expectations

`Lift` is the easy robosuite task, not a solved-in-ten-minutes one. With the shaped reward,
SAC typically shows a rising `eval/success_rate` in the low hundreds of thousands of steps.
If `rollout/ep_rew_mean` climbs while `eval/success_rate` stays at 0, the policy is farming
the shaping term (usually hovering near the cube) — a reward-hacking signal, not a bug in
this code.

Single-seed RL results aren't evidence. Run three seeds before believing a curve.

