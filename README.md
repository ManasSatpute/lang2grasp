# lang2grasp

Stock Stable-Baselines3 (PPO / SAC) on **robosuite `Lift`** with a **Franka Emika Panda**
arm, through a native **Gymnasium** interface, plus an LLM-driven pipeline that turns a
text description of an object into the physical parameters (`ObjectParams`) that
parameterise the sim. Built for a SLURM cluster with 1 GPU and 12 CPU cores per job.

No custom rewards, no custom policies — a baseline you can trust before changing things.

## Layout

```
lang2grasp/
├── requirements.txt
├── src/
│   ├── common/
│   │   └── utils.py            # logging, seeding, device, threads, run dirs
│   ├── configs/
│   │   ├── policy/
│   │   │   ├── sac.json        # 4 env workers, gradient_steps=4 (1:1 replay ratio)
│   │   │   └── ppo.json        # 10 env workers + 2 learner threads
│   │   └── objects/
│   │       ├── prompts.json    # 6 named text prompts, one per object
│   │       └── <name>.json     # ObjectParams snapshots written by extract_object_params.py
│   ├── objects/
│   │   ├── object_params.py    # ObjectParams: shape/size/mass/friction, validated + clamped
│   │   └── lift_object_task.py # ParamLift: Lift with the cube replaced by an ObjectParams object
│   ├── extraction/
│   │   ├── llm_backends.py     # Mock / Anthropic / OpenAI / Groq extraction backends
│   │   ├── param_prompts.py    # extraction prompt, JSON schema, offline priors
│   │   ├── param_extraction.py # prompt -> validated ObjectParams, with one retry
│   │   └── deligrasp_sim/      # standalone DeliGrasp thinker->coder reproduction (force-adaptive grasp)
│   ├── rl/
│   │   ├── env.py              # robosuite -> gymnasium.Env wrapper
│   │   ├── vec_env.py          # VecEnv + VecNormalize construction (shared)
│   │   ├── config.py           # typed config, JSON load/snapshot
│   │   ├── callbacks.py        # checkpoint / eval / graceful pre-emption
│   │   ├── train.py            # training entrypoint, resume-aware
│   │   └── rollout.py          # load-and-roll-out entrypoint
│   ├── scripts/
│   │   ├── check_gpu.py            # GPU hello-world
│   │   ├── extract_object_params.py # stage 1: prompt -> LLM -> ObjectParams JSON
│   │   ├── train_object.py          # stage 2: train one object's SAC policy
│   │   ├── train_all_objects.py     # stage 2: local sequential driver, all 6 objects
│   │   ├── rollout_all_objects.py   # stage 3: roll out + success-rate table, all 6 objects
│   │   └── run_experiment.py        # standalone DeliGrasp benchmark runner (uses extraction/deligrasp_sim)
│   ├── tests/
│   │   └── smoke_test.py       # end-to-end: check_env + train + save/load round-trip + rollout
│   ├── slurm/                  # placeholder for cluster job scripts (currently empty; see note below)
│   └── results/                # placeholder for experiment output (currently empty)
```

No install step. Every entry point below is written to be run **from the repo root**
with `PYTHONPATH=src`, which makes `common`, `rl`, `objects`, `extraction`, `scripts`
and `tests` importable as top-level packages.

> **Note on `src/slurm/`:** this directory is currently empty. The `sbatch`/`.slurm`
> commands in this README describe the intended cluster workflow (and the code in
> `rl/train.py`/`rl/callbacks.py` that supports it — checkpoint-on-signal, exit-code
> 42 for requeue, etc.), but the actual `.slurm` job scripts have not been (re)added
> to this repo yet. Everything under "Run it" that shells out to `slurm/*.slurm` will
> need those scripts written first; training/rollout via plain `python` works today.

## LLM-parameterised objects: prompt → SAC policy → Panda rollout

Three stages, each a separate script so they can run independently (extraction
needs an LLM/network; training and rollout never do):

```bash
# 1. Prompt -> LLM -> physical parameters, snapshotted to src/configs/objects/<name>.json.
#    --backend mock is offline/deterministic (no API key); anthropic/openai/groq call a real LLM.
PYTHONPATH=src python src/scripts/extract_object_params.py --backend mock

# 2. Train one SAC policy per object. Locally, sequentially:
PYTHONPATH=src python src/scripts/train_all_objects.py --base-config src/configs/policy/sac.json
#    ...or one object at a time:
PYTHONPATH=src python src/scripts/train_object.py \
    --object src/configs/objects/raw_egg.json --base-config src/configs/policy/sac.json
#    ...or on the cluster, one SLURM array task per object (once slurm/train_objects_array.slurm exists):
sbatch slurm/train_objects_array.slurm

# 3. Roll every trained policy out against the Panda arm in robosuite/MuJoCo.
PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
```

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
three groups of fields: simulation fields (`shape`, `size`, `density`, `friction`)
that map directly onto robosuite's primitive objects, grasp-descriptive fields
(`mass_class`, `fragile`, `grip_force_min_N`/`max_N`) that are carried through the
pipeline as metadata — they don't currently change the SAC reward or physics, but
are the natural hook for a future force-adaptive grasp reward — and two fields
(`spring_Npm`, `crush_force_N`) used only by `extraction/deligrasp`'s standalone
grasp-simulation benchmark, which builds its ground truth directly from this same
schema (`extraction/param_prompts.py`'s `PRIORS`) so both pipelines describe
identical objects. `ParamLift`
(`src/objects/lift_object_task.py`) is a `robosuite.Lift` subclass whose
`_load_model` builds the object from `shape`/`size`/`density`/`friction` instead of
the stock red cube — every other `Lift` method (`reward`, `_check_success`, ...)
references `self.cube` generically and needs no changes. `EnvConfig.object` (in
`rl/env.py`) is `None` by default, so every existing config/test is byte-for-byte
unaffected; setting it switches `RobosuiteLiftEnv` from `suite.make("Lift", ...)` to
`suite.make("ParamLift", object_params=..., ...)`.

Extraction is deliberately decoupled from training: `extract_object_params.py`
writes a plain JSON snapshot of `ObjectParams`, and everything downstream --
including a SLURM node with no internet -- reads that snapshot. No training
run ever calls an LLM.

## One edit before you start

Each `.slurm` file (once added under `slurm/`) should set a `#SBATCH --partition=gpuL`
line, and `train.slurm` should set:

```bash
RUNS_DIR="/scratch/${USER}/lang2grasp_runs"      # <-- must be shared storage
```

Shared storage matters: a requeued job may land on a different node and has to find
its own checkpoints. Lustre and NFS are fine; node-local `/tmp` is not.

## Run it

```bash
sbatch slurm/check_gpu.slurm     # 1. gate: "hello world from cuda:0 ... sum = 27.0"
sbatch slurm/smoke_test.slurm    # 2. gate: "SMOKE TEST PASSED"

JOB=$(sbatch --parsable slurm/train.slurm)      # 3.
tail -f logs/lift_train_${JOB}.out

sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_${JOB}_s0 \
       slurm/rollout.slurm                       # 4.
```

PPO instead of SAC: `sbatch --export=ALL,CONFIG=src/configs/policy/ppo.json slurm/train.slurm`

Cancel: `scancel $JOB`. Watch: `squeue --me`, `tensorboard --logdir /scratch/$USER/lang2grasp_runs`

**The job will vanish from `squeue` and reappear with the same ID.** That is the requeue
working, not a crash.

Without a SLURM cluster, run the same steps directly:

```bash
PYTHONPATH=src python src/scripts/check_gpu.py
PYTHONPATH=src python src/tests/smoke_test.py --algo SAC --steps 3000

PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json
PYTHONPATH=src python -m rl.rollout --run-dir runs/SAC_local --episodes 10
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
replay_buffer.pkl      # SAC only; without it a resume restarts the critic cold
vecnormalize.pkl       # PPO only
checkpoints/model_50000_steps.zip ...
eval/best_model.zip, evaluations.npz
tb/tb_1/events.out.tfevents...
```

## Sizing for 12 cores + 1 GPU

|     | `n_envs` | `n_threads` | why |
|-----|---------:|------------:|-----|
| SAC | 4        | 4           | gradient-bound; `gradient_steps=4` keeps the replay ratio at 1:1 |
| PPO | 10       | 2           | throughput-bound; MuJoCo stepping is the bottleneck |

Each `.slurm` script should export `OMP_NUM_THREADS=1`. Not cosmetic: every
`SubprocVecEnv` worker links BLAS, and 10 workers × an unpinned OpenMP pool each will
thrash a 12-core cgroup and run **slower than a single environment**.

PPO's `n_steps × n_envs = 5120` divides exactly by `batch_size = 1280`. If you change
`n_envs`, fix `batch_size` too.

> **MuJoCo physics is CPU-only.** The GPU accelerates the policy network and offscreen
> rendering, nothing else. SAC is gradient-bound and benefits. PPO with a `[256,256]` MLP
> on state observations may be faster on CPU — benchmark `--device cpu` before assuming.

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
statistics that keep moving. Hence `normalize_obs: false` for SAC, `true` for PPO. At load
time `rollout.py` reloads `vecnormalize.pkl` with `training=False, norm_reward=False` — a PPO
policy fed raw observations performs at chance and looks like a training failure.

**`SubprocVecEnv` + `fork`.** MuJoCo GL contexts do not survive `fork()`. `vec_env.py` forces
`start_method="spawn"`.

## Expectations

`Lift` is the easy robosuite task, not a solved-in-ten-minutes one. With the shaped reward,
SAC typically shows a rising `eval/success_rate` in the low hundreds of thousands of steps.
PPO needs roughly an order of magnitude more samples. If `rollout/ep_rew_mean` climbs while
`eval/success_rate` stays at 0, the policy is farming the shaping term (usually hovering near
the cube) — a reward-hacking signal, not a bug in this code.

Single-seed RL results aren't evidence. Run three seeds before believing a curve.
