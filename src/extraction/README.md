# DeliGrasp — LLM pipeline, simulation-only reproduction

A self-contained reproduction of the LLM half of *DeliGrasp: Inferring Object
Properties with LLMs for Adaptive Grasp Policies* (CoRL 2024). It keeps the
paper's **thinker → coder** prompting and property-inference pipeline intact and
replaces the real force-sensing gripper (Dynamixel AX-12 servos + UR5 arm +
RealSense cameras + Flask server) with a **physics simulator**, so the whole
thing runs on a laptop with no hardware.

The experiment reproduced here is the paper's core claim: an LLM that infers an
object's mass / friction / stiffness and sets an adaptive, force-controlled
grasp holds a wide range of objects **without dropping or crushing them**,
beating fixed-force baselines.

## What was kept vs. replaced

| Original component | Here |
|---|---|
| `mp_prompt_tc_phys.py` thinker + coder prompts | `deligrasp/prompts.py` (copied verbatim) |
| `conversation.py` two-LLM chain | `deligrasp/conversation.py` |
| `process_code.py` code extraction | `deligrasp/process_code.py` |
| OpenAI call | `deligrasp/llm.py` — `OpenAIBackend` (real) **or** `MockBackend` (offline) |
| `ConfirmationSafeExecutor` (subprocess) | `deligrasp/executor.py` (in-process, stdout captured) |
| `magpie/gripper.py` (real servos, `deligrasp()`, `check_slip`) | `deligrasp/gripper.py` (same API, simulated physics) |
| Flask `server.py`, perception, UR5, cameras | **dropped** — out of scope |

The benchmark objects (`deligrasp/objects.py`'s `BENCHMARK`, and the matching blind
`LLM_PRIORS` in `deligrasp/prompts.py`) are the same 6 objects, described by the
same `ObjectParams` schema (`objects/object_params.py`), as the object-parameter
extraction pipeline (`extraction/param_prompts.py`'s `PRIORS` / `configs/objects/
*.json`): `glass_bottle`, `steel_bolt`, `ceramic_mug`, `rice_bag`, `raw_egg`,
`brick`. `BENCHMARK` is built directly from `param_prompts.PRIORS`, so mass,
friction, and geometry (`mass_g`, `friction`, `rest_width_mm` -- all derived
properties on `ObjectParams`) are identical between the two pipelines; only
`spring_Npm` and `crush_force_N` exist purely for this benchmark's physics. The
thinker LLM (`prompts.LLM_PRIORS`) still only sees the object's text description
and must infer these values blind, same as the original paper.

Because only the `Gripper` class was hardware-bound, the LLM planning stack is
reused essentially unchanged.

## Install & run

```bash
pip install -r requirements.txt

# from the repo root -- offline, deterministic, no API key (default):
python src/scripts/run_experiment.py --plot

# reproduce with a real LLM exactly as the paper does:
export OPENAI_API_KEY=sk-...
python src/scripts/run_experiment.py --backend openai --model gpt-4-turbo --plot

# inspect one object's thinker/coder output:
python src/scripts/run_experiment.py --object raw_egg --verbose
```

Outputs: a per-grasp table, a success-rate summary, and (written to
`src/results/` by default) `results.csv` and, with `--plot`,
`deligrasp_trajectories.png`.

## Pipeline (per object)

1. Instruction `"grasp the <object>"` →
2. **Thinker LLM** fills the structured template: estimates mass (bracketed by a
   heavier and lighter reference object), compliance, spring constant, friction
   coefficient, goal aperture, and slip-recovery increments →
3. **Coder LLM** turns that into a gripper program that computes
   `initial_force = m·g / μ` and `additional_force = k·Δx·0.1` and calls
   `G.deligrasp(...)` →
4. **Executor** runs the program against the simulator (import is rewritten to
   the sim gripper, the object is injected) →
5. **Evaluator** scores the result against ground-truth physics as
   *held / dropped / crushed*.

## Simulator physics (`gripper.py`)

Per finger, an object of rest width `w`, stiffness `k` resists compression:

```
reaction(aperture) = k · max(0, w − aperture) / 1000      # N
measured_force     = min(force_limit, reaction)           # what a load cell reads
```

`deligrasp()` reproduces the real step-and-check loop: close to the goal
aperture, and while the object still slips (measured force below the per-finger
target `fc/2`), close an extra `Δx` and raise the force by `Δf`, until the
contact force is enough to hold. Ground truth for scoring:

```
required_per_finger = m·g / (2·μ)        # below this → dropped
crush_force_N                            # above this → crushed
```

## Example result (offline `MockBackend`)

```
SUCCESS RATE BY METHOD (held without crushing):
  deligrasp    6/6  (100%)   [dropped 0, crushed 0]
  min_force    0/6  (  0%)   [dropped 6, crushed 0]   # too weak: drops everything
  fixed_5N     2/6  ( 33%)   [dropped 4, crushed 0]   # one force can't fit all
  max_force    3/6  ( 50%)   [dropped 0, crushed 3]   # too strong: crushes fragile
```

DeliGrasp adapts force per object; the fixed baselines cannot satisfy both
fragile and heavy objects with a single force.

## Notes on faithfulness

- The offline `MockBackend` uses a small table of *plausible* property estimates
  (`LLM_PRIORS`) that stands in for GPT-4. It is intentionally close to ground
  truth, so DeliGrasp scores near-perfect. With `--backend openai`, real-LLM
  estimation error makes the numbers noisier — that variance is the interesting
  research signal, and this harness is what lets you measure it.
- To study robustness to bad inference, perturb `LLM_PRIORS` (e.g. inflate a
  mass or drop a friction estimate) and watch the evaluator flag drops/crushes.
- Ground-truth object properties live in `extraction/param_prompts.py`'s
  `PRIORS` (`deligrasp/objects.py`'s `BENCHMARK` is built from it); the LLM
  never sees them. Add an object there (with `spring_Npm`/`crush_force_N`
  filled in) and a matching blind prior in `deligrasp/prompts.py`'s
  `LLM_PRIORS` to extend the benchmark.

## Files

```
../scripts/run_experiment.py   # driver: LLM grasp vs baselines, table + CSV + plot
../scripts/plot_trajectory.py  # force/aperture trajectory plots
deligrasp/
  objects.py                 # BENCHMARK: ObjectParams built from param_prompts.PRIORS
  gripper.py                 # simulated force-sensing gripper + deligrasp loop
  prompts.py                 # real thinker/coder prompts + offline LLM priors
  llm.py                     # OpenAI + Mock backends
  conversation.py            # thinker -> coder orchestration
  process_code.py            # code-block extraction
  executor.py                # run LLM code against the sim
  evaluate.py                # held / dropped / crushed scoring
```
