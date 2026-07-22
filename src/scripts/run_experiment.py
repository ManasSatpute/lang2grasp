"""
DeliGrasp reproduction — simulation only, LLM-driven.

For every benchmark object:
  1. Build a natural-language instruction ("grasp the <object>").
  2. Run the DeliGrasp thinker->coder LLM pipeline to infer the object's
     physical properties and emit a grasp program.
  3. Execute that program against the physics simulator.
  4. Compare against three non-adaptive baselines (min / fixed / max force).
  5. Score every grasp as held / dropped / crushed against ground truth.

Backend selection:
  --backend mock    (default) deterministic, offline, no API key needed
  --backend openai  uses GPT via OPENAI_API_KEY (reproduces the paper exactly)

Usage (from the repo root):
  python src/scripts/run_experiment.py
  python src/scripts/run_experiment.py --backend openai --model gpt-4-turbo
  python src/scripts/run_experiment.py --object egg --verbose
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# deligrasp lives under src/extraction/, a sibling of this file's src/scripts/ dir.
# src/ itself is needed too: deligrasp/objects.py builds its BENCHMARK from
# objects.object_params.ObjectParams, the same schema `objects/` uses for training.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extraction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deligrasp import objects, evaluate, executor
from deligrasp.conversation import Conversation
from deligrasp.gripper import Gripper
from deligrasp import gripper as sim_gripper
from deligrasp.llm import MockBackend, OpenAIBackend

# per-finger force for the naive baselines
BASELINES = {"min_force": 0.15, "fixed_5N": 2.5, "max_force": 16.0}


def make_backend(name, model):
    if name == "openai":
        return OpenAIBackend(model=model)
    return MockBackend()


def run_deligrasp(conv, obj, verbose):
    _, code = conv.plan(f"grasp the {obj.name.replace('_', ' ')}")
    res = executor.run(code, obj)
    return evaluate.evaluate(obj, "deligrasp",
                             res["final_aperture_mm"],
                             res["applied_force_N"],
                             res["peak_contact_force_N"]), res


def run_baseline(obj, method, force):
    sim_gripper.ACTIVE_OBJECT = obj
    g = Gripper()
    ap, applied, _ = g.naive_grasp(obj.rest_width_mm, force)
    return evaluate.evaluate(obj, method, ap, applied, g.peak_contact_force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "openai"])
    ap.add_argument("--model", default="gpt-4-turbo")
    ap.add_argument("--object", default=None, help="run a single object")
    ap.add_argument("--verbose", action="store_true", help="print LLM I/O")
    ap.add_argument("--csv", default="src/results/results.csv")
    ap.add_argument("--plot", action="store_true", help="save trajectory plots")
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    conv = Conversation(backend, verbose=args.verbose)

    names = [args.object] if args.object else list(objects.BENCHMARK)
    all_outcomes = []
    trajectories = {}

    for name in names:
        obj = objects.get(name)
        dg_outcome, res = run_deligrasp(conv, obj, args.verbose)
        all_outcomes.append(dg_outcome)
        trajectories[name] = res["grasp_log"]
        for method, force in BASELINES.items():
            all_outcomes.append(run_baseline(obj, method, force))

    # ---- per-object table -------------------------------------------------
    print("\n" + "=" * 92)
    print("PER-GRASP OUTCOMES  (req = force needed to hold; crush = damage threshold)")
    print("=" * 92)
    order = ["deligrasp"] + list(BASELINES)
    for name in names:
        for method in order:
            o = next(x for x in all_outcomes if x.object_name == name and x.method == method)
            print(o.row())
        print("-" * 92)

    # ---- success-rate summary --------------------------------------------
    print("\nSUCCESS RATE BY METHOD (held without crushing):")
    for method in order:
        outs = [o for o in all_outcomes if o.method == method]
        n_ok = sum(o.success for o in outs)
        crushed = sum(o.outcome == "crushed" for o in outs)
        dropped = sum(o.outcome == "dropped" for o in outs)
        print(f"  {method:<12} {n_ok}/{len(outs)}  "
              f"({100*n_ok/len(outs):3.0f}%)   "
              f"[dropped {dropped}, crushed {crushed}]")

    # ---- write CSV --------------------------------------------------------
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "method", "outcome", "success",
                    "final_force_N", "peak_force_N", "required_force_N", "crush_force_N"])
        for o in all_outcomes:
            w.writerow([o.object_name, o.method, o.outcome, o.success,
                        o.final_force_N, o.peak_force_N, o.required_force_N, o.crush_force_N])
    print(f"\nWrote {csv_path}")

    if args.plot:
        from plot_trajectory import plot_all
        plot_path = csv_path.parent / "deligrasp_trajectories.png"
        plot_all(trajectories, [o for o in all_outcomes if o.method == "deligrasp"], path=str(plot_path))


if __name__ == "__main__":
    main()
