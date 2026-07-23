"""Plot per-object success-rate and return from rollout_all_objects.py and
scripts/compare_policies.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(rows, path="rollout_success_rate.png"):
    """rows: list of dicts (object, success_rate, mean_return, std_return, ...),
    already ordered the way they should appear on the x-axis."""
    names = [r["object"] for r in rows]
    success_pct = [100.0 * r["success_rate"] for r in rows]
    returns = [r["mean_return"] for r in rows]
    errs = [r["std_return"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(names) * 1.3), 6), sharex=True)

    ax1.bar(names, success_pct, color="tab:blue")
    ax1.set_ylabel("success rate (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_title("Rollout success rate by object")

    ax2.bar(names, returns, yerr=errs, color="tab:orange", capsize=4)
    ax2.set_ylabel("mean return")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("Rollout return by object (error bars = std across episodes)")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Wrote {path}")


def plot_comparison(rows, path="policy_comparison.png"):
    """Grouped bar chart: generic (no extracted params) vs. object-aware policy,
    per object. rows: list of dicts with object/generic_success_rate/
    generic_mean_return/generic_std_return/object_aware_success_rate/
    object_aware_mean_return/object_aware_std_return, already ordered for the x-axis."""
    names = [r["object"] for r in rows]
    x = range(len(names))
    width = 0.35

    generic_success = [100.0 * r["generic_success_rate"] for r in rows]
    aware_success = [100.0 * r["object_aware_success_rate"] for r in rows]
    generic_return = [r["generic_mean_return"] for r in rows]
    aware_return = [r["object_aware_mean_return"] for r in rows]
    generic_err = [r["generic_std_return"] for r in rows]
    aware_err = [r["object_aware_std_return"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(names) * 1.6), 7), sharex=True)

    ax1.bar([i - width / 2 for i in x], generic_success, width, color="tab:gray", label="generic")
    ax1.bar([i + width / 2 for i in x], aware_success, width, color="tab:blue", label="object-aware")
    ax1.set_ylabel("success rate (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_title("Success rate: generic policy vs. per-object policy")
    ax1.legend()

    ax2.bar([i - width / 2 for i in x], generic_return, width, yerr=generic_err,
            color="tab:gray", capsize=4, label="generic")
    ax2.bar([i + width / 2 for i in x], aware_return, width, yerr=aware_err,
            color="tab:blue", capsize=4, label="object-aware")
    ax2.set_ylabel("mean return")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("Return: generic policy vs. per-object policy (error bars = std)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Wrote {path}")
