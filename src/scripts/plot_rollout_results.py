"""Plot per-object success-rate and return from rollout_all_objects.py."""

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
