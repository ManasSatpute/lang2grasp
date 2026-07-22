"""Plot the adaptive-grasp force/aperture trajectories produced by deligrasp()."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_all(trajectories, dg_outcomes, path="deligrasp_trajectories.png"):
    outcome_by_name = {o.object_name: o for o in dg_outcomes}
    names = [n for n in trajectories if trajectories[n]]
    n = len(names)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), squeeze=False)

    for i, name in enumerate(names):
        log = trajectories[name]
        ax = axes[i // cols][i % cols]
        steps = [e["step"] for e in log]
        contact = [e["contact_force"] for e in log]
        applied = [e["applied_force"] for e in log]
        ax.plot(steps, applied, "--", color="tab:orange", label="applied force limit")
        ax.plot(steps, contact, "-o", color="tab:blue", ms=3, label="contact force")
        o = outcome_by_name.get(name)
        if o:
            ax.axhline(o.required_force_N, color="tab:green", lw=1, ls=":",
                       label="force to hold")
            title = f"{name}  ->  {o.outcome.upper()} ({'OK' if o.success else 'FAIL'})"
        else:
            title = name
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("control step")
        ax.set_ylabel("force (N)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("DeliGrasp adaptive grasp: contact force converges to holding force",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    print(f"Wrote {path}")
