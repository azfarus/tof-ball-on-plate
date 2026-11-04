"""Plots for one scenario and for the summary."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scenario_plot(path, title, trace, readings, corners, cfg):
    s = cfg["plate"]["size_m"] / 2 * 1000
    ok = trace[:, 5] > 0
    err = np.linalg.norm(trace[:, 1:3] - trace[:, 3:5], axis=1) * 1000
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))

    a.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], color="0.5", lw=1)
    a.plot(trace[:, 1] * 1000, trace[:, 2] * 1000, color="black", lw=1.5, label="true")
    est = np.where(ok[:, None], trace[:, 3:5] * 1000, np.nan)
    a.plot(est[:, 0], est[:, 1], color="tab:orange", lw=1, label="estimate")
    for c in corners:
        a.plot(c[0] * 1000, c[1] * 1000, "s", color="tab:blue", ms=7)
    acc = [r for r in readings if r["status"] == "accepted"]
    if acc:
        pts = np.array([[corners[r["sensor"]][0] + (r["range"] + 0.02) * np.cos(r["yaw"]),
                         corners[r["sensor"]][1] + (r["range"] + 0.02) * np.sin(r["yaw"])]
                        for r in acc]) * 1000
        a.plot(pts[:, 0], pts[:, 1], ".", color="tab:green", ms=2, alpha=0.4,
               label="accepted reading (beam axis)")
    a.set_aspect("equal")
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_title(title)
    a.legend(loc="upper right", fontsize=8)

    b.plot(trace[:, 0], np.where(ok, err, np.nan), color="tab:orange", lw=1)
    b.set_xlabel("time (s)")
    b.set_ylabel("position error (mm)")
    b.set_title("estimate error vs truth")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_plot(path, rows):
    names = sorted({r["run"].rsplit("_", 1)[0] for r in rows})
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (mode, color) in enumerate([("clean", "tab:blue"), ("interf", "tab:red")]):
        vals = []
        for n in names:
            hit = [r for r in rows if r["run"] == f"{n}_{mode}"]
            vals.append(hit[0]["rmse_mm"] if hit else np.nan)
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color,
               label="no interference" if mode == "clean" else "with interference")
    ax.set_xticks(x, names)
    ax.set_ylabel("position RMSE (mm)")
    ax.set_title("4-corner oscillating VL53L4CD, 18 deg steps")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ablation_plot(path, summary, variants, scenarios, key, title):
    """Heatmap of one metric: variants (rows) x scenarios (columns)."""
    grid = np.array([[next(r[key] for r in summary
                           if r["variant"] == v and r["scenario"] == s)
                      for s in scenarios] for v in variants])
    fig, ax = plt.subplots(figsize=(1.1 * len(scenarios) + 3, 0.5 * len(variants) + 2))
    shown = np.log10(np.clip(grid, 1, None))  # log colour: errors span 1-1000 mm
    im = ax.imshow(shown, cmap="viridis_r", aspect="auto")
    for i in range(len(variants)):
        for j in range(len(scenarios)):
            v = grid[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if shown[i, j] > np.nanmean(shown) else "black")
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=35, ha="right")
    ax.set_yticks(range(len(variants)), variants)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10(mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
