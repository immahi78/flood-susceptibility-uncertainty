"""Matplotlib-only figures.  Every figure is stamped SYNTHETIC DATA."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .validate import roc_points, success_rate_curve

STAMP = "SYNTHETIC DATA - not Karachi, not a real catchment"


def _finish(fig, title, path):
    fig.suptitle(f"{title}\n{STAMP}", fontsize=11, y=1.04)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _map(ax, arr, title, cmap, label, vmin=None, vmax=None):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def plot_factors(cat, names, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    units = {
        "elevation": ("Elevation (m)", "terrain"),
        "slope": ("Slope (deg)", "magma"),
        "drainage_distance": ("Distance to nullah (m)", "viridis"),
        "impervious": ("Impervious fraction", "cividis"),
        "rainfall": ("Event rainfall (mm)", "Blues"),
        "soil_permeability": ("Permeability index", "YlGn"),
    }
    for ax, n in zip(axes.flat, names[:6]):
        t, cm = units[n]
        _map(ax, cat.raw[n], t, cm, "")
    _map(axes.flat[6], cat.landuse_class, "Land-use class (0-3)", "viridis", "class", 0, 3)
    inv = axes.flat[7]
    inv.imshow(cat.true_susceptibility, cmap="Greys", alpha=0.6)
    yy, xx = np.nonzero(cat.inventory)
    inv.scatter(xx, yy, s=0.6, c="crimson", marker="s", linewidths=0)
    inv.set_title("Synthetic flood inventory (red) on hidden truth (grey)", fontsize=9)
    inv.set_xticks([])
    inv.set_yticks([])
    _finish(fig, "Synthetic factor rasters and flood inventory", path)


def plot_susceptibility(maps: dict, path, label="susceptibility (0-1)"):
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.6))
    for ax, (t, m) in zip(np.atleast_1d(axes), maps.items()):
        _map(ax, m, t, "inferno", label, 0, None)
    _finish(fig, "Susceptibility maps", path)


def plot_uncertainty(mc, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _map(axes[0], mc.mean, "MC mean susceptibility (AHP)", "inferno", "score")
    _map(axes[1], mc.std, "MC std (judgment uncertainty)", "viridis", "std of score")
    _map(axes[2], mc.p_high, "P(high class)", "magma", "probability", 0, 1)
    axes[2].contour(mc.baseline >= mc.threshold, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[2].text(0.02, 0.02, "cyan = baseline high-class boundary", transform=axes[2].transAxes,
                 color="black", fontsize=7, va="bottom",
                 bbox=dict(facecolor="white", alpha=0.8, pad=1.5, lw=0))
    _finish(fig, "Propagating pairwise-judgment uncertainty into the map", path)


def plot_roc(y, scores: dict, path, oracle_name=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    styles = ["-", "--", "-.", ":"]
    for (name, s), ls in zip(scores.items(), styles):
        fpr, tpr = roc_points(y, s)
        from .validate import auc

        a1.plot(fpr, tpr, ls, lw=1.8, label=f"{name} (AUC {auc(y, s):.3f})")
        x, h, _ = success_rate_curve(y, s)
        a2.plot(x, h, ls, lw=1.8, label=name)
    a1.plot([0, 1], [0, 1], color="grey", lw=0.8)
    a2.plot([0, 1], [0, 1], color="grey", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a2.set(xlabel="Fraction of area (most susceptible first)", ylabel="Fraction of flooded cells captured",
           title="Success-rate curve")
    a1.legend(fontsize=8, loc="lower right")
    a2.legend(fontsize=8, loc="lower right")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    _finish(fig, "AHP vs machine learning (ML = spatial block CV, out-of-fold)", path)


def plot_sensitivity(names, mc_weights, nominal_w, oat_df, removal_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.boxplot(mc_weights, vert=False, tick_labels=names, showfliers=False)
    ax.scatter(nominal_w, np.arange(1, len(names) + 1), color="crimson", zorder=3, label="nominal")
    ax.set_xlabel("AHP weight")
    ax.set_title("Weights under judgment perturbation (CR<=0.1)", fontsize=9)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax = axes[1]
    piv = oat_df.pivot(index="factor", columns="change", values="pct_cells_high_class_changed").loc[names]
    yv = np.arange(len(names))
    ax.barh(yv - 0.2, piv.iloc[:, 0], 0.4, label=piv.columns[0])
    ax.barh(yv + 0.2, piv.iloc[:, 1], 0.4, label=piv.columns[1])
    ax.set_yticks(yv, names)
    ax.invert_yaxis()
    ax.set_xlabel("% of cells whose high-class membership changes")
    ax.set_title("One-at-a-time weight change", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[2]
    r = removal_df.set_index("removed_factor").loc[names]
    ax.barh(yv, r["pct_cells_high_class_changed"], 0.5, color="tab:green")
    ax.set_yticks(yv, names)
    ax.invert_yaxis()
    ax.set_xlabel("% of cells whose high-class membership changes")
    ax.set_title("Map-removal (factor dropped)", fontsize=9)
    _finish(fig, "Sensitivity analysis", path)
