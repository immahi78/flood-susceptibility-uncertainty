"""End-to-end demo on a SYNTHETIC catchment.

Writes docs/figures/*.png, outputs/results.txt, outputs/susceptibility_cells.csv.
Run:  python scripts/run_demo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from floodunc import ahp as ahp_mod
from floodunc import ml, uncertainty as unc, validate as val, viz
from floodunc.factors import FACTOR_NAMES, generate_catchment

N, SEED = 200, 42
MC_SAMPLES, SIGMA = 1000, 0.5
BLOCK, FOLDS, BUFFER = 25, 5, 5
HIGH_Q = 0.8

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def main():
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    fig_dir = ROOT / "docs" / "figures"

    cat = generate_catchment(n=N, seed=SEED)
    stack = cat.criteria_stack()
    y = cat.labels()
    blocks = ml.block_ids(cat.shape, BLOCK).reshape(-1)

    out("SYNTHETIC DATA DEMO - results describe method behaviour, NOT Karachi")
    out("=" * 72)
    out(f"grid {N}x{N} = {N*N} cells, seed {SEED}, flooded cells {y.sum()} ({100*y.mean():.2f}%)")

    # ---- AHP ----
    A = ahp_mod.default_matrix()
    res = ahp_mod.ahp(A)
    out("\n[1] AHP (expert-style judgments, NOT estimated from data)")
    out("pairwise matrix (rows/cols = " + ", ".join(FACTOR_NAMES) + "):")
    out(pd.DataFrame(A, index=FACTOR_NAMES, columns=FACTOR_NAMES).round(3).to_string())
    out(f"lambda_max = {res.lambda_max:.4f}  CI = {res.ci:.4f}  CR = {res.cr:.4f}  (n=7, RI=1.32)")
    out("weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(FACTOR_NAMES, res.weights)))
    ahp_map = ahp_mod.weighted_overlay(stack, res.weights)

    # ---- Monte Carlo ----
    mc = unc.monte_carlo_susceptibility(
        A, stack, n_samples=MC_SAMPLES, sigma=SIGMA, seed=SEED, high_quantile=HIGH_Q, labels=y
    )
    out(f"\n[2] Monte Carlo on pairwise judgments (log-normal sigma={SIGMA}, reject CR>0.1)")
    out(f"accepted {MC_SAMPLES} samples from {mc.attempts} draws (acceptance {100*mc.acceptance_rate:.1f}%)")
    out(f"accepted-sample CR: mean {mc.cr.mean():.4f}, max {mc.cr.max():.4f}")
    out("weight mean +/- sd [5%,95%]:")
    for i, n in enumerate(FACTOR_NAMES):
        w = mc.weights[:, i]
        out(f"  {n:18s} nominal {res.weights[i]:.3f} | {w.mean():.3f} +/- {w.std():.3f} "
            f"[{np.percentile(w,5):.3f}, {np.percentile(w,95):.3f}]")
    out(f"per-cell std of score: mean {mc.std.mean():.4f}, 95th pct {np.percentile(mc.std,95):.4f}, max {mc.std.max():.4f}")
    ph = mc.p_high
    out(f"high-class threshold (baseline 80th pct score) = {mc.threshold:.4f}")
    out(f"cells with p_high > 0.5: {100*(ph>0.5).mean():.2f}% of area (baseline high class = 20.00%)")
    out(f"cells with 0.1 < p_high < 0.9 ('uncertain class'): {100*((ph>0.1)&(ph<0.9)).mean():.2f}% of area")
    out(f"cells with p_high >= 0.99: {100*(ph>=0.99).mean():.2f}%; p_high <= 0.01: {100*(ph<=0.01).mean():.2f}%")
    out(f"AUC vs inventory across MC maps: nominal {val.auc(y, ahp_map):.4f}; "
        f"MC mean {mc.aucs.mean():.4f}, 5-95% [{np.percentile(mc.aucs,5):.4f}, {np.percentile(mc.aucs,95):.4f}]")

    # ---- Sensitivity ----
    oat = unc.oat_weight_sensitivity(stack, res.weights, FACTOR_NAMES, 0.2, HIGH_Q, y)
    rem = unc.map_removal_sensitivity(stack, res.weights, FACTOR_NAMES, HIGH_Q, y)
    out("\n[3a] One-at-a-time weight sensitivity (+/-20% relative, others rescaled)")
    out(oat.round(4).to_string(index=False))
    out("\n[3b] Map-removal sensitivity")
    out(rem.round(4).to_string(index=False))

    # ---- ML with spatial CV ----
    X = cat.design_matrix()
    models = ml.make_models(SEED)
    sp = list(ml.spatial_block_splits(cat.shape, BLOCK, FOLDS, BUFFER, SEED))
    rnd = list(ml.random_splits(len(y), FOLDS, SEED))
    oof = {m: ml.cross_val_predict_splits(models[m], X, y, sp) for m in ("rf", "lr")}
    rf_random = ml.cross_val_predict_splits(models["rf"], X, y, rnd)
    out(f"\n[4] ML ({FOLDS}-fold spatial block CV, {BLOCK}x{BLOCK}-cell blocks, {BUFFER}-cell training buffer)")
    out("training cells per fold: " + ", ".join(str(len(tr)) for tr, _ in sp))
    out("test cells per fold:     " + ", ".join(str(len(te)) for _, te in sp))
    out(f"RF  AUC, random 5-fold CV (leaky):  {val.auc(y, rf_random):.4f}")
    out(f"RF  AUC, spatial block CV (honest): {val.auc(y, oof['rf']):.4f}")
    imp = ml.fit_full_rf_importance(X, y, FACTOR_NAMES, SEED)
    out("RF impurity importances (fit on all cells, descriptive): "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])))

    # ---- Validation ----
    scores = {
        "AHP (nominal)": ahp_map,
        "Logistic regression (block CV)": oof["lr"].reshape(cat.shape),
        "Random forest (block CV)": oof["rf"].reshape(cat.shape),
        "Hidden truth (ceiling)": cat.true_susceptibility,
    }
    tab = val.compare_models(y, scores, blocks, 0.2, 200, SEED)
    out("\n[5] Validation against the synthetic inventory (CI = spatial block bootstrap, 200 reps)")
    out("precision/recall flag the top 20% of cells as 'high'; random-guess precision = prevalence "
        f"= {y.mean():.4f}")
    out(tab.round(4).to_string(index=False))

    # ---- Figures ----
    viz.plot_factors(cat, FACTOR_NAMES, fig_dir / "01_factors_inventory.png")
    viz.plot_susceptibility(
        {"AHP weighted overlay": ahp_map, "Random forest (block-CV OOF)": oof["rf"].reshape(cat.shape),
         "Hidden true probability": cat.true_susceptibility},
        fig_dir / "02_susceptibility.png",
    )
    viz.plot_uncertainty(mc, fig_dir / "03_uncertainty.png")
    viz.plot_roc(y, scores, fig_dir / "04_roc_success_rate.png")
    viz.plot_sensitivity(FACTOR_NAMES, mc.weights, res.weights, oat, rem, fig_dir / "05_sensitivity.png")

    # ---- CSV ----
    yy, xx = np.mgrid[0:N, 0:N]
    df = pd.DataFrame({
        "x": xx.reshape(-1), "y": yy.reshape(-1),
        "ahp": ahp_map.reshape(-1), "ahp_std": mc.std.reshape(-1),
        "p_high": mc.p_high.reshape(-1), "rf": oof["rf"],
    }).round(5)
    df.to_csv(ROOT / "outputs" / "susceptibility_cells.csv", index=False)
    out(f"\nwrote {len(df)} rows to outputs/susceptibility_cells.csv and 5 figures to docs/figures/")

    (ROOT / "outputs" / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
