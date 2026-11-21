"""Uncertainty and sensitivity analysis for AHP weighted-overlay susceptibility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .ahp import consistency, high_class_threshold, principal_eigen, weighted_overlay

SAATY_MIN, SAATY_MAX = 1 / 9, 9.0


def perturb_matrix(A: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Multiply each upper-triangle judgment by exp(sigma * N(0,1)), clip to the
    Saaty range [1/9, 9] and restore reciprocity."""
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.clip(A[iu] * np.exp(sigma * rng.standard_normal(iu[0].size)), SAATY_MIN, SAATY_MAX)
    B = np.ones((n, n))
    B[iu] = vals
    B[(iu[1], iu[0])] = 1.0 / vals
    return B


def sample_consistent_weights(
    A: np.ndarray,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
    cr_max: float = 0.1,
    max_attempt_factor: int = 200,
):
    """Draw perturbed matrices, reject those with CR > cr_max.

    Returns (weights (m,k), cr (m,), attempts).  Raises if acceptance is hopeless.
    """
    W, CR = [], []
    attempts = 0
    limit = n_samples * max_attempt_factor
    while len(W) < n_samples:
        if attempts >= limit:
            raise RuntimeError(
                f"only {len(W)}/{n_samples} consistent samples after {attempts} attempts; "
                "reduce sigma or raise cr_max"
            )
        attempts += 1
        B = perturb_matrix(A, rng, sigma)
        _, _, cr = consistency(B)
        if cr <= cr_max:
            _, w = principal_eigen(B)
            W.append(w)
            CR.append(cr)
    return np.array(W), np.array(CR), attempts


@dataclass
class MCResult:
    mean: np.ndarray
    std: np.ndarray
    p_high: np.ndarray
    weights: np.ndarray  # (m, k) accepted weight vectors
    cr: np.ndarray
    acceptance_rate: float
    attempts: int
    threshold: float  # fixed 'high' cut-off taken from the baseline map
    baseline: np.ndarray
    aucs: np.ndarray | None  # per-sample AUC vs inventory (if labels supplied)


def monte_carlo_susceptibility(
    A: np.ndarray,
    stack: np.ndarray,
    n_samples: int = 1000,
    sigma: float = 0.25,
    seed: int = 0,
    high_quantile: float = 0.8,
    labels: np.ndarray | None = None,
    chunk: int = 100,
) -> MCResult:
    """Monte Carlo propagation of pairwise-judgment uncertainty into the map.

    Each accepted draw gives a weight vector and a full susceptibility map.  We
    accumulate per-cell mean, standard deviation and the probability that the
    cell exceeds a FIXED high-class threshold (the baseline map's
    `high_quantile` quantile).  Rejecting CR > 0.1 draws is part of the method
    and slightly narrows the spread compared with unconstrained perturbation.
    """
    rng = np.random.default_rng(seed)
    k, H, W_ = stack.shape
    _, w0 = principal_eigen(A)
    baseline = weighted_overlay(stack, w0)
    thr = high_class_threshold(baseline, high_quantile)

    weights, crs, attempts = sample_consistent_weights(A, n_samples, sigma, rng)
    flat = stack.reshape(k, -1)
    n_cells = flat.shape[1]
    s1 = np.zeros(n_cells)
    s2 = np.zeros(n_cells)
    hi = np.zeros(n_cells)
    aucs = [] if labels is not None else None
    y = None if labels is None else np.asarray(labels).reshape(-1)
    for a in range(0, n_samples, chunk):
        maps = weights[a : a + chunk] @ flat  # (c, n_cells)
        s1 += maps.sum(0)
        s2 += (maps**2).sum(0)
        hi += (maps >= thr).sum(0)
        if aucs is not None:
            aucs.extend(roc_auc_score(y, m) for m in maps)
    mean = s1 / n_samples
    var = np.maximum(s2 / n_samples - mean**2, 0.0) * n_samples / (n_samples - 1)
    return MCResult(
        mean=mean.reshape(H, W_),
        std=np.sqrt(var).reshape(H, W_),
        p_high=(hi / n_samples).reshape(H, W_),
        weights=weights,
        cr=crs,
        acceptance_rate=n_samples / attempts,
        attempts=attempts,
        threshold=thr,
        baseline=baseline,
        aucs=None if aucs is None else np.array(aucs),
    )


def _compare_maps(base, new, high_quantile, labels):
    b, n = base.reshape(-1), new.reshape(-1)
    bh = b >= np.quantile(b, high_quantile)
    nh = n >= np.quantile(n, high_quantile)
    row = {
        "mean_abs_change": float(np.mean(np.abs(n - b))),
        "spearman_rho": float(spearmanr(b, n)[0]),
        "pct_cells_high_class_changed": float(100 * np.mean(bh != nh)),
        "high_class_jaccard": float((bh & nh).sum() / max((bh | nh).sum(), 1)),
    }
    if labels is not None:
        y = np.asarray(labels).reshape(-1)
        row["delta_auc"] = float(roc_auc_score(y, n) - roc_auc_score(y, b))
    return row


def oat_weight_sensitivity(stack, weights, names, delta=0.2, high_quantile=0.8, labels=None) -> pd.DataFrame:
    """One-at-a-time: change one weight by +/- delta (relative) and rescale the
    remaining weights proportionally so that the sum stays 1."""
    weights = np.asarray(weights, float)
    base = weighted_overlay(stack, weights)
    rows = []
    for i, name in enumerate(names):
        for sign in (-1, 1):
            w = weights.copy()
            w[i] = weights[i] * (1 + sign * delta)
            others = np.arange(len(w)) != i
            w[others] = weights[others] * (1 - w[i]) / weights[others].sum()
            r = {"factor": name, "change": f"{sign * int(delta * 100):+d}%"}
            r.update(_compare_maps(base, weighted_overlay(stack, w), high_quantile, labels))
            rows.append(r)
    return pd.DataFrame(rows)


def map_removal_sensitivity(stack, weights, names, high_quantile=0.8, labels=None) -> pd.DataFrame:
    """Map-removal: drop one criterion layer, renormalise the other weights."""
    weights = np.asarray(weights, float)
    base = weighted_overlay(stack, weights)
    rows = []
    for i, name in enumerate(names):
        w = weights.copy()
        w[i] = 0.0
        w /= w.sum()
        r = {"removed_factor": name}
        r.update(_compare_maps(base, weighted_overlay(stack, w), high_quantile, labels))
        rows.append(r)
    return pd.DataFrame(rows)
