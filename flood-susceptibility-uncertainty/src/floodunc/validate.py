"""Validation of susceptibility maps against a flood inventory."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def roc_points(y, score):
    fpr, tpr, _ = roc_curve(np.asarray(y).reshape(-1), np.asarray(score).reshape(-1))
    return fpr, tpr


def auc(y, score) -> float:
    return float(roc_auc_score(np.asarray(y).reshape(-1), np.asarray(score).reshape(-1)))


def success_rate_curve(y, score):
    """Cells ranked from most to least susceptible: x = cumulative area fraction,
    y = cumulative fraction of inventory cells captured.  Returns (x, y, area_under)."""
    y = np.asarray(y).reshape(-1).astype(float)
    order = np.argsort(-np.asarray(score).reshape(-1), kind="stable")
    hits = np.cumsum(y[order]) / y.sum()
    x = np.arange(1, y.size + 1) / y.size
    x = np.r_[0.0, x]
    hits = np.r_[0.0, hits]
    return x, hits, float(np.trapezoid(hits, x))


def precision_recall_top(y, score, top_frac: float = 0.2):
    """Precision/recall when the top `top_frac` of cells by score are flagged 'high'."""
    y = np.asarray(y).reshape(-1).astype(bool)
    s = np.asarray(score).reshape(-1)
    flagged = s >= np.quantile(s, 1 - top_frac)
    tp = (flagged & y).sum()
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max(y.sum(), 1)
    return float(precision), float(recall)


def block_bootstrap_auc(y, score, blocks, n_boot: int = 200, seed: int = 0):
    """95% CI for AUC by resampling spatial blocks (respects autocorrelation)."""
    y = np.asarray(y).reshape(-1)
    s = np.asarray(score).reshape(-1)
    b = np.asarray(blocks).reshape(-1)
    groups = [np.flatnonzero(b == u) for u in np.unique(b)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def compare_models(y, scores: dict, blocks, top_frac=0.2, n_boot=200, seed=0) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        lo, hi = block_bootstrap_auc(y, s, blocks, n_boot, seed)
        p, r = precision_recall_top(y, s, top_frac)
        rows.append(
            {
                "model": name,
                "AUC": auc(y, s),
                "AUC_CI95_low": lo,
                "AUC_CI95_high": hi,
                "success_rate_AUC": success_rate_curve(y, s)[2],
                f"precision_top{int(top_frac*100)}%": p,
                f"recall_top{int(top_frac*100)}%": r,
            }
        )
    return pd.DataFrame(rows)
