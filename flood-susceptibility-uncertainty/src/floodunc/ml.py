"""Machine-learning susceptibility with SPATIAL block cross-validation.

Random k-fold CV on spatially autocorrelated rasters leaks information between
neighbouring cells and inflates skill.  Here whole blocks of the grid are held
out, and (optionally) a buffer of training cells around each held-out region is
dropped as well.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def block_ids(shape, block_size: int) -> np.ndarray:
    """(H,W) integer id of the square block each cell belongs to."""
    H, W = shape
    r, c = np.mgrid[0:H, 0:W]
    n_bc = -(-W // block_size)
    return (r // block_size) * n_bc + (c // block_size)


def assign_block_folds(blocks: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Assign whole blocks to folds at random (round-robin after shuffling)."""
    ub = np.unique(blocks)
    if len(ub) < n_folds:
        raise ValueError("fewer blocks than folds")
    perm = np.random.default_rng(seed).permutation(ub)
    fold_of = {b: i % n_folds for i, b in enumerate(perm)}
    return np.vectorize(fold_of.get)(blocks)


def spatial_block_splits(shape, block_size=25, n_folds=5, buffer_cells=5, seed=0):
    """Yield (train_idx, test_idx) as flat indices.

    Test = all cells of the held-out blocks.  Train = cells of other blocks that
    are more than `buffer_cells` away from any test cell.
    """
    blocks = block_ids(shape, block_size)
    folds = assign_block_folds(blocks, n_folds, seed)
    for k in range(n_folds):
        test_mask = folds == k
        train_mask = ~test_mask
        if buffer_cells > 0:
            train_mask &= distance_transform_edt(~test_mask) > buffer_cells
        yield np.flatnonzero(train_mask.reshape(-1)), np.flatnonzero(test_mask.reshape(-1))


def random_splits(n_cells, n_folds=5, seed=0):
    """Ordinary shuffled k-fold (for demonstrating leakage inflation only)."""
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(n_cells)):
        yield tr, te


def make_models(seed: int = 0) -> dict:
    return {
        "rf": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=20, n_jobs=-1, random_state=seed
        ),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def cross_val_predict_splits(model, X, y, splits) -> np.ndarray:
    """Out-of-fold P(flood) for every cell; each cell is predicted by a model that
    never saw it (or, for block CV, its neighbourhood)."""
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("some cells never appeared in a test fold")
    return oof


def fit_full_rf_importance(X, y, names, seed=0) -> dict:
    """Impurity-based importances of an RF fitted on all cells (descriptive only)."""
    rf = make_models(seed)["rf"].fit(X, y)
    return dict(zip(names, rf.feature_importances_))
