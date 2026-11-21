"""Analytic Hierarchy Process (AHP) weights and weighted-overlay susceptibility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Saaty's random consistency index (n = 1..10)
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SAATY_VALUES = np.array([1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])


@dataclass
class AHPResult:
    weights: np.ndarray
    lambda_max: float
    ci: float
    cr: float


def is_reciprocal(A: np.ndarray, tol: float = 1e-8) -> bool:
    A = np.asarray(A, float)
    return bool(np.allclose(np.diag(A), 1.0) and np.allclose(A * A.T, 1.0, atol=tol))


def principal_eigen(A: np.ndarray):
    """Return (lambda_max, normalised principal eigenvector)."""
    vals, vecs = np.linalg.eig(np.asarray(A, float))
    i = int(np.argmax(vals.real))
    w = np.abs(vecs[:, i].real)
    return float(vals[i].real), w / w.sum()


def consistency(A: np.ndarray):
    """Return (lambda_max, CI, CR) with CI=(lambda_max-n)/(n-1), CR=CI/RI."""
    A = np.asarray(A, float)
    n = A.shape[0]
    lam, _ = principal_eigen(A)
    if n <= 2:
        return lam, 0.0, 0.0
    ci = max((lam - n) / (n - 1), 0.0)
    return lam, ci, ci / RANDOM_INDEX[n]


def ahp(A: np.ndarray) -> AHPResult:
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("pairwise matrix must be square")
    if not is_reciprocal(A):
        raise ValueError("pairwise matrix must be positive reciprocal with unit diagonal")
    lam, ci, cr = consistency(A)
    _, w = principal_eigen(A)
    return AHPResult(weights=w, lambda_max=lam, ci=ci, cr=cr)


def saaty_round(x):
    """Snap ratio(s) to the nearest Saaty-scale value (nearest in log space)."""
    x = np.atleast_1d(np.asarray(x, float))
    idx = np.abs(np.log(x)[..., None] - np.log(SAATY_VALUES)).argmin(axis=-1)
    return SAATY_VALUES[idx]


def matrix_from_priorities(priorities) -> np.ndarray:
    """Pairwise matrix whose upper triangle is the Saaty-rounded ratio p_i / p_j.

    This is how the demo's "expert-style" judgments are encoded: a rough priority
    vector is converted to Saaty-scale judgments, which keeps them readable and
    (nearly) consistent.  It is a convenience for a synthetic demo, not an
    elicitation procedure.
    """
    p = np.asarray(priorities, float)
    n = len(p)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = float(saaty_round(p[i] / p[j])[0])
            A[i, j] = v
            A[j, i] = 1.0 / v
    return A


# Expert-style guesses (NOT estimated from data), order = factors.FACTOR_NAMES:
# elevation, slope, drainage_distance, impervious, rainfall, soil_permeability, land_use
DEFAULT_PRIORITIES = [0.24, 0.09, 0.22, 0.14, 0.15, 0.08, 0.08]


def default_matrix() -> np.ndarray:
    return matrix_from_priorities(DEFAULT_PRIORITIES)


def weighted_overlay(stack: np.ndarray, weights) -> np.ndarray:
    """Weighted linear combination of (k,H,W) criteria -> (H,W) score in 0..1."""
    w = np.asarray(weights, float)
    if stack.shape[0] != w.size:
        raise ValueError("number of weights must equal number of criteria layers")
    return np.tensordot(w, stack, axes=1)


def high_class_threshold(score: np.ndarray, quantile: float = 0.8) -> float:
    """Score value above which a cell is in the 'high' susceptibility class."""
    return float(np.quantile(score, quantile))
