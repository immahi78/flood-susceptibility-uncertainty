"""Synthetic urban catchment generator.

*** ALL DATA PRODUCED HERE IS SYNTHETIC. It is NOT Karachi data and does not
represent any real place. ***

The generator builds a square grid of cells and seven factor rasters
(elevation, slope, distance to drainage, impervious fraction, event rainfall,
soil permeability, land use).  A *hidden, nonlinear* "true" flood
susceptibility is computed from these factors (plus an unobserved spatially
correlated driver) and a binary flood inventory is drawn from it.  Because the
truth is known by construction, any susceptibility model can be validated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
)

FACTOR_NAMES = [
    "elevation",
    "slope",
    "drainage_distance",
    "impervious",
    "rainfall",
    "soil_permeability",
    "land_use",
]

LANDUSE_LABELS = [
    "open/vegetated",
    "low-density residential",
    "dense/informal residential",
    "commercial/industrial",
]
# Runoff-propensity score per land-use class (synthetic, non-monotonic on purpose).
LANDUSE_RUNOFF_SCORE = np.array([0.1, 0.4, 0.8, 0.6])


@dataclass
class SyntheticCatchment:
    """Container for one synthetic catchment (all arrays are H x W)."""

    shape: tuple
    cell_size_m: float
    seed: int
    raw: dict  # factor name -> raw-unit raster
    landuse_class: np.ndarray  # int classes 0..3
    criteria: dict  # factor name -> 0..1 raster, higher = more flood-prone
    true_susceptibility: np.ndarray  # hidden probability used to draw the inventory
    inventory: np.ndarray  # bool flood inventory

    def criteria_stack(self) -> np.ndarray:
        """(k, H, W) array of benefit-oriented criteria in FACTOR_NAMES order."""
        return np.stack([self.criteria[n] for n in FACTOR_NAMES])

    def design_matrix(self) -> np.ndarray:
        """(H*W, k) feature matrix for machine learning (row-major cell order)."""
        k = len(FACTOR_NAMES)
        return self.criteria_stack().reshape(k, -1).T

    def labels(self) -> np.ndarray:
        return self.inventory.reshape(-1).astype(int)


def _smooth_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    f = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="reflect")
    return (f - f.mean()) / f.std()


def _scale01(a: np.ndarray, invert: bool = False) -> np.ndarray:
    """Robust min-max scaling to 0..1 (1st-99th percentile), optionally inverted."""
    lo, hi = np.percentile(a, [1, 99])
    out = np.clip((a - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return 1.0 - out if invert else out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _drainage_mask(rng: np.random.Generator, n: int, n_channels: int = 3) -> np.ndarray:
    """Meandering north-to-south channels (synthetic nullahs)."""
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)
    for x0 in np.sort(rng.uniform(0.15, 0.85, n_channels)) * n:
        amp = rng.uniform(0.02, 0.06) * n
        lam = rng.uniform(0.4, 0.9) * n
        phase = rng.uniform(0, 2 * np.pi)
        wander = gaussian_filter1d(np.cumsum(rng.standard_normal(n)), 10)
        wander = (wander - wander.mean()) / (np.abs(wander - wander.mean()).max() + 1e-9)
        cols = x0 + amp * np.sin(2 * np.pi * rows / lam + phase) + 0.04 * n * wander
        cols = np.clip(np.round(cols).astype(int), 0, n - 1)
        mask[rows, cols] = True
    return binary_dilation(mask, structure=np.ones((3, 3), bool))


def generate_catchment(
    n: int = 200,
    seed: int = 42,
    cell_size_m: float = 30.0,
    target_prevalence: float = 0.10,
) -> SyntheticCatchment:
    """Generate a SYNTHETIC n x n catchment with factors and a flood inventory."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)

    # Drainage network and distance to it (m)
    dmask = _drainage_mask(rng, n)
    d_cells = distance_transform_edt(~dmask)
    drain_dist = d_cells * cell_size_m

    # Elevation (m): smooth ramp towards a "coast" at the bottom edge,
    # smooth undulation, and channels carved into the surface.
    elev = 2.0 + 25.0 * (1.0 - yy) ** 1.3 + 3.0 * _smooth_noise(rng, n, n / 12)
    elev = elev - 2.5 * np.exp(-d_cells / 6.0)
    elev = np.maximum(elev, 0.2)

    # Slope (degrees)
    gy, gx = np.gradient(elev, cell_size_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    # Impervious fraction: urban core + noise
    r2 = (xx - 0.55) ** 2 + (yy - 0.50) ** 2
    imp = 0.85 * np.exp(-r2 / (2 * 0.30**2)) + 0.12 * _smooth_noise(rng, n, n / 25) + 0.08
    imp = np.clip(imp, 0.02, 0.98)

    # Land use classes from (noisy) imperviousness
    lu_signal = imp + 0.08 * _smooth_noise(rng, n, n / 40)
    landuse = np.digitize(lu_signal, [0.2, 0.45, 0.7]).astype(int)
    lu_score = LANDUSE_RUNOFF_SCORE[landuse]

    # Event rainfall (mm): west-to-east gradient + smooth noise
    rain = 150.0 + 60.0 * xx + 15.0 * _smooth_noise(rng, n, n / 10)

    # Soil permeability index 0..1 (1 = very permeable)
    perm = _sigmoid(1.2 * _smooth_noise(rng, n, n / 14))

    raw = {
        "elevation": elev,
        "slope": slope,
        "drainage_distance": drain_dist,
        "impervious": imp,
        "rainfall": rain,
        "soil_permeability": perm,
        "land_use": lu_score,
    }

    # Benefit-oriented 0..1 criteria (higher = more flood-prone)
    criteria = {
        "elevation": _scale01(elev, invert=True),
        "slope": _scale01(slope, invert=True),
        "drainage_distance": _scale01(drain_dist, invert=True),
        "impervious": _scale01(imp),
        "rainfall": _scale01(rain),
        "soil_permeability": _scale01(perm, invert=True),
        "land_use": _scale01(lu_score),
    }

    # ---- Hidden nonlinear "truth" (not available to any model) ----
    low_elev = np.exp(-elev / 6.0)  # exponential decay with height
    flat = _sigmoid((np.median(slope) - slope) / (0.5 * slope.std() + 1e-9))
    near = np.exp(-drain_dist / 300.0)  # nullah proximity effect
    rain_imp = criteria["rainfall"] * criteria["impervious"]  # interaction
    low_perm = _sigmoid((0.4 - perm) * 8.0)  # threshold behaviour
    unobserved = _smooth_noise(rng, n, n / 8)  # e.g. blocked culverts; NOT a factor
    z = (
        2.2 * low_elev
        + 1.5 * near
        + 1.2 * flat
        + 2.0 * rain_imp
        + 0.8 * low_perm
        + 0.5 * lu_score
        + 0.9 * unobserved
    )
    z = (z - z.mean()) / z.std()
    k = 2.5
    lo, hi = -10.0, 10.0
    for _ in range(60):  # bisection on the intercept for the target prevalence
        c = 0.5 * (lo + hi)
        if _sigmoid(k * (z - c)).mean() > target_prevalence:
            lo = c
        else:
            hi = c
    true_susc = _sigmoid(k * (z - c))
    inventory = rng.random((n, n)) < true_susc

    return SyntheticCatchment(
        shape=(n, n),
        cell_size_m=cell_size_m,
        seed=seed,
        raw=raw,
        landuse_class=landuse,
        criteria=criteria,
        true_susceptibility=true_susc,
        inventory=inventory,
    )
