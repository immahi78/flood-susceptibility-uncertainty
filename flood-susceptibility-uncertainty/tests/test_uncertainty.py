import numpy as np

from floodunc import ahp, uncertainty as unc
from floodunc.factors import generate_catchment


def small():
    cat = generate_catchment(n=50, seed=1)
    return cat.criteria_stack(), cat.labels()


def test_mc_std_positive_and_ranges_valid():
    stack, y = small()
    mc = unc.monte_carlo_susceptibility(ahp.default_matrix(), stack, n_samples=120, sigma=0.25, seed=3, labels=y)
    assert mc.std.min() >= 0 and mc.std.mean() > 0
    assert (mc.p_high >= 0).all() and (mc.p_high <= 1).all()
    assert mc.mean.shape == stack.shape[1:]
    assert (mc.cr <= 0.1).all()
    assert 0 < mc.acceptance_rate <= 1
    assert mc.aucs.shape == (120,)


def test_mc_is_reproducible_and_matches_direct_computation():
    stack, _ = small()
    A = ahp.default_matrix()
    a = unc.monte_carlo_susceptibility(A, stack, n_samples=60, seed=5)
    b = unc.monte_carlo_susceptibility(A, stack, n_samples=60, seed=5)
    assert np.array_equal(a.std, b.std)
    maps = np.array([ahp.weighted_overlay(stack, w) for w in a.weights])
    assert np.allclose(maps.mean(0), a.mean)
    assert np.allclose(maps.std(0, ddof=1), a.std, atol=1e-9)
    assert np.allclose((maps >= a.threshold).mean(0), a.p_high)


def test_zero_sigma_gives_no_uncertainty():
    stack, _ = small()
    mc = unc.monte_carlo_susceptibility(ahp.default_matrix(), stack, n_samples=20, sigma=0.0, seed=0)
    assert mc.std.max() < 1e-7


def test_perturbed_matrix_stays_reciprocal_and_in_range():
    rng = np.random.default_rng(0)
    B = unc.perturb_matrix(ahp.default_matrix(), rng, 0.8)
    assert ahp.is_reciprocal(B)
    assert B.min() >= 1 / 9 - 1e-12 and B.max() <= 9 + 1e-12


def test_sensitivity_tables():
    stack, y = small()
    w = ahp.ahp(ahp.default_matrix()).weights
    names = [f"f{i}" for i in range(len(w))]
    oat = unc.oat_weight_sensitivity(stack, w, names, labels=y)
    rem = unc.map_removal_sensitivity(stack, w, names, labels=y)
    assert len(oat) == 2 * len(w) and len(rem) == len(w)
    assert (oat["mean_abs_change"] > 0).all()
    assert rem["spearman_rho"].between(-1, 1).all()
