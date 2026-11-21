import numpy as np
import pytest

from floodunc import ahp


def power_iteration(A, iters=500):
    w = np.ones(A.shape[0]) / A.shape[0]
    for _ in range(iters):
        w = A @ w
        w /= w.sum()
    lam = float(np.mean((A @ w) / w))
    return lam, w


KNOWN = np.array([[1, 2, 4], [1 / 2, 1, 3], [1 / 4, 1 / 3, 1]])


def test_weights_sum_to_one_and_positive():
    res = ahp.ahp(ahp.default_matrix())
    assert res.weights.sum() == pytest.approx(1.0)
    assert (res.weights > 0).all()


def test_consistent_matrix_has_zero_cr_and_exact_weights():
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1 / w)  # perfectly consistent
    res = ahp.ahp(A)
    assert res.lambda_max == pytest.approx(3.0)
    assert res.cr == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(res.weights, w)


def test_cr_known_matrix_against_independent_power_iteration():
    res = ahp.ahp(KNOWN)
    lam, w = power_iteration(KNOWN)  # independent algorithm
    assert res.lambda_max == pytest.approx(lam, rel=1e-6)
    assert np.allclose(res.weights, w, atol=1e-6)
    ci = (lam - 3) / 2
    assert res.ci == pytest.approx(ci, rel=1e-6)
    assert res.cr == pytest.approx(ci / 0.58, rel=1e-6)
    assert 0 < res.cr < 0.1  # mildly inconsistent but acceptable


def test_inconsistent_cyclic_matrix_fails_threshold():
    A = np.array([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])
    assert ahp.ahp(A).cr > 0.1


def test_non_reciprocal_rejected():
    with pytest.raises(ValueError):
        ahp.ahp(np.array([[1, 2], [2, 1]]))


def test_default_matrix_acceptable_and_saaty_scale():
    A = ahp.default_matrix()
    assert ahp.ahp(A).cr < 0.1
    assert np.isin(np.round(A, 9), np.round(ahp.SAATY_VALUES, 9)).all()


def test_weighted_overlay_is_convex_combination():
    stack = np.random.default_rng(0).random((3, 5, 5))
    s = ahp.weighted_overlay(stack, [0.5, 0.3, 0.2])
    assert s.min() >= stack.min() and s.max() <= stack.max()
    assert np.allclose(s, 0.5 * stack[0] + 0.3 * stack[1] + 0.2 * stack[2])
