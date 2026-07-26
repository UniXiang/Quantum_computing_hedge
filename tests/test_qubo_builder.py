"""Tests for src/qubo_builder.py — downside-semivariance portfolio QUBO."""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qubo_builder import (
    build_qubo, qubo_to_ising, ising_to_qubo_energy, downside_semivariance,
)
from solvers import solve_exact


def make_returns(T=60, N=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=T, freq="B").strftime("%Y-%m-%d")
    cols = [f"s{i}" for i in range(N)]
    return pd.DataFrame(rng.normal(1e-3, 0.02, size=(T, N)),
                        index=dates, columns=cols)


# ---------------------------------------------------------------------------
# 1. QUBO <-> Ising roundtrip consistency (including offset)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, seed", [(2, 0), (5, 1), (8, 2)])
def test_qubo_ising_roundtrip_random(n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, size=(n, n))
    Q = (Q + Q.T) / 2  # symmetric
    h, J, offset = qubo_to_ising(Q)
    assert h.shape == (n,) and J.shape == (n, n)
    assert np.allclose(J, J.T) and np.allclose(np.diag(J), 0.0)
    for _ in range(200):
        x = rng.integers(0, 2, size=n)
        e_qubo = float(x @ Q @ x)
        e_conv = ising_to_qubo_energy(x, h, J, offset)
        assert abs(e_qubo - e_conv) < 1e-9


def test_ising_to_qubo_energy_known_value():
    # n=2: Q = [[1, 2], [2, 3]] -> x'Qx for x=(1,0) is 1
    Q = np.array([[1.0, 2.0], [2.0, 3.0]])
    h, J, offset = qubo_to_ising(Q)
    assert ising_to_qubo_energy(np.array([1, 0]), h, J, offset) == pytest.approx(1.0)
    assert ising_to_qubo_energy(np.array([0, 1]), h, J, offset) == pytest.approx(3.0)
    assert ising_to_qubo_energy(np.array([1, 1]), h, J, offset) == pytest.approx(8.0)
    assert ising_to_qubo_energy(np.array([0, 0]), h, J, offset) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. build_qubo structure
# ---------------------------------------------------------------------------
def test_build_qubo_symmetric():
    r = make_returns()
    Q = build_qubo(r, K=3, lam=0.1, A=0.01)
    assert Q.shape == (6, 6)
    assert np.allclose(Q, Q.T, atol=1e-15)


def test_downside_semivariance_ignores_upside():
    # asset 0 alternates +r/-r with asset 1 perfectly correlated only on
    # downside; a pure-upside comovement must not enter Sigma_minus
    T = 100
    rng = np.random.default_rng(7)
    down = np.minimum(rng.normal(0, 0.02, T), 0)
    r = pd.DataFrame({
        "a": down,
        "b": 2.0 * down,               # same downside, double magnitude
        "c": np.abs(down) + 0.01,      # always positive -> zero downside
    })
    S = downside_semivariance(r, shrink=False)
    assert S.shape == (3, 3)
    assert np.allclose(S, S.T)
    assert np.all(np.diag(S) >= 0)
    assert S[2, 2] == pytest.approx(0.0, abs=1e-12)
    # cov(a,b) = 2 * var_downside(a)
    assert S[0, 1] == pytest.approx(2.0 * S[0, 0], rel=1e-6)


def test_downside_semivariance_shrinkage_is_psd():
    r = make_returns(T=60, N=6, seed=5)
    S_raw = downside_semivariance(r, shrink=False)
    S_shrunk = downside_semivariance(r, shrink=True)
    assert not np.allclose(S_raw, S_shrunk)
    # shrunk matrix is PSD with strictly positive diagonal
    assert np.all(np.linalg.eigvalsh(S_shrunk) >= -1e-15)
    assert np.all(np.diag(S_shrunk) > 0)


def test_cardinality_penalty_coefficients():
    """Pin Q entries against the documented objective with zero-data trick:
    constant tiny returns -> Sigma_minus ~ 0, mu known -> Q is dominated by
    the A*(sum x - K)^2 term plus the -lam*mu/K linear term."""
    T, N, K = 60, 5, 2
    r = pd.DataFrame(np.full((T, N), 1e-3),  # constant positive: no downside
                     columns=[f"s{i}" for i in range(N)])
    lam, A = 0.1, 0.05
    Q = build_qubo(r, K=K, lam=lam, A=A)
    # downside matrix is exactly 0 (no negative returns), so:
    #   Q_ii = -lam*mu_i/K + A*(1 - 2K),  Q_ij (i!=j) = A
    mu = 1e-3
    for i in range(N):
        assert Q[i, i] == pytest.approx(-lam * mu / K + A * (1 - 2 * K),
                                        rel=1e-6)
        for j in range(N):
            if i != j:
                assert Q[i, j] == pytest.approx(A, rel=1e-6)


def test_cardinality_energy_gap():
    """Energy of a K-subset vs a (K+1)-subset differs by the penalty A,
    up to the risk/return terms; with A huge, exact solver must pick
    exactly K assets."""
    r = make_returns(N=8, seed=3)
    K = 4
    Q = build_qubo(r, K=K, lam=0.05, A=1.0)
    x_best, _ = solve_exact(Q)
    assert x_best.sum() == K


def test_gamma_turnover_penalty():
    """With gamma and x_prev, flipping a held asset out costs gamma more
    than keeping it; verify via direct Q energy comparison."""
    T, N, K = 60, 4, 2
    r = pd.DataFrame(np.full((T, N), 1e-3), columns=[f"s{i}" for i in range(N)])
    x_prev = np.array([1, 1, 0, 0])
    gamma = 0.5
    Q = build_qubo(r, K=K, lam=0.0, A=0.1, gamma=gamma, x_prev=x_prev)
    e_keep = np.array([1, 1, 0, 0]) @ Q @ np.array([1, 1, 0, 0])
    e_swap = np.array([1, 0, 1, 0]) @ Q @ np.array([1, 0, 1, 0])
    # swapping out held asset 1 for new asset 2 incurs 2*gamma (one exit +
    # one entry) relative to staying
    assert e_swap - e_keep == pytest.approx(2 * gamma, rel=1e-9)


def test_build_qubo_input_validation():
    r = make_returns()
    with pytest.raises(ValueError):
        build_qubo(r, K=0, lam=0.1, A=0.01)
    with pytest.raises(ValueError):
        build_qubo(r, K=7, lam=0.1, A=0.01)  # K > N
    with pytest.raises(ValueError):
        build_qubo(r, K=3, lam=0.1, A=0.01,
                   gamma=0.1, x_prev=np.ones(3))  # wrong x_prev shape
