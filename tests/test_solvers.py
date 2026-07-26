"""Tests for src/solvers.py — exact enumeration + simulated annealing."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solvers import solve_exact, solve_sa, _flip_delta


def random_q(n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, size=(n, n))
    return (Q + Q.T) / 2


def brute_energy(Q, x):
    return float(x @ Q @ x)


# ---------------------------------------------------------------------------
# solve_exact
# ---------------------------------------------------------------------------
def test_solve_exact_handwritten():
    # n=3 diagonal Q: optimum picks the negative entries
    Q = np.diag([-1.0, 2.0, -0.5])
    x_best, e_best = solve_exact(Q)
    assert e_best == pytest.approx(-1.5)
    assert list(x_best) == [1, 0, 1]


def test_solve_exact_handwritten_coupling():
    # n=2: x'Qx = x0 + x1 - 4*x0*x1 -> optimum (1,1) with energy -2
    Q = np.array([[1.0, -2.0], [-2.0, 1.0]])
    x_best, e_best = solve_exact(Q)
    assert e_best == pytest.approx(-2.0)
    assert list(x_best) == [1, 1]


@pytest.mark.parametrize("n, seed", [(4, 0), (8, 1), (12, 2)])
def test_solve_exact_matches_python_brute_force(n, seed):
    Q = random_q(n, seed)
    x_best, e_best = solve_exact(Q)
    # independent brute force
    best = np.inf
    for m in range(2**n):
        x = np.array([(m >> i) & 1 for i in range(n)], dtype=float)
        best = min(best, brute_energy(Q, x))
    assert e_best == pytest.approx(best, abs=1e-9)
    assert brute_energy(Q, x_best) == pytest.approx(e_best, abs=1e-9)


# ---------------------------------------------------------------------------
# solve_sa
# ---------------------------------------------------------------------------
def test_solve_sa_reproducible():
    Q = random_q(10, seed=5)
    x1, e1 = solve_sa(Q, budget_s=0.5, seed=11)
    x2, e2 = solve_sa(Q, budget_s=0.5, seed=11)
    np.testing.assert_array_equal(x1, x2)
    assert e1 == e2


def test_solve_sa_n16_within_5pct_of_exact():
    Q = random_q(16, seed=6)
    _, e_exact = solve_exact(Q)
    _, e_sa = solve_sa(Q, budget_s=2.0, seed=42)
    gap = e_sa - e_exact
    assert gap >= -1e-9, "SA beat exhaustive enumeration (impossible)"
    assert gap <= 0.05 * abs(e_exact), (
        f"SA gap {gap:.6f} > 5% of |exact| {abs(e_exact):.6f}"
    )


def test_solve_sa_respects_budget():
    import time
    Q = random_q(14, seed=7)
    t0 = time.perf_counter()
    solve_sa(Q, budget_s=0.3, seed=1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0  # generous upper bound; must not run away


# ---------------------------------------------------------------------------
# _flip_delta (incremental SA energy update)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, seed", [(8, 20), (10, 21), (12, 22)])
def test_flip_delta_matches_full_reevaluation(n, seed):
    """200 random single-bit flips: after each incremental update via
    _flip_delta (and the solve_sa qx update rule), the running energy and
    qx must match a full re-evaluation x'Qx / Q@x to ~1e-10."""
    Q = random_q(n, seed)
    rng = np.random.default_rng(seed + 1000)
    x = rng.integers(0, 2, size=n).astype(np.float64)
    qx = x @ Q
    e = float(x @ qx)
    for _ in range(200):
        i = int(rng.integers(n))
        xi = x[i]
        delta = _flip_delta(Q, qx, x, i)
        # same update rule as solve_sa
        x[i] = 1.0 - xi
        qx += (1.0 - 2.0 * xi) * Q[:, i]
        e += delta
        np.testing.assert_allclose(qx, x @ Q, atol=1e-10)
        assert e == pytest.approx(float(x @ Q @ x), abs=1e-10)
