"""solvers.py — classical baselines for QUBO instances.

- solve_exact: vectorized exhaustive enumeration over 2^n states
  (practical up to n~20; milliseconds at n=16).
- solve_sa: simulated annealing with single-bit flips, incremental delta
  energies, geometric cooling, seeded RNG.

Reproducibility note: SA converts the wall-clock budget into a fixed
flip count (steps = budget_s * FLIPS_PER_SEC_16 * 16 / n, calibrated on
the dev machine at ~3e5 flips/s for n=16) and runs exactly that many
steps. Same (Q, budget_s, seed) therefore always yields the same result;
the actual wall-clock may differ from budget_s on much slower/faster
machines.

Convention: Q symmetric (N, N); energy E(x) = x'Qx for x in {0,1}^N.
"""
import numpy as np

# Calibrated flip rate at n=16 on the dev machine (see module docstring).
FLIPS_PER_SEC_16 = 200_000


def _check_q(Q):
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square, got {Q.shape}")
    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric")
    return Q


def energy_table(Q: np.ndarray) -> np.ndarray:
    """E(x) for all 2^n states; state m has x_i = (m >> i) & 1 (qubit 0 =
    LSB, matching IsingQAOA's bitstring convention)."""
    Q = _check_q(Q)
    n = Q.shape[0]
    m = np.arange(2**n, dtype=np.int64)
    bits = ((m[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return np.einsum("bi,ij,bj->b", bits, Q, bits, optimize=True)


def solve_exact(Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Exhaustive exact optimum. Returns (x_best, energy_best);
    x_best[i] = (argmin index >> i) & 1."""
    Q = _check_q(Q)
    e = energy_table(Q)
    m = int(np.argmin(e))
    n = Q.shape[0]
    x_best = np.array([(m >> i) & 1 for i in range(n)], dtype=np.int64)
    return x_best, float(e[m])


def solve_sa(Q: np.ndarray, budget_s: float, seed: int = 42,
             n_restarts: int = 4) -> tuple[np.ndarray, float]:
    """Simulated annealing under a (calibrated) time budget.

    Single-bit-flip neighborhood, incremental delta evaluation, geometric
    cooling from T0 to T0/1e4 over the step budget. ``n_restarts`` chains
    run round-robin from different seeded random starts; the best state
    seen across all chains is returned. Fully reproducible for a given
    (Q, budget_s, seed) — see module docstring.
    """
    Q = _check_q(Q)
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    total_steps = max(1, int(budget_s * FLIPS_PER_SEC_16 * 16.0 / n))

    # Temperature scale from typical flip delta magnitude.
    sample_x = rng.integers(0, 2, size=(64, n)).astype(np.float64)
    qdiag = Q.diagonal()
    d = (1 - 2 * sample_x) * (qdiag * (1 - 2 * sample_x)
                              + 2 * (sample_x @ Q))
    t0_scale = float(np.abs(d).mean()) or 1.0
    T_init, T_final = 2.0 * t0_scale, 2.0 * t0_scale / 1e4
    log_ratio = np.log(T_final / T_init)

    xs = rng.integers(0, 2, size=(n_restarts, n)).astype(np.float64)
    qxs = xs @ Q
    es = np.einsum("bi,bi->b", xs, qxs)
    best_flat = int(np.argmin(es))
    x_best = xs[best_flat].copy()
    e_best = float(es[best_flat])

    steps_per_chain = max(1, total_steps // n_restarts)
    for c in range(n_restarts):
        x, qx, e = xs[c], qxs[c], float(es[c])
        for step in range(steps_per_chain):
            T = T_init * np.exp(log_ratio * step / max(steps_per_chain - 1, 1))
            i = int(rng.integers(n))
            xi = x[i]
            # delta of flipping bit i: (1-2x_i) * (Q_ii*(1-2x_i) + 2*(Qx)_i)
            delta = (1.0 - 2.0 * xi) * (Q[i, i] * (1.0 - 2.0 * xi)
                                        + 2.0 * qx[i])
            if delta <= 0.0 or rng.random() < np.exp(-delta / T):
                x[i] = 1.0 - xi
                qx += (1.0 - 2.0 * xi) * Q[:, i]
                e += delta
                if e < e_best:
                    e_best = e
                    x_best = x.copy()
    return x_best.astype(np.int64), e_best
