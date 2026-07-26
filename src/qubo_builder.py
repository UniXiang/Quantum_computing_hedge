"""qubo_builder.py — downside-semivariance portfolio-selection QUBO.

Objective (binary selection x in {0,1}^N, equal weight w = x/K assumed so
the objective stays quadratic, K fixed):

    min  x' S x / K^2  -  lam * mu' x / K  +  A * (sum x - K)^2
         +  gamma * |x - x_prev|

Definitions
-----------
- S (Sigma_minus): downside semivariance matrix over the rolling window.
  Chosen definition (Estrada-style co-semivariance):
      D = X_down' X_down / T,  X_down[t, i] = min(r[t, i], 0)
  i.e. each asset's return series is truncated at 0 and the raw
  (uncentered) second-moment matrix of the truncated series is used.
  This keeps time alignment across assets (unlike masking r<0 per asset)
  and reduces to the downside semivariance on the diagonal. With
  ``shrink=True`` (default) D is Ledoit-Wolf shrunk
  (sklearn.covariance.ledoit_wolf, assume_centered=True, consistent with
  the uncentered definition).
- mu: mean daily return over the same window.
- A: cardinality penalty coefficient.
- gamma: turnover penalty; x_prev binary vector of previous holdings or
  None (first period -> no turnover term). For binary x,
  |x_i - x_prev_i| is linear in x_i, so it enters Q's diagonal only.
- The constant A*K^2 (+ gamma*|x_prev|_1) is dropped from Q: it does not
  affect the optimizer. Energies reported through Q therefore differ from
  the full objective by that constant.

QUBO -> Ising mapping
---------------------
With x_i = (1 - z_i) / 2 (matching IsingQAOA's z = 1 - 2x convention),
for symmetric Q:

    h_i    = -(1/2) * sum_j Q_ij          (row sum)
    J_ij   = Q_ij / 2        (i < j, zero diagonal)
    offset = (trace(Q) + 2*sum_{i<j} Q_ij) / 4
           = sum_i Q_ii/2 + sum_{i<j} Q_ij/2

so that  x' Q x == E_ising(z(x)) + offset  exactly, with
E_ising(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf


def downside_semivariance(returns: pd.DataFrame, shrink: bool = True) -> np.ndarray:
    """Downside semivariance matrix of the return window.

    D = X_down' X_down / T with X_down = min(r, 0); optionally
    Ledoit-Wolf shrunk (assume_centered=True). See module docstring.
    """
    X = np.minimum(returns.to_numpy(dtype=np.float64), 0.0)
    if np.all(X == 0.0):
        # no downside samples at all: covariance is exactly zero and
        # shrinkage would be degenerate (0/0); return zeros.
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    if shrink:
        shrunk, _ = ledoit_wolf(X, assume_centered=True)
        return np.asarray(shrunk, dtype=np.float64)
    return (X.T @ X) / X.shape[0]


def build_qubo(returns: pd.DataFrame, K: int, lam: float, A: float,
               gamma: float = 0.0,
               x_prev: np.ndarray | None = None) -> np.ndarray:
    """Build the symmetric QUBO matrix Q for portfolio selection.

    Parameters
    ----------
    returns : (T, N) DataFrame of daily returns (columns = assets).
    K : target number of holdings (1 <= K <= N).
    lam : expected-return weight.
    A : cardinality penalty coefficient.
    gamma : turnover penalty coefficient (requires x_prev to take effect).
    x_prev : (N,) binary array of previous holdings, or None for the
        first period (no turnover term).

    Returns
    -------
    (N, N) symmetric Q with x'Qx equal to the objective up to a constant
    (see module docstring).
    """
    R = returns.to_numpy(dtype=np.float64)
    T, N = R.shape
    if not (1 <= K <= N):
        raise ValueError(f"K must satisfy 1 <= K <= N, got K={K}, N={N}")
    if x_prev is not None:
        x_prev = np.asarray(x_prev, dtype=np.float64)
        if x_prev.shape != (N,):
            raise ValueError(
                f"x_prev must have shape ({N},), got {x_prev.shape}")
        if not np.all((x_prev == 0.0) | (x_prev == 1.0)):
            raise ValueError("x_prev must be binary")

    S = downside_semivariance(returns, shrink=True)
    mu = R.mean(axis=0)

    Q = S / (K * K)
    Q = Q + A * np.ones((N, N)) - A * np.eye(N)  # off-diagonal += A
    diag = np.diag_indices(N)
    Q[diag] += -lam * mu / K + A * (1.0 - 2.0 * K)
    if gamma != 0.0 and x_prev is not None:
        # |x_i - p_i| = x_i if p_i=0 else (1 - x_i); linear part enters
        # the diagonal with sign +1 / -1, constant gamma*|p|_1 dropped.
        Q[diag] += gamma * (1.0 - 2.0 * x_prev)
    # exact symmetry (guard against float asymmetry from S)
    return (Q + Q.T) / 2.0


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map symmetric QUBO matrix to Ising (h, J, offset).

    Guarantees x'Qx == E_ising(z(x)) + offset for x_i = (1 - z_i)/2.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square, got {Q.shape}")
    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric")
    n = Q.shape[0]
    h = -0.5 * Q.sum(axis=1)
    J = Q / 2.0
    np.fill_diagonal(J, 0.0)
    offset = float(np.trace(Q) / 2.0 + np.triu(Q, k=1).sum() / 2.0)
    return h, J, offset


def ising_to_qubo_energy(x: np.ndarray, h: np.ndarray, J: np.ndarray,
                         offset: float) -> float:
    """QUBO energy of binary x via its Ising image: E_ising(z(x)) + offset.

    Consistency anchor for tests: must equal x'Qx for the Q that produced
    (h, J, offset).
    """
    x = np.asarray(x, dtype=np.float64)
    z = 1.0 - 2.0 * x
    e = float(h @ z) + float(offset)
    iu = np.triu_indices(len(h), k=1)
    e += float(np.sum(J[iu] * z[iu[0]] * z[iu[1]]))
    return e
