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


def benchmark_downside_covariance(
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        shrink: bool = True,
        min_observations: int = 20) -> np.ndarray:
    """Covariance of full asset returns conditional on benchmark down days.

    Unlike :func:`downside_semivariance`, this keeps positive hedge returns
    on benchmark-down days. A hedge that rises while stocks fall therefore
    receives a negative stock/hedge covariance and can genuinely reduce
    ``w' Sigma_down w``.

    Inputs are inner-aligned by index and rows with any missing value are
    dropped. Raises when fewer than ``min_observations`` benchmark-down
    observations remain, because a 24-asset covariance from a tiny tail
    sample is not meaningful.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series")
    if min_observations < 2:
        raise ValueError("min_observations must be >= 2")

    benchmark_name = "__benchmark__"
    while benchmark_name in returns.columns:
        benchmark_name += "_"
    aligned = returns.join(
        benchmark_returns.rename(benchmark_name), how="inner").dropna()
    down = aligned.loc[aligned[benchmark_name] < 0.0, returns.columns]
    if len(down) < min_observations:
        raise ValueError(
            f"need at least {min_observations} aligned benchmark-down "
            f"observations, got {len(down)}")
    X = down.to_numpy(dtype=np.float64)
    if shrink:
        covariance, _ = ledoit_wolf(X, assume_centered=False)
    else:
        covariance = np.cov(X, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
    return np.asarray((covariance + covariance.T) / 2.0,
                      dtype=np.float64)


def build_flexible_selection_qubo(
        expected_excess: np.ndarray,
        downside_covariance: np.ndarray,
        *,
        exposure_signs: np.ndarray | None = None,
        exposure_scales: np.ndarray | None = None,
        betas: np.ndarray | None = None,
        target_beta: float = 0.6,
        lambda_return: float = 1.0,
        lambda_downside: float = 2.0,
        lambda_beta: float = 1.0,
        holding_cost: float | np.ndarray = 0.01,
        previous_selection: np.ndarray | None = None,
        lambda_turnover: float = 0.0,
        mutually_exclusive: list[tuple[int, int]] | None = None,
        conflict_penalty: float = 2.0) -> np.ndarray:
    """Build a variable-cardinality selection QUBO.

    The binary variables are asset/direction decisions, not continuous
    weights. No ``(sum(x)-K)^2`` term is present: the number selected is
    chosen by the return/risk trade-off plus ``holding_cost``. Continuous
    portfolio weights are allocated in a later classical stage.

    Objective, with signed proxy exposure
    ``v = exposure_signs * exposure_scales * x``::

        lambda_downside * v' Sigma_down v
        - lambda_return * expected_excess' v
        + lambda_beta * (beta' v - target_beta)^2
        + holding_cost' x
        + lambda_turnover * |x - previous_selection|
        + conflict_penalty * sum(x_i*x_j for mutually-exclusive pairs)

    Constants independent of ``x`` are dropped. For an OKX instrument,
    represent long and short as two variables with signs ``+1``/``-1``
    and add their pair to ``mutually_exclusive``. ``exposure_scales``
    converts a selected bit into a realistic proxy weight, so beta and
    covariance penalties do not treat every selected asset as a 100%
    position.
    """
    alpha = np.asarray(expected_excess, dtype=np.float64)
    covariance = np.asarray(downside_covariance, dtype=np.float64)
    if alpha.ndim != 1:
        raise ValueError("expected_excess must be 1-D")
    n = len(alpha)
    if covariance.shape != (n, n):
        raise ValueError(
            f"downside_covariance must have shape ({n}, {n}), got "
            f"{covariance.shape}")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("downside_covariance must be symmetric")

    signs = (np.ones(n, dtype=np.float64) if exposure_signs is None
             else np.asarray(exposure_signs, dtype=np.float64))
    if signs.shape != (n,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(
            f"exposure_signs must have shape ({n},) with values +/-1")
    scales = (np.ones(n, dtype=np.float64) if exposure_scales is None
              else np.asarray(exposure_scales, dtype=np.float64))
    if scales.shape != (n,) or not np.all(np.isfinite(scales)):
        raise ValueError(
            f"exposure_scales must contain finite values with shape ({n},)")
    if np.any(scales <= 0.0):
        raise ValueError("exposure_scales must be strictly positive")
    beta = (np.zeros(n, dtype=np.float64) if betas is None
            else np.asarray(betas, dtype=np.float64))
    if beta.shape != (n,):
        raise ValueError(f"betas must have shape ({n},), got {beta.shape}")

    costs = np.asarray(holding_cost, dtype=np.float64)
    if costs.ndim == 0:
        costs = np.full(n, float(costs), dtype=np.float64)
    if costs.shape != (n,):
        raise ValueError(
            f"holding_cost must be scalar or shape ({n},), got "
            f"{costs.shape}")

    signed_exposure = signs * scales
    signed_covariance = covariance * np.outer(
        signed_exposure, signed_exposure)
    signed_alpha = alpha * signed_exposure
    signed_beta = beta * signed_exposure
    Q = lambda_downside * signed_covariance
    Q = Q + lambda_beta * np.outer(signed_beta, signed_beta)
    diag = np.diag_indices(n)
    Q[diag] += (
        -lambda_return * signed_alpha
        - 2.0 * lambda_beta * target_beta * signed_beta
        + costs)

    if previous_selection is not None:
        previous = np.asarray(previous_selection, dtype=np.float64)
        if previous.shape != (n,) or not np.all(np.isin(previous, (0, 1))):
            raise ValueError(
                f"previous_selection must be binary shape ({n},)")
        # |x_i-p_i| has x-dependent coefficient (1-2p_i).
        Q[diag] += lambda_turnover * (1.0 - 2.0 * previous)

    for pair in mutually_exclusive or []:
        if len(pair) != 2:
            raise ValueError(
                "each mutually_exclusive entry must contain two indices")
        i, j = (int(pair[0]), int(pair[1]))
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"invalid mutually-exclusive pair {(i, j)}")
        # x'Qx counts symmetric off-diagonal entries twice.
        Q[i, j] += conflict_penalty / 2.0
        Q[j, i] += conflict_penalty / 2.0

    return np.asarray((Q + Q.T) / 2.0, dtype=np.float64)


def build_weighted_cardinality_qubo(
        expected_returns: np.ndarray,
        downside_covariance: np.ndarray,
        *,
        target_holdings: int,
        lambda_return: float = 1.0,
        lambda_downside: float = 1.0,
        cardinality_penalty: float = 1.0,
        previous_selection: np.ndarray | None = None,
        lambda_turnover: float = 0.0) -> np.ndarray:
    """Build a long-or-cash, fixed-cardinality portfolio QUBO.

    Every binary variable represents one asset: ``x_i=1`` means long and
    ``x_i=0`` means the asset is absent.  The quantum-stage portfolio uses
    equal proxy weights ``w=x/K`` so the objective remains exactly quadratic::

        lambda_downside * w' Sigma_down w
        - lambda_return * mu' w
        + A * (sum(x)-K)^2
        + lambda_turnover * |x-previous_selection|

    The constant ``A*K^2`` and the constant part of turnover are omitted.
    A later classical allocator may refine weights inside the selected set,
    but it must keep the same long-only and full-investment semantics.
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    covariance = np.asarray(downside_covariance, dtype=np.float64)
    if mu.ndim != 1:
        raise ValueError("expected_returns must be 1-D")
    n = len(mu)
    if covariance.shape != (n, n):
        raise ValueError(
            f"downside_covariance must have shape ({n}, {n}), got "
            f"{covariance.shape}")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("downside_covariance must be symmetric")
    if not (1 <= int(target_holdings) <= n):
        raise ValueError(
            f"target_holdings must satisfy 1 <= K <= {n}, got "
            f"{target_holdings}")
    if cardinality_penalty <= 0.0:
        raise ValueError("cardinality_penalty must be positive")
    K = int(target_holdings)
    Q = (
        float(lambda_downside) * covariance / (K * K)
        - np.diag(float(lambda_return) * mu / K)
    )
    # A(sum x-K)^2, remembering that x_i^2=x_i and x'Qx counts
    # symmetric off-diagonal terms twice.
    A = float(cardinality_penalty)
    Q += A * np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(Q, np.diag(Q) - 2.0 * A * K)

    if previous_selection is not None:
        previous = np.asarray(previous_selection, dtype=np.float64)
        if previous.shape != (n,) or not np.all(np.isin(previous, (0, 1))):
            raise ValueError(
                f"previous_selection must be binary shape ({n},)")
        diag = np.diag_indices(n)
        Q[diag] += float(lambda_turnover) * (1.0 - 2.0 * previous)
    return np.asarray((Q + Q.T) / 2.0, dtype=np.float64)


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
