"""Build and solve the 18-stock + 3-contract real n=24 portfolio.

The QUBO selects assets/directions with variable cardinality.  A selected
bit is mapped to a small proxy exposure while constructing the Hamiltonian;
the final continuous weights are solved separately under portfolio limits.

This module is intentionally strict about time alignment: every price
series is truncated at ``as_of`` before returns or factors are calculated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from data_loader import _file_for, _normalize_code, load_returns
from qubo_builder import (
    benchmark_downside_covariance,
    build_flexible_selection_qubo,
    qubo_to_ising,
)
from solvers import solve_sa


@dataclass
class MarketInputs:
    """Aligned return matrix and metadata used by one portfolio decision."""

    base_returns: pd.DataFrame
    benchmark_returns: pd.Series
    stock_liquidity: pd.Series
    as_of: str
    preferred_observations: int


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("portfolio config must be a YAML mapping")
    return config


def _contract_prices(path: str | Path, as_of: str) -> pd.Series:
    """Load confirmed closes, with a hard no-look-ahead cutoff."""
    frame = pd.read_csv(path)
    if "close" not in frame:
        raise ValueError(f"{path}: missing close column")
    if "confirm" in frame:
        confirmed = pd.to_numeric(frame["confirm"], errors="coerce")
        frame = frame.loc[confirmed == 1].copy()
    date_column = "date" if "date" in frame else "datetime"
    if date_column not in frame:
        raise ValueError(f"{path}: missing date/datetime column")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(
        closes.to_numpy(), index=dates.dt.strftime("%Y-%m-%d"),
        name=Path(path).stem,
    ).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.loc[series.index <= str(as_of)]


def _stock_liquidity(
        codes: list[str], as_of: str, window: int, cache_dir: str,
) -> pd.Series:
    """Trailing average traded amount, truncated before taking the window."""
    result: dict[str, float] = {}
    for code in codes:
        code6 = _normalize_code(code)
        frame = pd.read_pickle(_file_for(code6, cache_dir))
        frame = frame.loc[frame["date"].astype(str) <= str(as_of)].tail(window)
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        result[code6] = float(amount.mean())
    return pd.Series(result, dtype=np.float64)


def load_market_inputs(config: dict[str, Any]) -> MarketInputs:
    """Load stocks, benchmark, and contract returns on one A-share calendar."""
    data = config["data"]
    objective = config["objective"]
    candidates = config["universe"]["candidates"]
    codes = [str(item["code"]).zfill(6) for item in candidates]
    as_of = str(data["as_of"])
    preferred = int(objective["preferred_window"])
    benchmark = str(data["benchmark_code"]).zfill(6)
    stock_and_benchmark = load_returns(
        codes + [benchmark], as_of, preferred,
        cache_dir=str(data["stock_cache_dir"]),
    )
    calendar = stock_and_benchmark.index
    base = stock_and_benchmark[codes].copy()

    for symbol, path in data["contract_files"].items():
        prices = _contract_prices(path, as_of)
        # Select A-share dates at the price level first.  Monday's return
        # then includes the whole weekend rather than only one crypto day.
        aligned_prices = prices.reindex(calendar)
        base[str(symbol)] = aligned_prices.pct_change(fill_method=None)

    joined = base.join(
        stock_and_benchmark[benchmark].rename("__benchmark__"),
        how="inner",
    ).dropna()
    minimum = int(objective["min_common_observations"])
    if len(joined) < minimum:
        raise ValueError(
            f"only {len(joined)} common observations, need at least {minimum}")
    liquidity = _stock_liquidity(
        codes, as_of, preferred, str(data["stock_cache_dir"]))
    return MarketInputs(
        base_returns=joined[base.columns],
        benchmark_returns=joined["__benchmark__"],
        stock_liquidity=liquidity,
        as_of=as_of,
        preferred_observations=preferred,
    )


def _zscore(values: np.ndarray, clip: float = 2.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - values.mean()) / std, -clip, clip)


def price_factor_table(
        inputs: MarketInputs, config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate deterministic, cross-sectional price-factor signals."""
    returns = inputs.base_returns
    windows = [
        min(int(value), len(returns))
        for value in config["preselection"]["momentum_windows"]
    ]
    momentum_parts = []
    for window in windows:
        log_momentum = np.log1p(returns.tail(window)).sum(axis=0).to_numpy()
        momentum_parts.append(_zscore(log_momentum))
    momentum = np.mean(momentum_parts, axis=0)

    annualization = float(config["objective"]["annualization"])
    values = returns.to_numpy(dtype=np.float64)
    downside_deviation = np.sqrt(
        np.mean(np.minimum(values, 0.0) ** 2, axis=0) * annualization)
    annual_return = returns.mean(axis=0).to_numpy() * annualization
    downside_adjusted = _zscore(
        annual_return / np.maximum(downside_deviation, 1e-8))
    low_volatility = _zscore(
        -returns.std(axis=0).to_numpy() * np.sqrt(annualization))

    stock_codes = []
    for item in config["universe"]["candidates"]:
        raw_code = str(item["code"])
        stock_codes.append(
            raw_code.zfill(6) if raw_code.isdigit() else raw_code.upper()
        )
    liquidity = np.zeros(returns.shape[1], dtype=np.float64)
    liquidity[:len(stock_codes)] = _zscore(
        np.log1p(inputs.stock_liquidity.reindex(stock_codes).to_numpy()))

    weights = config["preselection"]["factors"]
    score = (
        float(weights["momentum"]) * momentum
        + float(weights["downside_adjusted_return"]) * downside_adjusted
        + float(weights["low_volatility"]) * low_volatility
        + float(weights["liquidity"]) * liquidity
    )
    scale = float(config["objective"]["signal_annual_return_scale"])
    return pd.DataFrame(
        {
            "momentum_score": momentum,
            "downside_adjusted_score": downside_adjusted,
            "low_volatility_score": low_volatility,
            "liquidity_score": liquidity,
            "factor_score": score,
            "expected_annual_return_proxy": scale * score,
        },
        index=returns.columns,
    )


def _market_betas(
        returns: pd.DataFrame, benchmark: pd.Series,
) -> pd.Series:
    benchmark_variance = float(np.var(benchmark, ddof=1))
    if benchmark_variance <= 0.0:
        raise ValueError("benchmark variance must be positive")
    betas = {
        column: np.cov(returns[column], benchmark, ddof=1)[0, 1]
        / benchmark_variance
        for column in returns
    }
    return pd.Series(betas, dtype=np.float64)


def build_real_qubo(
        inputs: MarketInputs, config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Create the n=24 QUBO and its auditable variable table."""
    candidates = config["universe"]["candidates"]
    factors = price_factor_table(inputs, config)
    betas = _market_betas(inputs.base_returns, inputs.benchmark_returns)
    rows: list[dict[str, Any]] = []
    source_columns: list[str] = []

    stock_scale = float(config["selection"]["stock_proxy_exposure"])
    hedge_scale = float(config["selection"]["hedge_proxy_exposure"])
    for item in candidates:
        code = str(item["code"]).zfill(6)
        rows.append({
            "variable": code,
            "underlying": code,
            "name": str(item["name"]),
            "asset_type": "stock",
            "direction": "long",
            "sign": 1.0,
            "proxy_exposure": stock_scale,
        })
        source_columns.append(code)
    for instrument in config["hedges"]["instruments"]:
        symbol = str(instrument).split("-", 1)[0]
        for direction, sign in (("long", 1.0), ("short", -1.0)):
            rows.append({
                "variable": f"{symbol}_{direction}",
                "underlying": symbol,
                "name": str(instrument),
                "asset_type": "hedge",
                "direction": direction,
                "sign": sign,
                "proxy_exposure": hedge_scale,
            })
            source_columns.append(symbol)
    variables = pd.DataFrame(rows)
    if len(variables) != int(config["selection"]["qubits"]):
        raise ValueError(
            f"variable table has {len(variables)} rows, expected "
            f"{config['selection']['qubits']}")

    variable_returns = inputs.base_returns[source_columns].copy()
    variable_returns.columns = variables["variable"]
    annualization = float(config["objective"]["annualization"])
    covariance = benchmark_downside_covariance(
        variable_returns,
        inputs.benchmark_returns,
        shrink=True,
        min_observations=int(
            config["objective"]["min_benchmark_down_observations"]),
    ) * annualization
    alpha = factors.loc[source_columns, "expected_annual_return_proxy"].to_numpy()
    beta = betas.loc[source_columns].to_numpy()
    variables["factor_score"] = factors.loc[
        source_columns, "factor_score"].to_numpy()
    variables["expected_annual_return_proxy"] = alpha
    variables["beta"] = beta

    costs = np.where(
        variables["asset_type"].to_numpy() == "stock",
        float(config["selection"]["stock_holding_cost"]),
        float(config["selection"]["hedge_direction_holding_cost"]),
    ) * variables["proxy_exposure"].to_numpy()
    conflict_pairs = [(18, 19), (20, 21), (22, 23)]
    objective = config["objective"]
    Q = build_flexible_selection_qubo(
        alpha,
        covariance,
        exposure_signs=variables["sign"].to_numpy(),
        exposure_scales=variables["proxy_exposure"].to_numpy(),
        betas=beta,
        target_beta=float(objective["beta_target"]),
        lambda_return=float(objective["expected_return_weight"]),
        lambda_downside=float(objective["downside_risk_weight"]),
        lambda_beta=float(objective["beta_penalty_weight"]),
        holding_cost=costs,
        mutually_exclusive=conflict_pairs,
        conflict_penalty=float(
            config["selection"]["direction_conflict_penalty"]),
    )
    meta = {
        "common_observations": len(inputs.base_returns),
        "window_start": str(inputs.base_returns.index[0]),
        "window_end": str(inputs.base_returns.index[-1]),
        "benchmark_down_observations": int(
            (inputs.benchmark_returns < 0.0).sum()),
        "annualization": annualization,
    }
    return Q, variables, covariance, meta


def allocate_selected_weights(
        selected: np.ndarray,
        variables: pd.DataFrame,
        covariance: np.ndarray,
        config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Continuous SLSQP allocation on the QUBO-selected subset."""
    selected = np.asarray(selected, dtype=np.int64)
    active = np.flatnonzero(selected)
    if not len(active):
        return np.zeros(len(selected)), {
            "gross_exposure": 0.0, "net_exposure": 0.0, "portfolio_beta": 0.0,
        }
    table = variables.iloc[active]
    signs = table["sign"].to_numpy(dtype=np.float64)
    alpha = table["expected_annual_return_proxy"].to_numpy(dtype=np.float64)
    signed_alpha = alpha * signs
    beta = table["beta"].to_numpy(dtype=np.float64) * signs
    cov = covariance[np.ix_(active, active)]
    allocation = config["allocation"]
    caps = np.where(
        table["asset_type"].to_numpy() == "stock",
        float(allocation["max_stock_weight"]),
        float(allocation["max_contract_abs_weight"]),
    )
    minimum_weight = float(allocation.get("min_selected_weight", 0.0))
    bounds = [(minimum_weight, float(cap)) for cap in caps]

    lower_net = float(allocation["min_net_exposure"])
    upper_net = float(allocation["max_net_exposure"])
    max_gross = float(allocation["max_gross_exposure"])
    target_beta = float(allocation["target_beta"])
    objective = config["objective"]
    risk_weight = float(objective["downside_risk_weight"])
    return_weight = float(objective["expected_return_weight"])
    beta_weight = float(objective["beta_penalty_weight"])

    def loss(magnitudes: np.ndarray) -> float:
        signed_weights = signs * magnitudes
        return float(
            risk_weight * signed_weights @ cov @ signed_weights
            - return_weight * signed_alpha @ magnitudes
            + beta_weight * (beta @ magnitudes - target_beta) ** 2)

    constraints = [
        {"type": "ineq", "fun": lambda w: signs @ w - lower_net},
        {"type": "ineq", "fun": lambda w: upper_net - signs @ w},
        {"type": "ineq", "fun": lambda w: max_gross - w.sum()},
    ]
    initial = np.minimum(caps, np.where(signs > 0.0, 0.06, 0.02))
    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"continuous allocation failed: {result.message}")
    weights = np.zeros(len(selected), dtype=np.float64)
    weights[active] = signs * result.x
    diagnostics = {
        "gross_exposure": float(np.abs(weights).sum()),
        "net_exposure": float(weights.sum()),
        "portfolio_beta": float(beta @ result.x),
        "objective": float(result.fun),
    }
    return weights, diagnostics


def solve_real_portfolio(
        config: dict[str, Any],
) -> dict[str, Any]:
    """Run the deterministic SA baseline and continuous allocation."""
    inputs = load_market_inputs(config)
    Q, variables, covariance, meta = build_real_qubo(inputs, config)
    qaoa_config = config["qaoa"]
    selected, energy = solve_sa(
        Q,
        budget_s=float(qaoa_config["sa_budget_seconds"]),
        seed=int(qaoa_config["seed"]),
        n_restarts=8,
    )
    weights, allocation = allocate_selected_weights(
        selected, variables, covariance, config)
    h, J, offset = qubo_to_ising(Q)
    output = variables.copy()
    output["selected"] = selected.astype(bool)
    output["weight"] = weights
    return {
        "inputs": inputs,
        "Q": Q,
        "h": h,
        "J": J,
        "ising_offset": offset,
        "qubo_energy": float(energy),
        "variables": output,
        "meta": meta,
        "allocation": allocation,
    }
