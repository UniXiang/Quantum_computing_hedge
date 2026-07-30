"""US-market preselection and signed 24-qubit portfolio construction.

The full parquet universe is first screened to a liquid 500-stock pool and
ranked without look-ahead.  Nine stocks survive sector-capped factor ranking.
Each of those stocks, plus BTC/CL/XAU, receives a long and a short bit:
12 underlyings * 2 directions = 24 qubits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qubo_builder import (
    benchmark_downside_covariance,
    build_flexible_selection_qubo,
    qubo_to_ising,
)
from real_portfolio import (
    MarketInputs,
    _contract_prices,
    _market_betas,
    _zscore,
    allocate_selected_weights,
    load_config,
    price_factor_table,
)
from solvers import solve_sa


@dataclass
class USPreselection:
    candidates: list[dict[str, str]]
    stock_returns: pd.DataFrame
    benchmark_returns: pd.Series
    liquidity: pd.Series
    ranking: pd.DataFrame
    stats: dict[str, Any]


def _validate_market_frame(frame: pd.DataFrame) -> None:
    required = {
        "date", "code", "asset_type", "close", "volume", "sector",
        "tradable",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market parquet missing columns: {missing}")


def _cross_sectional_factor_ranking(
    returns: pd.DataFrame,
    liquidity: pd.Series,
    sectors: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    annualization = float(config["objective"]["annualization"])
    factor_config = config["preselection"]
    momentum_parts: list[np.ndarray] = []
    for requested in factor_config["momentum_windows"]:
        window = min(int(requested), len(returns))
        raw = np.log1p(returns.tail(window)).sum(axis=0).to_numpy()
        momentum_parts.append(_zscore(raw))
    momentum = np.mean(momentum_parts, axis=0)
    values = returns.to_numpy(dtype=np.float64)
    downside = np.sqrt(
        np.nanmean(np.minimum(values, 0.0) ** 2, axis=0) * annualization
    )
    annual_return = np.nanmean(values, axis=0) * annualization
    downside_adjusted = _zscore(
        annual_return / np.maximum(downside, 1e-8)
    )
    low_volatility = _zscore(
        -np.nanstd(values, axis=0, ddof=1) * np.sqrt(annualization)
    )
    liquidity_score = _zscore(
        np.log1p(liquidity.reindex(returns.columns).to_numpy())
    )
    weights = factor_config["factors"]
    score = (
        float(weights["momentum"]) * momentum
        + float(weights["downside_adjusted_return"]) * downside_adjusted
        + float(weights["low_volatility"]) * low_volatility
        + float(weights["liquidity"]) * liquidity_score
    )
    ranking = pd.DataFrame(
        {
            "sector": sectors.reindex(returns.columns).fillna("Unknown"),
            "average_dollar_volume": liquidity.reindex(returns.columns),
            "observations": returns.notna().sum(axis=0),
            "momentum_score": momentum,
            "downside_adjusted_score": downside_adjusted,
            "low_volatility_score": low_volatility,
            "liquidity_score": liquidity_score,
            "factor_score": score,
        },
        index=returns.columns,
    )
    ranking.index.name = "code"
    return ranking.sort_values(
        ["factor_score", "average_dollar_volume"],
        ascending=[False, False],
    )


def preselect_us_stocks(
    config: dict[str, Any],
    market_frame: pd.DataFrame | None = None,
) -> USPreselection:
    """Select nine US stocks using only rows on or before ``data.as_of``."""
    universe = config["universe"]
    as_of = pd.Timestamp(config["data"]["as_of"])
    if market_frame is None:
        path = Path(universe["market_daily_file"])
        market_frame = pd.read_parquet(
            path,
            columns=[
                "date", "code", "asset_type", "close", "volume", "sector",
                "tradable",
            ],
        )
    frame = market_frame.copy()
    _validate_market_frame(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.loc[
        frame["asset_type"].eq("stock")
        & frame["tradable"].fillna(False).astype(bool)
        & frame["date"].le(as_of)
        & frame["close"].gt(0.0)
        & frame["volume"].ge(0.0)
    ].copy()
    frame = frame.sort_values(["date", "code"]).drop_duplicates(
        ["date", "code"], keep="last"
    )
    if frame.empty:
        raise ValueError(f"no eligible US stock rows on or before {as_of.date()}")

    preferred = int(config["objective"]["preferred_window"])
    min_history = int(universe["min_history_observations"])
    closes = frame.pivot(index="date", columns="code", values="close")
    closes = closes.sort_index().tail(preferred + 1)
    returns = closes.pct_change(fill_method=None).iloc[1:]
    counts = returns.notna().sum()
    eligible_codes = counts.index[counts.ge(min_history)]
    if len(eligible_codes) < int(universe["finalists"]):
        raise ValueError(
            f"only {len(eligible_codes)} stocks have {min_history} returns"
        )

    liquidity_window = int(universe["liquidity_window"])
    recent = frame.groupby("code", group_keys=False).tail(liquidity_window)
    dollar_volume = (recent["close"] * recent["volume"]).groupby(
        recent["code"]
    ).mean()
    min_liquidity = float(universe["min_average_dollar_volume"])
    liquid_codes = eligible_codes.intersection(
        dollar_volume.index[dollar_volume.ge(min_liquidity)]
    )
    pool_size = int(universe["stock_pool_size"])
    liquid_codes = dollar_volume.reindex(liquid_codes).nlargest(
        pool_size
    ).index
    pool_returns = returns.reindex(columns=liquid_codes)
    latest_sector = (
        frame.dropna(subset=["sector"])
        .drop_duplicates("code", keep="last")
        .set_index("code")["sector"]
    )
    ranking = _cross_sectional_factor_ranking(
        pool_returns, dollar_volume, latest_sector, config
    )

    finalists = int(universe["finalists"])
    max_per_sector = int(config["preselection"]["max_per_sector"])
    chosen: list[str] = []
    sector_counts: dict[str, int] = {}
    for code, row in ranking.iterrows():
        sector = str(row["sector"])
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        chosen.append(str(code))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(chosen) == finalists:
            break
    if len(chosen) != finalists:
        raise ValueError(
            f"sector cap left only {len(chosen)} finalists, need {finalists}"
        )

    benchmark = pool_returns.mean(axis=1, skipna=True).rename("__benchmark__")
    selected_returns = pool_returns[chosen].copy()
    company_names = universe.get("company_names", {})
    candidates = [
        {
            "code": code,
            "name": str(company_names.get(code, code)),
            "sector": str(ranking.loc[code, "sector"]),
        }
        for code in chosen
    ]
    stats = {
        "source_rows_through_as_of": int(len(frame)),
        "source_symbols_through_as_of": int(frame["code"].nunique()),
        "history_eligible_symbols": int(len(eligible_codes)),
        "liquid_pool_symbols": int(len(liquid_codes)),
        "finalists": int(len(chosen)),
        "as_of": str(as_of.date()),
        "history_start": str(returns.index.min().date()),
        "history_end": str(returns.index.max().date()),
    }
    return USPreselection(
        candidates=candidates,
        stock_returns=selected_returns,
        benchmark_returns=benchmark,
        liquidity=dollar_volume.reindex(chosen),
        ranking=ranking,
        stats=stats,
    )


def load_us_market_inputs(
    config: dict[str, Any],
    market_frame: pd.DataFrame | None = None,
) -> tuple[MarketInputs, USPreselection]:
    selected = preselect_us_stocks(config, market_frame=market_frame)
    as_of = str(config["data"]["as_of"])
    calendar = selected.stock_returns.index
    base = selected.stock_returns.copy()
    for symbol, path in config["data"]["contract_files"].items():
        prices = _contract_prices(path, as_of)
        prices.index = pd.to_datetime(prices.index)
        base[str(symbol)] = prices.reindex(calendar).pct_change(fill_method=None)
    joined = base.join(selected.benchmark_returns, how="inner").dropna()
    minimum = int(config["objective"]["min_common_observations"])
    if len(joined) < minimum:
        raise ValueError(
            f"only {len(joined)} common observations, need at least {minimum}"
        )
    inputs = MarketInputs(
        base_returns=joined[base.columns],
        benchmark_returns=joined["__benchmark__"],
        stock_liquidity=selected.liquidity,
        as_of=as_of,
        preferred_observations=int(config["objective"]["preferred_window"]),
    )
    return inputs, selected


def build_us_n24_qubo(
    inputs: MarketInputs,
    preselection: USPreselection,
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    factors_config = dict(config)
    factors_config["universe"] = dict(config["universe"])
    factors_config["universe"]["candidates"] = preselection.candidates
    factors = price_factor_table(inputs, factors_config)
    betas = _market_betas(inputs.base_returns, inputs.benchmark_returns)
    rows: list[dict[str, Any]] = []
    source_columns: list[str] = []
    conflict_pairs: list[tuple[int, int]] = []

    underlyings: list[dict[str, str]] = [
        {
            "symbol": item["code"],
            "name": item["name"],
            "asset_type": "stock",
            "sector": item["sector"],
        }
        for item in preselection.candidates
    ]
    for instrument in config["hedges"]["instruments"]:
        symbol = str(instrument).split("-", 1)[0]
        underlyings.append(
            {
                "symbol": symbol,
                "name": str(instrument),
                "asset_type": "hedge",
                "sector": "Alternative",
            }
        )
    for item in underlyings:
        start = len(rows)
        scale = float(
            config["selection"]["stock_proxy_exposure"]
            if item["asset_type"] == "stock"
            else config["selection"]["hedge_proxy_exposure"]
        )
        for direction, sign in (("long", 1.0), ("short", -1.0)):
            rows.append(
                {
                    "variable": f"{item['symbol']}_{direction}",
                    "underlying": item["symbol"],
                    "name": item["name"],
                    "asset_type": item["asset_type"],
                    "sector": item["sector"],
                    "direction": direction,
                    "sign": sign,
                    "proxy_exposure": scale,
                }
            )
            source_columns.append(item["symbol"])
        conflict_pairs.append((start, start + 1))

    variables = pd.DataFrame(rows)
    expected_qubits = int(config["selection"]["qubits"])
    if len(variables) != expected_qubits:
        raise ValueError(
            f"variable table has {len(variables)} rows, expected "
            f"{expected_qubits}"
        )
    variable_returns = inputs.base_returns[source_columns].copy()
    variable_returns.columns = variables["variable"]
    annualization = float(config["objective"]["annualization"])
    covariance = benchmark_downside_covariance(
        variable_returns,
        inputs.benchmark_returns,
        shrink=True,
        min_observations=int(
            config["objective"]["min_benchmark_down_observations"]
        ),
    ) * annualization
    alpha = factors.loc[
        source_columns, "expected_annual_return_proxy"
    ].to_numpy()
    beta = betas.loc[source_columns].to_numpy()
    variables["factor_score"] = factors.loc[
        source_columns, "factor_score"
    ].to_numpy()
    variables["expected_annual_return_proxy"] = alpha
    variables["beta"] = beta
    costs = np.where(
        variables["asset_type"].eq("stock"),
        float(config["selection"]["stock_direction_holding_cost"]),
        float(config["selection"]["hedge_direction_holding_cost"]),
    ) * variables["proxy_exposure"].to_numpy()
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
            config["selection"]["direction_conflict_penalty"]
        ),
    )
    meta = {
        **preselection.stats,
        "common_observations": int(len(inputs.base_returns)),
        "common_window_start": str(inputs.base_returns.index[0].date()),
        "common_window_end": str(inputs.base_returns.index[-1].date()),
        "benchmark_down_observations": int(
            (inputs.benchmark_returns < 0.0).sum()
        ),
        "variable_layout": "9 US stocks x long/short + BTC/CL/XAU x long/short",
        "mutually_exclusive_pairs": conflict_pairs,
    }
    return Q, variables, covariance, meta


def solve_us_n24_portfolio(config: dict[str, Any]) -> dict[str, Any]:
    inputs, preselection = load_us_market_inputs(config)
    Q, variables, covariance, meta = build_us_n24_qubo(
        inputs, preselection, config
    )
    selected, energy = solve_sa(
        Q,
        budget_s=float(config["qaoa"]["sa_budget_seconds"]),
        seed=int(config["qaoa"]["seed"]),
        n_restarts=8,
    )
    weights, allocation = allocate_selected_weights(
        selected, variables, covariance, config
    )
    h, J, offset = qubo_to_ising(Q)
    output = variables.copy()
    output["selected"] = selected.astype(bool)
    output["weight"] = weights
    return {
        "inputs": inputs,
        "preselection": preselection,
        "Q": Q,
        "h": h,
        "J": J,
        "ising_offset": offset,
        "qubo_energy": float(energy),
        "variables": output,
        "meta": meta,
        "allocation": allocation,
    }


__all__ = [
    "USPreselection",
    "build_us_n24_qubo",
    "load_config",
    "load_us_market_inputs",
    "preselect_us_stocks",
    "solve_us_n24_portfolio",
]
