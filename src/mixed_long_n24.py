"""Mixed A/US/alternative long-or-cash 24-qubit portfolio model."""
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
from real_portfolio import MarketInputs, _contract_prices, _market_betas, _zscore
from solvers import solve_sa


@dataclass
class MixedInputs:
    market_inputs: MarketInputs
    finalists: list[dict[str, str]]
    ranking: pd.DataFrame
    universe_stats: dict[str, Any]


def _cache_path(cache_dir: Path, market: str, code: str) -> Path:
    if market == "US":
        return cache_dir / f"us_{code.upper()}_1year.pkl"
    exchange = "sz" if code[0] in "023" else "sh"
    return cache_dir / f"{exchange}_{code.zfill(6)}_1year.pkl"


def _read_cache_return(
    cache_dir: Path, market: str, code: str, as_of: pd.Timestamp
) -> tuple[pd.Series, float, dict[str, Any]]:
    path = _cache_path(cache_dir, market, code)
    if not path.exists():
        raise FileNotFoundError(f"missing cache file: {path}")
    frame = pd.read_pickle(path)
    required = {"date", "close", "amount", "tradestatus", "pctChg"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["tradestatus"] = pd.to_numeric(
        data["tradestatus"], errors="coerce"
    ).fillna(0)
    data["pctChg"] = pd.to_numeric(data["pctChg"], errors="coerce")
    data = data.loc[data["date"].le(as_of)].sort_values("date")
    data = data.drop_duplicates("date", keep="last")
    if len(data) < 61:
        raise ValueError(f"{path.name}: only {len(data)} rows through {as_of.date()}")
    raw_return = data["pctChg"] / 100.0
    fallback = data["close"].pct_change(fill_method=None)
    values = raw_return.where(raw_return.notna(), fallback)
    values = values.where(data["tradestatus"].gt(0), 0.0)
    series = pd.Series(values.to_numpy(), index=data["date"], name=code)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    liquidity = float(data["amount"].tail(60).replace(0, np.nan).median())
    return series, liquidity, {
        "file": path.name,
        "rows": int(len(data)),
        "start": str(data["date"].min().date()),
        "end": str(data["date"].max().date()),
    }


def _rank_stocks(
    returns: pd.DataFrame,
    liquidity: pd.Series,
    markets: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    annualization = float(config["objective"]["annualization"])
    momentum_parts = []
    for requested in config["preselection"]["momentum_windows"]:
        window = min(int(requested), len(returns))
        momentum_parts.append(
            _zscore(np.log1p(returns.tail(window)).sum(axis=0).to_numpy())
        )
    momentum = np.mean(momentum_parts, axis=0)
    values = returns.to_numpy(dtype=np.float64)
    downside = np.sqrt(
        np.mean(np.minimum(values, 0.0) ** 2, axis=0) * annualization
    )
    annual_return = np.mean(values, axis=0) * annualization
    downside_adjusted = _zscore(annual_return / np.maximum(downside, 1e-8))
    low_volatility = _zscore(
        -np.std(values, axis=0, ddof=1) * np.sqrt(annualization)
    )
    # Currency units are not comparable, so liquidity is standardized inside
    # each market rather than across CNY and USD.
    liquidity_score = np.zeros(len(returns.columns), dtype=np.float64)
    for market in ("A", "US"):
        mask = markets.reindex(returns.columns).to_numpy() == market
        liquidity_score[mask] = _zscore(
            np.log1p(liquidity.reindex(returns.columns[mask]).to_numpy())
        )
    weights = config["preselection"]["factors"]
    score = (
        float(weights["momentum"]) * momentum
        + float(weights["downside_adjusted_return"]) * downside_adjusted
        + float(weights["low_volatility"]) * low_volatility
        + float(weights["liquidity"]) * liquidity_score
    )
    ranking = pd.DataFrame(
        {
            "market": markets.reindex(returns.columns),
            "observations": returns.notna().sum(),
            "median_amount_60d": liquidity.reindex(returns.columns),
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
        ["factor_score", "median_amount_60d"], ascending=[False, False]
    )


def load_mixed_inputs(config: dict[str, Any]) -> MixedInputs:
    universe = config["universe"]
    as_of = pd.Timestamp(config["data"]["as_of"])
    cache_dir = Path(config["data"]["cache_dir"])
    candidates: list[dict[str, str]] = []
    for item in universe["a_share_candidates"]:
        candidates.append(
            {
                "code": str(item["code"]).zfill(6),
                "name": str(item["name"]),
                "sector": str(item["sector"]),
                "market": "A",
            }
        )
    for item in universe["us_candidates"]:
        candidates.append(
            {
                "code": str(item["code"]).upper(),
                "name": str(item["name"]),
                "sector": str(item["sector"]),
                "market": "US",
            }
        )

    return_parts: dict[str, pd.Series] = {}
    liquidity: dict[str, float] = {}
    cache_audit: dict[str, Any] = {}
    for item in candidates:
        series, amount, audit = _read_cache_return(
            cache_dir, item["market"], item["code"], as_of
        )
        return_parts[item["code"]] = series
        liquidity[item["code"]] = amount
        cache_audit[item["code"]] = audit
    # A and US market holidays are explicit zero-return days on the global
    # union calendar. This avoids discarding most observations via inner join.
    stock_returns = pd.concat(return_parts, axis=1).sort_index().fillna(0.0)
    preferred = int(config["objective"]["preferred_window"])
    stock_returns = stock_returns.tail(preferred)
    markets = pd.Series({item["code"]: item["market"] for item in candidates})
    ranking = _rank_stocks(
        stock_returns, pd.Series(liquidity), markets, config
    )

    finalist_count = int(universe["finalists"])
    max_per_market = int(universe["max_per_market"])
    selected_codes: list[str] = []
    market_counts = {"A": 0, "US": 0}
    for code, row in ranking.iterrows():
        market = str(row["market"])
        if market_counts[market] >= max_per_market:
            continue
        selected_codes.append(str(code))
        market_counts[market] += 1
        if len(selected_codes) == finalist_count:
            break
    if len(selected_codes) != finalist_count:
        raise ValueError(
            f"market cap selected {len(selected_codes)}, need {finalist_count}"
        )
    by_code = {item["code"]: item for item in candidates}
    finalists = [by_code[code] for code in selected_codes]

    benchmark = stock_returns.mean(axis=1).rename("__benchmark__")
    selected_returns = stock_returns[selected_codes].copy()
    calendar = selected_returns.index
    for symbol, path in config["data"]["contract_files"].items():
        prices = _contract_prices(path, str(as_of.date()))
        prices.index = pd.to_datetime(prices.index)
        # Carry the most recent daily price onto the global stock calendar;
        # pct_change then gives zero on a no-move/non-trading day.
        aligned = prices.reindex(prices.index.union(calendar)).sort_index().ffill()
        selected_returns[str(symbol)] = (
            aligned.reindex(calendar).pct_change(fill_method=None)
        )
    joined = selected_returns.join(benchmark, how="inner").dropna()
    minimum = int(config["objective"]["min_common_observations"])
    if len(joined) < minimum:
        raise ValueError(f"only {len(joined)} common observations, need {minimum}")
    inputs = MarketInputs(
        base_returns=joined[selected_returns.columns],
        benchmark_returns=joined["__benchmark__"],
        stock_liquidity=pd.Series(liquidity).reindex(selected_codes),
        as_of=str(as_of.date()),
        preferred_observations=preferred,
    )
    stats = {
        "candidate_count": len(candidates),
        "candidate_markets": {"A": 18, "US": 18},
        "finalist_count": len(finalists),
        "finalist_markets": market_counts,
        "global_window_start": str(joined.index.min().date()),
        "global_window_end": str(joined.index.max().date()),
        "common_observations": int(len(joined)),
        "cache_audit": cache_audit,
    }
    return MixedInputs(inputs, finalists, ranking, stats)


def build_mixed_long_qubo(
    mixed: MixedInputs, config: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    inputs = mixed.market_inputs
    returns = inputs.base_returns
    factors = _rank_stocks(
        returns,
        pd.concat(
            [
                inputs.stock_liquidity,
                pd.Series({"BTC": 0.0, "CL": 0.0, "XAU": 0.0}),
            ]
        ),
        pd.Series(
            {
                **{item["code"]: item["market"] for item in mixed.finalists},
                "BTC": "ALT",
                "CL": "ALT",
                "XAU": "ALT",
            }
        ),
        config,
    )
    # Recompute factors for all 24 assets without using incommensurable
    # alternative liquidity. The rank helper's liquidity term is zero for ALT.
    betas = _market_betas(returns, inputs.benchmark_returns)
    finalist_map = {item["code"]: item for item in mixed.finalists}
    rows = []
    for symbol in returns.columns:
        if symbol in finalist_map:
            item = finalist_map[symbol]
            asset_type = "stock"
            market = item["market"]
            name = item["name"]
            sector = item["sector"]
            scale = float(config["selection"]["stock_proxy_exposure"])
            cost = float(config["selection"]["stock_holding_cost"])
        else:
            asset_type = "alternative"
            market = "GLOBAL"
            name = {"BTC": "Bitcoin", "CL": "Crude Oil", "XAU": "Gold"}[symbol]
            sector = "Alternative"
            scale = float(config["selection"]["alternative_proxy_exposure"])
            cost = float(config["selection"]["alternative_holding_cost"])
        rows.append(
            {
                "variable": symbol,
                "underlying": symbol,
                "name": name,
                "market": market,
                "asset_type": asset_type,
                "sector": sector,
                "direction": "long",
                "sign": 1.0,
                "proxy_exposure": scale,
                "holding_cost": cost,
            }
        )
    variables = pd.DataFrame(rows)
    n = int(config["selection"]["qubits"])
    if len(variables) != n:
        raise ValueError(f"variable table has {len(variables)} rows, expected {n}")

    annualization = float(config["objective"]["annualization"])
    covariance = benchmark_downside_covariance(
        returns,
        inputs.benchmark_returns,
        shrink=True,
        min_observations=int(
            config["objective"]["min_benchmark_down_observations"]
        ),
    ) * annualization
    # Factor-score columns are already aligned to return columns.
    score = factors.reindex(returns.columns)["factor_score"].to_numpy()
    alpha = float(config["objective"]["signal_annual_return_scale"]) * score
    beta = betas.reindex(returns.columns).to_numpy()
    variables["factor_score"] = score
    variables["expected_annual_return_proxy"] = alpha
    variables["beta"] = beta
    costs = (
        variables["holding_cost"].to_numpy()
        * variables["proxy_exposure"].to_numpy()
    )
    objective = config["objective"]
    Q = build_flexible_selection_qubo(
        alpha,
        covariance,
        exposure_signs=np.ones(n),
        exposure_scales=variables["proxy_exposure"].to_numpy(),
        betas=beta,
        target_beta=float(objective["beta_target"]),
        lambda_return=float(objective["expected_return_weight"]),
        lambda_downside=float(objective["downside_risk_weight"]),
        lambda_beta=float(objective["beta_penalty_weight"]),
        holding_cost=costs,
        mutually_exclusive=[],
    )
    meta = {
        **mixed.universe_stats,
        "n": n,
        "bit_semantics": {"0": "cash", "1": "long"},
        "benchmark_down_observations": int(
            (inputs.benchmark_returns < 0.0).sum()
        ),
        "annualization": annualization,
    }
    return Q, variables, covariance, meta


def prepare_problem(config: dict[str, Any]) -> dict[str, Any]:
    mixed = load_mixed_inputs(config)
    Q, variables, covariance, meta = build_mixed_long_qubo(mixed, config)
    selected, energy = solve_sa(
        Q,
        budget_s=float(config["qaoa"]["sa_budget_seconds"]),
        seed=int(config["qaoa"]["seed"]),
        n_restarts=8,
    )
    h, J, offset = qubo_to_ising(Q)
    return {
        "Q": Q,
        "h": h,
        "J": J,
        "offset": offset,
        "variables": variables,
        "covariance": covariance,
        "meta": meta,
        "ranking": mixed.ranking,
        "sa_selection": selected,
        "sa_energy": float(energy),
    }
