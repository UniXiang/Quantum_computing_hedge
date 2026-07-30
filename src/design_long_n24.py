"""Design-document aligned n=24 long-or-cash portfolio implementation.

The reduced n=24 profile uses one qubit per asset: 21 preselected stocks
plus BTC, crude oil, and gold.  It deliberately omits multiplicative hedge
ratio bits because selection_bit * ratio_bits would make the variance term
higher than quadratic.  Instead the QUBO uses equal proxy weights and a
classical long-only allocator refines weights after discrete selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mixed_long_n24 import _rank_stocks, _read_cache_return
from qubo_builder import (
    build_weighted_cardinality_qubo,
    downside_semivariance,
    qubo_to_ising,
)
from real_portfolio import _contract_prices
from solvers import solve_sa


@dataclass
class DesignInputs:
    returns: pd.DataFrame
    finalists: list[dict[str, str]]
    ranking: pd.DataFrame
    audit: dict[str, Any]


def _candidate_table(config: dict[str, Any]) -> list[dict[str, str]]:
    candidates = []
    for item in config["universe"]["a_share_candidates"]:
        candidates.append(
            {
                "code": str(item["code"]).zfill(6),
                "name": str(item["name"]),
                "sector": str(item["sector"]),
                "market": "A",
            }
        )
    for item in config["universe"]["us_candidates"]:
        candidates.append(
            {
                "code": str(item["code"]).upper(),
                "name": str(item["name"]),
                "sector": str(item["sector"]),
                "market": "US",
            }
        )
    return candidates


def _select_sector_balanced(
    ranking: pd.DataFrame,
    candidates: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    universe = config["universe"]
    count = int(universe["finalists"])
    max_market = int(universe["max_per_market"])
    max_sector = int(universe["max_per_sector"])
    lookup = {item["code"]: item for item in candidates}
    chosen = []
    market_counts: dict[str, int] = {}
    sector_counts: dict[str, int] = {}
    for code, row in ranking.iterrows():
        item = lookup[str(code)]
        market = item["market"]
        sector = item["sector"]
        if market_counts.get(market, 0) >= max_market:
            continue
        if sector_counts.get(sector, 0) >= max_sector:
            continue
        chosen.append(item)
        market_counts[market] = market_counts.get(market, 0) + 1
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(
            f"market/sector caps selected {len(chosen)} stocks, need {count}"
        )
    return chosen


def load_design_inputs(config: dict[str, Any]) -> DesignInputs:
    as_of = pd.Timestamp(config["data"]["as_of"])
    cache_dir = Path(config["data"]["cache_dir"])
    window = int(config["objective"]["window"])
    candidates = _candidate_table(config)
    series_map = {}
    liquidity = {}
    cache_audit = {}
    for item in candidates:
        series, amount, audit = _read_cache_return(
            cache_dir, item["market"], item["code"], as_of
        )
        series_map[item["code"]] = series
        liquidity[item["code"]] = amount
        cache_audit[item["code"]] = audit

    # Returns are returns of a globally held portfolio. On a date when one
    # market is closed, that market's marked close is unchanged, hence zero.
    stock_returns = pd.concat(series_map, axis=1).sort_index().fillna(0.0)
    stock_returns = stock_returns.tail(window)
    markets = pd.Series({item["code"]: item["market"] for item in candidates})
    ranking = _rank_stocks(
        stock_returns, pd.Series(liquidity), markets, config
    )
    finalists = _select_sector_balanced(ranking, candidates, config)
    codes = [item["code"] for item in finalists]
    returns = stock_returns[codes].copy()
    calendar = returns.index
    for symbol, path in config["data"]["contract_files"].items():
        prices = _contract_prices(path, str(as_of.date()))
        prices.index = pd.to_datetime(prices.index)
        aligned = prices.reindex(prices.index.union(calendar)).sort_index().ffill()
        returns[str(symbol)] = aligned.reindex(calendar).pct_change(
            fill_method=None
        )
    returns = returns.dropna()
    if len(returns) < window - 5:
        raise ValueError(
            f"only {len(returns)} complete observations for requested "
            f"{window}-observation window"
        )
    audit = {
        "as_of_session_date": str(as_of.date()),
        "decision_time": str(config["data"]["decision_time"]),
        "window_requested": window,
        "window_observations": int(len(returns)),
        "window_start": str(returns.index.min().date()),
        "window_end": str(returns.index.max().date()),
        "candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "finalist_markets": {
            market: sum(item["market"] == market for item in finalists)
            for market in ("A", "US")
        },
        "cache_audit": cache_audit,
    }
    return DesignInputs(returns, finalists, ranking, audit)


def build_design_problem(
    inputs: DesignInputs,
    config: dict[str, Any],
    previous_selection: np.ndarray | None = None,
) -> dict[str, Any]:
    returns = inputs.returns
    n = returns.shape[1]
    if n < 4:
        raise ValueError(f"expected at least 4 one-asset variables, got {n}")
    annualization = float(config["objective"]["annualization"])
    covariance = downside_semivariance(returns, shrink=True) * annualization
    raw_expected = returns.mean(axis=0).to_numpy() * annualization
    prior_observations = float(
        config["objective"].get("expected_return_prior_observations", 0.0)
    )
    reliability = len(returns) / (len(returns) + prior_observations)
    clip_low, clip_high = config["objective"]["expected_return_clip"]
    expected = np.clip(
        raw_expected * reliability,
        float(clip_low),
        float(clip_high),
    )
    K = int(config["objective"]["target_holdings"])
    # Calibrate A from the unconstrained objective. Violating cardinality by
    # one then costs comfortably more than any single-variable improvement.
    base = (
        float(config["objective"]["downside_risk_weight"])
        * covariance
        / (K * K)
        - np.diag(
            float(config["objective"]["expected_return_weight"])
            * expected
            / K
        )
    )
    coefficient_scale = max(
        float(np.max(np.abs(base).sum(axis=1))), 1e-6
    )
    cardinality_penalty = (
        float(config["objective"]["cardinality_penalty_multiplier"])
        * coefficient_scale
    )
    Q = build_weighted_cardinality_qubo(
        expected,
        covariance,
        target_holdings=K,
        lambda_return=float(config["objective"]["expected_return_weight"]),
        lambda_downside=float(config["objective"]["downside_risk_weight"]),
        cardinality_penalty=cardinality_penalty,
        previous_selection=previous_selection,
        lambda_turnover=float(config["objective"]["turnover_penalty"]),
    )
    h, J, offset = qubo_to_ising(Q)
    finalist_map = {item["code"]: item for item in inputs.finalists}
    rows = []
    for symbol in returns.columns:
        if symbol in finalist_map:
            item = finalist_map[symbol]
            rows.append(
                {
                    "qubit": len(rows),
                    "code": symbol,
                    "name": item["name"],
                    "market": item["market"],
                    "sector": item["sector"],
                    "asset_type": "stock",
                    "direction": "long",
                }
            )
        else:
            rows.append(
                {
                    "qubit": len(rows),
                    "code": symbol,
                    "name": {
                        "BTC": "Bitcoin",
                        "CL": "Crude Oil",
                        "XAU": "Gold",
                    }[symbol],
                    "market": "GLOBAL",
                    "sector": "Alternative",
                    "asset_type": "alternative",
                    "direction": "long",
                }
            )
    variables = pd.DataFrame(rows)
    variables["raw_sample_annual_return"] = raw_expected
    variables["expected_annual_return"] = expected
    variables["downside_deviation"] = np.sqrt(np.diag(covariance))
    meta = {
        **inputs.audit,
        "n": n,
        "target_holdings": K,
        "cardinality_penalty": cardinality_penalty,
        "bit_semantics": {"0": "asset_not_held", "1": "asset_long"},
        "short_selling": False,
        "risk_definition": "uncentered_downside_semivariance_ledoit_wolf",
        "expected_return_estimator": "sample_mean_shrunk_toward_zero",
        "expected_return_reliability": reliability,
        "expected_return_prior_observations": prior_observations,
        "full_investment_after_selection": True,
    }
    return {
        "Q": Q,
        "h": h,
        "J": J,
        "offset": offset,
        "covariance": covariance,
        "variables": variables,
        "meta": meta,
    }


def allocate_full_investment(
    selected: np.ndarray,
    variables: pd.DataFrame,
    covariance: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    selected = np.asarray(selected, dtype=np.int64)
    active = np.flatnonzero(selected)
    K = int(config["objective"]["target_holdings"])
    if len(active) != K:
        raise ValueError(f"selected {len(active)} assets, expected exactly {K}")
    table = variables.iloc[active]
    mu = table["expected_annual_return"].to_numpy(dtype=np.float64)
    cov = covariance[np.ix_(active, active)]
    alloc = config["allocation"]
    caps = np.where(
        table["asset_type"].eq("stock").to_numpy(),
        float(alloc["max_stock_weight"]),
        float(alloc["max_alternative_weight"]),
    )
    minimum = float(alloc["min_selected_weight"])
    target_sum = float(alloc["full_investment"])
    bounds = [(minimum, float(cap)) for cap in caps]
    risk_weight = float(config["objective"]["downside_risk_weight"])
    return_weight = float(config["objective"]["expected_return_weight"])

    def loss(weights: np.ndarray) -> float:
        return float(
            risk_weight * weights @ cov @ weights
            - return_weight * mu @ weights
        )

    result = minimize(
        loss,
        np.full(len(active), target_sum / len(active)),
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - target_sum},
        ],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"full-investment allocation failed: {result.message}")
    weights = np.zeros(len(selected), dtype=np.float64)
    weights[active] = result.x
    return weights, {
        "gross_exposure": float(weights.sum()),
        "net_exposure": float(weights.sum()),
        "cash_weight": float(1.0 - weights.sum()),
        "objective": float(result.fun),
        "selected_count": int(len(active)),
    }


def prepare_design_problem(config: dict[str, Any]) -> dict[str, Any]:
    inputs = load_design_inputs(config)
    result = build_design_problem(inputs, config)
    selected, energy = solve_sa(
        result["Q"],
        budget_s=float(config["qaoa"]["sa_budget_seconds"]),
        seed=int(config["qaoa"]["seed"]),
        n_restarts=8,
    )
    result["ranking"] = inputs.ranking
    result["sa_selection"] = selected
    result["sa_energy"] = float(energy)
    result["sa_cardinality"] = int(selected.sum())
    if result["sa_cardinality"] == int(config["objective"]["target_holdings"]):
        weights, diagnostics = allocate_full_investment(
            selected,
            result["variables"],
            result["covariance"],
            config,
        )
        result["sa_weights"] = weights
        result["sa_allocation"] = diagnostics
    return result
