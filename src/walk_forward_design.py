"""Cross-market point-in-time walk-forward validation for the n=24 model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_long_n24 import (
    _candidate_table,
    allocate_full_investment,
    build_design_problem,
    load_design_inputs,
)
from mixed_long_n24 import _cache_path
from real_portfolio import load_config
from solvers import solve_sa


def _stock_open_series(
    cache_dir: Path, market: str, code: str
) -> pd.Series:
    frame = pd.read_pickle(_cache_path(cache_dir, market, code)).copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    opens = pd.to_numeric(frame["open"], errors="coerce")
    status = pd.to_numeric(
        frame["tradestatus"], errors="coerce"
    ).fillna(0)
    series = pd.Series(opens.to_numpy(), index=dates, name=code)
    series = series.loc[status.to_numpy() > 0].dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _alternative_open_series(path: str | Path, symbol: str) -> pd.Series:
    frame = pd.read_csv(path)
    if "confirm" in frame:
        confirmed = pd.to_numeric(frame["confirm"], errors="coerce")
        frame = frame.loc[confirmed.eq(1)].copy()
    date_column = "date" if "date" in frame else "datetime"
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    opens = pd.to_numeric(frame["open"], errors="coerce")
    series = pd.Series(opens.to_numpy(), index=dates, name=symbol).dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _execution_panels(config: dict[str, Any]):
    cache_dir = Path(config["data"]["cache_dir"])
    candidates = _candidate_table(config)
    opens = {}
    market_dates = {}
    for item in candidates:
        series = _stock_open_series(
            cache_dir, item["market"], item["code"]
        )
        opens[item["code"]] = series
        market_dates.setdefault(item["market"], []).append(series.index)
    for symbol, path in config["data"]["contract_files"].items():
        opens[str(symbol)] = _alternative_open_series(path, str(symbol))
    # A decision date must offer an executable open in both stock markets.
    a_dates = market_dates["A"][0]
    for dates in market_dates["A"][1:]:
        a_dates = a_dates.intersection(dates)
    us_dates = market_dates["US"][0]
    for dates in market_dates["US"][1:]:
        us_dates = us_dates.intersection(dates)
    common = a_dates.intersection(us_dates).sort_values()
    return opens, common


def _open_to_open_return(
    series: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    if start not in series.index or end not in series.index:
        # Alternatives trade every day; stocks are guaranteed by common
        # calendar. Missing execution prices are never silently filled.
        raise ValueError(
            f"{series.name}: missing execution open for "
            f"{start.date()} or {end.date()}"
        )
    return float(series.loc[end] / series.loc[start] - 1.0)


def run_walk_forward(config: dict[str, Any]) -> dict[str, Any]:
    opens, common_dates = _execution_panels(config)
    backtest = config["backtest"]
    start = pd.Timestamp(backtest["start"])
    end = pd.Timestamp(backtest["end"])
    decisions = common_dates[
        (common_dates >= start) & (common_dates <= end)
    ]
    if len(decisions) < 2:
        raise ValueError("need at least two common execution dates")

    nav = float(backtest["initial_nav"])
    benchmark_nav = nav
    previous_weights: dict[str, float] = {}
    previous_codes: set[str] = set()
    records = []
    for position in range(len(decisions) - 1):
        decision_date = pd.Timestamp(decisions[position])
        next_date = pd.Timestamp(decisions[position + 1])
        # At 08:00 Beijing, only session labels through the previous
        # natural day are admitted. This is conservative for both markets.
        cutoff = decision_date - pd.Timedelta(days=1)
        daily_config = copy.deepcopy(config)
        daily_config["data"]["as_of"] = str(cutoff.date())
        daily_config["data"]["decision_time"] = (
            f"{decision_date.date()}T"
            f"{backtest['decision_time_local']}"
        )
        inputs = load_design_inputs(daily_config)
        current_columns = list(inputs.returns.columns)
        previous_selection = np.asarray(
            [int(code in previous_codes) for code in current_columns],
            dtype=np.int64,
        )
        problem = build_design_problem(
            inputs,
            daily_config,
            previous_selection=previous_selection,
        )
        selected, qubo_energy = solve_sa(
            problem["Q"],
            budget_s=float(config["qaoa"]["sa_budget_seconds"]),
            seed=int(config["qaoa"]["seed"]) + position,
            n_restarts=8,
        )
        if int(selected.sum()) != int(
            config["objective"]["target_holdings"]
        ):
            raise RuntimeError(
                f"{decision_date.date()}: SA selected "
                f"{int(selected.sum())} assets"
            )
        weights, allocation = allocate_full_investment(
            selected,
            problem["variables"],
            problem["covariance"],
            daily_config,
        )
        current_weights = {
            str(problem["variables"].iloc[index]["code"]): float(weights[index])
            for index in np.flatnonzero(selected)
        }
        all_codes = set(previous_weights) | set(current_weights)
        traded_notional = sum(
            abs(
                current_weights.get(code, 0.0)
                - previous_weights.get(code, 0.0)
            )
            for code in all_codes
        )
        cost = (
            traded_notional
            * float(backtest["transaction_cost_bps"])
            / 10_000.0
        )
        asset_returns = {
            code: _open_to_open_return(
                opens[code], decision_date, next_date
            )
            for code in current_weights
        }
        gross_return = sum(
            current_weights[code] * asset_returns[code]
            for code in current_weights
        )
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0
        nav *= 1.0 + net_return

        stock_codes = [
            item["code"] for item in _candidate_table(config)
        ]
        benchmark_returns = [
            _open_to_open_return(
                opens[code], decision_date, next_date
            )
            for code in stock_codes
        ]
        benchmark_return = float(np.mean(benchmark_returns))
        benchmark_nav *= 1.0 + benchmark_return
        records.append(
            {
                "decision_time": daily_config["data"]["decision_time"],
                "data_cutoff": str(cutoff.date()),
                "execution_open": str(decision_date.date()),
                "next_execution_open": str(next_date.date()),
                "selected_codes": sorted(current_weights),
                "weights": current_weights,
                "asset_returns": asset_returns,
                "qubo_energy": float(qubo_energy),
                "allocation_objective": allocation["objective"],
                "traded_notional": traded_notional,
                "transaction_cost": cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "nav": nav,
                "benchmark_return": benchmark_return,
                "benchmark_nav": benchmark_nav,
            }
        )
        previous_weights = current_weights
        previous_codes = set(current_weights)

    returns = np.asarray([row["net_return"] for row in records])
    benchmark_returns = np.asarray(
        [row["benchmark_return"] for row in records]
    )
    periods = len(returns)
    annualization = 252.0
    downside = np.sqrt(
        np.mean(np.minimum(returns, 0.0) ** 2) * annualization
    )
    nav_path = np.asarray([1.0] + [row["nav"] for row in records])
    running_peak = np.maximum.accumulate(nav_path)
    maximum_drawdown = float(np.min(nav_path / running_peak - 1.0))
    metrics = {
        "periods": periods,
        "total_return": float(nav - 1.0),
        "benchmark_total_return": float(benchmark_nav - 1.0),
        "excess_total_return": float(nav - benchmark_nav),
        "annualized_return": float(
            np.prod(1.0 + returns) ** (annualization / periods) - 1.0
        ),
        "annualized_volatility": float(
            np.std(returns, ddof=1) * np.sqrt(annualization)
        ),
        "downside_deviation": float(downside),
        "maximum_drawdown": maximum_drawdown,
        "average_traded_notional": float(
            np.mean([row["traded_notional"] for row in records])
        ),
        "total_transaction_cost": float(
            np.sum([row["transaction_cost"] for row in records])
        ),
        "positive_period_ratio": float(np.mean(returns > 0.0)),
        "correlation_to_benchmark": float(
            np.corrcoef(returns, benchmark_returns)[0, 1]
        ),
    }
    return {
        "method": "daily_cross_market_walk_forward_sa",
        "timing": {
            "decision": "08:00 Asia/Shanghai",
            "data_available_through": "previous natural day",
            "execution": "current common A/US session open",
            "holding_return": "current common open to next common open",
        },
        "config": config,
        "metrics": metrics,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_walk_forward(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
