"""Walk-forward, price-direction backtest for the quantum hedge strategy.

The framework deliberately models only the scope confirmed in
``portfolio_default.yaml``: a close-to-close return series, weekly signals,
next-session execution, and no trading or derivative implementation costs.
It does *not* turn an XAU-USDT price series into a tradeable futures PnL.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import _load_one, _normalize_code
from real_portfolio import (
    _code,
    _contract_prices,
    _hedge_assets,
    load_config,
    solve_real_portfolio,
)


@dataclass
class BacktestResult:
    """Auditable output of one walk-forward run."""

    daily: pd.DataFrame
    rebalances: pd.DataFrame
    metrics: pd.DataFrame
    effective_end_date: str


def effective_end_date(config: dict[str, Any]) -> str:
    """Last A-share benchmark session covered by every configured contract."""
    requested = str(config["data"]["as_of"])
    ends = [requested]
    for symbol, path in config["data"].get("contract_files", {}).items():
        prices = _contract_prices(path, requested)
        if prices.empty:
            raise ValueError(f"{symbol}: no contract close on or before {requested}")
        ends.append(str(prices.index[-1]))
    contract_limited_end = min(ends)
    benchmark = _normalize_code(str(config["data"]["benchmark_code"]))
    benchmark_returns = _load_one(benchmark, str(config["data"]["stock_cache_dir"]))
    sessions = benchmark_returns.loc[benchmark_returns.index <= contract_limited_end]
    if sessions.empty:
        raise ValueError(
            "no benchmark session is jointly available with the contract data")
    return str(sessions.index[-1])


def _history(config: dict[str, Any], end_date: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return asset and benchmark returns on observed A-share benchmark days.

    US closes are shifted by one A-share session, matching the signal path.
    Contract prices must be observed through ``end_date``; unlike the signal
    loader, this function never extends a stale final contract close into the
    reported performance period.
    """
    data = config["data"]
    cache_dir = str(data["stock_cache_dir"])
    benchmark = _normalize_code(str(data["benchmark_code"]))
    benchmark_returns = _load_one(benchmark, cache_dir)
    benchmark_returns = benchmark_returns.loc[benchmark_returns.index <= end_date]
    calendar = benchmark_returns.index
    if len(calendar) < 2:
        raise ValueError("backtest needs at least two benchmark sessions")

    candidates = config["universe"]["candidates"]
    hedges = _hedge_assets(config)
    equity_items = candidates + [item for item in hedges if item["kind"] == "equity"]
    equity_codes = [_code(item) for item in equity_items]
    mainland = [code for code in equity_codes if not code.startswith("us_")]
    us = [code for code in equity_codes if code.startswith("us_")]
    base = pd.DataFrame(index=calendar)
    for code in mainland:
        base[code] = _load_one(code, cache_dir).reindex(calendar).fillna(0.0)
    for code in us:
        closes_known_at_signal = _load_one(code, cache_dir).reindex(calendar).fillna(0.0)
        base[code] = closes_known_at_signal.shift(1).fillna(0.0)

    for item in hedges:
        if item["kind"] != "contract":
            continue
        symbol = _code(item)
        path = data["contract_files"][symbol]
        prices = _contract_prices(path, end_date)
        if prices.empty or str(prices.index[-1]) < end_date:
            raise ValueError(
                f"{symbol}: prices end at "
                f"{None if prices.empty else prices.index[-1]}, before {end_date}")
        aligned = prices.reindex(calendar).ffill(limit=5)
        base[symbol] = aligned.pct_change(fill_method=None)
    return base, benchmark_returns.reindex(calendar)


def weekly_signal_dates(calendar: pd.Index) -> list[str]:
    """Last available A-share session of every Monday-to-Friday week."""
    dates = pd.DatetimeIndex(pd.to_datetime(calendar))
    frame = pd.DataFrame({"date": dates}, index=dates.strftime("%Y-%m-%d"))
    frame["week"] = dates.to_period("W-FRI")
    return [str(value.date()) for value in frame.groupby("week")["date"].max()]


def _periods(calendar: pd.Index) -> list[tuple[str, str, str]]:
    """Return (signal, execution-start, holding-end) tuples without overlap."""
    ordered = [str(value) for value in calendar]
    positions = {date: index for index, date in enumerate(ordered)}
    signals = weekly_signal_dates(calendar)
    result: list[tuple[str, str, str]] = []
    for index, signal in enumerate(signals):
        signal_position = positions[signal]
        execution_position = signal_position + 1
        if execution_position >= len(ordered):
            continue
        end = signals[index + 1] if index + 1 < len(signals) else ordered[-1]
        if execution_position <= positions[end]:
            result.append((signal, ordered[execution_position], end))
    return result


def _performance_metrics(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Calculate a compact institutional-style, zero-risk-free scorecard."""
    if returns.empty:
        return {name: float("nan") for name in (
            "cumulative_return", "annualized_return", "annualized_volatility",
            "sharpe_zero_rf", "sortino_zero_rf", "calmar", "max_drawdown",
            "beta_to_csi300", "annualized_alpha_zero_rf", "tracking_error",
            "information_ratio", "daily_win_rate", "daily_var_95", "daily_cvar_95")}
    returns = returns.astype(float)
    benchmark = benchmark.reindex(returns.index).astype(float)
    nav = (1.0 + returns).cumprod()
    cumulative = float(nav.iloc[-1] - 1.0)
    annualized = float(nav.iloc[-1] ** (252.0 / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() * 252.0 / volatility) if volatility > 0.0 else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    downside_deviation = float(np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))
                               * np.sqrt(252.0))
    sortino = (float(returns.mean() * 252.0 / downside_deviation)
               if downside_deviation > 0.0 else float("nan"))
    calmar = (float(annualized / abs(max_drawdown))
              if max_drawdown < 0.0 else float("nan"))
    variance = float(benchmark.var(ddof=1)) if len(benchmark) > 1 else 0.0
    beta = (float(returns.cov(benchmark) / variance)
            if variance > 0.0 else float("nan"))
    active = returns - benchmark
    tracking_error = (float(active.std(ddof=1) * np.sqrt(252.0))
                      if len(active) > 1 else 0.0)
    information_ratio = (float(active.mean() * 252.0 / tracking_error)
                         if tracking_error > 0.0 else float("nan"))
    alpha = (float((returns.mean() - beta * benchmark.mean()) * 252.0)
             if np.isfinite(beta) else float("nan"))
    var_95 = float(np.quantile(returns, 0.05))
    tail = returns.loc[returns <= var_95]
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_zero_rf": sharpe,
        "sortino_zero_rf": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "beta_to_csi300": beta,
        "annualized_alpha_zero_rf": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "daily_win_rate": float((returns > 0.0).mean()),
        "daily_var_95": var_95,
        "daily_cvar_95": float(tail.mean()),
    }


def run_backtest(config: dict[str, Any]) -> BacktestResult:
    """Run the configured weekly walk-forward SA backtest.

    Each QUBO sees only prices through its signal close.  The selected target
    is then allocated with a one-way turnover cap
    ``0.5 * sum(abs(w_t - w_{t-1})) <= configured cap``.  Positions removed
    from a target may be carried temporarily (and only reduced) when needed
    to satisfy that cap.
    """
    end_date = effective_end_date(config)
    returns, benchmark = _history(config, end_date)
    candidates = [_code(item) for item in config["universe"]["candidates"]]
    previous_selection: np.ndarray | None = None
    previous_weights = np.zeros(
        int(config["selection"]["qubits"]), dtype=np.float64)
    daily_parts: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, Any]] = []

    for signal_date, execution_date, holding_end in _periods(returns.index):
        signal_config = deepcopy(config)
        signal_config["data"]["as_of"] = signal_date
        try:
            decision = solve_real_portfolio(
                signal_config,
                previous_selection=previous_selection,
                previous_weights=previous_weights,
                enforce_turnover_cap=True,
            )
        except ValueError as error:
            # Before the configured min-common/min-downside history exists,
            # there is deliberately no strategy exposure rather than a
            # shortened, undocumented model.
            if "observations" in str(error) or "history" in str(error):
                continue
            raise

        variables = decision["variables"]
        weights = variables["weight"].to_numpy(dtype=np.float64)
        underlying = variables["underlying"].tolist()
        period_index = returns.loc[execution_date:holding_end].index
        underlying_returns = returns.loc[period_index, underlying].to_numpy(dtype=np.float64)
        portfolio_returns = underlying_returns @ weights
        candidate_returns = returns.loc[period_index, candidates].mean(axis=1)
        period = pd.DataFrame({
            "strategy_return": portfolio_returns,
            "csi300_return": benchmark.loc[period_index].to_numpy(dtype=np.float64),
            "candidate_equal_weight_return": candidate_returns.to_numpy(dtype=np.float64),
            "signal_date": signal_date,
        }, index=period_index)
        daily_parts.append(period)

        selected_names = variables.loc[variables["selected"], "variable"].tolist()
        executed = variables.loc[np.abs(weights) > 1e-12, ["variable", "weight"]]
        rebalance_rows.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "holding_end": holding_end,
            "common_observations": decision["meta"]["common_observations"],
            "benchmark_down_observations": decision["meta"]["benchmark_down_observations"],
            "qubo_energy": decision["qubo_energy"],
            "selection_turnover_penalty": decision["objective_terms"]["selection_turnover"],
            "executed_one_way_turnover": decision["allocation"]["turnover"],
            "gross_exposure": decision["allocation"]["gross_exposure"],
            "net_exposure": decision["allocation"]["net_exposure"],
            "portfolio_beta": decision["allocation"]["portfolio_beta"],
            "selected_targets": ",".join(selected_names),
            "executed_weights": ",".join(
                f"{row.variable}:{row.weight:.6f}" for row in executed.itertuples()),
        })
        previous_selection = variables["selected"].to_numpy(dtype=np.int64)
        previous_weights = weights

    if not daily_parts:
        raise RuntimeError("no eligible rebalance dates; check history requirements")
    daily = pd.concat(daily_parts).sort_index()
    for column in ("strategy_return", "csi300_return", "candidate_equal_weight_return"):
        daily[column.replace("return", "nav")] = (1.0 + daily[column]).cumprod()
    metrics = pd.DataFrame.from_dict({
        "strategy": _performance_metrics(daily["strategy_return"], daily["csi300_return"]),
        "CSI300": _performance_metrics(daily["csi300_return"], daily["csi300_return"]),
        "candidate_equal_weight": _performance_metrics(
            daily["candidate_equal_weight_return"], daily["csi300_return"]),
    }, orient="index")
    return BacktestResult(
        daily=daily,
        rebalances=pd.DataFrame(rebalance_rows),
        metrics=metrics,
        effective_end_date=end_date,
    )


def _format_metric(value: float, kind: str) -> str:
    if not np.isfinite(value):
        return "N/A"
    if kind == "percent":
        return f"{value:.2%}"
    return f"{value:.3f}"


def _metrics_markdown(metrics: pd.DataFrame) -> str:
    rows = [
        ("Cumulative return", "cumulative_return", "percent"),
        ("Annualized return", "annualized_return", "percent"),
        ("Annualized volatility", "annualized_volatility", "percent"),
        ("Sharpe (rf=0)", "sharpe_zero_rf", "number"),
        ("Sortino (rf=0)", "sortino_zero_rf", "number"),
        ("Calmar", "calmar", "number"),
        ("Maximum drawdown", "max_drawdown", "percent"),
        ("Beta to CSI300", "beta_to_csi300", "number"),
        ("Annualized CAPM alpha (rf=0)", "annualized_alpha_zero_rf", "percent"),
        ("Tracking error", "tracking_error", "percent"),
        ("Information ratio", "information_ratio", "number"),
        ("Daily win rate", "daily_win_rate", "percent"),
        ("Daily VaR 95%", "daily_var_95", "percent"),
        ("Daily CVaR 95%", "daily_cvar_95", "percent"),
    ]
    header = "| Metric | Strategy | CSI300 | Candidate equal weight |\n|---|---:|---:|---:|"
    body = []
    for label, column, kind in rows:
        body.append("| " + label + " | " + " | ".join(
            _format_metric(float(metrics.loc[name, column]), kind)
            for name in ("strategy", "CSI300", "candidate_equal_weight")) + " |")
    return "\n".join([header, *body])


def _plot_performance(result: BacktestResult, path: Path) -> None:
    """Write a two-panel NAV and drawdown chart without a GUI dependency."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum_hedge_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    daily = result.daily
    dates = pd.to_datetime(daily.index)
    series = (
        ("strategy_nav", "Strategy", "#1f77b4"),
        ("csi300_nav", "CSI300", "#666666"),
        ("candidate_equal_weight_nav", "Candidate equal weight", "#ff7f0e"),
    )
    figure, (nav_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    for column, label, color in series:
        nav = daily[column].astype(float)
        nav_axis.plot(dates, nav, label=label, color=color, linewidth=1.8)
        drawdown = nav / nav.cummax() - 1.0
        drawdown_axis.plot(dates, drawdown, label=label, color=color, linewidth=1.3)
    nav_axis.set_title(
        f"Walk-forward backtest: NAV and drawdown (through {result.effective_end_date})")
    nav_axis.set_ylabel("Net asset value")
    nav_axis.grid(alpha=0.25)
    nav_axis.legend(loc="best")
    drawdown_axis.axhline(0.0, color="black", linewidth=0.8)
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_axis.set_xlabel("Date")
    drawdown_axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    drawdown_axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _professional_assessment(result: BacktestResult) -> str:
    """Return an evidence-led interpretation, not a claim of significance."""
    daily = result.daily
    strategy = result.metrics.loc["strategy"]
    benchmark = result.metrics.loc["CSI300"]
    candidates = result.metrics.loc["candidate_equal_weight"]
    relative = float(daily["strategy_nav"].iloc[-1] / daily["csi300_nav"].iloc[-1] - 1.0)
    candidate_relative = float(
        daily["strategy_nav"].iloc[-1] / daily["candidate_equal_weight_nav"].iloc[-1] - 1.0)
    average_turnover = float(result.rebalances["executed_one_way_turnover"].mean())
    max_turnover = float(result.rebalances["executed_one_way_turnover"].max())
    return "\n".join([
        "## Professional assessment",
        "",
        f"- **Relative performance:** strategy {'outperformed' if relative >= 0 else 'underperformed'} "
        f"CSI300 by {_format_metric(relative, 'percent')} cumulatively. It "
        f"{'underperformed' if candidate_relative < 0 else 'outperformed'} the candidate equal-weight "
        f"baseline by {_format_metric(abs(candidate_relative), 'percent')}; this is the more demanding "
        "stock-selection comparison.",
        f"- **Risk and market exposure:** realised beta was {_format_metric(float(strategy['beta_to_csi300']), 'number')} "
        f"versus the configured 0.60 target. Strategy maximum drawdown was "
        f"{_format_metric(float(strategy['max_drawdown']), 'percent')}, compared with "
        f"{_format_metric(float(benchmark['max_drawdown']), 'percent')} for CSI300 and "
        f"{_format_metric(float(candidates['max_drawdown']), 'percent')} for candidate equal weight.",
        f"- **Risk-adjusted return:** zero-risk-free Sharpe was "
        f"{_format_metric(float(strategy['sharpe_zero_rf']), 'number')}; information ratio versus CSI300 was "
        f"{_format_metric(float(strategy['information_ratio']), 'number')}. Annualized CAPM alpha is a "
        f"descriptive {_format_metric(float(strategy['annualized_alpha_zero_rf']), 'percent')}, not a significance-tested claim.",
        f"- **Implementation discipline:** {len(result.rebalances)} rebalances; mean one-way turnover "
        f"{_format_metric(average_turnover, 'percent')}; maximum "
        f"{_format_metric(max_turnover, 'percent')}. The configured 20% cap is "
        f"{'respected' if max_turnover <= 0.20 + 1e-8 else 'breached'}.",
        "- **Interpretation limit:** this is a short, price-direction-only experiment. It excludes "
        "commissions, slippage, funding, margin, liquidation, contract multipliers, basis, and roll. "
        "It is not evidence of implementable live net performance or statistical robustness.",
    ])


def _write_report(result: BacktestResult, path: Path) -> None:
    daily = result.daily
    rebalance = result.rebalances
    report = "\n".join([
        "# Walk-forward backtest report",
        "",
        f"- **Performance period:** {daily.index[0]} to {daily.index[-1]} "
        f"({len(daily)} trading days)",
        f"- **Effective data end date:** {result.effective_end_date}",
        f"- **Rebalances:** {len(rebalance)}",
        f"- **Training observations per rebalance:** "
        f"{int(rebalance['common_observations'].min())}–{int(rebalance['common_observations'].max())}",
        "",
        "![NAV and drawdown](performance.png)",
        "",
        "## Performance scorecard",
        "",
        _metrics_markdown(result.metrics),
        "",
        _professional_assessment(result),
        "",
        "## Methodology boundary",
        "",
        "Signals are formed at the weekly A-share close and take effect on the next A-share "
        "session. Every QUBO uses only data available on its signal date; US returns are delayed "
        "one A-share session. Results use the configured local simulated annealing solver and a "
        "20% one-way turnover cap. No implementation costs or derivative mechanics are modelled.",
        "",
        "Artifacts: `daily_returns.csv`, `rebalances.csv`, `metrics.csv`, and `performance.png`.",
    ])
    path.write_text(report + "\n", encoding="utf-8")


def save_backtest(result: BacktestResult, output_dir: str | Path) -> None:
    """Persist raw data plus a chart and professional evaluation on every run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(directory / "daily_returns.csv", index_label="date")
    result.rebalances.to_csv(directory / "rebalances.csv", index=False)
    result.metrics.to_csv(directory / "metrics.csv", index_label="portfolio")
    _plot_performance(result, directory / "performance.png")
    _write_report(result, directory / "report.md")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local-SA walk-forward backtest")
    parser.add_argument("--config", default="configs/portfolio_default.yaml")
    parser.add_argument("--output-dir", default="results/backtest_sa")
    args = parser.parse_args()
    result = run_backtest(load_config(args.config))
    save_backtest(result, args.output_dir)
    print(f"Effective end date: {result.effective_end_date}")
    print(result.metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print(_professional_assessment(result))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
