"""Unit tests for walk-forward scheduling and hard turnover execution."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import (
    BacktestResult,
    _periods,
    _performance_metrics,
    save_backtest,
    weekly_signal_dates,
)
from real_portfolio import allocate_selected_weights, load_config


def test_weekly_schedule_uses_last_session_and_next_session_execution():
    calendar = pd.Index([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14",
        "2026-01-15", "2026-01-16",
    ])
    assert weekly_signal_dates(calendar) == ["2026-01-09", "2026-01-16"]
    assert _periods(calendar) == [("2026-01-09", "2026-01-12", "2026-01-16")]


def test_metrics_uses_close_to_close_compounding():
    returns = pd.Series([0.10, -0.05])
    benchmark = pd.Series([0.02, -0.01])
    metrics = _performance_metrics(returns, benchmark)
    assert metrics["cumulative_return"] == pytest.approx(0.045)
    assert metrics["max_drawdown"] == pytest.approx(-0.05)
    assert metrics["daily_var_95"] <= 0.0
    assert "information_ratio" in metrics


def test_save_backtest_writes_chart_and_professional_report(tmp_path):
    index = pd.Index(["2026-01-05", "2026-01-06", "2026-01-07"])
    daily = pd.DataFrame({
        "strategy_return": [0.01, -0.02, 0.03],
        "csi300_return": [0.005, -0.01, 0.015],
        "candidate_equal_weight_return": [0.02, -0.01, 0.01],
        "signal_date": ["2026-01-02"] * 3,
    }, index=index)
    for column in ("strategy_return", "csi300_return", "candidate_equal_weight_return"):
        daily[column.replace("return", "nav")] = (1.0 + daily[column]).cumprod()
    metrics = pd.DataFrame.from_dict({
        "strategy": _performance_metrics(daily["strategy_return"], daily["csi300_return"]),
        "CSI300": _performance_metrics(daily["csi300_return"], daily["csi300_return"]),
        "candidate_equal_weight": _performance_metrics(
            daily["candidate_equal_weight_return"], daily["csi300_return"]),
    }, orient="index")
    result = BacktestResult(
        daily=daily,
        rebalances=pd.DataFrame({
            "common_observations": [60], "executed_one_way_turnover": [0.1],
        }),
        metrics=metrics,
        effective_end_date="2026-01-07",
    )
    save_backtest(result, tmp_path)
    assert (tmp_path / "performance.png").stat().st_size > 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Professional assessment" in report
    assert "Information ratio" in report


def test_turnover_cap_carries_old_weights_when_target_changes():
    config = load_config("configs/portfolio_default.yaml")
    n = 10
    variables = pd.DataFrame({
        "asset_type": ["equity"] * n,
        "sign": [1.0] * n,
        "expected_annual_return_proxy": [0.0] * n,
        "beta": [1.5] * n,
    })
    selected = np.array([0] * 5 + [1] * 5)
    previous = np.array([0.08] * 5 + [0.0] * 5)
    weights, diagnostics = allocate_selected_weights(
        selected, variables, np.eye(n) * 0.01, config,
        previous_weights=previous, max_turnover=0.20,
    )
    assert diagnostics["turnover"] <= 0.20 + 1e-8
    assert np.any(np.abs(weights[:5]) > 1e-12)  # residual carry, not a cap breach
    assert np.all(weights[5:] >= 0.005 - 1e-10)
