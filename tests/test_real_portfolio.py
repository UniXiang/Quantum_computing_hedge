"""Tests for the real variable-cardinality portfolio pipeline."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from real_portfolio import (
    MarketInputs,
    _contract_prices,
    allocate_selected_weights,
    build_real_qubo,
    load_config,
    load_market_inputs,
    price_factor_table,
)
from solvers import solve_sa


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "portfolio_default.yaml")
CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "bs_cache_1year")


def test_contract_prices_uses_only_confirmed_past_rows(tmp_path):
    path = tmp_path / "contract.csv"
    pd.DataFrame({
        "datetime": [
            "2026-01-01 16:00:00",
            "2026-01-02 16:00:00",
            "2026-01-03 16:00:00",
            "2026-01-04 16:00:00",
        ],
        "close": [100.0, 101.0, 999.0, 103.0],
        "confirm": [1, 1, 0, 1],
    }).to_csv(path, index=False)
    prices = _contract_prices(path, "2026-01-03")
    assert prices.index.tolist() == ["2026-01-01", "2026-01-02"]
    assert prices.tolist() == [100.0, 101.0]


def test_price_factor_table_is_finite_and_cross_sectional():
    rng = np.random.default_rng(8)
    columns = [f"{i:06d}" for i in range(18)] + ["BTC", "CL", "XAU"]
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(80, 21)), columns=columns)
    inputs = MarketInputs(
        base_returns=returns,
        benchmark_returns=pd.Series(
            rng.normal(0.0, 0.01, 80), index=returns.index),
        stock_liquidity=pd.Series(
            np.linspace(1e8, 2e8, 18), index=columns[:18]),
        as_of="2026-01-01",
        preferred_observations=120,
    )
    config = load_config(CONFIG_PATH)
    config["universe"]["candidates"] = [
        {"code": code, "name": code} for code in columns[:18]]
    factors = price_factor_table(inputs, config)
    assert factors.shape == (21, 6)
    assert np.isfinite(factors.to_numpy()).all()
    np.testing.assert_allclose(
        factors["expected_annual_return_proxy"],
        factors["factor_score"]
        * config["objective"]["signal_annual_return_scale"],
    )


@pytest.mark.skipif(
    not os.path.isdir(CACHE_DIR), reason="real stock cache unavailable")
def test_real_n24_qubo_and_allocation_constraints():
    config = load_config(CONFIG_PATH)
    inputs = load_market_inputs(config)
    Q, variables, covariance, meta = build_real_qubo(inputs, config)
    assert Q.shape == (24, 24)
    assert np.allclose(Q, Q.T)
    assert len(variables) == 24
    assert meta["common_observations"] >= 60
    assert meta["benchmark_down_observations"] >= 20

    selected, _ = solve_sa(Q, budget_s=0.5, seed=42, n_restarts=8)
    for long_index, short_index in [(18, 19), (20, 21), (22, 23)]:
        assert not (selected[long_index] and selected[short_index])
    weights, diagnostics = allocate_selected_weights(
        selected, variables, covariance, config)
    assert diagnostics["gross_exposure"] <= 1.30 + 1e-8
    assert diagnostics["net_exposure"] >= 0.40 - 1e-8
    assert diagnostics["net_exposure"] <= 1.00 + 1e-8
    assert np.all(weights[:18] >= -1e-12)
