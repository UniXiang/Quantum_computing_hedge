"""Tests for full-universe US preselection and signed n=24 encoding."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_n24_portfolio import (
    build_us_n24_qubo,
    load_config,
    preselect_us_stocks,
)
from real_portfolio import MarketInputs


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "portfolio_us_n24.yaml"
)


def _synthetic_market() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=130)
    rows = []
    rng = np.random.default_rng(19)
    for number in range(15):
        code = f"S{number:02d}"
        prices = 50.0 * np.cumprod(
            1.0 + rng.normal(0.0005 + number * 0.00005, 0.01, len(dates))
        )
        for date, close in zip(dates, prices):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "asset_type": "stock",
                    "close": close,
                    "volume": 2_000_000 + number * 100_000,
                    "sector": f"sector-{number % 6}",
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_preselection_is_nine_stocks_and_obeys_sector_cap():
    config = load_config(CONFIG_PATH)
    config["data"]["as_of"] = "2025-07-02"
    config["universe"]["stock_pool_size"] = 15
    result = preselect_us_stocks(config, market_frame=_synthetic_market())
    assert len(result.candidates) == 9
    sectors = pd.Series([item["sector"] for item in result.candidates])
    assert sectors.value_counts().max() <= 3
    assert result.stats["source_symbols_through_as_of"] == 15
    assert result.stock_returns.shape[1] == 9


def test_n24_layout_has_signed_mutually_exclusive_pairs():
    config = load_config(CONFIG_PATH)
    config["data"]["as_of"] = "2025-07-02"
    config["universe"]["stock_pool_size"] = 15
    preselection = preselect_us_stocks(
        config, market_frame=_synthetic_market()
    )
    rng = np.random.default_rng(23)
    columns = [item["code"] for item in preselection.candidates]
    columns += ["BTC", "CL", "XAU"]
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.012, size=(80, 12)),
        columns=columns,
        index=pd.bdate_range("2025-03-03", periods=80),
    )
    inputs = MarketInputs(
        base_returns=returns,
        benchmark_returns=pd.Series(
            rng.normal(0.0, 0.01, 80), index=returns.index
        ),
        stock_liquidity=preselection.liquidity,
        as_of="2025-07-02",
        preferred_observations=120,
    )
    Q, variables, covariance, meta = build_us_n24_qubo(
        inputs, preselection, config
    )
    assert Q.shape == (24, 24)
    assert covariance.shape == (24, 24)
    assert np.allclose(Q, Q.T)
    assert variables["asset_type"].value_counts().to_dict() == {
        "stock": 18,
        "hedge": 6,
    }
    for left, right in meta["mutually_exclusive_pairs"]:
        assert variables.iloc[left]["underlying"] == variables.iloc[right][
            "underlying"
        ]
        assert {
            variables.iloc[left]["direction"],
            variables.iloc[right]["direction"],
        } == {"long", "short"}
