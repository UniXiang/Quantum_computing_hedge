import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from design_long_n24 import allocate_full_investment
from mixed_long_n24 import _read_cache_return
from qubo_builder import build_weighted_cardinality_qubo
from solvers import solve_exact


def test_weighted_cardinality_qubo_has_exact_target_count():
    rng = np.random.default_rng(123)
    n, K = 10, 4
    raw = rng.normal(size=(n, n))
    covariance = raw.T @ raw / 100.0
    expected = rng.normal(0.08, 0.03, size=n)
    base_scale = max(
        np.max(np.abs(covariance / (K * K)).sum(axis=1)),
        np.max(np.abs(expected / K)),
    )
    Q = build_weighted_cardinality_qubo(
        expected,
        covariance,
        target_holdings=K,
        lambda_return=0.35,
        lambda_downside=1.0,
        cardinality_penalty=20.0 * base_scale,
    )
    selected, _ = solve_exact(Q)
    assert int(selected.sum()) == K


def test_turnover_term_prefers_previous_selection_on_tie():
    n, K = 6, 3
    previous = np.array([1, 0, 1, 0, 1, 0])
    Q = build_weighted_cardinality_qubo(
        np.zeros(n),
        np.zeros((n, n)),
        target_holdings=K,
        cardinality_penalty=1.0,
        previous_selection=previous,
        lambda_turnover=0.1,
    )
    selected, _ = solve_exact(Q)
    np.testing.assert_array_equal(selected, previous)


def test_full_allocator_is_long_only_and_has_no_cash():
    variables = pd.DataFrame(
        {
            "asset_type": ["stock"] * 4,
            "expected_annual_return": [0.1, 0.08, 0.06, 0.04],
        }
    )
    selected = np.ones(4, dtype=int)
    covariance = np.eye(4) * 0.04
    config = {
        "objective": {
            "target_holdings": 4,
            "downside_risk_weight": 1.0,
            "expected_return_weight": 0.35,
        },
        "allocation": {
            "min_selected_weight": 0.02,
            "max_stock_weight": 0.4,
            "max_alternative_weight": 0.4,
            "full_investment": 1.0,
        },
    }
    weights, diagnostics = allocate_full_investment(
        selected, variables, covariance, config
    )
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["cash_weight"] == pytest.approx(0.0, abs=1e-10)


def test_cache_reader_hard_cuts_future_rows(tmp_path):
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": np.arange(70, dtype=float) + 100.0,
            "amount": np.full(70, 1_000_000.0),
            "tradestatus": np.ones(70),
            "pctChg": np.full(70, 0.1),
        }
    ).astype(str)
    # This future outlier must never affect a cutoff through day 65.
    frame.loc[69, "pctChg"] = "9999"
    path = tmp_path / "us_TEST_1year.pkl"
    frame.to_pickle(path)
    cutoff = dates[64]
    returns, _, audit = _read_cache_return(
        tmp_path, "US", "TEST", cutoff
    )
    assert returns.index.max() == cutoff
    assert float(returns.max()) < 1.0
    assert audit["end"] == str(cutoff.date())
