"""Tests for src/data_loader.py — real bs_cache_1year pkl loading.

Uses three representative stocks:
  - 600000 (SH large-cap blue chip, clean history)
  - 000001 (SZ main board, clean history)
  - 000008 (has a 5-day suspension in July 2026 with empty pctChg)
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_returns, CACHE_DIR

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE_DIR), reason="bs_cache_1year not available"
)

CODES = ["600000", "000001", "000008"]
END = "2026-07-24"  # last trading day in cache
WINDOW = 60


def test_load_returns_shape_and_index():
    r = load_returns(CODES, END, WINDOW)
    assert r.shape == (WINDOW, len(CODES))
    assert list(r.columns) == CODES
    # index is ascending trading-day dates, last one == end_date
    assert list(r.index) == sorted(r.index)
    assert str(r.index[-1]) == END
    assert r.index.is_unique
    # returns are small daily numbers, no NaN
    assert not r.isna().any().any()
    assert (r.abs() < 0.25).all().all()


def test_load_returns_matches_pctchg():
    r = load_returns(["600000"], END, WINDOW)
    raw = pd.read_pickle(os.path.join(CACHE_DIR, "sh_600000_1year.pkl"))
    raw = raw[raw["date"] <= END].tail(WINDOW)
    expected = pd.to_numeric(raw["pctChg"], errors="coerce").fillna(0.0) / 100.0
    np.testing.assert_allclose(r["600000"].values, expected.values, atol=1e-12)


def test_code_normalization():
    r1 = load_returns(["600000"], END, 10)
    r2 = load_returns(["sh.600000"], END, 10)
    r3 = load_returns(["sh_600000"], END, 10)
    assert list(r2.columns) == ["600000"]
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(r1, r3)


def test_us_cache_code_normalization_and_cutoff():
    """US cache files use us_TICKER labels and obey the same cutoff rule."""
    r1 = load_returns(["us_AAPL"], "2026-07-03", 10)
    r2 = load_returns(["us.AAPL"], "2026-07-03", 10)
    assert list(r1.columns) == ["us_AAPL"]
    assert str(r1.index[-1]) <= "2026-07-03"
    pd.testing.assert_frame_equal(r1, r2)


def test_no_future_data_end_date_shift():
    """Shifting end_date back one trading day must only roll the window:
    overlapping dates keep identical values, and no data after the new
    end_date may appear (no look-ahead)."""
    r_full = load_returns(CODES, END, WINDOW)
    end_early = "2026-07-23"  # previous trading day
    r_early = load_returns(CODES, end_early, WINDOW)
    assert str(r_early.index[-1]) == end_early
    assert (r_early.index <= end_early).all()
    # the earlier window is the full window shifted by one day
    np.testing.assert_allclose(
        r_early.values[1:], r_full.values[:-1], atol=1e-12
    )


def test_suspension_days_get_zero_return():
    """000008 was suspended 2026-07-07..2026-07-13 (tradestatus=0, empty
    pctChg): strategy is return = 0 on suspended/missing days."""
    r = load_returns(["000008"], END, WINDOW)
    for d in ["2026-07-07", "2026-07-08", "2026-07-09",
              "2026-07-10", "2026-07-13"]:
        assert d in r.index
        assert r.loc[d, "000008"] == 0.0


def test_missing_code_raises():
    with pytest.raises(FileNotFoundError):
        load_returns(["999999"], END, WINDOW)
