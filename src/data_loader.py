"""data_loader.py — load daily returns from the bs_cache_1year pkl cache.

Follows the conventions of /mnt/f/Gaming/load_1year_data.py:
  - mainland files are named ``{exchange}_{code6}_1year.pkl`` (for example
    ``sh_600000_1year.pkl``); US files are ``us_{ticker}_1year.pkl``
  - each pkl is a DataFrame with string columns:
    date, code (e.g. 'sh.600000'), open/high/low/close/preclose, volume,
    amount, turn, tradestatus, pctChg, isST
  - ``pctChg`` is the baostock daily percent change computed against
    ``preclose``, so it is dividend/ex-rights adjusted; daily return =
    pctChg / 100.

Missing / suspension handling
-----------------------------
Suspended days appear in the pkl with ``tradestatus='0'`` and empty
``pctChg``/``volume``; occasionally a date row may be absent entirely.
Chosen strategy (keeps the multi-asset matrix aligned on a common
calendar): **return = 0 on any day the stock did not trade**. Rationale:
price did not move while suspended, and filling 0 keeps the trading
calendar shared across assets without look-ahead.

No-look-ahead guarantee
-----------------------
``load_returns`` filters every series to ``date <= end_date`` *before*
taking the trailing window, so no data after ``end_date`` can leak into
the result. This is a hard red line for the project.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

CACHE_DIR = "/mnt/f/Gaming/bs_cache_1year"


def _normalize_code(code: str) -> str:
    """Normalize mainland and US cache codes.

    Mainland examples ``'600000'``, ``'sh.600000'`` and ``'sh_600000'``
    become ``'600000'``. US examples ``'us_AAPL'``, ``'us.AAPL'`` and
    ``'US-AAPL'`` become ``'us_AAPL'``. Keeping the US prefix in the column
    label prevents collisions and makes the cache source explicit.
    """
    raw = str(code).strip()
    numeric = re.search(r"(\d{6})", raw)
    if numeric:
        return numeric.group(1)
    us = re.fullmatch(r"us[._-]([A-Za-z][A-Za-z0-9.-]*)", raw,
                      flags=re.IGNORECASE)
    if us:
        return f"us_{us.group(1).upper()}"
    raise ValueError(f"cannot parse stock code: {code!r}")


def _file_for(code6: str, cache_dir: str) -> Path:
    """Resolve a canonical mainland code or ``us_TICKER`` cache file."""
    if code6.startswith("us_"):
        path = Path(cache_dir) / f"{code6}_1year.pkl"
    else:
        # exchange prefix from the code (same convention as the cache)
        prefix = "sh" if code6[0] in ("5", "6", "9") else "sz"
        path = Path(cache_dir) / f"{prefix}_{code6}_1year.pkl"
    if not path.exists():
        # Defensive fallback for unusual mainland prefixes / cache casing.
        alt = list(Path(cache_dir).glob(f"*_{code6}_1year.pkl"))
        if alt:
            return alt[0]
        raise FileNotFoundError(f"no 1year pkl for code {code6} in {cache_dir}")
    return path


def _load_one(code6: str, cache_dir: str) -> pd.Series:
    """Daily return series (date -> float) for one stock, NaN-free."""
    df = pd.read_pickle(_file_for(code6, cache_dir))
    ret = pd.to_numeric(df["pctChg"], errors="coerce") / 100.0
    # suspended rows (tradestatus='0' / empty pctChg) -> return 0
    ret = ret.fillna(0.0)
    s = pd.Series(ret.values, index=df["date"].astype(str).values, name=code6)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def load_returns(codes: list[str], end_date: str, window: int,
                 cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Trailing ``window`` days of daily returns ending at ``end_date``.

    Parameters
    ----------
    codes : list of mainland codes in common formats ('600000',
        'sh.600000', 'sh_600000') and/or US codes ('us_AAPL', 'us.AAPL').
        Columns use canonical 6-digit / ``us_TICKER`` labels.
    end_date : 'YYYY-MM-DD', inclusive. Strictly no data after this date
        is used (no look-ahead).
    window : number of trading days in the returned matrix.

    Returns
    -------
    (window, len(codes)) DataFrame of daily returns, index = trading-day
    date strings (ascending, last row == the last trading day <= end_date
    on the union calendar of the requested codes).

    Raises
    ------
    FileNotFoundError if a code has no cached pkl.
    ValueError if fewer than ``window`` trading days are available.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    codes6 = [_normalize_code(c) for c in codes]
    end_date = str(end_date)

    series = {}
    for c6 in codes6:
        s = _load_one(c6, cache_dir)
        series[c6] = s[s.index <= end_date]  # hard no-look-ahead filter

    calendar = sorted({d for s in series.values() for d in s.index})
    if len(calendar) < window:
        raise ValueError(
            f"insufficient history: {len(calendar)} trading days <= "
            f"{end_date}, need {window}")
    calendar = calendar[-window:]

    out = pd.DataFrame(
        {c6: series[c6].reindex(calendar).fillna(0.0) for c6 in codes6}
    )
    out.index.name = "date"
    return out
