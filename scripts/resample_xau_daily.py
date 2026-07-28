"""Resample XAU 1h data → daily bars (Beijing-time 08:00 snapshot).

Uses the existing 1-year XAU 1h CSV from the crypto data lake.
Aligns to A-share trading calendar for direct use in qubo_builder.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── config ──────────────────────────────────────────────────
XAU_1H_PATH = Path(
    "/mnt/f/crypto_data/adaptive_btc_strategy/data/volatile_1m/"
    "XAU_USDT_USDT_1h.csv"
)
OUTPUT_DIR = Path("/mnt/f/Gaming/quantum_hedge/data/crypto_daily")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Beijing 08:00 snapshot = UTC 00:00
RESAMPLE_HOUR_UTC = 0


def load_xau_1h(path: Path) -> pd.DataFrame:
    """Load XAU 1h data, normalize timestamps."""
    df = pd.read_csv(path)

    # Detect timestamp column
    if "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        raise ValueError(f"Cannot find datetime column in {path}")

    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    print(f"Loaded {path.name}: {len(df)} rows, "
          f"{df.index[0]} → {df.index[-1]}")
    return df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h → daily at UTC 00:00 (= Beijing 08:00).

    Daily bar:
      open  = first hour's open
      high  = max of 24h highs
      low   = min of 24h lows
      close = last hour's close
      volume = sum of 24h volumes
    """
    daily = df.resample("24h", offset=f"{RESAMPLE_HOUR_UTC}h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    daily["symbol"] = "XAU/USDT"
    daily = daily.reset_index()
    print(f"Resampled to daily: {len(daily)} rows, "
          f"{daily['dt'].iloc[0]} → {daily['dt'].iloc[-1]}")
    return daily


def align_to_stock_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    """Add a 'date' column (YYYY-MM-DD, Beijing time) for A-share alignment.

    The resample anchor is UTC 00:00 = Beijing 08:00, so the date label
    corresponds to the A-share trading day that just opened.
    """
    # Convert to Beijing time (+8h) and extract date
    daily["date"] = (daily["dt"] + timedelta(hours=8)).dt.strftime("%Y-%m-%d")
    return daily


def main():
    print("=" * 55)
    print(" XAU 1h → Daily Resample")
    print("=" * 55)

    df_1h = load_xau_1h(XAU_1H_PATH)
    daily = resample_to_daily(df_1h)
    daily = align_to_stock_calendar(daily)

    # Output
    out_path = OUTPUT_DIR / "XAU_USDT_1d.csv"
    daily.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path} ({len(daily)} daily bars)")
    print(f"Columns: {list(daily.columns)}")
    print(f"Sample:")
    print(daily.head(3).to_string(index=False))
    print(f"...")
    print(daily.tail(3).to_string(index=False))

    # Sanity checks
    assert len(daily) > 200, f"Expected >200 daily bars, got {len(daily)}"
    assert not daily["close"].isnull().any(), "Null close values found"
    assert (daily["high"] >= daily["low"]).all(), "high < low detected"
    print("\n✅ Sanity checks passed.")


if __name__ == "__main__":
    main()
