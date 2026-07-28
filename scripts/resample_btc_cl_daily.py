"""Resample BTC & CL 1m data → daily bars (Beijing-time 08:00 snapshot).

Uses the 1m CSVs from the crypto data lake.
These have ~49 days of coverage (2026-05-24 → 2026-07-12).
Daily resample anchor: UTC 00:00 = Beijing 08:00, aligned with A-shares.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

# ── config ──────────────────────────────────────────────────
DATA_LAKE = Path("/mnt/f/crypto_data/adaptive_btc_strategy/data/volatile_1m")
SOURCES = {
    "BTC": DATA_LAKE / "BTC_USDT_USDT_1m.csv",
    "CL":  DATA_LAKE / "CL_USDT_USDT_1m.csv",
}
OUTPUT_DIR = Path("/mnt/f/Gaming/quantum_hedge/data/crypto_daily")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_1m(path: Path) -> pd.DataFrame:
    """Load 1m CSV, normalise datetime column."""
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    # Convert columns to numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resample_daily(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Resample 1m → daily at UTC 00:00 (Beijing 08:00)."""
    daily = df.resample("24h", offset="0h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Add Beijing date for A-share alignment
    daily["date"] = (daily.index + timedelta(hours=8)).strftime("%Y-%m-%d")
    daily["symbol"] = f"{label}/USDT"
    daily = daily.reset_index()

    print(f"  {label}: {len(daily)} daily bars, "
          f"{daily['dt'].iloc[0]} → {daily['dt'].iloc[-1]}")
    return daily


def main():
    print("=" * 55)
    print(" BTC & CL 1m → Daily Resample")
    print("=" * 55)

    for label, src_path in SOURCES.items():
        print(f"\n[{label}] Loading {src_path.name} ...")
        df_1m = load_1m(src_path)
        print(f"  {len(df_1m)} 1m rows loaded "
              f"({df_1m.index[0]} → {df_1m.index[-1]})")

        daily = resample_daily(df_1m, label)

        # Sanity
        assert len(daily) > 30, f"Expected >30 daily bars, got {len(daily)}"
        assert not daily["close"].isnull().any()
        assert (daily["high"] >= daily["low"]).all()

        out_path = OUTPUT_DIR / f"{label}_USDT_1d.csv"
        daily.to_csv(out_path, index=False)
        print(f"  → {out_path}")

        # Sample
        for _, row in daily.head(3).iterrows():
            print(f"    {row['date']}  O={row['open']:.2f}  C={row['close']:.2f}  "
                  f"H={row['high']:.2f}  L={row['low']:.2f}  V={row['volume']:.0f}")

    # ── summary ──
    print(f"\n{'='*55}")
    print(" Files in output dir:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    print("=" * 55)


if __name__ == "__main__":
    main()
