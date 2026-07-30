"""Download CL (crude oil) and BTC daily klines from OKX (1 year).

Uses OKX REST API directly via proxy (no ccxt dependency for network).
Public endpoint — no API key needed.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── config ──────────────────────────────────────────────────
PROXY_URL = (
    os.environ.get("http_proxy")
    or os.environ.get("HTTP_PROXY")
    or "http://172.31.208.1:7890"
)
PROXIES = {"http": PROXY_URL, "https": PROXY_URL}

# instId format on OKX: <base>-<quote>-SWAP for perpetuals
SYMBOLS = {
    "BTC": "BTC-USDT-SWAP",
    "CL": "CL-USDT-SWAP",
    "XAU": "XAU-USDT-SWAP",
}
BAR = "1D"
DAYS = 365 * 6  # 6年，API能返回多少就取多少
OUTPUT_DIR = Path("/mnt/f/Gaming/quantum_hedge/data/crypto_daily")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.okx.com/api/v5/market/candles"
SESSION = requests.Session()
# Set both env and session proxies
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL


def fetch_daily(inst_id: str, days: int) -> list:
    """Fetch daily OHLCV candles from OKX, paginate back in time.

    OKX candle format: [ts_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    Returns list of dicts sorted ascending by timestamp.
    """
    all_candles = []
    limit = 300  # OKX max

    print(f"  {inst_id}: ~{days} bars expected")
    print(f"  Fetching...", end="", flush=True)

    # OKX returns newest first, we paginate backwards
    after = None  # timestamp in ms for "candles before this"
    consecutive_errors = 0

    while len(all_candles) < days + 10:  # small buffer
        params = {"instId": inst_id, "bar": BAR, "limit": str(limit)}
        if after:
            params["after"] = str(after)

        try:
            resp = SESSION.get(BASE_URL, params=params,
                              proxies=PROXIES, timeout=30)
            data = resp.json()
        except Exception as e:
            consecutive_errors += 1
            print(f"\n  ⚠️  request error: {e}", flush=True)
            if consecutive_errors > 5:
                print(f"  ❌ Too many errors, giving up.", flush=True)
                break
            time.sleep(3)
            continue

        if data.get("code") != "0":
            print(f"\n  ⚠️  API error: {data.get('msg', 'unknown')}", flush=True)
            break

        candles = data["data"]
        if not candles:
            print(f" (no more data)", flush=True)
            break

        all_candles.extend(candles)
        consecutive_errors = 0
        after = int(candles[-1][0])  # oldest candle timestamp
        print(f" {len(all_candles)}", end="", flush=True)

        if len(candles) < limit:
            # Partial page = reached beginning of history
            break

        time.sleep(0.15)  # rate limit

    print(flush=True)

    # Deduplicate by timestamp, sort ascending
    seen = set()
    unique = []
    for c in all_candles:
        ts = int(c[0])
        if ts not in seen:
            seen.add(ts)
            unique.append(c)
    unique.sort(key=lambda c: int(c[0]))
    print(f"  ✅ {len(unique)} unique bars after dedup")
    return unique


def save_csv(candles: list, label: str) -> Path:
    """Convert OKX candles to DataFrame, save as CSV."""
    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp_ms", "open", "high", "low", "close",
            "volume_ct", "volume_ccy", "volume_ccy_quote", "confirm",
        ],
    )
    # Convert types
    for col in ["open", "high", "low", "close", "volume_ct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    out_path = OUTPUT_DIR / f"{label}_USDT_1d.csv"
    df.to_csv(out_path, index=False)

    d_min = df["datetime"].iloc[0]
    d_max = df["datetime"].iloc[-1]
    print(f"  💾 {out_path}  ({len(df)} rows, {d_min} → {d_max})")
    return out_path


def main():
    print(f"Downloading daily klines from OKX ({DAYS}d window)")
    print(f"Proxy: {PROXY_URL}")
    print(f"Symbols: {list(SYMBOLS.values())}")
    print(f"Output:  {OUTPUT_DIR.resolve()}")
    print()

    for label, inst_id in SYMBOLS.items():
        print(f"{'='*50}")
        print(f" {label}  ({inst_id})")
        print(f"{'='*50}")
        candles = fetch_daily(inst_id, DAYS)
        if candles:
            save_csv(candles, label)
        else:
            print(f"  ❌ No data returned")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
