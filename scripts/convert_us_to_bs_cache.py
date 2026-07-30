"""Convert selected Nasdaq rows into the project's 13-column cache format.

Nasdaq market_daily does not contain high, low, turnover, adjusted close, or
an authoritative suspension flag.  Those unavailable string fields remain
empty. ``preclose`` and ``pctChg`` are raw close-to-close values, not adjusted
returns.  Output files are named ``us_{TICKER}_1year.pkl``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = [
    "MU", "INTC", "AMD", "TSM", "ASML", "MRVL", "ARM", "GOOGL", "MSFT",
    "AMZN", "META", "PANW", "BAC", "HAL", "NEE", "ICE", "NOC", "LLY",
]
OUTPUT_COLUMNS = [
    "date", "code", "open", "high", "low", "close", "preclose", "volume",
    "amount", "turn", "tradestatus", "pctChg", "isST",
]


def _number_string(value: float, decimals: int = 4) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{decimals}f}"


def convert_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = frame.loc[
        frame["code"].astype(str).str.upper().eq(symbol)
        & frame["asset_type"].eq("stock")
    ].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data = data.dropna(subset=["date", "open", "close", "volume"])
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    if data.empty:
        raise ValueError(f"{symbol}: no valid stock rows")
    last_date = data["date"].max()
    data = data.loc[data["date"].ge(last_date - pd.Timedelta(days=370))]
    preclose = data["close"].shift(1)
    pct_change = (data["close"] / preclose - 1.0) * 100.0
    tradable = data["tradable"].fillna(False).astype(bool)
    output = pd.DataFrame(
        {
            "date": data["date"].dt.strftime("%Y-%m-%d"),
            "code": f"us.{symbol}",
            "open": data["open"].map(_number_string),
            "high": "",
            "low": "",
            "close": data["close"].map(_number_string),
            "preclose": preclose.map(_number_string),
            "volume": data["volume"].round().astype("int64").astype(str),
            "amount": (data["close"] * data["volume"]).map(_number_string),
            "turn": "",
            "tradestatus": np.where(tradable, "1", "0"),
            "pctChg": pct_change.map(_number_string),
            "isST": "0",
        },
        columns=OUTPUT_COLUMNS,
    )
    return output.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--symbols", default=",".join(DEFAULT_SYMBOLS),
        help="comma-separated US stock symbols",
    )
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    frame = pd.read_parquet(
        args.input,
        columns=[
            "date", "code", "asset_type", "open", "close", "volume",
            "tradable",
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for symbol in symbols:
        output = convert_symbol(frame, symbol)
        path = args.output_dir / f"us_{symbol}_1year.pkl"
        output.to_pickle(path)
        summary.append(
            (
                symbol,
                len(output),
                output["date"].iloc[0],
                output["date"].iloc[-1],
                path,
            )
        )
    for symbol, rows, start, end, path in summary:
        print(f"{symbol}: {rows} rows, {start}..{end}, {path}")


if __name__ == "__main__":
    main()
