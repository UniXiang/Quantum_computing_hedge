"""Download ETF daily kline data via baostock.

Adds ETF data to the existing bs_cache_1year directory in the same pkl format.
ETFs are identified by their listing exchange (sh.51xxxx or sz.159xxx).
"""

import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd

warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────
CACHE_DIR = Path("/mnt/f/Gaming/bs_cache_1year")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PAUSE = 0.05

# Target ETFs: (baostock_code, label, description)
TARGET_ETFS = [
    # Gold / Commodity ETFs
    ("sh.518880", "黄金ETF", "华安黄金ETF"),
    ("sz.159985", "豆粕ETF", "华夏饲料豆粕期货ETF"),
    ("sz.159981", "有色ETF", "南方中证申万有色金属ETF"),
    # Broad market ETFs
    ("sh.510050", "上证50ETF", "华夏上证50ETF"),
    ("sh.510300", "沪深300ETF", "华泰柏瑞沪深300ETF"),
    ("sh.510500", "中证500ETF", "南方中证500ETF"),
    ("sz.159915", "创业板ETF", "易方达创业板ETF"),
    ("sz.159919", "沪深300ETF", "嘉实沪深300ETF"),
    ("sz.159949", "创业板50ETF", "华安创业板50ETF"),
    # Overseas ETFs
    ("sh.513100", "纳指ETF", "国泰纳斯达克100ETF"),
    ("sh.513500", "标普500ETF", "博时标普500ETF"),
    # Oil / Energy ETFs
    ("sz.159930", "能源ETF", "汇添富中证能源ETF"),
    ("sh.510410", "资源ETF", "博时上证自然资源ETF"),
]

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"


def rs_to_df(rs):
    """Convert baostock result set to DataFrame."""
    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        return pd.DataFrame()
    return pd.DataFrame(data_list, columns=rs.fields)


def get_latest_trade_day():
    """Find the most recent trading day with available data."""
    today = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=15)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start, end_date=today)
    df = rs_to_df(rs)
    if df.empty:
        return None
    trading_days = df[df["is_trading_day"] == "1"]["calendar_date"].tolist()
    for d in reversed(trading_days):
        rs_k = bs.query_history_k_data_plus(
            "sh.600000", "date,close",
            start_date=d, end_date=d, frequency="d",
        )
        test = rs_to_df(rs_k)
        if not test.empty:
            return d
    return None


def download_one(code: str, end_date: str, cache_file: Path) -> pd.DataFrame | None:
    """Download 1 year of daily data, save as pkl. Returns the DataFrame."""
    one_year_ago = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
    ).strftime("%Y-%m-%d")

    rs = bs.query_history_k_data_plus(
        code, FIELDS,
        start_date=one_year_ago, end_date=end_date,
        frequency="d", adjustflag="3",
    )
    df = rs_to_df(rs)
    if df is None or df.empty:
        return None

    df.to_pickle(cache_file)
    return df


def main():
    lg = bs.login()
    if lg.error_code != "0":
        print(f"[FATAL] baostock login failed: {lg.error_msg}")
        return

    try:
        latest_date = get_latest_trade_day()
        if not latest_date:
            print("[FATAL] Cannot find any recent trading day with data.")
            return

        print(f"Latest trading day with data: {latest_date}")
        print(f"Target ETFs: {len(TARGET_ETFS)}")
        print(f"Cache dir: {CACHE_DIR.resolve()}")
        print()

        downloaded, skipped, failed = 0, 0, 0

        for code, label, desc in TARGET_ETFS:
            file_name = code.replace(".", "_") + "_1year.pkl"
            cache_file = CACHE_DIR / file_name

            if cache_file.exists():
                # Check date range
                try:
                    existing = pd.read_pickle(cache_file)
                    if not existing.empty and "date" in existing.columns:
                        existing_dates = pd.to_datetime(existing["date"], errors="coerce")
                        last_date = existing_dates.max().strftime("%Y-%m-%d") if not existing_dates.empty else "?"
                        print(f"  [{code}] {label} ({desc}) — already cached "
                              f"({len(existing)} rows, {last_date}), skip")
                        skipped += 1
                        continue
                except Exception:
                    pass  # re-download if unreadable

            print(f"  [{code}] {label} ({desc}) — downloading...", end=" ", flush=True)
            df = download_one(code, latest_date, cache_file)

            if df is not None and not df.empty:
                dates = pd.to_datetime(df["date"], errors="coerce")
                d_min = dates.min().strftime("%Y-%m-%d") if not dates.empty else "?"
                d_max = dates.max().strftime("%Y-%m-%d") if not dates.empty else "?"
                print(f"✅ {len(df)} rows ({d_min} → {d_max})")
                downloaded += 1
            else:
                print("❌ no data returned (ETF may be delisted or code changed)")
                failed += 1

            time.sleep(PAUSE)

        print(f"\n{'='*50}")
        print(f" Done: downloaded {downloaded}, skipped {skipped}, failed {failed}")
        print(f" Cache: {CACHE_DIR.resolve()}")
        print(f"{'='*50}")

    finally:
        bs.logout()


if __name__ == "__main__":
    main()
