#!/usr/bin/env python3
"""
从 market_daily.parquet 提取美股七姐妹 (Magnificent 7) 日线数据，
按 ../bs_cache_1year/ 的 pkl 格式输出。
"""
import pandas as pd
import numpy as np
from pathlib import Path

# === 配置 ===
PARQUET_PATH = Path("/mnt/f/Gaming/quantum_hedge/market_daily.parquet")
OUTPUT_DIR = Path("/mnt/f/Gaming/quantum_hedge")
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA"]
# 注意：KO (Coca-Cola) 不在该 parquet 数据集中

# bs_cache_1year 的标准列
COLS = ["date", "code", "open", "high", "low", "close", "preclose",
        "volume", "amount", "turn", "tradestatus", "pctChg", "isST"]


def build_mag7_pkl(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """将单只股票的 parquet 行转换为 bs_cache_1year 格式的 DataFrame"""
    sub = df[df["code"] == ticker].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=COLS)

    # 按日期排序
    sub = sub.sort_values("date").reset_index(drop=True)

    # 计算衍生字段
    opens = sub["open"].values
    closes = sub["close"].values
    vols = sub["volume"].values

    # preclose: 前一天的 close，第一天用 close 本身
    precloses = np.roll(closes, 1)
    precloses[0] = closes[0]

    # high/low: 只有 open/close，用 max/min 近似
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)

    # amount: 近似成交额 = volume * close
    amounts = vols * closes

    # pctChg: 百分比涨跌幅
    pct_chgs = (closes / precloses - 1.0) * 100.0

    result = pd.DataFrame({
        "date": sub["date"].astype(str).values,
        "code": f"us.{ticker}",
        "open": [f"{v:.4f}" for v in opens],
        "high": [f"{v:.4f}" for v in highs],
        "low": [f"{v:.4f}" for v in lows],
        "close": [f"{v:.4f}" for v in closes],
        "preclose": [f"{v:.4f}" for v in precloses],
        "volume": [f"{int(v)}" for v in vols],
        "amount": [f"{v:.4f}" for v in amounts],
        "turn": ["0"] * len(sub),
        "tradestatus": ["1"] * len(sub),
        "pctChg": [f"{v:.4f}" for v in pct_chgs],
        "isST": ["0"] * len(sub),
    }, columns=COLS)

    return result


def main():
    print("嘟嘟噜~ 真由理开始提取美股七姐妹数据啦！")

    # 读取 parquet
    print(f"正在读取 {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"总行数: {len(df):,}, 总代码数: {df['code'].nunique()}")

    # 检查哪些目标在数据中
    available = df["code"].unique()
    found = [t for t in TICKERS if t in available]
    missing = [t for t in TICKERS if t not in available]

    print(f"找到的股票: {found}")
    if missing:
        print(f"未找到: {missing}")

    for ticker in found:
        result = build_mag7_pkl(df, ticker)
        out_path = OUTPUT_DIR / f"us_{ticker}_1year.pkl"
        result.to_pickle(out_path)
        print(f"  ✅ {ticker}: {len(result)} 行 -> {out_path.name}")

    # KO 检查
    if "KO" not in available:
        print("\n⚠️ 可口可乐 (KO) 不在该 parquet 数据集中，已跳过。")

    # 展示第一个文件的样例以验证格式
    first_ticker = found[0]
    verify = pd.read_pickle(OUTPUT_DIR / f"us_{first_ticker}_1year.pkl")
    print(f"\n📋 格式验证 ({first_ticker}):")
    print(f"  列: {verify.columns.tolist()}")
    print(f"  dtypes: {dict(verify.dtypes)}")
    print(f"  前 2 行:")
    print(verify.head(2).to_string())
    print(f"  后 2 行:")
    print(verify.tail(2).to_string())

    print("\n✨ 完成！嘟嘟噜~")


if __name__ == "__main__":
    main()
