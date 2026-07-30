"""下载指定A股+沪深300的6年日线数据到 bs_cache_1year。

用法：python scripts/download_6year_data.py

数据源：baostock
缓存目录：/mnt/f/Gaming/bs_cache_1year/
目标：2020-07-28 ~ 最新交易日（约6年）
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path("/mnt/f/Gaming/bs_cache_1year")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PAUSE = 0.02  # 礼貌间隔

# 跟现有缓存一致的字段
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
# 指数用字段（无 turn/tradestatus/isST）
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

# ========== 目标标的 ==========

# alpha 候选（11只A股）
ALPHA_CODES = [
    ("sz.300308", "中际旭创"),
    ("sz.002475", "立讯精密"),
    ("sz.300502", "新易盛"),
    ("sh.603986", "兆易创新"),
    ("sh.688981", "中芯国际"),
    ("sz.300750", "宁德时代"),
    ("sh.601869", "长飞光纤"),
    ("sz.002371", "北方华创"),
    ("sh.601100", "恒立液压"),
    ("sh.688017", "绿的谐波"),
    ("sz.002185", "华天科技"),
]

# hedge A股（4只）
HEDGE_CODES = [
    ("sh.601288", "农业银行"),
    ("sh.600519", "贵州茅台"),
    ("sh.603259", "药明康德"),
    ("sz.300059", "东方财富"),
]

# 沪深300 指数
INDEX_CODES = [
    ("sh.000300", "沪深300"),
]

ALL_CODES = ALPHA_CODES + HEDGE_CODES + INDEX_CODES


def rs_to_df(rs):
    """baostock result set → DataFrame"""
    data_list = []
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        return pd.DataFrame()
    return pd.DataFrame(data_list, columns=rs.fields)


def get_latest_trade_day():
    """获取最新的有数据的交易日"""
    today = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=15)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start, end_date=today)
    df = rs_to_df(rs)
    trading_days = df[df["is_trading_day"] == "1"]["calendar_date"].tolist()
    for test_date in reversed(trading_days):
        rs_k = bs.query_history_k_data_plus(
            "sh.600000", "date,close",
            start_date=test_date, end_date=test_date, frequency="d",
        )
        if not rs_to_df(rs_k).empty:
            return test_date
    return None


def download_stock(code: str, name: str, start_date: str, end_date: str):
    """下载单只股票6年数据，覆盖缓存"""
    code_short = code.replace(".", "_")
    cache_file = CACHE_DIR / f"{code_short}_1year.pkl"

    rs = bs.query_history_k_data_plus(
        code, FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3",  # 后复权
    )
    df = rs_to_df(rs)
    if df.empty:
        print(f"  ⚠️  {name:6s} ({code}) → 空数据，跳过")
        return False

    df.to_pickle(cache_file)
    print(f"  ✅ {name:6s} ({code}) → {len(df)} 行 ({df['date'].min()} ~ {df['date'].max()})")
    return True


def download_index(code: str, name: str, start_date: str, end_date: str):
    """下载指数6年数据，覆盖缓存"""
    code_short = code.replace(".", "_")
    cache_file = CACHE_DIR / f"{code_short}_1year.pkl"

    rs = bs.query_history_k_data_plus(
        code, INDEX_FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2",  # 前复权（指数用）
    )
    df = rs_to_df(rs)
    if df.empty:
        print(f"  ⚠️  {name:6s} ({code}) → 空数据，跳过")
        return False

    df.to_pickle(cache_file)
    print(f"  ✅ {name:6s} ({code}) → {len(df)} 行 ({df['date'].min()} ~ {df['date'].max()})")
    return True


def main():
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock 登录失败: {lg.error_msg}")
        return
    print(f"✅ baostock 登录成功")

    try:
        latest_date = get_latest_trade_day()
        if not latest_date:
            print("❌ 无法获取最新交易日")
            return

        # 计算6年前的日期
        dt_latest = datetime.strptime(latest_date, "%Y-%m-%d")
        six_years_ago = (dt_latest - timedelta(days=6*365 + 1)).strftime("%Y-%m-%d")

        print(f"\n📅 最新交易日: {latest_date}")
        print(f"📅 下载起始日: {six_years_ago}（约6年）")
        print(f"📂 缓存目录: {CACHE_DIR}")
        print(f"🎯 共 {len(ALL_CODES)} 个标的\n")

        # 下载股票
        print("=" * 60)
        print("📈 下载 A 股日线（后复权）")
        print("=" * 60)
        for code, name in ALPHA_CODES + HEDGE_CODES:
            download_stock(code, name, six_years_ago, latest_date)
            time.sleep(PAUSE)

        # 下载指数
        print("\n" + "=" * 60)
        print("📊 下载指数日线（前复权）")
        print("=" * 60)
        for code, name in INDEX_CODES:
            download_index(code, name, six_years_ago, latest_date)
            time.sleep(PAUSE)

        print(f"\n{'=' * 60}")
        print(f"🎉 全部下载完成！")
        print(f"   标的数: {len(ALL_CODES)}")
        print(f"   日期范围: {six_years_ago} ~ {latest_date}")
        print(f"{'=' * 60}")

    finally:
        bs.logout()
        print("👋 baostock 已登出")


if __name__ == "__main__":
    main()
