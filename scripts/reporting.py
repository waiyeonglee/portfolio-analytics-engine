import os
from pathlib import Path

import numpy as np
import pandas as pd
from moomoo import TrdEnv

from .broker import get_portfolio_equity, get_position_status


def calculate_buy_hold(output_df):
    """Calculate buy-and-hold return from strategy candles."""

    df = output_df.loc[output_df["action"] != "INITIALIZING"].copy()

    df = df.sort_values("time")

    if df.empty:
        return None

    initial_price = float(df["close"].iloc[0])
    final_price = float(df["close"].iloc[-1])

    buy_hold_return = final_price / initial_price - 1

    return buy_hold_return * 100


def get_risk_free_rates():
    """Fetch historical US 3-month Treasury rates from FRED."""

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2026-01-01"

    rf = pd.read_csv(url)

    rf["DATE"] = pd.to_datetime(rf["observation_date"])

    rf["DTB3"] = pd.to_numeric(
        rf["DTB3"],
        errors="coerce",
    )

    rf = rf.dropna(subset=["DTB3"]).sort_values("DATE")

    rf["annual_rf"] = rf["DTB3"] / 100
    rf["daily_rf"] = rf["annual_rf"] / 252

    return rf[["DATE", "annual_rf", "daily_rf"]]


def get_daily_risk_free_rate(trading_date):
    """Get the latest available risk-free rate for a trading date."""

    rf = get_risk_free_rates()

    trading_date = pd.Timestamp(trading_date)

    if trading_date.tzinfo is not None:
        trading_date = trading_date.tz_localize(None)

    trading_date = trading_date.normalize()

    available = rf.loc[rf["DATE"] <= trading_date]

    if available.empty:
        raise ValueError(f"No Treasury rate available for {trading_date}")

    print("Daily Risk Free Date:", available["DATE"].iloc[-1].strftime("%Y-%m-%d"))
    print("Daily Risk Free Rate:", available["daily_rf"].iloc[-1])
    return float(available["daily_rf"].iloc[-1])


def calculate_intraday_sharpe(
    output_df,
    initial_cash,
    daily_rf=0.0,
    periods_per_day=390,
):
    """Calculate intraday portfolio equity, returns, and Sharpe ratio."""

    df = output_df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        format="mixed",
    )

    df = df.sort_values("time").reset_index(drop=True)

    action = df["action"]
    trade_qty = pd.to_numeric(df["trade_qty"], errors="raise")
    close = pd.to_numeric(df["close"], errors="raise")
    execution_price = pd.to_numeric(
        df["execution_price"],
        errors="coerce",
    ).fillna(close)
    fee = pd.to_numeric(df["fee_amount"], errors="coerce").fillna(0)
    position_qty = pd.to_numeric(df["Position Qty"], errors="raise")
    buy_mask = action.eq("BUY")
    sell_mask = action.eq("SELL")

    cash_change = (-trade_qty * execution_price - fee).where(
        buy_mask,
        0,
    ) + (trade_qty * execution_price - fee).where(sell_mask, 0)

    df["cash"] = float(initial_cash) + cash_change.cumsum()
    df["equity"] = df["cash"] + position_qty * close

    trading_mask = df["action"] != "INITIALIZING"
    df["returns"] = df["equity"].pct_change(fill_method=None).where(trading_mask)

    df["excess_return"] = np.nan

    df.loc[trading_mask, "excess_return"] = (
        df.loc[trading_mask, "returns"] - daily_rf / periods_per_day
    )

    excess = df.loc[trading_mask, "excess_return"].dropna()

    intraday_sharpe = np.nan

    if len(excess) >= 2:
        volatility = excess.std(ddof=1)

        if pd.notna(volatility) and volatility > 0:
            intraday_sharpe = excess.mean() / volatility * np.sqrt(periods_per_day)

    return df, intraday_sharpe


def compute_daily_pl(output_df):
    """Calculate net realized P/L from sell trades."""

    # Realized PL
    sell_mask = output_df["action"].eq("SELL")
    sell_df = output_df.loc[sell_mask].copy()
    sell_fees = sell_df["fee_amount"]
    sell_df["realized_pl"] = sell_df["realized_pl_gross"] - sell_fees
    realized_pl_sum = sell_df["realized_pl"].sum()

    print(f"Total Realized P/L: {realized_pl_sum:.0f}")

    return sell_df, realized_pl_sum


def get_daily_status(
    trade_ctx,
    runtime_settings,
    realized_pl_sum,
    initial_equity,
    intraday_sharpe_ratio,
    buy_hold_return_pct,
    logs_folder,
    daily_status_file_name,
):
    """Build a concise daily portfolio and P/L report."""

    # Unrealized PL
    df = get_position_status(
        trade_ctx,
        runtime_settings,
    )
    df["date"] = runtime_settings.market_datetime
    cost_column = (
        "average_cost" if runtime_settings.trade_env == TrdEnv.REAL else "diluted_cost"
    )
    cost_basis = pd.to_numeric(df[cost_column], errors="coerce")
    df["unrealized_pl_sum"] = (
        pd.to_numeric(df["nominal_price"], errors="coerce") - cost_basis
    ) * pd.to_numeric(df["qty"], errors="coerce")

    # Realized PL
    df["daily_realized_pl_sum"] = realized_pl_sum
    df["intraday_sharpe_ratio"] = intraday_sharpe_ratio
    df["buy_hold_return_pct"] = buy_hold_return_pct

    # Total Assets
    df["total_assets"] = get_portfolio_equity(
        trade_ctx,
        runtime_settings,
        currency="USD",
    )

    # Each saved status file contains the full history, so only load the newest one.
    latest_status_file = max(
        Path(logs_folder).glob(f"*{daily_status_file_name}"),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )
    previous_status = (
        pd.read_csv(latest_status_file)
        if latest_status_file is not None
        else pd.DataFrame()
    )
    if not previous_status.empty and "date" in previous_status:
        previous_status["_date_key"] = pd.to_datetime(
            previous_status["date"],
            errors="coerce",
        ).dt.date
        previous_status = previous_status.dropna(subset=["_date_key"])
    else:
        previous_status = pd.DataFrame()

    current_date = pd.Timestamp(df["date"].iloc[0]).date()
    prior_days = (
        previous_status.loc[previous_status["_date_key"] < current_date].copy()
        if not previous_status.empty
        else pd.DataFrame()
    )

    df = pd.concat(
        [prior_days, df],
        ignore_index=True,
        sort=False,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # Realized P/L is accumulated within each market date; unrealized P/L is a snapshot.
    df["daily_unrealized_pl_difference"] = df["unrealized_pl_sum"].diff()
    df["daily_total_pl_sum"] = (
        df["daily_realized_pl_sum"] + df["daily_unrealized_pl_difference"]
    )
    df["daily_total_pl_sum_pct"] = (
        df["daily_total_pl_sum"] / initial_equity * 100 if initial_equity > 0 else 0.0
    )
    df["asset_difference"] = pd.to_numeric(
        df["total_assets"],
        errors="coerce",
    ).diff()
    df["reconciliation_difference"] = df["asset_difference"] - df["daily_total_pl_sum"]

    output_columns = [
        "date",
        "code",
        "qty",
        "total_assets",
        "daily_realized_pl_sum",
        "unrealized_pl_sum",
        "daily_total_pl_sum",
        "daily_unrealized_pl_difference",
        "asset_difference",
        "reconciliation_difference",
        "daily_total_pl_sum_pct",
        "buy_hold_return_pct",
        "intraday_sharpe_ratio",
    ]

    return df[output_columns]


def save_results(
    strategy,
    trade_ctx,
    runtime_settings,
):
    """Save trading logs, P/L, and daily status."""

    logs_folder = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_folder, exist_ok=True)

    if runtime_settings.live_mode:
        mode = "live"
        timestamp = pd.Timestamp.today().strftime("%Y-%m-%d %H_%M_%S")

    else:
        mode = "backtest"
        timestamp = runtime_settings.market_datetime.strftime("%Y-%m-%d %H_%M_%S")

    # Filenames for logs
    trading_logs_file_name = f"{mode}_{runtime_settings.env_name}_trading_logs.csv"
    pl_file_name = f"{mode}_{runtime_settings.env_name}_pl.csv"
    daily_status_file_name = f"{mode}_{runtime_settings.env_name}_daily_status.csv"

    trading_logs_path = os.path.join(
        logs_folder,
        f"{timestamp} - {trading_logs_file_name}",
    )

    pl_path = os.path.join(
        logs_folder,
        f"{timestamp} - {pl_file_name}",
    )

    daily_status_path = os.path.join(
        logs_folder,
        f"{timestamp} - {daily_status_file_name}",
    )

    output_df = pd.DataFrame(strategy.output)

    if output_df.empty:
        return

    output_df["time_SG"] = (
        pd.to_datetime(output_df["time"])
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("Asia/Singapore")
    )

    daily_rf = get_daily_risk_free_rate(runtime_settings.market_datetime)

    output_df, intraday_sharpe_ratio = calculate_intraday_sharpe(
        output_df=output_df,
        initial_cash=strategy.initial_equity,
        daily_rf=daily_rf,
    )
    output_df.to_csv(trading_logs_path, index=False)

    buy_hold_return_pct = calculate_buy_hold(output_df)

    sell_df, realized_pl_sum = compute_daily_pl(
        output_df,
    )

    sell_df.to_csv(
        pl_path,
        index=False,
    )

    daily_status = get_daily_status(
        trade_ctx=trade_ctx,
        runtime_settings=runtime_settings,
        realized_pl_sum=realized_pl_sum,
        initial_equity=strategy.initial_equity,
        intraday_sharpe_ratio=intraday_sharpe_ratio,
        buy_hold_return_pct=buy_hold_return_pct,
        logs_folder=logs_folder,
        daily_status_file_name=(daily_status_file_name),
    )

    daily_status.to_csv(
        daily_status_path,
        index=False,
    )
