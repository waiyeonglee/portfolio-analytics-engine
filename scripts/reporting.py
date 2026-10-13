import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from moomoo import RET_OK

from broker import get_portfolio_equity

def calculate_buy_hold(output_df):
    """Calculate buy-and-hold return from strategy candles."""

    df = output_df.loc[
        output_df["action"] != "INITIALIZING"
    ].copy()

    df = df.sort_values("time")

    if df.empty:
        return None

    initial_price = float(df["close"].iloc[0])
    final_price = float(df["close"].iloc[-1])

    buy_hold_return = (
        final_price / initial_price - 1
    )

    return buy_hold_return * 100

def get_risk_free_rates():
    """Fetch historical US 3-month Treasury rates from FRED."""

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=DTB3"
    )

    rf = pd.read_csv(url)

    rf["DATE"] = pd.to_datetime(rf["observation_date"])

    rf["DTB3"] = pd.to_numeric(
        rf["DTB3"],
        errors="coerce",
    )

    rf = (
        rf.dropna(subset=["DTB3"])
        .sort_values("DATE")
    )

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

    available = rf.loc[
        rf["DATE"] <= trading_date
    ]

    if available.empty:
        raise ValueError(
            f"No Treasury rate available for {trading_date}"
        )

    return float(available["daily_rf"].iloc[-1])

def calculate_intraday_sharpe(
    output_df,
    initial_cash,
    daily_rf=0.0,
    periods_per_day=390,
):
    """Calculate intraday Sharpe and return the equity DataFrame."""

    df = output_df.loc[
        output_df["action"] != "INITIALIZING"
    ].sort_values("time").copy()

    cash = float(initial_cash)
    position_qty = 0

    cash_history = []
    position_history = []
    equity_history = []

    for _, row in df.iterrows():
        action = row["action"]
        qty = float(row["trade_qty"])
        price = float(row["close"])

        execution_price = pd.to_numeric(
            row.get("execution_price"),
            errors="coerce",
        )

        if pd.isna(execution_price):
            execution_price = price

        fee = pd.to_numeric(
            row.get("fee_amount", 0),
            errors="coerce",
        )

        if pd.isna(fee):
            fee = 0.0

        if action == "BUY":
            cash -= qty * execution_price + fee
            position_qty += qty

        elif action == "SELL":
            cash += qty * execution_price - fee
            position_qty -= qty

        cash_history.append(cash)
        position_history.append(position_qty)
        equity_history.append(
            cash + position_qty * price
        )

    df["cash"] = cash_history
    df["position_qty"] = position_history
    df["equity"] = equity_history

    df["returns"] = df["equity"].pct_change(
        fill_method=None
    )

    df["excess_return"] = (
        df["returns"] - daily_rf / periods_per_day
    )

    excess = df["excess_return"].dropna()

    intraday_sharpe = np.nan

    if len(excess) >= 2:
        volatility = excess.std(ddof=1)

        if pd.notna(volatility) and volatility > 0:
            intraday_sharpe = (
                excess.mean()
                / volatility
                * np.sqrt(periods_per_day)
            )

    return df, intraday_sharpe

def compute_daily_pl(
    output_df,
    price_column,
):
    """Calculate realized daily P/L and peak exposure."""

    sell_df = output_df.loc[
        output_df["action"] == "SELL"
    ].copy()

    exposure = 0.0
    peak_exposure = 0.0

    for _, row in output_df.iterrows():

        if row["action"] == "BUY":
            exposure += (
                row[price_column]
                * row["trade_qty"]
            )

        elif row["action"] == "SELL":
            exposure -= (
                row["cost_price"]
                * row["trade_qty"]
            )

        peak_exposure = max(
            peak_exposure,
            exposure,
        )

    print(
        f"Peak Exposure: "
        f"{peak_exposure:.0f}"
    )

    sell_df["realized_pl"] = (
        (
            sell_df[price_column]
            - sell_df["cost_price"]
        )
        * sell_df["trade_qty"]
    )

    realized_pl_sum = (
        sell_df["realized_pl"].sum()
    )

    if peak_exposure > 0:
        realized_pl_pct = (
            realized_pl_sum
            / peak_exposure
            * 100
        )
    else:
        realized_pl_pct = 0.0

    print(
        f"Total Return: "
        f"{realized_pl_sum:.0f}, "
        f"{realized_pl_pct:.3f}%"
    )

    return (
        sell_df,
        realized_pl_sum,
        peak_exposure,
        realized_pl_pct,
    )

def get_daily_status(
    trade_ctx,
    config,
    realized_pl_sum,
    peak_exposure,
    realized_pl_pct,
    intraday_sharpe,
    buy_hold_return,
    logs_folder,
    daily_status_file_name,
):
    """Build daily portfolio status with risk-adjusted metrics."""

    ret, positions = trade_ctx.position_list_query(
        trd_env=config.trade_env
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching positions: {positions}"
        )

    position_columns = [
        "code",
        "qty",
        "nominal_price",
        "cost_price",
        "average_cost",
        "market_val",
        "pl_ratio",
        "pl_ratio_avg_cost",
    ]

    df = positions.loc[
        (positions["code"] == config.symbol)
        & (positions["qty"] > 0),
        position_columns,
    ].copy()

    if df.empty:
        df = pd.DataFrame(
            [{
                "code": config.symbol,
                **{
                    col: 0
                    for col in position_columns
                    if col != "code"
                },
            }]
        )

    df["final_cost_price"] = (
        pd.to_numeric(
            df["average_cost"],
            errors="coerce",
        ).fillna(df["cost_price"])
    )

    df["unrealized_pl_ratio"] = (
        pd.to_numeric(
            df["pl_ratio_avg_cost"],
            errors="coerce",
        ).fillna(df["pl_ratio"])
    )

    df["date"] = config.market_datetime
    df["total_assets"] = get_portfolio_equity(trade_ctx, config, currency="SGD")

    df["calculated_realized_pl_sum"] = realized_pl_sum
    df["calculated_realized_pl_ratio"] = realized_pl_pct
    df["calculated_peak_exposure"] = peak_exposure

    # Load previous daily status
    files = list(
        Path(logs_folder).glob(
            f"*{daily_status_file_name}"
        )
    )

    if files:

        def extract_date(path):
            match = re.search(
                r"\d{4}-\d{2}-\d{2}",
                path.name,
            )
            return (
                pd.Timestamp(match.group())
                if match
                else pd.Timestamp.min
            )

        prev_file = max(files, key=extract_date)

        if extract_date(prev_file).date() != pd.Timestamp.today().date():
            print("Previous file:", prev_file)

            df = pd.concat(
                [pd.read_csv(prev_file), df],
                ignore_index=True,
            )

    # Portfolio P/L
    df["unrealized_pl_sum"] = (
        df["unrealized_pl_ratio"]
        / 100
        * df["market_val"]
    )

    df["calculated_pl_sum"] = (
        df["calculated_realized_pl_sum"]
        + df["unrealized_pl_sum"]
    )

    df["asset_difference"] = (
        df["total_assets"].diff()
    )

    df["asset_difference_ratio"] = (
        df["total_assets"].pct_change(fill_method=None) * 100
    )

    # Sharpe ratio
    df["intraday_sharpe"] = intraday_sharpe
    # Buy-and-hold return
    df["buy_hold_return"] = buy_hold_return

    output_columns = [
        "date",
        "code",
        "qty",
        "market_val",
        "total_assets",
        "calculated_pl_sum",
        "asset_difference",
        "asset_difference_ratio",
        "intraday_sharpe",
        "buy_hold_return",
    ]

    return df[output_columns]

def save_results(
    strategy,
    trade_ctx,
    config,
):
    """Save trading logs, P/L, and daily status."""

    logs_folder = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_folder,exist_ok=True)
    
    if config.live_mode:
        mode = "live"
        price_column = "execution_price"
        timestamp = (pd.Timestamp.today().strftime("%Y-%m-%d %H_%M_%S"))

    else:
        mode = "backtest"
        price_column = "close"
        timestamp = config.market_datetime.strftime("%Y-%m-%d %H_%M_%S")

    # Filenames for logs
    trading_logs_file_name = (f"{mode}_{config.env_name}""_trading_logs.csv")
    pl_file_name = (f"{mode}_{config.env_name}""_pl.csv")
    daily_status_file_name = (f"{mode}_{config.env_name}""_daily_status.csv")

    trading_logs_path = os.path.join(
        logs_folder,
        f"{timestamp} - "
        f"{trading_logs_file_name}",
    )

    pl_path = os.path.join(
        logs_folder,
        f"{timestamp} - {pl_file_name}",
    )

    daily_status_path = os.path.join(
        logs_folder,
        f"{timestamp} - "
        f"{daily_status_file_name}",
    )

    output_df = pd.DataFrame(strategy.output)

    if output_df.empty:
        return

    output_df["time_SG"] = (
        pd.to_datetime(output_df["time"]).
        dt.tz_localize("America/New_York").
        dt.tz_convert("Asia/Singapore"))

    daily_rf = get_daily_risk_free_rate(
        config.market_datetime
    )
    
    output_df, intraday_sharpe = calculate_intraday_sharpe(
        output_df=output_df,
        initial_cash=strategy.initial_equity,
        daily_rf=daily_rf,
    )
    output_df.to_csv(trading_logs_path,index=False)     
    
    buy_hold_return = calculate_buy_hold(output_df)

    (
        sell_df,
        realized_pl_sum,
        peak_exposure,
        realized_pl_pct,
    ) = compute_daily_pl(
        output_df,
        price_column,
    )

    sell_df.to_csv(
        pl_path,
        index=False,
    )

    daily_status = get_daily_status(
        trade_ctx=trade_ctx,
        config=config,
        realized_pl_sum=realized_pl_sum,
        peak_exposure=peak_exposure,
        realized_pl_pct=realized_pl_pct,
        intraday_sharpe=intraday_sharpe,
        buy_hold_return=buy_hold_return,
        logs_folder=logs_folder,
        daily_status_file_name=(
            daily_status_file_name
        ),
    )

    daily_status.to_csv(
        daily_status_path,
        index=False,
    )