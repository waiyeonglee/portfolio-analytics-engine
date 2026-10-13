import os
import re
from pathlib import Path

import pandas as pd

from moomoo import RET_OK

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
    """Fetch historical US 3-month Treasury bill rates."""

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

    rf = rf.dropna(subset=["DTB3"])

    rf["annual_rf"] = rf["DTB3"] / 100
    rf["daily_rf"] = rf["annual_rf"] / 252

    return rf[["DATE", "annual_rf", "daily_rf"]]

def add_sharpe_ratio(df):
    """Calculate historical excess returns and annualized Sharpe."""

    df = df.copy()

    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df["total_assets"] = pd.to_numeric(
        df["total_assets"],
        errors="coerce",
    )

    df["asset_difference"] = df["total_assets"].diff()

    df["asset_difference_ratio"] = (
        df["total_assets"].pct_change(fill_method=None) * 100
    )

    rf = get_risk_free_rates()

    df = pd.merge_asof(
        df,
        rf.sort_values("DATE"),
        left_on="date",
        right_on="DATE",
        direction="backward",
    )

    df["daily_return"] = (
        df["asset_difference_ratio"] / 100
    )

    df["excess_return"] = (
        df["daily_return"] - df["daily_rf"]
    )

    excess = df["excess_return"]

    mean = excess.expanding(min_periods=2).mean()
    std = excess.expanding(min_periods=2).std(ddof=1)

    df["sharpe_ratio"] = (
        mean / std.where(std > 0) * (252 ** 0.5)
    )

    return df.drop(columns=["DATE"])

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

    ret, account = trade_ctx.accinfo_query(
        trd_env=config.trade_env,
        currency="SGD",
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching account info: {account}"
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

    df["date"] = config.timezone_date
    df["total_assets"] = float(
        account["total_assets"].iloc[0]
    )

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

    # Risk-adjusted performance
    df = add_sharpe_ratio(df)

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
        "annual_rf",
        "sharpe_ratio",
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
    timestamp = (pd.Timestamp.today().strftime("%Y-%m-%d %H_%M_%S"))
    
    if config.live_mode:
        mode = "live"
        price_column = "execution_price"

    else:
        mode = "backtest"
        price_column = "close"

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