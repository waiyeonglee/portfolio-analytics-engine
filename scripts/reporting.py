import os
import re
from pathlib import Path

import pandas as pd

from moomoo import RET_OK


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
    logs_folder,
    daily_status_file_name,
):
    """Build the daily portfolio-status dataframe."""

    ret, positions = (
        trade_ctx.position_list_query(
            trd_env=config.trade_env
        )
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching positions: "
            f"{positions}"
        )

    columns = [
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
        (
            positions["code"]
            == config.symbol
        )
        & (
            positions["qty"] > 0
        ),
        columns,
    ].copy()

    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "code": config.symbol,
                    "qty": 0,
                    "nominal_price": 0,
                    "cost_price": 0,
                    "average_cost": 0,
                    "market_val": 0,
                    "pl_ratio": 0,
                    "pl_ratio_avg_cost": 0,
                }
            ]
        )

    ret, account = trade_ctx.accinfo_query(
        trd_env=config.trade_env,
        currency="SGD",
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching account info: "
            f"{account}"
        )

    total_assets = float(
        account["total_assets"].iloc[0]
    )

    average_cost = pd.to_numeric(
        df["average_cost"],
        errors="coerce",
    )

    df["final_cost_price"] = (
        average_cost.mask(
            average_cost.isna(),
            df["cost_price"],
        )
    )

    pl_avg = pd.to_numeric(
        df["pl_ratio_avg_cost"],
        errors="coerce",
    )

    df["unrealized_pl_ratio"] = (
        pl_avg.mask(
            pl_avg.isna(),
            df["pl_ratio"],
        )
    )

    df["date"] = config.timezone_date
    df["total_assets"] = total_assets

    df[
        "calculated_realized_pl_sum"
    ] = realized_pl_sum

    df[
        "calculated_realized_pl_ratio"
    ] = realized_pl_pct

    df[
        "calculated_peak_exposure"
    ] = peak_exposure

    files = list(
        Path(logs_folder).glob(
            f"*{daily_status_file_name}"
        )
    )

    prev_df = None

    if files:

        def extract_date(file):
            match = re.search(
                r"\d{4}-\d{2}-\d{2}",
                file.name,
            )

            if match:
                return pd.to_datetime(
                    match.group()
                )

            return pd.Timestamp.min

        prev_file = max(
            files,
            key=extract_date,
        )

        file_date = extract_date(
            prev_file
        ).date()

        today = pd.Timestamp.today().date()

        if file_date != today:
            print(
                "Previous file:",
                prev_file,
            )

            prev_df = pd.read_csv(
                prev_file
            )

    if prev_df is not None:
        df = pd.concat(
            [prev_df, df],
            ignore_index=True,
        )

    df["unrealized_pl_sum"] = (
        df["unrealized_pl_ratio"]
        / 100
        * df["market_val"]
    )

    df["asset_difference"] = (
        df["total_assets"].diff()
    )

    df["asset_difference_ratio"] = (
        df["total_assets"].pct_change()
        * 100
    )

    df["calculated_pl_sum"] = (
        df["calculated_realized_pl_sum"]
        + df["unrealized_pl_sum"]
    )

    columns = [
        "date",
        "code",
        "qty",
        "nominal_price",
        "cost_price",
        "average_cost",
        "final_cost_price",
        "market_val",
        "pl_ratio",
        "pl_ratio_avg_cost",
        "unrealized_pl_ratio",
        "unrealized_pl_sum",
        "calculated_realized_pl_ratio",
        "calculated_realized_pl_sum",
        "calculated_peak_exposure",
        "calculated_pl_sum",
        "total_assets",
        "asset_difference",
        "asset_difference_ratio",
    ]

    return df[columns]


def save_results(
    strategy,
    trade_ctx,
    config,
):
    """Save trading logs, P/L, and daily status."""

    output_df = pd.DataFrame(
        strategy.output
    )

    if output_df.empty:
        return

    output_df["time_SG"] = (
        pd.to_datetime(
            output_df["time"]
        )
        .dt.tz_localize(
            "America/New_York"
        )
        .dt.tz_convert(
            "Asia/Singapore"
        )
    )

    if config.live_mode:
        mode = "live"
        price_column = "execution_price"

    else:
        mode = "backtest"
        price_column = "close"

    trading_logs_file_name = (
        f"{mode}_{config.env_name}"
        "_trading_logs.csv"
    )

    pl_file_name = (
        f"{mode}_{config.env_name}"
        "_pl.csv"
    )

    daily_status_file_name = (
        f"{mode}_{config.env_name}"
        "_daily_status.csv"
    )

    logs_folder = os.path.join(
        os.getcwd(),
        "logs",
    )

    os.makedirs(
        logs_folder,
        exist_ok=True,
    )

    timestamp = (
        pd.Timestamp.today().strftime(
            "%Y-%m-%d %H_%M_%S"
        )
    )

    trading_logs_path = os.path.join(
        logs_folder,
        f"{timestamp} - "
        f"{trading_logs_file_name}",
    )

    output_df.to_csv(
        trading_logs_path,
        index=False,
    )

    (
        sell_df,
        realized_pl_sum,
        peak_exposure,
        realized_pl_pct,
    ) = compute_daily_pl(
        output_df,
        price_column,
    )

    pl_path = os.path.join(
        logs_folder,
        f"{timestamp} - {pl_file_name}",
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
        logs_folder=logs_folder,
        daily_status_file_name=(
            daily_status_file_name
        ),
    )

    daily_status_path = os.path.join(
        logs_folder,
        f"{timestamp} - "
        f"{daily_status_file_name}",
    )

    daily_status.to_csv(
        daily_status_path,
        index=False,
    )