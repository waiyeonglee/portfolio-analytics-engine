import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

from .broker import get_portfolio_equity, get_position_status

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
        "&cosd=2026-01-01"
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

    print("Daily Risk Free Date:", available["DATE"].iloc[-1].strftime("%Y-%m-%d"))
    print("Daily Risk Free Rate:", available["daily_rf"].iloc[-1])
    return float(available["daily_rf"].iloc[-1])

def calculate_intraday_sharpe(
    output_df,
    initial_cash,
    initial_position_qty=0,
    daily_rf=0.0,
    periods_per_day=390,
):
    """Calculate intraday Sharpe and return the equity DataFrame."""

    df = output_df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        format="mixed",
    )

    df = df.sort_values("time").reset_index(drop=True)

    cash = float(initial_cash)
    initial_position_qty = float(initial_position_qty)

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
            initial_position_qty += qty

        elif action == "SELL":
            cash += qty * execution_price - fee
            initial_position_qty -= qty

        cash_history.append(cash)
        position_history.append(initial_position_qty)
        equity_history.append(
            cash + initial_position_qty * price
        )

    df["cash"] = cash_history
    df["position_qty"] = position_history
    df["equity"] = equity_history

    trading_mask = df["action"] != "INITIALIZING"

    df["returns"] = np.nan

    df.loc[trading_mask, "returns"] = (
        df.loc[trading_mask, "equity"]
        .pct_change(fill_method=None)
    )

    df["excess_return"] = np.nan

    df.loc[trading_mask, "excess_return"] = (
        df.loc[trading_mask, "returns"]
        - daily_rf / periods_per_day
    )

    excess = df.loc[
        trading_mask, "excess_return"
    ].dropna()

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
    initial_position_qty=0,
    initial_cost_price=0.0,
):
    """Calculate realized daily P/L and peak exposure."""

    requested_qty = pd.to_numeric(
        output_df.get(
            "trade_qty",
            pd.Series(0.0, index=output_df.index),
        ),
        errors="coerce",
    )
    filled_qty = pd.to_numeric(
        output_df.get(
            "filled_qty",
            pd.Series(index=output_df.index, dtype=float),
        ),
        errors="coerce",
    )
    if "filled_qty" not in output_df:
        reported_qty = requested_qty
    elif "order_id" in output_df:
        # A submitted order with zero fills must not fall back to requested size.
        legacy_or_backtest = output_df["order_id"].isna()
        reported_qty = filled_qty.where(
            filled_qty > 0,
            requested_qty.where(legacy_or_backtest, 0),
        )
    else:
        reported_qty = filled_qty.where(
            filled_qty > 0,
            requested_qty,
        )
    reported_qty = reported_qty.fillna(0)

    actions = output_df["action"]
    prices = pd.to_numeric(
        output_df.get(
            price_column,
            pd.Series(index=output_df.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(0)
    cost_basis = pd.to_numeric(
        output_df.get(
            "cost_price",
            pd.Series(index=output_df.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(0)

    exposure_change = pd.Series(0.0, index=output_df.index)
    buy_mask = actions == "BUY"
    sell_mask = actions == "SELL"
    exposure_change.loc[buy_mask] = (
        prices.loc[buy_mask] * reported_qty.loc[buy_mask]
    )
    exposure_change.loc[sell_mask] = (
        -cost_basis.loc[sell_mask] * reported_qty.loc[sell_mask]
    )
    starting_exposure = (
        float(initial_position_qty) * float(initial_cost_price)
    )
    peak_exposure = max(
        starting_exposure,
        starting_exposure + exposure_change.cumsum().max(),
    )

    print(
        f"Peak Exposure: "
        f"{peak_exposure:.0f}"
    )

    sell_df = output_df.loc[sell_mask].copy()
    sell_df["reported_qty"] = reported_qty.loc[sell_mask]

    gross_realized = pd.to_numeric(
        sell_df.get(
            "realized_pl_gross",
            pd.Series(index=sell_df.index, dtype=float),
        ),
        errors="coerce",
    )
    filled_sell_qty = filled_qty.loc[sell_mask].fillna(0)
    execution_price = pd.to_numeric(
        sell_df.get(
            price_column,
            pd.Series(index=sell_df.index, dtype=float),
        ),
        errors="coerce",
    )
    sell_cost_basis = cost_basis.loc[sell_mask]
    fallback_realized = (
        (execution_price - sell_cost_basis)
        * sell_df["reported_qty"]
    )
    has_fill_level_realized = (
        gross_realized.notna()
        & (
            (filled_sell_qty > 0)
            | gross_realized.ne(0)
        )
    )
    sell_df["realized_pl_gross"] = gross_realized.where(
        has_fill_level_realized,
        fallback_realized,
    ).fillna(0)

    sell_fees = pd.to_numeric(
        sell_df.get(
            "fee_amount",
            pd.Series(0.0, index=sell_df.index),
        ),
        errors="coerce",
    ).fillna(0)
    sell_df["realized_pl"] = (
        sell_df["realized_pl_gross"] - sell_fees
    )

    realized_pl_sum = sell_df["realized_pl"].sum()

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
        realized_pl_pct
    )

def get_daily_status(
    trade_ctx,
    runtime_settings,
    realized_pl_sum,
    logs_folder,
    daily_status_file_name,
):
    """Build a concise daily portfolio and P/L report."""

    df = get_position_status(
        trade_ctx,
        runtime_settings,
    )

    cost_basis = pd.to_numeric(df["average_cost"], errors="coerce")
    df["position_pl_sum"] = (
        pd.to_numeric(df["nominal_price"], errors="coerce") - cost_basis
    ) * pd.to_numeric(df["qty"], errors="coerce")

    df["date"] = runtime_settings.market_datetime
    df["total_assets"] = get_portfolio_equity(
        trade_ctx,
        runtime_settings,
        currency="USD",
    )

    # Load prior snapshots so cumulative values survive restarts.
    files = Path(logs_folder).glob(f"*{daily_status_file_name}")
    previous_status = pd.DataFrame()
    snapshots = []
    for path in files:
        snapshot = pd.read_csv(path)
        if not snapshot.empty and "date" in snapshot:
            snapshot["_snapshot_order"] = path.stat().st_mtime
            snapshots.append(snapshot)

    if snapshots:
        previous_status = pd.concat(snapshots, ignore_index=True)
        previous_status["_date_key"] = pd.to_datetime(
            previous_status["date"].astype(str).str.slice(0, 10),
            errors="coerce",
        ).dt.date
        previous_status = (
            previous_status
            .dropna(subset=["_date_key"])
            .sort_values("_snapshot_order")
            .drop_duplicates("_date_key", keep="last")
        )

    current_date = pd.Timestamp(df["date"].iloc[0]).date()
    if previous_status.empty:
        prior_days = same_day = previous_status
    else:
        prior_days = previous_status.loc[
            previous_status["_date_key"] < current_date
        ].copy()
        same_day = previous_status.loc[
            previous_status["_date_key"] == current_date
        ]

    prior_realized = pd.to_numeric(
        same_day.get(
            "daily_realized_pl_sum",
            pd.Series(index=same_day.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(
        pd.to_numeric(
            same_day.get(
                "calculated_realized_pl_sum",
                pd.Series(0.0, index=same_day.index),
            ),
            errors="coerce",
        )
    ).fillna(0).sum()
    df["daily_realized_pl_sum"] = prior_realized + realized_pl_sum

    df = pd.concat(
        [prior_days, df],
        ignore_index=True,
        sort=False,
    )
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.slice(0, 10),
        errors="coerce",
    ).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    for column, legacy_column in (
        ("daily_realized_pl_sum", "calculated_realized_pl_sum"),
        ("unrealized_pl_sum", "position_pl_sum"),
    ):
        current_values = pd.to_numeric(
            df.get(column, pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        )
        legacy_values = pd.to_numeric(
            df.get(legacy_column, pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        )
        df[column] = current_values.fillna(legacy_values)

    df["daily_unrealized_pl_change"] = (
        df["unrealized_pl_sum"].diff()
    )
    df["daily_total_pl_sum"] = (
        df["daily_realized_pl_sum"]
        + df["daily_unrealized_pl_change"]
    )
    df["asset_change"] = pd.to_numeric(
        df["total_assets"],
        errors="coerce",
    ).diff()
    df["reconciliation_difference"] = (
        df["asset_change"] - df["daily_total_pl_sum"]
    )

    output_columns = [
        "date",
        "code",
        "qty",
        "total_assets",
        "daily_realized_pl_sum",
        "unrealized_pl_sum",
        "daily_unrealized_pl_change",
        "daily_total_pl_sum",
        "asset_change",
        "reconciliation_difference",
    ]

    return df[output_columns]

def save_results(
    strategy,
    trade_ctx,
    runtime_settings,
):
    """Save trading logs, P/L, and daily status."""

    logs_folder = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_folder,exist_ok=True)
    
    if runtime_settings.live_mode:
        mode = "live"
        price_column = "execution_price"
        timestamp = (pd.Timestamp.today().strftime("%Y-%m-%d %H_%M_%S"))

    else:
        mode = "backtest"
        price_column = "close"
        timestamp = runtime_settings.market_datetime.strftime("%Y-%m-%d %H_%M_%S")

    # Filenames for logs
    trading_logs_file_name = (f"{mode}_{runtime_settings.env_name}""_trading_logs.csv")
    pl_file_name = (f"{mode}_{runtime_settings.env_name}""_pl.csv")
    daily_status_file_name = (f"{mode}_{runtime_settings.env_name}""_daily_status.csv")

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
        runtime_settings.market_datetime
    )
    
    output_df, intraday_sharpe = calculate_intraday_sharpe(
        output_df=output_df,
        initial_cash=strategy.initial_equity,
        initial_position_qty=strategy.initial_position_qty,
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
        initial_position_qty=strategy.initial_position_qty,
        initial_cost_price=strategy.initial_cost_price,
    )

    sell_df.to_csv(
        pl_path,
        index=False,
    )

    daily_status = get_daily_status(
        trade_ctx=trade_ctx,
        runtime_settings=runtime_settings,
        realized_pl_sum=realized_pl_sum,
        logs_folder=logs_folder,
        daily_status_file_name=(
            daily_status_file_name
        ),
    )

    daily_status.to_csv(
        daily_status_path,
        index=False,
    )