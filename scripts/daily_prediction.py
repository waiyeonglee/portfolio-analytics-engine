import argparse

import numpy as np
import pandas as pd
from moomoo import RET_OK, AuType, Session, SubType
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .broker import (
    close_contexts,
    create_contexts,
    get_live_market_snapshot,
    get_position_status,
)
from .runtime import create_runtime

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "sma_5_gap",
    "sma_20_gap",
]
HISTORY_YEARS = 3
MAX_HISTORY_ROWS = 1000
TRAINING_WINDOW = 252
MINIMUM_TRAINING_ROWS = 60


def make_daily_features(daily_df):
    """Build daily technical features and next-session return targets."""

    df = daily_df.copy()
    df["time_key"] = pd.to_datetime(df["time_key"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("time_key").reset_index(drop=True)

    if df["time_key"].duplicated().any():
        raise ValueError("Duplicate daily candle timestamps")
    if not np.isfinite(df["close"]).all() or (df["close"] <= 0).any():
        raise ValueError("Daily closing prices must be positive and finite")

    close = df["close"]
    daily_return = close.pct_change(fill_method=None)
    df["return_1d"] = daily_return
    df["return_5d"] = close.pct_change(periods=5, fill_method=None)
    df["return_20d"] = close.pct_change(periods=20, fill_method=None)
    df["volatility_20d"] = daily_return.rolling(window=20).std()
    df["sma_5_gap"] = close / close.rolling(window=5).mean() - 1
    df["sma_20_gap"] = close / close.rolling(window=20).mean() - 1
    df["target"] = close.shift(-1) / close - 1

    df[FEATURES + ["target"]] = df[FEATURES + ["target"]].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def predict_daily_return(daily_df, current_price, as_of):
    """Fit Ridge on completed daily candles and forecast from the current quote."""

    current_price = float(current_price)
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current price must be a positive finite number")

    history = make_daily_features(daily_df)
    training = history.dropna(subset=FEATURES + ["target"]).tail(TRAINING_WINDOW)
    if len(training) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "Not enough completed daily history to train a forecast: "
            f"{len(training)}/{MINIMUM_TRAINING_ROWS} observations"
        )

    as_of = pd.to_datetime(as_of)
    prediction_history = history[["time_key", "close"]].copy()
    latest_time = prediction_history["time_key"].iloc[-1]
    if as_of.date() == latest_time.date():
        prediction_history.loc[prediction_history.index[-1], "close"] = current_price
    elif as_of <= latest_time:
        raise ValueError("Prediction time must not precede the latest daily candle")
    else:
        prediction_history = pd.concat(
            [
                prediction_history,
                pd.DataFrame([{"time_key": as_of, "close": current_price}]),
            ],
            ignore_index=True,
        )

    prediction_data = make_daily_features(prediction_history)
    latest = prediction_data[FEATURES].iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError("Could not calculate features for the current quote")

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(training[FEATURES], training["target"])
    return float(model.predict(latest)[0]), len(training)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forecast the next daily return of the configured Moomoo position."
    )
    parser.add_argument(
        "--env",
        choices=["real", "simulate"],
        default="simulate",
        help="Moomoo trading environment (default: simulate).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_settings = create_runtime(
        argparse.Namespace(
            env=args.env,
            date=pd.Timestamp.now(),
            live=False,
        )
    )

    quote_ctx = None
    trade_ctx = None
    try:
        quote_ctx, trade_ctx, _ = create_contexts(runtime_settings)
        position = get_position_status(trade_ctx, runtime_settings)
        quantity = int(position["qty"].iloc[0])
        if quantity <= 0:
            print(f"No open position in {runtime_settings.symbol}.")
            return

        snapshot = get_live_market_snapshot(quote_ctx, [runtime_settings.symbol])
        if snapshot is None:
            raise RuntimeError(
                f"Could not retrieve a live quote for {runtime_settings.symbol}"
            )
        quote = snapshot.loc[snapshot["code"] == runtime_settings.symbol]
        if quote.empty:
            raise RuntimeError(f"No live quote returned for {runtime_settings.symbol}")

        current_price = float(
            pd.to_numeric(quote["last_price"], errors="coerce").iloc[0]
        )
        if not np.isfinite(current_price) or current_price <= 0:
            raise RuntimeError(
                f"Invalid live price for {runtime_settings.symbol}: {current_price}"
            )

        market_now = pd.Timestamp.now(tz=runtime_settings.market_datetime.tz)
        start = (market_now - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
        end = market_now.strftime("%Y-%m-%d")
        ret, daily_df, _ = quote_ctx.request_history_kline(
            runtime_settings.symbol,
            start,
            end,
            ktype=SubType.K_DAY,
            autype=AuType.NONE,
            max_count=MAX_HISTORY_ROWS,
            session=Session.ALL,
        )
        if ret != RET_OK:
            raise RuntimeError(
                f"Error fetching daily history for {runtime_settings.symbol}: "
                f"{daily_df}"
            )

        daily_df["time_key"] = pd.to_datetime(daily_df["time_key"])
        completed_history = daily_df.loc[
            daily_df["time_key"].dt.date < market_now.date(),
            ["time_key", "close"],
        ].copy()
        if completed_history.empty:
            raise RuntimeError(
                f"No completed daily candles returned for {runtime_settings.symbol}"
            )

        predicted_return, training_rows = predict_daily_return(
            daily_df=completed_history,
            current_price=current_price,
            as_of=market_now.tz_localize(None),
        )
        current_value = quantity * current_price
        estimated_change = current_value * predicted_return

        print(f"Position: {runtime_settings.symbol} x {quantity}")
        print(f"Current price: {current_price:.4f} {runtime_settings.currency}")
        print(
            f"Predicted next daily return: {predicted_return:+.4%} "
            f"(Ridge, {training_rows} training observations)"
        )
        print(
            f"Estimated position value change: "
            f"{estimated_change:+,.2f} {runtime_settings.currency}"
        )
    finally:
        close_contexts(quote_ctx, trade_ctx)


if __name__ == "__main__":
    main()
