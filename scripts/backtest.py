import pandas as pd

from moomoo import (
    RET_OK,
    AuType,
    Session,
    SubType,
)

from broker import (
    get_available_qty,
    get_market_trend_simulation,
    get_position_status,
)

from strategy import (
    WINDOW_LENGTH,
    round_down_to_lot,
)


def initialize_backtest(
    strategy,
    trade_ctx,
    quote_ctx,
    config,
):
    """Load historical data and initialize backtest state."""

    timezone_date = config.timezone_date

    # ---------------------------------------------------------
    # Determine backtest session
    # ---------------------------------------------------------

    if timezone_date.time() >= pd.Timestamp(
        "16:00"
    ).time():
        session_date = pd.Timestamp(
            timezone_date.date()
        )
    else:
        session_date = (
            pd.Timestamp(timezone_date.date())
            - pd.offsets.BDay(1)
        )

    warmup_start = session_date.replace(
        hour=8,
        minute=30,
    )

    session_start = session_date.replace(
        hour=9,
        minute=30,
    )

    session_end = session_date.replace(
        hour=16,
        minute=0,
    )

    api_date = session_date.strftime(
        "%Y-%m-%d"
    )

    # ---------------------------------------------------------
    # Load symbol candles
    # ---------------------------------------------------------

    ret, historical_df, _ = (
        quote_ctx.request_history_kline(
            config.symbol,
            api_date,
            api_date,
            SubType.K_1M,
            AuType.NONE,
            session=Session.ALL,
        )
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching historical data: "
            f"{historical_df}"
        )

    historical_df["time_key"] = (
        pd.to_datetime(
            historical_df["time_key"]
        )
    )

    historical_df = (
        historical_df
        .sort_values("time_key")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Warm-up candles
    # ---------------------------------------------------------

    warmup_df = historical_df.loc[
        (
            historical_df["time_key"]
            >= warmup_start
        )
        & (
            historical_df["time_key"]
            < session_start
        )
    ].copy()

    df_past = warmup_df.iloc[
        -(WINDOW_LENGTH - 1):
    ].copy()

    if len(df_past) < WINDOW_LENGTH - 1:
        raise RuntimeError(
            "Not enough historical candles to "
            "initialize strategy. "
            f"Need {WINDOW_LENGTH - 1}, "
            f"got {len(df_past)}."
        )

    # ---------------------------------------------------------
    # Actual backtest candles
    # ---------------------------------------------------------

    df_current = historical_df.loc[
        (
            historical_df["time_key"]
            >= session_start
        )
        & (
            historical_df["time_key"]
            <= session_end
        )
    ].copy()

    if df_current.empty:
        raise RuntimeError(
            "No regular-session candles "
            "available for backtest."
        )

    # ---------------------------------------------------------
    # Initialize portfolio
    # ---------------------------------------------------------

    initial_price = float(
        df_past["close"].iloc[-1]
    )

    max_cash_buy, _ = get_available_qty(
        trade_ctx,
        config,
        initial_price,
    )

    (
        strategy.position_qty,
        strategy.cost_price,
    ) = get_position_status(
        trade_ctx,
        config,
    )

    strategy.position_qty = int(
        strategy.position_qty
    )

    strategy.cost_price = float(
        strategy.cost_price
    )

    strategy.position_open = (
        strategy.position_qty > 0
    )

    strategy.total_price = (
        strategy.cost_price
        * strategy.position_qty
    )

    # Convert broker buying capacity into simulated cash.
    strategy.cash = (
        float(max_cash_buy)
        * initial_price
    )

    strategy.max_cash_buy = int(
        max_cash_buy
    )

    strategy.max_position_sell = (
        strategy.position_qty
    )

    # ---------------------------------------------------------
    # Warm indicators
    # ---------------------------------------------------------

    for _, row in df_past.iterrows():

        strategy.update_state_from_row(
            row,
            init=True,
        )

        current_price = float(
            row["close"]
        )

        strategy.market_trend = 0.0

        strategy.unrealized_pl_pct = (
            strategy.compute_pl(
                current_price
            )
        )

        strategy.trade_qty = 0
        strategy.realized_pl_pct = 0.0

        strategy.save_output(
            row,
            "INITIALIZING",
            order_data=None,
        )

    # ---------------------------------------------------------
    # Load benchmark data
    # ---------------------------------------------------------

    df_market = get_market_trend_simulation(
        quote_ctx,
        config,
        session_date,
    )

    print(
        "Initialized time:",
        df_past["time_key"].iloc[-1],
    )

    print(
        "Initial simulated cash:",
        strategy.cash,
    )

    print(
        "Initial position:",
        strategy.position_qty,
    )

    return (
        df_current,
        df_market,
    )

def execute_backtest(
    strategy,
    df_current,
    df_market,
):
    """Run the candle-by-candle backtest."""

    for _, row in df_current.iterrows():

        candle_price = float(
            row["close"]
        )

        curr_time = row["time_key"]

        # -----------------------------------------------------
        # Historical market trend
        # -----------------------------------------------------

        market_row = df_market.loc[
            df_market["time_key"] == curr_time
        ]

        if market_row.empty:
            print(
                "Market trend data not found "
                f"for time: {curr_time}"
            )
            continue

        market_trend = float(
            market_row[
                "market_trend"
            ].iloc[0]
        )

        # -----------------------------------------------------
        # Simulated account capacity
        # -----------------------------------------------------

        max_cash_buy = (
            round_down_to_lot(
                strategy.cash
                / candle_price,
                strategy.lot_size,
            )
        )

        max_position_sell = (
            strategy.position_qty
        )

        # -----------------------------------------------------
        # Shared live/backtest strategy pipeline
        # -----------------------------------------------------

        (
            action,
            buy_qty,
            sell_qty,
            current_price,
        ) = strategy.process_candle(
            row=row,
            market_trend=market_trend,
            max_cash_buy=max_cash_buy,
            max_position_sell=max_position_sell,
        )

        # -----------------------------------------------------
        # Simulated BUY
        # -----------------------------------------------------

        if action == "BUY":

            trade_value = (
                current_price
                * buy_qty
            )

            if (
                buy_qty > 0
                and trade_value
                <= strategy.cash
            ):
                strategy.cash -= (
                    trade_value
                )

                strategy.apply_fill(
                    action="BUY",
                    price=current_price,
                    qty=buy_qty,
                )

            else:
                action = "HOLD"
                strategy.reset_trade_state()

        # -----------------------------------------------------
        # Simulated SELL
        # -----------------------------------------------------

        elif action == "SELL":

            if (
                sell_qty > 0
                and sell_qty
                <= strategy.position_qty
            ):
                strategy.cash += (
                    current_price
                    * sell_qty
                )

                strategy.apply_fill(
                    action="SELL",
                    price=current_price,
                    qty=sell_qty,
                )

            else:
                action = "HOLD"
                strategy.reset_trade_state()

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        strategy.save_output(
            row,
            action,
            order_data=None,
        )

def run_backtest(
    strategy,
    quote_ctx,
    trade_ctx,
    config,
):
    """Prepare and run a complete backtest."""

    (
        df_current,
        df_market,
    ) = initialize_backtest(
        strategy,
        trade_ctx,
        quote_ctx,
        config,
    )

    execute_backtest(
        strategy,
        df_current,
        df_market,
    )