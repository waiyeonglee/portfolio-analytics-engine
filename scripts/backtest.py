import pandas as pd

from .broker import (
    get_market_trend_simulation
)

from .strategy import (
    LONG_WINDOW,
    round_down_to_lot,
)

from .initialization import (
    load_historical_candles,
    initialize_portfolio
)

def initialize_backtest(
    strategy,
    trade_ctx,
    quote_ctx,
    runtime_settings,
):
    """Initialize historical backtesting."""

    market_datetime = runtime_settings.market_datetime

    if market_datetime.time() >= pd.Timestamp(
        "16:00"
    ).time():
        session_date = pd.Timestamp(
            market_datetime.date()
        )
    else:
        session_date = (
            pd.Timestamp(market_datetime.date())
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

    historical_df = load_historical_candles(
        quote_ctx,
        runtime_settings,
        session_date,
    )

    warmup_df = historical_df.loc[
        (historical_df["time_key"] >= warmup_start)
        & (historical_df["time_key"] < session_start)
    ].copy()

    df_past = warmup_df.iloc[
        -(LONG_WINDOW):
    ].copy()

    df_current = historical_df.loc[
        (historical_df["time_key"] >= session_start)
        & (historical_df["time_key"] <= session_end)
    ].copy()

    if df_past.empty or df_current.empty:
        raise RuntimeError(
            "Insufficient historical candles for backtest."
        )

    initialize_portfolio(
        strategy=strategy,
        trade_ctx=trade_ctx,
        runtime_settings=runtime_settings,
        current_price=float(df_past["close"].iloc[-1]),
        mode="backtest",
        initial_cash=1000.0,
    )

    if len(df_past) < LONG_WINDOW:
        raise RuntimeError(
            "Insufficient candles for indicator warm-up."
        )

    for _, row in df_past.iterrows():
        strategy.process_candle(
            row=row,
            market_trend=0.0,
            max_cash_buy=strategy.max_cash_buy,
            max_position_sell=strategy.max_position_sell,
            init=True,
        )

    print(
        "Initialized time:",
        df_past["time_key"].iloc[-1],
    )

    df_market = get_market_trend_simulation(
        quote_ctx,
        runtime_settings,
        session_date,
    )

    return df_current, df_market

def execute_backtest(
    strategy,
    df_current,
    df_market,
):
    """Run candle-by-candle historical simulation."""

    market_lookup = (
        df_market
        .set_index("time_key")["market_trend"]
    )

    def execute_order(action, qty, price):
        """Simulate an immediate order fill."""

        if qty <= 0:
            return False, None

        trade_value = price * qty

        if action == "BUY":
            if trade_value > strategy.cash:
                return False, None

            strategy.cash -= trade_value

        elif action == "SELL":
            if qty > strategy.position_qty:
                return False, None

            strategy.cash += trade_value

        else:
            return False, None

        strategy.apply_fill(
            action=action,
            price=price,
            qty=qty,
        )

        return True, None

    for _, row in df_current.iterrows():

        curr_time = row["time_key"]
        current_price = float(row["close"])

        if curr_time not in market_lookup.index:
            print(
                f"Market trend not found: {curr_time}"
            )
            continue

        market_trend = float(
            market_lookup.loc[curr_time]
        )

        max_cash_buy = round_down_to_lot(
            strategy.cash / current_price,
            strategy.lot_size,
        )

        max_position_sell = strategy.position_qty

        strategy.process_candle(
            row=row,
            market_trend=market_trend,
            max_cash_buy=max_cash_buy,
            max_position_sell=max_position_sell,
            execute_order=execute_order,
        )

def run_backtest(
    strategy,
    quote_ctx,
    trade_ctx,
    runtime_settings,
):
    """Prepare and run a complete backtest."""

    (
        df_current,
        df_market,
    ) = initialize_backtest(
        strategy,
        trade_ctx,
        quote_ctx,
        runtime_settings,
    )

    execute_backtest(
        strategy,
        df_current,
        df_market,
    )