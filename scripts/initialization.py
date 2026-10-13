import pandas as pd

from moomoo import (
    RET_OK,
    AuType,
    Session,
    SubType,
)
from broker import (
    get_available_qty,
    get_portfolio_equity,
    get_position_status,
)

from strategy import (
    WINDOW_LENGTH
)

def load_historical_candles(
    quote_ctx,
    config,
    session_date,
):
    """Load and sort historical 1-minute candles."""

    api_date = session_date.strftime("%Y-%m-%d")

    ret, df, _ = quote_ctx.request_history_kline(
        config.symbol,
        api_date,
        api_date,
        SubType.K_1M,
        AuType.NONE,
        session=Session.ALL,
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching historical data: {df}"
        )

    df["time_key"] = pd.to_datetime(
        df["time_key"]
    )

    return (
        df.sort_values("time_key")
        .reset_index(drop=True)
    )

def initialize_portfolio(
    strategy,
    trade_ctx,
    config,
    current_price,
    mode,
    initial_cash=1000.0,
):
    """Initialize portfolio state for live or backtest."""

    if mode == "backtest":
        strategy.initial_equity = float(initial_cash)
        strategy.cash = float(initial_cash)
        
        strategy.position_qty = 0
        strategy.cost_price = 0.0

        strategy.max_cash_buy = (
            int(strategy.initial_equity // current_price)
            // strategy.lot_size
            * strategy.lot_size
        )

        strategy.max_position_sell = 0

    elif mode == "live":
        strategy.initial_equity = get_portfolio_equity(
            trade_ctx,
            config,
        )

        (
            strategy.position_qty,
            strategy.cost_price,
        ) = get_position_status(
            trade_ctx,
            config,
        )

        (
            strategy.max_cash_buy,
            strategy.max_position_sell,
        ) = get_available_qty(
            trade_ctx,
            config,
            current_price,
        )

    else:
        raise ValueError(
            f"Unsupported initialization mode: {mode}"
        )

    strategy.position_open = (
        strategy.position_qty > 0
    )

    strategy.total_price = (
        strategy.cost_price
        * strategy.position_qty
    )

def warmup_strategy(
    strategy,
    df_past,
):
    """Warm up strategy indicators without trading."""

    if len(df_past) < WINDOW_LENGTH - 1:
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