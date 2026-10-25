import pandas as pd

from moomoo import (
    RET_OK,
    AuType,
    Session,
    SubType,
    TrdEnv
)
from .broker import (
    get_available_qty,
    get_portfolio_equity,
    get_position_status,
)

def load_historical_candles(
    quote_ctx,
    runtime_settings,
    session_date,
):
    """Load and sort historical 1-minute candles."""

    api_date = session_date.strftime("%Y-%m-%d")

    ret, df, _ = quote_ctx.request_history_kline(
        runtime_settings.symbol,
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
    runtime_settings,
    current_price,
    mode,
    initial_cash=1000.0,
):
    """Initialize portfolio state for live or backtest."""

    if mode == "backtest":
        strategy.initial_equity = float(initial_cash)
        strategy.cash = float(initial_cash)

        strategy.initial_position_qty = 0
        strategy.initial_cost_price = 0.0
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
            runtime_settings,
            currency=runtime_settings.currency,
        )
        
        # strategy.cash not used in live mode

        position_df = get_position_status(
            trade_ctx,
            runtime_settings,
        )
        
        strategy.initial_position_qty =  int(position_df["qty"].iloc[0])
        strategy.position_qty = strategy.initial_position_qty
        
        if runtime_settings.trade_env == TrdEnv.REAL:
            cost_column = "average_cost"
        else:
            cost_column = "cost_price"

        strategy.cost_price = float(
            pd.to_numeric(
                position_df[cost_column],
                errors="coerce",
            ).iloc[0]
        )
        strategy.initial_cost_price = strategy.cost_price

        (
            strategy.max_cash_buy,
            strategy.max_position_sell,
        ) = get_available_qty(
            trade_ctx,
            runtime_settings,
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