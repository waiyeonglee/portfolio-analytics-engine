import pandas as pd

from moomoo import (
    RET_OK,
    AuType,
    Session,
    SubType,
)

from scripts.broker import (
    get_available_qty,
    get_market_trend_simulation,
)
from scripts.strategy import WINDOW_LENGTH


def initialize_backtest(
    strategy,
    trade_ctx,
    quote_ctx,
    config,
):
    """Load historical candles and initialize strategy state."""

    timezone_date = config.timezone_date

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

    session_start = session_date.replace(
        hour=9,
        minute=30,
    )

    session_end = session_date.replace(
        hour=16,
        minute=0,
    )

    api_start = session_start.strftime(
        "%Y-%m-%d"
    )

    api_end = session_end.strftime(
        "%Y-%m-%d"
    )

    ret, historical_df, _ = (
        quote_ctx.request_history_kline(
            config.symbol,
            api_start,
            api_end,
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

    historical_df["time_key"] = pd.to_datetime(
        historical_df["time_key"]
    )

    pre_session_start = session_start.replace(
        hour=9,
        minute=0,
    )

    pre_session_end = session_start.replace(
        hour=9,
        minute=30,
    )

    prev_session_df = historical_df.loc[
        (
            historical_df["time_key"]
            >= pre_session_start
        )
        & (
            historical_df["time_key"]
            < pre_session_end
        )
    ].copy()

    df_past = prev_session_df.iloc[
        -WINDOW_LENGTH + 1:
    ].copy()

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

    if df_past.empty:
        raise RuntimeError(
            "Not enough historical candles "
            "to initialize strategy."
        )

    for i, (_, row) in enumerate(
        df_past.iterrows()
    ):
        strategy.update_state_from_row(
            row,
            init=True,
        )

        current_price = strategy.prices[-1]

        strategy.market_trend = 0

        if i == 0:
            (
                max_cash_buy,
                max_position_sell,
            ) = get_available_qty(
                trade_ctx,
                config,
                current_price,
            )

            # In simulation, treat existing position
            # capacity as available starting capital.
            strategy.max_cash_buy = (
                max_cash_buy
                + max_position_sell
            )

            strategy.max_position_sell = 0

        strategy.cost_price = 0
        strategy.unrealized_pl_pct = 0
        strategy.position_open = False
        strategy.trade_qty = 0
        strategy.realized_pl_pct = 0

        strategy.total_price = (
            strategy.cost_price
            * strategy.max_position_sell
        )

        strategy.save_output(
            row,
            "INITIALIZING",
            order_data=None,
        )

    print(
        "Initialized time:",
        df_past["time_key"].iloc[-1],
    )

    return df_current, session_end


def execute_backtest(
    strategy,
    df_current,
    df_market,
    config,
):
    """Run the candle-by-candle backtest."""

    next_max_cash_buy = (
        strategy.max_cash_buy
    )

    next_max_position_sell = (
        strategy.max_position_sell
    )

    for _, row in df_current.iterrows():
        strategy.update_state_from_row(
            row,
            init=False,
        )

        current_price = float(
            row["close"]
        )

        curr_time = row["time_key"]

        market_row = df_market.loc[
            df_market["time_key"] == curr_time
        ]

        if market_row.empty:
            print(
                "Market trend data not found "
                f"for time: {curr_time}"
            )
            continue

        strategy.market_trend = (
            market_row["close"].iloc[0]
            - market_row["open"].iloc[0]
        )

        strategy.max_cash_buy = (
            next_max_cash_buy
        )

        strategy.max_position_sell = (
            next_max_position_sell
        )

        if strategy.max_position_sell == 0:
            strategy.cost_price = 0

        strategy.unrealized_pl_pct = (
            strategy.compute_pl(
                current_price
            )
        )

        (
            action,
            buy_qty,
            sell_qty,
        ) = strategy.buy_or_sell(
            strategy.unrealized_pl_pct
        )

        if action == "BUY":
            next_max_cash_buy -= buy_qty
            next_max_position_sell += buy_qty

            strategy.apply_fill(
                action="BUY",
                price=current_price,
                qty=buy_qty,
                position_qty_after=(
                    next_max_position_sell
                ),
            )

            print(
                f"BUY | {strategy.trade_qty} "
                f"{config.symbol} "
                f"| Cost: "
                f"{strategy.cost_price:.2f}"
            )

        elif action == "SELL":
            next_max_cash_buy += sell_qty
            next_max_position_sell -= sell_qty

            strategy.apply_fill(
                action="SELL",
                price=current_price,
                qty=sell_qty,
                position_qty_after=(
                    next_max_position_sell
                ),
            )

            print(
                f"SELL | {strategy.trade_qty} "
                f"{config.symbol} "
                f"| Cost: "
                f"{strategy.cost_price:.2f} "
                f"| Profit: "
                f"{strategy.realized_pl_pct:.2f}"
            )

        else:
            strategy.reset_trade_state()

        strategy.position_open = (
            next_max_position_sell > 0
        )

        print(
            f"Current time: {row['time_key']}, "
            f"Current price: {current_price}, "
            f"Unrealized P/L: "
            f"{strategy.unrealized_pl_pct}"
        )

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

    df_current, last_day = initialize_backtest(
        strategy,
        trade_ctx,
        quote_ctx,
        config,
    )

    df_market = get_market_trend_simulation(
        quote_ctx,
        config,
        last_day,
    )

    execute_backtest(
        strategy,
        df_current,
        df_market,
        config,
    )