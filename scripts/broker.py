import pandas as pd

from moomoo import (
    RET_OK,
    AuType,
    OpenQuoteContext,
    OpenSecTradeContext,
    OrderType,
    SecurityFirm,
    SecurityType,
    Session,
    SubType,
    TrdEnv,
    TrdMarket,
)

from config import pwd_unlock


def get_trend_symbol(symbol):
    """Return the benchmark used to determine broad market direction."""

    if symbol.startswith("HK."):
        return "HK.800000"

    if symbol.startswith("US."):
        return "US.SPY"

    raise ValueError(f"Unsupported symbol: {symbol}")


def get_market_config(symbol):
    """Return Moomoo market enum and global-state key."""

    if symbol.startswith("HK."):
        return TrdMarket.HK, "market_hk"

    if symbol.startswith("US."):
        return TrdMarket.US, "market_us"

    raise ValueError(f"Unsupported symbol: {symbol}")


def create_contexts(config):
    """Create and initialize Moomoo quote/trade contexts."""

    trade_market, _ = get_market_config(config.symbol)

    quote_ctx = OpenQuoteContext(
        host="127.0.0.1",
        port=11111,
    )

    trade_ctx = OpenSecTradeContext(
        filter_trdmarket=trade_market,
        host="127.0.0.1",
        port=11111,
        security_firm=SecurityFirm.FUTUSG,
    )

    if config.trade_env == TrdEnv.REAL:
        ret, data = trade_ctx.unlock_trade(
            password_md5=pwd_unlock
        )

        if ret != RET_OK:
            quote_ctx.close()
            trade_ctx.close()
            raise RuntimeError(
                f"Unlock trade failed: {data}"
            )

    ret, stock_data = quote_ctx.get_stock_basicinfo(
        market=trade_market,
        stock_type=SecurityType.STOCK,
        code_list=[config.symbol],
    )

    if ret != RET_OK:
        quote_ctx.close()
        trade_ctx.close()
        raise RuntimeError(
            f"Failed to get stock information: {stock_data}"
        )

    lot_size = int(
        stock_data["lot_size"].iloc[0]
    )

    return quote_ctx, trade_ctx, lot_size


def close_contexts(quote_ctx, trade_ctx):
    """Close Moomoo contexts."""

    if quote_ctx is not None:
        quote_ctx.close()

    if trade_ctx is not None:
        trade_ctx.close()


def place_order(
    trade_ctx,
    config,
    price,
    qty,
    side,
    order_type,
):
    """Place an order."""

    ret, data = trade_ctx.place_order(
        price=price,
        qty=qty,
        code=config.symbol,
        trd_side=side,
        order_type=order_type,
        trd_env=config.trade_env,
    )

    if ret != RET_OK:
        print(
            f"❌ Order submission failed: "
            f"{side} {config.symbol} | {data}"
        )
        return None

    print(
        f"✅ Order submitted: "
        f"{side} {qty} {config.symbol}"
    )

    return data


def get_position_status(trade_ctx, config):
    """Return current cost price for the configured symbol."""

    ret, positions = trade_ctx.position_list_query(
        trd_env=config.trade_env
    )

    if ret != RET_OK:
        print(
            "Error fetching positions:",
            positions,
        )
        return 0.0

    symbol_positions = positions.loc[
        positions["code"] == config.symbol
    ]

    if symbol_positions.empty:
        return 0.0

    return float(
        symbol_positions["cost_price"].iloc[0]
    )


def get_available_qty(
    trade_ctx,
    config,
    current_price,
):
    """Return available buy and sell quantities."""

    ret, trading_info = trade_ctx.acctradinginfo_query(
        order_type=OrderType.NORMAL,
        code=config.symbol,
        price=current_price,
        trd_env=config.trade_env,
    )

    if ret != RET_OK:
        print(
            "Error fetching trading info:",
            trading_info,
        )
        return 0, 0

    max_cash_buy = int(
        trading_info["max_cash_buy"].iloc[0]
    )

    max_position_sell = int(
        trading_info["max_position_sell"].iloc[0]
    )

    return max_cash_buy, max_position_sell


def get_market_trend_live(
    quote_ctx,
    config,
):
    """Return live broad-market price change."""

    trend_code = get_trend_symbol(
        config.symbol
    )

    ret, df_market = quote_ctx.get_market_snapshot(
        [trend_code]
    )

    if ret != RET_OK:
        print(
            "Error fetching market trend:",
            df_market,
        )
        return None

    return (
        df_market.loc[0, "last_price"]
        - df_market.loc[0, "prev_close_price"]
    )


def get_market_trend_simulation(
    quote_ctx,
    config,
    last_day,
):
    """Load historical benchmark candles."""

    if last_day is None:
        raise ValueError(
            "last_day must be provided in backtest mode."
        )

    trend_code = get_trend_symbol(
        config.symbol
    )

    ret, df_market, _ = (
        quote_ctx.request_history_kline(
            trend_code,
            last_day.strftime("%Y-%m-%d"),
            last_day.strftime("%Y-%m-%d"),
            SubType.K_1M,
            AuType.NONE,
        )
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Error fetching market trend: {df_market}"
        )

    df_market["time_key"] = pd.to_datetime(
        df_market["time_key"]
    )

    return df_market