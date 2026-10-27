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

from .config import pwd_unlock

# these 3 dont support backtesting:
# get_position_status (Current position quantity and cost price)
# get_available_qty (Current available buy/sell quantity)
# get_portfolio_equity (Current total portfolio equity)


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


def create_contexts(runtime_settings):
    """Create and initialize Moomoo quote/trade contexts."""

    trade_market, _ = get_market_config(runtime_settings.symbol)

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

    if runtime_settings.trade_env == TrdEnv.REAL:
        ret, data = trade_ctx.unlock_trade(password_md5=pwd_unlock)

        if ret != RET_OK:
            quote_ctx.close()
            trade_ctx.close()
            raise RuntimeError(f"Unlock trade failed: {data}")

    ret, stock_data = quote_ctx.get_stock_basicinfo(
        market=trade_market,
        stock_type=SecurityType.STOCK,
        code_list=[runtime_settings.symbol],
    )

    if ret != RET_OK:
        quote_ctx.close()
        trade_ctx.close()
        raise RuntimeError(f"Failed to get stock information: {stock_data}")

    lot_size = int(stock_data["lot_size"].iloc[0])

    return quote_ctx, trade_ctx, lot_size


def close_contexts(quote_ctx, trade_ctx):
    """Close Moomoo contexts."""

    if quote_ctx is not None:
        quote_ctx.close()

    if trade_ctx is not None:
        trade_ctx.close()


def place_order(
    trade_ctx,
    runtime_settings,
    price,
    qty,
    side,
    order_type,
):
    """Place an order."""

    ret, data = trade_ctx.place_order(
        price=price,
        qty=qty,
        code=runtime_settings.symbol,
        trd_side=side,
        order_type=order_type,
        trd_env=runtime_settings.trade_env,
    )

    if ret != RET_OK:
        print(f"❌ Order submission failed: {side} {runtime_settings.symbol} | {data}")
        return None

    print(f"✅ Order submitted: {side} {qty} {runtime_settings.symbol}")

    return data


def get_position_status(trade_ctx, runtime_settings):
    """Get position details using the appropriate cost basis."""

    ret, positions = trade_ctx.position_list_query(trd_env=runtime_settings.trade_env)

    if ret != RET_OK:
        raise RuntimeError(f"Error fetching positions: {positions}")

    position_columns = [
        "code",
        "qty",
        "nominal_price",
        "cost_price",
        "diluted_cost",
        "average_cost",
        "market_val",
        "pl_ratio",
        "pl_ratio_avg_cost",
    ]

    df = positions.loc[
        (positions["code"] == runtime_settings.symbol) & (positions["qty"] > 0)
    ].copy()

    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "code": runtime_settings.symbol,
                    **{col: 0 for col in position_columns if col != "code"},
                }
            ]
        )

    return df


def get_available_qty(
    trade_ctx,
    runtime_settings,
    current_price,
):
    """Return available buy and sell quantities."""

    ret, trading_info = trade_ctx.acctradinginfo_query(
        order_type=OrderType.NORMAL,
        code=runtime_settings.symbol,
        price=current_price,
        trd_env=runtime_settings.trade_env,
    )

    if ret != RET_OK:
        print(
            "Error fetching trading info:",
            trading_info,
        )
        return 0, 0

    max_cash_buy = int(trading_info["max_cash_buy"].iloc[0])

    max_position_sell = int(trading_info["max_position_sell"].iloc[0])

    return max_cash_buy, max_position_sell


def get_live_market_snapshot(
    quote_ctx,
    codes,
):
    """Fetch a snapshot for the requested symbols."""

    ret, snapshot = quote_ctx.get_market_snapshot(codes)

    if ret != RET_OK:
        print(
            "Error fetching live market snapshot:",
            snapshot,
        )
        return None

    if snapshot.empty:
        print("No live market snapshot returned.")
        return None

    return snapshot


def get_live_market_data(quote_ctx, runtime_settings):
    """Return live benchmark price change."""

    trend_code = get_trend_symbol(runtime_settings.symbol)
    snapshot = get_live_market_snapshot(quote_ctx, [trend_code])
    if snapshot is None:
        return None

    benchmark_quote = snapshot.loc[snapshot["code"] == trend_code]

    if benchmark_quote.empty:
        print(f"No market snapshot found for benchmark {trend_code}.")
        return None

    last_price = pd.to_numeric(
        benchmark_quote["last_price"].iloc[0],
        errors="coerce",
    )
    previous_close = pd.to_numeric(
        benchmark_quote["prev_close_price"].iloc[0],
        errors="coerce",
    )
    if pd.isna(last_price) or pd.isna(previous_close):
        print(f"Invalid market trend snapshot for {trend_code}.")
        return None

    return float(last_price - previous_close)


def get_live_bid_ask(quote_ctx, runtime_settings):
    """Return the latest bid-ask spread as a decimal midpoint ratio."""

    symbol = runtime_settings.symbol
    snapshot = get_live_market_snapshot(quote_ctx, [symbol])
    if snapshot is None:
        return None

    symbol_quote = snapshot.loc[snapshot["code"] == symbol]
    if symbol_quote.empty:
        print(f"No bid/ask snapshot found for {symbol}.")
        return None

    quote = symbol_quote.iloc[0]
    bid = pd.to_numeric(quote["bid_price"], errors="coerce")
    ask = pd.to_numeric(quote["ask_price"], errors="coerce")
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or ask < bid:
        print(
            f"Invalid bid/ask quote for {symbol}: "
            f"bid={bid}, ask={ask}"
        )
        return None

    spread = float(ask - bid)
    midpoint = float((ask + bid) / 2)
    return spread / midpoint


def get_market_trend_simulation(
    quote_ctx,
    runtime_settings,
    session_date,
):
    """Load historical benchmark candles."""

    trend_code = get_trend_symbol(runtime_settings.symbol)

    api_date = session_date.strftime("%Y-%m-%d")

    ret, df_market, _ = quote_ctx.request_history_kline(
        trend_code,
        api_date,
        api_date,
        SubType.K_1M,
        AuType.NONE,
        session=Session.ALL,
    )

    if ret != RET_OK:
        raise RuntimeError(f"Error fetching market trend: {df_market}")

    df_market["time_key"] = pd.to_datetime(df_market["time_key"])

    df_market = df_market.sort_values("time_key").reset_index(drop=True)

    # Use 09:30 benchmark candle as reference.
    market_open_time = session_date.replace(
        hour=9,
        minute=30,
    )

    open_row = df_market.loc[df_market["time_key"] == market_open_time]

    if open_row.empty:
        raise RuntimeError("09:30 benchmark candle not available.")

    market_open_price = float(open_row["open"].iloc[0])

    df_market["market_trend"] = df_market["close"].astype(float) - market_open_price

    return df_market


def get_portfolio_equity(trade_ctx, runtime_settings, currency="USD"):
    """Get current total portfolio equity."""

    ret, account = trade_ctx.accinfo_query(
        trd_env=runtime_settings.trade_env,
        currency=currency,
    )

    if ret != RET_OK:
        raise RuntimeError(f"Failed to get portfolio equity: {account}")

    return float(account["total_assets"].iloc[0])
