import time

import pandas as pd
from moomoo import (
    RET_ERROR,
    RET_OK,
    CurKlineHandlerBase,
    OrderType,
    SubType,
    TradeOrderHandlerBase,
    TrdEnv,
    TrdSide,
)

from .broker import (
    get_available_qty,
    get_market_config,
    get_market_trend_live,
    place_order,
)
from .initialization import initialize_portfolio, load_historical_candles
from .strategy import LONG_WINDOW


class KlineHandler(CurKlineHandlerBase):
    def __init__(self, strategy, quote_ctx, trade_ctx, runtime_settings):
        super().__init__()

        self.strategy = strategy
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.runtime_settings = runtime_settings
        self.prev_candle = None

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)

        if ret != RET_OK:
            print("Kline error:", data)
            return RET_ERROR, data

        if data.empty:
            return RET_OK, data

        current_candle = data.iloc[-1]

        # Wait until the first candle is completed
        if self.prev_candle is None:
            self.prev_candle = current_candle
            return RET_OK, data

        if current_candle["time_key"] == self.prev_candle["time_key"]:
            return RET_OK, data

        candle = self.prev_candle
        self.prev_candle = current_candle
        current_price = float(candle["close"])

        # Retrieve live market data
        market_trend = get_market_trend_live(
            self.quote_ctx,
            self.runtime_settings,
        )

        max_cash_buy, max_position_sell = get_available_qty(
            self.trade_ctx,
            self.runtime_settings,
            current_price,
        )

        # Shared strategy pipeline
        self.strategy.process_candle(
            row=candle,
            market_trend=market_trend,
            max_cash_buy=max_cash_buy,
            max_position_sell=max_position_sell,
            execute_order=self.execute_order,
        )

        return RET_OK, data

    def execute_order(self, action, qty, price):
        """Submit a live Moomoo order."""

        # an order is already pending, do not submit a new one
        if self.strategy.pending_order:
            return False, None

        side = TrdSide.BUY if action == "BUY" else TrdSide.SELL

        # Mark the order as pending before placing it to prevent duplicate submissions
        self.strategy.pending_order = True
        try:
            order_data = place_order(
                trade_ctx=self.trade_ctx,
                runtime_settings=self.runtime_settings,
                price=price,
                qty=qty,
                side=side,
                order_type=OrderType.MARKET,
            )
        except Exception:
            # assume order submission failed, reset pending_order to allow future orders
            self.strategy.pending_order = False
            raise

        if order_data is None:
            # order submission failed, reset pending_order to allow future orders
            self.strategy.pending_order = False
            return False, None

        return True, order_data


class OrderHandler(TradeOrderHandlerBase):
    def __init__(self, strategy, trade_ctx, runtime_settings):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.runtime_settings = runtime_settings

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)

        if ret != RET_OK:
            print("❌ Order callback error:", data)
            return (RET_ERROR, data)

        if len(data) != 1:
            print(
                "❌ Order callback error: unexpected data length:",
                len(data),
            )
            return (RET_ERROR, data)

        order_id = data["order_id"].iloc[0]
        order_status = data["order_status"].iloc[0]

        terminal_statuses = {
            "SUBMIT_FAILED",
            "FAILED",
            "DISABLED",
            "DELETED",          
            "FILLED_ALL",
            "CANCELLED_PART",
            "CANCELLED_ALL"
        }
        if order_status not in terminal_statuses:
            # Keep the order pending while its status is nonterminal.
            return (RET_OK, data)

        self.strategy.pending_order = False

        for record in self.strategy.output:
            if record["order_id"] != order_id:
                continue

            record["order_status"] = order_status
            cumulative_qty = int(data["dealt_qty"].iloc[0])
            cumulative_price = float(data["dealt_avg_price"].iloc[0])

            # if there is no dealt_qty, skip
            if cumulative_qty <= 0 or order_status in {
                "SUBMIT_FAILED",
                "FAILED",
            }:
                break

            # account for partial fills, 
            # else can always assume previous_qty = 0
            previous_qty = int(record.get("trade_qty", 0) or 0)
            additional_qty = cumulative_qty - previous_qty
            if additional_qty < 0:
                print(
                    f"❌ Order callback error: dealt quantity decreased "
                    f"for order {order_id} ({previous_qty} to {cumulative_qty})"
                )
                return (RET_ERROR, data)
            elif additional_qty > 0:
                previous_notional = (
                    float(record["execution_price"]) * previous_qty
                    if previous_qty
                    else 0.0
                )
                # previous_notional = 0 for filled_all
                # cumulative_qty = additional_qty for filled_all
                # additional_price = cumulative_price for filled_all
                additional_price = (
                    cumulative_price * cumulative_qty - previous_notional
                ) / additional_qty
                action = data["trd_side"].iloc[0]
                cost_price_before = self.strategy.cost_price

                # Update trade_qty, total_price, position_qty, cost_price, position_open
                self.strategy.apply_fill(
                    action=action,
                    price=additional_price,
                    qty=additional_qty,
                )

                self._update_record(
                    record,
                    data,
                    cost_price_before,
                    cumulative_qty,
                    cumulative_price,
                    additional_qty,
                    additional_price,
                )

            if self.runtime_settings.trade_env == TrdEnv.REAL:
                ret, order_fee = self.trade_ctx.order_fee_query([order_id])
                if ret != RET_OK:
                    print("❌ Order Fee error:", order_fee)
                    return (RET_ERROR, order_fee)

                record["fee_amount"] = order_fee["fee_amount"].iloc[0]
                record["fee_details"] = order_fee["fee_details"].iloc[0]
                break

            break

        return (RET_OK, data)

    def _update_record(
        self,
        record,
        data,
        cost_price_before,
        cumulative_qty,
        cumulative_price,
        additional_qty,
        additional_price,
    ):
        
        action = data["trd_side"].iloc[0]

        # data['dealt_qty'] if filled_all
        record["trade_qty"] = cumulative_qty
        # data['dealt_avg_price'] if filled_all
        record["execution_price"] = cumulative_price
        record["execution_time"] = data["updated_time"].iloc[0]
        
        # Note: trade_qty, total_price, position_qty, cost_price, position_open updated in self.strategy.apply_fill()
        record["total_price"] = self.strategy.total_price

        if action == "SELL":
            # additional_price = cumulative_price = dealt_avg_price for filled_all
            # additional_qty = cumulative_qty = dealt_qty = trade_qtyfor filled_all
            record["realized_pl_gross"] += (
                additional_price - cost_price_before
            ) * additional_qty
            record["cost_price"] = cost_price_before

        elif action == "BUY":
            record["cost_price"] = self.strategy.cost_price

        record["Position"] = "OPEN" if self.strategy.position_open else "CLOSED"
        record["Position Qty"] = self.strategy.position_qty

        print(
            f"{self.runtime_settings.symbol} | "
            f"Price: {record['execution_price']:.2f} | "
            f"Action: {action} | "
            f"Time: {record['execution_time']}"
        )


def initialize_live(
    strategy,
    trade_ctx,
    quote_ctx,
    runtime_settings,
) -> bool:
    """Initialize live trading."""

    _, market_key = get_market_config(runtime_settings.symbol)

    while True:
        ret, state = quote_ctx.get_global_state()
        if ret != RET_OK:
            print(
                "[QUOTE] get_global_state failed during startup:",
                state,
            )
            time.sleep(60)
            continue

        market_state = state[market_key]
        if market_state in (
            "AFTER_HOURS_BEGIN",
            "CLOSED",
        ):
            print("LIVE INITIALIZATION SKIPPED: Market closed")
            return False

        market_now = pd.Timestamp.now(tz=runtime_settings.market_datetime.tz)
        session_date = pd.Timestamp(market_now.date())

        historical_df = load_historical_candles(
            quote_ctx,
            runtime_settings,
            session_date,
        )

        # Only use completed candles before current market time.
        historical_df = historical_df.loc[
            historical_df["time_key"] < market_now.tz_localize(None)
        ].copy()

        df_past = historical_df.iloc[-(LONG_WINDOW):].copy()

        if len(df_past) >= LONG_WINDOW:
            runtime_settings.market_datetime = market_now
            break

        print(
            "Waiting for completed market candles to initialize: "
            f"{len(df_past)}/{LONG_WINDOW}. Retrying in 60 seconds."
        )
        time.sleep(60)

    initialize_portfolio(
        strategy=strategy,
        trade_ctx=trade_ctx,
        runtime_settings=runtime_settings,
        current_price=float(df_past["close"].iloc[-1]),
        mode="live",
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
    return True


def run_live(
    strategy,
    quote_ctx,
    trade_ctx,
    runtime_settings,
):
    """Initialize and start live trading."""

    if not initialize_live(
        strategy,
        trade_ctx,
        quote_ctx,
        runtime_settings,
    ):
        return

    trade_ctx.set_handler(
        OrderHandler(
            strategy,
            trade_ctx,
            runtime_settings,
        )
    )

    quote_ctx.set_handler(
        KlineHandler(
            strategy,
            quote_ctx,
            trade_ctx,
            runtime_settings,
        )
    )

    ret, data = quote_ctx.subscribe(
        [runtime_settings.symbol],
        [SubType.K_1M],
        subscribe_push=True,
    )

    if ret != RET_OK:
        raise RuntimeError(f"Subscription failed: {data}")

    _, market_key = get_market_config(runtime_settings.symbol)

    print(f"🚀 Started LIVE TRADING in environment: {runtime_settings.trade_env}")

    print("Press Ctrl+C to exit.")

    while True:
        ret, state = quote_ctx.get_global_state()

        if ret != RET_OK:
            print(
                "[QUOTE] get_global_state failed:",
                state,
            )
            time.sleep(1)
            continue

        market_state = state[market_key]

        if market_state in (
            "AFTER_HOURS_BEGIN",
            "CLOSED",
        ):
            print("LOOP EXITED: Market closed")
            return

        time.sleep(1)
