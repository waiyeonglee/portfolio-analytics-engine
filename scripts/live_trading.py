import time
import pandas as pd

from moomoo import (
    RET_ERROR,
    RET_OK,
    CurKlineHandlerBase,
    OrderType,
    SubType,
    TradeDealHandlerBase,
    TradeOrderHandlerBase,
    TrdEnv,
    TrdSide,
)

from .initialization import (
    load_historical_candles,
    initialize_portfolio
)

from .broker import (
    get_available_qty,
    get_market_config,
    get_market_trend_live,
    place_order
)

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

        if (
            current_candle["time_key"]
            == self.prev_candle["time_key"]
        ):
            return RET_OK, data

        candle = self.prev_candle
        self.prev_candle = current_candle
        current_price = float(candle["close"])

        # Retrieve live market data
        market_trend = get_market_trend_live(
            self.quote_ctx,
            self.runtime_settings,
        )

        max_cash_buy, max_position_sell = (
            get_available_qty(
                self.trade_ctx,
                self.runtime_settings,
                current_price,
            )
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

        if self.strategy.pending_order:
            return False, None

        side = (
            TrdSide.BUY
            if action == "BUY"
            else TrdSide.SELL
        )

        order_data = place_order(
            trade_ctx=self.trade_ctx,
            runtime_settings=self.runtime_settings,
            price=price,
            qty=qty,
            side=side,
            order_type=OrderType.MARKET,
        )

        if order_data is None:
            return False, None

        self.strategy.pending_order = True

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
            "FILLED_ALL",
            "CANCELLED_ALL",
            "FAILED",
        }

        if order_status in terminal_statuses:
            self.strategy.pending_order = False

        if order_status not in {
            "FILLED_ALL",
            "CANCELLED_ALL",
        }:
            return (RET_OK, data)

        for record in self.strategy.output:

            if record["order_id"] != order_id:
                continue

            record["order_status"] = order_status

            if int(data["dealt_qty"].iloc[0]) <= 0:
                break

            if self.runtime_settings.trade_env == TrdEnv.REAL:
                ret2, order_fee = (
                    self.trade_ctx.order_fee_query(
                        [order_id]
                    )
                )

                if ret2 != RET_OK:
                    print(
                        "❌ Order Fee error:",
                        order_fee,
                    )
                    return (RET_ERROR, order_fee)

                record["fee_amount"] = (
                    order_fee["fee_amount"].iloc[0]
                )

                record["fee_details"] = (
                    order_fee["fee_details"].iloc[0]
                )

                # Real fills are handled by DealHandler.
                break

            action = data["trd_side"].iloc[0]

            current_price = float(
                data["dealt_avg_price"].iloc[0]
            )

            filled_qty = int(
                data["dealt_qty"].iloc[0]
            )

            cost_price_before = (
                self.strategy.cost_price
            )

            self.strategy.apply_fill(
                action=action,
                price=current_price,
                qty=filled_qty,
            )

            self._update_record(
                record,
                data,
                current_price,
                cost_price_before,
            )

            self.strategy.pending_order = False
            break

        return (RET_OK, data)

    def _update_record(
        self,
        record,
        data,
        current_price,
        cost_price_before,
    ):
        action = data["trd_side"].iloc[0]
        filled_qty = int(data["dealt_qty"].iloc[0])

        record["cost_price"] = (
            cost_price_before
            if action == "SELL"
            else self.strategy.cost_price
        )

        record["total_price"] = (
            self.strategy.total_price
        )

        record["execution_time"] = (
            data["updated_time"].iloc[0]
        )

        record["execution_price"] = current_price
        record["filled_qty"] = filled_qty
        record["filled_notional"] = current_price * filled_qty
        if action == "SELL":
            record["realized_pl_gross"] = (
                current_price - cost_price_before
            ) * filled_qty

        record["realized_pl_pct"] = (
            self.strategy.realized_pl_pct
        )

        record["Position"] = (
            "OPEN"
            if self.strategy.position_open
            else "CLOSED"
        )

        record["trade_qty"] = (
            filled_qty
        )

        record["Position Qty"] = (
            self.strategy.position_qty
        )

        print(
            f"{self.runtime_settings.symbol} | "
            f"Price: {current_price:.2f} | "
            f"Action: {action} | "
            f"Time: {record['execution_time']}"
        )

class DealHandler(TradeDealHandlerBase):

    def __init__(self, strategy, trade_ctx, runtime_settings):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.runtime_settings = runtime_settings

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)

        if ret != RET_OK:
            print("❌ Deal callback error:", data)
            return (RET_ERROR, data)

        # A callback can contain more than one deal.
        for _, deal in data.iterrows():

            order_id = deal["order_id"]

            record = next(
                (
                    r for r in self.strategy.output
                    if r["order_id"] == order_id
                ),
                None,
            )

            if record is None:
                continue

            action = deal["trd_side"]
            fill_price = float(deal["price"])
            fill_qty = int(deal["qty"])
            cost_price_before = self.strategy.cost_price

            if action == "SELL":
                record["realized_pl_gross"] = (
                    record.get("realized_pl_gross", 0.0)
                    + (fill_price - cost_price_before) * fill_qty
                )

            self.strategy.apply_fill(
                action=action,
                price=fill_price,
                qty=fill_qty
            )

            record["cost_price"] = (
                self.strategy.cost_price
            )

            record["total_price"] = (
                self.strategy.total_price
            )

            record["execution_time"] = (
                deal["create_time"]
            )

            previous_notional = record.get(
                "filled_notional",
                0.0,
            )
            previous_filled = record.get(
                "filled_qty",
                0,
            )
            filled_notional = (
                previous_notional + fill_price * fill_qty
            )
            filled_qty = previous_filled + fill_qty

            record["filled_notional"] = filled_notional
            record["filled_qty"] = filled_qty
            record["execution_price"] = (
                filled_notional / filled_qty
            )

            record["realized_pl_pct"] = (
                self.strategy.realized_pl_pct
            )

            record["Position"] = (
                "OPEN"
                if self.strategy.position_open
                else "CLOSED"
            )

            record["trade_qty"] = filled_qty
            record["Position Qty"] = (
                self.strategy.position_qty
            )
            if action == "SELL":
                record["cost_price"] = cost_price_before

            print(
                f"{self.runtime_settings.symbol} | "
                f"Fill: {fill_qty} | "
                f"Price: {fill_price:.2f} | "
                f"Action: {action} | "
                f"Filled: {record['filled_qty']}"
            )

        return (RET_OK, data)

def initialize_live(
    strategy,
    trade_ctx,
    quote_ctx,
    runtime_settings,
) -> bool:
    """Initialize live trading."""

    _, market_key = get_market_config(
        runtime_settings.symbol
    )

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
            print(
                "LIVE INITIALIZATION SKIPPED: Market closed"
            )
            return False

        market_now = pd.Timestamp.now(
            tz=runtime_settings.market_datetime.tz
        )
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

        df_past = historical_df.iloc[
            -(LONG_WINDOW):
        ].copy()

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

    if runtime_settings.trade_env == TrdEnv.REAL:
        trade_ctx.set_handler(
            DealHandler(
                strategy,
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
        raise RuntimeError(
            f"Subscription failed: {data}"
        )

    _, market_key = get_market_config(
        runtime_settings.symbol
    )

    print(
        "🚀 Started LIVE TRADING "
        f"in environment: {runtime_settings.trade_env}"
    )

    print("Press Ctrl+C to exit.")

    while True:
        ret, state = (
            quote_ctx.get_global_state()
        )

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
            print(
                "LOOP EXITED: Market closed"
            )
            return

        time.sleep(1)