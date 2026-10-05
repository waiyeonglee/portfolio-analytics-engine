import time

from moomoo import (
    RET_ERROR,
    RET_OK,
    AuType,
    CurKlineHandlerBase,
    OrderType,
    Session,
    SubType,
    TradeDealHandlerBase,
    TradeOrderHandlerBase,
    TrdEnv,
    TrdSide,
)

from broker import (
    get_available_qty,
    get_market_config,
    get_market_trend_live,
    get_position_status,
    place_order,
)

from strategy import WINDOW_LENGTH


class KlineHandler(CurKlineHandlerBase):

    def __init__(self, strategy, quote_ctx, trade_ctx, config):
        super().__init__()

        self.strategy = strategy
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.config = config
        self.prev_candle = None

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)

        if ret != RET_OK:
            print("Kline error:", data)
            return (RET_ERROR, data)

        current_candle = data.iloc[-1]

        if self.prev_candle is None:
            self.prev_candle = current_candle

            self.strategy.save_output(
                self.prev_candle,
                "INITIALIZING",
                order_data=None,
            )

            return (RET_OK, data)

        if current_candle["time_key"] == self.prev_candle["time_key"]:
            return (RET_OK, data)

        candle_to_process = self.prev_candle
        self.prev_candle = current_candle

        print(
            f"Current time: {candle_to_process['time_key']}, "
            f"Current price: {candle_to_process['close']}"
        )

        candle_price = float(
            candle_to_process["close"]
        )


        market_trend = get_market_trend_live(
            self.quote_ctx,
            self.config,
        )

        (
            max_cash_buy,
            max_position_sell,
        ) = get_available_qty(
            self.trade_ctx,
            self.config,
            candle_price,
        )

        (
            action,
            buy_qty,
            sell_qty,
            current_price,
        ) = self.strategy.process_candle(
            row=candle_to_process,
            market_trend=market_trend,
            max_cash_buy=max_cash_buy,
            max_position_sell=max_position_sell,
        )

        order_data = None
        self.strategy.trade_qty = 0

        if action in ("BUY", "SELL"):

            if action == "BUY":
                qty = buy_qty
                side = TrdSide.BUY
                max_qty = self.strategy.max_cash_buy

            else:
                qty = sell_qty
                side = TrdSide.SELL
                max_qty = self.strategy.max_position_sell

            print(
                f"Max QTY to {action.title()}: {max_qty}"
            )

            order_data = place_order(
                trade_ctx=self.trade_ctx,
                config=self.config,
                price=current_price,
                qty=qty,
                side=side,
                order_type=OrderType.MARKET,
            )

            if order_data is not None:
                self.strategy.pending_order = True

            else:
                action = "HOLD"

        else:
            self.strategy.reset_trade_state()

        self.strategy.save_output(
            candle_to_process,
            action,
            order_data,
        )

        return (RET_OK, data)


class OrderHandler(TradeOrderHandlerBase):

    def __init__(self, strategy, trade_ctx, config):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.config = config

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

        if order_status != "FILLED_ALL":
            return (RET_OK, data)

        for record in self.strategy.output:

            if record["order_id"] != order_id:
                continue

            record["order_status"] = order_status

            if self.config.trade_env == TrdEnv.REAL:
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

        record["realized_pl_pct"] = (
            self.strategy.realized_pl_pct
        )

        record["Position"] = (
            "OPEN"
            if self.strategy.position_open
            else "CLOSED"
        )

        print(
            f"{self.config.symbol} | "
            f"Price: {current_price:.2f} | "
            f"Action: {action} | "
            f"Time: {record['execution_time']}"
        )


class DealHandler(TradeDealHandlerBase):

    def __init__(self, strategy, trade_ctx, config):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.config = config

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

            record["execution_price"] = fill_price

            record["realized_pl_pct"] = (
                self.strategy.realized_pl_pct
            )

            record["Position"] = (
                "OPEN"
                if self.strategy.position_open
                else "CLOSED"
            )

            # Accumulate actual filled quantity.
            previous_filled = record.get(
                "filled_qty",
                0,
            )

            record["filled_qty"] = (
                previous_filled + fill_qty
            )

            print(
                f"{self.config.symbol} | "
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
    config,
):
    """
    Load enough historical candles to warm up
    the strategy before live trading starts.
    """

    api_date = config.timezone_date.strftime(
        "%Y-%m-%d"
    )

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
        historical_df["time_key"]
    )

    df_past = historical_df.iloc[
        -WINDOW_LENGTH + 1:
    ].copy()

    if df_past.empty:
        raise RuntimeError(
            "No historical candles available "
            "for strategy initialization."
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
                strategy.max_cash_buy,
                strategy.max_position_sell,
            ) = get_available_qty(
                trade_ctx,
                config,
                current_price,
            )

            (
                strategy.position_qty,
                strategy.cost_price,
            ) = get_position_status(
                trade_ctx,
                config,
            )

            strategy.position_open = (
                strategy.position_qty > 0
            )

            strategy.total_price = (
                strategy.cost_price
                * strategy.position_qty
            )

        strategy.unrealized_pl_pct = (
            strategy.compute_pl(current_price)
        )

        strategy.trade_qty = 0
        strategy.realized_pl_pct = 0

        strategy.save_output(
            row,
            "INITIALIZING",
            order_data=None,
        )

    print(
        "Initialized time:",
        df_past["time_key"].iloc[-1],
    )


def run_live(
    strategy,
    quote_ctx,
    trade_ctx,
    config,
):
    """Initialize and start live trading."""

    initialize_live(
        strategy,
        trade_ctx,
        quote_ctx,
        config,
    )

    trade_ctx.set_handler(
        OrderHandler(
            strategy,
            trade_ctx,
            config,
        )
    )

    quote_ctx.set_handler(
        KlineHandler(
            strategy,
            quote_ctx,
            trade_ctx,
            config,
        )
    )

    if config.trade_env == TrdEnv.REAL:
        trade_ctx.set_handler(
            DealHandler(
                strategy,
                trade_ctx,
                config,
            )
        )

    ret, data = quote_ctx.subscribe(
        [config.symbol],
        [SubType.K_1M],
        subscribe_push=True,
    )

    if ret != RET_OK:
        raise RuntimeError(
            f"Subscription failed: {data}"
        )

    _, market_key = get_market_config(
        config.symbol
    )

    print(
        "🚀 Started LIVE TRADING "
        f"in environment: {config.trade_env}"
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