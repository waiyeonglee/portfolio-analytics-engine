import time

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

from broker import (
    get_available_qty,
    get_market_config,
    get_market_trend_live,
    get_position_status,
    place_order,
)


class KlineHandler(CurKlineHandlerBase):

    def __init__(
        self,
        strategy,
        quote_ctx,
        trade_ctx,
        config,
    ):
        super().__init__()

        self.strategy = strategy
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.config = config

        self.prev_candle = None

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(
            rsp_pb
        )

        if ret != RET_OK:
            print("Kline error:", data)
            return RET_ERROR, data

        current_candle = data.iloc[-1]

        if self.prev_candle is None:
            self.prev_candle = current_candle

            self.strategy.save_output(
                self.prev_candle,
                "INITIALIZING",
                order_data=None,
            )

            return RET_OK, data

        if (
            current_candle["time_key"]
            == self.prev_candle["time_key"]
        ):
            return RET_OK, data

        candle_to_process = self.prev_candle
        self.prev_candle = current_candle

        print(
            f"Current time: "
            f"{candle_to_process['time_key']}, "
            f"Current price: "
            f"{candle_to_process['close']}"
        )

        self.strategy.update_state_from_row(
            candle_to_process,
            init=False,
        )

        current_price = (
            self.strategy.prices[-1]
        )

        market_trend = get_market_trend_live(
            self.quote_ctx,
            self.config,
        )

        self.strategy.market_trend = (
            market_trend
            if market_trend is not None
            else 0
        )

        (
            self.strategy.max_cash_buy,
            self.strategy.max_position_sell,
        ) = get_available_qty(
            self.trade_ctx,
            self.config,
            current_price,
        )

        if (
            self.strategy.max_position_sell
            == 0
        ):
            self.strategy.cost_price = 0

        self.strategy.unrealized_pl_pct = (
            self.strategy.compute_pl(
                current_price
            )
        )

        (
            action,
            buy_qty,
            sell_qty,
        ) = self.strategy.buy_or_sell(
            self.strategy.unrealized_pl_pct
        )

        if self.strategy.pending_order:
            print(
                "Pending order, "
                "skipping this candle."
            )
            action = "HOLD"

        order_data = None
        self.strategy.trade_qty = 0

        if action in ("BUY", "SELL"):

            if action == "BUY":
                qty = buy_qty
                side = TrdSide.BUY
                max_qty = (
                    self.strategy.max_cash_buy
                )

            else:
                qty = sell_qty
                side = TrdSide.SELL
                max_qty = (
                    self.strategy
                    .max_position_sell
                )

            print(
                f"Max QTY to "
                f"{action.title()}: {max_qty}"
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
                self.strategy.trade_qty = qty

            else:
                action = "HOLD"

        else:
            self.strategy.reset_trade_state()

        self.strategy.save_output(
            candle_to_process,
            action,
            order_data,
        )

        return RET_OK, data


class OrderHandler(TradeOrderHandlerBase):

    def __init__(
        self,
        strategy,
        trade_ctx,
        config,
    ):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.config = config

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(
            rsp_pb
        )

        if ret != RET_OK:
            print(
                "❌ Order callback error:",
                data,
            )
            return RET_ERROR, data

        if len(data) != 1:
            print(
                "❌ Order callback error: "
                "unexpected data length:",
                len(data),
            )
            return RET_ERROR, data

        order_id = data["order_id"].iloc[0]
        order_status = (
            data["order_status"].iloc[0]
        )

        if order_status != "FILLED_ALL":
            return RET_OK, data

        for record in self.strategy.output:

            if record["order_id"] != order_id:
                continue

            record["order_status"] = (
                order_status
            )

            if (
                self.config.trade_env
                == TrdEnv.REAL
            ):
                ret2, order_fee = (
                    self.trade_ctx
                    .order_fee_query(
                        [order_id]
                    )
                )

                if ret2 != RET_OK:
                    print(
                        "❌ Order Fee error:",
                        order_fee,
                    )
                    return RET_ERROR, order_fee

                record["fee_amount"] = (
                    order_fee[
                        "fee_amount"
                    ].iloc[0]
                )

                record["fee_details"] = (
                    order_fee[
                        "fee_details"
                    ].iloc[0]
                )

                # Real fills are handled
                # by DealHandler.
                break

            action = (
                data["trd_side"].iloc[0]
            )

            current_price = float(
                data[
                    "dealt_avg_price"
                ].iloc[0]
            )

            current_position = (
                self.strategy
                .max_position_sell
            )

            if action == "BUY":
                position_qty_after = (
                    current_position
                    + self.strategy.trade_qty
                )

            else:
                position_qty_after = max(
                    0,
                    current_position
                    - self.strategy.trade_qty,
                )

            self.strategy.apply_fill(
                action=action,
                price=current_price,
                qty=self.strategy.trade_qty,
                position_qty_after=(
                    position_qty_after
                ),
            )

            self._update_record(
                record,
                data,
                current_price,
            )

            self.strategy.pending_order = False

            break

        return RET_OK, data

    def _update_record(
        self,
        record,
        data,
        current_price,
    ):
        record["cost_price"] = (
            self.strategy.cost_price
        )

        record["total_price"] = (
            self.strategy.total_price
        )

        record["execution_time"] = (
            data["updated_time"].iloc[0]
        )

        record["execution_price"] = (
            current_price
        )

        record["realized_pl_pct"] = (
            self.strategy.realized_pl_pct
        )

        record["Position"] = (
            "OPEN"
            if self.strategy.position_open
            else "CLOSED"
        )

        action = data["trd_side"].iloc[0]

        print(
            f"{self.config.symbol} "
            f"| Price: {current_price:.2f} "
            f"| Action: {action} "
            f"| Time: "
            f"{record['execution_time']}"
        )


class DealHandler(TradeDealHandlerBase):

    def __init__(
        self,
        strategy,
        trade_ctx,
        config,
    ):
        super().__init__()

        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.config = config

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(
            rsp_pb
        )

        if ret != RET_OK:
            print(
                "❌ Deal callback error:",
                data,
            )
            return RET_ERROR, data

        print("Deal callback received!")

        order_id = data["order_id"].iloc[0]

        for record in self.strategy.output:

            if record["order_id"] != order_id:
                continue

            action = (
                data["trd_side"].iloc[0]
            )

            current_price = float(
                data["price"].iloc[0]
            )

            current_position = (
                self.strategy
                .max_position_sell
            )

            if action == "BUY":
                position_qty_after = (
                    current_position
                    + self.strategy.trade_qty
                )

            else:
                position_qty_after = max(
                    0,
                    current_position
                    - self.strategy.trade_qty,
                )

            self.strategy.apply_fill(
                action=action,
                price=current_price,
                qty=self.strategy.trade_qty,
                position_qty_after=(
                    position_qty_after
                ),
            )

            record["cost_price"] = (
                self.strategy.cost_price
            )

            record["total_price"] = (
                self.strategy.total_price
            )

            record["execution_time"] = (
                data["create_time"].iloc[0]
            )

            record["execution_price"] = (
                current_price
            )

            record["realized_pl_pct"] = (
                self.strategy.realized_pl_pct
            )

            record["Position"] = (
                "OPEN"
                if self.strategy.position_open
                else "CLOSED"
            )

            self.strategy.pending_order = False

            print(
                f"{self.config.symbol} "
                f"| Price: {current_price:.2f} "
                f"| Action: {action} "
                f"| Time: "
                f"{record['execution_time']}"
            )

            break

        return RET_OK, data


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

    from moomoo import (
        RET_OK,
        AuType,
        Session,
        SubType,
    )

    from strategy import WINDOW_LENGTH

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

        current_price = (
            strategy.prices[-1]
        )

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

            strategy.cost_price = (
                get_position_status(
                    trade_ctx,
                    config,
                )
            )

        strategy.unrealized_pl_pct = (
            strategy.compute_pl(
                current_price
            )
        )

        strategy.trade_qty = 0
        strategy.realized_pl_pct = 0

        strategy.position_open = (
            strategy.max_position_sell > 0
        )

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
        f"🚀 Started LIVE TRADING "
        f"in environment: "
        f"{config.trade_env}"
    )

    print("Press Ctrl+C to exit.")

    while True:
        ret, state = (
            quote_ctx.get_global_state()
        )

        if ret != RET_OK:
            print(
                "[QUOTE] "
                "get_global_state failed:",
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