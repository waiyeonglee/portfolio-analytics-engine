import numpy as np
import pandas as pd
import talib


# ============================================================
# CONFIG
# ============================================================

RSI_THRESHOLD_FOLLOW = 55
RSI_THRESHOLD_REVERT = 35

RSI_PERIOD = 14
SHORT_WINDOW = 12
LONG_WINDOW = 26
MACD_SIGNAL = 9

PROFIT_PCT = 1.5
LOSS_PCT = -1.0

WINDOW_LENGTH = max(
    RSI_PERIOD + 1,
    LONG_WINDOW + MACD_SIGNAL + 1,
)

def round_down_to_lot(qty, lot_size):
    """Round quantity down to the nearest valid lot size."""

    if qty <= 0:
        return 0

    return (int(qty) // lot_size) * lot_size

class MovingAverageStrategy:

    def __init__(self, lot_size=1):
        self.lot_size = int(lot_size)

        # Historical state
        self.prices = []
        self.output = []

        # Indicators
        self.pct_diff = 0.0
        self.cum_sum_pct = 0.0

        self.short_sma = 0.0
        self.long_sma = 0.0

        self.rsi = 0.0

        self.macd = 0.0
        self.macd_signal = 0.0
        self.macd_histogram = 0.0

        # VWAP
        self.prev_vwap = 0.0
        self.vwap = 0.0
        self.cum_turnover = 0.0
        self.cum_volume = 0.0

        # Market
        self.market_trend = 0.0

        # Position
        self.position_open = False
        self.position_qty = 0
        self.cost_price = 0.0
        self.total_price = 0.0

        # Available quantities
        self.max_cash_buy = 0
        self.max_position_sell = 0

        # Current trade
        self.trade_qty = 0
        self.unrealized_pl_pct = 0.0
        self.realized_pl_pct = 0.0
        self.pending_order = False

        self.initial_equity = None
        self.cash = None
        self.equity_history = []

    def process_candle(
        self,
        row,
        market_trend=0.0,
        max_cash_buy=0,
        max_position_sell=0,
        execute_order=None,
        init=False,
    ):
        """Process a candle for initialization or trading."""

        self.update_indicators(
            row,
            init=init,
        )

        current_price = self.prices[-1]

        self.market_trend = (
            market_trend
            if market_trend is not None
            else 0.0
        )

        self.max_cash_buy = int(max_cash_buy)
        self.max_position_sell = int(max_position_sell)

        self.unrealized_pl_pct = self.compute_pl(
            current_price
        )

        self.reset_trade_state()

        # Warm-up: update indicators only, no trading
        if init:
            self.save_output(
                row,
                "INITIALIZING",
                order_data=None,
            )
            return "INITIALIZING"

        # Generate trading signal
        action, buy_qty, sell_qty = self.buy_or_sell(
            self.unrealized_pl_pct
        )

        order_data = None

        if self.pending_order:
            action = "HOLD"

        elif action in ("BUY", "SELL"):
            qty = buy_qty if action == "BUY" else sell_qty

            if qty <= 0:
                action = "HOLD"

            elif execute_order is None:
                raise ValueError(
                    "execute_order is required for trading."
                )

            else:
                success, order_data = execute_order(
                    action,
                    qty,
                    current_price,
                )

                if not success:
                    action = "HOLD"

        if action == "HOLD":
            self.reset_trade_state()

        if self.cash is not None:
            self.equity_history.append(
                self.cash
                + self.position_qty * current_price
            )

        self.save_output(
            row,
            action,
            order_data=order_data,
        )

        return action

    def update_indicators(self, row, init=False):
        """Update price statistics and technical indicators."""
        price = float(row["close"])
        previous = self.prices[-1] if self.prices else None
        self.prices.append(price)

        self.pct_diff = (
            (price - previous) / previous * 100
            if previous is not None and previous != 0 else 0.0
        )
        self.cum_sum_pct += self.pct_diff

        if len(self.prices) >= SHORT_WINDOW:
            self.short_sma = float(np.mean(self.prices[-SHORT_WINDOW:]))
        else:
            self.short_sma = 0.0

        if len(self.prices) >= LONG_WINDOW:
            self.long_sma = float(np.mean(self.prices[-LONG_WINDOW:]))
        else:
            self.long_sma = 0.0

        if len(self.prices) >= RSI_PERIOD + 1:
            recent = np.asarray(self.prices[-(RSI_PERIOD + 1):], dtype=float)
            self.rsi = float(talib.RSI(recent, timeperiod=RSI_PERIOD)[-1])
        else:
            self.rsi = 0.0

        macd_length = LONG_WINDOW + MACD_SIGNAL + 1
        if len(self.prices) >= macd_length:
            recent = np.asarray(self.prices[-macd_length:], dtype=float)
            macd, signal, histogram = talib.MACD(
                recent,
                fastperiod=SHORT_WINDOW,
                slowperiod=LONG_WINDOW,
                signalperiod=MACD_SIGNAL,
            )
            self.macd = float(macd[-1])
            self.macd_signal = float(signal[-1])
            self.macd_histogram = float(histogram[-1])
        else:
            self.macd = self.macd_signal = self.macd_histogram = 0.0

        if not init:
            self.prev_vwap = self.vwap
            self.cum_turnover += float(row["turnover"])
            self.cum_volume += float(row["volume"])
            if self.cum_volume > 0:
                self.vwap = self.cum_turnover / self.cum_volume

    def compute_pl(
        self,
        current_price,
    ):
        """Calculate current percentage P/L."""

        if self.cost_price <= 0:
            return 0.0

        return (
            (float(current_price) - self.cost_price)
            / self.cost_price
            * 100
        )

    def buy_or_sell(self, pl_pct=0.0):
        """Calculate trade sizes and choose BUY, SELL, or HOLD."""
        strength = self.macd - self.macd_signal
        buy_ratio = min(0.7, abs(strength / 0.5))
        buy_qty = round_down_to_lot(
            self.max_cash_buy * buy_ratio, self.lot_size
        )

        available = min(self.position_qty, self.max_position_sell)
        sell_ratio = min(1.0, abs(pl_pct / LOSS_PCT))
        sell_qty = round_down_to_lot(
            available * sell_ratio, self.lot_size
        )

        if self.market_trend > 0:
            buy_signal = (
                buy_qty > 0
                and strength > 0
                and self.rsi > RSI_THRESHOLD_FOLLOW
            )
        else:
            buy_signal = (
                buy_qty > 0
                and strength < 0
                and self.rsi < RSI_THRESHOLD_REVERT
            )

        if pl_pct >= PROFIT_PCT:
            sell_signal = (
                sell_qty > 0
                and self.macd < self.macd_signal
                and self.rsi < RSI_THRESHOLD_REVERT
            )
        else:
            sell_signal = (
                sell_qty > 0
                and (self.macd < self.macd_signal or pl_pct <= LOSS_PCT)
            )

        if sell_signal:
            return "SELL", buy_qty, sell_qty
        if buy_signal:
            return "BUY", buy_qty, sell_qty
        return "HOLD", buy_qty, sell_qty

    def apply_fill(
        self,
        action,
        price,
        qty,
    ):
        """Update position/accounting after a completed fill."""

        price = float(price)
        qty = int(qty)

        if qty <= 0:
            return

        self.trade_qty = qty

        if action == "BUY":
            self.realized_pl_pct = 0.0
            self.total_price += (
                price * qty
            )

            self.position_qty += qty

            if self.position_qty > 0:
                self.cost_price = (
                    self.total_price
                    / self.position_qty
                )

        elif action == "SELL":
            
        # Calculate before changing cost basis.
            self.realized_pl_pct = self.compute_pl(
                price
            )
            
            if qty > self.position_qty:
                raise RuntimeError(
                    f"Position mismatch: "
                    f"SELL qty={qty}, "
                    f"position_qty={self.position_qty}"
                )

            self.total_price -= (
                self.cost_price * qty
            )

            self.position_qty -= qty

            if self.position_qty == 0:
                self.total_price = 0.0
                self.cost_price = 0.0

        else:
            raise ValueError(
                f"Unsupported action: {action}"
            )

        self.position_open = (
            self.position_qty > 0
        )

    def reset_trade_state(self):
        self.trade_qty = 0
        self.realized_pl_pct = 0.0

    def save_output(
        self,
        row,
        action,
        order_data=None,
    ):
        """Save current strategy state."""

        order_id = None
        order_status = None

        if order_data is not None:

            if (
                "order_id" in order_data.columns
                and not order_data.empty
            ):
                order_id = (
                    order_data["order_id"].iloc[0]
                )

            if (
                "order_status" in order_data.columns
                and not order_data.empty
            ):
                order_status = (
                    order_data["order_status"].iloc[0]
                )

        candle = {
            # Candle
            "code": row["code"],
            "time": row["time_key"],
            "open": row["open"],
            "close": row["close"],

            # Price statistics
            "pct_diff": self.pct_diff,
            "cum_sum_pct": self.cum_sum_pct,

            # Moving averages
            "short_sma": self.short_sma,
            "long_sma": self.long_sma,

            # RSI
            "RSI": self.rsi,

            # MACD
            "MACD": self.macd,
            "MACD Signal": self.macd_signal,
            "MACD Histogram": self.macd_histogram,
            "MACD_up": (
                self.macd
                > self.macd_signal
            ),

            # Market
            "market_trend": self.market_trend,

            # Position
            "Position": (
                "OPEN"
                if self.position_open
                else "CLOSED"
            ),
            "Position Qty": self.position_qty,
            "cost_price": self.cost_price,
            "total_price": self.total_price,

            # Capacity
            "max_cash_buy": self.max_cash_buy,
            "max_position_sell": (
                self.max_position_sell
            ),

            # Trade
            "action": action,
            "trade_qty": self.trade_qty,

            # P/L
            "unrealized_pl_pct": (
                self.unrealized_pl_pct
            ),
            "realized_pl_pct": (
                self.realized_pl_pct
            ),

            "hit_profit": (
                self.unrealized_pl_pct
                >= PROFIT_PCT
            ),
            "hit_loss": (
                self.unrealized_pl_pct
                <= LOSS_PCT
            ),

            # Order
            "order_id": order_id,
            "order_status": order_status,

            # Execution
            "execution_time": pd.NaT,
            "execution_price": np.nan,

            # Fees
            "fee_amount": 0.0,
            "fee_details": None,
        }

        self.output.append(candle)