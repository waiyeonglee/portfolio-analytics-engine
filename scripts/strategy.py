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


# ============================================================
# HELPERS
# ============================================================

def round_down_to_lot(qty, lot_size):
    """Round quantity down to the nearest valid lot size."""

    if qty <= 0:
        return 0

    return (
        int(qty) // lot_size
    ) * lot_size


# ============================================================
# STRATEGY
# ============================================================

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

    # ========================================================
    # STATE / INDICATORS
    # ========================================================

    def update_state_from_row(
        self,
        row,
        init=False,
    ):
        """Update indicators using the latest candle."""

        current_price = float(
            row["close"]
        )

        previous_price = (
            self.prices[-1]
            if self.prices
            else None
        )

        self.prices.append(
            current_price
        )

        self._update_price_change(
            current_price,
            previous_price,
        )

        self._update_moving_averages()
        self._update_rsi()
        self._update_macd()

        if not init:
            self._update_vwap(
                turnover=float(
                    row["turnover"]
                ),
                volume=float(
                    row["volume"]
                ),
            )

    def _update_price_change(
        self,
        current_price,
        previous_price,
    ):
        if (
            previous_price is None
            or previous_price == 0
        ):
            self.pct_diff = 0.0

        else:
            self.pct_diff = (
                (
                    current_price
                    - previous_price
                )
                / previous_price
                * 100
            )

        self.cum_sum_pct += (
            self.pct_diff
        )

    def _update_moving_averages(self):
        if (
            len(self.prices)
            >= SHORT_WINDOW
        ):
            self.short_sma = float(
                np.mean(
                    self.prices[
                        -SHORT_WINDOW:
                    ]
                )
            )

        else:
            self.short_sma = 0.0

        if (
            len(self.prices)
            >= LONG_WINDOW
        ):
            self.long_sma = float(
                np.mean(
                    self.prices[
                        -LONG_WINDOW:
                    ]
                )
            )

        else:
            self.long_sma = 0.0

    def _update_rsi(self):
        required_prices = (
            RSI_PERIOD + 1
        )

        if (
            len(self.prices)
            < required_prices
        ):
            self.rsi = 0.0
            return

        prices = np.asarray(
            self.prices[
                -required_prices:
            ],
            dtype=float,
        )

        rsi = talib.RSI(
            prices,
            timeperiod=RSI_PERIOD,
        )

        self.rsi = float(
            rsi[-1]
        )

    def _update_macd(self):
        required_prices = (
            LONG_WINDOW
            + MACD_SIGNAL
            + 1
        )

        if (
            len(self.prices)
            < required_prices
        ):
            self.macd = 0.0
            self.macd_signal = 0.0
            self.macd_histogram = 0.0
            return

        prices = np.asarray(
            self.prices[
                -required_prices:
            ],
            dtype=float,
        )

        (
            macd,
            signal,
            histogram,
        ) = talib.MACD(
            prices,
            fastperiod=SHORT_WINDOW,
            slowperiod=LONG_WINDOW,
            signalperiod=MACD_SIGNAL,
        )

        self.macd = float(
            macd[-1]
        )

        self.macd_signal = float(
            signal[-1]
        )

        self.macd_histogram = float(
            histogram[-1]
        )

    def _update_vwap(
        self,
        turnover,
        volume,
    ):
        self.prev_vwap = self.vwap

        self.cum_turnover += turnover
        self.cum_volume += volume

        if self.cum_volume > 0:
            self.vwap = (
                self.cum_turnover
                / self.cum_volume
            )

    # ========================================================
    # PROFIT / LOSS
    # ========================================================

    def compute_pl(
        self,
        current_price,
    ):
        """Calculate current percentage P/L."""

        if self.cost_price <= 0:
            return 0.0

        return (
            (
                float(current_price)
                - self.cost_price
            )
            / self.cost_price
            * 100
        )

    # ========================================================
    # POSITION SIZING
    # ========================================================

    def _calculate_buy_qty(self):
        if self.max_cash_buy <= 0:
            return 0

        trend_strength = (
            self.macd
            - self.macd_signal
        )

        buy_ratio = min(
            0.7,
            abs(
                trend_strength / 0.5
            ),
        )

        return round_down_to_lot(
            self.max_cash_buy
            * buy_ratio,
            self.lot_size,
        )

    def _calculate_sell_qty(
        self,
        pl_pct,
    ):
        if (
            self.max_position_sell
            <= 0
        ):
            return 0

        sell_ratio = min(
            1.0,
            abs(
                pl_pct / LOSS_PCT
            ),
        )

        return round_down_to_lot(
            self.max_position_sell
            * sell_ratio,
            self.lot_size,
        )

    # ========================================================
    # DECISION
    # ========================================================

    def buy_or_sell(
        self,
        pl_pct=0.0,
    ):
        buy_qty = (
            self._calculate_buy_qty()
        )

        sell_qty = (
            self._calculate_sell_qty(
                pl_pct
            )
        )

        trend_strength = (
            self.macd
            - self.macd_signal
        )

        if self.market_trend > 0:
            buy_signal = (
                buy_qty > 0
                and trend_strength > 0
                and self.rsi
                > RSI_THRESHOLD_FOLLOW
            )

        else:
            buy_signal = (
                buy_qty > 0
                and trend_strength < 0
                and self.rsi
                < RSI_THRESHOLD_REVERT
            )

        if pl_pct >= PROFIT_PCT:
            sell_signal = (
                sell_qty > 0
                and self.macd
                < self.macd_signal
                and self.rsi
                < RSI_THRESHOLD_REVERT
            )

        else:
            sell_signal = (
                sell_qty > 0
                and (
                    self.macd
                    < self.macd_signal
                    or pl_pct
                    <= LOSS_PCT
                )
            )

        if sell_signal:
            return (
                "SELL",
                buy_qty,
                sell_qty,
            )

        if buy_signal:
            return (
                "BUY",
                buy_qty,
                sell_qty,
            )

        return (
            "HOLD",
            buy_qty,
            sell_qty,
        )

    # ========================================================
    # EXECUTION ACCOUNTING
    # ========================================================

    def apply_fill(
        self,
        action,
        price,
        qty,
        position_qty_after,
    ):
        """Update accounting after a completed fill."""

        price = float(price)
        qty = int(qty)

        position_qty_after = int(
            position_qty_after
        )

        if qty <= 0:
            return

        self.trade_qty = qty

        # Must calculate before changing
        # the cost basis.
        self.realized_pl_pct = (
            self.compute_pl(price)
        )

        if action == "BUY":
            self.total_price += (
                price * qty
            )

            if position_qty_after > 0:
                self.cost_price = (
                    self.total_price
                    / position_qty_after
                )

        elif action == "SELL":
            self.total_price -= (
                self.cost_price
                * qty
            )

            if position_qty_after == 0:
                self.total_price = 0.0
                self.cost_price = 0.0

        else:
            raise ValueError(
                f"Unsupported action: "
                f"{action}"
            )

        self.position_open = (
            position_qty_after > 0
        )

        self.max_position_sell = (
            position_qty_after
        )

    def reset_trade_state(self):
        self.trade_qty = 0
        self.realized_pl_pct = 0.0

    # ========================================================
    # OUTPUT
    # ========================================================

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
                "order_id"
                in order_data.columns
                and not order_data.empty
            ):
                order_id = (
                    order_data[
                        "order_id"
                    ].iloc[0]
                )

            if (
                "order_status"
                in order_data.columns
                and not order_data.empty
            ):
                order_status = (
                    order_data[
                        "order_status"
                    ].iloc[0]
                )

        candle = {
            # Candle
            "code": row["code"],
            "time": row["time_key"],
            "open": row["open"],
            "close": row["close"],

            # Price statistics
            "pct_diff": self.pct_diff,
            "cum_sum_pct": (
                self.cum_sum_pct
            ),

            # Moving averages
            "short_sma": self.short_sma,
            "long_sma": self.long_sma,

            # RSI
            "RSI": self.rsi,

            # MACD
            "MACD": self.macd,
            "MACD Signal": (
                self.macd_signal
            ),
            "MACD Histogram": (
                self.macd_histogram
            ),
            "MACD_up": (
                self.macd
                > self.macd_signal
            ),

            # Market
            "market_trend": (
                self.market_trend
            ),

            # Position
            "Position": (
                "OPEN"
                if self.position_open
                else "CLOSED"
            ),
            "cost_price": (
                self.cost_price
            ),
            "total_price": (
                self.total_price
            ),

            # Capacity
            "max_cash_buy": (
                self.max_cash_buy
            ),
            "max_position_sell": (
                self.max_position_sell
            ),

            # Trade
            "action": action,
            "trade_qty": (
                self.trade_qty
            ),

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
            "order_status": (
                order_status
            ),

            # Execution
            "execution_time": pd.NaT,
            "execution_price": np.nan,

            # Fees
            "fee_amount": 0.0,
            "fee_details": None,
        }

        self.output.append(
            candle
        )