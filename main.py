import time
import os
import argparse
import talib
import re
import pandas as pd
import numpy as np
from moomoo import *
from pandas.tseries.offsets import BDay
from pathlib import Path
from config import pwd_unlock

# ================= CONFIG =================
# SYMBOL = "HK.00700"
SYMBOL = "US.AAPL"
RSI_threshold_follow = 55
RSI_threshold_revert = 35
RSI_PERIOD = 14
SHORT_WINDOW = 12
LONG_WINDOW = 26
MACD_SIGNAL = 9

window_length = max(RSI_PERIOD+1, LONG_WINDOW + MACD_SIGNAL+1)
PROFIT_PCT = 1.5
LOSS_PCT = -1.0

# ============================================================
# STRATEGY CLASS
# ============================================================

class MovingAverageStrategy:
    def __init__(self):
        self.output = []
        self.prices = []
        self.prev_vwap = 0
        self.vwap = 0
        self.cum_sum_pct = 0
        self.cum_turnover = 0
        self.cum_volume = 0

    def update_state_from_row(self, row, init=False):

        # From previous/saved data
        prev_price = self.prices[-1] if self.prices else 0

         # Current time
        current_price = row['close']
        self.prices.append(current_price)
        turnover = row['turnover']
        volume = row['volume']

        # Skip vwap computation during init
        if init == False:
            # Update vwap -> prev_vwap
            self.prev_vwap = self.vwap
            
            # Update cumulative vwap
            self.cum_turnover += turnover
            self.cum_volume += volume
            self.vwap = self.cum_turnover / self.cum_volume if self.cum_volume else 0
       
        # Compute pct_diff
        if len(self.prices) <= 1 :
            self.pct_diff = 0
        else:
            self.pct_diff = (current_price - prev_price) / prev_price * 100
        self.cum_sum_pct += self.pct_diff
        
        # Compute short SMA if enough prices, else 0
        if len(self.prices) >= SHORT_WINDOW:
            self.short_sma = sum(self.prices[-SHORT_WINDOW:]) / SHORT_WINDOW
        else:
            self.short_sma = 0

        # Compute long SMA if enough prices, else 0
        if len(self.prices) >= LONG_WINDOW:
            self.long_sma = sum(self.prices[-LONG_WINDOW:]) / LONG_WINDOW
        else:
            self.long_sma = 0

        # Compute RSI if enough prices, else 0
        if len(self.prices) >= RSI_PERIOD+1:
            self.rsi = talib.RSI(np.array(self.prices[-(RSI_PERIOD+1):]), timeperiod=RSI_PERIOD)[-1]
        else:
            self.rsi = 0

        # Compute MACD if enough prices, else 0
        if len(self.prices) >= LONG_WINDOW + MACD_SIGNAL+1:
            macd, macd_signal, macd_histogram = talib.MACD(
                np.array(self.prices[-(LONG_WINDOW + MACD_SIGNAL+1):]),
                fastperiod=SHORT_WINDOW,
                slowperiod=LONG_WINDOW,
                signalperiod=MACD_SIGNAL
            )
            self.macd = macd[-1]
            self.macd_signal = macd_signal[-1]
            self.macd_histogram = macd_histogram[-1]
        else:
            self.macd, self.macd_signal, self.macd_histogram = 0, 0, 0
    
    def compute_pl(self, current_price):
        # unrealized -> to compute pl before BUY/SELL (use current cost_price)
        # realized -> to compute pl after BUY/SELL (use previous cost_price)

        # position OPEN (BUY/SELL)
        if self.cost_price > 0:
            self.position_open = True
            pl_pct = (current_price - self.cost_price) / self.cost_price * 100
        # position CLOSED (SELL)
        else:
            self.position_open = False
            pl_pct = 0
    
        return pl_pct

    def buy_or_sell(self, pl_pct=0):
        action = "HOLD"
        
        # buy ratio 0 < x < 1
        trend_strength = self.macd - self.macd_signal
        buy_ratio = max(0, min(0.7, trend_strength/0.005))
        if self.max_cash_buy > 0:
            buy_qty = int(self.max_cash_buy * buy_ratio)
        else:
            buy_qty = 0

        # sell ratio 0 < x < 1
        sell_ratio = min(1, abs(pl_pct/LOSS_PCT))
        if self.max_position_sell > 0:
            sell_qty = max(1, int(self.max_position_sell * sell_ratio))
        else:
            sell_qty = 0
        
        # A: trend following
        if self.market_trend > 0:
            buy_signal = (
                buy_qty > 0
                and trend_strength > 0
                and self.rsi > RSI_threshold_follow
            )
        # B: mean reversion
        else:
             buy_signal = (
                buy_qty > 0
                and trend_strength < 0
                and self.rsi < RSI_threshold_revert
            )

        # if pl_pct >= PROFIT_PCT, only sell when hit loss pct or above, regardless of MACD signal (take profit)
        if pl_pct >= PROFIT_PCT:
            sell_signal = (
                sell_qty > 0
                and self.macd < self.macd_signal
                and self.rsi < RSI_threshold_revert
            )
        else:
        # if not hit profit pct, sell when MACD signal is unfavorable or hit loss pct (cut loss)
            sell_signal = (
                sell_qty > 0
                and (self.macd < self.macd_signal
                or pl_pct <= LOSS_PCT)
            )
        if buy_signal:
            action = "BUY"
        if sell_signal:
            action = "SELL"
            
        return action, buy_qty, sell_qty

    def save_output(self, row, action, order_data=None):
        candle_dict = {
            "code": row['code'],
            "time": row['time_key'],
            "open": row['open'],
            "close": row['close'],
            "pct_diff": self.pct_diff,
            "short_sma": self.short_sma,
            "long_sma": self.long_sma,
            # "vwap": self.vwap,
            # "prev_vwap": self.prev_vwap,
            # "vwap_up": self.vwap > self.prev_vwap,
            "RSI": self.rsi,
            "MACD": self.macd,
            "MACD Signal": self.macd_signal,
            "MACD_up": (self.macd - self.macd_signal) > 0,
            "hit_profit": self.unrealized_pl_pct >= PROFIT_PCT,
            "hit_loss": self.unrealized_pl_pct <= LOSS_PCT,
            "MACD Histogram": self.macd_histogram,
            "cost_price": self.cost_price,
            "total_price": self.total_price,
            "max_cash_buy": self.max_cash_buy,
            "max_position_sell": self.max_position_sell,
            "trade_qty": self.trade_qty,
            "action": action,
            "order_id": order_data['order_id'].iloc[0] if order_data is not None else None,
            "order_status": order_data['order_status'].iloc[0] if order_data is not None else None,
            "fee_amount": 0,
            "fee_details": 0,
            "execution_time": "NA",
            "execution_price": "NA",
            "Position": "OPEN" if self.position_open else "CLOSED",
            "unrealized_pl_pct": self.unrealized_pl_pct,
            "realized_pl_pct": self.realized_pl_pct,
            "cum_sum_pct": self.cum_sum_pct,
            "market_trend": self.market_trend
        }

        self.output.append(candle_dict)
# ============================================================
# MATCHING YOUR place_order FUNCTION
# ============================================================
def place_order(trade_ctx, price, symbol, qty, side, order_type, trd_env):
    """Place a LIMIT/MARKET order"""
    ret, data = trade_ctx.place_order(
        price=price,
        qty=qty,
        code=symbol,
        trd_side=side,
        order_type=order_type,
        trd_env=trd_env
    )
    if ret == RET_OK:
        print(f"✅ Order executed: {side} {qty} {symbol}")
    else:
        print(f"❌ Order failed: {side} {symbol} | {data}")
    return data

def get_position_status(trade_ctx, trade_env):
    """Check if there's an open position for the symbol and return positions"""
    ret, positions = trade_ctx.position_list_query(trd_env=trade_env)
    if ret != RET_OK:
        print("Error fetching positions:", positions)
        return None
    
    # for CLOSED positions
    cost_price = 0
    for _, row in positions.iterrows():
        # for OPEN positions, find latest cost price
        if SYMBOL == row['code']:
            cost_price = row['cost_price']
            break
    return cost_price
    
def get_available_qty(trade_ctx, current_price, lot_size, trade_env):
    ret, max_qty_to_trade = trade_ctx.acctradinginfo_query(order_type=OrderType.NORMAL, code=SYMBOL, price=current_price, trd_env=trade_env)
    if ret != RET_OK:
        print("Error fetching trading info:", max_qty_to_trade)
        return 0
    
    max_cash_buy = max_qty_to_trade['max_cash_buy'].iloc[0] // lot_size
    max_position_sell = max_qty_to_trade['max_position_sell'].iloc[0] // lot_size

    return max_cash_buy, max_position_sell

def get_market_trend_live(quote_ctx):
    trend_code = "HK.800000" if SYMBOL.startswith("HK.") else "US.SPY"
    ret, df_market = quote_ctx.get_market_snapshot([trend_code])

    if ret != RET_OK:
        print("Error fetching market trend:", df_market)
        return None

    return (
        df_market.loc[0, "last_price"]
        - df_market.loc[0, "prev_close_price"]
    )

def get_market_trend_simulation(quote_ctx, last_day=None):
    trend_code = "HK.800000" if SYMBOL.startswith("HK.") else "US.SPY"
    if last_day is None:
        raise ValueError("last_day must be provided in backtest mode.")

    ret, df_market, _ = quote_ctx.request_history_kline(
        trend_code,
        last_day.strftime("%Y-%m-%d"),
        last_day.strftime("%Y-%m-%d"),
        SubType.K_1M,
        AuType.NONE,
    )

    if ret != RET_OK:
        print("Error fetching market trend:", df_market)
        return None

    df_market["time_key"] = pd.to_datetime(df_market["time_key"])
    return df_market

def initialize_rows(strategy, trade_ctx, quote_ctx, job_start_date, lot_size):
    
    prev_date = (job_start_date - BDay(5)).strftime('%Y-%m-%d')
    end_date = job_start_date.strftime('%Y-%m-%d')

    full_df = []
    ret, historical_df, page_req_key = quote_ctx.request_history_kline(
        SYMBOL,
        prev_date,
        end_date,
        SubType.K_1M, 
        AuType.NONE
    )
    full_df.append(historical_df)
    while page_req_key != None: # Request all results after
        ret, historical_df, page_req_key = quote_ctx.request_history_kline(
        SYMBOL,
        prev_date,
        end_date,
        SubType.K_1M, 
        AuType.NONE,
        page_req_key=page_req_key
    )
        if ret != RET_OK:
            print("Error fetching historical data:", historical_df)
        full_df.append(historical_df)

    full_historical_df = pd.concat(full_df, ignore_index=True)
    full_historical_df["time_key"] = pd.to_datetime(full_historical_df["time_key"])
    full_historical_df["date"] = full_historical_df["time_key"].dt.date

    # Intialize first window_length-1 candles to fill the strategy state
    if live_mode:
        df_past = full_historical_df.iloc[-window_length+1:].copy()
        df_current = None
        last_day = None
    else:
        # get last 2 full trading days
        counts = full_historical_df.groupby("date").size()
        full_days = counts[counts >= 390].index.sort_values()
        # second last FULL day
        second_last_day = full_days[-2]   
        prev_day_df = full_historical_df[full_historical_df["date"] == second_last_day]
        df_past = prev_day_df.iloc[-window_length+1:].copy()
        # last FULL day
        last_day = full_days[-1]
        df_current = full_historical_df[full_historical_df["date"] == last_day].copy()

    for i in range(len(df_past)):
        
        row = df_past.iloc[i]

        # Update state
        strategy.update_state_from_row(row, init=True)
        current_price = strategy.prices[-1]

        # Get market trend
        strategy.market_trend = 0

        # Cannot call so many times in init, so only call once, since get_available_qty doesnt change during init
        if i == 0:
            max_cash_buy, max_position_sell = get_available_qty(trade_ctx, current_price, lot_size, trade_env)
            if live_mode:
                strategy.max_cash_buy = max_cash_buy
                strategy.max_position_sell = max_position_sell
            else:
                strategy.max_cash_buy = max_cash_buy + max_position_sell
                strategy.max_position_sell = 0

        if live_mode:
            # only call cost_price once during init, since cost_price is updated in OrderHandler after BUY/SELL
            # get_position_status also updates position_open
            strategy.cost_price = get_position_status(trade_ctx, trade_env)
            # compute unrealized P/L before buy_or_sell decision
            strategy.unrealized_pl_pct = strategy.compute_pl(current_price)
        else:   
            strategy.cost_price = 0
            strategy.unrealized_pl_pct = 0
            strategy.position_open = False

        # Decide action
        action = "INITIALIZING"

        # place_order + OrderHandler
        # Since no order, next_max_cash_buy and next_max_position_sell are not updated during init
        strategy.trade_qty = 0
        strategy.realized_pl_pct = 0

        strategy.total_price = strategy.cost_price * strategy.max_position_sell
        # cost price not updated during init
        strategy.save_output(row, action, order_data=None)

    print("Initialized time: ", df_past['time_key'].iloc[-1])
    return df_current, last_day

def compute_daily_pl(output_df, price):

    sell_df = output_df.loc[output_df['action'] == 'SELL'].copy()

    exposure = 0
    peak_exposure = 0

    for _, row in output_df.iterrows():

        if row["action"] == "BUY":
            exposure += (row[price] * row['trade_qty'])

        elif row["action"] == "SELL":
            exposure -= (row[price] * row['trade_qty'])

        # track peak capital used
        peak_exposure = max(peak_exposure, exposure)
    print(f"Peak Exposure: {peak_exposure:.0f}")
    
    sell_df['realized_pl'] = (sell_df[price] - sell_df['cost_price']) * sell_df['trade_qty']
    realized_pl_sum = sell_df['realized_pl'].sum()
    if peak_exposure > 0:
        realized_pl = realized_pl_sum/peak_exposure * 100
    else:
        realized_pl = 0

    print(f"Total Return: {realized_pl_sum:.0f}, {realized_pl:.3f}%")

    return sell_df, realized_pl_sum, peak_exposure, realized_pl

def get_daily_status(trade_ctx, realized_pl_sum, peak_exposure, realized_pl, logs_folder, daily_status_file_name, trade_env):
    # Open positions
    ret, positions = trade_ctx.position_list_query(trd_env=trade_env)
    if ret != RET_OK:
        print("Error fetching positions:", positions)
        return None  # or handle the error appropriately

    columns = [
        "code",
        "qty",
        "nominal_price",
        "cost_price",
        "average_cost",
        "market_val",
        "pl_ratio",
        "pl_ratio_avg_cost",
    ]

    # Filter to the target symbol with a positive position
    df = positions.loc[
        (positions["code"] == SYMBOL),
        columns
    ].copy()

    # Create a default row if no position exists
    if df.empty:
        df = pd.DataFrame([{
            "code": SYMBOL,
            "qty": 0,
            "nominal_price": 0,
            "cost_price": 0,
            "average_cost": 0,
            "market_val": 0,
            "pl_ratio_avg_cost": 0,
            "pl_ratio": 0,
        }])
    
    # Total assets
    ret, acc = trade_ctx.accinfo_query(trd_env=trade_env)

    if ret != RET_OK:
        print("Error fetching account info:", acc)

    total_assets = acc.loc[0, 'total_assets']

    # Create final cost
    df["final_cost_price"] = df["average_cost"].mask(
        df["average_cost"] == 0,
        df["cost_price"]
    )

    # Create final P/L ratio
    df["unrealized_pl_ratio"] = df["pl_ratio_avg_cost"].mask(
        df["pl_ratio_avg_cost"] == 0,
        df["pl_ratio"]
    )
    # Convert to numeric
    df["unrealized_pl_ratio"] = pd.to_numeric(
        df["unrealized_pl_ratio"],
        errors="coerce"
    )

    df['date'] = job_start_date
    df['total_assets'] = total_assets
    df['calculated_realized_pl_sum'] = realized_pl_sum
    df['calculated_realized_pl_ratio'] = realized_pl
    df['calculated_peak_exposure'] = peak_exposure

    files = list(Path(logs_folder).glob(f"*{daily_status_file_name}"))
    prev_df = None

    if files:

        def extract_date(f):
            match = re.search(r"\d{4}-\d{2}-\d{2}", f.name)
            return pd.to_datetime(match.group()) if match else pd.Timestamp.min

        prev_file = max(files, key=extract_date)
        file_date = extract_date(prev_file).date()

        if file_date != pd.Timestamp(pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')).date():
            print("Previous file: ", prev_file)
            prev_df = pd.read_csv(prev_file)

    if prev_df is not None:
        df = pd.concat([prev_df, df], ignore_index=True)
    
    df['unrealized_pl_sum'] = df['unrealized_pl_ratio']/100 * df['market_val']
    df['asset_difference'] = df['total_assets'].diff()
    df['asset_difference_ratio'] = df['total_assets'].pct_change() * 100
    df['calculated_pl_sum'] = df['calculated_realized_pl_sum'] + df['unrealized_pl_sum']

    df = df[['date', 'code', 'qty', 'nominal_price', 'cost_price', 'average_cost', 'final_cost_price', 'market_val', 'pl_ratio', 'pl_ratio_avg_cost', 'unrealized_pl_ratio', 'unrealized_pl_sum',
             'calculated_realized_pl_ratio', 'calculated_realized_pl_sum', 'calculated_peak_exposure', 'calculated_pl_sum',
             'total_assets', 'asset_difference', 'asset_difference_ratio']]
    return df
# ============================================================
# QUOTE CALLBACK
# ============================================================
class KlineHandler(CurKlineHandlerBase):
    
    def __init__(self, strategy, quote_ctx, trade_ctx, lot_size):
        super().__init__()
        self.strategy = strategy
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.lot_size = lot_size
        self.prev_candle = None

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret != RET_OK:
            print("Kline error:", data)
            return RET_ERROR, data

        current_candle = data.iloc[-1]
        # When new candle starts, process prev_candle
        if self.prev_candle is None:
            # first candle, no action, just save output
            self.prev_candle = current_candle
            action ='HOLD'
            self.strategy.save_output(self.prev_candle, action, order_data=None)

        if current_candle['time_key'] != self.prev_candle['time_key']:
            
            candle_to_process = self.prev_candle
            # Mark this candle as processed immediately
            self.prev_candle = current_candle
            print(f"Current time: {candle_to_process['time_key']}, Current price:  {candle_to_process['close']}")
            
            # Update state
            self.strategy.update_state_from_row(candle_to_process, init=False)
            current_price = self.strategy.prices[-1]

            # Get market trend
            self.strategy.market_trend = get_market_trend_live(self.quote_ctx)

            # get available qty for buy/sell        
            self.strategy.max_cash_buy, self.strategy.max_position_sell = get_available_qty(self.trade_ctx, current_price, self.lot_size, self.trade_ctx.get_trading_env())
            
            # update cost price if max_position_sell is 0 (position closed)
            if self.strategy.max_position_sell == 0:
                self.strategy.cost_price = 0

            # compute unrealized P/L before buy_or_sell decision
            self.strategy.unrealized_pl_pct = self.strategy.compute_pl(current_price)    
            # Decide action
            action, buy_qty, sell_qty = self.strategy.buy_or_sell(self.strategy.unrealized_pl_pct)

            BUY_QTY = self.lot_size * buy_qty
            SELL_QTY = self.lot_size * sell_qty
            # Execute action in live mode
            if action == "BUY":
                print("Max QTY to Buy:", self.strategy.max_cash_buy)
                order_data = place_order(self.trade_ctx, self.strategy.prices[-1], SYMBOL, BUY_QTY, TrdSide.BUY, OrderType.MARKET, trade_env)
                self.strategy.trade_qty = BUY_QTY
            elif action == "SELL":
                print("Max QTY to Sell:", self.strategy.max_position_sell)
                order_data = place_order(self.trade_ctx, self.strategy.prices[-1], SYMBOL, SELL_QTY, TrdSide.SELL, OrderType.MARKET, trade_env)
                self.strategy.trade_qty = SELL_QTY
            else:
                order_data = None
                self.strategy.realized_pl_pct = 0
                self.strategy.trade_qty = 0
            self.strategy.save_output(candle_to_process, action, order_data)

        return RET_OK, data

# ============================================================
# ORDER CALLBACK
# ============================================================
class OrderHandler(TradeOrderHandlerBase):
    
    def __init__(self, strategy, trade_ctx, lot_size, trade_env):
        super().__init__()
        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.lot_size = lot_size
        self.trade_env = trade_env

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret != RET_OK:
            print("❌ Order callback error:", data)
            return RET_ERROR, data
        
        if len(data) != 1:
            print("❌ Order callback error: unexpected data length:", len(data))
            return RET_ERROR, data
        
        for o in self.strategy.output:
            if o['order_id'] == data['order_id'].iloc[0]:
                order_status = data['order_status'].iloc[0]
                o['order_status'] = order_status

                # Real env ==> in DealHandler, not here
                if self.trade_env == TrdEnv.REAL:
                    ret2, order_fee = self.trade_ctx.order_fee_query(data['order_id'].iloc[0])
                    if ret2 != RET_OK:
                        print("❌ Order Fee error:", order_fee)
                        return RET_ERROR, order_fee
                    o['fee_amount'] = order_fee['fee_amount'].iloc[0]
                    o['fee_details'] = order_fee['fee_details'].iloc[0]

                elif self.trade_env == TrdEnv.SIMULATE and order_status == "FILLED_ALL":
                    action = data['trd_side'].iloc[0]
                    current_price = data['dealt_avg_price'].iloc[0]

                    # realized -> get cost_price after compute pl
                    self.strategy.realized_pl_pct = self.strategy.compute_pl(current_price)
                    match action:
                        # update cost price if BUY
                        case 'BUY':
                            # Total price changes after BUY AND SELL
                            self.strategy.total_price += current_price * self.strategy.trade_qty
                            # Cost price changes after BUY, but not after SELL
                            self.strategy.cost_price = self.strategy.total_price /(self.strategy.max_position_sell + self.strategy.trade_qty)
                            o['cost_price'] = self.strategy.cost_price

                        case 'SELL':
                            # Total price changes after BUY AND SELL
                            self.strategy.total_price -= self.strategy.cost_price * self.strategy.trade_qty
                            # Cost price not updated after SELL

                    o['total_price'] = self.strategy.total_price
                    o['execution_time'] = data['updated_time'].iloc[0]
                    o['execution_price'] = current_price
                    o['realized_pl_pct'] = self.strategy.realized_pl_pct
                    o['Position'] = "OPEN" if self.strategy.position_open else "CLOSED"
                    
                    print(f"{SYMBOL} | Price:{o['execution_price']:.2f} "
                    f"| Action:{action} "
                    f"| Time:{o['execution_time']}")
                    if action == 'SELL':
                        print(f"|Cost Price:{o['cost_price']}, Sell Price:{o['execution_price']},  Profit:{o['realized_pl_pct']:.2f}")
                    break

        return RET_OK, data

# ============================================================
# DEAL CALLBACK (REAL ENV ONLY)
# ============================================================
class DealHandler(TradeDealHandlerBase):
    def __init__(self, strategy, trade_ctx, lot_size):
        super().__init__()
        self.strategy = strategy
        self.trade_ctx = trade_ctx
        self.lot_size = lot_size

    def on_recv_rsp(self, rsp_pb):
        ret, data = super(DealHandler, self).on_recv_rsp(rsp_pb)
        if ret != RET_OK:
            print("❌ Deal callback error:", data)
            return RET_ERROR, data
        
        for o in self.strategy.output:
            if o['order_id'] == data['order_id'].iloc[0]:

                action = data['trd_side'].iloc[0]
                current_price = data['price'].iloc[0]

                # realized -> get cost_price after compute pl
                self.strategy.realized_pl_pct = self.strategy.compute_pl(current_price)
                match action:
                    # update cost price if BUY
                    case 'BUY':
                        # Total price changes after BUY AND SELL
                        self.strategy.total_price += current_price * self.strategy.trade_qty
                        # Cost price changes after BUY, but not after SELL
                        self.strategy.cost_price = self.strategy.total_price /(self.strategy.max_position_sell + self.strategy.trade_qty)
                        o['cost_price'] = self.strategy.cost_price

                    case 'SELL':
                        # Total price changes after BUY AND SELL
                        self.strategy.total_price -= self.strategy.cost_price * self.strategy.trade_qty
                        # Cost price not updated after SELL

                o['total_price'] = self.strategy.total_price
                o['execution_time'] = data['updated_time'].iloc[0]
                o['execution_price'] = current_price
                o['realized_pl_pct'] = self.strategy.realized_pl_pct
                o['Position'] = "OPEN" if self.strategy.position_open else "CLOSED"
                
                print(f"{SYMBOL} | Price:{o['execution_price']:.2f} "
                f"| Action:{action} "
                f"| Time:{o['execution_time']}")
                if action == 'SELL':
                    print(f"|Cost Price:{o['cost_price']}, Sell Price:{o['execution_price']},  Profit:{o['realized_pl_pct']:.2f}")
                break
        
        return RET_OK, data
# ============================================================
# START
# ============================================================

def start(job_start_date, trade_env):
    if SYMBOL.startswith("HK."):
        trade_market = TrdMarket.HK
        market = 'market_hk'
    elif SYMBOL.startswith("US."):
        trade_market = TrdMarket.US
        market = 'market_us'

    strategy = MovingAverageStrategy()

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_ctx = OpenSecTradeContext(
            filter_trdmarket=trade_market,
            host="127.0.0.1",
            port=11111,
            security_firm=SecurityFirm.FUTUSG
    )
    if trade_env == TrdEnv.REAL:
        ret, data = trade_ctx.unlock_trade(password_md5=pwd_unlock)
        if ret != RET_OK:
            print(f"Unlock trade failed: {data}")

    ret, stock_data = quote_ctx.get_stock_basicinfo(
        market=trade_market,
        stock_type=SecurityType.STOCK,
        code_list=SYMBOL
    )
    lot_size = stock_data['lot_size'].iloc[0]

    # df_current and last_day only used for backtesting, not live mode
    df_current, last_day = initialize_rows(strategy, trade_ctx, quote_ctx, job_start_date, lot_size)

    if live_mode:    
        trade_ctx.set_handler(OrderHandler(strategy, trade_ctx, lot_size, trade_env))
        quote_ctx.set_handler(KlineHandler(strategy, quote_ctx, trade_ctx, lot_size))
        ret, data = quote_ctx.subscribe([SYMBOL], [SubType.K_1M], subscribe_push=True)
        if ret != RET_OK:
            print(f"Subscription failed: {data}")

    if trade_env == TrdEnv.REAL:
        trade_ctx.set_handler(DealHandler(strategy, trade_ctx, lot_size))

    mode = "LIVE TRADING" if live_mode else "SIMULATION MODE"
    print(f"🚀 Started ({mode})")
    print("Press Ctrl+C to exit.")
    if live_mode:
        while True:
            ret, df_state = quote_ctx.get_global_state()
            if ret != RET_OK:
                print(f"[QUOTE] get_global_state failed, ret={df_state}")
            if df_state[market] in ['AFTER_HOURS_BEGIN', 'CLOSED']:
                print("LOOP EXITED: Market closed")
                return strategy, quote_ctx, trade_ctx
            time.sleep(1)
    else:
        # simulation mode only, call historical data for market trend
        df_market = get_market_trend_simulation(quote_ctx, last_day)
        # simulation mode only, initialize next_max_cash_buy and next_max_position_sell to current values, since they will be updated in the loop
        next_max_cash_buy = strategy.max_cash_buy
        next_max_position_sell = strategy.max_position_sell

        for _, row in df_current.iterrows():
            # Update state
            strategy.update_state_from_row(row, init=False)
            current_price = strategy.prices[-1]
            
            # Get market trend
            curr_time = row['time_key']
            strategy.market_trend = df_market.loc[df_market['time_key'] == curr_time, 'close'].iloc[0] - df_market.loc[df_market['time_key'] == curr_time, 'last_close'].iloc[0]
            
            # Manual function for get_available_qty
            strategy.max_cash_buy = next_max_cash_buy
            strategy.max_position_sell = next_max_position_sell

            # update cost price if max_position_sell is 0 (position closed)
            if strategy.max_position_sell == 0:
                strategy.cost_price = 0

            # compute unrealized P/L before buy_or_sell decision
            strategy.unrealized_pl_pct = strategy.compute_pl(current_price)
            # Decide action
            action, buy_qty, sell_qty = strategy.buy_or_sell(strategy.unrealized_pl_pct)
            
            # place_order + OrderHandler
            if action == 'BUY':
                next_max_cash_buy = strategy.max_cash_buy - buy_qty
                next_max_position_sell = strategy.max_position_sell + buy_qty

                strategy.trade_qty = buy_qty * lot_size
                strategy.realized_pl_pct = strategy.compute_pl(current_price)

                strategy.total_price += current_price * strategy.trade_qty
                strategy.cost_price = strategy.total_price / next_max_position_sell
                print(f"BUY | {strategy.trade_qty} {SYMBOL} | Cost: {strategy.cost_price:.2f}")
            elif action == 'SELL':
                next_max_cash_buy = strategy.max_cash_buy + sell_qty
                next_max_position_sell = strategy.max_position_sell - sell_qty

                strategy.trade_qty = sell_qty * lot_size
                strategy.realized_pl_pct = strategy.compute_pl(current_price)

                strategy.total_price -= strategy.cost_price * strategy.trade_qty
                # cost price not updated
                print(f"SELL | {strategy.trade_qty} {SYMBOL} | Cost: {strategy.cost_price:.2f} | Profit: {strategy.realized_pl_pct:.2f}")
            elif action == 'HOLD':
                next_max_cash_buy = strategy.max_cash_buy
                next_max_position_sell = strategy.max_position_sell
                strategy.trade_qty = 0
                strategy.realized_pl_pct = 0
                # total price, cost price remain the same

            # Manual function to update position status, as in compute_pl (after order execution), done before current iteration
            if next_max_position_sell > 0:
                strategy.position_open = True
            else:
                strategy.position_open = False

            print(f"Current time: {row['time_key']}, Current price:  {row['close']}, Unrealized P/L: {strategy.unrealized_pl_pct}")
            strategy.save_output(row, action, order_data=None)

    return strategy, quote_ctx, trade_ctx

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Run the trading bot in live or test mode.')
    argparser.add_argument('--live', default=False, action='store_true', help='Run the bot in live mode (default is test mode)')
    argparser.add_argument('--date', default=pd.Timestamp.today(), help='Current date, or previous date for backtesting')
    argparser.add_argument('--env', default='simulate', help='Trading environment (default is SIMULATE)')
    args = argparser.parse_args()
    live_mode = args.live
    match args.env:
        case 'real':
            trade_env = TrdEnv.REAL
        case 'simulate':
            trade_env = TrdEnv.SIMULATE

    if SYMBOL.startswith("HK."):
        timezone_date = pd.to_datetime(args.date).tz_localize("Asia/Hong_Kong")
    elif SYMBOL.startswith("US."):
        timezone_date = pd.to_datetime(args.date).tz_localize("America/New_York")

    if live_mode:
        job_start_date = timezone_date
    else:
        job_start_date = pd.to_datetime(str(timezone_date) + ' 23:59:00')
    
    try:
        strategy, quote_ctx, trade_ctx = start(job_start_date, trade_env)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if len(strategy.output):
            output_df = pd.DataFrame(strategy.output)
            output_df['time_SG'] = pd.to_datetime(output_df['time']).dt.tz_localize('America/New_York').dt.tz_convert('Asia/Singapore')
            
            if live_mode:
                mode = "live"
                price = "execution_price"
            else:
                mode = "backtest"
                price = "close"

            trading_logs_file_name = f"{mode}_{args.env}_trading_logs.csv"
            pl_file_name = f"{mode}_{args.env}_pl.csv"
            daily_status_file_name = f"{mode}_{args.env}_daily_status.csv"

            logs_folder = os.path.join(os.getcwd(), 'logs')
            trading_logs_path = os.path.join(logs_folder, f"{pd.Timestamp.today().strftime('%Y-%m-%d %H_%M_%S')} - {trading_logs_file_name}")
            output_df.to_csv(trading_logs_path)

            sell_df, realized_pl_sum, peak_exposure, realized_pl = compute_daily_pl(output_df, price)
            pl_path = os.path.join(logs_folder, f"{pd.Timestamp.today().strftime('%Y-%m-%d %H_%M_%S')} - {pl_file_name}")
            sell_df.to_csv(pl_path)

            daily_status = get_daily_status(trade_ctx, realized_pl_sum, peak_exposure, realized_pl, logs_folder, daily_status_file_name, trade_env)
            daily_status_path = os.path.join(logs_folder, f"{pd.Timestamp.today().strftime('%Y-%m-%d %H_%M_%S')} - {daily_status_file_name}")
            daily_status.to_csv(daily_status_path, index=False)

            quote_ctx.close()
            trade_ctx.close()
