from dataclasses import dataclass

import pandas as pd
from moomoo import TrdEnv


@dataclass
class RuntimeSettings:
    symbol: str
    trade_env: TrdEnv
    live_mode: bool
    market_datetime: pd.Timestamp
    env_name: str
    currency: str


def create_runtime(args):
    if args.env == "real":
        trade_env = TrdEnv.REAL
        symbol = "US.META"

    elif args.env == "simulate":
        trade_env = TrdEnv.SIMULATE
        symbol = "US.AAPL"

    else:
        raise ValueError(
            f"Unsupported environment: {args.env}"
        )

    market_datetime = pd.to_datetime(args.date)

    if market_datetime.tzinfo is None:
        market_datetime = market_datetime.tz_localize(
            "Asia/Singapore"
        )
        currency = "SGD"

    if symbol.startswith("HK."):
        market_datetime = market_datetime.tz_convert(
            "Asia/Hong_Kong"
        )
        currency = "HKD"


    elif symbol.startswith("US."):
        market_datetime = market_datetime.tz_convert(
            "America/New_York"
        )
        currency = "USD"

    return RuntimeSettings(
        symbol=symbol,
        trade_env=trade_env,
        live_mode=args.live,
        market_datetime=market_datetime,
        env_name=args.env,
        currency=currency
    )