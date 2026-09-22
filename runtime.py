from dataclasses import dataclass

import pandas as pd
from moomoo import TrdEnv


@dataclass
class RuntimeSettings:
    symbol: str
    trade_env: TrdEnv
    live_mode: bool
    timezone_date: pd.Timestamp
    env_name: str


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

    timezone_date = pd.to_datetime(args.date)

    if timezone_date.tzinfo is None:
        timezone_date = timezone_date.tz_localize(
            "Asia/Singapore"
        )

    if symbol.startswith("HK."):
        timezone_date = timezone_date.tz_convert(
            "Asia/Hong_Kong"
        )

    elif symbol.startswith("US."):
        timezone_date = timezone_date.tz_convert(
            "America/New_York"
        )

    return RuntimeSettings(
        symbol=symbol,
        trade_env=trade_env,
        live_mode=args.live,
        timezone_date=timezone_date,
        env_name=args.env,
    )