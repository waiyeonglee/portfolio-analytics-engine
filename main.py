import argparse

import pandas as pd

from backtest import run_backtest
from broker import (
    close_contexts,
    create_contexts,
)
from runtime import create_runtime
from live_trading import run_live
from reporting import save_results
from strategy import MovingAverageStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the trading bot in "
            "live or backtest mode."
        )
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode.",
    )

    parser.add_argument(
        "--date",
        default=pd.Timestamp.today(),
        help="Date for backtesting.",
    )

    parser.add_argument(
        "--env",
        choices=[
            "real",
            "simulate",
        ],
        default="simulate",
        help="Trading environment.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = create_runtime(args)

    quote_ctx = None
    trade_ctx = None
    strategy = None

    try:
        (
            quote_ctx,
            trade_ctx,
            lot_size,
        ) = create_contexts(config)

        strategy = MovingAverageStrategy(
            lot_size=lot_size
        )

        if config.live_mode:
            run_live(
                strategy,
                quote_ctx,
                trade_ctx,
                config,
            )

        else:
            run_backtest(
                strategy,
                quote_ctx,
                trade_ctx,
                config,
            )

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        if (
            strategy is not None
            and strategy.output
            and trade_ctx is not None
        ):
            save_results(
                strategy,
                trade_ctx,
                config,
            )

        close_contexts(
            quote_ctx,
            trade_ctx,
        )


if __name__ == "__main__":
    main()