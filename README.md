# Portfolio Analytics Engine

Python trading and portfolio analytics application for Moomoo/Futu. It supports historical backtesting and live trading, with a shared strategy and portfolio state engine.

## Requirements

- Python 3.10–3.14
- [`uv`](https://docs.astral.sh/uv/)
- Moomoo OpenD running and reachable by the application

## Local setup

```bash
git clone https://github.com/waiyeonglee/portfolio-analytics-engine.git
cd portfolio-analytics-engine
uv sync
```

Run a backtest:

```bash
uv run python -m scripts.main --date 2026-10-14
```

Run live in the simulated environment:

```bash
uv run python -m scripts.main --live --env simulate
```

For notebook development:

```bash
uv sync --group dev
uv run --group dev jupyter lab Moomoo.ipynb
```

`uv.lock` records the resolved dependencies. Use `uv sync --locked` in deployment environments so dependency installation does not change the lockfile.

**Real trading:** `--live --env real` connects to the real trading environment. Verify the environment and account settings before running it.

## EC2 boot updates

The EC2 updater, [`jobs/moomoo_update.sh`](./jobs/moomoo_update.sh), fast-forward pulls `main` and installs the locked runtime dependencies. Configure the local systemd units so `moomoo_real.service` and `moomoo_simulate.service` both require and start after `moomoo_update.service`. The update then runs before either trading service starts when EC2 boots.

`jobs/` contains the checked-in update, start, and stop scripts. `service_files/` contains machine-specific systemd unit files and is intentionally git-ignored; keep credentials and EC2-specific configuration out of GitHub. The EC2 checkout path used by the scripts is `/home/ubuntu/portfolio_analytics_engine`.

One-time EC2 setup:

1. Clone the repository to `/home/ubuntu/portfolio_analytics_engine` and install `uv` at `/home/ubuntu/.local/bin/uv`.
2. Ensure the `ubuntu` account can pull the repository from GitHub. For a private repository, configure a read-only deploy key on EC2.
3. Copy the local systemd units from `service_files/` to `/etc/systemd/system/`. Keep the existing `opend.service` dependency, and make both trading services require and order after `moomoo_update.service`.
4. Check that the trading units use these start and stop commands:

   ```ini
   # moomoo_real.service
   ExecStart=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/start_moomoo_real.sh
   ExecStopPost=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/end_moomoo.sh real

   # moomoo_simulate.service
   ExecStart=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/start_moomoo_simulate.sh
   ExecStopPost=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/end_moomoo.sh simulate
   ```

5. Reload systemd and enable the services you want to start at boot:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable opend moomoo_update moomoo_simulate
   ```

Enable `moomoo_real` only if you intend real trading to start automatically at boot. The updater is a shared oneshot dependency: if its GitHub pull or locked sync fails, the dependent trading services will not start. Live startup exits cleanly when the market is closed, and waits for at least 30 completed one-minute candles when the market is open but the strategy has not warmed up yet.

## Outputs

The application writes portfolio and trading results to local CSV logs. When a trading service stops, `jobs/end_moomoo.sh` saves that service's journal and the current boot's `moomoo_update` journal, then uploads them and matching environment logs to `s3://moomoos3/logs/`.

Live trading logs include the bid-ask spread-to-midpoint ratio as `bid_ask_spread_return`. It is a decimal (for example, `0.0025` means `0.25%`), matching Ridge's decimal return prediction. The snapshot is captured when the next candle arrives, then stored on the candle that just completed. If Moomoo does not return a valid quote, the field is blank and the candle continues through the strategy.

Moomoo's historical K-line API does not include historical bid/ask quotes. Backtests therefore use a fixed assumed `bid_ask_spread_return` of `0.003` (0.3%); it is an assumption, not historical market data.

The daily status CSV keeps the P/L view concise:

- `daily_realized_pl_sum`: realized P/L for the current run and market date, net of sell-order fees. Prior snapshots from today are excluded, so it does not carry realized P/L forward across same-day restarts or accumulate across dates.
- `unrealized_pl_sum`: current P/L on the open position at this snapshot. It is neither a daily total nor cumulative; it may include gains or losses accumulated over the lifetime of the open position.
- `daily_total_pl_sum`: daily realized P/L plus the daily change in unrealized P/L.
- `daily_unrealized_pl_difference`: daily difference in unrealized P/L, calculated as this snapshot minus the prior market-date snapshot.
- `asset_difference`: difference in total assets since the prior market-date snapshot.
- `reconciliation_difference`: `asset_difference - daily_total_pl_sum`; zero means the reported P/L components match the account asset difference.
- `daily_total_pl_sum_pct`: daily total P/L as a percentage of the current run's initial equity.
- `buy_hold_return_pct`: percentage return for a buy-and-hold benchmark over this run's candle period.
- `intraday_sharpe_ratio`: annualized Sharpe ratio calculated from this run's intraday equity returns and the daily risk-free rate.

The first recorded date has no prior snapshot, so daily unrealized difference, daily total, asset difference, and reconciliation are blank for that row. Unrealized P/L uses Moomoo average cost, keeping realized P/L separate for both real and simulated environments. Live realized P/L uses the actual price and quantity of every fill, including partial fills, and deducts the completed order's fee once. Differences from account asset differences can still occur due to cash flows, other holdings, or fees not included in the trade records.
