# Automated Trading Analytics & Portfolio Monitoring System

A Python-based portfolio analytics engine that integrates with broker APIs to compute real-time and historical portfolio performance, including realized and unrealized P/L. The system supports both live trading and backtesting with a unified portfolio state model.

---
## ⚙️ How to Run

```bash
git clone https://github.com/waiyeonglee/portfolio-analytics-engine.git
cd portfolio-analytics-engine

uv sync
```

Backtesting mode:
```bash
uv run python -m scripts.main
```

Live mode:
```bash
uv run python -m scripts.main --live
```

For notebook work, include the development dependencies:
```bash
uv sync --group dev
uv run --group dev jupyter lab Moomoo.ipynb
```

`uv.lock` pins the resolved dependencies.

## Deploying to EC2

When EC2 boots, systemd can pull the latest `main` commit from GitHub and sync its locked dependencies before starting either trading service. This works even when the instance was offline during a push. To deploy immediately while EC2 is running, start the **Deploy portfolio engine to EC2** workflow manually from GitHub Actions; it pulls the latest commit and restarts both services.

The directories have separate roles: `jobs/` contains the checked-in scripts that update, start, and stop the application. `service_files/` contains local EC2-specific systemd unit files and is intentionally git-ignored. Keep credentials and machine-specific service configuration there; do not commit it. Copy the required unit files to EC2 separately.

For manual deployments, configure these GitHub Actions repository secrets:

- `EC2_HOST`: EC2's public DNS name or IP address
- `EC2_SSH_KEY`: private SSH key for the `ubuntu` account
- `EC2_KNOWN_HOSTS`: verified SSH host-key entry for the EC2 host

One-time EC2 setup:

1. Clone this repository to `/home/ubuntu/portfolio_analytics_engine` and install `uv` at `/home/ubuntu/.local/bin/uv`.
2. If the repository is private, configure a read-only GitHub deploy key for the `ubuntu` account so `git pull` can access it.
3. Securely copy the local unit files from `service_files/` to `/etc/systemd/system/` on EC2. These local-only files are not included in GitHub. Check that the real and simulated units use the checked-in scripts:
   ```ini
   # moomoo_real.service
   WorkingDirectory=/home/ubuntu/portfolio_analytics_engine
   ExecStart=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/start_moomoo_real.sh
   ExecStopPost=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/end_moomoo.sh real

   # moomoo_simulate.service
   WorkingDirectory=/home/ubuntu/portfolio_analytics_engine
   ExecStart=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/start_moomoo_simulate.sh
   ExecStopPost=/bin/bash /home/ubuntu/portfolio_analytics_engine/jobs/end_moomoo.sh simulate
   ```
   The unit files should require and order after `opend.service` and `moomoo_update.service`. The updater should use `/home/ubuntu/portfolio_analytics_engine` as its working directory and run `jobs/moomoo_update.sh`.
4. Reload systemd and enable the updater and both trading services:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable moomoo_update moomoo_real moomoo_simulate
   ```

The updater is a shared oneshot dependency, so GitHub is pulled and `uv sync --locked` runs once before the trading services start. If the update fails, systemd will not start them. The manual GitHub Actions workflow also requires the `ubuntu` account to be able to run `sudo systemctl restart` for both services without an interactive password.

## 🚀 Key Features

- Real-time portfolio tracking via broker API integration  
- Unified computation of realized and unrealized P/L  
- Session-aware market data handling (full-day vs partial-day logic)  
- Live trading and backtesting using the same core analytics engine  
- Persistent daily logging of portfolio state for auditing and debugging  
- Robust handling of duplicate and fragmented position records from broker APIs  

---

## 🧠 Core Design

This project is built around a **portfolio state engine** that reconstructs and maintains accurate account performance over time.

It ensures:

- Consistent P/L computation across time  
- Clean separation between open positions and closed trades  
- Reliable handling of incomplete or inconsistent broker data  
- Reproducibility between live trading and backtest environments  

---

## 🏗 System Architecture

**Data Flow:**

Broker API → Data Ingestion → Position Normalization → P/L Engine → Daily Snapshot Logs

**Main Components:**

- **Data Layer**: Fetches account + position data from broker API  
- **Normalization Layer**: Cleans duplicate / fragmented position rows  
- **Analytics Engine**: Computes realized & unrealized P/L  
- **Session Manager**: Ensures correct trading-day boundaries  
- **Logging Layer**: Stores daily portfolio snapshots  

---

## 📊 Outputs

The engine generates:

- Total account value  
- Position-level breakdown  
- Realized P/L (daily + cumulative)  
- Unrealized P/L  
- Exposure tracking  
- Daily portfolio snapshots for historical analysis  

---

## 🔁 Live vs Backtest Mode

- **Live Mode**: Processes real-time broker data continuously  
- **Backtest Mode**: Reconstructs portfolio state from historical sessions  

Both modes share the same computation logic to ensure consistency.

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- Broker API (Moomoo / Futu OpenAPI)  
- NumPy (light usage for calculations)  
- Local file-based logging system (CSV snapshots)  

---

## 📌 Key Engineering Challenges Solved

- Handling inconsistent broker position structures  
- Preventing double counting of P/L from duplicated rows  
- Correctly segmenting trading sessions across days  
- Aligning live trading logic with historical backtesting  
- Building a reliable portfolio state reconstruction system  

---

## 📈 Future Improvements

- Strategy module (signal-based trading logic)  
- Performance metrics (Sharpe ratio, drawdown, win rate)  
- Visualization dashboard for portfolio analytics  
- Multi-asset portfolio support  
- Database-backed storage instead of CSV logs  

---

## 📄 License

MIT License
