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

`uv.lock` pins the resolved dependencies. Install `uv` on the EC2 host before using the deployment scripts; the scripts sync the lockfile and run the engine through `uv`.

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
