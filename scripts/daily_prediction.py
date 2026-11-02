import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from moomoo import RET_OK, AuType, Session, SubType
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .broker import (
    close_contexts,
    create_contexts,
    get_live_market_snapshot,
)
from .runtime import create_runtime
from .strategy import LONG_WINDOW, MACD_SIGNAL, RSI_PERIOD, SHORT_WINDOW

FEATURES = [
    "RSI",
    "MACD",
    "MACD Histogram",
    "sma_diff",
    "pct_diff",
]
HISTORY_YEARS = 3
MAX_HISTORY_ROWS = 1000
TRAINING_WINDOW = 252
MINIMUM_TRAINING_ROWS = 60
VALIDATION_FRACTION = 0.2
FORECAST_HORIZON = 5
TRADING_SESSIONS_PER_YEAR = 252
TARGET_COLUMNS = [f"target_day_{day}" for day in range(1, FORECAST_HORIZON + 1)]
MODEL_GRID = [
    ("ridge", alpha, None)
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
] + [
    ("lasso", alpha, None)
    for alpha in (0.00001, 0.0001, 0.001, 0.01, 0.1)
] + [
    ("elasticnet", alpha, l1_ratio)
    for alpha in (0.00001, 0.0001, 0.001, 0.01, 0.1)
    for l1_ratio in (0.1, 0.5, 0.9)
]


def make_daily_features(daily_df):
    """Build features at t and each daily return from t+1 close onward."""

    df = daily_df.copy()
    df["time_key"] = pd.to_datetime(df["time_key"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("time_key").reset_index(drop=True)

    if df["time_key"].duplicated().any():
        raise ValueError("Duplicate daily candle timestamps")
    if not np.isfinite(df["close"]).all() or (df["close"] <= 0).any():
        raise ValueError("Daily closing prices must be positive and finite")

    close = df["close"].to_numpy(dtype=float)
    df["short_sma"] = talib.SMA(close, timeperiod=SHORT_WINDOW)
    df["long_sma"] = talib.SMA(close, timeperiod=LONG_WINDOW)
    rsi_values = np.full(len(close), np.nan)
    macd_values = np.full(len(close), np.nan)
    macd_histograms = np.full(len(close), np.nan)
    macd_window = LONG_WINDOW + MACD_SIGNAL + 1
    for index in range(len(close)):
        if index + 1 >= RSI_PERIOD + 1:
            rsi_values[index] = talib.RSI(
                close[index - RSI_PERIOD : index + 1],
                timeperiod=RSI_PERIOD,
            )[-1]
        if index + 1 >= macd_window:
            macd, _, histogram = talib.MACD(
                close[index - macd_window + 1 : index + 1],
                fastperiod=SHORT_WINDOW,
                slowperiod=LONG_WINDOW,
                signalperiod=MACD_SIGNAL,
            )
            macd_values[index] = macd[-1]
            macd_histograms[index] = histogram[-1]
    df["RSI"] = rsi_values
    df["MACD"] = macd_values
    df["MACD Histogram"] = macd_histograms
    df["sma_diff"] = (df["short_sma"] - df["long_sma"]) / df["close"]
    df["pct_diff"] = df["close"].pct_change(fill_method=None) * 100
    # Signal at t, entry at t+1 close, then one target for each return day.
    for day in range(1, FORECAST_HORIZON + 1):
        df[f"target_day_{day}"] = (
            df["close"].shift(-(day + 1)) / df["close"].shift(-day) - 1
        )

    df[FEATURES + TARGET_COLUMNS] = df[FEATURES + TARGET_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def make_model(family, alpha, l1_ratio=None):
    """Build a scaled linear model for one model-grid candidate."""

    if family == "ridge":
        estimator = Ridge(alpha=alpha)
    elif family == "lasso":
        estimator = Lasso(alpha=alpha, max_iter=20_000)
    elif family == "elasticnet":
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return make_pipeline(StandardScaler(), estimator)


def build_pooled_panel(daily_histories):
    """Build a date-aligned panel with lagged cross-stock returns and ticker IDs."""

    features_by_symbol = {
        symbol: make_daily_features(history) for symbol, history in daily_histories.items()
    }
    returns_wide = pd.concat(
        {
            symbol: frame.set_index("time_key")["close"].pct_change(fill_method=None)
            for symbol, frame in features_by_symbol.items()
        },
        axis=1,
    ).sort_index()
    # Shift the union calendar so no prediction uses another market's same-date close.
    cross_returns = returns_wide.ffill().shift(1).fillna(0.0)
    cross_feature_names = {
        symbol: f"lagged_return_{symbol}" for symbol in daily_histories
    }
    cross_returns = cross_returns.rename(columns=cross_feature_names)
    ticker_feature_names = {
        symbol: f"ticker_{symbol}" for symbol in daily_histories
    }

    rows = []
    for symbol, frame in features_by_symbol.items():
        frame = frame.copy()
        frame["symbol"] = symbol
        aligned_cross_returns = cross_returns.reindex(frame["time_key"]).reset_index(
            drop=True
        )
        for column in aligned_cross_returns:
            frame[column] = aligned_cross_returns[column].to_numpy()
        for ticker, column in ticker_feature_names.items():
            frame[column] = float(ticker == symbol)
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True).sort_values(
        ["time_key", "symbol"]
    ).reset_index(drop=True)
    model_features = FEATURES + list(cross_feature_names.values()) + list(
        ticker_feature_names.values()
    )
    return panel, model_features


def _training_rows_before(panel, date, feature_cols):
    """Select up to 252 prior dates with all five labels known by ``date``."""

    dates = pd.DatetimeIndex(panel["time_key"].drop_duplicates().sort_values())
    date_index = dates.get_loc(date)
    cutoff_index = date_index - FORECAST_HORIZON
    if cutoff_index <= 0:
        return panel.iloc[0:0]
    cutoff_date = dates[cutoff_index]
    eligible = panel.loc[panel["time_key"] < cutoff_date].dropna(
        subset=feature_cols + TARGET_COLUMNS
    )
    training_dates = pd.DatetimeIndex(
        eligible["time_key"].drop_duplicates().sort_values()
    )[-TRAINING_WINDOW:]
    return eligible.loc[eligible["time_key"].isin(training_dates)]


def score_validation_candidate(panel, feature_cols, validation_dates, candidate):
    """Walk forward a pooled candidate and score its equal-weight long portfolio."""

    family, alpha, l1_ratio = candidate
    portfolio_returns = []
    for date in validation_dates:
        eval_rows = panel.loc[
            (panel["time_key"] == date)
            & panel[FEATURES + TARGET_COLUMNS].notna().all(axis=1)
        ]
        training = _training_rows_before(panel, date, feature_cols)
        if training["time_key"].nunique() < MINIMUM_TRAINING_ROWS:
            return np.nan, 0
        model = make_model(family, alpha, l1_ratio)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(training[feature_cols], training[TARGET_COLUMNS])
        daily_predictions = model.predict(eval_rows[feature_cols])
        daily_actuals = eval_rows[TARGET_COLUMNS].to_numpy(dtype=float)
        predicted_totals = np.prod(1 + daily_predictions, axis=1) - 1
        actual_totals = np.prod(1 + daily_actuals, axis=1) - 1
        predicted_total = dict(zip(eval_rows["symbol"], predicted_totals))
        actual_total = dict(zip(eval_rows["symbol"], actual_totals))
        selected = [symbol for symbol, value in predicted_total.items() if value > 0]
        portfolio_returns.append(
            float(np.mean([actual_total[symbol] for symbol in selected]))
            if selected
            else 0.0
        )

    if len(portfolio_returns) < 2:
        return np.nan, len(portfolio_returns)
    daily_std = np.asarray(portfolio_returns).std(ddof=1)
    if daily_std <= 0:
        return np.nan, len(portfolio_returns)
    annualization = np.sqrt(TRADING_SESSIONS_PER_YEAR / FORECAST_HORIZON)
    return (
        float(np.mean(portfolio_returns) / daily_std * annualization),
        len(portfolio_returns),
    )


def predict_pooled_returns(daily_histories):
    """Select one pooled model, then forecast five returns for every stock."""

    panel, feature_cols = build_pooled_panel(daily_histories)
    validation_candidates = panel.dropna(
        subset=feature_cols + TARGET_COLUMNS
    )
    candidate_dates = pd.DatetimeIndex(
        validation_candidates["time_key"].drop_duplicates().sort_values()
    )[-TRAINING_WINDOW:]
    validation_start = max(
        MINIMUM_TRAINING_ROWS + FORECAST_HORIZON,
        len(candidate_dates) - int(np.ceil(len(candidate_dates) * VALIDATION_FRACTION)),
    )
    validation_dates = candidate_dates[validation_start::FORECAST_HORIZON]
    grid_rows = []
    for candidate in MODEL_GRID:
        score, observations = score_validation_candidate(
            panel, feature_cols, validation_dates, candidate
        )
        family, alpha, l1_ratio = candidate
        grid_rows.append(
            {
                "model": family,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "validation_sharpe": score,
                "validation_periods": observations,
            }
        )
    viable = [row for row in grid_rows if np.isfinite(row["validation_sharpe"])]
    if not viable:
        raise ValueError("Validation grid search produced no finite Sharpe scores")
    best = max(viable, key=lambda row: row["validation_sharpe"])

    latest_rows = {}
    for symbol in daily_histories:
        symbol_rows = panel.loc[panel["symbol"] == symbol].dropna(subset=feature_cols)
        if symbol_rows.empty:
            raise ValueError(f"Could not calculate latest features for {symbol}")
        latest_rows[symbol] = symbol_rows.iloc[[-1]]

    training = panel.dropna(subset=feature_cols + TARGET_COLUMNS)
    training_dates = pd.DatetimeIndex(
        training["time_key"].drop_duplicates().sort_values()
    )[-TRAINING_WINDOW:]
    training = training.loc[training["time_key"].isin(training_dates)]
    if training["time_key"].nunique() < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "Not enough pooled history: "
            f"{training['time_key'].nunique()}/{MINIMUM_TRAINING_ROWS} dates"
        )
    model = make_model(best["model"], best["alpha"], best["l1_ratio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(training[feature_cols], training[TARGET_COLUMNS])

    symbol_order = list(latest_rows)
    latest_panel = pd.concat(
        [latest_rows[symbol] for symbol in symbol_order], ignore_index=True
    )
    predictions = model.predict(latest_panel[feature_cols])
    predictions_by_symbol = {
        symbol: {
            day: float(predictions[row_index, day_index])
            for day_index, day in enumerate(range(1, FORECAST_HORIZON + 1))
        }
        for row_index, symbol in enumerate(symbol_order)
    }
    training_counts = {
        day: int(training["time_key"].nunique())
        for day in range(1, FORECAST_HORIZON + 1)
    }

    return (
        predictions_by_symbol,
        training_counts,
        float(best["validation_sharpe"]),
        int(best["validation_periods"]),
        {"model": best["model"], "alpha": best["alpha"], "l1_ratio": best["l1_ratio"]},
        grid_rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Forecast five daily returns after next-session execution "
            "for open Moomoo positions."
        )
    )
    parser.add_argument(
        "--env",
        choices=["real", "simulate"],
        default="simulate",
        help="Moomoo trading environment (default: simulate).",
    )
    return parser.parse_args()


def get_open_positions(trade_ctx, trade_env):
    """Return positive-quantity positions with a normalized code-name column."""

    ret, positions = trade_ctx.position_list_query(trd_env=trade_env)
    if ret != RET_OK:
        raise RuntimeError(f"Error fetching positions: {positions}")

    positions = positions.loc[positions["qty"] > 0].copy()
    if "stock_name" in positions.columns:
        positions = positions[["code", "qty", "stock_name"]].rename(
            columns={"stock_name": "code_name"}
        )
    else:
        positions = positions[["code", "qty"]].copy()
        positions["code_name"] = ""
    return positions


def get_market_now(symbol):
    """Return the current timestamp in the symbol's exchange timezone."""

    market_tz = "Asia/Hong_Kong" if symbol.startswith("HK.") else "America/New_York"
    return pd.Timestamp.now(tz=market_tz)


def fetch_live_price(quote_ctx, symbol, code_name=""):
    """Return live price, best available instrument name, and any skip status."""

    snapshot = get_live_market_snapshot(quote_ctx, [symbol])
    if snapshot is None:
        return None, code_name, "quote_unavailable"

    if not code_name:
        name_column = next(
            (column for column in ("name", "stock_name") if column in snapshot),
            None,
        )
        if name_column:
            names = snapshot.loc[snapshot["code"] == symbol, name_column].dropna()
            if not names.empty:
                code_name = str(names.iloc[0])

    quote = snapshot.loc[snapshot["code"] == symbol]
    if quote.empty:
        return None, code_name, "quote_missing"

    current_price = float(
        pd.to_numeric(quote["last_price"], errors="coerce").iloc[0]
    )
    if not np.isfinite(current_price) or current_price <= 0:
        return current_price, code_name, "invalid_price"
    return current_price, code_name, None


def fetch_completed_history(quote_ctx, symbol, market_now):
    """Fetch daily candles and keep completed sessions only."""

    start = (market_now - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
    end = market_now.strftime("%Y-%m-%d")
    ret, daily_df, _ = quote_ctx.request_history_kline(
        symbol,
        start,
        end,
        ktype=SubType.K_DAY,
        autype=AuType.NONE,
        max_count=MAX_HISTORY_ROWS,
        session=Session.ALL,
    )
    if ret != RET_OK:
        return None, "history_unavailable", str(daily_df)

    daily_df["time_key"] = pd.to_datetime(daily_df["time_key"])
    completed_history = daily_df.loc[
        daily_df["time_key"].dt.date < market_now.date(),
        ["time_key", "close"],
    ].copy()
    if completed_history.empty:
        return None, "history_empty", "No completed daily candles returned"
    return completed_history, None, None


def fetch_forecast_dates(quote_ctx, symbol, feature_date):
    """Return the execution date and five subsequent trading-session dates."""

    start = (pd.Timestamp(feature_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(feature_date) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    ret, calendar = quote_ctx.request_trading_days(
        code=symbol,
        start=start,
        end=end,
    )
    if ret != RET_OK:
        return None, str(calendar)
    dates = [pd.Timestamp(day["time"]).date().isoformat() for day in calendar]
    if len(dates) < FORECAST_HORIZON + 1:
        return None, "Not enough future trading dates were returned"
    return dates[: FORECAST_HORIZON + 1], None


def prepare_position(position, quote_ctx, environment, run_time):
    """Fetch one position's quote and history before pooled forecasting."""

    symbol = position.code
    code_name = str(position.code_name) if pd.notna(position.code_name) else ""
    quantity = int(position.qty)
    result = {
        "run_time": run_time,
        "environment": environment,
        "symbol": symbol,
        "code_name": code_name,
        "quantity": quantity,
    }

    current_price, code_name, quote_status = fetch_live_price(
        quote_ctx, symbol, code_name
    )
    result["code_name"] = code_name
    if quote_status:
        result["current_price"] = current_price
        result["status"] = quote_status
        print(f"{symbol}: {quote_status.replace('_', ' ')}; skipping.")
        return result, None

    result["current_price"] = current_price
    market_now = get_market_now(symbol)
    history, history_status, message = fetch_completed_history(
        quote_ctx, symbol, market_now
    )
    if history_status:
        result["status"] = history_status
        result["message"] = message
        print(f"{symbol}: {message}; skipping.")
        return result, None

    feature_date = history["time_key"].iloc[-1].date().isoformat()
    result["feature_date"] = feature_date
    forecast_dates, calendar_error = fetch_forecast_dates(
        quote_ctx, symbol, feature_date
    )
    if calendar_error:
        result["status"] = "calendar_unavailable"
        result["message"] = calendar_error
        print(f"{symbol}: could not get forecast dates ({calendar_error}); skipping.")
        return result, None

    return result, {
        "run_time": run_time,
        "environment": environment,
        "symbol": symbol,
        "code_name": code_name,
        "quantity": quantity,
        "current_price": current_price,
        "currency": "HKD" if symbol.startswith("HK.") else "USD",
        "feature_date": feature_date,
        "forecast_dates": forecast_dates,
        "history": history,
    }


def finish_pooled_prediction(
    prepared, daily_returns, training_counts, validation_sharpe,
    validation_rows, selected_model,
):
    """Add pooled model forecasts and print one position's results."""

    symbol = prepared["symbol"]
    code_name = prepared["code_name"]
    quantity = prepared["quantity"]
    current_price = prepared["current_price"]
    currency = prepared["currency"]
    feature_date = prepared["feature_date"]
    forecast_dates = prepared["forecast_dates"]
    result = {
        "run_time": prepared["run_time"],
        "environment": prepared["environment"],
        "symbol": symbol,
        "code_name": code_name,
        "quantity": quantity,
        "current_price": current_price,
        "feature_date": feature_date,
    }

    total_return = float(np.prod(1 + np.asarray(list(daily_returns.values()))) - 1)
    estimated_change = quantity * current_price * total_return
    result.update(
        {
            "currency": currency,
            "execution_date": forecast_dates[0],
            "predicted_5d_return": total_return,
            "training_observations": min(training_counts.values()),
            "pooled_validation_sharpe": validation_sharpe,
            "validation_observations": validation_rows,
            "selected_model": selected_model["model"],
            "selected_alpha": selected_model["alpha"],
            "selected_l1_ratio": selected_model["l1_ratio"],
            "estimated_position_value_change": estimated_change,
            "status": "predicted",
        }
    )
    for day, predicted_return in daily_returns.items():
        return_date = forecast_dates[day]
        result[f"predicted_return_ending_{return_date}"] = predicted_return

    display_name = f" ({code_name})" if code_name else ""
    print(f"\nPosition: {symbol}{display_name} x {quantity}")
    print(f"Features through completed session: {feature_date}")
    print(f"Execution assumed at close: {forecast_dates[0]}")
    print(f"Current price: {current_price:.4f} {currency}")
    print(
        f"Selected model: {selected_model['model']} "
        f"(alpha={selected_model['alpha']:g}, "
        f"l1_ratio={selected_model['l1_ratio']})"
    )
    for day, predicted_return in daily_returns.items():
        print(
            f"Predicted return ending {forecast_dates[day]}: "
            f"{predicted_return:+.4%} "
            f"({selected_model['model']}, "
            f"{training_counts[day]} training observations)"
        )
    print(f"Predicted cumulative 5-session return: {total_return:+.4%}")
    if np.isfinite(validation_sharpe):
        print(
            f"Pooled walk-forward validation Sharpe: {validation_sharpe:.3f} "
            f"({validation_rows} non-overlapping periods; long when predicted return > 0)"
        )
    else:
        print("Walk-forward validation Sharpe: unavailable")
    print(f"Estimated position value change: {estimated_change:+,.2f} {currency}")
    return result


def write_prediction_log(results, grid_results):
    """Write prediction output and model-grid validation scores to timestamped CSVs."""

    if not results:
        return
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = logs_dir / f"daily_predictions_{timestamp}.csv"
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"\nPrediction log saved to {log_path}")
    if grid_results:
        grid_path = logs_dir / f"daily_prediction_grid_search_{timestamp}.csv"
        pd.DataFrame(grid_results).to_csv(grid_path, index=False)
        print(f"Validation grid scores saved to {grid_path}")


def main():
    args = parse_args()
    runtime_settings = create_runtime(
        argparse.Namespace(
            env=args.env,
            date=pd.Timestamp.now(),
            live=False,
        )
    )

    quote_ctx = None
    trade_ctx = None
    results = []
    grid_results = []
    run_time = pd.Timestamp.now().tz_localize(None).isoformat(timespec="seconds")
    try:
        quote_ctx, trade_ctx, _ = create_contexts(runtime_settings)
        positions = get_open_positions(
            trade_ctx,
            runtime_settings.trade_env,
        )
        if positions.empty:
            print(f"No open positions in the {args.env} account.")
            results.append(
                {
                    "run_time": run_time,
                    "environment": args.env,
                    "symbol": "",
                    "code_name": "",
                    "quantity": 0,
                    "status": "no_open_positions",
                }
            )
            return

        prepared_positions = []
        for position in positions.itertuples(index=False):
            result, prepared = prepare_position(
                position, quote_ctx, args.env, run_time
            )
            if prepared is None:
                results.append(result)
            else:
                prepared["base_result"] = result
                prepared_positions.append(prepared)

        if prepared_positions:
            histories = {
                prepared["symbol"]: prepared["history"]
                for prepared in prepared_positions
            }
            try:
                (
                    predictions_by_symbol,
                    training_counts,
                    validation_sharpe,
                    validation_rows,
                    selected_model,
                    candidate_scores,
                ) = predict_pooled_returns(histories)
            except ValueError as error:
                for prepared in prepared_positions:
                    result = prepared["base_result"]
                    result["status"] = "prediction_unavailable"
                    result["message"] = str(error)
                    print(
                        f"{prepared['symbol']}: pooled prediction unavailable "
                        f"({error}); skipping."
                    )
                    results.append(result)
            else:
                for prepared in prepared_positions:
                    results.append(
                        finish_pooled_prediction(
                            prepared,
                            predictions_by_symbol[prepared["symbol"]],
                            training_counts,
                            validation_sharpe,
                            validation_rows,
                            selected_model,
                        )
                    )
                universe = ",".join(sorted(histories))
                grid_results.extend(
                    {
                        "run_time": run_time,
                        "environment": args.env,
                        "universe": universe,
                        "selected": (
                            score["model"] == selected_model["model"]
                            and score["alpha"] == selected_model["alpha"]
                            and score["l1_ratio"] == selected_model["l1_ratio"]
                        ),
                        **score,
                    }
                    for score in candidate_scores
                )
    finally:
        close_contexts(quote_ctx, trade_ctx)
        write_prediction_log(results, grid_results)


if __name__ == "__main__":
    main()
