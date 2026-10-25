import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_features(df, features):
    """Prepare Ridge features and next-minute return targets."""

    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    if df["time"].duplicated().any():
        raise ValueError("Duplicate candle timestamps")

    numeric_columns = [
        "close",
        "short_sma",
        "long_sma",
        "RSI",
        "MACD",
        "MACD Histogram",
        "market_trend",
        "pct_diff",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Technical indicators
    df["sma_diff"] = (df["short_sma"] - df["long_sma"]) / df["close"]

    # Next-minute return target
    next_minute = df["time"].shift(-1) - df["time"] == pd.Timedelta(minutes=1)

    df["target"] = (df["close"].shift(-1) / df["close"] - 1).where(next_minute)

    # Replace invalid values
    df[features + ["target"]] = df[features + ["target"]].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


def train_ridge(
    training_df,
    features,
    long_window,
    alpha=1.0,
):
    """Train Ridge on the latest N valid observations."""

    train = training_df.dropna(subset=features + ["target"]).tail(long_window)

    if len(train) < long_window:
        return None

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(
        train[features],
        train["target"],
    )

    return model


def predict_ridge(
    previous_output,
    current_row,
    long_window,
    features,
    alpha=1.0,
):
    """Retrain on previous observations and predict next-minute return."""

    df = pd.concat(
        [
            pd.DataFrame(previous_output).tail(long_window + 1),
            pd.DataFrame([current_row]),
        ],
        ignore_index=True,
    )

    df = make_features(df, features)

    # Latest row is for prediction
    latest = df[features].iloc[[-1]]

    # Earlier rows have known next-minute targets
    training_df = df.iloc[:-1]

    model = train_ridge(
        training_df=training_df,
        features=features,
        long_window=long_window,
        alpha=alpha,
    )

    if model is None or latest.isna().any().any():
        return np.nan

    return float(model.predict(latest)[0])
