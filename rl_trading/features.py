"""
features.py — Feature engineering with look-ahead bias prevention.

Computes technical indicators on minute data, resamples to 15-min / 1-hour
for multi-timeframe context, and merges everything into a single feature matrix.
All operations are strictly causal (no future information leaks).

Uses the ``ta`` library (Technical Analysis Library in Python) which is
compatible with Python 3.14+.
"""

import logging

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core technical indicators (minute-level)
# ---------------------------------------------------------------------------

def compute_log_returns(close: pd.Series) -> pd.Series:
    """Causal log-returns: ln(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    return RSIIndicator(close=close, window=length).rsi()


def compute_macd(close: pd.Series) -> pd.DataFrame:
    """Returns MACD line, signal line, and histogram."""
    indicator = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    return pd.DataFrame({
        "macd": indicator.macd(),
        "macd_signal": indicator.macd_signal(),
        "macd_hist": indicator.macd_diff(),
    }, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 14) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()


def _ema(close: pd.Series, span: int) -> pd.Series:
    return EMAIndicator(close=close, window=span).ema_indicator()


def compute_emas(close: pd.Series) -> pd.DataFrame:
    """EMA-20, EMA-50, EMA-200."""
    return pd.DataFrame({
        "ema_20": _ema(close, 20),
        "ema_50": _ema(close, 50),
        "ema_200": _ema(close, 200),
    }, index=close.index)


# ---------------------------------------------------------------------------
# Multi-timeframe features
# ---------------------------------------------------------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample minute OHLCV to a lower frequency using last-known bars only."""
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


def compute_multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 15-min and 1-hour EMA-50, then forward-fill onto the minute index
    to avoid look-ahead. Returns relative price position vs 1H EMA-50.

    The forward-fill uses the *completed* bar value, so any bar still forming
    is represented by its predecessor — no future data leaks in.
    """
    features = pd.DataFrame(index=df.index)

    # --- 15-minute ---
    df_15m = _resample_ohlcv(df, "15min")
    ema50_15m = _ema(df_15m["close"], 50)
    ema50_15m.name = "ema50_15m"
    # Shift by 1 bar to use only completed 15-min bars (anti look-ahead)
    ema50_15m = ema50_15m.shift(1)
    features["ema50_15m"] = ema50_15m.reindex(df.index, method="ffill")

    # --- 1-hour ---
    df_1h = _resample_ohlcv(df, "1h")
    ema50_1h = _ema(df_1h["close"], 50)
    ema50_1h.name = "ema50_1h"
    ema50_1h = ema50_1h.shift(1)
    features["ema50_1h"] = ema50_1h.reindex(df.index, method="ffill")

    # Relative position of current price to 1H EMA-50
    features["price_vs_1h_ema50"] = (df["close"] - features["ema50_1h"]) / features["ema50_1h"]

    return features


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw minute OHLCV data.

    All indicators use only past/current data — no look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to the minute index (NaN rows at the start
        are trimmed by the caller / environment).
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Log-returns
    feats["log_return"] = compute_log_returns(df["close"])

    # 2. RSI
    feats["rsi"] = compute_rsi(df["close"])

    # 3. MACD
    macd_df = compute_macd(df["close"])
    feats = feats.join(macd_df)

    # 4. ATR
    feats["atr"] = compute_atr(df["high"], df["low"], df["close"])

    # 5. EMAs (minute-level)
    ema_df = compute_emas(df["close"])
    feats = feats.join(ema_df)

    # 6. EMA distances (normalised by close)
    for col in ["ema_20", "ema_50", "ema_200"]:
        feats[f"close_vs_{col}"] = (df["close"] - feats[col]) / feats[col]

    # 7. Multi-timeframe
    mtf = compute_multitimeframe_features(df)
    feats = feats.join(mtf)

    # Drop raw EMA / MTF levels (we keep only relative distances)
    feats.drop(columns=["ema_20", "ema_50", "ema_200",
                         "ema50_15m", "ema50_1h"], inplace=True)

    # 8. Volatility features — helps agent recognize high-vol regimes
    feats["realized_vol_20"] = feats["log_return"].rolling(20).std()
    feats["realized_vol_60"] = feats["log_return"].rolling(60).std()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    feats["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    # 9. Time features — cyclic encoding of hour-of-day and day-of-week
    hour = df.index.hour + df.index.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df.index.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    logger.info("Built %d features, total rows %d", feats.shape[1], len(feats))
    return feats
