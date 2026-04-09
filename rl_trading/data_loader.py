"""
data_loader.py — Robust CSV parser for HistData minute-bar forex data.

Parses <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL> format
into a clean pandas DataFrame with proper DatetimeIndex.
"""

import os
import glob
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "<TICKER>": "ticker",
    "<DTYYYYMMDD>": "date",
    "<TIME>": "time",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<VOL>": "volume",
}


def load_pair(filepath: str) -> pd.DataFrame:
    """
    Load a single forex pair CSV/TXT file and return a DatetimeIndex DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume. Index: datetime (UTC).
    """
    df = pd.read_csv(filepath)
    df.rename(columns=COLUMN_MAP, inplace=True)

    # Build datetime from DTYYYYMMDD + TIME (HHMMSS)
    df["date"] = df["date"].astype(str)
    df["time"] = df["time"].astype(str).str.zfill(6)
    df.index = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
    df.index.name = "datetime"

    df.drop(columns=["ticker", "date", "time"], inplace=True)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Remove duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]

    logger.info(
        "Loaded %s — %d rows from %s to %s",
        os.path.basename(filepath),
        len(df),
        df.index[0],
        df.index[-1],
    )
    return df


def load_all_pairs(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load every .txt file in *data_dir* and return a dict keyed by pair name.

    Parameters
    ----------
    data_dir : str
        Directory containing per-pair .txt files (e.g. EURUSD.txt).

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    pairs = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        name = os.path.splitext(os.path.basename(fp))[0].upper()
        pairs[name] = load_pair(fp)
    logger.info("Loaded %d pairs from %s", len(pairs), data_dir)
    return pairs
