"""
metrics.py — Performance metrics and equity-curve visualisation.

Computes: Total Return, Annualized Sharpe Ratio, Maximum Drawdown, Calmar Ratio.
Plots the RL equity curve vs Buy-and-Hold benchmark using matplotlib.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from rl_trading.wfo import WFOReport

logger = logging.getLogger(__name__)

MINUTES_PER_YEAR = 252 * 6.5 * 60  # approx. trading minutes in a year


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def total_return(balance_curve: np.ndarray) -> float:
    """Total return as a fraction (0.10 = +10%)."""
    return balance_curve[-1] / balance_curve[0] - 1.0


def annualized_sharpe(returns: np.ndarray, periods_per_year: float = MINUTES_PER_YEAR) -> float:
    """Annualized Sharpe Ratio (assumes zero risk-free rate)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(balance_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.15 = −15%)."""
    peak = np.maximum.accumulate(balance_curve)
    dd = (peak - balance_curve) / peak
    return float(np.max(dd))


def calmar_ratio(balance_curve: np.ndarray, returns: np.ndarray,
                 periods_per_year: float = MINUTES_PER_YEAR) -> float:
    """Calmar Ratio = annualized return / max drawdown."""
    mdd = max_drawdown(balance_curve)
    if mdd == 0:
        return 0.0
    n = len(returns)
    ann_return = (balance_curve[-1] / balance_curve[0]) ** (periods_per_year / max(n, 1)) - 1
    return float(ann_return / mdd)


def compute_summary(report: WFOReport) -> pd.DataFrame:
    """
    Compute a summary table for the aggregated OOS results.

    Returns
    -------
    pd.DataFrame
        Single-row table with key metrics.
    """
    balance = report.oos_balance
    returns = report.oos_returns

    metrics = {
        "Pair": report.pair,
        "Total Return (%)": round(total_return(balance) * 100, 2),
        "Ann. Sharpe Ratio": round(annualized_sharpe(returns), 4),
        "Max Drawdown (%)": round(max_drawdown(balance) * 100, 2),
        "Calmar Ratio": round(calmar_ratio(balance, returns), 4),
        "OOS Bars": len(returns),
        "Folds": len(report.folds),
    }
    return pd.DataFrame([metrics])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_equity_curve(
    report: WFOReport,
    output_dir: str = "results",
    show: bool = False,
) -> str:
    """
    Plot RL strategy equity vs Buy-and-Hold benchmark and save to disk.

    Parameters
    ----------
    report : WFOReport
    output_dir : str
        Directory to save the PNG.
    show : bool
        If True, call plt.show() (useful in notebooks).

    Returns
    -------
    str
        Path to the saved figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    balance = report.oos_balance
    prices = report.oos_prices
    timestamps = report.oos_timestamps

    # Buy-and-hold: invest initial_balance at first OOS price
    initial = balance[0]
    bnh = initial * prices / prices[0]

    # Align lengths (balance has +1 element from the initial value)
    n = min(len(timestamps), len(balance) - 1, len(bnh))
    ts = pd.to_datetime(timestamps[:n])
    rl_curve = balance[1: n + 1]
    bnh_curve = bnh[:n]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, rl_curve, label="RL Strategy", linewidth=1.0)
    ax.plot(ts, bnh_curve, label="Buy & Hold", linewidth=1.0, alpha=0.7)
    ax.set_title(f"{report.pair} — OOS Equity Curve (Walk-Forward)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Balance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    filepath = str(Path(output_dir) / f"{report.pair}_equity_curve.png")
    fig.savefig(filepath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    logger.info("Saved equity curve → %s", filepath)
    return filepath
