"""
metrics.py — Performance metrics and equity-curve visualisation.

Computes: Total Return, Annualized Sharpe Ratio, Maximum Drawdown, Calmar Ratio.
Plots the RL equity curve vs Buy-and-Hold and SMA-crossover benchmarks.
"""

import logging
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from rl_trading.wfo import WFOReport

logger = logging.getLogger(__name__)

MINUTES_PER_YEAR = 252 * 24 * 60  # forex trades ~24h, 252 days/year


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def total_return(balance_curve: np.ndarray) -> float:
    """Total return as a fraction (0.10 = +10%)."""
    if len(balance_curve) < 2 or abs(balance_curve[0]) <= 1e-12:
        return 0.0
    return balance_curve[-1] / balance_curve[0] - 1.0


def annualized_sharpe(returns: np.ndarray, periods_per_year: float = MINUTES_PER_YEAR) -> float:
    """Annualized Sharpe Ratio (assumes zero risk-free rate)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(balance_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.15 = −15%)."""
    if len(balance_curve) == 0:
        return 0.0
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
    if len(balance_curve) < 2 or balance_curve[0] <= 0 or balance_curve[-1] <= 0:
        return 0.0
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
    returns = report.oos_returns
    actions = report.oos_actions

    fold_returns = []
    for f in report.folds:
        if len(f.balance_curve) >= 2 and f.balance_curve[0] > 0:
            fold_returns.append(float(f.balance_curve[-1] / f.balance_curve[0] - 1.0))

    if fold_returns:
        compounded_total_return = float(np.prod(1.0 + np.array(fold_returns)) - 1.0)
        fold_equity = np.concatenate([[1.0], np.cumprod(1.0 + np.array(fold_returns))])
    else:
        compounded_total_return = 0.0
        fold_equity = np.array([1.0])

    # Position distribution
    action_counts = Counter(actions)
    total_actions = max(len(actions), 1)
    # Trade count = number of position changes
    trades = int(np.sum(np.diff(actions) != 0)) if len(actions) > 1 else 0

    metrics = {
        "Pair": report.pair,
        "Total Return (%)": round(compounded_total_return * 100, 2),
        "Ann. Sharpe Ratio": round(annualized_sharpe(returns), 4),
        "Max Drawdown (%)": round(max_drawdown(fold_equity) * 100, 2),
        "Calmar Ratio": round(calmar_ratio(fold_equity, returns), 4),
        "OOS Bars": len(returns),
        "Folds": len(report.folds),
        "Trades": trades,
        "Sell %": round(100 * action_counts.get(0, 0) / total_actions, 1),
        "Flat %": round(100 * action_counts.get(1, 0) / total_actions, 1),
        "Buy %": round(100 * action_counts.get(2, 0) / total_actions, 1),
    }
    return pd.DataFrame([metrics])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _sma_crossover_benchmark(
    prices: np.ndarray, fast: int = 50, slow: int = 200
) -> np.ndarray:
    """
    Simple SMA crossover benchmark: long when fast SMA > slow SMA, else flat.
    Returns a balance curve with the same initial value as the RL strategy.
    """
    s = pd.Series(prices)
    sma_f = s.rolling(fast, min_periods=fast).mean()
    sma_s = s.rolling(slow, min_periods=slow).mean()
    # Position: +1 when fast > slow, else 0
    pos = (sma_f > sma_s).astype(float).fillna(0).values
    ret = np.diff(prices) / prices[:-1]
    strat_ret = pos[:-1] * ret
    balance = np.ones(len(prices))
    balance[1:] = np.cumprod(1 + strat_ret)
    return balance


def plot_equity_curve(
    report: WFOReport,
    output_dir: str = "results",
    show: bool = False,
) -> str:
    """
    Plot RL strategy equity vs Buy-and-Hold and SMA crossover benchmarks.

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

    prices = report.oos_prices
    timestamps = report.oos_timestamps

    # Build chained RL balance curve from fold data
    rl_balance_parts = []
    for f in report.folds:
        # Skip the initial_balance entry (index 0) for all folds after the first
        if rl_balance_parts:
            rl_balance_parts.append(f.balance_curve[1:])
        else:
            rl_balance_parts.append(f.balance_curve)
    rl_balance = np.concatenate(rl_balance_parts) if rl_balance_parts else np.array([100_000.0])

    initial = rl_balance[0]

    # Buy-and-hold: invest initial_balance at first OOS price
    bnh = initial * prices / prices[0]

    # SMA crossover benchmark
    sma_curve = initial * _sma_crossover_benchmark(prices)

    # Align lengths
    n = min(len(timestamps), len(rl_balance) - 1, len(bnh))
    ts = pd.to_datetime(timestamps[:n])
    rl_curve = rl_balance[1: n + 1]
    bnh_curve = bnh[:n]
    sma_curve = sma_curve[:n] if len(sma_curve) >= n else np.pad(
        sma_curve, (0, n - len(sma_curve)), constant_values=sma_curve[-1]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # --- Equity curve ---
    ax = axes[0]
    ax.plot(ts, rl_curve, label="RL Strategy", linewidth=1.2, color="#2196F3")
    ax.plot(ts, bnh_curve, label="Buy & Hold", linewidth=1.0, alpha=0.7, color="#FF9800")
    ax.plot(ts, sma_curve, label="SMA 50/200 Crossover", linewidth=1.0, alpha=0.7,
            color="#4CAF50", linestyle="--")
    ax.set_title(f"{report.pair} — OOS Equity Curve (Walk-Forward)", fontsize=14)
    ax.set_ylabel("Account Balance")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Drawdown subplot ---
    ax2 = axes[1]
    peak = np.maximum.accumulate(rl_curve)
    dd = (peak - rl_curve) / peak * 100
    ax2.fill_between(ts, dd, 0, color="#F44336", alpha=0.4, label="Drawdown %")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()

    filepath = str(Path(output_dir) / f"{report.pair}_equity_curve.png")
    fig.savefig(filepath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    logger.info("Saved equity curve -> %s", filepath)
    return filepath
