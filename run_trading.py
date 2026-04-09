#!/usr/bin/env python
"""
run_trading.py — Main orchestrator for the RL trading system.

Usage
-----
    # Run on all pairs (default)
    python run_trading.py

    # Run on a single pair
    python run_trading.py --pair EURUSD

    # Customise training budget & window sizes
    python run_trading.py --pair EURUSD --timesteps 200000 --train-months 6 --test-months 1

    # Use GPU
    python run_trading.py --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from rl_trading.data_loader import load_all_pairs, load_pair
from rl_trading.metrics import compute_summary, plot_equity_curve
from rl_trading.wfo import run_wfo

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_trading")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Production-grade RL Forex Trading System"
    )
    p.add_argument(
        "--data-dir", type=str, default="forex_data_by_pair",
        help="Directory with per-pair .txt files",
    )
    p.add_argument(
        "--pair", type=str, default=None,
        help="Run on a single pair (e.g. EURUSD). Default: all pairs.",
    )
    p.add_argument(
        "--train-months", type=int, default=6,
        help="Training window in months (default: 6)",
    )
    p.add_argument(
        "--test-months", type=int, default=1,
        help="OOS test window in months (default: 1)",
    )
    p.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Training timesteps per WFO fold (default: 100000)",
    )
    p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Initial learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--lstm-size", type=int, default=128,
        help="LSTM hidden size (default: 128)",
    )
    p.add_argument(
        "--balance", type=float, default=100_000.0,
        help="Initial account balance (default: 100000)",
    )
    p.add_argument(
        "--pip-cost", type=float, default=0.0001,
        help="Transaction cost in price units (default: 0.0001 = 1 pip)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output plots and reports",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if args.pair:
        fp = Path(args.data_dir) / f"{args.pair.upper()}.txt"
        if not fp.exists():
            logger.error("File not found: %s", fp)
            sys.exit(1)
        pairs = {args.pair.upper(): load_pair(str(fp))}
    else:
        pairs = load_all_pairs(args.data_dir)

    if not pairs:
        logger.error("No data files found in %s", args.data_dir)
        sys.exit(1)

    # --- Run WFO for each pair ---
    all_summaries = []

    for pair_name, df in pairs.items():
        logger.info("=" * 70)
        logger.info("PAIR: %s  (%d rows)", pair_name, len(df))
        logger.info("=" * 70)

        report = run_wfo(
            pair_name=pair_name,
            df=df,
            train_months=args.train_months,
            test_months=args.test_months,
            initial_balance=args.balance,
            pip_cost=args.pip_cost,
            total_timesteps=args.timesteps,
            initial_lr=args.lr,
            lstm_hidden_size=args.lstm_size,
            device=args.device,
            seed=args.seed,
        )

        if not report.folds:
            logger.warning("No completed folds for %s — skipping", pair_name)
            continue

        # --- Metrics ---
        summary = compute_summary(report)
        all_summaries.append(summary)
        logger.info("\n%s", summary.to_string(index=False))

        # --- Equity curve ---
        plot_equity_curve(report, output_dir=str(output_dir))

    # --- Aggregate summary ---
    if all_summaries:
        full_table = pd.concat(all_summaries, ignore_index=True)
        csv_path = output_dir / "wfo_summary.csv"
        full_table.to_csv(csv_path, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD OPTIMISATION — ALL PAIRS SUMMARY")
        logger.info("=" * 70)
        logger.info("\n%s", full_table.to_string(index=False))
        logger.info("Summary saved → %s", csv_path)
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
