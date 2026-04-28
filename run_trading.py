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
    python run_trading.py --pair EURUSD --timesteps 200000 --train-months 12 --test-months 1

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
        "--train-months", type=int, default=12,
        help="Training window in months (default: 12)",
    )
    p.add_argument(
        "--test-months", type=int, default=1,
        help="OOS test window in months (default: 1)",
    )
    p.add_argument(
        "--max-folds", type=int, default=None,
        help="Cap number of WFO folds (default: all). Use e.g. 5 for quick diagnostics.",
    )
    p.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Training timesteps per WFO fold (default: 300000)",
    )
    p.add_argument(
        "--n-steps", type=int, default=4_096,
        help="Rollout horizon per PPO update (default: 4096)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    p.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Starting entropy coefficient (default: 0.01)",
    )
    p.add_argument(
        "--ent-coef-final", type=float, default=0.005,
        help="Final entropy coefficient after linear decay (default: 0.005)",
    )
    p.add_argument(
        "--trade-penalty", type=float, default=0.75,
        help="Per-trade penalty during training (default: 0.75)",
    )
    p.add_argument(
        "--whipsaw-window", type=int, default=30,
        help="Steps defining rapid trade reversals (default: 30)",
    )
    p.add_argument(
        "--whipsaw-penalty", type=float, default=1.25,
        help="Extra penalty for fast flip-flops (default: 1.25)",
    )
    p.add_argument(
        "--position-cost", type=float, default=0.002,
        help="Per-step cost while non-flat in training (default: 0.002)",
    )
    p.add_argument(
        "--min-hold-steps", type=int, default=5,
        help="Minimum steps to hold a position before switching (default: 5)",
    )
    p.add_argument(
        "--drawdown-penalty", type=float, default=0.5,
        help="Reward penalty multiplier for worsening drawdown (default: 0.5)",
    )
    p.add_argument(
        "--turnover-penalty", type=float, default=0.25,
        help="Extra training penalty per unit of position turnover (default: 0.25)",
    )
    p.add_argument(
        "--reward-scaling", type=float, default=100.0,
        help="Scale factor for the PnL reward term (default: 100.0)",
    )
    p.add_argument(
        "--reward-clip", type=float, default=1.0,
        help="Absolute clip applied to per-step training reward (default: 1.0)",
    )
    p.add_argument(
        "--lstm-size", type=int, default=128,
        help="LSTM hidden size (default: 128)",
    )
    p.add_argument(
        "--lstm-layers", type=int, default=1,
        help="Number of stacked LSTM layers (default: 1)",
    )
    p.add_argument(
        "--balance", type=float, default=1_000.0,
        help="Initial account balance (default: 1000)",
    )
    p.add_argument(
        "--lot-size", type=float, default=1_000.0,
        help="Trade position size in base currency units (default: 1000 = 0.01 std lot)",
    )
    p.add_argument(
        "--pip-cost", type=float, default=0.0001,
        help="Transaction cost in price units (default: 0.0001 = 1 pip)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto — uses GPU if available)",
    )
    p.add_argument(
        "--chain-balance", action="store_true",
        help="Chain account balance between folds (default: disabled)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output plots and reports",
    )
    p.add_argument(
        "--tensorboard-log", type=str, default="results/tensorboard",
        help="Directory for TensorBoard logs (default: results/tensorboard)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = Path(args.tensorboard_log)
    tensorboard_log.mkdir(parents=True, exist_ok=True)

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
            max_folds=args.max_folds,
            initial_balance=args.balance,
            lot_size=args.lot_size,
            pip_cost=args.pip_cost,
            total_timesteps=args.timesteps,
            initial_lr=args.lr,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            ent_coef_final=args.ent_coef_final,
            lstm_hidden_size=args.lstm_size,
            n_lstm_layers=args.lstm_layers,
            trade_penalty=args.trade_penalty,
            whipsaw_window=args.whipsaw_window,
            whipsaw_penalty=args.whipsaw_penalty,
            position_cost=args.position_cost,
            min_hold_steps=args.min_hold_steps,
            drawdown_penalty=args.drawdown_penalty,
            turnover_penalty=args.turnover_penalty,
            reward_scaling=args.reward_scaling,
            reward_clip=args.reward_clip,
            chain_balance=args.chain_balance,
            tensorboard_log=str(tensorboard_log),
            device=args.device,
            seed=args.seed,
        )

        if not report.folds:
            logger.warning("No completed folds for %s -- skipping", pair_name)
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
        logger.info("WALK-FORWARD OPTIMISATION -- ALL PAIRS SUMMARY")
        logger.info("=" * 70)
        logger.info("\n%s", full_table.to_string(index=False))
        logger.info("Summary saved -> %s", csv_path)
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
