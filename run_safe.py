"""Wrapper that runs the trading pipeline with all output captured to a log file."""
import sys
import os
import traceback

LOG_PATH = os.path.join("results", "run_output.log")
os.makedirs("results", exist_ok=True)

# Open log file FIRST, before any imports that could fail
log_file = open(LOG_PATH, "w", encoding="utf-8")

class Tee:
    """Write to both console and log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(original_stdout, log_file)
sys.stderr = Tee(original_stderr, log_file)

try:
    print("=== WRAPPER START ===", flush=True)
    
    import logging
    # Force all loggers to write to our tee'd stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    
    from rl_trading.data_loader import load_pair
    from rl_trading.features import build_features
    from rl_trading.wfo import run_wfo
    from rl_trading.metrics import compute_summary, plot_equity_curve
    import pandas as pd
    from pathlib import Path
    
    print("Imports OK", flush=True)
    
    # Load data
    pair = "EURUSD"
    fp = f"forex_data_by_pair/{pair}.txt"
    print(f"Loading {fp}...", flush=True)
    df = load_pair(fp)
    print(f"Loaded {len(df)} rows", flush=True)
    
    # Run WFO with 24 folds, 500k timesteps, CUDA
    print("Starting WFO...", flush=True)
    report = run_wfo(
        pair_name=pair,
        df=df,
        train_months=12,
        test_months=1,
        max_folds=24,
        initial_balance=1_000.0,
        lot_size=1_000.0,
        pip_cost=0.0001,
        total_timesteps=500_000,
        initial_lr=1e-4,
        ent_coef=0.01,
        lstm_hidden_size=128,
        trade_penalty=0.75,
        whipsaw_window=30,
        whipsaw_penalty=1.25,
        position_cost=0.002,
        min_hold_steps=5,
        drawdown_penalty=0.5,
        chain_balance=False,
        device="cuda",
        seed=42,
    )
    
    print(f"\nCompleted {len(report.folds)} folds", flush=True)
    
    if report.folds:
        summary = compute_summary(report)
        print(f"\n{summary.to_string(index=False)}", flush=True)
        
        plot_equity_curve(report, output_dir="results")
        print("\nEquity curve saved", flush=True)
        
        summary.to_csv("results/wfo_summary.csv", index=False)
        print("Summary CSV saved", flush=True)
    else:
        print("No folds completed!", flush=True)
    
    print("\n=== WRAPPER DONE ===", flush=True)

except Exception as e:
    print(f"\n=== ERROR ===\n{traceback.format_exc()}", flush=True)
finally:
    log_file.close()
