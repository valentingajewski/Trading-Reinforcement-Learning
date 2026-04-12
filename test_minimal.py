"""Minimal test: load data then build 1 fold."""
import sys
print("Step 0: starting", flush=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

print("Step 1: imports", flush=True)
from rl_trading.data_loader import load_pair
from rl_trading.features import build_features
from rl_trading.environment import ForexTradingEnv
from rl_trading.agent import build_agent
from sklearn.preprocessing import RobustScaler
import numpy as np
print("Step 2: imports done", flush=True)

# Load small subset
import pandas as pd
print("Step 3: loading data (nrows=50000)...", flush=True)
raw = pd.read_csv("forex_data_by_pair/EURUSD.txt", nrows=50000)
print(f"Step 4: loaded {len(raw)} rows", flush=True)

# Parse
from rl_trading.data_loader import COLUMN_MAP
raw.rename(columns=COLUMN_MAP, inplace=True)
raw["date"] = raw["date"].astype(str)
raw["time"] = raw["time"].astype(str).str.zfill(6)
raw.index = pd.to_datetime(raw["date"] + raw["time"], format="%Y%m%d%H%M%S")
raw.drop(columns=["ticker", "date", "time"], inplace=True)
for col in ["open", "high", "low", "close", "volume"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
raw.dropna(inplace=True)
print(f"Step 5: parsed {len(raw)} rows", flush=True)

# Build features
feats = build_features(raw)
valid = feats.dropna()
print(f"Step 6: features {valid.shape}", flush=True)

# Scale
scaler = RobustScaler()
scaled = scaler.fit_transform(valid.values)
prices = raw.loc[valid.index, "close"].values
print(f"Step 7: scaled {scaled.shape}", flush=True)

# Build env with episode cap
env = ForexTradingEnv(
    features=scaled[:10000],
    prices=prices[:10000],
    initial_balance=100_000.0,
    lot_size=100_000.0,
    pip_cost=0.0001,
    reward_scaling=1000.0,
    episode_length=4096,
)
print(f"Step 8: env created, obs_space={env.observation_space.shape}", flush=True)

# Build agent
print("Step 9: building agent...", flush=True)
model = build_agent(env=env, device="cpu", n_steps=128, batch_size=32, n_epochs=2)
print("Step 10: agent built", flush=True)

# Train briefly
print("Step 11: training 256 steps...", flush=True)
model.learn(total_timesteps=256, progress_bar=False)
print("Step 12: training done", flush=True)

# Evaluate
obs, _ = env.reset()
lstm_states = None
ep_start = np.ones((1,), dtype=bool)
actions = []
for i in range(500):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    ep_start = np.array([terminated or truncated], dtype=bool)
    actions.append(int(action))
    if terminated or truncated:
        break

from collections import Counter
print(f"Step 13: actions distribution = {Counter(actions)}", flush=True)
print(f"Step 14: trades = {env.trade_count}, balance = {env._balance:.2f}", flush=True)
print("ALL DONE", flush=True)
