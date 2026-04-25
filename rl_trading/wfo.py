"""
wfo.py — Walk-Forward Optimisation pipeline.

Rolling windows: 6 months training → 1 month OOS testing.
RobustScaler is fit ONLY on each training window to prevent data leakage.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from rl_trading.agent import LoggingCallback, build_agent
from rl_trading.environment import ForexTradingEnv
from rl_trading.features import build_features

logger = logging.getLogger(__name__)


@dataclass
class WFOResult:
    """Container for one OOS fold result."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    actions: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    balance_curve: np.ndarray
    timestamps: np.ndarray


@dataclass
class WFOReport:
    """Aggregated walk-forward results."""
    pair: str
    folds: list[WFOResult] = field(default_factory=list)

    @property
    def oos_prices(self) -> np.ndarray:
        return np.concatenate([f.prices for f in self.folds])

    @property
    def oos_returns(self) -> np.ndarray:
        return np.concatenate([f.returns for f in self.folds])

    @property
    def oos_balance(self) -> np.ndarray:
        return np.concatenate([f.balance_curve for f in self.folds])

    @property
    def oos_actions(self) -> np.ndarray:
        return np.concatenate([f.actions for f in self.folds])

    @property
    def oos_timestamps(self) -> np.ndarray:
        return np.concatenate([f.timestamps for f in self.folds])


# ---------------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------------

def _generate_wfo_splits(
    index: pd.DatetimeIndex,
    train_months: int = 6,
    test_months: int = 1,
):
    """
    Yield (train_start, train_end, test_start, test_end) tuples that tile the
    data with rolling windows.
    """
    start = index.min()
    end = index.max()

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        yield train_start, train_end, test_start, test_end
        cursor = test_start  # slide by test_months


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def _evaluate_fold(
    model,
    features_scaled: np.ndarray,
    prices: np.ndarray,
    initial_balance: float,
    lot_size: float,
    pip_cost: float,
    min_hold_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run a trained model on OOS data and collect actions, per-step returns,
    the balance curve, and trade diagnostics.
    """
    env = ForexTradingEnv(
        features=features_scaled,
        prices=prices,
        initial_balance=initial_balance,
        lot_size=lot_size,
        pip_cost=pip_cost,
        min_hold_steps=min_hold_steps,
    )
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    actions_list = []
    balance_list = [initial_balance]
    returns_list = []

    done = False
    while not done:
        prev_balance = env._balance
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        obs, _reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)

        executed_action = 0 if env._position == -1 else 1 if env._position == 0 else 2
        actions_list.append(executed_action)
        balance_list.append(env._balance)
        # OOS performance metrics must use true account return, not training reward.
        if abs(prev_balance) > 1e-12:
            returns_list.append((env._balance - prev_balance) / prev_balance)
        else:
            returns_list.append(0.0)

    # Trade diagnostics
    actions_arr = np.array(actions_list)
    positions = np.array(env.position_history)
    diagnostics = {
        "trade_count": env.trade_count,
        "position_sell_pct": round(100 * np.mean(positions == -1), 1),
        "position_flat_pct": round(100 * np.mean(positions == 0), 1),
        "position_buy_pct": round(100 * np.mean(positions == 1), 1),
        "turnover": env.trade_count / max(len(actions_list), 1),
    }

    return (
        actions_arr,
        np.array(returns_list),
        np.array(balance_list),
        diagnostics,
    )


# ---------------------------------------------------------------------------
# Main WFO runner
# ---------------------------------------------------------------------------

def run_wfo(
    pair_name: str,
    df: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 12,
    max_folds: Optional[int] = None,
    initial_balance: float = 1_000.0,
    lot_size: float = 1_000.0,
    pip_cost: float = 0.0001,
    total_timesteps: int = 100_000,
    initial_lr: float = 3e-4,
    n_steps: int = 8192,
    ent_coef: float = 0.05,
    ent_coef_final: float = 0.02,
    clip_range: float = 0.1,
    batch_size: int = 256,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    trade_penalty: float = 0.75,
    whipsaw_window: int = 30,
    whipsaw_penalty: float = 1.25,
    position_cost: float = 0.002,
    min_hold_steps: int = 15,
    drawdown_penalty: float = 1.0,
    reward_scaling: float = 10.0,
    chain_balance: bool = False,
    device: str = "auto",
    seed: int = 42,
) -> WFOReport:
    """
    Execute full walk-forward optimisation for one forex pair.

    Parameters
    ----------
    pair_name : str
        Identifier (e.g. "EURUSD").
    df : pd.DataFrame
        Raw minute OHLCV with DatetimeIndex.
    train_months / test_months : int
        Rolling window sizes.
    max_folds : int or None
        Cap the number of WFO folds (useful for fast diagnostics).
    lot_size : float
        Trade size in base currency units (1000 = 0.01 standard lot).
    total_timesteps : int
        Training budget per fold.

    Returns
    -------
    WFOReport
    """
    # Pre-compute features on the entire dataset (indicators only look backward)
    all_features = build_features(df)
    # Determine the first valid index after NaN warm-up
    valid_mask = all_features.notna().all(axis=1)
    first_valid = valid_mask.idxmax()
    all_features = all_features.loc[first_valid:]
    df = df.loc[first_valid:]

    report = WFOReport(pair=pair_name)

    splits = list(_generate_wfo_splits(
        df.index, train_months=train_months, test_months=test_months
    ))
    if max_folds is not None:
        splits = splits[:max_folds]
    logger.info(
        "WFO for %s -- %d folds (%dm train / %dm test)",
        pair_name, len(splits), train_months, test_months,
    )

    # Optional capital chaining across folds (disabled by default for cleaner WFO comparability).
    running_balance = initial_balance

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        logger.info(
            "Fold %d/%d  train [%s -> %s]  test [%s -> %s]",
            fold_idx + 1, len(splits),
            tr_s.date(), tr_e.date(), te_s.date(), te_e.date(),
        )

        # --- Slice ---
        train_feat = all_features.loc[tr_s:tr_e]
        train_prices = df.loc[tr_s:tr_e, "close"]
        test_feat = all_features.loc[te_s:te_e]
        test_prices = df.loc[te_s:te_e, "close"]

        if len(train_feat) < 1000 or len(test_feat) < 100:
            logger.warning("Fold %d skipped -- insufficient data", fold_idx + 1)
            continue

        # --- Scale (fit on train only) ---
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_feat.values)
        test_scaled = scaler.transform(test_feat.values)

        # --- Training env ---
        train_env = ForexTradingEnv(
            features=train_scaled,
            prices=train_prices.values,
            initial_balance=initial_balance,
            lot_size=lot_size,
            pip_cost=pip_cost,
            reward_scaling=reward_scaling,
            episode_length=20160,     # ~14 days — captures weekly/biweekly patterns
            trade_penalty=trade_penalty,
            whipsaw_window=whipsaw_window,
            whipsaw_penalty=whipsaw_penalty,
            position_cost=position_cost,
            min_hold_steps=min_hold_steps,
            drawdown_penalty=drawdown_penalty,
        )

        # --- Build & train agent ---
        model = build_agent(
            env=train_env,
            initial_lr=initial_lr,
            n_steps=n_steps,
            clip_range=clip_range,
            batch_size=batch_size,
            ent_coef=ent_coef,
            ent_coef_final=ent_coef_final,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            device=device,
            seed=seed,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=LoggingCallback(
                log_interval=25_000,
                ent_coef_start=ent_coef,
                ent_coef_end=ent_coef_final,
            ),
            progress_bar=False,
        )

        fold_initial_balance = running_balance if chain_balance else initial_balance
        if fold_initial_balance <= 0:
            logger.warning(
                "Stopping WFO at fold %d: non-positive starting balance %.2f",
                fold_idx + 1,
                fold_initial_balance,
            )
            break

        # --- OOS evaluation ---
        actions, returns, balance, diag = _evaluate_fold(
            model, test_scaled, test_prices.values,
            initial_balance=fold_initial_balance,
            lot_size=lot_size,
            pip_cost=pip_cost,
            min_hold_steps=min_hold_steps,
        )

        if chain_balance:
            running_balance = balance[-1]

        result = WFOResult(
            fold=fold_idx,
            train_start=tr_s,
            train_end=tr_e,
            test_start=te_s,
            test_end=te_e,
            actions=actions,
            prices=test_prices.values[:len(actions)],
            returns=returns,
            balance_curve=balance,
            timestamps=test_feat.index.values[:len(actions)],
        )
        report.folds.append(result)
        logger.info(
            "Fold %d OOS -- final balance: %.2f  (%.2f%%)  trades: %d  "
            "positions: sell=%.1f%% flat=%.1f%% buy=%.1f%%  turnover=%.4f",
            fold_idx + 1,
            balance[-1],
            100 * (balance[-1] / balance[0] - 1),
            diag["trade_count"],
            diag["position_sell_pct"],
            diag["position_flat_pct"],
            diag["position_buy_pct"],
            diag["turnover"],
        )

        if chain_balance and running_balance <= 0:
            logger.warning(
                "Account exhausted after fold %d (balance=%.2f). Stopping remaining folds.",
                fold_idx + 1,
                running_balance,
            )
            break

    return report
