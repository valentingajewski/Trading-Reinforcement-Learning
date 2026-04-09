"""
environment.py — Custom Gymnasium trading environment.

Implements:
- Discrete action space: 0 = Sell, 1 = Hold/Flat, 2 = Buy
- Differential Sharpe Ratio reward (Moody et al., 1998)
- Transaction cost penalty (1 pip per trade)
- Portfolio state: position, unrealized PnL, balance
"""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Maps discrete actions to position targets
ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}


class ForexTradingEnv(gym.Env):
    """
    Minute-level forex trading environment with Differential Sharpe Ratio reward.

    Parameters
    ----------
    features : np.ndarray, shape (T, n_features)
        Pre-computed, scaled feature matrix (no NaNs).
    prices : np.ndarray, shape (T,)
        Close prices aligned with *features*.
    initial_balance : float
        Starting account balance in quote currency units.
    pip_cost : float
        Transaction cost per trade in price units (default 0.0001 = 1 pip for
        major pairs).
    eta : float
        Adaptation rate for the differential Sharpe ratio EMA.
    episode_length : int or None
        If set, episodes are truncated after this many steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 100_000.0,
        pip_cost: float = 0.0001,
        eta: float = 0.001,
        episode_length: Optional[int] = None,
    ):
        super().__init__()

        assert len(features) == len(prices), "features / prices length mismatch"
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.initial_balance = initial_balance
        self.pip_cost = pip_cost
        self.eta = eta
        self.episode_length = episode_length

        self.n_steps = len(features)
        self.n_features = features.shape[1]

        # 3 portfolio state vars appended to observation: position, unrealized PnL, balance (normalised)
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Internal state — set in reset()
        self._current_step: int = 0
        self._position: int = 0       # -1, 0, 1
        self._entry_price: float = 0.0
        self._balance: float = initial_balance
        self._total_pnl: float = 0.0

        # Differential Sharpe Ratio running statistics
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._position = 0
        self._entry_price = 0.0
        self._balance = self.initial_balance
        self._total_pnl = 0.0
        self._A = 0.0
        self._B = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        target_position = ACTION_TO_POSITION[int(action)]
        price = self.prices[self._current_step]
        prev_position = self._position

        # --- Realise PnL on position change ---
        realised = 0.0
        if prev_position != target_position and prev_position != 0:
            realised = prev_position * (price - self._entry_price)
            self._balance += realised

        # --- Transaction cost ---
        trade_occurred = target_position != prev_position
        cost = self.pip_cost if trade_occurred else 0.0
        self._balance -= cost

        # --- Update position ---
        if target_position != 0 and target_position != prev_position:
            self._entry_price = price
        if target_position == 0:
            self._entry_price = 0.0
        self._position = target_position

        # --- Step return (for reward) ---
        step_return = self._compute_step_return(price, realised, cost)

        # --- Differential Sharpe Ratio reward ---
        reward = self._differential_sharpe(step_return)

        # --- Advance ---
        self._current_step += 1
        truncated = False
        terminated = False

        if self._current_step >= self.n_steps:
            terminated = True
        elif self.episode_length and self._current_step >= self.episode_length:
            truncated = True
        elif self._balance <= 0:
            terminated = True

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, float(reward), terminated, truncated, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        idx = min(self._current_step, self.n_steps - 1)
        market = self.features[idx]

        unrealised = 0.0
        if self._position != 0:
            unrealised = self._position * (
                self.prices[idx] - self._entry_price
            )

        portfolio = np.array([
            float(self._position),
            unrealised / self.initial_balance,          # normalised unPnL
            (self._balance - self.initial_balance) / self.initial_balance,  # normalised balance delta
        ], dtype=np.float32)

        return np.concatenate([market, portfolio])

    def _compute_step_return(self, price: float, realised: float,
                             cost: float) -> float:
        """Net return for this step (realised + mark-to-market change − cost)."""
        unrealised_change = 0.0
        if self._position != 0 and self._current_step > 0:
            prev_price = self.prices[self._current_step - 1]
            unrealised_change = self._position * (price - prev_price)
        return (realised + unrealised_change - cost) / self.initial_balance

    def _differential_sharpe(self, R_t: float) -> float:
        """
        Differential Sharpe Ratio (Moody & Saffell, 2001).

        D_t = (B_{t-1} * ΔA_t − 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} − A_{t-1}^2)^{3/2}

        Where ΔA_t = R_t − A_{t-1}, ΔB_t = R_t^2 − B_{t-1}.
        """
        delta_A = R_t - self._A
        delta_B = R_t ** 2 - self._B

        denom = (self._B - self._A ** 2)
        if denom <= 1e-12:
            dsr = R_t  # fallback for first steps
        else:
            dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)

        # Update EMAs
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B

        return dsr
