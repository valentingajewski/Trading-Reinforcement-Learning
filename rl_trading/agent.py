"""
agent.py — RecurrentPPO agent configuration with learning-rate schedule.

Wraps sb3-contrib's RecurrentPPO (Actor-Critic + LSTM) and exposes a simple
interface for the walk-forward optimisation pipeline.
"""

import logging
from typing import Optional

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def linear_schedule(initial_lr: float):
    """Return a callable that linearly decays the LR from *initial_lr* → 0."""
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0 during training
        return initial_lr * progress_remaining
    return _schedule


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LoggingCallback(BaseCallback):
    """Prints mean reward every *log_interval* steps."""

    def __init__(self, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = sum(
                    ep["r"] for ep in self.model.ep_info_buffer
                ) / len(self.model.ep_info_buffer)
                logger.info(
                    "Step %d | Mean episode reward: %.6f", self.n_calls, mean_rew
                )
        return True


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(
    env,
    initial_lr: float = 3e-4,
    n_steps: int = 4096,
    batch_size: int = 128,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    policy_kwargs: Optional[dict] = None,
    device: str = "auto",
    seed: int = 42,
) -> RecurrentPPO:
    """
    Construct a RecurrentPPO agent with LSTM policy.

    Parameters
    ----------
    env : gymnasium.Env
        The trading environment (must already be wrapped if needed).
    initial_lr : float
        Peak learning rate (decays linearly to 0).
    lstm_hidden_size : int
        Number of units in each LSTM layer.
    n_lstm_layers : int
        Number of stacked LSTM layers.

    Returns
    -------
    RecurrentPPO
    """
    if policy_kwargs is None:
        policy_kwargs = {}

    policy_kwargs.setdefault("lstm_hidden_size", lstm_hidden_size)
    policy_kwargs.setdefault("n_lstm_layers", n_lstm_layers)
    # Shared feature extractor fed into separate actor / critic heads
    policy_kwargs.setdefault("shared_lstm", False)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=linear_schedule(initial_lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
        seed=seed,
    )
    logger.info(
        "Built RecurrentPPO -- LSTM(%dx%d), LR %.1e -> 0",
        lstm_hidden_size, n_lstm_layers, initial_lr,
    )
    return model
