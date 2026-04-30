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

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Return a callable that linearly decays from *initial_value* to *final_value*."""
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0 during training
        return final_value + (initial_value - final_value) * progress_remaining
    return _schedule


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LoggingCallback(BaseCallback):
    """Prints mean reward every *log_interval* steps."""

    def __init__(
        self,
        log_interval: int = 10_000,
        verbose: int = 0,
        ent_coef_start: Optional[float] = None,
        ent_coef_end: Optional[float] = None,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end

    def _decayed_ent_coef(self) -> Optional[float]:
        if self.ent_coef_start is None or self.ent_coef_end is None:
            return None
        total = max(int(getattr(self.model, "_total_timesteps", 0)), 1)
        ratio = min(max(self.model.num_timesteps / total, 0.0), 1.0)
        return self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * ratio

    def _on_step(self) -> bool:
        decayed_ent = self._decayed_ent_coef()
        if decayed_ent is not None:
            self.model.ent_coef = float(decayed_ent)
            self.logger.record("train/ent_coef", float(decayed_ent))

        if self.n_calls % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = sum(
                    ep["r"] for ep in self.model.ep_info_buffer
                ) / len(self.model.ep_info_buffer)
                if decayed_ent is not None:
                    logger.info(
                        "Step %d | Mean episode reward: %.6f | ent_coef: %.4f",
                        self.n_calls,
                        mean_rew,
                        decayed_ent,
                    )
                else:
                    logger.info(
                        "Step %d | Mean episode reward: %.6f", self.n_calls, mean_rew
                    )
        return True


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(
    env,
    initial_lr: float = 1e-4,
    final_lr: Optional[float] = None,
    n_steps: int = 8192,
    batch_size: int = 128,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.02,
    ent_coef_final: float = 0.005,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    policy_kwargs: Optional[dict] = None,
    tensorboard_log: Optional[str] = None,
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
        Initial learning rate.
    final_lr : float or None
        Final learning rate after decay. When omitted, the schedule stays constant.
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

    lr_end = initial_lr if final_lr is None else final_lr

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=linear_schedule(initial_lr, lr_end),
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
        tensorboard_log=tensorboard_log,
        device=device,
        seed=seed,
    )
    logger.info(
        "Built RecurrentPPO -- LSTM(%dx%d), LR %.1e -> %.1e, ent %.3f -> %.3f",
        lstm_hidden_size, n_lstm_layers, initial_lr, lr_end, ent_coef, ent_coef_final,
    )
    return model
