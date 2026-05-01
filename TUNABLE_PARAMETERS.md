# Tunable Parameters Guide

This file lists the parameters you can tune in the current RL trading pipeline and what typically happens when you increase or decrease them.

## 1) Parameters you can tune directly from CLI (`run_trading.py`)

| Parameter | Current default | Increase / set higher | Decrease / set lower |
|---|---:|---|---|
| `--train-months` | `12` | More training history per fold, more stable policy, slower adaptation to regime changes | Faster adaptation to recent regime, but noisier policy and higher overfit risk |
| `--test-months` | `1` | Longer OOS period per fold, fewer fold transitions, smoother per-fold stats | Shorter OOS period, more folds, more variance in fold outcomes |
| `--max-folds` | `None` (all) | More robust backtest coverage, much longer runtime | Faster experiments, less statistical confidence |
| `--timesteps` | `300000` | Better convergence potential, longer runtime, possible overfit on fold | Faster runs, possible undertraining |
| `--n-steps` | `8192` | Longer rollout chunks, better long-context credit assignment, slower policy refresh | Faster refresh, noisier updates |
| `--lr` | `1e-4` | Faster learning, but more unstable updates and oscillation risk | Slower but steadier learning, may underfit if too small |
| `--final-lr` | `match --lr` | Higher final LR preserves larger updates later in training | Lower final LR makes late training more conservative; `0` decays to zero |
| `--ent-coef` | `0.02` | More exploration, usually more action switching and trade frequency | More exploitation, usually fewer switches, can get stuck |
| `--ent-coef-final` | `0.01` | Preserves more late-stage exploration | Faster convergence toward deterministic policy |
| `--trade-penalty` | `0.45` | Stronger discouragement of trading, lower churn, can undertrade | More willingness to trade, can increase churn |
| `--whipsaw-window` | `30` | More reversals counted as "too fast", fewer quick flips | Less strict reversal window, more rapid switching allowed |
| `--whipsaw-penalty` | `1.25` | Heavier punishment for flip-flop behavior, lower turnover | Weaker anti-whipsaw control, turnover can rise |
| `--position-cost` | `0.002` | More incentive to go flat when edge is weak, fewer long holds | Less cost to stay exposed, more continuous market exposure |
| `--min-hold-steps` | `15` | Forces longer holds, reduces micro-churn but slows reaction | Allows faster position changes, can overtrade |
| `--drawdown-penalty` | `1.0` | More risk-aversion when drawdown worsens, can reduce tail losses but mute upside | More return-seeking behavior, but larger tail-risk possible |
| `--turnover-penalty` | `0.25` | Adds explicit churn penalty scaled by position change size | Lets the agent reverse more freely |
| `--reward-scaling` | `100.0` | Stronger direct PnL signal in reward | Relative weight shifts toward DSR and penalties |
| `--reward-clip` | `1.0` | Tighter PPO-stable reward bounds | Allows larger per-step reward spikes |
| `--lstm-size` | `64` | Higher model capacity for complex patterns, more compute and overfit risk | Simpler model, faster training, may miss complex temporal structure |
| `--lstm-layers` | `1` | Deeper recurrent stack, more capacity and training cost | Shallower recurrent stack, simpler optimization |
| `--balance` | `1000.0` | Larger nominal account; in this setup many costs are account-normalized so behavior impact is limited | Smaller nominal account; similar behavior if other scales remain proportional |
| `--lot-size` | `1000.0` | Larger PnL and cost swings per move, higher risk per decision | Smaller per-trade impact, lower volatility |
| `--pip-cost` | `0.0001` | Higher effective friction, discourages frequent trading | Lower friction, strategy can trade more |
| `--device` | `auto` | `cuda` usually speeds training wall-clock time | `cpu` slower training wall-clock time |
| `--chain-balance` | `False` | `True`: folds compound gains/losses, more realistic path dependency | `False`: each fold starts same balance, cleaner fold comparability |
| `--seed` | `42` | Different random path, useful for robustness checks | Same as above; lower/higher number has no inherent quality ordering |
| `--pair` | `None` | Set to single pair for focused training and faster run | `None` runs all available pairs |
| `--data-dir` | `forex_data_by_pair` | Point to a different dataset source | Use current dataset source |
| `--output-dir` | `results` | Save reports elsewhere (no learning behavior change) | Save reports in default location |
| `--tensorboard-log` | `results/tensorboard` | More experiment visibility and entropy tracking | No effect if you do not use TensorBoard |

## 2) PPO model hyperparameters available in code (`rl_trading/agent.py`)

These are tunable by editing `build_agent(...)` inputs or defaults.

| Parameter | Current default | Increase / set higher | Decrease / set lower |
|---|---:|---|---|
| `n_steps` | `8192` | Longer rollout chunks, lower gradient noise, slower policy refresh | Faster policy refresh, noisier updates |
| `batch_size` | `128` | Smoother gradient estimate, more memory use | Noisier gradients, sometimes better regularization |
| `n_epochs` | `10` | More optimization per rollout, better sample usage, more overfit risk | Less optimization per rollout, faster but may underfit |
| `gamma` | `0.99` | More long-term reward focus | More short-term reward focus |
| `gae_lambda` | `0.95` | Lower bias / higher variance advantages | Higher bias / lower variance advantages |
| `clip_range` | `0.2` | Looser policy update constraint, potentially unstable jumps | Tighter update constraint, more conservative learning |
| `initial_lr` | `1e-4` | Faster optimization progress, but larger update instability risk | Slower but steadier optimization |
| `final_lr` | `match initial_lr` | Keeps updates larger later in training | More decay means more conservative late training |
| `ent_coef` | `0.02` | More exploration and action diversity | More deterministic exploitation |
| `ent_coef_final` | `0.01` | Maintains more late-stage stochasticity | Quicker convergence to lower exploration |
| `lstm_hidden_size` | `64` | More sequence modeling capacity, more compute and overfit risk | Lower capacity, faster but may miss temporal structure |
| `n_lstm_layers` | `1` | Deeper recurrent model, more capacity and training difficulty | Simpler recurrent stack |
| `max_grad_norm` | `0.5` | Weaker clipping, possible exploding updates | Stronger clipping, safer but can slow learning |
| `shared_lstm` (policy kwargs) | `False` | `True`: actor/critic share representation, fewer params | `False`: separate actor/critic representations, more flexible |

## 3) Environment and reward shaping parameters (`rl_trading/environment.py`)

| Parameter | Current default | Increase / set higher | Decrease / set lower |
|---|---:|---|---|
| `reward_scaling` | `100.0` (set in WFO training env) | Stronger PnL term influence in reward | DSR and penalties dominate relatively more |
| `eta` (DSR EMA rate) | `0.001` | DSR reacts faster to recent returns, noisier risk signal | Smoother/slower DSR signal |
| `episode_length` | `20160` (set in WFO training env) | Longer episodes, better long-context learning, slower resets | More frequent resets, faster feedback cycles |
| `trade_penalty` | `0.45` | Fewer trades, lower churn, potential undertrading | More trades, higher churn risk |
| `whipsaw_window` | `30` | More strict anti-flip region | Less strict anti-flip region |
| `whipsaw_penalty` | `1.25` | Stronger punishment for rapid reversals | Weaker punishment for rapid reversals |
| `position_cost` | `0.002` | More pressure to stay flat unless conviction is high | More willingness to stay in market |
| `min_hold_steps` | `15` | Forces commitment to positions, lower churn | Enables rapid switching |
| `drawdown_penalty` | `1.0` | More penalty when drawdown worsens, better tail-risk control | Less tail-risk control, potentially higher upside and deeper losses |
| `turnover_penalty` | `0.25` | Penalizes gross position changes, especially rapid full reversals | Reduces explicit churn control |
| `reward_clip` | `1.0` | Keeps per-step reward inside PPO-friendly bounds | Allows bigger reward outliers |
| `pip_cost` | `0.0001` | Simulates worse spread/fees, fewer trades | Simulates cheaper execution, more trades |
| `lot_size` | `1000.0` | Larger PnL/cost sensitivity per price move | Smaller PnL/cost sensitivity |
| `initial_balance` | `1000.0` | Changes nominal account level for env state normalization context | Same opposite direction |

## 4) WFO process controls in code (`rl_trading/wfo.py`)

| Parameter | Current default | Increase / set higher | Decrease / set lower |
|---|---:|---|---|
| `train_months` | `12` | More historical context per fold | More recency focus |
| `test_months` | `1` | Longer OOS spans per fold | Shorter OOS spans, more folds |
| `max_folds` | `None` | More coverage, slower run | Faster run, less confidence |
| `chain_balance` | `False` | `True` compounds fold path | `False` isolates fold starts |
| Fold skip thresholds (`len(train)<1000` or `len(test)<100`) | fixed in code | Higher thresholds = stricter data sufficiency | Lower thresholds = include thinner folds |

## 5) Feature engineering hyperparameters (`rl_trading/features.py`)

These require code edits.

The current pipeline also centers RSI into `[-1, 1]` before the train-only `RobustScaler` fit so bounded oscillators enter PPO in a stable numeric range.

| Parameter | Current default | Increase / set higher | Decrease / set lower |
|---|---:|---|---|
| RSI window | `14` | Smoother RSI, slower momentum signal | Faster/more reactive RSI, noisier |
| MACD fast window | `12` | Slower fast EMA component | Faster fast EMA component |
| MACD slow window | `26` | Even slower trend baseline | Faster baseline, less smoothing |
| MACD signal window | `9` | Smoother signal line | More reactive signal line |
| ATR window | `14` | Smoother volatility estimate | More reactive volatility estimate |
| EMA windows | `20, 50, 200` | Longer windows smooth trend signals | Shorter windows react faster |
| MTF resample rules | `15min`, `1h` | Coarser TFs reduce noise but lag | Finer TFs react faster but noisier |
| MTF EMA window | `50` | Smoother MTF trend estimate | Faster MTF trend estimate |
| Realized vol windows | `20`, `60` | Smoother volatility regimes | More reactive volatility regimes |
| Bollinger window | `20` | Smoother band width | More reactive band width |
| Bollinger stdev multiplier | `2` | Wider bands, fewer extreme hits | Narrower bands, more extreme hits |

## 6) Practical tuning order

Use this order for fastest gains with minimal instability:

1. `trade_penalty`, `whipsaw_penalty`, `position_cost`, `min_hold_steps` (control churn first)
2. `drawdown_penalty` (control tail-risk)
3. `timesteps`, `lr`, `ent_coef` (optimize learning dynamics)
4. `lstm_size`, `n_steps`, `batch_size` (capacity/optimizer refinement)
5. Feature windows (if behavior still regime-fragile)

## 7) Current run defaults in `run_safe.py`

Current wrapper run uses:

- `train_months=12`, `test_months=1`, `max_folds=24`
- `total_timesteps=500000`
- `initial_lr=1e-4`, `final_lr=1e-4`, `n_steps=8192`, `ent_coef=0.02 -> 0.01`
- `lstm_hidden_size=64`, `n_lstm_layers=2`
- `trade_penalty=0.45`, `whipsaw_window=30`, `whipsaw_penalty=1.25`
- `position_cost=0.002`, `min_hold_steps=15`, `drawdown_penalty=1.0`, `turnover_penalty=0.25`
- `reward_scaling=100.0`, `reward_clip=1.0`
- `chain_balance=False`, `device="cuda"`, `seed=42`
