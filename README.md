# Trading-Reinforcement-

A GitHub repository for a reinforcement-learning forex trading model.

Current strategy controls include:
- A 15-minute EMA trend gate that only allows long exposure when EMA-20 is above EMA-200.
- An out-of-sample fold kill-switch that forces the agent flat for the rest of the month after a 10% drawdown breach.
