# Model card

- **Models:** separate actor and critic MLPs (2x64 tanh). Categorical policy for discrete
  actions; diagonal Gaussian with state-independent log-std for continuous actions. They are
  trained per run and not committed.
- **Intended use:** a reference PPO implementation, and an empirical study of which
  implementation details matter, with uncertainty estimates.
- **Not intended for:** real-world control. Policies are trained in simulation only and are
  brittle outside their training environment.
- **Evaluation:** mean episodic return over the last 10% of training (stochastic policy),
  5 seeds, 95% t-intervals. Cross-checked against Stable-Baselines3 PPO with matched
  hyperparameters.
