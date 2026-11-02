# Design

## Algorithm
Clipped-surrogate PPO (Schulman et al., 2017): for each update, collect `num_envs ×
num_steps` transitions, compute GAE(γ=0.99, λ=0.95) advantages and returns, then run
`update_epochs` passes of minibatch SGD (Adam, ε=1e-5) on

`L = −E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)] + c_v·L_V − c_e·H[π]`, with `r = π/π_old`, ε=0.2.

## Conventions that are easy to get wrong
- **Done flags.** `done_buf[t]` means "the observation at step t starts a new episode".
  GAE then uses `1 − done_buf[t+1]` as the non-terminal mask for transition t, and the final
  `done` for the last step. A Hypothesis test checks this against an explicit per-env
  reference recursion.
- **Episode statistics with same-step autoreset (Gymnasium 1.x).** Finished episodes appear
  under `info["final_info"]["episode"]`, not `info["episode"]`. The first version read the
  wrong key and recorded nothing.
- **Reward scaling.** Rewards are divided by the running std of the discounted return, not
  of the reward, and the return accumulator resets at episode boundaries.
- **Value clipping.** It uses the old value predictions from the rollout:
  `V_clip = V_old + clip(V − V_old, −ε, ε)`, loss `max((V−R)², (V_clip−R)²)`.

## Ablation design
Each ablation turns off exactly one detail of the full configuration, holding everything
else fixed (one-at-a-time). This measures each detail's marginal contribution in the
presence of the others, not interactions. Observation and reward normalisation are only
ablated in MuJoCo, because the discrete presets do not use them.

## Presets
| | discrete (CartPole, Acrobot) | MuJoCo (Hopper, HalfCheetah) |
|---|---|---|
| envs × steps per rollout | 8 × 128 | 1 × 2048 |
| epochs × minibatches | 4 × 4 | 10 × 32 |
| learning rate (linearly annealed) | 2.5e-4 | 3e-4 |
| entropy coefficient | 0.01 | 0.0 |
| observation / reward normalisation | off | on |

These are the standard settings from the PPO paper and its widely used reimplementations.
They were not tuned here.
