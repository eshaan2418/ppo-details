"""python scripts/run.py --env Hopper-v5 --seed 0 [--ablate adv_norm] [--steps N]"""

import argparse

from ppo_details.ppo import PRESETS, PPOConfig, train

ENVS = {"CartPole-v1": ("discrete", 500_000), "Acrobot-v1": ("discrete", 500_000),
        "Hopper-v5": ("mujoco", 1_000_000), "HalfCheetah-v5": ("mujoco", 1_000_000)}
ABLATIONS = ["adv_norm", "value_clip", "obs_norm", "reward_norm", "ortho_init", "lr_anneal",
             "grad_clip"]

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=sorted(ENVS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ablate", default=None, choices=ABLATIONS)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--out", default="eval/results/runs")
a = ap.parse_args()
preset, steps = ENVS[a.env]
kw = dict(PRESETS[preset])
tag = "full"
if a.ablate:
    kw[a.ablate] = False
    tag = f"no_{a.ablate}"
cfg = PPOConfig(env_id=a.env, seed=a.seed, total_steps=a.steps or steps, tag=tag, **kw)
s = train(cfg, a.out)
print(f"{a.env} {tag} seed {a.seed}: final return {s['final_return']:.1f}, "
      f"wall {s['wall_s']:.0f}s")
