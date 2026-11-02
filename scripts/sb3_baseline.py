"""Stable-Baselines3 PPO with hyperparameters matched to ours (external correctness check).

    python scripts/sb3_baseline.py --env Hopper-v5 --seed 0
Writes eval/results/sb3/<env>_s<seed>.json with the same summary fields as our runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

SETTINGS = {
    "CartPole-v1": dict(n_envs=8, n_steps=128, batch=256, epochs=4, lr=2.5e-4, ent=0.01,
                        norm=False, steps=500_000),
    "Acrobot-v1": dict(n_envs=8, n_steps=128, batch=256, epochs=4, lr=2.5e-4, ent=0.01,
                       norm=False, steps=500_000),
    "Hopper-v5": dict(n_envs=1, n_steps=2048, batch=64, epochs=10, lr=3e-4, ent=0.0,
                      norm=True, steps=1_000_000),
    "HalfCheetah-v5": dict(n_envs=1, n_steps=2048, batch=64, epochs=10, lr=3e-4, ent=0.0,
                           norm=True, steps=1_000_000),
}


class Episodes(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episodes: list[tuple[int, float, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes.append((self.num_timesteps, float(info["episode"]["r"]),
                                      int(info["episode"]["l"])))
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = SETTINGS[a.env]
    torch.set_num_threads(1)
    env = make_vec_env(a.env, n_envs=s["n_envs"], seed=a.seed * 1000)
    if s["norm"]:
        env = VecNormalize(env, gamma=0.99)
    model = PPO("MlpPolicy", env, learning_rate=lambda f: s["lr"] * f, n_steps=s["n_steps"],
                batch_size=s["batch"], n_epochs=s["epochs"], gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, clip_range_vf=0.2, ent_coef=s["ent"], vf_coef=0.5,
                max_grad_norm=0.5, seed=a.seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                   activation_fn=torch.nn.Tanh))
    cb = Episodes()
    t0 = time.perf_counter()
    model.learn(s["steps"], callback=cb)
    wall = time.perf_counter() - t0
    cut = int(0.9 * s["steps"])
    final = [r for st, r, _ in cb.episodes if st >= cut]
    out = Path("eval/results/sb3")
    out.mkdir(parents=True, exist_ok=True)
    rec = {"env": a.env, "seed": a.seed, "final_return": float(np.mean(final)),
           "auc_return": float(np.mean([r for _, r, _ in cb.episodes])),
           "episodes": cb.episodes, "wall_s": wall, "settings": s,
           "sb3": __import__("stable_baselines3").__version__, "gymnasium": gym.__version__}
    (out / f"{a.env}_s{a.seed}.json").write_text(json.dumps(rec))
    print(f"SB3 {a.env} seed {a.seed}: final {rec['final_return']:.1f} wall {wall:.0f}s")


if __name__ == "__main__":
    main()
