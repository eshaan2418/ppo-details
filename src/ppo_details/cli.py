"""ppo-details CLI: train PPO and print the learning curve.

    ppo-details train --env CartPole-v1 --steps 100000 [--ablate reward_norm]
"""

from __future__ import annotations

import argparse
import tempfile

from .ppo import PRESETS, PPOConfig, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppo-details")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--env", default="CartPole-v1")
    t.add_argument("--steps", type=int, default=100_000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ablate", default=None)
    a = ap.parse_args(argv)
    preset = "discrete" if a.env in ("CartPole-v1", "Acrobot-v1") else "mujoco"
    kw: dict = dict(PRESETS[preset])
    if a.ablate:
        kw[a.ablate] = False
    cfg = PPOConfig(env_id=a.env, seed=a.seed, total_steps=a.steps,
                    tag=f"no_{a.ablate}" if a.ablate else "full", **kw)
    with tempfile.TemporaryDirectory() as d:
        s = train(cfg, d, log_every=max(1, a.steps // (cfg.num_envs * cfg.num_steps) // 10))
    for r in s["log"]:
        ret = r["mean_return_last20"]
        bar = "#" * int(max(0, ret or 0) / 10) if preset == "discrete" else ""
        print(f"step {r['step']:>8,}  return {ret:8.1f}  kl {r['approx_kl']:.4f}  "
              f"clipfrac {r['clipfrac']:.2f}  {bar}")
    print(f"final return (last 10% of training): {s['final_return']:.1f} "
          f"in {s['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
