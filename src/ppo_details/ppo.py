"""Proximal Policy Optimization (clipped surrogate), from scratch, with switchable details.

Every implementation detail studied in the ablations is a flag on :class:`PPOConfig`. The
defaults follow the widely used reference settings (Schulman et al. 2017; Huang et al. 2022,
"The 37 Implementation Details of PPO"): GAE(λ=0.95), clipped objective ε=0.2, advantage
normalisation per minibatch, clipped value loss, orthogonal init, linear LR annealing, global
gradient-norm clipping at 0.5, and, for continuous control, observation normalisation and
reward scaling by a running estimate of the discounted return's standard deviation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_steps: int = 500_000
    num_envs: int = 8
    num_steps: int = 128  # rollout length per env
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    # --- implementation details (ablation switches) ---
    adv_norm: bool = True
    value_clip: bool = True
    obs_norm: bool = False  # set True for continuous control presets
    reward_norm: bool = False
    ortho_init: bool = True
    lr_anneal: bool = True
    grad_clip: bool = True
    tag: str = "full"


PRESETS: dict[str, dict[str, Any]] = {
    "discrete": dict(num_envs=8, num_steps=128, lr=2.5e-4, update_epochs=4, num_minibatches=4,
                     ent_coef=0.01, obs_norm=False, reward_norm=False),
    "mujoco": dict(num_envs=1, num_steps=2048, lr=3e-4, update_epochs=10, num_minibatches=32,
                   ent_coef=0.0, obs_norm=True, reward_norm=True),
}


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


class Normalizer:
    """Observation normalisation and return-based reward scaling for a vector env."""

    def __init__(self, n_envs: int, obs_shape: tuple[int, ...], cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.obs_rms = RunningMeanStd(obs_shape)
        self.ret_rms = RunningMeanStd(())
        self.returns = np.zeros(n_envs)

    def obs(self, o: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.cfg.obs_norm:
            return o
        if update:
            self.obs_rms.update(o)
        return np.clip((o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

    def reward(self, r: np.ndarray, done: np.ndarray) -> np.ndarray:
        if not self.cfg.reward_norm:
            return r
        self.returns = self.returns * self.cfg.gamma + r
        self.ret_rms.update(self.returns)
        self.returns[done] = 0.0
        return np.clip(r / np.sqrt(self.ret_rms.var + 1e-8), -10, 10)


def layer(i: int, o: int, std: float, ortho: bool) -> nn.Linear:
    lin = nn.Linear(i, o)
    if ortho:
        nn.init.orthogonal_(lin.weight, std)
        nn.init.constant_(lin.bias, 0.0)
    return lin


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_space: gym.Space, cfg: PPOConfig) -> None:
        super().__init__()
        h, o = cfg.hidden, cfg.ortho_init
        s2 = float(np.sqrt(2))
        self.critic = nn.Sequential(layer(obs_dim, h, s2, o), nn.Tanh(), layer(h, h, s2, o),
                                    nn.Tanh(), layer(h, 1, 1.0, o))
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        n_out = int(act_space.n) if self.discrete else int(np.prod(act_space.shape or ()))  # type: ignore[attr-defined]
        self.actor = nn.Sequential(layer(obs_dim, h, s2, o), nn.Tanh(), layer(h, h, s2, o),
                                   nn.Tanh(), layer(h, n_out, 0.01, o))
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(n_out))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def dist(self, x: torch.Tensor) -> Categorical | Normal:
        out = self.actor(x)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def log_prob(self, d: Categorical | Normal, a: torch.Tensor) -> torch.Tensor:
        lp = d.log_prob(a)
        return lp if self.discrete else lp.sum(-1)

    def entropy(self, d: Categorical | Normal) -> torch.Tensor:
        e = d.entropy()
        return e if self.discrete else e.sum(-1)


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray,
        last_done: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over a (T, N) rollout.

    ``dones[t]`` marks that the observation at step t starts a new episode (the previous
    step ended one), the convention used by vectorised auto-resetting environments.
    """
    t_max = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(t_max)):
        if t == t_max - 1:
            nonterminal = 1.0 - last_done
            next_v = last_value
        else:
            nonterminal = 1.0 - dones[t + 1]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def train(cfg: PPOConfig, out_dir: str | Path = "eval/results/runs",
          log_every: int = 10) -> dict:
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id))
         for _ in range(cfg.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None
    agent = Agent(int(np.prod(obs_shape)), envs.single_action_space, cfg)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    norm = Normalizer(cfg.num_envs, obs_shape, cfg)
    T, N = cfg.num_steps, cfg.num_envs
    batch = T * N
    mb = batch // cfg.num_minibatches
    n_updates = cfg.total_steps // batch
    obs_buf = np.zeros((T, N, *obs_shape), dtype=np.float32)
    act_shape = envs.single_action_space.shape or ()
    act_buf = np.zeros((T, N, *act_shape), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)
    raw, _ = envs.reset(seed=[cfg.seed * 1000 + i for i in range(N)])
    obs = norm.obs(raw)
    done = np.zeros(N)
    episodes: list[tuple[int, float, int]] = []  # (global step, return, length)
    log = []
    step = 0
    t0 = time.perf_counter()
    for upd in range(n_updates):
        if cfg.lr_anneal:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - upd / n_updates)
        for t in range(T):
            step += N
            obs_buf[t] = obs
            done_buf[t] = done
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                d = agent.dist(x)
                a = d.sample()
                logp_buf[t] = agent.log_prob(d, a).numpy()
                val_buf[t] = agent.value(x).numpy()
            act_buf[t] = a.numpy()
            act = a.numpy()
            if continuous:
                low, high = envs.single_action_space.low, envs.single_action_space.high  # type: ignore[attr-defined]
                act = np.clip(act, low, high)
            raw, r, term, trunc, info = envs.step(act)
            done = np.logical_or(term, trunc).astype(np.float64)
            rew_buf[t] = norm.reward(r, done.astype(bool))
            obs = norm.obs(raw)
            # same-step autoreset: finished episodes' statistics are under "final_info"
            fin = info.get("final_info", {})
            if "episode" in fin:
                for i in np.nonzero(fin["_episode"])[0]:
                    episodes.append((step, float(fin["episode"]["r"][i]),
                                     int(fin["episode"]["l"][i])))
        with torch.no_grad():
            last_v = agent.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        adv, ret = gae(rew_buf, val_buf, done_buf, last_v, done, cfg.gamma, cfg.gae_lambda)
        b_obs = torch.as_tensor(obs_buf.reshape(batch, *obs_shape))
        b_act = torch.as_tensor(act_buf.reshape(batch, *act_shape))
        if not continuous:
            b_act = b_act.long()
        b_logp = torch.as_tensor(logp_buf.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        b_val = torch.as_tensor(val_buf.reshape(-1))
        clipfracs, approx_kls = [], []
        for _ in range(cfg.update_epochs):
            perm = rng.permutation(batch)
            for s in range(0, batch, mb):
                idx = torch.as_tensor(perm[s : s + mb])
                d = agent.dist(b_obs[idx])
                new_logp = agent.log_prob(d, b_act[idx])
                ratio = (new_logp - b_logp[idx]).exp()
                with torch.no_grad():
                    approx_kls.append(float(((ratio - 1) - (new_logp - b_logp[idx])).mean()))
                    clipfracs.append(float(((ratio - 1).abs() > cfg.clip_coef).float().mean()))
                a_mb = b_adv[idx]
                if cfg.adv_norm:
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v = agent.value(b_obs[idx])
                if cfg.value_clip:
                    v_clip = b_val[idx] + (v - b_val[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((v - b_ret[idx]) ** 2,
                                             (v_clip - b_ret[idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                ent = agent.entropy(d).mean()
                loss = pg - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
        if (upd + 1) % log_every == 0 or upd == n_updates - 1:
            recent = [e[1] for e in episodes[-20:]]
            log.append({"step": step, "mean_return_last20": float(np.mean(recent))
                        if recent else None, "approx_kl": float(np.mean(approx_kls)),
                        "clipfrac": float(np.mean(clipfracs)), "v_loss": float(v_loss.detach()),
                        "elapsed_s": time.perf_counter() - t0})
    envs.close()
    final_cut = int(0.9 * cfg.total_steps)
    final = [r for s, r, _ in episodes if s >= final_cut]
    summary = {
        "config": asdict(cfg), "episodes": episodes, "log": log,
        "final_return": float(np.mean(final)) if final else float("nan"),
        "auc_return": float(np.mean([r for _, r, _ in episodes])) if episodes else float("nan"),
        "wall_s": time.perf_counter() - t0, "git_commit": commit,
        "torch": torch.__version__, "gymnasium": gym.__version__,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.env_id}_{cfg.tag}_s{cfg.seed}"
    (out / f"{name}.json").write_text(json.dumps(summary))
    return summary
