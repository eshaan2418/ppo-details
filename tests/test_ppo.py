import gymnasium as gym
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ppo_details.ppo import Agent, PPOConfig, RunningMeanStd, gae, train


def gae_reference(rewards, values, ends_after, last_value, gamma, lam):
    """Per-env backward recursion with explicit 'episode ended after step t' flags."""
    T, N = rewards.shape
    adv = np.zeros((T, N))
    for n in range(N):
        a = 0.0
        for t in reversed(range(T)):
            next_v = last_value[n] if t == T - 1 else values[t + 1, n]
            nonterm = 0.0 if ends_after[t, n] else 1.0
            delta = rewards[t, n] + gamma * next_v * nonterm - values[t, n]
            a = delta + gamma * lam * nonterm * a
            adv[t, n] = a
    return adv


@settings(max_examples=50, deadline=None)
@given(st.integers(1, 12), st.integers(1, 3), st.integers(0, 10**6))
def test_gae_matches_reference(T, N, seed):
    rng = np.random.default_rng(seed)
    r, v = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    ends = rng.random((T, N)) < 0.2
    last_v = rng.normal(size=N)
    dones = np.zeros((T, N))
    dones[1:] = ends[:-1]  # done_buf[t] = episode ended after step t-1
    adv, ret = gae(r, v, dones, last_v, ends[-1].astype(float), 0.99, 0.95)
    ref = gae_reference(r, v, ends, last_v, 0.99, 0.95)
    np.testing.assert_allclose(adv, ref, atol=1e-10)
    np.testing.assert_allclose(ret, ref + v, atol=1e-10)


def test_gae_lambda1_is_discounted_return_minus_value():
    r = np.ones((5, 1))
    v = np.zeros((5, 1))
    adv, _ = gae(r, v, np.zeros((5, 1)), np.zeros(1), np.ones(1), 0.5, 1.0)
    expected = [sum(0.5 ** k for k in range(5 - t)) for t in range(5)]
    np.testing.assert_allclose(adv[:, 0], expected)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3, 2, (1000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.array_split(data, 17):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(0), atol=1e-3)
    np.testing.assert_allclose(rms.var, data.var(0), rtol=1e-3)


def test_clipped_objective_has_zero_gradient_outside_trust_region():
    adv = torch.tensor([1.0, -1.0])
    logp_old = torch.zeros(2)
    new = torch.tensor([0.5, -0.5], requires_grad=True)  # ratio e^0.5 > 1.2, e^-0.5 < 0.8
    ratio = (new - logp_old).exp()
    loss = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
    loss.backward()
    assert torch.allclose(new.grad, torch.zeros(2))


def test_agent_heads_and_init_switch():
    env = gym.make("Pendulum-v1")
    for ortho in (True, False):
        a = Agent(3, env.action_space, PPOConfig(ortho_init=ortho))
        x = torch.randn(5, 3)
        d = a.dist(x)
        assert d.sample().shape == (5, 1) and a.value(x).shape == (5,)
    a = Agent(3, env.action_space, PPOConfig(ortho_init=True))
    w = a.actor[0].weight
    assert torch.allclose(w @ w.T, 2 * torch.eye(w.shape[0])[: w.shape[0]], atol=1e-4) or \
        torch.allclose(w.T @ w, 2 * torch.eye(w.shape[1]), atol=1e-4)
    disc = Agent(4, gym.make("CartPole-v1").action_space, PPOConfig())
    assert disc.dist(torch.randn(2, 4)).sample().shape == (2,)


def test_cartpole_learns(tmp_path):
    """Smoke train: PPO must clearly beat a random policy (~22) within 60k steps."""
    s = train(PPOConfig(env_id="CartPole-v1", seed=0, total_steps=60_000), tmp_path)
    assert s["final_return"] > 100, s["final_return"]
