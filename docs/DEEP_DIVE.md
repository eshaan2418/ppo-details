# Deep dive: questions and answers

### 1. Why trust these ablation numbers more than a single-run blog post?
Each cell has 5 independent seeds, the CIs are t-intervals, and differences use Welch's interval. Runs with the same seed diverge after the first update, so they are treated as independent. p-values are Holm-corrected across the 7 Hopper ablations. The base implementation is also checked against Stable-Baselines3 with matched settings, and is statistically indistinguishable on Hopper (Δ −40 ± 399), HalfCheetah, and Acrobot. Without that check, an ablation could be measuring a bug rather than a design choice.

### 2. Why does reward scaling matter so much on Hopper?
Removing it drops the final return from 2537 to 500, consistently across all seeds. Hopper rewards ~1–3 per step over episodes of up to 1,000 steps give discounted returns in the hundreds. Unscaled value targets that large make the critic's squared error, and so its gradient, much larger than the policy gradient. Because both losses share one global gradient-norm clip and one Adam optimiser, the policy update is crowded out. Scaling rewards by the running std of the *discounted return* keeps targets O(1). This mechanism is the likely explanation. It is not directly measured, since per-term gradient norms were not logged.

### 3. Why divide by the std of the return rather than the reward?
The quantity that sets the scale of the value targets is the return, not the per-step reward. A per-step reward std of 1 can give returns of 100 with γ=0.99. The running return accumulator is reset at episode boundaries so that scaling tracks the episodic return distribution.

### 4. Several details had point estimates of −300 to −500 on Hopper but were "not significant". Do they not matter?
Absence of evidence is not evidence of absence. Per-arm std on Hopper is ~290, so with 5 seeds the 95% interval for a difference is about ±450 wide. Orthogonal init (−490, p=0.046 uncorrected) and LR annealing (−460, p=0.054) are suggestive. About 15 seeds per arm would be needed for 80% power to detect a 300-point effect. On the AUC measure, orthogonal init's effect (−173 ± 112) does exclude zero, meaning learning is slower even where final returns recover.

### 5. Why did nothing matter on CartPole?
CartPole is easy (return capped at 500), so the full method nearly saturates and there is little headroom. Its default preset also leaves out observation and reward normalisation. The value-clipping ablation's point estimate is −111 ± 142, the largest effect, but its CI includes zero.

### 6. CartPole's clip fraction is ~0. Is the clip doing anything?
With lr 2.5e-4, 4 epochs and 4 minibatches, the policy changes so little per update that the probability ratio almost never leaves [0.8, 1.2]: approx. KL ~1e-4. I confirmed SB3 shows the same with identical settings (clip fraction 0.00, KL ~3e-4), so it is a property of the hyperparameters, not a bug. On Hopper (10 epochs × 32 minibatches) the clip fraction is ~0.12 early in training, and there the clip is active.

### 7. Why is our CartPole result better than SB3's?
SB3's runs reach ~400 by 300k steps and then regress. Ours keep rising to 474. The candidate explanation is truncation handling: SB3 bootstraps the value at time-limit truncations, while this implementation treats truncation as termination. I have not run the controlled test (toggle bootstrapping in ours), so this is a hypothesis.

### 8. How is GAE verified, given how easy it is to get the done flags wrong?
`done_buf[t]` means "the observation at step t begins a new episode". The non-terminal mask for transition t is therefore `1 − done_buf[t+1]` (and the final `done` for the last step). A Hypothesis test generates random rewards, values, and episode boundaries for several environments, and compares against an explicit per-environment backward recursion written in terms of "episode ended after step t". A second test checks that λ=1 gives the discounted return minus the value.

### 9. What went wrong during development?
Gymnasium 1.x with same-step autoreset reports finished episodes under `info["final_info"]`. The first version read `info["episode"]`, recorded no episodes, and returned NaN final returns. The smoke run caught it immediately.

### 10. Why one-at-a-time ablations rather than a factorial design?
A full factorial over 7 binary details is 128 configurations, or 640 runs at 5 seeds, which is about 8× the budget. One-at-a-time answers "what do I lose if I drop this from a working configuration", which is the practitioner's question. It misses interactions (e.g. observation normalisation might partly substitute for orthogonal init).

### 11. Why measure "final return" from training episodes?
It avoids a separate evaluation environment whose observation normaliser would have to be frozen and copied, and it matches how the reference implementations report results. The last 10% of training covers hundreds of episodes on CartPole and tens to hundreds on MuJoCo. It reflects the stochastic policy, which slightly understates deterministic performance.

### 12. How was the compute spent?
100 runs: 80 of this implementation (20 full + 60 ablation) and 20 SB3. Each is single-threaded. MuJoCo runs take 3.5–9 minutes. 4 run concurrently, for 329 CPU-minutes, about 1.5 h of wall time. Small MLPs are faster on CPU than GPU, so parallelising across runs is the efficient choice.

### 13. With 100x compute?
20+ seeds per arm, to resolve the 300–500 point effects. A factorial or Shapley-style attribution over the details. More environments (Walker2d, Ant, Atari via MinAtar-like suites). A hyperparameter sweep crossed with the ablations, to test whether "unimportant" details become important at other learning rates. And the truncation-bootstrapping test from Q7.
