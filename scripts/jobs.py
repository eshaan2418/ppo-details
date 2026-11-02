"""Generate the experiment job list (one command per line) for scripts/run_all.sh."""

ENVS = ["CartPole-v1", "Acrobot-v1", "Hopper-v5", "HalfCheetah-v5"]
SEEDS = range(5)
DISCRETE_ABL = ["adv_norm", "value_clip", "ortho_init", "lr_anneal", "grad_clip"]
MUJOCO_ABL = DISCRETE_ABL + ["obs_norm", "reward_norm"]

jobs = []
for env in ENVS:
    for s in SEEDS:
        jobs.append(f"scripts/run.py --env {env} --seed {s}")
        jobs.append(f"scripts/sb3_baseline.py --env {env} --seed {s}")
for s in SEEDS:
    for ab in MUJOCO_ABL:
        jobs.append(f"scripts/run.py --env Hopper-v5 --seed {s} --ablate {ab}")
    for ab in DISCRETE_ABL:
        jobs.append(f"scripts/run.py --env CartPole-v1 --seed {s} --ablate {ab}")
# longest jobs first for better packing
jobs.sort(key=lambda j: ("Cheetah" not in j and "Hopper" not in j, "sb3" not in j))
print("\n".join(jobs))
