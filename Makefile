PY ?= .venv/bin/python
export OMP_NUM_THREADS=1

.PHONY: setup data train eval bench demo test lint plots smoke

setup:
	uv venv -q -p 3.12 && uv pip install -q -e ".[bench,dev]"

data:
	@echo "no datasets: environments are simulated (Gymnasium / MuJoCo)"

# all 100 runs (80 ours + 20 Stable-Baselines3), 4 parallel workers, ~1.5 h
train:
	./scripts/run_all.sh

eval: plots

bench: train

plots:
	$(PY) scripts/make_tables.py

demo:
	$(PY) -m ppo_details.cli train --env CartPole-v1 --steps 100000

smoke:
	$(PY) -m pytest -q tests/test_ppo.py -k cartpole

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy
