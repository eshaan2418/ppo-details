#!/bin/bash
# Run every job with 4 parallel single-thread workers, each under a 45-minute timeout.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/jobs.py | xargs -P 4 -I{} sh -c 'gtimeout 2700 nice -n 10 .venv/bin/python {} 2>/dev/null || echo "FAILED {}"'
echo ALLDONE
