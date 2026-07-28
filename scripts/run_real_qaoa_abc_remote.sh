#!/usr/bin/env bash
# A/B/C experiment for the mixed A-share + US-stock + XAU real n=24 QUBO.
# Results are one JSON file written only after all p=1/p=2 runs finish;
# nohup log remains available for progress monitoring.
# The vendor environment script reads optional variables that may be unset,
# so do not enable nounset before sourcing it.
set -eo pipefail

REPO=/workspace/quantum/quantum_hedge
ENV_SCRIPT=/usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh

source "$ENV_SCRIPT" >/dev/null
cd "$REPO"
mkdir -p results

PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  src/run_real_qaoa_experiment.py \
  --instance results/real_n24_us_instance.npz \
  --output results/real_n24_us_qaoa_abc.json \
  --device biren \
  --p1-iter 60 \
  --p1-seeds 42 137 271 \
  --p2-iter 40 \
  --p2-seeds 42 137 \
  --warm-jitter 0.03 \
  --top-k 65536 \
  --stop-on-ground-hit
