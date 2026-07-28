#!/usr/bin/env bash
# Task 3.3 one-shot Biren n=24 experiment. Results are checkpointed after
# every completed depth so a long sweep remains recoverable if interrupted.
set -euo pipefail

REPO=/workspace/quantum/quantum_hedge
ENV_SCRIPT=/usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh

source "$ENV_SCRIPT" >/dev/null
cd "$REPO"
mkdir -p results

/usr/bin/python3 src/validate_n24_n28.py \
  --ns 24 \
  --layers 1 2 4 \
  --max-iter 20 \
  --top-k 4096 \
  --device biren \
  --dtype complex64 \
  --checkpoint \
  --adjoint \
  --init random \
  --output results/t33_n24.json
