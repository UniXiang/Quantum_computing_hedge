#!/usr/bin/env bash
set -euo pipefail
cd /workspace/quantum/quantum_hedge
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export PYTHONPATH=src
python3 src/run_mixed_qaoa.py \
  --instance results/mixed_long_n24/mixed_long_n24_instance.npz \
  --context results/mixed_long_n24/mixed_long_n24_context.json \
  --output results/mixed_long_n24/mixed_long_n24_result.json \
  --device biren \
  --layers 1 \
  --max-iter 20 \
  --seed 42 \
  --top-k 4096
