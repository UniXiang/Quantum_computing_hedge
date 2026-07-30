#!/usr/bin/env bash
set -eo pipefail
cd /workspace/quantum/quantum_hedge
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export PYTHONPATH=src
python3 src/run_design_long_qaoa.py \
  --instance results/design_long_n24/design_long_n24_instance.npz \
  --context results/design_long_n24/design_long_n24_context.json \
  --output results/design_long_n24/design_long_n24_result.json \
  --device biren \
  --layers 2 \
  --init interp \
  --max-iter 15 \
  --seed 42 \
  --top-k 4096
