#!/usr/bin/env bash
set -eo pipefail

PROJECT=/workspace/quantum/quantum_hedge
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
set -u
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"

for k in 4 5 6 7 8 9; do
  echo "START_K=$k"
  python3 "$PROJECT/src/prepare_fixed_k_scale.py" \
    --config "$PROJECT/configs/design_long_n24.yaml" \
    --n 32 --K "$k" \
    --output-dir "$PROJECT/results/variable_k/n32/k$k"
  python3 "$PROJECT/src/run_fixed_k_warm_qaoa.py" \
    --instance "$PROJECT/results/variable_k/n32/k$k/instance.npz" \
    --context "$PROJECT/results/variable_k/n32/k$k/context.json" \
    --output "$PROJECT/results/variable_k/n32/k$k/result.json" \
    --device biren --sa-seconds 10 --p 1 --max-iter 3 \
    --warm-strength 8 --angle-scale 0.02
  echo "DONE_K=$k"
done

python3 "$PROJECT/src/summarize_variable_k.py" \
  --root "$PROJECT/results/variable_k/n32" \
  --k-min 4 --k-max 9 \
  --output "$PROJECT/results/variable_k/n32/summary.json"
