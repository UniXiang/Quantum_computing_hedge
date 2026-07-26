"""validate_n16.py — n=16 end-to-end correctness anchor.

Builds the downside-semivariance portfolio QUBO from real bs_cache_1year
data (16 liquid SSE stocks, K=8, window=60) and compares three solver
channels:

  1. exhaustive exact enumeration (ground truth)
  2. IsingQAOA (layers=4, autograd)
  3. simulated annealing with the same wall-clock budget as QAOA

Acceptance (design doc section 8): QAOA approximation ratio >= 0.95.

Approximation-ratio definition (energies may be negative, so a plain
E_qaoa/E_exact ratio is ill-defined; design doc suggestion adopted):

    ratio = (E_qaoa - E_worst) / (E_exact - E_worst)

where E_worst is the maximum QUBO energy over all 2^16 states (from the
same exhaustive pass). ratio = 1 at the exact optimum, 0 at the worst
state. All energies are QUBO energies x'Qx (Ising energy + offset).

Writes results/n16_validation.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_returns
from qubo_builder import build_qubo, qubo_to_ising, ising_to_qubo_energy
from solvers import solve_exact, solve_sa, energy_table
from ising_qaoa import IsingQAOA

# --- instance configuration (justification in the report) ---------------
CODES = ["600000", "600004", "600016", "600028", "600030", "600036",
         "600050", "600519", "601088", "601166", "601318", "601398",
         "601601", "601857", "601988", "601288"]  # liquid SSE large caps
END_DATE = "2026-07-24"   # last trading day in the cache
WINDOW = 60
K = 8
# Return scale: daily sigma ~2% -> risk term x'Sx/K^2 ~ 5e-5; mean daily
# return ~1e-3 -> lam*mu'x/K ~ lam*1e-3. lam=0.05 puts the return term on
# the same order as the risk term. A=0.01 is ~100x the per-asset
# risk/return contribution, so cardinality violations are strictly
# dominated. gamma=0: first period (x_prev=None), no turnover penalty.
LAM = 0.05
A = 0.01
GAMMA = 0.0

QAOA_LAYERS = 4
QAOA_MAX_ITER = 200
QAOA_SEED = 42
SA_SEED = 42
RATIO_THRESHOLD = 0.95

REPORT_PATH = HERE.parent / "results" / "n16_validation.md"


def main():
    print(f"[1/4] loading returns: {len(CODES)} stocks, window={WINDOW}, "
          f"end={END_DATE}")
    returns = load_returns(CODES, END_DATE, WINDOW)

    print("[2/4] building QUBO and Ising mapping")
    Q = build_qubo(returns, K=K, lam=LAM, A=A, gamma=GAMMA, x_prev=None)
    h, J, offset = qubo_to_ising(Q)

    # --- channel 1: exhaustive exact ------------------------------------
    print("[3/4] exhaustive exact solve (2^16 states)")
    t0 = time.perf_counter()
    x_exact, e_exact = solve_exact(Q)
    t_exact = time.perf_counter() - t0
    e_table = energy_table(Q)
    e_worst = float(e_table.max())
    n_optimal = int(np.sum(np.isclose(e_table, e_exact, atol=1e-12)))

    # --- channel 2: QAOA -------------------------------------------------
    print(f"      QAOA (layers={QAOA_LAYERS}, autograd, "
          f"max_iter={QAOA_MAX_ITER})")
    qaoa = IsingQAOA()
    t0 = time.perf_counter()
    res = qaoa.solve(h, J, layers=QAOA_LAYERS, device="cpu",
                     optimizer="autograd", max_iter=QAOA_MAX_ITER,
                     seed=QAOA_SEED)
    t_qaoa = time.perf_counter() - t0
    x_qaoa = np.array([int(b) for b in res["best_bitstring"]],
                      dtype=np.float64)
    e_qaoa = ising_to_qubo_energy(x_qaoa, h, J, offset)

    # --- channel 3: SA with the same wall-clock budget -------------------
    print(f"[4/4] SA with matched budget ({t_qaoa:.1f}s)")
    x_sa, e_sa = solve_sa(Q, budget_s=t_qaoa, seed=SA_SEED)

    def ratio(e):
        return (e - e_worst) / (e_exact - e_worst)

    r_qaoa, r_sa = ratio(e_qaoa), ratio(e_sa)
    passed = r_qaoa >= RATIO_THRESHOLD

    print("\n================ results ================")
    print(f"E_worst  = {e_worst:.8f}")
    print(f"E_exact  = {e_exact:.8f}  x*={x_exact.tolist()} "
          f"({n_optimal} degenerate optima), {t_exact:.3f}s")
    print(f"E_qaoa   = {e_qaoa:.8f}  x={x_qaoa.astype(int).tolist()}, "
          f"ratio={r_qaoa:.4f}, {t_qaoa:.1f}s")
    print(f"E_sa     = {e_sa:.8f}  x={x_sa.tolist()}, "
          f"ratio={r_sa:.4f}")
    print(f"ACCEPT: ratio_qaoa {r_qaoa:.4f} >= {RATIO_THRESHOLD}: {passed}")

    hist = res["energy_history"]
    qubo_hist = [e + offset for e in hist]
    report = f"""# n=16 Validation Report (Task 2)

## Instance

- 16 liquid SSE stocks: {", ".join(CODES)}
- end_date={END_DATE}, window={WINDOW} trading days, K={K}
- QUBO: min x'Sx/K^2 - lam*mu'x/K + A*(sum x - K)^2 + gamma*|x - x_prev|
  - S = downside semivariance (X_down=min(r,0), X_down'X_down/T), Ledoit-Wolf shrunk
  - lam={LAM}, A={A}, gamma={GAMMA} (x_prev=None, first period)
- Ising offset = {offset:.10f}

## Approximation-ratio definition

ratio = (E - E_worst) / (E_exact - E_worst), all energies are QUBO
energies x'Qx. 1.0 = exact optimum, 0.0 = worst of the 2^16 states.

## Three-channel comparison

| channel | energy (x'Qx) | ratio | selected (x) | wall time |
|---|---|---|---|---|
| exact  | {e_exact:.8f} | 1.0000 | {x_exact.tolist()} | {t_exact:.3f}s |
| QAOA   | {e_qaoa:.8f} | {r_qaoa:.4f} | {x_qaoa.astype(int).tolist()} | {t_qaoa:.1f}s |
| SA (budget=QAOA time) | {e_sa:.8f} | {r_sa:.4f} | {x_sa.tolist()} | {t_qaoa:.1f}s |

E_worst = {e_worst:.8f}; degenerate exact optima: {n_optimal}

Selected by exact: {[c for c, b in zip(CODES, x_exact) if b]}
Selected by QAOA:  {[c for c, b in zip(CODES, x_qaoa.astype(int)) if b]}
Selected by SA:    {[c for c, b in zip(CODES, x_sa) if b]}

## QAOA configuration

layers={QAOA_LAYERS}, optimizer=autograd, max_iter={QAOA_MAX_ITER},
seed={QAOA_SEED}, device=cpu, restarts=3 (IsingQAOA internal default)

## QAOA convergence (QUBO energy x'Qx per optimizer step, restarts concatenated)

{json.dumps([round(e, 8) for e in qubo_hist])}

## Conclusion

QAOA approximation ratio = **{r_qaoa:.4f}** vs threshold {RATIO_THRESHOLD}
-> **{"PASS" if passed else "FAIL"}** (design doc section 8 acceptance).
SA on the same wall-clock budget: ratio {r_sa:.4f}.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nreport written to {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
