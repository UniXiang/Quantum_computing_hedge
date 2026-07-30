"""Run fixed-K SA-warm-start QAOA and verify against exact enumeration."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cardinality_qaoa import CardinalityQAOA
from design_long_n24 import allocate_full_investment
from ising_qaoa import bitstring_of_index
from same_budget_compare import timed_fixed_cardinality_sa


def snapshot() -> str:
    return subprocess.run(
        ["brsmi"], capture_output=True, text=True, check=False
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--sa-seconds", type=float, default=10.0)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-strength", type=float, default=8.0)
    parser.add_argument("--angle-scale", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()
    instance = np.load(args.instance)
    Q = np.asarray(instance["Q"], dtype=np.float64)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    config = context["config"]
    K = int(context["config"]["objective"]["target_holdings"])
    n = len(h)
    started = datetime.now(timezone.utc)
    gpu_before = snapshot()

    incumbent, incumbent_qubo, sa_iterations, sa_elapsed = (
        timed_fixed_cardinality_sa(
            Q, K, args.sa_seconds, args.seed, chains=4
        )
    )
    incumbent_bitstring = bitstring_of_index(
        sum(int(bit) << i for i, bit in enumerate(incumbent)), n
    )
    construction_start = time.perf_counter()
    solver = CardinalityQAOA(n, K, device=args.device)
    construction_elapsed = time.perf_counter() - construction_start
    qaoa_start = time.perf_counter()
    result = solver.solve(
        h,
        J,
        layers=args.p,
        max_iter=args.max_iter,
        seed=args.seed,
        init="random",
        top_k=args.top_k,
        restarts=1,
        objective="expectation",
        warm_start_bitstring=incumbent_bitstring,
        warm_start_strength=args.warm_strength,
        initial_angle_scale=args.angle_scale,
        reference_bitstring=incumbent_bitstring,
    )
    qaoa_elapsed = time.perf_counter() - qaoa_start
    exact_qubo = result["exact_feasible_energy"] + offset
    qaoa_qubo = result["best_energy"] + offset
    exact_probability = result["exact_feasible_probability"]
    qaoa_bits = np.asarray(
        [int(bit) for bit in result["best_bitstring"]], dtype=np.int64
    )
    exact_bits = np.asarray(
        [int(bit) for bit in result["exact_feasible_bitstring"]],
        dtype=np.int64,
    )
    _, qaoa_allocation = allocate_full_investment(
        qaoa_bits, variables, covariance, config
    )
    _, exact_allocation = allocate_full_investment(
        exact_bits, variables, covariance, config
    )
    rng = np.random.default_rng(args.seed)
    shot_counts = [1_000, 10_000, 100_000]
    shot_hits = {
        str(shots): int(rng.binomial(shots, exact_probability))
        for shots in shot_counts
    }
    payload = {
        "run": {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "device": args.device,
            "n": n,
            "K": K,
            "state_construction_seconds": construction_elapsed,
            "sa_seconds": sa_elapsed,
            "qaoa_seconds": qaoa_elapsed,
            "gpu_before": gpu_before,
            "gpu_after": snapshot(),
        },
        "search_space": {
            "full_dimension": result["full_dimension"],
            "fixed_weight_dimension": result["feasible_dimension"],
        },
        "warm_start": {
            "source": "fixed_cardinality_SA_without_exact_answer",
            "bitstring_lsb_first": incumbent_bitstring,
            "qubo_energy": incumbent_qubo,
            "iterations": sa_iterations,
            "equals_posthoc_exact": (
                abs(incumbent_qubo - exact_qubo) <= 1e-8
            ),
            "strength": args.warm_strength,
            "initial_probability": (
                result["warm_start_initial_probability"]
            ),
        },
        "qaoa": {
            "p": args.p,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "angle_scale": args.angle_scale,
            "best_bitstring_lsb_first": result["best_bitstring"],
            "best_qubo_energy": qaoa_qubo,
            "exact_bitstring_lsb_first": (
                result["exact_feasible_bitstring"]
            ),
            "exact_qubo_energy": exact_qubo,
            "gap": qaoa_qubo - exact_qubo,
            "exact_probability": exact_probability,
            "exact_probability_rank": (
                result["exact_feasible_probability_rank"]
            ),
            "distribution_hits_exact": exact_probability > 0.0,
            "top_candidate_equals_exact": (
                result["best_bitstring"]
                == result["exact_feasible_bitstring"]
            ),
            "simulated_shot_hits": shot_hits,
            "probability_norm": result["probability_norm"],
            "continuous_allocation": qaoa_allocation,
            "exact_continuous_allocation": exact_allocation,
        },
        "interpretation": (
            "Hybrid SA-warm-start QAOA. Exact enumeration is used only "
            "post hoc for validation, not to initialize the quantum state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
