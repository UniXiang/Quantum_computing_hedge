"""Run the fixed-cardinality n=24 portfolio with a feasible-subspace mixer."""
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


def _gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["brsmi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except Exception as exc:
        return f"brsmi unavailable: {exc}"


def _bits(bitstring: str) -> np.ndarray:
    return np.asarray([int(bit) for bit in bitstring], dtype=np.int64)


def _decode(bits, variables, covariance, config):
    weights, allocation = allocate_full_investment(
        bits, variables, covariance, config
    )
    assets = []
    for index, row in variables.iterrows():
        assets.append(
            {
                "qubit": int(index),
                "bit": int(bits[index]),
                "state": "long" if bits[index] else "not_held",
                "code": str(row["code"]),
                "name": str(row["name"]),
                "market": str(row["market"]),
                "sector": str(row["sector"]),
                "asset_type": str(row["asset_type"]),
                "weight": float(weights[index]),
                "raw_sample_annual_return": float(
                    row["raw_sample_annual_return"]
                ),
                "expected_annual_return": float(
                    row["expected_annual_return"]
                ),
                "downside_deviation": float(row["downside_deviation"]),
            }
        )
    return {"allocation": allocation, "assets": assets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--init", choices=("random", "interp"), default="interp")
    args = parser.parse_args()

    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    config = context["config"]
    cardinality = int(config["objective"]["target_holdings"])

    started = datetime.now(timezone.utc)
    gpu_before = _gpu_snapshot()
    solver = CardinalityQAOA(
        len(h), cardinality, device=args.device
    )
    start = time.perf_counter()
    result = solver.solve(
        h,
        J,
        layers=args.layers,
        max_iter=args.max_iter,
        seed=args.seed,
        init=args.init,
        top_k=args.top_k,
    )
    elapsed = time.perf_counter() - start
    best_bits = _bits(result["best_bitstring"])
    decoded = _decode(best_bits, variables, covariance, config)
    best_qubo = float(result["best_energy"] + offset)
    exact_qubo = float(result["exact_feasible_energy"] + offset)
    gpu_after = _gpu_snapshot()
    payload = {
        "run": {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "device": args.device,
            "n": len(h),
            "layers": args.layers,
            "init": args.init,
            "max_iter_per_level": args.max_iter,
            "seed": args.seed,
            "qaoa_seconds": elapsed,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "model": context["meta"],
        "algorithm": {
            "mixer": "ring_xy_hamming_weight_preserving",
            "initial_state": "uniform_dicke_state_in_fixed_weight_subspace",
            "feasible_dimension": result["feasible_dimension"],
            "full_dimension": result["full_dimension"],
            "probability_norm": result["probability_norm"],
        },
        "qaoa_solution": {
            "bitstring_lsb_first": result["best_bitstring"],
            "selected_count": int(best_bits.sum()),
            "qubo_energy": best_qubo,
            "exact_feasible_qubo_energy": exact_qubo,
            "absolute_gap_to_exact_feasible": best_qubo - exact_qubo,
            "ground_state_hit": abs(best_qubo - exact_qubo) <= 1e-7,
            **decoded,
            "expectation_history": result["energy_history"],
            "top_probabilities": result["bitstrings"],
        },
        "classical_sa": {
            "qubo_energy": float(context["sa_energy"]),
            "selected_count": int(context["sa_cardinality"]),
            "absolute_gap_to_exact_feasible": (
                float(context["sa_energy"]) - exact_qubo
            ),
            "allocation": context.get("sa_allocation"),
        },
        "preselection_ranking": context["ranking"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
