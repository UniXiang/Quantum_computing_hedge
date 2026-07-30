"""Run the document-aligned long-only n=24 problem on Biren QAOA."""
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

from design_long_n24 import allocate_full_investment
from ising_qaoa import IsingQAOA


def _gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["brsmi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except Exception as exc:
        return f"brsmi unavailable: {exc}"


def _decode(
    bits: np.ndarray,
    variables: pd.DataFrame,
    covariance: np.ndarray,
    config: dict,
) -> dict:
    target = int(config["objective"]["target_holdings"])
    allocation = None
    weights = np.zeros(len(bits), dtype=np.float64)
    allocation_error = None
    if int(bits.sum()) == target:
        try:
            weights, allocation = allocate_full_investment(
                bits, variables, covariance, config
            )
        except Exception as exc:
            allocation_error = str(exc)
    else:
        allocation_error = (
            f"cardinality {int(bits.sum())} does not equal target {target}"
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
    return {
        "bitstring_lsb_first": "".join(str(int(bit)) for bit in bits),
        "selected_count": int(bits.sum()),
        "allocation": allocation,
        "allocation_error": allocation_error,
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=15)
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
    started = datetime.now(timezone.utc)
    gpu_before = _gpu_snapshot()

    qaoa = IsingQAOA(algo_dir=str(args.output.parent / "design_qaoa"))
    start = time.perf_counter()
    energy_vector = qaoa._energy_vector(h, J)
    exact_seconds = time.perf_counter() - start
    exact_index = int(np.argmin(energy_vector))
    exact_bits = (
        (exact_index >> np.arange(len(h), dtype=np.int64)) & 1
    ).astype(np.int64)

    start = time.perf_counter()
    result = qaoa.solve(
        h,
        J,
        layers=args.layers,
        device=args.device,
        optimizer="autograd",
        max_iter=args.max_iter,
        seed=args.seed,
        init=args.init,
        dtype="complex64",
        checkpoint=True,
        adjoint=True,
        final_backend="native",
        energy_vector=energy_vector,
        top_k=args.top_k,
    )
    qaoa_seconds = time.perf_counter() - start
    qaoa_bits = np.asarray(
        [int(bit) for bit in result["best_bitstring"]], dtype=np.int64
    )
    exact_qubo = float(energy_vector[exact_index] + offset)
    qaoa_qubo = float(result["best_energy"] + offset)
    sa_qubo = float(context["sa_energy"])
    exact_decoded = _decode(exact_bits, variables, covariance, config)
    qaoa_decoded = _decode(qaoa_bits, variables, covariance, config)
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
            "dtype": result["dtype"],
            "adjoint": result["adjoint"],
            "checkpoint": result["checkpoint"],
            "top_k": args.top_k,
            "classical_exact_enumeration_seconds": exact_seconds,
            "qaoa_seconds": qaoa_seconds,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "model": context["meta"],
        "exact_qubo_solution": {
            **exact_decoded,
            "qubo_energy": exact_qubo,
            "source": "classical_enumeration_of_2^24_states",
        },
        "qaoa_solution": {
            **qaoa_decoded,
            "qubo_energy": qaoa_qubo,
            "absolute_gap_to_exact": qaoa_qubo - exact_qubo,
            "ground_state_hit": abs(qaoa_qubo - exact_qubo) <= 1e-7,
            "expectation_history": result["energy_history"],
            "top_probabilities": result["bitstrings"],
        },
        "classical_sa": {
            "qubo_energy": sa_qubo,
            "selected_count": int(context["sa_cardinality"]),
            "absolute_gap_to_exact": sa_qubo - exact_qubo,
            "allocation": context.get("sa_allocation"),
        },
        "preselection_ranking": context["ranking"],
        "interpretation_warning": (
            "The exact state is the global optimum of the equal-weight "
            "selection QUBO, not a proof that the subsequent continuous "
            "portfolio is globally optimal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
