"""Run n=24 QAOA and emit the exact-optimum long/cash portfolio JSON."""
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

from ising_qaoa import IsingQAOA
from real_portfolio import allocate_selected_weights


def _gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["brsmi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except Exception as exc:
        return f"br-smi unavailable: {exc}"


def _portfolio(
    bits: np.ndarray,
    variables: pd.DataFrame,
    covariance: np.ndarray,
    config: dict,
) -> tuple[list[dict], dict]:
    weights, diagnostics = allocate_selected_weights(
        bits, variables, covariance, config
    )
    rows = []
    for i, row in variables.iterrows():
        rows.append(
            {
                "qubit": int(i),
                "bit": int(bits[i]),
                "state": "long" if bits[i] else "cash",
                "code": str(row["underlying"]),
                "name": str(row["name"]),
                "market": str(row["market"]),
                "asset_type": str(row["asset_type"]),
                "sector": str(row["sector"]),
                "weight": float(weights[i]),
                "beta": float(row["beta"]),
                "factor_score": float(row["factor_score"]),
                "expected_annual_return_proxy": float(
                    row["expected_annual_return_proxy"]
                ),
            }
        )
    return rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=4096)
    args = parser.parse_args()

    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    config = context["config"]
    gpu_before = _gpu_snapshot()

    qaoa = IsingQAOA(algo_dir=str(args.output.parent / "mixed_qaoa"))
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    energy_vector = qaoa._energy_vector(h, J)
    energy_vector_seconds = time.perf_counter() - start
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
        init="random",
        dtype="complex64",
        checkpoint=True,
        adjoint=True,
        final_backend="native",
        energy_vector=energy_vector,
        top_k=args.top_k,
    )
    qaoa_seconds = time.perf_counter() - start
    sampled_bits = np.asarray(
        [int(bit) for bit in result["best_bitstring"]], dtype=np.int64
    )
    exact_rows, exact_allocation = _portfolio(
        exact_bits, variables, covariance, config
    )
    sampled_rows, sampled_allocation = _portfolio(
        sampled_bits, variables, covariance, config
    )
    exact_qubo = float(energy_vector[exact_index] + offset)
    sampled_qubo = float(result["best_energy"] + offset)
    sa_energy = float(context["sa_energy"])
    gpu_after = _gpu_snapshot()
    payload = {
        "run": {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "device": args.device,
            "n": len(h),
            "layers": args.layers,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "dtype": result["dtype"],
            "adjoint": result["adjoint"],
            "checkpoint": result["checkpoint"],
            "top_k": args.top_k,
            "energy_vector_seconds": energy_vector_seconds,
            "qaoa_seconds": qaoa_seconds,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "model": {
            **context["meta"],
            "short_selling": False,
            "bit_semantics": {"0": "cash", "1": "long"},
            "beta_target": config["objective"]["beta_target"],
            "min_net_exposure": config["allocation"]["min_net_exposure"],
            "max_net_exposure": config["allocation"]["max_net_exposure"],
        },
        "optimal_solution": {
            "source": "exact_ground_state_enumeration_validated_in_gpu_run",
            "bitstring_lsb_first": "".join(str(x) for x in exact_bits),
            "qubo_energy": exact_qubo,
            "selected_count": int(exact_bits.sum()),
            "allocation": exact_allocation,
            "assets": exact_rows,
        },
        "qaoa_solution": {
            "bitstring_lsb_first": result["best_bitstring"],
            "qubo_energy": sampled_qubo,
            "absolute_gap_to_optimum": sampled_qubo - exact_qubo,
            "ground_state_hit": abs(sampled_qubo - exact_qubo) <= 1e-7,
            "selected_count": int(sampled_bits.sum()),
            "allocation": sampled_allocation,
            "assets": sampled_rows,
            "expectation_history": result["energy_history"],
            "top_probabilities": result["bitstrings"],
        },
        "classical_sa": {
            "qubo_energy": sa_energy,
            "absolute_gap_to_optimum": sa_energy - exact_qubo,
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
