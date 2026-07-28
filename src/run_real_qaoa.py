"""Run p=1 QAOA on an exported real n=24 Hamiltonian.

The input NPZ is produced locally from the audited financial pipeline so
the Biren host only needs the quantum kernel, not the private market cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from ising_qaoa import IsingQAOA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
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
    qaoa = IsingQAOA(algo_dir=str(args.output.parent / "real_qaoa"))

    start = time.perf_counter()
    energy_vector = qaoa._energy_vector(h, J)
    energy_vector_seconds = time.perf_counter() - start
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
    bitstring = result["best_bitstring"]
    selected = [int(bit) for bit in bitstring]
    exact_qubo = float(result["exact_energy"] + offset)
    sampled_qubo = float(result["best_energy"] + offset)
    payload = {
        "host": platform.node(),
        "n": len(h),
        "layers": args.layers,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "device": args.device,
        "dtype": result["dtype"],
        "adjoint": result["adjoint"],
        "checkpoint": result["checkpoint"],
        "top_k": args.top_k,
        "energy_vector_seconds": energy_vector_seconds,
        "qaoa_seconds": qaoa_seconds,
        "best_bitstring_lsb_first": bitstring,
        "selection": selected,
        "sampled_qubo_energy": sampled_qubo,
        "exact_qubo_energy": exact_qubo,
        "absolute_gap": sampled_qubo - exact_qubo,
        "ground_state_hit": abs(sampled_qubo - exact_qubo) <= 1e-7,
        "expectation_history": result["energy_history"],
        "top_probabilities": result["bitstrings"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
