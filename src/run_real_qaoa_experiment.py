"""A/B/C QAOA experiment for an exported real n=24 Hamiltonian.

A: p=1 multi-seed search with a larger on-device top-k pool.
B: interpolate the best p=1 parameters into p=2 and optimize from that
   true warm start (optionally with small deterministic jitters).
C: use the uniformly normalized Ising coefficients exported by the local
   financial pipeline. The original QUBO energy is always reported back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from ising_qaoa import IsingQAOA, bitstring_of_index, normalize_ising


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--p1-iter", type=int, default=60)
    parser.add_argument("--p1-seeds", type=int, nargs="+",
                        default=[42, 137, 271])
    parser.add_argument("--p2-iter", type=int, default=40)
    parser.add_argument("--p2-seeds", type=int, nargs="+",
                        default=[42, 137])
    parser.add_argument("--warm-jitter", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=65536)
    parser.add_argument("--stop-on-ground-hit", action="store_true",
                        help="Stop after a p=1 run reaches the exact ground state.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    instance = np.load(args.instance)
    raw_h = np.asarray(instance["h"], dtype=np.float64)
    raw_J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    if "scaled_h" in instance and "scaled_J" in instance:
        h = np.asarray(instance["scaled_h"], dtype=np.float64)
        J = np.asarray(instance["scaled_J"], dtype=np.float64)
        scale = float(instance["hamiltonian_scale"])
    else:
        h, J, scale = normalize_ising(raw_h, raw_J)

    qaoa = IsingQAOA(algo_dir=str(args.output.parent / "real_qaoa"))
    start = time.perf_counter()
    raw_e_vec = qaoa._energy_vector(raw_h, raw_J)
    energy_vector_seconds = time.perf_counter() - start
    # Uniform Hamiltonian scaling means this is exactly the scaled energy
    # vector; no second 2^24 enumeration is required.
    energy_vector = raw_e_vec / scale
    exact_index = int(np.argmin(raw_e_vec))
    ground_bitstring = bitstring_of_index(exact_index, len(raw_h))
    exact_qubo = float(raw_e_vec[exact_index] + offset)

    def run(label: str, layers: int, max_iter: int, seed: int,
            initial_params: np.ndarray | None = None) -> dict:
        started = time.perf_counter()
        result = qaoa.solve(
            h, J, layers=layers, device=args.device, optimizer="autograd",
            max_iter=max_iter, seed=seed, init="random", dtype="complex64",
            checkpoint=True, adjoint=True, final_backend="native",
            energy_vector=energy_vector, top_k=args.top_k,
            initial_params=initial_params,
        )
        bitstring = result["best_bitstring"]
        sampled_qubo = float(result["best_energy"] * scale + offset)
        top = result["bitstrings"]
        return {
            "label": label,
            "layers": layers,
            "max_iter": max_iter,
            "seed": seed,
            "wall_seconds": time.perf_counter() - started,
            "best_bitstring_lsb_first": bitstring,
            "selection": [int(bit) for bit in bitstring],
            "sampled_qubo_energy": sampled_qubo,
            "absolute_gap": sampled_qubo - exact_qubo,
            "ground_state_hit": abs(sampled_qubo - exact_qubo) <= 1e-7,
            "best_probability": float(top[bitstring]),
            "ground_state_in_top_k": ground_bitstring in top,
            "ground_state_probability": float(top.get(ground_bitstring, 0.0)),
            "minimum_expectation_scaled": float(min(result["energy_history"])),
            "best_params": result["best_params"].tolist(),
        }

    payload = {
        "host": platform.node(),
        "n": len(raw_h),
        "device": args.device,
        "dtype": "complex64",
        "adjoint": True,
        "checkpoint": True,
        "top_k": args.top_k,
        "hamiltonian_normalization": {
            "method": "max_abs",
            "positive_scale": scale,
            "raw_max_abs_h": float(np.max(np.abs(raw_h))),
            "raw_max_abs_J": float(np.max(np.abs(raw_J))),
        },
        "energy_vector_seconds": energy_vector_seconds,
        "exact_qubo_energy": exact_qubo,
        "ground_bitstring_lsb_first": ground_bitstring,
        "p1_runs": [],
        "p2_runs": [],
        "status": "running",
    }

    def checkpoint() -> None:
        completed = payload["p1_runs"] + payload["p2_runs"]
        if completed:
            payload["best_so_far"] = min(
                completed, key=lambda record: record["sampled_qubo_energy"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")

    for seed in args.p1_seeds:
        print(f"\n--- p=1 random seed={seed} ---", flush=True)
        record = run("p1_random", 1, args.p1_iter, seed)
        payload["p1_runs"].append(record)
        checkpoint()
        if args.stop_on_ground_hit and record["ground_state_hit"]:
            payload["best_run"] = record
            payload["status"] = "complete_early_ground_hit"
            checkpoint()
            print("p=1 reached exact ground state; stopping by request.",
                  flush=True)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return
    p1_runs = payload["p1_runs"]
    best_p1 = min(p1_runs, key=lambda record: record["sampled_qubo_energy"])
    warm_params = IsingQAOA._interp_extend(
        np.asarray(best_p1["best_params"], dtype=np.float64))
    for run_index, seed in enumerate(args.p2_seeds):
        # First p=2 trajectory is the exact INTERP continuation. Later
        # seeds probe a small, reproducible neighbourhood of it.
        if run_index == 0:
            initial = warm_params.copy()
        else:
            rng = np.random.default_rng(seed)
            initial = warm_params + rng.normal(
                0.0, args.warm_jitter, size=warm_params.shape)
        print(f"\n--- p=2 warm seed={seed} ---", flush=True)
        payload["p2_runs"].append(
            run("p2_warm", 2, args.p2_iter, seed, initial))
        checkpoint()

    p2_runs = payload["p2_runs"]
    all_runs = p1_runs + p2_runs
    best = min(all_runs, key=lambda record: record["sampled_qubo_energy"])
    payload["best_run"] = best
    payload["status"] = "complete"
    checkpoint()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
