"""Valid forward-only benchmark for seed-batched XY evolution on SUPA."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cardinality_qaoa import CardinalityQAOA, ising_energies_for_states


def snapshot() -> str:
    return subprocess.run(
        ["brsmi"], capture_output=True, text=True, check=False
    ).stdout


def synchronize(value: torch.Tensor) -> None:
    float(torch.sum(value.real).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    solver = CardinalityQAOA(len(h), 8, device=args.device)
    energy_np = ising_energies_for_states(h, J, solver.states)
    energies = torch.as_tensor(
        energy_np, dtype=torch.float32, device=solver.device
    )
    seeds = [11, 42, 73]
    params = torch.as_tensor(
        np.stack(
            [
                np.random.default_rng(seed).uniform(
                    0.0, 0.5, 2 * args.p
                )
                for seed in seeds
            ]
        ),
        dtype=torch.float32,
        device=solver.device,
    )
    with torch.no_grad():
        synchronize(solver._evolve_batch(params, energies))
        start = time.perf_counter()
        for _ in range(args.repeats):
            independent = torch.stack(
                [solver._evolve(row, energies) for row in params]
            )
        synchronize(independent)
        sequential_seconds = time.perf_counter() - start
        sequential_snapshot = snapshot()

        start = time.perf_counter()
        for _ in range(args.repeats):
            batched = solver._evolve_batch(params, energies)
        synchronize(batched)
        batched_seconds = time.perf_counter() - start
        batched_snapshot = snapshot()
        max_difference = float(
            torch.max(torch.abs(batched - independent)).cpu()
        )
    payload = {
        "scope": "forward_evaluation_only",
        "p": args.p,
        "seeds": seeds,
        "repeats": args.repeats,
        "sequential_seconds": sequential_seconds,
        "fused_batch_seconds": batched_seconds,
        "speedup": sequential_seconds / batched_seconds,
        "max_amplitude_abs_difference": max_difference,
        "safe_for": ["batched evaluation", "sampling", "SPSA forward pairs"],
        "not_safe_for": [
            "SUPA autograd training until custom backward is implemented"
        ],
        "observed_supa_gradient_mismatch": 128.66192626953125,
        "gpu_after_sequential": sequential_snapshot,
        "gpu_after_fused_batch": batched_snapshot,
    }
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
