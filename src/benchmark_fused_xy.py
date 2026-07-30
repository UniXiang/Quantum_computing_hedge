"""Benchmark sequential versus fused-seed-batch XY QAOA optimization."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cardinality_qaoa import (
    CardinalityQAOA,
    ising_energies_for_states,
)


def snapshot() -> str:
    return subprocess.run(
        ["brsmi"], capture_output=True, text=True, check=False
    ).stdout


def optimize_sequential(
    solver, initial, energies, iterations
) -> tuple[torch.Tensor, float]:
    outputs = []
    start = time.perf_counter()
    for row in initial:
        params = row.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=0.05)
        for _ in range(iterations):
            optimizer.zero_grad()
            psi = solver._evolve(params, energies)
            probabilities = psi.real.square() + psi.imag.square()
            loss = torch.sum(probabilities * energies)
            loss.backward()
            optimizer.step()
        outputs.append(params.detach())
    float(torch.sum(torch.stack(outputs)).cpu())
    return torch.stack(outputs), time.perf_counter() - start


def optimize_batch(
    solver, initial, energies, iterations
) -> tuple[torch.Tensor, float]:
    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=0.05)
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        psi = solver._evolve_batch(params, energies)
        probabilities = psi.real.square() + psi.imag.square()
        losses = torch.sum(probabilities * energies[None, :], dim=1)
        losses.sum().backward()
        optimizer.step()
    float(torch.sum(params).detach().cpu())
    return params.detach(), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--seeds", default="11,42,73")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    context = json.loads(args.context.read_text(encoding="utf-8"))
    target = int(context["config"]["objective"]["target_holdings"])
    solver = CardinalityQAOA(len(h), target, device=args.device)
    energy_np = ising_energies_for_states(h, J, solver.states)
    energies = torch.as_tensor(
        energy_np, dtype=torch.float32, device=solver.device
    )
    seeds = [int(value) for value in args.seeds.split(",")]
    initial_np = np.stack(
        [
            np.random.default_rng(seed).uniform(0.0, 0.5, 2 * args.p)
            for seed in seeds
        ]
    )
    initial = torch.as_tensor(
        initial_np, dtype=torch.float32, device=solver.device
    )
    # Warm-up excludes one-time device initialization from both timings.
    with torch.no_grad():
        solver._evolve_batch(initial[:, :2], energies)
    float(torch.sum(initial).cpu())

    before = snapshot()
    sequential_params, sequential_seconds = optimize_sequential(
        solver, initial, energies, args.iterations
    )
    middle = snapshot()
    batch_params, batch_seconds = optimize_batch(
        solver, initial, energies, args.iterations
    )
    after = snapshot()
    max_difference = float(
        torch.max(torch.abs(sequential_params - batch_params)).cpu()
    )
    payload = {
        "implementation": (
            "seed-batched XY edge schedule; one edge launch handles all seeds"
        ),
        "p": args.p,
        "seeds": seeds,
        "iterations": args.iterations,
        "sequential_seconds": sequential_seconds,
        "fused_batch_seconds": batch_seconds,
        "speedup": sequential_seconds / batch_seconds,
        "parameter_max_abs_difference": max_difference,
        "same_initialization_and_optimizer": True,
        "gpu_before": before,
        "gpu_after_sequential": middle,
        "gpu_after_fused_batch": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
