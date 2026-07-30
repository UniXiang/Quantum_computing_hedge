"""Validate batched XY evolution against independent n=24 evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cardinality_qaoa import CardinalityQAOA, ising_energies_for_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--p", type=int, default=4)
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
        batch = solver._evolve_batch(params, energies)
        independent = torch.stack(
            [solver._evolve(row, energies) for row in params]
        )
        amplitude_difference = torch.max(
            torch.abs(batch - independent)
        )
        batch_probabilities = batch.real.square() + batch.imag.square()
        independent_probabilities = (
            independent.real.square() + independent.imag.square()
        )
        objective_difference = torch.max(
            torch.abs(
                torch.sum(batch_probabilities * energies[None, :], dim=1)
                - torch.sum(
                    independent_probabilities * energies[None, :], dim=1
                )
            )
        )
        norm_error = torch.max(
            torch.abs(torch.sum(batch_probabilities, dim=1) - 1.0)
        )
    batch_params = params.clone().requires_grad_(True)
    independent_params = params.clone().requires_grad_(True)
    batch_state = solver._evolve_batch(batch_params, energies)
    batch_loss = torch.sum(
        (batch_state.real.square() + batch_state.imag.square())
        * energies[None, :]
    )
    batch_loss.backward()
    independent_loss = 0.0
    for row in independent_params:
        state = solver._evolve(row, energies)
        independent_loss = independent_loss + torch.sum(
            (state.real.square() + state.imag.square()) * energies
        )
    independent_loss.backward()
    gradient_difference = torch.max(
        torch.abs(batch_params.grad - independent_params.grad)
    )
    payload = {
        "p": args.p,
        "seeds": seeds,
        "max_amplitude_abs_difference": float(
            amplitude_difference.cpu()
        ),
        "max_objective_abs_difference": float(
            objective_difference.cpu()
        ),
        "max_probability_norm_error": float(norm_error.cpu()),
        "max_gradient_abs_difference": float(
            gradient_difference.cpu()
        ),
    }
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
