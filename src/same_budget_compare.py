"""Compare QAOA, fixed-cardinality SA and enumeration under one budget."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cardinality_qaoa import fixed_weight_states, ising_energies_for_states
from design_long_n24 import allocate_full_investment


def bits_from_string(value: str) -> np.ndarray:
    return np.asarray([int(bit) for bit in value], dtype=np.int64)


def bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def timed_fixed_cardinality_sa(
    Q: np.ndarray,
    cardinality: int,
    budget_seconds: float,
    seed: int,
    chains: int = 4,
) -> tuple[np.ndarray, float, int, float]:
    rng = np.random.default_rng(seed)
    n = len(Q)
    xs = np.zeros((chains, n), dtype=np.float64)
    selected = np.empty((chains, cardinality), dtype=np.int64)
    unselected = np.empty((chains, n - cardinality), dtype=np.int64)
    for chain in range(chains):
        chosen = rng.choice(n, size=cardinality, replace=False)
        xs[chain, chosen] = 1.0
        selected[chain] = np.flatnonzero(xs[chain])
        unselected[chain] = np.flatnonzero(1.0 - xs[chain])
    qxs = xs @ Q
    energies = np.einsum("bi,bi->b", xs, qxs)
    best_chain = int(np.argmin(energies))
    best_x = xs[best_chain].copy()
    best_energy = float(energies[best_chain])
    scale = float(np.mean(np.abs(Q))) or 1.0
    t_initial, t_final = 10.0 * scale, scale / 1000.0
    start = time.perf_counter()
    deadline = start + budget_seconds
    iterations = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        fraction = min((now - start) / budget_seconds, 1.0)
        temperature = t_initial * (t_final / t_initial) ** fraction
        chain = iterations % chains
        selected_position = int(rng.integers(cardinality))
        unselected_position = int(rng.integers(n - cardinality))
        i = int(selected[chain, selected_position])
        j = int(unselected[chain, unselected_position])
        # Remove selected i, then add unselected j.
        remove_delta = Q[i, i] - 2.0 * qxs[chain, i]
        qx_after_remove_j = qxs[chain, j] - Q[j, i]
        add_delta = Q[j, j] + 2.0 * qx_after_remove_j
        delta = float(remove_delta + add_delta)
        if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
            xs[chain, i] = 0.0
            xs[chain, j] = 1.0
            qxs[chain] += Q[:, j] - Q[:, i]
            energies[chain] += delta
            selected[chain, selected_position] = j
            unselected[chain, unselected_position] = i
            if energies[chain] < best_energy:
                best_energy = float(energies[chain])
                best_x = xs[chain].copy()
        iterations += 1
    return (
        best_x.astype(np.int64),
        best_energy,
        iterations,
        time.perf_counter() - start,
    )


def allocation(bits, variables, covariance, config) -> dict:
    _, summary = allocate_full_investment(
        bits, variables, covariance, config
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--qaoa-experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sa-seed", type=int, default=20260729)
    args = parser.parse_args()
    instance = np.load(args.instance)
    Q = np.asarray(instance["Q"], dtype=np.float64)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    experiment = json.loads(
        args.qaoa_experiment.read_text(encoding="utf-8")
    )
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    config = context["config"]
    cardinality = int(config["objective"]["target_holdings"])

    # Step 3 chooses the configuration. Its single-run wall time is the
    # common evaluation budget; hyperparameter search time is reported but
    # deliberately not charged to any one evaluation.
    qaoa = min(experiment["runs"], key=lambda row: row["gap"])
    budget = float(qaoa["seconds"])
    qaoa_bits = bits_from_string(qaoa["best_bitstring_lsb_first"])

    sa_bits, sa_energy, sa_iterations, sa_seconds = (
        timed_fixed_cardinality_sa(
            Q, cardinality, budget, args.sa_seed
        )
    )

    exact_start = time.perf_counter()
    states = fixed_weight_states(len(h), cardinality)
    exact_ising = ising_energies_for_states(h, J, states)
    exact_position = int(np.argmin(exact_ising))
    exact_state = int(states[exact_position])
    exact_bits = np.asarray(
        [(exact_state >> index) & 1 for index in range(len(h))],
        dtype=np.int64,
    )
    exact_energy = float(exact_ising[exact_position] + offset)
    exact_seconds = time.perf_counter() - exact_start

    rows = [
        {
            "solver": "QAOA",
            "seconds": budget,
            "within_budget": True,
            "qubo_energy": qaoa["best_qubo_energy"],
            "gap_to_exact": qaoa["best_qubo_energy"] - exact_energy,
            "bitstring_lsb_first": bitstring(qaoa_bits),
            "selected_count": int(qaoa_bits.sum()),
            "continuous_allocation": allocation(
                qaoa_bits, variables, covariance, config
            ),
            "detail": {
                "objective": qaoa["objective"],
                "p": qaoa["p"],
                "seed": qaoa["seed"],
                "top_k": 512,
                "hyperparameter_search_seconds_excluded": sum(
                    row["seconds"] for row in experiment["runs"]
                ),
            },
        },
        {
            "solver": "SA",
            "seconds": sa_seconds,
            "within_budget": sa_seconds <= budget * 1.01,
            "qubo_energy": sa_energy,
            "gap_to_exact": sa_energy - exact_energy,
            "bitstring_lsb_first": bitstring(sa_bits),
            "selected_count": int(sa_bits.sum()),
            "continuous_allocation": allocation(
                sa_bits, variables, covariance, config
            ),
            "detail": {
                "seed": args.sa_seed,
                "chains": 4,
                "iterations": sa_iterations,
                "move": "one_selected_one_unselected_swap",
            },
        },
        {
            "solver": "exact_enumeration",
            "seconds": exact_seconds,
            "within_budget": exact_seconds <= budget,
            "qubo_energy": exact_energy,
            "gap_to_exact": 0.0,
            "bitstring_lsb_first": bitstring(exact_bits),
            "selected_count": int(exact_bits.sum()),
            "continuous_allocation": allocation(
                exact_bits, variables, covariance, config
            ),
            "detail": {
                "states_enumerated": int(len(states)),
                "space": "C(24,8)",
            },
        },
    ]
    payload = {
        "common_time_budget_seconds": budget,
        "budget_definition": (
            "wall time of the Step-3 selected single QAOA evaluation"
        ),
        "selection_metric": "QUBO energy; lower is better",
        "all_solvers_fixed_cardinality": cardinality,
        "short_selling": False,
        "rows": rows,
        "interpretation_guardrail": (
            "QAOA hyperparameter-search time is excluded from its single-run "
            "evaluation budget and is disclosed separately."
        ),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
