"""Add exact feasible and SA portfolio mappings to an existing GPU JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardinality_qaoa import (
    fixed_weight_states,
    ising_energies_for_states,
)
from ising_qaoa import bitstring_of_index
from run_design_cardinality_qaoa import _decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    config = context["config"]
    cardinality = int(config["objective"]["target_holdings"])

    states = fixed_weight_states(len(h), cardinality)
    energies = ising_energies_for_states(h, J, states)
    best_position = int(np.argmin(energies))
    exact_state = int(states[best_position])
    exact_bitstring = bitstring_of_index(exact_state, len(h))
    exact_bits = np.asarray(
        [int(bit) for bit in exact_bitstring], dtype=np.int64
    )
    exact_qubo = float(energies[best_position] + offset)
    reported = float(
        payload["qaoa_solution"]["exact_feasible_qubo_energy"]
    )
    if abs(exact_qubo - reported) > 1e-9:
        raise RuntimeError(
            f"exact energy mismatch: recomputed {exact_qubo}, "
            f"GPU JSON reports {reported}"
        )
    payload["exact_feasible_solution"] = {
        "source": "classical_enumeration_of_C(n,K)_feasible_states",
        "bitstring_lsb_first": exact_bitstring,
        "selected_count": int(exact_bits.sum()),
        "qubo_energy": exact_qubo,
        **_decode(exact_bits, variables, covariance, config),
    }

    sa_bits = np.asarray(context["sa_selection"], dtype=np.int64)
    payload["classical_sa"].update(
        {
            "bitstring_lsb_first": "".join(str(int(bit)) for bit in sa_bits),
            **_decode(sa_bits, variables, covariance, config),
        }
    )
    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "result": str(args.result),
        "exact_bitstring": exact_bitstring,
        "exact_qubo_energy": exact_qubo,
        "sa_gap": payload["classical_sa"]["absolute_gap_to_exact_feasible"],
        "qaoa_gap": payload["qaoa_solution"][
            "absolute_gap_to_exact_feasible"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
