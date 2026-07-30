"""Select the best K by the executable continuous portfolio objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--k-min", type=int, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for K in range(args.k_min, args.k_max + 1):
        payload = json.loads(
            (args.root / f"k{K}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append({
            "K": K,
            "feasible_dimension": (
                payload["search_space"]["fixed_weight_dimension"]
            ),
            "qaoa_seconds": payload["run"]["qaoa_seconds"],
            "qaoa_gap": payload["qaoa"]["gap"],
            "qaoa_equals_exact": (
                payload["qaoa"]["top_candidate_equals_exact"]
            ),
            "exact_probability": (
                payload["qaoa"]["exact_probability"]
            ),
            "continuous_objective": (
                payload["qaoa"]["continuous_allocation"]["objective"]
            ),
            "bitstring_lsb_first": (
                payload["qaoa"]["best_bitstring_lsb_first"]
            ),
        })
    best = min(rows, key=lambda row: row["continuous_objective"])
    payload = {
        "method": (
            "outer discrete K search with SA-warm-start fixed-K QAOA; "
            "selection by continuous full-investment objective"
        ),
        "n": 32,
        "K_range": [args.k_min, args.k_max],
        "best_K": best["K"],
        "best": best,
        "best_at_search_boundary": best["K"] in (
            args.k_min, args.k_max
        ),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
