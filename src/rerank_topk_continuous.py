"""Rerank every QAOA top-K selection by its optimized continuous portfolio."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from design_long_n24 import allocate_full_investment


def _decode_assets(bits, weights, variables):
    rows = []
    for index in np.flatnonzero(bits):
        row = variables.iloc[index]
        rows.append(
            {
                "qubit": int(index),
                "code": str(row["code"]),
                "name": str(row["name"]),
                "market": str(row["market"]),
                "sector": str(row["sector"]),
                "asset_type": str(row["asset_type"]),
                "weight": float(weights[index]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    config = context["config"]
    variables = pd.DataFrame(context["variables"])
    covariance = np.asarray(context["covariance"], dtype=np.float64)
    target = int(config["objective"]["target_holdings"])
    probabilities = payload["qaoa_solution"]["top_probabilities"]

    candidates = []
    failures = []
    start = time.perf_counter()
    for probability_rank, (bitstring, probability) in enumerate(
        probabilities.items(), start=1
    ):
        bits = np.asarray([int(bit) for bit in bitstring], dtype=np.int64)
        if int(bits.sum()) != target:
            failures.append(
                {"bitstring": bitstring, "error": "wrong_cardinality"}
            )
            continue
        try:
            weights, allocation = allocate_full_investment(
                bits, variables, covariance, config
            )
        except Exception as exc:
            failures.append(
                {"bitstring": bitstring, "error": str(exc)}
            )
            continue
        candidates.append(
            {
                "bitstring_lsb_first": bitstring,
                "probability_rank": probability_rank,
                "probability": float(probability),
                "continuous_objective": float(allocation["objective"]),
                "allocation": allocation,
                "weights": weights,
                "bits": bits,
            }
        )
    candidates.sort(key=lambda item: item["continuous_objective"])
    elapsed = time.perf_counter() - start
    if not candidates:
        raise RuntimeError("no top-K candidate could be allocated")
    best = candidates[0]
    original_bitstring = payload["qaoa_solution"]["bitstring_lsb_first"]
    original_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate["bitstring_lsb_first"] == original_bitstring
        ),
        None,
    )
    payload["topk_continuous_rerank"] = {
        "candidates_requested": len(probabilities),
        "candidates_optimized": len(candidates),
        "failures": failures,
        "seconds": elapsed,
        "ranking_objective": (
            "continuous_long_only_full_investment_downside_risk_minus_return"
        ),
        "best": {
            "bitstring_lsb_first": best["bitstring_lsb_first"],
            "probability_rank": best["probability_rank"],
            "probability": best["probability"],
            "continuous_objective": best["continuous_objective"],
            "allocation": best["allocation"],
            "assets": _decode_assets(
                best["bits"], best["weights"], variables
            ),
        },
        "original_qaoa_candidate_continuous_rank": original_rank,
        "top_20": [
            {
                "bitstring_lsb_first": candidate["bitstring_lsb_first"],
                "probability_rank": candidate["probability_rank"],
                "probability": candidate["probability"],
                "continuous_objective": candidate["continuous_objective"],
            }
            for candidate in candidates[:20]
        ],
    }
    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        payload["topk_continuous_rerank"],
        ensure_ascii=False,
        indent=2,
        default=lambda _: None,
    ))


if __name__ == "__main__":
    main()
