"""CLI for the first real n=24 portfolio decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ising_qaoa import normalize_ising
from real_portfolio import load_config, solve_real_portfolio


def _serializable(result: dict) -> dict:
    selected = result["variables"].loc[result["variables"]["selected"]]
    return {
        "as_of": result["inputs"].as_of,
        "solver": "simulated_annealing",
        "qubo_energy": result["qubo_energy"],
        "meta": result["meta"],
        "allocation": result["allocation"],
        "qubo_objective_terms": result["objective_terms"],
        "selected": [
            {
                "variable": row["variable"],
                "code_or_instrument": row["underlying"],
                "name": row["name"],
                "asset_type": row["asset_type"],
                "direction": row["direction"],
                "factor_score": float(row["factor_score"]),
                "expected_annual_return_proxy": float(
                    row["expected_annual_return_proxy"]),
                "beta": float(row["beta"]),
                "weight": float(row["weight"]),
            }
            for _, row in selected.iterrows()
        ],
        "simplified_backtest_warning": (
            "本结果只按价格方向建模，未计手续费、滑点、资金费率、保证金、"
            "强平、合约乘数、基差和移仓，不代表可实现净收益。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/portfolio_default.yaml")
    parser.add_argument(
        "--output", default="results/real_portfolio_2026-07-03.json")
    parser.add_argument(
        "--instance-output", type=Path,
        help="Optional NPZ export for the GPU-only QAOA host.")
    args = parser.parse_args()
    result = solve_real_portfolio(load_config(args.config))
    payload = _serializable(result)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.instance_output:
        method = load_config(args.config)["qaoa"]["hamiltonian_normalization"]
        scaled_h, scaled_J, scale = normalize_ising(
            result["h"], result["J"], method=method)
        args.instance_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.instance_output,
            h=result["h"], J=result["J"], offset=result["ising_offset"],
            scaled_h=scaled_h, scaled_J=scaled_J,
            hamiltonian_scale=np.array(scale),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
