"""Prepare n=24..32 long-only fixed-K scaling instances."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from design_long_n24 import prepare_design_problem
from real_portfolio import load_config


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 24 <= args.n <= 32:
        raise ValueError("scale test n must be between 24 and 32")
    config = copy.deepcopy(load_config(args.config))
    config["universe"]["finalists"] = args.n - 3
    config["universe"]["max_per_market"] = 18
    config["universe"]["max_per_sector"] = 8
    if not 1 <= args.K < args.n:
        raise ValueError("K must satisfy 1 <= K < n")
    config["objective"]["target_holdings"] = args.K
    result = prepare_design_problem(config)
    if len(result["h"]) != args.n:
        raise RuntimeError(
            f"prepared {len(result['h'])} variables, expected {args.n}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "instance.npz",
        Q=result["Q"],
        h=result["h"],
        J=result["J"],
        offset=result["offset"],
    )
    context = {
        "config": config,
        "variables": records(result["variables"]),
        "covariance": result["covariance"].tolist(),
        "meta": result["meta"],
        "ranking": records(result["ranking"].reset_index()),
        "preparation_sa_selection": (
            result["sa_selection"].astype(int).tolist()
        ),
        "preparation_sa_energy": result["sa_energy"],
        "preparation_sa_cardinality": result["sa_cardinality"],
    }
    (args.output_dir / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n": args.n,
        "K": args.K,
        "variables": [
            row["code"] for row in context["variables"]
        ],
        "preparation_sa_cardinality": result["sa_cardinality"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
