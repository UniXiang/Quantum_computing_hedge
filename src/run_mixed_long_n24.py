"""Prepare the mixed long-only n=24 QUBO and its GPU context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mixed_long_n24 import prepare_problem
from real_portfolio import load_config


def _records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    result = prepare_problem(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "mixed_long_n24_instance.npz",
        Q=result["Q"],
        h=result["h"],
        J=result["J"],
        offset=result["offset"],
    )
    context = {
        "config": config,
        "variables": _records(result["variables"]),
        "covariance": result["covariance"].tolist(),
        "meta": result["meta"],
        "ranking": json.loads(
            result["ranking"].reset_index().to_json(
                orient="records", force_ascii=False
            )
        ),
        "sa_selection": result["sa_selection"].astype(int).tolist(),
        "sa_energy": result["sa_energy"],
    }
    (args.output_dir / "mixed_long_n24_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
