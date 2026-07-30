"""Run p=1..4, multi-seed expectation/CVaR QAOA experiments."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cardinality_qaoa import CardinalityQAOA


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["brsmi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except Exception as exc:
        return f"brsmi unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--seeds", default="11,42,73")
    parser.add_argument("--objectives", default="expectation,cvar")
    parser.add_argument("--cvar-alpha", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=512)
    args = parser.parse_args()

    instance = np.load(args.instance)
    h = np.asarray(instance["h"], dtype=np.float64)
    J = np.asarray(instance["J"], dtype=np.float64)
    offset = float(instance["offset"])
    context = json.loads(args.context.read_text(encoding="utf-8"))
    cardinality = int(
        context["config"]["objective"]["target_holdings"]
    )
    depths = [int(value) for value in args.depths.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    objectives = args.objectives.split(",")
    solver = CardinalityQAOA(len(h), cardinality, device=args.device)

    started = datetime.now(timezone.utc)
    before = gpu_snapshot()
    runs = []
    for objective in objectives:
        for depth in depths:
            for seed in seeds:
                start = time.perf_counter()
                result = solver.solve(
                    h,
                    J,
                    layers=depth,
                    max_iter=args.max_iter,
                    seed=seed,
                    init="random",
                    top_k=args.top_k,
                    restarts=1,
                    objective=objective,
                    cvar_alpha=args.cvar_alpha,
                )
                elapsed = time.perf_counter() - start
                runs.append(
                    {
                        "objective": objective,
                        "cvar_alpha": (
                            args.cvar_alpha if objective == "cvar" else None
                        ),
                        "p": depth,
                        "seed": seed,
                        "seconds": elapsed,
                        "best_bitstring_lsb_first": result["best_bitstring"],
                        "best_qubo_energy": result["best_energy"] + offset,
                        "exact_feasible_qubo_energy": (
                            result["exact_feasible_energy"] + offset
                        ),
                        "gap": (
                            result["best_energy"]
                            - result["exact_feasible_energy"]
                        ),
                        "optimized_objective": result["expectation"],
                        "probability_norm": result["probability_norm"],
                        "top_probability_mass": sum(
                            result["bitstrings"].values()
                        ),
                    }
                )
                print(json.dumps(runs[-1]), flush=True)

    payload = {
        "run": {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "device": args.device,
            "n": len(h),
            "target_holdings": cardinality,
            "depths": depths,
            "seeds": seeds,
            "objectives": objectives,
            "cvar_alpha": args.cvar_alpha,
            "max_iter": args.max_iter,
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
