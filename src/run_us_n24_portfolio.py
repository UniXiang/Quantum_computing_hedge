"""Run the US-stock + BTC/CL/XAU signed n=24 portfolio pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from us_n24_portfolio import load_config, solve_us_n24_portfolio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/portfolio_us_n24.yaml"
    )
    parser.add_argument(
        "--output", default="results/us_n24_portfolio.json"
    )
    parser.add_argument(
        "--instance", default="results/us_n24_instance.npz"
    )
    parser.add_argument(
        "--report", default=""
    )
    parser.add_argument(
        "--gpu-note",
        default=(
            "本命令未执行 QAOA；金融筛选、模拟退火和连续权重分配均在 CPU 完成。"
        ),
    )
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    config = load_config(args.config)
    result = solve_us_n24_portfolio(config)
    duration = time.perf_counter() - start_clock
    variables = result["variables"]
    selected = variables.loc[variables["selected"]]
    payload = {
        "as_of": result["inputs"].as_of,
        "solver": "simulated_annealing",
        "qubo_energy": result["qubo_energy"],
        "meta": result["meta"],
        "allocation": result["allocation"],
        "preselected_us_stocks": result["preselection"].ranking.loc[
            [item["code"] for item in result["preselection"].candidates]
        ].reset_index().to_dict(orient="records"),
        "selected_directions": selected[
            [
                "variable", "underlying", "asset_type", "sector",
                "name", "direction", "factor_score",
                "expected_annual_return_proxy", "beta", "weight",
            ]
        ].to_dict(orient="records"),
        "run": {
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "host": platform.node(),
        },
        "parameters": config,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    instance = Path(args.instance)
    instance.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        instance,
        h=result["h"],
        J=result["J"],
        offset=np.asarray(result["ising_offset"]),
        Q=result["Q"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved portfolio: {output}")
    print(f"Saved Hamiltonian: {instance}")
    if args.report:
        from portfolio_report import (
            collect_runtime_environment,
            generate_portfolio_report,
        )

        runtime = collect_runtime_environment(args.gpu_note)
        runtime.update(payload["run"])
        report = generate_portfolio_report(
            payload, config, args.report, runtime
        )
        print(f"Saved PDF report: {report}")


if __name__ == "__main__":
    main()
