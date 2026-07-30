"""Build one compact JSON summary for fixed-K scaling runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def gpu_metrics(snapshot: str) -> tuple[int | None, int | None]:
    memory = re.search(r"(\d+)MiB / \d+MiB", snapshot)
    utilization = re.search(r"\|\s+(\d+)%\s+Default", snapshot)
    return (
        int(memory.group(1)) if memory else None,
        int(utilization.group(1)) if utilization else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for n in (24, 28, 30, 32):
        payload = json.loads(
            (args.root / f"n{n}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        memory, utilization = gpu_metrics(
            payload["run"]["gpu_after"]
        )
        rows.append({
            "n": n,
            "K": payload["run"]["K"],
            "feasible_dimension": (
                payload["search_space"]["fixed_weight_dimension"]
            ),
            "state_construction_seconds": (
                payload["run"]["state_construction_seconds"]
            ),
            "sa_seconds": payload["run"]["sa_seconds"],
            "qaoa_seconds": payload["run"]["qaoa_seconds"],
            "gpu_memory_snapshot_mib": memory,
            "gpu_utilization_snapshot_pct": utilization,
            "sa_equals_exact": (
                payload["warm_start"]["equals_posthoc_exact"]
            ),
            "qaoa_equals_exact": (
                payload["qaoa"]["top_candidate_equals_exact"]
            ),
            "qaoa_gap": payload["qaoa"]["gap"],
            "exact_probability": (
                payload["qaoa"]["exact_probability"]
            ),
            "exact_probability_rank": (
                payload["qaoa"]["exact_probability_rank"]
            ),
            "shots_1000_hits": (
                payload["qaoa"]["simulated_shot_hits"]["1000"]
            ),
        })
    summary = {
        "method": "fixed-K SA-warm-start QAOA",
        "K": 8,
        "exact_answer_used_for_initialization": False,
        "exact_enumeration_role": "posthoc_validation_only",
        "all_qaoa_top_candidates_equal_exact": all(
            row["qaoa_equals_exact"] for row in rows
        ),
        "rows": rows,
        "next_stage": (
            "optimize K itself only after fixed-K scaling validation"
        ),
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
