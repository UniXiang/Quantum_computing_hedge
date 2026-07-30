"""Merge completed expectation log rows with the resumed CVaR JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expectation-log", required=True, type=Path)
    parser.add_argument("--cvar-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expectation = []
    for line in args.expectation_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("objective") == "expectation":
            expectation.append(row)
    cvar = json.loads(args.cvar_json.read_text(encoding="utf-8"))
    if len(expectation) != 12 or len(cvar["runs"]) != 12:
        raise RuntimeError(
            f"expected 12+12 rows, got {len(expectation)}+"
            f"{len(cvar['runs'])}"
        )
    payload = {
        "run": {
            **cvar["run"],
            "objectives": ["expectation", "cvar"],
            "note": (
                "expectation rows recovered from the completed first phase; "
                "CVaR resumed after a SUPA CumsumBackward/Flip workaround"
            ),
        },
        "runs": expectation + cvar["runs"],
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
