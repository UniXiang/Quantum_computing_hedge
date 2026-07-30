"""Repack 13-column cache CSV files into runtime-compatible pickle files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.dir.glob("*_1year.csv"))
    if not paths:
        raise RuntimeError(f"no *_1year.csv files in {args.dir}")
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        output = path.with_suffix(".pkl")
        frame.astype(str).to_pickle(output)
        print(f"{path.name} -> {output.name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
