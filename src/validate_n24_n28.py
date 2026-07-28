"""Task 3.3 — Biren n=24 QAOA vs matched-budget SA experiment.

The instance is a deterministic, data-independent portfolio QUBO built
from synthetic factor returns.  This keeps the accelerator experiment
reproducible on the remote host without syncing the private market-data
cache.  The downside-semivariance/return objective is normalized before
adding the cardinality penalty so the feasible-subspace landscape remains
numerically discriminative.

Primary metric (lower is better):

    feasible_relative_gap =
        (E_candidate - E_feasible_best)
        / (E_feasible_worst - E_feasible_best)

Only popcount-K candidates are scored.  The exact feasible envelope is
enumerated from the already-computed diagonal Ising energy vector in
bounded-memory chunks.  Writes JSON after every completed depth so a long
run remains recoverable if interrupted, then renders a Markdown report.
The CLI still accepts another ``--ns`` value for optional future resource
experiments, but n=24 is the accepted project target.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ising_qaoa import IsingQAOA, estimate_evolve_memory, ising_energy
from qubo_builder import build_qubo, qubo_to_ising
from solvers import solve_sa


def make_portfolio_instance(n: int, seed: int, window: int = 120):
    """Build a deterministic factor-return QUBO with target K=n//2."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.012, size=(window, 1))
    defensive = rng.normal(0.0001, 0.007, size=(window, 1))
    beta_market = rng.uniform(0.4, 1.4, size=(1, n))
    beta_defensive = rng.uniform(-0.5, 0.7, size=(1, n))
    alpha = rng.normal(0.00025, 0.00015, size=(1, n))
    idio = rng.normal(0.0, rng.uniform(0.004, 0.018, size=(1, n)),
                      size=(window, n))
    values = alpha + market * beta_market + defensive * beta_defensive + idio
    returns = pd.DataFrame(values, columns=[f"asset_{i:02d}" for i in range(n)])

    k = n // 2
    # Normalize only the financial objective. This is a change of units,
    # not a change of optimizer, and prevents raw daily-return magnitudes
    # (~1e-4) from being swallowed by float32 under the cardinality penalty.
    q_fin = build_qubo(returns, K=k, lam=0.05, A=0.0)
    scale = 0.01 / max(float(np.max(np.abs(q_fin))), np.finfo(float).tiny)
    q_fin *= scale

    penalty = 0.5
    q = q_fin + penalty * np.ones((n, n)) - penalty * np.eye(n)
    diag = np.diag_indices(n)
    q[diag] += penalty * (1.0 - 2.0 * k)
    q = (q + q.T) / 2.0
    return q, k, {
        "window": window,
        "seed": seed,
        "lambda": 0.05,
        "financial_scale": scale,
        "cardinality_penalty": penalty,
    }


def feasible_envelope(e_vec: np.ndarray, k: int,
                      chunk_size: int = 1 << 20) -> dict:
    """Exact min/max over states with popcount k, using bounded scratch."""
    dim = len(e_vec)
    n = int(math.log2(dim))
    if (1 << n) != dim:
        raise ValueError("energy vector length must be a power of two")
    best = math.inf
    worst = -math.inf
    seen = 0
    for start in range(0, dim, chunk_size):
        stop = min(start + chunk_size, dim)
        idx = np.arange(start, stop, dtype=np.uint64)
        counts = np.zeros(stop - start, dtype=np.uint8)
        for bit in range(n):
            counts += ((idx >> bit) & 1).astype(np.uint8)
        selected = e_vec[start:stop][counts == k]
        if selected.size:
            best = min(best, float(selected.min()))
            worst = max(worst, float(selected.max()))
            seen += int(selected.size)
    expected = math.comb(n, k)
    if seen != expected:
        raise RuntimeError(f"feasible enumeration saw {seen}, expected {expected}")
    return {"best": best, "worst": worst, "count": seen}


def best_feasible_qaoa(result: dict, h: np.ndarray, J: np.ndarray,
                       offset: float, k: int) -> tuple[str | None, float | None]:
    candidates = []
    for bitstring in result["bitstrings"]:
        if bitstring.count("1") == k:
            candidates.append((ising_energy(bitstring, h, J) + offset,
                               bitstring))
    if not candidates:
        return None, None
    energy, bitstring = min(candidates)
    return bitstring, float(energy)


def relative_gap(energy: float | None, best: float, worst: float):
    if energy is None:
        return None
    span = worst - best
    if span <= 0.0:
        return 0.0 if abs(energy - best) <= 1e-12 else None
    gap = float((energy - best) / span)
    # Exact endpoints can differ by a few ulps after Ising offset recovery.
    if -1e-9 < gap < 0.0:
        return 0.0
    if 1.0 < gap < 1.0 + 1e-9:
        return 1.0
    return gap


def render_markdown(payload: dict) -> str:
    lines = [
        "# Task 3.3 — Biren n=24 QAOA vs SA",
        "",
        f"- generated: {payload['generated_at']}",
        f"- device: `{payload['config']['device']}`",
        f"- dtype: `{payload['config']['dtype']}`; "
        f"adjoint={payload['config']['adjoint']}; "
        f"checkpoint={payload['config']['checkpoint']}",
        f"- max_iter per restart/level: {payload['config']['max_iter']}",
        f"- probability pool before K postselection: "
        f"{payload['config']['top_k']}",
        "",
        "Metric: `(E - E_feasible_best) / "
        "(E_feasible_worst - E_feasible_best)`; lower is better. "
        "Infeasible candidates are not scored.",
        "",
        "| n | p | init | QAOA gap | SA gap | QAOA feasible | "
        "SA feasible | QAOA s | SA s |",
        "|---:|---:|:---|---:|---:|:---:|:---:|---:|---:|",
    ]
    for case in payload["cases"]:
        for run in case["runs"]:
            qgap = run["qaoa"]["feasible_relative_gap"]
            sgap = run["sa"]["feasible_relative_gap"]
            lines.append(
                f"| {case['n']} | {run['layers']} | {run['init']} | "
                f"{'—' if qgap is None else f'{qgap:.6f}'} | "
                f"{'—' if sgap is None else f'{sgap:.6f}'} | "
                f"{'yes' if run['qaoa']['feasible'] else 'no'} | "
                f"{'yes' if run['sa']['feasible'] else 'no'} | "
                f"{run['qaoa']['wall_s']:.2f} | {run['sa']['wall_s']:.2f} |")
        lines += [
            "",
            f"n={case['n']} exact feasible envelope: "
            f"[{case['feasible']['best_qubo']:.8f}, "
            f"{case['feasible']['worst_qubo']:.8f}] over "
            f"{case['feasible']['count']:,} states. "
            f"Energy-vector build: {case['energy_vector_wall_s']:.2f}s; "
            f"adjoint peak model: {case['memory_model']['total_GiB']:.3f} GiB.",
            "",
        ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", nargs="+", type=int, default=[24])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--device", default="biren")
    parser.add_argument("--dtype", choices=["complex64", "complex128"],
                        default="complex64")
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--top-k", type=int, default=4096,
                        help="probability-ranked pool before K postselection")
    parser.add_argument("--init", choices=["random", "interp"],
                        default="random")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--adjoint", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/t33_n24_n28.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "suite": "t33_n24_n28",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "config": {
            "ns": args.ns,
            "layers": args.layers,
            "max_iter": args.max_iter,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "top_k": args.top_k,
            "init": args.init,
            "checkpoint": args.checkpoint,
            "adjoint": args.adjoint,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    qaoa = IsingQAOA(algo_dir=str(args.output.parent / "t33_qaoa"))

    def checkpoint_results():
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        args.output.with_suffix(".md").write_text(
            render_markdown(payload), encoding="utf-8")

    for n in args.ns:
        print(f"\n=== n={n}: instance and exact feasible envelope ===", flush=True)
        q, k, instance_meta = make_portfolio_instance(n, args.seed + n)
        h, J, offset = qubo_to_ising(q)
        t0 = time.perf_counter()
        e_vec = qaoa._energy_vector(h, J)
        evec_wall = time.perf_counter() - t0
        envelope = feasible_envelope(e_vec, k)
        all_best = float(e_vec.min())
        # The chosen penalty must make the unconstrained ground state
        # feasible; otherwise the QAOA Hamiltonian is not the intended one.
        if all_best < envelope["best"] - 1e-7:
            raise RuntimeError(
                f"n={n}: cardinality penalty insufficient: unconstrained "
                f"{all_best} < feasible {envelope['best']}")

        model = estimate_evolve_memory(
            n, max(args.layers), dtype=args.dtype,
            checkpoint=args.checkpoint, adjoint=args.adjoint)
        case = {
            "n": n,
            "k": k,
            "instance": instance_meta,
            "offset": offset,
            "energy_vector_wall_s": evec_wall,
            "feasible": {
                "best_ising": envelope["best"],
                "worst_ising": envelope["worst"],
                "best_qubo": envelope["best"] + offset,
                "worst_qubo": envelope["worst"] + offset,
                "count": envelope["count"],
            },
            "memory_model": {
                **model,
                "total_GiB": model["total_bytes"] / 2**30,
            },
            "runs": [],
        }
        payload["cases"].append(case)
        checkpoint_results()

        for layers in args.layers:
            print(f"\n--- n={n}, p={layers}, QAOA ---", flush=True)
            t0 = time.perf_counter()
            result = qaoa.solve(
                h, J, layers=layers, device=args.device,
                optimizer="autograd", max_iter=args.max_iter,
                seed=args.seed + 100 * n + layers, init=args.init,
                dtype=args.dtype, checkpoint=args.checkpoint,
                adjoint=args.adjoint, final_backend="native",
                energy_vector=e_vec, top_k=args.top_k)
            qaoa_wall = time.perf_counter() - t0
            qbits, qenergy = best_feasible_qaoa(
                result, h, J, offset, k)

            print(f"--- n={n}, p={layers}, matched SA "
                  f"budget={qaoa_wall:.2f}s ---", flush=True)
            t0 = time.perf_counter()
            x_sa, e_sa = solve_sa(
                q, budget_s=qaoa_wall,
                seed=args.seed + 10_000 + 100 * n + layers)
            sa_wall = time.perf_counter() - t0
            sa_feasible = int(x_sa.sum()) == k
            sa_energy = float(e_sa) if sa_feasible else None

            best_q = envelope["best"] + offset
            worst_q = envelope["worst"] + offset
            run = {
                "layers": layers,
                "init": args.init,
                "qaoa": {
                    "wall_s": qaoa_wall,
                    "bitstring": qbits,
                    "energy_qubo": qenergy,
                    "feasible": qbits is not None,
                    "feasible_relative_gap": relative_gap(
                        qenergy, best_q, worst_q),
                    "expectation_history": result["energy_history"],
                    "top_probabilities": result["bitstrings"],
                    "final_backend": result["final_backend"],
                },
                "sa": {
                    "budget_s": qaoa_wall,
                    "wall_s": sa_wall,
                    "x": x_sa.tolist(),
                    "energy_qubo": sa_energy,
                    "feasible": sa_feasible,
                    "feasible_relative_gap": relative_gap(
                        sa_energy, best_q, worst_q),
                },
            }
            case["runs"].append(run)
            checkpoint_results()
            print(
                f"n={n} p={layers}: qaoa_gap="
                f"{run['qaoa']['feasible_relative_gap']} sa_gap="
                f"{run['sa']['feasible_relative_gap']}", flush=True)

        # Release the multi-GiB host energy vector before the next n.
        del e_vec
        if args.device != "cpu":
            accelerator = getattr(torch, "supa", None)
            if accelerator is not None and hasattr(accelerator, "empty_cache"):
                accelerator.empty_cache()

    checkpoint_results()
    print(f"\nJSON: {args.output}")
    print(f"Markdown: {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
