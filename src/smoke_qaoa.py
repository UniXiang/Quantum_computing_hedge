"""Smoke test: environment check for the quantum core.

1. Runs the stock unitarylab QAOAAlgorithm MaxCut demo (n=6) to confirm
   the third-party library works in this environment.
2. Runs IsingQAOA on a random n=6 weighted-Ising instance.

Prints device, wall time, and energies for both.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm
from ising_qaoa import IsingQAOA

DEVICE = "cpu"
ALGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "smoke")


def smoke_stock_maxcut():
    print("=" * 60)
    print("Smoke 1: stock unitarylab QAOAAlgorithm (MaxCut, n=6)")
    print("=" * 60)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
    algo = QAOAAlgorithm(text_mode="plain",
                         algo_dir=os.path.join(ALGO_DIR, "maxcut"))
    t0 = time.time()
    algo.run(edges=edges, n=6, layers=2, max_iter=60,
             backend="torch", device=DEVICE)
    print(f"[smoke] stock MaxCut done in {time.time() - t0:.2f}s "
          f"on device={DEVICE}")


def smoke_ising():
    print("=" * 60)
    print("Smoke 2: IsingQAOA random weighted-Ising instance (n=6)")
    print("=" * 60)
    rng = np.random.default_rng(2026)
    n = 6
    h = rng.uniform(-1.0, 1.0, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    J = np.triu(J, k=1)
    J = J + J.T

    algo = IsingQAOA(algo_dir=os.path.join(ALGO_DIR, "ising"))
    for optimizer in ("autograd", "cobyla"):
        t0 = time.time()
        res = algo.solve(h, J, layers=4, device=DEVICE,
                         optimizer=optimizer, max_iter=200, seed=42)
        dt = time.time() - t0
        ratio = 1.0 - (res["best_energy"] - res["exact_energy"]) / abs(res["exact_energy"])
        print(f"[smoke] optimizer={optimizer:9s} device={res['device']} "
              f"time={dt:.2f}s best_energy={res['best_energy']:.6f} "
              f"exact={res['exact_energy']:.6f} approx_ratio={ratio:.4f} "
              f"best_bitstring={res['best_bitstring']}")


if __name__ == "__main__":
    smoke_stock_maxcut()
    smoke_ising()
    print("[smoke] OK")
