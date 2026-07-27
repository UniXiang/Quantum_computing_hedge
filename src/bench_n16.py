"""bench_n16.py — Task 3.1 benchmark: n=16 solve() wall time + e_vec memory.

Modes
-----
  python bench_n16.py solve   # time a full n=16 solve() (deterministic instance)
  python bench_n16.py mem     # e_vec memory peak (VmHWM delta) at n=20/22/24,
                              # chunked implementation vs the old dense algorithm
                              # (old replicated inline, n=20 only: 1.6GB intermediates)

The solve instance is synthetic (seeded) so the benchmark does not depend on
the bs_cache data and is reproducible across refactors.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ising_qaoa import IsingQAOA


def make_instance(n, seed=123):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    J = np.triu(J, k=1)
    return h, J + J.T


def vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmHWM not found")


def energy_vector_old(h, J):
    """Pre-refactor algorithm (kept inline for A/B memory comparison)."""
    n = len(h)
    idx = np.arange(2**n)
    shifts = np.arange(n)
    bits = (idx[:, None] >> shifts[None, :]) & 1
    z = 1.0 - 2.0 * bits.astype(np.float64)
    e = z @ h
    iu = np.triu_indices(n, k=1)
    e = e + (z[:, iu[0]] * z[:, iu[1]]) @ J[iu]
    return e


def bench_solve():
    h, J = make_instance(16)
    qaoa = IsingQAOA()
    t0 = time.perf_counter()
    res = qaoa.solve(h, J, layers=4, device="cpu", optimizer="autograd",
                     max_iter=200, seed=42)
    dt = time.perf_counter() - t0
    print(f"n=16 solve(): {dt:.1f}s  best_energy={res['best_energy']:.6f} "
          f"exact={res['exact_energy']:.6f}")


def _mem_one(n, impl):
    """Worker: measure VmHWM delta of one e_vec computation (fresh process)."""
    import gc
    h, J = make_instance(n)
    gc.collect()
    base = vmhwm_mb()
    if impl == "chunked":
        e = IsingQAOA._energy_vector(h, J)
    else:
        e = energy_vector_old(h, J)
    peak = vmhwm_mb() - base
    print(f"{impl} n={n}: peak_delta={peak:.0f}MB "
          f"(e_vec itself {e.nbytes / 1024**2:.0f}MB, "
          f"extra {peak - e.nbytes / 1024**2:.0f}MB)")


def bench_mem():
    # Each measurement runs in a FRESH subprocess: VmHWM is a monotonic
    # high-water mark, so sequential measurements in one process would
    # under-report everything after the first.
    import subprocess
    for n in (20, 22, 24):
        subprocess.run([sys.executable, __file__, "mem_one", str(n),
                        "chunked"], check=True)
    # old algorithm for reference (n=20 only: (2^20, 190) intermediate ~1.6GB)
    subprocess.run([sys.executable, __file__, "mem_one", "20", "old"],
                   check=True)
    # correctness cross-check on the same instance (small enough for both)
    n = 20
    h, J = make_instance(n)
    e_old = energy_vector_old(h, J)
    e_new = IsingQAOA._energy_vector(h, J)
    print(f"old-vs-chunked max abs diff at n={n}: "
          f"{float(np.max(np.abs(e_old - e_new))):.3e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"
    if mode == "mem_one":
        _mem_one(int(sys.argv[2]), sys.argv[3])
    else:
        {"solve": bench_solve, "mem": bench_mem}[mode]()
