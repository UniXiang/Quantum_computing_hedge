#!/usr/bin/env bash
# scripts/t32_remote_check.sh — Task 3.2 remote acceptance on the Biren host.
#
# Usage (from the local repo root, once SSH key auth is ready):
#   ssh biren 'bash -s' < scripts/t32_remote_check.sh
#
# Runs ON the Biren container:
#   1. activates the SUPA environment and enters the remote repo
#   2. n=16 smoke: complex64+checkpoint CPU vs biren best_energy consistency
#      (threshold 1e-3, same bound as the spike handbook for complex64)
#   3. n=24 p=4 single forward+backward peak-memory measurement on the
#      biren device via torch.supa memory stats (graceful null if the API
#      is absent), checked against the <8GB acceptance target
# Writes results/t32_remote_check.json and exits non-zero on failure.
set -euo pipefail

REPO=/workspace/quantum/quantum_hedge
ENV_SCRIPT=/usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh

source "$ENV_SCRIPT" >/dev/null
cd "$REPO"
mkdir -p results

python3 - <<'PY'
import json
import os
import sys
import time

import numpy as np
import torch
import torch_br  # noqa: F401  registers the 'supa' backend

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from ising_qaoa import IsingQAOA, estimate_evolve_memory, resolve_device

TOL_CONSISTENCY = 1e-3   # complex64 CPU/GPU threshold (spike handbook)
MEM_TARGET = 8 * 2**30   # n=24 p=4 fwd+bwd peak must stay under 8GB

out = {"suite": "t32_remote_check", "ok": True,
       "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    J = np.triu(J, k=1)
    return h, J + J.T


def check(name, cond, detail):
    out[name] = {"ok": bool(cond), **detail}
    out["ok"] = out["ok"] and bool(cond)
    print(f"[{name}] ok={cond} {detail}", flush=True)


algo = IsingQAOA(algo_dir="/tmp/t32_qaoa_out")
dev = resolve_device("biren")
out["device"] = str(dev)

# --- 1. n=16 CPU vs biren consistency, complex64 + checkpoint --------------
h, J = make_instance(16, seed=320)
common = dict(layers=2, optimizer="autograd", max_iter=30, seed=42,
              dtype="complex64", checkpoint=True)
r_cpu = algo.solve(h, J, device="cpu", **common)
r_brn = algo.solve(h, J, device="biren", **common)
scale = max(1.0, abs(r_cpu["exact_energy"]))
rel = abs(r_cpu["best_energy"] - r_brn["best_energy"]) / scale
check("n16_cpu_vs_biren", rel < TOL_CONSISTENCY,
      {"cpu_best": r_cpu["best_energy"], "biren_best": r_brn["best_energy"],
       "exact": r_cpu["exact_energy"], "rel_diff": rel, "tol": TOL_CONSISTENCY,
       "same_bitstring": r_cpu["best_bitstring"] == r_brn["best_bitstring"]})

# --- 2. n=24 p=4 single fwd+bwd peak memory on biren -----------------------
supa = getattr(torch, "supa", None)
mem_api = supa is not None and hasattr(supa, "max_memory_allocated")
model = estimate_evolve_memory(24, 4, dtype="complex64", checkpoint=True)
detail = {"model_estimate_bytes": model["total_bytes"],
          "model_estimate_GiB": round(model["total_bytes"] / 2**30, 3),
          "mem_api_available": mem_api}
peak = None
if mem_api:
    n, p = 24, 4
    h, J = make_instance(n, seed=321)
    e_vec = algo._energy_vector(h, J)  # chunked, ~640MB float64
    e_vec_t = torch.as_tensor(e_vec, dtype=torch.float32, device=dev)
    del e_vec
    params = torch.zeros(2 * p, dtype=torch.float32, device=dev,
                         requires_grad=True)
    with torch.no_grad():
        params.copy_(torch.as_tensor(
            np.random.default_rng(1).uniform(0.0, 0.5, size=2 * p),
            dtype=torch.float32))
    if hasattr(supa, "empty_cache"):
        supa.empty_cache()
    if hasattr(supa, "reset_peak_memory_stats"):
        supa.reset_peak_memory_stats()
    psi = algo._evolve_torch(params, h, J, device=dev, e_vec=e_vec_t,
                             dtype=torch.complex64, checkpoint=True)
    energy = torch.sum((psi.real**2 + psi.imag**2) * e_vec_t)
    energy.backward()
    peak = int(supa.max_memory_allocated())
    detail.update(peak_bytes=peak, peak_GiB=round(peak / 2**30, 3),
                  energy=float(energy.detach().cpu()))
check("n24_p4_fwd_bwd_peak",
      (peak is not None and peak < MEM_TARGET) or peak is None,
      {**detail, "target_bytes": MEM_TARGET,
       "note": None if peak is not None else
               "torch.supa memory stats unavailable; only the analytic "
               "model estimate is reported — verify with brsmi"})

with open("results/t32_remote_check.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2), flush=True)
sys.exit(0 if out["ok"] else 1)
PY
