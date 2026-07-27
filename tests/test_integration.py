"""tests/test_integration.py — cross-module integration (Task 3.1, deliverable 5).

Pins the full chain inside pytest (previously only verified outside pytest):

    build_qubo (synthetic returns) -> qubo_to_ising -> IsingQAOA().solve
    -> best_bitstring -> x -> QUBO energy

Assertions:
  1. ising_to_qubo_energy(x, h, J, offset) == x'Qx exactly (float64, the
     QUBO<->Ising mapping in qubo_builder is algebraically exact; both sides
     are sums of the same products, so agreement to ~1e-10 is expected and
     1e-8 leaves margin for summation-order differences).
  2. best_energy + offset == x'Qx(best_bitstring): solve()'s reported
     Ising energy must be self-consistent with the QUBO objective.
  3. exact_energy + offset == min over all 2^n states of x'Qx: the
     ground-state anchor crosses the module boundary correctly.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qubo_builder import build_qubo, qubo_to_ising, ising_to_qubo_energy
from ising_qaoa import IsingQAOA

TOL = 1e-8  # see module docstring: mapping is algebraically exact


def make_returns(n, T=60, seed=99):
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=1e-3, scale=0.02, size=(T, n))
    return pd.DataFrame(data, columns=[f"asset{i}" for i in range(n)])


@pytest.fixture()
def algo(tmp_path):
    return IsingQAOA(algo_dir=str(tmp_path / "qaoa_out"))


@pytest.mark.parametrize("n, K", [(8, 4), (10, 5)])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_full_chain_bitstring_to_qubo_energy(algo, n, K, dtype):
    returns = make_returns(n)
    Q = build_qubo(returns, K=K, lam=0.05, A=0.01, gamma=0.0, x_prev=None)
    h, J, offset = qubo_to_ising(Q)

    res = algo.solve(h, J, layers=2, device="cpu", optimizer="autograd",
                     max_iter=60, seed=42, dtype=dtype)

    # bitstring convention: s[i] = x_i (qubit i, LSB-first state index)
    x = np.array([int(b) for b in res["best_bitstring"]], dtype=np.float64)
    assert len(x) == n

    # (1) cross-module energy identity
    e_qubo_direct = float(x @ Q @ x)
    e_qubo_via_ising = ising_to_qubo_energy(x, h, J, offset)
    assert e_qubo_via_ising == pytest.approx(e_qubo_direct, abs=TOL)

    # (2) solve()-reported energy is self-consistent with the QUBO objective
    assert res["best_energy"] + offset == pytest.approx(e_qubo_direct,
                                                        abs=TOL)

    # (3) ground-state anchor: exact_energy + offset == min_x x'Qx (brute force)
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    e_all = np.einsum("bi,ij,bj->b", bits, Q, bits)
    assert res["exact_energy"] + offset == pytest.approx(float(e_all.min()),
                                                         abs=TOL)
    # variational principle on the QUBO side
    assert e_qubo_direct >= float(e_all.min()) - TOL


def test_full_chain_with_turnover(algo):
    """Same chain with gamma != 0 and x_prev: linear term on the diagonal."""
    n, K = 8, 3
    returns = make_returns(n, seed=100)
    rng = np.random.default_rng(101)
    x_prev = np.zeros(n)
    x_prev[rng.choice(n, size=K, replace=False)] = 1.0
    Q = build_qubo(returns, K=K, lam=0.05, A=0.01, gamma=0.005, x_prev=x_prev)
    h, J, offset = qubo_to_ising(Q)

    res = algo.solve(h, J, layers=2, device="cpu", optimizer="autograd",
                     max_iter=40, seed=7)
    x = np.array([int(b) for b in res["best_bitstring"]], dtype=np.float64)
    assert ising_to_qubo_energy(x, h, J, offset) == pytest.approx(
        float(x @ Q @ x), abs=TOL)
    assert res["best_energy"] + offset == pytest.approx(float(x @ Q @ x),
                                                        abs=TOL)
