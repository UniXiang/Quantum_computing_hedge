"""Tests for src/ising_qaoa.py — IsingQAOA weighted-Ising QAOA extension.

Bitstring convention (must match src/ising_qaoa docstring):
    The unitarylab statevector backend orders basis states with qubit 0 as
    the least significant bit of the state index. A basis state is reported
    as bitstring s of length n with s[i] = outcome x_i of qubit i, i.e.
    state_index = sum_i x_i * 2**i (see ising_qaoa.bitstring_of_index).
    Spin mapping: z_i = 1 - 2 * int(s[i])  (bit 0 -> z=+1, bit 1 -> z=-1).
    Ising energy: E(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
    (J symmetric, zero diagonal; each pair counted once via i<j).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ising_qaoa import IsingQAOA, ising_energy, bitstring_of_index


def make_instance(n, seed):
    """Random (h, J) instance: h in [-1, 1], J symmetric, zero diagonal."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    J = np.triu(J, k=1)
    J = J + J.T
    return h, J


def all_bitstrings(n):
    return [bitstring_of_index(i, n) for i in range(2**n)]


@pytest.fixture()
def algo(tmp_path):
    return IsingQAOA(algo_dir=str(tmp_path / "qaoa_out"))


# ---------------------------------------------------------------------------
# 1. Hamiltonian correctness: diagonal of _get_h_cost equals direct E(z)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, seed", [(4, 0), (5, 1), (5, 2), (6, 3), (6, 4)])
def test_hamiltonian_diagonal_matches_direct_energy(algo, n, seed):
    h, J = make_instance(n, seed)
    h_cost = algo._get_h_cost(h, J)
    assert h_cost.shape == (2**n, 2**n)
    diag = np.real(np.diag(h_cost))
    for idx, s in enumerate(all_bitstrings(n)):
        expected = ising_energy(s, h, J)
        assert abs(diag[idx] - expected) < 1e-10, (
            f"basis |{s}>: hamiltonian {diag[idx]} != direct {expected}"
        )


def test_ising_energy_helper_convention():
    # n=2, bit '0' -> z=+1, bit '1' -> z=-1
    h = np.array([0.5, -0.25])
    J = np.array([[0.0, 2.0], [2.0, 0.0]])
    # s='01': z = (+1, -1) -> E = 0.5*(+1) + (-0.25)*(-1) + 2.0*(+1)*(-1)
    assert ising_energy("01", h, J) == pytest.approx(0.5 + 0.25 - 2.0)
    # s='11': z = (-1, -1) -> E = -0.5 + 0.25 + 2.0
    assert ising_energy("11", h, J) == pytest.approx(-0.5 + 0.25 + 2.0)


# ---------------------------------------------------------------------------
# 2. solve interface contract + variational principle
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "bitstrings", "best_bitstring", "best_energy", "exact_energy",
    "energy_history", "n_qubits", "layers", "device", "optimizer",
}


@pytest.mark.parametrize("optimizer", ["autograd", "cobyla"])
def test_solve_interface_contract(algo, optimizer):
    h, J = make_instance(4, seed=10)
    res = algo.solve(h, J, layers=2, device="cpu", optimizer=optimizer,
                     max_iter=60, seed=1)
    assert REQUIRED_KEYS <= set(res.keys())
    assert isinstance(res["bitstrings"], dict) and len(res["bitstrings"]) > 0
    for s, p in res["bitstrings"].items():
        assert isinstance(s, str) and len(s) == 4
        assert 0.0 <= p <= 1.0
    assert res["best_bitstring"] in res["bitstrings"]
    assert res["n_qubits"] == 4 and res["layers"] == 2
    assert res["device"] == "cpu" and res["optimizer"] == optimizer
    assert isinstance(res["energy_history"], list) and len(res["energy_history"]) > 0
    # variational principle: no sampled state beats the exact ground state
    assert res["best_energy"] >= res["exact_energy"] - 1e-6
    # best_energy must equal the direct energy of best_bitstring
    assert res["best_energy"] == pytest.approx(ising_energy(res["best_bitstring"], h, J))


def test_exact_energy_matches_brute_force(algo):
    h, J = make_instance(5, seed=11)
    res = algo.solve(h, J, layers=2, max_iter=30, seed=2)
    brute = min(ising_energy(s, h, J) for s in all_bitstrings(5))
    assert res["exact_energy"] == pytest.approx(brute, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Convergence: n=6, layers=4 approximation ratio >= 0.9
# ---------------------------------------------------------------------------
def test_convergence_n6_layers4(algo):
    h, J = make_instance(6, seed=20)
    res = algo.solve(h, J, layers=4, device="cpu", optimizer="autograd",
                     max_iter=200, seed=42)
    gap = (res["best_energy"] - res["exact_energy"]) / abs(res["exact_energy"])
    ratio = 1.0 - gap
    assert ratio >= 0.9, (
        f"approximation ratio {ratio:.4f} < 0.9 "
        f"(best={res['best_energy']:.6f}, exact={res['exact_energy']:.6f})"
    )


# ---------------------------------------------------------------------------
# 4. Reproducibility: same seed twice -> same best_bitstring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("optimizer", ["autograd", "cobyla"])
def test_reproducibility(algo, optimizer):
    h, J = make_instance(5, seed=30)
    r1 = algo.solve(h, J, layers=3, optimizer=optimizer, max_iter=80, seed=7)
    r2 = algo.solve(h, J, layers=3, optimizer=optimizer, max_iter=80, seed=7)
    assert r1["best_bitstring"] == r2["best_bitstring"]
    assert r1["best_energy"] == pytest.approx(r2["best_energy"])
    assert r1["energy_history"] == pytest.approx(r2["energy_history"])


# ---------------------------------------------------------------------------
# 5. COBYLA cross-validation: both optimizers agree within 5% of |E_exact|
# ---------------------------------------------------------------------------
def test_cobyla_cross_validation(algo):
    h, J = make_instance(4, seed=40)
    common = dict(layers=4, device="cpu", max_iter=200, seed=42)
    r_auto = algo.solve(h, J, optimizer="autograd", **common)
    r_cob = algo.solve(h, J, optimizer="cobyla", **common)
    exact = r_auto["exact_energy"]
    assert exact == pytest.approx(r_cob["exact_energy"])
    diff = abs(r_auto["best_energy"] - r_cob["best_energy"])
    assert diff < 0.05 * abs(exact), (
        f"autograd {r_auto['best_energy']:.6f} vs cobyla "
        f"{r_cob['best_energy']:.6f}, diff {diff:.6f} >= 5% of |exact|"
    )


# ---------------------------------------------------------------------------
# 6. Simulator consistency: torch-native autograd evolution matches
#    unitarylab circuit execution at identical parameters.
# ---------------------------------------------------------------------------
def test_torch_evolution_matches_unitarylab_circuit(algo):
    n = 4
    h, J = make_instance(n, seed=50)
    layers = 2
    rng = np.random.default_rng(3)
    params = rng.uniform(0, np.pi, size=2 * layers)
    qc = algo._build_circuit(params, h, J)
    psi_circ = np.asarray(
        qc.execute(backend="torch", device="cpu").state
    ).flatten().astype(np.complex128)
    psi_torch = algo._evolve_torch(params, h, J, device="cpu").detach().cpu().numpy()
    np.testing.assert_allclose(np.abs(psi_circ), np.abs(psi_torch), atol=1e-5)
    # phases must match too (same unitary), up to numerical precision
    np.testing.assert_allclose(psi_circ, psi_torch, atol=1e-4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_inputs_raise(algo):
    h, J = make_instance(4, seed=60)
    with pytest.raises(ValueError):
        algo.solve(h, J, optimizer="not-an-optimizer")
    with pytest.raises(ValueError):
        algo.solve(h, np.zeros((5, 5)))  # shape mismatch
    with pytest.raises(ValueError):
        J_bad = np.zeros((4, 4)); J_bad[0, 1] = 1.0  # not symmetric
        algo.solve(h, J_bad)
