import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cardinality_qaoa import (
    CardinalityQAOA,
    fixed_weight_states,
    ising_energies_for_states,
)
from ising_qaoa import ising_energy
from ising_qaoa import IsingQAOA


def test_fixed_weight_states_are_complete_and_feasible():
    states = fixed_weight_states(6, 2)
    assert len(states) == 15
    assert np.all(states[1:] > states[:-1])
    assert all(int(state).bit_count() == 2 for state in states)


def test_restricted_energies_match_direct_ising():
    rng = np.random.default_rng(4)
    h = rng.normal(size=6)
    raw = rng.normal(size=(6, 6))
    J = np.triu(raw, 1)
    J = J + J.T
    states = fixed_weight_states(6, 2)
    energies = ising_energies_for_states(h, J, states, chunk_size=3)
    direct = [
        ising_energy(
            "".join(str((int(state) >> bit) & 1) for bit in range(6)),
            h,
            J,
        )
        for state in states
    ]
    np.testing.assert_allclose(energies, direct, atol=1e-12)


def test_xy_evolution_preserves_norm_and_has_gradient():
    solver = CardinalityQAOA(6, 2, device="cpu")
    params = torch.tensor(
        [0.2, 0.1], dtype=torch.float64, requires_grad=True
    )
    energies = torch.linspace(
        -1.0, 1.0, len(solver.states), dtype=torch.float64
    )
    state = solver._evolve(params, energies)
    norm = torch.sum(state.real.square() + state.imag.square())
    assert float(norm.detach()) == pytest.approx(1.0, abs=1e-10)
    expectation = torch.sum(
        (state.real.square() + state.imag.square()) * energies
    )
    expectation.backward()
    assert torch.all(torch.isfinite(params.grad))


def test_batched_xy_matches_independent_evolution():
    solver = CardinalityQAOA(6, 2, device="cpu")
    params = torch.tensor(
        [[0.2, 0.1], [0.35, 0.25]], dtype=torch.float64
    )
    energies = torch.linspace(
        -1.0, 1.0, len(solver.states), dtype=torch.float64
    )
    batched = solver._evolve_batch(params, energies)
    independent = torch.stack(
        [solver._evolve(row, energies) for row in params]
    )
    torch.testing.assert_close(batched, independent)


def test_solver_returns_only_feasible_bitstrings():
    rng = np.random.default_rng(8)
    h = rng.normal(size=6)
    raw = rng.normal(size=(6, 6))
    J = np.triu(raw, 1)
    J = J + J.T
    solver = CardinalityQAOA(6, 2, device="cpu")
    result = solver.solve(
        h, J, layers=1, max_iter=3, seed=1, top_k=10, restarts=1
    )
    assert result["probability_norm"] == pytest.approx(1.0, abs=1e-6)
    assert all(bits.count("1") == 2 for bits in result["bitstrings"])
    assert result["best_bitstring"].count("1") == 2


def test_cvar_solver_returns_finite_feasible_result():
    rng = np.random.default_rng(18)
    h = rng.normal(size=6)
    raw = rng.normal(size=(6, 6))
    J = np.triu(raw, 1)
    J = J + J.T
    solver = CardinalityQAOA(6, 2, device="cpu")
    result = solver.solve(
        h,
        J,
        layers=1,
        max_iter=3,
        seed=2,
        top_k=10,
        restarts=1,
        objective="cvar",
        cvar_alpha=0.2,
    )
    assert np.isfinite(result["expectation"])
    assert result["objective"] == "cvar"
    assert result["cvar_alpha"] == pytest.approx(0.2)
    assert all(bits.count("1") == 2 for bits in result["bitstrings"])


def test_warm_start_concentrates_probability_on_incumbent():
    h = np.zeros(6)
    J = np.zeros((6, 6))
    solver = CardinalityQAOA(6, 2, device="cpu")
    warm = "110000"
    result = solver.solve(
        h,
        J,
        layers=1,
        max_iter=1,
        seed=2,
        top_k=5,
        restarts=1,
        warm_start_bitstring=warm,
        warm_start_strength=8.0,
        initial_angle_scale=0.0,
        reference_bitstring=warm,
    )
    assert result["reference_probability_rank"] == 1
    assert result["reference_probability"] > 0.99
    assert result["warm_start_initial_probability"] > 0.99


def test_n24_restricted_energy_matches_full_vector_on_feasible_states():
    rng = np.random.default_rng(81)
    n, cardinality = 24, 8
    h = rng.normal(size=n)
    raw = rng.normal(size=(n, n))
    J = np.triu(raw, 1)
    J = J + J.T
    states = fixed_weight_states(n, cardinality)
    # A large enough prefix exercises the tall-skinny BLAS path that was
    # observed to corrupt rows under the Biren image's 64-thread OpenBLAS.
    sample = states[:100_000]
    restricted = ising_energies_for_states(
        h, J, sample, chunk_size=100_000
    )
    full = IsingQAOA._energy_vector(h, J, chunk_size=1 << 18)
    np.testing.assert_allclose(restricted, full[sample], atol=1e-10)
