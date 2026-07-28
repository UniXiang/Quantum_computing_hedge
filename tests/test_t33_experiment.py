"""Small-n correctness tests for the Task 3.3 experiment machinery."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ising_qaoa import IsingQAOA, bitstring_of_index
from qubo_builder import qubo_to_ising
from validate_n24_n28 import (
    best_feasible_qaoa,
    feasible_envelope,
    make_portfolio_instance,
    relative_gap,
)


def test_feasible_envelope_matches_direct_popcount():
    q, k, _ = make_portfolio_instance(8, seed=3308)
    h, J, _ = qubo_to_ising(q)
    e_vec = IsingQAOA._energy_vector(h, J, chunk_size=31)
    envelope = feasible_envelope(e_vec, k, chunk_size=29)
    feasible = np.array([
        e for i, e in enumerate(e_vec) if i.bit_count() == k
    ])
    assert envelope["count"] == len(feasible)
    assert envelope["best"] == pytest.approx(float(feasible.min()))
    assert envelope["worst"] == pytest.approx(float(feasible.max()))


def test_instance_penalty_makes_ground_state_feasible():
    q, k, _ = make_portfolio_instance(10, seed=3310)
    h, J, _ = qubo_to_ising(q)
    e_vec = IsingQAOA._energy_vector(h, J)
    ground = int(np.argmin(e_vec))
    assert bitstring_of_index(ground, 10).count("1") == k


def test_best_feasible_qaoa_filters_infeasible_top_states():
    h = np.array([1.0, -0.5, 0.25])
    J = np.zeros((3, 3))
    result = {
        "bitstrings": {
            "111": 0.5,  # infeasible for k=1
            "001": 0.3,
            "010": 0.2,
        }
    }
    bitstring, energy = best_feasible_qaoa(
        result, h, J, offset=2.0, k=1)
    assert bitstring == "001"
    assert energy == pytest.approx(2.25)


def test_relative_gap_endpoints_and_infeasible():
    assert relative_gap(2.0, 2.0, 6.0) == pytest.approx(0.0)
    assert relative_gap(6.0, 2.0, 6.0) == pytest.approx(1.0)
    assert relative_gap(3.0, 2.0, 6.0) == pytest.approx(0.25)
    assert relative_gap(None, 2.0, 6.0) is None
