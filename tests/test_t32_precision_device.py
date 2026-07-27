"""Tests for Task 3.2 — complex64 dtype switch, per-layer gradient
checkpointing, torch_br device adapter, and the autograd memory model.

Tolerance rationale (complex64):
  float32 eps ~ 1.2e-7; the training trajectory accumulates rounding over
  p layers x n mixer steps, so trained parameters drift at the ~1e-5..1e-4
  level. After Stage 3 re-measurement (always complex128 via the
  unitarylab circuit) and the best-among-top-K selection, the final
  best_energy typically agrees to << 1e-3 relative. The 1e-3 bound is the
  same CPU/GPU consistency threshold the Biren spike handbook recommends
  for complex64, so passing it locally means the dtype switch itself adds
  no error beyond the accepted cross-device noise floor.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ising_qaoa import (IsingQAOA, estimate_evolve_memory, ising_energy,
                        resolve_device)


def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    J = np.triu(J, k=1)
    J = J + J.T
    return h, J


@pytest.fixture()
def algo(tmp_path):
    return IsingQAOA(algo_dir=str(tmp_path / "qaoa_out"))


HAS_TORCH_BR = False
try:  # pragma: no cover - environment dependent
    import torch_br  # noqa: F401
    HAS_TORCH_BR = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1. dtype switch: validation + defaults unchanged
# ---------------------------------------------------------------------------
def test_solve_invalid_dtype_raises(algo):
    h, J = make_instance(4, seed=200)
    with pytest.raises(ValueError, match="dtype"):
        algo.solve(h, J, dtype="float16")


def test_default_dtype_unchanged(algo):
    """Default solve() must remain the complex128 path (same trajectory)."""
    h, J = make_instance(5, seed=201)
    common = dict(layers=3, max_iter=40, seed=11)
    r1 = algo.solve(h, J, **common)
    r2 = algo.solve(h, J, dtype="complex128", **common)
    assert r1["energy_history"] == pytest.approx(r2["energy_history"])
    assert r1["best_bitstring"] == r2["best_bitstring"]


# ---------------------------------------------------------------------------
# 2. complex64 precision regression (n=9)
# ---------------------------------------------------------------------------
def test_complex64_matches_complex128_final_energy(algo):
    h, J = make_instance(9, seed=202)
    common = dict(layers=3, device="cpu", optimizer="autograd",
                  max_iter=80, seed=42)
    r128 = algo.solve(h, J, dtype="complex128", **common)
    r64 = algo.solve(h, J, dtype="complex64", **common)
    exact = r128["exact_energy"]
    assert exact == pytest.approx(r64["exact_energy"])
    scale = max(1.0, abs(exact))
    rel = abs(r64["best_energy"] - r128["best_energy"]) / scale
    assert rel < 1e-3, (
        f"complex64 best_energy {r64['best_energy']:.6f} vs complex128 "
        f"{r128['best_energy']:.6f}: rel diff {rel:.2e} >= 1e-3")
    # both must still respect the variational principle
    assert r64["best_energy"] >= exact - 1e-4
    assert r64["best_energy"] == pytest.approx(
        ising_energy(r64["best_bitstring"], h, J), abs=1e-6)


@pytest.mark.parametrize("dtype,tol", [("complex128", 1e-12),
                                       ("complex64", 1e-4)])
def test_evolve_unitarity_by_dtype(algo, dtype, tol):
    """Evolution must stay unitary: sum |psi|^2 == 1 within dtype noise.

    complex64 bound 1e-4: float32 eps 1.2e-7 accumulated over p*n
    two-by-two mixer updates; observed drift is O(1e-6).
    """
    h, J = make_instance(6, seed=203)
    params = np.random.default_rng(3).uniform(0.0, 0.5, size=8)
    cdtype = getattr(torch, dtype)
    psi = algo._evolve_torch(params, h, J, device="cpu", dtype=cdtype)
    assert psi.dtype == cdtype
    norm = float(torch.sum(psi.real**2 + psi.imag**2))
    assert abs(norm - 1.0) < tol, f"{dtype}: |psi|^2 sum = {norm!r}"


# ---------------------------------------------------------------------------
# 3. Gradient checkpointing: trajectory must be identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,rtol", [("complex128", 1e-9),
                                        ("complex64", 1e-4)])
def test_checkpoint_matches_no_checkpoint(algo, dtype, rtol):
    """Same seed/init: checkpoint=True recomputes layer forwards during
    backward but must not change the training trajectory or the result."""
    h, J = make_instance(6, seed=204)
    common = dict(layers=3, device="cpu", optimizer="autograd",
                  max_iter=50, seed=7, dtype=dtype)
    r_plain = algo.solve(h, J, checkpoint=False, **common)
    r_ckpt = algo.solve(h, J, checkpoint=True, **common)
    assert r_ckpt["energy_history"] == pytest.approx(
        r_plain["energy_history"], rel=rtol, abs=1e-12)
    assert r_ckpt["best_bitstring"] == r_plain["best_bitstring"]
    assert r_ckpt["best_energy"] == pytest.approx(r_plain["best_energy"],
                                                  rel=rtol)


def test_checkpoint_gradient_matches_direct(algo):
    """Gradients through the checkpointed graph equal the direct graph."""
    h, J = make_instance(5, seed=205)
    e_vec_t = torch.as_tensor(algo._energy_vector(h, J),
                              dtype=torch.float64)
    init = np.random.default_rng(9).uniform(0.0, 0.5, size=6)

    grads = {}
    for ckpt in (False, True):
        p = torch.as_tensor(init, dtype=torch.float64).clone()
        p.requires_grad_(True)
        psi = algo._evolve_torch(p, h, J, device="cpu", e_vec=e_vec_t,
                                 checkpoint=ckpt)
        energy = torch.sum((psi.real**2 + psi.imag**2) * e_vec_t)
        energy.backward()
        grads[ckpt] = p.grad.detach().clone()
    torch.testing.assert_close(grads[True], grads[False],
                               rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. resolve_device adapter
# ---------------------------------------------------------------------------
def test_resolve_device_cpu(algo):
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_string(algo):
    # must not require CUDA hardware — string resolution only
    assert resolve_device("cuda:0") == torch.device("cuda:0")


@pytest.mark.skipif(HAS_TORCH_BR, reason="torch_br present: error path n/a")
def test_resolve_device_biren_without_torch_br(algo):
    with pytest.raises(ImportError, match="torch_br"):
        resolve_device("biren")


@pytest.mark.skipif(not HAS_TORCH_BR, reason="torch_br only on Biren hosts")
def test_resolve_device_biren_with_torch_br(algo):
    dev = resolve_device("biren")
    assert dev.type == "supa"


def test_resolve_device_invalid(algo):
    with pytest.raises(Exception):
        resolve_device("not-a-device")


# ---------------------------------------------------------------------------
# 5. Autograd memory model (CPU proxy for the unavailable 32GB card)
# ---------------------------------------------------------------------------
GiB = 2**30


def test_memory_model_n24_acceptance(algo):
    """n=24, p=4: complex128/no-ckpt must blow the 32GB card (motivation),
    complex64+ckpt must fit the <8GB acceptance target."""
    m_bad = estimate_evolve_memory(24, 4, dtype="complex128",
                                   checkpoint=False)
    assert m_bad["total_bytes"] > 32 * GiB, (
        f"model says complex128/no-ckpt n=24 p=4 needs "
        f"{m_bad['total_bytes']/GiB:.1f} GiB — expected > 32 GiB")
    m_good = estimate_evolve_memory(24, 4, dtype="complex64",
                                    checkpoint=True)
    assert m_good["total_bytes"] < 8 * GiB, (
        f"model says complex64+ckpt n=24 p=4 peaks at "
        f"{m_good['total_bytes']/GiB:.2f} GiB — misses the <8GB target")


def test_memory_model_checkpoint_monotonic(algo):
    for dtype in ("complex128", "complex64"):
        plain = estimate_evolve_memory(16, 4, dtype=dtype,
                                       checkpoint=False)
        ckpt = estimate_evolve_memory(16, 4, dtype=dtype,
                                      checkpoint=True)
        assert ckpt["total_bytes"] < plain["total_bytes"]
        # retained activations must shrink to ~one layer's worth
        assert ckpt["activation_bytes"] == pytest.approx(
            plain["activation_bytes"] / 4)


def test_memory_model_dtype_halves_state(algo):
    m128 = estimate_evolve_memory(20, 2, dtype="complex128")
    m64 = estimate_evolve_memory(20, 2, dtype="complex64")
    # statevector bytes: 2^20 * 16 vs 2^20 * 8
    assert m128["state_bytes"] == 2 * m64["state_bytes"]
