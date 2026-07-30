"""IsingQAOA — weighted-Ising QAOA solver with local fields.

Extends unitarylab's MaxCut-specific ``QAOAAlgorithm`` into a general
QUBO/Ising solver supporting arbitrary weighted couplings J_ij and local
fields h_i.

Hamiltonian
-----------
    H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

(J is assumed symmetric with zero diagonal; each pair is counted once via
the upper triangle i<j.)

Bitstring encoding convention
-----------------------------
The unitarylab statevector backend orders basis states with **qubit 0 as
the least significant bit** of the state index (verified empirically:
X on qubit i puts the amplitude at index 2**i). A measured basis state
is reported as a bitstring ``s`` of length n with

    s[i] = measurement outcome x_i of qubit i, i.e.
    state_index = sum_i x_i * 2**i  (= int(s[::-1], 2))

so bitstring position i always refers to qubit i.

Spin mapping (used consistently everywhere in this module):

    x_i = int(s[i]) in {0, 1};   z_i = 1 - 2 * x_i
    (bit '0' -> z = +1, bit '1' -> z = -1)

Ising energy of a basis state:

    E(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j

Cost unitary per QAOA layer (gamma):
    weighted ZZ term  J_ij Z_i Z_j :  cx(i, j) -> rz(2*gamma*J_ij, j) -> cx(i, j)
    local field term  h_i Z_i      :  rz(2*gamma*h_i, i)
Mixer per layer (beta) is unchanged from the base class: rx(2*beta) on
every qubit.

Training paths
--------------
``optimizer='autograd'`` (default): Adam on a torch-native statevector
evolution (``_evolve_torch``) that applies exactly the same unitary as
``_build_circuit``, so gradients backpropagate through the whole circuit.
This is the main path for GPU scaling. (unitarylab's compiled torch
executor bakes gate angles into constant matrices at circuit-build time
and therefore cannot propagate gradients through gate parameters; the
torch-native evolution is the faithful differentiable twin of the
circuit, verified against ``qc.execute`` in the test suite.)

``optimizer='cobyla'``: scipy COBYLA on the unitarylab circuit itself
(``qc.execute(backend='torch', device=device)``), kept as a
cross-validation path for small n.
"""
import time

import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from threadpoolctl import threadpool_limits

from unitarylab import Circuit
from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm

# Number of top-probability bitstrings returned in result['bitstrings'].
TOP_K = 16
# Internal random restarts for the autograd path (best expectation kept).
# All restarts draw from the same seeded RNG, so results stay reproducible.
N_RESTARTS = 3

# dtype switch: name -> (complex state dtype, real dtype for params/e_vec).
# complex64 keeps e_vec/params in float32: Ising energies are O(n * max|J|)
# (~1e2 at n=28), so float32's relative eps 1.2e-7 gives absolute energy
# error ~1e-5 — far below the 1e-3 CPU/GPU consistency threshold the Biren
# spike handbook recommends for complex64; the statevector phases are the
# precision-critical part and halving them halves the dominant memory term.
_PRECISION = {
    "complex128": (torch.complex128, torch.float64),
    "complex64": (torch.complex64, torch.float32),
}

# unitarylab's compiled executor names the Biren SUPA device 'gpu' (see
# GPU使用对接.md §3); torch_br tensors themselves live on device 'supa'.
_UNITARYLAB_DEVICE = {"biren": "gpu", "supa": "gpu"}


def _apply_mixer(psi: torch.Tensor, beta: torch.Tensor, n: int,
                 adjoint: bool = False) -> torch.Tensor:
    """Apply ``exp(-i beta sum_i X_i)`` (or its adjoint) to a flat state.

    This helper deliberately performs one qubit at a time.  During a
    grad-free forward/backward it needs only a constant number of full
    state buffers, unlike eager autograd which retains buffers for all
    ``n`` mixer gates.
    """
    c = torch.cos(beta)
    s = torch.sin(beta)
    zero = torch.zeros_like(c)
    sign = 1.0 if adjoint else -1.0
    offdiag = torch.complex(zero, sign * s)

    t = psi
    for k in range(n):
        # LSB convention: for qubit k, consecutive groups of 2^(k+1)
        # amplitudes split into the x_k=0 and x_k=1 halves. Keep this view
        # rank-3: torch_br/SUPA cannot represent a (2,)*n shape at n=24.
        pair = t.reshape(-1, 2, 1 << k)
        zero_row = c * pair[:, 0, :] + offdiag * pair[:, 1, :]
        one_row = offdiag * pair[:, 0, :] + c * pair[:, 1, :]
        next_t = torch.stack((zero_row, one_row), dim=1).reshape(-1)
        # pair keeps the previous full state alive; row temporaries keep
        # another full state between them. Drop all three before the next
        # qubit so SUPA's allocator can reuse the same buffers.
        del pair, zero_row, one_row
        t = next_t
    return t


def _sum_x_actions(psi: torch.Tensor, n: int) -> torch.Tensor:
    """Return ``sum_i X_i |psi>`` with constant-state scratch memory."""
    out = torch.zeros_like(psi)
    for k in range(n):
        pair = psi.reshape(-1, 2, 1 << k)
        out_pair = out.reshape(-1, 2, 1 << k)
        out_pair[:, 0, :].add_(pair[:, 1, :])
        out_pair[:, 1, :].add_(pair[:, 0, :])
    return out


class _AdjointQAOALayer(torch.autograd.Function):
    """One QAOA layer with an analytic, constant-scratch backward.

    For ``U(beta)=exp(-i beta sum X_i)`` and
    ``C(gamma)=exp(-i gamma H_C)``, the layer is ``U C``.  Its parameter
    derivatives are generated by ``sum X_i`` and ``H_C`` respectively,
    while the state gradient is obtained by applying ``C† U†``.  Saving
    only the layer input/output avoids retaining every single-qubit mixer
    activation; backend allocator/cache overhead remains platform-specific.
    """

    @staticmethod
    def forward(ctx, psi, gamma, beta, e_vec, n):
        phase = torch.complex(torch.cos(gamma * e_vec),
                              -torch.sin(gamma * e_vec))
        cost_state = psi * phase
        del phase
        out = _apply_mixer(cost_state, beta, n)
        ctx.n = n
        ctx.save_for_backward(psi, out, gamma, beta, e_vec)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        psi, out, gamma, beta, e_vec = ctx.saved_tensors
        n = ctx.n

        # d/dbeta U(beta)|v> = -i (sum_i X_i) U(beta)|v>.
        sum_x_out = _sum_x_actions(out, n)
        sum_x_out.mul_(complex(0.0, -1.0))
        grad_beta = torch.real(
            torch.sum(torch.conj(grad_out) * sum_x_out))
        del sum_x_out

        # Pull the cotangent through U first.  At the cost-layer output,
        # d/dgamma C(gamma)|psi> = -i H_C C(gamma)|psi>.
        grad_cost = _apply_mixer(grad_out, beta, n, adjoint=True)
        phase = torch.complex(torch.cos(gamma * e_vec),
                              -torch.sin(gamma * e_vec))
        cost_state = psi * phase
        cost_state.mul_(e_vec)
        cost_state.mul_(complex(0.0, -1.0))
        grad_gamma = torch.real(
            torch.sum(torch.conj(grad_cost) * cost_state))
        del cost_state
        grad_psi = grad_cost * phase.conj()
        return grad_psi, grad_gamma, grad_beta, None, None


def resolve_device(device_str) -> torch.device:
    """Resolve a user device string to a torch.device.

    'cpu' / 'cuda[:i]' pass straight through to ``torch.device``.
    'biren' (or 'supa') targets the Biren SUPA stack: it requires
    ``torch_br`` (only installed in the Biren container) and resolves to
    ``torch.device('supa')`` — the device torch_br tensors are created on
    (GPU使用对接.md §3/§8). Raises ImportError with an actionable message
    when torch_br is unavailable, so local (non-Biren) runs fail clearly
    at the call site instead of deep inside tensor allocation.
    """
    if isinstance(device_str, torch.device):
        device_str = str(device_str)
    name = str(device_str).lower()
    if name in ("biren", "supa") or name.startswith("supa:"):
        try:
            import torch_br  # noqa: F401  (registers the 'supa' backend)
        except ImportError as exc:
            raise ImportError(
                f"device '{device_str}' requires the Biren SUPA torch "
                "backend (import torch_br), which is only installed in the "
                "Biren container (source brsw_set_env.sh first, see "
                "GPU使用对接.md §3); run on the Biren host or use "
                "device='cpu'/'cuda' locally") from exc
        return torch.device("supa" if name == "biren" else name)
    return torch.device(device_str)


def estimate_evolve_memory(n: int, layers: int, dtype: str = "complex128",
                           checkpoint: bool = False,
                           adjoint: bool = False) -> dict:
    """Analytic peak-memory model for one _train_autograd fwd+bwd step.

    Model (dim d = 2^n, c = bytes/complex entry, r = bytes/real entry):
      - e_vec:                     d * r              (persistent input)
      - per-layer retained activations, no checkpoint:
          mixer per qubit: movedim().reshape(2,-1) copies (non-contiguous
            view) and each matmul output is saved for backward
            -> 2 complex buffers per qubit per layer  = 2n * d * c
          cost: product buffer d * r + phase d * c + layer-input psi d * c
          => per layer (2n + 2) * d * c + d * r, times p layers
      - checkpoint=True: only the p+1 layer-boundary states are retained
          ((p+1) * d * c); during backward each layer's activations are
          recomputed and freed one layer at a time, adding at most one
          layer's worth: (2n + 2) * d * c + d * r
      - adjoint=True: a custom analytic backward stores only the p+1 layer
          boundary states and uses at most four state buffers as scratch
          while applying U† and the parameter generators
      - backward grad of the live statevector: d * c
    Numbers are retained-activation estimates (no allocator/fragmentation
    overhead); the acceptance margin absorbs that.
    """
    if dtype not in _PRECISION:
        raise ValueError(f"unknown dtype '{dtype}', "
                         f"expected one of {sorted(_PRECISION)}")
    cbytes = 16 if dtype == "complex128" else 8
    rbytes = cbytes // 2
    d = 1 << n
    state_bytes = d * cbytes
    e_vec_bytes = d * rbytes
    per_layer = (2 * n + 2) * state_bytes + e_vec_bytes
    if adjoint:
        boundary_bytes = (layers + 1) * state_bytes
        activation_bytes = 4 * state_bytes
    elif checkpoint:
        boundary_bytes = (layers + 1) * state_bytes
        activation_bytes = per_layer  # one layer recomputed at a time
    else:
        boundary_bytes = 0
        activation_bytes = layers * per_layer
    grad_bytes = state_bytes
    total = e_vec_bytes + boundary_bytes + activation_bytes + grad_bytes
    return {
        "n": n, "layers": layers, "dtype": dtype, "checkpoint": checkpoint,
        "adjoint": adjoint,
        "state_bytes": state_bytes, "e_vec_bytes": e_vec_bytes,
        "boundary_bytes": boundary_bytes,
        "activation_bytes": activation_bytes,
        "grad_bytes": grad_bytes, "total_bytes": total,
    }


def ising_energy(bitstring: str, h: np.ndarray, J: np.ndarray) -> float:
    """Ising energy of one bitstring under the module convention.

    z_i = 1 - 2*int(bitstring[i]);  E = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
    """
    n = len(h)
    assert len(bitstring) == n
    z = 1.0 - 2.0 * np.array([int(b) for b in bitstring], dtype=np.float64)
    e = float(h @ z)
    iu = np.triu_indices(n, k=1)
    e += float(np.sum(J[iu] * z[iu[0]] * z[iu[1]]))
    return e


def bitstring_of_index(idx: int, n: int) -> str:
    """Basis-state index -> bitstring under the module convention.

    Qubit 0 is the least significant bit of the statevector index, so
    s[i] = str((idx >> i) & 1).
    """
    return "".join(str((idx >> i) & 1) for i in range(n))


class IsingQAOA(QAOAAlgorithm):
    """QAOA solver for general weighted Ising models with local fields."""

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------
    @staticmethod
    def _energy_vector(h: np.ndarray, J: np.ndarray,
                       chunk_size: int | None = None) -> np.ndarray:
        """E(z) for every computational basis state (diag of H_C).

        Computed in row blocks of ``chunk_size`` (default 2^20) so no
        (2^n, n) spin table and no (2^n, nnz) pair matrix is ever
        materialized: inside a block the spin signs are generated directly
        by bit shifts and the coupling term is evaluated as
        0.5 * rowsum(z * (z @ J)) (J symmetric, zero diagonal, so
        z'Jz = 2 * sum_{i<j} J_ij z_i z_j). Peak scratch memory is
        O(chunk_size * n) — at the default chunk and n=28 that is ~700MB,
        independent of 2^n.
        """
        n = len(h)
        dim = 1 << n
        if chunk_size is None:
            chunk_size = min(dim, 1 << 20)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        e = np.empty(dim, dtype=np.float64)
        shifts = np.arange(n, dtype=np.int64)
        # Biren's container ships OpenBLAS 0.3.20 configured for 64 threads.
        # Its very tall-skinny GEMM (2^20 x n @ n x n) was observed to
        # silently corrupt rows after the first block at n=24. One BLAS
        # thread is both correct and faster for these tiny inner dimensions;
        # lock it locally so callers do not need process-wide env settings.
        with threadpool_limits(limits=1, user_api="blas"):
            for start in range(0, dim, chunk_size):
                m = min(chunk_size, dim - start)
                idx = start + np.arange(m, dtype=np.int64)
                # (m, n) block of z values, qubit 0 = LSB (module convention)
                z = 1.0 - 2.0 * (
                    (idx[:, None] >> shifts[None, :]) & 1
                ).astype(np.float64)
                e[start:start + m] = (
                    z @ h + 0.5 * np.sum(z * (z @ J), axis=1))
        return e

    def _get_h_cost(self, h: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Dense (diagonal) cost Hamiltonian — SMALL-n TEST VALIDATION ONLY.

        Entry (idx, idx) is E(z) of basis state idx under the module
        bitstring convention. Raises for n > 12: the dense (2^n, 2^n)
        complex matrix is ~1GB already at n=13 and 68GB at n=16; the
        solve() path never needs it (H_C is diagonal — use
        ``_energy_vector`` for its diagonal).
        """
        h = np.asarray(h, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
        if len(h) > 12:
            raise ValueError(
                f"_get_h_cost builds a dense (2^n, 2^n) matrix and is "
                f"intended for small-n test validation only (n <= 12, got "
                f"n={len(h)}); the solve path never needs the dense matrix. "
                f"Use _energy_vector(h, J) for the diagonal of H_C.")
        e_vec = self._energy_vector(h, J)
        return np.diag(e_vec).astype(np.complex128)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self, params: np.ndarray, h: np.ndarray, J: np.ndarray) -> Circuit:
        """Weighted-Ising QAOA circuit.

        Per layer i (gamma = params[i], beta = params[p+i]):
          - for each i<j with J_ij != 0: cx(i,j) -> rz(2*gamma*J_ij, j) -> cx(i,j)
          - for each i with h_i != 0:   rz(2*gamma*h_i, i)
          - mixer: rx(2*beta) on all qubits
        """
        n = len(h)
        p = len(params) // 2
        gammas, betas = params[:p], params[p:]

        qc = Circuit(n)
        for i in range(n):
            qc.h(i)
        iu = np.triu_indices(n, k=1)
        edges = [(int(i), int(j), float(J[i, j])) for i, j in zip(*iu) if J[i, j] != 0.0]
        fields = [(int(i), float(h[i])) for i in range(n) if h[i] != 0.0]
        for layer in range(p):
            g = float(gammas[layer])
            for i, j, w in edges:
                qc.cx(i, j)
                qc.rz(2.0 * g * w, j)
                qc.cx(i, j)
            for i, hi in fields:
                qc.rz(2.0 * g * hi, i)
            b = float(betas[layer])
            for i in range(n):
                qc.rx(2.0 * b, i)
        return qc

    # ------------------------------------------------------------------
    # Torch-native differentiable evolution (autograd training path)
    # ------------------------------------------------------------------
    def _evolve_torch(self, params, h: np.ndarray, J: np.ndarray,
                      device: str = "cpu", e_vec=None,
                      dtype: torch.dtype = torch.complex128,
                      checkpoint: bool = False,
                      adjoint: bool = False) -> torch.Tensor:
        """Statevector after the QAOA circuit, as a torch tensor.

        Applies exactly the same unitary as ``_build_circuit``:
        |+>^n -> per layer [ exp(-i*gamma*H_C) ; prod_i RX(2*beta, i) ].
        H_C is diagonal, so the cost layer is an exact elementwise phase.
        Differentiable w.r.t. ``params`` if it requires grad.

        ``params``: array-like of length 2*layers (numpy or torch tensor).
        ``e_vec``: precomputed energy vector (numpy array or torch tensor).
            Pass the same tensor every call to avoid recomputation and
            numpy->torch transfers inside a training loop. ``None``
            (default) recomputes it from (h, J) — kept as the default so
            existing standalone callers/tests keep working without
            signature changes.
        ``dtype``: torch.complex128 (default, pre-T3.2 behavior) or
            torch.complex64; params/e_vec are cast to the matching real
            dtype (float64/float32 — see _PRECISION for the float32
            precision justification).
        ``checkpoint``: if True, wrap each QAOA layer (cost + mixer) in
            ``torch.utils.checkpoint`` (use_reentrant=False) so only the
            layer-boundary states are retained and each layer's forward
            is recomputed during backward; the trajectory is unchanged
            (recompute is deterministic). Only active when params
            requires grad — inference calls run the plain forward.
        ``adjoint``: if True, use an analytic custom backward for each
            layer. It stores only layer-boundary states and applies the
            inverse mixer/cost plus their generators during backward.
            This is gradient-equivalent to eager autograd but reduces
            checkpoint backward scratch from O(n * 2^n) to O(2^n).
        Returns a complex tensor of ``dtype`` and shape (2^n,) on
        ``device``.
        """
        cdtype = dtype
        rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
        n = len(h)
        p = len(params) // 2
        if not isinstance(params, torch.Tensor):
            params = torch.as_tensor(np.asarray(params, dtype=np.float64))
        params = params.to(device=device, dtype=rdtype)
        gammas, betas = params[:p], params[p:]

        if e_vec is None:
            e_vec = self._energy_vector(h, J)
        e_vec = torch.as_tensor(e_vec, dtype=rdtype, device=device)
        dim = 2**n

        def layer_fn(psi_in, g, b):
            # cost layer: diagonal phase exp(-i * gamma * E(z)), built in
            # the working precision (torch.complex of real-dtype parts)
            ang = g * e_vec
            phase = torch.complex(torch.cos(ang), -torch.sin(ang))
            return _apply_mixer(psi_in * phase, b, n)

        # |+>^n
        psi = torch.full((dim,), 2.0 ** (-n / 2),
                         dtype=cdtype, device=device)
        # Checkpointing only matters when building a graph; recomputing a
        # grad-free forward would just waste time.
        use_ckpt = (checkpoint and torch.is_grad_enabled()
                    and params.requires_grad)
        use_adjoint = (adjoint and torch.is_grad_enabled()
                       and params.requires_grad)
        for layer in range(p):
            if use_adjoint:
                psi = _AdjointQAOALayer.apply(
                    psi, gammas[layer], betas[layer], e_vec, n)
            elif use_ckpt:
                psi = _torch_checkpoint(layer_fn, psi, gammas[layer],
                                        betas[layer], use_reentrant=False)
            else:
                psi = layer_fn(psi, gammas[layer], betas[layer])
        return psi

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def run(self, *args, **kwargs):
        """The base-class MaxCut run(edges, n) interface is not supported."""
        raise NotImplementedError(
            "IsingQAOA does not support the MaxCut-oriented base-class "
            "run(edges, n) interface; use solve(h, J, ...) instead.")

    @staticmethod
    def _validate(h, J, optimizer):
        h = np.asarray(h, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
        if h.ndim != 1:
            raise ValueError("h must be a 1-D array of shape (n,)")
        n = len(h)
        if J.shape != (n, n):
            raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}")
        if not np.allclose(J, J.T, atol=1e-12):
            raise ValueError("J must be symmetric")
        if not np.allclose(np.diag(J), 0.0, atol=1e-12):
            raise ValueError("J must have zero diagonal")
        if optimizer not in ("autograd", "cobyla"):
            raise ValueError(f"unknown optimizer '{optimizer}', "
                             "expected 'autograd' or 'cobyla'")
        return h, J, n

    def _train_autograd(self, initial_params, h, J, layers, device,
                        max_iter, energy_history, e_vec=None,
                        dtype: torch.dtype = torch.complex128,
                        checkpoint: bool = False,
                        adjoint: bool = False):
        """Adam on the torch-native evolution; returns BEST-SEEN params.

        Tracks the lowest expectation energy over the whole trajectory
        (not the last iterate — Adam with a fixed lr does not converge
        monotonically) and returns the parameters that realized it.

        ``e_vec``: precomputed energy vector as a torch tensor on
        ``device`` (pass the same tensor across calls so the training loop
        does zero numpy->torch transfers and zero recomputation). ``None``
        recomputes it once here (compat path for standalone callers).
        ``dtype``/``checkpoint``/``adjoint``: forwarded to
        ``_evolve_torch``.

        Memory model for one fwd+bwd step (d = 2^n, c = bytes per complex
        entry of ``dtype``, r = c/2; same model as estimate_evolve_memory):
          - persistent: e_vec d*r, params/Adam state O(p)
          - checkpoint=False: every layer retains its full graph —
            mixer 2 complex buffers per qubit (movedim-reshape copy +
            matmul output saved for backward) + cost (phase d*c, product
            d*r, layer-input psi d*c) -> p * ((2n+2)*d*c + d*r)
          - checkpoint=True: only p+1 layer-boundary states retained
            ((p+1)*d*c); each layer's activations are recomputed one layer
            at a time during backward -> peak add (2n+2)*d*c + d*r
          - backward grad of the live state: d*c
        Anchor numbers (n=24, p=4, d*c complex128 = 268MB / complex64 =
        134MB): complex128 no-ckpt ~54.6GB (exceeds the 32GB Biren106M —
        this was final-review landmine #2); complex64 + ckpt ~7.7GB
        (boundaries 0.67GB + one-layer recompute 6.78GB + e_vec 0.06GB +
        grad 0.13GB) < 8GB acceptance target.
        """
        cdtype = dtype
        rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
        if e_vec is None:
            e_vec = torch.as_tensor(self._energy_vector(h, J),
                                    dtype=rdtype, device=device)
        params = torch.as_tensor(initial_params, dtype=rdtype,
                                 device=device).clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=0.05)
        best_energy = np.inf
        best_params = None
        for _ in range(max_iter):
            opt.zero_grad()
            psi = self._evolve_torch(params, h, J, device=device,
                                     e_vec=e_vec, dtype=cdtype,
                                     checkpoint=checkpoint,
                                     adjoint=adjoint)
            probs = psi.real**2 + psi.imag**2
            energy = torch.sum(probs * e_vec)
            energy.backward()
            # record BEFORE opt.step(): e_val is the energy at the current
            # params, so the best-seen snapshot must be taken pre-step
            e_val = float(energy.detach().cpu())
            energy_history.append(e_val)
            if e_val < best_energy:
                best_energy = e_val
                # clone: params is mutated in place by opt.step()
                best_params = params.detach().clone()
            opt.step()
        if best_params is None:  # max_iter == 0 edge case
            best_params = params.detach()
            best_energy = np.inf
        return best_params.cpu().numpy(), float(best_energy)

    def _train_cobyla(self, initial_params, h, J, layers, device,
                      max_iter, energy_history, e_vec=None):
        if e_vec is None:
            e_vec = self._energy_vector(h, J)

        def obj_func(p_flat):
            qc = self._build_circuit(p_flat, h, J)
            psi_out = qc.execute(backend="torch", device=device).state
            psi = np.asarray(psi_out, dtype=np.complex128).flatten()
            probs = np.abs(psi) ** 2
            energy = float(probs @ e_vec)
            energy_history.append(energy)
            return energy

        opt_res = minimize(obj_func, x0=initial_params, method="COBYLA",
                           options={"maxiter": max_iter})
        return np.asarray(opt_res.x, dtype=np.float64)

    @staticmethod
    def _interp_extend(params: np.ndarray) -> np.ndarray:
        """INTERP heuristic (Zhou et al. 2020): extend p-layer optima to
        (p+1)-layer initial values by linear interpolation.

        For each of the gamma and beta halves, with the sequence extended
        by zeros at both ends (v_0 = v_{p+1} = 0):

            v'_i = w * v_{i-1} + (1 - w) * v_i,   w = (i-1)/p,  i = 1..p+1

        (endpoints are preserved: v'_1 = v_1, v'_{p+1} = v_p).
        """
        params = np.asarray(params, dtype=np.float64)
        p = len(params) // 2
        out = np.empty(2 * (p + 1), dtype=np.float64)
        w = np.arange(p + 1) / p  # w[i-1] for i = 1..p+1
        for off, src in ((0, params[:p]), (p + 1, params[p:])):
            ext = np.concatenate(([0.0], src, [0.0]))
            out[off:off + p + 1] = w * ext[:-1] + (1.0 - w) * ext[1:]
        return out

    def solve(self, h: np.ndarray, J: np.ndarray, layers: int = 4,
              device: str = "cpu", optimizer: str = "autograd",
              max_iter: int = 200, seed: int = 42,
              init: str = "random", dtype: str = "complex128",
              checkpoint: bool = False, adjoint: bool = False,
              final_backend: str = "auto",
              energy_vector: np.ndarray | None = None,
              top_k: int = TOP_K) -> dict:
        """Solve a weighted Ising model with QAOA.

        Parameters:
            h: (n,) local field coefficients
            J: (n,n) symmetric coupling matrix with zero diagonal
            layers: QAOA depth p
            device: 'cpu', a torch device string (e.g. 'cuda:0'), or
                'biren'/'supa' for the Biren SUPA stack (resolved via
                resolve_device; requires torch_br, only present in the
                Biren container). Torch tensors go to the resolved device;
                unitarylab circuit executions (cobyla path and Stage 3)
                map biren/supa -> 'gpu'.
            optimizer: 'autograd' (Adam, torch backprop through the
                circuit) or 'cobyla' (scipy COBYLA cross-check path)
            max_iter: optimizer iterations per trained level
            seed: RNG seed for parameter initialization (fully reproducible)
            init: 'random' (default; seeded uniform small-parameter inits
                with N_RESTARTS restarts) or 'interp' (INTERP heuristic:
                train p=1 first, then grow one layer at a time, each level
                warm-started from the previous level's best parameters via
                _interp_extend; the p=1 level uses the same seeded restarts
                as 'random')
            dtype: 'complex128' (default, pre-T3.2 behavior) or
                'complex64' for the autograd training path. In complex64
                the statevector/evolution run in complex64 while e_vec and
                parameters use float32 (justification in _PRECISION:
                energies are O(n*max|J|) ~ 1e2, so float32's 1.2e-7
                relative eps is ~1e-5 absolute — far below the 1e-3
                cross-device consistency threshold). Stage 3 final
                measurement always re-runs in complex128 via the
                unitarylab circuit, so reported probabilities/bitstrings
                keep the float64 bit-exact anchor; dtype only affects the
                training trajectory.
            checkpoint: if True, per-layer gradient checkpointing on the
                autograd path (see _train_autograd docstring memory
                model); trajectory-identical, trades one extra forward
                recompute per layer for the activation memory.
            adjoint: if True, use the analytic per-layer adjoint backward.
                This is gradient-equivalent to eager autograd and has
                constant-state scratch, substantially reducing retained
                activations for large statevectors.
                It may be combined with ``checkpoint=True``; adjoint takes
                precedence because it has the lower memory bound.
            final_backend: 'auto' (native torch for n>20 or Biren/SUPA,
                otherwise the unitarylab circuit), 'native', or
                'unitarylab'. Native mode extracts top-K on-device and
                avoids copying a multi-GiB statevector to host memory.
            energy_vector: optional precomputed E(z) array of shape
                ``(2**n,)``. Large experiment sweeps can compute it once
                and reuse it across depths.
            top_k: number of highest-probability states to return. The
                default remains 16; cardinality-postselected experiments
                may request a larger pool without transferring the full
                probability vector to host.

        Returns:
            dict with keys: bitstrings (top-K bitstring -> probability),
            best_bitstring (lowest-energy bitstring among the top-K most
            probable outcomes), best_energy, exact_energy (ground-state
            energy; H is strictly diagonal so this is min of the energy
            vector, no dense diagonalization), energy_history (all
            restarts and, for init='interp', all levels concatenated in
            training order), n_qubits, layers, device, optimizer, dtype,
            checkpoint, adjoint, final_backend.
        """
        h, J, n = self._validate(h, J, optimizer)
        if init not in ("random", "interp"):
            raise ValueError(f"unknown init '{init}', "
                             "expected 'random' or 'interp'")
        if dtype not in _PRECISION:
            raise ValueError(f"unknown dtype '{dtype}', "
                             f"expected one of {sorted(_PRECISION)}")
        if final_backend not in ("auto", "native", "unitarylab"):
            raise ValueError(
                f"unknown final_backend '{final_backend}', expected one "
                f"of ['auto', 'native', 'unitarylab']")
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        cdtype, rdtype = _PRECISION[dtype]
        tdevice = resolve_device(device)
        udevice = _UNITARYLAB_DEVICE.get(str(device).lower(), device)
        rng = np.random.default_rng(seed)

        self.log(f"Stage 1: Computing exact ground-state energy (n={n})")
        # H is strictly diagonal in the computational basis, so the exact
        # ground-state energy is simply min over the energy vector; this
        # avoids the dense (2^n, 2^n) eigvalsh which OOMs at n=16.
        # The vector is computed ONCE here (chunked, bounded memory) and
        # reused by every training call and by Stage 3 below.
        if energy_vector is None:
            e_vec = self._energy_vector(h, J)
        else:
            e_vec = np.asarray(energy_vector, dtype=np.float64)
            if e_vec.shape != (1 << n,):
                raise ValueError(
                    f"energy_vector must have shape ({1 << n},), got "
                    f"{e_vec.shape}")
        exact_energy = float(e_vec.min())
        # Torch copy moved to the training device once; the training loop
        # itself does zero numpy->torch transfers and zero recomputation.
        e_vec_t = (torch.as_tensor(e_vec, dtype=rdtype, device=tdevice)
                   if optimizer == "autograd" else None)

        self.log(f"Stage 2: Variational optimization "
                 f"({optimizer}, layers={layers}, max_iter={max_iter}, "
                 f"init={init}, dtype={dtype}, checkpoint={checkpoint}, "
                 f"adjoint={adjoint})")
        energy_history: list = []
        q_start = time.time()

        def train_level(initial_params, level):
            if optimizer == "autograd":
                return self._train_autograd(
                    initial_params, h, J, level, tdevice, max_iter,
                    energy_history, e_vec=e_vec_t, dtype=cdtype,
                    checkpoint=checkpoint, adjoint=adjoint)
            params = self._train_cobyla(
                initial_params, h, J, level, udevice, max_iter,
                energy_history, e_vec=e_vec)
            return params, energy_history[-1]

        if init == "interp":
            # p=1: same seeded restarts as the random path.
            final_params, best_exp = None, np.inf
            for _ in range(N_RESTARTS):
                if optimizer == "autograd":
                    init_p = rng.uniform(0.0, 0.5, size=2)
                else:
                    init_p = rng.uniform(0.0, np.pi, size=2)
                params_r, exp_r = train_level(init_p, 1)
                if exp_r < best_exp or final_params is None:
                    best_exp, final_params = exp_r, params_r
            # Grow one layer at a time, warm-started from INTERP-extended
            # best parameters of the previous level.
            for level in range(2, layers + 1):
                init_p = self._interp_extend(final_params)
                final_params, best_exp = train_level(init_p, level)
            final_exp = best_exp
        elif optimizer == "autograd":
            # Small-parameter inits avoid the poor plateaus of uniform(0, pi)
            # inits; a few seeded restarts, keep the best expectation.
            # Note: energy_history concatenates all restarts.
            final_params, best_exp = None, np.inf
            for _ in range(N_RESTARTS):
                init_r = rng.uniform(0.0, 0.5, size=2 * layers)
                params_r, exp_r = train_level(init_r, layers)
                if exp_r < best_exp or final_params is None:
                    best_exp, final_params = exp_r, params_r
            final_exp = best_exp
        else:
            initial_params = rng.uniform(0.0, np.pi, size=2 * layers)
            final_params, final_exp = train_level(initial_params, layers)
        q_time = time.time() - q_start
        self.log(f"  optimization time: {q_time:.3f}s, "
                 f"final expectation: {final_exp:.6f}")

        k = min(int(top_k), 2**n)
        chosen_final_backend = final_backend
        if chosen_final_backend == "auto":
            device_name = str(device).lower()
            chosen_final_backend = (
                "native" if optimizer == "autograd"
                and (n > 20 or device_name in ("biren", "supa")
                     or device_name.startswith("supa:"))
                else "unitarylab")

        self.log("Stage 3: Final state measurement and bitstring decoding "
                 f"({chosen_final_backend})")
        if chosen_final_backend == "native":
            with torch.no_grad():
                final_params_t = torch.as_tensor(
                    final_params, dtype=rdtype, device=tdevice)
                psi_t = self._evolve_torch(
                    final_params_t, h, J, device=tdevice, e_vec=e_vec_t,
                    dtype=cdtype, checkpoint=False, adjoint=False)
                probs_t = psi_t.real.square() + psi_t.imag.square()
                probs_t = probs_t / probs_t.sum()
                top_prob_t, top_idx_t = torch.topk(
                    probs_t, k=k, largest=True, sorted=True)
                top_idx = top_idx_t.cpu().numpy().astype(np.int64)
                top_prob = top_prob_t.cpu().numpy().astype(np.float64)
            bitstrings = {
                bitstring_of_index(int(i), n): float(prob)
                for i, prob in zip(top_idx, top_prob)}
        else:
            # Small-n convention anchor through unitarylab. Large runs use
            # native mode above so a multi-GiB state never crosses to host.
            qc_final = self._build_circuit(final_params, h, J)
            psi_out = qc_final.execute(backend="torch", device=udevice).state
            psi = np.asarray(psi_out, dtype=np.complex128).flatten()
            probs = np.abs(psi) ** 2
            probs = probs / probs.sum()
            top_idx = np.argpartition(probs, -k)[-k:]
            top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
            bitstrings = {
                bitstring_of_index(int(i), n): float(probs[i])
                for i in top_idx}

        # Best sample: lowest-energy bitstring among the top-K most probable.
        best_idx = int(top_idx[np.argmin(e_vec[top_idx])])
        best_bitstring = bitstring_of_index(best_idx, n)
        best_energy = float(e_vec[best_idx])

        self.log(f"  best bitstring {best_bitstring}, "
                 f"energy {best_energy:.6f} (exact {exact_energy:.6f})")
        return {
            "bitstrings": bitstrings,
            "best_bitstring": best_bitstring,
            "best_energy": best_energy,
            "exact_energy": exact_energy,
            "energy_history": energy_history,
            "n_qubits": n,
            "layers": layers,
            "device": device,
            "optimizer": optimizer,
            "dtype": dtype,
            "checkpoint": checkpoint,
            "adjoint": adjoint,
            "final_backend": chosen_final_backend,
        }
