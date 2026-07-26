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

from unitarylab import Circuit
from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm

# Number of top-probability bitstrings returned in result['bitstrings'].
TOP_K = 16
# Internal random restarts for the autograd path (best expectation kept).
# All restarts draw from the same seeded RNG, so results stay reproducible.
N_RESTARTS = 3


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
    def _spin_table(n: int) -> np.ndarray:
        """(2^n, n) table of z values; row idx = basis state, qubit 0 = LSB
        (unitarylab statevector ordering)."""
        idx = np.arange(2**n)
        shifts = np.arange(n)
        bits = (idx[:, None] >> shifts[None, :]) & 1
        return 1.0 - 2.0 * bits.astype(np.float64)

    @staticmethod
    def _energy_vector(h: np.ndarray, J: np.ndarray) -> np.ndarray:
        """E(z) for every computational basis state (diag of H_C)."""
        n = len(h)
        z = IsingQAOA._spin_table(n)
        e = z @ h
        iu = np.triu_indices(n, k=1)
        e = e + (z[:, iu[0]] * z[:, iu[1]]) @ J[iu]
        return e

    def _get_h_cost(self, h: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Weighted couplings sum_{i<j} J_ij Z_i Z_j + local fields sum_i h_i Z_i.

        Returns the dense (diagonal) cost Hamiltonian, same layout as the
        base class: entry (idx, idx) is E(z) of basis state idx under the
        module bitstring convention.
        """
        h = np.asarray(h, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
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
                      device: str = "cpu") -> torch.Tensor:
        """Statevector after the QAOA circuit, as a torch tensor.

        Applies exactly the same unitary as ``_build_circuit``:
        |+>^n -> per layer [ exp(-i*gamma*H_C) ; prod_i RX(2*beta, i) ].
        H_C is diagonal, so the cost layer is an exact elementwise phase.
        Differentiable w.r.t. ``params`` if it requires grad.

        ``params``: array-like of length 2*layers (numpy or torch tensor).
        Returns a complex128 tensor of shape (2^n,) on ``device``.
        """
        n = len(h)
        p = len(params) // 2
        if not isinstance(params, torch.Tensor):
            params = torch.as_tensor(np.asarray(params, dtype=np.float64))
        params = params.to(device=device, dtype=torch.float64)
        gammas, betas = params[:p], params[p:]

        e_vec = torch.as_tensor(self._energy_vector(h, J),
                                dtype=torch.float64, device=device)
        dim = 2**n
        n_axes = tuple(range(n - 1, -1, -1))

        def to_axes(flat):
            # flat index = sum_k x_k * 2**k (qubit 0 = LSB, unitarylab
            # backend ordering) -> shape (2,)*n with axis k = qubit k.
            return flat.reshape((2,) * n).permute(n_axes)

        def to_flat(t):
            return t.permute(n_axes).reshape(dim)

        # |+>^n
        psi = torch.full((dim,), 2.0 ** (-n / 2),
                         dtype=torch.complex128, device=device)

        for layer in range(p):
            # cost layer: diagonal phase exp(-i * gamma * E(z))
            psi = psi * torch.exp(-1j * gammas[layer] * e_vec)
            # mixer layer: RX(2*beta) on every qubit
            theta = 2.0 * betas[layer]
            c = torch.cos(theta / 2.0)
            s = torch.sin(theta / 2.0)
            rx = torch.stack([
                torch.stack([c, -1j * s]),
                torch.stack([-1j * s, c]),
            ]).to(torch.complex128)
            t = to_axes(psi)
            for k in range(n):
                t = torch.movedim(t, k, 0).reshape(2, -1)
                t = (rx @ t).reshape((2,) * n)
                t = torch.movedim(t, 0, k)
            psi = to_flat(t)
        return psi

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
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
                        max_iter, energy_history):
        params = torch.as_tensor(initial_params, dtype=torch.float64,
                                 device=device).clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=0.05)
        e_vec = torch.as_tensor(self._energy_vector(h, J),
                                dtype=torch.float64, device=device)
        for _ in range(max_iter):
            opt.zero_grad()
            psi = self._evolve_torch(params, h, J, device=device)
            probs = psi.real**2 + psi.imag**2
            energy = torch.sum(probs * e_vec)
            energy.backward()
            opt.step()
            energy_history.append(float(energy.detach().cpu()))
        return params.detach().cpu().numpy(), float(energy_history[-1])

    def _train_cobyla(self, initial_params, h, J, layers, device,
                      max_iter, energy_history):
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

    def solve(self, h: np.ndarray, J: np.ndarray, layers: int = 4,
              device: str = "cpu", optimizer: str = "autograd",
              max_iter: int = 200, seed: int = 42) -> dict:
        """Solve a weighted Ising model with QAOA.

        Parameters:
            h: (n,) local field coefficients
            J: (n,n) symmetric coupling matrix with zero diagonal
            layers: QAOA depth p
            device: 'cpu' or torch device string (e.g. 'cuda:0'); passed
                through to torch tensors and to qc.execute(backend='torch',
                device=device) on the cobyla path
            optimizer: 'autograd' (Adam, torch backprop through the
                circuit) or 'cobyla' (scipy COBYLA cross-check path)
            max_iter: optimizer iterations
            seed: RNG seed for parameter initialization (fully reproducible)

        Returns:
            dict with keys: bitstrings (top-K bitstring -> probability),
            best_bitstring (lowest-energy bitstring among the top-K most
            probable outcomes), best_energy, exact_energy (dense
            diagonalization ground-state energy), energy_history,
            n_qubits, layers, device, optimizer.
        """
        h, J, n = self._validate(h, J, optimizer)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        self.log(f"Stage 1: Building Ising Hamiltonian (n={n})")
        h_cost = self._get_h_cost(h, J)
        exact_energy = float(np.linalg.eigvalsh(h_cost)[0])

        self.log(f"Stage 2: Variational optimization "
                 f"({optimizer}, layers={layers}, max_iter={max_iter})")
        energy_history: list = []
        q_start = time.time()
        final_exp = None
        if optimizer == "autograd":
            # Small-parameter inits avoid the poor plateaus of uniform(0, pi)
            # inits; a few seeded restarts, keep the best expectation.
            # Note: energy_history concatenates all restarts.
            final_params, best_exp = None, np.inf
            for _ in range(N_RESTARTS):
                init = rng.uniform(0.0, 0.5, size=2 * layers)
                params_r, exp_r = self._train_autograd(
                    init, h, J, layers, device, max_iter, energy_history)
                if exp_r < best_exp:
                    best_exp, final_params = exp_r, params_r
            final_exp = best_exp
        else:
            initial_params = rng.uniform(0.0, np.pi, size=2 * layers)
            final_params = self._train_cobyla(initial_params, h, J, layers,
                                              device, max_iter,
                                              energy_history)
            final_exp = energy_history[-1]
        q_time = time.time() - q_start
        self.log(f"  optimization time: {q_time:.3f}s, "
                 f"final expectation: {final_exp:.6f}")

        self.log("Stage 3: Final state measurement and bitstring decoding")
        # Final state via the unitarylab circuit (bit-exact convention anchor).
        qc_final = self._build_circuit(final_params, h, J)
        psi_out = qc_final.execute(backend="torch", device=device).state
        psi = np.asarray(psi_out, dtype=np.complex128).flatten()
        probs = np.abs(psi) ** 2
        probs = probs / probs.sum()

        e_vec = self._energy_vector(h, J)
        k = min(TOP_K, 2**n)
        top_idx = np.argpartition(probs, -k)[-k:]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
        bitstrings = {bitstring_of_index(int(i), n): float(probs[i])
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
        }
