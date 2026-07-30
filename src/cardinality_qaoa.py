"""Fixed-cardinality QAOA with a Hamming-weight-preserving XY mixer.

The standard X mixer explores all 2**n bitstrings and does not preserve a
portfolio cardinality constraint.  This solver works directly in the
``C(n, K)`` feasible subspace, initializes a Dicke state, and applies a
Trotterized ring-XY mixer.  Every state therefore has exactly K selected
assets and no penalty/postselection is needed to make a result executable.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ising_qaoa import bitstring_of_index, resolve_device


def fixed_weight_states(n: int, cardinality: int) -> np.ndarray:
    if not (1 <= cardinality <= n):
        raise ValueError(
            f"cardinality must satisfy 1 <= K <= {n}, got {cardinality}"
        )
    count = math.comb(n, cardinality)
    states = np.fromiter(
        (
            sum(1 << bit for bit in chosen)
            for chosen in itertools.combinations(range(n), cardinality)
        ),
        dtype=np.int64,
        count=count,
    )
    states.sort()
    return states


def ising_energies_for_states(
    h: np.ndarray,
    J: np.ndarray,
    states: np.ndarray,
    chunk_size: int = 1 << 18,
) -> np.ndarray:
    """Compute Ising energies only for supplied basis-state indices."""
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    n = len(h)
    if J.shape != (n, n):
        raise ValueError(f"J must have shape ({n}, {n})")
    shifts = np.arange(n, dtype=np.int64)
    energies = np.empty(len(states), dtype=np.float64)
    # The Biren image's OpenBLAS 0.3.20 can silently corrupt rows in very
    # tall-skinny GEMM when it dispatches 64 CPU threads. This is the same
    # failure mode guarded in IsingQAOA._energy_vector.
    with threadpool_limits(limits=1, user_api="blas"):
        for start in range(0, len(states), chunk_size):
            stop = min(start + chunk_size, len(states))
            z = 1.0 - 2.0 * (
                (states[start:stop, None] >> shifts[None, :]) & 1
            ).astype(np.float64)
            energies[start:stop] = (
                z @ h + 0.5 * np.sum(z * (z @ J), axis=1)
            )
    return energies


class CardinalityQAOA:
    """QAOA simulator restricted to a fixed-Hamming-weight subspace."""

    def __init__(self, n: int, cardinality: int, device: str = "cpu"):
        self.n = int(n)
        self.cardinality = int(cardinality)
        self.device_name = device
        self.device = resolve_device(device)
        self.states = fixed_weight_states(self.n, self.cardinality)
        self._edge_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Even/odd ring ordering reduces Trotter ordering bias while keeping
        # each individual gate a disjoint two-state rotation.
        ring = [(i, (i + 1) % self.n) for i in range(self.n)]
        ordered = ring[0::2] + ring[1::2]
        for left, right in ordered:
            left_bit = 1 << left
            right_bit = 1 << right
            mask = (
                ((self.states & left_bit) == 0)
                & ((self.states & right_bit) != 0)
            )
            source = np.flatnonzero(mask).astype(np.int64)
            target_states = self.states[source] ^ left_bit ^ right_bit
            target = np.searchsorted(
                self.states, target_states
            ).astype(np.int64)
            if not np.array_equal(self.states[target], target_states):
                raise RuntimeError("fixed-weight state lookup failed")
            self._edge_pairs.append(
                (
                    torch.as_tensor(
                        source, dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        target, dtype=torch.long, device=self.device
                    ),
                )
            )

    def _evolve(
        self,
        params: torch.Tensor,
        energies: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layers = len(params) // 2
        gammas = params[:layers]
        betas = params[layers:]
        real_dtype = params.dtype
        complex_dtype = (
            torch.complex64
            if real_dtype == torch.float32
            else torch.complex128
        )
        if initial_state is None:
            psi = torch.full(
                (len(self.states),),
                len(self.states) ** -0.5,
                dtype=complex_dtype,
                device=self.device,
            )
        else:
            psi = initial_state.to(dtype=complex_dtype)
        for layer in range(layers):
            angle = gammas[layer] * energies
            phase = torch.complex(torch.cos(angle), -torch.sin(angle))
            psi = psi * phase
            c = torch.cos(betas[layer])
            s = torch.sin(betas[layer])
            minus_i_s = torch.complex(torch.zeros_like(s), -s)
            for source, target in self._edge_pairs:
                left = psi[source]
                right = psi[target]
                updated = psi.clone()
                updated[source] = c * left + minus_i_s * right
                updated[target] = minus_i_s * left + c * right
                psi = updated
        return psi

    def _evolve_batch(
        self, params: torch.Tensor, energies: torch.Tensor
    ) -> torch.Tensor:
        """Evolve independent parameter sets in one fused seed batch."""
        if params.ndim != 2:
            raise ValueError("batched params must have shape (batch, 2*p)")
        layers = params.shape[1] // 2
        gammas = params[:, :layers]
        betas = params[:, layers:]
        complex_dtype = (
            torch.complex64
            if params.dtype == torch.float32
            else torch.complex128
        )
        psi = torch.full(
            (params.shape[0], len(self.states)),
            len(self.states) ** -0.5,
            dtype=complex_dtype,
            device=self.device,
        )
        for layer in range(layers):
            angle = gammas[:, layer, None] * energies[None, :]
            phase = torch.complex(torch.cos(angle), -torch.sin(angle))
            psi = psi * phase
            c = torch.cos(betas[:, layer])[:, None]
            s = torch.sin(betas[:, layer])[:, None]
            minus_i_s = torch.complex(torch.zeros_like(s), -s)
            for source, target in self._edge_pairs:
                left = psi[:, source]
                right = psi[:, target]
                updated = psi.clone()
                updated[:, source] = c * left + minus_i_s * right
                updated[:, target] = minus_i_s * left + c * right
                psi = updated
        return psi

    @staticmethod
    def _interp_extend(params: np.ndarray) -> np.ndarray:
        p = len(params) // 2
        output = np.empty(2 * (p + 1), dtype=np.float64)
        weights = np.arange(p + 1) / p
        for offset, source in ((0, params[:p]), (p + 1, params[p:])):
            extended = np.concatenate(([0.0], source, [0.0]))
            output[offset:offset + p + 1] = (
                weights * extended[:-1]
                + (1.0 - weights) * extended[1:]
            )
        return output

    def _train_level(
        self,
        initial: np.ndarray,
        energies: torch.Tensor,
        iterations: int,
        history: list[float],
        objective: str,
        cvar_alpha: float,
        energy_order: torch.Tensor | None,
        initial_state: torch.Tensor | None,
    ) -> tuple[np.ndarray, float]:
        params = torch.as_tensor(
            initial, dtype=energies.dtype, device=self.device
        ).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=0.05)
        best_energy = np.inf
        best_params = None
        for _ in range(iterations):
            optimizer.zero_grad()
            psi = self._evolve(params, energies, initial_state)
            probabilities = psi.real.square() + psi.imag.square()
            if objective == "expectation":
                loss = torch.sum(probabilities * energies)
            else:
                assert energy_order is not None
                sorted_probabilities = probabilities[energy_order]
                sorted_energies = energies[energy_order]
                # The cutoff is piecewise constant, so determine it from a
                # detached cumulative mass. This gives the exact a.e. CVaR
                # gradient and avoids CumsumBackward0, whose implementation
                # calls an unsupported Flip kernel on torch_br/SUPA.
                cumulative = torch.cumsum(
                    sorted_probabilities.detach(), dim=0
                )
                previous = cumulative - sorted_probabilities.detach()
                fully_in_tail = cumulative <= cvar_alpha
                crosses_cutoff = (
                    (previous < cvar_alpha)
                    & (cumulative >= cvar_alpha)
                )
                cutoff_energy = torch.sum(
                    crosses_cutoff.to(sorted_energies.dtype)
                    * sorted_energies
                )
                full_mass = torch.sum(
                    sorted_probabilities
                    * fully_in_tail.to(sorted_probabilities.dtype)
                )
                full_value = torch.sum(
                    sorted_probabilities
                    * sorted_energies
                    * fully_in_tail.to(sorted_probabilities.dtype)
                )
                loss = (
                    full_value
                    + (cvar_alpha - full_mass) * cutoff_energy
                ) / cvar_alpha
            loss.backward()
            value = float(loss.detach().cpu())
            history.append(value)
            if value < best_energy:
                best_energy = value
                best_params = params.detach().clone()
            optimizer.step()
        if best_params is None:
            best_params = params.detach()
            best_energy = float("inf")
        return best_params.cpu().numpy(), best_energy

    def solve(
        self,
        h: np.ndarray,
        J: np.ndarray,
        *,
        layers: int = 2,
        max_iter: int = 40,
        seed: int = 42,
        init: str = "interp",
        top_k: int = 4096,
        restarts: int = 3,
        objective: str = "expectation",
        cvar_alpha: float = 0.1,
        warm_start_bitstring: str | None = None,
        warm_start_strength: float = 0.0,
        initial_angle_scale: float = 0.5,
        reference_bitstring: str | None = None,
    ) -> dict[str, Any]:
        if layers < 1:
            raise ValueError("layers must be positive")
        if init not in ("random", "interp"):
            raise ValueError("init must be random or interp")
        if objective not in ("expectation", "cvar"):
            raise ValueError("objective must be expectation or cvar")
        if not (0.0 < cvar_alpha <= 1.0):
            raise ValueError("cvar_alpha must satisfy 0 < alpha <= 1")
        if initial_angle_scale < 0.0:
            raise ValueError("initial_angle_scale must be nonnegative")
        energies_np = ising_energies_for_states(
            h, J, self.states
        )
        energies = torch.as_tensor(
            energies_np, dtype=torch.float32, device=self.device
        )
        exact_position = int(np.argmin(energies_np))
        exact_bitstring = bitstring_of_index(
            int(self.states[exact_position]), self.n
        )
        rng = np.random.default_rng(seed)
        history: list[float] = []
        initial_state = None
        warm_start_initial_probability = None
        if warm_start_bitstring is not None:
            if len(warm_start_bitstring) != self.n:
                raise ValueError("warm-start bitstring has wrong length")
            warm_state = sum(
                int(bit) << index
                for index, bit in enumerate(warm_start_bitstring)
            )
            if int(warm_state).bit_count() != self.cardinality:
                raise ValueError("warm-start bitstring has wrong cardinality")
            selected_overlap = np.zeros(
                len(self.states), dtype=np.int16
            )
            for bit in range(self.n):
                if warm_state & (1 << bit):
                    selected_overlap += (
                        (self.states >> bit) & 1
                    ).astype(np.int16)
            log_probability = warm_start_strength * (
                selected_overlap - self.cardinality
            )
            log_probability -= float(log_probability.max())
            amplitude = np.exp(0.5 * log_probability)
            amplitude /= np.linalg.norm(amplitude)
            initial_state = torch.as_tensor(
                amplitude,
                dtype=torch.float32,
                device=self.device,
            )
            warm_position = int(
                np.searchsorted(self.states, warm_state)
            )
            warm_start_initial_probability = float(
                amplitude[warm_position] ** 2
            )
        energy_order = (
            torch.argsort(energies) if objective == "cvar" else None
        )

        def train_restarts(level: int) -> tuple[np.ndarray, float]:
            best_params = None
            best_expectation = np.inf
            for _ in range(restarts):
                initial = rng.uniform(
                    0.0, initial_angle_scale, size=2 * level
                )
                params, expectation = self._train_level(
                    initial,
                    energies,
                    max_iter,
                    history,
                    objective,
                    cvar_alpha,
                    energy_order,
                    initial_state,
                )
                if expectation < best_expectation:
                    best_params = params
                    best_expectation = expectation
            assert best_params is not None
            return best_params, best_expectation

        if init == "interp":
            params, expectation = train_restarts(1)
            for level in range(2, layers + 1):
                params, expectation = self._train_level(
                    self._interp_extend(params),
                    energies,
                    max_iter,
                    history,
                    objective,
                    cvar_alpha,
                    energy_order,
                    initial_state,
                )
        else:
            params, expectation = train_restarts(layers)

        with torch.no_grad():
            params_t = torch.as_tensor(
                params, dtype=torch.float32, device=self.device
            )
            psi = self._evolve(params_t, energies, initial_state)
            probabilities = psi.real.square() + psi.imag.square()
            probabilities = probabilities / probabilities.sum()
            k = min(int(top_k), len(self.states))
            top_probability, top_position = torch.topk(
                probabilities, k=k, largest=True, sorted=True
            )
            positions = top_position.cpu().numpy().astype(np.int64)
            probability_values = (
                top_probability.cpu().numpy().astype(np.float64)
            )
            reference_probability = None
            reference_probability_rank = None
            if reference_bitstring is not None:
                reference_state = sum(
                    int(bit) << index
                    for index, bit in enumerate(reference_bitstring)
                )
                reference_position = int(
                    np.searchsorted(self.states, reference_state)
                )
                if (
                    reference_position >= len(self.states)
                    or self.states[reference_position] != reference_state
                ):
                    raise ValueError(
                        "reference bitstring is outside feasible subspace"
                    )
                reference_probability_t = probabilities[
                    reference_position
                ]
                reference_probability = float(
                    reference_probability_t.cpu()
                )
                reference_probability_rank = int(
                    torch.sum(
                        probabilities > reference_probability_t
                    ).cpu()
                ) + 1
            exact_probability_t = probabilities[exact_position]
            exact_probability = float(exact_probability_t.cpu())
            exact_probability_rank = int(
                torch.sum(probabilities > exact_probability_t).cpu()
            ) + 1
        best_position = int(positions[np.argmin(energies_np[positions])])
        top = {
            bitstring_of_index(int(self.states[position]), self.n): float(prob)
            for position, prob in zip(positions, probability_values)
        }
        return {
            "best_bitstring": bitstring_of_index(
                int(self.states[best_position]), self.n
            ),
            "best_energy": float(energies_np[best_position]),
            "exact_feasible_energy": float(energies_np.min()),
            "exact_feasible_bitstring": exact_bitstring,
            "exact_feasible_probability": exact_probability,
            "exact_feasible_probability_rank": exact_probability_rank,
            "expectation": float(expectation),
            "energy_history": history,
            "bitstrings": top,
            "feasible_dimension": int(len(self.states)),
            "full_dimension": int(1 << self.n),
            "cardinality": self.cardinality,
            "layers": layers,
            "init": init,
            "max_iter": max_iter,
            "restarts": restarts,
            "device": self.device_name,
            "probability_norm": float(probabilities.sum().cpu()),
            "objective": objective,
            "cvar_alpha": cvar_alpha if objective == "cvar" else None,
            "warm_start_bitstring": warm_start_bitstring,
            "warm_start_strength": warm_start_strength,
            "warm_start_initial_probability": (
                warm_start_initial_probability
            ),
            "reference_bitstring": reference_bitstring,
            "reference_probability": reference_probability,
            "reference_probability_rank": reference_probability_rank,
        }
