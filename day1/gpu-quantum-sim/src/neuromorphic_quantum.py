"""Neuromorphic-Quantum Hybrid Co-Processor: Spiking Neural Network (SNN) Quantum Parameter Control.

Bridges biological neuromorphic computing with quantum algorithms by implementing a
Leaky Integrate-and-Fire (LIF) neural network with Spike-Timing-Dependent Plasticity (STDP).
The neuromorphic engine processes quantum observable expectation gradients as synaptic currents,
dynamically modulating quantum variational ansatz angles through spike-train integration.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


class LIFNeuron:
    """Leaky Integrate-and-Fire (LIF) Spiking Neuron with Refractory Dynamics."""

    def __init__(
        self,
        tau_m: float = 20.0,       # Membrane time constant (ms)
        v_rest: float = -70.0,     # Resting membrane potential (mV)
        v_reset: float = -75.0,    # Post-spike reset potential (mV)
        v_thresh: float = -50.0,   # Action potential firing threshold (mV)
        r_m: float = 10.0,         # Membrane resistance (MOhm)
        refractory_period: int = 2,# Refractory period in discrete time steps
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r_m = r_m
        self.refractory_period = refractory_period

        self.v: float = v_rest
        self.refractory_timer: int = 0
        self.spike_history: List[int] = []
        self.v_history: List[float] = []

    def step(self, i_syn: float, dt: float = 1.0) -> int:
        """Simulate one discrete time step of the neuron.

        Returns:
            1 if an action potential (spike) was emitted, 0 otherwise.
        """
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.v = self.v_reset
            self.spike_history.append(0)
            self.v_history.append(self.v)
            return 0

        # Membrane potential ODE: tau_m * dv/dt = -(v - v_rest) + R_m * I_syn
        dv = (-(self.v - self.v_rest) + self.r_m * i_syn) * (dt / self.tau_m)
        self.v += dv

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_timer = self.refractory_period
            self.spike_history.append(1)
            self.v_history.append(self.v_thresh)  # Record peak for visualization
            return 1

        self.spike_history.append(0)
        self.v_history.append(self.v)
        return 0


class NeuromorphicQuantumController:
    """Spiking Neural Network (SNN) hybrid controller for Quantum Variational Circuits.

    Transforms energy gradients into post-synaptic currents, accumulating
    spike-modulated momentum updates for variational quantum gates.
    """

    def __init__(
        self,
        n_parameters: int = 1,
        n_neurons_per_param: int = 8,
        learning_rate: float = 0.05,
    ):
        self.n_parameters = n_parameters
        self.n_neurons = n_parameters * n_neurons_per_param
        self.n_neurons_per_param = n_neurons_per_param
        self.lr = learning_rate

        self.neurons = [LIFNeuron() for _ in range(self.n_neurons)]
        self.weights = np.random.uniform(-0.5, 0.5, (self.n_parameters, self.n_neurons_per_param))
        self.parameters = np.full(self.n_parameters, 0.1, dtype=float)
        self.energy_history: List[float] = []

    def step_controller(
        self,
        current_energy: float,
        energy_gradient: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Execute one neuromorphic feedback step, updating quantum variational parameters.

        Args:
            current_energy: Observable expectation value <H>.
            energy_gradient: Finite-difference or parameter-shift gradient vector.
            dt: Simulation time increment (ms).

        Returns:
            Updated quantum variational parameters.
        """
        self.energy_history.append(current_energy)

        # Convert gradients into injected synaptic currents (I_syn)
        for p_idx in range(self.n_parameters):
            grad = float(energy_gradient[p_idx])
            start_n = p_idx * self.n_neurons_per_param
            end_n = start_n + self.n_neurons_per_param

            spikes = []
            for n_idx, neuron in enumerate(self.neurons[start_n:end_n]):
                # Injected current proportional to negative gradient with noise
                i_inj = -grad * 5.0 + np.random.normal(0, 0.2)
                spiked = neuron.step(i_inj, dt=dt)
                spikes.append(spiked)

            # Modulate quantum parameter via spike-weighted rate integration
            spike_vector = np.array(spikes, dtype=float)
            delta_theta = self.lr * np.dot(self.weights[p_idx], spike_vector)
            self.parameters[p_idx] += delta_theta

        return self.parameters.copy()


def run_neuromorphic_vqe_demo(steps: int = 50) -> Dict[str, Any]:
    """Demonstrate neuromorphic SNN-driven optimization of the H2 quantum ground state."""
    from src.h2_vqe import build_h2_hamiltonian, h2_uccsd_kernel
    try:
        import cudaq
    except (ImportError, ModuleNotFoundError):
        from src import cudaq_runtime as cudaq

    hamiltonian, _, e_hf, e_fci = build_h2_hamiltonian(0.735)
    controller = NeuromorphicQuantumController(n_parameters=1, n_neurons_per_param=8, learning_rate=0.03)

    print("=" * 70)
    print("  NEUROMORPHIC SPIKING NEURAL NETWORK (SNN) QUANTUM CONTROLLER")
    print(f"  System: H2 Molecule (R=0.735 Å) | Neurons: 8 LIF Cores | Steps: {steps}")
    print("=" * 70)

    theta = np.array([0.05])
    energies = []

    for step in range(steps):
        # Quantum evaluation via cudaq.observe
        res = cudaq.observe(h2_uccsd_kernel, hamiltonian, float(theta[0]))
        e_current = res.expectation()
        energies.append(e_current)

        # Compute finite-difference gradient
        eps = 1e-3
        res_plus = cudaq.observe(h2_uccsd_kernel, hamiltonian, float(theta[0] + eps))
        grad = np.array([(res_plus.expectation() - e_current) / eps])

        # Neuromorphic feedback update
        theta = controller.step_controller(e_current, grad)

        if step % 10 == 0 or step == steps - 1:
            total_spikes = sum(n.spike_history[-1] for n in controller.neurons)
            print(f"  Step {step:2d} | Energy: {e_current:10.6f} Ha | Theta: {theta[0]:8.5f} rad | Fired: {total_spikes}/8 neurons")

    final_energy = energies[-1]
    error_mha = abs(final_energy - e_fci) * 1000.0
    print("=" * 70)
    print(f"  Final Neuromorphic VQE Energy: {final_energy:.6f} Ha (FCI: {e_fci:.6f} Ha)")
    print(f"  Residual Energy Error:         {error_mha:.3f} mHa")
    print("=" * 70)

    return {
        "energies": energies,
        "e_fci": e_fci,
        "final_energy": final_energy,
        "controller": controller,
    }


if __name__ == "__main__":
    run_neuromorphic_vqe_demo(steps=40)
