"""Unit Tests for Neuromorphic Spiking Dynamics, STDP Learning, and Quantum Entropy."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from src.live_cockpit import ResearchCockpitSimulator


def test_neuromorphic_simulation_initialization():
    """Verify that the cockpit simulator initializes all physical constants correctly."""
    sim = ResearchCockpitSimulator(bond_distance=0.735, n_neurons=8)
    assert sim.n_neurons == 8
    assert sim.v_mem.shape == (8,)
    assert sim.W.shape == (8, 8)
    assert np.allclose(np.diag(sim.W), 0.0)
    assert -1.138 < sim.e_fci < -1.136


def test_quantum_state_entanglement_and_entropy():
    """Verify reduced density matrix properties and von Neumann entanglement entropy."""
    sim = ResearchCockpitSimulator(bond_distance=0.735)
    
    # At theta = 0, state is pure product state |1100>, entropy must be ~0
    psi_0, e_0, entropy_0 = sim.compute_quantum_state(0.0)
    assert np.isclose(np.linalg.norm(psi_0), 1.0)
    assert np.isclose(entropy_0, 0.0, atol=1e-3)

    # At theta > 0, entanglement increases
    psi_theta, e_theta, entropy_theta = sim.compute_quantum_state(0.12)
    assert np.isclose(np.linalg.norm(psi_theta), 1.0)
    assert entropy_theta > entropy_0
    assert entropy_theta > 0.1  # Significant quantum entanglement generated


def test_stdp_synaptic_plasticity_and_vqe_convergence():
    """Verify that LIF neurons spike, STDP updates weights, and energy descends towards FCI."""
    sim = ResearchCockpitSimulator(bond_distance=0.735, n_neurons=8)
    initial_energy = sim.compute_quantum_state(sim.theta)[1]

    # Run 35 steps
    for s in range(35):
        sim.step(s)

    # Total spikes fired across simulation
    all_spikes = np.array(sim.spike_histories)
    total_spikes = np.sum(all_spikes)
    assert total_spikes > 0, "LIF neurons must fire action potentials"

    # Synaptic weights must evolve under STDP
    initial_w = sim.weight_histories[0]
    final_w = sim.weight_histories[-1]
    assert not np.allclose(initial_w, final_w), "STDP must actively update synaptic weights"

    # Energy must descend towards exact FCI
    final_energy = sim.energy_history[-1]
    assert final_energy < initial_energy, "Neuromorphic-quantum controller must optimize energy"
    assert abs(final_energy - sim.e_fci) < 0.015, "Energy must approach exact FCI"
