"""Tests for H2 quantum chemistry, Jordan-Wigner Pauli terms, and VQE chemical accuracy."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from src.h2_vqe import build_h2_hamiltonian, run_vqe_single_point


def test_h2_pauli_term_count_15():
    """Verify that Jordan-Wigner mapped H2 in STO-3G has exactly 15 Pauli terms (including identity)."""
    hamiltonian, num_terms, e_hf, e_fci = build_h2_hamiltonian(0.735)
    assert num_terms == 15, f"Expected exactly 15 Pauli terms, got {num_terms}"
    assert len(hamiltonian) == 15


def test_h2_hf_energy_at_equilibrium():
    """Verify Hartree-Fock ground state energy at R = 0.735 Å."""
    _, _, e_hf, _ = build_h2_hamiltonian(0.735)
    # Expected STO-3G HF energy at 0.735 Å is ~ -1.116999 Ha
    assert pytest.approx(-1.116999, abs=1e-4) == e_hf


def test_h2_vqe_chemical_accuracy_at_equilibrium():
    """Verify VQE energy at R = 0.735 Å matches exact FCI within chemical accuracy (1.6 mHa)."""
    e_vqe, e_fci, e_hf, opt_theta, _ = run_vqe_single_point(
        r=0.735,
        initial_theta=0.1,
        method="COBYLA",
    )
    # Expected equilibrium minimum ~ -1.1373 Ha
    assert pytest.approx(-1.1373, abs=2e-3) == e_vqe
    assert pytest.approx(e_fci, abs=1e-5) == e_vqe

    abs_error_mha = abs(e_vqe - e_fci) * 1000.0
    print(f"Test VQE Error: {abs_error_mha:.6f} mHa")
    assert abs_error_mha < 1.60, f"VQE error {abs_error_mha:.4f} mHa exceeds chemical accuracy threshold (1.6 mHa)"


def test_h2_correlation_at_stretched_bond():
    """Verify that at stretched bond distance (R = 2.0 Å), VQE recovers correlation energy where HF fails."""
    e_vqe, e_fci, e_hf, _, _ = run_vqe_single_point(r=2.0, initial_theta=0.5)
    # At stretched bond, FCI and VQE are much lower than HF
    assert e_vqe < e_hf - 0.10, "VQE should significantly outperform RHF at bond dissociation"
    abs_error_mha = abs(e_vqe - e_fci) * 1000.0
    assert abs_error_mha < 1.60
