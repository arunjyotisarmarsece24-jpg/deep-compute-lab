"""Tests for statevector simulation memory estimator and safety validation."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from src.memory_estimator import (
    check_memory_fits,
    estimate_statevector_memory_bytes,
    estimate_statevector_memory_gb,
    get_available_system_memory,
)


def test_estimate_statevector_memory_bytes_complex64():
    """Verify byte calculation for 2^n * 8 bytes (complex64)."""
    assert estimate_statevector_memory_bytes(0, "complex64") == 8
    assert estimate_statevector_memory_bytes(1, "complex64") == 16
    assert estimate_statevector_memory_bytes(10, "complex64") == 1024 * 8
    # n=20: 1048576 * 8 = 8,388,608 bytes
    assert estimate_statevector_memory_bytes(20, "complex64") == (1 << 20) * 8
    # n=30: 2^30 * 8 bytes = 8 GiB
    assert estimate_statevector_memory_bytes(30, "complex64") == 8 * (1024**3)


def test_estimate_statevector_memory_bytes_complex128():
    """Verify byte calculation for 2^n * 16 bytes (complex128 / fp64)."""
    assert estimate_statevector_memory_bytes(0, "complex128") == 16
    assert estimate_statevector_memory_bytes(20, "fp64") == (1 << 20) * 16
    assert estimate_statevector_memory_bytes(30, "double") == 16 * (1024**3)


def test_estimate_statevector_memory_gb():
    """Verify GiB calculations."""
    # n=30 complex64: exactly 8.0 GiB
    assert pytest.approx(8.0, rel=1e-5) == estimate_statevector_memory_gb(30, "complex64")
    # n=30 complex128: exactly 16.0 GiB
    assert pytest.approx(16.0, rel=1e-5) == estimate_statevector_memory_gb(30, "complex128")
    # n=20 complex64: 8 MiB = 8 / 1024 GiB = 0.0078125 GiB
    assert pytest.approx(8.0 / 1024.0, rel=1e-5) == estimate_statevector_memory_gb(20, "complex64")


def test_memory_estimator_input_validation():
    """Verify input validation handles invalid inputs gracefully."""
    with pytest.raises(ValueError):
        estimate_statevector_memory_bytes(-1)

    with pytest.raises(ValueError):
        estimate_statevector_memory_bytes(10, precision="invalid_precision")


def test_get_available_system_memory():
    """Verify host RAM detection returns valid positive numbers."""
    mem = get_available_system_memory()
    assert "ram_available_gb" in mem
    assert "ram_total_gb" in mem
    assert mem["ram_available_gb"] > 0.0
    assert mem["ram_total_gb"] > 0.0
    assert mem["ram_available_gb"] <= mem["ram_total_gb"]


def test_check_memory_fits():
    """Verify memory safety checks allow safe qubit counts and reject excessive ones."""
    # Small qubit count (n=10, 8 KB) should always fit
    fits, msg = check_memory_fits(10, target="qpp-cpu")
    assert fits is True
    assert "Fits safely" in msg

    # Extreme qubit count (n=50 requires ~8 Petabytes) must always be rejected
    fits_huge, msg_huge = check_memory_fits(50, target="qpp-cpu")
    assert fits_huge is False
    assert "exceeds" in msg_huge
