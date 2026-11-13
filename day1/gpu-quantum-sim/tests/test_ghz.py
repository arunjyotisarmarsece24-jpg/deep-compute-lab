"""Tests for GHZ state preparation and measurement correctness."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from src.ghz_benchmark import ghz, verify_ghz_correctness

try:
    import cudaq
except (ImportError, ModuleNotFoundError):
    from src import cudaq_runtime as cudaq


def test_ghz_correctness_helper():
    """Verify verify_ghz_correctness passes on n=8 qubits."""
    assert verify_ghz_correctness(n=8, shots=1000, tolerance=0.15) is True


@pytest.mark.parametrize("n", [3, 5, 8])
def test_ghz_bitstrings_only_all_zeros_and_ones(n: int):
    """Assert only |00...0> and |11...1> bitstrings appear in GHZ measurement."""
    shots = 1000
    res = cudaq.sample(ghz, n, shots_count=shots)

    exp_0 = "0" * n
    exp_1 = "1" * n

    for k in res.keys():
        assert k in (exp_0, exp_1), f"Unexpected bitstring {k} in GHZ({n})"

    count_0 = res[exp_0]
    count_1 = res[exp_1]
    assert count_0 + count_1 == shots
    # Statistical tolerance: within ~15% of 50/50
    assert abs(count_0 / shots - 0.5) < 0.15
    assert abs(count_1 / shots - 0.5) < 0.15
