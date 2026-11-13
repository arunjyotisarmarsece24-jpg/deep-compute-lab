"""Memory estimation and hardware capacity validation for statevector simulation."""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Tuple

try:
    import psutil
except ImportError:
    psutil = None


def estimate_statevector_memory_bytes(n_qubits: int, precision: str = "complex64") -> int:
    """Calculate the theoretical memory in bytes required for an n-qubit statevector.

    Args:
        n_qubits: Number of qubits.
        precision: Floating point precision ("complex64", "fp32", "complex128", "fp64").

    Returns:
        Exact byte count required to store 2^n amplitudes.

    Raises:
        ValueError: If n_qubits < 0 or precision is unsupported.
    """
    if n_qubits < 0:
        raise ValueError(f"qubit count must be non-negative, got {n_qubits}")

    dim = 1 << n_qubits  # 2^n

    norm_prec = precision.lower().strip()
    if norm_prec in ("complex64", "fp32", "single"):
        bytes_per_amplitude = 8  # 2 * 4 bytes
    elif norm_prec in ("complex128", "fp64", "double"):
        bytes_per_amplitude = 16  # 2 * 8 bytes
    else:
        raise ValueError(
            f"Unsupported precision '{precision}'. Must be 'complex64'/'fp32' or 'complex128'/'fp64'."
        )

    return dim * bytes_per_amplitude


def estimate_statevector_memory_gb(n_qubits: int, precision: str = "complex64") -> float:
    """Calculate memory in gigabytes (GiB: 2^30 bytes) for an n-qubit statevector."""
    return estimate_statevector_memory_bytes(n_qubits, precision) / (1024.0**3)


def get_available_system_memory() -> Dict[str, float]:
    """Inspect physical host RAM and NVIDIA GPU VRAM.

    Returns:
        Dict containing:
            - 'ram_available_gb': available host RAM in GiB
            - 'ram_total_gb': total host RAM in GiB
            - 'vram_available_gb': free VRAM in GiB (or 0.0 if not detected)
            - 'vram_total_gb': total VRAM in GiB (or 0.0 if not detected)
    """
    mem_info = {
        "ram_available_gb": 4.0,
        "ram_total_gb": 8.0,
        "vram_available_gb": 0.0,
        "vram_total_gb": 0.0,
    }

    # Query Host RAM via psutil
    if psutil is not None:
        vm = psutil.virtual_memory()
        mem_info["ram_available_gb"] = vm.available / (1024.0**3)
        mem_info["ram_total_gb"] = vm.total / (1024.0**3)

    # Query GPU VRAM via nvidia-smi if available
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,nounits,noheader",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = lines[0].split(",")
            if len(parts) >= 2:
                free_mb = float(parts[0].strip())
                total_mb = float(parts[1].strip())
                mem_info["vram_available_gb"] = free_mb / 1024.0
                mem_info["vram_total_gb"] = total_mb / 1024.0
    except Exception:
        # nvidia-smi unavailable or no NVIDIA GPU connected
        pass

    return mem_info


def check_memory_fits(
    n_qubits: int,
    target: str = "qpp-cpu",
    precision: str = "complex64",
    safety_margin: float = 0.85,
) -> Tuple[bool, str]:
    """Verify whether a simulation of n_qubits fits safely in memory.

    Args:
        n_qubits: Number of qubits in statevector.
        target: Simulator target ("qpp-cpu" or "nvidia").
        precision: Floating-point precision ("complex64" or "complex128").
        safety_margin: Fractional threshold of free memory allowed (default: 85%).

    Returns:
        (fits: bool, message: str)
    """
    est_gb = estimate_statevector_memory_gb(n_qubits, precision)
    mem_info = get_available_system_memory()

    is_gpu = "nvidia" in target.lower()
    avail_gb = mem_info["vram_available_gb"] if is_gpu else mem_info["ram_available_gb"]
    dev_name = "VRAM" if is_gpu else "Host RAM"

    # If GPU target requested but no VRAM detected
    if is_gpu and avail_gb <= 0.01:
        return (
            False,
            f"Target '{target}' requested, but no active NVIDIA GPU VRAM detected.",
        )

    max_allowed = avail_gb * safety_margin
    if est_gb > max_allowed:
        return (
            False,
            f"Statevector for n={n_qubits} requires {est_gb:.3f} GiB {precision}, "
            f"which exceeds safe available {dev_name} ({max_allowed:.3f} GiB allowed of {avail_gb:.3f} GiB free).",
        )

    return (
        True,
        f"Fits safely: {est_gb:.4f} GiB needed <= {max_allowed:.3f} GiB allocatable on {dev_name}.",
    )
