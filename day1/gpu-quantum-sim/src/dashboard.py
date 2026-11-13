"""Interactive Research Telemetry Dashboard: Quantum-Neuromorphic Live Hardware Monitor."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from src.memory_estimator import get_available_system_memory
from src.neuromorphic_quantum import NeuromorphicQuantumController

try:
    import cudaq
except (ImportError, ModuleNotFoundError):
    from src import cudaq_runtime as cudaq

from src.h2_vqe import CHEMICAL_ACCURACY_MHA, build_h2_hamiltonian, h2_uccsd_kernel


def render_dashboard(
    steps: int = 25,
    save_path: str = "results/dashboard_telemetry.png",
    interactive: bool = False,
) -> None:
    """Generate and render the 4-panel Quantum-Neuromorphic Hardware Telemetry Dashboard."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    print("[DASHBOARD] Initializing Quantum Hamiltonian & Neuromorphic SNN Controller...")
    ham, _, e_hf, e_fci = build_h2_hamiltonian(0.735)
    controller = NeuromorphicQuantumController(n_parameters=1, n_neurons_per_param=6, learning_rate=0.035)

    mem_sys = get_available_system_memory()
    theta = 0.08
    vqe_energies = []
    thetas = []

    for s in range(steps):
        res = cudaq.observe(h2_uccsd_kernel, ham, theta)
        e = res.expectation()
        vqe_energies.append(e)
        thetas.append(theta)

        # Finite difference gradient
        res_plus = cudaq.observe(h2_uccsd_kernel, ham, theta + 1e-3)
        grad = np.array([(res_plus.expectation() - e) / 1e-3])
        theta = float(controller.step_controller(e, grad)[0])

    # 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.suptitle(
        "NVIDIA CUDA-Q & Neuromorphic SNN Hybrid Co-Processor Telemetry",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: VQE Energy Descent & Chemical Accuracy
    ax1 = axes[0, 0]
    ax1.plot(range(1, steps + 1), vqe_energies, marker="o", color="#1f77b4", lw=2, label="SNN-VQE Energy")
    ax1.axhline(e_fci, color="#2ca02c", lw=2, label=f"Exact FCI ({e_fci:.4f} Ha)")
    ax1.axhspan(
        e_fci - CHEMICAL_ACCURACY_MHA / 1000.0,
        e_fci + CHEMICAL_ACCURACY_MHA / 1000.0,
        color="#2ca02c",
        alpha=0.2,
        label=f"Chemical Accuracy (±{CHEMICAL_ACCURACY_MHA:.1f} mHa)",
    )
    ax1.set_title("Panel A: SNN-Guided Variational Ground State Energy", fontweight="bold")
    ax1.set_xlabel("SNN Iteration Step")
    ax1.set_ylabel("Molecular Energy (Hartree)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, ls="--", alpha=0.5)

    # Panel 2: Neuromorphic LIF Neuron Spike Raster
    ax2 = axes[0, 1]
    for n_idx, neuron in enumerate(controller.neurons):
        spike_times = [t for t, spk in enumerate(neuron.spike_history) if spk == 1]
        ax2.scatter(
            spike_times,
            [n_idx + 1] * len(spike_times),
            marker="|",
            s=120,
            color="#d62728",
            lw=2.5,
        )
    ax2.set_title("Panel B: LIF Neuromorphic Neuron Spike Raster", fontweight="bold")
    ax2.set_xlabel("Simulation Time Steps (ms)")
    ax2.set_ylabel("Neuron Core Index")
    ax2.set_yticks(range(1, len(controller.neurons) + 1))
    ax2.set_ylim(0.5, len(controller.neurons) + 0.5)
    ax2.grid(True, ls="--", alpha=0.5)

    # Panel 3: Final Statevector Probability Amplitudes
    ax3 = axes[1, 0]
    ctx = cudaq._build_circuit(h2_uccsd_kernel, theta)
    sv = cudaq.simulate_statevector(ctx, 4)
    probs = np.abs(sv) ** 2
    basis_labels = [f"|{i:04b}>" for i in range(16)]
    # Filter basis states with non-negligible probability
    active_idx = [i for i, p in enumerate(probs) if p > 0.005]
    ax3.bar(
        [basis_labels[i] for i in active_idx],
        [probs[i] for i in active_idx],
        color="#9467bd",
        edgecolor="black",
        alpha=0.85,
    )
    ax3.set_title("Panel C: Final 4-Qubit Statevector Superposition (|ψ|²)", fontweight="bold")
    ax3.set_xlabel("Computational Basis State")
    ax3.set_ylabel("Measurement Probability")
    ax3.grid(True, ls="--", alpha=0.5)

    # Panel 4: Hardware Memory Telemetry (Host RAM vs GPU VRAM)
    ax4 = axes[1, 1]
    categories = ["Host RAM\n(Allocated)", "Host RAM\n(Available)", "GPU VRAM\n(Free)", "GPU VRAM\n(Total)"]
    ram_alloc = max(0.1, mem_sys["ram_total_gb"] - mem_sys["ram_available_gb"])
    values = [
        ram_alloc,
        mem_sys["ram_available_gb"],
        mem_sys["vram_available_gb"],
        mem_sys["vram_total_gb"],
    ]
    bar_colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]
    bars = ax4.bar(categories, values, color=bar_colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        h_val = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val + 0.3,
            f"{h_val:.2f} GiB",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax4.set_title("Panel D: Real-Time Hardware Resource Footprint", fontweight="bold")
    ax4.set_ylabel("Memory Capacity (GiB)")
    ax4.set_ylim(0, max(values) * 1.25)
    ax4.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[DASHBOARD] Snapshot rendered successfully to {save_path}")
    if interactive:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum-Neuromorphic Live Research Dashboard")
    parser.add_argument("--steps", type=int, default=30, help="SNN simulation steps (default: 30)")
    parser.add_argument("--save-path", type=str, default="results/dashboard_telemetry.png")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    args = parser.parse_args()

    render_dashboard(steps=args.steps, save_path=args.save_path, interactive=args.show)


if __name__ == "__main__":
    main()
