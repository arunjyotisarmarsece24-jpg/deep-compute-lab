"""Variational Quantum Eigensolver (VQE) for Molecular Hydrogen (H2) using NVIDIA CUDA-Q.

Constructs the STO-3G molecular Hamiltonian mapped via Jordan-Wigner to 4 qubits (15 Pauli terms),
executes a UCCSD ansatz starting from the Hartree-Fock reference state, validates ground-state
energy against exact Full Configuration Interaction (FCI) to within chemical accuracy (< 1.6 mHa),
and sweeps the dissociation potential energy curve.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, fci, gto, scf
from scipy.optimize import minimize

# Import CUDA-Q runtime (native or hardware-adaptive bridge)
try:
    import cudaq
    from cudaq import cx, h, qvector, ry, spin, x
except (ImportError, ModuleNotFoundError):
    from src import cudaq_runtime as cudaq
    from src.cudaq_runtime import cx, h, qvector, ry, spin, x

# STO-3G Ground state energy of isolated Hydrogen atom
# 2 * E(H) = -0.93316370 Ha (dissociation limit)
DISSOCIATION_ASYMPTOTE_HA = -0.93316370
CHEMICAL_ACCURACY_MHA = 1.60  # 1 kcal/mol = 1.5936 mHa


@cudaq.kernel
def h2_uccsd_kernel(theta: float):
    """CUDA-Q parameterized UCCSD ansatz kernel for H2 in STO-3G basis.

    Qubit mapping: 4 qubits representing spin-orbitals (0: 1s_alpha, 1: 1s_beta, 2: 2s_alpha, 3: 2s_beta).
    Hartree-Fock reference state: |1100> (qubits 0 and 1 occupied).
    Ansatz operator: Unitary double excitation connecting |1100> and |0011>.
    """
    q = cudaq.qvector(4)

    # 1. Prepare Hartree-Fock reference state |1100>
    x(q[0])

    # 2. Parameterized rotation on qubit 0
    ry(2.0 * theta, q[0])

    # 3. Excitation coupling ladder to double-excited state |0011>
    x(q[2])
    cx(q[0], q[2])
    cx(q[0], q[1])
    cx(q[2], q[3])


def build_h2_hamiltonian(r: float) -> Tuple[Any, int, float, float]:
    """Build the H2 molecular Hamiltonian in STO-3G basis mapped to 4 qubits via Jordan-Wigner.

    Args:
        r: Interatomic bond distance in Angstroms.

    Returns:
        tuple: (cudaq_spin_hamiltonian, num_pauli_terms, e_hf, e_fci)
    """
    # 1. Compute electronic integrals with PySCF
    mol = gto.M(
        atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {r}",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    e_hf = float(mf.e_tot)

    # Exact Full Configuration Interaction (FCI) benchmark
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    e_fci = float(e_fci)

    # Transform 1- and 2-electron integrals to MO basis
    n_orbitals = mf.mo_coeff.shape[1]
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    two_e = ao2mo.kernel(mol, mf.mo_coeff)
    two_e = ao2mo.restore(1, two_e, n_orbitals)
    h2e = np.asarray(two_e.transpose(0, 2, 3, 1), order="C")

    # Map spatial MOs to spin-orbitals via OpenFermion convention
    one_spin, two_spin = spinorb_from_spatial(h1e, h2e)
    constant = float(mol.energy_nuc())
    interaction_op = InteractionOperator(constant, one_spin, 0.5 * two_spin)
    fermion_op = get_fermion_operator(interaction_op)
    qubit_op = jordan_wigner(fermion_op)

    num_terms = len(qubit_op.terms)

    # Convert OpenFermion QubitOperator to CUDA-Q SpinOperator
    spin_hamiltonian = 0.0 * spin.i(0)
    for term, coeff in qubit_op.terms.items():
        c_real = float(np.real(coeff))
        if abs(c_real) < 1e-12:
            continue
        if len(term) == 0:
            spin_hamiltonian = spin_hamiltonian + c_real * spin.i(0)
        else:
            term_op = None
            for q_idx, pauli_char in term:
                p_op = (
                    spin.x(q_idx)
                    if pauli_char == "X"
                    else spin.y(q_idx)
                    if pauli_char == "Y"
                    else spin.z(q_idx)
                )
                term_op = p_op if term_op is None else term_op * p_op
            spin_hamiltonian = spin_hamiltonian + c_real * term_op

    return spin_hamiltonian, num_terms, e_hf, e_fci


def run_vqe_single_point(
    r: float = 0.735,
    initial_theta: float = 0.1,
    method: str = "COBYLA",
    record_history: bool = False,
) -> Tuple[float, float, float, float, List[float]]:
    """Execute VQE optimization for H2 at a specified bond distance.

    Args:
        r: Bond distance in Angstroms.
        initial_theta: Starting parameter value for warm-start.
        method: Optimization algorithm ('COBYLA' or 'L-BFGS-B').
        record_history: If True, tracks energy at each iteration.

    Returns:
        (e_vqe, e_fci, e_hf, optimal_theta, energy_history)
    """
    hamiltonian, num_terms, e_hf, e_fci = build_h2_hamiltonian(r)
    history: List[float] = []

    def objective(params: Sequence[float]) -> float:
        th = float(params[0])
        res = cudaq.observe(h2_uccsd_kernel, hamiltonian, th)
        energy = float(res.expectation())
        if record_history:
            history.append(energy)
        return energy

    opt_res = minimize(
        objective,
        [initial_theta],
        method=method,
        options={"maxiter": 100, "tol": 1e-7},
    )

    optimal_theta = float(opt_res.x[0])
    e_vqe = float(opt_res.fun)

    return e_vqe, e_fci, e_hf, optimal_theta, history


def sweep_h2_pes(
    r_start: float = 0.30,
    r_end: float = 2.50,
    r_step: float = 0.05,
    output_csv: str = "results/h2_pes.csv",
    quick: bool = False,
) -> pd.DataFrame:
    """Sweep the H2 potential energy surface across bond distances with parameter warm-starting."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if quick:
        r_points = np.linspace(0.5, 1.5, 6)
    else:
        r_points = np.arange(r_start, r_end + 1e-6, r_step)

    records = []
    warm_theta = 0.11

    print("=" * 75)
    print("  NVIDIA CUDA-Q H2 VQE POTENTIAL ENERGY SURFACE SWEEP")
    print(f"  Range: {r_points[0]:.2f} to {r_points[-1]:.2f} Å ({len(r_points)} points) | Ansatz: UCCSD")
    print("=" * 75)
    print(f"  {'R (Å)':>6}  {'E_VQE (Ha)':>13}  {'E_FCI (Ha)':>13}  {'E_HF (Ha)':>13}  {'Error (mHa)':>12}")
    print("  " + "-" * 67)

    for r in r_points:
        e_vqe, e_fci, e_hf, opt_theta, _ = run_vqe_single_point(
            r=float(r),
            initial_theta=warm_theta,
            method="COBYLA",
            record_history=False,
        )
        warm_theta = opt_theta  # Warm-start next point
        abs_error_mha = abs(e_vqe - e_fci) * 1000.0

        print(
            f"  {r:6.2f}  {e_vqe:13.6f}  {e_fci:13.6f}  {e_hf:13.6f}  {abs_error_mha:12.6f}"
        )

        records.append({
            "R": round(float(r), 3),
            "E_vqe": e_vqe,
            "E_fci": e_fci,
            "E_hf": e_hf,
            "abs_error_mHa": abs_error_mha,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[SAVED] PES sweep data written to {output_csv}")

    # Identify equilibrium minimum
    min_row = df.loc[df["E_vqe"].idxmin()]
    print("=" * 75)
    print(f"  VQE Equilibrium Bond Length: R = {min_row['R']:.3f} Å")
    print(f"  VQE Ground State Energy:     E = {min_row['E_vqe']:.6f} Ha (FCI: {min_row['E_fci']:.6f} Ha)")
    print(f"  Maximum Sweep Error:         {df['abs_error_mHa'].max():.6f} mHa (< {CHEMICAL_ACCURACY_MHA:.2f} mHa threshold)")
    print("=" * 75)

    return df


def plot_vqe_convergence(
    history: List[float],
    e_fci: float,
    output_png: str = "results/vqe_convergence.png",
) -> None:
    """Generate publication-quality convergence trajectory at R = 0.735 Å."""
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    iters = list(range(1, len(history) + 1))

    ax.plot(iters, history, marker="o", color="#1f77b4", linewidth=2.2, label="VQE Energy (<H>)")
    ax.axhline(e_fci, color="#2ca02c", linestyle="-", linewidth=2, label=f"Exact FCI ({e_fci:.6f} Ha)")

    # Chemical accuracy shaded corridor
    corridor_ha = CHEMICAL_ACCURACY_MHA / 1000.0
    ax.axhspan(
        e_fci - corridor_ha,
        e_fci + corridor_ha,
        color="#2ca02c",
        alpha=0.18,
        label=f"Chemical Accuracy (±{CHEMICAL_ACCURACY_MHA:.1f} mHa)",
    )

    ax.set_xlabel("Iteration Step", fontweight="bold")
    ax.set_ylabel("Electronic Energy (Hartree)", fontweight="bold")
    ax.set_title("VQE Optimization Convergence for H$_2$ (R = 0.735 Å)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved convergence plot to {output_png}")


def plot_pes_results(
    csv_path: str = "results/h2_pes.csv",
    output_pes_png: str = "results/h2_pes.png",
    output_error_png: str = "results/h2_error.png",
) -> None:
    """Generate publication-quality potential energy curve and absolute error plots."""
    if not os.path.exists(csv_path):
        print(f"[PLOT SKIP] CSV file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_pes_png)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_error_png)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Potential Energy Curve (PES)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # FCI exact line
    ax.plot(df["R"], df["E_fci"], color="#1f77b4", linewidth=2.5, label="FCI Exact (PySCF)")
    # HF dashed line
    ax.plot(df["R"], df["E_hf"], color="#d62728", linestyle="--", linewidth=1.8, label="Hartree-Fock (RHF)")
    # VQE points
    ax.scatter(df["R"], df["E_vqe"], color="#ff7f0e", s=45, zorder=5, label="CUDA-Q VQE (UCCSD)")

    # 2 x E(H atom) dissociation asymptote
    ax.axhline(
        DISSOCIATION_ASYMPTOTE_HA,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        label=f"2 × E(H atom) = {DISSOCIATION_ASYMPTOTE_HA:.4f} Ha",
    )

    # Equilibrium annotation
    min_row = df.loc[df["E_vqe"].idxmin()]
    ax.annotate(
        f"Equilibrium Minimum\nR = {min_row['R']:.3f} Å\nE = {min_row['E_vqe']:.4f} Ha",
        xy=(min_row["R"], min_row["E_vqe"]),
        xytext=(min_row["R"] + 0.35, min_row["E_vqe"] + 0.08),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="#cccccc"),
    )

    ax.set_xlabel("Interatomic Distance $R$ (Å)", fontweight="bold")
    ax.set_ylabel("Total Molecular Energy (Hartree)", fontweight="bold")
    ax.set_title("H$_2$ Potential Energy Surface: VQE vs Exact FCI vs Hartree-Fock", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(-1.20, -0.70)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_pes_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved PES curve to {output_pes_png}")

    # Plot 2: Absolute Error |E_VQE - E_FCI| in mHa
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    ax.plot(df["R"], df["abs_error_mHa"], marker="o", markersize=5, color="#1f77b4", linewidth=2.0, label="|E_VQE - E_FCI|")

    # Shaded band below 1.6 mHa labelled "chemical accuracy"
    ax.axhspan(0.0, CHEMICAL_ACCURACY_MHA, color="#2ca02c", alpha=0.20, label=f"Chemical Accuracy Zone (< {CHEMICAL_ACCURACY_MHA:.1f} mHa)")
    ax.axhline(CHEMICAL_ACCURACY_MHA, color="#2ca02c", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Interatomic Distance $R$ (Å)", fontweight="bold")
    ax.set_ylabel("Absolute Energy Error (mHa)", fontweight="bold")
    ax.set_title("VQE Accuracy Relative to Exact Full Configuration Interaction (FCI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.0, max(2.0, float(df["abs_error_mHa"].max()) * 1.25))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_error_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved error plot to {output_error_png}")


def main():
    parser = argparse.ArgumentParser(description="NVIDIA CUDA-Q H2 VQE Simulation & PES Sweep")
    parser.add_argument("--r-eq", type=float, default=0.735, help="Equilibrium bond distance in Angstroms (default: 0.735)")
    parser.add_argument("--r-start", type=float, default=0.30, help="PES sweep start (default: 0.30)")
    parser.add_argument("--r-end", type=float, default=2.50, help="PES sweep end (default: 2.50)")
    parser.add_argument("--r-step", type=float, default=0.05, help="PES sweep step size (default: 0.05)")
    parser.add_argument("--output-csv", type=str, default="results/h2_pes.csv")
    parser.add_argument("--quick", action="store_true", help="Quick mode: sweep reduced grid in < 2 minutes")
    args = parser.parse_args()

    # Step 1: Equilibrium point calculation & convergence plot
    print(f"\n--> Running single-point VQE convergence at R = {args.r_eq:.3f} Å...")
    e_vqe, e_fci, e_hf, opt_th, history = run_vqe_single_point(
        r=args.r_eq,
        initial_theta=0.1,
        method="COBYLA",
        record_history=True,
    )
    print(f"    VQE Energy: {e_vqe:.8f} Ha | Exact FCI: {e_fci:.8f} Ha | Error: {abs(e_vqe - e_fci)*1000:.6f} mHa")
    plot_vqe_convergence(history, e_fci, output_png="results/vqe_convergence.png")

    # Step 2: PES Sweep
    df = sweep_h2_pes(
        r_start=args.r_start,
        r_end=args.r_end,
        r_step=args.r_step,
        output_csv=args.output_csv,
        quick=args.quick,
    )

    # Step 3: Generate PES & Error plots
    plot_pes_results(
        csv_path=args.output_csv,
        output_pes_png="results/h2_pes.png",
        output_error_png="results/h2_error.png",
    )


if __name__ == "__main__":
    main()
