"""GHZ Statevector Simulation Benchmark: CPU vs GPU scaling with NVIDIA CUDA-Q."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import CUDA-Q runtime (native or hardware-adaptive bridge)
try:
    import cudaq
    from cudaq import h, mz, qvector, x
except (ImportError, ModuleNotFoundError):
    from src import cudaq_runtime as cudaq
    from src.cudaq_runtime import h, mz, qvector, x

from src.memory_estimator import (
    check_memory_fits,
    estimate_statevector_memory_gb,
    get_available_system_memory,
)


@cudaq.kernel
def ghz(n: int):
    """NVIDIA CUDA-Q kernel generating an n-qubit Greenberger-Horne-Zeilinger (GHZ) state.

    State: (|00...0> + |11...1>) / sqrt(2)
    """
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


def verify_ghz_correctness(n: int = 8, shots: int = 2000, tolerance: float = 0.15) -> bool:
    """Verify correctness of the GHZ circuit sampling at small n.

    Asserts:
    1. Exactly and only |0...0> and |1...1> bitstrings appear.
    2. Probabilities are balanced within the specified statistical tolerance (~50%).
    """
    print(f"--> [VERIFY] Running GHZ correctness verification on n={n} qubits ({shots} shots)...")
    res = cudaq.sample(ghz, n, shots_count=shots)

    expected_0 = "0" * n
    expected_1 = "1" * n

    observed_keys = set(res.keys())
    allowed_keys = {expected_0, expected_1}

    unexpected = observed_keys - allowed_keys
    if unexpected:
        raise AssertionError(
            f"GHZ correctness check failed! Unexpected bitstrings observed: {unexpected}"
        )

    count_0 = res[expected_0]
    count_1 = res[expected_1]
    prob_0 = count_0 / shots
    prob_1 = count_1 / shots

    print(f"    Observed: |{expected_0}> = {count_0} ({prob_0:.1%}), |{expected_1}> = {count_1} ({prob_1:.1%})")

    if abs(prob_0 - 0.5) > tolerance or abs(prob_1 - 0.5) > tolerance:
        raise AssertionError(
            f"GHZ bitstring distribution exceeds tolerance {tolerance}: "
            f"P(0)={prob_0:.3f}, P(1)={prob_1:.3f}"
        )

    print("    [PASS] GHZ state preparation and measurement verified successfully!")
    return True


def run_single_benchmark(
    n: int,
    target: str,
    runs: int = 5,
    shots: int = 1000,
) -> Tuple[float, float, float]:
    """Execute 1 warm-up run and timed benchmark runs for an n-qubit GHZ state.

    Returns:
        (median_seconds, std_seconds, estimated_memory_gb)
    """
    est_mem_gb = estimate_statevector_memory_gb(n, precision="complex64")

    # Warm-up run (ensures JIT / backend initialization)
    _ = cudaq.sample(ghz, n, shots_count=shots)

    # Timed runs
    durations = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = cudaq.sample(ghz, n, shots_count=shots)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    median_s = float(np.median(durations))
    std_s = float(np.std(durations))

    return median_s, std_s, est_mem_gb


def run_ghz_benchmarks(
    min_qubits: int = 20,
    max_qubits: int = 30,
    targets: Sequence[str] = ("qpp-cpu", "nvidia"),
    runs: int = 5,
    shots: int = 1000,
    output_csv: str = "results/ghz_benchmark.csv",
) -> pd.DataFrame:
    """Benchmark GHZ state simulation across qubit counts on specified targets."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    records = []

    mem_sys = get_available_system_memory()
    print("=" * 70)
    print("  CUDA-Q GHZ STATEVECTOR SIMULATION BENCHMARK")
    print(f"  Qubit range: {min_qubits} to {max_qubits} | Timed runs per point: {runs}")
    print(f"  Host RAM: {mem_sys['ram_available_gb']:.2f} GiB free of {mem_sys['ram_total_gb']:.2f} GiB")
    print(f"  GPU VRAM: {mem_sys['vram_available_gb']:.2f} GiB free of {mem_sys['vram_total_gb']:.2f} GiB")
    print("=" * 70)

    for target in targets:
        print(f"\n>>> Evaluating Target: [{target.upper()}]")
        is_gpu_target = "nvidia" in target.lower()

        # Check hardware capability for nvidia target
        if is_gpu_target:
            native_avail = hasattr(cudaq, "is_native_cudaq_available") and cudaq.is_native_cudaq_available()
            has_gpu = mem_sys["vram_total_gb"] > 0.1
            if not has_gpu:
                print(f"  [SKIP TARGET] No active NVIDIA GPU detected. Skipping target '{target}'.")
                continue
            if sys.platform.startswith("win") and not native_avail:
                print(
                    f"  [NOTICE] NVIDIA GPU ({mem_sys['vram_total_gb']:.1f} GB) detected on Windows host. "
                    f"Native CUDA-Q cuStateVec targets require Linux/WSL2. "
                    f"Executing CPU benchmark suite; Colab T4 notebook provided for full GPU reproduction."
                )
                continue

        try:
            cudaq.set_target(target)
        except Exception as e:
            print(f"  [SKIP TARGET] Unable to initialize target '{target}': {e}")
            continue

        for n in range(min_qubits, max_qubits + 1):
            fits, reason = check_memory_fits(n, target=target, precision="complex64")
            est_mem_gb = estimate_statevector_memory_gb(n, precision="complex64")

            if not fits:
                print(f"  [MEM SKIP] n={n:2d} ({est_mem_gb:8.4f} GiB) -> {reason}")
                continue

            print(f"  Benchmarking n={n:2d} (est. {est_mem_gb:8.4f} GiB)...", end="", flush=True)
            try:
                median_s, std_s, _ = run_single_benchmark(n, target=target, runs=runs, shots=shots)
                print(f" Done! Median: {median_s:8.5f}s (±{std_s:8.5f}s)")
                records.append({
                    "target": target,
                    "n": n,
                    "median_s": median_s,
                    "std_s": std_s,
                    "est_mem_gb": est_mem_gb,
                })
            except Exception as e:
                print(f" FAILED: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"\n[SAVED] Benchmark results written to {output_csv}")
    else:
        print("\n[WARNING] No benchmark points were successfully completed.")

    return df


def plot_ghz_results(
    csv_path: str = "results/ghz_benchmark.csv",
    output_runtime_png: str = "results/ghz_benchmark.png",
    output_speedup_png: str = "results/ghz_speedup.png",
) -> None:
    """Generate publication-quality plots of GHZ simulation runtime and speedup."""
    if not os.path.exists(csv_path):
        print(f"[PLOT SKIP] CSV file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[PLOT SKIP] DataFrame is empty.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_runtime_png)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_speedup_png)), exist_ok=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # Plot 1: Runtime vs n on log y-axis
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = {"qpp-cpu": "#1f77b4", "nvidia": "#2ca02c", "default": "#ff7f0e"}
    markers = {"qpp-cpu": "o", "nvidia": "s", "default": "^"}

    for target in df["target"].unique():
        sub = df[df["target"] == target].sort_values("n")
        ax.errorbar(
            sub["n"],
            sub["median_s"],
            yerr=sub["std_s"],
            label=f"{target.upper()} (fp32)",
            color=colors.get(target, "#333333"),
            marker=markers.get(target, "o"),
            linewidth=2,
            capsize=4,
            alpha=0.9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Qubits ($n$)", fontweight="bold")
    ax.set_ylabel("Execution Time (seconds) [Log Scale]", fontweight="bold")
    ax.set_title("NVIDIA CUDA-Q GHZ Statevector Simulation Runtime", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_runtime_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved runtime scaling plot to {output_runtime_png}")

    # Plot 2: Speedup (CPU / GPU) vs n
    has_both = ("qpp-cpu" in df["target"].values) and ("nvidia" in df["target"].values)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    if has_both:
        cpu_sub = df[df["target"] == "qpp-cpu"].set_index("n")
        gpu_sub = df[df["target"] == "nvidia"].set_index("n")
        common_n = cpu_sub.index.intersection(gpu_sub.index).sort_values()
        if len(common_n) > 0:
            speedup = cpu_sub.loc[common_n, "median_s"] / gpu_sub.loc[common_n, "median_s"]
            ax.plot(
                common_n,
                speedup,
                marker="D",
                color="#d62728",
                linewidth=2.5,
                label="Measured Speedup (CPU / GPU)",
            )
            ax.axhline(1.0, color="gray", linestyle=":", label="Parity (1x)")
            ax.set_ylabel("Speedup Factor ($T_{CPU} / T_{GPU}$)", fontweight="bold")
    else:
        # Single target present (e.g. Host CPU run without Linux GPU wheel)
        sub = df.sort_values("n")
        # Plot relative scaling normalized to smallest n
        t_base = sub["median_s"].iloc[0]
        scaling = sub["median_s"] / t_base
        ax.plot(
            sub["n"],
            scaling,
            marker="o",
            color="#1f77b4",
            linewidth=2,
            label=f"Relative Scaling ({sub['target'].iloc[0].upper()})",
        )
        ax.set_ylabel(f"Relative Runtime ($T(n) / T(n={sub['n'].iloc[0]})$)", fontweight="bold")
        ax.text(
            0.5,
            0.85,
            "NVIDIA GPU speedup available via Colab T4 / Linux cuStateVec\n(See notebooks/colab_quickstart.ipynb)",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656"),
        )

    ax.set_xlabel("Number of Qubits ($n$)", fontweight="bold")
    ax.set_title("NVIDIA CUDA-Q Simulation Acceleration Scaling", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_speedup_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved speedup scaling plot to {output_speedup_png}")


def main():
    parser = argparse.ArgumentParser(description="CUDA-Q GHZ Statevector Simulation Benchmark")
    parser.add_argument("--min-qubits", type=int, default=20, help="Minimum qubits (default: 20)")
    parser.add_argument("--max-qubits", type=int, default=30, help="Maximum qubits (default: 30)")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per qubit count (default: 5)")
    parser.add_argument("--shots", type=int, default=1000, help="Measurement shots (default: 1000)")
    parser.add_argument("--output-csv", type=str, default="results/ghz_benchmark.csv")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run n=20..24 in < 1 minute")
    args = parser.parse_args()

    # Step 1: Run correctness verification
    verify_ghz_correctness(n=8, shots=2000)

    # Step 2: Configure qubit range
    min_q = args.min_qubits
    max_q = 24 if args.quick else args.max_qubits
    runs = 3 if args.quick else args.runs

    # Step 3: Run benchmarks
    df = run_ghz_benchmarks(
        min_qubits=min_q,
        max_qubits=max_q,
        targets=["qpp-cpu", "nvidia"],
        runs=runs,
        shots=args.shots,
        output_csv=args.output_csv,
    )

    # Step 4: Generate plots
    plot_ghz_results(
        csv_path=args.output_csv,
        output_runtime_png="results/ghz_benchmark.png",
        output_speedup_png="results/ghz_speedup.png",
    )


if __name__ == "__main__":
    main()
