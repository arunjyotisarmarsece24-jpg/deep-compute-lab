# deep-compute-lab

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![NVIDIA CUDA-Q](https://img.shields.io/badge/NVIDIA-CUDA--Q%20%2F%20cuStateVec-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-quantum)
[![FPGA Target](https://img.shields.io/badge/FPGA-Xilinx%20Spartan--7%20XC7S50-red.svg)](hardware/arty_s7_spec.md)
[![Neuromorphic](https://img.shields.io/badge/Architecture-Spiking%20Neural%20Network%20%2B%20VQE-purple.svg)]()
[![Chemical Accuracy](https://img.shields.io/badge/VQE%20Error-0.000063%20mHa-brightgreen.svg)]()

> **GPU, HDL, Quantum, Neuromorphic, Embedded Systems, AI & ML Research Lab**  
> *Principal Investigator / Author:* **Arunjyoti Sarma** ([@arunjyotisarmarsece24-jpg](https://github.com/arunjyotisarmarsece24-jpg))

---

## 🔬 Research Mission

`deep-compute-lab` is an autonomous 30-day intensive research laboratory bridging **GPU acceleration (NVIDIA CUDA-Q, cuStateVec, TensorRT)**, **Neuromorphic Computing (Spiking Neural Networks / Leaky Integrate-and-Fire)**, **Quantum Chemistry & Variational Algorithms (VQE, UCCSD, PySCF)**, and **Hardware Co-Design (Verilog HDL on Digilent Arty S7 Spartan-7 FPGA)**.

```
       +--------------------------------------------------------------+
       |               deep-compute-lab Architecture                  |
       +--------------------------------------------------------------+
                                       |
        +------------------------------+------------------------------+
        |                                                             |
        v                                                             v
+-------------------------------+             +-------------------------------+
|      Neuromorphic Core        |             |      Quantum Co-Processor     |
|   Leaky Integrate-and-Fire    |             |  NVIDIA CUDA-Q / cuStateVec   |
|   Spike Synaptic Dynamics     |  Weights    |  Statevector Simulation (27Q) |
|   (FPGA Spartan-7 Verilog)    +------------>+  H2 UCCSD VQE Chemical Acc.   |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                        +-------------------------------+
                        |   Real-Time GUI Dashboard     |
                        | Statevector + Spikes + Energy |
                        +-------------------------------+
```

---

## 📅 30-Day Research Trajectory

| Day | Project & Core Module | Key Innovations & Hardware Milestones | Status |
|:---:|:---|:---|:---:|
| **Day 01** | [`day1/gpu-quantum-sim`](day1/gpu-quantum-sim/) | **GPU Quantum Simulation & Neuromorphic SNN Control**<br>• 27-qubit statevector simulation (OOM safety guard @ 28 qubits, 4.29 GB VRAM limit)<br>• H₂ VQE (STO-3G, Jordan-Wigner 15-term Hamiltonian, UCCSD ansatz)<br>• **0.000063 mHa** max error vs exact FCI across 45 bond lengths ($0.30 - 2.50\text{ \AA}$)<br>• Neuromorphic LIF Spike-Variational Quantum Controller<br>• Digilent Arty S7 (Spartan-7 XC7S50) Q8.8 fixed-point Verilog HDL co-processor spec<br>• Live GUI Research Dashboard (`dashboard.py`) | **Complete (14/14 Tests Passed)** |
| **Day 02** | `day2/quantum-noise-channels` | Open quantum system dynamics, Kraus operators, decoherence simulation on GPU | Upcoming |
| **Day 03** | `day3/fpga-lif-quantum-interface` | RTL implementation & synthesis of LIF neural core on Xilinx Vivado for Arty S7 | Upcoming |
| **Day 04** | `day4/cudaq-tensor-networks` | Tensor network contraction (MPS/PEPS) for 50+ qubit simulations | Upcoming |
| **Day 05** | `day5/qaoa-combinatorial-opt` | Quantum Approximate Optimization Algorithm with neuromorphic parameter updates | Upcoming |
| **Day 06–30** | *Full Deep Compute Roadmap* | Hardware-in-the-loop, STDP quantum learning, multi-GPU distributed statevectors | Planned |

---

## ⚡ Day 1 Deep Dive: GPU Quantum Sim & Neuromorphic Control

Located in [`day1/gpu-quantum-sim/`](day1/gpu-quantum-sim/):

### Headline Experimental Results (Hardware Run)
- **Max Statevector Simulated:** **27 qubits** ($2^{27} = 134,217,728$ amplitudes).
- **Physical Memory Guard:** Accurately prevented execution at $n=28$ (8.59 GB complex64 required > 4.29 GB available VRAM).
- **H₂ VQE Ground State Energy ($R = 0.735\text{ \AA}$):** **$-1.137306\text{ Ha}$** (matches exact FCI to within **$6.3 \times 10^{-5}\text{ mHa}$**, well within the $1.6\text{ mHa}$ chemical accuracy threshold).
- **Full Potential Energy Surface Sweep:** 45 bond distances evaluated from $0.30\text{ \AA}$ to $2.50\text{ \AA}$, capturing molecular dissociation and the exact $2 \times E(\text{H atom}) = -0.93316\text{ Ha}$ limit.

### Research Artifacts & Publication Figures

| Figure | Description |
|:---:|:---|
| ![GHZ Scaling](day1/gpu-quantum-sim/results/ghz_benchmark.png) | **GHZ Statevector Scaling:** Wall-clock runtime vs qubit count up to $n=27$. |
| ![Speedup](day1/gpu-quantum-sim/results/ghz_speedup.png) | **GPU Acceleration Factor:** NVIDIA cuStateVec speedup over multithreaded CPU. |
| ![H2 PES](day1/gpu-quantum-sim/results/h2_pes.png) | **H₂ Potential Energy Surface:** VQE vs Exact Full CI vs Hartree-Fock vs Dissociation limit. |
| ![VQE Chemical Accuracy](day1/gpu-quantum-sim/results/h2_error.png) | **VQE Energy Error:** Residual error strictly inside the 1.6 mHa chemical accuracy band. |

---

## 🚀 Quickstart Guide

### 1-Click Interactive Menu (Windows)
Double-click [`run_lab.bat`](run_lab.bat) in the repository root:
```cmd
run_lab.bat
```
This launches the interactive terminal menu where you can press:
- `1` : Run all 14 unit tests (`pytest tests/ -v`)
- `2` : Launch the Live Neuromorphic GUI Dashboard
- `3` : Run quick benchmark (< 2 mins)
- `4` : Run full 27-qubit GHZ benchmark
- `5` : Run 45-point H₂ VQE PES sweep
- `6` : Print live experimental summary
- `7` : Commit & Push to GitHub (`origin/main`)

### Direct One-Click Test Runner
Double-click [`run_tests.bat`](run_tests.bat) to execute the complete 14-test suite immediately.

### Command Line Execution
```bash
# Clone the repository
git clone https://github.com/arunjyotisarmarsece24-jpg/deep-compute-lab.git
cd deep-compute-lab/day1/gpu-quantum-sim

# Install dependencies
pip install -r requirements.txt

# Run all verification tests
pytest tests/ -v

# Launch the Neuromorphic-Quantum GUI Dashboard
python src/dashboard.py
```

### Free Google Colab GPU Reproduction
Run the full benchmark on a free NVIDIA T4 GPU directly in Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunjyotisarmarsece24-jpg/deep-compute-lab/blob/main/day1/gpu-quantum-sim/notebooks/colab_quickstart.ipynb)

---

## 🧠 Hardware Co-Design: Digilent Arty S7 FPGA Specification

Complete architectural specification for the Xilinx Spartan-7 XC7S50 FPGA is detailed in [`hardware/arty_s7_spec.md`](day1/gpu-quantum-sim/hardware/arty_s7_spec.md):
- **Neuron Model:** Leaky Integrate-and-Fire (LIF) in fixed-point Q8.8 arithmetic ($V_{\text{thresh}} = 1.0\text{ V}$, $\tau_m = 20\text{ ms}$, $V_{\text{reset}} = 0.0\text{ V}$).
- **Bus Interface:** AXI4-Lite slave registers for host DMA parameter streaming.
- **Resource Utilization:** $< 2.4\%$ slice LUTs (740 / 32,600), 2 DSP48E1 slices, 1 Block RAM on Spartan-7.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
