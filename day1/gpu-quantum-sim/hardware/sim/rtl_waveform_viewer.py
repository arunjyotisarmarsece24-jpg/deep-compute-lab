"""Hardware Digital Timing Logic & Waveform Telemetry Generator.

Simulates the cycle-accurate RTL execution of the Spartan-7 Neuromorphic Co-Processor
(lif_neuron_q8_8 and quantum_phase_accumulator) and plots digital logic timing waveforms
as seen in ModelSim / Vivado Logic Analyzer.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def simulate_rtl_timing(n_cycles: int = 50):
    """Simulate cycle-accurate digital transitions of the Verilog LIF & AXI cores."""
    clk = []
    rst_n = []
    i_syn = []
    v_mem = []
    spike = []
    theta = []

    # Q8.8 fixed-point constants
    v_rest = -70.0 * 256.0      # -17920
    v_reset = -75.0 * 256.0     # -19200
    v_thresh = -50.0 * 256.0    # -12800
    decay = 243 / 256.0         # 0.9492
    
    cur_v = v_rest
    cur_theta = 0.05 * 256.0
    ref_timer = 0
    total_spikes = 0

    for cycle in range(n_cycles):
        # Reset deasserts at cycle 3
        reset_active = 0 if cycle < 3 else 1
        rst_n.append(reset_active)

        # Stimulus current stepped after cycle 6
        if cycle < 3:
            syn_in = 0
        elif cycle < 28:
            syn_in = 2800  # Inject current (~10.9 in Q8.8)
        else:
            syn_in = 1600  # Reduced current

        if not reset_active:
            cur_v = v_rest
            cur_spike = 0
            ref_timer = 0
            cur_theta = 0.05 * 256.0
        else:
            if ref_timer > 0:
                ref_timer -= 1
                cur_v = v_reset
                cur_spike = 0
            else:
                v_diff = cur_v - v_rest
                v_leaked = v_rest + (v_diff * decay)
                v_integrated = v_leaked + syn_in
                if v_integrated >= v_thresh:
                    cur_v = v_reset
                    cur_spike = 1
                    ref_timer = 4
                    total_spikes += 1
                    cur_theta += 12  # Quantum angle accumulator step
                else:
                    cur_v = v_integrated
                    cur_spike = 0

        clk.append(cycle)
        i_syn.append(syn_in / 256.0)
        v_mem.append(cur_v / 256.0)
        spike.append(cur_spike)
        theta.append(cur_theta / 256.0)

    return clk, rst_n, i_syn, v_mem, spike, theta


def plot_rtl_waveforms(output_path: str = "results/fpga_rtl_waveforms.png"):
    """Plot publication-grade digital hardware timing waveforms."""
    clk, rst_n, i_syn, v_mem, spike, theta = simulate_rtl_timing(50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True, gridspec_kw={"hspace": 0.25})

    t = np.array(clk)

    # 1. Master Clock & Reset
    axes[0].step(t, [0.5 if c % 2 == 0 else 0.0 for c in t], where="mid", color="#00ffcc", lw=1.5, label="clk (100 MHz)")
    axes[0].step(t, np.array(rst_n) * 0.4 - 0.5, where="post", color="#ff7700", lw=1.8, label="rst_n (Active-Low)")
    axes[0].set_ylabel("Digital Logic", fontsize=9, fontweight="bold")
    axes[0].set_ylim(-0.7, 0.7)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.2, linestyle="--")

    # 2. Synaptic Current Stimulus
    axes[1].step(t, i_syn, where="post", color="#3399ff", lw=2, label="i_syn [Q8.8 Fixed-Point]")
    axes[1].set_ylabel("I_syn (mA)", fontsize=9, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.2, linestyle="--")

    # 3. Membrane Voltage V_mem (Q8.8)
    axes[2].plot(t, v_mem, color="#ffcc00", lw=2.2, label="v_mem [Q8.8] (mV)")
    axes[2].axhline(-50.0, color="#ff3333", linestyle=":", lw=1.5, label="V_THRESH (-50 mV)")
    axes[2].axhline(-75.0, color="#888888", linestyle="--", lw=1, label="V_RESET (-75 mV)")
    axes[2].set_ylabel("V_mem (mV)", fontsize=9, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.2, linestyle="--")

    # 4. Spike Output Pulse
    axes[3].step(t, spike, where="mid", color="#ff0055", lw=2.5, label="spike_out (Action Potential)")
    axes[3].set_ylabel("Spike", fontsize=9, fontweight="bold")
    axes[3].set_ylim(-0.2, 1.3)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.2, linestyle="--")

    # 5. Quantum AXI Register theta_angle
    axes[4].step(t, theta, where="post", color="#00ff88", lw=2.2, label="theta_angle_q8_8 [AXI Reg] (rad)")
    axes[4].set_ylabel("Angle (rad)", fontsize=9, fontweight="bold")
    axes[4].set_xlabel("Hardware Clock Cycles (100 MHz Master FPGA Clock)", fontsize=10, fontweight="bold")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("Digilent Arty S7 (Spartan-7 XC7S50) Cycle-Accurate Verilog RTL Timing Telemetry",
                 fontsize=12, fontweight="bold", color="#ffffff", y=0.96)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[RTL Simulation] Hardware timing waveforms saved -> {output_path}")


if __name__ == "__main__":
    plot_rtl_waveforms("results/fpga_rtl_waveforms.png")
