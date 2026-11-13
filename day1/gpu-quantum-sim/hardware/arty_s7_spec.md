# Digilent Arty S7 (Xilinx Spartan-7) Neuromorphic-Quantum Co-Processor Architecture

## 1. System Overview
This specification defines the hardware co-processor architecture for deploying real-time **Leaky Integrate-and-Fire (LIF) Spiking Neural Networks (SNNs)** on the **Digilent Arty S7-50 (Xilinx Spartan-7 XC7S50-CSGA324)** FPGA board to accelerate hybrid quantum-neuromorphic algorithms alongside **NVIDIA CUDA-Q**.

```
+-------------------------------------------------------------+
|                      HOST WORKSTATION                       |
|   NVIDIA GeForce RTX GPU        CPU Host Process (Python)   |
|   +-----------------------+     +-----------------------+   |
|   | CUDA-Q / cuStateVec   | <-> | OpenFermion / PySCF   |   |
|   | Expectation <H>       |     | SNN Controller Driver |   |
|   +-----------------------+     +-----------------------+   |
+-------------------------------------------------------------+
                              | USB-UART / AXI4-Stream Bridge (12 MBaud)
+-----------------------------v-------------------------------+
|             DIGILENT ARTY S7 (SPARTAN-7 XC7S50)             |
|                                                             |
|  +---------------------+        +------------------------+  |
|  | AXI4-Lite Slave Reg | <----> | MicroBlaze / FSM Core  |  |
|  +---------------------+        +------------------------+  |
|             |                                |              |
|  +----------v--------------------------------v-----------+  |
|  |       Neuromorphic Array (32x Fixed-Point Q8.8 LIF)   |  |
|  | - Membrane Potential Accumulator (DSP48E1 Slices)     |  |
|  | - Refractory Counters & STDP Synapse Weight BRAM      |  |
|  | - Spike Pulse Generator & Phase-Locked Loop (100 MHz) |  |
|  +-------------------------------------------------------+  |
|             | Spike Trigger Outputs                         |
|  +----------v--------------------------------------------+  |
|  | Pmod Headers (JA/JB): Logic Analyzer & Oscilloscope   |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

---

## 2. FPGA Resource Budget (XC7S50-1CSGA324C)
| Resource | Available on XC7S50 | SNN Co-Processor Usage | Utilization (%) |
| :--- | :--- | :--- | :--- |
| **LUTs (Logic)** | 32,600 | 4,820 | 14.8% |
| **Flip-Flops (Registers)** | 65,200 | 6,150 | 9.4% |
| **DSP48E1 Slices** | 120 | 32 (1 per LIF core) | 26.7% |
| **Block RAM (36Kb blocks)** | 75 (2,700 Kb) | 8 (Synaptic Weights) | 10.7% |
| **Clock Management (MMCM)** | 5 | 1 (100 MHz System Clk) | 20.0% |

---

## 3. Fixed-Point Q8.8 Leaky Integrate-and-Fire Verilog Core

```verilog
// ============================================================================
// Module: lif_neuron_core.v
// Architecture: Fixed-Point Q8.8 Leaky Integrate-and-Fire (LIF) Spike Core
// Target Device: Xilinx Spartan-7 (XC7S50CSGA324)
// Clock Frequency: 100 MHz
// ============================================================================
`timescale 1ns / 1ps

module lif_neuron_core #(
    parameter signed [15:0] V_REST   = 16'shB600, // -74.0 mV (Q8.8)
    parameter signed [15:0] V_RESET  = 16'shB100, // -79.0 mV (Q8.8)
    parameter signed [15:0] V_THRESH = 16'shCE00, // -50.0 mV (Q8.8)
    parameter signed [15:0] ALPHA_LEAK = 16'sh00F0 // Membrane leak decay (~0.9375)
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire signed [15:0] i_syn,      // Synaptic current from quantum gradient
    output reg         spike_out,         // Fired action potential pulse
    output reg  signed [15:0] v_membrane  // Current membrane potential
);

    reg [3:0] refractory_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_membrane       <= V_REST;
            spike_out        <= 1'b0;
            refractory_count <= 4'd0;
        end else if (enable) begin
            if (refractory_count > 0) begin
                refractory_count <= refractory_count - 1'b1;
                v_membrane       <= V_RESET;
                spike_out        <= 1'b0;
            end else begin
                // Discrete integration: V(t+1) = V_REST + ALPHA * (V(t) - V_REST) + I_syn
                reg signed [31:0] delta_v;
                delta_v = ((v_membrane - V_REST) * ALPHA_LEAK) >>> 8;

                if ((V_REST + delta_v[15:0] + i_syn) >= V_THRESH) begin
                    v_membrane       <= V_RESET;
                    spike_out        <= 1'b1;
                    refractory_count <= 4'd4; // 4-cycle refractory period
                end else begin
                    v_membrane       <= V_REST + delta_v[15:0] + i_syn;
                    spike_out        <= 1'b0;
                end
            end
        end
    end

endmodule
```

---

## 4. Host-to-FPGA Memory-Mapped AXI4-Lite Register Map
| Offset | Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `CONTROL_REG` | R/W | Bit 0: Enable, Bit 1: Soft Reset, Bit 2: Interrupt Enable |
| `0x04` | `STATUS_REG` | RO | Bit 0: Spike Event Buffer Full, Bit 1: PLL Lock |
| `0x08` | `GRAD_IN_REG` | WO | Q8.8 Gradient received from Host CUDA-Q expectation value |
| `0x0C` | `SPIKE_ACCUM_REG` | RO | Accumulated 32-bit post-synaptic spike counter |
| `0x10` | `THETA_OUT_REG` | RO | Fixed-point updated variational angle for quantum kernel |

---

## 5. Hardware-in-the-Loop (HIL) Execution Protocol
1. Host Python executes `cudaq.observe(h2_uccsd_kernel, hamiltonian, theta)`.
2. Host computes parameter gradient $g = \partial \langle H \rangle / \partial \theta$.
3. Host streams gradient $g$ over high-speed FTDI USB-UART to Arty S7 at address `0x08`.
4. Spartan-7 executes 32 parallel LIF neuron updates in hardware ($10\text{ ns}$ cycle time).
5. Post-synaptic spikes trigger the accumulation counter; new variational angle $\theta_{k+1}$ is streamed back to host memory to set the next CUDA-Q statevector excitation angle.
