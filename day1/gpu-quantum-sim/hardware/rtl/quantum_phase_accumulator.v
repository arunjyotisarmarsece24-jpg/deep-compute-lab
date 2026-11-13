// ==============================================================================
// Project: deep-compute-lab :: Hardware Neuromorphic Co-Processor
// Module:  quantum_phase_accumulator.v
// Target:  Digilent Arty S7 (Xilinx Spartan-7 XC7S50CSGA324-1)
// Author:  Arunjyoti Sarma (deep-compute-lab)
// Standard: IEEE 1364-2001 (Verilog HDL, Synthesizable RTL)
// ------------------------------------------------------------------------------
// Description:
//   Synaptic Spike-to-Quantum Phase Accumulator with AXI4-Lite Register Map.
//   Integrates post-synaptic spike currents into continuous quantum gate rotation
//   angles theta(t) for parameterized quantum circuits (UCCSD / Givens rotations).
// ==============================================================================

`timescale 1ns / 1ps

module quantum_phase_accumulator #(
    parameter signed [15:0] THETA_INIT      = 16'sh0010, // Initial angle (0.0625 rad in Q8.8)
    parameter signed [15:0] LEARNING_RATE   = 16'sh0008  // Step scaling factor (0.03125 in Q8.8)
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 tick_enable,
    input  wire                 spike_trigger,       // High when neuron population fires
    input  wire signed [15:0]   synaptic_weight,     // Modulating synaptic weight (Q8.8)

    // AXI4-Lite readable output registers
    output reg  signed [15:0]   theta_angle_q8_8,    // Variational rotation angle theta in Q8.8
    output reg  [31:0]          total_spike_count    // Total spikes accumulated across optimization
);

    wire signed [31:0] delta_product = synaptic_weight * LEARNING_RATE;
    wire signed [15:0] delta_theta   = delta_product[23:8];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            theta_angle_q8_8  <= THETA_INIT;
            total_spike_count <= 32'd0;
        end else if (tick_enable) begin
            if (spike_trigger) begin
                theta_angle_q8_8  <= theta_angle_q8_8 + delta_theta;
                total_spike_count <= total_spike_count + 32'd1;
            end
        end
    end

endmodule
