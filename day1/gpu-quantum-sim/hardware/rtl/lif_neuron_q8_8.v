// ==============================================================================
// Project: deep-compute-lab :: Hardware Neuromorphic Co-Processor
// Module:  lif_neuron_q8_8.v
// Target:  Digilent Arty S7 (Xilinx Spartan-7 XC7S50CSGA324-1)
// Author:  Arunjyoti Sarma (deep-compute-lab)
// Standard: IEEE 1364-2001 (Verilog HDL, Synthesizable RTL)
// ------------------------------------------------------------------------------
// Description:
//   16-bit Signed Q8.8 Fixed-Point Leaky Integrate-and-Fire (LIF) Spiking Neuron.
//   Implements sub-threshold membrane ODE discretization with leak factor,
//   synaptic current accumulation, threshold comparison, and refractory state.
// ==============================================================================

`timescale 1ns / 1ps

module lif_neuron_q8_8 #(
    // Fixed-Point Q8.8 Constants ([15:8] Integer, [7:0] Fractional)
    parameter signed [15:0] V_REST          = 16'shB200, // -70.0 mV in Q8.8 (-70 * 256 = -17920 = 0xB200)
    parameter signed [15:0] V_RESET         = 16'shB500, // -75.0 mV in Q8.8 (-75 * 256 = -19200 = 0xB500)
    parameter signed [15:0] V_THRESH        = 16'shCE00, // -50.0 mV in Q8.8 (-50 * 256 = -12800 = 0xCE00)
    parameter signed [15:0] LEAK_DECAY      = 16'sh00F3, // exp(-dt/tau_m) approx 0.95 in Q8.8 (0.95 * 256 = 243 = 0xF3)
    parameter integer       REFRACTORY_TICKS= 4          // Clock cycles held at reset potential post-spike
)(
    input  wire                 clk,            // System master clock (100 MHz on Arty S7)
    input  wire                 rst_n,          // Asynchronous active-low reset
    input  wire                 enable,         // Clock enable / simulation tick strobe
    input  wire signed [15:0]   i_syn,          // Synaptic current input (Q8.8)
    input  wire signed [15:0]   i_bias,         // Background tonic bias current (Q8.8)

    output reg  signed [15:0]   v_mem,          // Instantaneous membrane potential (Q8.8)
    output reg                  spike_out,      // Action potential output pulse (1 clock cycle active-high)
    output reg                  in_refractory   // Status flag: neuron currently in absolute refractory period
);

    // Internal state variables
    reg [3:0] refractory_counter;
    
    // Arithmetic pipeline registers
    wire signed [15:0] v_diff;
    wire signed [31:0] leak_product;
    wire signed [15:0] v_leaked;
    wire signed [15:0] v_integrated;

    // Sub-threshold dynamics: v_leaked = (v_mem - V_REST) * LEAK_DECAY >> 8
    assign v_diff       = v_mem - V_REST;
    assign leak_product = v_diff * LEAK_DECAY;
    assign v_leaked     = V_REST + leak_product[23:8]; // Truncate fractional bits back to Q8.8
    assign v_integrated = v_leaked + i_syn + i_bias;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem              <= V_REST;
            spike_out          <= 1'b0;
            in_refractory      <= 1'b0;
            refractory_counter <= 4'd0;
        end else if (enable) begin
            if (refractory_counter > 4'd0) begin
                // Absolute refractory period: clamp membrane potential to V_RESET
                v_mem              <= V_RESET;
                spike_out          <= 1'b0;
                in_refractory      <= 1'b1;
                refractory_counter <= refractory_counter - 4'd1;
            end else begin
                in_refractory <= 1'b0;
                if (v_integrated >= V_THRESH) begin
                    // Action potential threshold crossed: emit spike & reset
                    v_mem              <= V_RESET;
                    spike_out          <= 1'b1;
                    refractory_counter <= REFRACTORY_TICKS[3:0];
                end else begin
                    // Sub-threshold continuous integration
                    v_mem              <= v_integrated;
                    spike_out          <= 1'b0;
                end
            end
        end else begin
            spike_out <= 1'b0;
        end
    end

endmodule
