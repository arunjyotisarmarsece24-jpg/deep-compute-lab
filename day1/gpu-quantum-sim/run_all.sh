#!/usr/bin/env bash
# ==============================================================================
# gpu-quantum-sim: End-to-End Automated Pipeline (Linux / macOS / Colab)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "  GPU QUANTUM SIMULATOR: REPRODUCIBLE BENCHMARK PIPELINE"
echo "======================================================================"

echo -e "\n[1/4] Installing / Verifying Dependencies..."
python -m pip install -r requirements.txt

echo -e "\n[2/4] Running Pytest Validation Suite..."
python -m pytest tests/ -v

echo -e "\n[3/4] Running GHZ Statevector Simulation Benchmark..."
python src/ghz_benchmark.py

echo -e "\n[4/4] Running H2 VQE Potential Energy Surface Sweep..."
python src/h2_vqe.py

echo -e "\n======================================================================"
echo "  ALL EXPERIMENTS & PLOTS REGENERATED SUCCESSFULLY!"
echo "  Artifacts saved to results/:"
ls -lh results/
echo "======================================================================"
