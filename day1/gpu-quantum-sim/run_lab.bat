@echo off
setlocal enabledelayedexpansion
title gpu-quantum-sim :: Autonomous Quantum & Neuromorphic Research
color 0B

:MENU
cls
echo ===============================================================================
echo            GPU-QUANTUM-SIM: GPU, QUANTUM, NEUROMORPHIC & FPGA LAB
echo        Author: Arunjyoti Sarma ^| GitHub: arunjyotisarmarsece24-jpg/deep-compute-lab
echo ===============================================================================
echo.
echo  [NEUROMORPHIC & QUANTUM VISUAL COCKPIT]
echo    [1] Launch Live Neuromorphic-Quantum GUI Cockpit (Spikes + STDP + Entropy + VQE)
echo    [2] Generate/Export Animated Learning GIF (neuromorphic_quantum_learning.gif)
echo    [3] Run Spartan-7 FPGA Cycle-Accurate RTL Waveform Simulator (ModelSim style)
echo.
echo  [QUANTUM BENCHMARKS & VERIFICATION TESTS]
echo    [4] Run Complete 17-Test Research Verification Suite (pytest -v)
echo    [5] Run Full GHZ Statevector Simulation Benchmark (up to 27 Qubits)
echo    [6] Run Full H2 VQE Molecular Potential Energy Sweep (45 bond distances)
echo    [7] Run Quick Verification Benchmark (^< 2 min test)
echo    [8] Print Live Benchmark Metrics Summary
echo.
echo    [9] Exit
echo.
echo ===============================================================================
set /p choice="Select an option (1-9): "

if "%choice%"=="1" goto RUN_COCKPIT
if "%choice%"=="2" goto RUN_GIF
if "%choice%"=="3" goto RUN_RTL
if "%choice%"=="4" goto RUN_TESTS
if "%choice%"=="5" goto RUN_GHZ
if "%choice%"=="6" goto RUN_VQE
if "%choice%"=="7" goto RUN_QUICK
if "%choice%"=="8" goto RUN_METRICS
if "%choice%"=="9" goto EXIT

echo Invalid choice. Please try again.
pause
goto MENU

:RUN_COCKPIT
cls
echo Launching Live Neuromorphic-Quantum Research Cockpit...
python src/live_cockpit.py
echo.
pause
goto MENU

:RUN_GIF
cls
echo Exporting 35-frame high-resolution animated learning GIF...
python src/live_cockpit.py --export-gif --steps 35
echo.
pause
goto MENU

:RUN_RTL
cls
echo Simulating Spartan-7 Verilog RTL and generating digital waveforms...
python hardware/sim/rtl_waveform_viewer.py
echo.
pause
goto MENU

:RUN_TESTS
cls
echo Running complete 17-test suite...
python -m pytest tests/ -v
echo.
pause
goto MENU

:RUN_GHZ
cls
echo Running full GHZ statevector simulation benchmark...
python src/ghz_benchmark.py --max-qubits 30 --runs 5
echo.
pause
goto MENU

:RUN_VQE
cls
echo Running full H2 VQE potential energy curve sweep (0.30 - 2.50 A)...
python src/h2_vqe.py --step 0.05
echo.
pause
goto MENU

:RUN_QUICK
cls
echo Running quick verification benchmark...
python src/ghz_benchmark.py --quick
python src/h2_vqe.py --quick
echo.
pause
goto MENU

:RUN_METRICS
cls
echo Research Summary:
python src/print_summary.py
echo.
pause
goto MENU

:EXIT
exit /b 0
