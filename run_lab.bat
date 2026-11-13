@echo off
setlocal enabledelayedexpansion
title deep-compute-lab :: Autonomous Quantum & Neuromorphic Research
color 0B

:MENU
cls
echo ===============================================================================
echo            DEEP-COMPUTE-LAB: GPU, QUANTUM, NEUROMORPHIC & FPGA LAB
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
echo  [GIT & REPOSITORY OPERATIONS]
echo    [9] Commit & Push Everything to GitHub (origin/main)
echo    [10] Exit
echo.
echo ===============================================================================
set /p choice="Select an option (1-10): "

if "%choice%"=="1" goto RUN_COCKPIT
if "%choice%"=="2" goto RUN_GIF
if "%choice%"=="3" goto RUN_RTL
if "%choice%"=="4" goto RUN_TESTS
if "%choice%"=="5" goto RUN_GHZ
if "%choice%"=="6" goto RUN_VQE
if "%choice%"=="7" goto RUN_QUICK
if "%choice%"=="8" goto RUN_METRICS
if "%choice%"=="9" goto RUN_PUSH
if "%choice%"=="10" goto EXIT

echo Invalid choice. Please try again.
pause
goto MENU

:RUN_COCKPIT
cls
echo [DEEP COMPUTE LAB] Launching Live Neuromorphic-Quantum Research Cockpit...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/live_cockpit.py
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_GIF
cls
echo [DEEP COMPUTE LAB] Exporting 35-frame high-resolution animated learning GIF...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/live_cockpit.py --export-gif --steps 35
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_RTL
cls
echo [DEEP COMPUTE LAB] Simulating Spartan-7 Verilog RTL and generating digital waveforms...
cd /d "%~dp0\day1\gpu-quantum-sim"
python hardware/sim/rtl_waveform_viewer.py
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_TESTS
cls
echo [DEEP COMPUTE LAB] Running complete 17-test suite...
cd /d "%~dp0\day1\gpu-quantum-sim"
python -m pytest tests/ -v
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_GHZ
cls
echo [DEEP COMPUTE LAB] Running full GHZ statevector simulation benchmark...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/ghz_benchmark.py --max-qubits 30 --runs 5
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_VQE
cls
echo [DEEP COMPUTE LAB] Running full H2 VQE potential energy curve sweep (0.30 - 2.50 A)...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/h2_vqe.py --step 0.05
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_QUICK
cls
echo [DEEP COMPUTE LAB] Running quick verification benchmark...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/ghz_benchmark.py --quick
python src/h2_vqe.py --quick
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_METRICS
cls
echo [DEEP COMPUTE LAB] Research Summary:
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/print_summary.py
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_PUSH
cls
echo [DEEP COMPUTE LAB] Pushing all updates to GitHub repository (origin/main)...
cd /d "%~dp0"
git add .
git commit -m "feat: Add Live Neuromorphic-Quantum Cockpit, animated telemetry GIF, synthesizable Verilog RTL, and 17 verification tests"
git push origin main
echo.
pause
goto MENU

:EXIT
echo Exiting Deep Compute Lab. Keep building beasts!
exit /b 0
