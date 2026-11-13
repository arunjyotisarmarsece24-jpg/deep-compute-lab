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
echo  [DAY 1: GPU QUANTUM SIMULATION & NEUROMORPHIC CONTROL]
echo    [1] Run All 14 Verification Tests (pytest)
echo    [2] Launch Live Neuromorphic Quantum GUI Dashboard (Spikes + Statevector)
echo    [3] Run Quick Benchmark ^& VQE Sweep (^< 2 min test)
echo    [4] Run Full GHZ Statevector Benchmark (20 to 27 Qubits)
echo    [5] Run Full H2 VQE Chemical Accuracy Sweep (45 bond distances)
echo    [6] Print Live Research Benchmark Metrics Summary
echo.
echo  [GIT ^& REPOSITORY OPERATIONS]
echo    [7] Push Current Workspace to GitHub (origin/main)
echo    [8] Check Git Status
echo.
echo    [9] Exit
echo.
echo ===============================================================================
set /p choice="Select an option (1-9): "

if "%choice%"=="1" goto RUN_TESTS
if "%choice%"=="2" goto RUN_DASHBOARD
if "%choice%"=="3" goto RUN_QUICK
if "%choice%"=="4" goto RUN_GHZ
if "%choice%"=="5" goto RUN_VQE
if "%choice%"=="6" goto RUN_METRICS
if "%choice%"=="7" goto RUN_PUSH
if "%choice%"=="8" goto RUN_STATUS
if "%choice%"=="9" goto EXIT

echo Invalid choice. Please try again.
pause
goto MENU

:RUN_TESTS
cls
echo [DEEP COMPUTE LAB] Running unit tests suite...
cd /d "%~dp0\day1\gpu-quantum-sim"
python -m pytest tests/ -v
cd /d "%~dp0"
echo.
pause
goto MENU

:RUN_DASHBOARD
cls
echo [DEEP COMPUTE LAB] Launching Live Neuromorphic Quantum GUI Dashboard...
cd /d "%~dp0\day1\gpu-quantum-sim"
python src/dashboard.py
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
echo [DEEP COMPUTE LAB] Pushing Day 1 to GitHub repository (origin/main)...
cd /d "%~dp0"
git add .
git commit -m "feat: Add Day 1 GPU Quantum Simulation, Neuromorphic LIF Controller, and Arty S7 FPGA specs"
git push -u origin main
echo.
pause
goto MENU

:RUN_STATUS
cls
echo [DEEP COMPUTE LAB] Git Status:
cd /d "%~dp0"
git status
echo.
pause
goto MENU

:EXIT
echo Exiting Deep Compute Lab. Keep building beasts!
exit /b 0
