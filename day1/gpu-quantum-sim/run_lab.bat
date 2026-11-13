@echo off
title GPU Quantum Simulator & Neuromorphic Research Lab [Day 1]
color 0B
cd /d "%~dp0"

:MENU
cls
echo ===============================================================================
echo   ____ ____  _   _    ___  _   _    _    _   _ _____ _   _ __  __ 
echo  / ___|  _ \| | | |  / _ \| | | |  / \  | \ | |_   _| | | |  \/  |
echo | |  _| |_) | | | | | | | | | | | / _ \ |  \| | | | | | | | |\/| |
echo | |_| |  __/| |_| | | |_| | |_| |/ ___ \| |\  | | | | |_| | |  | |
echo  \____|_|    \___/   \__\_\\___//_/   \_\_| \_| |_|  \___/|_|  |_|
echo           GPU-ACCELERATED QUANTUM SIMULATION & NEUROMORPHIC LAB
echo                 NVIDIA CUDA-Q | PySCF | OpenFermion | SNN
echo ===============================================================================
echo   Repository: C:\Users\arunj\Desktop\Arunjyoti_Sarma_all_For_Research
echo               \My_Coding_Cool\My_Quantum_computing_Project\day1\gpu-quantum-sim
echo ===============================================================================
echo.
echo   [1] RUN TEST SUITE (pytest - 14 Unit Tests)
echo   [2] RUN H2 VQE ELECTRONIC STRUCTURE & PES SWEEP
echo   [3] RUN GHZ QUANTUM SIMULATION BENCHMARK (up to 27+ Qubits)
echo   [4] LAUNCH NEUROMORPHIC SNN & TELEMETRY DASHBOARD
echo   [5] OPEN ALL RESULTS & PUBLICATION PLOTS IN WINDOWS
echo   [6] PRINT SUMMARY METRICS FROM CSVs (Chemical Accuracy & Speed)
echo   [7] RUN COMPLETE END-TO-END PIPELINE (All Benchmarks)
echo   [0] EXIT
echo.
echo ===============================================================================
set /p CHOICE="  Enter option [0-7] and press Enter: "

if "%CHOICE%"=="1" goto TEST
if "%CHOICE%"=="2" goto VQE
if "%CHOICE%"=="3" goto GHZ
if "%CHOICE%"=="4" goto DASHBOARD
if "%CHOICE%"=="5" goto VIEW_RESULTS
if "%CHOICE%"=="6" goto SUMMARY
if "%CHOICE%"=="7" goto ALL
if "%CHOICE%"=="0" goto EXIT

echo Invalid option. Please try again.
timeout /t 2 >nul
goto MENU

:TEST
cls
echo.
echo ===============================================================================
echo   RUNNING PYTEST TEST SUITE (14 Tests)
echo ===============================================================================
echo.
python -m pytest tests/ -v
echo.
echo ===============================================================================
pause
goto MENU

:VQE
cls
echo.
echo ===============================================================================
echo   RUNNING H2 VQE SIMULATION & POTENTIAL ENERGY SURFACE SWEEP
echo ===============================================================================
echo.
echo   Choose mode:
echo   [1] Fast Mode (6 points, < 15 seconds)
echo   [2] Full Research Sweep (45 points, R = 0.30 to 2.50 Angstroms)
echo.
set /p VQE_MODE="  Select [1 or 2]: "
if "%VQE_MODE%"=="1" (
    python src/h2_vqe.py --quick
) else (
    python src/h2_vqe.py --r-start 0.30 --r-end 2.50 --r-step 0.05
)
echo.
echo ===============================================================================
pause
goto MENU

:GHZ
cls
echo.
echo ===============================================================================
echo   RUNNING GHZ QUANTUM STATEVECTOR SIMULATION BENCHMARK
echo ===============================================================================
echo.
echo   Choose mode:
echo   [1] Quick Benchmark (n=20 to 24 qubits, < 10 seconds)
echo   [2] Deep Memory Benchmark (n=20 to 28 qubits, with OOM protection)
echo.
set /p GHZ_MODE="  Select [1 or 2]: "
if "%GHZ_MODE%"=="1" (
    python src/ghz_benchmark.py --quick
) else (
    python src/ghz_benchmark.py --min-qubits 20 --max-qubits 28 --runs 5
)
echo.
echo ===============================================================================
pause
goto MENU

:DASHBOARD
cls
echo.
echo ===============================================================================
echo   LAUNCHING NEUROMORPHIC SNN & QUANTUM TELEMETRY DASHBOARD
echo ===============================================================================
echo.
python src/neuromorphic_quantum.py
echo.
echo Rendering 4-Panel Research Dashboard...
python src/dashboard.py --show
echo.
echo Opening generated telemetry snapshot...
if exist "results\dashboard_telemetry.png" start "" "results\dashboard_telemetry.png"
echo.
echo ===============================================================================
pause
goto MENU

:VIEW_RESULTS
cls
echo.
echo ===============================================================================
echo   OPENING PUBLICATION PLOTS & RESULTS DIRECTORY
echo ===============================================================================
echo.
echo Opening plots in Windows default viewer...
if exist "results\h2_pes.png" start "" "results\h2_pes.png"
if exist "results\h2_error.png" start "" "results\h2_error.png"
if exist "results\vqe_convergence.png" start "" "results\vqe_convergence.png"
if exist "results\ghz_benchmark.png" start "" "results\ghz_benchmark.png"
if exist "results\dashboard_telemetry.png" start "" "results\dashboard_telemetry.png"
start "" "results"
echo Plots opened!
echo.
echo ===============================================================================
pause
goto MENU

:SUMMARY
cls
echo.
python src\print_summary.py
echo.
pause
goto MENU

:ALL
cls
echo.
echo ===============================================================================
echo   RUNNING COMPLETE REPRODUCIBLE RESEARCH PIPELINE
echo ===============================================================================
echo.
call "%~dp0run_all.bat" 2>nul || powershell -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
echo.
echo ===============================================================================
pause
goto MENU

:EXIT
cls
echo.
echo Exiting GPU Quantum Lab. Keep crushing it on your 30-day research journey!
echo.
timeout /t 2 >nul
exit /b 0
