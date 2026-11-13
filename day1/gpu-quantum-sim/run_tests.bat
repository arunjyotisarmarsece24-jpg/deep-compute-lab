@echo off
title GPU Quantum Sim - Unit Tests Runner
color 0A
cd /d "%~dp0"
cls
echo ===============================================================================
echo            GPU-QUANTUM-SIM - RUNNING COMPLETE TEST SUITE (PYTEST)
echo ===============================================================================
echo Working Directory: %CD%
echo Python:
python --version
echo.
echo Running 14 Unit Tests (GHZ Bitstrings, 15-term Pauli H2, VQE Chemical Accuracy, Memory Guard)...
echo -------------------------------------------------------------------------------
pytest tests/ -v
echo -------------------------------------------------------------------------------
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] ALL TESTS PASSED!
) else (
    echo [FAILED] Some tests encountered an issue.
)
echo.
pause
