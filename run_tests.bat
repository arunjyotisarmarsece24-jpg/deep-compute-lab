@echo off
title Deep Compute Lab - Unit Tests Runner
color 0A
cd /d "%~dp0\day1\gpu-quantum-sim"
cls
echo ===============================================================================
echo            DEEP COMPUTE LAB - RUNNING COMPLETE TEST SUITE (PYTEST)
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
    echo [SUCCESS] ALL TESTS PASSED! Research verification complete.
) else (
    echo [FAILED] Some tests encountered an issue. Please review the output above.
)
echo.
pause
