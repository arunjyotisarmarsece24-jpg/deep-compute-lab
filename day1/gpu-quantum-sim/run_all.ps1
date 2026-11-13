# ==============================================================================
# gpu-quantum-sim: End-to-End Automated Pipeline (Windows PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  GPU QUANTUM SIMULATOR: REPRODUCIBLE BENCHMARK PIPELINE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Installing / Verifying Dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

Write-Host "`n[2/4] Running Pytest Validation Suite..." -ForegroundColor Yellow
python -m pytest tests/ -v

Write-Host "`n[3/4] Running GHZ Statevector Simulation Benchmark..." -ForegroundColor Yellow
python src/ghz_benchmark.py

Write-Host "`n[4/4] Running H2 VQE Potential Energy Surface Sweep..." -ForegroundColor Yellow
python src/h2_vqe.py

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  ALL EXPERIMENTS & PLOTS REGENERATED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "  Artifacts saved to results/:" -ForegroundColor Green
Get-ChildItem results/ | Format-Table Name, Length, LastWriteTime
Write-Host "======================================================================" -ForegroundColor Green
