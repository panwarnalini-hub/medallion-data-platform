# Run Gesture Pipeline (Bronze -> Silver -> Gold)
# Usage: From project root: .\scripts\run-gesture-pipeline.ps1

$ErrorActionPreference = "Stop"

# Go to project root (scripts folder is inside project)
Set-Location "$PSScriptRoot\.."

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MEDALLION GESTURE PIPELINE" -ForegroundColor Cyan
Write-Host "  Bronze - Silver - Gold" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# Step 1: Bronze
Write-Host "`n[1/3] BRONZE : Raw Data Capture" -ForegroundColor Yellow
Write-Host "Hold hand in front of camera, press 'q' to stop" -ForegroundColor Gray
python .\src\bronze_ingestion.py

if ($LASTEXITCODE -ne 0) { Write-Host "Bronze failed!" -ForegroundColor Red; exit 1 }

# Step 2: Silver
Write-Host "`n[2/3] SILVER : Transform & Normalize" -ForegroundColor Yellow
python .\src\silver_transform.py

if ($LASTEXITCODE -ne 0) { Write-Host "Silver failed!" -ForegroundColor Red; exit 1 }

# Step 3: Gold
Write-Host "`n[3/3] GOLD : Feature Engineering" -ForegroundColor Yellow
python .\src\gold_features.py

if ($LASTEXITCODE -ne 0) { Write-Host "Gold failed!" -ForegroundColor Red; exit 1 }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Output folders:" -ForegroundColor Gray
Write-Host "  - bronze_data/" -ForegroundColor Gray
Write-Host "  - silver_data/" -ForegroundColor Gray
Write-Host "  - gold_data/" -ForegroundColor Gray
