# Run Gesture Demo
# Usage: From project root: .\scripts\run-gesture-demo.ps1

$ErrorActionPreference = "Stop"

# Go to project root (scripts folder is inside project)
Set-Location "$PSScriptRoot\.."

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GESTURE DEMO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# Add src to PYTHONPATH so demos can import gesture_classifier
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

Write-Host "`nStarting Classifier Demo..." -ForegroundColor Yellow
Write-Host "Press 'q' in camera window to quit`n" -ForegroundColor Gray

python .\demos\demo_classifier.py

Write-Host "`nDone!" -ForegroundColor Green
