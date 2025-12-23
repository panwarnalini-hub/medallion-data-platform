# Run Touchless Kiosk Demo
# Usage: From project root: .\scripts\run-kiosk.ps1

$ErrorActionPreference = "Stop"

# Go to project root
Set-Location "$PSScriptRoot\.."

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TOUCHLESS HOSPITAL KIOSK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# Add src to PYTHONPATH so demos can import gesture_classifier
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

Write-Host "`nControls:" -ForegroundColor Yellow
Write-Host "  Swipe LEFT/RIGHT - Navigate cards" -ForegroundColor Gray
Write-Host "  THUMBS UP        - Select" -ForegroundColor Gray
Write-Host "  FIST             - Back/Close" -ForegroundColor Gray
Write-Host "`nPress 'q' in camera window to quit`n" -ForegroundColor Gray

python .\demos\demo_kiosk.py

Write-Host "`nKiosk closed." -ForegroundColor Green
