$ErrorActionPreference = "Stop"

# Ensure we're in the project root
cd $PSScriptRoot

# Deactivate any active virtual environment
if ($env:VIRTUAL_ENV) {
    deactivate
}

# Activate local virtual environment
.\.venv\Scripts\Activate.ps1

# Run pipeline
python demo_classifier.py
python demo_kiosk.py