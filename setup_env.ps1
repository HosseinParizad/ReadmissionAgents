# PowerShell script to set up the virtual environment
Write-Host "Setting up Readmission Agents project..." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "Setup complete! Virtual environment is activated." -ForegroundColor Green
Write-Host "To activate manually, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan

