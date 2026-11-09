# PowerShell 1-Click Launcher for CHO Heat Exchanger Monitoring Platform
Set-Location -Path $PSScriptRoot
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Starting CHO 24/7 Heat Exchanger Monitoring Platform..." -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Opening browser at http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"
Write-Host "Starting backend server on port 8000 (Press Ctrl+C to stop)..." -ForegroundColor Green
& "C:\Users\Nayan preetham\OneDrive\Documents\PROJECTS\.venv\Scripts\python.exe" run_server.py --port 8000
