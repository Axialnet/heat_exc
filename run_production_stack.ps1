# Detached Background Stack Launcher for CHO Predictive Maintenance Platform
# Starts the FastAPI server and Cloudflare tunnel as independent Windows processes.

$ErrorActionPreference = "Continue"
$repoRoot = "C:\Users\Nayan preetham\OneDrive\Documents\CHO"
$pythonExe = "C:\Users\Nayan preetham\OneDrive\Documents\PROJECTS\.venv\Scripts\python.exe"
$cloudflaredExe = "$repoRoot\bin\cloudflared.exe"
$logDir = "$repoRoot\webapp\data"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$serverLog = "$logDir\server.log"
$tunnelLog = "$logDir\tunnel.log"

# Stop any previously running instances
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*run_server.py*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Milliseconds 500

# Clean old logs
"" | Out-File $serverLog -Encoding utf8
"" | Out-File $tunnelLog -Encoding utf8

Write-Host "1. Starting FastAPI Monitoring Server (Independent Process)..."
$serverProc = Start-Process -FilePath $pythonExe `
    -ArgumentList "run_server.py --port 8000 --log-file `"$serverLog`"" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Write-Host "   Server process started with PID: $($serverProc.Id)"

# Wait for server to become healthy (up to 12 seconds)
$serverHealthy = $false
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 1
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res.status -eq "HEALTHY") {
            $serverHealthy = $true
            Write-Host "   Server is HEALTHY on http://127.0.0.1:8000"
            break
        }
    } catch {}
}

if (-not $serverHealthy) {
    Write-Host "   Warning: Server taking longer than expected. Recent server log:"
    Get-Content $serverLog -Tail 10 -ErrorAction SilentlyContinue
}

Write-Host "2. Starting Cloudflare Public Tunnel (Independent Process)..."
$tunnelProc = Start-Process -FilePath $cloudflaredExe `
    -ArgumentList "tunnel --url http://127.0.0.1:8000 --logfile `"$tunnelLog`"" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Write-Host "   Tunnel process started with PID: $($tunnelProc.Id)"

# Extract public URL (up to 12 seconds)
$publicUrl = ""
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $match = Select-String -Path $tunnelLog -Pattern "https://[a-zA-Z0-9-]+\.trycloudflare\.com"
        if ($match) {
            $publicUrl = $match.Matches[0].Value
            Write-Host "   Found Public URL: $publicUrl"
            break
        }
    }
}

Write-Host "======================================================================="
Write-Host "STACK SUCCESSFULLY LAUNCHED AS DETACHED WINDOWS SERVICES"
Write-Host "   Server PID   : $($serverProc.Id)"
Write-Host "   Tunnel PID   : $($tunnelProc.Id)"
Write-Host "   Local URL    : http://127.0.0.1:8000"
Write-Host "   Public URL   : $publicUrl"
Write-Host "======================================================================="
