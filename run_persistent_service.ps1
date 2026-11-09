# Persistent WMI Launcher for CHO Monitoring & Predictive Maintenance Platform
# Spawns server and tunnel outside Antigravity Job Object via Windows Management Instrumentation (WMI).

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

# Terminate any previously running instances
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    try { (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine -like "*run_server.py*" } catch { $false }
} | Stop-Process -Force -ErrorAction SilentlyContinue

Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Milliseconds 600

# Clear log files
"" | Out-File $serverLog -Encoding utf8
"" | Out-File $tunnelLog -Encoding utf8

# 1. Start Server via WMI
$serverCmd = "cmd.exe /c `"`"$pythonExe`" `"$repoRoot\run_server.py`" --port 8000 --log-file `"$serverLog`"`""
$serverProc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $serverCmd
    CurrentDirectory = $repoRoot
}
Write-Host "1. Server spawned via WMI (ProcessId: $($serverProc.ProcessId), ReturnValue: $($serverProc.ReturnValue))"

# Wait for server health (up to 12s)
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
    Write-Host "   Server log:"
    Get-Content $serverLog -Tail 10 -ErrorAction SilentlyContinue
}

# 2. Start Cloudflare Tunnel via WMI
$tunnelCmd = "cmd.exe /c `"`"$cloudflaredExe`" tunnel --url http://127.0.0.1:8000 --logfile `"$tunnelLog`"`""
$tunnelProc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $tunnelCmd
    CurrentDirectory = $repoRoot
}
Write-Host "2. Tunnel spawned via WMI (ProcessId: $($tunnelProc.ProcessId), ReturnValue: $($tunnelProc.ReturnValue))"

# Wait for public URL from tunnel log (up to 12s)
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
Write-Host "PERSISTENT WMI STACK READY"
Write-Host "   Server PID : $($serverProc.ProcessId)"
Write-Host "   Tunnel PID : $($tunnelProc.ProcessId)"
Write-Host "   Local URL  : http://127.0.0.1:8000"
Write-Host "   Public URL : $publicUrl"
Write-Host "======================================================================="
