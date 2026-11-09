@echo off
title CHO Predictive Maintenance Platform - Public Web Tunnel
cd /d "%~dp0"
echo =======================================================================
echo   CHO 24/7 HEAT EXCHANGER MONITORING & PREDICTIVE MAINTENANCE
echo   Generating Public Cloudflare HTTPS Web Link...
echo =======================================================================
echo.
if not exist "bin\cloudflared.exe" (
    echo Error: bin\cloudflared.exe not found.
    pause
    exit /b 1
)

echo Starting tunnel for http://localhost:8000 ...
echo Look for the https://*.trycloudflare.com link below:
echo =======================================================================
"bin\cloudflared.exe" tunnel --url http://localhost:8000
pause
