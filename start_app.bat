@echo off
title CHO 24/7 Heat Exchanger Monitoring Platform
cd /d "%~dp0"
echo =======================================================================
echo   Starting CHO 24/7 Heat Exchanger Monitoring Platform...
echo =======================================================================
echo.
echo Opening browser at http://localhost:8000 ...
start http://localhost:8000
echo.
echo Starting backend server on port 8000 (Press Ctrl+C to stop)...
"C:\Users\Nayan preetham\OneDrive\Documents\PROJECTS\.venv\Scripts\python.exe" run_server.py --port 8000
pause
