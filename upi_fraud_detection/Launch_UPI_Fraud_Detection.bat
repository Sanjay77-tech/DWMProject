@echo off
title UPI Sentinel AI - Fraud Detection Platform
echo ======================================================================
echo    STARTING UPI SENTINEL AI PLATFORM (React + FastAPI)
echo ======================================================================
echo.
cd /d "%~dp0"
py run.py
if errorlevel 1 (
    python run.py
)
pause
