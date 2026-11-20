@echo off
title UPI Sentinel AI - Fraud Detection Platform
echo ======================================================================
echo    UPI SENTINEL AI - FINTECH FRAUD DETECTION PLATFORM
echo    React.js + FastAPI + Scikit-Learn Engine
echo ======================================================================
echo.
cd /d "%~dp0"
echo Current Directory: %CD%
echo Launching unified application on http://localhost:8000 ...
echo.
py run.py
if errorlevel 1 (
    echo.
    echo Trying fallback python command...
    python run.py
)
pause
