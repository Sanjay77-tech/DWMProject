@echo off
title UPI Fraud Detection AI Engine
echo ======================================================================
echo    UPI SCAM & FRAUD DETECTION AI - STARTING SYSTEM
echo ======================================================================
echo.
cd /d "C:\Users\Admin\OneDrive\Desktop\DWMProject\upi_fraud_detection"
echo Current Directory: %CD%
echo Launching Streamlit application...
echo.
py -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Trying fallback python command...
    python -m streamlit run app.py
)
pause
