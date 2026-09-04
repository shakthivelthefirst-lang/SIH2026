@echo off
cd /d "%~dp0\.."
echo ===================================================
echo Starting Causal DPCRN - Running All Pipeline Files
echo ===================================================
.venv\Scripts\python.exe run_all.py
pause
