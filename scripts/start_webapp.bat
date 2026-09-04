@echo off
cd /d "%~dp0\.."
echo ===================================================
echo Launching Causal DPCRN Web Application Dashboard
echo URL: http://127.0.0.1:8000
echo ===================================================
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
