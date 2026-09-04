@echo off
cd /d "%~dp0\.."
echo ===================================================
echo Running 6-Module Unit & Integration Test Suite
echo ===================================================
.venv\Scripts\python.exe test_system.py
pause
