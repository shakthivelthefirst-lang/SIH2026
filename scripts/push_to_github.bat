@echo off
cd /d "%~dp0\.."
echo ===================================================================
echo Pushing all files and phases to GitHub repository:
echo https://github.com/shakthivelthefirst-lang/SIH2026 (branch: training-2)
echo ===================================================================
echo.
git push -u origin training-2
echo.
pause
