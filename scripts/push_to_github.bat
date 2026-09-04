@echo off
cd /d "%~dp0\.."
echo ===================================================================
echo Pushing ALL files and phases to GitHub repository:
echo https://github.com/shakthivelthefirst-lang/SIH2026
echo Target Branch: training-2
echo ===================================================================
echo.
git push -u origin training-2 --force
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] All files successfully uploaded to branch training-2!
    echo Check your repository: https://github.com/shakthivelthefirst-lang/SIH2026/tree/training-2
) else (
    echo [FAILED] Push did not complete. Please check the message above.
)
echo.
pause
